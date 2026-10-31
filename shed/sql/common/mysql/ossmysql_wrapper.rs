/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is dual-licensed under either the MIT license found in the
 * LICENSE-MIT file in the root directory of this source tree or the Apache
 * License, Version 2.0 found in the LICENSE-APACHE file in the root directory
 * of this source tree. You may select, at your option, one of the
 * above-listed licenses.
 */

//! Module hides the implementation details of the Facebook Mysql client library
//! and provides API that is used in sql crate.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Error;
use futures::FutureExt;
use futures_stats::futures03::TimedFutureExt;
use mysql_async::Conn as MysqlConnection;
use mysql_async::QueryResult as MysqlQueryResult;
use mysql_async::Row;
use mysql_async::TextProtocol;
use mysql_async::Transaction;
use mysql_async::TxOpts;
use mysql_async::prelude::Queryable;
use stats::prelude::*;
use time_ext::DurationExt;

use crate::QueryTelemetry;
use crate::mysql::ConnectionStats;
use crate::mysql::IsolationLevel;
use crate::mysql::OssErrorKind;
use crate::mysql::OssMysqlError;
use crate::mysql::OssPool;
use crate::mysql::OssStmt;
use crate::mysql::WriteResult;

type QueryResult<'a> = MysqlQueryResult<'a, 'static, TextProtocol>;

/// Telemetry returned after a query is executed or transaction is committed.
pub type OssQueryTelemetry = HashMap<String, String>;

/// OssConnection is a wrapper around a MySQL async Pool
/// It provides read/write query and begin transaction API.
#[derive(Clone)]
pub struct OssConnection {
    /// Connection pool
    pub pool: OssPool,
    /// Stats struct for logging performance
    pub stats: Arc<ConnectionStats>,
    /// Isolation level of the transactions started on this connection.
    /// `None` keeps the server's default.
    isolation_level: Option<IsolationLevel>,
}

impl OssConnection {
    /// Creates OssConnection from a Pool or an OssPool object
    pub fn new(pool: impl Into<OssPool>, stats: Arc<ConnectionStats>) -> Self {
        Self {
            pool: pool.into(),
            stats,
            isolation_level: None,
        }
    }

    /// Sets the isolation level of the transactions started on this
    /// connection, unless their options set one. `None` keeps the server's
    /// default. Autocommit statements always run at the server's default:
    /// the pool may be shared with other connections, so the level is set
    /// for each transaction rather than for the session.
    pub fn set_isolation_level(&mut self, isolation_level: Option<IsolationLevel>) {
        self.isolation_level = isolation_level;
    }

    /// Checks out a connection from the pool while collecting stats. Errors
    /// of the pool arrive as `mysql_async::Error::Other(OssMysqlError)`.
    pub async fn get_conn_counted(
        pool: impl Into<OssPool>,
        stats: &ConnectionStats,
    ) -> Result<MysqlConnection, mysql_async::Error> {
        let pool = pool.into();
        let (st, conn) = pool.get_conn().timed().await;
        stats
            .get_connection_ms
            .add_value(st.completion_time.as_millis_unchecked() as i64);
        conn.map_err(OssMysqlError::into_mysql)
    }

    /// Executes a given query and returns the result while collecting stats
    pub async fn raw_query_counted<'a>(
        conn: &'a mut MysqlConnection,
        stats: &ConnectionStats,
        query: &'a str,
    ) -> Result<QueryResult<'a>, mysql_async::Error> {
        let (st, result) = conn.query_iter(query).timed().await;
        stats
            .raw_query_ms
            .add_value(st.completion_time.as_millis_unchecked() as i64);
        result
    }

    /// Performs a given query and returns the result as a QueryResult
    pub async fn read_query<'a>(
        &self,
        conn: &'a mut MysqlConnection,
        query: &'a str,
    ) -> Result<(QueryResult<'a>, Option<QueryTelemetry>), mysql_async::Error> {
        let res = OssConnection::raw_query_counted(conn, &self.stats, query).await?;
        Ok((res, None))
    }

    /// Performs a given read query and maps every row of its result with `f`.
    pub async fn read_rows<T, F>(&self, query: &str, f: F) -> Result<Vec<T>, Error>
    where
        F: FnMut(Row) -> Result<T, Error> + Send,
    {
        let mut conn = OssConnection::get_conn_counted(self.pool.clone(), &self.stats).await?;
        let result = self
            .pool
            .execute(OssStmt::Read, async {
                let mut result =
                    OssConnection::raw_query_counted(&mut conn, &self.stats, query).await?;
                result.map(f).await
            })
            .await;
        let rows = result.map_err(|err| discard_after_timeout(conn, err))?;
        rows.into_iter().collect()
    }

    /// Performs a given query and returns the write result.
    pub async fn write_query(&self, query: String) -> Result<WriteResult, Error> {
        let mut conn = OssConnection::get_conn_counted(self.pool.clone(), &self.stats).await?;
        let result = self
            .pool
            .execute(OssStmt::Write, async {
                let result =
                    OssConnection::raw_query_counted(&mut conn, &self.stats, &query).await?;
                Ok::<_, mysql_async::Error>((
                    result.last_insert_id().unwrap_or(0),
                    result.affected_rows(),
                ))
            })
            .await;
        let (last_insert_id, rows_affected) =
            result.map_err(|err| discard_after_timeout(conn, err))?;
        Ok(WriteResult::new(last_insert_id, rows_affected, None))
    }

    /// Begins transaction and returns Transaction object. The transaction
    /// gets the isolation level set with
    /// [`OssConnection::set_isolation_level`] unless `tx_opts` sets one.
    pub async fn begin_transaction(
        &self,
        mut tx_opts: TxOpts,
    ) -> Result<Transaction<'static>, Error> {
        if let (None, Some(isolation_level)) = (tx_opts.isolation_level(), self.isolation_level) {
            tx_opts.with_isolation_level(mysql_async::IsolationLevel::from(isolation_level));
        }
        let tr = self
            .pool
            .start_transaction(tx_opts)
            .await
            .map_err(OssMysqlError::into_mysql)?;

        Ok(tr)
    }
}

/// Converts `err`, the error of a statement run on `conn`, for the caller.
///
/// After a timeout, `conn` is closed rather than returned for reuse. Rows
/// are read outside the routine that marks a connection broken, so a read
/// that timed out while rows were arriving leaves a healthy-looking
/// connection with the rest of its result unread, and the pool would read
/// all of that, with no time limit, before reusing it. `Conn::disconnect`
/// marks the connection disconnected before it does anything else, and only
/// then reads that rest, so it is polled just once and dropped: the pool
/// discards the connection and closes its socket at once. The server only
/// notices when it next sends rows, which for a small result can be when
/// the query ends, so this frees the pool slot but does not stop the query.
fn discard_after_timeout(conn: MysqlConnection, err: OssMysqlError) -> mysql_async::Error {
    if matches!(err.kind, OssErrorKind::Timeout(_)) {
        let _ = conn.disconnect().now_or_never();
    }
    err.into_mysql()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use mysql_async::IsolationLevel as MysqlIsolationLevel;

    use super::*;
    use crate::mysql::OssPhase;
    use crate::mysql::OssPoolState;
    use crate::mysql::fake_mysqld::FakeMysqld;
    use crate::mysql::fake_mysqld::Reply;

    fn stats() -> Arc<ConnectionStats> {
        Arc::new(ConnectionStats::new("test".to_string()))
    }

    #[tokio::test]
    async fn read_timeout_discards_the_connection() {
        let server = FakeMysqld::start(|query| match query {
            "SELECT stalls" => Reply::PartialRows,
            _ => Reply::Ok,
        })
        .await;
        let pool = OssPool::builder(OssPoolState::Active {
            generation: 0,
            pool: server.pool(1),
            endpoint: Arc::from("fake"),
        })
        .checkout_timeout(Duration::from_secs(5))
        .query_timeout(Duration::from_millis(300))
        .build();
        let conn = OssConnection::new(pool, stats());

        // The server sends the header and the first row, then stalls.
        let err = conn.read_rows("SELECT stalls", Ok).await.unwrap_err();
        let err = err
            .chain()
            .find_map(|e| e.downcast_ref::<OssMysqlError>())
            .unwrap();
        assert_eq!((err.phase, err.stmt), (OssPhase::Execute, OssStmt::Read));
        assert!(matches!(err.kind, OssErrorKind::Timeout(_)));

        // The only slot is free again at once: its connection was closed,
        // not kept to read the rest of the result.
        let rows = conn.read_rows("SELECT 1", Ok).await.unwrap();
        assert!(rows.is_empty());
        let result = conn.write_query("UPDATE t SET x = 1".to_string()).await;
        assert_eq!(result.unwrap().rows_affected(), 1);
        assert_eq!(
            server.queries(),
            vec!["SELECT stalls", "SELECT 1", "UPDATE t SET x = 1"]
        );
    }

    #[tokio::test]
    async fn transactions_use_the_isolation_level() {
        async fn start(conn: &OssConnection) {
            let connection = crate::Connection::OssMysql(conn.clone());
            let transaction = connection.start_transaction().await.unwrap();
            transaction.commit().await.unwrap();
        }
        let server = FakeMysqld::start(|_| Reply::Ok).await;
        let mut conn = OssConnection::new(server.pool(1), stats());

        start(&conn).await;
        conn.set_isolation_level(Some(IsolationLevel::ReadCommitted));
        start(&conn).await;
        // Options that set a level take precedence.
        let mut tx_opts = TxOpts::default();
        tx_opts.with_isolation_level(MysqlIsolationLevel::Serializable);
        conn.begin_transaction(tx_opts)
            .await
            .unwrap()
            .commit()
            .await
            .unwrap();
        // Autocommit statements keep the server's default.
        conn.write_query("UPDATE t SET x = 1".to_string())
            .await
            .unwrap();
        conn.set_isolation_level(None);
        start(&conn).await;

        assert_eq!(
            server.queries(),
            vec![
                "START TRANSACTION",
                "COMMIT",
                "SET TRANSACTION ISOLATION LEVEL READ COMMITTED",
                "START TRANSACTION",
                "COMMIT",
                "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE",
                "START TRANSACTION",
                "COMMIT",
                "UPDATE t SET x = 1",
                "START TRANSACTION",
                "COMMIT",
            ]
        );
    }
}
