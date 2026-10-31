/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is dual-licensed under either the MIT license found in the
 * LICENSE-MIT file in the root directory of this source tree or the Apache
 * License, Version 2.0 found in the LICENSE-APACHE file in the root directory
 * of this source tree. You may select, at your option, one of the
 * above-listed licenses.
 */

//! A swappable handle over a `mysql_async` connection pool.
//!
//! [`OssPool`] lets its owner point it at a different [`Pool`] at run time,
//! for example to follow a change of MySQL primary, while every user keeps the
//! same handle. Callers that check out a connection while there is no usable
//! server wait for the next change, up to the checkout timeout. A checkout
//! still in progress on the previous pool when the state changes starts over
//! on the new state.
//!
//! Every error is an [`OssMysqlError`] that records at which [`OssPhase`]
//! the operation failed, so that callers can tell whether a retry is safe.
//!
//! An [`OssPool`] made with `From<Pool>` never changes and has no timeouts,
//! so it behaves like the [`Pool`] it wraps.

use std::any::Any;
use std::fmt;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;

use mysql_async::Conn;
use mysql_async::DriverError;
use mysql_async::Pool;
use mysql_async::Transaction;
use mysql_async::TxOpts;
use tokio::sync::watch;
use tokio::time::Instant;

/// Endpoint recorded for a pool made with `From<Pool>`.
const FIXED_ENDPOINT: &str = "fixed";
/// Endpoint recorded while an [`OssPool`] has no usable server.
const NO_ENDPOINT: &str = "<none>";

/// Statement timeout inside transactions, in milliseconds. 0 means none.
static OSS_TRANSACTION_STATEMENT_TIMEOUT_MS: AtomicU64 = AtomicU64::new(0);

/// The step of an OSS MySQL operation at which an error happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OssPhase {
    /// Getting a connection: waiting for a usable server, connecting,
    /// authenticating, running the pool's setup statements and, for a
    /// transaction, starting it. No statement of the caller has run.
    Checkout,
    /// Running an autocommit statement on a checked-out connection.
    Execute,
    /// Running a statement inside an open transaction.
    InTransaction,
    /// Committing a transaction.
    Commit,
}

/// Whether the statement that failed reads or writes data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OssStmt {
    /// A statement that does not change data.
    Read,
    /// A statement that may change data.
    Write,
    /// Not known: a checkout, or a statement inside a transaction.
    Unknown,
}

/// What went wrong in an [`OssMysqlError`].
#[derive(Debug, thiserror::Error)]
pub enum OssErrorKind {
    /// The MySQL client or server returned an error.
    #[error(transparent)]
    Mysql(mysql_async::Error),
    /// The operation did not finish within the given time and was dropped.
    #[error("timed out after {0:?}")]
    Timeout(Duration),
    /// The pool had no usable server until the checkout timeout.
    #[error("no usable MySQL primary: {0}")]
    Unavailable(Arc<str>),
}

/// An error of an [`OssPool`] operation or of a transaction statement,
/// tagged with where it happened.
#[derive(Debug, thiserror::Error)]
#[error(
    "OSS MySQL {phase:?}/{stmt:?} error on {endpoint}{}",
    generation_note(.phase, .generation)
)]
pub struct OssMysqlError {
    /// The step at which the error happened.
    pub phase: OssPhase,
    /// The kind of statement that failed.
    pub stmt: OssStmt,
    /// Where the pool pointed when the operation started. For a statement
    /// inside a transaction, where its connection is connected: see
    /// [`oss_tx_endpoint`].
    pub endpoint: Arc<str>,
    /// The pool generation when the operation started. Statements inside a
    /// transaction, its COMMIT included, do not know it: their errors have
    /// 0, and do not show it.
    pub generation: u64,
    /// What went wrong.
    #[source]
    pub kind: OssErrorKind,
}

/// How an [`OssMysqlError`] shows its generation, if it knows it.
fn generation_note(phase: &OssPhase, generation: &u64) -> String {
    match phase {
        OssPhase::Checkout | OssPhase::Execute => format!(" (generation {generation})"),
        OssPhase::InTransaction | OssPhase::Commit => String::new(),
    }
}

impl OssMysqlError {
    /// Wraps this error as `mysql_async::Error::Other`, for APIs that return
    /// `mysql_async::Error`. It can be found again by downcasting the sources
    /// in the error chain.
    pub fn into_mysql(self) -> mysql_async::Error {
        mysql_async::Error::Other(Box::new(self))
    }
}

/// Called with every error an [`OssPool`] returns from a checkout or a
/// statement, before the error is returned.
pub type OssErrorHook = Arc<dyn Fn(&OssMysqlError) + Send + Sync>;

/// Called when a checkout finds an [`OssPool`] unavailable, before it waits
/// for the next change, with the generation and reason of that state.
pub type OssWaitHook = Arc<dyn Fn(u64, &str) + Send + Sync>;

/// What an [`OssPool`] currently points at.
#[derive(Clone)]
pub enum OssPoolState {
    /// Connections come from `pool`.
    Active {
        /// Increased by one on every change of state.
        generation: u64,
        /// The pool connections are checked out from.
        pool: Pool,
        /// Where `pool` connects to, for errors and logs.
        endpoint: Arc<str>,
    },
    /// There is no usable server. Checkouts wait for the next change.
    Unavailable {
        /// Increased by one on every change of state.
        generation: u64,
        /// Why there is no usable server.
        reason: Arc<str>,
    },
}

impl OssPoolState {
    /// The generation of this state.
    pub fn generation(&self) -> u64 {
        match self {
            OssPoolState::Active { generation, .. } => *generation,
            OssPoolState::Unavailable { generation, .. } => *generation,
        }
    }
}

impl fmt::Debug for OssPoolState {
    // The pool is left out: its options include the password.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OssPoolState::Active {
                generation,
                endpoint,
                ..
            } => f
                .debug_struct("Active")
                .field("generation", generation)
                .field("endpoint", endpoint)
                .finish_non_exhaustive(),
            OssPoolState::Unavailable { generation, reason } => f
                .debug_struct("Unavailable")
                .field("generation", generation)
                .field("reason", reason)
                .finish(),
        }
    }
}

/// A handle over a MySQL connection pool that its owner can swap at run time.
///
/// Clones share the same state. Only one task should change it, with
/// [`OssPool::replace`] and [`OssPool::mark_unavailable`].
#[derive(Clone)]
pub struct OssPool {
    inner: Arc<OssPoolInner>,
    /// Kept alive by this handle and its clones: see
    /// [`OssPool::with_keepalive`].
    _keepalive: Option<Arc<dyn Any + Send + Sync>>,
}

struct OssPoolInner {
    state: watch::Sender<OssPoolState>,
    checkout_timeout: Option<Duration>,
    query_timeout: Option<Duration>,
    on_error: Option<OssErrorHook>,
    on_wait: Option<OssWaitHook>,
}

/// Builds an [`OssPool`]. Settings that are not given are off: no timeouts
/// and no hooks.
pub struct OssPoolBuilder {
    initial: OssPoolState,
    checkout_timeout: Option<Duration>,
    query_timeout: Option<Duration>,
    on_error: Option<OssErrorHook>,
    on_wait: Option<OssWaitHook>,
}

impl OssPoolBuilder {
    /// Limits the time a checkout may take in total, including the time
    /// spent waiting for a usable server.
    pub fn checkout_timeout(mut self, timeout: Duration) -> Self {
        self.checkout_timeout = Some(timeout);
        self
    }

    /// Limits the time a statement run with [`OssPool::execute`] may take.
    pub fn query_timeout(mut self, timeout: Duration) -> Self {
        self.query_timeout = Some(timeout);
        self
    }

    /// Sets the hook called with every error the pool returns.
    pub fn on_error(mut self, hook: OssErrorHook) -> Self {
        self.on_error = Some(hook);
        self
    }

    /// Sets the hook called each time a checkout finds the pool unavailable
    /// and starts to wait for the next change. A waiting checkout returns no
    /// error before the checkout timeout, or ever when there is none, so the
    /// error hook cannot tell that callers are waiting. This hook lets the
    /// owner react at once, for example by looking for a usable server. It
    /// runs on the caller's task, once for each checkout and unavailable
    /// state, so it should be cheap.
    pub fn on_wait(mut self, hook: OssWaitHook) -> Self {
        self.on_wait = Some(hook);
        self
    }

    /// Creates the pool.
    pub fn build(self) -> OssPool {
        OssPool {
            inner: Arc::new(OssPoolInner {
                state: watch::Sender::new(self.initial),
                checkout_timeout: self.checkout_timeout,
                query_timeout: self.query_timeout,
                on_error: self.on_error,
                on_wait: self.on_wait,
            }),
            _keepalive: None,
        }
    }
}

impl From<Pool> for OssPool {
    /// A pool that never changes, with no timeouts and no hooks: it behaves
    /// like `pool` itself.
    fn from(pool: Pool) -> Self {
        OssPool::builder(OssPoolState::Active {
            generation: 0,
            pool,
            endpoint: Arc::from(FIXED_ENDPOINT),
        })
        .build()
    }
}

impl OssPool {
    /// Starts building a pool with the given initial state.
    pub fn builder(initial: OssPoolState) -> OssPoolBuilder {
        OssPoolBuilder {
            initial,
            checkout_timeout: None,
            query_timeout: None,
            on_error: None,
            on_wait: None,
        }
    }

    /// A handle to the same pool that also keeps `keepalive` alive for as
    /// long as it or one of its clones exists. It replaces any keepalive of
    /// this handle, and other handles are not affected. This lets the owner
    /// of a pool give its users handles that keep the owner alive, while it
    /// keeps a handle without the keepalive for itself, so there is no
    /// reference cycle.
    pub fn with_keepalive(&self, keepalive: Arc<dyn Any + Send + Sync>) -> OssPool {
        OssPool {
            inner: self.inner.clone(),
            _keepalive: Some(keepalive),
        }
    }

    /// The current state.
    pub fn state(&self) -> OssPoolState {
        self.inner.state.borrow().clone()
    }

    /// The current generation.
    pub fn generation(&self) -> u64 {
        self.inner.state.borrow().generation()
    }

    /// Points the pool at `pool` and returns the new generation.
    ///
    /// The previous pool, if any, is disconnected in the background: its idle
    /// connections close at once and busy ones when they are returned.
    /// Checkouts in progress on it, whether queued for a free slot,
    /// connecting or checking an idle connection, give up their attempt and
    /// start over on `pool`, within their original checkout timeout. Callers
    /// that already hold a connection of the previous pool keep it. `pool`
    /// must be a new pool, not a clone of the current one.
    pub fn replace(&self, pool: Pool, endpoint: impl Into<Arc<str>>) -> u64 {
        let endpoint = endpoint.into();
        self.swap(move |generation| OssPoolState::Active {
            generation,
            pool,
            endpoint,
        })
    }

    /// Records that there is no usable server and returns the new
    /// generation. The previous pool, if any, is disconnected in the
    /// background. Checkouts, including those in progress on the previous
    /// pool, wait for the next change, up to the checkout timeout.
    pub fn mark_unavailable(&self, reason: impl Into<Arc<str>>) -> u64 {
        let reason = reason.into();
        self.swap(move |generation| OssPoolState::Unavailable { generation, reason })
    }

    /// Checks out a connection from the current pool.
    pub async fn get_conn(&self) -> Result<Conn, OssMysqlError> {
        self.checkout(|pool| async move { pool.get_conn().await })
            .await
    }

    /// Checks out a connection from the current pool and starts a
    /// transaction on it.
    pub async fn start_transaction(
        &self,
        options: TxOpts,
    ) -> Result<Transaction<'static>, OssMysqlError> {
        self.checkout(|pool| {
            let options = options.clone();
            async move { pool.start_transaction(options).await }
        })
        .await
    }

    /// Runs `f`, a statement on a connection from this pool, within the query
    /// timeout. Errors are tagged with [`OssPhase::Execute`], `stmt`, and the
    /// state current when the statement started.
    pub async fn execute<T>(
        &self,
        stmt: OssStmt,
        f: impl Future<Output = Result<T, mysql_async::Error>>,
    ) -> Result<T, OssMysqlError> {
        let (generation, endpoint) = self.current_endpoint();
        bounded(deadline_after(self.inner.query_timeout), f)
            .await
            .map_err(|kind| self.fail(OssPhase::Execute, stmt, endpoint, generation, kind))
    }

    fn swap(&self, new_state: impl FnOnce(u64) -> OssPoolState) -> u64 {
        let mut generation = 0;
        let mut old = None;
        self.inner.state.send_modify(|state| {
            generation = state.generation() + 1;
            old = Some(std::mem::replace(state, new_state(generation)));
        });
        if let Some(OssPoolState::Active { pool, .. }) = old {
            disconnect_in_background(pool);
        }
        generation
    }

    fn current_endpoint(&self) -> (u64, Arc<str>) {
        match &*self.inner.state.borrow() {
            OssPoolState::Active {
                generation,
                endpoint,
                ..
            } => (*generation, endpoint.clone()),
            OssPoolState::Unavailable { generation, .. } => (*generation, Arc::from(NO_ENDPOINT)),
        }
    }

    async fn checkout<T, F, Fut>(&self, mut op: F) -> Result<T, OssMysqlError>
    where
        F: FnMut(Pool) -> Fut,
        Fut: Future<Output = Result<T, mysql_async::Error>>,
    {
        let deadline = deadline_after(self.inner.checkout_timeout);
        let mut rx = self.inner.state.subscribe();
        loop {
            let state = rx.borrow_and_update().clone();
            match state {
                OssPoolState::Active {
                    generation,
                    pool,
                    endpoint,
                } => {
                    let result = tokio::select! {
                        biased;
                        result = bounded(deadline, op(pool)) => result,
                        // The state changed while this attempt was in
                        // progress. Disconnecting the old pool neither wakes
                        // callers queued for one of its slots nor stops a
                        // connect, which may never finish. Dropping the
                        // attempt gives its place in the old pool back.
                        Ok(()) = rx.changed() => continue,
                    };
                    match result {
                        // The state changed as the attempt finished. Dropping
                        // the connection returns it to the old pool, which
                        // discards it (rolling back a transaction first).
                        // This narrows the window in which a caller gets a
                        // connection of a superseded generation, but cannot
                        // close it: the state can change right after the
                        // check.
                        Ok(value) if self.generation() != generation => {
                            drop(value);
                            continue;
                        }
                        Ok(value) => return Ok(value),
                        // The pool was swapped and disconnected under this
                        // call: try again with the new state.
                        Err(OssErrorKind::Mysql(mysql_async::Error::Driver(
                            DriverError::PoolDisconnected,
                        ))) if self.generation() != generation => continue,
                        Err(kind) => {
                            return Err(self.fail(
                                OssPhase::Checkout,
                                OssStmt::Unknown,
                                endpoint,
                                generation,
                                kind,
                            ));
                        }
                    }
                }
                OssPoolState::Unavailable { generation, reason } => {
                    if let Some(on_wait) = &self.inner.on_wait {
                        on_wait(generation, &reason);
                    }
                    let changed = match deadline {
                        Some((deadline, _)) => {
                            matches!(
                                tokio::time::timeout_at(deadline, rx.changed()).await,
                                Ok(Ok(()))
                            )
                        }
                        None => rx.changed().await.is_ok(),
                    };
                    if !changed {
                        return Err(self.fail(
                            OssPhase::Checkout,
                            OssStmt::Unknown,
                            Arc::from(NO_ENDPOINT),
                            generation,
                            OssErrorKind::Unavailable(reason),
                        ));
                    }
                }
            }
        }
    }

    fn fail(
        &self,
        phase: OssPhase,
        stmt: OssStmt,
        endpoint: Arc<str>,
        generation: u64,
        kind: OssErrorKind,
    ) -> OssMysqlError {
        let err = OssMysqlError {
            phase,
            stmt,
            endpoint,
            generation,
            kind,
        };
        if let Some(on_error) = &self.inner.on_error {
            on_error(&err);
        }
        err
    }
}

/// Sets the timeout of every statement, including COMMIT, run inside an OSS
/// MySQL transaction. `None` means no timeout. It applies to the whole
/// process.
pub fn set_oss_transaction_statement_timeout(timeout: Option<Duration>) {
    let millis = match timeout {
        // Round sub-millisecond timeouts up so that they are not taken as none.
        Some(timeout) => u64::try_from(timeout.as_millis())
            .unwrap_or(u64::MAX)
            .max(1),
        None => 0,
    };
    OSS_TRANSACTION_STATEMENT_TIMEOUT_MS.store(millis, Ordering::Relaxed);
}

/// The timeout of statements inside OSS MySQL transactions, if any.
pub fn oss_transaction_statement_timeout() -> Option<Duration> {
    match OSS_TRANSACTION_STATEMENT_TIMEOUT_MS.load(Ordering::Relaxed) {
        0 => None,
        millis => Some(Duration::from_millis(millis)),
    }
}

/// Where `conn`, the connection of an OSS MySQL transaction, is connected,
/// for the errors of its statements: the server and database, as pools give
/// them, and the connection id, which the server's logs and process list
/// show.
pub fn oss_tx_endpoint(conn: &Conn) -> Arc<str> {
    let opts = conn.opts();
    let host = opts.ip_or_hostname();
    let host = if host.contains(':') {
        format!("[{host}]")
    } else {
        host.to_string()
    };
    Arc::from(format!(
        "{host}:{}/{} (connection {})",
        opts.tcp_port(),
        opts.db_name().unwrap_or_default(),
        conn.id()
    ))
}

/// Runs `f`, a statement inside an OSS MySQL transaction on the connection
/// at `endpoint` (see [`oss_tx_endpoint`]), within the transaction statement
/// timeout. Every error is returned as an [`OssMysqlError`] with the given
/// `phase`, wrapped by [`OssMysqlError::into_mysql`].
pub async fn oss_tx_guard<T>(
    phase: OssPhase,
    endpoint: Arc<str>,
    f: impl Future<Output = Result<T, mysql_async::Error>>,
) -> Result<T, mysql_async::Error> {
    bounded(deadline_after(oss_transaction_statement_timeout()), f)
        .await
        .map_err(|kind| {
            OssMysqlError {
                phase,
                stmt: OssStmt::Unknown,
                endpoint,
                generation: 0,
                kind,
            }
            .into_mysql()
        })
}

/// The deadline `timeout` from now, with the timeout itself. `None` when
/// there is no timeout or the deadline is out of range.
fn deadline_after(timeout: Option<Duration>) -> Option<(Instant, Duration)> {
    let timeout = timeout?;
    Some((Instant::now().checked_add(timeout)?, timeout))
}

/// Runs `f` until the deadline, if any.
async fn bounded<T>(
    deadline: Option<(Instant, Duration)>,
    f: impl Future<Output = Result<T, mysql_async::Error>>,
) -> Result<T, OssErrorKind> {
    match deadline {
        Some((deadline, timeout)) => match tokio::time::timeout_at(deadline, f).await {
            Ok(result) => result.map_err(OssErrorKind::Mysql),
            Err(_) => Err(OssErrorKind::Timeout(timeout)),
        },
        None => f.await.map_err(OssErrorKind::Mysql),
    }
}

fn disconnect_in_background(pool: Pool) {
    // Without a runtime nothing can drive the disconnect. Dropping the handle
    // lets the pool close once its other handles are gone.
    if let Ok(runtime) = tokio::runtime::Handle::try_current() {
        runtime.spawn(async move {
            let _ = pool.disconnect().await;
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use futures::future;
    use mysql_async::OptsBuilder;
    use mysql_async::PoolConstraints;
    use mysql_async::PoolOpts;
    use mysql_async::ServerError;
    use tokio::net::TcpListener;
    use tokio::net::TcpStream;
    use tokio::sync::Notify;
    use tokio::task::JoinHandle;

    use super::*;
    use crate::mysql::ConnectionStats;
    use crate::mysql::OssConnection;
    use crate::mysql::fake_mysqld::FakeMysqld;
    use crate::mysql::fake_mysqld::Reply;

    /// Held by the tests that change the transaction statement timeout,
    /// which is global.
    static STATEMENT_TIMEOUT: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// A pool to 127.0.0.1:`port` with at most `max` connections. Creating
    /// it does not connect.
    fn pool_to(port: u16, max: usize) -> Pool {
        let pool_opts = PoolOpts::default().with_constraints(PoolConstraints::new(0, max).unwrap());
        Pool::new(
            OptsBuilder::default()
                .ip_or_hostname("127.0.0.1")
                .tcp_port(port)
                .prefer_socket(false)
                .pool_opts(pool_opts),
        )
    }

    /// A listener that never answers: connections to it hang in the
    /// handshake.
    async fn silent_listener() -> (TcpListener, u16) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        (listener, port)
    }

    /// A port nothing listens on.
    async fn closed_port() -> u16 {
        let (_listener, port) = silent_listener().await;
        port
    }

    fn server_error(code: u16) -> mysql_async::Error {
        mysql_async::Error::Server(ServerError {
            code,
            message: format!("error {code}"),
            state: "HY000".to_string(),
        })
    }

    /// Paused time advances to timer ticks, which are rounded to milliseconds.
    fn assert_ten_seconds(elapsed: Duration) {
        assert!(
            elapsed >= Duration::from_secs(10) && elapsed < Duration::from_millis(10_010),
            "{elapsed:?}"
        );
    }

    fn unavailable(reason: &str) -> OssPoolState {
        OssPoolState::Unavailable {
            generation: 0,
            reason: Arc::from(reason),
        }
    }

    /// The phase and statement kind of every error a hook saw.
    type Seen = Arc<Mutex<Vec<(OssPhase, OssStmt)>>>;

    fn recording_hook() -> (OssErrorHook, Seen) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let hook: OssErrorHook = Arc::new({
            let seen = seen.clone();
            move |err: &OssMysqlError| seen.lock().unwrap().push((err.phase, err.stmt))
        });
        (hook, seen)
    }

    #[tokio::test]
    async fn replace_bumps_the_generation() {
        let pool = OssPool::from(pool_to(closed_port().await, 1));
        assert_eq!(pool.generation(), 0);

        assert_eq!(pool.replace(pool_to(closed_port().await, 1), "a:1"), 1);
        assert_eq!(pool.generation(), 1);
        assert!(matches!(
            pool.state(),
            OssPoolState::Active { generation: 1, endpoint, .. } if &*endpoint == "a:1"
        ));

        assert_eq!(pool.mark_unavailable("no primary"), 2);
        assert!(matches!(
            pool.state(),
            OssPoolState::Unavailable { generation: 2, reason } if &*reason == "no primary"
        ));

        assert_eq!(pool.replace(pool_to(closed_port().await, 1), "b:1"), 3);
        // Clones share the state.
        assert_eq!(pool.clone().generation(), 3);
    }

    /// A pool whose only slot is taken by a checkout through it that hangs
    /// in the handshake with `listener`, which accepts it. Nothing ever
    /// frees the slot. Returns the checkout and the accepted socket, which
    /// must be kept open.
    async fn stalled_checkout(
        pool: &OssPool,
        listener: &TcpListener,
    ) -> (JoinHandle<Result<Conn, OssMysqlError>>, TcpStream) {
        let checkout = tokio::spawn({
            let pool = pool.clone();
            async move { pool.get_conn().await }
        });
        let (socket, _) = listener.accept().await.unwrap();
        (checkout, socket)
    }

    #[tokio::test]
    async fn checkouts_in_progress_move_to_a_replaced_pool() {
        let (listener, port) = silent_listener().await;
        let server = FakeMysqld::start(|_| Reply::Ok).await;
        let pool = OssPool::builder(OssPoolState::Active {
            generation: 0,
            pool: pool_to(port, 1),
            endpoint: Arc::from("old"),
        })
        .checkout_timeout(Duration::from_secs(60))
        .build();

        // One checkout connects and never gets an answer. Another queues
        // for the slot it holds.
        let (connecting, _socket) = stalled_checkout(&pool, &listener).await;
        let queued = tokio::spawn({
            let pool = pool.clone();
            async move { pool.start_transaction(TxOpts::default()).await }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!connecting.is_finished());
        assert!(!queued.is_finished());

        // Both move to the new pool at once, long before their deadline.
        assert_eq!(pool.replace(server.pool(2), "new"), 1);
        let (connecting, queued) = tokio::time::timeout(Duration::from_secs(10), async {
            (connecting.await.unwrap(), queued.await.unwrap())
        })
        .await
        .expect("checkouts stayed on the old pool");
        let _conn = connecting.unwrap();
        let _transaction = queued.unwrap();
        assert_eq!(server.queries(), vec!["START TRANSACTION"]);
    }

    #[tokio::test]
    async fn checkouts_in_progress_wait_when_marked_unavailable() {
        let (listener, port) = silent_listener().await;
        let server = FakeMysqld::start(|_| Reply::Ok).await;
        let waits = Arc::new(Mutex::new(Vec::new()));
        let waiting = Arc::new(Notify::new());
        let hook: OssWaitHook = Arc::new({
            let waits = waits.clone();
            let waiting = waiting.clone();
            move |generation, reason: &str| {
                waits.lock().unwrap().push((generation, reason.to_string()));
                waiting.notify_one();
            }
        });
        let pool = OssPool::builder(OssPoolState::Active {
            generation: 0,
            pool: pool_to(port, 1),
            endpoint: Arc::from("old"),
        })
        .checkout_timeout(Duration::from_secs(60))
        .on_wait(hook)
        .build();
        let (checkout, _socket) = stalled_checkout(&pool, &listener).await;

        // The checkout leaves the old pool and waits for a usable server.
        assert_eq!(pool.mark_unavailable("split brain"), 1);
        tokio::time::timeout(Duration::from_secs(10), waiting.notified())
            .await
            .expect("the checkout stayed on the old pool");
        assert_eq!(*waits.lock().unwrap(), vec![(1, "split brain".to_string())]);
        assert!(!checkout.is_finished());

        assert_eq!(pool.replace(server.pool(1), "new"), 2);
        tokio::time::timeout(Duration::from_secs(10), checkout)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(waits.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn wait_hook_lets_the_owner_end_an_unbounded_wait() {
        // Like an owner that looks for a server only when asked to.
        let asked = Arc::new(Notify::new());
        let hook: OssWaitHook = Arc::new({
            let asked = asked.clone();
            move |_: u64, _: &str| asked.notify_one()
        });
        let pool = OssPool::builder(unavailable("no primary"))
            .on_wait(hook)
            .build();
        let owner = tokio::spawn({
            let pool = pool.clone();
            async move {
                asked.notified().await;
                pool.replace(pool_to(closed_port().await, 1), "new")
            }
        });

        // With no checkout timeout, nothing else would end this wait.
        let err = tokio::time::timeout(Duration::from_secs(10), pool.get_conn())
            .await
            .expect("the owner was not asked")
            .unwrap_err();
        assert_eq!(owner.await.unwrap(), 1);
        assert_eq!(&*err.endpoint, "new");
        assert!(matches!(
            err.kind,
            OssErrorKind::Mysql(mysql_async::Error::Io(_))
        ));
    }

    #[tokio::test]
    async fn checkout_retries_an_attempt_of_a_superseded_generation() {
        let pool = OssPool::from(pool_to(closed_port().await, 1));
        let pools = [
            pool_to(closed_port().await, 1),
            pool_to(closed_port().await, 1),
        ];

        // The first two attempts change the state as they finish, so that
        // the change and the result are ready together.
        let mut attempts = 0;
        let value = pool
            .checkout(|_| {
                attempts += 1;
                let result = match attempts {
                    1 => Ok(attempts),
                    2 => Err(mysql_async::Error::Driver(DriverError::PoolDisconnected)),
                    _ => return future::ready(Ok(attempts)),
                };
                pool.replace(pools[attempts - 1].clone(), "new");
                future::ready(result)
            })
            .await
            .unwrap();
        assert_eq!(value, 3);
        assert_eq!(pool.generation(), 2);
    }

    #[tokio::test]
    async fn keepalive_lives_as_long_as_its_handles() {
        let pool = OssPool::from(pool_to(closed_port().await, 1));
        let owner = Arc::new(());
        let weak = Arc::downgrade(&owner);
        let handle = pool.with_keepalive(owner);

        let clone = handle.clone();
        drop(handle);
        assert!(weak.upgrade().is_some());

        // Handles share the state either way.
        assert_eq!(clone.replace(pool_to(closed_port().await, 1), "new"), 1);
        assert_eq!(pool.generation(), 1);

        drop(clone);
        assert!(weak.upgrade().is_none());
        assert_eq!(pool.generation(), 1);
    }

    #[tokio::test]
    async fn get_conn_on_a_disconnected_pool_fails_when_not_swapped() {
        let old = pool_to(closed_port().await, 1);
        old.clone().disconnect().await.unwrap();
        let err = OssPool::from(old).get_conn().await.unwrap_err();
        assert_eq!(err.phase, OssPhase::Checkout);
        assert!(matches!(
            err.kind,
            OssErrorKind::Mysql(mysql_async::Error::Driver(DriverError::PoolDisconnected))
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn unavailable_parks_and_wakes_on_replace() {
        let (_listener, port) = silent_listener().await;
        let pool = OssPool::builder(unavailable("no primary"))
            .checkout_timeout(Duration::from_secs(10))
            .build();
        let start = Instant::now();
        let checkout = tokio::spawn({
            let pool = pool.clone();
            async move { pool.get_conn().await }
        });

        tokio::time::sleep(Duration::from_secs(4)).await;
        assert!(!checkout.is_finished());
        assert_eq!(pool.replace(pool_to(port, 1), "new"), 1);

        // The checkout moved to the new pool, whose server never answers, and
        // ran out of the time left.
        let err = checkout.await.unwrap().unwrap_err();
        assert_eq!(err.phase, OssPhase::Checkout);
        assert_eq!(&*err.endpoint, "new");
        assert_eq!(err.generation, 1);
        assert!(matches!(err.kind, OssErrorKind::Timeout(t) if t == Duration::from_secs(10)));
        assert_ten_seconds(start.elapsed());
    }

    #[tokio::test(start_paused = true)]
    async fn unavailable_times_out() {
        let pool = OssPool::builder(unavailable("split brain"))
            .checkout_timeout(Duration::from_secs(10))
            .build();
        let start = Instant::now();
        let checkout = tokio::spawn({
            let pool = pool.clone();
            async move { pool.get_conn().await }
        });

        // A change that leaves the pool unavailable keeps the checkout waiting.
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert_eq!(pool.mark_unavailable("still split"), 1);

        let err = checkout.await.unwrap().unwrap_err();
        assert_eq!(err.phase, OssPhase::Checkout);
        assert_eq!(err.stmt, OssStmt::Unknown);
        assert_eq!(&*err.endpoint, NO_ENDPOINT);
        assert_eq!(err.generation, 1);
        assert!(
            matches!(&err.kind, OssErrorKind::Unavailable(reason) if &**reason == "still split")
        );
        assert_ten_seconds(start.elapsed());

        // Transactions wait the same way.
        let err = pool.start_transaction(TxOpts::default()).await.unwrap_err();
        assert_eq!(err.phase, OssPhase::Checkout);
        assert!(matches!(err.kind, OssErrorKind::Unavailable(_)));
    }

    #[tokio::test(start_paused = true)]
    async fn silent_server_times_out_at_checkout() {
        let (_listener, port) = silent_listener().await;
        let pool = OssPool::builder(OssPoolState::Active {
            generation: 7,
            pool: pool_to(port, 1),
            endpoint: Arc::from("silent"),
        })
        .checkout_timeout(Duration::from_secs(2))
        .build();

        let err = pool.get_conn().await.unwrap_err();
        assert_eq!(err.phase, OssPhase::Checkout);
        assert_eq!(&*err.endpoint, "silent");
        assert_eq!(err.generation, 7);
        assert!(matches!(err.kind, OssErrorKind::Timeout(t) if t == Duration::from_secs(2)));

        let err = pool.start_transaction(TxOpts::default()).await.unwrap_err();
        assert_eq!(err.phase, OssPhase::Checkout);
        assert!(matches!(err.kind, OssErrorKind::Timeout(_)));
    }

    #[tokio::test(start_paused = true)]
    async fn execute_tags_errors() {
        let pool = OssPool::builder(OssPoolState::Active {
            generation: 3,
            pool: pool_to(closed_port().await, 1),
            endpoint: Arc::from("db:3306"),
        })
        .query_timeout(Duration::from_secs(1))
        .build();

        assert_eq!(
            pool.execute(OssStmt::Read, async { Ok(5) }).await.unwrap(),
            5
        );

        let err = pool
            .execute(OssStmt::Write, std::future::pending::<Result<(), _>>())
            .await
            .unwrap_err();
        assert_eq!(err.phase, OssPhase::Execute);
        assert_eq!(err.stmt, OssStmt::Write);
        assert_eq!(&*err.endpoint, "db:3306");
        assert_eq!(err.generation, 3);
        assert!(matches!(err.kind, OssErrorKind::Timeout(t) if t == Duration::from_secs(1)));

        let err = pool
            .execute(OssStmt::Read, async { Err::<(), _>(server_error(1290)) })
            .await
            .unwrap_err();
        assert_eq!(err.phase, OssPhase::Execute);
        assert_eq!(err.stmt, OssStmt::Read);
        assert!(matches!(
            err.kind,
            OssErrorKind::Mysql(mysql_async::Error::Server(ServerError { code: 1290, .. }))
        ));

        pool.mark_unavailable("gone");
        let err = pool
            .execute(OssStmt::Read, async { Err::<(), _>(server_error(1053)) })
            .await
            .unwrap_err();
        assert_eq!(&*err.endpoint, NO_ENDPOINT);
        assert_eq!(err.generation, 4);
    }

    #[tokio::test(start_paused = true)]
    async fn hook_sees_every_error() {
        let (hook, seen) = recording_hook();
        let pool = OssPool::builder(unavailable("none yet"))
            .checkout_timeout(Duration::from_secs(1))
            .query_timeout(Duration::from_secs(1))
            .on_error(hook)
            .build();

        pool.execute(OssStmt::Read, async { Ok(()) }).await.unwrap();
        assert!(seen.lock().unwrap().is_empty());

        pool.get_conn().await.unwrap_err();
        pool.execute(OssStmt::Write, async { Err::<(), _>(server_error(1290)) })
            .await
            .unwrap_err();
        pool.execute(OssStmt::Read, std::future::pending::<Result<(), _>>())
            .await
            .unwrap_err();

        assert_eq!(
            *seen.lock().unwrap(),
            vec![
                (OssPhase::Checkout, OssStmt::Unknown),
                (OssPhase::Execute, OssStmt::Write),
                (OssPhase::Execute, OssStmt::Read),
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn fixed_pool_has_no_timeouts() {
        let (_listener, port) = silent_listener().await;
        let pool = OssPool::from(pool_to(port, 1));
        assert_eq!(pool.generation(), 0);
        assert!(matches!(
            pool.state(),
            OssPoolState::Active { generation: 0, endpoint, .. } if &*endpoint == FIXED_ENDPOINT
        ));

        // Only the outer timeout ends these calls.
        let hour = Duration::from_secs(3600);
        assert!(tokio::time::timeout(hour, pool.get_conn()).await.is_err());
        assert!(
            tokio::time::timeout(
                hour,
                pool.execute(OssStmt::Read, std::future::pending::<Result<(), _>>())
            )
            .await
            .is_err()
        );
    }

    #[test]
    fn into_mysql_round_trips_through_anyhow() {
        let err = OssMysqlError {
            phase: OssPhase::Commit,
            stmt: OssStmt::Unknown,
            endpoint: Arc::from("db:3306"),
            generation: 9,
            kind: OssErrorKind::Mysql(server_error(1213)),
        };
        let err = anyhow::Error::from(err.into_mysql())
            .context("While executing")
            .context("outer");

        let found = err
            .chain()
            .find_map(|e| e.downcast_ref::<OssMysqlError>())
            .unwrap();
        assert_eq!(found.phase, OssPhase::Commit);
        assert_eq!(&*found.endpoint, "db:3306");
        assert_eq!(found.generation, 9);
        assert!(matches!(
            found.kind,
            OssErrorKind::Mysql(mysql_async::Error::Server(ServerError { code: 1213, .. }))
        ));
        assert!(
            err.chain()
                .any(|e| matches!(e.downcast_ref(), Some(mysql_async::Error::Other(_))))
        );
        let text = format!("{err:#}");
        // Errors in a transaction do not know its pool's generation.
        assert!(
            text.contains("OSS MySQL Commit/Unknown error on db:3306: "),
            "{text}"
        );
        assert!(text.contains("1213"), "{text}");
        let err = OssMysqlError {
            phase: OssPhase::Execute,
            stmt: OssStmt::Write,
            endpoint: Arc::from("db:3306"),
            generation: 9,
            kind: OssErrorKind::Mysql(server_error(1290)),
        };
        assert_eq!(
            err.to_string(),
            "OSS MySQL Execute/Write error on db:3306 (generation 9)"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn oss_tx_guard_tags_its_phase() {
        let _timeout = STATEMENT_TIMEOUT.lock().await;
        set_oss_transaction_statement_timeout(None);
        assert_eq!(oss_transaction_statement_timeout(), None);
        let endpoint = || Arc::from("db:3306/mononoke (connection 7)");
        assert_eq!(
            oss_tx_guard(OssPhase::InTransaction, endpoint(), async { Ok(1) })
                .await
                .unwrap(),
            1
        );

        for phase in [OssPhase::InTransaction, OssPhase::Commit] {
            let err = oss_tx_guard(phase, endpoint(), async {
                Err::<(), _>(server_error(1290))
            })
            .await
            .unwrap_err();
            let err = match err {
                mysql_async::Error::Other(err) => err,
                other => panic!("not wrapped: {other:?}"),
            };
            let err = err.downcast_ref::<OssMysqlError>().unwrap();
            assert_eq!(err.phase, phase);
            assert_eq!(err.stmt, OssStmt::Unknown);
            assert_eq!(&*err.endpoint, "db:3306/mononoke (connection 7)");
            assert!(matches!(
                err.kind,
                OssErrorKind::Mysql(mysql_async::Error::Server(ServerError { code: 1290, .. }))
            ));
        }

        set_oss_transaction_statement_timeout(Some(Duration::from_micros(10)));
        assert_eq!(
            oss_transaction_statement_timeout(),
            Some(Duration::from_millis(1))
        );
        set_oss_transaction_statement_timeout(Some(Duration::from_secs(10)));
        let start = Instant::now();
        let err = oss_tx_guard(
            OssPhase::Commit,
            endpoint(),
            std::future::pending::<Result<(), _>>(),
        )
        .await
        .unwrap_err();
        assert_ten_seconds(start.elapsed());
        let err = match err {
            mysql_async::Error::Other(err) => err,
            other => panic!("not wrapped: {other:?}"),
        };
        let err = err.downcast_ref::<OssMysqlError>().unwrap();
        assert_eq!(err.phase, OssPhase::Commit);
        assert!(matches!(err.kind, OssErrorKind::Timeout(t) if t == Duration::from_secs(10)));

        set_oss_transaction_statement_timeout(None);
        assert_eq!(oss_transaction_statement_timeout(), None);
    }

    #[tokio::test]
    async fn transaction_rollback_is_bounded() {
        let _timeout = STATEMENT_TIMEOUT.lock().await;
        let server = FakeMysqld::start(|query| match query {
            "ROLLBACK" => Reply::Stall,
            _ => Reply::Ok,
        })
        .await;
        let stats = Arc::new(ConnectionStats::new("test".to_string()));
        let connection = crate::Connection::OssMysql(OssConnection::new(server.pool(1), stats));
        let transaction = connection.start_transaction().await.unwrap();

        set_oss_transaction_statement_timeout(Some(Duration::from_millis(300)));
        let result = tokio::time::timeout(Duration::from_secs(10), transaction.rollback()).await;
        set_oss_transaction_statement_timeout(None);

        let err = result.expect("ROLLBACK was not bounded").unwrap_err();
        let err = err
            .chain()
            .find_map(|e| e.downcast_ref::<OssMysqlError>())
            .unwrap();
        assert_eq!(err.phase, OssPhase::InTransaction);
        assert!(matches!(err.kind, OssErrorKind::Timeout(t) if t == Duration::from_millis(300)));
        assert_eq!(server.queries(), vec!["START TRANSACTION", "ROLLBACK"]);
        // The error names the server the transaction was open on.
        let endpoint = format!("127.0.0.1:{}/", server.port());
        assert!(err.endpoint.starts_with(&endpoint), "{}", err.endpoint);
        assert!(err.endpoint.contains("(connection "), "{}", err.endpoint);
    }
}
