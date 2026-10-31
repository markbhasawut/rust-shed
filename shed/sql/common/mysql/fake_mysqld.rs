/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is dual-licensed under either the MIT license found in the
 * LICENSE-MIT file in the root directory of this source tree or the Apache
 * License, Version 2.0 found in the LICENSE-APACHE file in the root directory
 * of this source tree. You may select, at your option, one of the
 * above-listed licenses.
 */

//! A fake MySQL server for tests. It speaks just enough of the protocol for
//! mysql_async to connect and run text queries, and answers each query as
//! the test says.

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Mutex;

use mysql_async::OptsBuilder;
use mysql_async::Pool;
use mysql_async::PoolConstraints;
use mysql_async::PoolOpts;
use mysql_async::consts::CapabilityFlags;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::net::TcpStream;
use tokio::task::JoinHandle;
use tokio::task::JoinSet;

const COM_QUIT: u8 = 0x01;
const COM_QUERY: u8 = 0x03;

const SERVER_STATUS_AUTOCOMMIT: u16 = 0x0002;
const UTF8MB4_GENERAL_CI: u8 = 45;
const MYSQL_TYPE_VAR_STRING: u8 = 0xfd;

/// How the server answers a query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reply {
    /// Success, with one row affected.
    Ok,
    /// The start of a result set of one column: its header and one row,
    /// and then nothing more, as if the server stalled while sending it.
    PartialRows,
    /// No answer at all.
    Stall,
}

type Answer = dyn Fn(&str) -> Reply + Send + Sync;

/// A running fake server. It stops when dropped.
pub(crate) struct FakeMysqld {
    addr: SocketAddr,
    queries: Arc<Mutex<Vec<String>>>,
    task: JoinHandle<()>,
}

impl FakeMysqld {
    /// Starts a server on a free local port that accepts any user and
    /// password, and answers every query with what `answer` returns for it.
    /// Other commands succeed. After a reply other than [`Reply::Ok`], the
    /// connection answers nothing more.
    pub(crate) async fn start(answer: impl Fn(&str) -> Reply + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let queries = Arc::new(Mutex::new(Vec::new()));
        let answer: Arc<Answer> = Arc::new(answer);
        let task = tokio::spawn({
            let queries = queries.clone();
            async move {
                // Dropped with the accept loop, which ends every connection.
                let mut connections = JoinSet::new();
                let mut id = 0;
                while let Ok((stream, _)) = listener.accept().await {
                    id += 1;
                    connections.spawn(serve(stream, id, answer.clone(), queries.clone()));
                }
            }
        });
        Self {
            addr,
            queries,
            task,
        }
    }

    /// The port the server listens on.
    pub(crate) fn port(&self) -> u16 {
        self.addr.port()
    }

    /// The queries received so far, in order.
    pub(crate) fn queries(&self) -> Vec<String> {
        self.queries.lock().unwrap().clone()
    }

    /// A new pool of at most `max` connections to this server.
    pub(crate) fn pool(&self, max: usize) -> Pool {
        let pool_opts = PoolOpts::default()
            .with_constraints(PoolConstraints::new(0, max).unwrap())
            .with_reset_connection(false);
        Pool::new(
            OptsBuilder::default()
                .ip_or_hostname(self.addr.ip().to_string())
                .tcp_port(self.addr.port())
                .user(Some("test"))
                .pass(Some("secret"))
                .prefer_socket(false)
                // Given, so that connecting runs no query.
                .max_allowed_packet(Some(16 << 20))
                .wait_timeout(Some(28_800))
                .pool_opts(pool_opts),
        )
    }
}

impl Drop for FakeMysqld {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn serve(
    mut stream: TcpStream,
    connection_id: u32,
    answer: Arc<Answer>,
    queries: Arc<Mutex<Vec<String>>>,
) -> io::Result<()> {
    write_packet(&mut stream, 0, &handshake(connection_id)).await?;
    // Any credentials will do.
    let (seq, _response) = read_packet(&mut stream).await?;
    write_packet(&mut stream, seq.wrapping_add(1), &ok_packet()).await?;
    loop {
        let (mut seq, payload) = read_packet(&mut stream).await?;
        let reply = match payload.split_first() {
            Some((&COM_QUERY, query)) => {
                let query = String::from_utf8_lossy(query).into_owned();
                let reply = answer(&query);
                queries.lock().unwrap().push(query);
                reply
            }
            Some((&COM_QUIT, _)) => return Ok(()),
            _ => Reply::Ok,
        };
        let packets = match reply {
            Reply::Ok => vec![ok_packet()],
            Reply::PartialRows => {
                vec![vec![1], column_definition("x"), eof_packet(), vec![1, b'1']]
            }
            Reply::Stall => vec![],
        };
        for packet in packets {
            seq = seq.wrapping_add(1);
            write_packet(&mut stream, seq, &packet).await?;
        }
        if reply != Reply::Ok {
            // Wait, answering nothing, until the client goes away.
            let _ = stream.read_to_end(&mut Vec::new()).await;
            return Ok(());
        }
    }
}

/// A protocol 10 handshake that asks for mysql_native_password.
fn handshake(connection_id: u32) -> Vec<u8> {
    let capabilities = (CapabilityFlags::CLIENT_LONG_PASSWORD
        | CapabilityFlags::CLIENT_LONG_FLAG
        | CapabilityFlags::CLIENT_CONNECT_WITH_DB
        | CapabilityFlags::CLIENT_PROTOCOL_41
        | CapabilityFlags::CLIENT_TRANSACTIONS
        | CapabilityFlags::CLIENT_SECURE_CONNECTION
        | CapabilityFlags::CLIENT_MULTI_STATEMENTS
        | CapabilityFlags::CLIENT_MULTI_RESULTS
        | CapabilityFlags::CLIENT_PLUGIN_AUTH)
        .bits();
    let mut packet = vec![10];
    packet.extend_from_slice(b"8.0.36-fake\0");
    packet.extend_from_slice(&connection_id.to_le_bytes());
    // The scramble: 8 bytes here, 12 and a NUL below.
    packet.extend_from_slice(b"abcdefgh\0");
    packet.extend_from_slice(&(capabilities as u16).to_le_bytes());
    packet.push(UTF8MB4_GENERAL_CI);
    packet.extend_from_slice(&SERVER_STATUS_AUTOCOMMIT.to_le_bytes());
    packet.extend_from_slice(&((capabilities >> 16) as u16).to_le_bytes());
    packet.push(21);
    packet.extend_from_slice(&[0; 10]);
    packet.extend_from_slice(b"ijklmnopqrst\0");
    packet.extend_from_slice(b"mysql_native_password\0");
    packet
}

fn ok_packet() -> Vec<u8> {
    // Header, affected rows, last insert id, status flags, warnings.
    let mut packet = vec![0x00, 1, 0];
    packet.extend_from_slice(&SERVER_STATUS_AUTOCOMMIT.to_le_bytes());
    packet.extend_from_slice(&[0, 0]);
    packet
}

/// The definition of a text column called `name`.
fn column_definition(name: &str) -> Vec<u8> {
    let mut packet = Vec::new();
    // Catalog, schema, table, original table, name and original name.
    for field in ["def", "", "", "", name, ""] {
        packet.push(u8::try_from(field.len()).expect("short names only"));
        packet.extend_from_slice(field.as_bytes());
    }
    // Length of the fixed fields, character set, column length, type, then
    // flags, decimals and filler.
    packet.push(0x0c);
    packet.extend_from_slice(&u16::from(UTF8MB4_GENERAL_CI).to_le_bytes());
    packet.extend_from_slice(&255u32.to_le_bytes());
    packet.push(MYSQL_TYPE_VAR_STRING);
    packet.extend_from_slice(&[0; 5]);
    packet
}

fn eof_packet() -> Vec<u8> {
    // Header, warnings, status flags.
    let mut packet = vec![0xfe, 0, 0];
    packet.extend_from_slice(&SERVER_STATUS_AUTOCOMMIT.to_le_bytes());
    packet
}

async fn read_packet(stream: &mut TcpStream) -> io::Result<(u8, Vec<u8>)> {
    let mut header = [0; 4];
    stream.read_exact(&mut header).await?;
    let len = u32::from_le_bytes([header[0], header[1], header[2], 0]);
    let mut payload = vec![0; len as usize];
    stream.read_exact(&mut payload).await?;
    Ok((header[3], payload))
}

async fn write_packet(stream: &mut TcpStream, seq: u8, payload: &[u8]) -> io::Result<()> {
    let len = u32::try_from(payload.len()).expect("small packets only");
    let mut packet = len.to_le_bytes()[..3].to_vec();
    packet.push(seq);
    packet.extend_from_slice(payload);
    stream.write_all(&packet).await
}
