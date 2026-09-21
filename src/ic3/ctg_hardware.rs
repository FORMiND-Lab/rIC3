//! Opt-in persistent hardware CTG transport. Structural validation only: no SAT
//! double-check and no native-simulation receipt. Any ambiguity disables the
//! connection permanently; the external owner must retain BOs until reset.
use super::{
    IC3,
    ctg_native::{self, NativeLemma},
};
use anyhow::{Context, Result, bail, ensure};
use logicrs::LitVec;
use serde_json::{Value, json};
use std::{
    io,
    os::{fd::AsRawFd, unix::net::UnixStream},
    path::Path,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

pub(super) struct HardwareResult {
    pub(super) complete: bool,
    pub(super) cube: LitVec,
    pub(super) journal: Vec<NativeLemma>,
}
#[derive(Clone, Copy, Default)]
struct Authority {
    words: [u32; 8],
}
struct Client {
    socket: UnixStream,
    id: u32,
    authority: Authority,
    model: Value,
    disabled: bool,
}

// poll and MSG_NOSIGNAL/MSG_DONTWAIT keep even partial operations within one
// absolute deadline and avoid terminating rIC3 on a disconnected owner.
fn transfer(socket: &UnixStream, bytes: &mut [u8], write: bool, deadline: Instant) -> Result<()> {
    let mut at = 0;
    while at < bytes.len() {
        let left = deadline
            .checked_duration_since(Instant::now())
            .context("CTG socket deadline")?;
        let mut fd = libc::pollfd {
            fd: socket.as_raw_fd(),
            events: if write { libc::POLLOUT } else { libc::POLLIN },
            revents: 0,
        };
        let ms = left.as_millis().saturating_add(1).min(i32::MAX as u128) as i32;
        let ready = unsafe { libc::poll(&mut fd, 1, ms) };
        if ready < 0 {
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(e.into());
        }
        ensure!(ready > 0, "CTG socket deadline");
        let n = unsafe {
            if write {
                libc::send(
                    fd.fd,
                    bytes[at..].as_ptr().cast(),
                    bytes.len() - at,
                    libc::MSG_NOSIGNAL | libc::MSG_DONTWAIT,
                )
            } else {
                libc::recv(
                    fd.fd,
                    bytes[at..].as_mut_ptr().cast(),
                    bytes.len() - at,
                    libc::MSG_DONTWAIT,
                )
            }
        };
        if n < 0 {
            let e = io::Error::last_os_error();
            if matches!(
                e.kind(),
                io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock
            ) {
                continue;
            }
            return Err(e.into());
        }
        ensure!(n > 0, "CTG socket disconnected");
        at += n as usize;
        ensure!(Instant::now() <= deadline, "CTG socket deadline");
    }
    Ok(())
}
fn peer(socket: &UnixStream) -> Result<()> {
    let mut cred = libc::ucred {
        pid: 0,
        uid: 0,
        gid: 0,
    };
    let mut size = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    ensure!(
        unsafe {
            libc::getsockopt(
                socket.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                (&mut cred as *mut libc::ucred).cast(),
                &mut size,
            )
        } == 0,
        "CTG peer credentials: {}",
        io::Error::last_os_error()
    );
    ensure!(
        size as usize == std::mem::size_of::<libc::ucred>()
            && cred.uid == unsafe { libc::geteuid() },
        "CTG peer UID"
    );
    Ok(())
}
fn encode(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}
fn decode(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn result(words: &[u32], input: &Value, old: Authority) -> Result<(HardwareResult, Authority)> {
    ensure!((16..=11032).contains(&words.len()), "CTG reply extent");
    let w = words;
    ensure!(
        w[0] <= 1
            && w[1] > 0
            && w[1] <= 128
            && w[2] <= 10000
            && w[5] <= 128
            && w[14] == 0
            && w[15] == 0,
        "CTG result header"
    );
    let complete = w[0] == 1;
    ensure!(w[3] == if complete { w[1] } else { 0 }, "CTG final query");
    let mut authority = Authority::default();
    authority.words.copy_from_slice(&w[6..14]);
    let a = authority.words;
    let p = old.words;
    ensure!(
        a[0] != 0
            && a[1] == p[1].checked_add(1).context("CTG task wrap")?
            && a[7] == p[7].checked_add(1).context("CTG serial wrap")?
            && a[6] <= 8192,
        "CTG identity"
    );
    if p[0] != 0 {
        ensure!(a[0] == p[0], "CTG lease changed");
    }
    for i in [2, 4, 5] {
        ensure!(a[i] > p[i], "CTG authority not fresh");
    }
    ensure!(a[3] >= p[3], "CTG cookie regressed");
    ensure!(
        u64::from(a[3]) >= u64::from(p[3]) + u64::from(w[5])
            && u64::from(a[4]) >= u64::from(p[4]) + 1 + u64::from(w[5]),
        "CTG publication authority census"
    );
    ensure!(
        u64::from(a[6]) >= u64::from(p[6]) + u64::from(w[5]),
        "CTG physical census"
    );
    let nv = input["n_var"].as_u64().context("CTG nvar")? as usize;
    let init = ctg_native::words(&input["init_value_by_current"])?;
    ensure!(
        nv > 0 && nv <= 2048 && init.len() == nv && init.iter().all(|v| *v <= 2),
        "CTG snapshot init"
    );
    let maxframe = input["max_frame"].as_u64().context("CTG maxframe")?;
    let mut at = 16usize;
    let take = |at: &mut usize, n: u32| -> Result<Vec<u32>> {
        ensure!(
            n > 0 && n as usize <= nv && n as usize <= w.len() - *at,
            "CTG cube extent"
        );
        let cube = w[*at..*at + n as usize].to_vec();
        *at += n as usize;
        ctg_native::state_cube(&json!(cube), input)
    };
    let cube = take(&mut at, w[4])?;
    ensure!(
        ctg_native::ordered_subset(&cube, &ctg_native::words(&input["loop_initial"]["cube"])?),
        "CTG root subset"
    );
    let mut journal = Vec::new();
    let mut proof = 0;
    let mut literals = 0usize;
    for _ in 0..w[5] {
        ensure!(w.len() - at >= 4, "CTG journal truncation");
        let lo = w[at];
        let hi = w[at + 1];
        let q = w[at + 2];
        let count = w[at + 3];
        at += 4;
        ensure!(
            lo == 1
                && hi >= lo
                && u64::from(hi) <= maxframe
                && q > proof
                && q <= w[1]
                && (!complete || q < w[3]),
            "CTG journal proof/range"
        );
        literals += count as usize;
        ensure!(literals <= 8192, "CTG journal literals");
        let c = take(&mut at, count)?;
        journal.push(NativeLemma {
            hi: hi as usize,
            cube: ctg_native::to_litvec(&c),
        });
        proof = q;
    }
    ensure!(at == w.len(), "CTG result suffix");
    Ok((
        HardwareResult {
            complete,
            cube: ctg_native::to_litvec(&cube),
            journal,
        },
        authority,
    ))
}
impl Client {
    fn request(&mut self, input: &Value) -> Result<Option<HardwareResult>> {
        ensure!(!self.disabled, "CTG client permanently disabled");
        let payload = ctg_native::serialize(input)?;
        ensure!(payload.len() <= 2 * 1024 * 1024, "CTG payload capacity");
        // From here even a partial write may have mutated the device. Only a
        // fully validated accepted result or explicit clean refusal re-enables.
        self.disabled = true;
        self.id = self.id.checked_add(1).context("CTG request wrap")?;
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut header = encode(&[0x43545251, 1, self.id, 1, payload.len() as u32, 0]);
        transfer(&self.socket, &mut header, true, deadline)?;
        transfer(&self.socket, &mut payload.clone(), true, deadline)?;
        transfer(&self.socket, &mut header, false, deadline)?;
        let h = decode(&header);
        ensure!(
            h[0] == 0x43545250 && h[1] == 1 && h[2] == self.id && h[5] == 0,
            "CTG reply identity"
        );
        if h[3] == 1 {
            ensure!(h[4] == 0, "CTG refusal payload");
            self.disabled = false;
            return Ok(None);
        }
        if h[3] == 2 {
            ensure!(h[4] == 0, "CTG quarantine payload");
            bail!("CTG owner unavailable/quarantined")
        }
        ensure!(
            h[3] == 0 && h[4] % 4 == 0 && h[4] >= 64 && h[4] <= 11032 * 4,
            "CTG reply framing"
        );
        let mut bytes = vec![0; h[4] as usize];
        transfer(&self.socket, &mut bytes, false, deadline)?;
        let (adopted, authority) = result(&decode(&bytes), input, self.authority)?;
        self.authority = authority;
        self.disabled = false;
        Ok(Some(adopted))
    }
}
fn model(input: &Value) -> Value {
    json!([
        input["transition_only"],
        input["next_literal_by_current"],
        input["init_value_by_current"],
        input["latch_variables"],
        input["input_variables"]
    ])
}
fn connect(path: &Path, input: &Value) -> Result<Client> {
    ensure!(path.is_absolute(), "CTG socket must be absolute");
    // UNIX connect is nonblocking too: a saturated owner listen queue must not
    // stall the CPU forever. A failed attempt is never automatically retried.
    let raw = unsafe {
        libc::socket(
            libc::AF_UNIX,
            libc::SOCK_STREAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
            0,
        )
    };
    ensure!(raw >= 0, "CTG socket create");
    use std::os::fd::FromRawFd;
    let socket = unsafe { UnixStream::from_raw_fd(raw) };
    use std::os::unix::ffi::OsStrExt;
    let name = path.as_os_str().as_bytes();
    let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    ensure!(
        name.len() < address.sun_path.len() && !name.contains(&0),
        "CTG socket path extent"
    );
    address.sun_family = libc::AF_UNIX as libc::sa_family_t;
    for (to, from) in address.sun_path.iter_mut().zip(name) {
        *to = *from as libc::c_char;
    }
    ensure!(
        unsafe {
            libc::connect(
                raw,
                (&address as *const libc::sockaddr_un).cast(),
                std::mem::size_of_val(&address) as libc::socklen_t,
            )
        } == 0,
        "CTG socket connect (no retry)"
    );
    peer(&socket)?;
    Ok(Client {
        socket,
        id: 0,
        authority: Authority::default(),
        model: model(input),
        disabled: false,
    })
}
impl IC3 {
    pub(super) fn try_hardware_ctg_root(
        &self,
        frame: usize,
        cube: &LitVec,
        constraint: &[LitVec],
        level: usize,
        max: usize,
        limit: usize,
    ) -> Option<HardwareResult> {
        let path = std::env::var_os("INDUCTOR_CTG_HARDWARE_SOCKET")?;
        if level != 1
            || max != 3
            || limit != 1
            || !self.cfg.ctg
            || self.cfg.dynamic
            || self.cfg.mab
            || self.cfg.ctp
        {
            return None;
        }
        // One persistent model per process; failed initialization is cached.
        static CLIENT: OnceLock<Mutex<Option<Client>>> = OnceLock::new();
        if let Some(lock) = CLIENT.get() {
            let guard = lock.lock().ok()?;
            if guard.as_ref().is_none_or(|c| c.disabled) {
                return None;
            }
        }
        if std::env::vars_os().any(|(k, _)| {
            let k = k.to_string_lossy();
            k == "INDUCTOR_CTG_NATIVE_EXECUTABLE"
                || k == "INDUCTOR_ACCEL"
                || k == "INDUCTOR_MIC"
                || k == "INDUCTOR_ACTIVE_CDCL"
                || k.starts_with("INDUCTOR_CDCL_")
        }) {
            return None;
        }
        let input = match self.native_ctg_snapshot(frame, cube, constraint, 0) {
            Ok(i) => i,
            Err(_) => return None,
        };
        let lock = CLIENT.get_or_init(|| {
            Mutex::new(match connect(Path::new(&path), &input) {
                Ok(c) => Some(c),
                Err(e) => {
                    eprintln!("hardware CTG disabled: {e:#}");
                    None
                }
            })
        });
        let mut guard = lock.lock().ok()?;
        let client = guard.as_mut()?;
        if client.disabled {
            return None;
        }
        if client.model != model(&input) {
            client.disabled = true;
            eprintln!("hardware CTG disabled: model changed");
            return None;
        }
        let started = Instant::now();
        match client.request(&input) {
            Ok(out) => {
                eprintln!(
                    "hardware CTG request={} accepted={} complete={} journal={} elapsed_ns={} cpu_validation_solve=false",
                    client.id,
                    out.is_some(),
                    out.as_ref().is_some_and(|r| r.complete),
                    out.as_ref().map_or(0, |r| r.journal.len()),
                    started.elapsed().as_nanos()
                );
                out
            }
            Err(e) => {
                client.disabled = true;
                eprintln!("hardware CTG permanently disabled: {e:#}");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    fn input() -> Value {
        json!({"n_var":3,"max_frame":2,"frame":1,
        "loop_initial":{"cube":[0,2]},"next_literal_by_current":[0,2,4],
        "init_value_by_current":[0,0,2],"latch_variables":[0,1],"input_variables":[2],
        "constraints_localabs_filtered":[],"transition_only":[[0,4]],
        "frame_solvers":[{"lemmas":[]},{"lemmas":[]},{"lemmas":[]}]})
    }
    fn accepted() -> Vec<u32> {
        vec![
            1, 5, 8, 5, 1, 1, 37, 1, 10, 2, 3, 5, 2, 1, 0, 0, 0, 1, 2, 3, 1, 2,
        ]
    }
    fn client(socket: UnixStream) -> Client {
        Client {
            socket,
            id: 0,
            authority: Authority::default(),
            model: model(&input()),
            disabled: false,
        }
    }
    // Explicitly a socketpair simulation, not hardware qualification.
    fn request_read(socket: &mut UnixStream, id: u32) {
        socket
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        socket
            .set_write_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let mut h = [0; 24];
        socket.read_exact(&mut h).unwrap();
        let h = decode(&h);
        assert_eq!(h[0], 0x43545251);
        assert_eq!(h[2], id);
        assert_eq!(h[3], 1);
        let mut payload = vec![0; h[4] as usize];
        socket.read_exact(&mut payload).unwrap();
        assert_eq!(payload, ctg_native::serialize(&input()).unwrap());
    }
    #[test]
    fn hardware_wire_simulation_accept_refuse_and_fallback() {
        let (socket, mut server) = UnixStream::pair().unwrap();
        peer(&socket).unwrap();
        let mut c = client(socket);
        let fake = std::thread::spawn(move || {
            request_read(&mut server, 1);
            server
                .write_all(&encode(&[0x43545250, 1, 1, 1, 0, 0]))
                .unwrap();
            request_read(&mut server, 2);
            let words = accepted();
            let body = encode(&words);
            for b in encode(&[0x43545250, 1, 2, 0, body.len() as u32, 0])
                .into_iter()
                .chain(body)
            {
                server.write_all(&[b]).unwrap();
            }
            request_read(&mut server, 3);
            let mut words = accepted();
            words[0] = 0;
            words[3] = 0;
            words[7] = 2;
            words[8] = 20;
            words[9] = 3;
            words[10] = 5;
            words[11] = 10;
            words[12] = 3;
            words[13] = 2;
            server
                .write_all(&encode(&[0x43545250, 1, 3, 0, (words.len() * 4) as u32, 0]))
                .unwrap();
            server.write_all(&encode(&words)).unwrap();
            request_read(&mut server, 4);
            let mut words = accepted();
            words.truncate(17);
            words[5] = 0;
            words[7] = 3;
            words[8] = 30;
            words[9] = 3;
            words[10] = 6;
            words[11] = 15;
            words[12] = 3;
            words[13] = 3;
            server
                .write_all(&encode(&[0x43545250, 1, 4, 0, (words.len() * 4) as u32, 0]))
                .unwrap();
            server.write_all(&encode(&words)).unwrap();
        });
        assert!(c.request(&input()).unwrap().is_none());
        assert!(!c.disabled && c.authority.words[1] == 0);
        let r = c.request(&input()).unwrap().unwrap();
        assert!(r.complete && r.journal.len() == 1);
        let r = c.request(&input()).unwrap().unwrap();
        assert!(!r.complete && r.journal.len() == 1);
        assert_eq!(c.authority.words[7], 2);
        let r = c.request(&input()).unwrap().unwrap();
        assert!(r.complete && r.journal.is_empty());
        assert_eq!(c.authority.words[3], 3);
        assert_eq!(c.authority.words[7], 3);
        fake.join().unwrap();
    }
    #[test]
    fn hardware_structural_rejection_is_atomic() {
        let original = accepted();
        assert!(result(&original, &input(), Authority::default()).is_ok());
        for len in 0..original.len() {
            assert!(result(&original[..len], &input(), Authority::default()).is_err());
        }
        for (i, bad) in [
            (0, 2),
            (1, 0),
            (1, 129),
            (2, 10001),
            (3, 4),
            (4, 4),
            (5, 129),
            (6, 0),
            (7, 2),
            (8, 0),
            (10, 0),
            (11, 0),
            (12, 8193),
            (13, 2),
            (14, 1),
            (15, 1),
            (16, 4),
            (17, 0),
            (18, 3),
            (19, 5),
            (20, 0),
            (21, 4),
        ] {
            let mut w = original.clone();
            w[i] = bad;
            assert!(
                result(&w, &input(), Authority::default()).is_err(),
                "word {i}"
            );
        }
        let (_, authority) = result(&original, &input(), Authority::default()).unwrap();
        assert!(result(&original, &input(), authority).is_err());
        let mut suffix = original.clone();
        suffix.push(0);
        assert!(result(&suffix, &input(), Authority::default()).is_err());
    }
    #[test]
    fn hardware_socket_ambiguity_never_retries() {
        for mode in 0..5 {
            let (socket, mut server) = UnixStream::pair().unwrap();
            let mut c = client(socket);
            let fake = std::thread::spawn(move || {
                request_read(&mut server, 1);
                let h = match mode {
                    0 => vec![0x43545250, 1, 1, 2, 0, 0],
                    1 => vec![0x43545250, 1, 2, 1, 0, 0],
                    2 => vec![0x43545250, 1, 1, 1, 4, 0],
                    3 => vec![0x43545250, 1, 1, 0, 64, 0],
                    _ => vec![0x43545250],
                };
                server.write_all(&encode(&h)).unwrap();
            });
            assert!(c.request(&input()).is_err() && c.disabled);
            let id = c.id;
            assert!(c.request(&input()).is_err() && c.id == id);
            fake.join().unwrap();
        }
    }
    #[test]
    fn hardware_absolute_deadline_simulation() {
        let (socket, _server) = UnixStream::pair().unwrap();
        let mut bytes = [0; 24];
        let start = Instant::now();
        assert!(
            transfer(
                &socket,
                &mut bytes,
                false,
                start + Duration::from_millis(20)
            )
            .is_err()
        );
        assert!(start.elapsed() < Duration::from_secs(1));
    }
}
