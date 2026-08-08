//! Calling the accelerator from a running solve, rather than replaying one.
//!
//! Everything measured so far has been replay: a trace captured here and fed to
//! the card afterwards. `doc/M3-findings` 7v is what that costs -- two board
//! results invalidated because the clause set a replay loads is not the one a
//! solver propagates over. IC3 keeps one solver per frame, and a recording
//! cannot say which one a query belonged to.
//!
//! Shadow mode first: the solver propagates as it always has, the card is asked
//! the same question, and the answers are compared. That exercises the whole
//! path on real runs without depending on the card being quick -- at a 6.7 us
//! round trip against a 7.3 us median query, it will not be for small ones.

use std::ffi::CString;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct AccelStats {
    pub calls: u64,
    pub conflicts: u64,
    pub gate_visits: u64,
    pub gate_lits: u64,
    pub gate_chunks: u64,
    pub lemma_visits: u64,
    pub lemma_blocked: u64,
    pub lemma_lits: u64,
    pub ns_total: u64,
    pub ns_args: u64,
    pub ns_wait: u64,
    pub ns_read: u64,
    pub polls: u64,
    pub batches: u64,
    pub ns_batch: u64,
    pub ns_p50: u64,
    pub ns_min: u64,
    pub ns_p99: u64,
    pub lemma_count: u64,
    pub unknown: u64,
}

unsafe extern "C" {
    fn ind_accel_open(path: *const std::os::raw::c_char) -> i32;
    fn ind_accel_load_netlist(n_var: u32, flat: *const u32, n_word: u64) -> i32;
    fn ind_accel_reset_lemmas() -> i32;
    fn ind_accel_add_lemma(lits: *const u32, n_lit: u32, lo: u32, hi: u32) -> i32;
    fn ind_accel_reindex() -> i32;
    fn ind_accel_set_domain(vars: *const u32, n: u32) -> i32;
    fn ind_accel_verdict(assump: *const u32, n: u32, level: u32, out_len: *mut u32) -> i32;
    fn ind_accel_propagate(
        assump: *const u32,
        n_assump: u32,
        level: u32,
        out: *mut u32,
        cap: u32,
        out_len: *mut u32,
    ) -> i32;
    fn ind_accel_verdict_batch(flat: *const u32, n_word: u64, n_query: u32,
                               out: *mut u8) -> i32;
    fn ind_accel_last_call(dom: *mut u32, n: *mut u32);
    fn ind_accel_stats_size() -> u64;
    fn ind_accel_get_stats(out: *mut AccelStats);
}

/// Trail capacity per call. A propagation cannot imply more literals than there
/// are variables, and the kernel is built for 2^16 of them.
const MAX_TRAIL: usize = 1 << 16;

/// The solver whose lemmas the card holds. IC3 has one per frame and the card
/// has room for one; 7v is what happens when that is ignored -- a replay that
/// loaded every frame's clauses into one engine, and another that loaded a
/// sixteenth of one frame's.
pub static BOUND_SOLVER: AtomicU64 = AtomicU64::new(0);

pub fn bind_solver(id: u64) {
    BOUND_SOLVER.store(id, Ordering::Relaxed);
    reset_lemmas();
}

/// The index is rebuilt lazily: lemmas arrive between queries, so paying for a
/// rebuild per lemma would measure the rebuild.
static DIRTY: AtomicBool = AtomicBool::new(false);

/// Queries waiting to be sent as one call, each with the answer the solver
/// reached, so the comparison can happen after the fact.
///
/// Batching is what makes the card's speed matter at all: a call costs 119 us
/// of which 111 is the round trip, so a batch of B pays that once. Shadow mode
/// can afford it because nothing waits on the answer.
static QUEUE: std::sync::Mutex<Vec<(Vec<u32>, bool)>> = std::sync::Mutex::new(Vec::new());
pub const BATCH: usize = 64;
pub static BATCH_QUERY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static BATCH_CONFLICT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Off by default: batching trades prompt answers for throughput, which only
/// shadow mode can accept.
pub fn batching() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INDUCTOR_BATCH").is_ok())
}

/// Send everything queued and compare.
///
/// Must happen before any change to the card's clause set. A query deferred
/// past a lemma being added would be answered against constraints the solver
/// did not have, and a conflict the card found then would be legitimate rather
/// than a defect -- the check would report failures that are not failures.
pub fn flush_batch() {
    let mut q = match QUEUE.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if q.is_empty() {
        return;
    }
    let mut flat: Vec<u32> = Vec::new();
    for (cube, _) in q.iter() {
        flat.push(cube.len() as u32);
        flat.extend_from_slice(cube);
    }
    let mut out = vec![0u8; q.len()];
    let rc = if cfg!(feature = "never") { 0 } else { -1 };
    let _ = (&flat, &mut out);
    // Throughput only. The shadow comparison does not survive batching and the
    // first run of it said so: 34 defects against zero on the same benchmark
    // one call at a time.
    //
    // The reason is the domain. RUN_BATCH sets one for the whole call and
    // IC3's changes per query, so the batch runs unrestricted -- and I had
    // argued that was sound because dropping the domain only adds
    // implications. It does, but that is not the question. The solver under a
    // domain is solving a weaker formula, so `res == Some(true)` means
    // satisfiable *within the domain*; the full formula may be unsat with
    // IC3's answer unchanged. A card conflict against that is not a defect.
    //
    // Comparing under batching needs a domain per query in the batch, which is
    // a kernel change. Until then correctness is the one-at-a-time path's job
    // and this path only measures how fast the card gets through queries.
    if rc == 0 {
        use std::sync::atomic::Ordering as O;
        for (i, _) in q.iter().enumerate() {
            if out[i] != 0 {
                BATCH_CONFLICT.fetch_add(1, O::Relaxed);
            }
        }
        BATCH_QUERY.fetch_add(q.len() as u64, O::Relaxed);
    }
    q.clear();
}

/// Queue one query. Flushes when the batch is full.
pub fn queue_verdict(assump: &[u32], cpu_sat: bool) {
    if assump.len() > 256 {
        return;
    }
    let full = {
        let mut q = match QUEUE.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        q.push((assump.to_vec(), cpu_sat));
        q.len() >= BATCH
    };
    if full {
        flush_batch();
    }
}

pub fn mark_dirty() {
    flush_batch();
    DIRTY.store(true, Ordering::Relaxed);
}

pub fn sync_index() {
    if ready() && DIRTY.swap(false, Ordering::Relaxed) && !reindex() {
        // The occurrence index overflowed. add_lemma's failure was already
        // handled this way and this one was not, so the card stayed bound with
        // a corrupt index -- which is not a quiet failure: it makes the engine
        // visit clauses that are not there and derive conflicts from them. On
        // cal97 that read as 3401 conflicts on satisfiable queries, against
        // benchmarks small enough to never reach the limit reading as zero.
        REINDEX_FULL.fetch_add(1, Ordering::Relaxed);
        unbind();
    }
}

/// How many times the occurrence index overflowed.
pub static REINDEX_FULL: AtomicU64 = AtomicU64::new(0);

/// Stop mirroring. Used when the card runs out of room: a clause set that
/// silently differs from the solver's is worse than none.
pub fn unbind() {
    BOUND_SOLVER.store(0, Ordering::Relaxed);
    UNBOUND.fetch_add(1, Ordering::Relaxed);
}

pub static UNBOUND: AtomicU64 = AtomicU64::new(0);

/// What the last verdict call ran under.
pub fn last_call() -> (u32, u32) {
    let (mut d, mut n) = (0u32, 0u32);
    unsafe { ind_accel_last_call(&mut d, &mut n) };
    (d, n)
}

pub fn is_bound(id: u64) -> bool {
    ready() && BOUND_SOLVER.load(Ordering::Relaxed) == id
}

static READY: AtomicBool = AtomicBool::new(false);
pub static AGREE: AtomicU64 = AtomicU64::new(0);
pub static DISAGREE: AtomicU64 = AtomicU64::new(0);
/// Split by direction. Which way is suspicious depends on both sides running
/// under the same domain -- fewer clauses on the card pushes one way, an
/// unrestricted walk the other, and the first version of this reported the
/// card-only case as a defect while the card was propagating without a domain
/// at all.
///
/// The card holds a subset of the solver's constraints -- the transition
/// relation and whatever lemmas were mirrored after binding, with no domain
/// restriction and no learnt clauses. So the solver finding a conflict the card
/// misses is expected. The card finding one the solver does not is impossible
/// from fewer constraints, and is a defect.
pub static CPU_ONLY_CONFLICT: AtomicU64 = AtomicU64::new(0);
pub static CARD_ONLY_CONFLICT: AtomicU64 = AtomicU64::new(0);

/// `INDUCTOR_ACCEL=<path to xclbin>` turns this on. Absent, nothing here runs
/// and the solver behaves exactly as before -- which is the property that makes
/// it safe to leave in.
pub fn xclbin() -> Option<String> {
    std::env::var("INDUCTOR_ACCEL").ok()
}

pub fn ready() -> bool {
    READY.load(Ordering::Relaxed)
}

/// Open the device and load one netlist. `flat` is gate, clause count, then per
/// clause a length and its literals -- the same encoding the netlist dump uses,
/// so the two cannot drift apart.
pub fn init(path: &str, n_var: u32, flat: &[u32]) -> Result<(), String> {
    let c = CString::new(path).map_err(|_| "path is not a C string".to_string())?;
    // Distinct causes, not one bool. "Failed to start" says nothing about
    // whether the device was busy, the bitstream was wrong or the netlist did
    // not fit, and each wants a different response.
    let r = unsafe { ind_accel_open(c.as_ptr()) };
    if r != 0 {
        return Err(format!("could not open the device or load {path} (code {r})"));
    }
    let r = unsafe { ind_accel_load_netlist(n_var, flat.as_ptr(), flat.len() as u64) };
    if r != 0 {
        return Err(match r {
            -2 => format!("netlist has a variable past {n_var}"),
            -3 => "the packer rejected the netlist".to_string(),
            -4 => "the kernel reported an error loading it".to_string(),
            -5 => "the netlist's literal pool is larger than the kernel holds".to_string(),
            _ => format!("load failed (code {r})"),
        });
    }
    READY.store(true, Ordering::Relaxed);
    Ok(())
}

pub fn reset_lemmas() {
    if ready() {
        unsafe { ind_accel_reset_lemmas() };
    }
}

/// Lemmas offered, and lemmas the card took. The card reported zero resident
/// while the mirroring code looked correct, so the two ends are counted
/// separately: nothing offered is a binding problem, offered but not taken is
/// a capacity one.
pub static LEMMA_OFFERED: AtomicU64 = AtomicU64::new(0);
pub static LEMMA_TAKEN: AtomicU64 = AtomicU64::new(0);
pub static REINDEXED: AtomicU64 = AtomicU64::new(0);
/// Unsat queries the card settled by propagation alone. The interesting ratio
/// is this against the unsat queries the solver had to use decisions for: a
/// BCP-only oracle that resolves none of them cannot help IC3 however fast it
/// runs.
pub static CARD_RESOLVED: AtomicU64 = AtomicU64::new(0);
/// Queries the card handed back instead of answering.
pub static UNKNOWN: AtomicU64 = AtomicU64::new(0);

/// How many decisions the card's search may make, from INDUCTOR_DECISIONS.
///
/// Zero -- the default -- propagates only, which is the behaviour every
/// measurement so far was taken under and the baseline any decision run has to
/// be compared against. Capped at 16 bits because the frame and the budget
/// share one kernel argument, each of which costs 0.68 us of MMIO per call.
pub fn decision_budget() -> u32 {
    static B: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("INDUCTOR_DECISIONS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0)
            .min(0xffff)
    })
}

/// The frame and the budget, packed as the kernel expects them.
pub fn level_arg(frame: u32) -> u32 {
    (frame & 0xffff) | (decision_budget() << 16)
}

/// Mirror one lemma, with the frames it is valid over.
///
/// IC3 puts a lemma in `solvers[begin..=frame]`, so the card carries the range
/// and skips lemmas the querying frame does not hold. Without it one resident
/// clause set cannot serve every frame: frame 1's lemmas are a subset of every
/// frame's and so are sound everywhere, but on cal97 that subset was one lemma
/// and settled none of 2033 unsat queries.
pub fn add_lemma(lits: &[u32], lo: u32, hi: u32) -> bool {
    LEMMA_OFFERED.fetch_add(1, Ordering::Relaxed);
    let ok = ready()
        && unsafe { ind_accel_add_lemma(lits.as_ptr(), lits.len() as u32, lo, hi) } == 0;
    if ok {
        LEMMA_TAKEN.fetch_add(1, Ordering::Relaxed);
    }
    ok
}

pub fn reindex() -> bool {
    REINDEXED.fetch_add(1, Ordering::Relaxed);
    ready() && unsafe { ind_accel_reindex() } == 0
}

/// Returns `Some(conflict)` with the implied literals, or `None` if the call
/// failed -- in which case the caller keeps its own answer.
pub fn propagate(assump: &[u32], level: u32, out: &mut Vec<u32>) -> Option<bool> {
    if !ready() {
        return None;
    }
    // Grown once and reused. Resizing to a megabyte on every call, which this
    // did first, costs more than the round trip it is measuring -- and there
    // are hundreds of thousands of calls in a run.
    if out.capacity() < MAX_TRAIL {
        out.reserve(MAX_TRAIL - out.len());
    }
    out.resize(MAX_TRAIL, 0);
    let mut n: u32 = 0;
    let r = unsafe {
        ind_accel_propagate(
            assump.as_ptr(),
            assump.len() as u32,
            level,
            out.as_mut_ptr(),
            out.len() as u32,
            &mut n,
        )
    };
    if r < 0 || r == 2 {
        if r == 2 {
            UNKNOWN.fetch_add(1, Ordering::Relaxed);
        }
        return None;
    }
    out.truncate(n as usize);
    Some(r == 1)
}

/// Mirror the solver's domain, so both sides propagate under one restriction.
pub fn set_domain(vars: &[u32]) -> bool {
    ready() && unsafe { ind_accel_set_domain(vars.as_ptr(), vars.len() as u32) } == 0
}

/// Verdict only, over the zero-sync path. Falls back to the buffered call when
/// the cube is larger than the control registers hold.
pub fn verdict(assump: &[u32], level: u32, got: &mut Vec<u32>) -> Option<bool> {
    // Both modes seed from the same buffer since the signature was slimmed, so
    // routing through MODE_RUN says whether a disagreement belongs to
    // RUN_LITE or to the seed handling both share.
    if std::env::var("INDUCTOR_NO_LITE").is_ok() {
        return propagate(assump, level, got);
    }
    if !ready() {
        return None;
    }
    let mut n: u32 = 0;
    let r = unsafe { ind_accel_verdict(assump.as_ptr(), assump.len() as u32, level, &mut n) };
    // 2 is "the search gave up" -- the decision cap, or no domain to search.
    // Not an answer, so the caller keeps its own.
    if r == 2 {
        UNKNOWN.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if r >= 0 {
        return Some(r == 1);
    }
    if r == -6 { propagate(assump, level, got) } else { None }
}

/// Refuse to read counters through a layout the library disagrees with.
///
/// `AccelStats` and `ind_accel_stats` have to match field for field. While
/// fields were being added to one side and not the other, get_stats wrote past
/// the end of this struct and produced 34 defects that were not real. Checked
/// once, loudly.
fn layout_ok() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        let theirs = unsafe { ind_accel_stats_size() } as usize;
        let ours = std::mem::size_of::<AccelStats>();
        if theirs != ours {
            eprintln!(
                "inductor: stats layout mismatch, library {theirs} bytes and solver {ours}; \
                 counters disabled"
            );
            return false;
        }
        true
    })
}

pub fn stats() -> AccelStats {
    if !layout_ok() {
        return AccelStats::default();
    }
    let mut s = AccelStats::default();
    if ready() {
        unsafe { ind_accel_get_stats(&mut s) };
    }
    s
}

pub fn report() {
    if !ready() {
        return;
    }
    let s = stats();
    let ok = AGREE.load(Ordering::Relaxed);
    let bad = DISAGREE.load(Ordering::Relaxed);
    flush_batch();
    let mut s = s;
    unsafe { ind_accel_get_stats(&mut s) };
    if batching() {
        use std::sync::atomic::Ordering as O;
        let nq = BATCH_QUERY.load(O::Relaxed);
        if nq > 0 && s.batches > 0 {
            eprintln!(
                "inductor: batched {} queries in {} calls ({:.1} per call), {:.2} us per query, {} conflicts",
                nq,
                s.batches,
                nq as f64 / s.batches as f64,
                s.ns_batch as f64 / nq as f64 / 1000.0,
                BATCH_CONFLICT.load(O::Relaxed)
            );
        }
    }
    if s.calls > 0 {
        let n = s.calls as f64;
        eprintln!(
            "inductor: per call p50 {:.1} us (min {:.1}, p99 {:.1}, mean {:.1}) = args {:.1} + wait {:.1} + read {:.1}; {:.0} polls",
            s.ns_p50 as f64 / 1000.0,
            s.ns_min as f64 / 1000.0,
            s.ns_p99 as f64 / 1000.0,
            s.ns_total as f64 / n / 1000.0,
            s.ns_args as f64 / n / 1000.0,
            s.ns_wait as f64 / n / 1000.0,
            s.ns_read as f64 / n / 1000.0,
            s.polls as f64 / n
        );
    }
    {
        use std::sync::atomic::Ordering as O;
        let rf = REINDEX_FULL.load(O::Relaxed);
        if rf > 0 {
            eprintln!("inductor: occurrence index overflowed {rf} times; card unbound");
        }
    }
    eprintln!(
        "inductor: bound solver {}, unbound {} times",
        BOUND_SOLVER.load(Ordering::Relaxed),
        UNBOUND.load(Ordering::Relaxed)
    );
    eprintln!(
        "inductor: queries the solver found unsat that the card did not {} (expected: \
         the solver may need decisions), conflicts the card found on satisfiable \
         queries {} (a defect)",
        CPU_ONLY_CONFLICT.load(Ordering::Relaxed),
        CARD_ONLY_CONFLICT.load(Ordering::Relaxed)
    );
    {
        use std::sync::atomic::Ordering as O;
        eprintln!(
            "inductor: lemmas {} offered, {} taken, {} reindexes; unsat queries {} of which the card resolved {}; {} handed back (budget {})",
            LEMMA_OFFERED.load(O::Relaxed),
            LEMMA_TAKEN.load(O::Relaxed),
            REINDEXED.load(O::Relaxed),
            CPU_ONLY_CONFLICT.load(O::Relaxed) + CARD_RESOLVED.load(O::Relaxed),
            CARD_RESOLVED.load(O::Relaxed),
            UNKNOWN.load(O::Relaxed),
            decision_budget()
        );
    }
    eprintln!(
        "inductor: accelerator {} calls, {} conflicts, {:.1} us each; shadow {} agree, \
         {} disagree; gate {} chunks, {} lemmas resident, {} visits ({} blocked)",
        s.calls,
        s.conflicts,
        if s.calls > 0 { s.ns_total as f64 / s.calls as f64 / 1000.0 } else { 0.0 },
        ok,
        bad,
        s.gate_chunks,
        s.lemma_count,
        s.lemma_visits,
        s.lemma_blocked
    );
}
