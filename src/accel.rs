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

pub mod cdcl;
pub mod cdcl_host;

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
    pub cores: u64,
    pub ddr_overflow: u64,
    pub con_full_rebuilds: u64,
    pub cores_unminimised: u64,
    pub lem_full_rebuilds: u64,
    pub ns_down: u64,
    pub n_down: u64,
    pub ns_mic: u64,
    pub n_mic: u64,
    pub mic_tried: u64,
    pub mic_in: u64,
    pub mic_out: u64,
    pub ns_constraint: u64,
    pub n_constraint: u64,
    pub ns_core_probe: u64,
    pub ns_core_min: u64,
    pub ns_domain: u64,
    pub n_domain: u64,
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
    fn ind_accel_last_call(dom: *mut u32, n: *mut u32);
    fn ind_accel_down(con_flat: *const u32, n_con_word: u32, assump: *const u32,
                      n_assump: u32, level: u32, out_core: *mut u32, cap: u32,
                      out_len: *mut u32) -> i32;
    fn ind_accel_mic(extra: *const u32, n_extra_word: u32, pairs: *const u32, n_lit: u32,
                     level: u32, out_cube: *mut u32, cap: u32, out_len: *mut u32) -> i32;
    fn ind_accel_stats_size() -> u64;
    fn ind_accel_set_constraint(flat: *const u32, n_word: u32) -> i32;
    fn ind_accel_core(assump: *const u32, n: u32, level: u32, out: *mut u32, cap: u32,
                      out_len: *mut u32) -> i32;
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
    // The legacy batch ABI cannot carry a per-query decision domain, so it is
    // deliberately disabled rather than hidden behind an undeclared cfg.
    let rc: i32 = -1;
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
            -6 => format!(
                "the watched kernel holds at most 16384 variables (netlist has {n_var}); CPU fallback required"
            ),
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

/// Cube lengths seen at generalization, bucketed. The batch a speculative
/// round could issue is one query per literal, so this is the batch-size
/// distribution the design would actually see.
pub static MIC_N: AtomicU64 = AtomicU64::new(0);
pub static MIC_LITS: AtomicU64 = AtomicU64::new(0);
pub static MIC_GE8: AtomicU64 = AtomicU64::new(0);
pub static MIC_GE32: AtomicU64 = AtomicU64::new(0);
pub static MIC_MAX: AtomicU64 = AtomicU64::new(0);

pub fn note_mic(len: usize) {
    MIC_N.fetch_add(1, Ordering::Relaxed);
    MIC_LITS.fetch_add(len as u64, Ordering::Relaxed);
    if len >= 8 {
        MIC_GE8.fetch_add(1, Ordering::Relaxed);
    }
    if len >= 32 {
        MIC_GE32.fetch_add(1, Ordering::Relaxed);
    }
    MIC_MAX.fetch_max(len as u64, Ordering::Relaxed);
}

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

/// The unsat core for a query, or None if the card did not derive one.
///
/// `down()` turns the returned literals into a candidate generalized lemma, so
/// a verdict bit was never usable however fast it came back. IC3 does not trust
/// this result directly: the exact frame solver validates the candidate under
/// the same query constraints before adoption.
/// The clauses this query carries, replacing the last query's.
///
/// `down()` asks with `!cube` under `strengthen`; the card holding a subset of
/// the solver's clauses is sound, but a subset missing the one clause that
/// makes the query unsatisfiable produces no core at all.
pub fn set_constraint(flat: &[u32]) -> bool {
    if !ready() {
        return false;
    }
    let ok = unsafe { ind_accel_set_constraint(flat.as_ptr(), flat.len() as u32) } == 0;
    if flat.is_empty() {
        return ok;
    }
    CON_SET.fetch_add(1, Ordering::Relaxed);
    if !ok {
        CON_FAIL.fetch_add(1, Ordering::Relaxed);
    }
    ok
}

pub static CON_SET: AtomicU64 = AtomicU64::new(0);
pub static CON_FAIL: AtomicU64 = AtomicU64::new(0);

/// One down() iteration in one round trip: the constraint goes in, the query
/// is answered, the core comes back, and the constraint comes out again --
/// without the host issuing four calls to walk that sequence.
///
/// `None` means this bitstream predates the mode and the caller should fall
/// back to set_constraint/core/set_constraint. `Some(0)` means no conflict.
/// mic's drop loop, run on the card.
///
/// `pairs` is the cube as [current, next] literal pairs. What comes back is a
/// sub-cube that still blocks -- weaker than what the solver would find,
/// because the card has no model to shrink with on the satisfiable branch, but
/// never unsound: every literal it dropped was dropped because propagation
/// still derived a conflict without it.
///
/// `None` if the bitstream predates the mode or the call failed.
pub fn mic(extra: &[u32], pairs: &[u32], level: u32, out: &mut Vec<u32>) -> Option<usize> {
    if !ready() || pairs.is_empty() {
        return None;
    }
    let n_lit = (pairs.len() / 2) as u32;
    out.resize(n_lit as usize, 0);
    let mut n: u32 = 0;
    let rc = unsafe {
        ind_accel_mic(extra.as_ptr(), extra.len() as u32, pairs.as_ptr(), n_lit, level,
                      out.as_mut_ptr(), n_lit, &mut n)
    };
    if rc < 0 {
        return None;
    }
    out.truncate(n as usize);
    Some(n as usize)
}

/// Whether the card's drop loop is available in this bitstream.
pub fn have_mic() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        if !ready() {
            return false;
        }
        let pairs = [0u32, 0u32];
        let mut o: Vec<u32> = vec![0; 1];
        let mut n: u32 = 0;
        let rc = unsafe {
            ind_accel_mic(std::ptr::null(), 0, pairs.as_ptr(), 1, 0, o.as_mut_ptr(), 1, &mut n)
        };
        rc != -2
    })
}

pub fn down(con_flat: &[u32], assump: &[u32], level: u32, out: &mut Vec<u32>) -> Option<usize> {
    if !ready() || assump.is_empty() {
        return None;
    }
    CORE_ASKED.fetch_add(1, Ordering::Relaxed);
    out.resize(assump.len(), 0);
    let mut n: u32 = 0;
    let rc = unsafe {
        ind_accel_down(
            con_flat.as_ptr(),
            con_flat.len() as u32,
            assump.as_ptr(),
            assump.len() as u32,
            level,
            out.as_mut_ptr(),
            assump.len() as u32,
            &mut n,
        )
    };
    if rc == -2 {
        return None; // no fused mode in this bitstream
    }
    if rc < 0 {
        CON_FAIL.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    out.truncate(n as usize);
    if n > 0 {
        CORE_GOT.fetch_add(1, Ordering::Relaxed);
        CORE_IN.fetch_add(assump.len() as u64, Ordering::Relaxed);
        CORE_OUT.fetch_add(n as u64, Ordering::Relaxed);
    }
    Some(n as usize)
}

/// Whether the fused path is available, so the caller can skip building the
/// separate constraint payload when it is not.
pub fn have_down() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        if !ready() {
            return false;
        }
        let a = [0u32];
        let mut o: Vec<u32> = Vec::new();
        o.resize(1, 0);
        let mut n: u32 = 0;
        // A probe with one assumption. -2 is the only answer that matters;
        // anything else means the mode exists and this query simply did or did
        // not conflict.
        let rc = unsafe {
            ind_accel_down(std::ptr::null(), 0, a.as_ptr(), 1, 0, o.as_mut_ptr(), 1, &mut n)
        };
        rc != -2
    })
}

pub fn core(assump: &[u32], level: u32, out: &mut Vec<u32>) -> Option<usize> {
    if !ready() || assump.is_empty() {
        return None;
    }
    CORE_ASKED.fetch_add(1, Ordering::Relaxed);
    out.resize(assump.len(), 0);
    let mut n: u32 = 0;
    let r = unsafe {
        ind_accel_core(assump.as_ptr(), assump.len() as u32, level, out.as_mut_ptr(),
                       out.len() as u32, &mut n)
    };
    if r <= 0 || n == 0 {
        return None;
    }
    out.truncate(n as usize);
    CORE_GOT.fetch_add(1, Ordering::Relaxed);
    CORE_IN.fetch_add(assump.len() as u64, Ordering::Relaxed);
    CORE_OUT.fetch_add(n as u64, Ordering::Relaxed);
    Some(n as usize)
}

/// Off by default. Turning it on changes what IC3 computes -- the card's core
/// replaces the solver's -- so every run with it on is a different run, and
/// the baseline has to be the run with it off.
/// Whether to ask the card a question whose answer is thrown away.
///
/// The shadow check compares every query against the card and uses none of the
/// results. It is what proved the integration correct, and it is 12.4 s of a
/// 16.5 s run: 18,535 calls at 670 us each, because with a decision budget
/// every one of them runs a full on-card search. Off unless asked for, so a
/// run that is using the card does not also pay to audit it.
pub fn shadow() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INDUCTOR_SHADOW").is_ok())
}

pub static CORE_THIN: AtomicU64 = AtomicU64::new(0);
pub static MIC_TAKEN: AtomicU64 = AtomicU64::new(0);

/// Whether to run mic's drop loop on the card. Off unless INDUCTOR_MIC is set.
///
/// It is the largest piece of IC3 the card holds that is not propagation, and
/// it is sound, but measured it does not pay on the benchmark that has the
/// volume. On Problem03_label51 it spent 169 s over 4,945 calls and 339,436
/// attempted drops to remove 9,452 literals -- 1.03x -- and the cubes it
/// handed back sent IC3 down a path needing 474,188 down() calls against
/// 227,216 without it. The run went from about 310 s to 605 s and produced no
/// cores at all.
///
/// The reason is structural, not a tuning problem. The solver's loop shrinks
/// the cube from its model on the satisfiable branch, which is where most
/// drops land; the card has no model there and can only keep the literal. So
/// the card tries every drop and wins the few that propagation alone settles.
pub fn mic_offload() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INDUCTOR_MIC").is_ok())
}

/// How many literals the card must remove before its core is worth taking.
/// One by default: a core equal to the cube is sound but generalizes nothing,
/// and adopting it skips the solver's own `down`, which would have done better.
pub fn core_gain() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("INDUCTOR_CORE_GAIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
    })
}

/// Whether to bypass CPU validation of a card-proposed core.
///
/// Off by default. A gate-resident VCK5000 run on Problem04_label27 produced a
/// 212-to-5 literal core which changed a real SAT result into UNSAT. Until the
/// hardware path has a stronger proof boundary, every proposed core is only a
/// candidate and must be rechecked by BCP in a clone of the exact frame solver.
/// This switch is retained solely to reproduce old research measurements.
pub fn unchecked_core() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INDUCTOR_CORE_UNCHECKED").is_ok())
}

pub fn core_offload() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INDUCTOR_CORE").is_ok())
}

/// Whether core offload is restricted to queries which look profitable.
///
/// The card is an UNSAT/core co-processor, not a replacement SAT solver: a
/// satisfiable query pays the round trip and still has to run on the CPU. The
/// selector therefore learns, separately for coarse cube-length buckets, how
/// often CPU `down()` queries are UNSAT and how much time their BCP consumes.
/// A query is sent only after its bucket has enough observations and clears
/// all three gates: long cube, high UNSAT probability, expensive CPU BCP.
///
/// This is deliberately a second switch. `INDUCTOR_CORE=1` retains the old
/// ask-every-query experiment; adding `INDUCTOR_CORE_SELECTIVE=1` enables the
/// policy, so old measurements remain reproducible.
pub fn selective_core_offload() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INDUCTOR_CORE_SELECTIVE").is_ok())
}

const CORE_SELECT_BUCKETS: usize = 5;

#[derive(Clone, Copy, Debug)]
struct CoreSelectConfig {
    min_cube: usize,
    warmup: u64,
    min_unsat_pct: u64,
    min_bcp_ns: u64,
    card_warmup: u64,
    min_card_hit_pct: u64,
    reprobe_every: u64,
}

impl CoreSelectConfig {
    fn from_env() -> Self {
        fn value<T: std::str::FromStr>(name: &str, default: T) -> T {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        }
        Self {
            min_cube: value("INDUCTOR_SELECT_MIN_CUBE", 8),
            warmup: value("INDUCTOR_SELECT_WARMUP", 16),
            min_unsat_pct: value::<u64>("INDUCTOR_SELECT_MIN_UNSAT_PCT", 25).min(100),
            min_bcp_ns: value("INDUCTOR_SELECT_MIN_BCP_NS", 50_000),
            card_warmup: value("INDUCTOR_SELECT_CARD_WARMUP", 2),
            min_card_hit_pct: value::<u64>("INDUCTOR_SELECT_MIN_CARD_HIT_PCT", 1).min(100),
            reprobe_every: value("INDUCTOR_SELECT_REPROBE_EVERY", 4096),
        }
    }
}

fn core_select_config() -> &'static CoreSelectConfig {
    static CONFIG: std::sync::OnceLock<CoreSelectConfig> = std::sync::OnceLock::new();
    CONFIG.get_or_init(CoreSelectConfig::from_env)
}

/// Cube-length buckets: 0-3, 4-7, 8-15, 16-31, and 32+ literals.
#[inline]
fn core_select_bucket(len: usize) -> usize {
    match len {
        0..=3 => 0,
        4..=7 => 1,
        8..=15 => 2,
        16..=31 => 3,
        _ => 4,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CoreSelectDecision {
    Selected,
    Short,
    Warmup,
    LowUnsat,
    CheapBcp,
    LowCardHit,
}

fn core_select_decision(
    cfg: CoreSelectConfig,
    cube_len: usize,
    seen: u64,
    unsat: u64,
    bcp_ns: u64,
    card_seen: u64,
    card_hit: u64,
) -> CoreSelectDecision {
    if cube_len < cfg.min_cube {
        return CoreSelectDecision::Short;
    }
    // With a zero warm-up, a non-zero learned threshold still needs one real
    // observation. Setting all thresholds to zero intentionally restores
    // ask-every-eligible-query behaviour.
    if seen < cfg.warmup
        || (seen == 0 && (cfg.min_unsat_pct != 0 || cfg.min_bcp_ns != 0))
    {
        return CoreSelectDecision::Warmup;
    }
    if unsat.saturating_mul(100) < seen.saturating_mul(cfg.min_unsat_pct) {
        return CoreSelectDecision::LowUnsat;
    }
    if bcp_ns < seen.saturating_mul(cfg.min_bcp_ns) {
        return CoreSelectDecision::CheapBcp;
    }
    // CPU UNSAT is only a proxy for the event that saves work: propagation on
    // the weaker card clause set returning a core. Explore enough real card
    // queries to measure that event, then require an actual hit rate.
    if card_seen < cfg.card_warmup {
        return CoreSelectDecision::Selected;
    }
    if card_hit.saturating_mul(100)
        < card_seen.saturating_mul(cfg.min_card_hit_pct)
    {
        return CoreSelectDecision::LowCardHit;
    }
    CoreSelectDecision::Selected
}

static CORE_CPU_SEEN: [AtomicU64; CORE_SELECT_BUCKETS] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0),
];
static CORE_CPU_UNSAT: [AtomicU64; CORE_SELECT_BUCKETS] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0),
];
static CORE_CPU_BCP_NS: [AtomicU64; CORE_SELECT_BUCKETS] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0),
];
static CORE_CARD_SEEN: [AtomicU64; CORE_SELECT_BUCKETS] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0),
];
static CORE_CARD_HIT: [AtomicU64; CORE_SELECT_BUCKETS] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0),
];
static CORE_CARD_REJECTED: [AtomicU64; CORE_SELECT_BUCKETS] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0),
];

pub static CORE_SELECT_OFFERED: AtomicU64 = AtomicU64::new(0);
pub static CORE_SELECT_SELECTED: AtomicU64 = AtomicU64::new(0);
pub static CORE_SELECT_SHORT: AtomicU64 = AtomicU64::new(0);
pub static CORE_SELECT_WARMUP: AtomicU64 = AtomicU64::new(0);
pub static CORE_SELECT_LOW_UNSAT: AtomicU64 = AtomicU64::new(0);
pub static CORE_SELECT_CHEAP_BCP: AtomicU64 = AtomicU64::new(0);
pub static CORE_SELECT_LOW_CARD_HIT: AtomicU64 = AtomicU64::new(0);

/// Decide whether this `down()` query should be offered to the card.
///
/// Callers already check `core_offload()` and `ready()`. Returning true when
/// selection is disabled preserves the original full-offload path.
pub fn select_core_query(cube_len: usize) -> bool {
    if !selective_core_offload() {
        return true;
    }
    CORE_SELECT_OFFERED.fetch_add(1, Ordering::Relaxed);
    let bucket = core_select_bucket(cube_len);
    let cfg = *core_select_config();
    let mut decision = core_select_decision(
        cfg,
        cube_len,
        CORE_CPU_SEEN[bucket].load(Ordering::Relaxed),
        CORE_CPU_UNSAT[bucket].load(Ordering::Relaxed),
        CORE_CPU_BCP_NS[bucket].load(Ordering::Relaxed),
        CORE_CARD_SEEN[bucket].load(Ordering::Relaxed),
        CORE_CARD_HIT[bucket].load(Ordering::Relaxed),
    );
    // A cumulative hit rate deliberately damps noise, but can otherwise get
    // stuck after a cold phase. Keep one low-rate exploration stream so a
    // later frame with better card-resolvable conflicts can recover.
    if decision == CoreSelectDecision::LowCardHit && cfg.reprobe_every != 0 {
        let rejected = CORE_CARD_REJECTED[bucket].fetch_add(1, Ordering::Relaxed) + 1;
        if rejected.is_multiple_of(cfg.reprobe_every) {
            decision = CoreSelectDecision::Selected;
        }
    }
    let counter = match decision {
        CoreSelectDecision::Selected => &CORE_SELECT_SELECTED,
        CoreSelectDecision::Short => &CORE_SELECT_SHORT,
        CoreSelectDecision::Warmup => &CORE_SELECT_WARMUP,
        CoreSelectDecision::LowUnsat => &CORE_SELECT_LOW_UNSAT,
        CoreSelectDecision::CheapBcp => &CORE_SELECT_CHEAP_BCP,
        CoreSelectDecision::LowCardHit => &CORE_SELECT_LOW_CARD_HIT,
    };
    counter.fetch_add(1, Ordering::Relaxed);
    decision == CoreSelectDecision::Selected
}

/// Feed one CPU `down()` result back into the selector's online model.
///
/// Card hits return before the CPU solver runs and therefore do not have a CPU
/// BCP measurement. Misses and rejected queries do, and keep the model
/// adapting as the frame and lemma database evolve.
pub fn observe_core_query(cube_len: usize, unsat: bool, bcp_ns: u32) {
    if !selective_core_offload() {
        return;
    }
    let bucket = core_select_bucket(cube_len);
    CORE_CPU_SEEN[bucket].fetch_add(1, Ordering::Relaxed);
    if unsat {
        CORE_CPU_UNSAT[bucket].fetch_add(1, Ordering::Relaxed);
    }
    CORE_CPU_BCP_NS[bucket].fetch_add(bcp_ns as u64, Ordering::Relaxed);
}

/// Record whether an actually selected card query returned an UNSAT core.
pub fn observe_card_core_query(cube_len: usize, hit: bool) {
    if !selective_core_offload() {
        return;
    }
    let bucket = core_select_bucket(cube_len);
    CORE_CARD_SEEN[bucket].fetch_add(1, Ordering::Relaxed);
    if hit {
        CORE_CARD_HIT[bucket].fetch_add(1, Ordering::Relaxed);
    }
}

pub static CORE_USED: AtomicU64 = AtomicU64::new(0);
pub static CORE_ASKED: AtomicU64 = AtomicU64::new(0);
pub static CORE_GOT: AtomicU64 = AtomicU64::new(0);
pub static CORE_IN: AtomicU64 = AtomicU64::new(0);
pub static CORE_OUT: AtomicU64 = AtomicU64::new(0);
pub static CORE_VALIDATED: AtomicU64 = AtomicU64::new(0);
pub static CORE_VALIDATION_FAILED: AtomicU64 = AtomicU64::new(0);

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
    cdcl_host::flush_and_report();
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
        let n = MIC_N.load(O::Relaxed);
        if n > 0 {
            eprintln!(
                "inductor: {} generalizations, mean cube {:.1}, {:.0}% >= 8, {:.0}% >= 32, longest {}",
                n,
                MIC_LITS.load(O::Relaxed) as f64 / n as f64,
                100.0 * MIC_GE8.load(O::Relaxed) as f64 / n as f64,
                100.0 * MIC_GE32.load(O::Relaxed) as f64 / n as f64,
                MIC_MAX.load(O::Relaxed)
            );
        }
        if s.n_constraint > 0 || s.n_domain > 0 {
            eprintln!(
                "inductor: constraint {:.1} ms over {} calls, domain {:.1} ms over {}, \
                 core probe {:.1} ms, core minimise {:.1} ms",
                s.ns_constraint as f64 / 1e6, s.n_constraint,
                s.ns_domain as f64 / 1e6, s.n_domain,
                s.ns_core_probe as f64 / 1e6, s.ns_core_min as f64 / 1e6
            );
            // A constraint is inserted into slack reserved in the occurrence
            // index. When the slack does not fit, it is rebuilt in instead --
            // correct, and the cost the slack was there to remove. Silence on
            // this reads exactly like the optimisation not working.
            if s.cores_unminimised > 0 {
                eprintln!(
                    "inductor: {} cores returned unminimised (too short to be worth it)",
                    s.cores_unminimised
                );
            }
            if s.lem_full_rebuilds > 0 {
                eprintln!("inductor: {} lemma appends rebuilt the whole index",
                          s.lem_full_rebuilds);
            }
            if s.con_full_rebuilds > 0 {
                eprintln!(
                    "inductor: {} constraint appends rebuilt the whole index \
                     ({:.1}% of {}); the index slack is too small for this design",
                    s.con_full_rebuilds,
                    100.0 * s.con_full_rebuilds as f64 / s.n_constraint.max(1) as f64,
                    s.n_constraint
                );
            }
        }
        if s.n_down > 0 {
            eprintln!("inductor: fused down {:.1} ms over {} calls ({:.1} us each)",
                      s.ns_down as f64 / 1e6, s.n_down,
                      s.ns_down as f64 / 1e3 / s.n_down as f64);
        }
        if s.n_mic > 0 {
            eprintln!(
                "inductor: card mic {:.1} ms over {} calls, {} drops tried, {} literals in {} out ({:.2}x smaller)",
                s.ns_mic as f64 / 1e6, s.n_mic, s.mic_tried, s.mic_in, s.mic_out,
                if s.mic_out > 0 { s.mic_in as f64 / s.mic_out as f64 } else { 0.0 }
            );
        }
        let cs = CON_SET.load(O::Relaxed);
        if cs > 0 {
            eprintln!("inductor: constraints set {} times, {} refused by the card",
                      cs, CON_FAIL.load(O::Relaxed));
        }
        let mt = MIC_TAKEN.load(O::Relaxed);
        if mt > 0 {
            eprintln!("inductor: {} cubes started from the card's generalization", mt);
        }
        let ct = CORE_THIN.load(O::Relaxed);
        if ct > 0 {
            eprintln!("inductor: {} cores declined for generalizing less than {} literal(s)",
                      ct, core_gain());
        }
        let ca = CORE_ASKED.load(O::Relaxed);
        if ca > 0 {
            let cg = CORE_GOT.load(O::Relaxed);
            eprintln!(
                "inductor: cores asked {} got {} ({:.1}%), {} used by IC3, {} literals in {} out ({:.2}x smaller)",
                ca, cg, 100.0 * cg as f64 / ca as f64,
                CORE_USED.load(O::Relaxed),
                CORE_IN.load(O::Relaxed), CORE_OUT.load(O::Relaxed),
                if CORE_OUT.load(O::Relaxed) > 0 {
                    CORE_IN.load(O::Relaxed) as f64 / CORE_OUT.load(O::Relaxed) as f64
                } else { 0.0 }
            );
            if !unchecked_core() {
                eprintln!(
                    "inductor: card-core CPU validation passed {}, failed {}",
                    CORE_VALIDATED.load(O::Relaxed),
                    CORE_VALIDATION_FAILED.load(O::Relaxed),
                );
            }
        }
        if selective_core_offload() {
            let offered = CORE_SELECT_OFFERED.load(O::Relaxed);
            let selected = CORE_SELECT_SELECTED.load(O::Relaxed);
            let seen: u64 = CORE_CPU_SEEN.iter().map(|v| v.load(O::Relaxed)).sum();
            let unsat: u64 = CORE_CPU_UNSAT.iter().map(|v| v.load(O::Relaxed)).sum();
            let bcp_ns: u64 = CORE_CPU_BCP_NS.iter().map(|v| v.load(O::Relaxed)).sum();
            let card_seen: u64 = CORE_CARD_SEEN.iter().map(|v| v.load(O::Relaxed)).sum();
            let card_hit: u64 = CORE_CARD_HIT.iter().map(|v| v.load(O::Relaxed)).sum();
            let cfg = core_select_config();
            eprintln!(
                "inductor: selective core offered {}, selected {} ({:.1}%); rejected short {}, warm-up {}, low-UNSAT {}, cheap-BCP {}, low-card-hit {}",
                offered,
                selected,
                if offered > 0 { 100.0 * selected as f64 / offered as f64 } else { 0.0 },
                CORE_SELECT_SHORT.load(O::Relaxed),
                CORE_SELECT_WARMUP.load(O::Relaxed),
                CORE_SELECT_LOW_UNSAT.load(O::Relaxed),
                CORE_SELECT_CHEAP_BCP.load(O::Relaxed),
                CORE_SELECT_LOW_CARD_HIT.load(O::Relaxed),
            );
            eprintln!(
                "inductor: selective calibration {} CPU queries, {:.1}% UNSAT, mean BCP {:.1} us; thresholds cube >= {}, bucket warm-up {}, UNSAT >= {}%, BCP >= {:.1} us",
                seen,
                if seen > 0 { 100.0 * unsat as f64 / seen as f64 } else { 0.0 },
                if seen > 0 { bcp_ns as f64 / seen as f64 / 1000.0 } else { 0.0 },
                cfg.min_cube,
                cfg.warmup,
                cfg.min_unsat_pct,
                cfg.min_bcp_ns as f64 / 1000.0,
            );
            eprintln!(
                "inductor: selective card calibration {} probes, {} cores ({:.2}%); thresholds card warm-up {}, hit >= {}%, reprobe every {} rejects",
                card_seen,
                card_hit,
                if card_seen > 0 { 100.0 * card_hit as f64 / card_seen as f64 } else { 0.0 },
                cfg.card_warmup,
                cfg.min_card_hit_pct,
                cfg.reprobe_every,
            );
        }
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

#[cfg(test)]
mod core_select_tests {
    use super::{CoreSelectConfig, CoreSelectDecision, core_select_bucket, core_select_decision};

    fn cfg() -> CoreSelectConfig {
        CoreSelectConfig {
            min_cube: 8,
            warmup: 4,
            min_unsat_pct: 25,
            min_bcp_ns: 50_000,
            card_warmup: 2,
            min_card_hit_pct: 1,
            reprobe_every: 128,
        }
    }

    #[test]
    fn cube_length_buckets_cover_boundaries() {
        assert_eq!(core_select_bucket(3), 0);
        assert_eq!(core_select_bucket(4), 1);
        assert_eq!(core_select_bucket(8), 2);
        assert_eq!(core_select_bucket(16), 3);
        assert_eq!(core_select_bucket(32), 4);
        assert_eq!(core_select_bucket(10_000), 4);
    }

    #[test]
    fn selector_applies_all_three_profitability_gates() {
        assert_eq!(core_select_decision(cfg(), 7, 100, 100, 99_000_000, 2, 2), CoreSelectDecision::Short);
        assert_eq!(core_select_decision(cfg(), 8, 3, 3, 3_000_000, 2, 2), CoreSelectDecision::Warmup);
        assert_eq!(core_select_decision(cfg(), 8, 4, 0, 4_000_000, 2, 2), CoreSelectDecision::LowUnsat);
        assert_eq!(core_select_decision(cfg(), 8, 4, 1, 199_999, 2, 2), CoreSelectDecision::CheapBcp);
        assert_eq!(core_select_decision(cfg(), 8, 4, 1, 200_000, 0, 0), CoreSelectDecision::Selected);
        assert_eq!(core_select_decision(cfg(), 8, 4, 1, 200_000, 2, 0), CoreSelectDecision::LowCardHit);
        assert_eq!(core_select_decision(cfg(), 8, 4, 1, 200_000, 100, 1), CoreSelectDecision::Selected);
    }

    #[test]
    fn zero_thresholds_can_reproduce_unconditional_selection() {
        let all = CoreSelectConfig {
            min_cube: 0,
            warmup: 0,
            min_unsat_pct: 0,
            min_bcp_ns: 0,
            card_warmup: 0,
            min_card_hit_pct: 0,
            reprobe_every: 0,
        };
        assert_eq!(core_select_decision(all, 0, 0, 0, 0, 0, 0), CoreSelectDecision::Selected);
    }
}
