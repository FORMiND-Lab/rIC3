//! FastLane: software model of a resident hardware frame-solver lane.
//!
//! With `INDUCTOR_FASTLANE=model`, every relative-induction query
//! (`TransysSolver::inductive_with_constrain`) whose phase is selected by
//! `INDUCTOR_FASTLANE_PHASES` (default: gen,block,push) is answered by a lane
//! that mirrors the frame solver's permanent clauses.  The lane result is
//! installed into GipSAT's live model/core state so IC3 consumes it exactly as
//! a CPU answer.  The model counts the hardware events that the cycle model
//! weights; it never measures time.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

#[cfg(has_fastlane)]
mod ffi {
    #[repr(C)]
    pub struct FlLane {
        _private: [u8; 0],
    }
    unsafe extern "C" {
        pub fn fl_new(n_var: u32, dep_offsets: *const u32, deps: *const u32) -> *mut FlLane;
        pub fn fl_clone(lane: *const FlLane) -> *mut FlLane;
        pub fn fl_free(lane: *mut FlLane);
        pub fn fl_add_clause(lane: *mut FlLane, lits: *const u32, n: u32);
        pub fn fl_add_fixed_domain(lane: *mut FlLane, var: u32);
        pub fn fl_add_clause_ranged(lane: *mut FlLane, lits: *const u32, n: u32, lo: u32, hi: u32);
        pub fn fl_solve_frame(
            lane: *mut FlLane,
            assumptions: *const u32,
            n_assumptions: u32,
            temp_lits: *const u32,
            temp_lens: *const u32,
            n_temps: u32,
            conflict_budget: u64,
            frame: u32,
        ) -> i32;
        pub fn fl_set_ts(lane: *mut FlLane, next_of_var: *const u32, init_of_var: *const u8);
        pub fn fl_pmic(
            lanes: *mut *mut FlLane,
            k: u32,
            frame: u32,
            cube: *const u32,
            n_cube: u32,
            cons_lits: *const u32,
            cons_lens: *const u32,
            n_cons: u32,
            act_key: *const u64,
            out_cube: *mut u32,
            cycles: *mut f64,
            rounds: *mut u64,
            chains: *mut u64,
            wasted: *mut u64,
            queries: *mut u64,
        ) -> u32;
        pub fn fl_domain(lane: *const FlLane, out: *mut *const u32) -> u32;
        pub fn fl_model(lane: *const FlLane, out: *mut *const u32) -> u32;
        pub fn fl_core(lane: *const FlLane, out: *mut *const u32) -> u32;
        pub fn fl_core_used_temps(lane: *const FlLane) -> i32;
        pub fn fl_global_events(out: *mut u64);
        pub fn fl_board_ready() -> i32;
        pub fn fl_board_shutdown();
        pub fn fl_board_abort();
        pub fn fl_board_reset();
        pub fn fl_board_go();
        pub fn fl_solve_frame_submit(
            lane: *mut FlLane,
            assumptions: *const u32,
            n_assumptions: u32,
            temp_lits: *const u32,
            temp_lens: *const u32,
            n_temps: u32,
            conflict_budget: u64,
            frame: u32,
        ) -> i32;
        pub fn fl_board_poll() -> i32;
        pub fn fl_board_busy() -> i32;
        pub fn fl_solve_frame_collect(lane: *mut FlLane) -> i32;
        pub fn fl_batch_submit(lane: *mut FlLane, words: *const u32, offs: *const u32, n: u32) -> i32;
        pub fn fl_batch_collect() -> u32;
        pub fn fl_batch_take(lane: *mut FlLane, i: u32) -> i32;
        pub fn fl_learnts(lane: *const FlLane, out: *mut *const u32) -> u32;
        pub fn fl_board_kernel_ns() -> u64;
        pub fn fl_pmic_push(
            lanes: *mut *mut FlLane,
            k: u32,
            frame: u32,
            cube: *const u32,
            n_cube: u32,
            cons_lits: *const u32,
            cons_lens: *const u32,
            n_cons: u32,
            act_key: *const u64,
            out_cube: *mut u32,
            cycles: *mut f64,
            rounds: *mut u64,
            chains: *mut u64,
            wasted: *mut u64,
            queries: *mut u64,
            push_on: i32,
            push_from: u32,
            push_to: u32,
            push_frame: *mut u32,
            push_complete: *mut i32,
            push_cube: *mut u32,
            push_n: *mut u32,
        ) -> u32;
    }
}

pub const EVENT_NAMES: [&str; 32] = [
    "queries",
    "setup_domain_vars",
    "setup_dep_edges",
    "setup_temp_lits",
    "bcp_lits",
    "bcp_occ",
    "assigns",
    "props",
    "undo_lits",
    "undo_occ",
    "unassign",
    "decisions",
    "decide_pops",
    "conflicts",
    "analyze_lits",
    "analyze_steps",
    "minimize_lits",
    "learnt_lits",
    "bumps",
    "restarts",
    "core_steps",
    "model_vars",
    "reduce_runs",
    "gc_runs",
    "gc_lits",
    "clause_adds",
    "clause_add_lits",
    "wl_visits",
    "wl_blocked",
    "wl_clause_reads",
    "wl_scan_lits",
    "wl_moves",
];

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        cfg!(has_fastlane)
            && std::env::var("INDUCTOR_FASTLANE")
                .is_ok_and(|value| matches!(value.as_str(), "model" | "1" | "shadow"))
    })
}

/// Shadow mode: GipSAT answers (CPU trajectory unchanged); the lane solves
/// the same query alongside so per-query CPU time and lane events pair up.
pub fn shadow() -> bool {
    static SHADOW: OnceLock<bool> = OnceLock::new();
    *SHADOW.get_or_init(|| std::env::var("INDUCTOR_FASTLANE").is_ok_and(|v| v == "shadow"))
}

pub static SHADOW_MISMATCH: AtomicU64 = AtomicU64::new(0);
/// Shadow mode, queries both solvers answer UNSAT: count and the summed
/// core sizes (assumptions in the core) of GipSAT and of the lane.
pub static SHADOW_UNSAT: AtomicU64 = AtomicU64::new(0);
pub static SHADOW_CORE_CPU: AtomicU64 = AtomicU64::new(0);
pub static SHADOW_CORE_LANE: AtomicU64 = AtomicU64::new(0);

/// Validate every lane SAT model against the exact CPU formula before use.
pub fn check_models() -> bool {
    static CHECK: OnceLock<bool> = OnceLock::new();
    *CHECK.get_or_init(|| std::env::var_os("INDUCTOR_FASTLANE_CHECK").is_some())
}

/// Is a query of this phase (`inductor_trace::Phase as u8`) routed to a lane?
pub fn routes_phase(phase: u8) -> bool {
    static MASK: OnceLock<u8> = OnceLock::new();
    let mask = *MASK.get_or_init(|| {
        let spec = std::env::var("INDUCTOR_FASTLANE_PHASES")
            .unwrap_or_else(|_| "gen,block,push".to_string());
        let mut mask = 0u8;
        for name in spec.split(',') {
            let bit = match name.trim() {
                "gen" => inductor_trace::Phase::Gen as u8,
                "block" => inductor_trace::Phase::Block as u8,
                "push" => inductor_trace::Phase::Push as u8,
                "inf" => inductor_trace::Phase::Inf as u8,
                "other" => inductor_trace::Phase::Other as u8,
                _ => continue,
            };
            mask |= 1 << bit;
        }
        mask
    });
    phase < 8 && mask & (1 << phase) != 0
}

pub static LANE_NS: AtomicU64 = AtomicU64::new(0);

type LogWriter = std::sync::Mutex<Option<std::io::BufWriter<std::fs::File>>>;

fn log_writer() -> &'static LogWriter {
    use std::io::Write;
    static LOG: OnceLock<LogWriter> = OnceLock::new();
    LOG.get_or_init(|| {
        let writer = std::env::var("INDUCTOR_FASTLANE_LOG").ok().and_then(|path| {
            let mut w = std::io::BufWriter::new(std::fs::File::create(path).ok()?);
            let _ = writeln!(w, "phase\top\troot_op\tdown\tframe\tstatus\tmodel_ns\tcpu_ns\tcpu_setup_ns\tcpu_search_ns\tcpu_bcp_ns\tcpu_teardown_ns\tcpu_decide\tcpu_prop\tcpu_conflict\tcpu_learnt\t{}", EVENT_NAMES.join("\t"));
            Some(w)
        });
        std::sync::Mutex::new(writer)
    })
}

/// Optional per-query event log (`INDUCTOR_FASTLANE_LOG=<path>`).
pub fn log_query(
    phase: u8,
    frame: u32,
    status: u8,
    ns: u64,
    cpu_ns: u64,
    cpu_probe: &str,
    before: &[u64; 32],
    after: &[u64; 32],
) {
    use std::io::Write;
    let Ok(mut guard) = log_writer().lock() else {
        return;
    };
    let Some(w) = guard.as_mut() else { return };
    let (op, root) = crate::inductor::current_ops();
    let down = crate::inductor::current_down_id();
    let deltas: Vec<String> = before
        .iter()
        .zip(after)
        .map(|(b, a)| (a - b).to_string())
        .collect();
    let _ = writeln!(
        w,
        "{phase}\t{op}\t{root}\t{down}\t{frame}\t{status}\t{ns}\t{cpu_ns}\t{cpu_probe}\t{}",
        deltas.join("\t")
    );
}

pub fn flush_log() {
    use std::io::Write;
    if let Ok(mut guard) = log_writer().lock()
        && let Some(w) = guard.as_mut()
    {
        let _ = w.flush();
    }
}

pub static ROUTED: AtomicU64 = AtomicU64::new(0);
/// Raced queries the CPU solved alone because the card was still finishing
/// a stale (CPU-won) run, and the CPU time they took.
pub static RACE_BUSY: AtomicU64 = AtomicU64::new(0);
/// Highest frame solver bound to the pool, and raced queries by frame depth
/// below it (clipped at 7): submitted, and answered by the card.
pub static MAX_FRAME: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
pub static DEPTH_SUB: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
pub static DEPTH_WIN: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
pub fn frame_depth(frame: u32) -> usize {
    (MAX_FRAME.load(Ordering::Relaxed).saturating_sub(frame) as usize).min(7)
}
pub static RACE_BUSY_NS: AtomicU64 = AtomicU64::new(0);
pub static ROUTED_SAT: AtomicU64 = AtomicU64::new(0);
pub static ROUTED_UNSAT: AtomicU64 = AtomicU64::new(0);
pub static ROUTED_UNKNOWN: AtomicU64 = AtomicU64::new(0);
pub static INSTALL_FAILED: AtomicU64 = AtomicU64::new(0);
pub static PHASE_COUNTS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

pub enum LaneResult {
    Sat { domain: Vec<u32>, model: Vec<u32> },
    Unsat { core: Vec<u32>, used_temps: bool },
    Unknown,
}

/// Owned handle to one lane; cloning a frame solver clones its lane.
pub struct LaneHandle {
    #[cfg(has_fastlane)]
    ptr: *mut ffi::FlLane,
}

// IC3 owns solvers on one thread; the handle moves with its solver.
unsafe impl Send for LaneHandle {}
unsafe impl Sync for LaneHandle {}

impl LaneHandle {
    #[allow(unused_variables)]
    pub fn new(n_var: u32, dep_offsets: &[u32], deps: &[u32]) -> Option<Self> {
        #[cfg(has_fastlane)]
        {
            let ptr = unsafe { ffi::fl_new(n_var, dep_offsets.as_ptr(), deps.as_ptr()) };
            return (!ptr.is_null()).then_some(Self { ptr });
        }
        #[allow(unreachable_code)]
        None
    }

    #[allow(unused_variables)]
    pub fn add_clause(&mut self, lits: &[u32]) {
        #[cfg(has_fastlane)]
        unsafe {
            ffi::fl_add_clause(self.ptr, lits.as_ptr(), lits.len() as u32)
        }
    }

    #[allow(unused_variables)]
    pub fn add_fixed_domain(&mut self, var: u32) {
        #[cfg(has_fastlane)]
        unsafe {
            ffi::fl_add_fixed_domain(self.ptr, var)
        }
    }

    #[allow(unused_variables)]
    pub fn add_clause_ranged(&mut self, lits: &[u32], lo: u32, hi: u32) {
        #[cfg(has_fastlane)]
        unsafe {
            ffi::fl_add_clause_ranged(self.ptr, lits.as_ptr(), lits.len() as u32, lo, hi)
        }
    }

    pub fn solve(&mut self, assumptions: &[u32], temps: &[Vec<u32>], budget: u64) -> LaneResult {
        self.solve_frame(assumptions, temps, budget, 0)
    }

    #[allow(unused_variables)]
    pub fn solve_frame(
        &mut self,
        assumptions: &[u32],
        temps: &[Vec<u32>],
        budget: u64,
        frame: u32,
    ) -> LaneResult {
        #[cfg(has_fastlane)]
        unsafe {
            let lens: Vec<u32> = temps.iter().map(|t| t.len() as u32).collect();
            let flat: Vec<u32> = temps.iter().flatten().copied().collect();
            let r = ffi::fl_solve_frame(
                self.ptr,
                assumptions.as_ptr(),
                assumptions.len() as u32,
                flat.as_ptr(),
                lens.as_ptr(),
                lens.len() as u32,
                budget,
                frame,
            );
            return self.result(r);
        }
        #[allow(unreachable_code)]
        LaneResult::Unknown
    }

    /// Start this SOLVE on the card without waiting (board mode, replica
    /// 0); false if racing is not available.
    #[allow(unused_variables)]
    pub fn submit_frame(&mut self, assumptions: &[u32], temps: &[Vec<u32>], frame: u32) -> bool {
        #[cfg(has_fastlane)]
        unsafe {
            let lens: Vec<u32> = temps.iter().map(|t| t.len() as u32).collect();
            let flat: Vec<u32> = temps.iter().flatten().copied().collect();
            return ffi::fl_solve_frame_submit(
                self.ptr,
                assumptions.as_ptr(),
                assumptions.len() as u32,
                flat.as_ptr(),
                lens.as_ptr(),
                lens.len() as u32,
                0,
                frame,
            ) != 0;
        }
        #[allow(unreachable_code)]
        false
    }

    /// Start several SOLVEs at `frame` on the card in one run (board mode,
    /// replica 0); false if not available.
    #[allow(unused_variables)]
    pub fn submit_batch(&mut self, queries: &[(Vec<u32>, Vec<Vec<u32>>)], frame: u32) -> bool {
        #[cfg(has_fastlane)]
        unsafe {
            let mut words = Vec::new();
            let mut offs = vec![0u32];
            for (assumptions, temps) in queries {
                words.extend([frame, 0, assumptions.len() as u32]);
                words.extend_from_slice(assumptions);
                words.push(temps.len() as u32);
                for t in temps {
                    words.push(t.len() as u32);
                    words.extend_from_slice(t);
                }
                offs.push(words.len() as u32);
            }
            return ffi::fl_batch_submit(
                self.ptr,
                words.as_ptr(),
                offs.as_ptr(),
                queries.len() as u32,
            ) != 0;
        }
        #[allow(unreachable_code)]
        false
    }

    /// Result i of the batch collected by `batch_collect`.
    #[allow(unused_variables)]
    pub fn batch_take(&mut self, i: u32) -> LaneResult {
        #[cfg(has_fastlane)]
        unsafe {
            let r = ffi::fl_batch_take(self.ptr, i);
            return self.result(r);
        }
        #[allow(unreachable_code)]
        LaneResult::Unknown
    }

    /// The learnts exported with the last card result (`x_len`), as records
    /// [len, lo, hi, lits...].
    pub fn learnts(&self) -> Vec<u32> {
        #[cfg(has_fastlane)]
        unsafe {
            let mut p: *const u32 = std::ptr::null();
            let n = ffi::fl_learnts(self.ptr, &mut p);
            return if n == 0 { Vec::new() } else { std::slice::from_raw_parts(p, n as usize).to_vec() };
        }
        #[allow(unreachable_code)]
        Vec::new()
    }

    /// Result of the submitted SOLVE (waits for the card).
    pub fn collect_frame(&mut self) -> LaneResult {
        #[cfg(has_fastlane)]
        unsafe {
            let r = ffi::fl_solve_frame_collect(self.ptr);
            return self.result(r);
        }
        #[allow(unreachable_code)]
        LaneResult::Unknown
    }

    #[cfg(has_fastlane)]
    fn result(&self, r: i32) -> LaneResult {
        unsafe {
            let read = |f: unsafe extern "C" fn(*const ffi::FlLane, *mut *const u32) -> u32| {
                let mut p: *const u32 = std::ptr::null();
                let n = f(self.ptr, &mut p);
                if n == 0 {
                    Vec::new()
                } else {
                    std::slice::from_raw_parts(p, n as usize).to_vec()
                }
            };
            return match r {
                1 => LaneResult::Sat {
                    domain: read(ffi::fl_domain),
                    model: read(ffi::fl_model),
                },
                2 => LaneResult::Unsat {
                    core: read(ffi::fl_core),
                    used_temps: ffi::fl_core_used_temps(self.ptr) != 0,
                },
                _ => LaneResult::Unknown,
            };
        }
        #[allow(unreachable_code)]
        LaneResult::Unknown
    }
}

impl Clone for LaneHandle {
    #[cfg(has_fastlane)]
    fn clone(&self) -> Self {
        Self {
            ptr: unsafe { ffi::fl_clone(self.ptr) },
        }
    }

    #[cfg(not(has_fastlane))]
    fn clone(&self) -> Self {
        Self {}
    }
}

impl Drop for LaneHandle {
    fn drop(&mut self) {
        #[cfg(has_fastlane)]
        unsafe {
            ffi::fl_free(self.ptr)
        }
    }
}

/// Frame-ranged replica pool (`INDUCTOR_FASTLANE_POOL=1`): every replica
/// holds all IC3 frames, each clause tagged with the solver indices it is
/// active for, exactly as a hardware lane would.
pub fn pool_mode() -> bool {
    static POOL: OnceLock<bool> = OnceLock::new();
    *POOL.get_or_init(|| enabled() && std::env::var_os("INDUCTOR_FASTLANE_POOL").is_some())
}

pub fn replica_count() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("INDUCTOR_FASTLANE_REPLICAS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
            .clamp(1, 64)
    })
}

pub struct Pool {
    pub replicas: Vec<LaneHandle>,
    pub n_var: u32,
    init: PoolInit,
}

/// What the pool was created from (transition, constraints, dependency
/// graph), kept to rebuild it on a reload.
struct PoolInit {
    n_var: u32,
    dep_offsets: Vec<u32>,
    deps: Vec<u32>,
    trans: Vec<Vec<u32>>,
    base: Vec<Vec<u32>>,
}

impl PoolInit {
    fn build(&self) -> Option<Vec<LaneHandle>> {
        let mut replicas = Vec::new();
        for _ in 0..replica_count() {
            let mut lane = LaneHandle::new(self.n_var, &self.dep_offsets, &self.deps)?;
            lane.add_fixed_domain(0);
            for c in &self.trans {
                lane.add_clause(c);
            }
            for c in &self.base {
                lane.add_clause(c);
                for l in c {
                    lane.add_fixed_domain(l >> 1);
                }
            }
            replicas.push(lane);
        }
        Some(replicas)
    }
}

/// The card's lane ran out of capacity (it answers UNKNOWN from then on):
/// stop routing to it until IC3 rebuilds the pool from its live frames.
static RELOAD_WANTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static RELOADS: AtomicU64 = AtomicU64::new(0);

pub fn request_reload() {
    if pool_mode() {
        RELOAD_WANTED.store(true, Ordering::Relaxed);
    }
}

#[inline]
pub fn reload_pending() -> bool {
    RELOAD_WANTED.load(Ordering::Relaxed)
}

/// Rebuild the pool (card lane included) from the creation-time clauses and
/// the live frame lemmas `(clause, lo, hi)`.  Every dropped clause is a
/// subsumed or moved lemma copy implied by the live ones, or (frame band)
/// ends below `band_lo`.
pub fn pool_reload(lemmas: &[(Vec<u32>, u32, u32)], band_lo: u32) {
    BAND_LO.store(band_lo, Ordering::Relaxed);
    LOADED.store(lemmas.len() as u64, Ordering::Relaxed);
    RELOAD_AT_NS.store(reload_clock().elapsed().as_nanos() as u64, Ordering::Relaxed);
    WAIT_AT_RELOAD_NS.store(SPLIT_WAIT_NS.load(Ordering::Relaxed), Ordering::Relaxed);
    LOADED_SUM.fetch_add(lemmas.len() as u64, Ordering::Relaxed);
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let Some(old) = pool.take() else {
            return;
        };
        let init = old.init;
        drop(old.replicas);
        #[cfg(has_fastlane)]
        unsafe {
            ffi::fl_board_reset()
        }
        let Some(mut replicas) = init.build() else {
            return;
        };
        for (c, lo, hi) in lemmas {
            for lane in replicas.iter_mut() {
                lane.add_clause_ranged(c, *lo, *hi);
            }
        }
        *pool = Some(Pool {
            replicas,
            n_var: init.n_var,
            init,
        });
    });
    RELOAD_WANTED.store(false, Ordering::Relaxed);
    ANSWERED.store(false, Ordering::Relaxed);
    MIRRORED.store(0, Ordering::Relaxed);
    SLOW_RUNS.store(0, Ordering::Relaxed);
    SLOW_BASE_NS.store(0, Ordering::Relaxed);
    RELOADS.fetch_add(1, Ordering::Relaxed);
}

/// Speculative parallel MIC (`INDUCTOR_FASTLANE_PMIC=1`, pool mode, CTG off).
/// False while the card is still coming up (board mode): the model checker
/// answers on the CPU until then, so device bring-up overlaps useful work.
pub fn card_ready() -> bool {
    #[cfg(has_fastlane)]
    {
        static READY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if READY.load(Ordering::Relaxed) {
            return true;
        }
        let ready = unsafe { ffi::fl_board_ready() } != 0;
        if ready {
            READY.store(true, Ordering::Relaxed);
        }
        return ready;
    }
    #[allow(unreachable_code)]
    true
}

/// Conflict budget of a CPU query during card warm-up.
pub fn warmup_conflicts() -> u32 {
    static B: OnceLock<u32> = OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("INDUCTOR_FASTLANE_WARMUP_CONFLICTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1000)
    })
}

/// CPU-first conflict budget once the card is up (`INDUCTOR_FASTLANE_CPU_FIRST`).
pub fn cpu_first_conflicts() -> Option<u32> {
    static B: OnceLock<Option<u32>> = OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("INDUCTOR_FASTLANE_CPU_FIRST")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}

/// `INDUCTOR_FASTLANE_ADAPT=1`: the CPU-first budget follows the card's race
/// win rate.  Every 256 races it doubles when the card won fewer than 5%
/// (light queries: a race only costs the CPU a submission) and halves when
/// it won more than 20%, between `INDUCTOR_FASTLANE_CPU_FIRST` and 4096.
pub fn adapt_enabled() -> bool {
    static A: OnceLock<bool> = OnceLock::new();
    *A.get_or_init(|| std::env::var("INDUCTOR_FASTLANE_ADAPT").is_ok_and(|v| v == "1"))
}

static ADAPT_BUDGET: AtomicU32 = AtomicU32::new(0);
static ADAPT_RACES: AtomicU32 = AtomicU32::new(0);
static ADAPT_CARD: AtomicU32 = AtomicU32::new(0);
/// Budget changes made by the adaptation (up, down).
pub static ADAPT_UP: AtomicU64 = AtomicU64::new(0);
pub static ADAPT_DOWN: AtomicU64 = AtomicU64::new(0);

/// The CPU-first budget in effect (adaptive when enabled).
pub fn cpu_first_budget() -> Option<u32> {
    let base = cpu_first_conflicts()?;
    if !adapt_enabled() {
        return Some(base);
    }
    let b = ADAPT_BUDGET.load(Ordering::Relaxed);
    Some(if b == 0 { base } else { b })
}

/// Outcome of one race, for the adaptive budget.
pub fn note_race(card_won: bool) {
    if !adapt_enabled() {
        return;
    }
    let Some(base) = cpu_first_conflicts() else {
        return;
    };
    if card_won {
        ADAPT_CARD.fetch_add(1, Ordering::Relaxed);
    }
    if ADAPT_RACES.fetch_add(1, Ordering::Relaxed) + 1 < 256 {
        return;
    }
    let card = ADAPT_CARD.swap(0, Ordering::Relaxed);
    ADAPT_RACES.store(0, Ordering::Relaxed);
    let b = cpu_first_budget().unwrap_or(base);
    let nb = if card * 20 < 256 {
        (b * 2).min(4096)
    } else if card * 5 > 256 {
        (b / 2).max(base)
    } else {
        b
    };
    if nb > b {
        ADAPT_UP.fetch_add(1, Ordering::Relaxed);
    } else if nb < b {
        ADAPT_DOWN.fetch_add(1, Ordering::Relaxed);
    }
    ADAPT_BUDGET.store(nb, Ordering::Relaxed);
}

/// Race the card against the CPU on queries past the CPU-first budget
/// (`INDUCTOR_FASTLANE_RACE=1`): the first answer is used.
pub fn race_enabled() -> bool {
    static R: OnceLock<bool> = OnceLock::new();
    *R.get_or_init(|| std::env::var("INDUCTOR_FASTLANE_RACE").is_ok_and(|v| v == "1"))
}

/// Push split (`INDUCTOR_FASTLANE_PUSH_SPLIT=1`, board pool mode): in each
/// frame's push pass the card answers a suffix of the lemmas in one run
/// while the CPU checks the rest, sized from measured costs so both finish
/// together.  No query is solved twice.
pub fn push_split_enabled() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("INDUCTOR_FASTLANE_PUSH_SPLIT").is_ok_and(|v| v == "1"))
}

/// Control for the push split (`INDUCTOR_FASTLANE_PUSH_SPLIT_CPU=1`, no
/// card): the same split, sized the same way, but the CPU answers the tail
/// against the frame as it was at the split, exactly when the card would
/// have been sent it.  It isolates what the split itself changes in the
/// search (stale answers) from the work the card takes off the CPU.
/// Pushed lemmas and their cores, by who answered (card / CPU): whether the
/// card's cores shrink lemmas as much as GipSAT's.
static CORE_N: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static CORE_LEMMA_LITS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static CORE_LITS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
pub fn note_push_core(card: bool, lemma_len: usize, core_len: usize) {
    let k = card as usize;
    CORE_N[k].fetch_add(1, Ordering::Relaxed);
    CORE_LEMMA_LITS[k].fetch_add(lemma_len as u64, Ordering::Relaxed);
    CORE_LITS[k].fetch_add(core_len as u64, Ordering::Relaxed);
}

/// INDUCTOR_FASTLANE_SPLIT_LOCK=1: a split that leaves the search unchanged
/// whoever answers, for measuring the card against its CPU counterpart: the
/// card's part is a fixed fraction of each frame of 16 lemmas or more
/// (INDUCTOR_FASTLANE_SPLIT_LOCK_FRAC, default 0.25, at most 384), answered
/// by the card or else on a clone of the frame solver, and its pushed lemmas
/// keep all their literals.  The frame solvers then run the same checks with
/// the card, with the CPU-split control, or with the card unavailable.
pub fn split_lock() -> Option<f64> {
    static LOCK: OnceLock<Option<f64>> = OnceLock::new();
    *LOCK.get_or_init(|| {
        std::env::var("INDUCTOR_FASTLANE_SPLIT_LOCK").is_ok_and(|v| v == "1").then(|| {
            std::env::var("INDUCTOR_FASTLANE_SPLIT_LOCK_FRAC")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .map_or(0.25, |f| f.clamp(0.0, 1.0))
        })
    })
}

/// Lock mode: the card's part of a frame of `n` lemmas.
/// (INDUCTOR_FASTLANE_SPLIT_LOCK_MIN: smallest frame split, default 16)
pub fn lock_size(n: usize) -> usize {
    static MIN: OnceLock<usize> = OnceLock::new();
    let min = *MIN.get_or_init(|| {
        std::env::var("INDUCTOR_FASTLANE_SPLIT_LOCK_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(16)
    });
    match split_lock() {
        Some(f) if n >= min => ((n as f64 * f) as usize).min(384),
        _ => 0,
    }
}

/// INDUCTOR_FASTLANE_SPLIT_FB_BATCH=1: lock mode checks a split's missing
/// card answers on one clone (see `propagate`).
pub fn split_fb_batch() -> bool {
    static B: OnceLock<bool> = OnceLock::new();
    *B.get_or_init(|| std::env::var("INDUCTOR_FASTLANE_SPLIT_FB_BATCH").is_ok_and(|v| v == "1"))
}

/// INDUCTOR_FASTLANE_SPLIT_IMPORT=L: lock mode copies the clauses of at most
/// L literals that the clone learns on a split's CPU answers back into the
/// frame solver (0: off), as the card would return its learnts.
pub fn split_import() -> usize {
    static L: OnceLock<usize> = OnceLock::new();
    *L.get_or_init(|| std::env::var("INDUCTOR_FASTLANE_SPLIT_IMPORT").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// INDUCTOR_FASTLANE_X_LEN=L (bitstream with learnt export): the card returns
/// each push-split answer's learnts of at most L literals, and the frame
/// solver imports those valid at its frame (0: off).
pub fn x_len() -> usize {
    static L: OnceLock<usize> = OnceLock::new();
    *L.get_or_init(|| std::env::var("INDUCTOR_FASTLANE_X_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// INDUCTOR_FASTLANE_X_CHECK=1 (debug): check that every exported learnt is
/// implied by the frame solver before importing it.
pub fn x_check() -> bool {
    static B: OnceLock<bool> = OnceLock::new();
    *B.get_or_init(|| std::env::var("INDUCTOR_FASTLANE_X_CHECK").is_ok_and(|v| v == "1"))
}
pub static X_CHECK_OK: AtomicU64 = AtomicU64::new(0);
pub static X_CHECK_BAD: AtomicU64 = AtomicU64::new(0);
pub static X_CHECK_UNK: AtomicU64 = AtomicU64::new(0);

/// Learnt clauses imported into frame solvers (`split_import`, `x_len`).
pub static SPLIT_IMPORTED: AtomicU64 = AtomicU64::new(0);

/// Time spent cloning frame solvers for lock-mode CPU answers (ns).
pub static SPLIT_CLONE_NS: AtomicU64 = AtomicU64::new(0);

/// The card split stops for the rest of the run after
/// INDUCTOR_FASTLANE_SPLIT_MAX_RELOADS (default 30) pool reloads: the lemmas
/// then no longer fit the card, and every reload resends them all.
pub fn split_card_off() -> bool {
    static MAX: OnceLock<u64> = OnceLock::new();
    let max = *MAX.get_or_init(|| {
        std::env::var("INDUCTOR_FASTLANE_SPLIT_MAX_RELOADS").ok().and_then(|v| v.parse().ok()).unwrap_or(30)
    });
    // INDUCTOR_FASTLANE_SPLIT_CAP_OVF=1: count only the reloads an overflow
    // asked for; proactive ones (RELOAD_REL, RELOAD_SLOW) keep a lane fresh
    // that still fits, and on small designs they are cheap and frequent
    static OVF: OnceLock<bool> = OnceLock::new();
    let n = if *OVF.get_or_init(|| std::env::var("INDUCTOR_FASTLANE_SPLIT_CAP_OVF").is_ok_and(|v| v == "1")) {
        RELOADS.load(Ordering::Relaxed)
            .saturating_sub(REL_RELOADS.load(Ordering::Relaxed) + SLOW_RELOADS.load(Ordering::Relaxed))
    } else {
        RELOADS.load(Ordering::Relaxed)
    };
    n >= max || split_dead().is_some_and(|n| DEAD_RELOADS.load(Ordering::Relaxed) >= n)
}

/// INDUCTOR_FASTLANE_SPLIT_DEAD=<n>: the card split also stops after n
/// reloads in a row whose lane overflowed before it gave a single answer:
/// the live lemmas alone no longer fit it, and each reload resends them all.
fn split_dead() -> Option<u64> {
    static N: OnceLock<Option<u64>> = OnceLock::new();
    *N.get_or_init(|| std::env::var("INDUCTOR_FASTLANE_SPLIT_DEAD").ok().and_then(|v| v.parse().ok()))
}
static DEAD_RELOADS: AtomicU64 = AtomicU64::new(0);
static ANSWERED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The card answered a split query (SAT or UNSAT).
pub fn note_card_answer() {
    ANSWERED.store(true, Ordering::Relaxed);
    DEAD_RELOADS.store(0, Ordering::Relaxed);
}

/// The lane overflowed (an UNKNOWN answer): a reload follows.
pub fn note_card_overflow() {
    if !reload_pending() && !ANSWERED.swap(false, Ordering::Relaxed) {
        DEAD_RELOADS.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn push_split_cpu() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("INDUCTOR_FASTLANE_PUSH_SPLIT_CPU").is_ok_and(|v| v == "1"))
}

pub static SPLIT_BATCHES: AtomicU64 = AtomicU64::new(0);
pub static SPLIT_CARD_QUERIES: AtomicU64 = AtomicU64::new(0);
pub static SPLIT_CARD_UNSAT: AtomicU64 = AtomicU64::new(0);
pub static SPLIT_CARD_SAT: AtomicU64 = AtomicU64::new(0);
pub static SPLIT_CARD_FALLBACK: AtomicU64 = AtomicU64::new(0);
pub static SPLIT_CPU_CHECKS: AtomicU64 = AtomicU64::new(0);
pub static SPLIT_WAIT_NS: AtomicU64 = AtomicU64::new(0);
/// Kernel time of the push batches (with the clause records they carried).
pub static SPLIT_KERNEL_NS: AtomicU64 = AtomicU64::new(0);
// Cost model (ns): CPU push check and card query, running means.
static SPLIT_CPU_NS: AtomicU64 = AtomicU64::new(30_000);
static SPLIT_CARD_NS: AtomicU64 = AtomicU64::new(100_000);
/// Batch size cap: results (with SAT models) must fit the card's output
/// buffer; lowered when a batch comes back short, raised slowly otherwise.
static SPLIT_CAP: AtomicU64 = AtomicU64::new(256);
/// Card run overhead (ns) beyond the queries: start, kernel entry and exit,
/// result transfer; and the CPU's own share of it (command and result syncs).
const SPLIT_CARD_FIXED_NS: u64 = 40_000;
const SPLIT_CPU_FIXED_NS: u64 = 40_000;

fn ema(a: &AtomicU64, x: u64) {
    let old = a.load(Ordering::Relaxed);
    a.store(old - old / 8 + x / 8, Ordering::Relaxed);
}

/// A CPU push check took `ns`.
pub fn note_cpu_check(ns: u64) {
    ema(&SPLIT_CPU_NS, ns);
}

/// Collecting a batch the card has finished (wait, output sync, parse).
const SPLIT_COLLECT_NS: u64 = 60_000;

/// The CPU waited `wait_ns` for a batch of `k` queries that took `run_ns`
/// from its start to its collected results (the run also covers the clause
/// records queued since the last run).  A wait beyond the collect cost means
/// the card was the bottleneck: the run's time per query is then a sample of
/// its cost.  Otherwise the card finished early, and that time per query is
/// an upper bound: the estimate drops to it, or by 1/8, giving the card a
/// larger part next time.
pub fn note_card_wait(wait_ns: u64, run_ns: u64, k: u64) {
    let c = SPLIT_CARD_NS.load(Ordering::Relaxed);
    let s = run_ns.saturating_sub(SPLIT_CARD_FIXED_NS) / k.max(1);
    let nc = if wait_ns > SPLIT_COLLECT_NS { (3 * c + s) / 4 } else { (c - c / 8).min(s.max(2_000)) };
    SPLIT_CARD_NS.store(nc.clamp(2_000, 50_000_000), Ordering::Relaxed);
    if let Some((r, min, deg)) = reload_slow() {
        // the runs right after a reload carry its bulk clause load: skip them
        // runs 4..12 set the lane's fresh cost (their mean); only a lane that
        // got slower than deg times that since is stale
        let n = SLOW_RUNS.fetch_add(1, Ordering::Relaxed);
        if n >= 4 {
            if n == 4 {
                SLOW_NS.store(s, Ordering::Relaxed);
            } else {
                ema(&SLOW_NS, s);
            }
        }
        if (4..12).contains(&n) {
            SLOW_BASE_NS.fetch_add(s / 8, Ordering::Relaxed);
        }
        let slow = SLOW_NS.load(Ordering::Relaxed) as f64;
        if n >= 12
            && slow > r * SPLIT_CPU_NS.load(Ordering::Relaxed) as f64
            && slow > deg * SLOW_BASE_NS.load(Ordering::Relaxed) as f64
            && MIRRORED.load(Ordering::Relaxed) >= min
            && !reload_pending()
        {
            SLOW_RELOADS.fetch_add(1, Ordering::Relaxed);
            request_reload();
        }
    }
}
/// Card runs since the last reload, and their smoothed cost per query (ns).
static SLOW_RUNS: AtomicU64 = AtomicU64::new(0);
static SLOW_NS: AtomicU64 = AtomicU64::new(0);
static SLOW_BASE_NS: AtomicU64 = AtomicU64::new(0);

/// Adaptive reload (`INDUCTOR_FASTLANE_RELOAD_SLOW=<r>`): rebuild the pool
/// when the card's cost per query, smoothed over the runs since the last
/// reload (its first 4 skipped), exceeds both r CPU checks and
/// `INDUCTOR_FASTLANE_RELOAD_SLOW_DEG` (default 2) times the cost of the
/// fresh lane (runs 4..12), and at least
/// `INDUCTOR_FASTLANE_RELOAD_SLOW_MIN` (default 1000) clauses were mirrored
/// since the last reload.  A lane clogged with stale lemma copies searches
/// far longer than the CPU; a healthy one is left alone.
fn reload_slow() -> Option<(f64, u64, f64)> {
    static R: OnceLock<Option<(f64, u64, f64)>> = OnceLock::new();
    *R.get_or_init(|| {
        let r = std::env::var("INDUCTOR_FASTLANE_RELOAD_SLOW").ok()?.parse().ok()?;
        let min = std::env::var("INDUCTOR_FASTLANE_RELOAD_SLOW_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1000);
        let deg = std::env::var("INDUCTOR_FASTLANE_RELOAD_SLOW_DEG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2.0);
        Some((r, min, deg))
    })
}
static SLOW_RELOADS: AtomicU64 = AtomicU64::new(0);

/// How many of `n` push checks the card should take (0: none), so that the
/// CPU's part plus its sync cost ends when the card's part does.
pub fn split_size(n: usize) -> usize {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // INDUCTOR_FASTLANE_SPLIT_CARD_US: initial card cost per query (the
        // control keeps it, having no card to measure)
        if let Some(us) = std::env::var("INDUCTOR_FASTLANE_SPLIT_CARD_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
        {
            SPLIT_CARD_NS.store(us * 1000, Ordering::Relaxed);
        }
    });
    let c = SPLIT_CPU_NS.load(Ordering::Relaxed).max(1);
    // INDUCTOR_FASTLANE_SPLIT_CARD_RATIO (control mode, which has no card to
    // measure): the card cost per query as a multiple of the CPU's
    static RATIO: OnceLock<Option<f64>> = OnceLock::new();
    let ratio = *RATIO.get_or_init(|| {
        std::env::var("INDUCTOR_FASTLANE_SPLIT_CARD_RATIO").ok().and_then(|v| v.parse::<f64>().ok())
    });
    let k = match ratio {
        Some(r) if push_split_cpu() => ((c as f64) * r) as u64,
        _ => SPLIT_CARD_NS.load(Ordering::Relaxed),
    }
    .max(1);
    let total = n as u64 * c;
    if total <= SPLIT_CARD_FIXED_NS + SPLIT_CPU_FIXED_NS {
        return 0;
    }
    // (n - x) c + F_cpu = F_card + x k, scaled by INDUCTOR_FASTLANE_SPLIT_FRAC
    // (default 0.75): the CPU waiting on a late card loses its whole time,
    // a card finishing early only idles
    static FRAC: OnceLock<u64> = OnceLock::new();
    let frac = *FRAC.get_or_init(|| {
        std::env::var("INDUCTOR_FASTLANE_SPLIT_FRAC")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map_or(750, |f| (f.clamp(0.0, 1.0) * 1000.0) as u64)
    });
    let x = (total + SPLIT_CPU_FIXED_NS).saturating_sub(SPLIT_CARD_FIXED_NS) / (c + k) * frac / 1000;
    let x = (x as usize).min(n).min(SPLIT_CAP.load(Ordering::Relaxed) as usize);
    // worth it only if the CPU time it frees exceeds the CPU's sync cost
    if x < 2 || x as u64 * c < 2 * SPLIT_CPU_FIXED_NS {
        0
    } else {
        x
    }
}

/// Kernel time of all card runs so far (ns).
pub fn kernel_ns() -> u64 {
    #[cfg(has_fastlane)]
    return unsafe { ffi::fl_board_kernel_ns() };
    #[allow(unreachable_code)]
    0
}

/// Wait for the submitted batch; the number of results that came back.
pub fn batch_collect() -> u32 {
    #[cfg(has_fastlane)]
    return unsafe { ffi::fl_batch_collect() };
    #[allow(unreachable_code)]
    0
}

/// One frame's push split: lemmas `at..` went to the card as `cubes`.
pub struct PushSplit {
    pub at: usize,
    pub cubes: Vec<logicrs::LitVec>,
    kernel_ns0: u64,
    /// When the card's run started (None: control mode).
    started: Option<std::time::Instant>,
    results: Option<u32>,
    /// Control mode: the CPU's answers (blocked, core) for the tail.
    pub cpu: Option<Vec<(bool, Option<logicrs::LitVec>)>>,
}

impl PushSplit {
    /// Submit the card's part (`cubes` with their raw queries) at `frame`.
    pub fn submit(at: usize, cubes: Vec<logicrs::LitVec>, queries: Vec<Vec<u32>>, frame: u32) -> Option<Self> {
        let queries: Vec<(Vec<u32>, Vec<Vec<u32>>)> =
            queries.into_iter().map(|q| (q, Vec::new())).collect();
        if !with_pool(|pool| pool.replicas[0].submit_batch(&queries, frame)).unwrap_or(false) {
            return None;
        }
        // after the submit, which first ends any earlier run
        let kernel_ns0 = kernel_ns();
        let started = Some(std::time::Instant::now());
        SPLIT_BATCHES.fetch_add(1, Ordering::Relaxed);
        SPLIT_CARD_QUERIES.fetch_add(queries.len() as u64, Ordering::Relaxed);
        Some(Self {
            at,
            cubes,
            kernel_ns0,
            started,
            results: None,
            cpu: None,
        })
    }

    /// Control mode: the tail answered by the CPU at the split.
    pub fn cpu_answered(at: usize, answers: Vec<(bool, Option<logicrs::LitVec>)>) -> Self {
        SPLIT_BATCHES.fetch_add(1, Ordering::Relaxed);
        SPLIT_CARD_QUERIES.fetch_add(answers.len() as u64, Ordering::Relaxed);
        Self {
            at,
            cubes: Vec::new(),
            kernel_ns0: 0,
            started: None,
            results: Some(answers.len() as u32),
            cpu: Some(answers),
        }
    }

    /// Wait for the card (once); the number of results it returned.
    pub fn finish(&mut self) -> u32 {
        if let Some(n) = self.results {
            return n;
        }
        // the control's batch costs what a CPU check costs: note_cpu_check
        let t0 = std::time::Instant::now();
        let n = batch_collect();
        let wait = t0.elapsed().as_nanos() as u64;
        SPLIT_WAIT_NS.fetch_add(wait, Ordering::Relaxed);
        let k = self.cubes.len() as u64;
        let run = self.started.map_or(wait, |t| t.elapsed().as_nanos() as u64);
        note_card_wait(wait, run, k);
        SPLIT_KERNEL_NS.fetch_add(kernel_ns() - self.kernel_ns0, Ordering::Relaxed);
        let cap = SPLIT_CAP.load(Ordering::Relaxed);
        if split_lock().is_some() {
        } else if (n as u64) < k {
            SPLIT_CAP.store((n as u64 * 3 / 4).max(2), Ordering::Relaxed);
        } else if k >= cap {
            SPLIT_CAP.store((cap + cap / 8 + 1).min(1024), Ordering::Relaxed);
        }
        self.results = Some(n);
        n
    }

    /// The card's answer for frame lemma `index` (>= `at`).
    pub fn take(&mut self, index: usize) -> LaneResult {
        let i = index - self.at;
        if i as u32 >= self.finish() {
            return LaneResult::Unknown;
        }
        with_pool(|pool| pool.replicas[0].batch_take(i as u32)).unwrap_or(LaneResult::Unknown)
    }

    /// The learnts the card exported with the last `take`, valid at query
    /// frame `frame`.
    pub fn learnts(&self, frame: u32) -> Vec<logicrs::LitVec> {
        let w = with_pool(|pool| pool.replicas[0].learnts()).unwrap_or_default();
        let mut out = Vec::new();
        let mut at = 0;
        while at + 3 <= w.len() {
            let (n, lo, hi) = (w[at] as usize, w[at + 1], w[at + 2]);
            if at + 3 + n > w.len() {
                break;
            }
            if lo <= frame && frame <= hi {
                out.push(
                    w[at + 3..at + 3 + n]
                        .iter()
                        .map(|&r| logicrs::Lit::new(logicrs::Var::from(r >> 1), r & 1 == 0))
                        .collect(),
                );
            }
            at += 3 + n;
        }
        out
    }
}

impl Drop for PushSplit {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Set while a CPU search races a submitted card SOLVE.
pub static RACING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `INDUCTOR_FASTLANE_ABORT=1`: when the CPU answers a raced query first,
/// tell the card to stop it (frees the card for the next race).
pub fn abort_enabled() -> bool {
    static A: OnceLock<bool> = OnceLock::new();
    *A.get_or_init(|| std::env::var("INDUCTOR_FASTLANE_ABORT").is_ok_and(|v| v == "1"))
}

/// Stop the raced SOLVE the CPU answered first (no-op unless enabled).
pub fn card_abort() {
    #[cfg(has_fastlane)]
    if abort_enabled() {
        unsafe { ffi::fl_board_abort() };
    }
}

/// Is the card still busy with a raced SOLVE the CPU already answered?
pub fn card_busy() -> bool {
    #[cfg(has_fastlane)]
    return unsafe { ffi::fl_board_busy() } != 0;
    #[allow(unreachable_code)]
    false
}

/// Time-triggered race (`INDUCTOR_FASTLANE_RACE_US=<T>`): every routed query
/// starts on the CPU; once it has run T us (and the card is free) the search
/// itself submits it to the card and keeps going, and the first answer wins.
/// On HWMCC designs the queries the card wins are slow on the CPU but have
/// few conflicts, which a conflict budget never sends.
pub fn race_us() -> Option<u64> {
    static T: OnceLock<Option<u64>> = OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("INDUCTOR_FASTLANE_RACE_US")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}

#[inline]
fn tsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return core::arch::x86_64::_rdtsc();
    }
    #[allow(unreachable_code)]
    {
        static T0: OnceLock<std::time::Instant> = OnceLock::new();
        T0.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
    }
}

/// Timestamp-counter ticks per microsecond (measured once).
fn tsc_per_us() -> u64 {
    static R: OnceLock<u64> = OnceLock::new();
    *R.get_or_init(|| {
        let (t0, c0) = (std::time::Instant::now(), tsc());
        while t0.elapsed() < std::time::Duration::from_millis(5) {}
        ((tsc() - c0) as f64 / t0.elapsed().as_secs_f64() / 1e6).max(1.0) as u64
    })
}

struct Armed {
    assump: Vec<u32>,
    temps: Vec<Vec<u32>>,
    frame: u32,
    deadline: u64,
}

thread_local! {
    static ARMED: std::cell::RefCell<Option<Armed>> = const { std::cell::RefCell::new(None) };
}
static ARMED_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static TIMED_SUBMITS: AtomicU64 = AtomicU64::new(0);

/// Arm the time trigger for the query about to run on the CPU.
pub fn race_arm(assump: Vec<u32>, temps: Vec<Vec<u32>>, frame: u32, us: u64) {
    let deadline = tsc() + us * tsc_per_us();
    ARMED.with(|a| {
        *a.borrow_mut() = Some(Armed {
            assump,
            temps,
            frame,
            deadline,
        })
    });
    ARMED_ON.store(true, Ordering::Relaxed);
}

/// Disarm after the CPU search returned; true if the query was submitted.
pub fn race_disarm() -> bool {
    ARMED_ON.store(false, Ordering::Relaxed);
    ARMED.with(|a| *a.borrow_mut() = None);
    RACING.swap(false, Ordering::Relaxed)
}

/// Called by the CPU search at every conflict and decision: submits an
/// armed query once its deadline passes, and says whether the card has
/// answered a submitted one (the search then stops).
#[inline]
pub fn race_tick() -> bool {
    if RACING.load(Ordering::Relaxed) {
        return race_stop();
    }
    if ARMED_ON.load(Ordering::Relaxed) {
        race_tick_armed();
    }
    false
}

#[cold]
fn race_tick_armed() {
    let due = ARMED.with(|a| a.borrow().as_ref().is_some_and(|a| tsc() >= a.deadline));
    if !due || card_busy() {
        return;
    }
    let Some(armed) = ARMED.with(|a| a.borrow_mut().take()) else {
        return;
    };
    ARMED_ON.store(false, Ordering::Relaxed);
    let ok =
        with_pool(|pool| pool.replicas[0].submit_frame(&armed.assump, &armed.temps, armed.frame))
            .unwrap_or(false);
    if ok {
        TIMED_SUBMITS.fetch_add(1, Ordering::Relaxed);
        RACING.store(true, Ordering::Relaxed);
    }
}

/// Should the racing CPU search stop (the card has answered)?
#[inline]
pub fn race_stop() -> bool {
    #[cfg(has_fastlane)]
    if RACING.load(Ordering::Relaxed) {
        return unsafe { ffi::fl_board_poll() } != 0;
    }
    false
}

pub fn pmic_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pool_mode() && std::env::var_os("INDUCTOR_FASTLANE_PMIC").is_some())
}

// Virtual card clock: modeled cycles (x1000, integer) and host interactions.
pub static CARD_MILLICYCLES: AtomicU64 = AtomicU64::new(0);
pub static INTERACTIONS: AtomicU64 = AtomicU64::new(0);
pub static PMIC_CALLS: AtomicU64 = AtomicU64::new(0);
pub static PMIC_ROUNDS: AtomicU64 = AtomicU64::new(0);
pub static PMIC_CHAINS: AtomicU64 = AtomicU64::new(0);
pub static PMIC_WASTED: AtomicU64 = AtomicU64::new(0);
pub static PMIC_QUERIES: AtomicU64 = AtomicU64::new(0);

/// WIDE microarchitecture cycle model over an event delta (same weights as
/// scripts/fastlane_cycle_model.py and fastlane_pmic.cpp).
pub fn wide_cycles(before: &[u64; 32], after: &[u64; 32]) -> f64 {
    let d = |i: usize| (after[i] - before[i]) as f64;
    76.0 * d(0)
        + d(3)
        + 3.0 * d(4)
        + 0.25 * d(5)
        + 0.25 * d(27)
        + 3.0 * d(29)
        + d(31)
        + d(10)
        + d(12)
        + 2.0 * d(11)
        + d(14)
        + d(15)
        + d(16)
        + d(17)
        + d(20)
        + 0.125 * d(21)
}

pub fn note_card(cycles: f64) {
    CARD_MILLICYCLES.fetch_add((cycles * 1000.0) as u64, Ordering::Relaxed);
    INTERACTIONS.fetch_add(1, Ordering::Relaxed);
}

/// Run one MIC on all replicas; returns the generalized cube.
#[allow(unused_variables)]
/// Result of rIC3 push_lemma run on the card right after PMIC.
pub struct FusedPush {
    /// The PMIC result the push started from.
    pub input: Vec<u32>,
    /// First failing frame (level + 1 if none).
    pub frame: usize,
    /// False if a query ended UNKNOWN: continue push_lemma from `frame`.
    pub complete: bool,
    pub cube: Vec<u32>,
}

/// Fuse push_lemma into the PMIC call (`INDUCTOR_FASTLANE_FUSE_PUSH=1`).
pub fn fuse_push_enabled() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("INDUCTOR_FASTLANE_FUSE_PUSH").is_ok_and(|v| v == "1"))
}

pub static FUSED_PUSHES: AtomicU64 = AtomicU64::new(0);
pub static FUSED_PUSH_USED: AtomicU64 = AtomicU64::new(0);

/// PMIC, then (push = Some((from, to))) push_lemma of its result on the card.
pub fn pool_pmic_push(
    frame: u32,
    cube: &[u32],
    constraint: &[Vec<u32>],
    act_key: &[u64],
    next_of_var: &[u32],
    init_of_var: &[u8],
    push: Option<(u32, u32)>,
) -> Option<(Vec<u32>, Option<FusedPush>)> {
    #[cfg(has_fastlane)]
    {
        return POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            let pool = pool.as_mut()?;
            let mut ptrs: Vec<*mut ffi::FlLane> = pool.replicas.iter().map(|l| l.ptr).collect();
            for p in ptrs.iter() {
                unsafe { ffi::fl_set_ts(*p, next_of_var.as_ptr(), init_of_var.as_ptr()) };
            }
            let lens: Vec<u32> = constraint.iter().map(|c| c.len() as u32).collect();
            let flat: Vec<u32> = constraint.iter().flatten().copied().collect();
            let mut out = vec![0u32; cube.len()];
            let mut pcube = vec![0u32; cube.len()];
            let (mut cycles, mut rounds, mut chains, mut wasted, mut queries) =
                (0f64, 0u64, 0u64, 0u64, 0u64);
            let (mut pframe, mut pcomplete, mut pn) = (0u32, 0i32, 0u32);
            let (push_on, from, to) = match push {
                Some((f, t)) => (1, f, t),
                None => (0, 0, 0),
            };
            let started = std::time::Instant::now();
            let n = unsafe {
                ffi::fl_pmic_push(
                    ptrs.as_mut_ptr(),
                    ptrs.len() as u32,
                    frame,
                    cube.as_ptr(),
                    cube.len() as u32,
                    flat.as_ptr(),
                    lens.as_ptr(),
                    lens.len() as u32,
                    act_key.as_ptr(),
                    out.as_mut_ptr(),
                    &mut cycles,
                    &mut rounds,
                    &mut chains,
                    &mut wasted,
                    &mut queries,
                    push_on,
                    from,
                    to,
                    &mut pframe,
                    &mut pcomplete,
                    pcube.as_mut_ptr(),
                    &mut pn,
                )
            };
            LANE_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            note_card(cycles);
            PMIC_CALLS.fetch_add(1, Ordering::Relaxed);
            PMIC_ROUNDS.fetch_add(rounds, Ordering::Relaxed);
            PMIC_CHAINS.fetch_add(chains, Ordering::Relaxed);
            PMIC_WASTED.fetch_add(wasted, Ordering::Relaxed);
            PMIC_QUERIES.fetch_add(queries, Ordering::Relaxed);
            out.truncate(n as usize);
            let fused = (push_on != 0 && pframe != 0).then(|| {
                FUSED_PUSHES.fetch_add(1, Ordering::Relaxed);
                pcube.truncate(pn as usize);
                FusedPush {
                    input: out.clone(),
                    frame: pframe as usize,
                    complete: pcomplete != 0,
                    cube: pcube,
                }
            });
            Some((out, fused))
        });
    }
    #[allow(unreachable_code)]
    None
}

pub fn pool_pmic(
    frame: u32,
    cube: &[u32],
    constraint: &[Vec<u32>],
    act_key: &[u64],
    next_of_var: &[u32],
    init_of_var: &[u8],
) -> Option<Vec<u32>> {
    #[cfg(has_fastlane)]
    {
        return POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            let pool = pool.as_mut()?;
            let mut ptrs: Vec<*mut ffi::FlLane> = pool.replicas.iter().map(|l| l.ptr).collect();
            for p in ptrs.iter() {
                unsafe { ffi::fl_set_ts(*p, next_of_var.as_ptr(), init_of_var.as_ptr()) };
            }
            let lens: Vec<u32> = constraint.iter().map(|c| c.len() as u32).collect();
            let flat: Vec<u32> = constraint.iter().flatten().copied().collect();
            let mut out = vec![0u32; cube.len()];
            let (mut cycles, mut rounds, mut chains, mut wasted, mut queries) =
                (0f64, 0u64, 0u64, 0u64, 0u64);
            let started = std::time::Instant::now();
            let n = unsafe {
                ffi::fl_pmic(
                    ptrs.as_mut_ptr(),
                    ptrs.len() as u32,
                    frame,
                    cube.as_ptr(),
                    cube.len() as u32,
                    flat.as_ptr(),
                    lens.as_ptr(),
                    lens.len() as u32,
                    act_key.as_ptr(),
                    out.as_mut_ptr(),
                    &mut cycles,
                    &mut rounds,
                    &mut chains,
                    &mut wasted,
                    &mut queries,
                )
            };
            LANE_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            note_card(cycles);
            PMIC_CALLS.fetch_add(1, Ordering::Relaxed);
            PMIC_ROUNDS.fetch_add(rounds, Ordering::Relaxed);
            PMIC_CHAINS.fetch_add(chains, Ordering::Relaxed);
            PMIC_WASTED.fetch_add(wasted, Ordering::Relaxed);
            PMIC_QUERIES.fetch_add(queries, Ordering::Relaxed);
            out.truncate(n as usize);
            Some(out)
        });
    }
    #[allow(unreachable_code)]
    None
}

thread_local! {
    static POOL: std::cell::RefCell<Option<Pool>> = const { std::cell::RefCell::new(None) };
}

/// Propagate-share gate (`INDUCTOR_FASTLANE_PROP_GATE=<share>`): the card only
/// pays off when push checks are a large part of the run, so FastLane stays
/// inert (no mirroring, no split, card not opened; the run is plain rIC3)
/// until, at the end of a push phase at least `INDUCTOR_FASTLANE_PROP_GATE_S`
/// seconds (default 3) into the run, the propagate time so far is at least
/// that share of the elapsed time.  Then the card is opened and the pool
/// rebuilt from the live frames.  Re-checked after every push phase, so a run
/// whose push share grows later still opens.
fn prop_gate() -> Option<(f64, f64)> {
    static G: OnceLock<Option<(f64, f64)>> = OnceLock::new();
    *G.get_or_init(|| {
        let share = std::env::var("INDUCTOR_FASTLANE_PROP_GATE").ok()?.parse().ok()?;
        let after = std::env::var("INDUCTOR_FASTLANE_PROP_GATE_S")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3.0);
        Some((share, after))
    })
}
static GATE_T0: OnceLock<std::time::Instant> = OnceLock::new();
/// 0: no gate or open, 1: closed.
static GATE_CLOSED: AtomicU8 = AtomicU8::new(0);
pub static GATE_OPENED_AT_MS: AtomicU64 = AtomicU64::new(0);
pub static GATE_PHASES: AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn gate_open() -> bool {
    GATE_CLOSED.load(Ordering::Relaxed) == 0
}

fn gate_start() {
    GATE_T0.get_or_init(std::time::Instant::now);
    if prop_gate().is_some() {
        GATE_CLOSED.store(1, Ordering::Relaxed);
    }
}

/// End of a push phase; `prop` is the propagate time of the run so far.
pub fn gate_note(prop: std::time::Duration) {
    if gate_open() {
        return;
    }
    let Some((share, after)) = prop_gate() else {
        return;
    };
    let Some(t0) = GATE_T0.get() else {
        return;
    };
    let phases = GATE_PHASES.fetch_add(1, Ordering::Relaxed) + 1;
    let el = t0.elapsed().as_secs_f64();
    // INDUCTOR_FASTLANE_PROP_GATE_PHASE=<n>: open at the end of push phase n
    // instead (a gate decision recorded from an earlier run, `closed_phases`),
    // so that runs compared in lock mode open at the same point
    static AT: OnceLock<Option<u64>> = OnceLock::new();
    let at = *AT.get_or_init(|| std::env::var("INDUCTOR_FASTLANE_PROP_GATE_PHASE").ok().and_then(|v| v.parse().ok()));
    let open = match at {
        Some(n) => phases >= n,
        None => el >= after && prop.as_secs_f64() >= share * el,
    };
    if !open {
        return;
    }
    GATE_CLOSED.store(0, Ordering::Relaxed);
    GATE_OPENED_AT_MS.store((el * 1e3) as u64, Ordering::Relaxed);
    #[cfg(has_fastlane)]
    unsafe {
        ffi::fl_board_go()
    }
    request_reload();
}

/// Create the pool from the first frame solver: transition (full range),
/// plus every clause already added to it (constraints) as full range with
/// their variables in the persistent domain.
pub fn pool_init(
    n_var: u32,
    dep_offsets: &[u32],
    deps: &[u32],
    trans: &[Vec<u32>],
    base: &[Vec<u32>],
) {
    gate_start();
    if !pool_mode() {
        return;
    }
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.is_some() {
            return;
        }
        let init = PoolInit {
            n_var,
            dep_offsets: dep_offsets.to_vec(),
            deps: deps.to_vec(),
            trans: trans.to_vec(),
            base: base.to_vec(),
        };
        let Some(replicas) = init.build() else {
            return;
        };
        *pool = Some(Pool {
            replicas,
            n_var,
            init,
        });
    });
}

/// Frame band (`INDUCTOR_FASTLANE_BAND=<b>`): every reload keeps only the
/// lemmas of frames >= `BAND_LO` = frontier - b, clauses ending below it are
/// not mirrored, and only queries at frames >= `BAND_LO` are routed.  Exact:
/// a clause ranged `lo..=hi` with `hi < BAND_LO` is inactive for every routed
/// query, it only costs the card watch visits (GipSAT's per-frame solvers
/// never hold it).
pub fn band() -> Option<u32> {
    static B: OnceLock<Option<u32>> = OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("INDUCTOR_FASTLANE_BAND")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}
pub static BAND_LO: AtomicU32 = AtomicU32::new(0);
pub fn in_band(frame: Option<u32>) -> bool {
    frame.is_some_and(|f| f >= BAND_LO.load(Ordering::Relaxed))
}
/// A new frontier frame: reload once the band is more than one frame behind.
pub fn note_frontier(frontier: u32) {
    MAX_FRAME.fetch_max(frontier, Ordering::Relaxed);
    if let Some(b) = band()
        && pool_mode()
        && frontier.saturating_sub(BAND_LO.load(Ordering::Relaxed)) > b + 1
    {
        request_reload();
    }
}

/// Proactive reload (`INDUCTOR_FASTLANE_RELOAD_CLAUSES=<n>`): rebuild the
/// pool once n clauses were mirrored since it was (re)loaded.  The card lane
/// otherwise fills with stale lemma copies and compacts over and over near
/// its capacity before it finally overflows.
fn reload_clauses() -> Option<u64> {
    static N: OnceLock<Option<u64>> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("INDUCTOR_FASTLANE_RELOAD_CLAUSES")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}
static MIRRORED: AtomicU64 = AtomicU64::new(0);

/// Relative reload (`INDUCTOR_FASTLANE_RELOAD_REL=<b>`): rebuild the pool once
/// the clauses mirrored since the last reload reach b times the live lemmas
/// that reload loaded (at least 1000), i.e. once stale copies may outnumber
/// the live ones, and the CPU spent at least `INDUCTOR_FASTLANE_RELOAD_REL_WAIT`
/// (default 0.1) of the time since waiting on the card: a lane that keeps up
/// gains nothing from a reload, which resends every live lemma.
fn reload_rel() -> Option<(f64, f64)> {
    static B: OnceLock<Option<(f64, f64)>> = OnceLock::new();
    *B.get_or_init(|| {
        let b = std::env::var("INDUCTOR_FASTLANE_RELOAD_REL").ok()?.parse().ok()?;
        let w = std::env::var("INDUCTOR_FASTLANE_RELOAD_REL_WAIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.1);
        Some((b, w))
    })
}
fn reload_clock() -> &'static std::time::Instant {
    static T: OnceLock<std::time::Instant> = OnceLock::new();
    T.get_or_init(std::time::Instant::now)
}
/// When the last reload happened (ns on reload_clock) and the CPU's total
/// wait on the card then.
static RELOAD_AT_NS: AtomicU64 = AtomicU64::new(0);
static WAIT_AT_RELOAD_NS: AtomicU64 = AtomicU64::new(0);
/// Live lemmas loaded by the last reload, and their sum over all reloads.
static LOADED: AtomicU64 = AtomicU64::new(0);
static LOADED_SUM: AtomicU64 = AtomicU64::new(0);
static REL_RELOADS: AtomicU64 = AtomicU64::new(0);

/// Mirror one frame-ranged clause (IC3 solver indices lo..=hi).
pub fn pool_register(lits: &[u32], lo: u32, hi: u32) {
    if !pool_mode() || !gate_open() || hi < BAND_LO.load(Ordering::Relaxed) {
        return;
    }
    let mirrored = MIRRORED.fetch_add(1, Ordering::Relaxed) + 1;
    if let Some(n) = reload_clauses()
        && mirrored >= n
    {
        request_reload();
    }
    if let Some((b, w)) = reload_rel()
        && mirrored as f64 >= b * LOADED.load(Ordering::Relaxed).max(1000) as f64
        && !reload_pending()
        && {
            let since = (reload_clock().elapsed().as_nanos() as u64).saturating_sub(RELOAD_AT_NS.load(Ordering::Relaxed));
            (SPLIT_WAIT_NS.load(Ordering::Relaxed) - WAIT_AT_RELOAD_NS.load(Ordering::Relaxed)) as f64 >= w * since as f64
        }
    {
        REL_RELOADS.fetch_add(1, Ordering::Relaxed);
        request_reload();
    }
    POOL.with(|pool| {
        if let Some(pool) = pool.borrow_mut().as_mut() {
            for lane in pool.replicas.iter_mut() {
                lane.add_clause_ranged(lits, lo, hi);
            }
        }
    });
}

pub fn with_pool<R>(f: impl FnOnce(&mut Pool) -> R) -> Option<R> {
    POOL.with(|pool| pool.borrow_mut().as_mut().map(f))
}

pub fn global_events() -> [u64; 32] {
    #[allow(unused_mut)]
    let mut out = [0u64; 32];
    #[cfg(has_fastlane)]
    unsafe {
        ffi::fl_global_events(out.as_mut_ptr())
    }
    out
}

fn report_split() {
    let ratio = |k: usize| {
        CORE_LITS[k].load(Ordering::Relaxed) as f64 / CORE_LEMMA_LITS[k].load(Ordering::Relaxed).max(1) as f64
    };
    eprintln!("fastlane-split-lock: clone_ms={:.3} imported={}", SPLIT_CLONE_NS.load(Ordering::Relaxed) as f64 / 1e6, SPLIT_IMPORTED.load(Ordering::Relaxed));
    if x_check() {
        eprintln!(
            "fastlane-x-check: implied={} not_implied={} unknown={}",
            X_CHECK_OK.load(Ordering::Relaxed),
            X_CHECK_BAD.load(Ordering::Relaxed),
            X_CHECK_UNK.load(Ordering::Relaxed)
        );
    }
    if prop_gate().is_some() {
        eprintln!(
            "fastlane-gate: open={} opened_at_ms={} closed_phases={}",
            gate_open() as u8,
            GATE_OPENED_AT_MS.load(Ordering::Relaxed),
            GATE_PHASES.load(Ordering::Relaxed)
        );
    }
    eprintln!(
        "fastlane-split-core: cpu_pushes={} cpu_core_ratio={:.3} card_pushes={} card_core_ratio={:.3}",
        CORE_N[0].load(Ordering::Relaxed),
        ratio(0),
        CORE_N[1].load(Ordering::Relaxed),
        ratio(1)
    );
    eprintln!(
        "fastlane-split: batches={} card_queries={} card_unsat={} card_sat={} card_fallback={} cpu_checks={} wait_ms={:.3} cpu_us={:.1} card_us={:.1} batch_kernel_ms={:.3}",
        SPLIT_BATCHES.load(Ordering::Relaxed),
        SPLIT_CARD_QUERIES.load(Ordering::Relaxed),
        SPLIT_CARD_UNSAT.load(Ordering::Relaxed),
        SPLIT_CARD_SAT.load(Ordering::Relaxed),
        SPLIT_CARD_FALLBACK.load(Ordering::Relaxed),
        SPLIT_CPU_CHECKS.load(Ordering::Relaxed),
        SPLIT_WAIT_NS.load(Ordering::Relaxed) as f64 / 1e6,
        SPLIT_CPU_NS.load(Ordering::Relaxed) as f64 / 1e3,
        SPLIT_CARD_NS.load(Ordering::Relaxed) as f64 / 1e3,
        SPLIT_KERNEL_NS.load(Ordering::Relaxed) as f64 / 1e6
    );
}

pub fn report() {
    // the card bring-up thread must not outlive this point (see fl_board_shutdown)
    #[cfg(has_fastlane)]
    unsafe {
        ffi::fl_board_shutdown()
    };
    if !enabled() {
        // the CPU-split control runs without the card
        if push_split_cpu() {
            report_split();
        }
        return;
    }
    let events = global_events();
    let fields: Vec<String> = EVENT_NAMES
        .iter()
        .zip(events.iter())
        .map(|(name, value)| format!("{name}={value}"))
        .collect();
    eprintln!(
        "fastlane: routed={} sat={} unsat={} unknown={} install_failed={} phases(gen/block/push/inf/other)={}/{}/{}/{}/{}",
        ROUTED.load(Ordering::Relaxed),
        ROUTED_SAT.load(Ordering::Relaxed),
        ROUTED_UNSAT.load(Ordering::Relaxed),
        ROUTED_UNKNOWN.load(Ordering::Relaxed),
        INSTALL_FAILED.load(Ordering::Relaxed),
        PHASE_COUNTS[0].load(Ordering::Relaxed),
        PHASE_COUNTS[1].load(Ordering::Relaxed),
        PHASE_COUNTS[2].load(Ordering::Relaxed),
        PHASE_COUNTS[5].load(Ordering::Relaxed),
        PHASE_COUNTS[6].load(Ordering::Relaxed),
    );
    eprintln!("fastlane-events: {}", fields.join(" "));
    flush_log();
    eprintln!(
        "fastlane-vt: card_cycles={:.0} interactions={} pmic_calls={} pmic_rounds={} pmic_chains={} pmic_wasted={} pmic_queries={}",
        CARD_MILLICYCLES.load(Ordering::Relaxed) as f64 / 1000.0,
        INTERACTIONS.load(Ordering::Relaxed),
        PMIC_CALLS.load(Ordering::Relaxed),
        PMIC_ROUNDS.load(Ordering::Relaxed),
        PMIC_CHAINS.load(Ordering::Relaxed),
        PMIC_WASTED.load(Ordering::Relaxed),
        PMIC_QUERIES.load(Ordering::Relaxed)
    );
    eprintln!(
        "fastlane-race: busy_skips={} busy_cpu_ms={:.3} budget={} adapt_up={} adapt_down={} reloads={} slow_reloads={} rel_reloads={} loaded_last={} loaded_sum={} timed_submits={} dead_reloads={}",
        RACE_BUSY.load(Ordering::Relaxed),
        RACE_BUSY_NS.load(Ordering::Relaxed) as f64 / 1e6,
        cpu_first_budget().unwrap_or(0),
        ADAPT_UP.load(Ordering::Relaxed),
        ADAPT_DOWN.load(Ordering::Relaxed),
        RELOADS.load(Ordering::Relaxed),
        SLOW_RELOADS.load(Ordering::Relaxed),
        REL_RELOADS.load(Ordering::Relaxed),
        LOADED.load(Ordering::Relaxed),
        LOADED_SUM.load(Ordering::Relaxed),
        TIMED_SUBMITS.load(Ordering::Relaxed),
        DEAD_RELOADS.load(Ordering::Relaxed)
    );
    let hist = |h: &[AtomicU64; 8]| {
        h.iter()
            .map(|c| c.load(Ordering::Relaxed).to_string())
            .collect::<Vec<_>>()
            .join("/")
    };
    eprintln!(
        "fastlane-depth: max_frame={} submitted={} card_won={}",
        MAX_FRAME.load(Ordering::Relaxed),
        hist(&DEPTH_SUB),
        hist(&DEPTH_WIN)
    );
    report_split();
    eprintln!(
        "fastlane-fuse: pushes={} used={}",
        FUSED_PUSHES.load(Ordering::Relaxed),
        FUSED_PUSH_USED.load(Ordering::Relaxed)
    );
    eprintln!(
        "fastlane-time: lane_model_ms={:.3} shadow_mismatch={} shadow_unsat={} core_cpu={} core_lane={}",
        LANE_NS.load(Ordering::Relaxed) as f64 / 1e6,
        SHADOW_MISMATCH.load(Ordering::Relaxed),
        SHADOW_UNSAT.load(Ordering::Relaxed),
        SHADOW_CORE_CPU.load(Ordering::Relaxed),
        SHADOW_CORE_LANE.load(Ordering::Relaxed)
    );
}
