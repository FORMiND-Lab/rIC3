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
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

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
/// subsumed or moved lemma copy implied by the live ones.
pub fn pool_reload(lemmas: &[(Vec<u32>, u32, u32)]) {
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
    MIRRORED.store(0, Ordering::Relaxed);
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

/// Proactive reload (`INDUCTOR_FASTLANE_RELOAD_CLAUSES=<n>`): rebuild the
/// pool once n clauses were mirrored since it was (re)loaded.  The card lane
/// otherwise fills with stale lemma copies and compacts over and over near
/// its capacity before it finally overflows.
fn reload_clauses() -> Option<u64> {
    static N: OnceLock<Option<u64>> = OnceLock::new();
    *N.get_or_init(|| std::env::var("INDUCTOR_FASTLANE_RELOAD_CLAUSES").ok().and_then(|v| v.parse().ok()))
}
static MIRRORED: AtomicU64 = AtomicU64::new(0);

/// Mirror one frame-ranged clause (IC3 solver indices lo..=hi).
pub fn pool_register(lits: &[u32], lo: u32, hi: u32) {
    if !pool_mode() {
        return;
    }
    if let Some(n) = reload_clauses()
        && MIRRORED.fetch_add(1, Ordering::Relaxed) + 1 >= n
    {
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

pub fn report() {
    // the card bring-up thread must not outlive this point (see fl_board_shutdown)
    #[cfg(has_fastlane)]
    unsafe {
        ffi::fl_board_shutdown()
    };
    if !enabled() {
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
        "fastlane-race: busy_skips={} busy_cpu_ms={:.3} budget={} adapt_up={} adapt_down={} reloads={} timed_submits={}",
        RACE_BUSY.load(Ordering::Relaxed),
        RACE_BUSY_NS.load(Ordering::Relaxed) as f64 / 1e6,
        cpu_first_budget().unwrap_or(0),
        ADAPT_UP.load(Ordering::Relaxed),
        ADAPT_DOWN.load(Ordering::Relaxed),
        RELOADS.load(Ordering::Relaxed),
        TIMED_SUBMITS.load(Ordering::Relaxed)
    );
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
