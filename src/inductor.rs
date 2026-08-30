//! Query instrumentation for the Inductor FPGA co-processor project.
//!
//! This module is not part of upstream rIC3. It records one row per SAT query
//! so the Inductor workload analysis can answer: what fraction of wall-clock is
//! SAT, how long is a query really, and how much of a query is fixed overhead
//! rather than search.
//!
//! # Shape of the instrumentation
//!
//! - [`QueryProbe`] lives inside `DagCnfSolver` and accumulates timings and
//!   counters for the query in flight. GipSAT knows nothing about trace files.
//! - The IC3 layer labels each query with a [`Phase`] and frame index via
//!   [`set_context`], because only the caller knows whether a relative-induction
//!   query came from generalization, blocking, or propagation.
//! - A process-global writer, enabled by the `INDUCTOR_TRACE` environment
//!   variable, serializes the records.
//!
//! Leaving `INDUCTOR_TRACE` unset disables everything behind one relaxed atomic
//! load, which is how the control run for measuring instrumentation overhead is
//! produced.

use inductor_trace::{Header, Mode, Phase, PhaseRec, QResult, QueryRec, Summary, TraceWriter};
use std::cell::Cell;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);
static WRITER: OnceLock<Mutex<TraceWriter>> = OnceLock::new();
static SUMMARY: Mutex<Option<Summary>> = Mutex::new(None);

thread_local! {
    static PHASE: Cell<u8> = const { Cell::new(Phase::Other as u8) };
    static FRAME: Cell<u16> = const { Cell::new(0) };
    /// Macro-op currently in scope, or 0 for "none".
    static OP: Cell<u32> = const { Cell::new(0) };
    /// Outermost algorithm-owned macro-op. Nested operations (notably MIC
    /// inside BLOCK) keep this id so exact replay can model one resident
    /// program without losing the finer-grained OP used by the ordinary trace.
    static ROOT_OP: Cell<u32> = const { Cell::new(0) };
    /// Per-thread inquiry counter used only while a root-level CPU admission
    /// sample is active. Keeping it thread-local prevents another IC3 worker
    /// in the same process from contaminating the sample.
    static ROOT_QUERY_COUNTING: Cell<bool> = const { Cell::new(false) };
    static ROOT_QUERY_COUNT: Cell<u64> = const { Cell::new(0) };
}

pub struct RootQueryCounter {
    start: u64,
    previous: bool,
    active: bool,
}

impl RootQueryCounter {
    pub fn start() -> Self {
        let previous = ROOT_QUERY_COUNTING.with(|enabled| enabled.replace(true));
        let start = ROOT_QUERY_COUNT.with(Cell::get);
        Self {
            start,
            previous,
            active: true,
        }
    }

    pub fn finish(mut self) -> u64 {
        let end = ROOT_QUERY_COUNT.with(Cell::get);
        ROOT_QUERY_COUNTING.with(|enabled| enabled.set(self.previous));
        self.active = false;
        end.wrapping_sub(self.start)
    }
}

impl Drop for RootQueryCounter {
    fn drop(&mut self) {
        if self.active {
            ROOT_QUERY_COUNTING.with(|enabled| enabled.set(self.previous));
        }
    }
}

#[inline(always)]
pub fn note_root_query() {
    ROOT_QUERY_COUNTING.with(|enabled| {
        if enabled.get() {
            ROOT_QUERY_COUNT.with(|count| count.set(count.get().wrapping_add(1)));
        }
    });
}

/// Allocates macro-op ids. Starts at 1 so 0 can mean "no scope".
static OP_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

/// Whether tracing is on. One relaxed load; safe to call on the query path.
#[inline(always)]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Start tracing into the directory named by `INDUCTOR_TRACE`, if set.
///
/// Static shape of the transition relation, as the hardware has to store and
/// evaluate it. Grouped into a struct because every field here sizes something
/// specific in the HLS design and the list keeps growing.
#[derive(Clone, Copy, Default)]
pub struct NetlistShape {
    pub n_var: u32,
    pub n_clause: u32,
    pub n_gate: u32,
    /// Entries in the inverted fanin lists: the resident fanout CSR's size.
    pub n_fanout_total: u64,
    /// Clauses in the largest gate. Sizes the evaluator's unroll, which sits on
    /// the propagation recurrence, so this is a logic-depth bound.
    pub max_gate_clauses: u32,
    /// Distinct variables in the largest gate. Bounds the entry's literal slots.
    pub max_gate_slots: u32,
    /// Longest clause.
    pub max_clause_len: u32,
    /// Gates that do not fit the kernel's fixed-size record.
    pub n_gate_unfit: u32,
    /// Gates bucketed by clause count: <=4, <=6, <=8, <=16, <=64, >64.
    pub gate_clause_hist: [u32; 6],
    /// The same buckets weighted by fanout degree -- how often the datapath
    /// actually has to evaluate a gate of that size.
    pub visit_clause_hist: [u64; 6],
    /// Words the resident clause pool needs.
    pub pool_words: u64,
    /// Literals across all gates.
    pub total_lits: u64,
    /// Pool size with each clause's literal run padded to a lane boundary, for
    /// lane widths 4, 8 and 16.
    pub pool_words_aligned: [u64; 3],
}

/// Whether to write the replay stream, and how many queries of it.
///
/// A full run's assumptions dwarf its statistics -- 86 M queries at ~35 literals
/// is tens of gigabytes -- so replay is opt-in and bounded. `INDUCTOR_REPLAY=N`
/// records the first N queries' assumptions; the netlist they run against comes
/// from `INDUCTOR_DUMP_NETLIST`.
static REPLAY_LIMIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REPLAY_WRITTEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn replay_limit() -> u64 {
    REPLAY_LIMIT.load(Ordering::Relaxed)
}

/// Record one query's assumptions, if replay is on and the budget is not spent.
///
/// Must be called before the query's [`QueryRec`] is written: the writer stamps
/// both with the same id, which is what lets the two streams be joined.
#[inline]
pub fn replay_assumptions(assump: &[u32], domain: &[u32]) {
    if !enabled() {
        return;
    }
    let limit = replay_limit();
    if limit == 0 {
        return;
    }
    if REPLAY_WRITTEN.fetch_add(1, Ordering::Relaxed) >= limit {
        return;
    }
    if let Some(w) = WRITER.get() {
        w.lock().unwrap().replay(assump, domain);
    }
}

/// Record one lemma into the replay stream as it is added.
///
/// Without this the stream carries assumptions and domains only, and the
/// accelerator's second BCP path -- 30-65% of the literal work by 7p -- has no
/// real data to run against. Bounded by the same replay limit as queries: a
/// full run's lemmas are large where its statistics are not.
pub fn replay_lemma(lits: &[u32]) {
    if !enabled() || replay_limit() == 0 {
        return;
    }
    // Bounded separately from queries, and far more loosely.
    //
    // Tying lemmas to the query limit truncated them to the start of a run,
    // where IC3's cubes are short: a 4,000-query bound on token_ring covered
    // the first 4% and recorded nothing longer than 8 literals, while the
    // clauses propagation actually reads there average 21.1 and on cv32e40x
    // 319.2. The interesting lemmas arrive after the queries stop being
    // recorded, and they are cheap to keep -- tens of thousands of them are a
    // few megabytes where a full query stream is tens of gigabytes.
    let limit = std::env::var("INDUCTOR_REPLAY_LEMMAS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(200_000);
    if LEMMA_WRITTEN.fetch_add(1, Ordering::Relaxed) >= limit {
        return;
    }
    if let Some(w) = WRITER.get() {
        w.lock().unwrap().replay_lemma(lits);
    }
}

/// The transition relation in the flat form `ind_accel_load_netlist` takes:
/// per gate, its variable, its clause count, then per clause a length and its
/// literals. The same encoding the dump uses, so the two cannot drift.
pub fn netlist_flat(dc: &logicrs::DagCnf) -> (u32, Vec<u32>) {
    let mut out = Vec::with_capacity(1 << 16);
    for (v, cls) in dc.iter() {
        if cls.is_empty() {
            continue;
        }
        out.push(Into::<u32>::into(v));
        out.push(cls.len() as u32);
        for c in cls.iter() {
            out.push(c.len() as u32);
            for l in c.iter() {
                out.push(Into::<u32>::into(*l));
            }
        }
    }
    (dc.num_var() as u32, out)
}

/// Snapshot the live lemma set every `INDUCTOR_SNAPSHOT` queries, default 500.
///
/// Bounded by the query replay limit, unlike individual lemmas: a snapshot is
/// only useful for queries that are themselves replayed.
pub fn replay_lemma_snapshot(clauses: &[Vec<u32>]) {
    if !enabled() || replay_limit() == 0 {
        return;
    }
    if REPLAY_WRITTEN.load(Ordering::Relaxed) >= replay_limit() {
        return;
    }
    if let Some(w) = WRITER.get() {
        w.lock().unwrap().replay_lemma_snapshot(clauses);
    }
}

/// Report the solver fan-out: IC3 keeps one solver per frame, each with its own
/// lemma set, and the replay loads a snapshot from whichever one was running.
/// If a query's propagation reads far more than that snapshot holds, this is
/// where the difference is.
pub fn report_solver_fanout(n_solver: usize, per_solver: &[(usize, u64)]) {
    if !enabled() {
        return;
    }
    let total_cls: usize = per_solver.iter().map(|(c, _)| c).sum();
    let total_lits: u64 = per_solver.iter().map(|(_, l)| l).sum();
    let mx = per_solver.iter().map(|(c, _)| *c).max().unwrap_or(0);
    eprintln!(
        "inductor: {n_solver} solvers; lemma clauses {total_cls} total, {mx} in the \
         largest, {:.1} literals each",
        if total_cls > 0 {
            total_lits as f64 / total_cls as f64
        } else {
            0.0
        }
    );
}

pub fn snapshot_every() -> u64 {
    use std::sync::OnceLock;
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("INDUCTOR_SNAPSHOT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(500)
    })
}

/// Dump the transition relation to a file, if `INDUCTOR_DUMP_NETLIST` is set.
///
/// The point is to get *real* netlists in front of the hardware packer and the
/// csim engine. Everything they have been tested against so far is generated,
/// and generated inputs are exactly how the fixed-size gate record survived a
/// thorough testbench while being wrong about half of all real gates.
///
/// Format, little-endian, deliberately trivial to read from C++:
///
///   magic "INDNET\0\x01"      8 bytes
///   u32 n_var
///   u32 n_gate                 gates that follow
///   per gate:  u32 var, u32 n_clause, then per clause: u32 len, len x u32 lit
///
/// Literals use logicrs' encoding (variable << 1 | sign), which is what the
/// kernel expects, so nothing is translated on the way in or out.
pub fn dump_netlist(dc: &logicrs::DagCnf) {
    let Ok(path) = std::env::var("INDUCTOR_DUMP_NETLIST") else {
        return;
    };
    if path.is_empty() {
        return;
    }
    let mut out: Vec<u8> = Vec::with_capacity(1 << 20);
    out.extend_from_slice(b"INDNET\0\x01");
    out.extend_from_slice(&(dc.num_var() as u32).to_le_bytes());
    let n_gate = dc.iter().filter(|(_, c)| !c.is_empty()).count() as u32;
    out.extend_from_slice(&n_gate.to_le_bytes());
    for (v, cls) in dc.iter() {
        if cls.is_empty() {
            continue;
        }
        out.extend_from_slice(&(Into::<u32>::into(v)).to_le_bytes());
        out.extend_from_slice(&(cls.len() as u32).to_le_bytes());
        for c in cls.iter() {
            out.extend_from_slice(&(c.len() as u32).to_le_bytes());
            for l in c.iter() {
                let raw: u32 = Into::<u32>::into(*l);
                out.extend_from_slice(&raw.to_le_bytes());
            }
        }
    }
    match std::fs::write(&path, &out) {
        Ok(()) => log::info!("inductor: netlist dumped to {path} ({} bytes)", out.len()),
        Err(e) => log::warn!("inductor: could not dump netlist to {path}: {e}"),
    }
}

/// Start tracing, recording the netlist shape in the header.
///
/// These feed the capacity-envelope analysis and the HLS sizing. They are
/// recorded rather than estimated because the two obvious estimates for the
/// fanout total differ by 3x, and the clause bound had been a guess since the
/// first kernel. Returns whether tracing was enabled.
pub fn init(model: &str, shape: NetlistShape) -> bool {
    let NetlistShape {
        n_var,
        n_clause,
        n_gate,
        n_fanout_total,
        max_gate_clauses,
        max_gate_slots,
        max_clause_len,
        n_gate_unfit,
        gate_clause_hist,
        visit_clause_hist,
        pool_words,
        total_lits,
        pool_words_aligned,
    } = shape;
    let Ok(dir) = std::env::var("INDUCTOR_TRACE") else {
        return false;
    };
    if dir.is_empty() {
        return false;
    }
    let timer_overhead_ns = inductor_trace::timer_overhead_ns();
    let replay_n: u64 = std::env::var("INDUCTOR_REPLAY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    REPLAY_LIMIT.store(replay_n, Ordering::Relaxed);
    let header = Header {
        version: inductor_trace::FORMAT_VERSION,
        mode: if replay_n > 0 {
            Mode::Replay
        } else {
            Mode::Stats
        },
        n_var,
        n_clause,
        n_gate,
        timer_overhead_ns,
        n_fanout_total,
        max_gate_clauses,
        max_gate_slots,
        max_clause_len,
        n_gate_unfit,
        gate_clause_hist,
        visit_clause_hist,
        pool_words,
        total_lits,
        pool_words_aligned,
    };
    match TraceWriter::create(&dir, header) {
        Ok(w) => {
            if WRITER.set(Mutex::new(w)).is_err() {
                // Already initialized (portfolio mode spawns several engines).
                return enabled();
            }
            *SUMMARY.lock().unwrap() = Some(Summary {
                model: model.to_string(),
                result: "unknown".into(),
                n_var,
                n_clause,
                n_gate,
                n_fanout_total,
                max_gate_clauses,
                max_gate_slots,
                max_clause_len,
                n_gate_unfit,
                timer_overhead_ns,
                ..Default::default()
            });
            ENABLED.store(true, Ordering::Relaxed);
            true
        }
        Err(e) => {
            log::warn!("inductor: could not open trace dir {dir}: {e}");
            false
        }
    }
}

/// Restores the enclosing query label when dropped.
///
/// IC3's phases nest -- MIC runs inside blocking, which runs inside the top
/// level -- so a plain "set" would leave the innermost label smeared over every
/// later query. Binding the guard to `_ctx` keeps it alive for the enclosing
/// scope; binding to `_` would drop it immediately and restore too early.
#[must_use = "dropping the guard immediately restores the previous label"]
pub struct PhaseGuard(u8, u16, u32, u32);

impl Drop for PhaseGuard {
    #[inline]
    fn drop(&mut self) {
        PHASE.with(|p| p.set(self.0));
        FRAME.with(|f| f.set(self.1));
        OP.with(|o| o.set(self.2));
        ROOT_OP.with(|o| o.set(self.3));
    }
}

/// Label the queries issued for as long as the returned guard lives.
#[inline]
pub fn set_context(phase: Phase, frame: usize) -> PhaseGuard {
    let prev_p = PHASE.with(|p| p.replace(phase as u8));
    let prev_f = FRAME.with(|f| f.replace(frame.min(u16::MAX as usize) as u16));
    PhaseGuard(
        prev_p,
        prev_f,
        OP.with(|o| o.get()),
        ROOT_OP.with(|o| o.get()),
    )
}

/// Open a macro-op scope: every query issued while the guard lives shares one
/// id and can therefore be shipped to the accelerator as a single transaction.
///
/// This has to be driven by the algorithm. IC3 interleaves query kinds inside a
/// single logical operation -- MIC's `down()` emits predecessor-lifting queries
/// between drop probes -- so a consumer trying to recover macro-ops by scanning
/// for consecutive equal phase+frame finds runs of length ~1.4 and concludes,
/// wrongly, that batching is impossible.
#[inline]
pub fn macro_scope(phase: Phase, frame: usize) -> PhaseGuard {
    let g = set_context(phase, frame);
    let id = OP_COUNTER.fetch_add(1, Ordering::Relaxed);
    OP.with(|o| o.set(id));
    ROOT_OP.with(|o| {
        if o.get() == 0 {
            o.set(id);
        }
    });
    g
}

/// Return the algorithm-derived trace labels currently in scope.
///
/// The exact FPGA replay writer uses this even when the ordinary binary trace
/// is disabled.  Keeping the labels next to the IC3 `macro_scope` source avoids
/// reconstructing dependencies later from adjacent query shapes.
#[inline]
pub fn current_macro_context() -> (Phase, u32) {
    let phase = PHASE.with(|p| Phase::from_u8(p.get()).unwrap_or(Phase::Other));
    let op_id = ROOT_OP.with(|root| {
        let root = root.get();
        if root != 0 {
            root
        } else {
            OP.with(|op| op.get())
        }
    });
    (phase, op_id)
}

/// Run `body` as one IC3 phase, recording its span and the queries it issued.
///
/// This is the B layer: it is what lets the analysis say how much of wall-clock
/// each phase costs and how many independent queries a phase offers, which is
/// the upper bound on what query-level parallelism can buy.
#[inline]
pub fn in_phase<R>(phase: Phase, frame: usize, body: impl FnOnce() -> R) -> R {
    // Exact FPGA replay is intentionally usable without the much larger
    // ordinary trace. Keep the algorithm labels alive in that configuration;
    // only timing/QueryRec serialization is conditional on `enabled()`.
    let _guard = set_context(phase, frame);
    if !enabled() {
        return body();
    }
    let start = now_ns();
    let qid_begin = peek_qid();
    let r = body();
    let rec = PhaseRec {
        kind_u8: phase as u8,
        t_start_ns: start,
        t_end_ns: now_ns(),
        qid_begin,
        qid_end: peek_qid(),
    };
    if let Some(w) = WRITER.get() {
        w.lock().unwrap().phase(rec);
    }
    r
}

fn now_ns() -> u64 {
    WRITER
        .get()
        .map(|w| w.lock().unwrap().now_ns())
        .unwrap_or(0)
}

fn peek_qid() -> u64 {
    WRITER
        .get()
        .map(|w| w.lock().unwrap().peek_qid())
        .unwrap_or(0)
}

/// Emit the record for one completed query.
pub fn record(probe: &QueryProbe, result: Option<bool>) {
    if !enabled() {
        return;
    }
    let Some(w) = WRITER.get() else { return };
    let rec = QueryRec {
        qid: 0, // assigned by the writer
        phase_u8: PHASE.with(|p| p.get()),
        result_u8: match result {
            Some(false) => QResult::Unsat,
            Some(true) => QResult::Sat,
            None => QResult::Unknown,
        } as u8,
        frame: FRAME.with(|f| f.get()),
        t_total_ns: probe.t_total_ns,
        t_setup_ns: probe.t_setup_ns,
        t_search_ns: probe.t_search_ns,
        t_core_ns: probe.t_core_ns,
        t_teardown_ns: probe.t_teardown_ns,
        n_assump: probe.n_assump,
        n_constraint_lits: probe.n_constraint_lits,
        domain_size: probe.domain_size,
        n_var_total: probe.n_var_total,
        n_decide: probe.n_decide,
        n_prop: probe.n_prop,
        n_conflict: probe.n_conflict,
        n_learnt: probe.n_learnt,
        n_lemma: probe.n_lemma,
        n_fanout_visit: probe.n_fanout_visit.min(u32::MAX as u64) as u32,
        n_assign: probe.n_assign,
        bcp_cycles: [
            probe.bcp_cycles[0].min(u32::MAX as u64) as u32,
            probe.bcp_cycles[1].min(u32::MAX as u64) as u32,
            probe.bcp_cycles[2].min(u32::MAX as u64) as u32,
            probe.bcp_cycles[3].min(u32::MAX as u64) as u32,
        ],
        // Outside any macro-op scope each query stands alone, so give it a
        // fresh id rather than letting unrelated queries share id 0.
        op_id: match OP.with(|o| o.get()) {
            0 => OP_COUNTER.fetch_add(1, Ordering::Relaxed),
            id => id,
        },
    };
    let mut g = w.lock().unwrap();
    g.query(rec);
    let elapsed = g.now_ns();
    drop(g);

    let snapshot = {
        let Ok(mut lock) = SUMMARY.lock() else { return };
        let Some(s) = lock.as_mut() else { return };
        s.n_query += 1;
        s.sum_query_ns += probe.t_total_ns as u64;
        s.max_frame = s.max_frame.max(rec.frame as u32);
        // Most benchmarks in a full sweep hit the timeout and are killed, so
        // `finish` never runs. Refresh the summary periodically: without a
        // wall-clock figure a killed run contributes queries but no denominator,
        // which silently inflates the aggregate SAT share above 100%.
        if s.n_query % SUMMARY_FLUSH_EVERY == 0 {
            s.total_wall_ns = elapsed;
            s.result = "timeout".into();
            Some(s.clone())
        } else {
            None
        }
    };
    if let Some(s) = snapshot {
        write_summary(&s);
    }
}

/// How often to refresh `summary.json` mid-run. At ~50us per query this is a
/// write every few seconds -- negligible, and it bounds how much wall-clock a
/// killed run can lose.
const SUMMARY_FLUSH_EVERY: u64 = 65_536;

fn write_summary(s: &Summary) {
    if let Ok(dir) = std::env::var("INDUCTOR_TRACE") {
        let path = inductor_trace::TraceDir::new(dir).summary_path();
        if let Err(e) = s.write(&path) {
            log::warn!("inductor: could not write {}: {e}", path.display());
        }
    }
}

/// Record IC3-level counters and close out the trace.
///
/// `mic_drop` is (successes, attempts) from `Statistic::mic_drop`; the failure
/// rate it implies is what decides how deep MIC speculation can go.
/// Process-wide BCP and search totals. A separate accumulator rather than a new
/// trace field, so the record format and its reader stay as they are.
pub static BCP_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SEARCH_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TOTAL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static N_QUERY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LEMMA_WRITTEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Setup's three parts. `t_setup_ns` is 21.5% of a query and the largest block
/// after BCP, but it is not one thing: the domain is a backward reachability
/// walk over resident structure, which is the shape an accelerator is good at,
/// while database cleanup and simplification are control-heavy, which is the
/// shape it is worst at. Deciding whether any of setup belongs on a card needs
/// them apart.
pub static DOMAIN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static DB_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SETUP_BCP_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SETUP_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Search minus BCP is the second largest block, and it is two things with
/// opposite hardware prospects: conflict analysis resolves clauses literal by
/// literal, which is the shape BCP already is, while the decision heap is
/// pointer work an accelerator's clock is thirteen times worse at.
pub static ANALYZE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static DECIDE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// BCP work split by what kind of clause it lands on.
///
/// The accelerator implements gate implication over the transition relation
/// only; D2's second path, watched literals for lemma, learnt and temporary
/// clauses, is unbuilt. Every propagation figure measured so far therefore
/// covers `T` alone, and how much that leaves out has been assumed, not
/// measured. These count watcher visits and literal reads on each side.
/// Counting by kind costs two atomics on every watcher visit, in BCP's
/// innermost loop, and that inflates BCP's own share -- 63.6% became 69.1% on
/// token_ring with the counters in. The counts are still right; the *timings*
/// taken from the same run are not. Off unless `INDUCTOR_KIND` is set, so the
/// two cannot be read from one run by accident.
pub fn kind_counting() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("INDUCTOR_KIND").is_ok())
}

/// Sampled comparison of the two indexing schemes for the second BCP path.
/// Accumulated on one query in a thousand, because the lemma-database walk it
/// needs is what doubled runtime when it was done on every query.
pub static OCC_VISITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static OCC_WATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static OCC_LITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static OCC_SAMPLES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static OCC_SAT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static OCC_RAW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static OCC_BLK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub static W_TRANS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static W_OTHER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static L_TRANS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static L_OTHER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// "Other" split three ways. 7p lumped lemma, learnt and temporary together,
/// and the accelerator's second path stores only the first: lemmas persist
/// across queries and can be shipped and indexed between them, where learnt
/// clauses are produced by conflict analysis inside a query. The recorded
/// lemmas average 2.3-3.1 literals against the 76.3-143.9 that "other" reads
/// per visit, which is the gap this measures.
pub static W_LEMMA: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static L_LEMMA: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static W_LEARNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static L_LEARNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Print the running BCP share. Called periodically and at `finish`.
pub fn report_bcp_share() {
    let b = BCP_NS.load(Ordering::Relaxed) as f64;
    let se = SEARCH_NS.load(Ordering::Relaxed) as f64;
    let t = TOTAL_NS.load(Ordering::Relaxed) as f64;
    let n = N_QUERY.load(Ordering::Relaxed);
    if t <= 0.0 {
        return;
    }
    let dom = DOMAIN_NS.load(Ordering::Relaxed) as f64;
    let db = DB_NS.load(Ordering::Relaxed) as f64;
    let sbcp = SETUP_BCP_NS.load(Ordering::Relaxed) as f64;
    let setup = SETUP_NS.load(Ordering::Relaxed) as f64;
    {
        let wt = W_TRANS.load(Ordering::Relaxed) as f64;
        let wo = W_OTHER.load(Ordering::Relaxed) as f64;
        let lt = L_TRANS.load(Ordering::Relaxed) as f64;
        let lo = L_OTHER.load(Ordering::Relaxed) as f64;
        let n = N_QUERY.load(Ordering::Relaxed).max(1) as f64;
        if wt + wo > 0.0 {
            eprintln!(
                "inductor: bcp work by clause kind -- trans {:.1}% of visits, {:.1}% of \
                 literal reads; per query {:.0} trans + {:.0} other visits, {:.0} + {:.0} reads",
                100.0 * wt / (wt + wo),
                100.0 * lt / (lt + lo).max(1.0),
                wt / n,
                wo / n,
                lt / n,
                lo / n
            );
        }
    }
    {
        let sm = OCC_SAMPLES.load(Ordering::Relaxed) as f64;
        if sm > 0.0 {
            let ov = OCC_VISITS.load(Ordering::Relaxed) as f64;
            // Watched visits per query, from the run total. OCC_WATCH held a
            // cumulative count and was being read as if it were per query,
            // which made the ratio look like 0.0x.
            let ow = W_OTHER.load(Ordering::Relaxed) as f64
                / N_QUERY.load(Ordering::Relaxed).max(1) as f64;
            let ol = OCC_LITS.load(Ordering::Relaxed) as f64;
            let sat = OCC_SAT.load(Ordering::Relaxed) as f64;
            let raw = OCC_RAW.load(Ordering::Relaxed) as f64;
            let blk = OCC_BLK.load(Ordering::Relaxed) as f64;
            eprintln!(
                "inductor: blocker on the lemma side -- {:.0}% of visited clauses already \
                 satisfied; literal reads {:.0} -> {:.0} per query, {:.1}x less",
                if ov > 0.0 { 100.0 * sat / ov } else { 0.0 },
                raw / sm,
                blk / sm,
                if blk > 0.0 { raw / blk } else { 0.0 }
            );
            eprintln!(
                "inductor: second path indexing, {sm:.0} sampled queries -- occurrence \
                 index {:.0} clause visits/query against watched {:.0}, ratio {:.1}x; \
                 lemma db {:.0} literals",
                ov / sm,
                ow,
                if ow > 0.0 { (ov / sm) / ow } else { 0.0 },
                ol / sm
            );
        }
    }
    {
        let wl = W_LEMMA.load(Ordering::Relaxed) as f64;
        let ll = L_LEMMA.load(Ordering::Relaxed) as f64;
        let wr = W_LEARNT.load(Ordering::Relaxed) as f64;
        let lr = L_LEARNT.load(Ordering::Relaxed) as f64;
        let lo = L_OTHER.load(Ordering::Relaxed) as f64;
        if lo > 0.0 {
            eprintln!(
                "inductor: non-trans literal reads -- lemma {:.1}% ({:.1} per visit), \
                 learnt {:.1}% ({:.1} per visit), temporary {:.1}%",
                100.0 * ll / lo,
                if wl > 0.0 { ll / wl } else { 0.0 },
                100.0 * lr / lo,
                if wr > 0.0 { lr / wr } else { 0.0 },
                100.0 * (lo - ll - lr).max(0.0) / lo
            );
        }
    }
    let an = ANALYZE_NS.load(Ordering::Relaxed) as f64;
    let de = DECIDE_NS.load(Ordering::Relaxed) as f64;
    eprintln!(
        "inductor: search-minus-bcp = analyze {:.1}% + decide {:.1}% of query",
        100.0 * an / t,
        100.0 * de / t
    );
    eprintln!(
        "inductor: setup {:.1}% of query = domain {:.1}% + db {:.1}% + setup-bcp {:.1}% \
         (+{:.1}% unaccounted)",
        100.0 * setup / t,
        100.0 * dom / t,
        100.0 * db / t,
        100.0 * sbcp / t,
        100.0 * (setup - dom - db - sbcp).max(0.0) / t
    );
    eprintln!(
        "inductor: {n} queries; bcp {:.1}% of query, {:.1}% of search   \
         (bcp {:.1} ms, search {:.1} ms, query {:.1} ms)",
        100.0 * b / t,
        if se > 0.0 { 100.0 * b / se } else { 0.0 },
        b / 1e6,
        se / 1e6,
        t / 1e6
    );
}

pub fn finish(
    result: &str,
    total_wall_ns: u64,
    mic_drop: (u64, u64),
    num_down: u64,
    num_down_sat: u64,
) {
    if !enabled() {
        return;
    }
    let summary = {
        let mut g = SUMMARY.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        s.result = result.to_string();
        s.total_wall_ns = total_wall_ns;
        s.mic_drop_success = mic_drop.0;
        s.mic_drop_total = mic_drop.1;
        s.num_down = num_down;
        s.num_down_sat = num_down_sat;
        s.clone()
    };
    if let Some(w) = WRITER.get() {
        let mut g = w.lock().unwrap();
        let _ = g.finish();
    }
    write_summary(&summary);
    report_bcp_share();
    ENABLED.store(false, Ordering::Relaxed);
}

/// Per-query measurements, accumulated inside `DagCnfSolver`.
///
/// Cloned along with the solver when IC3 extends to a new frame; the counters
/// are per-query and reset at the start of each solve, so a stale clone carries
/// nothing meaningful.
#[derive(Debug, Clone, Copy, Default)]
pub struct QueryProbe {
    pub t_total_ns: u32,
    pub t_setup_ns: u32,
    pub t_search_ns: u32,
    /// Time inside `propagate()`, i.e. BCP alone.
    ///
    /// `t_search_ns` lumps decide, BCP and conflict analysis together, which is
    /// enough to say search dominates but not enough to say whether a wider BCP
    /// datapath is worth building: accelerating BCP to zero is bounded by BCP's
    /// own share, and everything else in search is serial control the
    /// accelerator's clock is thirteen times worse at.
    pub t_bcp_ns: u32,
    pub t_core_ns: u32,
    pub t_teardown_ns: u32,
    pub n_assump: u32,
    pub n_constraint_lits: u32,
    pub domain_size: u32,
    pub n_var_total: u32,
    pub n_decide: u32,
    pub n_prop: u32,
    pub n_conflict: u32,
    pub n_learnt: u32,
    pub n_lemma: u32,
    /// Gate visits a gate-implication BCP path would make. Accumulated in
    /// `assign()` from the precomputed fanout degrees.
    ///
    /// u64 because a long query can plausibly exceed 4.29e9 visits; the record
    /// field is u32 and saturates, so an overflow shows up pinned at the
    /// maximum instead of wrapping to a small, believable-looking number.
    pub n_fanout_visit: u64,
    pub n_assign: u32,
    /// BCP cycles at [`LANES`] gate-evaluation lanes, with ceil() semantics.
    pub bcp_cycles: [u64; 4],
}

impl QueryProbe {
    /// Clear the per-query fields at the start of a solve.
    #[inline]
    pub fn begin(&mut self) {
        *self = QueryProbe::default();
    }
}

/// Candidate gate-evaluation lane counts for the BCP datapath.
pub const LANES: [u32; 4] = [4, 8, 16, 32];

/// Per-thread CPU time for accelerator profitability decisions.
///
/// Wall time is the correct end-to-end metric, but it is a poor routing signal
/// inside a many-process portfolio: a microsecond GipSAT query can appear to
/// take milliseconds when its worker is descheduled. Using the calling
/// thread's CPU clock keeps the FPGA crossover about SAT work rather than host
/// contention. Non-Linux targets fall back to wall time.
#[derive(Clone, Copy)]
pub struct ThreadCpuTimer {
    wall: Instant,
    cpu_ns: Option<u64>,
}

static THREAD_CPU_TIMING: AtomicBool = AtomicBool::new(false);

impl ThreadCpuTimer {
    #[inline]
    pub fn enable() {
        THREAD_CPU_TIMING.store(true, Ordering::Relaxed);
    }

    #[inline]
    pub fn start() -> Self {
        Self {
            wall: Instant::now(),
            cpu_ns: THREAD_CPU_TIMING
                .load(Ordering::Relaxed)
                .then(thread_cpu_time_ns)
                .flatten(),
        }
    }

    #[inline]
    pub fn ns(&self) -> u64 {
        self.cpu_ns
            .zip(thread_cpu_time_ns())
            .and_then(|(start, end)| end.checked_sub(start))
            .unwrap_or_else(|| self.wall.elapsed().as_nanos().min(u64::MAX as u128) as u64)
    }
}

#[cfg(target_os = "linux")]
#[inline]
fn thread_cpu_time_ns() -> Option<u64> {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let status = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) };
    if status != 0 || time.tv_sec < 0 || time.tv_nsec < 0 {
        return None;
    }
    (time.tv_sec as u64)
        .checked_mul(1_000_000_000)
        .and_then(|seconds| seconds.checked_add(time.tv_nsec as u64))
}

#[cfg(not(target_os = "linux"))]
#[inline]
fn thread_cpu_time_ns() -> Option<u64> {
    None
}

/// A timestamp that costs nothing when tracing is off.
///
/// The four time splits each need a start/stop pair, so on a short query the
/// clock reads are a measurable share of what is being measured. Skipping them
/// entirely in the control run is what makes the overhead claim checkable.
#[derive(Clone, Copy)]
pub struct Timer(Option<Instant>);

impl Timer {
    #[inline(always)]
    pub fn start() -> Self {
        // Selective core offload learns from exact CPU BCP time even when a
        // trace file is not being written. This opt-in path pays the clock
        // reads intentionally; normal runs retain their zero-cost timestamps.
        static SELECT_TIMING: OnceLock<bool> = OnceLock::new();
        let timing = enabled()
            || *SELECT_TIMING.get_or_init(|| std::env::var("INDUCTOR_CORE_SELECTIVE").is_ok());
        Timer(timing.then(Instant::now))
    }

    #[inline(always)]
    pub fn ns(&self) -> u32 {
        match self.0 {
            // Saturate rather than wrap: a query longer than 4.29s is an
            // outlier we want pinned at the maximum, not folded back to zero.
            Some(t) => t.elapsed().as_nanos().min(u32::MAX as u128) as u32,
            None => 0,
        }
    }
}
