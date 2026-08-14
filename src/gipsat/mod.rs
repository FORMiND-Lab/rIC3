mod analyze;
mod cdb;
mod domain;
mod eq;
mod propagate;
mod query;
mod search;
mod simplify;
mod statistic;
mod ts;
mod vsids;

use crate::gipsat::eq::Eqc;
use analyze::Analyze;
pub use cdb::ClauseKind;
use cdb::{CREF_NONE, CRef, ClauseDB};
use domain::Domain;
use giputils::bitvec::BitVec;
use giputils::gvec::Gvec;
use giputils::ptr::Gptr;
use logicrs::satif::Satif;
use logicrs::{DagCnf, Lbool, VarAssign, VarRange};
use logicrs::{Lit, LitSet, LitVec, Var, VarMap};
use propagate::Watchers;
use rand::RngExt;
use rand::{SeedableRng, rngs::SmallRng};
use simplify::Simplify;
pub use statistic::SolverStatistic;
pub use query::{
    BatchDecodeError, IncrementalCdcl, IncrementalQuery, IncrementalResult, QueryBudget,
    decode_batch_results, pack_batch, solve_on_cpu_after_hardware_unknown,
    solve_with_cpu_fallback,
};
use std::iter::empty;
use std::time::Instant;
pub use ts::*;
use vsids::Vsids;

#[derive(Clone)]
pub struct DagCnfSolver {
    cdb: ClauseDB,
    watchers: Watchers,
    value: VarAssign,
    trail: Gvec<Lit>,
    pos_in_trail: Vec<u32>,
    level: VarMap<u32>,
    reason: VarMap<CRef>,
    propagated: u32,
    vsids: Vsids,
    phase_saving: VarMap<Lbool>,
    analyze: Analyze,
    simplify: Simplify,
    eqc: Eqc,
    unsat_core: LitSet,
    domain: Domain,
    temporary_domain: bool,
    prepared_vsids: bool,
    constrain_act: Var,
    dc: Gptr<DagCnf>,
    trivial_unsat: bool,
    /// Identifies this solver to the accelerator.
    ///
    /// IC3 keeps one solver per frame, each with its own lemmas, and builds
    /// them by cloning. A clone that carried this id would make one id name
    /// every frame: `is_bound` would be true everywhere and every frame's
    /// lemmas would go into the one engine on the card. That produced conflicts
    /// on satisfiable queries until each clone got a fresh id.
    pub accel_id: u64,
    /// The frame this solver belongs to. Queries pass it to the card, which
    /// holds every frame's lemmas and skips the ones this frame does not.
    pub accel_level: u32,
    /// Inductor instrumentation: per-query timings and counters.
    pub probe: crate::inductor::QueryProbe,
    /// Inductor: fanout degree per variable, i.e. how many gates a
    /// gate-implication BCP path would re-evaluate when this variable is
    /// assigned. Precomputed once so the hot path costs one lookup.
    fanout_len: VarMap<u32>,
    mark: LitSet,
    rng: SmallRng,
    pub cfg: Config,

    assump: LitVec,
    constraint: Vec<LitVec>,
    /// Exact transition CNF copied at construction. Keeping an owned copy is
    /// also important because `dc` is a non-owning pointer used by GipSAT.
    resident_trans: Vec<LitVec>,
    /// Exact permanent IC3 clauses as supplied through `Satif::add_clause`.
    /// ClauseDB may simplify a lemma into a level-zero assignment and no
    /// longer retain its original record, so the accelerator boundary keeps
    /// this semantic log instead of reconstructing clauses from the trail.
    resident_lemmas: Vec<LitVec>,

    statistic: SolverStatistic,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub phase_saving: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self { phase_saving: true }
    }
}

impl DagCnfSolver {
    /// A new identity, for a solver that must not share the card with another.
    pub fn fresh_accel_id() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(1);
        NEXT.fetch_add(1, Ordering::Relaxed)
    }

    /// Snapshot the persistent formula a full-search accelerator may retain.
    /// CPU learnts are deliberately omitted: transition clauses and IC3 frame
    /// lemmas define the query semantics, while both solvers may learn their
    /// own redundant clauses independently. Temporary constraints stay in the
    /// per-query payload.
    pub fn incremental_resident_snapshot(&self) -> (u32, u32, Vec<LitVec>) {
        let (n_var, frame, mut trans, lemmas) = self.incremental_resident_partition();
        trans.extend(lemmas);
        (n_var, frame, trans)
    }

    /// Split the immutable transition relation from the frame-specific lemma
    /// log. A batched accelerator can keep the former resident and attach the
    /// latter to each query, avoiding a context reload every time IC3 grows a
    /// frame while preserving the exact formula seen by GipSAT.
    pub fn incremental_resident_partition(
        &self,
    ) -> (u32, u32, Vec<LitVec>, Vec<LitVec>) {
        (
            self.num_var() as u32,
            self.accel_level,
            self.resident_trans.clone(),
            self.resident_lemmas.clone(),
        )
    }

    pub fn new(dc: &DagCnf) -> Self {
        let constrain_act = Var::CONST;
        let resident_trans = dc.clause().cloned().collect();
        let mut solver = Self {
            dc: Gptr::new(dc),
            cdb: Default::default(),
            watchers: Default::default(),
            value: VarAssign::new_with(constrain_act),
            trail: Default::default(),
            pos_in_trail: Default::default(),
            level: VarMap::new_with(constrain_act),
            reason: VarMap::new_with(constrain_act),
            propagated: Default::default(),
            vsids: Default::default(),
            phase_saving: Default::default(),
            analyze: Default::default(),
            simplify: Default::default(),
            eqc: Default::default(),
            unsat_core: Default::default(),
            domain: Domain::new(),
            temporary_domain: Default::default(),
            prepared_vsids: false,
            constrain_act,
            assump: Default::default(),
            constraint: Default::default(),
            resident_trans,
            resident_lemmas: Default::default(),
            statistic: Default::default(),
            trivial_unsat: false,
            accel_id: Self::fresh_accel_id(),
            accel_level: 0,
            probe: Default::default(),
            fanout_len: Default::default(),
            rng: SmallRng::seed_from_u64(0),
            cfg: Default::default(),
            mark: Default::default(),
        };
        while solver.num_var() < solver.dc.num_var() {
            solver.new_var();
        }
        // Invert the DagCnf dependency lists into fanout degrees. `dep(v)` is
        // v's fanins, so v contributes one fanout entry to each of them. The
        // gate that v itself defines must also be re-evaluated, which is the
        // `+1` applied below for non-leaf variables.
        solver.fanout_len.reserve(solver.dc.max_var());
        for v in solver.dc.var_iter() {
            for d in solver.dc.dep(v).iter() {
                solver.fanout_len[*d] += 1;
            }
        }
        for v in solver.dc.var_iter() {
            if !solver.dc.is_leaf(v) {
                solver.fanout_len[v] += 1;
            }
        }

        for cls in dc.clause() {
            solver.add_clause_inner(cls, ClauseKind::Trans);
        }
        assert!(solver.propagate() == CREF_NONE);
        solver
    }

    #[inline]
    #[allow(unused)]
    pub fn set_rseed(&mut self, rseed: u64) {
        self.rng = SmallRng::seed_from_u64(rseed);
    }

    fn simplify_clause(&mut self, clause: &[Lit]) -> Option<LitVec> {
        assert!(self.highest_level() == 0);
        let mut clause = logicrs::LitVec::from(clause);
        clause.sort();
        let clause = clause.ordered_simp(&self.value)?;
        if clause.is_empty() {
            self.trivial_unsat = true;
            return None;
        }
        Some(clause)
    }

    fn add_clause_inner(&mut self, clause: &[Lit], mut kind: ClauseKind) -> CRef {
        if let Some(clause) = self.simplify_clause(clause) {
            if clause.iter().any(|l| l.var() == self.constrain_act) {
                kind = ClauseKind::Temporary;
            }
            if clause.len() == 1 {
                assert!(clause[0].var() != self.constrain_act);
                match self.value.v(clause[0]) {
                    Lbool::TRUE | Lbool::FALSE => todo!(),
                    _ => {
                        self.assign(clause[0], CREF_NONE);
                        let probe_sbcp = crate::inductor::Timer::start();
        let setup_conflict = self.propagate();
        crate::inductor::SETUP_BCP_NS
            .fetch_add(probe_sbcp.ns() as u64, std::sync::atomic::Ordering::Relaxed);
        if setup_conflict != CREF_NONE {
                            self.trivial_unsat = true;
                        }
                        CREF_NONE
                    }
                }
            } else {
                self.attach_clause(&clause, kind)
            }
        } else {
            CREF_NONE
        }
    }

    pub fn add_eq(&mut self, x: Lit, y: Lit) {
        self.eqc.add_eq(x, y);
    }

    fn reset(&mut self) {
        self.backtrack(0, false);
        self.clean_temporary();
        self.prepared_vsids = false;
        self.domain.reset();
        assert!(!self.temporary_domain);
    }

    fn new_round(
        &mut self,
        domain: impl Iterator<Item = Var>,
        constraint: Vec<LitVec>,
        bucket: bool,
    ) -> bool {
        // GipSAT defers the previous query's teardown to here, so time it
        // separately: it is the previous query's cost showing up in this one's
        // setup, and the hardware design needs to know how big it is.
        let td = crate::inductor::Timer::start();
        self.backtrack(0, self.temporary_domain);
        self.clean_temporary();
        self.probe.t_teardown_ns = td.ns();
        self.prepared_vsids = false;

        for mut c in constraint {
            c.push(!self.constrain_act.lit());
            if let Some(c) = self.simplify_clause(&c) {
                assert!(!c.is_empty());
                if c.len() == 1 {
                    return false;
                }
                self.add_clause_inner(&c, ClauseKind::Temporary);
            }
        }

        if !self.temporary_domain {
            self.domain.enable_local(domain, &self.dc, &self.value);
            assert!(!self.domain.has(self.constrain_act));
            self.domain.insert(self.constrain_act);
            if bucket {
                self.vsids.enable_bucket = true;
                self.vsids.bucket.clear();
            } else {
                self.vsids.enable_bucket = false;
                self.vsids.heap.clear();
            }
        }
        self.statistic.avg_decide_var +=
            self.domain.len() as f64 / (self.dc.num_var() - self.trail.len()) as f64;
        self.probe.domain_size = self.domain.len();
        self.probe.n_var_total = self.dc.num_var() as u32;
        true
    }

    pub fn solve_with_param(
        &mut self,
        assump: &[Lit],
        constraint: Vec<LitVec>,
        domain: impl Iterator<Item = Var>,
        limit: Option<usize>,
    ) -> Option<bool> {
        self.assump = assump.into();
        self.constraint = constraint.clone();
        if self.trivial_unsat {
            self.unsat_core.clear();
            return Some(false);
        }
        self.statistic.num_solve += 1;
        let start = Instant::now();

        // --- Inductor instrumentation ---
        // `t_setup` runs from here to the first decision: domain computation,
        // temporary-clause install, learnt-DB cleanup, simplification. Together
        // with `t_core` it is the per-query fixed overhead that a resident
        // hardware engine can drive toward zero.
        self.probe.begin();
        let probe_total = crate::inductor::Timer::start();
        let probe_setup = crate::inductor::Timer::start();
        self.probe.n_assump = assump.len() as u32;
        self.probe.n_constraint_lits = constraint.iter().map(|c| c.len() as u32).sum();
        self.probe.n_learnt = self.cdb.num_learnt() as u32;
        self.probe.n_lemma = self.cdb.num_lemma() as u32;

        let mut assumption;
        if self.propagate() != CREF_NONE {
            self.trivial_unsat = true;
            self.unsat_core.clear();
            self.statistic.avg_solve_time += start.elapsed();
            self.probe.t_setup_ns = probe_setup.ns();
            self.probe.t_total_ns = probe_total.ns();
            crate::inductor::record(&self.probe, Some(false));
            return Some(false);
        }
        let assump = if !constraint.is_empty() {
            assumption = LitVec::new();
            assumption.push(self.constrain_act.lit());
            assumption.extend_from_slice(assump);
            let cc: Vec<Lit> = constraint.iter().flatten().copied().collect();
            let probe_dom = crate::inductor::Timer::start();
            let nr_ok = self.new_round(
                domain.chain(assump.iter().chain(cc.iter()).map(|l| l.var())),
                constraint,
                true,
            );
            crate::inductor::DOMAIN_NS
                .fetch_add(probe_dom.ns() as u64, std::sync::atomic::Ordering::Relaxed);
            if !nr_ok {
                self.unsat_core.clear();
                self.statistic.avg_solve_time += start.elapsed();
                self.probe.t_setup_ns = probe_setup.ns();
                self.probe.t_total_ns = probe_total.ns();
                crate::inductor::record(&self.probe, Some(false));
                return Some(false);
            };
            &assumption
        } else {
            let probe_dom2 = crate::inductor::Timer::start();
            assert!(self.new_round(domain.chain(assump.iter().map(|l| l.var())), vec![], true));
            crate::inductor::DOMAIN_NS
                .fetch_add(probe_dom2.ns() as u64, std::sync::atomic::Ordering::Relaxed);
            assump
        };
        // Replay: the assumption literals and the domain the solver just
        // computed, not merely their sizes. Recorded after `new_round` because
        // that is what fills the domain, and before the query record is written
        // because the writer stamps both streams with the same id.
        if crate::inductor::enabled() {
            // A snapshot before the query it applies to, so a reader can take
            // the most recent one at or before a query id.
            let n = crate::inductor::N_QUERY.load(std::sync::atomic::Ordering::Relaxed);
            if n % crate::inductor::snapshot_every() == 0 {
                crate::inductor::replay_lemma_snapshot(&self.cdb.lemma_snapshot());
            }
            let raw: Vec<u32> = assump.iter().map(|l| Into::<u32>::into(*l)).collect();
            let dom: Vec<u32> = (0..self.domain.len())
                .map(|i| Into::<u32>::into(self.domain[i]))
                .collect();
            crate::inductor::replay_assumptions(&raw, &dom);
        }
        let probe_db = crate::inductor::Timer::start();
        self.clean_learnt(true);
        self.simplify();
        crate::inductor::DB_NS
            .fetch_add(probe_db.ns() as u64, std::sync::atomic::Ordering::Relaxed);
        self.probe.t_setup_ns = probe_setup.ns();

        // GipSAT performs unrestricted level-zero propagation before its
        // local decision-domain search. Preserve that exact boundary for the
        // shadow query: the current HLS P1 propagator otherwise treats an
        // unassigned out-of-domain literal as a blocker and can miss a root
        // value already established by the CPU's full propagation.
        let shadow_root: LitVec = if crate::accel::cdcl_host::shadow_enabled() {
            self.trail
                .iter()
                .filter(|lit| self.level[lit.var()] == 0 && lit.var() != self.constrain_act)
                .copied()
                .collect()
        } else {
            LitVec::new()
        };

        let probe_search = crate::inductor::Timer::start();
        let res = self.search_with_restart(assump, limit);
        // Core extraction happens inside the search loop, so subtract it out to
        // leave `t_search` as decide/BCP/conflict-analysis only.
        self.probe.t_search_ns = probe_search.ns().saturating_sub(self.probe.t_core_ns);

        // Full-CDCL shadow mode batches real GipSAT inquiries that share an
        // identical resident transition/frame snapshot. It runs after the CPU
        // answer, so it cannot affect IC3 while the new proof boundary is
        // being validated on real workloads.
        if crate::accel::cdcl_host::shadow_enabled() {
            let mut domain: Vec<Var> = (0..self.domain.len())
                .map(|i| self.domain[i])
                .filter(|var| *var != self.constrain_act)
                .collect();
            let mut constraints = self.constraint.clone();
            for lit in shadow_root {
                constraints.push(LitVec::from([lit]));
                if !domain.contains(&lit.var()) {
                    domain.push(lit.var());
                }
            }
            // Correctness baseline: solve the complete resident CNF. The P1
            // scan propagator's approximation of GipSAT's watcher-dependent
            // local-domain semantics is still experimental and can be enabled
            // explicitly for diagnosis.
            if std::env::var_os("INDUCTOR_CDCL_SHADOW_LOCAL_DOMAIN").is_none() {
                domain = (0..self.num_var()).map(Var::from).collect();
            }
            // The card is a short-query accelerator: synchronous lanes must
            // not let one CDCL long tail stall the other results in a round.
            // UNKNOWN is only profiling here and is a proof-safe CPU retry in
            // active mode. Zero can still be selected explicitly for an
            // unrestricted diagnostic run.
            let conflict_budget = crate::accel::cdcl_host::shadow_conflict_budget();
            let query = IncrementalQuery {
                frame: self.accel_level,
                assumptions: self.assump.clone(),
                constraints,
                domain,
                budget: QueryBudget {
                    conflicts: conflict_budget,
                    ..QueryBudget::default()
                },
                // Keep the first semantic audit independent of any possible
                // cross-query learnt-clause contamination in the P1 engine.
                keep_learnts: false,
            };
            crate::accel::cdcl_host::queue_shadow(self, query, res);
        }

        // Shadow, checked as an implication rather than an equality.
        //
        // Three earlier versions compared things that are not comparable: the
        // trails (different clause sets, different domains), then the conflict
        // verdict against `trivial_unsat` -- which is whether level 0 conflicts
        // *before* the assumptions are asserted, while the card is given the
        // assumptions and run to a fixpoint. Different questions.
        //
        // What does hold: if propagating the assumptions conflicts, the query
        // is unsatisfiable. The converse does not -- the solver can need
        // decisions and conflict analysis to get there -- so only one direction
        // is a defect.
        if crate::accel::shadow() && crate::accel::ready() && !assump.is_empty() {
            crate::accel::sync_index();
            let dom: Vec<u32> = (0..self.domain.len())
                .map(|i| Into::<u32>::into(self.domain[i]))
                .collect();
            if !crate::accel::batching() {
                crate::accel::set_domain(&dom);
            }
            let raw: Vec<u32> = assump.iter().map(|l| Into::<u32>::into(*l)).collect();
            thread_local! {
                static GOT: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
            }
            let mut got: Vec<u32> = GOT.with(|g| std::mem::take(&mut *g.borrow_mut()));
            // Queued, not asked, when batching: per query the call costs more
            // than the datapath runs for, and nothing here waits on the answer.
            if crate::accel::batching() {
                crate::accel::queue_verdict(&raw, res == Some(true));
            } else if let Some(conflict) = crate::accel::verdict(&raw, crate::accel::level_arg(self.accel_level), &mut got) {
                use std::sync::atomic::Ordering as O;
                if conflict && res == Some(true) {
                    // The card derived a contradiction from a query the solver
                    // satisfied. It holds a subset of the constraints, so this
                    // cannot be right.
                    crate::accel::CARD_ONLY_CONFLICT.fetch_add(1, O::Relaxed);
                    crate::accel::DISAGREE.fetch_add(1, O::Relaxed);
                } else {
                    if !conflict && res == Some(false) {
                        crate::accel::CPU_ONLY_CONFLICT.fetch_add(1, O::Relaxed);
                    } else if conflict && res == Some(false) {
                        // Propagation alone settled a query the solver also
                        // found unsat. This is the only case where the card
                        // does work the solver would otherwise have to do.
                        crate::accel::CARD_RESOLVED.fetch_add(1, O::Relaxed);
                    }
                    crate::accel::AGREE.fetch_add(1, O::Relaxed);
                }
                GOT.with(|g| *g.borrow_mut() = got);
            }
        }
        self.probe.t_total_ns = probe_total.ns();
        if crate::inductor::kind_counting() {
            use std::sync::atomic::Ordering as O;
            // One query in a thousand. The walk is O(lemma literals); doing it
            // on every query doubled runtime and produced nothing usable.
            let n = crate::inductor::N_QUERY.load(O::Relaxed);
            if n % 1000 == 0 {
                let n_lit = (self.dc.num_var() + 1) * 2;
                let (visits, lits) = self.cdb.lemma_occurrence_visits(&self.trail, n_lit);
                let (_v, sat, raw, blk) = self.cdb.lemma_blocker_saving(&self.trail, n_lit, |l| {
                    match self.value.v(l) {
                        logicrs::Lbool::TRUE => Some(true),
                        logicrs::Lbool::FALSE => Some(false),
                        _ => None,
                    }
                });
                crate::inductor::OCC_SAT.fetch_add(sat, O::Relaxed);
                crate::inductor::OCC_RAW.fetch_add(raw, O::Relaxed);
                crate::inductor::OCC_BLK.fetch_add(blk, O::Relaxed);
                crate::inductor::OCC_VISITS.fetch_add(visits, O::Relaxed);
                crate::inductor::OCC_LITS.fetch_add(lits, O::Relaxed);
                crate::inductor::OCC_SAMPLES.fetch_add(1, O::Relaxed);
                crate::inductor::OCC_WATCH.store(
                    crate::inductor::W_OTHER.load(O::Relaxed),
                    O::Relaxed,
                );
            }
        }
        {
            use std::sync::atomic::Ordering as O;
            crate::inductor::BCP_NS.fetch_add(self.probe.t_bcp_ns as u64, O::Relaxed);
            crate::inductor::SEARCH_NS.fetch_add(self.probe.t_search_ns as u64, O::Relaxed);
            crate::inductor::TOTAL_NS.fetch_add(self.probe.t_total_ns as u64, O::Relaxed);
            crate::inductor::SETUP_NS.fetch_add(self.probe.t_setup_ns as u64, O::Relaxed);
            // Reported as the run goes, not only at the end: the benchmarks
            // this matters for do not terminate inside a useful timeout, and a
            // killed process never reaches `finish`.
            let n = crate::inductor::N_QUERY.fetch_add(1, O::Relaxed) + 1;
            if n % 20000 == 0 {
                crate::inductor::report_bcp_share();
            }
        }
        crate::inductor::record(&self.probe, res);

        self.statistic.avg_solve_time += start.elapsed();
        res
    }

    /// This solver's lemma set. IC3 keeps one solver per frame and `add_lemma`
    /// writes into a range of them, so the total across frames is far larger
    /// than any one -- which is the gap between a replay that loads one
    /// snapshot and the propagation figures measured across the whole run.
    pub fn lemma_size(&self) -> (usize, u64) {
        (self.cdb.num_lemma(), self.cdb.lemma_lits())
    }

    pub fn solve_with_restart_limit(
        &mut self,
        assumps: &[Lit],
        constraint: Vec<LitVec>,
        limit: usize,
    ) -> Option<bool> {
        self.solve_with_param(assumps, constraint, empty::<Var>(), Some(limit))
    }

    /// Return true only when setup plus assumption propagation proves UNSAT.
    ///
    /// This follows the ordinary solve path through temporary-constraint
    /// installation, domain construction and simplification, then asserts the
    /// assumptions one decision level at a time exactly as `search()` does.
    /// It deliberately stops after BCP: no branching, conflict analysis,
    /// learning or restart. The selective FPGA path runs it on a clone, so a
    /// rejected hardware core cannot perturb the live incremental solver.
    pub fn conflicts_by_propagation(
        &mut self,
        assump: &[Lit],
        constraint: Vec<LitVec>,
    ) -> bool {
        self.assump = assump.into();
        self.constraint = constraint.clone();
        if self.trivial_unsat {
            return true;
        }
        if self.propagate() != CREF_NONE {
            return true;
        }

        let mut activated;
        let assump = if constraint.is_empty() {
            assert!(self.new_round(assump.iter().map(|l| l.var()), vec![], true));
            assump
        } else {
            activated = LitVec::new();
            activated.push(self.constrain_act.lit());
            activated.extend_from_slice(assump);
            let constraint_lits: Vec<Lit> = constraint.iter().flatten().copied().collect();
            if !self.new_round(
                assump
                    .iter()
                    .chain(constraint_lits.iter())
                    .map(|l| l.var()),
                constraint,
                true,
            ) {
                return true;
            }
            &activated
        };

        self.clean_learnt(true);
        self.simplify();
        for &a in assump {
            match self.value.v(a) {
                Lbool::TRUE => self.new_level(),
                Lbool::FALSE => return true,
                _ => {
                    self.new_level();
                    self.assign(a, CREF_NONE);
                }
            }
            if self.propagate() != CREF_NONE {
                return true;
            }
        }
        false
    }

    pub fn solve_with_domain(
        &mut self,
        assumps: &[Lit],
        domain: impl Iterator<Item = Var>,
    ) -> bool {
        self.solve_with_param(assumps, vec![], domain, None)
            .unwrap()
    }

    #[allow(unused)]
    pub fn imply<'a>(
        &mut self,
        domain: impl Iterator<Item = Var>,
        assump: impl Iterator<Item = &'a Lit>,
    ) {
        self.reset();
        self.domain.enable_local(domain, &self.dc, &self.value);
        self.new_level();
        for a in assump {
            if let Lbool::FALSE = self.value.v(*a) {
                panic!();
            }
            self.assign(*a, CREF_NONE);
        }
        assert!(self.propagate() == CREF_NONE);
    }

    #[inline]
    #[allow(unused)]
    pub fn assert_value(&mut self, lit: Lit) -> Option<bool> {
        self.reset();
        self.value.v(lit).into()
    }

    #[inline]
    pub fn statistic(&self) -> &SolverStatistic {
        &self.statistic
    }

    #[allow(unused)]
    pub fn sat_value_bitvet(&mut self) -> BitVec {
        let mut res = BitVec::new();
        for v in VarRange::new_inclusive(Var::CONST, self.max_var()) {
            if let Some(v) = self.sat_value(v.lit()) {
                res.push(v);
            } else {
                res.push(self.rng.random_bool(0.5));
            }
        }
        res
    }

    #[allow(unused)]
    pub fn sat_value_iter(&self) -> impl Iterator<Item = &'_ Lit> {
        let constrain_act = self.constrain_act;
        self.trail.iter().filter(move |l| l.var() != constrain_act)
    }

    pub fn minimal_premise(
        &mut self,
        assump: &[Lit],
        premise: &[Lit],
        consequent: &[Lit],
    ) -> Option<LitVec> {
        let assump = LitVec::from_iter(assump.iter().chain(premise.iter()).copied());
        if self.solve_with_constraint(&assump, vec![LitVec::from(consequent)]) {
            return None;
        }
        Some(
            premise
                .iter()
                .filter(|l| self.unsat_has(**l))
                .copied()
                .collect(),
        )
    }
}

impl Satif for DagCnfSolver {
    #[inline]
    fn new_var(&mut self) -> Var {
        self.reset();
        let v = self.constrain_act;
        let var = Var::new(self.num_var() + 1);
        self.value.reserve(var);
        self.level.reserve(var);
        self.reason.reserve(var);
        self.watchers.reserve(var);
        self.vsids.reserve(var);
        self.phase_saving.reserve(var);
        self.eqc.reserve(var);
        self.analyze.reserve(var);
        self.unsat_core.reserve(var);
        self.domain.reserve(var);
        self.mark.reserve(var);
        // Must grow with every other var-indexed structure: VarMap indexes with
        // get_unchecked in release builds, so missing this is an out-of-bounds
        // read, not a panic. Auxiliary variables drive no gates, so 0 is right.
        self.fanout_len.reserve(var);
        self.constrain_act = var;
        v
    }

    #[inline]
    fn num_var(&self) -> usize {
        self.constrain_act.into()
    }

    fn add_clause(&mut self, clause: &[Lit]) {
        self.reset();
        self.resident_lemmas.push(LitVec::from(clause));
        for l in clause.iter() {
            self.add_domain(l.var(), true);
        }
        if crate::inductor::enabled() {
            let raw: Vec<u32> = clause.iter().map(|l| Into::<u32>::into(*l)).collect();
            crate::inductor::replay_lemma(&raw);
        }
        self.add_clause_inner(clause, ClauseKind::Lemma);
    }

    fn solve(&mut self, assumps: &[Lit]) -> bool {
        self.solve_with_param(assumps, vec![], empty::<Var>(), None)
            .unwrap()
    }

    fn solve_with_constraint(&mut self, assumps: &[Lit], constraint: Vec<LitVec>) -> bool {
        self.solve_with_param(assumps, constraint, empty::<Var>(), None)
            .unwrap()
    }

    #[inline]
    fn sat_value(&self, lit: Lit) -> Option<bool> {
        match self.value.v(lit) {
            Lbool::TRUE => Some(true),
            Lbool::FALSE => Some(false),
            _ => None,
        }
    }

    #[inline]
    fn unsat_has(&self, lit: Lit) -> bool {
        self.unsat_core.has(lit)
    }

    #[inline]
    fn flip_to_none(&mut self, var: Var) -> bool {
        self.flip_to_none_inner(var)
    }
}

#[cfg(test)]
mod propagation_validation_tests {
    use super::DagCnfSolver;
    use logicrs::satif::Satif;
    use logicrs::{DagCnf, LitVec};

    #[test]
    fn exact_bcp_validator_accepts_only_propagation_conflicts() {
        let mut dc = DagCnf::new();
        let a = dc.new_var().lit();
        let b = dc.new_var().lit();

        let mut implication = DagCnfSolver::new(&dc);
        implication.add_clause(&[!a, b]);
        assert!(implication.clone().conflicts_by_propagation(&[a, !b], vec![]));
        assert!(!implication.clone().conflicts_by_propagation(&[a, b], vec![]));
        assert!(implication.clone().conflicts_by_propagation(
            &[a],
            vec![LitVec::from([!a])],
        ));

        // Globally UNSAT but with no unit propagation at level zero. A
        // validator that branches would prove this; the FPGA proof boundary
        // must return false because BCP alone did not.
        let mut needs_search = DagCnfSolver::new(&dc);
        needs_search.add_clause(&[a, b]);
        needs_search.add_clause(&[a, !b]);
        needs_search.add_clause(&[!a, b]);
        needs_search.add_clause(&[!a, !b]);
        assert!(!needs_search.conflicts_by_propagation(&[], vec![]));
    }
}
