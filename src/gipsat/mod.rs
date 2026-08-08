mod analyze;
mod cdb;
mod domain;
mod eq;
mod propagate;
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
    pub fn new(dc: &DagCnf) -> Self {
        let constrain_act = Var::CONST;
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
            statistic: Default::default(),
            trivial_unsat: false,
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

        let probe_search = crate::inductor::Timer::start();
        let res = self.search_with_restart(assump, limit);
        // Core extraction happens inside the search loop, so subtract it out to
        // leave `t_search` as decide/BCP/conflict-analysis only.
        self.probe.t_search_ns = probe_search.ns().saturating_sub(self.probe.t_core_ns);
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
