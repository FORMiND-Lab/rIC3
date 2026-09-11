use super::{
    DagCnfSolver,
    cdb::{CREF_NONE, CRef, ClauseKind},
};
use log::debug;
use logicrs::{Lbool, Lit};

impl DagCnfSolver {
    #[inline]
    pub fn highest_level(&self) -> usize {
        self.pos_in_trail.len()
    }

    #[inline]
    pub fn assign(&mut self, lit: Lit, reason: CRef) {
        // Inductor: what a gate-implication BCP path would have to visit, and
        // what that costs at each candidate lane count. ceil() per assignment,
        // because a variable with fanout 3 occupies a 16-lane datapath for a
        // whole cycle and leaves 13 lanes idle -- dividing totals would hide
        // exactly the underutilisation we need to size the datapath against.
        let fan = self.fanout_len[lit.var()];
        self.probe.n_fanout_visit += fan as u64;
        for (i, lanes) in crate::inductor::LANES.iter().enumerate() {
            self.probe.bcp_cycles[i] += fan.div_ceil(*lanes) as u64;
        }
        self.probe.n_assign += 1;
        self.trail.push(lit);
        self.value.set(lit);
        self.reason[lit] = reason;
        self.level[lit] = self.highest_level() as u32;
    }

    #[inline]
    pub fn new_level(&mut self) {
        self.pos_in_trail.push(self.trail.len() as u32)
    }

    #[inline]
    pub fn backtrack(&mut self, level: usize, vsids: bool) {
        if self.highest_level() <= level {
            return;
        }
        while self.trail.len() as u32 > self.pos_in_trail[level] {
            let bt = self.trail.pop().unwrap();
            self.value.set_none(bt.var());
            if vsids {
                self.vsids.push(bt.var());
            }
            self.phase_saving[bt] = Lbool::from(bt.polarity());
        }
        self.propagated = self.pos_in_trail[level];
        self.pos_in_trail.truncate(level);
    }

    pub fn search_with_restart(
        &mut self,
        assumption: &[Lit],
        limit: Option<usize>,
        conflict_limit: Option<u32>,
        retain_learnts: bool,
    ) -> Option<bool> {
        let mut restarts = 0;
        loop {
            if conflict_limit.is_some_and(|limit| self.probe.n_conflict >= limit) {
                return None;
            }
            if let Some(limit) = limit
                && restarts >= limit as u32
            {
                return None;
            }
            if restarts > 10 && self.vsids.enable_bucket {
                self.vsids.enable_bucket = false;
                self.vsids.heap.clear();
                for d in self.domain.iter() {
                    if self.value.v(d.lit()).is_none() {
                        self.vsids.push(*d);
                    }
                }
            }
            let rest_base = luby(2.0, restarts);
            let restart_conflicts = rest_base * 100.0;
            let search_conflicts = conflict_limit.map_or(restart_conflicts, |limit| {
                restart_conflicts.min(limit.saturating_sub(self.probe.n_conflict) as f64)
            });
            match self.search(assumption, Some(search_conflicts), retain_learnts) {
                None => {
                    if conflict_limit.is_some_and(|limit| self.probe.n_conflict >= limit) {
                        return None;
                    }
                    restarts += 1;
                    if restarts % 10 == 0 {
                        debug!(
                            "gipsat restarted {restarts} times with {} learnt clauses",
                            self.cdb.num_learnt()
                        );
                    }
                }
                Some(r) => return Some(r),
            }
        }
    }

    pub fn search(
        &mut self,
        assumption: &[Lit],
        noc: Option<f64>,
        retain_learnts: bool,
    ) -> Option<bool> {
        let mut num_conflict = 0.0_f64;
        'ml: loop {
            // BCP alone, separated from the decide and analyse it shares
            // `t_search` with.
            let probe_bcp = crate::inductor::Timer::start();
            let conflict = self.propagate();
            self.probe.t_bcp_ns = self.probe.t_bcp_ns.saturating_add(probe_bcp.ns());
            if conflict != CREF_NONE {
                num_conflict += 1.0;
                self.probe.n_conflict += 1;
                if self.highest_level() == 0 {
                    self.unsat_core.clear();
                    return Some(false);
                }
                let probe_an = crate::inductor::Timer::start();
                let (learnt, btl) = self.analyze(conflict);
                crate::inductor::ANALYZE_NS
                    .fetch_add(probe_an.ns() as u64, std::sync::atomic::Ordering::Relaxed);
                self.backtrack(btl, true);
                if learnt.len() == 1 {
                    debug_assert!(btl == 0);
                    self.assign(learnt[0], CREF_NONE);
                } else {
                    let kind = if !retain_learnts
                        || learnt.iter().any(|l| self.constrain_act == l.var())
                    {
                        ClauseKind::Temporary
                    } else {
                        ClauseKind::Learnt
                    };
                    let learnt_id = self.attach_clause(&learnt, kind);
                    self.cdb.bump(learnt_id);
                    let assign = self.cdb.get(learnt_id)[0];
                    self.assign(assign, learnt_id);
                }
                self.vsids.decay();
                self.cdb.decay();
            } else {
                if let Some(noc) = noc
                    && num_conflict >= noc
                {
                    self.backtrack(assumption.len(), true);
                    return None;
                }
                self.clean_learnt(false);
                while self.highest_level() < assumption.len() {
                    let a = assumption[self.highest_level()];
                    match self.value.v(a) {
                        Lbool::TRUE => {
                            self.new_level();
                            if self.highest_level() == assumption.len() {
                                self.prepare_vsids();
                            }
                        }
                        Lbool::FALSE => {
                            // UNSAT-core extraction: the other half of the
                            // per-query fixed overhead, and the operation with
                            // essentially no hardware prior art.
                            let t = crate::inductor::Timer::start();
                            self.analyze_unsat_core(a);
                            self.probe.t_core_ns += t.ns();
                            return Some(false);
                        }
                        _ => {
                            self.new_level();
                            self.assign(a, CREF_NONE);
                            if self.highest_level() == assumption.len() {
                                self.prepare_vsids();
                            }
                            continue 'ml;
                        }
                    }
                }
                // Empty assumptions skip the loop that initializes the
                // decision frontier. Do not mistake that empty queue for SAT.
                if assumption.is_empty() {
                    self.prepare_vsids();
                }
                let probe_de = crate::inductor::Timer::start();
                let decided = self.decide();
                crate::inductor::DECIDE_NS
                    .fetch_add(probe_de.ns() as u64, std::sync::atomic::Ordering::Relaxed);
                if !decided {
                    return Some(true);
                }
            }
        }
    }
}

fn luby(y: f64, mut x: u32) -> f64 {
    let mut size = 1;
    let mut seq = 0;
    while size < x + 1 {
        seq += 1;
        size = 2 * size + 1
    }
    while size - 1 != x {
        size = (size - 1) >> 1;
        seq -= 1;
        x %= size;
    }
    y.powi(seq)
}

#[cfg(test)]
mod empty_assumption_tests {
    use super::DagCnfSolver;
    use logicrs::satif::Satif;
    use logicrs::{DagCnf, Lit, Var};

    // DagCnfSolver retains a non-owning pointer. Keep the boxed owner alive
    // through every query, including after moving this tuple to the caller.
    fn flat_solver(unsat: bool) -> (Box<DagCnf>, DagCnfSolver, Vec<[Lit; 2]>) {
        let mut dc = Box::new(DagCnf::new());
        dc.new_var_to(Var(2));
        let a = Var(1).lit();
        let b = Var(2).lit();
        let mut clauses = vec![[a, b], [a, !b], [!a, b]];
        if unsat {
            clauses.push([!a, !b]);
        }
        let mut solver = DagCnfSolver::new(&dc);
        // Match the flat-CNF adapter: variable zero is the explicit false
        // constant, not a third unconstrained Boolean input.
        solver.add_clause(&[!Var(0).lit()]);
        for clause in &clauses {
            solver.add_clause(clause);
        }
        (dc, solver, clauses)
    }

    fn assert_model(solver: &DagCnfSolver, clauses: &[[Lit; 2]]) {
        let model: Vec<_> = solver.sat_value_iter().copied().collect();
        assert!(model.contains(&!Var(0).lit()), "missing explicit false constant");
        for var in [Var(1), Var(2)] {
            let positive = model.contains(&var.lit());
            let negative = model.contains(&!var.lit());
            assert_ne!(positive, negative, "missing or contradictory model variable {var:?}");
        }
        for clause in clauses {
            assert!(clause.iter().any(|literal| model.contains(literal)),
                    "SAT witness violates original clause {clause:?}: {model:?}");
        }
    }

    #[test]
    fn empty_assumption_full_domain_nonunit_unsat() {
        for start in [1, 0] {
            let (_dc, mut solver, _clauses) = flat_solver(true);
            assert_eq!(solver.solve_with_param(&[], vec![], (start..=2).map(Var), None),
                       Some(false), "four nonunit clauses require search, domain start={start}");
        }
    }

    #[test]
    fn empty_assumption_full_domain_nonunit_sat_has_model() {
        for start in [1, 0] {
            let (_dc, mut solver, clauses) = flat_solver(false);
            assert_eq!(solver.solve_with_param(&[], vec![], (start..=2).map(Var), None),
                       Some(true));
            // A bare SAT status with an empty decision frontier is insufficient.
            assert_model(&solver, &clauses);
        }
    }

    #[test]
    fn empty_assumption_same_solver_reinitializes_frontier() {
        let (_dc, mut solver, clauses) = flat_solver(false);
        assert_eq!(solver.solve_with_param(&[Var(1).lit()], vec![], (0..=2).map(Var), None),
                   Some(true));
        assert_model(&solver, &clauses);
        // Interleave both full-domain spellings without cloning/reconstructing
        // the solver: new_round clears the previous query's decision frontier.
        for start in [1, 0, 1, 0] {
            assert_eq!(solver.solve_with_param(&[], vec![], (start..=2).map(Var), None),
                       Some(true));
            assert_model(&solver, &clauses);
        }
        solver.add_clause(&[!Var(1).lit(), !Var(2).lit()]);
        for start in [1, 0] {
            assert_eq!(solver.solve_with_param(&[], vec![], (start..=2).map(Var), None),
                       Some(false), "same solver must see the newly installed fourth clause");
        }
    }
}
