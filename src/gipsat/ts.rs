use crate::{
    gipsat::{DagCnfSolver, IncrementalQuery, QueryBudget, SolverStatistic},
    transys::{TransysCtx, TransysIf},
};
use giputils::ptr::Grc;
use logicrs::{Lit, LitVec, Var, satif::Satif};

#[derive(Clone)]
pub struct TransysSolver {
    pub dcs: DagCnfSolver,
    ts: Grc<TransysCtx>,

    relind: LitVec,
}

impl TransysSolver {
    pub fn new(ts: &Grc<TransysCtx>) -> Self {
        let mut dcs = DagCnfSolver::new(&ts.rel);
        for c in ts.constraint.iter() {
            dcs.add_clause(&[*c]);
        }
        Self {
            dcs,
            ts: ts.clone(),
            relind: Default::default(),
        }
    }

    #[inline]
    pub fn get_assump(&self) -> &LitVec {
        &self.dcs.assump
    }

    #[allow(unused)]
    pub fn trivial_pred(&mut self) -> (LitVec, LitVec) {
        let mut input = LitVec::new();
        for i in self.ts.input() {
            if let Some(v) = self.dcs.sat_value_lit(i) {
                input.push(v);
            }
        }
        let mut latch = LitVec::new();
        for l in self.ts.latch() {
            if let Some(v) = self.dcs.sat_value_lit(l) {
                latch.push(v);
            }
        }
        (input, latch)
    }

    pub fn inductive_with_constrain(
        &mut self,
        cube: &[Lit],
        strengthen: bool,
        mut constraint: Vec<LitVec>,
    ) -> bool {
        self.relind = LitVec::from(cube);
        let assump = self.ts.lits_next(cube);
        if strengthen {
            constraint.push(LitVec::from_iter(cube.iter().map(|l| !*l)));
        }
        !self.dcs.solve_with_constraint(&assump, constraint.clone())
    }

    pub fn inductive(&mut self, cube: &[Lit], strengthen: bool) -> bool {
        self.inductive_with_constrain(cube, strengthen, vec![])
    }

    /// Build the exact full-domain inquiry used by `inductive_with_constrain`
    /// without mutating GipSAT. IC3 uses this to batch independent push checks
    /// before deciding which answers can safely bypass CPU search.
    pub fn incremental_inductive_query(
        &self,
        cube: &[Lit],
        strengthen: bool,
        mut constraint: Vec<LitVec>,
    ) -> IncrementalQuery {
        if strengthen {
            constraint.push(LitVec::from_iter(cube.iter().map(|lit| !*lit)));
        }
        IncrementalQuery {
            frame: self.dcs.accel_level,
            assumptions: self.ts.lits_next(cube),
            constraints: constraint,
            domain: (0..self.dcs.num_var()).map(Var::from).collect(),
            budget: QueryBudget {
                conflicts: crate::accel::cdcl_host::active_conflict_budget(),
                ..QueryBudget::default()
            },
            keep_learnts: false,
        }
    }

    /// Stable positive-current-latch -> next-literal map registered with the
    /// resident BLOCK root controller. The value includes the next literal's
    /// polarity: TransysCtx may represent a latch through an inverted next
    /// literal, so a variable-only map silently flips Q_block assumptions.
    /// Non-latch variables are deliberately unmapped.
    pub fn resident_block_next_var_map(&self) -> Vec<u32> {
        let mut map = vec![u32::MAX; self.dcs.num_var()];
        for current in self.ts.latch() {
            let current_index = usize::from(current);
            let next = self.ts.next(current.lit());
            if current_index < map.len() {
                map[current_index] = u32::from(next);
            }
        }
        map
    }

    /// Projection metadata installed by the complete resident BLOCK root.
    /// Init uses 0/1 for constant latch values and 2 for symbolic variables;
    /// latch/input lists define the exact packed-model witness projection.
    pub fn resident_block_projection_metadata(&self) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let mut init = vec![2u32; self.dcs.num_var()];
        let latches = self
            .ts
            .latch()
            .map(|latch| {
                if let Some(value) = self.ts.init_map[latch].and_then(|lit| lit.try_constant()) {
                    init[usize::from(latch)] = u32::from(value);
                }
                u32::from(latch)
            })
            .collect();
        let inputs = self.ts.input().map(u32::from).collect();
        (init, latches, inputs)
    }

    pub fn install_incremental_sat_model(
        &mut self,
        query: &IncrementalQuery,
        model: &[Lit],
    ) -> bool {
        self.dcs.install_incremental_sat_model(query, model)
    }

    pub fn install_trusted_incremental_sat_model(
        &mut self,
        query: &IncrementalQuery,
        model: &[Lit],
    ) -> bool {
        self.dcs.install_trusted_incremental_sat_model(query, model)
    }

    pub fn validate_incremental_sat_model(&self, query: &IncrementalQuery, model: &[Lit]) -> bool {
        self.dcs.validate_incremental_sat_model(query, model)
    }

    pub fn trusted_incremental_sat_model_shape(
        &self,
        query: &IncrementalQuery,
        model: &[Lit],
    ) -> bool {
        self.dcs.trusted_incremental_sat_model_shape(query, model)
    }

    /// Validate a hardware assumption core with an exact reduced GipSAT
    /// solve. Keeping the original cube in `relind` preserves the literal
    /// mapping and initial-state repair used by `inductive_core`.
    pub fn validate_incremental_unsat_core(
        &mut self,
        cube: &[Lit],
        query: &IncrementalQuery,
        hardware_core: &[Lit],
    ) -> Option<usize> {
        self.relind = LitVec::from(cube);
        self.dcs
            .validate_incremental_unsat_core(query, hardware_core)
    }

    /// Restore an already-qualified core for the ordinary IC3 `inductive_core`
    /// consumer. Exact CPU preflight and explicit trusted-accelerator mode use
    /// this path; neither needs another proof solve.
    pub fn install_incremental_proven_unsat_core(
        &mut self,
        cube: &[Lit],
        query: &IncrementalQuery,
        core: &[Lit],
        used_constraints: bool,
    ) -> bool {
        if !self
            .dcs
            .install_incremental_proven_unsat_core(query, core, used_constraints)
        {
            return false;
        }
        self.relind = LitVec::from(cube);
        true
    }

    /// Check relative induction using setup and BCP only.
    pub fn inductive_by_propagation(
        &mut self,
        cube: &[Lit],
        strengthen: bool,
        mut constraint: Vec<LitVec>,
    ) -> bool {
        self.relind = LitVec::from(cube);
        let assump = self.ts.lits_next(cube);
        if strengthen {
            constraint.push(LitVec::from_iter(cube.iter().map(|l| !*l)));
        }
        self.dcs.conflicts_by_propagation(&assump, constraint)
    }

    pub fn inductive_core(&mut self) -> Option<LitVec> {
        let mut ans = LitVec::new();
        for &l in self.relind.iter() {
            let nl = self.ts.next(l);
            if self.dcs.unsat_has(nl) {
                ans.push(l);
            }
        }
        if self.ts.cube_subsume_init(&ans) {
            ans = LitVec::new();
            let new = self.relind.iter().find(|&&l| {
                self.ts.init_map[l.var()]
                    .and_then(|l| l.try_constant())
                    .is_some_and(|i| i != l.polarity())
            })?;
            for &l in self.relind.iter() {
                let nl = self.ts.next(l);
                if self.dcs.unsat_has(nl) {
                    ans.push(l);
                }
                if l.eq(new) {
                    ans.push(l);
                }
            }
            assert!(!self.ts.cube_subsume_init(&ans));
        }
        Some(ans)
    }

    #[inline]
    pub fn flip_to_none(&mut self, var: Var) -> bool {
        self.dcs.flip_to_none(var)
    }

    #[inline]
    pub fn set_domain(&mut self, domain: impl IntoIterator<Item = Lit>) {
        self.dcs.set_domain(domain);
    }

    #[inline]
    pub fn unset_domain(&mut self) {
        self.dcs.unset_domain();
    }

    #[inline]
    #[allow(unused)]
    pub fn add_domain(&mut self, var: Var, deps: bool) {
        self.dcs.add_domain(var, deps);
    }

    #[inline]
    pub fn statistic(&self) -> &SolverStatistic {
        self.dcs.statistic()
    }
}

impl Satif for TransysSolver {
    #[inline]
    fn new_var(&mut self) -> Var {
        self.dcs.new_var()
    }

    #[inline]
    fn num_var(&self) -> usize {
        self.dcs.num_var()
    }

    #[inline]
    fn add_clause(&mut self, clause: &[Lit]) {
        self.dcs.add_clause(clause);
    }

    #[inline]
    fn solve(&mut self, assumps: &[Lit]) -> bool {
        self.dcs.solve(assumps)
    }

    #[inline]
    fn solve_with_constraint(&mut self, assumps: &[Lit], constraint: Vec<LitVec>) -> bool {
        self.dcs.solve_with_constraint(assumps, constraint)
    }

    #[inline]
    fn sat_value(&self, lit: Lit) -> Option<bool> {
        self.dcs.sat_value(lit)
    }

    #[inline]
    fn unsat_has(&self, lit: Lit) -> bool {
        self.dcs.unsat_has(lit)
    }

    #[inline]
    fn flip_to_none(&mut self, var: Var) -> bool {
        self.dcs.flip_to_none(var)
    }
}
