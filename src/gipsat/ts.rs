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

#[inline]
fn raw_lit(raw: u32) -> Lit {
    Lit::new(Var::from(raw >> 1), raw & 1 == 0)
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
        if crate::fastlane::enabled()
            && crate::fastlane::routes_phase(crate::inductor::current_phase())
            && (!crate::fastlane::pool_mode() || crate::fastlane::in_band(self.dcs.fastlane_frame))
            && !crate::fastlane::reload_pending()
            && !crate::fastlane::push_split_enabled()
        {
            if crate::fastlane::shadow() {
                return self.inductive_shadowed(&assump, &constraint);
            }
            if let Some(us) = crate::fastlane::race_us()
                && crate::fastlane::card_ready()
            {
                return self.inductive_timed(cube, &assump, &constraint, us);
            }
            // While the card is coming up, light queries run on the CPU; a
            // query that exceeds the warm-up budget goes to the card (the
            // flush waits for it).
            // With INDUCTOR_FASTLANE_CPU_FIRST=<conflicts> every query first
            // runs on the CPU under that budget and only harder ones go on.
            let budget = if crate::fastlane::card_ready() {
                crate::fastlane::cpu_first_budget()
            } else {
                Some(crate::fastlane::warmup_conflicts())
            };
            if let Some(budget) = budget
                && let Some(sat) =
                    self.dcs
                        .solve_with_constraint_budget(&assump, constraint.clone(), budget)
            {
                return !sat;
            }
            if crate::fastlane::race_enabled()
                && crate::fastlane::card_ready()
                && let Some(blocked) = self.inductive_raced(cube, &assump, &constraint)
            {
                return blocked;
            }
            if let Some(blocked) = self.inductive_on_lane(cube, &assump, &constraint) {
                return blocked;
            }
        }
        !self.dcs.solve_with_constraint(&assump, constraint.clone())
    }

    /// Solve on this solver's private lane, or on pool replica 0 at this
    /// solver's frame in pool mode.
    fn lane_solve(&mut self, assump: &[u32], temps: &[Vec<u32>]) -> crate::fastlane::LaneResult {
        if crate::fastlane::pool_mode() {
            let frame = self.dcs.fastlane_frame.unwrap_or(0);
            return crate::fastlane::with_pool(|pool| {
                pool.replicas[0].solve_frame(assump, temps, 0, frame)
            })
            .unwrap_or(crate::fastlane::LaneResult::Unknown);
        }
        self.dcs.fastlane_lane().solve(assump, temps, 0)
    }

    /// GipSAT answers; the lane solves the same query for paired logging.
    fn inductive_shadowed(&mut self, assump: &LitVec, constraint: &[LitVec]) -> bool {
        use std::sync::atomic::Ordering;
        let started = std::time::Instant::now();
        let sat = self.dcs.solve_with_constraint(assump, constraint.to_vec());
        let cpu_ns = started.elapsed().as_nanos() as u64;
        let raw_assump: Vec<u32> = assump.iter().map(|l| u32::from(*l)).collect();
        let raw_temps: Vec<Vec<u32>> = constraint
            .iter()
            .map(|c| c.iter().map(|l| u32::from(*l)).collect())
            .collect();
        let before = crate::fastlane::global_events();
        let lane_started = std::time::Instant::now();
        let result = self.lane_solve(&raw_assump, &raw_temps);
        let ns = lane_started.elapsed().as_nanos() as u64;
        crate::fastlane::LANE_NS.fetch_add(ns, Ordering::Relaxed);
        let after = crate::fastlane::global_events();
        let status = match &result {
            crate::fastlane::LaneResult::Sat { .. } => 1,
            crate::fastlane::LaneResult::Unsat { .. } => 2,
            crate::fastlane::LaneResult::Unknown => 0,
        };
        if status != if sat { 1 } else { 2 } {
            crate::fastlane::SHADOW_MISMATCH.fetch_add(1, Ordering::Relaxed);
        }
        if let crate::fastlane::LaneResult::Unsat { core, .. } = &result
            && !sat
        {
            let cpu_core = assump.iter().filter(|&&l| self.dcs.unsat_has(l)).count();
            crate::fastlane::SHADOW_UNSAT.fetch_add(1, Ordering::Relaxed);
            crate::fastlane::SHADOW_CORE_CPU.fetch_add(cpu_core as u64, Ordering::Relaxed);
            crate::fastlane::SHADOW_CORE_LANE.fetch_add(core.len() as u64, Ordering::Relaxed);
        }
        crate::fastlane::log_query(
            crate::inductor::current_phase(),
            self.dcs.accel_level,
            status,
            ns,
            cpu_ns,
            &self.dcs.probe_summary(),
            &before,
            &after,
        );
        !sat
    }

    /// Answer one relative-induction query on this solver's FastLane model
    /// and install the result into GipSAT's live state, so IC3 reads the
    /// model or core exactly as after a CPU solve.  `None` falls back to CPU.
    fn inductive_on_lane(
        &mut self,
        cube: &[Lit],
        assump: &LitVec,
        constraint: &[LitVec],
    ) -> Option<bool> {
        use crate::fastlane::{LaneResult, ROUTED};
        use std::sync::atomic::Ordering;
        let phase = crate::inductor::current_phase() as usize;
        let raw_assump: Vec<u32> = assump.iter().map(|l| u32::from(*l)).collect();
        let raw_temps: Vec<Vec<u32>> = constraint
            .iter()
            .map(|c| c.iter().map(|l| u32::from(*l)).collect())
            .collect();
        ROUTED.fetch_add(1, Ordering::Relaxed);
        crate::fastlane::PHASE_COUNTS[phase.min(7)].fetch_add(1, Ordering::Relaxed);
        let before = crate::fastlane::global_events();
        let started = std::time::Instant::now();
        let result = self.lane_solve(&raw_assump, &raw_temps);
        let ns = started.elapsed().as_nanos() as u64;
        crate::fastlane::LANE_NS.fetch_add(ns, Ordering::Relaxed);
        let after = crate::fastlane::global_events();
        crate::fastlane::note_card(crate::fastlane::wide_cycles(&before, &after));
        let status = match &result {
            LaneResult::Sat { .. } => 1,
            LaneResult::Unsat { .. } => 2,
            LaneResult::Unknown => 0,
        };
        crate::fastlane::log_query(
            phase as u8,
            self.dcs.accel_level,
            status,
            ns,
            0,
            "0\t0\t0\t0\t0\t0\t0\t0",
            &before,
            &after,
        );
        self.install_lane_result(cube, assump, constraint, result)
    }

    /// Install a lane answer into GipSAT's live state (see
    /// `inductive_on_lane`); `None` if it cannot be used.
    fn install_lane_result(
        &mut self,
        cube: &[Lit],
        assump: &LitVec,
        constraint: &[LitVec],
        result: crate::fastlane::LaneResult,
    ) -> Option<bool> {
        use crate::fastlane::{LaneResult, ROUTED_SAT, ROUTED_UNKNOWN, ROUTED_UNSAT};
        use std::sync::atomic::Ordering;
        let query = |domain: Vec<Var>| IncrementalQuery {
            frame: self.dcs.accel_level,
            assumptions: assump.clone(),
            constraints: constraint.to_vec(),
            domain,
            budget: QueryBudget::default(),
            keep_learnts: false,
        };
        match result {
            LaneResult::Sat { domain, model } => {
                let q = query(domain.into_iter().map(|v| Var::from(v as usize)).collect());
                let model: Vec<Lit> = model.into_iter().map(raw_lit).collect();
                if !self.dcs.install_lane_sat_model(&q, &model) {
                    crate::fastlane::INSTALL_FAILED.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                ROUTED_SAT.fetch_add(1, Ordering::Relaxed);
                Some(false)
            }
            LaneResult::Unsat { core, used_temps } => {
                let q = query(Vec::new());
                let core: Vec<Lit> = core.into_iter().map(raw_lit).collect();
                if !self
                    .dcs
                    .install_incremental_proven_unsat_core(&q, &core, used_temps)
                {
                    crate::fastlane::INSTALL_FAILED.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                self.relind = LitVec::from(cube);
                ROUTED_UNSAT.fetch_add(1, Ordering::Relaxed);
                Some(true)
            }
            LaneResult::Unknown => {
                ROUTED_UNKNOWN.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Time-triggered race (see `fastlane::race_us`): the CPU search submits
    /// the query to the card once it has run `us` microseconds.
    fn inductive_timed(
        &mut self,
        cube: &[Lit],
        assump: &LitVec,
        constraint: &[LitVec],
        us: u64,
    ) -> bool {
        let raw_assump: Vec<u32> = assump.iter().map(|l| u32::from(*l)).collect();
        let raw_temps: Vec<Vec<u32>> = constraint
            .iter()
            .map(|c| c.iter().map(|l| u32::from(*l)).collect())
            .collect();
        let frame = self.dcs.fastlane_frame.unwrap_or(0);
        crate::fastlane::race_arm(raw_assump, raw_temps, frame, us);
        let cpu = self
            .dcs
            .solve_with_constraint_raced(assump, constraint.to_vec());
        let submitted = crate::fastlane::race_disarm();
        if let Some(sat) = cpu {
            if submitted {
                crate::fastlane::card_abort();
                crate::fastlane::note_race(false);
            }
            return !sat;
        }
        crate::fastlane::note_race(true);
        crate::fastlane::ROUTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let result = crate::fastlane::with_pool(|pool| pool.replicas[0].collect_frame())
            .unwrap_or(crate::fastlane::LaneResult::Unknown);
        // the raced SOLVE has no budget: UNKNOWN means the lane overflowed
        if matches!(result, crate::fastlane::LaneResult::Unknown) {
            crate::fastlane::request_reload();
        }
        match self.install_lane_result(cube, assump, constraint, result) {
            Some(blocked) => blocked,
            None => !self.dcs.solve_with_constraint(assump, constraint.to_vec()),
        }
    }

    /// Race: the card solves the query while the CPU keeps searching; the
    /// first answer is used (the card's then installed as from a lane).
    fn inductive_raced(
        &mut self,
        cube: &[Lit],
        assump: &LitVec,
        constraint: &[LitVec],
    ) -> Option<bool> {
        use std::sync::atomic::Ordering;
        let raw_assump: Vec<u32> = assump.iter().map(|l| u32::from(*l)).collect();
        let raw_temps: Vec<Vec<u32>> = constraint
            .iter()
            .map(|c| c.iter().map(|l| u32::from(*l)).collect())
            .collect();
        if crate::fastlane::card_busy() {
            let t0 = std::time::Instant::now();
            let sat = self.dcs.solve_with_constraint(assump, constraint.to_vec());
            crate::fastlane::RACE_BUSY.fetch_add(1, Ordering::Relaxed);
            crate::fastlane::RACE_BUSY_NS
                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            return Some(!sat);
        }
        let frame = self.dcs.fastlane_frame.unwrap_or(0);
        let submitted = crate::fastlane::with_pool(|pool| {
            pool.replicas[0].submit_frame(&raw_assump, &raw_temps, frame)
        })
        .unwrap_or(false);
        if !submitted {
            return None;
        }
        let depth = crate::fastlane::frame_depth(frame);
        crate::fastlane::DEPTH_SUB[depth].fetch_add(1, Ordering::Relaxed);
        crate::fastlane::RACING.store(true, Ordering::Relaxed);
        let cpu = self
            .dcs
            .solve_with_constraint_raced(assump, constraint.to_vec());
        crate::fastlane::RACING.store(false, Ordering::Relaxed);
        if let Some(sat) = cpu {
            crate::fastlane::card_abort();
            crate::fastlane::note_race(false);
            return Some(!sat);
        }
        crate::fastlane::note_race(true);
        crate::fastlane::DEPTH_WIN[depth].fetch_add(1, Ordering::Relaxed);
        crate::fastlane::ROUTED.fetch_add(1, Ordering::Relaxed);
        let phase = crate::inductor::current_phase() as usize;
        crate::fastlane::PHASE_COUNTS[phase.min(7)].fetch_add(1, Ordering::Relaxed);
        let result = crate::fastlane::with_pool(|pool| pool.replicas[0].collect_frame())
            .unwrap_or(crate::fastlane::LaneResult::Unknown);
        // A raced SOLVE has no budget: UNKNOWN means the lane overflowed.
        if matches!(result, crate::fastlane::LaneResult::Unknown) {
            crate::fastlane::request_reload();
        }
        match self.install_lane_result(cube, assump, constraint, result) {
            Some(blocked) => Some(blocked),
            // unusable card answer: the CPU solves the query after all
            None => Some(!self.dcs.solve_with_constraint(assump, constraint.to_vec())),
        }
    }

    /// Push split: the card query for `cube`, as
    /// `inductive_with_constrain(cube, false, vec![])` asks it (raw
    /// assumption literals, no temporary clauses).
    pub fn push_query_raw(&self, cube: &[Lit]) -> Vec<u32> {
        self.ts.lits_next(cube).iter().map(|l| u32::from(*l)).collect()
    }

    /// Install the card's answer to `push_query_raw(cube)` as if this
    /// solver had answered it; `None` if it cannot be used.
    pub fn install_push_result(
        &mut self,
        cube: &[Lit],
        result: crate::fastlane::LaneResult,
    ) -> Option<bool> {
        let assump = self.ts.lits_next(cube);
        self.install_lane_result(cube, &assump, &[], result)
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
        let assumptions = self.ts.lits_next(cube);
        let domain = if crate::gipsat::query::local_incremental_domain_enabled() {
            self.dcs.incremental_local_domain(
                assumptions
                    .iter()
                    .chain(constraint.iter().flatten())
                    .map(|lit| lit.var()),
            )
        } else {
            (0..self.dcs.num_var()).map(Var::from).collect()
        };
        IncrementalQuery {
            frame: self.dcs.accel_level,
            assumptions,
            constraints: constraint,
            domain,
            budget: QueryBudget {
                decisions: crate::accel::cdcl_host::active_decision_budget(),
                conflicts: crate::accel::cdcl_host::active_conflict_budget(),
                restarts: None,
            },
            keep_learnts: crate::accel::cdcl_host::active_keep_learnts(),
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
