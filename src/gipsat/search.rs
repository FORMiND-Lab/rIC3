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
