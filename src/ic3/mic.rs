use super::IC3;
use crate::{ic3::IC3Config, transys::TransysIf};
use giputils::hash::GHashSet;
use log::trace;
use logicrs::{Lit, LitOrdVec, LitVec, satif::Satif};
use rand::{RngExt, seq::SliceRandom};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
pub struct DropVarParameter {
    pub limit: usize,
    max: usize,
    level: usize,
}

impl DropVarParameter {
    #[inline]
    pub fn new(limit: usize, max: usize, level: usize) -> Self {
        Self { limit, max, level }
    }

    fn sub_level(self) -> Self {
        Self {
            limit: self.limit,
            max: self.max,
            level: self.level - 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum MicType {
    #[allow(unused)]
    NoMic,
    DropVar(DropVarParameter),
}

impl MicType {
    pub fn from_config(cfg: &IC3Config) -> Self {
        let p = if cfg.ctg {
            DropVarParameter {
                limit: cfg.ctg_limit,
                max: cfg.ctg_max,
                level: 1,
            }
        } else {
            DropVarParameter::default()
        };
        MicType::DropVar(p)
    }
}

impl IC3 {
    fn down(
        &mut self,
        frame: usize,
        cube: &LitVec,
        keep: &GHashSet<Lit>,
        full: &LitVec,
        constraint: &[LitVec],
        cex: &mut Vec<(LitOrdVec, LitOrdVec)>,
    ) -> Option<LitVec> {
        let mut cube = cube.clone();
        self.statistic.num_down += 1;
        loop {
            if self.tsctx.cube_subsume_init(&cube) {
                return None;
            }
            let lemma = LitOrdVec::new(cube.clone());
            if cex
                .iter()
                .any(|(s, t)| !lemma.subsume(s) && lemma.subsume(t))
            {
                return None;
            }
            self.statistic.num_down_sat += 1;

            // Ask the card before the solver, because asking after saves
            // nothing. It returns the unsat core -- the subset of the
            // next-state assumptions that conflicts -- which is exactly what
            // `inductive_core()` reconstructs and what becomes the lemma.
            //
            // Sound to use directly. The card holds a subset of this solver's
            // clauses and none of the per-query constraints, including the
            // `strengthen` clause, so it is strictly weaker: a conflict it
            // derives is a conflict here. A core that subsumes the initial
            // states is handed back rather than repaired, since
            // `inductive_core` has its own handling for that and duplicating
            // it here would be a second implementation of a subtle rule.
            if crate::accel::core_offload() && crate::accel::ready() {
                // The lemmas the frame has gained are not visible to the card
                // until its occurrence index is rebuilt, and that used to happen
                // in the shadow block. Gating the shadow took it with it, and
                // the card went on propagating over a stale index: cores fell
                // from 84 to 10 while the run got ten times faster, which is
                // the shape of an engine that has stopped seeing the lemmas
                // rather than one that got quicker.
                crate::accel::sync_index();
                let assump = self.tsctx.lits_next(&cube);
                let raw: Vec<u32> = assump.iter().map(|l| Into::<u32>::into(*l)).collect();
                // No domain restriction here.
                //
                // The solver's domain is the transitive closure `enable_local`
                // builds, and it is built inside `solve()` -- after this point.
                // Sending the surface set instead, the assumptions and the cube,
                // gave the card a domain so small it propagated almost nothing:
                // 5 cores from 2,892 asks with every constraint accepted.
                //
                // Dropping the restriction is sound in the direction that
                // matters. It only lets propagation reach further over clauses
                // that are all real constraints, so a conflict it derives is
                // still a conflict for the query; the domain can lose
                // implications, never invent them.
                crate::accel::set_domain(&[]);
                // The clauses this query carries. `down` calls `blocked` with
                // `.with_strengthen()`, which adds `!cube`, plus whatever the
                // caller passed; the card needs both or it is weaker than the
                // solver on exactly these queries.
                let mut flat: Vec<u32> = Vec::new();
                {
                    let mut push = |c: &LitVec| {
                        flat.push(c.len() as u32);
                        for l in c.iter() {
                            flat.push(Into::<u32>::into(*l));
                        }
                    };
                    push(&LitVec::from_iter(cube.iter().map(|l| !*l)));
                    for c in constraint.iter() {
                        push(c);
                    }
                }
                let mut got: Vec<u32> = Vec::new();
                let lvl = crate::accel::level_arg((frame - 1) as u32);
                // One round trip where there were four. The card installs the
                // constraint, answers, minimises, and takes the constraint
                // back out itself, so the drop below is only for the
                // bitstreams that predate the fused mode.
                let got_core = if crate::accel::have_down() {
                    crate::accel::down(&flat, &raw, lvl, &mut got).is_some()
                } else {
                    crate::accel::set_constraint(&flat);
                    let r = crate::accel::core(&raw, lvl, &mut got).is_some();
                    crate::accel::set_constraint(&[]);
                    r
                };
                if got_core {
                    let inset: std::collections::HashSet<u32> = got.into_iter().collect();
                    let mut ans = LitVec::new();
                    for &l in cube.iter() {
                        if inset.contains(&Into::<u32>::into(self.tsctx.next(l))) {
                            ans.push(l);
                        }
                    }
                    // Only when the card actually generalized something.
                    //
                    // A core equal to the cube is sound and IC3 will take it,
                    // but it is a lemma no stronger than what was asked about,
                    // and taking it skips the solver's own `down`, which would
                    // have returned a smaller one. Measured on
                    // Problem03_label51, 8,153 such cores went in at 2.00
                    // literals and came back at 2.00, and IC3 then called mic
                    // 24,514 times against the CPU-only run's 2,157 -- an
                    // eleven-fold increase that cost more than everything the
                    // card saved.
                    //
                    // `INDUCTOR_CORE_GAIN` is how many literals the card has to
                    // remove to be worth believing. 0 restores the old
                    // behaviour of taking every core.
                    let gain = cube.len().saturating_sub(ans.len());
                    if !ans.is_empty()
                        && gain >= crate::accel::core_gain()
                        && !self.tsctx.cube_subsume_init(&ans)
                    {
                        crate::accel::CORE_USED
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return Some(ans);
                    }
                    crate::accel::CORE_THIN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }

            if self
                .blocked(frame, &cube)
                .in_phase(inductor_trace::Phase::Gen)
                .with_act_order(false)
                .with_strengthen()
                .with_constraint(constraint)
                .check()
            {
                return Some(self.solvers[frame - 1].inductive_core().unwrap());
            }
            let mut ret = false;
            let mut cube_new = LitVec::new();
            for lit in cube {
                if keep.contains(&lit) {
                    if let Some(true) = self.solvers[frame - 1].sat_value(lit) {
                        cube_new.push(lit);
                    } else {
                        ret = true;
                        break;
                    }
                } else if let Some(true) = self.solvers[frame - 1].sat_value(lit)
                    && !self.solvers[frame - 1].flip_to_none(lit.var())
                {
                    cube_new.push(lit);
                }
            }
            cube = cube_new;
            let mut s = LitVec::new();
            let mut t = LitVec::new();
            for l in full.iter() {
                if let Some(v) = self.solvers[frame - 1].sat_value(*l)
                    && self.solvers[frame - 1].flip_to_none(l.var())
                {
                    s.push(l.not_if(!v));
                }
                if let Some(v) = self.solvers[frame - 1].sat_value(self.tsctx.next(*l)) {
                    t.push(l.not_if(!v));
                }
            }
            cex.push((LitOrdVec::new(s), LitOrdVec::new(t)));
            if ret {
                return None;
            }
        }
    }

    fn ctg_down(
        &mut self,
        frame: usize,
        cube: &LitVec,
        keep: &GHashSet<Lit>,
        full: &LitVec,
        parameter: DropVarParameter,
    ) -> Option<LitVec> {
        let mut cube = cube.clone();
        self.statistic.num_down += 1;
        let mut ctg = 0;
        loop {
            if self.tsctx.cube_subsume_init(&cube) {
                return None;
            }
            self.statistic.num_down_sat += 1;
            if self
                .blocked(frame, &cube)
                .in_phase(inductor_trace::Phase::Gen)
                .with_act_order(false)
                .with_strengthen()
                .check()
            {
                return Some(self.solvers[frame - 1].inductive_core().unwrap());
            }
            for lit in cube.iter() {
                if keep.contains(lit) && !self.solvers[frame - 1].sat_value(*lit).is_some_and(|v| v)
                {
                    return None;
                }
            }
            let (model, _) = self.get_pred(frame, false);
            let cex_set: GHashSet<Lit> = GHashSet::from_iter(model.iter().cloned());
            // for lit in cube.iter() {
            //     if keep.contains(lit) && !cex_set.contains(lit) {
            //         return None;
            //     }
            // }
            if ctg < parameter.max
                && frame > 1
                && !self.tsctx.cube_subsume_init(&model)
                && self.trivial_block(
                    frame - 1,
                    LitOrdVec::new(model.clone()),
                    &[!full.clone()],
                    parameter.sub_level(),
                )
            {
                ctg += 1;
                continue;
            }
            ctg = 0;
            let mut cube_new = LitVec::new();
            for lit in cube {
                if cex_set.contains(&lit) {
                    cube_new.push(lit);
                } else if keep.contains(&lit) {
                    return None;
                }
            }
            cube = cube_new;
        }
    }

    fn handle_down_success(
        &mut self,
        _frame: usize,
        cube: LitVec,
        i: usize,
        mut new_cube: LitVec,
    ) -> (LitVec, usize) {
        new_cube = cube
            .iter()
            .filter(|l| new_cube.contains(l))
            .cloned()
            .collect();
        let new_i = new_cube
            .iter()
            .position(|l| !(cube[0..i]).contains(l))
            .unwrap_or(new_cube.len());
        if new_i < new_cube.len() {
            assert!(!(cube[0..=i]).contains(&new_cube[new_i]))
        }
        (new_cube, new_i)
    }

    fn mic_by_drop_var(
        &mut self,
        frame: usize,
        mut cube: LitVec,
        constraint: &[LitVec],
        parameter: DropVarParameter,
    ) -> LitVec {
        let start = Instant::now();
        let _op = crate::inductor::macro_scope(inductor_trace::Phase::Gen, frame);
        if parameter.level == 0 {
            self.solvers[frame - 1].set_domain(
                self.tsctx
                    .lits_next(&cube)
                    .iter()
                    .copied()
                    .chain(cube.iter().copied()),
            );
        }
        self.statistic.avg_mic_cube_len += cube.len();
        self.statistic.num_mic += 1;
        // How many independent queries one generalization could issue at once.
        // Every speculative drop is a subset of this cube, so they share the
        // domain set just above and the clause set, which is fixed for the
        // duration of a mic -- the two conditions RUN_BATCH needs and the two
        // that the earlier per-query batching attempt could not meet.
        crate::accel::note_mic(cube.len());
        let mut cex = Vec::new();
        if self.rng.random_bool(0.2) {
            cube.shuffle(&mut self.rng);
        } else {
            self.activity.sort_by_activity(&mut cube, true);
        }
        if self.cfg.parent_lemma
            && let Some(parent) = self.frame.parent_lemma(&cube, frame)
        {
            let parent = GHashSet::from_iter(parent);
            cube.sort_by_key(|x| parent.contains(x));
        }
        let mut keep = GHashSet::new();

        // Let the card run the drop loop first.
        //
        // It tries the same removals this loop does, with propagation instead
        // of the solver, and returns a sub-cube that still blocks. Weaker: on
        // the satisfiable branch the solver shrinks the cube from its model
        // and the card has no model, so it keeps the literal. But every
        // literal it did drop was dropped because a conflict survived without
        // it, so what comes back is a sound starting point -- and the loop
        // below still runs, so nothing this misses is lost.
        //
        // One call for the whole loop. The assumptions and the constraint are
        // both derived from the cube and both change every time it shrinks,
        // which is why this could not be a batch of queries prepared here.
        if crate::accel::mic_offload() && crate::accel::ready() && crate::accel::have_mic() {
            crate::accel::sync_index();
            let mut pairs: Vec<u32> = Vec::with_capacity(cube.len() * 2);
            for l in cube.iter() {
                pairs.push(Into::<u32>::into(*l));
                pairs.push(Into::<u32>::into(self.tsctx.next(*l)));
            }
            let mut extra: Vec<u32> = Vec::new();
            for c in constraint.iter() {
                extra.push(c.len() as u32);
                for l in c.iter() {
                    extra.push(Into::<u32>::into(*l));
                }
            }
            let mut got: Vec<u32> = Vec::new();
            let lvl = crate::accel::level_arg((frame - 1) as u32);
            if crate::accel::mic(&extra, &pairs, lvl, &mut got).is_some()
                && !got.is_empty()
                && got.len() < cube.len()
            {
                let inset: std::collections::HashSet<u32> = got.into_iter().collect();
                let mut shrunk = LitVec::new();
                for l in cube.iter() {
                    if inset.contains(&Into::<u32>::into(*l)) {
                        shrunk.push(*l);
                    }
                }
                // A cube that subsumes the initial states is not a lemma. The
                // card does not test that, and `down` would have handed such a
                // cube back rather than repaired it.
                if !shrunk.is_empty() && !self.tsctx.cube_subsume_init(&shrunk) {
                    crate::accel::MIC_TAKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    cube = shrunk;
                }
            }
        }

        let mut i = 0;
        while i < cube.len() {
            if keep.contains(&cube[i]) {
                i += 1;
                continue;
            }
            let mut removed_cube = cube.clone();
            removed_cube.remove(i);
            let mic = if parameter.level == 0 {
                self.down(frame, &removed_cube, &keep, &cube, constraint, &mut cex)
            } else {
                self.ctg_down(frame, &removed_cube, &keep, &cube, parameter)
            };
            if let Some(new_cube) = mic {
                self.statistic.mic_drop.success();
                (cube, i) = self.handle_down_success(frame, cube, i, new_cube);
                if parameter.level == 0 {
                    self.solvers[frame - 1].unset_domain();
                    self.solvers[frame - 1].set_domain(
                        self.tsctx
                            .lits_next(&cube)
                            .iter()
                            .copied()
                            .chain(cube.iter().copied()),
                    );
                }
            } else {
                self.statistic.mic_drop.fail();
                keep.insert(cube[i]);
                i += 1;
            }
        }
        if parameter.level == 0 {
            self.solvers[frame - 1].unset_domain();
        }
        self.activity.bump_cube_activity(&cube);
        self.statistic.block.mic_time += start.elapsed();
        cube
    }

    pub(super) fn mic(
        &mut self,
        frame: usize,
        cube: LitVec,
        constraint: &[LitVec],
        mic_type: MicType,
    ) -> LitVec {
        let mic_olen = cube.len();
        let r = match mic_type {
            MicType::NoMic => cube,
            MicType::DropVar(parameter) => self.mic_by_drop_var(frame, cube, constraint, parameter),
        };
        trace!("mic from {} to {} len", mic_olen, r.len());
        r
    }
}
