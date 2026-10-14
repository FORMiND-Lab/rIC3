use crate::ic3::IC3;
use giputils::statistic::{Average, CountedDuration, RunningTime, SuccessRate};
use logicrs::Var;
use std::{cmp::Ordering, fmt::Debug, time::Duration};

#[derive(Debug, Clone, Default)]
pub struct Block {
    pub overall_time: Duration,
    pub get_bad_time: CountedDuration,
    pub blocked_time: CountedDuration,
    pub get_pred_time: Duration,
    pub mic_time: Duration,
    pub push_time: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct Propagate {
    pub overall_time: Duration,
    pub push_inf_time: Duration,
    /// First push check of each live lemma (the batchable part).
    pub check0: CountedDuration,
    /// Push checks retried after a CTP lemma was added.
    pub check_retry: CountedDuration,
    /// CTP generalization: get_pred, the CTP inductive check, mic, add_lemma.
    pub ctp_gen: CountedDuration,
    /// add_lemma of pushed lemmas.
    pub add_pushed: CountedDuration,
    /// (frame, pass) groups with at least one live lemma, and their lemmas.
    pub groups: usize,
    pub group_lemmas: usize,
    pub max_group: usize,
}

#[allow(unused)]
#[derive(Debug, Default)]
pub struct Statistic {
    pub time: RunningTime,

    pub num_mic: usize,
    pub avg_mic_cube_len: Average,
    pub avg_po_cube_len: Average,
    pub mic_drop: SuccessRate,
    pub num_down: usize,
    pub num_down_sat: usize,

    pub ctp: SuccessRate,

    pub block: Block,

    pub propagate: Propagate,

    pub xor_gen: SuccessRate,
    pub num_auxiliary_var: usize,

    pub test: SuccessRate,
}

impl IC3 {
    pub fn inn_cmp(&self, a: impl Into<Var>, b: impl Into<Var>) -> Ordering {
        let (a, b) = (a.into(), b.into());
        self.ts_top_lv[a]
            .cmp(&self.ts_top_lv[b])
            .then_with(|| self.activity.cmp(a, b))
    }
}
