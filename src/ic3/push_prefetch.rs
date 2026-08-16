use crate::{
    accel::cdcl::UnknownReason,
    gipsat::{DagCnfSolver, IncrementalQuery, IncrementalResult},
};
use logicrs::LitOrdVec;
use std::time::Instant;

fn retain_sat_results() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_PUSH_PREFETCH_SAT")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
    })
}

fn result_cacheable(result: &IncrementalResult, retain_sat: bool) -> bool {
    matches!(result, IncrementalResult::Unsat { .. })
        || retain_sat && matches!(result, IncrementalResult::Sat { .. })
}

#[derive(Clone)]
struct CachedPushInquiry {
    frame_idx: usize,
    lemma: LitOrdVec,
    result: IncrementalResult,
    cached_at: u64,
    batch_id: u64,
}

pub(super) struct PrefetchedPushInquiry {
    pub(super) result: IncrementalResult,
    pub(super) batch_id: u64,
}

struct FinishedPushBatch {
    batch_id: u64,
    n_queries: usize,
    inquiries: Vec<CachedPushInquiry>,
}

struct PendingPushBatch {
    batch_id: u64,
    keys: Vec<(usize, LitOrdVec)>,
    handle: Option<std::thread::JoinHandle<Vec<IncrementalResult>>>,
    launched_at: Instant,
}

impl PendingPushBatch {
    fn is_finished(&self) -> bool {
        self.handle
            .as_ref()
            .is_none_or(std::thread::JoinHandle::is_finished)
    }

    fn finish(&mut self) -> FinishedPushBatch {
        let join_start = Instant::now();
        let n_queries = self.keys.len();
        let results = self
            .handle
            .take()
            .and_then(|handle| handle.join().ok())
            .unwrap_or_else(|| {
                vec![IncrementalResult::Unknown(UnknownReason::BackendError); n_queries]
            });
        let join_ns = join_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let wall_ns = self.launched_at.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let inquiries: Vec<_> = std::mem::take(&mut self.keys)
            .into_iter()
            .zip(results)
            .filter_map(|((frame_idx, lemma), result)| {
                result_cacheable(&result, retain_sat_results()).then(|| {
                    crate::accel::cdcl_host::note_active_push_prefetch_ready_length(lemma.len());
                    CachedPushInquiry {
                        frame_idx,
                        lemma,
                        result,
                        cached_at: 0,
                        batch_id: self.batch_id,
                    }
                })
            })
            .collect();
        crate::accel::cdcl_host::note_active_push_prefetch_harvest(
            inquiries.len(),
            wall_ns,
            join_ns,
        );
        FinishedPushBatch {
            batch_id: self.batch_id,
            n_queries,
            inquiries,
        }
    }
}

impl Drop for PendingPushBatch {
    fn drop(&mut self) {
        if self.handle.is_some() {
            let _ = self.finish();
        }
    }
}

#[derive(Debug)]
struct PushBatchStat {
    batch_id: u64,
    n_queries: usize,
    ready: usize,
    harvested_at: u64,
    exact_hits: usize,
    used: usize,
    rejected: usize,
}

#[derive(Default)]
pub(super) struct PushPrefetchCache {
    ready: Vec<CachedPushInquiry>,
    pending: Option<PendingPushBatch>,
    batch_stats: Vec<PushBatchStat>,
    epoch: u64,
    next_batch_id: u64,
    next_probe_epoch: u64,
}

impl PushPrefetchCache {
    pub(super) fn launch_window() -> usize {
        use std::sync::OnceLock;
        static WINDOW: OnceLock<usize> = OnceLock::new();
        *WINDOW.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_PUSH_PREFETCH_WINDOW")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(512)
                .clamp(1, 4096)
        })
    }

    pub(super) fn max_lemma_len() -> usize {
        use std::sync::OnceLock;
        static MAX_LEN: OnceLock<usize> = OnceLock::new();
        *MAX_LEN.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_PUSH_PREFETCH_MAX_LEMMA")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0)
                .min(4096)
        })
    }

    fn retention_passes() -> u64 {
        use std::sync::OnceLock;
        static PASSES: OnceLock<u64> = OnceLock::new();
        *PASSES.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_PUSH_PREFETCH_PASSES")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(4)
                .clamp(1, 64)
        })
    }

    fn adaptive_enabled() -> bool {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_PUSH_PREFETCH_ADAPTIVE")
                .ok()
                .is_none_or(|value| !matches!(value.as_str(), "0" | "false" | "off"))
        })
    }

    fn admission_window() -> usize {
        use std::sync::OnceLock;
        static WINDOW: OnceLock<usize> = OnceLock::new();
        *WINDOW.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_PUSH_PREFETCH_ADAPTIVE_WINDOW")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(4)
                .clamp(1, 32)
        })
    }

    fn admission_min_batches() -> usize {
        use std::sync::OnceLock;
        static MIN_BATCHES: OnceLock<usize> = OnceLock::new();
        *MIN_BATCHES.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_PUSH_PREFETCH_MIN_PROBES")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(2)
                .clamp(1, 16)
        })
    }

    fn admission_min_use_pct() -> usize {
        use std::sync::OnceLock;
        static MIN_USE_PCT: OnceLock<usize> = OnceLock::new();
        *MIN_USE_PCT.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_PUSH_PREFETCH_MIN_USE_PCT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0)
                .clamp(0, 100)
        })
    }

    fn reprobe_passes() -> u64 {
        use std::sync::OnceLock;
        static PASSES: OnceLock<u64> = OnceLock::new();
        *PASSES.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_PUSH_PREFETCH_REPROBE_PASSES")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8)
                .clamp(1, 256)
        })
    }

    fn evaluated_admission_totals(
        stats: &[PushBatchStat],
        epoch: u64,
        window: usize,
    ) -> (usize, usize, usize) {
        let mut batches = 0usize;
        let mut queries = 0usize;
        let mut used = 0usize;
        for stat in stats
            .iter()
            .rev()
            .filter(|stat| stat.harvested_at < epoch)
            .take(window)
        {
            batches += 1;
            queries = queries.saturating_add(stat.n_queries);
            used = used.saturating_add(stat.used);
        }
        (batches, queries, used)
    }

    pub(super) fn should_launch(&mut self) -> bool {
        if !Self::adaptive_enabled() {
            return true;
        }
        self.should_launch_with_policy(
            Self::admission_window(),
            Self::admission_min_batches(),
            Self::admission_min_use_pct(),
            Self::reprobe_passes(),
        )
    }

    fn should_launch_with_policy(
        &mut self,
        window: usize,
        min_batches: usize,
        min_use_pct: usize,
        reprobe_passes: u64,
    ) -> bool {
        let (batches, queries, used) =
            Self::evaluated_admission_totals(&self.batch_stats, self.epoch, window);
        if batches < min_batches || queries == 0 {
            crate::accel::cdcl_host::note_active_push_prefetch_admission(
                true, false, queries, used,
            );
            return true;
        }
        let profitable = if min_use_pct == 0 {
            used != 0
        } else {
            (used as u128) * 100 >= (queries as u128) * (min_use_pct as u128)
        };
        if profitable {
            self.next_probe_epoch = 0;
            crate::accel::cdcl_host::note_active_push_prefetch_admission(
                true, false, queries, used,
            );
            return true;
        }
        if self.next_probe_epoch == 0 {
            self.next_probe_epoch = self.epoch.saturating_add(reprobe_passes);
            crate::accel::cdcl_host::note_active_push_prefetch_admission(
                false, false, queries, used,
            );
            return false;
        }
        let reprobe = self.epoch >= self.next_probe_epoch;
        if reprobe {
            self.next_probe_epoch = self.epoch.saturating_add(reprobe_passes);
        }
        crate::accel::cdcl_host::note_active_push_prefetch_admission(
            reprobe, reprobe, queries, used,
        );
        reprobe
    }

    pub(super) fn begin_pass(&mut self) {
        self.epoch = self.epoch.saturating_add(1);
        let before = self.ready.len();
        let epoch = self.epoch;
        self.ready
            .retain(|entry| epoch.saturating_sub(entry.cached_at) <= Self::retention_passes());
        crate::accel::cdcl_host::note_active_push_prefetch_evicted(
            before.saturating_sub(self.ready.len()),
        );
        self.harvest_ready(self.epoch.saturating_sub(1));
    }

    fn insert(&mut self, finished: FinishedPushBatch, harvested_at: u64) {
        let ready = finished.inquiries.len();
        self.batch_stats.push(PushBatchStat {
            batch_id: finished.batch_id,
            n_queries: finished.n_queries,
            ready,
            harvested_at,
            exact_hits: 0,
            used: 0,
            rejected: 0,
        });
        if self.batch_stats.len() > 64 {
            self.batch_stats.remove(0);
        }
        for mut inquiry in finished.inquiries {
            if !result_cacheable(&inquiry.result, true) {
                continue;
            }
            inquiry.cached_at = self.epoch;
            if let Some(index) = self.ready.iter().position(|entry| {
                entry.frame_idx == inquiry.frame_idx && entry.lemma == inquiry.lemma
            }) {
                self.ready.swap_remove(index);
                crate::accel::cdcl_host::note_active_push_prefetch_evicted(1);
            }
            self.ready.push(inquiry);
        }
    }

    fn harvest_ready(&mut self, harvested_at: u64) {
        if self
            .pending
            .as_ref()
            .is_some_and(PendingPushBatch::is_finished)
        {
            let mut pending = self.pending.take().unwrap();
            let finished = pending.finish();
            self.insert(finished, harvested_at);
        }
    }

    pub(super) fn take(
        &mut self,
        frame_idx: usize,
        lemma: &LitOrdVec,
    ) -> Option<PrefetchedPushInquiry> {
        self.harvest_ready(self.epoch);
        let index = self
            .ready
            .iter()
            .position(|entry| entry.frame_idx == frame_idx && &entry.lemma == lemma)?;
        let entry = self.ready.swap_remove(index);
        if let Some(stat) = self
            .batch_stats
            .iter_mut()
            .find(|stat| stat.batch_id == entry.batch_id)
        {
            stat.exact_hits = stat.exact_hits.saturating_add(1);
        }
        Some(PrefetchedPushInquiry {
            result: entry.result,
            batch_id: entry.batch_id,
        })
    }

    pub(super) fn note_validation(&mut self, batch_id: u64, accepted: bool) {
        if let Some(stat) = self
            .batch_stats
            .iter_mut()
            .find(|stat| stat.batch_id == batch_id)
        {
            if accepted {
                stat.used = stat.used.saturating_add(1);
            } else {
                stat.rejected = stat.rejected.saturating_add(1);
            }
        }
    }

    pub(super) fn busy(&mut self) -> bool {
        self.harvest_ready(self.epoch);
        self.pending.is_some()
    }

    pub(super) fn start(
        &mut self,
        keys: Vec<(usize, LitOrdVec)>,
        owned_solvers: Vec<DagCnfSolver>,
        owned_requests: Vec<(usize, IncrementalQuery)>,
        prepare_ns: u64,
    ) {
        debug_assert!(self.pending.is_none());
        self.next_batch_id = self.next_batch_id.saturating_add(1);
        let batch_id = self.next_batch_id;
        for (_, lemma) in &keys {
            crate::accel::cdcl_host::note_active_push_prefetch_submit_length(lemma.len());
        }
        let launched_at = Instant::now();
        let handle = std::thread::spawn(move || {
            let requests = owned_requests
                .into_iter()
                .map(|(solver_index, query)| (&owned_solvers[solver_index], query))
                .collect();
            crate::accel::cdcl_host::solve_active_batch(requests)
        });
        crate::accel::cdcl_host::note_active_push_prefetch_launch(keys.len(), prepare_ns);
        self.pending = Some(PendingPushBatch {
            batch_id,
            keys,
            handle: Some(handle),
            launched_at,
        });
    }

    pub(super) fn finish(&mut self) {
        if let Some(mut pending) = self.pending.take() {
            let finished = pending.finish();
            self.insert(finished, self.epoch);
        }
        if !self.batch_stats.is_empty() {
            let summary = self
                .batch_stats
                .iter()
                .map(|stat| {
                    format!(
                        "{}:{}/{}/{}/{}/{}",
                        stat.batch_id,
                        stat.n_queries,
                        stat.ready,
                        stat.exact_hits,
                        stat.used,
                        stat.rejected,
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            eprintln!(
                "inductor-cdcl: active push prefetch batch id:queries/ready/hits/used/rejected {}",
                summary,
            );
            self.batch_stats.clear();
        }
        crate::accel::cdcl_host::note_active_push_prefetch_evicted(self.ready.len());
        self.ready.clear();
    }
}

impl Drop for PushPrefetchCache {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        result_cacheable, CachedPushInquiry, FinishedPushBatch, PushBatchStat, PushPrefetchCache,
    };
    use crate::{accel::cdcl::UnknownReason, gipsat::IncrementalResult};
    use logicrs::{Lit, LitOrdVec, LitVec, Var};

    fn inquiry(frame_idx: usize, lit: Lit, result: IncrementalResult) -> CachedPushInquiry {
        CachedPushInquiry {
            frame_idx,
            lemma: LitOrdVec::new(LitVec::from([lit])),
            result,
            cached_at: 0,
            batch_id: 1,
        }
    }

    fn finished(inquiries: Vec<CachedPushInquiry>) -> FinishedPushBatch {
        FinishedPushBatch {
            batch_id: 1,
            n_queries: inquiries.len(),
            inquiries,
        }
    }

    #[test]
    fn cache_is_exact_bounded_and_conclusive_only() {
        let a = Lit::new(Var::from(0), true);
        let b = Lit::new(Var::from(1), true);
        let mut cache = PushPrefetchCache::default();
        cache.begin_pass();
        cache.insert(
            finished(vec![
                inquiry(
                    2,
                    a,
                    IncrementalResult::Sat {
                        model: LitVec::from([a]),
                    },
                ),
                inquiry(
                    2,
                    b,
                    IncrementalResult::Unknown(UnknownReason::ConflictBudget),
                ),
            ]),
            cache.epoch,
        );
        assert!(cache.take(1, &LitOrdVec::new(LitVec::from([a]))).is_none());
        assert!(cache.take(2, &LitOrdVec::new(LitVec::from([b]))).is_none());
        assert!(matches!(
            cache
                .take(2, &LitOrdVec::new(LitVec::from([a])))
                .map(|inquiry| inquiry.result),
            Some(IncrementalResult::Sat { .. })
        ));

        cache.insert(
            finished(vec![inquiry(
                3,
                a,
                IncrementalResult::Unsat {
                    core: LitVec::from([a]),
                    used_constraints: false,
                },
            )]),
            cache.epoch,
        );
        for _ in 0..=PushPrefetchCache::retention_passes() {
            cache.begin_pass();
        }
        assert!(cache.take(3, &LitOrdVec::new(LitVec::from([a]))).is_none());
    }

    #[test]
    fn production_prefetch_keeps_unsat_but_sat_is_explicit() {
        let a = Lit::new(Var::from(0), true);
        let sat = IncrementalResult::Sat {
            model: LitVec::from([a]),
        };
        let unsat = IncrementalResult::Unsat {
            core: LitVec::from([a]),
            used_constraints: false,
        };
        assert!(!result_cacheable(&sat, false));
        assert!(result_cacheable(&sat, true));
        assert!(result_cacheable(&unsat, false));
    }

    #[test]
    fn admission_uses_only_batches_that_had_a_consumption_pass() {
        let stats = vec![
            PushBatchStat {
                batch_id: 1,
                n_queries: 100,
                ready: 0,
                harvested_at: 2,
                exact_hits: 0,
                used: 0,
                rejected: 0,
            },
            PushBatchStat {
                batch_id: 2,
                n_queries: 100,
                ready: 8,
                harvested_at: 3,
                exact_hits: 8,
                used: 8,
                rejected: 0,
            },
        ];
        assert_eq!(
            PushPrefetchCache::evaluated_admission_totals(&stats, 3, 4),
            (1, 100, 0)
        );
        assert_eq!(
            PushPrefetchCache::evaluated_admission_totals(&stats, 4, 4),
            (2, 200, 8)
        );
    }

    #[test]
    fn admission_suppresses_zero_yield_and_reprobes_after_cooldown() {
        let mut cache = PushPrefetchCache::default();
        cache.epoch = 4;
        cache.batch_stats = vec![
            PushBatchStat {
                batch_id: 1,
                n_queries: 100,
                ready: 0,
                harvested_at: 1,
                exact_hits: 0,
                used: 0,
                rejected: 0,
            },
            PushBatchStat {
                batch_id: 2,
                n_queries: 100,
                ready: 0,
                harvested_at: 2,
                exact_hits: 0,
                used: 0,
                rejected: 0,
            },
        ];
        assert!(!cache.should_launch_with_policy(4, 2, 2, 8));
        assert_eq!(cache.next_probe_epoch, 12);
        cache.epoch = 11;
        assert!(!cache.should_launch_with_policy(4, 2, 2, 8));
        cache.epoch = 12;
        assert!(cache.should_launch_with_policy(4, 2, 2, 8));
        assert_eq!(cache.next_probe_epoch, 20);

        cache.batch_stats[1].used = 5;
        cache.epoch = 13;
        assert!(cache.should_launch_with_policy(4, 2, 2, 8));
        assert_eq!(cache.next_probe_epoch, 0);

        cache.batch_stats[1].used = 0;
        assert!(!cache.should_launch_with_policy(4, 2, 0, 8));
        cache.batch_stats[1].used = 1;
        assert!(cache.should_launch_with_policy(4, 2, 0, 8));
    }
}
