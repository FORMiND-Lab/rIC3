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
}

struct PendingPushBatch {
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

    fn finish(&mut self) -> Vec<CachedPushInquiry> {
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
                result_cacheable(&result, retain_sat_results()).then_some(CachedPushInquiry {
                    frame_idx,
                    lemma,
                    result,
                    cached_at: 0,
                })
            })
            .collect();
        crate::accel::cdcl_host::note_active_push_prefetch_harvest(
            inquiries.len(),
            wall_ns,
            join_ns,
        );
        inquiries
    }
}

impl Drop for PendingPushBatch {
    fn drop(&mut self) {
        if self.handle.is_some() {
            let _ = self.finish();
        }
    }
}

#[derive(Default)]
pub(super) struct PushPrefetchCache {
    ready: Vec<CachedPushInquiry>,
    pending: Option<PendingPushBatch>,
    epoch: u64,
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

    pub(super) fn begin_pass(&mut self) {
        self.epoch = self.epoch.saturating_add(1);
        let before = self.ready.len();
        let epoch = self.epoch;
        self.ready
            .retain(|entry| epoch.saturating_sub(entry.cached_at) <= Self::retention_passes());
        crate::accel::cdcl_host::note_active_push_prefetch_evicted(
            before.saturating_sub(self.ready.len()),
        );
        self.harvest_ready();
    }

    fn insert(&mut self, inquiries: Vec<CachedPushInquiry>) {
        for mut inquiry in inquiries {
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

    fn harvest_ready(&mut self) {
        if self
            .pending
            .as_ref()
            .is_some_and(PendingPushBatch::is_finished)
        {
            let mut pending = self.pending.take().unwrap();
            let inquiries = pending.finish();
            self.insert(inquiries);
        }
    }

    pub(super) fn take(
        &mut self,
        frame_idx: usize,
        lemma: &LitOrdVec,
    ) -> Option<IncrementalResult> {
        self.harvest_ready();
        let index = self
            .ready
            .iter()
            .position(|entry| entry.frame_idx == frame_idx && &entry.lemma == lemma)?;
        Some(self.ready.swap_remove(index).result)
    }

    pub(super) fn busy(&mut self) -> bool {
        self.harvest_ready();
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
            keys,
            handle: Some(handle),
            launched_at,
        });
    }

    pub(super) fn finish(&mut self) {
        if let Some(mut pending) = self.pending.take() {
            let inquiries = pending.finish();
            self.insert(inquiries);
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
    use super::{result_cacheable, CachedPushInquiry, PushPrefetchCache};
    use crate::{accel::cdcl::UnknownReason, gipsat::IncrementalResult};
    use logicrs::{Lit, LitOrdVec, LitVec, Var};

    fn inquiry(frame_idx: usize, lit: Lit, result: IncrementalResult) -> CachedPushInquiry {
        CachedPushInquiry {
            frame_idx,
            lemma: LitOrdVec::new(LitVec::from([lit])),
            result,
            cached_at: 0,
        }
    }

    #[test]
    fn cache_is_exact_bounded_and_conclusive_only() {
        let a = Lit::new(Var::from(0), true);
        let b = Lit::new(Var::from(1), true);
        let mut cache = PushPrefetchCache::default();
        cache.begin_pass();
        cache.insert(vec![
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
        ]);
        assert!(cache.take(1, &LitOrdVec::new(LitVec::from([a]))).is_none());
        assert!(cache.take(2, &LitOrdVec::new(LitVec::from([b]))).is_none());
        assert!(matches!(
            cache.take(2, &LitOrdVec::new(LitVec::from([a]))),
            Some(IncrementalResult::Sat { .. })
        ));

        cache.insert(vec![inquiry(
            3,
            a,
            IncrementalResult::Unsat {
                core: LitVec::from([a]),
                used_constraints: false,
            },
        )]);
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
}
