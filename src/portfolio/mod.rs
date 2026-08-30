mod lemma_mgr;
mod ui;

use self::lemma_mgr::LemmaMgr;
use self::ui::PortfolioUi;
use crate::config::{EngineConfig, EngineConfigBase, PreprocConfig, WorkerConfigs};
use crate::tracer::{Tracer, TracerIf};
use crate::transys::Transys;
use crate::transys::certify::{BlCex, BlProof, Restore};
use crate::ui::UiRenderer;
use crate::utils::{
    CertIpcRx, CertIpcTx, EngineCtrl, LemmaIpcRx, StateIpcTx, install_interrupt_handler,
};
use crate::{BlEngine, Engine, McBlCertificate, McResult, create_bl_engine, impl_config_deref};
use anyhow::Context;
use clap::{Args, Parser};
use giputils::TerminateCtrl;
use giputils::hash::GHashMap;
use giputils::logger::with_log_level;
use ipc_channel::ipc;
use ipc_channel::{
    TrySelectError,
    ipc::{IpcReceiverSet, IpcSelectionResult},
};
use log::{LevelFilter, info, set_max_level};
use logicrs::VarSymbols;
use nix::errno::Errno;
use nix::sys::wait::{WaitPidFlag, WaitStatus, waitpid};
use nix::unistd::Pid;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::iter;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{process::exit, sync::mpsc, thread::spawn};
use tempfile::TempDir;

#[derive(Args, Clone, Debug, Serialize, Deserialize)]
pub struct PortfolioConfig {
    #[command(flatten)]
    pub base: EngineConfigBase,

    #[command(flatten)]
    pub preproc: PreprocConfig,

    /// worker configuration
    #[arg(long = "config")]
    pub config: Option<String>,

    /// share lemma
    #[arg(long = "share-lemma")]
    pub share_lemma: bool,
}

impl_config_deref!(PortfolioConfig);

impl Default for PortfolioConfig {
    fn default() -> Self {
        let cfg = EngineConfig::parse_from(["", "portfolio"]);
        cfg.into_portfolio().unwrap()
    }
}

pub struct Portfolio {
    ots: Transys,
    ts: Transys,
    sym: VarSymbols,
    rst: Restore,
    cert: Option<McBlCertificate>,
    need_cert: bool,
    cfg: PortfolioConfig,
    engines: Vec<Worker>,
    running: GHashMap<Pid, usize>,
    winner_idx: Option<usize>,
    ctrl: Arc<EngineCtrl>,
    tracer: Tracer,
    ui: Option<PortfolioUi>,
    #[allow(unused)]
    temp_dir: TempDir,
    st_recv: IpcReceiverSet,
    // state tracer id to worker id
    stid_to_wid: GHashMap<u64, usize>,
}

struct Worker {
    name: String,
    cfg: EngineConfig,
    args: String,
    cert_tx: Option<CertIpcTx>,
    cert_rx: Option<CertIpcRx>,
    state: McResult,
}

impl Worker {
    fn run(
        &self,
        ts: &Transys,
        ots: &Transys,
        rst: &Restore,
        sym: &VarSymbols,
        tracer: StateIpcTx,
        extractor: Option<LemmaIpcRx>,
    ) -> ! {
        set_max_level(LevelFilter::Warn);
        let active_fpga = std::env::var_os("INDUCTOR_CDCL_ACTIVE").is_some();
        let trace_active = std::env::var_os("INDUCTOR_CDCL_TRACE_CSV").is_some()
            || std::env::var_os("INDUCTOR_CDCL_EXACT_REPLAY").is_some();
        let selected_worker = portfolio_worker_uses_fpga(
            &self.name,
            std::env::var("INDUCTOR_CDCL_PORTFOLIO_FPGA_WORKERS")
                .ok()
                .as_deref(),
            std::env::var("INDUCTOR_CDCL_BLOCK_FULL_ROOT")
                .ok()
                .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off")),
        );
        let selected_fpga = active_fpga && selected_worker;
        let selected_trace = trace_active && selected_worker;
        if active_fpga && !selected_fpga {
            // This runs in the freshly forked, single-threaded child before
            // the engine or accelerator client exists. The parent and sibling
            // workers retain their own environment and accelerator policy.
            unsafe {
                std::env::remove_var("INDUCTOR_CDCL_ACTIVE");
                std::env::remove_var("INDUCTOR_CDCL_SERVER");
            }
        }
        if trace_active && !selected_trace {
            // Portfolio trace capture uses one worker-scoped output file per
            // selected independent IC3 context. Nonselected workers must not
            // truncate the shared base path after fork.
            unsafe {
                std::env::remove_var("INDUCTOR_CDCL_TRACE_CSV");
                std::env::remove_var("INDUCTOR_CDCL_EXACT_REPLAY");
            }
        }
        if selected_fpga || selected_trace {
            // A portfolio time limit terminates children before their final
            // statistics report. Preserve the selected worker name in its
            // private post-fork environment so the lazy hardware connection
            // and trace writer can identify which policy crossed the route.
            unsafe {
                std::env::set_var("INDUCTOR_CDCL_PORTFOLIO_WORKER", &self.name);
            }
        }
        // We are already in the forked child, so take ownership of the inherited
        // in-memory model directly instead of reparsing or serializing it.
        let ts = unsafe { std::ptr::read(ts) };
        let sym = unsafe { std::ptr::read(sym) };
        let mut engine = create_bl_engine(self.cfg.clone(), ts, sym);
        engine.add_tracer(Box::new(tracer));
        extractor.map(|e| engine.set_extractor(Box::new(e)));
        // The portfolio parent stops losing or timed-out workers with SIGTERM.
        // Only FPGA-selected workers need a graceful path: their IC3 check
        // reports the compact per-process qualification counters before
        // returning, while all other workers retain the immediate termination
        // behavior. The ctrlc `termination` feature maps SIGTERM here.
        let _fpga_interrupt =
            (selected_fpga || selected_trace).then(|| install_interrupt_handler(engine.get_ctrl()));
        let res = engine.check();
        if let Some(cert_tx) = self.cert_tx.as_ref() {
            let certificate = match res {
                McResult::UNSAT => {
                    let cert = rst.restore_proof(engine.proof(), ots);
                    Some(McBlCertificate::UNSAT(cert))
                }
                McResult::SAT(_) => {
                    let cert = rst.restore_cex(&engine.cex());
                    Some(McBlCertificate::SAT(cert))
                }
                // A gracefully terminated FPGA worker has no certificate.
                McResult::Unknown(_) => None,
            };
            if let Some(certificate) = certificate {
                let _ = cert_tx.send(certificate);
            }
        };
        exit(0);
    }
}

const DEFAULT_FPGA_PORTFOLIO_WORKERS: [&str; 2] = ["ic3", "ic3_ctg_limit"];
const DEFAULT_FULL_ROOT_PORTFOLIO_WORKER: &str = "ic3_abs_all";

fn portfolio_worker_uses_fpga(
    name: &str,
    allowlist: Option<&str>,
    full_root_enabled: bool,
) -> bool {
    let allowlist = allowlist.map(str::trim);
    if allowlist.is_none() || allowlist == Some("auto") {
        // ic3/ic3_ctg_limit share one transition relation and supply the
        // qualified short-inquiry stream. ic3_abs_all owns the independently
        // qualified complete-root path; the joint two-lane replay includes
        // its view switches. Wider portfolios remain explicit because other
        // abstractions repeatedly invalidated queued requests on the board.
        return DEFAULT_FPGA_PORTFOLIO_WORKERS.contains(&name)
            || (full_root_enabled && name == DEFAULT_FULL_ROOT_PORTFOLIO_WORKER);
    }
    let allowlist = allowlist.unwrap();
    allowlist
        .split(',')
        .map(str::trim)
        .any(|candidate| candidate == "*" || candidate == "all" || candidate == name)
}

#[cfg(test)]
mod fpga_worker_tests {
    use super::portfolio_worker_uses_fpga;

    #[test]
    fn automatic_fpga_worker_policy_selects_the_qualified_mixed_set() {
        assert!(portfolio_worker_uses_fpga("ic3", None, false));
        assert!(portfolio_worker_uses_fpga("ic3_ctg_limit", None, false));
        assert!(!portfolio_worker_uses_fpga("ic3_abs_all", None, false));
        assert!(portfolio_worker_uses_fpga("ic3_abs_all", None, true));
        assert!(!portfolio_worker_uses_fpga("ic3_no_parent", None, true));
        assert!(!portfolio_worker_uses_fpga("ic3_inn", Some("auto"), true));
        assert!(portfolio_worker_uses_fpga("ic3", Some(" auto "), false));
    }

    #[test]
    fn explicit_fpga_worker_allowlist_is_exact_and_whitespace_tolerant() {
        assert!(portfolio_worker_uses_fpga(
            "ic3_inn",
            Some("ic3, ic3_inn"),
            false
        ));
        assert!(!portfolio_worker_uses_fpga(
            "ic3_abs_all",
            Some("ic3, ic3_inn"),
            true
        ));
        assert!(portfolio_worker_uses_fpga("anything", Some("all"), false));
        assert!(portfolio_worker_uses_fpga("anything", Some("*"), false));
        assert!(!portfolio_worker_uses_fpga("ic3", Some(""), false));
    }
}

impl Portfolio {
    pub fn new(
        ts: Transys,
        sym: VarSymbols,
        need_cert: bool,
        cfg: PortfolioConfig,
    ) -> anyhow::Result<Self> {
        let rst = Restore::new(&ts);
        let ots = ts.clone();
        let (ts, rst) = ts.preproc(&cfg.preproc, rst);
        // Let tempfile choose a private directory instead of relying on one
        // shared, pre-created path. The old /tmp/rIC3 parent could be left
        // owned by another account and make every portfolio run fail before
        // any worker started.
        let temp_dir =
            tempfile::TempDir::new().context("failed to create portfolio temporary directory")?;
        let mut engines = Vec::new();
        let mut new_engine = |name, args: &str| {
            let argv: Vec<_> = iter::once("").chain(args.split_whitespace()).collect();
            let cfg = EngineConfig::try_parse_from(argv)?;
            assert!(!cfg.is_wl());
            let (cert_tx, cert_rx) = if need_cert {
                let (cert_tx, cert_rx) = ipc::channel().unwrap();
                (Some(cert_tx), Some(cert_rx))
            } else {
                (None, None)
            };
            engines.push(Worker {
                name,
                cfg,
                args: args.to_string(),
                cert_tx,
                cert_rx,
                state: McResult::default(),
            });
            anyhow::Ok(())
        };
        let config = cfg.config.as_deref().unwrap_or("bl_default");
        let worker_cfgs = WorkerConfigs::from_toml(include_str!("portfolio.toml"), config);
        for (name, args) in worker_cfgs.iter_args(true) {
            new_engine(name.clone(), &args)
                .with_context(|| format!("invalid portfolio worker `{name}`"))?;
        }
        Ok(Self {
            ts,
            ots,
            sym,
            rst,
            cert: None,
            need_cert,
            cfg,
            engines,
            running: GHashMap::new(),
            winner_idx: None,
            temp_dir,
            ctrl: Arc::new(EngineCtrl::new()),
            tracer: Tracer::new(),
            ui: None,
            st_recv: IpcReceiverSet::new().unwrap(),
            stid_to_wid: GHashMap::new(),
        })
    }

    fn terminate_running(&mut self) {
        for &pid in self.running.keys() {
            let _ = nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM);
        }
        loop {
            match waitpid(None, None) {
                Ok(WaitStatus::Exited(pid, code)) => {
                    let worker_idx = self.running.remove(&pid).unwrap();
                    if code != 0 {
                        info!("{} exited with code {code}", self.engines[worker_idx].name);
                    }
                }
                Ok(WaitStatus::Signaled(pid, _, _)) => {
                    self.running.remove(&pid);
                }
                Err(Errno::EINTR) => continue,
                Err(Errno::ECHILD) => {
                    assert!(self.running.is_empty());
                    return;
                }
                Err(err) => panic!("portfolio waitpid failed: {err}"),
                _ => panic!(),
            }
        }
    }

    fn on_state_trace(&mut self, worker_idx: usize, prop: Option<usize>, res: McResult) {
        self.engines[worker_idx].state = res;
        self.tracer.trace_state(prop, res);
        if let Some(ui) = self.ui.as_mut() {
            ui.update(worker_idx, res);
        }
        let worker_name = self.engines[worker_idx].name.clone();
        let prop_prefix = prop.map(|p| format!("p{p}: ")).unwrap_or_default();
        match res {
            McResult::UNSAT => {
                info!("{worker_name}{prop_prefix} proved the property");
                self.accept_winner(worker_idx);
            }
            McResult::SAT(d) => {
                info!("{worker_name}{prop_prefix} found a counterexample at depth {d}");
                self.accept_winner(worker_idx);
            }
            McResult::Unknown(Some(d)) => {
                info!("{worker_name}{prop_prefix} proved at depth {d}");
            }
            McResult::Unknown(None) => {}
        }
    }

    fn accept_winner(&mut self, worker_idx: usize) {
        if self.winner_idx.is_some() {
            return;
        }
        let worker = &self.engines[worker_idx];
        info!(
            "best worker: {}, configuration: {}",
            worker.name, worker.args
        );
        self.winner_idx = Some(worker_idx);
        if self.need_cert {
            let cert = self.engines[worker_idx]
                .cert_rx
                .as_mut()
                .unwrap()
                .recv()
                .unwrap();
            self.tracer.trace_cert(&cert);
            self.cert = Some(cert);
        }
    }

    fn reap_child(&mut self) -> Option<McResult> {
        loop {
            match waitpid(None, Some(WaitPidFlag::WNOHANG)) {
                Ok(WaitStatus::StillAlive) => break,
                Ok(WaitStatus::Exited(pid, code)) => {
                    let worker_idx = self.running.remove(&pid).unwrap();
                    if code == 0 {
                        while self.winner_idx.is_none() {
                            self.poll_state_traces();
                        }
                        if self.winner_idx == Some(worker_idx) {
                            let res = self.engines[worker_idx].state;
                            assert!(!res.is_unknown());
                            return Some(res);
                        }
                    } else {
                        info!("{} exited with code {code}", self.engines[worker_idx].name);
                    }
                }
                Ok(WaitStatus::Signaled(pid, _, _)) => {
                    self.running.remove(&pid);
                }
                Err(Errno::EINTR) => continue,
                Err(Errno::ECHILD) => break,
                Err(err) => panic!("portfolio waitpid failed: {err}"),
                _ => panic!(),
            }
        }
        None
    }

    fn poll_state_traces(&mut self) {
        let events = match self.st_recv.try_select_timeout(Duration::from_millis(100)) {
            Ok(events) => events,
            Err(TrySelectError::Empty) => return,
            Err(err) => panic!("portfolio trace select failed: {err}"),
        };
        for event in events {
            match event {
                IpcSelectionResult::MessageReceived(id, message) => {
                    let Some(&worker_idx) = self.stid_to_wid.get(&id) else {
                        continue;
                    };
                    let (prop, res): (Option<usize>, McResult) = message.to().unwrap();
                    self.on_state_trace(worker_idx, prop, res);
                }
                IpcSelectionResult::ChannelClosed(id) => {
                    self.stid_to_wid.remove(&id);
                }
            }
        }
    }
}

impl Engine for Portfolio {
    fn check(&mut self) -> McResult {
        let mut lemma_mgr = self.cfg.share_lemma.then(LemmaMgr::new);
        for (worker_idx, worker) in self.engines.iter_mut().enumerate() {
            let (state_tx, state_rx) = ipc::channel().unwrap();
            let (lemma_send, lemma_recv) = if self.cfg.share_lemma {
                let (lemma_send, lemma_recv) = ipc::channel().unwrap();
                (Some(lemma_send), Some(lemma_recv))
            } else {
                (None, None)
            };
            match fork::fork().unwrap() {
                fork::Fork::Parent(child) => {
                    let state_trace_id = self.st_recv.add(state_rx).unwrap();
                    lemma_mgr.as_mut().map(|lemma_mgr| {
                        lemma_mgr
                            .add_worker(
                                worker.name.clone(),
                                lemma_recv.unwrap(),
                                lemma_send.unwrap(),
                            )
                            .unwrap()
                    });
                    let pid = Pid::from_raw(child);
                    info!("start engine {}", worker.name);
                    self.running.insert(pid, worker_idx);
                    self.stid_to_wid.insert(state_trace_id, worker_idx);
                }
                fork::Fork::Child => {
                    worker.run(
                        &self.ts, &self.ots, &self.rst, &self.sym, state_tx, lemma_recv,
                    );
                }
            }
        }
        let lemma_mgr_join = lemma_mgr.map(|lemma_mgr| spawn(move || lemma_mgr.run()));
        let interrupt = install_interrupt_handler(self.ctrl.clone());

        let start = Instant::now();
        loop {
            if self.ctrl.is_terminated() || self.cfg.time_limit_hit(start) {
                self.terminate_running();
                let _ = lemma_mgr_join.map(|j| j.join());
                if let Some(ui) = self.ui.as_ref() {
                    ui.finish(McResult::Unknown(None));
                }
                if interrupt.is_interrupted() {
                    exit(130);
                }
                return McResult::Unknown(None);
            }

            if self.running.is_empty() {
                let res = self
                    .winner_idx
                    .map(|winner_idx| self.engines[winner_idx].state)
                    .unwrap_or(McResult::Unknown(None));
                let _ = lemma_mgr_join.map(|j| j.join());
                if let Some(ui) = self.ui.as_ref() {
                    ui.finish(res);
                }
                return res;
            }

            self.poll_state_traces();

            if let Some(res) = self.reap_child() {
                self.terminate_running();
                let _ = lemma_mgr_join.map(|j| j.join());
                if let Some(ui) = self.ui.as_ref() {
                    ui.finish(res);
                }
                return res;
            }
        }
    }

    fn add_tracer(&mut self, tracer: Box<dyn TracerIf>) {
        self.tracer.add_tracer(tracer);
    }

    fn get_ctrl(&self) -> Arc<dyn TerminateCtrl> {
        self.ctrl.clone()
    }

    fn set_ui(&mut self, renderer: UiRenderer) {
        self.ui = Some(PortfolioUi::new(
            renderer,
            self.engines.iter().map(|worker| worker.name.clone()),
        ));
    }
}

impl BlEngine for Portfolio {
    fn proof(&mut self) -> BlProof {
        let Some(McBlCertificate::UNSAT(proof)) = self.cert.as_ref() else {
            panic!("no proof available");
        };
        proof.clone()
    }

    fn cex(&mut self) -> BlCex {
        let Some(McBlCertificate::SAT(cex)) = self.cert.as_ref() else {
            panic!("no counterexample available");
        };
        cex.clone()
    }
}

#[derive(Default, Clone)]
pub struct LightPortfolioConfig {
    pub time_limit: Option<usize>,
}

pub struct LightPortfolio {
    ts: Transys,
    sym: VarSymbols,
    cfg: LightPortfolioConfig,
    ecfgs: Vec<EngineConfig>,
    engines: Vec<Box<dyn BlEngine>>,
    ctrl: Arc<EngineCtrl>,
}

impl LightPortfolio {
    pub fn new(
        cfg: LightPortfolioConfig,
        ts: Transys,
        sym: VarSymbols,
        ecfgs: Vec<EngineConfig>,
    ) -> Self {
        Self {
            cfg,
            ecfgs,
            ts,
            sym,
            engines: Vec::new(),
            ctrl: Arc::new(EngineCtrl::new()),
        }
    }
}

impl Engine for LightPortfolio {
    fn check(&mut self) -> McResult {
        let engines: Vec<_> = self
            .ecfgs
            .clone()
            .into_par_iter()
            .map(|ecfg| create_bl_engine(ecfg, self.ts.clone(), self.sym.clone()))
            .collect();
        let ctrls: Vec<_> = engines.iter().map(|e| e.get_ctrl()).collect();
        let (tx, rx) = mpsc::channel();
        let mut joins = Vec::new();
        with_log_level(log::LevelFilter::Warn, || {
            let start = Instant::now();
            let num_engines = engines.len();
            for mut e in engines {
                let tx = tx.clone();
                let join = spawn(move || {
                    let res = e.check();
                    let _ = tx.send(res);
                    (e, res)
                });
                joins.push(join);
            }
            let mut res = McResult::Unknown(None);
            let mut finished = 0;
            while finished < num_engines {
                if self.ctrl.is_terminated() {
                    info!("LightPortfolio interrupted by external signal.");
                    break;
                }
                if let Some(t) = self.cfg.time_limit
                    && start.elapsed().as_secs() >= t as u64
                {
                    info!("LightPortfolio terminated by timeout.");
                    break;
                }
                match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                    Ok(r) => {
                        finished += 1;
                        if !r.is_unknown() {
                            res = r;
                            break;
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        break;
                    }
                }
            }
            for c in ctrls {
                c.terminate();
            }
            for j in joins {
                let (e, _) = j.join().unwrap();
                self.engines.push(e);
            }
            res
        })
    }

    fn get_ctrl(&self) -> Arc<dyn TerminateCtrl> {
        self.ctrl.clone()
    }
}
