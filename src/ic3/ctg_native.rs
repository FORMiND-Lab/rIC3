//! Opt-in live-CPU integration of the bounded native CTG architecture model.
//! This launches a separate simulation process, never an FPGA or CPU proof
//! double-check. A pinned, qualified candidate owns its own disposable context;
//! only a fully validated committed journal crosses back into the caller.
use super::IC3;
use anyhow::{Context, Result, bail, ensure};
use logicrs::{Lit, LitVec, Var};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

const N: usize = 2048;
const MAX_CLAUSES: usize = 8192;
const MAX_LITS: usize = 65536;
const FILE_LIMIT: u64 = 32 * 1024 * 1024;
static ORDINAL: AtomicUsize = AtomicUsize::new(0);

pub(super) struct NativeLemma {
    pub(super) hi: usize,
    pub(super) cube: LitVec,
}
pub(super) struct NativeCtgResult {
    pub(super) complete: bool,
    pub(super) cube: LitVec,
    pub(super) journal: Vec<NativeLemma>,
    pub(super) job_dir: PathBuf,
}
struct Settings {
    executable: PathBuf,
    sha: String,
    output: PathBuf,
    max_roots: usize,
}

fn settings() -> Option<&'static Settings> {
    static SETTINGS: OnceLock<Option<Settings>> = OnceLock::new();
    SETTINGS
        .get_or_init(|| {
            let executable = std::env::var_os("INDUCTOR_CTG_NATIVE_EXECUTABLE")?;
            let parsed = (|| -> Result<Settings> {
                // Library search/build paths are not activation switches. All known
                // hardware/query/root offload entry points must be absent, even =0.
                for key in [
                    "INDUCTOR_CTG_HARDWARE_SOCKET",
                    "INDUCTOR_ACCEL",
                "INDUCTOR_ACTIVE_CDCL",
                "INDUCTOR_CDCL_ACTIVE",
                "INDUCTOR_CDCL_SHADOW",
                "INDUCTOR_CDCL_PAIRED",
                "INDUCTOR_CDCL_TRACE_CSV",
                    "INDUCTOR_CDCL_BLOCK_ROOT_EXECUTOR",
                    "INDUCTOR_CDCL_BLOCK_FULL_ROOT",
                    "INDUCTOR_CDCL_BLOCK_CONTROLLER_OWNS_QUEUE",
                ] {
                    ensure!(
                        std::env::var_os(key).is_none(),
                        "native CTG conflicts with {key}"
                    );
                }
                let executable = PathBuf::from(executable);
                let output = PathBuf::from(
                    std::env::var_os("INDUCTOR_CTG_NATIVE_OUTPUT_DIR")
                        .context("missing native output directory")?,
                );
                let sha = std::env::var("INDUCTOR_CTG_NATIVE_SHA256")?.to_ascii_lowercase();
                ensure!(
                    sha.len() == 64 && sha.bytes().all(|c| c.is_ascii_hexdigit()),
                    "binary SHA256"
                );
                let max_roots = std::env::var("INDUCTOR_CTG_NATIVE_MAX_ROOTS")
                    .unwrap_or_else(|_| "8".into())
                    .parse::<usize>()?;
                ensure!((1..=128).contains(&max_roots), "native root budget");
                ensure!(
                    executable.is_absolute() && executable.is_file(),
                    "native executable path"
                );
                ensure!(
                    output.is_absolute() && output.is_dir(),
                    "existing private output directory"
                );
                Ok(Settings {
                    executable,
                    sha,
                    output,
                    max_roots,
                })
            })();
            match parsed {
                Ok(value) => Some(value),
                Err(error) => {
                    eprintln!("native CTG disabled: {error:#}");
                    None
                }
            }
        })
        .as_ref()
}
fn lits(xs: &[Lit]) -> Vec<u32> {
    xs.iter().map(|x| u32::from(*x)).collect()
}
fn clauses(xs: &[LitVec]) -> Vec<Vec<u32>> {
    xs.iter().map(|x| lits(x)).collect()
}
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    ensure!(
        file.metadata()?.len() <= limit,
        "file size limit: {}",
        path.display()
    );
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= limit, "file grew beyond limit");
    Ok(bytes)
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    ensure!(bytes.len() as u64 <= FILE_LIMIT, "output file size limit");
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(bytes)?;
    Ok(())
}
fn number(v: &Value, key: &str) -> Result<usize> {
    usize::try_from(
        v.get(key)
            .and_then(Value::as_u64)
            .with_context(|| format!("integer {key}"))?,
    )
    .context("integer extent")
}
fn array<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    v.get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("array {key}"))
}
pub(super) fn words(v: &Value) -> Result<Vec<u32>> {
    let a = v.as_array().context("word array")?;
    ensure!(a.len() <= MAX_LITS, "word array capacity");
    a.iter()
        .map(|x| u32::try_from(x.as_u64().context("word")?).context("u32 word"))
        .collect()
}
pub(super) fn state_cube(v: &Value, input: &Value) -> Result<Vec<u32>> {
    let c = words(v)?;
    let nv = number(input, "n_var")?;
    let latches = words(&input["latch_variables"])?;
    ensure!(!c.is_empty() && c.len() <= N, "nonempty bounded cube");
    for (i, lit) in c.iter().enumerate() {
        ensure!(
            (*lit >> 1) < nv as u32 && latches.contains(&(lit >> 1)),
            "state literal"
        );
        ensure!(
            !c[..i].iter().any(|old| old >> 1 == lit >> 1),
            "cube variable duplicate"
        );
    }
    let init = words(&input["init_value_by_current"])?;
    ensure!(
        c.iter().any(
            |l| init[(l >> 1) as usize] < 2 && init[(l >> 1) as usize] != u32::from(l & 1 == 0)
        ),
        "constant-init guard"
    );
    Ok(c)
}
pub(super) fn ordered_subset(cube: &[u32], source: &[u32]) -> bool {
    let mut next = 0;
    for lit in source {
        if next < cube.len() && cube[next] == *lit {
            next += 1;
        }
    }
    next == cube.len()
}
pub(super) fn to_litvec(cube: &[u32]) -> LitVec {
    cube.iter()
        .map(|lit| Lit::new(Var::from((lit >> 1) as usize), lit & 1 == 0))
        .collect()
}

impl IC3 {
    pub(super) fn native_ctg_snapshot(
        &self,
        frame: usize,
        cube: &LitVec,
        constraint: &[LitVec],
        ordinal: usize,
    ) -> Result<Value> {
        ensure!(
            frame > 0 && frame <= self.level() && self.solvers.len() == self.level() + 1,
            "frame extent"
        );
        ensure!(
            !cube.is_empty() && cube.len() <= N && constraint.is_empty(),
            "root continuation"
        );
        let solver = &self.solvers[frame - 1];
        let (init, latches, inputs) = solver.resident_block_projection_metadata();
        let next = solver.resident_block_next_var_map();
        let nv = next.len();
        ensure!(
            nv > 0 && nv <= N && self.tsctx.num_var() <= nv,
            "variable capacity"
        );
        ensure!(
            init.len() == nv && init.iter().all(|x| *x <= 2),
            "init geometry"
        );
        ensure!(
            latches.len() + inputs.len() <= N && self.solvers.len() <= 64,
            "projection/frame capacity"
        );
        for projection in [&latches, &inputs] {
            ensure!(
                projection.windows(2).all(|p| p[0] < p[1])
                    && projection.iter().all(|v| (*v as usize) < nv),
                "projection order/extent"
            );
        }
        ensure!(
            inputs.iter().all(|v| !latches.contains(v)),
            "projection overlap"
        );
        ensure!(
            next.iter().all(|n| *n == u32::MAX || (*n >> 1) < nv as u32),
            "next literal extent"
        );
        ensure!(
            latches.iter().all(|v| next[*v as usize] != u32::MAX),
            "mapped latch"
        );
        let mut frames = Vec::new();
        let mut common_t: Option<Vec<Vec<u32>>> = None;
        let mut clause_count = 0usize;
        let mut literal_count = 0usize;
        for (index, slv) in self.solvers.iter().enumerate() {
            let (n, tag, t, lemmas) = slv.dcs.incremental_resident_partition();
            ensure!(n as usize == nv, "frame variable identity");
            let t = clauses(&t);
            if let Some(ref expected) = common_t {
                ensure!(&t == expected, "frame T mismatch");
            } else {
                clause_count += t.len();
                literal_count += t.iter().map(Vec::len).sum::<usize>();
                common_t = Some(t);
            }
            let lemmas = clauses(&lemmas);
            clause_count += lemmas.len();
            literal_count += lemmas.iter().map(Vec::len).sum::<usize>();
            frames.push(
                json!({"solver_index":index,"reported_frame_tag":tag,"n_var":n,"lemmas":lemmas}),
            );
        }
        ensure!(
            clause_count <= MAX_CLAUSES && literal_count <= MAX_LITS,
            "resident formula capacity"
        );
        let t = common_t.context("frame transition missing")?;
        ensure!(!t.is_empty(), "transition body missing");
        let (lift_nv, _, lift_t, lift_lemmas) = self.lift.capture_transition_snapshot();
        ensure!(
            lift_nv as usize == nv && lift_lemmas.is_empty(),
            "lift background/extent"
        );
        ensure!(t == clauses(&lift_t), "frame/lift transition mismatch");
        let raw_t: Vec<Vec<u32>> = self.ts.rel.clause().map(|c| lits(c)).collect();
        ensure!(t == raw_t, "actual DAG transition mismatch");
        ensure!(
            t.iter()
                .all(|c| !c.is_empty() && c.len() <= N && c.iter().all(|l| (*l >> 1) < nv as u32)),
            "transition clause geometry"
        );
        for f in &frames {
            for c in array(f, "lemmas")? {
                let c = words(c)?;
                ensure!(
                    !c.is_empty() && c.len() <= N && c.iter().all(|l| (*l >> 1) < nv as u32),
                    "lemma geometry"
                );
            }
        }
        let refine: Vec<bool> = (0..self.tsctx.num_var())
            .map(|v| self.localabs.refine_has(Var::from(v)))
            .collect();
        let filtered: Vec<u32> = self
            .ts
            .constraint
            .iter()
            .filter(|l| self.localabs.refine_has(l.var()))
            .map(|l| u32::from(*l))
            .collect();
        ensure!(
            filtered.len() <= N && cube.len() + filtered.len() <= N,
            "target clause capacity"
        );
        let init_raw: Vec<Value> = self
            .tsctx
            .latch
            .iter()
            .map(|v| {
                json!({"var":u32::from(*v),
            "init_literal":self.tsctx.init_map[*v].map(u32::from)})
            })
            .collect();
        let init_cnf: Vec<Vec<u32>> = self.tsctx.init.iter().map(|c| lits(c)).collect();
        let deps: Vec<Vec<u32>> = (0..self.tsctx.num_var())
            .map(|v| {
                self.ts
                    .rel
                    .dep(Var::from(v))
                    .iter()
                    .map(|x| u32::from(*x))
                    .collect()
            })
            .collect();
        let input = json!({"schema":"inductor.ctg-root-input.v1","root_mic_ordinal":ordinal,
            "capture_point":"live native simulation after natural MIC ordering; before drop loop",
            "hardware_candidate_must_not_read_oracle":true,"frame":frame,"max_frame":self.level(),
            "level":1,"cube_entry":lits(cube),"entry_temporary_constraints":clauses(constraint),
            "loop_initial":{"kind":"mic_loop_entry","depth":1,"cube":lits(cube),"keep":[],"i":0,"level":1,"max":3,"limit":1},
            "n_var":nv,"ts_n_var":self.tsctx.num_var(),"transition_only":t,
            "transition_origin":"actual frame partition == TsLift.slv partition == preprocessed ts.rel clauses",
            "lift_n_var":lift_nv,"frame_solvers":frames,"frame_bookkeeping_image":self.frame.progress_image(),
            "next_literal_by_current":next,"init_value_by_current":init,"init_literal_by_latch":init_raw,
            "init_cnf":init_cnf,"latch_variables":latches,"input_variables":inputs,
            "constraints_raw":lits(&self.ts.constraint),"constraints_localabs_filtered":filtered,
            "localabs_refine_mask":refine,"dag_dependencies":deps,
            "config":{"ctg":self.cfg.ctg,"ctg_max":self.cfg.ctg_max,"ctg_limit":self.cfg.ctg_limit,
                "dynamic":self.cfg.dynamic,"mab":self.cfg.mab,"inn":self.cfg.inn,
                "parent_lemma":self.cfg.parent_lemma,"ctp":self.cfg.ctp},
            "learnts_omitted_as_redundant":true,"watchers_or_cpu_search_state_copied":false,
            "cpu_rng_state_copied":false,"identical_solver_trajectory_claimed":false});
        state_cube(&input["cube_entry"], &input)?;
        Ok(input)
    }

    /// Called only at the naturally ordered root drop-loop boundary. Default
    /// disabled: no solver snapshot, file writes, hashing or child execution.
    pub(super) fn try_native_ctg_root(
        &self,
        frame: usize,
        cube: &LitVec,
        constraint: &[LitVec],
        level: usize,
        max: usize,
        limit: usize,
    ) -> Option<NativeCtgResult> {
        let cfg = settings()?;
        if level != 1
            || max != 3
            || limit != 1
            || !self.cfg.ctg
            || self.cfg.dynamic
            || self.cfg.mab
            || self.cfg.ctp
        {
            return None;
        }
        let ordinal = ORDINAL.fetch_add(1, Ordering::Relaxed) + 1;
        if ordinal > cfg.max_roots {
            return None;
        }
        // Attempt quota is consumed before geometry/capture/execution checks.
        // No unsuccessful root is replaced by a later, more favorable root.
        let job = match tempfile::Builder::new()
            .prefix(&format!("job-{ordinal:04}-"))
            .tempdir_in(&cfg.output)
        {
            Ok(dir) => dir.keep(),
            Err(error) => {
                eprintln!("native CTG job {ordinal}: {error}");
                return None;
            }
        };
        let mut receipt = json!({"schema":"inductor.ctg-live-native-job.v1",
            "eligible_ordinal":ordinal,"hardware":false,"native_simulation":true,
            "accepted":false,"complete":false,"binary_sha256":cfg.sha,
            "cpu_proof_double_check":false,"independent_offline_check_pending":true,
            "cpu_adoption_record":"adoption.json (caller writes after actual integration)",
            "limits":{"child_seconds":10,"address_space_bytes":17179869184u64,
                "combined_log_bytes":67108864u64,"each_file_bytes":FILE_LIMIT}});
        let attempt = (|| -> Result<NativeCtgResult> {
            let input = self.native_ctg_snapshot(frame, cube, constraint, ordinal)?;
            let input_bytes = serde_json::to_vec_pretty(&input)?;
            let stdin = serialize(&input)?;
            write_new(&job.join("input.json"), &input_bytes)?;
            write_new(&job.join("stdin"), &stdin)?;
            receipt["input_sha256"] = json!(hash(&input_bytes));
            receipt["stdin_sha256"] = json!(hash(&stdin));
            // Copy then hash a private immutable invocation image, avoiding a
            // hash(path)/exec(path) race against later edits to the source path.
            let executable = read_bounded(&cfg.executable, FILE_LIMIT)?;
            ensure!(
                hash(&executable) == cfg.sha,
                "native executable SHA256 mismatch"
            );
            let program = job.join("native-executable");
            write_new(&program, &executable)?;
            fs::set_permissions(&program, fs::Permissions::from_mode(0o500))?;
            let terminal = run_child(&program, &job, &mut receipt);
            let stdout = read_bounded(&job.join("native.stdout"), FILE_LIMIT)?;
            let stderr = read_bounded(&job.join("native.stderr"), FILE_LIMIT)?;
            receipt["stdout_sha256"] = json!(hash(&stdout));
            receipt["stderr_sha256"] = json!(hash(&stderr));
            let code = terminal?;
            let mut result = validate_terminal(
                &input,
                std::str::from_utf8(&stdout)?,
                std::str::from_utf8(&stderr)?,
                code,
            )?;
            result.job_dir = job.clone();
            receipt["accepted"] = json!(true);
            receipt["complete"] = json!(result.complete);
            receipt["result_cube"] = json!(lits(&result.cube));
            receipt["adoptable_journal"] = json!(
                result
                    .journal
                    .iter()
                    .map(|e| json!({"hi":e.hi,"cube":lits(&e.cube)}))
                    .collect::<Vec<_>>()
            );
            Ok(result)
        })();
        if let Err(ref error) = attempt {
            receipt["error"] = json!(format!("{error:#}"));
        }
        if write_new(
            &job.join("receipt.json"),
            &serde_json::to_vec_pretty(&receipt).ok()?,
        )
        .is_err()
        {
            eprintln!("native CTG receipt failure: {}", job.display());
            return None;
        }
        match attempt {
            Ok(result) => Some(result),
            Err(error) => {
                eprintln!("native CTG job {ordinal} rejected: {error:#}");
                None
            }
        }
    }
}

pub(super) fn serialize(input: &Value) -> Result<Vec<u8>> {
    let mut out = vec![
        0x43544731u32,
        number(input, "n_var")? as u32,
        number(input, "max_frame")? as u32,
        number(input, "frame")? as u32,
        1,
        3,
        1,
    ];
    fn vector(out: &mut Vec<u32>, value: &Value) -> Result<()> {
        let v = words(value)?;
        out.push(v.len() as u32);
        out.extend(v);
        Ok(())
    }
    fn cnf(out: &mut Vec<u32>, value: &Value) -> Result<()> {
        let cs = value.as_array().context("CNF")?;
        out.push(cs.len() as u32);
        for c in cs {
            vector(out, c)?;
        }
        Ok(())
    }
    for value in [
        &input["loop_initial"]["cube"],
        &input["next_literal_by_current"],
        &input["init_value_by_current"],
        &input["latch_variables"],
        &input["input_variables"],
        &input["constraints_localabs_filtered"],
    ] {
        vector(&mut out, value)?;
    }
    cnf(&mut out, &input["transition_only"])?;
    let frames = array(input, "frame_solvers")?;
    out.push(frames.len() as u32);
    for frame in frames {
        cnf(&mut out, &frame["lemmas"])?;
    }
    let text = out.iter().map(u32::to_string).collect::<Vec<_>>().join(" ") + "\n";
    ensure!(text.len() as u64 <= FILE_LIMIT, "native stdin capacity");
    Ok(text.into_bytes())
}

fn run_child(program: &Path, job: &Path, receipt: &mut Value) -> Result<i32> {
    let argv = vec![
        "/usr/bin/taskset".to_owned(),
        "-c".into(),
        "112-115".into(),
        "/usr/bin/prlimit".into(),
        "--as=17179869184:17179869184".into(),
        "--fsize=33554432:33554432".into(),
        "--cpu=10:10".into(),
        "--core=0:0".into(),
        "--".into(),
        program.to_string_lossy().into_owned(),
    ];
    receipt["argv"] = json!(argv);
    let stdout = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(job.join("native.stdout"))?;
    let stderr = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(job.join("native.stderr"))?;
    // No setsid/new process group: an outer CPU guard can kill the whole group.
    // Native candidate is a single process; taskset/prlimit exec it directly.
    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .current_dir(job)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("LANG", "C")
        .env("LC_ALL", "C")
        .stdin(Stdio::from(File::open(job.join("stdin"))?))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()?;
    let started = Instant::now();
    let terminal = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if started.elapsed() < Duration::from_secs(10) => {
                std::thread::sleep(Duration::from_millis(2))
            }
            Ok(None) => {
                receipt["timed_out"] = json!(true);
                let _ = child.kill();
                let reaped = child.wait();
                receipt["reaped"] = json!(reaped.is_ok());
                receipt["exit_code"] = json!(reaped.ok().and_then(|s| s.code()));
                break Err(anyhow::anyhow!("native child timeout"));
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(error.into());
            }
        }
    };
    receipt["child_elapsed_ns"] = json!(started.elapsed().as_nanos() as u64);
    let status = terminal?;
    receipt["timed_out"] = json!(false);
    receipt["reaped"] = json!(true);
    receipt["exit_code"] = json!(status.code());
    let code = status.code().context("native child terminated by signal")?;
    ensure!(
        code == 0 || code == 2,
        "native child rejected/error exit {code}"
    );
    Ok(code)
}

// Structural transport/certificate provenance checks only. No SAT invocation,
// CPU oracle lookup, model minimization, or mutation of the running CPU solver.
fn validate_terminal(
    input: &Value,
    stdout: &str,
    stderr: &str,
    code: i32,
) -> Result<NativeCtgResult> {
    ensure!(code == 0 || code == 2, "unsupported native exit");
    let events: Vec<Value> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    ensure!(
        !events.is_empty() && events.len() <= 4096,
        "native event count"
    );
    let transport: Value = serde_json::from_str(stderr.trim())?;
    ensure!(
        transport["schema"] == "inductor.ctg-resident-native-transport.v1"
            && transport["hardware"] == false
            && transport["transactional"] == true
            && transport["drained"] == true
            && transport["receiver_ready"] == true,
        "transport identity/fence"
    );
    let maxframe = number(input, "max_frame")?;
    let frame = number(input, "frame")?;
    let mut queries: Vec<&Value> = Vec::new();
    let mut appends: Vec<&Value> = Vec::new();
    let result = events.last().context("terminal result")?;
    for event in &events[..events.len() - 1] {
        match event["kind"].as_str() {
            Some("query") => {
                ensure!(
                    number(event, "id")? == queries.len() + 1 && queries.len() < 128,
                    "query sequence/budget"
                );
                ensure!((1..=3).contains(&number(event, "status")?), "query status");
                queries.push(event);
            }
            Some("append") => {
                let cube = state_cube(&event["cube"], input)?;
                let hi = number(event, "hi")?;
                ensure!(
                    number(event, "lo")? == 1 && hi > 0 && hi <= maxframe,
                    "append range"
                );
                let proof = number(event, "proof_query_id")?;
                ensure!(proof > 0 && proof <= queries.len(), "append proof id");
                ensure!(
                    publish_proof(queries[proof - 1], hi - 1, &cube)?,
                    "highest append proof"
                );
                for lower in 0..hi {
                    let mut found = false;
                    for query in &queries {
                        if publish_proof(query, lower, &cube)? {
                            found = true;
                            break;
                        }
                    }
                    ensure!(found, "missing affected-frame publish proof");
                }
                appends.push(event);
            }
            _ => bail!("unexpected native event"),
        }
    }
    ensure!(
        result["kind"] == "result"
            && number(result, "queries")? == queries.len()
            && number(result, "steps")? <= 10000
            && number(result, "published")? == appends.len(),
        "terminal census"
    );
    let complete = result["status"] == "complete";
    ensure!(
        (code == 0 && complete) || (code == 2 && result["status"] == "fallback"),
        "exit/status agreement"
    );
    let cube = state_cube(&result["cube"], input)?;
    ensure!(
        ordered_subset(&cube, &words(&input["loop_initial"]["cube"])?),
        "ordered root subset"
    );
    let finalid = number(result, "final_query_id")?;
    if complete {
        ensure!(finalid > 0 && finalid == queries.len(), "final query id");
        let final_query = queries[finalid - 1];
        ensure!(
            final_query["purpose"] == "final"
                && number(final_query, "frame")? == frame - 1
                && number(final_query, "status")? == 2
                && words(&final_query["cube"])? == cube
                && words(&final_query["extra_full"])?.is_empty(),
            "final independent induction query"
        );
    } else {
        ensure!(finalid == 0, "fallback has final proof id");
    }
    let journal = array(&transport, "journal")?;
    ensure!(
        journal.len() == appends.len()
            && journal.len() <= 128
            && number(&transport, "journal_attempted")? == journal.len()
            && number(&transport, "journal_committed")? == journal.len(),
        "journal census"
    );
    let mut adopted = Vec::new();
    for (entry, event) in journal.iter().zip(&appends) {
        ensure!(
            entry["attempted"] == true && entry["committed"] == true,
            "unconfirmed mutation"
        );
        for key in ["lo", "hi", "cube", "proof_query_id"] {
            ensure!(
                entry.get(key).is_some() && entry[key] == event[key],
                "journal/append mismatch: {key}"
            );
        }
        adopted.push(NativeLemma {
            hi: number(entry, "hi")?,
            cube: to_litvec(&words(&entry["cube"])?),
        });
    }
    let mutations = 1 + adopted.len();
    ensure!(
        number(&transport, "begins")? == mutations
            && number(&transport, "commits")? == mutations
            && number(&transport, "cookie")? == mutations
            && number(&transport, "generation")? == mutations
            && number(&transport, "aborts")? == 0
            && number(&transport, "legacy_mutations")? == 0
            && number(&transport, "max_wire_words")? <= 8192,
        "transactional mutation census"
    );
    let chunks = number(&transport, "chunks")?;
    ensure!(
        chunks >= mutations
            && number(&transport, "transactions")? == queries.len() + 2 * mutations + chunks,
        "drained transaction census"
    );
    let mut initial_clauses = array(input, "transition_only")?.len();
    for f in array(input, "frame_solvers")? {
        initial_clauses += array(f, "lemmas")?.len();
    }
    ensure!(
        number(&transport, "clauses")? == initial_clauses + adopted.len(),
        "resident clause census"
    );
    Ok(NativeCtgResult {
        complete,
        cube: to_litvec(&cube),
        journal: adopted,
        job_dir: PathBuf::new(),
    })
}
fn publish_proof(query: &Value, frame: usize, cube: &[u32]) -> Result<bool> {
    Ok(query["purpose"] == "publish"
        && number(query, "frame")? == frame
        && number(query, "status")? == 2
        && words(&query["cube"])? == cube
        && words(&query["extra_full"])?.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(complete: bool) -> (Value, Vec<Value>, Value) {
        let input = json!({"n_var":4,"max_frame":1,"frame":1,
            "latch_variables":[1,2],"init_value_by_current":[2,0,0,2],
            "loop_initial":{"cube":[2,4]},"transition_only":[[1]],
            "frame_solvers":[{"lemmas":[]},{"lemmas":[]}]});
        let mut events = vec![
            json!({"kind":"query","id":1,"purpose":"publish","frame":0,
                "status":2,"cube":[4],"extra_full":[]}),
            json!({"kind":"append","lo":1,"hi":1,"cube":[4],"proof_query_id":1}),
        ];
        if complete {
            events.push(json!({"kind":"query","id":2,"purpose":"final",
            "frame":0,"status":2,"cube":[2],"extra_full":[]}));
        }
        let queries = if complete { 2 } else { 1 };
        events.push(
            json!({"kind":"result","status":if complete {"complete"}else{"fallback"},
            "cube":[2],"queries":queries,"steps":4,"published":1,
            "final_query_id":if complete {2}else{0}}),
        );
        let transport = json!({"schema":"inductor.ctg-resident-native-transport.v1",
            "hardware":false,"transactional":true,"drained":true,"receiver_ready":true,
            "journal_attempted":1,"journal_committed":1,"begins":2,"commits":2,
            "cookie":2,"generation":2,"aborts":0,"legacy_mutations":0,"max_wire_words":25,
            "chunks":2,"transactions":queries+6,"clauses":2,
            "journal":[{"lo":1,"hi":1,"proof_query_id":1,"cube":[4],"attempted":true,"committed":true}]});
        (input, events, transport)
    }
    fn validate(
        input: &Value,
        events: &[Value],
        transport: &Value,
        code: i32,
    ) -> Result<NativeCtgResult> {
        let stdout = events
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        validate_terminal(input, &stdout, &transport.to_string(), code)
    }
    #[test]
    fn complete_and_safe_fallback_transfer_the_same_committed_journal() {
        for complete in [false, true] {
            let (input, events, transport) = fixture(complete);
            let result =
                validate(&input, &events, &transport, if complete { 0 } else { 2 }).unwrap();
            assert_eq!(result.complete, complete);
            assert_eq!(lits(&result.cube), vec![2]);
            assert_eq!(result.journal.len(), 1);
            assert_eq!(result.journal[0].hi, 1);
            assert_eq!(lits(&result.journal[0].cube), vec![4]);
        }
    }
    #[test]
    fn all_journal_entries_validate_before_any_result_is_returned() {
        for which in 0..9 {
            let (input, mut events, mut transport) = fixture(true);
            match which {
                0 => transport["journal"][0]["committed"] = json!(false),
                1 => transport["journal"][0]["cube"] = json!([2]),
                2 => events[1]["hi"] = json!(2),
                3 => events[0]["extra_full"] = json!([2, 4]),
                4 => events[0]["status"] = json!(1),
                5 => transport["transactions"] = json!(9),
                6 => transport["receiver_ready"] = json!(false),
                7 => events[3]["cube"] = json!([4, 2]),
                _ => events[3]["final_query_id"] = json!(1),
            }
            assert!(
                validate(&input, &events, &transport, 0).is_err(),
                "mutation {which}"
            );
        }
    }
    #[test]
    fn errors_and_status_mismatch_never_adopt() {
        let (input, events, transport) = fixture(true);
        assert!(validate(&input, &events, &transport, 2).is_err());
        assert!(validate(&input, &events, &transport, 3).is_err());
        assert!(validate_terminal(&input, "{}", "{}", 0).is_err());
    }
    #[test]
    fn subset_order_is_not_set_membership() {
        assert!(ordered_subset(&[6, 4], &[6, 2, 4]));
        assert!(!ordered_subset(&[4, 6], &[6, 2, 4]));
        assert!(!ordered_subset(&[6, 6], &[6, 2, 4]));
    }
    #[test]
    fn literal_round_trip_and_init_guard() {
        assert_eq!(
            lits(&to_litvec(&[0, 1, 2, 3, 2046])),
            vec![0, 1, 2, 3, 2046]
        );
        let input = json!({"n_var":4,"latch_variables":[1,2],"init_value_by_current":[2,0,1,2]});
        assert_eq!(state_cube(&json!([2]), &input).unwrap(), vec![2]);
        assert!(state_cube(&json!([3]), &input).is_err());
        assert!(state_cube(&json!([2, 3]), &input).is_err());
        assert!(state_cube(&json!([6]), &input).is_err());
    }
}
