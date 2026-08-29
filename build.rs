// Link the accelerator library when it has been built.
//
// Absent, `INDUCTOR_ACCEL` simply never becomes ready and the solver runs
// exactly as before: the accelerator is an addition, not a dependency, and a
// machine without a card has to keep working.
fn main() {
    println!("cargo:rerun-if-env-changed=INDUCTOR_ACCEL_LIB");
    println!("cargo:rerun-if-env-changed=XRT_ROOT");
    let lib =
        std::env::var("INDUCTOR_ACCEL_LIB").unwrap_or_else(|_| "../../hls/build/hw".to_string());
    let legacy_accel = std::path::Path::new(&format!("{lib}/libinductor_accel.a")).exists();
    let cdcl_accel = std::path::Path::new(&format!("{lib}/libinductor_cdcl_host.a")).exists();
    if legacy_accel || cdcl_accel {
        println!("cargo:rustc-link-search=native={lib}");
    }
    if legacy_accel {
        println!("cargo:rustc-link-lib=static=inductor_accel");
        println!("cargo:rustc-cfg=has_accel");
    }
    if cdcl_accel {
        println!("cargo:rustc-link-lib=static=inductor_cdcl_host");
        println!("cargo:rustc-cfg=has_cdcl_accel");
    }
    if legacy_accel || cdcl_accel {
        let xrt = std::env::var("XRT_ROOT").unwrap_or_else(|_| "/opt/xilinx/xrt".to_string());
        println!("cargo:rustc-link-search=native={xrt}/lib");
        println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    // Relink when the library changes. Without this cargo tracks only build.rs,
    // so a rebuilt libinductor_accel.a leaves the old one baked into the binary
    // -- which cost two board runs: the kernel signature had changed, the
    // library had been rebuilt to match, and the binary still called the old
    // argument indices and threw on init.
    println!("cargo:rerun-if-changed={lib}/libinductor_accel.a");
    println!("cargo:rerun-if-changed={lib}/libinductor_cdcl_host.a");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(has_accel)");
    println!("cargo:rustc-check-cfg=cfg(has_cdcl_accel)");
}
