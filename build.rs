use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    let ptx = Path::new("src/conditional_corr.ptx");
    let source = Path::new("src/conditional_corr.cu");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo did not set OUT_DIR"))
        .join("conditional_corr.ptx");

    if ptx.is_file() {
        println!("cargo:rerun-if-changed={}", ptx.display());
        fs::copy(ptx, &output).expect("Could not copy the bundled conditional_corr.ptx");
        return;
    }

    // Watch the source directory so Cargo also notices if the bundled PTX is restored.
    println!("cargo:rerun-if-changed=src");
    let status = Command::new("nvcc")
        .arg("--ptx")
        .arg(source)
        .arg("-o")
        .arg(&output)
        .status()
        .expect("Could not run nvcc. Install the CUDA Toolkit to build missing PTX");
    assert!(
        status.success(),
        "nvcc failed to compile {} ({status})",
        source.display()
    );
}
