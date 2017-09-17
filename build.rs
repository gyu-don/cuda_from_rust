use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    Command::new("nvcc")
        .args(&["-c", "-arch=sm_20", "src/vectorAdd.cu", "-Xcompiler", "-fPIC", "-o"])
        .arg(&format!("{}/vectorAdd.o", out_dir))
        .status()
        .unwrap();
    /* You can choice nvcc or clang. */
    /*
    Command::new("clang")
        .args(&["src/vectorAdd.cu", "--cuda-path=/opt/cuda", "-c", "-o"])
        .arg(&format!("{}/vectorAdd.o", out_dir))
        .status()
        .unwrap();
    */

    Command::new("ar")
        .args(&["crus", "libgpu.a", "vectorAdd.o"])
        .current_dir(&Path::new(&out_dir))
        .status()
        .unwrap();

    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=gpu");
}
