fn copy_directory(
    source: impl AsRef<std::path::Path> + std::fmt::Debug,
    destination: impl AsRef<std::path::Path>,
) {
    let _ = std::fs::create_dir_all(&destination);
    let expect_message = format!("{:?} exists", &source);
    for entry in std::fs::read_dir(source).expect(&expect_message) {
        let entry = entry.unwrap();
        let file_type = entry.file_type().unwrap();
        if file_type.is_dir() {
            copy_directory(entry.path(), destination.as_ref().join(entry.file_name()));
        } else {
            std::fs::copy(entry.path(), destination.as_ref().join(entry.file_name())).unwrap();
        }
    }
}

fn bash() -> std::process::Command {
    match std::env::var("CARGO_CFG_TARGET_OS").unwrap().as_str() {
        "windows" => {
            let mut command = std::process::Command::new("bash");
            command.env("CC", "cl");
            command
        }
        _ => std::process::Command::new("bash"),
    }
}

fn make() -> std::process::Command {
    match std::env::var("CARGO_CFG_TARGET_OS").unwrap().as_str() {
        "windows" => {
            let mut command = std::process::Command::new("make");
            command.env("CC", "cl");
            command
        }
        _ => std::process::Command::new("make"),
    }
}

fn x264() -> &'static str {
    match std::env::var("CARGO_CFG_TARGET_OS").unwrap().as_str() {
        "windows" => "libx264",
        _ => "x264",
    }
}

fn main() -> std::io::Result<()> {
    let cargo_manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rerun-if-changed={cargo_manifest_dir}/src/mp4/x264.h",);
    let x264_directory: std::path::PathBuf =
        [&cargo_manifest_dir, "src", "mp4", "x264"].iter().collect();
    let x264_build_directory: std::path::PathBuf =
        [&cargo_manifest_dir, "src", "mp4", "x264-build"]
            .iter()
            .collect();
    copy_directory(x264_directory, &x264_build_directory);
    println!("cargo:rustc-link-search=native={cargo_manifest_dir}/src/mp4/x264-build/");
    println!("cargo:rustc-link-lib=static={}", x264());
    if !bash()
        .args([
            "configure",
            "--disable-cli",
            "--enable-static",
            "--disable-interlaced",
            "--bit-depth=8",
            "--enable-strip",
            "--disable-avs",
            "--disable-swscale",
            "--disable-lavf",
            "--disable-ffms",
            "--disable-gpac",
            "--disable-lsmash",
            "--enable-pic",
        ])
        .current_dir(&x264_build_directory)
        .status()
        .expect("Failed to spawn 'bash configure' for x264")
        .success()
    {
        panic!("Failed to configure x264");
    }
    if !make()
        .args(["clean"])
        .current_dir(&x264_build_directory)
        .status()
        .expect("Failed to spawn 'make clean' for x264")
        .success()
    {
        panic!("Failed to make clean x264");
    }
    if !make()
        .current_dir(&x264_build_directory)
        .status()
        .expect("Failed to spawn 'make' for x264")
        .success()
    {
        panic!("Failed to make x264");
    }
    let bindings = bindgen::Builder::default()
        .header("src/mp4/x264.h")
        .generate()
        .expect("Unable to generate bindings");
    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("x264_bindings.rs"))
        .expect("Couldn't write bindings");
    Ok(())
}
