use std::path::PathBuf;
use std::process::Command;

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let manifest = env!("CARGO_MANIFEST_DIR");

    // macOS refuses microphone / speech recognition access unless the process
    // has an Info.plist with usage descriptions. Embedding it in the binary
    // makes a plain `cargo run` work, not only the .app bundle.
    let plist = format!("{manifest}/macos/Info.plist");
    println!("cargo:rerun-if-changed={plist}");
    println!("cargo:rustc-link-arg=-Wl,-sectcreate,__TEXT,__info_plist,{plist}");

    // Compile the Swift shim around SpeechAnalyzer (a Swift-only API) into a
    // static library and link it, together with the Swift runtime.
    let source = format!("{manifest}/macos/Transcriber.swift");
    println!("cargo:rerun-if-changed={source}");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let arch = match std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "arm64",
        _ => "x86_64",
    };
    let status = Command::new("swiftc")
        .args(["-parse-as-library", "-emit-library", "-static", "-O"])
        .args(["-swift-version", "5", "-module-name", "TWASpeech"])
        .args(["-target", &format!("{arch}-apple-macos14.0")])
        .arg(&source)
        .arg("-o")
        .arg(out.join("libtwaspeech.a"))
        .status()
        .expect("swiftc not found; install the Xcode command line tools");
    assert!(status.success(), "swiftc failed to compile {source}");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=twaspeech");

    // Where the linker finds the Swift runtime and compatibility libraries.
    let run = |cmd: &str, args: &[&str]| {
        let o = Command::new(cmd).args(args).output().expect("xcrun failed");
        String::from_utf8(o.stdout).unwrap().trim().to_string()
    };
    let sdk = run("xcrun", &["--show-sdk-path"]);
    let swiftc = PathBuf::from(run("xcrun", &["--find", "swiftc"]));
    let toolchain_lib = swiftc.parent().unwrap().parent().unwrap().join("lib/swift/macosx");
    println!("cargo:rustc-link-search=native={sdk}/usr/lib/swift");
    println!("cargo:rustc-link-search=native={}", toolchain_lib.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
}
