//! Fetching model files: resumable, and verified against a known checksum
//! before anything uses them.

use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

/// Download `url` to `dest` (via `<dest>.part`, which an interrupted download
/// leaves behind and the next one continues). `progress` gets the bytes so
/// far and returns false to cancel.
pub fn fetch(url: &str, dest: &Path, sha256: &str, progress: &dyn Fn(u64) -> bool) -> Result<(), String> {
    let folder = dest.parent().ok_or("no folder")?;
    std::fs::create_dir_all(folder).map_err(|e| e.to_string())?;
    let mut part = dest.as_os_str().to_owned();
    part.push(".part");
    let part = Path::new(&part);
    let mut child = Command::new("curl")
        .args(["-fsSL", "--retry", "3", "-C", "-", "-o"])
        .arg(part)
        .arg(url)
        .spawn()
        .map_err(|e| format!("could not run curl: {e}"))?;
    loop {
        if !progress(std::fs::metadata(part).map(|m| m.len()).unwrap_or(0)) {
            let _ = child.kill();
            let _ = child.wait();
            return Err("cancelled".into());
        }
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) if status.success() => break,
            Some(status) => return Err(format!("curl exited with {status}")),
            None => std::thread::sleep(Duration::from_millis(500)),
        }
    }
    let found = checksum(part)?;
    if found != sha256 {
        let _ = std::fs::remove_file(part);
        return Err("the downloaded file is not the expected one (checksum mismatch)".into());
    }
    std::fs::rename(part, dest).map_err(|e| e.to_string())
}

fn checksum(path: &Path) -> Result<String, String> {
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    #[test]
    fn checksums_a_file() {
        let path = std::env::temp_dir().join(format!("twa-checksum-{}", std::process::id()));
        std::fs::write(&path, b"abc").unwrap();
        let sum = super::checksum(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(sum, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
}
