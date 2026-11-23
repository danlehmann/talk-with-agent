//! High-accuracy local transcription with Whisper (whisper.cpp).
//!
//! Platform-independent: a voice backend records the utterance as 16 kHz mono
//! samples and hands it over; the platform's own recognizer is only needed for
//! the live preview and as a fallback while the model is not available.
//! The model is downloaded once, on first launch.

use super::Waker;
use crate::state::data_dir;
use std::path::Path;
use std::process::Command;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub const SAMPLE_RATE: usize = 16_000;

const MODEL_NAME: &str = "Whisper large-v3";
const MODEL_FILE: &str = "ggml-large-v3.bin";
const MODEL_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin";
const MODEL_BYTES: u64 = 3095033483;

/// Nudges the model towards the vocabulary of these conversations.
const PROMPT: &str = "A spoken conversation with an AI coding agent about software development.";

#[derive(Clone)]
enum Status {
    Downloading(u64),
    Loading,
    Ready,
    Failed(String),
}

type Reply = Box<dyn FnOnce(Option<String>) + Send>;

pub struct Whisper {
    status: Arc<Mutex<Status>>,
    jobs: Sender<(Vec<f32>, Reply)>,
}

impl Whisper {
    /// Starts a worker thread that fetches and loads the model, then transcribes.
    pub fn start(waker: Waker) -> Self {
        let status = Arc::new(Mutex::new(Status::Loading));
        let (jobs, queue) = channel::<(Vec<f32>, Reply)>();
        let shared = status.clone();
        std::thread::spawn(move || {
            let set = |s: Status| {
                *shared.lock().unwrap() = s;
                waker();
            };
            let path = data_dir().join("models").join(MODEL_FILE);
            if !path.exists()
                && let Err(e) = download(&path, &|bytes| set(Status::Downloading(bytes)))
            {
                return set(Status::Failed(format!("model download failed: {e}")));
            }
            set(Status::Loading);
            whisper_rs::install_logging_hooks(); // keep whisper.cpp off stderr
            let context = WhisperContext::new_with_params(&path, WhisperContextParameters::default());
            let mut state = match context.as_ref().map_err(|e| e.to_string()).and_then(|c| {
                c.create_state().map_err(|e| e.to_string())
            }) {
                Ok(state) => state,
                Err(e) => return set(Status::Failed(format!("could not load model: {e}"))),
            };
            set(Status::Ready);
            // whisper.cpp's Metal backend asserts in its static destructors when
            // the process exits with the model still loaded, which macOS reports
            // as a crash on quit. Registered now, this handler runs before those
            // destructors and skips them. (All app state is saved as it changes.)
            unsafe { atexit(exit_now) };

            for (mut samples, reply) in queue {
                // whisper.cpp rejects clips shorter than a second.
                if samples.len() < SAMPLE_RATE * 11 / 10 {
                    samples.resize(SAMPLE_RATE * 11 / 10, 0.0);
                }
                let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                params.set_print_progress(false);
                params.set_print_realtime(false);
                params.set_print_special(false);
                params.set_print_timestamps(false);
                params.set_language(Some("auto"));
                params.set_no_context(true);
                params.set_suppress_nst(true);
                params.set_initial_prompt(PROMPT);
                let text = state.full(params, &samples).ok().map(|_| {
                    let parts: Vec<String> = state
                        .as_iter()
                        // Skip stretches the model itself considers silence.
                        .filter(|s| s.no_speech_probability() < 0.6)
                        .filter_map(|s| s.to_str_lossy().ok().map(|t| t.trim().to_string()))
                        .collect();
                    parts.join(" ").trim().to_string()
                });
                reply(text);
            }
        });
        Whisper { status, jobs }
    }

    /// Transcribe 16 kHz mono audio in the background; `reply` gets the text,
    /// or None if it failed. Returns false (without calling `reply`) if the
    /// model is not ready, so the caller can use its fallback.
    pub fn transcribe(&self, samples: Vec<f32>, reply: impl FnOnce(Option<String>) + Send + 'static) -> bool {
        matches!(*self.status.lock().unwrap(), Status::Ready)
            && self.jobs.send((samples, Box::new(reply))).is_ok()
    }

    /// One line for the UI.
    pub fn describe(&self) -> String {
        match &*self.status.lock().unwrap() {
            Status::Downloading(bytes) => format!(
                "Downloading {MODEL_NAME} ({}%)… using the system recognizer meanwhile",
                (bytes * 100 / MODEL_BYTES).min(99)
            ),
            Status::Loading => format!("Loading {MODEL_NAME}…"),
            Status::Ready => format!("Transcription: {MODEL_NAME}, on this device"),
            Status::Failed(e) => format!("{MODEL_NAME} unavailable ({e}); using the system recognizer"),
        }
    }
}

/// Whether `whisper`'s text can be a better transcription of the same
/// speech as `reference` (the live recognizer's text). Whisper sometimes
/// invents text on quiet audio, typically a repeated "Thank you."; that shows
/// as a transcript that shares few words with the reference or is much longer.
pub fn plausible(whisper: &str, reference: &str) -> bool {
    let words = |t: &str| -> Vec<String> {
        t.split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(str::to_lowercase)
            .collect()
    };
    let (ours, theirs) = (words(whisper), words(reference));
    if theirs.len() < 3 {
        // Too short to compare word by word ("Yeah." vs "Yes."): accept
        // anything short, except Whisper's stock inventions.
        let invented = ["thank", "thanks", "bye", "subscribe", "watching"];
        return ours.len() <= theirs.len() + 4
            && !ours.iter().any(|w| invented.contains(&w.as_str()) && !theirs.contains(w));
    }
    let shared = theirs.iter().filter(|w| ours.contains(w)).count();
    shared * 10 >= theirs.len() * 4 && ours.len() <= theirs.len() * 2 + 5
}

unsafe extern "C" {
    fn atexit(callback: extern "C" fn()) -> i32;
    fn _exit(code: i32) -> !;
}

extern "C" fn exit_now() {
    unsafe { _exit(0) }
}

fn download(path: &Path, progress: &dyn Fn(u64)) -> Result<(), String> {
    let dir = path.parent().unwrap();
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let part = path.with_extension("part");
    let mut child = Command::new("curl")
        .args(["-fsSL", "--retry", "3", "-C", "-", "-o"])
        .arg(&part)
        .arg(MODEL_URL)
        .spawn()
        .map_err(|e| format!("could not run curl: {e}"))?;
    loop {
        progress(std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0));
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) if status.success() => break,
            Some(status) => return Err(format!("curl exited with {status}")),
            None => std::thread::sleep(Duration::from_millis(500)),
        }
    }
    std::fs::rename(&part, path).map_err(|e| e.to_string())
}
