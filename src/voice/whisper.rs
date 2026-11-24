//! High-accuracy local transcription with Whisper (whisper.cpp).
//!
//! Platform-independent: a voice backend records the utterance as 16 kHz mono
//! samples and hands it over; the platform's own recognizer is only needed for
//! the live preview and as a fallback while no model is available.
//! Which model to use is the user's choice; it is downloaded once, and only
//! after the user agreed to downloads.

use super::download::fetch;
use super::Waker;
use crate::state::data_dir;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub const SAMPLE_RATE: usize = 16_000;

pub struct Model {
    pub id: &'static str,
    name: &'static str,
    /// What sets it apart, for the list of choices.
    note: &'static str,
    file: &'static str,
    bytes: u64,
    sha256: &'static str,
}

/// From https://huggingface.co/ggerganov/whisper.cpp
const MODELS: [Model; 2] = [
    Model {
        id: "large-v3-turbo-q5",
        name: "Whisper large-v3 turbo",
        note: "small and fast",
        file: "ggml-large-v3-turbo-q5_0.bin",
        bytes: 574_041_195,
        sha256: "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
    },
    Model {
        id: "large-v3",
        name: "Whisper large-v3",
        note: "most accurate, needs about 5 GB of memory",
        file: "ggml-large-v3.bin",
        bytes: 3_095_033_483,
        sha256: "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
    },
];
/// Id of the choice to do without Whisper.
const NONE: &str = "none";

impl Model {
    fn path(&self) -> PathBuf {
        data_dir().join("models").join(self.file)
    }

    fn url(&self) -> String {
        format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}", self.file)
    }
}

/// Nudges the model towards the vocabulary of these conversations.
const PROMPT: &str = "A spoken conversation with an AI coding agent about software development.";

#[derive(Clone)]
enum Status {
    /// The user chose to do without.
    Off,
    /// Not on disk, and downloads have not been agreed to.
    Missing,
    Downloading(u64),
    Loading,
    Ready,
    Failed(String),
}

type Reply = Box<dyn FnOnce(Option<String>) + Send>;

enum Job {
    Transcribe(Vec<f32>, Reply),
    /// The choice of model or the permission to download changed.
    Reconsider,
}

struct Wanted {
    /// Index into [`MODELS`]; None = do without.
    model: Option<usize>,
    may_download: bool,
}

struct Shared {
    status: Mutex<Status>,
    wanted: Mutex<Wanted>,
}

pub struct Whisper {
    shared: Arc<Shared>,
    jobs: Sender<Job>,
}

impl Whisper {
    /// Starts a worker thread that loads the chosen model (fetching it first
    /// if allowed), then transcribes. `model` is an id from [`Whisper::models`];
    /// without one, a model already on disk is used, else the smallest.
    pub fn start(waker: Waker, model: Option<&str>, may_download: bool) -> Self {
        let model = match model {
            Some(NONE) => None,
            _ => model
                .and_then(|id| MODELS.iter().position(|m| m.id == id))
                // No (valid) choice yet: the best model already on disk.
                .or_else(|| MODELS.iter().rposition(|m| m.path().exists()))
                .or(Some(0)),
        };
        let shared = Arc::new(Shared {
            status: Mutex::new(Status::Loading),
            wanted: Mutex::new(Wanted { model, may_download }),
        });
        let (jobs, queue) = channel::<Job>();
        let worker = shared.clone();
        std::thread::spawn(move || {
            let set = |s: Status| {
                *worker.status.lock().unwrap() = s;
                waker();
            };
            let wanted = || {
                let wanted = worker.wanted.lock().unwrap();
                (wanted.model, wanted.may_download)
            };
            whisper_rs::install_logging_hooks(); // keep whisper.cpp off stderr
            // The loaded model's state (which keeps the model itself alive).
            let mut loaded: Option<whisper_rs::WhisperState> = None;
            // The choice that has been dealt with, successfully or not.
            let mut settled: Option<Option<usize>> = None;
            loop {
                let (model, may_download) = wanted();
                if settled != Some(model) {
                    loaded = None; // frees the previous model first
                    let status = match model {
                        None => Status::Off,
                        Some(index) => {
                            let chosen = &MODELS[index];
                            let path = chosen.path();
                            let fetched = if path.exists() {
                                Ok(())
                            } else if may_download {
                                set(Status::Downloading(0));
                                fetch(&chosen.url(), &path, chosen.sha256, &|bytes| {
                                    set(Status::Downloading(bytes));
                                    wanted().0 == model // else the user chose another
                                })
                                .map_err(|e| Status::Failed(format!("download failed: {e}")))
                            } else {
                                Err(Status::Missing)
                            };
                            if wanted().0 != model {
                                continue; // cancelled for another choice
                            }
                            fetched
                                .and_then(|()| {
                                    set(Status::Loading);
                                    WhisperContext::new_with_params(&path, WhisperContextParameters::default())
                                        .and_then(|context| context.create_state())
                                        .map_err(|e| Status::Failed(format!("could not load model: {e}")))
                                })
                                .map(|state| {
                                    loaded = Some(state);
                                    // whisper.cpp's Metal backend asserts in its static
                                    // destructors when the process exits with a model loaded,
                                    // which macOS reports as a crash on quit. Registered now,
                                    // this handler runs before those destructors and skips
                                    // them. (All app state is saved as it changes.)
                                    static ONCE: std::sync::Once = std::sync::Once::new();
                                    ONCE.call_once(|| unsafe {
                                        atexit(exit_now);
                                    });
                                    Status::Ready
                                })
                                .unwrap_or_else(|status| status)
                        }
                    };
                    set(status);
                    settled = Some(model);
                }
                let Ok(job) = queue.recv() else { return };
                let Job::Transcribe(mut samples, reply) = job else {
                    // Try again what failed or was not allowed before.
                    if matches!(*worker.status.lock().unwrap(), Status::Missing | Status::Failed(_)) {
                        settled = None;
                    }
                    continue;
                };
                let Some(state) = loaded.as_mut() else {
                    reply(None);
                    continue;
                };
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
        Whisper { shared, jobs }
    }

    /// The choices, as (id, label).
    pub fn models() -> Vec<(String, String)> {
        let mut models: Vec<(String, String)> = MODELS
            .iter()
            .map(|m| {
                let size = if m.path().exists() { "installed".to_string() } else { super::megabytes(m.bytes) + " download" };
                (m.id.to_string(), format!("{} ({}; {size})", m.name, m.note))
            })
            .collect();
        models.push((NONE.to_string(), "System recognizer only".to_string()));
        models
    }

    pub fn model(&self) -> String {
        self.shared.wanted.lock().unwrap().model.map_or(NONE, |index| MODELS[index].id).to_string()
    }

    pub fn set_model(&self, id: &str) {
        let model = MODELS.iter().position(|m| m.id == id);
        if model.is_some() || id == NONE {
            self.shared.wanted.lock().unwrap().model = model;
            let _ = self.jobs.send(Job::Reconsider);
        }
    }

    /// Every model: (id, name, download size in bytes, whether it is on disk).
    pub fn all() -> Vec<(String, String, u64, bool)> {
        MODELS.iter().map(|m| (m.id.to_string(), m.name.to_string(), m.bytes, m.path().exists())).collect()
    }

    /// Delete a model from disk. If it was the one in use, the system
    /// recognizer takes over.
    pub fn remove(&self, id: &str) {
        let Some(index) = MODELS.iter().position(|m| m.id == id) else { return };
        if self.shared.wanted.lock().unwrap().model == Some(index) {
            self.set_model(NONE);
        }
        let _ = std::fs::remove_file(MODELS[index].path());
    }

    /// The user agreed to downloading models.
    pub fn allow_downloads(&self) {
        self.shared.wanted.lock().unwrap().may_download = true;
        let _ = self.jobs.send(Job::Reconsider);
    }

    /// The chosen model, if it still has to be downloaded and that has not
    /// been agreed to: its name and size in bytes.
    pub fn missing(&self) -> Option<(String, u64)> {
        let wanted = self.shared.wanted.lock().unwrap();
        let model = &MODELS[wanted.model?];
        (!wanted.may_download && !model.path().exists()).then(|| (model.name.to_string(), model.bytes))
    }

    fn name(&self) -> &'static str {
        self.shared.wanted.lock().unwrap().model.map_or("Whisper", |index| MODELS[index].name)
    }

    /// Transcribe 16 kHz mono audio in the background; `reply` gets the text,
    /// or None if it failed. Returns false (without calling `reply`) if the
    /// model is not ready, so the caller can use its fallback.
    pub fn transcribe(&self, samples: Vec<f32>, reply: impl FnOnce(Option<String>) + Send + 'static) -> bool {
        matches!(*self.shared.status.lock().unwrap(), Status::Ready)
            && self.jobs.send(Job::Transcribe(samples, Box::new(reply))).is_ok()
    }

    /// One line for the UI.
    pub fn describe(&self) -> String {
        let name = self.name();
        let total = self.shared.wanted.lock().unwrap().model.map_or(1, |index| MODELS[index].bytes);
        match &*self.shared.status.lock().unwrap() {
            Status::Off => "Transcription: system recognizer".to_string(),
            Status::Missing => format!("Transcription: system recognizer ({name} not downloaded)"),
            Status::Downloading(bytes) => format!(
                "Downloading {name} ({}%)… using the system recognizer meanwhile",
                (bytes * 100 / total).min(99)
            ),
            Status::Loading => format!("Loading {name}…"),
            Status::Ready => format!("Transcription: {name}, on this device"),
            Status::Failed(e) => format!("{name} unavailable ({e}); using the system recognizer"),
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
