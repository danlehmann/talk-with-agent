//! Platform voice abstraction: text-to-speech and speech-to-text.
//!
//! Everything OS-specific lives behind the [`Voice`] trait. To support a new
//! platform, add a module next to `macos.rs` that implements it and wire it
//! into [`create`]. Nothing else in the app touches platform APIs.

#[cfg(target_os = "macos")]
mod macos;
// Platform-independent, but so far only the macOS backend records audio for it.
#[cfg(target_os = "macos")]
mod download;
#[cfg(target_os = "macos")]
mod kokoro;
#[cfg(not(target_os = "macos"))]
mod unsupported;
#[cfg(target_os = "macos")]
mod whisper;

use std::sync::Arc;

/// A downloadable speech model, for the settings screen.
pub struct Model {
    pub id: String,
    pub name: String,
    /// Download size in bytes.
    pub bytes: u64,
    pub installed: bool,
}

/// A download size for display: "574 MB", "3.1 GB".
pub fn megabytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else {
        format!("{} MB", bytes / 1_000_000)
    }
}

pub enum VoiceEvent {
    /// Best transcription so far of what the user is currently saying.
    Partial(String),
    /// Listening has ended. Contains the final transcription (may be empty).
    Finished(String),
    /// Listening failed or could not start. Listening has ended.
    Error(String),
    /// The user pressed play/pause or mute on a headset or keyboard:
    /// Some(true) = pause, Some(false) = resume, None = toggle.
    PauseRequested(Option<bool>),
}

/// Wakes the UI loop; call it after queuing an event from a background thread.
pub type Waker = Arc<dyn Fn() + Send + Sync>;

/// All methods are called from the UI thread.
pub trait Voice {
    /// Queue text to be spoken after anything already being spoken. `id` is
    /// the caller's handle for this utterance.
    fn speak(&mut self, id: u64, text: &str);
    /// Stop now and drop everything queued. If that cut something off, returns
    /// the id of the utterance being spoken and the part of its text that was
    /// actually spoken (as exactly as the platform can tell).
    fn stop_speaking(&mut self) -> Option<(u64, String)>;
    fn is_speaking(&self) -> bool;
    /// Pause speech where it is (`true`) or carry on from there (`false`).
    /// Still counts as speaking while held.
    fn hold_speaking(&mut self, _hold: bool) {}
    /// The utterance being spoken right now and how many bytes of its text
    /// have been spoken so far, for highlighting it on screen.
    fn speaking_progress(&self) -> Option<(u64, usize)> {
        None
    }

    /// Start capturing the microphone and transcribing. Results arrive via
    /// [`Voice::poll`]: zero or more `Partial`s, then one `Finished` or `Error`.
    fn start_listening(&mut self);
    /// Stop capturing; a `Finished` event with the final text follows.
    fn stop_listening(&mut self);

    /// Forget the audio captured so far in the current listening session,
    /// except the last moment (the user may just have started talking).
    /// Called when the agent stops talking, so its voice isn't transcribed.
    fn trim_recording(&mut self) {}

    /// A hands-free conversation started or ended. While one is going on the
    /// platform should keep the computer awake, and report headset/media
    /// play-pause presses as [`VoiceEvent::PauseRequested`].
    fn set_conversing(&mut self, _on: bool) {}
    /// The conversation was paused or resumed (to show in system media controls).
    fn set_paused(&mut self, _paused: bool) {}

    /// Speaking voices the user can choose from, as (id, label). May be empty.
    fn voices(&self) -> Vec<(String, String)> {
        Vec::new()
    }
    /// Id of the voice in use, if there is a choice.
    fn voice(&self) -> Option<String> {
        None
    }
    /// Switch to another voice from [`Voice::voices`]; unknown ids are ignored.
    fn set_voice(&mut self, _id: &str) {}

    /// Transcription engines the user can choose from, as (id, label).
    fn transcribers(&self) -> Vec<(String, String)> {
        Vec::new()
    }
    fn transcriber(&self) -> Option<String> {
        None
    }
    fn set_transcriber(&mut self, _id: &str) {}

    /// Every model this backend can use.
    fn models(&self) -> Vec<Model> {
        Vec::new()
    }
    /// Delete a model from disk. If it was in use, the choice of voice or
    /// transcriber changes to do without it.
    fn remove_model(&mut self, _id: &str) {}

    /// Download a model (the user asked for exactly this one) and use it.
    fn download_model(&mut self, _id: &str) {}

    /// Models that are needed but not on disk, as (name, download size in
    /// bytes). They are only downloaded after [`Voice::allow_downloads`].
    fn missing_models(&self) -> Vec<(String, u64)> {
        Vec::new()
    }
    /// The user agreed to downloading models, now and in future.
    fn allow_downloads(&mut self) {}

    /// One line describing the transcription engine or its setup progress.
    fn status(&self) -> Option<String> {
        None
    }

    /// Drain pending events. Called every UI frame.
    fn poll(&mut self) -> Vec<VoiceEvent>;
}

/// `voice` and `transcriber` are the saved choices from [`Voice::voices`] and
/// [`Voice::transcribers`], if any; `may_download` whether the user has
/// agreed to downloading models.
pub fn create(
    waker: Waker,
    voice: Option<&str>,
    transcriber: Option<&str>,
    may_download: bool,
) -> Box<dyn Voice> {
    #[cfg(target_os = "macos")]
    return Box::new(macos::MacVoice::new(
        waker,
        voice,
        transcriber,
        may_download,
    ));
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (voice, transcriber, may_download);
        Box::new(unsupported::NoVoice::new(waker))
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use std::sync::Arc;

    /// With no models on disk and no consent, nothing is fetched: the models
    /// are reported missing, and choosing to do without Whisper clears it.
    #[test]
    fn nothing_downloads_without_consent() {
        let dir = crate::state::use_test_data_dir();
        let whisper = super::whisper::Whisper::start(Arc::new(|| {}), None, false);
        let kokoro = super::kokoro::Kokoro::start(Arc::new(|| {}), None, false);
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(whisper.missing().map(|(_, bytes)| bytes), Some(574_041_195));
        assert!(
            whisper.describe().contains("not downloaded"),
            "{}",
            whisper.describe()
        );
        assert!(kokoro.missing().is_some());
        assert!(!kokoro.ready());
        kokoro.set_voice("none");
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(kokoro.missing().is_none());
        assert_eq!(kokoro.describe(), "Voice: off");
        whisper.set_model("large-v3");
        assert_eq!(
            whisper.missing().map(|(_, bytes)| bytes),
            Some(3_095_033_483)
        );
        whisper.set_model("none");
        assert!(whisper.missing().is_none());
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(whisper.describe(), "Transcription: system recognizer");
        let leftovers: Vec<_> = std::fs::read_dir(dir.join("models"))
            .map(|d| d.flatten().collect())
            .unwrap_or_default();
        assert!(
            leftovers.is_empty(),
            "something was downloaded: {leftovers:?}"
        );
    }
}
