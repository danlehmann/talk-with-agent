//! Platform voice abstraction: text-to-speech and speech-to-text.
//!
//! Everything OS-specific lives behind the [`Voice`] trait. To support a new
//! platform, add a module next to `macos.rs` that implements it and wire it
//! into [`create`]. Nothing else in the app touches platform APIs.

#[cfg(target_os = "macos")]
mod macos;
// Platform-independent, but so far only the macOS backend records audio for it.
#[cfg(target_os = "macos")]
mod whisper;
#[cfg(target_os = "macos")]
mod kokoro;
#[cfg(not(target_os = "macos"))]
mod unsupported;

use std::sync::Arc;

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

    /// One line describing the transcription engine or its setup progress.
    fn status(&self) -> Option<String> {
        None
    }

    /// Drain pending events. Called every UI frame.
    fn poll(&mut self) -> Vec<VoiceEvent>;
}

pub fn create(waker: Waker) -> Box<dyn Voice> {
    #[cfg(target_os = "macos")]
    return Box::new(macos::MacVoice::new(waker));
    #[cfg(not(target_os = "macos"))]
    return Box::new(unsupported::NoVoice::new(waker));
}
