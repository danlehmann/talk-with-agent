//! macOS voice backend using the system frameworks.
//!
//! Output: the Kokoro neural voice (see kokoro.rs). Until its model has been
//! downloaded, replies are only shown, not spoken.
//! Input: SpeechAnalyzer on macOS 26+ (always on-device; reached through the
//! Swift shim in macos/Transcriber.swift), otherwise the older
//! SFSpeechRecognizer + AVAudioEngine.
//!
//! Requires NSMicrophoneUsageDescription and NSSpeechRecognitionUsageDescription
//! in the Info.plist (see macos/Info.plist; build.rs embeds it in the binary).

use super::kokoro::Kokoro;
use super::whisper::Whisper;
use super::{Voice, VoiceEvent, Waker};
use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::Bool;
use objc2::AnyThread;
use objc2_avf_audio::{
    AVAudioApplication, AVAudioEngine, AVAudioPCMBuffer, AVAudioTime,
};
use objc2_foundation::NSError;
use std::ffi::{c_char, c_void, CStr};
use objc2_speech::{
    SFSpeechAudioBufferRecognitionRequest, SFSpeechRecognitionResult, SFSpeechRecognitionTask,
    SFSpeechRecognizer, SFSpeechRecognizerAuthorizationStatus,
};
use std::ptr::NonNull;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};

// Implemented in macos/Transcriber.swift.
unsafe extern "C" {
    fn twa_transcriber_available() -> bool;
    fn twa_transcriber_start(
        generation: u64,
        context: *mut c_void,
        callback: extern "C" fn(*mut c_void, i32, u64, *const c_char, *const f32, usize),
    );
    fn twa_transcriber_stop();
    fn twa_transcriber_trim();
    fn twa_set_conversing(
        on: bool,
        context: *mut c_void,
        pause_requested: extern "C" fn(*mut c_void, i32),
    );
    fn twa_set_paused(paused: bool);
}

/// What the Swift shim calls back with: where to send messages, and how to wake the UI.
struct ShimContext {
    tx: Sender<Msg>,
    waker: Waker,
}

extern "C" fn shim_callback(
    context: *mut c_void,
    kind: i32,
    generation: u64,
    text: *const c_char,
    samples: *const f32,
    count: usize,
) {
    let context = unsafe { &*(context as *const ShimContext) };
    let text = if text.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
    };
    let audio = if samples.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(samples, count) }.to_vec()
    };
    let _ = context.tx.send(match kind {
        0 => Msg::Partial(generation, text),
        1 => Msg::Done(generation, Ok(text), audio),
        _ => Msg::Done(generation, Err(text), audio),
    });
    (context.waker)();
}

/// request: 0 = toggle, 1 = pause, 2 = resume.
extern "C" fn pause_callback(context: *mut c_void, request: i32) {
    let context = unsafe { &*(context as *const ShimContext) };
    let _ = context.tx.send(Msg::PauseRequested(match request {
        1 => Some(true),
        2 => Some(false),
        _ => None,
    }));
    (context.waker)();
}

/// Messages from framework callbacks (arbitrary threads) to the UI thread.
enum Msg {
    SpeechAuth(bool),
    MicAuth(bool),
    Partial(u64, String),
    /// Listening ended: the recognizer's text (or error), and the recorded
    /// utterance (16 kHz mono) if the recognizer captured it.
    Done(u64, Result<String, String>, Vec<f32>),
    /// Play/pause or mute was pressed on a headset or keyboard.
    PauseRequested(Option<bool>),
    /// Whisper finished re-transcribing an utterance.
    Refined(u64, String),
}

/// The SFSpeechRecognizer objects of a listening session (pre-macOS 26 path).
struct Legacy {
    engine: Retained<AVAudioEngine>,
    request: Retained<SFSpeechAudioBufferRecognitionRequest>,
    task: Retained<SFSpeechRecognitionTask>,
}

struct Active {
    generation: u64,
    /// None when the Swift SpeechAnalyzer shim is doing the listening.
    legacy: Option<Legacy>,
    last_partial: String,
    /// Set once the microphone has been stopped and we await the final result.
    stopping: Option<Instant>,
}

impl Active {
    fn stop_audio(&mut self) {
        if self.stopping.is_none() {
            match &self.legacy {
                Some(l) => unsafe {
                    l.engine.stop();
                    l.engine.inputNode().removeTapOnBus(0);
                    l.request.endAudio();
                },
                None => unsafe { twa_transcriber_stop() },
            }
            self.stopping = Some(Instant::now());
        }
    }
}

pub struct MacVoice {
    recognizer: Option<Retained<SFSpeechRecognizer>>,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    waker: Waker,
    speech_auth: Option<bool>,
    mic_auth: Option<bool>,
    auth_requested: bool,
    /// start_listening was called but we are still waiting for permissions.
    want_listen: bool,
    active: Option<Active>,
    generation: u64,
    out: Vec<VoiceEvent>,
    /// Handed to the Swift shim; lives as long as the app.
    shim_context: *const ShimContext,
    whisper: Whisper,
    kokoro: Kokoro,
    /// Generation of the utterance Whisper is transcribing right now.
    refining: Option<u64>,
}

impl MacVoice {
    pub fn new(waker: Waker, voice: Option<&str>, transcriber: Option<&str>, may_download: bool) -> Self {
        let (tx, rx) = channel();
        let shim_context = Box::into_raw(Box::new(ShimContext {
            tx: tx.clone(),
            waker: waker.clone(),
        }));
        MacVoice {
            whisper: Whisper::start(waker.clone(), transcriber, may_download),
            kokoro: Kokoro::start(waker.clone(), voice, may_download),
            refining: None,
            shim_context,
            recognizer: None,
            tx,
            rx,
            waker,
            speech_auth: None,
            mic_auth: None,
            auth_requested: false,
            want_listen: false,
            active: None,
            generation: 0,
            out: Vec::new(),
        }
    }

    fn request_permissions(&mut self) {
        if self.auth_requested {
            return;
        }
        self.auth_requested = true;

        if unsafe { twa_transcriber_available() } {
            // SpeechAnalyzer runs on-device and only needs the microphone; the
            // "send speech to Apple" consent is for the old recognizer.
            self.speech_auth = Some(true);
        } else {
            let (tx, waker) = (self.tx.clone(), self.waker.clone());
            let block = RcBlock::new(move |status: SFSpeechRecognizerAuthorizationStatus| {
                let ok = status == SFSpeechRecognizerAuthorizationStatus::Authorized;
                let _ = tx.send(Msg::SpeechAuth(ok));
                waker();
            });
            unsafe { SFSpeechRecognizer::requestAuthorization(&block) };
        }

        let (tx, waker) = (self.tx.clone(), self.waker.clone());
        let block = RcBlock::new(move |granted: Bool| {
            let _ = tx.send(Msg::MicAuth(granted.as_bool()));
            waker();
        });
        unsafe { AVAudioApplication::requestRecordPermissionWithCompletionHandler(&block) };
    }

    fn begin(&mut self) -> Result<(), String> {
        if unsafe { twa_transcriber_available() } {
            self.generation += 1;
            eprintln!("[voice] listening #{}: SpeechAnalyzer", self.generation);
            unsafe {
                twa_transcriber_start(self.generation, self.shim_context as *mut c_void, shim_callback)
            };
            self.active = Some(Active {
                generation: self.generation,
                legacy: None,
                last_partial: String::new(),
                stopping: None,
            });
            return Ok(());
        }

        if self.recognizer.is_none() {
            self.recognizer = unsafe { SFSpeechRecognizer::init(SFSpeechRecognizer::alloc()) };
        }
        let recognizer = self
            .recognizer
            .as_ref()
            .ok_or("Speech recognition does not support your system language.")?;
        if !unsafe { recognizer.isAvailable() } {
            return Err("Speech recognition is not available right now. Check that Dictation is enabled in System Settings › Keyboard.".into());
        }

        self.generation += 1;
        let generation = self.generation;

        let request = unsafe { SFSpeechAudioBufferRecognitionRequest::new() };
        unsafe {
            request.setShouldReportPartialResults(true);
            request.setAddsPunctuation(true);
            if recognizer.supportsOnDeviceRecognition() {
                // Private, and no one-minute limit.
                request.setRequiresOnDeviceRecognition(true);
            }
        }

        let engine = unsafe { AVAudioEngine::new() };
        let input = unsafe { engine.inputNode() };
        let format = unsafe { input.outputFormatForBus(0) };
        if unsafe { format.sampleRate() } <= 0.0 || unsafe { format.channelCount() } == 0 {
            return Err("No microphone available.".into());
        }

        eprintln!(
            "[voice] listening #{generation}: input {} Hz, {} ch, on-device: {}",
            unsafe { format.sampleRate() },
            unsafe { format.channelCount() },
            unsafe { recognizer.supportsOnDeviceRecognition() },
        );

        // Called on a realtime audio thread.
        let tap_request = request.clone();
        let tap = RcBlock::new(
            move |buffer: NonNull<AVAudioPCMBuffer>, _when: NonNull<AVAudioTime>| unsafe {
                tap_request.appendAudioPCMBuffer(buffer.as_ref());
            },
        );
        unsafe {
            input.installTapOnBus_bufferSize_format_block(
                0,
                1024,
                Some(&format),
                RcBlock::as_ptr(&tap),
            );
            engine.prepare();
        }
        if let Err(e) = unsafe { engine.startAndReturnError() } {
            unsafe { input.removeTapOnBus(0) };
            return Err(format!(
                "Could not start the microphone: {}",
                e.localizedDescription()
            ));
        }

        let (tx, waker) = (self.tx.clone(), self.waker.clone());
        let handler = RcBlock::new(
            move |result: *mut SFSpeechRecognitionResult, error: *mut NSError| {
                let msg = if let Some(result) = unsafe { result.as_ref() } {
                    let text = unsafe { result.bestTranscription().formattedString() }.to_string();
                    if unsafe { result.isFinal() } {
                        Msg::Done(generation, Ok(text), Vec::new())
                    } else {
                        Msg::Partial(generation, text)
                    }
                } else {
                    let why = unsafe { error.as_ref() }
                        .map(|e| e.localizedDescription().to_string())
                        .unwrap_or_else(|| "Speech recognition failed.".into());
                    Msg::Done(generation, Err(why), Vec::new())
                };
                let _ = tx.send(msg);
                waker();
            },
        );
        let task = unsafe { recognizer.recognitionTaskWithRequest_resultHandler(&request, &handler) };

        self.active = Some(Active {
            generation,
            legacy: Some(Legacy { engine, request, task }),
            last_partial: String::new(),
            stopping: None,
        });
        Ok(())
    }

    fn try_begin(&mut self) {
        match (self.speech_auth, self.mic_auth) {
            (Some(true), Some(true)) => {
                self.want_listen = false;
                if let Err(e) = self.begin() {
                    eprintln!("[voice] could not start listening: {e}");
                    self.out.push(VoiceEvent::Error(e));
                }
            }
            (Some(false), _) => {
                self.want_listen = false;
                self.out.push(VoiceEvent::Error("Speech recognition permission was denied. Enable it in System Settings › Privacy & Security › Speech Recognition.".into()));
            }
            (_, Some(false)) => {
                self.want_listen = false;
                self.out.push(VoiceEvent::Error("Microphone permission was denied. Enable it in System Settings › Privacy & Security › Microphone.".into()));
            }
            _ => self.request_permissions(),
        }
    }
}

impl Voice for MacVoice {
    fn speak(&mut self, id: u64, text: &str) {
        if self.kokoro.ready() {
            self.kokoro.speak(id, text);
        }
    }

    fn stop_speaking(&mut self) -> Option<(u64, String)> {
        self.kokoro.stop()
    }

    fn hold_speaking(&mut self, hold: bool) {
        if hold {
            self.kokoro.pause();
        } else {
            self.kokoro.resume();
        }
    }

    fn is_speaking(&self) -> bool {
        self.kokoro.is_speaking()
    }

    fn trim_recording(&mut self) {
        if self.active.as_ref().is_some_and(|a| a.legacy.is_none() && a.stopping.is_none()) {
            unsafe { twa_transcriber_trim() };
        }
    }

    fn set_conversing(&mut self, on: bool) {
        unsafe { twa_set_conversing(on, self.shim_context as *mut c_void, pause_callback) };
    }

    fn set_paused(&mut self, paused: bool) {
        unsafe { twa_set_paused(paused) };
    }

    fn voices(&self) -> Vec<(String, String)> {
        Kokoro::voices()
    }

    fn voice(&self) -> Option<String> {
        Some(self.kokoro.voice())
    }

    fn set_voice(&mut self, id: &str) {
        self.kokoro.set_voice(id);
    }

    fn transcribers(&self) -> Vec<(String, String)> {
        Whisper::models()
    }

    fn transcriber(&self) -> Option<String> {
        Some(self.whisper.model())
    }

    fn set_transcriber(&mut self, id: &str) {
        self.whisper.set_model(id);
    }

    fn models(&self) -> Vec<super::Model> {
        let (name, bytes, installed) = Kokoro::model();
        let whisper = Whisper::all().into_iter().map(|(id, name, bytes, installed)| super::Model {
            id: format!("whisper:{id}"),
            name,
            bytes,
            installed,
        });
        std::iter::once(super::Model { id: "kokoro".into(), name, bytes, installed }).chain(whisper).collect()
    }

    fn remove_model(&mut self, id: &str) {
        match id.strip_prefix("whisper:") {
            Some(model) => self.whisper.remove(model),
            None if id == "kokoro" => self.kokoro.remove(),
            None => {}
        }
    }

    fn download_model(&mut self, id: &str) {
        match id.strip_prefix("whisper:") {
            Some(model) => {
                self.whisper.set_model(model);
                self.whisper.allow_downloads();
            }
            None if id == "kokoro" => {
                if self.kokoro.voice() == "none" {
                    self.kokoro.set_voice(Kokoro::default_voice());
                }
                self.kokoro.allow_downloads();
            }
            None => {}
        }
    }

    fn missing_models(&self) -> Vec<(String, u64)> {
        self.kokoro.missing().into_iter().chain(self.whisper.missing()).collect()
    }

    fn allow_downloads(&mut self) {
        self.kokoro.allow_downloads();
        self.whisper.allow_downloads();
    }

    fn status(&self) -> Option<String> {
        Some(format!("{}  ·  {}", self.kokoro.describe(), self.whisper.describe()))
    }

    fn speaking_progress(&self) -> Option<(u64, usize)> {
        self.kokoro.progress()
    }

    fn start_listening(&mut self) {
        if self.active.is_some() || self.want_listen || self.refining.is_some() {
            return;
        }
        self.want_listen = true;
        self.try_begin();
    }

    fn stop_listening(&mut self) {
        if self.want_listen {
            self.want_listen = false;
            self.out.push(VoiceEvent::Finished(String::new()));
        }
        if let Some(active) = &mut self.active {
            active.stop_audio();
        }
    }

    fn poll(&mut self) -> Vec<VoiceEvent> {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::SpeechAuth(ok) => {
                    eprintln!("[voice] speech recognition permission: {ok}");
                    self.speech_auth = Some(ok);
                    if self.want_listen {
                        self.try_begin();
                    }
                }
                Msg::MicAuth(ok) => {
                    eprintln!("[voice] microphone permission: {ok}");
                    self.mic_auth = Some(ok);
                    if self.want_listen {
                        self.try_begin();
                    }
                }
                Msg::Partial(generation, text) => {
                    if let Some(active) = &mut self.active
                        && active.generation == generation
                    {
                        active.last_partial = text.clone();
                        self.out.push(VoiceEvent::Partial(text));
                    }
                }
                Msg::PauseRequested(pause) => self.out.push(VoiceEvent::PauseRequested(pause)),
                Msg::Refined(generation, text) => {
                    eprintln!("[voice] whisper #{generation}: {} chars", text.len());
                    if self.refining == Some(generation) {
                        self.refining = None;
                        self.out.push(VoiceEvent::Finished(text));
                    }
                }
                Msg::Done(generation, result, audio) => {
                    eprintln!(
                        "[voice] done #{generation}: {:?}, {:.1}s of audio",
                        result.as_ref().map(|t| t.len()),
                        audio.len() as f32 / super::whisper::SAMPLE_RATE as f32
                    );
                    if self.active.as_ref().is_some_and(|a| a.generation == generation) {
                        let mut active = self.active.take().unwrap();
                        let user_stopped = active.stopping.is_some();
                        active.stop_audio();
                        // If the system recognizer heard speech, let Whisper
                        // produce the final text from the recording.
                        // A result without a single word (just ".") is noise.
                        let result = result.map(|t| if t.chars().any(char::is_alphanumeric) { t } else { String::new() });
                        if let Ok(text) = &result
                            && !text.is_empty()
                            && !audio.is_empty()
                        {
                            let (tx, waker, fallback) = (self.tx.clone(), self.waker.clone(), text.clone());
                            if self.whisper.transcribe(audio, move |better| {
                                // Keep the live recognizer's text if Whisper's
                                // doesn't look like the same speech.
                                let text = better
                                    .filter(|t| !t.is_empty() && super::whisper::plausible(t, &fallback))
                                    .unwrap_or(fallback);
                                let _ = tx.send(Msg::Refined(generation, text));
                                waker();
                            }) {
                                self.refining = Some(generation);
                                continue;
                            }
                        }
                        self.out.push(match result {
                            Ok(text) => VoiceEvent::Finished(text),
                            // Typically "no speech detected" after stopping.
                            Err(_) if user_stopped || !active.last_partial.is_empty() => {
                                VoiceEvent::Finished(active.last_partial)
                            }
                            Err(e) => VoiceEvent::Error(e),
                        });
                    }
                }
            }
        }

        // Don't hang if the recognizer never delivers a final result.
        if self
            .active
            .as_ref()
            .and_then(|a| a.stopping)
            .is_some_and(|t| t.elapsed() > Duration::from_secs(4))
        {
            let active = self.active.take().unwrap();
            if let Some(l) = &active.legacy {
                unsafe { l.task.cancel() };
            }
            self.out.push(VoiceEvent::Finished(active.last_partial));
        }

        std::mem::take(&mut self.out)
    }
}
