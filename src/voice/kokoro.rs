//! Natural-sounding local text-to-speech with the Kokoro model (run by
//! sherpa-onnx), played through the default audio output.
//!
//! Platform-independent: a voice backend uses this once [`Kokoro::ready`] and
//! falls back to the system voice until then. The model is downloaded once.

use super::Waker;
use crate::state::data_dir;
use rodio::buffer::SamplesBuffer;
use rodio::{DeviceSinkBuilder, Player};
use sherpa_rs::tts::{KokoroTts, KokoroTtsConfig};
use sherpa_rs::OnnxConfig;
use std::collections::VecDeque;
use std::num::NonZero;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

const MODEL_NAME: &str = "Kokoro";
const MODEL_DIR: &str = "kokoro-multi-lang-v1_0";
const MODEL_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2";
const ARCHIVE_BYTES: u64 = 349_886_476;
/// The model's English voices; a voice's position here is its speaker id.
/// Names are <accent><gender>_<name>: a = American, b = British; f / m.
const VOICES: [&str; 28] = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
    "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
    "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma",
    "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
];
const DEFAULT_VOICE: usize = 3;

/// "af_heart" -> "Heart (American, female)".
fn label(voice: &str) -> String {
    let (kind, name) = voice.split_once('_').unwrap_or(("", voice));
    let accent = if kind.starts_with('b') { "British" } else { "American" };
    let gender = if kind.ends_with('m') { "male" } else { "female" };
    let mut name = name.to_string();
    if let Some(first) = name.get_mut(..1) {
        first.make_ascii_uppercase();
    }
    format!("{name} ({accent}, {gender})")
}

#[derive(Clone)]
enum Status {
    Downloading(u64),
    Loading,
    Ready,
    Failed(String),
}

/// A synthesized piece of the current utterance and when it plays.
struct Chunk {
    starts: Instant,
    lasts: Duration,
    /// Byte range of the utterance text this audio covers.
    text: std::ops::Range<usize>,
    samples: Vec<f32>,
    rate: NonZero<u32>,
}

struct State {
    status: Status,
    /// Utterances waiting to be spoken.
    queue: VecDeque<(u64, String)>,
    /// The utterance being synthesized / played.
    current: Option<(u64, String)>,
    chunks: Vec<Chunk>,
    player: Option<Arc<Player>>,
    /// Index into [`VOICES`]; applies from the next sentence synthesized.
    voice: usize,
    /// Bumped by `stop`, so the worker abandons what it was doing.
    epoch: u64,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

pub struct Kokoro {
    shared: Arc<Shared>,
}

impl Kokoro {
    /// Starts a worker thread that fetches and loads the model, then speaks.
    pub fn start(waker: Waker) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                status: Status::Loading,
                queue: VecDeque::new(),
                current: None,
                chunks: Vec::new(),
                player: None,
                voice: DEFAULT_VOICE,
                epoch: 0,
            }),
            wake: Condvar::new(),
        });
        let worker = shared.clone();
        std::thread::spawn(move || run(&worker, &waker));
        Kokoro { shared }
    }

    pub fn voices() -> Vec<(String, String)> {
        VOICES.iter().map(|v| (v.to_string(), label(v))).collect()
    }

    pub fn voice(&self) -> String {
        VOICES[self.shared.lock().voice].to_string()
    }

    pub fn set_voice(&self, id: &str) {
        if let Some(index) = VOICES.iter().position(|v| *v == id) {
            self.shared.lock().voice = index;
        }
    }

    pub fn ready(&self) -> bool {
        matches!(self.shared.lock().status, Status::Ready)
    }

    /// Queue text to be spoken after anything already queued.
    pub fn speak(&self, id: u64, text: &str) {
        self.shared.lock().queue.push_back((id, text.to_string()));
        self.shared.wake.notify_all();
    }

    pub fn is_speaking(&self) -> bool {
        let state = self.shared.lock();
        state.current.is_some() || !state.queue.is_empty()
    }

    /// The utterance being spoken and how many bytes of it have been played.
    pub fn progress(&self) -> Option<(u64, usize)> {
        let state = self.shared.lock();
        let (id, text) = state.current.as_ref()?;
        Some((*id, heard_bytes(&state.chunks, Instant::now(), text)))
    }

    /// Stop now and drop everything queued. Returns the id of the utterance
    /// that was cut off and (an estimate of) the part of it already heard.
    pub fn stop(&self) -> Option<(u64, String)> {
        let mut state = self.shared.lock();
        state.epoch += 1;
        if let Some(player) = state.player.take() {
            player.stop();
        }
        let cut = match state.current.take() {
            Some((id, text)) => {
                let heard = heard_bytes(&state.chunks, Instant::now(), &text);
                Some((id, text[..heard].to_string()))
            }
            None => state.queue.front().map(|(id, _)| (*id, String::new())),
        };
        state.queue.clear();
        state.chunks.clear();
        drop(state);
        self.shared.wake.notify_all();
        cut
    }

    /// One line for the UI.
    pub fn describe(&self) -> String {
        match &self.shared.lock().status {
            Status::Downloading(bytes) => format!(
                "Downloading {MODEL_NAME} voice ({}%)… using the system voice meanwhile",
                (bytes * 100 / ARCHIVE_BYTES).min(99)
            ),
            Status::Loading => format!("Loading {MODEL_NAME} voice…"),
            Status::Ready => format!("Voice: {MODEL_NAME}, on this device"),
            Status::Failed(e) => format!("{MODEL_NAME} voice unavailable ({e}); using the system voice"),
        }
    }
}

/// How much of `text` has been played at `now`, snapped to the end of a word.
fn heard_bytes(chunks: &[Chunk], now: Instant, text: &str) -> usize {
    let mut heard = 0;
    for chunk in chunks {
        if now >= chunk.starts + chunk.lasts {
            heard = chunk.text.end;
        } else if now > chunk.starts {
            let fraction = (now - chunk.starts).as_secs_f64() / chunk.lasts.as_secs_f64().max(0.001);
            heard = chunk.text.start + (chunk.text.len() as f64 * fraction) as usize;
        }
    }
    let mut heard = heard.min(text.len());
    while !text.is_char_boundary(heard) {
        heard += 1;
    }
    if heard == 0 {
        return 0; // nothing played yet
    }
    match text[heard..].find(char::is_whitespace) {
        Some(space) => heard + space,
        None => text.len(),
    }
}

/// Byte ranges of the sentences in `text`, so speech can start after the
/// first sentence is synthesized instead of after the whole message.
fn sentences(text: &str) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let end = i + c.len_utf8();
        let boundary = c == '\n'
            || (matches!(c, '.' | '!' | '?' | ':' | ';')
                && chars.peek().is_none_or(|(_, next)| next.is_whitespace()));
        if boundary && !text[start..end].trim().is_empty() {
            ranges.push(start..end);
            start = end;
        }
    }
    if !text[start..].trim().is_empty() {
        ranges.push(start..text.len());
    }
    ranges
}

fn run(shared: &Shared, waker: &Waker) {
    let set = |status: Status| {
        shared.lock().status = status;
        waker();
    };
    let dir = data_dir().join("models").join(MODEL_DIR);
    if !dir.join("model.onnx").exists()
        && let Err(e) = download(&dir, &|bytes| set(Status::Downloading(bytes)))
    {
        return set(Status::Failed(format!("download failed: {e}")));
    }
    set(Status::Loading);
    let file = |name: &str| dir.join(name).to_string_lossy().into_owned();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8)) as i32;
    let mut tts = KokoroTts::new(KokoroTtsConfig {
        model: file("model.onnx"),
        voices: file("voices.bin"),
        tokens: file("tokens.txt"),
        data_dir: file("espeak-ng-data"),
        dict_dir: file("dict"),
        lexicon: format!("{},{}", file("lexicon-us-en.txt"), file("lexicon-zh.txt")),
        length_scale: 1.0,
        onnx_config: OnnxConfig {
            provider: "cpu".into(),
            debug: false,
            num_threads: threads,
        },
        ..Default::default()
    });
    // The first synthesis is slow; get it out of the way.
    if tts.create("Ready.", DEFAULT_VOICE as i32, 1.0).is_err() {
        return set(Status::Failed("could not load the model".into()));
    }
    set(Status::Ready);

    loop {
        // Wait for the next utterance.
        let (id, text, epoch) = {
            let mut state = shared.lock();
            loop {
                if let Some((id, text)) = state.queue.pop_front() {
                    state.current = Some((id, text.clone()));
                    state.chunks.clear();
                    break (id, text, state.epoch);
                }
                state = shared.wake.wait(state).unwrap_or_else(|e| e.into_inner());
            }
        };
        let _ = id;
        waker();

        // A fresh output per utterance follows changes of the default device
        // (e.g. headphones connecting).
        let mut output = open_output();
        let mut player = output.as_ref().map(|o| Arc::new(Player::connect_new(o.mixer())));
        if let Some(player) = &player {
            let mut state = shared.lock();
            if state.epoch == epoch {
                state.player = Some(player.clone());
            }
        }
        // An output opened while the audio hardware is switching modes (a
        // headset changing profile when the microphone comes on) can be dead:
        // it accepts audio but never plays it. Watch that playback really
        // advances; if not, replace the output and start the utterance again.
        let mut reopens = 0;
        let mut reopen_if_needed = |output: &mut Option<rodio::MixerDeviceSink>,
                                    player: &mut Option<Arc<Player>>| {
            let state = shared.lock();
            let (Some(first), Some(p)) = (state.chunks.first(), player.as_ref()) else {
                return;
            };
            let total: Duration = state.chunks.iter().map(|c| c.lasts).sum();
            let expected = first.starts.elapsed().min(total);
            // Chunks that finished playing, plus the position in the current one.
            let finished = state.chunks.len().saturating_sub(p.len());
            let played = state.chunks[..finished].iter().map(|c| c.lasts).sum::<Duration>() + p.get_pos();
            let stalled = (expected > Duration::from_millis(700) && played < expected / 3)
                || expected > played + Duration::from_millis(1500);
            if !stalled || reopens >= 20 {
                return;
            }
            drop(state);
            reopens += 1;
            eprintln!("[voice] reopening audio output (expected {expected:?}, played {played:?})");
            *player = None;
            *output = None;
            std::thread::sleep(Duration::from_millis(300)); // let the hardware settle
            *output = open_output();
            *player = output.as_ref().map(|o| Arc::new(Player::connect_new(o.mixer())));
            let mut state = shared.lock();
            if state.epoch != epoch {
                return;
            }
            state.player = player.clone();
            let mut starts = Instant::now();
            for chunk in &mut state.chunks {
                chunk.starts = starts;
                starts += chunk.lasts;
                if let Some(player) = player {
                    player.append(SamplesBuffer::new(NonZero::<u16>::MIN, chunk.rate, chunk.samples.clone()));
                }
            }
        };

        for range in sentences(&text) {
            if shared.lock().epoch != epoch {
                break;
            }
            reopen_if_needed(&mut output, &mut player);
            let voice = shared.lock().voice as i32;
            let Ok(audio) = tts.create(&respell(text[range.clone()].trim(), &pronunciations()), voice, 1.0) else {
                continue;
            };
            let (Some(player), Some(rate)) = (&player, NonZero::new(audio.sample_rate)) else {
                break;
            };
            let lasts = Duration::from_secs_f64(audio.samples.len() as f64 / audio.sample_rate as f64);
            let mut state = shared.lock();
            if state.epoch != epoch {
                break;
            }
            let now = Instant::now();
            let starts = state.chunks.last().map_or(now, |c| (c.starts + c.lasts).max(now));
            player.append(SamplesBuffer::new(NonZero::<u16>::MIN, rate, audio.samples.clone()));
            state.chunks.push(Chunk { starts, lasts, text: range, samples: audio.samples, rate });
        }

        // Let it play out (or be stopped).
        loop {
            reopen_if_needed(&mut output, &mut player);
            let state = shared.lock();
            let done = state.chunks.last().is_none_or(|c| Instant::now() >= c.starts + c.lasts);
            if state.epoch != epoch || done {
                break;
            }
            drop(state);
            std::thread::sleep(Duration::from_millis(30));
        }
        let mut state = shared.lock();
        if state.epoch == epoch {
            state.current = None;
            state.chunks.clear();
            state.player = None;
        }
        drop(state);
        drop(player);
        drop(output);
        waker();
    }
}

/// Open the default audio output. This can fail for a moment while the
/// hardware is switching modes (e.g. the microphone just came on), so retry.
fn open_output() -> Option<rodio::MixerDeviceSink> {
    let started = Instant::now();
    loop {
        match DeviceSinkBuilder::open_default_sink() {
            Ok(mut output) => {
                output.log_on_drop(false);
                if started.elapsed() > Duration::from_millis(100) {
                    eprintln!("[voice] audio output opened after {:?}", started.elapsed());
                }
                return Some(output);
            }
            Err(e) if started.elapsed() > Duration::from_secs(30) => {
                eprintln!("[voice] could not open audio output: {e}");
                return None;
            }
            Err(_) => std::thread::sleep(Duration::from_millis(250)),
        }
    }
}

fn download(dir: &Path, progress: &dyn Fn(u64)) -> Result<(), String> {
    let models = dir.parent().unwrap();
    std::fs::create_dir_all(models).map_err(|e| e.to_string())?;
    let archive = models.join(format!("{MODEL_DIR}.tar.bz2.part"));
    let mut child = Command::new("curl")
        .args(["-fsSL", "--retry", "3", "-C", "-", "-o"])
        .arg(&archive)
        .arg(MODEL_URL)
        .spawn()
        .map_err(|e| format!("could not run curl: {e}"))?;
    loop {
        progress(std::fs::metadata(&archive).map(|m| m.len()).unwrap_or(0));
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) if status.success() => break,
            Some(status) => return Err(format!("curl exited with {status}")),
            None => std::thread::sleep(Duration::from_millis(500)),
        }
    }
    let status = Command::new("tar")
        .arg("-xjf")
        .arg(&archive)
        .arg("-C")
        .arg(models)
        .status()
        .map_err(|e| format!("could not run tar: {e}"))?;
    if !status.success() {
        return Err(format!("tar exited with {status}"));
    }
    let _ = std::fs::remove_file(&archive);
    Ok(())
}

/// The user's pronunciation fixes (`pronunciations.txt` in the data folder,
/// created with a starter list): lowercase word -> how to say it.
fn pronunciations() -> Vec<(String, String)> {
    let path = data_dir().join("pronunciations.txt");
    if !path.exists() {
        let _ = std::fs::write(&path, include_str!("pronunciations.txt"));
    }
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    text.lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .filter_map(|line| line.split_once('='))
        .map(|(word, say)| (word.trim().to_lowercase(), say.trim().to_string()))
        .filter(|(word, say)| !word.is_empty() && !say.is_empty())
        .collect()
}

/// Replace whole words of `text` that have a pronunciation fix, keeping a
/// plural or possessive ending.
fn respell(text: &str, fixes: &[(String, String)]) -> String {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    let lookup = |word: &str| -> Option<String> {
        let lower = word.to_lowercase();
        let find = |w: &str| fixes.iter().find(|(from, _)| from == w).map(|(_, say)| say.clone());
        find(&lower)
            .or_else(|| lower.strip_suffix('s').and_then(find).map(|say| say + "s"))
    };
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while !rest.is_empty() {
        let end = rest.find(|c| !is_word(c)).unwrap_or(rest.len());
        if end == 0 {
            let c = rest.chars().next().unwrap();
            out.push(c);
            rest = &rest[c.len_utf8()..];
        } else {
            out.push_str(&lookup(&rest[..end]).unwrap_or_else(|| rest[..end].to_string()));
            rest = &rest[end..];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::respell;

    #[test]
    fn respells_whole_words() {
        let fixes = vec![("callee".to_string(), "call E".to_string()), ("sp".to_string(), "S P".to_string())];
        assert_eq!(respell("The Callee's sp, callees; spin.", &fixes), "The call E's S P, call Es; spin.");
    }
}
