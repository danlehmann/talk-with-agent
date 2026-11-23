# Talk with Agent

A desktop app for talking to coding agents, such as Claude Code, by voice. It runs a local HTTP
server; an agent connects with `curl`, and from then on what the agent says is
read aloud and what you say is transcribed and sent back. Speech recognition
and the voice both run on your machine.

## Run

    scripts/bundle-macos.sh
    open "target/Talk with Agent.app"

(`cargo run` also works; macOS then attributes the microphone permission to your terminal.)

Building needs Rust, `cmake`, and the Xcode command line tools (for `swiftc`).
The build downloads prebuilt sherpa-onnx libraries. On first launch the app
downloads two models (about 3.4 GB together) into
`~/Library/Application Support/TalkWithAgent/models`; until they are ready
it uses the system recognizer and voice.

Click **Copy prompt** and paste it into an agent session (for example Claude Code). The agent fetches
its instructions from the URL (`http://<lan-ip>:7717/<access-key>`) and starts a session.
The server is reachable from your local network; the access key in the URL is what
protects it, so treat the URL as a secret. Traffic is plain HTTP.

- **Hands-free** (on by default) keeps the microphone open and sends what you
  said when you pause. Start talking while the agent is speaking to interrupt
  it; the agent is told how much you heard.
- With hands-free off, **Talk / Space** starts listening and **Send / Space** sends.
  **Interrupt** (or Space) cuts the agent off.
- The text being read aloud is highlighted as the voice progresses.
- Sessions and history are stored in
  `~/Library/Application Support/TalkWithAgent/sessions.json`; the ✕ next to a
  session deletes it.

## Layout

| File | Purpose |
| --- | --- |
| `src/state.rs` | Sessions, history, persistence |
| `src/server.rs` | HTTP API for agents (`GET /<key>` returns the agent instructions) |
| `src/app.rs` | UI (egui) and conversation logic, platform-independent |
| `src/voice/mod.rs` | `Voice` trait: the only platform boundary |
| `src/voice/whisper.rs` | Final transcription with Whisper large-v3 (platform-independent) |
| `src/voice/kokoro.rs` | Kokoro neural voice and playback (platform-independent) |
| `src/voice/pronunciations.txt` | Starter list of pronunciation fixes, copied to the data folder for editing |
| `src/voice/macos.rs` | macOS backend: microphone, live transcript, system-voice fallback |
| `macos/Transcriber.swift` | Swift shim for SpeechAnalyzer and echo-cancelled microphone capture |
| `src/voice/unsupported.rs` | Fallback for other platforms: typing only |

To port to another OS, implement `Voice` in a new `src/voice/<os>.rs` and select
it in `voice::create`. The backend needs to capture the microphone (ideally with
echo cancellation), give a live transcript for the preview and for detecting
pauses, and hand the recorded utterance to `whisper.rs`; `kokoro.rs` can be
used as is for speech.

## Disclaimer

This is an independent project. It is not affiliated with, endorsed by, or
sponsored by Anthropic. "Claude" and "Claude Code" are trademarks of Anthropic,
used here only to describe what the app works with.
