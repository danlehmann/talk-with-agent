// Microphone capture and live speech-to-text with Apple's on-device
// SpeechAnalyzer (macOS 26+).
//
// SpeechAnalyzer is a Swift-only API, so this small shim exposes it to the
// Rust macOS backend (src/voice/macos.rs) through a few C functions.
// build.rs compiles this file with swiftc and links it in.

import AVFAudio
import CoreMedia
import Foundation
import MediaPlayer
import Speech

/// kind: 0 = partial transcript, 1 = finished (final transcript), 2 = error.
/// "Finished" also carries the whole utterance as 16 kHz mono float samples,
/// so a better model can re-transcribe it. May be called from any thread; the
/// pointers are only valid during the call.
public typealias TWACallback = @convention(c) (
    UnsafeMutableRawPointer?, Int32, UInt64, UnsafePointer<CChar>?, UnsafePointer<Float>?, Int
) -> Void

private let recordingRate = 16_000.0
private let maxRecordedSamples = Int(recordingRate) * 600

private func log(_ message: String) {
    FileHandle.standardError.write(Data("[voice] \(message)\n".utf8))
}

private func fail(_ message: String) -> NSError {
    NSError(domain: "TWASpeech", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
}

/// The microphone, with echo cancellation. It keeps running between listening
/// sessions: stopping and restarting it loses words, and makes Bluetooth
/// headsets switch modes back and forth. Audio that arrives while no session
/// is attached is kept briefly and handed to the next one.
private final class Microphone: @unchecked Sendable {
    static let shared = Microphone()

    /// How long the microphone stays on after the last session, in seconds.
    private let linger = 10.0
    /// How much audio from between two sessions is carried over, in seconds.
    private let carryOver = 4.0

    private let engine = AVAudioEngine()
    // Recursive: starting the engine can report a configuration change on
    // the same thread.
    private let lock = NSRecursiveLock()
    private var running = false
    private var starting = false
    private var configured = false
    private var consumer: ((AVAudioPCMBuffer) -> Void)?
    private var backlog: [AVAudioPCMBuffer] = []
    private var detachedAt = Date.distantPast
    private var uses = 0

    /// Start delivering microphone audio to `consumer` (on an audio thread),
    /// beginning with anything captured since the previous session ended.
    func attach(_ consumer: @escaping (AVAudioPCMBuffer) -> Void) throws {
        lock.lock()
        defer { lock.unlock() }
        uses += 1
        if !running {
            backlog.removeAll()
            try start()
        }
        if Date().timeIntervalSince(detachedAt) < carryOver {
            backlog.forEach(consumer)
        }
        backlog.removeAll()
        self.consumer = consumer
    }

    /// Stop delivering audio. The microphone itself stops a little later,
    /// unless another session attaches first.
    func detach() {
        lock.lock()
        consumer = nil
        detachedAt = Date()
        let use = uses
        lock.unlock()
        DispatchQueue.global().asyncAfter(deadline: .now() + linger) { [self] in
            lock.lock()
            defer { lock.unlock() }
            if uses == use, consumer == nil, running {
                engine.stop()
                engine.inputNode.removeTap(onBus: 0)
                running = false
                backlog.removeAll()
            }
        }
    }

    /// Call with the lock held.
    private func start() throws {
        let input = engine.inputNode
        if !configured {
            configured = true
            // Echo cancellation: the microphone stays open while the agent's
            // voice plays, so that voice must not be transcribed back.
            do {
                try input.setVoiceProcessingEnabled(true)
                // Voice processing lowers other audio by default, including
                // the agent's voice. Keep that to the minimum.
                input.voiceProcessingOtherAudioDuckingConfiguration = .init(
                    enableAdvancedDucking: false, duckingLevel: .min)
            } catch {
                log("no echo cancellation: \(error.localizedDescription)")
            }
            // The audio hardware reconfigures itself when playback starts or
            // stops, or a headset connects; the engine stops when that
            // happens. Pick the microphone up again in its new format.
            NotificationCenter.default.addObserver(
                forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil
            ) { [self] _ in
                lock.lock()
                defer { lock.unlock() }
                guard running, !starting else { return }
                engine.inputNode.removeTap(onBus: 0)
                do {
                    try start()
                } catch {
                    running = false
                    log("microphone lost: \(error.localizedDescription)")
                }
            }
        }

        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw fail("No microphone available.")
        }
        log("microphone on: \(format.sampleRate) Hz, \(format.channelCount) ch, echo cancellation \(input.isVoiceProcessingEnabled)")
        let backlogLimit = Int(format.sampleRate * carryOver)
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [self] buffer, _ in
            lock.lock()
            defer { lock.unlock() }
            if let consumer {
                consumer(buffer)
            } else if let copy = Microphone.copy(buffer) {
                backlog.append(copy)
                var frames = backlog.reduce(0) { $0 + Int($1.frameLength) }
                while frames > backlogLimit, let first = backlog.first {
                    frames -= Int(first.frameLength)
                    backlog.removeFirst()
                }
            }
        }
        starting = true
        defer { starting = false }
        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            throw fail("Could not start the microphone: \(error.localizedDescription)")
        }
        running = true
    }

    /// The engine reuses its tap buffers, so audio that is kept needs a copy.
    private static func copy(_ buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        guard let copy = AVAudioPCMBuffer(pcmFormat: buffer.format, frameCapacity: buffer.frameLength)
        else { return nil }
        copy.frameLength = buffer.frameLength
        let source = UnsafeMutableAudioBufferListPointer(buffer.mutableAudioBufferList)
        let target = UnsafeMutableAudioBufferListPointer(copy.mutableAudioBufferList)
        for (from, to) in zip(source, target) {
            if let fromData = from.mData, let toData = to.mData {
                memcpy(toData, fromData, Int(min(from.mDataByteSize, to.mDataByteSize)))
            }
        }
        return copy
    }
}

/// One stretch of listening: from opening up to the final transcript.
@available(macOS 26.0, *)
private final class Session: @unchecked Sendable {
    let generation: UInt64
    let context: UnsafeMutableRawPointer?
    let callback: TWACallback

    private let lock = NSLock()
    private var stopRequested = false
    private var attached = false
    private var continuation: AsyncStream<AnalyzerInput>.Continuation?
    private var analyzer: SpeechAnalyzer?
    private var recorded: [Float] = []
    /// The part of `recorded` in which the recognizer heard speech. Long
    /// silence around it makes Whisper invent words ("Thank you.").
    private var speechStart: Int?
    private var speechEnd = 0

    // Used only on the microphone's delivery path.
    private var inputFormat: AVAudioFormat?
    private var toAnalyzer: AVAudioConverter?
    private var toRecording: AVAudioConverter?
    private let recordingFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: recordingRate, channels: 1, interleaved: false)!

    init(generation: UInt64, context: UnsafeMutableRawPointer?, callback: TWACallback) {
        self.generation = generation
        self.context = context
        self.callback = callback
    }

    private func emit(_ kind: Int32, _ text: String, samples: [Float] = []) {
        text.withCString { cString in
            samples.withUnsafeBufferPointer { buffer in
                callback(context, kind, generation, cString, buffer.baseAddress, buffer.count)
            }
        }
    }

    /// Converts one microphone buffer to `format`; nil if nothing came out.
    private static func convert(
        _ buffer: AVAudioPCMBuffer, with converter: AVAudioConverter, to format: AVAudioFormat
    ) -> AVAudioPCMBuffer? {
        let ratio = format.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
        guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { return nil }
        var consumed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if consumed {
                status.pointee = .noDataNow
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return buffer
        }
        return error == nil && out.frameLength > 0 ? out : nil
    }

    /// Microphone audio arrives here: feed the analyzer in its format, and
    /// keep a 16 kHz copy of the utterance.
    private func consume(
        _ buffer: AVAudioPCMBuffer, analyzerFormat: AVAudioFormat,
        continuation: AsyncStream<AnalyzerInput>.Continuation
    ) {
        if inputFormat != buffer.format {
            // First buffer, or the hardware changed format.
            inputFormat = buffer.format
            toAnalyzer = AVAudioConverter(from: buffer.format, to: analyzerFormat)
            toRecording = AVAudioConverter(from: buffer.format, to: recordingFormat)
            if buffer.format.channelCount > 1 {
                // Voice processing can expose several channels; the first is the voice.
                toAnalyzer?.channelMap = [0]
                toRecording?.channelMap = [0]
            }
        }
        if let toAnalyzer, let out = Session.convert(buffer, with: toAnalyzer, to: analyzerFormat) {
            continuation.yield(AnalyzerInput(buffer: out))
        }
        if let toRecording, let out = Session.convert(buffer, with: toRecording, to: recordingFormat),
           let channel = out.floatChannelData?[0] {
            let samples = UnsafeBufferPointer(start: channel, count: Int(out.frameLength))
            lock.withLock {
                if recorded.count < maxRecordedSamples {
                    recorded.append(contentsOf: samples)
                }
            }
        }
    }

    func run() async {
        do {
            guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale.current)
            else { throw fail("Speech recognition does not support your system language.") }
            let transcriber = SpeechTranscriber(locale: locale, preset: .progressiveTranscription)

            // First use: fetch the on-device model for this language.
            if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                try await request.downloadAndInstall()
            }

            let analyzer = SpeechAnalyzer(modules: [transcriber])
            guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])
            else { throw fail("No audio format available for speech recognition.") }

            let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
            try Microphone.shared.attach { [self] buffer in
                consume(buffer, analyzerFormat: format, continuation: continuation)
            }
            let stopAlready = lock.withLock {
                self.continuation = continuation
                self.analyzer = analyzer
                attached = true
                return stopRequested
            }

            try await analyzer.start(inputSequence: stream)
            if stopAlready { finish() }

            // Finalized text only grows; the volatile tail is the model's
            // current guess for the newest audio and gets replaced.
            var finalized = ""
            var volatile = ""
            var previous = ""
            for try await result in transcriber.results {
                let text = String(result.text.characters)
                if result.isFinal {
                    finalized += text
                    volatile = ""
                } else {
                    volatile = text
                }
                let transcript = finalized + volatile
                if transcript != previous, !transcript.trimmingCharacters(in: .whitespaces).isEmpty {
                    // New words: the recognizer runs a little behind the audio.
                    lock.withLock {
                        if speechStart == nil {
                            speechStart = max(0, recorded.count - Int(recordingRate * 3))
                        }
                        speechEnd = recorded.count
                    }
                }
                previous = transcript
                emit(0, transcript)
            }
            detach()
            let samples: [Float] = lock.withLock {
                guard let start = speechStart else { return [] }
                let end = min(recorded.count, speechEnd + Int(recordingRate * 0.4))
                return start < end ? Array(recorded[start..<end]) : []
            }
            emit(
                1, (finalized + volatile).trimmingCharacters(in: .whitespacesAndNewlines),
                samples: samples)
        } catch {
            detach()
            emit(2, error.localizedDescription)
        }
    }

    private func detach() {
        let wasAttached = lock.withLock {
            let was = attached
            attached = false
            return was
        }
        if wasAttached {
            Microphone.shared.detach()
        }
    }

    /// Stop taking audio and let the analyzer finish what it has heard.
    private func finish() {
        detach()
        let (continuation, analyzer) = lock.withLock { (self.continuation, self.analyzer) }
        continuation?.finish()
        if let analyzer {
            Task { try? await analyzer.finalizeAndFinishThroughEndOfInput() }
        }
    }

    /// Drop the recording so far, except the last 0.7 seconds.
    func trim() {
        lock.withLock {
            let keep = Int(recordingRate * 0.7)
            if recorded.count > keep {
                recorded.removeFirst(recorded.count - keep)
            }
            speechStart = nil
            speechEnd = 0
        }
    }

    func stop() {
        lock.withLock { stopRequested = true }
        finish()
    }
}

nonisolated(unsafe) private var awakeActivity: NSObjectProtocol?
nonisolated(unsafe) private var pauseHandler: ((Int32) -> Void)?
nonisolated(unsafe) private var mediaCommandsInstalled = false

/// A hands-free conversation started (or ended). While one is going on:
/// - the Mac and its display don't go to sleep when idle;
/// - the app is the system's "now playing" app, so play/pause on a headset
///   (the AirPods stem), keyboard or Control Center reaches it;
/// - the headset's mute gesture, which replaces play/pause while the
///   microphone is in use, reaches it too.
/// `pauseRequested` gets 0 = toggle, 1 = pause, 2 = resume. Main thread only.
@_cdecl("twa_set_conversing")
public func twaSetConversing(
    _ on: Bool, context: UnsafeMutableRawPointer?,
    pauseRequested: @convention(c) (UnsafeMutableRawPointer?, Int32) -> Void
) {
    if on, awakeActivity == nil {
        awakeActivity = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .idleDisplaySleepDisabled],
            reason: "Voice conversation with a coding agent")
    } else if !on, let activity = awakeActivity {
        ProcessInfo.processInfo.endActivity(activity)
        awakeActivity = nil
    }

    pauseHandler = on ? { request in pauseRequested(context, request) } : nil
    let commands = MPRemoteCommandCenter.shared()
    if !mediaCommandsInstalled {
        mediaCommandsInstalled = true
        commands.togglePlayPauseCommand.addTarget { _ in pauseHandler?(0); return .success }
        commands.pauseCommand.addTarget { _ in pauseHandler?(1); return .success }
        commands.playCommand.addTarget { _ in pauseHandler?(2); return .success }
        do {
            try AVAudioApplication.shared.setInputMuteStateChangeHandler { muted in
                DispatchQueue.main.async { pauseHandler?(muted ? 1 : 2) }
                return true
            }
        } catch {
            log("no headset mute gesture: \(error.localizedDescription)")
        }
    }
    for command in [commands.togglePlayPauseCommand, commands.pauseCommand, commands.playCommand] {
        command.isEnabled = on
    }
    let nowPlaying = MPNowPlayingInfoCenter.default()
    nowPlaying.nowPlayingInfo = on
        ? [MPMediaItemPropertyTitle: "Voice conversation", MPMediaItemPropertyArtist: "Talk with Agent"]
        : nil
    nowPlaying.playbackState = on ? .playing : .stopped
}

/// Show the conversation as paused or playing in the system media controls.
@_cdecl("twa_set_paused")
public func twaSetPaused(_ paused: Bool) {
    if pauseHandler != nil {
        MPNowPlayingInfoCenter.default().playbackState = paused ? .paused : .playing
    }
}

private let sessionLock = NSLock()
nonisolated(unsafe) private var currentSession: AnyObject?

@_cdecl("twa_transcriber_available")
public func twaTranscriberAvailable() -> Bool {
    if #available(macOS 26.0, *) {
        return SpeechTranscriber.isAvailable
    }
    return false
}

/// Start listening. Results arrive through `callback`, tagged with `generation`:
/// any number of partials, then exactly one finished or error.
@_cdecl("twa_transcriber_start")
public func twaTranscriberStart(
    generation: UInt64, context: UnsafeMutableRawPointer?, callback: TWACallback
) {
    guard #available(macOS 26.0, *) else { return }
    let session = Session(generation: generation, context: context, callback: callback)
    sessionLock.lock()
    currentSession = session
    sessionLock.unlock()
    Task.detached { await session.run() }
}

/// Forget the audio recorded so far (except the last moment).
@_cdecl("twa_transcriber_trim")
public func twaTranscriberTrim() {
    guard #available(macOS 26.0, *) else { return }
    sessionLock.lock()
    let session = currentSession as? Session
    sessionLock.unlock()
    session?.trim()
}

/// Stop listening; the finished callback follows with the final text.
@_cdecl("twa_transcriber_stop")
public func twaTranscriberStop() {
    guard #available(macOS 26.0, *) else { return }
    sessionLock.lock()
    let session = currentSession as? Session
    sessionLock.unlock()
    session?.stop()
}
