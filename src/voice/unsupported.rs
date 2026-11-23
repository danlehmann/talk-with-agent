//! Fallback for platforms without a voice backend yet: the app still works
//! with typed input and on-screen replies.

use super::{Voice, VoiceEvent, Waker};

pub struct NoVoice {
    events: Vec<VoiceEvent>,
}

impl NoVoice {
    pub fn new(_waker: Waker) -> Self {
        NoVoice { events: Vec::new() }
    }
}

impl Voice for NoVoice {
    fn speak(&mut self, _id: u64, _text: &str) {}
    fn stop_speaking(&mut self) -> Option<(u64, String)> {
        None
    }
    fn is_speaking(&self) -> bool {
        false
    }
    fn start_listening(&mut self) {
        self.events.push(VoiceEvent::Error(
            "Voice is not implemented on this platform yet; type instead.".into(),
        ));
    }
    fn stop_listening(&mut self) {}
    fn poll(&mut self) -> Vec<VoiceEvent> {
        std::mem::take(&mut self.events)
    }
}
