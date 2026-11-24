//! Sessions, message history and persistence. Shared between the HTTP server
//! threads and the UI.

use serde::{Deserialize, Serialize};
use std::hash::{BuildHasher, Hasher};
use std::path::PathBuf;
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Agent,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub text: String,
    /// Unix seconds.
    pub time: u64,
    /// User message: not yet delivered to the agent.
    /// Agent message: not yet spoken to the user.
    #[serde(default)]
    pub pending: bool,
    /// Agent message the user interrupted: the part that was spoken.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heard: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub name: String,
    pub created: u64,
    pub messages: Vec<Message>,
    /// The agent said goodbye (DELETE). History is kept until the user deletes it.
    #[serde(default)]
    pub ended: bool,
    /// Note for the agent about an interruption, delivered with the next user message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interruption: Option<String>,
    /// Number of agent requests currently blocked waiting for the user to speak.
    #[serde(skip)]
    pub listeners: usize,
    /// Last time the agent made any request, this run of the app.
    #[serde(skip)]
    pub last_seen: Option<Instant>,
}

impl Session {
    pub fn unheard(&self) -> usize {
        self.messages
            .iter()
            .filter(|m| m.role == Role::Agent && m.pending)
            .count()
    }
}

#[derive(Default, Serialize, Deserialize)]
pub struct Inner {
    pub sessions: Vec<Session>,
}

impl Inner {
    pub fn session(&mut self, id: &str) -> Option<&mut Session> {
        self.sessions.iter_mut().find(|s| s.id == id)
    }
}

pub struct NotFound;

/// What an agent receives when it listens.
pub struct Delivery {
    /// What the user said since the agent's latest message.
    pub text: String,
    /// What the user said before the agent's latest message was posted, but
    /// which the agent has not seen yet.
    pub earlier: String,
    /// Indexes of the delivered messages (for `undeliver`).
    pub indexes: Vec<usize>,
    /// Note about the user interrupting the agent's speech.
    pub interruption: Option<String>,
}

pub struct Store {
    inner: Mutex<Inner>,
    changed: Condvar,
    path: PathBuf,
    waker: OnceLock<Box<dyn Fn() + Send + Sync>>,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn random_id() -> String {
    let r = || std::collections::hash_map::RandomState::new().build_hasher().finish();
    format!("{:016x}{:016x}", r(), r())
}

/// Points the data folder at a scratch location, for tests.
#[cfg(test)]
pub fn use_test_data_dir() -> PathBuf {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("twa-test-{}", std::process::id()));
        unsafe { std::env::set_var("TALK_WITH_AGENT_DATA", &dir) };
        dir
    })
    .clone()
}

pub fn data_dir() -> PathBuf {
    // TALK_WITH_AGENT_DATA overrides the location (for trying things out
    // without touching the real sessions and models).
    let dir = std::env::var_os("TALK_WITH_AGENT_DATA").map(PathBuf::from).unwrap_or_else(|| {
        dirs::data_dir().unwrap_or_else(|| PathBuf::from(".")).join("TalkWithAgent")
    });
    let _ = std::fs::create_dir_all(&dir);
    dir
}

impl Store {
    pub fn load() -> Self {
        let path = data_dir().join("sessions.json");
        let inner = std::fs::read(&path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        Store {
            inner: Mutex::new(inner),
            changed: Condvar::new(),
            path,
            waker: OnceLock::new(),
        }
    }

    /// Called whenever state changes from a server thread, to repaint the UI.
    pub fn set_waker(&self, f: impl Fn() + Send + Sync + 'static) {
        let _ = self.waker.set(Box::new(f));
    }

    pub fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Persist and notify everyone (UI + long-polling agents). Call after any mutation.
    pub fn commit(&self, inner: MutexGuard<'_, Inner>) {
        if let Ok(json) = serde_json::to_vec_pretty(&*inner) {
            let tmp = self.path.with_extension("json.tmp");
            if std::fs::write(&tmp, json).is_ok() {
                let _ = std::fs::rename(&tmp, &self.path);
            }
        }
        drop(inner);
        self.notify();
    }

    fn notify(&self) {
        self.changed.notify_all();
        if let Some(w) = self.waker.get() {
            w();
        }
    }

    /// Create a session, or (unless `fresh`) pick up the latest existing one
    /// with the same name. Returns its id and whether it was resumed.
    pub fn open_session(&self, name: &str, fresh: bool) -> (String, bool) {
        let mut inner = self.lock();
        if !fresh
            && !name.is_empty()
            && let Some(at) = inner.sessions.iter().rposition(|s| s.name == name)
        {
            // Back in use: move it to the newest position.
            let mut session = inner.sessions.remove(at);
            session.ended = false;
            session.last_seen = Some(Instant::now());
            let id = session.id.clone();
            inner.sessions.push(session);
            self.commit(inner);
            return (id, true);
        }
        let id = random_id();
        let name = if name.is_empty() {
            format!("Session {}", inner.sessions.len() + 1)
        } else {
            name.chars().take(80).collect()
        };
        inner.sessions.push(Session {
            id: id.clone(),
            name,
            created: now(),
            messages: Vec::new(),
            ended: false,
            interruption: None,
            listeners: 0,
            last_seen: Some(Instant::now()),
        });
        self.commit(inner);
        (id, false)
    }

    /// The last few messages of a session, oldest first, as (role, text).
    pub fn recent(&self, id: &str, count: usize) -> Vec<(Role, String)> {
        let mut inner = self.lock();
        let Some(s) = inner.session(id) else { return Vec::new() };
        let from = s.messages.len().saturating_sub(count);
        s.messages[from..].iter().map(|m| (m.role, m.text.clone())).collect()
    }

    /// Speech was paused (not interrupted) at message `index`: queue it and
    /// the agent messages after it to be spoken again.
    pub fn requeue_speech(&self, id: &str, index: usize) {
        let mut inner = self.lock();
        if let Some(s) = inner.session(id) {
            for m in s.messages.iter_mut().skip(index) {
                if m.role == Role::Agent && m.heard.is_none() {
                    m.pending = true;
                }
            }
        }
        self.commit(inner);
    }

    fn push(&self, id: &str, role: Role, text: &str) -> Result<(), NotFound> {
        let mut inner = self.lock();
        let s = inner.session(id).ok_or(NotFound)?;
        if role == Role::Agent {
            s.last_seen = Some(Instant::now());
            s.ended = false;
        }
        s.messages.push(Message {
            role,
            text: text.to_string(),
            time: now(),
            pending: true,
            heard: None,
        });
        self.commit(inner);
        Ok(())
    }

    pub fn agent_says(&self, id: &str, text: &str) -> Result<(), NotFound> {
        self.push(id, Role::Agent, text)
    }

    pub fn user_says(&self, id: &str, text: &str) {
        let _ = self.push(id, Role::User, text);
    }

    /// Block until the user has said something in this session (or timeout).
    pub fn listen(&self, id: &str, timeout: Duration) -> Result<Option<Delivery>, NotFound> {
        let deadline = Instant::now() + timeout;
        let mut inner = self.lock();
        {
            let s = inner.session(id).ok_or(NotFound)?;
            s.listeners += 1;
            s.last_seen = Some(Instant::now());
            s.ended = false;
        }
        drop(inner);
        self.notify();
        inner = self.lock();
        let result = loop {
            let Some(s) = inner.session(id) else {
                return Err(NotFound); // deleted while waiting
            };
            let idx: Vec<usize> = (0..s.messages.len())
                .filter(|&i| s.messages[i].role == Role::User && s.messages[i].pending)
                .collect();
            if !idx.is_empty() {
                // What the user said before the agent's latest message was
                // posted is not a reply to it; keep the two apart.
                let last_agent = s.messages.iter().rposition(|m| m.role == Role::Agent);
                let (mut earlier, mut text) = (Vec::new(), Vec::new());
                for &i in &idx {
                    s.messages[i].pending = false;
                    let said = s.messages[i].text.clone();
                    if last_agent.is_some_and(|a| i < a) {
                        earlier.push(said);
                    } else {
                        text.push(said);
                    }
                }
                break Some(Delivery {
                    text: text.join("\n"),
                    earlier: earlier.join("\n"),
                    indexes: idx,
                    interruption: s.interruption.take(),
                });
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break None;
            }
            inner = self
                .changed
                .wait_timeout(inner, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        };
        if let Some(s) = inner.session(id) {
            s.listeners = s.listeners.saturating_sub(1);
            s.last_seen = Some(Instant::now());
        }
        self.commit(inner);
        Ok(result)
    }

    /// The response carrying these messages never reached the agent; queue them again.
    pub fn undeliver(&self, id: &str, idx: &[usize]) {
        let mut inner = self.lock();
        if let Some(s) = inner.session(id) {
            for &i in idx {
                if let Some(m) = s.messages.get_mut(i) {
                    m.pending = true;
                }
            }
        }
        self.commit(inner);
    }

    /// The user cut off the agent's speech while message `index` was being
    /// read, having heard only `heard`. Later queued messages were not heard.
    pub fn interrupted(&self, id: &str, index: usize, heard: &str) {
        let mut inner = self.lock();
        let Some(s) = inner.session(id) else { return };
        let Some(m) = s.messages.get_mut(index) else { return };
        let heard = heard.trim();
        let rest = m.text.get(heard.len()..).unwrap_or("").trim().to_string();
        m.heard = Some(heard.to_string());
        let mut unheard_later = 0;
        for later in s.messages[index + 1..].iter_mut() {
            if later.role == Role::Agent && !later.pending && later.heard.is_none() {
                later.heard = Some(String::new());
                unheard_later += 1;
            }
        }
        if rest.is_empty() && unheard_later == 0 {
            return self.commit(inner); // cut off at the very end: nothing was missed
        }
        let mut note = if heard.is_empty() {
            "The user interrupted you before any of your message was read aloud.".to_string()
        } else {
            format!(
                "The user interrupted you while your message was being read aloud. They heard only: \"{heard}\" — they did NOT hear the rest: \"{rest}\""
            )
        };
        if unheard_later > 0 {
            note += &format!(" Your {unheard_later} later message(s) were not heard at all.");
        }
        s.interruption = Some(match s.interruption.take() {
            Some(prev) => format!("{prev} {note}"),
            None => note,
        });
        self.commit(inner);
    }

    pub fn end_session(&self, id: &str) -> Result<(), NotFound> {
        let mut inner = self.lock();
        let s = inner.session(id).ok_or(NotFound)?;
        s.ended = true;
        s.last_seen = None;
        self.commit(inner);
        Ok(())
    }

    pub fn delete_session(&self, id: &str) {
        let mut inner = self.lock();
        inner.sessions.retain(|s| s.id != id);
        self.commit(inner);
    }
}
