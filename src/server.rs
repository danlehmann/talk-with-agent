//! The local HTTP API that agents use. Everything is plain curl-able:
//! text request bodies, JSON responses.

use crate::state::{NotFound, Store};
use serde_json::json;
use std::io::Read;
use std::sync::Arc;
use std::time::Duration;
use tiny_http::{Header, Method, Request, Response, Server};

pub const DEFAULT_PORT: u16 = 7717;
const DEFAULT_LISTEN_SECS: u64 = 90;
const MAX_LISTEN_SECS: u64 = 590;
const MAX_BODY: u64 = 1 << 20;

/// The secret path prefix that every request must carry. The server is
/// reachable from the local network, so this is what keeps strangers out.
fn access_key() -> String {
    let path = crate::state::data_dir().join("access-key");
    if let Ok(key) = std::fs::read_to_string(&path)
        && key.trim().len() >= 16
    {
        return key.trim().to_string();
    }
    let key = crate::state::random_id();
    let _ = std::fs::write(&path, &key);
    key
}

/// This machine's address on the local network, for agents on other machines.
/// (mDNS names like `my-mac.local` often don't resolve from containers/Linux.)
fn lan_address() -> String {
    // Connecting a UDP socket sends nothing; it just picks the outgoing interface.
    std::net::UdpSocket::bind("0.0.0.0:0")
        .and_then(|s| s.connect("192.0.2.1:9").and_then(|_| s.local_addr()))
        .map(|a| a.ip().to_string())
        .unwrap_or_else(|_| "127.0.0.1".into())
}

/// Serve on all interfaces in background threads. Returns the base URL.
pub fn start(store: Arc<Store>) -> Result<String, String> {
    let server = Server::http(("0.0.0.0", DEFAULT_PORT))
        .or_else(|_| Server::http(("0.0.0.0", 0)))
        .map_err(|e| format!("could not start HTTP server: {e}"))?;
    let port = server
        .server_addr()
        .to_ip()
        .map(|a| a.port())
        .unwrap_or(DEFAULT_PORT);
    let key = access_key();
    let base = format!("http://{}:{port}/{key}", lan_address());
    let base_for_threads = base.clone();
    std::thread::spawn(move || {
        for request in server.incoming_requests() {
            let store = store.clone();
            let base = base_for_threads.clone();
            // One thread per request: `listen` long-polls.
            let key = key.clone();
            std::thread::spawn(move || handle(request, &store, &base, &key));
        }
    });
    Ok(base)
}

fn header<'a>(req: &'a Request, name: &'static str) -> Option<&'a str> {
    req.headers()
        .iter()
        .find(|h| h.field.equiv(name))
        .map(|h| h.value.as_str())
}

fn respond(req: Request, status: u16, content_type: &str, body: String) -> bool {
    let resp = Response::from_string(body)
        .with_status_code(status)
        .with_header(Header::from_bytes("Content-Type", content_type).unwrap());
    req.respond(resp).is_ok()
}

fn respond_json(req: Request, status: u16, value: serde_json::Value) -> bool {
    respond(req, status, "application/json", format!("{value}\n"))
}

/// Body as text. Accepts plain text, or JSON like {"text": "..."} / {"name": "..."}.
fn body_text(req: &mut Request, json_key: &str) -> String {
    let mut body = String::new();
    let _ = req.as_reader().take(MAX_BODY).read_to_string(&mut body);
    let trimmed = body.trim();
    if trimmed.starts_with('{')
        && let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed)
        && let Some(s) = v.get(json_key).and_then(|s| s.as_str())
    {
        return s.trim().to_string();
    }
    trimmed.to_string()
}

fn handle(mut req: Request, store: &Store, base: &str, key: &str) {
    let url = req.url().to_string();
    // Requests must carry the access key as the first path segment, and must
    // not come from a web page (those carry an Origin header).
    let rest = url
        .strip_prefix('/')
        .and_then(|u| u.strip_prefix(key))
        .filter(|r| r.is_empty() || r.starts_with('/') || r.starts_with('?'));
    let (Some(url), None) = (rest, header(&req, "Origin")) else {
        respond_json(req, 403, json!({"status": "error", "error": "forbidden: use the full URL shown in the app, including the access key"}));
        return;
    };

    let (path, query) = url.split_once('?').unwrap_or((url, ""));
    let timeout = query
        .split('&')
        .find_map(|kv| kv.strip_prefix("timeout="))
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_LISTEN_SECS)
        .min(MAX_LISTEN_SECS);
    let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
    let method = req.method().clone();

    let not_found = |req: Request| {
        respond_json(
            req,
            404,
            json!({"status": "error", "error": format!("not found; GET {base}/ for instructions. If your session is gone, the user deleted it: stop, or create a new one.")}),
        );
    };

    match (&method, parts.as_slice()) {
        (Method::Get, []) => {
            respond(req, 200, "text/plain; charset=utf-8", instructions(base));
        }
        (Method::Post, ["sessions"]) => {
            let name = body_text(&mut req, "name");
            let fresh = query.split('&').any(|kv| kv == "new=1");
            let (id, resumed) = store.open_session(&name, fresh);
            let url = format!("{base}/sessions/{id}");
            let mut body = json!({"status": "ok", "session_id": id, "session_url": url, "resumed": resumed});
            if resumed {
                let history: Vec<_> = store
                    .recent(&id, 20)
                    .into_iter()
                    .map(|(role, text)| {
                        json!({"who": if role == crate::state::Role::User { "user" } else { "you" }, "text": text})
                    })
                    .collect();
                body["recent_history"] = json!(history);
                body["note"] = json!("An earlier conversation with this name exists and was resumed; recent_history shows how it ended. Greet the user as someone continuing that conversation.");
            }
            respond_json(req, 200, body);
        }
        (Method::Post, ["sessions", id, action @ ("say" | "ask")]) => {
            let text = body_text(&mut req, "text");
            if text.is_empty() {
                respond_json(req, 400, json!({"status": "error", "error": "empty body; send the text to speak as the request body"}));
                return;
            }
            match store.agent_says(id, &text) {
                Err(NotFound) => not_found(req),
                Ok(()) if *action == "say" => {
                    respond_json(req, 200, json!({"status": "ok"}));
                }
                Ok(()) => listen(req, store, id, timeout, &not_found),
            }
        }
        (Method::Get | Method::Post, ["sessions", id, "listen"]) => {
            listen(req, store, id, timeout, &not_found)
        }
        (Method::Delete, ["sessions", id]) => match store.end_session(id) {
            Ok(()) => {
                respond_json(req, 200, json!({"status": "ok"}));
            }
            Err(NotFound) => not_found(req),
        },
        _ => not_found(req),
    }
}

fn listen(req: Request, store: &Store, id: &str, timeout: u64, not_found: &dyn Fn(Request)) {
    match store.listen(id, Duration::from_secs(timeout)) {
        Err(NotFound) => not_found(req),
        Ok(None) => {
            respond_json(
                req,
                200,
                json!({"status": "timeout", "hint": "The user has not spoken yet. Call listen again."}),
            );
        }
        Ok(Some(delivery)) => {
            let mut body = json!({"status": "message", "text": delivery.text});
            if !delivery.earlier.is_empty() {
                body["said_before_your_last_message"] = json!(delivery.earlier);
                body["note"] = json!(if delivery.text.is_empty() {
                    "The user said this while you were busy, BEFORE your latest message was read to them. It is not a reply to that message. Take it into account; if you still need an answer to your latest message, call listen again."
                } else {
                    "said_before_your_last_message was spoken while you were busy, before your latest message was read to the user; it is not a reply to it. text is what they said afterwards."
                });
            }
            if let Some(note) = delivery.interruption {
                body["interrupted"] = json!(note);
            }
            if !respond_json(req, 200, body) {
                store.undeliver(id, &delivery.indexes);
            }
        }
    }
}

fn instructions(base: &str) -> String {
    format!(
        r#"TALK WITH AGENT — voice conversation bridge
==================================================

This server lets you (a coding agent) hold a spoken conversation with the
user. What you send is read aloud to them with text-to-speech; what they say
is transcribed and returned to you. Use curl from your shell.

1. Create a session (the body is a short name shown to the user, e.g. the
   project or task you are working on):

     curl -s -X POST {base}/sessions -d 'my-project: fixing login bug'

   -> {{"status":"ok","session_id":"ID","session_url":"{base}/sessions/ID"}}

   Use a stable name (project and topic). If a session with exactly that
   name already exists, it is resumed instead of creating a duplicate: the
   reply then has "resumed": true and "recent_history". To force a separate
   session, POST to {base}/sessions?new=1 instead. If you already hold a
   session ID from earlier in your own conversation, just keep using it.

2. Speak, then wait for the user's spoken reply (one call):

     curl -s --max-time 100 -X POST {base}/sessions/ID/ask --data-binary @- <<'EOF'
     Hi! I'm connected. What would you like to work on?
     EOF

   -> {{"status":"message","text":"what the user said"}}
   -> {{"status":"timeout"}}   (user has not replied yet: call listen, below)

3. Other calls:

     POST {base}/sessions/ID/say      speak without waiting (body = text),
                                      e.g. a progress update before long work
     GET  {base}/sessions/ID/listen   wait for the user to say something
     DELETE {base}/sessions/ID        end the conversation (say goodbye first)

   ask and listen block for up to 90 seconds by default; add ?timeout=SECONDS
   (max 590) to change that, and keep your shell command's own timeout larger.
   On {{"status":"timeout"}} just call listen again — the user may be thinking
   or away. Keep waiting until they reply.

HOW TO BEHAVE

- This is a conversation loop: ask -> do what the user asked with your normal
  tools -> ask again with the result. Never finish your turn without calling
  ask or listen again, until the user says they are done. Then say goodbye and
  DELETE the session.
- Everything you send is HEARD, not read. Write the way you would speak:
  short, plain sentences. No markdown, bullet lists, code, file paths, URLs,
  or long identifiers unless the user asks. Summarize; offer detail on request.
- Keep turns brief (a few sentences) and end with a clear question or next
  step so the user knows it is their turn.
- Before work that takes more than a few seconds, use say to tell the user
  what you are about to do. When it is done, report the outcome with ask.
- The user can interrupt you while your message is being read aloud. The
  reply then has an extra "interrupted" field saying how much they heard.
  Do not assume they know the part they did not hear, and take the hint:
  they probably want shorter answers or a change of direction.
- The user may speak while you are working. If they did so before your
  latest message, the reply has a "said_before_your_last_message" field
  (and "text" may be empty). That is not an answer to your latest message.
- The user's words come from speech recognition and may contain mistakes,
  especially for technical terms and names. Infer the likely meaning; if
  something is ambiguous and matters, ask them to confirm.
- A 404 for your session means the user deleted it. Stop the loop.
"#
    )
}
