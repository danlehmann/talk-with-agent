//! The window: session list on the left, conversation on the right.
//! Platform-independent; voice goes through the `Voice` trait.

use crate::settings::Settings;
use crate::state::{Role, Session, Store};
use crate::voice::{Voice, VoiceEvent, megabytes};
use eframe::egui::{self, Color32, RichText};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Spoken when the user picks a voice.
const VOICE_SAMPLE: &str = "This is how I sound.";

/// In hands-free mode, a pause this long after speech sends the message.
const PAUSE_TO_SEND: Duration = Duration::from_millis(2500);
/// How long the agent's voice waits to find out whether the user is cutting in.
const HOLD: Duration = Duration::from_millis(900);
/// How long after the agent stops its words can still turn up in the transcript.
const ECHO_TAIL: Duration = Duration::from_millis(2500);
/// How far the live transcript can lag behind the microphone.
const CATCH_UP: Duration = Duration::from_millis(1300);

const GREEN: Color32 = Color32::from_rgb(80, 180, 100);
const AMBER: Color32 = Color32::from_rgb(220, 160, 50);
const RED: Color32 = Color32::from_rgb(220, 80, 80);
const BLUE: Color32 = Color32::from_rgb(90, 150, 240);

pub struct App {
    store: Arc<Store>,
    voice: Box<dyn Voice>,
    /// Base URL of the HTTP server, or why it failed to start.
    server: Result<String, String>,
    selected: Option<String>,
    listening: bool,
    partial: String,
    partial_changed: Instant,
    discard_next: bool,
    hands_free: bool,
    /// Side-panel mode: only the conversation, always on top.
    compact: bool,
    conversing: bool,
    /// The user muted the conversation: nothing is spoken or heard until resumed.
    paused: bool,
    window_ready: bool,
    /// The full window's outer position and inner size from before switching
    /// to the side panel, to put it back exactly.
    full_window: Option<(egui::Pos2, egui::Vec2)>,
    /// Frames left until the full window is put in place again, and where.
    restore_window: Option<(u8, egui::Pos2, egui::Vec2)>,
    was_speaking: bool,
    /// What the agent is saying right now, to recognize its echo in the microphone.
    speaking_text: String,
    /// The agent's voice is paused because the user may have started talking:
    /// since when, and how many foreign words had been heard then.
    hold: Option<(Instant, usize)>,
    /// When the agent last finished talking.
    spoke_until: Instant,
    /// Foreign words in the live transcript that turned out to be nothing.
    dismissed: usize,
    draft: String,
    error: Option<String>,
    confirm_delete: Option<String>,
    copied: Option<(Instant, &'static str)>,
    settings: Settings,
    /// The user put off the question about downloading models, for this run.
    setup_dismissed: bool,
    show_settings: bool,
    /// Id of the model whose removal awaits confirmation.
    confirm_remove: Option<String>,
}

impl App {
    pub fn new(
        store: Arc<Store>,
        mut voice: Box<dyn Voice>,
        server: Result<String, String>,
    ) -> Self {
        let settings = Settings::load();
        if let Some(id) = &settings.voice {
            voice.set_voice(id);
        }
        App {
            settings,
            setup_dismissed: false,
            show_settings: false,
            confirm_remove: None,
            store,
            voice,
            server,
            selected: None,
            listening: false,
            partial: String::new(),
            partial_changed: Instant::now(),
            discard_next: false,
            hands_free: true,
            compact: false,
            conversing: false,
            paused: false,
            window_ready: false,
            full_window: None,
            restore_window: None,
            was_speaking: false,
            speaking_text: String::new(),
            hold: None,
            spoke_until: Instant::now(),
            dismissed: 0,
            draft: String::new(),
            error: None,
            confirm_delete: None,
            copied: None,
        }
    }

    fn start_listening(&mut self) {
        if self.listening || self.selected.is_none() {
            return;
        }
        self.error = None;
        self.partial.clear();
        self.discard_next = false;
        self.listening = true;
        self.voice.start_listening();
    }

    fn stop_listening(&mut self, discard: bool) {
        if self.listening {
            self.discard_next |= discard;
            if discard {
                self.partial.clear(); // show at once that it's thrown away
            }
            self.voice.stop_listening();
        }
    }

    /// Mute or unmute the conversation: while paused the microphone is closed
    /// and the agent's messages wait. Speech that was cut short starts again
    /// on resume, since a pause is not an interruption.
    fn set_paused(&mut self, paused: bool) {
        if self.paused == paused {
            return;
        }
        self.paused = paused;
        if paused {
            self.hold = None;
            if let Some((index, _)) = self.voice.stop_speaking()
                && let Some(id) = &self.selected
            {
                self.store.requeue_speech(id, index as usize);
            }
            self.stop_listening(true);
        }
        self.voice.set_paused(paused);
    }

    /// Cut the agent off mid-speech and record how far it got, so it can be told.
    fn interrupt(&mut self) {
        self.hold = None;
        if let Some((index, heard)) = self.voice.stop_speaking()
            && let Some(id) = &self.selected
        {
            self.store.interrupted(id, index as usize, &heard);
        }
    }

    fn send(&mut self, text: &str) {
        let text = text.trim();
        if let Some(id) = &self.selected
            && !text.is_empty()
        {
            self.store.user_says(id, text);
        }
    }

    fn select(&mut self, id: Option<String>) {
        if self.selected != id {
            self.stop_listening(true);
            self.interrupt();
            self.selected = id;
            self.confirm_delete = None;
        }
    }

    fn copy(&mut self, ctx: &egui::Context, what: &'static str, text: String) {
        ctx.copy_text(text);
        self.copied = Some((Instant::now(), what));
    }

    /// Switch between the full window and a slim always-on-top strip at the
    /// right edge of the screen that shows only the conversation.
    fn set_compact(&mut self, ctx: &egui::Context, compact: bool) {
        use egui::{ViewportCommand, WindowLevel};
        self.compact = compact;
        if compact {
            // Remember where the full window was. (Position is of the frame,
            // size of the content: those are what can be set again.)
            self.full_window = ctx.input(|i| {
                let viewport = i.viewport();
                Some((viewport.outer_rect?.min, viewport.inner_rect?.size()))
            });
            crate::window::set_shadow(false);
            ctx.send_viewport_cmd(ViewportCommand::Transparent(true));
            ctx.send_viewport_cmd(ViewportCommand::Decorations(false));
            ctx.send_viewport_cmd(ViewportCommand::WindowLevel(WindowLevel::AlwaysOnTop));
            let width = 340.0;
            if let Some(screen) = ctx.input(|i| i.viewport().monitor_size) {
                let height = (screen.y * 0.6).max(400.0);
                ctx.send_viewport_cmd(ViewportCommand::InnerSize(egui::vec2(width, height)));
                ctx.send_viewport_cmd(ViewportCommand::OuterPosition(egui::pos2(
                    screen.x - width - 12.0,
                    40.0, // just below the menu bar
                )));
            } else {
                ctx.send_viewport_cmd(ViewportCommand::InnerSize(egui::vec2(width, 700.0)));
            }
        } else {
            ctx.send_viewport_cmd(ViewportCommand::Transparent(false));
            ctx.send_viewport_cmd(ViewportCommand::Decorations(true));
            ctx.send_viewport_cmd(ViewportCommand::WindowLevel(WindowLevel::Normal));
            let (position, size) = self
                .full_window
                .take()
                .unwrap_or((egui::pos2(120.0, 80.0), egui::vec2(900.0, 640.0)));
            ctx.send_viewport_cmd(ViewportCommand::InnerSize(size));
            ctx.send_viewport_cmd(ViewportCommand::OuterPosition(position));
            crate::window::set_shadow(true);
            // The title bar is back only after these commands are processed;
            // place the window once more then, so it can't end up shifted.
            self.restore_window = Some((2, position, size));
        }
    }

    /// The side panel: a translucent overlay with just the conversation.
    fn overlay(&mut self, ui: &mut egui::Ui) {
        let name_and_status = self.selected.as_ref().and_then(|id| {
            let inner = self.store.lock();
            inner
                .sessions
                .iter()
                .find(|s| &s.id == id)
                .map(|s| (s.name.clone(), status(s)))
        });
        egui::Frame::new()
            .fill(Color32::from_rgba_unmultiplied(10, 10, 14, 215))
            .corner_radius(14)
            .inner_margin(14)
            .show(ui, |ui| {
                ui.set_min_size(ui.available_size());
                // Bright text on the dark glass.
                ui.visuals_mut().override_text_color = Some(Color32::from_gray(245));
                ui.visuals_mut().widgets.noninteractive.fg_stroke.color = Color32::from_gray(160);

                let header = ui
                    .horizontal(|ui| {
                        if let Some((name, (text, color))) = &name_and_status {
                            ui.colored_label(*color, "●").on_hover_text(*text);
                            ui.label(RichText::new(name).small().color(Color32::from_gray(170)));
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let close = egui::Button::new(
                                RichText::new("×").size(22.0).color(Color32::from_gray(220)),
                            )
                            .frame(false);
                            if ui
                                .add(close)
                                .on_hover_text("Back to the full window")
                                .clicked()
                            {
                                self.set_compact(ui.ctx(), false);
                            }
                        });
                    })
                    .response;
                // No title bar: drag the header to move the overlay.
                if header.interact(egui::Sense::drag()).drag_started() {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
                }
                ui.add_space(6.0);

                self.conversation(ui);

                if self.paused {
                    if ui
                        .button(
                            RichText::new("Paused. Click or press play to resume.").color(AMBER),
                        )
                        .clicked()
                    {
                        self.set_paused(false);
                    }
                } else if let Some(e) = &self.error {
                    ui.colored_label(RED, e);
                } else if self.listening && !self.partial.is_empty() {
                    ui.label(RichText::new(&self.partial).italics().color(BLUE));
                }
            });
    }

    fn sidebar(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.heading("Talk with Agent");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Where there is a menu bar, Settings lives there.
                if !crate::window::HAS_SETTINGS_MENU && ui.small_button("Settings").clicked() {
                    self.show_settings = !self.show_settings;
                }
                if ui
                    .small_button("Side panel")
                    .on_hover_text(
                        "Side panel: a slim always-on-top strip with just the conversation",
                    )
                    .clicked()
                {
                    self.set_compact(ui.ctx(), true);
                }
            });
        });
        ui.add_space(6.0);

        match self.server.clone() {
            Ok(url) => {
                ui.label(RichText::new("Give this URL to an agent:").small());
                ui.monospace(&url);
                ui.horizontal(|ui| {
                    if ui.button("Copy URL").clicked() {
                        self.copy(ui.ctx(), "url", url.clone());
                    }
                    if ui
                        .button("Copy prompt")
                        .on_hover_text("Copies a ready-to-paste instruction for a coding agent")
                        .clicked()
                    {
                        self.copy(ui.ctx(), "prompt", format!(
                            "Let's talk by voice. Run `curl -s {url}` and follow the instructions it returns. Speak in short sentences - this should be a conversation, not a dump of all the information all at once."
                        ));
                    }
                });
                if let Some((at, what)) = self.copied {
                    if at.elapsed() < Duration::from_secs(2) {
                        ui.label(RichText::new(format!("Copied {what}")).small().color(GREEN));
                        ui.ctx().request_repaint_after(Duration::from_secs(2));
                    } else {
                        self.copied = None;
                    }
                }
            }
            Err(e) => {
                ui.colored_label(RED, e);
            }
        }

        ui.add_space(6.0);
        ui.separator();
        ui.label(RichText::new("Sessions").strong());

        // Snapshot what we need so the lock isn't held while handling clicks.
        struct Row {
            id: String,
            name: String,
            status: (&'static str, Color32),
            unheard: usize,
        }
        let rows: Vec<Row> = self
            .store
            .lock()
            .sessions
            .iter()
            .rev()
            .map(|s| Row {
                id: s.id.clone(),
                name: s.name.clone(),
                status: status(s),
                unheard: s.unheard(),
            })
            .collect();

        if rows.is_empty() {
            ui.label(
                RichText::new(
                    "No sessions yet. Paste the prompt into a coding agent to start one.",
                )
                .weak(),
            );
        }

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for row in rows {
                    let is_selected = self.selected.as_deref() == Some(&row.id);
                    ui.horizontal(|ui| {
                        ui.colored_label(row.status.1, "●")
                            .on_hover_text(row.status.0);
                        let mut label = row.name.clone();
                        if row.unheard > 0 && !is_selected {
                            label = format!("{label}  ({} new)", row.unheard);
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if self.confirm_delete.as_deref() == Some(&row.id) {
                                if ui.button("Cancel").clicked() {
                                    self.confirm_delete = None;
                                }
                                if ui.button(RichText::new("Delete").color(RED)).clicked() {
                                    if is_selected {
                                        self.select(None);
                                    }
                                    self.store.delete_session(&row.id);
                                    self.confirm_delete = None;
                                }
                            } else if ui
                                .small_button("×")
                                .on_hover_text("Delete this session and its history")
                                .clicked()
                            {
                                self.confirm_delete = Some(row.id.clone());
                            }
                            ui.with_layout(
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .add(
                                            egui::Button::selectable(is_selected, label).truncate(),
                                        )
                                        .on_hover_text(row.status.0)
                                        .clicked()
                                    {
                                        self.select(Some(row.id.clone()));
                                    }
                                },
                            );
                        });
                    });
                }
            });
    }

    /// The choice of speaking voice.
    fn voice_picker(&mut self, ui: &mut egui::Ui) {
        let voices = self.voice.voices();
        if let Some(current) = self.voice.voice()
            && !voices.is_empty()
        {
            let shown = voices
                .iter()
                .find(|(id, _)| *id == current)
                .map_or(current.as_str(), |(_, label)| label);
            let mut chosen = None;
            egui::ComboBox::from_label("Voice")
                .selected_text(shown)
                .show_ui(ui, |ui| {
                    for (id, label) in &voices {
                        if ui.selectable_label(*id == current, label).clicked() && *id != current {
                            chosen = Some(id.clone());
                        }
                    }
                });
            if let Some(id) = chosen {
                self.voice.set_voice(&id);
                self.settings.voice = Some(id);
                self.settings.save();
                // Let the user hear the choice, unless the agent is talking.
                // (A sample still playing for the previous choice is cut off.)
                if self
                    .voice
                    .speaking_progress()
                    .is_some_and(|(id, _)| id == u64::MAX)
                {
                    self.voice.stop_speaking();
                }
                if !self.voice.is_speaking() && !self.paused {
                    self.speaking_text = VOICE_SAMPLE.to_string();
                    self.voice.speak(u64::MAX, VOICE_SAMPLE);
                }
            }
        }
    }

    /// The settings, in a window of their own.
    fn settings_window(&mut self, ctx: &egui::Context) {
        let builder = egui::ViewportBuilder::default()
            .with_title("Settings")
            .with_inner_size([500.0, 440.0])
            .with_resizable(false)
            .with_minimize_button(false)
            .with_maximize_button(false);
        ctx.show_viewport_immediate(settings_viewport(), builder, |ui, _class| {
            egui::CentralPanel::default().show(ui, |ui| self.settings(ui));
            if ui.input(|i| i.viewport().close_requested()) {
                self.show_settings = false;
            }
        });
    }

    /// Voice, transcription and the models they need.
    fn settings(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Speech").strong());
        self.voice_picker(ui);
        self.transcriber_picker(ui);
        ui.label(
            RichText::new("Both run on this computer. Without a transcription model, the system recognizer is used.")
                .small()
                .weak(),
        );

        ui.add_space(10.0);
        ui.separator();
        ui.label(RichText::new("Models").strong());
        let models = self.voice.models();
        let status = self.voice.status().unwrap_or_default();
        egui::Grid::new("models")
            .num_columns(3)
            .spacing([16.0, 8.0])
            .show(ui, |ui| {
                for model in &models {
                    ui.label(&model.name);
                    let size = megabytes(model.bytes);
                    if model.installed {
                        ui.label(RichText::new(format!("Installed  ·  {size}")).color(GREEN));
                        if self.confirm_remove.as_deref() == Some(model.id.as_str()) {
                            ui.horizontal(|ui| {
                                if ui.button(RichText::new("Delete").color(RED)).clicked() {
                                    self.voice.remove_model(&model.id);
                                    self.confirm_remove = None;
                                    self.remember_choices();
                                }
                                if ui.button("Keep").clicked() {
                                    self.confirm_remove = None;
                                }
                            });
                        } else if ui.button("Remove").clicked() {
                            self.confirm_remove = Some(model.id.clone());
                        }
                    } else {
                        ui.label(RichText::new(format!("Not installed  ·  {size}")).weak());
                        if ui.button("Download").clicked() {
                            self.voice.download_model(&model.id);
                            self.remember_choices();
                        }
                    }
                    ui.end_row();
                }
            });
        if !status.is_empty() {
            ui.label(RichText::new(status).small().weak());
        }

        ui.add_space(10.0);
        ui.separator();
        ui.label(RichText::new("Data").strong());
        let folder = crate::state::data_dir();
        ui.label(
            RichText::new(
                "Sessions, settings, pronunciations and models are all kept in one folder. \
                 To uninstall completely, quit the app and delete it.",
            )
            .small()
            .weak(),
        );
        ui.horizontal(|ui| {
            ui.monospace(folder.display().to_string());
        });
        if ui.button("Show in file manager").clicked() {
            crate::window::reveal(&folder);
        }
    }

    /// Save the choice of voice and transcriber as the backend now has them.
    fn remember_choices(&mut self) {
        self.settings.voice = self.voice.voice();
        self.settings.transcriber = self.voice.transcriber();
        self.settings.save();
    }

    /// The choice of transcription engine.
    fn transcriber_picker(&mut self, ui: &mut egui::Ui) {
        let choices = self.voice.transcribers();
        let Some(current) = self.voice.transcriber() else {
            return;
        };
        if choices.is_empty() {
            return;
        }
        // Short in the closed box, with the details in the list.
        let short = |label: &str| label.split(" (").next().unwrap_or(label).to_string();
        let shown = choices
            .iter()
            .find(|(id, _)| *id == current)
            .map_or(current.clone(), |(_, label)| short(label));
        let mut chosen = None;
        egui::ComboBox::from_label("Transcription")
            .selected_text(shown)
            .show_ui(ui, |ui| {
                for (id, label) in &choices {
                    if ui.selectable_label(*id == current, label).clicked() && *id != current {
                        chosen = Some(id.clone());
                    }
                }
            });
        if let Some(id) = chosen {
            self.voice.set_transcriber(&id);
            self.settings.transcriber = Some(id);
            self.settings.save();
            self.setup_dismissed = false;
        }
    }

    /// Asks before downloading speech models. Shown while some are missing
    /// and the user has not agreed to downloads yet.
    fn setup(&mut self, ui: &mut egui::Ui, missing: &[(String, u64)]) {
        ui.add_space(8.0);
        ui.label(RichText::new("Download speech models?").strong());
        ui.label(
            "Speech and transcription run on this computer, with models that are downloaded once. \
             Without them, replies are shown but not spoken, and the system recognizer transcribes.",
        );
        for (name, bytes) in missing {
            ui.label(format!("•  {name}: {}", megabytes(*bytes)));
        }
        self.transcriber_picker(ui);
        let total = megabytes(missing.iter().map(|(_, bytes)| bytes).sum());
        ui.label(
            RichText::new(format!(
                "{total} in total, from github.com and huggingface.co, saved in {}.",
                crate::state::data_dir().join("models").display()
            ))
            .small()
            .weak(),
        );
        ui.horizontal(|ui| {
            if ui.button(format!("Download ({total})")).clicked() {
                self.voice.allow_downloads();
                self.settings.downloads_allowed = true;
                self.settings.save();
            }
            if ui.button("Not now").clicked() {
                self.setup_dismissed = true;
            }
        });
        ui.add_space(8.0);
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        if let Some(e) = self.error.clone() {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(RED, e);
                if ui.small_button("Dismiss").clicked() {
                    self.error = None;
                }
            });
        }
        if self.listening {
            let shown = if self.partial.is_empty() {
                "Listening…"
            } else {
                &self.partial
            };
            ui.label(RichText::new(shown).italics().color(BLUE));
        }
        ui.horizontal(|ui| {
            if self.listening {
                if ui.button(RichText::new("Send").size(18.0).color(BLUE)).clicked() {
                    self.stop_listening(false);
                }
                if ui.button("Cancel").clicked() {
                    self.stop_listening(true);
                }
            } else if ui
                .button(RichText::new("Talk").size(18.0))
                .on_hover_text("Space")
                .clicked()
            {
                self.interrupt();
                self.start_listening();
            }
            if self.voice.is_speaking()
                && ui
                    .button("Interrupt")
                    .on_hover_text("Stop the agent talking. It is told how much you heard. (Space interrupts and starts listening.)")
                    .clicked()
            {
                self.interrupt();
            }
            if ui
                .button(if self.paused { "Resume" } else { "Pause" })
                .on_hover_text("Mute the conversation: nothing is spoken or heard until you resume. The play/pause button on your headset does the same.")
                .clicked()
            {
                self.set_paused(!self.paused);
            }
            ui.checkbox(&mut self.hands_free, "Hands-free").on_hover_text(
                "Keep the microphone open and send when you pause. Just start talking to interrupt the agent.",
            );
        });
        if !self.compact
            && let Some(status) = self.voice.status()
        {
            ui.label(RichText::new(status).small().weak());
        }
        ui.horizontal(|ui| {
            let send_clicked = ui
                .with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let clicked = ui.button("Send").clicked();
                    let edit = ui.add(
                        egui::TextEdit::singleline(&mut self.draft)
                            .hint_text("…or type a message")
                            .desired_width(f32::INFINITY),
                    );
                    let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if enter {
                        edit.request_focus();
                    }
                    clicked || enter
                })
                .inner;
            if send_clicked {
                let text = std::mem::take(&mut self.draft);
                self.send(&text);
            }
        });
        ui.add_space(6.0);
    }

    fn conversation(&mut self, ui: &mut egui::Ui) {
        let speaking = self.voice.speaking_progress();
        let store = self.store.clone();
        let inner = store.lock();
        let session = self
            .selected
            .as_ref()
            .and_then(|id| inner.sessions.iter().find(|s| &s.id == id));
        let Some(session) = session else {
            ui.centered_and_justified(|ui| {
                ui.label(
                    RichText::new(
                        "Select a session, or connect an agent using the URL on the left.",
                    )
                    .weak(),
                );
            });
            return;
        };

        if !self.compact {
            ui.horizontal(|ui| {
                ui.heading(&session.name);
                let (text, color) = status(session);
                ui.colored_label(color, format!("● {text}"));
            });
            ui.separator();
        }
        // In the overlay, leave room below for the live transcript.
        let height = ui.available_height() - if self.compact { 44.0 } else { 0.0 };
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .max_height(height)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for (index, m) in session.messages.iter().enumerate() {
                    let (who, color) = match m.role {
                        Role::User => ("You", BLUE),
                        Role::Agent => ("Agent", AMBER),
                    };
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(who).strong().color(color));
                        if let Some(time) = chrono::DateTime::from_timestamp(m.time as i64, 0) {
                            let time = time.with_timezone(&chrono::Local).format("%H:%M:%S");
                            ui.label(RichText::new(time.to_string()).small().weak());
                        }
                        if m.pending && m.role == Role::User {
                            ui.label(
                                RichText::new("queued — agent isn't listening yet")
                                    .small()
                                    .weak(),
                            );
                        }
                    });
                    match &m.heard {
                        Some(heard) => {
                            let rest = m.text.get(heard.len()..).unwrap_or("").trim();
                            if !heard.is_empty() {
                                ui.label(heard);
                            }
                            if !rest.is_empty() {
                                ui.label(
                                    RichText::new(format!("(interrupted, not heard: {rest})"))
                                        .weak(),
                                );
                            }
                        }
                        None => match speaking {
                            // Being read aloud: show how far the voice has got.
                            Some((id, spoken)) if id == index as u64 && m.role == Role::Agent => {
                                let mut spoken = spoken.min(m.text.len());
                                while !m.text.is_char_boundary(spoken) {
                                    spoken -= 1;
                                }
                                let font = egui::TextStyle::Body.resolve(ui.style());
                                let format = |color| egui::TextFormat {
                                    font_id: font.clone(),
                                    color,
                                    ..Default::default()
                                };
                                let mut job = egui::text::LayoutJob::default();
                                job.wrap.max_width = ui.available_width();
                                job.append(
                                    &m.text[..spoken],
                                    0.0,
                                    format(ui.visuals().strong_text_color()),
                                );
                                job.append(
                                    &m.text[spoken..],
                                    0.0,
                                    format(ui.visuals().weak_text_color()),
                                );
                                ui.label(job);
                            }
                            _ => {
                                ui.label(&m.text);
                            }
                        },
                    }
                    ui.add_space(8.0);
                }
            });
    }
}

fn settings_viewport() -> egui::ViewportId {
    egui::ViewportId::from_hash_of("settings")
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| w.chars().any(char::is_alphanumeric))
        .map(str::to_lowercase)
        .collect()
}

/// Whether two words are the same give or take a recognizer's slip
/// ("lane" / "lanes", "callee" / "colley").
fn sounds_alike(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let allowed = match a.len().min(b.len()) {
        0..=3 => return false,
        4 => 1,
        _ => 2,
    };
    if a.len().abs_diff(b.len()) > allowed {
        return false;
    }
    // Edit distance.
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let cost = diagonal + usize::from(x != y);
            diagonal = row[j + 1];
            row[j + 1] = cost.min(diagonal + 1).min(row[j] + 1);
        }
    }
    row[b.len()] <= allowed
}

/// What in a live transcript is not the agent's own voice coming back
/// through the microphone.
#[derive(Default, Clone, Copy, PartialEq, Debug)]
struct Foreign {
    /// Words in the whole transcript that the agent didn't say.
    total: usize,
    /// The same among the last few words, and how many words that is.
    recent: usize,
    of: usize,
}

impl Foreign {
    /// The newest words are mostly not the agent's: someone is talking.
    fn talking(self) -> bool {
        self.recent > 0 && self.recent * 10 > self.of * 4
    }
}

/// Compare `heard` (a live transcript) with `spoken` (what the agent is
/// saying). An echo repeats the agent's words in the agent's order, so the
/// words that don't fit into that order are someone else's.
fn foreign(heard: &str, spoken: &str) -> Foreign {
    const RECENT: usize = 6;
    let (heard, spoken) = (words(heard), words(spoken));
    // Longest common subsequence, keeping which heard words are part of it.
    let (n, m) = (heard.len(), spoken.len());
    let mut table = vec![0u16; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            table[at(i, j)] = if sounds_alike(&heard[i], &spoken[j]) {
                table[at(i + 1, j + 1)] + 1
            } else {
                table[at(i + 1, j)].max(table[at(i, j + 1)])
            };
        }
    }
    let mut result = Foreign {
        of: n.min(RECENT),
        ..Default::default()
    };
    let (mut i, mut j) = (0, 0);
    while i < n {
        if j < m
            && sounds_alike(&heard[i], &spoken[j])
            && table[at(i, j)] == table[at(i + 1, j + 1)] + 1
        {
            (i, j) = (i + 1, j + 1);
        } else if j < m && table[at(i, j + 1)] >= table[at(i + 1, j)] {
            j += 1;
        } else {
            result.total += 1;
            result.recent += usize::from(i + RECENT >= n);
            i += 1;
        }
    }
    result
}

/// Words that mean "be quiet" on their own.
fn asks_to_stop(heard: &str) -> bool {
    words(heard)
        .last()
        .is_some_and(|w| ["stop", "wait", "pause", "hold", "quiet"].contains(&w.as_str()))
}

fn status(s: &Session) -> (&'static str, Color32) {
    if s.ended {
        ("ended", Color32::GRAY)
    } else if s.listeners > 0 {
        ("waiting for you", GREEN)
    } else if s.last_seen.is_some() {
        ("working…", AMBER)
    } else {
        ("not connected", Color32::GRAY)
    }
}

impl eframe::App for App {
    /// Voice handling. Runs even when the window is hidden.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !std::mem::replace(&mut self.window_ready, true) {
            // The window is created able to be transparent (for the overlay),
            // but the regular view should be an ordinary opaque window.
            ctx.send_viewport_cmd(egui::ViewportCommand::Transparent(false));
            let waker = ctx.clone();
            crate::window::install_settings_menu(move || waker.request_repaint());
        }
        if crate::window::settings_requested() {
            if self.compact {
                self.set_compact(ctx, false);
            }
            self.show_settings = true;
            // If it is open already, bring it to the front.
            ctx.send_viewport_cmd_to(settings_viewport(), egui::ViewportCommand::Focus);
        }
        if let Some((frames, position, size)) = self.restore_window.take() {
            if frames > 0 {
                self.restore_window = Some((frames - 1, position, size));
                ctx.request_repaint();
            } else if !self.compact {
                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(size));
                ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(position));
                crate::window::set_shadow(true);
            }
        }
        for event in self.voice.poll() {
            match event {
                VoiceEvent::Partial(mut text) => {
                    if !text.chars().any(char::is_alphanumeric) {
                        text.clear(); // a lone "." is noise, not speech
                    }
                    if text != self.partial {
                        self.partial = text;
                        self.partial_changed = Instant::now();
                    }
                }
                VoiceEvent::Finished(text) => {
                    self.listening = false;
                    self.partial.clear();
                    if !std::mem::take(&mut self.discard_next) {
                        self.send(&text);
                    }
                }
                VoiceEvent::PauseRequested(pause) => {
                    self.set_paused(pause.unwrap_or(!self.paused));
                }
                VoiceEvent::Error(e) => {
                    self.listening = false;
                    self.discard_next = false;
                    self.partial.clear();
                    self.error = Some(e);
                    self.hands_free = false; // don't retry in a loop
                }
            }
        }

        // Keep the selection valid; pick the newest session if none is selected.
        let mut to_speak = Vec::new();
        let mut agent_present = false;
        {
            let store = self.store.clone();
            let mut inner = store.lock();
            if let Some(id) = &self.selected
                && !inner.sessions.iter().any(|s| &s.id == id)
            {
                self.selected = None;
                self.stop_listening(true);
            }
            if self.selected.is_none() {
                self.selected = inner.sessions.last().map(|s| s.id.clone());
            }
            // Speak what the agent said in the selected session (not over the user).
            if let Some(s) = self.selected.as_ref().and_then(|id| inner.session(id)) {
                agent_present = s.last_seen.is_some() && !s.ended;
                // In hands-free mode the microphone stays open while the agent
                // talks (so the user can cut in), but the agent doesn't start
                // talking while the user is in the middle of a sentence.
                let user_has_floor =
                    self.listening && (!self.hands_free || !self.partial.is_empty());
                if s.unheard() > 0 && !user_has_floor && !self.paused {
                    for (i, m) in s.messages.iter_mut().enumerate() {
                        if m.role == Role::Agent && m.pending {
                            m.pending = false;
                            to_speak.push((i as u64, m.text.clone()));
                        }
                    }
                }
            }
            if !to_speak.is_empty() {
                store.commit(inner);
            }
        }
        for (index, text) in &to_speak {
            self.voice.speak(*index, text);
            self.speaking_text.push_str(text);
            self.speaking_text.push(' ');
        }

        // A hands-free conversation shouldn't be cut short by the computer
        // dozing off while the agent works or the user thinks.
        let conversing = self.hands_free && agent_present;
        if conversing != self.conversing {
            self.conversing = conversing;
            self.voice.set_conversing(conversing);
            if !conversing && self.paused {
                self.set_paused(false);
            }
        }

        let speaking = self.voice.is_speaking();
        if self.hands_free && !self.paused {
            // The microphone is always open, so the conversation flows without
            // touching anything.
            if agent_present && !self.listening {
                self.start_listening();
            }
            // What the microphone picked up may be the agent's own voice, or
            // noise the recognizer made a word of.
            let heard = foreign(&self.partial, &self.speaking_text);
            if self.partial.is_empty() {
                self.dismissed = 0;
            }
            let fresh = heard.total > self.dismissed;
            if self.listening && speaking {
                match self.hold {
                    None if fresh && heard.talking() => {
                        if heard.recent >= 3 || asks_to_stop(&self.partial) {
                            eprintln!("[voice] interrupted");
                            self.interrupt(); // the user is talking over the agent
                        } else {
                            // A word or two: maybe the user, maybe a mishearing.
                            // Stop talking for a moment and see if more follows.
                            eprintln!("[voice] paused: possible interruption");
                            self.voice.hold_speaking(true);
                            self.hold = Some((Instant::now(), heard.total));
                        }
                    }
                    // The recognizer runs behind: for a moment it is still
                    // working on what it heard before the voice paused.
                    Some((since, at_first)) if since.elapsed() < CATCH_UP => {
                        self.hold = Some((since, at_first.max(heard.total)));
                    }
                    // After that nothing has been playing, so more words are
                    // the user's.
                    Some((_, at_first)) if heard.total > at_first => {
                        eprintln!("[voice] interrupted");
                        self.interrupt();
                    }
                    Some((since, _)) if since.elapsed() >= CATCH_UP + HOLD => {
                        // Nothing followed: carry on, and don't fall for those
                        // words again.
                        self.hold = None;
                        self.dismissed = heard.total;
                        self.voice.hold_speaking(false);
                    }
                    _ => {}
                }
            } else if self.listening && self.was_speaking {
                // The agent just finished uninterrupted: make sure its voice
                // doesn't end up in what the user says next.
                if self.partial.is_empty() {
                    self.voice.trim_recording();
                } else if !fresh || !heard.talking() {
                    self.stop_listening(true);
                }
            } else if self.listening
                && !self.partial.is_empty()
                && !self.speaking_text.is_empty()
                && (!fresh || !heard.talking())
            {
                // The end of what the agent said, transcribed a moment late.
                eprintln!("[voice] dropped late echo");
                self.stop_listening(true);
            } else if self.listening
                && !self.partial.is_empty()
                && self.partial_changed.elapsed() >= PAUSE_TO_SEND
            {
                self.stop_listening(false);
            }
        }
        if self.hold.is_some() && !(self.hands_free && self.listening && speaking) {
            // Whatever ended the wait, don't leave the voice paused.
            self.hold = None;
            self.voice.hold_speaking(false);
        }
        if self.was_speaking && !speaking {
            self.spoke_until = Instant::now();
        }
        // Keep what was said for a moment: the recognizer runs behind.
        if !speaking && self.spoke_until.elapsed() >= ECHO_TAIL {
            self.speaking_text.clear();
        }
        self.was_speaking = speaking;

        if speaking || self.listening {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    /// Only the overlay is see-through.
    fn clear_color(&self, visuals: &egui::Visuals) -> [f32; 4] {
        if self.compact {
            [0.0; 4]
        } else {
            visuals.panel_fill.to_normalized_gamma_f32()
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Space toggles the microphone when not typing.
        let typing = ui.ctx().memory(|m| m.focused().is_some());
        if !typing && ui.input(|i| i.key_pressed(egui::Key::Space)) {
            if self.voice.is_speaking() {
                self.interrupt();
            } else if self.listening {
                self.stop_listening(false);
            } else {
                self.start_listening();
            }
        }

        if self.compact {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| self.overlay(ui));
            return;
        }
        if self.show_settings {
            self.settings_window(ui.ctx());
        }
        let missing = self.voice.missing_models();
        if !missing.is_empty() && !self.setup_dismissed && !self.show_settings {
            egui::Panel::top("setup").show(ui, |ui| self.setup(ui, &missing));
        }
        egui::Panel::left("sessions")
            .default_size(260.0)
            .frame(
                egui::Frame::new()
                    .fill(crate::theme::sidebar_fill(ui.visuals()))
                    .inner_margin(12),
            )
            .show(ui, |ui| self.sidebar(ui));
        if self.selected.is_some() {
            egui::Panel::bottom("controls").show(ui, |ui| self.controls(ui));
        }
        egui::CentralPanel::default().show(ui, |ui| self.conversation(ui));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::Model;

    /// A backend with one model installed and one not.
    #[derive(Default)]
    struct FakeVoice {
        removed: Vec<String>,
        downloads_allowed: bool,
    }

    impl Voice for FakeVoice {
        fn speak(&mut self, _id: u64, _text: &str) {}
        fn stop_speaking(&mut self) -> Option<(u64, String)> {
            None
        }
        fn is_speaking(&self) -> bool {
            false
        }
        fn start_listening(&mut self) {}
        fn stop_listening(&mut self) {}
        fn poll(&mut self) -> Vec<VoiceEvent> {
            Vec::new()
        }
        fn voices(&self) -> Vec<(String, String)> {
            vec![
                ("a".into(), "Voice A".into()),
                ("none".into(), "Off".into()),
            ]
        }
        fn voice(&self) -> Option<String> {
            Some("a".into())
        }
        fn transcribers(&self) -> Vec<(String, String)> {
            vec![("small".into(), "Small (fast; 574 MB download)".into())]
        }
        fn transcriber(&self) -> Option<String> {
            Some("small".into())
        }
        fn models(&self) -> Vec<Model> {
            vec![
                Model {
                    id: "voice".into(),
                    name: "Voice model".into(),
                    bytes: 350_000_000,
                    installed: true,
                },
                Model {
                    id: "small".into(),
                    name: "Small".into(),
                    bytes: 574_000_000,
                    installed: false,
                },
            ]
        }
        fn remove_model(&mut self, id: &str) {
            self.removed.push(id.to_string());
        }
        fn missing_models(&self) -> Vec<(String, u64)> {
            if self.downloads_allowed {
                Vec::new()
            } else {
                vec![("Small".into(), 574_000_000)]
            }
        }
        fn allow_downloads(&mut self) {
            self.downloads_allowed = true;
        }
    }

    /// The main window, the download question and the settings window all
    /// lay out without trouble, in both themes.
    #[test]
    fn tells_the_user_from_the_echo() {
        let spoken = "Done, it's inlined. Line 140 now asks whether callee inputs of the function has a next item.";
        // Nothing but punctuation is not speech.
        assert!(!foreign(".", spoken).talking());
        assert!(!foreign("", spoken).talking());
        // The agent's own words, also misheard a little, are echo.
        assert!(!foreign("Done, it's in line. Line 140 now asks", spoken).talking());
        assert!(!foreign("whether colley inputs of the functions", spoken).talking());
        assert!(sounds_alike("accept", "except") && !sounds_alike("next", "item"));
        // Other words are someone talking; few of them is weak evidence.
        let weak = foreign("Concerned?", spoken);
        assert!(weak.talking() && weak.recent < 3);
        assert!(foreign("Hold on, I am not done here yet", spoken).recent >= 3);
        // Also after a long stretch of echo.
        let late = foreign(
            "Done, it's inlined. Line 140 now asks whether, no wait, go back please",
            spoken,
        );
        assert!(late.talking() && late.recent >= 3, "{late:?}");
        // The agent's words in another order are not an echo.
        assert!(foreign("item next has the function", spoken).talking());
        assert!(asks_to_stop("Stop.") && !asks_to_stop("Stop it now"));
    }

    #[test]
    fn screens_render() {
        crate::state::use_test_data_dir();
        let ctx = egui::Context::default();
        crate::theme::apply(&ctx);
        let store = Arc::new(Store::load());
        let mut app = App::new(
            store,
            Box::new(FakeVoice::default()),
            Ok("http://127.0.0.1:7717/key".into()),
        );
        for theme in [egui::Theme::Dark, egui::Theme::Light] {
            ctx.set_theme(theme);
            for settings in [false, true] {
                app.show_settings = settings;
                app.confirm_remove = settings.then(|| "voice".to_string());
                for _ in 0..3 {
                    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                        let missing = app.voice.missing_models();
                        if settings {
                            app.settings_window(ui.ctx());
                        } else {
                            egui::Panel::top("setup").show(ui, |ui| app.setup(ui, &missing));
                        }
                        egui::Panel::left("sessions").show(ui, |ui| app.sidebar(ui));
                        egui::CentralPanel::default().show(ui, |ui| app.conversation(ui));
                    });
                }
            }
        }
    }
}
