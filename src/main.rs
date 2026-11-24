mod app;
mod server;
mod settings;
mod state;
mod theme;
mod voice;
mod window;

use eframe::egui;
use std::sync::Arc;

fn main() -> eframe::Result {
    let store = Arc::new(state::Store::load());
    let server = server::start(store.clone());

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Talk with Agent")
            .with_inner_size([900.0, 640.0])
            .with_min_inner_size([280.0, 300.0])
            // For the translucent side-panel overlay.
            .with_transparent(true),
        ..Default::default()
    };
    eframe::run_native(
        "Talk with Agent",
        options,
        Box::new(move |cc| {
            theme::apply(&cc.egui_ctx);
            let ctx = cc.egui_ctx.clone();
            store.set_waker(move || ctx.request_repaint());
            let ctx = cc.egui_ctx.clone();
            let settings = settings::Settings::load();
            let voice = voice::create(
                Arc::new(move || ctx.request_repaint()),
                settings.voice.as_deref(),
                settings.transcriber.as_deref(),
                settings.downloads_allowed,
            );
            Ok(Box::new(app::App::new(store, voice, server)))
        }),
    )
}
