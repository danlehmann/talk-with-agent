mod app;
mod server;
mod settings;
mod state;
mod voice;

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
            // For the translucent side-panel overlay. (No shadow: macOS draws
            // stale shadows around transparent windows.)
            .with_transparent(true)
            .with_has_shadow(false),
        ..Default::default()
    };
    eframe::run_native(
        "Talk with Agent",
        options,
        Box::new(move |cc| {
            let ctx = cc.egui_ctx.clone();
            store.set_waker(move || ctx.request_repaint());
            let ctx = cc.egui_ctx.clone();
            let voice = voice::create(Arc::new(move || ctx.request_repaint()));
            Ok(Box::new(app::App::new(store, voice, server)))
        }),
    )
}
