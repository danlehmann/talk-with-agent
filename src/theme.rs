//! Look and feel: the system UI font and a calmer, flatter style than egui's
//! defaults, in both light and dark.

use eframe::egui::{self, Color32, CornerRadius, FontData, FontFamily, FontId, Stroke, TextStyle, Theme};
use std::sync::Arc;

/// (proportional, monospace) font files of the platform's UI typeface.
fn system_fonts() -> (&'static [&'static str], &'static [&'static str]) {
    if cfg!(target_os = "macos") {
        (&["/System/Library/Fonts/SFNS.ttf"], &["/System/Library/Fonts/SFNSMono.ttf"])
    } else if cfg!(target_os = "windows") {
        (&["C:\\Windows\\Fonts\\segoeui.ttf"], &["C:\\Windows\\Fonts\\consola.ttf"])
    } else {
        (&[], &[])
    }
}

/// The platform fonts first, egui's bundled ones as fallback (and for symbols).
pub fn fonts() -> egui::FontDefinitions {
    let mut fonts = egui::FontDefinitions::default();
    let (proportional, monospace) = system_fonts();
    for (family, paths) in [(FontFamily::Proportional, proportional), (FontFamily::Monospace, monospace)] {
        if let Some(bytes) = paths.iter().find_map(|path| std::fs::read(path).ok()) {
            let name = format!("system-{family}");
            fonts.font_data.insert(name.clone(), Arc::new(FontData::from_owned(bytes)));
            fonts.families.entry(family).or_default().insert(0, name);
        }
    }
    fonts
}

pub fn apply(ctx: &egui::Context) {
    ctx.set_fonts(fonts());
    ctx.all_styles_mut(|style| {
        style.text_styles = [
            (TextStyle::Small, FontId::proportional(11.0)),
            (TextStyle::Body, FontId::proportional(13.5)),
            (TextStyle::Button, FontId::proportional(13.5)),
            (TextStyle::Heading, FontId::proportional(18.0)),
            (TextStyle::Monospace, FontId::monospace(12.0)),
        ]
        .into();
        let spacing = &mut style.spacing;
        spacing.item_spacing = egui::vec2(8.0, 7.0);
        spacing.button_padding = egui::vec2(10.0, 4.0);
        spacing.interact_size.y = 26.0;
        spacing.scroll = egui::style::ScrollStyle::floating();

        let visuals = &mut style.visuals;
        let radius = CornerRadius::same(6);
        for widget in [
            &mut visuals.widgets.noninteractive,
            &mut visuals.widgets.inactive,
            &mut visuals.widgets.hovered,
            &mut visuals.widgets.active,
            &mut visuals.widgets.open,
        ] {
            widget.corner_radius = radius;
            widget.expansion = 0.0;
        }
        visuals.menu_corner_radius = CornerRadius::same(8);
        visuals.window_corner_radius = CornerRadius::same(10);
        visuals.selection.stroke = Stroke::new(1.0, Color32::WHITE);
    });
    ctx.style_mut_of(Theme::Dark, |style| {
        palette(&mut style.visuals, &Palette {
            panel: Color32::from_gray(30),
            raised: Color32::from_gray(42),
            sunken: Color32::from_gray(22),
            control: [Color32::from_gray(58), Color32::from_gray(72), Color32::from_gray(88)],
            border: Color32::from_gray(58),
            control_border: Color32::TRANSPARENT,
            text: Color32::from_gray(232),
            weak_text: Color32::from_gray(150),
            accent: Color32::from_rgb(10, 132, 255),
        });
    });
    ctx.style_mut_of(Theme::Light, |style| {
        palette(&mut style.visuals, &Palette {
            panel: Color32::from_gray(246),
            raised: Color32::WHITE,
            sunken: Color32::WHITE,
            control: [Color32::WHITE, Color32::from_gray(240), Color32::from_gray(226)],
            border: Color32::from_gray(214),
            control_border: Color32::from_gray(204),
            text: Color32::from_gray(28),
            weak_text: Color32::from_gray(120),
            accent: Color32::from_rgb(0, 122, 255),
        });
    });
}

/// The session list's background: a shade off the main panel, like a sidebar.
pub fn sidebar_fill(visuals: &egui::Visuals) -> Color32 {
    if visuals.dark_mode { Color32::from_gray(38) } else { Color32::from_gray(236) }
}

struct Palette {
    panel: Color32,
    /// Menus and popups.
    raised: Color32,
    /// Text boxes.
    sunken: Color32,
    /// Buttons: at rest, hovered, pressed.
    control: [Color32; 3],
    /// Separators and panel edges.
    border: Color32,
    control_border: Color32,
    text: Color32,
    weak_text: Color32,
    accent: Color32,
}

fn palette(visuals: &mut egui::Visuals, p: &Palette) {
    visuals.panel_fill = p.panel;
    visuals.window_fill = p.raised;
    visuals.window_stroke = Stroke::new(1.0, p.border);
    visuals.extreme_bg_color = p.sunken;
    visuals.faint_bg_color = p.control[0];
    visuals.hyperlink_color = p.accent;
    visuals.selection.bg_fill = p.accent;
    visuals.weak_text_color = Some(p.weak_text);

    let widgets = &mut visuals.widgets;
    widgets.noninteractive.bg_fill = p.panel;
    widgets.noninteractive.weak_bg_fill = p.panel;
    widgets.noninteractive.bg_stroke = Stroke::new(1.0, p.border);
    widgets.noninteractive.fg_stroke = Stroke::new(1.0, p.text);
    for (widget, fill) in [
        (&mut widgets.inactive, p.control[0]),
        (&mut widgets.hovered, p.control[1]),
        (&mut widgets.active, p.control[2]),
        (&mut widgets.open, p.control[1]),
    ] {
        widget.bg_fill = fill;
        widget.weak_bg_fill = fill;
        widget.bg_stroke = Stroke::new(1.0, p.control_border);
        widget.fg_stroke = Stroke::new(1.0, p.text);
    }
}

#[cfg(test)]
mod tests {
    use eframe::egui;

    /// The system font files load and lay out text (they are variable fonts).
    #[test]
    fn system_fonts_render() {
        let ctx = egui::Context::default();
        super::apply(&ctx);
        let mut width = 0.0;
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            width = ui.label("Talk with Agent × 123").rect.width();
            ui.monospace("http://127.0.0.1");
        });
        assert!(width > 50.0, "text laid out {width} wide");
    }
}
