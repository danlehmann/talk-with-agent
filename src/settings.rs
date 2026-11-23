//! User preferences, kept in `settings.json` in the data folder.

use crate::state::data_dir;
use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Id of the chosen speaking voice; None = the backend's default.
    pub voice: Option<String>,
}

impl Settings {
    pub fn load() -> Self {
        std::fs::read_to_string(data_dir().join("settings.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(data_dir().join("settings.json"), text);
        }
    }
}
