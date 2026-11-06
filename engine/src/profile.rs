use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoiConfig {
    pub x: i32,
    pub y: i32,
    #[serde(alias = "w")]
    pub width: u32,
    #[serde(alias = "h")]
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardConfig {
    pub rows: usize,
    pub cols: usize,
    pub roi: RoiConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatcherProfile {
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    #[serde(default = "default_empty_variance")]
    pub empty_variance_threshold: f32,
    #[serde(default = "default_template_dir")]
    pub template_dir: String,
}

fn default_threshold() -> f32 {
    0.75
}
fn default_empty_variance() -> f32 {
    15.0
}
fn default_template_dir() -> String {
    "templates".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingConfig {
    #[serde(default = "default_click_interval")]
    pub click_interval_ms: u64,
    #[serde(default = "default_animation_delay")]
    pub animation_delay_ms: u64,
    #[serde(default = "default_verify_timeout")]
    pub verify_timeout_ms: u64,
}

fn default_click_interval() -> u64 {
    35
}
fn default_animation_delay() -> u64 {
    120
}
fn default_verify_timeout() -> u64 {
    600
}

/// Optional game rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulesConfig {
    /// true  = a connecting line may run outside the board (around the outer ring).
    /// false = paths must stay inside the board (the outer ring is blocked).
    #[serde(default = "default_allow_outer_path")]
    pub allow_outer_path: bool,
    /// When no move exists, retry once treating obstacle blocks as passable.
    #[serde(default = "default_ignore_obstacles_when_stuck")]
    pub ignore_obstacles_when_stuck: bool,
}

fn default_ignore_obstacles_when_stuck() -> bool {
    true
}

fn default_allow_outer_path() -> bool {
    true
}

impl Default for RulesConfig {
    fn default() -> Self {
        Self {
            allow_outer_path: default_allow_outer_path(),
            ignore_obstacles_when_stuck: default_ignore_obstacles_when_stuck(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameProfile {
    pub name: String,
    pub board: BoardConfig,
    #[serde(default)]
    pub matcher: MatcherProfile,
    #[serde(default)]
    pub timing: TimingConfig,
    #[serde(default)]
    pub rules: RulesConfig,
}

impl Default for MatcherProfile {
    fn default() -> Self {
        Self {
            threshold: default_threshold(),
            empty_variance_threshold: default_empty_variance(),
            template_dir: default_template_dir(),
        }
    }
}

impl Default for TimingConfig {
    fn default() -> Self {
        Self {
            click_interval_ms: default_click_interval(),
            animation_delay_ms: default_animation_delay(),
            verify_timeout_ms: default_verify_timeout(),
        }
    }
}

impl GameProfile {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read game profile at {:?}", path.as_ref()))?;
        let profile: GameProfile = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse game profile JSON from {:?}", path.as_ref()))?;
        Ok(profile)
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path.as_ref(), json)?;
        Ok(())
    }
}
