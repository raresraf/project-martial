/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @a15bd23a-5752-4b72-994d-ef993fcae13e/components/config/opts.rs
 * @brief Global runtime configuration management for the Servo browser engine.
 * 
 * Functional Intent: Defines the schema and lifecycle for Servo's execution 
 * parameters. It aggregates standard web browser flags with deep engine-level 
 * debug toggles, facilitating fine-grained control over layout, rendering, 
 * and script execution behavior. Options are typically initialized from 
 * command-line arguments and made globally accessible via a thread-safe singleton.
 * 
 * Domain: Browser Configuration, Engine Tuning, Runtime Lifecycle.
 */

use std::default::Default;
use std::path::PathBuf;
use std::sync::{LazyLock, RwLock, RwLockReadGuard};

use serde::{Deserialize, Serialize};
use servo_url::ServoUrl;

/**
 * @brief Primary configuration container for a Servo execution instance.
 */
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Opts {
    /// Functional Utility: Synchronizes image capture with engine quiescence (stable state).
    pub wait_for_stable_image: bool,

    /// Functional Utility: Configures temporal performance metric collection.
    pub time_profiling: Option<OutputOptions>,

    pub time_profiler_trace_path: Option<String>,

    /// Block Logic: Layout strategy toggle.
    /// Invariant: If true, disables partial reflows, forcing full tree reconciliation.
    pub nonincremental_layout: bool,

    pub user_stylesheets: Vec<(Vec<u8>, ServoUrl)>,

    /// Functional Utility: Enforces immediate process termination on thread panic.
    pub hard_fail: bool,

    /// Block Logic: Low-level engine instrumentation.
    pub debug: DebugOptions,

    pub webdriver_port: Option<u16>,

    pub multiprocess: bool,

    pub background_hang_monitor: bool,

    pub sandbox: bool,

    /// Block Logic: Fuzzing and resilience testing.
    /// Logic: Simulates constellation instability for hardening verification.
    pub random_pipeline_closure_probability: Option<f32>,

    pub random_pipeline_closure_seed: Option<usize>,

    pub shaders_dir: Option<PathBuf>,

    pub config_dir: Option<PathBuf>,

    pub certificate_path: Option<String>,

    pub ignore_certificate_errors: bool,

    /// Block Logic: Source code auditing and de-obfuscation.
    pub unminify_js: bool,

    pub local_script_source: Option<String>,

    pub unminify_css: bool,

    pub print_pwm: bool,
}

/**
 * @brief Developer-centric debugging and tree visualization toggles.
 */
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DebugOptions {
    pub help: bool,

    /// Block Logic: Visual structure inspection.
    /// Logic: Serializes various internal engine trees to stdout for analysis.
    pub dump_style_tree: bool,
    pub dump_rule_tree: bool,
    pub dump_flow_tree: bool,
    pub dump_stacking_context_tree: bool,
    pub dump_scroll_tree: bool,
    pub dump_display_list: bool,

    pub relayout_event: bool,

    pub profile_script_events: bool,

    /// Block Logic: Deterministic debugging.
    /// Invariant: Enabling layout tracing forces sequential execution to maintain order.
    pub trace_layout: bool,

    pub disable_share_style_cache: bool,

    pub dump_style_statistics: bool,

    pub convert_mouse_to_touch: bool,

    pub gc_profile: bool,

    pub webrender_stats: bool,

    pub signpost: bool,
}

impl DebugOptions {
    /**
     * extend - Incremental configuration updates via string-based flags.
     * @debug_string: Comma-separated list of debug keys (e.g., from -Z).
     * 
     * Block Logic: String-to-feature mapping.
     * Logic: Iteratively parses flag tokens and activates corresponding 
     * boolean toggles in the DebugOptions struct.
     */
    pub fn extend(&mut self, debug_string: String) -> Result<(), String> {
        for option in debug_string.split(',') {
            match option {
                "help" => self.help = true,
                "convert-mouse-to-touch" => self.convert_mouse_to_touch = true,
                "disable-share-style-cache" => self.disable_share_style_cache = true,
                "dump-display-list" => self.dump_display_list = true,
                "dump-stacking-context-tree" => self.dump_stacking_context_tree = true,
                "dump-flow-tree" => self.dump_flow_tree = true,
                "dump-rule-tree" => self.dump_rule_tree = true,
                "dump-style-tree" => self.dump_style_tree = true,
                "dump-scroll-tree" => self.dump_scroll_tree = true,
                "gc-profile" => self.gc_profile = true,
                "profile-script-events" => self.profile_script_events = true,
                "relayout-event" => self.relayout_event = true,
                "signpost" => self.signpost = true,
                "dump-style-stats" => self.dump_style_statistics = true,
                "trace-layout" => self.trace_layout = true,
                "wr-stats" => self.webrender_stats = true,
                "" => {},
                _ => return Err(String::from(option)),
            };
        }

        Ok(())
    }
}

/**
 * @brief Output destinations for profiling data.
 */
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum OutputOptions {
    FileName(String),
    Stdout(f64),
}

impl Default for Opts {
    /**
     * @brief Baseline configuration with safety-first defaults.
     */
    fn default() -> Self {
        Self {
            wait_for_stable_image: false,
            time_profiling: None,
            time_profiler_trace_path: None,
            nonincremental_layout: false,
            user_stylesheets: Vec::new(),
            hard_fail: true,
            webdriver_port: None,
            multiprocess: false,
            background_hang_monitor: false,
            random_pipeline_closure_probability: None,
            random_pipeline_closure_seed: None,
            sandbox: false,
            debug: Default::default(),
            config_dir: None,
            shaders_dir: None,
            certificate_path: None,
            ignore_certificate_errors: false,
            unminify_js: false,
            local_script_source: None,
            unminify_css: false,
            print_pwm: false,
        }
    }
}

/**
 * Functional Utility: Global configuration singleton.
 * Logic: Uses RwLock for safe multi-threaded reads while allowing late-bound 
 * initialization from CLI parsers. Encapsulated in LazyLock for O(1) start-up 
 * overhead before options are needed.
 */
static OPTIONS: LazyLock<RwLock<Opts>> = LazyLock::new(|| RwLock::new(Opts::default()));

/**
 * @brief Commits a new configuration state to the global store.
 */
pub fn set_options(opts: Opts) {
    *OPTIONS.write().unwrap() = opts;
}

/**
 * @brief Retrieves a read-only handle to the active configuration.
 */
#[inline]
pub fn get() -> RwLockReadGuard<'static, Opts> {
    OPTIONS.read().unwrap()
}
