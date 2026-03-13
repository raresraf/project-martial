/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! This module serves as the main application entry point for the desktop `servoshell`.
//! It orchestrates the `winit` event loop, managing the interactions between the
//! Servo engine, the underlying windowing system, and the user interface components.
//! This includes initializing the `Servo` instance, handling `WebView` lifecycles,
//! and processing various events such as keyboard input, mouse wheel scrolls,
//! and UI interactions (e.g., navigation, tab management).

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;
use std::{env, fs};

use log::{info, trace, warn};
use servo::compositing::windowing::{AnimationState, WindowMethods};
use servo::config::opts::Opts;
use servo::config::prefs::Preferences;
use servo::servo_config::pref;
use servo::servo_url::ServoUrl;
use servo::webxr::glwindow::GlWindowDiscovery;
#[cfg(target_os = "windows")]
use servo::webxr::openxr::{AppInfo, OpenXrDiscovery};
use servo::{EventLoopWaker, Servo};
use url::Url;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::window::WindowId;

use super::app_state::AppState;
use super::events_loop::{EventsLoop, WakerEvent};
use super::minibrowser::{Minibrowser, MinibrowserEvent};
use super::{headed_window, headless_window};
use crate::desktop::app_state::RunningAppState;
use crate::desktop::embedder::{EmbedderCallbacks, XrDiscovery};
use crate::desktop::tracing::trace_winit_event;
use crate::desktop::window_trait::WindowPortsMethods;
use crate::parser::{get_default_url, location_bar_input_to_url};
use crate::prefs::ServoShellPreferences;

/// `App` represents the main application structure for the `servoshell` desktop port.
/// It holds the overall application state, configuration options, and manages
/// the interaction with the `winit` event loop and Servo engine.
pub struct App {
    /// Global Servo options, parsed from command-line arguments.
    opts: Opts,
    /// User preferences for Servo's behavior.
    preferences: Preferences,
    /// Preferences specific to the Servo shell.
    servoshell_preferences: ServoShellPreferences,
    /// A `Cell` indicating whether the application is currently suspended.
    suspended: Cell<bool>,
    /// A map of `WindowId` to `WindowPortsMethods`, managing open windows.
    windows: HashMap<WindowId, Rc<dyn WindowPortsMethods>>,
    /// An optional `Minibrowser` instance for UI elements.
    minibrowser: Option<Minibrowser>,
    /// A waker to interact with the event loop.
    waker: Box<dyn EventLoopWaker>,
    /// The initial URL to load when the application starts.
    initial_url: ServoUrl,
    /// The starting time of the application for timing calculations.
    t_start: Instant,
    /// The last recorded time for timing calculations.
    t: Instant,
    /// The current state of the application's lifecycle.
    state: AppState,
}

/// `PumpResult` indicates the outcome of processing events and whether the
/// application's event loop should continue or shut down.
pub(crate) enum PumpResult {
    /// The caller should shut down Servo and its related context.
    Shutdown,
    /// The event loop should continue, with flags indicating whether a UI update
    /// or window redraw is needed.
    Continue {
        need_update: bool,
        need_window_redraw: bool,
    },
}

impl App {
    /// Creates a new `App` instance, initializing its configuration and basic state.
    ///
    /// # Arguments
    /// * `opts` - Global configuration options for Servo.
    /// * `preferences` - User preferences for Servo.
    /// * `servo_shell_preferences` - Preferences specific to the Servo shell.
    /// * `events_loop` - A reference to the `EventsLoop` for creating a waker.
    ///
    /// Pre-condition: All input parameters are valid.
    /// Post-condition: A new `App` instance is returned in the `Initializing` state.
    pub fn new(
        opts: Opts,
        preferences: Preferences,
        servo_shell_preferences: ServoShellPreferences,
        events_loop: &EventsLoop,
    ) -> Self {
        let initial_url = get_default_url(
            servo_shell_preferences.url.as_deref(),
            env::current_dir().unwrap(),
            |path| fs::metadata(path).is_ok(),
            &servo_shell_preferences,
        );

        let t = Instant::now();
        App {
            opts,
            preferences,
            servoshell_preferences: servo_shell_preferences,
            suspended: Cell::new(false),
            windows: HashMap::new(),
            minibrowser: None,
            waker: events_loop.create_event_loop_waker(),
            initial_url: initial_url.clone(),
            t_start: t,
            t,
            state: AppState::Initializing,
        }
    }

    /// Initializes the application once the event loop starts running.
    ///
    /// This method sets up the main window (headed or headless), initializes Servo,
    /// creates the initial `WebView`, and sets up the `Minibrowser` UI if applicable.
    ///
    /// # Arguments
    /// * `event_loop` - An `Option` containing the `ActiveEventLoop` if running in headed mode.
    ///
    /// Pre-condition: The application is in the `Initializing` state.
    /// Post-condition: The application transitions to the `Running` state, with all core
    /// components initialized and ready.
    pub fn init(&mut self, event_loop: Option<&ActiveEventLoop>) {
        let headless = self.servoshell_preferences.headless;

        assert_eq!(headless, event_loop.is_none());
        let window = match event_loop {
            Some(event_loop) => {
                let window = headed_window::Window::new(&self.servoshell_preferences, event_loop);
                self.minibrowser = Some(Minibrowser::new(
                    window.offscreen_rendering_context(),
                    event_loop,
                    self.initial_url.clone(),
                ));
                Rc::new(window)
            },
            None => headless_window::Window::new(&self.servoshell_preferences),
        };

        self.windows.insert(window.id(), window);

        self.suspended.set(false);
        let (_, window) = self.windows.iter().next().unwrap();

        // Block Logic: Initializes WebXR discovery based on preferences and platform.
        let xr_discovery = if pref!(dom_webxr_openxr_enabled) && !headless {
            #[cfg(target_os = "windows")]
            let openxr = {
                let app_info = AppInfo::new("Servoshell", 0, "Servo", 0);
                Some(XrDiscovery::OpenXr(OpenXrDiscovery::new(None, app_info)))
            };
            #[cfg(not(target_os = "windows"))]
            let openxr = None;

            openxr
        } else if pref!(dom_webxr_glwindow_enabled) && !headless {
            let window = window.new_glwindow(event_loop.unwrap());
            Some(XrDiscovery::GlWindow(GlWindowDiscovery::new(window)))
        } else {
            None
        };

        // Implements embedder methods, used by libservo and constellation.
        let embedder = Box::new(EmbedderCallbacks::new(self.waker.clone(), xr_discovery));

        // TODO: Remove this once dyn upcasting coercion stabilises
        // <https://github.com/rust-lang/rust/issues/65991>
        /// Helper struct for upcasting `WindowPortsMethods` to `WindowMethods`.
        struct UpcastedWindow(Rc<dyn WindowPortsMethods>);
        impl WindowMethods for UpcastedWindow {
            /// Retrieves embedder coordinates from the underlying window.
            fn get_coordinates(&self) -> servo::compositing::windowing::EmbedderCoordinates {
                self.0.get_coordinates()
            }
            /// Sets the animation state of the underlying window.
            fn set_animation_state(&self, state: AnimationState) {
                self.0.set_animation_state(state);
            }
        }

        let servo = Servo::new(
            self.opts.clone(),
            self.preferences.clone(),
            window.rendering_context(),
            embedder,
            Rc::new(UpcastedWindow(window.clone())),
            self.servoshell_preferences.user_agent.clone(),
        );
        servo.setup_logging();

        let running_state = Rc::new(RunningAppState::new(
            servo,
            window.clone(),
            self.servoshell_preferences.clone(),
        ));
        running_state.new_toplevel_webview(self.initial_url.clone().into_url());

        // Block Logic: Updates the minibrowser UI if it exists.
        if let Some(ref mut minibrowser) = self.minibrowser {
            minibrowser.update(window.winit_window().unwrap(), &running_state, "init");
            window.set_toolbar_height(minibrowser.toolbar_height);
        }

        self.state = AppState::Running(running_state);
    }

    /// Checks if any window is currently animating.
    ///
    /// Post-condition: Returns `true` if any managed window reports an animating state, `false` otherwise.
    pub fn is_animating(&self) -> bool {
        self.windows.iter().any(|(_, window)| window.is_animating())
    }

    /// Handles events in headed mode with `winit`.
    ///
    /// # Arguments
    /// * `event_loop` - The active `winit` `ActiveEventLoop`.
    /// * `window` - The `WindowPortsMethods` instance for the active window.
    ///
    /// Pre-condition: The application is in the `Running` state.
    /// Post-condition: Processes Servo events, updates the UI, and manages `winit`'s control flow.
    pub fn handle_events_with_winit(
        &mut self,
        event_loop: &ActiveEventLoop,
        window: Rc<dyn WindowPortsMethods>,
    ) {
        let AppState::Running(state) = &self.state else {
            return;
        };

        match state.pump_event_loop() {
            PumpResult::Shutdown => {
                state.shutdown();
                self.state = AppState::ShuttingDown;
            },
            PumpResult::Continue {
                need_update: update,
                need_window_redraw,
            } => {
                // Block Logic: Updates the minibrowser UI if needed.
                let updated = match (update, &mut self.minibrowser) {
                    (true, Some(minibrowser)) => minibrowser.update_webview_data(state),
                    _ => false,
                };

                // If in headed mode, request a winit redraw event, so we can paint the minibrowser.
                if updated || need_window_redraw {
                    if let Some(window) = window.winit_window() {
                        window.request_redraw();
                    }
                }
            },
        }

        // Block Logic: If the application is shutting down, exit the event loop.
        if matches!(self.state, AppState::ShuttingDown) {
            event_loop.exit();
        }
    }

    /// Handles all Servo events in headless mode.
    ///
    /// Post-condition: Processes Servo events, repaints the Servo view if necessary, and
    /// returns `true` if the application should continue running, `false` otherwise.
    pub fn handle_events_with_headless(&mut self) -> bool {
        let now = Instant::now();
        let event = winit::event::Event::UserEvent(WakerEvent);
        trace_winit_event!(
            event,
            "@{:?} (+{:?}) {event:?}",
            now - self.t_start,
            now - self.t
        );
        self.t = now;

        // We should always be in the running state.
        let AppState::Running(state) = &self.state else {
            return false;
        };

        match state.pump_event_loop() {
            PumpResult::Shutdown => {
                state.shutdown();
                self.state = AppState::ShuttingDown;
            },
            PumpResult::Continue { .. } => state.repaint_servo_if_necessary(),
        }

        !matches!(self.state, AppState::ShuttingDown)
    }

    /// Takes any events generated during `egui` updates (from the minibrowser UI) and
    /// performs their corresponding actions.
    ///
    /// Pre-condition: The minibrowser exists and the application is in the `Running` state.
    /// Post-condition: UI events (e.g., Go, Back, Forward, Reload, NewWebView, CloseWebView)
    /// are processed, affecting the `WebView`s.
    fn handle_servoshell_ui_events(&mut self) {
        let Some(minibrowser) = self.minibrowser.as_ref() else {
            return;
        };
        // We should always be in the running state.
        let AppState::Running(state) = &self.state else {
            return;
        };

        // Block Logic: Iterates through events from the minibrowser and dispatches actions.
        for event in minibrowser.take_events() {
            match event {
                MinibrowserEvent::Go(location) => {
                    minibrowser.update_location_dirty(false);
                    let Some(url) = location_bar_input_to_url(
                        &location.clone(),
                        &self.servoshell_preferences.searchpage,
                    ) else {
                        warn!("failed to parse location");
                        break;
                    };
                    if let Some(focused_webview) = state.focused_webview() {
                        focused_webview.load(url.into_url());
                    }
                },
                MinibrowserEvent::Back => {
                    if let Some(focused_webview) = state.focused_webview() {
                        focused_webview.go_back(1);
                    }
                },
                MinibrowserEvent::Forward => {
                    if let Some(focused_webview) = state.focused_webview() {
                        focused_webview.go_forward(1);
                    }
                },
                MinibrowserEvent::Reload => {
                    minibrowser.update_location_dirty(false);
                    if let Some(focused_webview) = state.focused_webview() {
                        focused_webview.reload();
                    }
                },
                MinibrowserEvent::NewWebView => {
                    minibrowser.update_location_dirty(false);
                    state.new_toplevel_webview(Url::parse("servo:newtab").unwrap());
                },
                MinibrowserEvent::CloseWebView(id) => {
                    minibrowser.update_location_dirty(false);
                    state.close_webview(id);
                },
            }
        }
    }
}

/// `App` implements `winit`'s `ApplicationHandler` trait to manage the application's
/// lifecycle and respond to various windowing and user events.
impl ApplicationHandler<WakerEvent> for App {
    /// Handles the `resumed` event, performing application initialization.
    ///
    /// # Arguments
    /// * `event_loop` - The active `winit` `ActiveEventLoop`.
    ///
    /// Post-condition: Calls `init` to set up the application.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.init(Some(event_loop));
    }

    /// Handles `winit` window events.
    ///
    /// # Arguments
    /// * `event_loop` - The active `winit` `ActiveEventLoop`.
    /// * `window_id` - The `WindowId` of the window that generated the event.
    /// * `event` - The `WindowEvent` to process.
    ///
    /// Pre-condition: The application is in the `Running` state and the `window_id` is valid.
    /// Post-condition: Processes redraw requests, scale factor changes, and other window events,
    /// updating the minibrowser UI and managing `winit`'s control flow.
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let now = Instant::now();
        // Block Logic: Traces winit events for debugging and performance analysis.
        trace_winit_event!(
            event,
            "@{:?} (+{:?}) {event:?}",
            now - self.t_start,
            now - self.t
        );
        self.t = now;

        let AppState::Running(state) = &self.state else {
            return;
        };

        let Some(window) = self.windows.get(&window_id) else {
            return;
        };

        let window = window.clone();
        // Block Logic: Handles `RedrawRequested` events, updating and painting the minibrowser.
        if event == WindowEvent::RedrawRequested {
            // We need to redraw the window for some reason.
            trace!("RedrawRequested");

            // WARNING: do not defer painting or presenting to some later tick of the event
            // loop or servoshell may become unresponsive! (servo#30312)
            if let Some(ref mut minibrowser) = self.minibrowser {
                minibrowser.update(window.winit_window().unwrap(), state, "RedrawRequested");
                minibrowser.paint(window.winit_window().unwrap());
            }
        }

        // Handle the event
        let mut consumed = false;
        // Block Logic: Delegates window events to the minibrowser for UI interaction.
        if let Some(ref mut minibrowser) = self.minibrowser {
            match event {
                WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    // Intercept any ScaleFactorChanged events away from EguiGlow::on_window_event, so
                    // we can use our own logic for calculating the scale factor and set egui’s
                    // scale factor to that value manually.
                    let desired_scale_factor = window.hidpi_factor().get();
                    let effective_egui_zoom_factor = desired_scale_factor / scale_factor as f32;

                    info!(
                        "window scale factor changed to {}, setting egui zoom factor to {}",
                        scale_factor, effective_egui_zoom_factor
                    );

                    minibrowser
                        .context
                        .egui_ctx
                        .set_zoom_factor(effective_egui_zoom_factor);

                    // Request a winit redraw event, so we can recomposite, update and paint
                    // the minibrowser, and present the new frame.
                    window.winit_window().unwrap().request_redraw();
                },
                ref event => {
                    let response =
                        minibrowser.on_window_event(window.winit_window().unwrap(), event);
                    // Update minibrowser if there's resize event to sync up with window.
                    if let WindowEvent::Resized(_) = event {
                        minibrowser.update(
                            window.winit_window().unwrap(),
                            state,
                            "Sync WebView size with Window Resize event",
                        );
                    }
                    if response.repaint && *event != WindowEvent::RedrawRequested {
                        // Request a winit redraw event, so we can recomposite, update and paint
                        // the minibrowser, and present the new frame.
                        window.winit_window().unwrap().request_redraw();
                    }

                    // TODO how do we handle the tab key? (see doc for consumed)
                    // Note that servo doesn’t yet support tabbing through links and inputs
                    consumed = response.consumed;
                },
            }
        }
        // Block Logic: If the event was not consumed by the minibrowser, delegate it to the window.
        if !consumed {
            window.handle_winit_event(state.clone(), event);
        }

        let animating = self.is_animating();

        // Block Logic: Controls the winit event loop's polling behavior based on animation state.
        if !animating || self.suspended.get() {
            event_loop.set_control_flow(ControlFlow::Wait);
        } else {
            event_loop.set_control_flow(ControlFlow::Poll);
        }

        // Consume and handle any events from the servoshell UI.
        self.handle_servoshell_ui_events();

        self.handle_events_with_winit(event_loop, window);
    }

    /// Handles user-defined `WakerEvent`s.
    ///
    /// # Arguments
    /// * `event_loop` - The active `winit` `ActiveEventLoop`.
    /// * `event` - The `WakerEvent` to process.
    ///
    /// Pre-condition: The application is in the `Running` state.
    /// Post-condition: Spins the Servo event loop, processes UI events, and manages `winit`'s control flow.
    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: WakerEvent) {
        let now = Instant::now();
        // Block Logic: Traces winit events for debugging.
        let event = winit::event::Event::UserEvent(event);
        trace_winit_event!(
            event,
            "@{:?} (+{:?}) {event:?}",
            now - self.t_start,
            now - self.t
        );
        self.t = now;

        // Block Logic: Ensures the application is in the `Running` state before processing.
        if !matches!(self.state, AppState::Running(_)) {
            return;
        };
        let Some(window) = self.windows.values().next() else {
            return;
        };
        let window = window.clone();

        let animating = self.is_animating();

        // Block Logic: Controls `winit` event loop polling behavior.
        if !animating || self.suspended.get() {
            event_loop.set_control_flow(ControlFlow::Wait);
        } else {
            event_loop.set_control_flow(ControlFlow::Poll);
        }

        // Consume and handle any events from the Minibrowser.
        self.handle_servoshell_ui_events();

        self.handle_events_with_winit(event_loop, window);
    }

    /// Handles the `suspended` event, indicating the application is in the background.
    ///
    /// # Arguments
    /// * `_` - The active `winit` `ActiveEventLoop` (unused).
    ///
    /// Post-condition: The internal `suspended` flag is set to `true`.
    fn suspended(&mut self, _: &ActiveEventLoop) {
        self.suspended.set(true);
    }
}
