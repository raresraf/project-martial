/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! This module defines the core application state for `servoshell`'s desktop port.
//! It encapsulates the main components and logic required to manage `WebView`
//! instances, handle various application events, process user input, and
//! coordinate with Servo's internal engine components like the compositor and
//! constellation. The `AppState` transitions through `Initializing`, `Running`,
//! and `ShuttingDown` phases, ensuring a structured lifecycle for the application.

use std::cell::{Ref, RefCell, RefMut};
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::thread;

use euclid::Vector2D;
use image::DynamicImage;
use keyboard_types::{Key, KeyboardEvent, Modifiers, ShortcutMatcher};
use log::{error, info};
use servo::base::id::WebViewId;
use servo::config::pref;
use servo::ipc_channel::ipc::IpcSender;
use servo::webrender_api::units::{DeviceIntPoint, DeviceIntSize};
use servo::webrender_api::ScrollLocation;
use servo::{
    AllowOrDenyRequest, AuthenticationRequest, FilterPattern, GamepadHapticEffectType, LoadStatus,
    PermissionRequest, PromptDefinition, PromptOrigin, PromptResult, Servo, ServoDelegate,
    ServoError, TouchEventType, WebView, WebViewDelegate,
};
#[cfg(target_os = "linux")]
use tinyfiledialogs::MessageBoxIcon;
use url::Url;

use super::app::PumpResult;
use super::dialog::Dialog;
use super::gamepad::GamepadSupport;
use super::keyutils::CMD_OR_CONTROL;
use super::window_trait::{WindowPortsMethods, LINE_HEIGHT};
use crate::prefs::ServoShellPreferences;

/// `AppState` represents the overall state of the Servo application, managing
/// its lifecycle through different phases.
pub(crate) enum AppState {
    /// The application is in the process of being initialized.
    Initializing,
    /// The application is fully initialized and running, holding a reference to `RunningAppState`.
    Running(Rc<RunningAppState>),
    /// The application is in the process of shutting down.
    ShuttingDown,
}

/// `RunningAppState` holds the core operational state of the Servo application once it's initialized.
/// It contains the Servo engine instance, application preferences, and a mutable inner state.
pub(crate) struct RunningAppState {
    /// A handle to the Servo instance of the [`RunningAppState`]. This is not stored inside
    /// `inner` so that we can keep a reference to Servo in order to spin the event loop,
    /// which will in turn call delegates doing a mutable borrow on `inner`.
    servo: Servo,
    /// The preferences for this run of servoshell. This is not mutable, so doesn't need to
    /// be stored inside the [`RunningAppStateInner`].
    servoshell_preferences: ServoShellPreferences,
    /// A mutable cell holding the internal, mutable state of the running application.
    inner: RefCell<RunningAppStateInner>,
}

/// `RunningAppStateInner` contains the mutable internal state of the `servoshell` desktop application.
pub struct RunningAppStateInner {
    /// List of top-level browsing contexts.
    /// Modified by EmbedderMsg::WebViewOpened and EmbedderMsg::WebViewClosed,
    /// and we exit if it ever becomes empty.
    webviews: HashMap<WebViewId, WebView>,

    /// The order in which the webviews were created.
    creation_order: Vec<WebViewId>,

    /// The webview that is currently focused.
    /// Modified by EmbedderMsg::WebViewFocused and EmbedderMsg::WebViewBlurred.
    focused_webview_id: Option<WebViewId>,

    /// The current set of open dialogs.
    dialogs: HashMap<WebViewId, Vec<Dialog>>,

    /// A handle to the Window that Servo is rendering in -- either headed or headless.
    window: Rc<dyn WindowPortsMethods>,

    /// Gamepad support, which may be `None` if it failed to initialize.
    gamepad_support: Option<GamepadSupport>,

    /// Whether or not the application interface needs to be updated.
    need_update: bool,

    /// Whether or not Servo needs to repaint its display. Currently this is global
    /// because every `WebView` shares a `RenderingContext`.
    need_repaint: bool,
}

impl Drop for RunningAppState {
    /// Deinitializes the Servo instance when `RunningAppState` is dropped.
    ///
    /// Post-condition: `servo.deinit()` is called to release resources.
    fn drop(&mut self) {
        self.servo.deinit();
    }
}

impl RunningAppState {
    /// Creates a new `RunningAppState` instance.
    ///
    /// # Arguments
    /// * `servo` - The `Servo` instance to manage.
    /// * `window` - A reference to the windowing system methods.
    /// * `servoshell_preferences` - The preferences for the Servo shell.
    ///
    /// Post-condition: A new `RunningAppState` is returned, with the `Servo` delegate set
    /// and internal state initialized.
    pub fn new(
        servo: Servo,
        window: Rc<dyn WindowPortsMethods>,
        servoshell_preferences: ServoShellPreferences,
    ) -> RunningAppState {
        servo.set_delegate(Rc::new(ServoShellServoDelegate));
        RunningAppState {
            servo,
            servoshell_preferences,
            inner: RefCell::new(RunningAppStateInner {
                webviews: HashMap::default(),
                creation_order: Default::default(),
                focused_webview_id: None,
                dialogs: Default::default(),
                window,
                gamepad_support: GamepadSupport::maybe_new(),
                need_update: false,
                need_repaint: false,
            }),
        }
    }

    /// Creates and loads a new top-level `WebView` with the given URL.
    ///
    /// # Arguments
    /// * `url` - The `Url` to load in the new webview.
    ///
    /// Pre-condition: Servo is running and not shutting down.
    /// Post-condition: A new `WebView` is created, its delegate set, and added to the application's state.
    pub(crate) fn new_toplevel_webview(self: &Rc<Self>, url: Url) {
        let webview = self.servo().new_webview(url);
        webview.set_delegate(self.clone());
        self.add(webview);
    }

    /// Returns an immutable reference to the internal `RunningAppStateInner`.
    ///
    /// Post-condition: A `Ref` to `RunningAppStateInner` is returned, allowing read-only access.
    pub(crate) fn inner(&self) -> Ref<RunningAppStateInner> {
        self.inner.borrow()
    }

    /// Returns a mutable reference to the internal `RunningAppStateInner`.
    ///
    /// Post-condition: A `RefMut` to `RunningAppStateInner` is returned, allowing mutable access.
    pub(crate) fn inner_mut(&self) -> RefMut<RunningAppStateInner> {
        self.inner.borrow_mut()
    }

    /// Returns an immutable reference to the `Servo` instance.
    ///
    /// Post-condition: An immutable reference to the `Servo` engine is returned.
    pub(crate) fn servo(&self) -> &Servo {
        &self.servo
    }

    /// Saves the current rendered output of the active `WebView` to an image file if a path is specified in preferences.
    ///
    /// Pre-condition: `servoshell_preferences.output_image_path` is `Some`.
    /// Post-condition: The rendered output is read from the `RenderingContext` and saved to the specified path.
    /// Errors are logged if reading or saving fails.
    pub(crate) fn save_output_image_if_necessary(&self) {
        let Some(output_path) = self.servoshell_preferences.output_image_path.as_ref() else {
            return;
        };

        let inner = self.inner();
        let viewport_rect = inner
            .window
            .get_coordinates()
            .viewport
            .to_rect()
            .to_untyped()
            .to_u32();
        let Some(image) = inner
            .window
            .rendering_context()
            .read_to_image(viewport_rect)
        else {
            error!("Failed to read output image.");
            return;
        };

        if let Err(error) = DynamicImage::ImageRgba8(image).save(output_path) {
            error!("Failed to save {output_path}: {error}.");
        }
    }

    /// Repaints the Servo view if necessary, returning `true` if anything was actually
    /// painted or `false` otherwise. Something may not be painted if Servo is waiting
    /// for a stable image to paint.
    ///
    /// Pre-condition: `inner().need_repaint` is true.
    /// Post-condition: The active `WebView`'s content is painted, output image is saved (if configured),
    /// and the `RenderingContext` is presented. If `exit_after_stable_image` is true, Servo starts shutting down.
    pub(crate) fn repaint_servo_if_necessary(&self) {
        if !self.inner().need_repaint {
            return;
        }
        let Some(webview) = self.focused_webview() else {
            return;
        };
        if !webview.paint() {
            return;
        }

        // This needs to be done before presenting(), because `ReneringContext::read_to_image` reads
        // from the back buffer.
        self.save_output_image_if_necessary();

        let mut inner_mut = self.inner_mut();
        inner_mut.window.rendering_context().present();
        inner_mut.need_repaint = false;

        if self.servoshell_preferences.exit_after_stable_image {
            self.servo().start_shutting_down();
        }
    }

    /// Spins the internal application event loop.
    ///
    /// - Notifies Servo about incoming gamepad events
    /// - Spin the Servo event loop, which will run the compositor and trigger delegate methods.
    ///
    /// Post-condition: Gamepad events are handled, the Servo event loop is spun, and
    /// a `PumpResult` is returned indicating whether to continue or shut down.
    pub(crate) fn pump_event_loop(&self) -> PumpResult {
        if pref!(dom_gamepad_enabled) {
            self.handle_gamepad_events();
        }

        if !self.servo().spin_event_loop() {
            return PumpResult::Shutdown;
        }

        // Delegate handlers may have asked us to present or update compositor contents.
        // Currently, egui-file-dialog dialogs need to be constantly redrawn or animations aren't fluid.
        let need_window_redraw = self.inner().need_repaint || self.has_active_dialog();
        let need_update = std::mem::replace(&mut self.inner_mut().need_update, false);

        PumpResult::Continue {
            need_update,
            need_window_redraw,
        }
    }

    /// Adds a `WebView` to the application's managed list.
    ///
    /// # Arguments
    /// * `webview` - The `WebView` to add.
    ///
    /// Post-condition: The webview's ID is added to `creation_order`, and the webview itself
    /// is inserted into `webviews` map.
    pub(crate) fn add(&self, webview: WebView) {
        self.inner_mut().creation_order.push(webview.id());
        self.inner_mut().webviews.insert(webview.id(), webview);
    }

    /// Initiates the shutdown of all managed `WebView`s.
    ///
    /// Post-condition: All `WebView` instances are cleared from the application's state.
    pub(crate) fn shutdown(&self) {
        self.inner_mut().webviews.clear();
    }

    /// Iterates over all active dialogs for the focused webview, applying a callback.
    ///
    /// # Arguments
    /// * `callback` - A closure that takes a mutable reference to a `Dialog` and returns a boolean.
    ///   If the callback returns `false`, the dialog is removed.
    ///
    /// Pre-condition: There is at least one active `WebView`.
    /// Post-condition: The callback is applied to each dialog, and dialogs returning `false` from the callback are removed.
    pub(crate) fn for_each_active_dialog(&self, callback: impl Fn(&mut Dialog) -> bool) {
        let last_created_webview_id = self.inner().creation_order.last().cloned();
        let Some(webview_id) = self
            .focused_webview()
            .as_ref()
            .map(WebView::id)
            .or(last_created_webview_id)
        else {
            return;
        };

        if let Some(dialogs) = self.inner_mut().dialogs.get_mut(&webview_id) {
            dialogs.retain_mut(callback);
        }
    }

    /// Closes a `WebView` identified by its `WebViewId`.
    ///
    /// # Arguments
    /// * `webview_id` - The `WebViewId` of the webview to close.
    ///
    /// Post-condition: The specified webview is removed from `webviews` and `creation_order`,
    /// its dialogs are removed, and focus is shifted to the next most recently created webview,
    /// or Servo starts shutting down if no webviews remain.
    pub fn close_webview(&self, webview_id: WebViewId) {
        // This can happen because we can trigger a close with a UI action and then get the
        // close event from Servo later.
        let mut inner = self.inner_mut();
        if !inner.webviews.contains_key(&webview_id) {
            return;
        }

        inner.webviews.retain(|&id, _| id != webview_id);
        inner.creation_order.retain(|&id| id != webview_id);
        inner.focused_webview_id = None;
        inner.dialogs.remove(&webview_id);

        let last_created = inner
            .creation_order
            .last()
            .and_then(|id| inner.webviews.get(id));

        // Block Logic: If there is a last created webview, focus it, otherwise initiate Servo shutdown.
        match last_created {
            Some(last_created_webview) => last_created_webview.focus(),
            None => self.servo.start_shutting_down(),
        }
    }

    /// Returns the currently focused `WebView`.
    ///
    /// Post-condition: An `Option<WebView>` is returned, which is `Some` if a webview is focused,
    /// and `None` otherwise.
    pub fn focused_webview(&self) -> Option<WebView> {
        self.inner()
            .focused_webview_id
            .and_then(|id| self.inner().webviews.get(&id).cloned())
    }

    /// Returns a vector of all managed `WebView`s in their creation order.
    ///
    /// Post-condition: A `Vec<(WebViewId, WebView)>` is returned, sorted by creation time.
    pub fn webviews(&self) -> Vec<(WebViewId, WebView)> {
        let inner = self.inner();
        inner
            .creation_order
            .iter()
            .map(|id| (*id, inner.webviews.get(id).unwrap().clone()))
            .collect()
    }

    /// Handles incoming gamepad events, dispatching them to the active webview.
    ///
    /// Pre-condition: `dom_gamepad_enabled` preference is true.
    /// Post-condition: Gamepad events are processed by the `GamepadSupport` instance for the focused webview.
    pub fn handle_gamepad_events(&self) {
        let Some(active_webview) = self.focused_webview() else {
            return;
        };
        if let Some(gamepad_support) = self.inner_mut().gamepad_support.as_mut() {
            gamepad_support.handle_gamepad_events(active_webview);
        }
    }

    /// Focuses a `WebView` by its index in the creation order.
    ///
    /// # Arguments
    /// * `index` - The zero-based index of the webview to focus.
    ///
    /// Pre-condition: `index` is a valid index within the range of managed webviews.
    /// Post-condition: The specified webview is focused.
    pub(crate) fn focus_webview_by_index(&self, index: usize) {
        if let Some((_, webview)) = self.webviews().get(index) {
            webview.focus();
        }
    }

    /// Adds a `Dialog` to the specified `WebView`.
    ///
    /// # Arguments
    /// * `webview` - The `WebView` to which the dialog belongs.
    /// * `dialog` - The `Dialog` instance to add.
    ///
    /// Post-condition: The dialog is added to the `webview`'s list of dialogs, and `need_update` is set to `true`.
    fn add_dialog(&self, webview: servo::WebView, dialog: Dialog) {
        let mut inner_mut = self.inner_mut();
        inner_mut
            .dialogs
            .entry(webview.id())
            .or_default()
            .push(dialog);
        inner_mut.need_update = true;
    }

    /// Checks if there is any active dialog for the focused webview.
    ///
    /// Post-condition: Returns `true` if there is an active dialog, `false` otherwise.
    fn has_active_dialog(&self) -> bool {
        let Some(webview) = self.focused_webview() else {
            return false;
        };
        let inner = self.inner();
        let Some(dialogs) = inner.dialogs.get(&webview.id()) else {
            return false;
        };
        !dialogs.is_empty()
    }

    /// Returns the index of the currently focused `WebView` in the creation order.
    ///
    /// Post-condition: An `Option<usize>` is returned, which is `Some(index)` if a webview is focused,
    /// and `None` otherwise.
    pub(crate) fn get_focused_webview_index(&self) -> Option<usize> {
        let focused_id = self.inner().focused_webview_id?;
        self.webviews()
            .iter()
            .position(|webview| webview.0 == focused_id)
    }

    /// Handles `servoshell` key bindings that may have been prevented by the page in the focused webview.
    ///
    /// This method defines keyboard shortcuts for common actions like zooming, scrolling,
    /// and navigating the page. These shortcuts take precedence over page-defined handlers
    /// if the page does not explicitly prevent their default action.
    ///
    /// # Arguments
    /// * `webview` - The `WebView` where the key event occurred.
    /// * `event` - The `KeyboardEvent` to handle.
    ///
    /// Post-condition: If a matching shortcut is found, the corresponding action is performed
    /// on the `webview`.
    fn handle_overridable_key_bindings(&self, webview: ::servo::WebView, event: KeyboardEvent) {
        let origin = webview.rect().min.ceil().to_i32();
        ShortcutMatcher::from_event(event)
            .shortcut(CMD_OR_CONTROL, '=', || {
                webview.set_zoom(1.1);
            })
            .shortcut(CMD_OR_CONTROL, '+', || {
                webview.set_zoom(1.1);
            })
            .shortcut(CMD_OR_CONTROL, '-', || {
                webview.set_zoom(1.0 / 1.1);
            })
            .shortcut(CMD_OR_CONTROL, '0', || {
                webview.reset_zoom();
            })
            .shortcut(Modifiers::empty(), Key::PageDown, || {
                let scroll_location = ScrollLocation::Delta(Vector2D::new(
                    0.0,
                    -self.inner().window.page_height() + 2.0 * LINE_HEIGHT,
                ));
                webview.notify_scroll_event(scroll_location, origin, TouchEventType::Move);
            })
            .shortcut(Modifiers::empty(), Key::PageUp, || {
                let scroll_location = ScrollLocation::Delta(Vector2D::new(
                    0.0,
                    self.inner().window.page_height() - 2.0 * LINE_HEIGHT,
                ));
                webview.notify_scroll_event(scroll_location, origin, TouchEventType::Move);
            })
            .shortcut(Modifiers::empty(), Key::Home, || {
                webview.notify_scroll_event(ScrollLocation::Start, origin, TouchEventType::Move);
            })
            .shortcut(Modifiers::empty(), Key::End, || {
                webview.notify_scroll_event(ScrollLocation::End, origin, TouchEventType::Move);
            })
            .shortcut(Modifiers::empty(), Key::ArrowUp, || {
                let location = ScrollLocation::Delta(Vector2D::new(0.0, 3.0 * LINE_HEIGHT));
                webview.notify_scroll_event(location, origin, TouchEventType::Move);
            })
            .shortcut(Modifiers::empty(), Key::ArrowDown, || {
                let location = ScrollLocation::Delta(Vector2D::new(0.0, -3.0 * LINE_HEIGHT));
                webview.notify_scroll_event(location, origin, TouchEventType::Move);
            })
            .shortcut(Modifiers::empty(), Key::ArrowLeft, || {
                let location = ScrollLocation::Delta(Vector2D::new(LINE_HEIGHT, 0.0));
                webview.notify_scroll_event(location, origin, TouchEventType::Move);
            })
            .shortcut(Modifiers::empty(), Key::ArrowRight, || {
                let location = ScrollLocation::Delta(Vector2D::new(-LINE_HEIGHT, 0.0));
                webview.notify_scroll_event(location, origin, TouchEventType::Move);
            });
    }
}

/// `ServoShellServoDelegate` implements the `ServoDelegate` trait for `servoshell`.
/// It handles notifications and requests from the `Servo` engine that are relevant
/// to the embedding application.
struct ServoShellServoDelegate;
impl ServoDelegate for ServoShellServoDelegate {
    /// Notifies the delegate that the DevTools server has started.
    ///
    /// # Arguments
    /// * `_servo` - The `Servo` instance (unused).
    /// * `port` - The port on which the DevTools server is listening.
    /// * `_token` - The connection token (unused).
    ///
    /// Post-condition: An informational message is logged indicating the DevTools server port.
    fn notify_devtools_server_started(&self, _servo: &Servo, port: u16, _token: String) {
        info!("Devtools Server running on port {port}");
    }

    /// Requests permission for a DevTools connection.
    ///
    /// # Arguments
    /// * `_servo` - The `Servo` instance (unused).
    /// * `request` - The `AllowOrDenyRequest` for the connection.
    ///
    /// Post-condition: The request is automatically allowed.
    fn request_devtools_connection(&self, _servo: &Servo, request: AllowOrDenyRequest) {
        request.allow();
    }

    /// Notifies the delegate of a `ServoError`.
    ///
    /// # Arguments
    /// * `_servo` - The `Servo` instance (unused).
    /// * `error` - The `ServoError` that occurred.
    ///
    /// Post-condition: An error message is logged with details of the `ServoError`.
    fn notify_error(&self, _servo: &Servo, error: ServoError) {
        error!("Saw Servo error: {error:?}!");
    }
}

/// `RunningAppState` implements `WebViewDelegate` to handle various events and
/// requests from individual `WebView` instances.
impl WebViewDelegate for RunningAppState {
    /// Notifies the delegate that the `WebView`'s status text has changed.
    ///
    /// # Arguments
    /// * `_webview` - The `WebView` instance (unused).
    /// * `_status` - The new status text (unused).
    ///
    /// Post-condition: `need_update` is set to `true`, indicating a UI update is required.
    fn notify_status_text_changed(&self, _webview: servo::WebView, _status: Option<String>) {
        self.inner_mut().need_update = true;
    }

    /// Notifies the delegate that the `WebView`'s page title has changed.
    ///
    /// # Arguments
    /// * `webview` - The `WebView` instance whose title changed.
    /// * `title` - The new page title.
    ///
    /// Pre-condition: The `webview` is focused.
    /// Post-condition: The window's title is updated, and `need_update` is set to `true`.
    fn notify_page_title_changed(&self, webview: servo::WebView, title: Option<String>) {
        if webview.focused() {
            let window_title = format!("{} - Servo", title.clone().unwrap_or_default());
            self.inner().window.set_title(&window_title);
            self.inner_mut().need_update = true;
        }
    }

    /// Requests the delegate to move the `WebView`'s window to a new position.
    ///
    /// # Arguments
    /// * `_` - The `WebView` instance (unused).
    /// * `new_position` - The new `DeviceIntPoint` for the window's top-left corner.
    ///
    /// Post-condition: The underlying window's position is set.
    fn request_move_to(&self, _: servo::WebView, new_position: DeviceIntPoint) {
        self.inner().window.set_position(new_position);
    }

    /// Requests the delegate to resize the `WebView`'s window to a new size.
    ///
    /// # Arguments
    /// * `webview` - The `WebView` instance to resize.
    /// * `new_size` - The new `DeviceIntSize` for the window.
    ///
    /// Post-condition: The webview's internal rectangle is updated, and the underlying
    /// window is requested to resize.
    fn request_resize_to(&self, webview: servo::WebView, new_size: DeviceIntSize) {
        let mut rect = webview.rect();
        rect.set_size(new_size.to_f32());
        webview.move_resize(rect);
        self.inner().window.request_resize(&webview, new_size);
    }

    /// Displays a prompt (alert, confirm, or input) to the user.
    ///
    /// # Arguments
    /// * `webview` - The `WebView` from which the prompt originated.
    /// * `definition` - The `PromptDefinition` specifying the type and content of the prompt.
    /// * `_origin` - The `PromptOrigin` (unused).
    ///
    /// Pre-condition: `servoshell_preferences.headless` is `false`.
    /// Post-condition: A `Dialog` is created and added to the webview's active dialogs.
    /// In headless mode, prompts are automatically responded to with default values.
    fn show_prompt(
        &self,
        webview: servo::WebView,
        definition: PromptDefinition,
        _origin: PromptOrigin,
    ) {
        // Block Logic: If in headless mode, respond to prompts automatically with default values.
        if self.servoshell_preferences.headless {
            let _ = match definition {
                PromptDefinition::Alert(_message, sender) => sender.send(()),
                PromptDefinition::OkCancel(_message, sender) => sender.send(PromptResult::Primary),
                PromptDefinition::Input(_message, default, sender) => {
                    sender.send(Some(default.to_owned()))
                },
            };
            return;
        }
        // Block Logic: In non-headless mode, create and add a dialog to the webview.
        match definition {
            PromptDefinition::Alert(message, sender) => {
                let alert_dialog = Dialog::new_alert_dialog(message, sender);
                self.add_dialog(webview, alert_dialog);
            },
            PromptDefinition::OkCancel(message, sender) => {
                let okcancel_dialog = Dialog::new_okcancel_dialog(message, sender);
                self.add_dialog(webview, okcancel_dialog);
            },
            PromptDefinition::Input(message, default, sender) => {
                let input_dialog = Dialog::new_input_dialog(message, default, sender);
                self.add_dialog(webview, input_dialog);
            },
        }
    }

    /// Requests authentication from the user.
    ///
    /// # Arguments
    /// * `webview` - The `WebView` instance requesting authentication.
    /// * `authentication_request` - The `AuthenticationRequest` details.
    ///
    /// Pre-condition: `servoshell_preferences.headless` is `false`.
    /// Post-condition: An authentication dialog is added to the webview's active dialogs.
    /// In headless mode, authentication requests are ignored.
    fn request_authentication(
        &self,
        webview: WebView,
        authentication_request: AuthenticationRequest,
    ) {
        // Block Logic: If in headless mode, ignore authentication requests.
        if self.servoshell_preferences.headless {
            return;
        }

        self.add_dialog(
            webview,
            Dialog::new_authentication_dialog(authentication_request),
        );
    }

    /// Requests to open a new auxiliary `WebView`.
    ///
    /// # Arguments
    /// * `parent_webview` - The `WebView` from which the request originated.
    ///
    /// Post-condition: A new auxiliary `WebView` is created, its delegate is set
    /// to the parent's delegate, and it's added to the application's state.
    /// Returns `Some(WebView)` if successful.
    fn request_open_auxiliary_webview(
        &self,
        parent_webview: servo::WebView,
    ) -> Option<servo::WebView> {
        let webview = self.servo.new_auxiliary_webview();
        webview.set_delegate(parent_webview.delegate());
        self.add(webview.clone());
        Some(webview)
    }

    /// Notifies the delegate that a `WebView` is ready to be shown.
    ///
    /// # Arguments
    /// * `webview` - The `WebView` instance that is ready.
    ///
    /// Post-condition: The webview is focused, moved/resized to fill the viewport,
    /// brought to the top, and its rendering context is notified of a resize.
    fn notify_ready_to_show(&self, webview: servo::WebView) {
        let rect = self
            .inner()
            .window
            .get_coordinates()
            .get_viewport()
            .to_f32();

        webview.focus();
        webview.move_resize(rect);
        webview.raise_to_top(true);
        webview.notify_rendering_context_resized();
    }

    /// Notifies the delegate that a `WebView` has been closed.
    ///
    /// # Arguments
    /// * `webview` - The `WebView` instance that was closed.
    ///
    /// Post-condition: The webview is closed by `close_webview`.
    fn notify_closed(&self, webview: servo::WebView) {
        self.close_webview(webview.id());
    }

    /// Notifies the delegate that the focus state of a `WebView` has changed.
    ///
    /// # Arguments
    /// * `webview` - The `WebView` instance whose focus state changed.
    /// * `focused` - `true` if the webview is now focused, `false` otherwise.
    ///
    /// Post-condition: If focused, the webview is shown, `need_update` is set,
    /// and `focused_webview_id` is updated. If blurred, `focused_webview_id` is cleared.
    fn notify_focus_changed(&self, webview: servo::WebView, focused: bool) {
        let mut inner_mut = self.inner_mut();
        if focused {
            webview.show(true);
            inner_mut.need_update = true;
            inner_mut.focused_webview_id = Some(webview.id());
        } else if inner_mut.focused_webview_id == Some(webview.id()) {
            inner_mut.focused_webview_id = None;
        }
    }

    /// Notifies the delegate of a keyboard event.
    ///
    /// # Arguments
    /// * `webview` - The `WebView` instance that received the event.
    /// * `keyboard_event` - The `KeyboardEvent` to handle.
    ///
    /// Post-condition: The `handle_overridable_key_bindings` method is called to process
    /// the keyboard event.
    fn notify_keyboard_event(&self, webview: servo::WebView, keyboard_event: KeyboardEvent) {
        self.handle_overridable_key_bindings(webview, keyboard_event);
    }

    /// Notifies the delegate that the cursor for a `WebView` has changed.
    ///
    /// # Arguments
    /// * `_webview` - The `WebView` instance (unused).
    /// * `cursor` - The new `servo::Cursor` style.
    ///
    /// Post-condition: The underlying window's cursor is set.
    fn notify_cursor_changed(&self, _webview: servo::WebView, cursor: servo::Cursor) {
        self.inner().window.set_cursor(cursor);
    }

    /// Notifies the delegate that the load status of a `WebView` has changed.
    ///
    /// # Arguments
    /// * `_webview` - The `WebView` instance (unused).
    /// * `_status` - The new `LoadStatus` (unused).
    ///
    /// Post-condition: `need_update` is set to `true`, indicating a UI update is required.
    fn notify_load_status_changed(&self, _webview: servo::WebView, _status: LoadStatus) {
        self.inner_mut().need_update = true;
    }

    /// Notifies the delegate that the fullscreen state of a `WebView` has changed.
    ///
    /// # Arguments
    /// * `_webview` - The `WebView` instance (unused).
    /// * `fullscreen_state` - `true` if now in fullscreen, `false` otherwise.
    ///
    /// Post-condition: The underlying window's fullscreen state is set.
    fn notify_fullscreen_state_changed(&self, _webview: servo::WebView, fullscreen_state: bool) {
        self.inner().window.set_fullscreen(fullscreen_state);
    }

    /// Displays a Bluetooth device selection dialog.
    ///
    /// # Arguments
    /// * `webview` - The `WebView` instance requesting the dialog.
    /// * `devices` - A list of available Bluetooth device names.
    /// * `response_sender` - An `IpcSender` to send the selected device name back.
    ///
    /// Post-condition: A platform-specific dialog is shown, and the selected device
    /// (or `None`) is sent back via `response_sender`.
    fn show_bluetooth_device_dialog(
        &self,
        webview: servo::WebView,
        devices: Vec<String>,
        response_sender: IpcSender<Option<String>>,
    ) {
        let selected = platform_get_selected_devices(devices);
        if let Err(e) = response_sender.send(selected) {
            webview.send_error(format!(
                "Failed to send GetSelectedBluetoothDevice response: {e}"
            ));
        }
    }

    /// Displays a file selection dialog.
    ///
    /// # Arguments
    /// * `webview` - The `WebView` instance requesting the dialog.
    /// * `filter_pattern` - A vector of `FilterPattern`s for file types.
    /// * `allow_select_mutiple` - `true` if multiple files can be selected.
    /// * `response_sender` - An `IpcSender` to send the selected file paths back.
    ///
    /// Post-condition: A file selection `Dialog` is created and added to the webview's active dialogs.
    fn show_file_selection_dialog(
        &self,
        webview: servo::WebView,
        filter_pattern: Vec<FilterPattern>,
        allow_select_mutiple: bool,
        response_sender: IpcSender<Option<Vec<PathBuf>>>,
    ) {
        let file_dialog =
            Dialog::new_file_dialog(allow_select_mutiple, response_sender, filter_pattern);
        self.add_dialog(webview, file_dialog);
    }

    /// Prompts the user for a specific permission.
    ///
    /// # Arguments
    /// * `_webview` - The `WebView` instance (unused).
    /// * `request` - The `PermissionRequest` details.
    ///
    /// Pre-condition: `servoshell_preferences.headless` is `false`.
    /// Post-condition: A platform-specific user prompt is shown, and the response is handled.
    /// In headless mode, requests are denied by default.
    fn request_permission(&self, _webview: servo::WebView, request: PermissionRequest) {
        if !self.servoshell_preferences.headless {
            prompt_user(request);
        }
    }

    /// Notifies the delegate that a new frame is ready for rendering.
    ///
    /// # Arguments
    /// * `_webview` - The `WebView` instance (unused).
    ///
    /// Post-condition: `need_repaint` is set to `true`, indicating a repaint is required.
    fn notify_new_frame_ready(&self, _webview: servo::WebView) {
        self.inner_mut().need_repaint = true;
    }

    /// Plays a gamepad haptic effect.
    ///
    /// # Arguments
    /// * `_webview` - The `WebView` instance (unused).
    /// * `index` - The index of the gamepad.
    /// * `effect_type` - The `GamepadHapticEffectType` to play.
    /// * `effect_complete_sender` - An `IpcSender` to signal when the effect is complete.
    ///
    /// Post-condition: The haptic effect is played on the specified gamepad,
    /// or `false` is sent back if `GamepadSupport` is not initialized.
    fn play_gamepad_haptic_effect(
        &self,
        _webview: servo::WebView,
        index: usize,
        effect_type: GamepadHapticEffectType,
        effect_complete_sender: IpcSender<bool>,
    ) {
        // Block Logic: If gamepad support is available, play the haptic effect; otherwise, send false.
        match self.inner_mut().gamepad_support.as_mut() {
            Some(gamepad_support) => {
                gamepad_support.play_haptic_effect(index, effect_type, effect_complete_sender);
            },
            None => {
                let _ = effect_complete_sender.send(false);
            },
        }
    }

    /// Stops a gamepad haptic effect.
    ///
    /// # Arguments
    /// * `_webview` - The `WebView` instance (unused).
    /// * `index` - The index of the gamepad.
    /// * `haptic_stop_sender` - An `IpcSender` to signal when the effect is stopped.
    ///
    /// Post-condition: The haptic effect is stopped on the specified gamepad,
    /// or `false` is sent back if `GamepadSupport` is not initialized.
    fn stop_gamepad_haptic_effect(
        &self,
        _webview: servo::WebView,
        index: usize,
        haptic_stop_sender: IpcSender<bool>,
    ) {
        // Block Logic: If gamepad support is available, stop the haptic effect; otherwise, send false.
        let stopped = match self.inner_mut().gamepad_support.as_mut() {
            Some(gamepad_support) => gamepad_support.stop_haptic_effect(index),
            None => false,
        };
        let _ = haptic_stop_sender.send(stopped);
    }
    /// Shows the Input Method Editor (IME).
    ///
    /// # Arguments
    /// * `_webview` - The `WebView` instance (unused).
    /// * `input_type` - The type of input method to use.
    /// * `text` - Optional pre-existing text content and insertion point.
    /// * `multiline` - `true` if the input is multiline, `false` otherwise.
    /// * `position` - The position of the input field.
    ///
    /// Post-condition: The `window`'s `show_ime` method is called.
    fn show_ime(
        &self,
        _webview: WebView,
        input_type: servo::InputMethodType,
        text: Option<(String, i32)>,
        multiline: bool,
        position: servo::webrender_api::units::DeviceIntRect,
    ) {
        self.inner()
            .window
            .show_ime(input_type, text, multiline, position);
    }

    /// Hides the Input Method Editor (IME).
    ///
    /// # Arguments
    /// * `_webview` - The `WebView` instance (unused).
    ///
    /// Post-condition: The `window`'s `hide_ime` method is called.
    fn hide_ime(&self, _webview: WebView) {
        self.inner().window.hide_ime();
    }
}

/// Displays a permission request dialog to the user on Linux.
///
/// # Arguments
/// * `request` - The `PermissionRequest` to prompt for.
///
/// Pre-condition: Running on Linux.
/// Post-condition: A tinyfiledialogs message box is shown, and the permission is allowed or denied based on user input.
#[cfg(target_os = "linux")]
fn prompt_user(request: PermissionRequest) {
    use tinyfiledialogs::YesNo;

    let message = format!(
        "Do you want to grant permission for {:?}?",
        request.feature()
    );
    match tinyfiledialogs::message_box_yes_no(
        "Permission request dialog",
        &message,
        MessageBoxIcon::Question,
        YesNo::No,
    ) {
        YesNo::Yes => request.allow(),
        YesNo::No => request.deny(),
    }
}

/// Placeholder for `prompt_user` on non-Linux platforms. Requests are denied by default.
#[cfg(not(target_os = "linux"))]
fn prompt_user(_request: PermissionRequest) {
    // Requests are denied by default.
}

/// On Linux, displays a dialog for selecting Bluetooth devices.
///
/// # Arguments
/// * `devices` - A vector of strings representing available Bluetooth devices.
///
/// Pre-condition: Running on Linux.
/// Post-condition: A tinyfiledialogs list dialog is shown, and the selected device's
/// address (or `None`) is returned.
///
/// Note: The device string format is "Address|Name". This function extracts the address part.
#[cfg(target_os = "linux")]
fn platform_get_selected_devices(devices: Vec<String>) -> Option<String> {
    thread::Builder::new()
        .name("DevicePicker".to_owned())
        .spawn(move || {
            let dialog_rows: Vec<&str> = devices.iter().map(|s| s.as_ref()).collect();
            let dialog_rows: Option<&[&str]> = Some(dialog_rows.as_slice());

            match tinyfiledialogs::list_dialog("Choose a device", &["Id", "Name"], dialog_rows) {
                Some(device) => {
                    // The device string format will be "Address|Name". We need the first part of it.
                    device.split('|').next().map(|s| s.to_string())
                },
                None => None,
            }
        })
        .unwrap()
        .join()
        .expect("Thread spawning failed")
}

/// Placeholder for `platform_get_selected_devices` on non-Linux platforms.
/// Selects the first available device by default.
///
/// # Arguments
/// * `devices` - A vector of strings representing available Bluetooth devices.
///
/// Pre-condition: Not running on Linux.
/// Post-condition: Returns the address of the first device in the list, or `None` if the list is empty.
#[cfg(not(target_os = "linux"))]
fn platform_get_selected_devices(devices: Vec<String>) -> Option<String> {
    for device in devices {
        if let Some(address) = device.split('|').next().map(|s| s.to_string()) {
            return Some(address);
        }
    }
    None
}
