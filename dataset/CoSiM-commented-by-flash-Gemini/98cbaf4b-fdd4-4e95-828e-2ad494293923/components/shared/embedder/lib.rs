/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! This module defines the core traits, enums, and structs that facilitate
//! communication between the Servo engine and its embedding application (the "embedder").
//! It establishes the interface for embedder-specific functionalities such as
//! event handling, UI prompts, resource requests, and various browser-level
//! notifications. The types defined here enable a clean separation of concerns,
//! allowing Servo to be integrated into diverse host environments.

pub mod input_events;
pub mod resources;

use std::fmt::{Debug, Error, Formatter};
use std::path::PathBuf;

use base::id::{PipelineId, WebViewId};
use crossbeam_channel::Sender;
use http::{HeaderMap, Method, StatusCode};
use ipc_channel::ipc::IpcSender;
pub use keyboard_types::{KeyboardEvent, Modifiers};
use log::warn;
use malloc_size_of_derive::MallocSizeOf;
use num_derive::FromPrimitive;
use serde::{Deserialize, Serialize};
use servo_url::ServoUrl;
use webrender_api::units::{DeviceIntPoint, DeviceIntRect, DeviceIntSize};

pub use crate::input_events::*;

/// Tracks whether Servo isn't shutting down, is in the process of shutting down,
/// or has finished shutting down.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShutdownState {
    /// Servo is operating normally and is not in a shutdown state.
    NotShuttingDown,
    /// Servo has begun the shutdown process.
    ShuttingDown,
    /// Servo has completed its shutdown and is no longer active.
    FinishedShuttingDown,
}

/// A cursor for the window. This is different from a CSS cursor (see
/// `CursorKind`) in that it has no `Auto` value.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Deserialize, Eq, FromPrimitive, PartialEq, Serialize)]
pub enum Cursor {
    None,
    Default,
    Pointer,
    ContextMenu,
    Help,
    Progress,
    Wait,
    Cell,
    Crosshair,
    Text,
    VerticalText,
    Alias,
    Copy,
    Move,
    NoDrop,
    NotAllowed,
    Grab,
    Grabbing,
    EResize,
    NResize,
    NeResize,
    NwResize,
    SResize,
    SeResize,
    SwResize,
    WResize,
    EwResize,
    NsResize,
    NeswResize,
    NwseResize,
    ColResize,
    RowResize,
    AllScroll,
    ZoomIn,
    ZoomOut,
}

#[cfg(feature = "webxr")]
/// `EventLoopWaker` is a trait that allows waking up the main event loop from another thread.
/// This is used for WebXR-specific functionalities when the `webxr` feature is enabled.
pub use webxr_api::MainThreadWaker as EventLoopWaker;
#[cfg(not(feature = "webxr"))]
/// `EventLoopWaker` is a trait that allows waking up the main event loop from another thread.
///
/// This trait provides a mechanism for other threads to signal the main event loop
/// that there are pending events to be processed, ensuring responsiveness.
pub trait EventLoopWaker: 'static + Send {
    /// Clones the `EventLoopWaker` into a boxed trait object.
    ///
    /// Post-condition: A new `Box<dyn EventLoopWaker>` is returned, allowing for
    /// multiple independent wakers to be created.
    fn clone_box(&self) -> Box<dyn EventLoopWaker>;
    /// Wakes up the associated event loop.
    ///
    /// Post-condition: The event loop is signaled to process pending events.
    fn wake(&self);
}

#[cfg(not(feature = "webxr"))]
impl Clone for Box<dyn EventLoopWaker> {
    /// Clones a boxed `EventLoopWaker`.
    ///
    /// Post-condition: A new `Box<dyn EventLoopWaker>` is returned, representing a clone
    /// of the original waker.
    fn clone(&self) -> Self {
        EventLoopWaker::clone_box(self.as_ref())
    }
}

/// `EmbedderProxy` provides a mechanism for Servo to send messages to the embedding application.
pub struct EmbedderProxy {
    /// The sender half of a channel used to send `EmbedderMsg` to the embedder.
    pub sender: Sender<EmbedderMsg>,
    /// A waker used to signal the embedder's event loop when a message is sent.
    pub event_loop_waker: Box<dyn EventLoopWaker>,
}

impl EmbedderProxy {
    /// Sends an `EmbedderMsg` to the embedder and then wakes up the embedder's event loop.
    ///
    /// # Arguments
    /// * `message` - The `EmbedderMsg` to be sent.
    ///
    /// Post-condition: The `message` is sent to the embedder's channel, and its event loop
    /// is awakened. A warning is logged if the message fails to send.
    pub fn send(&self, message: EmbedderMsg) {
        // Send a message and kick the OS event loop awake.
        if let Err(err) = self.sender.send(message) {
            warn!("Failed to send response ({:?}).", err);
        }
        self.event_loop_waker.wake();
    }
}

impl Clone for EmbedderProxy {
    /// Creates a new `EmbedderProxy` that shares the same sender and event loop waker.
    ///
    /// Post-condition: A new `EmbedderProxy` instance is returned, which is a clone
    /// of the original.
    fn clone(&self) -> EmbedderProxy {
        EmbedderProxy {
            sender: self.sender.clone(),
            event_loop_waker: self.event_loop_waker.clone(),
        }
    }
}

/// `ContextMenuResult` enumerates the possible outcomes when a context menu is displayed.
#[derive(Deserialize, Serialize)]
pub enum ContextMenuResult {
    /// The context menu was dismissed by the user without making a selection.
    Dismissed,
    /// The user interacted with the context menu, but the action was ignored.
    Ignored,
    /// An item at the specified index was selected from the context menu.
    Selected(usize),
}

/// `PromptDefinition` describes the type and content of a prompt to be displayed to the user.
#[derive(Deserialize, Serialize)]
pub enum PromptDefinition {
    /// A simple alert message with an "OK" button.
    Alert(String, IpcSender<()>),
    /// A confirmation dialog with "OK" and "Cancel" buttons, returning a `PromptResult`.
    OkCancel(String, IpcSender<PromptResult>),
    /// An input dialog that allows the user to enter text, with an initial value.
    Input(String, String, IpcSender<Option<String>>),
}

/// `AuthenticationResponse` holds the username and password for HTTP authentication requests.
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct AuthenticationResponse {
    /// Username for http request authentication
    pub username: String,
    /// Password for http request authentication
    pub password: String,
}

/// `PromptOrigin` indicates whether a prompt originated from trusted Servo code or untrusted web content.
#[derive(Deserialize, PartialEq, Serialize)]
pub enum PromptOrigin {
    /// The prompt was triggered by untrusted content (e.g., `window.prompt`, `alert`, `confirm`).
    /// The message content is assumed to be potentially malicious.
    Untrusted,
    /// The prompt was triggered by trusted Servo code (e.g., asking for permissions, showing errors).
    /// The message content is considered safe.
    Trusted,
}

/// `PromptResult` indicates the user's interaction with a prompt dialog.
#[derive(Deserialize, PartialEq, Serialize)]
pub enum PromptResult {
    /// The user clicked the primary action button (e.g., "OK", "Yes").
    Primary,
    /// The user clicked the secondary action button (e.g., "Cancel", "No").
    Secondary,
    /// The prompt was dismissed without a specific action (e.g., closing the dialog).
    Dismissed,
}

/// `AllowOrDeny` is a simple enum used to represent a binary choice: allow or deny an action.
#[derive(Clone, Copy, Deserialize, PartialEq, Serialize)]
pub enum AllowOrDeny {
    /// The action is allowed.
    Allow,
    /// The action is denied.
    Deny,
}

/// `EmbedderMsg` defines the types of messages that Servo can send to its embedding application.
/// These messages inform the embedder about various browser events, UI requests,
/// and status updates.
#[derive(Deserialize, Serialize)]
pub enum EmbedderMsg {
    /// A status message to be displayed by the browser chrome (e.g., in the status bar).
    Status(WebViewId, Option<String>),
    /// Alerts the embedder that the current page has changed its title.
    ChangePageTitle(WebViewId, Option<String>),
    /// Requests the embedder to move the window to a specific point.
    MoveTo(WebViewId, DeviceIntPoint),
    /// Requests the embedder to resize the window to a specific size.
    ResizeTo(WebViewId, DeviceIntSize),
    /// Requests the embedder to display a UI prompt to the user.
    Prompt(WebViewId, PromptDefinition, PromptOrigin),
    /// Requests authentication from the embedder for a load or navigation.
    RequestAuthentication(
        WebViewId,
        ServoUrl,
        bool, /* for proxy */
        IpcSender<Option<AuthenticationResponse>>,
    ),
    /// Requests the embedder to show a context menu.
    ShowContextMenu(
        WebViewId,
        IpcSender<ContextMenuResult>,
        Option<String>,
        Vec<String>,
    ),
    /// Whether or not to allow a pipeline to load a url.
    AllowNavigationRequest(WebViewId, PipelineId, ServoUrl),
    /// Whether or not to allow script to open a new tab/browser.
    AllowOpeningWebView(WebViewId, IpcSender<Option<WebViewId>>),
    /// Notifies the embedder that a new webview was created.
    WebViewOpened(WebViewId),
    /// Notifies the embedder that a webview was destroyed.
    WebViewClosed(WebViewId),
    /// Notifies the embedder that a webview gained focus for keyboard events.
    WebViewFocused(WebViewId),
    /// Notifies the embedder that all webviews lost focus for keyboard events.
    WebViewBlurred,
    /// Requests the embedder whether to allow a document to unload.
    AllowUnload(WebViewId, IpcSender<AllowOrDeny>),
    /// Sends an unconsumed keyboard event back to the embedder.
    Keyboard(WebViewId, KeyboardEvent),
    /// Instructs the embedder to clear the system clipboard.
    ClearClipboard(WebViewId),
    /// Requests the system clipboard contents from the embedder.
    GetClipboardText(WebViewId, IpcSender<Result<String, String>>),
    /// Instructs the embedder to set the system clipboard contents.
    SetClipboardText(WebViewId, String),
    /// Requests the embedder to change the mouse cursor.
    SetCursor(WebViewId, Cursor),
    /// Notifies the embedder that a new favicon was detected.
    NewFavicon(WebViewId, ServoUrl),
    /// Notifies the embedder that the history state has changed for a webview.
    HistoryChanged(WebViewId, Vec<ServoUrl>, usize),
    /// Notifies the embedder that the fullscreen state of a webview has changed.
    NotifyFullscreenStateChanged(WebViewId, bool),
    /// Notifies the embedder that the `LoadStatus` of the given `WebView` has changed.
    NotifyLoadStatusChanged(WebViewId, LoadStatus),
    /// Requests a web resource from the embedder.
    WebResourceRequested(
        Option<WebViewId>,
        WebResourceRequest,
        IpcSender<WebResourceResponseMsg>,
    ),
    /// Notifies the embedder that a pipeline panicked, providing a reason and optional backtrace.
    Panic(WebViewId, String, Option<String>),
    /// Requests the embedder to open a dialog to select a Bluetooth device.
    GetSelectedBluetoothDevice(WebViewId, Vec<String>, IpcSender<Option<String>>),
    /// Requests the embedder to open a file dialog to select files.
    SelectFiles(
        WebViewId,
        Vec<FilterPattern>,
        bool,
        IpcSender<Option<Vec<PathBuf>>>,
    ),
    /// Requests the embedder to prompt for a specific permission.
    PromptPermission(WebViewId, PermissionFeature, IpcSender<AllowOrDeny>),
    /// Requests the embedder to present an Input Method Editor (IME) to the user.
    ShowIME(
        WebViewId,
        InputMethodType,
        Option<(String, i32)>,
        bool,
        DeviceIntRect,
    ),
    /// Requests the embedder to hide the Input Method Editor (IME).
    HideIME(WebViewId),
    /// Reports a complete sampled profile to the embedder.
    ReportProfile(Vec<u8>),
    /// Notifies the embedder about media session events.
    MediaSessionEvent(WebViewId, MediaSessionEvent),
    /// Reports the status of the DevTools Server (started or failed) to the embedder.
    OnDevtoolsStarted(Result<u16, ()>, String),
    /// Requests the user's permission for a DevTools client to connect.
    RequestDevtoolsConnection(IpcSender<AllowOrDeny>),
    /// Requests the embedder to play a haptic effect on a connected gamepad.
    PlayGamepadHapticEffect(WebViewId, usize, GamepadHapticEffectType, IpcSender<bool>),
    /// Requests the embedder to stop a haptic effect on a connected gamepad.
    StopGamepadHapticEffect(WebViewId, usize, IpcSender<bool>),
    /// Informs the embedder that the constellation has completed shutdown.
    ShutdownComplete,
}

impl Debug for EmbedderMsg {
    /// Formats the `EmbedderMsg` for debugging purposes.
    ///
    /// # Arguments
    /// * `f` - A mutable reference to the `Formatter`.
    ///
    /// Post-condition: The message is written to the formatter in a human-readable format.
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match *self {
            EmbedderMsg::Status(..) => write!(f, "Status"),
            EmbedderMsg::ChangePageTitle(..) => write!(f, "ChangePageTitle"),
            EmbedderMsg::MoveTo(..) => write!(f, "MoveTo"),
            EmbedderMsg::ResizeTo(..) => write!(f, "ResizeTo"),
            EmbedderMsg::Prompt(..) => write!(f, "Prompt"),
            EmbedderMsg::RequestAuthentication(..) => write!(f, "RequestAuthentication"),
            EmbedderMsg::AllowUnload(..) => write!(f, "AllowUnload"),
            EmbedderMsg::AllowNavigationRequest(..) => write!(f, "AllowNavigationRequest"),
            EmbedderMsg::Keyboard(..) => write!(f, "Keyboard"),
            EmbedderMsg::ClearClipboard(..) => write!(f, "ClearClipboard"),
            EmbedderMsg::GetClipboardText(..) => write!(f, "GetClipboardText"),
            EmbedderMsg::SetClipboardText(..) => write!(f, "SetClipboardText"),
            EmbedderMsg::SetCursor(..) => write!(f, "SetCursor"),
            EmbedderMsg::NewFavicon(..) => write!(f, "NewFavicon"),
            EmbedderMsg::HistoryChanged(..) => write!(f, "HistoryChanged"),
            EmbedderMsg::NotifyFullscreenStateChanged(..) => {
                write!(f, "NotifyFullscreenStateChanged")
            },
            EmbedderMsg::NotifyLoadStatusChanged(_, status) => {
                write!(f, "NotifyLoadStatusChanged({status:?})")
            },
            EmbedderMsg::WebResourceRequested(..) => write!(f, "WebResourceRequested"),
            EmbedderMsg::Panic(..) => write!(f, "Panic"),
            EmbedderMsg::GetSelectedBluetoothDevice(..) => write!(f, "GetSelectedBluetoothDevice"),
            EmbedderMsg::SelectFiles(..) => write!(f, "SelectFiles"),
            EmbedderMsg::PromptPermission(..) => write!(f, "PromptPermission"),
            EmbedderMsg::ShowIME(..) => write!(f, "ShowIME"),
            EmbedderMsg::HideIME(..) => write!(f, "HideIME"),
            EmbedderMsg::AllowOpeningWebView(..) => write!(f, "AllowOpeningWebView"),
            EmbedderMsg::WebViewOpened(..) => write!(f, "WebViewOpened"),
            EmbedderMsg::WebViewClosed(..) => write!(f, "WebViewClosed"),
            EmbedderMsg::WebViewFocused(..) => write!(f, "WebViewFocused"),
            EmbedderMsg::WebViewBlurred => write!(f, "WebViewBlurred"),
            EmbedderMsg::ReportProfile(..) => write!(f, "ReportProfile"),
            EmbedderMsg::MediaSessionEvent(..) => write!(f, "MediaSessionEvent"),
            EmbedderMsg::OnDevtoolsStarted(..) => write!(f, "OnDevtoolsStarted"),
            EmbedderMsg::RequestDevtoolsConnection(..) => write!(f, "RequestDevtoolsConnection"),
            EmbedderMsg::ShowContextMenu(..) => write!(f, "ShowContextMenu"),
            EmbedderMsg::PlayGamepadHapticEffect(..) => write!(f, "PlayGamepadHapticEffect"),
            EmbedderMsg::StopGamepadHapticEffect(..) => write!(f, "StopGamepadHapticEffect"),
            EmbedderMsg::ShutdownComplete => write!(f, "ShutdownComplete"),
        }
    }
}

/// Filter for file selection;
/// the `String` content is expected to be extension (e.g, "doc", without the prefixing ".")
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FilterPattern(pub String);

/// `MediaMetadata` represents the metadata of currently playing media.
///
/// <https://w3c.github.io/mediasession/#mediametadata>
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MediaMetadata {
    /// Title of the media.
    pub title: String,
    /// Artist of the media.
    pub artist: String,
    /// Album of the media.
    pub album: String,
}

impl MediaMetadata {
    /// Creates a new `MediaMetadata` instance with a given title and empty artist/album.
    ///
    /// # Arguments
    /// * `title` - The title of the media.
    ///
    /// Post-condition: A new `MediaMetadata` struct is returned.
    pub fn new(title: String) -> Self {
        Self {
            title,
            artist: "".to_owned(),
            album: "".to_owned(),
        }
    }
}

/// `MediaSessionPlaybackState` defines the playback state of a media session.
///
/// <https://w3c.github.io/mediasession/#enumdef-mediasessionplaybackstate>
#[repr(i32)]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum MediaSessionPlaybackState {
    /// The browsing context does not specify whether it’s playing or paused.
    None_ = 1,
    /// The browsing context is currently playing media and it can be paused.
    Playing,
    /// The browsing context has paused media and it can be resumed.
    Paused,
}

/// `MediaPositionState` represents the current position state of media playback.
///
/// <https://w3c.github.io/mediasession/#dictdef-mediapositionstate>
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MediaPositionState {
    /// The duration of the media in seconds.
    pub duration: f64,
    /// The current playback rate (e.g., 1.0 for normal speed).
    pub playback_rate: f64,
    /// The current position of the playback in seconds.
    pub position: f64,
}

impl MediaPositionState {
    /// Creates a new `MediaPositionState` instance.
    ///
    /// # Arguments
    /// * `duration` - The total duration of the media.
    /// * `playback_rate` - The current playback speed.
    /// * `position` - The current playback position.
    ///
    /// Post-condition: A new `MediaPositionState` struct is returned.
    pub fn new(duration: f64, playback_rate: f64, position: f64) -> Self {
        Self {
            duration,
            playback_rate,
            position,
        }
    }
}

/// `MediaSessionEvent` defines the types of events related to media sessions that can be sent from script to the embedder.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum MediaSessionEvent {
    /// Indicates that the media metadata is available or has changed.
    SetMetadata(MediaMetadata),
    /// Indicates that the playback state (playing, paused, etc.) has changed.
    PlaybackStateChange(MediaSessionPlaybackState),
    /// Indicates that the position state (duration, playback rate, current position) has been set or updated.
    SetPositionState(MediaPositionState),
}

/// `PermissionFeature` enumerates the different types of permissions that can be requested by web content.
///
/// This enum's variants match the DOM `PermissionName` enum.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum PermissionFeature {
    Geolocation,
    Notifications,
    Push,
    Midi,
    Camera,
    Microphone,
    Speaker,
    DeviceInfo,
    BackgroundSync,
    Bluetooth,
    PersistentStorage,
}

/// `InputMethodType` specifies the kind of input method editor appropriate for editing a field.
///
/// This is a subset of `htmlinputelement::InputType` because some variants of `InputType`
/// don't make sense in this context.
#[derive(Debug, Deserialize, Serialize)]
pub enum InputMethodType {
    Color,
    Date,
    DatetimeLocal,
    Email,
    Month,
    Number,
    Password,
    Search,
    Tel,
    Text,
    Time,
    Url,
    Week,
}

/// `DualRumbleEffectParams` holds parameters for a dual-rumble haptic effect on a gamepad.
///
/// <https://w3.org/TR/gamepad/#dom-gamepadhapticeffecttype-dual-rumble>
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DualRumbleEffectParams {
    /// The duration of the effect in seconds.
    pub duration: f64,
    /// The start delay of the effect in seconds.
    pub start_delay: f64,
    /// The magnitude of the strong rumble motor (normalized to 0.0-1.0).
    pub strong_magnitude: f64,
    /// The magnitude of the weak rumble motor (normalized to 0.0-1.0).
    pub weak_magnitude: f64,
}

/// `GamepadHapticEffectType` enumerates the different types of haptic effects supported by gamepads.
///
/// <https://w3.org/TR/gamepad/#dom-gamepadhapticeffecttype>
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum GamepadHapticEffectType {
    /// A dual-rumble effect with specified parameters.
    DualRumble(DualRumbleEffectParams),
}

/// `WebResourceRequest` encapsulates a web resource request, including its method, headers, and URL.
#[derive(Clone, Debug, Deserialize, MallocSizeOf, Serialize)]
pub struct WebResourceRequest {
    /// The HTTP method of the request (e.g., GET, POST).
    #[serde(
        deserialize_with = "::hyper_serde::deserialize",
        serialize_with = "::hyper_serde::serialize"
    )]
    #[ignore_malloc_size_of = "Defined in hyper"]
    pub method: Method,
    /// The HTTP headers associated with the request.
    #[serde(
        deserialize_with = "::hyper_serde::deserialize",
        serialize_with = "::hyper_serde::serialize"
    )]
    #[ignore_malloc_size_of = "Defined in hyper"]
    pub headers: HeaderMap,
    /// The URL of the requested resource.
    pub url: ServoUrl,
    /// True if the request is for the main frame of a webview.
    pub is_for_main_frame: bool,
    /// True if the request is a redirect.
    pub is_redirect: bool,
}

impl WebResourceRequest {
    /// Creates a new `WebResourceRequest` instance.
    ///
    /// # Arguments
    /// * `method` - The HTTP method.
    /// * `headers` - The request headers.
    /// * `url` - The request URL.
    /// * `is_for_main_frame` - Whether it's for the main frame.
    /// * `is_redirect` - Whether it's a redirect.
    ///
    /// Post-condition: A new `WebResourceRequest` struct is returned.
    pub fn new(
        method: Method,
        headers: HeaderMap,
        url: ServoUrl,
        is_for_main_frame: bool,
        is_redirect: bool,
    ) -> Self {
        WebResourceRequest {
            method,
            url,
            headers,
            is_for_main_frame,
            is_redirect,
        }
    }
}

/// `WebResourceResponseMsg` defines messages related to web resource responses.
#[derive(Clone, Deserialize, Serialize)]
pub enum WebResourceResponseMsg {
    /// The start of a web resource response, including the response headers and status.
    Start(WebResourceResponse),
    /// A chunk of the response body.
    Body(HttpBodyData),
    /// Instructs not to override the response, allowing the default behavior.
    None,
}

/// `HttpBodyData` represents a chunk of HTTP response body data or a completion/cancellation signal.
#[derive(Clone, Deserialize, Serialize)]
pub enum HttpBodyData {
    /// A chunk of bytes from the HTTP response body.
    Chunk(Vec<u8>),
    /// Signals that the entire HTTP response body has been sent.
    Done,
    /// Signals that the HTTP response transfer was cancelled.
    Cancelled,
}

/// `WebResourceResponse` encapsulates a web resource response, including its URL, headers, and status code.
#[derive(Clone, Debug, Deserialize, MallocSizeOf, Serialize)]
pub struct WebResourceResponse {
    /// The URL of the responded resource.
    pub url: ServoUrl,
    /// The HTTP headers of the response.
    #[serde(
        deserialize_with = "::hyper_serde::deserialize",
        serialize_with = "::hyper_serde::serialize"
    )]
    #[ignore_malloc_size_of = "Defined in hyper"]
    pub headers: HeaderMap,
    /// The HTTP status code of the response (e.g., 200 OK, 404 Not Found).
    #[serde(
        deserialize_with = "::hyper_serde::deserialize",
        serialize_with = "::hyper_serde::serialize"
    )]
    #[ignore_malloc_size_of = "Defined in hyper"]
    pub status_code: StatusCode,
    /// The status message accompanying the status code (e.g., "OK").
    pub status_message: Vec<u8>,
}

impl WebResourceResponse {
    /// Creates a new `WebResourceResponse` instance with default values.
    ///
    /// # Arguments
    /// * `url` - The URL of the resource.
    ///
    /// Post-condition: A new `WebResourceResponse` is returned with an `OK` status
    /// and no headers.
    pub fn new(url: ServoUrl) -> WebResourceResponse {
        WebResourceResponse {
            url,
            headers: HeaderMap::new(),
            status_code: StatusCode::OK,
            status_message: b"OK".to_vec(),
        }
    }

    /// Sets the HTTP headers for the `WebResourceResponse`.
    ///
    /// # Arguments
    /// * `headers` - The `HeaderMap` to set.
    ///
    /// Post-condition: The response's headers are updated.
    /// Returns `self` for chaining.
    pub fn headers(mut self, headers: HeaderMap) -> WebResourceResponse {
        self.headers = headers;
        self
    }

    /// Sets the HTTP status code for the `WebResourceResponse`.
    ///
    /// # Arguments
    /// * `status_code` - The `StatusCode` to set.
    ///
    /// Post-condition: The response's status code is updated.
    /// Returns `self` for chaining.
    pub fn status_code(mut self, status_code: StatusCode) -> WebResourceResponse {
        self.status_code = status_code;
        self
    }

    /// Sets the HTTP status message for the `WebResourceResponse`.
    ///
    /// # Arguments
    /// * `status_message` - The status message bytes.
    ///
    /// Post-condition: The response's status message is updated.
    /// Returns `self` for chaining.
    pub fn status_message(mut self, status_message: Vec<u8>) -> WebResourceResponse {
        self.status_message = status_message;
        self
    }
}

/// `TraversalDirection` specifies the direction and amount for navigating through session history.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum TraversalDirection {
    /// Travel forward the given number of documents in history.
    Forward(usize),
    /// Travel backward the given number of documents in history.
    Back(usize),
}

/// `Theme` represents the type of platform theme preference (e.g., light or dark mode).
#[derive(Clone, Copy, Debug, Deserialize, Eq, MallocSizeOf, PartialEq, Serialize)]
pub enum Theme {
    /// Light theme preference.
    Light,
    /// Dark theme preference.
    Dark,
}
// The type of MediaSession action.
/// `MediaSessionActionType` enumerates actions that can be performed on a media session.
///
/// <https://w3c.github.io/mediasession/#enumdef-mediasessionaction>
#[derive(Clone, Debug, Deserialize, Eq, Hash, MallocSizeOf, PartialEq, Serialize)]
pub enum MediaSessionActionType {
    /// The action intent is to resume playback.
    Play,
    /// The action intent is to pause the currently active playback.
    Pause,
    /// The action intent is to move the playback time backward by a short period (i.e. a few
    /// seconds).
    SeekBackward,
    /// The action intent is to move the playback time forward by a short period (i.e. a few
    /// seconds).
    SeekForward,
    /// The action intent is to either start the current playback from the beginning if the
    /// playback has a notion, of beginning, or move to the previous item in the playlist if the
    /// playback has a notion of playlist.
    PreviousTrack,
    /// The action is to move to the playback to the next item in the playlist if the playback has
    /// a notion of playlist.
    NextTrack,
    /// The action intent is to skip the advertisement that is currently playing.
    SkipAd,
    /// The action intent is to stop the playback and clear the state if appropriate.
    Stop,
    /// The action intent is to move the playback time to a specific time.
    SeekTo,
}

/// `LoadStatus` describes the current loading status of a `WebView`'s document.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum LoadStatus {
    /// The load has started, but the headers have not yet been parsed.
    Started,
    /// The `<head>` tag has been parsed in the currently loading page. At this point the page's
    /// `HTMLBodyElement` is now available in the DOM.
    HeadParsed,
    /// The `Document` and all subresources have loaded. This is equivalent to
    /// `document.readyState` == `complete`.
    /// See <https://developer.mozilla.org/en-US/docs/Web/API/Document/readyState>
    Complete,
}
