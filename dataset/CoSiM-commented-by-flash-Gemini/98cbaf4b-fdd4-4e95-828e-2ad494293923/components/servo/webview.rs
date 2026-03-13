/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! @3f57f4bb-7c2b-4797-ab43-b75f5f704df1/components/servo/webview.rs
//! @brief Handle to a Servo webview instance.
//!
//! This module defines the `WebView` struct, which serves as a primary handle
//! for interacting with a single webview instance within Servo. It provides
//! a high-level API for managing the webview's lifecycle, controlling its
//! content (e.g., loading URLs), handling user input events, and coordinating
//! with the underlying rendering engine (`compositor`) and the central
//! coordination unit (`constellation`).
//!
//! The `WebView` is designed to be cloneable, where cloning creates a new handle
//! to the *same* webview, not a new webview instance. Resources associated with a
//! webview are only cleaned up when the last handle to it is dropped.
//!
//! ## Rendering Model
//!
//! Every [`WebView`] has an associated [`RenderingContext`](crate::RenderingContext).
//! The embedding application (`embedder`) is responsible for managing when the
//! contents of the [`WebView`] are painted to this context. When a [`WebView`] needs
//! to be updated (e.g., due to content changes), Servo signals this by calling
//! [`WebViewDelegate::notify_new_frame_ready`]. The application should then
//! repaint the `WebView` by calling [`WebView::paint`].
//!
//! An example of the rendering flow:
//!
//! 1. [`WebViewDelegate::notify_new_frame_ready`] is invoked, indicating new content.
//!    The application should respond by queuing a repaint for the window containing this `WebView`.
//! 2. During the window's repaint cycle, the application calls [`WebView::paint`]. This
//!    updates the contents of the `RenderingContext` associated with the `WebView`.
//! 3. If the `RenderingContext` is double-buffered, the application then calls
//!    [`crate::RenderingContext::present()`] to swap the back buffer to the front,
//!    making the updated `WebView` contents visible to the user.
//!
//! Even if the `WebView` contents haven't changed, a repaint might be necessary (e.g.,
//! window damage). In such cases, the application can directly call steps 2 and 3;
//! Servo will still repaint without a prior `notify_new_frame_ready` call.

use std::cell::{Ref, RefCell, RefMut};
use std::hash::Hash;
use std::rc::{Rc, Weak};
use std::time::Duration;

use base::id::WebViewId;
use compositing::windowing::WebRenderDebugOption;
use compositing::IOCompositor;
use compositing_traits::ConstellationMsg;
use embedder_traits::{
    Cursor, InputEvent, LoadStatus, MediaSessionActionType, Theme, TouchEventType,
    TraversalDirection,
};
use url::Url;
use webrender_api::units::{DeviceIntPoint, DeviceRect};
use webrender_api::ScrollLocation;

use crate::clipboard_delegate::{ClipboardDelegate, DefaultClipboardDelegate};
use crate::webview_delegate::{DefaultWebViewDelegate, WebViewDelegate};
use crate::ConstellationProxy;

/// A handle to a Servo webview. If you clone this handle, it does not create a new webview,
/// but instead creates a new handle to the webview. Once the last handle is dropped, Servo
/// considers that the webview has closed and will clean up all associated resources related
/// to this webview.
///
/// ## Rendering Model
///
/// Every [`WebView`] has a [`RenderingContext`](crate::RenderingContext). The embedder manages when
/// the contents of the [`WebView`] paint to the [`RenderingContext`](crate::RenderingContext). When
/// a [`WebView`] needs to be painted, for instance, because its contents have changed, Servo will
/// call [`WebViewDelegate::notify_new_frame_ready`] in order to signal that it is time to repaint
/// the [`WebView`] using [`WebView::paint`].
///
/// An example of how this flow might work is:
///
/// 1. [`WebViewDelegate::notify_new_frame_ready`] is called. The applications triggers a request
///    to repaint the window that contains this [`WebView`].
/// 2. During window repainting, the application calls [`WebView::paint`] and the contents of the
///    [`RenderingContext`][crate::RenderingContext] are updated.
/// 3. If the [`RenderingContext`][crate::RenderingContext] is double-buffered, the
///    application then calls [`crate::RenderingContext::present()`] in order to swap the back buffer
///    to the front, finally displaying the updated [`WebView`] contents.
///
/// In cases where the [`WebView`] contents have not been updated, but a repaint is necessary, for
/// instance when repainting a window due to damage, an application may simply perform the final two
/// steps and Servo will repaint even without first calling the
/// [`WebViewDelegate::notify_new_frame_ready`] method.
#[derive(Clone)]
pub struct WebView(Rc<RefCell<WebViewInner>>);

impl PartialEq for WebView {
    /// Compares two `WebView` instances for equality based on their internal `WebViewId`.
    ///
    /// # Arguments
    /// * `other` - The other `WebView` instance to compare against.
    ///
    /// Post-condition: Returns `true` if both `WebView`s refer to the same underlying
    /// webview instance, `false` otherwise.
    fn eq(&self, other: &Self) -> bool {
        self.inner().id == other.inner().id
    }
}

impl Hash for WebView {
    /// Hashes the `WebView` based on its internal `WebViewId`.
    ///
    /// # Arguments
    /// * `state` - The hasher to feed the `WebViewId` into.
    ///
    /// Post-condition: The `WebViewId` is hashed, ensuring consistent hashing
    /// for identical `WebView` instances.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner().id.hash(state);
    }
}

/// `WebViewInner` holds the internal state and core components of a single webview instance.
pub(crate) struct WebViewInner {
    // TODO: ensure that WebView instances interact with the correct Servo instance
    /// The unique identifier for this webview.
    pub(crate) id: WebViewId,
    /// A proxy for communicating with the `Constellation`.
    pub(crate) constellation_proxy: ConstellationProxy,
    /// A reference to the `IOCompositor` responsible for rendering this webview.
    pub(crate) compositor: Rc<RefCell<IOCompositor>>,
    /// The delegate for handling webview-specific events and requests.
    pub(crate) delegate: Rc<dyn WebViewDelegate>,
    /// The delegate for handling clipboard operations.
    pub(crate) clipboard_delegate: Rc<dyn ClipboardDelegate>,

    /// The current rectangular bounds of the webview in device coordinates.
    rect: DeviceRect,
    /// The current loading status of the webview's document.
    load_status: LoadStatus,
    /// The URL of the currently loaded document.
    url: Option<Url>,
    /// The status text displayed for the webview.
    status_text: Option<String>,
    /// The title of the currently loaded web page.
    page_title: Option<String>,
    /// The URL of the favicon for the currently loaded web page.
    favicon_url: Option<Url>,
    /// A boolean indicating whether the webview is currently focused.
    focused: bool,
    /// The current mouse cursor style for the webview.
    cursor: Cursor,
}

impl Drop for WebViewInner {
    /// Sends a `CloseWebView` message to the `Constellation` when the `WebViewInner` is dropped.
    ///
    /// Post-condition: The `Constellation` is notified that this webview should be closed,
    /// triggering cleanup of associated resources.
    fn drop(&mut self) {
        self.constellation_proxy
            .send(ConstellationMsg::CloseWebView(self.id));
    }
}

impl WebView {
    /// Creates a new `WebView` instance.
    ///
    /// # Arguments
    /// * `constellation_proxy` - A reference to the `ConstellationProxy` for communication.
    /// * `compositor` - A reference-counted, mutable cell containing the `IOCompositor`.
    ///
    /// Post-condition: A new `WebView` handle is returned, initialized with default values
    /// and connected to the `Constellation` and `Compositor`.
    pub(crate) fn new(
        constellation_proxy: &ConstellationProxy,
        compositor: Rc<RefCell<IOCompositor>>,
    ) -> Self {
        Self(Rc::new(RefCell::new(WebViewInner {
            id: WebViewId::new(),
            constellation_proxy: constellation_proxy.clone(),
            compositor,
            delegate: Rc::new(DefaultWebViewDelegate),
            clipboard_delegate: Rc::new(DefaultClipboardDelegate),
            rect: DeviceRect::zero(),
            load_status: LoadStatus::Complete,
            url: None,
            status_text: None,
            page_title: None,
            favicon_url: None,
            focused: false,
            cursor: Cursor::Pointer,
        })))
    }

    /// Returns an immutable reference to the internal `WebViewInner` data.
    ///
    /// Post-condition: A `Ref<'_, WebViewInner>` is returned, allowing read-only access
    /// to the webview's internal state.
    fn inner(&self) -> Ref<'_, WebViewInner> {
        self.0.borrow()
    }

    /// Returns a mutable reference to the internal `WebViewInner` data.
    ///
    /// Post-condition: A `RefMut<'_, WebViewInner>` is returned, allowing mutable access
    /// to the webview's internal state.
    fn inner_mut(&self) -> RefMut<'_, WebViewInner> {
        self.0.borrow_mut()
    }

    /// Creates a `WebView` handle from a weak reference to its `WebViewInner`.
    ///
    /// # Arguments
    /// * `inner` - A `Weak<RefCell<WebViewInner>>` handle.
    ///
    /// Post-condition: An `Option<Self>` is returned, which is `Some` if the weak reference
    /// can be upgraded (i.e., the webview is still alive), and `None` otherwise.
    pub(crate) fn from_weak_handle(inner: &Weak<RefCell<WebViewInner>>) -> Option<Self> {
        inner.upgrade().map(WebView)
    }

    /// Returns a weak reference to the internal `WebViewInner` data.
    ///
    /// Post-condition: A `Weak<RefCell<WebViewInner>>` is returned, which can be
    /// used to check the liveness of the `WebView` without extending its lifetime.
    pub(crate) fn weak_handle(&self) -> Weak<RefCell<WebViewInner>> {
        Rc::downgrade(&self.0)
    }

    /// Returns a reference-counted instance of the current `WebViewDelegate`.
    ///
    /// Post-condition: A `Rc<dyn WebViewDelegate>` is returned, allowing interaction
    /// with the embedding application's specific implementations for this webview.
    pub fn delegate(&self) -> Rc<dyn WebViewDelegate> {
        self.inner().delegate.clone()
    }

    /// Sets a new delegate for this `WebView`.
    ///
    /// # Arguments
    /// * `delegate` - The new `Rc<dyn WebViewDelegate>` to set.
    ///
    /// Post-condition: The internal delegate is updated to the provided one.
    pub fn set_delegate(&self, delegate: Rc<dyn WebViewDelegate>) {
        self.inner_mut().delegate = delegate;
    }

    /// Returns a reference-counted instance of the current `ClipboardDelegate`.
    ///
    /// Post-condition: A `Rc<dyn ClipboardDelegate>` is returned, allowing interaction
    /// with the embedding application's specific implementations for clipboard operations.
    pub fn clipboard_delegate(&self) -> Rc<dyn ClipboardDelegate> {
        self.inner().clipboard_delegate.clone()
    }

    /// Sets a new clipboard delegate for this `WebView`.
    ///
    /// # Arguments
    /// * `delegate` - The new `Rc<dyn ClipboardDelegate>` to set.
    ///
    /// Post-condition: The internal clipboard delegate is updated to the provided one.
    pub fn set_clipboard_delegate(&self, delegate: Rc<dyn ClipboardDelegate>) {
        self.inner_mut().clipboard_delegate = delegate;
    }

    /// Returns the unique identifier of this `WebView`.
    ///
    /// Post-condition: The `WebViewId` of this instance is returned.
    pub fn id(&self) -> WebViewId {
        self.inner().id
    }

    /// Returns the current loading status of the webview's document.
    ///
    /// Post-condition: A `LoadStatus` enum indicating the current state of the page load.
    pub fn load_status(&self) -> LoadStatus {
        self.inner().load_status
    }

    /// Sets the loading status of the webview and notifies its delegate if the status changes.
    ///
    /// # Arguments
    /// * `new_value` - The new `LoadStatus` for the webview.
    ///
    /// Pre-condition: `new_value` is a valid `LoadStatus`.
    /// Post-condition: The internal `load_status` is updated, and `notify_load_status_changed`
    /// is called on the delegate if the status has changed.
    pub(crate) fn set_load_status(self, new_value: LoadStatus) {
        if self.inner().load_status == new_value {
            return;
        }
        self.inner_mut().load_status = new_value;
        self.delegate().notify_load_status_changed(&self, new_value);
    }

    /// Returns the URL of the currently loaded document in this `WebView`.
    ///
    /// Post-condition: An `Option<Url>` is returned, which is `Some` if a URL is loaded,
    /// and `None` otherwise.
    pub fn url(&self) -> Option<Url> {
        self.inner().url.clone()
    }

    /// Sets the URL of the webview and notifies its delegate if the URL changes.
    ///
    /// # Arguments
    /// * `new_value` - The new `Url` for the webview.
    ///
    /// Pre-condition: `new_value` is a valid `Url`.
    /// Post-condition: The internal `url` is updated, and `notify_url_changed` is called
    /// on the delegate if the URL has changed.
    pub(crate) fn set_url(self, new_value: Url) {
        if self
            .inner()
            .url
            .as_ref()
            .is_some_and(|url| url == &new_value)
        {
            return;
        }
        self.inner_mut().url = Some(new_value.clone());
        self.delegate().notify_url_changed(&self, new_value);
    }

    /// Returns the status text displayed for this `WebView`.
    ///
    /// Post-condition: An `Option<String>` is returned, containing the status text if set,
    /// or `None` otherwise.
    pub fn status_text(&self) -> Option<String> {
        self.inner().status_text.clone()
    }

    /// Sets the status text for the webview and notifies its delegate if the text changes.
    ///
    /// # Arguments
    /// * `new_value` - The new status `String` for the webview.
    ///
    /// Pre-condition: `new_value` is a valid `Option<String>`.
    /// Post-condition: The internal `status_text` is updated, and `notify_status_text_changed`
    /// is called on the delegate if the status text has changed.
    pub(crate) fn set_status_text(self, new_value: Option<String>) {
        if self.inner().status_text == new_value {
            return;
        }
        self.inner_mut().status_text = new_value.clone();
        self.delegate().notify_status_text_changed(&self, new_value);
    }

    /// Returns the page title of the currently loaded document.
    ///
    /// Post-condition: An `Option<String>` is returned, containing the page title if set,
    /// or `None` otherwise.
    pub fn page_title(&self) -> Option<String> {
        self.inner().page_title.clone()
    }

    /// Sets the page title for the webview and notifies its delegate if the title changes.
    ///
    /// # Arguments
    /// * `new_value` - The new page title `String` for the webview.
    ///
    /// Pre-condition: `new_value` is a valid `Option<String>`.
    /// Post-condition: The internal `page_title` is updated, and `notify_page_title_changed`
    /// is called on the delegate if the page title has changed.
    pub(crate) fn set_page_title(self, new_value: Option<String>) {
        if self.inner().page_title == new_value {
            return;
        }
        self.inner_mut().page_title = new_value.clone();
        self.delegate().notify_page_title_changed(&self, new_value);
    }

    /// Returns the favicon URL of the currently loaded document.
    ///
    /// Post-condition: An `Option<Url>` is returned, containing the favicon URL if set,
    /// or `None` otherwise.
    pub fn favicon_url(&self) -> Option<Url> {
        self.inner().favicon_url.clone()
    }

    /// Sets the favicon URL for the webview and notifies its delegate if the URL changes.
    ///
    /// # Arguments
    /// * `new_value` - The new favicon `Url` for the webview.
    ///
    /// Pre-condition: `new_value` is a valid `Url`.
    /// Post-condition: The internal `favicon_url` is updated, and `notify_favicon_url_changed`
    /// is called on the delegate if the favicon URL has changed.
    pub(crate) fn set_favicon_url(self, new_value: Url) {
        if self
            .inner()
            .favicon_url
            .as_ref()
            .is_some_and(|url| url == &new_value)
        {
            return;
        }
        self.inner_mut().favicon_url = Some(new_value.clone());
        self.delegate().notify_favicon_url_changed(&self, new_value);
    }

    /// Returns `true` if this `WebView` is currently focused, `false` otherwise.
    ///
    /// Post-condition: A boolean indicating the focus state of the webview is returned.
    pub fn focused(&self) -> bool {
        self.inner().focused
    }

    /// Sets the focus state of the webview and notifies its delegate if the state changes.
    ///
    /// # Arguments
    /// * `new_value` - `true` to focus the webview, `false` to blur it.
    ///
    /// Pre-condition: `new_value` is a boolean.
    /// Post-condition: The internal `focused` state is updated, and `notify_focus_changed`
    /// is called on the delegate if the focus state has changed.
    pub(crate) fn set_focused(self, new_value: bool) {
        if self.inner().focused == new_value {
            return;
        }
        self.inner_mut().focused = new_value;
        self.delegate().notify_focus_changed(&self, new_value);
    }

    /// Returns the current mouse cursor style for this `WebView`.
    ///
    /// Post-condition: A `Cursor` enum representing the current cursor style.
    pub fn cursor(&self) -> Cursor {
        self.inner().cursor
    }

    /// Sets the mouse cursor style for the webview and notifies its delegate if the style changes.
    ///
    /// # Arguments
    /// * `new_value` - The new `Cursor` style for the webview.
    ///
    /// Pre-condition: `new_value` is a valid `Cursor` enum.
    /// Post-condition: The internal `cursor` style is updated, and `notify_cursor_changed`
    /// is called on the delegate if the cursor style has changed.
    pub(crate) fn set_cursor(self, new_value: Cursor) {
        if self.inner().cursor == new_value {
            return;
        }
        self.inner_mut().cursor = new_value;
        self.delegate().notify_cursor_changed(&self, new_value);
    }

    /// Requests that this `WebView` be focused.
    ///
    /// Post-condition: A `ConstellationMsg::FocusWebView` message is sent to the `Constellation`,
    /// requesting that this webview be brought into focus.
    pub fn focus(&self) {
        self.inner()
            .constellation_proxy
            .send(ConstellationMsg::FocusWebView(self.id()));
    }

    /// Requests that this `WebView` be blurred (unfocused).
    ///
    /// Post-condition: A `ConstellationMsg::BlurWebView` message is sent to the `Constellation`,
    /// requesting that the currently focused webview be blurred.
    pub fn blur(&self) {
        self.inner()
            .constellation_proxy
            .send(ConstellationMsg::BlurWebView);
    }

    /// Returns the current rectangular bounds of the webview in device coordinates.
    ///
    /// Post-condition: A `DeviceRect` representing the position and size of the webview.
    pub fn rect(&self) -> DeviceRect {
        self.inner().rect
    }

    /// Moves and/or resizes the `WebView` to the specified `rect`.
    ///
    /// # Arguments
    /// * `rect` - The new `DeviceRect` for the webview.
    ///
    /// Pre-condition: `rect` is a valid `DeviceRect`.
    /// Post-condition: The internal `rect` is updated, and a `move_resize_webview`
    /// command is sent to the compositor if the rectangle has changed.
    pub fn move_resize(&self, rect: DeviceRect) {
        if self.inner().rect == rect {
            return;
        }

        self.inner_mut().rect = rect;
        self.inner()
            .compositor
            .borrow_mut()
            .move_resize_webview(self.id(), rect);
    }

    /// Makes this `WebView` visible.
    ///
    /// # Arguments
    /// * `hide_others` - If `true`, all other webviews will be hidden.
    ///
    /// Post-condition: The `WebView` is made visible, and other webviews may be hidden.
    /// Panics if the internal `WebView` instance is invalid.
    pub fn show(&self, hide_others: bool) {
        self.inner()
            .compositor
            .borrow_mut()
            .show_webview(self.id(), hide_others)
            .expect("BUG: invalid WebView instance");
    }

    /// Hides this `WebView`.
    ///
    /// Post-condition: The `WebView` is hidden.
    /// Panics if the internal `WebView` instance is invalid.
    pub fn hide(&self) {
        self.inner()
            .compositor
            .borrow_mut()
            .hide_webview(self.id())
            .expect("BUG: invalid WebView instance");
    }

    /// Brings this `WebView` to the top of the z-order.
    ///
    /// # Arguments
    /// * `hide_others` - If `true`, all other webviews will be hidden.
    ///
    /// Post-condition: The `WebView` is brought to the top, and other webviews may be hidden.
    /// Panics if the internal `WebView` instance is invalid.
    pub fn raise_to_top(&self, hide_others: bool) {
        self.inner()
            .compositor
            .borrow_mut()
            .raise_webview_to_top(self.id(), hide_others)
            .expect("BUG: invalid WebView instance");
    }

    /// Notifies the `Constellation` of a theme change.
    ///
    /// # Arguments
    /// * `theme` - The new `Theme` to apply.
    ///
    /// Post-condition: A `ConstellationMsg::ThemeChange` message is sent to the `Constellation`.
    pub fn notify_theme_change(&self, theme: Theme) {
        self.inner()
            .constellation_proxy
            .send(ConstellationMsg::ThemeChange(theme))
    }

    /// Loads a new URL into this `WebView`.
    ///
    /// # Arguments
    /// * `url` - The `Url` to load.
    ///
    /// Post-condition: A `ConstellationMsg::LoadUrl` message is sent to the `Constellation`,
    /// initiating the loading of the specified URL in this webview.
    pub fn load(&self, url: Url) {
        self.inner()
            .constellation_proxy
            .send(ConstellationMsg::LoadUrl(self.id(), url.into()))
    }

    /// Reloads the current page in this `WebView`.
    ///
    /// Post-condition: A `ConstellationMsg::Reload` message is sent to the `Constellation`,
    /// requesting a reload of the current document in this webview.
    pub fn reload(&self) {
        self.inner()
            .constellation_proxy
            .send(ConstellationMsg::Reload(self.id()))
    }

    /// Navigates back in the `WebView`'s history by `amount` entries.
    ///
    /// # Arguments
    /// * `amount` - The number of history entries to go back.
    ///
    /// Post-condition: A `ConstellationMsg::TraverseHistory` message with `TraversalDirection::Back`
    /// is sent to the `Constellation`.
    pub fn go_back(&self, amount: usize) {
        self.inner()
            .constellation_proxy
            .send(ConstellationMsg::TraverseHistory(
                self.id(),
                TraversalDirection::Back(amount),
            ))
    }

    /// Navigates forward in the `WebView`'s history by `amount` entries.
    ///
    /// # Arguments
    /// * `amount` - The number of history entries to go forward.
    ///
    /// Post-condition: A `ConstellationMsg::TraverseHistory` message with `TraversalDirection::Forward`
    /// is sent to the `Constellation`.
    pub fn go_forward(&self, amount: usize) {
        self.inner()
            .constellation_proxy
            .send(ConstellationMsg::TraverseHistory(
                self.id(),
                TraversalDirection::Forward(amount),
            ))
    }

    /// Notifies the `WebView` of a scroll event.
    ///
    /// # Arguments
    /// * `location` - The `ScrollLocation` (e.g., delta, start, end).
    /// * `point` - The `DeviceIntPoint` representing the cursor position.
    /// * `touch_event_action` - The `TouchEventType` associated with the scroll.
    ///
    /// Post-condition: The scroll event is forwarded to the compositor.
    pub fn notify_scroll_event(
        &self,
        location: ScrollLocation,
        point: DeviceIntPoint,
        touch_event_action: TouchEventType,
    ) {
        self.inner()
            .compositor
            .borrow_mut()
            .on_scroll_event(location, point, touch_event_action);
    }

    /// Notifies the `WebView` of an input event.
    ///
    /// # Arguments
    /// * `event` - The `InputEvent` to process.
    ///
    /// Pre-condition: `event` is a valid `InputEvent`.
    /// Post-condition: If the event has a point, it's sent to the compositor for hit testing;
    /// otherwise, it's sent directly to the `Constellation`.
    pub fn notify_input_event(&self, event: InputEvent) {
        // Events with a `point` first go to the compositor for hit testing.
        if event.point().is_some() {
            self.inner().compositor.borrow_mut().on_input_event(event);
            return;
        }

        self.inner()
            .constellation_proxy
            .send(ConstellationMsg::ForwardInputEvent(
                event, None, /* hit_test */
            ))
    }

    /// Notifies the `WebView` of a media session action event.
    ///
    /// # Arguments
    /// * `event` - The `MediaSessionActionType` to process.
    ///
    /// Post-condition: A `ConstellationMsg::MediaSessionAction` message is sent to the `Constellation`.
    pub fn notify_media_session_action_event(&self, event: MediaSessionActionType) {
        self.inner()
            .constellation_proxy
            .send(ConstellationMsg::MediaSessionAction(event));
    }

    /// Notifies the `WebView` of a vertical synchronization (vsync) event.
    ///
    /// Post-condition: The `on_vsync` method of the compositor is called,
    /// typically to process any pending fling actions.
    pub fn notify_vsync(&self) {
        self.inner().compositor.borrow_mut().on_vsync();
    }

    /// Notifies the `WebView` that its rendering context has been resized.
    ///
    /// Post-condition: The `on_rendering_context_resized` method of the compositor is called,
    /// triggering necessary rendering updates.
    pub fn notify_rendering_context_resized(&self) {
        self.inner()
            .compositor
            .borrow_mut()
            .on_rendering_context_resized();
    }

    /// Notifies the `WebView` that the embedder window has been moved.
    ///
    /// Post-condition: The `on_embedder_window_moved` method of the compositor is called,
    /// updating its internal coordinate system.
    pub fn notify_embedder_window_moved(&self) {
        self.inner()
            .compositor
            .borrow_mut()
            .on_embedder_window_moved();
    }

    /// Sets the page zoom level for this `WebView`.
    ///
    /// # Arguments
    /// * `new_zoom` - The new zoom factor (e.g., 1.0 for 100%).
    ///
    /// Post-condition: The `on_zoom_window_event` method of the compositor is called,
    /// adjusting the page zoom.
    pub fn set_zoom(&self, new_zoom: f32) {
        self.inner()
            .compositor
            .borrow_mut()
            .on_zoom_window_event(new_zoom);
    }

    /// Resets the page zoom level to its default (100%).
    ///
    /// Post-condition: The `on_zoom_reset_window_event` method of the compositor is called,
    /// resetting the page zoom.
    pub fn reset_zoom(&self) {
        self.inner()
            .compositor
            .borrow_mut()
            .on_zoom_reset_window_event();
    }

    /// Sets the pinch zoom level for this `WebView`.
    ///
    /// # Arguments
    /// * `new_pinch_zoom` - The new pinch zoom factor.
    ///
    /// Post-condition: The `on_pinch_zoom_window_event` method of the compositor is called,
    /// adjusting the pinch zoom.
    pub fn set_pinch_zoom(&self, new_pinch_zoom: f32) {
        self.inner()
            .compositor
            .borrow_mut()
            .on_pinch_zoom_window_event(new_pinch_zoom);
    }

    /// Requests to exit fullscreen mode for this `WebView`.
    ///
    /// Post-condition: A `ConstellationMsg::ExitFullScreen` message is sent to the `Constellation`.
    pub fn exit_fullscreen(&self) {
        self.inner()
            .constellation_proxy
            .send(ConstellationMsg::ExitFullScreen(self.id()));
    }

    /// Sets the throttled state for this `WebView`.
    ///
    /// When throttled, the webview may reduce its activity (e.g., script execution,
    /// rendering updates) to conserve resources.
    ///
    /// # Arguments
    /// * `throttled` - `true` to throttle the webview, `false` to unthrottle.
    ///
    /// Post-condition: A `ConstellationMsg::SetWebViewThrottled` message is sent to the `Constellation`.
    pub fn set_throttled(&self, throttled: bool) {
        self.inner()
            .constellation_proxy
            .send(ConstellationMsg::SetWebViewThrottled(self.id(), throttled));
    }

    /// Toggles WebRender debugging options for this `WebView`.
    ///
    /// # Arguments
    /// * `debugging` - The `WebRenderDebugOption` to toggle.
    ///
    /// Post-condition: The `toggle_webrender_debug` method of the compositor is called.
    pub fn toggle_webrender_debugging(&self, debugging: WebRenderDebugOption) {
        self.inner()
            .compositor
            .borrow_mut()
            .toggle_webrender_debug(debugging);
    }

    /// Captures the current WebRender output.
    ///
    /// Post-condition: The `capture_webrender` method of the compositor is called,
    /// typically triggering a screenshot or debug capture.
    pub fn capture_webrender(&self) {
        self.inner().compositor.borrow_mut().capture_webrender();
    }

    /// Toggles the sampling profiler with a specified rate and maximum duration.
    ///
    /// # Arguments
    /// * `rate` - The sampling interval.
    /// * `max_duration` - The maximum duration for profiling.
    ///
    /// Post-condition: A `ConstellationMsg::ToggleProfiler` message is sent to the `Constellation`.
    pub fn toggle_sampling_profiler(&self, rate: Duration, max_duration: Duration) {
        self.inner()
            .constellation_proxy
            .send(ConstellationMsg::ToggleProfiler(rate, max_duration));
    }

    /// Sends an error message related to this `WebView` to the `Constellation`.
    ///
    /// # Arguments
    /// * `message` - The error message `String`.
    ///
    /// Post-condition: A `ConstellationMsg::SendError` message is sent to the `Constellation`.
    pub fn send_error(&self, message: String) {
        self.inner()
            .constellation_proxy
            .send(ConstellationMsg::SendError(Some(self.id()), message));
    }

    /// Paints the contents of this [`WebView`] into its `RenderingContext`.
    ///
    /// This will always trigger a paint operation, unless the `Opts::wait_for_stable_image`
    /// option is enabled, in which case it might do nothing if the image is not yet stable.
    ///
    /// Post-condition: The `render` method of the compositor is called.
    /// Returns `true` if a paint was actually performed, `false` otherwise.
    pub fn paint(&self) -> bool {
        self.inner().compositor.borrow_mut().render()
    }
}
