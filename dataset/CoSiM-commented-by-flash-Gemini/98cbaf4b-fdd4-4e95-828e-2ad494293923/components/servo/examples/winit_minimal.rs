/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! This module provides a minimal example of how to embed Servo into a `winit`-based
//! application. It demonstrates the basic setup of a `winit` event loop, creation
//! of a window, initialization of Servo, and the integration of a `WebView` for
//! rendering web content. The example handles fundamental window events such as
//! closing, redrawing, and mouse wheel input, showcasing the interaction between
//! the `winit` windowing system and the Servo rendering engine.

use std::cell::{Cell, RefCell};
use std::error::Error;
use std::rc::Rc;

use compositing::windowing::{AnimationState, EmbedderMethods, WindowMethods};
use euclid::{Point2D, Scale, Size2D};
use servo::{RenderingContext, Servo, TouchEventType, WebView, WindowRenderingContext};
use servo_geometry::DeviceIndependentPixel;
use tracing::warn;
use url::Url;
use webrender_api::units::{DeviceIntPoint, DeviceIntRect, DevicePixel, LayoutVector2D};
use webrender_api::ScrollLocation;
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{MouseScrollDelta, WindowEvent};
use winit::event_loop::EventLoop;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

/// Main entry point for the `winit_minimal` example.
///
/// This function initializes the Rustls crypto provider, sets up the `winit` event loop,
/// creates the application instance, runs the event loop, and deinitializes Servo
/// upon application exit.
///
/// Pre-condition: `rustls::crypto::aws_lc_rs::default_provider()` is available.
/// Post-condition: The application runs until exit, and Servo resources are cleaned up.
fn main() -> Result<(), Box<dyn Error>> {
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .expect("Failed to install crypto provider");

    let event_loop = EventLoop::with_user_event()
        .build()
        .expect("Failed to create EventLoop");
    let mut app = App::new(&event_loop);
    event_loop.run_app(&mut app)?;

    // Block Logic: Deinitializes Servo if the application state is `Running`.
    // Invariant: Ensures that Servo resources are properly released when the application exits.
    if let App::Running(state) = app {
        if let Some(state) = Rc::into_inner(state) {
            state.servo.deinit();
        }
    }

    Ok(())
}

/// `AppState` holds the core components of the application, including the Servo instance,
/// rendering context, and managed WebViews.
struct AppState {
    /// The `WindowDelegate` responsible for handling window-related events.
    window_delegate: Rc<WindowDelegate>,
    /// The main Servo engine instance.
    servo: Servo,
    /// The rendering context used for displaying WebView content.
    rendering_context: Rc<WindowRenderingContext>,
    /// A mutable cell containing a vector of `WebView` instances managed by this application.
    webviews: RefCell<Vec<WebView>>,
}

impl ::servo::WebViewDelegate for AppState {
    /// Notifies the delegate that a `WebView` is ready to be shown.
    ///
    /// # Arguments
    /// * `webview` - The `WebView` instance that is ready.
    ///
    /// Post-condition: The webview is focused, moved/resized to fill the viewport,
    /// and brought to the top.
    fn notify_ready_to_show(&self, webview: WebView) {
        let rect = self
            .window_delegate
            .get_coordinates()
            .get_viewport()
            .to_f32();
        webview.focus();
        webview.move_resize(rect);
        webview.raise_to_top(true);
    }

    /// Notifies the delegate that a new frame is ready for rendering.
    ///
    /// # Arguments
    /// * `_` - The `WebView` instance (unused).
    ///
    /// Post-condition: The `winit` window is requested to redraw.
    fn notify_new_frame_ready(&self, _: WebView) {
        self.window_delegate.window.request_redraw();
    }

    /// Requests to open a new auxiliary `WebView`.
    ///
    /// # Arguments
    /// * `parent_webview` - The parent `WebView` from which the request originated.
    ///
    /// Post-condition: A new auxiliary `WebView` is created, its delegate is set
    /// to the parent's delegate, and it's added to the list of managed webviews.
    /// Returns `Some(WebView)` if successful, `None` otherwise.
    fn request_open_auxiliary_webview(&self, parent_webview: WebView) -> Option<WebView> {
        let webview = self.servo.new_auxiliary_webview();
        webview.set_delegate(parent_webview.delegate());
        self.webviews.borrow_mut().push(webview.clone());
        Some(webview)
    }
}

/// `App` represents the state of the `winit` application, transitioning from
/// an `Initial` state to a `Running` state after initialization.
enum App {
    /// Initial state before the `winit` event loop has resumed.
    Initial(Waker),
    /// Running state after `winit` has resumed, holding the `AppState`.
    Running(Rc<AppState>),
}

impl App {
    /// Creates a new `App` instance in its `Initial` state.
    ///
    /// # Arguments
    /// * `event_loop` - A reference to the `winit` `EventLoop`.
    ///
    /// Post-condition: A new `App::Initial` instance is returned.
    fn new(event_loop: &EventLoop<WakerEvent>) -> Self {
        Self::Initial(Waker::new(event_loop))
    }
}

impl ApplicationHandler<WakerEvent> for App {
    /// Handles the `Resumed` event from `winit`, performing application initialization.
    ///
    /// # Arguments
    /// * `event_loop` - The active `winit` `EventLoop`.
    ///
    /// Pre-condition: The application is in the `Initial` state.
    /// Post-condition: Initializes the window, rendering context, Servo, and creates
    /// the initial `WebView`, transitioning the application to the `Running` state.
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // Block Logic: Initializes the application if it's in the `Initial` state.
        // Invariant: Ensures that `Servo`, `RenderingContext`, and the initial `WebView` are
        //            set up once the event loop resumes.
        if let Self::Initial(waker) = self {
            let display_handle = event_loop
                .display_handle()
                .expect("Failed to get display handle");
            let window = event_loop
                .create_window(Window::default_attributes())
                .expect("Failed to create winit Window");
            let window_handle = window.window_handle().expect("Failed to get window handle");

            let rendering_context = Rc::new(
                WindowRenderingContext::new(display_handle, window_handle, &window.inner_size())
                    .expect("Could not create RenderingContext for window."),
            );
            let window_delegate = Rc::new(WindowDelegate::new(window));

            let _ = rendering_context.make_current();

            let servo = Servo::new(
                Default::default(),
                Default::default(),
                rendering_context.clone(),
                Box::new(EmbedderDelegate {
                    waker: waker.clone(),
                }),
                window_delegate.clone(),
                Default::default(),
            );
            servo.setup_logging();

            let app_state = Rc::new(AppState {
                window_delegate,
                servo,
                rendering_context,
                webviews: Default::default(),
            });

            // Make a new WebView and assign the `AppState` as the delegate.
            let url = Url::parse("https://demo.servo.org/experiments/twgl-tunnel/")
                .expect("Guaranteed by argument");
            let webview = app_state.servo.new_webview(url);
            webview.set_delegate(app_state.clone());
            app_state.webviews.borrow_mut().push(webview);

            *self = Self::Running(app_state);
        }
    }

    /// Handles user-defined events.
    ///
    /// # Arguments
    /// * `_event_loop` - The active `winit` `EventLoop` (unused).
    /// * `_event` - The `WakerEvent` (unused).
    ///
    /// Post-condition: The Servo event loop is spun.
    fn user_event(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop, _event: WakerEvent) {
        // Block Logic: Spins the Servo event loop.
        // Invariant: Ensures Servo processes its internal messages and updates.
        if let Self::Running(state) = self {
            state.servo.spin_event_loop();
        }
    }

    /// Handles various window events from `winit`.
    ///
    /// # Arguments
    /// * `event_loop` - The active `winit` `EventLoop`.
    /// * `_window_id` - The ID of the window that generated the event (unused).
    /// * `event` - The `WindowEvent` to process.
    ///
    /// Post-condition: Processes window events like `CloseRequested`, `RedrawRequested`,
    /// `MouseWheel`, and `KeyboardInput`, affecting the application state and Servo's behavior.
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        // Block Logic: Spins the Servo event loop before handling specific window events.
        // Invariant: Ensures Servo processes any pending events before responding to window-specific
        //            interactions.
        if let Self::Running(state) = self {
            state.servo.spin_event_loop();
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            },
            WindowEvent::RedrawRequested => {
                // Block Logic: Triggers a paint operation for the last WebView and presents the rendering context.
                // Invariant: Ensures the WebView's content is rendered and displayed when a redraw is requested.
                if let Self::Running(state) = self {
                    state.webviews.borrow().last().unwrap().paint();
                    state.rendering_context.present();
                }
            },
            WindowEvent::MouseWheel { delta, .. } => {
                // Block Logic: Processes mouse wheel events, converting them into scroll events for the WebView.
                // Invariant: The last active WebView receives the scroll event.
                if let Self::Running(state) = self {
                    if let Some(webview) = state.webviews.borrow().last() {
                        let moved_by = match delta {
                            MouseScrollDelta::LineDelta(horizontal, vertical) => {
                                LayoutVector2D::new(20. * horizontal, 20. * vertical)
                            },
                            MouseScrollDelta::PixelDelta(pos) => {
                                LayoutVector2D::new(pos.x as f32, pos.y as f32)
                            },
                        };
                        webview.notify_scroll_event(
                            ScrollLocation::Delta(moved_by),
                            DeviceIntPoint::new(10, 10),
                            TouchEventType::Down,
                        );
                    }
                }
            },
            WindowEvent::KeyboardInput { event, .. } => {
                // Block Logic: Handles keyboard input, specifically the 'q' key for closing WebViews.
                // Invariant: If 'q' is pressed, the last WebView is closed, or the application exits if none remain.
                if event.logical_key.to_text() == Some("q") {
                    if let Self::Running(state) = self {
                        let _ = state.webviews.borrow_mut().pop();
                        match state.webviews.borrow().last() {
                            Some(last) => last.show(true),
                            None => event_loop.exit(),
                        }
                    }
                }
            },
            _ => (),
        }
    }
}

/// `EmbedderDelegate` implements `EmbedderMethods` to bridge `winit`'s event loop
/// waker to Servo's embedder API.
struct EmbedderDelegate {
    /// A `Waker` instance used to wake up the `winit` event loop.
    waker: Waker,
}

impl EmbedderMethods for EmbedderDelegate {
    /// Creates an `EventLoopWaker` suitable for the `winit` event loop.
    ///
    /// Post-condition: A `Box<dyn embedder_traits::EventLoopWaker>` is returned,
    /// allowing Servo to request event loop processing.
    fn create_event_loop_waker(&mut self) -> Box<dyn embedder_traits::EventLoopWaker> {
        Box::new(self.waker.clone())
    }
}

/// `Waker` is a wrapper around `winit::event_loop::EventLoopProxy` to implement
/// `embedder_traits::EventLoopWaker`, enabling Servo to wake up the `winit` event loop.
#[derive(Clone)]
struct Waker(winit::event_loop::EventLoopProxy<WakerEvent>);
/// A marker struct representing a user-defined event for `winit`.
#[derive(Debug)]
struct WakerEvent;

impl Waker {
    /// Creates a new `Waker` from a `winit` `EventLoop`.
    ///
    /// # Arguments
    /// * `event_loop` - A reference to the `winit` `EventLoop`.
    ///
    /// Post-condition: A new `Waker` instance is returned.
    fn new(event_loop: &EventLoop<WakerEvent>) -> Self {
        Self(event_loop.create_proxy())
    }
}

impl embedder_traits::EventLoopWaker for Waker {
    /// Clones the `Waker` into a boxed trait object.
    ///
    /// Post-condition: A new `Box<dyn embedder_traits::EventLoopWaker>` is returned.
    fn clone_box(&self) -> Box<dyn embedder_traits::EventLoopWaker> {
        Box::new(Self(self.0.clone()))
    }

    /// Wakes up the `winit` event loop by sending a `WakerEvent`.
    ///
    /// Post-condition: A `WakerEvent` is sent to the event loop, or a warning is logged
    /// if sending fails.
    fn wake(&self) {
        if let Err(error) = self.0.send_event(WakerEvent) {
            warn!(?error, "Failed to wake event loop");
        }
    }
}

/// `WindowDelegate` implements `WindowMethods` to provide Servo with information
/// about the `winit` window and to control its animation state.
struct WindowDelegate {
    /// The `winit` window instance.
    window: Window,
    /// The current animation state of the window.
    animation_state: Cell<AnimationState>,
}

impl WindowDelegate {
    /// Creates a new `WindowDelegate` instance.
    ///
    /// # Arguments
    /// * `window` - The `winit` window to delegate.
    ///
    /// Post-condition: A new `WindowDelegate` is returned with an `Idle` animation state.
    fn new(window: Window) -> Self {
        Self {
            window,
            animation_state: Cell::new(AnimationState::Idle),
        }
    }
}

impl WindowMethods for WindowDelegate {
    /// Retrieves the embedder-specific coordinates and sizing information for the window.
    ///
    /// Post-condition: An `EmbedderCoordinates` struct is returned, containing information
    /// about the window's dimensions, HIDPI factor, screen size, and viewport.
    fn get_coordinates(&self) -> compositing::windowing::EmbedderCoordinates {
        let monitor = self
            .window
            .current_monitor()
            .or_else(|| self.window.available_monitors().nth(0))
            .expect("Failed to get winit monitor");
        let scale =
            Scale::<f64, DeviceIndependentPixel, DevicePixel>::new(self.window.scale_factor());
        let window_size = winit_size_to_euclid_size(self.window.outer_size()).to_i32();
        let window_origin = self.window.outer_position().unwrap_or_default();
        let window_origin = winit_position_to_euclid_point(window_origin).to_i32();
        let window_rect = DeviceIntRect::from_origin_and_size(window_origin, window_size);
        let viewport_origin = DeviceIntPoint::zero(); // bottom left
        let viewport_size = winit_size_to_euclid_size(self.window.inner_size()).to_f32();
        let viewport = DeviceIntRect::from_origin_and_size(viewport_origin, viewport_size.to_i32());

        compositing::windowing::EmbedderCoordinates {
            hidpi_factor: Scale::new(self.window.scale_factor() as f32),
            screen_size: (winit_size_to_euclid_size(monitor.size()).to_f64() / scale).to_i32(),
            available_screen_size: (winit_size_to_euclid_size(monitor.size()).to_f64() / scale)
                .to_i32(),
            window_rect: (window_rect.to_f64() / scale).to_i32(),
            framebuffer: viewport.size(),
            viewport,
        }
    }

    /// Sets the animation state of the window.
    ///
    /// # Arguments
    /// * `state` - The new `AnimationState` for the window.
    ///
    /// Post-condition: The internal `animation_state` is updated.
    fn set_animation_state(&self, state: compositing::windowing::AnimationState) {
        self.animation_state.set(state);
    }
}

/// Converts a `winit` `PhysicalSize` to an `euclid` `Size2D`.
///
/// # Type Parameters
/// * `T` - The numeric type of the size components.
///
/// # Arguments
/// * `size` - The `winit::dpi::PhysicalSize` to convert.
///
/// Post-condition: An `euclid::Size2D` with `DevicePixel` units is returned.
pub fn winit_size_to_euclid_size<T>(size: PhysicalSize<T>) -> Size2D<T, DevicePixel> {
    Size2D::new(size.width, size.height)
}

/// Converts a `winit` `PhysicalPosition` to an `euclid` `Point2D`.
///
/// # Type Parameters
/// * `T` - The numeric type of the position components.
///
/// # Arguments
/// * `position` - The `winit::dpi::PhysicalPosition` to convert.
///
/// Post-condition: An `euclid::Point2D` with `DevicePixel` units is returned.
pub fn winit_position_to_euclid_point<T>(position: PhysicalPosition<T>) -> Point2D<T, DevicePixel> {
    Point2D::new(position.x, position.y)
}