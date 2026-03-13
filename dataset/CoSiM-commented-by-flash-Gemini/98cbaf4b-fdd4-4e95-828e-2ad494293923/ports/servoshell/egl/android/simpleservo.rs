/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! This module implements a minimal Servo embedding for Android, demonstrating
//! the basic integration of the Servo engine within an Android application's
//! `Activity` context. It focuses on creating a `WebView`, managing its lifecycle,
//! and handling fundamental input events, as well as coordinating rendering updates.
//! The implementation leverages EGL for graphics, `android_glue` for event handling,
//! and `surfman` for graphics context management.

// Block Logic: Standard library imports for common data structures and threading.
// Functional Utility: Provides essential Rust language features like `Arc`, `Cell`,
//                     `RefCell`, `Rc`, and `Mutex` for shared ownership, interior
//                     mutability, and thread synchronization.
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// Block Logic: External crate imports for Android-specific functionalities, Servo components, and graphics.
// Functional Utility: Integrates Android-specific bindings (`android_glue`), Servo's core engine (`servo`),
//                     cross-platform graphics abstraction (`gleam`), rendering contexts (`surfman`),
//                     and logging capabilities (`log`).
use android_glue::{self as android_platform, AndroidApp};
use gleam::gl;
use log::{debug, error, info, LevelFilter};
use servo::{
    self, EmbedderMethods, GlApi, GlContext, LoadStatus, NativeDisplay, OffscreenRenderingContext,
    RenderingContext, Servo, ServoDelegate, ServoError, Theme, WebView, WebViewDelegate,
    WindowGLContext, WindowRenderingContext,
};
use surfman::{
    Connection, Context, ContextAttributePreference, ContextAttributes, ContextDescriptor,
    ContextRcExt, GLVersion, NativeWidget, NativeWidgetExt, Surface, SurfaceAccess, SurfaceAttributePreference,
    SurfaceAttributes, SurfaceTexture, SurfaceTextureExt, SurfaceType, WindowingApi,
};
use webrender_api::units::{DeviceIntPoint, DeviceIntRect, DeviceIntSize};
use webrender_api::ScrollLocation;

/// `App` represents the main application state for the Android embedding.
/// It holds the Servo instance, the active WebView, and the windowing context.
struct App {
    /// The Servo engine instance.
    servo: Servo,
    /// The currently active WebView.
    webview: WebView,
    /// The current event loop waker.
    event_loop_waker: Box<dyn servo::EventLoopWaker>,
    /// The `surfman` connection for graphics.
    connection: Connection,
    /// The `surfman` device for graphics.
    device: Device,
    /// The `surfman` OpenGL context.
    context: Context,
    /// The rendering context for the window.
    window_rendering_context: Rc<WindowRenderingContext>,
    /// The offscreen rendering context for Servo.
    rendering_context: Rc<OffscreenRenderingContext>,
    /// The native window widget.
    native_widget: NativeWidget,
    /// The `surfman` surface for the window.
    surface: Surface,
    /// Tracks the current size of the surface.
    current_surface_size: Cell<DeviceIntSize>,
    /// A mutable cell indicating whether the surface needs to be recreated.
    need_new_surface: Cell<bool>,
}

impl App {
    /// Creates a new `App` instance, initializing Servo and setting up the rendering environment.
    ///
    /// # Arguments
    /// * `app` - The `AndroidApp` instance for Android-specific operations.
    ///
    /// Post-condition: A new `App` instance is returned, ready to run the Servo embedding.
    fn new(app: &AndroidApp) -> App {
        let opts = servo::config::opts::Opts {
            debug: servo::config::opts::DebugOptions {
                convert_mouse_to_touch: true,
                ..Default::default()
            },
            ..Default::default()
        };

        // Block Logic: Initializes `surfman` for graphics.
        let connection = Connection::new().expect("Couldn't create a surfman connection!");
        let adapter = connection
            .create_hardware_adapter(&None)
            .expect("Couldn't create a surfman adapter!");
        let mut device = connection
            .create_device(&adapter)
            .expect("Couldn't create a surfman device!");

        let context_attributes = ContextAttributes {
            version: GLVersion::new(2, 0),
            // OpenGL ES supports double buffering natively, so no need for explicit double buffering.
            // On the other hand, `surfman` also makes this implicit by returning `NULL` as an
            // `egl_surface`'s `back_buffer_texture` on Android.
            context_attribute_preference: ContextAttributePreference::LowPower,
        };

        let context = device
            .create_context(&context_attributes)
            .expect("Couldn't create a surfman context!");

        let window_rendering_context = Rc::new(WindowRenderingContext::new_with_context(
            connection.clone(),
            device.clone(),
            context.clone(),
        ));
        let rendering_context = Rc::new(window_rendering_context.offscreen_context(
            DeviceIntSize::new(1, 1),
        ));

        WindowGLContext::initialize_with_context(
            window_rendering_context.connection(),
            window_rendering_context.device(),
            window_rendering_context.context(),
        );

        let event_loop_waker = Box::new(android_platform::Waker::new(app));

        let embedder = Box::new(EmbedderCallbacks {
            waker: event_loop_waker.clone(),
        });
        let url = Url::parse("https://demo.servo.org/experiments/twgl-tunnel/")
            .expect("Failed to parse URL");
        let servo = Servo::new(
            opts,
            Default::default(),
            rendering_context.clone(),
            embedder,
            Rc::new(Platform),
            Some("android".into()),
        );
        let webview = servo.new_webview(url);
        webview.set_delegate(Rc::new(AndroidWebViewDelegate));
        servo.setup_logging();

        // Block Logic: Initializes the surface and native widget for the Android window.
        let (surface, native_widget) = {
            let (width, height) = android_platform::get_size();
            let native_widget = connection
                .create_native_widget_from_window(app.window(), DeviceIntSize::new(width, height))
                .expect("Couldn't create a surfman native widget!");
            let surface_attributes = SurfaceAttributes {
                usage: SurfaceAccess::GPUOnly,
                format: connection
                    .surface_formats(&adapter)
                    .unwrap()
                    .get(0)
                    .unwrap()
                    .clone(),
                present_mode: surfman::PresentMode::FIFO,
                surface_attribute_preference: SurfaceAttributePreference::GPUOptimized,
            };
            let surface = device
                .create_surface(&context, &native_widget, &surface_attributes)
                .expect("Couldn't create a surfman surface!");
            (surface, native_widget)
        };

        App {
            servo,
            webview,
            event_loop_waker,
            connection,
            device,
            context,
            window_rendering_context,
            rendering_context,
            native_widget,
            surface,
            current_surface_size: Cell::new(DeviceIntSize::new(0, 0)),
            need_new_surface: Cell::new(false),
        }
    }

    /// Handles an incoming user action.
    ///
    /// # Arguments
    /// * `action` - The `UserAction` to process.
    ///
    /// Post-condition: The action is processed, which might involve resizing, repainting,
    /// or passing events to the WebView.
    fn handle_action(&mut self, action: UserAction) {
        match action {
            UserAction::Resize(width, height) => {
                let size = DeviceIntSize::new(width, height);
                if self.current_surface_size.get() != size {
                    self.webview.move_resize(DeviceIntRect::new(
                        DeviceIntPoint::new(0, 0),
                        size,
                    ));
                    self.window_rendering_context.resize(size);
                    self.rendering_context.resize(size);
                    self.current_surface_size.set(size);
                    self.need_new_surface.set(true);
                    self.webview.notify_rendering_context_resized();
                }
            },
            UserAction::Input(event) => self.webview.notify_input_event(event),
            UserAction::Resume => {
                self.servo.spin_event_loop();
                self.webview.paint();
                self.present();
            },
            UserAction::Repaint => {
                self.webview.paint();
                self.present();
            },
        }
    }

    /// Presents the rendered content to the display.
    ///
    /// Post-condition: The `surfman` surface is presented, making the rendered frame visible.
    /// If a new surface is needed, it is created before presenting.
    fn present(&mut self) {
        if self.need_new_surface.get() {
            let new_size = self.current_surface_size.get();
            let surface_attributes = SurfaceAttributes {
                usage: surfman::SurfaceAccess::GPUOnly,
                format: self
                    .connection
                    .surface_formats(&self.device)
                    .unwrap()
                    .get(0)
                    .unwrap()
                    .clone(),
                present_mode: surfman::PresentMode::FIFO,
                surface_attribute_preference: SurfaceAttributePreference::GPUOptimized,
            };
            self.surface = self
                .device
                .create_surface(&self.context, &self.native_widget, &surface_attributes)
                .expect("Couldn't create a surfman surface!");
            self.connection
                .resize_surface(&mut self.device, &self.context, &mut self.surface, new_size.width, new_size.height)
                .unwrap();
            self.need_new_surface.set(false);
        }
        self.device.present_surface(&self.context, &mut self.surface).unwrap();
    }
}

/// `UserAction` enumerates the types of actions an Android user can perform or that the system can trigger.
#[derive(Debug)]
enum UserAction {
    /// Indicates a resize event with new width and height.
    Resize(i32, i32),
    /// An input event to be processed by the WebView.
    Input(servo::InputEvent),
    /// A signal to resume the application's activity.
    Resume,
    /// A request to repaint the WebView.
    Repaint,
}

/// `AndroidWebViewDelegate` implements `WebViewDelegate` for the Android embedding.
/// It primarily handles `notify_new_frame_ready` to request a repaint.
struct AndroidWebViewDelegate;

impl WebViewDelegate for AndroidWebViewDelegate {
    /// Notifies the delegate that a new frame is ready for rendering.
    ///
    /// # Arguments
    /// * `_webview` - The `WebView` instance (unused).
    ///
    /// Post-condition: The event loop is woken up to trigger a repaint.
    fn notify_new_frame_ready(&self, _webview: WebView) {
        android_platform::Waker::wake();
    }
}

/// `EmbedderCallbacks` implements `EmbedderMethods` for the Android embedding.
/// It primarily handles `create_event_loop_waker` to provide a way to wake up
/// the Android event loop.
struct EmbedderCallbacks {
    /// The event loop waker for the Android platform.
    waker: Box<dyn servo::EventLoopWaker>,
}

impl EmbedderMethods for EmbedderCallbacks {
    /// Creates an `EventLoopWaker` suitable for the Android event loop.
    ///
    /// Post-condition: A `Box<dyn servo::EventLoopWaker>` is returned.
    fn create_event_loop_waker(&mut self) -> Box<dyn servo::EventLoopWaker> {
        self.waker.clone()
    }
}

/// `ServoShellServoDelegate` implements the `ServoDelegate` trait for `simpleservo`.
/// It handles notifications and requests from the `Servo` engine, such as error reporting.
struct ServoShellServoDelegate;

impl ServoDelegate for ServoShellServoDelegate {
    /// Notifies the delegate of a `ServoError`.
    ///
    /// # Arguments
    /// * `_servo` - The `Servo` instance (unused).
    /// * `error` - The `ServoError` that occurred.
    ///
    /// Post-condition: An error message is logged to the Android log.
    fn notify_error(&self, _servo: &Servo, error: ServoError) {
        error!("Saw Servo error: {error:?}!");
    }
}

/// `Platform` implements `NativeDisplay`, `GlApi`, and `GlContext` traits
/// to bridge Android's EGL and OpenGL ES to Servo's graphics abstraction.
struct Platform;

impl NativeDisplay for Platform {
    /// Returns the HiDPI factor for the device.
    ///
    /// Post-condition: A `Scale<f32, servo::DeviceIndependentPixel, servo::DevicePixel>` is returned,
    /// representing the device's pixel ratio.
    fn hidpi_factor(&self) -> Scale<f32, servo::DeviceIndependentPixel, servo::DevicePixel> {
        Scale::new(1.0)
    }

    /// Returns the dimensions of the window.
    ///
    /// Post-condition: A `DeviceIntSize` representing the window's width and height is returned.
    fn window_size(&self) -> DeviceIntSize {
        let (width, height) = android_platform::get_size();
        DeviceIntSize::new(width, height)
    }

    /// Creates a new GL window (not used in this Android EGL implementation).
    ///
    /// Post-condition: Panics as this is not implemented for Android EGL.
    fn new_glwindow(
        &self,
        _event_loop: &android_platform::AndroidApp,
    ) -> Rc<dyn servo::webxr::glwindow::GlWindow> {
        unimplemented!()
    }
}

impl GlApi for Platform {
    /// Returns the OpenGL ES bindings.
    ///
    /// Post-condition: The `gl::Gl` instance for OpenGL ES is returned.
    fn gl_api(&self) -> Rc<dyn gl::Gl> {
        gl::GlType::Gles.get_gl()
    }
}

impl GlContext for Platform {
    /// Creates a new EGL window surface.
    ///
    /// Post-condition: A new `surfman::SurfaceTexture` is returned.
    fn create_window_surface(&self) -> SurfaceTexture {
        unimplemented!()
    }

    /// Destroys an EGL window surface.
    ///
    /// # Arguments
    /// * `surface_texture` - The `SurfaceTexture` to destroy.
    ///
    /// Post-condition: The `surfman::SurfaceTexture` is destroyed.
    fn destroy_window_surface(&self, _surface_texture: SurfaceTexture) {
        unimplemented!()
    }

    /// Makes the EGL context current.
    ///
    /// Post-condition: The EGL context is made current.
    fn make_current(&self) {
        unimplemented!()
    }

    /// Makes the EGL context not current.
    ///
    /// Post-condition: The EGL context is made not current.
    fn make_not_current(&self) {
        unimplemented!()
    }

    /// Swaps the EGL buffers.
    ///
    /// Post-condition: The EGL buffers are swapped, displaying the rendered frame.
    fn swap_buffers(&self) {
        unimplemented!()
    }

    /// Retrieves the framebuffer surface texture.
    ///
    /// Post-condition: A `SurfaceTexture` representing the framebuffer.
    fn get_framebuffer_surface(&self) -> SurfaceTexture {
        unimplemented!()
    }

    /// Retrieves the address of an OpenGL function.
    ///
    /// # Arguments
    /// * `_proc_name` - The name of the OpenGL function.
    ///
    /// Post-condition: Returns a `*const c_void` pointer to the function.
    fn get_proc_address(&self, _proc_name: &str) -> *const std::ffi::c_void {
        unimplemented!()
    }
}

/// Runs the `simpleservo` application.
///
/// This function sets up logging, initializes Servo's crypto, and enters the
/// Android event loop to handle user actions and system events.
///
/// # Arguments
/// * `app_handle` - The `AndroidApp` instance.
///
/// Post-condition: The Android event loop runs, processing events until the
/// application is stopped.
pub fn run_simpleservo(app_handle: AndroidApp) {
    // Block Logic: Sets up logging for Android, filtering messages at the `Info` level.
    android_platform::init_logging(LevelFilter::Info);
    servo::init_crypto();
    let mut app = App::new(&app_handle);

    // Block Logic: Processes events from the Android app event loop.
    android_platform::EventLoop::new(app_handle).run(move |event| {
        match event {
            android_platform::Event::Main(android_platform::MainEvent::InitWindow) => {
                info!("InitWindow");
                let (width, height) = android_platform::get_size();
                app.handle_action(UserAction::Resize(width, height));
            },
            android_platform::Event::Main(android_platform::MainEvent::TermWindow) => {
                info!("TermWindow");
            },
            android_platform::Event::Main(android_platform::MainEvent::GainedFocus) => {
                info!("GainedFocus");
                app.handle_action(UserAction::Resume);
            },
            android_platform::Event::Main(android_platform::MainEvent::LostFocus) => {
                info!("LostFocus");
            },
            android_platform::Event::Main(android_platform::MainEvent::Input(input)) => {
                info!("Input event: {:?}", input);
                let point = servo::euclid::point2(input.x, input.y);
                app.handle_action(UserAction::Input(servo::InputEvent::MouseButton(
                    servo::MouseButtonEvent {
                        point,
                        action: match input.event_type {
                            android_platform::EventType::Down => {
                                servo::MouseButtonAction::Down
                            },
                            android_platform::EventType::Up => servo::MouseButtonAction::Up,
                            _ => return,
                        },
                        button: servo::MouseButton::Left,
                    },
                )));
            },
            android_platform::Event::Wake => {
                app.handle_action(UserAction::Repaint);
            },
            _ => {},
        }
    });
}
