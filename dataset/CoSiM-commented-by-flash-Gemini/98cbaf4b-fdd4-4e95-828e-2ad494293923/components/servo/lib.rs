/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Servo, the mighty web browser engine from the future.
//!
//! This is a very simple library that wires all of Servo's components
//! together as type `Servo`, along with a generic client
//! implementing the `WindowMethods` trait, to create a working web
//! browser.
//!
//! The `Servo` type is responsible for configuring a
//! `Constellation`, which does the heavy lifting of coordinating all
//! of Servo's internal subsystems, including the `ScriptThread` and the
//! `LayoutThread`, as well maintains the navigation context.
//!
//! `Servo` is fed events from a generic type that implements the
//! `WindowMethods` trait.

mod clipboard_delegate;
mod proxies;
mod servo_delegate;
mod webview;
mod webview_delegate;

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::cmp::max;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::{Rc, Weak};
use std::sync::{Arc, Mutex};
use std::thread;

pub use base::id::TopLevelBrowsingContextId;
use base::id::{PipelineNamespace, PipelineNamespaceId, WebViewId};
#[cfg(feature = "bluetooth")]
use bluetooth::BluetoothThreadFactory;
#[cfg(feature = "bluetooth")]
use bluetooth_traits::BluetoothRequest;
use canvas::canvas_paint_thread::CanvasPaintThread;
use canvas::WebGLComm;
use canvas_traits::webgl::{GlType, WebGLThreads};
use clipboard_delegate::StringRequest;
use compositing::windowing::{EmbedderMethods, WindowMethods};
use compositing::{IOCompositor, InitialCompositorState};
use compositing_traits::{CompositorMsg, CompositorProxy, CompositorReceiver, ConstellationMsg};
#[cfg(all(
    not(target_os = "windows"),
    not(target_os = "ios"),
    not(target_os = "android"),
    not(target_arch = "arm"),
    not(target_arch = "aarch64"),
    not(target_env = "ohos"),
))]
use constellation::content_process_sandbox_profile;
use constellation::{
    Constellation, FromCompositorLogger, FromScriptLogger, InitialConstellationState,
    UnprivilegedContent,
};
use crossbeam_channel::{unbounded, Receiver, Sender};
pub use embedder_traits::*;
use env_logger::Builder as EnvLoggerBuilder;
use euclid::Scale;
use fonts::SystemFontService;
#[cfg(all(
    not(target_os = "windows"),
    not(target_os = "ios"),
    not(target_os = "android"),
    not(target_arch = "arm"),
    not(target_arch = "aarch64"),
    not(target_env = "ohos"),
))]
use gaol::sandbox::{ChildSandbox, ChildSandboxMethods};
pub use gleam::gl;
use gleam::gl::RENDERER;
use ipc_channel::ipc::{self, IpcSender};
use ipc_channel::router::ROUTER;
pub use keyboard_types::*;
#[cfg(feature = "layout_2013")]
pub use layout_thread_2013;
use log::{debug, warn, Log, Metadata, Record};
use media::{GlApi, NativeDisplay, WindowGLContext};
use net::protocols::ProtocolRegistry;
use net::resource_thread::new_resource_threads;
use profile::{mem as profile_mem, time as profile_time};
use profile_traits::{mem, time};
use script::{JSEngineSetup, ServiceWorkerManager};
use script_layout_interface::LayoutFactory;
use script_traits::{ScriptToConstellationChan, WindowSizeData};
use servo_config::opts::Opts;
use servo_config::prefs::Preferences;
use servo_config::{opts, pref, prefs};
use servo_delegate::DefaultServoDelegate;
use servo_media::player::context::GlContext;
use servo_media::ServoMedia;
use servo_url::ServoUrl;
#[cfg(feature = "webgpu")]
pub use webgpu;
#[cfg(feature = "webgpu")]
use webgpu::swapchain::WGPUImageMap;
use webrender::{RenderApiSender, ShaderPrecacheFlags, UploadMethod, ONE_TIME_USAGE_HINT};
use webrender_api::{ColorF, DocumentId, FramePublishId};
pub use webrender_traits::rendering_context::{
    OffscreenRenderingContext, RenderingContext, SoftwareRenderingContext, SurfmanRenderingContext,
    WindowRenderingContext,
};
use webrender_traits::{
    CrossProcessCompositorApi, WebrenderExternalImageHandlers, WebrenderExternalImageRegistry,
    WebrenderImageHandlerType,
};
use webview::WebViewInner;
#[cfg(feature = "webxr")]
pub use webxr;
pub use {
    background_hang_monitor, base, canvas, canvas_traits, compositing, devtools, devtools_traits,
    euclid, fonts, ipc_channel, layout_thread_2020, media, net, net_traits, profile,
    profile_traits, script, script_layout_interface, script_traits, servo_config as config,
    servo_config, servo_geometry, servo_url, style, style_traits, webrender_api,
};
#[cfg(feature = "bluetooth")]
pub use {bluetooth, bluetooth_traits};

use crate::proxies::ConstellationProxy;
pub use crate::servo_delegate::{ServoDelegate, ServoError};
pub use crate::webview::WebView;
pub use crate::webview_delegate::{
    AllowOrDenyRequest, AuthenticationRequest, NavigationRequest, PermissionRequest,
    WebViewDelegate,
};

/// Starts the WebDriver server if the `webdriver` feature is enabled.
///
/// # Arguments
/// * `port` - The port number on which the WebDriver server should listen.
/// * `constellation` - A `Sender` to communicate with the `Constellation`.
#[cfg(feature = "webdriver")]
fn webdriver(port: u16, constellation: Sender<ConstellationMsg>) {
    webdriver_server::start_server(port, constellation);
}

/// Placeholder function for `webdriver` when the feature is not enabled.
#[cfg(not(feature = "webdriver"))]
fn webdriver(_port: u16, _constellation: Sender<ConstellationMsg>) {}

/// Media platform specific initialization for GStreamer backend.
#[cfg(feature = "media-gstreamer")]
mod media_platform {
    #[cfg(any(windows, target_os = "macos"))]
    mod gstreamer_plugins {
        include!(concat!(env!("OUT_DIR"), "/gstreamer_plugins.rs"));
    }

    use servo_media_gstreamer::GStreamerBackend;

    use super::ServoMedia;

    /// Initializes GStreamer with platform-specific plugins.
    #[cfg(any(windows, target_os = "macos"))]
    pub fn init() {
        ServoMedia::init_with_backend(|| {
            let mut plugin_dir = std::env::current_exe().unwrap();
            plugin_dir.pop();

            if cfg!(target_os = "macos") {
                plugin_dir.push("lib");
            }

            match GStreamerBackend::init_with_plugins(
                plugin_dir,
                gstreamer_plugins::GSTREAMER_PLUGINS,
            ) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("Error initializing GStreamer: {:?}", e);
                    std::process::exit(1);
                },
            }
        });
    }

    /// Initializes GStreamer with default backend for other platforms.
    #[cfg(not(any(windows, target_os = "macos")))]
    pub fn init() {
        ServoMedia::init::<GStreamerBackend>();
    }
}

/// Placeholder for media platform initialization when `media-gstreamer` feature is not enabled.
#[cfg(not(feature = "media-gstreamer"))]
mod media_platform {
    use super::ServoMedia;
    pub fn init() {
        ServoMedia::init::<servo_media_dummy::DummyBackend>();
    }
}

/// The in-process interface to Servo.
///
/// It does everything necessary to render the web, primarily
/// orchestrating the interaction between JavaScript, CSS layout,
/// rendering, and the client window.
///
/// Clients create a `Servo` instance for a given reference-counted type
/// implementing `WindowMethods`, which is the bridge to whatever
/// application Servo is embedded in. Clients then create an event
/// loop to pump messages between the embedding application and
/// various browser components.
pub struct Servo {
    /// The current delegate responsible for handling Servo-specific events and requests.
    delegate: RefCell<Rc<dyn ServoDelegate>>,
    /// A reference-counted, mutable cell containing the `IOCompositor`, which manages rendering.
    compositor: Rc<RefCell<IOCompositor>>,
    /// A proxy for communicating with the `Constellation`, Servo's central coordination unit.
    constellation_proxy: ConstellationProxy,
    /// A receiver for messages originating from the embedder.
    embedder_receiver: Receiver<EmbedderMsg>,
    /// Tracks whether we are in the process of shutting down, or have shut down.
    /// This is shared with `WebView`s and the `ServoRenderer`.
    shutdown_state: Rc<Cell<ShutdownState>>,
    /// A map of `WebViewId` to weak references of `WebViewInner` instances managed by this `Servo` instance.
    /// Weak references allow the embedding application to control the lifetime of `WebView`s.
    webviews: RefCell<HashMap<WebViewId, Weak<RefCell<WebViewInner>>>>,
    /// For single-process Servo instances, this field controls the initialization
    /// and deinitialization of the JS Engine. Multiprocess Servo instances have their
    /// own instance that exists in the content process instead.
    _js_engine_setup: Option<JSEngineSetup>,
}

/// `RenderNotifier` implements `webrender_api::RenderNotifier` to bridge WebRender
/// frame-ready notifications to Servo's compositor.
#[derive(Clone)]
struct RenderNotifier {
    /// A proxy to the compositor, used to send messages about new frames.
    compositor_proxy: CompositorProxy,
}

impl RenderNotifier {
    /// Creates a new `RenderNotifier` instance.
    ///
    /// # Arguments
    /// * `compositor_proxy` - A `CompositorProxy` to communicate with the compositor.
    ///
    /// Post-condition: A new `RenderNotifier` is returned.
    pub fn new(compositor_proxy: CompositorProxy) -> RenderNotifier {
        RenderNotifier { compositor_proxy }
    }
}

impl webrender_api::RenderNotifier for RenderNotifier {
    /// Clones the `RenderNotifier` instance.
    ///
    /// Post-condition: A new boxed `RenderNotifier` instance is returned.
    fn clone(&self) -> Box<dyn webrender_api::RenderNotifier> {
        Box::new(RenderNotifier::new(self.compositor_proxy.clone()))
    }

    /// Wakes up the rendering system (no-op in this implementation).
    fn wake_up(&self, _composite_needed: bool) {}

    /// Notifies the compositor that a new WebRender frame is ready.
    ///
    /// # Arguments
    /// * `document_id` - The `DocumentId` for which the frame is ready.
    /// * `_scrolled` - A boolean indicating if the frame was scrolled (unused).
    /// * `composite_needed` - A boolean indicating if a new composite is needed.
    /// * `_frame_publish_id` - The `FramePublishId` of the new frame (unused).
    ///
    /// Post-condition: A `CompositorMsg::NewWebRenderFrameReady` message is sent
    /// to the compositor.
    fn new_frame_ready(
        &self,
        document_id: DocumentId,
        _scrolled: bool,
        composite_needed: bool,
        _frame_publish_id: FramePublishId,
    ) {
        self.compositor_proxy
            .send(CompositorMsg::NewWebRenderFrameReady(
                document_id,
                composite_needed,
            ));
    }
}

impl Servo {
    /// Creates a new `Servo` instance, initializing all its core components.
    ///
    /// # Arguments
    /// * `opts` - Global configuration options for Servo.
    /// * `preferences` - User preferences for various Servo behaviors.
    /// * `rendering_context` - The rendering context for graphical operations.
    /// * `embedder` - An instance of `EmbedderMethods` for interacting with the embedding application.
    /// * `window` - An instance of `WindowMethods` for managing window-related events.
    /// * `user_agent` - An optional user agent string to override the default.
    ///
    /// Pre-condition: All input parameters are valid and configured.
    /// Post-condition: A fully initialized `Servo` instance is returned, with its `Constellation`,
    /// `Compositor`, and other subsystems set up.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            skip(preferences, rendering_context, embedder, window),
            fields(servo_profiling = true),
            level = "trace",
        )
    )]
    pub fn new(
        opts: Opts,
        preferences: Preferences,
        rendering_context: Rc<dyn RenderingContext>,
        mut embedder: Box<dyn EmbedderMethods>,
        window: Rc<dyn WindowMethods>,
        user_agent: Option<String>,
    ) -> Self {
        // Global configuration options, parsed from the command line.
        opts::set_options(opts);
        let opts = opts::get();

        // Set the preferences globally.
        // TODO: It would be better to make these private to a particular Servo instance.
        servo_config::prefs::set(preferences);

        use std::sync::atomic::Ordering;

        style::context::DEFAULT_DISABLE_STYLE_SHARING_CACHE
            .store(opts.debug.disable_share_style_cache, Ordering::Relaxed);
        style::context::DEFAULT_DUMP_STYLE_STATISTICS
            .store(opts.debug.dump_style_statistics, Ordering::Relaxed);
        style::traversal::IS_SERVO_NONINCREMENTAL_LAYOUT
            .store(opts.nonincremental_layout, Ordering::Relaxed);

        if !opts.multiprocess {
            media_platform::init();
        }

        let user_agent = match user_agent {
            Some(ref ua) if ua == "ios" => default_user_agent_string_for(UserAgent::iOS).into(),
            Some(ref ua) if ua == "android" => {
                default_user_agent_string_for(UserAgent::Android).into()
            },
            Some(ref ua) if ua == "desktop" => {
                default_user_agent_string_for(UserAgent::Desktop).into()
            },
            Some(ref ua) if ua == "ohos" => {
                default_user_agent_string_for(UserAgent::OpenHarmony).into()
            },
            Some(ua) => ua.into(),
            None => embedder
                .get_user_agent_string()
                .map(Into::into)
                .unwrap_or(default_user_agent_string_for(DEFAULT_USER_AGENT).into()),
        };

        // Get GL bindings
        let webrender_gl = rendering_context.gl_api();

        // Make sure the gl context is made current.
        if let Err(err) = rendering_context.make_current() {
            warn!("Failed to make the rendering context current: {:?}", err);
        }
        debug_assert_eq!(webrender_gl.get_error(), gleam::gl::NO_ERROR,);

        // Reserving a namespace to create TopLevelBrowsingContextId.
        PipelineNamespace::install(PipelineNamespaceId(0));

        // Get both endpoints of a special channel for communication between
        // the client window and the compositor. This channel is unique because
        // messages to client may need to pump a platform-specific event loop
        // to deliver the message.
        let event_loop_waker = embedder.create_event_loop_waker();
        let (compositor_proxy, compositor_receiver) =
            create_compositor_channel(event_loop_waker.clone());
        let (embedder_proxy, embedder_receiver) = create_embedder_channel(event_loop_waker.clone());
        let time_profiler_chan = profile_time::Profiler::create(
            &opts.time_profiling,
            opts.time_profiler_trace_path.clone(),
        );
        let mem_profiler_chan = profile_mem::Profiler::create(opts.mem_profiler_period);

        let devtools_sender = if pref!(devtools_server_enabled) {
            Some(devtools::start_server(
                pref!(devtools_server_port) as u16,
                embedder_proxy.clone(),
            ))
        } else {
            None
        };

        let coordinates: compositing::windowing::EmbedderCoordinates = window.get_coordinates();
        let device_pixel_ratio = coordinates.hidpi_factor.get();
        let viewport_size = coordinates.viewport.size().to_f32() / device_pixel_ratio;

        let (mut webrender, webrender_api_sender) = {
            let mut debug_flags = webrender::DebugFlags::empty();
            debug_flags.set(
                webrender::DebugFlags::PROFILER_DBG,
                opts.debug.webrender_stats,
            );

            rendering_context.prepare_for_rendering();
            let render_notifier = Box::new(RenderNotifier::new(compositor_proxy.clone()));
            let clear_color = servo_config::pref!(shell_background_color_rgba);
            let clear_color = ColorF::new(
                clear_color[0] as f32,
                clear_color[1] as f32,
                clear_color[2] as f32,
                clear_color[3] as f32,
            );

            // Use same texture upload method as Gecko with ANGLE:
            // https://searchfox.org/mozilla-central/source/gfx/webrender_bindings/src/bindings.rs#1215-1219
            let upload_method = if webrender_gl.get_string(RENDERER).starts_with("ANGLE") {
                UploadMethod::Immediate
            } else {
                UploadMethod::PixelBuffer(ONE_TIME_USAGE_HINT)
            };
            let worker_threads = thread::available_parallelism()
                .map(|i| i.get())
                .unwrap_or(pref!(threadpools_fallback_worker_num) as usize)
                .min(pref!(threadpools_webrender_workers_max).max(1) as usize);
            let workers = Some(Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(worker_threads)
                    .thread_name(|idx| format!("WRWorker#{}", idx))
                    .build()
                    .unwrap(),
            ));
            webrender::create_webrender_instance(
                webrender_gl.clone(),
                render_notifier,
                webrender::WebRenderOptions {
                    // We force the use of optimized shaders here because rendering is broken
                    // on Android emulators with unoptimized shaders. This is due to a known
                    // issue in the emulator's OpenGL emulation layer.
                    // See: https://github.com/servo/servo/issues/31726
                    use_optimized_shaders: true,
                    resource_override_path: opts.shaders_dir.clone(),
                    debug_flags,
                    precache_flags: if pref!(gfx_precache_shaders) {
                        ShaderPrecacheFlags::FULL_COMPILE
                    } else {
                        ShaderPrecacheFlags::empty()
                    },
                    enable_aa: pref!(gfx_text_antialiasing_enabled),
                    enable_subpixel_aa: pref!(gfx_subpixel_text_antialiasing_enabled),
                    allow_texture_swizzling: pref!(gfx_texture_swizzling_enabled),
                    clear_color,
                    upload_method,
                    workers,
                    ..Default::default()
                },
                None,
            )
            .expect("Unable to initialize webrender!")
        };

        let webrender_api = webrender_api_sender.create_api();
        let webrender_document = webrender_api.add_document(coordinates.get_viewport().size());

        // Important that this call is done in a single-threaded fashion, we
        // can't defer it after `create_constellation` has started.
        let js_engine_setup = if !opts.multiprocess {
            Some(script::init())
        } else {
            None
        };

        // Create the webgl thread
        let gl_type = match webrender_gl.get_type() {
            gleam::gl::GlType::Gl => GlType::Gl,
            gleam::gl::GlType::Gles => GlType::Gles,
        };

        let (external_image_handlers, external_images) = WebrenderExternalImageHandlers::new();
        let mut external_image_handlers = Box::new(external_image_handlers);

        let WebGLComm {
            webgl_threads,
            #[cfg(feature = "webxr")]
            webxr_layer_grand_manager,
            image_handler,
        } = WebGLComm::new(
            rendering_context.clone(),
            webrender_api.create_sender(),
            webrender_document,
            external_images.clone(),
            gl_type,
        );

        // Set webrender external image handler for WebGL textures
        external_image_handlers.set_handler(image_handler, WebrenderImageHandlerType::WebGL);

        // Create the WebXR main thread
        #[cfg(feature = "webxr")]
        let mut webxr_main_thread =
            webxr::MainThreadRegistry::new(event_loop_waker, webxr_layer_grand_manager)
                .expect("Failed to create WebXR device registry");
        #[cfg(feature = "webxr")]
        if pref!(dom_webxr_enabled) {
            embedder.register_webxr(&mut webxr_main_thread, embedder_proxy.clone());
        }

        #[cfg(feature = "webgpu")]
        let wgpu_image_handler = webgpu::WGPUExternalImages::default();
        #[cfg(feature = "webgpu")]
        let wgpu_image_map = wgpu_image_handler.images.clone();
        #[cfg(feature = "webgpu")]
        external_image_handlers.set_handler(
            Box::new(wgpu_image_handler),
            WebrenderImageHandlerType::WebGPU,
        );

        WindowGLContext::initialize_image_handler(
            &mut external_image_handlers,
            external_images.clone(),
        );

        webrender.set_external_image_handler(external_image_handlers);

        // The division by 1 represents the page's default zoom of 100%,
        // and gives us the appropriate CSSPixel type for the viewport.
        let window_size = WindowSizeData {
            initial_viewport: viewport_size / Scale::new(1.0),
            device_pixel_ratio: Scale::new(device_pixel_ratio),
        };

        // Create the constellation, which maintains the engine pipelines, including script and
        // layout, as well as the navigation context.
        let mut protocols = ProtocolRegistry::with_internal_protocols();
        protocols.merge(embedder.get_protocol_handlers());

        let constellation_chan = create_constellation(
            user_agent,
            opts.config_dir.clone(),
            embedder_proxy,
            compositor_proxy.clone(),
            time_profiler_chan.clone(),
            mem_profiler_chan.clone(),
            devtools_sender,
            webrender_document,
            webrender_api_sender,
            #[cfg(feature = "webxr")]
            webxr_main_thread.registry(),
            Some(webgl_threads),
            window_size,
            external_images,
            #[cfg(feature = "webgpu")]
            wgpu_image_map,
            protocols,
        );

        if cfg!(feature = "webdriver") {
            if let Some(port) = opts.webdriver_port {
                webdriver(port, constellation_chan.clone());
            }
        }

        // The compositor coordinates with the client window to create the final
        // rendered page and display it somewhere.
        let shutdown_state = Rc::new(Cell::new(ShutdownState::NotShuttingDown));
        let compositor = IOCompositor::new(
            window,
            InitialCompositorState {
                sender: compositor_proxy,
                receiver: compositor_receiver,
                constellation_chan: constellation_chan.clone(),
                time_profiler_chan,
                mem_profiler_chan,
                webrender,
                webrender_document,
                webrender_api,
                rendering_context,
                webrender_gl,
                #[cfg(feature = "webxr")]
                webxr_main_thread,
                shutdown_state: shutdown_state.clone(),
            },
            opts.debug.convert_mouse_to_touch,
            embedder.get_version_string().unwrap_or_default(),
        );

        Self {
            delegate: RefCell::new(Rc::new(DefaultServoDelegate)),
            compositor: Rc::new(RefCell::new(compositor)),
            constellation_proxy: ConstellationProxy::new(constellation_chan),
            embedder_receiver,
            shutdown_state,
            webviews: Default::default(),
            _js_engine_setup: js_engine_setup,
        }
    }

    /// Returns a reference-counted instance of the current `ServoDelegate`.
    ///
    /// Post-condition: A `Rc<dyn ServoDelegate>` is returned, allowing interaction
    /// with the embedding application's specific implementations.
    pub fn delegate(&self) -> Rc<dyn ServoDelegate> {
        self.delegate.borrow().clone()
    }

    /// Sets a new delegate for the `Servo` instance.
    ///
    /// # Arguments
    /// * `delegate` - The new `Rc<dyn ServoDelegate>` to set.
    ///
    /// Post-condition: The internal delegate is updated to the provided one.
    pub fn set_delegate(&self, delegate: Rc<dyn ServoDelegate>) {
        *self.delegate.borrow_mut() = delegate;
    }

    /// **EXPERIMENTAL:** Intialize GL accelerated media playback. This currently only works on a limited number
    /// of platforms. This should be run *before* calling [`Servo::new`] and creating the first [`WebView`].
    ///
    /// # Arguments
    /// * `display` - The native display context.
    /// * `api` - The GL API to use (e.g., OpenGL, OpenGL ES).
    /// * `context` - The GL context for rendering.
    ///
    /// Pre-condition: Called before `Servo::new` and before creating any `WebView`s.
    /// Post-condition: GL accelerated media playback is initialized.
    pub fn initialize_gl_accelerated_media(display: NativeDisplay, api: GlApi, context: GlContext) {
        WindowGLContext::initialize(display, api, context)
    }

    /// Spins the Servo event loop.
    ///
    /// This method performs several crucial tasks:
    /// - Processes incoming messages for the compositor, such as queued pinch zoom events.
    /// - Executes delegate methods on all `WebView` instances and the `Servo` instance itself.
    /// - Optionally updates the rendered compositor output, without performing buffer swaps.
    ///
    /// Post-condition: Returns `true` if Servo is still running and the event loop should continue,
    /// `false` if Servo has finished shutting down.
    pub fn spin_event_loop(&self) -> bool {
        if self.shutdown_state.get() == ShutdownState::FinishedShuttingDown {
            return false;
        }

        self.compositor.borrow_mut().receive_messages();

        // Only handle incoming embedder messages if the compositor hasn't already started shutting down.
        // Block Logic: Processes incoming messages from the embedder.
        // Invariant: Iterates as long as there are messages and Servo is not fully shut down.
        while let Ok(message) = self.embedder_receiver.try_recv() {
            self.handle_embedder_message(message);

            if self.shutdown_state.get() == ShutdownState::FinishedShuttingDown {
                break;
            }
        }

        if self.constellation_proxy.disconnected() {
            self.delegate()
                .notify_error(self, ServoError::LostConnectionWithBackend);
        }

        self.compositor.borrow_mut().perform_updates();
        self.send_new_frame_ready_messages();
        self.clean_up_destroyed_webview_handles();

        if self.shutdown_state.get() == ShutdownState::FinishedShuttingDown {
            return false;
        }

        true
    }

    /// Sends `notify_new_frame_ready` messages to all `WebView` delegates if a repaint is needed.
    ///
    /// Pre-condition: The compositor has indicated that a repaint is necessary.
    /// Post-condition: `WebViewDelegate::notify_new_frame_ready` is called for each active webview.
    fn send_new_frame_ready_messages(&self) {
        if !self.compositor.borrow().needs_repaint() {
            return;
        }

        // Block Logic: Iterates through all active WebView handles.
        // Invariant: Only valid, non-destroyed WebView handles are processed.
        for webview in self
            .webviews
            .borrow()
            .values()
            .filter_map(WebView::from_weak_handle)
        {
            webview.delegate().notify_new_frame_ready(webview);
        }
    }

    /// Cleans up destroyed `WebView` handles from the internal map.
    ///
    /// Post-condition: Any `Weak` references to `WebViewInner` that no longer have a strong count
    /// (i.e., the `WebView` has been destroyed) are removed from the `webviews` map.
    fn clean_up_destroyed_webview_handles(&self) {
        // Remove any webview handles that have been destroyed and would not be upgradable.
        // Note that `retain` is O(capacity) because it visits empty buckets, so it may be worth
        // calling `shrink_to_fit` at some point to deal with cases where a long-running Servo
        // instance goes from many open webviews to only a few.
        self.webviews
            .borrow_mut()
            .retain(|_webview_id, webview| webview.strong_count() > 0);
    }

    /// Returns the current pinch zoom level of the compositor.
    ///
    /// Post-condition: The `f32` value representing the pinch zoom level is returned.
    pub fn pinch_zoom_level(&self) -> f32 {
        self.compositor.borrow_mut().pinch_zoom_level().get()
    }

    /// Sets up the global logging infrastructure for Servo.
    ///
    /// This involves configuring `env_logger` and setting up a composite logger
    /// that sends messages to both `env_logger` and the `Constellation`.
    ///
    /// Pre-condition: The `ConstellationProxy` is initialized.
    /// Post-condition: The global logger is configured, and logging messages are routed
    /// to both the environment logger and the Constellation.
    pub fn setup_logging(&self) {
        let constellation_chan = self.constellation_proxy.sender();
        let env = env_logger::Env::default();
        let env_logger = EnvLoggerBuilder::from_env(env).build();
        let con_logger = FromCompositorLogger::new(constellation_chan);

        let filter = max(env_logger.filter(), con_logger.filter());
        let logger = BothLogger(env_logger, con_logger);

        log::set_boxed_logger(Box::new(logger)).expect("Failed to set logger.");
        log::set_max_level(filter);
    }

    /// Initiates the shutdown process for Servo.
    ///
    /// This sends an `Exit` message to the `Constellation` and sets the internal
    /// `shutdown_state` to `ShuttingDown`.
    ///
    /// Pre-condition: Servo is not already in the process of shutting down.
    /// Post-condition: An `Exit` message is sent to the `Constellation`, and `shutdown_state`
    /// is set to `ShuttingDown`.
    pub fn start_shutting_down(&self) {
        if self.shutdown_state.get() != ShutdownState::NotShuttingDown {
            warn!("Requested shutdown while already shutting down");
            return;
        }

        debug!("Sending Exit message to Constellation");
        self.constellation_proxy.send(ConstellationMsg::Exit);
        self.shutdown_state.set(ShutdownState::ShuttingDown);
    }

    /// Finalizes the shutdown process of Servo.
    ///
    /// This method is called when the `Constellation` has completed its shutdown.
    /// It updates the `shutdown_state` to `FinishedShuttingDown` and instructs
    /// the compositor to finish its shutdown sequence.
    ///
    /// Pre-condition: The `Constellation` has signaled that its shutdown is complete.
    /// Post-condition: `shutdown_state` is set to `FinishedShuttingDown`, and the
    /// compositor's shutdown is finalized.
    fn finish_shutting_down(&self) {
        debug!("Servo received message that Constellation shutdown is complete");
        self.shutdown_state.set(ShutdownState::FinishedShuttingDown);
        self.compositor.borrow_mut().finish_shutting_down();
    }

    /// Deinitializes the compositor, releasing its resources.
    ///
    /// Post-condition: The `IOCompositor`'s `deinit` method is called.
    pub fn deinit(&self) {
        self.compositor.borrow_mut().deinit();
    }

    /// Creates and returns a new `WebView` instance.
    ///
    /// # Arguments
    /// * `url` - The initial `url::Url` to load in the new `WebView`.
    ///
    /// Pre-condition: Servo is initialized and not shutting down.
    /// Post-condition: A new `WebView` is created, registered internally, and
    /// a `ConstellationMsg::NewWebView` message is sent to the `Constellation`.
    /// The newly created `WebView` is returned.
    pub fn new_webview(&self, url: url::Url) -> WebView {
        let webview = WebView::new(&self.constellation_proxy, self.compositor.clone());
        self.webviews
            .borrow_mut()
            .insert(webview.id(), webview.weak_handle());
        self.constellation_proxy
            .send(ConstellationMsg::NewWebView(url.into(), webview.id()));
        webview
    }

    /// Creates and returns a new auxiliary `WebView` instance.
    ///
    /// This is typically used for pop-up windows or other auxiliary browsing contexts.
    ///
    /// Pre-condition: Servo is initialized and not shutting down.
    /// Post-condition: A new auxiliary `WebView` is created and registered internally.
    /// The newly created `WebView` is returned.
    pub fn new_auxiliary_webview(&self) -> WebView {
        let webview = WebView::new(&self.constellation_proxy, self.compositor.clone());
        self.webviews
            .borrow_mut()
            .insert(webview.id(), webview.weak_handle());
        webview
    }

    /// Retrieves a `WebView` handle by its `WebViewId`.
    ///
    /// # Arguments
    /// * `id` - The `WebViewId` of the `WebView` to retrieve.
    ///
    /// Post-condition: An `Option<WebView>` is returned, which is `Some` if a matching
    /// `WebView` is found and is still active (not destroyed), and `None` otherwise.
    fn get_webview_handle(&self, id: WebViewId) -> Option<WebView> {
        self.webviews
            .borrow()
            .get(&id)
            .and_then(WebView::from_weak_handle)
    }

    /// Handles incoming messages from the embedder.
    ///
    /// # Arguments
    /// * `message` - The `EmbedderMsg` received from the embedding application.
    ///
    /// Pre-condition: `message` is a valid `EmbedderMsg`.
    /// Post-condition: The message is processed by dispatching it to the appropriate
    /// internal handler or `WebView` delegate.
    fn handle_embedder_message(&self, message: EmbedderMsg) {
        match message {
            EmbedderMsg::ShutdownComplete => self.finish_shutting_down(),
            EmbedderMsg::Status(webview_id, status_text) => {
                // Block Logic: Updates the status text of the specified WebView.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.set_status_text(status_text);
                }
            },
            EmbedderMsg::ChangePageTitle(webview_id, title) => {
                // Block Logic: Updates the page title of the specified WebView.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.set_page_title(title);
                }
            },
            EmbedderMsg::MoveTo(webview_id, position) => {
                // Block Logic: Requests the specified WebView delegate to move.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.delegate().request_move_to(webview, position);
                }
            },
            EmbedderMsg::ResizeTo(webview_id, size) => {
                // Block Logic: Requests the specified WebView delegate to resize.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.delegate().request_resize_to(webview, size);
                }
            },
            EmbedderMsg::Prompt(webview_id, prompt_definition, prompt_origin) => {
                // Block Logic: Displays a prompt via the specified WebView delegate.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview
                        .delegate()
                        .show_prompt(webview, prompt_definition, prompt_origin);
                }
            },
            EmbedderMsg::ShowContextMenu(webview_id, ipc_sender, title, items) => {
                // Block Logic: Displays a context menu via the specified WebView delegate.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview
                        .delegate()
                        .show_context_menu(webview, ipc_sender, title, items);
                }
            },
            EmbedderMsg::AllowNavigationRequest(webview_id, pipeline_id, servo_url) => {
                // Block Logic: Requests permission for navigation via the specified WebView delegate.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    let request = NavigationRequest {
                        url: servo_url.into_url(),
                        pipeline_id,
                        constellation_proxy: self.constellation_proxy.clone(),
                        response_sent: false,
                    };
                    webview.delegate().request_navigation(webview, request);
                }
            },
            EmbedderMsg::AllowOpeningWebView(webview_id, response_sender) => {
                // Block Logic: Requests permission to open a new auxiliary WebView.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    let new_webview = webview.delegate().request_open_auxiliary_webview(webview);
                    let _ = response_sender.send(new_webview.map(|webview| webview.id()));
                }
            },
            EmbedderMsg::WebViewOpened(webview_id) => {
                // Block Logic: Notifies the specified WebView delegate that it has been opened.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.delegate().notify_ready_to_show(webview);
                }
            },
            EmbedderMsg::WebViewClosed(webview_id) => {
                // Block Logic: Notifies the specified WebView delegate that it has been closed.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.delegate().notify_closed(webview);
                }
            },
            EmbedderMsg::WebViewFocused(webview_id) => {
                // Block Logic: Iterates through all WebViews to set focus state.
                // Invariant: Each WebView is processed to update its focused status.
                for id in self.webviews.borrow().keys() {
                    if let Some(webview) = self.get_webview_handle(*id) {
                        let focused = webview.id() == webview_id;
                        webview.set_focused(focused);
                    }
                }
            },
            EmbedderMsg::WebViewBlurred => {
                // Block Logic: Iterates through all WebViews to clear focus state.
                // Invariant: Each WebView is processed to clear its focused status.
                for id in self.webviews.borrow().keys() {
                    if let Some(webview) = self.get_webview_handle(*id) {
                        webview.set_focused(false);
                    }
                }
            },
            EmbedderMsg::AllowUnload(webview_id, response_sender) => {
                // Block Logic: Requests permission to unload a WebView.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    let request = AllowOrDenyRequest {
                        response_sender,
                        response_sent: false,
                        default_response: AllowOrDeny::Allow,
                    };
                    webview.delegate().request_unload(webview, request);
                }
            },
            EmbedderMsg::Keyboard(webview_id, keyboard_event) => {
                // Block Logic: Notifies the specified WebView delegate of a keyboard event.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview
                        .delegate()
                        .notify_keyboard_event(webview, keyboard_event);
                }
            },
            EmbedderMsg::ClearClipboard(webview_id) => {
                // Block Logic: Clears the clipboard via the specified WebView's clipboard delegate.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.clipboard_delegate().clear(webview);
                }
            },
            EmbedderMsg::GetClipboardText(webview_id, result_sender) => {
                // Block Logic: Requests clipboard text via the specified WebView's clipboard delegate.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview
                        .clipboard_delegate()
                        .get_text(webview, StringRequest::from(result_sender));
                }
            },
            EmbedderMsg::SetClipboardText(webview_id, string) => {
                // Block Logic: Sets clipboard text via the specified WebView's clipboard delegate.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.clipboard_delegate().set_text(webview, string);
                }
            },
            EmbedderMsg::SetCursor(webview_id, cursor) => {
                // Block Logic: Sets the cursor for the specified WebView.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.set_cursor(cursor);
                }
            },
            EmbedderMsg::NewFavicon(webview_id, url) => {
                // Block Logic: Sets the favicon URL for the specified WebView.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.set_favicon_url(url.into_url());
                }
            },
            EmbedderMsg::NotifyLoadStatusChanged(webview_id, load_status) => {
                // Block Logic: Notifies the specified WebView of a load status change.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.set_load_status(load_status);
                }
            },
            EmbedderMsg::HistoryChanged(webview_id, urls, current_index) => {
                // Block Logic: Notifies the specified WebView delegate of a history change.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    let urls: Vec<_> = urls.into_iter().map(ServoUrl::into_url).collect();
                    let current_url = urls[current_index].clone();

                    webview
                        .delegate()
                        .notify_history_changed(webview.clone(), urls, current_index);
                    webview.set_url(current_url);
                }
            },
            EmbedderMsg::NotifyFullscreenStateChanged(webview_id, fullscreen) => {
                // Block Logic: Notifies the specified WebView delegate of a fullscreen state change.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview
                        .delegate()
                        .notify_fullscreen_state_changed(webview, fullscreen);
                }
            },
            EmbedderMsg::WebResourceRequested(
                webview_id,
                web_resource_request,
                response_sender,
            ) => {
                // Block Logic: Intercepts web resource requests via the WebView delegate or Servo delegate.
                // Invariant: The WebView (if specified) and Servo must be active.
                let webview = webview_id.and_then(|webview_id| self.get_webview_handle(webview_id));
                if let Some(webview) = webview.clone() {
                    webview.delegate().intercept_web_resource_load(
                        webview,
                        &web_resource_request,
                        response_sender.clone(),
                    );
                }

                self.delegate().intercept_web_resource_load(
                    webview,
                    &web_resource_request,
                    response_sender,
                );
            },
            EmbedderMsg::Panic(webview_id, reason, backtrace) => {
                // Block Logic: Notifies the specified WebView delegate of a crash.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview
                        .delegate()
                        .notify_crashed(webview, reason, backtrace);
                }
            },
            EmbedderMsg::GetSelectedBluetoothDevice(webview_id, items, response_sender) => {
                // Block Logic: Displays a Bluetooth device selection dialog via the WebView delegate.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.delegate().show_bluetooth_device_dialog(
                        webview,
                        items,
                        response_sender,
                    );
                }
            },
            EmbedderMsg::SelectFiles(
                webview_id,
                filter_patterns,
                allow_select_multiple,
                response_sender,
            ) => {
                // Block Logic: Displays a file selection dialog via the WebView delegate.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.delegate().show_file_selection_dialog(
                        webview,
                        filter_patterns,
                        allow_select_multiple,
                        response_sender,
                    );
                }
            },
            EmbedderMsg::RequestAuthentication(webview_id, url, for_proxy, response_sender) => {
                // Block Logic: Requests authentication via the specified WebView delegate.
                // Invariant: The WebView with `webview_id` must be active.
                let authentication_request = AuthenticationRequest {
                    url: url.into_url(),
                    for_proxy,
                    response_sender,
                    response_sent: false,
                };
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview
                        .delegate()
                        .request_authentication(webview, authentication_request);
                }
            },
            EmbedderMsg::PromptPermission(webview_id, requested_feature, response_sender) => {
                // Block Logic: Prompts for permission via the specified WebView delegate.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    let permission_request = PermissionRequest {
                        requested_feature,
                        allow_deny_request: AllowOrDenyRequest {
                            response_sender,
                            response_sent: false,
                            default_response: AllowOrDeny::Deny,
                        },
                    };
                    webview
                        .delegate()
                        .request_permission(webview, permission_request);
                }
            },
            EmbedderMsg::ShowIME(webview_id, input_method_type, text, multiline, position) => {
                // Block Logic: Shows IME via the specified WebView delegate.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.delegate().show_ime(
                        webview,
                        input_method_type,
                        text,
                        multiline,
                        position,
                    );
                }
            },
            EmbedderMsg::HideIME(webview_id) => {
                // Block Logic: Hides IME via the specified WebView delegate.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.delegate().hide_ime(webview);
                }
            },
            EmbedderMsg::ReportProfile(_items) => {},
            EmbedderMsg::MediaSessionEvent(webview_id, media_session_event) => {
                // Block Logic: Notifies the specified WebView delegate of a media session event.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview
                        .delegate()
                        .notify_media_session_event(webview, media_session_event);
                }
            },
            EmbedderMsg::OnDevtoolsStarted(port, token) => match port {
                Ok(port) => self
                    .delegate()
                    .notify_devtools_server_started(self, port, token),
                Err(()) => self
                    .delegate()
                    .notify_error(self, ServoError::DevtoolsFailedToStart),
            },
            EmbedderMsg::RequestDevtoolsConnection(response_sender) => {
                // Block Logic: Requests a DevTools connection via the Servo delegate.
                // Invariant: Servo must be active.
                self.delegate().request_devtools_connection(
                    self,
                    AllowOrDenyRequest {
                        response_sender,
                        response_sent: false,
                        default_response: AllowOrDeny::Deny,
                    },
                );
            },
            EmbedderMsg::PlayGamepadHapticEffect(
                webview_id,
                gamepad_index,
                gamepad_haptic_effect_type,
                ipc_sender,
            ) => {
                // Block Logic: Plays a gamepad haptic effect via the specified WebView delegate.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.delegate().play_gamepad_haptic_effect(
                        webview,
                        gamepad_index,
                        gamepad_haptic_effect_type,
                        ipc_sender,
                    );
                }
            },
            EmbedderMsg::StopGamepadHapticEffect(webview_id, gamepad_index, ipc_sender) => {
                // Block Logic: Stops a gamepad haptic effect via the specified WebView delegate.
                // Invariant: The WebView with `webview_id` must be active.
                if let Some(webview) = self.get_webview_handle(webview_id) {
                    webview.delegate().stop_gamepad_haptic_effect(
                        webview,
                        gamepad_index,
                        ipc_sender,
                    );
                }
            },
        }
    }
}

/// Creates a pair of channels for communication with the embedder.
///
/// # Arguments
/// * `event_loop_waker` - A `Box<dyn EventLoopWaker>` to wake up the event loop.
///
/// Post-condition: A `(EmbedderProxy, Receiver<EmbedderMsg>)` tuple is returned,
/// providing the sender and receiver for embedder messages.
fn create_embedder_channel(
    event_loop_waker: Box<dyn EventLoopWaker>,
) -> (EmbedderProxy, Receiver<EmbedderMsg>) {
    let (sender, receiver) = unbounded();
    (
        EmbedderProxy {
            sender,
            event_loop_waker,
        },
        receiver,
    )
}

/// Creates a pair of channels for communication with the compositor.
///
/// This sets up a cross-process communication channel for compositor messages,
/// utilizing `ipc_channel` and routing messages through `ROUTER`.
///
/// # Arguments
/// * `event_loop_waker` - A `Box<dyn EventLoopWaker>` to wake up the event loop.
///
/// Post-condition: A `(CompositorProxy, CompositorReceiver)` tuple is returned,
/// providing the sender and receiver for compositor messages, including cross-process APIs.
fn create_compositor_channel(
    event_loop_waker: Box<dyn EventLoopWaker>,
) -> (CompositorProxy, CompositorReceiver) {
    let (sender, receiver) = unbounded();

    let (compositor_ipc_sender, compositor_ipc_receiver) =
        ipc::channel().expect("ipc channel failure");

    let cross_process_compositor_api = CrossProcessCompositorApi(compositor_ipc_sender);
    let compositor_proxy = CompositorProxy {
        sender,
        cross_process_compositor_api,
        event_loop_waker,
    };

    let compositor_proxy_clone = compositor_proxy.clone();
    ROUTER.add_typed_route(
        compositor_ipc_receiver,
        Box::new(move |message| {
            compositor_proxy_clone.send(CompositorMsg::CrossProcess(
                message.expect("Could not convert Compositor message"),
            ));
        }),
    );

    (compositor_proxy, CompositorReceiver { receiver })
}

/// Determines and returns the appropriate layout factory based on the `legacy_layout` flag.
///
/// # Arguments
/// * `legacy_layout` - A boolean indicating whether to use the legacy layout engine.
///
/// Pre-condition: If `legacy_layout` is true, the `layout_2013` feature must be enabled
/// at compile time.
/// Post-condition: An `Arc<dyn LayoutFactory>` is returned, providing an instance of
/// either `layout_thread_2013::LayoutFactoryImpl` or `layout_thread_2020::LayoutFactoryImpl`.
fn get_layout_factory(legacy_layout: bool) -> Arc<dyn LayoutFactory> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "layout_2013")] {
            if legacy_layout {
                return Arc::new(layout_thread_2013::LayoutFactoryImpl());
            }
        } else {
            if legacy_layout {
                panic!("Runtime option `legacy_layout` was enabled, but the `layout_2013` 
                feature was not enabled at compile time! ");
           }
        }
    }
    Arc::new(layout_thread_2020::LayoutFactoryImpl())
}

/// Creates and initializes the `Constellation`, Servo's central coordination unit.
///
/// This function sets up all the necessary channels and initial state for the
/// `Constellation` to manage various Servo components, including communication
/// with the embedder, compositor, profilers, DevTools, and resource threads.
///
/// # Arguments
/// * `user_agent` - The user agent string for Servo.
/// * `config_dir` - Optional path to the configuration directory.
/// * `embedder_proxy` - A proxy for communicating with the embedder.
/// * `compositor_proxy` - A proxy for communicating with the compositor.
/// * `time_profiler_chan` - Channel for time profiling data.
/// * `mem_profiler_chan` - Channel for memory profiling data.
/// * `devtools_sender` - Optional sender for DevTools control messages.
/// * `webrender_document` - The `DocumentId` for WebRender.
/// * `webrender_api_sender` - Sender for WebRender API commands.
/// * `webxr_registry` - WebXR registry instance (if feature enabled).
/// * `webgl_threads` - Optional `WebGLThreads` instance.
/// * `initial_window_size` - The initial size of the top-level window.
/// * `external_images` - External image registry for WebRender.
/// * `wgpu_image_map` - WGPU image map (if feature enabled).
/// * `protocols` - Custom protocol handlers.
///
/// Pre-condition: All input parameters are valid and configured.
/// Post-condition: A `Sender<ConstellationMsg>` is returned, providing a channel
/// to communicate with the newly started `Constellation` thread.
#[allow(clippy::too_many_arguments)]
fn create_constellation(
    user_agent: Cow<'static, str>,
    config_dir: Option<PathBuf>,
    embedder_proxy: EmbedderProxy,
    compositor_proxy: CompositorProxy,
    time_profiler_chan: time::ProfilerChan,
    mem_profiler_chan: mem::ProfilerChan,
    devtools_sender: Option<Sender<devtools_traits::DevtoolsControlMsg>>,
    webrender_document: DocumentId,
    webrender_api_sender: RenderApiSender,
    #[cfg(feature = "webxr")] webxr_registry: webxr_api::Registry,
    webgl_threads: Option<WebGLThreads>,
    initial_window_size: WindowSizeData,
    external_images: Arc<Mutex<WebrenderExternalImageRegistry>>,
    #[cfg(feature = "webgpu")] wgpu_image_map: WGPUImageMap,
    protocols: ProtocolRegistry,
) -> Sender<ConstellationMsg> {
    // Global configuration options, parsed from the command line.
    let opts = opts::get();

    #[cfg(feature = "bluetooth")]
    let bluetooth_thread: IpcSender<BluetoothRequest> =
        BluetoothThreadFactory::new(embedder_proxy.clone());

    let (public_resource_threads, private_resource_threads) = new_resource_threads(
        user_agent.clone(),
        devtools_sender.clone(),
        time_profiler_chan.clone(),
        mem_profiler_chan.clone(),
        embedder_proxy.clone(),
        config_dir,
        opts.certificate_path.clone(),
        opts.ignore_certificate_errors,
        Arc::new(protocols),
    );

    let system_font_service = Arc::new(
        SystemFontService::spawn(compositor_proxy.cross_process_compositor_api.clone()).to_proxy(),
    );

    let (canvas_create_sender, canvas_ipc_sender) = CanvasPaintThread::start(
        compositor_proxy.cross_process_compositor_api.clone(),
        system_font_service.clone(),
        public_resource_threads.clone(),
    );

    let initial_state = InitialConstellationState {
        compositor_proxy,
        embedder_proxy,
        devtools_sender,
        #[cfg(feature = "bluetooth")]
        bluetooth_thread,
        system_font_service,
        public_resource_threads,
        private_resource_threads,
        time_profiler_chan,
        mem_profiler_chan,
        webrender_document,
        webrender_api_sender,
        #[cfg(feature = "webxr")]
        webxr_registry: Some(webxr_registry),
        #[cfg(not(feature = "webxr"))]
        webxr_registry: None,
        webgl_threads,
        user_agent,
        webrender_external_images: external_images,
        #[cfg(feature = "webgpu")]
        wgpu_image_map,
    };

    let layout_factory: Arc<dyn LayoutFactory> = get_layout_factory(opts::get().legacy_layout);

    Constellation::<script::ScriptThread, script::ServiceWorkerManager>::start(
        initial_state,
        layout_factory,
        initial_window_size,
        opts.random_pipeline_closure_probability,
        opts.random_pipeline_closure_seed,
        opts.hard_fail,
        canvas_create_sender,
        canvas_ipc_sender,
    )
}

/// A custom logger that multiplexes log messages to two underlying loggers.
/// This is useful for sending logs to both a standard output (e.g., console)
/// and an internal component (e.g., Constellation).
///
/// # Type Parameters
/// * `Log1` - The first logger type.
/// * `Log2` - The second logger type.
struct BothLogger<Log1, Log2>(Log1, Log2);

impl<Log1, Log2> Log for BothLogger<Log1, Log2>
where
    Log1: Log,
    Log2: Log,
{
    /// Checks if logging is enabled for the given metadata in either logger.
    fn enabled(&self, metadata: &Metadata) -> bool {
        self.0.enabled(metadata) || self.1.enabled(metadata)
    }

    /// Logs a record to both underlying loggers.
    fn log(&self, record: &Record) {
        self.0.log(record);
        self.1.log(record);
    }

    /// Flushes both underlying loggers.
    fn flush(&self) {
        self.0.flush();
        self.1.flush();
    }
}

/// Sets up the global logger for content processes.
///
/// This function configures `env_logger` and a `FromScriptLogger` to route
/// log messages from script threads to the `Constellation`.
///
/// # Arguments
/// * `script_to_constellation_chan` - A `ScriptToConstellationChan` for sending
/// log messages to the `Constellation`.
///
/// Post-condition: The global logger is configured to forward script-generated
/// log messages to the `Constellation` and to the environment logger.
pub fn set_logger(script_to_constellation_chan: ScriptToConstellationChan) {
    let con_logger = FromScriptLogger::new(script_to_constellation_chan);
    let env = env_logger::Env::default();
    let env_logger = EnvLoggerBuilder::from_env(env).build();

    let filter = max(env_logger.filter(), con_logger.filter());
    let logger = BothLogger(env_logger, con_logger);

    log::set_boxed_logger(Box::new(logger)).expect("Failed to set logger.");
    log::set_max_level(filter);
}

/// The entry point for content processes in a multi-process Servo setup.
///
/// This function is responsible for:
/// - Establishing IPC communication with the main Servo process.
/// - Setting up configuration options and preferences.
/// - Initializing sandboxing if enabled.
/// - Initializing the JavaScript engine.
/// - Starting either a pipeline or a service worker.
///
/// # Arguments
/// * `token` - A string token used to establish IPC connection with the main process.
///
/// Pre-condition: `token` is a valid IPC connection token.
/// Post-condition: The content process is initialized and starts its designated role (pipeline or service worker).
pub fn run_content_process(token: String) {
    let (unprivileged_content_sender, unprivileged_content_receiver) =
        ipc::channel::<UnprivilegedContent>().unwrap();
    let connection_bootstrap: IpcSender<IpcSender<UnprivilegedContent>> =
        IpcSender::connect(token).unwrap();
    connection_bootstrap
        .send(unprivileged_content_sender)
        .unwrap();

    let unprivileged_content = unprivileged_content_receiver.recv().unwrap();
    opts::set_options(unprivileged_content.opts());
    prefs::set(unprivileged_content.prefs().clone());

    // Enter the sandbox if necessary.
    if opts::get().sandbox {
        create_sandbox();
    }

    let _js_engine_setup = script::init();

    match unprivileged_content {
        UnprivilegedContent::Pipeline(mut content) => {
            media_platform::init();

            set_logger(content.script_to_constellation_chan().clone());

            let background_hang_monitor_register = content.register_with_background_hang_monitor();
            let layout_factory: Arc<dyn LayoutFactory> =
                get_layout_factory(opts::get().legacy_layout);

            content.start_all::<script::ScriptThread>(
                true,
                layout_factory,
                background_hang_monitor_register,
            );
        },
        UnprivilegedContent::ServiceWorker(content) => {
            content.start::<ServiceWorkerManager>();
        },
    }
}

/// Creates and activates a sandboxed environment for content processes.
///
/// This function is enabled for Linux (x86_64) and other non-Windows/iOS/Android ARM targets
/// and utilizes `gaol` for sandboxing.
///
/// Pre-condition: The necessary sandboxing libraries are available.
/// Post-condition: A child sandbox is created and activated, confining the content process.
#[cfg(all(
    not(target_os = "windows"),
    not(target_os = "ios"),
    not(target_os = "android"),
    not(target_arch = "arm"),
    not(target_arch = "aarch64"),
    not(target_env = "ohos"),
))]
fn create_sandbox() {
    ChildSandbox::new(content_process_sandbox_profile())
        .activate()
        .expect("Failed to activate sandbox!");
}

/// Placeholder for `create_sandbox` on unsupported platforms.
///
/// This function panics if called on platforms where sandboxing is not supported
/// (Windows, iOS, ARM targets, Android).
///
/// Pre-condition: None.
/// Post-condition: Panics if called, indicating sandboxing is not available.
#[cfg(any(
    target_os = "windows",
    target_os = "ios",
    target_os = "android",
    target_arch = "arm",
    target_arch = "aarch64",
    target_env = "ohos",
))]
fn create_sandbox() {
    panic!("Sandboxing is not supported on Windows, iOS, ARM targets and android.");
}

/// Enumerates the different types of user agents that Servo can emulate.
enum UserAgent {
    /// A desktop user agent.
    Desktop,
    /// An Android mobile user agent.
    Android,
    /// An OpenHarmony mobile user agent.
    OpenHarmony,
    /// An iOS mobile user agent.
    #[allow(non_camel_case_types)]
    iOS,
}

/// Returns the current version string of Servo.
///
/// Post-condition: A static string slice containing the Cargo package version is returned.
fn get_servo_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Generates a default user agent string based on the specified `UserAgent` type and platform.
///
/// # Arguments
/// * `agent` - The `UserAgent` type for which to generate the string.
///
/// Post-condition: A `String` containing the appropriate user agent string is returned.
fn default_user_agent_string_for(agent: UserAgent) -> String {
    let servo_version = get_servo_version();

    #[cfg(all(target_os = "linux", target_arch = "x86_64", not(target_env = "ohos")))]
    let desktop_ua_string =
        format!("Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Servo/{servo_version} Firefox/128.0");
    #[cfg(all(
        target_os = "linux",
        not(target_arch = "x86_64"),
        not(target_env = "ohos")
    ))]
    let desktop_ua_string =
        format!("Mozilla/5.0 (X11; Linux i686; rv:128.0) Servo/{servo_version} Firefox/128.0");

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    let desktop_ua_string = format!(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Servo/{servo_version} Firefox/128.0"
    );
    #[cfg(all(target_os = "windows", not(target_arch = "x86_64")))]
    let desktop_ua_string =
        format!("Mozilla/5.0 (Windows NT 10.0; rv:128.0) Servo/{servo_version} Firefox/128.0");

    #[cfg(target_os = "macos")]
    let desktop_ua_string = format!(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:128.0) Servo/{servo_version} Firefox/128.0"
    );

    #[cfg(any(target_os = "android", target_env = "ohos"))]
    let desktop_ua_string = "".to_string();

    match agent {
        UserAgent::Desktop => desktop_ua_string,
        UserAgent::Android => format!(
            "Mozilla/5.0 (Android; Mobile; rv:128.0) Servo/{servo_version} Firefox/128.0"
        ),
        UserAgent::OpenHarmony => format!(
            "Mozilla/5.0 (OpenHarmony; Mobile; rv:128.0) Servo/{servo_version} Firefox/128.0"
        ),
        UserAgent::iOS => format!(
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X; rv:128.0) Servo/{servo_version} Firefox/128.0"
        ),
    }
}

/// The default user agent for Android.
#[cfg(target_os = "android")]
const DEFAULT_USER_AGENT: UserAgent = UserAgent::Android;

/// The default user agent for OpenHarmony.
#[cfg(target_env = "ohos")]
const DEFAULT_USER_AGENT: UserAgent = UserAgent::OpenHarmony;

/// The default user agent for iOS.
#[cfg(target_os = "ios")]
const DEFAULT_USER_AGENT: UserAgent = UserAgent::iOS;

/// The default user agent for desktop platforms (non-Android, non-iOS, non-OpenHarmony).
#[cfg(not(any(target_os = "android", target_os = "ios", target_env = "ohos")))]
const DEFAULT_USER_AGENT: UserAgent = UserAgent::Desktop;
