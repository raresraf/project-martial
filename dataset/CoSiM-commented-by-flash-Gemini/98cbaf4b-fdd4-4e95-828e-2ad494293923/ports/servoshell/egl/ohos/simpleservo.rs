/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */
//! This module implements a minimal Servo embedding for the OpenHarmony platform.
//! It mirrors the functionality of `simpleservo` for Android, adapting it to
//! OpenHarmony's specific windowing and graphics APIs. The code is responsible
//! for initializing the Servo engine, parsing command-line arguments to configure
//! the environment, setting up the rendering context using `surfman` with
//! OpenHarmony's native window handles, and configuring callbacks for the host
//! environment to facilitate communication between Servo and the OpenHarmony application.
//! It focuses on the core setup needed to get a basic Servo WebView running on the platform.

use std::cell::RefCell;
use std::os::raw::c_void;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::rc::Rc;

use log::{debug, info};
use raw_window_handle::{
    DisplayHandle, OhosDisplayHandle, OhosNdkWindowHandle, RawDisplayHandle, RawWindowHandle,
    WindowHandle,
};
/// The EventLoopWaker::wake function will be called from any thread.
/// It will be called to notify embedder that some events are available,
/// and that perform_updates need to be called
pub use servo::EventLoopWaker;
use servo::{self, resources, Servo, WindowRenderingContext};
use xcomponent_sys::OH_NativeXComponent;

use crate::egl::app_state::{
    Coordinates, RunningAppState, ServoEmbedderCallbacks, ServoWindowCallbacks,
};
use crate::egl::host_trait::HostTrait;
use crate::egl::ohos::resources::ResourceReaderInstance;
use crate::egl::ohos::InitOpts;
use crate::prefs::{parse_command_line_arguments, ArgumentParsingResult};

/// Initializes the Servo embedding for the OpenHarmony platform.
///
/// This function sets up the necessary environment for Servo to run, including
/// cryptography, resource management, command-line argument parsing, and
/// the rendering context. It creates a `RunningAppState` which encapsulates
/// the Servo instance and its associated components.
///
/// # Arguments
/// * `options` - Initialization options for Servo on OpenHarmony.
/// * `native_window` - A raw pointer to the native window handle.
/// * `xcomponent` - A raw pointer to the `OH_NativeXComponent` for graphics.
/// * `waker` - An `EventLoopWaker` to wake up the event loop.
/// * `callbacks` - A boxed trait object for host-specific callbacks.
///
/// Pre-condition: `native_window` and `xcomponent` are valid pointers.
/// Post-condition: A `Result` is returned, containing an `Rc<RunningAppState>` on success,
/// or an error message if initialization fails.
pub fn init(
    options: InitOpts,
    native_window: *mut c_void,
    xcomponent: *mut OH_NativeXComponent,
    waker: Box<dyn EventLoopWaker>,
    callbacks: Box<dyn HostTrait>,
) -> Result<Rc<RunningAppState>, &'static str> {
    info!("Entered simpleservo init function");
    // Functional Utility: Initializes Servo's cryptographic components.
    crate::init_crypto();
    // Functional Utility: Sets up Servo's resource reader for OpenHarmony.
    let resource_dir = PathBuf::from(&options.resource_dir).join("servo");
    resources::set(Box::new(ResourceReaderInstance::new(resource_dir)));

    // It would be nice if `from_cmdline_args()` could accept str slices, to avoid allocations here.
    // Then again, this code could and maybe even should be disabled in production builds.
    let mut args = vec!["servoshell".to_string()];
    // Block Logic: Extends command-line arguments from initialization options.
    args.extend(
        options
            .commandline_args
            .split("\u{1f}")
            .map(|arg| arg.to_string()),
    );
    debug!("Servo commandline args: {:?}", args);

    // Block Logic: Parses command-line arguments to determine Servo's configuration.
    let (opts, preferences, servoshell_preferences) = match parse_command_line_arguments(args) {
        // Invariant: OpenHarmony does not support multiprocess mode yet.
        ArgumentParsingResult::ContentProcess(..) => {
            unreachable!("OHOS does not have support for multiprocess yet.")
        },
        ArgumentParsingResult::ChromeProcess(opts, preferences, servoshell_preferences) => {
            (opts, preferences, servoshell_preferences)
        },
    };

    // Functional Utility: Initializes tracing based on the parsed configuration.
    crate::init_tracing(servoshell_preferences.tracing_filter.as_deref());

    // Block Logic: Retrieves the size of the XComponent.
    let Ok(window_size) = (unsafe { super::get_xcomponent_size(xcomponent, native_window) }) else {
        return Err("Failed to get xcomponent size");
    };
    // Functional Utility: Creates `Coordinates` object based on the retrieved window size.
    let coordinates = Coordinates::new(
        0,
        0,
        window_size.width,
        window_size.height,
        window_size.width,
        window_size.height,
    );

    // Functional Utility: Creates raw display handle for OpenHarmony.
    let display_handle = RawDisplayHandle::Ohos(OhosDisplayHandle::new());
    let display_handle = unsafe { DisplayHandle::borrow_raw(display_handle) };

    // Functional Utility: Creates raw window handle for OpenHarmony.
    let native_window = NonNull::new(native_window).expect("Could not get native window");
    let window_handle = RawWindowHandle::OhosNdk(OhosNdkWindowHandle::new(native_window));
    let window_handle = unsafe { WindowHandle::borrow_raw(window_handle) };

    // Functional Utility: Creates a `WindowRenderingContext` for the OpenHarmony window.
    let rendering_context = Rc::new(
        WindowRenderingContext::new(
            display_handle,
            window_handle,
            &coordinates.framebuffer_size(),
        )
        .expect("Could not create RenderingContext"),
    );

    info!("before ServoWindowCallbacks...");

    // Functional Utility: Initializes `ServoWindowCallbacks` with host-specific callbacks and display density.
    let window_callbacks = Rc::new(ServoWindowCallbacks::new(
        callbacks,
        RefCell::new(coordinates),
        options.display_density as f32,
    ));

    // Functional Utility: Initializes `ServoEmbedderCallbacks`.
    let embedder_callbacks = Box::new(ServoEmbedderCallbacks::new(
        waker,
        #[cfg(feature = "webxr")]
        None,
    ));

    // Functional Utility: Creates the main Servo engine instance.
    let servo = Servo::new(
        opts,
        preferences,
        rendering_context.clone(),
        embedder_callbacks,
        window_callbacks.clone(),
        None, /* user_agent */
    );

    // Functional Utility: Creates the `RunningAppState` which manages the Servo instance.
    let app_state = RunningAppState::new(
        Some(options.url),
        rendering_context,
        servo,
        window_callbacks,
        servoshell_preferences,
    );

    Ok(app_state)
}
