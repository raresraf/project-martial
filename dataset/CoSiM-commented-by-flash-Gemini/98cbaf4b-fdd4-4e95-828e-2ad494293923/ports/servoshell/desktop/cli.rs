/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! This module provides the command-line interface (CLI) entry point for the desktop
//! `servoshell` application. It is responsible for initializing fundamental system
//! components such as crash handling, cryptography, resource management, and tracing.
//! The CLI parses command-line arguments to determine the operational mode
//! (Chrome process or Content process) and then launches the appropriate application
//! loop, orchestrating the startup and shutdown of the Servo shell.

use std::{env, panic};

use crate::desktop::app::App;
use crate::desktop::events_loop::EventsLoop;
use crate::panic_hook;
use crate::prefs::{parse_command_line_arguments, ArgumentParsingResult};

/// The main entry point for the `servoshell` desktop application.
///
/// This function initializes various global components, parses command-line arguments,
/// and then dispatches to either the Chrome process or a Content process,
/// depending on the arguments provided. It sets up crash handling, cryptography,
/// resource loading, and tracing.
///
/// Post-condition: The application's event loop is started, or a content process is initiated.
/// Upon exit, platform-specific deinitialization is performed.
pub fn main() {
    crate::crash_handler::install();
    crate::init_crypto();
    crate::resources::init();

    // TODO: once log-panics is released, can this be replaced by
    // log_panics::init()?
    // Functional Utility: Sets a custom panic hook to handle panics gracefully.
    panic::set_hook(Box::new(panic_hook::panic_hook));

    let args = env::args().collect();
    // Block Logic: Parses command-line arguments to determine the application's role (Chrome or Content process).
    let (opts, preferences, servoshell_preferences) = match parse_command_line_arguments(args) {
        // If it's a content process, run the content process and return.
        ArgumentParsingResult::ContentProcess(token) => return servo::run_content_process(token),
        // Otherwise, extract the options for the Chrome process.
        ArgumentParsingResult::ChromeProcess(opts, preferences, servoshell_preferences) => {
            (opts, preferences, servoshell_preferences)
        },
    };

    // Functional Utility: Initializes tracing based on the parsed configuration.
    crate::init_tracing(servoshell_preferences.tracing_filter.as_deref());

    let clean_shutdown = servoshell_preferences.clean_shutdown;
    let has_output_file = servoshell_preferences.output_image_path.is_some();
    // Functional Utility: Creates the main event loop for the application.
    let event_loop = EventsLoop::new(servoshell_preferences.headless, has_output_file)
        .expect("Failed to create events loop");

    // Block Logic: Creates and runs the main application `App` instance within the event loop.
    {
        let mut app = App::new(opts, preferences, servoshell_preferences, &event_loop);
        event_loop.run_app(&mut app);
    }

    // Functional Utility: Performs platform-specific deinitialization tasks.
    crate::platform::deinit(clean_shutdown)
}
