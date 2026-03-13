//! This crate provides the core components for a GPU-accelerated compositor,
//! handling rendering, window management, and inter-process communication
//! with other parts of the system like Constellation and WebRender.
//!
//! The primary structure, `InitialCompositorState`, encapsulates all necessary
//! dependencies and communication channels for initializing and operating
//! the compositor.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

#![deny(unsafe_code)]

use std::cell::Cell;
use std::rc::Rc;

use compositing_traits::{CompositorProxy, CompositorReceiver, ConstellationMsg};
use crossbeam_channel::Sender;
use embedder_traits::ShutdownState;
use profile_traits::{mem, time};
use webrender::RenderApi;
use webrender_api::DocumentId;
use webrender_traits::rendering_context::RenderingContext;

pub use crate::compositor::IOCompositor;

#[macro_use]
mod tracing;

mod compositor;
mod touch;
pub mod webview;
pub mod windowing;

/// `InitialCompositorState` encapsulates all the necessary data and communication
/// channels required to construct and initialize the compositor.
/// This includes proxies for inter-process communication, profiler channels,
/// shutdown state, and WebRender API instances.
pub struct InitialCompositorState {
    /// A channel for sending messages to the compositor.
    pub sender: CompositorProxy,
    /// A port for receiving messages destined for the compositor.
    pub receiver: CompositorReceiver,
    /// A channel for sending messages to the `Constellation` component.
    pub constellation_chan: Sender<ConstellationMsg>,
    /// A channel for communicating with the time profiler thread.
    pub time_profiler_chan: time::ProfilerChan,
    /// A channel for communicating with the memory profiler thread.
    pub mem_profiler_chan: mem::ProfilerChan,
    /// A shared state indicating whether Servo has started or is in the process of
    /// shutting down. This is an `Rc<Cell<T>>` to allow interior mutability and shared
    /// ownership among multiple components that need to observe or update the shutdown status.
    pub shutdown_state: Rc<Cell<ShutdownState>>,
    /// An instance of the `webrender::Renderer` used for managing rendering processes.
    pub webrender: webrender::Renderer,
    /// The `DocumentId` associated with the main WebRender document.
    pub webrender_document: DocumentId,
    /// The `RenderApi` instance from WebRender, providing an interface for rendering operations.
    pub webrender_api: RenderApi,
    /// A reference-counted rendering context, enabling shared access to GPU resources
    /// and rendering capabilities.
    pub rendering_context: Rc<dyn RenderingContext>,
    /// A reference-counted instance of `gleam::gl::Gl`, providing access to OpenGL
    /// functions for low-level GPU operations.
    pub webrender_gl: Rc<dyn gleam::gl::Gl>,
    #[cfg(feature = "webxr")]
    /// Registry for WebXR main thread interactions, enabled only when the "webxr" feature is compiled.
    pub webxr_main_thread: webxr::MainThreadRegistry,
}
