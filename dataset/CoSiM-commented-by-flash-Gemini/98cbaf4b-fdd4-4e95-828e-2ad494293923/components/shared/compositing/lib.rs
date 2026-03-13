/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! This module defines the communication interfaces and message types for inter-thread
//! communication with the compositor thread in Servo. It provides `CompositorProxy`
//! for sending messages and `CompositorReceiver` for handling incoming messages,
//! enabling various components to interact with the rendering engine. The `CompositorMsg`
//! enum encapsulates a wide range of commands and notifications, from animation state
//! changes to WebDriver events.

mod constellation_msg;

use std::fmt::{Debug, Error, Formatter};

use base::id::{PipelineId, TopLevelBrowsingContextId};
use base::Epoch;
pub use constellation_msg::ConstellationMsg;
use crossbeam_channel::{Receiver, Sender};
use embedder_traits::{EventLoopWaker, MouseButton, MouseButtonAction};
use euclid::Rect;
use ipc_channel::ipc::IpcSender;
use log::warn;
use pixels::Image;
use script_traits::{AnimationState, EventResult, ScriptThreadMessage};
use style_traits::CSSPixel;
use webrender_api::DocumentId;
use webrender_traits::{CrossProcessCompositorApi, CrossProcessCompositorMessage};

/// `CompositorProxy` provides an interface for sending messages to the compositor thread.
///
/// It encapsulates both a direct `crossbeam_channel::Sender` for in-process messages
/// and a `CrossProcessCompositorApi` for messages that need to be sent across IPC boundaries.
#[derive(Clone)]
pub struct CompositorProxy {
    /// The direct sender for `CompositorMsg` to the compositor thread.
    pub sender: Sender<CompositorMsg>,
    /// Access to [`Self::sender`] that is possible to send across an IPC
    /// channel. These messages are routed via the router thread to
    /// [`Self::sender`].
    pub cross_process_compositor_api: CrossProcessCompositorApi,
    /// A waker to notify the event loop that new messages are available.
    pub event_loop_waker: Box<dyn EventLoopWaker>,
}

impl CompositorProxy {
    /// Sends a `CompositorMsg` to the compositor thread and wakes up its event loop.
    ///
    /// # Arguments
    /// * `msg` - The `CompositorMsg` to send.
    ///
    /// Post-condition: The message is sent to the compositor, and its event loop is awakened.
    /// A warning is logged if the message fails to send.
    pub fn send(&self, msg: CompositorMsg) {
        if let Err(err) = self.sender.send(msg) {
            warn!("Failed to send response ({:?}).", err);
        }
        self.event_loop_waker.wake();
    }
}

/// `CompositorReceiver` is the receiving endpoint for messages destined for the compositor thread.
pub struct CompositorReceiver {
    /// The receiver for `CompositorMsg` from other threads.
    pub receiver: Receiver<CompositorMsg>,
}

impl CompositorReceiver {
    /// Attempts to receive a `CompositorMsg` without blocking.
    ///
    /// Post-condition: Returns `Some(CompositorMsg)` if a message is available, `None` otherwise.
    pub fn try_recv_compositor_msg(&mut self) -> Option<CompositorMsg> {
        self.receiver.try_recv().ok()
    }
    /// Receives a `CompositorMsg`, blocking until a message is available.
    ///
    /// Post-condition: Returns the received `CompositorMsg`. Panics if the sender is disconnected.
    pub fn recv_compositor_msg(&mut self) -> CompositorMsg {
        self.receiver.recv().unwrap()
    }
}

/// `CompositorMsg` defines the types of messages that can be sent to the compositor thread.
/// These messages instruct the compositor to perform various rendering-related tasks,
/// manage webviews, or communicate with other components.
pub enum CompositorMsg {
    /// Alerts the compositor that the given pipeline has changed whether it is running animations.
    ChangeRunningAnimationsState(PipelineId, AnimationState),
    /// Create or update a webview, given its frame tree.
    CreateOrUpdateWebView(SendableFrameTree),
    /// Remove a webview.
    RemoveWebView(TopLevelBrowsingContextId),
    /// Script has handled a touch event, and either prevented or allowed default actions.
    TouchEventProcessed(EventResult),
    /// Composite to a PNG file and return the Image over a passed channel.
    CreatePng(Option<Rect<f32, CSSPixel>>, IpcSender<Option<Image>>),
    /// A reply to the compositor asking if the output image is stable.
    IsReadyToSaveImageReply(bool),
    /// Set whether to use less resources by stopping animations.
    SetThrottled(PipelineId, bool),
    /// WebRender has produced a new frame. This message informs the compositor that
    /// the frame is ready. It contains a bool to indicate if it needs to composite and the
    /// `DocumentId` of the new frame.
    NewWebRenderFrameReady(DocumentId, bool),
    /// A pipeline was shut down.
    // This message acts as a synchronization point between the constellation,
    // when it shuts down a pipeline, to the compositor; when the compositor
    // sends a reply on the IpcSender, the constellation knows it's safe to
    // tear down the other threads associated with this pipeline.
    PipelineExited(PipelineId, IpcSender<()>),
    /// Indicates to the compositor that it needs to record the time when the frame with
    /// the given ID (epoch) is painted and report it to the layout of the given
    /// pipeline ID.
    PendingPaintMetric(PipelineId, Epoch),
    /// The load of a page has completed
    LoadComplete(TopLevelBrowsingContextId),
    /// WebDriver mouse button event
    WebDriverMouseButtonEvent(MouseButtonAction, MouseButton, f32, f32),
    /// WebDriver mouse move event
    WebDriverMouseMoveEvent(f32, f32),

    /// Messages forwarded to the compositor by the constellation from other crates. These
    /// messages are mainly passed on from the compositor to WebRender.
    CrossProcess(CrossProcessCompositorMessage),
}

/// `SendableFrameTree` represents a serializable structure of a frame tree,
/// used for transmitting frame hierarchy information across threads or processes.
pub struct SendableFrameTree {
    /// The `CompositionPipeline` associated with this frame tree node.
    pub pipeline: CompositionPipeline,
    /// A vector of child `SendableFrameTree` instances, representing nested frames.
    pub children: Vec<SendableFrameTree>,
}

/// `CompositionPipeline` contains the essential subset of pipeline information
/// required for layer composition by the compositor.
#[derive(Clone)]
pub struct CompositionPipeline {
    /// The unique identifier for this pipeline.
    pub id: PipelineId,
    /// The unique identifier of the top-level browsing context this pipeline belongs to.
    pub top_level_browsing_context_id: TopLevelBrowsingContextId,
    /// An IPC sender for messages to the script thread associated with this pipeline.
    pub script_chan: IpcSender<ScriptThreadMessage>,
}

impl Debug for CompositorMsg {
    /// Formats the `CompositorMsg` for debugging purposes.
    ///
    /// # Arguments
    /// * `f` - A mutable reference to the `Formatter`.
    ///
    /// Post-condition: The message is written to the formatter in a human-readable format.
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match *self {
            CompositorMsg::ChangeRunningAnimationsState(_, state) => {
                write!(f, "ChangeRunningAnimationsState({:?})", state)
            },
            CompositorMsg::CreateOrUpdateWebView(..) => write!(f, "CreateOrUpdateWebView"),
            CompositorMsg::RemoveWebView(..) => write!(f, "RemoveWebView"),
            CompositorMsg::TouchEventProcessed(..) => write!(f, "TouchEventProcessed"),
            CompositorMsg::CreatePng(..) => write!(f, "CreatePng"),
            CompositorMsg::IsReadyToSaveImageReply(..) => write!(f, "IsReadyToSaveImageReply"),
            CompositorMsg::SetThrottled(..) => write!(f, "SetThrottled"),
            CompositorMsg::PipelineExited(..) => write!(f, "PipelineExited"),
            CompositorMsg::NewWebRenderFrameReady(..) => write!(f, "NewWebRenderFrameReady"),
            CompositorMsg::PendingPaintMetric(..) => write!(f, "PendingPaintMetric"),
            CompositorMsg::LoadComplete(..) => write!(f, "LoadComplete"),
            CompositorMsg::WebDriverMouseButtonEvent(..) => write!(f, "WebDriverMouseButtonEvent"),
            CompositorMsg::WebDriverMouseMoveEvent(..) => write!(f, "WebDriverMouseMoveEvent"),
            CompositorMsg::CrossProcess(..) => write!(f, "CrossProcess"),
        }
    }
}
