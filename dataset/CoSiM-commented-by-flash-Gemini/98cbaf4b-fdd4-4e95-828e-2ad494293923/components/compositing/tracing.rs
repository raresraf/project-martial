//! This module provides tracing utilities for logging events originating from the
//! `constellation` component within the compositor. It defines a macro for
//! conditional trace logging and a trait to determine the appropriate log target
//! for different message types, facilitating granular control over tracing output.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/// Log an event from constellation at trace level.
/// - To disable tracing: RUST_LOG='compositor<constellation@=off'
/// - To enable tracing: RUST_LOG='compositor<constellation@'
macro_rules! trace_msg_from_constellation {
    // This macro only exists to put the docs in the same file as the target prefix,
    // so the macro definition is always the same.
    ($event:expr, $($rest:tt)+) => {
        ::log::trace!(target: $crate::tracing::LogTarget::log_target(&$event), $($rest)+)
    };
}

/// Provides a method to determine the static log target string for a given type,
/// typically used for messages originating from the `constellation` component.
pub(crate) trait LogTarget {
    fn log_target(&self) -> &'static str;
}

mod from_constellation {
    use super::LogTarget;

    macro_rules! target {
        ($($name:literal)+) => {
            concat!("compositor<constellation@", $($name),+)
        };
    }

    /// Implements `LogTarget` for `compositing_traits::CompositorMsg` to provide
    /// specific log targets based on the message variant. This allows for
    /// fine-grained control over tracing output for different compositor messages.
    impl LogTarget for compositing_traits::CompositorMsg {
        fn log_target(&self) -> &'static str {
            match self {
                Self::ChangeRunningAnimationsState(..) => target!("ChangeRunningAnimationsState"),
                Self::CreateOrUpdateWebView(..) => target!("CreateOrUpdateWebView"),
                Self::RemoveWebView(..) => target!("RemoveWebView"),
                Self::TouchEventProcessed(..) => target!("TouchEventProcessed"),
                Self::CreatePng(..) => target!("CreatePng"),
                Self::IsReadyToSaveImageReply(..) => target!("IsReadyToSaveImageReply"),
                Self::SetThrottled(..) => target!("SetThrottled"),
                Self::NewWebRenderFrameReady(..) => target!("NewWebRenderFrameReady"),
                Self::PipelineExited(..) => target!("PipelineExited"),
                Self::PendingPaintMetric(..) => target!("PendingPaintMetric"),
                Self::LoadComplete(..) => target!("LoadComplete"),
                Self::WebDriverMouseButtonEvent(..) => target!("WebDriverMouseButtonEvent"),
                Self::WebDriverMouseMoveEvent(..) => target!("WebDriverMouseMoveEvent"),
                Self::CrossProcess(_) => target!("CrossProcess"),
            }
        }
    }
}