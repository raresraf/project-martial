/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Defines a generic mechanism for notifying the DevTools client about the
//! availability or modification of server-side resources.
//!
//! In the DevTools protocol, many events are communicated by informing the client
//! that a new resource is available (e.g., a new network request) or an existing
//! one has been updated (e.g., a network request has now received a response).
//! This module provides a standardized trait and message format for these notifications.

use std::net::TcpStream;

use serde::Serialize;

use crate::protocol::JsonPacketStream;

/// An enum to distinguish between a new resource and an updated one.
pub enum ResourceArrayType {
    /// Indicates that one or more new resources are available.
    Available,
    /// Indicates that one or more existing resources have been updated.
    Updated,
}

/// A struct representing the JSON message sent to the client to announce
/// available or updated resources. This is a generic structure that can hold
/// any serializable resource type `T`.
#[derive(Serialize)]
pub(crate) struct ResourceAvailableReply<T: Serialize> {
    /// The name of the actor from which this message originates.
    pub from: String,
    /// The type of the notification, either "resources-available-array" or "resources-updated-array".
    #[serde(rename = "type")]
    pub type_: String,
    /// An array containing the resources. The format is a list of tuples,
    /// where each tuple contains the resource type as a string and a vector of the actual resources.
    pub array: Vec<(String, Vec<T>)>,
}

/// A trait for actors that need to notify clients about resources.
/// It provides a common interface for sending resource-related messages.
pub(crate) trait ResourceAvailable {
    /// Returns the name of the actor, used as the `from` field in notification messages.
    fn actor_name(&self) -> String;

    /**
     * @brief Sends a notification for a single resource.
     *
     * @param resource The serializable resource to send.
     * @param resource_type A string identifier for the type of resource (e.g., "network-event").
     * @param array_type Specifies whether the resource is new (`Available`) or modified (`Updated`).
     * @param stream The TCP stream to write the notification to.
     */
    fn resource_array<T: Serialize>(
        &self,
        resource: T,
        resource_type: String,
        array_type: ResourceArrayType,
        stream: &mut TcpStream,
    ) {
        self.resources_array(vec![resource], resource_type, array_type, stream);
    }

    /**
     * @brief Sends a notification for multiple resources of the same type.
     *
     * @param resources A vector of serializable resources to send.
     * @param resource_type A string identifier for the type of the resources.
     * @param array_type Specifies whether the resources are new or updated.
     * @param stream The TCP stream to write the notification to.
     */
    fn resources_array<T: Serialize>(
        &self,
        resources: Vec<T>,
        resource_type: String,
        array_type: ResourceArrayType,
        stream: &mut TcpStream,
    ) {
        // Construct the reply message based on the array type.
        let msg = ResourceAvailableReply::<T> {
            from: self.actor_name(),
            type_: match array_type {
                ResourceArrayType::Available => "resources-available-array".to_string(),
                ResourceArrayType::Updated => "resources-updated-array".to_string(),
            },
            array: vec![(resource_type, resources)],
        };

        // Serialize the message to JSON and send it over the stream.
        let _ = stream.write_json_packet(&msg);
    }
}