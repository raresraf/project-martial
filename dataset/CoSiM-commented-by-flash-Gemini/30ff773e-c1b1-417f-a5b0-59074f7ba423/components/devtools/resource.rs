/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @file resource.rs
 * @brief This file defines traits and data structures for handling resource availability
 *        and updates within the devtools server. It provides a standardized way for
 *        actors to notify connected devtools clients about new or changed resources.
 * Algorithm: Implements a trait `ResourceAvailable` that actors can use to send
 *            structured messages about resources (available or updated) to connected
 *            clients via JSON packets.
 * Time Complexity: O(1) for generating and sending resource messages, assuming
 *                  serialization and TCP write operations are efficient.
 * Space Complexity: O(N) where N is the size of the resource data being serialized.
 */

use std::net::TcpStream;

use serde::Serialize;

use crate::protocol::JsonPacketStream;

/**
 * @brief Enum to specify the type of resource array being sent.
 *
 * This enum differentiates between resources that have become newly available
 * and resources that have been updated.
 */
pub enum ResourceArrayType {
    /// Indicates that the resources in the array are newly available.
    Available,
    /// Indicates that the resources in the array have been updated.
    Updated,
}

/**
 * @brief Represents a reply message containing an array of resources.
 *
 * This struct is used to serialize and send information about resources
 * (either newly available or updated) to devtools clients.
 *
 * @tparam T The type of the resource being serialized.
 */
#[derive(Serialize)]
pub(crate) struct ResourceAvailableReply<T: Serialize> {
    // from: The name of the actor sending the reply.
    pub from: String,
    // type_: The type of the reply, indicating if resources are available or updated.
    #[serde(rename = "type")]
    pub type_: String,
    // array: A vector of tuples, where each tuple contains the resource type as a string
    //        and a vector of the actual resource data.
    pub array: Vec<(String, Vec<T>)>,
}

/**
 * @brief Trait for actors that can report resource availability or updates.
 *
 * This trait provides methods for actors to send messages to devtools clients
 * indicating that certain resources are available or have been updated.
 */
pub(crate) trait ResourceAvailable {
    /// @brief Returns the name of the actor implementing this trait.
    fn actor_name(&self) -> String;

    /**
     * @brief Sends a message to a devtools client about a single resource.
     *
     * @tparam T The type of the resource to be serialized.
     * @param resource The single resource data.
     * @param resource_type A string identifying the type of the resource (e.g., "network-event", "source").
     * @param array_type The `ResourceArrayType` indicating if the resource is available or updated.
     * @param stream A mutable reference to the `TcpStream` of the client.
     */
    fn resource_array<T: Serialize>(
        &self,
        resource: T,
        resource_type: String,
        array_type: ResourceArrayType,
        stream: &mut TcpStream,
    ) {
        // Block Logic: Delegates to `resources_array` method for a single resource.
        self.resources_array(vec![resource], resource_type, array_type, stream);
    }

    /**
     * @brief Sends a message to a devtools client about multiple resources.
     *
     * @tparam T The type of the resources to be serialized.
     * @param resources A vector of resource data.
     * @param resource_type A string identifying the type of the resources.
     * @param array_type The `ResourceArrayType` indicating if the resources are available or updated.
     * @param stream A mutable reference to the `TcpStream` of the client.
     */
    fn resources_array<T: Serialize>(
        &self,
        resources: Vec<T>,
        resource_type: String,
        array_type: ResourceArrayType,
        stream: &mut TcpStream,
    ) {
        // Block Logic: Constructs the `ResourceAvailableReply` message.
        let msg = ResourceAvailableReply::<T> {
            from: self.actor_name(),
            type_: match array_type {
                ResourceArrayType::Available => "resources-available-array".to_string(),
                ResourceArrayType::Updated => "resources-updated-array".to_string(),
            },
            array: vec![(resource_type, resources)],
        };

        // Block Logic: Writes the JSON message to the client stream.
        let _ = stream.write_json_packet(&msg);
    }
}
