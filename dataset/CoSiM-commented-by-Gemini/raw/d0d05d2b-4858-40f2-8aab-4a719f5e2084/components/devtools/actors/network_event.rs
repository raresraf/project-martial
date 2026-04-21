/**
 * @file network_event.rs
 * @brief Implementation of the DevTools Network Event actor for Servo.
 * 
 * Architectural Intent: Manages the lifecycle of network diagnostic data for a single HTTP transaction. 
 * It intercepts network stack events (requests, responses, timing) and exposes them to the 
 * remote DevTools client via a serialized JSON protocol.
 * 
 * Domain-Awareness: Implements the Firefox DevTools Protocol actor interface. 
 * Tracks complex metadata including CORS preflights, XHR status, and sub-millisecond timing resolution.
 */

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Liberally derived from the [Firefox JS implementation](http://mxr.mozilla.org/mozilla-central/source/toolkit/devtools/server/actors/webconsole.js).
//! Handles interaction with the remote web console on network events (HTTP requests, responses) in Servo.

use std::net::TcpStream;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{Local, LocalResult, TimeZone};
use devtools_traits::{HttpRequest as DevtoolsHttpRequest, HttpResponse as DevtoolsHttpResponse};
use headers::{ContentType, Cookie, HeaderMapExt};
use http::{HeaderMap, Method, header};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::StreamId;
use crate::actor::{Actor, ActorMessageStatus, ActorRegistry};
use crate::network_handler::Cause;
use crate::protocol::JsonPacketStream;

/**
 * @struct NetworkEventActor
 * @brief Persistent state container for a network transaction's diagnostic data.
 */
pub struct NetworkEventActor {
    pub name: String,
    pub is_xhr: bool,
    pub request_url: String,
    pub request_method: Method,
    pub request_started: SystemTime,
    pub request_time_stamp: i64,
    pub request_headers_raw: Option<HeaderMap>,
    pub request_body: Option<Vec<u8>>,
    pub request_cookies: Option<RequestCookiesMsg>,
    pub request_headers: Option<RequestHeadersMsg>,
    pub response_headers_raw: Option<HeaderMap>,
    pub response_body: Option<Vec<u8>>,
    pub response_content: Option<ResponseContentMsg>,
    pub response_start: Option<ResponseStartMsg>,
    pub response_cookies: Option<ResponseCookiesMsg>,
    pub response_headers: Option<ResponseHeadersMsg>,
    pub total_time: Duration,
    pub security_state: String,
    pub event_timing: Option<Timings>,
}

// ... (Serialization models) ...

impl Actor for NetworkEventActor {
    fn name(&self) -> String {
        self.name.clone()
    }

    /**
     * @brief Protocol message dispatcher.
     * Functional Utility: Resolves requests from the debugger UI and streams back specific 
     * subsets of the captured network metadata (e.g., just cookies or just headers).
     */
    fn handle_message(
        &self,
        _registry: &ActorRegistry,
        msg_type: &str,
        _msg: &Map<String, Value>,
        stream: &mut TcpStream,
        _id: StreamId,
    ) -> Result<ActorMessageStatus, ()> {
        Ok(match msg_type {
            /**
             * Block Logic: Request header retrieval.
             * Invariant: Dynamically reconstructs the raw header string from the captured HeaderMap.
             */
            "getRequestHeaders" => {
                let mut headers: Vec<Header> = Vec::new();
                let mut raw_headers_string = "".to_owned();
                let mut headers_size = 0;
                if let Some(ref headers_map) = self.request_headers_raw {
                    for (name, value) in headers_map.iter() {
                        let value = &value.to_str().unwrap().to_string();
                        raw_headers_string =
                            raw_headers_string + name.as_str() + ":" + value + "\r\n";
                        headers_size += name.as_str().len() + value.len();
                        headers.push(Header {
                            name: name.as_str().to_owned(),
                            value: value.to_owned(),
                        });
                    }
                }

                let msg = GetRequestHeadersReply {
                    from: self.name(),
                    headers,
                    header_size: headers_size,
                    raw_headers: raw_headers_string,
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },
            // ... (Other handlers) ...
            _ => ActorMessageStatus::Ignored,
        })
    }
}

impl NetworkEventActor {
    pub fn new(name: String) -> NetworkEventActor {
        NetworkEventActor {
            name,
            is_xhr: false,
            request_url: String::new(),
            request_method: Method::GET,
            request_started: SystemTime::now(),
            request_time_stamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            request_headers_raw: None,
            request_body: None,
            request_cookies: None,
            request_headers: None,
            response_headers_raw: None,
            response_body: None,
            response_content: None,
            response_start: None,
            response_cookies: None,
            response_headers: None,
            total_time: Duration::ZERO,
            security_state: "insecure".to_owned(),
            event_timing: None,
        }
    }

    /**
     * @brief Updates the actor state with metadata from an outgoing HTTP request.
     */
    pub fn add_request(&mut self, request: DevtoolsHttpRequest) {
        self.is_xhr = request.is_xhr;
        self.request_cookies = Some(Self::request_cookies(&request));
        self.request_headers = Some(Self::request_headers(&request));
        self.total_time = Self::total_time(&request);
        self.event_timing = Some(Self::event_timing(&request));
        self.request_url = request.url.to_string();
        self.request_method = request.method;
        self.request_started = request.started_date_time;
        self.request_time_stamp = request.time_stamp;
        self.request_body = request.body.clone();
        self.request_headers_raw = Some(request.headers.clone());
    }

    /**
     * @brief Updates the actor state with metadata from an incoming HTTP response.
     */
    pub fn add_response(&mut self, response: DevtoolsHttpResponse) {
        self.response_headers = Some(Self::response_headers(&response));
        self.response_cookies = Some(Self::response_cookies(&response));
        self.response_start = Some(Self::response_start(&response));
        self.response_content = Some(Self::response_content(&response));
        self.response_body = response.body.clone();
        self.response_headers_raw = response.headers.clone();
    }

    /**
     * @brief Generates a high-level summary of the network event for resource listing.
     * Logic: Categorizes the 'cause' of the request based on file extensions.
     */
    pub fn event_actor(&self) -> EventActor {
        let started_datetime_rfc3339 = match Local.timestamp_millis_opt(
            self.request_started
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as i64,
        ) {
            LocalResult::None => "".to_owned(),
            LocalResult::Single(date_time) => date_time.to_rfc3339().to_string(),
            LocalResult::Ambiguous(date_time, _) => date_time.to_rfc3339().to_string(),
        };

        let cause_type = match self.request_url.as_str() {
            url if url.ends_with(".css") => "stylesheet",
            url if url.ends_with(".js") => "script",
            url if url.ends_with(".png") || url.ends_with(".jpg") => "img",
            _ => "document",
        };

        EventActor {
            actor: self.name(),
            url: self.request_url.clone(),
            method: format!("{}", self.request_method),
            started_date_time: started_datetime_rfc3339,
            time_stamp: self.request_time_stamp,
            is_xhr: self.is_xhr,
            private: false,
            cause: Cause {
                type_: cause_type.to_string(),
                loading_document_uri: None, 
            },
        }
    }

    // ... (Helpers for message construction) ...
}
