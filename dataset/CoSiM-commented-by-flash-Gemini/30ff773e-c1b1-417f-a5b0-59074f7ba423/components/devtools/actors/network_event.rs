/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @file network_event.rs
 * @brief Implements the `NetworkEventActor` for the devtools server, handling
 *        network events (HTTP requests and responses) and mediating their
 *        interaction with the remote web console.
 * Algorithm: This actor captures and processes HTTP request and response data,
 *            stores relevant information, and provides methods to expose this
 *            data to devtools clients. It tracks the state of network resources
 *            and reports updates.
 * Time Complexity: Handling of individual network events is generally O(1),
 *                  with serialization overhead.
 * Space Complexity: O(N_events * S_event) where N_events is the number of tracked
 *                   network events and S_event is the memory consumed per event.
 */

use std::net::TcpStream;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{Local, LocalResult, TimeZone};
use devtools_traits::{HttpRequest as DevtoolsHttpRequest, HttpResponse as DevtoolsHttpResponse};
use headers::{ContentType, Cookie, HeaderMapExt};
use http::{HeaderMap, Method, header};
use net_traits::http_status::HttpStatus;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::StreamId;
use crate::actor::{Actor, ActorMessageStatus, ActorRegistry};
use crate::network_handler::Cause;
use crate::protocol::JsonPacketStream;

/**
 * @brief Represents an HTTP request for network event tracking.
 */
#[derive(Clone)]
struct HttpRequest {
    // url: The URL of the request.
    url: String,
    // method: The HTTP method used (e.g., GET, POST).
    method: Method,
    // headers: The HTTP request headers.
    headers: HeaderMap,
    // body: The request body, if any.
    body: Option<Vec<u8>>,
    // started_date_time: The system time when the request started.
    started_date_time: SystemTime,
    // time_stamp: A timestamp associated with the request.
    time_stamp: i64,
    // connect_time: The duration spent connecting to the server.
    connect_time: Duration,
    // send_time: The duration spent sending the request.
    send_time: Duration,
}

/**
 * @brief Represents an HTTP response for network event tracking.
 */
#[derive(Clone)]
struct HttpResponse {
    // headers: The HTTP response headers, if any.
    headers: Option<HeaderMap>,
    // status: The HTTP status code and text.
    status: HttpStatus,
    // body: The response body, if any.
    body: Option<Vec<u8>>,
}

pub struct NetworkEventActor {
    // name: The unique name of this actor.
    pub name: String,
    // request: The HTTP request data associated with this network event.
    pub request: HttpRequest,
    // response: The HTTP response data associated with this network event.
    pub response: HttpResponse,
    // is_xhr: Indicates if the request was an XMLHttpRequest.
    pub is_xhr: bool,
    // response_content: Optional message containing response content details.
    pub response_content: Option<ResponseContentMsg>,
    // response_start: Optional message containing response start details.
    pub response_start: Option<ResponseStartMsg>,
    // response_cookies: Optional message containing response cookie details.
    pub response_cookies: Option<ResponseCookiesMsg>,
    // response_headers: Optional message containing response header details.
    pub response_headers: Option<ResponseHeadersMsg>,
    // request_cookies: Optional message containing request cookie details.
    pub request_cookies: Option<RequestCookiesMsg>,
    // request_headers: Optional message containing request header details.
    pub request_headers: Option<RequestHeadersMsg>,
    // total_time: The total time taken for the network event.
    pub total_time: Duration,
    // security_state: The security state of the connection (e.g., "insecure", "secure").
    pub security_state: String,
    // event_timing: Optional message containing detailed event timings.
    pub event_timing: Option<Timings>,
}

/**
 * @brief Represents a network event resource to be sent to devtools clients.
 *
 * This struct encapsulates the details of a network event, including its ID,
 * various updates, and associated browsing context information.
 */
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkEventResource {
    // resource_id: A unique identifier for the network resource.
    pub resource_id: u64,
    // resource_updates: A map containing various updated properties of the resource.
    pub resource_updates: Map<String, Value>,
    // browsing_context_id: The ID of the browsing context where the event occurred.
    pub browsing_context_id: u64,
    // inner_window_id: The ID of the inner window where the event occurred.
    pub inner_window_id: u64,
}

/**
 * @brief Represents an event actor for a network request, containing serializable details.
 */
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EventActor {
    // actor: The actor ID of this network event.
    pub actor: String,
    // url: The URL of the network request.
    pub url: String,
    // method: The HTTP method of the request.
    pub method: String,
    // started_date_time: The start date and time of the request in RFC3339 format.
    pub started_date_time: String,
    // time_stamp: A timestamp for the event.
    pub time_stamp: i64,
    // is_xhr: Flag indicating if the request was an XMLHttpRequest.
    #[serde(rename = "isXHR")]
    pub is_xhr: bool,
    // private: Flag indicating if the request was private.
    pub private: bool,
    // cause: The cause of the network event.
    pub cause: Cause,
}

/**
 * @brief Represents a message containing response cookies information.
 */
#[derive(Serialize)]
pub struct ResponseCookiesMsg {
    // cookies: The number of cookies in the response.
    pub cookies: usize,
}

/**
 * @brief Represents a message containing response start details.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResponseStartMsg {
    // http_version: The HTTP protocol version used.
    pub http_version: String,
    // remote_address: The IP address of the remote server.
    pub remote_address: String,
    // remote_port: The port number of the remote server.
    pub remote_port: u32,
    // status: The HTTP status code as a string.
    pub status: String,
    // status_text: The HTTP status text (e.g., "OK", "Not Found").
    pub status_text: String,
    // headers_size: The size of the response headers in bytes.
    pub headers_size: usize,
    // discard_response_body: Flag indicating if the response body was discarded.
    pub discard_response_body: bool,
}

/**
 * @brief Represents a message containing response content details.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResponseContentMsg {
    // mime_type: The MIME type of the response content.
    pub mime_type: String,
    // content_size: The actual size of the response content in bytes.
    pub content_size: u32,
    // transferred_size: The size of the response content transferred over the network in bytes.
    pub transferred_size: u32,
    // discard_response_body: Flag indicating if the response body was discarded.
    pub discard_response_body: bool,
}

/**
 * @brief Represents a message containing response headers details.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResponseHeadersMsg {
    // headers: The number of response headers.
    pub headers: usize,
    // headers_size: The total size of the response headers in bytes.
    pub headers_size: usize,
}

/**
 * @brief Represents a message containing request cookies information.
 */
#[derive(Serialize)]
pub struct RequestCookiesMsg {
    // cookies: The number of cookies in the request.
    pub cookies: usize,
}

/**
 * @brief Represents a message containing request headers details.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestHeadersMsg {
    // headers: The number of request headers.
    headers: usize,
    // headers_size: The total size of the request headers in bytes.
    headers_size: usize,
}

/**
 * @brief Reply message containing request headers information.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GetRequestHeadersReply {
    // from: The name of the actor sending the reply.
    from: String,
    // headers: A list of request headers.
    headers: Vec<Header>,
    // header_size: The total size of the request headers.
    header_size: usize,
    // raw_headers: The raw string representation of the request headers.
    raw_headers: String,
}

/**
 * @brief Represents a single HTTP header.
 */
#[derive(Serialize)]
struct Header {
    // name: The name of the header.
    name: String,
    // value: The value of the header.
    value: String,
}

/**
 * @brief Reply message containing response headers information.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GetResponseHeadersReply {
    // from: The name of the actor sending the reply.
    from: String,
    // headers: A list of response headers.
    headers: Vec<Header>,
    // header_size: The total size of the response headers.
    header_size: usize,
    // raw_headers: The raw string representation of the response headers.
    raw_headers: String,
}

/**
 * @brief Reply message containing response content information.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GetResponseContentReply {
    // from: The name of the actor sending the reply.
    from: String,
    // content: The content of the response body.
    content: Option<Vec<u8>>,
    // content_discarded: Flag indicating if the content was discarded.
    content_discarded: bool,
}

/**
 * @brief Reply message containing request post data information.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GetRequestPostDataReply {
    // from: The name of the actor sending the reply.
    from: String,
    // post_data: The post data of the request.
    post_data: Option<Vec<u8>>,
    // post_data_discarded: Flag indicating if the post data was discarded.
    post_data_discarded: bool,
}

/**
 * @brief Reply message containing request cookies information.
 */
#[derive(Serialize)]
struct GetRequestCookiesReply {
    // from: The name of the actor sending the reply.
    from: String,
    // cookies: The cookies in the request.
    cookies: Vec<u8>,
}

/**
 * @brief Reply message containing response cookies information.
 */
#[derive(Serialize)]
struct GetResponseCookiesReply {
    // from: The name of the actor sending the reply.
    from: String,
    // cookies: The cookies in the response.
    cookies: Vec<u8>,
}

/**
 * @brief Represents detailed timing information for a network event.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Timings {
    // blocked: Time spent blocked before sending the request.
    blocked: u32,
    // dns: Time spent in DNS resolution.
    dns: u32,
    // connect: Time spent establishing a connection.
    connect: u64,
    // send: Time spent sending the request.
    send: u64,
    // wait: Time spent waiting for the first byte of the response.
    wait: u32,
    // receive: Time spent receiving the response.
    receive: u32,
}

/**
 * @brief Reply message containing event timing information for a network request.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GetEventTimingsReply {
    // from: The name of the actor sending the reply.
    from: String,
    // timings: Detailed timing information for the event.
    timings: Timings,
    // total_time: The total time taken for the event.
    total_time: u64,
}

/**
 * @brief Represents security information for a network connection.
 */
#[derive(Serialize)]
struct SecurityInfo {
    // state: The security state of the connection (e.g., "insecure", "secure").
    state: String,
}

/**
 * @brief Reply message containing security information for a network connection.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GetSecurityInfoReply {
    // from: The name of the actor sending the reply.
    from: String,
    // security_info: Detailed security information.
    security_info: SecurityInfo,
}

impl Actor for NetworkEventActor {
    /// @brief Returns the unique name of this actor.
    fn name(&self) -> String {
        self.name.clone()
    }

    /**
     * @brief Handles incoming messages for the `NetworkEventActor`.
     *
     * Dispatches messages based on their type to retrieve specific network event details
     * (e.g., request headers, response content, timings) and sends them back to the client.
     *
     * @param _registry A reference to the actor registry.
     * @param msg_type The type of the incoming message.
     * @param _msg The message payload as a JSON `Map`.
     * @param stream A mutable reference to the `TcpStream` of the client.
     * @param _id The `StreamId` of the client connection.
     * @return A `Result` indicating whether the message was processed or ignored.
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
            // Block Logic: Handles requests for HTTP request headers.
            "getRequestHeaders" => {
                // headers: A vector to store serialized header key-value pairs.
                let mut headers = Vec::new();
                // raw_headers_string: String representation of all raw headers.
                let mut raw_headers_string = "".to_owned();
                // headers_size: Accumulates the total size of header names and values.
                let mut headers_size = 0;
                // Block Logic: Iterate through request headers to populate `headers` and calculate size.
                for (name, value) in self.request.headers.iter() {
                    let value = &value.to_str().unwrap().to_string();
                    raw_headers_string = raw_headers_string + name.as_str() + ":" + value + "\r\n";
                    headers_size += name.as_str().len() + value.len();
                    headers.push(Header {
                        name: name.as_str().to_owned(),
                        value: value.to_owned(),
                    });
                }
                // Block Logic: Construct and send the reply message.
                let msg = GetRequestHeadersReply {
                    from: self.name(),
                    headers,
                    header_size: headers_size,
                    raw_headers: raw_headers_string,
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },
            // Block Logic: Handles requests for HTTP request cookies.
            "getRequestCookies" => {
                // cookies: Vector to store raw cookie bytes.
                let mut cookies = Vec::new();

                // Block Logic: Extract cookie values from request headers.
                for cookie in self.request.headers.get_all(header::COOKIE) {
                    if let Ok(cookie_value) = String::from_utf8(cookie.as_bytes().to_vec()) {
                        cookies = cookie_value.into_bytes();
                    }
                }

                // Block Logic: Construct and send the reply message.
                let msg = GetRequestCookiesReply {
                    from: self.name(),
                    cookies,
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },
            // Block Logic: Handles requests for HTTP request post data (body).
            "getRequestPostData" => {
                // Block Logic: Construct and send the reply message with post data.
                let msg = GetRequestPostDataReply {
                    from: self.name(),
                    post_data: self.request.body.clone(),
                    post_data_discarded: false,
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },
            // Block Logic: Handles requests for HTTP response headers.
            "getResponseHeaders" => {
                if let Some(ref response_headers) = self.response.headers {
                    // headers: A vector to store serialized header key-value pairs.
                    let mut headers = vec![];
                    // raw_headers_string: String representation of all raw headers.
                    let mut raw_headers_string = "".to_owned();
                    // headers_size: Accumulates the total size of header names and values.
                    let mut headers_size = 0;
                    // Block Logic: Iterate through response headers to populate `headers` and calculate size.
                    for (name, value) in response_headers.iter() {
                        headers.push(Header {
                            name: name.as_str().to_owned(),
                            value: value.to_str().unwrap().to_owned(),
                        });
                        headers_size += name.as_str().len() + value.len();
                        raw_headers_string.push_str(name.as_str());
                        raw_headers_string.push(':');
                        raw_headers_string.push_str(value.to_str().unwrap());
                        raw_headers_string.push_str("\r\n");
                    }
                    // Block Logic: Construct and send the reply message.
                    let msg = GetResponseHeadersReply {
                        from: self.name(),
                        headers,
                        header_size: headers_size,
                        raw_headers: raw_headers_string,
                    };
                    let _ = stream.write_json_packet(&msg);
                }
                ActorMessageStatus::Processed
            },
            // Block Logic: Handles requests for HTTP response cookies.
            "getResponseCookies" => {
                // cookies: Vector to store raw cookie bytes.
                let mut cookies = Vec::new();
                // TODO: This seems quite broken
                // Block Logic: Extract cookie values from response headers.
                for cookie in self.request.headers.get_all(header::SET_COOKIE) {
                    if let Ok(cookie_value) = String::from_utf8(cookie.as_bytes().to_vec()) {
                        cookies = cookie_value.into_bytes();
                    }
                }

                // Block Logic: Construct and send the reply message.
                let msg = GetResponseCookiesReply {
                    from: self.name(),
                    cookies,
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },
            // Block Logic: Handles requests for HTTP response content (body).
            "getResponseContent" => {
                // Block Logic: Construct and send the reply message with response content.
                let msg = GetResponseContentReply {
                    from: self.name(),
                    content: self.response.body.clone(),
                    content_discarded: self.response.body.is_none(),
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },
            // Block Logic: Handles requests for network event timings.
            "getEventTimings" => {
                // TODO: This is a fake timings msg
                // timings_obj: Constructed Timings object with request time data.
                let timings_obj = Timings {
                    blocked: 0,
                    dns: 0,
                    connect: self.request.connect_time.as_millis() as u64,
                    send: self.request.send_time.as_millis() as u64,
                    wait: 0,
                    receive: 0,
                };
                // total: Calculated total time for the event.
                let total = timings_obj.connect + timings_obj.send;
                // Block Logic: Construct and send the reply message with event timings.
                // TODO: Send the correct values for all these fields.
                let msg = GetEventTimingsReply {
                    from: self.name(),
                    timings: timings_obj,
                    total_time: total,
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },
            // Block Logic: Handles requests for security information.
            "getSecurityInfo" => {
                // Block Logic: Construct and send the reply message with security info.
                // TODO: Send the correct values for securityInfo.
                let msg = GetSecurityInfoReply {
                    from: self.name(),
                    security_info: SecurityInfo {
                        state: "insecure".to_owned(),
                    },
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },
            _ => ActorMessageStatus::Ignored,
        })
    }
}

impl NetworkEventActor {
    /**
     * @brief Creates a new `NetworkEventActor` instance.
     *
     * Initializes the actor with default HttpRequest and HttpResponse data,
     * and sets up initial flags and security state.
     *
     * @param name The unique name for this actor instance.
     * @return A new `NetworkEventActor` instance.
     */
    pub fn new(name: String) -> NetworkEventActor {
        // request: Initializes an empty HttpRequest object.
        let request = HttpRequest {
            url: String::new(),
            method: Method::GET,
            headers: HeaderMap::new(),
            body: None,
            started_date_time: SystemTime::now(),
            time_stamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            send_time: Duration::ZERO,
            connect_time: Duration::ZERO,
        };
        // response: Initializes an empty HttpResponse object.
        let response = HttpResponse {
            headers: None,
            status: HttpStatus::default(),
            body: None,
        };

        NetworkEventActor {
            name,
            request: request.clone(),
            response: response.clone(),
            is_xhr: false,
            response_content: None,
            response_start: None,
            response_cookies: None,
            response_headers: None,
            request_cookies: None,
            request_headers: None,
            total_time: Self::total_time(&request),
            security_state: "insecure".to_owned(), // Default security state
            event_timing: None,
        }
    }

    pub fn add_request(&mut self, request: DevtoolsHttpRequest) {
        // url: The URL of the request.
        request.url.as_str().clone_into(&mut self.request.url);

        // method: The HTTP method used (e.g., GET, POST).
        self.request.method = request.method.clone();
        // headers: The HTTP request headers.
        self.request.headers = request.headers.clone();
        // body: The request body, if any.
        self.request.body = request.body;
        // started_date_time: The system time when the request started.
        self.request.started_date_time = request.started_date_time;
        // time_stamp: A timestamp associated with the request.
        self.request.time_stamp = request.time_stamp;
        // connect_time: The duration spent connecting to the server.
        self.request.connect_time = request.connect_time;
        // send_time: The duration spent sending the request.
        self.request.send_time = request.send_time;
        // is_xhr: Indicates if the request was an XMLHttpRequest.
        self.is_xhr = request.is_xhr;
    }

    /**
     * @brief Adds HTTP response data to the network event actor.
     * @param response The `DevtoolsHttpResponse` containing the response data.
     */
    pub fn add_response(&mut self, response: DevtoolsHttpResponse) {
        // headers: The HTTP response headers, if any.
        self.response.headers.clone_from(&response.headers);
        // status: The HTTP status code and text.
        self.response.status = response.status;
        // body: The response body, if any.
        self.response.body = response.body;
    }

    /**
     * @brief Creates an `EventActor` representing the current state of the network event.
     *
     * This function formats the stored HTTP request data into a serializable `EventActor`
     * object, suitable for sending to devtools clients.
     *
     * @return An `EventActor` instance with details about the network request.
     */
    pub fn event_actor(&self) -> EventActor {
        // started_datetime_rfc3339: Formatted start date and time of the request.
        // TODO: Send the correct values for startedDateTime, isXHR, private
        let started_datetime_rfc3339 = match Local.timestamp_millis_opt(
            self.request
                .started_date_time
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as i64,
        ) {
            LocalResult::None => "".to_owned(),
            LocalResult::Single(date_time) => date_time.to_rfc3339().to_string(),
            LocalResult::Ambiguous(date_time, _) => date_time.to_rfc3339().to_string(),
        };

        // cause_type: Determines the type of cause for the network event based on URL extension.
        // Block Logic: Determine the cause type based on the request URL's file extension.
        let cause_type = match self.request.url.as_str() {
            // Adjust based on request data
            url if url.ends_with(".css") => "stylesheet",
            url if url.ends_with(".js") => "script",
            url if url.ends_with(".png") || url.ends_with(".jpg") => "img",
            _ => "document",
        };

        EventActor {
            actor: self.name(),
            url: self.request.url.clone(),
            method: format!("{}", self.request.method),
            started_date_time: started_datetime_rfc3339,
            time_stamp: self.request.time_stamp,
            is_xhr: self.is_xhr,
            private: false,
            cause: Cause {
                type_: cause_type.to_string(),
                loading_document_uri: None, // Set if available
            },
        }
    }

    /**
     * @brief Creates a `ResponseStartMsg` from the given HTTP response.
     *
     * This function extracts relevant information from an `HttpResponse` to
     * populate a `ResponseStartMsg` object, suitable for client notification.
     *
     * @param response A reference to the `HttpResponse` data.
     * @return A `ResponseStartMsg` instance.
     */
    #[allow(dead_code)]
    pub fn response_start(response: &HttpResponse) -> ResponseStartMsg {
        // h_size: The number of headers in the response.
        let h_size = response.headers.as_ref().map(|h| h.len()).unwrap_or(0);
        // status: The HTTP status of the response.
        let status = &response.status;

        // TODO: Send the correct values for remoteAddress and remotePort and http_version
        ResponseStartMsg {
            http_version: "HTTP/1.1".to_owned(),
            remote_address: "63.245.217.43".to_owned(),
            remote_port: 443,
            status: status.code().to_string(),
            status_text: String::from_utf8_lossy(status.message()).to_string(),
            headers_size: h_size,
            discard_response_body: false,
        }
    }

    /**
     * @brief Creates a `ResponseContentMsg` from the given HTTP response.
     *
     * Extracts the MIME type, content size, and transferred size from the response.
     *
     * @param response A reference to the `HttpResponse` data.
     * @return A `ResponseContentMsg` instance.
     */
    #[allow(dead_code)]
    pub fn response_content(response: &HttpResponse) -> ResponseContentMsg {
        // mime_type: The MIME type of the content, extracted from headers.
        let mime_type = if let Some(ref headers) = response.headers {
            match headers.typed_get::<ContentType>() {
                Some(ct) => ct.to_string(),
                None => "".to_string(),
            }
        } else {
            "".to_string()
        };
        // TODO: Set correct values when response's body is sent to the devtools in http_loader.
        ResponseContentMsg {
            mime_type,
            content_size: 0,
            transferred_size: 0,
            discard_response_body: true,
        }
    }

    /**
     * @brief Creates a `ResponseCookiesMsg` from the given HTTP response.
     *
     * Counts the number of cookies present in the response headers.
     *
     * @param response A reference to the `HttpResponse` data.
     * @return A `ResponseCookiesMsg` instance.
     */
    #[allow(dead_code)]
    pub fn response_cookies(response: &HttpResponse) -> ResponseCookiesMsg {
        // cookies_size: The number of cookies in the response.
        let mut cookies_size = 0;
        // Block Logic: Extract the number of cookies from the response headers.
        if let Some(ref headers) = response.headers {
            cookies_size = match headers.typed_get::<Cookie>() {
                Some(ref cookie) => cookie.len(),
                _ => 0,
            };
        }
        ResponseCookiesMsg {
            cookies: cookies_size,
        }
    }

    /**
     * @brief Creates a `ResponseHeadersMsg` from the given HTTP response.
     *
     * Extracts the number of headers and their total size from the response.
     *
     * @param response A reference to the `HttpResponse` data.
     * @return A `ResponseHeadersMsg` instance.
     */
    #[allow(dead_code)]
    pub fn response_headers(response: &HttpResponse) -> ResponseHeadersMsg {
        // headers_size: The number of headers.
        let mut headers_size = 0;
        // headers_byte_count: The total size of headers in bytes.
        let mut headers_byte_count = 0;
        // Block Logic: Iterate through response headers to count them and calculate total size.
        if let Some(ref headers) = response.headers {
            headers_size = headers.len();
            for (name, value) in headers.iter() {
                headers_byte_count += name.as_str().len() + value.len();
            }
        }
        ResponseHeadersMsg {
            headers: headers_size,
            headers_size: headers_byte_count,
        }
    }

    /**
     * @brief Creates a `RequestHeadersMsg` from the given HTTP request.
     *
     * Calculates the number of headers and their total size from the request.
     *
     * @param request A reference to the `HttpRequest` data.
     * @return A `RequestHeadersMsg` instance.
     */
    #[allow(dead_code)]
    pub fn request_headers(request: &HttpRequest) -> RequestHeadersMsg {
        // size: The total size of the request headers in bytes.
        let size = request.headers.iter().fold(0, |acc, (name, value)| {
            acc + name.as_str().len() + value.len()
        });
        RequestHeadersMsg {
            headers: request.headers.len(),
            headers_size: size,
        }
    }

    /**
     * @brief Creates a `RequestCookiesMsg` from the given HTTP request.
     *
     * Counts the number of cookies present in the request headers.
     *
     * @param request A reference to the `HttpRequest` data.
     * @return A `RequestCookiesMsg` instance.
     */
    #[allow(dead_code)]
    pub fn request_cookies(request: &HttpRequest) -> RequestCookiesMsg {
        // cookies_size: The number of cookies in the request.
        let cookies_size = match request.headers.typed_get::<Cookie>() {
            Some(ref cookie) => cookie.len(),
            _ => 0,
        };
        RequestCookiesMsg {
            cookies: cookies_size,
        }
    }

    /**
     * @brief Calculates the total time taken for the network event.
     * @param request A reference to the `HttpRequest` data.
     * @return The total time as a `Duration`.
     */
    pub fn total_time(request: &HttpRequest) -> Duration {
        request.connect_time + request.send_time
    }

    /**
     * @brief Inserts a serialized optional object into a JSON `Map`.
     *
     * If the optional object is `Some` and can be serialized to a JSON object,
     * its key-value pairs are inserted into the provided `map`.
     *
     * @param map A mutable reference to the JSON `Map` to insert into.
     * @param obj An optional object to serialize and insert.
     */
    fn insert_serialized_map<T: Serialize>(map: &mut Map<String, Value>, obj: &Option<T>) {
        if let Some(value) = obj {
            // Block Logic: Attempt to serialize the object into a JSON `Value::Object`.
            if let Ok(Value::Object(serialized)) = serde_json::to_value(value) {
                // Block Logic: Insert all key-value pairs from the serialized object into the map.
                for (key, val) in serialized {
                    map.insert(key, val);
                }
            }
        }
    }

    /**
     * @brief Generates a `NetworkEventResource` representing the current updates to a network event.
     *
     * This function constructs a `NetworkEventResource` object by gathering
     * various state flags and serialized details of the HTTP request and response.
     *
     * @return A `NetworkEventResource` instance containing the updated resource information.
     */
    pub fn resource_updates(&self) -> NetworkEventResource {
        // resource_updates: A JSON `Map` to store the updated properties of the resource.
        let mut resource_updates = Map::new();

        // Block Logic: Insert flags indicating the availability of request/response cookies and headers.
        resource_updates.insert(
            "requestCookiesAvailable".to_owned(),
            Value::Bool(self.request_cookies.is_some()),
        );

        resource_updates.insert(
            "requestHeadersAvailable".to_owned(),
            Value::Bool(self.request_headers.is_some()),
        );

        resource_updates.insert(
            "responseHeadersAvailable".to_owned(),
            Value::Bool(self.response_headers.is_some()),
        );
        resource_updates.insert(
            "responseCookiesAvailable".to_owned(),
            Value::Bool(self.response_cookies.is_some()),
        );
        resource_updates.insert(
            "responseStartAvailable".to_owned(),
            Value::Bool(self.response_start.is_some()),
        );
        resource_updates.insert(
            "responseContentAvailable".to_owned(),
            Value::Bool(self.response_content.is_some()),
        );

        // Block Logic: Insert total time and security state.
        resource_updates.insert(
            "totalTime".to_string(),
            Value::from(self.total_time.as_secs_f64()),
        );

        resource_updates.insert(
            "securityState".to_string(),
            Value::String(self.security_state.clone()),
        );
        resource_updates.insert(
            "eventTimingsAvailable".to_owned(),
            Value::Bool(self.event_timing.is_some()),
        );

        // Block Logic: Insert serialized optional messages into the resource updates map.
        Self::insert_serialized_map(&mut resource_updates, &self.response_content);
        Self::insert_serialized_map(&mut resource_updates, &self.response_headers);
        Self::insert_serialized_map(&mut resource_updates, &self.response_cookies);
        Self::insert_serialized_map(&mut resource_updates, &self.request_headers);
        Self::insert_serialized_map(&mut resource_updates, &self.request_cookies);
        Self::insert_serialized_map(&mut resource_updates, &self.response_start);
        Self::insert_serialized_map(&mut resource_updates, &self.event_timing);

        // TODO: Set the correct values for these fields
        NetworkEventResource {
            resource_id: 0,
            resource_updates,
            browsing_context_id: 0,
            inner_window_id: 0,
        }
    }
}
