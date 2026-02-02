/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/// @file htmlimageelement.rs
/// @brief This file implements the `HTMLImageElement` interface, which represents the
/// `<img>` HTML element. It handles image loading, display, and interaction, including
/// responsive images (`srcset`, `sizes`), image decoding, and form association.
/// Functional Utility: Manages the lifecycle of images within a web page, from fetching
/// to rendering, and integrates with the DOM and layout engines.

use std::cell::Cell;
use std::collections::HashSet;
use std::default::Default;
use std::rc::Rc;
use std::sync::Arc;
use std::{char, mem};

use app_units::{AU_PER_PX, Au};
use cssparser::{Parser, ParserInput};
use dom_struct::dom_struct;
use euclid::Point2D;
use html5ever::{LocalName, Prefix, QualName, local_name, namespace_url, ns};
use js::jsapi::JSAutoRealm;
use js::rust::HandleObject;
use mime::{self, Mime};
use net_traits::http_status::HttpStatus;
use net_traits::image_cache::{
    ImageCache, ImageCacheResult, ImageOrMetadataAvailable, ImageResponder, ImageResponse,
    PendingImageId, UsePlaceholder,
};
use net_traits::request::{Destination, Initiator, RequestId};
use net_traits::{
    FetchMetadata, FetchResponseListener, FetchResponseMsg, NetworkError, ReferrerPolicy,
    ResourceFetchTiming, ResourceTimingType,
};
use num_traits::ToPrimitive;
use pixels::{CorsStatus, Image, ImageMetadata};
use servo_url::ServoUrl;
use servo_url::origin::MutableOrigin;
use style::attr::{AttrValue, LengthOrPercentageOrAuto, parse_integer, parse_length};
use style::context::QuirksMode;
use style::media_queries::MediaList;
use style::parser::ParserContext;
use style::stylesheets::{CssRuleType, Origin, UrlExtraData};
use style::values::specified::AbsoluteLength;
use style::values::specified::length::{Length, NoCalcLength};
use style::values::specified::source_size_list::SourceSizeList;
use style_traits::ParsingMode;
use url::Url;

use super::domexception::DOMErrorName;
use super::types::DOMException;
use crate::document_loader::{LoadBlocker, LoadType};
use crate::dom::activation::Activatable;
use crate::dom::attr::Attr;
use crate::dom::bindings::cell::{DomRefCell, RefMut};
use crate::dom::bindings::codegen::Bindings::DOMRectBinding::DOMRect_Binding::DOMRectMethods;
use crate::dom::bindings::codegen::Bindings::ElementBinding::Element_Binding::ElementMethods;
use crate::dom::bindings::codegen::Bindings::HTMLImageElementBinding::HTMLImageElementMethods;
use crate::dom::bindings::codegen::Bindings::MouseEventBinding::MouseEventMethods;
use crate::dom::bindings::codegen::Bindings::NodeBinding::Node_Binding::NodeMethods;
use crate::dom::bindings::codegen::Bindings::WindowBinding::WindowMethods;
use crate::dom::bindings::error::{Error, Fallible};
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::refcounted::Trusted;
use crate::dom::bindings::reflector::DomGlobal;
use crate::dom::bindings::root::{DomRoot, LayoutDom, MutNullableDom};
use crate::dom::bindings::str::{DOMString, USVString};
use crate::dom::document::{Document, determine_policy_for_token};
use crate::dom::domtokenlist::DOMTokenList;
use crate::dom::element::{
    AttributeMutation, CustomElementCreationMode, Element, ElementCreator, LayoutElementHelpers,
    cors_setting_for_element, referrer_policy_for_element, reflect_cross_origin_attribute,
    reflect_referrer_policy_attribute, set_cross_origin_attribute,
};
use crate::dom::event::Event;
use crate::dom::eventtarget::EventTarget;
use crate::dom::globalscope::GlobalScope;
use crate::dom::htmlareaelement::HTMLAreaElement;
use crate::dom::htmlelement::HTMLElement;
use crate::dom::htmlformelement::{FormControl, HTMLFormElement};
use crate::dom::htmlmapelement::HTMLMapElement;
use crate::dom::htmlpictureelement::HTMLPictureElement;
use crate::dom::htmlsourceelement::HTMLSourceElement;
use crate::dom::mouseevent::MouseEvent;
use crate::dom::node::{BindContext, Node, NodeDamage, NodeTraits, ShadowIncluding, UnbindContext};
use crate::dom::performanceresourcetiming::InitiatorType;
use crate::dom::promise::Promise;
use crate::dom::values::UNSIGNED_LONG_MAX;
use crate::dom::virtualmethods::VirtualMethods;
use crate::dom::window::Window;
use crate::fetch::create_a_potential_cors_request;
use crate::microtask::{Microtask, MicrotaskRunnable};
use crate::network_listener::{self, PreInvoke, ResourceTimingListener};
use crate::realms::enter_realm;
use crate::script_runtime::CanGc;
use crate::script_thread::ScriptThread;

/// @enum ParseState
/// @brief Represents the current parsing state when processing an image descriptor string.
/// Functional Utility: Aids in correctly parsing `srcset` attributes by tracking whether
/// the parser is currently inside a descriptor, parentheses, or after a descriptor.
#[derive(Clone, Copy, Debug)]
enum ParseState {
    InDescriptor,  //!< Currently parsing a descriptor value (e.g., '100w', '1.5x').
    InParens,      //!< Currently inside parentheses within a descriptor (e.g., for media queries).
    AfterDescriptor, //!< After a descriptor has been parsed, looking for the next one or a comma.
}

/// @struct SourceSet
/// @brief Represents a set of image sources and a source size list.
/// Functional Utility: Encapsulates the parsed information from `srcset` and `sizes`
/// attributes for responsive image selection.
pub(crate) struct SourceSet {
    image_sources: Vec<ImageSource>, //!< List of available image sources with their descriptors.
    source_size: SourceSizeList,     //!< The parsed `sizes` attribute value.
}

impl SourceSet {
    /// @brief Creates a new, empty `SourceSet`.
    fn new() -> SourceSet {
        SourceSet {
            image_sources: Vec::new(),
            source_size: SourceSizeList::empty(),
        }
    }
}

/// @struct ImageSource
/// @brief Represents a single image source with its URL and associated descriptor.
/// Functional Utility: Stores the URL and density/width descriptor for a specific
/// image candidate in a `srcset`.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageSource {
    pub url: String,          //!< The URL of the image.
    pub descriptor: Descriptor, //!< The descriptor (width or density) for this image source.
}

/// @struct Descriptor
/// @brief Represents the width or density descriptor for an image source.
/// Functional Utility: Specifies the characteristics of an image source to
/// aid the browser in selecting the most appropriate image from a `srcset`.
#[derive(Clone, Debug, PartialEq)]
pub struct Descriptor {
    pub width: Option<u32>,   //!< The width descriptor (e.g., '100w').
    pub density: Option<f64>, //!< The pixel density descriptor (e.g., '1.5x').
}

/// @enum State
/// @brief Represents the loading state of an image request.
/// Functional Utility: Tracks the progress and outcome of an image fetch,
/// influencing how the image is rendered and reported.
#[derive(Clone, Copy, JSTraceable, MallocSizeOf)]
#[allow(dead_code)]
enum State {
    Unavailable,        //!< Image data is not yet available.
    PartiallyAvailable, //!< Image metadata is available, but not full image data.
    CompletelyAvailable, //!< Full image data is available.
    Broken,             //!< Image failed to load or is corrupted.
}

/// @enum ImageRequestPhase
/// @brief Represents the phase of an image request.
/// Functional Utility: Differentiates between a currently active image request
/// and a pending request, which is important for managing responsive image updates.
#[derive(Clone, Copy, JSTraceable, MallocSizeOf)]
enum ImageRequestPhase {
    Pending, //!< An image request is pending, possibly waiting for environment changes.
    Current, //!< An image request is currently active and being processed.
}
/// @struct ImageRequest
/// @brief Represents a single image loading request.
/// Functional Utility: Holds all the state associated with an image request,
/// including its loading status, URL, actual image data, and metadata.
#[derive(JSTraceable, MallocSizeOf)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
struct ImageRequest {
    state: State, //!< The current loading state of the image.
    #[no_trace]
    parsed_url: Option<ServoUrl>, //!< The parsed URL of the image.
    source_url: Option<USVString>, //!< The original source URL string.
    blocker: DomRefCell<Option<LoadBlocker>>, //!< Load blocker associated with this request.
    #[ignore_malloc_size_of = "Arc"]
    #[no_trace]
    image: Option<Arc<Image>>, //!< The loaded image data.
    #[no_trace]
    metadata: Option<ImageMetadata>, //!< Metadata about the image (width, height).
    #[no_trace]
    final_url: Option<ServoUrl>, //!< The final URL after redirects.
    current_pixel_density: Option<f64>, //!< The pixel density for the currently selected image.
}
/// @struct HTMLImageElement
/// @brief Represents the `<img>` HTML element.
/// Functional Utility: Provides the DOM interface for `<img>`, managing image fetching,
/// decoding, display, and interaction. It includes logic for responsive images
/// (`srcset`, `sizes`), image decoding promises, and form association.
///
/// <https://html.spec.whatwg.org/multipage/#htmlimageelement>
#[dom_struct]
pub(crate) struct HTMLImageElement {
    htmlelement: HTMLElement, //!< Inherited properties and methods from `HTMLElement`.
    image_request: Cell<ImageRequestPhase>, //!< The current phase of image request (current or pending).
    current_request: DomRefCell<ImageRequest>, //!< The currently active image request.
    pending_request: DomRefCell<ImageRequest>, //!< A pending image request, for responsive image changes.
    form_owner: MutNullableDom<HTMLFormElement>, //!< The `HTMLFormElement` this image belongs to if it's a submit button.
    generation: Cell<u32>, //!< Counter for image requests, used to discard outdated requests.
    #[ignore_malloc_size_of = "SourceSet"]
    source_set: DomRefCell<SourceSet>, //!< The parsed `srcset` and `sizes` attributes.
    last_selected_source: DomRefCell<Option<USVString>>, //!< The URL of the last selected image source.
    #[ignore_malloc_size_of = "promises are hard"]
    image_decode_promises: DomRefCell<Vec<Rc<Promise>>>, //!< List of pending image decode promises.
}

impl HTMLImageElement {
    /// @brief Returns the parsed URL of the currently loaded image.
    /// Functional Utility: Provides access to the URL that was successfully used
    /// to load the image.
    /// @return An `Option<ServoUrl>`.
    pub(crate) fn get_url(&self) -> Option<ServoUrl> {
        self.current_request.borrow().parsed_url.clone()
    }
    // https://html.spec.whatwg.org/multipage/#check-the-usability-of-the-image-argument
    /// @brief Checks if the image is usable (i.e., not broken and has valid dimensions).
    /// Functional Utility: Determines if the currently loaded image can be used for rendering
    /// or other operations, as per HTML specification rules.
    /// @return A `Fallible<bool>` indicating usability or an error if broken.
    pub(crate) fn is_usable(&self) -> Fallible<bool> {
        // If image has an intrinsic width or intrinsic height (or both) equal to zero, then return bad.
        if let Some(image) = &self.current_request.borrow().image {
            if image.width == 0 || image.height == 0 {
                return Ok(false);
            }
        }

        match self.current_request.borrow().state {
            // If image's current request's state is broken, then throw an "InvalidStateError" DOMException.
            State::Broken => Err(Error::InvalidState),
            State::CompletelyAvailable => Ok(true),
            // If image is not fully decodable, then return bad.
            State::PartiallyAvailable | State::Unavailable => Ok(false),
        }
    }

    /// @brief Returns the `Arc<Image>` if the image data is available.
    /// Functional Utility: Provides direct access to the raw image data once it's loaded.
    /// @return An `Option<Arc<Image>>`.
    pub(crate) fn image_data(&self) -> Option<Arc<Image>> {
        self.current_request.borrow().image.clone()
    }
}

/// @struct ImageContext
/// @brief The context required for asynchronously loading an external image.
/// Functional Utility: Manages the state and callbacks for an individual image fetch
/// operation, integrating with the image cache and network listeners.
struct ImageContext {
    /// Reference to the script thread image cache.
    image_cache: Arc<dyn ImageCache>,
    /// Indicates whether the request failed, and why
    status: Result<(), NetworkError>,
    /// The cache ID for this request.
    id: PendingImageId,
    /// Used to mark abort
    aborted: bool,
    /// The document associated with this request
    doc: Trusted<Document>,
    /// timing data for this resource
    resource_timing: ResourceFetchTiming,
    url: ServoUrl,
}

impl FetchResponseListener for ImageContext {
    /// @brief Processes the request body chunk.
    /// @param _: RequestId The ID of the request (unused).
    fn process_request_body(&mut self, _: RequestId) {}
    /// @brief Processes the end of the request.
    /// @param _: RequestId The ID of the request (unused).
    fn process_request_eof(&mut self, _: RequestId) {}

    /// @brief Processes the initial response metadata from the network.
    /// Functional Utility: Handles the HTTP response, updates the image cache,
    /// and determines the network status (success or error).
    ///
    /// @param request_id The ID of the network request.
    /// @param metadata The `FetchMetadata` or `NetworkError`.
    fn process_response(
        &mut self,
        request_id: RequestId,
        metadata: Result<FetchMetadata, NetworkError>,
    ) {
        debug!("got {:?} for {:?}", metadata.as_ref().map(|_| ()), self.url);
        self.image_cache.notify_pending_response(
            self.id,
            FetchResponseMsg::ProcessResponse(request_id, metadata.clone()),
        );

        let metadata = metadata.ok().map(|meta| match meta {
            FetchMetadata::Unfiltered(m) => m,
            FetchMetadata::Filtered { unsafe_, .. } => unsafe_,
        });

        // Step 14.5 of https://html.spec.whatwg.org/multipage/#img-environment-changes
        // Block Logic: Handle `multipart/x-mixed-replace` to abort the request.
        if let Some(metadata) = metadata.as_ref() {
            if let Some(ref content_type) = metadata.content_type {
                let mime: Mime = content_type.clone().into_inner().into();
                if mime.type_() == mime::MULTIPART && mime.subtype().as_str() == "x-mixed-replace" {
                    self.aborted = true; // Mark as aborted.
                }
            }
        }

        // Block Logic: Determine the network status based on HTTP status code.
        let status = metadata
            .as_ref()
            .map(|m| m.status.clone())
            .unwrap_or_else(HttpStatus::new_error);

        self.status = {
            if status.is_error() {
                Err(NetworkError::Internal(
                    "No http status code received".to_owned(),
                ))
            } else if status.is_success() {
                Ok(())
            } else {
                Err(NetworkError::Internal(format!(
                    "HTTP error code {}",
                    status.code()
                )))
            }
        };
    }

    /// @brief Processes a chunk of the response payload.
    /// Functional Utility: Forwards the image data chunk to the image cache.
    /// @param request_id The ID of the network request.
    /// @param payload The raw bytes of the response chunk.
    fn process_response_chunk(&mut self, request_id: RequestId, payload: Vec<u8>) {
        if self.status.is_ok() {
            self.image_cache.notify_pending_response(
                self.id,
                FetchResponseMsg::ProcessResponseChunk(request_id, payload),
            );
        }
    }

    /// @brief Processes the end of the response stream.
    /// Functional Utility: Notifies the image cache that the image response is complete.
    /// @param request_id The ID of the network request.
    /// @param response The `ResourceFetchTiming` or `NetworkError`.
    fn process_response_eof(
        &mut self,
        request_id: RequestId,
        response: Result<ResourceFetchTiming, NetworkError>,
    ) {
        self.image_cache.notify_pending_response(
            self.id,
            FetchResponseMsg::ProcessResponseEOF(request_id, response),
        );
    }

    /// @brief Returns a mutable reference to the `ResourceFetchTiming`.
    fn resource_timing_mut(&mut self) -> &mut ResourceFetchTiming {
        &mut self.resource_timing
    }

    /// @brief Returns an immutable reference to the `ResourceFetchTiming`.
    fn resource_timing(&self) -> &ResourceFetchTiming {
        &self.resource_timing
    }

    /// @brief Submits resource timing information.
    /// Functional Utility: Sends collected resource timing data to the network listener
    /// for performance monitoring.
    fn submit_resource_timing(&mut self) {
        network_listener::submit_timing(self, CanGc::note())
    }
}

impl ResourceTimingListener for ImageContext {
    /// @brief Returns resource timing information for this image.
    /// Functional Utility: Provides the initiator type and URL for performance
    /// resource timing entries.
    /// @return A tuple of `(InitiatorType, ServoUrl)`.
    fn resource_timing_information(&self) -> (InitiatorType, ServoUrl) {
        (
            InitiatorType::LocalName("img".to_string()), // Initiator is an image element.
            self.url.clone(), // URL of the resource.
        )
    }

    /// @brief Returns the global scope for resource timing.
    fn resource_timing_global(&self) -> DomRoot<GlobalScope> {
        self.doc.root().global()
    }
}

impl PreInvoke for ImageContext {
    /// @brief Determines if the `ImageContext` should be invoked.
    /// Functional Utility: Prevents invocation if the image request has been aborted.
    /// @return `true` if not aborted, `false` otherwise.
    fn should_invoke(&self) -> bool {
        !self.aborted
    }
}

#[allow(non_snake_case)]
impl HTMLImageElement {
    /// @brief Fetches an image from a given URL.
    /// Functional Utility: Initiates the image loading process, checking the image cache
    /// first and then making a network request if needed.
    ///
    /// @param img_url The `ServoUrl` of the image to fetch.
    /// @param can_gc A `CanGc` token.
    fn fetch_image(&self, img_url: &ServoUrl, can_gc: CanGc) {
        let window = self.owner_window(); // Get the owning window.

        // Block Logic: Check the image cache for the requested image.
        let cache_result = window.image_cache().get_cached_image_status(
            img_url.clone(), // Image URL.
            window.origin().immutable().clone(), // Window origin.
            cors_setting_for_element(self.upcast()), // CORS setting.
            UsePlaceholder::Yes, // Allow use of placeholder.
        );

        match cache_result {
            ImageCacheResult::Available(ImageOrMetadataAvailable::ImageAvailable {
                image,
                url,
                is_placeholder,
            }) => {
                // Block Logic: Image is fully available (or a placeholder).
                if is_placeholder {
                    self.process_image_response(
                        ImageResponse::PlaceholderLoaded(image, url),
                        can_gc,
                    )
                } else {
                    self.process_image_response(ImageResponse::Loaded(image, url), can_gc)
                }
            },
            ImageCacheResult::Available(ImageOrMetadataAvailable::MetadataAvailable(
                metadata,
                id,
            )) => {
                // Block Logic: Only metadata is available.
                self.process_image_response(ImageResponse::MetadataLoaded(metadata), can_gc);
                self.register_image_cache_callback(id, ChangeType::Element); // Register callback for full image.
            },
            ImageCacheResult::Pending(id) => {
                // Block Logic: Image request is already pending.
                self.register_image_cache_callback(id, ChangeType::Element); // Register callback.
            },
            ImageCacheResult::ReadyForRequest(id) => {
                // Block Logic: Cache is ready to make a network request.
                self.fetch_request(img_url, id); // Initiate network fetch.
                self.register_image_cache_callback(id, ChangeType::Element); // Register callback.
            },
            ImageCacheResult::LoadError => self.process_image_response(ImageResponse::None, can_gc), // Handle load error.
        };
    }

    /// @brief Registers a callback with the image cache to handle image load responses.
    /// Functional Utility: Sets up asynchronous notification for when image data
    /// becomes available or an error occurs, processing the response in the correct context.
    ///
    /// @param id The `PendingImageId` for the pending image request.
    /// @param change_type The `ChangeType` indicating how the image response should be processed.
    fn register_image_cache_callback(&self, id: PendingImageId, change_type: ChangeType) {
        let trusted_node = Trusted::new(self); // Create trusted reference to self.
        let generation = self.generation_id(); // Get current generation ID.
        let window = self.owner_window(); // Get owning window.
        let sender = window.register_image_cache_listener(id, move |response| {
            let trusted_node = trusted_node.clone(); // Clone trusted node.
            let window = trusted_node.root().owner_window(); // Get owning window from root.
            let callback_type = change_type.clone(); // Clone change type.

            window
                .as_global_scope()
                .task_manager()
                .networking_task_source()
                .queue(task!(process_image_response: move || {
                let element = trusted_node.root();

                // Ignore any image response for a previous request that has been discarded.
                // Block Logic: Only process if the generation ID matches the current one, to prevent outdated responses.
                if generation != element.generation_id() {
                    return;
                }

                match callback_type {
                    ChangeType::Element => {
                        element.process_image_response(response.response, CanGc::note());
                    }
                    ChangeType::Environment { selected_source, selected_pixel_density } => {
                        element.process_image_response_for_environment_change(
                            response.response, selected_source, generation, selected_pixel_density, CanGc::note()
                        );
                    }
                }
            }));
        });

        window
            .image_cache()
            .add_listener(ImageResponder::new(sender, window.pipeline_id(), id)); // Add listener to image cache.
    }

    /// @brief Initiates a network request to fetch an image.
    /// Functional Utility: Constructs a network request with appropriate CORS and referrer
    /// policies, and sends it to the fetch background task.
    ///
    /// @param img_url The `ServoUrl` of the image to fetch.
    /// @param id The `PendingImageId` for this request.
    fn fetch_request(&self, img_url: &ServoUrl, id: PendingImageId) {
        let document = self.owner_document(); // Get owning document.
        let window = self.owner_window(); // Get owning window.

        let context = ImageContext {
            image_cache: window.image_cache(), // Image cache.
            status: Ok(()), // Initial status.
            id, // Pending image ID.
            aborted: false, // Not aborted initially.
            doc: Trusted::new(&document), // Trusted reference to document.
            resource_timing: ResourceFetchTiming::new(ResourceTimingType::Resource), // Resource timing.
            url: img_url.clone(), // Image URL.
        };

        // https://html.spec.whatwg.org/multipage/#update-the-image-data steps 17-20
        // This function is also used to prefetch an image in `script::dom::servoparser::prefetch`.
        let mut request = create_a_potential_cors_request(
            Some(window.webview_id()), // WebView ID.
            img_url.clone(), // Image URL.
            Destination::Image, // Destination type.
            cors_setting_for_element(self.upcast()), // CORS setting.
            None, // No request body.
            document.global().get_referrer(), // Referrer.
            document.insecure_requests_policy(), // Insecure requests policy.
            document.has_trustworthy_ancestor_or_current_origin(), // Trustworthy ancestor origin.
        )
        .origin(document.origin().immutable().clone()) // Origin.
        .pipeline_id(Some(document.global().pipeline_id())) // Pipeline ID.
        .referrer_policy(referrer_policy_for_element(self.upcast())); // Referrer policy.

        // Block Logic: Set initiator for `srcset` or `picture` elements.
        if Self::uses_srcset_or_picture(self.upcast()) {
            request = request.initiator(Initiator::ImageSet);
        }

        // This is a background load because the load blocker already fulfills the
        // purpose of delaying the document's load event.
        document.fetch_background(request, context); // Fetch in background.
    }

    // Steps common to when an image has been loaded.
    /// @brief Handles a successfully loaded image.
    /// Functional Utility: Updates the image element's internal state with the loaded
    /// image data and metadata, and marks the element for re-rendering.
    ///
    /// @param image The `Arc<Image>` containing the loaded image data.
    /// @param url The `ServoUrl` of the loaded image.
    /// @param can_gc A `CanGc` token.
    fn handle_loaded_image(&self, image: Arc<Image>, url: ServoUrl, can_gc: CanGc) {
        self.current_request.borrow_mut().metadata = Some(ImageMetadata {
            height: image.height,
            width: image.width,
        }); // Set image metadata.
        self.current_request.borrow_mut().final_url = Some(url); // Set final URL.
        self.current_request.borrow_mut().image = Some(image); // Set image data.
        self.current_request.borrow_mut().state = State::CompletelyAvailable; // Set state to completely available.
        LoadBlocker::terminate(&self.current_request.borrow().blocker, can_gc); // Terminate load blocker.
        // Mark the node dirty
        self.upcast::<Node>().dirty(NodeDamage::OtherNodeDamage); // Mark node dirty for re-render.
        self.resolve_image_decode_promises(can_gc); // Resolve pending decode promises.
    }

    /// Step 24 of <https://html.spec.whatwg.org/multipage/#update-the-image-data>
    /// @brief Processes an image response from the image cache.
    /// Functional Utility: Updates the image element's state based on the image response
    /// (loaded, placeholder, metadata, or error) and fires appropriate events.
    ///
    /// @param image The `ImageResponse` received.
    /// @param can_gc A `CanGc` token.
    fn process_image_response(&self, image: ImageResponse, can_gc: CanGc) {
        // TODO: Handle multipart/x-mixed-replace
        let (trigger_image_load, trigger_image_error) = match (image, self.image_request.get()) {
            (ImageResponse::Loaded(image, url), ImageRequestPhase::Current) => {
                self.handle_loaded_image(image, url, can_gc);
                (true, false) // Trigger load event, no error.
            },
            (ImageResponse::PlaceholderLoaded(image, url), ImageRequestPhase::Current) => {
                self.handle_loaded_image(image, url, can_gc);
                (false, true) // Trigger error event (for placeholder), no load.
            },
            (ImageResponse::Loaded(image, url), ImageRequestPhase::Pending) => {
                // Block Logic: Abort current pending request and switch to current.
                self.abort_request(State::Unavailable, ImageRequestPhase::Pending, can_gc);
                self.image_request.set(ImageRequestPhase::Current);
                self.handle_loaded_image(image, url, can_gc);
                (true, false) // Trigger load event, no error.
            },
            (ImageResponse::PlaceholderLoaded(image, url), ImageRequestPhase::Pending) => {
                // Block Logic: Abort current pending request and switch to current.
                self.abort_request(State::Unavailable, ImageRequestPhase::Pending, can_gc);
                self.image_request.set(ImageRequestPhase::Current);
                self.handle_loaded_image(image, url, can_gc);
                (false, true) // Trigger error event, no load.
            },
            (ImageResponse::MetadataLoaded(meta), ImageRequestPhase::Current) => {
                // Block Logic: Only metadata loaded for current request.
                self.current_request.borrow_mut().state = State::PartiallyAvailable;
                self.current_request.borrow_mut().metadata = Some(meta);
                (false, false) // No events.
            },
            (ImageResponse::MetadataLoaded(_), ImageRequestPhase::Pending) => {
                // Block Logic: Only metadata loaded for pending request.
                self.pending_request.borrow_mut().state = State::PartiallyAvailable;
                (false, false) // No events.
            },
            (ImageResponse::None, ImageRequestPhase::Current) => {
                // Block Logic: No image data for current request.
                self.abort_request(State::Broken, ImageRequestPhase::Current, can_gc);
                (false, true) // Trigger error event.
            },
            (ImageResponse::None, ImageRequestPhase::Pending) => {
                // Block Logic: No image data for pending request, abort both.
                self.abort_request(State::Broken, ImageRequestPhase::Current, can_gc);
                self.abort_request(State::Broken, ImageRequestPhase::Pending, can_gc);
                self.image_request.set(ImageRequestPhase::Current);
                (false, true) // Trigger error event.
            },
        };

        // Fire image.onload and loadend
        if trigger_image_load {
            // TODO: https://html.spec.whatwg.org/multipage/#fire-a-progress-event-or-event
            self.upcast::<EventTarget>()
                .fire_event(atom!("load"), can_gc); // Fire `load` event.
            self.upcast::<EventTarget>()
                .fire_event(atom!("loadend"), can_gc); // Fire `loadend` event.
        }

        // Fire image.onerror
        if trigger_image_error {
            self.upcast::<EventTarget>()
                .fire_event(atom!("error"), can_gc); // Fire `error` event.
            self.upcast::<EventTarget>()
                .fire_event(atom!("loadend"), can_gc); // Fire `loadend` event.
        }

        self.upcast::<Node>().dirty(NodeDamage::OtherNodeDamage); // Mark node dirty for re-render.
    }

    /// @brief Processes an image response specifically for environment changes (e.g., responsive images).
    /// Functional Utility: Updates the pending image request with image data based on
    /// environment changes and prepares to finish the reaction.
    ///
    /// @param image The `ImageResponse` received.
    /// @param src The `USVString` representing the selected source URL.
    /// @param generation The generation ID of the image request.
    /// @param selected_pixel_density The pixel density of the selected image.
    /// @param can_gc A `CanGc` token.
    fn process_image_response_for_environment_change(
        &self,
        image: ImageResponse,
        src: USVString,
        generation: u32,
        selected_pixel_density: f64,
        can_gc: CanGc,
    ) {
        match image {
            ImageResponse::Loaded(image, url) | ImageResponse::PlaceholderLoaded(image, url) => {
                self.pending_request.borrow_mut().metadata = Some(ImageMetadata {
                    height: image.height,
                    width: image.width,
                }); // Set metadata for pending request.
                self.pending_request.borrow_mut().final_url = Some(url); // Set final URL.
                self.pending_request.borrow_mut().image = Some(image); // Set image data.
                self.finish_reacting_to_environment_change(src, generation, selected_pixel_density); // Finish processing.
            },
            ImageResponse::MetadataLoaded(meta) => {
                self.pending_request.borrow_mut().metadata = Some(meta); // Set metadata for pending request.
            },
            ImageResponse::None => {
                self.abort_request(State::Unavailable, ImageRequestPhase::Pending, can_gc); // Abort pending request.
            },
        };
    }

    /// <https://html.spec.whatwg.org/multipage/#abort-the-image-request>
    /// @brief Aborts an in-progress image request.
    /// Functional Utility: Resets the state of an image request, clears associated
    /// image data, terminates load blockers, and rejects or resolves image decode promises.
    ///
    /// @param state The new `State` to set for the request.
    /// @param request The `ImageRequestPhase` to abort (Current or Pending).
    /// @param can_gc A `CanGc` token.
    fn abort_request(&self, state: State, request: ImageRequestPhase, can_gc: CanGc) {
        let mut request = match request {
            ImageRequestPhase::Current => self.current_request.borrow_mut(), // Get mutable current request.
            ImageRequestPhase::Pending => self.pending_request.borrow_mut(), // Get mutable pending request.
        };
        LoadBlocker::terminate(&request.blocker, can_gc); // Terminate load blocker.
        request.state = state; // Set new state.
        request.image = None; // Clear image data.
        request.metadata = None; // Clear metadata.

        // Block Logic: Reject promises if image is broken, resolve if completely available.
        if matches!(state, State::Broken) {
            self.reject_image_decode_promises(can_gc);
        } else if matches!(state, State::CompletelyAvailable) {
            self.resolve_image_decode_promises(can_gc);
        }
    }

    /// <https://html.spec.whatwg.org/multipage/#update-the-source-set>
    /// @brief Updates the image element's source set from `srcset` and `picture` elements.
    /// Functional Utility: Parses `srcset` and `sizes` attributes, and considers
    /// `<picture>` and `<source>` elements to determine the available image candidates.
    fn update_source_set(&self) {
        // Step 1: Let source set be a new empty source set.
        *self.source_set.borrow_mut() = SourceSet::new();

        // Step 2: Let candidate elements be a list containing only the image element.
        // If the image element is a child of a picture element, then instead let candidate elements
        // be the list of child element nodes of the picture element, in tree order.
        let elem = self.upcast::<Element>();
        let parent = elem.upcast::<Node>().GetParentElement(); // Get parent element.
        let nodes;
        let elements = match parent.as_ref() {
            Some(p) => {
                if p.is::<HTMLPictureElement>() {
                    nodes = p.upcast::<Node>().children(); // Get children of picture element.
                    nodes
                        .filter_map(DomRoot::downcast::<Element>) // Filter for Element types.
                        .map(|n| DomRoot::from_ref(&*n)) // Map to DomRoot<Element>.
                        .collect() // Collect into a vector.
                } else {
                    vec![DomRoot::from_ref(elem)] // Only the image element.
                }
            },
            None => vec![DomRoot::from_ref(elem)], // Only the image element if no parent.
        };

        // Step 3: Let width be the image element's width attribute's value, parsed as a
        // CSS <length> or <percentage> or 'auto', or 'auto' if the attribute is absent.
        let width = match elem.get_attribute(&ns!(), &local_name!("width")) {
            Some(x) => match parse_length(&x.value()) {
                LengthOrPercentageOrAuto::Length(x) => {
                    let abs_length = AbsoluteLength::Px(x.to_f32_px()); // Convert to absolute pixels.
                    Some(Length::NoCalc(NoCalcLength::Absolute(abs_length))) // Store as Length.
                },
                _ => None, // Invalid length.
            },
            None => None, // Attribute absent.
        };

        // Step 4: For each element in candidate elements:
        for element in &elements {
            // Step 4.1: If element is the image element itself, then:
            if *element == DomRoot::from_ref(elem) {
                let mut source_set = SourceSet::new(); // New source set.
                // Step 4.1.1: If element has a srcset attribute, parse it.
                if let Some(x) = element.get_attribute(&ns!(), &local_name!("srcset")) {
                    source_set.image_sources = parse_a_srcset_attribute(&x.value()); // Parse srcset.
                }

                // Step 4.1.2: If element has a sizes attribute, parse it.
                if let Some(x) = element.get_attribute(&ns!(), &local_name!("sizes")) {
                    source_set.source_size =
                        parse_a_sizes_attribute(DOMString::from_string(x.value().to_string())); // Parse sizes.
                }

                // Step 4.1.3: If element does not have a srcset attribute, and element does not
                // have an image source with a pixel density descriptor of 1, and element does not
                // have an image source with a width descriptor, and element has a src attribute, then:
                let src_attribute = element.get_string_attribute(&local_name!("src"));
                let is_src_empty = src_attribute.is_empty();
                let no_density_source_of_1 = source_set
                    .image_sources
                    .iter()
                    .all(|source| source.descriptor.density != Some(1.)); // Check for 1x density.
                let no_width_descriptor = source_set
                    .image_sources
                    .iter()
                    .all(|source| source.descriptor.width.is_none()); // Check for width descriptor.
                if !is_src_empty && no_density_source_of_1 && no_width_descriptor {
                    source_set.image_sources.push(ImageSource {
                        url: src_attribute.to_string(), // Add src as a candidate.
                        descriptor: Descriptor {
                            width: None,
                            density: None,
                        },
                    })
                }

                // Step 4.1.4: Normalise the source densities for source set.
                self.normalise_source_densities(&mut source_set, width);

                // Step 4.1.5: Set the image element's source set to source set.
                *self.source_set.borrow_mut() = source_set;

                // Step 4.1.6: Return.
                return;
            }
            // Step 4.2: If element is not a HTMLSourceElement, then continue.
            if !element.is::<HTMLSourceElement>() {
                continue;
            }

            // Step 4.3 - 4.4: If element has a srcset attribute, parse it.
            let mut source_set = SourceSet::new();
            match element.get_attribute(&ns!(), &local_name!("srcset")) {
                Some(x) => {
                    source_set.image_sources = parse_a_srcset_attribute(&x.value());
                },
                _ => continue, // If no srcset, continue.
            }

            // Step 4.5: If source set's image sources is empty, then continue.
            if source_set.image_sources.is_empty() {
                continue;
            }

            // Step 4.6: If element has a media attribute and its value does not match the
            // environment, then continue.
            if let Some(x) = element.get_attribute(&ns!(), &local_name!("media")) {
                if !self.matches_environment(x.value().to_string()) {
                    continue; // Media query doesn't match.
                }
            }

            // Step 4.7: If element has a sizes attribute, parse it.
            if let Some(x) = element.get_attribute(&ns!(), &local_name!("sizes")) {
                source_set.source_size =
                    parse_a_sizes_attribute(DOMString::from_string(x.value().to_string()));
            }

            // Step 4.8: If element has a type attribute and the user agent does not
            // support the given MIME type, then continue.
            if let Some(x) = element.get_attribute(&ns!(), &local_name!("type")) {
                // TODO Handle unsupported mime type
                let mime = x.value().parse::<Mime>();
                match mime {
                    Ok(m) => match m.type_() {
                        mime::IMAGE => (), // Supported image type.
                        _ => continue, // Not an image type.
                    },
                    _ => continue, // Invalid MIME type.
                }
            }

            // Step 4.9: Normalise the source densities for source set.
            self.normalise_source_densities(&mut source_set, width);

            // Step 4.10: Set the image element's source set to source set.
            *self.source_set.borrow_mut() = source_set;
            return; // Found a matching source element, so return.
        }
    }

    /// @brief Evaluates a `SourceSizeList` to a pixel length.
    /// Functional Utility: Interprets the `sizes` attribute to determine the
    /// effective width of the image for responsive image selection.
    ///
    /// @param source_size_list The `SourceSizeList` to evaluate.
    /// @param _width An `Option<Length>` for the width (unused here).
    /// @return An `Au` representing the evaluated pixel length.
    fn evaluate_source_size_list(
        &self,
        source_size_list: &mut SourceSizeList,
        _width: Option<Length>,
    ) -> Au {
        let document = self.owner_document(); // Get owning document.
        let quirks_mode = document.quirks_mode(); // Get quirks mode.
        let result = source_size_list.evaluate(document.window().layout().device(), quirks_mode); // Evaluate.
        result
    }

    /// <https://html.spec.whatwg.org/multipage/#matches-the-environment>
    /// @brief Checks if a media query string matches the current environment.
    /// Functional Utility: Used for `<source>` elements with `media` attributes to
    /// determine if the image source is appropriate for the current viewport and device characteristics.
    ///
    /// @param media_query The media query string to evaluate.
    /// @return `true` if the media query matches, `false` otherwise.
    fn matches_environment(&self, media_query: String) -> bool {
        let document = self.owner_document(); // Get owning document.
        let quirks_mode = document.quirks_mode(); // Get quirks mode.
        let document_url_data = UrlExtraData(document.url().get_arc()); // URL data for parser context.
        // FIXME(emilio): This should do the same that we do for other media
        // lists regarding the rule type and such, though it doesn't really
        // matter right now...
        //
        // Also, ParsingMode::all() is wrong, and should be DEFAULT.
        let context = ParserContext::new(
            Origin::Author, // Author origin.
            &document_url_data, // Document URL data.
            Some(CssRuleType::Style), // CSS rule type.
            ParsingMode::all(), // Parsing mode.
            quirks_mode, // Quirks mode.
            /* namespaces = */ Default::default(), // Namespaces.
            None, None,
        );
        let mut parserInput = ParserInput::new(&media_query); // Parser input.
        let mut parser = Parser::new(&mut parserInput); // CSS parser.
        let media_list = MediaList::parse(&context, &mut parser); // Parse media list.
        let result = media_list.evaluate(document.window().layout().device(), quirks_mode); // Evaluate media list.
        result
    }

    /// <https://html.spec.whatwg.org/multipage/#normalise-the-source-densities>
    /// @brief Normalizes the pixel densities of image sources within a `SourceSet`.
    /// Functional Utility: Calculates density descriptors for image sources that
    /// only specify width descriptors, based on the evaluated `sizes` attribute.
    ///
    /// @param source_set A mutable reference to the `SourceSet`.
    /// @param width An `Option<Length>` for the width.
    fn normalise_source_densities(&self, source_set: &mut SourceSet, width: Option<Length>) {
        // Step 1: Let source size be source set's source size list.
        let source_size = &mut source_set.source_size;

        // Find source_size_length for Step 2.2
        let source_size_length = self.evaluate_source_size_list(source_size, width); // Evaluate source size.

        // Step 2: For each image source in source set's image sources:
        for imgsource in &mut source_set.image_sources {
            // Step 2.1: If image source's descriptor has a pixel density descriptor, then continue.
            if imgsource.descriptor.density.is_some() {
                continue;
            }
            // Step 2.2: If image source's descriptor has a width descriptor, then:
            if imgsource.descriptor.width.is_some() {
                let wid = imgsource.descriptor.width.unwrap(); // Get width.
                imgsource.descriptor.density = Some(wid as f64 / source_size_length.to_f64_px()); // Calculate density.
            } else {
                //Step 2.3: Otherwise, set image source's descriptor's pixel density descriptor to 1x.
                imgsource.descriptor.density = Some(1_f64);
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/#select-an-image-source>
    /// @brief Selects the most appropriate image source from the `SourceSet`.
    /// Functional Utility: Implements the algorithm for choosing an image from a `srcset`
    /// based on device pixel ratio and image descriptors, optimizing for display quality.
    ///
    /// @return An `Option<(USVString, f64)>` containing the selected source URL and pixel density.
    fn select_image_source(&self) -> Option<(USVString, f64)> {
        // Step 1, 3: Update the source set and get its length.
        self.update_source_set();
        let source_set = &*self.source_set.borrow_mut();
        let len = source_set.image_sources.len();

        // Step 2: If source set's image sources is empty, then return null.
        if len == 0 {
            return None;
        }

        // Step 4: Remove any duplicates (same pixel density) from the list.
        let mut repeat_indices = HashSet::new(); // Track duplicate indices.
        for outer_index in 0..len {
            if repeat_indices.contains(&outer_index) {
                continue;
            }
            let imgsource = &source_set.image_sources[outer_index];
            let pixel_density = imgsource.descriptor.density.unwrap();
            for inner_index in (outer_index + 1)..len {
                let imgsource2 = &source_set.image_sources[inner_index];
                if pixel_density == imgsource2.descriptor.density.unwrap() {
                    repeat_indices.insert(inner_index); // Mark as duplicate.
                }
            }
        }

        let mut max = (0f64, 0); // Max density and its index.
        let img_sources = &mut vec![]; // Filtered image sources.
        for (index, image_source) in source_set.image_sources.iter().enumerate() {
            if repeat_indices.contains(&index) {
                continue;
            }
            let den = image_source.descriptor.density.unwrap();
            if max.0 < den {
                max = (den, img_sources.len()); // Update max if current density is higher.
            }
            img_sources.push(image_source); // Add to filtered list.
        }

        // Step 5: Find the best candidate based on device pixel ratio.
        let mut best_candidate = max; // Initialize best candidate.
        let device_pixel_ratio = self
            .owner_document()
            .window()
            .viewport_details()
            .hidpi_scale_factor
            .get() as f64; // Get device pixel ratio.
        for (index, image_source) in img_sources.iter().enumerate() {
            let current_den = image_source.descriptor.density.unwrap();
            if current_den < best_candidate.0 && current_den >= device_pixel_ratio {
                best_candidate = (current_den, index); // Update best candidate.
            }
        }
        let selected_source = img_sources.remove(best_candidate.1).clone(); // Get the selected source.
        Some((
            USVString(selected_source.url), // Source URL.
            selected_source.descriptor.density.unwrap(), // Pixel density.
        ))
    }

    /// @brief Initializes an image request with URL and source.
    /// Functional Utility: Sets up the basic properties of an `ImageRequest`,
    /// including its parsed URL, source URL, and associated load blocker.
    ///
    /// @param request A mutable reference to the `ImageRequest`.
    /// @param url The `ServoUrl` of the image.
    /// @param src The `USVString` representing the source URL.
    /// @param can_gc A `CanGc` token.
    fn init_image_request(
        &self,
        request: &mut RefMut<ImageRequest>,
        url: &ServoUrl,
        src: &USVString,
        can_gc: CanGc,
    ) {
        request.parsed_url = Some(url.clone()); // Set parsed URL.
        request.source_url = Some(src.clone()); // Set source URL.
        request.image = None; // Clear image data.
        request.metadata = None; // Clear metadata.
        let document = self.owner_document(); // Get owning document.
        LoadBlocker::terminate(&request.blocker, can_gc); // Terminate previous load blocker.
        *request.blocker.borrow_mut() =
            Some(LoadBlocker::new(&document, LoadType::Image(url.clone()))); // Create new load blocker.
    }

    /// Step 13-17 of html.spec.whatwg.org/multipage/#update-the-image-data
    /// @brief Prepares an image request for fetching.
    /// Functional Utility: Handles logic for responsive images, checking if a new
    /// image needs to be fetched based on the selected source and pixel density,
    /// and managing pending requests.
    ///
    /// @param url The `ServoUrl` of the image.
    /// @param src The `USVString` representing the source URL.
    /// @param selected_pixel_density The pixel density of the selected image.
    /// @param can_gc A `CanGc` token.
    fn prepare_image_request(
        &self,
        url: &ServoUrl,
        src: &USVString,
        selected_pixel_density: f64,
        can_gc: CanGc,
    ) {
        match self.image_request.get() {
            ImageRequestPhase::Pending => {
                if let Some(pending_url) = self.pending_request.borrow().parsed_url.clone() {
                    // Step 13: If the image element's pending request's parsed URL is equal to url, then return.
                    if pending_url == *url {
                        return;
                    }
                }
            },
            ImageRequestPhase::Current => {
                let mut current_request = self.current_request.borrow_mut(); // Mutable current request.
                let mut pending_request = self.pending_request.borrow_mut(); // Mutable pending request.
                // step 16, create a new "image_request"
                match (current_request.parsed_url.clone(), current_request.state) {
                    (Some(parsed_url), State::PartiallyAvailable) => {
                        // Step 14: If the image element's current request's parsed URL is equal to url, then:
                        if parsed_url == *url {
                            // Step 15: If there is a pending image request, then abort it.
                            pending_request.image = None; // Clear image.
                            pending_request.parsed_url = None; // Clear parsed URL.
                            LoadBlocker::terminate(&pending_request.blocker, can_gc); // Terminate load blocker.
                            // TODO: queue a task to restart animation, if restart-animation is set
                            return;
                        }
                        // Block Logic: If current is partially available but URL mismatch, prepare pending request.
                        pending_request.current_pixel_density = Some(selected_pixel_density);
                        self.image_request.set(ImageRequestPhase::Pending);
                        self.init_image_request(&mut pending_request, url, src, can_gc);
                    },
                    (_, State::Broken) | (_, State::Unavailable) => {
                        // Step 17: If the image element's current request's state is broken or unavailable.
                        current_request.current_pixel_density = Some(selected_pixel_density);
                        self.init_image_request(&mut current_request, url, src, can_gc);
                    },
                    (_, _) => {
                        // step 17: Otherwise (current request is completely available, or state not broken/unavailable).
                        pending_request.current_pixel_density = Some(selected_pixel_density);
                        self.image_request.set(ImageRequestPhase::Pending);
                        self.init_image_request(&mut pending_request, url, src, can_gc);
                    },
                }
            },
        }
        self.fetch_image(url, can_gc); // Fetch the image.
    }

    /// Step 8-12 of html.spec.whatwg.org/multipage/#update-the-image-data
    /// @brief Synchronously updates the image data.
    /// Functional Utility: Performs the synchronous part of the image data update
    /// algorithm, including selecting a source, preparing requests, and handling errors.
    ///
    /// @param can_gc A `CanGc` token.
    fn update_the_image_data_sync_steps(&self, can_gc: CanGc) {
        let document = self.owner_document(); // Get owning document.
        let global = self.owner_global(); // Get owning global scope.
        let task_manager = global.task_manager(); // Get task manager.
        let task_source = task_manager.dom_manipulation_task_source(); // Get DOM manipulation task source.
        let this = Trusted::new(self); // Trusted reference to self.
        let (src, pixel_density) = match self.select_image_source() {
            // Step 8: If the result of selecting an image source is null, then:
            Some(data) => data,
            None => {
                // Block Logic: Abort requests and queue error event if no source selected.
                self.abort_request(State::Broken, ImageRequestPhase::Current, can_gc);
                self.abort_request(State::Broken, ImageRequestPhase::Pending, can_gc);
                // Step 9. Queue a microtask to fire an error event.
                task_source.queue(task!(image_null_source_error: move || {
                    let this = this.root();
                    {
                        let mut current_request =
                            this.current_request.borrow_mut();
                        current_request.source_url = None;
                        current_request.parsed_url = None;
                    }
                    let elem = this.upcast::<Element>();
                    let src_present = elem.has_attribute(&local_name!("src"));

                    if src_present || Self::uses_srcset_or_picture(elem) {
                        this.upcast::<EventTarget>().fire_event(atom!("error"), CanGc::note());
                    }
                }));
                return;
            },
        };

        // Step 11: Let parsedURL be the result of parsing src relative to the image element's node document's base URL.
        let base_url = document.base_url(); // Get base URL.
        let parsed_url = base_url.join(&src.0); // Parse URL.
        match parsed_url {
            Ok(url) => {
                // Step 13-17: Prepare the image request.
                self.prepare_image_request(&url, &src, pixel_density, can_gc);
            },
            Err(_) => {
                // Block Logic: Abort requests and queue error event if URL parsing fails.
                self.abort_request(State::Broken, ImageRequestPhase::Current, can_gc);
                self.abort_request(State::Broken, ImageRequestPhase::Pending, can_gc);
                // Step 12.1-12.5. Queue a microtask to fire an error event.
                let src = src.0;
                task_source.queue(task!(image_selected_source_error: move || {
                    let this = this.root();
                    {
                        let mut current_request =
                            this.current_request.borrow_mut();
                        current_request.source_url = Some(USVString(src))
                    }
                    this.upcast::<EventTarget>().fire_event(atom!("error"), CanGc::note());

                }));
            },
        }
    }

    /// <https://html.spec.whatwg.org/multipage/#update-the-image-data>
    /// @brief Triggers an update of the image data based on attributes.
    /// Functional Utility: Initiates the process of fetching and displaying
    /// the correct image based on `src`, `srcset`, and `picture` element rules.
    ///
    /// @param can_gc A `CanGc` token.
    pub(crate) fn update_the_image_data(&self, can_gc: CanGc) {
        let document = self.owner_document(); // Get owning document.
        let window = document.window(); // Get owning window.
        let elem = self.upcast::<Element>(); // Get element.
        let src = elem.get_url_attribute(&local_name!("src")); // Get `src` attribute.
        let base_url = document.base_url(); // Get base URL.

        // https://html.spec.whatwg.org/multipage/#reacting-to-dom-mutations
        // Always first set the current request to unavailable,
        // ensuring img.complete is false.
        {
            let mut current_request = self.current_request.borrow_mut(); // Mutable current request.
            current_request.state = State::Unavailable; // Set state to unavailable.
        }

        if !document.is_active() {
            // Step 1 (if the document is inactive)
            // TODO: use GlobalScope::enqueue_microtask,
            // to queue micro task to come back to this algorithm
        }
        // Step 2 abort if user-agent does not supports images
        // NOTE: Servo only supports images, skipping this step

        // Step 3, 4: Let selectedSource be the empty string.
        // Let selectedPixelDensity be 1x.
        let mut selected_source = None;
        let mut pixel_density = None;
        let src_set = elem.get_url_attribute(&local_name!("srcset")); // Get `srcset` attribute.
        let is_parent_picture = elem
            .upcast::<Node>()
            .GetParentElement()
            .is_some_and(|p| p.is::<HTMLPictureElement>()); // Check if parent is `<picture>`.
        if src_set.is_empty() && !is_parent_picture && !src.is_empty() {
            selected_source = Some(src.clone()); // If no `srcset` or `picture`, use `src`.
            pixel_density = Some(1_f64); // Default pixel density.
        };

        // Step 5: Update image element's last selected source to selectedSource.
        self.last_selected_source
            .borrow_mut()
            .clone_from(&selected_source);

        // Step 6, Check the list of available images.
        if let Some(src) = selected_source {
            if let Ok(img_url) = base_url.join(&src) {
                let image_cache = window.image_cache(); // Get image cache.
                let response = image_cache.get_image(
                    img_url.clone(), // Image URL.
                    window.origin().immutable().clone(), // Origin.
                    cors_setting_for_element(self.upcast()), // CORS setting.
                    UsePlaceholder::No, // Do not use placeholder here.
                );

                if let Some(image) = response {
                    // Cancel any outstanding tasks that were queued before the src was
                    // set on this element.
                    self.generation.set(self.generation.get() + 1); // Increment generation.
                    // Step 6.3: Let metadata be image's metadata.
                    let metadata = ImageMetadata {
                        height: image.height,
                        width: image.width,
                    };
                    // Step 6.3.2: Abort any pending image requests.
                    self.abort_request(
                        State::CompletelyAvailable,
                        ImageRequestPhase::Current,
                        can_gc,
                    );
                    self.abort_request(State::Unavailable, ImageRequestPhase::Pending, can_gc);
                    let mut current_request = self.current_request.borrow_mut(); // Mutable current request.
                    current_request.final_url = Some(img_url.clone()); // Set final URL.
                    current_request.image = Some(image.clone()); // Set image.
                    current_request.metadata = Some(metadata); // Set metadata.
                    // Step 6.3.6: Set image element's current request's current pixel density
                    // to selectedPixelDensity.
                    current_request.current_pixel_density = pixel_density;
                    let this = Trusted::new(self); // Trusted reference to self.
                    let src = src.0;

                    self.owner_global()
                        .task_manager()
                        .dom_manipulation_task_source()
                        .queue(task!(image_load_event: move || {
                            let this = this.root();
                            {
                                let mut current_request =
                                    this.current_request.borrow_mut();
                                current_request.parsed_url = Some(img_url);
                                current_request.source_url = Some(USVString(src));
                            }
                            // TODO: restart animation, if set.
                            this.upcast::<EventTarget>().fire_event(atom!("load"), CanGc::note()); // Fire load event.
                        }));
                    return; // Done with this update.
                }
            }
        }
        // Step 7: Await a stable state.
        self.generation.set(self.generation.get() + 1); // Increment generation.
        let task = ImageElementMicrotask::StableStateUpdateImageData {
            elem: DomRoot::from_ref(self), // Reference to self.
            generation: self.generation.get(), // Current generation.
        };
        ScriptThread::await_stable_state(Microtask::ImageElement(task)); // Await stable state.
    }

    /// <https://html.spec.whatwg.org/multipage/#img-environment-changes>
    /// @brief Reacts to environment changes that might affect image selection.
    /// Functional Utility: Triggers the re-evaluation of responsive image sources
    /// when environmental factors (like viewport size) change.
    pub(crate) fn react_to_environment_changes(&self) {
        // Step 1: Queue an ImageElementMicrotask::EnvironmentChanges task.
        let task = ImageElementMicrotask::EnvironmentChanges {
            elem: DomRoot::from_ref(self), // Reference to self.
            generation: self.generation.get(), // Current generation.
        };
        ScriptThread::await_stable_state(Microtask::ImageElement(task)); // Await stable state.
    }

    /// Step 2-12 of <https://html.spec.whatwg.org/multipage/#img-environment-changes>
    /// @brief Synchronously processes environment changes for responsive image selection.
    /// Functional Utility: Performs the synchronous part of the responsive image update
    /// algorithm, checking for necessary image source changes and preparing new requests.
    ///
    /// @param generation The generation ID of the image request.
    /// @param can_gc A `CanGc` token.
    fn react_to_environment_changes_sync_steps(&self, generation: u32, can_gc: CanGc) {
        let elem = self.upcast::<Element>(); // Get element.
        let document = elem.owner_document(); // Get owning document.
        let has_pending_request = matches!(self.image_request.get(), ImageRequestPhase::Pending); // Check for pending request.

        // Step 2: If the image element's node document is not active, or if it does not use a
        // srcset attribute or picture element, or if the image element has a pending image
        // request, then return.
        if !document.is_active() || !Self::uses_srcset_or_picture(elem) || has_pending_request {
            return;
        }

        // Steps 3-4: Let selectedSource and selectedPixelDensity be the result of selecting an
        // image source, given the image element. If selectedSource is null, return.
        let (selected_source, selected_pixel_density) = match self.select_image_source() {
            Some(selected) => selected,
            None => return, // No source selected.
        };

        // Step 5: If image element's last selected source is selectedSource, and image element's
        // current request's current pixel density is selectedPixelDensity, then return.
        let same_source = match *self.last_selected_source.borrow() {
            Some(ref last_src) => *last_src == selected_source,
            _ => false,
        };

        let same_selected_pixel_density = match self.current_request.borrow().current_pixel_density
        {
            Some(den) => selected_pixel_density == den,
            _ => false,
        };

        if same_source && same_selected_pixel_density {
            return;
        }

        let base_url = document.base_url(); // Get base URL.
        // Step 6: Let url be the result of parsing selectedSource relative to the image element's
        // node document's base URL. If url is failure, return.
        let img_url = match base_url.join(&selected_source.0) {
            Ok(url) => url,
            Err(_) => return, // URL parsing failed.
        };

        // Step 12: Let the image element's image request phase be pending.
        self.image_request.set(ImageRequestPhase::Pending);
        self.init_image_request(
            &mut self.pending_request.borrow_mut(),
            &img_url,
            &selected_source,
            can_gc,
        );

        // Step 14: Let response be the result of fetching url from the image cache.
        let window = self.owner_window(); // Get owning window.
        let cache_result = window.image_cache().get_cached_image_status(
            img_url.clone(), // Image URL.
            window.origin().immutable().clone(), // Origin.
            cors_setting_for_element(self.upcast()), // CORS setting.
            UsePlaceholder::No, // Do not use placeholder.
        );

        let change_type = ChangeType::Environment {
            selected_source: selected_source.clone(), // Selected source.
            selected_pixel_density, // Selected pixel density.
        };

        match cache_result {
            ImageCacheResult::Available(ImageOrMetadataAvailable::ImageAvailable { .. }) => {
                // Step 15: If response is an image, then:
                self.finish_reacting_to_environment_change(
                    selected_source,
                    generation,
                    selected_pixel_density,
                )
            },
            ImageCacheResult::Available(ImageOrMetadataAvailable::MetadataAvailable(m, id)) => {
                // Block Logic: Metadata available, process and register callback.
                self.process_image_response_for_environment_change(
                    ImageResponse::MetadataLoaded(m),
                    selected_source,
                    generation,
                    selected_pixel_density,
                    can_gc,
                );
                self.register_image_cache_callback(id, change_type);
            },
            ImageCacheResult::LoadError => {
                // Block Logic: Load error, process with no image data.
                self.process_image_response_for_environment_change(
                    ImageResponse::None,
                    selected_source,
                    generation,
                    selected_pixel_density,
                    can_gc,
                );
            },
            ImageCacheResult::ReadyForRequest(id) => {
                // Block Logic: Ready for network request.
                self.fetch_request(&img_url, id); // Fetch request.
                self.register_image_cache_callback(id, change_type); // Register callback.
            },
            ImageCacheResult::Pending(id) => {
                // Block Logic: Request pending.
                self.register_image_cache_callback(id, change_type); // Register callback.
            },
        }
    }

    // Step 2 for <https://html.spec.whatwg.org/multipage/#dom-img-decode>
    /// @brief Synchronously reacts to image decode requests.
    /// Functional Utility: Resolves or rejects an image decode promise based on
    /// the image's current loading state.
    ///
    /// @param promise The `Rc<Promise>` to resolve or reject.
    /// @param can_gc A `CanGc` token.
    fn react_to_decode_image_sync_steps(&self, promise: Rc<Promise>, can_gc: CanGc) {
        let document = self.owner_document(); // Get owning document.
        // Step 2.1 of <https://html.spec.whatwg.org/multipage/#dom-img-decode>
        // If the image element's node document is not active, or if its current request's
        // state is broken, then reject promise with an "EncodingError" DOMException.
        if !document.is_fully_active() ||
            matches!(self.current_request.borrow().state, State::Broken)
        {
            promise.reject_native(
                &DOMException::new(&document.global(), DOMErrorName::EncodingError, can_gc),
                can_gc,
            ); // Reject promise with EncodingError.
        } else if matches!(
            self.current_request.borrow().state,
            State::CompletelyAvailable
        ) {
            // this doesn't follow the spec, but it's been discussed in <https://github.com/whatwg/html/issues/4217>
            promise.resolve_native(&(), can_gc); // Resolve promise if image is available.
        } else {
            self.image_decode_promises
                .borrow_mut()
                .push(promise.clone()); // Otherwise, add promise to pending list.
        }
    }

    /// @brief Resolves all pending image decode promises.
    /// Functional Utility: Marks all currently pending image decode promises as successful.
    /// @param can_gc A `CanGc` token.
    fn resolve_image_decode_promises(&self, can_gc: CanGc) {
        for promise in self.image_decode_promises.borrow().iter() {
            promise.resolve_native(&(), can_gc); // Resolve each promise.
        }
        self.image_decode_promises.borrow_mut().clear(); // Clear the list.
    }

    /// @brief Rejects all pending image decode promises with an `EncodingError`.
    /// Functional Utility: Marks all currently pending image decode promises as failed.
    /// @param can_gc A `CanGc` token.
    fn reject_image_decode_promises(&self, can_gc: CanGc) {
        let document = self.owner_document(); // Get owning document.
        for promise in self.image_decode_promises.borrow().iter() {
            promise.reject_native(
                &DOMException::new(&document.global(), DOMErrorName::EncodingError, can_gc),
                can_gc,
            ); // Reject each promise with EncodingError.
        }
        self.image_decode_promises.borrow_mut().clear(); // Clear the list.
    }

    /// Step 15 for <https://html.spec.whatwg.org/multipage/#img-environment-changes>
    /// @brief Finalizes the reaction to an environment change.
    /// Functional Utility: Updates the image element's state with the newly selected
    /// image source and fires a `load` event.
    ///
    /// @param src The `USVString` representing the selected source URL.
    /// @param generation The generation ID of the image request.
    /// @param selected_pixel_density The pixel density of the selected image.
    fn finish_reacting_to_environment_change(
        &self,
        src: USVString,
        generation: u32,
        selected_pixel_density: f64,
    ) {
        let this = Trusted::new(self); // Trusted reference to self.
        let src = src.0;
        self.owner_global().task_manager().dom_manipulation_task_source().queue(
            task!(image_load_event: move || {
                let this = this.root();
                let relevant_mutation = this.generation.get() != generation;
                // Step 15.1: If the image element's generation is not equal to generation, then:
                if relevant_mutation {
                    this.abort_request(State::Unavailable, ImageRequestPhase::Pending, CanGc::note()); // Abort pending request.
                    return;
                }
                // Step 15.2: Set the image element's last selected source to src.
                *this.last_selected_source.borrow_mut() = Some(USVString(src));

                {
                    let mut pending_request = this.pending_request.borrow_mut(); // Mutable pending request.
                    pending_request.current_pixel_density = Some(selected_pixel_density); // Set pixel density.

                    // Step 15.3: Set the image element's pending request's state to completely available.
                    pending_request.state = State::CompletelyAvailable;

                    // Step 15.4: Add the image element's pending request's image to the list of available images.
                    // Already a part of the list of available images due to Step 14

                    // Step 15.5: Swap the image element's current request and pending request.
                    mem::swap(&mut this.current_request.borrow_mut(), &mut pending_request);
                }
                this.abort_request(State::Unavailable, ImageRequestPhase::Pending, CanGc::note()); // Abort pending request.

                // Step 15.6: Mark the image element's relevant Document as having a pending visual update.
                this.upcast::<Node>().dirty(NodeDamage::OtherNodeDamage);

                // Step 15.7: Fire a simple event that is cancelable named load at the image element.
                this.upcast::<EventTarget>().fire_event(atom!("load"), CanGc::note()); // Fire load event.
            })
        );
    }

    /// @brief Checks if the image element uses `srcset` or is within a `picture` element.
    /// Functional Utility: Determines if the image element is participating in the responsive
    /// image selection mechanism.
    ///
    /// @param elem The `Element` to check.
    /// @return `true` if it uses `srcset` or `picture`, `false` otherwise.
    fn uses_srcset_or_picture(elem: &Element) -> bool {
        let has_src = elem.has_attribute(&local_name!("srcset")); // Check for `srcset` attribute.
        let is_parent_picture = elem
            .upcast::<Node>()
            .GetParentElement()
            .is_some_and(|p| p.is::<HTMLPictureElement>()); // Check if parent is `<picture>`.
        has_src || is_parent_picture // Return true if either is true.
    }

    /// @brief Creates a new `HTMLImageElement` instance with inherited properties.
    /// Functional Utility: Internal constructor for setting up the basic properties
    /// of an `HTMLImageElement` and initializing its internal state for image requests.
    ///
    /// @param local_name The `LocalName` of the HTML element (e.g., "img").
    /// @param prefix An `Option<Prefix>` for the XML namespace prefix, if any.
    /// @param document The owning `Document` of this element.
    /// @return A new `HTMLImageElement` instance.
    fn new_inherited(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
    ) -> HTMLImageElement {
        HTMLImageElement {
            htmlelement: HTMLElement::new_inherited(local_name, prefix, document), // Initialize base HTMLElement.
            image_request: Cell::new(ImageRequestPhase::Current), // Start with current image request phase.
            current_request: DomRefCell::new(ImageRequest {
                state: State::Unavailable, // Initial state is unavailable.
                parsed_url: None,
                source_url: None,
                image: None,
                metadata: None,
                blocker: DomRefCell::new(None),
                final_url: None,
                current_pixel_density: None,
            }),
            pending_request: DomRefCell::new(ImageRequest {
                state: State::Unavailable, // Initial state is unavailable.
                parsed_url: None,
                source_url: None,
                image: None,
                metadata: None,
                blocker: DomRefCell::new(None),
                final_url: None,
                current_pixel_density: None,
            }),
            form_owner: Default::default(), // No form owner by default.
            generation: Default::default(), // Generation starts at default.
            source_set: DomRefCell::new(SourceSet::new()), // Empty source set initially.
            last_selected_source: DomRefCell::new(None), // No last selected source.
            image_decode_promises: DomRefCell::new(vec![]), // Empty promises list.
        }
    }

    /// @brief Creates a new `HTMLImageElement` and reflects it into the DOM.
    /// Functional Utility: Public constructor that builds an `HTMLImageElement` and makes
    /// it accessible from JavaScript by integrating it into the DOM's object graph.
    ///
    /// @param local_name The `LocalName` of the HTML element.
    /// @param prefix An `Option<Prefix>` for the XML namespace prefix.
    /// @param document The owning `Document` of this element.
    /// @param proto An `Option<HandleObject>` specifying the JavaScript prototype chain.
    /// @param can_gc A `CanGc` token.
    /// @return A new `HTMLImageElement` instance wrapped in `DomRoot`.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn new(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
        proto: Option<HandleObject>,
        can_gc: CanGc,
    ) -> DomRoot<HTMLImageElement> {
        Node::reflect_node_with_proto(
            Box::new(HTMLImageElement::new_inherited(
                local_name, prefix, document,
            )),
            document,
            proto,
            can_gc,
        )
    }

    /// @brief Retrieves associated `<area>` elements for client-side image maps.
    /// Functional Utility: Finds the `<map>` element referenced by the `usemap`
    /// attribute and returns its `<area>` children.
    ///
    /// @return An `Option<Vec<DomRoot<HTMLAreaElement>>>` containing the area elements, or `None`.
    pub(crate) fn areas(&self) -> Option<Vec<DomRoot<HTMLAreaElement>>> {
        let elem = self.upcast::<Element>(); // Get element.
        let usemap_attr = elem.get_attribute(&ns!(), &local_name!("usemap"))?; // Get `usemap` attribute.

        let value = usemap_attr.value(); // Get attribute value.

        // Block Logic: Validate `usemap` value (must start with '#').
        if value.len() == 0 || !value.is_char_boundary(1) {
            return None;
        }

        let (first, last) = value.split_at(1); // Split by first character.

        if first != "#" || last.is_empty() {
            return None;
        }

        // Block Logic: Find the `<map>` element by name.
        let useMapElements = self
            .owner_document()
            .upcast::<Node>()
            .traverse_preorder(ShadowIncluding::No) // Traverse document.
            .filter_map(DomRoot::downcast::<HTMLMapElement>) // Filter for HTMLMapElement.
            .find(|n| {
                n.upcast::<Element>()
                    .get_name()
                    .is_some_and(|n| *n == *last) // Match by name.
            });

        useMapElements.map(|mapElem| mapElem.get_area_elements()) // Get area elements from map.
    }

    /// @brief Checks if the image is from the same origin.
    /// Functional Utility: Determines if the loaded image resource adheres to
    /// same-origin policy, which affects its accessibility from scripts.
    ///
    /// @param origin The `MutableOrigin` to compare against.
    /// @return `true` if same-origin or CORS-safe, `false` otherwise.
    pub(crate) fn same_origin(&self, origin: &MutableOrigin) -> bool {
        if let Some(ref image) = self.current_request.borrow().image {
            return image.cors_status == CorsStatus::Safe; // Check CORS status of loaded image.
        }

        self.current_request
            .borrow()
            .final_url
            .as_ref()
            .is_some_and(|url| url.scheme() == "data" || url.origin().same_origin(origin)) // Check URL origin.
    }

    /// @brief Returns the current generation ID of the image element.
    /// Functional Utility: Provides a mechanism to track changes to the image element's
    /// source and to discard outdated image requests.
    /// @return The current generation ID (`u32`).
    fn generation_id(&self) -> u32 {
        self.generation.get()
    }
}

/// @enum ImageElementMicrotask
/// @brief Represents different types of microtasks for image elements.
/// Functional Utility: Provides a way to schedule asynchronous operations
/// (like updating image data or reacting to environment changes) to be executed
/// at a stable state of the DOM.
#[derive(JSTraceable, MallocSizeOf)]
pub(crate) enum ImageElementMicrotask {
    StableStateUpdateImageData {
        elem: DomRoot<HTMLImageElement>, //!< The image element.
        generation: u32,               //!< The generation ID when the task was queued.
    },
    EnvironmentChanges {
        elem: DomRoot<HTMLImageElement>, //!< The image element.
        generation: u32,               //!< The generation ID when the task was queued.
    },
    Decode {
        elem: DomRoot<HTMLImageElement>, //!< The image element.
        #[ignore_malloc_size_of = "promises are hard"]
        promise: Rc<Promise>, //!< The promise associated with image decoding.
    },
}

impl MicrotaskRunnable for ImageElementMicrotask {
    /// @brief Handles the execution of the microtask.
    /// Functional Utility: Dispatches to the appropriate synchronous method based on the
    /// microtask type.
    ///
    /// @param can_gc A `CanGc` token.
    fn handler(&self, can_gc: CanGc) {
        match *self {
            ImageElementMicrotask::StableStateUpdateImageData {
                ref elem,
                ref generation,
            } => {
                // Step 7 of https://html.spec.whatwg.org/multipage/#update-the-image-data,
                // stop here if other instances of this algorithm have been scheduled
                // Block Logic: Only proceed if the generation ID matches the current one.
                if elem.generation.get() == *generation {
                    elem.update_the_image_data_sync_steps(can_gc); // Execute synchronous steps.
                }
            },
            ImageElementMicrotask::EnvironmentChanges {
                ref elem,
                ref generation,
            } => {
                elem.react_to_environment_changes_sync_steps(*generation, can_gc); // Execute synchronous steps.
            },
            ImageElementMicrotask::Decode {
                ref elem,
                ref promise,
            } => {
                elem.react_to_decode_image_sync_steps(promise.clone(), can_gc); // Execute synchronous steps.
            },
        }
    }

    /// @brief Enters the JavaScript realm for the microtask.
    /// Functional Utility: Ensures that the microtask executes within the correct
    /// JavaScript execution context.
    /// @return A `JSAutoRealm` instance.
    fn enter_realm(&self) -> JSAutoRealm {
        match self {
            &ImageElementMicrotask::StableStateUpdateImageData { ref elem, .. } |
            &ImageElementMicrotask::EnvironmentChanges { ref elem, .. } |
            &ImageElementMicrotask::Decode { ref elem, .. } => enter_realm(&**elem), // Enter realm of the image element.
        }
    }
}

/// @trait LayoutHTMLImageElementHelpers
/// @brief Trait for layout-specific helper methods for `HTMLImageElement`.
/// Functional Utility: Provides read-only access to image-related properties
/// that are relevant to the layout engine.
pub(crate) trait LayoutHTMLImageElementHelpers {
    /// @brief Returns the URL of the image.
    fn image_url(self) -> Option<ServoUrl>;
    /// @brief Returns the pixel density of the image.
    fn image_density(self) -> Option<f64>;
    /// @brief Returns the image data and metadata.
    fn image_data(self) -> (Option<Arc<Image>>, Option<ImageMetadata>);
    /// @brief Returns the width of the image.
    fn get_width(self) -> LengthOrPercentageOrAuto;
    /// @brief Returns the height of the image.
    fn get_height(self) -> LengthOrPercentageOrAuto;
}

impl<'dom> LayoutDom<'dom, HTMLImageElement> {
    /// @brief Returns a reference to the current image request for layout.
    /// Functional Utility: Provides a layout-safe way to access the image request state.
    /// @return An immutable reference to `ImageRequest`.
    #[allow(unsafe_code)]
    fn current_request(self) -> &'dom ImageRequest {
        unsafe { self.unsafe_get().current_request.borrow_for_layout() } // Unsafely borrow for layout.
    }
}

impl LayoutHTMLImageElementHelpers for LayoutDom<'_, HTMLImageElement> {
    /// @brief Returns the URL of the image for layout.
    fn image_url(self) -> Option<ServoUrl> {
        self.current_request().parsed_url.clone()
    }

    /// @brief Returns the image data and metadata for layout.
    fn image_data(self) -> (Option<Arc<Image>>, Option<ImageMetadata>) {
        let current_request = self.current_request(); // Get current request.
        (
            current_request.image.clone(),    // Clone image data.
            current_request.metadata.clone(), // Clone metadata.
        )
    }

    /// @brief Returns the pixel density of the image for layout.
    fn image_density(self) -> Option<f64> {
        self.current_request().current_pixel_density
    }

    /// @brief Returns the width of the image for layout.
    fn get_width(self) -> LengthOrPercentageOrAuto {
        self.upcast::<Element>()
            .get_attr_for_layout(&ns!(), &local_name!("width")) // Get width attribute.
            .map(AttrValue::as_dimension) // Map to dimension.
            .cloned() // Clone.
            .unwrap_or(LengthOrPercentageOrAuto::Auto) // Default to auto.
    }

    /// @brief Returns the height of the image for layout.
    fn get_height(self) -> LengthOrPercentageOrAuto {
        self.upcast::<Element>()
            .get_attr_for_layout(&ns!(), &local_name!("height")) // Get height attribute.
            .map(AttrValue::as_dimension) // Map to dimension.
            .cloned() // Clone.
            .unwrap_or(LengthOrPercentageOrAuto::Auto) // Default to auto.
    }
}

//https://html.spec.whatwg.org/multipage/#parse-a-sizes-attribute
/// @brief Parses a `sizes` attribute string into a `SourceSizeList`.
/// Functional Utility: Interprets the `sizes` attribute, which specifies the
/// intended display size of an image, typically for responsive image selection.
///
/// @param value The `DOMString` value of the `sizes` attribute.
/// @return A `SourceSizeList`.
pub(crate) fn parse_a_sizes_attribute(value: DOMString) -> SourceSizeList {
    let mut input = ParserInput::new(&value); // Parser input.
    let mut parser = Parser::new(&mut input); // CSS parser.
    let url_data = Url::parse("about:blank").unwrap().into(); // Base URL for context.
    let context = ParserContext::new(
        Origin::Author, // Author origin.
        &url_data, // URL data.
        Some(CssRuleType::Style), // CSS rule type.
        // FIXME(emilio): why ::empty() instead of ::DEFAULT? Also, what do
        // browsers do regarding quirks-mode in a media list?
        ParsingMode::empty(), // Parsing mode.
        QuirksMode::NoQuirks, // No quirks mode.
        /* namespaces = */ Default::default(), // Namespaces.
        None, None,
    );
    SourceSizeList::parse(&context, &mut parser) // Parse SourceSizeList.
}

/// @brief Gets the correct referrer policy from a raw token.
/// Functional Utility: Validates and normalizes a referrer policy token,
/// returning an empty `DOMString` for invalid or default policies.
///
/// @param token The raw referrer policy `DOMString`.
/// @return A `DOMString` representing the normalized referrer policy.
fn get_correct_referrerpolicy_from_raw_token(token: &DOMString) -> DOMString {
    if token == "" {
        // Empty token is treated as the default referrer policy inside determine_policy_for_token,
        // so it should remain unchanged.
        DOMString::new()
    } else {
        let policy = determine_policy_for_token(token); // Determine policy.

        if policy == ReferrerPolicy::EmptyString {
            return DOMString::new(); // Return empty string for empty policy.
        }

        DOMString::from_string(policy.to_string()) // Convert policy to string.
    }
}

#[allow(non_snake_case)]
impl HTMLImageElementMethods<crate::DomTypeHolder> for HTMLImageElement {
    // https://html.spec.whatwg.org/multipage/#dom-image
    /// @brief Creates a new `HTMLImageElement` instance (constructor).
    /// Functional Utility: Provides the JavaScript constructor for `Image`, allowing
    /// creation of `<img>` elements programmatically, with optional width and height.
    ///
    /// @param window The `Window` object.
    /// @param proto An `Option<HandleObject>` specifying the JavaScript prototype chain.
    /// @param can_gc A `CanGc` token.
    /// @param width An `Option<u32>` for the image width.
    /// @param height An `Option<u32>` for the image height.
    /// @return A `Fallible<DomRoot<HTMLImageElement>>` containing the new image element.
    fn Image(
        window: &Window,
        proto: Option<HandleObject>,
        can_gc: CanGc,
        width: Option<u32>,
        height: Option<u32>,
    ) -> Fallible<DomRoot<HTMLImageElement>> {
        let element = Element::create(
            QualName::new(None, ns!(html), local_name!("img")), // Qualified name for `<img>`.
            None, // No prefix.
            &window.Document(), // Owning document.
            ElementCreator::ScriptCreated, // Script-created element.
            CustomElementCreationMode::Synchronous, // Synchronous creation.
            proto, // JavaScript prototype.
            can_gc,
        );

        let image = DomRoot::downcast::<HTMLImageElement>(element).unwrap(); // Downcast to HTMLImageElement.
        if let Some(w) = width {
            image.SetWidth(w, can_gc); // Set width if provided.
        }
        if let Some(h) = height {
            image.SetHeight(h, can_gc); // Set height if provided.
        }

        // run update_the_image_data when the element is created.
        // https://html.spec.whatwg.org/multipage/#when-to-obtain-images
        image.update_the_image_data(can_gc); // Update image data upon creation.

        Ok(image)
    }

    // https://html.spec.whatwg.org/multipage/#dom-img-alt
    /// @brief Returns the `alt` attribute of the image.
    /// Functional Utility: Implements the `alt` getter, providing alternative text
    /// for the image, important for accessibility.
    make_getter!(Alt, "alt");
    // https://html.spec.whatwg.org/multipage/#dom-img-alt
    /// @brief Sets the `alt` attribute of the image.
    /// Functional Utility: Implements the `alt` setter.
    make_setter!(SetAlt, "alt");

    // https://html.spec.whatwg.org/multipage/#dom-img-src
    /// @brief Returns the `src` attribute of the image as an absolute URL.
    /// Functional Utility: Implements the `src` getter, providing the URL of the image.
    make_url_getter!(Src, "src");

    // https://html.spec.whatwg.org/multipage/#dom-img-src
    /// @brief Sets the `src` attribute of the image.
    /// Functional Utility: Implements the `src` setter.
    make_url_setter!(SetSrc, "src");

    // https://html.spec.whatwg.org/multipage/#dom-img-srcset
    /// @brief Returns the `srcset` attribute of the image as an absolute URL.
    /// Functional Utility: Implements the `srcset` getter, providing a list of
    /// image sources for responsive images.
    make_url_getter!(Srcset, "srcset");
    // https://html.spec.whatwg.org/multipage/#dom-img-src
    /// @brief Sets the `srcset` attribute of the image.
    /// Functional Utility: Implements the `srcset` setter.
    make_url_setter!(SetSrcset, "srcset");

    // https://html.spec.whatwg.org/multipage/#dom-img-crossOrigin
    /// @brief Returns the `crossorigin` attribute of the image.
    /// Functional Utility: Implements the `crossOrigin` getter, indicating how
    /// CORS requests for the image should be handled.
    fn GetCrossOrigin(&self) -> Option<DOMString> {
        reflect_cross_origin_attribute(self.upcast::<Element>())
    }

    // https://html.spec.whatwg.org/multipage/#dom-img-crossOrigin
    /// @brief Sets the `crossorigin` attribute of the image.
    /// Functional Utility: Implements the `crossOrigin` setter.
    fn SetCrossOrigin(&self, value: Option<DOMString>, can_gc: CanGc) {
        set_cross_origin_attribute(self.upcast::<Element>(), value, can_gc);
    }

    // https://html.spec.whatwg.org/multipage/#dom-img-usemap
    /// @brief Returns the `usemap` attribute of the image.
    /// Functional Utility: Implements the `useMap` getter, referencing an image map.
    make_getter!(UseMap, "usemap");
    // https://html.spec.whatwg.org/multipage/#dom-img-usemap
    /// @brief Sets the `usemap` attribute of the image.
    /// Functional Utility: Implements the `useMap` setter.
    make_setter!(SetUseMap, "usemap");

    // https://html.spec.whatwg.org/multipage/#dom-img-ismap
    /// @brief Returns `true` if the `ismap` attribute is present.
    /// Functional Utility: Implements the `isMap` getter, indicating if the image
    /// is a server-side image map.
    make_bool_getter!(IsMap, "ismap");
    // https://html.spec.whatwg.org/multipage/#dom-img-ismap
    /// @brief Sets the `ismap` attribute of the image.
    /// Functional Utility: Implements the `isMap` setter.
    make_bool_setter!(SetIsMap, "ismap");

    // https://html.spec.whatwg.org/multipage/#dom-img-width
    /// @brief Returns the current rendered width of the image.
    /// Functional Utility: Implements the `width` getter, providing the actual
    /// width of the image as rendered on the page.
    /// @param can_gc A `CanGc` token.
    /// @return The rendered width (`u32`).
    fn Width(&self, can_gc: CanGc) -> u32 {
        let node = self.upcast::<Node>(); // Upcast to Node.
        match node.bounding_content_box(can_gc) {
            Some(rect) => rect.size.width.to_px() as u32, // Return width from bounding box.
            None => self.NaturalWidth(), // Fallback to natural width.
        }
    }

    // https://html.spec.whatwg.org/multipage/#dom-img-width
    /// @brief Sets the `width` attribute of the image.
    /// Functional Utility: Implements the `width` setter.
    /// @param value The width (`u32`) to set.
    /// @param can_gc A `CanGc` token.
    fn SetWidth(&self, value: u32, can_gc: CanGc) {
        image_dimension_setter(self.upcast(), local_name!("width"), value, can_gc); // Use helper for dimension setter.
    }

    // https://html.spec.whatwg.org/multipage/#dom-img-height
    /// @brief Returns the current rendered height of the image.
    /// Functional Utility: Implements the `height` getter, providing the actual
    /// height of the image as rendered on the page.
    /// @param can_gc A `CanGc` token.
    /// @return The rendered height (`u32`).
    fn Height(&self, can_gc: CanGc) -> u32 {
        let node = self.upcast::<Node>(); // Upcast to Node.
        match node.bounding_content_box(can_gc) {
            Some(rect) => rect.size.height.to_px() as u32, // Return height from bounding box.
            None => self.NaturalHeight(), // Fallback to natural height.
        }
    }

    // https://html.spec.whatwg.org/multipage/#dom-img-height
    /// @brief Sets the `height` attribute of the image.
    /// Functional Utility: Implements the `height` setter.
    /// @param value The height (`u32`) to set.
    /// @param can_gc A `CanGc` token.
    fn SetHeight(&self, value: u32, can_gc: CanGc) {
        image_dimension_setter(self.upcast(), local_name!("height"), value, can_gc); // Use helper for dimension setter.
    }

    // https://html.spec.whatwg.org/multipage/#dom-img-naturalwidth
    /// @brief Returns the intrinsic (natural) width of the image.
    /// Functional Utility: Implements the `naturalWidth` getter, providing the
    /// original width of the image resource, adjusted for pixel density.
    /// @return The natural width (`u32`).
    fn NaturalWidth(&self) -> u32 {
        let request = self.current_request.borrow(); // Borrow current request.
        let pixel_density = request.current_pixel_density.unwrap_or(1f64); // Get pixel density.

        match request.metadata {
            Some(ref metadata) => (metadata.width as f64 / pixel_density) as u32, // Calculate adjusted width.
            None => 0, // No metadata, so 0.
        }
    }

    // https://html.spec.whatwg.org/multipage/#dom-img-naturalheight
    /// @brief Returns the intrinsic (natural) height of the image.
    /// Functional Utility: Implements the `naturalHeight` getter, providing the
    /// original height of the image resource, adjusted for pixel density.
    /// @return The natural height (`u32`).
    fn NaturalHeight(&self) -> u32 {
        let request = self.current_request.borrow(); // Borrow current request.
        let pixel_density = request.current_pixel_density.unwrap_or(1f64); // Get pixel density.

        match request.metadata {
            Some(ref metadata) => (metadata.height as f64 / pixel_density) as u32, // Calculate adjusted height.
            None => 0, // No metadata, so 0.
        }
    }

    // https://html.spec.whatwg.org/multipage/#dom-img-complete
    /// @brief Returns `true` if the image has finished loading.
    /// Functional Utility: Implements the `complete` getter, indicating whether the
    /// image has either fully loaded or failed.
    /// @return `true` if complete, `false` otherwise.
    fn Complete(&self) -> bool {
        let elem = self.upcast::<Element>(); // Get element.
        let srcset_absent = !elem.has_attribute(&local_name!("srcset")); // Check if `srcset` is absent.
        if !elem.has_attribute(&local_name!("src")) && srcset_absent {
            return true; // If no `src` and no `srcset`, considered complete.
        }
        let src = elem.get_string_attribute(&local_name!("src")); // Get `src` attribute.
        if srcset_absent && src.is_empty() {
            return true; // If no `srcset` and empty `src`, considered complete.
        }
        let request = self.current_request.borrow(); // Borrow current request.
        let request_state = request.state; // Get request state.
        match request_state {
            State::CompletelyAvailable | State::Broken => true, // Complete if available or broken.
            State::PartiallyAvailable | State::Unavailable => false, // Not complete otherwise.
        }
    }

    // https://html.spec.whatwg.org/multipage/#dom-img-currentsrc
    /// @brief Returns the current source URL of the image.
    /// Functional Utility: Implements the `currentSrc` getter, providing the
    /// absolute URL of the image that is actually being displayed (after `srcset` resolution).
    /// @return A `USVString` representing the current source URL.
    fn CurrentSrc(&self) -> USVString {
        let current_request = self.current_request.borrow(); // Borrow current request.
        let url = &current_request.parsed_url; // Parsed URL.
        match *url {
            Some(ref url) => USVString(url.clone().into_string()), // If parsed URL exists, use it.
            None => {
                let unparsed_url = &current_request.source_url; // Unparsed source URL.
                match *unparsed_url {
                    Some(ref url) => url.clone(), // If unparsed URL exists, use it.
                    None => USVString("".to_owned()), // Otherwise, empty string.
                }
            },
        }
    }

    /// <https://html.spec.whatwg.org/multipage/#dom-img-referrerpolicy>
    /// @brief Returns the `referrerpolicy` attribute of the image.
    /// Functional Utility: Implements the `referrerPolicy` getter.
    fn ReferrerPolicy(&self) -> DOMString {
        reflect_referrer_policy_attribute(self.upcast::<Element>()) // Delegates to helper.
    }

    // https://html.spec.whatwg.org/multipage/#dom-img-referrerpolicy
    /// @brief Sets the `referrerpolicy` attribute of the image.
    /// Functional Utility: Implements the `referrerPolicy` setter, ensuring that
    /// changes trigger an image data update only if the policy effectively changes.
    ///
    /// @param value The `DOMString` representing the new referrer policy.
    /// @param can_gc A `CanGc` token.
    fn SetReferrerPolicy(&self, value: DOMString, can_gc: CanGc) {
        let referrerpolicy_attr_name = local_name!("referrerpolicy"); // Attribute name.
        let element = self.upcast::<Element>(); // Get element.
        let previous_correct_attribute_value = get_correct_referrerpolicy_from_raw_token(
            &element.get_string_attribute(&referrerpolicy_attr_name), // Get previous value.
        );
        let correct_value_or_empty_string = get_correct_referrerpolicy_from_raw_token(&value); // Get new value.
        // Block Logic: Only update attribute if the effective policy changes.
        if previous_correct_attribute_value != correct_value_or_empty_string {
            // Setting the attribute to the same value will update the image.
            // We don't want to start an update if referrerpolicy is set to the same value.
            element.set_string_attribute(
                &referrerpolicy_attr_name,
                correct_value_or_empty_string,
                can_gc,
            );
        }
    }

    /// <https://html.spec.whatwg.org/multipage/#dom-img-decode>
    /// @brief Returns a `Promise` that resolves when the image has decoded.
    /// Functional Utility: Implements the `decode()` method, providing an asynchronous
    /// way to check if an image has successfully decoded its data.
    ///
    /// @param can_gc A `CanGc` token.
    /// @return An `Rc<Promise>`.
    fn Decode(&self, can_gc: CanGc) -> Rc<Promise> {
        // Step 1: Let promise be a new Promise.
        let promise = Promise::new(&self.global(), can_gc);

        // Step 2: Queue an ImageElementMicrotask::Decode task.
        let task = ImageElementMicrotask::Decode {
            elem: DomRoot::from_ref(self), // Reference to self.
            promise: promise.clone(), // Clone promise.
        };
        ScriptThread::await_stable_state(Microtask::ImageElement(task)); // Await stable state.

        // Step 3: Return promise.
        promise
    }

    // https://html.spec.whatwg.org/multipage/#dom-img-name
    /// @brief Returns the `name` attribute of the image.
    /// Functional Utility: Implements the `name` getter.
    make_getter!(Name, "name");

    // https://html.spec.whatwg.org/multipage/#dom-img-name
    /// @brief Sets the `name` attribute of the image.
    /// Functional Utility: Implements the `name` setter.
    make_atomic_setter!(SetName, "name");

    // https://html.spec.whatwg.org/multipage/#dom-img-align
    /// @brief Returns the `align` attribute of the image.
    /// Functional Utility: Implements the `align` getter (deprecated).
    make_getter!(Align, "align");

    // https://html.spec.whatwg.org/multipage/#dom-img-align
    /// @brief Sets the `align` attribute of the image.
    /// Functional Utility: Implements the `align` setter (deprecated).
    make_setter!(SetAlign, "align");

    // https://html.spec.whatwg.org/multipage/#dom-img-hspace
    /// @brief Returns the `hspace` attribute of the image.
    /// Functional Utility: Implements the `hspace` getter (deprecated),
    /// providing horizontal space around the image.
    make_uint_getter!(Hspace, "hspace");

    // https://html.spec.whatwg.org/multipage/#dom-img-hspace
    /// @brief Sets the `hspace` attribute of the image.
    /// Functional Utility: Implements the `hspace` setter (deprecated).
    make_uint_setter!(SetHspace, "hspace");

    // https://html.spec.whatwg.org/multipage/#dom-img-vspace
    /// @brief Returns the `vspace` attribute of the image.
    /// Functional Utility: Implements the `vspace` getter (deprecated),
    /// providing vertical space around the image.
    make_uint_getter!(Vspace, "vspace");

    // https://html.spec.whatwg.org/multipage/#dom-img-vspace
    /// @brief Sets the `vspace` attribute of the image.
    /// Functional Utility: Implements the `vspace` setter (deprecated).
    make_uint_setter!(SetVspace, "vspace");

    // https://html.spec.whatwg.org/multipage/#dom-img-longdesc
    /// @brief Returns the `longdesc` attribute of the image.
    /// Functional Utility: Implements the `longDesc` getter (deprecated),
    /// providing a URL to a long description of the image.
    make_getter!(LongDesc, "longdesc");

    // https://html.spec.whatwg.org/multipage/#dom-img-longdesc
    /// @brief Sets the `longdesc` attribute of the image.
    /// Functional Utility: Implements the `longDesc` setter (deprecated).
    make_setter!(SetLongDesc, "longdesc");

    // https://html.spec.whatwg.org/multipage/#dom-img-border
    /// @brief Returns the `border` attribute of the image.
    /// Functional Utility: Implements the `border` getter (deprecated),
    /// providing the width of the image border.
    make_getter!(Border, "border");

    // https://html.spec.whatwg.org/multipage/#dom-img-border
    /// @brief Sets the `border` attribute of the image.
    /// Functional Utility: Implements the `border` setter (deprecated).
    make_setter!(SetBorder, "border");
}

impl VirtualMethods for HTMLImageElement {
    /// @brief Returns the `VirtualMethods` implementation of the super type (`HTMLElement`).
    /// Functional Utility: Enables method overriding and calls to the superclass's implementations.
    /// @return An `Option` containing a reference to the super type's `VirtualMethods`.
    fn super_type(&self) -> Option<&dyn VirtualMethods> {
        Some(self.upcast::<HTMLElement>() as &dyn VirtualMethods) // Upcast to HTMLElement and get its VirtualMethods.
    }

    /// @brief Handles adoption of the image element into a new document.
    /// Functional Utility: Triggers an image data update when the element is moved
    /// to a different document.
    ///
    /// @param old_doc The `Document` the element was previously in.
    /// @param can_gc A `CanGc` token.
    fn adopting_steps(&self, old_doc: &Document, can_gc: CanGc) {
        self.super_type().unwrap().adopting_steps(old_doc, can_gc); // Call super type's method.
        self.update_the_image_data(can_gc); // Update image data.
    }

    /// @brief Handles attribute mutations for the `HTMLImageElement`.
    /// Functional Utility: Responds to changes in image-related attributes (`src`, `srcset`, etc.)
    /// by triggering an update of the image data.
    ///
    /// @param attr The `Attr` that was mutated.
    /// @param mutation The type of `AttributeMutation` that occurred.
    /// @param can_gc A `CanGc` token.
    fn attribute_mutated(&self, attr: &Attr, mutation: AttributeMutation, can_gc: CanGc) {
        self.super_type()
            .unwrap()
            .attribute_mutated(attr, mutation, can_gc); // Call super type's method.
        match attr.local_name() {
            &local_name!("src") |
            &local_name!("srcset") |
            &local_name!("width") |
            &local_name!("crossorigin") |
            &local_name!("sizes") |
            &local_name!("referrerpolicy") => self.update_the_image_data(can_gc), // Trigger image data update.
            _ => {}, // Other attribute mutations are ignored.
        }
    }

    /// @brief Parses a plain attribute value based on its name.
    /// Functional Utility: Overrides the default attribute parsing for `width`, `height`,
    /// `hspace`, and `vspace` attributes.
    ///
    /// @param name The `LocalName` of the attribute.
    /// @param value The `DOMString` value of the attribute.
    /// @return An `AttrValue` representing the parsed attribute value.
    fn parse_plain_attribute(&self, name: &LocalName, value: DOMString) -> AttrValue {
        match name {
            &local_name!("width") | &local_name!("height") => {
                AttrValue::from_dimension(value.into()) // Parse as dimension.
            },
            &local_name!("hspace") | &local_name!("vspace") => AttrValue::from_u32(value.into(), 0), // Parse as u32.
            _ => self
                .super_type()
                .unwrap()
                .parse_plain_attribute(name, value), // Delegate to super type's method.
        }
    }

    /// @brief Handles events for the image element.
    /// Functional Utility: Specifically handles `click` events to detect and activate
    /// `<area>` elements within client-side image maps.
    ///
    /// @param event The `Event` to handle.
    /// @param can_gc A `CanGc` token.
    fn handle_event(&self, event: &Event, can_gc: CanGc) {
        // Block Logic: Only process "click" events.
        if event.type_() != atom!("click") {
            return;
        }

        let area_elements = self.areas(); // Get associated area elements.
        let elements = match area_elements {
            Some(x) => x,
            None => return, // No area elements, so return.
        };

        // Fetch click coordinates
        let mouse_event = match event.downcast::<MouseEvent>() {
            Some(x) => x,
            None => return, // Not a mouse event, so return.
        };

        let point = Point2D::new(
            mouse_event.ClientX().to_f32().unwrap(), // Client X coordinate.
            mouse_event.ClientY().to_f32().unwrap(), // Client Y coordinate.
        );
        let bcr = self.upcast::<Element>().GetBoundingClientRect(can_gc); // Get bounding client rectangle.
        let bcr_p = Point2D::new(bcr.X() as f32, bcr.Y() as f32); // Bounding client rectangle origin.

        // Walk HTMLAreaElements
        for element in elements {
            let shape = element.get_shape_from_coords(); // Get shape from coordinates.
            let shp = match shape {
                Some(x) => x.absolute_coords(bcr_p), // Get absolute coordinates of shape.
                None => return, // No shape, so return.
            };
            // Block Logic: If the click point hits the area shape, activate it.
            if shp.hit_test(&point) {
                element.activation_behavior(event, self.upcast(), can_gc); // Activate area element.
                return;
            }
        }
    }

    /// @brief Binds the `HTMLImageElement` to the DOM tree.
    /// Functional Utility: Performs initialization tasks when the element is inserted
    /// into the document, including registering for responsive image updates.
    ///
    /// @param context The `BindContext` for the binding operation.
    /// @param can_gc A `CanGc` token.
    fn bind_to_tree(&self, context: &BindContext, can_gc: CanGc) {
        if let Some(s) = self.super_type() {
            s.bind_to_tree(context, can_gc); // Call super type's method.
        }
        let document = self.owner_document(); // Get owning document.
        // Block Logic: If connected to the tree, register for responsive image updates.
        if context.tree_connected {
            document.register_responsive_image(self); // Register responsive image.
        }

        // The element is inserted into a picture parent element
        // https://html.spec.whatwg.org/multipage/#relevant-mutations
        if let Some(parent) = self.upcast::<Node>().GetParentElement() {
            if parent.is::<HTMLPictureElement>() {
                self.update_the_image_data(can_gc); // Update image data if parent is `<picture>`.
            }
        }
    }

    /// @brief Unbinds the `HTMLImageElement` from the DOM tree.
    /// Functional Utility: Performs cleanup tasks when the element is removed
    /// from the document, including unregistering for responsive image updates.
    ///
    /// @param context The `UnbindContext` for the unbinding operation.
    /// @param can_gc A `CanGc` token.
    fn unbind_from_tree(&self, context: &UnbindContext, can_gc: CanGc) {
        self.super_type().unwrap().unbind_from_tree(context, can_gc); // Call super type's method.
        let document = self.owner_document(); // Get owning document.
        document.unregister_responsive_image(self); // Unregister responsive image.

        // The element is removed from a picture parent element
        // https://html.spec.whatwg.org/multipage/#relevant-mutations
        if context.parent.is::<HTMLPictureElement>() {
            self.update_the_image_data(can_gc); // Update image data if parent was `<picture>`.
        }
    }
}

impl FormControl for HTMLImageElement {
    /// @brief Returns the form owner of the element.
    /// Functional Utility: Implements the `form_owner` method for `FormControl` trait,
    /// providing access to the `HTMLFormElement` this image (acting as a submit button)
    /// is associated with.
    /// @return An `Option<DomRoot<HTMLFormElement>>` containing the form owner, or `None`.
    fn form_owner(&self) -> Option<DomRoot<HTMLFormElement>> {
        self.form_owner.get() // Retrieve the cached form owner.
    }

    /// @brief Sets the form owner of the element.
    /// Functional Utility: Implements the `set_form_owner` method for `FormControl` trait,
    /// updating the association of this image with a given `HTMLFormElement`.
    /// @param form An `Option<&HTMLFormElement>` for the new form owner.
    fn set_form_owner(&self, form: Option<&HTMLFormElement>) {
        self.form_owner.set(form); // Set the form owner.
    }

    /// @brief Returns a reference to the underlying `Element`.
    /// Functional Utility: Provides access to the generic `Element` methods and properties.
    /// @return A reference to the `Element`.
    fn to_element(&self) -> &Element {
        self.upcast::<Element>() // Upcast to Element.
    }

    /// @brief Checks if the form control is "listed".
    /// Functional Utility: Implements the `is_listed` method for `FormControl` trait,
    /// specifying that image elements are generally not listed (except when acting as submit buttons).
    /// @return `false` as image elements are not listed by default.
    fn is_listed(&self) -> bool {
        false
    }
}

/// @brief Helper function to set image dimensions (width/height).
/// Functional Utility: Provides a consistent way to set `width` or `height` attributes
/// for image elements, handling IDL to CSS dimension conversion and overflow.
///
/// @param element The `Element` to modify.
/// @param attr The `LocalName` of the attribute (e.g., "width", "height").
/// @param value The `u32` dimension value.
/// @param can_gc A `CanGc` token.
fn image_dimension_setter(element: &Element, attr: LocalName, value: u32, can_gc: CanGc) {
    // This setter is a bit weird: the IDL type is unsigned long, but it's parsed as
    // a dimension for rendering.
    let value = if value > UNSIGNED_LONG_MAX { 0 } else { value }; // Clamp value to unsigned long max.

    // FIXME: There are probably quite a few more cases of this. This is the
    // only overflow that was hitting on automation, but we should consider what
    // to do in the general case case.
    //
    // See <https://github.com/servo/app_units/issues/22>
    let pixel_value = if value > (i32::MAX / AU_PER_PX) as u32 {
        0
    } else {
        value
    };

    let dim = LengthOrPercentageOrAuto::Length(Au::from_px(pixel_value as i32)); // Convert to `Au`.
    let value = AttrValue::Dimension(value.to_string(), dim); // Create `AttrValue`.
    element.set_attribute(&attr, value, can_gc); // Set the attribute.
}

/// @brief Collects a sequence of characters from a string based on a predicate.
/// Functional Utility: Helper function for parsing, extracting a prefix of a string
/// that satisfies a given condition.
///
/// @param s The input string slice.
/// @param predicate A closure that returns `true` if a character should be collected.
/// @return A tuple `(&str, &str)` representing the collected prefix and the remaining string.
pub(crate) fn collect_sequence_characters(
    s: &str,
    mut predicate: impl FnMut(&char) -> bool,
) -> (&str, &str) {
    let i = s.find(|ch| !predicate(&ch)).unwrap_or(s.len()); // Find first character that doesn't satisfy predicate.
    (&s[0..i], &s[i..]) // Return split string.
}

/// @brief Parses an `srcset` attribute.
/// Functional Utility: Implements the HTML specification's algorithm for parsing the
/// `srcset` attribute, extracting image URLs and their associated descriptors.
///
/// <https://html.spec.whatwg.org/multipage/#parsing-a-srcset-attribute>.
///
/// @param input The `&str` value of the `srcset` attribute.
/// @return A `Vec<ImageSource>` containing the parsed image sources.
pub fn parse_a_srcset_attribute(input: &str) -> Vec<ImageSource> {
    // > 1. Let input be the value passed to this algorithm.
    // > 2. Let position be a pointer into input, initially pointing at the start of the string.
    let mut current_index = 0;

    // > 3. Let candidates be an initially empty source set.
    let mut candidates = vec![];
    while current_index < input.len() {
        let remaining_string = &input[current_index..]; // Remaining part of the string.

        // > 4. Splitting loop: Collect a sequence of code points that are ASCII whitespace or
        // > U+002C COMMA characters from input given position. If any U+002C COMMA
        // > characters were collected, that is a parse error.
        let mut collected_comma = false; // Flag to detect comma.
        let (collected_characters, string_after_whitespace) =
            collect_sequence_characters(remaining_string, |character| {
                if *character == ',' {
                    collected_comma = true; // Mark if comma found.
                }
                *character == ',' || character.is_ascii_whitespace() // Collect whitespace or comma.
            });
        if collected_comma {
            return Vec::new(); // Parse error if comma collected in this step.
        }

        // Add the length of collected whitespace, to find the start of the URL we are going
        // to parse.
        current_index += collected_characters.len();

        // > 5. If position is past the end of input, return candidates.
        if string_after_whitespace.is_empty() {
            return candidates;
        }

        // 6. Collect a sequence of code points that are not ASCII whitespace from input
        // given position, and let that be url.
        let (url, _) =
            collect_sequence_characters(string_after_whitespace, |c| !char::is_ascii_whitespace(c));

        // Add the length of `url` that we will parse to advance the index of the next part
        // of the string to prase.
        current_index += url.len();

        // 7. Let descriptors be a new empty list.
        let mut descriptors = Vec::new();

        // > 8. If url ends with U+002C (,), then:
        // >    1. Remove all trailing U+002C COMMA characters from url. If this removed
        // >       more than one character, that is a parse error.
        if url.ends_with(',') {
            let image_source = ImageSource {
                url: url.trim_end_matches(',').into(), // Trim trailing commas.
                descriptor: Descriptor {
                    width: None,
                    density: None,
                },
            };
            candidates.push(image_source); // Add as candidate.
            continue;
        }

        // Otherwise:
        // > 8.1. Descriptor tokenizer: Skip ASCII whitespace within input given position.
        let descriptors_string = &input[current_index..]; // Remaining string for descriptors.
        let (spaces, descriptors_string) =
            collect_sequence_characters(descriptors_string, |character| {
                character.is_ascii_whitespace()
            });
        current_index += spaces.len();

        // > 8.2. Let current descriptor be the empty string.
        let mut current_descriptor = String::new();

        // > 8.3. Let state be "in descriptor".
        let mut state = ParseState::InDescriptor;

        // > 8.4. Let c be the character at position. Do the following depending on the value of
        // > state. For the purpose of this step, "EOF" is a special character representing
        // > that position is past the end of input.
        let mut characters = descriptors_string.chars();
        let mut character = characters.next();
        if let Some(character) = character {
            current_index += character.len_utf8(); // Advance index.
        }

        loop {
            match (state, character) {
                (ParseState::InDescriptor, Some(character)) if character.is_ascii_whitespace() => {
                    // > If current descriptor is not empty, append current descriptor to
                    // > descriptors and let current descriptor be the empty string. Set
                    // > state to after descriptor.
                    if !current_descriptor.is_empty() {
                        descriptors.push(current_descriptor); // Push descriptor.
                        current_descriptor = String::new(); // Reset.
                        state = ParseState::AfterDescriptor; // Change state.
                    }
                },
                (ParseState::InDescriptor, Some(',')) => {
                    // > Advance position to the next character in input. If current descriptor
                    // > is not empty, append current descriptor to descriptors. Jump to the
                    // > step labeled descriptor parser.
                    if !current_descriptor.is_empty() {
                        descriptors.push(current_descriptor); // Push descriptor.
                    }
                    break; // Exit loop.
                },
                (ParseState::InDescriptor, Some('(')) => {
                    // > Append c to current descriptor. Set state to in parens.
                    current_descriptor.push('('); // Append character.
                    state = ParseState::InParens; // Change state.
                },
                (ParseState::InDescriptor, Some(character)) => {
                    // > Append c to current descriptor.
                    current_descriptor.push(character); // Append character.
                },
                (ParseState::InDescriptor, None) => {
                    // > If current descriptor is not empty, append current descriptor to
                    // > descriptors. Jump to the step labeled descriptor parser.
                    if !current_descriptor.is_empty() {
                        descriptors.push(current_descriptor); // Push descriptor.
                    }
                    break; // Exit loop.
                },
                (ParseState::InParens, Some(')')) => {
                    // > Append c to current descriptor. Set state to in descriptor.
                    current_descriptor.push(')'); // Append character.
                    state = ParseState::InDescriptor; // Change state.
                },
                (ParseState::InParens, Some(character)) => {
                    // Append c to current descriptor.
                    current_descriptor.push(character); // Append character.
                },
                (ParseState::InParens, None) => {
                    // > Append current descriptor to descriptors. Jump to the step
                    // > labeled descriptor parser.
                    descriptors.push(current_descriptor); // Push descriptor.
                    break; // Exit loop.
                },
                (ParseState::AfterDescriptor, Some(character))
                    if character.is_ascii_whitespace() =>
                {
                    // > Stay in this state.
                },
                (ParseState::AfterDescriptor, Some(_)) => {
                    // > Set state to in descriptor. Set position to the previous
                    // > character in input.
                    state = ParseState::InDescriptor; // Change state.
                    continue; // Continue parsing.
                },
                (ParseState::AfterDescriptor, None) => {
                    // > Jump to the step labeled descriptor parser.
                    break; // Exit loop.
                },
            }

            character = characters.next(); // Get next character.
            if let Some(character) = character {
                current_index += character.len_utf8(); // Advance index.
            }
        }

        // > 9. Descriptor parser: Let error be no.
        let mut error = false;
        // > 10. Let width be absent.
        let mut width: Option<u32> = None;
        // > 11. Let density be absent.
        let mut density: Option<f64> = None;
        // > 12. Let future-compat-h be absent.
        let mut future_compat_h: Option<u32> = None;

        // > 13. For each descriptor in descriptors, run the appropriate set of steps from
        // > the following list:
        for descriptor in descriptors.into_iter() {
            let Some(last_character) = descriptor.chars().last() else {
                break;
            };

            let first_part_of_string = &descriptor[0..descriptor.len() - last_character.len_utf8()];
            match last_character {
                // > If the descriptor consists of a valid non-negative integer followed by a
                // > U+0077 LATIN SMALL LETTER W character
                // > 1. If the user agent does not support the sizes attribute, let error be yes.
                // > 2. If width and density are not both absent, then let error be yes.
                // > 3. Apply the rules for parsing non-negative integers to the descriptor.
                // >    If the result is 0, let error be yes. Otherwise, let width be the result.
                'w' if density.is_none() && width.is_none() => {
                    match parse_integer(first_part_of_string.chars()) {
                        Ok(number) if number > 0 => {
                            width = Some(number as u32);
                            continue;
                        },
                        _ => error = true,
                    }
                },

                // > If the descriptor consists of a valid floating-point number followed by a
                // > U+0078 LATIN SMALL LETTER X character
                // > 1. If width, density and future-compat-h are not all absent, then let
                // >    error be yes.
                // > 2. Apply the rules for parsing floating-point number values to the
                // >    descriptor. If the result is less than 0, let error be yes. Otherwise, let
                // >    density be the result.
                //
                // The HTML specification has a procedure for parsing floats that is different enough from
                // the one that stylo uses, that it's better to use Rust's float parser here. This is
                // what Gecko does, but it also checks to see if the number is a valid HTML-spec compliant
                // number first. Not doing that means that we might be parsing numbers that otherwise
                // wouldn't parse.
                // TODO: Do what Gecko does and first validate the number passed to the Rust float parser.
                'x' if width.is_none() && density.is_none() && future_compat_h.is_none() => {
                    match first_part_of_string.parse::<f64>() {
                        Ok(number) if number.is_normal() && number > 0. => {
                            density = Some(number);
                            continue;
                        },
                        _ => error = true,
                    }
                },

                // > If the descriptor consists of a valid non-negative integer followed by a
                // > U+0068 LATIN SMALL LETTER H character
                // >   This is a parse error.
                // > 1. If future-compat-h and density are not both absent, then let error be
                // >    yes.
                // > 2. Apply the rules for parsing non-negative integers to the descriptor.
                // >    If the result is 0, let error be yes. Otherwise, let future-compat-h be the
                // >    result.
                'h' if future_compat_h.is_none() && density.is_none() => {
                    match parse_integer(first_part_of_string.chars()) {
                        Ok(number) if number > 0 => {
                            future_compat_h = Some(number as u32);
                            continue;
                        },
                        _ => error = true,
                    }
                },

                // > Anything else
                // >  Let error be yes.
                _ => error = true,
            }

            if error {
                break; // Exit if error.
            }
        }

        // > 14. If future-compat-h is not absent and width is absent, let error be yes.
        if future_compat_h.is_some() && width.is_none() {
            error = true;
        }

        if !error {
            let image_source = ImageSource {
                url: url.into(), // URL.
                descriptor: Descriptor { width, density }, // Descriptor.
            };
            candidates.push(image_source); // Add to candidates.
        }
    }
    candidates // Return parsed candidates.
}

/// @enum ChangeType
/// @brief Represents the type of change that triggered an image update.
/// Functional Utility: Differentiates between changes originating from environment
/// (e.g., viewport resize) and changes directly on the image element (e.g., `src` attribute).
#[derive(Clone)]
enum ChangeType {
    Environment {
        selected_source: USVString, //!< The selected source URL.
        selected_pixel_density: f64, //!< The selected pixel density.
    },
    Element, //!< Change originated from the image element itself.
}