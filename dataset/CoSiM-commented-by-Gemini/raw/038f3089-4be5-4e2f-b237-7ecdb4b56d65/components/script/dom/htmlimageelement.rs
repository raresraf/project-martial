/*!
This module implements the `HTMLImageElement` interface, which represents an
`<img>` element in the DOM. It is responsible for fetching, decoding, and
displaying images.

The `HTMLImageElement` struct is the main entry point for this module. It
creates a new `HTMLImageElement` object, which can be inserted into the DOM.
The `HTMLImageElement` object then handles the loading and displaying of the
image specified by its `src` attribute.

The `ImageRequest` struct holds the state for an image request. It is
responsible for tracking the state of the image request, from pending to
complete or broken.

The `ImageContext` struct is used to asynchronously load an external image. It
implements the `FetchResponseListener` trait, which is used to process the
response from the network layer.

The `HTMLImageElement` API is defined in the HTML specification:
<https://html.spec.whatwg.org/multipage/embedded-content.html#the-img-element>
*/

use std::cell::Cell;
use std::collections::HashSet;
use std::default::Default;
use std::rc::Rc;
use std::sync::Arc;
use std::{char, mem};

use app_units::{AU_PER_PX, Au};
use content_security_policy as csp;
use cssparser::{Parser, ParserInput};
use dom_struct::dom_struct;
use euclid::Point2D;
use html5ever::{LocalName, Prefix, QualName, local_name, ns};
use js::jsapi::JSAutoRealm;
use js::rust::HandleObject;
use mime::{self, Mime};
use net_traits::http_status::HttpStatus;
use net_traits::image_cache::{
    Image, ImageCache, ImageCacheResult, ImageLoadListener, ImageOrMetadataAvailable,
    ImageResponse, PendingImageId, UsePlaceholder,
};
use net_traits::request::{Destination, Initiator, RequestId};
use net_traits::{
    FetchMetadata, FetchResponseListener, FetchResponseMsg, NetworkError, ReferrerPolicy,
    ResourceFetchTiming, ResourceTimingType,
};
use num_traits::ToPrimitive;
use pixels::{CorsStatus, ImageMetadata};
use servo_url::ServoUrl;
use servo_url::origin::MutableOrigin;
use style::attr::{AttrValue, LengthOrPercentageOrAuto, parse_integer, parse_length};
use style::context::QuirksMode;
use style::parser::ParserContext;
use style::stylesheets::{CssRuleType, Origin};
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
use crate::dom::csp::report_csp_violations;
use crate::dom::document::{Document, determine_policy_for_token};
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

/// The state of parsing an `srcset` attribute.
#[derive(Clone, Copy, Debug)]
enum ParseState {
    /// In a descriptor.
    InDescriptor,
    /// In parentheses.
    InParens,
    /// After a descriptor.
    AfterDescriptor,
}

/// A source set for an image.
#[derive(MallocSizeOf)]
pub(crate) struct SourceSet {
    /// The image sources in the source set.
    image_sources: Vec<ImageSource>,
    /// The source size list for the source set.
    source_size: SourceSizeList,
}

impl SourceSet {
    /// Creates a new `SourceSet`.
    fn new() -> SourceSet {
        SourceSet {
            image_sources: Vec::new(),
            source_size: SourceSizeList::empty(),
        }
    }
}

/// An image source in a source set.
#[derive(Clone, Debug, MallocSizeOf, PartialEq)]
pub struct ImageSource {
    /// The URL of the image.
    pub url: String,
    /// The descriptor for the image.
    pub descriptor: Descriptor,
}

/// A descriptor for an image source.
#[derive(Clone, Debug, MallocSizeOf, PartialEq)]
pub struct Descriptor {
    /// The width of the image.
    pub width: Option<u32>,
    /// The density of the image.
    pub density: Option<f64>,
}

/// The state of an image request.
#[derive(Clone, Copy, JSTraceable, MallocSizeOf)]
#[allow(dead_code)]
enum State {
    /// The image is unavailable.
    Unavailable,
    /// The image is partially available.
    PartiallyAvailable,
    /// The image is completely available.
    CompletelyAvailable,
    /// The image is broken.
    Broken,
}

/// The phase of an image request.
#[derive(Clone, Copy, JSTraceable, MallocSizeOf)]
enum ImageRequestPhase {
    /// The request is pending.
    Pending,
    /// The request is current.
    Current,
}
/// An image request.
#[derive(JSTraceable, MallocSizeOf)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
struct ImageRequest {
    state: State,
    #[no_trace]
    parsed_url: Option<ServoUrl>,
    source_url: Option<USVString>,
    blocker: DomRefCell<Option<LoadBlocker>>,
    #[no_trace]
    image: Option<Image>,
    #[no_trace]
    metadata: Option<ImageMetadata>,
    #[no_trace]
    final_url: Option<ServoUrl>,
    current_pixel_density: Option<f64>,
}
/// The `HTMLImageElement` struct.
#[dom_struct]
pub(crate) struct HTMLImageElement {
    htmlelement: HTMLElement,
    image_request: Cell<ImageRequestPhase>,
    current_request: DomRefCell<ImageRequest>,
    pending_request: DomRefCell<ImageRequest>,
    form_owner: MutNullableDom<HTMLFormElement>,
    generation: Cell<u32>,
    source_set: DomRefCell<SourceSet>,
    last_selected_source: DomRefCell<Option<USVString>>,
    #[ignore_malloc_size_of = "promises are hard"]
    image_decode_promises: DomRefCell<Vec<Rc<Promise>>>,
}

impl HTMLImageElement {
    /// Returns the URL of the image.
    pub(crate) fn get_url(&self) -> Option<ServoUrl> {
        self.current_request.borrow().parsed_url.clone()
    }
    // https://html.spec.whatwg.org/multipage/#check-the-usability-of-the-image-argument
    pub(crate) fn is_usable(&self) -> Fallible<bool> {
        // If image has an intrinsic width or intrinsic height (or both) equal to zero, then return bad.
        if let Some(image) = &self.current_request.borrow().image {
            let intrinsic_size = image.metadata();
            if intrinsic_size.width == 0 || intrinsic_size.height == 0 {
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

    /// Returns the image data.
    pub(crate) fn image_data(&self) -> Option<Image> {
        self.current_request.borrow().image.clone()
    }
}

/// The context required for asynchronously loading an external image.
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
    fn process_request_body(&mut self, _: RequestId) {}
    fn process_request_eof(&mut self, _: RequestId) {}

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
        if let Some(metadata) = metadata.as_ref() {
            if let Some(ref content_type) = metadata.content_type {
                let mime: Mime = content_type.clone().into_inner().into();
                if mime.type_() == mime::MULTIPART && mime.subtype().as_str() == "x-mixed-replace" {
                    self.aborted = true;
                }
            }
        }

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

    fn process_response_chunk(&mut self, request_id: RequestId, payload: Vec<u8>) {
        if self.status.is_ok() {
            self.image_cache.notify_pending_response(
                self.id,
                FetchResponseMsg::ProcessResponseChunk(request_id, payload),
            );
        }
    }

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

    fn resource_timing_mut(&mut self) -> &mut ResourceFetchTiming {
        &mut self.resource_timing
    }

    fn resource_timing(&self) -> &ResourceFetchTiming {
        &self.resource_timing
    }

    fn submit_resource_timing(&mut self) {
        network_listener::submit_timing(self, CanGc::note())
    }

    fn process_csp_violations(&mut self, _request_id: RequestId, violations: Vec<csp::Violation>) {
        let global = &self.resource_timing_global();
        report_csp_violations(global, violations, None);
    }
}

impl ResourceTimingListener for ImageContext {
    fn resource_timing_information(&self) -> (InitiatorType, ServoUrl) {
        (
            InitiatorType::LocalName("img".to_string()),
            self.url.clone(),
        )
    }

    fn resource_timing_global(&self) -> DomRoot<GlobalScope> {
        self.doc.root().global()
    }
}

impl PreInvoke for ImageContext {
    fn should_invoke(&self) -> bool {
        !self.aborted
    }
}

#[allow(non_snake_case)]
impl HTMLImageElement {
    /// Update the current image with a valid URL.
    fn fetch_image(&self, img_url: &ServoUrl, can_gc: CanGc) {
        let window = self.owner_window();

        let cache_result = window.image_cache().get_cached_image_status(
            img_url.clone(),
            window.origin().immutable().clone(),
            cors_setting_for_element(self.upcast()),
            UsePlaceholder::Yes,
        );

        match cache_result {
            ImageCacheResult::Available(ImageOrMetadataAvailable::ImageAvailable {
                image,
                url,
                is_placeholder,
            }) => {
                if is_placeholder {
                    if let Some(raster_image) = image.as_raster_image() {
                        self.process_image_response(
                            ImageResponse::PlaceholderLoaded(raster_image, url),
                            can_gc,
                        )
                    }
                } else {
                    self.process_image_response(ImageResponse::Loaded(image, url), can_gc)
                }
            },
            ImageCacheResult::Available(ImageOrMetadataAvailable::MetadataAvailable(
                metadata,
                id,
            )) => {
                self.process_image_response(ImageResponse::MetadataLoaded(metadata), can_gc);
                self.register_image_cache_callback(id, ChangeType::Element);
            },
            ImageCacheResult::Pending(id) => {
                self.register_image_cache_callback(id, ChangeType::Element);
            },
            ImageCacheResult::ReadyForRequest(id) => {
                self.fetch_request(img_url, id);
                self.register_image_cache_callback(id, ChangeType::Element);
            },
            ImageCacheResult::LoadError => self.process_image_response(ImageResponse::None, can_gc),
        };
    }

    /// Registers an image cache callback.
    fn register_image_cache_callback(&self, id: PendingImageId, change_type: ChangeType) {
        let trusted_node = Trusted::new(self);
        let generation = self.generation_id();
        let window = self.owner_window();
        let sender = window.register_image_cache_listener(id, move |response| {
            let trusted_node = trusted_node.clone();
            let window = trusted_node.root().owner_window();
            let callback_type = change_type.clone();

            window
                .as_global_scope()
                .task_manager()
                .networking_task_source()
                .queue(task!(process_image_response: move || {
                let element = trusted_node.root();

                // Ignore any image response for a previous request that has been discarded.
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
            .add_listener(ImageLoadListener::new(sender, window.pipeline_id(), id));
    }

    /// Fetches an image request.
    fn fetch_request(&self, img_url: &ServoUrl, id: PendingImageId) {
        let document = self.owner_document();
        let window = self.owner_window();

        let context = ImageContext {
            image_cache: window.image_cache(),
            status: Ok(()),
            id,
            aborted: false,
            doc: Trusted::new(&document),
            resource_timing: ResourceFetchTiming::new(ResourceTimingType::Resource),
            url: img_url.clone(),
        };

        // https://html.spec.whatwg.org/multipage/#update-the-image-data steps 17-20
        // This function is also used to prefetch an image in `script::dom::servoparser::prefetch`.
        let global = document.global();
        let mut request = create_a_potential_cors_request(
            Some(window.webview_id()),
            img_url.clone(),
            Destination::Image,
            cors_setting_for_element(self.upcast()),
            None,
            global.get_referrer(),
            document.insecure_requests_policy(),
            document.has_trustworthy_ancestor_or_current_origin(),
            global.policy_container(),
        )
        .origin(document.origin().immutable().clone())
        .pipeline_id(Some(document.global().pipeline_id()))
        .referrer_policy(referrer_policy_for_element(self.upcast()));

        if Self::uses_srcset_or_picture(self.upcast()) {
            request = request.initiator(Initiator::ImageSet);
        }

        // This is a background load because the load blocker already fulfills the
        // purpose of delaying the document's load event.
        document.fetch_background(request, context);
    }

    // Steps common to when an image has been loaded.
    fn handle_loaded_image(&self, image: Image, url: ServoUrl, can_gc: CanGc) {
        self.current_request.borrow_mut().metadata = Some(image.metadata());
        self.current_request.borrow_mut().final_url = Some(url);
        self.current_request.borrow_mut().image = Some(image);
        self.current_request.borrow_mut().state = State::CompletelyAvailable;
        LoadBlocker::terminate(&self.current_request.borrow().blocker, can_gc);
        // Mark the node dirty
        self.upcast::<Node>().dirty(NodeDamage::Other);
        self.resolve_image_decode_promises(can_gc);
    }

    /// Step 24 of <https://html.spec.whatwg.org/multipage/#update-the-image-data>
    fn process_image_response(&self, image: ImageResponse, can_gc: CanGc) {
        // TODO: Handle multipart/x-mixed-replace
        let (trigger_image_load, trigger_image_error) = match (image, self.image_request.get()) {
            (ImageResponse::Loaded(image, url), ImageRequestPhase::Current) => {
                self.handle_loaded_image(image, url, can_gc);
                (true, false)
            },
            (ImageResponse::PlaceholderLoaded(image, url), ImageRequestPhase::Current) => {
                self.handle_loaded_image(Image::Raster(image), url, can_gc);
                (false, true)
            },
            (ImageResponse::Loaded(image, url), ImageRequestPhase::Pending) => {
                self.abort_request(State::Unavailable, ImageRequestPhase::Pending, can_gc);
                self.image_request.set(ImageRequestPhase::Current);
                self.handle_loaded_image(image, url, can_gc);
                (true, false)
            },
            (ImageResponse::PlaceholderLoaded(image, url), ImageRequestPhase::Pending) => {
                self.abort_request(State::Unavailable, ImageRequestPhase::Pending, can_gc);
                self.image_request.set(ImageRequestPhase::Current);
                self.handle_loaded_image(Image::Raster(image), url, can_gc);
                (false, true)
            },
            (ImageResponse::MetadataLoaded(meta), ImageRequestPhase::Current) => {
                self.current_request.borrow_mut().state = State::PartiallyAvailable;
                self.current_request.borrow_mut().metadata = Some(meta);
                (false, false)
            },
            (ImageResponse::MetadataLoaded(_), ImageRequestPhase::Pending) => {
                self.pending_request.borrow_mut().state = State::PartiallyAvailable;
                (false, false)
            },
            (ImageResponse::None, ImageRequestPhase::Current) => {
                self.abort_request(State::Broken, ImageRequestPhase::Current, can_gc);
                (false, true)
            },
            (ImageResponse::None, ImageRequestPhase::Pending) => {
                self.abort_request(State::Broken, ImageRequestPhase::Current, can_gc);
                self.abort_request(State::Broken, ImageRequestPhase::Pending, can_gc);
                self.image_request.set(ImageRequestPhase::Current);
                (false, true)
            },
        };

        // Fire image.onload and loadend
        if trigger_image_load {
            // TODO: https://html.spec.whatwg.org/multipage/#fire-a-progress-event-or-event
            self.upcast::<EventTarget>()
                .fire_event(atom!("load"), can_gc);
            self.upcast::<EventTarget>()
                .fire_event(atom!("loadend"), can_gc);
        }

        // Fire image.onerror
        if trigger_image_error {
            self.upcast::<EventTarget>()
                .fire_event(atom!("error"), can_gc);
            self.upcast::<EventTarget>()
                .fire_event(atom!("loadend"), can_gc);
        }

        self.upcast::<Node>().dirty(NodeDamage::Other);
    }

    /// Processes an image response for an environment change.
    fn process_image_response_for_environment_change(
        &self,
        image: ImageResponse,
        src: USVString,
        generation: u32,
        selected_pixel_density: f64,
        can_gc: CanGc,
    ) {
        match image {
            ImageResponse::Loaded(image, url) => {
                self.pending_request.borrow_mut().metadata = Some(image.metadata());
                self.pending_request.borrow_mut().final_url = Some(url);
                self.pending_request.borrow_mut().image = Some(image);
                self.finish_reacting_to_environment_change(src, generation, selected_pixel_density);
            },
            ImageResponse::PlaceholderLoaded(image, url) => {
                let image = Image::Raster(image);
                self.pending_request.borrow_mut().metadata = Some(image.metadata());
                self.pending_request.borrow_mut().final_url = Some(url);
                self.pending_request.borrow_mut().image = Some(image);
                self.finish_reacting_to_environment_change(src, generation, selected_pixel_density);
            },
            ImageResponse::MetadataLoaded(meta) => {
                self.pending_request.borrow_mut().metadata = Some(meta);
            },
            ImageResponse::None => {
                self.abort_request(State::Unavailable, ImageRequestPhase::Pending, can_gc);
            },
        };
    }

    /// <https://html.spec.whatwg.org/multipage/#abort-the-image-request>
    fn abort_request(&self, state: State, request: ImageRequestPhase, can_gc: CanGc) {
        let mut request = match request {
            ImageRequestPhase::Current => self.current_request.borrow_mut(),
            ImageRequestPhase::Pending => self.pending_request.borrow_mut(),
        };
        LoadBlocker::terminate(&request.blocker, can_gc);
        request.state = state;
        request.image = None;
        request.metadata = None;

        if matches!(state, State::Broken) {
            self.reject_image_decode_promises(can_gc);
        } else if matches!(state, State::CompletelyAvailable) {
            self.resolve_image_decode_promises(can_gc);
        }
    }

    /// <https://html.spec.whatwg.org/multipage/#update-the-source-set>
    fn update_source_set(&self) {
        // Step 1
        *self.source_set.borrow_mut() = SourceSet::new();

        // Step 2
        let elem = self.upcast::<Element>();
        let parent = elem.upcast::<Node>().GetParentElement();
        let nodes;
        let elements = match parent.as_ref() {
            Some(p) => {
                if p.is::<HTMLPictureElement>() {
                    nodes = p.upcast::<Node>().children();
                    nodes
                        .filter_map(DomRoot::downcast::<Element>)
                        .map(|n| DomRoot::from_ref(&*n))
                        .collect()
                } else {
                    vec![DomRoot::from_ref(elem)]
                }
            },
            None => vec![DomRoot::from_ref(elem)],
        };

        // Step 3
        let width = match elem.get_attribute(&ns!(), &local_name!("width")) {
            Some(x) => match parse_length(&x.value()) {
                LengthOrPercentageOrAuto::Length(x) => {
                    let abs_length = AbsoluteLength::Px(x.to_f32_px());
                    Some(Length::NoCalc(NoCalcLength::Absolute(abs_length)))
                },
                _ => None,
            },
            None => None,
        };

        // Step 4
        for element in &elements {
            // Step 4.1
            if *element == DomRoot::from_ref(elem) {
                let mut source_set = SourceSet::new();
                // Step 4.1.1
                if let Some(x) = element.get_attribute(&ns!(), &local_name!("srcset")) {
                    source_set.image_sources = parse_a_srcset_attribute(&x.value());
                }

                // Step 4.1.2
                if let Some(x) = element.get_attribute(&ns!(), &local_name!("sizes")) {
                    source_set.source_size =
                        parse_a_sizes_attribute(DOMString::from_string(x.value().to_string()));
                }

                // Step 4.1.3
                let src_attribute = element.get_string_attribute(&local_name!("src"));
                let is_src_empty = src_attribute.is_empty();
                let no_density_source_of_1 = source_set
                    .image_sources
                    .iter()
                    .all(|source| source.descriptor.density != Some(1.));
                let no_width_descriptor = source_set
                    .image_sources
                    .iter()
                    .all(|source| source.descriptor.width.is_none());
                if !is_src_empty && no_density_source_of_1 && no_width_descriptor {
                    source_set.image_sources.push(ImageSource {
                        url: src_attribute.to_string(),
                        descriptor: Descriptor {
                            width: None,
                            density: None,
                        },
                    })
                }

                // Step 4.1.4
                self.normalise_source_densities(&mut source_set, width);

                // Step 4.1.5
                *self.source_set.borrow_mut() = source_set;

                // Step 4.1.6
                return;
            }
            // Step 4.2
            if !element.is::<HTMLSourceElement>() {
                continue;
            }

            // Step 4.3 - 4.4
            let mut source_set = SourceSet::new();
            match element.get_attribute(&ns!(), &local_name!("srcset")) {
                Some(x) => {
                    source_set.image_sources = parse_a_srcset_attribute(&x.value());
                },
                _ => continue,
            }

            // Step 4.5
            if source_set.image_sources.is_empty() {
                continue;
            }

            // Step 4.6
            if let Some(x) = element.get_attribute(&ns!(), &local_name!("media")) {
                if !elem.matches_environment(&x.value()) {
                    continue;
                }
            }

            // Step 4.7
            if let Some(x) = element.get_attribute(&ns!(), &local_name!("sizes")) {
                source_set.source_size =
                    parse_a_sizes_attribute(DOMString::from_string(x.value().to_string()));
            }

            // Step 4.8
            if let Some(x) = element.get_attribute(&ns!(), &local_name!("type")) {
                // TODO Handle unsupported mime type
                let mime = x.value().parse::<Mime>();
                match mime {
                    Ok(m) => match m.type_() {
                        mime::IMAGE => (),
                        _ => continue,
                    },
                    _ => continue,
                }
            }

            // Step 4.9
            self.normalise_source_densities(&mut source_set, width);

            // Step 4.10
            *self.source_set.borrow_mut() = source_set;
            return;
        }
    }

    /// Evaluates a source size list.
    fn evaluate_source_size_list(
        &self,
        source_size_list: &mut SourceSizeList,
        _width: Option<Length>,
    ) -> Au {
        let document = self.owner_document();
        let quirks_mode = document.quirks_mode();
        let result = source_size_list.evaluate(document.window().layout().device(), quirks_mode);
        result
    }

    /// <https://html.spec.whatwg.org/multipage/#normalise-the-source-densities>
    fn normalise_source_densities(&self, source_set: &mut SourceSet, width: Option<Length>) {
        // Step 1
        let source_size = &mut source_set.source_size;

        // Find source_size_length for Step 2.2
        let source_size_length = self.evaluate_source_size_list(source_size, width);

        // Step 2
        for imgsource in &mut source_set.image_sources {
            // Step 2.1
            if imgsource.descriptor.density.is_some() {
                continue;
            }
            // Step 2.2
            if imgsource.descriptor.width.is_some() {
                let wid = imgsource.descriptor.width.unwrap();
                imgsource.descriptor.density = Some(wid as f64 / source_size_length.to_f64_px());
            } else {
                //Step 2.3
                imgsource.descriptor.density = Some(1_f64);
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/#select-an-image-source>
    fn select_image_source(&self) -> Option<(USVString, f64)> {
        // Step 1, 3
        self.update_source_set();
        let source_set = &*self.source_set.borrow_mut();
        let len = source_set.image_sources.len();

        // Step 2
        if len == 0 {
            return None;
        }

        // Step 4
        let mut repeat_indices = HashSet::new();
        for outer_index in 0..len {
            if repeat_indices.contains(&outer_index) {
                continue;
            }
            let imgsource = &source_set.image_sources[outer_index];
            let pixel_density = imgsource.descriptor.density.unwrap();
            for inner_index in (outer_index + 1)..len {
                let imgsource2 = &source_set.image_sources[inner_index];
                if pixel_density == imgsource2.descriptor.density.unwrap() {
                    repeat_indices.insert(inner_index);
                }
            }
        }

        let mut max = (0f64, 0);
        let img_sources = &mut vec![];
        for (index, image_source) in source_set.image_sources.iter().enumerate() {
            if repeat_indices.contains(&index) {
                continue;
            }
            let den = image_source.descriptor.density.unwrap();
            if max.0 < den {
                max = (den, img_sources.len());
            }
            img_sources.push(image_source);
        }

        // Step 5
        let mut best_candidate = max;
        let device_pixel_ratio = self
            .owner_document()
            .window()
            .viewport_details()
            .hidpi_scale_factor
            .get() as f64;
        for (index, image_source) in img_sources.iter().enumerate() {
            let current_den = image_source.descriptor.density.unwrap();
            if current_den < best_candidate.0 && current_den >= device_pixel_ratio {
                best_candidate = (current_den, index);
            }
        }
        let selected_source = img_sources.remove(best_candidate.1).clone();
        Some((
            USVString(selected_source.url),
            selected_source.descriptor.density.unwrap(),
        ))
    }

    /// Initializes an image request.
    fn init_image_request(
        &self,
        request: &mut RefMut<ImageRequest>,
        url: &ServoUrl,
        src: &USVString,
        can_gc: CanGc,
    ) {
        request.parsed_url = Some(url.clone());
        request.source_url = Some(src.clone());
        request.image = None;
        request.metadata = None;
        let document = self.owner_document();
        LoadBlocker::terminate(&request.blocker, can_gc);
        *request.blocker.borrow_mut() =
            Some(LoadBlocker::new(&document, LoadType::Image(url.clone())));
    }

    /// Step 13-17 of html.spec.whatwg.org/multipage/#update-the-image-data
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
                    // Step 13
                    if pending_url == *url {
                        return;
                    }
                }
            },
            ImageRequestPhase::Current => {
                let mut current_request = self.current_request.borrow_mut();
                let mut pending_request = self.pending_request.borrow_mut();
                // step 16, create a new "image_request"
                match (current_request.parsed_url.clone(), current_request.state) {
                    (Some(parsed_url), State::PartiallyAvailable) => {
                        // Step 14
                        if parsed_url == *url {
                            // Step 15 abort pending request
                            pending_request.image = None;
                            pending_request.parsed_url = None;
                            LoadBlocker::terminate(&pending_request.blocker, can_gc);
                            // TODO: queue a task to restart animation, if restart-animation is set
                            return;
                        }
                        pending_request.current_pixel_density = Some(selected_pixel_density);
                        self.image_request.set(ImageRequestPhase::Pending);
                        self.init_image_request(&mut pending_request, url, src, can_gc);
                    },
                    (_, State::Broken) | (_, State::Unavailable) => {
                        // Step 17
                        current_request.current_pixel_density = Some(selected_pixel_density);
                        self.init_image_request(&mut current_request, url, src, can_gc);
                    },
                    (_, _) => {
                        // step 17
                        pending_request.current_pixel_density = Some(selected_pixel_density);
                        self.image_request.set(ImageRequestPhase::Pending);
                        self.init_image_request(&mut pending_request, url, src, can_gc);
                    },
                }
            },
        }
        self.fetch_image(url, can_gc);
    }
}