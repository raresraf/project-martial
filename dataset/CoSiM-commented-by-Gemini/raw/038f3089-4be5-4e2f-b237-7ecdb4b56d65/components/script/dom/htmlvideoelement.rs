/*!
This module implements the `HTMLVideoElement` interface, which represents a
`<video>` element in the DOM. It extends the `HTMLMediaElement` interface and
provides additional properties and methods for controlling video playback.

The `HTMLVideoElement` struct is the main entry point for this module. It
creates a new `HTMLVideoElement` object, which can be inserted into the DOM.
The `HTMLVideoElement` object then handles the loading and playback of the
video resource specified by its `src` attribute.

The `HTMLVideoElement` API is defined in the HTML specification:
<https://html.spec.whatwg.org/multipage/media.html#the-video-element>
*/

use std::cell::Cell;
use std::sync::Arc;

use content_security_policy as csp;
use dom_struct::dom_struct;
use euclid::default::Size2D;
use html5ever::{LocalName, Prefix, local_name, ns};
use js::rust::HandleObject;
use layout_api::{HTMLMediaData, MediaMetadata};
use net_traits::image_cache::{
    ImageCache, ImageCacheResult, ImageLoadListener, ImageOrMetadataAvailable, ImageResponse,
    PendingImageId, UsePlaceholder,
};
use net_traits::request::{CredentialsMode, Destination, RequestBuilder, RequestId};
use net_traits::{
    FetchMetadata, FetchResponseListener, FetchResponseMsg, NetworkError, ReferrerPolicy,
    ResourceFetchTiming, ResourceTimingType,
};
use pixels::{Snapshot, SnapshotAlphaMode, SnapshotPixelFormat};
use servo_media::player::video::VideoFrame;
use servo_url::ServoUrl;
use style::attr::AttrValue;
use style::context::QuirksMode;
use style::parser::ParserContext;
use style::stylesheets::{CssRuleType, Origin};
use style::values::specified::length::{Length, LengthOrPercentageOrAuto, NoCalcLength};
use style::values::specified::source_size_list::SourceSizeList;
use style_traits::ParsingMode;
use url::Url;

use super::domexception::DOMErrorName;
use super::types::DOMException;
use crate::document_loader::{LoadBlocker, LoadType};
use crate::dom::attr::Attr;
use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::AttrBinding::AttrMethods;
use crate::dom::bindings::codegen::Bindings::DOMRectBinding::DOMRect_Binding::DOMRectMethods;
use crate::dom::bindings::codegen::Bindings::ElementBinding::Element_Binding::ElementMethods;
use crate::dom::bindings::codegen::Bindings::HTMLVideoElementBinding::HTMLVideoElementMethods;
use crate::dom::bindings::codegen::Bindings::MouseEventBinding::MouseEventMethods;
use crate::dom::bindings::codegen::Bindings::NodeBinding::Node_Binding::NodeMethods;
use crate::dom::bindings::codegen::Bindings::WindowBinding::WindowMethods;
use crate::dom::bindings::error::{Error, Fallible};
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::refcounted::Trusted;
use crate::dom::bindings::reflector::DomGlobal;
use crate::dom::bindings::root::{DomRoot, LayoutDom};
use crate::dom::bindings::str::DOMString;
use crate::dom::csp::report_csp_violations;
use crate::dom::document::Document;
use crate::dom::element::{
    AttributeMutation, Element, ElementCreator, LayoutElementHelpers,
    cors_setting_for_element, referrer_policy_for_element, reflect_cross_origin_attribute,
    reflect_referrer_policy_attribute, set_cross_origin_attribute,
};
use crate::dom::event::Event;
use crate::dom::eventtarget::EventTarget;
use crate::dom::globalscope::GlobalScope;
use crate::dom::htmlmediaelement::{HTMLMediaElement, NetworkState, ReadyState};
use crate::dom::node::{BindContext, Node, NodeDamage, NodeTraits, UnbindContext};
use crate::dom::performanceresourcetiming::InitiatorType;
use crate::dom::promise::Promise;
use crate::dom::values::UNSIGNED_LONG_MAX;
use crate::dom::virtualmethods::VirtualMethods;
use crate::fetch::{create_a_potential_cors_request, FetchCanceller};
use crate::network_listener::{self, PreInvoke, ResourceTimingListener};
use crate::realms::enter_realm;
use crate::script_runtime::CanGc;
use crate::task_source::{SendableTaskSource, TaskSourceName};
use crate::unminify::{create_output_file, create_temp_files, execute_js_beautify};

/// The `HTMLVideoElement` struct.
#[dom_struct]
pub(crate) struct HTMLVideoElement {
    htmlmediaelement: HTMLMediaElement,
    /// <https://html.spec.whatwg.org/multipage/#dom-video-videowidth>
    video_width: Cell<Option<u32>>,
    /// <https://html.spec.whatwg.org/multipage/#dom-video-videoheight>
    video_height: Cell<Option<u32>>,
    /// Incremented whenever tasks associated with this element are cancelled.
    generation_id: Cell<u32>,
    /// Load event blocker. Will block the load event while the poster frame
    /// is being fetched.
    load_blocker: DomRefCell<Option<LoadBlocker>>,
    /// A copy of the last frame
    #[ignore_malloc_size_of = "VideoFrame"]
    #[no_trace]
    last_frame: DomRefCell<Option<VideoFrame>>,
    /// Indicates if it has already sent a resize event for a given size
    sent_resize: Cell<Option<(u32, u32)>>,
}

impl HTMLVideoElement {
    fn new_inherited(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
    ) -> HTMLVideoElement {
        HTMLVideoElement {
            htmlmediaelement: HTMLMediaElement::new_inherited(local_name, prefix, document),
            video_width: Cell::new(None),
            video_height: Cell::new(None),
            generation_id: Cell::new(0),
            load_blocker: Default::default(),
            last_frame: Default::default(),
            sent_resize: Cell::new(None),
        }
    }

    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn new(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
        proto: Option<HandleObject>,
        can_gc: CanGc,
    ) -> DomRoot<HTMLVideoElement> {
        Node::reflect_node_with_proto(
            Box::new(HTMLVideoElement::new_inherited(
                local_name, prefix, document,
            )),
            document,
            proto,
            can_gc,
        )
    }

    /// Returns the width of the video.
    pub(crate) fn get_video_width(&self) -> Option<u32> {
        self.video_width.get()
    }

    /// Returns the height of the video.
    pub(crate) fn get_video_height(&self) -> Option<u32> {
        self.video_height.get()
    }

    /// <https://html.spec.whatwg.org/multipage#event-media-resize>
    pub(crate) fn resize(&self, width: Option<u32>, height: Option<u32>) -> Option<(u32, u32)> {
        self.video_width.set(width);
        self.video_height.set(height);

        let width = width?;
        let height = height?;
        if self.sent_resize.get() == Some((width, height)) {
            return None;
        }

        let sent_resize = if self.htmlmediaelement.get_ready_state() == ReadyState::HaveNothing {
            None
        } else {
            self.owner_global()
                .task_manager()
                .media_element_task_source()
                .queue_simple_event(self.upcast(), atom!("resize"));
            Some((width, height))
        };

        self.sent_resize.set(sent_resize);
        sent_resize
    }

    /// Gets the copy of the video frame at the current playback position,
    /// if that is available, or else (e.g. when the video is seeking or buffering)
    /// its previous appearance, if any.
    pub(crate) fn get_current_frame_data(&self) -> Option<Snapshot> {
        let frame = self.htmlmediaelement.get_current_frame();
        if frame.is_some() {
            *self.last_frame.borrow_mut() = frame;
        }

        match self.last_frame.borrow().as_ref() {
            Some(frame) => {
                let size = Size2D::new(frame.get_width() as u32, frame.get_height() as u32);
                if !frame.is_gl_texture() {
                    let alpha_mode = SnapshotAlphaMode::Transparent {
                        premultiplied: false,
                    };

                    Some(Snapshot::from_vec(
                        size.cast(),
                        SnapshotPixelFormat::BGRA,
                        alpha_mode,
                        frame.get_data().to_vec(),
                    ))
                } else {
                    // XXX(victor): here we only have the GL texture ID.
                    Some(Snapshot::cleared(size.cast()))
                }
            },
            None => None,
        }
    }

    /// <https://html.spec.whatwg.org/multipage/#poster-frame>
    fn update_poster_frame(&self, poster_url: Option<&str>, can_gc: CanGc) {
        // Step 1. If there is an existing instance of this algorithm running
        // for this video element, abort that instance of this algorithm without
        // changing the poster frame.
        self.generation_id.set(self.generation_id.get() + 1);

        // Step 2. If the poster attribute's value is the empty string or
        // if the attribute is absent, then there is no poster frame; return.
        let Some(poster_url) = poster_url.filter(|poster_url| !poster_url.is_empty()) else {
            self.htmlmediaelement.set_poster_frame(None);
            return;
        };

        // Step 3. Let url be the result of encoding-parsing a URL given
        // the poster attribute's value, relative to the element's node
        // document.
        // Step 4. If url is failure, then return. There is no poster frame.
        let poster_url = match self.owner_document().encoding_parse_a_url(poster_url) {
            Ok(url) => url,
            Err(_) => {
                self.htmlmediaelement.set_poster_frame(None);
                return;
            },
        };

        // We use the image cache for poster frames so we save as much
        // network activity as possible.
        let window = self.owner_window();
        let image_cache = window.image_cache();
        let cache_result = image_cache.get_cached_image_status(
            poster_url.clone(),
            window.origin().immutable().clone(),
            None,
            UsePlaceholder::No,
        );

        let id = match cache_result {
            ImageCacheResult::Available(ImageOrMetadataAvailable::ImageAvailable {
                image,
                url,
                ..
            }) => {
                self.process_image_response(ImageResponse::Loaded(image, url), can_gc);
                return;
            },
            ImageCacheResult::Available(ImageOrMetadataAvailable::MetadataAvailable(_, id)) => id,
            ImageCacheResult::ReadyForRequest(id) => {
                self.do_fetch_poster_frame(poster_url, id, can_gc);
                id
            },
            ImageCacheResult::LoadError => {
                self.process_image_response(ImageResponse::None, can_gc);
                return;
            },
            ImageCacheResult::Pending(id) => id,
        };

        let trusted_node = Trusted::new(self);
        let generation = self.generation_id();
        let sender = window.register_image_cache_listener(id, move |response| {
            let element = trusted_node.root();

            // Ignore any image response for a previous request that has been discarded.
            if generation != element.generation_id() {
                return;
            }
            element.process_image_response(response.response, CanGc::note());
        });

        image_cache.add_listener(ImageLoadListener::new(sender, window.pipeline_id(), id));
    }

    /// <https://html.spec.whatwg.org/multipage/#poster-frame>
    fn do_fetch_poster_frame(&self, poster_url: ServoUrl, id: PendingImageId, can_gc: CanGc) {
        // Step 5. Let request be a new request whose URL is url, client is the element's node
        // document's relevant settings object, destination is "image", initiator type is "video",
        // credentials mode is "include", and whose use-URL-credentials flag is set.
        let document = self.owner_document();
        let request = RequestBuilder::new(
            Some(document.webview_id()),
            poster_url.clone(),
            document.global().get_referrer(),
        )
        .destination(Destination::Image)
        .credentials_mode(CredentialsMode::Include)
        .use_url_credentials(true)
        .origin(document.origin().immutable().clone())
        .pipeline_id(Some(document.global().pipeline_id()))
        .insecure_requests_policy(document.insecure_requests_policy())
        .has_trustworthy_ancestor_origin(document.has_trustworthy_ancestor_origin())
        .policy_container(document.policy_container().to_owned());

        // Step 6. Fetch request. This must delay the load event of the element's node document.
        // This delay must be independent from the ones created by HTMLMediaElement during
        // its media load algorithm, otherwise a code like
        // <video poster="poster.png"></video>
        // (which triggers no media load algorithm unless a explicit call to .load() is done)
        // will block the document's load event forever.
        let blocker = &self.load_blocker;
        LoadBlocker::terminate(blocker, can_gc);
        *blocker.borrow_mut() = Some(LoadBlocker::new(
            &self.owner_document(),
            LoadType::Image(poster_url.clone()),
        ));

        let context = PosterFrameFetchContext::new(self, poster_url, id, request.id);
        self.owner_document().fetch_background(request, context);
    }

    /// Returns the generation ID.
    fn generation_id(&self) -> u32 {
        self.generation_id.get()
    }

    /// <https://html.spec.whatwg.org/multipage/#poster-frame>
    fn process_image_response(&self, response: ImageResponse, can_gc: CanGc) {
        // Step 7. If an image is thus obtained, the poster frame is that image.
        // Otherwise, there is no poster frame.
        match response {
            ImageResponse::Loaded(image, url) => {
                debug!("Loaded poster image for video element: {:?}", url);
                match image.as_raster_image() {
                    Some(image) => self.htmlmediaelement.set_poster_frame(Some(image)),
                    None => warn!("Vector images are not yet supported in video poster"),
                }
                LoadBlocker::terminate(&self.load_blocker, can_gc);
            },
            ImageResponse::MetadataLoaded(..) => {},
            // The image cache may have loaded a placeholder for an invalid poster url
            ImageResponse::PlaceholderLoaded(..) | ImageResponse::None => {
                self.htmlmediaelement.set_poster_frame(None);
                // A failed load should unblock the document load.
                LoadBlocker::terminate(&self.load_blocker, can_gc);
            },
        }
    }

    /// <https://html.spec.whatwg.org/multipage/#check-the-usability-of-the-image-argument>
    pub(crate) fn is_usable(&self) -> bool {
        !matches!(
            self.htmlmediaelement.get_ready_state(),
            ReadyState::HaveNothing | ReadyState::HaveMetadata
        )
    }

    /// Returns whether the origin is clean.
    pub(crate) fn origin_is_clean(&self) -> bool {
        self.htmlmediaelement.origin_is_clean()
    }

    /// Returns whether the network state is empty.
    pub(crate) fn is_network_state_empty(&self) -> bool {
        self.htmlmediaelement.network_state() == NetworkState::Empty
    }
}