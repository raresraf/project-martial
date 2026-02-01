/*!
This module implements the `HTMLMediaElement` interface, which is the base
interface for `<audio>` and `<video>` elements. It provides the common
functionality for both elements, such as playback controls, event handling,
and resource fetching.

The `HTMLMediaElement` struct is the main entry point for this module. It
creates a new `HTMLMediaElement` object, which can be inserted into the DOM.
The `HTMLMediaElement` object then handles the loading and playback of the
media resource specified by its `src` attribute.

The `MediaFrameRenderer` struct is used to render video frames to the
compositor. It implements the `VideoFrameRenderer` trait from the `media`
crate, which is used by the media backend to render video frames.

The `HTMLMediaElement` API is defined in the HTML specification:
<https://html.spec.whatwg.org/multipage/media.html#htmlmediaelement>
*/

use std::cell::Cell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{f64, mem};

use compositing_traits::{CrossProcessCompositorApi, ImageUpdate, SerializableImageData};
use content_security_policy as csp;
use dom_struct::dom_struct;
use embedder_traits::{MediaPositionState, MediaSessionEvent, MediaSessionPlaybackState};
use euclid::default::Size2D;
use headers::{ContentLength, ContentRange, HeaderMapExt};
use html5ever::{LocalName, Prefix, local_name, ns};
use http::StatusCode;
use http::header::{self, HeaderMap, HeaderValue};
use ipc_channel::ipc::{self, IpcSharedMemory, channel};
use ipc_channel::router::ROUTER;
use js::jsapi::JSAutoRealm;
use layout_api::MediaFrame;
use media::{GLPlayerMsg, GLPlayerMsgForward, WindowGLContext};
use net_traits::request::{Destination, RequestId};
use net_traits::{
    FetchMetadata, FetchResponseListener, FilteredMetadata, Metadata, NetworkError,
    ResourceFetchTiming, ResourceTimingType,
};
use pixels::RasterImage;
use script_bindings::codegen::GenericBindings::TimeRangesBinding::TimeRangesMethods;
use script_bindings::codegen::InheritTypes::{
    ElementTypeId, HTMLElementTypeId, HTMLMediaElementTypeId, NodeTypeId,
};
use servo_config::pref;
use servo_media::player::audio::AudioRenderer;
use servo_media::player::video::{VideoFrame, VideoFrameRenderer};
use servo_media::player::{PlaybackState, Player, PlayerError, PlayerEvent, SeekLock, StreamType};
use servo_media::{ClientContextId, ServoMedia, SupportsMediaType};
use servo_url::ServoUrl;
use webrender_api::{
    ExternalImageData, ExternalImageId, ExternalImageType, ImageBufferKind, ImageDescriptor,
    ImageDescriptorFlags, ImageFormat, ImageKey,
};

use crate::document_loader::{LoadBlocker, LoadType};
use crate::dom::attr::Attr;
use crate::dom::audiotrack::AudioTrack;
use crate::dom::audiotracklist::AudioTrackList;
use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::AttrBinding::AttrMethods;
use crate::dom::bindings::codegen::Bindings::HTMLMediaElementBinding::{
    CanPlayTypeResult, HTMLMediaElementConstants, HTMLMediaElementMethods,
};
use crate::dom::bindings::codegen::Bindings::MediaErrorBinding::MediaErrorConstants::*;
use crate::dom::bindings::codegen::Bindings::MediaErrorBinding::MediaErrorMethods;
use crate::dom::bindings::codegen::Bindings::NavigatorBinding::Navigator_Binding::NavigatorMethods;
use crate::dom::bindings::codegen::Bindings::NodeBinding::Node_Binding::NodeMethods;
use crate::dom::bindings::codegen::Bindings::ShadowRootBinding::{
    ShadowRootMode, SlotAssignmentMode,
};
use crate::dom::bindings::codegen::Bindings::TextTrackBinding::{TextTrackKind, TextTrackMode};
use crate::dom::bindings::codegen::Bindings::URLBinding::URLMethods;
use crate::dom::bindings::codegen::Bindings::WindowBinding::Window_Binding::WindowMethods;
use crate::dom::bindings::codegen::UnionTypes::{
    MediaStreamOrBlob, VideoTrackOrAudioTrackOrTextTrack,
};
use crate::dom::bindings::error::{Error, ErrorResult, Fallible};
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::num::Finite;
use crate::dom::bindings::refcounted::Trusted;
use crate::dom::bindings::reflector::DomGlobal;
use crate::dom::bindings::root::{Dom, DomRoot, MutNullableDom};
use crate::dom::bindings::str::{DOMString, USVString};
use crate::dom::blob::Blob;
use crate::dom::csp::report_csp_violations;
use crate::dom::document::Document;
use crate::dom::element::{
    AttributeMutation, Element, ElementCreator, cors_setting_for_element,
    reflect_cross_origin_attribute, set_cross_origin_attribute,
};
use crate::dom::event::Event;
use crate::dom::eventtarget::EventTarget;
use crate::dom::globalscope::GlobalScope;
use crate::dom::htmlelement::HTMLElement;
use crate::dom::htmlscriptelement::HTMLScriptElement;
use crate::dom::htmlsourceelement::HTMLSourceElement;
use crate::dom::htmlstyleelement::HTMLStyleElement;
use crate::dom::htmlvideoelement::HTMLVideoElement;
use crate::dom::mediaerror::MediaError;
use crate::dom::mediafragmentparser::MediaFragmentParser;
use crate::dom::mediastream::MediaStream;
use crate::dom::node::{Node, NodeDamage, NodeTraits, UnbindContext};
use crate::dom::performanceresourcetiming::InitiatorType;
use crate::dom::promise::Promise;
use crate::dom::shadowroot::IsUserAgentWidget;
use crate::dom::texttrack::TextTrack;
use crate::dom::texttracklist::TextTrackList;
use crate::dom::timeranges::{TimeRanges, TimeRangesContainer};
use crate::dom::trackevent::TrackEvent;
use crate::dom::url::URL;
use crate::dom::videotrack::VideoTrack;
use crate::dom::videotracklist::VideoTrackList;
use crate::dom::virtualmethods::VirtualMethods;
use crate::fetch::{FetchCanceller, create_a_potential_cors_request};
use crate::microtask::{Microtask, MicrotaskRunnable};
use crate::network_listener::{self, PreInvoke, ResourceTimingListener};
use crate::realms::{InRealm, enter_realm};
use crate::script_runtime::CanGc;
use crate::script_thread::ScriptThread;

/// A CSS file to style the media controls.
static MEDIA_CONTROL_CSS: &str = include_str!("../resources/media-controls.css");

/// A JS file to control the media controls.
static MEDIA_CONTROL_JS: &str = include_str!("../resources/media-controls.js");

/// The status of a video frame.
#[derive(PartialEq)]
enum FrameStatus {
    /// The frame is locked and cannot be modified.
    Locked,
    /// The frame is unlocked and can be modified.
    Unlocked,
}

/// A holder for a video frame.
struct FrameHolder(FrameStatus, VideoFrame);

impl FrameHolder {
    /// Creates a new `FrameHolder`.
    fn new(frame: VideoFrame) -> FrameHolder {
        FrameHolder(FrameStatus::Unlocked, frame)
    }

    /// Locks the frame.
    fn lock(&mut self) {
        if self.0 == FrameStatus::Unlocked {
            self.0 = FrameStatus::Locked;
        };
    }

    /// Unlocks the frame.
    fn unlock(&mut self) {
        if self.0 == FrameStatus::Locked {
            self.0 = FrameStatus::Unlocked;
        };
    }

    /// Sets the frame.
    fn set(&mut self, new_frame: VideoFrame) {
        if self.0 == FrameStatus::Unlocked {
            self.1 = new_frame
        };
    }

    /// Returns the texture ID, size, and a dummy value.
    fn get(&self) -> (u32, Size2D<i32>, usize) {
        if self.0 == FrameStatus::Locked {
            (
                self.1.get_texture_id(),
                Size2D::new(self.1.get_width(), self.1.get_height()),
                0,
            )
        } else {
            unreachable!();
        }
    }

    /// Returns the frame.
    fn get_frame(&self) -> VideoFrame {
        self.1.clone()
    }
}

/// A renderer for media frames.
pub(crate) struct MediaFrameRenderer {
    player_id: Option<u64>,
    compositor_api: CrossProcessCompositorApi,
    current_frame: Option<MediaFrame>,
    old_frame: Option<ImageKey>,
    very_old_frame: Option<ImageKey>,
    current_frame_holder: Option<FrameHolder>,
    /// <https://html.spec.whatwg.org/multipage/#poster-frame>
    poster_frame: Option<MediaFrame>,
}

impl MediaFrameRenderer {
    /// Creates a new `MediaFrameRenderer`.
    fn new(compositor_api: CrossProcessCompositorApi) -> Self {
        Self {
            player_id: None,
            compositor_api,
            current_frame: None,
            old_frame: None,
            very_old_frame: None,
            current_frame_holder: None,
            poster_frame: None,
        }
    }

    /// Sets the poster frame.
    fn set_poster_frame(&mut self, image: Option<Arc<RasterImage>>) {
        self.poster_frame = image.and_then(|image| {
            image.id.map(|image_key| MediaFrame {
                image_key,
                width: image.metadata.width as i32,
                height: image.metadata.height as i32,
            })
        });
    }
}

impl VideoFrameRenderer for MediaFrameRenderer {
    fn render(&mut self, frame: VideoFrame) {
        let mut updates = vec![];

        if let Some(old_image_key) = mem::replace(&mut self.very_old_frame, self.old_frame.take()) {
            updates.push(ImageUpdate::DeleteImage(old_image_key));
        }

        let descriptor = ImageDescriptor::new(
            frame.get_width(),
            frame.get_height(),
            ImageFormat::BGRA8,
            ImageDescriptorFlags::empty(),
        );

        match &mut self.current_frame {
            Some(current_frame)
                if current_frame.width == frame.get_width() &&
                    current_frame.height == frame.get_height() =>
            {
                if !frame.is_gl_texture() {
                    updates.push(ImageUpdate::UpdateImage(
                        current_frame.image_key,
                        descriptor,
                        SerializableImageData::Raw(IpcSharedMemory::from_bytes(&frame.get_data())),
                    ));
                }

                self.current_frame_holder
                    .get_or_insert_with(|| FrameHolder::new(frame.clone()))
                    .set(frame);

                if let Some(old_image_key) = self.old_frame.take() {
                    updates.push(ImageUpdate::DeleteImage(old_image_key));
                }
            },
            Some(current_frame) => {
                self.old_frame = Some(current_frame.image_key);

                let Some(new_image_key) = self.compositor_api.generate_image_key() else {
                    return;
                };

                /* update current_frame */
                current_frame.image_key = new_image_key;
                current_frame.width = frame.get_width();
                current_frame.height = frame.get_height();

                let image_data = if frame.is_gl_texture() && self.player_id.is_some() {
                    let texture_target = if frame.is_external_oes() {
                        ImageBufferKind::TextureExternal
                    } else {
                        ImageBufferKind::Texture2D
                    };

                    SerializableImageData::External(ExternalImageData {
                        id: ExternalImageId(self.player_id.unwrap()),
                        channel_index: 0,
                        image_type: ExternalImageType::TextureHandle(texture_target),
                        normalized_uvs: false,
                    })
                } else {
                    SerializableImageData::Raw(IpcSharedMemory::from_bytes(&frame.get_data()))
                };

                self.current_frame_holder
                    .get_or_insert_with(|| FrameHolder::new(frame.clone()))
                    .set(frame);

                updates.push(ImageUpdate::AddImage(new_image_key, descriptor, image_data));
            },
            None => {
                let Some(image_key) = self.compositor_api.generate_image_key() else {
                    return;
                };

                self.current_frame = Some(MediaFrame {
                    image_key,
                    width: frame.get_width(),
                    height: frame.get_height(),
                });

                let image_data = if frame.is_gl_texture() && self.player_id.is_some() {
                    let texture_target = if frame.is_external_oes() {
                        ImageBufferKind::TextureExternal
                    } else {
                        ImageBufferKind::Texture2D
                    };

                    SerializableImageData::External(ExternalImageData {
                        id: ExternalImageId(self.player_id.unwrap()),
                        channel_index: 0,
                        image_type: ExternalImageType::TextureHandle(texture_target),
                        normalized_uvs: false,
                    })
                } else {
                    SerializableImageData::Raw(IpcSharedMemory::from_bytes(&frame.get_data()))
                };

                self.current_frame_holder = Some(FrameHolder::new(frame));

                updates.push(ImageUpdate::AddImage(image_key, descriptor, image_data));
            },
        }
        self.compositor_api.update_images(updates);
    }
}

/// The source object for a media element.
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
#[derive(JSTraceable, MallocSizeOf)]
enum SrcObject {
    /// A `MediaStream` object.
    MediaStream(Dom<MediaStream>),
    /// A `Blob` object.
    Blob(Dom<Blob>),
}

impl From<MediaStreamOrBlob> for SrcObject {
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    fn from(src_object: MediaStreamOrBlob) -> SrcObject {
        match src_object {
            MediaStreamOrBlob::Blob(blob) => SrcObject::Blob(Dom::from_ref(&*blob)),
            MediaStreamOrBlob::MediaStream(stream) => {
                SrcObject::MediaStream(Dom::from_ref(&*stream))
            },
        }
    }
}

/// A struct that automatically cleans up a media element when it is dropped.
#[derive(JSTraceable, MallocSizeOf)]
struct DroppableHtmlMediaElement {
    /// Player Id reported the player thread
    player_id: Cell<u64>,
    #[ignore_malloc_size_of = "Defined in other crates"]
    #[no_trace]
    player_context: WindowGLContext,
}

impl DroppableHtmlMediaElement {
    /// Creates a new `DroppableHtmlMediaElement`.
    fn new(player_id: Cell<u64>, player_context: WindowGLContext) -> Self {
        Self {
            player_id,
            player_context,
        }
    }

    /// Sets the player ID.
    pub(crate) fn set_player_id(&self, id: u64) {
        self.player_id.set(id);
    }
}

impl Drop for DroppableHtmlMediaElement {
    fn drop(&mut self) {
        self.player_context
            .send(GLPlayerMsg::UnregisterPlayer(self.player_id.get()));
    }
}
/// The `HTMLMediaElement` struct.
#[dom_struct]
#[allow(non_snake_case)]
pub(crate) struct HTMLMediaElement {
    htmlelement: HTMLElement,
    /// <https://html.spec.whatwg.org/multipage/#dom-media-networkstate>
    network_state: Cell<NetworkState>,
    /// <https://html.spec.whatwg.org/multipage/#dom-media-readystate>
    ready_state: Cell<ReadyState>,
    /// <https://html.spec.whatwg.org/multipage/#dom-media-srcobject>
    src_object: DomRefCell<Option<SrcObject>>,
    /// <https://html.spec.whatwg.org/multipage/#dom-media-currentsrc>
    current_src: DomRefCell<String>,
    /// Incremented whenever tasks associated with this element are cancelled.
    generation_id: Cell<u32>,
    /// <https://html.spec.whatwg.org/multipage/#fire-loadeddata>
    ///
    /// Reset to false every time the load algorithm is invoked.
    fired_loadeddata_event: Cell<bool>,
    /// <https://html.spec.whatwg.org/multipage/#dom-media-error>
    error: MutNullableDom<MediaError>,
    /// <https://html.spec.whatwg.org/multipage/#dom-media-paused>
    paused: Cell<bool>,
    /// <https://html.spec.whatwg.org/multipage/#dom-media-defaultplaybackrate>
    defaultPlaybackRate: Cell<f64>,
    /// <https://html.spec.whatwg.org/multipage/#dom-media-playbackrate>
    playbackRate: Cell<f64>,
    /// <https://html.spec.whatwg.org/multipage/#attr-media-autoplay>
    autoplaying: Cell<bool>,
    /// <https://html.spec.whatwg.org/multipage/#delaying-the-load-event-flag>
    delaying_the_load_event_flag: DomRefCell<Option<LoadBlocker>>,
    /// <https://html.spec.whatwg.org/multipage/#list-of-pending-play-promises>
    #[ignore_malloc_size_of = "promises are hard"]
    pending_play_promises: DomRefCell<Vec<Rc<Promise>>>,
    /// Play promises which are soon to be fulfilled by a queued task.
    #[allow(clippy::type_complexity)]
    #[ignore_malloc_size_of = "promises are hard"]
    in_flight_play_promises_queue: DomRefCell<VecDeque<(Box<[Rc<Promise>]>, ErrorResult)>>,
    #[ignore_malloc_size_of = "servo_media"]
    #[no_trace]
    player: DomRefCell<Option<Arc<Mutex<dyn Player>>>>,
    #[ignore_malloc_size_of = "Arc"]
    #[no_trace]
    video_renderer: Arc<Mutex<MediaFrameRenderer>>,
    #[ignore_malloc_size_of = "Arc"]
    #[no_trace]
    audio_renderer: DomRefCell<Option<Arc<Mutex<dyn AudioRenderer>>>>,
    /// <https://html.spec.whatwg.org/multipage/#show-poster-flag>
    show_poster: Cell<bool>,
    /// <https://html.spec.whatwg.org/multipage/#dom-media-duration>
    duration: Cell<f64>,
    /// <https://html.spec.whatwg.org/multipage/#official-playback-position>
    playback_position: Cell<f64>,
    /// <https://html.spec.whatwg.org/multipage/#default-playback-start-position>
    default_playback_start_position: Cell<f64>,
    /// <https://html.spec.whatwg.org/multipage/#dom-media-volume>
    volume: Cell<f64>,
    /// <https://html.spec.whatwg.org/multipage/#dom-media-seeking>
    seeking: Cell<bool>,
    /// <https://html.spec.whatwg.org/multipage/#dom-media-muted>
    muted: Cell<bool>,
    /// URL of the media resource, if any.
    #[no_trace]
    resource_url: DomRefCell<Option<ServoUrl>>,
    /// URL of the media resource, if the resource is set through the src_object attribute and it
    /// is a blob.
    #[no_trace]
    blob_url: DomRefCell<Option<ServoUrl>>,
    /// <https://html.spec.whatwg.org/multipage/#dom-media-played>
    #[ignore_malloc_size_of = "Rc"]
    played: DomRefCell<TimeRangesContainer>,
    // https://html.spec.whatwg.org/multipage/#dom-media-audiotracks
    audio_tracks_list: MutNullableDom<AudioTrackList>,
    // https://html.spec.whatwg.org/multipage/#dom-media-videotracks
    video_tracks_list: MutNullableDom<VideoTrackList>,
    /// <https://html.spec.whatwg.org/multipage/#dom-media-texttracks>
    text_tracks_list: MutNullableDom<TextTrackList>,
    /// Time of last timeupdate notification.
    #[ignore_malloc_size_of = "Defined in std::time"]
    next_timeupdate_event: Cell<Instant>,
    /// Latest fetch request context.
    current_fetch_context: DomRefCell<Option<HTMLMediaElementFetchContext>>,
    /// Media controls id.
    /// In order to workaround the lack of privileged JS context, we secure the
    /// the access to the "privileged" document.servoGetMediaControls(id) API by
    /// keeping a whitelist of media controls identifiers.
    media_controls_id: DomRefCell<Option<String>>,
    droppable: DroppableHtmlMediaElement,
}

/// <https://html.spec.whatwg.org/multipage/#dom-media-networkstate>
#[derive(Clone, Copy, JSTraceable, MallocSizeOf, PartialEq)]
#[repr(u8)]
pub(crate) enum NetworkState {
    Empty = HTMLMediaElementConstants::NETWORK_EMPTY as u8,
    Idle = HTMLMediaElementConstants::NETWORK_IDLE as u8,
    Loading = HTMLMediaElementConstants::NETWORK_LOADING as u8,
    NoSource = HTMLMediaElementConstants::NETWORK_NO_SOURCE as u8,
}
