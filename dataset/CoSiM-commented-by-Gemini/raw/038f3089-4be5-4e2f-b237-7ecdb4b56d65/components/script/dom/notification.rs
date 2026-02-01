/*!
This module implements the Notification API, which allows web pages to
display notifications to the user.

The `Notification` struct is the main entry point for this module. It creates
a new `Notification` object, which can be used to display a notification.

The `Notification` API is defined in the HTML specification:
<https://notifications.spec.whatwg.org/>
*/

use std::collections::HashSet;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use content_security_policy as csp;
use content_security_policy::Destination;
use dom_struct::dom_struct;
use embedder_traits::{
    EmbedderMsg, Notification as EmbedderNotification,
    NotificationAction as EmbedderNotificationAction,
};
use ipc_channel::ipc;
use ipc_channel::router::ROUTER;
use js::jsapi::Heap;
use js::jsval::JSVal;
use js::rust::{HandleObject, MutableHandleValue};
use net_traits::http_status::HttpStatus;
use net_traits::image_cache::{
    ImageCache, ImageCacheResponseMessage, ImageCacheResult, ImageLoadListener,
    ImageOrMetadataAvailable, ImageResponse, PendingImageId, UsePlaceholder,
};
use net_traits::request::{RequestBuilder, RequestId};
use net_traits::{
    FetchMetadata, FetchResponseListener, FetchResponseMsg, NetworkError, ResourceFetchTiming,
    ResourceTimingType,
};
use pixels::RasterImage;
use servo_url::{ImmutableOrigin, ServoUrl};
use uuid::Uuid;

use super::bindings::cell::DomRefCell;
use super::bindings::refcounted::{Trusted, TrustedPromise};
use super::bindings::reflector::DomGlobal;
use super::bindings::trace::RootedTraceableBox;
use super::performanceresourcetiming::InitiatorType;
use super::permissionstatus::PermissionStatus;
use crate::dom::bindings::callback::ExceptionHandling;
use crate::dom::bindings::codegen::Bindings::NotificationBinding::{
    NotificationAction, NotificationDirection, NotificationMethods, NotificationOptions,
    NotificationPermission, NotificationPermissionCallback,
};
use crate::dom::bindings::codegen::Bindings::PermissionStatusBinding::PermissionStatus_Binding::PermissionStatusMethods;
use crate::dom::bindings::codegen::Bindings::PermissionStatusBinding::{
    PermissionDescriptor, PermissionName, PermissionState,
};
use crate::dom::bindings::codegen::UnionTypes::UnsignedLongOrUnsignedLongSequence;
use crate::dom::bindings::error::{Error, Fallible};
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::reflector::reflect_dom_object_with_proto;
use crate::dom::bindings::root::{Dom, DomRoot};
use crate::dom::bindings::str::DOMString;
use crate::dom::csp::report_csp_violations;
use crate::dom::eventtarget::EventTarget;
use crate::dom::globalscope::GlobalScope;
use crate::dom::permissions::{PermissionAlgorithm, Permissions, descriptor_permission_state};
use crate::dom::promise::Promise;
use crate::dom::serviceworkerglobalscope::ServiceWorkerGlobalScope;
use crate::dom::serviceworkerregistration::ServiceWorkerRegistration;
use crate::fetch::create_a_potential_cors_request;
use crate::network_listener::{self, PreInvoke, ResourceTimingListener};
use crate::script_runtime::CanGc;

// TODO: Service Worker API (persistent notification)
// https://notifications.spec.whatwg.org/#service-worker-api

/// <https://notifications.spec.whatwg.org/#notifications>
#[dom_struct]
pub(crate) struct Notification {
    eventtarget: EventTarget,
    /// <https://notifications.spec.whatwg.org/#service-worker-registration>
    serviceworker_registration: Option<Dom<ServiceWorkerRegistration>>,
    /// <https://notifications.spec.whatwg.org/#concept-title>
    title: DOMString,
    /// <https://notifications.spec.whatwg.org/#body>
    body: DOMString,
    /// <https://notifications.spec.whatwg.org/#data>
    #[ignore_malloc_size_of = "mozjs"]
    data: Heap<JSVal>,
    /// <https://notifications.spec.whatwg.org/#concept-direction>
    dir: NotificationDirection,
    /// <https://notifications.spec.whatwg.org/#image-url>
    image: Option<USVString>,
    /// <https://notifications.spec.whatwg.org/#icon-url>
    icon: Option<USVString>,
    /// <https://notifications.spec.whatwg.org/#badge-url>
    badge: Option<USVString>,
    /// <https://notifications.spec.whatwg.org/#concept-language>
    lang: DOMString,
    /// <https://notifications.spec.whatwg.org/#silent-preference-flag>
    silent: Option<bool>,
    /// <https://notifications.spec.whatwg.org/#tag>
    tag: DOMString,
    /// <https://notifications.spec.whatwg.org/#concept-origin>
    #[no_trace] // ImmutableOrigin is not traceable
    origin: ImmutableOrigin,
    /// <https://notifications.spec.whatwg.org/#vibration-pattern>
    vibration_pattern: Vec<u32>,
    /// <https://notifications.spec.whatwg.org/#timestamp>
    timestamp: u64,
    /// <https://notifications.spec.whatwg.org/#renotify-preference-flag>
    renotify: bool,
    /// <https://notifications.spec.whatwg.org/#require-interaction-preference-flag>
    require_interaction: bool,
    /// <https://notifications.spec.whatwg.org/#actions>
    actions: Vec<Action>,
    /// Pending image, icon, badge, action icon resource request's id
    #[no_trace] // RequestId is not traceable
    pending_request_ids: DomRefCell<HashSet<RequestId>>,
    /// <https://notifications.spec.whatwg.org/#image-resource>
    #[ignore_malloc_size_of = "Arc"]
    #[no_trace]
    image_resource: DomRefCell<Option<Arc<RasterImage>>>,
    /// <https://notifications.spec.whatwg.org/#icon-resource>
    #[ignore_malloc_size_of = "Arc"]
    #[no_trace]
    icon_resource: DomRefCell<Option<Arc<RasterImage>>>,
    /// <https://notifications.spec.whatwg.org/#badge-resource>
    #[ignore_malloc_size_of = "Arc"]
    #[no_trace]
    badge_resource: DomRefCell<Option<Arc<RasterImage>>>,
}

impl Notification {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        global: &GlobalScope,
        title: DOMString,
        options: RootedTraceableBox<NotificationOptions>,
        origin: ImmutableOrigin,
        base_url: ServoUrl,
        fallback_timestamp: u64,
        proto: Option<HandleObject>,
        can_gc: CanGc,
    ) -> DomRoot<Self> {
        let notification = reflect_dom_object_with_proto(
            Box::new(Notification::new_inherited(
                global,
                title,
                &options,
                origin,
                base_url,
                fallback_timestamp,
            )),
            global,
            proto,
            can_gc,
        );

        notification.data.set(options.data.get());

        notification
    }

    /// partial implementation of <https://notifications.spec.whatwg.org/#create-a-notification>
    fn new_inherited(
        global: &GlobalScope,
        title: DOMString,
        options: &RootedTraceableBox<NotificationOptions>,
        origin: ImmutableOrigin,
        base_url: ServoUrl,
        fallback_timestamp: u64,
    ) -> Self {
        // TODO: missing call to https://html.spec.whatwg.org/multipage/#structuredserializeforstorage
        // may be find in `dom/bindings/structuredclone.rs`
        let data = Heap::default();

        let title = title.clone();
        let dir = options.dir;
        let lang = options.lang.clone();
        let body = options.body.clone();
        let tag = options.tag.clone();

        // If options["image"] exists, then parse it using baseURL, and if that does not return failure,
        // set notification’s image URL to the return value. (Otherwise notification’s image URL is not set.)
        let image = options.image.as_ref().and_then(|image_url| {
            ServoUrl::parse_with_base(Some(&base_url), image_url.as_ref())
                .map(|url| USVString::from(url.to_string()))
                .ok()
        });
        // If options["icon"] exists, then parse it using baseURL, and if that does not return failure,
        // set notification’s icon URL to the return value. (Otherwise notification’s icon URL is not set.)
        let icon = options.icon.as_ref().and_then(|icon_url| {
            ServoUrl::parse_with_base(Some(&base_url), icon_url.as_ref())
                .map(|url| USVString::from(url.to_string()))
                .ok()
        });
        // If options["badge"] exists, then parse it using baseURL, and if that does not return failure,
        // set notification’s badge URL to the return value. (Otherwise notification’s badge URL is not set.)
        let badge = options.badge.as_ref().and_then(|badge_url| {
            ServoUrl::parse_with_base(Some(&base_url), badge_url.as_ref())
                .map(|url| USVString::from(url.to_string()))
                .ok()
        });
        // If options["vibrate"] exists, then validate and normalize it and
        // set notification’s vibration pattern to the return value.
        let vibration_pattern = match &options.vibrate {
            Some(pattern) => validate_and_normalize_vibration_pattern(pattern),
            None => Vec::new(),
        };
        // If options["timestamp"] exists, then set notification’s timestamp to the value.
        // Otherwise, set notification’s timestamp to fallbackTimestamp.
        let timestamp = options.timestamp.unwrap_or(fallback_timestamp);
        let renotify = options.renotify;
        let silent = options.silent;
        let require_interaction = options.requireInteraction;

        // For each entry in options["actions"]
        // up to the maximum number of actions supported (skip any excess entries):
        let mut actions: Vec<Action> = Vec::new();
        let max_actions = Notification::MaxActions(global);
        for action in options.actions.iter().take(max_actions as usize) {
            actions.push(Action {
                id: Uuid::new_v4().simple().to_string(),
                name: action.action.clone(),
                title: action.title.clone(),
                // If entry["icon"] exists, then parse it using baseURL, and if that does not return failure
                // set action’s icon URL to the return value. (Otherwise action’s icon URL remains null.)
                icon_url: action.icon.as_ref().and_then(|icon_url| {
                    ServoUrl::parse_with_base(Some(&base_url), icon_url.as_ref())
                        .map(|url| USVString::from(url.to_string()))
                        .ok()
                }),
                icon_resource: DomRefCell::new(None),
            });
        }

        Self {
            eventtarget: EventTarget::new_inherited(),
            // A non-persistent notification is a notification whose service worker registration is null.
            serviceworker_registration: None,
            title,
            body,
            data,
            dir,
            image,
            icon,
            badge,
            lang,
            silent,
            origin,
            vibration_pattern,
            timestamp,
            renotify,
            tag,
            require_interaction,
            actions,
            pending_request_ids: DomRefCell::new(HashSet::new()),
            image_resource: DomRefCell::new(None),
            icon_resource: DomRefCell::new(None),
            badge_resource: DomRefCell::new(None),
        }
    }

    /// <https://notifications.spec.whatwg.org/#notification-show-steps>
    fn show(&self) {
        // step 3: set shown to false
        let shown = false;

        // TODO: step 4: Let oldNotification be the notification in the list of notifications
        //               whose tag is not the empty string and is notification’s tag,
        //               and whose origin is same origin with notification’s origin,
        //               if any, and null otherwise.

        // TODO: step 5: If oldNotification is non-null, then:
        // TODO:   step 5.1: Handle close events with oldNotification.
        // TODO:   step 5.2: If the notification platform supports replacement, then:
        // TODO:     step 5.2.1: Replace oldNotification with notification, in the list of notifications.
        // TODO:     step 5.2.2: Set shown to true.
        // TODO:   step 5.3: Otherwise, remove oldNotification from the list of notifications.

        // step 6: If shown is false, then:
        if !shown {
            // TODO: step 6.1: Append notification to the list of notifications.
            // step 6.2: Display notification on the device
            self.global()
                .send_to_embedder(EmbedderMsg::ShowNotification(
                    self.global().webview_id(),
                    self.to_embedder_notification(),
                ));
        }

        // TODO: step 7: If shown is false or oldNotification is non-null,
        //               and notification’s renotify preference is true,
        //               then run the alert steps for notification.

        // step 8: If notification is a non-persistent notification,
        //         then queue a task to fire an event named show on
        //         the Notification object representing notification.
        if self.serviceworker_registration.is_none() {
            self.global()
                .task_manager()
                .dom_manipulation_task_source()
                .queue_simple_event(self.upcast(), atom!("show"));
        }
    }

    /// Create an [`embedder_traits::Notification`].
    fn to_embedder_notification(&self) -> EmbedderNotification {
        EmbedderNotification {
            title: self.title.to_string(),
            body: self.body.to_string(),
            tag: self.tag.to_string(),
            language: self.lang.to_string(),
            require_interaction: self.require_interaction,
            silent: self.silent,
            icon_url: self
                .icon
                .as_ref()
                .and_then(|icon| ServoUrl::parse(icon).ok()),
            badge_url: self
                .badge
                .as_ref()
                .and_then(|badge| ServoUrl::parse(badge).ok()),
            image_url: self
                .image
                .as_ref()
                .and_then(|image| ServoUrl::parse(image).ok()),
            actions: self
                .actions
                .iter()
                .map(|action| EmbedderNotificationAction {
                    name: action.name.to_string(),
                    title: action.title.to_string(),
                    icon_url: action
                        .icon_url
                        .as_ref()
                        .and_then(|icon| ServoUrl::parse(icon).ok()),
                    icon_resource: action.icon_resource.borrow().clone(),
                })
                .collect(),
            icon_resource: self.icon_resource.borrow().clone(),
            badge_resource: self.badge_resource.borrow().clone(),
            image_resource: self.image_resource.borrow().clone(),
        }
    }
}