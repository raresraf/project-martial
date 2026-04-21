/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! # JSAPI Proxy Handler Utilities
//!
//! This module provides the infrastructure for implementing custom JavaScript
//! proxy handlers within the script engine. It specifically targets DOM proxy
//! objects, which require specialized behavior for property resolution,
//! expando management, and cross-origin security boundaries.
//!
//! ## Architectural Intent
//!
//! The utilities here bridge the gap between Rust-based DOM implementations
//! and the underlying SpiderMonkey JSAPI, ensuring that proxy traps (like
//! `get`, `set`, and `defineProperty`) adhere to both the ECMAScript and HTML
//! specifications regarding platform objects.
//!
//! DOM proxies are unique because they often represent "live" collections or
//! objects whose properties are not fixed. They also need to support "expandos"
//! (arbitrary properties added by scripts) while maintaining strict security
//! invariants, especially in cross-origin scenarios.

use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;

use js::conversions::ToJSValConvertible;
use js::glue::{
    GetProxyHandler, GetProxyHandlerFamily, GetProxyPrivate, InvokeGetOwnPropertyDescriptor,
    SetProxyPrivate,
};
use js::jsapi::{
    DOMProxyShadowsResult, GetStaticPrototype, GetWellKnownSymbol, Handle as RawHandle,
    HandleId as RawHandleId, HandleObject as RawHandleObject, HandleValue as RawHandleValue,
    JS_AtomizeAndPinString, JS_DefinePropertyById, JS_GetOwnPropertyDescriptorById,
    JS_IsExceptionPending, JSAutoRealm, JSContext, JSErrNum, JSFunctionSpec, JSObject,
    JSPropertySpec, MutableHandle as RawMutableHandle,
    MutableHandleIdVector as RawMutableHandleIdVector,
    MutableHandleObject as RawMutableHandleObject, MutableHandleValue as RawMutableHandleValue,
    ObjectOpResult, PropertyDescriptor, SetDOMProxyInformation, SymbolCode, jsid,
};
use js::jsid::SymbolId;
use js::jsval::{ObjectValue, UndefinedValue};
use js::rust::wrappers::{
    AppendToIdVector, JS_AlreadyHasOwnPropertyById, JS_NewObjectWithGivenProto,
    RUST_INTERNED_STRING_TO_JSID, SetDataPropertyDescriptor,
};
use js::rust::{Handle, HandleObject, HandleValue, MutableHandle, MutableHandleObject};
use js::{jsapi, rooted};

use crate::DomTypes;
use crate::conversions::{is_dom_proxy, jsid_to_string, jsstring_to_str};
use crate::error::Error;
use crate::interfaces::{DomHelpers, GlobalScopeHelpers};
use crate::realms::{AlreadyInRealm, InRealm};
use crate::reflector::DomObject;
use crate::root::DomRoot;
use crate::script_runtime::{CanGc, JSContext as SafeJSContext};
use crate::str::DOMString;
use crate::utils::delete_property_by_id;

/// Determine if a given property identifier shadows any existing properties on this proxy.
///
/// This callback is used by SpiderMonkey's internal proxy logic to efficiently check
/// if a property access should be handled by the proxy's own traps or if it might
/// be "shadowed" by an expando property.
///
/// # Returns
/// - `DOMProxyShadowsResult::ShadowsViaDirectExpando` if the property exists on the expando object.
/// - `DOMProxyShadowsResult::DoesntShadow` otherwise.
///
/// # Safety
/// `cx` must point to a valid, non-null JSContext.
pub(crate) unsafe extern "C" fn shadow_check_callback(
    cx: *mut JSContext,
    object: RawHandleObject,
    id: RawHandleId,
) -> DOMProxyShadowsResult {
    // TODO: support OverrideBuiltins when #12978 is fixed.

    // Retrieve the expando object associated with this proxy, if any.
    rooted!(in(cx) let mut expando = ptr::null_mut::<JSObject>());
    get_expando_object(object, expando.handle_mut());
    
    // If an expando object exists, check if it contains the property 'id'.
    if !expando.get().is_null() {
        let mut has_own = false;
        let raw_id = Handle::from_raw(id);

        // Perform a direct "own property" check on the expando object.
        if !JS_AlreadyHasOwnPropertyById(cx, expando.handle(), raw_id, &mut has_own) {
            return DOMProxyShadowsResult::ShadowCheckFailed;
        }

        if has_own {
            return DOMProxyShadowsResult::ShadowsViaDirectExpando;
        }
    }

    // Our expando, if any, didn't shadow, so we're not shadowing at all.
    DOMProxyShadowsResult::DoesntShadow
}

/// Initialize the global DOM proxy infrastructure.
///
/// This function registers the proxy handler family and the shadow check callback
/// with SpiderMonkey, enabling the specialized DOM proxy behaviors defined in this module.
pub fn init() {
    unsafe {
        SetDOMProxyInformation(
            GetProxyHandlerFamily(),
            Some(shadow_check_callback),
            ptr::null(),
        );
    }
}

/// Defines a property on the proxy's expando object.
///
/// In the DOM specification, platform objects can have properties added to them
/// that are not part of their IDL-defined interface. These are stored on a
/// hidden "expando" object to separate them from the native implementation.
///
/// # Safety
/// - `cx` must be a valid pointer to a JSContext.
/// - `proxy` and `id` must be valid handles within the current realm.
/// - `desc` must be a valid property descriptor.
/// - `result` must be a valid pointer to an ObjectOpResult for reporting success/failure.
pub(crate) unsafe extern "C" fn define_property(
    cx: *mut JSContext,
    proxy: RawHandleObject,
    id: RawHandleId,
    desc: RawHandle<PropertyDescriptor>,
    result: *mut ObjectOpResult,
) -> bool {
    rooted!(in(cx) let mut expando = ptr::null_mut::<JSObject>());
    // Ensure the expando object exists before attempting to define a property on it.
    ensure_expando_object(cx, proxy, expando.handle_mut());
    JS_DefinePropertyById(cx, expando.handle().into(), id, desc, result)
}

/// Deletes a property from the proxy's expando object.
///
/// If the expando object does not exist, the operation is a no-op and returns success.
///
/// # Safety
/// - `cx` must be a valid pointer to a JSContext.
/// - `proxy` and `id` must be valid handles.
/// - `bp` must be a valid pointer to an ObjectOpResult.
pub(crate) unsafe extern "C" fn delete(
    cx: *mut JSContext,
    proxy: RawHandleObject,
    id: RawHandleId,
    bp: *mut ObjectOpResult,
) -> bool {
    rooted!(in(cx) let mut expando = ptr::null_mut::<JSObject>());
    get_expando_object(proxy, expando.handle_mut());
    
    // If there's no expando object, there's nothing to delete.
    if expando.is_null() {
        (*bp).code_ = 0 /* OkCode */;
        return true;
    }

    delete_property_by_id(cx, expando.handle(), Handle::from_raw(id), bp)
}

/// Implements the `[[PreventExtensions]]` trap for DOM proxies.
///
/// By default, DOM proxies are non-extensible in a way that prevents changing
/// the extensible bit. This trap always reports an error when attempting to
/// prevent extensions.
///
/// # Safety
/// `result` must point to a valid, non-null ObjectOpResult.
pub(crate) unsafe extern "C" fn prevent_extensions(
    _cx: *mut JSContext,
    _proxy: RawHandleObject,
    result: *mut ObjectOpResult,
) -> bool {
    // Explicitly prohibit preventing extensions on DOM proxies.
    (*result).code_ = JSErrNum::JSMSG_CANT_PREVENT_EXTENSIONS as ::libc::uintptr_t;
    true
}

/// Implements the `[[IsExtensible]]` trap for DOM proxies.
///
/// DOM platform objects are generally extensible (allowing for expandos).
///
/// # Safety
/// `succeeded` must point to a valid, non-null bool.
pub(crate) unsafe extern "C" fn is_extensible(
    _cx: *mut JSContext,
    _proxy: RawHandleObject,
    succeeded: *mut bool,
) -> bool {
    // DOM proxies are considered extensible by default.
    *succeeded = true;
    true
}

/// Implements a trap to retrieve the prototype if the object behaves like an ordinary object.
///
/// If `proxy` (underneath any functionally-transparent wrapper proxies) has as
/// its `[[GetPrototypeOf]]` trap the ordinary `[[GetPrototypeOf]]` behavior
/// defined for ordinary objects, set `*is_ordinary` to true and store `obj`'s
/// prototype in `proto`. Otherwise set `*is_ordinary` to false.
///
/// # Architectural Note
/// This implementation handles the case of ordinary `[[GetPrototypeOf]]` behavior.
/// Special logic is required for cross-origin objects (like `Window` or `Location`)
/// which have non-standard prototype resolution rules.
///
/// # Safety
/// `is_ordinary` must point to a valid, non-null bool.
pub(crate) unsafe extern "C" fn get_prototype_if_ordinary(
    _: *mut JSContext,
    proxy: RawHandleObject,
    is_ordinary: *mut bool,
    proto: RawMutableHandleObject,
) -> bool {
    *is_ordinary = true;
    // Retrieve the static prototype assigned during object creation.
    proto.set(GetStaticPrototype(proxy.get()));
    true
}

/// Internal helper to retrieve the expando object associated with a proxy.
///
/// The expando object is stored in the "private" slot of the proxy.
pub(crate) fn get_expando_object(obj: RawHandleObject, mut expando: MutableHandleObject) {
    unsafe {
        assert!(is_dom_proxy(obj.get()));
        let val = &mut UndefinedValue();
        // Extract the value from the proxy's internal storage.
        GetProxyPrivate(obj.get(), val);
        expando.set(if val.is_undefined() {
            ptr::null_mut()
        } else {
            val.to_object()
        });
    }
}

/// Internal helper to ensure an expando object exists, creating it if necessary.
///
/// This is used when defining new properties that don't belong to the native implementation.
///
/// # Safety
/// `cx` must point to a valid, non-null JSContext.
pub(crate) unsafe fn ensure_expando_object(
    cx: *mut JSContext,
    obj: RawHandleObject,
    mut expando: MutableHandleObject,
) {
    assert!(is_dom_proxy(obj.get()));
    get_expando_object(obj, expando.reborrow());
    
    // Create a new ordinary object to serve as the expando holder if one doesn't exist.
    if expando.is_null() {
        expando.set(JS_NewObjectWithGivenProto(
            cx,
            ptr::null_mut(),
            HandleObject::null(),
        ));
        assert!(!expando.is_null());

        // Store the newly created expando object in the proxy's private slot.
        SetProxyPrivate(obj.get(), &ObjectValue(expando.get()));
    }
}

/// Configures a property descriptor with a specific value and attributes.
///
/// This is a convenience wrapper around `SetDataPropertyDescriptor`.
pub fn set_property_descriptor(
    desc: MutableHandle<PropertyDescriptor>,
    value: HandleValue,
    attrs: u32,
    is_none: &mut bool,
) {
    unsafe {
        SetDataPropertyDescriptor(desc, value, attrs);
    }
    *is_none = false;
}

/// Converts a JS ID to its string representation for debugging or diagnostic purposes.
///
/// This effectively performs a `ValueToSource` conversion on the underlying property ID.
pub(crate) fn id_to_source(cx: SafeJSContext, id: RawHandleId) -> Option<DOMString> {
    unsafe {
        rooted!(in(*cx) let mut value = UndefinedValue());
        rooted!(in(*cx) let mut jsstr = ptr::null_mut::<jsapi::JSString>());
        jsapi::JS_IdToValue(*cx, id.get(), value.handle_mut().into())
            .then(|| {
                jsstr.set(jsapi::JS_ValueToSource(*cx, value.handle().into()));
                jsstr.get()
            })
            .and_then(ptr::NonNull::new)
            .map(|jsstr| jsstring_to_str(*cx, jsstr))
    }
}

/// Encapsulates properties and methods that are accessible cross-origin.
///
/// According to the HTML spec, certain objects (like `Window` or `Location`)
/// allow a limited set of properties to be accessed even from different origins.
///
/// See: [`CrossOriginProperties(O)`](https://html.spec.whatwg.org/multipage/#crossoriginproperties-(-o-))
pub(crate) struct CrossOriginProperties {
    pub(crate) attributes: &'static [JSPropertySpec],
    pub(crate) methods: &'static [JSFunctionSpec],
}

impl CrossOriginProperties {
    /// Returns an iterator over the raw string names of the cross-origin properties.
    fn keys(&self) -> impl Iterator<Item = *const c_char> + '_ {
        // Safety: All cross-origin property keys are strings, not symbols.
        self.attributes
            .iter()
            .map(|spec| unsafe { spec.name.string_ })
            .chain(self.methods.iter().map(|spec| unsafe { spec.name.string_ }))
            .filter(|ptr| !ptr.is_null())
    }
}

/// Implements the `[[OwnPropertyKeys]]` trap for cross-origin objects.
///
/// This ensures that only allowlisted properties are visible when enumerating
/// properties on a cross-origin object.
///
/// See: [`CrossOriginOwnPropertyKeys`](https://html.spec.whatwg.org/multipage/#crossoriginownpropertykeys-(-o-))
pub(crate) fn cross_origin_own_property_keys(
    cx: SafeJSContext,
    _proxy: RawHandleObject,
    cross_origin_properties: &'static CrossOriginProperties,
    props: RawMutableHandleIdVector,
) -> bool {
    // 1. Append keys from the specific cross-origin properties definition.
    for key in cross_origin_properties.keys() {
        unsafe {
            rooted!(in(*cx) let rooted = JS_AtomizeAndPinString(*cx, key));
            rooted!(in(*cx) let mut rooted_jsid: jsid);
            RUST_INTERNED_STRING_TO_JSID(*cx, rooted.handle().get(), rooted_jsid.handle_mut());
            AppendToIdVector(props, rooted_jsid.handle());
        }
    }

    // 2. Append globally allowlisted properties (e.g., "then", Symbol.toStringTag).
    append_cross_origin_allowlisted_prop_keys(cx, props);

    true
}

/// A raw JSAPI callback for retrieving the prototype of a potentially cross-origin object.
///
/// Cross-origin objects have a custom `[[GetPrototypeOf]]` trap that typically returns null
/// to prevent leaking the prototype chain across origin boundaries.
///
/// # Safety
/// `is_ordinary` must point to a valid, non-null bool.
pub(crate) unsafe extern "C" fn maybe_cross_origin_get_prototype_if_ordinary_rawcx(
    _: *mut JSContext,
    _proxy: RawHandleObject,
    is_ordinary: *mut bool,
    _proto: RawMutableHandleObject,
) -> bool {
    // We have a custom `[[GetPrototypeOf]]` for security, so this is NOT an ordinary object.
    *is_ordinary = false;
    true
}

/// Implementation of `[[SetPrototypeOf]]` for [`Location`] and [`WindowProxy`].
///
/// These objects have "immutable prototypes" or specific restrictions on changing
/// their prototype to maintain security and consistency.
///
/// # Safety
/// `result` must point to a valid, non-null ObjectOpResult.
pub(crate) unsafe extern "C" fn maybe_cross_origin_set_prototype_rawcx(
    cx: *mut JSContext,
    proxy: RawHandleObject,
    proto: RawHandleObject,
    result: *mut ObjectOpResult,
) -> bool {
    // Retrieve the current prototype.
    rooted!(in(cx) let mut current = ptr::null_mut::<JSObject>());
    if !jsapi::GetObjectProto(cx, proxy, current.handle_mut().into()) {
        return false;
    }

    // If the new prototype is the same as the current one, it's a no-op (success).
    if proto.get() == current.get() {
        (*result).code_ = 0 /* OkCode */;
        return true;
    }

    // Otherwise, prevent changing the prototype to maintain the "immutable prototype" invariant.
    (*result).code_ = JSErrNum::JSMSG_CANT_SET_PROTO as usize;
    true
}

/// Extracts the getter function from a property descriptor.
pub(crate) fn get_getter_object(d: &PropertyDescriptor, out: RawMutableHandleObject) {
    if d.hasGetter_() {
        out.set(d.getter_);
    }
}

/// Extracts the setter function from a property descriptor.
pub(crate) fn get_setter_object(d: &PropertyDescriptor, out: RawMutableHandleObject) {
    if d.hasSetter_() {
        out.set(d.setter_);
    }
}

/// Checks if the descriptor represents an accessor (getter/setter) property.
pub(crate) fn is_accessor_descriptor(d: &PropertyDescriptor) -> bool {
    d.hasSetter_() || d.hasGetter_()
}

/// Checks if the descriptor represents a data property.
pub(crate) fn is_data_descriptor(d: &PropertyDescriptor) -> bool {
    d.hasWritable_() || d.hasValue_()
}

/// Determines if a cross-origin object has a specific own property.
///
/// # Safety
/// `bp` must point to a valid, non-null bool.
pub(crate) unsafe fn cross_origin_has_own(
    cx: SafeJSContext,
    _proxy: RawHandleObject,
    cross_origin_properties: &'static CrossOriginProperties,
    id: RawHandleId,
    bp: *mut bool,
) -> bool {
    // Check if the ID matches any of the names in the cross-origin allowlist.
    *bp = jsid_to_string(*cx, Handle::from_raw(id)).is_some_and(|key| {
        cross_origin_properties.keys().any(|defined_key| {
            let defined_key = CStr::from_ptr(defined_key);
            defined_key.to_bytes() == key.as_bytes()
        })
    });

    true
}

/// Helper to get an own property descriptor for a cross-origin object.
///
/// It uses a "holder" object that contains the actual descriptors for the
/// allowlisted cross-origin properties.
pub(crate) fn cross_origin_get_own_property_helper(
    cx: SafeJSContext,
    proxy: RawHandleObject,
    cross_origin_properties: &'static CrossOriginProperties,
    id: RawHandleId,
    desc: RawMutableHandle<PropertyDescriptor>,
    is_none: &mut bool,
) -> bool {
    rooted!(in(*cx) let mut holder = ptr::null_mut::<JSObject>());

    // Ensure we have a holder object populated with the correct cross-origin members.
    ensure_cross_origin_property_holder(
        cx,
        proxy,
        cross_origin_properties,
        holder.handle_mut().into(),
    );

    // Delegate the descriptor lookup to the holder object.
    unsafe { JS_GetOwnPropertyDescriptorById(*cx, holder.handle().into(), id, desc, is_none) }
}

/// Symbols that are always allowed to be accessed cross-origin.
const ALLOWLISTED_SYMBOL_CODES: &[SymbolCode] = &[
    SymbolCode::toStringTag,
    SymbolCode::hasInstance,
    SymbolCode::isConcatSpreadable,
];

/// Verifies if a property ID is allowlisted for cross-origin access.
pub(crate) fn is_cross_origin_allowlisted_prop(cx: SafeJSContext, id: RawHandleId) -> bool {
    unsafe {
        // The "then" property is special-cased for Promise interoperability.
        if jsid_to_string(*cx, Handle::from_raw(id)).is_some_and(|st| st == "then") {
            return true;
        }

        rooted!(in(*cx) let mut allowed_id: jsid);
        ALLOWLISTED_SYMBOL_CODES.iter().any(|&allowed_code| {
            allowed_id.set(SymbolId(GetWellKnownSymbol(*cx, allowed_code)));
            // Compare IDs referentially for well-known symbols.
            allowed_id.get().asBits_ == id.asBits_
        })
    }
}

/// Appends universal cross-origin allowlisted keys to a vector of IDs.
fn append_cross_origin_allowlisted_prop_keys(cx: SafeJSContext, props: RawMutableHandleIdVector) {
    unsafe {
        rooted!(in(*cx) let mut id: jsid);

        // Add "then"
        let jsstring = JS_AtomizeAndPinString(*cx, c"then".as_ptr());
        rooted!(in(*cx) let rooted = jsstring);
        RUST_INTERNED_STRING_TO_JSID(*cx, rooted.handle().get(), id.handle_mut());
        AppendToIdVector(props, id.handle());

        // Add allowlisted symbols.
        for &allowed_code in ALLOWLISTED_SYMBOL_CODES.iter() {
            id.set(SymbolId(GetWellKnownSymbol(*cx, allowed_code)));
            AppendToIdVector(props, id.handle());
        }
    }
}

/// Lazily creates and returns a "holder" object for cross-origin properties.
///
/// The holder is an ordinary object defined with the specific attributes and methods
/// that a cross-origin proxy is allowed to expose.
fn ensure_cross_origin_property_holder(
    cx: SafeJSContext,
    _proxy: RawHandleObject,
    cross_origin_properties: &'static CrossOriginProperties,
    out_holder: RawMutableHandleObject,
) -> bool {
    // TODO: Ideally, this holder should be cached in a slot on the proxy itself.
    // Currently, it's recreated on every access which is inefficient.

    unsafe {
        out_holder.set(jsapi::JS_NewObjectWithGivenProto(
            *cx,
            ptr::null_mut(),
            RawHandleObject::null(),
        ));

        if out_holder.get().is_null() ||
            !jsapi::JS_DefineProperties(
                *cx,
                out_holder.handle(),
                cross_origin_properties.attributes.as_ptr(),
            ) ||
            !jsapi::JS_DefineFunctions(
                *cx,
                out_holder.handle(),
                cross_origin_properties.methods.as_ptr(),
            )
        {
            return false;
        }
    }

    true
}

/// Throws a SecurityError DOMException when an unauthorized cross-origin access is attempted.
///
/// This is used to implement the "Throw a SecurityError" operation in the HTML spec.
pub(crate) fn report_cross_origin_denial<D: DomTypes>(
    cx: SafeJSContext,
    id: RawHandleId,
    access: &str,
) -> bool {
    let source;
    let js_is_exception_pending;
    let mut global: Option<DomRoot<D::GlobalScope>> = None;
    let in_realm_proof = AlreadyInRealm::assert_for_cx(cx);
    
    unsafe {
        source = id_to_source(cx, id);
        js_is_exception_pending = JS_IsExceptionPending(*cx);
        // Only attempt to throw if there isn't already a pending exception.
        if !js_is_exception_pending {
            global = Some(D::GlobalScope::from_context(
                *cx,
                InRealm::Already(&in_realm_proof),
            ));
        }
    }
    
    debug!(
        "permission denied to {} property {} on cross-origin object",
        access,
        source.as_deref().unwrap_or("< error >"),
    );
    
    if !js_is_exception_pending && global.is_some() {
        <D as DomHelpers<D>>::throw_dom_exception(
            cx,
            &(global.unwrap()),
            Error::Security,
            CanGc::note(),
        );
    }
    false
}

/// Implements the `[[Set]]` trap for potentially cross-origin objects.
///
/// If the object is same-origin, it falls back to ordinary property setting logic.
/// If it's cross-origin, it enforces strict cross-origin setting rules.
pub(crate) unsafe extern "C" fn maybe_cross_origin_set_rawcx<D: DomTypes>(
    cx: *mut JSContext,
    proxy: RawHandleObject,
    id: RawHandleId,
    v: RawHandleValue,
    receiver: RawHandleValue,
    result: *mut ObjectOpResult,
) -> bool {
    let cx = SafeJSContext::from_ptr(cx);

    // Security check: Determine if we can treat this as a standard local object.
    if !<D as DomHelpers<D>>::is_platform_object_same_origin(cx, proxy) {
        return cross_origin_set::<D>(cx, proxy, id, v, receiver, result);
    }

    // Enter the Realm of the proxy to perform ordinary operations.
    let _ac = JSAutoRealm::new(*cx, proxy.get());

    // OrdinarySet implementation.
    rooted!(in(*cx) let mut own_desc = PropertyDescriptor::default());
    let mut is_none = false;
    if !InvokeGetOwnPropertyDescriptor(
        GetProxyHandler(*proxy),
        *cx,
        proxy,
        id,
        own_desc.handle_mut().into(),
        &mut is_none,
    ) {
        return false;
    }

    // Delegate to the specialized setter that ignores named getters (per spec).
    js::jsapi::SetPropertyIgnoringNamedGetter(
        *cx,
        proxy,
        id,
        v,
        receiver,
        own_desc.handle().into(),
        result,
    )
}

/// Implements the `[[GetPrototypeOf]]` trap with cross-origin awareness.
///
/// Same-origin proxies return their real prototype, while cross-origin ones return null.
pub(crate) unsafe fn maybe_cross_origin_get_prototype<D: DomTypes>(
    cx: SafeJSContext,
    proxy: RawHandleObject,
    get_proto_object: unsafe fn(cx: SafeJSContext, global: HandleObject, rval: MutableHandleObject),
    proto: RawMutableHandleObject,
) -> bool {
    if <D as DomHelpers<D>>::is_platform_object_same_origin(cx, proxy) {
        let ac = JSAutoRealm::new(*cx, proxy.get());
        let global = D::GlobalScope::from_context(*cx, InRealm::Entered(&ac));
        get_proto_object(
            cx,
            global.reflector().get_jsobject(),
            MutableHandleObject::from_raw(proto),
        );
        return !proto.is_null();
    }

    // Cross-origin objects must hide their prototype.
    proto.set(ptr::null_mut());
    true
}

/// Implementation of the `CrossOriginGet` operation from the HTML spec.
///
/// This enforces that only allowlisted properties are readable cross-origin.
pub(crate) unsafe fn cross_origin_get<D: DomTypes>(
    cx: SafeJSContext,
    proxy: RawHandleObject,
    receiver: RawHandleValue,
    id: RawHandleId,
    vp: RawMutableHandleValue,
) -> bool {
    // 1. Get the property descriptor.
    rooted!(in(*cx) let mut descriptor = PropertyDescriptor::default());
    let mut is_none = false;
    if !InvokeGetOwnPropertyDescriptor(
        GetProxyHandler(*proxy),
        *cx,
        proxy,
        id,
        descriptor.handle_mut().into(),
        &mut is_none,
    ) {
        return false;
    }

    assert!(!is_none, "Descriptor lookup failed unexpectedly.");

    // 2. If it's a data descriptor, return its value directly.
    if is_data_descriptor(&descriptor) {
        vp.set(descriptor.value_);
        return true;
    }

    // 3. If it's an accessor, call the getter if authorized.
    assert!(is_accessor_descriptor(&descriptor));

    rooted!(in(*cx) let mut getter = ptr::null_mut::<JSObject>());
    get_getter_object(&descriptor, getter.handle_mut().into());
    if getter.get().is_null() {
        return report_cross_origin_denial::<D>(cx, id, "get");
    }

    rooted!(in(*cx) let mut getter_jsval = UndefinedValue());
    getter.get().to_jsval(*cx, getter_jsval.handle_mut());

    // Call the getter function with the receiver as 'this'.
    jsapi::Call(
        *cx,
        receiver,
        getter_jsval.handle().into(),
        &jsapi::HandleValueArray::empty(),
        vp,
    )
}

/// Implementation of the `CrossOriginSet` operation from the HTML spec.
///
/// Only allowlisted properties with defined setters can be modified cross-origin.
pub(crate) unsafe fn cross_origin_set<D: DomTypes>(
    cx: SafeJSContext,
    proxy: RawHandleObject,
    id: RawHandleId,
    v: RawHandleValue,
    receiver: RawHandleValue,
    result: *mut ObjectOpResult,
) -> bool {
    rooted!(in(*cx) let mut descriptor = PropertyDescriptor::default());
    let mut is_none = false;
    if !InvokeGetOwnPropertyDescriptor(
        GetProxyHandler(*proxy),
        *cx,
        proxy,
        id,
        descriptor.handle_mut().into(),
        &mut is_none,
    ) {
        return false;
    }

    assert!(!is_none, "Descriptor lookup failed unexpectedly.");

    rooted!(in(*cx) let mut setter = ptr::null_mut::<JSObject>());
    get_setter_object(&descriptor, setter.handle_mut().into());
    if setter.get().is_null() {
        // No authorized setter found for this cross-origin access.
        return report_cross_origin_denial::<D>(cx, id, "set");
    }

    rooted!(in(*cx) let mut setter_jsval = UndefinedValue());
    setter.get().to_jsval(*cx, setter_jsval.handle_mut());

    // Execute the setter.
    rooted!(in(*cx) let mut ignored = UndefinedValue());
    if !jsapi::Call(
        *cx,
        receiver,
        setter_jsval.handle().into(),
        &jsapi::HandleValueArray {
            length_: 1,
            elements_: v.ptr,
        },
        ignored.handle_mut().into(),
    ) {
        return false;
    }

    (*result).code_ = 0 /* OkCode */;
    true
}

/// Fallback logic for cross-origin property access when a property is not found.
///
/// This handles properties like `then` or symbols that should be treated as
/// existing but undefined for security/interoperability reasons.
pub(crate) unsafe fn cross_origin_property_fallback<D: DomTypes>(
    cx: SafeJSContext,
    _proxy: RawHandleObject,
    id: RawHandleId,
    desc: RawMutableHandle<PropertyDescriptor>,
    is_none: &mut bool,
) -> bool {
    assert!(*is_none, "Fallback called even though property was found.");

    // If the property is in the allowlist but not physically present, return a default descriptor.
    if is_cross_origin_allowlisted_prop(cx, id) {
        set_property_descriptor(
            MutableHandle::from_raw(desc),
            HandleValue::undefined(),
            jsapi::JSPROP_READONLY as u32,
            is_none,
        );
        return true;
    }

    // Otherwise, deny the access entirely.
    report_cross_origin_denial::<D>(cx, id, "access")
}
