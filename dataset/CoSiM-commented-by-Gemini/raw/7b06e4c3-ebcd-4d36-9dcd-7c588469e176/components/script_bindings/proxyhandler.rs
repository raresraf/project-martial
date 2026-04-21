/**
 * @file proxyhandler.rs
 * @brief Infrastructure for JSAPI proxy handlers and cross-origin security in Servo.
 * 
 * Architectural Intent: Implements the low-level bindings between the SpiderMonkey JS engine 
 * and the DOM proxy system. It enforces W3C cross-origin security boundaries by intercepting 
 * property access traps ([[Get]], [[Set]], [[DefineProperty]]) and validating them against 
 * origin-based policies.
 * 
 * Domain-Awareness: Manages "expando" objects (per-proxy dynamic property storage) and 
 * implements spec-compliant behavior for potentially cross-origin objects like 'Window' or 'Location'.
 */

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Utilities for the implementation of JSAPI proxy handlers.

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
use crate::script_runtime::{CanGc, JSContext as SafeJSContext};
use crate::str::DOMString;
use crate::utils::delete_property_by_id;

/**
 * @brief Determine if a property id shadows any existing properties for this proxy.
 * Functional Utility: Optimizes property lookups by signaling if the engine needs 
 * to check the proxy's prototype chain or if the property is definitely on the expando.
 */
pub(crate) unsafe extern "C" fn shadow_check_callback(
    cx: *mut JSContext,
    object: RawHandleObject,
    id: RawHandleId,
) -> DOMProxyShadowsResult {
    // TODO: support OverrideBuiltins when #12978 is fixed.

    rooted!(in(cx) let mut expando = ptr::null_mut::<JSObject>());
    get_expando_object(object, expando.handle_mut());
    
    /**
     * Block Logic: Expando existence check.
     * Invariant: If no expando object is attached to the proxy, no shadowing occurs.
     */
    if !expando.get().is_null() {
        let mut has_own = false;
        let raw_id = Handle::from_raw(id);

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

/// Initialize the infrastructure for DOM proxy objects.
pub fn init() {
    unsafe {
        SetDOMProxyInformation(
            GetProxyHandlerFamily(),
            Some(shadow_check_callback),
            ptr::null(),
        );
    }
}

/**
 * @brief Defines a property on the proxy's expando object.
 * Functional Utility: Intercepts 'Object.defineProperty' to ensure properties 
 * are stored in the side-loaded expando rather than the proxy host.
 */
pub(crate) unsafe extern "C" fn define_property(
    cx: *mut JSContext,
    proxy: RawHandleObject,
    id: RawHandleId,
    desc: RawHandle<PropertyDescriptor>,
    result: *mut ObjectOpResult,
) -> bool {
    rooted!(in(cx) let mut expando = ptr::null_mut::<JSObject>());
    ensure_expando_object(cx, proxy, expando.handle_mut());
    JS_DefinePropertyById(cx, expando.handle().into(), id, desc, result)
}

/**
 * @brief Deletes a property from the proxy's expando.
 */
pub(crate) unsafe extern "C" fn delete(
    cx: *mut JSContext,
    proxy: RawHandleObject,
    id: RawHandleId,
    bp: *mut ObjectOpResult,
) -> bool {
    rooted!(in(cx) let mut expando = ptr::null_mut::<JSObject>());
    get_expando_object(proxy, expando.handle_mut());
    
    /**
     * Block Logic: Pre-condition check for property existence.
     * Invariant: Returns success immediately if no expando exists, as there is nothing to delete.
     */
    if expando.is_null() {
        (*bp).code_ = 0 /* OkCode */;
        return true;
    }

    delete_property_by_id(cx, expando.handle(), Handle::from_raw(id), bp)
}

/// Controls whether the Extensible bit can be changed
pub(crate) unsafe extern "C" fn prevent_extensions(
    _cx: *mut JSContext,
    _proxy: RawHandleObject,
    result: *mut ObjectOpResult,
) -> bool {
    // Architectural Invariant: DOM proxies are always extensible.
    (*result).code_ = JSErrNum::JSMSG_CANT_PREVENT_EXTENSIONS as ::libc::uintptr_t;
    true
}

/// Reports whether the object is Extensible
pub(crate) unsafe extern "C" fn is_extensible(
    _cx: *mut JSContext,
    _proxy: RawHandleObject,
    succeeded: *mut bool,
) -> bool {
    *succeeded = true;
    true
}

/**
 * @brief Retrieves the static prototype of the proxy host.
 */
pub(crate) unsafe extern "C" fn get_prototype_if_ordinary(
    _: *mut JSContext,
    proxy: RawHandleObject,
    is_ordinary: *mut bool,
    proto: RawMutableHandleObject,
) -> bool {
    *is_ordinary = true;
    proto.set(GetStaticPrototype(proxy.get()));
    true
}

/**
 * @brief Retrieves the private expando object from the proxy's private slot.
 */
pub(crate) fn get_expando_object(obj: RawHandleObject, mut expando: MutableHandleObject) {
    unsafe {
        assert!(is_dom_proxy(obj.get()));
        let val = &mut UndefinedValue();
        GetProxyPrivate(obj.get(), val);
        expando.set(if val.is_undefined() {
            ptr::null_mut()
        } else {
            val.to_object()
        });
    }
}

/**
 * @brief Lazy-initializes the expando object if it doesn't exist.
 */
pub(crate) unsafe fn ensure_expando_object(
    cx: *mut JSContext,
    obj: RawHandleObject,
    mut expando: MutableHandleObject,
) {
    assert!(is_dom_proxy(obj.get()));
    get_expando_object(obj, expando.reborrow());
    
    /**
     * Block Logic: Allocation of new JSObject for expandos.
     * Invariant: New object is successfully created and linked to the proxy's private slot.
     */
    if expando.is_null() {
        expando.set(JS_NewObjectWithGivenProto(
            cx,
            ptr::null_mut(),
            HandleObject::null(),
        ));
        assert!(!expando.is_null());

        SetProxyPrivate(obj.get(), &ObjectValue(expando.get()));
    }
}

// ... (Rest of utility implementation) ...

/**
 * @brief Implementation of the [[Set]] trap for potentially cross-origin objects.
 * Algorithm: Checks origin affinity. If same-origin, performs ordinary set. 
 * If cross-origin, delegates to specific security-filtered set logic.
 */
pub(crate) unsafe extern "C" fn maybe_cross_origin_set_rawcx<D: DomTypes>(
    cx: *mut JSContext,
    proxy: RawHandleObject,
    id: RawHandleId,
    v: RawHandleValue,
    receiver: RawHandleValue,
    result: *mut ObjectOpResult,
) -> bool {
    let cx = SafeJSContext::from_ptr(cx);

    /**
     * Block Logic: Origin validation barrier.
     */
    if !<D as DomHelpers<D>>::is_platform_object_same_origin(cx, proxy) {
        return cross_origin_set::<D>(cx, proxy, id, v, receiver, result);
    }

    // Safe to enter the Realm of proxy now.
    let _ac = JSAutoRealm::new(*cx, proxy.get());

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

/**
 * @brief Implementation of [[CrossOriginGet]] as per HTML spec.
 * Logic: Validates if the property is allowlisted (e.g., 'window.location'). 
 * Throws SecurityError if access to restricted data is attempted.
 */
pub(crate) unsafe fn cross_origin_get<D: DomTypes>(
    cx: SafeJSContext,
    proxy: RawHandleObject,
    receiver: RawHandleValue,
    id: RawHandleId,
    vp: RawMutableHandleValue,
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

    assert!(
        !is_none,
        "Callees should throw in all cases when they are not finding \
        a property decriptor"
    );

    if is_data_descriptor(&descriptor) {
        vp.set(descriptor.value_);
        return true;
    }

    assert!(is_accessor_descriptor(&descriptor));

    rooted!(in(*cx) let mut getter = ptr::null_mut::<JSObject>());
    get_getter_object(&descriptor, getter.handle_mut().into());
    
    /**
     * Block Logic: Security check for cross-origin getters.
     * Invariant: Access denied if the descriptor lacks a valid getter function.
     */
    if getter.get().is_null() {
        return report_cross_origin_denial::<D>(cx, id, "get");
    }

    rooted!(in(*cx) let mut getter_jsval = UndefinedValue());
    getter.get().to_jsval(*cx, getter_jsval.handle_mut());

    jsapi::Call(
        *cx,
        receiver,
        getter_jsval.handle().into(),
        &jsapi::HandleValueArray::empty(),
        vp,
    )
}
