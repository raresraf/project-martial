/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Native representation of JS Promise values.
//!
//! This implementation differs from the traditional Rust DOM object, because the reflector
//! is provided by SpiderMonkey and has no knowledge of an associated native representation
//! (ie. dom::Promise). This means that native instances use native reference counting (Rc)
//! to ensure that no memory is leaked, which means that there can be multiple instances of
//! native Promise values that refer to the same JS value yet are distinct native objects
//! (ie. address equality for the native objects is meaningless).

use std::ptr;
use std::rc::Rc;

use dom_struct::dom_struct;
use js::conversions::ToJSValConvertible;
use js::jsapi::{
    AddRawValueRoot, CallArgs, GetFunctionNativeReserved, Heap, JSAutoRealm, JSContext, JSObject,
    JS_ClearPendingException, JS_GetFunctionObject, JS_NewFunction, NewFunctionWithReserved,
    PromiseState, PromiseUserInputEventHandlingState, RemoveRawValueRoot,
    SetFunctionNativeReserved,
};
use js::jsval::{Int32Value, JSVal, ObjectValue, UndefinedValue};
use js::rust::wrappers::{
    AddPromiseReactions, CallOriginalPromiseReject, CallOriginalPromiseResolve,
    GetPromiseIsHandled, GetPromiseState, IsPromiseObject, NewPromiseObject, RejectPromise,
    ResolvePromise, SetAnyPromiseIsHandled, SetPromiseUserInputEventHandlingState,
};
use js::rust::{HandleObject, HandleValue, MutableHandleObject, Runtime};

use crate::dom::bindings::conversions::root_from_object;
use crate::dom::bindings::error::{Error, ErrorToJsval};
use crate::dom::bindings::reflector::{DomGlobal, DomObject, MutDomObject, Reflector};
use crate::dom::bindings::settings_stack::AutoEntryScript;
use crate::dom::globalscope::GlobalScope;
use crate::dom::promisenativehandler::PromiseNativeHandler;
use crate::realms::{enter_realm, AlreadyInRealm, InRealm};
use crate::script_runtime::{CanGc, JSContext as SafeJSContext};
use crate::script_thread::ScriptThread;

/// Native representation of a JavaScript Promise object.
///
/// Functional Utility: This struct wraps a SpiderMonkey `JSObject` that represents
/// a Promise, allowing Rust code to interact with it using native Rust types and
/// patterns while managing the underlying JavaScript object's lifecycle.
#[dom_struct]
#[cfg_attr(crown, crown::unrooted_must_root_lint::allow_unrooted_in_rc)]
pub(crate) struct Promise {
    /// The `Reflector` provides the bridge between the native Rust object and its JS counterpart.
    reflector: Reflector,
    /// Since Promise values are natively reference counted without the knowledge of
    /// the SpiderMonkey GC, an explicit root for the reflector is stored while any
    /// native instance exists. This ensures that the reflector will never be GCed
    /// while native code could still interact with its native representation.
    /// Memory Management: This `Heap` root ensures the underlying JS `JSVal` (which is an `ObjectValue`)
    /// is protected from garbage collection as long as the native `Promise` instance exists.
    #[ignore_malloc_size_of = "SM handles JS values"]
    permanent_js_root: Heap<JSVal>,
}

/// Private helper trait to enable adding new methods to `Rc<Promise>`.
/// Functional Utility: Provides an extension point for `Rc<Promise>` to
/// add initialization logic that requires the `Rc` context.
trait PromiseHelper {
    /// Initializes the internal state of the `Promise` after its creation.
    /// Functional Utility: Sets up the permanent JavaScript root for the
    /// associated `JSObject` to protect it from garbage collection.
    /// Precondition: The `Promise`'s `reflector` must already be initialized with a valid `JSObject`.
    /// Postcondition: A raw value root is added in SpiderMonkey for the promise's `JSObject`.
    /// @param cx The current JavaScript context.
    fn initialize(&self, cx: SafeJSContext);
}

impl PromiseHelper for Rc<Promise> {
    #[allow(unsafe_code)]
    fn initialize(&self, cx: SafeJSContext) {
        let obj = self.reflector().get_jsobject();
        self.permanent_js_root.set(ObjectValue(*obj));
        unsafe {
            // Memory Management: Add a raw value root to prevent the JS Promise object from being GC'd.
            assert!(AddRawValueRoot(
                *cx,
                self.permanent_js_root.get_unsafe(),
                c"Promise::root".as_ptr(),
            ));
        }
    }
}

impl Drop for Promise {
    /// Implements the `Drop` trait for `Promise`.
    /// Functional Utility: Ensures that the permanent JavaScript root is removed
    /// when the native `Promise` instance is dropped, allowing the `JSObject`
    /// to be garbage collected if no other references exist.
    /// Precondition: The `permanent_js_root` must have been successfully added.
    /// Postcondition: The raw value root in SpiderMonkey for the promise's `JSObject` is removed.
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        unsafe {
            let object = self.permanent_js_root.get().to_object();
            // Invariant: The object pointer should not be null when dropping if it was rooted.
            assert!(!object.is_null());
            if let Some(cx) = Runtime::get() {
                // Memory Management: Remove the raw value root.
                RemoveRawValueRoot(cx.as_ptr(), self.permanent_js_root.get_unsafe());
            }
        }
    }
}

impl Promise {
    /// Creates a new `Promise` in the current `GlobalScope`.
    /// Functional Utility: Provides a high-level API to construct a new JavaScript Promise
    /// that can be managed from Rust.
    /// @param global The `GlobalScope` in which the Promise should be created.
    /// @param can_gc A `CanGc` token.
    /// @return An `Rc<Promise>` representing the newly created Promise.
    pub(crate) fn new(global: &GlobalScope, can_gc: CanGc) -> Rc<Promise> {
        let realm = enter_realm(global);
        let comp = InRealm::Entered(&realm);
        Promise::new_in_current_realm(comp, can_gc)
    }

    /// Creates a new `Promise` within the current JavaScript realm.
    /// Functional Utility: Internal helper to create the underlying JavaScript Promise
    /// object and wrap it in a native `Promise` instance.
    /// Precondition: A JavaScript context must be active and a realm entered.
    /// @param _comp The current `InRealm` context.
    /// @param can_gc A `CanGc` token.
    /// @return An `Rc<Promise>` representing the newly created Promise.
    pub(crate) fn new_in_current_realm(_comp: InRealm, can_gc: CanGc) -> Rc<Promise> {
        let cx = GlobalScope::get_cx();
        rooted!(in(*cx) let mut obj = ptr::null_mut::<JSObject>());
        Promise::create_js_promise(cx, obj.handle_mut(), can_gc);
        Promise::new_with_js_promise(obj.handle(), cx)
    }

    /// Duplicates the native `Promise` reference, pointing to the same JavaScript Promise.
    /// Functional Utility: Allows multiple native Rust `Rc<Promise>` instances to share
    /// ownership of a single underlying JavaScript Promise.
    /// @return An `Rc<Promise>` that refers to the same JavaScript Promise object.
    #[allow(unsafe_code)]
    pub(crate) fn duplicate(&self) -> Rc<Promise> {
        let cx = GlobalScope::get_cx();
        Promise::new_with_js_promise(self.reflector().get_jsobject(), cx)
    }

    /// Creates a new `Promise` instance from an existing JavaScript Promise `JSObject`.
    /// Functional Utility: Internal constructor used to wrap an already existing
    /// JavaScript Promise object in a Rust `Promise` instance and establish its GC rooting.
    /// Precondition: `obj` must be a valid `HandleObject` pointing to a JavaScript Promise.
    /// Postcondition: A new `Rc<Promise>` instance is returned, with the JS object rooted.
    /// @param obj A `HandleObject` to the existing JavaScript Promise.
    /// @param cx The current JavaScript context.
    /// @return An `Rc<Promise>` representing the JavaScript Promise.
    #[allow(unsafe_code)]
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn new_with_js_promise(obj: HandleObject, cx: SafeJSContext) -> Rc<Promise> {
        unsafe {
            // Invariant: Ensure the provided `HandleObject` is indeed a Promise object.
            assert!(IsPromiseObject(obj));
            let promise = Promise {
                reflector: Reflector::new(),
                permanent_js_root: Heap::default(),
            };
            let promise = Rc::new(promise);
            promise.init_reflector(obj.get()); // Initialize the reflector with the JS object.
            promise.initialize(cx); // Set up GC rooting.
            promise
        }
    }

    /// Creates the underlying JavaScript Promise object.
    /// Functional Utility: This low-level function directly interacts with SpiderMonkey's
    /// JS API to instantiate a new Promise object. It uses a `do_nothing_promise_executor`
    /// as the initial executor, reflecting that the resolution/rejection will be handled
    /// directly by native code.
    /// Precondition: `cx` is a valid JavaScript context and `obj` is a mutable handle for the new JS object.
    /// Postcondition: `obj` will point to a newly created JavaScript Promise object, with its
    ///                user input event handling state set.
    /// @param cx The current JavaScript context.
    /// @param mut obj A mutable handle where the new JS Promise object will be stored.
    /// @param _can_gc A `CanGc` token (may trigger GC internally).
    #[allow(unsafe_code)]
    // The apparently-unused CanGc parameter reflects the fact that the JS API calls
    // like JS_NewFunction can trigger a GC.
    fn create_js_promise(cx: SafeJSContext, mut obj: MutableHandleObject, _can_gc: CanGc) {
        unsafe {
            let do_nothing_func = JS_NewFunction(
                *cx,
                Some(do_nothing_promise_executor), // Custom executor function.
                /* nargs = */ 2, // resolve, reject arguments.
                /* flags = */ 0,
                ptr::null(),
            );
            // Invariant: The new function object must not be null.
            assert!(!do_nothing_func.is_null());
            rooted!(in(*cx) let do_nothing_obj = JS_GetFunctionObject(do_nothing_func));
            assert!(!do_nothing_obj.is_null());
            // Create a new Promise object using the executor.
            obj.set(NewPromiseObject(*cx, do_nothing_obj.handle()));
            assert!(!obj.is_null());
            // Set the promise's user input event handling state.
            let is_user_interacting = if ScriptThread::is_user_interacting() {
                PromiseUserInputEventHandlingState::HadUserInteractionAtCreation
            } else {
                PromiseUserInputEventHandlingState::DidntHaveUserInteractionAtCreation
            };
            SetPromiseUserInputEventHandlingState(obj.handle(), is_user_interacting);
        }
    }

    /// Creates a new `Promise` that is immediately resolved with a given value.
    /// Functional Utility: Simplifies the creation of already-resolved promises,
    /// common in asynchronous programming patterns.
    /// Precondition: `value` is convertible to a `JSVal`.
    /// Postcondition: A new `Rc<Promise>` is returned representing a resolved Promise.
    /// @param global The `GlobalScope` context.
    /// @param cx The JavaScript context.
    /// @param value The value to resolve the Promise with.
    /// @param _can_gc A `CanGc` token.
    /// @return An `Rc<Promise>` representing the resolved Promise.
    #[allow(unsafe_code)]
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn new_resolved(
        global: &GlobalScope,
        cx: SafeJSContext,
        value: impl ToJSValConvertible,
        _can_gc: CanGc,
    ) -> Rc<Promise> {
        let _ac = JSAutoRealm::new(*cx, global.reflector().get_jsobject().get());
        unsafe {
            rooted!(in(*cx) let mut rval = UndefinedValue());
            value.to_jsval(*cx, rval.handle_mut()); // Convert Rust value to JSVal.
            // Call the original Promise.resolve to create a resolved promise.
            rooted!(in(*cx) let p = CallOriginalPromiseResolve(*cx, rval.handle()));
            assert!(!p.handle().is_null());
            Promise::new_with_js_promise(p.handle(), cx)
        }
    }

    /// Creates a new `Promise` that is immediately rejected with a given value.
    /// Functional Utility: Simplifies the creation of already-rejected promises.
    /// Precondition: `value` is convertible to a `JSVal`.
    /// Postcondition: A new `Rc<Promise>` is returned representing a rejected Promise.
    /// @param global The `GlobalScope` context.
    /// @param cx The JavaScript context.
    /// @param value The value to reject the Promise with.
    /// @param _can_gc A `CanGc` token.
    /// @return An `Rc<Promise>` representing the rejected Promise.
    #[allow(unsafe_code)]
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn new_rejected(
        global: &GlobalScope,
        cx: SafeJSContext,
        value: impl ToJSValConvertible,
        _can_gc: CanGc,
    ) -> Rc<Promise> {
        let _ac = JSAutoRealm::new(*cx, global.reflector().get_jsobject().get());
        unsafe {
            rooted!(in(*cx) let mut rval = UndefinedValue());
            value.to_jsval(*cx, rval.handle_mut()); // Convert Rust value to JSVal.
            // Call the original Promise.reject to create a rejected promise.
            rooted!(in(*cx) let p = CallOriginalPromiseReject(*cx, rval.handle()));
            assert!(!p.handle().is_null());
            Promise::new_with_js_promise(p.handle(), cx)
        }
    }

    /// Resolves the JavaScript Promise with a native Rust value.
    /// Functional Utility: Provides a type-safe way to resolve the underlying
    /// JavaScript Promise from Rust code.
    /// Precondition: The Promise must be in a pending state.
    /// Postcondition: The Promise transitions to a fulfilled state with `val`.
    /// @param val The native value to resolve the Promise with (must be `ToJSValConvertible`).
    /// @param can_gc A `CanGc` token.
    pub(crate) fn resolve_native<T>(&self, val: &T, can_gc: CanGc)
    where
        T: ToJSValConvertible,
    {
        let cx = GlobalScope::get_cx();
        let _ac = enter_realm(self);
        rooted!(in(*cx) let mut v = UndefinedValue());
        unsafe {
            val.to_jsval(*cx, v.handle_mut()); // Convert native value to JSVal.
        }
        self.resolve(cx, v.handle(), can_gc);
    }

    /// Resolves the JavaScript Promise with a given JavaScript `HandleValue`.
    /// Functional Utility: Low-level method to directly resolve the underlying
    /// JavaScript Promise object.
    /// Precondition: The Promise must be in a pending state.
    /// Postcondition: The Promise transitions to a fulfilled state with `value`.
    /// @param cx The JavaScript context.
    /// @param value The JavaScript `HandleValue` to resolve the Promise with.
    /// @param _can_gc A `CanGc` token.
    #[allow(unsafe_code)]
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn resolve(&self, cx: SafeJSContext, value: HandleValue, _can_gc: CanGc) {
        unsafe {
            // Attempt to resolve the promise; clear any pending exception if it fails.
            if !ResolvePromise(*cx, self.promise_obj(), value) {
                JS_ClearPendingException(*cx);
            }
        }
    }

    /// Rejects the JavaScript Promise with a native Rust value.
    /// Functional Utility: Provides a type-safe way to reject the underlying
    /// JavaScript Promise from Rust code.
    /// Precondition: The Promise must be in a pending state.
    /// Postcondition: The Promise transitions to a rejected state with `val`.
    /// @param val The native value to reject the Promise with (must be `ToJSValConvertible`).
    pub(crate) fn reject_native<T>(&self) -> Rc<Promise> {
        let cx = GlobalScope::get_cx();
        let _ac = enter_realm(self);
        rooted!(in(*cx) let mut v = UndefinedValue());
        unsafe {
            val.to_jsval(*cx, v.handle_mut()); // Convert native value to JSVal.
        }
        self.reject(cx, v.handle());
    }

    /// Rejects the JavaScript Promise with an `Error` object.
    /// Functional Utility: Converts a Rust `Error` into a JavaScript error
    /// and rejects the Promise with it.
    /// Precondition: The Promise must be in a pending state.
    /// Postcondition: The Promise transitions to a rejected state with the `error`.
    /// @param error The Rust `Error` to reject the Promise with.
    pub(crate) fn reject_error(&self, error: Error) {
        let cx = GlobalScope::get_cx();
        let _ac = enter_realm(self);
        rooted!(in(*cx) let mut v = UndefinedValue());
        error.to_jsval(cx, &self.global(), v.handle_mut()); // Convert Rust Error to JSVal.
        self.reject(cx, v.handle());
    }

    /// Rejects the JavaScript Promise with a given JavaScript `HandleValue`.
    /// Functional Utility: Low-level method to directly reject the underlying
    /// JavaScript Promise object.
    /// Precondition: The Promise must be in a pending state.
    /// Postcondition: The Promise transitions to a rejected state with `value`.
    /// @param cx The JavaScript context.
    /// @param value The JavaScript `HandleValue` to reject the Promise with.
    #[allow(unsafe_code)]
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn reject(&self, cx: SafeJSContext, value: HandleValue) {
        unsafe {
            // Attempt to reject the promise; clear any pending exception if it fails.
            if !RejectPromise(*cx, self.promise_obj(), value) {
                JS_ClearPendingException(*cx);
            }
        }
    }

    /// Checks if the Promise is in a fulfilled or rejected state.
    /// Functional Utility: Provides a way to query the final settlement status of the Promise.
    /// @return `true` if the Promise is fulfilled or rejected, `false` if pending.
    #[allow(unsafe_code)]
    pub(crate) fn is_fulfilled(&self) -> bool {
        let state = unsafe { GetPromiseState(self.promise_obj()) };
        matches!(state, PromiseState::Rejected | PromiseState::Fulfilled)
    }

    /// Checks if the Promise is in a rejected state.
    /// Functional Utility: Provides a way to query if the Promise has failed.
    /// @return `true` if the Promise is rejected, `false` otherwise.
    #[allow(unsafe_code)]
    pub(crate) fn is_rejected(&self) -> bool {
        let state = unsafe { GetPromiseState(self.promise_obj()) };
        matches!(state, PromiseState::Rejected)
    }

    /// Checks if the Promise is in a pending state.
    /// Functional Utility: Provides a way to query if the Promise is still unsettled.
    /// @return `true` if the Promise is pending, `false` otherwise.
    #[allow(unsafe_code)]
    pub(crate) fn is_pending(&self) -> bool {
        let state = unsafe { GetPromiseState(self.promise_obj()) };
        matches!(state, PromiseState::Pending)
    }

    /// Returns a `HandleObject` to the underlying JavaScript Promise object.
    /// Functional Utility: Provides direct access to the raw SpiderMonkey Promise object,
    /// typically for low-level JS API interactions.
    /// Precondition: The `reflector` must be initialized with a valid `JSObject`.
    /// Postcondition: The returned `HandleObject` is guaranteed to be a Promise object.
    /// @return A `HandleObject` pointing to the JavaScript Promise.
    #[allow(unsafe_code)]
    pub(crate) fn promise_obj(&self) -> HandleObject {
        let obj = self.reflector().get_jsobject();
        unsafe {
            assert!(IsPromiseObject(obj));
        }
        obj
    }

    /// Appends a native Rust promise handler (for `.then()` or `.catch()` semantics)
    /// to the JavaScript Promise.
    /// Functional Utility: Integrates Rust-defined callbacks to react to Promise
    /// resolution or rejection, allowing native code to participate in the Promise chain.
    /// Precondition: `handler` is a valid `PromiseNativeHandler`.
    /// Postcondition: The native handler functions are added as reactions to the JS Promise.
    /// @param handler The `PromiseNativeHandler` containing the Rust callbacks.
    /// @param _comp The current `InRealm` context.
    /// @param can_gc A `CanGc` token.
    #[allow(unsafe_code)]
    pub(crate) fn append_native_handler(
        &self,
        handler: &PromiseNativeHandler,
        _comp: InRealm,
        can_gc: CanGc,
    ) {
        let _ais = AutoEntryScript::new(&handler.global());
        let cx = GlobalScope::get_cx();
        // Create a JS function for the resolve callback.
        rooted!(in(*cx) let resolve_func =
                create_native_handler_function(*cx,
                                               handler.reflector().get_jsobject(),
                                               NativeHandlerTask::Resolve,
                                               can_gc));

        // Create a JS function for the reject callback.
        rooted!(in(*cx) let reject_func =
                create_native_handler_function(*cx,
                                               handler.reflector().get_jsobject(),
                                               NativeHandlerTask::Reject,
                                               can_gc));

        unsafe {
            // Add these JS functions as reactions to the underlying Promise.
            let ok = AddPromiseReactions(
                *cx,
                self.promise_obj(),
                resolve_func.handle(),
                reject_func.handle(),
            );
            // Invariant: Adding promise reactions must succeed.
            assert!(ok);
        }
    }

    /// Checks if the JavaScript Promise has been handled (i.e., has a reaction).
    /// Functional Utility: Reflects the internal state of the Promise regarding
    /// whether its completion has been observed by a `.then()` or `.catch()` handler.
    /// @return `true` if the Promise is handled, `false` otherwise.
    #[allow(unsafe_code)]
    pub(crate) fn get_promise_is_handled(&self) -> bool {
        unsafe { GetPromiseIsHandled(self.reflector().get_jsobject()) }
    }

    /// Marks the JavaScript Promise as handled.
    /// Functional Utility: Explicitly sets the Promise's internal handled flag,
    /// typically to suppress unhandled promise rejection warnings.
    /// @return `true` if the operation was successful.
    #[allow(unsafe_code)]
    pub(crate) fn set_promise_is_handled(&self) -> bool {
        let cx = GlobalScope::get_cx();
        unsafe { SetAnyPromiseIsHandled(*cx, self.reflector().get_jsobject()) }
    }
}

/// A C-style extern function acting as the executor for a newly created JavaScript Promise.
/// Functional Utility: This executor is deliberately designed to do nothing, as the
/// resolution/rejection of promises created via `Promise::new` is intended to be
/// controlled explicitly by native Rust code.
/// @param _cx The JavaScript context (unused).
/// @param argc The number of arguments (unused).
/// @param vp Pointer to the argument array (unused).
/// @return `true` on completion.
#[allow(unsafe_code)]
unsafe extern "C" fn do_nothing_promise_executor(
    _cx: *mut JSContext,
    argc: u32,
    vp: *mut JSVal,
) -> bool {
    let args = CallArgs::from_vp(vp, argc);
    *args.rval() = UndefinedValue(); // Return undefined.
    true
}

/// Slot index for storing the native handler object in a JS function's reserved slots.
const SLOT_NATIVEHANDLER: usize = 0;
/// Slot index for storing the native handler task type in a JS function's reserved slots.
const SLOT_NATIVEHANDLER_TASK: usize = 1;

/// Enum representing the type of task a native promise handler function should perform.
/// Functional Utility: Differentiates between a resolve task and a reject task,
/// allowing a single callback function to handle both based on this value.
#[derive(PartialEq)]
enum NativeHandlerTask {
    /// Indicates the task is to resolve the promise.
    Resolve = 0,
    /// Indicates the task is to reject the promise.
    Reject = 1,
}

/// A C-style extern function serving as the callback for JavaScript Promise reactions.
/// Functional Utility: This function is invoked by SpiderMonkey when a Promise it's
/// attached to resolves or rejects. It extracts the appropriate native Rust handler
/// and dispatches the corresponding `resolved_callback` or `rejected_callback`.
/// Precondition: The JS function's reserved slots must contain valid native handler and task values.
/// Postcondition: The native `PromiseNativeHandler`'s callback is executed.
/// @param cx The JavaScript context.
/// @param argc The number of arguments.
/// @param vp Pointer to the argument array.
/// @return `true` on completion.
#[allow(unsafe_code)]
unsafe extern "C" fn native_handler_callback(
    cx: *mut JSContext,
    argc: u32,
    vp: *mut JSVal,
) -> bool {
    let cx = SafeJSContext::from_ptr(cx);
    let in_realm_proof = AlreadyInRealm::assert_for_cx(cx);

    let args = CallArgs::from_vp(vp, argc);
    // Block Logic: Retrieve the native handler object from the function's reserved slot.
    rooted!(in(*cx) let v = *GetFunctionNativeReserved(args.callee(), SLOT_NATIVEHANDLER));
    // Invariant: The value in the reserved slot must be a JS object.
    assert!(v.get().is_object());

    // Root and cast the JS object back to a native `PromiseNativeHandler`.
    let handler = root_from_object::<PromiseNativeHandler>(v.to_object(), *cx)
        .expect("unexpected value for native handler in promise native handler callback");

    // Block Logic: Retrieve the task type from the function's reserved slot and dispatch.
    rooted!(in(*cx) let v = *GetFunctionNativeReserved(args.callee(), SLOT_NATIVEHANDLER_TASK));
    match v.to_int32() {
        v if v == NativeHandlerTask::Resolve as i32 => handler.resolved_callback(
            *cx,
            HandleValue::from_raw(args.get(0)), // First argument is the resolved value.
            InRealm::Already(&in_realm_proof),
            CanGc::note(),
        ),
        v if v == NativeHandlerTask::Reject as i32 => handler.rejected_callback(
            *cx,
            HandleValue::from_raw(args.get(0)), // First argument is the rejection reason.
            InRealm::Already(&in_realm_proof),
            CanGc::note(),
        ),
        _ => panic!("unexpected native handler task value"), // Invariant: Task value must be valid.
    };

    true
}

/// Creates a JavaScript function object that will invoke a native handler callback.
/// Functional Utility: Generates a bridge function in JavaScript that, when called,
/// dispatches to the Rust `native_handler_callback`, effectively connecting JS Promise
/// reactions to native Rust logic.
/// Precondition: `cx` is a valid JavaScript context, `holder` is a valid JS object, and `task` is a `NativeHandlerTask`.
/// Postcondition: A new JS function object is returned, with the native handler and task stored in its reserved slots.
/// @param cx The JavaScript context.
/// @param holder A `HandleObject` to the native handler's reflector (used for GC rooting).
/// @param task The `NativeHandlerTask` (Resolve or Reject) this function should perform.
/// @param _can_gc A `CanGc` token (may trigger GC internally).
#[allow(unsafe_code)]
// The apparently-unused CanGc argument reflects the fact that the JS API calls
// like NewFunctionWithReserved can trigger a GC.
fn create_native_handler_function(
    cx: *mut JSContext,
    holder: HandleObject,
    task: NativeHandlerTask,
    _can_gc: CanGc,
) -> *mut JSObject {
    unsafe {
        // Create a new JS function with `native_handler_callback` as its native entry point.
        let func = NewFunctionWithReserved(cx, Some(native_handler_callback), 1, 0, ptr::null());
        assert!(!func.is_null());

        rooted!(in(cx) let obj = JS_GetFunctionObject(func));
        assert!(!obj.is_null());
        // Store the native handler's JS object and the task type in the function's reserved slots.
        SetFunctionNativeReserved(obj.get(), SLOT_NATIVEHANDLER, &ObjectValue(*holder));
        SetFunctionNativeReserved(obj.get(), SLOT_NATIVEHANDLER_TASK, &Int32Value(task as i32));
        obj.get()
    }
}
/// Trait for operations that must be invoked from the generated bindings.
/// Functional Utility: Provides an abstraction layer for promise helper functions
/// that are expected to be called from auto-generated binding code.
pub(crate) trait PromiseHelpers<D: crate::DomTypes> {
    /// Creates a new `Promise` that is immediately resolved with a given value.
    /// Functional Utility: Offers a convenient way for generated bindings to
    /// create resolved promises.
    /// @param global The `GlobalScope` context.
    /// @param cx The JavaScript context.
    /// @param value The value to resolve the Promise with.
    /// @return An `Rc<D::Promise>` representing the resolved Promise.
    fn new_resolved(
        global: &D::GlobalScope,
        cx: SafeJSContext,
        value: impl ToJSValConvertible,
    ) -> Rc<D::Promise>;
}

impl PromiseHelpers<crate::DomTypeHolder> for Promise {
    fn new_resolved(
        global: &GlobalScope,
        cx: SafeJSContext,
        value: impl ToJSValConvertible,
    ) -> Rc<Promise> {
        Promise::new_resolved(global, cx, value, CanGc::note())
    }
}
