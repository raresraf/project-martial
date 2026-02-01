//! This module defines the `TestBinding` struct and implements the `TestBindingMethods` trait,
//! serving as a comprehensive test harness and example for generated WebIDL bindings in Rust.
//! It covers a wide array of WebIDL features, including various IDL types (e.g., boolean, byte,
//! string, enum, interface, union), nullable and optional types, sequences, records, promises,
//! event listeners, callbacks, preferences, and method overloading.
//!
//! Functional Utility: This file is crucial for validating the correctness and completeness
//! of the WebIDL binding generation process, ensuring that native Rust code can effectively
//! interact with JavaScript DOM objects and Web APIs as defined by WebIDL specifications.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

// check-tidy: no specs after this line

use std::borrow::ToOwned;
use std::ptr::{self, NonNull};
use std::rc::Rc;
use std::time::Duration;

use dom_struct::dom_struct;
use js::jsapi::{Heap, JSObject, JS_NewPlainObject};
use js::jsval::JSVal;
use js::rust::{CustomAutoRooterGuard, HandleObject, HandleValue, MutableHandleValue};
use js::typedarray::{self, Uint8ClampedArray};
use script_traits::serializable::BlobImpl;
use servo_config::prefs;

use crate::dom::bindings::buffer_source::create_buffer_source;
use crate::dom::bindings::callback::ExceptionHandling;
use crate::dom::bindings::codegen::Bindings::EventListenerBinding::EventListener;
use crate::dom::bindings::codegen::Bindings::FunctionBinding::Function;
use crate::dom::bindings::codegen::Bindings::TestBindingBinding::{
    SimpleCallback, TestBindingMethods, TestDictionary, TestDictionaryDefaults,
    TestDictionaryParent, TestDictionaryWithParent, TestEnum, TestURLLike,
};
use crate::dom::bindings::codegen::UnionTypes;
use crate::dom::bindings::codegen::UnionTypes::{
    BlobOrBlobSequence, BlobOrBoolean, BlobOrString, BlobOrUnsignedLong, ByteStringOrLong,
    ByteStringSequenceOrLong, ByteStringSequenceOrLongOrString, EventOrString, EventOrUSVString,
    HTMLElementOrLong, HTMLElementOrUnsignedLongOrStringOrBoolean, LongOrLongSequenceSequence,
    LongSequenceOrBoolean, StringOrBoolean, StringOrLongSequence, StringOrStringSequence,
    StringOrUnsignedLong, StringSequenceOrUnsignedLong, UnsignedLongOrBoolean,
};
use crate::dom::bindings::error::{Error, Fallible};
use crate::dom::bindings::num::Finite;
use crate::dom::bindings::record::Record;
use crate::dom::bindings::refcounted::TrustedPromise;
use crate::dom::bindings::reflector::{reflect_dom_object_with_proto, DomGlobal, Reflector};
use crate::dom::bindings::root::DomRoot;
use crate::dom::bindings::str::{ByteString, DOMString, USVString};
use crate::dom::bindings::trace::RootedTraceableBox;
use crate::dom::bindings::weakref::MutableWeakRef;
use crate::dom::blob::Blob;
use crate::dom::globalscope::GlobalScope;
use crate::dom::node::Node;
use crate::dom::promise::Promise;
use crate::dom::promisenativehandler::{Callback, PromiseNativeHandler};
use crate::dom::url::URL;
use crate::realms::InRealm;
use crate::script_runtime::{CanGc, JSContext as SafeJSContext};
use crate::timers::OneshotTimerCallback;

/// Represents a native binding for testing various WebIDL features.
/// Functional Utility: This struct serves as a test subject for the WebIDL binding generator,
/// allowing verification of how different IDL types, attributes, and methods
/// are translated and handled in Rust code.
#[dom_struct]
pub(crate) struct TestBinding {
    /// The `Reflector` provides the bridge between the native Rust object and its JS counterpart.
    reflector_: Reflector,
    /// A mutable weak reference to a `URL` object, used for testing nullable interface attributes.
    url: MutableWeakRef<URL>,
}

#[allow(non_snake_case)]
impl TestBinding {
    /// Internal constructor for creating an inherited `TestBinding` instance.
    /// Functional Utility: Used internally during the reflection process to create
    /// a base `TestBinding` object without immediate JavaScript association.
    /// @return A new `TestBinding` instance.
    fn new_inherited() -> TestBinding {
        TestBinding {
            reflector_: Reflector::new(),
            url: MutableWeakRef::new(None),
        }
    }

    /// Creates a new `DomRoot<TestBinding>` instance, reflecting it into the JavaScript DOM.
    /// Functional Utility: This is the primary entry point for instantiating `TestBinding`
    /// objects that are accessible from JavaScript, ensuring they are properly
    /// rooted and managed by the DOM.
    /// @param global The `GlobalScope` in which the object is created.
    /// @param proto An optional `HandleObject` to use as the JavaScript prototype.
    /// @param can_gc A `CanGc` token.
    /// @return A `DomRoot<TestBinding>` representing the new instance.
    fn new(
        global: &GlobalScope,
        proto: Option<HandleObject>,
        can_gc: CanGc,
    ) -> DomRoot<TestBinding> {
        reflect_dom_object_with_proto(
            Box::new(TestBinding::new_inherited()),
            global,
            proto,
            can_gc,
        )
    }
}
impl TestBindingMethods<crate::DomTypeHolder> for TestBinding {
    /// Implements the default constructor for `TestBinding`.
    /// Functional Utility: Tests the basic instantiation of a WebIDL interface
    /// without any arguments.
    /// @param global The `GlobalScope` context.
    /// @param proto An optional `HandleObject` for the JavaScript prototype.
    /// @param can_gc A `CanGc` token.
    /// @return A `Fallible<DomRoot<TestBinding>>` for the new instance.
    fn Constructor(
        global: &GlobalScope,
        proto: Option<HandleObject>,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<TestBinding>> {
        Ok(TestBinding::new(global, proto, can_gc))
    }

    /// Implements an overloaded constructor for `TestBinding` taking a sequence of numbers.
    /// Functional Utility: Tests WebIDL's ability to handle overloaded constructors
    /// with different argument signatures, specifically a `sequence<double>`.
    /// @param global The `GlobalScope` context.
    /// @param proto An optional `HandleObject` for the JavaScript prototype.
    /// @param can_gc A `CanGc` token.
    /// @param nums A `Vec` of `f64` representing the input sequence.
    /// @return A `Fallible<DomRoot<TestBinding>>` for the new instance.
    #[allow(unused_variables)]
    fn Constructor_(
        global: &GlobalScope,
        proto: Option<HandleObject>,
        can_gc: CanGc,
        nums: Vec<f64>,
    ) -> Fallible<DomRoot<TestBinding>> {
        Ok(TestBinding::new(global, proto, can_gc))
    }

    /// Implements another overloaded constructor for `TestBinding` taking a single number.
    /// Functional Utility: Further tests WebIDL's ability to handle overloaded constructors
    /// with different argument signatures, specifically a single `double`.
    /// @param global The `GlobalScope` context.
    /// @param proto An optional `HandleObject` for the JavaScript prototype.
    /// @param can_gc A `CanGc` token.
    /// @param num A `f64` representing the input number.
    /// @return A `Fallible<DomRoot<TestBinding>>` for the new instance.
    #[allow(unused_variables)]
    fn Constructor__(
        global: &GlobalScope,
        proto: Option<HandleObject>,
        can_gc: CanGc,
        num: f64,
    ) -> Fallible<DomRoot<TestBinding>> {
        Ok(TestBinding::new(global, proto, can_gc))
    }

    /// Getter for a boolean attribute.
    /// Functional Utility: Tests the binding of a simple `boolean` IDL attribute.
    /// @return The boolean value of the attribute.
    fn BooleanAttribute(&self) -> bool {
        false
    }
    /// Setter for a boolean attribute.
    /// Functional Utility: Tests setting a simple `boolean` IDL attribute.
    /// @param _ The boolean value to set.
    fn SetBooleanAttribute(&self, _: bool) {}
    /// Getter for a byte attribute.
    /// Functional Utility: Tests the binding of a `byte` IDL attribute (signed 8-bit integer).
    /// @return The byte value of the attribute.
    fn ByteAttribute(&self) -> i8 {
        0
    }
    /// Setter for a byte attribute.
    /// Functional Utility: Tests setting a `byte` IDL attribute.
    /// @param _ The byte value to set.
    fn SetByteAttribute(&self, _: i8) {}
    /// Getter for an octet attribute.
    /// Functional Utility: Tests the binding of an `octet` IDL attribute (unsigned 8-bit integer).
    /// @return The octet value of the attribute.
    fn OctetAttribute(&self) -> u8 {
        0
    }
    /// Setter for an octet attribute.
    /// Functional Utility: Tests setting an `octet` IDL attribute.
    /// @param _ The octet value to set.
    fn SetOctetAttribute(&self, _: u8) {}
    /// Getter for a short attribute.
    /// Functional Utility: Tests the binding of a `short` IDL attribute (signed 16-bit integer).
    /// @return The short value of the attribute.
    fn ShortAttribute(&self) -> i16 {
        0
    }
    /// Setter for a short attribute.
    /// Functional Utility: Tests setting a `short` IDL attribute.
    /// @param _ The short value to set.
    fn SetShortAttribute(&self, _: i16) {}
    /// Getter for an unsigned short attribute.
    /// Functional Utility: Tests the binding of an `unsigned short` IDL attribute (unsigned 16-bit integer).
    /// @return The unsigned short value of the attribute.
    fn UnsignedShortAttribute(&self) -> u16 {
        0
    }
    /// Setter for an unsigned short attribute.
    /// Functional Utility: Tests setting an `unsigned short` IDL attribute.
    /// @param _ The unsigned short value to set.
    fn SetUnsignedShortAttribute(&self, _: u16) {}
    /// Getter for a long attribute.
    /// Functional Utility: Tests the binding of a `long` IDL attribute (signed 32-bit integer).
    /// @return The long value of the attribute.
    fn LongAttribute(&self) -> i32 {
        0
    }
    /// Setter for a long attribute.
    /// Functional Utility: Tests setting a `long` IDL attribute.
    /// @param _ The long value to set.
    fn SetLongAttribute(&self, _: i32) {}
    /// Getter for an unsigned long attribute.
    /// Functional Utility: Tests the binding of an `unsigned long` IDL attribute (unsigned 32-bit integer).
    /// @return The unsigned long value of the attribute.
    fn UnsignedLongAttribute(&self) -> u32 {
        0
    }
    /// Setter for an unsigned long attribute.
    /// Functional Utility: Tests setting an `unsigned long` IDL attribute.
    /// @param _ The unsigned long value to set.
    fn SetUnsignedLongAttribute(&self, _: u32) {}
    /// Getter for a long long attribute.
    /// Functional Utility: Tests the binding of a `long long` IDL attribute (signed 64-bit integer).
    /// @return The long long value of the attribute.
    fn LongLongAttribute(&self) -> i64 {
        0
    }
    /// Setter for a long long attribute.
    /// Functional Utility: Tests setting a `long long` IDL attribute.
    /// @param _ The long long value to set.
    fn SetLongLongAttribute(&self, _: i64) {}
    /// Getter for an unsigned long long attribute.
    /// Functional Utility: Tests the binding of an `unsigned long long` IDL attribute (unsigned 64-bit integer).
    /// @return The unsigned long long value of the attribute.
    fn UnsignedLongLongAttribute(&self) -> u64 {
        0
    }
    /// Setter for an unsigned long long attribute.
    /// Functional Utility: Tests setting an `unsigned long long` IDL attribute.
    /// @param _ The unsigned long long value to set.
    fn SetUnsignedLongLongAttribute(&self, _: u64) {}
    /// Getter for an unrestricted float attribute.
    /// Functional Utility: Tests the binding of an `unrestricted float` IDL attribute (32-bit floating-point).
    /// @return The unrestricted float value of the attribute.
    fn UnrestrictedFloatAttribute(&self) -> f32 {
        0.
    }
    /// Setter for an unrestricted float attribute.
    /// Functional Utility: Tests setting an `unrestricted float` IDL attribute.
    /// @param _ The unrestricted float value to set.
    fn SetUnrestrictedFloatAttribute(&self, _: f32) {}
    /// Getter for a float attribute.
    /// Functional Utility: Tests the binding of a `float` IDL attribute (32-bit floating-point, guaranteed finite).
    /// @return The finite float value of the attribute.
    fn FloatAttribute(&self) -> Finite<f32> {
        Finite::wrap(0.)
    }
    /// Setter for a float attribute.
    /// Functional Utility: Tests setting a `float` IDL attribute.
    /// @param _ The finite float value to set.
    fn SetFloatAttribute(&self, _: Finite<f32>) {}
    /// Getter for an unrestricted double attribute.
    /// Functional Utility: Tests the binding of an `unrestricted double` IDL attribute (64-bit floating-point).
    /// @return The unrestricted double value of the attribute.
    fn UnrestrictedDoubleAttribute(&self) -> f64 {
        0.
    }
    /// Setter for an unrestricted double attribute.
    /// Functional Utility: Tests setting an `unrestricted double` IDL attribute.
    /// @param _ The unrestricted double value to set.
    fn SetUnrestrictedDoubleAttribute(&self, _: f64) {}
    /// Getter for a double attribute.
    /// Functional Utility: Tests the binding of a `double` IDL attribute (64-bit floating-point, guaranteed finite).
    /// @return The finite double value of the attribute.
    fn DoubleAttribute(&self) -> Finite<f64> {
        Finite::wrap(0.)
    }
    /// Setter for a double attribute.
    /// Functional Utility: Tests setting a `double` IDL attribute.
    /// @param _ The finite double value to set.
    fn SetDoubleAttribute(&self, _: Finite<f64>) {}
    /// Getter for a string attribute.
    /// Functional Utility: Tests the binding of a `DOMString` IDL attribute.
    /// @return The `DOMString` value of the attribute.
    fn StringAttribute(&self) -> DOMString {
        DOMString::new()
    }
    /// Setter for a string attribute.
    /// Functional Utility: Tests setting a `DOMString` IDL attribute.
    /// @param _ The `DOMString` value to set.
    fn SetStringAttribute(&self, _: DOMString) {}
    /// Getter for a USVString attribute.
    /// Functional Utility: Tests the binding of a `USVString` IDL attribute (Unicode Scalar Value string).
    /// @return The `USVString` value of the attribute.
    fn UsvstringAttribute(&self) -> USVString {
        USVString("".to_owned())
    }
    /// Setter for a USVString attribute.
    /// Functional Utility: Tests setting a `USVString` IDL attribute.
    /// @param _ The `USVString` value to set.
    fn SetUsvstringAttribute(&self, _: USVString) {}
    /// Getter for a ByteString attribute.
    /// Functional Utility: Tests the binding of a `ByteString` IDL attribute (sequence of 8-bit bytes).
    /// @return The `ByteString` value of the attribute.
    fn ByteStringAttribute(&self) -> ByteString {
        ByteString::new(vec![])
    }
    /// Setter for a ByteString attribute.
    /// Functional Utility: Tests setting a `ByteString` IDL attribute.
    /// @param _ The `ByteString` value to set.
    fn SetByteStringAttribute(&self, _: ByteString) {}
    /// Getter for an enum attribute.
    /// Functional Utility: Tests the binding of an `enum` IDL attribute.
    /// @return The `TestEnum` value of the attribute.
    fn EnumAttribute(&self) -> TestEnum {
        TestEnum::_empty
    }
    /// Setter for an enum attribute.
    /// Functional Utility: Tests setting an `enum` IDL attribute.
    /// @param _ The `TestEnum` value to set.
    fn SetEnumAttribute(&self, _: TestEnum) {}
    /// Getter for an interface attribute.
    /// Functional Utility: Tests the binding of an interface type (`Blob`) as an IDL attribute.
    /// @param can_gc A `CanGc` token.
    /// @return A `DomRoot<Blob>` representing the attribute's value.
    fn InterfaceAttribute(&self, can_gc: CanGc) -> DomRoot<Blob> {
        Blob::new(
            &self.global(),
            BlobImpl::new_from_bytes(vec![], "".to_owned()),
            can_gc,
        )
    }
    /// Setter for an interface attribute.
    /// Functional Utility: Tests setting an interface type (`Blob`) as an IDL attribute.
    /// @param _ The `Blob` object to set.
    fn SetInterfaceAttribute(&self, _: &Blob) {}
    /// Getter for a union attribute (`HTMLElementOrLong`).
    /// Functional Utility: Tests the binding of a union type attribute with different possible types.
    /// @return An `HTMLElementOrLong` value of the attribute.
    fn UnionAttribute(&self) -> HTMLElementOrLong {
        HTMLElementOrLong::Long(0)
    }
    /// Setter for a union attribute (`HTMLElementOrLong`).
    /// Functional Utility: Tests setting a union type attribute.
    /// @param _ The `HTMLElementOrLong` value to set.
    fn SetUnionAttribute(&self, _: HTMLElementOrLong) {}
    /// Getter for a union attribute (`EventOrString`).
    /// Functional Utility: Tests the binding of another union type attribute with different possible types.
    /// @return An `EventOrString` value of the attribute.
    fn Union2Attribute(&self) -> EventOrString {
        EventOrString::String(DOMString::new())
    }
    /// Setter for a union attribute (`EventOrString`).
    /// Functional Utility: Tests setting a union type attribute.
    /// @param _ The `EventOrString` value to set.
    fn SetUnion2Attribute(&self, _: EventOrString) {}
    /// Getter for a union attribute (`EventOrUSVString`).
    /// Functional Utility: Tests the binding of a union type with `Event` or `USVString`.
    /// @return An `EventOrUSVString` value of the attribute.
    fn Union3Attribute(&self) -> EventOrUSVString {
        EventOrUSVString::USVString("".to_owned())
    }
    /// Setter for a union attribute (`EventOrUSVString`).
    /// Functional Utility: Tests setting a union type attribute.
    /// @param _ The `EventOrUSVString` value to set.
    fn SetUnion3Attribute(&self, _: EventOrUSVString) {}
    /// Getter for a union attribute (`StringOrUnsignedLong`).
    /// Functional Utility: Tests the binding of a union type with `DOMString` or `unsigned long`.
    /// @return A `StringOrUnsignedLong` value of the attribute.
    fn Union4Attribute(&self) -> StringOrUnsignedLong {
        StringOrUnsignedLong::UnsignedLong(0u32)
    }
    /// Setter for a union attribute (`StringOrUnsignedLong`).
    /// Functional Utility: Tests setting a union type attribute.
    /// @param _ The `StringOrUnsignedLong` value to set.
    fn SetUnion4Attribute(&self, _: StringOrUnsignedLong) {}
    /// Getter for a union attribute (`StringOrBoolean`).
    /// Functional Utility: Tests the binding of a union type with `DOMString` or `boolean`.
    /// @return A `StringOrBoolean` value of the attribute.
    fn Union5Attribute(&self) -> StringOrBoolean {
        StringOrBoolean::Boolean(true)
    }
    /// Setter for a union attribute (`StringOrBoolean`).
    /// Functional Utility: Tests setting a union type attribute.
    /// @param _ The `StringOrBoolean` value to set.
    fn SetUnion5Attribute(&self, _: StringOrBoolean) {}
    /// Getter for a union attribute (`UnsignedLongOrBoolean`).
    /// Functional Utility: Tests the binding of a union type with `unsigned long` or `boolean`.
    /// @return An `UnsignedLongOrBoolean` value of the attribute.
    fn Union6Attribute(&self) -> UnsignedLongOrBoolean {
        UnsignedLongOrBoolean::Boolean(true)
    }
    /// Setter for a union attribute (`UnsignedLongOrBoolean`).
    /// Functional Utility: Tests setting a union type attribute.
    /// @param _ The `UnsignedLongOrBoolean` value to set.
    fn SetUnion6Attribute(&self, _: UnsignedLongOrBoolean) {}
    /// Getter for a union attribute (`BlobOrBoolean`).
    /// Functional Utility: Tests the binding of a union type with `Blob` or `boolean`.
    /// @return A `BlobOrBoolean` value of the attribute.
    fn Union7Attribute(&self) -> BlobOrBoolean {
        BlobOrBoolean::Boolean(true)
    }
    /// Setter for a union attribute (`BlobOrBoolean`).
    /// Functional Utility: Tests setting a union type attribute.
    /// @param _ The `BlobOrBoolean` value to set.
    fn SetUnion7Attribute(&self, _: BlobOrBoolean) {}
    /// Getter for a union attribute (`BlobOrUnsignedLong`).
    /// Functional Utility: Tests the binding of a union type with `Blob` or `unsigned long`.
    /// @return A `BlobOrUnsignedLong` value of the attribute.
    fn Union8Attribute(&self) -> BlobOrUnsignedLong {
        BlobOrUnsignedLong::UnsignedLong(0u32)
    }
    /// Setter for a union attribute (`BlobOrUnsignedLong`).
    /// Functional Utility: Tests setting a union type attribute.
    /// @param _ The `BlobOrUnsignedLong` value to set.
    fn SetUnion8Attribute(&self, _: BlobOrUnsignedLong) {}
    /// Getter for a union attribute (`ByteStringOrLong`).
    /// Functional Utility: Tests the binding of a union type with `ByteString` or `long`.
    /// @return A `ByteStringOrLong` value of the attribute.
    fn Union9Attribute(&self) -> ByteStringOrLong {
        ByteStringOrLong::ByteString(ByteString::new(vec![]))
    }
    /// Setter for a union attribute (`ByteStringOrLong`).
    /// Functional Utility: Tests setting a union type attribute.
    /// @param _ The `ByteStringOrLong` value to set.
    fn SetUnion9Attribute(&self, _: ByteStringOrLong) {}
    /// Getter for an array attribute (specifically `Uint8ClampedArray`).
    /// Functional Utility: Tests the binding of an IDL `Uint8ClampedArray` attribute,
    /// returning a newly created instance.
    /// @param cx The JavaScript context.
    /// @return A `Uint8ClampedArray` value of the attribute.
    fn ArrayAttribute(&self, cx: SafeJSContext) -> Uint8ClampedArray {
        let data: [u8; 16] = [0; 16];

        rooted!(in (*cx) let mut array = ptr::null_mut::<JSObject>());
        create_buffer_source(cx, &data, array.handle_mut(), CanGc::note())
            .expect("Creating ClampedU8 array should never fail")
    }
    /// Getter for an `any` attribute.
    /// Functional Utility: Tests the binding of an `any` IDL attribute, which can hold any JavaScript value.
    /// @param _cx The JavaScript context.
    /// @param _v A `MutableHandleValue` to receive the attribute's value.
    fn AnyAttribute(&self, _: SafeJSContext, _: MutableHandleValue) {}
    /// Setter for an `any` attribute.
    /// Functional Utility: Tests setting an `any` IDL attribute.
    /// @param _cx The JavaScript context.
    /// @param _v The `HandleValue` to set as the attribute's value.
    fn SetAnyAttribute(&self, _: SafeJSContext, _: HandleValue) {}
    /// Getter for an object attribute.
    /// Functional Utility: Tests the binding of an IDL `object` type attribute,
    /// returning a new plain JavaScript object.
    /// @param cx The JavaScript context.
    /// @return A `NonNull<JSObject>` representing the attribute's value.
    #[allow(unsafe_code)]
    fn ObjectAttribute(&self, cx: SafeJSContext) -> NonNull<JSObject> {
        unsafe {
            rooted!(in(*cx) let obj = JS_NewPlainObject(*cx));
            NonNull::new(obj.get()).expect("got a null pointer")
        }
    }
    /// Setter for an object attribute.
    /// Functional Utility: Tests setting an IDL `object` type attribute.
    /// @param _cx The JavaScript context.
    /// @param _ The raw pointer to the `JSObject` to set.
    fn SetObjectAttribute(&self, _: SafeJSContext, _: *mut JSObject) {}

    /// Getter for a nullable boolean attribute.
    /// Functional Utility: Tests the binding of a `boolean?` IDL attribute (nullable boolean).
    /// @return An `Option<bool>` representing the nullable boolean value.
    fn GetBooleanAttributeNullable(&self) -> Option<bool> {
        Some(false)
    }
    /// Setter for a nullable boolean attribute.
    /// Functional Utility: Tests setting a `boolean?` IDL attribute.
    /// @param _ The `Option<bool>` value to set.
    fn SetBooleanAttributeNullable(&self, _: Option<bool>) {}
    /// Getter for a nullable byte attribute.
    /// Functional Utility: Tests the binding of a `byte?` IDL attribute (nullable signed 8-bit integer).
    /// @return An `Option<i8>` representing the nullable byte value.
    fn GetByteAttributeNullable(&self) -> Option<i8> {
        Some(0)
    }
    /// Setter for a nullable byte attribute.
    /// Functional Utility: Tests setting a `byte?` IDL attribute.
    /// @param _ The `Option<i8>` value to set.
    fn SetByteAttributeNullable(&self, _: Option<i8>) {}
    /// Getter for a nullable octet attribute.
    /// Functional Utility: Tests the binding of an `octet?` IDL attribute (nullable unsigned 8-bit integer).
    /// @return An `Option<u8>` representing the nullable octet value.
    fn GetOctetAttributeNullable(&self) -> Option<u8> {
        Some(0)
    }
    /// Setter for a nullable octet attribute.
    /// Functional Utility: Tests setting an `octet?` IDL attribute.
    /// @param _ The `Option<u8>` value to set.
    fn SetOctetAttributeNullable(&self, _: Option<u8>) {}
    /// Getter for a nullable short attribute.
    /// Functional Utility: Tests the binding of a `short?` IDL attribute (nullable signed 16-bit integer).
    /// @return An `Option<i16>` representing the nullable short value.
    fn GetShortAttributeNullable(&self) -> Option<i16> {
        Some(0)
    }
    /// Setter for a nullable short attribute.
    /// Functional Utility: Tests setting a `short?` IDL attribute.
    /// @param _ The `Option<i16>` value to set.
    fn SetShortAttributeNullable(&self, _: Option<i16>) {}
    /// Getter for a nullable unsigned short attribute.
    /// Functional Utility: Tests the binding of an `unsigned short?` IDL attribute (nullable unsigned 16-bit integer).
    /// @return An `Option<u16>` representing the nullable unsigned short value.
    fn GetUnsignedShortAttributeNullable(&self) -> Option<u16> {
        Some(0)
    }
    /// Setter for a nullable unsigned short attribute.
    /// Functional Utility: Tests setting an `unsigned short?` IDL attribute.
    /// @param _ The `Option<u16>` value to set.
    fn SetUnsignedShortAttributeNullable(&self, _: Option<u16>) {}
    /// Getter for a nullable long attribute.
    /// Functional Utility: Tests the binding of a `long?` IDL attribute (nullable signed 32-bit integer).
    /// @return An `Option<i32>` representing the nullable long value.
    fn GetLongAttributeNullable(&self) -> Option<i32> {
        Some(0)
    }
    /// Setter for a nullable long attribute.
    /// Functional Utility: Tests setting a `long?` IDL attribute.
    /// @param _ The `Option<i32>` value to set.
    fn SetLongAttributeNullable(&self, _: Option<i32>) {}
    /// Getter for a nullable unsigned long attribute.
    /// Functional Utility: Tests the binding of an `unsigned long?` IDL attribute (nullable unsigned 32-bit integer).
    /// @return An `Option<u32>` representing the nullable unsigned long value.
    fn GetUnsignedLongAttributeNullable(&self) -> Option<u32> {
        Some(0)
    }
    /// Setter for a nullable unsigned long attribute.
    /// Functional Utility: Tests setting an `unsigned long?` IDL attribute.
    /// @param _ The `Option<u32>` value to set.
    fn SetUnsignedLongAttributeNullable(&self, _: Option<u32>) {}
    /// Getter for a nullable long long attribute.
    /// Functional Utility: Tests the binding of a `long long?` IDL attribute (nullable signed 64-bit integer).
    /// @return An `Option<i64>` representing the nullable long long value.
    fn GetLongLongAttributeNullable(&self) -> Option<i64> {
        Some(0)
    }
    /// Setter for a nullable long long attribute.
    /// Functional Utility: Tests setting a `long long?` IDL attribute.
    /// @param _ The `Option<i64>` value to set.
    fn SetLongLongAttributeNullable(&self, _: Option<i64>) {}
    /// Getter for a nullable unsigned long long attribute.
    /// Functional Utility: Tests the binding of an `unsigned long long?` IDL attribute (nullable unsigned 64-bit integer).
    /// @return An `Option<u64>` representing the nullable unsigned long long value.
    fn GetUnsignedLongLongAttributeNullable(&self) -> Option<u64> {
        Some(0)
    }
    /// Setter for a nullable unsigned long long attribute.
    /// Functional Utility: Tests setting an `unsigned long long?` IDL attribute.
    /// @param _ The `Option<u64>` value to set.
    fn SetUnsignedLongLongAttributeNullable(&self, _: Option<u64>) {}
    /// Getter for a nullable unrestricted float attribute.
    /// Functional Utility: Tests the binding of an `unrestricted float?` IDL attribute (nullable 32-bit floating-point).
    /// @return An `Option<f32>` representing the nullable unrestricted float value.
    fn GetUnrestrictedFloatAttributeNullable(&self) -> Option<f32> {
        Some(0.)
    }
    /// Setter for a nullable unrestricted float attribute.
    /// Functional Utility: Tests setting an `unrestricted float?` IDL attribute.
    /// @param _ The `Option<f32>` value to set.
    fn SetUnrestrictedFloatAttributeNullable(&self, _: Option<f32>) {}
    /// Getter for a nullable float attribute.
    /// Functional Utility: Tests the binding of a `float?` IDL attribute (nullable 32-bit floating-point, guaranteed finite).
    /// @return An `Option<Finite<f32>>` representing the nullable finite float value.
    fn GetFloatAttributeNullable(&self) -> Option<Finite<f32>> {
        Some(Finite::wrap(0.))
    }
    /// Setter for a nullable float attribute.
    /// Functional Utility: Tests setting a `float?` IDL attribute.
    /// @param _ The `Option<Finite<f32>>` value to set.
    fn SetFloatAttributeNullable(&self, _: Option<Finite<f32>>) {}
    /// Getter for a nullable unrestricted double attribute.
    /// Functional Utility: Tests the binding of an `unrestricted double?` IDL attribute (nullable 64-bit floating-point).
    /// @return An `Option<f64>` representing the nullable unrestricted double value.
    fn GetUnrestrictedDoubleAttributeNullable(&self) -> Option<f64> {
        Some(0.)
    }
    /// Setter for a nullable unrestricted double attribute.
    /// Functional Utility: Tests setting an `unrestricted double?` IDL attribute.
    /// @param _ The `Option<f64>` value to set.
    fn SetUnrestrictedDoubleAttributeNullable(&self, _: Option<f64>) {}
    /// Getter for a nullable double attribute.
    /// Functional Utility: Tests the binding of a `double?` IDL attribute (nullable 64-bit floating-point, guaranteed finite).
    /// @return An `Option<Finite<f64>>` representing the nullable finite double value.
    fn GetDoubleAttributeNullable(&self) -> Option<Finite<f64>> {
        Some(Finite::wrap(0.))
    }
    /// Setter for a nullable double attribute.
    /// Functional Utility: Tests setting a `double?` IDL attribute.
    /// @param _ The `Option<Finite<f64>>` value to set.
    fn SetDoubleAttributeNullable(&self, _: Option<Finite<f64>>) {}
    /// Getter for a nullable ByteString attribute.
    /// Functional Utility: Tests the binding of a `ByteString?` IDL attribute (nullable sequence of 8-bit bytes).
    /// @return An `Option<ByteString>` representing the nullable ByteString value.
    fn GetByteStringAttributeNullable(&self) -> Option<ByteString> {
        Some(ByteString::new(vec![]))
    }
    /// Setter for a nullable ByteString attribute.
    /// Functional Utility: Tests setting a `ByteString?` IDL attribute.
    /// @param _ The `Option<ByteString>` value to set.
    fn SetByteStringAttributeNullable(&self, _: Option<ByteString>) {}
    /// Getter for a nullable String attribute.
    /// Functional Utility: Tests the binding of a `DOMString?` IDL attribute (nullable DOMString).
    /// @return An `Option<DOMString>` representing the nullable String value.
    fn GetStringAttributeNullable(&self) -> Option<DOMString> {
        Some(DOMString::new())
    }
    /// Setter for a nullable String attribute.
    /// Functional Utility: Tests setting a `DOMString?` IDL attribute.
    /// @param _ The `Option<DOMString>` value to set.
    fn SetStringAttributeNullable(&self, _: Option<DOMString>) {}
    /// Getter for a nullable USVString attribute.
    /// Functional Utility: Tests the binding of a `USVString?` IDL attribute (nullable Unicode Scalar Value string).
    /// @return An `Option<USVString>` representing the nullable USVString value.
    fn GetUsvstringAttributeNullable(&self) -> Option<USVString> {
        Some(USVString("".to_owned()))
    }
    /// Setter for a nullable USVString attribute.
    /// Functional Utility: Tests setting a `USVString?` IDL attribute.
    /// @param _ The `Option<USVString>` value to set.
    fn SetUsvstringAttributeNullable(&self, _: Option<USVString>) {}
    /// Setter for a binary renamed attribute.
    /// Functional Utility: Tests how WebIDL handles attribute names that are keywords in Rust,
    /// requiring renaming in the generated binding. This specific setter tests `binary` renamed.
    /// @param _ The `DOMString` value to set.
    fn SetBinaryRenamedAttribute(&self, _: DOMString) {}
    /// Getter for a forwarded attribute that returns the `TestBinding` itself.
    /// Functional Utility: Tests the binding of an attribute that returns a reference
    /// to the current object, often used for chaining or self-referencing patterns.
    /// @return A `DomRoot<TestBinding>` representing the current object.
    fn ForwardedAttribute(&self) -> DomRoot<TestBinding> {
        DomRoot::from_ref(self)
    }
    /// Getter for a binary renamed attribute.
    /// Functional Utility: Tests how WebIDL handles attribute names that are keywords in Rust,
    /// requiring renaming in the generated binding. This specific getter tests `binary` renamed.
    /// @return The `DOMString` value of the attribute.
    fn BinaryRenamedAttribute(&self) -> DOMString {
        DOMString::new()
    }
    /// Setter for a second binary renamed attribute.
    /// Functional Utility: Tests another instance of WebIDL attribute renaming
    /// due to Rust keyword conflicts.
    /// @param _ The `DOMString` value to set.
    fn SetBinaryRenamedAttribute2(&self, _: DOMString) {}
    /// Getter for a second binary renamed attribute.
    /// Functional Utility: Tests another instance of WebIDL attribute renaming
    /// due to Rust keyword conflicts.
    /// @return The `DOMString` value of the attribute.
    fn BinaryRenamedAttribute2(&self) -> DOMString {
        DOMString::new()
    }
    /// Getter for an attribute that should be automatically renamed by the binding generator.
    /// Functional Utility: Tests automatic renaming conventions in WebIDL bindings.
    /// @return The `DOMString` value of the attribute.
    fn Attr_to_automatically_rename(&self) -> DOMString {
        DOMString::new()
    }
    /// Setter for an attribute that should be automatically renamed.
    /// Functional Utility: Tests setting an attribute that undergoes automatic renaming.
    /// @param _ The `DOMString` value to set.
    fn SetAttr_to_automatically_rename(&self, _: DOMString) {}
    /// Getter for a nullable enum attribute.
    /// Functional Utility: Tests the binding of an `enum?` IDL attribute (nullable enum).
    /// @return An `Option<TestEnum>` representing the nullable enum value.
    fn GetEnumAttributeNullable(&self) -> Option<TestEnum> {
        Some(TestEnum::_empty)
    }
    /// Getter for a nullable interface attribute.
    /// Functional Utility: Tests the binding of an interface type (`Blob?`) as a nullable IDL attribute.
    /// @param can_gc A `CanGc` token.
    /// @return An `Option<DomRoot<Blob>>` representing the nullable Blob object.
    fn GetInterfaceAttributeNullable(&self, can_gc: CanGc) -> Option<DomRoot<Blob>> {
        Some(Blob::new(
            &self.global(),
            BlobImpl::new_from_bytes(vec![], "".to_owned()),
            can_gc,
        ))
    }
    /// Setter for a nullable interface attribute.
    /// Functional Utility: Tests setting a `Blob?` IDL attribute.
    /// @param _ The `Option<&Blob>` object to set.
    fn SetInterfaceAttributeNullable(&self, _: Option<&Blob>) {}
    /// Getter for a weak nullable interface attribute (`URL?`).
    /// Functional Utility: Tests the binding of a nullable interface type that is held as a weak reference.
    /// @return An `Option<DomRoot<URL>>` representing the nullable URL object, if available.
    fn GetInterfaceAttributeWeak(&self) -> Option<DomRoot<URL>> {
        self.url.root()
    }
    /// Setter for a weak nullable interface attribute (`URL?`).
    /// Functional Utility: Tests setting a nullable interface attribute held as a weak reference.
    /// @param url The `Option<&URL>` object to set as a weak reference.
    fn SetInterfaceAttributeWeak(&self, url: Option<&URL>) {
        self.url.set(url);
    }
    /// Getter for a nullable object attribute.
    /// Functional Utility: Tests the binding of an `object?` IDL attribute (nullable JavaScript object).
    /// @param _cx The JavaScript context.
    /// @return An `Option<NonNull<JSObject>>` representing the nullable object.
    fn GetObjectAttributeNullable(&self, _: SafeJSContext) -> Option<NonNull<JSObject>> {
        None
    }
    /// Setter for a nullable object attribute.
    /// Functional Utility: Tests setting an `object?` IDL attribute.
    /// @param _cx The JavaScript context.
    /// @param _ The raw pointer to the `JSObject` to set.
    fn SetObjectAttributeNullable(&self, _: SafeJSContext, _: *mut JSObject) {}
    /// Getter for a nullable union attribute (`HTMLElementOrLong?`).
    /// Functional Utility: Tests the binding of a nullable union type attribute.
    /// @return An `Option<HTMLElementOrLong>` representing the nullable union value.
    fn GetUnionAttributeNullable(&self) -> Option<HTMLElementOrLong> {
        Some(HTMLElementOrLong::Long(0))
    }
    /// Setter for a nullable union attribute (`HTMLElementOrLong?`).
    /// Functional Utility: Tests setting a nullable union type attribute.
    /// @param _ The `Option<HTMLElementOrLong>` value to set.
    fn SetUnionAttributeNullable(&self, _: Option<HTMLElementOrLong>) {}
    /// Getter for a nullable union attribute (`EventOrString?`).
    /// Functional Utility: Tests the binding of another nullable union type attribute.
    /// @return An `Option<EventOrString>` representing the nullable union value.
    fn GetUnion2AttributeNullable(&self) -> Option<EventOrString> {
        Some(EventOrString::String(DOMString::new()))
    }
    /// Setter for a nullable union attribute (`EventOrString?`).
    /// Functional Utility: Tests setting another nullable union type attribute.
    /// @param _ The `Option<EventOrString>` value to set.
    fn SetUnion2AttributeNullable(&self, _: Option<EventOrString>) {}
    /// Getter for a nullable union attribute (`BlobOrBoolean?`).
    /// Functional Utility: Tests the binding of a nullable union type with `Blob` or `boolean`.
    /// @return An `Option<BlobOrBoolean>` representing the nullable union value.
    fn GetUnion3AttributeNullable(&self) -> Option<BlobOrBoolean> {
        Some(BlobOrBoolean::Boolean(true))
    }
    /// Setter for a nullable union attribute (`BlobOrBoolean?`).
    /// Functional Utility: Tests setting a nullable union type attribute.
    /// @param _ The `Option<BlobOrBoolean>` value to set.
    fn SetUnion3AttributeNullable(&self, _: Option<BlobOrBoolean>) {}
    /// Getter for a nullable union attribute (`UnsignedLongOrBoolean?`).
    /// Functional Utility: Tests the binding of a nullable union type with `unsigned long` or `boolean`.
    /// @return An `Option<UnsignedLongOrBoolean>` representing the nullable union value.
    fn GetUnion4AttributeNullable(&self) -> Option<UnsignedLongOrBoolean> {
        Some(UnsignedLongOrBoolean::Boolean(true))
    }
    /// Setter for a nullable union attribute (`UnsignedLongOrBoolean?`).
    /// Functional Utility: Tests setting a nullable union type attribute.
    /// @param _ The `Option<UnsignedLongOrBoolean>` value to set.
    fn SetUnion4AttributeNullable(&self, _: Option<UnsignedLongOrBoolean>) {}
    /// Getter for a nullable union attribute (`StringOrBoolean?`).
    /// Functional Utility: Tests the binding of a nullable union type with `DOMString` or `boolean`.
    /// @return An `Option<StringOrBoolean>` representing the nullable union value.
    fn GetUnion5AttributeNullable(&self) -> Option<StringOrBoolean> {
        Some(StringOrBoolean::Boolean(true))
    }
    /// Setter for a nullable union attribute (`StringOrBoolean?`).
    /// Functional Utility: Tests setting a nullable union type attribute.
    /// @param _ The `Option<StringOrBoolean>` value to set.
    fn SetUnion5AttributeNullable(&self, _: Option<StringOrBoolean>) {}
    /// Getter for a nullable union attribute (`ByteStringOrLong?`).
    /// Functional Utility: Tests the binding of a nullable union type with `ByteString` or `long`.
    /// @return An `Option<ByteStringOrLong>` representing the nullable union value.
    fn GetUnion6AttributeNullable(&self) -> Option<ByteStringOrLong> {
        Some(ByteStringOrLong::ByteString(ByteString::new(vec![])))
    }
    /// Setter for a nullable union attribute (`ByteStringOrLong?`).
    /// Functional Utility: Tests setting a nullable union type attribute.
    /// @param _ The `Option<ByteStringOrLong>` value to set.
    fn SetUnion6AttributeNullable(&self, _: Option<ByteStringOrLong>) {}
    /// A method testing binary renamed.
    /// Functional Utility: Tests how WebIDL handles method names that are keywords in Rust,
    /// requiring renaming in the generated binding. This specific method tests `binary` renamed.
    fn BinaryRenamedMethod(&self) {}
    /// A method that receives no arguments and returns nothing.
    /// Functional Utility: Tests the basic binding of a void method with no parameters.
    fn ReceiveVoid(&self) {}
    /// A method that receives no arguments and returns a boolean.
    /// Functional Utility: Tests the basic binding of a method that returns a `boolean`.
    /// @return A boolean value.
    fn ReceiveBoolean(&self) -> bool {
        false
    }
    /// A method that receives no arguments and returns a byte.
    /// Functional Utility: Tests the basic binding of a method that returns a `byte`.
    /// @return A signed 8-bit integer value.
    fn ReceiveByte(&self) -> i8 {
        0
    }
    /// A method that receives no arguments and returns an octet.
    /// Functional Utility: Tests the basic binding of a method that returns an `octet`.
    /// @return An unsigned 8-bit integer value.
    fn ReceiveOctet(&self) -> u8 {
        0
    }
    /// A method that receives no arguments and returns a short.
    /// Functional Utility: Tests the basic binding of a method that returns a `short`.
    /// @return A signed 16-bit integer value.
    fn ReceiveShort(&self) -> i16 {
        0
    }
    /// A method that receives no arguments and returns an unsigned short.
    /// Functional Utility: Tests the basic binding of a method that returns an `unsigned short`.
    /// @return An unsigned 16-bit integer value.
    fn ReceiveUnsignedShort(&self) -> u16 {
        0
    }
    /// A method that receives no arguments and returns a long.
    /// Functional Utility: Tests the basic binding of a method that returns a `long`.
    /// @return A signed 32-bit integer value.
    fn ReceiveLong(&self) -> i32 {
        0
    }
    /// A method that receives no arguments and returns an unsigned long.
    /// Functional Utility: Tests the basic binding of a method that returns an `unsigned long`.
    /// @return An unsigned 32-bit integer value.
    fn ReceiveUnsignedLong(&self) -> u32 {
        0
    }
    /// A method that receives no arguments and returns a long long.
    /// Functional Utility: Tests the basic binding of a method that returns a `long long`.
    /// @return A signed 64-bit integer value.
    fn ReceiveLongLong(&self) -> i64 {
        0
    }
    /// A method that receives no arguments and returns an unsigned long long.
    /// Functional Utility: Tests the basic binding of a method that returns an `unsigned long long`.
    /// @return An unsigned 64-bit integer value.
    fn ReceiveUnsignedLongLong(&self) -> u64 {
        0
    }
    /// A method that receives no arguments and returns an unrestricted float.
    /// Functional Utility: Tests the basic binding of a method that returns an `unrestricted float`.
    /// @return A 32-bit floating-point value.
    fn ReceiveUnrestrictedFloat(&self) -> f32 {
        0.
    }
    /// A method that receives no arguments and returns a float.
    /// Functional Utility: Tests the basic binding of a method that returns a `float`.
    /// @return A finite 32-bit floating-point value.
    fn ReceiveFloat(&self) -> Finite<f32> {
        Finite::wrap(0.)
    }
    /// A method that receives no arguments and returns an unrestricted double.
    /// Functional Utility: Tests the basic binding of a method that returns an `unrestricted double`.
    /// @return A 64-bit floating-point value.
    fn ReceiveUnrestrictedDouble(&self) -> f64 {
        0.
    }
    /// A method that receives no arguments and returns a double.
    /// Functional Utility: Tests the basic binding of a method that returns a `double`.
    /// @return A finite 64-bit floating-point value.
    fn ReceiveDouble(&self) -> Finite<f64> {
        Finite::wrap(0.)
    }
    /// A method that receives no arguments and returns a DOMString.
    /// Functional Utility: Tests the basic binding of a method that returns a `DOMString`.
    /// @return A `DOMString` value.
    fn ReceiveString(&self) -> DOMString {
        DOMString::new()
    }
    /// A method that receives no arguments and returns a USVString.
    /// Functional Utility: Tests the basic binding of a method that returns a `USVString`.
    /// @return A `USVString` value.
    fn ReceiveUsvstring(&self) -> USVString {
        USVString("".to_owned())
    }
    /// A method that receives no arguments and returns a ByteString.
    /// Functional Utility: Tests the basic binding of a method that returns a `ByteString`.
    /// @return A `ByteString` value.
    fn ReceiveByteString(&self) -> ByteString {
        ByteString::new(vec![])
    }
    /// A method that receives no arguments and returns an enum.
    /// Functional Utility: Tests the basic binding of a method that returns an `enum`.
    /// @return A `TestEnum` value.
    fn ReceiveEnum(&self) -> TestEnum {
        TestEnum::_empty
    }
    /// A method that receives no arguments and returns an interface type (`Blob`).
    /// Functional Utility: Tests the basic binding of a method that returns an interface type.
    /// @param can_gc A `CanGc` token.
    /// @return A `DomRoot<Blob>` value.
    fn ReceiveInterface(&self, can_gc: CanGc) -> DomRoot<Blob> {
        Blob::new(
            &self.global(),
            BlobImpl::new_from_bytes(vec![], "".to_owned()),
            can_gc,
        )
    }
    /// A method that receives no arguments and returns an `any` type.
    /// Functional Utility: Tests the basic binding of a method that returns an `any` type.
    /// @param _cx The JavaScript context.
    /// @param _v A `MutableHandleValue` to receive the returned `any` value.
    fn ReceiveAny(&self, _: SafeJSContext, _: MutableHandleValue) {}
    /// A method that receives no arguments and returns an object type.
    /// Functional Utility: Tests the basic binding of a method that returns an `object` type.
    /// @param cx The JavaScript context.
    /// @return A `NonNull<JSObject>` representing the returned object.
    fn ReceiveObject(&self, cx: SafeJSContext) -> NonNull<JSObject> {
        self.ObjectAttribute(cx)
    }
    /// A method that receives no arguments and returns a union type (`HTMLElementOrLong`).
    /// Functional Utility: Tests the basic binding of a method that returns a union type.
    /// @return An `HTMLElementOrLong` value.
    fn ReceiveUnion(&self) -> HTMLElementOrLong {
        HTMLElementOrLong::Long(0)
    }
    /// A method that receives no arguments and returns a union type (`EventOrString`).
    /// Functional Utility: Tests the basic binding of a method that returns a union type.
    /// @return An `EventOrString` value.
    fn ReceiveUnion2(&self) -> EventOrString {
        EventOrString::String(DOMString::new())
    }
    /// A method that receives no arguments and returns a union type (`StringOrLongSequence`).
    /// Functional Utility: Tests the binding of a method that returns a union of `DOMString` or `sequence<long>`.
    /// @return A `StringOrLongSequence` value.
    fn ReceiveUnion3(&self) -> StringOrLongSequence {
        StringOrLongSequence::LongSequence(vec![])
    }
    /// A method that receives no arguments and returns a union type (`StringOrStringSequence`).
    /// Functional Utility: Tests the binding of a method that returns a union of `DOMString` or `sequence<DOMString>`.
    /// @return A `StringOrStringSequence` value.
    fn ReceiveUnion4(&self) -> StringOrStringSequence {
        StringOrStringSequence::StringSequence(vec![])
    }
    /// A method that receives no arguments and returns a union type (`BlobOrBlobSequence`).
    /// Functional Utility: Tests the binding of a method that returns a union of `Blob` or `sequence<Blob>`.
    /// @return A `BlobOrBlobSequence` value.
    fn ReceiveUnion5(&self) -> BlobOrBlobSequence {
        BlobOrBlobSequence::BlobSequence(vec![])
    }
    /// A method that receives no arguments and returns a union type (`StringOrUnsignedLong`).
    /// Functional Utility: Tests the binding of a method that returns a union of `DOMString` or `unsigned long`.
    /// @return A `StringOrUnsignedLong` value.
    fn ReceiveUnion6(&self) -> StringOrUnsignedLong {
        StringOrUnsignedLong::String(DOMString::new())
    }
    /// A method that receives no arguments and returns a union type (`StringOrBoolean`).
    /// Functional Utility: Tests the binding of a method that returns a union of `DOMString` or `boolean`.
    /// @return A `StringOrBoolean` value.
    fn ReceiveUnion7(&self) -> StringOrBoolean {
        StringOrBoolean::Boolean(true)
    }
    /// A method that receives no arguments and returns a union type (`UnsignedLongOrBoolean`).
    /// Functional Utility: Tests the binding of a method that returns a union of `unsigned long` or `boolean`.
    /// @return An `UnsignedLongOrBoolean` value.
    fn ReceiveUnion8(&self) -> UnsignedLongOrBoolean {
        UnsignedLongOrBoolean::UnsignedLong(0u32)
    }
    /// A method that receives no arguments and returns a union type (`HTMLElementOrUnsignedLongOrStringOrBoolean`).
    /// Functional Utility: Tests the binding of a method that returns a complex union type.
    /// @return An `HTMLElementOrUnsignedLongOrStringOrBoolean` value.
    fn ReceiveUnion9(&self) -> HTMLElementOrUnsignedLongOrStringOrBoolean {
        HTMLElementOrUnsignedLongOrStringOrBoolean::Boolean(true)
    }
    /// A method that receives no arguments and returns a union type (`ByteStringOrLong`).
    /// Functional Utility: Tests the binding of a method that returns a union of `ByteString` or `long`.
    /// @return A `ByteStringOrLong` value.
    fn ReceiveUnion10(&self) -> ByteStringOrLong {
        ByteStringOrLong::ByteString(ByteString::new(vec![]))
    }
    /// A method that receives no arguments and returns a union type (`ByteStringSequenceOrLongOrString`).
    /// Functional Utility: Tests the binding of a method that returns a complex union type with sequences.
    /// @return A `ByteStringSequenceOrLongOrString` value.
    fn ReceiveUnion11(&self) -> ByteStringSequenceOrLongOrString {
        ByteStringSequenceOrLongOrString::ByteStringSequence(vec![ByteString::new(vec![])])
    }
    /// A method that receives no arguments and returns a sequence of longs.
    /// Functional Utility: Tests the basic binding of a method that returns a `sequence<long>`.
    /// @return A `Vec<i32>` representing the sequence.
    fn ReceiveSequence(&self) -> Vec<i32> {
        vec![1]
    }
    /// A method that receives no arguments and returns a sequence of interface types (`sequence<Blob>`).
    /// Functional Utility: Tests the binding of a method that returns a sequence of interface types.
    /// @param can_gc A `CanGc` token.
    /// @return A `Vec<DomRoot<Blob>>` representing the sequence of Blob objects.
    fn ReceiveInterfaceSequence(&self, can_gc: CanGc) -> Vec<DomRoot<Blob>> {
        vec![Blob::new(
            &self.global(),
            BlobImpl::new_from_bytes(vec![], "".to_owned()),
            can_gc,
        )]
    }
    /// A method that receives a union type and returns it unchanged.
    /// Functional Utility: Tests the binding of a method that takes and returns a union type (`StringOrObject`),
    /// verifying identity transformation.
    /// @param _cx The JavaScript context.
    /// @param arg The input `UnionTypes::StringOrObject` value.
    /// @return The input `UnionTypes::StringOrObject` value.
    fn ReceiveUnionIdentity(
        &self,
        _: SafeJSContext,
        arg: UnionTypes::StringOrObject,
    ) -> UnionTypes::StringOrObject {
        arg
    }

    /// A method that receives no arguments and returns a nullable boolean.
    /// Functional Utility: Tests the basic binding of a method that returns a `boolean?`.
    /// @return An `Option<bool>` value.
    fn ReceiveNullableBoolean(&self) -> Option<bool> {
        Some(false)
    }
    /// A method that receives no arguments and returns a nullable byte.
    /// Functional Utility: Tests the basic binding of a method that returns a `byte?`.
    /// @return An `Option<i8>` value.
    fn ReceiveNullableByte(&self) -> Option<i8> {
        Some(0)
    }
    /// A method that receives no arguments and returns a nullable octet.
    /// Functional Utility: Tests the basic binding of a method that returns an `octet?`.
    /// @return An `Option<u8>` value.
    fn ReceiveNullableOctet(&self) -> Option<u8> {
        Some(0)
    }
    /// A method that receives no arguments and returns a nullable short.
    /// Functional Utility: Tests the basic binding of a method that returns a `short?`.
    /// @return An `Option<i16>` value.
    fn ReceiveNullableShort(&self) -> Option<i16> {
        Some(0)
    }
    /// A method that receives no arguments and returns a nullable unsigned short.
    /// Functional Utility: Tests the basic binding of a method that returns an `unsigned short?`.
    /// @return An `Option<u16>` value.
    fn ReceiveNullableUnsignedShort(&self) -> Option<u16> {
        Some(0)
    }
    /// A method that receives no arguments and returns a nullable long.
    /// Functional Utility: Tests the basic binding of a method that returns a `long?`.
    /// @return An `Option<i32>` value.
    fn ReceiveNullableLong(&self) -> Option<i32> {
        Some(0)
    }
    /// A method that receives no arguments and returns a nullable unsigned long.
    /// Functional Utility: Tests the basic binding of a method that returns an `unsigned long?`.
    /// @return An `Option<u32>` value.
    fn ReceiveNullableUnsignedLong(&self) -> Option<u32> {
        Some(0)
    }
    /// A method that receives no arguments and returns a nullable long long.
    /// Functional Utility: Tests the basic binding of a method that returns a `long long?`.
    /// @return An `Option<i64>` value.
    fn ReceiveNullableLongLong(&self) -> Option<i64> {
        Some(0)
    }
    /// A method that receives no arguments and returns a nullable unsigned long long.
    /// Functional Utility: Tests the basic binding of a method that returns an `unsigned long long?`.
    /// @return An `Option<u64>` value.
    fn ReceiveNullableUnsignedLongLong(&self) -> Option<u64> {
        Some(0)
    }
    /// A method that receives no arguments and returns a nullable unrestricted float.
    /// Functional Utility: Tests the basic binding of a method that returns an `unrestricted float?`.
    /// @return An `Option<f32>` value.
    fn ReceiveNullableUnrestrictedFloat(&self) -> Option<f32> {
        Some(0.)
    }
    /// A method that receives no arguments and returns a nullable float.
    /// Functional Utility: Tests the basic binding of a method that returns a `float?`.
    /// @return An `Option<Finite<f32>>` value.
    fn ReceiveNullableFloat(&self) -> Option<Finite<f32>> {
        Some(Finite::wrap(0.))
    }
    /// A method that receives no arguments and returns a nullable unrestricted double.
    /// Functional Utility: Tests the basic binding of a method that returns an `unrestricted double?`.
    /// @return An `Option<f64>` value.
    fn ReceiveNullableUnrestrictedDouble(&self) -> Option<f64> {
        Some(0.)
    }
    /// A method that receives no arguments and returns a nullable double.
    /// Functional Utility: Tests the basic binding of a method that returns a `double?`.
    /// @return An `Option<Finite<f64>>` value.
    fn ReceiveNullableDouble(&self) -> Option<Finite<f64>> {
        Some(Finite::wrap(0.))
    }
    /// A method that receives no arguments and returns a nullable DOMString.
    /// Functional Utility: Tests the basic binding of a method that returns a `DOMString?`.
    /// @return An `Option<DOMString>` value.
    fn ReceiveNullableString(&self) -> Option<DOMString> {
        Some(DOMString::new())
    }
    /// A method that receives no arguments and returns a nullable USVString.
    /// Functional Utility: Tests the basic binding of a method that returns a `USVString?`.
    /// @return An `Option<USVString>` value.
    fn ReceiveNullableUsvstring(&self) -> Option<USVString> {
        Some(USVString("".to_owned()))
    }
    /// A method that receives no arguments and returns a nullable ByteString.
    /// Functional Utility: Tests the basic binding of a method that returns a `ByteString?`.
    /// @return An `Option<ByteString>` value.
    fn ReceiveNullableByteString(&self) -> Option<ByteString> {
        Some(ByteString::new(vec![]))
    }
    /// A method that receives no arguments and returns a nullable enum.
    /// Functional Utility: Tests the basic binding of a method that returns an `enum?`.
    /// @return An `Option<TestEnum>` value.
    fn ReceiveNullableEnum(&self) -> Option<TestEnum> {
        Some(TestEnum::_empty)
    }
    /// A method that receives no arguments and returns a nullable interface type (`Blob?`).
    /// Functional Utility: Tests the basic binding of a method that returns a nullable interface type.
    /// @param can_gc A `CanGc` token.
    /// @return An `Option<DomRoot<Blob>>` value.
    fn ReceiveNullableInterface(&self, can_gc: CanGc) -> Option<DomRoot<Blob>> {
        Some(Blob::new(
            &self.global(),
            BlobImpl::new_from_bytes(vec![], "".to_owned()),
            can_gc,
        ))
    }
    /// A method that receives no arguments and returns a nullable object.
    /// Functional Utility: Tests the basic binding of a method that returns an `object?`.
    /// @param cx The JavaScript context.
    /// @return An `Option<NonNull<JSObject>>` value.
    fn ReceiveNullableObject(&self, cx: SafeJSContext) -> Option<NonNull<JSObject>> {
        self.GetObjectAttributeNullable(cx)
    }
    /// A method that receives no arguments and returns a nullable union type (`HTMLElementOrLong?`).
    /// Functional Utility: Tests the basic binding of a method that returns a nullable union type.
    /// @return An `Option<HTMLElementOrLong>` value.
    fn ReceiveNullableUnion(&self) -> Option<HTMLElementOrLong> {
        Some(HTMLElementOrLong::Long(0))
    }
    /// A method that receives no arguments and returns a nullable union type (`EventOrString?`).
    /// Functional Utility: Tests the basic binding of a method that returns a nullable union type.
    /// @return An `Option<EventOrString>` value.
    fn ReceiveNullableUnion2(&self) -> Option<EventOrString> {
        Some(EventOrString::String(DOMString::new()))
    }
    /// A method that receives no arguments and returns a nullable union type (`StringOrLongSequence?`).
    /// Functional Utility: Tests the binding of a method that returns a nullable union of `DOMString` or `sequence<long>`.
    /// @return An `Option<StringOrLongSequence>` value.
    fn ReceiveNullableUnion3(&self) -> Option<StringOrLongSequence> {
        Some(StringOrLongSequence::String(DOMString::new()))
    }
    /// A method that receives no arguments and returns a nullable union type (`LongSequenceOrBoolean?`).
    /// Functional Utility: Tests the binding of a method that returns a nullable union of `sequence<long>` or `boolean`.
    /// @return An `Option<LongSequenceOrBoolean>` value.
    fn ReceiveNullableUnion4(&self) -> Option<LongSequenceOrBoolean> {
        Some(LongSequenceOrBoolean::Boolean(true))
    }
    /// A method that receives no arguments and returns a nullable union type (`UnsignedLongOrBoolean?`).
    /// Functional Utility: Tests the binding of a method that returns a nullable union of `unsigned long` or `boolean`.
    /// @return An `Option<UnsignedLongOrBoolean>` value.
    fn ReceiveNullableUnion5(&self) -> Option<UnsignedLongOrBoolean> {
        Some(UnsignedLongOrBoolean::UnsignedLong(0u32))
    }
    /// A method that receives no arguments and returns a nullable union type (`ByteStringOrLong?`).
    /// Functional Utility: Tests the binding of a method that returns a nullable union of `ByteString` or `long`.
    /// @return An `Option<ByteStringOrLong>` value.
    fn ReceiveNullableUnion6(&self) -> Option<ByteStringOrLong> {
        Some(ByteStringOrLong::ByteString(ByteString::new(vec![])))
    }
    /// A method that receives no arguments and returns a nullable sequence of longs.
    /// Functional Utility: Tests the basic binding of a method that returns a `sequence<long>?`.
    /// @return An `Option<Vec<i32>>` representing the nullable sequence.
    fn ReceiveNullableSequence(&self) -> Option<Vec<i32>> {
        Some(vec![1])
    }
    /// A method that returns a `TestDictionary` with specific values, including one on a keyword.
    /// Functional Utility: Tests the binding of dictionaries, including how keywords
    /// might be handled as member names (e.g., `type_`).
    /// @return A `RootedTraceableBox<TestDictionary>` containing a predefined dictionary instance.
    fn ReceiveTestDictionaryWithSuccessOnKeyword(&self) -> RootedTraceableBox<TestDictionary> {
        RootedTraceableBox::new(TestDictionary {
            anyValue: RootedTraceableBox::new(Heap::default()),
            booleanValue: None,
            byteValue: None,
            dict: RootedTraceableBox::new(TestDictionaryDefaults {
                UnrestrictedDoubleValue: 0.0,
                anyValue: RootedTraceableBox::new(Heap::default()),
                arrayValue: Vec::new(),
                booleanValue: false,
                bytestringValue: ByteString::new(vec![]),
                byteValue: 0,
                doubleValue: Finite::new(1.0).unwrap(),
                enumValue: TestEnum::Foo,
                floatValue: Finite::new(1.0).unwrap(),
                longLongValue: 54,
                longValue: 12,
                nullableBooleanValue: None,
                nullableBytestringValue: None,
                nullableByteValue: None,
                nullableDoubleValue: None,
                nullableFloatValue: None,
                nullableLongLongValue: None,
                nullableLongValue: None,
                nullableObjectValue: RootedTraceableBox::new(Heap::default()),
                nullableOctetValue: None,
                nullableShortValue: None,
                nullableStringValue: None,
                nullableUnrestrictedDoubleValue: None,
                nullableUnrestrictedFloatValue: None,
                nullableUnsignedLongLongValue: None,
                nullableUnsignedLongValue: None,
                nullableUnsignedShortValue: None,
                nullableUsvstringValue: None,
                octetValue: 0,
                shortValue: 0,
                stringValue: DOMString::new(),
                unrestrictedFloatValue: 0.0,
                unsignedLongLongValue: 0,
                unsignedLongValue: 0,
                unsignedShortValue: 0,
                usvstringValue: USVString("".to_owned()),
            }),
            doubleValue: None,
            enumValue: None,
            floatValue: None,
            interfaceValue: None,
            longLongValue: None,
            longValue: None,
            objectValue: None,
            octetValue: None,
            requiredValue: true,
            seqDict: None,
            elementSequence: None,
            shortValue: None,
            stringValue: None,
            type_: Some(DOMString::from("success")),
            unrestrictedDoubleValue: None,
            unrestrictedFloatValue: None,
            unsignedLongLongValue: None,
            unsignedLongValue: None,
            unsignedShortValue: None,
            usvstringValue: None,
            nonRequiredNullable: None,
            nonRequiredNullable2: Some(None),
            noCallbackImport: None,
            noCallbackImport2: None,
        })
    }

    /// A method that takes a `TestDictionary` and checks if its values match expected criteria.
    /// Functional Utility: Tests the correct passing and interpretation of dictionary types
    /// from JavaScript to Rust, including checks for specific member values and nullability.
    /// @param arg The `RootedTraceableBox<TestDictionary>` argument to validate.
    /// @return `true` if the dictionary matches the expected values, `false` otherwise.
    fn DictMatchesPassedValues(&self, arg: RootedTraceableBox<TestDictionary>) -> bool {
        arg.type_.as_ref().map(|s| s == "success").unwrap_or(false) &&
            arg.nonRequiredNullable.is_none() &&
            arg.nonRequiredNullable2 == Some(None) &&
            arg.noCallbackImport.is_none() &&
            arg.noCallbackImport2.is_none()
    }

    /// A method that takes a boolean argument.
    /// Functional Utility: Tests passing a `boolean` IDL argument to a Rust method.
    /// @param _ The boolean value to pass.
    fn PassBoolean(&self, _: bool) {}
    /// A method that takes a byte argument.
    /// Functional Utility: Tests passing a `byte` IDL argument (signed 8-bit integer) to a Rust method.
    /// @param _ The byte value to pass.
    fn PassByte(&self, _: i8) {}
    /// A method that takes an octet argument.
    /// Functional Utility: Tests passing an `octet` IDL argument (unsigned 8-bit integer) to a Rust method.
    /// @param _ The octet value to pass.
    fn PassOctet(&self, _: u8) {}
    /// A method that takes a short argument.
    /// Functional Utility: Tests passing a `short` IDL argument (signed 16-bit integer) to a Rust method.
    /// @param _ The short value to pass.
    fn PassShort(&self, _: i16) {}
    /// A method that takes an unsigned short argument.
    /// Functional Utility: Tests passing an `unsigned short` IDL argument (unsigned 16-bit integer) to a Rust method.
    /// @param _ The unsigned short value to pass.
    fn PassUnsignedShort(&self, _: u16) {}
    /// A method that takes a long argument.
    /// Functional Utility: Tests passing a `long` IDL argument (signed 32-bit integer) to a Rust method.
    /// @param _ The long value to pass.
    fn PassLong(&self, _: i32) {}
    /// A method that takes an unsigned long argument.
    /// Functional Utility: Tests passing an `unsigned long` IDL argument (unsigned 32-bit integer) to a Rust method.
    /// @param _ The unsigned long value to pass.
    fn PassUnsignedLong(&self, _: u32) {}
    /// A method that takes a long long argument.
    /// Functional Utility: Tests passing a `long long` IDL argument (signed 64-bit integer) to a Rust method.
    /// @param _ The long long value to pass.
    fn PassLongLong(&self, _: i64) {}
    /// A method that takes an unsigned long long argument.
    /// Functional Utility: Tests passing an `unsigned long long` IDL argument (unsigned 64-bit integer) to a Rust method.
    /// @param _ The unsigned long long value to pass.
    fn PassUnsignedLongLong(&self, _: u64) {}
    /// A method that takes an unrestricted float argument.
    /// Functional Utility: Tests passing an `unrestricted float` IDL argument (32-bit floating-point) to a Rust method.
    /// @param _ The unrestricted float value to pass.
    fn PassUnrestrictedFloat(&self, _: f32) {}
    /// A method that takes a float argument.
    /// Functional Utility: Tests passing a `float` IDL argument (32-bit floating-point, guaranteed finite) to a Rust method.
    /// @param _ The finite float value to pass.
    fn PassFloat(&self, _: Finite<f32>) {}
    /// A method that takes an unrestricted double argument.
    /// Functional Utility: Tests passing an `unrestricted double` IDL argument (64-bit floating-point) to a Rust method.
    /// @param _ The unrestricted double value to pass.
    fn PassUnrestrictedDouble(&self, _: f64) {}
    /// A method that takes a double argument.
    /// Functional Utility: Tests passing a `double` IDL argument (64-bit floating-point, guaranteed finite) to a Rust method.
    /// @param _ The finite double value to pass.
    fn PassDouble(&self, _: Finite<f64>) {}
    /// A method that takes a DOMString argument.
    /// Functional Utility: Tests passing a `DOMString` IDL argument to a Rust method.
    /// @param _ The `DOMString` value to pass.
    fn PassString(&self, _: DOMString) {}
    /// A method that takes a USVString argument.
    /// Functional Utility: Tests passing a `USVString` IDL argument to a Rust method.
    /// @param _ The `USVString` value to pass.
    fn PassUsvstring(&self, _: USVString) {}
    /// A method that takes a ByteString argument.
    /// Functional Utility: Tests passing a `ByteString` IDL argument to a Rust method.
    /// @param _ The `ByteString` value to pass.
    fn PassByteString(&self, _: ByteString) {}
    /// A method that takes an enum argument.
    /// Functional Utility: Tests passing an `enum` IDL argument to a Rust method.
    /// @param _ The `TestEnum` value to pass.
    fn PassEnum(&self, _: TestEnum) {}
    /// A method that takes an interface argument (`Blob`).
    /// Functional Utility: Tests passing an interface type (`Blob`) IDL argument to a Rust method.
    /// @param _ The `Blob` object to pass.
    fn PassInterface(&self, _: &Blob) {}
    /// A method that takes a TypedArray argument (`Int8Array`).
    /// Functional Utility: Tests passing an IDL `TypedArray` (`Int8Array`) to a Rust method.
    /// @param _ The `CustomAutoRooterGuard<typedarray::Int8Array>` value to pass.
    fn PassTypedArray(&self, _: CustomAutoRooterGuard<typedarray::Int8Array>) {}
    /// A method that takes a TypedArray argument (`ArrayBuffer`).
    /// Functional Utility: Tests passing an IDL `TypedArray` (`ArrayBuffer`) to a Rust method.
    /// @param _ The `CustomAutoRooterGuard<typedarray::ArrayBuffer>` value to pass.
    fn PassTypedArray2(&self, _: CustomAutoRooterGuard<typedarray::ArrayBuffer>) {}
    /// A method that takes a TypedArray argument (`ArrayBufferView`).
    /// Functional Utility: Tests passing an IDL `TypedArray` (`ArrayBufferView`) to a Rust method.
    /// @param _ The `CustomAutoRooterGuard<typedarray::ArrayBufferView>` value to pass.
    fn PassTypedArray3(&self, _: CustomAutoRooterGuard<typedarray::ArrayBufferView>) {}    /// A method that takes a union argument (`HTMLElementOrLong`).
    /// Functional Utility: Tests passing a union type IDL argument to a Rust method.
    /// @param _ The `HTMLElementOrLong` value to pass.
    fn PassUnion(&self, _: HTMLElementOrLong) {}
    /// A method that takes a union argument (`EventOrString`).
    /// Functional Utility: Tests passing a union type IDL argument to a Rust method.
    /// @param _ The `EventOrString` value to pass.
    fn PassUnion2(&self, _: EventOrString) {}
    /// A method that takes a union argument (`BlobOrString`).
    /// Functional Utility: Tests passing a union type IDL argument to a Rust method.
    /// @param _ The `BlobOrString` value to pass.
    fn PassUnion3(&self, _: BlobOrString) {}
    /// A method that takes a union argument (`StringOrStringSequence`).
    /// Functional Utility: Tests passing a union type IDL argument to a Rust method.
    /// @param _ The `StringOrStringSequence` value to pass.
    fn PassUnion4(&self, _: StringOrStringSequence) {}
    /// A method that takes a union argument (`StringOrBoolean`).
    /// Functional Utility: Tests passing a union type IDL argument to a Rust method.
    /// @param _ The `StringOrBoolean` value to pass.
    fn PassUnion5(&self, _: StringOrBoolean) {}
    /// A method that takes a union argument (`UnsignedLongOrBoolean`).
    /// Functional Utility: Tests passing a union type IDL argument to a Rust method.
    /// @param _ The `UnsignedLongOrBoolean` value to pass.
    fn PassUnion6(&self, _: UnsignedLongOrBoolean) {}
    /// A method that takes a union argument (`StringSequenceOrUnsignedLong`).
    /// Functional Utility: Tests passing a union type IDL argument to a Rust method.
    /// @param _ The `StringSequenceOrUnsignedLong` value to pass.
    fn PassUnion7(&self, _: StringSequenceOrUnsignedLong) {}
    /// A method that takes a union argument (`ByteStringSequenceOrLong`).
    /// Functional Utility: Tests passing a union type IDL argument to a Rust method.
    /// @param _ The `ByteStringSequenceOrLong` value to pass.
    fn PassUnion8(&self, _: ByteStringSequenceOrLong) {}
    /// A method that takes a union argument (`TestDictionaryOrLong`).
    /// Functional Utility: Tests passing a union type IDL argument to a Rust method.
    /// @param _ The `UnionTypes::TestDictionaryOrLong` value to pass.
    fn PassUnion9(&self, _: UnionTypes::TestDictionaryOrLong) {}
    /// A method that takes a union argument (`StringOrObject`).
    /// Functional Utility: Tests passing a union type IDL argument to a Rust method.
    /// @param _cx The JavaScript context.
    /// @param _ The `UnionTypes::StringOrObject` value to pass.
    fn PassUnion10(&self, _: SafeJSContext, _: UnionTypes::StringOrObject) {}
    /// A method that takes a union argument (`ArrayBufferOrArrayBufferView`).
    /// Functional Utility: Tests passing a union type IDL argument to a Rust method.
    /// @param _ The `UnionTypes::ArrayBufferOrArrayBufferView` value to pass.
    fn PassUnion11(&self, _: UnionTypes::ArrayBufferOrArrayBufferView) {}
    /// A method that takes a union argument with a typedef (`DocumentOrStringOrURLOrBlob`).
    /// Functional Utility: Tests passing a union type IDL argument that includes a typedef.
    /// @param _ The `UnionTypes::DocumentOrStringOrURLOrBlob` value to pass.
    fn PassUnionWithTypedef(&self, _: UnionTypes::DocumentOrStringOrURLOrBlob) {}
    /// A method that takes a union argument with a typedef (`LongSequenceOrStringOrURLOrBlob`).
    /// Functional Utility: Tests passing another union type IDL argument that includes a typedef.
    /// @param _ The `UnionTypes::LongSequenceOrStringOrURLOrBlob` value to pass.
    fn PassUnionWithTypedef2(&self, _: UnionTypes::LongSequenceOrStringOrURLOrBlob) {}
    /// A method that takes an `any` argument.
    /// Functional Utility: Tests passing an `any` IDL argument to a Rust method.
    /// @param _cx The JavaScript context.
    /// @param _ The `HandleValue` to pass.
    fn PassAny(&self, _: SafeJSContext, _: HandleValue) {}
    /// A method that takes an object argument.
    /// Functional Utility: Tests passing an IDL `object` to a Rust method.
    /// @param _cx The JavaScript context.
    /// @param _ The raw pointer to the `JSObject` to pass.
    fn PassObject(&self, _: SafeJSContext, _: *mut JSObject) {}
    /// A method that takes a callback function argument.
    /// Functional Utility: Tests passing an IDL `Function` callback to a Rust method.
    /// @param _ The `Rc<Function>` object representing the callback.
    fn PassCallbackFunction(&self, _: Rc<Function>) {}
    /// A method that takes a callback interface argument.
    /// Functional Utility: Tests passing an IDL `EventListener` callback interface to a Rust method.
    /// @param _ The `Rc<EventListener>` object representing the callback interface.
    fn PassCallbackInterface(&self, _: Rc<EventListener>) {}
    /// A method that takes a sequence argument (`sequence<long>`).
    /// Functional Utility: Tests passing an IDL `sequence` to a Rust method.
    /// @param _ The `Vec<i32>` representing the sequence of long values.
    fn PassSequence(&self, _: Vec<i32>) {}
    /// A method that takes a sequence of `any` arguments (`sequence<any>`).
    /// Functional Utility: Tests passing an IDL `sequence<any>` to a Rust method.
    /// @param _cx The JavaScript context.
    /// @param _ A `CustomAutoRooterGuard<Vec<JSVal>>` representing the sequence of `any` values.
    fn PassAnySequence(&self, _: SafeJSContext, _: CustomAutoRooterGuard<Vec<JSVal>>) {}
    /// A method that takes a sequence of `any` arguments and returns it unchanged.
    /// Functional Utility: Tests passing and returning an IDL `sequence<any>`, verifying data integrity.
    /// @param _cx The JavaScript context.
    /// @param seq A `CustomAutoRooterGuard<Vec<JSVal>>` representing the input sequence.
    /// @return A `Vec<JSVal>` that is a clone of the input sequence.
    fn AnySequencePassthrough(
        &self,
        _: SafeJSContext,
        seq: CustomAutoRooterGuard<Vec<JSVal>>,
    ) -> Vec<JSVal> {
        (*seq).clone()
    }
    /// A method that takes a sequence of `object` arguments (`sequence<object>`).
    /// Functional Utility: Tests passing an IDL `sequence<object>` to a Rust method.
    /// @param _cx The JavaScript context.
    /// @param _ A `CustomAutoRooterGuard<Vec<*mut JSObject>>` representing the sequence of `JSObject` pointers.
    fn PassObjectSequence(&self, _: SafeJSContext, _: CustomAutoRooterGuard<Vec<*mut JSObject>>) {}
    /// A method that takes a sequence of `DOMString` arguments (`sequence<DOMString>`).
    /// Functional Utility: Tests passing an IDL `sequence<DOMString>` to a Rust method.
    /// @param _ The `Vec<DOMString>` representing the sequence.
    fn PassStringSequence(&self, _: Vec<DOMString>) {}
    /// A method that takes a sequence of interface arguments (`sequence<Blob>`).
    /// Functional Utility: Tests passing an IDL `sequence<Blob>` to a Rust method.
    /// @param _ The `Vec<DomRoot<Blob>>` representing the sequence of Blob objects.
    fn PassInterfaceSequence(&self, _: Vec<DomRoot<Blob>>) {}

    /// Overloaded method. Tests passing an `ArrayBuffer`.
    /// Functional Utility: Tests how WebIDL handles overloaded methods with an `ArrayBuffer` argument.
    /// @param _ The `CustomAutoRooterGuard<typedarray::ArrayBuffer>` value to pass.
    fn PassOverloaded(&self, _: CustomAutoRooterGuard<typedarray::ArrayBuffer>) {}
    /// Overloaded method. Tests passing a `DOMString`.
    /// Functional Utility: Tests how WebIDL handles overloaded methods with a `DOMString` argument.
    /// @param _ The `DOMString` value to pass.
    fn PassOverloaded_(&self, _: DOMString) {}

    /// Overloaded method. Tests passing a `Node` dictionary.
    /// Functional Utility: Tests how WebIDL handles overloaded methods where one overload
    /// takes an interface type (`Node`).
    /// @param _ The `Node` object to pass.
    /// @return A `DOMString` indicating the type of argument received.
    fn PassOverloadedDict(&self, _: &Node) -> DOMString {
        "node".into()
    }

    /// Overloaded method. Tests passing a `TestURLLike` dictionary.
    /// Functional Utility: Tests how WebIDL handles overloaded methods where another overload
    /// takes a dictionary type (`TestURLLike`).
    /// @param u The `TestURLLike` dictionary to pass.
    /// @return The `href` member of the `TestURLLike` dictionary.
    fn PassOverloadedDict_(&self, u: &TestURLLike) -> DOMString {
        u.href.clone()
    }

    /// A method that takes a nullable boolean argument.
    /// Functional Utility: Tests passing a `boolean?` IDL argument to a Rust method.
    /// @param _ The `Option<bool>` value to pass.
    fn PassNullableBoolean(&self, _: Option<bool>) {}
    /// A method that takes a nullable byte argument.
    /// Functional Utility: Tests passing a `byte?` IDL argument to a Rust method.
    /// @param _ The `Option<i8>` value to pass.
    fn PassNullableByte(&self, _: Option<i8>) {}
    /// A method that takes a nullable octet argument.
    /// Functional Utility: Tests passing an `octet?` IDL argument to a Rust method.
    /// @param _ The `Option<u8>` value to pass.
    fn PassNullableOctet(&self, _: Option<u8>) {}
    /// A method that takes a nullable short argument.
    /// Functional Utility: Tests passing a `short?` IDL argument to a Rust method.
    /// @param _ The `Option<i16>` value to pass.
    fn PassNullableShort(&self, _: Option<i16>) {}
    /// A method that takes a nullable unsigned short argument.
    /// Functional Utility: Tests passing an `unsigned short?` IDL argument to a Rust method.
    /// @param _ The `Option<u16>` value to pass.
    fn PassNullableUnsignedShort(&self, _: Option<u16>) {}
    /// A method that takes a nullable long argument.
    /// Functional Utility: Tests passing a `long?` IDL argument to a Rust method.
    /// @param _ The `Option<i32>` value to pass.
    fn PassNullableLong(&self, _: Option<i32>) {}
    /// A method that takes a nullable unsigned long argument.
    /// Functional Utility: Tests passing an `unsigned long?` IDL argument to a Rust method.
    /// @param _ The `Option<u32>` value to pass.
    fn PassNullableUnsignedLong(&self, _: Option<u32>) {}
    /// A method that takes a nullable long long argument.
    /// Functional Utility: Tests passing a `long long?` IDL argument to a Rust method.
    /// @param _ The `Option<i64>` value to pass.
    fn PassNullableLongLong(&self, _: Option<i64>) {}
    /// A method that takes a nullable unsigned long long argument.
    /// Functional Utility: Tests passing an `unsigned long long?` IDL argument to a Rust method.
    /// @param _ The `Option<u64>` value to pass.
    fn PassNullableUnsignedLongLong(&self, _: Option<u64>) {}
    /// A method that takes a nullable unrestricted float argument.
    /// Functional Utility: Tests passing an `unrestricted float?` IDL argument to a Rust method.
    /// @param _ The `Option<f32>` value to pass.
    fn PassNullableUnrestrictedFloat(&self, _: Option<f32>) {}
    /// A method that takes a nullable float argument.
    /// Functional Utility: Tests passing a `float?` IDL argument to a Rust method.
    /// @param _ The `Option<Finite<f32>>` value to pass.
    fn PassNullableFloat(&self, _: Option<Finite<f32>>) {}
    /// A method that takes a nullable unrestricted double argument.
    /// Functional Utility: Tests passing an `unrestricted double?` IDL argument to a Rust method.
    /// @param _ The `Option<f64>` value to pass.
    fn PassNullableUnrestrictedDouble(&self, _: Option<f64>) {}
    /// A method that takes a nullable double argument.
    /// Functional Utility: Tests passing a `double?` IDL argument to a Rust method.
    /// @param _ The `Option<Finite<f64>>` value to pass.
    fn PassNullableDouble(&self, _: Option<Finite<f64>>) {}
    /// A method that takes a nullable DOMString argument.
    /// Functional Utility: Tests passing a `DOMString?` IDL argument to a Rust method.
    /// @param _ The `Option<DOMString>` value to pass.
    fn PassNullableString(&self, _: Option<DOMString>) {}
    /// A method that takes a nullable USVString argument.
    /// Functional Utility: Tests passing a `USVString?` IDL argument to a Rust method.
    /// @param _ The `Option<USVString>` value to pass.
    fn PassNullableUsvstring(&self, _: Option<USVString>) {}
    /// A method that takes a nullable ByteString argument.
    /// Functional Utility: Tests passing a `ByteString?` IDL argument to a Rust method.
    /// @param _ The `Option<ByteString>` value to pass.
    fn PassNullableByteString(&self, _: Option<ByteString>) {}
    // fn PassNullableEnum(self, _: Option<TestEnum>) {}
    /// A method that takes a nullable interface argument (`Blob?`).
    /// Functional Utility: Tests passing an interface type (`Blob?`) IDL argument to a Rust method.
    /// @param _ The `Option<&Blob>` object to pass.
    fn PassNullableInterface(&self, _: Option<&Blob>) {}
    /// A method that takes a nullable object argument.
    /// Functional Utility: Tests passing an IDL `object?` to a Rust method.
    /// @param _cx The JavaScript context.
    /// @param _ The raw pointer to the `JSObject` to pass.
    fn PassNullableObject(&self, _: SafeJSContext, _: *mut JSObject) {}
    /// A method that takes a nullable TypedArray argument (`Int8Array?`).
    /// Functional Utility: Tests passing an IDL `TypedArray?` (nullable `Int8Array`) to a Rust method.
    /// @param _ The `CustomAutoRooterGuard<Option<typedarray::Int8Array>>` value to pass.
    fn PassNullableTypedArray(&self, _: CustomAutoRooterGuard<Option<typedarray::Int8Array>>) {}
    /// A method that takes a nullable union argument (`HTMLElementOrLong?`).
    /// Functional Utility: Tests passing a nullable union type IDL argument to a Rust method.
    /// @param _ The `Option<HTMLElementOrLong>` value to pass.
    fn PassNullableUnion(&self, _: Option<HTMLElementOrLong>) {}
    /// A method that takes a nullable union argument (`EventOrString?`).
    /// Functional Utility: Tests passing a nullable union type IDL argument to a Rust method.
    /// @param _ The `Option<EventOrString>` value to pass.
    fn PassNullableUnion2(&self, _: Option<EventOrString>) {}
    /// A method that takes a nullable union argument (`StringOrLongSequence?`).
    /// Functional Utility: Tests passing a nullable union type IDL argument to a Rust method.
    /// @param _ The `Option<StringOrLongSequence>` value to pass.
    fn PassNullableUnion3(&self, _: Option<StringOrLongSequence>) {}
    /// A method that takes a nullable union argument (`LongSequenceOrBoolean?`).
    /// Functional Utility: Tests passing a nullable union type IDL argument to a Rust method.
    /// @param _ The `Option<LongSequenceOrBoolean>` value to pass.
    fn PassNullableUnion4(&self, _: Option<LongSequenceOrBoolean>) {}
    /// A method that takes a nullable union argument (`UnsignedLongOrBoolean?`).
    /// Functional Utility: Tests passing a nullable union type IDL argument to a Rust method.
    /// @param _ The `Option<UnsignedLongOrBoolean>` value to pass.
    fn PassNullableUnion5(&self, _: Option<UnsignedLongOrBoolean>) {}
    /// A method that takes a nullable union argument (`ByteStringOrLong?`).
    /// Functional Utility: Tests passing a nullable union type IDL argument to a Rust method.
    /// @param _ The `Option<ByteStringOrLong>` value to pass.
    fn PassNullableUnion6(&self, _: Option<ByteStringOrLong>) {}
    /// A method that takes a nullable callback function argument (`Function?`).
    /// Functional Utility: Tests passing a nullable IDL `Function` callback to a Rust method.
    /// @param _ The `Option<Rc<Function>>` object representing the callback.
    fn PassNullableCallbackFunction(&self, _: Option<Rc<Function>>) {}
    /// A method that takes a nullable callback interface argument (`EventListener?`).
    /// Functional Utility: Tests passing a nullable IDL `EventListener` callback interface to a Rust method.
    /// @param _ The `Option<Rc<EventListener>>` object representing the callback interface.
    fn PassNullableCallbackInterface(&self, _: Option<Rc<EventListener>>) {}
    /// A method that takes a nullable sequence of longs argument (`sequence<long>?`).
    /// Functional Utility: Tests passing a nullable IDL `sequence` to a Rust method.
    /// @param _ The `Option<Vec<i32>>` representing the nullable sequence of long values.
    fn PassNullableSequence(&self, _: Option<Vec<i32>>) {}

    /// A method that takes an optional boolean argument.
    /// Functional Utility: Tests passing an optional `boolean` IDL argument (`boolean=true`) to a Rust method.
    /// @param _ The `Option<bool>` value to pass.
    fn PassOptionalBoolean(&self, _: Option<bool>) {}
    /// A method that takes an optional byte argument.
    /// Functional Utility: Tests passing an optional `byte` IDL argument to a Rust method.
    /// @param _ The `Option<i8>` value to pass.
    fn PassOptionalByte(&self, _: Option<i8>) {}
    /// A method that takes an optional octet argument.
    /// Functional Utility: Tests passing an optional `octet` IDL argument to a Rust method.
    /// @param _ The `Option<u8>` value to pass.
    fn PassOptionalOctet(&self, _: Option<u8>) {}
    /// A method that takes an optional short argument.
    /// Functional Utility: Tests passing an optional `short` IDL argument to a Rust method.
    /// @param _ The `Option<i16>` value to pass.
    fn PassOptionalShort(&self, _: Option<i16>) {}
    /// A method that takes an optional unsigned short argument.
    /// Functional Utility: Tests passing an optional `unsigned short` IDL argument to a Rust method.
    /// @param _ The `Option<u16>` value to pass.
    fn PassOptionalUnsignedShort(&self, _: Option<u16>) {}
    /// A method that takes an optional long argument.
    /// Functional Utility: Tests passing an optional `long` IDL argument to a Rust method.
    /// @param _ The `Option<i32>` value to pass.
    fn PassOptionalLong(&self, _: Option<i32>) {}
    /// A method that takes an optional unsigned long argument.
    /// Functional Utility: Tests passing an optional `unsigned long` IDL argument to a Rust method.
    /// @param _ The `Option<u32>` value to pass.
    fn PassOptionalUnsignedLong(&self, _: Option<u32>) {}
    /// A method that takes an optional long long argument.
    /// Functional Utility: Tests passing an optional `long long` IDL argument to a Rust method.
    /// @param _ The `Option<i64>` value to pass.
    fn PassOptionalLongLong(&self, _: Option<i64>) {}
    /// A method that takes an optional unsigned long long argument.
    /// Functional Utility: Tests passing an optional `unsigned long long` IDL argument to a Rust method.
    /// @param _ The `Option<u64>` value to pass.
    fn PassOptionalUnsignedLongLong(&self, _: Option<u64>) {}
    /// A method that takes an optional unrestricted float argument.
    /// Functional Utility: Tests passing an optional `unrestricted float` IDL argument to a Rust method.
    /// @param _ The `Option<f32>` value to pass.
    fn PassOptionalUnrestrictedFloat(&self, _: Option<f32>) {}
    /// A method that takes an optional float argument.
    /// Functional Utility: Tests passing an optional `float` IDL argument to a Rust method.
    /// @param _ The `Option<Finite<f32>>` value to pass.
    fn PassOptionalFloat(&self, _: Option<Finite<f32>>) {}
    /// A method that takes an optional unrestricted double argument.
    /// Functional Utility: Tests passing an optional `unrestricted double` IDL argument to a Rust method.
    /// @param _ The `Option<f64>` value to pass.
    fn PassOptionalUnrestrictedDouble(&self, _: Option<f64>) {}
    /// A method that takes an optional double argument.
    /// Functional Utility: Tests passing an optional `double` IDL argument to a Rust method.
    /// @param _ The `Option<Finite<f64>>` value to pass.
    fn PassOptionalDouble(&self, _: Option<Finite<f64>>) {}
    /// A method that takes an optional DOMString argument.
    /// Functional Utility: Tests passing an optional `DOMString` IDL argument to a Rust method.
    /// @param _ The `Option<DOMString>` value to pass.
    fn PassOptionalString(&self, _: Option<DOMString>) {}
    /// A method that takes an optional USVString argument.
    /// Functional Utility: Tests passing an optional `USVString` IDL argument to a Rust method.
    /// @param _ The `Option<USVString>` value to pass.
    fn PassOptionalUsvstring(&self, _: Option<USVString>) {}
    /// A method that takes an optional ByteString argument.
    /// Functional Utility: Tests passing an optional `ByteString` IDL argument to a Rust method.
    /// @param _ The `Option<ByteString>` value to pass.
    fn PassOptionalByteString(&self, _: Option<ByteString>) {}
    /// A method that takes an optional enum argument.
    /// Functional Utility: Tests passing an optional `enum` IDL argument to a Rust method.
    /// @param _ The `Option<TestEnum>` value to pass.
    fn PassOptionalEnum(&self, _: Option<TestEnum>) {}
    /// A method that takes an optional interface argument (`Blob`).
    /// Functional Utility: Tests passing an optional `Blob` interface IDL argument to a Rust method.
    /// @param _ The `Option<&Blob>` value to pass.
    fn PassOptionalInterface(&self, _: Option<&Blob>) {}
    /// A method that takes an optional union argument (`HTMLElementOrLong`).
    /// Functional Utility: Tests passing an optional union type IDL argument to a Rust method.
    /// @param _ The `Option<HTMLElementOrLong>` value to pass.
    fn PassOptionalUnion(&self, _: Option<HTMLElementOrLong>) {}
    /// A method that takes an optional union argument (`EventOrString`).
    /// Functional Utility: Tests passing an optional union type IDL argument to a Rust method.
    /// @param _ The `Option<EventOrString>` value to pass.
    fn PassOptionalUnion2(&self, _: Option<EventOrString>) {}
    /// A method that takes an optional union argument (`StringOrLongSequence`).
    /// Functional Utility: Tests passing an optional union type IDL argument to a Rust method.
    /// @param _ The `Option<StringOrLongSequence>` value to pass.
    fn PassOptionalUnion3(&self, _: Option<StringOrLongSequence>) {}
    /// A method that takes an optional union argument (`LongSequenceOrBoolean`).
    /// Functional Utility: Tests passing an optional union type IDL argument to a Rust method.
    /// @param _ The `Option<LongSequenceOrBoolean>` value to pass.
    fn PassOptionalUnion4(&self, _: Option<LongSequenceOrBoolean>) {}
    /// A method that takes an optional union argument (`UnsignedLongOrBoolean`).
    /// Functional Utility: Tests passing an optional union type IDL argument to a Rust method.
    /// @param _ The `Option<UnsignedLongOrBoolean>` value to pass.
    fn PassOptionalUnion5(&self, _: Option<UnsignedLongOrBoolean>) {}
    /// A method that takes an optional union argument (`ByteStringOrLong`).
    /// Functional Utility: Tests passing an optional union type IDL argument to a Rust method.
    /// @param _ The `Option<ByteStringOrLong>` value to pass.
    fn PassOptionalUnion6(&self, _: Option<ByteStringOrLong>) {}
    /// A method that takes an optional `any` argument.
    /// Functional Utility: Tests passing an optional `any` IDL argument to a Rust method.
    /// @param _cx The JavaScript context.
    /// @param _ The `HandleValue` to pass.
    fn PassOptionalAny(&self, _: SafeJSContext, _: HandleValue) {}
    /// A method that takes an optional object argument.
    /// Functional Utility: Tests passing an optional IDL `object` to a Rust method.
    /// @param _cx The JavaScript context.
    /// @param _ The `Option<*mut JSObject>` value to pass.
    fn PassOptionalObject(&self, _: SafeJSContext, _: Option<*mut JSObject>) {}
    /// A method that takes an optional callback function argument.
    /// Functional Utility: Tests passing an optional IDL `Function` callback to a Rust method.
    /// @param _ The `Option<Rc<Function>>` object representing the callback.
    fn PassOptionalCallbackFunction(&self, _: Option<Rc<Function>>) {}
    /// A method that takes an optional callback interface argument.
    /// Functional Utility: Tests passing an optional IDL `EventListener` callback interface to a Rust method.
    /// @param _ The `Option<Rc<EventListener>>` object representing the callback interface.
    fn PassOptionalCallbackInterface(&self, _: Option<Rc<EventListener>>) {}
    /// A method that takes an optional sequence of longs argument (`sequence<long>`).
    /// Functional Utility: Tests passing an optional IDL `sequence` to a Rust method.
    /// @param _ The `Option<Vec<i32>>` representing the optional sequence of long values.
    fn PassOptionalSequence(&self, _: Option<Vec<i32>>) {}

    /// A method that takes an optional nullable boolean argument.
    /// Functional Utility: Tests passing an optional `boolean?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<bool>>` value to pass.
    fn PassOptionalNullableBoolean(&self, _: Option<Option<bool>>) {}
    /// A method that takes an optional nullable byte argument.
    /// Functional Utility: Tests passing an optional `byte?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<i8>>` value to pass.
    fn PassOptionalNullableByte(&self, _: Option<Option<i8>>) {}
    /// A method that takes an optional nullable octet argument.
    /// Functional Utility: Tests passing an optional `octet?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<u8>>` value to pass.
    fn PassOptionalNullableOctet(&self, _: Option<Option<u8>>) {}
    /// A method that takes an optional nullable short argument.
    /// Functional Utility: Tests passing an optional `short?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<i16>>` value to pass.
    fn PassOptionalNullableShort(&self, _: Option<Option<i16>>) {}
    /// A method that takes an optional nullable unsigned short argument.
    /// Functional Utility: Tests passing an optional `unsigned short?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<u16>>` value to pass.
    fn PassOptionalNullableUnsignedShort(&self, _: Option<Option<u16>>) {}
    /// A method that takes an optional nullable long argument.
    /// Functional Utility: Tests passing an optional `long?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<i32>>` value to pass.
    fn PassOptionalNullableLong(&self, _: Option<Option<i32>>) {}
    /// A method that takes an optional nullable unsigned long argument.
    /// Functional Utility: Tests passing an optional `unsigned long?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<u32>>` value to pass.
    fn PassOptionalNullableUnsignedLong(&self, _: Option<Option<u32>>) {}
    /// A method that takes an optional nullable long long argument.
    /// Functional Utility: Tests passing an optional `long long?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<i64>>` value to pass.
    fn PassOptionalNullableLongLong(&self, _: Option<Option<i64>>) {}
    /// A method that takes an optional nullable unsigned long long argument.
    /// Functional Utility: Tests passing an optional `unsigned long long?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<u64>>` value to pass.
    fn PassOptionalNullableUnsignedLongLong(&self, _: Option<Option<u64>>) {}
    /// A method that takes an optional nullable unrestricted float argument.
    /// Functional Utility: Tests passing an optional `unrestricted float?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<f32>>` value to pass.
    fn PassOptionalNullableUnrestrictedFloat(&self, _: Option<Option<f32>>) {}
    /// A method that takes an optional nullable float argument.
    /// Functional Utility: Tests passing an optional `float?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<Finite<f32>>>` value to pass.
    fn PassOptionalNullableFloat(&self, _: Option<Option<Finite<f32>>>) {}
    /// A method that takes an optional nullable unrestricted double argument.
    /// Functional Utility: Tests passing an optional `unrestricted double?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<f64>>` value to pass.
    fn PassOptionalNullableUnrestrictedDouble(&self, _: Option<Option<f64>>) {}
    /// A method that takes an optional nullable double argument.
    /// Functional Utility: Tests passing an optional `double?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<Finite<f64>>>` value to pass.
    fn PassOptionalNullableDouble(&self, _: Option<Option<Finite<f64>>>) {}
    /// A method that takes an optional nullable DOMString argument.
    /// Functional Utility: Tests passing an optional `DOMString?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<DOMString>>` value to pass.
    fn PassOptionalNullableString(&self, _: Option<Option<DOMString>>) {}
    /// A method that takes an optional nullable USVString argument.
    /// Functional Utility: Tests passing an optional `USVString?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<USVString>>` value to pass.
    fn PassOptionalNullableUsvstring(&self, _: Option<Option<USVString>>) {}
    /// A method that takes an optional nullable ByteString argument.
    /// Functional Utility: Tests passing an optional `ByteString?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<ByteString>>` value to pass.
    fn PassOptionalNullableByteString(&self, _: Option<Option<ByteString>>) {}
    // fn PassOptionalNullableEnum(self, _: Option<Option<TestEnum>>) {}
    /// A method that takes an optional nullable interface argument (`Blob?`).
    /// Functional Utility: Tests passing an optional `Blob?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<&Blob>>` value to pass.
    fn PassOptionalNullableInterface(&self, _: Option<Option<&Blob>>) {}
    /// A method that takes an optional nullable object argument.
    /// Functional Utility: Tests passing an optional `object?` IDL argument to a Rust method.
    /// @param _cx The JavaScript context.
    /// @param _ The `Option<*mut JSObject>` value to pass.
    fn PassOptionalNullableObject(&self, _: SafeJSContext, _: Option<*mut JSObject>) {}
    /// A method that takes an optional nullable union argument (`HTMLElementOrLong?`).
    /// Functional Utility: Tests passing an optional `HTMLElementOrLong?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<HTMLElementOrLong>>` value to pass.
    fn PassOptionalNullableUnion(&self, _: Option<Option<HTMLElementOrLong>>) {}
    /// A method that takes an optional nullable union argument (`EventOrString?`).
    /// Functional Utility: Tests passing an optional `EventOrString?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<EventOrString>>` value to pass.
    fn PassOptionalNullableUnion2(&self, _: Option<Option<EventOrString>>) {}
    /// A method that takes an optional nullable union argument (`StringOrLongSequence?`).
    /// Functional Utility: Tests passing an optional `StringOrLongSequence?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<StringOrLongSequence>>` value to pass.
    fn PassOptionalNullableUnion3(&self, _: Option<Option<StringOrLongSequence>>) {}
    /// A method that takes an optional nullable union argument (`LongSequenceOrBoolean?`).
    /// Functional Utility: Tests passing an optional `LongSequenceOrBoolean?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<LongSequenceOrBoolean>>` value to pass.
    fn PassOptionalNullableUnion4(&self, _: Option<Option<LongSequenceOrBoolean>>) {}
    /// A method that takes an optional nullable union argument (`UnsignedLongOrBoolean?`).
    /// Functional Utility: Tests passing an optional `UnsignedLongOrBoolean?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<UnsignedLongOrBoolean>>` value to pass.
    fn PassOptionalNullableUnion5(&self, _: Option<Option<UnsignedLongOrBoolean>>) {}
    /// A method that takes an optional nullable union argument (`ByteStringOrLong?`).
    /// Functional Utility: Tests passing an optional `ByteStringOrLong?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<ByteStringOrLong>>` value to pass.
    fn PassOptionalNullableUnion6(&self, _: Option<Option<ByteStringOrLong>>) {}
    /// A method that takes an optional nullable callback function argument (`Function?`).
    /// Functional Utility: Tests passing an optional `Function?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<Rc<Function>>>` value to pass.
    fn PassOptionalNullableCallbackFunction(&self, _: Option<Option<Rc<Function>>>) {}
    /// A method that takes an optional nullable callback interface argument (`EventListener?`).
    /// Functional Utility: Tests passing an optional `EventListener?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<Rc<EventListener>>>` value to pass.
    fn PassOptionalNullableCallbackInterface(&self, _: Option<Option<Rc<EventListener>>>) {}
    /// A method that takes an optional nullable sequence of longs argument (`sequence<long>?`).
    /// Functional Utility: Tests passing an optional `sequence<long>?` IDL argument to a Rust method.
    /// @param _ The `Option<Option<Vec<i32>>>` value to pass.
    fn PassOptionalNullableSequence(&self, _: Option<Option<Vec<i32>>>) {}

    /// A method that takes an optional boolean argument with a default value.
    /// Functional Utility: Tests passing an optional `boolean` IDL argument with a default.
    /// @param _ The boolean value to pass.
    fn PassOptionalBooleanWithDefault(&self, _: bool) {}
    /// A method that takes an optional byte argument with a default value.
    /// Functional Utility: Tests passing an optional `byte` IDL argument with a default.
    /// @param _ The byte value to pass.
    fn PassOptionalByteWithDefault(&self, _: i8) {}
    /// A method that takes an optional octet argument with a default value.
    /// Functional Utility: Tests passing an optional `octet` IDL argument with a default.
    /// @param _ The octet value to pass.
    fn PassOptionalOctetWithDefault(&self, _: u8) {}
    /// A method that takes an optional short argument with a default value.
    /// Functional Utility: Tests passing an optional `short` IDL argument with a default.
    /// @param _ The short value to pass.
    fn PassOptionalShortWithDefault(&self, _: i16) {}
    /// A method that takes an optional unsigned short argument with a default value.
    /// Functional Utility: Tests passing an optional `unsigned short` IDL argument with a default.
    /// @param _ The unsigned short value to pass.
    fn PassOptionalUnsignedShortWithDefault(&self, _: u16) {}
    /// A method that takes an optional long argument with a default value.
    /// Functional Utility: Tests passing an optional `long` IDL argument with a default.
    /// @param _ The long value to pass.
    fn PassOptionalLongWithDefault(&self, _: i32) {}
    /// A method that takes an optional unsigned long argument with a default value.
    /// Functional Utility: Tests passing an optional `unsigned long` IDL argument with a default.
    /// @param _ The unsigned long value to pass.
    fn PassOptionalUnsignedLongWithDefault(&self, _: u32) {}
    /// A method that takes an optional long long argument with a default value.
    /// Functional Utility: Tests passing an optional `long long` IDL argument with a default.
    /// @param _ The long long value to pass.
    fn PassOptionalLongLongWithDefault(&self, _: i64) {}
    /// A method that takes an optional unsigned long long argument with a default value.
    /// Functional Utility: Tests passing an optional `unsigned long long` IDL argument with a default.
    /// @param _ The unsigned long long value to pass.
    fn PassOptionalUnsignedLongLongWithDefault(&self, _: u64) {}
    /// A method that takes an optional DOMString argument with a default value.
    /// Functional Utility: Tests passing an optional `DOMString` IDL argument with a default.
    /// @param _ The `DOMString` value to pass.
    fn PassOptionalStringWithDefault(&self, _: DOMString) {}
    /// A method that takes an optional USVString argument with a default value.
    /// Functional Utility: Tests passing an optional `USVString` IDL argument with a default.
    /// @param _ The `USVString` value to pass.
    fn PassOptionalUsvstringWithDefault(&self, _: USVString) {}
    /// A method that takes an optional ByteString argument with a default value.
    /// Functional Utility: Tests passing an optional `ByteString` IDL argument with a default.
    /// @param _ The `ByteString` value to pass.
    fn PassOptionalBytestringWithDefault(&self, _: ByteString) {}
    /// A method that takes an optional enum argument with a default value.
    /// Functional Utility: Tests passing an optional `enum` IDL argument with a default.
    /// @param _ The `TestEnum` value to pass.
    fn PassOptionalEnumWithDefault(&self, _: TestEnum) {}
    /// A method that takes an optional sequence of longs argument with a default value.
    /// Functional Utility: Tests passing an optional `sequence<long>` IDL argument with a default.
    /// @param _ The `Vec<i32>` representing the sequence of long values to pass.
    fn PassOptionalSequenceWithDefault(&self, _: Vec<i32>) {}

    /// A method that takes an optional nullable boolean argument with a default value.
    /// Functional Utility: Tests passing an optional `boolean?` IDL argument with a default.
    /// @param _ The `Option<bool>` value to pass.
    fn PassOptionalNullableBooleanWithDefault(&self, _: Option<bool>) {}
    /// A method that takes an optional nullable byte argument with a default value.
    /// Functional Utility: Tests passing an optional `byte?` IDL argument with a default.
    /// @param _ The `Option<i8>` value to pass.
    fn PassOptionalNullableByteWithDefault(&self, _: Option<i8>) {}
    /// A method that takes an optional nullable octet argument with a default value.
    /// Functional Utility: Tests passing an optional `octet?` IDL argument with a default.
    /// @param _ The `Option<u8>` value to pass.
    fn PassOptionalNullableOctetWithDefault(&self, _: Option<u8>) {}
    /// A method that takes an optional nullable short argument with a default value.
    /// Functional Utility: Tests passing an optional `short?` IDL argument with a default.
    /// @param _ The `Option<i16>` value to pass.
    fn PassOptionalNullableShortWithDefault(&self, _: Option<i16>) {}
    /// A method that takes an optional nullable unsigned short argument with a default value.
    /// Functional Utility: Tests passing an optional `unsigned short?` IDL argument with a default.
    /// @param _ The `Option<u16>` value to pass.
    fn PassOptionalNullableUnsignedShortWithDefault(&self, _: Option<u16>) {}
    /// A method that takes an optional nullable long argument with a default value.
    /// Functional Utility: Tests passing an optional `long?` IDL argument with a default.
    /// @param _ The `Option<i32>` value to pass.
    fn PassOptionalNullableLongWithDefault(&self, _: Option<i32>) {}
    /// A method that takes an optional nullable unsigned long argument with a default value.
    /// Functional Utility: Tests passing an optional `unsigned long?` IDL argument with a default.
    /// @param _ The `Option<u32>` value to pass.
    fn PassOptionalNullableUnsignedLongWithDefault(&self, _: Option<u32>) {}
    /// A method that takes an optional nullable long long argument with a default value.
    /// Functional Utility: Tests passing an optional `long long?` IDL argument with a default.
    /// @param _ The `Option<i64>` value to pass.
    fn PassOptionalNullableLongLongWithDefault(&self, _: Option<i64>) {}
    /// A method that takes an optional nullable unsigned long long argument with a default value.
    /// Functional Utility: Tests passing an optional `unsigned long long?` IDL argument with a default.
    /// @param _ The `Option<u64>` value to pass.
    fn PassOptionalNullableUnsignedLongLongWithDefault(&self, _: Option<u64>) {}
    // fn PassOptionalNullableUnrestrictedFloatWithDefault(self, _: Option<f32>) {}
    // fn PassOptionalNullableFloatWithDefault(self, _: Option<Finite<f32>>) {}
    // fn PassOptionalNullableUnrestrictedDoubleWithDefault(self, _: Option<f64>) {}
    // fn PassOptionalNullableDoubleWithDefault(self, _: Option<Finite<f64>>) {}
    /// A method that takes an optional nullable DOMString argument with a default value.
    /// Functional Utility: Tests passing an optional `DOMString?` IDL argument with a default.
    /// @param _ The `Option<DOMString>` value to pass.
    fn PassOptionalNullableStringWithDefault(&self, _: Option<DOMString>) {}
    /// A method that takes an optional nullable USVString argument with a default value.
    /// Functional Utility: Tests passing an optional `USVString?` IDL argument with a default.
    /// @param _ The `Option<USVString>` value to pass.
    fn PassOptionalNullableUsvstringWithDefault(&self, _: Option<USVString>) {}
    /// A method that takes an optional nullable ByteString argument with a default value.
    /// Functional Utility: Tests passing an optional `ByteString?` IDL argument with a default.
    /// @param _ The `Option<ByteString>` value to pass.
    fn PassOptionalNullableByteStringWithDefault(&self, _: Option<ByteString>) {}
    // fn PassOptionalNullableEnumWithDefault(self, _: Option<TestEnum>) {}
    /// A method that takes an optional nullable interface argument (`Blob?`) with a default value.
    /// Functional Utility: Tests passing an optional `Blob?` IDL argument with a default.
    /// @param _ The `Option<&Blob>` value to pass.
    fn PassOptionalNullableInterfaceWithDefault(&self, _: Option<&Blob>) {}
    /// A method that takes an optional nullable object argument with a default value.
    /// Functional Utility: Tests passing an optional `object?` IDL argument with a default.
    /// @param _cx The JavaScript context.
    /// @param _ The `*mut JSObject` value to pass.
    fn PassOptionalNullableObjectWithDefault(&self, _: SafeJSContext, _: *mut JSObject) {}
    /// A method that takes an optional nullable union argument (`HTMLElementOrLong?`) with a default value.
    /// Functional Utility: Tests passing an optional `HTMLElementOrLong?` IDL argument with a default.
    /// @param _ The `Option<HTMLElementOrLong>` value to pass.
    fn PassOptionalNullableUnionWithDefault(&self, _: Option<HTMLElementOrLong>) {}
    /// A method that takes an optional nullable union argument (`EventOrString?`) with a default value.
    /// Functional Utility: Tests passing an optional `EventOrString?` IDL argument with a default.
    /// @param _ The `Option<EventOrString>` value to pass.
    fn PassOptionalNullableUnion2WithDefault(&self, _: Option<EventOrString>) {}
    // fn PassOptionalNullableCallbackFunctionWithDefault(self, _: Option<Function>) {}
    /// A method that takes an optional nullable callback interface argument (`EventListener?`) with a default value.
    /// Functional Utility: Tests passing an optional `EventListener?` IDL argument with a default.
    /// @param _ The `Option<Rc<EventListener>>` value to pass.
    fn PassOptionalNullableCallbackInterfaceWithDefault(&self, _: Option<Rc<EventListener>>) {}
    /// A method that takes an optional `any` argument with a default value.
    /// Functional Utility: Tests passing an optional `any` IDL argument with a default.
    /// @param _cx The JavaScript context.
    /// @param _ The `HandleValue` to pass.
    fn PassOptionalAnyWithDefault(&self, _: SafeJSContext, _: HandleValue) {}

    /// A method that takes an optional nullable boolean argument with a non-null default value.
    /// Functional Utility: Tests passing an optional `boolean?` IDL argument with a non-null default.
    /// @param _ The `Option<bool>` value to pass.
    fn PassOptionalNullableBooleanWithNonNullDefault(&self, _: Option<bool>) {}
    /// A method that takes an optional nullable byte argument with a non-null default value.
    /// Functional Utility: Tests passing an optional `byte?` IDL argument with a non-null default.
    /// @param _ The `Option<i8>` value to pass.
    fn PassOptionalNullableByteWithNonNullDefault(&self, _: Option<i8>) {}
    /// A method that takes an optional nullable octet argument with a non-null default value.
    /// Functional Utility: Tests passing an optional `octet?` IDL argument with a non-null default.
    /// @param _ The `Option<u8>` value to pass.
    fn PassOptionalNullableOctetWithNonNullDefault(&self, _: Option<u8>) {}
    /// A method that takes an optional nullable short argument with a non-null default value.
    /// Functional Utility: Tests passing an optional `short?` IDL argument with a non-null default.
    /// @param _ The `Option<i16>` value to pass.
    fn PassOptionalNullableShortWithNonNullDefault(&self, _: Option<i16>) {}
    /// A method that takes an optional nullable unsigned short argument with a non-null default value.
    /// Functional Utility: Tests passing an optional `unsigned short?` IDL argument with a non-null default.
    /// @param _ The `Option<u16>` value to pass.
    fn PassOptionalNullableUnsignedShortWithNonNullDefault(&self, _: Option<u16>) {}
    /// A method that takes an optional nullable long argument with a non-null default value.
    /// Functional Utility: Tests passing an optional `long?` IDL argument with a non-null default.
    /// @param _ The `Option<i32>` value to pass.
    fn PassOptionalNullableLongWithNonNullDefault(&self, _: Option<i32>) {}
    /// A method that takes an optional nullable unsigned long argument with a non-null default value.
    /// Functional Utility: Tests passing an optional `unsigned long?` IDL argument with a non-null default.
    /// @param _ The `Option<u32>` value to pass.
    fn PassOptionalNullableUnsignedLongWithNonNullDefault(&self, _: Option<u32>) {}
    /// A method that takes an optional nullable long long argument with a non-null default value.
    /// Functional Utility: Tests passing an optional `long long?` IDL argument with a non-null default.
    /// @param _ The `Option<i64>` value to pass.
    fn PassOptionalNullableLongLongWithNonNullDefault(&self, _: Option<i64>) {}
    /// A method that takes an optional nullable unsigned long long argument with a non-null default value.
    /// Functional Utility: Tests passing an optional `unsigned long long?` IDL argument with a non-null default.
    /// @param _ The `Option<u64>` value to pass.
    fn PassOptionalNullableUnsignedLongLongWithNonNullDefault(&self, _: Option<u64>) {}
    // fn PassOptionalNullableUnrestrictedFloatWithNonNullDefault(self, _: Option<f32>) {}
    // fn PassOptionalNullableFloatWithNonNullDefault(self, _: Option<Finite<f32>>) {}
    // fn PassOptionalNullableUnrestrictedDoubleWithNonNullDefault(self, _: Option<f64>) {}
    // fn PassOptionalNullableDoubleWithNonNullDefault(self, _: Option<Finite<f64>>) {}
    /// A method that takes an optional nullable DOMString argument with a non-null default value.
    /// Functional Utility: Tests passing an optional `DOMString?` IDL argument with a non-null default.
    /// @param _ The `Option<DOMString>` value to pass.
    fn PassOptionalNullableStringWithNonNullDefault(&self, _: Option<DOMString>) {}
    /// A method that takes an optional nullable USVString argument with a non-null default value.
    /// Functional Utility: Tests passing an optional `USVString?` IDL argument with a non-null default.
    /// @param _ The `Option<USVString>` value to pass.
    fn PassOptionalNullableUsvstringWithNonNullDefault(&self, _: Option<USVString>) {}
    // fn PassOptionalNullableEnumWithNonNullDefault(self, _: Option<TestEnum>) {}
    /// Overloaded method. Tests passing a `TestBinding` object and two unsigned long arguments.
    /// Functional Utility: Tests an overloaded method where the first overload takes
    /// an interface type and two unsigned long integers.
    /// @param a The `TestBinding` object to pass.
    /// @param _ The first unsigned long value.
    /// @param _ The second unsigned long value.
    /// @return A `DomRoot<TestBinding>` referencing the passed `TestBinding` object.
    fn PassOptionalOverloaded(&self, a: &TestBinding, _: u32, _: u32) -> DomRoot<TestBinding> {
        DomRoot::from_ref(a)
    }
    /// Overloaded method. Tests passing a `Blob` object and an unsigned long argument.
    /// Functional Utility: Tests another overload for the `PassOptionalOverloaded` method,
    /// where this overload takes a `Blob` interface and an unsigned long integer.
    /// @param _ The `Blob` object to pass.
    /// @param _ The unsigned long value.
    fn PassOptionalOverloaded_(&self, _: &Blob, _: u32) {}

    /// A method that takes a variadic boolean argument.
    /// Functional Utility: Tests passing a variadic `boolean` IDL argument to a Rust method.
    /// @param _ The `Vec<bool>` representing the variadic boolean arguments.
    fn PassVariadicBoolean(&self, _: Vec<bool>) {}
    /// A method that takes a boolean argument and a variadic boolean argument.
    /// Functional Utility: Tests passing a `boolean` argument followed by a variadic `boolean` argument.
    /// @param _ The initial boolean value.
    /// @param _ The `Vec<bool>` representing the variadic boolean arguments.
    fn PassVariadicBooleanAndDefault(&self, _: bool, _: Vec<bool>) {}
    /// A method that takes a variadic byte argument.
    /// Functional Utility: Tests passing a variadic `byte` IDL argument (sequence of signed 8-bit integers) to a Rust method.
    /// @param _ The `Vec<i8>` representing the variadic byte arguments.
    fn PassVariadicByte(&self, _: Vec<i8>) {}
    /// A method that takes a variadic octet argument.
    /// Functional Utility: Tests passing a variadic `octet` IDL argument (sequence of unsigned 8-bit integers) to a Rust method.
    /// @param _ The `Vec<u8>` representing the variadic octet arguments.
    fn PassVariadicOctet(&self, _: Vec<u8>) {}
    /// A method that takes a variadic short argument.
    /// Functional Utility: Tests passing a variadic `short` IDL argument (sequence of signed 16-bit integers) to a Rust method.
    /// @param _ The `Vec<i16>` representing the variadic short arguments.
    fn PassVariadicShort(&self, _: Vec<i16>) {}
    /// A method that takes a variadic unsigned short argument.
    /// Functional Utility: Tests passing a variadic `unsigned short` IDL argument (sequence of unsigned 16-bit integers) to a Rust method.
    /// @param _ The `Vec<u16>` representing the variadic unsigned short arguments.
    fn PassVariadicUnsignedShort(&self, _: Vec<u16>) {}
    /// A method that takes a variadic long argument.
    /// Functional Utility: Tests passing a variadic `long` IDL argument (sequence of signed 32-bit integers) to a Rust method.
    /// @param _ The `Vec<i32>` representing the variadic long arguments.
    fn PassVariadicLong(&self, _: Vec<i32>) {}
    /// A method that takes a variadic unsigned long argument.
    /// Functional Utility: Tests passing a variadic `unsigned long` IDL argument (sequence of unsigned 32-bit integers) to a Rust method.
    /// @param _ The `Vec<u32>` representing the variadic unsigned long arguments.
    fn PassVariadicUnsignedLong(&self, _: Vec<u32>) {}
    /// A method that takes a variadic long long argument.
    /// Functional Utility: Tests passing a variadic `long long` IDL argument (sequence of signed 64-bit integers) to a Rust method.
    /// @param _ The `Vec<i64>` representing the variadic long long arguments.
    fn PassVariadicLongLong(&self, _: Vec<i64>) {}
    /// A method that takes a variadic unsigned long long argument.
    /// Functional Utility: Tests passing a variadic `unsigned long long` IDL argument (sequence of unsigned 64-bit integers) to a Rust method.
    /// @param _ The `Vec<u64>` representing the variadic unsigned long long arguments.
    fn PassVariadicUnsignedLongLong(&self, _: Vec<u64>) {}
    /// A method that takes a variadic unrestricted float argument.
    /// Functional Utility: Tests passing a variadic `unrestricted float` IDL argument (sequence of 32-bit floating-point) to a Rust method.
    /// @param _ The `Vec<f32>` representing the variadic unrestricted float arguments.
    fn PassVariadicUnrestrictedFloat(&self, _: Vec<f32>) {}
    /// A method that takes a variadic float argument.
    /// Functional Utility: Tests passing a variadic `float` IDL argument (sequence of finite 32-bit floating-point) to a Rust method.
    /// @param _ The `Vec<Finite<f32>>` representing the variadic float arguments.
    fn PassVariadicFloat(&self, _: Vec<Finite<f32>>) {}
    /// A method that takes a variadic unrestricted double argument.
    /// Functional Utility: Tests passing a variadic `unrestricted double` IDL argument (sequence of 64-bit floating-point) to a Rust method.
    /// @param _ The `Vec<f64>` representing the variadic unrestricted double arguments.
    fn PassVariadicUnrestrictedDouble(&self, _: Vec<f64>) {}
    /// A method that takes a variadic double argument.
    /// Functional Utility: Tests passing a variadic `double` IDL argument (sequence of finite 64-bit floating-point) to a Rust method.
    /// @param _ The `Vec<Finite<f64>>` representing the variadic double arguments.
    fn PassVariadicDouble(&self, _: Vec<Finite<f64>>) {}
    /// A method that takes a variadic DOMString argument.
    /// Functional Utility: Tests passing a variadic `DOMString` IDL argument to a Rust method.
    /// @param _ The `Vec<DOMString>` representing the variadic DOMString arguments.
    fn PassVariadicString(&self, _: Vec<DOMString>) {}
    /// A method that takes a variadic USVString argument.
    /// Functional Utility: Tests passing a variadic `USVString` IDL argument to a Rust method.
    /// @param _ The `Vec<USVString>` representing the variadic USVString arguments.
    fn PassVariadicUsvstring(&self, _: Vec<USVString>) {}
    /// A method that takes a variadic ByteString argument.
    /// Functional Utility: Tests passing a variadic `ByteString` IDL argument to a Rust method.
    /// @param _ The `Vec<ByteString>` representing the variadic ByteString arguments.
    fn PassVariadicByteString(&self, _: Vec<ByteString>) {}
    /// A method that takes a variadic enum argument.
    /// Functional Utility: Tests passing a variadic `enum` IDL argument to a Rust method.
    /// @param _ The `Vec<TestEnum>` representing the variadic enum arguments.
    fn PassVariadicEnum(&self, _: Vec<TestEnum>) {}
    /// A method that takes a variadic interface argument (`Blob`).
    /// Functional Utility: Tests passing a variadic `Blob` interface IDL argument to a Rust method.
    /// @param _ The `&[&Blob]` representing the variadic Blob objects.
    fn PassVariadicInterface(&self, _: &[&Blob]) {}
    /// A method that takes a variadic union argument (`HTMLElementOrLong`).
    /// Functional Utility: Tests passing a variadic union type IDL argument to a Rust method.
    /// @param _ The `Vec<HTMLElementOrLong>` representing the variadic union arguments.
    fn PassVariadicUnion(&self, _: Vec<HTMLElementOrLong>) {}
    /// A method that takes a variadic union argument (`EventOrString`).
    /// Functional Utility: Tests passing a variadic union type IDL argument to a Rust method.
    /// @param _ The `Vec<EventOrString>` representing the variadic union arguments.
    fn PassVariadicUnion2(&self, _: Vec<EventOrString>) {}
    /// A method that takes a variadic union argument (`BlobOrString`).
    /// Functional Utility: Tests passing a variadic union type IDL argument to a Rust method.
    /// @param _ The `Vec<BlobOrString>` representing the variadic union arguments.
    fn PassVariadicUnion3(&self, _: Vec<BlobOrString>) {}
    /// A method that takes a variadic union argument (`BlobOrBoolean`).
    /// Functional Utility: Tests passing a variadic union type IDL argument to a Rust method.
    /// @param _ The `Vec<BlobOrBoolean>` representing the variadic union arguments.
    fn PassVariadicUnion4(&self, _: Vec<BlobOrBoolean>) {}
    /// A method that takes a variadic union argument (`StringOrUnsignedLong`).
    /// Functional Utility: Tests passing a variadic union type IDL argument to a Rust method.
    /// @param _ The `Vec<StringOrUnsignedLong>` representing the variadic union arguments.
    fn PassVariadicUnion5(&self, _: Vec<StringOrUnsignedLong>) {}
    /// A method that takes a variadic union argument (`UnsignedLongOrBoolean`).
    /// Functional Utility: Tests passing a variadic union type IDL argument to a Rust method.
    /// @param _ The `Vec<UnsignedLongOrBoolean>` representing the variadic union arguments.
    fn PassVariadicUnion6(&self, _: Vec<UnsignedLongOrBoolean>) {}
    /// A method that takes a variadic union argument (`ByteStringOrLong`).
    /// Functional Utility: Tests passing a variadic union type IDL argument to a Rust method.
    /// @param _ The `Vec<ByteStringOrLong>` representing the variadic union arguments.
    fn PassVariadicUnion7(&self, _: Vec<ByteStringOrLong>) {}
    /// A method that takes a variadic `any` argument.
    /// Functional Utility: Tests passing a variadic `any` IDL argument (sequence of any JavaScript values) to a Rust method.
    /// @param _cx The JavaScript context.
    /// @param _ The `Vec<HandleValue>` representing the variadic `any` arguments.
    fn PassVariadicAny(&self, _: SafeJSContext, _: Vec<HandleValue>) {}
    /// A method that takes a variadic `object` argument.
    /// Functional Utility: Tests passing a variadic `object` IDL argument (sequence of raw JavaScript objects) to a Rust method.
    /// @param _cx The JavaScript context.
    /// @param _ The `Vec<*mut JSObject>` representing the variadic object arguments.
    fn PassVariadicObject(&self, _: SafeJSContext, _: Vec<*mut JSObject>) {}
    /// A method that retrieves a boolean Mozilla preference.
    /// Functional Utility: Tests reading a boolean preference value via the bindings.
    /// @param pref_name The name of the preference to retrieve.
    /// @return The boolean value of the preference, or `false` if not found or invalid.
    fn BooleanMozPreference(&self, pref_name: DOMString) -> bool {
        prefs::get()
            .get_value(pref_name.as_ref())
            .try_into()
            .unwrap_or(false)
    }
    /// A method that retrieves a string Mozilla preference.
    /// Functional Utility: Tests reading a string preference value via the bindings.
    /// @param pref_name The name of the preference to retrieve.
    /// @return The `DOMString` value of the preference, or an empty string if not found or invalid.
    fn StringMozPreference(&self, pref_name: DOMString) -> DOMString {
        DOMString::from_string(
            prefs::get()
                .get_value(pref_name.as_ref())
                .try_into()
                .unwrap_or_default(),
        )
    }
    /// A getter for a boolean attribute controlled by a disabled preference.
    /// Functional Utility: Tests the behavior of attributes that are enabled/disabled
    /// based on a Mozilla preference that is currently disabled.
    /// @return `false`, indicating the attribute is disabled.
    fn PrefControlledAttributeDisabled(&self) -> bool {
        false
    }
    /// A getter for a boolean attribute controlled by an enabled preference.
    /// Functional Utility: Tests the behavior of attributes that are enabled/disabled
    /// based on a Mozilla preference that is currently enabled.
    /// @return `false`, indicating the attribute is enabled. (Note: The return value is hardcoded for testing purposes.)
    fn PrefControlledAttributeEnabled(&self) -> bool {
        false
    }
    /// A method controlled by a disabled preference.
    /// Functional Utility: Tests the behavior of methods that are enabled/disabled
    /// based on a Mozilla preference that is currently disabled.
    fn PrefControlledMethodDisabled(&self) {}
    /// A method controlled by an enabled preference.
    /// Functional Utility: Tests the behavior of methods that are enabled/disabled
    /// based on a Mozilla preference that is currently enabled.
    fn PrefControlledMethodEnabled(&self) {}
    /// A getter for a boolean attribute controlled by a disabled function.
    /// Functional Utility: Tests the behavior of attributes that are enabled/disabled
    /// based on a specific function that is currently disabled.
    /// @return `false`, indicating the attribute is disabled.
    fn FuncControlledAttributeDisabled(&self) -> bool {
        false
    }
    /// A getter for a boolean attribute controlled by an enabled function.
    /// Functional Utility: Tests the behavior of attributes that are enabled/disabled
    /// based on a specific function that is currently enabled.
    /// @return `false`, indicating the attribute is enabled. (Note: The return value is hardcoded for testing purposes.)
    fn FuncControlledAttributeEnabled(&self) -> bool {
        false
    }
    /// A method controlled by a disabled function.
    /// Functional Utility: Tests the behavior of methods that are enabled/disabled
    /// based on a specific function that is currently disabled.
    fn FuncControlledMethodDisabled(&self) {}
    /// A method controlled by an enabled function.
    /// Functional Utility: Tests the behavior of methods that are enabled/disabled
    /// based on a specific function that is currently enabled.
    fn FuncControlledMethodEnabled(&self) {}

    /// A method that takes a record argument (`Record<DOMString, long>`).
    /// Functional Utility: Tests passing an IDL `Record` type to a Rust method.
    /// @param _ The `Record<DOMString, i32>` value to pass.
    fn PassRecord(&self, _: Record<DOMString, i32>) {}
    /// A method that takes a record with USVString keys.
    /// Functional Utility: Tests passing an IDL `Record` where keys are `USVString`s.
    /// @param _ The `Record<USVString, i32>` value to pass.
    fn PassRecordWithUSVStringKey(&self, _: Record<USVString, i32>) {}
    /// A method that takes a record with ByteString keys.
    /// Functional Utility: Tests passing an IDL `Record` where keys are `ByteString`s.
    /// @param _ The `Record<ByteString, i32>` value to pass.
    fn PassRecordWithByteStringKey(&self, _: Record<ByteString, i32>) {}
    /// A method that takes a nullable record argument (`Record<DOMString, long>?`).
    /// Functional Utility: Tests passing a nullable IDL `Record` to a Rust method.
    /// @param _ The `Option<Record<DOMString, i32>>` value to pass.
    fn PassNullableRecord(&self, _: Option<Record<DOMString, i32>>) {}
    /// A method that takes a record with nullable integer values (`Record<DOMString, long?>`).
    /// Functional Utility: Tests passing an IDL `Record` where its member values are nullable.
    /// @param _ The `Record<DOMString, Option<i32>>` value to pass.
    fn PassRecordOfNullableInts(&self, _: Record<DOMString, Option<i32>>) {}
    /// A method that takes an optional record with nullable integer values.
    /// Functional Utility: Tests passing an optional IDL `Record` where its member values are nullable.
    /// @param _ The `Option<Record<DOMString, Option<i32>>>` value to pass.
    fn PassOptionalRecordOfNullableInts(&self, _: Option<Record<DOMString, Option<i32>>>) {}
    /// A method that takes an optional nullable record of nullable integer values.
    /// Functional Utility: Tests passing an optional `Record?` where its member values are also nullable.
    /// @param _ The `Option<Option<Record<DOMString, Option<i32>>>>` value to pass.
    fn PassOptionalNullableRecordOfNullableInts(
        &self,
        _: Option<Option<Record<DOMString, Option<i32>>>>,
    ) {
    }
    /// A method that takes a record with castable object values.
    /// Functional Utility: Tests passing an IDL `Record` where its member values are
    /// references to other DOM objects that are castable (e.g., `TestBinding`).
    /// @param _ The `Record<DOMString, DomRoot<TestBinding>>` value to pass.
    fn PassCastableObjectRecord(&self, _: Record<DOMString, DomRoot<TestBinding>>) {}
    /// A method that takes a record with nullable castable object values.
    /// Functional Utility: Tests passing an IDL `Record` where its member values are
    /// nullable references to other DOM objects.
    /// @param _ The `Record<DOMString, Option<DomRoot<TestBinding>>>` value to pass.
    fn PassNullableCastableObjectRecord(&self, _: Record<DOMString, Option<DomRoot<TestBinding>>>) {
    }
    /// A method that takes an optional record with castable object values.
    /// Functional Utility: Tests passing an optional IDL `Record?` where its member values are
    /// references to other DOM objects.
    /// @param _ The `Option<Record<DOMString, DomRoot<TestBinding>>>` value to pass.
    fn PassCastableObjectNullableRecord(&self, _: Option<Record<DOMString, DomRoot<TestBinding>>>) {
    }
    /// A method that takes an optional record with optional castable object values.
    /// Functional Utility: Tests passing an optional IDL `Record?` where its member values are
    /// also nullable references to other DOM objects.
    /// @param _ The `Option<Record<DOMString, Option<DomRoot<TestBinding>>>>` value to pass.
    fn PassNullableCastableObjectNullableRecord(
        &self,
        _: Option<Record<DOMString, Option<DomRoot<TestBinding>>>>,
    ) {
    }
    /// A method that takes an optional record argument (`Record<DOMString, long>?`).
    /// Functional Utility: Tests passing an optional IDL `Record` to a Rust method.
    /// @param _ The `Option<Record<DOMString, i32>>` value to pass.
    fn PassOptionalRecord(&self, _: Option<Record<DOMString, i32>>) {}
    /// A method that takes an optional nullable record argument (`Record<DOMString, long>?`).
    /// Functional Utility: Tests passing an optional nullable IDL `Record` to a Rust method.
    /// @param _ The `Option<Option<Record<DOMString, i32>>>` value to pass.
    fn PassOptionalNullableRecord(&self, _: Option<Option<Record<DOMString, i32>>>) {}
    /// A method that takes an optional nullable record argument (`Record<DOMString, long>?`) with a default value.
    /// Functional Utility: Tests passing an optional nullable IDL `Record` with a default value to a Rust method.
    /// @param _ The `Option<Record<DOMString, i32>>` value to pass.
    fn PassOptionalNullableRecordWithDefaultValue(&self, _: Option<Record<DOMString, i32>>) {}
    /// A method that takes an optional record with object values.
    /// Functional Utility: Tests passing an optional IDL `Record` where its member values are
    /// references to other DOM objects (`TestBinding`).
    /// @param _ The `Option<Record<DOMString, DomRoot<TestBinding>>>` value to pass.
    fn PassOptionalObjectRecord(&self, _: Option<Record<DOMString, DomRoot<TestBinding>>>) {}
    /// A method that takes a record with string keys and string values (`Record<DOMString, DOMString>`).
    /// Functional Utility: Tests passing an IDL `Record` with `DOMString` keys and `DOMString` values.
    /// @param _ The `Record<DOMString, DOMString>` value to pass.
    fn PassStringRecord(&self, _: Record<DOMString, DOMString>) {}
    /// A method that takes a record with string keys and bytestring values (`Record<DOMString, ByteString>`).
    /// Functional Utility: Tests passing an IDL `Record` with `DOMString` keys and `ByteString` values.
    /// @param _ The `Record<DOMString, ByteString>` value to pass.
    fn PassByteStringRecord(&self, _: Record<DOMString, ByteString>) {}
    /// A method that takes a record of records (`Record<DOMString, Record<DOMString, long>>`).
    /// Functional Utility: Tests passing an IDL `Record` where its values are themselves `Record`s.
    /// @param _ The `Record<DOMString, Record<DOMString, i32>>` value to pass.
    fn PassRecordOfRecords(&self, _: Record<DOMString, Record<DOMString, i32>>) {}
    /// A method that takes a union argument containing a record (`LongOrStringByteStringRecord`).
    /// Functional Utility: Tests passing a complex union type that includes a `Record`.
    /// @param _ The `UnionTypes::LongOrStringByteStringRecord` value to pass.
    fn PassRecordUnion(&self, _: UnionTypes::LongOrStringByteStringRecord) {}
    /// A method that takes a union argument containing a record (`TestBindingOrStringByteStringRecord`).
    /// Functional Utility: Tests passing a complex union type that includes a `Record`.
    /// @param _ The `UnionTypes::TestBindingOrStringByteStringRecord` value to pass.
    fn PassRecordUnion2(&self, _: UnionTypes::TestBindingOrStringByteStringRecord) {}
    /// A method that takes a union argument containing a record (`TestBindingOrByteStringSequenceSequenceOrStringByteStringRecord`).
    /// Functional Utility: Tests passing a very complex union type that includes interfaces, sequences, and records.
    /// @param _ The `UnionTypes::TestBindingOrByteStringSequenceSequenceOrStringByteStringRecord` value to pass.
    fn PassRecordUnion3(
        &self,
        _: UnionTypes::TestBindingOrByteStringSequenceSequenceOrStringByteStringRecord,
    ) {
    }
    /// A method that receives no arguments and returns a record (`Record<DOMString, long>`).
    /// Functional Utility: Tests the basic binding of a method that returns an IDL `Record`.
    /// @return A new empty `Record<DOMString, i32>`.
    fn ReceiveRecord(&self) -> Record<DOMString, i32> {
        Record::new()
    }
    /// A method that receives no arguments and returns a record with USVString keys.
    /// Functional Utility: Tests the basic binding of a method that returns an IDL `Record`
    /// where its keys are `USVString`s.
    /// @return A new empty `Record<USVString, i32>`.
    fn ReceiveRecordWithUSVStringKey(&self) -> Record<USVString, i32> {
        Record::new()
    }
    /// A method that receives no arguments and returns a record with ByteString keys.
    /// Functional Utility: Tests the basic binding of a method that returns an IDL `Record`
    /// where its keys are `ByteString`s.
    /// @return A new empty `Record<ByteString, i32>`.
    fn ReceiveRecordWithByteStringKey(&self) -> Record<ByteString, i32> {
        Record::new()
    }
    /// A method that receives no arguments and returns a nullable record (`Record<DOMString, long>?`).
    /// Functional Utility: Tests the basic binding of a method that returns a nullable IDL `Record`.
    /// @return An `Option<Record<DOMString, i32>>` representing the nullable Record.
    fn ReceiveNullableRecord(&self) -> Option<Record<DOMString, i32>> {
        Some(Record::new())
    }
    /// A method that receives no arguments and returns a record of nullable integer values (`Record<DOMString, long?>`).
    /// Functional Utility: Tests the basic binding of a method that returns an IDL `Record`
    /// where its member values are nullable.
    /// @return A new empty `Record<DOMString, Option<i32>>`.
    fn ReceiveRecordOfNullableInts(&self) -> Record<DOMString, Option<i32>> {
        Record::new()
    }
    /// A method that receives no arguments and returns a nullable record of nullable integer values (`Record<DOMString, long?>?`).
    /// Functional Utility: Tests the basic binding of a method that returns a nullable IDL `Record`
    /// where its member values are also nullable.
    /// @return An `Option<Record<DOMString, Option<i32>>>` representing the nullable Record.
    fn ReceiveNullableRecordOfNullableInts(&self) -> Option<Record<DOMString, Option<i32>>> {
        Some(Record::new())
    }
    /// A method that receives no arguments and returns a record of records (`Record<DOMString, Record<DOMString, long>>`).
    /// Functional Utility: Tests the basic binding of a method that returns a nested IDL `Record`.
    /// @return A new empty `Record<DOMString, Record<DOMString, i32>>`.
    fn ReceiveRecordOfRecords(&self) -> Record<DOMString, Record<DOMString, i32>> {
        Record::new()
    }
    /// A method that receives no arguments and returns a record with `any` values (`Record<DOMString, any>`).
    /// Functional Utility: Tests the basic binding of a method that returns an IDL `Record` with `any` values.
    /// @return A new empty `Record<DOMString, JSVal>`.
    fn ReceiveAnyRecord(&self) -> Record<DOMString, JSVal> {
        Record::new()
    }

    /// A method that returns an immediately resolved `Promise`.
    /// Functional Utility: Tests the binding of methods that return an already-resolved `Promise`.
    /// @param cx The JavaScript context.
    /// @param v The `HandleValue` to resolve the Promise with.
    /// @return An `Rc<Promise>` that is already resolved.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    fn ReturnResolvedPromise(&self, cx: SafeJSContext, v: HandleValue) -> Rc<Promise> {
        Promise::new_resolved(&self.global(), cx, v, CanGc::note())
    }

    /// A method that returns an immediately rejected `Promise`.
    /// Functional Utility: Tests the binding of methods that return an already-rejected `Promise`.
    /// @param cx The JavaScript context.
    /// @param v The `HandleValue` to reject the Promise with.
    /// @return An `Rc<Promise>` that is already rejected.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    fn ReturnRejectedPromise(&self, cx: SafeJSContext, v: HandleValue) -> Rc<Promise> {
        Promise::new_rejected(&self.global(), cx, v, CanGc::note())
    }

    /// A method that resolves a given `Promise` with a JavaScript value.
    /// Functional Utility: Tests the ability to resolve an existing `Promise` from Rust code.
    /// @param cx The JavaScript context.
    /// @param p The `Promise` object to resolve.
    /// @param v The `HandleValue` to resolve the Promise with.
    /// @param can_gc A `CanGc` token.
    fn PromiseResolveNative(&self, cx: SafeJSContext, p: &Promise, v: HandleValue, can_gc: CanGc) {
        p.resolve(cx, v, can_gc);
    }

    /// A method that rejects a given `Promise` with a JavaScript value.
    /// Functional Utility: Tests the ability to reject an existing `Promise` from Rust code.
    /// @param cx The JavaScript context.
    /// @param p The `Promise` object to reject.
    /// @param v The `HandleValue` to reject the Promise with.
    fn PromiseRejectNative(&self, cx: SafeJSContext, p: &Promise, v: HandleValue) {
        p.reject(cx, v);
    }

    /// A method that rejects a given `Promise` with a `TypeError`.
    /// Functional Utility: Tests the ability to reject an existing `Promise` from Rust
    /// with a specific `TypeError` message.
    /// @param p The `Promise` object to reject.
    /// @param s The `USVString` containing the error message.
    fn PromiseRejectWithTypeError(&self, p: &Promise, s: USVString) {
        p.reject_error(Error::Type(s.0));
    }

    /// A method that resolves a `Promise` after a specified delay.
    /// Functional Utility: Tests the ability to schedule a delayed resolution of a `Promise`
    /// from Rust code, demonstrating asynchronous operations.
    /// @param p The `Promise` object to resolve.
    /// @param value The `DOMString` value to resolve the Promise with.
    /// @param delay The delay in milliseconds before resolving the Promise.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    fn ResolvePromiseDelayed(&self, p: &Promise, value: DOMString, delay: u64) {
        let promise = p.duplicate();
        let cb = TestBindingCallback {
            promise: TrustedPromise::new(promise),
            value,
        };
        let _ = self.global().schedule_callback(
            OneshotTimerCallback::TestBindingCallback(cb),
            Duration::from_millis(delay),
        );
    }

    /// A method that tests native promise handlers.
    /// Functional Utility: Tests the integration of Rust-defined callbacks as
    /// JavaScript Promise `.then()` and `.catch()` handlers.
    /// @param resolve An optional `SimpleCallback` for the resolve handler.
    /// @param reject An optional `SimpleCallback` for the reject handler.
    /// @param comp The `InRealm` context.
    /// @param can_gc A `CanGc` token.
    /// @return A new `Rc<Promise>` with the native handlers appended.
    fn PromiseNativeHandler(
        &self,
        resolve: Option<Rc<SimpleCallback>>,
        reject: Option<Rc<SimpleCallback>>,
        comp: InRealm,
        can_gc: CanGc,
    ) -> Rc<Promise> {
        let global = self.global();
        let handler = PromiseNativeHandler::new(
            &global,
            resolve.map(SimpleHandler::new_boxed),
            reject.map(SimpleHandler::new_boxed),
            can_gc,
        );
        let p = Promise::new_in_current_realm(comp, can_gc);
        p.append_native_handler(&handler, comp, can_gc);
        return p;

        /// Private helper struct that acts as a simple wrapper for a `SimpleCallback`.
        #[derive(JSTraceable, MallocSizeOf)]
        struct SimpleHandler {
            #[ignore_malloc_size_of = "Rc has unclear ownership semantics"]
            handler: Rc<SimpleCallback>,
        }
        impl SimpleHandler {
            /// Creates a new boxed `SimpleHandler` from an `Rc<SimpleCallback>`.
            fn new_boxed(callback: Rc<SimpleCallback>) -> Box<dyn Callback> {
                Box::new(SimpleHandler { handler: callback })
            }
        }
        impl Callback for SimpleHandler {
            /// Implements the callback logic for `SimpleHandler`.
            /// Functional Utility: Invokes the wrapped `SimpleCallback` with the
            /// resolved or rejected value from the JavaScript Promise.
            fn callback(&self, cx: SafeJSContext, v: HandleValue, realm: InRealm, _can_gc: CanGc) {
                let global = GlobalScope::from_safe_context(cx, realm);
                let _ = self.handler.Call_(&*global, v, ExceptionHandling::Report);
            }
        }
    }

    /// A method that returns a new `Promise` object.
    /// Functional Utility: Tests the binding of methods that return a `Promise`.
    /// @param comp The `InRealm` context.
    /// @param can_gc A `CanGc` token.
    /// @return A new `Rc<Promise>` instance.
    fn PromiseAttribute(&self, comp: InRealm, can_gc: CanGc) -> Rc<Promise> {
        Promise::new_in_current_realm(comp, can_gc)
    }

    /// A method that accepts a `Promise` as an argument.
    /// Functional Utility: Tests the basic binding of methods that take a `Promise` as a parameter.
    /// @param _promise The `Promise` object to accept.
    fn AcceptPromise(&self, _promise: &Promise) {}

    /// A method that takes a sequence of sequences of longs argument (`sequence<sequence<long>>`).
    /// Functional Utility: Tests passing a nested sequence IDL argument to a Rust method.
    /// @param _seq The `Vec<Vec<i32>>` representing the sequence of sequences.
    fn PassSequenceSequence(&self, _seq: Vec<Vec<i32>>) {}
    /// A method that receives no arguments and returns a sequence of sequences of longs.
    /// Functional Utility: Tests the basic binding of a method that returns a `sequence<sequence<long>>`.
    /// @return An empty `Vec<Vec<i32>>`.
    fn ReturnSequenceSequence(&self) -> Vec<Vec<i32>> {
        vec![]
    }
    /// A method that takes a union argument (`LongOrLongSequenceSequence`).
    /// Functional Utility: Tests passing a complex union type that includes a
    /// `long` or a `sequence<sequence<long>>`.
    /// @param seq The `LongOrLongSequenceSequence` value to pass.
    fn PassUnionSequenceSequence(&self, seq: LongOrLongSequenceSequence) {
        match seq {
            LongOrLongSequenceSequence::Long(_) => (),
            LongOrLongSequenceSequence::LongSequenceSequence(seq) => {
                let _seq: Vec<Vec<i32>> = seq;
            },
        }
    }

    /// A method that intentionally crashes the process.
    /// Functional Utility: Tests how the binding generator handles methods that can lead
    /// to a hard crash (e.g., dereferencing a null pointer), verifying error handling
    /// or testing specific crash reporting mechanisms.
    #[allow(unsafe_code)]
    fn CrashHard(&self) {
        unsafe { std::ptr::null_mut::<i32>().write(42) }
    }

    /// A method that advances the animation clock.
    /// Functional Utility: Tests the ability to control time-based events within the DOM
    /// (e.g., animations, timers) for testing purposes.
    /// @param ms The number of milliseconds to advance the clock.
    fn AdvanceClock(&self, ms: i32) {
        self.global().as_window().advance_animation_clock(ms);
    }

    /// A method that intentionally triggers a Rust panic.
    /// Functional Utility: Tests how the binding generator handles methods that can lead
    /// to a Rust panic, verifying error handling or crash reporting mechanisms for panics.
    fn Panic(&self) {
        panic!("explicit panic from script")
    }

    /// A method that returns the entry `GlobalScope`.
    /// Functional Utility: Tests the binding of a method that returns the initial
    /// `GlobalScope` (often the main window's global object).
    /// @return The `DomRoot<GlobalScope>` representing the entry global scope.
    fn EntryGlobal(&self) -> DomRoot<GlobalScope> {
        GlobalScope::entry()
    }
    /// A method that returns the incumbent `GlobalScope`.
    /// Functional Utility: Tests the binding of a method that returns the currently
    /// active `GlobalScope` (the one whose script is currently executing).
    /// @return The `DomRoot<GlobalScope>` representing the incumbent global scope.
    fn IncumbentGlobal(&self) -> DomRoot<GlobalScope> {
        GlobalScope::incumbent().unwrap()
    }

    /// A method that returns a boolean from a semi-exposed interface.
    /// Functional Utility: Tests the binding of methods that are "semi-exposed" (i.e.,
    /// conditionally exposed based on specific build configurations or flags) and
    /// return a boolean value.
    /// @return `true` for testing purposes.
    fn SemiExposedBoolFromInterface(&self) -> bool {
        true
    }

    /// A method that returns a boolean from a semi-exposed partial interface.
    /// Functional Utility: Tests the binding of methods that are "semi-exposed"
    /// within a partial interface and return a boolean value.
    /// @return `true` for testing purposes.
    fn BoolFromSemiExposedPartialInterface(&self) -> bool {
        true
    }

    /// A method that returns a boolean from a semi-exposed partial interface.
    /// Functional Utility: Tests the binding of methods that are "semi-exposed"
    /// within a partial interface and return a boolean value. This is functionally
    /// similar to `BoolFromSemiExposedPartialInterface` but might represent a
    /// different test case or IDL definition.
    /// @return `true` for testing purposes.
    fn SemiExposedBoolFromPartialInterface(&self) -> bool {
        true
    }

    /// A method that returns a `TestDictionaryWithParent` instance.
    /// Functional Utility: Tests the binding of dictionaries with inheritance,
    /// ensuring that members from both the parent and child dictionaries are correctly handled.
    /// @param s1 A `DOMString` for the `parentStringMember` of the parent dictionary.
    /// @param s2 A `DOMString` for the `stringMember` of the child dictionary.
    /// @return A `TestDictionaryWithParent` instance populated with the input strings.
    fn GetDictionaryWithParent(&self, s1: DOMString, s2: DOMString) -> TestDictionaryWithParent {
        TestDictionaryWithParent {
            parent: TestDictionaryParent {
                parentStringMember: Some(s1),
            },
            stringMember: Some(s2),
        }
    }

    /// A method that immediately throws an error, intended to reject a promise.
    /// Functional Utility: Tests the binding generator's ability to translate Rust `Err` results
    /// into a JavaScript Promise rejection.
    /// @return A `Fallible<Rc<Promise>>` that will always return an `Err`.
    fn MethodThrowToRejectPromise(&self) -> Fallible<Rc<Promise>> {
        Err(Error::Type("test".to_string()))
    }

    /// A getter method that immediately throws an error, intended to reject a promise.
    /// Functional Utility: Tests the binding generator's ability to translate `Err` results
    /// from attribute getters into a JavaScript Promise rejection.
    /// @return A `Fallible<Rc<Promise>>` that will always return an `Err`.
    fn GetGetterThrowToRejectPromise(&self) -> Fallible<Rc<Promise>> {
        Err(Error::Type("test".to_string()))
    }

    /// A method that is expected to throw internally and reject a promise.
    /// Functional Utility: Tests scenarios where a method's implementation
    /// is expected to `panic!` or otherwise throw an error before reaching
    /// the JavaScript promise resolution logic.
    /// @param _arg An unused `u64` argument.
    /// @return A `Rc<Promise>` (though this line is unreachable due to the internal throw).
    fn MethodInternalThrowToRejectPromise(&self, _arg: u64) -> Rc<Promise> {
        unreachable!("Method should already throw")
    }

    /// A static method that immediately throws an error, intended to reject a promise.
    /// Functional Utility: Tests the binding generator's ability to translate Rust `Err` results
    /// from a static method into a JavaScript Promise rejection.
    /// @param _ The `GlobalScope` context (unused).
    /// @return A `Fallible<Rc<Promise>>` that will always return an `Err`.
    fn StaticThrowToRejectPromise(_: &GlobalScope) -> Fallible<Rc<Promise>> {
        Err(Error::Type("test".to_string()))
    }

    /// A static method that is expected to throw internally and reject a promise.
    /// Functional Utility: Tests scenarios where a static method's implementation
    /// is expected to `panic!` or otherwise throw an error before reaching
    /// the JavaScript promise resolution logic.
    /// @param _ The `GlobalScope` context (unused).
    /// @param _arg An unused `u64` argument.
    /// @return A `Rc<Promise>` (though this line is unreachable due to the internal throw).
    fn StaticInternalThrowToRejectPromise(_: &GlobalScope, _arg: u64) -> Rc<Promise> {
        unreachable!("Method should already throw")
    }

    /// Static getter for a boolean attribute.
    /// Functional Utility: Tests the binding of a static `boolean` IDL attribute.
    /// @param _ The `GlobalScope` context (unused).
    /// @return The boolean value of the attribute.
    fn BooleanAttributeStatic(_: &GlobalScope) -> bool {
        false
    }
    /// Static setter for a boolean attribute.
    /// Functional Utility: Tests setting a static `boolean` IDL attribute.
    /// @param _ The `GlobalScope` context (unused).
    /// @param _ The boolean value to set.
    fn SetBooleanAttributeStatic(_: &GlobalScope, _: bool) {}
    /// Static method that receives no arguments and returns nothing.
    /// Functional Utility: Tests the basic binding of a static void method with no parameters.
    /// @param _ The `GlobalScope` context (unused).
    fn ReceiveVoidStatic(_: &GlobalScope) {}
    /// Static getter for a boolean attribute controlled by a disabled preference.
    /// Functional Utility: Tests the behavior of static attributes that are enabled/disabled
    /// based on a Mozilla preference that is currently disabled.
    /// @param _ The `GlobalScope` context (unused).
    /// @return `false`, indicating the attribute is disabled.
    fn PrefControlledStaticAttributeDisabled(_: &GlobalScope) -> bool {
        false
    }
    /// Static getter for a boolean attribute controlled by an enabled preference.
    /// Functional Utility: Tests the behavior of static attributes that are enabled/disabled
    /// based on a Mozilla preference that is currently enabled.
    /// @param _ The `GlobalScope` context (unused).
    /// @return `false`, indicating the attribute is enabled. (Note: The return value is hardcoded for testing purposes.)
    fn PrefControlledStaticAttributeEnabled(_: &GlobalScope) -> bool {
        false
    }
    /// Static method controlled by a disabled preference.
    /// Functional Utility: Tests the behavior of static methods that are enabled/disabled
    /// based on a Mozilla preference that is currently disabled.
    /// @param _ The `GlobalScope` context (unused).
    fn PrefControlledStaticMethodDisabled(_: &GlobalScope) {}
    /// Static method controlled by an enabled preference.
    /// Functional Utility: Tests the behavior of static methods that are enabled/disabled
    /// based on a Mozilla preference that is currently enabled.
    /// @param _ The `GlobalScope` context (unused).
    fn PrefControlledStaticMethodEnabled(_: &GlobalScope) {}
    /// Static getter for a boolean attribute controlled by a disabled function.
    /// Functional Utility: Tests the behavior of static attributes that are enabled/disabled
    /// based on a specific function that is currently disabled.
    /// @param _ The `GlobalScope` context (unused).
    /// @return `false`, indicating the attribute is disabled.
    fn FuncControlledStaticAttributeDisabled(_: &GlobalScope) -> bool {
        false
    }
    /// Static getter for a boolean attribute controlled by an enabled function.
    /// Functional Utility: Tests the behavior of static attributes that are enabled/disabled
    /// based on a specific function that is currently enabled.
    /// @param _ The `GlobalScope` context (unused).
    /// @return `false`, indicating the attribute is enabled. (Note: The return value is hardcoded for testing purposes.)
    fn FuncControlledStaticAttributeEnabled(_: &GlobalScope) -> bool {
        false
    }
    /// Static method controlled by a disabled function.
    /// Functional Utility: Tests the behavior of static methods that are enabled/disabled
    /// based on a specific function that is currently disabled.
    /// @param _ The `GlobalScope` context (unused).
    fn FuncControlledStaticMethodDisabled(_: &GlobalScope) {}
    /// Static method controlled by an enabled function.
    /// Functional Utility: Tests the behavior of static methods that are enabled/disabled
    /// based on a specific function that is currently enabled.
    /// @param _ The `GlobalScope` context (unused).
    fn FuncControlledStaticMethodEnabled(_: &GlobalScope) {}
}

impl TestBinding {
    /// A condition function that always returns `true`.
    /// Functional Utility: Used for testing conditional bindings where the condition
    /// is expected to be satisfied.
    /// @param _cx The JavaScript context.
    /// @param _ An unused `HandleObject`.
    /// @return Always `true`.
    pub(crate) fn condition_satisfied(_: SafeJSContext, _: HandleObject) -> bool {
        true
    }
    /// A condition function that always returns `false`.
    /// Functional Utility: Used for testing conditional bindings where the condition
    /// is expected to be unsatisfied.
    /// @param _cx The JavaScript context.
    /// @param _ An unused `HandleObject`.
    /// @return Always `false`.
    pub(crate) fn condition_unsatisfied(_: SafeJSContext, _: HandleObject) -> bool {
        false
    }
}
/// A callback structure for testing delayed promise resolution.
/// Functional Utility: Encapsulates the necessary data to resolve a `Promise`
/// after a delay, used in conjunction with `ResolvePromiseDelayed`.
#[derive(JSTraceable, MallocSizeOf)]
pub(crate) struct TestBindingCallback {
    /// The `TrustedPromise` to be resolved.
    #[ignore_malloc_size_of = "unclear ownership semantics"]
    promise: TrustedPromise,
    /// The `DOMString` value to resolve the promise with.
    value: DOMString,
}

impl TestBindingCallback {
    /// Invokes the callback, resolving the stored `Promise` with the stored value.
    /// Functional Utility: Executes the delayed promise resolution logic.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn invoke(self) {
        self.promise
            .root()
            .resolve_native(&self.value, CanGc::note());
    }
}
