/**
 * @file test.rs
 * @brief Diagnostic and testing utilities for Servo's script component.
 * 
 * Architectural Intent: Provides a central interface for internal unit tests to access 
 * DOM bindings and verify structural properties of the engine.
 * 
 * Functional Utility: Primarily used for memory-profiling and compile-time validation. 
 * The 'size_of' module allows empirical measurement of core DOM object footprints.
 */

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

// For compile-fail tests only.
pub use crate::dom::bindings::refcounted::TrustedPromise;
pub use crate::dom::bindings::str::{ByteString, DOMString};

pub mod area {
    pub use crate::dom::htmlareaelement::{Area, Shape};
}

/**
 * @mod size_of
 * @brief Memory footprint diagnostics for core DOM types.
 */
#[allow(non_snake_case)]
pub mod size_of {
    use std::mem::size_of;

    use crate::dom::characterdata::CharacterData;
    use crate::dom::element::Element;
    use crate::dom::eventtarget::EventTarget;
    use crate::dom::htmldivelement::HTMLDivElement;
    use crate::dom::htmlelement::HTMLElement;
    use crate::dom::htmlspanelement::HTMLSpanElement;
    use crate::dom::node::Node;
    use crate::dom::text::Text;

    /**
     * Block Logic: Size resolution.
     * Logic: Returns the byte size of the type using Rust's static memory layout.
     */
    pub fn CharacterData() -> usize {
        size_of::<CharacterData>()
    }

    pub fn Element() -> usize {
        size_of::<Element>()
    }

    pub fn EventTarget() -> usize {
        size_of::<EventTarget>()
    }

    pub fn HTMLDivElement() -> usize {
        size_of::<HTMLDivElement>()
    }

    pub fn HTMLElement() -> usize {
        size_of::<HTMLElement>()
    }

    pub fn HTMLSpanElement() -> usize {
        size_of::<HTMLSpanElement>()
    }

    pub fn Node() -> usize {
        size_of::<Node>()
    }

    pub fn Text() -> usize {
        size_of::<Text>()
    }
}

pub mod srcset {
    pub use crate::dom::htmlimageelement::{Descriptor, ImageSource, parse_a_srcset_attribute};
}

pub mod timeranges {
    pub use crate::dom::timeranges::TimeRangesContainer;
}

pub mod textinput {
    pub use crate::clipboard_provider::ClipboardProvider;
    pub use crate::textinput::{
        Direction, Lines, Selection, SelectionDirection, TextInput, TextPoint, UTF8Bytes,
        UTF16CodeUnits,
    };
}
