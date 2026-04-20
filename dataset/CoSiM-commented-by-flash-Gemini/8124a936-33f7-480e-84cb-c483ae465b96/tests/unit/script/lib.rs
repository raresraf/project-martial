/**
 * @8124a936-33f7-480e-84cb-c483ae465b96/tests/unit/script/lib.rs
 * @brief Aggregate root for unit tests targeting the 'script' crate.
 * 
 * Functional Intent: Orchestrates the execution of granular unit tests for 
 * various DOM elements and internal script utility types. It includes 
 * architecture-specific memory layout checks (size_of) and static verification 
 * of trait bounds (e.g., ensuring TrustedPromise remains non-Clone).
 * 
 * Domain: Browser Engine, DOM Testing, Rust Static Verification.
 */

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

#[cfg(test)]
mod htmlareaelement;
#[cfg(test)]
mod htmlimageelement;
#[cfg(test)]
mod origin;
#[cfg(all(test, target_pointer_width = "64"))]
mod size_of;
#[cfg(test)]
mod textinput;
#[cfg(test)]
mod timeranges;

/**
 * Functional Utility: Verifies that `TrustedPromise` does not implement `Clone`.
 * Logic: Employs a documentation test with `compile_fail` to ensure that 
 * security-sensitive promise objects cannot be duplicated, preventing potential 
 * state inconsistencies or safety violations.
```compile_fail,E0277
extern crate script;

fn cloneable<T: Clone>() {}

fn main() {
    cloneable::<script::test::TrustedPromise>();
}
```
*/
pub fn trustedpromise_does_not_impl_clone() {}
