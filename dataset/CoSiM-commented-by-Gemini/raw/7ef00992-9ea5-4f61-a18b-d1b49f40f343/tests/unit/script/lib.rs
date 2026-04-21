/**
 * @raw/7ef00992-9ea5-4f61-a18b-d1b49f40f343/tests/unit/script/lib.rs
 * @brief Core functionality implementation.
 * Intent: Execute functional units and state management.
 * Algorithm: Iterative or sequential execution logic.
 * Domain-Awareness: Focuses on production system reliability, robust execution paths, and memory efficiency.
 */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

#[cfg(test)]
mod headers;
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
```compile_fail,E0277
extern crate script;

fn cloneable<T: Clone>() {}

fn main() {
    cloneable::<script::test::TrustedPromise>();
}
```
*/
pub fn trustedpromise_does_not_impl_clone() {}
