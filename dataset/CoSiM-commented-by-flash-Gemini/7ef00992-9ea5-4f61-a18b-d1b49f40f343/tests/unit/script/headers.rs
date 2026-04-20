/**
 * @7ef00992-9ea5-4f61-a18b-d1b49f40f343/tests/unit/script/headers.rs
 * @brief Unit tests for HTTP header value normalization.
 * 
 * Functional Intent: Validates the stripping of leading and trailing HTTP whitespace 
 * (SP, HTAB, VT, FF, CR, LF) from ByteStrings. It ensures that internal 
 * whitespace is preserved while edge-case whitespace is correctly purged, 
 * complying with the Fetch specification for header value processing.
 * 
 * Domain: Networking, HTTP Protocol, Data Sanitization.
 */

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use script::test::{ByteString, normalize_value};

/**
 * @brief Logic: Verifies that an empty input results in an empty output.
 */
#[test]
fn test_normalize_empty_bytestring() {
    let empty_bytestring = ByteString::new(vec![]);
    let actual = normalize_value(empty_bytestring);
    let expected = ByteString::new(vec![]);
    assert_eq!(actual, expected);
}

/**
 * @brief Logic: Verifies that a string containing ONLY whitespace is fully collapsed.
 */
#[test]
fn test_normalize_all_whitespace_bytestring() {
    let all_whitespace_bytestring = ByteString::new(vec![b'\t', b'\n', b'\r', b' ']);
    let actual = normalize_value(all_whitespace_bytestring);
    let expected = ByteString::new(vec![]);
    assert_eq!(actual, expected);
}

/**
 * @brief Logic: Verifies that already-normalized strings remain unchanged.
 */
#[test]
fn test_normalize_non_empty_no_whitespace_bytestring() {
    let no_whitespace_bytestring = ByteString::new(vec![b'S', b'!']);
    let actual = normalize_value(no_whitespace_bytestring);
    let expected = ByteString::new(vec![b'S', b'!']);
    assert_eq!(actual, expected);
}

/**
 * @brief Logic: Verifies the purging of diverse leading whitespace characters.
 */
#[test]
fn test_normalize_non_empty_leading_whitespace_bytestring() {
    let leading_whitespace_bytestring =
        ByteString::new(vec![b'\t', b'\n', b' ', b'\r', b'S', b'!']);
    let actual = normalize_value(leading_whitespace_bytestring);
    let expected = ByteString::new(vec![b'S', b'!']);
    assert_eq!(actual, expected);
}

/**
 * @brief Logic: Verifies the purging of diverse trailing whitespace characters.
 */
#[test]
fn test_normalize_non_empty_no_leading_whitespace_trailing_whitespace_bytestring() {
    let trailing_whitespace_bytestring =
        ByteString::new(vec![b'S', b'!', b'\t', b'\n', b' ', b'\r']);
    let actual = normalize_value(trailing_whitespace_bytestring);
    let expected = ByteString::new(vec![b'S', b'!']);
    assert_eq!(actual, expected);
}

/**
 * @brief Logic: Verifies simultaneous leading and trailing normalization.
 */
#[test]
fn test_normalize_non_empty_leading_and_trailing_whitespace_bytestring() {
    let whitespace_sandwich_bytestring = ByteString::new(vec![
        b'\t', b'\n', b' ', b'\r', b'S', b'!', b'\t', b'\n', b' ', b'\r',
    ]);
    let actual = normalize_value(whitespace_sandwich_bytestring);
    let expected = ByteString::new(vec![b'S', b'!']);
    assert_eq!(actual, expected);
}

/**
 * @brief Invariant: Internal whitespace must remain intact during normalization.
 * Logic: Validates that while boundaries are purged, whitespace between 
 * printable characters is preserved.
 */
#[test]
fn test_normalize_non_empty_leading_trailing_and_internal_whitespace_bytestring() {
    let whitespace_bigmac_bytestring = ByteString::new(vec![
        b'\t', b'\n', b' ', b'\r', b'S', b'\t', b'\n', b' ', b'\r', b'!', b'\t', b'\n', b' ', b'\r',
    ]);
    let actual = normalize_value(whitespace_bigmac_bytestring);
    let expected = ByteString::new(vec![b'S', b'\t', b'\n', b' ', b'\r', b'!']);
    assert_eq!(actual, expected);
}
