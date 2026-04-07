/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
/**
 * @file line.ts
 * @brief Defines the `Line` class, a token representing a single line of text.
 *
 * This module introduces the `Line` class, which extends `BaseToken` and
 * encapsulates a line of text along with its positional information
 * (line number and character range). It serves as a fundamental building block
 * within the linesCodec tokenization system for editors, providing a structured
 * representation of textual content for further processing and rendering.
 * Domain: Text Editor, Tokenization, Data Structure.
 */

import { BaseToken } from '../../baseToken.js';
import { assert } from '../../../../../base/common/assert.js';
import { Range } from '../../../../../editor/common/core/range.js';

/**
 * @class Line
 * @augments BaseToken
 * @brief Represents a tokenized line of text with positional metadata.
 *
 * Functional Utility: This class extends `BaseToken` to specifically
 * represent a single line of text from a document. Its primary functional
 * utility is to encapsulate the actual text content (`text`) along with a
 * `Range` object that precisely defines the line's start and end positions
 * (including line number and column offsets) within the original source data.
 * This ensures that each `Line` token carries sufficient information for
 * accurate rendering, navigation, and text manipulation operations within
 * an editor context.
 * Invariant: A `Line` token always represents a contiguous sequence of characters on a single line within the document,
 *            and its `Range` precisely corresponds to its `text` content.
 */
export class Line extends BaseToken {
	/**
	 * @brief Constructs a new `Line` token.
	 * @param lineNumber The 1-based index of the line in the original document.
	 * @param text The string content of the line.
	 *
	 * Functional Utility: Initializes a `Line` token by establishing its
	 * textual content and its absolute position within the document.
	 * Assertions ensure that `lineNumber` is a valid, positive integer,
	 * preventing invalid state and ensuring correct positional mapping.
	 */
	constructor(
		// the line index
		// Note! 1-based indexing
		lineNumber: number,
		// the line contents
		public readonly text: string, /**< The immutable string content of the line. */
	) {
		// Block Logic: Ensures that the provided line number is a valid numerical value.
		// Pre-condition: `lineNumber` must not be `NaN`.
		assert(
			!isNaN(lineNumber),
			`The line number must not be a NaN.`,
		);

		// Block Logic: Enforces that the line number is a positive integer, as lines are 1-based indexed.
		// Pre-condition: `lineNumber` must be greater than 0.
		assert(
			lineNumber > 0,
			`The line number must be >= 1, got "${lineNumber}".`,
		);

		super(
			new Range(
				lineNumber,
				1,
				lineNumber,
				text.length + 1,
			),
		);
	}

	/**
	 * @brief Compares this `Line` token with another token for equality.
	 * @param other The other token to compare against.
	 * @returns `true` if the tokens are semantically equivalent, `false` otherwise.
	 *
	 * Functional Utility: This method determines if two `Line` tokens represent
	 * the same logical entity. It first defers to the `BaseToken`'s equality
	 * check (which likely compares their `Range` properties) and then
	 * performs a deep comparison of their `text` content. This ensures that
	 * two `Line` tokens are considered equal only if they occupy the same
	 * position and contain identical textual data.
	 */
	public override equals<T extends BaseToken>(other: T): boolean {
		// Block Logic: Delegates initial equality check to the `BaseToken` superclass.
		// Functional Utility: Efficiently prunes non-matching tokens based on their fundamental range properties.
		// Pre-condition: `other` is a `BaseToken` or a subclass.
		// Invariant: If `BaseToken`'s equality check fails, the tokens are definitively not equal.
		if (!super.equals(other)) {
			return false;
		}

		// Block Logic: Verifies that the `other` token is specifically an instance of `Line`.
		// Functional Utility: Ensures type compatibility for a semantic comparison, as only two `Line` tokens can be truly equal.
		// Pre-condition: `other` is a valid `BaseToken` instance.
		// Invariant: If `other` is not a `Line` token, it cannot be semantically equal to this `Line` token.
		if (!(other instanceof Line)) {
			return false;
		}

		return this.text === other.text;
	}

	/**
	 * @brief Returns a string representation of the `Line` token.
	 * @returns A string representing the token, including its truncated text and range.
	 *
	 * Functional Utility: This method generates a human-readable string
	 * for debugging and logging purposes. It provides a concise summary of
	 * the `Line` token, showing a shortened version of its `text` content
	 * and its `Range` information, which helps in quickly identifying
	 * the token's identity and location.
	 */
	public override toString(): string {
		return `line("${this.shortText()}")${this.range}`;
	}
}
