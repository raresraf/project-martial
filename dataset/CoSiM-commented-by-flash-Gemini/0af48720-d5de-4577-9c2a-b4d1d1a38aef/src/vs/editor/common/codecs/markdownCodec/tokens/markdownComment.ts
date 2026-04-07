/**
 * @file markdownComment.ts
 * @brief Defines the `MarkdownComment` class, a token representing a Markdown comment.
 *
 * This module introduces the `MarkdownComment` class, which extends `MarkdownToken`
 * to specifically handle Markdown comment syntax (e.g., `<!-- ... -->`). It encapsulates
 * the raw text of the comment along with its positional information within the document.
 * This class facilitates parsing, rendering, and structural analysis of Markdown content
 * by providing a structured representation for comments.
 * Domain: Markdown Parsing, Text Editor, Tokenization.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { BaseToken } from '../../baseToken.js';
import { Range } from '../../../core/range.js';
import { MarkdownToken } from './markdownToken.js';
import { assert } from '../../../../../base/common/assert.js';

/**
 * @class MarkdownComment
 * @augments MarkdownToken
 * @brief Represents a tokenized Markdown comment (e.g., `<!-- comment -->`).
 *
 * Functional Utility: This class extends `MarkdownToken` to specifically encapsulate
 * a Markdown comment, including its textual content and positional information.
 * It provides methods to verify the comment's structure and compare it with other tokens.
 * Domain: Markdown Parsing, Text Editor, Tokenization.
 * Invariant: A `MarkdownComment` token's `text` property always begins with `<!--` and,
 *            if complete, ends with `-->`. Its `Range` precisely corresponds to its `text` content.
 */
export class MarkdownComment extends MarkdownToken {
	/**
	 * @brief Constructs a new `MarkdownComment` token.
	 * @param range The range of the comment token in the original document.
	 * @param text The raw string content of the Markdown comment.
	 * Functional Utility: Initializes a `MarkdownComment` token, validating that the provided text
	 *                     conforms to the expected Markdown comment starting syntax.
	 * Pre-condition: `text` must be a string starting with "<!--".
	 */
	constructor(
		range: Range,
		public readonly text: string,
	) {
		// Block Logic: Ensures that the comment's text begins with the standard Markdown comment opening delimiter.
		// Pre-condition: `text` must start with '<!--'.
		assert(
			text.startsWith('<!--'),
			`The comment must start with '<!--', got '${text.substring(0, 10)}'.`,
		);

		super(range);
	}

	/**
	 * @property hasEndMarker
	 * @brief Indicates whether the Markdown comment token includes an ending delimiter (`-->`).
	 * Functional Utility: Provides a quick check for the structural completeness of a Markdown comment.
	 * @returns `true` if the comment text ends with `-->`, `false` otherwise.
	 * Invariant: The value is derived directly from the `text` property and reflects its suffix.
	 */
	public get hasEndMarker(): boolean {
		return this.text.endsWith('-->');
	}

	/**
	 * @method equals
	 * @brief Compares this `MarkdownComment` token with another token for semantic equality.
	 * Functional Utility: Determines if two comment tokens are equivalent by comparing their range and textual content.
	 * @param other The other token to compare against.
	 * @returns `true` if the tokens are semantically equivalent, `false` otherwise.
	 * Pre-condition: `other` must be a `BaseToken` or a subclass.
	 * Invariant: Two `MarkdownComment` tokens are equal if they occupy the same `Range` and have identical `text` content.
	 */
	public override equals<T extends BaseToken>(other: T): boolean {
		// Block Logic: Checks if the tokens occupy the same textual range.
		// Functional Utility: Prunes non-matching tokens based on their positional information.
		if (!super.sameRange(other.range)) {
			return false;
		}

		// Block Logic: Verifies that the `other` token is specifically an instance of `MarkdownComment`.
		// Functional Utility: Ensures type compatibility for a semantic comparison.
		// Invariant: Only two `MarkdownComment` tokens can be truly equal.
		if (!(other instanceof MarkdownComment)) {
			return false;
		}

		return this.text === other.text;
	}

	/**
	 * @method toString
	 * @brief Returns a string representation of the `MarkdownComment` token for debugging.
	 * Functional Utility: Provides a concise string output for debugging, indicating the token type, a shortened text preview, and its range.
	 * @returns A string describing the `MarkdownComment` token.
	 * Invariant: The string representation accurately reflects the token's type, content (truncated), and location.
	 */
	public override toString(): string {
		return `md-comment("${this.shortText()}")${this.range}`;
	}
}
