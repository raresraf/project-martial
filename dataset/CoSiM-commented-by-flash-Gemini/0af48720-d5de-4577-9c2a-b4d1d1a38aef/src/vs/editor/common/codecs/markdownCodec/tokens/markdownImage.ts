/**
 * @file markdownImage.ts
 * @brief Defines the `MarkdownImage` class, a token representing a Markdown image.
 *
 * This module introduces the `MarkdownImage` class, extending `MarkdownToken`
 * to specifically handle Markdown image syntax (e.g., `![alt text](image.png)`).
 * It encapsulates the image's caption and reference, along with positional information.
 * This class provides functionality to validate the image's source as a URL and to
 * extract the range of the link part within the document.
 * Domain: Markdown Parsing, Text Editor, Tokenization, Hypermedia.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { BaseToken } from '../../baseToken.js';
import { MarkdownToken } from './markdownToken.js';
import { IRange, Range } from '../../../core/range.js';
import { assert } from '../../../../../base/common/assert.js';

/**
 * @class MarkdownImage
 * @augments MarkdownToken
 * @brief Represents a tokenized Markdown image (e.g., `![alt text](image.png)`).
 *
 * Functional Utility: This class extends `MarkdownToken` to encapsulate the structure
 * and content of a Markdown image, including its caption (alt text) and reference (path/URL).
 * It provides mechanisms to determine if the image source is a URL and to extract
 * the range of the link part.
 * Domain: Markdown Parsing, Text Editor, Tokenization, Hypermedia.
 * Invariant: A `MarkdownImage` token always starts with `!`, its caption is enclosed
 *            in square brackets `[]`, and its reference is enclosed in parentheses `()`.
 *            Its `Range` precisely corresponds to the combined span of its caption and reference.
 */
export class MarkdownImage extends MarkdownToken {
	/**
	 * @property isURL
	 * @brief Indicates whether the image's reference path is a syntactically valid URL.
	 * Functional Utility: Facilitates distinguishing between local file paths and external web resources for images.
	 * Invariant: This property is determined at construction time based on the `path` property and is immutable.
	 */
	public readonly isURL: boolean;

	/**
	 * @brief Constructs a new `MarkdownImage` token.
	 * @param lineNumber The 1-based starting line number of the image.
	 * @param columnNumber The 1-based starting column number of the image.
	 * @param caption The full caption text, including the leading '!' and enclosing square brackets (e.g., `![alt text]`).
	 * @param reference The full reference path/URL, including enclosing parentheses (e.g., `(image.png)`).
	 * Functional Utility: Initializes a Markdown image token, performing structural validation on its components
	 *                     and determining if its reference is a valid URL.
	 * Pre-condition: `lineNumber` and `columnNumber` must be positive integers.
	 *                `caption` must start with '!' and be enclosed in '[]'.
	 *                `reference` must be enclosed in '()'.
	 * Post-condition: `isURL` property is correctly set based on the `path`.
	 */
	constructor(
		/**
		 * The starting line number of the image (1-based indexing).
		 */
		lineNumber: number,
		/**
		 * The starting column number of the image (1-based indexing).
		 */
		columnNumber: number,
		/**
		 * The caption of the image, including the `!` and `square brackets`.
		 */
		private readonly caption: string,
		/**
		 * The reference of the image, including the parentheses.
		 */
		private readonly reference: string,
	) {
		// Block Logic: Ensures `lineNumber` is a valid number.
		assert(
			!isNaN(lineNumber),
			`The line number must not be a NaN.`,
		);

		// Block Logic: Enforces `lineNumber` to be a positive integer.
		assert(
			lineNumber > 0,
			`The line number must be >= 1, got "${lineNumber}".`,
		);

		// Block Logic: Enforces `columnNumber` to be a positive integer.
		assert(
			columnNumber > 0,
			`The column number must be >= 1, got "${columnNumber}".`,
		);

		// Block Logic: Validates that the caption starts with the Markdown image indicator.
		assert(
			caption[0] === '!',
			`The caption must start with '!' character, got "${caption}".`,
		);

		// Block Logic: Validates that the caption (alt text) is correctly enclosed in square brackets.
		assert(
			caption[1] === '[' && caption[caption.length - 1] === ']',
			`The caption must be enclosed in square brackets, got "${caption}".`,
		);

		// Block Logic: Validates that the image reference (path/URL) is correctly enclosed in parentheses.
		assert(
			reference[0] === '(' && reference[reference.length - 1] === ')',
			`The reference must be enclosed in parentheses, got "${reference}".`,
		);

		super(
			new Range(
				lineNumber,
				columnNumber,
				lineNumber,
				columnNumber + caption.length + reference.length,
			),
		);

		// Block Logic: Determines if the image reference is a valid URL by attempting to construct a URL object.
		// Invariant: `isURL` flag is set immutably during construction.
		try {
			new URL(this.path);
			this.isURL = true;
		} catch {
			this.isURL = false;
		}
	}

	/**
	 * @property text
	 * @brief Returns the full textual representation of the Markdown image token.
	 * Functional Utility: Concatenates the caption and reference to reconstruct the original image syntax.
	 * @returns The combined string of the image caption and reference.
	 * Invariant: The returned string always adheres to the `![caption](reference)` Markdown image format.
	 */
	public override get text(): string {
		return `${this.caption}${this.reference}`;
	}

	/**
	 * @property path
	 * @brief Extracts the raw path or URL from the image's reference, removing the enclosing parentheses.
	 * Functional Utility: Provides direct access to the image's source for loading or validation.
	 * @returns The string content of the reference without '()' delimiters.
	 * Pre-condition: `reference` property is a non-empty string enclosed in parentheses.
	 */
	public get path(): string {
		return this.reference.slice(1, this.reference.length - 1);
	}

	/**
	 * @method equals
	 * @brief Compares this `MarkdownImage` token with another token for semantic equality.
	 * Functional Utility: Determines if two image tokens are equivalent by comparing their range and combined textual content.
	 * @param other The other token to compare against.
	 * @returns `true` if the tokens are semantically equivalent, `false` otherwise.
	 * Pre-condition: `other` must be a `BaseToken` or a subclass.
	 * Invariant: Two `MarkdownImage` tokens are equal if they occupy the same `Range` and have identical combined `caption` and `reference` content.
	 */
	public override equals<T extends BaseToken>(other: T): boolean {
		// Block Logic: Checks if the tokens occupy the same textual range.
		// Functional Utility: Efficiently prunes non-matching tokens based on their positional information.
		if (!super.sameRange(other.range)) {
			return false;
		}

		// Block Logic: Verifies that the `other` token is specifically an instance of `MarkdownImage`.
		// Functional Utility: Ensures type compatibility for a semantic comparison.
		// Invariant: Only two `MarkdownImage` tokens can be truly equal.
		if (!(other instanceof MarkdownImage)) {
			return false;
		}

		return this.text === other.text;
	}

	/**
	 * @property linkRange
	 * @brief Calculates the `Range` of the image's reference (the path/URL part) within the document.
	 * Functional Utility: Provides precise positional information for the clickable or referencable part of the image,
	 *                     useful for navigation, validation, or UI interactions.
	 * @returns An `IRange` object representing the link's position, or `undefined` if the path is empty.
	 * Invariant: The calculated range is always a sub-range of the overall `MarkdownImage` token's `range`.
	 *            The `startColumn` correctly accounts for the caption's length and the opening parenthesis.
	 * Pre-condition: `range` and `caption` properties must be valid.
	 */
	public get linkRange(): IRange | undefined {
		// Block Logic: Returns undefined if the image path is empty, as there is no link to range.
		if (this.path.length === 0) {
			return undefined;
		}

		const { range } = this;

		// Inline: Calculate start column by adding caption length and accounting for the opening parenthesis of the reference.
		const startColumn = range.startColumn + this.caption.length + 1;
		const endColumn = startColumn + this.path.length;

		return new Range(
			range.startLineNumber,
			startColumn,
			range.endLineNumber,
			endColumn,
		);
	}

	/**
	 * @method toString
	 * @brief Returns a string representation of the `MarkdownImage` token for debugging.
	 * Functional Utility: Provides a concise string output for debugging, indicating the token type, a shortened text preview, and its range.
	 * @returns A string describing the `MarkdownImage` token.
	 * Invariant: The string representation accurately reflects the token's type, content (truncated), and location.
	 */
	public override toString(): string {
		return `md-image("${this.shortText()}")${this.range}`;
	}
}
