/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
/**
 * @file word.ts
 * @brief Defines the `Word` class, a token representing a sequence of continuous characters.
 *
 * This module introduces the `Word` class, extending `BaseToken`. It is designed
 * to represent a "word" as a fundamental token, characterized by a continuous
 * sequence of characters uninterrupted by common delimiters (like spaces, tabs,
 * or newlines). This class encapsulates the word's text content and its
 * `Range` within the document, providing essential functionality for text
 * processing, analysis, and editor operations.
 * Domain: Text Editor, Tokenization, Lexical Analysis, Text Processing.
 */

import { BaseToken } from '../../baseToken.js';
import { Line } from '../../linesCodec/tokens/line.js';
import { Range } from '../../../../../editor/common/core/range.js';
import { Position } from '../../../../../editor/common/core/position.js';

/**
 * @class Word
 * @augments BaseToken
 * @brief Represents a tokenized word within a text document.
 *
 * Functional Utility: This class extends `BaseToken` to represent a "word"
 * as a distinct lexical unit. A word is defined here as a continuous sequence
 * of characters that are not considered stop characters (e.g., spaces, tabs,
 * newlines). The `Word` token encapsulates both the textual content of the
 * word and its `Range` within the document, making it a critical component
 * for lexical analysis, syntax highlighting, and text editing operations
 * that operate on individual words.
 * Invariant: A `Word` token always represents a contiguous sequence of characters that form a single word,
 *            and its `Range` precisely corresponds to its `text` content.
 */
export class Word extends BaseToken {
	/**
	 * @brief Constructs a new `Word` token.
	 * @param range The `Range` object defining the start and end positions of the word in the document.
	 * @param text The string value of the word.
	 *
	 * Functional Utility: Initializes a `Word` token by associating its textual
	 * content with its precise location within the document. This is crucial
	 * for any operation that needs to know both *what* the word is and *where*
	 * it is located, such as selection, replacement, or semantic analysis.
	 */
	constructor(
		/**
		 * The word range.
		 */
		range: Range,

		/**
		 * The string value of the word.
		 */
		public readonly text: string,
	) {
		super(range);
	}

	/**
	 * @method newOnLine
	 * @brief Creates a new `Word` token, inferring its range based on a line and column.
	 * @param text The string content of the word.
	 * @param line The `Line` token within which the new word token resides.
	 * @param atColumnNumber The 1-based starting column number of the word within the line.
	 * @returns A new `Word` token instance.
	 *
	 * Functional Utility: This static factory method simplifies the creation of `Word` tokens
	 *                     by automatically calculating their `Range` from the provided line,
	 *                     column, and text. It's particularly useful during parsing when
	 *                     words are identified within a known line context.
	 * Pre-condition: `atColumnNumber` must be a positive integer and `text` must be non-empty.
	 *                The calculated range must be valid within the `line`'s boundaries.
	 * Post-condition: A `Word` token is returned with a `Range` accurately representing its
	 *                 position within the specified `line`.
	 */
	public static newOnLine(
		text: string,
		line: Line,
		atColumnNumber: number,
	): Word {
		const { range } = line;

		// Block Logic: Constructs `Position` objects to define the start and end of the word.
		const startPosition = new Position(range.startLineNumber, atColumnNumber);
		const endPosition = new Position(range.startLineNumber, atColumnNumber + text.length);

		// Functional Utility: Creates a new `Word` token with the calculated range and provided text.
		return new Word(
			Range.fromPositions(startPosition, endPosition),
			text,
		);
	}

	/**
	 * @method equals
	 * @brief Compares this `Word` token with another token for semantic equality.
	 * Functional Utility: Determines if two word tokens are equivalent by comparing their range and textual content.
	 * @param other The other token to compare against.
	 * @returns `true` if the tokens are semantically equivalent, `false` otherwise.
	 * Pre-condition: `other` must be a `BaseToken` or a subclass.
	 * Invariant: Two `Word` tokens are equal if they occupy the same `Range` and have identical `text` content.
	 */
	public override equals<T extends BaseToken>(other: T): boolean {
		// Block Logic: Delegates initial equality check to the `BaseToken` superclass.
		// Functional Utility: Efficiently prunes non-matching tokens based on their fundamental range properties.
		// Pre-condition: `other` is a `BaseToken` or a subclass.
		// Invariant: If `BaseToken`'s equality check fails, the tokens are definitively not equal.
		if (!super.equals(other)) {
			return false;
		}

		// Block Logic: Verifies that the `other` token is specifically an instance of `Word`.
		// Functional Utility: Ensures type compatibility for a semantic comparison.
		// Invariant: Only two `Word` tokens can be truly equal.
		if (!(other instanceof Word)) {
			return false;
		}

		return this.text === other.text;
	}

	/**
	 * @method toString
	 * @brief Returns a string representation of the `Word` token for debugging.
	 * Functional Utility: Provides a concise string output for debugging, indicating the token type, a shortened text preview, and its range.
	 * @returns A string describing the `Word` token.
	 * Invariant: The string representation accurately reflects the token's type, content (truncated), and location.
	 */
	public override toString(): string {
		return `word("${this.shortText()}")${this.range}`;
	}
}
