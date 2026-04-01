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
	 * Create new `Word` token with the given `text` and the range
	 * inside the given `Line` at the specified `column number`.
	 */
	public static newOnLine(
		text: string,
		line: Line,
		atColumnNumber: number,
	): Word {
		const { range } = line;

		const startPosition = new Position(range.startLineNumber, atColumnNumber);
		const endPosition = new Position(range.startLineNumber, atColumnNumber + text.length);

		return new Word(
			Range.fromPositions(startPosition, endPosition),
			text,
		);
	}

	/**
	 * Check if this token is equal to another one.
	 */
	public override equals<T extends BaseToken>(other: T): boolean {
		if (!super.equals(other)) {
			return false;
		}

		if (!(other instanceof Word)) {
			return false;
		}

		return this.text === other.text;
	}

	/**
	 * Returns a string representation of the token.
	 */
	public override toString(): string {
		return `word("${this.shortText()}")${this.range}`;
	}
}
