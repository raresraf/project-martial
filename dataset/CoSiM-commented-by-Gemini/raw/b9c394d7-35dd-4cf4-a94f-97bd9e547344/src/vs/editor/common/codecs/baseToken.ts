/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { pick } from '../../../base/common/arrays.js';
import { assert } from '../../../base/common/assert.js';
import { IRange, Range } from '../../../editor/common/core/range.js';

/**
 * @fileoverview
 * This file defines the foundational token structures used for text processing and analysis within the editor.
 * It provides an abstract `BaseToken` class that represents a segment of text with a specific range,
 * and a concrete `Text` class for representing a sequence of other tokens. These classes are essential
 * for tokenization, parsing, and semantic analysis tasks.
 */

/**
 * @description Base class for all tokens with a `range` that
 * reflects token position in the original data. This class provides
 * a foundational structure for more specialized token types.
 */
export abstract class BaseToken {
	constructor(
		private _range: Range,
	) { }

	/**
	 * @description The range of the token in the source document, indicating its start and end position.
	 */
	public get range(): Range {
		return this._range;
	}

	/**
	 * @description Return text representation of the token.
	 * This is an abstract property that must be implemented by subclasses.
	 */
	public abstract get text(): string;

	/**
	 * @description Check if this token has the same range as another one.
	 * @param {Range} other The range to compare against.
	 * @returns {boolean} True if the ranges are identical, false otherwise.
	 */
	public sameRange(other: Range): boolean {
		return this.range.equalsRange(other);
	}

	/**
	 * @description Returns a string representation of the token for debugging and logging purposes.
	 * This is an abstract method that must be implemented by subclasses.
	 */
	public abstract toString(): string;

	/**
	 * @description Check if this token is equal to another one by comparing their constructor,
	 * text content, and range.
	 * @param {BaseToken} other The token to compare against.
	 * @returns {boolean} True if the tokens are equal, false otherwise.
	 */
	public equals(other: BaseToken): other is typeof this {
		if (other.constructor !== this.constructor) {
			return false;
		}

		if (this.text.length !== other.text.length) {
			return false;
		}

		if (this.text !== other.text) {
			return false;
		}

		return this.sameRange(other.range);
	}

	/**
	 * @description Creates a new instance of the token with a modified `range`.
	 * This allows for the creation of tokens with adjusted positions without mutating the original.
	 * @param {Partial<IRange>} components The components of the new range to apply.
	 * @returns {this} A new token instance with the updated range.
	 */
	public withRange(components: Partial<IRange>): this {
		this._range = new Range(
			components.startLineNumber ?? this.range.startLineNumber,
			components.startColumn ?? this.range.startColumn,
			components.endLineNumber ?? this.range.endLineNumber,
			components.endColumn ?? this.range.endColumn,
		);

		return this;
	}

	/**
	 * @description Render a list of tokens into a single concatenated string.
	 * @param {readonly BaseToken[]} tokens The list of tokens to render.
	 * @returns {string} The combined text of all tokens.
	 */
	public static render(tokens: readonly BaseToken[]): string {
		return tokens.map(pick('text')).join('');
	}

	/**
	 * @description Returns the full range that encompasses a list of tokens, from the start
	 * of the first token to the end of the last one.
	 *
	 * @param {readonly BaseToken[]} tokens The sequence of tokens.
	 * @returns {Range} The encompassing range.
	 * @throws if:
	 * 	- provided {@link tokens} list is empty
	 *  - the first token start number is greater than the start line of the last token
	 *  - if the first and last token are on the same line, the first token start column must
	 * 	  be smaller than the start column of the last token
	 */
	public static fullRange(tokens: readonly BaseToken[]): Range {
		assert(
			tokens.length > 0,
			'Cannot get full range for an empty list of tokens.',
		);

		const firstToken = tokens[0];
		const lastToken = tokens[tokens.length - 1];

		// Invariant: The start of the first token must not be after the start of the last token.
		assert(
			firstToken.range.startLineNumber <= lastToken.range.startLineNumber,
			'First token must start on previous or the same line as the last token.',
		);
		// Invariant: If on the same line, the first token must not start after the last one.
		if ((firstToken !== lastToken) && (firstToken.range.startLineNumber === lastToken.range.startLineNumber)) {
			assert(
				firstToken.range.endColumn <= lastToken.range.startColumn,
				[
					'First token must end at least on previous or the same column as the last token.',
					`First token: ${firstToken}; Last token: ${lastToken}.`,
				].join('\n'),
			);
		}

		return new Range(
			firstToken.range.startLineNumber,
			firstToken.range.startColumn,
			lastToken.range.endLineNumber,
			lastToken.range.endColumn,
		);
	}

	/**
	 * @description Provides a truncated version of the token's text, suitable for display in summaries or logs.
	 * @param {number} maxLength The maximum length of the returned string.
	 * @returns {string} The shortened text.
	 */
	public shortText(
		maxLength: number = 32,
	): string {
		if (this.text.length <= maxLength) {
			return this.text;
		}

		return `${this.text.slice(0, maxLength - 1)}...`;
	}
}

/**
 * @description Represents a sequence of tokens that does not hold a specific
 * semantic meaning on its own, but serves as a container for other tokens.
 * This is useful for grouping tokens into a single logical unit.
 * @template TToken The type of tokens contained within this text sequence.
 */
export class Text<TToken extends BaseToken = BaseToken> extends BaseToken {
	/**
	 * @description The combined text representation of all tokens in the sequence.
	 */
	public get text(): string {
		return BaseToken.render(this.tokens);
	}

	constructor(
		range: Range,
		/**
		 * @description The readonly array of tokens that constitute this text block.
		 */
		public readonly tokens: readonly TToken[],
	) {
		super(range);
	}

	/**
	 * @description Creates a new `Text` instance from a provided list of tokens, automatically
	 * inferring its range from the first and last tokens in the list.
	 *
	 * @template TToken The type of tokens to be included.
	 * @param {readonly TToken[]} tokens The list of tokens.
	 * @returns {Text<TToken>} A new `Text` token encompassing the provided tokens.
	 * @throws if the provided tokens list is empty, as range inference is not possible.
	 */
	public static fromTokens<TToken extends BaseToken = BaseToken>(
		tokens: readonly TToken[],
	): Text<TToken> {
		assert(
			tokens.length > 0,
			'Cannot infer range from an empty list of tokens.',
		);

		const range = BaseToken.fullRange(tokens);

		return new Text(range, tokens);
	}

	/**
	 * @description Returns a string representation of the `Text` token for debugging purposes.
	 * @returns {string} A compact string showing the token's type, short text, and range.
	 */
	public override toString(): string {
		return `text(${this.shortText()})${this.range}`;
	}
}
