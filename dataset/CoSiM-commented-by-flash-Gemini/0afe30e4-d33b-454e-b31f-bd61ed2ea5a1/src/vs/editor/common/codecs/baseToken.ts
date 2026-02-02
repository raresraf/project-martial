/**
 * @file baseToken.ts
 * @brief Defines foundational structures for representing and managing tokens within a text editor.
 *
 * This module introduces `BaseToken`, an abstract class providing core functionalities for tokens,
 * including range management and basic comparisons. It also defines the `Text` class, which
 * represents a sequence of tokens without additional semantic meaning, primarily used for grouping.
 *
 * Functional Utility: These classes serve as building blocks for syntax highlighting, code parsing,
 * and structural analysis within the editor, enabling efficient manipulation and querying of text segments.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { pick } from '../../../base/common/arrays.js';
import { assert } from '../../../base/common/assert.js';
import { IRange, Range } from '../../../editor/common/core/range.js';

/**
 * @class BaseToken
 * @brief Abstract base class for all tokens, providing fundamental properties and methods.
 * Functional Utility: Establishes a common interface for tokens, ensuring consistent
 * handling of their textual range, textual representation, and equality comparisons.
 * It serves as a contract for concrete token implementations, centralizing common token logic.
 */
export abstract class BaseToken {
	constructor(
		private _range: Range,
	) { }

	/**
	 * Range of the token in the original text.
	 */
	public get range(): Range {
		return this._range;
	}

	/**
	 * @property text
	 * @brief Returns the textual representation of the token.
	 * Functional Utility: Provides direct access to the string content that the token represents in the source code.
	 * @returns The string content of the token.
	 */
	public abstract get text(): string;

	/**
	 * @method sameRange
	 * @brief Checks if this token has the same range as another provided range.
	 * Functional Utility: Facilitates efficient comparison of token positions, essential for tasks like merging or diffing tokens.
	 * @param other The range to compare against.
	 * @returns True if the ranges are identical, false otherwise.
	 */
	public sameRange(other: Range): boolean {
		return this.range.equalsRange(other);
	}

	/**
	 * @method toString
	 * @brief Returns a string representation of the token for debugging purposes.
	 * Functional Utility: Provides a human-readable representation of the token instance, useful for logging and diagnostics.
	 * @returns A string describing the token.
	 */
	public abstract toString(): string;

	/**
	 * @method equals
	 * @brief Compares this token with another token for equality, including type, text content, and range.
	 * Functional Utility: Provides a comprehensive equality check for tokens, ensuring that two tokens are identical in their semantic and positional attributes.
	 * @param other The other token to compare.
	 * @returns True if the tokens are equal in type, text content, and range, false otherwise.
	 * Pre-condition: `other` must be an instance of `BaseToken` or a subclass.
	 * Invariant: If two tokens are equal, their constructor, text, and range must be identical.
	 */
	public equals(other: BaseToken): other is typeof this {
		/**
		 * Block Logic: Type checking to ensure comparison is between compatible token types.
		 * Invariant: If constructors differ, tokens are not equal.
		 */
		if (other.constructor !== this.constructor) {
			return false;
		}

		/**
		 * Block Logic: Length comparison for text content.
		 * Invariant: If text lengths differ, text content cannot be equal.
		 */
		if (this.text.length !== other.text.length) {
			return false;
		}

		/**
		 * Block Logic: Direct text content comparison.
		 * Invariant: If text content differs, tokens are not equal.
		 */
		if (this.text !== other.text) {
			return false;
		}

		// Functional Utility: Final range comparison to confirm full equality.
		return this.sameRange(other.range);
	}

	/**
	 * @method withRange
	 * @brief Creates a new token instance with modified range components.
	 * Functional Utility: Allows for immutable modification of a token's range, returning a new instance with updated position data.
	 * @param components An object containing partial range components to update (startLineNumber, startColumn, endLineNumber, endColumn).
	 * @returns A new instance of the token with the updated range.
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
	 * @method collapseRangeToStart
	 * @brief Collapses the token's range to its starting position.
	 * Functional Utility: Modifies the token's range to represent a single point at its beginning, useful for operations that only care about the start of a token.
	 * @returns The current token instance with its range collapsed to its start.
	 * See {@link Range.collapseToStart} for more details.
	 */
	public collapseRangeToStart(): this {
		this._range = this._range.collapseToStart();

		return this;
	}

	/**
	 * @method render
	 * @brief Concatenates the text representation of a list of tokens.
	 * Functional Utility: Reconstructs a string from an array of tokens, effectively reversing the tokenization process.
	 * @param tokens A readonly array of BaseToken instances.
	 * @returns A single string formed by joining the text of all tokens.
	 */
	public static render(tokens: readonly BaseToken[]): string {
		return tokens.map(pick('text')).join('');
	}

	/**
	 * @method fullRange
	 * @brief Calculates the encompassing range for a sequence of tokens.
	 * Functional Utility: Determines the start of the first token and the end of the last token to define the overall span of a token list.
	 * @param tokens A readonly array of BaseToken instances.
	 * @returns A Range object representing the combined span of all tokens.
	 * @throws Assertion error if:
	 * 	- provided {@link tokens} list is empty
	 *  - the first token start number is greater than the start line of the last token
	 *  - if the first and last token are on the same line, the first token start column must
	 * 	  be smaller than the start column of the last token
	 * Pre-condition: The `tokens` array must not be empty.
	 * Invariant: The calculated range will always start at or before the end of the last token.
	 */
	public static fullRange(tokens: readonly BaseToken[]): Range {
		assert(
			tokens.length > 0,
			'Cannot get full range for an empty list of tokens.',
		);

		const firstToken = tokens[0];
		const lastToken = tokens[tokens.length - 1];

		/**
		 * Block Logic: Sanity check to ensure the first token's start line is not after the last token's start line.
		 * Invariant: `firstToken.range.startLineNumber` <= `lastToken.range.startLineNumber`.
		 */
		assert(
			firstToken.range.startLineNumber <= lastToken.range.startLineNumber,
			'First token must start on previous or the same line as the last token.',
		);
		/**
		 * Block Logic: Conditional assertion for tokens on the same line, ensuring correct column order.
		 * Pre-condition: `firstToken` and `lastToken` are on the same line.
		 * Invariant: The end column of the first token must be less than or equal to the start column of the last token.
		 */
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
	 * @method shortText
	 * @brief Returns a truncated version of the token's text.
	 * Functional Utility: Provides a concise textual representation, primarily for display in limited UI spaces or for debugging summaries.
	 * @param maxLength The maximum desired length of the returned string. Defaults to 32.
	 * @returns The full text if it's within `maxLength`, otherwise a truncated string with '...' appended.
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
 * @class Text
 * @brief Represents a token that is a composition of other tokens, without introducing new semantic meaning beyond concatenation.
 * Functional Utility: Used to group a sequence of `BaseToken` instances into a single logical unit, simplifying hierarchical token structures.
 * @template TToken The type of tokens contained within this Text token.
 */
export class Text<TToken extends BaseToken = BaseToken> extends BaseToken {
	/**
	 * @property text
	 * @brief Returns the concatenated text of all contained tokens.
	 * Functional Utility: Provides the complete string content represented by this composite Text token.
	 * @returns The combined text of its child tokens.
	 */
	public override get text(): string {
		return BaseToken.render(this.tokens);
	}

	constructor(
		range: Range,
		public readonly tokens: readonly TToken[],
	) {
		super(range);
	}

	/**
	 * @method fromTokens
	 * @brief Creates a new Text token from an array of existing tokens.
	 * Functional Utility: A static factory method that constructs a `Text` token and automatically derives its range from the provided list of child tokens.
	 * @template TToken The type of tokens contained within the new Text token.
	 * @param tokens A readonly array of `BaseToken` instances to compose.
	 * @returns A new `Text` token instance.
	 * @throws Assertion error if the provided `tokens` list is empty.
	 * Pre-condition: The `tokens` array must not be empty to infer the range.
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
	 * @method toString
	 * @brief Returns a string representation of the Text token for debugging.
	 * Functional Utility: Provides a concise string output for debugging, indicating the token type, a shortened text preview, and its range.
	 * @returns A string describing the Text token.
	 */
	public override toString(): string {
		return `text(${this.shortText()})${this.range}`;
	}
}
