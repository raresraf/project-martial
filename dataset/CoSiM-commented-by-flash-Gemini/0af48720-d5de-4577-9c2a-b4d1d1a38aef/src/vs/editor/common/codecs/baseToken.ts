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
 * Domain: Text Editor, Tokenization, Abstract Syntax Tree (AST), UI/UX (Syntax Highlighting).
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
 * Invariant: All concrete token implementations derived from `BaseToken` must maintain
 * a valid `Range` and a non-empty `text` representation.
 */
export abstract class BaseToken {
	constructor(
		private _range: Range,
	) { }

	/**
	 * @property range
	 * @brief Retrieves the immutable range (start and end positions) of the token within the document.
	 * Functional Utility: Provides the precise location of the token, crucial for selection,
	 * highlighting, and other editor features that operate on text spans.
	 * @returns A `Range` object representing the token's position.
	 */
	public get range(): Range {
		return this._range;
	}

	/**
	 * @property text
	 * @brief Returns the textual representation of the token.
	 * Functional Utility: Provides direct access to the string content that the token represents in the source code.
	 * @returns The string content of the token.
	 * Pre-condition: Implementations must ensure this returns the actual substring of the document represented by the token.
	 */
	public abstract get text(): string;

	/**
	 * @method sameRange
	 * @brief Checks if this token occupies the same range as another provided range.
	 * Functional Utility: Facilitates efficient comparison of token positions, essential for tasks like merging or diffing tokens.
	 * @param other The range to compare against.
	 * @returns True if the ranges are identical, false otherwise.
	 * Pre-condition: `other` must be a valid `Range` object.
	 */
	public sameRange(other: Range): boolean {
		return this.range.equalsRange(other);
	}

	/**
	 * @method toString
	 * @brief Returns a string representation of the token for debugging purposes.
	 * Functional Utility: Provides a human-readable representation of the token instance, useful for logging and diagnostics.
	 * @returns A string describing the token.
	 * Invariant: The string representation should include key token properties for easy identification.
	 */
	public abstract toString(): string;

	/**
	 * @method equals
	 * @brief Compares this token with another token for equality.
	 * Functional Utility: Determines if two token instances are logically equivalent, based on their type and range.
	 * @param other The other token to compare.
	 * @returns True if the tokens are equal, false otherwise.
	 * Pre-condition: `other` must be an instance of `BaseToken` or a subclass.
	 * Invariant: Equality implies that both tokens are of the same conceptual type and occupy the exact same textual range.
	 */
	public equals<T extends BaseToken>(other: T): boolean {
		// Block Logic: Ensures type compatibility before comparing ranges.
		// Pre-condition: `other` should be a `BaseToken` subclass.
		// Invariant: Only tokens of the same exact class can be considered equal for structural integrity.
		if (!(other instanceof this.constructor)) {
			return false;
		}

		return this.sameRange(other.range);
	}

	/**
	 * @method withRange
	 * @brief Creates a new token instance with modified range components.
	 * Functional Utility: Allows for immutable modification of a token's range, returning a new instance with updated position data.
	 * @param components An object containing partial range components to update (startLineNumber, startColumn, endLineNumber, endColumn).
	 * @returns A new instance of the token with the updated range.
	 * Pre-condition: `components` must contain valid (partial) range coordinates.
	 * Post-condition: The returned token will have an updated `_range` property reflecting the new components,
	 *                 while other properties remain unchanged.
	 */
	public withRange(components: Partial<IRange>): this {
		// Block Logic: Constructs a new `Range` object by merging existing range components with provided updates.
		// Invariant: The resulting range remains valid and consistent with editor's coordinate system.
		this._range = new Range(
			components.startLineNumber ?? this.range.startLineNumber,
			components.startColumn ?? this.range.startColumn,
			components.endLineNumber ?? this.range.endLineNumber,
			components.endColumn ?? this.range.endColumn,
		);

		return this;
	}

	/**
	 * @method render
	 * @brief Concatenates the text representation of a list of tokens.
	 * Functional Utility: Reconstructs a string from an array of tokens, effectively reversing the tokenization process.
	 * @param tokens A readonly array of BaseToken instances.
	 * @returns A single string formed by joining the text of all tokens.
	 * Pre-condition: `tokens` array contains valid `BaseToken` instances.
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
	 * @throws Assertion error if the provided `tokens` list is empty, or if the token ranges are inconsistent (e.g., first token starts after the last token).
	 * Pre-condition: The `tokens` array must not be empty.
	 * Invariant: The calculated range will always start at or before the end of the last token.
	 */
	public static fullRange(tokens: readonly BaseToken[]): Range {
		// Block Logic: Ensures that a non-empty list of tokens is provided to calculate a valid range.
		// Pre-condition: The `tokens` array must contain at least one element.
		assert(
			tokens.length > 0,
			'Cannot get full range for an empty list of tokens.',
		);

		const firstToken = tokens[0];
		const lastToken = tokens[tokens.length - 1];

		// sanity checks for the full range we would construct
		// Block Logic: Ensures that the first token does not start after the last token in terms of line number.
		// Invariant: Line numbers for tokens in the array should be monotonically increasing or remain the same.
		assert(
			firstToken.range.startLineNumber <= lastToken.range.startLineNumber,
			'First token must start on previous or the same line as the last token.',
		);
		/**
		 * Block Logic: Validates token order when start lines are identical.
		 * Pre-condition: `firstToken` and `lastToken` are on the same line.
		 * Invariant: The end column of the first token must be less than or equal to the start column of the last token to ensure non-overlapping or contiguous ranges.
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
	 * Pre-condition: `maxLength` must be a non-negative integer.
	 */
	public shortText(
		maxLength: number = 32,
	): string {
		// Block Logic: Truncates the token's text if its length exceeds the specified maximum.
		// Invariant: The returned string's length will not exceed `maxLength`, unless the original text is already shorter.
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
 * Invariant: The `Text` token's range must encompass the full range of all its constituent `tokens`.
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
	 * Post-condition: A `Text` token is returned whose range is derived from the `fullRange` of the input `tokens`.
	 */
	public static fromTokens<TToken extends BaseToken = BaseToken>(
		tokens: readonly TToken[],
	): Text<TToken> {
		// Block Logic: Asserts that the input token array is not empty to ensure a valid range can be inferred.
		// Pre-condition: `tokens` array must contain at least one element.
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
