/**
 * @file baseToken.ts
 * @brief Foundational abstractions for text tokenization and spatial range management.
 * @details Defines the base schema for elements extracted from source data, ensuring 
 * each token maintains a strict mapping to its original coordinates (line and column) 
 * through the `Range` object. Provides utilities for token serialization and range composition.
 * 
 * Domain: Compilers, Text Processing, Editor Foundations.
 */

import { pick } from '../../../base/common/arrays.js';
import { assert } from '../../../base/common/assert.js';
import { IRange, Range } from '../../../editor/common/core/range.js';

/**
 * @class BaseToken
 * @brief Abstract representation of a discrete atomic unit of text.
 * Functional Utility: Serves as the parent for all domain-specific tokens, 
 * encapsulating positioning and textual representation logic.
 */
export abstract class BaseToken {
	constructor(
		/**
		 * Spatial Metadata: The precise coordinates of this token in the source buffer.
		 */
		private _range: Range,
	) { }

	public get range(): Range {
		return this._range;
	}

	/**
	 * @property text
	 * @brief The raw textual content identified by this token.
	 */
	public abstract get text(): string;

	/**
	 * @brief Structural equality check for ranges.
	 */
	public sameRange(other: Range): boolean {
		return this.range.equalsRange(other);
	}

	public abstract toString(): string;

	/**
	 * @brief Deep equality check for tokens.
	 * Logic: Compares constructor type, text content length, text content, and range.
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
	 * @brief Immutable-style range modification.
	 * @return A new instance or updated self with the modified range components.
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
	 * @brief Static Utility: Serializes a sequence of tokens back into a contiguous string.
	 */
	public static render(tokens: readonly BaseToken[]): string {
		return tokens.map(pick('text')).join('');
	}

	/**
	 * @brief Aggregation: Computes the minimal bounding range covering a sequence of tokens.
	 * Pre-condition: Tokens must be provided in sequential order.
	 * @param tokens Sequence of tokens to envelope.
	 * @throws Assertion error if the list is empty or non-contiguous.
	 */
	public static fullRange(tokens: readonly BaseToken[]): Range {
		assert(
			tokens.length > 0,
			'Cannot get full range for an empty list of tokens.',
		);

		const firstToken = tokens[0];
		const lastToken = tokens[tokens.length - 1];

		// Consistency Check: Ensures the tokens actually form a valid forward-moving sequence.
		assert(
			firstToken.range.startLineNumber <= lastToken.range.startLineNumber,
			'First token must start on previous or the same line as the last token.',
		);
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
	 * @brief Debug Utility: Provides a truncated preview of the token's text.
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
 * @brief Represents a composite token formed by a sequence of sub-tokens.
 * Functional Utility: Used to group tokens that collectively form a plain text 
 * segment without specific semantic meaning.
 */
export class Text<TToken extends BaseToken = BaseToken> extends BaseToken {
	public get text(): string {
		// Rendering: Reconstructs the segment by concatenating sub-token text.
		return BaseToken.render(this.tokens);
	}

	constructor(
		range: Range,
		public readonly tokens: readonly TToken[],
	) {
		super(range);
	}

	/**
	 * @brief Factory: Constructs a Text token from a list, automatically inferring the range.
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

	public override toString(): string {
		return `text(${this.shortText()})${this.range}`;
	}
}
