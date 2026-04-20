/**
 * @b9c394d7-35dd-4cf4-a94f-97bd9e547344/src/vs/editor/common/codecs/baseToken.ts
 * @brief Foundational abstractions for text tokenization and spatial range management.
 * 
 * Functional Intent: Defines the base schema for elements extracted from source 
 * data, ensuring each token maintains a strict mapping to its original coordinates 
 * (line and column). Provides utilities for token serialization, deep equality 
 * comparison, and range-based aggregation.
 * 
 * Domain: Compilers, Text Processing, Editor Foundations.
 */

import { pick } from '../../../base/common/arrays.js';
import { assert } from '../../../base/common/assert.js';
import { IRange, Range } from '../../../editor/common/core/range.js';

/**
 * @brief Abstract representation of a discrete atomic unit of text.
 * 
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
	 * equals - Deep equality check for tokens.
	 * Logic: Compares constructor type, text content length, text content, and range 
	 * to ensure both identity and value equivalence.
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
	 * withRange - Immutable-style range modification.
	 * Logic: Merges existing range components with provided overrides to produce 
	 * a updated coordinate set for the token.
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
	 * fullRange - Computes the minimal bounding range covering a sequence of tokens.
	 * 
	 * Block Logic: Range aggregation and consistency verification.
	 * Pre-condition: Tokens must be provided in sequential order and non-empty.
	 * Logic: 
	 * 1. Identifies the start of the first token and the end of the last.
	 * 2. Enforces structural invariants (non-decreasing line/column numbers) 
	 *    to detect non-contiguous or reversed input sequences.
	 * 
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
 * @brief Represents a composite token formed by a sequence of sub-tokens.
 * 
 * Functional Utility: Groups discrete tokens that collectively form a logical 
 * text segment, facilitating tree-based token structures.
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
	 * fromTokens - Constructs a Text token from a list with inferred range.
	 * Logic: Automatically calculates the bounding envelope for the provided tokens 
	 * and creates a composite parent.
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
