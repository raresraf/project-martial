/**
 * @file promptTemplateVariable.ts
 * @brief Defines the `PromptTemplateVariable` class, representing a template variable token in a chatbot prompt.
 *
 * This module introduces the `PromptTemplateVariable` class, extending `PromptToken`.
 * It encapsulates template variables (e.g., `${variableName}`) within prompt syntax,
 * storing their internal content and providing mechanisms to reconstruct their full
 * textual representation. This token type is crucial for parameterized prompts in
 * conversational AI systems.
 * Domain: Chatbot, Prompt Engineering, Tokenization, Template Processing.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { PromptToken } from './promptToken.js';
import { Range } from '../../../../../../../editor/common/core/range.js';
import { BaseToken } from '../../../../../../../editor/common/codecs/baseToken.js';
import { DollarSign } from '../../../../../../../editor/common/codecs/simpleCodec/tokens/dollarSign.js';
import { LeftCurlyBrace, RightCurlyBrace } from '../../../../../../../editor/common/codecs/simpleCodec/tokens/curlyBraces.js';

/**
 * @class PromptTemplateVariable
 * @augments PromptToken
 * @brief Represents a tokenized template variable within a chatbot prompt.
 *
 * Functional Utility: This class extends `PromptToken` to specifically encapsulate
 * template variables (e.g., `${variableName}`). It stores the internal content
 * of the variable (excluding the `${}` delimiters) and provides a mechanism
 * to reconstruct its full textual representation. This token type is crucial
 * for parameterized prompts in conversational AI systems.
 * Domain: Chatbot, Prompt Engineering, Tokenization, Template Processing.
 * Invariant: A `PromptTemplateVariable` token always starts with `${` and ends with `}`,
 *            and its `contents` property represents the text within these delimiters.
 */
export class PromptTemplateVariable extends PromptToken {
	/**
	 * @brief Constructs a new `PromptTemplateVariable` token.
	 * @param range The `Range` object defining the start and end positions of the template variable in the prompt.
	 * @param contents The internal content of the template variable, excluding the `${}` delimiters.
	 * Functional Utility: Initializes a `PromptTemplateVariable` token, storing its range and internal content.
	 * Pre-condition: `range` must be a valid `Range` object; `contents` can be any string.
	 */
	constructor(
		range: Range,
		/**
		 * The contents of the template variable, excluding
		 * the surrounding `${}` characters.
		 */
		public readonly contents: string,
	) {
		super(range);
	}

	/**
	 * @property text
	 * @brief Returns the full textual representation of the `PromptTemplateVariable` token, including its `${}` delimiters.
	 * Functional Utility: Reconstructs the original template variable string as it would appear in the prompt.
	 * @returns The complete template variable string (e.g., "${myVar}").
	 * Invariant: The returned string always adheres to the `${contents}` format.
	 */
	public get text(): string {
		return [
			DollarSign.symbol,
			LeftCurlyBrace.symbol,
			this.contents,
			RightCurlyBrace.symbol,
		].join('');
	}

	/**
	 * @method equals
	 * @brief Compares this `PromptTemplateVariable` token with another token for semantic equality.
	 * Functional Utility: Determines if two template variable tokens are equivalent by comparing their range and textual content.
	 * @param other The other token to compare against.
	 * @returns `true` if the tokens are semantically equivalent, `false` otherwise.
	 * Pre-condition: `other` must be a `BaseToken` or a subclass.
	 * Invariant: Two `PromptTemplateVariable` tokens are equal if they occupy the same `Range` and have identical `text` content.
	 */
	public override equals<T extends BaseToken>(other: T): boolean {
		// Block Logic: Checks if the tokens occupy the same textual range.
		// Functional Utility: Efficiently prunes non-matching tokens based on their positional information.
		if (!super.sameRange(other.range)) {
			return false;
		}

		// Block Logic: Verifies that the `other` token is specifically an instance of `PromptTemplateVariable`.
		// Functional Utility: Ensures type compatibility for a semantic comparison.
		// Invariant: Only two `PromptTemplateVariable` tokens can be truly equal.
		if (!(other instanceof PromptTemplateVariable)) {
			return false;
		}

		// Block Logic: Optimizes comparison by quickly ruling out tokens with different total text lengths.
		// Functional Utility: Prevents more expensive string content comparison if lengths do not match.
		if (this.text.length !== other.text.length) {
			return false;
		}

		return this.text === other.text;
	}

	/**
	 * @method toString
	 * @brief Returns a string representation of the `PromptTemplateVariable` token for debugging.
	 * Functional Utility: Provides a concise string output for debugging, indicating the token's text and its range.
	 * @returns A string describing the `PromptTemplateVariable` token.
	 * Invariant: The string representation accurately reflects the token's content and location.
	 */
	public override toString(): string {
		return `${this.text}${this.range}`;
	}
}
