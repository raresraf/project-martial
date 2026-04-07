/**
 * @file promptVariable.ts
 * @brief Defines `PromptVariable` and `PromptVariableWithData` classes for tokenizing chatbot prompt variables.
 *
 * This module introduces foundational classes for representing variables within chatbot prompts.
 * `PromptVariable` handles basic `#variable` tokens, while `PromptVariableWithData` extends
 * this to include associated data (e.g., `#file:/path/to/file.md`). Both classes provide
 * validation and methods for reconstructing and comparing variable tokens.
 * Domain: Chatbot, Prompt Engineering, Tokenization, Variable Resolution, Data Handling.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { PromptToken } from './promptToken.js';
import { assert } from '../../../../../../../base/common/assert.js';
import { IRange, Range } from '../../../../../../../editor/common/core/range.js';
import { BaseToken } from '../../../../../../../editor/common/codecs/baseToken.js';
import { INVALID_NAME_CHARACTERS, STOP_CHARACTERS } from '../parsers/promptVariableParser.js';

/**
 * @const START_CHARACTER
 * @brief Defines the mandatory starting character for all prompt variables.
 * Functional Utility: Establishes a clear syntactic convention for recognizing variable tokens.
 * Invariant: The value of this constant is always '#'.
 */
const START_CHARACTER: string = '#';

/**
 * @const DATA_SEPARATOR
 * @brief Defines the character used to separate a prompt variable's name from its associated data.
 * Functional Utility: Provides a clear demarcation within variable tokens that carry additional data.
 * Invariant: The value of this constant is always ':'.
 */
const DATA_SEPARATOR: string = ':';

/**
 * @class PromptVariable
 * @augments PromptToken
 * @brief Represents a tokenized generic variable within a chatbot prompt.
 *
 * Functional Utility: This class extends `PromptToken` to encapsulate basic
 * prompt variables (e.g., `#context`). It stores the name of the variable
 * and provides validation to ensure the name contains only allowed characters.
 * This token type is a fundamental building block for structured prompts.
 * Domain: Chatbot, Prompt Engineering, Tokenization, Variable Resolution.
 * Invariant: A `PromptVariable` token always starts with `START_CHARACTER` ('#'),
 *            and its `name` property consists only of valid characters
 *            (i.e., not `INVALID_NAME_CHARACTERS` or `STOP_CHARACTERS`).
 */
export class PromptVariable extends PromptToken {
	/**
	 * @brief Constructs a new `PromptVariable` token.
	 * @param range The `Range` object defining the start and end positions of the variable in the prompt.
	 * @param name The name of the variable, excluding the leading '#' character.
	 * Functional Utility: Initializes a `PromptVariable` token, performing character-level
	 *                     validation to ensure the variable's name adheres to defined rules.
	 * Pre-condition: `range` must be a valid `Range` object. `name` must be a string that
	 *                does not contain any `INVALID_NAME_CHARACTERS` or `STOP_CHARACTERS`.
	 */
	constructor(
		range: Range,
		/**
		 * The name of a prompt variable, excluding the `#` character at the start.
		 */
		public readonly name: string,
	) {
		// Block Logic: Iterates through each character of the provided name to validate its composition.
		// Invariant: Each character in `name` must not be present in `INVALID_NAME_CHARACTERS` or `STOP_CHARACTERS`.
		for (const character of name) {
			assert(
				(INVALID_NAME_CHARACTERS.includes(character) === false) &&
				(STOP_CHARACTERS.includes(character) === false),
				`Variable 'name' cannot contain character '${character}', got '${name}'.`,
			);
		}

		super(range);
	}

	/**
	 * @property text
	 * @brief Returns the full textual representation of the `PromptVariable` token, including the starting '#' character.
	 * Functional Utility: Reconstructs the original variable string as it would appear in the prompt.
	 * @returns The complete variable string (e.g., "#context").
	 * Invariant: The returned string always starts with `START_CHARACTER` followed by the `name` property.
	 */
	public get text(): string {
		return `${START_CHARACTER}${this.name}`;
	}

	/**
	 * @method equals
	 * @brief Compares this `PromptVariable` token with another token for semantic equality.
	 * Functional Utility: Determines if two variable tokens are equivalent by comparing their range and textual content.
	 * @param other The other token to compare against.
	 * @returns `true` if the tokens are semantically equivalent, `false` otherwise.
	 * Pre-condition: `other` must be a `BaseToken` or a subclass.
	 * Invariant: Two `PromptVariable` tokens are equal if they occupy the same `Range` and have identical `text` content.
	 */
	public override equals<T extends BaseToken>(other: T): boolean {
		// Block Logic: Checks if the tokens occupy the same textual range.
		// Functional Utility: Efficiently prunes non-matching tokens based on their positional information.
		if (!super.sameRange(other.range)) {
			return false;
		}

		// Block Logic: Verifies that the `other` token is specifically an instance of `PromptVariable`.
		// Functional Utility: Ensures type compatibility for a semantic comparison.
		// Invariant: Only two `PromptVariable` tokens can be truly equal.
		if ((other instanceof PromptVariable) === false) {
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
	 * @brief Returns a string representation of the `PromptVariable` token for debugging.
	 * Functional Utility: Provides a concise string output for debugging, indicating the token's text and its range.
	 * @returns A string describing the `PromptVariable` token.
	 * Invariant: The string representation accurately reflects the token's content and location.
	 */
	public override toString(): string {
		return `${this.text}${this.range}`;
	}
}

/**
 * @class PromptVariableWithData
 * @augments PromptVariable
 * @brief Represents a tokenized prompt variable that includes additional data.
 *
 * Functional Utility: This class extends `PromptVariable` to handle tokens
 * that not only have a name but also an associated data value (e.g., `#file:/path/to/file.md`).
 * It encapsulates both the variable name and its data, with validation for the data
 * component. This is critical for implementing rich, data-driven prompt interactions.
 * Domain: Chatbot, Prompt Engineering, Tokenization, Data Handling.
 * Invariant: A `PromptVariableWithData` token always includes `DATA_SEPARATOR` (':')
 *            between its `name` and `data` properties, and its `data` property
 *            consists only of valid characters (i.e., not `STOP_CHARACTERS`).
 */
export class PromptVariableWithData extends PromptVariable {
	/**
	 * @brief Constructs a new `PromptVariableWithData` token.
	 * @param fullRange The `Range` object defining the start and end positions of the entire token in the prompt.
	 * @param name The name of the variable, excluding the leading '#' character.
	 * @param data The data associated with the variable, excluding the leading ':' character.
	 * Functional Utility: Initializes a `PromptVariableWithData` token, handling both the variable
	 *                     name and its associated data, with validation for the data component.
	 * Pre-condition: `fullRange` must be a valid `Range` object. `name` must be a valid variable name.
	 *                `data` must be a string that does not contain any `STOP_CHARACTERS`.
	 */
	constructor(
		fullRange: Range,
		/**
		 * The name of the variable, excluding the starting `#` character.
		 */
		name: string,

		/**
		 * The data of the variable, excluding the starting {@link DATA_SEPARATOR} character.
		 */
		public readonly data: string,
	) {
		super(fullRange, name);

		// Block Logic: Iterates through each character of the provided data to validate its composition.
		// Invariant: Each character in `data` must not be present in `STOP_CHARACTERS`.
		for (const character of data) {
			assert(
				(STOP_CHARACTERS.includes(character) === false),
				`Variable 'data' cannot contain character '${character}', got '${data}'.`,
			);
		}
	}

	/**
	 * Get full text of the token.
	 */
	/**
	 * @property text
	 * @brief Returns the full textual representation of the `PromptVariableWithData` token, including all its parts.
	 * Functional Utility: Reconstructs the original variable string with its name and data as it would appear in the prompt.
	 * @returns The complete variable string (e.g., "#file:/path/to/file.md").
	 * Invariant: The returned string always adheres to the `#name:data` format.
	 */
	public override get text(): string {
		return `${START_CHARACTER}${this.name}${DATA_SEPARATOR}${this.data}`;
	}

	/**
	 * @method equals
	 * @brief Compares this `PromptVariableWithData` token with another token for semantic equality.
	 * Functional Utility: Determines if two data-carrying variable tokens are equivalent by first
	 *                     checking type compatibility and then delegating to the superclass
	 *                     for range and base variable comparison.
	 * @param other The other token to compare against.
	 * @returns `true` if the tokens are semantically equivalent, `false` otherwise.
	 * Pre-condition: `other` must be a `BaseToken` or a subclass.
	 * Invariant: Two `PromptVariableWithData` tokens are equal if they are of the same type and
	 *            their `range`, `name`, and `data` are identical.
	 */
	public override equals<T extends BaseToken>(other: T): boolean {
		// Block Logic: Verifies that the `other` token is an instance of `PromptVariableWithData`.
		// Functional Utility: Ensures type compatibility for a semantic comparison.
		if ((other instanceof PromptVariableWithData) === false) {
			return false;
		}

		// Block Logic: Delegates the actual comparison of range and base variable properties to the `PromptVariable` superclass.
		// Functional Utility: Leverages existing comparison logic for shared properties of prompt variables.
		return super.equals(other);
	}

	/**
	 * @property dataRange
	 * @brief Calculates the `IRange` of the data part within the prompt variable token.
	 * Functional Utility: Provides precise positional information of the data segment,
	 *                     which can be used for UI interactions like highlighting or data extraction.
	 * @returns An `IRange` object representing the data's position, or `undefined` if the data part is empty.
	 * Invariant: The calculated range is always a sub-range of the overall token's `range` and
	 *            starts immediately after the variable name and data separator.
	 */
	public get dataRange(): IRange | undefined {
		const { range } = this;

		// Block Logic: Calculates the starting column of the data part by summing the lengths of
		//              the start character, variable name, and data separator.
		const dataStartColumn = range.startColumn +
			START_CHARACTER.length + this.name.length +
			DATA_SEPARATOR.length;

		// Block Logic: Constructs a new `Range` object specifically for the data part of the variable.
		const result = new Range(
			range.startLineNumber,
			dataStartColumn,
			range.endLineNumber,
			range.endColumn,
		);

		// Block Logic: Checks if the calculated data range is empty, indicating no actual data is present.
		// Functional Utility: Returns `undefined` if no data exists, preventing operations on empty ranges.
		if (result.isEmpty()) {
			return undefined;
		}

		return result;
	}
}
