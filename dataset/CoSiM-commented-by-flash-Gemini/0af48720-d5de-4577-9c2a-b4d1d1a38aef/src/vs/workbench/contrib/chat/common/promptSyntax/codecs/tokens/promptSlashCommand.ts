/**
 * @file promptSlashCommand.ts
 * @brief Defines the `PromptSlashCommand` class, representing a slash command token in a chatbot prompt.
 *
 * This module introduces the `PromptSlashCommand` class, extending `PromptToken`.
 * It encapsulates slash commands (e.g., `/help` or `/search`) within prompt syntax,
 * storing the name of the command and validating its character composition.
 * This token type is crucial for identifying executable instructions given to a
 * conversational AI system.
 * Domain: Chatbot, Prompt Engineering, Tokenization, Command Parsing.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { PromptToken } from './promptToken.js';
import { assert } from '../../../../../../../base/common/assert.js';
import { Range } from '../../../../../../../editor/common/core/range.js';
import { BaseToken } from '../../../../../../../editor/common/codecs/baseToken.js';
import { INVALID_NAME_CHARACTERS, STOP_CHARACTERS } from '../parsers/promptSlashCommandParser.js';

/**
 * @const START_CHARACTER
 * @brief Defines the mandatory starting character for all prompt slash commands.
 * Functional Utility: Establishes a clear syntactic convention for recognizing command tokens.
 * Invariant: The value of this constant is always '/'.
 */
const START_CHARACTER: string = '/';

/**
 * @class PromptSlashCommand
 * @augments PromptToken
 * @brief Represents a tokenized slash command within a chatbot prompt.
 *
 * Functional Utility: This class extends `PromptToken` to specifically encapsulate
 * slash commands (e.g., `/help` or `/search`). It stores the name of the command
 * and provides validation to ensure the name contains only allowed characters.
 * This token type is crucial for identifying executable instructions given to a
 * conversational AI system.
 * Domain: Chatbot, Prompt Engineering, Tokenization, Command Parsing.
 * Invariant: A `PromptSlashCommand` token always starts with `START_CHARACTER` ('/'),
 *            and its `name` property consists only of valid characters
 *            (i.e., not `INVALID_NAME_CHARACTERS` or `STOP_CHARACTERS`).
 */
export class PromptSlashCommand extends PromptToken {
	/**
	 * @brief Constructs a new `PromptSlashCommand` token.
	 * @param range The `Range` object defining the start and end positions of the command in the prompt.
	 * @param name The name of the command, excluding the leading '/' character.
	 * Functional Utility: Initializes a `PromptSlashCommand` token, performing character-level
	 *                     validation to ensure the command's name adheres to defined rules.
	 * Pre-condition: `range` must be a valid `Range` object. `name` must be a string that
	 *                does not contain any `INVALID_NAME_CHARACTERS` or `STOP_CHARACTERS`.
	 */
	constructor(
		range: Range,
		/**
		 * The name of a command, excluding the `/` character at the start.
		 */
		public readonly name: string,
	) {
		// Block Logic: Iterates through each character of the provided name to validate its composition.
		// Invariant: Each character in `name` must not be present in `INVALID_NAME_CHARACTERS` or `STOP_CHARACTERS`.
		for (const character of name) {
			assert(
				(INVALID_NAME_CHARACTERS.includes(character) === false) &&
				(STOP_CHARACTERS.includes(character) === false),
				`Slash command 'name' cannot contain character '${character}', got '${name}'.`,
			);
		}

		super(range);
	}

	/**
	 * @property text
	 * @brief Returns the full textual representation of the `PromptSlashCommand` token, including the starting '/' character.
	 * Functional Utility: Reconstructs the original command string as it would appear in the prompt.
	 * @returns The complete command string (e.g., "/help").
	 * Invariant: The returned string always starts with `START_CHARACTER` followed by the `name` property.
	 */
	public get text(): string {
		return `${START_CHARACTER}${this.name}`;
	}

	/**
	 * @method equals
	 * @brief Compares this `PromptSlashCommand` token with another token for semantic equality.
	 * Functional Utility: Determines if two command tokens are equivalent by comparing their range and textual content.
	 * @param other The other token to compare against.
	 * @returns `true` if the tokens are semantically equivalent, `false` otherwise.
	 * Pre-condition: `other` must be a `BaseToken` or a subclass.
	 * Invariant: Two `PromptSlashCommand` tokens are equal if they occupy the same `Range` and have identical `text` content.
	 */
	public override equals<T extends BaseToken>(other: T): boolean {
		// Block Logic: Checks if the tokens occupy the same textual range.
		// Functional Utility: Efficiently prunes non-matching tokens based on their positional information.
		if (!super.sameRange(other.range)) {
			return false;
		}

		// Block Logic: Verifies that the `other` token is specifically an instance of `PromptSlashCommand`.
		// Functional Utility: Ensures type compatibility for a semantic comparison.
		// Invariant: Only two `PromptSlashCommand` tokens can be truly equal.
		if ((other instanceof PromptSlashCommand) === false) {
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
	 * @brief Returns a string representation of the `PromptSlashCommand` token for debugging.
	 * Functional Utility: Provides a concise string output for debugging, indicating the token's text and its range.
	 * @returns A string describing the `PromptSlashCommand` token.
	 * Invariant: The string representation accurately reflects the token's content and location.
	 */
	public override toString(): string {
		return `${this.text}${this.range}`;
	}
}
