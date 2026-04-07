/**
 * @file fileReference.ts
 * @brief Defines the `FileReference` class, a token representing a file reference in a chatbot prompt.
 *
 * This module introduces the `FileReference` class, extending `PromptVariableWithData`.
 * It specifically handles file references within prompt syntax, encapsulating the file's
 * path and providing methods for creation from generic prompt variables and comparison.
 * Domain: Chatbot, Prompt Engineering, Tokenization, File System Integration.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/


import { PromptVariableWithData } from './promptVariable.js';
import { assert } from '../../../../../../../base/common/assert.js';
import { IRange, Range } from '../../../../../../../editor/common/core/range.js';
import { BaseToken } from '../../../../../../../editor/common/codecs/baseToken.js';

/**
 * @const VARIABLE_NAME
 * @brief Defines the fixed name for file reference variables within chatbot prompts.
 * Functional Utility: Ensures consistency and proper identification of file reference tokens.
 * Invariant: The value of this constant is always 'file'.
 */
const VARIABLE_NAME: string = 'file';

/**
 * @class FileReference
 * @augments PromptVariableWithData
 * @brief Represents a tokenized reference to a file within a chatbot prompt.
 *
 * Functional Utility: This class extends `PromptVariableWithData` to specifically
 * encapsulate file references in chatbot prompts. It stores the file's path
 * and provides methods to create instances from generic prompt variables,
 * and to compare with other tokens.
 * Domain: Chatbot, Prompt Engineering, Tokenization, File System Integration.
 * Invariant: A `FileReference` token always has a fixed `VARIABLE_NAME` of 'file',
 *            and its `data` property represents the referenced file's path.
 */
export class FileReference extends PromptVariableWithData {
	/**
	 * @brief Constructs a new `FileReference` token.
	 * @param range The `Range` object defining the start and end positions of the file reference in the prompt.
	 * @param path The string representing the file path being referenced.
	 * Functional Utility: Initializes a `FileReference` token by setting its range,
	 *                     fixed variable name, and the specific file path data.
	 * Pre-condition: `range` must be a valid `Range` object; `path` must be a non-empty string representing a file path.
	 */
	constructor(
		range: Range,
		public readonly path: string,
	) {
		super(range, VARIABLE_NAME, path);
	}

	/**
	 * @method from
	 * @brief Creates a new `FileReference` token from a generic `PromptVariableWithData` instance.
	 * @param variable The `PromptVariableWithData` instance to convert.
	 * @returns A new `FileReference` token.
	 * @throws Error if the `variable.name` does not match `VARIABLE_NAME`.
	 *
	 * Functional Utility: Acts as a factory method to safely cast a `PromptVariableWithData`
	 *                     into a `FileReference`, ensuring that the variable represents a file.
	 * Pre-condition: `variable` must be an instance of `PromptVariableWithData` with a `name`
	 *                property equal to `VARIABLE_NAME`.
	 * Post-condition: A `FileReference` token is returned with its `range` and `path`
	 *                 derived from the input `variable`.
	 */
	public static from(variable: PromptVariableWithData) {
		// Block Logic: Asserts that the name of the prompt variable matches the predefined file variable name.
		// Pre-condition: `variable.name` must be identical to `VARIABLE_NAME`.
		assert(
			variable.name === VARIABLE_NAME,
			`Variable name must be '${VARIABLE_NAME}', got '${variable.name}'.`,
		);

		return new FileReference(
			variable.range,
			variable.data,
		);
	}

	/**
	 * @method equals
	 * @brief Compares this `FileReference` token with another token for semantic equality.
	 * Functional Utility: Determines if two file reference tokens are equivalent by first
	 *                     checking type compatibility and then delegating to the superclass
	 *                     for range and data comparison.
	 * @param other The other token to compare against.
	 * @returns `true` if the tokens are semantically equivalent, `false` otherwise.
	 * Pre-condition: `other` must be a `BaseToken` or a subclass.
	 * Invariant: Two `FileReference` tokens are equal if they are of the same type and
	 *            their `range`, `VARIABLE_NAME`, and `path` are identical.
	 */
	public override equals<T extends BaseToken>(other: T): boolean {
		// Block Logic: Verifies that the `other` token is an instance of `FileReference`.
		// Functional Utility: Ensures type compatibility for a semantic comparison.
		if ((other instanceof FileReference) === false) {
			return false;
		}

		// Block Logic: Delegates the actual comparison of range and data to the `PromptVariableWithData` superclass.
		// Functional Utility: Leverages existing comparison logic for shared properties of prompt variables.
		return super.equals(other);
	}

	/**
	 * @property linkRange
	 * @brief Retrieves the `IRange` of the file path within the file reference token.
	 * Functional Utility: Provides the precise positional information of the actual file path
	 *                     segment, which can be used for UI interactions like linking or highlighting.
	 * @returns An `IRange` object representing the file path, or `undefined` if the path is empty.
	 * Invariant: This property directly maps to the `dataRange` from the `PromptVariableWithData` superclass.
	 */
	public get linkRange(): IRange | undefined {
		return super.dataRange;
	}
}
