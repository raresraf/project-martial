/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
/**
 * @file baseToken.test.ts
 * @brief Test suite for the BaseToken class and associated utilities.
 *
 * This file contains unit tests for the `BaseToken` class's `render` and `fullRange` methods,
 * as well as utility functions like `randomRange` and `randomSimpleToken`.
 * It validates the correct handling of token ranges and text rendering within a tokenization context.
 */

import assert from 'assert';
import { Range } from '../../../common/core/range.js';
import { randomInt } from '../../../../base/common/numbers.js';
import { BaseToken } from '../../../common/codecs/baseToken.js';
import { assertDefined } from '../../../../base/common/types.js';
import { randomBoolean } from '../../../../base/test/common/testUtils.js';
import { NewLine } from '../../../common/codecs/linesCodec/tokens/newLine.js';
import { CarriageReturn } from '../../../common/codecs/linesCodec/tokens/carriageReturn.js';
import { ensureNoDisposablesAreLeakedInTestSuite } from '../../../../base/test/common/utils.js';
import { TSimpleToken, WELL_KNOWN_TOKENS } from '../../../common/codecs/simpleCodec/simpleDecoder.js';
import { ISimpleTokenClass, SimpleToken } from '../../../common/codecs/simpleCodec/tokens/simpleToken.js';
import { At, Colon, DollarSign, ExclamationMark, Hash, LeftAngleBracket, LeftBracket, LeftCurlyBrace, RightAngleBracket, RightBracket, RightCurlyBrace, Slash, Space, Word } from '../../../common/codecs/simpleCodec/tokens/index.js';

/**
 * @function randomRange
 * @description Generates a random {@link Range} object, which is a fundamental data structure
 *   representing a contiguous selection within a text document (defined by start and end line/column numbers).
 *   This utility is crucial for creating diverse and realistic test data for token-related operations,
 *   ensuring robust validation of token parsing, rendering, and range manipulation.
 * @param {number} maxNumber - The upper bound for random line and column numbers. Defaults to 1_000.
 *   This parameter controls the scale of the generated ranges, influencing the depth and breadth
 *   of test scenarios.
 * @returns {Range} A randomly generated Range object.
 * @throws {Error} If `maxNumber` is less than `2`, `NaN`, or `infinite`, ensuring valid input for range generation.
 *
 * @algorithm
 * 1. **Input Validation**: Checks if `maxNumber` is valid (greater than 1, not NaN, not infinite).
 * 2. **Start Line Number**: Generates a random integer between 1 and `maxNumber` for `startLineNumber`.
 * 3. **End Line Number**:
 *    - Randomly decides (50% chance) if `endLineNumber` should be the same as `startLineNumber`.
 *    - If different, generates a random integer between `startLineNumber` and `2 * maxNumber`.
 *    - This ensures that `endLineNumber` is always greater than or equal to `startLineNumber`,
 *      covering single-line and multi-line ranges.
 * 4. **Start Column Number**: Generates a random integer between 1 and `maxNumber` for `startColumnNumber`.
 * 5. **End Column Number**:
 *    - Randomly decides (50% chance) if `endColumnNumber` should be `startColumnNumber + 1`.
 *    - If different, generates a random integer between `startColumnNumber + 1` and `2 * maxNumber`.
 *    - This guarantees `endColumnNumber` is always greater than `startColumnNumber` (a range must span at least one character)
 *      and handles both single-character and multi-character column ranges.
 * 6. **Range Instantiation**: Constructs and returns a new `Range` object with the generated values.
 */
const randomRange = (
	maxNumber: number = 1_000,
): Range => {
	assert(
		maxNumber > 1,
		`Max number must be greater than 1, got '${maxNumber}'.`,
	);

	const startLineNumber = randomInt(maxNumber, 1);
	const endLineNumber = (randomBoolean() === true)
		? startLineNumber
		: randomInt(2 * maxNumber, startLineNumber);

	const startColumnNumber = randomInt(maxNumber, 1);
	const endColumnNumber = (randomBoolean() === true)
		? startColumnNumber + 1
		: randomInt(2 * maxNumber, startColumnNumber + 1);

	return new Range(
		startLineNumber,
		startColumnNumber,
		endLineNumber,
		endColumnNumber,
	);
};

/**
 * List of simple tokens to randomly select from
 * in the {@link randomSimpleToken} utility.
 */
const TOKENS: readonly ISimpleTokenClass<TSimpleToken>[] = Object.freeze([
	...WELL_KNOWN_TOKENS,
	CarriageReturn,
	NewLine,
]);

/**
 * @function randomSimpleToken
 * @description Generates a random {@link TSimpleToken} instance, a simplified representation
 *   of a lexical token used in the editor's tokenization process. This function is essential
 *   for creating varied and unpredictable token inputs, crucial for comprehensive testing
 *   of token-based algorithms, rendering, and manipulation.
 * @returns {TSimpleToken} A randomly generated SimpleToken object, encapsulating a specific
 *   token type and a randomly determined range within a hypothetical document.
 * @throws {Error} If a constructor object for a well-known token cannot be found at a random index,
 *   indicating a potential misconfiguration of the `TOKENS` array.
 *
 * @algorithm
 * 1. **Random Index Generation**: A random integer `index` is generated within the bounds
 *    of the `TOKENS` array length, utilizing `randomInt(TOKENS.length - 1)`. This ensures
 *    a uniform probability distribution for selecting any defined token type.
 * 2. **Constructor Retrieval**: The `Constructor` for the chosen token type is retrieved
 *    from the `TOKENS` array using the generated `index`.
 * 3. **Assertion**: An assertion (`assertDefined`) verifies that a valid constructor
 *    was found, preventing runtime errors if the `TOKENS` array is improperly managed.
 * 4. **Instance Creation**: A new instance of the `Constructor` is created, passing a
 *    randomly generated `Range` (obtained via `randomRange()`) to its constructor.
 *    This combines a random token type with a random position in the document.
 * 5. **Return Value**: The newly instantiated `TSimpleToken` object is returned, ready
 *    for use in testing scenarios.
 */
const randomSimpleToken = (): TSimpleToken => {

	const index = randomInt(TOKENS.length - 1);

	const Constructor = TOKENS[index];
	assertDefined(
		Constructor,
		`Cannot find a constructor object for a well-known token at index '${index}'.`,
	);

	return new Constructor(randomRange());
};

/**
 * @testSuite BaseToken
 * @description Comprehensive test suite for the `BaseToken` class, ensuring its core functionalities
 *   related to text rendering and range calculations behave as expected. This suite is critical
 *   for maintaining the integrity and correctness of the editor's tokenization infrastructure.
 */
suite('BaseToken', () => {

	ensureNoDisposablesAreLeakedInTestSuite();

	suite('• render', () => {
		/**
		 * @testCase Verifies the `BaseToken.render` method's ability to correctly concatenate
		 *   the textual representation of a diverse list of tokens. It ensures that the
		 *   rendered string matches the expected output for various token combinations.
		 *
		 * Note: The range information of individual tokens is deliberately ignored by the `render` method,
		 *       which is why random ranges are generated for each token in this test case.
		 */
		test('• a list of tokens', () => {
			const tests: readonly [string, BaseToken[]][] = [
				['/textoftheword$#', [
					new Slash(randomRange()),
					new Word(randomRange(), 'textoftheword'),
					new DollarSign(randomRange()),
					new Hash(randomRange()),
				]],
				['<:👋helou👋:>', [
					new LeftAngleBracket(randomRange()),
					new Colon(randomRange()),
					new Word(randomRange(), '👋helou👋'),
					new Colon(randomRange()),
					new RightAngleBracket(randomRange()),
				]],
				[' {$#[ !@! ]#$} ', [
					new Space(randomRange()),
					new LeftCurlyBrace(randomRange()),
					new DollarSign(randomRange()),
					new Hash(randomRange()),
					new LeftBracket(randomRange()),
					new Space(randomRange()),
					new ExclamationMark(randomRange()),
					new At(randomRange()),
					new ExclamationMark(randomRange()),
					new Space(randomRange()),
					new RightBracket(randomRange()),
					new Hash(randomRange()),
					new DollarSign(randomRange()),
					new RightCurlyBrace(randomRange()),
					new Space(randomRange()),
				]],
			];

			for (const test of tests) {
				const [expectedText, tokens] = test;

				assert.strictEqual(
					expectedText,
					BaseToken.render(tokens),
				);
			}
		});

		test('• an empty list of tokens', () => {
			/**
			 * @testCase Validates that `BaseToken.render` returns an empty string when provided
			 *   with an empty array of tokens, ensuring correct handling of edge cases.
			 */
			assert.strictEqual(
				'',
				BaseToken.render([]),
				`Must correctly render and empty list of tokens.`,
			);
		});
	});

	suite('• fullRange', () => {
		/**
		 * @testSuite fullRange
		 * @description Tests the `BaseToken.fullRange` static method, which calculates the
		 *   minimal bounding {@link Range} that encloses all tokens in a given list.
		 *   This suite ensures that the method correctly identifies the earliest start
		 *   and latest end positions across a collection of tokens.
		 */
		suite('• throws', () => {
			test('• if empty list provided', () => {
				/**
				 * @testCase Verifies that `BaseToken.fullRange` throws an error when an empty
				 *   array of tokens is provided, as a valid encompassing range cannot be
				 *   determined from an empty set. This tests the method's robustness against invalid input.
				 */
				assert.throws(() => {
					BaseToken.fullRange([]);
				});
			});

			test('• if start line number of the first token is greater than one of the last token', () => {
				/**
				 * @testCase Validates that `BaseToken.fullRange` throws an error if the logical order
				 *   of tokens is violated, specifically when the `startLineNumber` of the first token
				 *   is greater than that of the last token. This enforces the precondition that tokens
				 *   must be ordered sequentially for correct range computation.
				 */
				assert.throws(() => {
					const lastToken = randomSimpleToken();

					// generate a first token with starting line number that is
					// greater than the start line number of the last token
					const startLineNumber = lastToken.range.startLineNumber + randomInt(10, 1);
					const firstToken = new Colon(
						new Range(
							startLineNumber,
							lastToken.range.startColumn,
							startLineNumber,
							lastToken.range.startColumn + 1,
						),
					);

					BaseToken.fullRange([
						firstToken,
						// tokens in the middle are ignored, so we
						// generate random ones to fill the gap
						randomSimpleToken(),
						randomSimpleToken(),
						randomSimpleToken(),
						randomSimpleToken(),
						randomSimpleToken(),
						// -
						lastToken,
					]);
				});
			});

			test('• if start line numbers are equal and end of the first token is greater than the start of the last token', () => {
				/**
				 * @testCase Ensures `BaseToken.fullRange` throws an error when tokens are on the same
				 *   line but provided in an overlapping or inverted order, specifically if the `endColumn`
				 *   of the first token is greater than the `startColumn` of the last token. This validates
				 *   the method's adherence to a strictly sequential and non-overlapping token input for
				 *   accurate range aggregation.
				 */
				assert.throws(() => {
					const firstToken = randomSimpleToken();

					const lastToken = new Hash(
						new Range(
							firstToken.range.startLineNumber,
							firstToken.range.endColumn - 1,
							firstToken.range.startLineNumber + randomInt(10),
							firstToken.range.endColumn,
						),
					);

					BaseToken.fullRange([
						firstToken,
						// tokens in the middle are ignored, so we
						// generate random ones to fill the gap
						randomSimpleToken(),
						randomSimpleToken(),
						randomSimpleToken(),
						randomSimpleToken(),
						randomSimpleToken(),
						// -
						lastToken,
					]);
				});
			});
		});
	});
});
