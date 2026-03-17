/**
 * @fileoverview This file contains unit tests for the static utility methods
 * of the `BaseToken` class, which appears to be a foundational component in a
 * text processing or tokenization engine, likely for an editor. The tests
 * validate the rendering of token sequences into strings and the calculation
 * of a bounding range for a sequence of tokens.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

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
 * Generates a random {@link Range} object.
 *
 * @throws if {@link maxNumber} argument is less than `2`,
 *         is equal to `NaN` or is `infinite`.
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
 * Generates a random {@link SimpleToken} instance.
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
 * @description Test suite for the static methods of the `BaseToken` class.
 */
suite('BaseToken', () => {
	ensureNoDisposablesAreLeakedInTestSuite();

	/**
	 * @description This suite tests the `BaseToken.render()` method, which is responsible
	 * for converting an array of token objects into its corresponding string representation.
	 */
	suite('• render', () => {
		/**
		 * Note! Range of tokens is ignored by the render method, hence
		 *       we generate random ranges for each token in this test.
		 */
		/**
		 * @description Validates that `render` correctly concatenates the string
		 * representations of various sequences of tokens.
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

		/**
		 * @description Checks the edge case where an empty list of tokens is provided,
		 * which should result in an empty string.
		 */
		test('• an empty list of tokens', () => {
			assert.strictEqual(
				'',
				BaseToken.render([]),
				`Must correctly render and empty list of tokens.`,
			);
		});
	});

	/**
	 * @description This suite tests the `BaseToken.fullRange()` method, which calculates
	 * the bounding range that encompasses all tokens in a given list.
	 */
	suite('• fullRange', () => {
		/**
		 * @description This sub-suite verifies that the method correctly throws errors
		 * when provided with invalid input.
		 */
		suite('• throws', () => {
			/**
			 * @description Ensures that the method throws an error if an empty array
			 * is passed, as a range cannot be determined.
			 */
			test('• if empty list provided', () => {
				assert.throws(() => {
					BaseToken.fullRange([]);
				});
			});

			/**
			 * @description Tests for an invalid logical ordering of tokens, where the
			 * first token in the array starts on a line after the last token.
			 */
			test('• if start line number of the first token is greater than one of the last token', () => {
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

			/**
			 * @description Tests for an invalid logical ordering on the same line, where
			 * the first token ends after the last token starts.
			 */
			test('• if start line numbers are equal and end of the first token is greater than the start of the last token', () => {
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
