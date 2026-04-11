/**
 * @file This file contains unit tests for the `BaseToken` class and its static methods.
 * @author Microsoft Corporation
 * @license MIT License
 *
 * @description
 * The tests are structured into suites for each major functionality of `BaseToken`:
 * - `render`: Verifies that a list of tokens can be correctly converted into a string representation.
 * - `fullRange`: Checks the logic for calculating the combined range of a list of tokens,
 *   including error handling for invalid inputs.
 * - `equals`: Tests the instance method for comparing two tokens for equality, covering
 *   cases where tokens differ by class, text content, or range.
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
 * Generates a random {@link Range} object for testing purposes.
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
 * Generates a random {@link SimpleToken} instance for testing purposes.
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
 * @suite BaseToken
 * @description A test suite for the BaseToken class.
 */
suite('BaseToken', () => {
	ensureNoDisposablesAreLeakedInTestSuite();

	/**
	 * @suite • render
	 * @description Tests the static `render` method, which concatenates the text
	 * representation of a list of tokens.
	 */
	suite('• render', () => {
		/**
		 * Note! Range of tokens is ignored by the render method, hence
		 *       we generate random ranges for each token in this test.
		 */
		/**
		 * @test • a list of tokens
		 * @description Verifies that a sequence of different token types is rendered
		 * into the correct string output.
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
		 * @test • an empty list of tokens
		 * @description Ensures that rendering an empty list of tokens results in an empty string.
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
	 * @suite • fullRange
	 * @description Tests the static `fullRange` method, which calculates the total
	 * range spanning from the start of the first token to the end of the last token.
	 */
	suite('• fullRange', () => {
		/**
		 * @suite • throws
		 * @description Contains tests that verify error conditions for `fullRange`.
		 */
		suite('• throws', () => {
			/**
			 * @test • if empty list provided
			 * @description Ensures an error is thrown when trying to get the range of an empty token list.
			 */
			test('• if empty list provided', () => {
				assert.throws(() => {
					BaseToken.fullRange([]);
				});
			});

			/**
			 * @test • if start line number of the first token is greater than one of the last token
			 * @description Checks for an error when the token list is not properly ordered by line number.
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
			 * @test • if start line numbers are equal and end of the first token is greater than the start of the last token
			 * @description Checks for an error when tokens on the same line are not properly ordered by column.
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

	/**
	 * @suite • equals()
	 * @description Tests the instance method `equals()`, which compares two token instances.
	 */
	suite('• equals()', () => {
		/**
		 * @suite • false
		 * @description Groups tests that should result in `equals()` returning false.
		 */
		suite('• false', () => {
			/**
			 * @test • different constructor
			 * @description Verifies that tokens created from different classes are not equal, even if they share a base class and range.
			 */
			test('• different constructor', () => {
				test('• same base class', () => {
					class TestToken1 extends BaseToken {
						public override get text(): string {
							throw new Error('Method not implemented.');
						}

						public override toString(): string {
							throw new Error('Method not implemented.');
						}
					}

					class TestToken2 extends BaseToken {
						public override get text(): string {
							throw new Error('Method not implemented.');
						}

						public override toString(): string {
							throw new Error('Method not implemented.');
						}
					}

					const range = randomRange();
					const token1 = new TestToken1(range);
					const token2 = new TestToken2(range);

					assert.strictEqual(
						token1.equals(token2),
						false,
						`Token of type '${token1.constructor.name}' must not be equal to token of type '${token2.constructor.name}'.`,
					);

					assert.strictEqual(
						token2.equals(token1),
						false,
						`Token of type '${token2.constructor.name}' must not be equal to token of type '${token1.constructor.name}'.`,
					);
				});
			});

			/**
			 * @test • child
			 * @description Verifies that a base class token is not equal to a token from a derived class.
			 */
			test('• child', () => {
				class TestToken1 extends BaseToken {
					public override get text(): string {
						throw new Error('Method not implemented.');
					}

					public override toString(): string {
						throw new Error('Method not implemented.');
					}
				}

				class TestToken2 extends TestToken1 {
					constructor(
						range: Range,
					) {
						super(range);
					}
				}

				const range = randomRange();
				const token1 = new TestToken1(range);
				const token2 = new TestToken2(range);

				assert.strictEqual(
					token1.equals(token2),
					false,
					`Token of type '${token1.constructor.name}' must not be equal to token of type '${token2.constructor.name}'.`,
				);

				assert.strictEqual(
					token2.equals(token1),
					false,
					`Token of type '${token2.constructor.name}' must not be equal to token of type '${token1.constructor.name}'.`,
				);
			});

			/**
			 * @test • different text
			 * @description Verifies that tokens with different text values are not equal.
			 */
			test('• different text', () => {
				class TestToken extends BaseToken {
					constructor(
						private readonly value: string,
					) {
						super(new Range(1, 1, 1, 1 + value.length));
					}

					public override get text(): string {
						return this.value;
					}

					public override toString(): string {
						throw new Error('Method not implemented.');
					}
				}

				const token1 = new TestToken('text1');
				const token2 = new TestToken('text2');

				assert.strictEqual(
					token1.equals(token2),
					false,
					`Token of type '${token1.constructor.name}' must not be equal to token of type '${token2.constructor.name}'.`,
				);

				assert.strictEqual(
					token2.equals(token1),
					false,
					`Token of type '${token2.constructor.name}' must not be equal to token of type '${token1.constructor.name}'.`,
				);
			});

			/**
			 * @test • different range
			 * @description Verifies that tokens with different range values are not equal.
			 */
			test('• different range', () => {
				class TestToken extends BaseToken {
					public override get text(): string {
						return 'some text value';
					}

					public override toString(): string {
						throw new Error('Method not implemented.');
					}
				}

				const range1 = randomRange();
				const token1 = new TestToken(range1);

				// TODO: @legomushroom - generate range different from the one of token1
				const range2 = randomRange();
				const token2 = new TestToken(range2);

				assert.strictEqual(
					token1.equals(token2),
					false,
					`Token of type '${token1.constructor.name}' must not be equal to token of type '${token2.constructor.name}'.`,
				);

				assert.strictEqual(
					token2.equals(token1),
					false,
					`Token of type '${token2.constructor.name}' must not be equal to token of type '${token1.constructor.name}'.`,
				);
			});
		});
	});
});
