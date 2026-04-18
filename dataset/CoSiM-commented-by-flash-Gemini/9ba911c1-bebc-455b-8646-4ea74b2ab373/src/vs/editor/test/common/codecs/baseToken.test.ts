/**
 * @9ba911c1-bebc-455b-8646-4ea74b2ab373/src/vs/editor/test/common/codecs/baseToken.test.ts
 * @brief Unit tests for the BaseToken class and its related codec primitives.
 * Domain: Editor Infrastructure, Tokenization, Text Decoding.
 * Architecture: Employs Mocha test suites and randomized fuzz-style input generation for validation of token lifecycle methods (render, range aggregation, equality).
 * Functional Utility: Ensures structural and semantic integrity of tokens used in VS Code's editor-level text processing pipeline.
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
 * @brief Generates a randomized Range object for fuzz testing.
 * Logic: Randomly selects start/end line and column numbers while maintaining valid temporal/spatial invariants.
 * @throws if maxNumber constraints are violated.
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
 * @brief Utility for generating random token instances for property-based testing.
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

suite('BaseToken', () => {
	// Synchronization: Invariant check to prevent resource leaks during testing.
	ensureNoDisposablesAreLeakedInTestSuite();

	suite('• render', () => {
		/**
		 * Functional Utility: Verifies that a sequence of tokens correctly reconstructs the source string.
		 * Invariant: Token ranges should not affect the final string output during rendering.
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
			assert.strictEqual(
				'',
				BaseToken.render([]),
				`Must correctly render and empty list of tokens.`,
			);
		});
	});

	suite('• fullRange', () => {
		/**
		 * Functional Utility: Validates the calculation of an aggregate range spanning a list of tokens.
		 * Logic: The resulting range must start at the first token and end at the last.
		 */
		suite('• throws', () => {
			test('• if empty list provided', () => {
				assert.throws(() => {
					BaseToken.fullRange([]);
				});
			});

			test('• if start line number of the first token is greater than one of the last token', () => {
				assert.throws(() => {
					const lastToken = randomSimpleToken();

					// Block Logic: Constructing a temporally impossible range (Start > End).
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

	suite('• equals()', () => {
		/**
		 * Functional Utility: Verifies strict equality between token instances.
		 * Invariants: Tokens are considered unequal if they differ in type (constructor), text content, or spatial range.
		 */
		suite('• false', () => {
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
