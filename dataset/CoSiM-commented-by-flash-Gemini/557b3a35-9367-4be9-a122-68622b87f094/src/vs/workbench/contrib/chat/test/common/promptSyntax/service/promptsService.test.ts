/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file promptsService.test.ts
 * @brief Unit tests for the PromptsService, focusing on parser caching and lifecycle management.
 * 
 * Functional Intent: Ensures that the PromptsService correctly manages syntax parsers for 
 * text models. Validates that parsers are properly cached (returning the same instance 
 * for the same model), correctly track model edits, and are gracefully disposed of 
 * when the underlying text model is destroyed.
 * 
 * Domain: Production Systems, AI-Assisted Development, Automated Testing.
 */

import assert from 'assert';
import { createURI } from '../testUtils/createUri.js';
import { URI } from '../../../../../../../base/common/uri.js';
import { Range } from '../../../../../../../editor/common/core/range.js';
import { assertDefined } from '../../../../../../../base/common/types.js';
import { waitRandom } from '../../../../../../../base/test/common/testUtils.js';
import { IPromptsService } from '../../../../common/promptSyntax/service/types.js';
import { IFileService } from '../../../../../../../platform/files/common/files.js';
import { IPromptFileReference } from '../../../../common/promptSyntax/parsers/types.js';
import { FileService } from '../../../../../../../platform/files/common/fileService.js';
import { createTextModel } from '../../../../../../../editor/test/common/testTextModel.js';
import { ILogService, NullLogService } from '../../../../../../../platform/log/common/log.js';
import { PromptsService } from '../../../../common/promptSyntax/service/promptsService.js';
import { TextModelPromptParser } from '../../../../common/promptSyntax/parsers/textModelPromptParser.js';
import { ensureNoDisposablesAreLeakedInTestSuite } from '../../../../../../../base/test/common/utils.js';
import { IConfigurationService } from '../../../../../../../platform/configuration/common/configuration.js';
import { TestInstantiationService } from '../../../../../../../platform/instantiation/test/common/instantiationServiceMock.js';
import { TestConfigurationService } from '../../../../../../../platform/configuration/test/common/testConfigurationService.js';

/**
 * @class ExpectedLink
 * @brief Helper for validating parsed file references within prompt text.
 */
class ExpectedLink {
	constructor(
		public readonly uri: URI,
		public readonly fullRange: Range,
		public readonly linkRange: Range,
	) { }

	/**
	 * @brief Performs strict structural and semantic equality checks on a parsed link.
	 */
	public assertEqual(link: IPromptFileReference) {
		assert.strictEqual(
			link.type,
			'file',
			'Link must have correct type.',
		);

		assert.strictEqual(
			link.uri.toString(),
			this.uri.toString(),
			'Link must have correct URI.',
		);

		assert(
			this.fullRange.equalsRange(link.range),
			`Full range must be '${this.fullRange}', got '${link.range}'.`,
		);

		assertDefined(
			link.linkRange,
			'Link must have a link range.',
		);

		assert(
			this.linkRange.equalsRange(link.linkRange),
			`Link range must be '${this.linkRange}', got '${link.linkRange}'.`,
		);
	}
}

/**
 * @brief Batch validation utility for multiple parsed references.
 */
const assertLinks = (
	links: readonly IPromptFileReference[],
	expectedLinks: readonly ExpectedLink[],
) => {
	for (let i = 0; i < links.length; i++) {
		try {
			expectedLinks[i].assertEqual(links[i]);
		} catch (error) {
			throw new Error(`link#${i}: ${error}`);
		}
	}

	assert.strictEqual(
		links.length,
		expectedLinks.length,
		`Links count must be correct.`,
	);
};

suite('PromptSyntaxService', () => {
	const disposables = ensureNoDisposablesAreLeakedInTestSuite();

	let service: IPromptsService;
	let instantiationService: TestInstantiationService;

	setup(async () => {
		// Synchronization: Mock service layer setup.
		instantiationService = disposables.add(new TestInstantiationService());
		instantiationService.stub(ILogService, new NullLogService());
		instantiationService.stub(IConfigurationService, new TestConfigurationService());
		instantiationService.stub(IFileService, disposables.add(instantiationService.createInstance(FileService)));

		service = disposables.add(instantiationService.createInstance(PromptsService));
	});

	suite('getParserFor', () => {
		/**
		 * Test: provides cached parser instance.
		 * 
		 * Logic:
		 * 1. Verifies that a parser is created and correctly identifies file links.
		 * 2. Confirms that subsequent requests for the same model return the same object (caching).
		 * 3. Validates that multiple models can have independent parsers concurrently.
		 * 4. Ensures disposal of one parser doesn't affect others.
		 */
		test('provides cached parser instance', async () => {
			const langId = 'fooLang';

			// Block Logic: Model 1 initialization and parsing.
			const model1 = disposables.add(createTextModel(
				'test1\n\t#file:./file.md\n\n\n   [bin file](/root/tmp.bin)\t\n',
				langId,
				undefined,
				createURI('/Users/vscode/repos/test/file1.txt'),
			));

			const parser1 = service.getSyntaxParserFor(model1);
			assert.strictEqual(
				parser1.uri.toString(),
				model1.uri.toString(),
				'Must create parser1 with the correct URI.',
			);

			assert(
				!parser1.disposed,
				'Parser1 must not be disposed.',
			);

			// Synchronization: Wait for async parser resolution.
			await parser1.settled();
			assertLinks(
				parser1.allReferences,
				[
					new ExpectedLink(
						createURI('/Users/vscode/repos/test/file.md'),
						new Range(2, 2, 2, 2 + 15),
						new Range(2, 8, 2, 8 + 9),
					),
					new ExpectedLink(
						createURI('/root/tmp.bin'),
						new Range(5, 4, 5, 4 + 25),
						new Range(5, 15, 5, 15 + 13),
					),
				],
			);

			await waitRandom(5);

			// Block Logic: Cache hit validation.
			const parser1_1 = service.getSyntaxParserFor(model1);
			assert.strictEqual(
				parser1,
				parser1_1,
				'Must return the same parser object.',
			);

			// Block Logic: Parallel model validation.
			const model2 = disposables.add(createTextModel(
				'some text #file:/absolute/path.txt  \t\ntest-text2',
				langId,
				undefined,
				createURI('/Users/vscode/repos/test/some-folder/file.md'),
			));

			await waitRandom(5);

			const parser2 = service.getSyntaxParserFor(model2);

			assert.strictEqual(
				parser2.uri.toString(),
				model2.uri.toString(),
				'Must create parser2 with the correct URI.',
			);

			await parser2.settled();

			assertLinks(
				parser2.allReferences,
				[
					new ExpectedLink(
						createURI('/absolute/path.txt'),
						new Range(1, 11, 1, 11 + 24),
						new Range(1, 17, 1, 17 + 18),
					),
				],
			);

			// Invariant: Concurrent parsers must maintain distinct state.
			await parser1_1.settled();
			assertLinks(
				parser1_1.allReferences,
				[
					new ExpectedLink(
						createURI('/Users/vscode/repos/test/file.md'),
						new Range(2, 2, 2, 2 + 15),
						new Range(2, 8, 2, 8 + 9),
					),
					new ExpectedLink(
						createURI('/root/tmp.bin'),
						new Range(5, 4, 5, 4 + 25),
						new Range(5, 15, 5, 15 + 13),
					),
				],
			);

			// Block Logic: Disposal propagation.
			parser1.dispose();

			assert(
				parser1.disposed,
				'Parser1 must be disposed.',
			);

			assert(
				parser1_1.disposed,
				'Parser1_1 must be disposed.',
			);

			assert(
				!parser2.disposed,
				'Parser2 must not be disposed.',
			);

			// Block Logic: Cache eviction and re-creation.
			const parser1_2 = service.getSyntaxParserFor(model1);

			assert(
				!parser1_2.disposed,
				'Parser1_2 must not be disposed.',
			);

			assert.notStrictEqual(
				parser1_2,
				parser1,
				'Must create a new parser object after disposal.',
			);

			// Block Logic: Model-driven disposal.
			model2.dispose();

			assert(
				parser2.disposed,
				'Parser must be disposed when the underlying model is destroyed.',
			);

			const model2_1 = disposables.add(createTextModel(
				'some text #file:/absolute/path.txt  \n [caption](.copilot/prompts/test.prompt.md)\t\n\t\n more text',
				langId,
				undefined,
				createURI('/Users/vscode/repos/test/some-folder/file.md'),
			));
			const parser2_1 = service.getSyntaxParserFor(model2_1);

			await parser2_1.settled();

			assertLinks(
				parser2_1.allReferences,
				[
					new ExpectedLink(
						createURI('/absolute/path.txt'),
						new Range(1, 11, 1, 11 + 24),
						new Range(1, 17, 1, 17 + 18),
					),
					new ExpectedLink(
						createURI('/Users/vscode/repos/test/some-folder/.copilot/prompts/test.prompt.md'),
						new Range(2, 2, 2, 2 + 42),
						new Range(2, 12, 2, 12 + 31),
					),
				],
			);
		});

		/**
		 * Test: auto-updated on model changes.
		 * 
		 * Logic: Verifies that the parser re-evaluates the syntax tree when 
		 * edits are applied to the text model.
		 */
		test('auto-updated on model changes', async () => {
			const langId = 'bazLang';

			const model = disposables.add(createTextModel(
				' \t #file:../file.md\ntest1\n\t\n  [another file](/Users/root/tmp/file2.txt)\t\n',
				langId,
				undefined,
				createURI('/repos/test/file1.txt'),
			));

			const parser = service.getSyntaxParserFor(model);

			await parser.settled();

			assertLinks(
				parser.allReferences,
				[
					new ExpectedLink(
						createURI('/repos/file.md'),
						new Range(1, 4, 1, 4 + 16),
						new Range(1, 10, 1, 10 + 10),
					),
					new ExpectedLink(
						createURI('/Users/root/tmp/file2.txt'),
						new Range(4, 3, 4, 3 + 41),
						new Range(4, 18, 4, 18 + 25),
					),
				],
			);

			// Block Logic: Incremental update injection.
			model.applyEdits([
				{
					range: new Range(4, 18, 4, 18 + 25),
					text: '/Users/root/tmp/file3.txt',
				},
			]);

			// Synchronization: Re-wait for parser convergence.
			await parser.settled();

			assertLinks(
				parser.allReferences,
				[
					new ExpectedLink(
						createURI('/repos/file.md'),
						new Range(1, 4, 1, 4 + 16),
						new Range(1, 10, 1, 10 + 10),
					),
					new ExpectedLink(
						createURI('/Users/root/tmp/file3.txt'),
						new Range(4, 3, 4, 3 + 41),
						new Range(4, 18, 4, 18 + 25),
					),
				],
			);
		});

		/**
		 * Test: throws if disposed model provided.
		 * 
		 * Logic: Defensive check for invalid API usage.
		 */
		test('throws if disposed model provided', async function () {
			const model = disposables.add(createTextModel(
				'test1\ntest2\n\ntest3\t\n',
				'barLang',
				undefined,
				URI.parse('./github/prompts/file.prompt.md'),
			));

			model.dispose();

			assert.throws(() => {
				service.getSyntaxParserFor(model);
			}, 'Cannot create a prompt parser for a disposed model.');
		});
	});
});
