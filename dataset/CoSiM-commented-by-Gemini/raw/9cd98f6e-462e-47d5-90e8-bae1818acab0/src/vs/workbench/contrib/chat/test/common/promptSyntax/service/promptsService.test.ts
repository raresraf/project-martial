/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file promptsService.test.ts
 * @brief Unit tests for the PromptsService.
 *
 * This file contains tests for the `PromptsService`, which is responsible for
 * managing and providing syntax parsers for text models that may contain
 * prompt syntax (e.g., file references). The tests cover parser lifecycle,
 * caching, automatic updates on model changes, and error handling.
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
 * @brief A helper class to encapsulate the expected properties of a parsed file link
 * for easier and more readable assertions.
 */
class ExpectedLink {
	constructor(
		public readonly uri: URI,
		public readonly fullRange: Range,
		public readonly linkRange: Range,
	) { }

	/**
	 * Asserts that a given prompt file reference matches the expected properties.
	 * @param link The `IPromptFileReference` to validate.
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
 * Asserts that two arrays of prompt file references are equal.
 * @param links The actual links parsed from the model.
 * @param expectedLinks The expected links to compare against.
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

suite('PromptsService', () => {
	const disposables = ensureNoDisposablesAreLeakedInTestSuite();

	let service: IPromptsService;
	let instantiationService: TestInstantiationService;

	/**
	 * Test Setup: Initializes the services required for the PromptsService,
	 * including a mock instantiation service and stubbed dependencies.
	 */
	setup(async () => {
		instantiationService = disposables.add(new TestInstantiationService());
		instantiationService.stub(ILogService, new NullLogService());
		instantiationService.stub(IConfigurationService, new TestConfigurationService());
		instantiationService.stub(IFileService, disposables.add(instantiationService.createInstance(FileService)));

		service = disposables.add(instantiationService.createInstance(PromptsService));
	});

	suite('• getParserFor', () => {
		/**
		 * Test Case: Verifies that the service correctly provides and caches
		 * parser instances for text models. It also tests the lifecycle of
		 * parsers, including their creation, disposal, and re-creation.
		 */
		test('• provides cached parser instance', async () => {
			const langId = 'fooLang';

			/**
			 * Arrange: Create the first text model and get a parser for it.
			 */
			const model1 = disposables.add(createTextModel(
				'test1
	#file:./file.md


   [bin file](/root/tmp.bin)	
',
				langId,
				undefined,
				createURI('/Users/vscode/repos/test/file1.txt'),
			));

			/**
			 * Act & Assert: Get the parser and perform initial validations.
			 */
			const parser1 = service.getSyntaxParserFor(model1);
			assert.strictEqual(
				parser1.uri.toString(),
				model1.uri.toString(),
				'Must create parser1 with the correct URI.',
			);
			assert(!parser1.disposed, 'Parser1 must not be disposed.');
			assert(parser1 instanceof TextModelPromptParser, 'Parser1 must be an instance of TextModelPromptParser.');

			/**
			 * Assert: Validate that the links within the model are correctly parsed.
			 */
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

			/**
			 * Act & Assert: Get the parser for the same model again and verify
			 * that the cached instance is returned.
			 */
			const parser1_1 = service.getSyntaxParserFor(model1);
			assert.strictEqual(parser1, parser1_1, 'Must return the same parser object.');
			assert.strictEqual(parser1_1.uri.toString(), model1.uri.toString(), 'Must create parser1_1 with the correct URI.');

			/**
			 * Arrange: Create a second, different text model.
			 */
			const model2 = disposables.add(createTextModel(
				'some text #file:/absolute/path.txt  	
test-text2',
				langId,
				undefined,
				createURI('/Users/vscode/repos/test/some-folder/file.md'),
			));

			await waitRandom(5);

			/**
			 * Act & Assert: Get a parser for the second model and verify its properties
			 * and that it has not affected the first parser.
			 */
			const parser2 = service.getSyntaxParserFor(model2);
			assert.strictEqual(parser2.uri.toString(), model2.uri.toString(), 'Must create parser2 with the correct URI.');
			assert(!parser2.disposed, 'Parser2 must not be disposed.');
			assert(parser2 instanceof TextModelPromptParser, 'Parser2 must be an instance of TextModelPromptParser.');
			assert(!parser1.disposed, 'Parser1 must not be disposed after creating parser2.');

			/**
			 * Assert: Validate that the links in the second model are correctly parsed.
			 */
			await parser2.settled();
			assert.notStrictEqual(parser1.uri.toString(), parser2.uri.toString(), 'Parser2 must have its own URI.');
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

			/**
			 * Assert: Re-validate the first parser to ensure it was not affected.
			 */
			await parser1_1.settled();
			assertLinks(
				parser1_1.allReferences,
				[
					new ExpectedLink(createURI('/Users/vscode/repos/test/file.md'), new Range(2, 2, 2, 2 + 15), new Range(2, 8, 2, 8 + 9)),
					new ExpectedLink(createURI('/root/tmp.bin'), new Range(5, 4, 5, 4 + 25), new Range(5, 15, 5, 15 + 13)),
				],
			);

			await waitRandom(5);

			/**
			 * Act & Assert: Dispose the first parser and verify the disposed state
			 * while ensuring the second parser remains active.
			 */
			parser1.dispose();
			assert(parser1.disposed, 'Parser1 must be disposed.');
			assert(parser1_1.disposed, 'Cached instance parser1_1 must also be disposed.');
			assert(!parser2.disposed, 'Parser2 must not be affected by parser1 disposal.');

			/**
			 * Act & Assert: Get a parser for the first model again. Confirm that a new,
			 * non-disposed parser object is created.
			 */
			const parser1_2 = service.getSyntaxParserFor(model1);
			assert(!parser1_2.disposed, 'Parser1_2 must not be disposed.');
			assert.notStrictEqual(parser1_2, parser1, 'Must create a new parser object for model1.');
			assert.strictEqual(parser1_2.uri.toString(), model1.uri.toString(), 'Must create parser1_2 with the correct URI.');

			/**
			 * Assert: Validate the links of the newly created parser.
			 */
			await parser1_2.settled();
			assertLinks(
				parser1_2.allReferences,
				[
					new ExpectedLink(createURI('/Users/vscode/repos/test/file.md'), new Range(2, 2, 2, 2 + 15), new Range(2, 8, 2, 8 + 9)),
					new ExpectedLink(createURI('/root/tmp.bin'), new Range(5, 4, 5, 4 + 25), new Range(5, 15, 5, 15 + 13)),
				],
			);

			await waitRandom(5);

			/**
			 * Act & Assert: Dispose the *model* of the second parser and verify that
			 * the parser itself is automatically disposed.
			 */
			model2.dispose();
			assert(parser2.disposed, 'Parser2 must be disposed when its model is disposed.');
			assert(!parser1_2.disposed, 'The other parser (parser1_2) must not be affected.');

			/**
			 * Arrange: Create a new model and parser to replace the disposed one,
			 * this time with different content to test updates.
			 */
			const model2_1 = disposables.add(createTextModel(
				'some text #file:/absolute/path.txt  
 [caption](.copilot/prompts/test.prompt.md)	
	
 more text',
				langId,
				undefined,
				createURI('/Users/vscode/repos/test/some-folder/file.md'),
			));
			const parser2_1 = service.getSyntaxParserFor(model2_1);

			assert(!parser2_1.disposed, 'Parser2_1 must not be disposed.');
			assert.notStrictEqual(parser2_1, parser2, 'Parser2_1 must be a new object.');
			assert.strictEqual(parser2_1.uri.toString(), model2.uri.toString(), 'Must create parser2_1 with the correct URI.');

			/**
			 * Assert: Validate that the new model's contents are parsed correctly.
			 */
			await parser2_1.settled();
			assertLinks(
				parser2_1.allReferences,
				[
					new ExpectedLink(createURI('/absolute/path.txt'), new Range(1, 11, 1, 11 + 24), new Range(1, 17, 1, 17 + 18)),
					new ExpectedLink(createURI('/Users/vscode/repos/test/some-folder/.copilot/prompts/test.prompt.md'), new Range(2, 2, 2, 2 + 42), new Range(2, 12, 2, 12 + 31)),
				],
			);
		});

		/**
		 * Test Case: Verifies that the parser automatically updates its findings
		 * when the underlying text model content changes.
		 */
		test('• auto-updated on model changes', async () => {
			const langId = 'bazLang';

			/**
			 * Arrange: Create a model with initial content and get its parser.
			 */
			const model = disposables.add(createTextModel(
				' 	 #file:../file.md
test1
	
  [another file](/Users/root/tmp/file2.txt)	
',
				langId,
				undefined,
				createURI('/repos/test/file1.txt'),
			));
			const parser = service.getSyntaxParserFor(model);

			await parser.settled();

			/**
			 * Assert: Check initial links.
			 */
			assertLinks(
				parser.allReferences,
				[
					new ExpectedLink(createURI('/repos/file.md'), new Range(1, 4, 1, 4 + 16), new Range(1, 10, 1, 10 + 10)),
					new ExpectedLink(createURI('/Users/root/tmp/file2.txt'), new Range(4, 3, 4, 3 + 41), new Range(4, 18, 4, 18 + 25)),
				],
			);

			/**
			 * Act: Apply an edit to the model, changing one of the links.
			 */
			model.applyEdits([
				{
					range: new Range(4, 18, 4, 18 + 25),
					text: '/Users/root/tmp/file3.txt',
				},
			]);

			await parser.settled();

			/**
			 * Assert: Verify that the parser has updated its links to reflect the change.
			 */
			assertLinks(
				parser.allReferences,
				[
					// The first link should remain unchanged.
					new ExpectedLink(createURI('/repos/file.md'), new Range(1, 4, 1, 4 + 16), new Range(1, 10, 1, 10 + 10)),
					// The second link's URI should be updated.
					new ExpectedLink(createURI('/Users/root/tmp/file3.txt'), new Range(4, 3, 4, 3 + 41), new Range(4, 18, 4, 18 + 25)),
				],
			);
		});

		/**
		 * Test Case: Ensures that attempting to get a parser for a model
		 * that has already been disposed results in an error.
		 */
		test('• throws if disposed model provided', async function () {
			/**
			 * Arrange: Create and immediately dispose of a text model.
			 */
			const model = disposables.add(createTextModel(
				'test1
test2

test3	
',
				'barLang',
				undefined,
				URI.parse('./github/prompts/file.prompt.md'),
			));
			model.dispose();

			/**
			 * Act & Assert: Expect an error when getting the parser.
			 */
			assert.throws(() => {
				service.getSyntaxParserFor(model);
			}, 'Cannot create a prompt parser for a disposed model.');
		});
	});
});
