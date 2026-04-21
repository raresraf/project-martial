/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file promptsService.ts
 * @brief Orchestration service for chat prompt syntax discovery and parser lifecycle management.
 * 
 * Functional Intent: Provides high-level interfaces for identifying prompt files 
 * within the workspace and managing the computational overhead of syntax parsing. 
 * It ensures that each active text model has a dedicated, cached parser instance 
 * to provide real-time syntax highlighting and reference resolution for AI prompts.
 * 
 * Domain: Production Systems, AI-Assisted Development, Prompt Engineering.
 */

import { IPromptsService } from './types.js';
import { URI } from '../../../../../../base/common/uri.js';
import { assert } from '../../../../../../base/common/assert.js';
import { PromptFilesLocator } from '../utils/promptFilesLocator.js';
import { ITextModel } from '../../../../../../editor/common/model.js';
import { Disposable } from '../../../../../../base/common/lifecycle.js';
import { ObjectCache } from '../../../../../../base/common/objectCache.js';
import { TextModelPromptParser } from '../parsers/textModelPromptParser.js';
import { IInstantiationService } from '../../../../../../platform/instantiation/common/instantiation.js';

/**
 * @class PromptsService
 * @brief Singleton service for coordinating prompt discovery and parsing tasks.
 */
export class PromptsService extends Disposable implements IPromptsService {
	declare readonly _serviceBrand: undefined;

	/**
	 * Cache of text model content prompt parsers.
	 * Logic: Prevents redundant re-instantiation of parsers by binding their 
	 * lifecycle to the underlying text models.
	 */
	private readonly cache: ObjectCache<TextModelPromptParser, ITextModel>;

	/**
	 * Prompt files locator utility.
	 */
	private readonly fileLocator = this.initService.createInstance(PromptFilesLocator);

	constructor(
		@IInstantiationService private readonly initService: IInstantiationService,
	) {
		super();

		// Block Logic: Parser factory for the object cache.
		// Invariant: Returns a fully initialized and active parser instance for the provided model.
		this.cache = this._register(
			new ObjectCache((model) => {
				/**
				 * Optimization Note: `seenReferences` tracking is required here to prevent 
				 * infinite recursion during nested prompt file inclusion. Currently 
				 * initialized with an empty set for basic model parsing.
				 */
				const parser: TextModelPromptParser = initService.createInstance(
					TextModelPromptParser,
					model,
					[],
				);

				parser.start();

				// Synchronization: Ensures that the cache contract is met (never returns disposed objects).
				parser.assertNotDisposed(
					'Created prompt parser must not be disposed.',
				);

				return parser;
			})
		);
	}

	/**
	 * getSyntaxParserFor - Retrieves or generates the syntax tree parser for a model.
	 * 
	 * Logic: 
	 * 1. Validates that the target model is still active.
	 * 2. Performs a cache lookup; if missing, triggers the internal factory.
	 * 3. Guarantees a non-disposed return value.
	 * 
	 * @throws {Error} If the model is already disposed or initialization fails.
	 */
	public getSyntaxParserFor(
		model: ITextModel,
	): TextModelPromptParser & { disposed: false } {
		assert(
			!model.isDisposed(),
			'Cannot create a prompt syntax parser for a disposed model.',
		);

		return this.cache.get(model);
	}

	/**
	 * @brief Enumerates all prompt files discovered in the current workspace context.
	 */
	public async listPromptFiles(): Promise<readonly URI[]> {
		return await this.fileLocator.listFiles([]);
	}
}
