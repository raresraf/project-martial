/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { IPrompt, IPromptsService } from './types.js';
import { assert } from '../../../../../../base/common/assert.js';
import { PromptFilesLocator } from '../utils/promptFilesLocator.js';
import { ITextModel } from '../../../../../../editor/common/model.js';
import { Disposable } from '../../../../../../base/common/lifecycle.js';
import { ObjectCache } from '../../../../../../base/common/objectCache.js';
import { TextModelPromptParser } from '../parsers/textModelPromptParser.js';
import { IInstantiationService } from '../../../../../../platform/instantiation/common/instantiation.js';

/**
 * Implements the {@link IPromptsService} interface, providing services for
 * discovering and parsing chat prompts within the workspace.
 */
export class PromptsService extends Disposable implements IPromptsService {
	declare readonly _serviceBrand: undefined;

	/**
	 * Cache of text model content prompt parsers.
	 * This cache ensures that only one parser is created per text model, optimizing
	 * resource usage. The parser is created on-demand when first requested.
	 */
	private readonly cache: ObjectCache<TextModelPromptParser, ITextModel>;

	/**
	 * A utility class instance for locating prompt files in the filesystem.
	 */
	private readonly fileLocator = this.initService.createInstance(PromptFilesLocator);

	constructor(
		@IInstantiationService private readonly initService: IInstantiationService,
	) {
		super();

		// Block Logic: Initialize the ObjectCache for parsers.
		// The factory function is called by the cache when a parser for a new model
		// is requested. It creates a new parser, starts it, and returns it.
		this.cache = this._register(
			new ObjectCache((model) => {
				/**
				 * Note! When/if shared with "file" prompts, the `seenReferences` array below must be taken into account.
				 * Otherwise consumers will either see incorrect failing or incorrect successful results, based on their
				 * use case, timing of their calls to the {@link getSyntaxParserFor} function, and state of this service.
				 */
				const parser: TextModelPromptParser = initService.createInstance(
					TextModelPromptParser,
					model,
					[],
				);

				// Kicks off the parsing process for the model.
				parser.start();

				// this is a sanity check and the contract of the object cache,
				// we must return a non-disposed object from this factory function
				parser.assertNotDisposed(
					'Created prompt parser must not be disposed.',
				);

				return parser;
			})
		);
	}

	/**
	 * Retrieves a cached or creates a new syntax parser for the given text model.
	 * @param model The text model for which to get the parser.
	 * @returns An active and non-disposed {@link TextModelPromptParser}.
	 *
	 * @throws {Error} if:
	 * 	- the provided model is disposed
	 * 	- a newly created parser is disposed immediately on initialization.
	 * 	  See factory function in the {@link constructor} for more info.
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
	 * Discovers all `.prompt` files within the workspace.
	 * @returns A promise that resolves to an array of {@link IPrompt} objects.
	 */
	public async listPromptFiles(): Promise<readonly IPrompt[]> {
		const promptFiles = await this.fileLocator.listFiles([]);

		return promptFiles.map((uri) => {
			return {
				uri,
				// right now all prompts are coming from the local disk
				source: 'local',
			};
		});
	}
}
