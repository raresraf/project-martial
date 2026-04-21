/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file promptsService.ts
 * @brief Orchestration service for chat prompt syntax parsing, file management, and metadata aggregation.
 * 
 * Functional Intent: Provides a centralized registry and processing engine for chat prompts. 
 * It manages the discovery of prompt files across user profiles and local workspaces, 
 * handles hierarchical metadata resolution for nested prompt references, and 
 * coordinates the identification of required AI tools based on active chat modes.
 * 
 * Domain: Production Systems, AI-Assisted Development, Prompt Engineering.
 */

import { ChatMode } from '../../constants.js';
import { localize } from '../../../../../../nls.js';
import { PROMPT_LANGUAGE_ID } from '../constants.js';
import { flatten, forEach } from '../utils/treeUtils.js';
import { PromptParser } from '../parsers/promptParser.js';
import { match } from '../../../../../../base/common/glob.js';
import { pick } from '../../../../../../base/common/arrays.js';
import { type URI } from '../../../../../../base/common/uri.js';
import { type IPromptFileReference } from '../parsers/types.js';
import { assert } from '../../../../../../base/common/assert.js';
import { basename } from '../../../../../../base/common/path.js';
import { ResourceSet } from '../../../../../../base/common/map.js';
import { PromptFilesLocator } from '../utils/promptFilesLocator.js';
import { Disposable } from '../../../../../../base/common/lifecycle.js';
import { type ITextModel } from '../../../../../../editor/common/model.js';
import { ObjectCache } from '../../../../../../base/common/objectCache.js';
import { ILogService } from '../../../../../../platform/log/common/log.js';
import { TextModelPromptParser } from '../parsers/textModelPromptParser.js';
import { ILabelService } from '../../../../../../platform/label/common/label.js';
import { IModelService } from '../../../../../../editor/common/services/model.js';
import { logTime, TLogFunction } from '../../../../../../base/common/decorators/logTime.js';
import { PROMPT_FILE_EXTENSION } from '../../../../../../platform/prompts/common/constants.js';
import { IInstantiationService } from '../../../../../../platform/instantiation/common/instantiation.js';
import { IUserDataProfileService } from '../../../../../services/userDataProfile/common/userDataProfile.js';
import type { IChatPromptSlashCommand, TCombinedToolsMetadata, IMetadata, IPromptPath, IPromptsService, TPromptsStorage, TPromptsType } from './types.js';

/**
 * @class PromptsService
 * @brief Singleton service for managing the lifecycle and discovery of chat prompts.
 */
export class PromptsService extends Disposable implements IPromptsService {
	public declare readonly _serviceBrand: undefined;

	/**
	 * Cache of text model content prompt parsers to minimize redundant re-parsing.
	 */
	private readonly cache: ObjectCache<TextModelPromptParser, ITextModel>;

	/**
	 * Prompt files locator utility.
	 */
	private readonly fileLocator: PromptFilesLocator;

	/**
	 * Function used by the `@logTime` decorator to log
	 * execution time of some of the decorated methods.
	 */
	public logTime: TLogFunction;

	constructor(
		@ILogService public readonly logger: ILogService,
		@ILabelService private readonly labelService: ILabelService,
		@IModelService private readonly modelService: IModelService,
		@IInstantiationService private readonly initService: IInstantiationService,
		@IUserDataProfileService private readonly userDataService: IUserDataProfileService,
	) {
		super();

		this.fileLocator = this.initService.createInstance(PromptFilesLocator);
		this.logTime = this.logger.trace.bind(this.logger);

		// Block Logic: Factory for syntax parsers.
		// Invariant: Maintains exactly one active parser per open text model.
		this.cache = this._register(
			new ObjectCache((model) => {
				assert(
					model.isDisposed() === false,
					'Text model must not be disposed.',
				);

				/**
				 * Optimization Note: `seenReferences` tracking is required here to prevent 
				 * infinite recursion during nested prompt file inclusion.
				 */
				const parser: TextModelPromptParser = initService.createInstance(
					TextModelPromptParser,
					model,
					{ seenReferences: [] },
				).start();

				// Synchronization: Ensures that the cache contract is met (never returns disposed objects).
				parser.assertNotDisposed(
					'Created prompt parser must not be disposed.',
				);

				return parser;
			})
		);
	}

	/**
	 * getSyntaxParserFor - Retrieves or creates a syntax parser for the provided text model.
	 * 
	 * @throws {Error} if the provided model is disposed or parser initialization fails.
	 */
	public getSyntaxParserFor(
		model: ITextModel,
	): TextModelPromptParser & { disposed: false } {
		assert(
			model.isDisposed() === false,
			'Cannot create a prompt syntax parser for a disposed model.',
		);

		return this.cache.get(model);
	}

	/**
	 * listPromptFiles - Enumerates all available prompt files from user and local storage.
	 * 
	 * Algorithm: Parallel discovery across storage backends.
	 */
	public async listPromptFiles(type: TPromptsType): Promise<readonly IPromptPath[]> {
		const userLocations = [this.userDataService.currentProfile.promptsHome];

		const prompts = await Promise.all([
			this.fileLocator.listFilesIn(userLocations, type)
				.then(withType('user', type)),
			this.fileLocator.listFiles(type)
				.then(withType('local', type)),
		]);

		return prompts.flat();
	}

	public getSourceFolders(type: TPromptsType): readonly IPromptPath[] {
		assert(
			type === 'prompt' || type === 'instructions',
			`Unknown prompt type '${type}'.`,
		);

		const result: IPromptPath[] = [];

		for (const uri of this.fileLocator.getConfigBasedSourceFolders(type)) {
			result.push({ uri, storage: 'local', type });
		}
		const userHome = this.userDataService.currentProfile.promptsHome;
		result.push({ uri: userHome, storage: 'user', type });

		return result;
	}

	public asPromptSlashCommand(command: string): IChatPromptSlashCommand | undefined {
		if (command.match(/^[\w_\-\.]+/)) {
			return { command, detail: localize('prompt.file.detail', 'Prompt file: {0}', command) };
		}
		return undefined;
	}

	/**
	 * resolvePromptSlashCommand - Maps a slash command back to its physical prompt resource.
	 * 
	 * Logic: Searches active list of prompt files or open text models to resolve the command context.
	 */
	public async resolvePromptSlashCommand(data: IChatPromptSlashCommand): Promise<IPromptPath | undefined> {
		if (data.promptPath) {
			return data.promptPath;
		}
		const files = await this.listPromptFiles('prompt');
		const command = data.command;
		const result = files.find(file => getPromptCommandName(file.uri.path) === command);
		if (result) {
			return result;
		}
		const textModel = this.modelService.getModels().find(model => model.getLanguageId() === PROMPT_LANGUAGE_ID && getPromptCommandName(model.uri.path) === command);
		if (textModel) {
			return { uri: textModel.uri, storage: 'local', type: 'prompt' };
		}
		return undefined;
	}

	public async findPromptSlashCommands(): Promise<IChatPromptSlashCommand[]> {
		const promptFiles = await this.listPromptFiles('prompt');
		return promptFiles.map(promptPath => {
			const command = getPromptCommandName(promptPath.uri.path);
			return {
				command,
				detail: localize('prompt.file.detail', 'Prompt file: {0}', this.labelService.getUriLabel(promptPath.uri, { relative: true })),
				promptPath
			};
		});
	}

	/**
	 * findInstructionFilesFor - Identifies relevant instruction files for a set of target files.
	 * 
	 * Algorithm: Glob-based metadata matching.
	 * Logic: Scans instruction prompts and filters those whose 'applyTo' policy covers the input file set.
	 */
	@logTime()
	public async findInstructionFilesFor(
		files: readonly URI[],
	): Promise<readonly URI[]> {
		const result: URI[] = [];

		const instructionFiles = await this.listPromptFiles('instructions');
		if (instructionFiles.length === 0) {
			return result;
		}

		const instructions = await this.getAllMetadata(
			instructionFiles.map(pick('uri')),
		);

		/**
		 * Block Logic: Policy evaluation loop.
		 * Invariant: Aggregates instructions whose scope covers the current editor context.
		 */
		for (const instruction of instructions.flatMap(flatten)) {
			const { metadata, uri } = instruction;
			const { applyTo } = metadata;

			if (applyTo === undefined) {
				continue;
			}

			// Optimization: Wildcard matching for global-scope instructions.
			if ((applyTo === '**') || (applyTo === '**/*')) {
				result.push(uri);

				continue;
			}

			for (const file of files) {
				if (match(applyTo, file.fsPath)) {
					result.push(uri);
				}
			}
		}

		return [...new ResourceSet(result)];
	}

	/**
	 * getAllMetadata - Batch parsing of prompt URIs into structured metadata trees.
	 */
	@logTime()
	public async getAllMetadata(
		promptUris: readonly URI[],
	): Promise<IMetadata[]> {
		const metadata = await Promise.all(
			promptUris.map(async (uri) => {
				let parser: PromptParser | undefined;
				try {
					parser = this.initService.createInstance(
						PromptParser,
						uri,
						{ allowNonPromptFiles: true },
					).start();

					await parser.allSettled();

					return collectMetadata(parser);
				} finally {
					parser?.dispose();
				}
			}),
		);

		return metadata;
	}

	/**
	 * getCombinedToolsMetadata - Aggregates required tools and privileges across multiple prompts.
	 * 
	 * Algorithm: Hierarchical capability reduction.
	 * Logic: 
	 * 1. Flattens nested metadata trees from multiple source prompts.
	 * 2. Resolves conflicting chat modes by picking the most privileged (Agent > Edit > Ask).
	 * 3. Deduplicates and merges tool lists required by the aggregate state.
	 */
	@logTime()
	public async getCombinedToolsMetadata(
		promptUris: readonly URI[],
	): Promise<TCombinedToolsMetadata | null> {
		if (promptUris.length === 0) {
			return null;
		}

		const filesMetadata = await this.getAllMetadata(promptUris);

		const allTools = filesMetadata
			.map((fileMetadata) => {
				const result: string[] = [];

				let isFirst = true;
				let isRootInAgentMode = false;
				let hasTools = false;

				let chatMode: ChatMode | undefined;

				// Block Logic: Capability discovery traversal.
				forEach((node) => {
					const { metadata } = node;
					const { mode, tools } = metadata;

					if (isFirst === true) {
						isFirst = false;

						if ((mode === ChatMode.Agent) || (tools !== undefined)) {
							isRootInAgentMode = true;

							chatMode = ChatMode.Agent;
						}
					}

					chatMode ??= mode;

					if (chatMode && mode) {
						chatMode = morePrivilegedChatMode(
							chatMode,
							mode,
						);
					}

					if (isRootInAgentMode && tools !== undefined) {
						result.push(...tools);
						hasTools = true;
					}

					return false;
				}, fileMetadata);

				if (chatMode === ChatMode.Agent) {
					return {
						tools: (hasTools)
							? [...new Set(result)]
							: undefined,
						mode: ChatMode.Agent,
					};
				}

				return {
					mode: chatMode,
				};
			});

		let hasAnyTools = false;
		let resultingChatMode: ChatMode | undefined;

		const result: string[] = [];
		/**
		 * Block Logic: Final privilege consolidation.
		 */
		for (const { tools, mode } of allTools) {
			resultingChatMode ??= mode;

			if (resultingChatMode && mode) {
				resultingChatMode = morePrivilegedChatMode(
					resultingChatMode,
					mode,
				);
			}

			if (tools) {
				result.push(...tools);
				hasAnyTools = true;
			}
		}

		if (resultingChatMode === ChatMode.Agent) {
			return {
				tools: (hasAnyTools)
					? [...new Set(result)]
					: undefined,
				mode: resultingChatMode,
			};
		}

		return {
			tools: undefined,
			mode: resultingChatMode,
		};
	}
}

/**
 * morePrivilegedChatMode - Implements strict ordering of Chat capability levels.
 */
const morePrivilegedChatMode = (
	chatMode1: ChatMode,
	chatMode2: ChatMode,
): ChatMode => {
	if (chatMode1 === chatMode2) {
		return chatMode1;
	}

	// Logic: 'Agent' mode subsumes all other capabilities.
	if ((chatMode1 === ChatMode.Agent) || (chatMode2 === ChatMode.Agent)) {
		return ChatMode.Agent;
	}

	// Logic: 'Edit' mode is prioritized over 'Ask'.
	if ((chatMode1 === ChatMode.Edit) || (chatMode2 === ChatMode.Edit)) {
		return ChatMode.Edit;
	}

	throw new Error(
		[
			'Invalid logic encountered: ',
			`at this point modes '${chatMode1}' and '${chatMode2}' are different, but`,
			`both must have be equal to '${ChatMode.Ask}' at the same time.`,
		].join(' '),
	);
};

/**
 * collectMetadata - Recursively transforms a file-reference graph into a metadata tree.
 */
const collectMetadata = (
	reference: Pick<IPromptFileReference, 'uri' | 'metadata' | 'references'>,
): IMetadata => {
	const childMetadata = [];
	for (const child of reference.references) {
		if (child.errorCondition !== undefined) {
			continue;
		}

		childMetadata.push(collectMetadata(child));
	}

	const children = (childMetadata.length > 0)
		? childMetadata
		: undefined;

	return {
		uri: reference.uri,
		metadata: reference.metadata,
		children,
	};
};

export function getPromptCommandName(path: string): string {
	const name = basename(path, PROMPT_FILE_EXTENSION);
	return name;
}

const addType = (
	storage: TPromptsStorage,
	type: TPromptsType,
): (uri: URI) => IPromptPath => {
	return (uri) => {
		return { uri, storage, type };
	};
};

const withType = (
	storage: TPromptsStorage,
	type: TPromptsType,
): (uris: readonly URI[]) => (readonly IPromptPath[]) => {
	return (uris) => {
		return uris
			.map(addType(storage, type));
	};
};
