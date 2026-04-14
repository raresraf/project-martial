/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file promptsService.ts
 * @brief This file defines the `PromptsService`, which is responsible for managing,
 * parsing, and locating prompt files (e.g., `.prompt`, `.instructions`) within the
 * workspace and user profiles. It provides functionalities for parsing prompt
 * syntax, resolving slash commands, and aggregating metadata from these files.
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
 * @brief Provides services for handling chat prompts, including locating prompt files,
 * parsing their syntax, and managing their lifecycle.
 *
 * This service acts as a central hub for prompt-related functionalities, caching
 * parsers for performance and providing utilities to resolve prompt commands and
f* ind relevant instruction files based on context.
 */
export class PromptsService extends Disposable implements IPromptsService {
	public declare readonly _serviceBrand: undefined;

	/**
	 * @property
	 * @description A cache for `TextModelPromptParser` instances, keyed by `ITextModel`.
	 * This prevents re-parsing of unchanged models, improving performance.
	 */
	private readonly cache: ObjectCache<TextModelPromptParser, ITextModel>;

	/**
	 * @property
	 * @description A utility class to locate prompt files in various locations
	 * (user profiles, workspace, etc.).
	 */
	private readonly fileLocator: PromptFilesLocator;

	/**
	 * @property
	 * @description A logging function assigned from the ILogService, used by the
	 * `@logTime` decorator to measure and log method execution times for debugging.
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

		/**
		 * Functional Utility: The ObjectCache is initialized with a factory function.
		 * This function is invoked only when a parser for a given text model is
		 * not already in the cache. It creates, starts, and returns a new parser.
		 */
		this.cache = this._register(
			new ObjectCache((model) => {
				assert(
					model.isDisposed() === false,
					'Text model must not be disposed.',
				);

				const parser: TextModelPromptParser = initService.createInstance(
					TextModelPromptParser,
					model,
					{ seenReferences: [] },
				).start();

				parser.assertNotDisposed(
					'Created prompt parser must not be disposed.',
				);

				return parser;
			})
		);
	}

	/**
	 * Retrieves a syntax parser for a given text model.
	 * It uses a cache to avoid creating new parsers for the same model.
	 * @param model The `ITextModel` to be parsed.
	 * @returns A non-disposed `TextModelPromptParser` instance for the given model.
	 * @throws {Error} If the provided model is already disposed.
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
	 * Lists all prompt files of a specific type ('prompt' or 'instructions') found
	 * within user and workspace locations.
	 * @param type The type of prompt files to list.
	 * @returns A promise that resolves to a readonly array of `IPromptPath` objects.
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

	/**
	 * Gets the source folders where prompt files of a given type are located.
	 * @param type The type of prompt.
	 * @returns A readonly array of `IPromptPath` objects representing source folders.
	 */
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

	/**
	 * Converts a string into a chat prompt slash command object if it matches the valid format.
	 * @param command The string to convert.
	 * @returns An `IChatPromptSlashCommand` object or undefined if the format is invalid.
	 */
	public asPromptSlashCommand(command: string): IChatPromptSlashCommand | undefined {
		if (command.match(/^[\w_\-\.]+/)) {
			return { command, detail: localize('prompt.file.detail', 'Prompt file: {0}', command) };
		}
		return undefined;
	}

	/**
	 * Resolves a slash command to its corresponding prompt file path.
	 * @param data The slash command data to resolve.
	 * @returns A promise that resolves to the `IPromptPath` of the file, or undefined if not found.
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

	/**
	 * Finds and lists all available prompt slash commands.
	 * @returns A promise that resolves to an array of `IChatPromptSlashCommand` objects.
	 */
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
	 * Finds all instruction files that apply to a given set of files.
	 * @param files An array of file URIs to check against.
	 * @returns A promise that resolves to a deduplicated array of URIs for the applicable instruction files.
	 *
	 * Algorithm: This method retrieves all `.instructions` files, parses their metadata,
	 * and checks if the `applyTo` glob pattern in the metadata matches any of the
	 * provided file URIs.
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

		// Block Logic: Iterate through each instruction file's metadata tree.
		for (const instruction of instructions.flatMap(flatten)) {
			const { metadata, uri } = instruction;
			const { applyTo } = metadata;

			if (applyTo === undefined) {
				continue;
			}

			// Pre-condition: Handle special wildcard patterns that apply globally.
			if ((applyTo === '**') || (applyTo === '**/*')) {
				result.push(uri);
				continue;
			}

			// Block Logic: Match each file against the glob pattern.
			for (const file of files) {
				if (match(applyTo, file.fsPath)) {
					result.push(uri);
				}
			}
		}

		// Return a unique set of instruction file URIs.
		return [...new ResourceSet(result)];
	}

	/**
	 * Retrieves the hierarchical metadata for a given list of prompt file URIs.
	 * @param promptUris An array of prompt file URIs.
	 * @returns A promise that resolves to an array of `IMetadata` trees.
	 */
	@logTime()
	public async getAllMetadata(
		promptUris: readonly URI[],
	): Promise<IMetadata[]> {
		const metadata = await Promise.all(
			promptUris.map(async (uri) => {
				let parser: PromptParser | undefined;
				try {
					// Block Logic: For each URI, create a temporary parser to extract metadata.
					// The parser is disposed of immediately after use.
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
	 * Parses and combines metadata from multiple prompt files, determining the
	 * resulting chat mode and aggregating all specified tools.
	 * @param promptUris An array of prompt file URIs to process.
	 * @returns A promise that resolves to a `TCombinedToolsMetadata` object, or null
	 * if no URIs are provided.
	 */
	@logTime()
	public async getCombinedToolsMetadata(
		promptUris: readonly URI[],
	): Promise<TCombinedToolsMetadata | null> {
		if (promptUris.length === 0) {
			return null;
		}

		const filesMetadata = await this.getAllMetadata(promptUris);

		// Block Logic: Process metadata from each file to determine chat mode and collect tools.
		const allTools = filesMetadata
			.map((fileMetadata) => {
				const result: string[] = [];

				let isFirst = true;
				let isRootInAgentMode = false;
				let hasTools = false;
				let chatMode: ChatMode | undefined;

				// Invariant: Traverses the metadata tree of a single file.
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

					// Block Logic: The more privileged chat mode wins (Agent > Edit > Ask).
					if (chatMode && mode) {
						chatMode = morePrivilegedChatMode(chatMode, mode);
					}

					if (isRootInAgentMode && tools !== undefined) {
						result.push(...tools);
						hasTools = true;
					}

					return false;
				}, fileMetadata);

				if (chatMode === ChatMode.Agent) {
					return { tools: (hasTools) ? [...new Set(result)] : undefined, mode: ChatMode.Agent };
				}

				return { mode: chatMode };
			});

		// Block Logic: Aggregate results from all processed files.
		let hasAnyTools = false;
		let resultingChatMode: ChatMode | undefined;

		const result: string[] = [];
		for (const { tools, mode } of allTools) {
			resultingChatMode ??= mode;

			if (resultingChatMode && mode) {
				resultingChatMode = morePrivilegedChatMode(resultingChatMode, mode);
			}

			if (tools) {
				result.push(...tools);
				hasAnyTools = true;
			}
		}

		if (resultingChatMode === ChatMode.Agent) {
			return { tools: (hasAnyTools) ? [...new Set(result)] : undefined, mode: resultingChatMode };
		}

		return { tools: undefined, mode: resultingChatMode };
	}
}

/**
 * Determines which of two chat modes is more privileged.
 * The order of privilege is: Agent > Edit > Ask.
 * @param chatMode1 The first chat mode.
 * @param chatMode2 The second chat mode.
 * @returns The more privileged `ChatMode`.
 */
const morePrivilegedChatMode = (
	chatMode1: ChatMode,
	chatMode2: ChatMode,
): ChatMode => {
	// when modes equal, return one of them
	if (chatMode1 === chatMode2) {
		return chatMode1;
	}

	// when modes are different but one of them is 'agent', use 'agent'
	if ((chatMode1 === ChatMode.Agent) || (chatMode2 === ChatMode.Agent)) {
		return ChatMode.Agent;
	}

	// when modes are different, none of them is 'agent', but one of them
	// is 'edit', use 'edit'
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
 * Recursively collects metadata from a prompt file reference and its children
 * into a hierarchical tree structure.
 * @param reference The root `IPromptFileReference` to start from.
 * @returns An `IMetadata` object representing the tree of metadata.
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

/**
 * Extracts the command name from a prompt file path.
 * (e.g., "/path/to/my-prompt.prompt.md" -> "my-prompt").
 * @param path The file path.
 * @returns The extracted command name.
 */
export function getPromptCommandName(path: string): string {
	const name = basename(path, PROMPT_FILE_EXTENSION);
	return name;
}

/**
 * A higher-order function that returns a function to create an `IPromptPath`
 * object from a URI, adding the specified storage and type.
 * @param storage The storage location ('user' or 'local').
 * @param type The prompt type ('prompt' or 'instructions').
 * @returns A function that takes a URI and returns an `IPromptPath`.
 */
const addType = (
	storage: TPromptsStorage,
	type: TPromptsType,
): (uri: URI) => IPromptPath => {
	return (uri) => {
		return { uri, storage, type };
	};
};

/**
 * A higher-order function that returns a function to map an array of URIs to
 * an array of `IPromptPath` objects with a specified storage and type.
 * @param storage The storage location ('user' or 'local').
 * @param type The prompt type ('prompt' or 'instructions').
 * @returns A function that takes an array of URIs and returns an array of `IPromptPath`.
 */
const withType = (
	storage: TPromptsStorage,
	type: TPromptsType,
): (uris: readonly URI[]) => (readonly IPromptPath[]) => {
	return (uris) => {
		return uris
			.map(addType(storage, type));
	};
};
