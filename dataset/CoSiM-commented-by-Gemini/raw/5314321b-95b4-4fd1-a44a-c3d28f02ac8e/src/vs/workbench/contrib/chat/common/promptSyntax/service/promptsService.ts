/**
 * @file This file defines the `PromptsService`, a core component in the chat feature
 * responsible for locating, parsing, and managing chat prompts and instructions
 * defined in `.prompt` files. It bridges the gap between the raw prompt files on disk
 * and the structured data needed by the chat system to execute prompts with variables,
 * tools, and context-awareness.
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
 * Manages the discovery, parsing, and retrieval of chat prompts defined in `.prompt` files.
 * This service is responsible for locating prompt files from various sources (user, workspace, extensions),
 * parsing their content to extract metadata and tool definitions, and providing this information
 * to the chat system. It also handles the resolution of slash commands to specific prompt files.
 */
export class PromptsService extends Disposable implements IPromptsService {
	public declare readonly _serviceBrand: undefined;

	/**
	 * A cache for `TextModelPromptParser` instances, keyed by their `ITextModel`.
	 * This improves performance by avoiding re-parsing of prompt files that are open in the editor.
	 * The cache ensures that for a given text model, only one active parser exists.
	 */
	private readonly cache: ObjectCache<TextModelPromptParser, ITextModel>;

	/**
	 * Utility for finding `.prompt` files within the workspace, user profiles, and extensions.
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

		// The factory function creates a new parser for a given model.
		// It ensures that we don't create parsers for disposed models and that the
		// created parser is itself not disposed, maintaining the contract of the ObjectCache.
		this.cache = this._register(
			new ObjectCache((model) => {
				assert(
					model.isDisposed() === false,
					'Text model must not be disposed.',
				);

				/**
				 * Note! When/if shared with "file" prompts, the `seenReferences` array below must be taken into account.
				 * Otherwise consumers will either see incorrect failing or incorrect successful results, based on their
				 * use case, timing of their calls to the {@link getSyntaxParserFor} function, and state of this service.
				 */
				const parser: TextModelPromptParser = initService.createInstance(
					TextModelPromptParser,
					model,
					{ seenReferences: [] },
				).start();

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
	 * Retrieves a `TextModelPromptParser` for a given `ITextModel`.
	 * The parser is retrieved from a cache if it already exists for the model,
	 * otherwise a new one is created and cached. This is efficient for prompt
	 * files that are open in an editor.
	 *
	 * @param model The text model for which to get the parser.
	 * @returns A non-disposed `TextModelPromptParser` instance.
	 * @throws {Error} if the provided model is disposed.
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
	 * Searches for and lists all `.prompt` files of a specific type (e.g., 'prompt' or 'instructions')
	 * available in the user's profile and the current workspace.
	 * @param type The type of prompt files to list.
	 * @returns A promise that resolves to an array of prompt paths.
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
	 * This includes configured workspace folders and the user's profile prompt home.
	 * @param type The type of prompt ('prompt' or 'instructions').
	 * @returns An array of prompt paths representing the source folders.
	 */
	public getSourceFolders(type: TPromptsType): readonly IPromptPath[] {
		// sanity check to make sure we don't miss a new
		// prompt type that could be added in the future
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
	 * Converts a string into a slash command object if it matches the valid format.
	 * @param command The string to convert.
	 * @returns An `IChatPromptSlashCommand` object, or `undefined` if the string is not a valid command.
	 */
	public asPromptSlashCommand(command: string): IChatPromptSlashCommand | undefined {
		if (command.match(/^[\w_\-\.]+/)) {
			return { command, detail: localize('prompt.file.detail', 'Prompt file: {0}', command) };
		}
		return undefined;
	}

	/**
	 * Resolves a slash command to its corresponding prompt file path.
	 * It searches through file-based prompts and currently open prompt files in text models.
	 * @param data The slash command data to resolve.
	 * @returns A promise that resolves to the `IPromptPath` of the found prompt file, or `undefined`.
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
	 * Finds all available prompt files and returns them as a list of slash commands.
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
	 * Finds all "instructions" prompt files that apply to a given set of file URIs.
	 * An instruction file is considered applicable if its `applyTo` metadata field,
	 * which is a glob pattern, matches any of the provided file URIs.
	 * @param files The list of file URIs to check against.
	 * @returns A promise that resolves to a readonly array of URIs for the applicable instruction files.
	 */
	@logTime()
	public async findInstructionFilesFor(
		files: readonly URI[],
	): Promise<readonly URI[]> {
		const instructionFiles = await this.listPromptFiles('instructions');
		if (instructionFiles.length === 0) {
			return [];
		}

		const instructions = await this.getAllMetadata(
			instructionFiles.map(pick('uri')),
		);

		const foundFiles = new ResourceSet();
		for (const instruction of instructions.flatMap(flatten)) {
			const { metadata, uri } = instruction;
			const { applyTo } = metadata;

			if (applyTo === undefined) {
				continue;
			}

			// if glob pattern is one of the special wildcard values,
			// add the instructions file event if no files are attached
			if ((applyTo === '**') || (applyTo === '**/*')) {
				foundFiles.add(uri);

				continue;
			}

			// match each attached file with each glob pattern and
			// add the instructions file if its rule matches the file
			for (const file of files) {
				if (match(applyTo, file.fsPath)) {
					foundFiles.add(uri);
				}
			}
		}

		return [...foundFiles];
	}

	/**
	 * Parses a list of prompt URIs and retrieves all their metadata.
	 * This method creates a temporary parser for each URI to extract its metadata tree.
	 * @param promptUris The URIs of the prompt files to parse.
	 * @returns A promise that resolves to an array of metadata trees.
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
	 * Aggregates metadata about tools and chat modes from a list of prompt files.
	 * It resolves conflicts by selecting the most privileged chat mode (`Agent` > `Edit` > `Ask`)
	 * and combines all unique tool definitions.
	 * @param promptUris The URIs of the prompt files to analyze.
	 * @returns A promise that resolves to the combined metadata, or `null` if no prompts were provided.
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

					// if both chat modes are set, pick the more privileged one
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
		for (const { tools, mode } of allTools) {
			resultingChatMode ??= mode;

			// if both chat modes are set, pick the more privileged one
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
 * Determines the more privileged chat mode between two given modes.
 * The order of privilege is `Agent` > `Edit` > `Ask`.
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
 * into a hierarchical tree structure. It omits references that resulted in an error.
 * @param reference The root prompt file reference.
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
 * Extracts the "command name" from a prompt file path.
 * This is typically the basename of the file without the `.prompt` extension.
 * @param path The path to the prompt file.
 * @returns The command name.
 */
export function getPromptCommandName(path: string): string {
	const name = basename(path, PROMPT_FILE_EXTENSION);
	return name;
}

/**
 * A higher-order function that returns a function to create an `IPromptPath` object
 * by adding `storage` and `type` attributes to a given URI.
 * @param storage The storage location ('user' or 'local').
 * @param type The type of prompt ('prompt' or 'instructions').
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
 * A higher-order function that returns a function to map an array of URIs
 * to an array of `IPromptPath` objects, adding the specified `storage` and `type`.
 * @param storage The storage location.
 * @param type The prompt type.
 * @returns A function that transforms an array of URIs.
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
