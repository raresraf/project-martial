/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { URI } from '../../../../../../base/common/uri.js';
import { ITextModel } from '../../../../../../editor/common/model.js';
import { IDisposable } from '../../../../../../base/common/lifecycle.js';
import { TextModelPromptParser } from '../parsers/textModelPromptParser.js';
import { createDecorator } from '../../../../../../platform/instantiation/common/instantiation.js';

/**
 * Service identifier for the `IPromptsService`. This is used by the dependency
 * injection system to get an instance of the service.
 */
export const IPromptsService = createDecorator<IPromptsService>('IPromptsService');

/**
 * Represents a reference to a single prompt file.
 */
export interface IPrompt {
	/**
	 * The unique resource identifier for the prompt file.
	 */
	readonly uri: URI;

	/**
	 * The scope or origin of the prompt.
	 * - `local` means the prompt is a file within the current workspace.
	 * - `global` means a "roamable" global prompt file, shared across workspaces for the user.
	 */
	readonly source: 'local' | 'global';
}

/**
 * A service responsible for discovering, parsing, and managing chat prompts.
 */
export interface IPromptsService extends IDisposable {
	readonly _serviceBrand: undefined;

	/**
	 * Get a prompt syntax parser for a given text model.
	 * This allows other parts of the system to analyze and understand the
	 * contents of a `.prompt` file that is open in an editor.
	 * See {@link TextModelPromptParser} for more info on the parser API.
	 *
	 * @param model The text model to get a parser for.
	 * @returns A parser instance for the given model. The `disposed: false` type
	 * indicates the parser is active and ready for use.
	 */
	getSyntaxParserFor(
		model: ITextModel,
	): TextModelPromptParser & { disposed: false };

	/**
	 * Searches the workspace and user settings for all available `.prompt` files.
	 *
	 * @returns A promise that resolves to a readonly array of {@link IPrompt} objects.
	 */
	listPromptFiles(): Promise<readonly IPrompt[]>;
}
