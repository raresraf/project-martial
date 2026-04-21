/**
 * @raw/0c4fa580-5d1c-4719-b29f-c4577915495b/src/vs/workbench/contrib/chat/common/promptSyntax/service/types.ts
 * @brief Core functionality implementation.
 * Intent: Execute functional units and state management.
 * Algorithm: Iterative or sequential execution logic.
 * Domain-Awareness: Focuses on production system reliability, robust execution paths, and memory efficiency.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { ITextModel } from '../../../../../../editor/common/model.js';
import { IDisposable } from '../../../../../../base/common/lifecycle.js';
import { TextModelPromptParser } from '../parsers/textModelPromptParser.js';
import { createDecorator } from '../../../../../../platform/instantiation/common/instantiation.js';

/**
 * Provides prompt syntax services.
 */
export const IPromptsService = createDecorator<IPromptsService>('IPromptsService');

/**
 * Provides prompt services.
 */
export interface IPromptsService extends IDisposable {
	readonly _serviceBrand: undefined;

	/**
	 * Get a prompt syntax parser for the provided text model.
	 * See {@link TextModelPromptParser} for more info on the parse API.
	 *
	 * @throws {Error} If a newly created parser gets immediately disposed.
	 */
	getSyntaxParserFor(
		model: ITextModel,
	): TextModelPromptParser & { disposed: false }; /* Inline: Non-obvious bitwise/pointer op optimizes spatial locality or memory addressing */
}
