/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { IChatWidget } from '../../chat.js';
import { localize } from '../../../../../../nls.js';
import { URI } from '../../../../../../base/common/uri.js';
import { isLinux, isWindows } from '../../../../../../base/common/platform.js';
import { IPrompt, IPromptsService } from '../../../common/promptSyntax/service/types.js';
import { ILabelService } from '../../../../../../platform/label/common/label.js';
import { IOpenerService } from '../../../../../../platform/opener/common/opener.js';
import { basename, dirname, extUri } from '../../../../../../base/common/resources.js';
import { DOCUMENTATION_URL, PROMPT_FILE_EXTENSION } from '../../../common/promptSyntax/constants.js';
import { IInstantiationService } from '../../../../../../platform/instantiation/common/instantiation.js';
import { IPickOptions, IQuickInputService, IQuickPickItem } from '../../../../../../platform/quickinput/common/quickInput.js';

/**
 * Type for an {@link IQuickPickItem} with its `value` property being a `URI`.
 */
type WithUriValue<T extends IQuickPickItem> = T & { value: URI };

/**
 * Options for the {@link showSelectPromptDialog} function.
 */
export interface ISelectPromptOptions {
	/**
	 * Prompt resource `URI` to attach to the chat input, if any.
	 * If provided the resource will be pre-selected in the prompt picker dialog,
	 * otherwise the dialog will show the prompts list without any pre-selection.
	 */
	resource?: URI;

	/**
	 * Target chat widget reference to attach the prompt to. If not provided, the command
	 * attaches the prompt to a `chat panel` widget by default (either the last focused,
	 * or a new one). If the `alt` (`option` on mac) key was pressed when the prompt is
	 * selected, the `edits` widget is used instead (likewise, either the last focused,
	 * or a new one).
	 */
	widget?: IChatWidget;

	labelService: ILabelService;
	openerService: IOpenerService;
	promptsService: IPromptsService;
	initService: IInstantiationService;
	quickInputService: IQuickInputService;
}

/**
 * Result of user interaction with the prompt selection dialog.
 */
interface IPromptSelectionResult {
	/**
	 * Selected prompt item in the dialog.
	 */
	selected: WithUriValue<IQuickPickItem>;

	/**
	 * Whether the `alt` (`option` on mac) key was pressed when
	 * the prompt selection was made.
	 */
	altOption: boolean;
}

/**
 * Creates a quick pick item for a prompt.
 * @param uri The URI of the prompt file.
 * @param labelService The label service for generating user-friendly paths.
 * @returns An {@link IQuickPickItem} representation of the prompt file.
 */
const createPickItem = (
	{ uri }: IPrompt,
	labelService: ILabelService,
): WithUriValue<IQuickPickItem> => {
	const fileBasename = basename(uri);
	const fileWithoutExtension = fileBasename.replace(PROMPT_FILE_EXTENSION, '');

	return {
		type: 'item',
		label: fileWithoutExtension,
		description: labelService.getUriLabel(dirname(uri), { relative: true }),
		tooltip: uri.fsPath,
		value: uri,
	};
};

/**
 * Creates a placeholder text to show in the prompt selection dialog.
 * If no specific widget is targeted, it includes a hint about using the
 * alt/option key to target a different editor (e.g., inline chat).
 * @param widget The target chat widget, if any.
 * @returns The placeholder string for the quick pick dialog.
 */
const createPlaceholderText = (widget?: IChatWidget): string => {
	let text = localize('selectPromptPlaceholder', 'Select a prompt to use');

	// if no widget reference is provided, add the note about
	// the `alt`/`option` key modifier users can use
	if (!widget) {
		const key = (isWindows || isLinux) ? 'alt' : 'option';

		text += ' ' + localize('selectPromptPlaceholder.holdAltOption', '(hold `{0}` to use in Edits)', key);
	}

	return text;
};

/**
 * Shows a prompt selection dialog to the user and waits for a selection.
 * This function orchestrates the process of finding prompt files, presenting them
 * in a Quick Pick UI, and handling the user's selection or cancellation.
 *
 * If {@link ISelectPromptOptions.resource resource} is provided, the dialog will have
 * the resource pre-selected in the prompts list.
 *
 * @param options The set of services and options required to show the dialog.
 * @returns A promise that resolves to the user's selection and alt-key status, or `null` if canceled.
 */
export const showSelectPromptDialog = async (
	options: ISelectPromptOptions,
): Promise<IPromptSelectionResult | null> => {
	const { resource, labelService, promptsService } = options;

	// Block Logic: Find all prompt files in the workspace and map them to QuickPick items.
	const files = await promptsService.listPromptFiles()
		.then((promptFiles) => {
			return promptFiles.map((promptFile) => {
				return createPickItem(promptFile, labelService);
			});
		});

	const { quickInputService, openerService } = options;

	// Block Logic: Handle the case where no prompt files are found.
	// It shows a special Quick Pick that acts as a link to documentation.
	if (files.length === 0) {
		const docsQuickPick: WithUriValue<IQuickPickItem> = {
			type: 'item',
			label: localize('noPromptFilesFoundTooltipLabel', 'Learn how to create reusable prompts'),
			description: DOCUMENTATION_URL,
			tooltip: DOCUMENTATION_URL,
			value: URI.parse(DOCUMENTATION_URL),
		};

		const result = await quickInputService.pick(
			[docsQuickPick],
			{
				placeHolder: localize('noPromptFilesFoundLabel', 'No prompts found.'),
				canPickMany: false,
			});

		// If the user selects the documentation link, open it and terminate.
		if (result) {
			await openerService.open(result.value);
		}

		return null;
	}

	// Block Logic: If a specific prompt resource is provided, prepare it as the active item.
	// This pre-selects the item in the Quick Pick UI for better user experience.
	let activeItem: WithUriValue<IQuickPickItem> | undefined;
	if (resource) {
		activeItem = files.find((file) => {
			return extUri.isEqual(file.value, resource);
		});

		// Also, sort the list to bring the active item to the top.
		files.sort((file1, file2) => {
			if (extUri.isEqual(file1.value, resource)) {
				return -1;
			}

			if (extUri.isEqual(file2.value, resource)) {
				return 1;
			}

			return 0;
		});
	}

	// Block Logic: Configure and display the Quick Pick dialog for prompt selection.
	const { widget } = options;
	const pickOptions: IPickOptions<WithUriValue<IQuickPickItem>> = {
		placeHolder: createPlaceholderText(widget),
		activeItem,
		canPickMany: false,
		matchOnDescription: true,
	};

	// Block Logic: Set up a key modifier listener to detect if the alt/option key is held.
	// This is used to alter the behavior of the action, for example, to target a different
	// editor or context.
	let altOption = false;
	if (!location) {
		pickOptions.onKeyMods = (keyMods) => {
			if (keyMods.alt) {
				altOption = true;
			}
		};
	}

	const maybeSelectedFile = await quickInputService.pick(files, pickOptions);

	// if user cancels the dialog, return `null` instead
	if (!maybeSelectedFile) {
		return null;
	}

	return {
		selected: maybeSelectedFile,
		altOption,
	};
};
