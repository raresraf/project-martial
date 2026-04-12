/**
 * @file This file defines the `AiStatsStatusBar` class, which is responsible for creating and managing a status bar
 * item in the editor to display AI-assisted typing statistics.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { n } from '../../../../../base/browser/dom.js';
import { ActionBar, IActionBarOptions, IActionOptions } from '../../../../../base/browser/ui/actionbar/actionbar.js';
import { IAction } from '../../../../../base/common/actions.js';
import { Codicon } from '../../../../../base/common/codicons.js';
import { createHotClass } from '../../../../../base/common/hotReloadHelpers.js';
import { Disposable, DisposableStore } from '../../../../../base/common/lifecycle.js';
import { autorun, derived } from '../../../../../base/common/observable.js';
import { ThemeIcon } from '../../../../../base/common/themables.js';
import { localize } from '../../../../../nls.js';
import { ICommandService } from '../../../../../platform/commands/common/commands.js';
import { IStatusbarService, StatusbarAlignment } from '../../../../services/statusbar/browser/statusbar.js';
import { AI_STATS_SETTING_ID } from '../settingIds.js';
import type { AiStatsFeature } from './aiStatsFeature.js';
import './media.css';

/**
 * @class AiStatsStatusBar
 * @description Creates a status bar entry to visualize AI usage statistics.
 * This includes a progress-bar-like indicator in the status bar and a detailed
 * hover view with more information and actions.
 */
export class AiStatsStatusBar extends Disposable {
	public static readonly hot = createHotClass(AiStatsStatusBar);

	/**
	 * @param _aiStatsFeature The feature providing the AI statistics data.
	 * @param _statusbarService The service for adding items to the status bar.
	 * @param _commandService The service for executing commands.
	 */
	constructor(
		private readonly _aiStatsFeature: AiStatsFeature,
		@IStatusbarService private readonly _statusbarService: IStatusbarService,
		@ICommandService private readonly _commandService: ICommandService,
	) {
		super();

		// This autorun block creates and maintains the status bar item.
		// It will re-run whenever an observable it reads changes.
		this._register(autorun((reader) => {
			const statusBarItem = this._createStatusBar().keepUpdated(reader.store);

			const store = this._register(new DisposableStore());

			// Adds the status bar entry to the UI.
			reader.store.add(this._statusbarService.addEntry({
				name: localize('inlineSuggestions', "Inline Suggestions"),
				ariaLabel: localize('inlineSuggestionsStatusBar', "Inline suggestions status bar"),
				text: '',
				tooltip: {
					element: async (_token) => {
						store.clear();
						const elem = this._createStatusBarHover();
						return elem.keepUpdated(store).element;
					},
					markdownNotSupportedFallback: undefined,
				},
				content: statusBarItem.element,
			}, 'aiStatsStatusBar', StatusbarAlignment.RIGHT, 100));
		}));
	}


	/**
	 * @description Builds the DOM structure for the status bar item itself.
	 * This renders a visual indicator (a progress bar) representing the AI contribution rate.
	 * @returns A DOM node to be rendered in the status bar.
	 */
	private _createStatusBar() {
		return n.div({
			style: {
				height: '100%',
				display: 'flex',
				alignItems: 'center',
				justifyContent: 'center',
			}
		}, [
			n.div(
				{
					class: 'ai-stats-status-bar',
					style: {
						display: 'flex',
						flexDirection: 'column',

						width: 50,
						height: 6,

						borderRadius: 6,
						border: '1px solid var(--vscode-statusBar-foreground)',
					}
				},
				[
					n.div({
						style: {
							flex: 1,

							display: 'flex',
							overflow: 'hidden',

							borderRadius: 6,
							border: '1px solid transparent',
						}
					}, [
						n.div({
							style: {
								width: this._aiStatsFeature.aiRate.map(v => `${v * 100}%`),
								backgroundColor: 'var(--vscode-statusBar-foreground)',
							}
						})
					])
				]
			)
		]);
	}

	/**
	 * @description Builds the DOM structure for the detailed hover view of the status bar item.
	 * This view shows the AI usage ratio, the number of accepted suggestions, and an
	 * action button to open relevant settings.
	 * @returns A DOM node to be rendered in the hover view.
	 */
	private _createStatusBarHover() {
		const aiRatePercent = this._aiStatsFeature.aiRate.map(r => `${Math.round(r * 100)}%`);

		return n.div({
			class: 'ai-stats-status-bar',
		}, [
			n.div({
				class: 'header',
				style: {
					fontWeight: 'bold',
					fontSize: '14px',
					marginBottom: '4px',
					minWidth: '200px',
				}
			},
				[
					n.div({ style: { flex: 1 } }, [localize('aiStatsStatusBarHeader', "AI Usage Statistics")]),
					// Action bar with a settings gear icon
					n.div({ style: { marginLeft: 'auto' } }, actionBar([
						{
							action: {
								id: 'foo',
								label: '',
								enabled: true,
								run: () => openSettingsCommand({ ids: [AI_STATS_SETTING_ID] }).run(this._commandService),
								class: ThemeIcon.asClassName(Codicon.gear),
								tooltip: ''
							},
							options: { icon: true, label: false, }
						}
					]))
				]
			),

			n.div({ style: { display: 'flex' } }, [
				n.div({ style: { flex: 1 } }, [
					localize('text1', "Manual vs. AI typing ratio: {0}", aiRatePercent.get()),
				]),
				/*
				TODO: Write article that explains the ratio and link to it.

				n.div({ style: { marginLeft: 'auto' } }, actionBar([
					{
						action: {
							id: 'aiStatsStatusBar.openSettings',
							label: '',
							enabled: true,
							run: () => { },
							class: ThemeIcon.asClassName(Codicon.info),
							tooltip: ''
						},
						options: { icon: true, label: true, }
					}
				]))*/
			]),

			localize('text2', "Accepted inline suggestions today: {0}", this._aiStatsFeature.acceptedInlineSuggestionsToday.get()),
		]);
	}
}

/**
 * @description A helper function to create a generic ActionBar component with a set of actions.
 * @param actions An array of actions and their options to be added to the action bar.
 * @param options Optional configuration for the ActionBar itself.
 * @returns A derived observable that builds and manages the ActionBar's DOM element.
 */
function actionBar(actions: { action: IAction; options: IActionOptions }[], options?: IActionBarOptions) {
	return derived((_reader) => n.div({
		class: [],
		style: {
		},
		ref: elem => {
			const actionBar = _reader.store.add(new ActionBar(elem, options));
			for (const { action, options } of actions) {
				actionBar.push(action, options);
			}
		}
	}));
}

/**
 * @class CommandWithArgs
 * @description A simple class to encapsulate a command ID and its arguments for easier execution.
 */
class CommandWithArgs {
	constructor(
		public readonly commandId: string,
		public readonly args: unknown[] = [],
	) { }

	/**
	 * @description Executes the command using the provided command service.
	 * @param commandService The service to use for command execution.
	 */
	public run(commandService: ICommandService): void {
		commandService.executeCommand(this.commandId, ...this.args);
	}
}

/**
 * @description Creates a `CommandWithArgs` instance specifically for opening the settings UI.
 * @param options Allows specifying a list of setting IDs to filter the settings UI.
 * @returns A `CommandWithArgs` object configured to open the settings view.
 */
function openSettingsCommand(options: { ids?: string[] } = {}) {
	return new CommandWithArgs('workbench.action.openSettings', [{
		query: options.ids ? options.ids.map(id => `@id:${id}`).join(' ') : undefined,
	}]);
}
