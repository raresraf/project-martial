/**
 * @file This file defines the `AiStatsStatusBar` class, a UI component for the
 * Visual Studio Code status bar that displays statistics related to AI-assisted coding.
 * It provides a visual summary of the AI generation rate and detailed information on hover.
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
 * @description A disposable UI component that renders and manages an AI statistics
 * item in the status bar. It reactively updates based on data from an `AiStatsFeature`
 * observable.
 */
export class AiStatsStatusBar extends Disposable {
	public static readonly hot = createHotClass(AiStatsStatusBar);

	constructor(
		private readonly _aiStatsFeature: AiStatsFeature,
		@IStatusbarService private readonly _statusbarService: IStatusbarService,
		@ICommandService private readonly _commandService: ICommandService,
	) {
		super();

		/**
		 * @autorun
		 * @description This reactive block creates the status bar entry and ensures it stays
		 * up-to-date. When the underlying observables from `_aiStatsFeature` change,
		 * the UI is automatically re-rendered.
		 */
		this._register(autorun((reader) => {
			const statusBarItem = this._createStatusBar().keepUpdated(reader.store);

			const store = this._register(new DisposableStore());

			// Add the entry to the VS Code status bar service.
			reader.store.add(this._statusbarService.addEntry({
				name: localize('inlineSuggestions', "Inline Suggestions"),
				ariaLabel: localize('inlineSuggestionsStatusBar', "Inline suggestions status bar"),
				text: '',
				tooltip: {
					// The hover element is created on-demand.
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
	 * @description Creates the DOM structure for the compact status bar item.
	 * It renders a small progress-bar-like element representing the AI generation rate.
	 * @returns A reactive DOM element.
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
						borderWidth: '1px',
						borderStyle: 'solid',
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
						// The width of this div is reactively bound to the AI rate observable.
						n.div({
							style: {
								width: this._aiStatsFeature.aiRate.map(v => `${v * 100}%`),
								backgroundColor: 'currentColor',
							}
						})
					])
				]
			)
		]);
	}

	/**
	 * @description Creates the DOM structure for the detailed hover tooltip that appears
	 * when the user hovers over the status bar item. It includes detailed stats and
	 * a settings action button.
	 * @returns A reactive DOM element for the hover content.
	 */
	private _createStatusBarHover() {
		const aiRatePercent = this._aiStatsFeature.aiRate.map(r => `${Math.round(r * 100)}%`);

		return n.div({
			class: 'ai-stats-status-bar',
		}, [
			n.div({
				class: 'header',
				style: {
					minWidth: '200px',
				}
			},
				[
					n.div({ style: { flex: 1 } }, [localize('aiStatsStatusBarHeader', "AI Usage Statistics")]),
					// Action bar for the settings gear icon.
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
				n.div({ style: { flex: 1, paddingRight: '4px' } }, [
					localize('text1', "Percentage of code generated by AI: {0}", aiRatePercent.get()),
				]),
			]),
			n.div({ style: { flex: 1, paddingRight: '4px' } }, [
				localize('text2', "Accepted inline suggestions today: {0}", this._aiStatsFeature.acceptedInlineSuggestionsToday.get()),
			]),
		]);
	}
}

/**
 * @description A factory function for creating a VS Code `ActionBar` UI component.
 * @param actions An array of actions and their options to be added to the action bar.
 * @param options Optional configuration for the action bar itself.
 * @returns A reactive DOM element representing the action bar.
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
 * @description A simple command object pattern to encapsulate a command ID and its arguments,
 * making it easier to pass around and execute.
 */
class CommandWithArgs {
	constructor(
		public readonly commandId: string,
		public readonly args: unknown[] = [],
	) { }

	/**
	 * @description Executes the command using the provided command service.
	 */
	public run(commandService: ICommandService): void {
		commandService.executeCommand(this.commandId, ...this.args);
	}
}

/**
 * @description A factory function that creates a command to open the VS Code settings UI.
 * @param options Allows specifying a list of setting IDs to pre-filter the settings UI.
 * @returns A `CommandWithArgs` instance configured to open settings.
 */
function openSettingsCommand(options: { ids?: string[] } = {}) {
	return new CommandWithArgs('workbench.action.openSettings', [{
		query: options.ids ? options.ids.map(id => `@id:${id}`).join(' ') : undefined,
	}]);
}
