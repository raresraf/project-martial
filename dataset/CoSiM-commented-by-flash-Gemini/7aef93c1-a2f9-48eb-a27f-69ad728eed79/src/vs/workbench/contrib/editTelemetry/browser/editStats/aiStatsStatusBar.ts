/**
 * @7aef93c1-a2f9-48eb-a27f-69ad728eed79/src/vs/workbench/contrib/editTelemetry/browser/editStats/aiStatsStatusBar.ts
 * @brief Workbench contribution for visualizing AI-driven code generation metrics in the status bar.
 * Domain: VS Code Workbench, Telemetry, UI Extensibility.
 * Architecture: Implements a reactive UI component using the VS Code Observable pattern to track and display real-time AI usage statistics.
 * Functional Utility: Provides a visual progress-style bar in the status bar and a detailed hover tooltip containing aggregate statistics (AI-generated code percentage, accepted suggestions).
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
 * @brief Core status bar contribution class.
 * Lifecycle: Managed via Disposable pattern.
 * Synchronization: Uses 'autorun' to automatically update the status bar entry when the underlying AiStatsFeature data changes.
 */
export class AiStatsStatusBar extends Disposable {
	public static readonly hot = createHotClass(AiStatsStatusBar);

	constructor(
		private readonly _aiStatsFeature: AiStatsFeature,
		@IStatusbarService private readonly _statusbarService: IStatusbarService,
		@ICommandService private readonly _commandService: ICommandService,
	) {
		super();

		this._register(autorun((reader) => {
			const statusBarItem = this._createStatusBar().keepUpdated(reader.store);

			const store = this._register(new DisposableStore());

			// Block Logic: Registration of the UI entry into the global StatusbarService.
			// Alignment: Positions the item on the right side of the status bar with priority 100.
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
	 * @brief Generates the primary DOM structure for the status bar indicator.
	 * Logic: Creates a horizontal bar where the filled width represents the AI generation rate.
	 * @return Reactive DOM node.
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
						n.div({
							style: {
								// Synchronization: width is reactively bound to the AI rate percentage.
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
	 * @brief Constructs the detailed overlay (hover) UI.
	 * Logic: Displays numerical percentages and absolute counts of AI-accepted suggestions.
	 * Includes a gear icon to jump to relevant settings.
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
					n.div({ style: { marginLeft: 'auto' } }, actionBar([
						{
							action: {
								id: 'foo',
								label: '',
								enabled: true,
								// Functional Utility: Navigation to configuration page.
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
			n.div({ style: { flex: 1, paddingRight: '4px' } }, [
				localize('text2', "Accepted inline suggestions today: {0}", this._aiStatsFeature.acceptedInlineSuggestionsToday.get()),
			]),
		]);
	}
}

/**
 * @brief Helper for constructing reactive ActionBar components within the DOM.
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
 * @brief Immutable command object for executing VS Code platform actions with specific arguments.
 */
class CommandWithArgs {
	constructor(
		public readonly commandId: string,
		public readonly args: unknown[] = [],
	) { }

	public run(commandService: ICommandService): void {
		commandService.executeCommand(this.commandId, ...this.args);
	}
}

/**
 * @brief Factory for the 'Open Settings' command scoped to AI statistics.
 */
function openSettingsCommand(options: { ids?: string[] } = {}) {
	return new CommandWithArgs('workbench.action.openSettings', [{
		query: options.ids ? options.ids.map(id => `@id:${id}`).join(' ') : undefined,
	}]);
}
