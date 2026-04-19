/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file aiStatsStatusBar.ts
 * @brief Status bar UI component for displaying AI adoption metrics.
 * 
 * This module implements a reactive status bar entry that visualizes the ratio 
 * of AI-assisted vs. manual typing. It utilizes VS Code's observable framework 
 * to provide real-time updates and includes a detailed hover tooltip with 
 * summary statistics and configuration shortcuts.
 * 
 * Domain: UI Component Architecture, Reactive Programming, Telemetry Visualization.
 */

import { n } from '../../../../../base/browser/dom.js';
...
import type { AiStatsFeature } from './aiStatsFeature.js';
import './media.css';

/**
 * @class AiStatsStatusBar
 * @brief Orchestrates the rendering and lifecycle of the AI statistics status bar entry.
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
		 * Logic: Reactive lifecycle orchestration.
		 * Invariant: The status bar entry is automatically re-registered and updated 
		 * whenever the underlying AI adoption data changes.
		 */
		this._register(autorun((reader) => {
			const statusBarItem = this._createStatusBar().keepUpdated(reader.store);

			const store = this._register(new DisposableStore());

			// Structural Intent: Registers a right-aligned entry in the workbench status bar.
			reader.store.add(this._statusbarService.addEntry({
				name: localize('inlineSuggestions', "Inline Suggestions"),
				ariaLabel: localize('inlineSuggestionsStatusBar', "Inline suggestions status bar"),
				text: '',
				tooltip: {
					element: async (_token) => {
						store.clear();
						// Hover Logic: Dynamic construction of the summary tooltip.
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
	 * @brief Constructs the visual progress bar entry for the status bar.
	 * 
	 * Logic: Creates a stylized progress indicator where the width represents 
	 * the proportion of AI-accepted characters.
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
								// Dynamic Styling: Reactively updates the progress width based on AI adoption rate.
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
	 * @brief Constructs the detailed summary view displayed on hover.
	 */
	private _createStatusBarHover() {
		const aiRatePercent = this._aiStatsFeature.aiRate.map(r => `${Math.round(r * 100)}%`);

		return n.div({
			class: 'ai-stats-status-bar',
		}, [
			// Block Logic: Tooltip Header and Settings access.
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
					n.div({ style: { marginLeft: 'auto' } }, actionBar([
						{
							action: {
								id: 'foo',
								label: '',
								enabled: true,
								// Functional Utility: Invokes the global command service to open relevant settings.
								run: () => openSettingsCommand({ ids: [AI_STATS_SETTING_ID] }).run(this._commandService),
								class: ThemeIcon.asClassName(Codicon.gear),
								tooltip: ''
							},
							options: { icon: true, label: false, }
						}
					]))
				]
			),

			// Block Logic: Adoption metrics display.
			n.div({ style: { display: 'flex' } }, [
				n.div({ style: { flex: 1 } }, [
					localize('text1', "Manual vs. AI typing ratio: {0}", aiRatePercent.get()),
				]),
			]),

			localize('text2', "Accepted inline suggestions today: {0}", this._aiStatsFeature.acceptedInlineSuggestionsToday.get()),
		]);
	}
}

/**
 * @brief Functional helper to create a reactive action bar.
 */
function actionBar(actions: { action: IAction; options: IActionOptions }[], options?: IActionBarOptions) {
	return derived((_reader) => n.div({
		class: [],
		style: {
		},
		ref: elem => {
			// Resource Management: Registers the action bar into the observable store.
			const actionBar = _reader.store.add(new ActionBar(elem, options));
			for (const { action, options } of actions) {
				actionBar.push(action, options);
			}
		}
	}));
}

/**
 * @class CommandWithArgs
 * @brief Encapsulates a command and its parameters for unified execution.
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
 * @brief Factory function for the open settings command.
 */
function openSettingsCommand(options: { ids?: string[] } = {}) {
	return new CommandWithArgs('workbench.action.openSettings', [{
		query: options.ids ? options.ids.map(id => `@id:${id}`).join(' ') : undefined,
	}]);
}
