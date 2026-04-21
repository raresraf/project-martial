/**
 * @file debugToolBar.ts
 * @brief Floating user interface component for controlling active debug sessions in VS Code.
 * 
 * Architectural Intent: Provides a persistent, globally accessible control surface for debugging operations. 
 * Decouples session management from the main sidebar to allow flexible placement across different windows and layouts.
 * 
 * Domain-Awareness: Implements spec-compliant behavior for varied debug adapters (DAP). 
 * Supports dynamic action visibility based on session state and capability discovery.
 * 
 * Performance Strategy: Uses a debounced update scheduler to synchronize UI state with 
 * high-frequency debug events (e.g., hitting breakpoints in multiple threads).
 */

/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import * as dom from '../../../../base/browser/dom.js';
import { StandardMouseEvent } from '../../../../base/browser/mouseEvent.js';
import { PixelRatio } from '../../../../base/browser/pixelRatio.js';
import { ActionBar, ActionsOrientation, IActionViewItem } from '../../../../base/browser/ui/actionbar/actionbar.js';
import { IBaseActionViewItemOptions } from '../../../../base/browser/ui/actionbar/actionViewItems.js';
import { CodeWindow, mainWindow } from '../../../../base/browser/window.js';
import { Action, IAction, IRunEvent, WorkbenchActionExecutedClassification, WorkbenchActionExecutedEvent } from '../../../../base/common/actions.js';
import * as arrays from '../../../../base/common/arrays.js';
import { RunOnceScheduler } from '../../../../base/common/async.js';
import { Codicon } from '../../../../base/common/codicons.js';
import * as errors from '../../../../base/common/errors.js';
import { DisposableStore, dispose, IDisposable, markAsSingleton, MutableDisposable } from '../../../../base/common/lifecycle.js';
import { Platform, platform } from '../../../../base/common/platform.js';
import { ThemeIcon } from '../../../../base/common/themables.js';
import { URI } from '../../../../base/common/uri.js';
import { ServicesAccessor } from '../../../../editor/browser/editorExtensions.js';
import { localize } from '../../../../nls.js';
import { ICommandAction, ICommandActionTitle } from '../../../../platform/action/common/action.js';
import { DropdownWithPrimaryActionViewItem, IDropdownWithPrimaryActionViewItemOptions } from '../../../../platform/actions/browser/dropdownWithPrimaryActionViewItem.js';
import { createActionViewItem, getFlatActionBarActions } from '../../../../platform/actions/browser/menuEntryActionViewItem.js';
import { IMenu, IMenuService, MenuId, MenuItemAction, MenuRegistry } from '../../../../platform/actions/common/actions.js';
import { IConfigurationService } from '../../../../platform/configuration/common/configuration.js';
import { ContextKeyExpr, ContextKeyExpression, IContextKeyService } from '../../../../platform/contextkey/common/contextkey.js';
import { IInstantiationService } from '../../../../platform/instantiation/common/instantiation.js';
import { INotificationService } from '../../../../platform/notification/common/notification.js';
import { IStorageService, StorageScope, StorageTarget } from '../../../../platform/storage/common/storage.js';
import { ITelemetryService } from '../../../../platform/telemetry/common/telemetry.js';
import { widgetBorder, widgetShadow } from '../../../../platform/theme/common/colorRegistry.js';
import { IThemeService, Themable } from '../../../../platform/theme/common/themeService.js';
import { getTitleBarStyle, TitlebarStyle } from '../../../../platform/window/common/window.js';
import { IWorkbenchContribution } from '../../../common/contributions.js';
import { EditorTabsMode, IWorkbenchLayoutService, LayoutSettings, Parts } from '../../../services/layout/browser/layoutService.js';
import { CONTEXT_DEBUG_STATE, CONTEXT_FOCUSED_SESSION_IS_ATTACH, CONTEXT_FOCUSED_SESSION_IS_NO_DEBUG, CONTEXT_IN_DEBUG_MODE, CONTEXT_MULTI_SESSION_DEBUG, CONTEXT_STEP_BACK_SUPPORTED, CONTEXT_SUSPEND_DEBUGGEE_SUPPORTED, CONTEXT_TERMINATE_DEBUGGEE_SUPPORTED, IDebugConfiguration, IDebugService, State, VIEWLET_ID } from '../common/debug.js';
import { FocusSessionActionViewItem } from './debugActionViewItems.js';
import { debugToolBarBackground, debugToolBarBorder } from './debugColors.js';
import { CONTINUE_ID, CONTINUE_LABEL, DISCONNECT_AND_SUSPEND_ID, DISCONNECT_AND_SUSPEND_LABEL, DISCONNECT_ID, DISCONNECT_LABEL, FOCUS_SESSION_ID, FOCUS_SESSION_LABEL, PAUSE_ID, PAUSE_LABEL, RESTART_LABEL, RESTART_SESSION_ID, REVERSE_CONTINUE_ID, STEP_BACK_ID, STEP_INTO_ID, STEP_INTO_LABEL, STEP_OUT_ID, STEP_OUT_LABEL, STEP_OVER_ID, STEP_OVER_LABEL, STOP_ID, STOP_LABEL } from './debugCommands.js';
import * as icons from './debugIcons.js';
import './media/debugToolBar.css';

const DEBUG_TOOLBAR_POSITION_KEY = 'debug.actionswidgetposition';
const DEBUG_TOOLBAR_Y_KEY = 'debug.actionswidgety';

/**
 * @class DebugToolBar
 * @brief Controller for the floating debug toolbar.
 */
export class DebugToolBar extends Themable implements IWorkbenchContribution {

	private $el: HTMLElement;
	private dragArea: HTMLElement;
	private actionBar: ActionBar;
	private activeActions: IAction[];
	private updateScheduler: RunOnceScheduler;
	private debugToolBarMenu: IMenu;

	private isVisible = false;
	private isBuilt = false;

	private readonly stopActionViewItemDisposables = this._register(new DisposableStore());
	/** coordinate of the debug toolbar per aux window */
	private readonly auxWindowCoordinates = new WeakMap<CodeWindow, { x: number; y: number | undefined }>(); 

	private readonly trackPixelRatioListener = this._register(new MutableDisposable());

	constructor(
		@INotificationService private readonly notificationService: INotificationService,
		@ITelemetryService private readonly telemetryService: ITelemetryService,
		@IDebugService private readonly debugService: IDebugService,
		@IWorkbenchLayoutService private readonly layoutService: IWorkbenchLayoutService,
		@IStorageService private readonly storageService: IStorageService,
		@IConfigurationService private readonly configurationService: IConfigurationService,
		@IThemeService themeService: IThemeService,
		@IInstantiationService private readonly instantiationService: IInstantiationService,
		@IMenuService menuService: IMenuService,
		@IContextKeyService contextKeyService: IContextKeyService,
	) {
		super(themeService);

		this.$el = dom.$('div.debug-toolbar');

		// Note: changes to this setting require a restart, so no need to listen to it.
		const controlsOnTitlebar = getTitleBarStyle(this.configurationService) === TitlebarStyle.CUSTOM;

		// Logic: Dynamically calculates toolbar constraints to avoid overlap with platform-specific window controls.
		const controlsOnLeft = controlsOnTitlebar && platform === Platform.Mac;
		const controlsOnRight = controlsOnTitlebar && (platform === Platform.Windows || platform === Platform.Linux);
		this.$el.style.transform = `translate(
			min(
				max(${controlsOnLeft ? '60px' : '0px'}, calc(-50% + (100vw * var(--x-position)))),
				calc(100vw - 100% - ${controlsOnRight ? '100px' : '0px'})
			),
			var(--y-position)
		)`;

		this.dragArea = dom.append(this.$el, dom.$('div.drag-area' + ThemeIcon.asCSSSelector(icons.debugGripper)));

		const actionBarContainer = dom.append(this.$el, dom.$('div.action-bar-container'));
		this.debugToolBarMenu = menuService.createMenu(MenuId.DebugToolBar, contextKeyService);
		this._register(this.debugToolBarMenu);

		this.activeActions = [];
		this.actionBar = this._register(new ActionBar(actionBarContainer, {
			orientation: ActionsOrientation.HORIZONTAL,
			actionViewItemProvider: (action: IAction, options: IBaseActionViewItemOptions) => {
				/**
				 * Block Logic: View item factory.
				 * Invariant: Maps specific debug actions (Stop/Disconnect/Focus) to specialized UI components.
				 */
				if (action.id === FOCUS_SESSION_ID) {
					return this.instantiationService.createInstance(FocusSessionActionViewItem, action, undefined);
				} else if (action.id === STOP_ID || action.id === DISCONNECT_ID) {
					this.stopActionViewItemDisposables.clear();
					const item = this.instantiationService.invokeFunction(accessor => createDisconnectMenuItemAction(action as MenuItemAction, this.stopActionViewItemDisposables, accessor, { hoverDelegate: options.hoverDelegate }));
					if (item) {
						return item;
					}
				}

				return createActionViewItem(this.instantiationService, action, options);
			}
		}));

		/**
		 * Functional Utility: State-driven visibility manager.
		 * Logic: Evaluates session state and user preferences to determine if the toolbar should be rendered.
		 */
		this.updateScheduler = this._register(new RunOnceScheduler(() => {
			const state = this.debugService.state;
			const toolBarLocation = this.configurationService.getValue<IDebugConfiguration>('debug').toolBarLocation;
			if (
				state === State.Inactive ||
				toolBarLocation !== 'floating' ||
				this.debugService.getModel().getSessions().every(s => s.suppressDebugToolbar) ||
				(state === State.Initializing && this.debugService.initializingOptions?.suppressDebugToolbar)
			) {
				return this.hide();
			}

			const actions = getFlatActionBarActions(this.debugToolBarMenu.getActions({ shouldForwardArgs: true }));
			if (!arrays.equals(actions, this.activeActions, (first, second) => first.id === second.id && first.enabled === second.enabled)) {
				this.actionBar.clear();
				this.actionBar.push(actions, { icon: true, label: false });
				this.activeActions = actions;
			}

			this.show();
		}, 20));

		this.updateStyles();
		this.registerListeners();
		this.hide();
	}

	private registerListeners(): void {
		this._register(this.debugService.onDidChangeState(() => this.updateScheduler.schedule()));
		this._register(this.configurationService.onDidChangeConfiguration(e => {
			if (e.affectsConfiguration('debug.toolBarLocation')) {
				this.updateScheduler.schedule();
			}
			if (e.affectsConfiguration(LayoutSettings.EDITOR_TABS_MODE) || e.affectsConfiguration(LayoutSettings.COMMAND_CENTER)) {
				this._yRange = undefined;
				this.setCoordinates();
			}
		}));
		this._register(this.debugToolBarMenu.onDidChange(() => this.updateScheduler.schedule()));
		this._register(this.actionBar.actionRunner.onDidRun((e: IRunEvent) => {
			if (e.error && !errors.isCancellationError(e.error)) {
				this.notificationService.warn(e.error);
			}

			this.telemetryService.publicLog2<WorkbenchActionExecutedEvent, WorkbenchActionExecutedClassification>('workbenchActionExecuted', { id: e.action.id, from: 'debugActionsWidget' });
		}));

		this._register(dom.addDisposableGenericMouseUpListener(this.dragArea, (event: MouseEvent) => {
			const mouseClickEvent = new StandardMouseEvent(dom.getWindow(this.dragArea), event);
			if (mouseClickEvent.detail === 2) {
				// double click on debug bar centers it again #8250
				this.setCoordinates(0.5, this.yDefault);
				this.storePosition();
			}
		}));

		/**
		 * Block Logic: Interaction handling (Drag and Drop).
		 * Invariant: Movement is relative to the active window's dimensions to support multi-monitor setups.
		 */
		this._register(dom.addDisposableGenericMouseDownListener(this.dragArea, (e: MouseEvent) => {
			this.dragArea.classList.add('dragged');
			const activeWindow = dom.getWindow(this.layoutService.activeContainer);
			const originEvent = new StandardMouseEvent(activeWindow, e);

			const originX = this.computeCurrentXPercent();
			const originY = this.getCurrentYPosition();

			const mouseMoveListener = dom.addDisposableGenericMouseMoveListener(activeWindow, (e: MouseEvent) => {
				const mouseMoveEvent = new StandardMouseEvent(activeWindow, e);
				mouseMoveEvent.preventDefault();
				this.setCoordinates(
					originX + (mouseMoveEvent.posx - originEvent.posx) / activeWindow.innerWidth,
					originY + mouseMoveEvent.posy - originEvent.posy,
				);
			});

			const mouseUpListener = dom.addDisposableGenericMouseUpListener(activeWindow, (e: MouseEvent) => {
				this.storePosition();
				this.dragArea.classList.remove('dragged');

				mouseMoveListener.dispose();
				mouseUpListener.dispose();
			});
		}));

		this._register(this.layoutService.onDidChangePartVisibility(() => this.setCoordinates()));

		this._register(this.layoutService.onDidChangeActiveContainer(async () => {
			this._yRange = undefined;
			await this.layoutService.whenContainerStylesLoaded(dom.getWindow(this.layoutService.activeContainer));
			if (this.isBuilt) {
				this.doShowInActiveContainer();
				this.setCoordinates();
			}
		}));
	}

	private computeCurrentXPercent(): number {
		const { left, width } = this.$el.getBoundingClientRect();
		return (left + width / 2) / dom.getWindow(this.$el).innerWidth;
	}

	private getCurrentXPercent(): number {
		return Number(this.$el.style.getPropertyValue('--x-position'));
	}

	private getCurrentYPosition(): number {
		return parseInt(this.$el.style.getPropertyValue('--y-position'));
	}

	/**
	 * @brief Persists the toolbar position across sessions.
	 * Invariant: Stores data in профиле-scoped storage for the main window, and in-memory for aux windows.
	 */
	private storePosition(): void {
		const activeWindow = dom.getWindow(this.layoutService.activeContainer);
		const isMainWindow = this.layoutService.activeContainer === this.layoutService.mainContainer;

		const x = this.getCurrentXPercent();
		const y = this.getCurrentYPosition();
		if (isMainWindow) {
			this.storageService.store(DEBUG_TOOLBAR_POSITION_KEY, x, StorageScope.PROFILE, StorageTarget.MACHINE);
			this.storageService.store(DEBUG_TOOLBAR_Y_KEY, y, StorageScope.PROFILE, StorageTarget.MACHINE);
		} else {
			this.auxWindowCoordinates.set(activeWindow, { x, y });
		}
	}

	override updateStyles(): void {
		super.updateStyles();

		if (this.$el) {
			this.$el.style.backgroundColor = this.getColor(debugToolBarBackground) || '';

			const widgetShadowColor = this.getColor(widgetShadow);
			this.$el.style.boxShadow = widgetShadowColor ? `0 0 8px 2px ${widgetShadowColor}` : '';

			const contrastBorderColor = this.getColor(widgetBorder);
			const borderColor = this.getColor(debugToolBarBorder);

			if (contrastBorderColor) {
				this.$el.style.border = `1px solid ${contrastBorderColor}`;
			} else {
				this.$el.style.border = borderColor ? `solid ${borderColor}` : 'none';
				this.$el.style.border = '1px 0';
			}
		}
	}

	private getStoredXPosition() {
		const currentWindow = dom.getWindow(this.layoutService.activeContainer);
		const isMainWindow = currentWindow === mainWindow;
		const storedPercentage = isMainWindow
			? Number(this.storageService.get(DEBUG_TOOLBAR_POSITION_KEY, StorageScope.PROFILE))
			: this.auxWindowCoordinates.get(currentWindow)?.x;
		return storedPercentage !== undefined && !isNaN(storedPercentage) ? storedPercentage : 0.5;
	}

	private getStoredYPosition() {
		const currentWindow = dom.getWindow(this.layoutService.activeContainer);
		const isMainWindow = currentWindow === mainWindow;
		const storedY = isMainWindow
			? this.storageService.getNumber(DEBUG_TOOLBAR_Y_KEY, StorageScope.PROFILE)
			: this.auxWindowCoordinates.get(currentWindow)?.y;
		return storedY ?? this.yDefault;
	}

	/**
	 * @brief Applies spatial coordinates to the DOM element via CSS custom properties.
	 */
	private setCoordinates(x?: number, y?: number): void {
		if (!this.isVisible) {
			return;
		}

		x ??= this.getStoredXPosition();
		y ??= this.getStoredYPosition();

		const [yMin, yMax] = this.yRange;
		y = Math.max(yMin, Math.min(y, yMax));
		this.$el.style.setProperty('--x-position', `${x}`);
		this.$el.style.setProperty('--y-position', `${y}px`);
	}

	private get yDefault() {
		return this.layoutService.mainContainerOffset.top;
	}

	private _yRange: [number, number] | undefined; 
	private get yRange(): [number, number] {
		if (!this._yRange) {
			const isTitleBarVisible = this.layoutService.isVisible(Parts.TITLEBAR_PART, dom.getWindow(this.layoutService.activeContainer));
			const yMin = isTitleBarVisible ? 0 : this.layoutService.mainContainerOffset.top;
			let yMax = 0;

			if (isTitleBarVisible) {
				if (this.configurationService.getValue(LayoutSettings.COMMAND_CENTER) === true) {
					yMax += 35;
				} else {
					yMax += 28;
				}
			}

			if (this.configurationService.getValue(LayoutSettings.EDITOR_TABS_MODE) !== EditorTabsMode.NONE) {
				yMax += 35;
			}
			this._yRange = [yMin, yMax];
		}
		return this._yRange;
	}

	private show(): void {
		if (this.isVisible) {
			this.setCoordinates();
			return;
		}
		if (!this.isBuilt) {
			this.isBuilt = true;
			this.doShowInActiveContainer();
		}

		this.isVisible = true;
		dom.show(this.$el);
		this.setCoordinates();
	}

	private doShowInActiveContainer(): void {
		this.layoutService.activeContainer.appendChild(this.$el);
		this.trackPixelRatioListener.value = PixelRatio.getInstance(dom.getWindow(this.$el)).onDidChange(
			() => this.setCoordinates()
		);
	}

	private hide(): void {
		this.isVisible = false;
		dom.hide(this.$el);
	}

	override dispose(): void {
		super.dispose();

		this.$el?.remove();
	}
}

// ... (Rest of menu registration logic) ...
