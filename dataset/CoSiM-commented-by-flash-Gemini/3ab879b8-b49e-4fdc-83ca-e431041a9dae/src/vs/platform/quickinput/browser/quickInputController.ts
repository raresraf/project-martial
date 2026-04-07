/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import * as dom from '../../../base/browser/dom.js';
import * as domStylesheetsJs from '../../../base/browser/domStylesheets.js';
import { ActionBar } from '../../../base/browser/ui/actionbar/actionbar.js';
import { ActionViewItem } from '../../../base/browser/ui/actionbar/actionViewItems.js';
import { Button } from '../../../base/browser/ui/button/button.js';
import { CountBadge } from '../../../base/browser/ui/countBadge/countBadge.js';
import { ProgressBar } from '../../../base/browser/ui/progressbar/progressbar.js';
import { CancellationToken } from '../../../base/common/cancellation.js';
import { Emitter, Event } from '../../../base/common/event.js';
import { KeyCode } from '../../../base/common/keyCodes.js';
import { Disposable, DisposableStore, dispose } from '../../../base/common/lifecycle.js';
import Severity from '../../../base/common/severity.js';
import { isString } from '../../../base/common/types.js';
import { localize } from '../../../nls.js';
import { IInputBox, IInputOptions, IKeyMods, IPickOptions, IQuickInput, IQuickInputButton, IQuickNavigateConfiguration, IQuickPick, IQuickPickItem, IQuickWidget, QuickInputHideReason, QuickPickInput, QuickPickFocus } from '../common/quickInput.js';
import { QuickInputBox } from './quickInputBox.js';
import { QuickInputUI, Writeable, IQuickInputStyles, IQuickInputOptions, QuickPick, backButton, InputBox, Visibilities, QuickWidget, InQuickInputContextKey, QuickInputTypeContextKey, EndOfQuickInputBoxContextKey, QuickInputAlignmentContextKey } from './quickInput.js';
import { ILayoutService } from '../../layout/browser/layoutService.js';
import { mainWindow } from '../../../base/browser/window.js';
import { IInstantiationService } from '../../instantiation/common/instantiation.js';
import { QuickInputTree } from './quickInputTree.js';
import { IContextKeyService } from '../../contextkey/common/contextkey.js';
import './quickInputActions.js';
import { autorun, observableValue } from '../../../base/common/observable.js';
import { StandardMouseEvent } from '../../../base/browser/mouseEvent.js';
import { IStorageService, StorageScope, StorageTarget } from '../../storage/common/storage.js';
import { IConfigurationService } from '../../configuration/common/configuration.js';
import { Platform, platform } from '../../../base/common/platform.js';
import { getTitleControlsStyle, TitleControlsStyle } from '../../window/common/window.js';
import { getZoomFactor } from '../../../base/browser/browser.js';

const $ = dom.$;

const VIEWSTATE_STORAGE_KEY = 'workbench.quickInput.viewState';

type QuickInputViewState = {
	readonly top?: number;
	readonly left?: number;
};

/**
 * @file quickInputController.ts
 * @brief Implements the controller for the Quick Input UI, managing its lifecycle, user interactions, and layout.
 *
 * This module defines the `QuickInputController` class, which is responsible for orchestrating the
 * quick pick and input box functionalities. It handles rendering, events, and state management
 * for the quick input widget, ensuring a consistent user experience. It integrates with various
 * services like layout, instantiation, and context key services to provide a flexible and
 * extensible quick input mechanism.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @class QuickInputController
 * @augments Disposable
 * @brief Manages the lifecycle, display, and interaction logic for the Quick Input UI.
 *
 * Functional Utility: This class orchestrates the Quick Input widget, handling its creation,
 *                     showing, hiding, layout, and event processing. It serves as the central
 *                     component for presenting interactive quick picks and input boxes to the user.
 *                     It also manages the internal state related to key modifiers and the currently
 *                     active quick input.
 * Domain: User Interface, Quick Input, Event Handling, State Management.
 */
export class QuickInputController extends Disposable {
	private static readonly MAX_WIDTH = 600; // Max total width of quick input widget

	private idPrefix: string;
	private ui: QuickInputUI | undefined;
	private dimension?: dom.IDimension;
	private titleBarOffset?: number;
	private enabled = true;
	private readonly onDidAcceptEmitter = this._register(new Emitter<void>());
	private readonly onDidCustomEmitter = this._register(new Emitter<void>());
	private readonly onDidTriggerButtonEmitter = this._register(new Emitter<IQuickInputButton>());
	private keyMods: Writeable<IKeyMods> = { ctrlCmd: false, alt: false };

	private controller: IQuickInput | null = null;
	get currentQuickInput() { return this.controller ?? undefined; }

	private _container: HTMLElement;
	get container() { return this._container; }

	private styles: IQuickInputStyles;

	private onShowEmitter = this._register(new Emitter<void>());
	readonly onShow = this.onShowEmitter.event;

	private onHideEmitter = this._register(new Emitter<void>());
	readonly onHide = this.onHideEmitter.event;

	private previousFocusElement?: HTMLElement;

	private viewState: QuickInputViewState | undefined;
	private dndController: QuickInputDragAndDropController | undefined;

	private readonly inQuickInputContext = InQuickInputContextKey.bindTo(this.contextKeyService);
	private readonly quickInputTypeContext = QuickInputTypeContextKey.bindTo(this.contextKeyService);
	private readonly endOfQuickInputBoxContext = EndOfQuickInputBoxContextKey.bindTo(this.contextKeyService);

	constructor(
		private options: IQuickInputOptions,
		@ILayoutService private readonly layoutService: ILayoutService,
		@IInstantiationService private readonly instantiationService: IInstantiationService,
		@IContextKeyService private readonly contextKeyService: IContextKeyService,
		@IStorageService private readonly storageService: IStorageService
	) {
		super();
		this.idPrefix = options.idPrefix;
		this._container = options.container;
		this.styles = options.styles;
		// Block Logic: Registers listeners for key modifier events (Ctrl/Cmd, Alt) on all windows.
		// Functional Utility: Captures the state of modifier keys during user interaction with the quick input.
		this._register(Event.runAndSubscribe(dom.onDidRegisterWindow, ({ window, disposables }) => this.registerKeyModsListeners(window, disposables), { window: mainWindow, disposables: this._store }));
		// Block Logic: Handles the scenario where the window containing the quick input is about to be unregistered.
		//              It re-parents the UI to the main container and re-layouts it to prevent loss of functionality.
		// Pre-condition: A window where the quick input is contained is being unregistered.
		this._register(dom.onWillUnregisterWindow(window => {
			if (this.ui && dom.getWindow(this.ui.container) === window) {
				// The window this quick input is contained in is about to
				// close, so we have to make sure to reparent it back to an
				// existing parent to not loose functionality.
				// (https://github.com/microsoft/vscode/issues/195870)
				this.reparentUI(this.layoutService.mainContainer);
				this.layout(this.layoutService.mainContainerDimension, this.layoutService.mainContainerOffset.quickPickTop);
			}
		}));
		// Inline: Loads the previously saved view state (position and size) from storage.
		this.viewState = this.loadViewState();
	}

	/**
	 * @brief Registers event listeners to track the state of modifier keys (Ctrl/Cmd, Alt).
	 * Functional Utility: Captures the current state of modifier keys during keyboard and mouse events
	 *                     to inform quick input actions that might depend on these modifiers.
	 * @param window The {@link Window} object to attach the event listeners to.
	 * @param disposables A {@link DisposableStore} to manage the lifecycle of the registered listeners.
	 * Pre-condition: `window` is a valid Window object.
	 * Post-condition: Event listeners for `keydown`, `keyup`, and `mousedown` are active on the specified window,
	 *                 updating `this.keyMods` with the current state of Ctrl/Cmd and Alt keys.
	 */
	private registerKeyModsListeners(window: Window, disposables: DisposableStore): void {
		const listener = (e: KeyboardEvent | MouseEvent) => {
			this.keyMods.ctrlCmd = e.ctrlKey || e.metaKey;
			this.keyMods.alt = e.altKey;
		};

		for (const event of [dom.EventType.KEY_DOWN, dom.EventType.KEY_UP, dom.EventType.MOUSE_DOWN]) {
			disposables.add(dom.addDisposableListener(window, event, listener, true));
		}
	}

	/**
	 * @brief Retrieves or creates the Quick Input UI components.
	 * Functional Utility: This method ensures that the UI components for the quick input are
	 *                     initialized and properly parented within the DOM. It also handles
	 *                     re-parenting the UI if the active container changes (e.g., multi-window support).
	 * @param showInActiveContainer Optional. If true, the UI will be re-parented to the currently active container.
	 * @returns The {@link QuickInputUI} object containing all managed UI elements.
	 * Pre-condition: The necessary services (layout, instantiation) are available.
	 * Post-condition: The `this.ui` property is initialized and returned, with all sub-components created
	 *                 and event listeners attached. Styles are applied, and the drag-and-drop controller is set up.
	 */
	private getUI(showInActiveContainer?: boolean) {
		// Block Logic: If the UI has already been created, return the existing instance.
		//              Handles re-parenting for multi-window support if `showInActiveContainer` is true.
		// Pre-condition: `this.ui` might be null or undefined.
		if (this.ui) {
			// In order to support aux windows, re-parent the controller
			// if the original event is from a different document
			if (showInActiveContainer) {
				if (dom.getWindow(this._container) !== dom.getWindow(this.layoutService.activeContainer)) {
					this.reparentUI(this.layoutService.activeContainer);
					this.layout(this.layoutService.activeContainerDimension, this.layoutService.activeContainerOffset.quickPickTop);
				}
			}

			return this.ui;
		}

		// Block Logic: Creates the main HTML container for the quick input widget.
		//              Sets its initial styles and attributes.
		const container = dom.append(this._container, $('.quick-input-widget.show-file-icons'));
		container.tabIndex = -1;
		container.style.display = 'none';

		const styleSheet = domStylesheetsJs.createStyleSheet(container);

		// Block Logic: Sets up the title bar section of the quick input, including left and right action bars.
		const titleBar = dom.append(container, $('.quick-input-titlebar'));

		const leftActionBar = this._register(new ActionBar(titleBar, { hoverDelegate: this.options.hoverDelegate }));
		leftActionBar.domNode.classList.add('quick-input-left-action-bar');

		const title = dom.append(titleBar, $('.quick-input-title'));

		const rightActionBar = this._register(new ActionBar(titleBar, { hoverDelegate: this.options.hoverDelegate }));
		rightActionBar.domNode.classList.add('quick-input-right-action-bar');

		const headerContainer = dom.append(container, $('.quick-input-header'));

		// Block Logic: Sets up the "Check All" checkbox for quick picks that allow multiple selections.
		//              Attaches event listeners to update the list's checked state and to manage focus.
		const checkAll = <HTMLInputElement>dom.append(headerContainer, $('input.quick-input-check-all'));
		checkAll.type = 'checkbox';
		checkAll.setAttribute('aria-label', localize('quickInput.checkAll', "Toggle all checkboxes"));
		this._register(dom.addStandardDisposableListener(checkAll, dom.EventType.CHANGE, e => {
			const checked = checkAll.checked;
			list.setAllVisibleChecked(checked);
		}));
		this._register(dom.addDisposableListener(checkAll, dom.EventType.CLICK, e => {
			if (e.x || e.y) { // Avoid 'click' triggered by 'space'...
				inputBox.setFocus();
			}
		}));

		const description2 = dom.append(headerContainer, $('.quick-input-description'));
		const inputContainer = dom.append(headerContainer, $('.quick-input-and-message'));
		const filterContainer = dom.append(inputContainer, $('.quick-input-filter'));

		// Block Logic: Initializes the input box component, applying specific styles and accessibility attributes.
		const inputBox = this._register(new QuickInputBox(filterContainer, this.styles.inputBox, this.styles.toggle));
		inputBox.setAttribute('aria-describedby', `${this.idPrefix}message`);

		// Block Logic: Sets up the "visible count" badge, which displays how many items are currently shown in the list.
		const visibleCountContainer = dom.append(filterContainer, $('.quick-input-visible-count'));
		visibleCountContainer.setAttribute('aria-live', 'polite');
		visibleCountContainer.setAttribute('aria-atomic', 'true');
		const visibleCount = this._register(new CountBadge(visibleCountContainer, { countFormat: localize({ key: 'quickInput.visibleCount', comment: ['This tells the user how many items are shown in a list of items to select from. The items can be anything. Currently not visible, but read by screen readers.'] }, "{0} Results") }, this.styles.countBadge));

		// Block Logic: Sets up the "selected count" badge, which displays how many items are currently selected in the list.
		const countContainer = dom.append(filterContainer, $('.quick-input-count'));
		countContainer.setAttribute('aria-live', 'polite');
		const count = this._register(new CountBadge(countContainer, { countFormat: localize({ key: 'quickInput.countSelected', comment: ['This tells the user how many items are selected in a list of items to select from. The items can be anything.'] }, "{0} Selected") }, this.styles.countBadge));

		const inlineActionBar = this._register(new ActionBar(headerContainer, { hoverDelegate: this.options.hoverDelegate }));
		inlineActionBar.domNode.classList.add('quick-input-inline-action-bar');

		// Block Logic: Creates and configures the "OK" button, attaching an event listener for user acceptance.
		const okContainer = dom.append(headerContainer, $('.quick-input-action'));
		const ok = this._register(new Button(okContainer, this.styles.button));
		ok.label = localize('ok', "OK");
		this._register(ok.onDidClick(e => {
			this.onDidAcceptEmitter.fire();
		}));

		// Block Logic: Creates and configures the "Custom" button, attaching an event listener for custom actions.
		const customButtonContainer = dom.append(headerContainer, $('.quick-input-action'));
		const customButton = this._register(new Button(customButtonContainer, { ...this.styles.button, supportIcons: true }));
		customButton.label = localize('custom', "Custom");
		this._register(customButton.onDidClick(e => {
			this.onDidCustomEmitter.fire();
		}));

		const message = dom.append(inputContainer, $(`#${this.idPrefix}message.quick-input-message`));

		// Inline: Initializes the progress bar component.
		const progressBar = this._register(new ProgressBar(container, this.styles.progressBar));
		progressBar.getContainer().classList.add('quick-input-progress');

		const widget = dom.append(container, $('.quick-input-html-widget'));
		widget.tabIndex = -1;

		const description1 = dom.append(container, $('.quick-input-description'));

		// Block Logic: Initializes the `QuickInputTree` (the list component) which displays the quick pick items.
		//              Attaches various event listeners to handle focus, checked states, and item selection.
		const listId = this.idPrefix + 'list';
		const list = this._register(this.instantiationService.createInstance(QuickInputTree, container, this.options.hoverDelegate, this.options.linkOpenerDelegate, listId));
		inputBox.setAttribute('aria-controls', listId);
		this._register(list.onDidChangeFocus(() => {
			inputBox.setAttribute('aria-activedescendant', list.getActiveDescendant() ?? '');
		}));
		this._register(list.onChangedAllVisibleChecked(checked => {
			checkAll.checked = checked;
		}));
		this._register(list.onChangedVisibleCount(c => {
			visibleCount.setCount(c);
		}));
		this._register(list.onChangedCheckedCount(c => {
			count.setCount(c);
		}));
		this._register(list.onLeave(() => {
			// Defer to avoid the input field reacting to the triggering key.
			// TODO@TylerLeonhardt https://github.com/microsoft/vscode/issues/203675
			setTimeout(() => {
				if (!this.controller) {
					return;
				}
				inputBox.setFocus();
				if (this.controller instanceof QuickPick && this.controller.canSelectMany) {
					list.clearFocus();
				}
			}, 0);
		}));

		const focusTracker = dom.trackFocus(container);
		this._register(focusTracker);
		// Block Logic: Handles focus events within the quick input container.
		//              Updates context keys related to input box selection and tracks the previously focused element.
		this._register(dom.addDisposableListener(container, dom.EventType.FOCUS, e => {
			const ui = this.getUI();
			if (dom.isAncestor(e.relatedTarget as HTMLElement, ui.inputContainer)) {
				const value = ui.inputBox.isSelectionAtEnd();
				if (this.endOfQuickInputBoxContext.get() !== value) {
					this.endOfQuickInputBoxContext.set(value);
				}
			}
			// Ignore focus events within container
			if (dom.isAncestor(e.relatedTarget as HTMLElement, ui.container)) {
				return;
			}
			this.inQuickInputContext.set(true);
			this.previousFocusElement = dom.isHTMLElement(e.relatedTarget) ? e.relatedTarget : undefined;
		}, true));
		// Block Logic: Handles blur events for the quick input container.
		//              If focus moves outside the container and `ignoreFocusOut` is not set, it hides the quick input.
		this._register(focusTracker.onDidBlur(() => {
			if (!this.getUI().ignoreFocusOut && !this.options.ignoreFocusOut()) {
				this.hide(QuickInputHideReason.Blur);
			}
			this.inQuickInputContext.set(false);
			this.endOfQuickInputBoxContext.set(false);
			this.previousFocusElement = undefined;
		}));
		// Block Logic: Manages `aria-activedescendant` for accessibility and updates `EndOfQuickInputBoxContextKey`.
		this._register(inputBox.onKeyDown(_ => {
			const value = this.getUI().inputBox.isSelectionAtEnd();
			if (this.endOfQuickInputBoxContext.get() !== value) {
				this.endOfQuickInputBoxContext.set(value);
			}
			// Allow screenreaders to read what's in the input
			// Note: this works for arrow keys and selection changes,
			// but not for deletions since that often triggers a
			// change in the list.
			inputBox.removeAttribute('aria-activedescendant');
		}));
		// Block Logic: Ensures the input box regains focus when the container receives focus.
		this._register(dom.addDisposableListener(container, dom.EventType.FOCUS, (e: FocusEvent) => {
			inputBox.setFocus();
		}));
		// TODO: Turn into commands instead of handling KEY_DOWN
		// Block Logic: Handles global keydown events for the quick input widget.
		//              Includes logic for Enter (accept), Escape (hide), and Tab (focus navigation).
		this._register(dom.addStandardDisposableListener(container, dom.EventType.KEY_DOWN, (event) => {
			// Pre-condition: Event target is not inside the custom widget area.
			if (dom.isAncestor(event.target, widget)) {
				return; // Ignore event if target is inside widget to allow the widget to handle the event.
			}
			switch (event.keyCode) {
				case KeyCode.Enter:
					dom.EventHelper.stop(event, true);
					// Inline: Triggers the accept action if the quick input is enabled.
					if (this.enabled) {
						this.onDidAcceptEmitter.fire();
					}
					break;
				case KeyCode.Escape:
					dom.EventHelper.stop(event, true);
					this.hide(QuickInputHideReason.Gesture);
					break;
				case KeyCode.Tab:
					// Block Logic: Custom tab navigation logic for accessibility within the quick input.
					//              Determines tab stops based on visible elements and wraps focus if needed.
					// Invariant: Ensures consistent tab order even with dynamically visible elements.
					if (!event.altKey && !event.ctrlKey && !event.metaKey) {
						// detect only visible actions
						const selectors = [
							'.quick-input-list .monaco-action-bar .always-visible',
							'.quick-input-list-entry:hover .monaco-action-bar',
							'.monaco-list-row.focused .monaco-action-bar'
						];

						if (container.classList.contains('show-checkboxes')) {
							selectors.push('input');
						} else {
							selectors.push('input[type=text]');
						}
						if (this.getUI().list.displayed) {
							selectors.push('.monaco-list');
						}
						// focus links if there are any
						if (this.getUI().message) {
							selectors.push('.quick-input-message a');
						}

						if (this.getUI().widget) {
							// Inline: Allow the custom widget to handle its own tab navigation.
							if (dom.isAncestor(event.target, this.getUI().widget)) {
								// let the widget control tab
								break;
							}
							selectors.push('.quick-input-html-widget');
						}
						const stops = container.querySelectorAll<HTMLElement>(selectors.join(', '));
						if (!event.shiftKey && dom.isAncestor(event.target, stops[stops.length - 1])) {
							dom.EventHelper.stop(event, true);
							stops[0].focus();
						}
						if (event.shiftKey && dom.isAncestor(event.target, stops[0])) {
							dom.EventHelper.stop(event, true);
							stops[stops.length - 1].focus();
						}
					}
					break;
			}
		}));

		// Drag and Drop support
		this.dndController = this._register(this.instantiationService.createInstance(
			QuickInputDragAndDropController,
			this._container,
			container,
			[
				{
					node: titleBar,
					includeChildren: true
				},
				{
					node: headerContainer,
					includeChildren: false
				}
			],
			this.viewState
		));

		// Block Logic: Listens for updates from the DND controller and updates the quick input's layout and saved view state.
		this._register(autorun(reader => {
			const dndViewState = this.dndController?.dndViewState.read(reader);
			if (!dndViewState) {
				return;
			}

			// Inline: Updates the view state with new position if provided by DND controller.
			if (dndViewState.top !== undefined && dndViewState.left !== undefined) {
				this.viewState = {
					...this.viewState,
					top: dndViewState.top,
					left: dndViewState.left
				};
			} else {
				// Reset position/size
				this.viewState = undefined;
			}

			this.updateLayout();

			// Inline: Saves the updated view state to storage when DND operation is complete.
			if (dndViewState.done) {
				this.saveViewState(this.viewState);
			}
		}));

		// Block Logic: Populates the `this.ui` object with all created UI components and their associated functionalities.
		this.ui = {
			container,
			styleSheet,
			leftActionBar,
			titleBar,
			title,
			description1,
			description2,
			widget,
			rightActionBar,
			inlineActionBar,
			checkAll,
			inputContainer,
			filterContainer,
			inputBox,
			visibleCountContainer,
			visibleCount,
			countContainer,
			count,
			okContainer,
			ok,
			message,
			customButtonContainer,
			customButton,
			list,
			progressBar,
			onDidAccept: this.onDidAcceptEmitter.event,
			onDidCustom: this.onDidCustomEmitter.event,
			onDidTriggerButton: this.onDidTriggerButtonEmitter.event,
			ignoreFocusOut: false,
			keyMods: this.keyMods,
			show: controller => this.show(controller),
			hide: () => this.hide(),
			setVisibilities: visibilities => this.setVisibilities(visibilities),
			setEnabled: enabled => this.setEnabled(enabled),
			setContextKey: contextKey => this.options.setContextKey(contextKey),
			linkOpenerDelegate: content => this.options.linkOpenerDelegate(content)
		};
		this.updateStyles();
		return this.ui;
	}

	/**
	 * @brief Re-parents the Quick Input UI to a new container.
	 * Functional Utility: Used primarily in multi-window scenarios to move the quick input widget
	 *                     from one DOM container to another (e.g., when the active window changes).
	 * @param container The new {@link HTMLElement} to append the quick input UI to.
	 * Pre-condition: `this.ui` must be initialized.
	 * Post-condition: The quick input's DOM element is moved to the new `container`, and the
	 *                 `dndController` is also informed of the change.
	 */
	private reparentUI(container: HTMLElement): void {
		if (this.ui) {
			this._container = container;
			dom.append(this._container, this.ui.container);
			this.dndController?.reparentUI(this._container);
		}
	}

	/**
	 * @brief Presents a quick pick UI to the user, allowing them to select one or more items from a list.
	 * Functional Utility: Provides a versatile and configurable way to prompt the user for a selection
	 *                     from a list of options, supporting single or multi-selection, dynamic updates,
	 *                     and custom actions.
	 * @param picks A {@link Promise} or array of {@link QuickPickInput} items to display in the quick pick.
	 * @param options Configuration options for the quick pick, including title, placeholder, validation, etc.
	 * @param token An optional {@link CancellationToken} to cancel the quick pick operation.
	 * @returns A {@link Promise} that resolves to the selected item(s) or `undefined` if cancelled.
	 * @template T The type of the items in the quick pick list, extending {@link IQuickPickItem}.
	 * @template O The type of the options object, extending {@link IPickOptions}.
	 * Pre-condition: `picks` contains valid items, and `options` are correctly configured.
	 * Post-condition: The quick pick UI is displayed, and the promise resolves upon user interaction (selection or cancellation).
	 */
	pick<T extends IQuickPickItem, O extends IPickOptions<T>>(picks: Promise<QuickPickInput<T>[]> | QuickPickInput<T>[], options: IPickOptions<T> = {}, token: CancellationToken = CancellationToken.None): Promise<(O extends { canPickMany: true } ? T[] : T) | undefined> {
		type R = (O extends { canPickMany: true } ? T[] : T) | undefined;
		return new Promise<R>((doResolve, reject) => {
			let resolve = (result: R) => {
				resolve = doResolve;
				options.onKeyMods?.(input.keyMods);
				doResolve(result);
			};
			// Block Logic: Checks if the operation has already been cancelled. If so, resolves immediately with undefined.
			if (token.isCancellationRequested) {
				resolve(undefined);
				return;
			}
			// Block Logic: Creates a new QuickPick instance configured to use separators.
			const input = this.createQuickPick<T>({ useSeparators: true });
			let activeItem: T | undefined;
			// Block Logic: Sets up a collection of disposables to manage the lifecycle of the QuickPick and its listeners.
			const disposables = [
				input,
				// Block Logic: Handles the acceptance of items in the QuickPick.
				//              If `canSelectMany` is true, it resolves with all selected items; otherwise, with the active item.
				input.onDidAccept(() => {
					if (input.canSelectMany) {
						resolve(<R>input.selectedItems.slice());
						input.hide();
					} else {
						const result = input.activeItems[0];
						if (result) {
							resolve(<R>result);
							input.hide();
						}
					}
				}),
				// Block Logic: Notifies `options.onDidFocus` when the active item in the QuickPick changes.
				input.onDidChangeActive(items => {
					const focused = items[0];
					if (focused && options.onDidFocus) {
						options.onDidFocus(focused);
					}
				}),
				// Block Logic: Handles item selection in the QuickPick.
				//              If `canSelectMany` is false, it resolves with the selected item and hides the QuickPick.
				input.onDidChangeSelection(items => {
					if (!input.canSelectMany) {
						const result = items[0];
						if (result) {
							resolve(<R>result);
							input.hide();
						}
					}
				}),
				// Block Logic: Handles when a button on an item is triggered.
				//              Allows for custom actions like removing the item from the list.
				input.onDidTriggerItemButton(event => options.onDidTriggerItemButton && options.onDidTriggerItemButton({
					...event,
					removeItem: () => {
						const index = input.items.indexOf(event.item);
						if (index !== -1) {
							const items = input.items.slice();
							const removed = items.splice(index, 1);
							const activeItems = input.activeItems.filter(activeItem => activeItem !== removed[0]);
							const keepScrollPositionBefore = input.keepScrollPosition;
							input.keepScrollPosition = true;
							input.items = items;
							if (activeItems) {
								input.activeItems = activeItems;
							}
							input.keepScrollPosition = keepScrollPositionBefore;
						}
					}
				})),
				// Block Logic: Handles when a button on a separator is triggered, delegating to `options.onDidTriggerSeparatorButton`.
				input.onDidTriggerSeparatorButton(event => options.onDidTriggerSeparatorButton?.(event)),
				// Block Logic: Manages the active item when the input value changes, ensuring it remains focused if appropriate.
				input.onDidChangeValue(value => {
					if (activeItem && !value && (input.activeItems.length !== 1 || input.activeItems[0] !== activeItem)) {
						input.activeItems = [activeItem];
					}
				}),
				// Block Logic: Hides the QuickPick if the cancellation token is requested.
				token.onCancellationRequested(() => {
					input.hide();
				}),
				// Block Logic: Disposes of resources and resolves the promise with `undefined` when the QuickPick is hidden.
				input.onDidHide(() => {
					dispose(disposables);
					resolve(undefined);
				}),
			];
			// Block Logic: Configures the QuickPick instance with the provided options.
			input.title = options.title;
			if (options.value) {
				input.value = options.value;
			}
			input.canSelectMany = !!options.canPickMany;
			input.placeholder = options.placeHolder;
			input.ignoreFocusOut = !!options.ignoreFocusLost;
			input.matchOnDescription = !!options.matchOnDescription;
			input.matchOnDetail = !!options.matchOnDetail;
			input.matchOnLabel = (options.matchOnLabel === undefined) || options.matchOnLabel; // default to true
			input.quickNavigate = options.quickNavigate;
			input.hideInput = !!options.hideInput;
			input.contextKey = options.contextKey;
			input.busy = true;
			// Block Logic: Populates the QuickPick with items and sets the active item once promises resolve.
			Promise.all([picks, options.activeItem])
				.then(([items, _activeItem]) => {
					activeItem = _activeItem;
					input.busy = false;
					input.items = items;
					if (input.canSelectMany) {
						input.selectedItems = items.filter(item => item.type !== 'separator' && item.picked) as T[];
					}
					if (activeItem) {
						input.activeItems = [activeItem];
					}
				});
			// Block Logic: Displays the configured QuickPick to the user.
			input.show();
			// Block Logic: Handles potential errors during the promise resolution of `picks`, rejecting the main promise.
			Promise.resolve(picks).then(undefined, err => {
				reject(err);
				input.hide();
			});
		});
	}

	/**
	 * @brief Applies a validation result to an input box, setting its severity and validation message.
	 * Functional Utility: Standardizes the display of validation feedback (errors, warnings, info)
	 *                     within an {@link IInputBox}.
	 * @param input The {@link IInputBox} to apply the validation to.
	 * @param validationResult The validation result, which can be a string (error message),
	 *                         an object with content and severity, or null/undefined for no validation.
	 * Pre-condition: `input` is a valid `IInputBox` instance.
	 * Post-condition: The `input.severity` and `input.validationMessage` properties are updated
	 *                 based on the provided `validationResult`.
	 */
	private setValidationOnInput(input: IInputBox, validationResult: string | {
		content: string;
		severity: Severity;
	} | null | undefined) {
		if (validationResult && isString(validationResult)) {
			input.severity = Severity.Error;
			input.validationMessage = validationResult;
		} else if (validationResult && !isString(validationResult)) {
			input.severity = validationResult.severity;
			input.validationMessage = validationResult.content;
		} else {
			input.severity = Severity.Ignore;
			input.validationMessage = undefined;
		}
	}

	input(options: IInputOptions = {}, token: CancellationToken = CancellationToken.None): Promise<string | undefined> {
		return new Promise<string | undefined>((resolve) => {
			if (token.isCancellationRequested) {
				resolve(undefined);
				return;
			}
			const input = this.createInputBox();
			const validateInput = options.validateInput || (() => <Promise<undefined>>Promise.resolve(undefined));
			const onDidValueChange = Event.debounce(input.onDidChangeValue, (last, cur) => cur, 100);
			let validationValue = options.value || '';
			let validation = Promise.resolve(validateInput(validationValue));
			const disposables = [
				input,
				onDidValueChange(value => {
					if (value !== validationValue) {
						validation = Promise.resolve(validateInput(value));
						validationValue = value;
					}
					validation.then(result => {
						if (value === validationValue) {
							this.setValidationOnInput(input, result);
						}
					});
				}),
				input.onDidAccept(() => {
					const value = input.value;
					if (value !== validationValue) {
						validation = Promise.resolve(validateInput(value));
						validationValue = value;
					}
					validation.then(result => {
						if (!result || (!isString(result) && result.severity !== Severity.Error)) {
							resolve(value);
							input.hide();
						} else if (value === validationValue) {
							this.setValidationOnInput(input, result);
						}
					});
				}),
				token.onCancellationRequested(() => {
					input.hide();
				}),
				input.onDidHide(() => {
					dispose(disposables);
					resolve(undefined);
				}),
			];

			input.title = options.title;
			input.value = options.value || '';
			input.valueSelection = options.valueSelection;
			input.prompt = options.prompt;
			input.placeholder = options.placeHolder;
			input.password = !!options.password;
			input.ignoreFocusOut = !!options.ignoreFocusLost;
			input.show();
		});
	}

	backButton = backButton;

	createQuickPick<T extends IQuickPickItem>(options: { useSeparators: true }): IQuickPick<T, { useSeparators: true }>;
	createQuickPick<T extends IQuickPickItem>(options?: { useSeparators: boolean }): IQuickPick<T, { useSeparators: false }>;
	createQuickPick<T extends IQuickPickItem>(options: { useSeparators: boolean } = { useSeparators: false }): IQuickPick<T, { useSeparators: boolean }> {
		const ui = this.getUI(true);
		return new QuickPick<T, typeof options>(ui);
	}

	createInputBox(): IInputBox {
		const ui = this.getUI(true);
		return new InputBox(ui);
	}

	setAlignment(alignment: 'top' | 'center' | { top: number; left: number }): void {
		this.dndController?.setAlignment(alignment);
	}

	createQuickWidget(): IQuickWidget {
		const ui = this.getUI(true);
		return new QuickWidget(ui);
	}

	private show(controller: IQuickInput) {
		const ui = this.getUI(true);
		this.onShowEmitter.fire();
		const oldController = this.controller;
		this.controller = controller;
		oldController?.didHide();

		this.setEnabled(true);
		ui.leftActionBar.clear();
		ui.title.textContent = '';
		ui.description1.textContent = '';
		ui.description2.textContent = '';
		dom.reset(ui.widget);
		ui.rightActionBar.clear();
		ui.inlineActionBar.clear();
		ui.checkAll.checked = false;
		// ui.inputBox.value = ''; Avoid triggering an event.
		ui.inputBox.placeholder = '';
		ui.inputBox.password = false;
		ui.inputBox.showDecoration(Severity.Ignore);
		ui.visibleCount.setCount(0);
		ui.count.setCount(0);
		dom.reset(ui.message);
		ui.progressBar.stop();
		ui.list.setElements([]);
		ui.list.matchOnDescription = false;
		ui.list.matchOnDetail = false;
		ui.list.matchOnLabel = true;
		ui.list.sortByLabel = true;
		ui.ignoreFocusOut = false;
		ui.inputBox.toggles = undefined;

		const backKeybindingLabel = this.options.backKeybindingLabel();
		backButton.tooltip = backKeybindingLabel ? localize('quickInput.backWithKeybinding', "Back ({0})", backKeybindingLabel) : localize('quickInput.back', "Back");

		ui.container.style.display = '';
		this.updateLayout();
		this.dndController?.layoutContainer();
		ui.inputBox.setFocus();
		this.quickInputTypeContext.set(controller.type);
	}

	isVisible(): boolean {
		return !!this.ui && this.ui.container.style.display !== 'none';
	}

	private setVisibilities(visibilities: Visibilities) {
		const ui = this.getUI();
		ui.title.style.display = visibilities.title ? '' : 'none';
		ui.description1.style.display = visibilities.description && (visibilities.inputBox || visibilities.checkAll) ? '' : 'none';
		ui.description2.style.display = visibilities.description && !(visibilities.inputBox || visibilities.checkAll) ? '' : 'none';
		ui.checkAll.style.display = visibilities.checkAll ? '' : 'none';
		ui.inputContainer.style.display = visibilities.inputBox ? '' : 'none';
		ui.filterContainer.style.display = visibilities.inputBox ? '' : 'none';
		ui.visibleCountContainer.style.display = visibilities.visibleCount ? '' : 'none';
		ui.countContainer.style.display = visibilities.count ? '' : 'none';
		ui.okContainer.style.display = visibilities.ok ? '' : 'none';
		ui.customButtonContainer.style.display = visibilities.customButton ? '' : 'none';
		ui.message.style.display = visibilities.message ? '' : 'none';
		ui.progressBar.getContainer().style.display = visibilities.progressBar ? '' : 'none';
		ui.list.displayed = !!visibilities.list;
		ui.container.classList.toggle('show-checkboxes', !!visibilities.checkBox);
		ui.container.classList.toggle('hidden-input', !visibilities.inputBox && !visibilities.description);
		this.updateLayout(); // TODO
	}

	private setEnabled(enabled: boolean) {
		if (enabled !== this.enabled) {
			this.enabled = enabled;
			for (const item of this.getUI().leftActionBar.viewItems) {
				(item as ActionViewItem).action.enabled = enabled;
			}
			for (const item of this.getUI().rightActionBar.viewItems) {
				(item as ActionViewItem).action.enabled = enabled;
			}
			this.getUI().checkAll.disabled = !enabled;
			this.getUI().inputBox.enabled = enabled;
			this.getUI().ok.enabled = enabled;
			this.getUI().list.enabled = enabled;
		}
	}

	hide(reason?: QuickInputHideReason) {
		const controller = this.controller;
		if (!controller) {
			return;
		}
		controller.willHide(reason);

		const container = this.ui?.container;
		const focusChanged = container && !dom.isAncestorOfActiveElement(container);
		this.controller = null;
		this.onHideEmitter.fire();
		if (container) {
			container.style.display = 'none';
		}
		if (!focusChanged) {
			let currentElement = this.previousFocusElement;
			while (currentElement && !currentElement.offsetParent) {
				currentElement = currentElement.parentElement ?? undefined;
			}
			if (currentElement?.offsetParent) {
				currentElement.focus();
				this.previousFocusElement = undefined;
			} else {
				this.options.returnFocus();
			}
		}
		controller.didHide(reason);
	}

	focus() {
		if (this.isVisible()) {
			const ui = this.getUI();
			if (ui.inputBox.enabled) {
				ui.inputBox.setFocus();
			} else {
				ui.list.domFocus();
			}
		}
	}

	toggle() {
		if (this.isVisible() && this.controller instanceof QuickPick && this.controller.canSelectMany) {
			this.getUI().list.toggleCheckbox();
		}
	}

	toggleHover() {
		if (this.isVisible() && this.controller instanceof QuickPick) {
			this.getUI().list.toggleHover();
		}
	}

	navigate(next: boolean, quickNavigate?: IQuickNavigateConfiguration) {
		if (this.isVisible() && this.getUI().list.displayed) {
			this.getUI().list.focus(next ? QuickPickFocus.Next : QuickPickFocus.Previous);
			if (quickNavigate && this.controller instanceof QuickPick) {
				this.controller.quickNavigate = quickNavigate;
			}
		}
	}

	async accept(keyMods: IKeyMods = { alt: false, ctrlCmd: false }) {
		// When accepting the item programmatically, it is important that
		// we update `keyMods` either from the provided set or unset it
		// because the accept did not happen from mouse or keyboard
		// interaction on the list itself
		this.keyMods.alt = keyMods.alt;
		this.keyMods.ctrlCmd = keyMods.ctrlCmd;

		this.onDidAcceptEmitter.fire();
	}

	async back() {
		this.onDidTriggerButtonEmitter.fire(this.backButton);
	}

	async cancel() {
		this.hide();
	}

	layout(dimension: dom.IDimension, titleBarOffset: number): void {
		this.dimension = dimension;
		this.titleBarOffset = titleBarOffset;
		this.updateLayout();
	}

	private updateLayout() {
		if (this.ui && this.isVisible()) {
			const style = this.ui.container.style;
			const width = Math.min(this.dimension!.width * 0.62 /* golden cut */, QuickInputController.MAX_WIDTH);
			style.width = width + 'px';

			// Position
			style.top = `${this.viewState?.top ? Math.round(this.dimension!.height * this.viewState.top) : this.titleBarOffset}px`;
			style.left = `${Math.round((this.dimension!.width * (this.viewState?.left ?? 0.5 /* center */)) - (width / 2))}px`;

			this.ui.inputBox.layout();
			this.ui.list.layout(this.dimension && this.dimension.height * 0.4);
		}
	}

	applyStyles(styles: IQuickInputStyles) {
		this.styles = styles;
		this.updateStyles();
	}

	private updateStyles() {
		if (this.ui) {
			const {
				quickInputTitleBackground, quickInputBackground, quickInputForeground, widgetBorder, widgetShadow,
			} = this.styles.widget;
			this.ui.titleBar.style.backgroundColor = quickInputTitleBackground ?? '';
			this.ui.container.style.backgroundColor = quickInputBackground ?? '';
			this.ui.container.style.color = quickInputForeground ?? '';
			this.ui.container.style.border = widgetBorder ? `1px solid ${widgetBorder}` : '';
			this.ui.container.style.boxShadow = widgetShadow ? `0 0 8px 2px ${widgetShadow}` : '';
			this.ui.list.style(this.styles.list);

			const content: string[] = [];
			if (this.styles.pickerGroup.pickerGroupBorder) {
				content.push(`.quick-input-list .quick-input-list-entry { border-top-color:  ${this.styles.pickerGroup.pickerGroupBorder}; }`);
			}
			if (this.styles.pickerGroup.pickerGroupForeground) {
				content.push(`.quick-input-list .quick-input-list-separator { color:  ${this.styles.pickerGroup.pickerGroupForeground}; }`);
			}
			if (this.styles.pickerGroup.pickerGroupForeground) {
				content.push(`.quick-input-list .quick-input-list-separator-as-item { color: var(--vscode-descriptionForeground); }`);
			}

			if (this.styles.keybindingLabel.keybindingLabelBackground ||
				this.styles.keybindingLabel.keybindingLabelBorder ||
				this.styles.keybindingLabel.keybindingLabelBottomBorder ||
				this.styles.keybindingLabel.keybindingLabelShadow ||
				this.styles.keybindingLabel.keybindingLabelForeground) {
				content.push('.quick-input-list .monaco-keybinding > .monaco-keybinding-key {');
				if (this.styles.keybindingLabel.keybindingLabelBackground) {
					content.push(`background-color: ${this.styles.keybindingLabel.keybindingLabelBackground};`);
				}
				if (this.styles.keybindingLabel.keybindingLabelBorder) {
					// Order matters here. `border-color` must come before `border-bottom-color`.
					content.push(`border-color: ${this.styles.keybindingLabel.keybindingLabelBorder};`);
				}
				if (this.styles.keybindingLabel.keybindingLabelBottomBorder) {
					content.push(`border-bottom-color: ${this.styles.keybindingLabel.keybindingLabelBottomBorder};`);
				}
				if (this.styles.keybindingLabel.keybindingLabelShadow) {
					content.push(`box-shadow: inset 0 -1px 0 ${this.styles.keybindingLabel.keybindingLabelShadow};`);
				}
				if (this.styles.keybindingLabel.keybindingLabelForeground) {
					content.push(`color: ${this.styles.keybindingLabel.keybindingLabelForeground};`);
				}
				content.push('}');
			}

			const newStyles = content.join('\n');
			if (newStyles !== this.ui.styleSheet.textContent) {
				this.ui.styleSheet.textContent = newStyles;
			}
		}
	}

	private loadViewState(): QuickInputViewState | undefined {
		try {
			const data = JSON.parse(this.storageService.get(VIEWSTATE_STORAGE_KEY, StorageScope.APPLICATION, '{}'));
			if (data.top !== undefined || data.left !== undefined) {
				return data;
			}
		} catch { }

		return undefined;
	}

	private saveViewState(viewState: QuickInputViewState | undefined): void {
		const isMainWindow = this.layoutService.activeContainer === this.layoutService.mainContainer;
		if (!isMainWindow) {
			return;
		}

		if (viewState !== undefined) {
			this.storageService.store(VIEWSTATE_STORAGE_KEY, JSON.stringify(viewState), StorageScope.APPLICATION, StorageTarget.MACHINE);
		} else {
			this.storageService.remove(VIEWSTATE_STORAGE_KEY, StorageScope.APPLICATION);
		}
	}
}

export interface IQuickInputControllerHost extends ILayoutService { }

class QuickInputDragAndDropController extends Disposable {
	readonly dndViewState = observableValue<{ top?: number; left?: number; done: boolean } | undefined>(this, undefined);

	private readonly _snapThreshold = 20;
	private readonly _snapLineHorizontalRatio = 0.25;

	private readonly _controlsOnLeft: boolean;
	private readonly _controlsOnRight: boolean;

	private _quickInputAlignmentContext = QuickInputAlignmentContextKey.bindTo(this._contextKeyService);

	constructor(
		private _container: HTMLElement,
		private readonly _quickInputContainer: HTMLElement,
		private _quickInputDragAreas: { node: HTMLElement; includeChildren: boolean }[],
		initialViewState: QuickInputViewState | undefined,
		@ILayoutService private readonly _layoutService: ILayoutService,
		@IContextKeyService private readonly _contextKeyService: IContextKeyService,
		@IConfigurationService private readonly configurationService: IConfigurationService
	) {
		super();
		const customTitleControls = getTitleControlsStyle(this.configurationService) === TitleControlsStyle.CUSTOM;

		// Do not allow the widget to overflow or underflow window controls.
		// Use CSS calculations to avoid having to force layout with `.clientWidth`
		this._controlsOnLeft = customTitleControls && platform === Platform.Mac;
		this._controlsOnRight = customTitleControls && (platform === Platform.Windows || platform === Platform.Linux);
		this._registerLayoutListener();
		this.registerMouseListeners();
		this.dndViewState.set({ ...initialViewState, done: true }, undefined);
	}

	reparentUI(container: HTMLElement): void {
		this._container = container;
	}

	layoutContainer(dimension = this._layoutService.activeContainerDimension): void {
		const state = this.dndViewState.get();
		const dragAreaRect = this._quickInputContainer.getBoundingClientRect();
		if (state?.top && state?.left) {
			const a = Math.round(state.left * 1e2) / 1e2;
			const b = dimension.width;
			const c = dragAreaRect.width;
			const d = a * b - c / 2;
			this._layout(state.top * dimension.height, d);
		}
	}

	setAlignment(alignment: 'top' | 'center' | { top: number; left: number }, done = true): void {
		if (alignment === 'top') {
			this.dndViewState.set({
				top: this._getTopSnapValue() / this._container.clientHeight,
				left: (this._getCenterXSnapValue() + (this._quickInputContainer.clientWidth / 2)) / this._container.clientWidth,
				done
			}, undefined);
			this._quickInputAlignmentContext.set('top');
		} else if (alignment === 'center') {
			this.dndViewState.set({
				top: this._getCenterYSnapValue() / this._container.clientHeight,
				left: (this._getCenterXSnapValue() + (this._quickInputContainer.clientWidth / 2)) / this._container.clientWidth,
				done
			}, undefined);
			this._quickInputAlignmentContext.set('center');
		} else {
			this.dndViewState.set({ top: alignment.top, left: alignment.left, done }, undefined);
			this._quickInputAlignmentContext.set(undefined);
		}
	}

	private _registerLayoutListener() {
		this._register(Event.filter(this._layoutService.onDidLayoutContainer, e => e.container === this._container)((e) => this.layoutContainer(e.dimension)));
	}

	private registerMouseListeners(): void {
		const dragArea = this._quickInputContainer;

		// Double click
		this._register(dom.addDisposableGenericMouseUpListener(dragArea, (event: MouseEvent) => {
			const originEvent = new StandardMouseEvent(dom.getWindow(dragArea), event);
			if (originEvent.detail !== 2) {
				return;
			}

			// Ignore event if the target is not the drag area
			if (!this._quickInputDragAreas.some(({ node, includeChildren }) => includeChildren ? dom.isAncestor(originEvent.target as HTMLElement, node) : originEvent.target === node)) {
				return;
			}

			this.dndViewState.set({ top: undefined, left: undefined, done: true }, undefined);
		}));

		// Mouse down
		this._register(dom.addDisposableGenericMouseDownListener(dragArea, (e: MouseEvent) => {
			const activeWindow = dom.getWindow(this._layoutService.activeContainer);
			const originEvent = new StandardMouseEvent(activeWindow, e);

			// Ignore event if the target is not the drag area
			if (!this._quickInputDragAreas.some(({ node, includeChildren }) => includeChildren ? dom.isAncestor(originEvent.target as HTMLElement, node) : originEvent.target === node)) {
				return;
			}

			// Mouse position offset relative to dragArea
			const dragAreaRect = this._quickInputContainer.getBoundingClientRect();
			const dragOffsetX = originEvent.browserEvent.clientX - dragAreaRect.left;
			const dragOffsetY = originEvent.browserEvent.clientY - dragAreaRect.top;

			let isMovingQuickInput = false;
			const mouseMoveListener = dom.addDisposableGenericMouseMoveListener(activeWindow, (e: MouseEvent) => {
				const mouseMoveEvent = new StandardMouseEvent(activeWindow, e);
				mouseMoveEvent.preventDefault();

				if (!isMovingQuickInput) {
					isMovingQuickInput = true;
				}

				this._layout(e.clientY - dragOffsetY, e.clientX - dragOffsetX);
			});
			const mouseUpListener = dom.addDisposableGenericMouseUpListener(activeWindow, (e: MouseEvent) => {
				if (isMovingQuickInput) {
					// Save position
					const state = this.dndViewState.get();
					this.dndViewState.set({ top: state?.top, left: state?.left, done: true }, undefined);
				}

				// Dispose listeners
				mouseMoveListener.dispose();
				mouseUpListener.dispose();
			});
		}));
	}

	private _layout(topCoordinate: number, leftCoordinate: number) {
		const snapCoordinateYTop = this._getTopSnapValue();
		const snapCoordinateY = this._getCenterYSnapValue();
		const snapCoordinateX = this._getCenterXSnapValue();
		// Make sure the quick input is not moved outside the container
		topCoordinate = Math.max(0, Math.min(topCoordinate, this._container.clientHeight - this._quickInputContainer.clientHeight));

		if (topCoordinate < this._layoutService.activeContainerOffset.top) {
			if (this._controlsOnLeft) {
				leftCoordinate = Math.max(leftCoordinate, 80 / getZoomFactor(dom.getActiveWindow()));
			} else if (this._controlsOnRight) {
				leftCoordinate = Math.min(leftCoordinate, this._container.clientWidth - this._quickInputContainer.clientWidth - (140 / getZoomFactor(dom.getActiveWindow())));
			}
		}

		const snappingToTop = Math.abs(topCoordinate - snapCoordinateYTop) < this._snapThreshold;
		topCoordinate = snappingToTop ? snapCoordinateYTop : topCoordinate;
		const snappingToCenter = Math.abs(topCoordinate - snapCoordinateY) < this._snapThreshold;
		topCoordinate = snappingToCenter ? snapCoordinateY : topCoordinate;
		const top = topCoordinate / this._container.clientHeight;

		// Make sure the quick input is not moved outside the container
		leftCoordinate = Math.max(0, Math.min(leftCoordinate, this._container.clientWidth - this._quickInputContainer.clientWidth));
		const snappingToCenterX = Math.abs(leftCoordinate - snapCoordinateX) < this._snapThreshold;
		leftCoordinate = snappingToCenterX ? snapCoordinateX : leftCoordinate;

		const b = this._container.clientWidth;
		const c = this._quickInputContainer.clientWidth;
		const d = leftCoordinate;
		const left = (d + c / 2) / b;

		this.dndViewState.set({ top, left, done: false }, undefined);
		if (snappingToCenterX) {
			if (snappingToTop) {
				this._quickInputAlignmentContext.set('top');
				return;
			} else if (snappingToCenter) {
				this._quickInputAlignmentContext.set('center');
				return;
			}
		}
		this._quickInputAlignmentContext.set(undefined);
	}

	private _getTopSnapValue() {
		return this._layoutService.activeContainerOffset.quickPickTop;
	}

	private _getCenterYSnapValue() {
		return Math.round(this._container.clientHeight * this._snapLineHorizontalRatio);
	}

	private _getCenterXSnapValue() {
		return Math.round(this._container.clientWidth / 2) - Math.round(this._quickInputContainer.clientWidth / 2);
	}
}
