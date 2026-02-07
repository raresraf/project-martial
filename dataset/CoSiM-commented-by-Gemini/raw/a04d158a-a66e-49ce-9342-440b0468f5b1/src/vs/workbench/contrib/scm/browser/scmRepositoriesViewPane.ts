/**
 * @file scmRepositoriesViewPane.ts
 * @brief Implements the Source Control Management (SCM) repositories view pane in Visual Studio Code.
 * This module is responsible for rendering, managing, and interacting with the list of SCM repositories,
 * providing a user interface to display and interact with various source control providers.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import './media/scm.css';
import { localize } from '../../../../nls.js';
import { Event } from '../../../../base/common/event.js';
import { ViewPane, IViewPaneOptions } from '../../../browser/parts/views/viewPane.js';
import { append, $ } from '../../../../base/browser/dom.js';
import { IListVirtualDelegate, IListContextMenuEvent, IListEvent, IListRenderer } from '../../../../base/browser/ui/list/list.js';
import { ISCMRepository, ISCMViewService } from '../common/scm.js';
import { IInstantiationService } from '../../../../platform/instantiation/common/instantiation.js';
import { IContextMenuService } from '../../../../platform/contextview/browser/contextView.js';
import { IContextKeyService } from '../../../../platform/contextkey/common/contextkey.js';
import { IKeybindingService } from '../../../../platform/keybinding/common/keybinding.js';
import { IThemeService } from '../../../../platform/theme/common/themeService.js';
import { WorkbenchList } from '../../../../platform/list/browser/listService.js';
import { IConfigurationService } from '../../../../platform/configuration/common/configuration.js';
import { IViewDescriptorService } from '../../../common/views.js';
import { IOpenerService } from '../../../../platform/opener/common/opener.js';
import { RepositoryActionRunner, RepositoryRenderer } from './scmRepositoryRenderer.js';
import { collectContextMenuActions, getActionViewItemProvider } from './util.js';
import { Orientation } from '../../../../base/browser/ui/sash/sash.js';
import { Iterable } from '../../../../base/common/iterator.js';
import { DisposableStore } from '../../../../base/common/lifecycle.js';
import { MenuId } from '../../../../platform/actions/common/actions.js';
import { IHoverService } from '../../../../platform/hover/browser/hover.js';

/**
 * @brief Implements `IListVirtualDelegate` to provide sizing and template information for the SCM repositories list.
 * This delegate ensures that each repository item in the list is rendered consistently.
 */
class ListDelegate implements IListVirtualDelegate<ISCMRepository> {

	/**
	 * @brief Returns the height of each item in the list.
	 * @return The fixed height of 22 pixels for each repository item.
	 */
	getHeight(): number {
		return 22;
	}

	/**
	 * @brief Returns the template ID for rendering list items.
	 * @return The `TEMPLATE_ID` from `RepositoryRenderer`, which identifies the renderer responsible for SCM repository items.
	 */
	getTemplateId(): string {
		return RepositoryRenderer.TEMPLATE_ID;
	}
}

/**
 * @brief Represents the main view pane for displaying Source Control Management (SCM) repositories.
 * This class extends `ViewPane` and manages the lifecycle, rendering, and interactions
 * of a list of SCM repositories within the VS Code workbench.
 */
export class SCMRepositoriesViewPane extends ViewPane {

	private list!: WorkbenchList<ISCMRepository>;
	private readonly disposables = new DisposableStore();

	/**
	 * @brief Constructs a new `SCMRepositoriesViewPane` instance.
	 * Initializes the view pane with necessary services for SCM interaction, UI rendering,
	 * context menus, keybindings, and configuration.
	 * @param options The view pane options inherited from the base class.
	 * @param scmViewService The SCM view service for managing SCM-related UI state and data.
	 * @param keybindingService The keybinding service for handling keyboard shortcuts.
	 * @param contextMenuService The context menu service for displaying context-sensitive menus.
	 * @param instantiationService The instantiation service for creating new service instances.
	 * @param viewDescriptorService The view descriptor service for managing view registrations.
	 * @param contextKeyService The context key service for managing UI context keys.
	 * @param configurationService The configuration service for accessing user settings.
	 * @param openerService The opener service for opening URIs.
	 * @param themeService The theme service for applying UI themes.
	 * @param hoverService The hover service for displaying hover widgets.
	 */
	constructor(
		options: IViewPaneOptions,
		@ISCMViewService protected scmViewService: ISCMViewService,
		@IKeybindingService keybindingService: IKeybindingService,
		@IContextMenuService contextMenuService: IContextMenuService,
		@IInstantiationService instantiationService: IInstantiationService,
		@IViewDescriptorService viewDescriptorService: IViewDescriptorService,
		@IContextKeyService contextKeyService: IContextKeyService,
		@IConfigurationService configurationService: IConfigurationService,
		@IOpenerService openerService: IOpenerService,
		@IThemeService themeService: IThemeService,
		@IHoverService hoverService: IHoverService
	) {
		super({ ...options, titleMenuId: MenuId.SCMSourceControlTitle }, keybindingService, contextMenuService, configurationService, contextKeyService, viewDescriptorService, instantiationService, openerService, themeService, hoverService);
	}

	/**
	 * @brief Renders the main body of the SCM repositories view pane.
	 * This method initializes the list UI component, sets up rendering delegates and
	 * renderers, configures accessibility, and registers event listeners for SCM
	 * repository changes and configuration updates. It also manages the visibility
	 * of provider count badges based on user settings.
	 * @param container The HTML element into which the view pane's body content will be rendered.
	 * Pre-condition: The `container` element must be a valid DOM element ready to receive content.
	 * Post-condition: The SCM repositories list is initialized and bound to SCM service events.
	 */
	protected override renderBody(container: HTMLElement): void {
		super.renderBody(container);

		const listContainer = append(container, $('.scm-view.scm-repositories-view'));

		// Block Logic: Manages the visibility of provider count badges based on the 'scm.providerCountBadge' configuration setting.
		// It toggles CSS classes to hide or automatically display the counts.
		const updateProviderCountVisibility = () => {
			const value = this.configurationService.getValue<'hidden' | 'auto' | 'visible'>('scm.providerCountBadge');
			listContainer.classList.toggle('hide-provider-counts', value === 'hidden');
			listContainer.classList.toggle('auto-provider-counts', value === 'auto');
		};
		this._register(Event.filter(this.configurationService.onDidChangeConfiguration, e => e.affectsConfiguration('scm.providerCountBadge'), this.disposables)(updateProviderCountVisibility));
		updateProviderCountVisibility();

		const delegate = new ListDelegate();
		const renderer = this.instantiationService.createInstance(RepositoryRenderer, MenuId.SCMSourceControlInline, getActionViewItemProvider(this.instantiationService));
		const identityProvider = { getId: (r: ISCMRepository) => r.provider.id };

		// Block Logic: Initializes the WorkbenchList component which is responsible for rendering the SCM repositories.
		// It configures the list with the necessary delegate, renderer, identity provider, and accessibility settings.
		this.list = this.instantiationService.createInstance(WorkbenchList, `SCM Main`, listContainer, delegate, [renderer as IListRenderer<ISCMRepository, any>], {
			identityProvider,
			horizontalScrolling: false,
			overrideStyles: this.getLocationBasedColors().listOverrideStyles,
			accessibilityProvider: {
				getAriaLabel(r: ISCMRepository) {
					return r.provider.label;
				},
				getWidgetAriaLabel() {
					return localize('scm', "Source Control Repositories");
				}
			}
		}) as WorkbenchList<ISCMRepository>;

		this._register(this.list);
		// Block Logic: Registers event listeners for list selection, focus, and context menu events to enable user interaction.
		this._register(this.list.onDidChangeSelection(this.onListSelectionChange, this));
		this._register(this.list.onDidChangeFocus(this.onDidChangeFocus, this));
		this._register(this.list.onContextMenu(this.onListContextMenu, this));

		// Block Logic: Registers event listeners for SCM service changes to dynamically update the list of repositories.
		this._register(this.scmViewService.onDidChangeRepositories(this.onDidChangeRepositories, this));
		this._register(this.scmViewService.onDidChangeVisibleRepositories(this.updateListSelection, this));

		// Block Logic: If the view pane orientation is vertical, listens for configuration changes
		// that affect the visibility of SCM repositories to re-evaluate the body size.
		if (this.orientation === Orientation.VERTICAL) {
			this._register(this.configurationService.onDidChangeConfiguration(e => {
				if (e.affectsConfiguration('scm.repositories.visible')) {
					this.updateBodySize();
				}
			}));
		}

		this.onDidChangeRepositories();
		this.updateListSelection();
	}

	/**
	 * @brief Handles changes in the SCM repositories.
	 * Updates the internal list model with the latest repositories from the SCM view service
	 * and recalculates the view pane's body size.
	 * Post-condition: The list UI is synchronized with the current set of SCM repositories.
	 */
	private onDidChangeRepositories(): void {
		this.list.splice(0, this.list.length, this.scmViewService.repositories);
		this.updateBodySize();
	}

	/**
	 * @brief Sets the UI focus to the SCM repositories list.
	 * This ensures that keyboard navigation and other focus-related interactions
	 * are directed to the repository list.
	 * Post-condition: The `WorkbenchList` component has DOM focus.
	 */
	override focus(): void {
		super.focus();
		this.list.domFocus();
	}

	/**
	 * @brief Lays out the body of the SCM repositories view pane.
	 * Specifically, it forwards the layout request to the internal `WorkbenchList` component
	 * to ensure the list is rendered with the correct dimensions.
	 * @param height The available height for the body.
	 * @param width The available width for the body.
	 */
	protected override layoutBody(height: number, width: number): void {
		super.layoutBody(height, width);
		this.list.layout(height, width);
	}

	/**
	 * @brief Updates the minimum and maximum body size of the view pane.
	 * This is particularly relevant when the view pane is oriented vertically
	 * and the number of visible SCM repositories is configured, allowing the pane
	 * to dynamically resize.
	 * Pre-condition: `this.orientation` must be `Orientation.VERTICAL` for this method to have an effect on size calculation beyond initial values.
	 * Post-condition: `this.minimumBodySize` and `this.maximumBodySize` are updated based on configuration and list content.
	 */
	private updateBodySize(): void {
		// Pre-condition: This update is primarily relevant for vertical orientation.
		if (this.orientation === Orientation.HORIZONTAL) {
			return;
		}

		const visibleCount = this.configurationService.getValue<number>('scm.repositories.visible');
		const empty = this.list.length === 0;
		// Functional Utility: Calculates the desired size based on the number of visible repositories and a fixed item height.
		const size = Math.min(this.list.length, visibleCount) * 22;

		// Invariant: If `visibleCount` is 0, the pane can expand infinitely. Otherwise, it's constrained by the calculated size.
		this.minimumBodySize = visibleCount === 0 ? 22 : size;
		this.maximumBodySize = visibleCount === 0 ? Number.POSITIVE_INFINITY : empty ? Number.POSITIVE_INFINITY : size;
	}

	/**
	 * @brief Handles context menu events triggered on items within the SCM repositories list.
	 * It retrieves context-specific actions for the selected SCM provider and displays
	 * a context menu, allowing users to perform actions related to that repository.
	 * @param e The `IListContextMenuEvent` containing information about the context menu invocation.
	 * Pre-condition: `e.element` must be defined, indicating that a valid list element was right-clicked.
	 * Post-condition: A context menu is displayed with actions relevant to the selected SCM repository.
	 */
	private onListContextMenu(e: IListContextMenuEvent<ISCMRepository>): void {
		if (!e.element) {
			return;
		}

		const provider = e.element.provider;
		const menus = this.scmViewService.menus.getRepositoryMenus(provider);
		const menu = menus.repositoryContextMenu;
		const actions = collectContextMenuActions(menu);

		const actionRunner = new RepositoryActionRunner(() => {
			return this.list.getSelectedElements();
		});
		actionRunner.onWillRun(() => this.list.domFocus());

		this.contextMenuService.showContextMenu({
			actionRunner,
			getAnchor: () => e.anchor,
			getActions: () => actions,
			getActionsContext: () => provider,
			onHide: () => actionRunner.dispose()
		});
	}

	/**
	 * @brief Responds to changes in the selection of the SCM repositories list.
	 * When the selection changes (e.g., due to user interaction), this method updates
	 * the `scmViewService.visibleRepositories` to reflect the newly selected items
	 * and preserves the scroll position of the list.
	 * @param e The `IListEvent` containing information about the selection change.
	 * Pre-condition: `e.browserEvent` must be present, indicating a user-initiated change.
	 * Post-condition: `scmViewService.visibleRepositories` is updated, and list scroll position is maintained.
	 */
	private onListSelectionChange(e: IListEvent<ISCMRepository>): void {
		if (e.browserEvent && e.elements.length > 0) {
			const scrollTop = this.list.scrollTop;
			this.scmViewService.visibleRepositories = e.elements;
			this.list.scrollTop = scrollTop;
		}
	}

	/**
	 * @brief Handles changes in the focus of the SCM repositories list.
	 * When a list item gains focus, this method informs the `scmViewService` to
	 * focus on the corresponding SCM repository.
	 * @param e The `IListEvent` containing information about the focus change.
	 * Pre-condition: `e.browserEvent` must be present, indicating a user-initiated change.
	 * Post-condition: The `scmViewService` is instructed to focus on the first focused element.
	 */
	private onDidChangeFocus(e: IListEvent<ISCMRepository>): void {
		if (e.browserEvent && e.elements.length > 0) {
			this.scmViewService.focus(e.elements[0]);
		}
	}

	/**
	 * @brief Synchronizes the selection state of the internal `WorkbenchList` with the
	 * `scmViewService.visibleRepositories`.
	 * This method calculates which repositories have been added or removed from the visible
	 * set and updates the list's selection accordingly.
	 * Post-condition: The `WorkbenchList`'s selection accurately reflects `scmViewService.visibleRepositories`.
	 */
	private updateListSelection(): void {
		const oldSelection = this.list.getSelection();
		const oldSet = new Set(Iterable.map(oldSelection, i => this.list.element(i)));
		const set = new Set(this.scmViewService.visibleRepositories);
		const added = new Set(Iterable.filter(set, r => !oldSet.has(r)));
		const removed = new Set(Iterable.filter(oldSet, r => !set.has(r)));

		if (added.size === 0 && removed.size === 0) {
			return;
		}

		const selection = oldSelection
			.filter(i => !removed.has(this.list.element(i)));

		// Block Logic: Iterates through all list elements to add newly visible repositories to the selection.
		for (let i = 0; i < this.list.length; i++) {
			if (added.has(this.list.element(i))) {
				selection.push(i);
			}
		}

		this.list.setSelection(selection);

		// Block Logic: Adjusts the list's focus and anchor if the selection changed and the focused element is no longer selected.
		if (selection.length > 0 && selection.indexOf(this.list.getFocus()[0]) === -1) {
			this.list.setAnchor(selection[0]);
			this.list.setFocus([selection[0]]);
		}
	}

	/**
	 * @brief Disposes of all resources owned by this view pane.
	 * This includes all disposables managed by the `disposables` `DisposableStore`.
	 * Post-condition: All event listeners and disposable resources are released.
	 */
	override dispose(): void {
		this.disposables.dispose();
		super.dispose();
	}}
