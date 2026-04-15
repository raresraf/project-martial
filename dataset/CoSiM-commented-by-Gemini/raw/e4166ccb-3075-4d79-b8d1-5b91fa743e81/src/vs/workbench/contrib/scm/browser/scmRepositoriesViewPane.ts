/**
 * @file scmRepositoriesViewPane.ts
 * @brief Defines the view pane for displaying Source Control Management (SCM) repositories in Visual Studio Code.
 *
 * This file contains the implementation of the `SCMRepositoriesViewPane`, a UI component
 * that renders a list or tree of available SCM repositories. It uses a `WorkbenchCompressibleAsyncDataTree`
 * to display the repositories and handles user interactions like selection, focus, and context menus.
 * It also responds to changes in the underlying SCM data model and configuration settings.
 */

/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import './media/scm.css';
import { localize } from '../../../../nls.js';
import { ViewPane, IViewPaneOptions } from '../../../browser/parts/views/viewPane.js';
import { append, $ } from '../../../../base/browser/dom.js';
import { IListVirtualDelegate, IIdentityProvider } from '../../../../base/browser/ui/list/list.js';
import { IAsyncDataSource, ITreeEvent, ITreeContextMenuEvent } from '../../../../base/browser/ui/tree/tree.js';
import { WorkbenchCompressibleAsyncDataTree } from '../../../../platform/list/browser/listService.js';
import { ISCMRepository, ISCMViewService } from '../common/scm.js';
import { IInstantiationService } from '../../../../platform/instantiation/common/instantiation.js';
import { IContextMenuService } from '../../../../platform/contextview/browser/contextView.js';
import { IContextKeyService } from '../../../../platform/contextkey/common/contextkey.js';
import { IKeybindingService } from '../../../../platform/keybinding/common/keybinding.js';
import { IThemeService } from '../../../../platform/theme/common/themeService.js';
import { Disposable, DisposableStore } from '../../../../base/common/lifecycle.js';
import { IConfigurationService } from '../../../../platform/configuration/common/configuration.js';
import { IViewDescriptorService } from '../../../common/views.js';
import { IOpenerService } from '../../../../platform/opener/common/opener.js';
import { RepositoryActionRunner, RepositoryRenderer } from './scmRepositoryRenderer.js';
import { collectContextMenuActions, getActionViewItemProvider, isSCMRepository } from './util.js';
import { Orientation } from '../../../../base/browser/ui/sash/sash.js';
import { Iterable } from '../../../../base/common/iterator.js';
import { MenuId } from '../../../../platform/actions/common/actions.js';
import { IHoverService } from '../../../../platform/hover/browser/hover.js';
import { observableConfigValue } from '../../../../platform/observable/common/platformObservableUtils.js';
import { autorun, IObservable, observableSignalFromEvent } from '../../../../base/common/observable.js';
import { Sequencer } from '../../../../base/common/async.js';

/**
 * @class ListDelegate
 * @implements {IListVirtualDelegate<ISCMRepository>}
 * @brief Provides layout information for the repository list.
 *
 * This class is responsible for telling the list view how to render each item,
 * specifically by providing the height and template ID for each repository.
 */
class ListDelegate implements IListVirtualDelegate<ISCMRepository> {

	/**
	 * @brief Gets the height of a single repository item in pixels.
	 * @returns The height of the item.
	 */
	getHeight(): number {
		return 22;
	}

	/**
	 * @brief Gets the template ID for the renderer of a repository item.
	 * @returns The template ID.
	 */
	getTemplateId(): string {
		return RepositoryRenderer.TEMPLATE_ID;
	}
}

/**
 * @class RepositoryTreeDataSource
 * @implements {IAsyncDataSource<ISCMViewService, ISCMRepository>}
 * @brief Provides the data and structure for the repository tree.
 *
 * This class acts as the data source for the tree view, fetching repository data
 * from the `ISCMViewService` and defining the parent-child relationships,
 * which allows for hierarchical display of repositories (e.g., submodules).
 */
class RepositoryTreeDataSource extends Disposable implements IAsyncDataSource<ISCMViewService, ISCMRepository> {
	constructor(@ISCMViewService private readonly scmViewService: ISCMViewService) {
		super();
	}

	/**
	 * @brief Gets the children of a given element.
	 * @param inputOrElement The input service or a parent repository.
	 * @returns An iterable of child repositories.
	 */
	getChildren(inputOrElement: ISCMViewService | ISCMRepository): Iterable<ISCMRepository> {
		const parentId = isSCMRepository(inputOrElement)
			? inputOrElement.provider.id
			: undefined;

		const repositories = this.scmViewService.repositories
			.filter(r => r.provider.parentId === parentId);

		return repositories;
	}

	/**
	 * @brief Gets the parent of a given repository.
	 * @param element The repository to get the parent of.
	 * @returns The parent repository or the view service if it's a root repository.
	 */
	getParent(element: ISCMViewService | ISCMRepository): ISCMViewService | ISCMRepository {
		if (!isSCMRepository(element)) {
			throw new Error('Unexpected call to getParent');
		}

		const repository = this.scmViewService.repositories
			.find(r => r.provider.id === element.provider.parentId);
		if (!repository) {
			throw new Error('Invalid element passed to getParent');
		}

		return repository;
	}

	/**
	 * @brief Checks if a given element has child repositories.
	 * @param inputOrElement The element to check.
	 * @returns `true` if the element has children, `false` otherwise.
	 */
	hasChildren(inputOrElement: ISCMViewService | ISCMRepository): boolean {
		const parentId = isSCMRepository(inputOrElement)
			? inputOrElement.provider.id
			: undefined;

		const repositories = this.scmViewService.repositories
			.filter(r => r.provider.parentId === parentId);

		return repositories.length > 0;
	}
}

/**
 * @class RepositoryTreeIdentityProvider
 * @implements {IIdentityProvider<ISCMRepository>}
 * @brief Provides a unique ID for each repository in the tree.
 *
 * This is crucial for the tree to maintain its state (e.g., selection, expansion)
 * across data updates. The provider ID is used as the unique identifier.
 */
class RepositoryTreeIdentityProvider implements IIdentityProvider<ISCMRepository> {
	getId(element: ISCMRepository): string {
		return element.provider.id;
	}
}

/**
 * @class SCMRepositoriesViewPane
 * @extends {ViewPane}
 * @brief A view pane that displays a list of SCM repositories.
 *
 * This class is responsible for rendering the SCM repositories view, handling user
 * interactions, and keeping the view in sync with the SCM service and user settings.
 */
export class SCMRepositoriesViewPane extends ViewPane {

	private tree!: WorkbenchCompressibleAsyncDataTree<ISCMViewService, ISCMRepository, any>;
	private treeDataSource!: RepositoryTreeDataSource;
	private treeIdentityProvider!: RepositoryTreeIdentityProvider;
	private readonly treeOperationSequencer = new Sequencer();

	private readonly visibleCountObs: IObservable<number>;
	private readonly providerCountBadgeObs: IObservable<'hidden' | 'auto' | 'visible'>;

	private readonly visibilityDisposables = new DisposableStore();

	constructor(
		options: IViewPaneOptions,
		@ISCMViewService private readonly scmViewService: ISCMViewService,
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

		// Observe configuration settings for the view.
		this.visibleCountObs = observableConfigValue('scm.repositories.visible', 10, this.configurationService);
		this.providerCountBadgeObs = observableConfigValue<'hidden' | 'auto' | 'visible'>('scm.providerCountBadge', 'hidden', this.configurationService);
	}

	/**
	 * @brief Renders the body of the view pane.
	 * @param container The parent HTML element.
	 *
	 * This method creates the tree view and sets up listeners to react to visibility
	 * changes and data updates.
	 */
	protected override renderBody(container: HTMLElement): void {
		super.renderBody(container);

		const treeContainer = append(container, $('.scm-view.scm-repositories-view'));

		// React to the `scm.providerCountBadge` setting to show/hide provider counts.
		this._register(autorun(reader => {
			const providerCountBadge = this.providerCountBadgeObs.read(reader);
			treeContainer.classList.toggle('hide-provider-counts', providerCountBadge === 'hidden');
			treeContainer.classList.toggle('auto-provider-counts', providerCountBadge === 'auto');
		}));

		this.createTree(treeContainer);

		// Set up logic to run when the view becomes visible.
		this.onDidChangeBodyVisibility(async visible => {
			if (!visible) {
				this.visibilityDisposables.clear();
				return;
			}

			// Use a sequencer to ensure tree operations run in order.
			this.treeOperationSequencer.queue(async () => {
				// Initial rendering of the tree with data from the SCM service.
				await this.tree.setInput(this.scmViewService);

				// React to the `scm.repositories.visible` setting to adjust the view size.
				this.visibilityDisposables.add(autorun(reader => {
					const visibleCount = this.visibleCountObs.read(reader);
					this.updateBodySize(this.tree.contentHeight, visibleCount);
				}));

				// Update the tree when the list of repositories changes.
				const onDidChangeRepositoriesSignal = observableSignalFromEvent(
					this, this.scmViewService.onDidChangeRepositories);

				this.visibilityDisposables.add(autorun(async reader => {
					onDidChangeRepositoriesSignal.read(reader);
					await this.treeOperationSequencer.queue(() => this.updateChildren());
				}));

				// Update the tree selection when the set of visible repositories changes.
				const onDidChangeVisibleRepositoriesSignal = observableSignalFromEvent(
					this, this.scmViewService.onDidChangeVisibleRepositories);

				this.visibilityDisposables.add(autorun(async reader => {
					onDidChangeVisibleRepositoriesSignal.read(reader);
					await this.treeOperationSequencer.queue(() => this.updateTreeSelection());
				}));
			});
		}, this, this._store);
	}

	/**
	 * @brief Lays out the body of the view pane.
	 * @param height The height of the view pane.
	 * @param width The width of the view pane.
	 *
	 * This method is called when the view is resized, and it adjusts the layout of the tree.
	 */
	protected override layoutBody(height: number, width: number): void {
		super.layoutBody(height, width);
		this.tree.layout(height, width);
	}

	/**
	 * @brief Sets focus on the view pane.
	 *
	 * This method is called to give focus to the repository tree.
	 */
	override focus(): void {
		super.focus();
		this.tree.domFocus();
	}

	/**
	 * @brief Creates and configures the repository tree.
	 * @param container The parent HTML element for the tree.
	 */
	private createTree(container: HTMLElement): void {
		this.treeIdentityProvider = new RepositoryTreeIdentityProvider();
		this.treeDataSource = this.instantiationService.createInstance(RepositoryTreeDataSource);
		this._register(this.treeDataSource);

		const compressionEnabled = observableConfigValue('scm.compactFolders', true, this.configurationService);

		this.tree = this.instantiationService.createInstance(
			WorkbenchCompressibleAsyncDataTree,
			'SCM Repositories',
			container,
			new ListDelegate(),
			{
				// Incompressible items are repositories that cannot be compacted into a single folder.
				isIncompressible: () => true
			},
			[
				this.instantiationService.createInstance(RepositoryRenderer, MenuId.SCMSourceControlInline, getActionViewItemProvider(this.instantiationService))
			],
			this.treeDataSource,
			{
				identityProvider: this.treeIdentityProvider,
				horizontalScrolling: false,
				compressionEnabled: compressionEnabled.get(),
				overrideStyles: this.getLocationBasedColors().listOverrideStyles,
				expandOnDoubleClick: false,
				expandOnlyOnTwistieClick: true,
				accessibilityProvider: {
					getAriaLabel(r: ISCMRepository) {
						return r.provider.label;
					},
					getWidgetAriaLabel() {
						return localize('scm', "Source Control Repositories");
					}
				}
			}
		) as WorkbenchCompressibleAsyncDataTree<ISCMViewService, ISCMRepository, any>;
		this._register(this.tree);

		// Register listeners for tree events.
		this._register(this.tree.onDidChangeSelection(this.onTreeSelectionChange, this));
		this._register(this.tree.onDidChangeFocus(this.onTreeDidChangeFocus, this));
		this._register(this.tree.onContextMenu(this.onTreeContextMenu, this));
		this._register(this.tree.onDidChangeContentHeight(this.onTreeContentHeightChange, this));
	}

	/**
	 * @brief Handles the context menu event on the tree.
	 * @param e The tree context menu event.
	 *
	 * This method collects the appropriate context menu actions for the selected
	 * repository and displays them.
	 */
	private onTreeContextMenu(e: ITreeContextMenuEvent<ISCMRepository>): void {
		if (!e.element) {
			return;
		}

		const provider = e.element.provider;
		const menus = this.scmViewService.menus.getRepositoryMenus(provider);
		const menu = menus.repositoryContextMenu;
		const actions = collectContextMenuActions(menu);

		const disposables = new DisposableStore();
		const actionRunner = new RepositoryActionRunner(() => {
			return this.tree.getSelection();
		});
		disposables.add(actionRunner);
		disposables.add(actionRunner.onWillRun(() => this.tree.domFocus()));

		this.contextMenuService.showContextMenu({
			actionRunner,
			getAnchor: () => e.anchor,
			getActions: () => actions,
			getActionsContext: () => provider,
			onHide: () => disposables.dispose()
		});
	}

	/**
	 * @brief Handles the selection change event on the tree.
	 * @param e The tree selection event.
	 *
	 * This method updates the `visibleRepositories` in the `ISCMViewService`
	 * to reflect the user's selection in the tree.
	 */
	private onTreeSelectionChange(e: ITreeEvent<ISCMRepository>): void {
		if (e.browserEvent && e.elements.length > 0) {
			const scrollTop = this.tree.scrollTop;
			this.scmViewService.visibleRepositories = e.elements;
			this.tree.scrollTop = scrollTop;
		}
	}

	/**
	 * @brief Handles the focus change event on the tree.
	 * @param e The tree focus event.
	 *
	 * This method informs the `ISCMViewService` which repository currently has focus.
	 */
	private onTreeDidChangeFocus(e: ITreeEvent<ISCMRepository>): void {
		if (e.browserEvent && e.elements.length > 0) {
			this.scmViewService.focus(e.elements[0]);
		}
	}

	/**
	 * @brief Handles the content height change event of the tree.
	 * @param height The new content height.
	 */
	private onTreeContentHeightChange(height: number): void {
		this.updateBodySize(height);
	}

	/**
	 * @brief Asynchronously updates the children of the tree.
	 *
	 * This is called when the underlying repository data changes.
	 */
	private async updateChildren(): Promise<void> {
		await this.tree.updateChildren();
		this.updateBodySize(this.tree.contentHeight);
	}

	/**
	 * @brief Updates the size of the view pane's body.
	 * @param contentHeight The height of the tree's content.
	 * @param visibleCount The number of items that should be visible.
	 *
	 * This method calculates and applies the minimum and maximum body size based on
	 * the content height and the `scm.repositories.visible` setting.
	 */
	private updateBodySize(contentHeight: number, visibleCount?: number): void {
		if (this.orientation === Orientation.HORIZONTAL) {
			return;
		}

		visibleCount = visibleCount ?? this.visibleCountObs.get();
		const empty = this.scmViewService.repositories.length === 0;
		const size = Math.min(contentHeight / 22, visibleCount) * 22;

		this.minimumBodySize = visibleCount === 0 ? 22 : size;
		this.maximumBodySize = visibleCount === 0 ? Number.POSITIVE_INFINITY : empty ? Number.POSITIVE_INFINITY : size;
	}

	/**
	 * @brief Asynchronously updates the tree selection.
	 *
	 * This method synchronizes the tree's selection with the `visibleRepositories`
	 * from the `ISCMViewService`.
	 */
	private async updateTreeSelection(): Promise<void> {
		const oldSelection = this.tree.getSelection();
		const oldSet = new Set(oldSelection);

		const set = new Set(this.scmViewService.visibleRepositories);
		const added = new Set(Iterable.filter(set, r => !oldSet.has(r)));
		const removed = new Set(Iterable.filter(oldSet, r => !set.has(r)));

		if (added.size === 0 && removed.size === 0) {
			return;
		}

		const selection = oldSelection.filter(repo => !removed.has(repo));

		for (const repo of this.scmViewService.repositories) {
			if (added.has(repo)) {
				selection.push(repo);
			}
		}

		// Expand all selected items to ensure they are visible.
		for (const item of selection) {
			await this.tree.expandTo(item);
		}
		this.tree.setSelection(selection);

		if (selection.length > 0 && !this.tree.getFocus().includes(selection[0])) {
			this.tree.setAnchor(selection[0]);
			this.tree.setFocus([selection[0]]);
		}
	}

	override dispose(): void {
		this.visibilityDisposables.dispose();
		super.dispose();
	}
}
