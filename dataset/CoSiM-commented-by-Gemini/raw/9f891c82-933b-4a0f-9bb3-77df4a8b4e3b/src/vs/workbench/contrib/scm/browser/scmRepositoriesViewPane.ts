/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file scmRepositoriesViewPane.ts
 * @brief This file defines a view pane for displaying a list of Source Control Management (SCM) repositories.
 *
 * It uses a WorkbenchCompressibleAsyncDataTree to render a tree of repositories,
 * allowing for features like nested repositories. The view is reactive to changes in
 * SCM state and user configuration, leveraging an observable-based pattern for UI updates.
 */

import './media/scm.css';
import { localize } from '../../../../nls.js';
import { ViewPane, IViewPaneOptions } from '../../../browser/parts/views/viewPane.js';
import { append, $ } from '../../../../base/browser/dom.js';
import { IListVirtualDelegate, IIdentityProvider } from '../../../../base/browser/ui/list/list.js';
import { IAsyncDataSource, ITreeEvent, ITreeContextMenuEvent } from '../../../../base/browser/ui/tree/tree.js';
import { WorkbenchCompressibleAsyncDataTree } from '../../../../platform/list/browser/listService.js';
import { ISCMRepository, ISCMService, ISCMViewService } from '../common/scm.js';
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
import { autorun, IObservable, observableFromEvent, observableSignalFromEvent } from '../../../../base/common/observable.js';
import { Sequencer } from '../../../../base/common/async.js';

/**
 * @class ListDelegate
 * @brief A standard list delegate for the repository tree.
 * It provides a fixed height for each item and a template ID for the renderer.
 */
class ListDelegate implements IListVirtualDelegate<ISCMRepository> {

	getHeight(): number {
		return 22;
	}

	getTemplateId(): string {
		return RepositoryRenderer.TEMPLATE_ID;
	}
}

/**
 * @class RepositoryTreeDataSource
 * @brief Implements the async data source for the repository tree.
 * It fetches repository data from the ISCMViewService, supporting a hierarchical
 * structure where repositories can be nested under a parent.
 */
class RepositoryTreeDataSource extends Disposable implements IAsyncDataSource<ISCMViewService, ISCMRepository> {
	constructor(@ISCMViewService private readonly scmViewService: ISCMViewService) {
		super();
	}

	/**
	 * Gets the children of a given element. If the element is the root (ISCMViewService),
	 * it returns the top-level repositories. If it's a repository, it returns its sub-repositories.
	 * @param inputOrElement The root or a parent repository.
	 * @returns An iterable of child repositories.
	 */
	getChildren(inputOrElement: ISCMViewService | ISCMRepository): Iterable<ISCMRepository> {
		const parentId = isSCMRepository(inputOrElement)
			? inputOrElement.provider.id
			: undefined;

		return this.scmViewService.repositories
			.filter(r => r.provider.parentId === parentId);
	}

	/**
	 * Checks if a given element has child repositories.
	 * @param inputOrElement The root or a parent repository.
	 * @returns True if the element has children, false otherwise.
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
 * @brief Provides a unique and stable identity for each repository in the tree.
 * This is crucial for the tree to maintain selection and expansion state across updates.
 */
class RepositoryTreeIdentityProvider implements IIdentityProvider<ISCMRepository> {
	getId(element: ISCMRepository): string {
		return element.provider.id;
	}
}

/**
 * @class SCMRepositoriesViewPane
 * @brief A view pane that displays a tree of SCM repositories.
 */
export class SCMRepositoriesViewPane extends ViewPane {

	private tree!: WorkbenchCompressibleAsyncDataTree<ISCMViewService, ISCMRepository, any>;
	private treeDataSource!: RepositoryTreeDataSource;
	private treeIdentityProvider!: RepositoryTreeIdentityProvider;
	/** A sequencer to ensure that tree operations run one after another, preventing race conditions. */
	private readonly treeOperationSequencer = new Sequencer();

	/** Observables for user configuration settings. */
	private readonly visibleCountObs: IObservable<number>;
	private readonly providerCountBadgeObs: IObservable<'hidden' | 'auto' | 'visible'>;

	/** A store for disposables that are only active when the view is visible. */
	private readonly visibilityDisposables = new DisposableStore();

	constructor(
		options: IViewPaneOptions,
		@ISCMService private readonly scmService: ISCMService,
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

		/**
		 * Functional Utility: Create observables from configuration values.
		 * This allows the UI to react automatically when the user changes these settings.
		 */
		this.visibleCountObs = observableConfigValue('scm.repositories.visible', 10, this.configurationService);
		this.providerCountBadgeObs = observableConfigValue<'hidden' | 'auto' | 'visible'>('scm.providerCountBadge', 'hidden', this.configurationService);
	}

	protected override renderBody(container: HTMLElement): void {
		super.renderBody(container);

		const treeContainer = append(container, $('.scm-view.scm-repositories-view'));

		/**
		 * Block Logic: An autorun block that reacts to changes in the `scm.providerCountBadge`
		 * setting. It toggles CSS classes to show, hide, or automatically manage the
		 * visibility of repository provider counts in the UI.
		 */
		this._register(autorun(reader => {
			const providerCountBadge = this.providerCountBadgeObs.read(reader);
			treeContainer.classList.toggle('hide-provider-counts', providerCountBadge === 'hidden');
			treeContainer.classList.toggle('auto-provider-counts', providerCountBadge === 'auto');
		}));

		this.createTree(treeContainer);

		/**
		 * Block Logic: Sets up listeners and performs initial rendering only when the
		 * view becomes visible. This is an optimization to avoid work when the
		 * view is hidden.
		 */
		this.onDidChangeBodyVisibility(async visible => {
			if (!visible) {
				this.visibilityDisposables.clear();
				return;
			}

			this.treeOperationSequencer.queue(async () => {
				// Initial rendering of the tree with the SCM view service as input.
				await this.tree.setInput(this.scmViewService);

				/**
				 * Block Logic: Reacts to `scm.repositories.visible` setting changes to
				 * dynamically adjust the view's body size.
				 */
				this.visibilityDisposables.add(autorun(reader => {
					const visibleCount = this.visibleCountObs.read(reader);
					this.updateBodySize(this.tree.contentHeight, visibleCount);
				}));

				/**
				 * Block Logic: Reacts to repositories being added or removed from the SCM service.
				 * This ensures the tree view stays in sync with the underlying data model.
				 */
				const addedRepositoryObs = observableFromEvent(this, this.scmService.onDidAddRepository, e => e);
				const removedRepositoryObs = observableFromEvent(this, this.scmService.onDidRemoveRepository, e => e);

				this.visibilityDisposables.add(autorun(async reader => {
					const addedRepository = addedRepositoryObs.read(reader);
					const removedRepository = removedRepositoryObs.read(reader);

					if (addedRepository === undefined && removedRepository === undefined) {
						await this.updateChildren();
						return;
					}

					if (addedRepository) {
						await this.updateRepository(addedRepository);
					}

					if (removedRepository) {
						await this.updateRepository(removedRepository);
					}
				}));

				/**
				 * Block Logic: Reacts to changes in the set of *visible* repositories (e.g., selection)
				 * and updates the tree's selection state accordingly.
				 */
				const onDidChangeVisibleRepositoriesSignal = observableSignalFromEvent(this, this.scmViewService.onDidChangeVisibleRepositories);

				this.visibilityDisposables.add(autorun(async reader => {
					onDidChangeVisibleRepositoriesSignal.read(reader);
					await this.treeOperationSequencer.queue(() => this.updateTreeSelection());
				}));
			});
		}, this, this._store);
	}

	protected override layoutBody(height: number, width: number): void {
		super.layoutBody(height, width);
		this.tree.layout(height, width);
	}

	override focus(): void {
		super.focus();
		this.tree.domFocus();
	}

	/**
	 * Creates and configures the `WorkbenchCompressibleAsyncDataTree` instance used to
	 * display the repositories.
	 * @param container The HTML element to render the tree into.
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
				// Repositories are always incompressible in this view.
				isIncompressible: () => true
			},
			[
				this.instantiationService.createInstance(RepositoryRenderer, MenuId.SCMSourceControlInline, getActionViewItemProvider(this.instantiationService))
			],
			this.treeDataSource,
			{
				identityProvider: this.treeIdentityProvider,
				horizontalScrolling: false,
				// Logic to determine initial expansion state of a repository.
				collapseByDefault: (e: unknown) => {
					if (isSCMRepository(e) && e.provider.parentId === undefined) {
						// Don't collapse top-level repositories by default.
						return false;
					}
					return true;
				},
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

		// Register event listeners for tree interactions.
		this._register(this.tree.onDidChangeSelection(this.onTreeSelectionChange, this));
		this._register(this.tree.onDidChangeFocus(this.onTreeDidChangeFocus, this));
		this._register(this.tree.onContextMenu(this.onTreeContextMenu, this));
		this._register(this.tree.onDidChangeContentHeight(this.onTreeContentHeightChange, this));
	}

	/**
	 * Handles the context menu event on a tree item.
	 * @param e The tree context menu event.
	 */
	private onTreeContextMenu(e: ITreeContextMenuEvent<ISCMRepository>): void {
		if (!e.element) {
			return;
		}

		// Block Logic: Gathers context menu actions for the selected repository and displays them.
		const provider = e.element.provider;
		const menus = this.scmViewService.menus.getRepositoryMenus(provider);
		const menu = menus.repositoryContextMenu;
		const actions = collectContextMenuActions(menu);

		const disposables = new DisposableStore();
		const actionRunner = new RepositoryActionRunner(() => this.tree.getSelection());
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
	 * Handles tree selection changes and updates the SCM view service.
	 * @param e The tree selection change event.
	 */
	private onTreeSelectionChange(e: ITreeEvent<ISCMRepository>): void {
		if (e.browserEvent && e.elements.length > 0) {
			const scrollTop = this.tree.scrollTop;
			this.scmViewService.visibleRepositories = e.elements;
			this.tree.scrollTop = scrollTop;
		}
	}

	private onTreeDidChangeFocus(e: ITreeEvent<ISCMRepository>): void {
		if (e.browserEvent && e.elements.length > 0) {
			this.scmViewService.focus(e.elements[0]);
		}
	}

	/**
	 * Adjusts the view pane's size when the tree's content height changes.
	 * @param height The new content height.
	 */
	private onTreeContentHeightChange(height: number): void {
		this.updateBodySize(height);

		// Refresh the selection to ensure it's still valid after a potential size change.
		this.treeOperationSequencer.queue(() => this.updateTreeSelection());
	}

	/**
	 * Asynchronously updates the children of a given element in the tree.
	 * @param element The element whose children need updating. Defaults to the root.
	 */
	private async updateChildren(element?: ISCMRepository): Promise<void> {
		await this.treeOperationSequencer.queue(async () => {
			if (element && this.tree.hasNode(element)) {
				await this.tree.updateChildren(element, true);
			} else {
				await this.tree.updateChildren(undefined, true);
			}
		});
	}

	/**
	 * Ensures a specific element is expanded in the tree.
	 * @param element The element to expand.
	 */
	private async expand(element: ISCMRepository): Promise<void> {
		await this.treeOperationSequencer.queue(() => this.tree.expand(element, true));
	}

	/**
	 * Updates the tree when a repository is added or removed.
	 * @param repository The repository that changed.
	 */
	private async updateRepository(repository: ISCMRepository): Promise<void> {
		// If it's a top-level repository, refresh the whole tree.
		if (repository.provider.parentId === undefined) {
			await this.updateChildren();
			return;
		}

		// Otherwise, just refresh the parent.
		await this.updateParentRepository(repository);
	}

	/**
	 * Finds the parent of a given repository and updates its children in the tree.
	 * @param repository The repository whose parent needs updating.
	 */
	private async updateParentRepository(repository: ISCMRepository): Promise<void> {
		const parentRepository = this.scmViewService.repositories
			.find(r => r.provider.id === repository.provider.parentId);
		if (!parentRepository) {
			return;
		}

		await this.updateChildren(parentRepository);
		await this.expand(parentRepository);
	}

	/**
	 * Calculates and sets the minimum and maximum body size of the view pane.
	 * @param contentHeight The current height of the tree's content.
	 * @param visibleCount The configured number of repositories to show.
	 */
	private updateBodySize(contentHeight: number, visibleCount?: number): void {
		if (this.orientation === Orientation.HORIZONTAL) {
			return;
		}

		visibleCount = visibleCount ?? this.visibleCountObs.get();
		const empty = this.scmViewService.repositories.length === 0;
		const rowHeight = 22;
		const size = Math.min(contentHeight / rowHeight, visibleCount) * rowHeight;

		this.minimumBodySize = visibleCount === 0 ? rowHeight : size;
		this.maximumBodySize = visibleCount === 0 ? Number.POSITIVE_INFINITY : empty ? Number.POSITIVE_INFINITY : size;
	}

	/**
	 * Synchronizes the tree's selection state with the `visibleRepositories`
	 * from the SCM view service.
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

		const visibleSelection = selection.filter(s => this.tree.hasNode(s));

		this.tree.setSelection(visibleSelection);

		if (visibleSelection.length > 0 && !this.tree.getFocus().includes(visibleSelection[0])) {
			this.tree.setAnchor(visibleSelection[0]);
			this.tree.setFocus([visibleSelection[0]]);
		}
	}

	override dispose(): void {
		this.visibilityDisposables.dispose();
		super.dispose();
	}
}
