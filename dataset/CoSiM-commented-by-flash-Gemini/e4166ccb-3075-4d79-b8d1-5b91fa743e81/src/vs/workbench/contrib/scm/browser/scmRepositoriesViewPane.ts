/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file scmRepositoriesViewPane.ts
 * @brief Workbench view pane for managing and orchestrating multiple SCM repositories.
 * 
 * Functional Intent: Provides a hierarchical tree-view of all active Source Control 
 * repositories in the workspace. It leverages a reactive architecture (Observables/autorun) 
 * to keep the UI in sync with the underlying SCM service state, handling repository 
 * registration, visibility toggling, and multi-selection workflows.
 * 
 * Domain: Production Systems, SCM (Source Control Management), Reactive UI, Tree Views.
 */

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
 * @brief Static configuration for repository list item dimensions and templating.
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
 * @brief Resolves hierarchical parent-child relationships for SCM repositories.
 * 
 * Logic: Filters the global SCM service repository list based on provider parent-child 
 * linkages to build a multi-level tree structure.
 */
class RepositoryTreeDataSource extends Disposable implements IAsyncDataSource<ISCMViewService, ISCMRepository> {
	constructor(@ISCMViewService private readonly scmViewService: ISCMViewService) {
		super();
	}

	getChildren(inputOrElement: ISCMViewService | ISCMRepository): Iterable<ISCMRepository> {
		const parentId = isSCMRepository(inputOrElement)
			? inputOrElement.provider.id
			: undefined;

		const repositories = this.scmViewService.repositories
			.filter(r => r.provider.parentId === parentId);

		return repositories;
	}

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

	hasChildren(inputOrElement: ISCMViewService | ISCMRepository): boolean {
		const parentId = isSCMRepository(inputOrElement)
			? inputOrElement.provider.id
			: undefined;

		const repositories = this.scmViewService.repositories
			.filter(r => r.provider.parentId === parentId);

		return repositories.length > 0;
	}
}

class RepositoryTreeIdentityProvider implements IIdentityProvider<ISCMRepository> {
	getId(element: ISCMRepository): string {
		return element.provider.id;
	}
}

/**
 * @class SCMRepositoriesViewPane
 * @brief The primary view container for the "Source Control Repositories" section.
 * 
 * Logic: Coordinates the lifecycle of the tree widget and its integration 
 * with the Workbench. Uses a Sequencer to ensure tree updates (data fetching, 
 * selection reconciliation) are executed atomically and in order.
 */
export class SCMRepositoriesViewPane extends ViewPane {

	private tree!: WorkbenchCompressibleAsyncDataTree<ISCMViewService, ISCMRepository, any>;
	private treeDataSource!: RepositoryTreeDataSource;
	private treeIdentityProvider!: RepositoryTreeIdentityProvider;
	// treeOperationSequencer - Guards against race conditions in asynchronous tree refreshes.
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

		// Logic: Reactive configuration values that trigger UI updates on change.
		this.visibleCountObs = observableConfigValue('scm.repositories.visible', 10, this.configurationService);
		this.providerCountBadgeObs = observableConfigValue<'hidden' | 'auto' | 'visible'>('scm.providerCountBadge', 'hidden', this.configurationService);
	}

	protected override renderBody(container: HTMLElement): void {
		super.renderBody(container);

		const treeContainer = append(container, $('.scm-view.scm-repositories-view'));

		// Block Logic: Dynamic styling based on user settings.
		// Invariant: CSS classes on the container reflect the 'scm.providerCountBadge' state.
		this._register(autorun(reader => {
			const providerCountBadge = this.providerCountBadgeObs.read(reader);
			treeContainer.classList.toggle('hide-provider-counts', providerCountBadge === 'hidden');
			treeContainer.classList.toggle('auto-provider-counts', providerCountBadge === 'auto');
		}));

		this.createTree(treeContainer);

		/**
		 * Block Logic: Viewport visibility management.
		 * Logic: Subscribes to SCM service events only when the pane is visible 
		 * to minimize background compute. Uses 'autorun' to react to repository 
		 * list mutations and visibility changes.
		 */
		this.onDidChangeBodyVisibility(async visible => {
			if (!visible) {
				this.visibilityDisposables.clear();
				return;
			}

			this.treeOperationSequencer.queue(async () => {
				await this.tree.setInput(this.scmViewService);

				this.visibilityDisposables.add(autorun(reader => {
					const visibleCount = this.visibleCountObs.read(reader);
					this.updateBodySize(this.tree.contentHeight, visibleCount);
				}));

				// Synchronization: Reacts to SCM repository registration/unregistration.
				const onDidChangeRepositoriesSignal = observableSignalFromEvent(
					this, this.scmViewService.onDidChangeRepositories);

				this.visibilityDisposables.add(autorun(async reader => {
					onDidChangeRepositoriesSignal.read(reader);
					await this.treeOperationSequencer.queue(() => this.updateChildren());
				}));

				// Synchronization: Reacts to repository visibility (selection) changes.
				const onDidChangeVisibleRepositoriesSignal = observableSignalFromEvent(
					this, this.scmViewService.onDidChangeVisibleRepositories);

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

		// Event Wiring: Binds tree interactions to SCM service updates.
		this._register(this.tree.onDidChangeSelection(this.onTreeSelectionChange, this));
		this._register(this.tree.onDidChangeFocus(this.onTreeDidChangeFocus, this));
		this._register(this.tree.onContextMenu(this.onTreeContextMenu, this));
		this._register(this.tree.onDidChangeContentHeight(this.onTreeContentHeightChange, this));
	}

	/**
	 * onTreeContextMenu - Dispatches context menu resolution to repository-specific providers.
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

	private onTreeSelectionChange(e: ITreeEvent<ISCMRepository>): void {
		// Logic: Selection in the tree defines the set of "visible" repositories in the primary SCM view.
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

	private onTreeContentHeightChange(height: number): void {
		this.updateBodySize(height);
	}

	private async updateChildren(): Promise<void> {
		await this.tree.updateChildren();
		this.updateBodySize(this.tree.contentHeight);
	}

	/**
	 * updateBodySize - Dynamically adjusts the pane height to fit its content.
	 * 
	 * Logic: Computes the required height based on the number of repositories 
	 * (up to a user-configured limit) and the tree's actual content height.
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
	 * updateTreeSelection - Synchronizes the tree selection state with the SCM service.
	 * 
	 * Logic: Performs a delta calculation between the tree's current selection 
	 * and the service's visible list. Expands tree nodes as needed to ensure 
	 * selected items are visible.
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

		// Block Logic: UI state reconciliation.
		// Invariant: All selected repositories must be expanded in the tree.
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
