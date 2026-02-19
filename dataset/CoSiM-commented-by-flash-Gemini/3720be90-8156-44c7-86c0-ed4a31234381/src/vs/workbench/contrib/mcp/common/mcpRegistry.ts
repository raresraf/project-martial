/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
/**
 * @file mcpRegistry.ts
 * @module vs/workbench/contrib/mcp/common/mcpRegistry
 * @description Implements the central registry for Multi-Cloud Platform (MCP) collections and servers.
 *              This file provides mechanisms for registering MCP collections, managing their trust,
 *              storing and resolving configuration inputs, and resolving connections to MCP servers.
 *              It acts as a core component for the MCP feature's lifecycle and interaction logic.
 */

// Functional Utility: Imports Emitter for event handling.
import { Emitter } from '../../../../base/common/event.js';
// Functional Utility: Imports StringSHA1 for generating SHA1 hashes from strings.
import { StringSHA1 } from '../../../../base/common/hash.js';
// Functional Utility: Imports MarkdownString for creating rich markdown content, typically for UI.
import { MarkdownString } from '../../../../base/common/htmlContent.js';
// Functional Utility: Imports Lazy for deferred initialization of properties.
import { Lazy } from '../../../../base/common/lazy.js';
// Functional Utility: Imports Disposable for managing disposable resources and IDisposable interface.
import { Disposable, IDisposable } from '../../../../base/common/lifecycle.js';
// Functional Utility: Imports observable utilities for reactive programming and state management.
import { derived, IObservable, observableValue } from '../../../../base/common/observable.js';
// Functional Utility: Imports basename utility for extracting the base name from a URI.
import { basename } from '../../../../base/common/resources.js';
// Functional Utility: Imports localization function.
import { localize } from '../../../../nls.js';
// Functional Utility: Imports ConfigurationTarget for specifying where configuration changes apply.
import { ConfigurationTarget } from '../../../../platform/configuration/common/configuration.js';
// Functional Utility: Imports IDialogService for interacting with user dialogs.
import { IDialogService } from '../../../../platform/dialogs/common/dialogs.js';
// Functional Utility: Imports IInstantiationService for creating objects with dependency injection.
import { IInstantiationService } from '../../../../platform/instantiation/common/instantiation.js';
// Functional Utility: Imports INotificationService and Severity for displaying notifications.
import { INotificationService, Severity } from '../../../../platform/notification/common/notification.js';
// Functional Utility: Imports observableMemento for managing observable memento state.
import { observableMemento } from '../../../../platform/observable/common/observableMemento.js';
// Functional Utility: Imports IProductService for accessing product-specific information.
import { IProductService } from '../../../../platform/product/common/productService.js';
// Functional Utility: Imports IStorageService, StorageScope, and StorageTarget for persistent storage.
import { IStorageService, StorageScope, StorageTarget } from '../../../../platform/storage/common/storage.js';
// Functional Utility: Imports IWorkspaceFolderData for workspace folder information.
import { IWorkspaceFolderData } from '../../../../platform/workspace/common/workspace.js';
// Functional Utility: Imports IConfigurationResolverService for resolving configuration variables.
import { IConfigurationResolverService } from '../../../services/configurationResolver/common/configurationResolver.js';
// Functional Utility: Imports ConfigurationResolverExpression for parsing and resolving configuration variables, and IResolvedValue type.
import { ConfigurationResolverExpression, IResolvedValue } from '../../../services/configurationResolver/common/configurationResolverExpression.js';
// Functional Utility: Imports IEditorService for opening editors.
import { IEditorService } from '../../../services/editor/common/editorService.js';
// Functional Utility: Imports McpRegistryInputStorage for managing stored MCP inputs.
import { McpRegistryInputStorage } from './mcpRegistryInputStorage.js';
// Functional Utility: Imports interfaces and types related to MCP registry delegates and connection options.
import { IMcpHostDelegate, IMcpRegistry, IMcpResolveConnectionOptions } from './mcpRegistryTypes.js';
// Functional Utility: Imports McpServerConnection class.
import { McpServerConnection } from './mcpServerConnection.js';
// Functional Utility: Imports interfaces and types related to MCP server connections, collections, and definitions.
import { IMcpServerConnection, LazyCollectionState, McpCollectionDefinition, McpCollectionReference, McpServerDefinition, McpServerLaunch } from './mcpTypes.js';

// Constant: Observable memento for storing trusted MCP collections.
const createTrustMemento = observableMemento<Readonly<Record<string, boolean>>>({
	defaultValue: {},
	key: 'mcp.trustedCollections'
});

// Constant: Defines the length of the tool prefix generated from collection IDs.
const collectionPrefixLen = 3;

/**
 * @class McpRegistry
 * @extends Disposable
 * @implements IMcpRegistry
 * @brief Central registry for managing Multi-Cloud Platform (MCP) collections and servers.
 *
 * This class provides mechanisms for:
 * - Registering and managing MCP collection definitions.
 * - Handling trust prompts for unknown collections.
 * - Storing and resolving configuration inputs (e.g., for server launch variables).
 * - Discovering and activating lazy-loaded collections.
 * - Resolving connections to MCP servers based on registered delegates.
 */
export class McpRegistry extends Disposable implements IMcpRegistry {
	declare public readonly _serviceBrand: undefined; // Functional Utility: Marks this class as implementing a service.

	// Map: Stores promises for ongoing trust prompts to prevent multiple dialogs for the same collection.
	private readonly _trustPrompts = new Map</* collection ID */string, Promise<boolean | undefined>>();

	// ObservableValue: Holds the list of all registered MCP collection definitions.
	private readonly _collections = observableValue<readonly McpCollectionDefinition[]>('collections', []);
	// Array: Stores delegates capable of starting MCP server connections.
	private readonly _delegates: IMcpHostDelegate[] = [];
	// IObservable: Publicly exposed observable of registered MCP collections.
	public readonly collections: IObservable<readonly McpCollectionDefinition[]> = this._collections;

	// IObservable: Derived observable that maps collection IDs to their generated tool prefixes.
	private readonly _collectionToPrefixes = this._collections.map(c => {
		// Block Logic: Generates tool prefixes based on a SHA1 hash of the collection ID.
		// This ensures stable, unique prefixes that are short enough to avoid errors.
		type CollectionHash = { view: number; hash: string; collection: McpCollectionDefinition };

		const hashes = c.map((collection): CollectionHash => {
			const sha = new StringSHA1();
			sha.update(collection.id);
			return { view: 0, hash: sha.digest(), collection };
		});

		// Functional Utility: Helper function to extract a view of the hash for collision checking.
		const view = (h: CollectionHash) => h.hash.slice(h.view, h.view + collectionPrefixLen);

		let collided = false;
		// Block Logic: Collision resolution loop for generated prefixes.
		// If prefixes collide, the 'view' (starting index of the hash slice) is adjusted.
		do {
			// Sorts hashes to group potential collisions.
			hashes.sort((a, b) => view(a).localeCompare(view(b)) || a.collection.id.localeCompare(b.collection.id));
			collided = false;
			for (let i = 1; i < hashes.length; i++) {
				const prev = hashes[i - 1];
				const curr = hashes[i];
				// If a collision is detected and more of the hash is available, adjust the 'view'.
				if (view(prev) === view(curr) && curr.view + collectionPrefixLen < curr.hash.length) {
					curr.view++;
					collided = true;
				}
			}
		} while (collided); // Continues until no more collisions.

		// Functional Utility: Maps each collection ID to its resolved, unique prefix.
		return Object.fromEntries(hashes.map(h => [h.collection.id, view(h) + '.']));
	});

	// Lazy: Lazily initialized storage for workspace-scoped MCP inputs.
	private readonly _workspaceStorage = new Lazy(() => this._register(this._instantiationService.createInstance(McpRegistryInputStorage, StorageScope.WORKSPACE, StorageTarget.USER)));
	// Lazy: Lazily initialized storage for profile-scoped MCP inputs.
	private readonly _profileStorage = new Lazy(() => this._register(this._instantiationService.createInstance(McpRegistryInputStorage, StorageScope.PROFILE, StorageTarget.USER)));

	// Lazy: Lazily initialized memento for storing trust decisions for MCP collections.
	private readonly _trustMemento = new Lazy(() => this._register(createTrustMemento(StorageScope.APPLICATION, StorageTarget.MACHINE, this._storageService)));
	// Set: Stores IDs of lazy collections that are currently being updated.
	private readonly _lazyCollectionsToUpdate = new Set</* collection ID*/string>();
	// ObservableValue: Tracks the number of ongoing lazy activation processes.
	private readonly _ongoingLazyActivations = observableValue(this, 0);

	// IObservable: Derived observable that reflects the overall state of lazy collections (loading, has unknown, all known).
	public readonly lazyCollectionState = derived(reader => {
		// Block Logic: If there are ongoing lazy activations, the state is LoadingUnknown.
		if (this._ongoingLazyActivations.read(reader) > 0) {
			return LazyCollectionState.LoadingUnknown;
		}
		// Block Logic: Checks if any lazy collection is not yet cached.
		const collections = this._collections.read(reader);
		return collections.some(c => c.lazy && c.lazy.isCached === false) ? LazyCollectionState.HasUnknown : LazyCollectionState.AllKnown;
	});

	// Getter: Returns a read-only array of registered MCP host delegates.
	public get delegates(): readonly IMcpHostDelegate[] {
		return this._delegates;
	}

	// Emitter: Event fired when MCP inputs change.
	private readonly _onDidChangeInputs = this._register(new Emitter<void>());
	// Event: Publicly exposed event for when MCP inputs change.
	public readonly onDidChangeInputs = this._onDidChangeInputs.event;

	constructor(
		@IInstantiationService private readonly _instantiationService: IInstantiationService,
		@IConfigurationResolverService private readonly _configurationResolverService: IConfigurationResolverService,
		@IDialogService private readonly _dialogService: IDialogService,
		@IStorageService private readonly _storageService: IStorageService,
		@IProductService private readonly _productService: IProductService,
		@INotificationService private readonly _notificationService: INotificationService,
		@IEditorService private readonly _editorService: IEditorService,
	) {
		super();
	}

	/**
	 * @brief Registers a new MCP host delegate.
	 *
	 * Delegates are responsible for specific aspects of MCP server management,
	 * such as starting/stopping servers or resolving connections.
	 *
	 * @param delegate (IMcpHostDelegate): The delegate to register.
	 * @returns (IDisposable): A disposable to unregister the delegate.
	 */
	public registerDelegate(delegate: IMcpHostDelegate): IDisposable {
		this._delegates.push(delegate);
		return {
			dispose: () => {
				const index = this._delegates.indexOf(delegate);
				if (index !== -1) {
					this._delegates.splice(index, 1);
				}
			}
		};
	}

	/**
	 * @brief Registers a new MCP collection definition.
	 *
	 * If an incoming collection is a non-lazy version of an existing lazy collection,
	 * it replaces the lazy one. Otherwise, it's added to the list of collections.
	 *
	 * @param collection (McpCollectionDefinition): The collection definition to register.
	 * @returns (IDisposable): A disposable to unregister the collection.
	 */
	public registerCollection(collection: McpCollectionDefinition): IDisposable {
		const currentCollections = this._collections.get();
		// Block Logic: Checks if the incoming collection is a replacement for a lazy collection.
		const toReplace = currentCollections.find(c => c.lazy && c.id === collection.id);

		// Block Logic: Incoming collections replace the "lazy" versions.
		if (toReplace) {
			this._lazyCollectionsToUpdate.add(collection.id); // Inline: Marks the collection for update.
			// Functional Utility: Replaces the lazy collection with the new, non-lazy one.
			this._collections.set(currentCollections.map(c => c === toReplace ? collection : c), undefined);
		} else {
			// Functional Utility: Adds the new collection to the existing list.
			this._collections.set([...currentCollections, collection], undefined);
		}

		return {
			dispose: () => {
				const currentCollections = this._collections.get();
				// Functional Utility: Removes the collection from the list upon disposal.
				this._collections.set(currentCollections.filter(c => c !== collection), undefined);
			}
		};
	}

	/**
	 * @brief Returns an observable for the tool prefix associated with a given MCP collection.
	 *
	 * The tool prefix is a short, unique identifier derived from the collection ID,
	 * used to disambiguate tools from different collections.
	 *
	 * @param collection (McpCollectionReference): The reference to the MCP collection.
	 * @returns (IObservable<string>): An observable that provides the tool prefix.
	 */
	public collectionToolPrefix(collection: McpCollectionReference): IObservable<string> {
		return this._collectionToPrefixes.map(p => p[collection.id] ?? '');
	}

	/**
	 * @brief Discovers and loads lazy-loaded MCP collections.
	 *
	 * This method triggers the loading of collections marked as lazy and not yet cached.
	 * It updates the `_ongoingLazyActivations` observable to reflect the loading state.
	 *
	 * @returns (Promise<McpCollectionDefinition[]>): A promise that resolves to an array of discovered non-lazy collections.
	 */
	public async discoverCollections(): Promise<McpCollectionDefinition[]> {
		// Block Logic: Identifies lazy collections that need to be discovered.
		const toDiscover = this._collections.get().filter(c => c.lazy && !c.lazy.isCached);

		// Block Logic: Increments the count of ongoing lazy activations.
		this._ongoingLazyActivations.set(this._ongoingLazyActivations.get() + 1, undefined);
		// Block Logic: Triggers the loading of lazy collections and decrements the counter when all are loaded.
		await Promise.all(toDiscover.map(c => c.lazy?.load())).finally(() => {
			this._ongoingLazyActivations.set(this._ongoingLazyActivations.get() - 1, undefined);
		});

		const found: McpCollectionDefinition[] = [];
		const current = this._collections.get();
		// Block Logic: Processes discovered collections, handling cases where lazy collections might not have been replaced.
		for (const collection of toDiscover) {
			const rec = current.find(c => c.id === collection.id);
			if (!rec) {
				// Inline: Collection was ignored or removed.
			} else if (rec.lazy) {
				rec.lazy.removed?.(); // Functional Utility: Calls 'removed' hook if the lazy version wasn't replaced.
			} else {
				found.push(rec); // Inline: Adds the newly found non-lazy collection to the result.
			}
		}

		return found;
	}

	/**
	 * @brief Retrieves the appropriate input storage instance based on the given storage scope.
	 * @param scope (StorageScope): The storage scope (workspace or profile).
	 * @returns (McpRegistryInputStorage): The corresponding input storage instance.
	 */
	private _getInputStorage(scope: StorageScope): McpRegistryInputStorage {
		return scope === StorageScope.WORKSPACE ? this._workspaceStorage.value : this._profileStorage.value;
	}

	/**
	 * @brief Clears saved MCP inputs from storage.
	 *
	 * Can clear a specific input by ID or all inputs within a given scope.
	 *
	 * @param scope (StorageScope): The storage scope from which to clear inputs.
	 * @param inputId (string): Optional. The ID of the specific input to clear. If omitted, all inputs in the scope are cleared.
	 * @returns (Promise<void>): A promise that resolves when the inputs are cleared.
	 */
	public async clearSavedInputs(scope: StorageScope, inputId?: string) {
		const storage = this._getInputStorage(scope);
		if (inputId) {
			await storage.clear(inputId); // Inline: Clears a specific input.
		} else {
			storage.clearAll(); // Inline: Clears all inputs in the scope.
		}

		this._onDidChangeInputs.fire(); // Inline: Fires an event indicating input changes.
	}

	/**
	 * @brief Initiates an interactive process to edit a saved MCP input.
	 *
	 * This method uses the configuration resolver service to prompt the user
	 * for a new value for a given input, and then updates the stored value.
	 *
	 * @param inputId (string): The ID of the input to edit.
	 * @param folderData (IWorkspaceFolderData): Optional. Workspace folder data for context.
	 * @param configSection (string): The configuration section where the input is used.
	 * @param target (ConfigurationTarget): The configuration target (e.g., Global, Workspace).
	 * @returns (Promise<void>): A promise that resolves when the input is edited.
	 */
	public async editSavedInput(inputId: string, folderData: IWorkspaceFolderData | undefined, configSection: string, target: ConfigurationTarget): Promise<void> {
		// Block Logic: Determines the correct input storage based on the configuration target.
		const storage = this._getInputStorage(target === ConfigurationTarget.WORKSPACE || target === ConfigurationTarget.WORKSPACE_FOLDER ? StorageScope.WORKSPACE : StorageTarget.PROFILE);
		// Block Logic: Parses the input ID into a ConfigurationResolverExpression.
		const expr = ConfigurationResolverExpression.parse(inputId);

		// Block Logic: Retrieves previously stored inputs.
		const stored = await storage.getMap();
		const previous = stored[inputId].value;
		// Functional Utility: Interactively resolves the expression, prompting the user for input if necessary.
		await this._configurationResolverService.resolveWithInteraction(folderData, expr, configSection, previous ? { [inputId]: previous } : {}, target);
		// Block Logic: Updates the storage with the newly resolved expression inputs.
		await this._updateStorageWithExpressionInputs(storage, expr);
	}

	/**
	 * @brief Retrieves all saved MCP inputs for a given storage scope.
	 * @param scope (StorageScope): The storage scope from which to retrieve inputs.
	 * @returns (Promise<{ [id: string]: IResolvedValue }>): A promise that resolves to a map of input IDs to their resolved values.
	 */
	public getSavedInputs(scope: StorageScope): Promise<{ [id: string]: IResolvedValue }> {
		return this._getInputStorage(scope).getMap();
	}

	/**
	 * @brief Resets all trust decisions for MCP collections.
	 *
	 * This clears the stored trust memento, effectively revoking trust
	 * for all previously trusted collections.
	 */
	public resetTrust(): void {
		this._trustMemento.value.set({}, undefined); // Inline: Sets the trust memento back to an empty object.
	}

	/**
	 * @brief Retrieves the trust status of a specific MCP collection.
	 *
	 * Returns an observable that reflects whether the collection is trusted,
	 * not trusted, or if no decision has been made yet.
	 *
	 * @param collectionRef (McpCollectionReference): The reference to the MCP collection.
	 * @returns (IObservable<boolean | undefined>): An observable of the trust status.
	 */
	public getTrust(collectionRef: McpCollectionReference): IObservable<boolean | undefined> {
		// Derived Observable: Reacts to changes in registered collections and trust memento.
		return derived(reader => {
			const collection = this._collections.read(reader).find(c => c.id === collectionRef.id);
			// Block Logic: If the collection is not found or trusted by default, return true.
			if (!collection || collection.isTrustedByDefault) {
				return true;
			}

			// Block Logic: Retrieves the trust decision from the memento.
			const memento = this._trustMemento.value.read(reader);
			return memento.hasOwnProperty(collection.id) ? memento[collection.id] : undefined;
		});
	}

	/**
	 * @brief Prompts the user for a trust decision for a given MCP collection.
	 *
	 * This internal method ensures that only one trust prompt dialog is open
	 * at a time for a specific collection, preventing redundant prompts.
	 *
	 * @param collection (McpCollectionDefinition): The collection to prompt for trust.
	 * @returns (Promise<boolean | undefined>): A promise that resolves to the user's trust decision (true, false, or undefined if dismissed).
	 */
	private _promptForTrust(collection: McpCollectionDefinition): Promise<boolean | undefined> {
		// Block Logic: Collects all trust prompts for a single config to prevent N dialogs.
		let resultPromise = this._trustPrompts.get(collection.id);
		// Block Logic: If no ongoing prompt, create a new one.
		resultPromise ??= this._promptForTrustOpenDialog(collection).finally(() => {
			this._trustPrompts.delete(collection.id); // Inline: Clears the promise from the map after completion.
		});
		this._trustPrompts.set(collection.id, resultPromise); // Inline: Stores the promise to track ongoing prompts.

		return resultPromise;
	}

	/**
	 * @brief Opens a dialog to prompt the user for trust for an MCP collection.
	 *
	 * This is the actual dialog presentation logic, separated to allow `_promptForTrust`
	 * to manage promise deduplication.
	 *
	 * @param collection (McpCollectionDefinition): The collection to prompt for trust.
	 * @returns (Promise<boolean | undefined>): A promise that resolves to the user's trust decision.
	 */
	private async _promptForTrustOpenDialog(collection: McpCollectionDefinition): Promise<boolean | undefined> {
		const originURI = collection.presentation?.origin;
		// Functional Utility: Formats a label that includes the origin URI if available.
		const labelWithOrigin = originURI ? `[\`${basename(originURI)}\`](${originURI})` : collection.label;

		// Block Logic: Presents a dialog to the user asking for trust.
		const result = await this._dialogService.prompt(
			{
				message: localize('trustTitleWithOrigin', 'Trust MCP servers from {0}?', collection.label),
				custom: {
					markdownDetails: [{
						markdown: new MarkdownString(localize('mcp.trust.details', '{0} discovered Model Context Protocol servers from {1} (`{2}`). {0} can use their capabilities in Chat.\n\nDo you want to allow running MCP servers from {3}?', this._productService.nameShort, collection.label, collection.serverDefinitions.get().map(s => s.label).join('`, `'), labelWithOrigin)),
						dismissOnLinkClick: true,
					}]
				},
				buttons: [
					{ label: localize('mcp.trust.yes', 'Trust'), run: () => true },
					{ label: localize('mcp.trust.no', 'Do not trust'), run: () => false }
				],
			},
		);

		return result.result;
	}

	/**
	 * @brief Updates the input storage with resolved expression inputs.
	 *
	 * Separates resolved inputs into plain text inputs and secrets,
	 * then stores them appropriately.
	 *
	 * @param inputStorage (McpRegistryInputStorage): The storage instance to update.
	 * @param expr (ConfigurationResolverExpression<unknown>): The expression containing resolved inputs.
	 * @returns (Promise<void>): A promise that resolves when storage is updated.
	 */
	private async _updateStorageWithExpressionInputs(inputStorage: McpRegistryInputStorage, expr: ConfigurationResolverExpression<unknown>): Promise<void> {
		const secrets: Record<string, IResolvedValue> = {};
		const inputs: Record<string, IResolvedValue> = {};
		// Block Logic: Categorizes resolved inputs into secrets or plain text.
		for (const [replacement, resolved] of expr.resolved()) {
			if (resolved.input?.type === 'promptString' && resolved.input.password) {
				secrets[replacement.id] = resolved;
			} else {
				inputs[replacement.id] = resolved;
			}
		}

		inputStorage.setPlainText(inputs); // Inline: Stores plain text inputs.
		await inputStorage.setSecrets(secrets); // Inline: Stores secret inputs.
		this._onDidChangeInputs.fire(); // Inline: Fires an event to notify of input changes.
	}

	/**
	 * @brief Replaces variables in an `McpServerLaunch` object using configuration resolution.
	 *
	 * This method resolves variables within a server launch configuration, potentially
	 * prompting the user for interactive inputs and storing the resolved values.
	 *
	 * @param definition (McpServerDefinition): The definition of the MCP server.
	 * @param launch (McpServerLaunch): The server launch configuration with variables to resolve.
	 * @returns (Promise<McpServerLaunch>): A promise that resolves to the launch configuration with all variables resolved.
	 */
	private async _replaceVariablesInLaunch(definition: McpServerDefinition, launch: McpServerLaunch): Promise<McpServerLaunch> {
		if (!definition.variableReplacement) {
			return launch; // Inline: No variable replacement needed.
		}

		const { section, target, folder } = definition.variableReplacement;
		// Block Logic: Determines the appropriate input storage scope.
		const inputStorage = target === ConfigurationTarget.WORKSPACE ? this._workspaceStorage.value : this._profileStorage.value;
		const previouslyStored = await inputStorage.getMap(); // Inline: Retrieves previously stored inputs.

		// Block Logic: Pre-fills variables that were previously resolved to avoid extra prompting.
		const expr = ConfigurationResolverExpression.parse(launch); // Inline: Parses the launch object for variables.
		for (const replacement of expr.unresolved()) {
			if (previouslyStored.hasOwnProperty(replacement.id)) {
				expr.resolve(replacement, previouslyStored[replacement.id]); // Inline: Resolves with previously stored value.
			}
		}

		// Block Logic: Interactively resolves variables that require user input.
		await this._configurationResolverService.resolveWithInteraction(folder, expr, section, undefined, target);

		// Block Logic: Updates the storage with any new or modified expression inputs.
		await this._updateStorageWithExpressionInputs(inputStorage, expr);

		// Block Logic: Resolves any remaining non-interactive variables and returns the final object.
		return await this._configurationResolverService.resolveAsync(folder, expr) as McpServerLaunch;
	}

	/**
	 * @brief Resolves an MCP server connection, handling trust and variable replacement.
	 *
	 * This is the core method for establishing a connection to an MCP server.
	 * It involves:
	 * 1. Finding the collection and server definition.
	 * 2. Identifying a suitable delegate to handle the connection.
	 * 3. Prompting for user trust if the collection is not trusted by default.
	 * 4. Resolving variables within the server's launch configuration.
	 * 5. Instantiating and returning an `IMcpServerConnection`.
	 *
	 * @param options (IMcpResolveConnectionOptions): Options for resolving the connection.
	 * @returns (Promise<IMcpServerConnection | undefined>): A promise that resolves to the server connection, or undefined if not trusted or an error occurs.
	 */
	public async resolveConnection({ collectionRef, definitionRef, forceTrust }: IMcpResolveConnectionOptions): Promise<IMcpServerConnection | undefined> {
		// Block Logic: Finds the collection and server definition based on references.
		const collection = this._collections.get().find(c => c.id === collectionRef.id);
		const definition = collection?.serverDefinitions.get().find(s => s.id === definitionRef.id);
		if (!collection || !definition) {
			throw new Error(`Collection or definition not found for ${collectionRef.id} and ${definitionRef.id}`);
		}

		// Block Logic: Finds a registered delegate that can start this specific server.
		const delegate = this._delegates.find(d => d.canStart(collection, definition));
		if (!delegate) {
			throw new Error('No delegate found that can handle the connection');
		}

		// Block Logic: Handles trust checking for the collection.
		if (!collection.isTrustedByDefault) {
			const memento = this._trustMemento.value.get();
			const trusted = memento.hasOwnProperty(collection.id) ? memento[collection.id] : undefined;

			if (trusted) {
				// Inline: Collection is explicitly trusted, continue.
			} else if (trusted === undefined || forceTrust) {
				// Block Logic: Prompts the user for a trust decision.
				const trustValue = await this._promptForTrust(collection);
				if (trustValue !== undefined) {
					// Inline: Stores the user's trust decision.
					this._trustMemento.value.set({ ...memento, [collection.id]: trustValue }, undefined);
				}
				if (!trustValue) {
					return; // Inline: If not trusted, return undefined.
				}
			} else /** trusted === false && !forceTrust */ {
				return undefined; // Inline: Explicitly not trusted, return undefined.
			}
		}

		let launch: McpServerLaunch | undefined;
		try {
			// Block Logic: Replaces variables in the server's launch configuration.
			launch = await this._replaceVariablesInLaunch(definition, definition.launch);
		} catch (e) {
			// Block Logic: Handles errors during variable replacement.
			this._notificationService.notify({
				severity: Severity.Error,
				message: localize('mcp.launchError', 'Error starting {0}: {1}', definition.label, String(e)),
				actions: {
					primary: collection.presentation?.origin && [
						{
							id: 'mcp.launchError.openConfig',
							class: undefined,
							enabled: true,
							tooltip: '',
							label: localize('mcp.launchError.openConfig', 'Open Configuration'),
							run: () => this._editorService.openEditor({
								resource: collection.presentation!.origin,
								options: { selection: definition.presentation?.origin?.range }
							}),
						}
					]
				}
			});
			return; // Inline: Return undefined due to launch error.
		}

		// Functional Utility: Instantiates and returns a new McpServerConnection object.
		return this._instantiationService.createInstance(
			McpServerConnection,
			collection,
			definition,
			delegate,
			launch,
		);
	}
}
