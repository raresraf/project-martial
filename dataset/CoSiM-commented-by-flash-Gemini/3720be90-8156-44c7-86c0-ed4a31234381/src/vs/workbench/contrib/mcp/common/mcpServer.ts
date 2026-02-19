/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
/**
 * @file mcpServer.ts
 * @module vs/workbench/contrib/mcp/common/mcpServer
 * @description Defines the core classes for managing Multi-Cloud Platform (MCP) server instances
 *              and the tools they provide. This includes handling server connections, caching
 *              tool metadata, and exposing server-bound tool interactions.
 */

// Functional Utility: Imports raceCancellationError for handling cancellation in promises and Sequencer for sequential promise execution.
import { raceCancellationError, Sequencer } from '../../../../base/common/async.js';
// Functional Utility: Imports CancellationToken and CancellationTokenSource for managing cancellable operations.
import { CancellationToken, CancellationTokenSource } from '../../../../base/common/cancellation.js';
// Functional Utility: Imports Disposable for managing disposable resources, DisposableStore for a collection of disposables, and toDisposable helper.
import { Disposable, DisposableStore, IDisposable, toDisposable } from '../../../../base/common/lifecycle.js';
// Functional Utility: Imports LRUCache for managing a least-recently-used cache.
import { LRUCache } from '../../../../base/common/map.js';
// Functional Utility: Imports observable utilities for reactive programming and state management, including autorun, derived, and observable values.
import { autorun, autorunWithStore, derived, disposableObservableValue, IObservable, ITransaction, observableFromEvent, ObservablePromise, observableValue, transaction } from '../../../../base/common/observable.js';
// Functional Utility: Imports IStorageService, StorageScope, and StorageTarget for persistent storage.
import { IStorageService, StorageScope, StorageTarget } from '../../../../platform/storage/common/storage.js';
// Functional Utility: Imports IWorkspaceContextService for workspace folder information.
import { IWorkspaceContextService } from '../../../../platform/workspace/common/workspace.js';
// Functional Utility: Imports IExtensionService for interacting with extensions.
import { IExtensionService } from '../../../services/extensions/common/extensions.js';
// Functional Utility: Imports mcpActivationEvent for determining extension activation events.
import { mcpActivationEvent } from './mcpConfiguration.js';
// Functional Utility: Imports IMcpRegistry interface for interacting with the MCP registry.
import { IMcpRegistry } from './mcpRegistryTypes.js';
// Functional Utility: Imports McpServerRequestHandler for handling MCP server requests.
import { McpServerRequestHandler } from './mcpServerRequestHandler.js';
// Functional Utility: Imports interfaces and types related to MCP servers, connections, tools, and states.
import { extensionMcpCollectionPrefix, IMcpServer, IMcpServerConnection, IMcpTool, McpCollectionReference, McpConnectionFailedError, McpConnectionState, McpDefinitionReference, McpServerDefinition, McpServerToolsState } from './mcpTypes.js';
// Functional Utility: Imports MCP protocol types, specifically MCP.Tool and MCP.JSONRPCMessage.
import { MCP } from './modelContextProtocol.js';


/**
 * @interface IToolCacheEntry
 * @brief Represents a cached entry for MCP tools.
 *
 * @property tools (readonly MCP.Tool[]): An array of cached MCP tool definitions.
 */
interface IToolCacheEntry {
	/** Cached tools so we can show what's available before it's started */
	readonly tools: readonly MCP.Tool[];
}

/**
 * @interface IServerCacheEntry
 * @brief Represents a cached entry for MCP server definitions.
 *
 * @property servers (readonly McpServerDefinition.Serialized[]): An array of cached serialized MCP server definitions.
 */
interface IServerCacheEntry {
	readonly servers: readonly McpServerDefinition.Serialized[];
}

/**
 * @class McpServerMetadataCache
 * @extends Disposable
 * @brief Manages caching of MCP server metadata, including tools and server definitions.
 *
 * This cache persists server and tool information across sessions, allowing the workbench
 * to display available tools even before a server fully starts or an extension activates.
 */
export class McpServerMetadataCache extends Disposable {
	private didChange = false; // Flag to track if the cache has changed and needs to be saved.
	private readonly cache = new LRUCache<string, IToolCacheEntry>(128); // LRU cache for server tools.
	private readonly extensionServers = new Map</* collection ID */string, IServerCacheEntry>(); // Map for extension servers cache.

	constructor(
		scope: StorageScope,
		@IStorageService storageService: IStorageService,
	) {
		super();

		type StoredType = { // Type definition for the data stored in persistent storage.
			extensionServers: [string, IServerCacheEntry][];
			serverTools: [string, IToolCacheEntry][];
		};

		const storageKey = 'mcpToolCache';
		// Block Logic: Registers a listener to save the cache state when the application is about to save its state.
		this._register(storageService.onWillSaveState(() => {
			if (this.didChange) { // Precondition: Only save if changes have occurred.
				storageService.store(storageKey, {
					extensionServers: [...this.extensionServers],
					serverTools: this.cache.toJSON(),
				} satisfies StoredType, scope, StorageTarget.MACHINE);
				this.didChange = false; // Inline: Resets the change flag.
			}
		}));

		// Block Logic: Attempts to load cached data from storage during initialization.
		try {
			const cached: StoredType | undefined = storageService.getObject(storageKey, scope);
			this.extensionServers = new Map(cached?.extensionServers ?? []); // Functional Utility: Restores extension servers from cache.
			cached?.serverTools?.forEach(([k, v]) => this.cache.set(k, v)); // Functional Utility: Restores server tools from cache.
		} catch {
			// Inline: If there's an error loading the cache, it's ignored.
		}
	}

	/**
	 * @brief Resets the entire cache for both tools and extension servers.
	 *
	 * This clears all cached metadata and marks the cache as changed so it
	 * will not be re-persisted.
	 */
	reset(): void {
		this.cache.clear();
		this.extensionServers.clear();
		this.didChange = true; // Inline: Marks the cache as dirty, ensuring it gets reset in storage.
	}

	/**
	 * @brief Retrieves cached tools for a specific server definition ID.
	 * @param definitionId (string): The ID of the server definition.
	 * @returns (readonly MCP.Tool[] | undefined): An array of cached tools, or undefined if not found.
	 */
	getTools(definitionId: string): readonly MCP.Tool[] | undefined {
		return this.cache.get(definitionId)?.tools;
	}

	/**
	 * @brief Stores tools in the cache for a specific server definition ID.
	 * @param definitionId (string): The ID of the server definition.
	 * @param tools (readonly MCP.Tool[]): The array of tools to cache.
	 */
	storeTools(definitionId: string, tools: readonly MCP.Tool[]): void {
		this.cache.set(definitionId, { ...this.cache.get(definitionId), tools });
		this.didChange = true; // Inline: Marks the cache as dirty.
	}

	/**
	 * @brief Retrieves cached server definitions for a specific collection ID.
	 * @param collectionId (string): The ID of the collection.
	 * @returns (IServerCacheEntry | undefined): The cached server entry, or undefined if not found.
	 */
	getServers(collectionId: string): IServerCacheEntry | undefined {
		return this.extensionServers.get(collectionId);
	}

	/**
	 * @brief Stores server definitions in the cache for a specific collection ID.
	 * @param collectionId (string): The ID of the collection.
	 * @param entry (IServerCacheEntry | undefined): The server entry to cache. If undefined, the entry is deleted.
	 */
	storeServers(collectionId: string, entry: IServerCacheEntry | undefined): void {
		if (entry) {
			this.extensionServers.set(collectionId, entry);
		} else {
			this.extensionServers.delete(collectionId);
		}
		this.didChange = true; // Inline: Marks the cache as dirty.
	}
}

/**
 * @class McpServer
 * @extends Disposable
 * @implements IMcpServer
 * @brief Represents a single MCP server instance, managing its connection, state, and provided tools.
 *
 * This class handles the lifecycle of an MCP server, including starting, stopping,
 * monitoring its connection state, caching its tools, and integrating with the
 * extension activation process.
 */
export class McpServer extends Disposable implements IMcpServer {
	private readonly _connectionSequencer = new Sequencer(); // Sequencer to ensure connection operations run serially.
	// Disposable ObservableValue: Holds the active MCP server connection.
	private readonly _connection = this._register(disposableObservableValue<IMcpServerConnection | undefined>(this, undefined));

	// IObservable: Publicly exposed observable of the current MCP server connection.
	public readonly connection = this._connection;
	// IObservable: Derived observable reflecting the server's connection state.
	public readonly connectionState: IObservable<McpConnectionState> = derived(reader => this._connection.read(reader)?.state.read(reader) ?? { state: McpConnectionState.Kind.Stopped });

	// Getter: Retrieves tools from the cache for this server.
	private get toolsFromCache(): readonly MCP.Tool[] | undefined {
		return this._toolCache.getTools(this.definition.id);
	}
	// ObservableValue: Holds a promise that resolves to tools obtained directly from the server.
	private readonly toolsFromServerPromise = observableValue<ObservablePromise<readonly MCP.Tool[]> | undefined>(this, undefined);
	// IObservable: Derived observable of tools obtained directly from the server (resolved promise data).
	private readonly toolsFromServer = derived(reader => this.toolsFromServerPromise.read(reader)?.promiseResult.read(reader)?.data);

	// IObservable: Publicly exposed observable of all available IMcpTool instances for this server.
	public readonly tools: IObservable<readonly IMcpTool[]>;

	// IObservable: Derived observable reflecting the current state of tools for this server.
	public readonly toolsState = derived(reader => {
		const fromServer = this.toolsFromServerPromise.read(reader);
		const connectionState = this.connectionState.read(reader);
		// Block Logic: Checks if the server is idle (stopped/can be started and no tools promise is active).
		const isIdle = McpConnectionState.canBeStarted(connectionState.state) && !fromServer;
		if (isIdle) {
			// Functional Utility: If idle, determine state based on cache presence.
			return this.toolsFromCache ? McpServerToolsState.Cached : McpServerToolsState.Unknown;
		}

		// Block Logic: If not idle, check the result of the toolsFromServerPromise.
		const fromServerResult = fromServer?.promiseResult.read(reader);
		if (!fromServerResult) {
			// Functional Utility: If still loading from server, indicate refreshing state based on cache presence.
			return this.toolsFromCache ? McpServerToolsState.RefreshingFromCached : McpServerToolsState.RefreshingFromUnknown;
		}

		// Functional Utility: If promise resolved, check for errors or live state.
		return fromServerResult.error ? (this.toolsFromCache ? McpServerToolsState.Cached : McpServerToolsState.Unknown) : McpServerToolsState.Live;
	});

	// IObservable: Publicly exposed observable reflecting the trust status of the collection this server belongs to.
	public get trusted(): IObservable<boolean | undefined> {
		return this._mcpRegistry.getTrust(this.collection);
	}

	constructor(
		public readonly collection: McpCollectionReference,
		public readonly definition: McpDefinitionReference,
		private readonly _requiresExtensionActivation: boolean | undefined,
		private readonly _toolCache: McpServerMetadataCache,
		@IMcpRegistry private readonly _mcpRegistry: IMcpRegistry,
		@IWorkspaceContextService workspacesService: IWorkspaceContextService,
		@IExtensionService private readonly _extensionService: IExtensionService,
	) {
		super();

		// 1. Block Logic: Reflects workspace folder changes into the MCP server's root paths.
		const workspaces = observableFromEvent(
			this,
			workspacesService.onDidChangeWorkspaceFolders,
			() => workspacesService.getWorkspace().folders,
		);

		this._register(autorunWithStore(reader => {
			const cnx = this._connection.read(reader)?.handler.read(reader);
			if (!cnx) {
				return;
			}

			// Functional Utility: Updates the connection's root paths with current workspace folders.
			cnx.roots = workspaces.read(reader).map(wf => ({
				uri: wf.uri.toString(),
				name: wf.name,
			}));
		}));

		// 2. Block Logic: Populates live tool data when a server connection is established.
		this._register(autorunWithStore((reader, store) => {
			const cnx = this._connection.read(reader)?.handler.read(reader);
			if (cnx) {
				this.populateLiveData(cnx, store); // Functional Utility: Calls helper to populate live data.
			} else {
				this.resetLiveData(); // Functional Utility: Resets live data when connection is lost.
			}
		}));

		// 3. Block Logic: Updates the tool cache when new tools are received from the server.
		this._register(autorun(reader => {
			const tools = this.toolsFromServer.read(reader);
			if (tools) {
				this._toolCache.storeTools(definition.id, tools); // Functional Utility: Stores tools in the cache.
			}
		}));

		// 4. Block Logic: Publishes (provides) the tools offered by this server.
		const toolPrefix = this._mcpRegistry.collectionToolPrefix(this.collection);
		this.tools = derived(reader => {
			const serverTools = this.toolsFromServer.read(reader);
			// Functional Utility: Uses tools from server, then cache, then empty array as fallback.
			const definitions = serverTools ?? this.toolsFromCache ?? [];
			const prefix = toolPrefix.read(reader);
			return definitions.map(def => new McpTool(this, prefix, def)); // Functional Utility: Maps raw tool definitions to McpTool instances.
		});
	}

	/**
	 * @brief Shows the output channel associated with this MCP server.
	 */
	public showOutput(): void {
		this._connection.get()?.showOutput();
	}

	/**
	 * @brief Starts the MCP server connection.
	 *
	 * This method handles extension activation if required, resolves the connection
	 * via the registry, and manages the connection lifecycle.
	 *
	 * @param isFromInteraction (boolean): True if the start request originated from user interaction (e.g., explicit start button).
	 * @returns (Promise<McpConnectionState>): A promise that resolves to the final connection state.
	 */
	public start(isFromInteraction?: boolean): Promise<McpConnectionState> {
		// Block Logic: Queues the connection operation to ensure sequential execution.
		return this._connectionSequencer.queue(async () => {
			const activationEvent = mcpActivationEvent(this.collection.id.slice(extensionMcpCollectionPrefix.length));
			// Block Logic: Handles extension activation if required for the collection.
			// Precondition: `_requiresExtensionActivation` is true and the extension has not been activated yet.
			if (this._requiresExtensionActivation && !this._extensionService.activationEventIsDone(activationEvent)) {
				await this._extensionService.activateByEvent(activationEvent); // Functional Utility: Activates extension by event.
				// Block Logic: Waits for initial provider promises from all delegates to be resolved.
				await Promise.all(this._mcpRegistry.delegates
					.map(r => r.waitForInitialProviderPromises()));
				// Precondition: Checks if the server was disposed during activation.
				if (this._store.isDisposed) {
					return { state: McpConnectionState.Kind.Stopped }; // Inline: Returns stopped state if disposed.
				}
			}

			let connection = this._connection.get();
			// Block Logic: If an existing connection can be started (e.g., stopped or error), dispose it and reset.
			if (connection && McpConnectionState.canBeStarted(connection.state.get().state)) {
				connection.dispose();
				connection = undefined;
				this._connection.set(connection, undefined);
			}

			// Block Logic: If no active connection, resolve a new one.
			if (!connection) {
				connection = await this._mcpRegistry.resolveConnection({
					collectionRef: this.collection,
					definitionRef: this.definition,
					forceTrust: isFromInteraction, // Inline: Force trust prompt if from user interaction.
				});
				if (!connection) {
					return { state: McpConnectionState.Kind.Stopped }; // Inline: Return stopped state if connection resolution fails.
				}

				// Precondition: Checks if the server was disposed during connection resolution.
				if (this._store.isDisposed) {
					connection.dispose(); // Inline: Dispose the connection if the server is gone.
					return { state: McpConnectionState.Kind.Stopped };
				}

				this._connection.set(connection, undefined); // Inline: Sets the new active connection.
			}

			return connection.start(); // Functional Utility: Starts the resolved connection.
		});
	}

	/**
	 * @brief Stops the MCP server connection.
	 * @returns (Promise<void>): A promise that resolves when the server is stopped.
	 */
	public stop(): Promise<void> {
		return this._connection.get()?.stop() || Promise.resolve();
	}

	/**
	 * @brief Resets the live data associated with the server (e.g., tools from the server).
	 *
	 * This typically happens when the server connection is lost.
	 */
	private resetLiveData(): void {
		transaction(tx => {
			this.toolsFromServerPromise.set(undefined, tx); // Inline: Clears the tools from server promise.
		});
	}

	/**
	 * @brief Populates live data (e.g., tools) from the MCP server.
	 *
	 * This method subscribes to changes in the tool list from the server handler
	 * and updates the `toolsFromServerPromise` observable.
	 *
	 * @param handler (McpServerRequestHandler): The request handler for the MCP server.
	 * @param store (DisposableStore): A disposable store to manage subscriptions.
	 */
	private populateLiveData(handler: McpServerRequestHandler, store: DisposableStore): void {
		const cts = new CancellationTokenSource();
		store.add(toDisposable(() => cts.dispose(true))); // Inline: Disposes cancellation token source when store is disposed.

		// todo: add more than just tools here

		// Functional Utility: Updates the toolsFromServerPromise by fetching tools from the handler.
		const updateTools = (tx: ITransaction | undefined) => {
			const toolPromise = handler.capabilities.tools ? handler.listTools({}, cts.token) : Promise.resolve([]);
			const toolPromiseSafe = toolPromise.then(tools => {
				handler.logger.info(`Discovered ${tools.length} tools`);
				return tools.map(tool => {
					if (!tool.description) {
						// Functional Utility: Ensures each tool has a description, logging a warning if missing.
						handler.logger.warn(`Tool ${tool.name} does not have a description. Tools must be accurately described to be called`);
						tool.description = '<empty>'; // Inline: Provides a default description if missing.
					}

					return tool;
				});
			});
			this.toolsFromServerPromise.set(new ObservablePromise(toolPromiseSafe), tx);
		};

		// Block Logic: Subscribes to tool list changes from the server handler and refreshes tools.
		store.add(handler.onDidChangeToolList(() => {
			handler.logger.info('Tool list changed, refreshing tools...');
			updateTools(undefined); // Inline: Updates tools without a transaction.
		}));

		transaction(tx => { // Functional Utility: Performs initial tools update within a transaction.
			updateTools(tx);
		});
	}

	/**
	 * @brief Helper function to call a function on the MCP server request handler once it's online.
	 *
	 * Ensures the server is started before attempting to call the function on its handler.
	 * Handles connection failures and server stopping.
	 *
	 * @param fn (function): The function to call on the `McpServerRequestHandler`.
	 * @param token (CancellationToken): An optional cancellation token.
	 * @returns (Promise<R>): A promise that resolves with the result of `fn`, or rejects on error.
	 */
	public async callOn<R>(fn: (handler: McpServerRequestHandler) => Promise<R>, token: CancellationToken = CancellationToken.None): Promise<R> {

		await this.start(); // Functional Utility: Ensures the server is started (idempotent operation).

		let ranOnce = false; // Flag to ensure the function `fn` is called only once.
		let d: IDisposable; // Disposable for the autorun reaction.

		// Block Logic: Creates a promise that resolves when the handler is available and `fn` is called.
		const callPromise = new Promise<R>((resolve, reject) => {

			// Functional Utility: Uses autorun to react to changes in the connection handler.
			d = autorun(reader => {
				const connection = this._connection.read(reader);
				if (!connection || ranOnce) {
					return;
				}

				const handler = connection.handler.read(reader);
				if (!handler) { // Block Logic: If handler is not yet available, check connection state.
					const state = connection.state.read(reader);
					if (state.state === McpConnectionState.Kind.Error) {
						reject(new McpConnectionFailedError(`MCP server could not be started: ${state.message}`)); // Inline: Rejects on connection error.
						return;
					} else if (state.state === McpConnectionState.Kind.Stopped) {
						reject(new McpConnectionFailedError('MCP server has stopped')); // Inline: Rejects if server stopped.
						return;
					} else {
						// Inline: Keep waiting for the handler to become available.
						return;
					}
				}

				resolve(fn(handler)); // Inline: Resolves the promise by calling `fn` on the handler.
				ranOnce = true; // Inline: Marks that `fn` has been called.
			});
		});

		// Block Logic: Races the call promise against a cancellation error and ensures cleanup.
		return raceCancellationError(callPromise, token).finally(() => d.dispose());
	}
}

/**
 * @class McpTool
 * @implements IMcpTool
 * @brief Represents a single tool provided by an MCP server.
 *
 * This class wraps a raw `MCP.Tool` definition from the server,
 * providing an ID and a method to remotely call the tool on its server.
 */
export class McpTool implements IMcpTool {

	readonly id: string; // Unique identifier for the tool.

	constructor(
		private readonly _server: McpServer,
		idPrefix: string,
		public readonly definition: MCP.Tool,
	) {
		// Functional Utility: Constructs the tool ID by combining the prefix and tool name,
		// replacing dots with underscores for compatibility.
		this.id = (idPrefix + definition.name).replaceAll('.', '_');
	}

	/**
	 * @brief Calls the tool on its associated MCP server.
	 * @param params (Record<string, unknown>): Parameters to pass to the tool call.
	 * @param token (CancellationToken): An optional cancellation token.
	 * @returns (Promise<MCP.CallToolResult>): A promise that resolves with the result of the tool call.
	 */
	call(params: Record<string, unknown>, token?: CancellationToken): Promise<MCP.CallToolResult> {
		return this._server.callOn(h => h.callTool({ name: this.definition.name, arguments: params }), token);
	}
}
