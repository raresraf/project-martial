/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
/**
 * @file mcpRegistryTypes.ts
 * @module vs/workbench/contrib/mcp/common/mcpRegistryTypes
 * @description Defines interfaces and types that form the contract for the Multi-Cloud Platform (MCP)
 *              registry and its interactions within the VS Code workbench. These types enable
 *              dependency injection and ensure consistent communication between MCP components.
 */

// Functional Utility: Imports Event for event handling.
import { Event } from '../../../../base/common/event.js';
// Functional Utility: Imports IDisposable for managing disposable resources.
import { IDisposable } from '../../../../base/common/lifecycle.js';
// Functional Utility: Imports IObservable for reactive programming and state management.
import { IObservable } from '../../../../base/common/observable.js';
// Functional Utility: Imports ConfigurationTarget for specifying where configuration changes apply.
import { ConfigurationTarget } from '../../../../platform/configuration/common/configuration.js';
// Functional Utility: Imports createDecorator for creating service decorators for dependency injection.
import { createDecorator } from '../../../../platform/instantiation/common/instantiation.js';
// Functional Utility: Imports StorageScope for defining storage scopes.
import { StorageScope } from '../../../../platform/storage/common/storage.js';
// Functional Utility: Imports IWorkspaceFolderData for workspace folder information.
import { IWorkspaceFolderData } from '../../../../platform/workspace/common/workspace.js';
// Functional Utility: Imports IResolvedValue type for resolved configuration variables.
import { IResolvedValue } from '../../../services/configurationResolver/common/configurationResolverExpression.js';
// Functional Utility: Imports IMcpServerConnection, LazyCollectionState, McpCollectionDefinition, etc., related to MCP server types.
import { IMcpServerConnection, LazyCollectionState, McpCollectionDefinition, McpCollectionReference, McpConnectionState, McpDefinitionReference, McpServerDefinition, McpServerLaunch } from './mcpTypes.js';
// Functional Utility: Imports MCP types from 'modelContextProtocol.js', specifically MCP.JSONRPCMessage.
import { MCP } from './modelContextProtocol.js';

/**
 * @const IMcpRegistry
 * @brief A unique service identifier for the MCP Registry.
 *
 * This decorator is used to retrieve the `IMcpRegistry` instance via dependency injection.
 */
export const IMcpRegistry = createDecorator<IMcpRegistry>('mcpRegistry');

/**
 * @interface IMcpMessageTransport
 * @brief Defines the interface for message transport to a single MCP server.
 *
 * This interface abstracts the communication layer with an MCP server, providing
 * methods to send and receive JSON-RPC messages and observe connection state and logs.
 */
export interface IMcpMessageTransport extends IDisposable {
	/**
	 * @property state
	 * @brief An observable representing the current connection state of the MCP server.
	 */
	readonly state: IObservable<McpConnectionState>;
	/**
	 * @property onDidLog
	 * @brief An event that fires when a new log message is received from the MCP server.
	 */
	readonly onDidLog: Event<string>;
	/**
	 * @property onDidReceiveMessage
	 * @brief An event that fires when a JSON-RPC message is received from the MCP server.
	 */
	readonly onDidReceiveMessage: Event<MCP.JSONRPCMessage>;
	/**
	 * @method send
	 * @brief Sends a JSON-RPC message to the MCP server.
	 * @param message (MCP.JSONRPCMessage): The JSON-RPC message to send.
	 */
	send(message: MCP.JSONRPCMessage): void;
	/**
	 * @method stop
	 * @brief Stops the message transport connection to the MCP server.
	 */
	stop(): void;
}

/**
 * @interface IMcpHostDelegate
 * @brief Defines the interface for a delegate that can handle starting MCP servers.
 *
 * Host delegates provide environment-specific logic for managing MCP server processes.
 */
export interface IMcpHostDelegate {
	/**
	 * @method waitForInitialProviderPromises
	 * @brief Waits for any initial provider promises to resolve before proceeding.
	 * @returns (Promise<void>): A promise that resolves when initial providers are ready.
	 */
	waitForInitialProviderPromises(): Promise<void>;
	/**
	 * @method canStart
	 * @brief Checks if the delegate is capable of starting a given MCP server.
	 * @param collectionDefinition (McpCollectionDefinition): The definition of the MCP collection.
	 * @param serverDefinition (McpServerDefinition): The definition of the MCP server.
	 * @returns (boolean): True if the delegate can start the server, False otherwise.
	 */
	canStart(collectionDefinition: McpCollectionDefinition, serverDefinition: McpServerDefinition): boolean;
	/**
	 * @method start
	 * @brief Starts the MCP server and returns an `IMcpMessageTransport` for communication.
	 * @param collectionDefinition (McpCollectionDefinition): The definition of the MCP collection.
	 * @param serverDefinition (McpServerDefinition): The definition of the MCP server.
	 * @param resolvedLaunch (McpServerLaunch): The resolved launch configuration for the server.
	 * @returns (IMcpMessageTransport): The message transport instance for the started server.
	 */
	start(collectionDefinition: McpCollectionDefinition, serverDefinition: McpServerDefinition, resolvedLaunch: McpServerLaunch): IMcpMessageTransport;
}

/**
 * @interface IMcpResolveConnectionOptions
 * @brief Defines options for resolving an MCP server connection.
 */
export interface IMcpResolveConnectionOptions {
	/**
	 * @property collectionRef
	 * @brief Reference to the MCP collection.
	 */
	collectionRef: McpCollectionReference;
	/**
	 * @property definitionRef
	 * @brief Reference to the specific MCP server definition within the collection.
	 */
	definitionRef: McpDefinitionReference;
	/**
	 * @property forceTrust
	 * @brief Optional. If true, the user will be prompted to trust the collection
	 *        even if they had previously untrusted it.
	 */
	forceTrust?: boolean;
}

/**
 * @interface IMcpRegistry
 * @brief Defines the interface for the MCP Registry service.
 *
 * This service is responsible for managing MCP collections, trust, configuration inputs,
 * and facilitating connections to MCP servers.
 */
export interface IMcpRegistry {
	readonly _serviceBrand: undefined; // Functional Utility: Marks this interface as a service.

	/**
	 * @property onDidChangeInputs
	 * @brief An event that fires when the user provides or changes inputs when creating a connection.
	 */
	readonly onDidChangeInputs: Event<void>;

	/**
	 * @property collections
	 * @brief An observable providing a read-only array of all registered MCP collection definitions.
	 */
	readonly collections: IObservable<readonly McpCollectionDefinition[]>;
	/**
	 * @property delegates
	 * @brief A read-only array of all registered MCP host delegates.
	 */
	readonly delegates: readonly IMcpHostDelegate[];
	/**
	 * @property lazyCollectionState
	 * @brief An observable reflecting the current state of lazy collections,
	 *        indicating whether there are new collections to discover.
	 */
	readonly lazyCollectionState: IObservable<LazyCollectionState>;

	/**
	 * @method collectionToolPrefix
	 * @brief Gets the prefix that should be applied to a collection's tools to avoid ID conflicts.
	 * @param collection (McpCollectionReference): The reference to the MCP collection.
	 * @returns (IObservable<string>): An observable that provides the tool prefix.
	 */
	collectionToolPrefix(collection: McpCollectionReference): IObservable<string>;

	/**
	 * @method discoverCollections
	 * @brief Discovers new (lazy-loaded) collections and returns any newly-discovered ones.
	 * @returns (Promise<McpCollectionDefinition[]>): A promise that resolves to an array of newly discovered collection definitions.
	 */
	discoverCollections(): Promise<McpCollectionDefinition[]>;

	/**
	 * @method registerDelegate
	 * @brief Registers a new MCP host delegate.
	 * @param delegate (IMcpHostDelegate): The delegate to register.
	 * @returns (IDisposable): A disposable to unregister the delegate.
	 */
	registerDelegate(delegate: IMcpHostDelegate): IDisposable;
	/**
	 * @method registerCollection
	 * @brief Registers a new MCP collection definition.
	 * @param collection (McpCollectionDefinition): The collection definition to register.
	 * @returns (IDisposable): A disposable to unregister the collection.
	 */
	registerCollection(collection: McpCollectionDefinition): IDisposable;

	/**
	 * @method resetTrust
	 * @brief Resets the trust state of all MCP collections, effectively untrusting them.
	 */
	resetTrust(): void;

	/**
	 * @method getTrust
	 * @brief Gets the trust status of a specific MCP collection.
	 * @param collection (McpCollectionReference): The reference to the MCP collection.
	 * @returns (IObservable<boolean | undefined>): An observable of the trust status (true for trusted, false for untrusted, undefined for no decision).
	 */
	getTrust(collection: McpCollectionReference): IObservable<boolean | undefined>;

	/**
	 * @method clearSavedInputs
	 * @brief Clears any saved inputs for a given input ID, or all saved inputs within a scope.
	 * @param scope (StorageScope): The storage scope from which to clear inputs.
	 * @param inputId (string): Optional. The ID of the specific input to clear.
	 * @returns (Promise<void>): A promise that resolves when the inputs are cleared.
	 */
	clearSavedInputs(scope: StorageScope, inputId?: string): Promise<void>;
	/**
	 * @method editSavedInput
	 * @brief Allows editing a previously-saved input, potentially prompting the user for new values.
	 * @param inputId (string): The ID of the input to edit.
	 * @param folderData (IWorkspaceFolderData): Optional. Workspace folder data for context.
	 * @param configSection (string): The configuration section where the input is used.
	 * @param target (ConfigurationTarget): The configuration target for saving the edited input.
	 * @returns (Promise<void>): A promise that resolves when the input is edited.
	 */
	editSavedInput(inputId: string, folderData: IWorkspaceFolderData | undefined, configSection: string, target: ConfigurationTarget): Promise<void>;
	/**
	 * @method getSavedInputs
	 * @brief Retrieves saved inputs from storage for a given scope.
	 * @param scope (StorageScope): The storage scope from which to retrieve inputs.
	 * @returns (Promise<{ [id: string]: IResolvedValue }>): A promise that resolves to a map of input IDs to their resolved values.
	 */
	getSavedInputs(scope: StorageScope): Promise<{ [id: string]: IResolvedValue }>;
	/**
	 * @method resolveConnection
	 * @brief Resolves and establishes a connection for the specified MCP collection and server definition.
	 *
	 * This method handles trust checking, variable replacement in launch configurations,
	 * and delegates to the appropriate host to start the server.
	 *
	 * @param options (IMcpResolveConnectionOptions): Options for resolving the connection.
	 * @returns (Promise<IMcpServerConnection | undefined>): A promise that resolves to the server connection,
	 *          or undefined if the connection cannot be resolved (e.g., not trusted, error).
	 */
	resolveConnection(options: IMcpResolveConnectionOptions): Promise<IMcpServerConnection | undefined>;
}
