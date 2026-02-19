/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
/**
 * @file mcpTypes.ts
 * @module vs/workbench/contrib/mcp/common/mcpTypes
 * @description Defines all essential interfaces, enums, and types for the Multi-Cloud Platform (MCP)
 *              feature within the VS Code workbench. This includes fundamental data structures
 *              for MCP collections, server definitions, connection states, and tool interactions.
 */

// Functional Utility: Imports assertNever utility for exhaustiveness checks in switch statements.
import { assertNever } from '../../../../base/common/assert.js';
// Functional Utility: Imports CancellationToken for managing cancellable operations.
import { CancellationToken } from '../../../../base/common/cancellation.js';
// Functional Utility: Imports IDisposable for managing disposable resources.
import { IDisposable } from '../../../../base/common/lifecycle.js';
// Functional Utility: Imports objectsEqual for deep comparison of objects.
import { equals as objectsEqual } from '../../../../base/common/objects.js';
// Functional Utility: Imports IObservable for reactive programming and state management.
import { IObservable } from '../../../../base/common/observable.js';
// Functional Utility: Imports URI and UriComponents for handling uniform resource identifiers.
import { URI, UriComponents } from '../../../../base/common/uri.js';
// Functional Utility: Imports Location type from editor common languages.
import { Location } from '../../../../editor/common/languages.js';
// Functional Utility: Imports localization function.
import { localize } from '../../../../nls.js';
// Functional Utility: Imports ConfigurationTarget for specifying where configuration changes apply.
import { ConfigurationTarget } from '../../../../platform/configuration/common/configuration.js';
// Functional Utility: Imports ExtensionIdentifier for working with extension identifiers.
import { ExtensionIdentifier } from '../../../../platform/extensions/common/extensions.js';
// Functional Utility: Imports createDecorator for creating service decorators for dependency injection.
import { createDecorator } from '../../../../platform/instantiation/common/instantiation.js';
// Functional Utility: Imports StorageScope for defining storage scopes.
import { StorageScope } from '../../../../platform/storage/common/storage.js';
// Functional Utility: Imports IWorkspaceFolderData for workspace folder information.
import { IWorkspaceFolderData } from '../../../../platform/workspace/common/workspace.js';
// Functional Utility: Imports McpServerRequestHandler for server request handling.
import { McpServerRequestHandler } from './mcpServerRequestHandler.js';
// Functional Utility: Imports MCP protocol types, specifically MCP.Tool and MCP.JSONRPCMessage.
import { MCP } from './modelContextProtocol.js';

/**
 * @const extensionMcpCollectionPrefix
 * @brief Prefix used for MCP collection IDs that originate from extensions.
 */
export const extensionMcpCollectionPrefix = 'ext.';

/**
 * @function extensionPrefixedIdentifier
 * @brief Generates a prefixed identifier for an MCP collection that originates from an extension.
 * @param identifier (ExtensionIdentifier): The identifier of the extension.
 * @param id (string): The unique ID provided by the extension.
 * @returns (string): The prefixed identifier (e.g., "publisher.extensionId/customId").
 */
export function extensionPrefixedIdentifier(identifier: ExtensionIdentifier, id: string): string {
	return ExtensionIdentifier.toKey(identifier) + '/' + id;
}

/**
 * @interface McpCollectionDefinition
 * @brief Defines the structure for an MCP collection.
 *
 * An `McpCollection` represents a logical grouping of `McpServers`.
 * There can be multiple collections, potentially discovered from different sources
 * or locations.
 */
export interface McpCollectionDefinition {
	/**
	 * @property remoteAuthority
	 * @brief The authority from which this collection was discovered (e.g., remote host URI), or null if local.
	 */
	readonly remoteAuthority: string | null;
	/**
	 * @property id
	 * @brief A globally-unique, stable identifier for this definition.
	 */
	readonly id: string;
	/**
	 * @property label
	 * @brief A human-readable label for the definition.
	 */
	readonly label: string;
	/**
	 * @property serverDefinitions
	 * @brief An observable providing a read-only array of `McpServerDefinition` objects
	 *        contained within this collection.
	 */
	readonly serverDefinitions: IObservable<readonly McpServerDefinition[]>;
	/**
	 * @property isTrustedByDefault
	 * @brief If 'false', explicit user consent (trust) is required before any MCP servers
	 *        in this collection are automatically launched.
	 */
	readonly isTrustedByDefault: boolean;
	/**
	 * @property scope
	 * @brief The storage scope where associated collection info should be stored.
	 */
	readonly scope: StorageScope;

	/**
	 * @property lazy
	 * @brief Optional properties for lazy-loaded collections.
	 *
	 * This property exists only for collections that are initially placeholder
	 * and need to be fully loaded (e.g., from an extension activation).
	 */
	readonly lazy?: {
		/**
		 * @property isCached
		 * @brief True if `serverDefinitions` for this lazy collection were loaded from a cache.
		 */
		isCached: boolean;
		/**
		 * @method load
		 * @brief Triggers the loading of the real server definition for this lazy collection.
		 *        The loaded definition should then be pushed to the `IMcpRegistry`.
		 * @returns (Promise<void>): A promise that resolves when the loading is complete.
		 */
		load(): Promise<void>;
		/**
		 * @method removed
		 * @brief Called after `load()` if the lazy collection's providing extension is not found
		 *        or the collection is removed without being replaced by a non-lazy version.
		 */
		removed?(): void;
	};

	/**
	 * @property presentation
	 * @brief Optional presentation-related properties for the collection.
	 */
	readonly presentation?: {
		/**
		 * @property order
		 * @brief Defines the sort order of the collection in UI displays.
		 */
		readonly order?: number;
		/**
		 * @property origin
		 * @brief The URI pointing to where this collection is configured (e.g., a settings file).
		 *        Used in workspace trust prompts and "show config" actions.
		 */
		readonly origin?: URI;
	};
}

/**
 * @enum McpCollectionSortOrder
 * @brief Defines standard sort order values for MCP collections.
 *
 * These values are used in `McpCollectionDefinition.presentation.order`
 * to ensure consistent ordering of collections in the UI.
 */
export const enum McpCollectionSortOrder {
	Workspace = 0,
	User = 100,
	Extension = 200,
	Filesystem = 300,

	RemotePenalty = 50, // Penalty added for remote collections to sort them lower.
}

export namespace McpCollectionDefinition {
	/**
	 * @interface FromExtHost
	 * @brief Defines the structure of an MCP collection definition as received from an extension host.
	 */
	export interface FromExtHost {
		readonly id: string;
		readonly label: string;
		readonly isTrustedByDefault: boolean;
		readonly scope: StorageScope;
	}

	/**
	 * @function equals
	 * @brief Compares two `McpCollectionDefinition` objects for equality.
	 * @param a (McpCollectionDefinition): The first collection definition.
	 * @param b (McpCollectionDefinition): The second collection definition.
	 * @returns (boolean): True if the collections are equal, False otherwise.
	 */
	export function equals(a: McpCollectionDefinition, b: McpCollectionDefinition): boolean {
		return a.id === b.id
			&& a.remoteAuthority === b.remoteAuthority
			&& a.label === b.label
			&& a.isTrustedByDefault === b.isTrustedByDefault;
	}
}

/**
 * @interface McpServerDefinition
 * @brief Defines the structure for a single MCP server.
 *
 * An `McpServerDefinition` specifies how to launch and identify an MCP server.
 */
export interface McpServerDefinition {
	/**
	 * @property id
	 * @brief A globally-unique, stable identifier for this server definition.
	 */
	readonly id: string;
	/**
	 * @property label
	 * @brief A human-readable label for this server, used in UI.
	 */
	readonly label: string;
	/**
	 * @property launch
	 * @brief A descriptor defining how the MCP server process should be launched.
	 */
	readonly launch: McpServerLaunch;
	/**
	 * @property variableReplacement
	 * @brief Optional. If set, allows configuration variables to be resolved in the {@link launch} with the given context.
	 */
	readonly variableReplacement?: McpServerDefinitionVariableReplacement;

	/**
	 * @property presentation
	 * @brief Optional presentation-related properties for the server.
	 */
	readonly presentation?: {
		/**
		 * @property order
		 * @brief Defines the sort order of the server definition in UI displays.
		 */
		readonly order?: number;
		/**
		 * @property origin
		 * @brief The location (URI and optional range) where this server is configured.
		 *        Used in "show config" actions.
		 */
		readonly origin?: Location;
	};
}

export namespace McpServerDefinition {
	/**
	 * @interface Serialized
	 * @brief Represents a serializable version of `McpServerDefinition`.
	 *
	 * This is used for caching or persisting server definitions.
	 */
	export interface Serialized {
		readonly id: string;
		readonly label: string;
		readonly launch: McpServerLaunch.Serialized;
		readonly variableReplacement?: McpServerDefinitionVariableReplacement.Serialized;
	}

	/**
	 * @function toSerialized
	 * @brief Converts an `McpServerDefinition` to its serializable format.
	 * @param def (McpServerDefinition): The definition to serialize.
	 * @returns (McpServerDefinition.Serialized): The serialized definition.
	 */
	export function toSerialized(def: McpServerDefinition): McpServerDefinition.Serialized {
		// Functional Utility: Direct conversion is possible as properties are compatible or handled by sub-serialization.
		return def;
	}

	/**
	 * @function fromSerialized
	 * @brief Converts a serialized `McpServerDefinition` back to its original format.
	 * @param def (McpServerDefinition.Serialized): The serialized definition.
	 * @returns (McpServerDefinition): The deserialized definition.
	 */
	export function fromSerialized(def: McpServerDefinition.Serialized): McpServerDefinition {
		return {
			id: def.id,
			label: def.label,
			launch: McpServerLaunch.fromSerialized(def.launch),
			variableReplacement: def.variableReplacement ? McpServerDefinitionVariableReplacement.fromSerialized(def.variableReplacement) : undefined,
		};
	}

	/**
	 * @function equals
	 * @brief Compares two `McpServerDefinition` objects for equality.
	 * @param a (McpServerDefinition): The first server definition.
	 * @param b (McpServerDefinition): The second server definition.
	 * @returns (boolean): True if the server definitions are equal, False otherwise.
	 */
	export function equals(a: McpServerDefinition, b: McpServerDefinition): boolean {
		return a.id === b.id
			&& a.label === b.label
			&& objectsEqual(a.launch, b.launch) // Functional Utility: Deep comparison for launch configuration.
			&& objectsEqual(a.variableReplacement, b.variableReplacement); // Functional Utility: Deep comparison for variable replacement.
	}
}


/**
 * @interface McpServerDefinitionVariableReplacement
 * @brief Defines the context for resolving variables within an `McpServerLaunch` configuration.
 */
export interface McpServerDefinitionVariableReplacement {
	/**
	 * @property section
	 * @brief Optional. The configuration section (e.g., 'mcp') where variables might be defined.
	 */
	section?: string; // e.g. 'mcp'
	/**
	 * @property folder
	 * @brief Optional. Workspace folder data to provide context for variable resolution.
	 */
	folder?: IWorkspaceFolderData;
	/**
	 * @property target
	 * @brief The configuration target (e.g., Global, Workspace) for the variables.
	 */
	target: ConfigurationTarget;
}

export namespace McpServerDefinitionVariableReplacement {
	/**
	 * @interface Serialized
	 * @brief Represents a serializable version of `McpServerDefinitionVariableReplacement`.
	 */
	export interface Serialized {
		target: ConfigurationTarget;
		section?: string;
		folder?: { name: string; index: number; uri: UriComponents }; // UriComponents for serializing URI.
	}

	/**
	 * @function toSerialized
	 * @brief Converts an `McpServerDefinitionVariableReplacement` to its serializable format.
	 * @param def (McpServerDefinitionVariableReplacement): The definition to serialize.
	 * @returns (McpServerDefinitionVariableReplacement.Serialized): The serialized definition.
	 */
	export function toSerialized(def: McpServerDefinitionVariableReplacement): McpServerDefinitionVariableReplacement.Serialized {
		// Functional Utility: Direct conversion is possible as properties are compatible or handled by URI serialization.
		return def;
	}

	/**
	 * @function fromSerialized
	 * @brief Converts a serialized `McpServerDefinitionVariableReplacement` back to its original format.
	 * @param def (McpServerDefinitionVariableReplacement.Serialized): The serialized definition.
	 * @returns (McpServerDefinitionVariableReplacement): The deserialized definition.
	 */
	export function fromSerialized(def: McpServerDefinitionVariableReplacement.Serialized): McpServerDefinitionVariableReplacement {
		return {
			section: def.section,
			folder: def.folder ? { ...def.folder, uri: URI.revive(def.folder.uri) } : undefined, // Functional Utility: Revives URI from UriComponents.
			target: def.target,
		};
	}
}

/**
 * @interface IMcpService
 * @brief Defines the interface for the MCP service.
 *
 * This service manages the overall state and interactions with MCP servers
 * and their collections across the workbench.
 */
export interface IMcpService {
	_serviceBrand: undefined; // Functional Utility: Marks this interface as a service.
	/**
	 * @property servers
	 * @brief An observable providing a read-only array of all active `IMcpServer` instances.
	 */
	readonly servers: IObservable<readonly IMcpServer[]>;

	/**
	 * @method resetCaches
	 * @brief Resets all cached MCP tools and server metadata.
	 */
	resetCaches(): void;

	/**
	 * @property lazyCollectionState
	 * @brief An observable indicating the state of lazy collections (e.g., if there are
	 *        extensions that register MCP servers that have never been activated).
	 */
	readonly lazyCollectionState: IObservable<LazyCollectionState>;
	/**
	 * @method activateCollections
	 * @brief Activates extensions that provide MCP collections and runs their associated servers.
	 * @returns (Promise<void>): A promise that resolves when collections are activated.
	 */
	activateCollections(): Promise<void>;
}

/**
 * @enum LazyCollectionState
 * @brief Represents the state of lazy-loaded MCP collections.
 */
export const enum LazyCollectionState {
	/** There are collections that exist but haven't been fully discovered/activated yet. */
	HasUnknown,
	/** Currently loading (activating) unknown collections. */
	LoadingUnknown,
	/** All known collections have been discovered and activated. */
	AllKnown,
}

/**
 * @const IMcpService
 * @brief A unique service identifier for the MCP Service.
 *
 * This decorator is used to retrieve the `IMcpService` instance via dependency injection.
 */
export const IMcpService = createDecorator<IMcpService>('IMcpService');

/**
 * @interface McpCollectionReference
 * @brief A lightweight reference to an MCP collection.
 */
export interface McpCollectionReference {
	/**
	 * @property id
	 * @brief The unique ID of the collection.
	 */
	id: string;
	/**
	 * @property label
	 * @brief The human-readable label of the collection.
	 */
	label: string;
	/**
	 * @property presentation
	 * @brief Optional presentation-related properties of the collection.
	 */
	presentation?: McpCollectionDefinition['presentation'];
}

/**
 * @interface McpDefinitionReference
 * @brief A lightweight reference to an MCP server definition.
 */
export interface McpDefinitionReference {
	/**
	 * @property id
	 * @brief The unique ID of the server definition.
	 */
	id: string;
	/**
	 * @property label
	 * @brief The human-readable label of the server definition.
	 */
	label: string;
}

/**
 * @interface IMcpServer
 * @extends IDisposable
 * @brief Represents an active MCP server instance.
 *
 * This interface provides access to the server's state, connection, tools,
 * and methods to control its lifecycle.
 */
export interface IMcpServer extends IDisposable {
	/**
	 * @property collection
	 * @brief Reference to the MCP collection this server belongs to.
	 */
	readonly collection: McpCollectionReference;
	/**
	 * @property definition
	 * @brief Reference to the specific MCP server definition for this server.
	 */
	readonly definition: McpDefinitionReference;
	/**
	 * @property connection
	 * @brief An observable providing the current `IMcpServerConnection` instance, or undefined if not connected.
	 */
	readonly connection: IObservable<IMcpServerConnection | undefined>;
	/**
	 * @property connectionState
	 * @brief An observable reflecting the current connection state of the MCP server.
	 */
	readonly connectionState: IObservable<McpConnectionState>;
	/**
	 * @property trusted
	 * @brief An observable reflecting the trust state of the MCP server's collection.
	 *        True if trusted, false if untrusted, undefined if consent is required but not indicated.
	 */
	readonly trusted: IObservable<boolean | undefined>;

	/**
	 * @method showOutput
	 * @brief Displays the output channel associated with this server.
	 */
	showOutput(): void;
	/**
	 * @method start
	 * @brief Starts the MCP server if it's currently stopped.
	 * @param isFromInteraction (boolean): Optional. True if the start request originated from user interaction.
	 * @returns (Promise<McpConnectionState>): A promise that resolves to the server's final connection state
	 *          (Running, Error, or Stopped if disposed/cancelled).
	 */
	start(isFromInteraction?: boolean): Promise<McpConnectionState>;
	/**
	 * @method stop
	 * @brief Stops the MCP server.
	 * @returns (Promise<void>): A promise that resolves when the server is fully stopped.
	 */
	stop(): Promise<void>;

	/**
	 * @property toolsState
	 * @brief An observable reflecting the current state of tools provided by this server.
	 */
	readonly toolsState: IObservable<McpServerToolsState>;
	/**
	 * @property tools
	 * @brief An observable providing a read-only array of `IMcpTool` instances available from this server.
	 */
	readonly tools: IObservable<readonly IMcpTool[]>;
}

/**
 * @enum McpServerToolsState
 * @brief Represents the discovery and availability state of tools for an MCP server.
 */
export const enum McpServerToolsState {
	/** Tools have not yet been read or discovered. */
	Unknown,
	/** Tools were read from a persistent cache (server not actively connected or refreshing). */
	Cached,
	/** Tools are currently being refreshed from the server for the first time. */
	RefreshingFromUnknown,
	/** Tools are currently being refreshed from the server, but cached tools are available. */
	RefreshingFromCached,
	/** The tool state is live and synchronized with the actively connected server. */
	Live,
}

/**
 * @interface IMcpTool
 * @brief Defines the interface for an MCP tool provided by a server.
 *
 * This interface represents a callable tool, providing its definition and
 * a method to invoke it on the server.
 */
export interface IMcpTool {
	/**
	 * @property id
	 * @brief A unique identifier for the tool, typically prefixed by its collection.
	 */
	readonly id: string;

	/**
	 * @property definition
	 * @brief The raw `MCP.Tool` definition as provided by the server.
	 */
	readonly definition: MCP.Tool;

	/**
	 * @method call
	 * @brief Calls (executes) the tool on its associated MCP server.
	 * @param params (Record<string, unknown>): Parameters to pass to the tool call.
	 * @param token (CancellationToken): Optional. A cancellation token to abort the call.
	 * @returns (Promise<MCP.CallToolResult>): A promise that resolves with the result of the tool call.
	 * @throws {@link MpcResponseError} if the tool fails to execute on the server.
	 * @throws {@link McpConnectionFailedError} if the connection to the server fails.
	 */
	call(params: Record<string, unknown>, token?: CancellationToken): Promise<MCP.CallToolResult>;
}

/**
 * @enum McpServerTransportType
 * @brief Defines the available transport mechanisms for communicating with an MCP server.
 */
export const enum McpServerTransportType {
	/** An MCP server communicating over standard input/output streams. */
	Stdio = 1 << 0,
	/** An MCP server that communicates using Server-Sent Events (SSE) over HTTP. */
	SSE = 1 << 1,
}

/**
 * @interface McpServerTransportStdio
 * @brief Defines the launch configuration for an MCP server communicating via standard I/O.
 * @see https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio
 */
export interface McpServerTransportStdio {
	/**
	 * @property type
	 * @brief The transport type, always `McpServerTransportType.Stdio`.
	 */
	readonly type: McpServerTransportType.Stdio;
	/**
	 * @property cwd
	 * @brief The current working directory for the server process, or undefined.
	 */
	readonly cwd: URI | undefined;
	/**
	 * @property command
	 * @brief The executable command to launch the server.
	 */
	readonly command: string;
	/**
	 * @property args
	 * @brief A read-only array of string arguments to pass to the command.
	 */
	readonly args: readonly string[];
	/**
	 * @property env
	 * @brief A record of environment variables to set for the server process.
	 */
	readonly env: Record<string, string | number | null>;
	/**
	 * @property envFile
	 * @brief An optional path to a file containing environment variables.
	 */
	readonly envFile: string | undefined;
}

/**
 * @interface McpServerTransportSSE
 * @brief Defines the launch configuration for an MCP server communicating via Server-Sent Events (SSE).
 * @see https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse
 */
export interface McpServerTransportSSE {
	/**
	 * @property type
	 * @brief The transport type, always `McpServerTransportType.SSE`.
	 */
	readonly type: McpServerTransportType.SSE;
	/**
	 * @property uri
	 * @brief The URI endpoint for the SSE connection.
	 */
	readonly uri: URI;
	/**
	 * @property headers
	 * @brief A read-only array of HTTP headers to send with the SSE request.
	 */
	readonly headers: [string, string][];
}

/**
 * @type McpServerLaunch
 * @brief A union type representing the launch configuration for an MCP server,
 *        which can be either `Stdio` or `SSE` based.
 */
export type McpServerLaunch =
	| McpServerTransportStdio
	| McpServerTransportSSE;

export namespace McpServerLaunch {
	/**
	 * @interface Serialized
	 * @brief Represents a serializable version of `McpServerLaunch`.
	 *
	 * Uses `UriComponents` for serializing URI objects.
	 */
	export type Serialized =
		| { type: McpServerTransportType.SSE; uri: UriComponents; headers: [string, string][] }
		| { type: McpServerTransportType.Stdio; cwd: UriComponents | undefined; command: string; args: readonly string[]; env: Record<string, string | number | null>; envFile: string | undefined };

	/**
	 * @function toSerialized
	 * @brief Converts an `McpServerLaunch` object to its serializable format.
	 * @param launch (McpServerLaunch): The launch configuration to serialize.
	 * @returns (McpServerLaunch.Serialized): The serialized launch configuration.
	 */
	export function toSerialized(launch: McpServerLaunch): McpServerLaunch.Serialized {
		// Functional Utility: Direct conversion as properties are compatible or handled by URI serialization.
		return launch;
	}

	/**
	 * @function fromSerialized
	 * @brief Converts a serialized `McpServerLaunch` object back to its original format.
	 * @param launch (McpServerLaunch.Serialized): The serialized launch configuration.
	 * @returns (McpServerLaunch): The deserialized launch configuration.
	 */
	export function fromSerialized(launch: McpServerLaunch.Serialized): McpServerLaunch {
		switch (launch.type) {
			case McpServerTransportType.SSE:
				return { type: launch.type, uri: URI.revive(launch.uri), headers: launch.headers };
			case McpServerTransportType.Stdio:
				return {
					type: launch.type,
					cwd: launch.cwd ? URI.revive(launch.cwd) : undefined, // Functional Utility: Revives URI from UriComponents for cwd.
					command: launch.command,
					args: launch.args,
					env: launch.env,
					envFile: launch.envFile,
				};
		}
	}
}

/**
 * @interface IMcpServerConnection
 * @extends IDisposable
 * @brief Defines the interface for an instance that manages a live connection to an MCP server.
 *
 * This interface provides methods to control the connection's lifecycle
 * and access its current state and request handler.
 */
export interface IMcpServerConnection extends IDisposable {
	/**
	 * @property definition
	 * @brief The `McpServerDefinition` for the server this connection is managing.
	 */
	readonly definition: McpServerDefinition;
	/**
	 * @property state
	 * @brief An observable reflecting the current connection state of the MCP server.
	 */
	readonly state: IObservable<McpConnectionState>;
	/**
	 * @property handler
	 * @brief An observable providing the `McpServerRequestHandler` when the server is
	 *        in a running state, or undefined otherwise.
	 */
	readonly handler: IObservable<McpServerRequestHandler | undefined>;

	/**
	 * @method showOutput
	 * @brief Shows the output channel associated with this server connection.
	 */
	showOutput(): void;

	/**
	 * @method start
	 * @brief Starts the MCP server if it is currently stopped.
	 * @returns (Promise<McpConnectionState>): A promise that resolves once the server
	 *          exits a 'starting' state (e.g., Running, Error, Stopped).
	 */
	start(): Promise<McpConnectionState>;

	/**
	 * @method stop
	 * @brief Stops the MCP server.
	 * @returns (Promise<void>): A promise that resolves when the server is fully stopped.
	 */
	stop(): Promise<void>;
}

/**
 * @namespace McpConnectionState
 * @brief Defines the connection states for an MCP server and related utility functions.
 */
export namespace McpConnectionState {
	/**
	 * @enum Kind
	 * @brief Represents the distinct kinds of connection states for an MCP server.
	 */
	export const enum Kind {
		Stopped,
		Starting,
		Running,
		Error,
	}

	/**
	 * @function toString
	 * @brief Converts an `McpConnectionState` object into a human-readable string.
	 * @param s (McpConnectionState): The connection state object.
	 * @returns (string): A localized string representation of the state.
	 */
	export const toString = (s: McpConnectionState): string => {
		switch (s.state) {
			case Kind.Stopped:
				return localize('mcpstate.stopped', 'Stopped');
			case Kind.Starting:
				return localize('mcpstate.starting', 'Starting');
			case Kind.Running:
				return localize('mcpstate.running', 'Running');
			case Kind.Error:
				return localize('mcpstate.error', 'Error {0}', s.message);
			default:
				assertNever(s); // Inline: Ensures all enum cases are handled for exhaustiveness.
		}
	};

	/**
	 * @function canBeStarted
	 * @brief Checks if a given `McpConnectionState.Kind` allows the server to be started.
	 * @param s (Kind): The connection state kind.
	 * @returns (boolean): True if the server can be started from this state (Error or Stopped), False otherwise.
	 */
	export const canBeStarted = (s: Kind) => s === Kind.Error || s === Kind.Stopped;

	/**
	 * @function isRunning
	 * @brief Checks if the given `McpConnectionState` represents a running state.
	 * @param s (McpConnectionState): The connection state object.
	 * @returns (boolean): True if the server is running, False otherwise.
	 */
	export const isRunning = (s: McpConnectionState) => !canBeStarted(s.state);

	/**
	 * @interface Stopped
	 * @brief Represents the `Stopped` connection state.
	 */
	export interface Stopped {
		readonly state: Kind.Stopped;
	}

	/**
	 * @interface Starting
	 * @brief Represents the `Starting` connection state.
	 */
	export interface Starting {
		readonly state: Kind.Starting;
	}

	/**
	 * @interface Running
	 * @brief Represents the `Running` connection state.
	 */
	export interface Running {
		readonly state: Kind.Running;
	}

	/**
	 * @interface Error
	 * @brief Represents the `Error` connection state, including an error message.
	 */
	export interface Error {
		readonly state: Kind.Error;
		readonly message: string;
	}
}

/**
 * @type McpConnectionState
 * @brief A union type representing all possible connection states for an MCP server.
 */
export type McpConnectionState =
	| McpConnectionState.Stopped
	| McpConnectionState.Starting
	| McpConnectionState.Running
	| McpConnectionState.Error;

/**
 * @class MpcResponseError
 * @extends Error
 * @brief Represents an error returned from an MCP server response.
 *
 * @property message (string): The error message.
 * @property code (number): The error code from the MCP response.
 * @property data (unknown): Additional error data.
 */
export class MpcResponseError extends Error {
	constructor(message: string, public readonly code: number, public readonly data: unknown) {
		super(`MPC ${code}: ${message}`);
	}
}

/**
 * @class McpConnectionFailedError
 * @extends Error
 * @brief Represents an error indicating that the connection to an MCP server failed.
 */
export class McpConnectionFailedError extends Error { }
