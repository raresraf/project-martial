/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
/**
 * @file mcpServerConnection.ts
 * @module vs/workbench/contrib/mcp/common/mcpServerConnection
 * @description Implements the connection management for a single Multi-Cloud Platform (MCP) server.
 *              This class is responsible for launching and stopping the server, monitoring its
 *              connection state, handling its output, and managing its request handler.
 */

// Functional Utility: Imports CancellationTokenSource for managing cancellable operations.
import { CancellationTokenSource } from '../../../../base/common/cancellation.js';
// Functional Utility: Imports Disposable for managing disposable resources, DisposableStore for a collection of disposables, IReference for managing references, and MutableDisposable for disposable values that can change.
import { Disposable, DisposableStore, IReference, MutableDisposable, toDisposable } from '../../../../base/common/lifecycle.js';
// Functional Utility: Imports observable utilities for reactive programming and state management, including autorun and observableValue.
import { autorun, IObservable, observableValue } from '../../../../base/common/observable.js';
// Functional Utility: Imports localization function.
import { localize } from '../../../../nls.js';
// Functional Utility: Imports IInstantiationService for creating objects with dependency injection.
import { IInstantiationService } from '../../../../platform/instantiation/common/instantiation.js';
// Functional Utility: Imports ILogger and ILoggerService for logging functionality.
import { ILogger, ILoggerService } from '../../../../platform/log/common/log.js';
// Functional Utility: Imports IOutputService for managing output channels.
import { IOutputService } from '../../../services/output/common/output.js';
// Functional Utility: Imports IMcpHostDelegate and IMcpMessageTransport interfaces.
import { IMcpHostDelegate, IMcpMessageTransport } from './mcpRegistryTypes.js';
// Functional Utility: Imports McpServerRequestHandler for handling MCP server requests.
import { McpServerRequestHandler } from './mcpServerRequestHandler.js';
// Functional Utility: Imports McpCollectionDefinition, IMcpServerConnection, etc., related to MCP server types.
import { McpCollectionDefinition, IMcpServerConnection, McpServerDefinition, McpConnectionState, McpServerLaunch } from './mcpTypes.js';

/**
 * @class McpServerConnection
 * @extends Disposable
 * @implements IMcpServerConnection
 * @brief Manages the lifecycle and state of a single connection to an MCP server.
 *
 * This class handles the actual process of starting and stopping an MCP server,
 * adapting its communication transport, logging its output, and providing a
 * request handler for interacting with the server's capabilities.
 */
export class McpServerConnection extends Disposable implements IMcpServerConnection {
	// MutableDisposable: Holds a reference to the message transport for the server launch.
	private readonly _launch = this._register(new MutableDisposable<IReference<IMcpMessageTransport>>());
	// ObservableValue: Tracks the current connection state of the MCP server.
	private readonly _state = observableValue<McpConnectionState>('mcpServerState', { state: McpConnectionState.Kind.Stopped });
	// ObservableValue: Holds the request handler for the MCP server when connected.
	private readonly _requestHandler = observableValue<McpServerRequestHandler | undefined>('mcpServerRequestHandler', undefined);

	// IObservable: Publicly exposed observable of the connection state.
	public readonly state: IObservable<McpConnectionState> = this._state;
	// IObservable: Publicly exposed observable of the request handler.
	public readonly handler: IObservable<McpServerRequestHandler | undefined> = this._requestHandler;

	private readonly _loggerId: string; // Unique ID for the logger.
	private readonly _logger: ILogger; // Logger instance for the server.
	private _launchId = 0; // Counter to track launch attempts and prevent stale connections/handlers.

	constructor(
		private readonly _collection: McpCollectionDefinition,
		public readonly definition: McpServerDefinition,
		private readonly _delegate: IMcpHostDelegate,
		public readonly launchDefinition: McpServerLaunch,
		@ILoggerService private readonly _loggerService: ILoggerService,
		@IOutputService private readonly _outputService: IOutputService,
		@IInstantiationService private readonly _instantiationService: IInstantiationService,
	) {
		super();
		this._loggerId = `mcpServer/${definition.id}`;
		// Block Logic: Creates a logger for this MCP server, with a hidden output channel by default.
		this._logger = this._register(_loggerService.createLogger(this._loggerId, { hidden: true, name: `MCP: ${definition.label}` }));
		// Block Logic: Ensures the logger is deregistered when this connection is disposed.
		this._register(toDisposable(() => _loggerService.deregisterLogger(this._loggerId)));
	}

	/** @inheritdoc */
	public showOutput(): void {
		this._loggerService.setVisibility(this._loggerId, true); // Inline: Makes the logger visible.
		this._outputService.showChannel(this._loggerId); // Inline: Shows the output channel in the UI.
	}

	/** @inheritdoc */
	public async start(): Promise<McpConnectionState> {
		const currentState = this._state.get();
		// Precondition: If the server is already starting or running, wait for it to reach a stable state.
		if (!McpConnectionState.canBeStarted(currentState.state)) {
			return this._waitForState(McpConnectionState.Kind.Running, McpConnectionState.Kind.Error);
		}

		const launchId = ++this._launchId; // Inline: Increments launch ID to invalidate older launch attempts.
		this._launch.value = undefined; // Inline: Clears any previous launch.
		this._state.set({ state: McpConnectionState.Kind.Starting }, undefined); // Inline: Sets state to Starting.
		this._logger.info(localize('mcpServer.starting', 'Starting server {0}', this.definition.label)); // Inline: Logs server start attempt.

		try {
			// Block Logic: Delegates to the host to start the actual message transport.
			const launch = this._delegate.start(this._collection, this.definition, this.launchDefinition);
			this._launch.value = this.adoptLaunch(launch, launchId); // Functional Utility: Adopts the new launch and manages its lifecycle.
			// Block Logic: Waits for the server to transition to Running or Error state.
			return this._waitForState(McpConnectionState.Kind.Running, McpConnectionState.Kind.Error);
		} catch (e) {
			// Block Logic: Handles errors during the server launch process.
			const errorState: McpConnectionState = {
				state: McpConnectionState.Kind.Error,
				message: e instanceof Error ? e.message : String(e)
			};
			this._state.set(errorState, undefined); // Inline: Sets connection state to Error.
			return errorState;
		}
	}

	/**
	 * @brief Adopts a new message transport launch and manages its lifecycle.
	 *
	 * This method sets up listeners for the launch's state and log events,
	 * and creates a `McpServerRequestHandler` when the connection is running.
	 * It also handles potential race conditions with multiple launch attempts.
	 *
	 * @param launch (IMcpMessageTransport): The message transport instance.
	 * @param launchId (number): The unique ID of this launch attempt.
	 * @returns (IReference<IMcpMessageTransport>): A reference to the message transport.
	 */
	private adoptLaunch(launch: IMcpMessageTransport, launchId: number): IReference<IMcpMessageTransport> {
		const store = new DisposableStore();
		const cts = new CancellationTokenSource(); // Functional Utility: CancellationTokenSource for the request handler.

		store.add(toDisposable(() => cts.dispose(true))); // Inline: Disposes CTS when store is disposed.
		store.add(launch); // Inline: Adds the launch transport to the disposable store.
		// Block Logic: Logs all messages from the transport to the server's logger.
		store.add(launch.onDidLog(msg => {
			this._logger.info(msg);
		}));

		let didStart = false; // Flag to ensure handler creation happens only once per running state.
		// Block Logic: Reacts to changes in the launch's connection state.
		store.add(autorun(reader => {
			const state = launch.state.read(reader);
			this._state.set(state, undefined); // Inline: Updates the connection state observable.
			this._logger.info(localize('mcpServer.state', 'Connection state: {0}', McpConnectionState.toString(state))); // Inline: Logs state changes.

			// Block Logic: When the connection becomes Running and the handler hasn't been created yet.
			if (state.state === McpConnectionState.Kind.Running && !didStart) {
				didStart = true;
				// Block Logic: Creates an McpServerRequestHandler instance.
				McpServerRequestHandler.create(this._instantiationService, launch, this._logger, cts.token).then(
					handler => {
						// Block Logic: Checks if this is still the active launch before setting the handler.
						if (this._launchId === launchId) {
							this._requestHandler.set(handler, undefined); // Inline: Sets the request handler.
						} else {
							handler.dispose(); // Inline: Dispose the handler if it's for an outdated launch.
						}
					},
					err => {
						store.dispose(); // Inline: Dispose all resources if handler creation fails.
						// Block Logic: If still the active launch, log error and set state.
						if (this._launchId === launchId) {
							this._logger.error(err);
							this._state.set({ state: McpConnectionState.Kind.Error, message: `Could not initialize MCP server: ${err.message}` }, undefined);
						}
					},
				);
			}
		}));

		return { dispose: () => store.dispose(), object: launch };
	}

	/** @inheritdoc */
	public async stop(): Promise<void> {
		this._launchId = -1; // Inline: Invalidates current launch ID to prevent subsequent handler creation.
		this._logger.info(localize('mcpServer.stopping', 'Stopping server {0}', this.definition.label)); // Inline: Logs server stop attempt.
		this._launch.value?.object.stop(); // Inline: Calls stop on the active message transport.
		// Block Logic: Waits for the server to transition to Stopped or Error state after stopping.
		await this._waitForState(McpConnectionState.Kind.Stopped, McpConnectionState.Kind.Error);
	}

	/** @inheritdoc */
	public override dispose(): void {
		this._launchId = -1; // Inline: Ensures no new handlers are set after disposal.
		this._requestHandler.get()?.dispose(); // Inline: Disposes the request handler if it exists.
		super.dispose(); // Inline: Calls the base Disposable dispose method.
		this._state.set({ state: McpConnectionState.Kind.Stopped }, undefined); // Inline: Sets final state to Stopped.
	}

	/**
	 * @brief Waits for the server connection to reach one of the specified states.
	 *
	 * If the current state is already one of the target kinds, the promise resolves immediately.
	 * Otherwise, it sets up an autorun to wait for a state change.
	 *
	 * @param kinds (McpConnectionState.Kind[]): An array of target connection states to wait for.
	 * @returns (Promise<McpConnectionState>): A promise that resolves to the connection state when one of the target kinds is reached.
	 */
	private _waitForState(...kinds: McpConnectionState.Kind[]): Promise<McpConnectionState> {
		const current = this._state.get();
		// Precondition: If the current state is already a target state, resolve immediately.
		if (kinds.includes(current.state)) {
			return Promise.resolve(current);
		}

		// Block Logic: Creates a promise that resolves when the state transitions to one of the target kinds.
		return new Promise(resolve => {
			const disposable = autorun(reader => {
				const state = this._state.read(reader);
				if (kinds.includes(state.state)) {
					disposable.dispose(); // Inline: Disposes the autorun once the target state is reached.
					resolve(state);
				}
			});
		});
	}
}
