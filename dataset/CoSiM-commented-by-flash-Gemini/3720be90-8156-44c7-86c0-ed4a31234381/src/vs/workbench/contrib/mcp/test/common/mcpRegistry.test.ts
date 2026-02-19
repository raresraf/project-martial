/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
/**
 * @file mcpRegistry.test.ts
 * @module vs/workbench/contrib/mcp/test/common/mcpRegistry.test
 * @description Unit tests for the `McpRegistry` class.
 *              These tests ensure the correct functionality of MCP collection registration,
 *              host delegate management, connection resolution with variable replacement,
 *              trust management, and lazy collection handling.
 */

// Functional Utility: Imports Node.js 'assert' module for assertion functions.
import * as assert from 'assert';
// Functional Utility: Imports sinon for creating test spies, stubs, and mocks.
import * as sinon from 'sinon';
// Functional Utility: Imports timeout utility for asynchronous delays.
import { timeout } from '../../../../../base/common/async.js';
// Functional Utility: Imports observable types for testing observable properties.
import { ISettableObservable, observableValue } from '../../../../../base/common/observable.js';
// Functional Utility: Imports upcast for type casting.
import { upcast } from '../../../../../base/common/types.js';
// Functional Utility: Imports URI for handling uniform resource identifiers.
import { URI } from '../../../../../base/common/uri.js';
// Functional Utility: Imports ensureNoDisposablesAreLeakedInTestSuite for memory leak detection in tests.
import { ensureNoDisposablesAreLeakedInTestSuite } from '../../../../../base/test/common/utils.js';
// Functional Utility: Imports ConfigurationTarget for configuration scope.
import { ConfigurationTarget } from '../../../../../platform/configuration/common/configuration.js';
// Functional Utility: Imports IDialogService for interacting with user dialogs.
import { IDialogService } from '../../../../../platform/dialogs/common/dialogs.js';
// Functional Utility: Imports ServiceCollection for dependency injection setup.
import { ServiceCollection } from '../../../../../platform/instantiation/common/serviceCollection.js';
// Functional Utility: Imports TestInstantiationService for testing dependency injection.
import { TestInstantiationService } from '../../../../../platform/instantiation/test/common/instantiationServiceMock.js';
// Functional Utility: Imports ILoggerService for logging.
import { ILoggerService } from '../../../../../platform/log/common/log.js';
// Functional Utility: Imports IProductService for product-specific information.
import { IProductService } from '../../../../../platform/product/common/productService.js';
// Functional Utility: Imports ISecretStorageService for secret management.
import { ISecretStorageService } from '../../../../../platform/secrets/common/secrets.js';
// Functional Utility: Imports TestSecretStorageService for testing secret storage.
import { TestSecretStorageService } from '../../../../../platform/secrets/test/common/testSecretStorageService.js';
// Functional Utility: Imports IStorageService and StorageScope for persistent storage.
import { IStorageService, StorageScope } from '../../../../../platform/storage/common/storage.js';
// Functional Utility: Imports IConfigurationResolverService for resolving configuration variables.
import { IConfigurationResolverService } from '../../../../services/configurationResolver/common/configurationResolver.js';
// Functional Utility: Imports IOutputService for output channels.
import { IOutputService } from '../../../../services/output/common/output.js';
// Functional Utility: Imports TestLoggerService and TestStorageService for testing services.
import { TestLoggerService, TestStorageService } from '../../../../test/common/workbenchTestServices.js';
// Functional Utility: Imports the McpRegistry class under test.
import { McpRegistry } from '../../common/mcpRegistry.js';
// Functional Utility: Imports IMcpHostDelegate and IMcpMessageTransport interfaces.
import { IMcpHostDelegate, IMcpMessageTransport } from '../../common/mcpRegistryTypes.js';
// Functional Utility: Imports McpServerConnection class.
import { McpServerConnection } from '../../common/mcpServerConnection.js';
// Functional Utility: Imports types and enums related to MCP collections and servers.
import { LazyCollectionState, McpCollectionDefinition, McpCollectionReference, McpServerDefinition, McpServerTransportType } from '../../common/mcpTypes.js';
// Functional Utility: Imports TestMcpMessageTransport for mocking message transport.
import { TestMcpMessageTransport } from './mcpRegistryTypes.js'; // Note: This import path suggests TestMcpMessageTransport is defined in mcpRegistryTypes.ts in this test directory.
// Functional Utility: Imports ConfigurationResolverExpression for variable resolution.
import { ConfigurationResolverExpression } from '../../../../services/configurationResolver/common/configurationResolverExpression.js';

/**
 * @class TestConfigurationResolverService
 * @brief A test stub for `IConfigurationResolverService`.
 *
 * This class provides a simplified implementation of `IConfigurationResolverService`
 * for use in unit tests, allowing control over variable resolution behavior
 * and simulating interactive prompts.
 */
class TestConfigurationResolverService implements Partial<IConfigurationResolverService> {
	declare readonly _serviceBrand: undefined;

	private interactiveCounter = 0; // Counter for unique interactive values.
	private readonly resolvedVariables = new Map<string, string>(); // Stores simulated resolved variables.

	constructor() {
		// Block Logic: Initializes with some predefined test variables.
		this.resolvedVariables.set('workspaceFolder', '/test/workspace');
		this.resolvedVariables.set('fileBasename', 'test.txt');
	}

	/**
	 * @brief Simulates asynchronous variable resolution.
	 * @param folder (any): Workspace folder context.
	 * @param value (any): The value containing variables to resolve.
	 * @returns (Promise<any>): A promise resolving to the object with variables resolved.
	 */
	resolveAsync(folder: any, value: any): Promise<any> {
		const parsed = ConfigurationResolverExpression.parse(value);
		// Block Logic: Resolves unresolved variables with simulated values.
		for (const variable of parsed.unresolved()) {
			const resolved = this.resolvedVariables.get(variable.inner);
			if (resolved) {
				parsed.resolve(variable, resolved);
			}
		}
		return Promise.resolve(parsed.toObject());
	}

	/**
	 * @brief Simulates interactive variable resolution.
	 * @param folder (any): Workspace folder context.
	 * @param config (any): The configuration to resolve.
	 * @param section (string): Optional. The configuration section.
	 * @param variables (Record<string, string>): Optional. Pre-provided variables.
	 * @param target (ConfigurationTarget): Optional. The configuration target.
	 * @returns (Promise<Map<string, string> | undefined>): A promise resolving to a map of resolved interactive variables.
	 */
	resolveWithInteraction(folder: any, config: any, section?: string, variables?: Record<string, string>, target?: ConfigurationTarget): Promise<Map<string, string> | undefined> {
		const parsed = ConfigurationResolverExpression.parse(config);
		// For testing, we simulate interaction by returning a map with some variables
		const result = new Map<string, string>();
		result.set('input:testInteractive', `interactiveValue${this.interactiveCounter++}`);
		result.set('command:testCommand', `commandOutput${this.interactiveCounter++}}`);

		// If variables are provided, include those too
		for (const [k, v] of result.entries()) {
			parsed.resolve({ id: '${' + k + '}' } as any, v);
		}
		return Promise.resolve(result);
	}
}

/**
 * @class TestMcpHostDelegate
 * @implements IMcpHostDelegate
 * @brief A test stub for `IMcpHostDelegate`.
 *
 * This delegate always indicates it can start a server and returns a mock
 * message transport.
 */
class TestMcpHostDelegate implements IMcpHostDelegate {
	/**
	 * @brief Always returns true, indicating it can start any server.
	 * @returns (boolean): True.
	 */
	canStart(): boolean {
		return true;
	}

	/**
	 * @brief Returns a new instance of `TestMcpMessageTransport`.
	 * @returns (IMcpMessageTransport): A mock message transport.
	 */
	start(): IMcpMessageTransport {
		return new TestMcpMessageTransport();
	}

	/**
	 * @brief Always resolves immediately, simulating initial provider readiness.
	 * @returns (Promise<void>): A promise that resolves immediately.
	 */
	waitForInitialProviderPromises(): Promise<void> {
		return Promise.resolve();
	}
}

/**
 * @class TestDialogService
 * @brief A test stub for `IDialogService`.
 *
 * This service allows setting a predefined result for dialog prompts and
 * tracking if `prompt` was called.
 */
class TestDialogService implements Partial<IDialogService> {
	declare readonly _serviceBrand: undefined;

	private _promptResult: boolean | undefined; // The predefined result for the next prompt.
	private _promptSpy: sinon.SinonStub; // Sinon stub to track calls to prompt.

	constructor() {
		this._promptSpy = sinon.stub();
		// Block Logic: Configures the stub to return a promise that resolves with the predefined result.
		this._promptSpy.callsFake(() => {
			return Promise.resolve({ result: this._promptResult });
		});
	}

	/**
	 * @brief Sets the result that `prompt` will return.
	 * @param result (boolean | undefined): The boolean result (true/false/undefined for dismiss).
	 */
	setPromptResult(result: boolean | undefined): void {
		this._promptResult = result;
	}

	/**
	 * @property promptSpy
	 * @brief Getter for the Sinon stub controlling `prompt` calls.
	 */
	get promptSpy(): sinon.SinonStub {
		return this._promptSpy;
	}

	/**
	 * @brief Simulates a dialog prompt.
	 * @param options (any): The dialog options.
	 * @returns (Promise<any>): A promise that resolves with the predefined result.
	 */
	prompt(options: any): Promise<any> {
		return this._promptSpy(options);
	}
}

/**
 * @suite Workbench - MCP - Registry
 * @brief Test suite for the `McpRegistry` class functionality.
 */
suite('Workbench - MCP - Registry', () => {
	const store = ensureNoDisposablesAreLeakedInTestSuite();

	let registry: McpRegistry;
	let testStorageService: TestStorageService;
	let testConfigResolverService: TestConfigurationResolverService;
	let testDialogService: TestDialogService;
	let testCollection: McpCollectionDefinition & { serverDefinitions: ISettableObservable<McpServerDefinition[]> };
	let baseDefinition: McpServerDefinition;

	setup(() => {
		testConfigResolverService = new TestConfigurationResolverService();
		testStorageService = store.add(new TestStorageService());
		testDialogService = new TestDialogService();

		const services = new ServiceCollection(
			[IConfigurationResolverService, testConfigResolverService],
			[IStorageService, testStorageService],
			[ISecretStorageService, new TestSecretStorageService()],
			[ILoggerService, store.add(new TestLoggerService())],
			[IOutputService, upcast({ showChannel: () => { } })],
			[IDialogService, testDialogService],
			[IProductService, upcast({})], // Functional Utility: Use upcast for partial mock of IProductService
		);

		const instaService = store.add(new TestInstantiationService(services));
		registry = store.add(instaService.createInstance(McpRegistry));

		// Create test collection that can be reused across tests.
		testCollection = {
			id: 'test-collection',
			label: 'Test Collection',
			remoteAuthority: null,
			serverDefinitions: observableValue('serverDefs', []), // Settable observable for server definitions.
			isTrustedByDefault: true,
			scope: StorageScope.APPLICATION
		};

		// Create base definition that can be reused across tests.
		baseDefinition = {
			id: 'test-server',
			label: 'Test Server',
			launch: {
				type: McpServerTransportType.Stdio,
				command: 'test-command',
				args: [],
				env: {},
				envFile: undefined,
				cwd: URI.parse('file:///test')
			}
		};
	});

	/**
	 * @test registerCollection adds collection to registry
	 * @brief Tests that `registerCollection` correctly adds and removes collections from the registry.
	 */
	test('registerCollection adds collection to registry', () => {
		const disposable = registry.registerCollection(testCollection);
		store.add(disposable);

		assert.strictEqual(registry.collections.get().length, 1);
		assert.strictEqual(registry.collections.get()[0], testCollection);

		disposable.dispose(); // Functional Utility: Dispose the registration.
		assert.strictEqual(registry.collections.get().length, 0);
	});

	/**
	 * @test registerDelegate adds delegate to registry
	 * @brief Tests that `registerDelegate` correctly adds and removes delegates from the registry.
	 */
	test('registerDelegate adds delegate to registry', () => {
		const delegate = new TestMcpHostDelegate();
		const disposable = registry.registerDelegate(delegate);
		store.add(disposable);

		assert.strictEqual(registry.delegates.length, 1);
		assert.strictEqual(registry.delegates[0], delegate);

		disposable.dispose(); // Functional Utility: Dispose the registration.
		assert.strictEqual(registry.delegates.length, 0);
	});

	/**
	 * @test resolveConnection creates connection with resolved variables and memorizes them until cleared
	 * @brief Tests that `resolveConnection` correctly resolves variables in the launch configuration
	 *        and persists these resolved values across subsequent calls until cleared.
	 */
	test('resolveConnection creates connection with resolved variables and memorizes them until cleared', async () => {
		const definition: McpServerDefinition = {
			...baseDefinition,
			launch: { // Functional Utility: Defines a launch configuration with variables to be resolved.
				type: McpServerTransportType.Stdio,
				command: '${workspaceFolder}/cmd',
				args: ['--file', '${fileBasename}'],
				env: {
					PATH: '${input:testInteractive}' // Functional Utility: Variable that requires interactive resolution.
				},
				envFile: undefined,
				cwd: URI.parse('file:///test')
			},
			variableReplacement: { // Functional Utility: Specifies the variable replacement context.
				section: 'mcp',
				target: ConfigurationTarget.WORKSPACE,
			}
		};

		const delegate = new TestMcpHostDelegate();
		store.add(registry.registerDelegate(delegate));
		testCollection.serverDefinitions.set([definition], undefined);
		store.add(registry.registerCollection(testCollection));

		// Block Logic: First connection resolution - triggers interactive prompt for 'testInteractive'.
		const connection = await registry.resolveConnection({ collectionRef: testCollection, definitionRef: definition }) as McpServerConnection;

		assert.ok(connection);
		assert.strictEqual(connection.definition, definition);
		// Assertion: Check if workspaceFolder variable is resolved.
		assert.strictEqual((connection.launchDefinition as any).command, '/test/workspace/cmd');
		// Assertion: Check if interactive variable is resolved and correctly captured.
		assert.strictEqual((connection.launchDefinition as any).env.PATH, 'interactiveValue0');
		connection.dispose(); // Functional Utility: Dispose the connection.

		// Block Logic: Second connection resolution - should use memorized interactive value.
		const connection2 = await registry.resolveConnection({ collectionRef: testCollection, definitionRef: definition }) as McpServerConnection;

		assert.ok(connection2);
		// Assertion: Interactive variable should be resolved from memory, not re-prompted.
		assert.strictEqual((connection2.launchDefinition as any).env.PATH, 'interactiveValue0');
		connection2.dispose();

		// Block Logic: Clear saved inputs and try again - should trigger a new interactive prompt.
		registry.clearSavedInputs(StorageScope.WORKSPACE);

		const connection3 = await registry.resolveConnection({ collectionRef: testCollection, definitionRef: definition }) as McpServerConnection;

		assert.ok(connection3);
		// Assertion: New interactive value indicates a fresh prompt after clearing inputs.
		// Note: The counter increments for each simulated interaction.
		assert.strictEqual((connection3.launchDefinition as any).env.PATH, 'interactiveValue4');
		connection3.dispose();
	});

	/**
	 * @suite Trust Management
	 * @brief Test suite for the trust management functionality of `McpRegistry`.
	 *
	 * These tests verify how the registry handles trusted and untrusted collections,
	 * including user prompts and persistence of trust decisions.
	 */
	suite('Trust Management', () => {
		setup(() => {
			const delegate = new TestMcpHostDelegate();
			store.add(registry.registerDelegate(delegate));
		});

		/**
		 * @test resolveConnection connects to server when trusted by default
		 * @brief Tests that `resolveConnection` proceeds without a prompt if the collection is trusted by default.
		 */
		test('resolveConnection connects to server when trusted by default', async () => {
			const definition = { ...baseDefinition };
			store.add(registry.registerCollection(testCollection));
			testCollection.serverDefinitions.set([definition], undefined);

			const connection = await registry.resolveConnection({ collectionRef: testCollection, definitionRef: definition });

			assert.ok(connection);
			// Assertion: No prompt should have been displayed.
			assert.strictEqual(testDialogService.promptSpy.called, false);
			connection?.dispose();
		});

		/**
		 * @test resolveConnection prompts for confirmation when not trusted by default
		 * @brief Tests that `resolveConnection` prompts the user for trust when a collection is not trusted by default.
		 *        Also verifies that the trust decision is remembered for subsequent calls.
		 */
		test('resolveConnection prompts for confirmation when not trusted by default', async () => {
			const untrustedCollection: McpCollectionDefinition = {
				...testCollection,
				isTrustedByDefault: false // Functional Utility: Mark collection as not trusted by default.
			};

			const definition = { ...baseDefinition };
			store.add(registry.registerCollection(untrustedCollection));
			testCollection.serverDefinitions.set([definition], undefined);

			testDialogService.setPromptResult(true); // Functional Utility: Simulate user trusting the collection.

			const connection = await registry.resolveConnection({
				collectionRef: untrustedCollection,
				definitionRef: definition
			});

			assert.ok(connection);
			// Assertion: Prompt should have been called once.
			assert.strictEqual(testDialogService.promptSpy.called, true);
			connection?.dispose();

			testDialogService.promptSpy.resetHistory(); // Functional Utility: Reset spy to check for subsequent calls.
			// Block Logic: Second call - should not prompt again if trust decision was remembered.
			const connection2 = await registry.resolveConnection({
				collectionRef: untrustedCollection,
				definitionRef: definition
			});

			assert.ok(connection2);
			// Assertion: Prompt should not be called again.
			assert.strictEqual(testDialogService.promptSpy.called, false);
			connection2?.dispose();
		});

		/**
		 * @test resolveConnection returns undefined when user does not trust the server
		 * @brief Tests that `resolveConnection` returns undefined if the user explicitly
		 *        declines to trust an untrusted-by-default collection.
		 */
		test('resolveConnection returns undefined when user does not trust the server', async () => {
			const untrustedCollection: McpCollectionDefinition = {
				...testCollection,
				isTrustedByDefault: false
			};

			const definition = { ...baseDefinition };
			store.add(registry.registerCollection(untrustedCollection));
			testCollection.serverDefinitions.set([definition], undefined);

			testDialogService.setPromptResult(false); // Functional Utility: Simulate user declining trust.

			const connection = await registry.resolveConnection({
				collectionRef: untrustedCollection,
				definitionRef: definition
			});

			// Assertion: Connection should not be established.
			assert.strictEqual(connection, undefined);
			// Assertion: Prompt should have been called once.
			assert.strictEqual(testDialogService.promptSpy.called, true);

			testDialogService.promptSpy.resetHistory();
			// Block Logic: Second call - should not prompt again if negative trust decision was remembered.
			const connection2 = await registry.resolveConnection({
				collectionRef: untrustedCollection,
				definitionRef: definition
			});

			// Assertion: Connection should still be undefined.
			assert.strictEqual(connection2, undefined);
			// Assertion: Prompt should not be called again.
			assert.strictEqual(testDialogService.promptSpy.called, false);
		});

		/**
		 * @test resolveConnection honors forceTrust parameter
		 * @brief Tests that the `forceTrust` parameter overrides previously stored
		 *        negative trust decisions, forcing a new prompt.
		 */
		test('resolveConnection honors forceTrust parameter', async () => {
			const untrustedCollection: McpCollectionDefinition = {
				...testCollection,
				isTrustedByDefault: false
			};

			const definition = { ...baseDefinition };
			store.add(registry.registerCollection(untrustedCollection));
			testCollection.serverDefinitions.set([definition], undefined);

			testDialogService.setPromptResult(false); // Functional Utility: User initially declines trust.

			const connection1 = await registry.resolveConnection({
				collectionRef: untrustedCollection,
				definitionRef: definition
			});

			assert.strictEqual(connection1, undefined); // Assertion: Connection fails as expected.

			testDialogService.promptSpy.resetHistory();
			testDialogService.setPromptResult(true); // Functional Utility: Simulate user trusting this time.

			// Block Logic: Call with forceTrust: true, should re-prompt even if previously declined.
			const connection2 = await registry.resolveConnection({
				collectionRef: untrustedCollection,
				definitionRef: definition,
				forceTrust: true
			});

			assert.ok(connection2);
			// Assertion: Prompt should be called again due to forceTrust.
			assert.strictEqual(testDialogService.promptSpy.called, true);
			connection2?.dispose();

			testDialogService.promptSpy.resetHistory();
			// Block Logic: Subsequent call should now remember the new positive trust decision.
			const connection3 = await registry.resolveConnection({
				collectionRef: untrustedCollection,
				definitionRef: definition
			});

			assert.ok(connection3);
			// Assertion: No prompt, as new trust decision is remembered.
			assert.strictEqual(testDialogService.promptSpy.called, false);
			connection3?.dispose();
		});
	});

	/**
	 * @suite Lazy Collections
	 * @brief Test suite for handling lazy-loaded MCP collections.
	 *
	 * These tests ensure that lazy collections are correctly registered,
	 * replaced by non-lazy versions, and that their loading state is managed.
	 */
	suite('Lazy Collections', () => {
		let lazyCollection: McpCollectionDefinition;
		let normalCollection: McpCollectionDefinition;
		let removedCalled: boolean;

		setup(() => {
			removedCalled = false;
			// Functional Utility: Defines a lazy collection for testing.
			lazyCollection = {
				...testCollection,
				id: 'lazy-collection',
				lazy: {
					isCached: false,
					load: () => Promise.resolve(), // Mock load function.
					removed: () => { removedCalled = true; } // Spy on removed callback.
				}
			};
			// Functional Utility: Defines a normal (non-lazy) collection for testing replacement.
			normalCollection = {
				...testCollection,
				id: 'lazy-collection', // Same ID as lazyCollection to simulate replacement.
				serverDefinitions: observableValue('serverDefs', [baseDefinition])
			};
		});

		/**
		 * @test registers lazy collection
		 * @brief Tests that a lazy collection is correctly registered and the lazyCollectionState is updated.
		 */
		test('registers lazy collection', () => {
			const disposable = registry.registerCollection(lazyCollection);
			store.add(disposable);

			assert.strictEqual(registry.collections.get().length, 1);
			assert.strictEqual(registry.collections.get()[0], lazyCollection);
			// Assertion: State should be 'HasUnknown' as there's a lazy, uncached collection.
			assert.strictEqual(registry.lazyCollectionState.get(), LazyCollectionState.HasUnknown);
		});

		/**
		 * @test lazy collection is replaced by normal collection
		 * @brief Tests that a lazy collection is correctly replaced by a non-lazy collection with the same ID.
		 */
		test('lazy collection is replaced by normal collection', () => {
			store.add(registry.registerCollection(lazyCollection));
			store.add(registry.registerCollection(normalCollection)); // Functional Utility: Register normal collection with same ID.

			const collections = registry.collections.get();
			assert.strictEqual(collections.length, 1);
			// Assertion: The lazy collection should be replaced by the normal one.
			assert.strictEqual(collections[0], normalCollection);
			assert.strictEqual(collections[0].lazy, undefined); // Assertion: The replaced collection should no longer be lazy.
			// Assertion: State should be 'AllKnown' after replacement.
			assert.strictEqual(registry.lazyCollectionState.get(), LazyCollectionState.AllKnown);
		});

		/**
		 * @test lazyCollectionState updates correctly during loading
		 * @brief Tests that `lazyCollectionState` transitions correctly during the discovery process.
		 */
		test('lazyCollectionState updates correctly during loading', async () => {
			// Block Logic: Overrides the lazy collection's load method to simulate async loading.
			lazyCollection = {
				...lazyCollection,
				lazy: {
					...lazyCollection.lazy!,
					load: async () => {
						await timeout(0); // Functional Utility: Simulate async work.
						store.add(registry.registerCollection(normalCollection)); // Functional Utility: Simulate registration of normal collection during lazy load.
						return Promise.resolve();
					}
				}
			};

			store.add(registry.registerCollection(lazyCollection));
			assert.strictEqual(registry.lazyCollectionState.get(), LazyCollectionState.HasUnknown); // Assertion: Initial state.

			const loadingPromise = registry.discoverCollections();
			// Assertion: State should be 'LoadingUnknown' during discovery.
			assert.strictEqual(registry.lazyCollectionState.get(), LazyCollectionState.LoadingUnknown);

			await loadingPromise; // Functional Utility: Wait for discovery to complete.

			// Assertion: The collection was replaced by `normalCollection` during load, so the state should be AllKnown.
			assert.strictEqual(registry.collections.get().length, 1);
			assert.strictEqual(registry.lazyCollectionState.get(), LazyCollectionState.AllKnown);
			// Assertion: `removed` should not have been called because it was replaced.
			assert.strictEqual(removedCalled, false);
		});

		/**
		 * @test removed callback is called when lazy collection is not replaced
		 * @brief Tests that the `removed` callback is invoked if a lazy collection is discovered
		 *        but not replaced by a non-lazy version.
		 */
		test('removed callback is called when lazy collection is not replaced', async () => {
			store.add(registry.registerCollection(lazyCollection));
			await registry.discoverCollections(); // Functional Utility: Discover the lazy collection.

			// Assertion: `removedCalled` should be true as it was not replaced.
			assert.strictEqual(removedCalled, true);
		});

		/**
		 * @test cached lazy collections are tracked correctly
		 * @brief Tests that `lazyCollectionState` correctly reflects the presence of cached lazy collections.
		 */
		test('cached lazy collections are tracked correctly', () => {
			lazyCollection.lazy!.isCached = true; // Functional Utility: Mark lazy collection as cached.
			store.add(registry.registerCollection(lazyCollection));

			// Assertion: State should be 'AllKnown' because the lazy collection is cached.
			assert.strictEqual(registry.lazyCollectionState.get(), LazyCollectionState.AllKnown);

			// Block Logic: Add another lazy collection that is not cached.
			const uncachedLazy = {
				...lazyCollection,
				id: 'uncached-lazy',
				lazy: {
					...lazyCollection.lazy!,
					isCached: false // Functional Utility: Mark as uncached.
				}
			};
			store.add(registry.registerCollection(uncachedLazy));

			// Assertion: State should revert to 'HasUnknown' due to the new uncached lazy collection.
			assert.strictEqual(registry.lazyCollectionState.get(), LazyCollectionState.HasUnknown);
		});
	});

	/**
	 * @suite Collection Tool Prefixes
	 * @brief Test suite for the generation and management of unique tool prefixes for collections.
	 *
	 * These tests ensure that prefixes are unique, handle collisions, and update correctly
	 * with collection lifecycle.
	 */
	suite('Collection Tool Prefixes', () => {
		/**
		 * @test assigns unique prefixes to collections
		 * @brief Tests that `collectionToolPrefix` assigns unique, 3-character hexadecimal
		 *        prefixes to different collections.
		 */
		test('assigns unique prefixes to collections', () => {
			const collection1: McpCollectionDefinition = {
				id: 'collection1',
				label: 'Collection 1',
				remoteAuthority: null,
				serverDefinitions: observableValue('serverDefs', []),
				isTrustedByDefault: true,
				scope: StorageScope.APPLICATION
			};

			const collection2: McpCollectionDefinition = {
				id: 'collection2',
				label: 'Collection 2',
				remoteAuthority: null,
				serverDefinitions: observableValue('serverDefs', []),
				isTrustedByDefault: true,
				scope: StorageScope.APPLICATION
			};

			store.add(registry.registerCollection(collection1));
			store.add(registry.registerCollection(collection2));

			const prefix1 = registry.collectionToolPrefix(collection1).get();
			const prefix2 = registry.collectionToolPrefix(collection2).get();

			// Assertion: Prefixes should be different for different collections.
			assert.notStrictEqual(prefix1, prefix2);
			// Assertion: Prefixes should be 3-character hexadecimal strings followed by a dot.
			assert.ok(/^[a-f0-9]{3}\.$/.test(prefix1));
			assert.ok(/^[a-f0-9]{3}\.$/.test(prefix2));
		});

		/**
		 * @test handles hash collisions by incrementing view
		 * @brief Tests that `collectionToolPrefix` handles hash collisions by adjusting
		 *        the prefix generation, ensuring uniqueness even with colliding SHA1 hashes.
		 */
		test('handles hash collisions by incrementing view', () => {
			// Functional Utility: Use strings known to have SHA1 hash collisions in their first few characters.
			const collection1: McpCollectionDefinition = {
				id: 'potato',
				label: 'Collection 1',
				remoteAuthority: null,
				serverDefinitions: observableValue('serverDefs', []),
				isTrustedByDefault: true,
				scope: StorageScope.APPLICATION
			};

			const collection2: McpCollectionDefinition = {
				id: 'candidate_83048',
				label: 'Collection 2',
				remoteAuthority: null,
				serverDefinitions: observableValue('serverDefs', []),
				isTrustedByDefault: true,
				scope: StorageScope.APPLICATION
			};

			store.add(registry.registerCollection(collection1));
			store.add(registry.registerCollection(collection2));

			const prefix1 = registry.collectionToolPrefix(collection1).get();
			const prefix2 = registry.collectionToolPrefix(collection2).get();

			// Assertion: Even with colliding initial hashes, prefixes should be unique.
			assert.notStrictEqual(prefix1, prefix2);
			assert.ok(/^[a-f0-9]{3}\.$/.test(prefix1));
			assert.ok(/^[a-f0-9]{3}\.$/.test(prefix2));
		});

		/**
		 * @test prefix changes when collections change
		 * @brief Tests that the tool prefix for a collection becomes empty when the collection is disposed.
		 */
		test('prefix changes when collections change', () => {
			const collection1: McpCollectionDefinition = {
				id: 'collection1',
				label: 'Collection 1',
				remoteAuthority: null,
				serverDefinitions: observableValue('serverDefs', []),
				isTrustedByDefault: true,
				scope: StorageScope.APPLICATION
			};

			const disposable = registry.registerCollection(collection1);
			store.add(disposable);

			const prefix1 = registry.collectionToolPrefix(collection1).get();
			assert.ok(!!prefix1); // Assertion: Prefix should exist initially.

			disposable.dispose(); // Functional Utility: Dispose the collection registration.

			const prefix2 = registry.collectionToolPrefix(collection1).get();

			// Assertion: Prefix should be empty after the collection is disposed.
			assert.strictEqual(prefix2, '');
		});

		/**
		 * @test prefix is empty for unknown collections
		 * @brief Tests that `collectionToolPrefix` returns an empty string for unregistered/unknown collections.
		 */
		test('prefix is empty for unknown collections', () => {
			const unknownCollection: McpCollectionReference = {
				id: 'unknown',
				label: 'Unknown'
			};

			const prefix = registry.collectionToolPrefix(unknownCollection).get();
			assert.strictEqual(prefix, ''); // Assertion: Prefix should be empty for an unknown collection.
		});
	});
});
