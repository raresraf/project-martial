/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
/**
 * @file mcpLanguageFeatures.ts
 * @module vs/workbench/contrib/mcp/browser/mcpLanguageFeatures
 * @description Provides language features (CodeLenses and Inlay Hints) for MCP-related
 *              configuration files (e.g., `mcp.json`). These features enhance the
 *              editing experience by displaying server status, available actions,
 *              and resolved input values directly in the editor.
 */

// Functional Utility: Imports Emitter and Event for event handling.
import { Emitter, Event } from '../../../../base/common/event.js';
// Functional Utility: Imports utilities for creating Markdown command links and Markdown strings for rich content.
import { markdownCommandLink, MarkdownString } from '../../../../base/common/htmlContent.js';
// Functional Utility: Imports JSON parsing utilities to navigate JSON abstract syntax trees.
import { findNodeAtLocation, Node, parseTree } from '../../../../base/common/json.js';
// Functional Utility: Imports classes for managing disposable resources and mutable disposables.
import { Disposable, DisposableStore, IDisposable, MutableDisposable } from '../../../../base/common/lifecycle.js';
// Functional Utility: Imports Observable interface for reactive programming.
import { IObservable } from '../../../../base/common/observable.js';
// Functional Utility: Imports utility for comparing URIs.
import { isEqual } from '../../../../base/common/resources.js';
// Functional Utility: Imports Range class for defining text ranges in the editor.
import { Range } from '../../../../editor/common/core/range.js';
// Functional Utility: Imports interfaces for CodeLens and InlayHint providers.
import { CodeLensList, CodeLensProvider, InlayHint, InlayHintList } from '../../../../editor/common/languages.js';
// Functional Utility: Imports ITextModel interface for interacting with editor text models.
import { ITextModel } from '../../../../editor/common/model.js';
// Functional Utility: Imports service for registering language features.
import { ILanguageFeaturesService } from '../../../../editor/common/services/languageFeatures.js';
// Functional Utility: Imports localization function.
import { localize } from '../../../../nls.js';
// Functional Utility: Imports workbench contribution interface.
import { IWorkbenchContribution } from '../../../common/contributions.js';
// Functional Utility: Imports ConfigurationResolverExpression for parsing and resolving configuration variables.
import { ConfigurationResolverExpression, IResolvedValue } from '../../../services/configurationResolver/common/configurationResolverExpression.js';
// Functional Utility: Imports interface for MCP configuration paths service.
import { IMcpConfigPathsService } from '../common/mcpConfigPathsService.js';
// Functional Utility: Imports interface for MCP registry.
import { IMcpRegistry } from '../common/mcpRegistryTypes.js';
// Functional Utility: Imports interfaces and types for MCP service and connection states.
import { IMcpService, McpConnectionState } from '../common/mcpTypes.js';
// Functional Utility: Imports MCP commands for use in CodeLenses and Inlay Hints.
import { EditStoredInput, RemoveStoredInput, ShowOutput, StartServer, StopServer } from './mcpCommands.js';

/**
 * @class McpLanguageFeatures
 * @extends Disposable
 * @implements IWorkbenchContribution
 * @brief Provides language features like CodeLenses and Inlay Hints for MCP configuration files.
 *
 * This class registers providers for CodeLenses and Inlay Hints, which
 * dynamically annotate MCP-related JSON files. CodeLenses provide contextual
 * actions (e.g., start/stop server) and status indicators. Inlay Hints show
 * resolved values of configuration variables.
 */
export class McpLanguageFeatures extends Disposable implements IWorkbenchContribution {
	// MutableDisposable: Caches the parsed JSON tree and its associated model for efficiency.
	private readonly _cachedMcpSection = this._register(new MutableDisposable<{ model: ITextModel; node: Node } & IDisposable>());

	constructor(
		@ILanguageFeaturesService languageFeaturesService: ILanguageFeaturesService,
		@IMcpRegistry private readonly _mcpRegistry: IMcpRegistry,
		@IMcpConfigPathsService private readonly _mcpConfigPathsService: IMcpConfigPathsService,
		@IMcpService private readonly _mcpService: IMcpService,
	) {
		super();

		// Constant: Defines the file patterns for which these language features should be active.
		const patterns = [{ pattern: '**/.vscode/mcp.json' }, { pattern: '**/settings.json' }];

		// Event Emitter: Used to signal changes in CodeLens data, triggering a re-render.
		const onDidChangeCodeLens = this._register(new Emitter<CodeLensProvider>());
		// CodeLensProvider: Defines the contract for providing CodeLenses.
		const codeLensProvider: CodeLensProvider = {
			onDidChange: onDidChangeCodeLens.event, // Event for when CodeLenses change.
			// Functional Utility: Provides CodeLenses for the given text model and range.
			provideCodeLenses: (model, range) => this._provideCodeLenses(model, () => onDidChangeCodeLens.fire(codeLensProvider)),
		};
		// Block Logic: Registers the CodeLens provider for the specified file patterns.
		this._register(languageFeaturesService.codeLensProvider.register(patterns, codeLensProvider));

		// Block Logic: Registers the Inlay Hints provider for the specified file patterns.
		this._register(languageFeaturesService.inlayHintsProvider.register(patterns, {
			// Functional Utility: Uses the MCP registry's onDidChangeInputs event to signal changes in inlay hints.
			onDidChangeInlayHints: _mcpRegistry.onDidChangeInputs,
			// Functional Utility: Provides Inlay Hints for the given text model and range.
			provideInlayHints: (model, range) => this._provideInlayHints(model, range),
		}
		));
	}

	/**
	 * @brief Parses the ITextModel into a JSON Abstract Syntax Tree (AST) and caches the result.
	 *
	 * This method provides a simple caching mechanism to avoid redundant JSON parsing
	 * when both CodeLenses and Inlay Hints need to access the model's AST.
	 *
	 * @param model (ITextModel): The text model to parse.
	 * @returns (Node): The root node of the parsed JSON AST.
	 */
	private _parseModel(model: ITextModel): Node {
		// Block Logic: Checks if the model is already cached.
		if (this._cachedMcpSection.value?.model === model) {
			return this._cachedMcpSection.value.node; // Inline: Returns the cached AST node.
		}

		// Block Logic: Parses the model's content into a JSON AST.
		const tree = parseTree(model.getValue());
		// Event Listener: Clears the cache if the model's content changes.
		const listener = model.onDidChangeContent(() => this._cachedMcpSection.clear());
		// Block Logic: Caches the new model, AST node, and a dispose function for the listener.
		this._cachedMcpSection.value = { model, node: tree, dispose: () => listener.dispose() };
		return tree;
	}

	/**
	 * @brief Provides CodeLenses for MCP server configurations in the editor.
	 *
	 * CodeLenses display the status of MCP servers (running, starting, error, stopped)
	 * and provide clickable actions (e.g., start, stop, restart, show output).
	 *
	 * @param model (ITextModel): The text model for which to provide CodeLenses.
	 * @param onDidChangeCodeLens (function): Callback to fire when CodeLens data changes.
	 * @returns (Promise<CodeLensList | undefined>): A promise that resolves to the list of CodeLenses, or undefined.
	 */
	private async _provideCodeLenses(model: ITextModel, onDidChangeCodeLens: () => void): Promise<CodeLensList | undefined> {
		// Block Logic: Checks if the current model's URI is part of an MCP configuration path.
		const inConfig = this._mcpConfigPathsService.paths.find(u => isEqual(u.uri, model.uri));
		if (!inConfig) {
			return undefined; // Inline: No MCP config found for this model.
		}

		// Block Logic: Parses the model to get the JSON AST.
		const tree = this._parseModel(model);
		// Block Logic: Finds the 'servers' node in the JSON tree.
		const serversNode = findNodeAtLocation(tree, inConfig.section ? [inConfig.section, 'servers'] : ['servers']);
		if (!serversNode) {
			return undefined; // Inline: No 'servers' section found.
		}

		const store = new DisposableStore();
		const lenses: CodeLensList = { lenses: [], dispose: () => store.dispose() };
		// Functional Utility: Helper function to read observable values and register for changes.
		const read = <T>(observable: IObservable<T>): T => {
			store.add(Event.fromObservableLight(observable)(onDidChangeCodeLens)); // Inline: Registers for observable changes.
			return observable.get(); // Inline: Returns the current value of the observable.
		};

		// Block Logic: Finds the relevant MCP collection for the current model.
		const collection = read(this._mcpRegistry.collections).find(c => isEqual(c.presentation?.origin, model.uri));
		if (!collection) {
			return lenses; // Inline: No collection found.
		}

		// Block Logic: Filters MCP servers belonging to the current collection.
		const mcpServers = read(this._mcpService.servers).filter(s => s.collection.id === collection.id);
		// Block Logic: Iterates through each server definition node in the JSON.
		for (const node of serversNode.children || []) {
			if (node.type !== 'property' || node.children?.[0]?.type !== 'string') {
				continue; // Inline: Skip if not a valid server property node.
			}

			// Inline: Extracts the server name from the JSON node.
			const name = node.children[0].value as string;
			// Block Logic: Finds the corresponding MCP server object.
			const server = mcpServers.find(s => s.definition.label === name);
			if (!server) {
				continue; // Inline: Skip if server object not found.
			}

			// Block Logic: Creates a Range object for the server name in the editor.
			const range = Range.fromPositions(model.getPositionAt(node.children[0].offset));
			// Block Logic: Generates CodeLenses based on the server's connection state.
			switch (read(server.connectionState).state) {
				case McpConnectionState.Kind.Error:
					// Functional Utility: Pushes CodeLenses for error state: Show Output and Restart.
					lenses.lenses.push({
						range,
						command: {
							id: ShowOutput.ID,
							title: '$(error) ' + localize('server.error', 'Error'),
							arguments: [server.definition.id],
						},
					}, {
						range,
						command: {
							id: StartServer.ID,
							title: localize('mcp.restart', "Restart"),
							arguments: [server.definition.id],
						},
					});
					break;
				case McpConnectionState.Kind.Starting:
					// Functional Utility: Pushes CodeLens for starting state: Show Output (with spinning icon).
					lenses.lenses.push({
						range,
						command: {
							id: ShowOutput.ID,
							title: '$(loading~spin) ' + localize('server.starting', 'Starting'),
							arguments: [server.definition.id],
						},
					});
					break;
				case McpConnectionState.Kind.Running:
					// Functional Utility: Pushes CodeLenses for running state: Show Output, Stop, Restart, and Tool Count.
					lenses.lenses.push({
						range,
						command: {
							id: ShowOutput.ID,
							title: '$(check) ' + localize('server.running', 'Running'),
							arguments: [server.definition.id],
						},
					}, {
						range,
						command: {
							id: StopServer.ID,
							title: localize('mcp.stop', "Stop"),
							arguments: [server.definition.id],
						},
					}, {
						range,
						command: {
							id: StartServer.ID,
							title: localize('mcp.restart', "Restart"),
							arguments: [server.definition.id],
						},
					}, {
						range,
						command: {
							id: 'workbench.action.chat.attachTools', // Placeholder command, assumed to be related to MCP tools.
							title: localize('server.toolCount', '{0} tools', read(server.tools).length),
						},
					});
					break;
				case McpConnectionState.Kind.Stopped:
					// Functional Utility: Pushes CodeLens for stopped state: Start Server.
					lenses.lenses.push({
						range,
						command: {
							id: StartServer.ID,
							title: '$(debug-start) ' + localize('mcp.start', "Start"),
							arguments: [server.definition.id],
						},
					});
			}
		}

		return lenses;
	}

	/**
	 * @brief Provides Inlay Hints for MCP configuration files, showing resolved variable values.
	 *
	 * Inlay Hints display the resolved runtime values of configuration variables
	 * (e.g., `${input:myInputId}`) directly in the editor, along with actions
	 * to edit or clear these stored values.
	 *
	 * @param model (ITextModel): The text model for which to provide Inlay Hints.
	 * @param range (Range): The range in the text model to consider for hints.
	 * @returns (Promise<InlayHintList | undefined>): A promise that resolves to the list of Inlay Hints, or undefined.
	 */
	private async _provideInlayHints(model: ITextModel, range: Range): Promise<InlayHintList | undefined> {
		// Block Logic: Checks if the current model's URI is part of an MCP configuration path.
		const inConfig = this._mcpConfigPathsService.paths.find(u => isEqual(u.uri, model.uri));
		if (!inConfig) {
			return undefined; // Inline: No MCP config found for this model.
		}

		// Block Logic: Parses the model to get the JSON AST.
		const tree = this._parseModel(model);
		// Block Logic: Finds the relevant MCP section in the JSON tree.
		const mcpSection = inConfig.section ? findNodeAtLocation(tree, [inConfig.section]) : tree;
		if (!mcpSection) {
			return undefined; // Inline: No MCP section found.
		}

		// Block Logic: Finds the 'inputs' node within the MCP section.
		const inputsNode = findNodeAtLocation(mcpSection, ['inputs']);
		if (!inputsNode) {
			return undefined; // Inline: No 'inputs' section found.
		}

		// Block Logic: Retrieves all saved inputs for the current configuration scope.
		const inputs = await this._mcpRegistry.getSavedInputs(inConfig.scope);
		const hints: InlayHint[] = [];

		// Block Logic: Annotates server definitions with hints if present.
		const serversNode = findNodeAtLocation(mcpSection, ['servers']);
		if (serversNode) {
			annotateServers(serversNode);
		}
		// Block Logic: Annotates input definitions with hints.
		annotateInputs(inputsNode);

		return { hints, dispose: () => { } }; // Inline: Returns the collected hints.

		/**
		 * @brief Recursively annotates server configuration nodes with Inlay Hints.
		 *
		 * This function traverses the JSON tree of server definitions, identifies
		 * configuration variables (e.g., `${input:myInputId}`), and, if resolved,
		 * pushes an Inlay Hint showing its value.
		 *
		 * @param node (Node): The current JSON AST node to process.
		 */
		function annotateServers(node: Node) {
			// Block Logic: Checks if the node is a string value containing a configuration variable.
			if (node.type === 'string' && typeof node.value === 'string' && node.value.includes(ConfigurationResolverExpression.VARIABLE_LHS)) {
				// Block Logic: Parses the string value to extract configuration variables.
				const expr = ConfigurationResolverExpression.parse(node.value);
				// Block Logic: Iterates through unresolved variables to find corresponding saved inputs.
				for (const { id } of expr.unresolved()) {
					const saved = inputs[id];
					if (saved) {
						// Functional Utility: Pushes an Inlay Hint showing the resolved value.
						pushAnnotation(id, node.offset + node.value.indexOf(id) + id.length, '', saved);
					}
				}

			} else if (node.type === 'property') {
				// Block Logic: If it's a property node, recursively calls `annotateServers` on its children (value part).
				node.children?.slice(1).forEach(annotateServers);
			} else {
				// Block Logic: For other node types, recursively calls `annotateServers` on all children.
				node.children?.forEach(annotateServers);
			}
		}

		/**
		 * @brief Annotates input definitions with Inlay Hints.
		 *
		 * This function specifically processes the 'inputs' array in the JSON
		 * configuration, identifying defined inputs and pushing Inlay Hints
		 * to show their resolved values and associated actions.
		 *
		 * @param node (Node): The JSON AST node representing the 'inputs' array.
		 */
		function annotateInputs(node: Node) {
			// Precondition: Node must be an array type with children.
			if (node.type !== 'array' || !node.children) {
				return;
			}

			// Block Logic: Iterates through each input object in the 'inputs' array.
			for (const input of node.children) {
				// Precondition: Each input must be an object type with children.
				if (input.type !== 'object' || !input.children) {
					continue;
				}

				// Block Logic: Finds the 'id' property within the input object.
				const idProp = input.children.find(c => c.type === 'property' && c.children?.[0].value === 'id');
				if (!idProp) {
					continue;
				}

				// Block Logic: Extracts the value of the 'id' property.
				const id = idProp.children![1];
				// Precondition: The ID node must be a string with a value.
				if (!id || id.type !== 'string' || !id.value) {
					continue;
				}

				// Block Logic: Constructs the full variable ID string (e.g., `${input:myInputId}`).
				const savedId = '${input:' + id.value + '}';
				// Block Logic: Retrieves the resolved value for this saved input.
				const saved = inputs[savedId];
				if (saved) {
					// Functional Utility: Pushes an Inlay Hint showing the resolved input value.
					const hint = pushAnnotation(savedId, id.offset + 1 + id.length, localize('input', 'Value'), saved);
					hint.paddingLeft = true; // Inline: Adds left padding for better visual separation.
				}
			}
		}

		/**
		 * @brief Helper function to create and push an Inlay Hint.
		 *
		 * Constructs an Inlay Hint object with a label, position, and rich tooltip
		 * containing actions to edit or clear the stored input.
		 *
		 * @param savedId (string): The full ID of the saved input (e.g., `${input:myInputId}`).
		 * @param offset (number): The character offset in the model where the hint should be displayed.
		 * @param prefix (string): A prefix to display before the resolved value (e.g., "Value").
		 * @param saved (IResolvedValue): The resolved value object for the input.
		 * @returns (InlayHint): The created Inlay Hint object.
		 */
		function pushAnnotation(savedId: string, offset: number, prefix: string, saved: IResolvedValue): InlayHint {
			// Block Logic: Creates a MarkdownString for the tooltip, including clickable command links.
			const tooltip = new MarkdownString([
				markdownCommandLink({ id: EditStoredInput.ID, title: localize('edit', 'Edit'), arguments: [savedId, model.uri, inConfig!.section, inConfig!.target] }),
				markdownCommandLink({ id: RemoveStoredInput.ID, title: localize('clear', 'Clear'), arguments: [inConfig!.scope, savedId] }),
				markdownCommandLink({ id: RemoveStoredInput.ID, title: localize('clearAll', 'Clear All'), arguments: [inConfig!.scope] }),
			].join(' | '), { isTrusted: true });

			// Block Logic: Constructs the Inlay Hint object.
			const hint: InlayHint = {
				// Functional Utility: Formats the hint label, masking passwords.
				label: prefix + ': ' + (saved.input?.type === 'promptString' && saved.input.password ? '*'.repeat(10) : (saved.value || '')),
				position: model.getPositionAt(offset), // Inline: Sets the hint's position.
				tooltip, // Inline: Attaches the rich tooltip.
			};

			hints.push(hint); // Inline: Adds the hint to the list.
			return hint; // Functional Utility: Returns the created hint.
		}
	}
}


