/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
/**
 * @file mcpCommands.ts
 * @module vs/workbench/contrib/mcp/browser/mcpCommands
 * @description Defines various commands and actions related to the Multi-Cloud Platform (MCP)
 *              feature within the VS Code workbench. These commands provide user interaction
 *              for managing MCP servers, configurations, and related tools.
 */

// Functional Utility: Imports utility for creating HTML elements, likely for UI rendering.
import { h } from '../../../../base/browser/dom.js';
// Functional Utility: Imports a utility to ensure all cases in a switch statement are handled.
import { assertNever } from '../../../../base/common/assert.js';
// Functional Utility: Imports utility for grouping items by a common key.
import { groupBy } from '../../../../base/common/collections.js';
// Functional Utility: Imports Codicon definitions for consistent iconography.
import { Codicon } from '../../../../base/common/codicons.js';
// Functional Utility: Imports the Event class for event handling.
import { Event } from '../../../../base/common/event.js';
// Functional Utility: Imports JSON with comments parser.
import { parse as jsoncParse } from '../../../../base/common/jsonc.js';
// Functional Utility: Imports classes for managing disposable resources.
import { Disposable, DisposableStore } from '../../../../base/common/lifecycle.js';
// Functional Utility: Imports observable utilities for reactive programming.
import { autorun, derived } from '../../../../base/common/observable.js';
// Functional Utility: Imports ThemeIcon for themable iconography.
import { ThemeIcon } from '../../../../base/common/themables.js';
// Functional Utility: Imports URI for handling uniform resource identifiers.
import { URI } from '../../../../base/common/uri.js';
// Functional Utility: Imports IModelService for interacting with text models.
import { IModelService } from '../../../../editor/common/services/model.js';
// Functional Utility: Imports localization functions.
import { ILocalizedString, localize, localize2 } from '../../../../nls.js';
// Functional Utility: Imports service for action view items.
import { IActionViewItemService } from '../../../../platform/actions/browser/actionViewItemService.js';
// Functional Utility: Imports action view item for menu entries.
import { MenuEntryActionViewItem } from '../../../../platform/actions/browser/menuEntryActionViewItem.js';
// Functional Utility: Imports core action definitions and menu IDs.
import { Action2, MenuId, MenuItemAction } from '../../../../platform/actions/common/actions.js';
// Functional Utility: Imports command service for executing commands.
import { ICommandService } from '../../../../platform/commands/common/commands.js';
// Functional Utility: Imports ConfigurationTarget for specifying where configuration changes apply.
import { ConfigurationTarget } from '../../../../platform/configuration/common/configuration.js';
// Functional Utility: Imports ContextKeyExpr for defining context key expressions.
import { ContextKeyExpr } from '../../../../platform/contextkey/common/contextkey.js';
// Functional Utility: Imports instantiation service for creating objects with dependency injection.
import { IInstantiationService, ServicesAccessor } from '../../../../platform/instantiation/common/instantiation.js';
// Functional Utility: Imports quick input service for user selection.
import { IQuickInputService, IQuickPickItem } from '../../../../platform/quickinput/common/quickInput.js';
// Functional Utility: Imports StorageScope for defining storage scopes.
import { StorageScope } from '../../../../platform/storage/common/storage.js';
// Functional Utility: Imports spinning loading icon.
import { spinningLoading } from '../../../../platform/theme/common/iconRegistry.js';
// Functional Utility: Imports workspace context service.
import { IWorkspaceContextService } from '../../../../platform/workspace/common/workspace.js';
// Functional Utility: Imports active editor context key.
import { ActiveEditorContext, ResourceContextKey } from '../../../common/contextkeys.js';
// Functional Utility: Imports workbench contribution interface.
import { IWorkbenchContribution } from '../../../common/contributions.js';
// Functional Utility: Imports editor service for opening editors.
import { IEditorService } from '../../../services/editor/common/editorService.js';
// Functional Utility: Imports ChatContextKeys for chat-related context.
import { ChatContextKeys } from '../../chat/common/chatContextKeys.js';
// Functional Utility: Imports ChatMode enum.
import { ChatMode } from '../../chat/common/constants.js';
// Functional Utility: Imports TEXT_FILE_EDITOR_ID.
import { TEXT_FILE_EDITOR_ID } from '../../files/common/files.js';
// Functional Utility: Imports MCP context keys.
import { McpContextKeys } from '../common/mcpContextKeys.js';
// Functional Utility: Imports MCP registry types.
import { IMcpRegistry } from '../common/mcpRegistryTypes.js';
// Functional Utility: Imports MCP service types and interfaces.
import { IMcpServer, IMcpService, LazyCollectionState, McpConnectionState, McpServerToolsState } from '../common/mcpTypes.js';
// Functional Utility: Imports the MCP Add Configuration Command.
import { McpAddConfigurationCommand } from './mcpCommandsAddConfiguration.js';
// Functional Utility: Imports the MCP URL Handler.
import { McpUrlHandler } from './mcpUrlHandler.js';

// Constant: Defines the category string for all MCP commands.
// It's intentionally not localized as per comment.
const category: ILocalizedString = {
	original: 'MCP',
	value: 'MCP',
};

/**
 * @class ListMcpServerCommand
 * @extends Action2
 * @brief Command to list available MCP servers and allow the user to select one for further actions.
 *
 * This command displays a quick pick list of all registered MCP servers,
 * showing their connection state. Upon selection, it triggers the
 * `McpServerOptionsCommand` for the chosen server.
 */
export class ListMcpServerCommand extends Action2 {
	public static readonly id = 'workbench.mcp.listServer';
	constructor() {
		super({
			id: ListMcpServerCommand.id,
			title: localize2('mcp.list', 'List Servers'),
			icon: Codicon.server,
			category,
			f1: true,
			menu: {
				when: ContextKeyExpr.and(
					ContextKeyExpr.or(McpContextKeys.hasUnknownTools, McpContextKeys.hasServersWithErrors),
					ChatContextKeys.chatMode.isEqualTo(ChatMode.Agent)
				),
				id: MenuId.ChatInputAttachmentToolbar,
				group: 'navigation',
				order: 0
			},
		});
	}

	/**
	 * @brief Executes the `ListMcpServerCommand`.
	 *
	 * Displays a quick pick UI to the user, listing all available MCP servers
	 * with their current connection status. When a server is selected,
	 * `McpServerOptionsCommand` is invoked for that server.
	 *
	 * @param accessor (ServicesAccessor): Provides access to various services.
	 */
	override async run(accessor: ServicesAccessor) {
		const mcpService = accessor.get(IMcpService);
		const commandService = accessor.get(ICommandService);
		const quickInput = accessor.get(IQuickInputService);

		type ItemType = { id: string } & IQuickPickItem;

		const store = new DisposableStore();
		const pick = quickInput.createQuickPick<ItemType>({ useSeparators: true });
		pick.placeholder = localize('mcp.selectServer', 'Select an MCP Server');

		store.add(pick);
		// Block Logic: Automatically updates the quick pick items based on changes in MCP server data.
		store.add(autorun(reader => {
			const servers = groupBy(mcpService.servers.read(reader).slice().sort((a, b) => (a.collection.presentation?.order || 0) - (b.collection.presentation?.order || 0)), s => s.collection.id);
			// Block Logic: Formats the grouped servers into quick pick items, adding separators for collections.
			pick.items = Object.values(servers).flatMap(servers => {
				return [
					{ type: 'separator', label: servers[0].collection.label, id: servers[0].collection.id },
					...servers.map(server => ({
						id: server.definition.id,
						label: server.definition.label,
						description: McpConnectionState.toString(server.connectionState.read(reader)),
					})),
				];
			});
		}));

		// Block Logic: Waits for the user to pick an item or dismiss the quick pick.
		const picked = await new Promise<ItemType | undefined>(resolve => {
			store.add(pick.onDidAccept(() => {
				resolve(pick.activeItems[0]);
			}));
			store.add(pick.onDidHide(() => {
				resolve(undefined);
			}));
			pick.show();
		});

		store.dispose(); // Inline: Disposes of the disposable store to clean up resources.

		// Block Logic: If an item was picked, execute the server options command for the selected server.
		if (picked) {
			commandService.executeCommand(McpServerOptionsCommand.id, picked.id);
		}
	}
}

/**
 * @class McpServerOptionsCommand
 * @extends Action2
 * @brief Command to display options and actions for a specific MCP server.
 *
 * This command presents a quick pick list of actions (start, stop, restart,
 * show output, show configuration) for a given MCP server, allowing the user
 * to manage its state and view details.
 */
export class McpServerOptionsCommand extends Action2 {

	static readonly id = 'workbench.mcp.serverOptions';

	constructor() {
		super({
			id: McpServerOptionsCommand.id,
			title: localize2('mcp.options', 'Server Options'),
			category,
			f1: false, // Functional Utility: Not directly discoverable via F1, typically triggered by other commands.
		});
	}

	/**
	 * @brief Executes the `McpServerOptionsCommand`.
	 *
	 * Displays a quick pick UI with available actions for the specified MCP server.
	 * Actions include starting, stopping, restarting the server, showing its output,
	 * or revealing its configuration.
	 *
	 * @param accessor (ServicesAccessor): Provides access to various services.
	 * @param id (string): The ID of the MCP server to show options for.
	 * @returns (Promise<void>): A promise that resolves when the action is complete.
	 */
	override async run(accessor: ServicesAccessor, id: string): Promise<void> {
		const mcpService = accessor.get(IMcpService);
		const quickInputService = accessor.get(IQuickInputService);
		const mcpRegistry = accessor.get(IMcpRegistry);
		const editorService = accessor.get(IEditorService);

		// Block Logic: Finds the specified MCP server.
		const server = mcpService.servers.get().find(s => s.definition.id === id);
		if (!server) {
			return; // Inline: Exits if the server is not found.
		}

		// Block Logic: Retrieves the server's collection and definition for configuration details.
		const collection = mcpRegistry.collections.get().find(c => c.id === server.collection.id);
		const serverDefinition = collection?.serverDefinitions.get().find(s => s.id === server.definition.id);

		interface ActionItem extends IQuickPickItem {
			action: 'start' | 'stop' | 'restart' | 'showOutput' | 'config';
		}

		const items: ActionItem[] = [];
		const serverState = server.connectionState.get();

		// Block Logic: Conditionally adds 'start', 'stop', or 'restart' actions based on current server state.
		// Precondition: Only show 'start' if the server can be started (stopped or in error).
		if (McpConnectionState.canBeStarted(serverState.state)) {
			items.push({
				label: localize('mcp.start', 'Start Server'),
				action: 'start'
			});
		} else { // Functional Utility: If server is running, offer stop and restart.
			items.push({
				label: localize('mcp.stop', 'Stop Server'),
				action: 'stop'
			});
			items.push({
				label: localize('mcp.restart', 'Restart Server'),
				action: 'restart'
			});
		}

		// Block Logic: Adds 'Show Output' action to the list.
		items.push({
			label: localize('mcp.showOutput', 'Show Output'),
			action: 'showOutput'
		});

		// Block Logic: Adds 'Show Configuration' action if a configuration target is available.
		const configTarget = serverDefinition?.presentation?.origin || collection?.presentation?.origin;
		if (configTarget) {
			items.push({
				label: localize('mcp.config', 'Show Configuration'),
				action: 'config',
			});
		}

		// Block Logic: Displays a quick pick to the user with the available actions.
		const pick = await quickInputService.pick(items, {
			title: server.definition.label,
			placeHolder: localize('mcp.selectAction', 'Select Server Action')
		});

		if (!pick) {
			return; // Inline: Exits if no action was picked.
		}

		// Block Logic: Executes the selected action.
		switch (pick.action) {
			case 'start':
				await server.start(true); // Inline: Starts the server and forces showing output.
				server.showOutput();
				break;
			case 'stop':
				await server.stop(); // Inline: Stops the server.
				break;
			case 'restart':
				await server.stop(); // Inline: Stops, then starts the server.
				await server.start(true);
				break;
			case 'showOutput':
				server.showOutput(); // Inline: Shows the server's output channel.
				break;
			case 'config':
				// Block Logic: Opens the server's configuration file in an editor.
				editorService.openEditor({
					resource: URI.isUri(configTarget) ? configTarget : configTarget!.uri,
					options: { selection: URI.isUri(configTarget) ? undefined : configTarget!.range }
				});
				break;
			default:
				assertNever(pick.action); // Inline: Ensures all possible actions are handled.
		}
	}
}

/**
 * @class MCPServerActionRendering
 * @extends Disposable
 * @implements IWorkbenchContribution
 * @brief Workbench contribution responsible for rendering MCP server actions in the chat input attachment toolbar.
 *
 * This class dynamically updates the appearance and behavior of an MCP-related
 * action button in the chat input attachment toolbar based on the state of
 * MCP servers (e.g., new tools available, errors, refreshing).
 */
export class MCPServerActionRendering extends Disposable implements IWorkbenchContribution {
	public static readonly ID = 'workbench.contrib.mcp.discovery';

	constructor(
		@IActionViewItemService actionViewItemService: IActionViewItemService,
		@IMcpService mcpService: IMcpService,
		@IInstantiationService instaService: IInstantiationService,
		@ICommandService commandService: ICommandService,
	) {
		super();

		// Enum: Defines the different states for displaying the MCP action button.
		const enum DisplayedState {
			None,         // No special state to display.
			NewTools,     // New tools are available.
			Error,        // An error occurred.
			Refreshing,   // Tools are being discovered/refreshed.
		}

		// Block Logic: Derived observable that calculates the overall display state based on all MCP servers.
		const displayedState = derived((reader) => {
			const servers = mcpService.servers.read(reader);
			const serversPerState: IMcpServer[][] = []; // Group servers by their determined DisplayedState.
			for (const server of servers) {
				let thisState = DisplayedState.None;
				switch (server.toolsState.read(reader)) {
					case McpServerToolsState.Unknown:
						// Block Logic: If tools state is unknown, check trusted status and connection error.
						if (server.trusted.read(reader) === false) {
							thisState = DisplayedState.None;
						} else {
							thisState = server.connectionState.read(reader).state === McpConnectionState.Kind.Error ? DisplayedState.Error : DisplayedState.NewTools;
						}
						break;
					case McpServerToolsState.RefreshingFromUnknown:
						thisState = DisplayedState.Refreshing;
						break;
					default:
						// Block Logic: If tools state is known, only show error if connection has an error.
						thisState = server.connectionState.read(reader).state === McpConnectionState.Kind.Error ? DisplayedState.Error : DisplayedState.None;
						break;
				}

				serversPerState[thisState] ??= []; // Initialize array if null
				serversPerState[thisState].push(server);
			}

			// Block Logic: Checks lazy collection state for overall unknown/loading states.
			const unknownServerStates = mcpService.lazyCollectionState.read(reader);
			if (unknownServerStates === LazyCollectionState.LoadingUnknown) {
				serversPerState[DisplayedState.Refreshing] ??= [];
			} else if (unknownServerStates === LazyCollectionState.HasUnknown) {
				serversPerState[DisplayedState.NewTools] ??= [];
			}

			// Inline: Determines the highest priority state to display.
			const maxState = (serversPerState.length - 1) as DisplayedState;
			return { state: maxState, servers: serversPerState[maxState] || [] };
		});

		// Block Logic: Registers a custom action view item for the MCP server command in the chat input toolbar.
		this._store.add(actionViewItemService.register(MenuId.ChatInputAttachmentToolbar, ListMcpServerCommand.id, (action, options) => {
			if (!(action instanceof MenuItemAction)) {
				return undefined; // Inline: Only handle MenuItemAction.
			}

			// Block Logic: Returns an instance of a custom MenuEntryActionViewItem for rendering.
			return instaService.createInstance(class extends MenuEntryActionViewItem {

				/**
				 * @brief Renders the MCP action button in the container.
				 * @param container (HTMLElement): The HTML element to render the action into.
				 */
				override render(container: HTMLElement): void {
					super.render(container);
					container.classList.add('chat-mcp'); // Inline: Adds a CSS class for styling.

					// Functional Utility: Creates the button element for the MCP action.
					const action = h('button.chat-mcp-action', [h('span@icon')]);

					// Block Logic: Uses autorun to reactively update the UI based on `displayedState`.
					this._register(autorun(r => {
						const { state } = displayedState.read(r);
						const { root, icon } = action;
						this.updateTooltip(); // Inline: Updates the tooltip text.
						container.classList.toggle('chat-mcp-has-action', state !== DisplayedState.None);

						if (!root.parentElement) {
							container.appendChild(root); // Inline: Appends the button if not already in container.
						}

						root.ariaLabel = this.getLabelForState(displayedState.read(r)); // Inline: Sets ARIA label for accessibility.
						root.className = 'chat-mcp-action'; // Inline: Resets and applies base class.
						icon.className = ''; // Inline: Clears existing icon classes.

						// Block Logic: Applies specific styling and icons based on the `displayedState`.
						if (state === DisplayedState.NewTools) {
							root.classList.add('chat-mcp-action-new');
							icon.classList.add(...ThemeIcon.asClassNameArray(Codicon.refresh)); // Inline: Adds refresh icon.
						} else if (state === DisplayedState.Error) {
							root.classList.add('chat-mcp-action-error');
							icon.classList.add(...ThemeIcon.asClassNameArray(Codicon.warning)); // Inline: Adds warning icon.
						} else if (state === DisplayedState.Refreshing) {
							root.classList.add('chat-mcp-action-refreshing');
							icon.classList.add(...ThemeIcon.asClassNameArray(spinningLoading)); // Inline: Adds spinning loading icon.
						} else {
							root.remove(); // Inline: Removes the button if state is None.
						}
					}));
				}

				/**
				 * @brief Handles the click event on the MCP action button.
				 * @param e (MouseEvent): The mouse event.
				 * @returns (Promise<void>): A promise that resolves when the click action is handled.
				 */
				override async onClick(e: MouseEvent): Promise<void> {
					e.preventDefault(); // Inline: Prevents default browser action.
					e.stopPropagation(); // Inline: Stops event propagation.

					const { state, servers } = displayedState.get();
					// Block Logic: Executes actions based on the current `displayedState`.
					if (state === DisplayedState.NewTools) {
						servers.forEach(server => server.start()); // Inline: Starts all servers with new tools.
						mcpService.activateCollections(); // Inline: Activates MCP collections.
					} else if (state === DisplayedState.Refreshing) {
						servers.at(-1)?.showOutput(); // Inline: Shows output for the last refreshing server.
					} else if (state === DisplayedState.Error) {
						const server = servers.at(-1);
						if (server) {
							commandService.executeCommand(McpServerOptionsCommand.id, server.definition.id); // Inline: Shows options for the last server with an error.
						}
					} else {
						commandService.executeCommand(ListMcpServerCommand.id); // Inline: Shows the list of MCP servers.
					}
				}

				/**
				 * @brief Overrides the default tooltip behavior to provide a dynamic tooltip based on state.
				 * @returns (string): The tooltip string.
				 */
				protected override getTooltip(): string {
					return this.getLabelForState() || super.getTooltip();
				}

				/**
				 * @brief Generates a localized label for the current display state.
				 * @param displayedStateObj (object): Optional, the current displayed state object.
				 * @returns (string | null): The localized label string, or null if no label.
				 */
				private getLabelForState({ state, servers } = displayedState.get()) {
					if (state === DisplayedState.NewTools) {
						return localize('mcp.newTools', "New tools available ({0})", servers.length || 1);
					} else if (state === DisplayedState.Error) {
						return localize('mcp.toolError', "Error loading {0} tool(s)", servers.length || 1);
					} else if (state === DisplayedState.Refreshing) {
						return localize('mcp.toolRefresh', "Discovering tools...");
					} else {
						return null;
					}
				}
			}, action, { ...options, keybindingNotRenderedWithLabel: true });

		}, Event.fromObservable(displayedState)));
	}
}

/**
 * @class ResetMcpTrustCommand
 * @extends Action2
 * @brief Command to reset the trust settings for MCP servers.
 *
 * This command allows users to clear any previously established trust
 * relationships with MCP servers.
 */
export class ResetMcpTrustCommand extends Action2 {
	static readonly ID = 'workbench.mcp.resetTrust';

	constructor() {
		super({
			id: ResetMcpTrustCommand.ID,
			title: localize2('mcp.resetTrust', "Reset Trust"),
			category,
			f1: true,
			precondition: McpContextKeys.toolsCount.greater(0), // Precondition: Only enable if there are tools to manage trust for.
		});
	}

	/**
	 * @brief Executes the `ResetMcpTrustCommand`.
	 *
	 * Invokes the `resetTrust` method on the `IMcpRegistry` service to clear
	 * all saved trust settings.
	 *
	 * @param accessor (ServicesAccessor): Provides access to various services.
	 */
	run(accessor: ServicesAccessor): void {
		const mcpService = accessor.get(IMcpRegistry);
		mcpService.resetTrust(); // Functional Utility: Calls the registry method to reset trust.
	}
}

/**
 * @class ResetMcpCachedTools
 * @extends Action2
 * @brief Command to reset the cached MCP tools.
 *
 * This command allows users to clear any cached information about MCP tools,
 * forcing a fresh discovery or re-evaluation of available tools.
 */
export class ResetMcpCachedTools extends Action2 {
	static readonly ID = 'workbench.mcp.resetCachedTools';

	constructor() {
		super({
			id: ResetMcpCachedTools.ID,
			title: localize2('mcp.resetCachedTools', "Reset Cached Tools"),
			category,
			f1: true,
			precondition: McpContextKeys.toolsCount.greater(0), // Precondition: Only enable if there are tools to manage caches for.
		});
	}

	/**
	 * @brief Executes the `ResetMcpCachedTools` command.
	 *
	 * Invokes the `resetCaches` method on the `IMcpService` to clear
	 * cached tool information.
	 *
	 * @param accessor (ServicesAccessor): Provides access to various services.
	 */
	run(accessor: ServicesAccessor): void {
		const mcpService = accessor.get(IMcpService);
		mcpService.resetCaches(); // Functional Utility: Calls the service method to reset caches.
	}
}

/**
 * @class AddConfigurationAction
 * @extends Action2
 * @brief Action to add a new MCP server configuration.
 *
 * This action initiates the process of adding a new MCP server configuration,
 * typically by modifying the `mcp.json` settings file.
 */
export class AddConfigurationAction extends Action2 {
	static readonly ID = 'workbench.mcp.addConfiguration';

	constructor() {
		super({
			id: AddConfigurationAction.ID,
			title: localize2('mcp.addConfiguration', "Add Server..."),
			metadata: {
				description: localize2('mcp.addConfiguration.description', "Installs a new Model Context protocol to the mcp.json settings"),
			},
			category,
			f1: true,
			menu: {
				id: MenuId.EditorContent,
				when: ContextKeyExpr.and(
					// Precondition: Only available in the context of editing a `.vscode/mcp.json` file.
					ContextKeyExpr.regex(ResourceContextKey.Path.key, /\.vscode[/\\]mcp\.json$/),
					ActiveEditorContext.isEqualTo(TEXT_FILE_EDITOR_ID)
				)
			}
		});
	}

	/**
	 * @brief Executes the `AddConfigurationAction`.
	 *
	 * Creates an instance of `McpAddConfigurationCommand` and runs it to
	 * guide the user through adding a new MCP server configuration.
	 *
	 * @param accessor (ServicesAccessor): Provides access to various services.
	 * @param configUri (string): Optional URI of the configuration file.
	 * @returns (Promise<void>): A promise that resolves when the command is complete.
	 */
	async run(accessor: ServicesAccessor, configUri?: string): Promise<void> {
		return accessor.get(IInstantiationService).createInstance(McpAddConfigurationCommand, configUri).run();
	}
}

/**
 * @class RemoveStoredInput
 * @extends Action2
 * @brief Command to remove stored MCP inputs.
 *
 * This command allows users to clear specific saved inputs related to MCP,
 * identified by their storage scope and optional ID.
 */
export class RemoveStoredInput extends Action2 {
	static readonly ID = 'workbench.mcp.removeStoredInput';

	constructor() {
		super({
			id: RemoveStoredInput.ID,
			title: localize2('mcp.removeStoredInput', "Remove Stored Input"), // Corrected title
			category,
			f1: false,
		});
	}

	/**
	 * @brief Executes the `RemoveStoredInput` command.
	 *
	 * Invokes the `clearSavedInputs` method on the `IMcpRegistry` service
	 * to remove stored inputs.
	 *
	 * @param accessor (ServicesAccessor): Provides access to various services.
	 * @param scope (StorageScope): The storage scope of the input to remove.
	 * @param id (string): Optional ID of the specific input to remove.
	 */
	run(accessor: ServicesAccessor, scope: StorageScope, id?: string): void {
		accessor.get(IMcpRegistry).clearSavedInputs(scope, id);
	}
}

/**
 * @class EditStoredInput
 * @extends Action2
 * @brief Command to edit stored MCP inputs.
 *
 * This command allows users to modify previously saved inputs related to MCP.
 */
export class EditStoredInput extends Action2 {
	static readonly ID = 'workbench.mcp.editStoredInput';

	constructor() {
		super({
			id: EditStoredInput.ID,
			title: localize2('mcp.editStoredInput', "Edit Stored Input"),
			category,
			f1: false,
		});
	}

	/**
	 * @brief Executes the `EditStoredInput` command.
	 *
	 * Invokes the `editSavedInput` method on the `IMcpRegistry` service
	 * to allow editing of a specific stored input.
	 *
	 * @param accessor (ServicesAccessor): Provides access to various services.
	 * @param inputId (string): The ID of the input to edit.
	 * @param uri (URI): Optional URI of the resource associated with the input.
	 * @param configSection (string): The configuration section.
	 * @param target (ConfigurationTarget): The configuration target.
	 */
	run(accessor: ServicesAccessor, inputId: string, uri: URI | undefined, configSection: string, target: ConfigurationTarget): void {
		// Block Logic: Determines the workspace folder from the provided URI.
		const workspaceFolder = uri && accessor.get(IWorkspaceContextService).getWorkspaceFolder(uri);
		accessor.get(IMcpRegistry).editSavedInput(inputId, workspaceFolder || undefined, configSection, target);
	}
}

/**
 * @class ShowOutput
 * @extends Action2
 * @brief Command to show the output channel for a specific MCP server.
 *
 * This command allows users to view the runtime output logs of a designated
 * MCP server.
 */
export class ShowOutput extends Action2 {
	static readonly ID = 'workbench.mcp.showOutput';

	constructor() {
		super({
			id: ShowOutput.ID,
			title: localize2('mcp.command.showOutput', "Show Output"),
			category,
			f1: false,
		});
	}

	/**
	 * @brief Executes the `ShowOutput` command.
	 *
	 * Finds the specified MCP server and calls its `showOutput` method
	 * to display its output channel.
	 *
	 * @param accessor (ServicesAccessor): Provides access to various services.
	 * @param serverId (string): The ID of the MCP server whose output to show.
	 */
	run(accessor: ServicesAccessor, serverId: string): void {
		accessor.get(IMcpService).servers.get().find(s => s.definition.id === serverId)?.showOutput();
	}
}

/**
 * @class StartServer
 * @extends Action2
 * @brief Command to start a specific MCP server.
 *
 * This command allows users to initiate the startup process of a designated
 * MCP server.
 */
export class StartServer extends Action2 {
	static readonly ID = 'workbench.mcp.startServer';

	constructor() {
		super({
			id: StartServer.ID,
			title: localize2('mcp.command.startServer', "Start Server"),
			category,
			f1: false,
		});
	}

	/**
	 * @brief Executes the `StartServer` command.
	 *
	 * Finds the specified MCP server, stops it if running, and then starts it.
	 *
	 * @param accessor (ServicesAccessor): Provides access to various services.
	 * @param serverId (string): The ID of the MCP server to start.
	 * @returns (Promise<void>): A promise that resolves when the server start operation is complete.
	 */
	async run(accessor: ServicesAccessor, serverId: string) {
		const s = accessor.get(IMcpService).servers.get().find(s => s.definition.id === serverId);
		await s?.stop(); // Inline: Stops the server first to ensure a clean start.
		await s?.start(); // Inline: Starts the server.
	}
}

/**
 * @class StopServer
 * @extends Action2
 * @brief Command to stop a specific MCP server.
 *
 * This command allows users to terminate the execution of a designated
 * MCP server.
 */
export class StopServer extends Action2 {
	static readonly ID = 'workbench.mcp.stopServer';

	constructor() {
		super({
			id: StopServer.ID,
			title: localize2('mcp.command.stopServer', "Stop Server"),
			category,
			f1: false,
		});
	}

	/**
	 * @brief Executes the `StopServer` command.
	 *
	 * Finds the specified MCP server and calls its `stop` method
	 * to terminate its execution.
	 *
	 * @param accessor (ServicesAccessor): Provides access to various services.
	 * @param serverId (string): The ID of the MCP server to stop.
	 * @returns (Promise<void>): A promise that resolves when the server stop operation is complete.
	 */
	async run(accessor: ServicesAccessor, serverId: string) {
		const s = accessor.get(IMcpService).servers.get().find(s => s.definition.id === serverId);
		await s?.stop();
	}
}

/**
 * @class InstallFromActivation
 * @extends Action2
 * @brief Command to install an MCP configuration from an activation URI.
 *
 * This command is triggered when an MCP activation URI is handled, allowing
 * the user to install the associated MCP configuration.
 */
export class InstallFromActivation extends Action2 {
	static readonly ID = 'workbench.mcp.installFromActivation';

	constructor() {
		super({
			id: InstallFromActivation.ID,
			title: localize2('mcp.command.installFromActivation', "Install..."),
			category,
			f1: false,
			menu: {
				id: MenuId.EditorContent,
				// Precondition: Only available when the resource scheme matches the MCP URL handler scheme.
				when: ContextKeyExpr.equals('resourceScheme', McpUrlHandler.scheme)
			}
		});
	}

	/**
	 * @brief Executes the `InstallFromActivation` command.
	 *
	 * Processes an MCP activation URI to install the associated configuration.
	 * It uses `McpAddConfigurationCommand` to handle the installation logic.
	 *
	 * @param accessor (ServicesAccessor): Provides access to various services.
	 * @param uri (URI): The activation URI containing the MCP configuration.
	 * @returns (Promise<void>): A promise that resolves when the installation is complete.
	 */
	async run(accessor: ServicesAccessor, uri: URI) {
		const editorService = accessor.get(IModelService);
		// Block Logic: Creates an instance of McpAddConfigurationCommand.
		const addConfigHelper = accessor.get(IInstantiationService).createInstance(McpAddConfigurationCommand, undefined);

		// Functional Utility: Uses the helper to pick and process the configuration from the URL handler.
		// jsoncParse is used to parse the content of the model associated with the URI.
		addConfigHelper.pickForUrlHandler(uri, jsoncParse(editorService.getModel(uri)!.getValue()));
	}
}
