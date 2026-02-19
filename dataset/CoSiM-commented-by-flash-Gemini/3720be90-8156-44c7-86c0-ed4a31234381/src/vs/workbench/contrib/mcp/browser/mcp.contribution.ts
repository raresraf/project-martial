/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
/**
 * @file mcp.contribution.ts
 * @module vs/workbench/contrib/mcp/browser/mcp.contribution
 * @description Entry point for registering and configuring the Multi-Cloud Platform (MCP) feature
 *              within the VS Code workbench. This file handles the dependency injection setup,
 *              discovery mechanisms, workbench contributions, command registrations,
 *              and JSON schema registration related to MCP functionality.
 */

// Functional Utility: Imports a utility to register command actions within the workbench.
import { registerAction2 } from '../../../../platform/actions/common/actions.js';
// Functional Utility: Imports SyncDescriptor for defining synchronous service descriptors.
import { SyncDescriptor } from '../../../../platform/instantiation/common/descriptors.js';
// Functional Utility: Imports utilities for registering singletons in the dependency injection container.
import { InstantiationType, registerSingleton } from '../../../../platform/instantiation/common/extensions.js';
// Functional Utility: Imports the JSON schema registry for contributing schemas.
import * as jsonContributionRegistry from '../../../../platform/jsonschemas/common/jsonContributionRegistry.js';
// Functional Utility: Imports the central platform registry.
import { Registry } from '../../../../platform/registry/common/platform.js';
// Functional Utility: Imports utilities for registering workbench contributions.
import { registerWorkbenchContribution2, WorkbenchPhase } from '../../../common/contributions.js';
// Functional Utility: Imports the schema ID for MCP configuration.
import { mcpSchemaId } from '../../../services/configuration/common/configuration.js';
// Functional Utility: Imports the configuration-based MCP discovery mechanism.
import { ConfigMcpDiscovery } from '../common/discovery/configMcpDiscovery.js';
// Functional Utility: Imports the extension-based MCP discovery mechanism.
import { ExtensionMcpDiscovery } from '../common/discovery/extensionMcpDiscovery.js';
// Functional Utility: Imports the MCP discovery registry.
import { mcpDiscoveryRegistry } from '../common/discovery/mcpDiscovery.js';
// Functional Utility: Imports the remote native MCP discovery mechanism.
import { RemoteNativeMpcDiscovery } from '../common/discovery/nativeMcpRemoteDiscovery.js';
// Functional Utility: Imports the service interface and implementation for MCP configuration paths.
import { IMcpConfigPathsService, McpConfigPathsService } from '../common/mcpConfigPathsService.js';
// Functional Utility: Imports the JSON schema definition for MCP server configuration.
import { mcpServerSchema } from '../common/mcpConfiguration.js';
// Functional Utility: Imports the controller for MCP context keys.
import { McpContextKeysController } from '../common/mcpContextKeys.js';
// Functional Utility: Imports the MCP registry implementation.
import { McpRegistry } from '../common/mcpRegistry.js';
// Functional Utility: Imports the MCP registry interface.
import { IMcpRegistry } from '../common/mcpRegistryTypes.js';
// Functional Utility: Imports the MCP service implementation.
import { McpService } from '../common/mcpService.js';
// Functional Utility: Imports the MCP service interface.
import { IMcpService } from '../common/mcpTypes.js';
// Functional Utility: Imports various MCP-related commands from the local 'mcpCommands' module.
import { AddConfigurationAction, EditStoredInput, InstallFromActivation, ListMcpServerCommand, MCPServerActionRendering, McpServerOptionsCommand, RemoveStoredInput, ResetMcpCachedTools, ResetMcpTrustCommand, ShowOutput, StartServer, StopServer } from './mcpCommands.js';
// Functional Utility: Imports the MCP discovery workbench contribution.
import { McpDiscovery } from './mcpDiscovery.js';
// Functional Utility: Imports the MCP language features workbench contribution.
import { McpLanguageFeatures } from './mcpLanguageFeatures.js';
// Functional Utility: Imports the MCP URL handler workbench contribution.
import { McpUrlHandler } from './mcpUrlHandler.js';

// Block Logic: Registers the IMcpRegistry interface with its McpRegistry implementation as a delayed singleton.
// This ensures that McpRegistry is instantiated only when it's first needed.
registerSingleton(IMcpRegistry, McpRegistry, InstantiationType.Delayed);
// Block Logic: Registers the IMcpService interface with its McpService implementation as a delayed singleton.
registerSingleton(IMcpService, McpService, InstantiationType.Delayed);
// Block Logic: Registers the IMcpConfigPathsService interface with its McpConfigPathsService implementation as a delayed singleton.
registerSingleton(IMcpConfigPathsService, McpConfigPathsService, InstantiationType.Delayed);

// Block Logic: Registers different MCP discovery mechanisms with the mcpDiscoveryRegistry.
// These descriptors tell the system how to create instances of each discovery type.
mcpDiscoveryRegistry.register(new SyncDescriptor(RemoteNativeMpcDiscovery));
mcpDiscoveryRegistry.register(new SyncDescriptor(ConfigMcpDiscovery));
mcpDiscoveryRegistry.register(new SyncDescriptor(ExtensionMcpDiscovery));

// Block Logic: Registers various workbench contributions related to MCP.
// These contributions define lifecycle hooks for different workbench phases.
registerWorkbenchContribution2('mcpDiscovery', McpDiscovery, WorkbenchPhase.AfterRestored);
registerWorkbenchContribution2('mcpContextKeys', McpContextKeysController, WorkbenchPhase.BlockRestore);
registerWorkbenchContribution2('mcpLanguageFeatures', McpLanguageFeatures, WorkbenchPhase.Eventually);
registerWorkbenchContribution2('mcpUrlHandler', McpUrlHandler, WorkbenchPhase.BlockRestore);

// Block Logic: Registers various MCP-related commands and actions.
// These commands will be available in the workbench (e.g., through the command palette).
registerAction2(ListMcpServerCommand);
registerAction2(McpServerOptionsCommand);
registerAction2(ResetMcpTrustCommand);
registerAction2(ResetMcpCachedTools);
registerAction2(AddConfigurationAction);
registerAction2(RemoveStoredInput);
registerAction2(EditStoredInput);
registerAction2(StartServer);
registerAction2(StopServer);
registerAction2(ShowOutput);
registerAction2(InstallFromActivation);

// Block Logic: Registers a workbench contribution specifically for MCP action rendering.
registerWorkbenchContribution2('mcpActionRendering', MCPServerActionRendering, WorkbenchPhase.BlockRestore);

// Block Logic: Retrieves the JSON contribution registry and registers the MCP server schema.
// This allows for validation and IntelliSense for MCP-related JSON configuration files.
const jsonRegistry = <jsonContributionRegistry.IJSONContributionRegistry>Registry.as(jsonContributionRegistry.Extensions.JSONContribution);
jsonRegistry.registerSchema(mcpSchemaId, mcpServerSchema);
