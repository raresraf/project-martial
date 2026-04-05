/**
 * @file This file contains the logic for displaying the startup page in Visual Studio Code,
 * commonly known as the 'Welcome' or 'Getting Started' page. It determines whether to show
 * this page, a README.md file, or a terminal on startup based on user configuration,
 * workspace state, and other factors.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { URI } from '../../../../base/common/uri.js';
import { ICommandService } from '../../../../platform/commands/common/commands.js';
import * as arrays from '../../../../base/common/arrays.js';
import { IWorkbenchContribution } from '../../../common/contributions.js';
import { IInstantiationService } from '../../../../platform/instantiation/common/instantiation.js';
import { IEditorService } from '../../../services/editor/common/editorService.js';
import { onUnexpectedError } from '../../../../base/common/errors.js';
import { IWorkspaceContextService, UNKNOWN_EMPTY_WINDOW_WORKSPACE, WorkbenchState } from '../../../../platform/workspace/common/workspace.js';
import { IConfigurationService } from '../../../../platform/configuration/common/configuration.js';
import { IWorkingCopyBackupService } from '../../../services/workingCopy/common/workingCopyBackup.js';
import { ILifecycleService, LifecyclePhase, StartupKind } from '../../../services/lifecycle/common/lifecycle.js';
import { Disposable } from '../../../../base/common/lifecycle.js';
import { IFileService } from '../../../../platform/files/common/files.js';
import { joinPath } from '../../../../base/common/resources.js';
import { IWorkbenchLayoutService } from '../../../services/layout/browser/layoutService.js';
import { GettingStartedEditorOptions, GettingStartedInput, gettingStartedInputTypeId } from './gettingStartedInput.js';
import { IWorkbenchEnvironmentService } from '../../../services/environment/common/environmentService.js';
import { IStorageService, StorageScope, StorageTarget } from '../../../../platform/storage/common/storage.js';
import { getTelemetryLevel } from '../../../../platform/telemetry/common/telemetryUtils.js';
import { TelemetryLevel } from '../../../../platform/telemetry/common/telemetry.js';
import { IProductService } from '../../../../platform/product/common/productService.js';
import { INotificationService } from '../../../../platform/notification/common/notification.js';
import { localize } from '../../../../nls.js';
import { IEditorResolverService, RegisteredEditorPriority } from '../../../services/editor/common/editorResolverService.js';
import { TerminalCommandId } from '../../terminal/common/terminal.js';

export const restoreWalkthroughsConfigurationKey = 'workbench.welcomePage.restorableWalkthroughs';
export type RestoreWalkthroughsConfigurationValue = { folder: string; category?: string; step?: string };

const configurationKey = 'workbench.startupEditor';
const oldConfigurationKey = 'workbench.welcome.enabled';
const telemetryOptOutStorageKey = 'workbench.telemetryOptOutShown';

/**
 * @class StartupPageEditorResolverContribution
 * @brief Registers the 'Getting Started' page as a custom editor.
 * @details This class is a workbench contribution that ensures the 'Getting Started'
 * page can be opened as a regular editor within VS Code. It maps the custom
 * `getting-started:` URI scheme to the `GettingStartedInput` editor.
 */
export class StartupPageEditorResolverContribution implements IWorkbenchContribution {

	static readonly ID = 'workbench.contrib.startupPageEditorResolver';

	constructor(
		@IInstantiationService private readonly instantiationService: IInstantiationService,
		@IEditorResolverService editorResolverService: IEditorResolverService
	) {
		// Registers an editor for resources with the 'getting-started' scheme.
		editorResolverService.registerEditor(
			`${GettingStartedInput.RESOURCE.scheme}:/**`,
			{
				id: GettingStartedInput.ID,
				label: localize('welcome.displayName', "Welcome Page"),
				priority: RegisteredEditorPriority.builtin,
			},
			{
				singlePerResource: false,
				canSupportResource: uri => uri.scheme === GettingStartedInput.RESOURCE.scheme,
			},
			{
				createEditorInput: ({ resource, options }) => {
					return {
						editor: this.instantiationService.createInstance(GettingStartedInput, options as GettingStartedEditorOptions),
						options: {
							...options,
							pinned: false
						}
					};
				}
			}
		);
	}
}

/**
 * @class StartupPageRunnerContribution
 * @brief Orchestrates the logic for opening a startup editor (e.g., Welcome Page, README) on workbench startup.
 * @details This class is the primary driver for the startup page functionality. It listens for the
 * workbench to be restored and then decides whether to open a specific editor based on user settings,
 * workspace state, and whether there are pending backups.
 */
export class StartupPageRunnerContribution extends Disposable implements IWorkbenchContribution {

	static readonly ID = 'workbench.contrib.startupPageRunner';

	constructor(
		@IConfigurationService private readonly configurationService: IConfigurationService,
		@IEditorService private readonly editorService: IEditorService,
		@IWorkingCopyBackupService private readonly workingCopyBackupService: IWorkingCopyBackupService,
		@IFileService private readonly fileService: IFileService,
		@IWorkspaceContextService private readonly contextService: IWorkspaceContextService,
		@ILifecycleService private readonly lifecycleService: ILifecycleService,
		@IWorkbenchLayoutService private readonly layoutService: IWorkbenchLayoutService,
		@IProductService private readonly productService: IProductService,
		@ICommandService private readonly commandService: ICommandService,
		@IWorkbenchEnvironmentService private readonly environmentService: IWorkbenchEnvironmentService,
		@IStorageService private readonly storageService: IStorageService,
		@INotificationService private readonly notificationService: INotificationService
	) {
		super();
		this.run().then(undefined, onUnexpectedError);
		this._register(this.editorService.onDidCloseEditor((e) => {
			if (e.editor instanceof GettingStartedInput) {
				e.editor.selectedCategory = undefined;
				e.editor.selectedStep = undefined;
			}
		}));
	}

	/**
	 * @brief Determines and executes the appropriate startup action.
	 * @details This is the core logic that runs after the workbench is restored. It checks various
	 * conditions to decide whether to open a walkthrough, a README, the 'Getting Started' page,
	 * or a terminal.
	 */
	private async run() {
		// Wait for the workbench to be fully restored to avoid impacting startup performance.
		await this.lifecycleService.when(LifecyclePhase.Restored);

		// For first-time launches where telemetry is enabled, always show the welcome page
		// to present the telemetry opt-out option.
		if (
			this.productService.enableTelemetry
			&& this.productService.showTelemetryOptOut
			&& getTelemetryLevel(this.configurationService) !== TelemetryLevel.NONE
			&& !this.environmentService.skipWelcome
			&& !this.storageService.get(telemetryOptOutStorageKey, StorageScope.PROFILE)
		) {
			this.storageService.store(telemetryOptOutStorageKey, true, StorageScope.PROFILE, StorageTarget.USER);
		}

		// If a walkthrough was in progress for the current folder, restore it.
		if (this.tryOpenWalkthroughForFolder()) {
			return;
		}

		const enabled = isStartupPageEnabled(this.configurationService, this.contextService, this.environmentService);
		// Pre-condition: Do not open a startup editor if the window is being reloaded.
		if (enabled && this.lifecycleService.startupKind !== StartupKind.ReloadedWindow) {
			const hasBackups = await this.workingCopyBackupService.hasBackups();
			// Pre-condition: If there are file backups to restore, do not open a startup page
			// as the user likely wants to restore their previous session.
			if (hasBackups) { return; }

			// Open a startup editor if no other editor is active or if only default editors were opened.
			if (!this.editorService.activeEditor || this.layoutService.openedDefaultEditors) {
				const startupEditorSetting = this.configurationService.inspect<string>(configurationKey);

				// Based on the 'workbench.startupEditor' setting, open the corresponding editor.
				if (startupEditorSetting.value === 'readme') {
					await this.openReadme();
				} else if (startupEditorSetting.value === 'welcomePage' || startupEditorSetting.value === 'welcomePageInEmptyWorkbench') {
					await this.openGettingStarted();
				} else if (startupEditorSetting.value === 'terminal') {
					this.commandService.executeCommand(TerminalCommandId.CreateTerminalEditor);
				}
			}
		}
	}

	/**
	 * @brief Attempts to restore a walkthrough that was open when the window was last closed.
	 * @returns `true` if a walkthrough was successfully restored, `false` otherwise.
	 */
	private tryOpenWalkthroughForFolder(): boolean {
		const toRestore = this.storageService.get(restoreWalkthroughsConfigurationKey, StorageScope.PROFILE);
		if (!toRestore) {
			return false;
		}
		else {
			const restoreData: RestoreWalkthroughsConfigurationValue = JSON.parse(toRestore);
			const currentWorkspace = this.contextService.getWorkspace();
			// Check if the stored folder matches the current workspace.
			if (restoreData.folder === UNKNOWN_EMPTY_WINDOW_WORKSPACE.id || restoreData.folder === currentWorkspace.folders[0].uri.toString()) {
				const options: GettingStartedEditorOptions = { selectedCategory: restoreData.category, selectedStep: restoreData.step, pinned: false };
				// Re-open the 'Getting Started' page to the specific category and step.
				this.editorService.openEditor({
					resource: GettingStartedInput.RESOURCE,
					options
				});
				this.storageService.remove(restoreWalkthroughsConfigurationKey, StorageScope.PROFILE);
				return true;
			}
		}
		return false;
	}

	/**
	 * @brief Finds and opens the README.md file from the root of the workspace.
	 * @details If multiple README files are found, it prioritizes `README.md`. If no README
	 * is found, it defaults to opening the 'Getting Started' page.
	 */
	private async openReadme() {
		const readmes = arrays.coalesce(
			await Promise.all(this.contextService.getWorkspace().folders.map(
				async folder => {
					const folderUri = folder.uri;
					const folderStat = await this.fileService.resolve(folderUri).catch(onUnexpectedError);
					const files = folderStat?.children ? folderStat.children.map(child => child.name).sort() : [];
					// Find a file named 'readme.md' (case-insensitive) or starting with 'readme'.
					const file = files.find(file => file.toLowerCase() === 'readme.md') || files.find(file => file.toLowerCase().startsWith('readme'));
					if (file) { return joinPath(folderUri, file); }
					else { return undefined; }
				})));

		// Only open a README if no other editor is already active.
		if (!this.editorService.activeEditor) {
			if (readmes.length) {
				const isMarkDown = (readme: URI) => readme.path.toLowerCase().endsWith('.md');
				// Open markdown files in preview mode and other files in a standard editor.
				await Promise.all([
					this.commandService.executeCommand('markdown.showPreview', null, readmes.filter(isMarkDown), { locked: true }).catch(error => {
						this.notificationService.error(localize('startupPage.markdownPreviewError', 'Could not open markdown preview: {0}.\n\nPlease make sure the markdown extension is enabled.', error.message));
					}),
					this.editorService.openEditors(readmes.filter(readme => !isMarkDown(readme)).map(readme => ({ resource: readme }))),
				]);
			} else {
				// If no readme is found, fall back to showing the welcome page.
				await this.openGettingStarted();
			}
		}
	}

	/**
	 * @brief Opens the 'Getting Started' editor page.
	 * @details It avoids opening the page if it's already open to prevent duplicates.
	 */
	private async openGettingStarted(showTelemetryNotice?: boolean) {
		const startupEditorTypeID = gettingStartedInputTypeId;
		const editor = this.editorService.activeEditor;

		// Invariant: Do not open the 'Getting Started' page if it is already open or active.
		if (editor?.typeId === startupEditorTypeID || this.editorService.editors.some(e => e.typeId === startupEditorTypeID)) {
			return;
		}

		const options: GettingStartedEditorOptions = editor ? { pinned: false, index: 0, showTelemetryNotice } : { pinned: false, showTelemetryNotice };
		// Programmatically open the editor using its registered type ID.
		if (startupEditorTypeID === gettingStartedInputTypeId) {
			this.editorService.openEditor({
				resource: GettingStartedInput.RESOURCE,
				options,
			});
		}
	}
}

/**
 * @brief Checks if any startup page functionality is enabled via configuration.
 * @param configurationService Service to inspect user/workspace settings.
 * @param contextService Service to get information about the current workspace.
 * @param environmentService Service to check for environment flags like `skipWelcome`.
 * @returns `true` if a startup editor should be considered, `false` otherwise.
 */
function isStartupPageEnabled(configurationService: IConfigurationService, contextService: IWorkspaceContextService, environmentService: IWorkbenchEnvironmentService) {
	// Startup page can be disabled globally via an environment flag.
	if (environmentService.skipWelcome) {
		return false;
	}

	const startupEditor = configurationService.inspect<string>(configurationKey);
	// For backward compatibility, check the old 'workbench.welcome.enabled' setting if the new one isn't set.
	if (!startupEditor.userValue && !startupEditor.workspaceValue) {
		const welcomeEnabled = configurationService.inspect(oldConfigurationKey);
		if (welcomeEnabled.value !== undefined && welcomeEnabled.value !== null) {
			return welcomeEnabled.value;
		}
	}

	// Check for the various values of the 'workbench.startupEditor' setting.
	return startupEditor.value === 'welcomePage'
		|| startupEditor.value === 'readme'
		|| (contextService.getWorkbenchState() === WorkbenchState.EMPTY && startupEditor.value === 'welcomePageInEmptyWorkbench')
		|| startupEditor.value === 'terminal';
}
