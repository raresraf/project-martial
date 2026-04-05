/**
 * @file This file is responsible for the logic that determines which page, if any,
 * is shown to the user on workbench startup. It handles showing the 'Welcome'
 * page, a project's README file, a terminal, or restoring a previous
 * walkthrough, based on a variety of configuration settings and workspace contexts.
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
import { ILogService } from '../../../../platform/log/common/log.js';
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
 * @implements {IWorkbenchContribution}
 * @brief Registers the 'Welcome Page' as a custom editor within the workbench.
 * @details This contribution allows the 'Welcome Page' to be treated like any other
 * editor, with a custom URI scheme (`getting-started:`). It defines how to
 * create an instance of the `GettingStartedInput` when such a URI is opened.
 */
export class StartupPageEditorResolverContribution implements IWorkbenchContribution {

	static readonly ID = 'workbench.contrib.startupPageEditorResolver';

	constructor(
		@IInstantiationService private readonly instantiationService: IInstantiationService,
		@IEditorResolverService editorResolverService: IEditorResolverService
	) {
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
 * @implements {IWorkbenchContribution}
 * @brief Manages the logic for opening a startup page when the workbench launches.
 * @details This is the core class that orchestrates the startup page behavior. It
 * waits for the workbench to be restored, then evaluates various conditions
 * (e.g., first launch, user settings, workspace state) to decide whether to open
 * the 'Welcome' page, a README file, or another configured startup editor.
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
		@ILogService private readonly logService: ILogService,
		@INotificationService private readonly notificationService: INotificationService
	) {
		super();
		this.run().then(undefined, onUnexpectedError);
		// When a getting started editor is closed, reset its internal navigation state.
		this._register(this.editorService.onDidCloseEditor((e) => {
			if (e.editor instanceof GettingStartedInput) {
				e.editor.selectedCategory = undefined;
				e.editor.selectedStep = undefined;
			}
		}));
	}

	/**
	 * @brief Core logic to determine and open a startup editor.
	 * This method is executed after the workbench has been restored. It contains
	 * the primary decision tree for the startup page feature.
	 */
	private async run() {
		// Defer execution until the workbench is restored to avoid impacting startup time.
		await this.lifecycleService.when(LifecyclePhase.Restored);

		// Special case: On first launch with telemetry enabled, always show the Welcome
		// page to present the user with telemetry settings and opt-out information.
		if (
			this.productService.enableTelemetry
			&& this.productService.showTelemetryOptOut
			&& getTelemetryLevel(this.configurationService) !== TelemetryLevel.NONE
			&& !this.environmentService.skipWelcome
			&& !this.storageService.get(telemetryOptOutStorageKey, StorageScope.PROFILE)
		) {
			this.storageService.store(telemetryOptOutStorageKey, true, StorageScope.PROFILE, StorageTarget.USER);
			await this.openGettingStarted(true);
			return;
		}

		// If a walkthrough was in progress, attempt to restore it.
		if (this.tryOpenWalkthroughForFolder()) {
			return;
		}

		const enabled = isStartupPageEnabled(this.configurationService, this.contextService, this.environmentService);
		// Pre-condition: Only run on first launch, not on window reload.
		if (enabled && this.lifecycleService.startupKind !== StartupKind.ReloadedWindow) {
			const hasBackups = await this.workingCopyBackupService.hasBackups();
			// Pre-condition: If there are dirty files to restore, suppress the startup page.
			if (hasBackups) { return; }

			// Open a startup editor only if no other editor is already active or if only default editors were opened.
			if (!this.editorService.activeEditor || this.layoutService.openedDefaultEditors) {
				const startupEditorSetting = this.configurationService.inspect<string>(configurationKey);

				const isStartupEditorReadme = startupEditorSetting.value === 'readme';
				// The 'readme' setting is only honored if it comes from user or default settings,
				// not from workspace settings, to prevent potentially untrusted code execution.
				const isStartupEditorUserReadme = startupEditorSetting.userValue === 'readme';
				const isStartupEditorDefaultReadme = startupEditorSetting.defaultValue === 'readme';

				if (isStartupEditorReadme && (!isStartupEditorUserReadme && !isStartupEditorDefaultReadme)) {
					this.logService.warn(`Warning: 'workbench.startupEditor: readme' setting ignored due to being set somewhere other than user or default settings (user=${startupEditorSetting.userValue}, default=${startupEditorSetting.defaultValue})`);
				}

				const openWithReadme = isStartupEditorReadme && (isStartupEditorUserReadme || isStartupEditorDefaultReadme);
				if (openWithReadme) {
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
	 * @brief Restores a walkthrough page if one was active in the folder.
	 * @returns True if a walkthrough was restored, false otherwise.
	 */
	private tryOpenWalkthroughForFolder(): boolean {
		const toRestore = this.storageService.get(restoreWalkthroughsConfigurationKey, StorageScope.PROFILE);
		if (!toRestore) {
			return false;
		}
		else {
			const restoreData: RestoreWalkthroughsConfigurationValue = JSON.parse(toRestore);
			const currentWorkspace = this.contextService.getWorkspace();
			// Check if the stored folder matches the current workspace before restoring.
			if (restoreData.folder === UNKNOWN_EMPTY_WINDOW_WORKSPACE.id || restoreData.folder === currentWorkspace.folders[0].uri.toString()) {
				const options: GettingStartedEditorOptions = { selectedCategory: restoreData.category, selectedStep: restoreData.step, pinned: false };
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
	 * @brief Finds and opens a README file from the workspace.
	 * Defaults to the Welcome page if no README is found.
	 */
	private async openReadme() {
		const readmes = arrays.coalesce(
			await Promise.all(this.contextService.getWorkspace().folders.map(
				async folder => {
					const folderUri = folder.uri;
					const folderStat = await this.fileService.resolve(folderUri).catch(onUnexpectedError);
					const files = folderStat?.children ? folderStat.children.map(child => child.name).sort() : [];
					// Find a file named 'readme.md' (case-insensitive) or any file starting with 'readme'.
					const file = files.find(file => file.toLowerCase() === 'readme.md') || files.find(file => file.toLowerCase().startsWith('readme'));
					if (file) { return joinPath(folderUri, file); }
					else { return undefined; }
				})));

		if (!this.editorService.activeEditor) {
			if (readmes.length) {
				const isMarkDown = (readme: URI) => readme.path.toLowerCase().endsWith('.md');
				// Open markdown files in preview mode and other files (e.g., .txt) in a standard editor.
				await Promise.all([
					this.commandService.executeCommand('markdown.showPreview', null, readmes.filter(isMarkDown), { locked: true }).catch(error => {
						this.notificationService.error(localize('startupPage.markdownPreviewError', 'Could not open markdown preview: {0}.\n\nPlease make sure the markdown extension is enabled.', error.message));
					}),
					this.editorService.openEditors(readmes.filter(readme => !isMarkDown(readme)).map(readme => ({ resource: readme }))),
				]);
			} else {
				// Fallback: If no README is found, open the Welcome page instead.
				await this.openGettingStarted();
			}
		}
	}

	/**
	 * @brief Opens the 'Getting Started' (Welcome) editor.
	 * @param showTelemetryNotice - If true, indicates that the page should include a notice about telemetry.
	 * It ensures that the welcome editor is not opened more than once.
	 */
	private async openGettingStarted(showTelemetryNotice?: boolean) {
		const startupEditorTypeID = gettingStartedInputTypeId;
		const editor = this.editorService.activeEditor;

		// Invariant: Do not open the 'Getting Started' editor if one is already open.
		if (editor?.typeId === startupEditorTypeID || this.editorService.editors.some(e => e.typeId === startupEditorTypeID)) {
			return;
		}

		const options: GettingStartedEditorOptions = editor ? { pinned: false, index: 0, showTelemetryNotice } : { pinned: false, showTelemetryNotice };
		if (startupEditorTypeID === gettingStartedInputTypeId) {
			this.editorService.openEditor({
				resource: GettingStartedInput.RESOURCE,
				options,
			});
		}
	}
}

/**
 * @brief Determines if any startup page functionality is enabled.
 * @returns True if a startup editor should be considered, false otherwise.
 * @details This function checks multiple configuration settings, including the current
 * `workbench.startupEditor` and the legacy `workbench.welcome.enabled`, as well as
 * environment flags.
 */
function isStartupPageEnabled(configurationService: IConfigurationService, contextService: IWorkspaceContextService, environmentService: IWorkbenchEnvironmentService) {
	// Check for a global override flag.
	if (environmentService.skipWelcome) {
		return false;
	}

	const startupEditor = configurationService.inspect<string>(configurationKey);
	// For backward compatibility, if the new key is not set, check the old key.
	if (!startupEditor.userValue && !startupEditor.workspaceValue) {
		const welcomeEnabled = configurationService.inspect(oldConfigurationKey);
		if (welcomeEnabled.value !== undefined && welcomeEnabled.value !== null) {
			return welcomeEnabled.value;
		}
	}
	// Check for the various valid settings that enable a startup editor.
	return startupEditor.value === 'welcomePage'
		|| startupEditor.value === 'readme' && (startupEditor.userValue === 'readme' || startupEditor.defaultValue === 'readme')
		|| (contextService.getWorkbenchState() === WorkbenchState.EMPTY && startupEditor.value === 'welcomePageInEmptyWorkbench')
		|| startupEditor.value === 'terminal';
}
