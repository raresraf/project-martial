/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { CancellationToken } from '../../../../../base/common/cancellation.js';
import { Event } from '../../../../../base/common/event.js';
import { Disposable, DisposableStore } from '../../../../../base/common/lifecycle.js';
import { ResourceMap } from '../../../../../base/common/map.js';
import { URI } from '../../../../../base/common/uri.js';
import { localize } from '../../../../../nls.js';
import { IWorkbenchContribution } from '../../../../common/contributions.js';
import { ChatAgentLocation } from '../../common/chatAgents.js';
import { ChatEditingSessionChangeType, IChatEditingService, IChatEditingSession, WorkingSetEntryRemovalReason, WorkingSetEntryState } from '../../common/chatEditingService.js';
import { IChatWidget, IChatWidgetService } from '../chat.js';

/**
 * A workbench contribution that provides related file suggestions for chat editing sessions.
 * This class monitors chat widgets and, when a relevant editing session is active,
 * it dynamically fetches and suggests files that are related to the user's input.
 */
export class ChatRelatedFilesContribution extends Disposable implements IWorkbenchContribution {
	static readonly ID = 'chat.relatedFilesWorkingSet';

	private readonly chatEditingSessionDisposables = new Map<string, DisposableStore>();
	private _currentRelatedFilesRetrievalOperation: Promise<void> | undefined;

	constructor(
		@IChatEditingService private readonly chatEditingService: IChatEditingService,
		@IChatWidgetService private readonly chatWidgetService: IChatWidgetService
	) {
		super();

		this._register(
			this.chatWidgetService.onDidAddWidget(widget => {
				// When a new chat widget is added, check if it's for an editing session.
				if (widget.location === ChatAgentLocation.EditingSession && widget.viewModel?.sessionId) {
					const editingSession = this.chatEditingService.getEditingSession(widget.viewModel.sessionId);
					if (editingSession) {
						this._handleNewEditingSession(editingSession, widget);
					}
				}
			}),
		);
	}

	/**
	 * Fetches and updates related file suggestions for a given chat editing session.
	 * This method is asynchronous and debounced to prevent excessive calls.
	 * @param currentEditingSession The active chat editing session.
	 * @param widget The associated chat widget.
	 */
	private _updateRelatedFileSuggestions(currentEditingSession: IChatEditingSession, widget: IChatWidget) {
		// Prevent multiple concurrent retrieval operations.
		if (this._currentRelatedFilesRetrievalOperation) {
			return;
		}

		// Only provide suggestions for the initial empty working set state.
		const workingSetEntries = currentEditingSession.entries.get();
		if (workingSetEntries.length > 0) {
			return;
		}

		// Asynchronously get related files from the editing service.
		this._currentRelatedFilesRetrievalOperation = this.chatEditingService.getRelatedFiles(currentEditingSession.chatSessionId, widget.getInput(), CancellationToken.None)
			.then((files) => {
				if (!files?.length) {
					return;
				}

				// Ensure the session is still valid and unchanged.
				const currentEditingSession = this.chatEditingService.globalEditingSessionObs.get();
				if (!currentEditingSession || currentEditingSession.chatSessionId !== widget.viewModel?.sessionId || currentEditingSession.entries.get().length) {
					return;
				}

				// Determine the number of suggestions to show (up to 2).
				const maximumRelatedFiles = Math.min(2, this.chatEditingService.editingSessionFileLimit - widget.input.chatEditWorkingSetFiles.length);
				const newSuggestions = new ResourceMap<{ description: string; group: string }>();
				for (const group of files) {
					for (const file of group.files) {
						if (newSuggestions.size >= maximumRelatedFiles) {
							break;
						}
						newSuggestions.set(file.uri, { group: group.group, description: file.description });
					}
				}

				// Remove previous suggestions that are no longer relevant.
				const existingSuggestedEntriesToRemove: URI[] = [];
				for (const entry of currentEditingSession.workingSet) {
					if (entry[1].state === WorkingSetEntryState.Suggested && !newSuggestions.has(entry[0])) {
						existingSuggestedEntriesToRemove.push(entry[0]);
					}
				}
				currentEditingSession?.remove(WorkingSetEntryRemovalReason.Programmatic, ...existingSuggestedEntriesToRemove);

				// Add the new suggestions to the working set.
				for (const [uri, data] of newSuggestions) {
					currentEditingSession.addFileToWorkingSet(uri, localize('relatedFile', "{0} (Suggested)", data.description), WorkingSetEntryState.Suggested);
				}
			})
			.finally(() => {
				this._currentRelatedFilesRetrievalOperation = undefined;
			});

	}

	/**
	 * Sets up event listeners for a new chat editing session to handle related file suggestions.
	 * @param currentEditingSession The new chat editing session.
	 * @param widget The associated chat widget.
	 */
	private _handleNewEditingSession(currentEditingSession: IChatEditingSession, widget: IChatWidget) {
		const disposableStore = new DisposableStore();
		disposableStore.add(currentEditingSession.onDidDispose(() => {
			disposableStore.clear();
		}));
		this._updateRelatedFileSuggestions(currentEditingSession, widget);
		
		// Debounce input changes to avoid excessive updates.
		const onDebouncedType = Event.debounce(widget.inputEditor.onDidChangeModelContent, () => null, 3000);
		disposableStore.add(onDebouncedType(() => {
			this._updateRelatedFileSuggestions(currentEditingSession, widget);
		}));
		
		// Also update on explicit working set changes.
		disposableStore.add(currentEditingSession.onDidChange((e) => {
			if (e === ChatEditingSessionChangeType.WorkingSet) {
				this._updateRelatedFileSuggestions(currentEditingSession, widget);
			}
		}));
		
		// Clean up when the session is disposed.
		disposableStore.add(currentEditingSession.onDidDispose(() => {
			disposableStore.dispose();
		}));
		this.chatEditingSessionDisposables.set(currentEditingSession.chatSessionId, disposableStore);
	}

	override dispose() {
		// Dispose all stored disposables for the chat editing sessions.
		for (const store of this.chatEditingSessionDisposables.values()) {
			store.dispose();
		}
		super.dispose();
	}
}
