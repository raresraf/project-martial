/**
 * @fileoverview This file defines the `ChatRelatedFilesContribution`, a workbench contribution
 * responsible for suggesting files related to the user's input within a chat editing session.
 * It automatically fetches relevant files and adds them to the session's working set as
 * suggestions, updating them dynamically as the user types.
 */
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
 * A workbench contribution that manages suggesting related files in a chat editing session.
 */
export class ChatRelatedFilesContribution extends Disposable implements IWorkbenchContribution {
	static readonly ID = 'chat.relatedFilesWorkingSet';

	/**
	 * A map to store disposable stores for each chat editing session to manage listeners.
	 */
	private readonly chatEditingSessionDisposables = new Map<string, DisposableStore>();
	/**
	 * A promise that tracks the current in-progress file retrieval operation to prevent concurrent calls.
	 */
	private _currentRelatedFilesRetrievalOperation: Promise<void> | undefined;

	constructor(
		@IChatEditingService private readonly chatEditingService: IChatEditingService,
		@IChatWidgetService private readonly chatWidgetService: IChatWidgetService
	) {
		super();

		/**
		 * Register a listener for when a new chat widget is added. If it's part of an
		 * editing session, set up the necessary handlers for that session.
		 */
		this._register(
			this.chatWidgetService.onDidAddWidget(widget => {
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
	 * Asynchronously fetches and updates related file suggestions for the current chat editing session.
	 * This method is designed to provide initial file suggestions when the working set is empty.
	 * @param currentEditingSession The active chat editing session.
	 * @param widget The associated chat widget.
	 */
	private _updateRelatedFileSuggestions(currentEditingSession: IChatEditingSession, widget: IChatWidget) {
		// Block concurrent retrieval operations.
		if (this._currentRelatedFilesRetrievalOperation) {
			return;
		}

		// Only provide initial suggestions if the working set is empty.
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

				const currentEditingSession = this.chatEditingService.globalEditingSessionObs.get();
				// Check if the session is still valid and unchanged.
				if (!currentEditingSession || currentEditingSession.chatSessionId !== widget.viewModel?.sessionId || currentEditingSession.entries.get().length) {
					return; // Might have been disposed or modified while calculating.
				}

				// Determine the maximum number of suggestions to add (up to 2).
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

				// Identify and remove any existing suggestions that are no longer relevant.
				const existingSuggestedEntriesToRemove: URI[] = [];
				for (const entry of currentEditingSession.workingSet) {
					if (entry[1].state === WorkingSetEntryState.Suggested && !newSuggestions.has(entry[0])) {
						existingSuggestedEntriesToRemove.push(entry[0]);
					}
				}
				currentEditingSession?.remove(WorkingSetEntryRemovalReason.Programmatic, ...existingSuggestedEntriesToRemove);

				// Add the new file suggestions to the working set.
				for (const [uri, data] of newSuggestions) {
					currentEditingSession.addFileToWorkingSet(uri, localize('relatedFile', "{0} (Suggested)", data.description), WorkingSetEntryState.Suggested);
				}
			})
			.finally(() => {
				// Release the operation lock.
				this._currentRelatedFilesRetrievalOperation = undefined;
			});

	}

	/**
	 * Sets up listeners for a new chat editing session to dynamically update related file suggestions.
	 * @param currentEditingSession The new chat editing session.
	 * @param widget The associated chat widget.
	 */
	private _handleNewEditingSession(currentEditingSession: IChatEditingSession, widget: IChatWidget) {
		const disposableStore = new DisposableStore();
		disposableStore.add(currentEditingSession.onDidDispose(() => {
			disposableStore.clear();
		}));

		// Trigger an initial update for suggestions.
		this._updateRelatedFileSuggestions(currentEditingSession, widget);

		// Set up a debounced listener to update suggestions as the user types.
		const onDebouncedType = Event.debounce(widget.inputEditor.onDidChangeModelContent, () => null, 3000);
		disposableStore.add(onDebouncedType(() => {
			this._updateRelatedFileSuggestions(currentEditingSession, widget);
		}));

		// Listen for manual changes to the working set to re-evaluate suggestions.
		disposableStore.add(currentEditingSession.onDidChange((e) => {
			if (e === ChatEditingSessionChangeType.WorkingSet) {
				this._updateRelatedFileSuggestions(currentEditingSession, widget);
			}
		}));

		// Ensure cleanup when the session is disposed.
		disposableStore.add(currentEditingSession.onDidDispose(() => {
			disposableStore.dispose();
		}));
		this.chatEditingSessionDisposables.set(currentEditingSession.chatSessionId, disposableStore);
	}

	/**
	 * Disposes of the contribution, cleaning up all listeners for all tracked sessions.
	 */
	override dispose() {
		for (const store of this.chatEditingSessionDisposables.values()) {
			store.dispose();
		}
		super.dispose();
	}
}