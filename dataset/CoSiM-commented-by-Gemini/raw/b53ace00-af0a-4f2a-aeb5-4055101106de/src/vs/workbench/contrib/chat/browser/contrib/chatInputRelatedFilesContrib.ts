/**
 * @fileoverview
 * @module chatInputRelatedFilesContrib
 * @description
 * This workbench contribution is responsible for providing related file suggestions within a chat session.
 * It observes the active chat editing session and, based on user input,
 * asynchronously fetches and displays relevant files that can be added to the chat's working set.
 * The entire mechanism is reactive, leveraging an observable pattern to respond to changes
 * in the chat session state.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { CancellationToken } from '../../../../../base/common/cancellation.js';
import { Event } from '../../../../../base/common/event.js';
import { Disposable, DisposableStore } from '../../../../../base/common/lifecycle.js';
import { ResourceMap } from '../../../../../base/common/map.js';
import { autorun } from '../../../../../base/common/observable.js';
import { URI } from '../../../../../base/common/uri.js';
import { localize } from '../../../../../nls.js';
import { IWorkbenchContribution } from '../../../../common/contributions.js';
import { ChatEditingSessionChangeType, IChatEditingService, IChatEditingSession, WorkingSetEntryRemovalReason, WorkingSetEntryState } from '../../common/chatEditingService.js';
import { IChatWidgetService } from '../chat.js';

/**
 * @class ChatRelatedFilesContribution
 * @description A workbench contribution that enhances the chat experience by suggesting related files.
 * It acts as a controller that listens to the global chat editing session and triggers
 * the retrieval and display of file suggestions in the chat widget.
 */
export class ChatRelatedFilesContribution extends Disposable implements IWorkbenchContribution {
	static readonly ID = 'chat.relatedFilesWorkingSet';

	private readonly chatEditingSessionDisposables = new DisposableStore();
	private _currentRelatedFilesRetrievalOperation: Promise<void> | undefined;

	constructor(
		/**
		 * @property {IChatEditingService} chatEditingService
		 * @description Service for managing the state of the chat editing session, including the working set of files.
		 */
		@IChatEditingService private readonly chatEditingService: IChatEditingService,
		/**
		 * @property {IChatWidgetService} chatWidgetService
		 * @description Service for accessing and interacting with chat widget instances.
		 */
		@IChatWidgetService private readonly chatWidgetService: IChatWidgetService
	) {
		super();

		/**
		 * @description This autorun block is the entry point for the contribution's logic.
		 * It reactively listens for changes to the global chat editing session. When a new
		 * session becomes active, it sets up the necessary event listeners for that session.
		 * This ensures that file suggestions are always relevant to the current chat context.
		 */
		this._register(autorun(r => {
			this.chatEditingSessionDisposables.clear();
			const session = this.chatEditingService.globalEditingSessionObs.read(r);
			if (session) {
				this._handleNewEditingSession(session);
			}
		}));
	}

	/**
	 * @private
	 * @method _updateRelatedFileSuggestions
	 * @description Orchestrates the retrieval and updating of related file suggestions.
	 * This method is the core of the contribution's logic. It ensures that only one
	 * retrieval operation is active at a time and fetches suggestions only when
	 * the working set is in its initial, empty state. It then updates the working
	 * set with new suggestions, limited to a small number to avoid cluttering the UI.
	 */
	private _updateRelatedFileSuggestions() {
		// Block Logic: Ensures that only one file retrieval operation can be in progress at any given time,
		// preventing redundant or conflicting updates.
		if (this._currentRelatedFilesRetrievalOperation) {
			return;
		}

		const currentEditingSession = this.chatEditingService.globalEditingSessionObs.get();
		if (!currentEditingSession) {
			return;
		}
		// Block Logic: Fetches suggestions only for the initial state of the working set (when it's empty).
		// This is intended to provide helpful context at the beginning of a chat interaction.
		const workingSetEntries = currentEditingSession.entries.get();
		if (workingSetEntries.length > 0) {
			// Do this only for the initial working set state
			return;
		}

		const widget = this.chatWidgetService.getWidgetBySessionId(currentEditingSession.chatSessionId);
		if (!widget) {
			return;
		}

		// Asynchronously fetch related files from the editing service. This operation is cancellable.
		this._currentRelatedFilesRetrievalOperation = this.chatEditingService.getRelatedFiles(currentEditingSession.chatSessionId, widget.getInput(), CancellationToken.None)
			.then((files) => {
				if (!files?.length) {
					return;
				}

				const currentEditingSession = this.chatEditingService.globalEditingSessionObs.get();
				// Pre-condition: Check if the session is still valid and unchanged before applying updates.
				// This is a safeguard against race conditions where the session might have changed during the async operation.
				if (!currentEditingSession || currentEditingSession.chatSessionId !== widget.viewModel?.sessionId || currentEditingSession.entries.get().length) {
					return; // Might have disposed while we were calculating
				}

				// Block Logic: Limits the number of suggestions to a maximum of 2, or the available space in the working set.
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

				// Block Logic: Performs a differential update of the working set. It removes any
				// existing suggestions that are not in the new set of suggestions.
				const existingSuggestedEntriesToRemove: URI[] = [];
				for (const entry of currentEditingSession.workingSet) {
					if (entry[1].state === WorkingSetEntryState.Suggested && !newSuggestions.has(entry[0])) {
						existingSuggestedEntriesToRemove.push(entry[0]);
					}
				}
				currentEditingSession?.remove(WorkingSetEntryRemovalReason.Programmatic, ...existingSuggestedEntriesToRemove);

				// Block Logic: Adds the new file suggestions to the working set with a "Suggested" state.
				for (const [uri, data] of newSuggestions) {
					currentEditingSession.addFileToWorkingSet(uri, localize('relatedFile', "{0} (Suggested)", data.description), WorkingSetEntryState.Suggested);
				}
			})
			.finally(() => {
				// Resets the operation lock to allow for future updates.
				this._currentRelatedFilesRetrievalOperation = undefined;
			});

	}

	/**
	 * @private
	 * @method _handleNewEditingSession
	 * @param {IChatEditingSession} currentEditingSession The newly activated chat editing session.
	 * @description Sets up all necessary event listeners for a new chat session. This includes
	 * listening for input changes (debounced to avoid excessive updates), changes to the
	 * working set, and the disposal of the session itself.
	 */
	private _handleNewEditingSession(currentEditingSession: IChatEditingSession) {

		const widget = this.chatWidgetService.getWidgetBySessionId(currentEditingSession.chatSessionId);
		if (!widget || widget.viewModel?.sessionId !== currentEditingSession.chatSessionId) {
			return;
		}
		// Invariant: The disposables store is cleared when the session is disposed,
		// ensuring no memory leaks from dangling event listeners.
		this.chatEditingSessionDisposables.add(currentEditingSession.onDidDispose(() => {
			this.chatEditingSessionDisposables.clear();
		}));
		this._updateRelatedFileSuggestions();
		// Functional Utility: Debounces the input change event to trigger suggestion updates
		// only after the user has paused typing for a certain duration (3000ms).
		const onDebouncedType = Event.debounce(widget.inputEditor.onDidChangeModelContent, () => null, 3000);
		this.chatEditingSessionDisposables.add(onDebouncedType(() => {
			this._updateRelatedFileSuggestions();
		}));
		this.chatEditingSessionDisposables.add(currentEditingSession.onDidChange((e) => {
			// Pre-condition: Only update suggestions if the working set has changed.
			if (e === ChatEditingSessionChangeType.WorkingSet) {
				this._updateRelatedFileSuggestions();
			}
		}));
	}

	override dispose() {
		this.chatEditingSessionDisposables.dispose();
		super.dispose();
	}
}