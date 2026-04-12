/**
 * @file chatInputRelatedFilesContrib.ts
 * @brief Workbench contribution for automated related-file suggestions in chat editing sessions.
 * @details Monitors active chat sessions and user input to proactively identify and suggest 
 * relevant workspace files for the current editing context. Leverages debounced event 
 * processing and reactive state management.
 * 
 * Domain: AI Chat, File Contextualization, Reactive UI.
 */

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
 * @brief Manages the lifecycle of "Suggested" file entries in a chat's working set.
 * Functional Utility: Enhances chat context by automatically proposing files that 
 * are topologically or semantically related to the user's active prompt.
 */
export class ChatRelatedFilesContribution extends Disposable implements IWorkbenchContribution {
	static readonly ID = 'chat.relatedFilesWorkingSet';

	/**
	 * Lifecycle: Aggregated disposables for the current editing session.
	 */
	private readonly chatEditingSessionDisposables = new DisposableStore();
	
	/**
	 * State: Reference to any in-flight asynchronous retrieval task to avoid duplicate operations.
	 */
	private _currentRelatedFilesRetrievalOperation: Promise<void> | undefined;

	constructor(
		@IChatEditingService private readonly chatEditingService: IChatEditingService,
		@IChatWidgetService private readonly chatWidgetService: IChatWidgetService
	) {
		super();

		/**
		 * Initialization: Sets up a reactive observer on the global editing session.
		 * Logic: Automatically binds/unbinds session handlers as sessions are created or disposed.
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
	 * @brief Orchestrates the retrieval and application of related file suggestions.
	 * Algorithm: Fetches relevant files based on widget input and merges them into 
	 * the session's working set using a least-recently-suggested eviction policy.
	 */
	private _updateRelatedFileSuggestions() {
		// Protocol: Ensure atomicity by blocking concurrent retrieval operations.
		if (this._currentRelatedFilesRetrievalOperation) {
			return;
		}

		const currentEditingSession = this.chatEditingService.globalEditingSessionObs.get();
		if (!currentEditingSession) {
			return;
		}
		
		/**
		 * Condition: Suggestions are only populated for fresh sessions with empty working sets 
		 * to avoid overriding manual user selections.
		 */
		const workingSetEntries = currentEditingSession.entries.get();
		if (workingSetEntries.length > 0) {
			return;
		}

		const widget = this.chatWidgetService.getWidgetBySessionId(currentEditingSession.chatSessionId);
		if (!widget) {
			return;
		}

		/**
		 * Asynchronous Task: Queries the chat service for files related to the current prompt.
		 */
		this._currentRelatedFilesRetrievalOperation = this.chatEditingService.getRelatedFiles(currentEditingSession.chatSessionId, widget.getInput(), CancellationToken.None)
			.then((files) => {
				if (!files?.length) {
					return;
				}

				// Validation: Ensure the session state is still consistent after the async jump.
				const currentEditingSession = this.chatEditingService.globalEditingSessionObs.get();
				if (!currentEditingSession || currentEditingSession.chatSessionId !== widget.viewModel?.sessionId || currentEditingSession.entries.get().length) {
					return;
				}

				/**
				 * Logic: Capacity-bounded suggestion generation.
				 * Limits the number of suggested files to 2 or the remaining space in the working set.
				 */
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

				/**
				 * Functional Utility: Working set pruning.
				 * Logic: Identifies previously suggested files that are no longer relevant 
				 * and marks them for removal.
				 */
				const existingSuggestedEntriesToRemove: URI[] = [];
				for (const entry of currentEditingSession.workingSet) {
					if (entry[1].state === WorkingSetEntryState.Suggested && !newSuggestions.has(entry[0])) {
						existingSuggestedEntriesToRemove.push(entry[0]);
					}
				}
				currentEditingSession?.remove(WorkingSetEntryRemovalReason.Programmatic, ...existingSuggestedEntriesToRemove);

				/**
				 * Update: Injects the newly identified suggestions into the session.
				 */
				for (const [uri, data] of newSuggestions) {
					currentEditingSession.addFileToWorkingSet(uri, localize('relatedFile', "{0} (Suggested)", data.description), WorkingSetEntryState.Suggested);
				}
			})
			.finally(() => {
				// Protocol: Unlocks the retrieval operation for future triggers.
				this._currentRelatedFilesRetrievalOperation = undefined;
			});

	}

	/**
	 * @brief Session lifecycle hook.
	 * @param currentEditingSession The active editing session instance.
	 * Logic: Sets up event listeners for user input (debounced) and state changes 
	 * to trigger re-evaluation of suggestions.
	 */
	private _handleNewEditingSession(currentEditingSession: IChatEditingSession) {
		const widget = this.chatWidgetService.getWidgetBySessionId(currentEditingSession.chatSessionId);
		if (!widget || widget.viewModel?.sessionId !== currentEditingSession.chatSessionId) {
			return;
		}

		// Resource Tracking: Ensures handlers are cleaned up on session disposal.
		this.chatEditingSessionDisposables.add(currentEditingSession.onDidDispose(() => {
			this.chatEditingSessionDisposables.clear();
		}));

		// Initial evaluation.
		this._updateRelatedFileSuggestions();

		/**
		 * Performance Optimization: Debounces input-driven re-evaluations (3000ms) 
		 * to prevent excessive computation during rapid typing.
		 */
		const onDebouncedType = Event.debounce(widget.inputEditor.onDidChangeModelContent, () => null, 3000);
		this.chatEditingSessionDisposables.add(onDebouncedType(() => {
			this._updateRelatedFileSuggestions();
		}));

		/**
		 * Logic: Responds to structural changes in the working set.
		 */
		this.chatEditingSessionDisposables.add(currentEditingSession.onDidChange((e) => {
			if (e === ChatEditingSessionChangeType.WorkingSet) {
				this._updateRelatedFileSuggestions();
			}
		}));
	}

	/**
	 * @brief Clean teardown of all reactive listeners and disposables.
	 */
	override dispose() {
		this.chatEditingSessionDisposables.dispose();
		super.dispose();
	}
}
