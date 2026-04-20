/**
 * @b53ace00-af0a-4f2a-aeb5-4055101106de/src/vs/workbench/contrib/chat/browser/contrib/chatInputRelatedFilesContrib.ts
 * @brief Workbench contribution for automated related-file suggestions in chat editing sessions.
 * 
 * Functional Intent: Proactively identifies and suggests workspace files that are 
 * semantically or topologically related to the user's active chat prompt. It 
 * manages the lifecycle of 'Suggested' working set entries, ensuring they 
 * remain relevant to the conversation context without interfering with manual 
 * user selections.
 * 
 * Domain: AI-Assisted Development, Contextual Suggestion, Reactive UI.
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
 * @brief Manages the lifecycle and state transitions of suggested files in a chat's working set.
 * 
 * Logic: Leverages a reactive observer on the global editing session to bind 
 * listeners to user input and structural session changes.
 */
export class ChatRelatedFilesContribution extends Disposable implements IWorkbenchContribution {
	static readonly ID = 'chat.relatedFilesWorkingSet';

	private readonly chatEditingSessionDisposables = new DisposableStore();
	private _currentRelatedFilesRetrievalOperation: Promise<void> | undefined;

	constructor(
		@IChatEditingService private readonly chatEditingService: IChatEditingService,
		@IChatWidgetService private readonly chatWidgetService: IChatWidgetService
	) {
		super();

		// Block Logic: Global session lifecycle tracking.
		// Invariant: Maintains exactly one set of handlers for the active global session.
		this._register(autorun(r => {
			this.chatEditingSessionDisposables.clear();
			const session = this.chatEditingService.globalEditingSessionObs.read(r);
			if (session) {
				this._handleNewEditingSession(session);
			}
		}));
	}

	/**
	 * _updateRelatedFileSuggestions - Orchestrates retrieval and pruning of suggestions.
	 * 
	 * Algorithm: Debounced context-aware suggestions.
	 * Logic: 
	 * 1. Blocks concurrent retrieval attempts (atomicity).
	 * 2. Only triggers for fresh sessions with empty working sets (safety).
	 * 3. Invokes AI-driven file resolution based on current widget input.
	 * 4. Merges results into the working set, applying a capacity limit (max 2) 
	 *    and evicting stale suggestions that no longer appear in the resolve set.
	 */
	private _updateRelatedFileSuggestions() {
		if (this._currentRelatedFilesRetrievalOperation) {
			return;
		}

		const currentEditingSession = this.chatEditingService.globalEditingSessionObs.get();
		if (!currentEditingSession) {
			return;
		}
		
		const workingSetEntries = currentEditingSession.entries.get();
		if (workingSetEntries.length > 0) {
			return;
		}

		const widget = this.chatWidgetService.getWidgetBySessionId(currentEditingSession.chatSessionId);
		if (!widget) {
			return;
		}

		this._currentRelatedFilesRetrievalOperation = this.chatEditingService.getRelatedFiles(currentEditingSession.chatSessionId, widget.getInput(), CancellationToken.None)
			.then((files) => {
				if (!files?.length) {
					return;
				}

				const currentEditingSession = this.chatEditingService.globalEditingSessionObs.get();
				if (!currentEditingSession || currentEditingSession.chatSessionId !== widget.viewModel?.sessionId || currentEditingSession.entries.get().length) {
					return;
				}

				// Block Logic: Result set pruning and transformation.
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

				// Block Logic: Delta application to the working set.
				const existingSuggestedEntriesToRemove: URI[] = [];
				for (const entry of currentEditingSession.workingSet) {
					if (entry[1].state === WorkingSetEntryState.Suggested && !newSuggestions.has(entry[0])) {
						existingSuggestedEntriesToRemove.push(entry[0]);
					}
				}
				currentEditingSession?.remove(WorkingSetEntryRemovalReason.Programmatic, ...existingSuggestedEntriesToRemove);

				for (const [uri, data] of newSuggestions) {
					currentEditingSession.addFileToWorkingSet(uri, localize('relatedFile', "{0} (Suggested)", data.description), WorkingSetEntryState.Suggested);
				}
			})
			.finally(() => {
				this._currentRelatedFilesRetrievalOperation = undefined;
			});

	}

	/**
	 * _handleNewEditingSession - Binds session-specific observers.
	 * Logic: Sets up debounced listeners (3000ms) on the editor content to trigger 
	 * re-evaluations only during typing pauses, minimizing service pressure.
	 */
	private _handleNewEditingSession(currentEditingSession: IChatEditingSession) {
		const widget = this.chatWidgetService.getWidgetBySessionId(currentEditingSession.chatSessionId);
		if (!widget || widget.viewModel?.sessionId !== currentEditingSession.chatSessionId) {
			return;
		}

		this.chatEditingSessionDisposables.add(currentEditingSession.onDidDispose(() => {
			this.chatEditingSessionDisposables.clear();
		}));

		this._updateRelatedFileSuggestions();

		const onDebouncedType = Event.debounce(widget.inputEditor.onDidChangeModelContent, () => null, 3000);
		this.chatEditingSessionDisposables.add(onDebouncedType(() => {
			this._updateRelatedFileSuggestions();
		}));

		this.chatEditingSessionDisposables.add(currentEditingSession.onDidChange((e) => {
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
