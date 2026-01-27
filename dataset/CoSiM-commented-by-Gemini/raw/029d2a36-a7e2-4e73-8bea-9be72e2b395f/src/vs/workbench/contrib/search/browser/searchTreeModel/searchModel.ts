/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file searchModel.ts
 * @brief Implements the core search model for the VS Code workbench.
 *
 * This file defines the `SearchModelImpl` class, which manages search queries,
 * results, and replacement operations. It interacts with the `ISearchService`
 * for actual search execution and `ITelemetryService` for logging.
 * The `SearchViewModelWorkbenchService` provides a singleton instance of the
 * `SearchModelImpl`.
 *
 * Key Components:
 * - `SearchModelImpl`: Manages search state, initiates searches, processes results,
 *   and handles cancellation. Supports both regular and AI-powered searches.
 * - `SearchResultImpl`: Stores and organizes the search results.
 * - `ReplacePattern`: Utility for handling find/replace patterns.
 *
 * Functional Utility:
 * - Provides a clear separation of concerns for managing search UI state and business logic.
 * - Supports asynchronous search operations with cancellation.
 * - Integrates with telemetry for performance and feature insight.
 * - Handles synchronization of search results between UI and background tasks.
 *
 * Algorithms:
 * - `doSearch`: Orchestrates both synchronous and asynchronous search result processing.
 * - `onSearchProgress`: Efficiently batches and adds search results to the `SearchResultImpl`.
 *
 * Time Complexity:
 * - Search operations are delegated to `ISearchService` and `INotebookSearchService`,
 *   their complexity depends on the underlying implementations.
 * - Processing results in `onSearchProgress` is generally efficient (amortized O(1) for adding to queue,
 *   then O(k) for adding to `SearchResultImpl` where k is number of matches in a batch).
 *
 * Space Complexity:
 * - Primarily depends on the number and size of search results stored in `_searchResult` and internal queues.
 */

import { CancellationToken, CancellationTokenSource } from '../../../../../base/common/cancellation.js';
import * as errors from '../../../../../base/common/errors.js';
import { Emitter, Event, PauseableEmitter } from '../../../../../base/common/event.js';
import { Lazy } from '../../../../../base/common/lazy.js';
import { Disposable, IDisposable } from '../../../../../base/common/lifecycle.js';
import { Schemas } from '../../../../../base/common/network.js';
import { URI } from '../../../../../base/common/uri.js';
import { IConfigurationService } from '../../../../../platform/configuration/common/configuration.js';
import { IInstantiationService } from '../../../../../platform/instantiation/common/instantiation.js';
import { ILogService } from '../../../../../platform/log/common/log.js';
import { ITelemetryService } from '../../../../../platform/telemetry/common/telemetry.js';
import { INotebookSearchService } from '../../common/notebookSearch.js';
import { ReplacePattern } from '../../../../services/search/common/replace.js';
import { IFileMatch, IPatternInfo, ISearchConfigurationProperties, ISearchComplete, ISearchProgressItem, ISearchService, ITextQuery, ITextSearchStats, QueryType, SearchCompletionExitCode } from '../../../../services/search/common/search.js';
import { IChangeEvent, mergeSearchResultEvents, SearchModelLocation, ISearchModel, ISearchResult, SEARCH_MODEL_PREFIX } from './searchTreeCommon.js';
import { SearchResultImpl } from './searchResult.js';
import { ISearchViewModelWorkbenchService } from './searchViewModelWorkbenchService.js';

/**
 * @class SearchModelImpl
 * @extends Disposable
 * @implements ISearchModel
 * @brief Implements the core logic for managing search operations and results within the workbench.
 *
 * This class orchestrates search queries, processes results, handles replacement logic,
 * and manages the lifecycle of search operations including cancellation. It supports
 * both traditional text search and AI-powered text search.
 */
export class SearchModelImpl extends Disposable implements ISearchModel {

	private _searchResult: ISearchResult;
	private _searchQuery: ITextQuery | null = null;
	private _replaceActive: boolean = false;
	private _replaceString: string | null = null;
	private _replacePattern: ReplacePattern | null = null;
	private _preserveCase: boolean = false;
	private _startStreamDelay: Promise<void> = Promise.resolve();
	private readonly _resultQueue: IFileMatch[] = [];
	private readonly _aiResultQueue: IFileMatch[] = [];

	private readonly _onReplaceTermChanged: Emitter<void> = this._register(new Emitter<void>());
	/**
	 * @event onReplaceTermChanged
	 * @brief An event that fires when the replace term changes.
	 */
	readonly onReplaceTermChanged: Event<void> = this._onReplaceTermChanged.event;

	private readonly _onSearchResultChanged = this._register(new PauseableEmitter<IChangeEvent>({
		merge: mergeSearchResultEvents
	}));
	/**
	 * @event onSearchResultChanged
	 * @brief An event that fires when search results change (e.g., new results are added, results are cleared).
	 */
	readonly onSearchResultChanged: Event<IChangeEvent> = this._onSearchResultChanged.event;

	private currentCancelTokenSource: CancellationTokenSource | null = null;
	private currentAICancelTokenSource: CancellationTokenSource | null = null;
	private searchCancelledForNewSearch: boolean = false;
	private aiSearchCancelledForNewSearch: boolean = false;
	/**
	 * @property location
	 * @brief The location of the search model (e.g., in a panel).
	 */
	public location: SearchModelLocation = SearchModelLocation.PANEL;
	private readonly _aiTextResultProviderName: Lazy<Promise<string | undefined>>;

	private readonly _id: string;

	/**
	 * @brief Constructs a new `SearchModelImpl` instance.
	 * @param searchService The search service to use for executing searches.
	 * @param telemetryService The telemetry service for logging events.
	 * @param configurationService The configuration service for accessing settings.
	 * @param instantiationService The instantiation service for creating dependent objects.
	 * @param logService The log service for logging messages.
	 * @param notebookSearchService The notebook search service for notebook-specific searches.
	 */
	constructor(
		@ISearchService private readonly searchService: ISearchService,
		@ITelemetryService private readonly telemetryService: ITelemetryService,
		@IConfigurationService private readonly configurationService: IConfigurationService,
		@IInstantiationService private readonly instantiationService: IInstantiationService,
		@ILogService private readonly logService: ILogService,
		@INotebookSearchService private readonly notebookSearchService: INotebookSearchService,
	) {
		super();
		this._searchResult = this.instantiationService.createInstance(SearchResultImpl, this);
		this._register(this._searchResult.onChange((e) => this._onSearchResultChanged.fire(e)));

		this._aiTextResultProviderName = new Lazy(async () => this.searchService.getAIName());
		this._id = SEARCH_MODEL_PREFIX + Date.now().toString();
	}

	/**
	 * @brief Returns the unique identifier for this search model.
	 * @returns A string representing the ID.
	 */
	id(): string {
		return this._id;
	}

	/**
	 * @brief Retrieves the name of the AI text result provider.
	 * @returns A promise that resolves to the AI provider's name.
	 * @throws Error if no AI provider is present.
	 */
	async getAITextResultProviderName(): Promise<string> {
		const result = await this._aiTextResultProviderName.value;
		if (!result) {
			throw Error('Fetching AI name when no provider present.');
		}
		return result;
	}

	/**
	 * @brief Checks if replace functionality is currently active.
	 * @returns True if replace is active, false otherwise.
	 */
	isReplaceActive(): boolean {
		return this._replaceActive;
	}

	/**
	 * @brief Sets whether replace functionality is active.
	 */
	set replaceActive(replaceActive: boolean) {
		this._replaceActive = replaceActive;
	}

	/**
	 * @property replacePattern
	 * @brief The current replace pattern.
	 */
	get replacePattern(): ReplacePattern | null {
		return this._replacePattern;
	}

	/**
	 * @property replaceString
	 * @brief The current replace string.
	 */
	get replaceString(): string {
		return this._replaceString || '';
	}

	/**
	 * @property preserveCase
	 * @brief Whether case should be preserved during replace operations.
	 */
	set preserveCase(value: boolean) {
		this._preserveCase = value;
	}

	/**
	 * @property preserveCase
	 * @brief Whether case should be preserved during replace operations.
	 */
	get preserveCase(): boolean {
		return this._preserveCase;
	}

	/**
	 * @brief Sets the replace string and updates the replace pattern.
	 * @param replaceString The new replace string.
	 */
	set replaceString(replaceString: string) {
		this._replaceString = replaceString;
		// Block Logic: If a search query exists, create a new ReplacePattern.
		if (this._searchQuery) {
			this._replacePattern = new ReplacePattern(replaceString, this._searchQuery.contentPattern);
		}
		// Functional Utility: Fires an event to notify listeners of the change.
		this._onReplaceTermChanged.fire();
	}

	/**
	 * @property searchResult
	 * @brief The search results object managed by this model.
	 */
	get searchResult(): ISearchResult {
		return this._searchResult;
	}

	/**
	 * @brief Initiates an AI-powered search.
	 * @returns A promise that resolves to the search completion result.
	 * @throws Error if AI results already exist or no search query is set.
	 */
	aiSearch(): Promise<ISearchComplete> {
		// Block Logic: Checks if AI results already exist or are pending.
		if (this.hasAIResults) {
			// already has matches or pending matches
			throw Error('AI results already exist');
		}
		// Block Logic: Ensures a search query is set.
		if (!this._searchQuery) {
			throw Error('No search query');
		}

		const searchInstanceID = Date.now().toString();
		const tokenSource = new CancellationTokenSource();
		this.currentAICancelTokenSource = tokenSource;
		const start = Date.now();
		// Functional Utility: Delegates AI text search to the search service.
		const asyncAIResults = this.searchService.aiTextSearch(
			{ ...this._searchQuery, contentPattern: this._searchQuery.contentPattern.pattern, type: QueryType.aiText },
			tokenSource.token,
			// Functional Utility: Progress callback for AI search.
			async (p: ISearchProgressItem) => {
				this.onSearchProgress(p, searchInstanceID, false, true);
			}).finally(() => {
				// Functional Utility: Disposes the cancellation token source on completion or cancellation.
				tokenSource.dispose(true);
			}).then(
				// Functional Utility: Success callback for AI search.
				value => {
					this.onSearchCompleted(value, Date.now() - start, searchInstanceID, true);
					return value;
				},
				// Functional Utility: Error callback for AI search.
				e => {
					this.onSearchError(e, Date.now() - start, true);
					throw e;
				});
		return asyncAIResults;
	}

	/**
	 * @brief Performs the actual search operation, handling both synchronous and asynchronous results.
	 * @param query The text query to execute.
	 * @param progressEmitter An Emitter to signal progress.
	 * @param searchQuery The search query object.
	 * @param searchInstanceID A unique ID for the current search instance.
	 * @param onProgress Optional callback for progress updates.
	 * @param callerToken Optional cancellation token from the caller.
	 * @returns An object containing a promise for async results and an array of sync results.
	 */
	private doSearch(query: ITextQuery, progressEmitter: Emitter<void>, searchQuery: ITextQuery, searchInstanceID: string, onProgress?: (result: ISearchProgressItem) => void, callerToken?: CancellationToken): {
		asyncResults: Promise<ISearchComplete>;
		syncResults: IFileMatch<URI>[];
	} {
		// Functional Utility: Asynchronous progress handler.
		const asyncGenerateOnProgress = async (p: ISearchProgressItem) => {
			progressEmitter.fire();
			this.onSearchProgress(p, searchInstanceID, false, false);
			onProgress?.(p);
		};

		// Functional Utility: Synchronous progress handler.
		const syncGenerateOnProgress = (p: ISearchProgressItem) => {
			progressEmitter.fire();
			this.onSearchProgress(p, searchInstanceID, true);
			onProgress?.(p);
		};
		// Functional Utility: Creates a cancellation token source for the current search.
		const tokenSource = this.currentCancelTokenSource = new CancellationTokenSource(callerToken);

		// Functional Utility: Initiates notebook search.
		const notebookResult = this.notebookSearchService.notebookSearch(query, tokenSource.token, searchInstanceID, asyncGenerateOnProgress);
		// Functional Utility: Initiates text search, splitting into sync and async parts.
		const textResult = this.searchService.textSearchSplitSyncAsync(
			searchQuery,
			tokenSource.token, asyncGenerateOnProgress,
			notebookResult.openFilesToScan,
			notebookResult.allScannedFiles,
		);

		const syncResults = textResult.syncResults.results;
		// Functional Utility: Processes synchronous results.
		syncResults.forEach(p => { if (p) { syncGenerateOnProgress(p); } });

		// Functional Utility: Defines the asynchronous result resolution logic.
		const getAsyncResults = async (): Promise<ISearchComplete> => {
			const searchStart = Date.now();

			// resolve async parts of search
			const allClosedEditorResults = await textResult.asyncResults;
			const resolvedNotebookResults = await notebookResult.completeData;
			const searchLength = Date.now() - searchStart;
			// Functional Utility: Merges results from text search and notebook search.
			const resolvedResult: ISearchComplete = {
				results: [...allClosedEditorResults.results, ...resolvedNotebookResults.results],
				messages: [...allClosedEditorResults.messages, ...resolvedNotebookResults.messages],
				limitHit: allClosedEditorResults.limitHit || resolvedNotebookResults.limitHit,
				exit: allClosedEditorResults.exit,
				stats: allClosedEditorResults.stats,
			};
			this.logService.trace(`whole search time | ${searchLength}ms`);
			return resolvedResult;
		};
		return {
			asyncResults: getAsyncResults()
				.finally(() => tokenSource.dispose(true)),
			syncResults
		};
	}

	/**
	 * @property hasAIResults
	 * @brief Checks if there are any AI search results or pending AI searches.
	 */
	get hasAIResults(): boolean {
		return !!(this.searchResult.getCachedSearchComplete(true)) || (!!this.currentAICancelTokenSource && !this.currentAICancelTokenSource.token.isCancellationRequested);
	}

	/**
	 * @property hasPlainResults
	 * @brief Checks if there are any regular (non-AI) search results or pending regular searches.
	 */
	get hasPlainResults(): boolean {
		return !!(this.searchResult.getCachedSearchComplete(false)) || (!!this.currentCancelTokenSource && !this.currentCancelTokenSource.token.isCancellationRequested);
	}

	/**
	 * @brief Initiates a new search operation.
	 * @param query The text query to execute.
	 * @param onProgress Optional callback for progress updates.
	 * @param callerToken Optional cancellation token from the caller.
	 * @returns An object containing a promise for async results and an array of sync results.
	 */
	search(query: ITextQuery, onProgress?: (result: ISearchProgressItem) => void, callerToken?: CancellationToken): {
		asyncResults: Promise<ISearchComplete>;
		syncResults: IFileMatch<URI>[];
	} {
		// Functional Utility: Cancels any ongoing search before starting a new one.
		this.cancelSearch(true);

		this._searchQuery = query;
		// Block Logic: Clears existing search results if search-on-type is disabled.
		if (!this.searchConfig.searchOnType) {
			this.searchResult.clear();
		}
		const searchInstanceID = Date.now().toString();

		this._searchResult.query = this._searchQuery;

		const progressEmitter = this._register(new Emitter<void>());
		// Functional Utility: Creates a replace pattern based on the current search query.
		this._replacePattern = new ReplacePattern(this.replaceString, this._searchQuery.contentPattern);

		// In search on type case, delay the streaming of results just a bit, so that we don't flash the only "local results" fast path
		// Functional Utility: Introduces a slight delay for streaming results in search-on-type mode to prevent UI flicker.
		this._startStreamDelay = new Promise(resolve => setTimeout(resolve, this.searchConfig.searchOnType ? 150 : 0));

		const req = this.doSearch(query, progressEmitter, this._searchQuery, searchInstanceID, onProgress, callerToken);
		const asyncResults = req.asyncResults;
		const syncResults = req.syncResults;

		// Block Logic: If a progress callback is provided, immediately notify it of synchronous results.
		if (onProgress) {
			syncResults.forEach(p => {
				if (p) {
					onProgress(p);
				}
			});
		}

		const start = Date.now();
		let event: IDisposable | undefined;

		// Functional Utility: Sets up a promise to resolve when the first progress event is fired.
		const progressEmitterPromise = new Promise(resolve => {
			event = Event.once(progressEmitter.event)(resolve);
			return event;
		});

		// Block Logic: Races between async results completion and the first progress event for telemetry.
		Promise.race([asyncResults, progressEmitterPromise]).finally(() => {
			/* __GDPR__
				"searchResultsFirstRender" : {
					"owner": "roblourens",
					"duration" : { "classification": "SystemMetaData", "purpose": "PerformanceAndHealth", "isMeasurement": true }
				}
			*/
			event?.dispose();
			this.telemetryService.publicLog('searchResultsFirstRender', { duration: Date.now() - start });
		});

		try {
			return {
				asyncResults: asyncResults.then(
					// Functional Utility: Success callback for search completion.
					value => {
						this.onSearchCompleted(value, Date.now() - start, searchInstanceID, false);
						return value;
					},
					// Functional Utility: Error callback for search.
					e => {
						this.onSearchError(e, Date.now() - start, false);
						throw e;
					}),
				syncResults
			};
		} finally {
			/* __GDPR__
				"searchResultsFinished" : {
					"owner": "roblourens",
					"duration" : { "classification": "SystemMetaData", "purpose": "PerformanceAndHealth", "isMeasurement": true }
				}
			*/
			this.telemetryService.publicLog('searchResultsFinished', { duration: Date.now() - start });
		}
	}

	/**
	 * @brief Handles the completion of a search operation.
	 * @param completed The completed search result object, or undefined if cancelled.
	 * @param duration The duration of the search in milliseconds.
	 * @param searchInstanceID The unique ID of the search instance.
	 * @param ai True if this is an AI search, false otherwise.
	 * @returns The completed search result object.
	 * @throws Error if `onSearchCompleted` is called before a search is started.
	 */
	private onSearchCompleted(completed: ISearchComplete | undefined, duration: number, searchInstanceID: string, ai: boolean): ISearchComplete | undefined {
		// Block Logic: Ensures a search query is set.
		if (!this._searchQuery) {
			throw new Error('onSearchCompleted must be called after a search is started');
		}

		// Block Logic: Adds results from the appropriate queue to the search result object.
		if (ai) {
			this._searchResult.add(this._aiResultQueue, searchInstanceID, true);
			this._aiResultQueue.length = 0;
		} else {
			this._searchResult.add(this._resultQueue, searchInstanceID, false);
			this._resultQueue.length = 0;
		}

		// Functional Utility: Caches the completed search results.
		this.searchResult.setCachedSearchComplete(completed, ai);

		const options: IPatternInfo = Object.assign({}, this._searchQuery.contentPattern);
		delete (options as any).pattern; // Functional Utility: Removes the pattern from options for telemetry.

		const stats = completed && completed.stats as ITextSearchStats;

		// Functional Utility: Determines the scheme type for telemetry.
		const fileSchemeOnly = this._searchQuery.folderQueries.every(fq => fq.folder.scheme === Schemas.file);
		const otherSchemeOnly = this._searchQuery.folderQueries.every(fq => fq.folder.scheme !== Schemas.file);
		const scheme = fileSchemeOnly ? Schemas.file :
			otherSchemeOnly ? 'other' :
				'mixed';

		/* __GDPR__
			"searchResultsShown" : {
				"owner": "roblourens",
				"count" : { "classification": "SystemMetaData", "purpose": "FeatureInsight", "isMeasurement": true },
				"fileCount": { "classification": "SystemMetaData", "purpose": "FeatureInsight", "isMeasurement": true },
				"options": { "${inline}": [ "${IPatternInfo}" ] },
				"duration": { "classification": "SystemMetaData", "purpose": "PerformanceAndHealth", "isMeasurement": true },
				"type" : { "classification": "SystemMetaData", "purpose": "PerformanceAndHealth" },
				"scheme" : { "classification": "SystemMetaData", "purpose": "PerformanceAndHealth" },
				"searchOnTypeEnabled" : { "classification": "SystemMetaData", "purpose": "FeatureInsight" }
			}
		*/
		// Functional Utility: Logs search results telemetry.
		this.telemetryService.publicLog('searchResultsShown', {
			count: this._searchResult.count(),
			fileCount: this._searchResult.fileCount(),
			options,
			duration,
			type: stats && stats.type,
			scheme,
			searchOnTypeEnabled: this.searchConfig.searchOnType
		});
		return completed;
	}

	/**
	 * @brief Handles errors that occur during a search operation.
	 * @param e The error object.
	 * @param duration The duration of the search up to the error.
	 * @param ai True if this is an AI search, false otherwise.
	 */
	private onSearchError(e: any, duration: number, ai: boolean): void {
		// Block Logic: If the error is a cancellation error, handles it by calling `onSearchCompleted` with a cancellation exit code.
		if (errors.isCancellationError(e)) {
			this.onSearchCompleted(
				(ai ? this.aiSearchCancelledForNewSearch : this.searchCancelledForNewSearch)
					? { exit: SearchCompletionExitCode.NewSearchStarted, results: [], messages: [] }
					: undefined,
				duration, '', ai);
			// Functional Utility: Resets the cancellation flag.
			if (ai) {
				this.aiSearchCancelledForNewSearch = false;
			} else {
				this.searchCancelledForNewSearch = false;
			}
		}
	}

	/**
	 * @brief Processes search progress items, adding them to the appropriate result queue.
	 * @param p The search progress item.
	 * @param searchInstanceID The unique ID of the search instance.
	 * @param sync True if the progress is from a synchronous search, false otherwise.
	 * @param ai True if the progress is from an AI search, false otherwise.
	 */
	private onSearchProgress(p: ISearchProgressItem, searchInstanceID: string, sync = true, ai: boolean = false) {
		const targetQueue = ai ? this._aiResultQueue : this._resultQueue;
		// Block Logic: If the progress item is a file match, adds it to the target queue.
		if ((<IFileMatch>p).resource) {
			targetQueue.push(<IFileMatch>p);
			// Block Logic: If synchronous, immediately adds queued results to the search result.
			if (sync) {
				if (targetQueue.length) {
					this._searchResult.add(targetQueue, searchInstanceID, false, true);
					targetQueue.length = 0;
				}
			} else {
				// Block Logic: For asynchronous results, waits for a delay before adding to prevent UI flashing.
				this._startStreamDelay.then(() => {
					if (targetQueue.length) {
						this._searchResult.add(targetQueue, searchInstanceID, ai, !ai);
						targetQueue.length = 0;
					}
				});
			}

		}
	}

	/**
	 * @property searchConfig
	 * @brief Retrieves the current search configuration properties.
	 */
	private get searchConfig() {
		return this.configurationService.getValue<ISearchConfigurationProperties>('search');
	}

	/**
	 * @brief Cancels the current ongoing search (non-AI).
	 * @param cancelledForNewSearch True if the cancellation is due to a new search starting, false otherwise.
	 * @returns True if a search was cancelled, false otherwise.
	 */
	cancelSearch(cancelledForNewSearch = false): boolean {
		// Block Logic: If a cancellation token source exists, cancels the search.
		if (this.currentCancelTokenSource) {
			this.searchCancelledForNewSearch = cancelledForNewSearch;
			this.currentCancelTokenSource.cancel();
			return true;
		}
		return false;
	}

	/**
	 * @brief Cancels the current ongoing AI search.
	 * @param cancelledForNewSearch True if the cancellation is due to a new search starting, false otherwise.
	 * @returns True if an AI search was cancelled, false otherwise.
	 */
	cancelAISearch(cancelledForNewSearch = false): boolean {
		// Block Logic: If an AI cancellation token source exists, cancels the AI search.
		if (this.currentAICancelTokenSource) {
			this.aiSearchCancelledForNewSearch = cancelledForNewSearch;
			this.currentAICancelTokenSource.cancel();
			return true;
		}
		return false;
	}

	/**
	 * @brief Clears all AI search results.
	 */
	clearAiSearchResults(): void {
		this._aiResultQueue.length = 0;
		// it's not clear all as we are only clearing the AI results
		this._searchResult.aiTextSearchResult.clear(false);
	}

	/**
	 * @brief Disposes of the search model, canceling any active searches and releasing resources.
	 */
	override dispose(): void {
		this.cancelSearch();
		this.cancelAISearch();
		this.searchResult.dispose();
		super.dispose();
	}

}

/**
 * @class SearchViewModelWorkbenchService
 * @implements ISearchViewModelWorkbenchService
 * @brief Provides access to the singleton `SearchModelImpl` instance for the workbench.
 *
 * This service ensures that there is a single instance of the `SearchModelImpl`
 * managing search operations across the workbench.
 */
export class SearchViewModelWorkbenchService implements ISearchViewModelWorkbenchService {

	declare readonly _serviceBrand: undefined;
	private _searchModel: SearchModelImpl | null = null;

	/**
	 * @brief Constructs a new `SearchViewModelWorkbenchService` instance.
	 * @param instantiationService The instantiation service for creating the `SearchModelImpl`.
	 */
	constructor(@IInstantiationService private readonly instantiationService: IInstantiationService) {
	}

	/**
	 * @property searchModel
	 * @brief The singleton instance of `SearchModelImpl`. Creates one if it doesn't exist.
	 */
	get searchModel(): SearchModelImpl {
		// Block Logic: Lazily creates the SearchModelImpl instance if it does not already exist.
		if (!this._searchModel) {
			this._searchModel = this.instantiationService.createInstance(SearchModelImpl);
		}
		return this._searchModel;
	}

	/**
	 * @property searchModel
	 * @brief Sets the current `SearchModelImpl` instance, disposing of the old one if it exists.
	 */
	set searchModel(searchModel: SearchModelImpl) {
		// Block Logic: Disposes of the existing search model before assigning a new one.
		this._searchModel?.dispose();
		this._searchModel = searchModel;
	}
}