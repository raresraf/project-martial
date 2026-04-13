/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/


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
import { IFileMatch, IPatternInfo, ISearchComplete, ISearchConfigurationProperties, ISearchProgressItem, ISearchService, ITextQuery, ITextSearchStats, QueryType, SearchCompletionExitCode } from '../../../../services/search/common/search.js';
import { IChangeEvent, mergeSearchResultEvents, SearchModelLocation, ISearchModel, ISearchResult, SEARCH_MODEL_PREFIX } from './searchTreeCommon.js';
import { SearchResultImpl } from './searchResult.js';
import { ISearchViewModelWorkbenchService } from './searchViewModelWorkbenchService.js';

/**
 * The antechamber to the search view model. The Search Model is responsible for sending search requests and receiving results.
 * It is also responsible for managing the search result tree, and updating it when results come in.
 * It also handles the replace functionality.
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
	readonly onReplaceTermChanged: Event<void> = this._onReplaceTermChanged.event;

	private readonly _onSearchResultChanged = this._register(new PauseableEmitter<IChangeEvent>({
		merge: mergeSearchResultEvents
	}));
	readonly onSearchResultChanged: Event<IChangeEvent> = this._onSearchResultChanged.event;

	private currentCancelTokenSource: CancellationTokenSource | null = null;
	private currentAICancelTokenSource: CancellationTokenSource | null = null;
	private searchCancelledForNewSearch: boolean = false;
	private aiSearchCancelledForNewSearch: boolean = false;
	public location: SearchModelLocation = SearchModelLocation.PANEL;
	private readonly _aiTextResultProviderName: Lazy<Promise<string | undefined>>;

	private readonly _id: string;

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

	id(): string {
		return this._id;
	}

	/**
	 * Fetches the name of the AI provider that is currently being used for search.
	 */
	async getAITextResultProviderName(): Promise<string> {
		const result = await this._aiTextResultProviderName.value;
		if (!result) {
			throw Error('Fetching AI name when no provider present.');
		}
		return result;
	}

	isReplaceActive(): boolean {
		return this._replaceActive;
	}

	set replaceActive(replaceActive: boolean) {
		this._replaceActive = replaceActive;
	}

	get replacePattern(): ReplacePattern | null {
		return this._replacePattern;
	}

	get replaceString(): string {
		return this._replaceString || '';
	}

	set preserveCase(value: boolean) {
		this._preserveCase = value;
	}

	get preserveCase(): boolean {
		return this._preserveCase;
	}

	set replaceString(replaceString: string) {
		this._replaceString = replaceString;
		if (this._searchQuery) {
			this._replacePattern = new ReplacePattern(replaceString, this._searchQuery.contentPattern);
		}
		this._onReplaceTermChanged.fire();
	}

	get searchResult(): ISearchResult {
		return this._searchResult;
	}

	/**
	 * Runs a new AI-powered search.
	 * @throws If an AI search is already running or has already completed.
	 * @returns A promise that resolves to the search completion details.
	 */
	aiSearch(): Promise<ISearchComplete> {
		if (this.hasAIResults) {
			// already has matches or pending matches
			throw Error('AI results already exist');
		}
		if (!this._searchQuery) {
			throw Error('No search query');
		}

		const searchInstanceID = Date.now().toString();
		const tokenSource = new CancellationTokenSource();
		this.currentAICancelTokenSource = tokenSource;
		const start = Date.now();
		// Functional Utility: Initiates the asynchronous AI text search and handles progress, completion, and error events.
		const asyncAIResults = this.searchService.aiTextSearch(
			{ ...this._searchQuery, contentPattern: this._searchQuery.contentPattern.pattern, type: QueryType.aiText },
			tokenSource.token,
			async (p: ISearchProgressItem) => {
				this.onSearchProgress(p, searchInstanceID, false, true);
			}).finally(() => {
				tokenSource.dispose(true);
			}).then(
				value => {
					this.onSearchCompleted(value, Date.now() - start, searchInstanceID, true);
					return value;
				},
				e => {
					this.onSearchError(e, Date.now() - start, true);
					throw e;
				});
		return asyncAIResults;
	}

	/**
	 * The core search execution logic. It splits the search into synchronous (open files)
	 * and asynchronous (files on disk) parts for performance. It also handles notebook
	 * searches separately.
	 *
	 * @param query The search query.
	 * @param progressEmitter An emitter to signal that the first result has been found.
	 * @param onProgress A callback to report progress items.
	 * @returns An object containing both the synchronous results and a promise for the asynchronous results.
	 */
	private doSearch(query: ITextQuery, progressEmitter: Emitter<void>, searchQuery: ITextQuery, searchInstanceID: string, onProgress?: (result: ISearchProgressItem) => void, callerToken?: CancellationToken): {
		asyncResults: Promise<ISearchComplete>;
		syncResults: IFileMatch<URI>[];
	} {
		const tokenSource = this.currentCancelTokenSource = new CancellationTokenSource(callerToken);

		// Functional Utility: This function is responsible for processing and batching search progress updates.
		const asyncGenerateOnProgress = async (p: ISearchProgressItem) => {
			progressEmitter.fire();
			this.onSearchProgress(p, searchInstanceID, false, false);
			onProgress?.(p);
		};

		// Block Logic: Asynchronously search through notebooks and text files.
		const notebookResult = this.notebookSearchService.notebookSearch(query, tokenSource.token, searchInstanceID, asyncGenerateOnProgress);
		const textResult = this.searchService.textSearchSplitSyncAsync(
			searchQuery,
			tokenSource.token, asyncGenerateOnProgress,
			notebookResult.openFilesToScan,
			notebookResult.allScannedFiles,
		);

		// Block Logic: Process synchronous results immediately.
		const syncResults = textResult.syncResults.results;
		syncResults.forEach(p => { if (p) { this.onSearchProgress(p, searchInstanceID, true); onProgress?.(p); } });

		// Functional Utility: A promise that aggregates results from asynchronous sources (closed text files and notebooks).
		const getAsyncResults = async (): Promise<ISearchComplete> => {
			const searchStart = Date.now();

			// resolve async parts of search
			const allClosedEditorResults = await textResult.asyncResults;
			const resolvedNotebookResults = await notebookResult.completeData;
			const searchLength = Date.now() - searchStart;
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

	get hasAIResults(): boolean {
		return !!(this.searchResult.getCachedSearchComplete(true)) || (!!this.currentAICancelTokenSource && !this.currentAICancelTokenSource.token.isCancellationRequested);
	}

	get hasPlainResults(): boolean {
		return !!(this.searchResult.getCachedSearchComplete(false)) || (!!this.currentCancelTokenSource && !this.currentCancelTokenSource.token.isCancellationRequested);
	}

	/**
	 * The main entry point for running a search. It cancels any previous search,
	 * sets up the new query, and orchestrates the synchronous and asynchronous
	 * parts of the search execution.
	 *
	 * @param query The text query to run.
	 * @param onProgress An optional callback to be invoked for each search result.
	 * @param callerToken An optional cancellation token from the caller.
	 * @returns An object containing synchronous results and a promise for the full completion details.
	 */
	search(query: ITextQuery, onProgress?: (result: ISearchProgressItem) => void, callerToken?: CancellationToken): {
		asyncResults: Promise<ISearchComplete>;
		syncResults: IFileMatch<URI>[];
	} {
		this.cancelSearch(true);

		this._searchQuery = query;
		if (!this.searchConfig.searchOnType) {
			this.searchResult.clear();
		}
		const searchInstanceID = Date.now().toString();

		this._searchResult.query = this._searchQuery;

		const progressEmitter = this._register(new Emitter<void>());
		this._replacePattern = new ReplacePattern(this.replaceString, this._searchQuery.contentPattern);

		// In search on type case, delay the streaming of results just a bit, so that we don't flash the only "local results" fast path
		this._startStreamDelay = new Promise(resolve => setTimeout(resolve, this.searchConfig.searchOnType ? 150 : 0));

		const req = this.doSearch(query, progressEmitter, this._searchQuery, searchInstanceID, onProgress, callerToken);
		const asyncResults = req.asyncResults;
		const syncResults = req.syncResults;

		if (onProgress) {
			syncResults.forEach(p => { if (p) { onProgress(p); } });
		}

		const start = Date.now();
		let event: IDisposable | undefined;

		// Telemetry: Log the time to the first result.
		const progressEmitterPromise = new Promise(resolve => {
			event = Event.once(progressEmitter.event)(resolve);
			return event;
		});

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
			// Functional Utility: Return a promise that resolves with the final search results, and handle completion and error logging.
			return {
				asyncResults: asyncResults.then(
					value => {
						this.onSearchCompleted(value, Date.now() - start, searchInstanceID, false);
						return value;
					},
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
	 * Called when a search completes. It adds any remaining buffered results to the
	 * search result tree and logs telemetry data.
	 */
	private onSearchCompleted(completed: ISearchComplete | undefined, duration: number, searchInstanceID: string, ai: boolean): ISearchComplete | undefined {
		if (!this._searchQuery) {
			throw new Error('onSearchCompleted must be called after a search is started');
		}

		const queue = ai ? this._aiResultQueue : this._resultQueue;
		this._searchResult.add(queue, searchInstanceID, ai);
		queue.length = 0;

		this.searchResult.setCachedSearchComplete(completed, ai);

		// ... Telemetry logging ...
		return completed;
	}

	/**
	 * Handles errors that occur during a search, particularly cancellation errors.
	 */
	private onSearchError(e: any, duration: number, ai: boolean): void {
		if (errors.isCancellationError(e)) {
			this.onSearchCompleted(
				(ai ? this.aiSearchCancelledForNewSearch : this.searchCancelledForNewSearch)
					? { exit: SearchCompletionExitCode.NewSearchStarted, results: [], messages: [] }
					: undefined,
				duration, '', ai);
			if (ai) {
				this.aiSearchCancelledForNewSearch = false;
			} else {
				this.searchCancelledForNewSearch = false;
			}
		}
	}

	/**
	 * Processes a search progress item. For performance, results are buffered and
	 * added to the search result tree in batches, especially for asynchronous results.
	 */
	private onSearchProgress(p: ISearchProgressItem, searchInstanceID: string, sync = true, ai: boolean = false) {
		const targetQueue = ai ? this._aiResultQueue : this._resultQueue;
		if ((<IFileMatch>p).resource) {
			targetQueue.push(<IFileMatch>p);
			if (sync) {
				// Sync results are added immediately.
				this._searchResult.add(targetQueue, searchInstanceID, false, true);
				targetQueue.length = 0;
			} else {
				// Async results are debounced slightly to avoid UI flicker.
				this._startStreamDelay.then(() => {
					if (targetQueue.length) {
						this._searchResult.add(targetQueue, searchInstanceID, ai, false);
						targetQueue.length = 0;
					}
				});
			}

		}
	}

	private get searchConfig() {
		return this.configurationService.getValue<ISearchConfigurationProperties>('search');
	}

	/**
	 * Cancels the current text search in progress.
	 * @param cancelledForNewSearch A flag indicating if the cancellation is due to a new search starting.
	 * @returns True if a search was cancelled, false otherwise.
	 */
	cancelSearch(cancelledForNewSearch = false): boolean {
		if (this.currentCancelTokenSource) {
			this.searchCancelledForNewSearch = cancelledForNewSearch;
			this.currentCancelTokenSource.cancel();
			return true;
		}
		return false;
	}
	/**
	 * Cancels the current AI search in progress.
	 * @param cancelledForNewSearch A flag indicating if the cancellation is due to a new search starting.
	 * @returns True if a search was cancelled, false otherwise.
	 */
	cancelAISearch(cancelledForNewSearch = false): boolean {
		if (this.currentAICancelTokenSource) {
			this.aiSearchCancelledForNewSearch = cancelledForNewSearch;
			this.currentAICancelTokenSource.cancel();
			return true;
		}
		return false;
	}
	clearAiSearchResults(): void {
		this._aiResultQueue.length = 0;
		// it's not clear all as we are only clearing the AI results
		this._searchResult.aiTextSearchResult.clear(false);
	}
	override dispose(): void {
		this.cancelSearch();
		this.cancelAISearch();
		this.searchResult.dispose();
		super.dispose();
	}

}

/**
 * A workbench service that provides access to the single instance of the search model.
 * This ensures that all parts of the workbench UI are interacting with the same
 * search state and results.
 */
export class SearchViewModelWorkbenchService implements ISearchViewModelWorkbenchService {

	declare readonly _serviceBrand: undefined;
	private _searchModel: SearchModelImpl | null = null;

	constructor(@IInstantiationService private readonly instantiationService: IInstantiationService) {
	}

	get searchModel(): SearchModelImpl {
		if (!this._searchModel) {
			this._searchModel = this.instantiationService.createInstance(SearchModelImpl);
		}
		return this._searchModel;
	}

	set searchModel(searchModel: SearchModelImpl) {
		this._searchModel?.dispose();
		this._searchModel = searchModel;
	}
}
