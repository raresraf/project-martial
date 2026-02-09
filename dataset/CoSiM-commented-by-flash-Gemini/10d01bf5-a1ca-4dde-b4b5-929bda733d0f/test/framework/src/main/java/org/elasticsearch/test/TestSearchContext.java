/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.test;

import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TotalHits;
import org.elasticsearch.action.search.SearchType;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.index.IndexService;
import org.elasticsearch.index.cache.bitset.BitsetFilterCache;
import org.elasticsearch.index.mapper.IdLoader;
import org.elasticsearch.index.mapper.SourceLoader;
import org.elasticsearch.index.query.ParsedQuery;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.index.shard.IndexShard;
import org.elasticsearch.index.shard.ShardId;
import org.elasticsearch.search.SearchExtBuilder;
import org.elasticsearch.search.SearchShardTarget;
import org.elasticsearch.search.aggregations.SearchContextAggregations;
import org.elasticsearch.search.collapse.CollapseContext;
import org.elasticsearch.search.dfs.DfsSearchResult;
import org.elasticsearch.search.fetch.FetchPhase;
import org.elasticsearch.search.fetch.FetchSearchResult;
import org.elasticsearch.search.fetch.StoredFieldsContext;
import org.elasticsearch.search.fetch.subphase.FetchDocValuesContext;
import org.elasticsearch.search.fetch.subphase.FetchFieldsContext;
import org.elasticsearch.search.fetch.subphase.FetchSourceContext;
import org.elasticsearch.search.fetch.subphase.ScriptFieldsContext;
import org.elasticsearch.search.fetch.subphase.highlight.SearchHighlightContext;
import org.elasticsearch.search.internal.AliasFilter;
import org.elasticsearch.search.internal.ContextIndexSearcher;
import org.elasticsearch.search.internal.ReaderContext;
import org.elasticsearch.search.internal.ScrollContext;
import org.elasticsearch.search.internal.SearchContext;
import org.elasticsearch.search.internal.ShardSearchContextId;
import org.elasticsearch.search.internal.ShardSearchRequest;
import org.elasticsearch.search.profile.Profilers;
import org.elasticsearch.search.query.QuerySearchResult;
import org.elasticsearch.search.rank.context.QueryPhaseRankShardContext;
import org.elasticsearch.search.rank.feature.RankFeatureResult;
import org.elasticsearch.search.rescore.RescoreContext;
import org.elasticsearch.search.sort.SortAndFormats;
import org.elasticsearch.search.suggest.SuggestionSearchContext;
import org.elasticsearch.tasks.CancellableTask;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.util.Collections.emptyMap;

/**
 * @brief Functional description of the TestSearchContext class.
 *        This is a placeholder for detailed semantic documentation.
 *        Further analysis will elaborate on its algorithm, complexity, and invariants.
 */
public class TestSearchContext extends SearchContext {
    final IndexService indexService;
    final BitsetFilterCache fixedBitSetFilterCache;
    final IndexShard indexShard;
    final QuerySearchResult queryResult = new QuerySearchResult();
    final SearchExecutionContext searchExecutionContext;
    ParsedQuery originalQuery;
    ParsedQuery postFilter;
    Query query;
    Float minScore;
    CancellableTask task;
    SortAndFormats sort;
    boolean trackScores = false;
    int trackTotalHitsUpTo = SearchContext.DEFAULT_TRACK_TOTAL_HITS_UP_TO;
    QueryPhaseRankShardContext queryPhaseRankShardContext;
    ContextIndexSearcher searcher;
    int from;
    int size;
    private int terminateAfter = DEFAULT_TERMINATE_AFTER;
    private SearchContextAggregations aggregations;
    private ScrollContext scrollContext;
    private FieldDoc searchAfter;
    private final ShardSearchRequest request;

    private final Map<String, SearchExtBuilder> searchExtBuilders = new HashMap<>();

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param indexService: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public TestSearchContext(IndexService indexService) {
        this.indexService = indexService;
        this.fixedBitSetFilterCache = indexService.cache().bitsetFilterCache();
        this.indexShard = indexService.getShardOrNull(0);
        searchExecutionContext = indexService.newSearchExecutionContext(0, 0, null, () -> 0L, null, emptyMap());
        this.request = new ShardSearchRequest(indexShard.shardId(), 0L, AliasFilter.EMPTY);
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param searchExecutionContext: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public TestSearchContext(SearchExecutionContext searchExecutionContext) {
        this(searchExecutionContext, null, null, null);
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param searchExecutionContext: [Description]
     * @param indexShard: [Description]
     * @param searcher: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public TestSearchContext(SearchExecutionContext searchExecutionContext, IndexShard indexShard, ContextIndexSearcher searcher) {
        this(searchExecutionContext, indexShard, searcher, null);
    }

    public TestSearchContext(
        SearchExecutionContext searchExecutionContext,
        IndexShard indexShard,
        ContextIndexSearcher searcher,
        ScrollContext scrollContext
    ) {
        this.indexService = null;
        this.fixedBitSetFilterCache = null;
        this.indexShard = indexShard;
        this.searchExecutionContext = searchExecutionContext;
        this.searcher = searcher;
        this.scrollContext = scrollContext;
        ShardId shardId = indexShard != null ? indexShard.shardId() : new ShardId("N/A", "N/A", 0);
        this.request = new ShardSearchRequest(shardId, 0L, AliasFilter.EMPTY);
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param searcher: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void setSearcher(ContextIndexSearcher searcher) {
        this.searcher = searcher;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void preProcess() {}

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param q: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Query buildFilteredQuery(Query q) {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ShardSearchContextId id() {
        return new ShardSearchContextId("", 0);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public String source() {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ShardSearchRequest request() {
        return request;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchType searchType() {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchShardTarget shardTarget() {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public int numberOfShards() {
        return 1;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ScrollContext scrollContext() {
        return scrollContext;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContextAggregations aggregations() {
        return aggregations;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param searchContextAggregations: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext aggregations(SearchContextAggregations searchContextAggregations) {
        this.aggregations = searchContextAggregations;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param name: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchExtBuilder getSearchExt(String name) {
        return searchExtBuilders.get(name);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchHighlightContext highlight() {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param highlight: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void highlight(SearchHighlightContext highlight) {}

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SuggestionSearchContext suggest() {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public List<RescoreContext> rescore() {
        return Collections.emptyList();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean hasScriptFields() {
        return false;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ScriptFieldsContext scriptFields() {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean sourceRequested() {
        return false;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public FetchSourceContext fetchSourceContext() {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param fetchSourceContext: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext fetchSourceContext(FetchSourceContext fetchSourceContext) {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public FetchDocValuesContext docValuesContext() {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param docValuesContext: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext docValuesContext(FetchDocValuesContext docValuesContext) {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public FetchFieldsContext fetchFieldsContext() {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param fetchFieldsContext: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext fetchFieldsContext(FetchFieldsContext fetchFieldsContext) {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ContextIndexSearcher searcher() {
        return searcher;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public IndexShard indexShard() {
        return indexShard;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public BitsetFilterCache bitsetFilterCache() {
        return fixedBitSetFilterCache;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public TimeValue timeout() {
        return TimeValue.ZERO;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public int terminateAfter() {
        return terminateAfter;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param terminateAfter: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void terminateAfter(int terminateAfter) {
        this.terminateAfter = terminateAfter;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean lowLevelCancellation() {
        return false;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param minimumScore: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext minimumScore(float minimumScore) {
        this.minScore = minimumScore;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Float minimumScore() {
        return minScore;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param sortAndFormats: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext sort(SortAndFormats sortAndFormats) {
        this.sort = sortAndFormats;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SortAndFormats sort() {
        return sort;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param shouldTrackScores: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext trackScores(boolean shouldTrackScores) {
        this.trackScores = shouldTrackScores;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean trackScores() {
        return trackScores;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param trackTotalHitsUpToValue: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext trackTotalHitsUpTo(int trackTotalHitsUpToValue) {
        this.trackTotalHitsUpTo = trackTotalHitsUpToValue;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public int trackTotalHitsUpTo() {
        return trackTotalHitsUpTo;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param searchAfterDoc: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext searchAfter(FieldDoc searchAfterDoc) {
        this.searchAfter = searchAfterDoc;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public FieldDoc searchAfter() {
        return searchAfter;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public CollapseContext collapse() {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param postFilterQuery: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext parsedPostFilter(ParsedQuery postFilterQuery) {
        this.postFilter = postFilterQuery;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ParsedQuery parsedPostFilter() {
        return postFilter;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param parsedQuery: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext parsedQuery(ParsedQuery parsedQuery) {
        this.originalQuery = parsedQuery;
        this.query = parsedQuery.query();
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ParsedQuery parsedQuery() {
        return originalQuery;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Query query() {
        return query;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public int from() {
        return from;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param fromValue: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext from(int fromValue) {
        this.from = fromValue;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public int size() {
        return size;
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param size: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void setSize(int size) {
        this.size = size;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param sizeValue: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext size(int sizeValue) {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean hasStoredFields() {
        return false;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public StoredFieldsContext storedFieldsContext() {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param storedFieldsContext: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext storedFieldsContext(StoredFieldsContext storedFieldsContext) {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean explain() {
        return false;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param explain: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void explain(boolean explain) {}

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public List<String> groupStats() {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean version() {
        return false;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param version: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void version(boolean version) {}

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean seqNoAndPrimaryTerm() {
        return false;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param seqNoAndPrimaryTerm: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void seqNoAndPrimaryTerm(boolean seqNoAndPrimaryTerm) {

    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public DfsSearchResult dfsResult() {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addDfsResult() {
        // this space intentionally left blank
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public QuerySearchResult queryResult() {
        return queryResult;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addQueryResult() {
        // this space intentionally left blank
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public TotalHits getTotalHits() {
        return queryResult.getTotalHits();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public float getMaxScore() {
        return queryResult.getMaxScore();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addRankFeatureResult() {
        // this space intentionally left blank
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public RankFeatureResult rankFeatureResult() {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public FetchSearchResult fetchResult() {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addFetchResult() {
        // this space intentionally left blank
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public FetchPhase fetchPhase() {
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public long getRelativeTimeInMillis() {
        return 0L;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Profilers getProfilers() {
        return null; // no profiling
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchExecutionContext getSearchExecutionContext() {
        return searchExecutionContext;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param task: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void setTask(CancellableTask task) {
        this.task = task;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public CancellableTask getTask() {
        return task;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean isCancelled() {
        return task.isCancelled();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public QueryPhaseRankShardContext queryPhaseRankShardContext() {
        return queryPhaseRankShardContext;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param queryPhaseRankContext: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void queryPhaseRankShardContext(QueryPhaseRankShardContext queryPhaseRankContext) {
        this.queryPhaseRankShardContext = queryPhaseRankContext;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param rescore: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addRescore(RescoreContext rescore) {

    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ReaderContext readerContext() {
        throw new UnsupportedOperationException();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SourceLoader newSourceLoader() {
        return searchExecutionContext.newSourceLoader(false);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public IdLoader newIdLoader() {
        throw new UnsupportedOperationException();
    }
}
