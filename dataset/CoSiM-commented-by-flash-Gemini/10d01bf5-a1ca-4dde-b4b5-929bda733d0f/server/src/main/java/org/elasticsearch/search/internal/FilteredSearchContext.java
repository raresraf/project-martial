/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.internal;

import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TotalHits;
import org.elasticsearch.action.search.SearchType;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.index.cache.bitset.BitsetFilterCache;
import org.elasticsearch.index.mapper.IdLoader;
import org.elasticsearch.index.mapper.SourceLoader;
import org.elasticsearch.index.query.ParsedQuery;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.index.shard.IndexShard;
import org.elasticsearch.search.SearchExtBuilder;
import org.elasticsearch.search.SearchShardTarget;
import org.elasticsearch.search.aggregations.SearchContextAggregations;
import org.elasticsearch.search.collapse.CollapseContext;
import org.elasticsearch.search.dfs.DfsSearchResult;
import org.elasticsearch.search.fetch.FetchPhase;
import org.elasticsearch.search.fetch.FetchSearchResult;
import org.elasticsearch.search.fetch.StoredFieldsContext;
import org.elasticsearch.search.fetch.subphase.FetchSourceContext;
import org.elasticsearch.search.fetch.subphase.InnerHitsContext;
import org.elasticsearch.search.fetch.subphase.ScriptFieldsContext;
import org.elasticsearch.search.fetch.subphase.highlight.SearchHighlightContext;
import org.elasticsearch.search.profile.Profilers;
import org.elasticsearch.search.query.QuerySearchResult;
import org.elasticsearch.search.rank.context.QueryPhaseRankShardContext;
import org.elasticsearch.search.rank.feature.RankFeatureResult;
import org.elasticsearch.search.rescore.RescoreContext;
import org.elasticsearch.search.sort.SortAndFormats;
import org.elasticsearch.search.suggest.SuggestionSearchContext;
import org.elasticsearch.tasks.CancellableTask;

import java.util.List;

/**
 * @brief Functional description of the FilteredSearchContext class.
 *        This is a placeholder for detailed semantic documentation.
 *        Further analysis will elaborate on its algorithm, complexity, and invariants.
 */
public abstract class FilteredSearchContext extends SearchContext {

    private final SearchContext in;

    public FilteredSearchContext(SearchContext in) {
        this.in = in;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean hasStoredFields() {
        return in.hasStoredFields();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public StoredFieldsContext storedFieldsContext() {
        return in.storedFieldsContext();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param storedFieldsContext: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext storedFieldsContext(StoredFieldsContext storedFieldsContext) {
        return in.storedFieldsContext(storedFieldsContext);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void preProcess() {
        in.preProcess();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param query: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Query buildFilteredQuery(Query query) {
        return in.buildFilteredQuery(query);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ShardSearchContextId id() {
        return in.id();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public String source() {
        return in.source();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ShardSearchRequest request() {
        return in.request();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchType searchType() {
        return in.searchType();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchShardTarget shardTarget() {
        return in.shardTarget();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public int numberOfShards() {
        return in.numberOfShards();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ScrollContext scrollContext() {
        return in.scrollContext();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContextAggregations aggregations() {
        return in.aggregations();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param aggregations: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext aggregations(SearchContextAggregations aggregations) {
        return in.aggregations(aggregations);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchHighlightContext highlight() {
        return in.highlight();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param highlight: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void highlight(SearchHighlightContext highlight) {
        in.highlight(highlight);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public InnerHitsContext innerHits() {
        return in.innerHits();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SuggestionSearchContext suggest() {
        return in.suggest();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public QueryPhaseRankShardContext queryPhaseRankShardContext() {
        return in.queryPhaseRankShardContext();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param queryPhaseRankShardContext: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void queryPhaseRankShardContext(QueryPhaseRankShardContext queryPhaseRankShardContext) {
        in.queryPhaseRankShardContext(queryPhaseRankShardContext);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public List<RescoreContext> rescore() {
        return in.rescore();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean hasScriptFields() {
        return in.hasScriptFields();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ScriptFieldsContext scriptFields() {
        return in.scriptFields();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean sourceRequested() {
        return in.sourceRequested();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public FetchSourceContext fetchSourceContext() {
        return in.fetchSourceContext();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param fetchSourceContext: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext fetchSourceContext(FetchSourceContext fetchSourceContext) {
        return in.fetchSourceContext(fetchSourceContext);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ContextIndexSearcher searcher() {
        return in.searcher();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public IndexShard indexShard() {
        return in.indexShard();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public BitsetFilterCache bitsetFilterCache() {
        return in.bitsetFilterCache();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public TimeValue timeout() {
        return in.timeout();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public int terminateAfter() {
        return in.terminateAfter();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param terminateAfter: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void terminateAfter(int terminateAfter) {
        in.terminateAfter(terminateAfter);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean lowLevelCancellation() {
        return in.lowLevelCancellation();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param minimumScore: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext minimumScore(float minimumScore) {
        return in.minimumScore(minimumScore);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Float minimumScore() {
        return in.minimumScore();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param sort: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext sort(SortAndFormats sort) {
        return in.sort(sort);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SortAndFormats sort() {
        return in.sort();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param trackScores: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext trackScores(boolean trackScores) {
        return in.trackScores(trackScores);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean trackScores() {
        return in.trackScores();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param trackTotalHitsUpTo: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext trackTotalHitsUpTo(int trackTotalHitsUpTo) {
        return in.trackTotalHitsUpTo(trackTotalHitsUpTo);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public int trackTotalHitsUpTo() {
        return in.trackTotalHitsUpTo();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param searchAfter: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext searchAfter(FieldDoc searchAfter) {
        return in.searchAfter(searchAfter);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public FieldDoc searchAfter() {
        return in.searchAfter();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param postFilter: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext parsedPostFilter(ParsedQuery postFilter) {
        return in.parsedPostFilter(postFilter);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ParsedQuery parsedPostFilter() {
        return in.parsedPostFilter();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param query: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext parsedQuery(ParsedQuery query) {
        return in.parsedQuery(query);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ParsedQuery parsedQuery() {
        return in.parsedQuery();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Query query() {
        return in.query();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public int from() {
        return in.from();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param from: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext from(int from) {
        return in.from(from);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public int size() {
        return in.size();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param size: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext size(int size) {
        return in.size(size);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean explain() {
        return in.explain();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param explain: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void explain(boolean explain) {
        in.explain(explain);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public List<String> groupStats() {
        return in.groupStats();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean version() {
        return in.version();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param version: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void version(boolean version) {
        in.version(version);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean seqNoAndPrimaryTerm() {
        return in.seqNoAndPrimaryTerm();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param seqNoAndPrimaryTerm: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void seqNoAndPrimaryTerm(boolean seqNoAndPrimaryTerm) {
        in.seqNoAndPrimaryTerm(seqNoAndPrimaryTerm);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public DfsSearchResult dfsResult() {
        return in.dfsResult();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addDfsResult() {
        in.addDfsResult();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public QuerySearchResult queryResult() {
        return in.queryResult();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addQueryResult() {
        in.addQueryResult();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public TotalHits getTotalHits() {
        return in.getTotalHits();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public float getMaxScore() {
        return in.getMaxScore();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addRankFeatureResult() {
        in.addRankFeatureResult();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public RankFeatureResult rankFeatureResult() {
        return in.rankFeatureResult();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public FetchSearchResult fetchResult() {
        return in.fetchResult();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addFetchResult() {
        in.addFetchResult();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public FetchPhase fetchPhase() {
        return in.fetchPhase();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public long getRelativeTimeInMillis() {
        return in.getRelativeTimeInMillis();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param name: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchExtBuilder getSearchExt(String name) {
        return in.getSearchExt(name);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Profilers getProfilers() {
        return in.getProfilers();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchExecutionContext getSearchExecutionContext() {
        return in.getSearchExecutionContext();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param task: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void setTask(CancellableTask task) {
        in.setTask(task);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public CancellableTask getTask() {
        return in.getTask();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean isCancelled() {
        return in.isCancelled();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public CollapseContext collapse() {
        return in.collapse();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param rescore: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addRescore(RescoreContext rescore) {
        in.addRescore(rescore);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ReaderContext readerContext() {
        return in.readerContext();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SourceLoader newSourceLoader() {
        return in.newSourceLoader();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public IdLoader newIdLoader() {
        return in.newIdLoader();
    }
}
