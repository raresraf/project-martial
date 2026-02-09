/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.OrdinalMap;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.util.NumericUtils;
import org.elasticsearch.action.search.SearchType;
import org.elasticsearch.cluster.routing.IndexRouting;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.core.Releasable;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.index.IndexMode;
import org.elasticsearch.index.IndexService;
import org.elasticsearch.index.IndexSettings;
import org.elasticsearch.index.IndexVersions;
import org.elasticsearch.index.cache.bitset.BitsetFilterCache;
import org.elasticsearch.index.engine.Engine;
import org.elasticsearch.index.fielddata.FieldDataContext;
import org.elasticsearch.index.fielddata.IndexFieldData;
import org.elasticsearch.index.fielddata.IndexNumericFieldData;
import org.elasticsearch.index.fielddata.IndexOrdinalsFieldData;
import org.elasticsearch.index.mapper.IdLoader;
import org.elasticsearch.index.mapper.KeywordFieldMapper;
import org.elasticsearch.index.mapper.MappedFieldType;
import org.elasticsearch.index.mapper.Mapper;
import org.elasticsearch.index.mapper.NestedLookup;
import org.elasticsearch.index.mapper.SourceLoader;
import org.elasticsearch.index.query.AbstractQueryBuilder;
import org.elasticsearch.index.query.ParsedQuery;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.index.search.NestedHelper;
import org.elasticsearch.index.shard.IndexShard;
import org.elasticsearch.search.aggregations.SearchContextAggregations;
import org.elasticsearch.search.builder.SearchSourceBuilder;
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
import org.elasticsearch.search.rescore.RescorePhase;
import org.elasticsearch.search.slice.SliceBuilder;
import org.elasticsearch.search.sort.SortAndFormats;
import org.elasticsearch.search.suggest.SuggestionSearchContext;
import org.elasticsearch.tasks.CancellableTask;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.Executor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.LongSupplier;
import java.util.function.ToLongFunction;

import static org.elasticsearch.search.SearchService.DEFAULT_SIZE;

/**
 * @brief Functional description of the DefaultSearchContext class.
 *        This is a placeholder for detailed semantic documentation.
 *        Further analysis will elaborate on its algorithm, complexity, and invariants.
 */
final class DefaultSearchContext extends SearchContext {

    private final ReaderContext readerContext;
    private final ShardSearchRequest request;
    private final SearchShardTarget shardTarget;
    private final LongSupplier relativeTimeSupplier;
    private final SearchType searchType;
    private final IndexShard indexShard;
    private final IndexService indexService;
    private final ContextIndexSearcher searcher;
    private DfsSearchResult dfsResult;
    private QuerySearchResult queryResult;
    private RankFeatureResult rankFeatureResult;
    private FetchSearchResult fetchResult;
    private final float queryBoost;
    private final boolean lowLevelCancellation;
    private TimeValue timeout;
    // terminate after count
    private int terminateAfter = DEFAULT_TERMINATE_AFTER;
    private List<String> groupStats;
    private boolean explain;
    private boolean version = false; // by default, we don't return versions
    private boolean seqAndPrimaryTerm = false;
    private StoredFieldsContext storedFields;
    private ScriptFieldsContext scriptFields;
    private FetchSourceContext fetchSourceContext;
    private FetchDocValuesContext docValuesContext;
    private FetchFieldsContext fetchFieldsContext;
    private int from = -1;
    private int size = -1;
    private SortAndFormats sort;
    private Float minimumScore;
    private boolean trackScores = false; // when sorting, track scores as well...
    private int trackTotalHitsUpTo = SearchContext.DEFAULT_TRACK_TOTAL_HITS_UP_TO;
    private FieldDoc searchAfter;
    private CollapseContext collapse;
    // filter for sliced scroll
    private SliceBuilder sliceBuilder;
    private CancellableTask task;
    private QueryPhaseRankShardContext queryPhaseRankShardContext;

    /**
     * The original query as sent by the user without the types and aliases
     * applied. Putting things in here leaks them into highlighting so don't add
     * things like the type filter or alias filters.
     */
    private ParsedQuery originalQuery;

    /**
     * The query to actually execute.
     */
    private Query query;
    private ParsedQuery postFilter;
    private Query aliasFilter;
    private SearchContextAggregations aggregations;
    private SearchHighlightContext highlight;
    private SuggestionSearchContext suggest;
    private List<RescoreContext> rescore;
    private Profilers profilers;

    private final Map<String, SearchExtBuilder> searchExtBuilders = new HashMap<>();
    private final SearchExecutionContext searchExecutionContext;
    private final FetchPhase fetchPhase;

    DefaultSearchContext(
        ReaderContext readerContext,
        ShardSearchRequest request,
        SearchShardTarget shardTarget,
        LongSupplier relativeTimeSupplier,
        TimeValue timeout,
        FetchPhase fetchPhase,
        boolean lowLevelCancellation,
        Executor executor,
        SearchService.ResultsType resultsType,
        boolean enableQueryPhaseParallelCollection,
        int minimumDocsPerSlice
    ) throws IOException {
        this.readerContext = readerContext;
        this.request = request;
        this.fetchPhase = fetchPhase;
        boolean success = false;
        try {
            this.searchType = request.searchType();
            this.shardTarget = shardTarget;
            this.indexService = readerContext.indexService();
            this.indexShard = readerContext.indexShard();

            Engine.Searcher engineSearcher = readerContext.acquireSearcher("search");
            int maximumNumberOfSlices = determineMaximumNumberOfSlices(
                executor,
                request,
                resultsType,
                enableQueryPhaseParallelCollection,
                field -> getFieldCardinality(field, readerContext.indexService(), engineSearcher.getDirectoryReader())
            );
            if (executor == null || maximumNumberOfSlices <= 1) {
                this.searcher = new ContextIndexSearcher(
                    engineSearcher.getIndexReader(),
                    engineSearcher.getSimilarity(),
                    engineSearcher.getQueryCache(),
                    engineSearcher.getQueryCachingPolicy(),
                    lowLevelCancellation
                );
            } else {
                this.searcher = new ContextIndexSearcher(
                    engineSearcher.getIndexReader(),
                    engineSearcher.getSimilarity(),
                    engineSearcher.getQueryCache(),
                    engineSearcher.getQueryCachingPolicy(),
                    lowLevelCancellation,
                    wrapExecutor(executor),
                    maximumNumberOfSlices,
                    minimumDocsPerSlice
                );
            }
            releasables.addAll(List.of(engineSearcher, searcher));
            this.relativeTimeSupplier = relativeTimeSupplier;
            this.timeout = timeout;
            searchExecutionContext = indexService.newSearchExecutionContext(
                request.shardId().id(),
                request.shardRequestIndex(),
                searcher,
                request::nowInMillis,
                shardTarget.getClusterAlias(),
                request.getRuntimeMappings(),
                request.source() == null ? null : request.source().size()
            );
            queryBoost = request.indexBoost();
            this.lowLevelCancellation = lowLevelCancellation;
            success = true;
        } finally {
            if (success == false) {
                close();
            }
        }
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param executor: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    private static Executor wrapExecutor(Executor executor) {
        if (executor instanceof ThreadPoolExecutor tpe) {
            // let this searcher fork to a limited maximum number of tasks, to protect against situations where Lucene may
            // submit too many segment level tasks. With enough parallel search requests and segments per shards, they may all see
            // an empty queue and start parallelizing, filling up the queue very quickly and causing rejections, due to
            // many small tasks in the queue that become no-op because the active caller thread will execute them instead.
            // Note that despite all tasks are completed, TaskExecutor#invokeAll leaves the leftover no-op tasks in queue hence
            // they contribute to the queue size until they are removed from it.
            AtomicInteger segmentLevelTasks = new AtomicInteger(0);
            return command -> {
                if (segmentLevelTasks.incrementAndGet() > tpe.getMaximumPoolSize()) {
                    try {
                        command.run();
                    } finally {
                        segmentLevelTasks.decrementAndGet();
                    }
                } else {
                    executor.execute(() -> {
                        try {
                            command.run();
                        } finally {
                            segmentLevelTasks.decrementAndGet();
                        }
                    });
                }
            };
        }
        return executor;
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param field: [Description]
     * @param indexService: [Description]
     * @param directoryReader: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    static long getFieldCardinality(String field, IndexService indexService, DirectoryReader directoryReader) {
        MappedFieldType mappedFieldType = indexService.mapperService().fieldType(field);
        if (mappedFieldType == null) {
            return -1;
        }
        IndexFieldData<?> indexFieldData;
        try {
            indexFieldData = indexService.loadFielddata(mappedFieldType, FieldDataContext.noRuntimeFields("field cardinality"));
        } catch (Exception e) {
            // loading fielddata for runtime fields will fail, that's ok
            return -1;
        }
        return getFieldCardinality(indexFieldData, directoryReader);
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param indexFieldData: [Description]
     * @param directoryReader: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    static long getFieldCardinality(IndexFieldData<?> indexFieldData, DirectoryReader directoryReader) {
        if (indexFieldData instanceof IndexOrdinalsFieldData indexOrdinalsFieldData) {
            if (indexOrdinalsFieldData.supportsGlobalOrdinalsMapping()) {
                IndexOrdinalsFieldData global = indexOrdinalsFieldData.loadGlobal(directoryReader);
                OrdinalMap ordinalMap = global.getOrdinalMap();
                if (ordinalMap != null) {
                    return ordinalMap.getValueCount();
                }
                if (directoryReader.leaves().isEmpty()) {
                    return 0;
                }
                return global.load(directoryReader.leaves().get(0)).getOrdinalsValues().getValueCount();
            }
        } else if (indexFieldData instanceof IndexNumericFieldData indexNumericFieldData) {
            final IndexNumericFieldData.NumericType type = indexNumericFieldData.getNumericType();
            try {
                if (type == IndexNumericFieldData.NumericType.INT || type == IndexNumericFieldData.NumericType.SHORT) {
                    final IndexReader reader = directoryReader.getContext().reader();
                    final byte[] min = PointValues.getMinPackedValue(reader, indexFieldData.getFieldName());
                    final byte[] max = PointValues.getMaxPackedValue(reader, indexFieldData.getFieldName());
                    if (min != null && max != null) {
                        return NumericUtils.sortableBytesToInt(max, 0) - NumericUtils.sortableBytesToInt(min, 0) + 1;
                    }
                } else if (type == IndexNumericFieldData.NumericType.LONG) {
                    final IndexReader reader = directoryReader.getContext().reader();
                    final byte[] min = PointValues.getMinPackedValue(reader, indexFieldData.getFieldName());
                    final byte[] max = PointValues.getMaxPackedValue(reader, indexFieldData.getFieldName());
                    if (min != null && max != null) {
                        return NumericUtils.sortableBytesToLong(max, 0) - NumericUtils.sortableBytesToLong(min, 0) + 1;
                    }
                }
            } catch (IOException ioe) {
                return -1L;
            }
        }
        //
        return -1L;
    }

    static int determineMaximumNumberOfSlices(
        Executor executor,
        ShardSearchRequest request,
        SearchService.ResultsType resultsType,
        boolean enableQueryPhaseParallelCollection,
        ToLongFunction<String> fieldCardinality
    ) {
        // Note: although this method refers to parallel collection, it affects any kind of parallelism, including query rewrite,
        // given that if 1 is the returned value, no executor is provided to the searcher.
        return executor instanceof ThreadPoolExecutor tpe
            && tpe.getQueue().size() <= tpe.getMaximumPoolSize()
            && isParallelCollectionSupportedForResults(resultsType, request.source(), fieldCardinality, enableQueryPhaseParallelCollection)
                ? tpe.getMaximumPoolSize()
                : 1;
    }

    static boolean isParallelCollectionSupportedForResults(
        SearchService.ResultsType resultsType,
        SearchSourceBuilder source,
        ToLongFunction<String> fieldCardinality,
        boolean isQueryPhaseParallelismEnabled
    ) {
        if (resultsType == SearchService.ResultsType.DFS) {
            return true;
        }
        if (resultsType == SearchService.ResultsType.QUERY && isQueryPhaseParallelismEnabled) {
            return source == null || source.supportsParallelCollection(fieldCardinality);
        }
        return false;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addRankFeatureResult() {
        this.rankFeatureResult = new RankFeatureResult(this.readerContext.id(), this.shardTarget, this.request);
        addReleasable(rankFeatureResult::decRef);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public RankFeatureResult rankFeatureResult() {
        return rankFeatureResult;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addFetchResult() {
        this.fetchResult = new FetchSearchResult(this.readerContext.id(), this.shardTarget);
        addReleasable(fetchResult::decRef);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addQueryResult() {
        this.queryResult = new QuerySearchResult(this.readerContext.id(), this.shardTarget, this.request);
        addReleasable(queryResult::decRef);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addDfsResult() {
        this.dfsResult = new DfsSearchResult(this.readerContext.id(), this.shardTarget, this.request);
    }

    /**
     * Should be called before executing the main query and after all other parameters have been set.
     */
    @Override
    public void preProcess() {
        if (hasOnlySuggest()) {
            return;
        }
        long from = from() == -1 ? 0 : from();
        long size = size() == -1 ? DEFAULT_SIZE : size();
        long resultWindow = from + size;
        int maxResultWindow = indexService.getIndexSettings().getMaxResultWindow();

        if (resultWindow > maxResultWindow) {
            if (scrollContext() == null) {
                throw new IllegalArgumentException(
                    "Result window is too large, from + size must be less than or equal to: ["
                        + maxResultWindow
                        + "] but was ["
                        + resultWindow
                        + "]. See the scroll api for a more efficient way to request large data sets. "
                        + "This limit can be set by changing the ["
                        + IndexSettings.MAX_RESULT_WINDOW_SETTING.getKey()
                        + "] index level setting."
                );
            }
            throw new IllegalArgumentException(
                "Batch size is too large, size must be less than or equal to: ["
                    + maxResultWindow
                    + "] but was ["
                    + resultWindow
                    + "]. Scroll batch sizes cost as much memory as result windows so they are controlled by the ["
                    + IndexSettings.MAX_RESULT_WINDOW_SETTING.getKey()
                    + "] index level setting."
            );
        }
        if (rescore != null) {
            if (RescorePhase.validateSort(sort) == false) {
                throw new IllegalArgumentException("Cannot use [sort] option in conjunction with [rescore].");
            }
            int maxWindow = indexService.getIndexSettings().getMaxRescoreWindow();
            for (RescoreContext rescoreContext : rescore()) {
                if (rescoreContext.getWindowSize() > maxWindow) {
                    throw new IllegalArgumentException(
                        "Rescore window ["
                            + rescoreContext.getWindowSize()
                            + "] is too large. "
                            + "It must be less than ["
                            + maxWindow
                            + "]. This prevents allocating massive heaps for storing the results "
                            + "to be rescored. This limit can be set by changing the ["
                            + IndexSettings.MAX_RESCORE_WINDOW_SETTING.getKey()
                            + "] index level setting."
                    );
                }
            }
        }

        if (sliceBuilder != null && scrollContext() != null) {
            int sliceLimit = indexService.getIndexSettings().getMaxSlicesPerScroll();
            int numSlices = sliceBuilder.getMax();
            if (numSlices > sliceLimit) {
                throw new IllegalArgumentException(
                    "The number of slices ["
                        + numSlices
                        + "] is too large. It must "
                        + "be less than ["
                        + sliceLimit
                        + "]. This limit can be set by changing the ["
                        + IndexSettings.MAX_SLICES_PER_SCROLL.getKey()
                        + "] index level setting."
                );
            }
        }

        // initialize the filtering alias based on the provided filters
        try {
            final QueryBuilder queryBuilder = request.getAliasFilter().getQueryBuilder();
            aliasFilter = queryBuilder == null ? null : queryBuilder.toQuery(searchExecutionContext);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }

        if (query == null) {
            parsedQuery(ParsedQuery.parsedMatchAllQuery());
        }
        if (queryBoost != AbstractQueryBuilder.DEFAULT_BOOST) {
            parsedQuery(new ParsedQuery(new BoostQuery(query, queryBoost), parsedQuery()));
        }
        this.query = buildFilteredQuery(query);
        if (lowLevelCancellation) {
            searcher().addQueryCancellation(() -> {
                final CancellableTask task = getTask();
                if (task != null) {
                    task.ensureNotCancelled();
                }
            });
        }
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param query: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Query buildFilteredQuery(Query query) {
        List<Query> filters = new ArrayList<>();
        NestedLookup nestedLookup = searchExecutionContext.nestedLookup();
        NestedHelper nestedHelper = new NestedHelper(nestedLookup, searchExecutionContext::isFieldMapped);
        if (nestedLookup != NestedLookup.EMPTY
            && nestedHelper.mightMatchNestedDocs(query)
            && (aliasFilter == null || nestedHelper.mightMatchNestedDocs(aliasFilter))) {
            filters.add(Queries.newNonNestedFilter(searchExecutionContext.indexVersionCreated()));
        }

        if (aliasFilter != null) {
            filters.add(aliasFilter);
        }

        if (sliceBuilder != null) {
            Query slicedQuery = sliceBuilder.toFilter(request, searchExecutionContext);
            if (slicedQuery instanceof MatchNoDocsQuery) {
                return slicedQuery;
            } else {
                filters.add(slicedQuery);
            }
        }

        if (filters.isEmpty()) {
            return query;
        } else {
            BooleanQuery.Builder builder = new BooleanQuery.Builder();
            builder.add(query, Occur.MUST);
            for (Query filter : filters) {
                builder.add(filter, Occur.FILTER);
            }
            return builder.build();
        }
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ShardSearchContextId id() {
        return readerContext.id();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public String source() {
        return "search";
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ShardSearchRequest request() {
        return this.request;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchType searchType() {
        return this.searchType;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchShardTarget shardTarget() {
        return this.shardTarget;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public int numberOfShards() {
        return request.numberOfShards();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ScrollContext scrollContext() {
        return readerContext.scrollContext();
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
     * @param aggregations: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext aggregations(SearchContextAggregations aggregations) {
        this.aggregations = aggregations;
        return this;
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param searchExtBuilder: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addSearchExt(SearchExtBuilder searchExtBuilder) {
        // it's ok to use the writeable name here given that we enforce it to be the same as the name of the element that gets
        // parsed by the corresponding parser. There is one single name and one single way to retrieve the parsed object from the context.
        searchExtBuilders.put(searchExtBuilder.getWriteableName(), searchExtBuilder);
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
        return highlight;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param highlight: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void highlight(SearchHighlightContext highlight) {
        this.highlight = highlight;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SuggestionSearchContext suggest() {
        return suggest;
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param suggest: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void suggest(SuggestionSearchContext suggest) {
        this.suggest = suggest;
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
     * @param queryPhaseRankShardContext: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void queryPhaseRankShardContext(QueryPhaseRankShardContext queryPhaseRankShardContext) {
        this.queryPhaseRankShardContext = queryPhaseRankShardContext;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public List<RescoreContext> rescore() {
        if (rescore == null) {
            return List.of();
        }
        return rescore;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param rescore: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addRescore(RescoreContext rescore) {
        if (this.rescore == null) {
            this.rescore = new ArrayList<>();
        }
        this.rescore.add(rescore);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean hasScriptFields() {
        return scriptFields != null && scriptFields.fields().isEmpty() == false;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ScriptFieldsContext scriptFields() {
        if (scriptFields == null) {
            scriptFields = new ScriptFieldsContext();
        }
        return this.scriptFields;
    }

    /**
     * A shortcut function to see whether there is a fetchSourceContext and it says the source is requested.
     */
    @Override
    public boolean sourceRequested() {
        return fetchSourceContext != null && fetchSourceContext.fetchSource();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public FetchSourceContext fetchSourceContext() {
        return this.fetchSourceContext;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param fetchSourceContext: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext fetchSourceContext(FetchSourceContext fetchSourceContext) {
        this.fetchSourceContext = fetchSourceContext;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public FetchDocValuesContext docValuesContext() {
        return docValuesContext;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param docValuesContext: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext docValuesContext(FetchDocValuesContext docValuesContext) {
        this.docValuesContext = docValuesContext;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public FetchFieldsContext fetchFieldsContext() {
        return fetchFieldsContext;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param fetchFieldsContext: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext fetchFieldsContext(FetchFieldsContext fetchFieldsContext) {
        this.fetchFieldsContext = fetchFieldsContext;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ContextIndexSearcher searcher() {
        return this.searcher;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public IndexShard indexShard() {
        return this.indexShard;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public BitsetFilterCache bitsetFilterCache() {
        return indexService.cache().bitsetFilterCache();
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public TimeValue timeout() {
        return timeout;
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param timeout: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void timeout(TimeValue timeout) {
        this.timeout = timeout;
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
     * @param minimumScore: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext minimumScore(float minimumScore) {
        this.minimumScore = minimumScore;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Float minimumScore() {
        return this.minimumScore;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param sort: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext sort(SortAndFormats sort) {
        this.sort = sort;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SortAndFormats sort() {
        return this.sort;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param trackScores: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext trackScores(boolean trackScores) {
        this.trackScores = trackScores;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean trackScores() {
        return this.trackScores;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param trackTotalHitsUpTo: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext trackTotalHitsUpTo(int trackTotalHitsUpTo) {
        this.trackTotalHitsUpTo = trackTotalHitsUpTo;
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
     * @param searchAfter: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext searchAfter(FieldDoc searchAfter) {
        this.searchAfter = searchAfter;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean lowLevelCancellation() {
        return lowLevelCancellation;
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

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param collapse: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext collapse(CollapseContext collapse) {
        this.collapse = collapse;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public CollapseContext collapse() {
        return collapse;
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param sliceBuilder: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext sliceBuilder(SliceBuilder sliceBuilder) {
        this.sliceBuilder = sliceBuilder;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param postFilter: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext parsedPostFilter(ParsedQuery postFilter) {
        this.postFilter = postFilter;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ParsedQuery parsedPostFilter() {
        return this.postFilter;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param query: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext parsedQuery(ParsedQuery query) {
        this.originalQuery = query;
        this.query = query.query();
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public ParsedQuery parsedQuery() {
        return this.originalQuery;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Query query() {
        return this.query;
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
     * @param from: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext from(int from) {
        this.from = from;
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

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param size: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext size(int size) {
        this.size = size;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean hasStoredFields() {
        return storedFields != null && storedFields.fieldNames() != null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public StoredFieldsContext storedFieldsContext() {
        return storedFields;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param storedFieldsContext: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SearchContext storedFieldsContext(StoredFieldsContext storedFieldsContext) {
        this.storedFields = storedFieldsContext;
        return this;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean explain() {
        return explain;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param explain: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void explain(boolean explain) {
        this.explain = explain;
    }

    @Override
    @Nullable
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public List<String> groupStats() {
        return this.groupStats;
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param groupStats: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void groupStats(List<String> groupStats) {
        this.groupStats = groupStats;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean version() {
        return version;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param version: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void version(boolean version) {
        this.version = version;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public boolean seqNoAndPrimaryTerm() {
        return seqAndPrimaryTerm;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param seqNoAndPrimaryTerm: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void seqNoAndPrimaryTerm(boolean seqNoAndPrimaryTerm) {
        this.seqAndPrimaryTerm = seqNoAndPrimaryTerm;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public DfsSearchResult dfsResult() {
        return dfsResult;
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

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param releasable: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void addQuerySearchResultReleasable(Releasable releasable) {
        queryResult.addReleasable(releasable);
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public TotalHits getTotalHits() {
        if (queryResult != null) {
            return queryResult.getTotalHits();
        }
        return null;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public float getMaxScore() {
        if (queryResult != null) {
            return queryResult.getMaxScore();
        }
        return Float.NaN;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public FetchPhase fetchPhase() {
        return fetchPhase;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public FetchSearchResult fetchResult() {
        return fetchResult;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public long getRelativeTimeInMillis() {
        return relativeTimeSupplier.getAsLong();
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
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public Profilers getProfilers() {
        return profilers;
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param profilers: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public void setProfilers(Profilers profilers) {
        this.profilers = profilers;
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
    public ReaderContext readerContext() {
        return readerContext;
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public SourceLoader newSourceLoader() {
        return searchExecutionContext.newSourceLoader(request.isForceSyntheticSource());
    }

    @Override
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public IdLoader newIdLoader() {
        if (indexService.getIndexSettings().getMode() == IndexMode.TIME_SERIES) {
            IndexRouting.ExtractFromSource indexRouting = null;
            List<String> routingPaths = null;
            if (indexService.getIndexSettings().getIndexVersionCreated().before(IndexVersions.TIME_SERIES_ROUTING_HASH_IN_ID)) {
                indexRouting = (IndexRouting.ExtractFromSource) indexService.getIndexSettings().getIndexRouting();
                routingPaths = indexService.getMetadata().getRoutingPaths();
                for (String routingField : routingPaths) {
                    if (routingField.contains("*")) {
                        // In case the routing fields include path matches, find any matches and add them as distinct fields
                        // to the routing path.
                        Set<String> matchingRoutingPaths = new TreeSet<>(routingPaths);
                        for (Mapper mapper : indexService.mapperService().mappingLookup().fieldMappers()) {
                            if (mapper instanceof KeywordFieldMapper && indexRouting.matchesField(mapper.fullPath())) {
                                matchingRoutingPaths.add(mapper.fullPath());
                            }
                        }
                        routingPaths = new ArrayList<>(matchingRoutingPaths);
                        break;
                    }
                }
            }
            return IdLoader.createTsIdLoader(indexRouting, routingPaths);
        } else {
            return IdLoader.fromLeafStoredFieldLoader();
        }
    }
}
