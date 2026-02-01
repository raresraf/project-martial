/**
 * A search source builder allowing to easily build search source. Simple
 * construction using {@link SearchSourceBuilder#searchSource()}.
 *
 * This class provides a comprehensive fluent API for defining all aspects of an Elasticsearch search request.
 * It acts as a central component for constructing complex search queries, including queries, post-filters,
 * aggregations, sorting, highlighting, fetching source fields, and various other search features.
 *
 * Architectural Intent: To abstract the complexity of building raw JSON search requests into a programmatic,
 * type-safe, and readable format, facilitating the construction and manipulation of search queries.
 * It integrates deeply with other builder classes for specific search functionalities like highlighting,
 * aggregations, and sorting.
 *
 * @see SearchRequest#source(SearchSourceBuilder)
 */
public final class SearchSourceBuilder implements Writeable, ToXContentObject, Rewriteable<SearchSourceBuilder> {

    public static final ParseField FROM_FIELD = new ParseField("from");
    public static final ParseField SIZE_FIELD = new ParseField("size");
    public static final ParseField TIMEOUT_FIELD = new ParseField("timeout");
    public static final ParseField TERMINATE_AFTER_FIELD = new ParseField("terminate_after");
    public static final ParseField QUERY_FIELD = new ParseField("query");
    @UpdateForV10(owner = UpdateForV10.Owner.SEARCH_RELEVANCE) // remove [sub_searches] and [rank] support in 10.0
    public static final ParseField SUB_SEARCHES_FIELD = new ParseField("sub_searches").withAllDeprecated("retriever");
    public static final ParseField RANK_FIELD = new ParseField("rank").withAllDeprecated("retriever");
    public static final ParseField POST_FILTER_FIELD = new ParseField("post_filter");
    public static final ParseField KNN_FIELD = new ParseField("knn");
    public static final ParseField MIN_SCORE_FIELD = new ParseField("min_score");
    public static final ParseField VERSION_FIELD = new ParseField("version");
    public static final ParseField SEQ_NO_PRIMARY_TERM_FIELD = new ParseField("seq_no_primary_term");
    public static final ParseField EXPLAIN_FIELD = new ParseField("explain");
    public static final ParseField _SOURCE_FIELD = new ParseField("_source");
    public static final ParseField STORED_FIELDS_FIELD = new ParseField("stored_fields");
    public static final ParseField DOCVALUE_FIELDS_FIELD = new ParseField("docvalue_fields");
    public static final ParseField FETCH_FIELDS_FIELD = new ParseField("fields");
    public static final ParseField SCRIPT_FIELDS_FIELD = new ParseField("script_fields");
    public static final ParseField SCRIPT_FIELD = new ParseField("script");
    public static final ParseField IGNORE_FAILURE_FIELD = new ParseField("ignore_failure");
    public static final ParseField SORT_FIELD = new ParseField("sort");
    public static final ParseField TRACK_SCORES_FIELD = new ParseField("track_scores");
    public static final ParseField TRACK_TOTAL_HITS_FIELD = new ParseField("track_total_hits");
    public static final ParseField INDICES_BOOST_FIELD = new ParseField("indices_boost");
    public static final ParseField AGGREGATIONS_FIELD = new ParseField("aggregations");
    public static final ParseField AGGS_FIELD = new ParseField("aggs");
    public static final ParseField HIGHLIGHT_FIELD = new ParseField("highlight");
    public static final ParseField SUGGEST_FIELD = new ParseField("suggest");
    public static final ParseField RESCORE_FIELD = new ParseField("rescore");
    public static final ParseField STATS_FIELD = new ParseField("stats");
    public static final ParseField EXT_FIELD = new ParseField("ext");
    public static final ParseField PROFILE_FIELD = new ParseField("profile");
    public static final ParseField SEARCH_AFTER = new ParseField("search_after");
    public static final ParseField COLLAPSE = new ParseField("collapse");
    public static final ParseField SLICE = new ParseField("slice");
    public static final ParseField POINT_IN_TIME = new ParseField("pit");
    public static final ParseField RUNTIME_MAPPINGS_FIELD = new ParseField("runtime_mappings");
    public static final ParseField RETRIEVER = new ParseField("retriever");

    private static final boolean RANK_SUPPORTED = Booleans.parseBoolean(System.getProperty("es.search.rank_supported"), true);
    // Functional Utility: Flag to indicate if the 'rank' feature is supported, controlled by a system property.

    /**
     * A static factory method to construct a new search source.
     */
    public static SearchSourceBuilder searchSource() {
        // Functional Utility: Provides a static factory method to create a new instance of SearchSourceBuilder.
        return new SearchSourceBuilder();
    }

    /**
     * A static factory method to construct new search highlights.
     */
    public static HighlightBuilder highlight() {
        // Functional Utility: Provides a static factory method to create a new instance of HighlightBuilder.
        return new HighlightBuilder();
    }

    private transient RetrieverBuilder retrieverBuilder; // Transient field for retriever configuration.

    private List<SubSearchSourceBuilder> subSearchSourceBuilders = new ArrayList<>(); // List of sub-search source builders.

    private QueryBuilder postQueryBuilder; // Query builder for post-filtering search results.

    private List<KnnSearchBuilder> knnSearch = new ArrayList<>(); // List of k-Nearest Neighbors search builders.

    private RankBuilder rankBuilder = null; // Rank builder for advanced ranking strategies.

    private int from = -1; // Starting offset for search results.

    private int size = -1; // Number of search results to return.

    private Boolean explain; // Flag to indicate if explanations should be returned for search hits.

    private Boolean version; // Flag to indicate if document versions should be returned.

    private Boolean seqNoAndPrimaryTerm; // Flag to indicate if sequence number and primary term should be returned.

    private List<SortBuilder<?>> sorts; // List of sort builders for ordering search results.

    private boolean trackScores = false; // Flag to indicate if scores should be tracked for sorting.

    private Integer trackTotalHitsUpTo; // Limit for tracking total hits.

    private SearchAfterBuilder searchAfterBuilder; // Builder for pagination using search_after.

    private SliceBuilder sliceBuilder; // Builder for slicing search results.

    private Float minScore; // Minimum score for search hits.

    private TimeValue timeout = null; // Timeout for the search request.
    private int terminateAfter = SearchContext.DEFAULT_TERMINATE_AFTER; // Number of documents to terminate the search after.

    private StoredFieldsContext storedFieldsContext; // Context for fetching stored fields.
    private List<FieldAndFormat> docValueFields; // List of doc value fields to fetch.
    private List<ScriptField> scriptFields; // List of script fields to include.
    private FetchSourceContext fetchSourceContext; // Context for fetching source fields.
    private List<FieldAndFormat> fetchFields; // List of fields to fetch.

    private AggregatorFactories.Builder aggregations; // Builder for aggregations.

    private HighlightBuilder highlightBuilder; // Builder for highlighting.

    private SuggestBuilder suggestBuilder; // Builder for suggestions.

    @SuppressWarnings("rawtypes")
    private List<RescorerBuilder> rescoreBuilders; // List of rescorer builders.

    private List<IndexBoost> indexBoosts = new ArrayList<>(); // List of index boosts.

    private List<String> stats; // List of stats groups.

    private List<SearchExtBuilder> extBuilders = Collections.emptyList(); // List of search extension builders.

    private boolean profile = false; // Flag to indicate if query profiling should be enabled.

    private CollapseBuilder collapse = null; // Builder for collapsing search results.

    private PointInTimeBuilder pointInTimeBuilder = null; // Builder for point in time search.

    private Map<String, Object> runtimeMappings = emptyMap(); // Runtime mappings for fields.

    private boolean skipInnerHits = false; // Flag to skip inner hits.

    /**
     * Constructs a new search source builder.
     */
    public SearchSourceBuilder() {}

    /**
     * Read from a stream.
     */
    public SearchSourceBuilder(StreamInput in) throws IOException {
        // Functional Utility: Reconstructs a SearchSourceBuilder instance from a StreamInput.
        // This constructor is used for deserialization in a distributed environment, reading
        // various search options and builders from the stream, handling version compatibility.
        aggregations = in.readOptionalWriteable(AggregatorFactories.Builder::new);
        explain = in.readOptionalBoolean();
        fetchSourceContext = in.readOptionalWriteable(FetchSourceContext::readFrom);
        // Conditional Logic: Read docValueFields if present.
        if (in.readBoolean()) {
            docValueFields = in.readCollectionAsList(FieldAndFormat::new);
        } else {
            docValueFields = null;
        }
        storedFieldsContext = in.readOptionalWriteable(StoredFieldsContext::new);
        from = in.readVInt();
        highlightBuilder = in.readOptionalWriteable(HighlightBuilder::new);
        indexBoosts = in.readCollectionAsList(IndexBoost::new);
        minScore = in.readOptionalFloat();
        postQueryBuilder = in.readOptionalNamedWriteable(QueryBuilder.class);
        // Conditional Logic: Handle subSearchSourceBuilders based on transport version.
        if (in.getTransportVersion().onOrAfter(TransportVersions.V_8_9_X)) {
            subSearchSourceBuilders = in.readCollectionAsList(SubSearchSourceBuilder::new);
        } else {
            // Block Logic: Fallback for older versions, reading a single query builder.
            QueryBuilder queryBuilder = in.readOptionalNamedWriteable(QueryBuilder.class);
            if (queryBuilder != null) {
                subSearchSourceBuilders.add(new SubSearchSourceBuilder(queryBuilder));
            }
        }
        // Conditional Logic: Read rescoreBuilders if present.
        if (in.readBoolean()) {
            rescoreBuilders = in.readNamedWriteableCollectionAsList(RescorerBuilder.class);
        }
        // Conditional Logic: Read scriptFields if present.
        if (in.readBoolean()) {
            scriptFields = in.readCollectionAsList(ScriptField::new);
        }
        size = in.readVInt();
        // Conditional Logic: Read sorts if present.
        if (in.readBoolean()) {
            int size = in.readVInt();
            sorts = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                sorts.add(in.readNamedWriteable(SortBuilder.class));
            }
        }
        // Conditional Logic: Read stats if present.
        if (in.readBoolean()) {
            stats = in.readStringCollectionAsList();
        }
        suggestBuilder = in.readOptionalWriteable(SuggestBuilder::new);
        terminateAfter = in.readVInt();
        timeout = in.readOptionalTimeValue();
        trackScores = in.readBoolean();
        version = in.readOptionalBoolean();
        seqNoAndPrimaryTerm = in.readOptionalBoolean();
        extBuilders = in.readNamedWriteableCollectionAsList(SearchExtBuilder.class);
        profile = in.readBoolean();
        searchAfterBuilder = in.readOptionalWriteable(SearchAfterBuilder::new);
        sliceBuilder = in.readOptionalWriteable(SliceBuilder::new);
        collapse = in.readOptionalWriteable(CollapseBuilder::new);
        trackTotalHitsUpTo = in.readOptionalInt();
        // Conditional Logic: Read fetchFields if present.
        if (in.readBoolean()) {
            fetchFields = in.readCollectionAsList(FieldAndFormat::new);
        }
        pointInTimeBuilder = in.readOptionalWriteable(PointInTimeBuilder::new);
        runtimeMappings = in.readGenericMap();
        // Conditional Logic: Read knnSearch based on transport version.
        if (in.getTransportVersion().onOrAfter(TransportVersions.V_8_4_0)) {
            if (in.getTransportVersion().before(TransportVersions.V_8_7_0)) {
                KnnSearchBuilder searchBuilder = in.readOptionalWriteable(KnnSearchBuilder::new);
                knnSearch = searchBuilder != null ? List.of(searchBuilder) : List.of();
            } else {
                knnSearch = in.readCollectionAsList(KnnSearchBuilder::new);
            }
        }
        // Conditional Logic: Read rankBuilder based on transport version.
        if (in.getTransportVersion().onOrAfter(TransportVersions.V_8_8_0)) {
            rankBuilder = in.readOptionalNamedWriteable(RankBuilder.class);
        }
        // Conditional Logic: Read skipInnerHits based on transport version.
        if (in.getTransportVersion().onOrAfter(TransportVersions.V_8_16_1)) {
            skipInnerHits = in.readBoolean();
        } else {
            skipInnerHits = false;
        }
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        // Functional Utility: Writes the state of the SearchSourceBuilder to a StreamOutput.
        // This method serializes all configured search options and builders, handling version compatibility.
        // Conditional Logic: Throw exception if builder is not rewritten before serialization.
        if (retrieverBuilder != null) {
            throw new IllegalStateException("SearchSourceBuilder should be rewritten first");
        }
        out.writeOptionalWriteable(aggregations);
        out.writeOptionalBoolean(explain);
        out.writeOptionalWriteable(fetchSourceContext);
        out.writeBoolean(docValueFields != null);
        // Conditional Logic: Write docValueFields if present.
        if (docValueFields != null) {
            out.writeCollection(docValueFields);
        }
        out.writeOptionalWriteable(storedFieldsContext);
        out.writeVInt(from);
        out.writeOptionalWriteable(highlightBuilder);
        out.writeCollection(indexBoosts);
        out.writeOptionalFloat(minScore);
        out.writeOptionalNamedWriteable(postQueryBuilder);
        // Conditional Logic: Write subSearchSourceBuilders based on transport version.
        if (out.getTransportVersion().onOrAfter(TransportVersions.V_8_9_X)) {
            out.writeCollection(subSearchSourceBuilders);
        } else if (out.getTransportVersion().before(TransportVersions.V_8_4_0) && subSearchSourceBuilders.size() >= 2) {
            // Error Handling: Throw exception if multiple sub_searches are not supported by the target version.
            throw new IllegalArgumentException(
                "cannot serialize [sub_searches] to version [" + out.getTransportVersion().toReleaseVersion() + "]"
            );
        } else {
            out.writeOptionalNamedWriteable(query());
        }
        boolean hasRescoreBuilders = rescoreBuilders != null;
        out.writeBoolean(hasRescoreBuilders);
        // Conditional Logic: Write rescoreBuilders if present.
        if (hasRescoreBuilders) {
            out.writeNamedWriteableCollection(rescoreBuilders);
        }
        boolean hasScriptFields = scriptFields != null;
        out.writeBoolean(hasScriptFields);
        // Conditional Logic: Write scriptFields if present.
        if (hasScriptFields) {
            out.writeCollection(scriptFields);
        }
        out.writeVInt(size);
        boolean hasSorts = sorts != null;
        out.writeBoolean(hasSorts);
        // Conditional Logic: Write sorts if present.
        if (hasSorts) {
            out.writeNamedWriteableCollection(sorts);
        }
        boolean hasStats = stats != null;
        out.writeBoolean(hasStats);
        // Conditional Logic: Write stats if present.
        if (hasStats) {
            out.writeStringCollection(stats);
        }
        out.writeOptionalWriteable(suggestBuilder);
        out.writeVInt(terminateAfter);
        out.writeOptionalTimeValue(timeout);
        out.writeBoolean(trackScores);
        out.writeOptionalBoolean(version);
        out.writeOptionalBoolean(seqNoAndPrimaryTerm);
        out.writeNamedWriteableCollection(extBuilders);
        out.writeBoolean(profile);
        out.writeOptionalWriteable(searchAfterBuilder);
        out.writeOptionalWriteable(sliceBuilder);
        out.writeOptionalWriteable(collapse);
        out.writeOptionalInt(trackTotalHitsUpTo);
        out.writeBoolean(fetchFields != null);
        // Conditional Logic: Write fetchFields if present.
        if (fetchFields != null) {
            out.writeCollection(fetchFields);
        }
        out.writeOptionalWriteable(pointInTimeBuilder);
        out.writeGenericMap(runtimeMappings);
        // Conditional Logic: Write knnSearch based on transport version.
        if (out.getTransportVersion().onOrAfter(TransportVersions.V_8_4_0)) {
            if (out.getTransportVersion().before(TransportVersions.V_8_7_0)) {
                // Conditional Logic: Throw exception if multiple KNN search clauses are not supported by the target version.
                if (knnSearch.size() > 1) {
                    throw new IllegalArgumentException(
                        "Versions before ["
                            + TransportVersions.V_8_7_0.toReleaseVersion()
                            + "] don't support multiple [knn] search clauses and search was sent to ["
                            + out.getTransportVersion().toReleaseVersion()
                            + "]"
                    );
                }
                out.writeOptionalWriteable(knnSearch.isEmpty() ? null : knnSearch.get(0));
            } else {
                out.writeCollection(knnSearch);
            }
        }
        // Conditional Logic: Write rankBuilder based on transport version.
        if (out.getTransportVersion().onOrAfter(TransportVersions.V_8_8_0)) {
            out.writeOptionalNamedWriteable(rankBuilder);
        } else if (rankBuilder != null) {
            // Error Handling: Throw exception if rank is not supported by the target version.
            throw new IllegalArgumentException("cannot serialize [rank] to version [" + out.getTransportVersion().toReleaseVersion() + "]");
        }
        // Conditional Logic: Write skipInnerHits based on transport version.
        if (out.getTransportVersion().onOrAfter(TransportVersions.V_8_16_1)) {
            out.writeBoolean(skipInnerHits);
        }
    }
                while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
                    if (token == XContentParser.Token.FIELD_NAME) {
                        currentFieldName = parser.currentName();
                    } else if (token.isValue()) {
                        if (SCRIPT_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                            script = Script.parse(parser);
                        } else if (IGNORE_FAILURE_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                            ignoreFailure = parser.booleanValue();
                        } else {
                            throw new ParsingException(
                                parser.getTokenLocation(),
                                "Unknown key for a " + token + " in [" + currentFieldName + "].",
                                parser.getTokenLocation()
                            );
                        }
                    } else if (token == XContentParser.Token.START_OBJECT) {
                        if (SCRIPT_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                            script = Script.parse(parser);
                        } else {
                            throw new ParsingException(
                                parser.getTokenLocation(),
                                "Unknown key for a " + token + " in [" + currentFieldName + "].",
                                parser.getTokenLocation()
                            );
                        }
                    } else {
                        throw new ParsingException(
                            parser.getTokenLocation(),
                            "Unknown key for a " + token + " in [" + currentFieldName + "].",
                            parser.getTokenLocation()
                        );
                    }
                }
                this.ignoreFailure = ignoreFailure;
                this.fieldName = scriptFieldName;
                this.script = script;
            } else {
                throw new ParsingException(
                    parser.getTokenLocation(),
                    "Expected [" + XContentParser.Token.START_OBJECT + "] in [" + parser.currentName() + "] but found [" + token + "]",
                    parser.getTokenLocation()
                );
            }
        }

        public String fieldName() {
            return fieldName;
        }

        public Script script() {
            return script;
        }

        public boolean ignoreFailure() {
            return ignoreFailure;
        }

        @Override
        public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
            builder.startObject(fieldName);
            builder.field(SCRIPT_FIELD.getPreferredName(), script);
            builder.field(IGNORE_FAILURE_FIELD.getPreferredName(), ignoreFailure);
            builder.endObject();
            return builder;
        }

        @Override
        public int hashCode() {
            return Objects.hash(fieldName, script, ignoreFailure);
        }

        @Override
        public boolean equals(Object obj) {
            if (obj == null) {
                return false;
            }
            if (getClass() != obj.getClass()) {
                return false;
            }
            ScriptField other = (ScriptField) obj;
            return Objects.equals(fieldName, other.fieldName)
                && Objects.equals(script, other.script)
                && Objects.equals(ignoreFailure, other.ignoreFailure);
        }
    }

    @Override
    public int hashCode() {
        return Objects.hash(
            aggregations,
            explain,
            fetchSourceContext,
            fetchFields,
            docValueFields,
            storedFieldsContext,
            from,
            highlightBuilder,
            indexBoosts,
            minScore,
            subSearchSourceBuilders,
            postQueryBuilder,
            knnSearch,
            rankBuilder,
            rescoreBuilders,
            scriptFields,
            size,
            sorts,
            searchAfterBuilder,
            sliceBuilder,
            stats,
            suggestBuilder,
            terminateAfter,
            timeout,
            trackScores,
            version,
            seqNoAndPrimaryTerm,
            profile,
            extBuilders,
            collapse,
            trackTotalHitsUpTo,
            pointInTimeBuilder,
            runtimeMappings,
            skipInnerHits
        );
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (obj.getClass() != getClass()) {
            return false;
        }
        SearchSourceBuilder other = (SearchSourceBuilder) obj;
        return Objects.equals(aggregations, other.aggregations)
            && Objects.equals(explain, other.explain)
            && Objects.equals(fetchSourceContext, other.fetchSourceContext)
            && Objects.equals(fetchFields, other.fetchFields)
            && Objects.equals(docValueFields, other.docValueFields)
            && Objects.equals(storedFieldsContext, other.storedFieldsContext)
            && Objects.equals(from, other.from)
            && Objects.equals(highlightBuilder, other.highlightBuilder)
            && Objects.equals(indexBoosts, other.indexBoosts)
            && Objects.equals(minScore, other.minScore)
            && Objects.equals(subSearchSourceBuilders, other.subSearchSourceBuilders)
            && Objects.equals(postQueryBuilder, other.postQueryBuilder)
            && Objects.equals(knnSearch, other.knnSearch)
            && Objects.equals(rankBuilder, other.rankBuilder)
            && Objects.equals(rescoreBuilders, other.rescoreBuilders)
            && Objects.equals(scriptFields, other.scriptFields)
            && Objects.equals(size, other.size)
            && Objects.equals(sorts, other.sorts)
            && Objects.equals(searchAfterBuilder, other.searchAfterBuilder)
            && Objects.equals(sliceBuilder, other.sliceBuilder)
            && Objects.equals(stats, other.stats)
            && Objects.equals(suggestBuilder, other.suggestBuilder)
            && Objects.equals(terminateAfter, other.terminateAfter)
            && Objects.equals(timeout, other.timeout)
            && Objects.equals(trackScores, other.trackScores)
            && Objects.equals(version, other.version)
            && Objects.equals(seqNoAndPrimaryTerm, other.seqNoAndPrimaryTerm)
            && Objects.equals(profile, other.profile)
            && Objects.equals(extBuilders, other.extBuilders)
            && Objects.equals(collapse, other.collapse)
            && Objects.equals(trackTotalHitsUpTo, other.trackTotalHitsUpTo)
            && Objects.equals(pointInTimeBuilder, other.pointInTimeBuilder)
            && Objects.equals(runtimeMappings, other.runtimeMappings)
            && Objects.equals(skipInnerHits, other.skipInnerHits);
    }

    @Override
    public String toString() {
        return toString(EMPTY_PARAMS);
    }

    public String toString(Params params) {
        try {
            return XContentHelper.toXContent(this, XContentType.JSON, params, true).utf8ToString();
        } catch (IOException e) {
            throw new ElasticsearchException(e);
        }
    }

    public boolean supportsParallelCollection(ToLongFunction<String> fieldCardinality) {
        if (profile) return false;

        if (sorts != null) {
            // the implicit sorting is by _score, which supports parallel collection
            for (SortBuilder<?> sortBuilder : sorts) {
                if (sortBuilder.supportsParallelCollection() == false) return false;
            }
        }

        return collapse == null && (aggregations == null || aggregations.supportsParallelCollection(fieldCardinality));
    }

    private void validate() throws ValidationException {
        var exceptions = validate(null, false, false);
        if (exceptions != null) {
            throw exceptions;
        }
    }

    public ActionRequestValidationException validate(
        ActionRequestValidationException validationException,
        boolean isScroll,
        boolean allowPartialSearchResults
    ) {
        if (retriever() != null) {
            validationException = retriever().validate(this, validationException, isScroll, allowPartialSearchResults);
            List<String> specified = new ArrayList<>();
            if (subSearches().isEmpty() == false) {
                specified.add(QUERY_FIELD.getPreferredName());
            }
            if (knnSearch().isEmpty() == false) {
                specified.add(KNN_FIELD.getPreferredName());
            }
            if (searchAfter() != null) {
                specified.add(SEARCH_AFTER.getPreferredName());
            }
            if (terminateAfter() != DEFAULT_TERMINATE_AFTER) {
                specified.add(TERMINATE_AFTER_FIELD.getPreferredName());
            }
            if (sorts() != null) {
                specified.add(SORT_FIELD.getPreferredName());
            }
            if (rankBuilder() != null) {
                specified.add(RANK_FIELD.getPreferredName());
            }
            if (rescores() != null) {
                specified.add(RESCORE_FIELD.getPreferredName());
            }
            if (specified.isEmpty() == false) {
                validationException = addValidationError(
                    "cannot specify [" + RETRIEVER.getPreferredName() + "] and " + specified,
                    validationException
                );
            }
        }
        if (isScroll) {
            if (trackTotalHitsUpTo() != null && trackTotalHitsUpTo() != SearchContext.TRACK_TOTAL_HITS_ACCURATE) {
                validationException = addValidationError(
                    "disabling [track_total_hits] is not allowed in a scroll context",
                    validationException
                );
            }
            if (from() > 0) {
                validationException = addValidationError("using [from] is not allowed in a scroll context", validationException);
            }
            if (size() == 0) {
                validationException = addValidationError("[size] cannot be [0] in a scroll context", validationException);
            }
            if (rescores() != null && rescores().isEmpty() == false) {
                validationException = addValidationError("using [rescore] is not allowed in a scroll context", validationException);
            }
            if (CollectionUtils.isEmpty(searchAfter()) == false) {
                validationException = addValidationError("[search_after] cannot be used in a scroll context", validationException);
            }
            if (collapse() != null) {
                validationException = addValidationError("cannot use `collapse` in a scroll context", validationException);
            }
        }
        if (slice() != null) {
            if (pointInTimeBuilder() == null && (isScroll == false)) {
                validationException = addValidationError(
                    "[slice] can only be used with [scroll] or [point-in-time] requests",
                    validationException
                );
            }
        }
        if (from() > 0 && CollectionUtils.isEmpty(searchAfter()) == false) {
            validationException = addValidationError("[from] parameter must be set to 0 when [search_after] is used", validationException);
        }
        if (storedFields() != null) {
            if (storedFields().fetchFields() == false) {
                if (fetchSource() != null && fetchSource().fetchSource()) {
                    validationException = addValidationError(
                        "[stored_fields] cannot be disabled if [_source] is requested",
                        validationException
                    );
                }
                if (fetchFields() != null) {
                    validationException = addValidationError(
                        "[stored_fields] cannot be disabled when using the [fields] option",
                        validationException
                    );
                }
            }
        }
        if (subSearches().size() >= 2 && rankBuilder() == null) {
            validationException = addValidationError("[sub_searches] requires [rank]", validationException);
        }
        if (aggregations() != null) {
            validationException = aggregations().validate(validationException);
        }

        if (rankBuilder() != null) {
            int s = size() == -1 ? SearchService.DEFAULT_SIZE : size();
            if (s == 0) {
                validationException = addValidationError("[rank] requires [size] greater than [0]", validationException);
            }
            if (s > rankBuilder().rankWindowSize()) {
                validationException = addValidationError(
                    "[rank] requires [rank_window_size: "
                        + rankBuilder().rankWindowSize()
                        + "]"
                        + " be greater than or equal to [size: "
                        + s
                        + "]",
                    validationException
                );
            }
            int queryCount = subSearches().size() + knnSearch().size();
            if (rankBuilder().isCompoundBuilder() && queryCount < 2) {
                validationException = addValidationError(
                    "[rank] requires a minimum of [2] result sets using a combination of sub searches and/or knn searches",
                    validationException
                );
            }
            if (isScroll) {
                validationException = addValidationError("[rank] cannot be used in a scroll context", validationException);
            }
            if (rescores() != null && rescores().isEmpty() == false) {
                validationException = addValidationError("[rank] cannot be used with [rescore]", validationException);
            }

            if (suggest() != null && suggest().getSuggestions().isEmpty() == false) {
                validationException = addValidationError("[rank] cannot be used with [suggest]", validationException);
            }
        }

        if (rescores() != null) {
            for (@SuppressWarnings("rawtypes")
            var rescorer : rescores()) {
                validationException = rescorer.validate(this, validationException);
            }
        }
        return validationException;
    }
}
