/**
 * A {@code RankBuilder} that enables ranking with text similarity model inference. Supports parameters for configuring the inference call.
 *
 * This class extends the abstract {@link RankBuilder} to provide a concrete implementation
 * for re-ranking search results based on text similarity models. It allows users to specify
 * an inference model ID, the text to be used for inference, the target field for comparison,
 * minimum score thresholds, and options for handling inference failures.
 *
 * Architectural Intent: To integrate machine learning-based text similarity ranking directly
 * into Elasticsearch's search capabilities, enabling more sophisticated relevance judgments
 * beyond traditional keyword matching. It leverages external inference models for scoring documents.
 */
public class TextSimilarityRankBuilder extends RankBuilder {

    public static final String NAME = "text_similarity_reranker";
    // Functional Utility: Defines the unique name for this text similarity reranker.

    /**
     * The default token size limit of the Elastic reranker.
     */
    private static final int RERANK_TOKEN_SIZE_LIMIT = 512;
    // Functional Utility: Specifies the maximum token size limit for the Elastic reranker model.

    /**
     * A safe default token size limit for other reranker models.
     * Reranker models with smaller token limits will be truncated.
     */
    private static final int DEFAULT_TOKEN_SIZE_LIMIT = 4096;
    // Functional Utility: Provides a safe default token size limit for other reranker models.

    public static final LicensedFeature.Momentary TEXT_SIMILARITY_RERANKER_FEATURE = LicensedFeature.momentary(
        null,
        "text-similarity-reranker",
        License.OperationMode.ENTERPRISE
    );
    // Functional Utility: Defines the licensing feature for the text similarity reranker, indicating it's an Enterprise feature.

    private final String inferenceId; // The ID of the inference model to use for text similarity.
    private final String inferenceText; // The input text for the inference model.
    private final String field; // The target field in the document to compare against the inference text.
    private final Float minScore; // The minimum score required for a document to be included after re-ranking.
    private final boolean failuresAllowed; // Flag indicating whether inference failures should be allowed (true) or result in an error (false).
    private final RerankSnippetConfig snippets; // Configuration for generating snippets from re-ranked results.

    public TextSimilarityRankBuilder(
        String field,
        String inferenceId,
        String inferenceText,
        int rankWindowSize,
        Float minScore,
        boolean failuresAllowed,
        RerankSnippetConfig snippets
    ) {
        // Functional Utility: Initializes a new TextSimilarityRankBuilder instance with all necessary parameters
        // for configuring text similarity-based ranking.
        super(rankWindowSize);
        this.inferenceId = inferenceId;
        this.inferenceText = inferenceText;
        this.field = field;
        this.minScore = minScore;
        this.failuresAllowed = failuresAllowed;
        this.snippets = snippets;
    }

    public TextSimilarityRankBuilder(StreamInput in) throws IOException {
        // Functional Utility: Reconstructs a TextSimilarityRankBuilder instance from a StreamInput.
        // This constructor handles deserialization of all text similarity ranking parameters,
        // including version-dependent reading for 'failuresAllowed' and 'snippets'.
        super(in);
        // rankWindowSize deserialization is handled by the parent class RankBuilder
        this.inferenceId = in.readString();
        this.inferenceText = in.readString();
        this.field = in.readString();
        this.minScore = in.readOptionalFloat();
        // Conditional Logic: Read 'failuresAllowed' field based on transport version.
        if (in.getTransportVersion().isPatchFrom(TransportVersions.RERANKER_FAILURES_ALLOWED_8_19)
            || in.getTransportVersion().onOrAfter(TransportVersions.RERANKER_FAILURES_ALLOWED)) {
            this.failuresAllowed = in.readBoolean();
        } else {
            this.failuresAllowed = false;
        }
        // Conditional Logic: Read 'snippets' field based on transport version.
        if (in.getTransportVersion().onOrAfter(TransportVersions.RERANK_SNIPPETS)) {
            this.snippets = in.readOptionalWriteable(RerankSnippetConfig::new);
        } else {
            this.snippets = null;
        }
    }

    @Override
    public String getWriteableName() {
        // Functional Utility: Returns the unique name used to identify this RankBuilder in a writeable context.
        return NAME;
    }

    @Override
    public TransportVersion getMinimalSupportedVersion() {
        // Functional Utility: Returns the minimal transport version supported by this RankBuilder.
        // This ensures compatibility across different Elasticsearch versions during serialization/deserialization.
        return TransportVersions.V_8_15_0;
    }

    @Override
    public void doWriteTo(StreamOutput out) throws IOException {
        // Functional Utility: Writes the state of the TextSimilarityRankBuilder to a StreamOutput.
        // This method handles serialization of all text similarity ranking parameters,
        // including version-dependent writing for 'failuresAllowed' and 'snippets'.
        // rankWindowSize serialization is handled by the parent class RankBuilder
        out.writeString(inferenceId);
        out.writeString(inferenceText);
        out.writeString(field);
        out.writeOptionalFloat(minScore);
        // Conditional Logic: Write 'failuresAllowed' field based on transport version.
        if (out.getTransportVersion().isPatchFrom(TransportVersions.RERANKER_FAILURES_ALLOWED_8_19)
            || out.getTransportVersion().onOrAfter(TransportVersions.RERANKER_FAILURES_ALLOWED)) {
            out.writeBoolean(failuresAllowed);
        }
        // Conditional Logic: Write 'snippets' field based on transport version.
        if (out.getTransportVersion().onOrAfter(TransportVersions.RERANK_SNIPPETS)) {
            out.writeOptionalWriteable(snippets);
        }
    }

    @Override
    public void doXContent(XContentBuilder builder, Params params) throws IOException {
        // Functional Utility: Converts the TextSimilarityRankBuilder instance into XContent (e.g., JSON).
        // This method writes the text similarity ranking configuration, including inference details and thresholds.
        // this object is not parsed, but it sometimes needs to be output as xcontent
        // rankWindowSize serialization is handled by the parent class RankBuilder
        builder.field(INFERENCE_ID_FIELD.getPreferredName(), inferenceId);
        builder.field(INFERENCE_TEXT_FIELD.getPreferredName(), inferenceText);
        builder.field(FIELD_FIELD.getPreferredName(), field);
        // Conditional Logic: Write minScore if it is not null.
        if (minScore != null) {
            builder.field(MIN_SCORE_FIELD.getPreferredName(), minScore);
        }
        // Conditional Logic: Write failuresAllowed if it is true.
        if (failuresAllowed) {
            builder.field(FAILURES_ALLOWED_FIELD.getPreferredName(), true);
        }
        // Conditional Logic: Write snippets if present.
        if (snippets != null) {
            builder.field(SNIPPETS_FIELD.getPreferredName(), snippets);
        }
    }

    @Override
    public boolean isCompoundBuilder() {
        // Functional Utility: Indicates that this TextSimilarityRankBuilder is not a compound builder.
        // It typically involves a single inference call rather than multiple chained queries or ranking steps.
        return false;
    }

    @Override
    public Explanation explainHit(Explanation baseExplanation, RankDoc scoreDoc, List<String> queryNames) {
        // Functional Utility: Generates a human-readable explanation of the score for a document
        // after being re-ranked by the text similarity model.
        // Conditional Logic: Return base explanation if scoreDoc is null or if base explanation is not a match.
        if (scoreDoc == null) {
            return baseExplanation;
        }
        if (false == baseExplanation.isMatch()) {
            return baseExplanation;
        }

        // Precondition: scoreDoc must be an instance of RankFeatureDoc for this explanation.
        assert scoreDoc instanceof RankFeatureDoc : "ScoreDoc is not an instance of RankFeatureDoc";
        RankFeatureDoc rrfRankDoc = (RankFeatureDoc) scoreDoc;

        // Block Logic: Construct a detailed explanation including rank, score, inference ID, and field.
        return Explanation.match(
            rrfRankDoc.score,
            "rank after reranking: ["
                + rrfRankDoc.rank
                + "] with score: ["
                + rrfRankDoc.score
                + "], using inference endpoint: ["
                + inferenceId
                + "] on document field: ["
                + field
                + "]",
            baseExplanation
        );
    }

    @Override
    public QueryPhaseRankShardContext buildQueryPhaseShardContext(List<Query> queries, int from) {
        // Functional Utility: Returns null, indicating that this RankBuilder does not participate
        // in the query phase at the shard level directly. Its logic is primarily in the RankFeature phase.
        return null;
    }

    @Override
    public QueryPhaseRankCoordinatorContext buildQueryPhaseCoordinatorContext(int size, int from) {
        // Functional Utility: Returns null, indicating that this RankBuilder does not participate
        // in coordinating query phase results. Its logic is primarily in the RankFeature phase.
        return null;
    }

    @Override
    public RankFeaturePhaseRankShardContext buildRankFeaturePhaseShardContext() {
        // Functional Utility: Builds a context for executing the rank feature phase on a shard.
        // This context prepares the necessary data for text similarity re-ranking at the shard level.
        return new RerankingRankFeaturePhaseRankShardContext(field, snippets);
    }

    @Override
    public RankFeaturePhaseRankCoordinatorContext buildRankFeaturePhaseCoordinatorContext(int size, int from, Client client) {
        // Functional Utility: Builds a context for coordinating the rank feature phase results globally.
        // This context combines shard-level inference results and applies global re-ranking logic.
        return new TextSimilarityRankFeaturePhaseRankCoordinatorContext(
            size,
            from,
            rankWindowSize(),
            client,
            inferenceId,
            inferenceText,
            minScore,
            failuresAllowed,
            snippets != null ? new SnippetRankInput(snippets, inferenceText, tokenSizeLimit()) : null
        );
    }

    /**
     * @return The token size limit to apply to this rerank context.
     * This is not yet available so we are hardcoding it for now.
     */
    public Integer tokenSizeLimit() {
        // Functional Utility: Determines the token size limit to apply for the reranker.
        // This limit is chosen based on the specific inference model being used (Elastic reranker vs. other models).
        // Conditional Logic: If using the default Elastic reranker ID, apply its specific limit.
        if (inferenceId.equals(DEFAULT_RERANK_ID) || inferenceId.equals(RERANKER_ID)) {
            return RERANK_TOKEN_SIZE_LIMIT;
        }

        // Block Logic: Otherwise, apply the general default token size limit.
        return DEFAULT_TOKEN_SIZE_LIMIT;
    }

    public String field() {
        // Functional Utility: Returns the target field name for text similarity comparison.
        return field;
    }

    public String inferenceId() {
        // Functional Utility: Returns the ID of the inference model used for text similarity.
        return inferenceId;
    }

    public String inferenceText() {
        // Functional Utility: Returns the input text used for the inference model.
        return inferenceText;
    }

    public Float minScore() {
        // Functional Utility: Returns the minimum score threshold after re-ranking.
        return minScore;
    }

    public boolean failuresAllowed() {
        // Functional Utility: Returns true if inference failures are allowed, false otherwise.
        return failuresAllowed;
    }

    @Override
    protected boolean doEquals(RankBuilder other) {
        // Functional Utility: Compares subclass-specific fields of two TextSimilarityRankBuilder instances for equality.
        // It checks the equality of inference ID, inference text, field, minimum score, and failures allowed flag.
        TextSimilarityRankBuilder that = (TextSimilarityRankBuilder) other;
        return Objects.equals(inferenceId, that.inferenceId)
            && Objects.equals(inferenceText, that.inferenceText)
            && Objects.equals(field, that.field)
            && Objects.equals(minScore, that.minScore)
            && failuresAllowed == that.failuresAllowed;
    }

    @Override
    protected int doHashCode() {
        // Functional Utility: Computes a subclass-specific hash code for TextSimilarityRankBuilder.
        // It combines the hash codes of inference ID, inference text, field, minimum score, and failures allowed flag.
        return Objects.hash(inferenceId, inferenceText, field, minScore, failuresAllowed);
    }
}
