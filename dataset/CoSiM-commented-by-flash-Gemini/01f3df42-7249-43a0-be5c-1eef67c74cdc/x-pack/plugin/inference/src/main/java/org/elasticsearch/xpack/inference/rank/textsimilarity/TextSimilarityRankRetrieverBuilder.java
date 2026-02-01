/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.rank.textsimilarity;

import org.apache.lucene.search.ScoreDoc;
import org.elasticsearch.features.NodeFeature;
import org.elasticsearch.index.query.MatchQueryBuilder;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.QueryRewriteContext;
import org.elasticsearch.license.LicenseUtils;
import org.elasticsearch.license.XPackLicenseState;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.search.rank.RankDoc;
import org.elasticsearch.search.rank.feature.RerankSnippetConfig;
import org.elasticsearch.search.retriever.CompoundRetrieverBuilder;
import org.elasticsearch.search.retriever.RetrieverBuilder;
import org.elasticsearch.search.retriever.RetrieverParserContext;
import org.elasticsearch.xcontent.ConstructingObjectParser;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.xpack.inference.queries.SemanticQueryBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.elasticsearch.search.rank.RankBuilder.DEFAULT_RANK_WINDOW_SIZE;
import static org.elasticsearch.xcontent.ConstructingObjectParser.constructorArg;
import static org.elasticsearch.xcontent.ConstructingObjectParser.optionalConstructorArg;
import static org.elasticsearch.xpack.inference.services.elasticsearch.ElasticsearchInternalService.DEFAULT_RERANK_ID;

/**
 * /**
 * @file TextSimilarityRankRetrieverBuilder.java
 * @brief Represents a builder for a text similarity reranker retriever in Elasticsearch.
 * This component is responsible for integrating text similarity ranking capabilities into search requests,
 * allowing for the reranking of search results based on semantic relevance derived from inference models.
 * It extends {@link CompoundRetrieverBuilder} to chain multiple retrieval and ranking stages.
 * Architectural Intent: Facilitates the use of machine learning inference models to improve search result relevance
 * by applying a secondary ranking pass on an initial set of retrieved documents.
 * It handles the parsing of XContent, construction of the reranker, and integration with the search source builder.
 */
public class TextSimilarityRankRetrieverBuilder extends CompoundRetrieverBuilder<TextSimilarityRankRetrieverBuilder> {

    // Functional Utility: NodeFeature flags to indicate support for specific text similarity reranker features.
    // These features ensure compatibility and proper behavior across different Elasticsearch node versions.
    public static final NodeFeature TEXT_SIMILARITY_RERANKER_ALIAS_HANDLING_FIX = new NodeFeature(
        "text_similarity_reranker_alias_handling_fix"
    );
    public static final NodeFeature TEXT_SIMILARITY_RERANKER_MINSCORE_FIX = new NodeFeature("text_similarity_reranker_minscore_fix");
    public static final NodeFeature TEXT_SIMILARITY_RERANKER_SNIPPETS = new NodeFeature("text_similarity_reranker_snippets");

    // Functional Utility: ParseField definitions for XContent parsing.
    // These constants define the field names used when serializing/deserializing the reranker builder.
    public static final ParseField RETRIEVER_FIELD = new ParseField("retriever");
    public static final ParseField INFERENCE_ID_FIELD = new ParseField("inference_id");
    public static final ParseField INFERENCE_TEXT_FIELD = new ParseField("inference_text");
    public static final ParseField FIELD_FIELD = new ParseField("field");
    public static final ParseField FAILURES_ALLOWED_FIELD = new ParseField("allow_rerank_failures");
    public static final ParseField SNIPPETS_FIELD = new ParseField("snippets");
    public static final ParseField NUM_SNIPPETS_FIELD = new ParseField("num_snippets");

    /**
     * Functional Utility: ConstructingObjectParser for {@code TextSimilarityRankRetrieverBuilder}.
     * This parser defines the structure for how a text similarity reranker is constructed from XContent,
     * mapping XContent fields to constructor arguments.
     */
    public static final ConstructingObjectParser<TextSimilarityRankRetrieverBuilder, RetrieverParserContext> PARSER =
        new ConstructingObjectParser<>(TextSimilarityRankBuilder.NAME, args -> {
            RetrieverBuilder retrieverBuilder = (RetrieverBuilder) args[0];
            String inferenceId = args[1] == null ? DEFAULT_RERANK_ID : (String) args[1];
            String inferenceText = (String) args[2];
            String field = (String) args[3];
            int rankWindowSize = args[4] == null ? DEFAULT_RANK_WINDOW_SIZE : (int) args[4];
            boolean failuresAllowed = args[5] != null && (Boolean) args[5];
            RerankSnippetConfig snippets = (RerankSnippetConfig) args[6];

            return new TextSimilarityRankRetrieverBuilder(
                retrieverBuilder,
                inferenceId,
                inferenceText,
                field,
                rankWindowSize,
                failuresAllowed,
                snippets
            );
        });

    /**
     * Functional Utility: ConstructingObjectParser for {@code RerankSnippetConfig}.
     * This parser specifically handles the 'snippets' configuration, allowing definition of
     * properties like the number of snippets to generate for reranking.
     */
    private static final ConstructingObjectParser<RerankSnippetConfig, RetrieverParserContext> SNIPPETS_PARSER =
        new ConstructingObjectParser<>(SNIPPETS_FIELD.getPreferredName(), true, args -> {
            Integer numSnippets = (Integer) args[0];
            return new RerankSnippetConfig(numSnippets);
        });

    static {
        // Block Logic: Register arguments for the main PARSER.
        PARSER.declareNamedObject(constructorArg(), (p, c, n) -> {
            RetrieverBuilder innerRetriever = p.namedObject(RetrieverBuilder.class, n, c);
            c.trackRetrieverUsage(innerRetriever.getName());
            return innerRetriever;
        }, RETRIEVER_FIELD);
        PARSER.declareString(optionalConstructorArg(), INFERENCE_ID_FIELD);
        PARSER.declareString(constructorArg(), INFERENCE_TEXT_FIELD);
        PARSER.declareString(constructorArg(), FIELD_FIELD);
        PARSER.declareInt(optionalConstructorArg(), RANK_WINDOW_SIZE_FIELD);
        PARSER.declareBoolean(optionalConstructorArg(), FAILURES_ALLOWED_FIELD);
        PARSER.declareObject(optionalConstructorArg(), SNIPPETS_PARSER, SNIPPETS_FIELD);
        // Block Logic: Register arguments for the SNIPPETS_PARSER.
        SNIPPETS_PARSER.declareInt(optionalConstructorArg(), NUM_SNIPPETS_FIELD);

        // Functional Utility: Declares base parser fields common to all RetrieverBuilders.
        RetrieverBuilder.declareBaseParserFields(PARSER);
    }

    /**
     * Functional Utility: Parses XContent to construct a {@code TextSimilarityRankRetrieverBuilder}.
     * This static method handles license checking before delegating to the parser.
     * @param parser The XContentParser to use for parsing.
     * @param context The RetrieverParserContext providing parsing context.
     * @param licenceState The current XPackLicenseState for feature checking.
     * @return A new {@code TextSimilarityRankRetrieverBuilder} instance.
     * @throws IOException If an I/O error occurs during parsing.
     */
    public static TextSimilarityRankRetrieverBuilder fromXContent(
        XContentParser parser,
        RetrieverParserContext context,
        XPackLicenseState licenceState
    ) throws IOException {
        // Precondition: Check if the text similarity reranker feature is allowed by the current license.
        if (TextSimilarityRankBuilder.TEXT_SIMILARITY_RERANKER_FEATURE.check(licenceState) == false) {
            throw LicenseUtils.newComplianceException(TextSimilarityRankBuilder.NAME);
        }
        return PARSER.apply(parser, context);
    }

    // Functional Utility: Stores the ID of the inference model to use for text similarity.
    private final String inferenceId;
    // Functional Utility: Stores the text to use for inference, e.g., the user query.
    private final String inferenceText;
    // Functional Utility: Stores the field in the document to apply text similarity against.
    private final String field;
    // Functional Utility: Indicates whether failures during reranking should be ignored.
    private final boolean failuresAllowed;
    // Functional Utility: Configuration for generating snippets for reranking.
    private final RerankSnippetConfig snippets;

    /**
     * Functional Utility: Constructs a new {@code TextSimilarityRankRetrieverBuilder}.
     * This constructor is typically used when building the retriever from parsed XContent,
     * defining the core parameters for the text similarity reranker.
     * @param retrieverBuilder The inner retriever builder that provides the initial set of documents.
     * @param inferenceId The ID of the inference model.
     * @param inferenceText The text for inference.
     * @param field The field to analyze for similarity.
     * @param rankWindowSize The number of documents to rerank.
     * @param failuresAllowed Whether to allow reranking failures.
     * @param snippets The snippet configuration for reranking.
     */
    public TextSimilarityRankRetrieverBuilder(
        RetrieverBuilder retrieverBuilder,
        String inferenceId,
        String inferenceText,
        String field,
        int rankWindowSize,
        boolean failuresAllowed,
        RerankSnippetConfig snippets
    ) {
        super(List.of(RetrieverSource.from(retrieverBuilder)), rankWindowSize);
        this.inferenceId = inferenceId;
        this.inferenceText = inferenceText;
        this.field = field;
        this.failuresAllowed = failuresAllowed;
        this.snippets = snippets;
    }

    /**
     * Functional Utility: Constructs a new {@code TextSimilarityRankRetrieverBuilder} with a more comprehensive set of parameters.
     * This constructor is used internally, especially during the rewrite process, to create a new builder
     * instance while preserving most of the current state, including minScore, retrieverName, and preFilterQueryBuilders.
     * @param retrieverSource A list of RetrieverSource objects providing documents.
     * @param inferenceId The ID of the inference model.
     * @param inferenceText The text for inference.
     * @param field The field to analyze for similarity.
     * @param rankWindowSize The number of documents to rerank.
     * @param minScore The minimum score for documents to be included after reranking.
     * @param failuresAllowed Whether to allow reranking failures.
     * @param retrieverName The name of the retriever.
     * @param preFilterQueryBuilders A list of QueryBuilders to apply as a pre-filter.
     * @param snippets The snippet configuration for reranking.
     */
    public TextSimilarityRankRetrieverBuilder(
        List<RetrieverSource> retrieverSource,
        String inferenceId,
        String inferenceText,
        String field,
        int rankWindowSize,
        Float minScore,
        boolean failuresAllowed,
        String retrieverName,
        List<QueryBuilder> preFilterQueryBuilders,
        RerankSnippetConfig snippets
    ) {
        super(retrieverSource, rankWindowSize);
        // Precondition: Ensure there is exactly one inner retriever.
        if (retrieverSource.size() != 1) {
            throw new IllegalArgumentException("[" + getName() + "] retriever should have exactly one inner retriever");
        }
        // Precondition: If snippets are configured, ensure numSnippets is positive.
        if (snippets != null && snippets.numSnippets() != null && snippets.numSnippets() < 1) {
            throw new IllegalArgumentException("num_snippets must be greater than 0, was: " + snippets.numSnippets());
        }
        this.inferenceId = inferenceId;
        this.inferenceText = inferenceText;
        this.field = field;
        this.minScore = minScore;
        this.failuresAllowed = failuresAllowed;
        this.retrieverName = retrieverName;
        this.preFilterQueryBuilders = preFilterQueryBuilders;
        this.snippets = snippets;
    }

    /**
     * Functional Utility: Creates a new instance of {@code TextSimilarityRankRetrieverBuilder} with updated child retrievers and pre-filter queries.
     * This method is part of the cloning mechanism for the builder, maintaining immutability while allowing modifications to internal components.
     * @param newChildRetrievers The updated list of child retrievers.
     * @param newPreFilterQueryBuilders The updated list of pre-filter query builders.
     * @return A new {@code TextSimilarityRankRetrieverBuilder} instance.
     */
    @Override
    protected TextSimilarityRankRetrieverBuilder clone(
        List<RetrieverSource> newChildRetrievers,
        List<QueryBuilder> newPreFilterQueryBuilders
    ) {
        return new TextSimilarityRankRetrieverBuilder(
            newChildRetrievers,
            inferenceId,
            inferenceText,
            field,
            rankWindowSize,
            minScore,
            failuresAllowed,
            retrieverName,
            newPreFilterQueryBuilders,
            snippets
        );
    }

    /**
     * Functional Utility: Combines and filters the results from the inner retriever.
     * This method applies the `minScore` filtering *after* the initial reranking, ensuring that
     * documents that might have been low-scoring initially but high-scoring after reranking are not prematurely excluded.
     * @param rankResults A list of ScoreDoc arrays from inner retrievers. Expects exactly one element.
     * @param explain A boolean indicating if explanations should be included in the RankDoc.
     * @return An array of {@code RankDoc} representing the combined and filtered results.
     */
    @Override
    protected RankDoc[] combineInnerRetrieverResults(List<ScoreDoc[]> rankResults, boolean explain) {
        assert rankResults.size() == 1; // Invariant: Only one inner retriever's results are expected.
        ScoreDoc[] scoreDocs = rankResults.getFirst();
        List<TextSimilarityRankDoc> filteredDocs = new ArrayList<>();
        // Block Logic: Filtering by min_score must be done here, after reranking.
        // Invariant: Documents with scores below minScore are excluded.
        // Precondition: minScore is non-null or scoreDoc.score is greater than or equal to minScore.
        // Applying min_score in the child retriever could prematurely exclude documents that would receive high scores from the reranker.
        for (int i = 0; i < scoreDocs.length; i++) {
            ScoreDoc scoreDoc = scoreDocs[i];
            assert scoreDoc.score >= 0;
            if (minScore == null || scoreDoc.score >= minScore) {
                // Conditional Logic: Include inference details if explanation is requested.
                if (explain) {
                    filteredDocs.add(new TextSimilarityRankDoc(scoreDoc.doc, scoreDoc.score, scoreDoc.shardIndex, inferenceId, field));
                } else {
                    filteredDocs.add(new TextSimilarityRankDoc(scoreDoc.doc, scoreDoc.score, scoreDoc.shardIndex));
                }
            }
        }
        return filteredDocs.toArray(new TextSimilarityRankDoc[0]);
    }

    /**
     * Functional Utility: Finalizes the {@code SearchSourceBuilder} by adding the text similarity rank builder.
     * This method integrates the specific text similarity ranking logic into the overall search request.
     * @param sourceBuilder The {@code SearchSourceBuilder} to finalize.
     * @return The finalized {@code SearchSourceBuilder}.
     */
    @Override
    protected SearchSourceBuilder finalizeSourceBuilder(SearchSourceBuilder sourceBuilder) {
        sourceBuilder.rankBuilder(
            new TextSimilarityRankBuilder(field, inferenceId, inferenceText, rankWindowSize, minScore, failuresAllowed, snippets)
        );
        return sourceBuilder;
    }

    /**
     * Functional Utility: Rewrites this {@code RetrieverBuilder} into its primitive form.
     * This process optimizes queries and transforms builders into their executable representations,
     * handling potential recursion and ensuring all components are in their final state before execution.
     * It specifically rewrites the snippet query builder if present.
     * @param ctx The QueryRewriteContext for rewriting queries.
     * @return The rewritten {@code RetrieverBuilder} instance.
     * @throws IOException If an I/O error occurs during rewriting.
     */
    @Override
    protected RetrieverBuilder doRewrite(QueryRewriteContext ctx) throws IOException {
        // Block Logic: Handle rewriting of snippet configuration if present.
        if (snippets != null) {
            QueryBuilder snippetQueryBuilder = snippets.snippetQueryBuilder();
            // Conditional Logic: If snippet query builder is null, create a new one with a default MatchQueryBuilder.
            if (snippetQueryBuilder == null) {
                return new TextSimilarityRankRetrieverBuilder(
                    innerRetrievers,
                    inferenceId,
                    inferenceText,
                    field,
                    rankWindowSize,
                    minScore,
                    failuresAllowed,
                    retrieverName,
                    preFilterQueryBuilders,
                    new RerankSnippetConfig(snippets.numSnippets(), new MatchQueryBuilder(field, inferenceText))
                );
            } else {
                // Conditional Logic: Rewrite the existing snippet query builder.
                QueryBuilder rewrittenSnippetQueryBuilder = snippetQueryBuilder.rewrite(ctx);
                // Invariant: If the snippet query builder was rewritten, return a new instance with the rewritten builder.
                if (snippetQueryBuilder != rewrittenSnippetQueryBuilder) {
                    return new TextSimilarityRankRetrieverBuilder(
                        innerRetrievers,
                        inferenceId,
                        inferenceText,
                        field,
                        rankWindowSize,
                        minScore,
                        failuresAllowed,
                        retrieverName,
                        preFilterQueryBuilders,
                        new RerankSnippetConfig(snippets.numSnippets(), rewrittenSnippetQueryBuilder)
                    );
                }
            }
        }

        return this; // Invariant: If no rewriting was needed, return the current instance.
    }

    /**
     * Functional Utility: Returns the name of this retriever.
     * @return The name of the retriever, which is {@code TextSimilarityRankBuilder.NAME}.
     */
    @Override
    public String getName() {
        return TextSimilarityRankBuilder.NAME;
    }

    /**
     * Functional Utility: Returns the inference ID used by this reranker.
     * @return The inference model ID.
     */
    public String inferenceId() {
        return inferenceId;
    }

    /**
     * Functional Utility: Returns whether reranking failures are allowed.
     * @return True if failures are allowed, false otherwise.
     */
    public boolean failuresAllowed() {
        return failuresAllowed;
    }

    /**
     * Functional Utility: Converts the {@code TextSimilarityRankRetrieverBuilder} to its XContent representation.
     * This method serializes the builder's properties into an XContentBuilder, which can then be used
     * for communication with Elasticsearch.
     * @param builder The XContentBuilder to write to.
     * @param params The parameters for XContent generation.
     * @throws IOException If an I/O error occurs during XContent generation.
     */
    @Override
    protected void doToXContent(XContentBuilder builder, Params params) throws IOException {
        builder.field(RETRIEVER_FIELD.getPreferredName(), innerRetrievers.getFirst().retriever());
        builder.field(INFERENCE_ID_FIELD.getPreferredName(), inferenceId);
        builder.field(INFERENCE_TEXT_FIELD.getPreferredName(), inferenceText);
        builder.field(FIELD_FIELD.getPreferredName(), field);
        builder.field(RANK_WINDOW_SIZE_FIELD.getPreferredName(), rankWindowSize);
        // Conditional Logic: Only write failuresAllowed field if it's true.
        if (failuresAllowed) {
            builder.field(FAILURES_ALLOWED_FIELD.getPreferredName(), failuresAllowed);
        }
        // Conditional Logic: Write snippets configuration if present.
        if (snippets != null) {
            builder.startObject(SNIPPETS_FIELD.getPreferredName());
            // Conditional Logic: Write numSnippets field if present.
            if (snippets.numSnippets() != null) {
                builder.field(NUM_SNIPPETS_FIELD.getPreferredName(), snippets.numSnippets());
            }
            builder.endObject();
        }
    }

    /**
     * Functional Utility: Compares this {@code TextSimilarityRankRetrieverBuilder} with another object for equality.
     * Equality is determined by comparing all relevant fields, including inherited fields.
     * @param other The object to compare with.
     * @return True if the objects are equal, false otherwise.
     */
    @Override
    public boolean doEquals(Object other) {
        TextSimilarityRankRetrieverBuilder that = (TextSimilarityRankRetrieverBuilder) other;
        return super.doEquals(other)
            && Objects.equals(inferenceId, that.inferenceId)
            && Objects.equals(inferenceText, that.inferenceText)
            && Objects.equals(field, that.field)
            && rankWindowSize == that.rankWindowSize
            && Objects.equals(minScore, that.minScore)
            && failuresAllowed == that.failuresAllowed
            && Objects.equals(snippets, that.snippets);
    }

    /**
     * Functional Utility: Generates a hash code for this {@code TextSimilarityRankRetrieverBuilder}.
     * The hash code is based on all relevant fields to ensure consistency with the {@code equals} method.
     * @return The hash code.
     */
    @Override
    public int doHashCode() {
        return Objects.hash(inferenceId, inferenceText, field, rankWindowSize, minScore, failuresAllowed, snippets);
    }
}