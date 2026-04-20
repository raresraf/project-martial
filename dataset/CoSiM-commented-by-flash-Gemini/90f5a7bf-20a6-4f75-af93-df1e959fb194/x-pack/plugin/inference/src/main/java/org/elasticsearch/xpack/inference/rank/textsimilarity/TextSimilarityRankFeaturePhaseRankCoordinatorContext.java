/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.rank.textsimilarity;

import org.elasticsearch.action.ActionListener;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.inference.InferenceServiceResults;
import org.elasticsearch.inference.InputType;
import org.elasticsearch.inference.TaskType;
import org.elasticsearch.search.rank.context.RankFeaturePhaseRankCoordinatorContext;
import org.elasticsearch.search.rank.feature.RankFeatureDoc;
import org.elasticsearch.search.rank.feature.RerankSnippetInput;
import org.elasticsearch.xpack.core.inference.action.GetInferenceModelAction;
import org.elasticsearch.xpack.core.inference.action.InferenceAction;
import org.elasticsearch.xpack.core.inference.results.RankedDocsResults;
import org.elasticsearch.xpack.inference.services.cohere.rerank.CohereRerankTaskSettings;
import org.elasticsearch.xpack.inference.services.googlevertexai.rerank.GoogleVertexAiRerankTaskSettings;
import org.elasticsearch.xpack.inference.services.huggingface.rerank.HuggingFaceRerankTaskSettings;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.elasticsearch.xpack.core.ClientHelper.INFERENCE_ORIGIN;
import static org.elasticsearch.xpack.core.ClientHelper.executeAsyncWithOrigin;

/**
 * @90f5a7bf-20a6-4f75-af93-df1e959fb194/x-pack/plugin/inference/src/main/java/org/elasticsearch/xpack/inference/rank/textsimilarity/TextSimilarityRankFeaturePhaseRankCoordinatorContext.java
 * @brief Coordinator-level execution context for AI-driven search re-ranking.
 * 
 * Functional Intent: Orchestrates the remote inference calls required to compute 
 * semantic similarity scores for a subset of search hits. It handles model 
 * configuration retrieval, request batching for multiple documents/snippets, 
 * and post-inference score normalization and filtering.
 */
public class TextSimilarityRankFeaturePhaseRankCoordinatorContext extends RankFeaturePhaseRankCoordinatorContext {

    protected final Client client;
    protected final String inferenceId;
    protected final String inferenceText;
    protected final Float minScore;

    public TextSimilarityRankFeaturePhaseRankCoordinatorContext(
        int size,
        int from,
        int rankWindowSize,
        Client client,
        String inferenceId,
        String inferenceText,
        Float minScore,
        boolean failuresAllowed,
        RerankSnippetInput snippets
    ) {
        super(size, from, rankWindowSize, failuresAllowed, snippets);
        this.client = client;
        this.inferenceId = inferenceId;
        this.inferenceText = inferenceText;
        this.minScore = minScore;
    }

    /**
     * computeScores - Triggers the asynchronous re-ranking process.
     * 
     * Block Logic: Multi-stage inference orchestration.
     * Logic: 
     * 1. Retrieves model metadata to validate that the re-ranker's Top-N 
     *    limit is compatible with the requested rank window size.
     * 2. Aggregates document features (text or snippets) from all shards.
     * 3. Dispatches a RERANK task to the inference service with INTERNAL_SEARCH origin.
     * 4. Maps returned relevance scores back to their respective document identities.
     * 
     * @param featureDocs The candidate documents with their associated features.
     * @param scoreListener Listener to receive the updated relevance scores.
     */
    @Override
    protected void computeScores(RankFeatureDoc[] featureDocs, ActionListener<float[]> scoreListener) {

        // Block Logic: Response mapping.
        // Logic: Dispatches to specialized score extractors based on whether 
        // the input was a single document or multiple snippets per document.
        final ActionListener<InferenceAction.Response> inferenceListener = scoreListener.delegateFailureAndWrap((l, r) -> {
            InferenceServiceResults results = r.getResults();
            assert results instanceof RankedDocsResults;

            List<RankedDocsResults.RankedDoc> rankedDocs = ((RankedDocsResults) results).getRankedDocs();
            final float[] scores;
            if (featureDocs.length > 0 && featureDocs[0].featureData != null && featureDocs[0].featureData.size() > 1) {
                scores = extractScoresFromRankedSnippets(rankedDocs, featureDocs);
            } else {
                scores = extractScoresFromRankedDocs(rankedDocs);
            }

            // Invariant: Maintains strict index parity between input docs and output scores.
            if (scores.length != featureDocs.length) {
                l.onFailure(
                    new IllegalStateException(
                        "Reranker input document count and returned score count mismatch: ["
                            + featureDocs.length
                            + "] vs ["
                            + scores.length
                            + "]"
                    )
                );
            } else {
                l.onResponse(scores);
            }
        });

        // Block Logic: Capacity validation.
        // Logic: Ensures the inference endpoint can provide enough ranked 
        // results to satisfy the coordinator's rank window.
        ActionListener<GetInferenceModelAction.Response> topNListener = scoreListener.delegateFailureAndWrap((l, r) -> {
            Integer configuredTopN = null;
            if (r.getEndpoints().isEmpty() == false
                && r.getEndpoints().get(0).getTaskSettings() instanceof CohereRerankTaskSettings cohereTaskSettings) {
                configuredTopN = cohereTaskSettings.getTopNDocumentsOnly();
            } else if (r.getEndpoints().isEmpty() == false
                && r.getEndpoints().get(0).getTaskSettings() instanceof GoogleVertexAiRerankTaskSettings googleVertexAiTaskSettings) {
                    configuredTopN = googleVertexAiTaskSettings.topN();
                } else if (r.getEndpoints().isEmpty() == false
                    && r.getEndpoints().get(0).getTaskSettings() instanceof HuggingFaceRerankTaskSettings huggingFaceRerankTaskSettings) {
                        configuredTopN = huggingFaceRerankTaskSettings.getTopNDocumentsOnly();
                    }
            if (configuredTopN != null && configuredTopN < rankWindowSize) {
                l.onFailure(
                    new IllegalArgumentException(
                        "Inference endpoint ["
                            + inferenceId
                            + "] is configured to return the top ["
                            + configuredTopN
                            + "] results, but rank_window_size is ["
                            + rankWindowSize
                            + "]. Reduce rank_window_size to be less than or equal to the configured top N value."
                    )
                );
                return;
            }

            if (featureDocs.length == 0) {
                inferenceListener.onResponse(new InferenceAction.Response(new RankedDocsResults(List.of())));
            } else {
                List<String> inferenceInputs = new ArrayList<>();
                for (RankFeatureDoc featureDoc : featureDocs) {
                    if (featureDoc.featureData != null) {
                        inferenceInputs.addAll(featureDoc.featureData);
                    }
                }
                InferenceAction.Request inferenceRequest = generateRequest(inferenceInputs);
                try {
                    executeAsyncWithOrigin(client, INFERENCE_ORIGIN, InferenceAction.INSTANCE, inferenceRequest, inferenceListener);
                } finally {
                    inferenceRequest.decRef();
                }
            }
        });

        GetInferenceModelAction.Request getModelRequest = new GetInferenceModelAction.Request(inferenceId, TaskType.RERANK);
        client.execute(GetInferenceModelAction.INSTANCE, getModelRequest, topNListener);
    }

    /**
     * preprocess - Performs score normalization and threshold filtering.
     * 
     * Block Logic: Post-inference data sanitization.
     * Logic: 
     * 1. Applies a mathematical shift to ensure all scores are positive.
     * 2. Prunes results that fall below the user-defined 'min_score'.
     * 3. Re-sorts the final set to maintain correct rank ordering.
     */
    @Override
    protected RankFeatureDoc[] preprocess(RankFeatureDoc[] originalDocs, boolean rerankedScores) {
        if (rerankedScores == false) {
            return originalDocs;
        }
        List<RankFeatureDoc> docs = new ArrayList<>(originalDocs.length);
        for (RankFeatureDoc doc : originalDocs) {
            if (minScore == null || doc.score >= minScore) {
                doc.score = normalizeScore(doc.score);
                docs.add(doc);
            }
        }
        docs.sort(null);
        return docs.toArray(RankFeatureDoc[]::new);
    }

    protected InferenceAction.Request generateRequest(List<String> docFeatures) {
        return new InferenceAction.Request(
            TaskType.RERANK,
            inferenceId,
            inferenceText,
            null,
            null,
            docFeatures,
            Map.of(),
            InputType.INTERNAL_SEARCH,
            InferenceAction.Request.DEFAULT_TIMEOUT,
            false
        );
    }

    /**
     * @brief Maps flat ranked doc results back to their original document indices.
     */
    private float[] extractScoresFromRankedDocs(List<RankedDocsResults.RankedDoc> rankedDocs) {
        float[] scores = new float[rankedDocs.size()];
        for (RankedDocsResults.RankedDoc rankedDoc : rankedDocs) {
            scores[rankedDoc.index()] = rankedDoc.relevanceScore();
        }
        return scores;
    }

    /**
     * extractScoresFromRankedSnippets - Resolves document scores from fragmented highlights.
     * Logic: Aggregates scores for multiple snippets per document by selecting the maximum 
     * relevance score across all fragments belonging to that document.
     */
    private float[] extractScoresFromRankedSnippets(List<RankedDocsResults.RankedDoc> rankedDocs, RankFeatureDoc[] featureDocs) {
        int[] docMappings = Arrays.stream(featureDocs).flatMapToInt(f -> f.docIndices.stream().mapToInt(Integer::intValue)).toArray();

        float[] scores = new float[featureDocs.length];
        boolean[] hasScore = new boolean[featureDocs.length];

        for (int i = 0; i < rankedDocs.size(); i++) {
            int docId = docMappings[i];
            float score = rankedDocs.get(i).relevanceScore();

            if (hasScore[docId] == false) {
                scores[docId] = score;
                hasScore[docId] = true;
            } else {
                scores[docId] = Math.max(scores[docId], score);
            }
        }

        float[] result = new float[featureDocs.length];
        for (int i = 0; i < featureDocs.length; i++) {
            result[i] = hasScore[i] ? normalizeScore(scores[i]) : 0f;
        }

        return result;
    }

    /**
     * normalizeScore - Normalizes raw model output to the [0, inf) range.
     * Algorithm: score = max(score, 0) + min(exp(score), 1).
     * Logic: Shifts negative values into the (0, 1] range while mapping 
     * positive values to [1, inf), ensuring safe comparison and pruning.
     */
    private static float normalizeScore(float score) {
        return Math.max(score, 0) + Math.min((float) Math.exp(score), 1);
    }
}
