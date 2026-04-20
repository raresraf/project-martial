/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.rank.rerank;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.common.document.DocumentField;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.SearchHits;
import org.elasticsearch.search.fetch.subphase.highlight.HighlightField;
import org.elasticsearch.search.rank.RankShardResult;
import org.elasticsearch.search.rank.context.RankFeaturePhaseRankShardContext;
import org.elasticsearch.search.rank.feature.RankFeatureDoc;
import org.elasticsearch.search.rank.feature.RankFeatureShardResult;
import org.elasticsearch.search.rank.feature.RerankSnippetInput;
import org.elasticsearch.xcontent.Text;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * @9020c670-79a4-462c-bc71-13e0071c8621/server/src/main/java/org/elasticsearch/search/rank/rerank/RerankingRankFeaturePhaseRankShardContext.java
 * @brief Shard-level execution context for extracting re-ranking features.
 * 
 * Functional Intent: Orchestrates the extraction of raw text or highlighted 
 * snippets from search hits to be used as features in the re-ranking phase. 
 * It transforms Lucene-level search results into specialized RankFeatureDoc 
 * containers, handling both direct field access and high-level snippet highlighting.
 */
public class RerankingRankFeaturePhaseRankShardContext extends RankFeaturePhaseRankShardContext {

    private static final Logger logger = LogManager.getLogger(RerankingRankFeaturePhaseRankShardContext.class);
    private final RerankSnippetInput snippets;

    public RerankingRankFeaturePhaseRankShardContext(String field) {
        this(field, null);
    }

    /**
     * @brief Constructs a re-ranking context.
     * @param field The source field for feature extraction.
     * @param snippets Optional configuration for snippet-based feature extraction.
     */
    public RerankingRankFeaturePhaseRankShardContext(String field, RerankSnippetInput snippets) {
        super(field);
        this.snippets = snippets;
    }

    /**
     * Block Logic: Post-search feature extraction.
     * Logic: 
     * 1. Iterates through search hits produced by the initial retrieval phase.
     * 2. If snippets are disabled: Extracts raw field values from the document.
     * 3. If highlighting is present: Collects generated fragments and associates 
     *    them with the document index for coordinate-level re-ranking.
     * 4. Invariant: Propagates shard metadata and document scores into the final result.
     * 
     * @param hits The initial set of search hits from the shard.
     * @param shardId Identifier of the source shard.
     * @return A consolidated RankFeatureShardResult containing extracted features.
     */
    @Override
    public RankShardResult buildRankFeatureShardResult(SearchHits hits, int shardId) {
        try {
            RankFeatureDoc[] rankFeatureDocs = new RankFeatureDoc[hits.getHits().length];
            int docIndex = 0;
            for (int i = 0; i < hits.getHits().length; i++) {
                rankFeatureDocs[i] = new RankFeatureDoc(hits.getHits()[i].docId(), hits.getHits()[i].getScore(), shardId);
                SearchHit hit = hits.getHits()[i];
                
                // Block Logic: Raw field value extraction.
                DocumentField docField = hit.field(field);
                if (docField != null && snippets == null) {
                    rankFeatureDocs[i].featureData(List.of(docField.getValue().toString()));
                }

                // Block Logic: Snippet-based feature extraction (Highlighting).
                // Logic: Maps pre-calculated highlights to the feature data buffer, 
                // tracking the relative document index for multi-snippet alignment.
                Map<String, HighlightField> highlightFields = hit.getHighlightFields();
                if (highlightFields != null) {
                    if (highlightFields.containsKey(field)) {
                        List<String> snippets = Arrays.stream(highlightFields.get(field).fragments()).map(Text::string).toList();
                        List<Integer> docIndices = new ArrayList<>();
                        for (String s : snippets) {
                            docIndices.add(docIndex);
                        }
                        rankFeatureDocs[i].featureData(snippets);
                        rankFeatureDocs[i].docIndices(docIndices);
                    }
                }
                docIndex++;
            }
            return new RankFeatureShardResult(rankFeatureDocs);
        } catch (Exception ex) {
            // Block Logic: Graceful failure handling.
            // Logic: Logs extraction failures while returning null to avoid crashing 
            // the entire search request; re-ranking will be skipped for this shard.
            logger.warn(
                "Error while fetching feature data for {field: ["
                    + field
                    + "]} and {docids: ["
                    + Arrays.stream(hits.getHits()).map(SearchHit::docId).toList()
                    + "]}.",
                ex
            );
            return null;
        }
    }
}
