/**
 * @raw/ad140ded-80c4-4d29-bdb6-f90ea8fbeed6/server/src/main/java/org/elasticsearch/search/rank/rerank/RerankingRankFeaturePhaseRankShardContext.java
 * @brief Core functionality implementation.
 * Intent: Execute functional units and state management.
 * Algorithm: Iterative or sequential execution logic.
 * Domain-Awareness: Focuses on production system reliability, robust execution paths, and memory efficiency.
 */
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
import org.elasticsearch.xcontent.Text;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * The {@code ReRankingRankFeaturePhaseRankShardContext} is handles the {@code SearchHits} generated from the {@code RankFeatureShardPhase}
 * and builds the {@code RankFeatureShardResult} for the reranking phase, by reading the field info for the specified {@code field} during
 * construction.
 */
public class RerankingRankFeaturePhaseRankShardContext extends RankFeaturePhaseRankShardContext {

    private static final Logger logger = LogManager.getLogger(RerankingRankFeaturePhaseRankShardContext.class);

    public RerankingRankFeaturePhaseRankShardContext(String field) {
        super(field);
    }

    @Override
    public RankShardResult buildRankFeatureShardResult(SearchHits hits, int shardId) {
        try {
            RankFeatureDoc[] rankFeatureDocs = new RankFeatureDoc[hits.getHits().length];
            int docIndex = 0;
            /**
             * Block Logic: Orchestrates the temporal progression of the iteration.
             * Invariant: At the start of each iteration, loop structures maintain boundary and locality.
             */
            for (int i = 0; i < hits.getHits().length; i++) {
                rankFeatureDocs[i] = new RankFeatureDoc(hits.getHits()[i].docId(), hits.getHits()[i].getScore(), shardId);
                SearchHit hit = hits.getHits()[i];
                DocumentField docField = hit.field(field);
                /**
                 * Block Logic: Conditional evaluation for divergent control flow.
                 * Invariant: Taken branch maintains control flow invariants.
                 */
                if (docField != null) {
                    rankFeatureDocs[i].featureData(docField.getValue().toString());
                }
                Map<String, HighlightField> highlightFields = hit.getHighlightFields();
                /**
                 * Block Logic: Conditional evaluation for divergent control flow.
                 * Invariant: Taken branch maintains control flow invariants.
                 */
                if (highlightFields != null) {
                    /**
                     * Block Logic: Conditional evaluation for divergent control flow.
                     * Invariant: Taken branch maintains control flow invariants.
                     */
                    if (highlightFields.containsKey(field)) {
                        List<String> snippets = Arrays.stream(highlightFields.get(field).fragments()).map(Text::string).toList();
                        List<Integer> docIndices = new ArrayList<>();
                        /**
                         * Block Logic: Orchestrates the temporal progression of the iteration.
                         * Invariant: At the start of each iteration, loop structures maintain boundary and locality.
                         */
                        for (String snippet : snippets) {
                            docIndices.add(docIndex);
                        }
                        rankFeatureDocs[i].snippets(snippets);
                        rankFeatureDocs[i].docIndices(docIndices);
                    }
                }
                docIndex++;
            }
            return new RankFeatureShardResult(rankFeatureDocs);
        } catch (Exception ex) {
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
