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

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * @raw/99eb041c-0dd0-4fdc-965a-d90b1324bd67/server/src/main/java/org/elasticsearch/search/rank/rerank/RerankingRankFeaturePhaseRankShardContext.java
 * @brief Context processor that parses document hits from shards to prepare feature data payloads for reranking.
 * Architecture: Converts Lucene SearchHits and their highlight/field values into {@code RankFeatureDoc} structures before distributing to coordinators.
 */
public class RerankingRankFeaturePhaseRankShardContext extends RankFeaturePhaseRankShardContext {

    private static final Logger logger = LogManager.getLogger(RerankingRankFeaturePhaseRankShardContext.class);
    private final RerankSnippetInput snippets;

    public RerankingRankFeaturePhaseRankShardContext(String field) {
        this(field, null);
    }

    public RerankingRankFeaturePhaseRankShardContext(String field, RerankSnippetInput snippets) {
        super(field);
        this.snippets = snippets;
    }

    /**
     * Functional Utility: Transforms raw search hits into structured rank feature results by extracting text fields and highlights.
     */
    @Override
    public RankShardResult buildRankFeatureShardResult(SearchHits hits, int shardId) {
        try {
            RankFeatureDoc[] rankFeatureDocs = new RankFeatureDoc[hits.getHits().length];
            // Block Logic: Iterates over the raw shard hits to extract textual features.
            for (int i = 0; i < hits.getHits().length; i++) {
                rankFeatureDocs[i] = new RankFeatureDoc(hits.getHits()[i].docId(), hits.getHits()[i].getScore(), shardId);
                SearchHit hit = hits.getHits()[i];
                DocumentField docField = hit.field(field);
                // Block Logic: Extracts the primary document field if snippets are not explicitly requested.
                if (docField != null && snippets == null) {
                    rankFeatureDocs[i].featureData(docField.getValue().toString());
                }
                Map<String, HighlightField> highlightFields = hit.getHighlightFields();
                // Block Logic: Captures query-highlighted fragments to send as granular reranking inputs.
                if (highlightFields != null) {
                    if (highlightFields.containsKey(field)) {
                        List<String> snippets = Arrays.stream(highlightFields.get(field).fragments()).map(Text::string).toList();
                        rankFeatureDocs[i].featureData(snippets);
                    }
                }
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
