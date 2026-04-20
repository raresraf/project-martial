/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.rank.feature;

import org.apache.lucene.search.Explanation;
import org.elasticsearch.TransportVersions;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.search.rank.RankDoc;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

/**
 * @99eb041c-0dd0-4fdc-965a-d90b1324bd67/server/src/main/java/org/elasticsearch/search/rank/feature/RankFeatureDoc.java
 * @brief Specialized RankDoc container for transporting feature data during search re-ranking.
 * 
 * Functional Intent: Extends the base RankDoc to include raw feature values and 
 * associated document indices. This data is extracted at the shard level and 
 * transmitted to the coordinator node, where it serves as input for complex 
 * re-ranking algorithms (e.g., LTR or learning-to-rank models).
 */
public class RankFeatureDoc extends RankDoc {

    public static final String NAME = "rank_feature_doc";

    /**
     * Functional Utility: Buffers for feature extraction.
     * featureData: Raw string representations of extracted document features.
     * docIndices: Mapping of features to their original document context.
     */
    public List<String> featureData;
    public List<Integer> docIndices;

    public RankFeatureDoc(int doc, float score, int shardIndex) {
        super(doc, score, shardIndex);
    }

    /**
     * Block Logic: Version-aware deserialization.
     * Logic: Handles legacy single-string feature data for older transport versions 
     * while supporting efficient collection streaming for newer versions (RERANK_SNIPPETS).
     */
    public RankFeatureDoc(StreamInput in) throws IOException {
        super(in);
        if (in.getTransportVersion().onOrAfter(TransportVersions.RERANK_SNIPPETS)) {
            featureData = in.readOptionalStringCollectionAsList();
            docIndices = in.readOptionalCollectionAsList(StreamInput::readVInt);
        } else {
            String featureDataString = in.readOptionalString();
            featureData = featureDataString == null ? null : List.of(featureDataString);
        }
    }

    @Override
    public Explanation explain(Explanation[] sources, String[] queryNames) {
        throw new UnsupportedOperationException("explain is not supported for {" + getClass() + "}");
    }

    public void featureData(List<String> featureData) {
        this.featureData = featureData;
    }

    public void docIndices(List<Integer> docIndices) {
        this.docIndices = docIndices;
    }

    /**
     * Block Logic: Version-aware serialization.
     * Logic: Maintains backwards compatibility by down-sampling multi-feature lists 
     * to a single element when communicating with older cluster nodes.
     */
    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        if (out.getTransportVersion().onOrAfter(TransportVersions.RERANK_SNIPPETS)) {
            out.writeOptionalStringCollection(featureData);
            out.writeOptionalCollection(docIndices, StreamOutput::writeVInt);
        } else {
            // Functional Utility: Legacy fallback (first feature only).
            out.writeOptionalString(featureData.get(0));
        }
    }

    @Override
    protected boolean doEquals(RankDoc rd) {
        RankFeatureDoc other = (RankFeatureDoc) rd;
        return Objects.equals(this.featureData, other.featureData) && Objects.equals(this.docIndices, other.docIndices);
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(featureData, docIndices);
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    /**
     * Block Logic: JSON serialization for diagnostic endpoints.
     */
    @Override
    protected void doToXContent(XContentBuilder builder, Params params) throws IOException {
        builder.array("featureData", featureData);
        builder.array("docIndices", docIndices);
    }
}
