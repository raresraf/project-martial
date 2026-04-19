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
 * @raw/99eb041c-0dd0-4fdc-965a-d90b1324bd67/server/src/main/java/org/elasticsearch/search/rank/feature/RankFeatureDoc.java
 * @brief Represents a ranking document enriched with feature data (like text snippets) destined for the reranking inference phase.
 * Architecture: Inherits from {@link RankDoc} and acts as a transport-friendly DTO carrying unstructured text arrays to be scored.
 */
public class RankFeatureDoc extends RankDoc {

    public static final String NAME = "rank_feature_doc";

    // TODO: update to support more than 1 fields; and not restrict to string data
    public List<String> featureData;

    /**
     * Functional Utility: Instantiates a basic ranking document with initial score and shard routing metadata.
     */
    public RankFeatureDoc(int doc, float score, int shardIndex) {
        super(doc, score, shardIndex);
    }

    /**
     * Functional Utility: Deserializes the document over the network, providing backward compatibility for pre-snippet versions.
     */
    public RankFeatureDoc(StreamInput in) throws IOException {
        super(in);
        // Block Logic: Checks transport protocol version to conditionally read multi-snippet collections.
        if (in.getTransportVersion().onOrAfter(TransportVersions.RERANK_SNIPPETS)) {
            featureData = in.readOptionalStringCollectionAsList();
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

    public void featureData(String featureData) {
        this.featureData = List.of(featureData);
    }

    /**
     * Functional Utility: Serializes the document for network transit, handling version-dependent snippet formatting.
     */
    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        // Block Logic: Ensures older nodes receive only a single text snippet to prevent protocol breakdown.
        if (out.getTransportVersion().onOrAfter(TransportVersions.RERANK_SNIPPETS)) {
            out.writeOptionalStringCollection(featureData);
        } else {
            out.writeOptionalString(featureData.get(0));
        }
    }

    @Override
    protected boolean doEquals(RankDoc rd) {
        RankFeatureDoc other = (RankFeatureDoc) rd;
        return Objects.equals(this.featureData, other.featureData);
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(featureData);
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    protected void doToXContent(XContentBuilder builder, Params params) throws IOException {
        builder.array("featureData", featureData);
    }
}
