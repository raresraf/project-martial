/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.rank;

import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.Query;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.VersionedNamedWriteable;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.core.UpdateForV10;
import org.elasticsearch.features.NodeFeature;
import org.elasticsearch.search.SearchService;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.search.rank.context.QueryPhaseRankCoordinatorContext;
import org.elasticsearch.search.rank.context.QueryPhaseRankShardContext;
import org.elasticsearch.search.rank.context.RankFeaturePhaseRankCoordinatorContext;
import org.elasticsearch.search.rank.context.RankFeaturePhaseRankShardContext;
import org.elasticsearch.search.retriever.RetrieverBuilder;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.function.Predicate;

/**
 * {@code RankBuilder} is used as a base class to manage input, parsing, and subsequent generation of appropriate contexts
 * for handling searches that require multiple queries and/or ranking steps for global rank relevance.
 *
 * This abstract class serves as a blueprint for implementing custom ranking strategies within Elasticsearch.
 * It provides an interface for defining how search queries are executed in phases, how results from different
 * shards are coordinated, and how final relevance scores are explained.
 *
 * Architectural Intent: To abstract the complexities of multi-stage search execution and distributed ranking,
 * allowing for flexible and pluggable ranking algorithms. It supports both query-phase ranking and a dedicated
 * rank feature phase for more advanced relevance models.
 */
public abstract class RankBuilder implements VersionedNamedWriteable, ToXContentObject {

    public static final ParseField RANK_WINDOW_SIZE_FIELD = new ParseField("rank_window_size");

    public static final int DEFAULT_RANK_WINDOW_SIZE = SearchService.DEFAULT_SIZE;

    private final int rankWindowSize;

    public RankBuilder(int rankWindowSize) {
        // Functional Utility: Initializes a new RankBuilder instance with a specified rank window size.
        // The rank window size determines the number of top hits considered for re-ranking or further processing.
        this.rankWindowSize = rankWindowSize;
    }

    public RankBuilder(StreamInput in) throws IOException {
        // Functional Utility: Reconstructs a RankBuilder instance from a StreamInput.
        // This constructor is used during deserialization in a distributed environment.
        rankWindowSize = in.readVInt();
    }

    public final void writeTo(StreamOutput out) throws IOException {
        // Functional Utility: Writes the state of the RankBuilder to a StreamOutput.
        // This method is used for serializing the object for network transfer or persistence.
        out.writeVInt(rankWindowSize);
        doWriteTo(out);
    }

    protected abstract void doWriteTo(StreamOutput out) throws IOException;
    // Functional Utility: Abstract method to allow concrete RankBuilder implementations to write
    // their specific state to a StreamOutput. This ensures that custom fields are also serialized.

    @Override
    public final XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        // Functional Utility: Converts the RankBuilder instance into XContent (e.g., JSON).
        // This method is used for rendering the builder's configuration as part of a search request.
        builder.startObject();
        builder.startObject(getWriteableName());
        builder.field(RANK_WINDOW_SIZE_FIELD.getPreferredName(), rankWindowSize);
        doXContent(builder, params);
        builder.endObject();
        builder.endObject();
        return builder;
    }

    protected abstract void doXContent(XContentBuilder builder, Params params) throws IOException;
    // Functional Utility: Abstract method to allow concrete RankBuilder implementations to write
    // their specific state into an XContentBuilder. This ensures custom fields are included in the XContent representation.

    public int rankWindowSize() {
        // Functional Utility: Returns the configured rank window size for this RankBuilder.
        return rankWindowSize;
    }

    /**
     * Specify whether this rank builder is a compound builder or not. A compound builder is a rank builder that requires
     * two or more queries to be executed in order to generate the final result.
     */
    public abstract boolean isCompoundBuilder();
    // Functional Utility: Abstract method to indicate whether this RankBuilder implementation requires
    // multiple queries or ranking steps to produce a final result.

    /**
     * Generates an {@code Explanation} on how the final score for the provided {@code RankDoc} is computed for the given `RankBuilder`.
     * In addition to the base explanation to enrich, we also have access to the query names that were provided in the request,
     * so that we can have direct association with the user provided query.
     */
    public abstract Explanation explainHit(Explanation baseExplanation, RankDoc scoreDoc, List<String> queryNames);
    // Functional Utility: Abstract method to generate a detailed explanation of how the final score
    // for a specific document (RankDoc) was computed by this RankBuilder.

    /**
     * Generates a context used to execute required searches during the query phase on the shard.
     */
    public abstract QueryPhaseRankShardContext buildQueryPhaseShardContext(List<Query> queries, int from);
    // Functional Utility: Abstract method to create a context for executing ranking logic on a search shard
    // during the query phase. This context encapsulates the queries and state relevant for shard-level processing.

    /**
     * Generates a context used to be executed on the coordinating node, that would combine all individual shard results.
     */
    public abstract QueryPhaseRankCoordinatorContext buildQueryPhaseCoordinatorContext(int size, int from);
    // Functional Utility: Abstract method to create a context for coordinating and combining
    // ranking results from multiple shards on the coordinating node during the query phase.

    /**
     * Generates a context used to execute the rank feature phase on the shard. This is responsible for retrieving any needed
     * feature data, and passing them back to the coordinator through the appropriate {@link  RankShardResult}.
     */
    public abstract RankFeaturePhaseRankShardContext buildRankFeaturePhaseShardContext();
    // Functional Utility: Abstract method to create a context for executing a dedicated rank feature phase
    // on a search shard. This phase typically involves retrieving feature data for advanced ranking.

    /**
     * Generates a context used to perform global ranking during the RankFeature phase,
     * on the coordinator based on all the individual shard results. The output of this will be a `size` ranked list of ordered results,
     * which will then be passed to fetch phase.
     */
    public abstract RankFeaturePhaseRankCoordinatorContext buildRankFeaturePhaseCoordinatorContext(int size, int from, Client client);
    // Functional Utility: Abstract method to create a context for performing global ranking on the coordinating node
    // during the rank feature phase. This combines shard-level feature data to produce a globally ranked list.

    /**
     * Transforms the specific rank builder (as parsed through SearchSourceBuilder) to the corresponding retriever.
     * This is used to ensure smooth deprecation of `rank` and `sub_searches` and move towards the retriever framework
     */
    @UpdateForV10(owner = UpdateForV10.Owner.SEARCH_RELEVANCE) // remove for 10.0 once we remove support for the rank parameter in SearchAPI
    @Nullable
    public RetrieverBuilder toRetriever(SearchSourceBuilder searchSourceBuilder, Predicate<NodeFeature> clusterSupportsFeature) {
        // Functional Utility: Transforms this RankBuilder into a RetrieverBuilder.
        // This method facilitates the migration from the older `rank` API to the newer `retriever` framework.
        return null; // Block Logic: Default implementation returns null, indicating no direct retriever conversion for this builder.
    }

    @Override
    public final boolean equals(Object obj) {
        // Functional Utility: Compares this RankBuilder instance with another object for equality.
        // It checks if the objects are of the same class and have the same rank window size,
        // delegating to `doEquals` for subclass-specific comparisons.
        if (this == obj) {
            return true;
        }
        if (obj == null || getClass() != obj.getClass()) {
            return false;
        }
        RankBuilder other = (RankBuilder) obj;
        // Block Logic: Compare basic fields and then delegate to subclass for specific comparisons.
        return rankWindowSize == other.rankWindowSize && doEquals(other);
    }

    protected abstract boolean doEquals(RankBuilder other);
    // Functional Utility: Abstract method to allow concrete RankBuilder implementations to define
    // their specific equality comparison logic.

    @Override
    public final int hashCode() {
        // Functional Utility: Computes the hash code for this RankBuilder instance.
        // It combines the hash code of the class, rank window size, and subclass-specific hash code.
        return Objects.hash(getClass(), rankWindowSize, doHashCode());
    }

    protected abstract int doHashCode();
    // Functional Utility: Abstract method to allow concrete RankBuilder implementations to define
    // their specific hash code computation logic.

    @Override
    public String toString() {
        // Functional Utility: Returns a string representation of this RankBuilder instance.
        return Strings.toString(this, true, true);
    }
}
