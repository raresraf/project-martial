/**
 * @file LookupJoin.java
 * @brief Implements a specialized logical plan operator for "lookup joins" within Elasticsearch ESQL.
 * A `LookupJoin` is a form of LEFT (OUTER) JOIN where the right-hand side originates from a lookup index.
 * This class provides logic for constructing, transforming, and validating such join operations
 * within the ESQL query planning process, particularly considering remote execution and
 * compatibility with other logical plan operators like `Aggregate` and `Enrich`.
 */
/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.esql.plan.logical.join;

import org.elasticsearch.xpack.esql.capabilities.PostAnalysisVerificationAware;
import org.elasticsearch.xpack.esql.capabilities.TelemetryAware;
import org.elasticsearch.xpack.esql.common.Failures;
import org.elasticsearch.xpack.esql.core.expression.Attribute;
import org.elasticsearch.xpack.esql.core.tree.NodeInfo;
import org.elasticsearch.xpack.esql.core.tree.Source;
import org.elasticsearch.xpack.esql.plan.logical.Aggregate;
import org.elasticsearch.xpack.esql.plan.logical.Enrich;
import org.elasticsearch.xpack.esql.plan.logical.LogicalPlan;
import org.elasticsearch.xpack.esql.plan.logical.SurrogateLogicalPlan;
import org.elasticsearch.xpack.esql.plan.logical.UnaryPlan;
import org.elasticsearch.xpack.esql.plan.logical.join.JoinTypes.UsingJoinType;

import java.util.List;

import static java.util.Collections.emptyList;
import static org.elasticsearch.xpack.esql.common.Failure.fail;
import static org.elasticsearch.xpack.esql.plan.logical.join.JoinTypes.LEFT;

/**
 * @brief Represents a specialized logical plan operator for "lookup joins" within Elasticsearch ESQL.
 * This class extends `Join` and implements `SurrogateLogicalPlan`, `PostAnalysisVerificationAware`,
 * and `TelemetryAware` to provide custom behavior for LEFT (OUTER) JOINs where the right-hand side
 * is sourced from a lookup index (`index_mode = lookup`).
 */
public class LookupJoin extends Join implements SurrogateLogicalPlan, PostAnalysisVerificationAware, TelemetryAware {

    private boolean isRemote = false; // Flag indicating if the lookup join involves remote indices.

    /**
     * @brief Constructs a new `LookupJoin` with `UsingJoinType` and LEFT join type.
     * @param source The `Source` information for this plan node.
     * @param left The left-hand side `LogicalPlan` of the join.
     * @param right The right-hand side `LogicalPlan` (lookup index) of the join.
     * @param joinFields A `List` of `Attribute`s representing the fields used for joining.
     */
    public LookupJoin(Source source, LogicalPlan left, LogicalPlan right, List<Attribute> joinFields) {
        this(source, left, right, new UsingJoinType(LEFT, joinFields), emptyList(), emptyList(), emptyList());
    }

    /**
     * @brief Constructs a new `LookupJoin` with explicit `JoinType` and fields.
     * @param source The `Source` information for this plan node.
     * @param left The left-hand side `LogicalPlan` of the join.
     * @param right The right-hand side `LogicalPlan` (lookup index) of the join.
     * @param type The `JoinType` (e.g., LEFT, INNER) of the join.
     * @param joinFields A `List` of `Attribute`s representing the fields used for joining.
     * @param leftFields A `List` of `Attribute`s from the left side involved in the join.
     * @param rightFields A `List` of `Attribute`s from the right side involved in the join.
     */
    public LookupJoin(
        Source source,
        LogicalPlan left,
        LogicalPlan right,
        JoinType type,
        List<Attribute> joinFields,
        List<Attribute> leftFields,
        List<Attribute> rightFields
    ) {
        this(source, left, right, new JoinConfig(type, joinFields, leftFields, rightFields));
    }

    /**
     * @brief Constructs a new `LookupJoin` with a pre-configured `JoinConfig`.
     * @param source The `Source` information for this plan node.
     * @param left The left-hand side `LogicalPlan` of the join.
     * @param right The right-hand side `LogicalPlan` (lookup index) of the join.
     * @param joinConfig The `JoinConfig` encapsulating the join type and fields.
     */
    public LookupJoin(Source source, LogicalPlan left, LogicalPlan right, JoinConfig joinConfig) {
        super(source, left, right, joinConfig);
    }

    /**
     * @brief Transforms this `LookupJoin` into a regular `Join` with a `Projection` on top.
     * This method is part of the `SurrogateLogicalPlan` interface, providing a way
     * to translate the specialized `LookupJoin` into a more generic `Join` for
     * further planning stages, particularly concerning serialization and optimization.
     * @return A `LogicalPlan` representing the surrogate `Join` operation.
     */
    @Override
    public LogicalPlan surrogate() {
        // TODO: decide whether to introduce USING or just basic ON semantics - keep the ordering out for now
        return new Join(source(), left(), right(), config());
    }

    /**
     * @brief Replaces the child logical plans of this `LookupJoin` with new ones.
     * This method is typically used during plan optimization or transformation
     * to reconstruct the join with modified left and right children.
     * @param left The new left-hand side `LogicalPlan`.
     * @param right The new right-hand side `LogicalPlan`.
     * @return A new `LookupJoin` instance with the updated children.
     */
    @Override
    public Join replaceChildren(LogicalPlan left, LogicalPlan right) {
        return new LookupJoin(source(), left, right, config());
    }

    /**
     * @brief Provides node information for this `LookupJoin` instance.
     * This method is used by the tree traversal mechanism to recreate or inspect
     * the node's properties.
     * @return A `NodeInfo` object containing information necessary to reconstruct this `LookupJoin` node.
     */
    @Override
    protected NodeInfo<Join> info() {
        return NodeInfo.create(
            this,
            LookupJoin::new, // Functional Utility: Provides a constructor reference for recreation.
            left(),
            right(),
            config().type(),
            config().matchFields(),
            config().leftFields(),
            config().rightFields()
        );
    }

    /**
     * @brief Provides a label for telemetry reporting for this `LookupJoin` operation.
     * This label is used to identify the type of join in monitoring and analytics data.
     * @return A `String` representing the telemetry label for this join type.
     */
    @Override
    public String telemetryLabel() {
        return "LOOKUP JOIN";
    }

    /**
     * @brief Performs post-analysis verification checks specific to `LookupJoin`.
     * This method extends the base class's verification and additionally
     * checks for potential issues related to remote joins if the `isRemote` flag is set.
     * @param failures A `Failures` object to which any discovered issues are added.
     */
    @Override
    public void postAnalysisVerification(Failures failures) {
        super.postAnalysisVerification(failures);
        if (isRemote) {
            checkRemoteJoin(failures); // Functional Utility: Performs checks specific to remote lookup joins.
        }
    }

    /**
     * @brief Performs specific validation checks for remote lookup joins.
     * This method traverses the logical plan upwards to identify if a remote
     * lookup join is incorrectly preceded by an `Aggregate` operation or an
     * `Enrich` operation with a coordinator policy. Such sequences are invalid
     * and will result in adding failures to the provided `Failures` object.
     * @param failures A `Failures` object to which any discovered validation issues are added.
     */
    private void checkRemoteJoin(Failures failures) {
        boolean[] agg = { false }; // Flag to track if an Aggregate operation is found upstream.
        boolean[] enrichCoord = { false }; // Flag to track if an Enrich operation with coordinator mode is found upstream.

        // Block Logic: Traverses the logical plan tree upwards to identify problematic upstream operators.
        this.forEachUp(UnaryPlan.class, u -> {
            if (u instanceof Aggregate) {
                agg[0] = true; // Set flag if Aggregate is found.
            } else if (u instanceof Enrich enrich && enrich.mode() == Enrich.Mode.COORDINATOR) {
                enrichCoord[0] = true; // Set flag if Enrich with coordinator mode is found.
            }
            // Block Logic: If a remote Enrich operation is encountered, check for invalid upstream operators.
            if (u instanceof Enrich enrich && enrich.mode() == Enrich.Mode.REMOTE) {
                // Invariant: Remote lookup joins cannot be executed after STATS (Aggregate).
                if (agg[0]) {
                    failures.add(fail(enrich, "LOOKUP JOIN with remote indices can't be executed after STATS"));
                }
                // Invariant: Remote lookup joins cannot be executed after ENRICH with coordinator policy.
                if (enrichCoord[0]) {
                    failures.add(fail(enrich, "LOOKUP JOIN with remote indices can't be executed after ENRICH with coordinator policy"));
                }
            }
        });
    }

    /**
     * @brief Checks if this `LookupJoin` is configured to operate on remote indices.
     * @return `true` if the join involves remote indices, `false` otherwise.
     */
    public boolean isRemote() {
        return isRemote;
    }

    /**
     * @brief Sets the remote flag for this `LookupJoin` operation.
     * This indicates whether the join should operate on remote indices.
     * @param remote `true` to configure the join for remote indices, `false` otherwise.
     * @return This `LookupJoin` instance, allowing for method chaining.
     */
    public LookupJoin setRemote(boolean remote) {
        isRemote = remote;
        return this;
    }
}
