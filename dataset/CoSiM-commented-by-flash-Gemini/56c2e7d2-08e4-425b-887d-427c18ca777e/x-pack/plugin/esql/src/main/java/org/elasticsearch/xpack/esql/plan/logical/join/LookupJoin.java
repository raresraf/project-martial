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
 * @file LookupJoin.java
 * @brief Represents a logical plan for a specialized LOOKUP JOIN operation in ESQL.
 *
 * This class extends the {@link Join} logical plan to specifically handle LOOKUP JOINs,
 * which are a form of LEFT (OUTER) JOIN. It is designed for scenarios where the left side
 * of the join is the main data source and the right side is a lookup index (index_mode = lookup).
 *
 * It implements {@link SurrogateLogicalPlan} to allow translation into a regular join for
 * serialization and processing, and includes post-analysis verification logic for remote joins.
 * It also provides telemetry information as indicated by {@link TelemetryAware}.
 *
 * Algorithm: Logical plan representation for left outer join with specific optimizations/constraints
 * for lookup indices. Includes verification rules for remote execution context.
 * Time Complexity: Operations on the logical plan itself (e.g., replacement, information retrieval)
 * are typically O(1) or O(number of attributes/children). The actual join execution complexity
 * is handled by downstream physical planning and execution engines.
 * Space Complexity: O(number of attributes/children) for storing plan configuration and fields.
 */
public class LookupJoin extends Join implements SurrogateLogicalPlan, PostAnalysisVerificationAware, TelemetryAware {

    /**
     * @brief Flag indicating whether this lookup join involves a remote data source.
     * This flag is used in post-analysis verification to apply specific rules for remote joins.
     */
    private boolean isRemote = false;

    /**
     * @brief Constructor for a LookupJoin with default LEFT join type and specified join fields.
     * @param source The source of the logical plan node.
     * @param left The left child logical plan.
     * @param right The right child logical plan (expected to be a lookup index).
     * @param joinFields A list of attributes used for the join condition.
     */
    public LookupJoin(Source source, LogicalPlan left, LogicalPlan right, List<Attribute> joinFields) {
        this(source, left, right, new UsingJoinType(LEFT, joinFields), emptyList(), emptyList(), emptyList());
    }

    /**
     * @brief Constructor for a LookupJoin with detailed join configuration.
     * @param source The source of the logical plan node.
     * @param left The left child logical plan.
     * @param right The right child logical plan.
     * @param type The type of join (e.g., LEFT).
     * @param joinFields A list of attributes used for the join condition.
     * @param leftFields A list of fields from the left side.
     * @param rightFields A list of fields from the right side.
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
     * @brief Constructor for a LookupJoin using a pre-configured {@link JoinConfig}.
     * @param source The source of the logical plan node.
     * @param left The left child logical plan.
     * @param right The right child logical plan.
     * @param joinConfig The configuration object for the join operation.
     */
    public LookupJoin(Source source, LogicalPlan left, LogicalPlan right, JoinConfig joinConfig) {
        super(source, left, right, joinConfig);
    }

    /**
     * @brief Provides a surrogate logical plan representation for this LookupJoin.
     *
     * This method translates the specialized LookupJoin into a more generic {@link Join}
     * logical plan, typically for purposes like serialization or when the specific
     * lookup semantics are no longer required at a later stage of planning.
     *
     * @return A new {@link Join} instance representing the surrogate plan.
     */
    @Override
    public LogicalPlan surrogate() {
        // TODO: decide whether to introduce USING or just basic ON semantics - keep the ordering out for now
        // Block Logic: Creates a new generic Join instance using the configuration from this LookupJoin.
        // This effectively "despecializes" the lookup join for further processing.
        return new Join(source(), left(), right(), config());
    }

    /**
     * @brief Creates a new LookupJoin instance with updated child logical plans.
     * This is part of the tree transformation process in logical plan optimization.
     * @param left The new left child logical plan.
     * @param right The new right child logical plan.
     * @return A new {@link LookupJoin} instance with the specified children and existing configuration.
     */
    @Override
    public Join replaceChildren(LogicalPlan left, LogicalPlan right) {
        return new LookupJoin(source(), left, right, config());
    }

    /**
     * @brief Provides node information for this LookupJoin, used for reconstruction and debugging.
     * @return A {@link NodeInfo} object containing constructor references and current state.
     */
    @Override
    protected NodeInfo<Join> info() {
        return NodeInfo.create(
            this,
            LookupJoin::new,
            left(),
            right(),
            config().type(),
            config().matchFields(),
            config().leftFields(),
            config().rightFields()
        );
    }

    /**
     * @brief Returns a label for telemetry purposes, identifying this operation.
     * @return A string label "LOOKUP JOIN".
     */
    @Override
    public String telemetryLabel() {
        return "LOOKUP JOIN";
    }

    /**
     * @brief Performs post-analysis verification specific to this LookupJoin.
     *
     * This method is invoked after the initial analysis phase to check for semantic
     * or logical inconsistencies, especially concerning remote joins.
     * @param failures A {@link Failures} object to accumulate any verification errors.
     */
    @Override
    public void postAnalysisVerification(Failures failures) {
        super.postAnalysisVerification(failures);
        // Block Logic: If the join is marked as remote, perform additional checks.
        // Pre-condition: The 'isRemote' flag has been set to true for this LookupJoin.
        if (isRemote) {
            checkRemoteJoin(failures);
        }
    }

    /**
     * @brief Performs specific validation checks for remote lookup joins.
     *
     * This method checks for invalid combinations of operations when a LOOKUP JOIN
     * involves remote indices, specifically disallowing certain operations like
     * AGGREGATE (STATS) or ENRICH with a COORDINATOR policy to precede the remote enrich operation.
     *
     * @param failures A {@link Failures} object to which validation errors are added.
     */
    private void checkRemoteJoin(Failures failures) {
        // agg: Flag to track if an Aggregate (STATS) operation is encountered in the plan.
        boolean[] agg = { false };
        // enrichCoord: Flag to track if an Enrich operation with COORDINATOR mode is encountered.
        boolean[] enrichCoord = { false };

        // Block Logic: Traverses the logical plan tree upwards, looking for specific unary plan nodes.
        // It checks if an Enrich REMOTe operation is preceded by an Aggregate or Enrich COORDINATOR.
        this.forEachUp(UnaryPlan.class, u -> {
            // Inline: Checks if the current UnaryPlan 'u' is an instance of Aggregate.
            if (u instanceof Aggregate) {
                agg[0] = true;
            // Inline: Checks if the current UnaryPlan 'u' is an Enrich operation with COORDINATOR mode.
            } else if (u instanceof Enrich enrich && enrich.mode() == Enrich.Mode.COORDINATOR) {
                enrichCoord[0] = true;
            }
            // Block Logic: If the current UnaryPlan 'u' is an Enrich operation with REMOTE mode,
            // then check for invalid preceding operations.
            if (u instanceof Enrich enrich && enrich.mode() == Enrich.Mode.REMOTE) {
                // If an Aggregate was found earlier, fail the verification.
                // Invariant: Aggregate operations cannot precede a remote lookup join that is enriched remotely.
                if (agg[0]) {
                    failures.add(fail(enrich, "LOOKUP JOIN with remote indices can't be executed after STATS"));
                }
                // If an Enrich with COORDINATOR mode was found earlier, fail the verification.
                // Invariant: Enrich operations with COORDINATOR policy cannot precede a remote lookup join that is enriched remotely.
                if (enrichCoord[0]) {
                    failures.add(fail(enrich, "LOOKUP JOIN with remote indices can't be executed after ENRICH with coordinator policy"));
                }
            }
        });
    }

    /**
     * @brief Checks if this lookup join involves remote indices.
     * @return true if the join is remote, false otherwise.
     */
    public boolean isRemote() {
        return isRemote;
    }

    /**
     * @brief Sets the remote status of this lookup join.
     * @param remote true to mark the join as remote, false otherwise.
     * @return This {@link LookupJoin} instance for method chaining.
     */
    public LookupJoin setRemote(boolean remote) {
        isRemote = remote;
        return this;
    }
}
