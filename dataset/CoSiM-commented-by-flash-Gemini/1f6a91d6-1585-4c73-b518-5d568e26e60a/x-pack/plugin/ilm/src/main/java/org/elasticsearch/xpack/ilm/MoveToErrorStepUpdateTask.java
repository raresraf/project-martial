/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.elasticsearch.xpack.ilm;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.ExceptionsHelper;
import org.elasticsearch.cluster.ClusterState;
import org.elasticsearch.cluster.NotMasterException;
import org.elasticsearch.cluster.coordination.FailedToCommitClusterStateException;
import org.elasticsearch.cluster.metadata.IndexMetadata;
import org.elasticsearch.cluster.metadata.LifecycleExecutionState;
import org.elasticsearch.common.Strings;
import org.elasticsearch.index.Index;
import org.elasticsearch.xpack.core.ilm.Step;

import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.LongSupplier;

 /**
  * @brief Functional description of the MoveToErrorStepUpdateTask class.
  *        This is a placeholder for detailed semantic documentation.
  *        Further analysis will elaborate on its algorithm, complexity, and invariants.
  */
public class MoveToErrorStepUpdateTask extends IndexLifecycleClusterStateUpdateTask {

    private static final Logger logger = LogManager.getLogger(MoveToErrorStepUpdateTask.class);

     /**
      * @brief [Functional description for field index]: Describe purpose here.
      */
    private final Index index;
     /**
      * @brief [Functional description for field policy]: Describe purpose here.
      */
    private final String policy;
     /**
      * @brief [Functional description for field currentStepKey]: Describe purpose here.
      */
    private final Step.StepKey currentStepKey;
    private final BiFunction<IndexMetadata, Step.StepKey, Step> stepLookupFunction;
     /**
      * @brief [Functional description for field stateChangeConsumer]: Describe purpose here.
      */
    private final Consumer<ClusterState> stateChangeConsumer;
     /**
      * @brief [Functional description for field nowSupplier]: Describe purpose here.
      */
    private final LongSupplier nowSupplier;
     /**
      * @brief [Functional description for field cause]: Describe purpose here.
      */
    private final Exception cause;

    /**
     * @brief [Functional Utility for MoveToErrorStepUpdateTask]: Describe purpose here.
     */
    public MoveToErrorStepUpdateTask(
        Index index,
        String policy,
        Step.StepKey currentStepKey,
        Exception cause,
        LongSupplier nowSupplier,
        BiFunction<IndexMetadata, Step.StepKey, Step> stepLookupFunction,
        Consumer<ClusterState> stateChangeConsumer
    ) {
        super(index, currentStepKey);
        this.index = index;
        this.policy = policy;
        this.currentStepKey = currentStepKey;
        this.cause = cause;
        this.nowSupplier = nowSupplier;
        this.stepLookupFunction = stepLookupFunction;
        this.stateChangeConsumer = stateChangeConsumer;
    }

    @Override
    protected ClusterState doExecute(ClusterState currentState) throws Exception {
         /**
          * @brief [Functional description for field project]: Describe purpose here.
          */
        final var project = currentState.getMetadata().getProject();
         /**
          * @brief [Functional description for field idxMeta]: Describe purpose here.
          */
        IndexMetadata idxMeta = project.index(index);
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (idxMeta == null) {
            // Index must have been since deleted, ignore it
             /**
              * @brief [Functional description for field currentState]: Describe purpose here.
              */
            return currentState;
        }
         /**
          * @brief [Functional description for field lifecycleState]: Describe purpose here.
          */
        LifecycleExecutionState lifecycleState = idxMeta.getLifecycleExecutionState();
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (policy.equals(idxMeta.getLifecyclePolicyName()) && currentStepKey.equals(Step.getCurrentStepKey(lifecycleState))) {
            return ClusterState.builder(currentState)
                .putProjectMetadata(IndexLifecycleTransition.moveIndexToErrorStep(index, project, cause, nowSupplier, stepLookupFunction))
                .build();
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        } else {
            // either the policy has changed or the step is now
            // not the same as when we submitted the update task. In
            // either case we don't want to do anything now
             /**
              * @brief [Functional description for field currentState]: Describe purpose here.
              */
            return currentState;
        }
    }

    @Override
    public void onClusterStateProcessed(ClusterState newState) {
        stateChangeConsumer.accept(newState);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MoveToErrorStepUpdateTask that = (MoveToErrorStepUpdateTask) o;
        // We don't have a stable equals on the cause and shouldn't have simultaneous moves to error step to begin with when deduplicating
        // tasks so we only compare the current state here and in the hashcode.
        return index.equals(that.index) && policy.equals(that.policy) && currentStepKey.equals(that.currentStepKey);
    }

    @Override
    public int hashCode() {
        return Objects.hash(index, policy, currentStepKey);
    }

    @Override
    protected void handleFailure(Exception e) {
        Level level;
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (ExceptionsHelper.unwrap(e, NotMasterException.class, FailedToCommitClusterStateException.class) != null) {
            level = Level.DEBUG;
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        } else {
            level = Level.ERROR;
            assert false : new AssertionError("unexpected exception", e);
        }
        logger.log(
            level,
            () -> Strings.format(
                "policy [%s] for index [%s] failed trying to move from step [%s] to the ERROR step.",
                policy,
                index.getName(),
                currentStepKey
            )
        );
    }
}
