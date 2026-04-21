/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ilm;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.cluster.ClusterState;
import org.elasticsearch.cluster.ClusterStateUpdateTask;
import org.elasticsearch.cluster.metadata.IndexMetadata;
import org.elasticsearch.cluster.metadata.LifecycleExecutionState;
import org.elasticsearch.cluster.metadata.Metadata;
import org.elasticsearch.common.Strings;
import org.elasticsearch.index.Index;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xpack.core.ilm.ClusterStateActionStep;
import org.elasticsearch.xpack.core.ilm.ClusterStateWaitStep;
import org.elasticsearch.xpack.core.ilm.ErrorStep;
import org.elasticsearch.xpack.core.ilm.LifecycleSettings;
import org.elasticsearch.xpack.core.ilm.Step;
import org.elasticsearch.xpack.core.ilm.TerminalPolicyStep;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.LongSupplier;

import static org.elasticsearch.core.Strings.format;

/**
 * @file ExecuteStepsUpdateTask.java
 * @brief Cluster state update task for executing a sequence of Index Lifecycle Management (ILM) steps.
 * 
 * Functional Intent: Orchestrates the sequential execution of ILM steps for a 
 * specific index. It handles both immediate actions (mutating cluster metadata) 
 * and wait conditions (polling for external state changes). Manages state transitions, 
 * error recovery, and async task triggering after state publication.
 * 
 * Domain: Production Systems, Distributed Databases, Life-cycle Orchestration.
 */
public class ExecuteStepsUpdateTask extends IndexLifecycleClusterStateUpdateTask {
    private static final Logger logger = LogManager.getLogger(ExecuteStepsUpdateTask.class);
    private final String policy;
    private final Step startStep;
    private final PolicyStepsRegistry policyStepsRegistry;
    private final IndexLifecycleRunner lifecycleRunner;
    private final LongSupplier nowSupplier;
    private final Map<String, Step.StepKey> indexToStepKeysForAsyncActions;
    private Step.StepKey nextStepKey = null;
    private Exception failure = null;

    /**
     * @brief Initializes the task with the target policy and starting execution point.
     */
    public ExecuteStepsUpdateTask(
        String policy,
        Index index,
        Step startStep,
        PolicyStepsRegistry policyStepsRegistry,
        IndexLifecycleRunner lifecycleRunner,
        LongSupplier nowSupplier
    ) {
        super(index, startStep.getKey());
        this.policy = policy;
        this.startStep = startStep;
        this.policyStepsRegistry = policyStepsRegistry;
        this.nowSupplier = nowSupplier;
        this.lifecycleRunner = lifecycleRunner;
        this.indexToStepKeysForAsyncActions = new HashMap<>();
    }

    String getPolicy() {
        return policy;
    }

    Step getStartStep() {
        return startStep;
    }

    Step.StepKey getNextStepKey() {
        return nextStepKey;
    }

    /**
     * doExecute - Core state transition logic.
     * 
     * Algorithm: Finite State Machine traversal.
     * Logic: 
     * 1. Validates that the index still exists and is at the expected step.
     * 2. Iteratively executes cluster-state steps (Action/Wait) within the same phase.
     * 3. Terminates if a step is not a cluster-state step, a wait condition is not met, 
     *    or a phase transition occurs.
     * 
     * Invariant: Each loop iteration represents an atomic step execution that potentially 
     * mutates the cluster state.
     */
    @Override
    public ClusterState doExecute(final ClusterState currentState) throws IOException {
        Step currentStep = startStep;
        IndexMetadata indexMetadata = currentState.metadata().getProject().index(index);
        
        // Pre-condition: Index existence check.
        if (indexMetadata == null) {
            logger.debug("lifecycle for index [{}] executed but index no longer exists", index.getName());
            return currentState;
        }
        
        // Synchronization: Verification that the index hasn't transitioned to a different step concurrently.
        Step registeredCurrentStep = IndexLifecycleRunner.getCurrentStep(policyStepsRegistry, policy, indexMetadata);
        if (currentStep.equals(registeredCurrentStep) == false) {
            return currentState;
        }
        
        ClusterState state = currentState;
        
        /**
         * Block Logic: Sequential step processing loop.
         * Invariant: Continues as long as steps can be executed within a single cluster state update.
         */
        while (currentStep instanceof ClusterStateActionStep || currentStep instanceof ClusterStateWaitStep) {
            try {
                if (currentStep instanceof ClusterStateActionStep) {
                    state = executeActionStep(state, currentStep);
                } else {
                    state = executeWaitStep(state, currentStep);
                }
            } catch (Exception exception) {
                // Error Handling: Automatic transition to a terminal ERROR state upon failure.
                return moveToErrorStep(state, currentStep.getKey(), exception);
            }
            
            if (nextStepKey == null) {
                return state;
            } else {
                state = moveToNextStep(state);
            }
            
            // Logic: Prevents cross-phase execution in a single update task to allow phase-transition bookkeeping.
            if (currentStep.getKey().phase().equals(currentStep.getNextStepKey().phase()) == false) {
                return state;
            }
            currentStep = policyStepsRegistry.getStep(indexMetadata, currentStep.getNextStepKey());
        }
        return state;
    }

    /**
     * @brief Executes a metadata-mutating action.
     */
    private ClusterState executeActionStep(ClusterState state, Step currentStep) {
        logger.trace(
            "[{}] performing cluster state action ({}) [{}]",
            index.getName(),
            currentStep.getClass().getSimpleName(),
            currentStep.getKey()
        );
        ClusterStateActionStep actionStep = (ClusterStateActionStep) currentStep;
        state = actionStep.performAction(index, state);
        
        // Logic: Tracking child indices that may require async follow-up (e.g., after a shrink).
        Optional.ofNullable(actionStep.indexForAsyncInvocation())
            .ifPresent(tuple -> indexToStepKeysForAsyncActions.put(tuple.v1(), tuple.v2()));
            
        nextStepKey = currentStep.getNextStepKey();
        return state;
    }

    /**
     * @brief Evaluates a wait condition for a lifecycle step.
     */
    private ClusterState executeWaitStep(ClusterState state, Step currentStep) {
        logger.trace(
            "[{}] waiting for cluster state step condition ({}) [{}]",
            index.getName(),
            currentStep.getClass().getSimpleName(),
            currentStep.getKey()
        );
        ClusterStateWaitStep.Result result = ((ClusterStateWaitStep) currentStep).isConditionMet(index, state);
        
        nextStepKey = currentStep.getNextStepKey();
        
        if (result.complete()) {
            logger.trace(
                "[{}] cluster state step condition met successfully ({}) [{}]",
                index.getName(),
                currentStep.getClass().getSimpleName(),
                currentStep.getKey()
            );
            return state;
        } else {
            // Logic: Condition not met, attach diagnostic info to the state and halt sequence.
            final ToXContentObject stepInfo = result.informationContext();
            nextStepKey = null;
            if (stepInfo == null) {
                return state;
            }
            return IndexLifecycleTransition.addStepInfoToClusterState(index, state, stepInfo);
        }
    }

    /**
     * @brief Transitions the index to the next planned step in the metadata.
     */
    private ClusterState moveToNextStep(ClusterState state) {
        if (nextStepKey == null) {
            return state;
        }
        logger.trace("[{}] moving cluster state to next step [{}]", index.getName(), nextStepKey);
        return ClusterState.builder(state)
            .putProjectMetadata(
                IndexLifecycleTransition.moveIndexToStep(
                    index,
                    state.metadata().getProject(),
                    nextStepKey,
                    nowSupplier,
                    policyStepsRegistry,
                    false
                )
            )
            .build();
    }

    /**
     * onClusterStateProcessed - Post-publication hook.
     * 
     * Functional Intent: Triggers exactly-once async actions after the cluster state 
     * update has been successfully applied to all nodes.
     */
    @Override
    public void onClusterStateProcessed(ClusterState newState) {
        final Metadata metadata = newState.metadata();
        final IndexMetadata indexMetadata = metadata.getProject().index(index);
        
        if (indexMetadata != null) {
            LifecycleExecutionState exState = indexMetadata.getLifecycleExecutionState();
            
            // Logic: Reporting successful or failed execution to the runner.
            if (ErrorStep.NAME.equals(exState.step()) && this.failure != null) {
                lifecycleRunner.registerFailedOperation(indexMetadata, failure);
            } else {
                lifecycleRunner.registerSuccessfulOperation(indexMetadata);
            }

            if (nextStepKey != null && nextStepKey != TerminalPolicyStep.KEY) {
                // Logic: Dispatch async operations (e.g., allocating shards, calling external APIs).
                lifecycleRunner.maybeRunAsyncAction(newState, indexMetadata, policy, nextStepKey);
            }
        }
        
        // Block Logic: Handling spawned indices (e.g. from Follower/Shrink actions).
        assert indexToStepKeysForAsyncActions.size() <= 1 : "we expect a maximum of one single spawned index currently";
        for (Map.Entry<String, Step.StepKey> indexAndStepKey : indexToStepKeysForAsyncActions.entrySet()) {
            final String indexName = indexAndStepKey.getKey();
            final Step.StepKey nextStep = indexAndStepKey.getValue();
            final IndexMetadata indexMeta = metadata.getProject().index(indexName);
            if (indexMeta != null) {
                if (newState.metadata().getProject().isIndexManagedByILM(indexMeta)) {
                    if (nextStep != null && nextStep != TerminalPolicyStep.KEY) {
                        final String policyName = LifecycleSettings.LIFECYCLE_NAME_SETTING.get(indexMeta.getSettings());
                        lifecycleRunner.maybeRunAsyncAction(newState, indexMeta, policyName, nextStep);
                    }
                }
            }
        }
    }

    @Override
    public void handleFailure(Exception e) {
        logger.warn(() -> format("policy [%s] for index [%s] failed on step [%s].", policy, index, startStep.getKey()), e);
    }

    /**
     * @brief Forcibly transitions the FSM to an error state due to unhandled exceptions.
     */
    private ClusterState moveToErrorStep(final ClusterState state, Step.StepKey currentStepKey, Exception cause) {
        this.failure = cause;
        logger.warn(
            () -> format(
                "policy [%s] for index [%s] failed on cluster state step [%s]. Moving to ERROR step",
                policy,
                index.getName(),
                currentStepKey
            ),
            cause
        );
        return IndexLifecycleTransition.moveClusterStateToErrorStep(index, state, cause, nowSupplier, policyStepsRegistry::getStep);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ExecuteStepsUpdateTask that = (ExecuteStepsUpdateTask) o;
        return policy.equals(that.policy) && index.equals(that.index) && Objects.equals(startStep, that.startStep);
    }

    @Override
    public int hashCode() {
        return Objects.hash(policy, index, startStep);
    }
}
