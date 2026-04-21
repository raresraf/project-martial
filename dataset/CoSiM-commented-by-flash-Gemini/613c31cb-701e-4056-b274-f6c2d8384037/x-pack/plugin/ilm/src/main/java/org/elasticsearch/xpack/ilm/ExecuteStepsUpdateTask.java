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
 * @brief Orchestrator for sequential execution of Index Lifecycle Management (ILM) steps.
 * 
 * Functional Intent: Manages the atomic transition of an index through various 
 * lifecycle stages (Phases, Actions, Steps). It executes a chain of metadata-only 
 * operations (ClusterStateAction/Wait) in a single update cycle and coordinates 
 * exactly-once execution of asynchronous background tasks (like shard relocation).
 * 
 * Domain: Production Systems, Distributed Databases, State Management.
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
     * doExecute - Executes a sequence of ILM steps that modify the cluster state.
     * 
     * Algorithm: Finite State Machine (FSM) sweep.
     * Logic: 
     * 1. Validates index residency and step alignment (consensus check).
     * 2. Iteratively processes ClusterStateActionStep and ClusterStateWaitStep.
     * 3. Terminates if a non-cluster-state step is encountered or a phase transition occurs.
     * 
     * Invariant: All steps within a single loop must belong to the same lifecycle phase 
     * to maintain audit trail consistency.
     */
    @Override
    public ClusterState doExecute(final ClusterState currentState) throws IOException {
        Step currentStep = startStep;
        IndexMetadata indexMetadata = currentState.metadata().index(index);
        
        // Pre-condition: Index must exist in the current metadata version.
        if (indexMetadata == null) {
            logger.debug("lifecycle for index [{}] executed but index no longer exists", index.getName());
            return currentState;
        }
        
        // Synchronization: Ensures the index is still at the starting step (prevents race with other master updates).
        Step registeredCurrentStep = IndexLifecycleRunner.getCurrentStep(policyStepsRegistry, policy, indexMetadata);
        if (currentStep.equals(registeredCurrentStep) == false) {
            return currentState;
        }
        
        ClusterState state = currentState;
        
        /**
         * Block Logic: Batch step execution loop.
         * Invariant: Successfully executes a chain of immediate actions as long as conditions are met.
         */
        while (currentStep instanceof ClusterStateActionStep || currentStep instanceof ClusterStateWaitStep) {
            try {
                if (currentStep instanceof ClusterStateActionStep) {
                    state = executeActionStep(state, currentStep);
                } else {
                    state = executeWaitStep(state, currentStep);
                }
            } catch (Exception exception) {
                // Error Handling: Forced transition to terminal ERROR state.
                return moveToErrorStep(state, currentStep.getKey(), exception);
            }
            
            if (nextStepKey == null) {
                return state;
            } else {
                state = moveToNextStep(state);
            }
            
            // Logic: Halt sequence at phase boundaries to allow specialized phase-entry logic.
            if (currentStep.getKey().phase().equals(currentStep.getNextStepKey().phase()) == false) {
                return state;
            }
            currentStep = policyStepsRegistry.getStep(indexMetadata, currentStep.getNextStepKey());
        }
        return state;
    }

    /**
     * @brief Performs a metadata-only action step (e.g. updating index settings).
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
        
        // Logic: Tracking child indices that may have been spawned (e.g., from shrinking).
        Optional.ofNullable(actionStep.indexForAsyncInvocation())
            .ifPresent(tuple -> indexToStepKeysForAsyncActions.put(tuple.v1(), tuple.v2()));
            
        nextStepKey = currentStep.getNextStepKey();
        return state;
    }

    /**
     * @brief Evaluates a wait condition against the cluster state.
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
            // Logic: Condition not met, attach diagnostic info for observability.
            final ToXContentObject stepInfo = result.informationContext();
            nextStepKey = null;
            if (stepInfo == null) {
                return state;
            }
            return IndexLifecycleTransition.addStepInfoToClusterState(index, state, stepInfo);
        }
    }

    /**
     * @brief Transitions the index to the next planned lifecycle step.
     */
    private ClusterState moveToNextStep(ClusterState state) {
        if (nextStepKey == null) {
            return state;
        }
        logger.trace("[{}] moving cluster state to next step [{}]", index.getName(), nextStepKey);
        return IndexLifecycleTransition.moveClusterStateToStep(
            index,
            state,
            nextStepKey,
            nowSupplier,
            policyStepsRegistry,
            false
        );
    }

    /**
     * onClusterStateProcessed - Post-publication hook for exactly-once async dispatch.
     * 
     * Logic: Triggers background actions (like shard allocation) after the metadata 
     * change has been successfully replicated across the cluster.
     */
    @Override
    public void onClusterStateProcessed(ClusterState newState) {
        final Metadata metadata = newState.metadata();
        final IndexMetadata indexMetadata = metadata.index(index);
        if (indexMetadata != null) {

            LifecycleExecutionState exState = indexMetadata.getLifecycleExecutionState();
            // Invariant: Reports the result of the metadata update to the runner.
            if (ErrorStep.NAME.equals(exState.step()) && this.failure != null) {
                lifecycleRunner.registerFailedOperation(indexMetadata, failure);
            } else {
                lifecycleRunner.registerSuccessfulOperation(indexMetadata);
            }

            if (nextStepKey != null && nextStepKey != TerminalPolicyStep.KEY) {
                // Functional Intent: Initiate heavy background operations after state sync.
                lifecycleRunner.maybeRunAsyncAction(newState, indexMetadata, policy, nextStepKey);
            }
        }
        
        // Block Logic: Handling spawned indices from previous action steps.
        assert indexToStepKeysForAsyncActions.size() <= 1 : "we expect a maximum of one single spawned index currently";
        for (Map.Entry<String, Step.StepKey> indexAndStepKey : indexToStepKeysForAsyncActions.entrySet()) {
            final String indexName = indexAndStepKey.getKey();
            final Step.StepKey nextStep = indexAndStepKey.getValue();
            final IndexMetadata indexMeta = metadata.index(indexName);
            if (indexMeta != null) {
                if (newState.metadata().isIndexManagedByILM(indexMeta)) {
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
     * @brief Transitions the state machine into an error state due to execution failure.
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
