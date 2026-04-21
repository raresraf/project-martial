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
 * @brief Transactional executor for Index Lifecycle Management (ILM) synchronous steps.
 * 
 * Functional Intent: Orchestrates a contiguous sequence of cluster-state mutating 
 * steps for a specific index. It executes all available synchronous transitions 
 * (Actions and Wait conditions) in a single atomic cluster state update, 
 * optimizing the lifecycle progression by minimizing expensive master node 
 * state publications. It also serves as a bridge between synchronous metadata 
 * changes and triggered asynchronous operations.
 * 
 * Domain: Production Systems, Finite State Machines (FSM), Cluster Coordination.
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
     * doExecute - Core execution engine for ILM step sequences.
     * 
     * Algorithm: Contiguous transition sweep.
     * Logic: 
     * 1. Validates that the index still exists and is at the expected step.
     * 2. While the current step is a synchronous mutation (Action) or a met condition (Wait):
     *    a. Executes the step logic against the current state.
     *    b. Updates the in-flight cluster state with the new lifecycle metadata.
     *    c. Checks for phase transitions to break the atomic update (safety boundary).
     *    d. Fetches the next step from the registry and repeats.
     * 3. Handles exceptions by diverting the lifecycle to an explicit 'ERROR' state.
     */
    @Override
    public ClusterState doExecute(final ClusterState currentState) throws IOException {
        Step currentStep = startStep;
        IndexMetadata indexMetadata = currentState.metadata().getProject().index(index);
        
        // Pre-condition: Target index must exist in the cluster metadata.
        if (indexMetadata == null) {
            logger.debug("lifecycle for index [{}] executed but index no longer exists", index.getName());
            return currentState;
        }

        // Logic: Ensures atomicity by verifying the index hasn't drifted to a different step 
        // while this task was queued in the master's update loop.
        Step registeredCurrentStep = IndexLifecycleRunner.getCurrentStep(policyStepsRegistry, policy, indexMetadata);
        if (currentStep.equals(registeredCurrentStep)) {
            ClusterState state = currentState;
            
            /**
             * Block Logic: Contiguous step execution loop.
             * Invariant: Transitions persist as long as the next step is synchronous and belongs to the same phase.
             */
            while (currentStep instanceof ClusterStateActionStep || currentStep instanceof ClusterStateWaitStep) {
                if (currentStep instanceof ClusterStateActionStep) {
                    // Logic: Synchronous state mutation (e.g. metadata update).
                    logger.trace(
                        "[{}] performing cluster state action ({}) [{}]",
                        index.getName(),
                        currentStep.getClass().getSimpleName(),
                        currentStep.getKey()
                    );
                    try {
                        ClusterStateActionStep actionStep = (ClusterStateActionStep) currentStep;
                        state = actionStep.performAction(index, state);
                        
                        // Synchronization: Captures spawned indices that need async task triggering post-publication.
                        Optional.ofNullable(actionStep.indexForAsyncInvocation())
                            .ifPresent(tuple -> indexToStepKeysForAsyncActions.put(tuple.v1(), tuple.v2()));
                    } catch (Exception exception) {
                        return moveToErrorStep(state, currentStep.getKey(), exception);
                    }
                    
                    nextStepKey = currentStep.getNextStepKey();
                    if (nextStepKey == null) {
                        return state;
                    } else {
                        // Logic: Commits the transition within the current batch.
                        logger.trace("[{}] moving cluster state to next step [{}]", index.getName(), nextStepKey);
                        state = ClusterState.builder(state)
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
                } else {
                    // Logic: Synchronous condition evaluation (e.g. checking if a shard is moved).
                    logger.trace(
                        "[{}] waiting for cluster state step condition ({}) [{}]",
                        index.getName(),
                        currentStep.getClass().getSimpleName(),
                        currentStep.getKey()
                    );
                    ClusterStateWaitStep.Result result;
                    try {
                        result = ((ClusterStateWaitStep) currentStep).isConditionMet(index, state);
                    } catch (Exception exception) {
                        return moveToErrorStep(state, currentStep.getKey(), exception);
                    }
                    
                    nextStepKey = currentStep.getNextStepKey();
                    if (result.complete()) {
                        // Logic: Condition met; advance the lifecycle to the next step.
                        if (nextStepKey == null) {
                            return state;
                        } else {
                            state = ClusterState.builder(state)
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
                    } else {
                        // Pre-condition: If wait condition fails, the atomic batch terminates immediately.
                        final ToXContentObject stepInfo = result.informationContext();
                        nextStepKey = null;
                        if (stepInfo == null) {
                            return state;
                        } else {
                            return IndexLifecycleTransition.addStepInfoToClusterState(index, state, stepInfo);
                        }
                    }
                }
                
                // Logic: Safety boundary; phase transitions must always result in a state publication 
                // to allow external controllers (like Allocation Deciders) to re-evaluate context.
                if (currentStep.getKey().phase().equals(currentStep.getNextStepKey().phase()) == false) {
                    return state;
                }
                currentStep = policyStepsRegistry.getStep(indexMetadata, currentStep.getNextStepKey());
            }
            return state;
        } else {
            // Guard: Rejects execution if mastership changed or the step was mutated concurrently.
            return currentState;
        }
    }

    /**
     * onClusterStateProcessed - Post-publication hook for async task triggering.
     * 
     * Logic: Invoked after the master has successfully distributed the new cluster 
     * state. It identifies if the current step is an 'AsyncAction' and initiates 
     * its out-of-band execution exactly once.
     */
    @Override
    public void onClusterStateProcessed(ClusterState newState) {
        final Metadata metadata = newState.metadata();
        final IndexMetadata indexMetadata = metadata.getProject().index(index);
        if (indexMetadata != null) {

            LifecycleExecutionState exState = indexMetadata.getLifecycleExecutionState();
            // Logic: Tracks success/failure for reporting and metrics.
            if (ErrorStep.NAME.equals(exState.step()) && this.failure != null) {
                lifecycleRunner.registerFailedOperation(indexMetadata, failure);
            } else {
                lifecycleRunner.registerSuccessfulOperation(indexMetadata);
            }

            if (nextStepKey != null && nextStepKey != TerminalPolicyStep.KEY) {
                // Logic: Triggers asynchronous logic iff the destination step requires it.
                lifecycleRunner.maybeRunAsyncAction(newState, indexMetadata, policy, nextStepKey);
            }
        }
        
        // Logic: Handles spawned indices (e.g. from a shrink/split operation).
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
     * moveToErrorStep - Diverts state to the ILM error handling terminal.
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
