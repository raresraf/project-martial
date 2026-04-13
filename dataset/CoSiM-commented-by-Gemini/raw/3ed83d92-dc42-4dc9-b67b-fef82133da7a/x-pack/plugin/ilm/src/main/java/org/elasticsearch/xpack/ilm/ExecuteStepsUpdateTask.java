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
 * A {@link ClusterStateUpdateTask} that executes a specific sequence of ILM
 * {@link Step}s for a given index.
 *
 * This task is the primary engine for advancing an index through its lifecycle policy.
 * It is designed to execute synchronous steps (those that modify the cluster state directly
 * or wait for a cluster state condition) in a batch. When it encounters a step
 * that requires a long-running asynchronous action (like shrink or force-merge),
 * it transitions the state and then triggers the {@link IndexLifecycleRunner} to
 * perform the action outside of the cluster state update thread.
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
     * Executes one or more ILM steps for an index.
     *
     * The method iterates through consecutive steps as long as they are synchronous
     * ({@link ClusterStateActionStep} or {@link ClusterStateWaitStep}) and within the same phase.
     * It stops when it encounters an asynchronous step, a wait step whose condition is not met,
     * or a phase transition, returning the modified {@link ClusterState}.
     *
     * @param currentState The current state to execute the <code>startStep</code> with.
     * @return The new cluster state after applying the step's actions.
     * @throws IOException if any exceptions occur during step execution.
     */
    @Override
    public ClusterState doExecute(final ClusterState currentState) throws IOException {
        Step currentStep = startStep;
        IndexMetadata indexMetadata = currentState.metadata().getProject().index(index);
        if (indexMetadata == null) {
            // Index has been deleted, nothing to do.
            logger.debug("lifecycle for index [{}] executed but index no longer exists", index.getName());
            return currentState;
        }

        // Pre-condition: Verify that the step we intend to execute is still the current step in the cluster state.
        // This is a safeguard against race conditions where the state might have changed since the task was submitted.
        Step registeredCurrentStep = IndexLifecycleRunner.getCurrentStep(policyStepsRegistry, policy, indexMetadata);
        if (currentStep.equals(registeredCurrentStep) == false) {
            logger.debug(
                "index [{}] has changed step from [{}] to [{}], skipping execution",
                index.getName(),
                currentStep.getKey(),
                registeredCurrentStep == null ? "null" : registeredCurrentStep.getKey()
            );
            return currentState;
        }

        ClusterState state = currentState;
        // Block Logic: This loop executes multiple synchronous steps in a single cluster state update.
        // It continues as long as the steps are safe to run sequentially on the master node.
        while (currentStep instanceof ClusterStateActionStep || currentStep instanceof ClusterStateWaitStep) {
            try {
                if (currentStep instanceof ClusterStateActionStep) {
                    state = executeActionStep(state, currentStep);
                } else {
                    state = executeWaitStep(state, currentStep);
                }
            } catch (Exception exception) {
                // If any step fails, move the index to the ERROR step and halt execution.
                return moveToErrorStep(state, currentStep.getKey(), exception);
            }

            // If a wait-step condition was not met, nextStepKey will be null, and we exit.
            if (nextStepKey == null) {
                return state;
            } else {
                state = moveToNextStep(state);
            }

            // If the next step is in a new phase, exit the loop to allow for phase-transition logic.
            if (currentStep.getKey().phase().equals(currentStep.getNextStepKey().phase()) == false) {
                return state;
            }
            currentStep = policyStepsRegistry.getStep(indexMetadata, currentStep.getNextStepKey());
        }
        return state;
    }

    private ClusterState executeActionStep(ClusterState state, Step currentStep) {
        logger.trace(
            "[{}] performing cluster state action ({}) [{}]",
            index.getName(),
            currentStep.getClass().getSimpleName(),
            currentStep.getKey()
        );
        ClusterStateActionStep actionStep = (ClusterStateActionStep) currentStep;
        state = actionStep.performAction(index, state);

        // This allows a step to signal that a newly created index (e.g. from a shrink action)
        // should also have its ILM process initiated.
        Optional.ofNullable(actionStep.indexForAsyncInvocation())
            .ifPresent(tuple -> indexToStepKeysForAsyncActions.put(tuple.v1(), tuple.v2()));
        nextStepKey = currentStep.getNextStepKey();
        return state;
    }

    private ClusterState executeWaitStep(ClusterState state, Step currentStep) {
        logger.trace(
            "[{}] waiting for cluster state step condition ({}) [{}]",
            index.getName(),
            currentStep.getClass().getSimpleName(),
            currentStep.getKey()
        );
        ClusterStateWaitStep.Result result = ((ClusterStateWaitStep) currentStep).isConditionMet(index, state);
        nextStepKey = currentStep.getNextStepKey();

        // Check if the wait condition is met.
        if (result.complete()) {
            logger.trace(
                "[{}] cluster state step condition met successfully ({}) [{}]",
                index.getName(),
                currentStep.getClass().getSimpleName(),
                currentStep.getKey()
            );
            return state;
        } else {
            // Condition not met, so we stop execution for this cycle.
            final ToXContentObject stepInfo = result.informationContext();
            if (logger.isTraceEnabled()) {
                logger.trace(
                    "[{}] condition not met ({}) [{}], returning existing state (info: {})",
                    index.getName(),
                    currentStep.getClass().getSimpleName(),
                    currentStep.getKey(),
                    stepInfo == null ? "null" : Strings.toString(stepInfo)
                );
            }
            // Halt advancement by clearing the nextStepKey.
            nextStepKey = null;
            if (stepInfo == null) {
                return state;
            }
            // Add context information about the wait condition to the cluster state.
            return IndexLifecycleTransition.addStepInfoToClusterState(index, state, stepInfo);
        }
    }

    /**
     * Updates the cluster state to move the index to the determined next step.
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
     * This callback is executed after the updated cluster state has been successfully published.
     * Its primary role is to trigger any asynchronous actions required by the new step.
     */
    @Override
    public void onClusterStateProcessed(ClusterState newState) {
        final Metadata metadata = newState.metadata();
        final IndexMetadata indexMetadata = metadata.getProject().index(index);
        if (indexMetadata != null) {

            LifecycleExecutionState exState = indexMetadata.getLifecycleExecutionState();
            if (ErrorStep.NAME.equals(exState.step()) && this.failure != null) {
                lifecycleRunner.registerFailedOperation(indexMetadata, failure);
            } else {
                lifecycleRunner.registerSuccessfulOperation(indexMetadata);
            }

            // If we successfully moved to a new step, check if it requires an async action.
            if (nextStepKey != null && nextStepKey != TerminalPolicyStep.KEY) {
                logger.trace(
                    "[{}] step sequence starting with {} has completed, running next step {} if it is an async action",
                    index.getName(),
                    startStep.getKey(),
                    nextStepKey
                );
                // Trigger the runner to execute the async action (e.g., shrink, force-merge).
                // This happens outside the cluster state update thread.
                lifecycleRunner.maybeRunAsyncAction(newState, indexMetadata, policy, nextStepKey);
            }
        }
        assert indexToStepKeysForAsyncActions.size() <= 1 : "we expect a maximum of one single spawned index currently";
        // Also trigger async actions for any newly spawned indices (e.g., from a shrink).
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
     * Moves the index into an ERROR step in the cluster state.
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
