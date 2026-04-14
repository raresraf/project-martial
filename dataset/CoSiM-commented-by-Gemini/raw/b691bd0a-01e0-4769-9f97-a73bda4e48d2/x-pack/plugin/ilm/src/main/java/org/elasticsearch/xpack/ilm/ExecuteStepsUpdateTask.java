/**
 * @file ExecuteStepsUpdateTask.java
 * @brief This file defines the core task for executing Index Lifecycle Management (ILM) steps in Elasticsearch.
 *
 * @details This class, `ExecuteStepsUpdateTask`, is a `ClusterStateUpdateTask`, which is the fundamental
 * mechanism in Elasticsearch for applying changes to the cluster state in a safe, atomic manner. This
 * specific task is responsible for advancing an index through its lifecycle policy by executing
 * one or more consecutive steps.
 *
 * Algorithm:
 * The task operates as a state machine. It begins at a specific `startStep` for a given index and policy.
 * It executes steps sequentially in a `while` loop as long as they are of type `ClusterStateActionStep`
 * or a met `ClusterStateWaitStep`.
 * - `ClusterStateActionStep`: These steps directly modify the `ClusterState` (e.g., changing index settings).
 *   After execution, the task transitions the index's lifecycle state to the next step.
 * - `ClusterStateWaitStep`: These steps check a condition against the `ClusterState`. If the condition is met,
 *   the task transitions to the next step. If not, the task terminates, leaving the index in the wait step
 *   to be re-evaluated later.
 * The loop terminates when it encounters a step that requires an asynchronous action (like shrinking an index),
 * a wait condition that is not met, or a phase transition. Asynchronous actions are then triggered in the
 * `onClusterStateProcessed` callback after the state has been successfully updated.
 *
 * Production Systems:
 * This is a critical component of Elasticsearch's ILM feature. Its correct and robust execution ensures that
 * indices are managed according to their policies, which is essential for data retention, performance, and cost
 * management in a production cluster. Error handling is crucial; if a step fails, the task moves the index
 * to a dedicated ERROR step to allow for manual intervention.
 */
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
 * A {@link ClusterStateUpdateTask} that executes a specific {@link Step} for a given
 * index and policy. It is the primary mechanism for advancing an index's lifecycle.
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
     * Constructs a new task to execute ILM steps.
     * @param policy The name of the ILM policy being executed.
     * @param index The index that the policy is being applied to.
     * @param startStep The first step in the sequence to be executed by this task.
     * @param policyStepsRegistry A registry to look up step definitions.
     * @param lifecycleRunner A runner to coordinate execution and handle async actions.
     * @param nowSupplier A supplier for the current time in milliseconds, used for timestamps.
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
     * Executes a sequence of ILM steps against the current cluster state.
     *
     * {@link Step}s for the current index and policy are executed in succession until the next step to be
     * executed is not a {@link ClusterStateActionStep}, or not a {@link ClusterStateWaitStep}, or does not
     * belong to the same phase as the executed step. All other types of steps are executed outside of this
     * {@link ClusterStateUpdateTask}, so they are of no concern here.
     *
     * @param currentState The current state to execute the <code>startStep</code> with
     * @return the new cluster state after cluster-state operations and step transitions are applied
     * @throws IOException if any exceptions occur
     */
    @Override
    public ClusterState doExecute(final ClusterState currentState) throws IOException {
        Step currentStep = startStep;
        IndexMetadata indexMetadata = currentState.metadata().index(index);
        // Pre-condition: If the index has been deleted, there's nothing to do.
        if (indexMetadata == null) {
            logger.debug("lifecycle for index [{}] executed but index no longer exists", index.getName());
            return currentState;
        }
        Step registeredCurrentStep = IndexLifecycleRunner.getCurrentStep(policyStepsRegistry, policy, indexMetadata);
        // Pre-condition: Ensure that the step we are about to execute is still the current step for the index.
        // This prevents executing a stale task if the cluster state has changed since the task was submitted.
        if (currentStep.equals(registeredCurrentStep)) {
            ClusterState state = currentState;
            // Block Logic: This `while` loop is the core engine. It executes consecutive steps that can be
            // performed atomically within a single cluster state update.
            // Invariant: `currentStep` is always a `ClusterStateActionStep` or `ClusterStateWaitStep` inside the loop.
            while (currentStep instanceof ClusterStateActionStep || currentStep instanceof ClusterStateWaitStep) {
                if (currentStep instanceof ClusterStateActionStep) {
                    // This is a step that directly modifies the cluster state.
                    logger.trace(
                        "[{}] performing cluster state action ({}) [{}]",
                        index.getName(),
                        currentStep.getClass().getSimpleName(),
                        currentStep.getKey()
                    );
                    try {
                        ClusterStateActionStep actionStep = (ClusterStateActionStep) currentStep;
                        // Perform the action and get the modified cluster state.
                        state = actionStep.performAction(index, state);
                        // Some actions may spawn new indices (e.g., shrink) which also need ILM management.
                        // This registers the new index for potential async action invocation.
                        Optional.ofNullable(actionStep.indexForAsyncInvocation())
                            .ifPresent(tuple -> indexToStepKeysForAsyncActions.put(tuple.v1(), tuple.v2()));
                    } catch (Exception exception) {
                        // If the action fails, move the index to the ERROR step.
                        return moveToErrorStep(state, currentStep.getKey(), exception);
                    }
                    // The action was successful, so prepare to transition to the next step.
                    nextStepKey = currentStep.getNextStepKey();
                    if (nextStepKey == null) {
                        return state;
                    } else {
                        logger.trace("[{}] moving cluster state to next step [{}]", index.getName(), nextStepKey);
                        state = IndexLifecycleTransition.moveClusterStateToStep(
                            index,
                            state,
                            nextStepKey,
                            nowSupplier,
                            policyStepsRegistry,
                            false
                        );
                    }
                } else {
                    // This is a step that waits for a specific condition to be met.
                    logger.trace(
                        "[{}] waiting for cluster state step condition ({}) [{}]",
                        index.getName(),
                        currentStep.getClass().getSimpleName(),
                        currentStep.getKey()
                    );
                    ClusterStateWaitStep.Result result;
                    try {
                        // Evaluate the wait condition.
                        result = ((ClusterStateWaitStep) currentStep).isConditionMet(index, state);
                    } catch (Exception exception) {
                        return moveToErrorStep(state, currentStep.getKey(), exception);
                    }

                    nextStepKey = currentStep.getNextStepKey();
                    // Block Logic: If the wait condition is met, transition to the next step.
                    if (result.complete()) {
                        logger.trace(
                            "[{}] cluster state step condition met successfully ({}) [{}], moving to next step {}",
                            index.getName(),
                            currentStep.getClass().getSimpleName(),
                            currentStep.getKey(),
                            nextStepKey
                        );
                        if (nextStepKey == null) {
                            return state;
                        } else {
                            state = IndexLifecycleTransition.moveClusterStateToStep(
                                index,
                                state,
                                nextStepKey,
                                nowSupplier,
                                policyStepsRegistry,
                                false
                            );
                        }
                    } else {
                        // Condition not met, so stop execution and wait for the next trigger.
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
                        nextStepKey = null; // Do not attempt to run an async action.
                        if (stepInfo == null) {
                            return state;
                        } else {
                            // Update the cluster state with information about the wait condition.
                            return IndexLifecycleTransition.addStepInfoToClusterState(index, state, stepInfo);
                        }
                    }
                }
                // Pre-condition: Check if a phase transition is about to occur.
                // If so, exit the loop to allow for specific phase-transition logic to run.
                if (currentStep.getKey().phase().equals(currentStep.getNextStepKey().phase()) == false) {
                    return state;
                }
                // Advance to the next step to continue the loop.
                currentStep = policyStepsRegistry.getStep(indexMetadata, currentStep.getNextStepKey());
            }
            return state;
        } else {
            // The current step in the cluster state is different from the one this task was created for.
            // This indicates a stale task, so we do nothing.
            return currentState;
        }
    }

    /**
     * Callback executed after the cluster state has been successfully updated.
     * Its primary role is to trigger any necessary asynchronous actions for the new current step.
     * @param newState The newly applied cluster state.
     */
    @Override
    public void onClusterStateProcessed(ClusterState newState) {
        final Metadata metadata = newState.metadata();
        final IndexMetadata indexMetadata = metadata.index(index);
        if (indexMetadata != null) {
            // Register the operation's outcome (success or failure) with the lifecycle runner.
            LifecycleExecutionState exState = indexMetadata.getLifecycleExecutionState();
            if (ErrorStep.NAME.equals(exState.step()) && this.failure != null) {
                lifecycleRunner.registerFailedOperation(indexMetadata, failure);
            } else {
                lifecycleRunner.registerSuccessfulOperation(indexMetadata);
            }

            // Block Logic: If the state transition resulted in a new step, and that step is not the terminal step,
            // check if the new step requires an asynchronous action (e.g., shrink, force merge).
            // If it does, the `lifecycleRunner` will execute it.
            if (nextStepKey != null && nextStepKey != TerminalPolicyStep.KEY) {
                logger.trace(
                    "[{}] step sequence starting with {} has completed, running next step {} if it is an async action",
                    index.getName(),
                    startStep.getKey(),
                    nextStepKey
                );
                lifecycleRunner.maybeRunAsyncAction(newState, indexMetadata, policy, nextStepKey);
            }
        }
        // Post-condition: Check for any newly spawned indices that need their own async actions triggered.
        assert indexToStepKeysForAsyncActions.size() <= 1 : "we expect a maximum of one single spawned index currently";
        for (Map.Entry<String, Step.StepKey> indexAndStepKey : indexToStepKeysForAsyncActions.entrySet()) {
            final String indexName = indexAndStepKey.getKey();
            final Step.StepKey nextStep = indexAndStepKey.getValue();
            final IndexMetadata indexMeta = metadata.index(indexName);
            if (indexMeta != null) {
                if (newState.metadata().isIndexManagedByILM(indexMeta)) {
                    if (nextStep != null && nextStep != TerminalPolicyStep.KEY) {
                        logger.trace(
                            "[{}] index has been spawed from a different index's ({}) "
                                + "ILM execution, running next step {} if it is an async action",
                            indexName,
                            index,
                            nextStep
                        );
                        final String policyName = LifecycleSettings.LIFECYCLE_NAME_SETTING.get(indexMeta.getSettings());
                        lifecycleRunner.maybeRunAsyncAction(newState, indexMeta, policyName, nextStep);
                    }
                }
            }
        }
    }

    /**
     * Callback executed if the cluster state update task itself fails to be applied.
     * @param e The exception that caused the failure.
     */
    @Override
    public void handleFailure(Exception e) {
        logger.warn(() -> format("policy [%s] for index [%s] failed on step [%s].", policy, index, startStep.getKey()), e);
    }

    /**
     * A helper method to transition the index into the ERROR step.
     * @param state The current cluster state.
     * @param currentStepKey The key of the step that failed.
     * @param cause The exception that caused the failure.
     * @return A new ClusterState where the index is in the ERROR step.
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

    /**
     * Equality check for task deduplication in the cluster state update queue.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ExecuteStepsUpdateTask that = (ExecuteStepsUpdateTAgE_SETTING.get(indexMeta.getSettings());
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
