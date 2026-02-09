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
  * @brief Functional description of the ExecuteStepsUpdateTask class.
  *        This is a placeholder for detailed semantic documentation.
  *        Further analysis will elaborate on its algorithm, complexity, and invariants.
  */
public class ExecuteStepsUpdateTask extends IndexLifecycleClusterStateUpdateTask {
    private static final Logger logger = LogManager.getLogger(ExecuteStepsUpdateTask.class);
     /**
      * @brief [Functional description for field policy]: Describe purpose here.
      */
    private final String policy;
     /**
      * @brief [Functional description for field startStep]: Describe purpose here.
      */
    private final Step startStep;
     /**
      * @brief [Functional description for field policyStepsRegistry]: Describe purpose here.
      */
    private final PolicyStepsRegistry policyStepsRegistry;
     /**
      * @brief [Functional description for field lifecycleRunner]: Describe purpose here.
      */
    private final IndexLifecycleRunner lifecycleRunner;
     /**
      * @brief [Functional description for field nowSupplier]: Describe purpose here.
      */
    private final LongSupplier nowSupplier;
    private final Map<String, Step.StepKey> indexToStepKeysForAsyncActions;
     /**
      * @brief [Functional description for field nextStepKey]: Describe purpose here.
      */
    private Step.StepKey nextStepKey = null;
     /**
      * @brief [Functional description for field failure]: Describe purpose here.
      */
    private Exception failure = null;

    /**
     * @brief [Functional Utility for ExecuteStepsUpdateTask]: Describe purpose here.
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
         /**
          * @brief [Functional description for field policy]: Describe purpose here.
          */
        return policy;
    }

    Step getStartStep() {
        return startStep;
    }

    Step.StepKey getNextStepKey() {
        return nextStepKey;
    }

    /**
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
        IndexMetadata indexMetadata = currentState.metadata().getProject().index(index);
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (indexMetadata == null) {
            logger.debug("lifecycle for index [{}] executed but index no longer exists", index.getName());
            // This index doesn't exist any more, there's nothing to execute currently
            return currentState;
        }
        Step registeredCurrentStep = IndexLifecycleRunner.getCurrentStep(policyStepsRegistry, policy, indexMetadata);
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (currentStep.equals(registeredCurrentStep)) {
            ClusterState state = currentState;
            // We can do cluster state steps all together until we
            // either get to a step that isn't a cluster state step or a
            // cluster state wait step returns not completed
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            while (currentStep instanceof ClusterStateActionStep || currentStep instanceof ClusterStateWaitStep) {
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (currentStep instanceof ClusterStateActionStep) {
                    // cluster state action step so do the action and
                    // move the cluster state to the next step
                    logger.trace(
                        "[{}] performing cluster state action ({}) [{}]",
                        index.getName(),
                        currentStep.getClass().getSimpleName(),
                        currentStep.getKey()
                    );
                    try {
                        ClusterStateActionStep actionStep = (ClusterStateActionStep) currentStep;
                        state = actionStep.performAction(index, state);
                        // If this step (usually a CopyExecutionStateStep step) has brought the
                        // index to where it needs to have async actions invoked, then add that
                        // index to the list so that when the new cluster state has been
                        // processed, the new indices will have their async actions invoked.
                        Optional.ofNullable(actionStep.indexForAsyncInvocation())
                            .ifPresent(tuple -> indexToStepKeysForAsyncActions.put(tuple.v1(), tuple.v2()));
                    } catch (Exception exception) {
                        return moveToErrorStep(state, currentStep.getKey(), exception);
                    }
                    // set here to make sure that the clusterProcessed knows to execute the
                    // correct step if it an async action
                    nextStepKey = currentStep.getNextStepKey();
                     // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                     // Invariant: [State condition that holds true before and after each iteration/execution]\n                    if (nextStepKey == null) {
                        return state;
                     // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                     // Invariant: [State condition that holds true before and after each iteration/execution]\n                    } else {
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
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                } else {
                    // cluster state wait step so evaluate the
                    // condition, if the condition is met move to the
                    // next step, if its not met return the current
                    // cluster state so it can be applied and we will
                    // wait for the next trigger to evaluate the
                    // condition again
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
                    // some steps can decide to change the next step to execute after waiting for some time for the condition
                    // to be met (eg. {@link LifecycleSettings#LIFECYCLE_STEP_WAIT_TIME_THRESHOLD_SETTING}, so it's important we
                    // re-evaluate what the next step is after we evaluate the condition
                    nextStepKey = currentStep.getNextStepKey();
                     // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                     // Invariant: [State condition that holds true before and after each iteration/execution]\n                    if (result.complete()) {
                        logger.trace(
                            "[{}] cluster state step condition met successfully ({}) [{}], moving to next step {}",
                            index.getName(),
                            currentStep.getClass().getSimpleName(),
                            currentStep.getKey(),
                            nextStepKey
                        );
                         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                         // Invariant: [State condition that holds true before and after each iteration/execution]\n                        if (nextStepKey == null) {
                            return state;
                         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                         // Invariant: [State condition that holds true before and after each iteration/execution]\n                        } else {
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
                     // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                     // Invariant: [State condition that holds true before and after each iteration/execution]\n                    } else {
                        final ToXContentObject stepInfo = result.informationContext();
                         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                         // Invariant: [State condition that holds true before and after each iteration/execution]\n                        if (logger.isTraceEnabled()) {
                            logger.trace(
                                "[{}] condition not met ({}) [{}], returning existing state (info: {})",
                                index.getName(),
                                currentStep.getClass().getSimpleName(),
                                currentStep.getKey(),
                                stepInfo == null ? "null" : Strings.toString(stepInfo)
                            );
                        }
                        // We may have executed a step and set "nextStepKey" to
                        // a value, but in this case, since the condition was
                        // not met, we can't advance any way, so don't attempt
                        // to run the current step
                        nextStepKey = null;
                         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                         // Invariant: [State condition that holds true before and after each iteration/execution]\n                        if (stepInfo == null) {
                            return state;
                         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                         // Invariant: [State condition that holds true before and after each iteration/execution]\n                        } else {
                            return ClusterState.builder(state)
                                .putProjectMetadata(
                                    IndexLifecycleTransition.addStepInfoToClusterState(index, state.metadata().getProject(), stepInfo)
                                )
                                .build();
                        }
                    }
                }
                // There are actions we need to take in the event a phase
                // transition happens, so even if we would continue in the while
                // loop, if we are about to go into a new phase, return so that
                // other processing can occur
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (currentStep.getKey().phase().equals(currentStep.getNextStepKey().phase()) == false) {
                    return state;
                }
                currentStep = policyStepsRegistry.getStep(indexMetadata, currentStep.getNextStepKey());
            }
            return state;
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        } else {
            // either we are no longer the master or the step is now
            // not the same as when we submitted the update task. In
            // either case we don't want to do anything now
            return currentState;
        }
    }

    @Override
    public void onClusterStateProcessed(ClusterState newState) {
        final Metadata metadata = newState.metadata();
        final IndexMetadata indexMetadata = metadata.getProject().index(index);
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (indexMetadata != null) {

            LifecycleExecutionState exState = indexMetadata.getLifecycleExecutionState();
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (ErrorStep.NAME.equals(exState.step()) && this.failure != null) {
                lifecycleRunner.registerFailedOperation(indexMetadata, failure);
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            } else {
                lifecycleRunner.registerSuccessfulOperation(indexMetadata);
            }

             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (nextStepKey != null && nextStepKey != TerminalPolicyStep.KEY) {
                logger.trace(
                    "[{}] step sequence starting with {} has completed, running next step {} if it is an async action",
                    index.getName(),
                    startStep.getKey(),
                    nextStepKey
                );
                // After the cluster state has been processed and we have moved
                // to a new step, we need to conditionally execute the step iff
                // it is an `AsyncAction` so that it is executed exactly once.
                lifecycleRunner.maybeRunAsyncAction(newState, indexMetadata, policy, nextStepKey);
            }
        }
        assert indexToStepKeysForAsyncActions.size() <= 1 : "we expect a maximum of one single spawned index currently";
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        for (Map.Entry<String, Step.StepKey> indexAndStepKey : indexToStepKeysForAsyncActions.entrySet()) {
            final String indexName = indexAndStepKey.getKey();
            final Step.StepKey nextStep = indexAndStepKey.getValue();
            final IndexMetadata indexMeta = metadata.getProject().index(indexName);
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (indexMeta != null) {
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (newState.metadata().getProject().isIndexManagedByILM(indexMeta)) {
                     // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                     // Invariant: [State condition that holds true before and after each iteration/execution]\n                    if (nextStep != null && nextStep != TerminalPolicyStep.KEY) {
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
        final var project = state.metadata().getProject();
        return ClusterState.builder(state)
            .putProjectMetadata(
                IndexLifecycleTransition.moveIndexToErrorStep(index, project, cause, nowSupplier, policyStepsRegistry::getStep)
            )
            .build();
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
