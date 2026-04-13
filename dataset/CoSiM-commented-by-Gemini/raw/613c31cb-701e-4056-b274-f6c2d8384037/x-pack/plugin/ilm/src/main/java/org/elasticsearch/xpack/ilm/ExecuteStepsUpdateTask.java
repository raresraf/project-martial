/**
 * This file defines the ExecuteStepsUpdateTask class, a core component of the
 * Index Lifecycle Management (ILM) feature in Elasticsearch. This task is responsible
 * for atomically executing one or more steps of an index's lifecycle policy within
 * the cluster state.
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
 * A {@link ClusterStateUpdateTask} that executes a sequence of ILM steps for a
 * given index. It is responsible for processing steps that can be executed
 * directly on the cluster state, such as {@link ClusterStateActionStep} and
 * {@link ClusterStateWaitStep}.
 *
 * The task iterates through steps, applying changes to the cluster state,
 * until it encounters a step that requires an asynchronous action or when a
 * phase transition occurs.
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
     *
     * @param policy              The name of the ILM policy being executed.
     * @param index               The index being managed.
     * @param startStep           The first step to execute in this task.
     * @param policyStepsRegistry A registry to look up step definitions.
     * @param lifecycleRunner     The runner responsible for coordinating ILM execution.
     * @param nowSupplier         A supplier for the current time in milliseconds.
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
     * Executes the sequence of ILM steps.
     * <p>
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
        if (indexMetadata == null) {
            logger.debug("lifecycle for index [{}] executed but index no longer exists", index.getName());
            // This index doesn't exist any more, there's nothing to execute currently
            return currentState;
        }
        Step registeredCurrentStep = IndexLifecycleRunner.getCurrentStep(policyStepsRegistry, policy, indexMetadata);
        if (currentStep.equals(registeredCurrentStep) == false) {
            // either we are no longer the master or the step is now
            // not the same as when we submitted the update task. In
            // either case we don't want to do anything now
            return currentState;
        }
        ClusterState state = currentState;
        // We can do cluster state steps all together until we
        // either get to a step that isn't a cluster state step or a
        // cluster state wait step returns not completed
        while (currentStep instanceof ClusterStateActionStep || currentStep instanceof ClusterStateWaitStep) {
            try {
                if (currentStep instanceof ClusterStateActionStep) {
                    state = executeActionStep(state, currentStep);
                } else {
                    state = executeWaitStep(state, currentStep);
                }
            } catch (Exception exception) {
                return moveToErrorStep(state, currentStep.getKey(), exception);
            }
            if (nextStepKey == null) {
                // The wait step condition was not met, so we stop and wait for the next trigger.
                return state;
            } else {
                state = moveToNextStep(state);
            }
            // There are actions we need to take in the event a phase
            // transition happens, so even if we would continue in the while
            // loop, if we are about to go into a new phase, return so that
            // other processing can occur
            if (currentStep.getKey().phase().equals(currentStep.getNextStepKey().phase()) == false) {
                return state;
            }
            currentStep = policyStepsRegistry.getStep(indexMetadata, currentStep.getNextStepKey());
        }
        return state;
    }

    /**
     * Executes a {@link ClusterStateActionStep}, which directly modifies the cluster state.
     *
     * @param state       The current cluster state.
     * @param currentStep The step to execute.
     * @return The modified cluster state.
     */
    private ClusterState executeActionStep(ClusterState state, Step currentStep) {
        // cluster state action step so do the action and
        // move the cluster state to the next step
        logger.trace(
            "[{}] performing cluster state action ({}) [{}]",
            index.getName(),
            currentStep.getClass().getSimpleName(),
            currentStep.getKey()
        );
        ClusterStateActionStep actionStep = (ClusterStateActionStep) currentStep;
        state = actionStep.performAction(index, state);
        // If this step (usually a CopyExecutionStateStep step) has brought the
        // index to where it needs to have async actions invoked, then add that
        // index to the list so that when the new cluster state has been
        // processed, the new indices will have their async actions invoked.
        Optional.ofNullable(actionStep.indexForAsyncInvocation())
            .ifPresent(tuple -> indexToStepKeysForAsyncActions.put(tuple.v1(), tuple.v2()));
        // set here to make sure that the clusterProcessed knows to execute the
        // correct step if it an async action
        nextStepKey = currentStep.getNextStepKey();
        return state;
    }

    /**
     * Executes a {@link ClusterStateWaitStep}, which checks if a condition is met.
     * If the condition is met, the process continues to the next step. If not, the
     * task ends, and the system will wait for a future trigger.
     *
     * @param state       The current cluster state.
     * @param currentStep The step to execute.
     * @return The cluster state, which may be modified with step information.
     */
    private ClusterState executeWaitStep(ClusterState state, Step currentStep) {
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
        ClusterStateWaitStep.Result result = ((ClusterStateWaitStep) currentStep).isConditionMet(index, state);
        // some steps can decide to change the next step to execute after waiting for some time for the condition
        // to be met (eg. {@link LifecycleSettings#LIFECYCLE_STEP_WAIT_TIME_THRESHOLD_SETTING}, so it's important we
        // re-evaluate what the next step is after we evaluate the condition
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
            // We may have executed a step and set "nextStepKey" to
            // a value, but in this case, since the condition was
            // not met, we can't advance any way, so don't attempt
            // to run the current step
            nextStepKey = null;
            if (stepInfo == null) {
                return state;
            }
            // Add information about the wait condition to the cluster state
            return IndexLifecycleTransition.addStepInfoToClusterState(index, state, stepInfo);
        }
    }

    /**
     * Updates the cluster state to reflect the transition to the next step.
     *
     * @param state The current cluster state.
     * @return The new cluster state with updated lifecycle metadata.
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
     * A callback executed after the cluster state has been successfully updated.
     * It is responsible for triggering any asynchronous actions required by the new step
     * and registering the operation's success or failure.
     *
     * @param newState The cluster state after this task's execution.
     */
    @Override
    public void onClusterStateProcessed(ClusterState newState) {
        final Metadata metadata = newState.metadata();
        final IndexMetadata indexMetadata = metadata.index(index);
        if (indexMetadata != null) {

            LifecycleExecutionState exState = indexMetadata.getLifecycleExecutionState();
            if (ErrorStep.NAME.equals(exState.step()) && this.failure != null) {
                lifecycleRunner.registerFailedOperation(indexMetadata, failure);
            } else {
                lifecycleRunner.registerSuccessfulOperation(indexMetadata);
            }

            if (nextStepKey != null && nextStepKey != TerminalPolicyStep.KEY) {
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
     * A callback to handle failures during the execution of this cluster state update task.
     * @param e The exception that caused the failure.
     */
    @Override
    public void handleFailure(Exception e) {
        logger.warn(() -> format("policy [%s] for index [%s] failed on step [%s].", policy, index, startStep.getKey()), e);
    }

    /**
     * Moves the index into the ERROR step. This is called when an exception
     * occurs during the execution of a step.
     *
     * @param state          The current cluster state.
     * @param currentStepKey The key of the step that failed.
     * @param cause          The exception that caused the failure.
     * @return The new cluster state with the index moved to the ERROR step.
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
