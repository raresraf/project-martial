/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.elasticsearch.xpack.ilm;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.cluster.ClusterState;
import org.elasticsearch.cluster.ClusterStateObserver;
import org.elasticsearch.cluster.ClusterStateTaskExecutor;
import org.elasticsearch.cluster.metadata.IndexMetadata;
import org.elasticsearch.cluster.metadata.LifecycleExecutionState;
import org.elasticsearch.cluster.metadata.Metadata;
import org.elasticsearch.cluster.service.ClusterService;
import org.elasticsearch.cluster.service.MasterServiceTaskQueue;
import org.elasticsearch.common.Priority;
import org.elasticsearch.common.Strings;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.core.SuppressForbidden;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.core.Tuple;
import org.elasticsearch.index.Index;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xpack.core.ilm.AsyncActionStep;
import org.elasticsearch.xpack.core.ilm.AsyncWaitStep;
import org.elasticsearch.xpack.core.ilm.ClusterStateActionStep;
import org.elasticsearch.xpack.core.ilm.ClusterStateWaitStep;
import org.elasticsearch.xpack.core.ilm.ErrorStep;
import org.elasticsearch.xpack.core.ilm.LifecycleSettings;
import org.elasticsearch.xpack.core.ilm.PhaseCompleteStep;
import org.elasticsearch.xpack.core.ilm.Step;
import org.elasticsearch.xpack.core.ilm.Step.StepKey;
import org.elasticsearch.xpack.core.ilm.TerminalPolicyStep;
import org.elasticsearch.xpack.ilm.history.ILMHistoryItem;
import org.elasticsearch.xpack.ilm.history.ILMHistoryStore;

import java.util.Collections;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.function.LongSupplier;

import static org.elasticsearch.core.Strings.format;
import static org.elasticsearch.index.IndexSettings.LIFECYCLE_ORIGINATION_DATE;

 /**
  * @brief Functional description of the IndexLifecycleRunner class.
  *        This is a placeholder for detailed semantic documentation.
  *        Further analysis will elaborate on its algorithm, complexity, and invariants.
  */
class IndexLifecycleRunner {
    private static final Logger logger = LogManager.getLogger(IndexLifecycleRunner.class);
     /**
      * @brief [Functional description for field threadPool]: Describe purpose here.
      */
    private final ThreadPool threadPool;
     /**
      * @brief [Functional description for field clusterService]: Describe purpose here.
      */
    private final ClusterService clusterService;
     /**
      * @brief [Functional description for field stepRegistry]: Describe purpose here.
      */
    private final PolicyStepsRegistry stepRegistry;
     /**
      * @brief [Functional description for field ilmHistoryStore]: Describe purpose here.
      */
    private final ILMHistoryStore ilmHistoryStore;
     /**
      * @brief [Functional description for field nowSupplier]: Describe purpose here.
      */
    private final LongSupplier nowSupplier;
     /**
      * @brief [Functional description for field masterServiceTaskQueue]: Describe purpose here.
      */
    private final MasterServiceTaskQueue<IndexLifecycleClusterStateUpdateTask> masterServiceTaskQueue;

    @SuppressWarnings("Convert2Lambda") // can't SuppressForbidden on a lambda
    private static final ClusterStateTaskExecutor<IndexLifecycleClusterStateUpdateTask> ILM_TASK_EXECUTOR =
        new ClusterStateTaskExecutor<>() {
            @Override
            @SuppressForbidden(reason = "consuming published cluster state for legacy reasons")
            public ClusterState execute(BatchExecutionContext<IndexLifecycleClusterStateUpdateTask> batchExecutionContext) {
                 /**
                  * @brief [Functional description for field state]: Describe purpose here.
                  */
                ClusterState state = batchExecutionContext.initialState();
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                for (final var taskContext : batchExecutionContext.taskContexts()) {
                    try {
                         /**
                          * @brief [Functional description for field task]: Describe purpose here.
                          */
                        final var task = taskContext.getTask();
                        try (var ignored = taskContext.captureResponseHeaders()) {
                            state = task.execute(state);
                        }
                        taskContext.success(
                            publishedState -> task.clusterStateProcessed(batchExecutionContext.initialState(), publishedState)
                        );
                    } catch (Exception e) {
                        taskContext.onFailure(e);
                    }
                }
                return state;
            }
        };

    /**
     * @brief [Functional Utility for IndexLifecycleRunner]: Describe purpose here.
     */
    IndexLifecycleRunner(
        PolicyStepsRegistry stepRegistry,
        ILMHistoryStore ilmHistoryStore,
        ClusterService clusterService,
        ThreadPool threadPool,
        LongSupplier nowSupplier
    ) {
        this.stepRegistry = stepRegistry;
        this.ilmHistoryStore = ilmHistoryStore;
        this.clusterService = clusterService;
        this.nowSupplier = nowSupplier;
        this.threadPool = threadPool;
        this.masterServiceTaskQueue = clusterService.createTaskQueue("ilm-runner", Priority.NORMAL, ILM_TASK_EXECUTOR);
    }

    /**
     * Retrieve the index's current step.
     */
    static Step getCurrentStep(PolicyStepsRegistry stepRegistry, String policy, IndexMetadata indexMetadata) {
        LifecycleExecutionState lifecycleState = indexMetadata.getLifecycleExecutionState();
        return getCurrentStep(stepRegistry, policy, indexMetadata, lifecycleState);
    }

    /**
     * @brief [Functional Utility for getCurrentStep]: Describe purpose here.
     * @return Step: [Description]\n     */
    static Step getCurrentStep(
        PolicyStepsRegistry stepRegistry,
        String policy,
        IndexMetadata indexMetadata,
        LifecycleExecutionState lifecycleState
    ) {
        StepKey currentStepKey = Step.getCurrentStepKey(lifecycleState);
        logger.trace("[{}] retrieved current step key: {}", indexMetadata.getIndex().getName(), currentStepKey);
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (currentStepKey == null) {
            return stepRegistry.getFirstStep(policy);
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        } else {
            return stepRegistry.getStep(indexMetadata, currentStepKey);
        }
    }

    /**
     * Calculate the index's origination time (in milliseconds) based on its
     * metadata. Returns null if there is no lifecycle date and the origination
     * date is not set.
     */
    @Nullable
    private static Long calculateOriginationMillis(final IndexMetadata indexMetadata) {
        LifecycleExecutionState lifecycleState = indexMetadata.getLifecycleExecutionState();
        Long originationDate = indexMetadata.getSettings().getAsLong(LIFECYCLE_ORIGINATION_DATE, -1L);
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (lifecycleState.lifecycleDate() == null && originationDate == -1L) {
            return null;
        }
        return originationDate == -1L ? lifecycleState.lifecycleDate() : originationDate;
    }

    /**
     * Return true or false depending on whether the index is ready to be in {@code phase}
     */
    boolean isReadyToTransitionToThisPhase(final String policy, final IndexMetadata indexMetadata, final String phase) {
        final Long lifecycleDate = calculateOriginationMillis(indexMetadata);
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (lifecycleDate == null) {
            logger.trace("[{}] no index creation or origination date has been set yet", indexMetadata.getIndex().getName());
            return true;
        }
        final TimeValue after = stepRegistry.getIndexAgeForPhase(policy, phase);
        final long now = nowSupplier.getAsLong();
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (logger.isTraceEnabled()) {
            final long ageMillis = now - lifecycleDate;
            final TimeValue age;
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (ageMillis >= 0) {
                age = new TimeValue(ageMillis);
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            } else if (ageMillis == Long.MIN_VALUE) {
                age = new TimeValue(Long.MAX_VALUE);
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            } else {
                age = new TimeValue(-ageMillis);
            }
            logger.trace(
                "[{}] checking for index age to be at least [{}] before performing actions in "
                    + "the \"{}\" phase. Now: {}, lifecycle date: {}, age: [{}{}/{}s]",
                indexMetadata.getIndex().getName(),
                after,
                phase,
                new TimeValue(now).seconds(),
                new TimeValue(lifecycleDate).seconds(),
                ageMillis < 0 ? "-" : "",
                age,
                age.seconds()
            );
        }
        return now >= lifecycleDate + after.getMillis();
    }

    /**
     * Run the current step, only if it is an asynchronous wait step. These
     * wait criteria are checked periodically from the ILM scheduler
     */
    void runPeriodicStep(String policy, Metadata metadata, IndexMetadata indexMetadata) {
        String index = indexMetadata.getIndex().getName();
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (LifecycleSettings.LIFECYCLE_SKIP_SETTING.get(indexMetadata.getSettings())) {
            logger.info("[{}] skipping policy [{}] because [{}] is true", index, policy, LifecycleSettings.LIFECYCLE_SKIP);
            return;
        }
        LifecycleExecutionState lifecycleState = indexMetadata.getLifecycleExecutionState();
        final Step currentStep;
        try {
            currentStep = getCurrentStep(stepRegistry, policy, indexMetadata, lifecycleState);
        } catch (Exception e) {
            markPolicyRetrievalError(policy, indexMetadata.getIndex(), lifecycleState, e);
            return;
        }

         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (currentStep == null) {
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (stepRegistry.policyExists(policy) == false) {
                markPolicyDoesNotExist(policy, indexMetadata.getIndex(), lifecycleState);
                return;
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            } else {
                Step.StepKey currentStepKey = Step.getCurrentStepKey(lifecycleState);
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (TerminalPolicyStep.KEY.equals(currentStepKey)) {
                    // This index is a leftover from before we halted execution on the final phase
                    // instead of going to the completed phase, so it's okay to ignore this index
                    // for now
                    return;
                }
                logger.error("current step [{}] for index [{}] with policy [{}] is not recognized", currentStepKey, index, policy);
                return;
            }
        }

         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (currentStep instanceof TerminalPolicyStep) {
            logger.debug("policy [{}] for index [{}] complete, skipping execution", policy, index);
            return;
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        } else if (currentStep instanceof ErrorStep) {
            onErrorMaybeRetryFailedStep(policy, currentStep.getKey(), indexMetadata);
            return;
        }

        logger.trace(
            "[{}] maybe running periodic step ({}) with current step {}",
            index,
            currentStep.getClass().getSimpleName(),
            currentStep.getKey()
        );
        // Only phase changing and async wait steps should be run through periodic polling
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (currentStep instanceof PhaseCompleteStep) {
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (currentStep.getNextStepKey() == null) {
                logger.debug(
                    "[{}] stopping in the current phase ({}) as there are no more steps in the policy",
                    index,
                    currentStep.getKey().phase()
                );
                return;
            }
            // Only proceed to the next step if enough time has elapsed to go into the next phase
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (isReadyToTransitionToThisPhase(policy, indexMetadata, currentStep.getNextStepKey().phase())) {
                moveToStep(indexMetadata.getIndex(), policy, currentStep.getKey(), currentStep.getNextStepKey());
            }
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        } else if (currentStep instanceof AsyncWaitStep) {
            logger.debug("[{}] running periodic policy with current-step [{}]", index, currentStep.getKey());
            ((AsyncWaitStep) currentStep).evaluateCondition(metadata, indexMetadata.getIndex(), new AsyncWaitStep.Listener() {

                @Override
                public void onResponse(boolean conditionMet, ToXContentObject stepInfo) {
                    logger.trace("cs-change-async-wait-callback, [{}] current-step: {}", index, currentStep.getKey());
                     // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                     // Invariant: [State condition that holds true before and after each iteration/execution]\n                    if (conditionMet) {
                        moveToStep(indexMetadata.getIndex(), policy, currentStep.getKey(), currentStep.getNextStepKey());
                     // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                     // Invariant: [State condition that holds true before and after each iteration/execution]\n                    } else if (stepInfo != null) {
                        setStepInfo(indexMetadata.getIndex(), policy, currentStep.getKey(), stepInfo);
                    }
                }

                @Override
                public void onFailure(Exception e) {
                    moveToErrorStep(indexMetadata.getIndex(), policy, currentStep.getKey(), e);
                }
            }, TimeValue.MAX_VALUE);
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        } else {
            logger.trace("[{}] ignoring non periodic step execution from step transition [{}]", index, currentStep.getKey());
        }
    }

    /**
     * Given the policy and index metadata for an index, this moves the index's
     * execution state to the previously failed step, incrementing the retry
     * counter.
     */
    void onErrorMaybeRetryFailedStep(String policy, StepKey currentStep, IndexMetadata indexMetadata) {
        String index = indexMetadata.getIndex().getName();
        LifecycleExecutionState lifecycleState = indexMetadata.getLifecycleExecutionState();
        Step failedStep = stepRegistry.getStep(
            indexMetadata,
            new StepKey(lifecycleState.phase(), lifecycleState.action(), lifecycleState.failedStep())
        );
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (failedStep == null) {
            logger.warn(
                "failed step [{}] for index [{}] is not part of policy [{}] anymore, or it is invalid. skipping execution",
                lifecycleState.failedStep(),
                index,
                policy
            );
            return;
        }

         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (lifecycleState.isAutoRetryableError() != null && lifecycleState.isAutoRetryableError()) {
            int currentRetryAttempt = lifecycleState.failedStepRetryCount() == null ? 1 : 1 + lifecycleState.failedStepRetryCount();
            logger.info(
                "policy [{}] for index [{}] on an error step due to a transient error, moving back to the failed "
                    + "step [{}] for execution. retry attempt [{}]",
                policy,
                index,
                lifecycleState.failedStep(),
                currentRetryAttempt
            );
            // we can afford to drop these requests if they timeout as on the next {@link
            // IndexLifecycleRunner#runPeriodicStep} run the policy will still be in the ERROR step, as we haven't been able
            // to move it back into the failed step, so we'll try again
            submitUnlessAlreadyQueued(
                Strings.format("ilm-retry-failed-step {policy [%s], index [%s], failedStep [%s]}", policy, index, failedStep.getKey()),
                new MoveToRetryFailedStepUpdateTask(indexMetadata.getIndex(), policy, currentStep, failedStep)
            );
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        } else {
            logger.debug("policy [{}] for index [{}] on an error step after a terminal error, skipping execution", policy, index);
        }
    }

    /**
     * If the current step (matching the expected step key) is an asynchronous action step, run it
     */
    void maybeRunAsyncAction(ClusterState currentState, IndexMetadata indexMetadata, String policy, StepKey expectedStepKey) {
        String index = indexMetadata.getIndex().getName();
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (LifecycleSettings.LIFECYCLE_SKIP_SETTING.get(indexMetadata.getSettings())) {
            logger.info("[{}] skipping policy [{}] because [{}] is true", index, policy, LifecycleSettings.LIFECYCLE_SKIP);
            return;
        }
        LifecycleExecutionState lifecycleState = indexMetadata.getLifecycleExecutionState();
        final Step currentStep;
        try {
            currentStep = getCurrentStep(stepRegistry, policy, indexMetadata, lifecycleState);
        } catch (Exception e) {
            markPolicyRetrievalError(policy, indexMetadata.getIndex(), lifecycleState, e);
            return;
        }
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (currentStep == null) {
            Step.StepKey currentStepKey = Step.getCurrentStepKey(lifecycleState);
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (TerminalPolicyStep.KEY.equals(currentStepKey)) {
                // This index is a leftover from before we halted execution on the final phase
                // instead of going to the completed phase, so it's okay to ignore this index
                // for now
                return;
            }
            logger.warn("current step [{}] for index [{}] with policy [{}] is not recognized", currentStepKey, index, policy);
            return;
        }
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (expectedStepKey.phase() == null && expectedStepKey.name() == null && expectedStepKey.action() == null) {
            // ILM is stopped, so do not try to run async action
            logger.debug("expected step for index [{}] with policy [{}] is [{}], not running async action", index, policy, expectedStepKey);
            return;
        }
        logger.trace(
            "[{}] maybe running async action step ({}) with current step {}",
            index,
            currentStep.getClass().getSimpleName(),
            currentStep.getKey()
        );
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (currentStep.getKey().equals(expectedStepKey) == false) {
            throw new IllegalStateException(
                "expected index ["
                    + indexMetadata.getIndex().getName()
                    + "] with policy ["
                    + policy
                    + "] to have current step consistent with provided step key ("
                    + expectedStepKey
                    + ") but it was "
                    + currentStep.getKey()
            );
        }
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (currentStep instanceof AsyncActionStep) {
            logger.debug("[{}] running policy with async action step [{}]", index, currentStep.getKey());
            ((AsyncActionStep) currentStep).performAction(
                indexMetadata,
                currentState,
                new ClusterStateObserver(clusterService, null, logger, threadPool.getThreadContext()),
                new ActionListener<>() {

                    @Override
                    public void onResponse(Void unused) {
                        logger.trace("cs-change-async-action-callback, [{}], current-step: {}", index, currentStep.getKey());
                         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                         // Invariant: [State condition that holds true before and after each iteration/execution]\n                        if (((AsyncActionStep) currentStep).indexSurvives()) {
                            moveToStep(indexMetadata.getIndex(), policy, currentStep.getKey(), currentStep.getNextStepKey());
                         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                         // Invariant: [State condition that holds true before and after each iteration/execution]\n                        } else {
                            // Delete needs special handling, because after this step we
                            // will no longer have access to any information about the
                            // index since it will be... deleted.
                            registerDeleteOperation(indexMetadata);
                        }
                    }

                    @Override
                    public void onFailure(Exception e) {
                        moveToErrorStep(indexMetadata.getIndex(), policy, currentStep.getKey(), e);
                    }
                }
            );
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        } else {
            logger.trace("[{}] ignoring non async action step execution from step transition [{}]", index, currentStep.getKey());
        }
    }

    /**
     * Run the current step that either waits for index age, or updates/waits-on cluster state.
     * Invoked after the cluster state has been changed
     */
    void runPolicyAfterStateChange(String policy, IndexMetadata indexMetadata) {
        String index = indexMetadata.getIndex().getName();
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (LifecycleSettings.LIFECYCLE_SKIP_SETTING.get(indexMetadata.getSettings())) {
            logger.info("[{}] skipping policy [{}] because [{}] is true", index, policy, LifecycleSettings.LIFECYCLE_SKIP);
            return;
        }
        LifecycleExecutionState lifecycleState = indexMetadata.getLifecycleExecutionState();
        final StepKey currentStepKey = Step.getCurrentStepKey(lifecycleState);
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (busyIndices.contains(Tuple.tuple(indexMetadata.getIndex(), currentStepKey))) {
            // try later again, already doing work for this index at this step, no need to check for more work yet
            return;
        }
        final Step currentStep;
        try {
            currentStep = getCurrentStep(stepRegistry, policy, indexMetadata, lifecycleState);
        } catch (Exception e) {
            markPolicyRetrievalError(policy, indexMetadata.getIndex(), lifecycleState, e);
            return;
        }
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (currentStep == null) {
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (stepRegistry.policyExists(policy) == false) {
                markPolicyDoesNotExist(policy, indexMetadata.getIndex(), lifecycleState);
                return;
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            } else {
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (TerminalPolicyStep.KEY.equals(currentStepKey)) {
                    // This index is a leftover from before we halted execution on the final phase
                    // instead of going to the completed phase, so it's okay to ignore this index
                    // for now
                    return;
                }
                logger.error("current step [{}] for index [{}] with policy [{}] is not recognized", currentStepKey, index, policy);
                return;
            }
        }

         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (currentStep instanceof TerminalPolicyStep) {
            logger.debug("policy [{}] for index [{}] complete, skipping execution", policy, index);
            return;
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        } else if (currentStep instanceof ErrorStep) {
            logger.debug("policy [{}] for index [{}] on an error step, skipping execution", policy, index);
            return;
        }

        logger.trace(
            "[{}] maybe running step ({}) after state change: {}",
            index,
            currentStep.getClass().getSimpleName(),
            currentStep.getKey()
        );
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (currentStep instanceof PhaseCompleteStep) {
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (currentStep.getNextStepKey() == null) {
                logger.debug(
                    "[{}] stopping in the current phase ({}) as there are no more steps in the policy",
                    index,
                    currentStep.getKey().phase()
                );
                return;
            }
            // Only proceed to the next step if enough time has elapsed to go into the next phase
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (isReadyToTransitionToThisPhase(policy, indexMetadata, currentStep.getNextStepKey().phase())) {
                moveToStep(indexMetadata.getIndex(), policy, currentStep.getKey(), currentStep.getNextStepKey());
            }
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        } else if (currentStep instanceof ClusterStateActionStep || currentStep instanceof ClusterStateWaitStep) {
            logger.debug("[{}] running policy with current-step [{}]", indexMetadata.getIndex().getName(), currentStep.getKey());
            submitUnlessAlreadyQueued(
                Strings.format("ilm-execute-cluster-state-steps [%s]", currentStep),
                new ExecuteStepsUpdateTask(policy, indexMetadata.getIndex(), currentStep, stepRegistry, this, nowSupplier)
            );
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        } else {
            logger.trace("[{}] ignoring step execution from cluster state change event [{}]", index, currentStep.getKey());
        }
    }

    /**
     * Move the index to the given {@code newStepKey}, always checks to ensure that the index's
     * current step matches the {@code currentStepKey} prior to changing the state.
     */
    private void moveToStep(Index index, String policy, Step.StepKey currentStepKey, Step.StepKey newStepKey) {
        logger.debug("[{}] moving to step [{}] {} -> {}", index.getName(), policy, currentStepKey, newStepKey);
        submitUnlessAlreadyQueued(
            Strings.format(
                "ilm-move-to-step {policy [%s], index [%s], currentStep [%s], nextStep [%s]}",
                policy,
                index.getName(),
                currentStepKey,
                newStepKey
            ),
            new MoveToNextStepUpdateTask(index, policy, currentStepKey, newStepKey, nowSupplier, stepRegistry, clusterState -> {
                IndexMetadata indexMetadata = clusterState.metadata().getProject().index(index);
                registerSuccessfulOperation(indexMetadata);
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (newStepKey != null && newStepKey != TerminalPolicyStep.KEY && indexMetadata != null) {
                    maybeRunAsyncAction(clusterState, indexMetadata, policy, newStepKey);
                }
            })
        );
    }

    /**
     * Move the index to the ERROR step.
     */
    private void moveToErrorStep(Index index, String policy, Step.StepKey currentStepKey, Exception e) {
        logger.error(
            () -> format("policy [%s] for index [%s] failed on step [%s]. Moving to ERROR step", policy, index.getName(), currentStepKey),
            e
        );
        submitUnlessAlreadyQueued(
            Strings.format("ilm-move-to-error-step {policy [%s], index [%s], currentStep [%s]}", policy, index.getName(), currentStepKey),
            new MoveToErrorStepUpdateTask(index, policy, currentStepKey, e, nowSupplier, stepRegistry::getStep, clusterState -> {
                IndexMetadata indexMetadata = clusterState.metadata().getProject().index(index);
                registerFailedOperation(indexMetadata, e);
            })
        );
    }

    /**
     * Set step info for the given index inside of its {@link LifecycleExecutionState} without
     * changing other execution state.
     */
    private void setStepInfo(Index index, String policy, @Nullable Step.StepKey currentStepKey, ToXContentObject stepInfo) {
        submitUnlessAlreadyQueued(
            Strings.format("ilm-set-step-info {policy [%s], index [%s], currentStep [%s]}", policy, index.getName(), currentStepKey),
            new SetStepInfoUpdateTask(index, policy, currentStepKey, stepInfo)
        );
    }

    /**
     * Mark the index with step info explaining that the policy doesn't exist.
     */
    private void markPolicyDoesNotExist(String policyName, Index index, LifecycleExecutionState executionState) {
        markPolicyRetrievalError(
            policyName,
            index,
            executionState,
            new IllegalArgumentException("policy [" + policyName + "] does not exist")
        );
    }

    /**
     * Mark the index with step info for a given error encountered while retrieving policy
     * information. This is opposed to lifecycle execution errors, which would cause a transition to
     * the ERROR step, however, the policy may be unparseable in which case there is no way to move
     * to the ERROR step, so this is the best effort at capturing the error retrieving the policy.
     */
    private void markPolicyRetrievalError(String policyName, Index index, LifecycleExecutionState executionState, Exception e) {
        logger.debug(
            () -> format(
                "unable to retrieve policy [%s] for index [%s], recording this in step_info for this index",
                policyName,
                index.getName()
            ),
            e
        );
        setStepInfo(index, policyName, Step.getCurrentStepKey(executionState), new SetStepInfoUpdateTask.ExceptionWrapper(e));
    }

    /**
     * For the given index metadata, register (index a document) that the index has transitioned
     * successfully into this new state using the {@link ILMHistoryStore}
     */
    void registerSuccessfulOperation(IndexMetadata indexMetadata) {
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (indexMetadata == null) {
            // This index may have been deleted and has no metadata, so ignore it
            return;
        }
        Long origination = calculateOriginationMillis(indexMetadata);
        ilmHistoryStore.putAsync(
            ILMHistoryItem.success(
                indexMetadata.getIndex().getName(),
                indexMetadata.getLifecyclePolicyName(),
                nowSupplier.getAsLong(),
                origination == null ? null : (nowSupplier.getAsLong() - origination),
                indexMetadata.getLifecycleExecutionState()
            )
        );
    }

    /**
     * For the given index metadata, register (index a document) that the index
     * has been deleted by ILM using the {@link ILMHistoryStore}
     */
    void registerDeleteOperation(IndexMetadata metadataBeforeDeletion) {
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (metadataBeforeDeletion == null) {
            throw new IllegalStateException("cannot register deletion of an index that did not previously exist");
        }
        Long origination = calculateOriginationMillis(metadataBeforeDeletion);
                    // Register that the delete phase is now "complete"
        ilmHistoryStore.putAsync(
            ILMHistoryItem.success(
                metadataBeforeDeletion.getIndex().getName(),
                metadataBeforeDeletion.getLifecyclePolicyName(),
                nowSupplier.getAsLong(),
                origination == null ? null : (nowSupplier.getAsLong() - origination),
                LifecycleExecutionState.builder(metadataBeforeDeletion.getLifecycleExecutionState())
                    .setStep(PhaseCompleteStep.NAME)
                    .build()
            )
        );
    }

    /**
     * For the given index metadata, register (index a document) that the index has transitioned
     * into the ERROR state using the {@link ILMHistoryStore}
     */
    void registerFailedOperation(IndexMetadata indexMetadata, Exception failure) {
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (indexMetadata == null) {
            // This index may have been deleted and has no metadata, so ignore it
            return;
        }
        Long origination = calculateOriginationMillis(indexMetadata);
        ilmHistoryStore.putAsync(
            ILMHistoryItem.failure(
                indexMetadata.getIndex().getName(),
                indexMetadata.getLifecyclePolicyName(),
                nowSupplier.getAsLong(),
                origination == null ? null : (nowSupplier.getAsLong() - origination),
                indexMetadata.getLifecycleExecutionState(),
                failure
            )
        );
    }

    private final Set<IndexLifecycleClusterStateUpdateTask> executingTasks = Collections.synchronizedSet(new HashSet<>());

    /**
     * Set of all index and current step key combinations that have an in-flight cluster state update at the moment. Used to not inspect
     * indices that are already executing an update at their current step on cluster state update thread needlessly.
     */
    private final Set<Tuple<Index, StepKey>> busyIndices = Collections.synchronizedSet(new HashSet<>());

    /**
     * Tracks already executing {@link IndexLifecycleClusterStateUpdateTask} tasks in {@link #executingTasks} to prevent queueing up
     * duplicate cluster state updates.
     * TODO: refactor ILM logic so that this is not required any longer. It is unreasonably expensive to only filter out duplicate tasks at
     *       this point given how these tasks are mostly set up on the cluster state applier thread.
     *
     * @param source source string as used in {@link ClusterService#submitUnbatchedStateUpdateTask}
     * @param task   task to submit unless already tracked in {@link #executingTasks}.
     */
    private void submitUnlessAlreadyQueued(String source, IndexLifecycleClusterStateUpdateTask task) {
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (executingTasks.add(task)) {
            final Tuple<Index, StepKey> dedupKey = Tuple.tuple(task.index, task.currentStepKey);
            // index+step-key combination on a best-effort basis to skip checking for more work for an index on CS application
            busyIndices.add(dedupKey);
            task.addListener(ActionListener.running(() -> {
                final boolean removed = executingTasks.remove(task);
                busyIndices.remove(dedupKey);
                assert removed : "tried to unregister unknown task [" + task + "]";
            }));
            masterServiceTaskQueue.submitTask(source, task, null);
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        } else {
            logger.trace("skipped redundant execution of [{}]", source);
        }
    }

    private final class MoveToRetryFailedStepUpdateTask extends IndexLifecycleClusterStateUpdateTask {

        private final String policy;
        private final Step failedStep;

        MoveToRetryFailedStepUpdateTask(Index index, String policy, StepKey currentStep, Step failedStep) {
            super(index, currentStep);
            this.policy = policy;
            this.failedStep = failedStep;
        }

        @Override
        protected ClusterState doExecute(ClusterState currentState) {
            final var updatedProject = IndexLifecycleTransition.moveClusterStateToPreviouslyFailedStep(
                currentState.metadata().getProject(),
                index.getName(),
                nowSupplier,
                stepRegistry,
                true
            );
            return ClusterState.builder(currentState).putProjectMetadata(updatedProject).build();
        }

        @Override
        public boolean equals(Object other) {
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (this == other) {
                return true;
            }
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (other instanceof MoveToRetryFailedStepUpdateTask == false) {
                return false;
            }
            final MoveToRetryFailedStepUpdateTask that = (MoveToRetryFailedStepUpdateTask) other;
            return index.equals(that.index)
                && policy.equals(that.policy)
                && currentStepKey.equals(that.currentStepKey)
                && this.failedStep.equals(that.failedStep);
        }

        @Override
        public int hashCode() {
            return Objects.hash(index, policy, currentStepKey, failedStep);
        }

        @Override
        protected void handleFailure(Exception e) {
            logger.error(() -> format("retry execution of step [%s] for index [%s] failed", failedStep.getKey().name(), index), e);
        }

        @Override
        protected void onClusterStateProcessed(ClusterState newState) {
            IndexMetadata newIndexMeta = newState.metadata().getProject().index(index);
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (newIndexMeta == null) {
                // index was deleted
                return;
            }
            Step indexMetaCurrentStep = getCurrentStep(stepRegistry, policy, newIndexMeta);
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (indexMetaCurrentStep == null) {
                // no step found
                return;
            }
            StepKey stepKey = indexMetaCurrentStep.getKey();
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (stepKey != null && stepKey != TerminalPolicyStep.KEY) {
                logger.trace(
                    "policy [{}] for index [{}] was moved back on the failed step for as part of an automatic "
                        + "retry. Attempting to execute the failed step [{}] if it's an async action",
                    policy,
                    index,
                    stepKey
                );
                maybeRunAsyncAction(newState, newIndexMeta, policy, stepKey);
            }

        }
    }
}
