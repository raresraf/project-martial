/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.rest.action;

import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.ActionRequest;
import org.elasticsearch.action.ActionResponse;
import org.elasticsearch.action.ActionType;
import org.elasticsearch.action.admin.cluster.node.tasks.cancel.CancelTasksRequest;
import org.elasticsearch.action.admin.cluster.node.tasks.cancel.TransportCancelTasksAction;
import org.elasticsearch.action.search.SearchRequest;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.action.search.TransportSearchAction;
import org.elasticsearch.action.support.PlainActionFuture;
import org.elasticsearch.action.support.SubscribableListener;
import org.elasticsearch.client.internal.node.NodeClient;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.util.set.Sets;
import org.elasticsearch.http.HttpChannel;
import org.elasticsearch.http.HttpResponse;
import org.elasticsearch.tasks.Task;
import org.elasticsearch.tasks.TaskId;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.threadpool.TestThreadPool;
import org.elasticsearch.threadpool.ThreadPool;
import org.junit.After;
import org.junit.Before;

import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongSupplier;

/**
 * @brief Unit tests for `RestCancellableNodeClient`, verifying its behavior
 * related to task management, HTTP channel lifecycle, and task cancellation
 * in concurrent scenarios.
 *
 * Functional Utility: These tests ensure that the `RestCancellableNodeClient`
 * correctly tracks tasks associated with HTTP channels, cancels tasks when
 * channels are closed, and handles various edge cases of concurrency
 * between task execution and channel closure.
 *
 * Concurrency: Extensively uses `ThreadPool`, `CountDownLatch`, `AtomicBoolean`,
 * `AtomicInteger`, and `CopyOnWriteArraySet` to simulate concurrent operations
 * and verify thread-safe behavior.
 *
 * Error Handling Patterns: Verifies proper task cancellation upon channel closure
 * and robust handling of scenarios where channels are already closed or tasks
 * complete before being fully tracked.
 */
public class RestCancellableNodeClientTests extends ESTestCase {

    private ThreadPool threadPool; // Thread pool for managing concurrent operations in tests.

    /**
     * @brief Sets up a new `TestThreadPool` before each test.
     */
    @Before
    public void createThreadPool() {
        threadPool = new TestThreadPool(RestCancellableNodeClientTests.class.getName());
    }

    /**
     * @brief Shuts down the `ThreadPool` after each test, ensuring all tasks complete.
     */
    @After
    public void stopThreadPool() {
        ThreadPool.terminate(threadPool, 5, TimeUnit.SECONDS); // Terminate thread pool with a 5-second timeout.
    }

    /**
     * @brief Verifies that no tasks are left in the internal tracking map
     * (`RestCancellableNodeClient.tasks`) when tasks complete normally.
     * This test simulates a scenario where tasks may complete even before they
     * are fully associated with their corresponding HTTP channel.
     * @throws Exception If any future execution fails or is interrupted.
     */
    public void testCompletedTasks() throws Exception {
        final var testClient = new TestClient(Settings.EMPTY, threadPool, false); // Create a TestClient that does not timeout.
        int initialHttpChannels = RestCancellableNodeClient.getNumChannels(); // Get initial number of tracked HTTP channels.
        int totalSearches = 0; // Counter for total search requests.
        List<Future<?>> futures = new ArrayList<>(); // List to hold all submitted futures.
        int numChannels = randomIntBetween(1, 30); // Random number of HTTP channels.

        // Block Logic: Iterate to create multiple HTTP channels and submit tasks.
        for (int i = 0; i < numChannels; i++) {
            int numTasks = randomIntBetween(1, 30); // Random number of tasks per channel.
            TestHttpChannel channel = new TestHttpChannel(); // Create a new test HTTP channel.
            totalSearches += numTasks; // Accumulate total search count.
            // Block Logic: Submit multiple search tasks for the current channel.
            for (int j = 0; j < numTasks; j++) {
                PlainActionFuture<SearchResponse> actionFuture = new PlainActionFuture<>(); // Future to track action completion.
                RestCancellableNodeClient client = new RestCancellableNodeClient(testClient, channel); // Create client for this channel.
                futures.add(
                    threadPool.generic().submit(() -> client.execute(TransportSearchAction.TYPE, new SearchRequest(), actionFuture))
                ); // Submit task to thread pool.
                futures.add(actionFuture); // Add action future to list for waiting.
            }
        }
        // Block Logic: Wait for all submitted futures to complete.
        for (Future<?> future : futures) {
            future.get(); // Blocks until task completes.
        }
        // Invariant: No channels get explicitly closed in this test, so the number of tracked channels should increase.
        assertEquals(initialHttpChannels + numChannels, RestCancellableNodeClient.getNumChannels());
        // Invariant: All tasks should have completed and been removed from tracking.
        assertEquals(0, RestCancellableNodeClient.getNumTasks());
        // Invariant: The total number of search requests initiated by the test client should match.
        assertEquals(totalSearches, testClient.searchRequests.get());
    }

    /**
     * @brief Verifies the behavior when an HTTP channel is explicitly closed.
     * Expected outcome: The channel should be removed from tracking, and all
     * its corresponding tasks should be cancelled.
     * @throws Exception If any future execution fails or is interrupted.
     */
    public void testCancelledTasks() throws Exception {
        final var nodeClient = new TestClient(Settings.EMPTY, threadPool, true); // Create a TestClient that times out.
        int initialHttpChannels = RestCancellableNodeClient.getNumChannels(); // Get initial number of tracked HTTP channels.
        int numChannels = randomIntBetween(1, 30); // Random number of HTTP channels.
        int totalSearches = 0; // Counter for total search requests.
        List<TestHttpChannel> channels = new ArrayList<>(numChannels); // List to hold created test channels.

        // Block Logic: Create multiple HTTP channels and submit tasks.
        for (int i = 0; i < numChannels; i++) {
            TestHttpChannel channel = new TestHttpChannel(); // Create a new test HTTP channel.
            channels.add(channel); // Add channel to list.
            int numTasks = randomIntBetween(1, 30); // Random number of tasks per channel.
            totalSearches += numTasks; // Accumulate total search count.
            RestCancellableNodeClient client = new RestCancellableNodeClient(nodeClient, channel); // Create client for this channel.
            // Block Logic: Submit multiple search tasks for the current channel.
            for (int j = 0; j < numTasks; j++) {
                client.execute(TransportSearchAction.TYPE, new SearchRequest(), null); // Submit task with a null listener (not waiting for response).
            }
            // Invariant: Verify that the correct number of tasks are tracked for this specific channel.
            assertEquals(numTasks, RestCancellableNodeClient.getNumTasks(channel));
        }
        // Invariant: Verify total number of tracked HTTP channels after setup.
        assertEquals(initialHttpChannels + numChannels, RestCancellableNodeClient.getNumChannels());

        // Block Logic: Close each created HTTP channel and wait for the close process.
        for (TestHttpChannel channel : channels) {
            channel.awaitClose(); // Close channel and wait for close listener to execute.
        }
        // Invariant: All created channels should have been removed from tracking.
        assertEquals(initialHttpChannels, RestCancellableNodeClient.getNumChannels());
        // Invariant: The total number of search requests initiated should match.
        assertEquals(totalSearches, nodeClient.searchRequests.get());
        // Invariant: All initiated tasks should have been marked as cancelled.
        assertEquals(totalSearches, nodeClient.cancelledTasks.size());
    }

    /**
     * @brief Verifies the behavior when a request arrives, but its corresponding
     * HTTP channel is already closed.
     * Expected outcome: The close listener is executed immediately, and the task is cancelled.
     * This may result in registering a close listener multiple times, but only the newly
     * added listener is invoked at registration time because the channel is already closed.
     */
    public void testChannelAlreadyClosed() {
        final var testClient = new TestClient(Settings.EMPTY, threadPool, true); // Create a TestClient that times out.
        int initialHttpChannels = RestCancellableNodeClient.getNumChannels(); // Get initial number of tracked HTTP channels.
        int numChannels = randomIntBetween(1, 30); // Random number of HTTP channels.
        int totalSearches = 0; // Counter for total search requests.

        // Block Logic: Create channels, close them immediately, then submit tasks.
        for (int i = 0; i < numChannels; i++) {
            TestHttpChannel channel = new TestHttpChannel(); // Create a new test HTTP channel.
            // Invariant: Close the channel immediately. No need to wait, as no external close listener is registered yet.
            channel.close();
            int numTasks = randomIntBetween(1, 5); // Random number of tasks.
            totalSearches += numTasks; // Accumulate total search count.
            RestCancellableNodeClient client = new RestCancellableNodeClient(testClient, channel); // Create client for this channel.
            // Block Logic: Submit multiple search tasks. The channel will be registered and immediately removed.
            for (int j = 0; j < numTasks; j++) {
                // Invariant: Here, the channel will be first registered, then immediately removed from the map
                // as its close listener is invoked upon registration due to the channel already being closed.
                client.execute(TransportSearchAction.TYPE, new SearchRequest(), null);
            }
        }
        // Invariant: All created channels should have been registered and then immediately deregistered.
        assertEquals(initialHttpChannels, RestCancellableNodeClient.getNumChannels());
        // Invariant: The total number of search requests initiated should match.
        assertEquals(totalSearches, testClient.searchRequests.get());
        // Invariant: All initiated tasks should have been marked as cancelled.
        assertEquals(totalSearches, testClient.cancelledTasks.size());
    }

    /**
     * @brief Verifies the concurrent execution of tasks and channel closure.
     * This test ensures that when tasks are being executed concurrently while
     * an HTTP channel is closing, all tasks associated with that channel are
     * correctly cancelled and the channel is removed from tracking.
     */
    public void testConcurrentExecuteAndClose() {
        final var testClient = new TestClient(Settings.EMPTY, threadPool, true); // Create a TestClient that times out.
        int initialHttpChannels = RestCancellableNodeClient.getNumChannels(); // Get initial number of tracked HTTP channels.
        int numTasks = randomIntBetween(1, 30); // Random number of tasks to submit.
        TestHttpChannel channel = new TestHttpChannel(); // Create a single test HTTP channel.
        final var startLatch = new CountDownLatch(1); // Latch to signal when tasks have started submission.
        final var doneLatch = new CountDownLatch(numTasks + 1); // Latch to signal overall test completion.
        final var expectedTasks = Sets.<TaskId>newHashSetWithExpectedSize(numTasks); // Set to store expected cancelled tasks.

        // Block Logic: Submit multiple tasks concurrently.
        for (int j = 0; j < numTasks; j++) {
            RestCancellableNodeClient client = new RestCancellableNodeClient(testClient, channel); // Client for this channel.
            threadPool.generic().execute(() -> {
                client.execute(TransportSearchAction.TYPE, new SearchRequest(), ActionListener.running(ESTestCase::fail));
                startLatch.countDown(); // Signal that a task has started submission.
                doneLatch.countDown(); // Signal that this task's submission is complete.
            });
            expectedTasks.add(new TaskId(testClient.getLocalNodeId(), j)); // Add expected TaskId to the set.
        }

        // Block Logic: Concurrently close the channel.
        threadPool.generic().execute(() -> {
            try {
                safeAwait(startLatch); // Wait until at least one task has started submission.
                channel.awaitClose(); // Close the channel and wait for its close listener to complete.
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new AssertionError(e);
            } finally {
                doneLatch.countDown(); // Signal that channel closure process is complete.
            }
        });
        safeAwait(doneLatch); // Wait for all tasks and channel closure to complete.
        // Invariant: The channel should have been removed from tracking.
        assertEquals(initialHttpChannels, RestCancellableNodeClient.getNumChannels());
        // Invariant: All tasks that were initiated should have been marked as cancelled.
        assertEquals(expectedTasks, testClient.cancelledTasks);
    }

    /**
     * @brief Mock `NodeClient` implementation used for testing.
     * It simulates task creation and cancellation behavior without actual Elasticsearch operations.
     */
    private static class TestClient extends NodeClient {
        // AtomicLongs to generate unique Task IDs for search and cancel operations.
        private final LongSupplier searchTaskIdGenerator = new AtomicLong(0)::getAndIncrement;
        private final LongSupplier cancelTaskIdGenerator = new AtomicLong(1000)::getAndIncrement;
        // Thread-safe set to track tasks that were cancelled.
        private final Set<TaskId> cancelledTasks = new CopyOnWriteArraySet<>();
        // AtomicInteger to count the number of search requests received.
        private final AtomicInteger searchRequests = new AtomicInteger(0);
        private final boolean timeout; // Flag to control whether tasks will 'timeout' (complete asynchronously).

        /**
         * @brief Constructor for `TestClient`.
         * @param settings Elasticsearch settings.
         * @param threadPool Thread pool for task submission.
         * @param timeout If true, simulated tasks will not complete immediately (allowing for cancellation tests).
         */
        TestClient(Settings settings, ThreadPool threadPool, boolean timeout) {
            super(settings, threadPool);
            this.timeout = timeout;
        }

        /**
         * @brief Simulates local execution of an action.
         * @param action The type of action to execute (e.g., Search, CancelTasks).
         * @param request The action request.
         * @param listener The listener to notify upon completion.
         * @return A simulated `Task` object.
         */
        @Override
        public <Request extends ActionRequest, Response extends ActionResponse> Task executeLocally(
            ActionType<Response> action,
            Request request,
            ActionListener<Response> listener
        ) {
            // Block Logic: Handle different action types.
            switch (action.name()) {
                case TransportCancelTasksAction.NAME -> { // Case for CancelTasks action.
                    // Invariant: Ensure task cancellation is not attempted more than once for the same task.
                    assertTrue(
                        "tried to cancel the same task more than once",
                        cancelledTasks.add(asInstanceOf(CancelTasksRequest.class, request).getTargetTaskId())
                    );
                    Task task = request.createTask( // Create a simulated task for cancellation.
                        cancelTaskIdGenerator.getAsLong(),
                        "cancel_task",
                        action.name(),
                        null,
                        Collections.emptyMap()
                    );
                    // Randomly simulate success or failure of cancellation.
                    if (randomBoolean()) {
                        listener.onResponse(null); // Simulate successful response.
                    } else {
                        // Invariant: Test that cancel tasks is best effort, failures are not propagated.
                        listener.onFailure(new IllegalStateException()); // Simulate failure.
                    }
                    return task;
                }
                case TransportSearchAction.NAME -> { // Case for Search action.
                    searchRequests.incrementAndGet(); // Increment search request counter.
                    Task searchTask = request.createTask( // Create a simulated search task.
                        searchTaskIdGenerator.getAsLong(),
                        "search",
                        action.name(),
                        null,
                        Collections.emptyMap()
                    );
                    // Block Logic: Simulate task completion based on the `timeout` flag.
                    if (timeout == false) {
                        if (rarely()) {
                            // Invariant: Rarely, complete the search from the same thread.
                            listener.onResponse(null);
                        } else {
                            // Invariant: Usually, complete the search asynchronously in a generic thread.
                            threadPool().generic().submit(() -> listener.onResponse(null));
                        }
                    }
                    return searchTask; // Return the simulated search task.
                }
                default -> throw new AssertionError("unexpected action " + action.name()); // Handle unexpected actions.
            }

        }

        /**
         * @brief Returns a mock local node ID.
         * @return A string representing the local node ID.
         */
        @Override
        public String getLocalNodeId() {
            return "node";
        }
    }

    /**
     * @brief Mock `HttpChannel` implementation used for testing.
     * Simulates the lifecycle of an HTTP channel, including opening, closing,
     * and adding close listeners.
     */
    private class TestHttpChannel implements HttpChannel {
        // AtomicBoolean to track if the channel is open.
        private final AtomicBoolean open = new AtomicBoolean(true);
        // A subscribable listener to manage callbacks registered for channel closure.
        private final SubscribableListener<ActionListener<Void>> closeListener = new SubscribableListener<>();
        // Latch to wait for the channel's close listener to complete its execution.
        private final CountDownLatch closeLatch = new CountDownLatch(1);

        @Override
        public void sendResponse(HttpResponse response, ActionListener<Void> listener) {}

        @Override
        public InetSocketAddress getLocalAddress() {
            return null;
        }

        @Override
        public InetSocketAddress getRemoteAddress() {
            return null;
        }

        /**
         * @brief Simulates closing the HTTP channel.
         * Sets the channel's `open` status to false and triggers any registered
         * close listeners.
         */
        @Override
        public void close() {
            // Invariant: Ensure the channel is not closed more than once.
            assertTrue("HttpChannel is already closed", open.compareAndSet(true, false));
            // Block Logic: Notify all registered close listeners.
            closeListener.andThenAccept(listener -> {
                boolean failure = randomBoolean(); // Randomly simulate success or failure of close notification.
                threadPool.generic().submit(() -> {
                    if (failure) {
                        listener.onFailure(new IllegalStateException()); // Simulate failure.
                    } else {
                        listener.onResponse(null); // Simulate successful response.
                    }
                    closeLatch.countDown(); // Decrement latch after listener execution.
                });
            });
        }

        /**
         * @brief Waits for the channel's close process to complete.
         * Calls `close()` and then blocks until `closeLatch` counts down.
         * @throws InterruptedException If the current thread is interrupted while waiting.
         */
        private void awaitClose() throws InterruptedException {
            close(); // Initiate channel closure.
            closeLatch.await(); // Wait for the close listener to complete.
        }

        /**
         * @brief Checks if the HTTP channel is open.
         * @return True if the channel is open, false otherwise.
         */
        @Override
        public boolean isOpen() {
            return open.get();
        }

        /**
         * @brief Adds a listener to be notified when the channel closes.
         * If the channel is already closed, the listener is notified immediately.
         * @param listener The `ActionListener` to be called upon channel closure.
         */
        @Override
        public void addCloseListener(ActionListener<Void> listener) {
            // Block Logic: If the channel is already closed, notify the listener immediately.
            if (open.get() == false) {
                listener.onResponse(null);
                // Invariant: Handle scenario where `awaitClose()` was called before any `addCloseListener()`
                // to ensure `closeLatch` is decremented.
                if (closeListener.isDone() == false) {
                    closeListener.onResponse(ActionListener.noop()); // Trigger internal close listener.
                }
            } else {
                // Invariant: Ensure only one primary close listener is set if the channel is still open.
                assertFalse("close listener already set, only one is allowed!", closeListener.isDone());
                closeListener.onResponse(ActionListener.assertOnce(listener)); // Register the listener.
            }
        }
    }
}