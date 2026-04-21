/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
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
 * @file RestCancellableNodeClientTests.java
 * @brief Unit tests for validating task cancellation upon HTTP channel closure.
 * 
 * Functional Intent: Ensures that the RestCancellableNodeClient correctly tracks 
 * active tasks associated with HTTP channels and triggers cancellation if the 
 * channel is closed before task completion. Tests various race conditions 
 * including tasks completing before registration and channels closing concurrently 
 * with task execution.
 * 
 * Domain: Production Systems, Distributed Computing, Resource Cleanup.
 */
public class RestCancellableNodeClientTests extends ESTestCase {

    private ThreadPool threadPool;

    @Before
    public void createThreadPool() {
        threadPool = new TestThreadPool(RestCancellableNodeClientTests.class.getName());
    }

    @After
    public void stopThreadPool() {
        ThreadPool.terminate(threadPool, 5, TimeUnit.SECONDS);
    }

    /**
     * testCompletedTasks - Verifies cleanup of successfully finished tasks.
     * 
     * Algorithm: Stress test with random load.
     * Logic: 
     * 1. Spawns multiple HTTP channels and associated search tasks.
     * 2. Simulates task completion (sometimes even before channel association).
     * 3. Confirms that no orphaned tasks remain in the tracking map.
     */
    public void testCompletedTasks() throws Exception {
        final var testClient = new TestClient(Settings.EMPTY, threadPool, false);
        int initialHttpChannels = RestCancellableNodeClient.getNumChannels();
        int totalSearches = 0;
        List<Future<?>> futures = new ArrayList<>();
        int numChannels = randomIntBetween(1, 30);
        for (int i = 0; i < numChannels; i++) {
            int numTasks = randomIntBetween(1, 30);
            TestHttpChannel channel = new TestHttpChannel();
            totalSearches += numTasks;
            for (int j = 0; j < numTasks; j++) {
                PlainActionFuture<SearchResponse> actionFuture = new PlainActionFuture<>();
                RestCancellableNodeClient client = new RestCancellableNodeClient(testClient, channel);
                futures.add(
                    threadPool.generic().submit(() -> client.execute(TransportSearchAction.TYPE, new SearchRequest(), actionFuture))
                );
                futures.add(actionFuture);
            }
        }
        for (Future<?> future : futures) {
            future.get();
        }
        
        // Invariant: Completed tasks must be removed from the tracker regardless of channel state.
        assertEquals(initialHttpChannels + numChannels, RestCancellableNodeClient.getNumChannels());
        assertEquals(0, RestCancellableNodeClient.getNumTasks());
        assertEquals(totalSearches, testClient.searchRequests.get());
    }

    /**
     * testCancelledTasks - Verifies that closing a channel aborts all associated tasks.
     * 
     * Logic: 
     * 1. Associates multiple tasks with several HTTP channels.
     * 2. Closes each channel.
     * 3. Validates that the tracker successfully triggered cancellation for every task.
     */
    public void testCancelledTasks() throws Exception {
        final var nodeClient = new TestClient(Settings.EMPTY, threadPool, true);
        int initialHttpChannels = RestCancellableNodeClient.getNumChannels();
        int numChannels = randomIntBetween(1, 30);
        int totalSearches = 0;
        List<TestHttpChannel> channels = new ArrayList<>(numChannels);
        for (int i = 0; i < numChannels; i++) {
            TestHttpChannel channel = new TestHttpChannel();
            channels.add(channel);
            int numTasks = randomIntBetween(1, 30);
            totalSearches += numTasks;
            RestCancellableNodeClient client = new RestCancellableNodeClient(nodeClient, channel);
            for (int j = 0; j < numTasks; j++) {
                client.execute(TransportSearchAction.TYPE, new SearchRequest(), null);
            }
            assertEquals(numTasks, RestCancellableNodeClient.getNumTasks(channel));
        }
        assertEquals(initialHttpChannels + numChannels, RestCancellableNodeClient.getNumChannels());
        for (TestHttpChannel channel : channels) {
            channel.awaitClose();
        }
        
        // Invariant: Closing the channel must purge it from the tracker and cancel pending work.
        assertEquals(initialHttpChannels, RestCancellableNodeClient.getNumChannels());
        assertEquals(totalSearches, nodeClient.searchRequests.get());
        assertEquals(totalSearches, nodeClient.cancelledTasks.size());
    }

    /**
     * testChannelAlreadyClosed - Validates immediate cancellation for dead channels.
     */
    public void testChannelAlreadyClosed() {
        final var testClient = new TestClient(Settings.EMPTY, threadPool, true);
        int initialHttpChannels = RestCancellableNodeClient.getNumChannels();
        int numChannels = randomIntBetween(1, 30);
        int totalSearches = 0;
        for (int i = 0; i < numChannels; i++) {
            TestHttpChannel channel = new TestHttpChannel();
            channel.close();
            int numTasks = randomIntBetween(1, 5);
            totalSearches += numTasks;
            RestCancellableNodeClient client = new RestCancellableNodeClient(testClient, channel);
            for (int j = 0; j < numTasks; j++) {
                // Logic: Client must detect the closed state and cancel the task upon execution.
                client.execute(TransportSearchAction.TYPE, new SearchRequest(), null);
            }
        }
        assertEquals(initialHttpChannels, RestCancellableNodeClient.getNumChannels());
        assertEquals(totalSearches, testClient.searchRequests.get());
        assertEquals(totalSearches, testClient.cancelledTasks.size());
    }

    /**
     * testConcurrentExecuteAndClose - Validates thread-safety during simultaneous execution and shutdown.
     */
    public void testConcurrentExecuteAndClose() {
        final var testClient = new TestClient(Settings.EMPTY, threadPool, true);
        int initialHttpChannels = RestCancellableNodeClient.getNumChannels();
        int numTasks = randomIntBetween(1, 30);
        TestHttpChannel channel = new TestHttpChannel();
        final var startLatch = new CountDownLatch(1);
        final var doneLatch = new CountDownLatch(numTasks + 1);
        final var expectedTasks = Sets.<TaskId>newHashSetWithExpectedSize(numTasks);
        for (int j = 0; j < numTasks; j++) {
            RestCancellableNodeClient client = new RestCancellableNodeClient(testClient, channel);
            threadPool.generic().execute(() -> {
                client.execute(TransportSearchAction.TYPE, new SearchRequest(), ActionListener.running(ESTestCase::fail));
                startLatch.countDown();
                doneLatch.countDown();
            });
            expectedTasks.add(new TaskId(testClient.getLocalNodeId(), j));
        }
        threadPool.generic().execute(() -> {
            try {
                // Synchronization: Ensure at least one task has started before closing the channel.
                safeAwait(startLatch);
                channel.awaitClose();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new AssertionError(e);
            } finally {
                doneLatch.countDown();
            }
        });
        safeAwait(doneLatch);
        assertEquals(initialHttpChannels, RestCancellableNodeClient.getNumChannels());
        assertEquals(expectedTasks, testClient.cancelledTasks);
    }

    /**
     * @class TestClient
     * @brief Mock implementation of NodeClient for tracking request counts and cancellations.
     */
    private static class TestClient extends NodeClient {
        private final LongSupplier searchTaskIdGenerator = new AtomicLong(0)::getAndIncrement;
        private final LongSupplier cancelTaskIdGenerator = new AtomicLong(1000)::getAndIncrement;
        private final Set<TaskId> cancelledTasks = new CopyOnWriteArraySet<>();
        private final AtomicInteger searchRequests = new AtomicInteger(0);
        private final boolean timeout;

        TestClient(Settings settings, ThreadPool threadPool, boolean timeout) {
            super(settings, threadPool);
            this.timeout = timeout;
        }

        @Override
        public <Request extends ActionRequest, Response extends ActionResponse> Task executeLocally(
            ActionType<Response> action,
            Request request,
            ActionListener<Response> listener
        ) {
            switch (action.name()) {
                case TransportCancelTasksAction.NAME -> {
                    // Logic: Validates that cancellation is only triggered once per task.
                    assertTrue(
                        "tried to cancel the same task more than once",
                        cancelledTasks.add(asInstanceOf(CancelTasksRequest.class, request).getTargetTaskId())
                    );
                    Task task = request.createTask(
                        cancelTaskIdGenerator.getAsLong(),
                        "cancel_task",
                        action.name(),
                        null,
                        Collections.emptyMap()
                    );
                    if (randomBoolean()) {
                        listener.onResponse(null);
                    } else {
                        listener.onFailure(new IllegalStateException());
                    }
                    return task;
                }
                case TransportSearchAction.NAME -> {
                    searchRequests.incrementAndGet();
                    Task searchTask = request.createTask(
                        searchTaskIdGenerator.getAsLong(),
                        "search",
                        action.name(),
                        null,
                        Collections.emptyMap()
                    );
                    if (timeout == false) {
                        if (rarely()) {
                            listener.onResponse(null);
                        } else {
                            threadPool().generic().submit(() -> listener.onResponse(null));
                        }
                    }
                    return searchTask;
                }
                default -> throw new AssertionError("unexpected action " + action.name());
            }

        }

        @Override
        public String getLocalNodeId() {
            return "node";
        }
    }

    /**
     * @class TestHttpChannel
     * @brief Mock HTTP channel implementation with subscribable close notifications.
     */
    private class TestHttpChannel implements HttpChannel {
        private final AtomicBoolean open = new AtomicBoolean(true);
        private final SubscribableListener<ActionListener<Void>> closeListener = new SubscribableListener<>();
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

        @Override
        public void close() {
            // Synchronization: Atomic transition to closed state.
            assertTrue("HttpChannel is already closed", open.compareAndSet(true, false));
            closeListener.andThenAccept(listener -> {
                boolean failure = randomBoolean();
                threadPool.generic().submit(() -> {
                    if (failure) {
                        listener.onFailure(new IllegalStateException());
                    } else {
                        listener.onResponse(null);
                    }
                    closeLatch.countDown();
                });
            });
        }

        private void awaitClose() throws InterruptedException {
            close();
            closeLatch.await();
        }

        @Override
        public boolean isOpen() {
            return open.get();
        }

        @Override
        public void addCloseListener(ActionListener<Void> listener) {
            // Logic: Immediate execution if the channel is already dead.
            if (open.get() == false) {
                listener.onResponse(null);
            } else {
                assertFalse("close listener already set, only one is allowed!", closeListener.isDone());
                closeListener.onResponse(ActionListener.assertOnce(listener));
            }
        }
    }
}
