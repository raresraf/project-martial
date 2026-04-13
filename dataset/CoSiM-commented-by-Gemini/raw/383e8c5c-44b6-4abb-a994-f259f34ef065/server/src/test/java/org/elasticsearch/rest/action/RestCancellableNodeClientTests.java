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
 * This class contains tests for the {@link RestCancellableNodeClient}.
 * The tests verify that tasks executed via this client are correctly tracked
 * against their corresponding HTTP channels and are properly cancelled when the
 * channel is closed. This is critical for preventing orphaned tasks on the cluster
 * when a client disconnects.
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
     * This test verifies the "happy path" where tasks complete successfully. It ensures
     * that the internal tracking maps within {@link RestCancellableNodeClient} are cleaned
     * up correctly, preventing memory leaks, even when the task finishes before the channel closes.
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
        // no channels get closed in this test, hence we expect as many channels as we created in the map
        assertEquals(initialHttpChannels + numChannels, RestCancellableNodeClient.getNumChannels());
        // All tasks should have been removed from the map upon completion.
        assertEquals(0, RestCancellableNodeClient.getNumTasks());
        assertEquals(totalSearches, testClient.searchRequests.get());
    }

    /**
     * This test verifies the core cancellation logic. It simulates multiple long-running
     * tasks and then closes their associated HTTP channels. It asserts that closing the
     * channel triggers a cancellation for every task that was running on it.
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
        // Close all channels, which should trigger cancellations.
        for (TestHttpChannel channel : channels) {
            channel.awaitClose();
        }
        // Verify that all channels and their tasks have been removed from tracking.
        assertEquals(initialHttpChannels, RestCancellableNodeClient.getNumChannels());
        assertEquals(totalSearches, nodeClient.searchRequests.get());
        // Verify that a cancellation was issued for every task.
        assertEquals(totalSearches, nodeClient.cancelledTasks.size());
    }

    /**
     * This test verifies the edge case where a request is executed on an HTTP channel
     * that has already been closed. It ensures that the task is immediately cancelled
     * and that no resources are leaked.
     */
    public void testChannelAlreadyClosed() {
        final var testClient = new TestClient(Settings.EMPTY, threadPool, true);
        int initialHttpChannels = RestCancellableNodeClient.getNumChannels();
        int numChannels = randomIntBetween(1, 30);
        int totalSearches = 0;
        for (int i = 0; i < numChannels; i++) {
            TestHttpChannel channel = new TestHttpChannel();
            // Close the channel before executing any requests on it.
            channel.close();
            int numTasks = randomIntBetween(1, 5);
            totalSearches += numTasks;
            RestCancellableNodeClient client = new RestCancellableNodeClient(testClient, channel);
            for (int j = 0; j < numTasks; j++) {
                // The close listener should be invoked immediately upon registration, cancelling the task.
                client.execute(TransportSearchAction.TYPE, new SearchRequest(), null);
            }
        }
        assertEquals(initialHttpChannels, RestCancellableNodeClient.getNumChannels());
        assertEquals(totalSearches, testClient.searchRequests.get());
        assertEquals(totalSearches, testClient.cancelledTasks.size());
    }

    /**
     * This test checks for race conditions between concurrent task execution and channel closure.
     * It submits multiple tasks and a channel close operation simultaneously to ensure that
     * the tracking and cancellation logic is thread-safe and that all tasks are
     * eventually cancelled correctly, regardless of timing.
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
                // Wait until at least one task has started before closing the channel.
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
     * A mock {@link NodeClient} implementation for testing purposes.
     * It intercepts action executions to track which tasks are started and which are cancelled.
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
                case TransportCancelTasksAction.NAME: {
                    // Track which tasks are being cancelled.
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
                    // Simulate successful or failed cancellation response.
                    if (randomBoolean()) {
                        listener.onResponse(null);
                    } else {
                        // test that cancel tasks is best effort, failure received are not propagated
                        listener.onFailure(new IllegalStateException());
                    }
                    return task;
                }
                case TransportSearchAction.NAME: {
                    searchRequests.incrementAndGet();
                    Task searchTask = request.createTask(
                        searchTaskIdGenerator.getAsLong(),
                        "search",
                        action.name(),
                        null,
                        Collections.emptyMap()
                    );
                    // If timeout is false, the task completes successfully, either immediately or asynchronously.
                    // If timeout is true, the listener is never called, simulating a long-running task.
                    if (timeout == false) {
                        if (rarely()) {
                            // make sure that search is sometimes also called from the same thread before the task is returned
                            listener.onResponse(null);
                        } else {
                            threadPool().generic().submit(() -> listener.onResponse(null));
                        }
                    }
                    return searchTask;
                }
                default:
                    throw new AssertionError("unexpected action " + action.name());
            }

        }

        @Override
        public String getLocalNodeId() {
            return "node";
        }
    }

    /**
     * A mock {@link HttpChannel} implementation for testing purposes.
     * It allows tests to control the channel's lifecycle (open/closed) and
     * inspect the close listeners that are registered.
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
            // Atomically close the channel and invoke any registered close listener.
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
            // if the channel is already closed, the listener gets notified immediately, from the same thread.
            if (open.get() == false) {
                listener.onResponse(null);
                // Handle scenario where awaitClose() was called before any calls to addCloseListener(), this ensures closeLatch is pulled.
                if (closeListener.isDone() == false) {
                    closeListener.onResponse(ActionListener.noop());
                }
            } else {
                assertFalse("close listener already set, only one is allowed!", closeListener.isDone());
                closeListener.onResponse(ActionListener.assertOnce(listener));
            }
        }
    }
}
