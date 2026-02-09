/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.compute.operator.exchange;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.ElasticsearchTimeoutException;
import org.elasticsearch.ResourceNotFoundException;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.ActionListenerResponseHandler;
import org.elasticsearch.action.ActionResponse;
import org.elasticsearch.action.support.ChannelActionListener;
import org.elasticsearch.action.support.SubscribableListener;
import org.elasticsearch.common.breaker.CircuitBreakingException;
import org.elasticsearch.common.component.AbstractLifecycleComponent;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.util.concurrent.AbstractRunnable;
import org.elasticsearch.common.util.concurrent.ConcurrentCollections;
import org.elasticsearch.common.util.concurrent.EsRejectedExecutionException;
import org.elasticsearch.compute.data.BlockFactory;
import org.elasticsearch.compute.data.BlockStreamInput;
import org.elasticsearch.compute.data.Page;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.tasks.CancellableTask;
import org.elasticsearch.tasks.Task;
import org.elasticsearch.tasks.TaskCancelledException;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.transport.AbstractTransportRequest;
import org.elasticsearch.transport.Transport;
import org.elasticsearch.transport.TransportChannel;
import org.elasticsearch.transport.TransportRequestHandler;
import org.elasticsearch.transport.TransportRequestOptions;
import org.elasticsearch.transport.TransportService;
import org.elasticsearch.transport.Transports;

import java.io.IOException;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * {@link ExchangeService} is responsible for exchanging pages between exchange sinks and sources on the same or different nodes.
 * It holds a map of {@link ExchangeSinkHandler} instances for each node in the cluster to serve {@link ExchangeRequest}s
 * To connect exchange sources to exchange sinks,
 * use {@link ExchangeSourceHandler#addRemoteSink(RemoteSink, boolean, Runnable, int, ActionListener)}.
 *
 * This service manages the lifecycle of data exchanges, including handling remote requests,
 * creating and finishing sink handlers, and cleaning up inactive resources. It relies on
 * Elasticsearch's transport service for inter-node communication and thread pools for
 * asynchronous operations.
 *
 * Algorithm: Distributed page exchange via request/response model with active/inactive state management.
 * Time Complexity: O(1) for most operations (e.g., getting/creating handlers) due to map-based lookups,
 *                  O(N) for periodic cleanup of inactive sinks where N is the number of sinks.
 * Space Complexity: O(S) for storing sink and source handlers, where S is the number of active exchanges.
 */
public final class ExchangeService extends AbstractLifecycleComponent {
    // TODO: Make this a child action of the data node transport to ensure that exchanges
    // are accessed only by the user initialized the session.
    /**
     * @brief The name of the transport action used for exchanging pages between nodes.
     * Functional Utility: Identifies the action for sending and receiving data pages during an exchange operation.
     */
    public static final String EXCHANGE_ACTION_NAME = "internal:data/read/esql/exchange";
    /**
     * @brief The name of the cross-cluster transport action used for exchanging pages between nodes,
     *        specifically for cross-cluster search (CCS) scenarios.
     * Functional Utility: Extends page exchange functionality to data nodes across different clusters.
     */
    public static final String EXCHANGE_ACTION_NAME_FOR_CCS = "cluster:internal:data/read/esql/exchange";

    /**
     * @brief The name of the transport action used to open a new exchange session on a remote node.
     * Functional Utility: Initiates the setup of an exchange sink handler on a target node.
     */
    public static final String OPEN_EXCHANGE_ACTION_NAME = "internal:data/read/esql/open_exchange";
    /**
     * @brief The name of the cross-cluster transport action used to open a new exchange session on a remote node,
     *        specifically for cross-cluster search (CCS) scenarios.
     * Functional Utility: Enables the initiation of exchange sessions across different clusters.
     */
    private static final String OPEN_EXCHANGE_ACTION_NAME_FOR_CCS = "cluster:internal:data/read/esql/open_exchange";

    /**
     * @brief Setting key for the time interval after which an exchange sink handler is considered inactive.
     * Functional Utility: Configures the timeout for automatic cleanup of unused exchange sink resources.
     */
    public static final String INACTIVE_SINKS_INTERVAL_SETTING = "esql.exchange.sink_inactive_interval";
    /**
     * @brief Default value for the inactive sinks interval setting.
     * Functional Utility: Provides a standard timeout duration for inactive exchange sinks.
     */
    public static final TimeValue INACTIVE_SINKS_INTERVAL_DEFAULT = TimeValue.timeValueMinutes(5);

    private static final Logger LOGGER = LogManager.getLogger(ExchangeService.class);

    private final ThreadPool threadPool;
    private final Executor executor;
    private final BlockFactory blockFactory;

    private final Map<String, ExchangeSinkHandler> sinks = ConcurrentCollections.newConcurrentMap();
    private final Map<String, ExchangeSourceHandler> exchangeSources = ConcurrentCollections.newConcurrentMap();

    /**
     * @brief Constructs a new ExchangeService.
     * @param settings The Elasticsearch settings for configuration.
     * @param threadPool The thread pool for scheduling tasks.
     * @param executorName The name of the executor to use from the thread pool.
     * @param blockFactory The factory for creating and managing data blocks.
     * Functional Utility: Initializes the ExchangeService, setting up internal components and scheduling the inactive sink reaper.
     */
    public ExchangeService(Settings settings, ThreadPool threadPool, String executorName, BlockFactory blockFactory) {
        this.threadPool = threadPool;
        this.executor = threadPool.executor(executorName);
        this.blockFactory = blockFactory;
        final var inactiveInterval = settings.getAsTime(INACTIVE_SINKS_INTERVAL_SETTING, INACTIVE_SINKS_INTERVAL_DEFAULT);
        /**
         * Block Logic: Schedules a periodic task to clean up inactive exchange sinks.
         * Functional Utility: Prevents resource leaks by automatically removing {@link ExchangeSinkHandler} instances
         *                     that have not been used for a configured duration, optimizing memory usage.
         * Invariant: The `InactiveSinksReaper` runs at regular intervals to maintain a clean state of active sinks.
         */
        this.threadPool.scheduleWithFixedDelay(
            new InactiveSinksReaper(LOGGER, threadPool, inactiveInterval),
            TimeValue.timeValueMillis(Math.max(1, inactiveInterval.millis() / 2)),
            executor
        );
    }

    /**
     * @brief Registers the transport request handlers for page exchange operations.
     * @param transportService The TransportService instance to register handlers with.
     * Functional Utility: Sets up the communication endpoints for both regular and cross-cluster
     *                     exchange requests, allowing other nodes to send data pages and initiate exchanges.
     */
    public void registerTransportHandler(TransportService transportService) {
        /**
         * Block Logic: Registers the primary handler for inter-node page exchange requests.
         * Functional Utility: Enables the receiving node to accept and process incoming {@link ExchangeRequest}s
         *                     for data pages, executing them on the dedicated executor.
         */
        transportService.registerRequestHandler(EXCHANGE_ACTION_NAME, this.executor, ExchangeRequest::new, new ExchangeTransportAction());
        /**
         * Block Logic: Registers the handler for requests to open new exchange sessions.
         * Functional Utility: Allows remote nodes to request the creation of a new {@link ExchangeSinkHandler}
         *                     on this node, specifying the session ID and buffer size.
         */
        transportService.registerRequestHandler(
            OPEN_EXCHANGE_ACTION_NAME,
            this.executor,
            OpenExchangeRequest::new,
            new OpenExchangeRequestHandler()
        );

        // This allows the system user access this action when executed over CCS and the API key based security model is in use
        /**
         * Block Logic: Registers the cross-cluster handler for inter-node page exchange requests.
         * Functional Utility: Extends the data exchange capability to nodes in different clusters,
         *                     facilitating cross-cluster search (CCS) operations.
         */
        transportService.registerRequestHandler(
            EXCHANGE_ACTION_NAME_FOR_CCS,
            this.executor,
            ExchangeRequest::new,
            new ExchangeTransportAction()
        );
        /**
         * Block Logic: Registers the cross-cluster handler for requests to open new exchange sessions.
         * Functional Utility: Allows remote nodes in different clusters to request the creation of a
         *                     new {@link ExchangeSinkHandler} on this node.
         */
        transportService.registerRequestHandler(
            OPEN_EXCHANGE_ACTION_NAME_FOR_CCS,
            this.executor,
            OpenExchangeRequest::new,
            new OpenExchangeRequestHandler()
        );
    }

    /**
     * @brief Creates and registers a new {@link ExchangeSinkHandler} for a given exchange ID.
     * @param exchangeId The unique identifier for the exchange.
     * @param maxBufferSize The maximum size of the buffer for the sink handler.
     * @return The newly created {@link ExchangeSinkHandler} instance.
     * @throws IllegalStateException if a sink handler with the provided `exchangeId` already exists.
     * Functional Utility: Provides a mechanism to set up a receiver endpoint for data pages for a specific exchange session.
     * Pre-condition: No {@link ExchangeSinkHandler} with the given `exchangeId` should already be registered.
     * Post-condition: A new {@link ExchangeSinkHandler} is created and mapped to `exchangeId` within the service.
     */
    public ExchangeSinkHandler createSinkHandler(String exchangeId, int maxBufferSize) {
        ExchangeSinkHandler sinkHandler = new ExchangeSinkHandler(blockFactory, maxBufferSize, threadPool.relativeTimeInMillisSupplier());
        if (sinks.putIfAbsent(exchangeId, sinkHandler) != null) {
            throw new IllegalStateException("sink exchanger for id [" + exchangeId + "] already exists");
        }
        return sinkHandler;
    }

    /**
     * @brief Retrieves an existing {@link ExchangeSinkHandler} associated with the given exchange ID.
     * @param exchangeId The unique identifier of the exchange sink handler to retrieve.
     * @return The {@link ExchangeSinkHandler} instance.
     * @throws ResourceNotFoundException if no sink handler for the given ID exists.
     * Functional Utility: Allows components to access a previously created and registered sink handler to interact with its data.
     * Pre-condition: An {@link ExchangeSinkHandler} with the given `exchangeId` must have been previously created.
     */
    public ExchangeSinkHandler getSinkHandler(String exchangeId) {
        ExchangeSinkHandler sinkHandler = sinks.get(exchangeId);
        if (sinkHandler == null) {
            throw new ResourceNotFoundException("sink exchanger for id [{}] doesn't exist", exchangeId);
        }
        return sinkHandler;
    }

    /**
     * @brief Removes and optionally aborts the {@link ExchangeSinkHandler} associated with the given exchange ID.
     * @param exchangeId The unique identifier of the exchange sink handler to finish.
     * @param failure An optional exception indicating a failure reason; if not null, the sink handler will be aborted with this exception.
     * Functional Utility: Cleans up resources associated with a completed or failed exchange session, ensuring proper state management.
     * Pre-condition: The sink handler identified by `exchangeId` may or may not exist.
     * Post-condition: The {@link ExchangeSinkHandler} is removed from the service's active sinks map.
     */
    public void finishSinkHandler(String exchangeId, @Nullable Exception failure) {
        /**
         * Block Logic: Checks if a {@link ExchangeSinkHandler} exists for the given `exchangeId` before proceeding.
         * Functional Utility: Prevents `NullPointerException` and ensures that cleanup operations are only applied
         *                     to active sink handlers. If a `failure` is provided, the sink handler is explicitly
         *                     aborted to propagate the error.
         * Invariant: If `sinkHandler` is found, its lifecycle is managed, and it is asserted to be finished post-operation.
         */
        final ExchangeSinkHandler sinkHandler = sinks.remove(exchangeId);
        if (sinkHandler != null) {
            if (failure != null) {
                sinkHandler.onFailure(failure);
            }
            assert sinkHandler.isFinished() : "Exchange sink " + exchangeId + " wasn't finished yet";
        }
    }

    /**
     * @brief Initiates the opening of a remote {@link ExchangeSinkHandler} on a specified remote node.
     * @param transportService The {@link TransportService} to send the request.
     * @param connection The {@link Transport.Connection} to the remote node.
     * @param sessionId The unique identifier for the exchange session.
     * @param exchangeBuffer The desired buffer size for the remote sink handler.
     * @param responseExecutor The {@link Executor} to handle the asynchronous response.
     * @param listener The {@link ActionListener} to be notified upon completion.
     * Functional Utility: Establishes a communication channel and sets up a data receiving endpoint on a remote node,
     *                     preparing it to receive pages for a specific exchange session.
     * Post-condition: A request is sent to the remote node to create and prepare an {@link ExchangeSinkHandler}.
     */
    public static void openExchange(
        TransportService transportService,
        Transport.Connection connection,
        String sessionId,
        int exchangeBuffer,
        Executor responseExecutor,
        ActionListener<Void> listener
    ) {
        transportService.sendRequest(
            connection,
            OPEN_EXCHANGE_ACTION_NAME,
            new OpenExchangeRequest(sessionId, exchangeBuffer),
            TransportRequestOptions.EMPTY,
            new ActionListenerResponseHandler<>(listener.map(unused -> null), in -> ActionResponse.Empty.INSTANCE, responseExecutor)
        );
    }

    /**
     * @brief Adds or updates the {@link ExchangeSourceHandler} for a given session ID.
     * @param sessionId The unique identifier for the exchange session.
     * @param sourceHandler The {@link ExchangeSourceHandler} instance to associate with the session ID.
     * Functional Utility: Stores a reference to the source handler, enabling retrieval for subsequent
     *                     async operations, cancellation, or early termination related to that session.
     * Post-condition: The provided `sourceHandler` is mapped to `sessionId` in the internal `exchangeSources` map.
     */
    public void addExchangeSourceHandler(String sessionId, ExchangeSourceHandler sourceHandler) {
        exchangeSources.put(sessionId, sourceHandler);
    }

    /**
     * @brief Removes and returns the {@link ExchangeSourceHandler} associated with the given session ID.
     * @param sessionId The unique identifier for the exchange session.
     * @return The removed {@link ExchangeSourceHandler} instance, or `null` if no handler was found for the session ID.
     * Functional Utility: Disassociates a source handler from a session, typically as part of session cleanup or transfer of responsibility.
     * Post-condition: The {@link ExchangeSourceHandler} for the given `sessionId` is no longer managed by this service.
     */
    public ExchangeSourceHandler removeExchangeSourceHandler(String sessionId) {
        return exchangeSources.remove(sessionId);
    }

    /**
     * @brief Terminates an exchange session prematurely, ensuring results are returned rather than discarded.
     *        This is typically invoked by asynchronous or stop API requests.
     * @param sessionId The unique identifier of the exchange session to terminate.
     * @param listener The {@link ActionListener} to be notified upon the session's early completion.
     * Functional Utility: Allows for controlled termination of ongoing data exchanges, preserving any partial results
     *                     and providing a mechanism for graceful shutdown or retrieval of intermediate states.
     * Pre-condition: This method should be called on the node coordinating the asynchronous request.
     * Post-condition: The {@link ExchangeSourceHandler} for the specified session is removed and its `finishEarly` method is invoked.
     */
    public void finishSessionEarly(String sessionId, ActionListener<Void> listener) {
        /**
         * Block Logic: Retrieves and removes the {@link ExchangeSourceHandler} associated with the `sessionId`.
         * Functional Utility: If a source handler exists, it is instructed to finish early, ensuring any accumulated
         *                     results are processed before the session fully terminates. Otherwise, the listener
         *                     is immediately notified of completion.
         * Invariant: The session either finishes gracefully (if a handler exists) or acknowledges immediate completion.
         */
        ExchangeSourceHandler exchangeSource = removeExchangeSourceHandler(sessionId);
        if (exchangeSource != null) {
            exchangeSource.finishEarly(false, listener);
        } else {
            listener.onResponse(null);
        }
    }

    /**
     * @brief Represents a request to open an exchange session on a remote node.
     * Functional Utility: Carries the necessary information (session ID and exchange buffer size)
     *                     to a remote node to initiate the creation of an {@link ExchangeSinkHandler}.
     *
     * This request is part of the distributed execution model, allowing tasks to dynamically
     * establish data transfer channels between different nodes in the cluster.
     */
    private static class OpenExchangeRequest extends AbstractTransportRequest {
        private final String sessionId;
        private final int exchangeBuffer;

        /**
         * @brief Constructs a new OpenExchangeRequest.
         * @param sessionId The unique identifier for the exchange session.
         * @param exchangeBuffer The desired buffer size for the remote sink handler.
         */
        OpenExchangeRequest(String sessionId, int exchangeBuffer) {
            this.sessionId = sessionId;
            this.exchangeBuffer = exchangeBuffer;
        }

        /**
         * @brief Constructs a new OpenExchangeRequest by reading from a {@link StreamInput}.
         * @param in The {@link StreamInput} to read the request data from.
         * @throws IOException if an I/O error occurs during deserialization.
         * Functional Utility: Enables the deserialization of an OpenExchangeRequest object
         *                     when it is received over the transport layer.
         */
        OpenExchangeRequest(StreamInput in) throws IOException {
            super(in);
            this.sessionId = in.readString();
            this.exchangeBuffer = in.readVInt();
        }

        /**
         * @brief Writes the OpenExchangeRequest's data to a {@link StreamOutput}.
         * @param out The {@link StreamOutput} to write the request data to.
         * @throws IOException if an I/O error occurs during serialization.
         * Functional Utility: Enables the serialization of an OpenExchangeRequest object
         *                     for transmission over the transport layer.
         */
        @Override
        public void writeTo(StreamOutput out) throws IOException {
            super.writeTo(out);
            out.writeString(sessionId);
            out.writeVInt(exchangeBuffer);
        }
    }

    /**
     * @brief Handles incoming {@link OpenExchangeRequest}s.
     * Functional Utility: Processes requests from remote nodes to establish new exchange sessions on the current node.
     *
     * This handler is responsible for creating an {@link ExchangeSinkHandler} based on the request
     * parameters and sending an empty response to acknowledge the setup.
     */
    private class OpenExchangeRequestHandler implements TransportRequestHandler<OpenExchangeRequest> {
        /**
         * @brief Receives and processes an {@link OpenExchangeRequest}.
         * @param request The incoming {@link OpenExchangeRequest} containing session details.
         * @param channel The {@link TransportChannel} to send the response.
         * @param task The {@link Task} associated with this request.
         * @throws Exception if an error occurs during sink handler creation or response sending.
         * Functional Utility: Creates a new {@link ExchangeSinkHandler} on the local node using the provided
         *                     session ID and buffer size from the request, effectively opening a new exchange endpoint.
         * Post-condition: A new {@link ExchangeSinkHandler} is registered, and an empty response is sent back to the requester.
         */
        @Override
        public void messageReceived(OpenExchangeRequest request, TransportChannel channel, Task task) throws Exception {
            createSinkHandler(request.sessionId, request.exchangeBuffer);
            channel.sendResponse(ActionResponse.Empty.INSTANCE);
        }
    }

    /**
     * @brief Handles incoming {@link ExchangeRequest}s, retrieving data pages from an {@link ExchangeSinkHandler}.
     * Functional Utility: Serves as the receiving endpoint for actual data transfer, fetching pages
     *                     from the appropriate sink handler and sending them back to the requester.
     *
     * This handler also integrates with task cancellation mechanisms to gracefully manage ongoing exchanges.
     */
    private class ExchangeTransportAction implements TransportRequestHandler<ExchangeRequest> {
        /**
         * @brief Receives and processes an {@link ExchangeRequest}.
         * @param request The incoming {@link ExchangeRequest} specifying the exchange ID and status.
         * @param channel The {@link TransportChannel} to send the response.
         * @param exchangeTask The {@link Task} associated with this request, allowing for cancellation.
         * Functional Utility: Retrieves the relevant {@link ExchangeSinkHandler}, fetches data pages
         *                     asynchronously, and sends them back to the requesting node. Handles cases
         *                     where the sink handler might not exist or the task is cancelled.
         */
        @Override
        public void messageReceived(ExchangeRequest request, TransportChannel channel, Task exchangeTask) {
            final String exchangeId = request.exchangeId();
            ActionListener<ExchangeResponse> listener = new ChannelActionListener<>(channel);
            final ExchangeSinkHandler sinkHandler = sinks.get(exchangeId);
            /**
             * Block Logic: Checks if an {@link ExchangeSinkHandler} exists for the requested `exchangeId`.
             * Functional Utility: If the sink handler is not found, it indicates an invalid or expired exchange
             *                     session, and an empty response signifying completion is sent.
             * Pre-condition: `sinkHandler` is the result of a lookup using `exchangeId`.
             * Invariant: Ensures that requests for non-existent sink handlers are handled gracefully without errors.
             */
            if (sinkHandler == null) {
                listener.onResponse(new ExchangeResponse(blockFactory, null, true));
            } else {
                /**
                 * Block Logic: Attaches a listener to the `exchangeTask` for cancellation handling and fetches
                 *              data pages from the `sinkHandler`.
                 * Functional Utility: Enables the graceful termination of data fetching if the parent task
                 *                     is cancelled, and retrieves the next available data page from the sink.
                 * Pre-condition: `sinkHandler` is a valid, active {@link ExchangeSinkHandler} instance.
                 * Invariant: Asynchronous page fetching is initiated, and any task cancellation will trigger
                 *            an appropriate failure notification on the sink handler.
                 */
                final CancellableTask task = (CancellableTask) exchangeTask;
                task.addListener(() -> sinkHandler.onFailure(new TaskCancelledException("request cancelled " + task.getReasonCancelled())));
                sinkHandler.fetchPageAsync(request.sourcesFinished(), listener);
            }
        }
    }

    /**
     * @brief A background task responsible for identifying and cleaning up inactive {@link ExchangeSinkHandler}s.
     * Functional Utility: Prevents resource exhaustion by periodically removing sink handlers that are no longer
     *                     actively processing or expecting data, freeing up system resources.
     *
     * This reaper helps maintain the health and efficiency of the ExchangeService by ensuring that only
     * actively used exchange endpoints consume resources.
     */
    private final class InactiveSinksReaper extends AbstractRunnable {
        private final Logger logger;
        private final TimeValue keepAlive;
        private final ThreadPool threadPool;

        /**
         * @brief Constructs a new InactiveSinksReaper.
         * @param logger The logger instance for logging events.
         * @param threadPool The thread pool for access to time utilities.
         * @param keepAlive The duration after which a sink is considered inactive and eligible for reaping.
         */
        InactiveSinksReaper(Logger logger, ThreadPool threadPool, TimeValue keepAlive) {
            this.logger = logger;
            this.keepAlive = keepAlive;
            this.threadPool = threadPool;
        }

        /**
         * @brief Handles unexpected failures during the execution of the reaper task.
         * @param e The exception that caused the failure.
         * Functional Utility: Logs the error and triggers an assertion failure in debug mode, indicating a critical issue.
         */
        @Override
        public void onFailure(Exception e) {
            logger.error("unexpected error when closing inactive sinks", e);
            assert false : e;
        }

        /**
         * @brief Handles rejections of the reaper task when the executor's queue is full.
         * @param e The exception indicating the rejection.
         * Functional Utility: Differentiates between a normal shutdown rejection and other execution rejections,
         *                     logging debug information for the former and calling `onFailure` for the latter.
         */
        @Override
        public void onRejection(Exception e) {
            if (e instanceof EsRejectedExecutionException esre && esre.isExecutorShutdown()) {
                logger.debug("rejected execution when closing inactive sinks");
            } else {
                onFailure(e);
            }
        }

        /**
         * @brief Determines if the reaper task should be force-executed even if the queue is full.
         * @return Always `true`, indicating that this task should bypass queue capacity checks.
         * Functional Utility: Ensures that the critical task of resource cleanup is not prevented by
         *                     a saturated thread pool, prioritizing system stability.
         */
        @Override
        public boolean isForceExecution() {
            // mustn't reject this task even if the queue is full
            return true;
        }

        /**
         * @brief The main logic for identifying and removing inactive sink handlers.
         * Functional Utility: Iterates through all registered sink handlers, checks their activity status,
         *                     and removes those that have been inactive beyond the configured `keepAlive` duration.
         * Pre-condition: This method runs on a non-transport and non-schedule thread to avoid performance bottlenecks.
         * Post-condition: Inactive {@link ExchangeSinkHandler}s are removed from the `sinks` map, and their resources are cleaned up.
         */
        @Override
        protected void doRun() {
            assert Transports.assertNotTransportThread("reaping inactive exchanges can be expensive");
            assert ThreadPool.assertNotScheduleThread("reaping inactive exchanges can be expensive");
            logger.debug("start removing inactive sinks");
            final long nowInMillis = threadPool.relativeTimeInMillis();
            /**
             * Block Logic: Iterates through all active {@link ExchangeSinkHandler}s managed by the service.
             * Functional Utility: Examines each sink to determine its activity status and eligibility for removal.
             * Invariant: Each sink handler currently registered in the `sinks` map is checked against the inactivity criteria.
             */
            for (Map.Entry<String, ExchangeSinkHandler> e : sinks.entrySet()) {
                ExchangeSinkHandler sink = e.getValue();
                /**
                 * Block Logic: Skips active sinks that either still hold data or have active listeners.
                 * Functional Utility: Ensures that sink handlers currently serving data or awaiting responses are not prematurely removed.
                 * Pre-condition: `sink` is a valid {@link ExchangeSinkHandler} instance.
                 * Invariant: Only sinks that are neither holding data nor actively listening are considered for inactivity.
                 */
                if (sink.hasData() && sink.hasListeners()) {
                    continue;
                }
                long elapsedInMillis = nowInMillis - sink.lastUpdatedTimeInMillis();
                /**
                 * Block Logic: Checks if the elapsed time since the last update exceeds the `keepAlive` interval.
                 * Functional Utility: Identifies sink handlers that have been idle for too long and are therefore candidates for removal.
                 * Pre-condition: `elapsedInMillis` is the duration of inactivity for the current `sink`.
                 * Invariant: If the inactivity threshold is met, the sink is logged as removed and its cleanup is initiated.
                 */
                if (elapsedInMillis > keepAlive.millis()) {
                    TimeValue elapsedTime = TimeValue.timeValueMillis(elapsedInMillis);
                    logger.debug("removed sink {} inactive for {}", e.getKey(), elapsedTime);
                    finishSinkHandler(
                        e.getKey(),
                        new ElasticsearchTimeoutException("Exchange sink {} has been inactive for {}", e.getKey(), elapsedTime)
                    );
                }
            }
        }
    }

    /**
     * @brief Creates a new {@link RemoteSink} instance responsible for fetching pages from a remote {@link ExchangeSinkHandler}.
     * @param parentTask The parent {@link Task} that initialized the ESQL request, used for task-related context.
     * @param exchangeId The unique identifier of the exchange session.
     * @param transportService The {@link TransportService} for sending requests to remote nodes.
     * @param conn The {@link Transport.Connection} to the remote node where the corresponding {@link ExchangeSinkHandler} resides.
     * @return A new {@link RemoteSink} instance configured to interact with the remote sink handler.
     * Functional Utility: Provides a client-side representation of a remote data source, allowing local components
     *                     to transparently pull data pages from a sink located on another node.
     */
    public RemoteSink newRemoteSink(Task parentTask, String exchangeId, TransportService transportService, Transport.Connection conn) {
        return new TransportRemoteSink(transportService, blockFactory, conn, parentTask, exchangeId, executor);
    }

    static final class TransportRemoteSink implements RemoteSink {
        final TransportService transportService;
        final BlockFactory blockFactory;
        final Transport.Connection connection;
        final Task parentTask;
        final String exchangeId;
        final Executor responseExecutor;

        final AtomicLong estimatedPageSizeInBytes = new AtomicLong(0L);
        final AtomicReference<SubscribableListener<Void>> completionListenerRef = new AtomicReference<>(null);

        /**
         * @brief Constructs a new TransportRemoteSink.
         * @param transportService The {@link TransportService} for sending requests.
         * @param blockFactory The {@link BlockFactory} for creating and managing data blocks.
         * @param connection The {@link Transport.Connection} to the remote node.
         * @param parentTask The parent {@link Task} associated with this sink.
         * @param exchangeId The unique identifier for the exchange.
         * @param responseExecutor The {@link Executor} for handling responses.
         * Functional Utility: Initializes the remote sink with all necessary components for communicating
         *                     with a remote {@link ExchangeSinkHandler} and managing data transfer.
         */
        TransportRemoteSink(
            TransportService transportService,
            BlockFactory blockFactory,
            Transport.Connection connection,
            Task parentTask,
            String exchangeId,
            Executor responseExecutor
        ) {
            this.transportService = transportService;
            this.blockFactory = blockFactory;
            this.connection = connection;
            this.parentTask = parentTask;
            this.exchangeId = exchangeId;
            this.responseExecutor = responseExecutor;
        }

        /**
         * @brief Asynchronously fetches a data page from the remote sink.
         * @param allSourcesFinished A boolean indicating if all sources contributing to the exchange are finished.
         * @param listener The {@link ActionListener} to be notified with the fetched {@link ExchangeResponse}.
         * Functional Utility: Orchestrates the request for the next data page from the remote sink,
         *                     handling cases where all sources are finished or the operation has already completed.
         */
        @Override
        public void fetchPageAsync(boolean allSourcesFinished, ActionListener<ExchangeResponse> listener) {
            /**
             * Block Logic: Checks if all sources involved in the exchange have already completed.
             * Functional Utility: If all sources are finished, it immediately closes the sink,
             *                     signaling that no further data is expected.
             * Pre-condition: `allSourcesFinished` indicates the global state of contributing sources.
             * Invariant: Prevents unnecessary data fetches if the exchange is logically complete.
             */
            if (allSourcesFinished) {
                close(listener.map(unused -> new ExchangeResponse(blockFactory, null, true)));
                return;
            }
            // already finished
            SubscribableListener<Void> completionListener = completionListenerRef.get();
            /**
             * Block Logic: Checks if the remote sink has already signaled its completion.
             * Functional Utility: If the sink is already completed, it adds the current listener to the existing
             *                     completion listener, ensuring delayed notifications without re-fetching data.
             * Invariant: Ensures that multiple calls to `fetchPageAsync` after completion are handled gracefully.
             */
            if (completionListener != null) {
                completionListener.addListener(listener.map(unused -> new ExchangeResponse(blockFactory, null, true)));
                return;
            }
            // Delegate to the actual asynchronous fetch operation, wrapping the listener
            // to handle potential completion signals and error closures.
            doFetchPageAsync(false, ActionListener.wrap(r -> {
                if (r.finished()) {
                    completionListenerRef.compareAndSet(null, SubscribableListener.nullSuccess());
                }
                listener.onResponse(r);
            }, e -> close(ActionListener.running(() -> listener.onFailure(e)))));
        }

        /**
         * @brief Performs the actual asynchronous request to fetch a data page from the remote {@link ExchangeSinkHandler}.
         * @param allSourcesFinished A flag indicating if all local sources are considered finished.
         * @param listener The {@link ActionListener} to be notified with the {@link ExchangeResponse}.
         * Functional Utility: Sends a request to the remote node for a page of data, managing circuit breaking
         *                     for memory estimation and deserializing the incoming response.
         * Pre-condition: `listener` is a valid {@link ActionListener} to receive the response.
         * Post-condition: A request is sent over the transport layer, and `listener` is invoked upon response or failure.
         */
        private void doFetchPageAsync(boolean allSourcesFinished, ActionListener<ExchangeResponse> listener) {
            final long reservedBytes = allSourcesFinished ? 0 : estimatedPageSizeInBytes.get();
            /**
             * Block Logic: Attempts to reserve estimated memory for the incoming page to prevent circuit breaking.
             * Functional Utility: Proactively checks for available memory before fetching potentially large data pages,
             *                     mitigating OutOfMemoryErrors. If a circuit breaking exception occurs, it fails early.
             * Pre-condition: `reservedBytes` is a positive estimate of the next page's size.
             * Invariant: The listener is wrapped to release the reserved memory after the response is processed,
             *            regardless of success or failure.
             */
            if (reservedBytes > 0) {
                // This doesn't fully protect ESQL from OOM, but reduces the likelihood.
                try {
                    blockFactory.breaker().addEstimateBytesAndMaybeBreak(reservedBytes, "fetch page");
                } catch (Exception e) {
                    assert e instanceof CircuitBreakingException : new AssertionError(e);
                    listener.onFailure(e);
                    return;
                }
                listener = ActionListener.runAfter(listener, () -> blockFactory.breaker().addWithoutBreaking(-reservedBytes));
            }
            transportService.sendChildRequest(
                connection,
                EXCHANGE_ACTION_NAME,
                new ExchangeRequest(exchangeId, allSourcesFinished),
                parentTask,
                TransportRequestOptions.EMPTY,
                new ActionListenerResponseHandler<>(listener, in -> {
                    try (BlockStreamInput bsi = new BlockStreamInput(in, blockFactory)) {
                        final ExchangeResponse resp = new ExchangeResponse(bsi);
                        final long responseBytes = resp.ramBytesUsedByPage();
                        estimatedPageSizeInBytes.getAndUpdate(curr -> Math.max(responseBytes, curr / 2));
                        return resp;
                    }
                }, responseExecutor)
            );
        }

        /**
         * @brief Closes the remote sink, signaling that no more data will be fetched.
         * @param listener The {@link ActionListener} to be notified when the close operation completes.
         * Functional Utility: Ensures that resources on both the local and remote ends are properly
         *                     released when the remote sink is no longer needed.
         * Post-condition: The completion listener is updated, and a final request is sent to the remote
         *                 sink to signal completion and allow for remote resource cleanup.
         */
        @Override
        public void close(ActionListener<Void> listener) {
            final SubscribableListener<Void> candidate = new SubscribableListener<>();
            final SubscribableListener<Void> actual = completionListenerRef.updateAndGet(
                curr -> Objects.requireNonNullElse(curr, candidate)
            );
            actual.addListener(listener);
            if (candidate == actual) {
                doFetchPageAsync(true, ActionListener.wrap(r -> {
                    final Page page = r.takePage();
                    if (page != null) {
                        page.releaseBlocks();
                    }
                    candidate.onResponse(null);
                }, e -> candidate.onResponse(null)));
            }
        }
    }

    // For testing
    /**
     * @brief Checks if there are any active exchange sink handlers managed by this service.
     * @return `true` if no sink handlers are present, `false` otherwise.
     * Functional Utility: Provides a way to ascertain if the service is actively involved in any data exchange sessions.
     */
    public boolean isEmpty() {
        return sinks.isEmpty();
    }

    /**
     * @brief Returns a set of all active exchange IDs (keys) for sink handlers managed by this service.
     * @return A {@link Set} of {@link String}s representing the unique identifiers of active sink handlers.
     * Functional Utility: Allows external components to inspect the currently active exchange sessions.
     */
    public Set<String> sinkKeys() {
        return sinks.keySet();
    }

    /**
     * @brief Lifecycle method invoked when the service is starting.
     * Functional Utility: Placeholder for any initialization logic required at service startup.
     * Invariant: Currently, this method performs no specific actions upon starting.
     */
    @Override
    protected void doStart() {

    }

    /**
     * @brief Lifecycle method invoked when the service is stopping.
     * Functional Utility: Placeholder for any cleanup logic required at service shutdown.
     * Invariant: Currently, this method performs no specific actions upon stopping.
     */
    @Override
    protected void doStop() {

    }

    /**
     * @brief Lifecycle method invoked when the service is closing.
     * Functional Utility: Delegates to `doStop()` for cleanup, ensuring resources are released.
     * Invariant: This method ensures that the stopping logic is executed when the service is closed.
     */
    @Override
    protected void doClose() {
        doStop();
    }

    /**
     * @brief Provides a string representation of the ExchangeService.
     * @return A string showing the class name and the keys of currently active sinks.
     * Functional Utility: Useful for debugging and logging to inspect the state of the ExchangeService.
     */
    @Override
    public String toString() {
        return "ExchangeService{" + "sinks=" + sinks.keySet() + '}';
    }
}
