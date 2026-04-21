/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.search;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.core.config.Configurator;
import org.elasticsearch.client.Request;
import org.elasticsearch.client.Response;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.util.CollectionUtils;
import org.elasticsearch.common.xcontent.XContentHelper;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.search.ErrorTraceHelper;
import org.elasticsearch.search.SearchService;
import org.elasticsearch.test.ESIntegTestCase;
import org.elasticsearch.test.MockLog;
import org.elasticsearch.test.transport.MockTransportService;
import org.elasticsearch.xcontent.XContentType;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;

import java.io.IOException;
import java.util.Collection;
import java.util.Map;
import java.util.function.BooleanSupplier;

/**
 * @file AsyncSearchErrorTraceIT.java
 * @brief Integration tests for error trace propagation in asynchronous search operations.
 * 
 * Functional Intent: Validates the behavior of the `error_trace` parameter across 
 * different stages of an async search (submission vs retrieval). It ensures that 
 * stack traces are only sent over the wire when explicitly requested, while 
 * maintaining persistent visibility in data node logs regardless of the request flag.
 * 
 * Domain: Production Systems, Distributed Search, Error Handling.
 */
public class AsyncSearchErrorTraceIT extends ESIntegTestCase {
    /**
     * Observable for detecting if a transport message payload contains a stack trace.
     */
    private BooleanSupplier transportMessageHasStackTrace;

    @Override
    protected boolean addMockHttpTransport() {
        return false; // Logic: Explicitly enables HTTP for REST client interaction.
    }

    @Override
    @SuppressWarnings("unchecked")
    protected Collection<Class<? extends Plugin>> nodePlugins() {
        // Domain Awareness: Plugs in AsyncSearch and MockTransport to allow fine-grained message inspection.
        return CollectionUtils.appendToCopyNoNullElements(super.nodePlugins(), AsyncSearch.class, MockTransportService.TestPlugin.class);
    }

    /**
     * @brief Elevates SearchService logs to DEBUG to allow verification of logged stack traces.
     */
    @BeforeClass
    public static void setDebugLogLevel() {
        Configurator.setLevel(SearchService.class, Level.DEBUG);
    }

    /**
     * @brief Initializes the transport listener and disables batched query execution for trace isolation.
     */
    @Before
    public void setupMessageListener() {
        transportMessageHasStackTrace = ErrorTraceHelper.setupErrorTraceListener(internalCluster());
        // Optimization: Disables batching to ensure a 1:1 mapping between shards and error logs.
        updateClusterSettings(Settings.builder().put(SearchService.BATCHED_QUERY_PHASE.getKey(), false));
    }

    @After
    public void resetSettings() {
        updateClusterSettings(Settings.builder().putNull(SearchService.BATCHED_QUERY_PHASE.getKey()));
    }

    /**
     * @brief Bootstraps indices with conflicting data types to trigger query-time failures.
     */
    private void setupIndexWithDocs() {
        createIndex("test1", "test2");
        // Logic: Creates a type mismatch (string vs numeric) to trigger a dynamic mapping error or search failure.
        indexRandom(
            true,
            prepareIndex("test1").setId("1").setSource("field", "foo"),
            prepareIndex("test2").setId("10").setSource("field", 5)
        );
        refresh();
    }

    /**
     * testAsyncSearchFailingQueryErrorTraceDefault - Verifies default suppression of stack traces.
     * 
     * Invariant: By default, stack traces must NOT be sent over the transport layer.
     */
    public void testAsyncSearchFailingQueryErrorTraceDefault() throws IOException, InterruptedException {
        setupIndexWithDocs();

        Request searchRequest = new Request("POST", "/_async_search");
        searchRequest.setJsonEntity("""
            {
                "query": {
                    "simple_query_string" : {
                        "query": "foo",
                        "fields": ["field"]
                    }
                }
            }
            """);
        searchRequest.addParameter("keep_on_completion", "true");
        searchRequest.addParameter("wait_for_completion_timeout", "0ms");
        Map<String, Object> responseEntity = performRequestAndGetResponseEntityAfterDelay(searchRequest, TimeValue.ZERO);
        String asyncExecutionId = (String) responseEntity.get("id");
        Request request = new Request("GET", "/_async_search/" + asyncExecutionId);
        
        /**
         * Block Logic: Polling for async completion.
         * Pre-condition: Search task is executing in the background.
         */
        while (responseEntity.get("is_running") instanceof Boolean isRunning && isRunning) {
            responseEntity = performRequestAndGetResponseEntityAfterDelay(request, TimeValue.timeValueSeconds(1L));
        }
        
        // Validation: check that the stack trace was not sent from the data node to the coordinating node
        assertFalse(transportMessageHasStackTrace.getAsBoolean());
    }

    /**
     * testAsyncSearchFailingQueryErrorTraceTrue - Verifies explicit enablement of trace propagation.
     */
    public void testAsyncSearchFailingQueryErrorTraceTrue() throws IOException, InterruptedException {
        setupIndexWithDocs();

        Request searchRequest = new Request("POST", "/_async_search");
        searchRequest.setJsonEntity("""
            {
                "query": {
                    "simple_query_string" : {
                        "query": "foo",
                        "fields": ["field"]
                    }
                }
            }
            """);
        searchRequest.addParameter("error_trace", "true");
        searchRequest.addParameter("keep_on_completion", "true");
        searchRequest.addParameter("wait_for_completion_timeout", "0ms");
        Map<String, Object> responseEntity = performRequestAndGetResponseEntityAfterDelay(searchRequest, TimeValue.ZERO);
        String asyncExecutionId = (String) responseEntity.get("id");
        Request request = new Request("GET", "/_async_search/" + asyncExecutionId);
        request.addParameter("error_trace", "true");
        
        while (responseEntity.get("is_running") instanceof Boolean isRunning && isRunning) {
            responseEntity = performRequestAndGetResponseEntityAfterDelay(request, TimeValue.timeValueSeconds(1L));
        }
        
        // Validation: check that the stack trace was sent from the data node to the coordinating node
        assertTrue(transportMessageHasStackTrace.getAsBoolean());
    }

    /**
     * testAsyncSearchFailingQueryErrorTraceFalse - Verifies explicit suppression of trace propagation.
     */
    public void testAsyncSearchFailingQueryErrorTraceFalse() throws IOException, InterruptedException {
        setupIndexWithDocs();

        Request searchRequest = new Request("POST", "/_async_search");
        searchRequest.setJsonEntity("""
            {
                "query": {
                    "simple_query_string" : {
                        "query": "foo",
                        "fields": ["field"]
                    }
                }
            }
            """);
        searchRequest.addParameter("error_trace", "false");
        searchRequest.addParameter("keep_on_completion", "true");
        searchRequest.addParameter("wait_for_completion_timeout", "0ms");
        Map<String, Object> responseEntity = performRequestAndGetResponseEntityAfterDelay(searchRequest, TimeValue.ZERO);
        String asyncExecutionId = (String) responseEntity.get("id");
        Request request = new Request("GET", "/_async_search/" + asyncExecutionId);
        request.addParameter("error_trace", "false");
        
        while (responseEntity.get("is_running") instanceof Boolean isRunning && isRunning) {
            responseEntity = performRequestAndGetResponseEntityAfterDelay(request, TimeValue.timeValueSeconds(1L));
        }
        
        // Validation: check that the stack trace was not sent from the data node to the coordinating node
        assertFalse(transportMessageHasStackTrace.getAsBoolean());
    }

    /**
     * testDataNodeLogsStackTrace - Ensures internal logging remains robust regardless of API parameters.
     * 
     * Invariant: Even if the user suppresses the API response trace, the system must 
     * log the failure context locally on the data node for diagnostic purposes.
     */
    public void testDataNodeLogsStackTrace() throws IOException, InterruptedException {
        setupIndexWithDocs();

        Request searchRequest = new Request("POST", "/_async_search");
        searchRequest.setJsonEntity("""
            {
                "query": {
                    "simple_query_string" : {
                        "query": "foo",
                        "fields": ["field"]
                    }
                }
            }
            """);

        // Logic: Randomizes the error_trace parameter to verify global logging persistence.
        int errorTraceValue = randomIntBetween(0, 2);
        if (errorTraceValue == 0) {
            searchRequest.addParameter("error_trace", "true");
        } else if (errorTraceValue == 1) {
            searchRequest.addParameter("error_trace", "false");
        } // else empty

        searchRequest.addParameter("keep_on_completion", "true");
        searchRequest.addParameter("wait_for_completion_timeout", "0ms");

        String errorTriggeringIndex = "test2";
        int numShards = getNumShards(errorTriggeringIndex).numPrimaries;
        
        /**
         * Block Logic: Log capture orchestration.
         * Logic: Intercepts SearchService log events and validates that an exception 
         * trace was emitted for every failing shard.
         */
        try (var mockLog = MockLog.capture(SearchService.class)) {
            ErrorTraceHelper.addSeenLoggingExpectations(numShards, mockLog, errorTriggeringIndex);

            Map<String, Object> responseEntity = performRequestAndGetResponseEntityAfterDelay(searchRequest, TimeValue.ZERO);
            String asyncExecutionId = (String) responseEntity.get("id");
            Request request = new Request("GET", "/_async_search/" + asyncExecutionId);

            if (errorTraceValue == 0) {
                request.addParameter("error_trace", "true");
            } else if (errorTraceValue == 1) {
                request.addParameter("error_trace", "false");
            } // else empty

            while (responseEntity.get("is_running") instanceof Boolean isRunning && isRunning) {
                responseEntity = performRequestAndGetResponseEntityAfterDelay(request, TimeValue.timeValueSeconds(1L));
            }

            // Invariant: All expected shard-level error logs must have been triggered.
            mockLog.assertAllExpectationsMatched();
        }
    }

    /**
     * testAsyncSearchFailingQueryErrorTraceFalseOnSubmitAndTrueOnGet - Validates precedence of submission flags.
     * 
     * Logic: If a task is submitted without traces, subsequent 'get' requests with 
     * traces enabled should still respect the initial task security/trace state.
     */
    public void testAsyncSearchFailingQueryErrorTraceFalseOnSubmitAndTrueOnGet() throws IOException, InterruptedException {
        setupIndexWithDocs();

        Request searchRequest = new Request("POST", "/_async_search");
        searchRequest.setJsonEntity("""
            {
                "query": {
                    "simple_query_string" : {
                        "query": "foo",
                        "fields": ["field"]
                    }
                }
            }
            """);
        searchRequest.addParameter("error_trace", "false");
        searchRequest.addParameter("keep_on_completion", "true");
        searchRequest.addParameter("wait_for_completion_timeout", "0ms");
        Map<String, Object> responseEntity = performRequestAndGetResponseEntityAfterDelay(searchRequest, TimeValue.ZERO);
        String asyncExecutionId = (String) responseEntity.get("id");
        Request request = new Request("GET", "/_async_search/" + asyncExecutionId);
        request.addParameter("error_trace", "true");
        
        while (responseEntity.get("is_running") instanceof Boolean isRunning && isRunning) {
            responseEntity = performRequestAndGetResponseEntityAfterDelay(request, TimeValue.timeValueSeconds(1L));
        }
        
        // check that the stack trace was not sent from the data node to the coordinating node
        assertFalse(transportMessageHasStackTrace.getAsBoolean());
    }

    /**
     * testAsyncSearchFailingQueryErrorTraceTrueOnSubmitAndFalseOnGet - Validates persistence of trace context.
     */
    public void testAsyncSearchFailingQueryErrorTraceTrueOnSubmitAndFalseOnGet() throws IOException, InterruptedException {
        setupIndexWithDocs();

        Request searchRequest = new Request("POST", "/_async_search");
        searchRequest.setJsonEntity("""
            {
                "query": {
                    "simple_query_string" : {
                        "query": "foo",
                        "fields": ["field"]
                    }
                }
            }
            """);
        searchRequest.addParameter("error_trace", "true");
        searchRequest.addParameter("keep_on_completion", "true");
        searchRequest.addParameter("wait_for_completion_timeout", "0ms");
        Map<String, Object> responseEntity = performRequestAndGetResponseEntityAfterDelay(searchRequest, TimeValue.ZERO);
        String asyncExecutionId = (String) responseEntity.get("id");
        Request request = new Request("GET", "/_async_search/" + asyncExecutionId);
        request.addParameter("error_trace", "false");
        
        while (responseEntity.get("is_running") instanceof Boolean isRunning && isRunning) {
            responseEntity = performRequestAndGetResponseEntityAfterDelay(request, TimeValue.timeValueSeconds(1L));
        }
        
        // Logic: Since the original task submission authorized traces, they may be sent across nodes.
        assertTrue(transportMessageHasStackTrace.getAsBoolean());
    }

    /**
     * @brief Helper for REST execution with deserialization into a Map.
     */
    private Map<String, Object> performRequestAndGetResponseEntityAfterDelay(Request r, TimeValue sleep) throws IOException,
        InterruptedException {
        Thread.sleep(sleep.millis());
        Response response = getRestClient().performRequest(r);
        XContentType entityContentType = XContentType.fromMediaType(response.getEntity().getContentType().getValue());
        return XContentHelper.convertToMap(entityContentType.xContent(), response.getEntity().getContent(), false);
    }
}
