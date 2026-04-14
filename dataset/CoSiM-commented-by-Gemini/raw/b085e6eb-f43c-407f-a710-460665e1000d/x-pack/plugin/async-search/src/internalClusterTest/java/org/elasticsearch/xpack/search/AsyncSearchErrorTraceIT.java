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
 * Integration tests for the `error_trace` functionality in async search.
 *
 * This test suite verifies that stack traces from failed shard searches are only
 * sent from data nodes to the coordinating node if the `error_trace` parameter
 * is set to true on the initial async search submission. It also confirms that
 * data nodes always log the stack trace locally upon failure, regardless of the
 * `error_trace` setting.
 */
public class AsyncSearchErrorTraceIT extends ESIntegTestCase {
    private BooleanSupplier transportMessageHasStackTrace;

    @Override
    protected boolean addMockHttpTransport() {
        return false; // enable http
    }

    @Override
    @SuppressWarnings("unchecked")
    protected Collection<Class<? extends Plugin>> nodePlugins() {
        return CollectionUtils.appendToCopyNoNullElements(super.nodePlugins(), AsyncSearch.class, MockTransportService.TestPlugin.class);
    }

    @BeforeClass
    public static void setDebugLogLevel() {
        Configurator.setLevel(SearchService.class, Level.DEBUG);
    }

    /**
     * Sets up a transport listener before each test to detect if any internode
     * transport messages contain a stack trace. Also disables batched query
     * execution as the listener does not support it.
     */
    @Before
    public void setupMessageListener() {
        transportMessageHasStackTrace = ErrorTraceHelper.setupErrorTraceListener(internalCluster());
        // TODO: make this test work with batched query execution by enhancing ErrorTraceHelper.setupErrorTraceListener
        updateClusterSettings(Settings.builder().put(SearchService.BATCHED_QUERY_PHASE.getKey(), false));
    }

    @After
    public void resetSettings() {
        updateClusterSettings(Settings.builder().putNull(SearchService.BATCHED_QUERY_PHASE.getKey()));
    }

    /**
     * Sets up two indices: 'test1' with a text field and 'test2' with a numeric
     * field. This setup is designed to cause a parsing failure on 'test2' when a
     * simple_query_string query expecting a text field is executed.
     */
    private void setupIndexWithDocs() {
        createIndex("test1", "test2");
        indexRandom(
            true,
            prepareIndex("test1").setId("1").setSource("field", "foo"),
            prepareIndex("test2").setId("10").setSource("field", 5)
        );
        refresh();
    }

    /**
     * Tests that a failing async search does NOT include a stack trace in the
     * transport message by default (when `error_trace` is not specified).
     */
    public void testAsyncSearchFailingQueryErrorTraceDefault() throws IOException, InterruptedException {
        setupIndexWithDocs();

        // Arrange: Create a search request that will fail on one shard.
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

        // Act: Submit the async search and poll for its completion.
        Map<String, Object> responseEntity = performRequestAndGetResponseEntityAfterDelay(searchRequest, TimeValue.ZERO);
        String asyncExecutionId = (String) responseEntity.get("id");
        Request request = new Request("GET", "/_async_search/" + asyncExecutionId);
        while (responseEntity.get("is_running") instanceof Boolean isRunning && isRunning) {
            responseEntity = performRequestAndGetResponseEntityAfterDelay(request, TimeValue.timeValueSeconds(1L));
        }

        // Assert: Verify that no stack trace was sent from the data node.
        assertFalse(transportMessageHasStackTrace.getAsBoolean());
    }

    /**
     * Tests that a failing async search DOES include a stack trace in the
     * transport message when `error_trace=true` is specified.
     */
    public void testAsyncSearchFailingQueryErrorTraceTrue() throws IOException, InterruptedException {
        setupIndexWithDocs();

        // Arrange: Create a search request with error_trace=true.
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

        // Act: Submit the async search and poll for its completion.
        Map<String, Object> responseEntity = performRequestAndGetResponseEntityAfterDelay(searchRequest, TimeValue.ZERO);
        String asyncExecutionId = (String) responseEntity.get("id");
        Request request = new Request("GET", "/_async_search/" + asyncExecutionId);
        request.addParameter("error_trace", "true");
        while (responseEntity.get("is_running") instanceof Boolean isRunning && isRunning) {
            responseEntity = performRequestAndGetResponseEntityAfterDelay(request, TimeValue.timeValueSeconds(1L));
        }

        // Assert: Verify that a stack trace was sent from the data node.
        assertTrue(transportMessageHasStackTrace.getAsBoolean());
    }

    /**
     * Tests that a failing async search does NOT include a stack trace in the
     * transport message when `error_trace=false` is specified.
     */
    public void testAsyncSearchFailingQueryErrorTraceFalse() throws IOException, InterruptedException {
        setupIndexWithDocs();

        // Arrange: Create a search request with error_trace=false.
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

        // Act: Submit the async search and poll for its completion.
        Map<String, Object> responseEntity = performRequestAndGetResponseEntityAfterDelay(searchRequest, TimeValue.ZERO);
        String asyncExecutionId = (String) responseEntity.get("id");
        Request request = new Request("GET", "/_async_search/" + asyncExecutionId);
        request.addParameter("error_trace", "false");
        while (responseEntity.get("is_running") instanceof Boolean isRunning && isRunning) {
            responseEntity = performRequestAndGetResponseEntityAfterDelay(request, TimeValue.timeValueSeconds(1L));
        }

        // Assert: Verify that no stack trace was sent from the data node.
        assertFalse(transportMessageHasStackTrace.getAsBoolean());
    }

    /**
     * Tests that the data node where the shard failure occurs always logs the
     * stack trace, regardless of the `error_trace` parameter's value.
     */
    public void testDataNodeLogsStackTrace() throws IOException, InterruptedException {
        setupIndexWithDocs();

        // Arrange: Create a search request. The error_trace parameter will be randomized.
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

        // No matter the value of error_trace (empty, true, or false) we should see stack traces logged
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

        // Act & Assert: Capture logs from SearchService and assert that the expected error messages are logged.
        try (var mockLog = MockLog.capture(SearchService.class)) {
            ErrorTraceHelper.addSeenLoggingExpectations(numShards, mockLog, errorTriggeringIndex);

            Map<String, Object> responseEntity = performRequestAndGetResponseEntityAfterDelay(searchRequest, TimeValue.ZERO);
            String asyncExecutionId = (String) responseEntity.get("id");
            Request request = new Request("GET", "/_async_search/" + asyncExecutionId);

            // Use the same value of error_trace as the search request
            if (errorTraceValue == 0) {
                request.addParameter("error_trace", "true");
            } else if (errorTraceValue == 1) {
                request.addParameter("error_trace", "false");
            } // else empty

            while (responseEntity.get("is_running") instanceof Boolean isRunning && isRunning) {
                responseEntity = performRequestAndGetResponseEntityAfterDelay(request, TimeValue.timeValueSeconds(1L));
            }

            mockLog.assertAllExpectationsMatched();
        }
    }

    /**
     * Tests that if `error_trace=false` on submission, the stack trace is not
     * captured, even if `error_trace=true` is used when retrieving the result.
     */
    public void testAsyncSearchFailingQueryErrorTraceFalseOnSubmitAndTrueOnGet() throws IOException, InterruptedException {
        setupIndexWithDocs();

        // Arrange: Submit with error_trace=false.
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

        // Act: Poll for completion, but request the error trace on the GET request.
        Map<String, Object> responseEntity = performRequestAndGetResponseEntityAfterDelay(searchRequest, TimeValue.ZERO);
        String asyncExecutionId = (String) responseEntity.get("id");
        Request request = new Request("GET", "/_async_search/" + asyncExecutionId);
        request.addParameter("error_trace", "true");
        while (responseEntity.get("is_running") instanceof Boolean isRunning && isRunning) {
            responseEntity = performRequestAndGetResponseEntityAfterDelay(request, TimeValue.timeValueSeconds(1L));
        }

        // Assert: The decision was made at submission time, so no trace should have been sent.
        assertFalse(transportMessageHasStackTrace.getAsBoolean());
    }

    /**
     * Tests that if `error_trace=true` on submission, the stack trace is
     * captured and sent, even if `error_trace=false` is used on retrieval.
     */
    public void testAsyncSearchFailingQueryErrorTraceTrueOnSubmitAndFalseOnGet() throws IOException, InterruptedException {
        setupIndexWithDocs();

        // Arrange: Submit with error_trace=true.
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

        // Act: Poll for completion with error_trace=false.
        Map<String, Object> responseEntity = performRequestAndGetResponseEntityAfterDelay(searchRequest, TimeValue.ZERO);
        String asyncExecutionId = (String) responseEntity.get("id");
        Request request = new Request("GET", "/_async_search/" + asyncExecutionId);
        request.addParameter("error_trace", "false");
        while (responseEntity.get("is_running") instanceof Boolean isRunning && isRunning) {
            responseEntity = performRequestAndGetResponseEntityAfterDelay(request, TimeValue.timeValueSeconds(1L));
        }

        // Assert: The trace was sent during the execution phase triggered by the initial POST.
        assertTrue(transportMessageHasStackTrace.getAsBoolean());
    }

    private Map<String, Object> performRequestAndGetResponseEntityAfterDelay(Request r, TimeValue sleep) throws IOException,
        InterruptedException {
        Thread.sleep(sleep.millis());
        Response response = getRestClient().performRequest(r);
        XContentType entityContentType = XContentType.fromMediaType(response.getEntity().getContentType().getValue());
        return XContentHelper.convertToMap(entityContentType.xContent(), response.getEntity().getContent(), false);
    }
}
