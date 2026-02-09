/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.openai;

import org.elasticsearch.inference.InferenceServiceResults;
import org.elasticsearch.rest.RestStatus;
import org.elasticsearch.xpack.core.inference.results.StreamingUnifiedChatCompletionResults;
import org.elasticsearch.xpack.core.inference.results.UnifiedChatCompletionException;
import org.elasticsearch.xpack.inference.external.http.HttpResult;
import org.elasticsearch.xpack.inference.external.http.retry.ErrorResponse;
import org.elasticsearch.xpack.inference.external.http.retry.ResponseParser;
import org.elasticsearch.xpack.inference.external.request.Request;
import org.elasticsearch.xpack.inference.external.response.streaming.ServerSentEventParser;
import org.elasticsearch.xpack.inference.external.response.streaming.ServerSentEventProcessor;
import org.elasticsearch.xpack.inference.external.response.streaming.StreamingErrorResponse;

import java.util.Locale;
import java.util.concurrent.Flow;
import java.util.function.Function;

import static org.elasticsearch.core.Strings.format;

/**
 * Handles streaming chat completion responses and error parsing for OpenAI inference endpoints.
 * This handler is designed to work with the unified OpenAI chat completion API.
 */
public class OpenAiUnifiedChatCompletionResponseHandler extends OpenAiChatCompletionResponseHandler {
    /**
     * @brief [Functional Utility for OpenAiUnifiedChatCompletionResponseHandler]: Describe purpose here.
     * @param requestType: [Description]
     * @param parseFunction: [Description]
     * @return [ReturnType]: [Description]
     */
    public OpenAiUnifiedChatCompletionResponseHandler(String requestType, ResponseParser parseFunction) {
        super(requestType, parseFunction, StreamingErrorResponse::fromResponse);
    }

    public OpenAiUnifiedChatCompletionResponseHandler(
        String requestType,
        ResponseParser parseFunction,
        Function<HttpResult, ErrorResponse> errorParseFunction
    ) {
        super(requestType, parseFunction, errorParseFunction);
    }

    @Override
    /**
     * @brief [Functional Utility for parseResult]: Describe purpose here.
     * @param request: [Description]
     * @param flow: [Description]
     * @return [ReturnType]: [Description]
     */
    public InferenceServiceResults parseResult(Request request, Flow.Publisher<HttpResult> flow) {
        var serverSentEventProcessor = new ServerSentEventProcessor(new ServerSentEventParser());
        var openAiProcessor = new OpenAiUnifiedStreamingProcessor((m, e) -> buildMidStreamError(request, m, e));
        flow.subscribe(serverSentEventProcessor);
        serverSentEventProcessor.subscribe(openAiProcessor);
        return new StreamingUnifiedChatCompletionResults(openAiProcessor);
    }

    @Override
    /**
     * @brief [Functional Utility for buildError]: Describe purpose here.
     * @param message: [Description]
     * @param request: [Description]
     * @param result: [Description]
     * @param errorResponse: [Description]
     * @return [ReturnType]: [Description]
     */
    protected Exception buildError(String message, Request request, HttpResult result, ErrorResponse errorResponse) {
        assert request.isStreaming() : "Only streaming requests support this format";
        var responseStatusCode = result.response().getStatusLine().getStatusCode();
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (request.isStreaming()) {
            var errorMessage = errorMessage(message, request, result, errorResponse, responseStatusCode);
            var restStatus = toRestStatus(responseStatusCode);
            return errorResponse instanceof StreamingErrorResponse oer
                ? new UnifiedChatCompletionException(restStatus, errorMessage, oer.type(), oer.code(), oer.param())
                : new UnifiedChatCompletionException(
                    restStatus,
                    errorMessage,
                    createErrorType(errorResponse),
                    restStatus.name().toLowerCase(Locale.ROOT)
                );
        // Block Logic: [Describe purpose of this else/else if block]
        } else {
            return super.buildError(message, request, result, errorResponse);
        }
    }

    /**
     * @brief [Functional Utility for createErrorType]: Describe purpose here.
     * @param errorResponse: [Description]
     * @return [ReturnType]: [Description]
     */
    protected static String createErrorType(ErrorResponse errorResponse) {
        return errorResponse != null ? errorResponse.getClass().getSimpleName() : "unknown";
    }

    /**
     * @brief [Functional Utility for buildMidStreamError]: Describe purpose here.
     * @param request: [Description]
     * @param message: [Description]
     * @param e: [Description]
     * @return [ReturnType]: [Description]
     */
    protected Exception buildMidStreamError(Request request, String message, Exception e) {
        return buildMidStreamError(request.getInferenceEntityId(), message, e);
    }

    /**
     * @brief [Functional Utility for buildMidStreamError]: Describe purpose here.
     * @param inferenceEntityId: [Description]
     * @param message: [Description]
     * @param e: [Description]
     * @return [ReturnType]: [Description]
     */
    public static UnifiedChatCompletionException buildMidStreamError(String inferenceEntityId, String message, Exception e) {
        var errorResponse = StreamingErrorResponse.fromString(message);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (errorResponse instanceof StreamingErrorResponse oer) {
            return new UnifiedChatCompletionException(
                RestStatus.INTERNAL_SERVER_ERROR,
                format(
                    "%s for request from inference entity id [%s]. Error message: [%s]",
                    SERVER_ERROR_OBJECT,
                    inferenceEntityId,
                    errorResponse.getErrorMessage()
                ),
                oer.type(),
                oer.code(),
                oer.param()
            );
        // Block Logic: [Describe purpose of this else/else if block]
        } else if (e != null) {
            return UnifiedChatCompletionException.fromThrowable(e);
        // Block Logic: [Describe purpose of this else/else if block]
        } else {
            return new UnifiedChatCompletionException(
                RestStatus.INTERNAL_SERVER_ERROR,
                format("%s for request from inference entity id [%s]", SERVER_ERROR_OBJECT, inferenceEntityId),
                createErrorType(errorResponse),
                "stream_error"
            );
        }
    }
}
