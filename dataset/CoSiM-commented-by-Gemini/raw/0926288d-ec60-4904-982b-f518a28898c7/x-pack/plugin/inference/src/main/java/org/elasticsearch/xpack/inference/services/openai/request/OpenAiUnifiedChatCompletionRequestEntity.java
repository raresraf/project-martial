/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
/**
 * @file This file defines the request entity for OpenAI's unified chat completion API.
 *
 * This file is responsible for creating the JSON request body that is sent to
 * the OpenAI API for chat completion tasks. It combines the chat input with
 * model-specific settings and task configurations to generate a request that
 * can be processed by the OpenAI service.
 */

package org.elasticsearch.xpack.inference.services.openai.request;

import org.elasticsearch.common.Strings;
import org.elasticsearch.inference.UnifiedCompletionRequest;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.inference.external.http.sender.UnifiedChatInput;
import org.elasticsearch.xpack.inference.external.unified.UnifiedChatCompletionRequestEntity;
import org.elasticsearch.xpack.inference.services.openai.completion.OpenAiChatCompletionModel;

import java.io.IOException;
import java.util.Objects;

/**
 * Represents the request entity for an OpenAI unified chat completion.
 * This class is responsible for building the JSON request body that is sent to the OpenAI API.
 */
public class OpenAiUnifiedChatCompletionRequestEntity implements ToXContentObject {

    public static final String USER_FIELD = "user";
    private static final String MODEL_FIELD = "model";
    private static final String MAX_COMPLETION_TOKENS_FIELD = "max_completion_tokens";

    private final UnifiedChatInput unifiedChatInput;
    private final OpenAiChatCompletionModel model;
    private final UnifiedChatCompletionRequestEntity unifiedRequestEntity;

    /**
     * Constructs a new OpenAiUnifiedChatCompletionRequestEntity.
     *
     * @param unifiedChatInput The unified chat input containing the messages.
     * @param model            The OpenAI chat completion model settings.
     */
    public OpenAiUnifiedChatCompletionRequestEntity(UnifiedChatInput unifiedChatInput, OpenAiChatCompletionModel model) {
        this.unifiedChatInput = Objects.requireNonNull(unifiedChatInput);
        this.unifiedRequestEntity = new UnifiedChatCompletionRequestEntity(unifiedChatInput);
        this.model = Objects.requireNonNull(model);
    }

    /**
     * Converts the request entity to XContent, which is used to build the JSON request body.
     *
     * @param builder The XContentBuilder to write the content to.
     * @param params  The parameters for the XContent conversion.
     * @return The XContentBuilder with the request entity content.
     * @throws IOException If an I/O error occurs.
     */
    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        unifiedRequestEntity.toXContent(
            builder,
            UnifiedCompletionRequest.withMaxCompletionTokensTokens(model.getServiceSettings().modelId(), params)
        );

        if (Strings.isNullOrEmpty(model.getTaskSettings().user()) == false) {
            builder.field(USER_FIELD, model.getTaskSettings().user());
        }

        builder.endObject();

        return builder;
    }
}