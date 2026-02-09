/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
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
 * Represents the request entity for an OpenAI chat completion request.
 *
 * This class wraps a generic unified chat completion request and enhances it
 * with OpenAI-specific details, such as the model and an optional 'user' field,
 * before serialization to JSON (XContent).
 */
public class OpenAiUnifiedChatCompletionRequestEntity implements ToXContentObject {

    public static final String USER_FIELD = "user";
    private final OpenAiChatCompletionModel model;
    private final UnifiedChatCompletionRequestEntity unifiedRequestEntity;

    /**
     * Constructs a new OpenAI chat completion request entity.
     *
     * @param unifiedChatInput The standardized chat input containing messages and other details.
     * @param model The OpenAI-specific model configuration, containing service and task settings.
     */
    public OpenAiUnifiedChatCompletionRequestEntity(UnifiedChatInput unifiedChatInput, OpenAiChatCompletionModel model) {
        this.unifiedRequestEntity = new UnifiedChatCompletionRequestEntity(unifiedChatInput);
        this.model = Objects.requireNonNull(model);
    }

    /**
     * Serializes the request object to an XContentBuilder (JSON).
     *
     * This method first serializes the wrapped unified request and then appends
     * the OpenAI-specific 'user' field if it is configured in the task settings.
     *
     * @param builder The XContentBuilder to write the JSON to.
     * @param params  Parameters for XContent serialization.
     * @return The XContentBuilder with the serialized object.
     * @throws IOException If an error occurs during serialization.
     */
    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        // Block Logic: Serialize the base unified chat request, passing along
        // model-specific parameters like max completion tokens.
        unifiedRequestEntity.toXContent(
            builder,
            UnifiedCompletionRequest.withMaxCompletionTokensTokens(model.getServiceSettings().modelId(), params)
        );

        // Block Logic: If a 'user' identifier is specified in the task settings,
        // add it to the request body. This is an OpenAI-specific field.
        if (Strings.isNullOrEmpty(model.getTaskSettings().user()) == false) {
            builder.field(USER_FIELD, model.getTaskSettings().user());
        }

        builder.endObject();

        return builder;
    }
}