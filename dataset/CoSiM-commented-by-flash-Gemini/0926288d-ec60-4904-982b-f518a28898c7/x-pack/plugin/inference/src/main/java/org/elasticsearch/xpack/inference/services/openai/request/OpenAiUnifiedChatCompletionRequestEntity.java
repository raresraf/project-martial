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
 * @class OpenAiUnifiedChatCompletionRequestEntity
 * @brief Represents a unified chat completion request entity specifically tailored for OpenAI services.
 * This class serves as an adapter, translating a generic {@link UnifiedChatInput} into a format
 * suitable for consumption by OpenAI's chat completion API, while also incorporating model-specific
 * settings and user identification.
 */
public class OpenAiUnifiedChatCompletionRequestEntity implements ToXContentObject {

    // Defines the field name for the user identifier in the JSON request payload.
    public static final String USER_FIELD = "user";
    // Defines the field name for the model identifier in the JSON request payload.
    private static final String MODEL_FIELD = "model"; // This field is actually handled within UnifiedCompletionRequest.withMaxCompletionTokensTokens
    // Defines the field name for the maximum completion tokens in the JSON request payload.
    private static final String MAX_COMPLETION_TOKENS_FIELD = "max_completion_tokens"; // This field is actually handled within UnifiedCompletionRequest.withMaxCompletionTokensTokens

    // The raw unified chat input provided by the client.
    private final UnifiedChatInput unifiedChatInput;
    // The OpenAI chat completion model configuration to be used for this request.
    private final OpenAiChatCompletionModel model;
    // An internal unified chat completion request entity that handles the core request structure.
    private final UnifiedChatCompletionRequestEntity unifiedRequestEntity;

    /**
     * @brief Constructs a new OpenAiUnifiedChatCompletionRequestEntity.
     * Initializes the request with the chat input and the specific OpenAI model configuration.
     *
     * @param unifiedChatInput The unified chat input containing messages and other generic chat parameters.
     * @param model The OpenAI chat completion model configuration.
     * @throws NullPointerException if unifiedChatInput or model is null.
     */
    public OpenAiUnifiedChatCompletionRequestEntity(UnifiedChatInput unifiedChatInput, OpenAiChatCompletionModel model) {
        // Ensures that the provided chat input is not null.
        this.unifiedChatInput = Objects.requireNonNull(unifiedChatInput);
        // Initializes the internal unified request entity with the chat input.
        this.unifiedRequestEntity = new UnifiedChatCompletionRequestEntity(unifiedChatInput);
        // Ensures that the provided model configuration is not null.
        this.model = Objects.requireNonNull(model);
    }

    /**
     * @brief Converts this object into an {@link XContentBuilder} for serialization, typically to JSON.
     * This method customizes the serialization process to include OpenAI-specific fields and
     * model settings, ensuring the generated request payload conforms to the OpenAI API specification.
     *
     * @param builder The XContentBuilder instance to write the content to.
     * @param params Additional parameters for XContent serialization.
     * @return The XContentBuilder with the object's content written.
     * @throws IOException If an I/O error occurs during the serialization process.
     */
    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        // Starts a new JSON object for the request payload.
        builder.startObject();
        // Delegates to the internal unified request entity to write common chat completion request fields.
        // It also incorporates the model ID and max completion tokens from the model settings.
        unifiedRequestEntity.toXContent(
            builder,
            UnifiedCompletionRequest.withMaxCompletionTokensTokens(model.getServiceSettings().modelId(), params)
        );

        // Block Logic: Conditionally adds the user field to the request payload.
        // Pre-condition: The user ID from the model's task settings is not null or empty.
        if (Strings.isNullOrEmpty(model.getTaskSettings().user()) == false) {
            // Adds the user ID as a field to the JSON object.
            builder.field(USER_FIELD, model.getTaskSettings().user());
        }

        // Ends the JSON object.
        builder.endObject();

        return builder;
    }
}