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
 * @4ab88809-51d3-41e5-a6a6-8ba64c7b2ac7/x-pack/plugin/inference/src/main/java/org/elasticsearch/xpack/inference/services/openai/request/OpenAiUnifiedChatCompletionRequestEntity.java
 * @brief Represents a request entity for OpenAI's unified chat completion, adapting a generic
 * unified request to OpenAI-specific requirements, including model settings and user information.
 *
 * This class serves as a wrapper that takes a generic `UnifiedChatInput` and an
 * `OpenAiChatCompletionModel` to construct a request body suitable for OpenAI's
 * chat completion API, while also conforming to Elasticsearch's `ToXContentObject`
 * for serialization.
 */
public class OpenAiUnifiedChatCompletionRequestEntity implements ToXContentObject {

    /**
     * @brief Field name for the user identifier in the serialized request.
     */
    public static final String USER_FIELD = "user";
    /**
     * @brief The OpenAI chat completion model associated with this request.
     */
    private final OpenAiChatCompletionModel model;
    /**
     * @brief The underlying unified chat completion request entity.
     */
    private final UnifiedChatCompletionRequestEntity unifiedRequestEntity;

    /**
     * @brief Constructs a new `OpenAiUnifiedChatCompletionRequestEntity`.
     * @param unifiedChatInput The unified chat input containing messages and other generic settings.
     * @param model The specific OpenAI chat completion model to be used for this request.
     * @throws NullPointerException if the provided model is null.
     */
    public OpenAiUnifiedChatCompletionRequestEntity(UnifiedChatInput unifiedChatInput, OpenAiChatCompletionModel model) {
        this.unifiedRequestEntity = new UnifiedChatCompletionRequestEntity(unifiedChatInput);
        this.model = Objects.requireNonNull(model);
    }

    /**
     * @brief Converts this object into an XContent (Elasticsearch's content format) representation.
     * This method serializes the unified request entity and optionally adds a user field
     * based on the model's task settings.
     * @param builder The XContentBuilder to write the content to.
     * @param params Additional parameters for XContent serialization.
     * @return The XContentBuilder instance after writing the object's content.
     * @throws IOException If an I/O error occurs during XContent building.
     */
    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        // Block Logic: Starts a new JSON object for the request.
        builder.startObject();
        // Block Logic: Delegates the serialization of the core unified request entity.
        // It injects the model's ID into the serialization parameters for max completion tokens.
        unifiedRequestEntity.toXContent(
            builder,
            UnifiedCompletionRequest.withMaxCompletionTokensTokens(model.getServiceSettings().modelId(), params)
        );

        // Block Logic: Conditionally adds the user field to the request if it is not null or empty
        // in the model's task settings.
        if (Strings.isNullOrEmpty(model.getTaskSettings().user()) == false) {
            builder.field(USER_FIELD, model.getTaskSettings().user());
        }

        // Block Logic: Ends the JSON object.
        builder.endObject();

        return builder;
    }
}
