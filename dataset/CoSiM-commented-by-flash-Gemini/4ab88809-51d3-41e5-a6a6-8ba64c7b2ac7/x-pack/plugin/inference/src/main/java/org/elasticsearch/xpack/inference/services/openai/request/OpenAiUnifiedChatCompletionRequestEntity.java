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
 * @brief OpenAI-specific serializer for unified chat completion requests.
 * 
 * Functional Intent: Adapts generic chat completion inputs to conform to the 
 * OpenAI API specification. It manages model-specific settings, user identifiers, 
 * and token limits during the serialization process.
 */
public class OpenAiUnifiedChatCompletionRequestEntity implements ToXContentObject {

    public static final String USER_FIELD = "user";
    private final OpenAiChatCompletionModel model;
    private final UnifiedChatCompletionRequestEntity unifiedRequestEntity;

    /**
     * @brief Constructs an OpenAI request entity.
     * @param unifiedChatInput The source chat data.
     * @param model The target model configuration containing service and task settings.
     */
    public OpenAiUnifiedChatCompletionRequestEntity(UnifiedChatInput unifiedChatInput, OpenAiChatCompletionModel model) {
        this.unifiedRequestEntity = new UnifiedChatCompletionRequestEntity(unifiedChatInput);
        this.model = Objects.requireNonNull(model);
    }

    /**
     * Block Logic: Serializes the request into an OpenAI-compatible JSON object.
     * Logic: 
     * 1. Initiates the JSON object structure.
     * 2. Delegates core field serialization while injecting model-specific limits (max tokens).
     * 3. Conditionally appends the 'user' metadata if defined in task settings.
     * 
     * @param builder The XContentBuilder to populate.
     * @param params Base serialization parameters.
     * @return The populated builder.
     * @throws IOException If serialization fails.
     */
    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        
        // Functional Utility: Injects model-specific constraints into the unified serialization path.
        unifiedRequestEntity.toXContent(
            builder,
            UnifiedCompletionRequest.withMaxCompletionTokensTokens(model.getServiceSettings().modelId(), params)
        );

        // Block Logic: Inclusion of user attribution for safety and auditing.
        // Invariant: Only writes the field if a non-empty user identifier is present.
        if (Strings.isNullOrEmpty(model.getTaskSettings().user()) == false) {
            builder.field(USER_FIELD, model.getTaskSettings().user());
        }

        builder.endObject();

        return builder;
    }
}
