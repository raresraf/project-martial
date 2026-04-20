/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.mistral.request.completion;

import org.elasticsearch.inference.UnifiedCompletionRequest;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.inference.external.http.sender.UnifiedChatInput;
import org.elasticsearch.xpack.inference.external.unified.UnifiedChatCompletionRequestEntity;
import org.elasticsearch.xpack.inference.services.mistral.completion.MistralChatCompletionModel;

import java.io.IOException;
import java.util.Objects;

/**
 * @37b6b7a1-f8d1-41b2-8338-955f8d8d749c/x-pack/plugin/inference/src/main/java/org/elasticsearch/xpack/inference/services/mistral/request/completion/MistralChatCompletionRequestEntity.java
 * @brief Mistral-specific serializer for unified chat completion requests.
 * 
 * Functional Intent: Adapts generic chat completion inputs to conform to the 
 * Mistral AI API specification. It leverages the unified request infrastructure 
 * while applying Mistral-specific token limits and streaming configurations 
 * during serialization.
 */
public class MistralChatCompletionRequestEntity implements ToXContentObject {

    private final MistralChatCompletionModel model;
    private final UnifiedChatCompletionRequestEntity unifiedRequestEntity;

    /**
     * @brief Constructs a Mistral request entity.
     * @param unifiedChatInput The source chat data.
     * @param model The Mistral model configuration containing service settings.
     */
    public MistralChatCompletionRequestEntity(UnifiedChatInput unifiedChatInput, MistralChatCompletionModel model) {
        this.unifiedRequestEntity = new UnifiedChatCompletionRequestEntity(unifiedChatInput);
        this.model = Objects.requireNonNull(model);
    }

    /**
     * Block Logic: Serializes the request into a Mistral-compatible JSON object.
     * Logic: 
     * 1. Initiates the JSON object structure.
     * 2. Delegates serialization to the unified request entity.
     * 3. Configures parameters to include model ID, enforce token limits, and 
     *    conditionally skip stream options fields based on Mistral's requirements.
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
            UnifiedCompletionRequest.withMaxTokensAndSkipStreamOptionsField(model.getServiceSettings().modelId(), params)
        );
        
        builder.endObject();
        return builder;
    }
}
