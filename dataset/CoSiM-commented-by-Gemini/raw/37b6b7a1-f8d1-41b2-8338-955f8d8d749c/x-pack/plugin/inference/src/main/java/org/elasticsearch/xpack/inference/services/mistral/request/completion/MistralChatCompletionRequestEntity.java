/**
 * @raw/37b6b7a1-f8d1-41b2-8338-955f8d8d749c/x-pack/plugin/inference/src/main/java/org/elasticsearch/xpack/inference/services/mistral/request/completion/MistralChatCompletionRequestEntity.java
 * @brief Core functionality implementation.
 * Intent: Execute functional units and state management.
 * Algorithm: Iterative or sequential execution logic.
 * Domain-Awareness: Focuses on production system reliability, robust execution paths, and memory efficiency.
 */
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
 * MistralChatCompletionRequestEntity is responsible for creating the request entity for Mistral chat completion.
 * It implements ToXContentObject to allow serialization to XContent format.
 */
public class MistralChatCompletionRequestEntity implements ToXContentObject {

    private final MistralChatCompletionModel model;
    private final UnifiedChatCompletionRequestEntity unifiedRequestEntity;

    public MistralChatCompletionRequestEntity(UnifiedChatInput unifiedChatInput, MistralChatCompletionModel model) {
        this.unifiedRequestEntity = new UnifiedChatCompletionRequestEntity(unifiedChatInput);
        this.model = Objects.requireNonNull(model);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        unifiedRequestEntity.toXContent(
            builder,
            UnifiedCompletionRequest.withMaxTokensAndSkipStreamOptionsField(model.getServiceSettings().modelId(), params)
        );
        builder.endObject();
        return builder;
    }
}
