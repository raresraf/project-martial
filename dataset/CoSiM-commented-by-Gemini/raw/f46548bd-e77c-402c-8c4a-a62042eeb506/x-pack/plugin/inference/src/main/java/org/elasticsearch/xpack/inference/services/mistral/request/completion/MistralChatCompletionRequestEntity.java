/**
 * @raw/f46548bd-e77c-402c-8c4a-a62042eeb506/x-pack/plugin/inference/src/main/java/org/elasticsearch/xpack/inference/services/mistral/request/completion/MistralChatCompletionRequestEntity.java
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

import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.inference.external.http.sender.UnifiedChatInput;
import org.elasticsearch.xpack.inference.services.mistral.completion.MistralChatCompletionModel;

import java.io.IOException;
import java.util.Objects;

import static org.elasticsearch.xpack.inference.services.mistral.MistralConstants.MAX_TOKENS_FIELD;
import static org.elasticsearch.xpack.inference.services.mistral.MistralConstants.MODEL_FIELD;

/**
 * MistralChatCompletionRequestEntity is responsible for creating the request entity for Mistral chat completion.
 * It implements ToXContentObject to allow serialization to XContent format.
 */
public class MistralChatCompletionRequestEntity implements ToXContentObject {

    private final UnifiedChatInput unifiedChatInput;
    private final MistralChatCompletionModel model;
    private final MistralUnifiedChatCompletionRequestEntity unifiedRequestEntity;

    public MistralChatCompletionRequestEntity(UnifiedChatInput unifiedChatInput, MistralChatCompletionModel model) {
        this.unifiedChatInput = Objects.requireNonNull(unifiedChatInput);
        this.unifiedRequestEntity = new MistralUnifiedChatCompletionRequestEntity(unifiedChatInput);
        this.model = Objects.requireNonNull(model);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        unifiedRequestEntity.toXContent(builder, params);
        builder.field(MODEL_FIELD, model.getServiceSettings().modelId());
        /**
         * Block Logic: Conditional evaluation for divergent control flow.
         * Invariant: Taken branch maintains control flow invariants.
         */
        if (unifiedChatInput.getRequest().maxCompletionTokens() != null) {
            builder.field(MAX_TOKENS_FIELD, unifiedChatInput.getRequest().maxCompletionTokens());
        }
        builder.endObject();
        return builder;
    }
}
