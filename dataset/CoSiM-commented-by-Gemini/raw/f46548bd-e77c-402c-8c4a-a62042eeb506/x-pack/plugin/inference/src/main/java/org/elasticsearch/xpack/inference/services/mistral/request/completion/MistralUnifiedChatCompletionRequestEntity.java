/**
 * @raw/f46548bd-e77c-402c-8c4a-a62042eeb506/x-pack/plugin/inference/src/main/java/org/elasticsearch/xpack/inference/services/mistral/request/completion/MistralUnifiedChatCompletionRequestEntity.java
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

import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.inference.external.http.sender.UnifiedChatInput;
import org.elasticsearch.xpack.inference.external.unified.UnifiedChatCompletionRequestEntity;

/**
 * This class encapsulates the unified request for Mistral.
 * The main difference between this class and {@link UnifiedChatCompletionRequestEntity} is that Mistral does not support
 * stream options field.
 */
public class MistralUnifiedChatCompletionRequestEntity extends UnifiedChatCompletionRequestEntity {
    public MistralUnifiedChatCompletionRequestEntity(UnifiedChatInput unifiedChatInput) {
        super(unifiedChatInput);
    }

    /**
     * Mistral does not support stream options field, so this method is overridden to do nothing.
     * The stream options field is not applicable for Mistral's unified chat completion request.
     */
    @Override
    protected void fillStreamOptionsFields(XContentBuilder builder) {
        // Mistral does not support stream options field
    }
}
