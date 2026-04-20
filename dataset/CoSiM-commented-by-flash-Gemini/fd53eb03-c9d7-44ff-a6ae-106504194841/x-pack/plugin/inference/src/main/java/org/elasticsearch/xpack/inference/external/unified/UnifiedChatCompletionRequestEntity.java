/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.external.unified;

import org.elasticsearch.inference.UnifiedCompletionRequest;
import org.elasticsearch.xcontent.ToXContentFragment;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.inference.external.http.sender.UnifiedChatInput;

import java.io.IOException;
import java.util.Objects;

import static org.elasticsearch.inference.UnifiedCompletionRequest.SKIP_STREAM_OPTIONS_PARAM;

/**
 * @fd53eb03-c9d7-44ff-a6ae-106504194841/x-pack/plugin/inference/src/main/java/org/elasticsearch/xpack/inference/external/unified/UnifiedChatCompletionRequestEntity.java
 * @brief Serializer for unified chat completion requests within the Elasticsearch inference plugin.
 * 
 * Functional Intent: Encapsulates the logic for converting internal chat completion 
 * request models into an XContent (JSON) format compatible with external providers 
 * (primarily OpenAI-compatible APIs). It handles provider-specific defaults and 
 * streaming options.
 */
public class UnifiedChatCompletionRequestEntity implements ToXContentFragment {

    public static final String STREAM_FIELD = "stream";
    private static final String NUMBER_OF_RETURNED_CHOICES_FIELD = "n";
    private static final String STREAM_OPTIONS_FIELD = "stream_options";
    private static final String INCLUDE_USAGE_FIELD = "include_usage";

    private final UnifiedCompletionRequest unifiedRequest;
    private final boolean stream;

    /**
     * @brief Constructs a request entity from a UnifiedChatInput.
     * @param unifiedChatInput High-level chat input containing the base request and streaming preference.
     */
    public UnifiedChatCompletionRequestEntity(UnifiedChatInput unifiedChatInput) {
        this(Objects.requireNonNull(unifiedChatInput).getRequest(), Objects.requireNonNull(unifiedChatInput).stream());
    }

    /**
     * @brief Constructs a request entity from raw components.
     * @param unifiedRequest The core completion request parameters.
     * @param stream Flag indicating whether the response should be streamed.
     */
    public UnifiedChatCompletionRequestEntity(UnifiedCompletionRequest unifiedRequest, boolean stream) {
        this.unifiedRequest = Objects.requireNonNull(unifiedRequest);
        this.stream = stream;
    }

    /**
     * Block Logic: Serializes the chat completion request to XContent format.
     * Logic: 
     * 1. Delegates base request serialization to the unifiedRequest object.
     * 2. Overrides the choice count to 1 (provider compatibility requirement).
     * 3. Appends the streaming flag and conditional stream options (e.g., usage reports).
     * 
     * @param builder The XContentBuilder to write into.
     * @param params Serialization parameters.
     * @return The updated XContentBuilder.
     * @throws IOException If an error occurs during serialization.
     */
    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        unifiedRequest.toXContent(builder, params);

        // Functional Utility: Enforces a single choice return for OpenAI provider compatibility.
        builder.field(NUMBER_OF_RETURNED_CHOICES_FIELD, 1);

        builder.field(STREAM_FIELD, stream);
        
        // Block Logic: Conditional inclusion of stream metadata.
        // Logic: If streaming is enabled and not explicitly suppressed via the 
        // SKIP_STREAM_OPTIONS_PARAM, requests token usage statistics in the stream metadata.
        if (stream == true && params.paramAsBoolean(SKIP_STREAM_OPTIONS_PARAM, false) == false) {
            builder.startObject(STREAM_OPTIONS_FIELD);
            builder.field(INCLUDE_USAGE_FIELD, true);
            builder.endObject();
        }

        return builder;
    }
}
