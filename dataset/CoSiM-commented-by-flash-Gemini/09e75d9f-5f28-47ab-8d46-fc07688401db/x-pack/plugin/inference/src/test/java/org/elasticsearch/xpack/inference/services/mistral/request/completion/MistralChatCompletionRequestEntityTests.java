/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.mistral.request.completion;

import org.elasticsearch.common.Strings;
import org.elasticsearch.common.xcontent.XContentHelper;
import org.elasticsearch.inference.UnifiedCompletionRequest;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.xcontent.ToXContent;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.json.JsonXContent;
import org.elasticsearch.xpack.inference.external.http.sender.UnifiedChatInput;
import org.elasticsearch.xpack.inference.services.mistral.completion.MistralChatCompletionModel;

import java.io.IOException;
import java.util.ArrayList;

import static org.elasticsearch.xpack.inference.services.mistral.completion.MistralChatCompletionModelTests.createCompletionModel;

/**
 * @09e75d9f-5f28-47ab-8d46-fc07688401db/x-pack/plugin/inference/src/test/java/org/elasticsearch/xpack/inference/services/mistral/request/completion/MistralChatCompletionRequestEntityTests.java
 * @brief Unit tests for the Mistral API request serialization logic.
 * 
 * Functional Intent: Validates the transformation of unified completion requests 
 * into JSON payloads compatible with Mistral's chat completion endpoint. 
 * Ensures message content, roles, and model-specific parameters (like streaming 
 * and choice count) are correctly mapped.
 */
public class MistralChatCompletionRequestEntityTests extends ESTestCase {

    private static final String ROLE = "user";

    /**
     * Block Logic: Validates serialization of chat messages and model metadata.
     * Logic: 
     * 1. Constructs a high-level message sequence.
     * 2. Orchestrates the creation of a Mistral-specific request entity.
     * 3. Performs JSON serialization and asserts structural equivalence with 
     *    the Mistral API specification (messages, model, streaming flag).
     * 
     * @throws IOException If serialization fails.
     */
    public void testModelUserFieldsSerialization() throws IOException {
        // Block Logic: Construct a unified completion request with a single user message.
        // Invariant: The message object accurately represents a user's input with content and role.
        UnifiedCompletionRequest.Message message = new UnifiedCompletionRequest.Message(
            new UnifiedCompletionRequest.ContentString("Hello, world!"),
            ROLE,
            null,
            null
        );
        var messageList = new ArrayList<UnifiedCompletionRequest.Message>();
        messageList.add(message);

        var unifiedRequest = UnifiedCompletionRequest.of(messageList);

        // Block Logic: Initialize state for the request entity.
        UnifiedChatInput unifiedChatInput = new UnifiedChatInput(unifiedRequest, true);
        MistralChatCompletionModel model = createCompletionModel("api-key", "test-endpoint");

        MistralChatCompletionRequestEntity entity = new MistralChatCompletionRequestEntity(unifiedChatInput, model);

        // Block Logic: Execution of serialization and verification.
        XContentBuilder builder = JsonXContent.contentBuilder();
        entity.toXContent(builder, ToXContent.EMPTY_PARAMS);
        String expectedJson = """
            {
                "messages": [
                    {
                        "content": "Hello, world!",
                        "role": "user"
                    }
                ],
                "model": "test-endpoint",
                "n": 1,
                "stream": true
            }
            """;
        
        // Final assertion to ensure the generated payload meets API requirements.
        assertEquals(XContentHelper.stripWhitespace(expectedJson), Strings.toString(builder));
    }
}
