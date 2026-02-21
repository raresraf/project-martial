/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.mistral.request.completion;

import org.elasticsearch.common.Strings;
import org.elasticsearch.inference.UnifiedCompletionRequest;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.xcontent.ToXContent;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.json.JsonXContent;
import org.elasticsearch.xpack.inference.external.http.sender.UnifiedChatInput;
import org.elasticsearch.xpack.inference.services.mistral.completion.MistralChatCompletionModel;

import java.io.IOException;
import java.util.ArrayList;

import static org.elasticsearch.xpack.inference.Utils.assertJsonEquals;
import static org.elasticsearch.xpack.inference.services.mistral.completion.MistralChatCompletionModelTests.createCompletionModel;

/**
 * @file MistralChatCompletionRequestEntityTests.java
 * @brief Unit tests for the MistralChatCompletionRequestEntity class.
 *
 * This file contains tests to ensure that the `MistralChatCompletionRequestEntity`
 * correctly serializes unified chat completion requests into the format expected
 * by the Mistral API. It focuses on verifying the JSON output for various input scenarios.
 */
public class MistralChatCompletionRequestEntityTests extends ESTestCase {

    private static final String ROLE = "user";

    /**
     * @brief Tests the serialization of user fields within the Mistral chat completion request entity.
     *
     * Functional Utility: This test verifies that the `MistralChatCompletionRequestEntity`
     * correctly transforms a `UnifiedCompletionRequest` containing chat messages
     * into a JSON structure that adheres to the Mistral chat completion API specification.
     * It checks for correct mapping of message content, role, model name, and streaming preference.
     *
     * @throws IOException if there is an error during JSON serialization.
     */
    public void testModelUserFieldsSerialization() throws IOException {
        // Block Logic: Prepare a single chat message for the unified request.
        UnifiedCompletionRequest.Message message = new UnifiedCompletionRequest.Message(
            new UnifiedCompletionRequest.ContentString("Hello, world!"),
            ROLE,
            null, // No name specified for this message
            null  // No tool calls specified for this message
        );
        // Block Logic: Create a list containing the prepared message.
        var messageList = new ArrayList<UnifiedCompletionRequest.Message>();
        messageList.add(message);

        // Block Logic: Construct a unified completion request from the message list.
        var unifiedRequest = UnifiedCompletionRequest.of(messageList);

        // Block Logic: Create a UnifiedChatInput with the unified request and set streaming to true.
        UnifiedChatInput unifiedChatInput = new UnifiedChatInput(unifiedRequest, true);
        // Block Logic: Create a mock MistralChatCompletionModel for testing purposes.
        MistralChatCompletionModel model = createCompletionModel("api-key", "test-endpoint");

        // Block Logic: Instantiate the MistralChatCompletionRequestEntity with the prepared input and model.
        MistralChatCompletionRequestEntity entity = new MistralChatCompletionRequestEntity(unifiedChatInput, model);

        // Block Logic: Use XContentBuilder to serialize the entity into a JSON string.
        XContentBuilder builder = JsonXContent.contentBuilder();
        entity.toXContent(builder, ToXContent.EMPTY_PARAMS);

        String jsonString = Strings.toString(builder);
        // Block Logic: Define the expected JSON output string for comparison.
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
        // Block Logic: Assert that the generated JSON string matches the expected JSON string.
        // Functional Utility: `assertJsonEquals` compares two JSON strings for structural and value equality.
        assertJsonEquals(jsonString, expectedJson);
    }
}