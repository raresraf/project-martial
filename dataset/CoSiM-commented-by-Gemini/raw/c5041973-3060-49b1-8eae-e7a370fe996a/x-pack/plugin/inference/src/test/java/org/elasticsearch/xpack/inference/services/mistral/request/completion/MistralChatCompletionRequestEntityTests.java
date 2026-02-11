/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

/**
 * @file MistralChatCompletionRequestEntityTests.java
 * @brief This file contains the unit tests for the {@link MistralChatCompletionRequestEntity} class.
 *
 * The tests in this file are designed to verify the correct serialization of the
 * {@link MistralChatCompletionRequestEntity} to JSON format, ensuring that the
 * request payload is compliant with the Mistral API specifications for chat
 * completion requests.
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
 * @class MistralChatCompletionRequestEntityTests
 * @brief Test suite for {@link MistralChatCompletionRequestEntity}.
 *
 * This class contains unit tests to validate the JSON serialization of the
 * {@link MistralChatCompletionRequestEntity}. It ensures that the generated JSON
 * structure and values correctly represent the chat completion request sent to
 * the Mistral API.
 */
public class MistralChatCompletionRequestEntityTests extends ESTestCase {

    private static final String ROLE = "user";

    /**
     * @brief Tests the serialization of a chat completion request with user-defined fields.
     *
     * This test method verifies that a {@link MistralChatCompletionRequestEntity}
     * object, when constructed with a simple user message, serializes to the
     * expected JSON format. The test ensures that the "messages" array, "model"
     * name, and other request parameters are correctly formatted for the Mistral API.
     *
     * @throws IOException If an I/O error occurs during JSON serialization.
     */
    public void testModelUserFieldsSerialization() throws IOException {
        UnifiedCompletionRequest.Message message = new UnifiedCompletionRequest.Message(
            new UnifiedCompletionRequest.ContentString("Hello, world!"),
            ROLE,
            null,
            null
        );
        var messageList = new ArrayList<UnifiedCompletionRequest.Message>();
        messageList.add(message);

        var unifiedRequest = UnifiedCompletionRequest.of(messageList);

        UnifiedChatInput unifiedChatInput = new UnifiedChatInput(unifiedRequest, true);
        MistralChatCompletionModel model = createCompletionModel("api-key", "test-endpoint");

        MistralChatCompletionRequestEntity entity = new MistralChatCompletionRequestEntity(unifiedChatInput, model);

        XContentBuilder builder = JsonXContent.contentBuilder();
        entity.toXContent(builder, ToXContent.EMPTY_PARAMS);

        String jsonString = Strings.toString(builder);
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
        assertJsonEquals(jsonString, expectedJson);
    }
}
