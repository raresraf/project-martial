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
 * @brief Unit tests for `MistralChatCompletionRequestEntity`.
 * This class verifies the correct serialization of Mistral chat completion requests,
 * ensuring that the request entity correctly transforms unified completion requests
 * into the Mistral API format, specifically focusing on user-defined fields and model parameters.
 * Domain: Inference, Machine Learning, Testing, API Serialization.
 */
public class MistralChatCompletionRequestEntityTests extends ESTestCase {

    private static final String ROLE = "user";

    /**
     * @brief Tests the serialization of user-defined fields within the Mistral chat completion request.
     * This method ensures that the `MistralChatCompletionRequestEntity` correctly serializes
     * a `UnifiedCompletionRequest` containing a user message into the expected JSON format
     * for the Mistral API, including the message content, role, model, and streaming parameters.
     * @throws IOException If an I/O error occurs during XContent serialization.
     * Pre-condition: UnifiedCompletionRequest with user message is properly formed.
     * Post-condition: Generated JSON matches the expected Mistral API request structure.
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

        // Block Logic: Initialize `UnifiedChatInput` and `MistralChatCompletionModel` for the request entity.
        // Invariant: Model and input objects are correctly configured for the test scenario.
        UnifiedChatInput unifiedChatInput = new UnifiedChatInput(unifiedRequest, true);
        MistralChatCompletionModel model = createCompletionModel("api-key", "test-endpoint");

        // Block Logic: Create the `MistralChatCompletionRequestEntity` from the unified input and model.
        MistralChatCompletionRequestEntity entity = new MistralChatCompletionRequestEntity(unifiedChatInput, model);

        // Block Logic: Serialize the request entity to JSON using XContentBuilder.
        // Invariant: The XContentBuilder correctly captures the entity's state in JSON.
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
        // Block Logic: Assert that the generated JSON matches the expected JSON structure.
        // This verifies the correct serialization of the request for the Mistral API.
        assertEquals(XContentHelper.stripWhitespace(expectedJson), Strings.toString(builder));
    }
}
