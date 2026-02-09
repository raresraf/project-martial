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
 * Tests the JSON serialization of {@link MistralChatCompletionRequestEntity}.
 * This class ensures that the request entity for Mistral chat completions
 * is correctly formatted as a JSON payload, adhering to the expected structure
 * for the external Mistral service.
 */
public class MistralChatCompletionRequestEntityTests extends ESTestCase {

    private static final String ROLE = "user";

    /**
     * Verifies that a {@link UnifiedChatInput} object containing a simple user message
     * is correctly serialized into the JSON format expected by the Mistral API.
     * This test checks for the presence and correctness of key fields such as 'messages',
     * 'model', 'n', and 'stream' to ensure API compatibility.
     *
     * @throws IOException If an error occurs during JSON serialization.
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
        assertEquals(XContentHelper.stripWhitespace(expectedJson), Strings.toString(builder));
    }
}
