/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.mistral.completion;

import org.elasticsearch.TransportVersion;
import org.elasticsearch.TransportVersions;
import org.elasticsearch.common.ValidationException;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.inference.ModelConfigurations;
import org.elasticsearch.inference.ServiceSettings;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.inference.services.ConfigurationParseContext;
import org.elasticsearch.xpack.inference.services.mistral.MistralService;
import org.elasticsearch.xpack.inference.services.settings.FilteredXContentObject;
import org.elasticsearch.xpack.inference.services.settings.RateLimitSettings;

import java.io.IOException;
import java.util.Map;
import java.util.Objects;

import static org.elasticsearch.xpack.inference.services.ServiceUtils.extractRequiredString;
import static org.elasticsearch.xpack.inference.services.mistral.MistralConstants.MODEL_FIELD;

/**
 * @88a80b66-f6ee-496f-ad14-2f85779222d5/x-pack/plugin/inference/src/main/java/org/elasticsearch/xpack/inference/services/mistral/completion/MistralChatCompletionServiceSettings.java
 * @brief Configuration container for Mistral AI chat completion service settings.
 * 
 * Functional Intent: Encapsulates immutable settings required for interacting with 
 * Mistral's API, including the specific model identifier and client-side rate 
 * limiting thresholds. It provides logic for deserialization from both maps 
 * (cluster state) and streams (inter-node transport).
 */
public class MistralChatCompletionServiceSettings extends FilteredXContentObject implements ServiceSettings {
    public static final String NAME = "mistral_completions_service_settings";

    private final String modelId;
    private final RateLimitSettings rateLimitSettings;

    // Functional Utility: Defines conservative default rate limits for Mistral (4 req/sec) 
    // to prevent aggressive throttling by the external provider.
    protected static final RateLimitSettings DEFAULT_RATE_LIMIT_SETTINGS = new RateLimitSettings(240);

    /**
     * Block Logic: Deserialization from configuration map.
     * Logic: 
     * 1. Extracts the mandatory model identifier.
     * 2. Resolves rate limit settings, falling back to system defaults if unspecified.
     * 3. Aggregates and throws validation errors if any required fields are missing.
     * 
     * @param map Raw configuration map from the REST request or cluster metadata.
     * @param context Context for parsing (e.g., handling sensitive fields).
     * @return Initialized settings object.
     */
    public static MistralChatCompletionServiceSettings fromMap(Map<String, Object> map, ConfigurationParseContext context) {
        ValidationException validationException = new ValidationException();

        String model = extractRequiredString(map, MODEL_FIELD, ModelConfigurations.SERVICE_SETTINGS, validationException);
        RateLimitSettings rateLimitSettings = RateLimitSettings.of(
            map,
            DEFAULT_RATE_LIMIT_SETTINGS,
            validationException,
            MistralService.NAME,
            context
        );

        if (validationException.validationErrors().isEmpty() == false) {
            throw validationException;
        }

        return new MistralChatCompletionServiceSettings(model, rateLimitSettings);
    }

    /**
     * @brief Deserializes settings from a binary stream.
     */
    public MistralChatCompletionServiceSettings(StreamInput in) throws IOException {
        this.modelId = in.readString();
        this.rateLimitSettings = new RateLimitSettings(in);
    }

    /**
     * @brief Direct constructor with optional rate limits.
     * @param modelId External model name (e.g., 'mistral-large-latest').
     * @param rateLimitSettings Specific rate limits, or null to use defaults.
     */
    public MistralChatCompletionServiceSettings(String modelId, @Nullable RateLimitSettings rateLimitSettings) {
        this.modelId = modelId;
        this.rateLimitSettings = Objects.requireNonNullElse(rateLimitSettings, DEFAULT_RATE_LIMIT_SETTINGS);
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    public TransportVersion getMinimalSupportedVersion() {
        return TransportVersions.ML_INFERENCE_MISTRAL_CHAT_COMPLETION_ADDED;
    }

    @Override
    public String modelId() {
        return this.modelId;
    }

    public RateLimitSettings rateLimitSettings() {
        return this.rateLimitSettings;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(modelId);
        rateLimitSettings.writeTo(out);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        this.toXContentFragmentOfExposedFields(builder, params);
        builder.endObject();
        return builder;
    }

    /**
     * Block Logic: Serializes public fields to XContent format.
     * Logic: Writes the model ID and flattens rate limit settings into the current JSON object level.
     */
    @Override
    protected XContentBuilder toXContentFragmentOfExposedFields(XContentBuilder builder, Params params) throws IOException {
        builder.field(MODEL_FIELD, this.modelId);

        rateLimitSettings.toXContent(builder, params);

        return builder;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MistralChatCompletionServiceSettings that = (MistralChatCompletionServiceSettings) o;
        return Objects.equals(modelId, that.modelId) && Objects.equals(rateLimitSettings, that.rateLimitSettings);
    }

    @Override
    public int hashCode() {
        return Objects.hash(modelId, rateLimitSettings);
    }

}
