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
 * Represents the settings for the Mistral chat completion service.
 * This class acts as a data transfer object (DTO) for configuration that is both
 * serializable for network transport ({@link org.elasticsearch.common.io.stream.Writeable})
 * and renderable as XContent for APIs. It encapsulates the model ID and rate limit
 * settings required to interact with the Mistral chat completion service.
 */
public class MistralChatCompletionServiceSettings extends FilteredXContentObject implements ServiceSettings {
    /**
     * Unique identifier for stream serialization.
     */
    public static final String NAME = "mistral_completions_service_settings";

    private final String modelId;
    private final RateLimitSettings rateLimitSettings;

    /**
     * Default rate limit settings for the Mistral service. Based on public documentation,
     * Mistral's default is 5 requests/sec. This is set to a conservative 240 requests/min
     * (4 requests/sec) as a safe default for the Elasticsearch integration.
     */
    protected static final RateLimitSettings DEFAULT_RATE_LIMIT_SETTINGS = new RateLimitSettings(240);

    /**
     * Factory method to parse settings from a map structure, typically derived from JSON or YAML configuration.
     *
     * @param map The map containing the raw configuration values.
     * @param context The parsing context, used for features like rate limiting.
     * @return A new {@link MistralChatCompletionServiceSettings} instance.
     * @throws ValidationException if required fields are missing or invalid.
     */
    public static MistralChatCompletionServiceSettings fromMap(Map<String, Object> map, ConfigurationParseContext context) {
        ValidationException validationException = new ValidationException();

        // Pre-condition: Ensure the 'model' field is present and is a string.
        String model = extractRequiredString(map, MODEL_FIELD, ModelConfigurations.SERVICE_SETTINGS, validationException);
        // Invariant: If rate limit settings are not provided, the default is used.
        RateLimitSettings rateLimitSettings = RateLimitSettings.of(
            map,
            DEFAULT_RATE_LIMIT_SETTINGS,
            validationException,
            MistralService.NAME,
            context
        );

        // Post-condition: If any validation errors were collected, throw an exception.
        if (validationException.validationErrors().isEmpty() == false) {
            throw validationException;
        }

        return new MistralChatCompletionServiceSettings(model, rateLimitSettings);
    }

    /**
     * Deserialization constructor. Reads the object state from a stream.
     * @param in The stream input.
     * @throws IOException If an I/O error occurs.
     */
    public MistralChatCompletionServiceSettings(StreamInput in) throws IOException {
        this.modelId = in.readString();
        this.rateLimitSettings = new RateLimitSettings(in);
    }

    /**
     * Instantiates a new Mistral chat completion service settings object.
     * @param modelId The identifier of the Mistral model to use.
     * @param rateLimitSettings The rate limit settings. If null, a default is applied.
     */
    public MistralChatCompletionServiceSettings(String modelId, @Nullable RateLimitSettings rateLimitSettings) {
        this.modelId = modelId;
        this.rateLimitSettings = Objects.requireNonNullElse(rateLimitSettings, DEFAULT_RATE_LIMIT_SETTINGS);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String getWriteableName() {
        return NAME;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public TransportVersion getMinimalSupportedVersion() {
        return TransportVersions.ML_INFERENCE_MISTRAL_CHAT_COMPLETION_ADDED;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String modelId() {
        return this.modelId;
    }

    /**
     * @return The configured rate limit settings for the service.
     */
    public RateLimitSettings rateLimitSettings() {
        return this.rateLimitSettings;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(modelId);
        rateLimitSettings.writeTo(out);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        this.toXContentFragmentOfExposedFields(builder, params);
        builder.endObject();
        return builder;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected XContentBuilder toXContentFragmentOfExposedFields(XContentBuilder builder, Params params) throws IOException {
        builder.field(MODEL_FIELD, this.modelId);
        rateLimitSettings.toXContent(builder, params);
        return builder;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MistralChatCompletionServiceSettings that = (MistralChatCompletionServiceSettings) o;
        return Objects.equals(modelId, that.modelId) && Objects.equals(rateLimitSettings, that.rateLimitSettings);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int hashCode() {
        return Objects.hash(modelId, rateLimitSettings);
    }

}
