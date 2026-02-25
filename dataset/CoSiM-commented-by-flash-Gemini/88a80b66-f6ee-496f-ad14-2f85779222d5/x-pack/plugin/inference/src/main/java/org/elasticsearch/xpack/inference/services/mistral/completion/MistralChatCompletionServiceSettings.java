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
 * Represents the settings for the Mistral chat completion service within Elasticsearch X-Pack Inference.
 * This class encapsulates configuration parameters such as the model ID and rate limit
 * settings that govern the behavior of the Mistral chat completion service.
 * It extends {@link FilteredXContentObject} for XContent serialization and
 * implements {@link ServiceSettings} to integrate with the inference service framework.
 */
public class MistralChatCompletionServiceSettings extends FilteredXContentObject implements ServiceSettings {
    /**
     * The unique name for these service settings.
     */
    public static final String NAME = "mistral_completions_service_settings";

    /**
     * The identifier of the Mistral model to be used for chat completions.
     */
    private final String modelId;
    /**
     * Settings for controlling the rate limits of requests to the Mistral service.
     */
    private final RateLimitSettings rateLimitSettings;

    /**
     * Default rate limit settings for the Mistral service.
     * Mistral's default is 5 requests/sec, set to 240 (4 requests/sec) as a sane default for internal use.
     */
    protected static final RateLimitSettings DEFAULT_RATE_LIMIT_SETTINGS = new RateLimitSettings(240);

    /**
     * Creates a {@link MistralChatCompletionServiceSettings} instance from a map of configuration values.
     *
     * This method parses the provided map, extracting the model ID and rate limit settings.
     * It performs validation and throws a {@link ValidationException} if any required fields are missing
     * or if configuration is invalid.
     *
     * @param map The map containing configuration key-value pairs.
     * @param context The parsing context for error reporting and additional configuration details.
     * @return A new {@link MistralChatCompletionServiceSettings} instance.
     * @throws ValidationException If the map contains invalid or missing required configuration.
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
     * Constructs {@link MistralChatCompletionServiceSettings} by reading from a {@link StreamInput}.
     *
     * This constructor is used for deserialization of the settings object.
     *
     * @param in The {@link StreamInput} to read from.
     * @throws IOException If an I/O error occurs during reading.
     */
    public MistralChatCompletionServiceSettings(StreamInput in) throws IOException {
        this.modelId = in.readString();
        this.rateLimitSettings = new RateLimitSettings(in);
    }

    /**
     * Constructs {@link MistralChatCompletionServiceSettings} with specified model ID and rate limit settings.
     *
     * If {@code rateLimitSettings} is null, {@link #DEFAULT_RATE_LIMIT_SETTINGS} will be used.
     *
     * @param modelId The identifier of the Mistral model.
     * @param rateLimitSettings The rate limit settings for the service, or null to use defaults.
     */
    public MistralChatCompletionServiceSettings(String modelId, @Nullable RateLimitSettings rateLimitSettings) {
        this.modelId = modelId;
        this.rateLimitSettings = Objects.requireNonNullElse(rateLimitSettings, DEFAULT_RATE_LIMIT_SETTINGS);
    }

    /**
     * Returns the writeable name of these settings.
     *
     * @return The string name {@link #NAME}.
     */
    @Override
    public String getWriteableName() {
        return NAME;
    }

    /**
     * Returns the minimal supported {@link TransportVersion} for these settings.
     *
     * @return The {@link TransportVersion} indicating the earliest compatible version.
     */
    @Override
    public TransportVersion getMinimalSupportedVersion() {
        return TransportVersions.ML_INFERENCE_MISTRAL_CHAT_COMPLETION_ADDED;
    }

    /**
     * Returns the model ID configured for the Mistral chat completion service.
     *
     * @return The model ID string.
     */
    @Override
    public String modelId() {
        return this.modelId;
    }

    /**
     * Returns the rate limit settings for the Mistral chat completion service.
     *
     * @return The {@link RateLimitSettings} instance.
     */
    public RateLimitSettings rateLimitSettings() {
        return this.rateLimitSettings;
    }

    /**
     * Writes the settings object to a {@link StreamOutput}.
     *
     * This method is used for serialization of the settings object.
     *
     * @param out The {@link StreamOutput} to write to.
     * @throws IOException If an I/O error occurs during writing.
     */
    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(modelId);
        rateLimitSettings.writeTo(out);
    }

    /**
     * Converts the settings object to XContent format, wrapping it in a root object.
     *
     * @param builder The {@link XContentBuilder} to write to.
     * @param params The parameters for XContent serialization.
     * @return The {@link XContentBuilder} with the settings written.
     * @throws IOException If an I/O error occurs during writing.
     */
    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        this.toXContentFragmentOfExposedFields(builder, params);
        builder.endObject();
        return builder;
    }

    /**
     * Converts only the exposed fields of the settings object to XContent format.
     *
     * This method is used by {@link #toXContent} and allows subclasses to control
     * which fields are exposed during serialization.
     *
     * @param builder The {@link XContentBuilder} to write to.
     * @param params The parameters for XContent serialization.
     * @return The {@link XContentBuilder} with the exposed fields written.
     * @throws IOException If an I/O error occurs during writing.
     */
    @Override
    protected XContentBuilder toXContentFragmentOfExposedFields(XContentBuilder builder, Params params) throws IOException {
        builder.field(MODEL_FIELD, this.modelId);

        rateLimitSettings.toXContent(builder, params);

        return builder;
    }

    /**
     * Indicates whether some other object is "equal to" this one.
     *
     * Two {@link MistralChatCompletionServiceSettings} objects are considered equal
     * if their {@code modelId} fields are equal.
     *
     * @param o The reference object with which to compare.
     * @return {@code true} if this object is the same as the obj argument; {@code false} otherwise.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MistralChatCompletionServiceSettings that = (MistralChatCompletionServiceSettings) o;
        return Objects.equals(modelId, that.modelId);
    }

    /**
     * Returns a hash code value for the object.
     *
     * The hash code is based on the {@code modelId} field.
     *
     * @return A hash code value for this object.
     */
    @Override
    public int hashCode() {
        return Objects.hash(modelId);
    }

}
