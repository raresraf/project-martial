/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.inference;

import org.elasticsearch.TransportVersion;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.inference.validation.ServiceIntegrationValidator;

import java.io.Closeable;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * @9622b0cb-ede4-44e2-8b0e-831e0a51db4c/server/src/main/java/org/elasticsearch/inference/InferenceService.java
 * @brief SPI (Service Provider Interface) for integrating AI/ML inference providers with Elasticsearch.
 * 
 * Functional Intent: Defines the contract for all inference engines (e.g., OpenAI, HuggingFace, 
 * local models). It handles model lifecycle management (start/stop), configuration 
 * serialization, and the execution of diverse task types like text embeddings, 
 * completion, and re-ranking.
 */
public interface InferenceService extends Closeable {

    /**
     * @brief Post-initialization hook.
     * @param client The internal Elasticsearch client for inter-node or inter-index operations.
     */
    default void init(Client client) {}

    /**
     * @brief Canonical identifier for the service provider (e.g., 'openai').
     */
    String name();

    /**
     * @brief Alternative identifiers for service resolution.
     */
    default List<String> aliases() {
        return List.of();
    }

    /**
     * Block Logic: Configuration ingestion.
     * Logic: Parses a raw request map, identifying and extracting sensitive credentials 
     * (secrets) alongside public service settings.
     * Invariant: Modifies the input map by removing processed keys.
     */
    void parseRequestConfig(String modelId, TaskType taskType, Map<String, Object> config, ActionListener<Model> parsedModelListener);

    /**
     * Block Logic: Persistence-to-model mapping (with secrets).
     * Logic: Reconstructs a model instance from separate settings and secure maps.
     */
    Model parsePersistedConfigWithSecrets(String modelId, TaskType taskType, Map<String, Object> config, Map<String, Object> secrets);

    /**
     * Block Logic: Persistence-to-model mapping (settings only).
     */
    Model parsePersistedConfig(String modelId, TaskType taskType, Map<String, Object> config);

    /**
     * @brief Retrieves the static service capabilities and constraints.
     */
    InferenceServiceConfiguration getConfiguration();

    /**
     * @brief Visibility control for administrative APIs.
     */
    default boolean hideFromConfigurationApi() {
        return false;
    }

    /**
     * @brief Returns the set of task archetypes (e.g., completion, embedding) supported by this provider.
     */
    EnumSet<TaskType> supportedTaskTypes();

    /**
     * infer - Primary execution path for standard inference tasks.
     * 
     * Block Logic: Request dispatch.
     * Logic: Processes raw text input through the specified model, applying 
     * overrides from taskSettings. Handles optional re-ranking parameters (query, topN).
     * Invariant: Asynchronous execution via the provided listener.
     */
    void infer(
        Model model,
        @Nullable String query,
        @Nullable Boolean returnDocuments,
        @Nullable Integer topN,
        List<String> input,
        boolean stream,
        Map<String, Object> taskSettings,
        InputType inputType,
        @Nullable TimeValue timeout,
        ActionListener<InferenceServiceResults> listener
    );

    /**
     * unifiedCompletionInfer - Specialized path for chat/text completion using a normalized schema.
     */
    void unifiedCompletionInfer(
        Model model,
        UnifiedCompletionRequest request,
        TimeValue timeout,
        ActionListener<InferenceServiceResults> listener
    );

    /**
     * chunkedInfer - Advanced inference for oversized inputs.
     * Logic: Handles input segmentation and parallel/sequential processing of document chunks.
     */
    void chunkedInfer(
        Model model,
        @Nullable String query,
        List<ChunkInferenceInput> input,
        Map<String, Object> taskSettings,
        InputType inputType,
        TimeValue timeout,
        ActionListener<List<ChunkedInference>> listener
    );

    /**
     * @brief Prepares hardware or network resources for model execution.
     */
    void start(Model model, TimeValue timeout, ActionListener<Boolean> listener);

    /**
     * @brief Tears down resources associated with a specific model deployment.
     */
    default void stop(Model model, ActionListener<Boolean> listener) {
        listener.onResponse(true);
    }

    /**
     * @brief Metadata enrichment for embedding models.
     */
    default Model updateModelWithEmbeddingDetails(Model model, int embeddingSize) {
        return model;
    }

    /**
     * @brief Metadata enrichment for completion models.
     */
    default Model updateModelWithChatCompletionDetails(Model model) {
        return model;
    }

    /**
     * @brief Minimum required TransportVersion for inter-node compatibility.
     */
    TransportVersion getMinimalSupportedVersion();

    /**
     * @brief Identifies tasks where the provider supports reactive streaming responses.
     */
    default Set<TaskType> supportedStreamingTasks() {
        return Set.of();
    }

    /**
     * @brief Predicate to check if a specific task can utilize the stream interface.
     */
    default boolean canStream(TaskType taskType) {
        return supportedStreamingTasks().contains(taskType);
    }

    record DefaultConfigId(String inferenceId, MinimalServiceSettings settings, InferenceService service) {};

    /**
     * @brief Returns a list of out-of-the-box model configurations provided by this service.
     */
    default List<DefaultConfigId> defaultConfigIds() {
        return List.of();
    }

    /**
     * @brief Hydrates and returns full model configurations for default IDs.
     */
    default void defaultConfigs(ActionListener<List<Model>> defaultsListener) {
        defaultsListener.onResponse(List.of());
    }

    /**
     * @brief Dynamic field expansion hook.
     */
    default void updateModelsWithDynamicFields(List<Model> model, ActionListener<List<Model>> listener) {
        listener.onResponse(model);
    }

    /**
     * @brief Lifecycle hook triggered when the host node reaches a healthy state.
     */
    default void onNodeStarted() {}

    /**
     * @brief Retrieves specialized validation logic for a given task type.
     */
    default ServiceIntegrationValidator getServiceIntegrationValidator(TaskType taskType) {
        return null;
    }
}
