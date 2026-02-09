/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services;

import org.elasticsearch.ElasticsearchStatusException;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.cluster.service.ClusterService;
import org.elasticsearch.common.ValidationException;
import org.elasticsearch.core.IOUtils;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.core.Strings;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.inference.ChunkInferenceInput;
import org.elasticsearch.inference.ChunkedInference;
import org.elasticsearch.inference.InferenceService;
import org.elasticsearch.inference.InferenceServiceResults;
import org.elasticsearch.inference.InputType;
import org.elasticsearch.inference.Model;
import org.elasticsearch.inference.TaskType;
import org.elasticsearch.inference.UnifiedCompletionRequest;
import org.elasticsearch.rest.RestStatus;
import org.elasticsearch.xpack.inference.InferencePlugin;
import org.elasticsearch.xpack.inference.external.http.sender.ChatCompletionInput;
import org.elasticsearch.xpack.inference.external.http.sender.EmbeddingsInput;
import org.elasticsearch.xpack.inference.external.http.sender.HttpRequestSender;
import org.elasticsearch.xpack.inference.external.http.sender.InferenceInputs;
import org.elasticsearch.xpack.inference.external.http.sender.QueryAndDocsInputs;
import org.elasticsearch.xpack.inference.external.http.sender.Sender;
import org.elasticsearch.xpack.inference.external.http.sender.UnifiedChatInput;

import java.io.IOException;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * @brief Functional description of the SenderService class.
 *        This is a placeholder for detailed semantic documentation.
 *        Further analysis will elaborate on its algorithm, complexity, and invariants.
 */
public abstract class SenderService implements InferenceService {
    protected static final Set<TaskType> COMPLETION_ONLY = EnumSet.of(TaskType.COMPLETION);
    /**
     * @brief [Functional description for field sender]: Describe purpose here.
     */
    private final Sender sender;
    /**
     * @brief [Functional description for field serviceComponents]: Describe purpose here.
     */
    private final ServiceComponents serviceComponents;
    /**
     * @brief [Functional description for field clusterService]: Describe purpose here.
     */
    private final ClusterService clusterService;

    /**
     * @brief [Functional Utility for SenderService]: Describe purpose here.
     * @param factory: [Description]
     * @param serviceComponents: [Description]
     * @param clusterService: [Description]
     * @return [ReturnType]: [Description]
     */
    public SenderService(HttpRequestSender.Factory factory, ServiceComponents serviceComponents, ClusterService clusterService) {
        Objects.requireNonNull(factory);
        sender = factory.createSender();
        this.serviceComponents = Objects.requireNonNull(serviceComponents);
        this.clusterService = Objects.requireNonNull(clusterService);
    }

    /**
     * @brief [Functional Utility for getSender]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public Sender getSender() {
    /**
     * @brief [Functional description for field sender]: Describe purpose here.
     */
        return sender;
    }

    /**
     * @brief [Functional Utility for getServiceComponents]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    protected ServiceComponents getServiceComponents() {
    /**
     * @brief [Functional description for field serviceComponents]: Describe purpose here.
     */
        return serviceComponents;
    }

    @Override
    public void infer(
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
    ) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (timeout == null) {
            timeout = clusterService.getClusterSettings().get(InferencePlugin.INFERENCE_QUERY_TIMEOUT);
        }
        init();
        var chunkInferenceInput = input.stream().map(i -> new ChunkInferenceInput(i, null)).toList();
        var inferenceInput = createInput(this, model, chunkInferenceInput, inputType, query, returnDocuments, topN, stream);
        doInfer(model, inferenceInput, taskSettings, timeout, listener);
    }

    private static InferenceInputs createInput(
        SenderService service,
        Model model,
        List<ChunkInferenceInput> input,
        InputType inputType,
        @Nullable String query,
        @Nullable Boolean returnDocuments,
        @Nullable Integer topN,
        boolean stream
    ) {
        List<String> textInput = ChunkInferenceInput.inputs(input);
        return switch (model.getTaskType()) {
            case COMPLETION, CHAT_COMPLETION -> new ChatCompletionInput(textInput, stream);
            case RERANK -> {
                ValidationException validationException = new ValidationException();
                service.validateRerankParameters(returnDocuments, topN, validationException);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (validationException.validationErrors().isEmpty() == false) {
    /**
     * @brief [Functional description for field validationException]: Describe purpose here.
     */
                    throw validationException;
                }
                yield new QueryAndDocsInputs(query, textInput, returnDocuments, topN, stream);
            }
            case TEXT_EMBEDDING, SPARSE_EMBEDDING -> {
                ValidationException validationException = new ValidationException();
                service.validateInputType(inputType, model, validationException);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                if (validationException.validationErrors().isEmpty() == false) {
    /**
     * @brief [Functional description for field validationException]: Describe purpose here.
     */
                    throw validationException;
                }
                yield new EmbeddingsInput(input, inputType, stream);
            }
            default -> throw new ElasticsearchStatusException(
                Strings.format("Invalid task type received when determining input type: [%s]", model.getTaskType().toString()),
                RestStatus.BAD_REQUEST
            );
        };
    }

    @Override
    public void unifiedCompletionInfer(
        Model model,
        UnifiedCompletionRequest request,
        TimeValue timeout,
        ActionListener<InferenceServiceResults> listener
    ) {
        init();
        doUnifiedCompletionInfer(model, new UnifiedChatInput(request, true), timeout, listener);
    }

    @Override
    public void chunkedInfer(
        Model model,
        @Nullable String query,
        List<ChunkInferenceInput> input,
        Map<String, Object> taskSettings,
        InputType inputType,
        TimeValue timeout,
        ActionListener<List<ChunkedInference>> listener
    ) {
        init();

        ValidationException validationException = new ValidationException();
        validateInputType(inputType, model, validationException);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (validationException.validationErrors().isEmpty() == false) {
    /**
     * @brief [Functional description for field validationException]: Describe purpose here.
     */
            throw validationException;
        }

        // a non-null query is not supported and is dropped by all providers
        doChunkedInfer(model, new EmbeddingsInput(input, inputType), taskSettings, inputType, timeout, listener);
    }

    protected abstract void doInfer(
        Model model,
        InferenceInputs inputs,
        Map<String, Object> taskSettings,
        TimeValue timeout,
        ActionListener<InferenceServiceResults> listener
    );

    protected abstract void validateInputType(InputType inputType, Model model, ValidationException validationException);

    protected void validateRerankParameters(Boolean returnDocuments, Integer topN, ValidationException validationException) {}

    protected abstract void doUnifiedCompletionInfer(
        Model model,
        UnifiedChatInput inputs,
        TimeValue timeout,
        ActionListener<InferenceServiceResults> listener
    );

    protected abstract void doChunkedInfer(
        Model model,
        EmbeddingsInput inputs,
        Map<String, Object> taskSettings,
        InputType inputType,
        TimeValue timeout,
        ActionListener<List<ChunkedInference>> listener
    );

    /**
     * @brief [Functional Utility for start]: Describe purpose here.
     * @param model: [Description]
     * @param listener: [Description]
     * @return [ReturnType]: [Description]
     */
    public void start(Model model, ActionListener<Boolean> listener) {
        init();
        doStart(model, listener);
    }

    @Override
    /**
     * @brief [Functional Utility for start]: Describe purpose here.
     * @param model: [Description]
     * @param unused: [Description]
     * @param listener: [Description]
     * @return [ReturnType]: [Description]
     */
    public void start(Model model, @Nullable TimeValue unused, ActionListener<Boolean> listener) {
        start(model, listener);
    }

    /**
     * @brief [Functional Utility for doStart]: Describe purpose here.
     * @param model: [Description]
     * @param listener: [Description]
     * @return [ReturnType]: [Description]
     */
    protected void doStart(Model model, ActionListener<Boolean> listener) {
        listener.onResponse(true);
    }

    /**
     * @brief [Functional Utility for init]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    private void init() {
        sender.start();
    }

    @Override
    /**
     * @brief [Functional Utility for close]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void close() throws IOException {
        IOUtils.closeWhileHandlingException(sender);
    }
}
