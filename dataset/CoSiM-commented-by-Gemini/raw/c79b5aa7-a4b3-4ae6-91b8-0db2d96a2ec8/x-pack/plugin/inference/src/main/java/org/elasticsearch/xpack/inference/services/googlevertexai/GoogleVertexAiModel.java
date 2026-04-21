/**
 * @raw/c79b5aa7-a4b3-4ae6-91b8-0db2d96a2ec8/x-pack/plugin/inference/src/main/java/org/elasticsearch/xpack/inference/services/googlevertexai/GoogleVertexAiModel.java
 * @brief Core functionality implementation.
 * Intent: Execute functional units and state management.
 * Algorithm: Iterative or sequential execution logic.
 * Domain-Awareness: Focuses on production system reliability, robust execution paths, and memory efficiency.
 */
/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.services.googlevertexai;

import org.elasticsearch.inference.Model;
import org.elasticsearch.inference.ModelConfigurations;
import org.elasticsearch.inference.ModelSecrets;
import org.elasticsearch.inference.ServiceSettings;
import org.elasticsearch.inference.TaskSettings;
import org.elasticsearch.xpack.inference.external.action.ExecutableAction;
import org.elasticsearch.xpack.inference.services.googlevertexai.action.GoogleVertexAiActionVisitor;

import java.net.URI;
import java.util.Map;
import java.util.Objects;

public abstract class GoogleVertexAiModel extends Model {

    private final GoogleVertexAiRateLimitServiceSettings rateLimitServiceSettings;

    protected URI uri;

    public GoogleVertexAiModel(
        ModelConfigurations configurations,
        ModelSecrets secrets,
        GoogleVertexAiRateLimitServiceSettings rateLimitServiceSettings
    ) {
        super(configurations, secrets);

        this.rateLimitServiceSettings = Objects.requireNonNull(rateLimitServiceSettings);
    }

    public GoogleVertexAiModel(GoogleVertexAiModel model, ServiceSettings serviceSettings) {
        super(model, serviceSettings);

        uri = model.uri();
        rateLimitServiceSettings = model.rateLimitServiceSettings();
    }

    public GoogleVertexAiModel(GoogleVertexAiModel model, TaskSettings taskSettings) {
        super(model, taskSettings);

        uri = model.uri();
        rateLimitServiceSettings = model.rateLimitServiceSettings();
    }

    public abstract ExecutableAction accept(GoogleVertexAiActionVisitor creator, Map<String, Object> taskSettings);

    public GoogleVertexAiRateLimitServiceSettings rateLimitServiceSettings() {
        return rateLimitServiceSettings;
    }

    public URI uri() {
        return uri;
    }
}
