/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.logstashbridge.ingest;

import org.elasticsearch.common.bytes.BytesArray;
import org.elasticsearch.ingest.PipelineConfiguration;
import org.elasticsearch.logstashbridge.StableBridgeAPI;
import org.elasticsearch.xcontent.XContentType;

import java.util.Map;

/**
 * An external bridge for {@link PipelineConfiguration}
 */
public class PipelineConfigurationBridge extends StableBridgeAPI.ProxyInternal<PipelineConfiguration> {
    /**
     * @brief [Functional Utility for PipelineConfigurationBridge]: Describe purpose here.
     * @param delegate: [Description]
     * @return [ReturnType]: [Description]
     */
    public PipelineConfigurationBridge(final PipelineConfiguration delegate) {
        super(delegate);
    }

    /**
     * @brief [Functional Utility for PipelineConfigurationBridge]: Describe purpose here.
     * @param pipelineId: [Description]
     * @param jsonEncodedConfig: [Description]
     * @return [ReturnType]: [Description]
     */
    public PipelineConfigurationBridge(final String pipelineId, final String jsonEncodedConfig) {
        this(new PipelineConfiguration(pipelineId, new BytesArray(jsonEncodedConfig), XContentType.JSON));
    }

    /**
     * @brief [Functional Utility for getId]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public String getId() {
        return internalDelegate.getId();
    }

    public Map<String, Object> getConfig() {
        return internalDelegate.getConfig();
    }

    public Map<String, Object> getConfig(final boolean unmodifiable) {
        return internalDelegate.getConfig(unmodifiable);
    }

    @Override
    /**
     * @brief [Functional Utility for hashCode]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public int hashCode() {
        return internalDelegate.hashCode();
    }

    @Override
    /**
     * @brief [Functional Utility for toString]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public String toString() {
        return internalDelegate.toString();
    }

    @Override
    /**
     * @brief [Functional Utility for equals]: Describe purpose here.
     * @param obj: [Description]
     * @return [ReturnType]: [Description]
     */
    public boolean equals(final Object obj) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
        if (this == obj) {
    /**
     * @brief [Functional description for field true]: Describe purpose here.
     */
            return true;
        // Block Logic: [Describe purpose of this else/else if block]
        } else if (obj instanceof PipelineConfigurationBridge other) {
            return internalDelegate.equals(other.internalDelegate);
        // Block Logic: [Describe purpose of this else/else if block]
        } else {
    /**
     * @brief [Functional description for field false]: Describe purpose here.
     */
            return false;
        }
    }
}
