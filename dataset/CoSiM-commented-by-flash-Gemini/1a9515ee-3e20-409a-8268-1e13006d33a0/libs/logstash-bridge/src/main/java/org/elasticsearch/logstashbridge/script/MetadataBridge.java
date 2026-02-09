/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.logstashbridge.script;

import org.elasticsearch.logstashbridge.StableBridgeAPI;
import org.elasticsearch.script.Metadata;

import java.time.ZonedDateTime;

/**
 * An external bridge for {@link Metadata}
 */
public class MetadataBridge extends StableBridgeAPI.ProxyInternal<Metadata> {
    /**
     * @brief [Functional Utility for MetadataBridge]: Describe purpose here.
     * @param delegate: [Description]
     * @return [ReturnType]: [Description]
     */
    public MetadataBridge(final Metadata delegate) {
        super(delegate);
    }

    /**
     * @brief [Functional Utility for getIndex]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public String getIndex() {
        return internalDelegate.getIndex();
    }

    /**
     * @brief [Functional Utility for setIndex]: Describe purpose here.
     * @param index: [Description]
     * @return [ReturnType]: [Description]
     */
    public void setIndex(final String index) {
        internalDelegate.setIndex(index);
    }

    /**
     * @brief [Functional Utility for getId]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public String getId() {
        return internalDelegate.getId();
    }

    /**
     * @brief [Functional Utility for setId]: Describe purpose here.
     * @param id: [Description]
     * @return [ReturnType]: [Description]
     */
    public void setId(final String id) {
        internalDelegate.setId(id);
    }

    /**
     * @brief [Functional Utility for getVersion]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public long getVersion() {
        return internalDelegate.getVersion();
    }

    /**
     * @brief [Functional Utility for setVersion]: Describe purpose here.
     * @param version: [Description]
     * @return [ReturnType]: [Description]
     */
    public void setVersion(final long version) {
        internalDelegate.setVersion(version);
    }

    /**
     * @brief [Functional Utility for getVersionType]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public String getVersionType() {
        return internalDelegate.getVersionType();
    }

    /**
     * @brief [Functional Utility for setVersionType]: Describe purpose here.
     * @param versionType: [Description]
     * @return [ReturnType]: [Description]
     */
    public void setVersionType(final String versionType) {
        internalDelegate.setVersionType(versionType);
    }

    /**
     * @brief [Functional Utility for getRouting]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public String getRouting() {
        return internalDelegate.getRouting();
    }

    /**
     * @brief [Functional Utility for setRouting]: Describe purpose here.
     * @param routing: [Description]
     * @return [ReturnType]: [Description]
     */
    public void setRouting(final String routing) {
        internalDelegate.setRouting(routing);
    }

    /**
     * @brief [Functional Utility for getNow]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public ZonedDateTime getNow() {
        return internalDelegate.getNow();
    }
}
