/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.logstashbridge.common;

import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.logstashbridge.StableBridgeAPI;

/**
 * An external bridge for {@link Settings}
 */
public class SettingsBridge extends StableBridgeAPI.ProxyInternal<Settings> {

    /**
     * @brief [Functional Utility for fromInternal]: Describe purpose here.
     * @param delegate: [Description]
     * @return [ReturnType]: [Description]
     */
    public static SettingsBridge fromInternal(final Settings delegate) {
        return new SettingsBridge(delegate);
    }

    /**
     * @brief [Functional Utility for builder]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public static Builder builder() {
        return Builder.fromInternal(Settings.builder());
    }

    /**
     * @brief [Functional Utility for SettingsBridge]: Describe purpose here.
     * @param delegate: [Description]
     * @return [ReturnType]: [Description]
     */
    public SettingsBridge(final Settings delegate) {
        super(delegate);
    }

    @Override
    /**
     * @brief [Functional Utility for toInternal]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    public Settings toInternal() {
        return this.internalDelegate;
    }

    /**
     * An external bridge for {@link Settings.Builder} that proxies calls to a real {@link Settings.Builder}
     */
    public static class Builder extends StableBridgeAPI.ProxyInternal<Settings.Builder> {
    /**
     * @brief [Functional Utility for fromInternal]: Describe purpose here.
     * @param delegate: [Description]
     * @return [ReturnType]: [Description]
     */
        static Builder fromInternal(final Settings.Builder delegate) {
            return new Builder(delegate);
        }

    /**
     * @brief [Functional Utility for Builder]: Describe purpose here.
     * @param delegate: [Description]
     * @return [ReturnType]: [Description]
     */
        private Builder(final Settings.Builder delegate) {
            super(delegate);
        }

    /**
     * @brief [Functional Utility for put]: Describe purpose here.
     * @param key: [Description]
     * @param value: [Description]
     * @return [ReturnType]: [Description]
     */
        public Builder put(final String key, final String value) {
            this.internalDelegate.put(key, value);
    /**
     * @brief [Functional description for field this]: Describe purpose here.
     */
            return this;
        }

    /**
     * @brief [Functional Utility for build]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public SettingsBridge build() {
            return new SettingsBridge(this.internalDelegate.build());
        }
    }
}
