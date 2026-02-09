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
import org.elasticsearch.script.TemplateScript;

/**
 * An external bridge for {@link TemplateScript}
 */
public class TemplateScriptBridge {

    /**
     * An external bridge for {@link TemplateScript.Factory}
     */
    public static class Factory extends StableBridgeAPI.ProxyInternal<TemplateScript.Factory> {
    /**
     * @brief [Functional Utility for fromInternal]: Describe purpose here.
     * @param delegate: [Description]
     * @return [ReturnType]: [Description]
     */
        public static Factory fromInternal(final TemplateScript.Factory delegate) {
            return new Factory(delegate);
        }

    /**
     * @brief [Functional Utility for Factory]: Describe purpose here.
     * @param delegate: [Description]
     * @return [ReturnType]: [Description]
     */
        public Factory(final TemplateScript.Factory delegate) {
            super(delegate);
        }

        @Override
    /**
     * @brief [Functional Utility for toInternal]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public TemplateScript.Factory toInternal() {
            return this.internalDelegate;
        }
    }
}
