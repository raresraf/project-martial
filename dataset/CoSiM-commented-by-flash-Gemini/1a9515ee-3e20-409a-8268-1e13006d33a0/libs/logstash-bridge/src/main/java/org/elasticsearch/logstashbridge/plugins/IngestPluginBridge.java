/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.logstashbridge.plugins;

import org.elasticsearch.logstashbridge.StableBridgeAPI;
import org.elasticsearch.logstashbridge.ingest.ProcessorBridge;
import org.elasticsearch.plugins.IngestPlugin;

import java.io.Closeable;
import java.io.IOException;
import java.util.Map;

/**
 * An external bridge for {@link IngestPlugin}
 */
public interface IngestPluginBridge {
    Map<String, ProcessorBridge.Factory> getProcessors(ProcessorBridge.Parameters parameters);

    /**
     * @brief [Functional Utility for fromInternal]: Describe purpose here.
     * @param delegate: [Description]
     * @return [ReturnType]: [Description]
     */
    static ProxyInternal fromInternal(final IngestPlugin delegate) {
        return new ProxyInternal(delegate);
    }

    /**
     * An implementation of {@link IngestPluginBridge} that proxies calls to an internal {@link IngestPlugin}
     */
    class ProxyInternal extends StableBridgeAPI.ProxyInternal<IngestPlugin> implements IngestPluginBridge, Closeable {

    /**
     * @brief [Functional Utility for ProxyInternal]: Describe purpose here.
     * @param delegate: [Description]
     * @return [ReturnType]: [Description]
     */
        private ProxyInternal(final IngestPlugin delegate) {
            super(delegate);
        }

        public Map<String, ProcessorBridge.Factory> getProcessors(final ProcessorBridge.Parameters parameters) {
            return StableBridgeAPI.fromInternal(this.internalDelegate.getProcessors(parameters.toInternal()),
                                                ProcessorBridge.Factory::fromInternal);
        }

        @Override
    /**
     * @brief [Functional Utility for toInternal]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
        public IngestPlugin toInternal() {
            return this.internalDelegate;
        }

        @Override
    /**
     * @brief [Functional Utility for close]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
        public void close() throws IOException {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
            if (this.internalDelegate instanceof Closeable closeableDelegate) {
                closeableDelegate.close();
            }
        }
    }
}
