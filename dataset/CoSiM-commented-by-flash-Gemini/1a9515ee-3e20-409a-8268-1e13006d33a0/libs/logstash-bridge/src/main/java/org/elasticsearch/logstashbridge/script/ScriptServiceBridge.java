/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.logstashbridge.script;

import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.ingest.common.ProcessorsWhitelistExtension;
import org.elasticsearch.logstashbridge.StableBridgeAPI;
import org.elasticsearch.logstashbridge.common.SettingsBridge;
import org.elasticsearch.painless.PainlessPlugin;
import org.elasticsearch.painless.PainlessScriptEngine;
import org.elasticsearch.painless.spi.PainlessExtension;
import org.elasticsearch.painless.spi.Whitelist;
import org.elasticsearch.plugins.ExtensiblePlugin;
import org.elasticsearch.script.IngestConditionalScript;
import org.elasticsearch.script.IngestScript;
import org.elasticsearch.script.ScriptContext;
import org.elasticsearch.script.ScriptEngine;
import org.elasticsearch.script.ScriptModule;
import org.elasticsearch.script.ScriptService;
import org.elasticsearch.script.mustache.MustacheScriptEngine;
import org.elasticsearch.xpack.constantkeyword.ConstantKeywordPainlessExtension;
import org.elasticsearch.xpack.spatial.SpatialPainlessExtension;
import org.elasticsearch.xpack.wildcard.WildcardPainlessExtension;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.LongSupplier;

/**
 * An external bridge for {@link ScriptService}
 */
public class ScriptServiceBridge extends StableBridgeAPI.ProxyInternal<ScriptService> implements Closeable {
    /**
     * @brief [Functional Utility for fromInternal]: Describe purpose here.
     * @param delegate: [Description]
     * @return [ReturnType]: [Description]
     */
    public ScriptServiceBridge fromInternal(final ScriptService delegate) {
        return new ScriptServiceBridge(delegate);
    }

    /**
     * @brief [Functional Utility for ScriptServiceBridge]: Describe purpose here.
     * @param settingsBridge: [Description]
     * @param timeProvider: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public ScriptServiceBridge(final SettingsBridge settingsBridge, final LongSupplier timeProvider) throws IOException {
        super(getScriptService(settingsBridge.toInternal(), timeProvider));
    }

    /**
     * @brief [Functional Utility for ScriptServiceBridge]: Describe purpose here.
     * @param delegate: [Description]
     * @return [ReturnType]: [Description]
     */
    public ScriptServiceBridge(ScriptService delegate) {
        super(delegate);
    }

    /**
     * @brief [Functional Utility for getScriptService]: Describe purpose here.
     * @param settings: [Description]
     * @param timeProvider: [Description]
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    private static ScriptService getScriptService(final Settings settings, final LongSupplier timeProvider) throws IOException {
        final List<Whitelist> painlessBaseWhitelist = getPainlessBaseWhiteList();
        final Map<ScriptContext<?>, List<Whitelist>> scriptContexts = Map.of(
            IngestScript.CONTEXT,
            painlessBaseWhitelist,
            IngestConditionalScript.CONTEXT,
            painlessBaseWhitelist
        );
        final Map<String, ScriptEngine> scriptEngines = Map.of(
            PainlessScriptEngine.NAME,
            getPainlessScriptEngine(settings),
            MustacheScriptEngine.NAME,
    /**
     * @brief [Functional Utility for MustacheScriptEngine]: Describe purpose here.
     * @param settings: [Description]
     * @return [ReturnType]: [Description]
     */
            new MustacheScriptEngine(settings)
        );
        return new ScriptService(settings, scriptEngines, ScriptModule.CORE_CONTEXTS, timeProvider);
    }

    /**
     * @brief [Functional Utility for getPainlessBaseWhiteList]: Describe purpose here.
     * @return [ReturnType]: [Description]
     */
    private static List<Whitelist> getPainlessBaseWhiteList() {
        return PainlessPlugin.baseWhiteList();
    }

    /**
     * @param settings the Elasticsearch settings object
     * @return a {@link ScriptEngine} for painless scripts for use in {@link IngestScript} and
     *         {@link IngestConditionalScript} contexts, including all available {@link PainlessExtension}s.
     * @throws IOException when the underlying script engine cannot be created
     */
    private static ScriptEngine getPainlessScriptEngine(final Settings settings) throws IOException {
        try (PainlessPlugin painlessPlugin = new PainlessPlugin()) {
            painlessPlugin.loadExtensions(new ExtensiblePlugin.ExtensionLoader() {
                @Override
                @SuppressWarnings("unchecked")
                public <T> List<T> loadExtensions(Class<T> extensionPointType) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                    if (extensionPointType.isAssignableFrom(PainlessExtension.class)) {
                        final List<PainlessExtension> extensions = new ArrayList<>();

                        extensions.add(new ConstantKeywordPainlessExtension());  // module: constant-keyword
                        extensions.add(new ProcessorsWhitelistExtension());      // module: ingest-common
                        extensions.add(new SpatialPainlessExtension());          // module: spatial
                        extensions.add(new WildcardPainlessExtension());         // module: wildcard

                        return (List<T>) extensions;
        // Block Logic: [Describe purpose of this else/else if block]
                    } else {
                        return List.of();
                    }
                }
            });

            return painlessPlugin.getScriptEngine(settings, Set.of(IngestScript.CONTEXT, IngestConditionalScript.CONTEXT));
        }
    }

    @Override
    /**
     * @brief [Functional Utility for close]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void close() throws IOException {
        this.internalDelegate.close();
    }
}
