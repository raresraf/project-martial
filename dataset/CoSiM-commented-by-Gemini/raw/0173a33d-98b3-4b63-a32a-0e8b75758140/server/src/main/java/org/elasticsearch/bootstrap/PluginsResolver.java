/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.bootstrap;

import org.elasticsearch.plugins.PluginsLoader;

import java.util.HashMap;
import java.util.Map;

/**
 * @file PluginsResolver.java
 * @brief Resolves a Java class to its corresponding Elasticsearch plugin name.
 *
 * This utility class builds a mapping from Java Modules to plugin names. This allows
 * the entitlement system to identify which plugin a given class belongs to, which is
 * essential for applying the correct security policy.
 */
class PluginsResolver {
    private final Map<Module, String> pluginNameByModule;

    private PluginsResolver(Map<Module, String> pluginNameByModule) {
        this.pluginNameByModule = pluginNameByModule;
    }

    /**
     * @brief Creates and initializes a `PluginsResolver`.
     *
     * It iterates through all loaded plugins, determines their modules (both named and unnamed),
     * and populates a map to associate each module with its parent plugin's name.
     *
     * @param pluginsLoader The `PluginsLoader` containing information about all loaded plugins.
     * @return A new, fully initialized `PluginsResolver` instance.
     */
    public static PluginsResolver create(PluginsLoader pluginsLoader) {
        Map<Module, String> pluginNameByModule = new HashMap<>();

        pluginsLoader.pluginLayers().forEach(pluginLayer -> {
            var pluginName = pluginLayer.pluginBundle().pluginDescriptor().getName();
            // Block Logic: Differentiates between modular and non-modular (JAR-based) plugins.
            if (pluginLayer.pluginModuleLayer() != null && pluginLayer.pluginModuleLayer() != ModuleLayer.boot()) {
                // This plugin is a Java Module, so iterate through its modules.
                for (var module : pluginLayer.pluginModuleLayer().modules()) {
                    pluginNameByModule.put(module, pluginName);
                }
            } else {
                // This plugin is a traditional JAR on the classpath, so it belongs to the unnamed module of its classloader.
                pluginNameByModule.put(pluginLayer.pluginClassLoader().getUnnamedModule(), pluginName);
            }
        });

        return new PluginsResolver(pluginNameByModule);
    }

    /**
     * @brief Resolves the plugin name for a given class.
     *
     * @param clazz The class to resolve.
     * @return The name of the plugin that contains the class, or `null` if the class does not
     *         belong to any known plugin.
     */
    public String resolveClassToPluginName(Class<?> clazz) {
        var module = clazz.getModule();
        return pluginNameByModule.get(module);
    }
}