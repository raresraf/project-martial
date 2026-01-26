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
import java.util.function.Function;

/**
 * @class PluginsResolver
 * @brief This class is responsible for resolving which plugin a given Java class belongs to.
 *
 * Functional Utility: In the context of the Java Module System and Elasticsearch's plugin
 * architecture, this resolver maps `java.lang.Module` objects (and unnamed modules
 * for non-modularized plugins) to their corresponding plugin names. This mapping
 * is crucial for the entitlement system to apply security policies based on a class's origin.
 */
class PluginsResolver {
    /** @brief A map storing the association between a `Module` (or an unnamed module)
     *         and the `String` name of the plugin it belongs to.
     */
    private final Map<Module, String> pluginNameByModule;

    /**
     * @brief Private constructor to initialize the resolver with the pre-built map.
     * @param pluginNameByModule The map containing module-to-plugin name associations.
     */
    private PluginsResolver(Map<Module, String> pluginNameByModule) {
        this.pluginNameByModule = pluginNameByModule;
    }

    /**
     * @brief Static factory method to create and initialize a `PluginsResolver` instance.
     * Functional Utility: It iterates through plugin layers provided by the `PluginsLoader`
     * and populates the internal `pluginNameByModule` map. It correctly differentiates
     * between modularized plugins (which have a defined Java Module) and non-modularized
     * plugins (which reside in an unnamed module).
     * @param pluginsLoader The {@link PluginsLoader} instance providing plugin layer information.
     * @return A newly created and initialized `PluginsResolver` instance.
     */
    public static PluginsResolver create(PluginsLoader pluginsLoader) {
        Map<Module, String> pluginNameByModule = new HashMap<>();

        // Iterate through each plugin layer loaded by the PluginsLoader.
        pluginsLoader.pluginLayers().forEach(pluginLayer -> {
            var pluginName = pluginLayer.pluginBundle().pluginDescriptor().getName(); // Get the name of the plugin.
            // Block Logic: Handle modularized plugins (Java Modules).
            if (pluginLayer.pluginModuleLayer() != null && pluginLayer.pluginModuleLayer() != ModuleLayer.boot()) {
                // If the plugin has its own module layer (i.e., it's a Java Module).
                for (var module : pluginLayer.pluginModuleLayer().modules()) {
                    // Map each module within this plugin's layer to the plugin name.
                    pluginNameByModule.put(module, pluginName);
                }
            }
            // Block Logic: Handle non-modularized plugins.
            else {
                // If the plugin is not modularized, its classes typically reside in the unnamed module
                // of the class loader associated with the plugin layer.
                pluginNameByModule.put(pluginLayer.pluginClassLoader().getUnnamedModule(), pluginName);
            }
        });

        return new PluginsResolver(pluginNameByModule);
    }

    /**
     * @brief Resolves the plugin name for a given Java class.
     * Functional Utility: Retrieves the `Module` associated with the provided class
     * and uses the internal map to find the corresponding plugin name.
     * @param clazz The Java {@link Class} to resolve.
     * @return The `String` name of the plugin the class belongs to, or `null` if the module is not mapped.
     */
    public String resolveClassToPluginName(Class<?> clazz) {
        var module = clazz.getModule(); // Get the module of the class.
        return pluginNameByModule.get(module); // Look up the plugin name in the map.
    }
}
