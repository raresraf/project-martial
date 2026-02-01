/**
 * @file EntitlementBootstrap.java
 * @brief This file contains the bootstrap logic for activating the Elasticsearch entitlement agent.
 *
 * @details
 * The `EntitlementBootstrap` class is the main entry point for dynamically attaching a Java agent
 * that enforces entitlement checks at runtime. The core functionality is to load and initialize
 * this agent, which then uses Java Instrumentation to intercept method calls and verify whether
 * the caller has the required entitlements based on predefined policies.
 *
 * The bootstrapping process involves:
 * 1.  Collecting all necessary configuration and policy data into a `BootstrapArgs` record.
 *     This includes security policies for the server and plugins, functions for resolving
 *     class scopes and settings, and paths to various Elasticsearch directories.
 * 2.  Dynamically finding and attaching the entitlement agent JAR to the running JVM using the
 *     `com.sun.tools.attach` API. This is a critical step that enables runtime instrumentation.
 * 3.  Exporting the `EntitlementInitialization` package to the agent, allowing the agent to
 *     access the bootstrap arguments and complete its initialization.
 *
 * Once bootstrapped, the agent enforces access control, throwing a `NotEntitledException`
 * for unauthorized operations.
 */
/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.entitlement.bootstrap;

import com.sun.tools.attach.AgentInitializationException;
import com.sun.tools.attach.AgentLoadException;
import com.sun.tools.attach.AttachNotSupportedException;
import com.sun.tools.attach.VirtualMachine;

import org.elasticsearch.core.Nullable;
import org.elasticsearch.core.PathUtils;
import org.elasticsearch.core.SuppressForbidden;
import org.elasticsearch.entitlement.initialization.EntitlementInitialization;
import org.elasticsearch.entitlement.runtime.policy.PathLookup;
import org.elasticsearch.entitlement.runtime.policy.PathLookupImpl;
import org.elasticsearch.entitlement.runtime.policy.Policy;
import org.elasticsearch.entitlement.runtime.policy.PolicyManager;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Stream;

import static java.util.Objects.requireNonNull;

public class EntitlementBootstrap {

    /**
     * A record to hold all the necessary arguments for bootstrapping the entitlement agent.
     * This serves as a data carrier to pass configuration from the main application to the agent.
     *
     * @param serverPolicyPatch      A policy patch to augment the base server entitlements.
     * @param pluginPolicies         A map of policies for each plugin, keyed by plugin name.
     * @param scopeResolver          A function that maps a Java Class to its corresponding component and module scope.
     * @param pathLookup             An object that provides access to various Elasticsearch directory paths.
     * @param sourcePaths            A map from plugin/module name to the path of its JAR file.
     * @param suppressFailureLogClasses A set of classes for which entitlement failures should not be logged.
     */
    public record BootstrapArgs(
        @Nullable Policy serverPolicyPatch,
        Map<String, Policy> pluginPolicies,
        Function<Class<?>, PolicyManager.PolicyScope> scopeResolver,
        PathLookup pathLookup,
        Map<String, Path> sourcePaths,
        Set<Class<?>> suppressFailureLogClasses
    ) {
        public BootstrapArgs {
            requireNonNull(pluginPolicies);
            requireNonNull(scopeResolver);
            requireNonNull(pathLookup);
            requireNonNull(sourcePaths);
            requireNonNull(suppressFailureLogClasses);
        }
    }

    private static BootstrapArgs bootstrapArgs;

    public static BootstrapArgs bootstrapArgs() {
        return bootstrapArgs;
    }

    /**
     * Activates entitlement checking by configuring and loading the Java agent.
     * Once this method returns, the agent is active, and methods protected by entitlements
     * will be enforced.
     *
     * @param serverPolicyPatch A policy with additional entitlements for the server layer.
     * @param pluginPolicies A map of policies for plugins, keyed by plugin name.
     * @param scopeResolver A function to resolve the component scope of a given class.
     * @param settingResolver A function to resolve Elasticsearch setting names.
     * @param dataDirs An array of data directories.
     * @param sharedRepoDirs An array of shared repository directories.
     * @param configDir The configuration directory.
     * @param libDir The library directory.
     * @param modulesDir The modules directory.
     * @param pluginsDir The plugins directory.
     * @param sourcePaths A map from plugin/module name to its JAR file path.
     * @param logsDir The log directory.
     * @param tempDir The temporary directory.
     * @param pidFile Path to the process ID file.
     * @param suppressFailureLogClasses A set of classes for which to suppress entitlement failure logs.
     */
    public static void bootstrap(
        Policy serverPolicyPatch,
        Map<String, Policy> pluginPolicies,
        Function<Class<?>, PolicyManager.PolicyScope> scopeResolver,
        Function<String, Stream<String>> settingResolver,
        Path[] dataDirs,
        Path[] sharedRepoDirs,
        Path configDir,
        Path libDir,
        Path modulesDir,
        Path pluginsDir,
        Map<String, Path> sourcePaths,
        Path logsDir,
        Path tempDir,
        Path pidFile,
        Set<Class<?>> suppressFailureLogClasses
    ) {
        logger.debug("Loading entitlement agent");
        if (EntitlementBootstrap.bootstrapArgs != null) {
            throw new IllegalStateException("plugin data is already set");
        }
        // Functional Utility: Gathers all configuration into a single, immutable record,
        // which is then passed to the agent for initialization.
        EntitlementBootstrap.bootstrapArgs = new BootstrapArgs(
            serverPolicyPatch,
            pluginPolicies,
            scopeResolver,
            new PathLookupImpl(
                getUserHome(),
                configDir,
                dataDirs,
                sharedRepoDirs,
                libDir,
                modulesDir,
                pluginsDir,
                logsDir,
                tempDir,
                pidFile,
                settingResolver
            ),
            sourcePaths,
            suppressFailureLogClasses
        );
        // Makes the bootstrap arguments accessible to the agent.
        exportInitializationToAgent();
        // Finds and dynamically loads the agent into the running JVM.
        loadAgent(findAgentJar());
    }

    private static Path getUserHome() {
        String userHome = System.getProperty("user.home");
        if (userHome == null) {
            throw new IllegalStateException("user.home system property is required");
        }
        return PathUtils.get(userHome);
    }

    /**
     * Dynamically attaches the Java agent to the current JVM process.
     * @param agentPath The file path to the entitlement agent JAR.
     */
    @SuppressForbidden(reason = "The VirtualMachine API is the only way to attach a java agent dynamically")
    private static void loadAgent(String agentPath) {
        try {
            // Functional Utility: Uses the Attach API to connect to the current JVM process.
            VirtualMachine vm = VirtualMachine.attach(Long.toString(ProcessHandle.current().pid()));
            try {
                // Load the agent JAR, which will trigger its `agentmain` method.
                vm.loadAgent(agentPath);
            } finally {
                vm.detach();
            }
        } catch (AttachNotSupportedException | IOException | AgentLoadException | AgentInitializationException e) {
            throw new IllegalStateException("Unable to attach entitlement agent", e);
        }
    }

    /**
     * Exports the package containing the initialization data to the agent's classloader.
     * This is necessary for the agent, which runs in an unnamed module, to access the
     * `bootstrapArgs` set by the main application.
     */
    private static void exportInitializationToAgent() {
        String initPkg = EntitlementInitialization.class.getPackageName();
        // The agent will be loaded into the unnamed module.
        Module unnamedModule = ClassLoader.getSystemClassLoader().getUnnamedModule();
        // Grant the unnamed module access to the initialization package.
        EntitlementInitialization.class.getModule().addExports(initPkg, unnamedModule);
    }

    /**
     * Finds the entitlement agent JAR file within the application's lib directory.
     * @return The absolute path to the agent JAR file as a string.
     */
    private static String findAgentJar() {
        String propertyName = "es.entitlement.agentJar";
        String propertyValue = System.getProperty(propertyName);
        if (propertyValue != null) {
            return propertyValue;
        }

        Path dir = Path.of("lib", "entitlement-agent");
        if (Files.exists(dir) == false) {
            throw new IllegalStateException("Directory for entitlement jar does not exist: " + dir);
        }
        try (var s = Files.list(dir)) {
            var candidates = s.limit(2).toList();
            if (candidates.size() != 1) {
                throw new IllegalStateException("Expected one jar in " + dir + "; found " + candidates.size());
            }
            return candidates.get(0).toString();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to list entitlement jars in: " + dir, e);
        }
    }

    private static final Logger logger = LogManager.getLogger(EntitlementBootstrap.class);
}
