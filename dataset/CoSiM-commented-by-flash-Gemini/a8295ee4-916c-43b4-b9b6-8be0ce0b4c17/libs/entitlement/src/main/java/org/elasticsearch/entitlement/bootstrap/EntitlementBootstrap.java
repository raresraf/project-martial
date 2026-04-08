/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
/**
 * @file EntitlementBootstrap.java
 * @brief Provides the bootstrapping mechanism for the Elasticsearch entitlement system.
 * Functional Utility: This class is responsible for dynamically attaching and loading a Java Agent
 *                     that enforces security policies and entitlements within the JVM at runtime.
 * Domain: Security, JVM Agents, Distributed Systems, Elasticsearch.
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
import org.elasticsearch.entitlement.runtime.policy.PathLookupImpl;
import org.elasticsearch.entitlement.runtime.policy.Policy;
import org.elasticsearch.entitlement.runtime.policy.PolicyManager;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * @class EntitlementBootstrap
 * @brief Handles the initialization and loading of the entitlement Java Agent into the current JVM.
 *
 * Functional Utility: This class serves as the primary entry point for activating the entitlement
 *                     checking mechanism. It gathers necessary configuration and path information,
 *                     packages it for the agent, and then uses the Java Attach API to load the
 *                     agent dynamically.
 * Domain: Security, JVM Agents, Configuration Management.
 */
public class EntitlementBootstrap {

    /**
     * @brief Main entry point that activates entitlement checking.
     * Functional Utility: Configures and initializes the entitlement agent, then dynamically loads it
     *                     into the current JVM. This method must be called to enable runtime entitlement
     *                     enforcement.
     * Post-conditions: Once this method returns, calls to methods protected by entitlements from
     *                  classes without a valid policy will throw {@link org.elasticsearch.entitlement.runtime.api.NotEntitledException}.
     *                  Throws {@link IllegalStateException} if initialization data is already set.
     *
     * @param serverPolicyPatch            additional entitlements to patch the embedded server layer policy.
     * @param pluginPolicies               maps each plugin name to the corresponding {@link Policy}.
     * @param scopeResolver                a functor to map a Java Class to the component and module it belongs to.
     * @param settingResolver              a functor to resolve a setting name pattern for one or more Elasticsearch settings.
     * @param dataDirs                     data directories for Elasticsearch.
     * @param sharedRepoDirs               shared repository directories for Elasticsearch.
     * @param configDir                    the config directory for Elasticsearch.
     * @param libDir                       the lib directory for Elasticsearch.
     * @param modulesDir                   the directory where Elasticsearch modules are.
     * @param pluginsDir                   the directory where plugins are installed for Elasticsearch.
     * @param pluginSourcePaths            maps each plugin name to the location of that plugin's code.
     * @param tempDir                      the temp directory for Elasticsearch.
     * @param logsDir                      the log directory for Elasticsearch.
     * @param pidFile                      path to a pid file for Elasticsearch, or {@code null} if one was not specified.
     * @param suppressFailureLogPackages   packages for which we do not need or want to log Entitlements failures.
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
        Map<String, Collection<Path>> pluginSourcePaths,
        Path logsDir,
        Path tempDir,
        @Nullable Path pidFile,
        Set<Package> suppressFailureLogPackages
    ) {
        logger.debug("Loading entitlement agent");
        // Block Logic: Ensures that the entitlement initialization process is not run multiple times.
        // Pre-condition: `EntitlementInitialization.initializeArgs` must be null.
        if (EntitlementInitialization.initializeArgs != null) {
            throw new IllegalStateException("initialization data is already set");
        }
        // Block Logic: Stores all necessary configuration arguments for the entitlement agent.
        // Functional Utility: Bundles configuration, policies, and path lookups for the agent's startup.
        EntitlementInitialization.initializeArgs = new EntitlementInitialization.InitializeArgs(
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
            pluginSourcePaths,
            suppressFailureLogPackages
        );
        // Functional Utility: Exports necessary initialization data from the current module to the agent's unnamed module.
        exportInitializationToAgent();
        // Functional Utility: Dynamically loads the entitlement agent into the JVM.
        loadAgent(findAgentJar(), EntitlementInitialization.class.getName());
    }

    /**
     * @brief Retrieves the user's home directory.
     * Functional Utility: Provides a portable way to access the current user's home directory,
     *                     which might be used in path lookups.
     * @return {@link Path}: The path to the user's home directory.
     * @throws IllegalStateException if the "user.home" system property is not set.
     */
    private static Path getUserHome() {
        String userHome = System.getProperty("user.home");
        // Block Logic: Checks for the presence of the "user.home" system property.
        // Invariant: The "user.home" property must be defined for proper operation.
        if (userHome == null) {
            throw new IllegalStateException("user.home system property is required");
        }
        return PathUtils.get(userHome);
    }

    /**
     * @brief Dynamically attaches a Java Agent to the current running JVM.
     * Functional Utility: Uses the Java Attach API to inject the entitlement agent into the
     *                     running process, allowing it to instrument code and enforce policies.
     * @param agentPath                        The file system path to the agent JAR.
     * @param entitlementInitializationClassName The fully qualified name of the agent's initialization class.
     * @throws IllegalStateException if attaching or loading the agent fails.
     * @see SuppressForbidden The VirtualMachine API is the only way to attach a java agent dynamically.
     */
    @SuppressForbidden(reason = "The VirtualMachine API is the only way to attach a java agent dynamically")
    static void loadAgent(String agentPath, String entitlementInitializationClassName) {
        try {
            // Block Logic: Attaches to the current JVM process.
            VirtualMachine vm = VirtualMachine.attach(Long.toString(ProcessHandle.current().pid()));
            try {
                // Block Logic: Loads the specified agent JAR with its initialization class.
                vm.loadAgent(agentPath, entitlementInitializationClassName);
            } finally {
                // Block Logic: Detaches from the JVM process, releasing resources.
                vm.detach();
            }
        } catch (AttachNotSupportedException | IOException | AgentLoadException | AgentInitializationException e) {
            // Functional Utility: Catches various exceptions that can occur during agent attachment/loading
            //                     and re-throws them as an IllegalStateException for consistent error handling.
            throw new IllegalStateException("Unable to attach entitlement agent", e);
        }
    }

    /**
     * @brief Exports necessary packages from the EntitlementInitialization module to the agent's unnamed module.
     * Functional Utility: Facilitates communication and class sharing between the main application
     *                     and the dynamically loaded agent, especially in a modular Java environment.
     */
    private static void exportInitializationToAgent() {
        String initPkg = EntitlementInitialization.class.getPackageName();
        // Block Logic: Retrieves the unnamed module of the system class loader, where dynamically loaded agents reside.
        Module unnamedModule = ClassLoader.getSystemClassLoader().getUnnamedModule();
        // Functional Utility: Adds an export from the EntitlementInitialization's module to the unnamed module,
        //                     making its package visible to the agent.
        EntitlementInitialization.class.getModule().addExports(initPkg, unnamedModule);
    }

    /**
     * @brief Locates the entitlement agent JAR file.
     * Functional Utility: Determines the file system path to the agent JAR, first by checking a system property,
     *                     then by searching a predefined directory relative to `es.path.home`.
     * @return {@link String}: The absolute path to the agent JAR.
     * @throws IllegalStateException if the agent JAR cannot be found or if there are multiple candidates.
     */
    static String findAgentJar() {
        String propertyName = "es.entitlement.agentJar";
        String propertyValue = System.getProperty(propertyName);
        // Block Logic: Checks if the agent JAR path is specified via a system property.
        if (propertyValue != null) {
            return propertyValue;
        }

        // Block Logic: If not specified by property, constructs the expected path based on "es.path.home".
        Path esHome = Path.of(System.getProperty("es.path.home"));
        Path dir = esHome.resolve("lib/entitlement-agent");
        // Block Logic: Verifies that the expected agent directory exists.
        if (Files.exists(dir) == false) {
            throw new IllegalStateException("Directory for entitlement jar does not exist: " + dir);
        }
        try (var s = Files.list(dir)) {
            // Block Logic: Lists files in the agent directory and checks for exactly one JAR file.
            var candidates = s.limit(2).toList();
            // Invariant: Expects exactly one JAR file in the agent directory.
            if (candidates.size() != 1) {
                throw new IllegalStateException("Expected one jar in " + dir + "; found " + candidates.size());
            }
            return candidates.get(0).toString();
        } catch (IOException e) {
            // Functional Utility: Handles I/O errors during directory listing and wraps them in an IllegalStateException.
            throw new IllegalStateException("Failed to list entitlement jars in: " + dir, e);
        }
    }

    /**
     * @brief Logger instance for recording events and debugging information related to entitlement bootstrapping.
     */
    private static final Logger logger = LogManager.getLogger(EntitlementBootstrap.class);
}
