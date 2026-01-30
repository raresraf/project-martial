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
 * @brief Manages the activation and bootstrapping of the entitlement checking mechanism in Elasticsearch.
 *
 * This file contains the core logic for dynamically loading a Java agent into the running JVM to enforce
 * security policies. It handles the initialization of entitlement arguments, agent loading, and ensures
 * compatibility with the Java Platform Module System.
 *
 * Performance Optimization: Dynamic agent loading minimizes application startup overhead by delaying
 *                           instrumentation until required.
 * Functional Utility: Centralizes the control over feature and plugin entitlement verification, preventing
 *                     unauthorized operations within the Elasticsearch ecosystem.
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

/**
 * @class EntitlementBootstrap
 * @brief This class is responsible for activating and bootstrapping the entitlement
 *        checking mechanism for Elasticsearch.
 *
 * It manages the dynamic loading of a Java agent into the currently running JVM,
 * which then enforces security policies related to various features and plugins.
 * This ensures that only entitled operations are allowed to proceed.
 */
public class EntitlementBootstrap {

    /**
     * @record BootstrapArgs
     * @brief A record (data class) that encapsulates all necessary arguments
     *        for bootstrapping the entitlement agent.
     *
     * This record provides a concise way to pass a collection of configuration
     * parameters required by the entitlement system.
     */
    public record BootstrapArgs(
        @Nullable Policy serverPolicyPatch, /**< An optional policy with additional entitlements to patch the embedded server layer policy. */
        Map<String, Policy> pluginPolicies, /**< A map holding policies for plugins (and modules), by plugin (or module) name. */
        Function<Class<?>, String> pluginResolver, /**< A functor to map a Java Class to the plugin it belongs to (the plugin name). */
        PathLookup pathLookup, /**< A component for resolving various Elasticsearch paths (e.g., config, data, logs). */
        Map<String, Path> sourcePaths, /**< A map holding the path to each plugin or module JAR, by plugin (or module) name. */
        Set<Class<?>> suppressFailureLogClasses /**< A set of classes for which entitlement failures should not be logged. */
    ) {
        /**
         * @brief Canonical constructor for `BootstrapArgs`.
         * Functional Utility: Ensures that essential arguments are not null upon instantiation.
         * Precondition: `pluginPolicies`, `pluginResolver`, `pathLookup`, `sourcePaths`, and `suppressFailureLogClasses` must not be null.
         * Postcondition: A valid `BootstrapArgs` instance is created with non-null essential fields.
         * @throws NullPointerException if any required non-nullable argument is null.
         */
        public BootstrapArgs {
            requireNonNull(pluginPolicies);
            requireNonNull(pluginResolver);
            requireNonNull(pathLookup);
            requireNonNull(sourcePaths);
            requireNonNull(suppressFailureLogClasses);
        }
    }

    /**
     * @brief Static field to hold the single instance of `BootstrapArgs` once initialized.
     * Invariant: This field is set once during the `bootstrap` process and remains immutable thereafter.
     * Functional Utility: Ensures that entitlement configuration is globally accessible and consistently applied.
     */
    private static BootstrapArgs bootstrapArgs;

    /**
     * @brief Static getter method to retrieve the initialized `BootstrapArgs` instance.
     * @return The `BootstrapArgs` instance containing the entitlement bootstrap configuration.
     * Precondition: The `bootstrap` method must have been successfully called, ensuring `bootstrapArgs` is initialized.
     * Postcondition: Returns the globally stored `BootstrapArgs` instance.
     * @throws IllegalStateException if `bootstrapArgs` has not been initialized yet.
     */
    public static BootstrapArgs bootstrapArgs() {
        return bootstrapArgs;
    }

    /**
     * @brief Activates the entitlement checking mechanism.
     *
     * Functional Utility: This is the main entry point for configuring and launching the
     * entitlement agent. Once this method successfully returns, calls to methods protected
     * by Entitlements from classes without a valid policy will throw
     * {@link org.elasticsearch.entitlement.runtime.api.NotEntitledException}.
     * It initializes the global `bootstrapArgs` record, exports necessary data for the
     * agent, and then dynamically attaches the agent to the JVM.
     *
     * @param serverPolicyPatch           An optional policy with additional entitlements to patch the embedded server layer policy.
     * @param pluginPolicies              A map holding policies for plugins (and modules), by plugin (or module) name.
     * @param pluginResolver              A functor to map a Java Class to the plugin it belongs to (the plugin name).
     * @param settingResolver             A functor to resolve a setting name pattern for one or more Elasticsearch settings.
     * @param dataDirs                    An array of data directories for Elasticsearch.
     * @param sharedRepoDirs              An array of shared repository directories for Elasticsearch.
     * @param configDir                   The configuration directory for Elasticsearch.
     * @param libDir                      The library directory for Elasticsearch.
     * @param modulesDir                  The directory where Elasticsearch modules are located.
     * @param pluginsDir                  The directory where plugins are installed for Elasticsearch.
     * @param sourcePaths                 A map holding the path to each plugin or module JAR, by plugin (or module) name.
     * @param logsDir                     The logging directory for Elasticsearch.
     * @param tempDir                     The temporary directory for Elasticsearch.
     * @param pidFile                     Path to a PID file for Elasticsearch, or {@code null} if one was not specified.
     * @param suppressFailureLogClasses   A set of classes for which we do not need or want to log Entitlements failures.
     * Precondition: This method must only be called once during the application's lifecycle.
     * Postcondition: The global `bootstrapArgs` is initialized, the entitlement agent is attached, and entitlement checking is active.
     * @throws IllegalStateException      If entitlement checking has already been bootstrapped, or if unable to attach the entitlement agent.
     */
    public static void bootstrap(
        Policy serverPolicyPatch,
        Map<String, Policy> pluginPolicies,
        Function<Class<?>, String> pluginResolver,
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
        /**
         * Block Logic: Ensures the entitlement bootstrapping process is idempotent.
         * Invariant: `EntitlementBootstrap.bootstrapArgs` must be null before initialization.
         */
        if (EntitlementBootstrap.bootstrapArgs != null) {
            throw new IllegalStateException("plugin data is already set");
        }
        // Initialize the static `bootstrapArgs` record with all provided configuration.
        EntitlementBootstrap.bootstrapArgs = new BootstrapArgs(
            serverPolicyPatch,
            pluginPolicies,
            pluginResolver,
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
        // Export initialization data to make it accessible to the agent.
        exportInitializationToAgent();
        // Locate and load the Java agent dynamically.
        loadAgent(findAgentJar());
    }

    /**
     * @brief Retrieves the user's home directory path.
     * @return The `Path` object representing the user's home directory.
     * Precondition: The "user.home" system property must be set in the JVM environment.
     * Postcondition: Returns a valid `Path` object for the user's home directory.
     * @throws IllegalStateException If the "user.home" system property is not set.
     */
    private static Path getUserHome() {
        String userHome = System.getProperty("user.home");
        /**
         * Block Logic: Validates the existence of the "user.home" system property.
         * Invariant: The "user.home" property is a mandatory component for path resolution.
         */
        if (userHome == null) {
            throw new IllegalStateException("user.home system property is required");
        }
        return PathUtils.get(userHome);
    }

    /**
     * @brief Dynamically attaches a Java agent to the currently running JVM.
     *
     * Functional Utility: Uses the `VirtualMachine` API (from `com.sun.tools.attach`)
     * to attach the agent specified by `agentPath`. This allows the agent to
     * instrument bytecode at runtime, enforcing entitlement policies.
     *
     * @param agentPath The absolute path to the Java agent JAR file.
     * Precondition: The `agentPath` must point to a valid and accessible Java agent JAR file.
     * Postcondition: The Java agent is loaded and active within the current JVM.
     * @throws IllegalStateException If any error occurs during the attachment process (e.g., agent not found, security issues).
     */
    @SuppressForbidden(reason = "The VirtualMachine API is the only way to attach a java agent dynamically")
    private static void loadAgent(String agentPath) {
        VirtualMachine vm = null; // Declare vm outside try block for finally.
        try {
            // Attach to the current JVM process using its PID.
            vm = VirtualMachine.attach(Long.toString(ProcessHandle.current().pid()));
            // Load the Java agent JAR.
            vm.loadAgent(agentPath);
        } catch (AttachNotSupportedException | IOException | AgentLoadException | AgentInitializationException e) {
            // Error Handling: Catch various attachment-related exceptions and wrap them.
            throw new IllegalStateException("Unable to attach entitlement agent", e);
        } finally {
            /**
             * Block Logic: Ensures proper cleanup by detaching from the VirtualMachine.
             * Invariant: The VirtualMachine connection should be closed regardless of whether agent loading succeeded or failed.
             */
            if (vm != null) {
                try {
                    vm.detach(); // Always detach the VirtualMachine.
                } catch (IOException e) {
                    // Logging: Records any failures encountered during the detachment process.
                    logger.warn("Failed to detach VirtualMachine", e); // Log detach failure.
                }
            }
        }
    }

    /**
     * @brief Exports the entitlement initialization package to the unnamed module
     *        of the system class loader.
     *
     * Functional Utility: This is a crucial step for Java Platform Module System
     * compatibility. It ensures that the dynamically loaded agent (which typically
     * resides in the unnamed module) can reflectively access classes within
     * the `org.elasticsearch.entitlement.initialization` package to receive its configuration.
     * Precondition: The `EntitlementInitialization` class must be accessible.
     * Postcondition: The `org.elasticsearch.entitlement.initialization` package is exported to the unnamed module, enabling agent access.
     */
    private static void exportInitializationToAgent() {
        String initPkg = EntitlementInitialization.class.getPackageName();
        Module unnamedModule = ClassLoader.getSystemClassLoader().getUnnamedModule();
        // Module System Compatibility: Explicitly exports the initialization package to allow reflective access from the unnamed module.
        EntitlementInitialization.class.getModule().addExports(initPkg, unnamedModule);
    }

    /**
     * @brief Locates the entitlement agent JAR file.
     *
     * Functional Utility: Determines the path to the agent JAR. It first checks for
     * a system property `es.entitlement.agentJar`. If not set, it scans a predefined
     * directory (`lib/entitlement-agent`) and expects to find exactly one JAR file.
     *
     * @return The absolute path to the entitlement agent JAR file.
     * Precondition: The system property `es.entitlement.agentJar` may or may not be set. If not set, a `lib/entitlement-agent` directory must exist and contain exactly one JAR.
     * Postcondition: Returns the validated path to the entitlement agent JAR.
     * @throws IllegalStateException If the system property is not set and the default
     *                               directory is missing, contains multiple JARs, or
     *                               an I/O error occurs.
     */
    private static String findAgentJar() {
        String propertyName = "es.entitlement.agentJar";
        String propertyValue = System.getProperty(propertyName);
        /**
         * Block Logic: Prioritizes a system property for the agent JAR path.
         * Invariant: If the property is set, its value is directly used as the agent path.
         */
        if (propertyValue != null) {
            return propertyValue; // Use path from system property if available.
        }

        // Default location if no system property is set.
        Path dir = Path.of("lib", "entitlement-agent");
        /**
         * Block Logic: Validates the existence of the default agent directory.
         * Invariant: The directory specified by `dir` must exist for agent discovery to proceed.
         */
        if (Files.exists(dir) == false) {
            throw new IllegalStateException("Directory for entitlement jar does not exist: " + dir);
        }
        try (var s = Files.list(dir)) { // List files in the directory.
            var candidates = s.limit(2).toList(); // Get up to 2 candidates to check for uniqueness.
            /**
             * Block Logic: Verifies that exactly one agent JAR is present in the default directory.
             * Invariant: The `candidates` list must contain precisely one element to unambiguously identify the agent JAR.
             */
            if (candidates.size() != 1) {
                throw new IllegalStateException("Expected one jar in " + dir + "; found " + candidates.size());
            }
            return candidates.get(0).toString(); // Return the path to the single found JAR.
        } catch (IOException e) {
            // Error Handling: Catch and wrap I/O exceptions during directory listing.
            throw new IllegalStateException("Failed to list entitlement jars in: " + dir, e);
        }
    }

    /**
     * @brief Static `Logger` instance for logging messages related to entitlement bootstrapping.
     * Functional Utility: Provides a centralized mechanism for logging debug, info, warn, and error
     *                     messages during the entitlement initialization process.
     * Invariant: The logger is initialized once when the class is loaded.
     */
    private static final Logger logger = LogManager.getLogger(EntitlementBootstrap.class);
}