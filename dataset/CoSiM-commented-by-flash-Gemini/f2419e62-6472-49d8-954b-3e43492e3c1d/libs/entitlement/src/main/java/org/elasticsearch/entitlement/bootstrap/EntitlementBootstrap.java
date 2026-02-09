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
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * @brief Main entry point for activating and configuring the entitlement checking mechanism within Elasticsearch.
 *
 * This class is responsible for orchestrating the loading of a Java agent, which then
 * performs bytecode instrumentation to inject entitlement checks into various methods.
 * It also initializes the {@link PolicyManager} with the necessary access policies
 * and environmental context (like directory paths).
 *
 * The bootstrapping process ensures that once this method completes, calls to
 * protected methods without proper entitlements will result in a
 * {@link org.elasticsearch.entitlement.runtime.api.NotEntitledException}.
 *
 * Functional Utility: Centralizes the setup of a robust security layer for access control
 *                     to internal Elasticsearch functionalities.
 * Architecture: Utilizes Java Agent technology for dynamic bytecode instrumentation,
 *               decoupling entitlement logic from core application code.
 */
public class EntitlementBootstrap {

    /**
     * @brief Main entry point that activates entitlement checking.
     * Functional Utility: Once this method returns, calls to methods protected by entitlements from
     *                     classes without a valid policy will throw {@link org.elasticsearch.entitlement.runtime.api.NotEntitledException}.
     *                     It orchestrates the initialization of the entitlement framework, including
     *                     loading the Java agent for bytecode instrumentation and setting up the policy manager.
     * Pre-condition: This method should be called only once during the application's lifecycle.
     * Post-condition: The entitlement agent is loaded, and the {@link EntitlementInitialization}
     *                 data is set, enabling runtime entitlement checks.
     *
     * @param serverPolicyPatch Additional entitlements to patch the embedded server layer policy.
     * @param pluginPolicies A map where each plugin name is mapped to its corresponding {@link Policy}.
     * @param scopeResolver A functor to map a Java Class to the component and module it belongs to.
     * @param settingResolver A functor to resolve a setting name pattern for one or more Elasticsearch settings.
     * @param dataDirs Array of data directories for Elasticsearch.
     * @param sharedRepoDirs Array of shared repository directories for Elasticsearch.
     * @param configDir The configuration directory for Elasticsearch.
     * @param libDir The library directory for Elasticsearch.
     * @param modulesDir The directory where Elasticsearch modules are located.
     * @param pluginsDir The directory where plugins are installed for Elasticsearch.
     * @param pluginSourcePaths A map where each plugin name is mapped to the location of that plugin's code.
     * @param logsDir The log directory for Elasticsearch.
     * @param tempDir The temporary directory for Elasticsearch.
     * @param pidFile Path to a PID file for Elasticsearch, or {@code null} if one was not specified.
     * @param suppressFailureLogPackages A set of packages for which entitlement failures should not be logged.
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
        /**
         * Block Logic: Checks if entitlement initialization data has already been set.
         * Functional Utility: Prevents multiple initializations of the entitlement framework,
         *                     ensuring idempotence and avoiding potential inconsistencies.
         * Pre-condition: `EntitlementInitialization.initializeArgs` should be null.
         * Invariant: An {@link IllegalStateException} is thrown if initialization is attempted more than once.
         */
        if (EntitlementInitialization.initializeArgs != null) {
            throw new IllegalStateException("initialization data is already set");
        }
        /**
         * Block Logic: Initializes the {@link PathLookupImpl} with various directory paths and a setting resolver.
         * Functional Utility: Creates a centralized component for resolving file system paths, which is critical
         *                     for policy enforcement and agent operations across the Elasticsearch environment.
         * Pre-condition: All required directory paths and the setting resolver function are provided as arguments.
         * Post-condition: A fully configured {@link PathLookup} instance is available for the entitlement system.
         */
        PathLookupImpl pathLookup = new PathLookupImpl(
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
        );
        /**
         * Block Logic: Assigns the initialized {@link PathLookup} and policy manager to {@link EntitlementInitialization.initializeArgs}.
         * Functional Utility: Makes the core initialization data available to the Java agent,
         *                     which will use it to configure the entitlement checks dynamically.
         * Pre-condition: `pathLookup` and `createPolicyManager` return valid instances.
         * Post-condition: The static `initializeArgs` field in {@link EntitlementInitialization} is populated.
         */
        EntitlementInitialization.initializeArgs = new EntitlementInitialization.InitializeArgs(
            pathLookup,
            suppressFailureLogPackages,
            createPolicyManager(pluginPolicies, pathLookup, serverPolicyPatch, scopeResolver, pluginSourcePaths)
        );
        /**
         * Block Logic: Exports initialization data to the Java agent.
         * Functional Utility: Prepares the environment for the agent by ensuring it has access to
         *                     necessary initialization classes and data.
         * Pre-condition: {@link EntitlementInitialization.initializeArgs} has been successfully populated.
         */
        exportInitializationToAgent();
        /**
         * Block Logic: Loads the Java agent dynamically into the currently running JVM.
         * Functional Utility: Activates the bytecode instrumentation process, enabling runtime
         *                     entitlement checks as defined by the loaded policies.
         * Pre-condition: The entitlement agent JAR is locatable, and `entitlementInitializationClassName` is valid.
         * Post-condition: The Java agent is attached, and its `agentmain` method is invoked.
         */
        loadAgent(findAgentJar(), EntitlementInitialization.class.getName());
    }

    /**
     * @brief Retrieves the user's home directory path.
     * @return A {@link Path} object representing the user's home directory.
     * @throws IllegalStateException if the "user.home" system property is not set.
     * Functional Utility: Provides a portable way to access the user's home directory,
     *                     which is often needed for resolving relative paths or configurations.
     */
    private static Path getUserHome() {
        String userHome = System.getProperty("user.home");
        /**
         * Block Logic: Checks if the "user.home" system property is null.
         * Functional Utility: Ensures that the user home directory is resolvable,
         *                     as it's a critical component for {@link PathLookupImpl} initialization.
         * Pre-condition: The "user.home" system property should be set by the JVM.
         * Invariant: An {@link IllegalStateException} is thrown if the property is missing, halting execution.
         */
        if (userHome == null) {
            throw new IllegalStateException("user.home system property is required");
        }
        return PathUtils.get(userHome);
    }

    @SuppressForbidden(reason = "The VirtualMachine API is the only way to attach a java agent dynamically")
    /**
     * @brief Dynamically loads a Java agent into the currently running JVM process.
     * @param agentPath The absolute path to the agent JAR file.
     * @param entitlementInitializationClassName The fully qualified class name of the agent's initialization class (agent class).
     * Functional Utility: Attaches the agent to the JVM, enabling it to perform bytecode instrumentation
     *                     and other agent-specific tasks at runtime. This is crucial for dynamic
     *                     entitlement checking.
     * Pre-condition: The JVM must support the Attach API, and the `agentPath` must point to a valid JAR.
     * Post-condition: The Java agent is loaded and its `agentmain` method is invoked with `entitlementInitializationClassName` as argument.
     * @throws IllegalStateException if there's any failure during the agent attachment process.
     */
    static void loadAgent(String agentPath, String entitlementInitializationClassName) {
        /**
         * Block Logic: Attaches the current JVM process to itself and loads the specified Java agent.
         * Functional Utility: Uses the Java Attach API to dynamically inject and activate the
         *                     entitlement agent. The `try-finally` block ensures that the attachment
         *                     is always detached, preventing resource leaks.
         * Pre-condition: `agentPath` and `entitlementInitializationClassName` are valid.
         * Invariant: The agent is loaded, and the `VirtualMachine` connection is detached.
         */
        try {
            // Inline: Attaches to the current JVM process using its Process ID.
            VirtualMachine vm = VirtualMachine.attach(Long.toString(ProcessHandle.current().pid()));
            try {
                // Inline: Loads the agent JAR and passes the initialization class name as an argument to its agentmain method.
                vm.loadAgent(agentPath, entitlementInitializationClassName);
            } finally {
                // Inline: Detaches from the JVM, releasing the resources associated with the attachment.
                vm.detach();
            }
        } catch (AttachNotSupportedException | IOException | AgentLoadException | AgentInitializationException e) {
            throw new IllegalStateException("Unable to attach entitlement agent", e);
        }
    }

    /**
     * @brief Exports the {@link EntitlementInitialization} package to the unnamed module of the system class loader.
     * Functional Utility: Ensures that the dynamically loaded Java agent, which typically resides in an
     *                     unnamed module, can access classes within the `org.elasticsearch.entitlement.initialization`
     *                     package. This is crucial for the agent to retrieve and utilize the initialization data.
     * Pre-condition: {@link EntitlementInitialization.initializeArgs} has been populated with necessary data.
     * Post-condition: Classes within the `initPkg` are accessible to the agent.
     */
    private static void exportInitializationToAgent() {
        String initPkg = EntitlementInitialization.class.getPackageName();
        // agent will live in unnamed module
        Module unnamedModule = ClassLoader.getSystemClassLoader().getUnnamedModule();
        EntitlementInitialization.class.getModule().addExports(initPkg, unnamedModule);
    }

    /**
     * @brief Locates the path to the entitlement agent JAR file.
     * @return The absolute path to the agent JAR file as a {@link String}.
     * @throws IllegalStateException if the agent JAR path is not specified, the directory does not exist,
     *                               or more than one JAR is found in the expected directory.
     * Functional Utility: Provides a robust mechanism to discover the agent JAR required for dynamic loading,
     *                     supporting both explicit property configuration and convention-based lookup.
     */
    static String findAgentJar() {
        String propertyName = "es.entitlement.agentJar";
        String propertyValue = System.getProperty(propertyName);
        /**
         * Block Logic: Checks if the agent JAR path is explicitly provided via a system property.
         * Functional Utility: Allows for overriding the default agent JAR location, useful in custom deployment scenarios.
         * Pre-condition: The system property `es.entitlement.agentJar` may or may not be set.
         * Invariant: If set, its value is used directly; otherwise, a convention-based lookup proceeds.
         */
        if (propertyValue != null) {
            return propertyValue;
        }

        Path dir = Path.of("lib", "entitlement-agent");
        /**
         * Block Logic: Verifies the existence of the expected directory for the agent JAR.
         * Functional Utility: Ensures that the convention-based lookup path is valid before attempting to list files.
         * Pre-condition: `dir` is the expected path relative to the current working directory.
         * Invariant: An {@link IllegalStateException} is thrown if the directory does not exist.
         */
        if (Files.exists(dir) == false) {
            throw new IllegalStateException("Directory for entitlement jar does not exist: " + dir);
        }
        /**
         * Block Logic: Lists files in the agent directory and validates that exactly one JAR is present.
         * Functional Utility: Identifies the specific agent JAR to load when its path is not explicitly configured,
         *                     enforcing the expectation of a single agent JAR for clarity and correctness.
         * Pre-condition: `dir` exists and is accessible.
         * Invariant: If more or less than one JAR is found, an {@link IllegalStateException} is thrown, indicating
         *            an ambiguous or incorrect setup.
         */
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

    /**
     * @brief Creates and initializes a {@link PolicyManager} instance based on provided policies and context.
     * @param pluginPolicies A map of plugin names to their respective {@link Policy} definitions.
     * @param pathLookup An initialized {@link PathLookup} instance for resolving file system paths.
     * @param serverPolicyPatch A {@link Policy} containing additional entitlements to apply to the server's policy.
     * @param scopeResolver A function to determine the {@link PolicyManager.PolicyScope} for a given class.
     * @param pluginSourcePaths A map of plugin names to their source code paths.
     * @return A fully configured {@link PolicyManager} ready for entitlement checks.
     * Functional Utility: Aggregates server, agent, and plugin-specific entitlement policies into a unified manager,
     *                     which will be used by the entitlement agent to enforce access control.
     * Pre-condition: Plugin policies and path lookup information are valid.
     * Post-condition: A {@link PolicyManager} is returned, capable of evaluating entitlement requests against loaded policies.
     */
    private static PolicyManager createPolicyManager(
        Map<String, Policy> pluginPolicies,
        PathLookup pathLookup,
        Policy serverPolicyPatch,
        Function<Class<?>, PolicyManager.PolicyScope> scopeResolver,
        Map<String, Collection<Path>> pluginSourcePaths
    ) {
        FilesEntitlementsValidation.validate(pluginPolicies, pathLookup);

        return new PolicyManager(
            HardcodedEntitlements.serverPolicy(pathLookup.pidFile(), serverPolicyPatch),
            HardcodedEntitlements.agentEntitlements(),
            pluginPolicies,
            scopeResolver,
            pluginSourcePaths,
            pathLookup
        );
    }

    private static final Logger logger = LogManager.getLogger(EntitlementBootstrap.class);
}
