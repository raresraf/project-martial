/**
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
 * @brief Handles the bootstrapping of the Elasticsearch entitlement system.
 *
 * @details This class is responsible for the critical task of dynamically attaching a Java agent
 * to the currently running JVM. This agent is the core of the entitlement enforcement
 * mechanism, intercepting method calls to check for appropriate licenses or permissions.
 * The process involves preparing initialization data, making it accessible to the agent,
 * and using the Attach API to load the agent into the VM.
 */
public class EntitlementBootstrap {

    private static final Logger logger = LogManager.getLogger(EntitlementBootstrap.class);

    /**
     * @brief Main entry point that activates entitlement checking.
     *
     * @details Once this method returns, the entitlement agent is active. Calls to methods protected
     * by entitlements from classes without a valid policy will throw
     * {@link org.elasticsearch.entitlement.runtime.api.NotEntitledException}.
     *
     * This method orchestrates three main steps:
     * 1. Bundling all configuration and policy data into a static context
     *    ({@link EntitlementInitialization.InitializeArgs}).
     * 2. Exporting the package containing this context so the dynamically loaded agent can access it.
     * 3. Finding and loading the entitlement agent JAR into the current JVM.
     *
     * @param serverPolicyPatch          Additional entitlements to patch the embedded server layer policy.
     * @param pluginPolicies             Maps each plugin name to its corresponding {@link Policy}.
     * @param scopeResolver              A function to map a Java Class to its component and module.
     * @param settingResolver            A function to resolve setting name patterns.
     * @param dataDirs                   Data directories for Elasticsearch.
     * @param sharedRepoDirs             Shared repository directories.
     * @param configDir                  The config directory.
     * @param libDir                     The lib directory.
     * @param modulesDir                 The modules directory.
     * @param pluginsDir                 The plugins directory.
     * @param pluginSourcePaths          Maps each plugin name to the location of its code.
     * @param logsDir                    The log directory.
     * @param tempDir                    The temp directory.
     * @param pidFile                    Path to the pid file, or {@code null}.
     * @param suppressFailureLogPackages Packages for which to suppress entitlement failure logs.
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
        if (EntitlementInitialization.initializeArgs != null) {
            throw new IllegalStateException("initialization data is already set");
        }
        // Step 1: Prepare the initialization data payload for the agent.
        // This object is stored in a static field, acting as a side-channel to pass
        // complex state to the agent, which cannot receive such objects directly.
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
        // Step 2: Make the initialization payload visible to the agent's classloader.
        exportInitializationToAgent();
        // Step 3: Find the agent JAR and load it into the running JVM.
        loadAgent(findAgentJar(), EntitlementInitialization.class.getName());
    }

    /**
     * @brief Retrieves the user's home directory path.
     */
    private static Path getUserHome() {
        String userHome = System.getProperty("user.home");
        if (userHome == null) {
            throw new IllegalStateException("user.home system property is required");
        }
        return PathUtils.get(userHome);
    }

    /**
     * @brief Dynamically attaches a Java agent to the current JVM.
     * @param agentPath The file path to the agent JAR.
     * @param entitlementInitializationClassName The fully qualified name of the class to pass to the agent.
     *
     * @details This method uses the non-standard but widely used Attach API (`com.sun.tools.attach`)
     * to connect to its own JVM process and command it to load the agent.
     */
    @SuppressForbidden(reason = "The VirtualMachine API is the only way to attach a java agent dynamically")
    static void loadAgent(String agentPath, String entitlementInitializationClassName) {
        try {
            // Get the current process ID and attach a virtual machine instance to it.
            VirtualMachine vm = VirtualMachine.attach(Long.toString(ProcessHandle.current().pid()));
            try {
                // Instruct the JVM to load the agent JAR. This triggers the agent's `agentmain` method.
                vm.loadAgent(agentPath, entitlementInitializationClassName);
            } finally {
                vm.detach();
            }
        } catch (AttachNotSupportedException | IOException | AgentLoadException | AgentInitializationException e) {
            throw new IllegalStateException("Unable to attach entitlement agent", e);
        }
    }

    /**
     * @brief Modifies module permissions to allow the agent to access initialization data.
     *
     * @details The entitlement agent is loaded into the "unnamed module" by the JVM.
     * For the agent to be able to read the static `EntitlementInitialization.initializeArgs` field,
     * the module where `EntitlementInitialization` resides must explicitly export its package
     * to the unnamed module. This method performs that runtime module modification.
     */
    private static void exportInitializationToAgent() {
        String initPkg = EntitlementInitialization.class.getPackageName();
        Module unnamedModule = ClassLoader.getSystemClassLoader().getUnnamedModule();
        EntitlementInitialization.class.getModule().addExports(initPkg, unnamedModule);
    }

    /**
     * @brief Finds the path to the entitlement agent JAR file.
     * @return The absolute path to the agent JAR.
     *
     * @details It first checks the system property "es.entitlement.agentJar" for a direct path.
     * If not found, it defaults to looking for a single JAR file within the
     * `lib/entitlement-agent` directory of the Elasticsearch home path.
     */
    static String findAgentJar() {
        String propertyName = "es.entitlement.agentJar";
        String propertyValue = System.getProperty(propertyName);
        if (propertyValue != null) {
            return propertyValue;
        }

        Path esHome = Path.of(System.getProperty("es.path.home"));
        Path dir = esHome.resolve("lib/entitlement-agent");
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
}
