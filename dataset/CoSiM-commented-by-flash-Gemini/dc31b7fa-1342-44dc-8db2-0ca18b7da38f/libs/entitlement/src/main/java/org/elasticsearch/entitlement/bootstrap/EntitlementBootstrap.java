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
 * @dc31b7fa-1342-44dc-8db2-0ca18b7da38f/libs/entitlement/src/main/java/org/elasticsearch/entitlement/bootstrap/EntitlementBootstrap.java
 * @brief Bootstrapping component for Elasticsearch's entitlement subsystem using dynamic Java agent attachment.
 * * Domain: Production Systems, Security Entitlements, Runtime Resource Access Control.
 * * Functional Utility: Configures the runtime environment for entitlement checks by injecting a Java agent 
 *   into the current process, enabling instrumented security policies across the cluster and plugins.
 */
public class EntitlementBootstrap {

    /**
     * @brief Entry point for activating the entitlement checking engine.
     * Logic: Initializes the path lookup implementation, bundles initialization arguments, 
     * exports required packages to the agent, and performs the dynamic attachment.
     * 
     * @param serverPolicyPatch            Policy augmentations for the core server.
     * @param pluginPolicies               Map of per-plugin security policies.
     * @param scopeResolver                Function to map classes to entitlement scopes.
     * @param settingResolver              Function to resolve Elasticsearch settings patterns.
     * @param dataDirs                     Array of filesystem paths for data storage.
     * @param sharedRepoDirs               Paths for shared snapshots/repositories.
     * @param configDir                    Location of node configuration files.
     * @param libDir                       Location of core library files.
     * @param modulesDir                   Location of internal system modules.
     * @param pluginsDir                   Location of external plugin installations.
     * @param pluginSourcePaths            Map of plugin names to their source code locations.
     * @param logsDir                      Location for system audit and error logs.
     * @param tempDir                      Location for transient runtime files.
     * @param pidFile                      Path to the process identifier file.
     * @param suppressFailureLogPackages   Set of packages exempted from entitlement failure logging.
     * 
     * @throws IllegalStateException if initialization data is already set or agent attachment fails.
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
        
        // Logic: Ensures idempotency of the bootstrap process.
        if (EntitlementInitialization.initializeArgs != null) {
            throw new IllegalStateException("initialization data is already set");
        }
        
        // Block Logic: Construct the runtime path context for policy evaluation.
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
        
        // Logic: Packages the static configuration needed by the agent upon startup.
        EntitlementInitialization.initializeArgs = new EntitlementInitialization.InitializeArgs(
            serverPolicyPatch,
            pluginPolicies,
            scopeResolver,
            pathLookup,
            pluginSourcePaths,
            suppressFailureLogPackages
        );
        
        exportInitializationToAgent();
        loadAgent(findAgentJar(), EntitlementInitialization.class.getName());
    }

    /**
     * @brief Retrieves the user's home directory from system properties.
     * @return Path object representing the user home.
     */
    private static Path getUserHome() {
        String userHome = System.getProperty("user.home");
        if (userHome == null) {
            throw new IllegalStateException("user.home system property is required");
        }
        return PathUtils.get(userHome);
    }

    /**
     * @brief Dynamically attaches the Java agent to the current process.
     * * Optimization: Uses Sun's Attach API to inject the entitlement logic at runtime 
     *   without requiring -javaagent at startup.
     * 
     * @param agentPath                         Filesystem path to the agent JAR.
     * @param entitlementInitializationClassName The class responsible for agent entry.
     */
    @SuppressForbidden(reason = "The VirtualMachine API is the only way to attach a java agent dynamically")
    static void loadAgent(String agentPath, String entitlementInitializationClassName) {
        try {
            // Logic: Attaches to the current PID to load the instrumentation agent.
            VirtualMachine vm = VirtualMachine.attach(Long.toString(ProcessHandle.current().pid()));
            try {
                vm.loadAgent(agentPath, entitlementInitializationClassName);
            } finally {
                // Post-condition: Detaches from the VM once the agent is successfully loaded.
                vm.detach();
            }
        } catch (AttachNotSupportedException | IOException | AgentLoadException | AgentInitializationException e) {
            throw new IllegalStateException("Unable to attach entitlement agent", e);
        }
    }

    /**
     * @brief Exports the initialization package to the unnamed module of the system class loader.
     * Functional Utility: Facilitates interoperability between the core application modules and 
     * the dynamically loaded agent which resides in the unnamed module.
     */
    private static void exportInitializationToAgent() {
        String initPkg = EntitlementInitialization.class.getPackageName();
        // agent will live in unnamed module
        Module unnamedModule = ClassLoader.getSystemClassLoader().getUnnamedModule();
        EntitlementInitialization.class.getModule().addExports(initPkg, unnamedModule);
    }

    /**
     * @brief Locates the entitlement agent JAR file within the Elasticsearch installation.
     * Algorithm: Priority-based lookup (System Property -> Installation Directory scan).
     * @return Full path to the agent JAR.
     */
    static String findAgentJar() {
        String propertyName = "es.entitlement.agentJar";
        String propertyValue = System.getProperty(propertyName);
        if (propertyValue != null) {
            return propertyValue;
        }

        // Logic: Defaults to scanning the standard 'lib/entitlement-agent' folder.
        Path esHome = Path.of(System.getProperty("es.path.home"));
        Path dir = esHome.resolve("lib/entitlement-agent");
        if (Files.exists(dir) == false) {
            throw new IllegalStateException("Directory for entitlement jar does not exist: " + dir);
        }
        
        try (var s = Files.list(dir)) {
            // Logic: Validates that exactly one JAR exists in the target directory.
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
