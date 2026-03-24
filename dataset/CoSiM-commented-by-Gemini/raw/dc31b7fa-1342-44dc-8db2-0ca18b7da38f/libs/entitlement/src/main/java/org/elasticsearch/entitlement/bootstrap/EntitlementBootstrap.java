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
 * A bootstrap utility responsible for dynamically attaching a Java agent at runtime.
 * This agent is used to enforce entitlement policies within the Elasticsearch application,
 * effectively acting as a gate for feature access control.
 */
public class EntitlementBootstrap {

    /**
     * Main entry point that activates entitlement checking. This method prepares all necessary
     * configuration and policies, then loads a Java agent into the current JVM to enforce them.
     * Once this method returns, calls to methods protected by entitlements from classes without
     * a valid policy will throw {@link org.elasticsearch.entitlement.runtime.api.NotEntitledException}.
     *
     * @param serverPolicyPatch            additional entitlements to patch the embedded server layer policy
     * @param pluginPolicies               maps each plugin name to the corresponding {@link Policy}
     * @param scopeResolver                a functor to map a Java Class to the component and module it belongs to.
     * @param settingResolver              a functor to resolve a setting name pattern for one or more Elasticsearch settings.
     * @param dataDirs                     data directories for Elasticsearch
     * @param sharedRepoDirs               shared repository directories for Elasticsearch
     * @param configDir                    the config directory for Elasticsearch
     * @param libDir                       the lib directory for Elasticsearch
     * @param modulesDir                   the directory where Elasticsearch modules are
     * @param pluginsDir                   the directory where plugins are installed for Elasticsearch
     * @param pluginSourcePaths            maps each plugin name to the location of that plugin's code
     * @param tempDir                      the temp directory for Elasticsearch
     * @param logsDir                      the log directory for Elasticsearch
     * @param pidFile                      path to a pid file for Elasticsearch, or {@code null} if one was not specified
     * @param suppressFailureLogPackages   packages for which we do not need or want to log Entitlements failures
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
        // Pre-condition: Ensure that the bootstrap process has not already been run.
        if (EntitlementInitialization.initializeArgs != null) {
            throw new IllegalStateException("initialization data is already set");
        }
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
        // Architectural Pattern: A static field is used to hold initialization arguments,
        // making them accessible to the agent code after it's loaded.
        EntitlementInitialization.initializeArgs = new EntitlementInitialization.InitializeArgs(
            serverPolicyPatch,
            pluginPolicies,
            scopeResolver,
            pathLookup,
            pluginSourcePaths,
            suppressFailureLogPackages
        );
        // Expose the initialization arguments to the agent's classloader module.
        exportInitializationToAgent();
        // Locate and dynamically load the agent into the running JVM.
        loadAgent(findAgentJar(), EntitlementInitialization.class.getName());
    }

    private static Path getUserHome() {
        String userHome = System.getProperty("user.home");
        if (userHome == null) {
            throw new IllegalStateException("user.home system property is required");
        }
        return PathUtils.get(userHome);
    }

    /**
     * Performs the dynamic attachment of the Java agent to the current JVM using the Attach API.
     * This is a privileged operation that allows modifying the running application's code.
     * @param agentPath The file path to the agent JAR.
     * @param entitlementInitializationClassName The fully qualified name of the class used for initialization.
     */
    @SuppressForbidden(reason = "The VirtualMachine API is the only way to attach a java agent dynamically")
    static void loadAgent(String agentPath, String entitlementInitializationClassName) {
        try {
            // Functional Utility: Attaches to the current JVM process.
            VirtualMachine vm = VirtualMachine.attach(Long.toString(ProcessHandle.current().pid()));
            try {
                // Triggers the agent's agentmain method and passes initialization arguments.
                vm.loadAgent(agentPath, entitlementInitializationClassName);
            } finally {
                vm.detach();
            }
        } catch (AttachNotSupportedException | IOException | AgentLoadException | AgentInitializationException e) {
            // If the agent cannot be attached, it's a fatal error for this feature.
            throw new IllegalStateException("Unable to attach entitlement agent", e);
        }
    }

    /**
     * Manages cross-module visibility in the Java Platform Module System (JPMS).
     * This method explicitly exports the package containing the initialization data
     * to the agent's module, which is the "unnamed module", allowing it to be accessed.
     */
    private static void exportInitializationToAgent() {
        String initPkg = EntitlementInitialization.class.getPackageName();
        // agent will live in unnamed module
        Module unnamedModule = ClassLoader.getSystemClassLoader().getUnnamedModule();
        // Inline: Makes the `initPkg` package from this class's module available to the unnamed module.
        EntitlementInitialization.class.getModule().addExports(initPkg, unnamedModule);
    }

    /**
     * Locates the entitlement agent JAR file.
     * It first checks for a specific system property (`es.entitlement.agentJar`) and falls back
     * to a conventional location (`lib/entitlement-agent`) within the Elasticsearch home directory.
     * @return The absolute path to the agent JAR file as a string.
     */
    static String findAgentJar() {
        String propertyName = "es.entitlement.agentJar";
        String propertyValue = System.getProperty(propertyName);
        if (propertyValue != null) {
            return propertyValue;
        }

        Path esHome = Path.of(System.getProperty("es.path.home"));
        Path dir = esHome.resolve("lib/entitlement-agent");
        // Pre-condition: The agent directory must exist.
        if (Files.exists(dir) == false) {
            throw new IllegalStateException("Directory for entitlement jar does not exist: " + dir);
        }
        try (var s = Files.list(dir)) {
            var candidates = s.limit(2).toList();
            // Invariant: Expects exactly one JAR file in the agent directory to avoid ambiguity.
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
