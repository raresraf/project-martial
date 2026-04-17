/**
 * @file EntitlementBootstrap.java
 * @brief Dynamic bootstrap mechanism for Elasticsearch's entitlement enforcement agent.
 * This class orchestrates the runtime activation of security and licensing checks 
 * by dynamically attaching a Java agent to the current JVM process. It manages 
 * complex dependency injection of security policies, path lookups, and module 
 * exports to ensure that the entitlement runtime has full visibility and 
 * enforcement capabilities over protected components and plugins.
 *
 * Domain: Enterprise Security, Java Instrumentation, Dynamic Agent Loading.
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

public class EntitlementBootstrap {

    /**
     * Primary activation sequence for the entitlement enforcement subsystem.
     * Logic: Aggregates system paths and policies into an initialization context, 
     * exports required modules, and triggers the dynamic loading of the agent JAR.
     * Once completed, any unauthorized access to protected methods will trigger 
     * entitlement exceptions.
     *
     * @param serverPolicyPatch            additional entitlements for the core server layer.
     * @param pluginPolicies               mapping of plugin identifies to their respective security policies.
     * @param scopeResolver                functor to resolve class-to-module architectural mappings.
     * @param settingResolver              functor for resolving Elasticsearch configuration patterns.
     * @param dataDirs                     Elasticsearch data storage paths.
     * @param sharedRepoDirs               Paths for shared repositories (backups, etc).
     * @param configDir                    Directory containing system configuration files.
     * @param libDir                       Directory for core library JARs.
     * @param modulesDir                   Path to Elasticsearch internal modules.
     * @param pluginsDir                   Root directory for third-party or optional plugins.
     * @param pluginSourcePaths            mapping of plugins to their physical source locations for auditing.
     * @param logsDir                      Directory for operational log storage.
     * @param tempDir                      Path for transient file storage.
     * @param pidFile                      Optional path to the process ID file.
     * @param suppressFailureLogPackages   Package whitelist for filtering entitlement failure noise.
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
        
        // Invariant: Entitlement agent can only be initialized once per process lifecycle.
        if (EntitlementInitialization.initializeArgs != null) {
            throw new IllegalStateException("initialization data is already set");
        }

        // Functional Utility: Encapsulates filesystem visibility rules for the enforcement agent.
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

        // State Capture: prepares the cross-module arguments for the incoming agent.
        EntitlementInitialization.initializeArgs = new EntitlementInitialization.InitializeArgs(
            serverPolicyPatch,
            pluginPolicies,
            scopeResolver,
            pathLookup,
            pluginSourcePaths,
            suppressFailureLogPackages
        );

        // Security Protocol: ensures the agent can access initialization data across module boundaries.
        exportInitializationToAgent();

        // Execution: dynamically injects the agent into the running JVM.
        loadAgent(findAgentJar(), EntitlementInitialization.class.getName());
    }

    /**
     * Retrieves the current user's home directory.
     * @return Path representing the user home.
     * @throws IllegalStateException if the 'user.home' property is missing.
     */
    private static Path getUserHome() {
        String userHome = System.getProperty("user.home");
        if (userHome == null) {
            throw new IllegalStateException("user.home system property is required");
        }
        return PathUtils.get(userHome);
    }

    /**
     * Core dynamic instrumentation logic using the Java Attach API.
     * Logic: Connects to the current process via its PID and commands the JVM 
     * to load the specified agent JAR.
     * @param agentPath Filesystem path to the agent JAR.
     * @param entitlementInitializationClassName Fully qualified name of the agent entry class.
     */
    @SuppressForbidden(reason = "The VirtualMachine API is the only way to attach a java agent dynamically")
    static void loadAgent(String agentPath, String entitlementInitializationClassName) {
        try {
            // Attach to self: enables dynamic bytecode manipulation and enforcement.
            VirtualMachine vm = VirtualMachine.attach(Long.toString(ProcessHandle.current().pid()));
            try {
                vm.loadAgent(agentPath, entitlementInitializationClassName);
            } finally {
                // Ensure the attachment is closed after the agent is loaded.
                vm.detach();
            }
        } catch (AttachNotSupportedException | IOException | AgentLoadException | AgentInitializationException e) {
            throw new IllegalStateException("Unable to attach entitlement agent", e);
        }
    }

    /**
     * Configures JPMS module exports for cross-layer communication.
     * Logic: Explicitly exports the initialization package to the unnamed module 
     * where the agent typically resides.
     */
    private static void exportInitializationToAgent() {
        String initPkg = EntitlementInitialization.class.getPackageName();
        // Target: System class loader's unnamed module (default agent location).
        Module unnamedModule = ClassLoader.getSystemClassLoader().getUnnamedModule();
        EntitlementInitialization.class.getModule().addExports(initPkg, unnamedModule);
    }

    /**
     * Resolves the location of the entitlement agent JAR.
     * Logic: Prioritizes system properties, falling back to the standard Elasticsearch 
     * installation layout (lib/entitlement-agent).
     * @return String path to the validated agent JAR.
     */
    static String findAgentJar() {
        String propertyName = "es.entitlement.agentJar";
        String propertyValue = System.getProperty(propertyName);
        if (propertyValue != null) {
            return propertyValue;
        }

        Path esHome = Path.of(System.getProperty("es.path.home"));
        Path dir = esHome.resolve("lib/entitlement-agent");
        
        // Block Logic: Filesystem validation.
        if (Files.exists(dir) == false) {
            throw new IllegalStateException("Directory for entitlement jar does not exist: " + dir);
        }
        try (var s = Files.list(dir)) {
            // Invariant: Expects exactly one JAR in the entitlement-agent directory.
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
