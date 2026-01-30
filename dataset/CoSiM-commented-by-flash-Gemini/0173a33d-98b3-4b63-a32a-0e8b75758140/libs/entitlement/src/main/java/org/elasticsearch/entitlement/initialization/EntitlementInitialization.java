/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

/**
 * @file EntitlementInitialization.java
 * @brief Serves as the primary entry point for the Elasticsearch entitlement Java agent, executed during the JVM's `agentmain` phase.
 *
 * This file orchestrates the setup and configuration of the entire entitlement system within Elasticsearch.
 * It dynamically loads and applies security policies, instruments methods to enforce those policies,
 * and ensures compatibility across various Java versions and platform-specific behaviors.
 */
package org.elasticsearch.entitlement.initialization;

import org.elasticsearch.core.Booleans;
import org.elasticsearch.core.PathUtils;
import org.elasticsearch.core.Strings;
import org.elasticsearch.core.internal.provider.ProviderLocator;
import org.elasticsearch.entitlement.bootstrap.EntitlementBootstrap;
import org.elasticsearch.entitlement.bridge.EntitlementChecker;
import org.elasticsearch.entitlement.instrumentation.CheckMethod;
import org.elasticsearch.entitlement.instrumentation.InstrumentationService;
import org.elasticsearch.entitlement.instrumentation.Instrumenter;
import org.elasticsearch.entitlement.instrumentation.MethodKey;
import org.elasticsearch.entitlement.instrumentation.Transformer;
import org.elasticsearch.entitlement.runtime.api.ElasticsearchEntitlementChecker;
import org.elasticsearch.entitlement.runtime.policy.FileAccessTree;
import org.elasticsearch.entitlement.runtime.policy.PathLookup;
import org.elasticsearch.entitlement.runtime.policy.Policy;
import org.elasticsearch.entitlement.runtime.policy.PolicyManager;
import org.elasticsearch.entitlement.runtime.policy.PolicyUtils;
import org.elasticsearch.entitlement.runtime.policy.Scope;
import org.elasticsearch.entitlement.runtime.policy.entitlements.CreateClassLoaderEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.Entitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.ExitVMEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.FilesEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.FilesEntitlement.FileData;
import org.elasticsearch.entitlement.runtime.policy.entitlements.InboundNetworkEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.LoadNativeLibrariesEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.ManageThreadsEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.OutboundNetworkEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.ReadStoreAttributesEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.SetHttpsConnectionPropertiesEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.WriteSystemPropertiesEntitlement;

import java.lang.instrument.Instrumentation;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.net.URI;
import java.nio.channels.spi.SelectorProvider;
import java.nio.file.AccessMode;
import java.nio.file.CopyOption;
import java.nio.file.DirectoryStream;
import java.nio.file.FileStore;
import java.nio.file.FileSystems;
import java.nio.file.LinkOption;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.WatchEvent;
import java.nio.file.WatchService;
import java.nio.file.attribute.FileAttribute;
import java.nio.file.spi.FileSystemProvider;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.CONFIG;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.DATA;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.LIB;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.LOGS;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.MODULES;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.PLUGINS;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.SHARED_REPO;
import static org.elasticsearch.entitlement.runtime.policy.Platform.LINUX;
import static org.elasticsearch.entitlement.runtime.policy.entitlements.FilesEntitlement.Mode.READ;
import static org.elasticsearch.entitlement.runtime.policy.entitlements.FilesEntitlement.Mode.READ_WRITE;

/**
 * @class EntitlementInitialization
 * @brief This class acts as the main entry point for the Elasticsearch entitlement
 *        Java agent, invoked during JVM {@code agentmain} phase.
 *
 * It is responsible for configuring the entire entitlement system:
 * <ol>
 * <li>Loading and configuring security policies ({@link PolicyManager}).</li>
 * <li>Instantiating a version-specific {@link EntitlementChecker}.</li>
 * <li>Making the checker available to the bootstrap library.</li>
 * <li>Installing a {@link Transformer} that uses an {@link Instrumenter} to
 *     inject bytecode instrumentation into specified methods.</li>
 * <li>Re-transforming already loaded classes to apply instrumentation retroactively.</li>
 * </ol>
 * This ensures that method calls protected by entitlements are intercepted and
 * validated against defined policies at runtime.
 */
public class EntitlementInitialization {

    /** @brief Package name for APM agents, used for specific entitlement handling. */
    private static final String AGENTS_PACKAGE_NAME = "co.elastic.apm.agent";
    /** @brief Reference to the module containing the core entitlement policies. */
    private static final Module ENTITLEMENTS_MODULE = PolicyManager.class.getModule();

    /** @brief Static instance of {@link ElasticsearchEntitlementChecker} that manages all entitlement checks. */
    private static ElasticsearchEntitlementChecker manager;

    /**
     * @interface InstrumentationInfoFactory
     * @brief A factory interface for creating {@link InstrumentationService.InstrumentationInfo} objects.
     * Functional Utility: Abstracts the creation of instrumentation information, particularly
     * for dynamically looked-up implementation methods.
     */
    interface InstrumentationInfoFactory {
        /**
         * @brief Creates an {@link InstrumentationService.InstrumentationInfo} for a given method.
         * @param methodName The name of the method to instrument.
         * @param parameterTypes An array of `Class` objects representing the parameter types of the method.
         * @return An `InstrumentationInfo` object containing details for instrumentation.
         * @throws ClassNotFoundException if a class specified in `parameterTypes` is not found.
         * @throws NoSuchMethodException if the specified method is not found in the target class.
         */
        InstrumentationService.InstrumentationInfo of(String methodName, Class<?>... parameterTypes) throws ClassNotFoundException,
            NoSuchMethodException;
    }

    /**
     * @brief Provides access to the initialized {@link EntitlementChecker} instance.
     * Functional Utility: This method is referenced by the bridge library reflectively
     * to obtain the active checker instance.
     * Precondition: `initChecker()` must have been successfully called, ensuring the `manager` field is initialized.
     * Postcondition: Returns the active `EntitlementChecker` instance.
     * @return The active `EntitlementChecker`.
     */
    // Note: referenced by bridge reflectively
    public static EntitlementChecker checker() {
        return manager;
    }

    /**
     * @brief Initializes the Entitlement system.
     *
     * Functional Utility: This is the main initialization method called by the Java agent.
     * It orchestrates the entire setup of the entitlement enforcement mechanism, including:
     * <ol>
     * <li>Finding and initializing a version-specific subclass of {@link EntitlementChecker}.</li>
     * <li>Building a comprehensive set of methods to instrument, including static lookups
     *     and dynamic resolution of platform-specific implementation methods (e.g., for {@link FileSystemProvider}).</li>
     * <li>Creating and registering an {@link Instrumenter} and {@link Transformer} to inject
     *     bytecode instrumentation at class loading time.</li>
     * <li>Re-transforming all already loaded classes that are targeted for instrumentation
     *     to ensure policies are applied to code already in memory.</li>
     * </ol>
     * @param inst The JVM {@link Instrumentation} class instance, provided by the Java agent framework.
     * Precondition: The `inst` parameter must be a valid `Instrumentation` instance provided by the JVM.
     * Postcondition: The entitlement system is fully set up, policies are loaded, and bytecode instrumentation is applied.
     * @throws Exception if any error occurs during initialization, class loading, or instrumentation.
     */
    public static void initialize(Instrumentation inst) throws Exception {
        // Step 1: Initialize the core EntitlementChecker manager.
        manager = initChecker();

        // Step 2: Determine the latest version-specific checker interface compatible with the current JVM.
        var latestCheckerInterface = getVersionSpecificCheckerClass(EntitlementChecker.class);
        // Step 3: Check if bytecode verification is enabled.
        var verifyBytecode = Booleans.parseBoolean(System.getProperty("es.entitlements.verify_bytecode", "false"));

        // Step 4: If verification is enabled, ensure sensitive classes are initialized to avoid circularity issues.
        /**
         * Block Logic: Conditional class initialization to prevent bytecode verification issues.
         * Invariant: Classes sensitive to bytecode verification should be initialized before re-transformation if verification is active.
         */
        if (verifyBytecode) {
            ensureClassesSensitiveToVerificationAreInitialized();
        }

        // Step 5: Build a map of methods to instrument.
        // It starts with methods directly looked up from the checker interface.
        Map<MethodKey, CheckMethod> checkMethods = new HashMap<>(INSTRUMENTATION_SERVICE.lookupMethods(latestCheckerInterface));
        
        // Step 6: Augment the set of methods to instrument dynamically, especially for platform-specific implementations.
        /**
         * Block Logic: Dynamically adds instrumentation targets for various file system related methods.
         * Functional Utility: Ensures comprehensive coverage of file system operations across different JDK implementations.
         */
        Stream.of(
            fileSystemProviderChecks(), // Add checks for FileSystemProvider methods.
            fileStoreChecks(),          // Add checks for FileStore methods.
            pathChecks(),               // Add checks for Path methods.
            Stream.of(
                // Add a specific check for SelectorProvider's inheritedChannel method.
                INSTRUMENTATION_SERVICE.lookupImplementationMethod(
                    SelectorProvider.class,
                    "inheritedChannel",
                    SelectorProvider.provider().getClass(),
                    EntitlementChecker.class,
                    "check" + Character.toUpperCase( "inheritedChannel".charAt(0)) + "inheritedChannel".substring(1), // Inline: dynamically generates the check method name.
                    // Renamed for clarity: checkSelectorProviderInheritedChannel
                    "checkSelectorProviderInheritedChannel"
                )
            )
        )
            .flatMap(Function.identity()) // Flatten the stream of streams.
            .forEach(instrumentation -> checkMethods.put(instrumentation.targetMethod(), instrumentation.checkMethod())); // Add to the map.

        // Step 7: Identify classes that need to be transformed based on the methods to instrument.
        var classesToTransform = checkMethods.keySet().stream().map(MethodKey::className).collect(Collectors.toSet());

        // Step 8: Create the Instrumenter and Transformer.
        Instrumenter instrumenter = INSTRUMENTATION_SERVICE.newInstrumenter(latestCheckerInterface, checkMethods);
        var transformer = new Transformer(instrumenter, classesToTransform, verifyBytecode);
        // Step 9: Add the transformer to the JVM instrumentation, enabling class file transformation.
        inst.addTransformer(transformer, true);

        // Step 10: Find and re-transform already loaded classes to apply instrumentation retroactively.
        var classesToRetransform = findClassesToRetransform(inst.getAllLoadedClasses(), classesToTransform);
        try {
            inst.retransformClasses(classesToRetransform);
        } catch (VerifyError e) {
            /**
             * Error Handling: If re-transformation fails with a `VerifyError`, the system attempts a more granular re-transformation.
             * Functional Utility: This block provides a fallback mechanism for bytecode verification issues, retrying class transformation individually.
             * Invariant: If a `VerifyError` occurs during bulk re-transformation, individual re-transformation attempts are made to isolate the problem.
             */
            transformer.enableClassVerification(); // Turn on verification for the transformer.

            for (var classToRetransform : classesToRetransform) {
                inst.retransformClasses(classToRetransform); // Retransform individually.
            }

            // If we somehow didn't catch the error in the loop, rethrow.
            throw e;
        }
    }

    /**
     * @brief Filters currently loaded classes to find those that need to be re-transformed.
     * @param loadedClasses An array of all classes currently loaded by the JVM.
     * @param classesToTransform A set of fully qualified class names that are targeted for instrumentation.
     * @return An array of `Class<?>` objects that need to be re-transformed.
     * Precondition: `loadedClasses` contains all currently loaded classes, and `classesToTransform` specifies target classes.
     * Postcondition: Returns an array containing only those loaded classes that match the `classesToTransform` set.
     */
    private static Class<?>[] findClassesToRetransform(Class<?>[] loadedClasses, Set<String> classesToTransform) {
        List<Class<?>> retransform = new ArrayList<>();
        /**
         * Block Logic: Iterates through all loaded classes to identify those requiring re-transformation.
         * Invariant: Only classes whose names (in "com/example/Class" format) exist in `classesToTransform` are added to the list.
         */
        for (Class<?> loadedClass : loadedClasses) {
            // Converts class name from "com.example.Class" to "com/example/Class" for comparison.
            if (classesToTransform.contains(loadedClass.getName().replace(".", "/"))) {
                retransform.add(loadedClass);
            }
        }
        return retransform.toArray(new Class<?>[0]); // Convert list to array.
    }

    /**
     * @brief Creates and configures the {@link PolicyManager} based on {@link EntitlementBootstrap.BootstrapArgs}.
     *
     * Functional Utility: This method is central to defining the security policies that
     * the entitlement system will enforce. It constructs a set of default server-level
     * scopes with various entitlements (e.g., file access, network access, thread management),
     * applies any provided server policy patches, and integrates plugin-specific policies.
     * It also includes platform-specific file access rules (e.g., for Linux OS files).
     *
     * @return A fully configured {@link PolicyManager} instance.
     * Precondition: `EntitlementBootstrap.bootstrapArgs()` must have been called and initialized.
     * Postcondition: A `PolicyManager` is returned, encapsulating all server-level and plugin-specific security policies.
     */
    private static PolicyManager createPolicyManager() {
        EntitlementBootstrap.BootstrapArgs bootstrapArgs = EntitlementBootstrap.bootstrapArgs();
        Map<String, Policy> pluginPolicies = bootstrapArgs.pluginPolicies();
        PathLookup pathLookup = bootstrapArgs.pathLookup();

        List<Scope> serverScopes = new ArrayList<>();
        List<FileData> serverModuleFileDatas = new ArrayList<>();
        
        /**
         * Block Logic: Defines default file access entitlements for core Elasticsearch directories and essential OS files.
         * These `FileData` objects specify read/write permissions for the server module based on base directories and platform.
         * Invariant: A comprehensive set of file access rules is established for fundamental system and application paths.
         */
        Collections.addAll(
            serverModuleFileDatas,
            // Base ES directories
            FileData.ofBaseDirPath(PLUGINS, READ),
            FileData.ofBaseDirPath(MODULES, READ),
            FileData.ofBaseDirPath(CONFIG, READ),
            FileData.ofBaseDirPath(LOGS, READ_WRITE),
            FileData.ofBaseDirPath(LIB, READ),
            FileData.ofBaseDirPath(DATA, READ_WRITE),
            FileData.ofBaseDirPath(SHARED_REPO, READ_WRITE),
            // exclusive settings file
            FileData.ofRelativePath(Path.of("operator/settings.json"), CONFIG, READ_WRITE).withExclusive(true),

            // OS release on Linux (specific file paths for Linux platforms).
            FileData.ofPath(Path.of("/etc/os-release"), READ).withPlatform(LINUX),
            FileData.ofPath(Path.of("/etc/system-release"), READ).withPlatform(LINUX),
            FileData.ofPath(Path.of("/usr/lib/os-release"), READ).withPlatform(LINUX),
            // read max virtual memory areas
            FileData.ofPath(Path.of("/proc/sys/vm/max_map_count"), READ).withPlatform(LINUX),
            FileData.ofPath(Path.of("/proc/meminfo"), READ).withPlatform(LINUX),
            // load averages on Linux
            FileData.ofPath(Path.of("/proc/loadavg"), READ).withPlatform(LINUX),
            // control group stats on Linux. cgroup v2 stats are in an unpredicable
            // location under `/sys/fs/cgroup`, so unfortunately we have to allow
            // read access to the entire directory hierarchy.
            FileData.ofPath(Path.of("/proc/self/cgroup"), READ).withPlatform(LINUX),
            FileData.ofPath(Path.of("/sys/fs/cgroup/"), READ).withPlatform(LINUX),
            // // io stats on Linux
            FileData.ofPath(Path.of("/proc/self/mountinfo"), READ).withPlatform(LINUX),
            FileData.ofPath(Path.of("/proc/diskstats"), READ).withPlatform(LINUX)
        );
        /**
         * Block Logic: Conditionally adds file access for the PID file.
         * Invariant: PID file access is only granted if a PID file path has been configured.
         */
        if (pathLookup.pidFile() != null) {
            serverModuleFileDatas.add(FileData.ofPath(pathLookup.pidFile(), READ_WRITE));
        }

        /**
         * Block Logic: Defines various security scopes with their associated entitlements for different Elasticsearch components.
         * Functional Utility: Each scope represents a logical module or component and lists the specific
         * permissions (entitlements) it requires to operate correctly.
         * Invariant: This block comprehensively sets up granular permissions for different parts of the Elasticsearch system.
         */
        Collections.addAll(
            serverScopes,
            new Scope(
                "org.elasticsearch.base",
                List.of(
                    new CreateClassLoaderEntitlement(), // Permission to create class loaders.
                    new FilesEntitlement(                // File access permissions.
                        List.of(
                            // TODO: what in es.base is accessing shared repo?
                            FileData.ofBaseDirPath(SHARED_REPO, READ_WRITE),
                            FileData.ofBaseDirPath(DATA, READ_WRITE)
                        )
                    )
                )
            ),
            new Scope("org.elasticsearch.xcontent", List.of(new CreateClassLoaderEntitlement())),
            new Scope(
                "org.elasticsearch.server",
                List.of(
                    new ExitVMEntitlement(),              // Permission to exit the JVM.
                    new ReadStoreAttributesEntitlement(), // Permission to read file store attributes.
                    new CreateClassLoaderEntitlement(),   // Permission to create class loaders.
                    new InboundNetworkEntitlement(),      // Permission for inbound network connections.
                    new LoadNativeLibrariesEntitlement(), // Permission to load native libraries.
                    new ManageThreadsEntitlement(),       // Permission to manage threads.
                    new FilesEntitlement(serverModuleFileDatas) // File access based on previously defined rules.
                )
            ),
            new Scope("java.desktop", List.of(new LoadNativeLibrariesEntitlement())),
            new Scope("org.apache.httpcomponents.httpclient", List.of(new OutboundNetworkEntitlement())),
            new Scope(
                "org.apache.lucene.core",
                List.of(
                    new LoadNativeLibrariesEntitlement(),
                    new ManageThreadsEntitlement(),
                    new FilesEntitlement(List.of(FileData.ofBaseDirPath(CONFIG, READ), FileData.ofBaseDirPath(DATA, READ_WRITE)))
                )
            ),
            new Scope("org.apache.lucene.misc", List.of(FileData.ofBaseDirPath(DATA, READ_WRITE)).stream().collect(Collectors.toList())), // Functional Utility: Collects file data for Lucene misc.
            new Scope(
                "org.apache.logging.log4j.core",
                List.of(new ManageThreadsEntitlement(), new FilesEntitlement(List.of(FileData.ofBaseDirPath(LOGS, READ_WRITE))))
            ),
            new Scope(
                "org.elasticsearch.nativeaccess",
                List.of(new LoadNativeLibrariesEntitlement(), new FilesEntitlement(List.of(FileData.ofBaseDirPath(DATA, READ_WRITE))))
            )
        );

        /**
         * Block Logic: Conditionally adds FIPS-related entitlements if FIPS-only mode is enforced.
         * Functional Utility: Integrates specific security permissions required when running in a FIPS-compliant environment,
         *                     including access to the trust store and management of threads and network for BouncyCastle FIPS modules.
         * Invariant: FIPS entitlements are only enabled if the `org.bouncycastle.fips.approved_only` system property is true.
         */
        if (Booleans.parseBoolean(System.getProperty("org.bouncycastle.fips.approved_only"), false)) {
            // Determine the trust store path, either custom or default JDK.
            String trustStore = System.getProperty("javax.net.ssl.trustStore");
            Path trustStorePath = trustStore != null
                ? Path.of(trustStore)
                : Path.of(System.getProperty("java.home")).resolve("lib/security/jssecacerts");

            Collections.addAll(
                serverScopes,
                new Scope(
                    "org.bouncycastle.fips.tls",
                    List.of(
                        new FilesEntitlement(List.of(FileData.ofPath(trustStorePath, READ))),
                        new ManageThreadsEntitlement(),
                        new OutboundNetworkEntitlement()
                    )
                ),
                new Scope(
                    "org.bouncycastle.fips.core",
                    // read to lib dir is required for checksum validation
                    List.of(new FilesEntitlement(List.of(FileData.ofBaseDirPath(LIB, READ))), new ManageThreadsEntitlement())
                )
            );
        }

        // Create the server policy, merging with any provided patch policy.
        var serverPolicy = new Policy(
            "server",
            bootstrapArgs.serverPolicyPatch() == null
                ? serverScopes
                : PolicyUtils.mergeScopes(serverScopes, bootstrapArgs.serverPolicyPatch().scopes())
        );

        /**
         * Block Logic: Defines specific entitlements required by the APM agent due to its dynamic nature and module system interaction.
         * Functional Utility: Provides necessary permissions for APM agents, which often operate outside standard module boundaries,
         *                     to perform tasks like class loading, thread management, and network operations.
         * Invariant: These entitlements are a temporary hack and should be re-evaluated for future module system improvements.
         */
        // agents run without a module, so this is a special hack for the apm agent
        // this should be removed once https://github.com/elastic/elasticsearch/issues/109335 is completed
        // See also modules/apm/src/main/plugin-metadata/entitlement-policy.yaml
        List<Entitlement> agentEntitlements = List.of(
            new CreateClassLoaderEntitlement(),
            new ManageThreadsEntitlement(),
            new SetHttpsConnectionPropertiesEntitlement(),
            new OutboundNetworkEntitlement(),
            new WriteSystemPropertiesEntitlement(Set.of("AsyncProfiler.safemode")),
            new LoadNativeLibrariesEntitlement(),
            new FilesEntitlement(
                List.of(
                    FileData.ofBaseDirPath(LOGS, READ_WRITE),
                    FileData.ofPath(Path.of("/proc/meminfo"), READ).withPlatform(LINUX),
                    FileData.ofPath(Path.of("/sys/fs/cgroup/"), READ).withPlatform(LINUX) // Explicitly specify LINUX
                )
            )
        );

        // Validate file entitlements for all plugins against forbidden paths.
        validateFilesEntitlements(pluginPolicies, pathLookup);

        // Return a new PolicyManager instance with all configured policies and resolvers.
        return new PolicyManager(
            serverPolicy,
            agentEntitlements,
            pluginPolicies,
            EntitlementBootstrap.bootstrapArgs().pluginResolver(),
            EntitlementBootstrap.bootstrapArgs().sourcePaths(),
            AGENTS_PACKAGE_NAME,
            ENTITLEMENTS_MODULE,
            pathLookup,
            bootstrapArgs.suppressFailureLogClasses()
        );
    }

    /**
     * @brief Validates file entitlements for all plugins against forbidden paths.
     *
     * Functional Utility: This method ensures that plugin policies do not inadvertently grant
     * access to sensitive directories (e.g., other plugin directories, module directories)
     * which could lead to security vulnerabilities.
     *
     * @param pluginPolicies A map of plugin policies.
     * @param pathLookup A {@link PathLookup} instance for resolving base directories.
     * Precondition: `pluginPolicies` contains all defined plugin security policies, and `pathLookup` is properly initialized for directory resolution.
     * Postcondition: All plugin file access entitlements are verified against a set of forbidden paths.
     * @throws IllegalArgumentException if a plugin's policy grants forbidden file access.
     */
    // package visible for tests
    static void validateFilesEntitlements(Map<String, Policy> pluginPolicies, PathLookup pathLookup) {
        // Build sets of forbidden paths for read and write access.
        Set<Path> readAccessForbidden = new HashSet<>();
        pathLookup.getBaseDirPaths(PLUGINS).forEach(p -> readAccessForbidden.add(p.toAbsolutePath().normalize()));
        pathLookup.getBaseDirPaths(MODULES).forEach(p -> readAccessForbidden.add(p.toAbsolutePath().normalize()));
        pathLookup.getBaseDirPaths(LIB).forEach(p -> readAccessForbidden.add(p.toAbsolutePath().normalize()));
        Set<Path> writeAccessForbidden = new HashSet<>();
        pathLookup.getBaseDirPaths(CONFIG).forEach(p -> writeAccessForbidden.add(p.toAbsolutePath().normalize()));
        
        /**
         * Block Logic: Iterates through each plugin's policy to validate its file access entitlements.
         * Invariant: No plugin is allowed to have read or write access to paths designated as forbidden.
         */
        for (var pluginPolicy : pluginPolicies.entrySet()) {
            for (var scope : pluginPolicy.getValue().scopes()) {
                var filesEntitlement = scope.entitlements()
                    .stream()
                    .filter(x -> x instanceof FilesEntitlement)
                    .map(x -> ((FilesEntitlement) x))
                    .findFirst();
                
                // If a FilesEntitlement is present in the scope, validate it.
                /**
                 * Block Logic: Validates file access permissions if a `FilesEntitlement` is present in the current scope.
                 * Invariant: If a `FilesEntitlement` exists, its read and write permissions must adhere to the forbidden path rules.
                 */
                if (filesEntitlement.isPresent()) {
                    // Create a FileAccessTree for the entitlement.
                    var fileAccessTree = FileAccessTree.withoutExclusivePaths(filesEntitlement.get(), pathLookup, null);
                    // Validate read and write access.
                    validateReadFilesEntitlements(pluginPolicy.getKey(), scope.moduleName(), fileAccessTree, readAccessForbidden);
                    validateWriteFilesEntitlements(pluginPolicy.getKey(), scope.moduleName(), fileAccessTree, writeAccessForbidden);
                }
            }
        }
    }

    /**
     * @brief Helper method to construct an `IllegalArgumentException` for file entitlement validation failures.
     * @param componentName The name of the component (e.g., plugin name).
     * @param moduleName The name of the module within the component.
     * @param forbiddenPath The path that was attempted to be accessed in a forbidden manner.
     * @param mode The access mode (READ or READ_WRITE) that was forbidden.
     * @return An `IllegalArgumentException` with a descriptive error message.
     * Precondition: A file access violation has been detected.
     * Postcondition: Returns a well-formatted exception detailing the violation.
     */
    private static IllegalArgumentException buildValidationException(
        String componentName,
        String moduleName,
        Path forbiddenPath,
        FilesEntitlement.Mode mode
    ) {
        return new IllegalArgumentException(
            Strings.format(
                "policy for module [%s] in [%s] has an invalid file entitlement. Any path under [%s] is forbidden for mode [%s].",
                moduleName,
                componentName,
                forbiddenPath,
                mode
            )
        );
    }

    /**
     * @brief Validates read access file entitlements for a given component and module.
     * @param componentName The name of the component (e.g., plugin name).
     * @param moduleName The name of the module.
     * @param fileAccessTree The {@link FileAccessTree} representing allowed file access.
     * @param readForbiddenPaths A set of paths that are forbidden for read access.
     * Precondition: `fileAccessTree` accurately reflects the allowed file access for the component, and `readForbiddenPaths` defines restricted locations.
     * Postcondition: Throws `IllegalArgumentException` if any read access is granted to a forbidden path.
     * @throws IllegalArgumentException if read access is granted to a forbidden path.
     */
    private static void validateReadFilesEntitlements(
        String componentName,
        String moduleName,
        FileAccessTree fileAccessTree,
        Set<Path> readForbiddenPaths
    ) {
        /**
         * Block Logic: Iterates through each forbidden path and checks if the `fileAccessTree` allows read access to it.
         * Invariant: Read access is strictly prohibited for any path present in `readForbiddenPaths`.
         */
        for (Path forbiddenPath : readForbiddenPaths) {
            if (fileAccessTree.canRead(forbiddenPath)) {
                throw buildValidationException(componentName, moduleName, forbiddenPath, READ);
            }
        }
    }

    /**
     * @brief Validates write access file entitlements for a given component and module.
     * @param componentName The name of the component (e.g., plugin name).
     * @param moduleName The name of the module.
     * @param fileAccessTree The {@link FileAccessTree} representing allowed file access.
     * @param writeForbiddenPaths A set of paths that are forbidden for write access.
     * Precondition: `fileAccessTree` accurately reflects the allowed file access for the component, and `writeForbiddenPaths` defines restricted locations.
     * Postcondition: Throws `IllegalArgumentException` if any write access is granted to a forbidden path.
     * @throws IllegalArgumentException if write access is granted to a forbidden path.
     */
    private static void validateWriteFilesEntitlements(
        String componentName,
        String moduleName,
        FileAccessTree fileAccessTree,
        Set<Path> writeForbiddenPaths
    ) {
        /**
         * Block Logic: Iterates through each forbidden path and checks if the `fileAccessTree` allows write access to it.
         * Invariant: Write access is strictly prohibited for any path present in `writeForbiddenPaths`.
         */
        for (Path forbiddenPath : writeForbiddenPaths) {
            if (fileAccessTree.canWrite(forbiddenPath)) {
                throw buildValidationException(componentName, moduleName, forbiddenPath, READ_WRITE);
            }
        }
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
     * @brief Dynamically identifies and prepares {@link InstrumentationService.InstrumentationInfo}
     *        for {@link FileSystemProvider} methods based on the concrete implementation
     *        used by the JVM.
     *
     * Functional Utility: This method is crucial because the JDK exposes some APIs through
     * interfaces (like `FileSystemProvider`) that have different internal implementations
     * depending on the JVM host platform (e.g., `UnixFileSystemProvider` or
     * `WindowsFileSystemProvider`). Since interfaces cannot be directly instrumented,
     * this dynamically finds the concrete class and prepares to instrument its methods.
     *
     * @return A {@link Stream} of `InstrumentationInfo` objects for `FileSystemProvider` methods.
     * @throws ClassNotFoundException if required classes are not found during reflective lookup.
     * @throws NoSuchMethodException if a method for instrumentation is not found during reflective lookup.
     */
    private static Stream<InstrumentationService.InstrumentationInfo> fileSystemProviderChecks() throws ClassNotFoundException,
        NoSuchMethodException {
        // Get the concrete class of the default FileSystemProvider.
        var fileSystemProviderClass = FileSystems.getDefault().provider().getClass();

        // Create an InstrumentationInfoFactory tailored for FileSystemProvider methods.
        var instrumentation = new InstrumentationInfoFactory() {
            @Override
            public InstrumentationService.InstrumentationInfo of(String methodName, Class<?>... parameterTypes)
                throws ClassNotFoundException, NoSuchMethodException {
                // Look up the implementation method on the concrete FileSystemProvider class.
                return INSTRUMENTATION_SERVICE.lookupImplementationMethod(
                    FileSystemProvider.class,      // Interface being implemented.
                    methodName,                    // Method name.
                    fileSystemProviderClass,       // Concrete implementation class.
                    EntitlementChecker.class,      // Checker class.
                    "check" + Character.toUpperCase(methodName.charAt(0)) + methodName.substring(1), // Corresponding check method name.
                    parameterTypes                 // Method parameter types.
                );
            }
        };

        // Define a stream of `InstrumentationInfo` for various FileSystemProvider methods.
        return Stream.of(
            instrumentation.of("newFileSystem", URI.class, Map.class),
            instrumentation.of("newFileSystem", Path.class, Map.class),
            instrumentation.of("newInputStream", Path.class, OpenOption[].class),
            instrumentation.of("newOutputStream", Path.class, OpenOption[].class),
            instrumentation.of("newFileChannel", Path.class, Set.class, FileAttribute[].class),
            instrumentation.of("newAsynchronousFileChannel", Path.class, Set.class, ExecutorService.class, FileAttribute[].class),
            instrumentation.of("newByteChannel", Path.class, Set.class, FileAttribute[].class),
            instrumentation.of("newDirectoryStream", Path.class, DirectoryStream.Filter.class),
            instrumentation.of("createDirectory", Path.class, FileAttribute[].class),
            instrumentation.of("createSymbolicLink", Path.class, Path.class, FileAttribute[].class),
            instrumentation.of("createLink", Path.class, Path.class),
            instrumentation.of("delete", Path.class),
            instrumentation.of("deleteIfExists", Path.class),
            instrumentation.of("readSymbolicLink", Path.class),
            instrumentation.of("copy", Path.class, Path.class, CopyOption[].class),
            instrumentation.of("move", Path.class, Path.class, CopyOption[].class),
            instrumentation.of("isSameFile", Path.class, Path.class),
            instrumentation.of("isHidden", Path.class),
            instrumentation.of("getFileStore", Path.class),
            instrumentation.of("checkAccess", Path.class, AccessMode[].class),
            instrumentation.of("getFileAttributeView", Path.class, Class.class, LinkOption[].class),
            instrumentation.of("readAttributes", Path.class, Class.class, LinkOption[].class),
            instrumentation.of("readAttributes", Path.class, String.class, LinkOption[].class),
            instrumentation.of("readAttributesIfExists", Path.class, Class.class, LinkOption[].class),
            instrumentation.of("setAttribute", Path.class, String.class, Object.class, LinkOption[].class),
            instrumentation.of("exists", Path.class, LinkOption[].class)
        );
    }

    /**
     * @brief Dynamically identifies and prepares {@link InstrumentationService.InstrumentationInfo}
     *        for {@link FileStore} methods.
     *
     * Functional Utility: Scans available `FileStore` implementations at runtime to
     * determine their concrete classes and prepares methods from these classes for instrumentation.
     * This addresses the challenge of instrumenting methods exposed via interfaces
     * that have various concrete implementations across different platforms or JVMs.
     *
     * @return A {@link Stream} of `InstrumentationInfo` objects for `FileStore` methods.
     * @throws RuntimeException if an underlying `NoSuchMethodException` or `ClassNotFoundException` occurs
     *         during reflective lookup, indicating an unexpected issue in the JVM environment.
     */
    private static Stream<InstrumentationService.InstrumentationInfo> fileStoreChecks() {
        // Get distinct concrete `FileStore` classes from the default file system.
        var fileStoreClasses = StreamSupport.stream(FileSystems.getDefault().getFileStores().spliterator(), false)
            .map(FileStore::getClass)
            .distinct();
        
        /**
         * Block Logic: For each unique `FileStore` class, it prepares methods for instrumentation.
         * Invariant: All relevant `FileStore` methods from distinct implementations are targeted for entitlement checks.
         */
        return fileStoreClasses.flatMap(fileStoreClass -> {
            var instrumentation = new InstrumentationInfoFactory() {
                @Override
                public InstrumentationService.InstrumentationInfo of(String methodName, Class<?>... parameterTypes)
                    throws ClassNotFoundException, NoSuchMethodException {
                    return INSTRUMENTATION_SERVICE.lookupImplementationMethod(
                        FileStore.class,
                        methodName,
                        fileStoreClass,
                        EntitlementChecker.class,
                        "check" + Character.toUpperCase(methodName.charAt(0)) + methodName.substring(1),
                        parameterTypes
                    );
                }
            };

            try {
                return Stream.of(
                    instrumentation.of("getFileStoreAttributeView", Class.class),
                    instrumentation.of("getAttribute", String.class),
                    instrumentation.of("getBlockSize"),
                    instrumentation.of("getTotalSpace"),
                    instrumentation.of("getUnallocatedSpace"),
                    instrumentation.of("getUsableSpace"),
                    instrumentation.of("isReadOnly"),
                    instrumentation.of("name"),
                    instrumentation.of("type")

                );
            } catch (NoSuchMethodException | ClassNotFoundException e) {
                // Error Handling: Wrap reflection-related exceptions in a RuntimeException.
                throw new RuntimeException(e);
            }
        });
    }

    /**
     * @brief Dynamically identifies and prepares {@link InstrumentationService.InstrumentationInfo}
     *        for {@link Path} methods.
     *
     * Functional Utility: Similar to `fileSystemProviderChecks`, this method determines
     * the concrete `Path` implementation classes used by the JVM and prepares specific
     * methods from these classes for bytecode instrumentation. This ensures that
     * operations performed via `Path` objects are subject to entitlement checks.
     *
     * @return A {@link Stream} of `InstrumentationInfo` objects for `Path` methods.
     * @throws RuntimeException if an underlying `NoSuchMethodException` or `ClassNotFoundException` occurs
     *         during reflective lookup, indicating an unexpected issue in the JVM environment.
     */
    private static Stream<InstrumentationService.InstrumentationInfo> pathChecks() {
        // Get distinct concrete `Path` classes from the default file system's root directories.
        var pathClasses = StreamSupport.stream(FileSystems.getDefault().getRootDirectories().spliterator(), false)
            .map(Path::getClass)
            .distinct();
        
        /**
         * Block Logic: For each unique `Path` class, it prepares methods for instrumentation.
         * Invariant: All relevant `Path` methods from distinct implementations are targeted for entitlement checks.
         */
        return pathClasses.flatMap(pathClass -> {
            InstrumentationInfoFactory instrumentation = (String methodName, Class<?>... parameterTypes) -> INSTRUMENTATION_SERVICE
                .lookupImplementationMethod(
                    Path.class,
                    methodName,
                    pathClass,
                    EntitlementChecker.class,
                    "checkPath" + Character.toUpperCase(methodName.charAt(0)) + methodName.substring(1),
                    parameterTypes
                );

            try {
                return Stream.of(
                    instrumentation.of("toRealPath", LinkOption[].class),
                    instrumentation.of("register", WatchService.class, WatchEvent.Kind[].class),
                    instrumentation.of("register", WatchService.class, WatchEvent.Kind[].class, WatchEvent.Modifier[].class)
                );
            } catch (NoSuchMethodException | ClassNotFoundException e) {
                // Error Handling: Wrap reflection-related exceptions in a RuntimeException.
                throw new RuntimeException(e);
            }
        });
    }

    /**
     * @brief Forces the initialization of certain classes that are sensitive to bytecode
     *        transformation and verification.
     *
     * Functional Utility: If bytecode verification is enabled, the order of class
     * transformation and verification matters. This method ensures that these classes
     * are loaded and initialized *before* they are potentially re-transformed,
     * thereby avoiding complex circularity errors that can arise during verification
     * of transformed bytecode.
     * Precondition: Called when bytecode verification is enabled.
     * Postcondition: Specified sensitive classes are initialized, reducing the risk of `VerifyError` during re-transformation.
     */
    private static void ensureClassesSensitiveToVerificationAreInitialized() {
        // A set of fully qualified class names that require pre-initialization.
        var classesToInitialize = Set.of("sun.net.www.protocol.http.HttpURLConnection");
        /**
         * Block Logic: Iterates through a predefined list of class names and forces their loading and initialization.
         * Invariant: Each class in `classesToInitialize` is explicitly loaded before any potential re-transformation.
         */
        for (String className : classesToInitialize) {
            try {
                Class.forName(className); // Force class loading and initialization.
            } catch (ClassNotFoundException unexpected) {
                // Error Handling: This should not happen if the class name is correct and available.
                throw new AssertionError(unexpected);
            }
        }
    }

    /**
     * @brief Returns the "most recent" {@link EntitlementChecker} class compatible
     *        with the current runtime Java version.
     *
     * Functional Utility: This method supports version-specific `EntitlementChecker`
     * implementations. It dynamically selects the appropriate checker class
     * (e.g., `Java23EntitlementChecker` for Java 23+) based on the JVM's runtime version.
     * This allows the entitlement system to leverage new Java features or APIs
     * while maintaining compatibility with older JVM versions.
     *
     * @param baseClass The base {@link EntitlementChecker} interface class (e.g., `EntitlementChecker.class`).
     * @return A `Class<?>` object representing the version-specific `EntitlementChecker` implementation.
     * Precondition: `baseClass` is a valid base interface for entitlement checking.
     * Postcondition: Returns the dynamically determined, version-compatible `EntitlementChecker` class.
     * @throws AssertionError if the required version-specific class cannot be found.
     */
    private static Class<?> getVersionSpecificCheckerClass(Class<?> baseClass) {
        String packageName = baseClass.getPackageName();
        String baseClassName = baseClass.getSimpleName();
        int javaVersion = Runtime.version().feature();

        final String classNamePrefix;
        /**
         * Block Logic: Determines the appropriate `EntitlementChecker` class name prefix based on the current Java version.
         * Invariant: Java versions 23 and above use a specific checker, while others default to no prefix.
         */
        if (javaVersion >= 23) {
            // All Java versions from 23 onwards will use checks in the Java23EntitlementChecker.
            classNamePrefix = "Java23";
        } else {
            // For any other Java version, the basic EntitlementChecker is used.
            classNamePrefix = "";
        }
        final String className = packageName + "." + classNamePrefix + baseClassName;
        Class<?> clazz;
        try {
            clazz = Class.forName(className); // Attempt to load the dynamically constructed class name.
        } catch (ClassNotFoundException e) {
            // Error Handling: If the version-specific class is not found, it indicates a configuration error.
            throw new AssertionError("entitlement lib cannot find entitlement class " + className, e);
        }
        return clazz;
    }

    /**
     * @brief Initializes and instantiates the version-specific {@link ElasticsearchEntitlementChecker}.
     *
     * Functional Utility: This method creates the {@link PolicyManager} and then
     * uses reflection to instantiate the correct `ElasticsearchEntitlementChecker`
     * implementation based on the current Java version, passing the `PolicyManager`
     * to its constructor.
     *
     * @return An instantiated {@link ElasticsearchEntitlementChecker}.
     * Precondition: All necessary `EntitlementBootstrap.BootstrapArgs` are initialized.
     * Postcondition: A version-specific `ElasticsearchEntitlementChecker` is instantiated and ready for use.
     * @throws AssertionError if the required constructor is missing or instantiation fails.
     */
    private static ElasticsearchEntitlementChecker initChecker() {
        final PolicyManager policyManager = createPolicyManager(); // Create the PolicyManager.

        final Class<?> clazz = getVersionSpecificCheckerClass(ElasticsearchEntitlementChecker.class); // Get version-specific class.

        Constructor<?> constructor;
        try {
            // Look up the constructor that takes a `PolicyManager`.
            constructor = clazz.getConstructor(PolicyManager.class);
        } catch (NoSuchMethodException e) {
            // Error Handling: If the constructor is not found, it's a critical implementation error.
            throw new AssertionError("entitlement impl is missing no arg constructor", e);
        }
        try {
            // Instantiate the checker using the found constructor and the policy manager.
            return (ElasticsearchEntitlementChecker) constructor.newInstance(policyManager);
        } catch (IllegalAccessException | InvocationTargetException | InstantiationException e) {
            // Error Handling: Catch various reflection-related instantiation errors.
            throw new AssertionError(e);
        }
    }

    /**
     * @brief Static instance of {@link InstrumentationService} for managing bytecode instrumentation.
     * Functional Utility: This service provides the core functionality for looking up methods
     * for instrumentation, creating instrumenters, and managing bytecode transformation.
     * Invariant: This service is initialized once as a singleton for the entitlement system.
     */
    private static final InstrumentationService INSTRUMENTATION_SERVICE = new ProviderLocator<>(
        "entitlement",
        InstrumentationService.class,
        "org.elasticsearch.entitlement.instrumentation",
        Set.of()
    ).get();
}