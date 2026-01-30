/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

/**
 * @file PolicyManager.java
 * @brief This file contains the core logic for managing and enforcing security policies within Elasticsearch's entitlement system.
 *
 * It defines how access requests are evaluated against predefined policies for various components
 * (system, server, plugins, agents) and handles the intricate logic of resolving class origins
 * and applying granular entitlements to ensure secure operation.
 */
package org.elasticsearch.entitlement.runtime.policy;

import org.elasticsearch.core.PathUtils;
import org.elasticsearch.core.Strings;
import org.elasticsearch.core.SuppressForbidden;
import org.elasticsearch.entitlement.instrumentation.InstrumentationService;
import org.elasticsearch.entitlement.runtime.api.NotEntitledException;
import org.elasticsearch.entitlement.runtime.policy.FileAccessTree.ExclusiveFileEntitlement;
import org.elasticsearch.entitlement.runtime.policy.FileAccessTree.ExclusivePath;
import org.elasticsearch.entitlement.runtime.policy.entitlements.CreateClassLoaderEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.Entitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.ExitVMEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.FilesEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.InboundNetworkEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.LoadNativeLibrariesEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.ManageThreadsEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.OutboundNetworkEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.ReadStoreAttributesEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.SetHttpsConnectionPropertiesEntitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.WriteSystemPropertiesEntitlement;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;

import java.io.File;
import java.io.IOException;
import java.lang.StackWalker.StackFrame;
import java.lang.module.ModuleFinder;
import java.lang.module.ModuleReference;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.lang.StackWalker.Option.RETAIN_CLASS_REFERENCE;
import static java.util.Objects.requireNonNull;
import static java.util.function.Predicate.not;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.toUnmodifiableMap;
import static java.util.zip.ZipFile.OPEN_DELETE;
import static java.util.zip.ZipFile.OPEN_READ;
import static org.elasticsearch.entitlement.bridge.Util.NO_CLASS;
import static org.elasticsearch.entitlement.runtime.policy.PathLookup.BaseDir.TEMP;

/**
 * @class PolicyManager
 * @brief This class is the core component for managing security policies and
 *        performing runtime entitlement checks within Elasticsearch.
 *
 * It is responsible for:
 * <ul>
 * <li>Finding the <strong>component</strong> (system, server, plugin, agent) to which a
 *     caller class belongs.</li>
 * <li>Retrieving the appropriate policy and entitlements for that component.</li>
 * <li>Checking requested actions against the component's entitlements.</li>
 * </ul>
 * If an action is not permitted by the policy, a {@link NotEntitledException} is thrown.
 * <p>
 * <strong>Component Identification Logic:</strong>
 * <ol>
 * <li><strong>System Component:</strong> Modules contained in {@link PolicyManager#SYSTEM_LAYER_MODULES},
 *     which includes modules in the boot layer, defined by {@link ModuleFinder#ofSystem()},
 *     and not explicitly excluded (e.g., {@code java.desktop}). Operations from system components
 *     are trivially allowed.</li>
 * <li><strong>Server Component:</strong> Modules in the boot layer but not in the system layer,
 *     identified by {@link PolicyManager#SERVER_LAYER_MODULES}.</li>
 * <li><strong>Plugin/Module Component:</strong> Identified using a `pluginResolver` function
 *     which maps a Java Class to its plugin/module name.</li>
 * <li><strong>Known Agent Component:</strong> Specifically handles known agents like APM agents.</li>
 * <li><strong>Unknown Component:</strong> Any other component not falling into the above categories.</li>
 * </ol>
 * <p>
 * <strong>Entitlement Check Flow:</strong>
 * All check methods first determine the requesting class's module. If the module is
 * part of the system layer, the check is trivially allowed. Otherwise,
 * {@link ModuleEntitlements} (containing all entitlements for that module) are
 * lazily computed and cached. Subsequent checks use these cached entitlements.
 */
public class PolicyManager {
    /** @brief Logger for general messages within the policy manager. */
    private static final Logger generalLogger = LogManager.getLogger(PolicyManager.class);

    /** @brief Constant for component names that cannot be identified. */
    static final String UNKNOWN_COMPONENT_NAME = "(unknown)";
    /** @brief Constant for the server component name. */
    static final String SERVER_COMPONENT_NAME = "(server)";
    /** @brief Constant for the APM agent component name. */
    static final String APM_AGENT_COMPONENT_NAME = "(APM agent)";

    /** @brief Reference to the class of the default file system's provider. */
    static final Class<?> DEFAULT_FILESYSTEM_CLASS = PathUtils.getDefaultFileSystem().getClass();

    /** @brief Set of module names to exclude from the "system" component definition. */
    static final Set<String> MODULES_EXCLUDED_FROM_SYSTEM_MODULES = Set.of("java.desktop");

    /**
     * @record ModuleEntitlements
     * @brief A record encapsulating all entitlements for a specific module or component.
     *
     * Functional Utility: This record serves as a cached representation of the security
     * policy applicable to a particular module. It organizes entitlements by type
     * and provides a {@link FileAccessTree} for efficient file access checks.
     */
    record ModuleEntitlements(
        String componentName, /**< The name of the component (e.g., plugin name, "server", "APM agent"). */
        Map<Class<? extends Entitlement>, List<Entitlement>> entitlementsByType, /**< Map of entitlement types to a list of specific entitlement instances. */
        FileAccessTree fileAccess, /**< A tree structure for efficient file access entitlement checks. */
        Logger logger /**< A logger instance specific to this component/module. */
    ) {

        /**
         * @brief Canonical constructor for `ModuleEntitlements`.
         * Functional Utility: Ensures that the `entitlementsByType` map is immutable
         * upon record creation.
         */
        ModuleEntitlements {
            entitlementsByType = Map.copyOf(entitlementsByType);
        }

        /**
         * @brief Checks if this module has a specific type of entitlement.
         * @param entitlementClass The class of the entitlement to check for.
         * @return `true` if an entitlement of the given type is present, `false` otherwise.
         */
        public boolean hasEntitlement(Class<? extends Entitlement> entitlementClass) {
            return entitlementsByType.containsKey(entitlementClass);
        }

        /**
         * @brief Retrieves a stream of entitlements of a specific type.
         * @param entitlementClass The class of the entitlement to retrieve.
         * @param <E> The type of the entitlement.
         * @return A `Stream` of entitlements of the specified type, or an empty stream if none are present.
         */
        public <E extends Entitlement> Stream<E> getEntitlements(Class<E> entitlementClass) {
            var entitlements = entitlementsByType.get(entitlementClass);
            if (entitlements == null) {
                return Stream.empty();
            }
            return entitlements.stream().map(entitlementClass::cast);
        }
    }

    /**
     * @brief Provides a default {@link FileAccessTree} for a component path when no specific
     *        `FilesEntitlement` is defined.
     * @param componentPath The base path of the component.
     * @return A `FileAccessTree` with no specific file access rules other than what's implied by the path.
     * Functional Utility: Ensures that components without explicit file entitlements are still handled
     *                     gracefully with a basic file access tree.
     */
    private FileAccessTree getDefaultFileAccess(Path componentPath) {
        return FileAccessTree.withoutExclusivePaths(FilesEntitlement.EMPTY, pathLookup, componentPath);
    }

    /**
     * @brief Creates {@link ModuleEntitlements} with default (no specific policy) settings.
     *        Used when a module has no explicit entitlement policy.
     * @param componentName The name of the component.
     * @param componentPath The base path of the component.
     * @param moduleName The name of the module.
     * @return A `ModuleEntitlements` instance with default file access and a logger.
     * Functional Utility: Provides a fallback `ModuleEntitlements` for modules without defined policies.
     */
    // pkg private for testing
    ModuleEntitlements defaultEntitlements(String componentName, Path componentPath, String moduleName) {
        return new ModuleEntitlements(componentName, Map.of(), getDefaultFileAccess(componentPath), getLogger(componentName, moduleName));
    }

    /**
     * @brief Creates {@link ModuleEntitlements} based on a specific list of entitlements.
     * @param componentName The name of the component.
     * @param componentPath The base path of the component.
     * @param moduleName The name of the module.
     * @param entitlements A list of {@link Entitlement} instances for this module.
     * @return A `ModuleEntitlements` instance with defined file access and entitlements.
     * Functional Utility: Constructs a `ModuleEntitlements` object by grouping provided entitlements by type
     *                     and building a corresponding `FileAccessTree`.
     */
    // pkg private for testing
    ModuleEntitlements policyEntitlements(String componentName, Path componentPath, String moduleName, List<Entitlement> entitlements) {
        FilesEntitlement filesEntitlement = FilesEntitlement.EMPTY;
        // Extract the FilesEntitlement if present, otherwise use an empty one.
        /**
         * Block Logic: Iterates through the provided entitlements to identify and extract the `FilesEntitlement`.
         * Invariant: If a `FilesEntitlement` is present, it will be used to construct the `FileAccessTree`; otherwise, an empty one is used.
         */
        for (Entitlement entitlement : entitlements) {
            if (entitlement instanceof FilesEntitlement) {
                filesEntitlement = (FilesEntitlement) entitlement;
            }
        }
        return new ModuleEntitlements(
            componentName,
            entitlements.stream().collect(groupingBy(Entitlement::getClass)), // Group entitlements by their type.
            FileAccessTree.of(componentName, moduleName, filesEntitlement, pathLookup, componentPath, exclusivePaths), // Build file access tree.
            getLogger(componentName, moduleName) // Get a specific logger.
        );
    }

    /** @brief A concurrent map caching {@link Module} to {@link ModuleEntitlements} mappings for performance.
     *         Functional Utility: Improves performance by avoiding redundant computation of entitlements for modules.
     */
    final Map<Module, ModuleEntitlements> moduleEntitlementsMap = new ConcurrentHashMap<>();

    /** @brief Map of server-specific scope names to their entitlements. */
    private final Map<String, List<Entitlement>> serverEntitlements;
    /** @brief List of entitlements specifically for the APM agent. */
    private final List<Entitlement> apmAgentEntitlements;
    /** @brief Map of plugin names to their scope-based entitlements. */
    private final Map<String, Map<String, List<Entitlement>>> pluginsEntitlements;
    /** @brief Function to resolve a {@link Class} to its plugin name. */
    private final Function<Class<?>, String> pluginResolver;
    /** @brief {@link PathLookup} instance for resolving various file system paths. */
    private final PathLookup pathLookup;
    /** @brief Set of classes for which entitlement failures should not be logged. */
    private final Set<Class<?>> mutedClasses;

    /** @brief Constant for the name representing all unnamed modules in a policy. */
    public static final String ALL_UNNAMED = "ALL-UNNAMED";

    /** @brief A set of {@link Module}s considered part of the system layer.
     *         Functional Utility: Modules in this set are implicitly trusted and bypass explicit entitlement checks.
     */
    private static final Set<Module> SYSTEM_LAYER_MODULES = findSystemLayerModules();

    /**
     * @brief Helper method to identify and collect modules belonging to the system layer.
     * Functional Utility: Determines which modules are considered "system" modules,
     * which are implicitly trusted and bypass entitlement checks. This includes
     * modules in the boot layer, those found by {@link ModuleFinder#ofSystem()},
     * and not explicitly excluded (e.g., {@code java.desktop}).
     * @return An immutable `Set` of system layer {@link Module}s.
     * Postcondition: Returns a `Set` containing all identified system modules.
     */
    private static Set<Module> findSystemLayerModules() {
        var systemModulesDescriptors = ModuleFinder.ofSystem()
            .findAll()
            .stream()
            .map(ModuleReference::descriptor)
            .collect(Collectors.toUnmodifiableSet());
        return Stream.concat(
            // The entitlements module itself is always part of the system layer.
            Stream.of(PolicyManager.class.getModule()),
            // Any other module in the boot layer, provided it's a system module and not excluded.
            ModuleLayer.boot()
                .modules()
                .stream()
                .filter(
                    m -> systemModulesDescriptors.contains(m.getDescriptor())
                        && MODULES_EXCLUDED_FROM_SYSTEM_MODULES.contains(m.getName()) == false
                )
        ).collect(Collectors.toUnmodifiableSet());
    }

    /** @brief An immutable `Set` of {@link Module}s considered part of the server layer.
     *         Functional Utility: Contains all modules in the boot layer that are not part of the system layer, representing core server components.
     */
    // Anything in the boot layer that is not in the system layer, is in the server layer
    public static final Set<Module> SERVER_LAYER_MODULES = ModuleLayer.boot()
        .modules()
        .stream()
        .filter(m -> SYSTEM_LAYER_MODULES.contains(m) == false)
        .collect(Collectors.toUnmodifiableSet());

    /** @brief Map of component names to their source paths. */
    private final Map<String, Path> sourcePaths;
    /** @brief The package name containing classes from the APM agent. */
    private final String apmAgentPackageName;

    /** @brief Reference to the module containing the entitlements library itself.
     *         Frames originating from this module are ignored in the permission logic.
     */
    private final Module entitlementsModule;

    /**
     * @brief List of {@link ExclusivePath} objects derived from policies.
     * Functional Utility: These paths are only allowed for a single module. They are
     * used to generate structures in {@link FileAccessTree}s to indicate that
     * other modules are not allowed to use these files.
     */
    private final List<ExclusivePath> exclusivePaths;

    /**
     * @brief Constructs a new `PolicyManager`.
     *
     * Functional Utility: Initializes the policy manager with various security policies
     * for the server, APM agent, and plugins. It builds internal data structures
     * for efficient entitlement lookups and performs initial validation of file entitlements.
     *
     * @param serverPolicy The {@link Policy} for the server component.
     * @param apmAgentEntitlements A list of {@link Entitlement}s specifically for the APM agent.
     * @param pluginPolicies A map of plugin names to their {@link Policy}.
     * @param pluginResolver A function to resolve a {@link Class} to its plugin name.
     * @param sourcePaths A map holding the path to each plugin or module JAR, by plugin (or module) name.
     * @param apmAgentPackageName The package name containing classes from the APM agent.
     * @param entitlementsModule The module containing the entitlements library itself.
     * @param pathLookup A {@link PathLookup} instance for resolving base directories.
     * @param suppressFailureLogClasses A set of classes for which entitlement failures should not be logged.
     * Precondition: All input parameters are valid and non-null (where applicable).
     * Postcondition: The `PolicyManager` is initialized with all specified security policies, and exclusive paths are validated.
     */
    public PolicyManager(
        Policy serverPolicy,
        List<Entitlement> apmAgentEntitlements,
        Map<String, Policy> pluginPolicies,
        Function<Class<?>, String> pluginResolver,
        Map<String, Path> sourcePaths,
        String apmAgentPackageName,
        Module entitlementsModule,
        PathLookup pathLookup,
        Set<Class<?>> suppressFailureLogClasses
    ) {
        this.serverEntitlements = buildScopeEntitlementsMap(requireNonNull(serverPolicy));
        this.apmAgentEntitlements = apmAgentEntitlements;
        this.pluginsEntitlements = requireNonNull(pluginPolicies).entrySet()
            .stream()
            .collect(toUnmodifiableMap(Map.Entry::getKey, e -> buildScopeEntitlementsMap(e.getValue())));
        this.pluginResolver = pluginResolver;
        this.sourcePaths = sourcePaths;
        this.apmAgentPackageName = apmAgentPackageName;
        this.entitlementsModule = entitlementsModule;
        this.pathLookup = requireNonNull(pathLookup);
        this.mutedClasses = suppressFailureLogClasses;

        List<ExclusiveFileEntitlement> exclusiveFileEntitlements = new ArrayList<>();
        // Validate server entitlements and collect exclusive file entitlements.
        /**
         * Block Logic: Validates server entitlements and populates the `exclusiveFileEntitlements` list.
         * Invariant: All entitlements defined for the server component are processed.
         */
        for (var e : serverEntitlements.entrySet()) {
            validateEntitlementsPerModule(SERVER_COMPONENT_NAME, e.getKey(), e.getValue(), exclusiveFileEntitlements);
        }
        // Validate APM agent entitlements.
        validateEntitlementsPerModule(APM_AGENT_COMPONENT_NAME, ALL_UNNAMED, apmAgentEntitlements, exclusiveFileEntitlements);
        // Validate plugin entitlements.
        /**
         * Block Logic: Iterates through all plugin policies to validate their entitlements and collect exclusive file entitlements.
         * Invariant: All entitlements for all registered plugins are processed.
         */
        for (var p : pluginsEntitlements.entrySet()) {
            for (var m : p.getValue().entrySet()) {
                validateEntitlementsPerModule(p.getKey(), m.getKey(), m.getValue(), exclusiveFileEntitlements);
            }
        }
        // Build and validate the list of exclusive paths.
        List<ExclusivePath> exclusivePaths = FileAccessTree.buildExclusivePathList(exclusiveFileEntitlements, pathLookup);
        FileAccessTree.validateExclusivePaths(exclusivePaths);
        this.exclusivePaths = exclusivePaths;
    }

    /**
     * @brief Builds a map of scope names to their entitlements from a given {@link Policy}.
     * @param policy The policy containing the scopes.
     * @return An immutable map where keys are scope names and values are lists of {@link Entitlement}s.
     * Functional Utility: Transforms a `Policy` object into a more readily accessible map for entitlement lookups.
     * Precondition: `policy` is a valid `Policy` object.
     * Postcondition: Returns an unmodifiable map of scope names to their entitlements.
     */
    private static Map<String, List<Entitlement>> buildScopeEntitlementsMap(Policy policy) {
        return policy.scopes().stream().collect(toUnmodifiableMap(Scope::moduleName, Scope::entitlements));
    }

    /**
     * @brief Validates entitlements for a specific component and module, ensuring no duplicate
     *        entitlement types and collecting any {@link ExclusiveFileEntitlement}s.
     * @param componentName The name of the component.
     * @param moduleName The name of the module.
     * @param entitlements A list of entitlements for this module.
     * @param exclusiveFileEntitlements A list to collect any `ExclusiveFileEntitlement`s found.
     * Precondition: `entitlements` is a list of entitlements for the given component and module.
     * Postcondition: `exclusiveFileEntitlements` is updated with any found `FilesEntitlement`s marked as exclusive, and an exception is thrown if duplicate entitlement types are found.
     * @throws IllegalArgumentException if a duplicate entitlement type is found within the list.
     */
    private static void validateEntitlementsPerModule(
        String componentName,
        String moduleName,
        List<Entitlement> entitlements,
        List<ExclusiveFileEntitlement> exclusiveFileEntitlements
    ) {
        Set<Class<? extends Entitlement>> found = new HashSet<>();
        /**
         * Block Logic: Iterates through the entitlements to check for duplicates and collect exclusive file entitlements.
         * Invariant: Each entitlement in the list is checked for uniqueness of its type within the module.
         */
        for (var e : entitlements) {
            /**
             * Block Logic: Checks for duplicate entitlement types.
             * Invariant: Each entitlement type must appear at most once within a module's policy.
             */
            if (found.contains(e.getClass())) {
                throw new IllegalArgumentException(
                    "[" + componentName + "] using module [" + moduleName + "] found duplicate entitlement [" + e.getClass().getName() + "]"
                );
            }
            found.add(e.getClass());
            // If the entitlement is a FilesEntitlement, add it to the exclusive list if applicable.
            if (e instanceof FilesEntitlement fe) {
                exclusiveFileEntitlements.add(new ExclusiveFileEntitlement(componentName, moduleName, fe));
            }
        }
    }

    /**
     * @brief Checks entitlement for starting a new process. This operation is never entitled.
     * Functional Utility: Enforces that no component can directly initiate a new process for security reasons.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting to start a process.
     * Postcondition: A `NotEntitledException` is thrown unless the caller is trivially allowed.
     */
    public void checkStartProcess(Class<?> callerClass) {
        neverEntitled(callerClass, () -> "start process");
    }

    /**
     * @brief Checks entitlement for writing file store attributes. This operation is never entitled.
     * Functional Utility: Prevents any component from modifying file store attributes, which are typically system-level configurations.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting to write file store attributes.
     * Postcondition: A `NotEntitledException` is thrown unless the caller is trivially allowed.
     */
    public void checkWriteStoreAttributes(Class<?> callerClass) {
        neverEntitled(callerClass, () -> "change file store attributes");
    }

    /**
     * @brief Checks entitlement for reading file store attributes.
     * Functional Utility: Verifies if the calling class has been explicitly granted permission to read file store attributes.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting to read file store attributes.
     * Postcondition: If the caller is not entitled, a `NotEntitledException` is thrown.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    public void checkReadStoreAttributes(Class<?> callerClass) {
        checkEntitlementPresent(callerClass, ReadStoreAttributesEntitlement.class);
    }

    /**
     * @brief Helper method for operations that are never entitled.
     * Functional Utility: Determines if the caller is trivially allowed (system module),
     * otherwise throws a `NotEntitledException`.
     * @param callerClass The class initiating the operation.
     * @param operationDescription A supplier for a descriptive string of the operation.
     * Precondition: `callerClass` is the class making an unauthorized request.
     * Postcondition: An exception is thrown, or the check is skipped if the class is trivially allowed.
     * @throws NotEntitledException always, if not trivially allowed.
     */
    private void neverEntitled(Class<?> callerClass, Supplier<String> operationDescription) {
        var requestingClass = requestingClass(callerClass);
        /**
         * Block Logic: Allows the operation if the requesting class is from a trivially allowed module.
         * Invariant: Operations from system modules are always permitted.
         */
        if (isTriviallyAllowed(requestingClass)) {
            return;
        }

        ModuleEntitlements entitlements = getEntitlements(requestingClass);
        notEntitled(
            Strings.format(
                "component [%s], module [%s], class [%s], operation [%s]",
                entitlements.componentName(),
                getModuleName(requestingClass),
                requestingClass,
                operationDescription.get()
            ),
            callerClass,
            entitlements
        );
    }

    /**
     * @brief Checks entitlement for exiting the JVM.
     * Functional Utility: Controls which components have the authority to terminate the JVM process.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting to exit the JVM.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    public void checkExitVM(Class<?> callerClass) {
        checkEntitlementPresent(callerClass, ExitVMEntitlement.class);
    }

    /**
     * @brief Checks entitlement for creating a class loader.
     * Functional Utility: Restricts which components can create new class loaders, a sensitive operation.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting to create a class loader.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    public void checkCreateClassLoader(Class<?> callerClass) {
        checkEntitlementPresent(callerClass, CreateClassLoaderEntitlement.class);
    }

    /**
     * @brief Checks entitlement for setting HTTPS connection properties.
     * Functional Utility: Governs which components can modify properties related to HTTPS connections.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting to set HTTPS connection properties.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    public void checkSetHttpsConnectionProperties(Class<?> callerClass) {
        checkEntitlementPresent(callerClass, SetHttpsConnectionPropertiesEntitlement.class);
    }

    /**
     * @brief Checks entitlement for changing JVM global state.
     * Functional Utility: This is a generic check for operations that modify the JVM's
     * global configuration or behavior. It dynamically looks up the specific method
     * that triggered the check for an informative error message.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting to change JVM global state.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    public void checkChangeJVMGlobalState(Class<?> callerClass) {
        neverEntitled(callerClass, () -> walkStackForCheckMethodName().orElse("change JVM global state"));
    }

    /**
     * @brief Checks entitlement for creating a logging file handler. This operation is never entitled.
     * Functional Utility: Prevents unauthorized creation of logging file handlers for security and control.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting to create a logging file handler.
     * Postcondition: A `NotEntitledException` is thrown unless the caller is trivially allowed.
     */
    public void checkLoggingFileHandler(Class<?> callerClass) {
        neverEntitled(callerClass, () -> walkStackForCheckMethodName().orElse("create logging file handler"));
    }

    /**
     * @brief Walks the stack to find the check method name.
     * Functional Utility: Used to compose informative error messages by identifying
     * the specific `check$` method within the entitlement system that triggered the failure.
     * @return An `Optional` containing the descriptive method name, or empty if not found.
     * Postcondition: Returns an `Optional` of the method name, if found in the stack trace.
     */
    private Optional<String> walkStackForCheckMethodName() {
        // Look up the check$ method to compose an informative error message.
        // This way, we don't need to painstakingly describe every individual global-state change.
        return StackWalker.getInstance()
            .walk(
                frames -> frames.map(StackFrame::getMethodName)
                    .dropWhile(not(methodName -> methodName.startsWith(InstrumentationService.CHECK_METHOD_PREFIX)))
                    .findFirst()
            )
            .map(this::operationDescription);
    }

    /**
     * @brief Checks entitlement for operations that modify how network operations are handled.
     * Functional Utility: Delegates to `checkChangeJVMGlobalState` as network handling changes are considered JVM global state modifications.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting to modify network handling.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled to change JVM global state.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    public void checkChangeNetworkHandling(Class<?> callerClass) {
        checkChangeJVMGlobalState(callerClass);
    }

    /**
     * @brief Checks entitlement for operations that modify how file operations are handled.
     * Functional Utility: Delegates to `checkChangeJVMGlobalState` as file handling changes are considered JVM global state modifications.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting to modify file operations.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled to change JVM global state.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    public void checkChangeFilesHandling(Class<?> callerClass) {
        checkChangeJVMGlobalState(callerClass);
    }

    /**
     * @brief Checks entitlement for reading a {@link File}.
     * Functional Utility: Converts the `File` to a `Path` and delegates to `checkFileRead(Class, Path)`.
     * @param callerClass The class initiating the operation.
     * @param file The {@link File} object being accessed.
     * Precondition: `callerClass` is the class attempting to read the file.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled to read the file.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    @SuppressForbidden(reason = "Explicitly checking File apis")
    public void checkFileRead(Class<?> callerClass, File file) {
        checkFileRead(callerClass, file.toPath());
    }

    /**
     * @brief Checks if a given {@link Path} is on the default file system.
     * @param path The {@link Path} to check.
     * @return `true` if the path belongs to the default file system, `false` otherwise.
     * Functional Utility: Optimizes entitlement checks by ignoring paths that are not managed by the default file system provider.
     * Precondition: `path` is a valid `Path` object.
     * Postcondition: Returns `true` if the `Path` is associated with the default file system, `false` otherwise.
     */
    private static boolean isPathOnDefaultFilesystem(Path path) {
        var pathFileSystemClass = path.getFileSystem().getClass();
        /**
         * Block Logic: Determines if the file system associated with the path is the default one.
         * Invariant: Only operations on the default file system are subject to granular entitlement checks here.
         */
        if (path.getFileSystem().getClass() != DEFAULT_FILESYSTEM_CLASS) {
            generalLogger.trace(
                () -> Strings.format(
                    "File entitlement trivially allowed: path [%s] is for a different FileSystem class [%s], default is [%s]",
                    path.toString(),
                    pathFileSystemClass.getName(),
                    DEFAULT_FILESYSTEM_CLASS.getName()
                )
            );
            return false;
        }
        return true;
    }

    /**
     * @brief Checks entitlement for reading a {@link Path}.
     * @param callerClass The class initiating the operation.
     * @param path The {@link Path} object being accessed.
     * Precondition: `callerClass` is the class attempting to read the path.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled to read the path.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     * @throws AssertionError if `NoSuchFileException` is thrown inappropriately (i.e., when not following links).
     */
    public void checkFileRead(Class<?> callerClass, Path path) {
        try {
            checkFileRead(callerClass, path, false);
        } catch (NoSuchFileException e) {
            assert false : "NoSuchFileException should only be thrown when following links";
            var notEntitledException = new NotEntitledException(e.getMessage());
            notEntitledException.addSuppressed(e);
            throw notEntitledException;
        }
    }

    /**
     * @brief Checks entitlement for reading a {@link Path}, with an option to follow symbolic links.
     * @param callerClass The class initiating the operation.
     * @param path The {@link Path} object being accessed.
     * @param followLinks `true` if symbolic links should be followed, `false` otherwise.
     * Precondition: `callerClass` is the class attempting to read the path.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled to read the path (or its real path if links are followed).
     * @throws NoSuchFileException if `followLinks` is `true` and the symbolic link target does not exist.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    public void checkFileRead(Class<?> callerClass, Path path, boolean followLinks) throws NoSuchFileException {
        // Trivial allowance if path is not on the default filesystem.
        /**
         * Block Logic: Skips entitlement check if the path is not part of the default file system, as such paths are not typically managed by internal policies.
         * Invariant: Entitlement checks are only applied to paths on the default file system.
         */
        if (isPathOnDefaultFilesystem(path) == false) {
            return;
        }
        var requestingClass = requestingClass(callerClass);
        // Trivial allowance if the requesting class is implicitly trusted.
        /**
         * Block Logic: Allows the operation without further checks if the requesting class is from a trivially allowed module.
         * Invariant: Operations from system modules are always permitted.
         */
        if (isTriviallyAllowed(requestingClass)) {
            return;
        }

        ModuleEntitlements entitlements = getEntitlements(requestingClass);

        Path realPath = null;
        boolean canRead = entitlements.fileAccess().canRead(path);
        // If following links, resolve the real path and check entitlements against it.
        /**
         * Block Logic: Resolves the real path if symbolic links are to be followed and re-evaluates read access against the real path.
         * Invariant: If `followLinks` is true, entitlements are checked against the canonical, non-symlink path.
         */
        if (canRead && followLinks) {
            try {
                realPath = path.toRealPath();
                if (realPath.equals(path) == false) {
                    canRead = entitlements.fileAccess().canRead(realPath);
                }
            } catch (NoSuchFileException e) {
                throw e; // rethrow NoSuchFileException if it occurs during toRealPath.
            } catch (IOException e) {
                canRead = false; // Treat other I/O exceptions as denied access.
            }
        }

        // If access is still not granted, throw NotEntitledException.
        /**
         * Block Logic: Throws a `NotEntitledException` if read access is not granted after all checks.
         * Invariant: The operation is denied if `canRead` is false.
         */
        if (canRead == false) {
            notEntitled(
                Strings.format(
                    "component [%s], module [%s], class [%s], entitlement [file], operation [read], path [%s]",
                    entitlements.componentName(),
                    getModuleName(requestingClass),
                    requestingClass,
                    realPath == null ? path : Strings.format("%s -> %s", path, realPath) // Include real path if resolved.
                ),
                callerClass,
                entitlements
            );
        }
    }

    /**
     * @brief Checks entitlement for writing to a {@link File}.
     * Functional Utility: Converts the `File` to a `Path` and delegates to `checkFileWrite(Class, Path)`.
     * @param callerClass The class initiating the operation.
     * @param file The {@link File} object being accessed.
     * Precondition: `callerClass` is the class attempting to write to the file.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled to write to the file.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    @SuppressForbidden(reason = "Explicitly checking File apis")
    public void checkFileWrite(Class<?> callerClass, File file) {
        checkFileWrite(callerClass, file.toPath());
    }

    /**
     * @brief Checks entitlement for writing to a {@link Path}.
     * @param callerClass The class initiating the operation.
     * @param path The {@link Path} object being accessed.
     * Precondition: `callerClass` is the class attempting to write to the path.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled to write to the path.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    public void checkFileWrite(Class<?> callerClass, Path path) {
        // Trivial allowance if path is not on the default filesystem.
        /**
         * Block Logic: Skips entitlement check if the path is not part of the default file system.
         * Invariant: Entitlement checks are only applied to paths on the default file system.
         */
        if (isPathOnDefaultFilesystem(path) == false) {
            return;
        }
        var requestingClass = requestingClass(callerClass);
        // Trivial allowance if the requesting class is implicitly trusted.
        /**
         * Block Logic: Allows the operation without further checks if the requesting class is from a trivially allowed module.
         * Invariant: Operations from system modules are always permitted.
         */
        if (isTriviallyAllowed(requestingClass)) {
            return;
        }

        ModuleEntitlements entitlements = getEntitlements(requestingClass);
        // Check if write access is permitted by the file access policy.
        /**
         * Block Logic: Determines if the current module's file access policy permits writing to the specified path.
         * Invariant: If `canWrite` returns false, a `NotEntitledException` is thrown.
         */
        if (entitlements.fileAccess().canWrite(path) == false) {
            notEntitled(
                Strings.format(
                    "component [%s], module [%s], class [%s], entitlement [file], operation [write], path [%s]",
                    entitlements.componentName(),
                    getModuleName(requestingClass),
                    requestingClass,
                    path
                ),
                callerClass,
                entitlements
            );
        }
    }

    /**
     * @brief Checks entitlement for creating a temporary file.
     * Functional Utility: This check delegates to `checkFileWrite` on the
     * designated temporary directory path, assuming there is only one temp
     * directory in production.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting to create a temporary file.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled to write to the temporary directory.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    public void checkCreateTempFile(Class<?> callerClass) {
        // In production, there should only ever be a single temp directory.
        // So we can safely assume we only need to check the sole element in this stream.
        checkFileWrite(callerClass, pathLookup.getBaseDirPaths(TEMP).findFirst().get());
    }

    /**
     * @brief Checks entitlement for file access with specific ZIP modes.
     * Functional Utility: This method handles `OPEN_READ` and `OPEN_DELETE` flags
     * for ZIP file operations, delegating to `checkFileWrite` (which implies
     * both read and write) or `checkFileRead` as appropriate.
     * @param callerClass The class initiating the operation.
     * @param file The {@link File} object being accessed.
     * @param zipMode The ZIP file access mode (e.g., `OPEN_READ`, `OPEN_DELETE`).
     * Precondition: `callerClass` is the class attempting file access with ZIP modes.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled to the requested file operation.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    @SuppressForbidden(reason = "Explicitly checking File apis")
    public void checkFileWithZipMode(Class<?> callerClass, File file, int zipMode) {
        assert zipMode == OPEN_READ || zipMode == (OPEN_READ | OPEN_DELETE);
        /**
         * Block Logic: Determines the type of file access (read-only or read-write/delete) based on the `zipMode` flags.
         * Invariant: The appropriate entitlement check (`checkFileWrite` or `checkFileRead`) is invoked based on the `zipMode`.
         */
        if ((zipMode & OPEN_DELETE) == OPEN_DELETE) {
            // This needs both read and write, but we happen to know that checkFileWrite
            // actually checks both.
            checkFileWrite(callerClass, file);
        } else {
            checkFileRead(callerClass, file);
        }
    }

    /**
     * @brief Checks entitlement for reading a file descriptor. This operation is never entitled.
     * Functional Utility: Prohibits direct reading of file descriptors, a low-level operation that could bypass higher-level security.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting to read a file descriptor.
     * Postcondition: A `NotEntitledException` is thrown unless the caller is trivially allowed.
     */
    public void checkFileDescriptorRead(Class<?> callerClass) {
        neverEntitled(callerClass, () -> "read file descriptor");
    }

    /**
     * @brief Checks entitlement for writing to a file descriptor. This operation is never entitled.
     * Functional Utility: Prohibits direct writing to file descriptors, a low-level operation that could bypass higher-level security.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting to write to a file descriptor.
     * Postcondition: A `NotEntitledException` is thrown unless the caller is trivially allowed.
     */
    public void checkFileDescriptorWrite(Class<?> callerClass) {
        neverEntitled(callerClass, () -> "write file descriptor");
    }

    /**
     * @brief Checks entitlement for getting an arbitrary {@code FileAttributeView}.
     * Functional Utility: This operation is over-approximated and directly denied
     * because `FileAttributeView`s can modify sensitive attributes like owner,
     * and introducing granular checks for each operation is complex.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting to get a file attribute view.
     * Postcondition: A `NotEntitledException` is thrown unless the caller is trivially allowed.
     */
    public void checkGetFileAttributeView(Class<?> callerClass) {
        neverEntitled(callerClass, () -> "get file attribute view");
    }

    /**
     * @brief Checks entitlement for loading native libraries.
     * Functional Utility: Controls which components are allowed to load native libraries, a potential security risk.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting to load native libraries.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    public void checkLoadingNativeLibraries(Class<?> callerClass) {
        checkEntitlementPresent(callerClass, LoadNativeLibrariesEntitlement.class);
    }

    /**
     * @brief Extracts a human-readable operation description from a check method name.
     * @param methodName The name of the check method (e.g., `checkStartProcess`).
     * @return A descriptive string for the operation.
     * @note This might need to be more sophisticated for better descriptions.
     * Functional Utility: Converts internal check method names into more user-friendly operation descriptions for logging and exceptions.
     * Precondition: `methodName` is a string representing a check method.
     * Postcondition: Returns a string representing the operation that was checked.
     */
    private String operationDescription(String methodName) {
        // TODO: Use a more human-readable description. Perhaps share code with InstrumentationServiceImpl.parseCheckerMethodName
        return methodName.substring(methodName.indexOf('$')); // Assumes method names start with "check" followed by dollar sign.
    }

    /**
     * @brief Checks entitlement for inbound network access.
     * Functional Utility: Verifies if the calling class has been explicitly granted permission for inbound network connections.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting inbound network access.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    public void checkInboundNetworkAccess(Class<?> callerClass) {
        checkEntitlementPresent(callerClass, InboundNetworkEntitlement.class);
    }

    /**
     * @brief Checks entitlement for outbound network access.
     * Functional Utility: Verifies if the calling class has been explicitly granted permission for outbound network connections.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting outbound network access.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    public void checkOutboundNetworkAccess(Class<?> callerClass) {
        checkEntitlementPresent(callerClass, OutboundNetworkEntitlement.class);
    }

    /**
     * @brief Checks entitlement for both inbound and outbound network access.
     * Functional Utility: Performs two separate entitlement checks for each direction
     * of network access.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting network access.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled to perform either inbound or outbound network access.
     * @throws NotEntitledException if the caller class is not entitled to perform either inbound or outbound network access.
     */
    public void checkAllNetworkAccess(Class<?> callerClass) {
        var requestingClass = requestingClass(callerClass);
        /**
         * Block Logic: Allows the operation without further checks if the requesting class is from a trivially allowed module.
         * Invariant: Operations from system modules are always permitted.
         */
        if (isTriviallyAllowed(requestingClass)) {
            return;
        }

        var classEntitlements = getEntitlements(requestingClass);
        checkFlagEntitlement(classEntitlements, InboundNetworkEntitlement.class, requestingClass, callerClass);
        checkFlagEntitlement(classEntitlements, OutboundNetworkEntitlement.class, requestingClass, callerClass);
    }

    /**
     * @brief Checks entitlement for using an unsupported URL protocol connection. This operation is never entitled.
     * Functional Utility: Prevents the use of unauthorized or unknown URL protocols for security and stability.
     * @param callerClass The class initiating the operation.
     * @param protocol The unsupported URL protocol.
     * Precondition: `callerClass` is the class attempting to use an unsupported URL protocol.
     * Postcondition: A `NotEntitledException` is thrown unless the caller is trivially allowed.
     */
    public void checkUnsupportedURLProtocolConnection(Class<?> callerClass, String protocol) {
        neverEntitled(callerClass, () -> Strings.format("unsupported URL protocol [%s]", protocol));
    }

    /**
     * @brief Helper method to check if a specific flag-based entitlement is present.
     * Functional Utility: Used for simple entitlements that only require checking
     * their presence (e.g., network access flags). Throws `NotEntitledException`
     * if the entitlement is missing.
     * @param classEntitlements The {@link ModuleEntitlements} for the requesting class.
     * @param entitlementClass The class of the entitlement to check.
     * @param requestingClass The class identified as requesting the entitlement.
     * @param callerClass The class directly calling the check method.
     * Precondition: `classEntitlements` is initialized for the `requestingClass`.
     * Postcondition: If the `entitlementClass` is not present, a `NotEntitledException` is thrown.
     * @throws NotEntitledException if the entitlement is not present.
     */
    private void checkFlagEntitlement(
        ModuleEntitlements classEntitlements,
        Class<? extends Entitlement> entitlementClass,
        Class<?> requestingClass,
        Class<?> callerClass
    ) {
        /**
         * Block Logic: Checks if the requested entitlement is present in the module's policy.
         * Invariant: If the entitlement is not found, an exception is thrown.
         */
        if (classEntitlements.hasEntitlement(entitlementClass) == false) {
            notEntitled(
                Strings.format(
                    "component [%s], module [%s], class [%s], entitlement [%s]",
                    classEntitlements.componentName(),
                    getModuleName(requestingClass),
                    requestingClass,
                    PolicyParser.buildEntitlementNameFromClass(entitlementClass)
                ),
                callerClass,
                classEntitlements
            );
        }
        classEntitlements.logger()
            .debug(
                () -> Strings.format(
                    "Entitled: component [%s], module [%s], class [%s], entitlement [%s]",
                    classEntitlements.componentName(),
                    getModuleName(requestingClass),
                    requestingClass,
                    PolicyParser.buildEntitlementNameFromClass(entitlementClass)
                )
            );
    }

    /**
     * @brief Checks entitlement for writing a system property.
     * Functional Utility: Determines if the requesting class has specific permission
     * to write the given system property.
     * @param callerClass The class initiating the operation.
     * @param property The name of the system property to write.
     * Precondition: `callerClass` is the class attempting to write the system property.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled to write the specific property.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    public void checkWriteProperty(Class<?> callerClass, String property) {
        var requestingClass = requestingClass(callerClass);
        /**
         * Block Logic: Allows the operation without further checks if the requesting class is from a trivially allowed module.
         * Invariant: Operations from system modules are always permitted.
         */
        if (isTriviallyAllowed(requestingClass)) {
            return;
        }

        ModuleEntitlements entitlements = getEntitlements(requestingClass);
        // Check if any `WriteSystemPropertiesEntitlement` allows writing this specific property.
        /**
         * Block Logic: Checks if any `WriteSystemPropertiesEntitlement` in the module's policy explicitly permits writing the target property.
         * Invariant: If a matching entitlement is found, the operation is allowed.
         */
        if (entitlements.getEntitlements(WriteSystemPropertiesEntitlement.class).anyMatch(e -> e.properties().contains(property))) {
            entitlements.logger()
                .debug(
                    () -> Strings.format(
                        "Entitled: component [%s], module [%s], class [%s], entitlement [write_system_properties], property [%s]",
                        entitlements.componentName(),
                        getModuleName(requestingClass),
                        requestingClass,
                        property
                    )
                );
            return;
        }
        notEntitled(
            Strings.format(
                "component [%s], module [%s], class [%s], entitlement [write_system_properties], property [%s]",
                entitlements.componentName(),
                getModuleName(requestingClass),
                requestingClass,
                property
            ),
            callerClass,
            entitlements
        );
    }

    /**
     * @brief Throws a {@link NotEntitledException} and logs the entitlement failure.
     * @param message The detailed error message.
     * @param callerClass The class that directly called the check method.
     * @param entitlements The {@link ModuleEntitlements} of the requesting class.
     * Precondition: An entitlement check has failed.
     * Postcondition: A `NotEntitledException` is thrown with a detailed message, and a warning is logged (unless the caller class is muted).
     * @throws NotEntitledException always.
     */
    private void notEntitled(String message, Class<?> callerClass, ModuleEntitlements entitlements) {
        var exception = new NotEntitledException(message);
        // Don't emit a log for muted classes, e.g. classes containing self tests
        /**
         * Block Logic: Logs the entitlement failure unless the `callerClass` is explicitly muted from logging.
         * Invariant: Entitlement failures are logged for unmuted classes to provide visibility into security violations.
         */
        if (mutedClasses.contains(callerClass) == false) {
            entitlements.logger().warn("Not entitled: {}", message, exception);
        }
        throw exception;
    }

    /**
     * @brief Retrieves or creates a logger specific to a component and module.
     * Functional Utility: Ensures that logging messages are attributed to the
     * correct source within the entitlement system.
     * @param componentName The name of the component.
     * @param moduleName The name of the module.
     * @return A {@link Logger} instance.
     * Postcondition: Returns a `Logger` instance uniquely identified by the component and module names.
     */
    private static Logger getLogger(String componentName, String moduleName) {
        var loggerSuffix = "." + componentName + "." + ((moduleName == null) ? ALL_UNNAMED : moduleName);
        return MODULE_LOGGERS.computeIfAbsent(PolicyManager.class.getName() + loggerSuffix, LogManager::getLogger);
    }

    /**
     * @brief A concurrent map caching {@link Logger} instances.
     * Functional Utility: Ensures that the same `Logger` object is used for a
     * given name, which is important for `ModuleEntitlements`' `equals` and `hashCode`
     * methods to work correctly (if `Logger`s were part of their state).
     */
    private static final ConcurrentHashMap<String, Logger> MODULE_LOGGERS = new ConcurrentHashMap<>();

    /**
     * @brief Checks entitlement for managing threads.
     * @param callerClass The class initiating the operation.
     * Precondition: `callerClass` is the class attempting to manage threads.
     * Postcondition: A `NotEntitledException` is thrown if the caller is not entitled.
     * @throws NotEntitledException if the caller class is not entitled to perform this action.
     */
    public void checkManageThreadsEntitlement(Class<?> callerClass) {
        checkEntitlementPresent(callerClass, ManageThreadsEntitlement.class);
    }

    /**
     * @brief Helper method to check if a specific {@link Entitlement} class is present for a caller.
     * Functional Utility: A common pattern for checking entitlements that act as simple flags.
     * @param callerClass The class directly calling the check method.
     * @param entitlementClass The class of the entitlement to check.
     * Precondition: `callerClass` is the class requesting a specific `entitlementClass`.
     * Postcondition: If the `requestingClass` is not trivially allowed and the `entitlementClass` is not found, a `NotEntitledException` is thrown.
     * @throws NotEntitledException if the entitlement is not present.
     */
    private void checkEntitlementPresent(Class<?> callerClass, Class<? extends Entitlement> entitlementClass) {
        var requestingClass = requestingClass(callerClass);
        // Trivial allowance if the requesting class is implicitly trusted.
        /**
         * Block Logic: Allows the operation without further checks if the requesting class is from a trivially allowed module.
         * Invariant: Operations from system modules are always permitted.
         */
        if (isTriviallyAllowed(requestingClass)) {
            return;
        }
        checkFlagEntitlement(getEntitlements(requestingClass), entitlementClass, requestingClass, callerClass);
    }

    /**
     * @brief Lazily computes and caches {@link ModuleEntitlements} for a given requesting class.
     * Functional Utility: Optimizes performance by computing `ModuleEntitlements` only once per module and storing them in a cache.
     * @param requestingClass The class for which to retrieve entitlements.
     * @return The {@link ModuleEntitlements} for the requesting class's module.
     * Postcondition: Returns the `ModuleEntitlements` for the `requestingClass`'s module, either from cache or newly computed.
     */
    ModuleEntitlements getEntitlements(Class<?> requestingClass) {
        return moduleEntitlementsMap.computeIfAbsent(requestingClass.getModule(), m -> computeEntitlements(requestingClass));
    }

    /**
     * @brief Determines and computes the {@link ModuleEntitlements} for a requesting class based on its module's component.
     * Functional Utility: Identifies whether the module belongs to the server, a plugin, or an agent,
     * and constructs the appropriate `ModuleEntitlements` instance.
     * @param requestingClass The class for which to compute entitlements.
     * @return The computed `ModuleEntitlements`.
     * Precondition: `requestingClass` is a valid `Class` object.
     * Postcondition: Returns a `ModuleEntitlements` instance corresponding to the `requestingClass`'s component.
     */
    private ModuleEntitlements computeEntitlements(Class<?> requestingClass) {
        Module requestingModule = requestingClass.getModule();
        // Check if it's a server module.
        /**
         * Block Logic: Determines if the requesting module is a server module and retrieves its entitlements.
         * Invariant: Server modules are identified by being named and part of the boot layer.
         */
        if (isServerModule(requestingModule)) {
            return getModuleScopeEntitlements(
                serverEntitlements,
                requestingModule.getName(),
                SERVER_COMPONENT_NAME,
                getComponentPathFromClass(requestingClass)
            );
        }

        // Check if it's a plugin module.
        var pluginName = pluginResolver.apply(requestingClass);
        /**
         * Block Logic: Determines if the requesting class belongs to a plugin and retrieves its entitlements.
         * Invariant: Plugin entitlements are looked up by the plugin name resolved from the class.
         */
        if (pluginName != null) {
            var pluginEntitlements = pluginsEntitlements.get(pluginName);
            /**
             * Block Logic: Handles cases where a plugin is identified but has no explicit entitlements defined.
             * Invariant: If no specific plugin entitlements exist, default entitlements are applied.
             */
            if (pluginEntitlements == null) {
                // If plugin entitlements are not explicitly defined, use default.
                return defaultEntitlements(pluginName, sourcePaths.get(pluginName), requestingModule.getName());
            } else {
                return getModuleScopeEntitlements(
                    pluginEntitlements,
                    getScopeName(requestingModule),
                    pluginName,
                    sourcePaths.get(pluginName)
                );
            }
        }

        // Special handling for the APM agent, which might run non-modular.
        /**
         * Block Logic: Specifically handles the case of the APM agent, which may operate as a non-modular entity.
         * Invariant: APM agent entitlements are applied if the class belongs to the APM agent package and is non-modular.
         */
        if (requestingModule.isNamed() == false && requestingClass.getPackageName().startsWith(apmAgentPackageName)) {
            // The APM agent is the only thing running non-modular in the system classloader.
            return policyEntitlements(
                APM_AGENT_COMPONENT_NAME,
                getComponentPathFromClass(requestingClass),
                ALL_UNNAMED,
                apmAgentEntitlements
            );
        }

        // Default entitlements for unknown components.
        return defaultEntitlements(UNKNOWN_COMPONENT_NAME, null, requestingModule.getName());
    }

    /**
     * @brief Returns the scope name for a given module.
     * Functional Utility: For named modules, it returns the module's name. For unnamed modules,
     * it returns {@link #ALL_UNNAMED}.
     * @param requestingModule The module.
     * @return The scope name as it would appear in an entitlement policy.
     * Precondition: `requestingModule` is a valid `Module` object.
     * Postcondition: Returns the module's name if named, or `ALL_UNNAMED` if unnamed.
     */
    private static String getScopeName(Module requestingModule) {
        /**
         * Block Logic: Determines the scope name based on whether the module is named or unnamed.
         * Invariant: Named modules use their actual name as the scope, while unnamed modules use `ALL_UNNAMED`.
         */
        if (requestingModule.isNamed() == false) {
            return ALL_UNNAMED;
        } else {
            return requestingModule.getName();
        }
    }

    /**
     * @brief Extracts the component path (JAR location) from a class.
     * @param requestingClass The class for which to find the component path.
     * @return A {@link Path} object representing the location of the class's code source, or `null` if not found or invalid.
     * Functional Utility: Determines the physical location of the code source for a given class, which is vital for path-based entitlement checks.
     * Precondition: `requestingClass` is a valid `Class` object.
     * Postcondition: Returns the `Path` to the component's JAR or directory, or `null` if unable to determine.
     */
    // pkg private for testing
    static Path getComponentPathFromClass(Class<?> requestingClass) {
        var codeSource = requestingClass.getProtectionDomain().getCodeSource();
        /**
         * Block Logic: If no code source is available for the class, return null.
         * Invariant: A valid code source is required to determine the component path.
         */
        if (codeSource == null) {
            return null;
        }
        try {
            return Paths.get(codeSource.getLocation().toURI());
        } catch (Exception e) {
            // If we get a URISyntaxException, or any other Exception due to an invalid URI, we return null to safely skip this location
            generalLogger.info(
                "Cannot get component path for [{}]: [{}] cannot be converted to a valid Path",
                requestingClass.getName(),
                codeSource.getLocation().toString()
            );
            return null;
        }
    }

    /**
     * @brief Retrieves module-specific scope entitlements.
     * Functional Utility: Looks up the entitlements for a specific scope name within a component.
     * If no explicit entitlements are found for that scope, it provides default entitlements.
     * @param scopeEntitlements Map of scope names to their entitlements.
     * @param scopeName The name of the scope.
     * @param componentName The name of the component.
     * @param componentPath The path of the component.
     * @return The {@link ModuleEntitlements} for the specified scope.
     * Precondition: `scopeEntitlements` is a map of entitlements, and `scopeName`, `componentName`, `componentPath` are valid.
     * Postcondition: Returns the `ModuleEntitlements` for the given scope, either defined or default.
     */
    private ModuleEntitlements getModuleScopeEntitlements(
        Map<String, List<Entitlement>> scopeEntitlements,
        String scopeName,
        String componentName,
        Path componentPath
    ) {
        var entitlements = scopeEntitlements.get(scopeName);
        /**
         * Block Logic: If no specific entitlements are found for the scope, provides default entitlements.
         * Invariant: Every scope will have an associated `ModuleEntitlements` object, even if it's a default one.
         */
        if (entitlements == null) {
            return defaultEntitlements(componentName, componentPath, scopeName);
        }
        return policyEntitlements(componentName, componentPath, scopeName, entitlements);
    }

    /**
     * @brief Checks if a given module is considered a server module.
     * Functional Utility: A server module is a named module that is part of the boot layer.
     * @param requestingModule The module to check.
     * @return `true` if the module is a server module, `false` otherwise.
     * Precondition: `requestingModule` is a valid `Module` object.
     * Postcondition: Returns `true` if the module is a named module in the boot layer, `false` otherwise.
     */
    private static boolean isServerModule(Module requestingModule) {
        return requestingModule.isNamed() && requestingModule.getLayer() == ModuleLayer.boot();
    }

    /**
     * @brief Walks the stack to determine which class should be checked for entitlements.
     *
     * Functional Utility: This method is used when the direct `callerClass` is not
     * sufficient or provided. It traverses the call stack to find the first class
     * outside of the entitlement library itself that initiated the operation.
     *
     * @param callerClass when non-null will be returned;
     *                    this is a fast-path check that can avoid the stack walk
     *                    in cases where the caller class is available.
     * @return the requesting class, or {@code null} if the entire call stack
     * comes from the entitlement library itself.
     * Postcondition: Returns the `Class` object that initiated the entitlement-checked operation, or `null`.
     */
    Class<?> requestingClass(Class<?> callerClass) {
        /**
         * Block Logic: Provides a fast-path if `callerClass` is directly provided, avoiding a stack walk.
         * Invariant: If `callerClass` is non-null, it is assumed to be the correct requesting class.
         */
        if (callerClass != null) {
            // Fast path: if callerClass is directly provided, use it.
            return callerClass;
        }
        // If no callerClass, perform a stack walk.
        Optional<Class<?>> result = StackWalker.getInstance(RETAIN_CLASS_REFERENCE)
            .walk(frames -> findRequestingFrame(frames).map(StackFrame::getDeclaringClass));
        return result.orElse(null);
    }

    /**
     * @brief Given a stream of {@link StackFrame}s, identifies the one whose entitlements should be checked.
     * Functional Utility: Filters out stack frames originating from the entitlement library itself
     * and skips the immediate caller, returning the first relevant frame.
     * @param frames A `Stream` of `StackFrame`s representing the current call stack.
     * @return An `Optional` containing the {@link StackFrame} of the requesting class, or empty if not found.
     * Postcondition: Returns an `Optional` of the first `StackFrame` that is outside the entitlement module and after the immediate caller.
     */
    Optional<StackFrame> findRequestingFrame(Stream<StackFrame> frames) {
        return frames.filter(f -> f.getDeclaringClass().getModule() != entitlementsModule) // Ignore frames from entitlement library.
            .skip(1) // Skip the sensitive caller method (the check method itself).
            .findFirst();
    }

    /**
     * @brief Determines if an operation is "trivially allowed" (i.e., granted permission
     *        regardless of specific entitlements).
     *
     * Functional Utility: Operations initiated by trusted components (like system modules)
     * are always allowed without further policy checks. This method identifies such cases.
     *
     * @param requestingClass The class identified as requesting the entitlement.
     * @return `true` if permission is granted regardless of specific entitlements, `false` otherwise.
     * Postcondition: Returns `true` if the `requestingClass` is associated with a system layer module or an outermost frame, `false` otherwise.
     */
    private static boolean isTriviallyAllowed(Class<?> requestingClass) {
        /**
         * Block Logic: Logs the stack trace for trivially allowed checks for debugging purposes.
         * Invariant: This logging only occurs when trace level is enabled.
         */
        if (generalLogger.isTraceEnabled()) {
            generalLogger.trace("Stack trace for upcoming trivially-allowed check", new Exception());
        }
        // If no requesting class is found (e.g., stack entirely within entitlement lib).
        /**
         * Block Logic: Handles cases where no requesting class can be identified, implying the call originates from within the entitlement library.
         * Invariant: If `requestingClass` is null, the operation is trivially allowed.
         */
        if (requestingClass == null) {
            generalLogger.debug("Entitlement trivially allowed: no caller frames outside the entitlement library");
            return true;
        }
        // Special marker indicating outermost frame.
        /**
         * Block Logic: Identifies a special marker class that signifies the outermost frame, which is always trivially allowed.
         * Invariant: Operations originating from the outermost frame are always permitted.
         */
        if (requestingClass == NO_CLASS) {
            generalLogger.debug("Entitlement trivially allowed from outermost frame");
            return true;
        }
        // If the requesting class belongs to a system layer module.
        /**
         * Block Logic: Checks if the requesting class belongs to a module defined as part of the system layer.
         * Invariant: Operations from system layer modules are inherently trusted and allowed.
         */
        if (SYSTEM_LAYER_MODULES.contains(requestingClass.getModule())) {
            generalLogger.debug("Entitlement trivially allowed from system module [{}]", requestingClass.getModule().getName());
            return true;
        }
        generalLogger.trace("Entitlement not trivially allowed");
        return false;
    }

    /**
     * @brief Returns the module name of a requesting class as it would appear in an entitlement policy file.
     * Functional Utility: Provides a consistent way to represent module names for policy lookups,
     * handling unnamed modules by assigning {@link #ALL_UNNAMED}.
     * @param requestingClass The class for which to get the module name.
     * @return The module name string.
     * Postcondition: Returns the module's name if named, or `ALL_UNNAMED` if unnamed.
     */
    private static String getModuleName(Class<?> requestingClass) {
        String name = requestingClass.getModule().getName();
        /**
         * Block Logic: Assigns `ALL_UNNAMED` to modules that do not have an explicit name.
         * Invariant: All modules are represented by a non-null string name for policy evaluation.
         */
        return (name == null) ? ALL_UNNAMED : name;
    }

    /**
     * @brief Provides a string representation of the `PolicyManager` instance.
     * @return A string detailing the server and plugin entitlements.
     */
    @Override
    public String toString() {
        return "PolicyManager{" + "serverEntitlements=" + serverEntitlements + ", pluginsEntitlements=" + pluginsEntitlements + '}';
    }
}