/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.entitlement.runtime.policy;

import org.elasticsearch.entitlement.runtime.policy.FileAccessTree.ExclusiveFileEntitlement;
import org.elasticsearch.entitlement.runtime.policy.FileAccessTree.ExclusivePath;
import org.elasticsearch.entitlement.runtime.policy.entitlements.Entitlement;
import org.elasticsearch.entitlement.runtime.policy.entitlements.FilesEntitlement;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;

import java.lang.module.ModuleFinder;
import java.lang.module.ModuleReference;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.toUnmodifiableMap;
import static org.elasticsearch.entitlement.bridge.Util.NO_CLASS;
import static org.elasticsearch.entitlement.runtime.policy.PolicyManager.ComponentKind.APM_AGENT;
import static org.elasticsearch.entitlement.runtime.policy.PolicyManager.ComponentKind.PLUGIN;
import static org.elasticsearch.entitlement.runtime.policy.PolicyManager.ComponentKind.SERVER;
import static org.elasticsearch.entitlement.runtime.policy.PolicyManager.ComponentKind.UNKNOWN;

/**
 * @brief Manages and evaluates entitlement policies to control access for various caller classes within Elasticsearch.
 *
 * This class is a central component of the entitlement system. It determines, based on predefined
 * policy information, which {@link Entitlement}s are granted to a given caller class.
 * It also handles the concept of "trivially allowed" classes, which are always granted
 * access regardless of specific policies (e.g., core JDK classes).
 *
 * Functional Utility: Serves as the decision-making engine for security, ensuring that
 *                     components (server, plugins, agents) operate within their defined
 *                     security boundaries. It consolidates entitlement definitions and
 *                     provides a lookup mechanism for runtime checks.
 * Architectural Role: Orchestrates the application of different policy types (server-level,
 *                     plugin-specific, agent-specific) and integrates with bytecode
 *                     instrumentation to enforce these policies dynamically. It manages
 *                     a mapping of modules to their granted entitlements.
 */
public class PolicyManager {
    /**
     * @brief A special module name used for classes that do not belong to a named Java module (e.g., classes in the unnamed module).
     * Functional Utility: Provides a conventional identifier for code that cannot be associated with a specific modular scope,
     *                     allowing entitlements to be applied to it (e.g., for non-modular plugins).
     */
    public static final String ALL_UNNAMED = "ALL-UNNAMED";
    /**
     * @brief A general-purpose {@link Logger} instance for the PolicyManager.
     * Functional Utility: Used for logging information or warnings not specific to any particular module or component,
     *                     or when a module-specific logger is not yet available.
     * Rationale: Marked as `static` and `final` to ensure consistent logging behavior and avoid repeated logger instantiation.
     */
    static final Logger generalLogger = LogManager.getLogger(PolicyManager.class);

    /**
     * @brief A set of Java modules that are explicitly excluded from being considered "system modules" for entitlement purposes.
     * Functional Utility: Prevents certain JDK modules (e.g., `java.desktop`) from being automatically
     *                     granted "trivially allowed" status, allowing for more specific control over their access.
     * Rationale: Some modules, while part of the JDK, might expose APIs or functionalities that require
     *            entitlement checks for security reasons, even if they are in the boot layer.
     */
    static final Set<String> MODULES_EXCLUDED_FROM_SYSTEM_MODULES = Set.of("java.desktop");

    /**
     * @brief Identifies a particular entitlement {@link Scope} within a {@link Policy}.
     *        This record uniquely identifies a logical boundary for applying entitlements,
     *        combining the kind of component, its specific name, and the module within it.
     *
     * @param kind The {@link ComponentKind} (e.g., SERVER, PLUGIN) that this scope belongs to.
     * @param componentName The specific name of the component (e.g., a plugin's name, "(server)").
     * @param moduleName The name of the Java module within the component (or {@link #ALL_UNNAMED} for non-modular code).
     */
    public record PolicyScope(ComponentKind kind, String componentName, String moduleName) {
        /**
         * Functional Utility: Ensures that all components of a {@link PolicyScope} are valid upon construction.
         * Invariants:
         * - `kind`, `componentName`, `moduleName` are non-null.
         * - If `kind` corresponds to a single predefined component (e.g., `SERVER`), `componentName` must match that component's name.
         */
        public PolicyScope {
            requireNonNull(kind);
            requireNonNull(componentName);
            requireNonNull(moduleName);
            assert kind.componentName == null || kind.componentName.equals(componentName);
        }

        /**
         * @brief Creates a {@link PolicyScope} for an unknown or unidentifiable component.
         * @param moduleName The module name to associate with the unknown scope.
         * @return A new {@link PolicyScope} instance of kind {@link ComponentKind#UNKNOWN}.
         */
        public static PolicyScope unknown(String moduleName) {
            return new PolicyScope(UNKNOWN, UNKNOWN.componentName, moduleName);
        }

        /**
         * @brief Creates a {@link PolicyScope} for a server-level module.
         * @param moduleName The name of the server module.
         * @return A new {@link PolicyScope} instance of kind {@link ComponentKind#SERVER}.
         */
        public static PolicyScope server(String moduleName) {
            return new PolicyScope(SERVER, SERVER.componentName, moduleName);
        }

        /**
         * @brief Creates a {@link PolicyScope} for an APM agent.
         * @param moduleName The module name to associate with the APM agent scope.
         * @return A new {@link PolicyScope} instance of kind {@link ComponentKind#APM_AGENT}.
         */
        public static PolicyScope apmAgent(String moduleName) {
            return new PolicyScope(APM_AGENT, APM_AGENT.componentName, moduleName);
        }

        /**
         * @brief Creates a {@link PolicyScope} for a plugin module.
         * @param componentName The name of the plugin.
         * @param moduleName The name of the plugin's module.
         * @return A new {@link PolicyScope} instance of kind {@link ComponentKind#PLUGIN}.
         */
        public static PolicyScope plugin(String componentName, String moduleName) {
            return new PolicyScope(PLUGIN, componentName, moduleName);
        }
    }

    public enum ComponentKind {
        /**
         * @brief Represents a component of unknown origin or type.
         */
        UNKNOWN("(unknown)"),
        /**
         * @brief Represents the Elasticsearch server component.
         */
        SERVER("(server)"),
        /**
         * @brief Represents an APM agent component.
         */
        APM_AGENT("(APM agent)"),
        /**
         * @brief Represents a plugin component.
         * Rationale: Plugins do not have a fixed component name like SERVER or APM_AGENT,
         *            as their names are dynamic.
         */
        PLUGIN(null);

        /**
         * @brief If this kind corresponds to a single predefined component (e.g., SERVER), this is that component's name.
         *        Otherwise, it is `null`, indicating that the component name is dynamic (e.g., for plugins).
         */
        final String componentName;

        ComponentKind(String componentName) {
            this.componentName = componentName;
        }
    }

    /**
     * @brief Encapsulates all entitlements granted to a specific module within a component, including file access.
     *
     * This record serves as a comprehensive view of the permissions available to a module. It stores
     * entitlements categorized by their type, a dedicated {@link FileAccessTree} for efficient filesystem checks,
     * and a logger for module-specific logging.
     *
     * Functional Utility: Provides a single, convenient object to query for any type of entitlement
     *                     granted to a particular module, simplifying access control decisions.
     * Architectural Role: Acts as a cached representation of a module's policy, created once upon first access
     *                     and then reused, optimizing performance for repeated entitlement checks.
     *
     * @param componentName The name of the component (e.g., plugin name, "(server)") this module belongs to.
     * @param entitlementsByType A map where keys are {@link Entitlement} classes and values are lists of
     *                           corresponding entitlement instances.
     * @param fileAccess A {@link FileAccessTree} instance configured for this module's file access policies.
     * @param logger A module-specific {@link Logger} for detailed logging related to this module's entitlements.
     */
    record ModuleEntitlements(
        String componentName,
        Map<Class<? extends Entitlement>, List<Entitlement>> entitlementsByType,
        FileAccessTree fileAccess,
        Logger logger
    ) {

        /**
         * Functional Utility: Ensures that the internal map of entitlements is immutable after construction,
         *                     guaranteeing thread safety and preventing external modification of policies.
         */
        public ModuleEntitlements {
            entitlementsByType = Map.copyOf(entitlementsByType);
        }

        /**
         * @brief Checks if this module has been granted any entitlement of a specific type.
         * @param entitlementClass The {@link Class} of the {@link Entitlement} to check for.
         * @return `true` if at least one entitlement of the given type exists, `false` otherwise.
         * Functional Utility: Provides a quick way to determine if a module has a certain category of permission.
         */
        public boolean hasEntitlement(Class<? extends Entitlement> entitlementClass) {
            return entitlementsByType.containsKey(entitlementClass);
        }

        /**
         * @brief Retrieves all entitlements of a specific type granted to this module.
         * @param entitlementClass The {@link Class} of the {@link Entitlement} to retrieve.
         * @return A {@link Stream} of entitlements of the specified type. Returns an empty stream if no such
         *         entitlements are found.
         * Functional Utility: Allows fine-grained access to all instances of a particular entitlement type,
         *                     enabling detailed policy evaluation.
         */
        public <E extends Entitlement> Stream<E> getEntitlements(Class<E> entitlementClass) {
            var entitlements = entitlementsByType.get(entitlementClass);
            if (entitlements == null) {
                return Stream.empty();
            }
            return entitlements.stream().map(entitlementClass::cast);
        }
    }

    private FileAccessTree getDefaultFileAccess(Collection<Path> componentPaths) {
        /**
         * @brief Creates a default {@link FileAccessTree} with no specific file entitlements beyond implicit access to component paths.
         * @param componentPaths A collection of paths associated with the current component.
         * @return A {@link FileAccessTree} instance configured with an empty {@link FilesEntitlement}.
         * Functional Utility: Provides a baseline file access tree for modules that either do not declare
         *                     specific file entitlements or are implicitly granted access to their own
         *                     component paths.
         */
        return FileAccessTree.withoutExclusivePaths(FilesEntitlement.EMPTY, pathLookup, componentPaths);
    }

    /**
     * @brief Creates a {@link ModuleEntitlements} instance representing a default set of entitlements.
     * @param componentName The name of the component (e.g., plugin name) this module belongs to.
     * @param componentPaths A collection of paths associated with the current component.
     * @param moduleName The name of the Java module for which to create default entitlements.
     * @return A {@link ModuleEntitlements} instance with an empty set of entitlements and a default {@link FileAccessTree}.
     * Functional Utility: Provides a fallback or baseline entitlement set for modules that do not have
     *                     explicitly defined policies, ensuring they operate with minimal permissions.
     */
    ModuleEntitlements defaultEntitlements(String componentName, Collection<Path> componentPaths, String moduleName) {
        return new ModuleEntitlements(componentName, Map.of(), getDefaultFileAccess(componentPaths), getLogger(componentName, moduleName));
    }

    /**
     * @brief Creates a {@link ModuleEntitlements} instance based on a specific list of entitlements.
     * @param componentName The name of the component (e.g., plugin name) this module belongs to.
     * @param componentPaths A collection of paths associated with the current component.
     * @param moduleName The name of the Java module.
     * @param entitlements A list of {@link Entitlement} objects applicable to this module.
     * @return A {@link ModuleEntitlements} instance with the specified entitlements and a configured {@link FileAccessTree}.
     * Functional Utility: Builds a complete set of module entitlements from a provided list, including
     *                     setting up the {@link FileAccessTree} to reflect file access policies.
     * Pre-condition: `entitlements` may contain a {@link FilesEntitlement}.
     */
    ModuleEntitlements policyEntitlements(
        String componentName,
        Collection<Path> componentPaths,
        String moduleName,
        List<Entitlement> entitlements
    ) {
        FilesEntitlement filesEntitlement = FilesEntitlement.EMPTY;
        /**
         * Block Logic: Iterates through the provided entitlements to find and extract the {@link FilesEntitlement} instance.
         * Functional Utility: Separates the file access policies from other entitlements, as file access requires
         *                     specialized handling via {@link FileAccessTree}.
         * Pre-condition: `entitlements` is a list of {@link Entitlement} objects.
         * Invariant: `filesEntitlement` will either be {@link FilesEntitlement#EMPTY} or the found {@link FilesEntitlement}.
         */
        for (Entitlement entitlement : entitlements) {
            if (entitlement instanceof FilesEntitlement) {
                filesEntitlement = (FilesEntitlement) entitlement;
            }
        }
        return new ModuleEntitlements(
            componentName,
            entitlements.stream().collect(groupingBy(Entitlement::getClass)),
            FileAccessTree.of(componentName, moduleName, filesEntitlement, pathLookup, componentPaths, exclusivePaths),
            getLogger(componentName, moduleName)
        );
    }

    /**
     * @brief A concurrent map caching {@link ModuleEntitlements} instances for each Java {@link Module}.
     * Functional Utility: Improves performance by storing computed entitlement sets, avoiding redundant
     *                     calculations for modules whose entitlements have already been determined.
     * Invariant: Each entry maps a {@link Module} to its corresponding {@link ModuleEntitlements}.
     */
    final Map<Module, ModuleEntitlements> moduleEntitlementsMap = new ConcurrentHashMap<>();

    /**
     * @brief Stores the server-level entitlements, mapped by module name (scope name).
     * Functional Utility: Provides quick lookup for entitlements applicable to core Elasticsearch server modules.
     * Invariant: Each entry maps a module name to a list of {@link Entitlement}s applicable to that module.
     */
    private final Map<String, List<Entitlement>> serverEntitlements;
    /**
     * @brief Stores the entitlements specifically for the APM agent.
     * Functional Utility: Centralizes the permissions granted to the APM agent, which operates as a special component.
     * Invariant: Contains a list of {@link Entitlement}s applicable to the APM agent.
     */
    private final List<Entitlement> apmAgentEntitlements;
    /**
     * @brief Stores plugin-specific entitlements, structured as `pluginName -> moduleName -> List<Entitlement>`.
     * Functional Utility: Enables the {@link PolicyManager} to retrieve entitlements for any plugin module.
     * Invariant: Each plugin name maps to another map, which in turn maps module names to their respective entitlements.
     */
    private final Map<String, Map<String, List<Entitlement>>> pluginsEntitlements;
    /**
     * @brief A function that resolves a calling class into its corresponding {@link PolicyScope}.
     * Functional Utility: Dynamically determines the context (component, module) of a class attempting to perform an action,
     *                     allowing the {@link PolicyManager} to retrieve the correct set of entitlements.
     */
    private final Function<Class<?>, PolicyScope> scopeResolver;
    /**
     * @brief An instance of {@link PathLookup} used for resolving file system paths.
     * Functional Utility: Provides an abstraction for looking up various base directories and resolving paths from settings,
     *                     crucial for evaluating {@link FilesEntitlement} policies.
     */
    private final PathLookup pathLookup;

    private static final Set<Module> SYSTEM_LAYER_MODULES = findSystemLayerModules();

    /**
     * @brief Identifies and collects all modules that are considered part of the system layer and are not explicitly excluded.
     * @return An unmodifiable {@link Set} of {@link Module} objects representing the system layer.
     * Functional Utility: Establishes the baseline of highly trusted code that is "trivially allowed"
     *                     to perform operations without explicit entitlement checks.
     * Rationale: Includes the entitlements module itself, as well as modules from the boot layer
     *            that are system modules and not in the exclusion list (e.g., `java.desktop`).
     */
    private static Set<Module> findSystemLayerModules() {
        /**
         * Block Logic: Discovers all modules provided by the system module finder and collects their descriptors.
         * Functional Utility: Creates a set of module descriptors for efficient lookup, used to filter modules
         *                     from the boot layer that are genuinely system modules.
         */
        var systemModulesDescriptors = ModuleFinder.ofSystem()
            .findAll()
            .stream()
            .map(ModuleReference::descriptor)
            .collect(Collectors.toUnmodifiableSet());
        /**
         * Block Logic: Combines the entitlements module with filtered modules from the boot layer to form the complete set of system modules.
         * Functional Utility: Defines the precise set of modules that are considered part of the trusted system.
         * Pre-condition: `systemModulesDescriptors` contains the descriptors of all system modules.
         * Invariant: The resulting set is unmodifiable and does not contain any modules from {@link #MODULES_EXCLUDED_FROM_SYSTEM_MODULES}.
         */
        return Stream.concat(
            // entitlements is a "system" module, we can do anything from it
            Stream.of(PolicyManager.class.getModule()),
            // anything in the boot layer is also part of the system
            ModuleLayer.boot()
                .modules()
                .stream()
                .filter(
                    m -> systemModulesDescriptors.contains(m.getDescriptor())
                        && MODULES_EXCLUDED_FROM_SYSTEM_MODULES.contains(m.getName()) == false
                )
        ).collect(Collectors.toUnmodifiableSet());
    }

    // Anything in the boot layer that is not in the system layer, is in the server layer
    /**
     * @brief A set of {@link Module}s that constitute the "server layer" in the application.
     * Functional Utility: Defines the modules that are part of the core Elasticsearch server,
     *                     distinguishing them from system-level and plugin modules.
     * Rationale: These modules are in the boot layer but are not considered part of the
     *            trivially allowed system modules, meaning their actions are subject to entitlement checks.
     */
    public static final Set<Module> SERVER_LAYER_MODULES = ModuleLayer.boot()
        .modules()
        .stream()
        .filter(m -> SYSTEM_LAYER_MODULES.contains(m) == false)
        .collect(Collectors.toUnmodifiableSet());

    /**
     * @brief Stores the base paths of each plugin, mapped by plugin name.
     * Functional Utility: Provides the physical locations from which plugins are loaded,
     *                     which can be used to grant plugins default read access to their own files.
     */
    private final Map<String, Collection<Path>> pluginSourcePaths;

    /**
     * @brief Contains a consolidated list of paths that are exclusively controlled by certain modules/components.
     * Functional Utility: These paths are used to prevent other modules/components from gaining access,
     *                     enforcing strict isolation for critical resources.
     * Invariant: The list is sorted and validated to ensure no overlaps or duplicates, ensuring
     *            that exclusive claims are unambiguous.
     */
    private final List<ExclusivePath> exclusivePaths;

    /**
     * @brief Constructs a new {@link PolicyManager} instance, initializing it with all relevant policies and resolvers.
     *        This involves processing server, APM agent, and plugin policies, building exclusive path lists,
     *        and validating for conflicts.
     * @param serverPolicy The {@link Policy} defining server-level entitlements.
     * @param apmAgentEntitlements A list of {@link Entitlement}s specifically for the APM agent.
     * @param pluginPolicies A map of plugin names to their respective {@link Policy} objects.
     * @param scopeResolver A function to resolve a {@link Class} into its {@link PolicyScope}.
     * @param pluginSourcePaths A map of plugin names to their source {@link Path}s.
     * @param pathLookup An instance of {@link PathLookup} for resolving file system paths.
     * Functional Utility: Sets up the entire entitlement evaluation framework, preparing it to answer
     *                     runtime entitlement queries efficiently and securely.
     * Pre-condition: All input policies and resolvers are valid and properly configured.
     * Post-condition: The {@link PolicyManager} is fully initialized with parsed and validated policies,
     *                 and `exclusivePaths` are built and validated.
     * @throws IllegalArgumentException if duplicate entitlements are found within a module or
     *                                  if exclusive paths overlap.
     */
    public PolicyManager(
        Policy serverPolicy,
        List<Entitlement> apmAgentEntitlements,
        Map<String, Policy> pluginPolicies,
        Function<Class<?>, PolicyScope> scopeResolver,
        Map<String, Collection<Path>> pluginSourcePaths,
        PathLookup pathLookup
    ) {
        /**
         * Block Logic: Builds a map of server entitlements, grouped by module name, from the provided `serverPolicy`.
         * Functional Utility: Organizes server policies for efficient lookup by module.
         * Invariant: `serverPolicy` must not be null.
         */
        this.serverEntitlements = buildScopeEntitlementsMap(requireNonNull(serverPolicy));
        this.apmAgentEntitlements = apmAgentEntitlements;
        /**
         * Block Logic: Processes `pluginPolicies` to create a nested map of plugin entitlements by plugin and module name.
         * Functional Utility: Transforms raw plugin policies into a structured, easily queryable format.
         * Invariant: `pluginPolicies` must not be null.
         */
        this.pluginsEntitlements = requireNonNull(pluginPolicies).entrySet()
            .stream()
            .collect(toUnmodifiableMap(Map.Entry::getKey, e -> buildScopeEntitlementsMap(e.getValue())));
        this.scopeResolver = scopeResolver;
        this.pluginSourcePaths = pluginSourcePaths;
        this.pathLookup = requireNonNull(pathLookup);

        List<ExclusiveFileEntitlement> exclusiveFileEntitlements = new ArrayList<>();
        /**
         * Block Logic: Iterates through server entitlements to extract any exclusive file entitlements and validate them.
         * Functional Utility: Collects all explicitly marked exclusive file paths from server policies.
         * Invariant: No duplicate entitlements are allowed per module.
         */
        for (var e : serverEntitlements.entrySet()) {
            validateEntitlementsPerModule(SERVER.componentName, e.getKey(), e.getValue(), exclusiveFileEntitlements);
        }
        /**
         * Block Logic: Validates APM agent entitlements and extracts any exclusive file entitlements.
         * Functional Utility: Ensures APM agent policies are correctly formed and exclusive paths are identified.
         * Invariant: APM agent entitlements are treated as a single module with the name `ALL_UNNAMED`.
         */
        validateEntitlementsPerModule(APM_AGENT.componentName, ALL_UNNAMED, apmAgentEntitlements, exclusiveFileEntitlements);
        /**
         * Block Logic: Iterates through plugin entitlements to extract any exclusive file entitlements and validate them.
         * Functional Utility: Collects all explicitly marked exclusive file paths from plugin policies.
         * Invariant: No duplicate entitlements are allowed per module within each plugin.
         */
        for (var p : pluginsEntitlements.entrySet()) {
            for (var m : p.getValue().entrySet()) {
                validateEntitlementsPerModule(p.getKey(), m.getKey(), m.getValue(), exclusiveFileEntitlements);
            }
        }
        /**
         * Block Logic: Builds a consolidated list of all unique and sorted exclusive paths from the collected exclusive file entitlements.
         * Functional Utility: Creates a final, validated list of paths that are reserved for exclusive use.
         * Pre-condition: `exclusiveFileEntitlements` contains all found exclusive entitlements.
         * Invariant: `exclusivePaths` is a sorted list of unique {@link ExclusivePath} objects.
         */
        List<ExclusivePath> exclusivePaths = FileAccessTree.buildExclusivePathList(
            exclusiveFileEntitlements,
            pathLookup,
            FileAccessTree.DEFAULT_COMPARISON
        );
        /**
         * Block Logic: Validates that there are no overlapping or duplicate exclusive paths in the final list.
         * Functional Utility: Ensures the integrity of the exclusive path configuration, preventing ambiguities.
         * Invariant: An {@link IllegalArgumentException} is thrown if any conflicts are detected.
         */
        FileAccessTree.validateExclusivePaths(exclusivePaths, FileAccessTree.DEFAULT_COMPARISON);
        this.exclusivePaths = exclusivePaths;
    }

    /**
     * @brief Transforms a {@link Policy} object into a map of entitlements, keyed by module name (scope name).
     * @param policy The {@link Policy} to process.
     * @return An unmodifiable {@link Map} where keys are module names and values are lists of {@link Entitlement}s.
     * Functional Utility: Provides a structured representation of a policy's scopes, making it easier to
     *                     access entitlements for specific modules.
     * Invariant: The returned map is immutable, reflecting the static nature of policies after loading.
     */
    private static Map<String, List<Entitlement>> buildScopeEntitlementsMap(Policy policy) {
        return policy.scopes().stream().collect(toUnmodifiableMap(Scope::moduleName, Scope::entitlements));
    }

    private static void validateEntitlementsPerModule(
        String componentName,
        String moduleName,
        List<Entitlement> entitlements,
        List<ExclusiveFileEntitlement> exclusiveFileEntitlements
    ) {
        Set<Class<? extends Entitlement>> found = new HashSet<>();
        /**
         * Block Logic: Iterates through each {@link Entitlement} granted to a specific module within a component.
         * Functional Utility: Performs two key tasks:
         *                     1. Checks for duplicate entitlements of the same type, ensuring policy clarity.
         *                     2. Extracts {@link FilesEntitlement} instances to build a global list of exclusive file paths.
         * Pre-condition: `entitlements` is a list of all entitlements for the given module.
         * Invariant: An {@link IllegalArgumentException} is thrown if a duplicate entitlement type is found for the same module.
         */
        for (var e : entitlements) {
            /**
             * Block Logic: Detects if an entitlement of the same class has already been processed for this module.
             * Functional Utility: Enforces a "one entitlement type per module" rule to prevent ambiguous or conflicting policies.
             * Invariant: An {@link IllegalArgumentException} is thrown if a duplicate entitlement type is detected.
             */
            if (found.contains(e.getClass())) {
                throw new IllegalArgumentException(
                    "[" + componentName + "] using module [" + moduleName + "] found duplicate entitlement [" + e.getClass().getName() + "]"
                );
            }
            found.add(e.getClass());
            /**
             * Block Logic: If the current entitlement is a {@link FilesEntitlement}, it is wrapped into an
             *              {@link ExclusiveFileEntitlement} and added to the list.
             * Functional Utility: Gathers all file-related entitlements that declare exclusive paths, which
             *                     are then processed globally to prevent conflicts across components.
             * Invariant: All {@link FilesEntitlement} instances are collected for later exclusive path validation.
             */
            if (e instanceof FilesEntitlement fe) {
                exclusiveFileEntitlements.add(new ExclusiveFileEntitlement(componentName, moduleName, fe));
            }
        }
    }

    /**
     * @brief Retrieves a {@link Logger} instance for a specific component and module combination.
     * @param componentName The name of the component (e.g., plugin name, "(server)").
     * @param moduleName The name of the module (or {@link #ALL_UNNAMED}).
     * @return A {@link Logger} instance uniquely identified by the component and module.
     * Functional Utility: Provides dedicated logging channels for different modules, allowing for
     *                     more granular control over log levels and destinations for entitlement-related messages.
     * Invariant: Ensures that the same {@link Logger} object is returned for identical component/module pairs.
     */
    private static Logger getLogger(String componentName, String moduleName) {
        var loggerSuffix = "." + componentName + "." + ((moduleName == null) ? ALL_UNNAMED : moduleName);
        return MODULE_LOGGERS.computeIfAbsent(PolicyManager.class.getName() + loggerSuffix, LogManager::getLogger);
    }

    /**
     * @brief A concurrent hash map to cache {@link Logger} instances for different modules.
     * Functional Utility: Ensures that the same {@link Logger} object is used for a given component-module
     *                     combination, which is important for {@link ModuleEntitlements#equals(Object)}
     *                     and {@link ModuleEntitlements#hashCode()} to work correctly.
     * Rationale: {@link LogManager#getLogger(String)} does not guarantee memoization of loggers,
     *            hence this explicit caching mechanism is necessary.
     */
    private static final ConcurrentHashMap<String, Logger> MODULE_LOGGERS = new ConcurrentHashMap<>();

    /**
     * @brief Retrieves the {@link ModuleEntitlements} for a given requesting {@link Class}.
     *        This method leverages a cache to avoid redundant computation of entitlements.
     * @param requestingClass The {@link Class} for which to retrieve entitlements.
     * @return The {@link ModuleEntitlements} associated with the module of the `requestingClass`.
     * Functional Utility: Provides the primary entry point for obtaining the full set of
     *                     entitlements applicable to a piece of code at runtime.
     * Invariant: Entitlements are computed only once per module and then cached for subsequent requests.
     */
    ModuleEntitlements getEntitlements(Class<?> requestingClass) {
        return moduleEntitlementsMap.computeIfAbsent(requestingClass.getModule(), m -> computeEntitlements(requestingClass));
    }

    private ModuleEntitlements computeEntitlements(Class<?> requestingClass) {
        var policyScope = scopeResolver.apply(requestingClass);
        var componentName = policyScope.componentName();
        var moduleName = policyScope.moduleName();

        /**
         * Block Logic: Determines the type of component (SERVER, APM_AGENT, UNKNOWN, PLUGIN) based on the {@link PolicyScope}.
         * Functional Utility: Dispatches to the appropriate logic for retrieving or constructing entitlements
         *                     depending on whether the requesting class belongs to the server, an agent,
         *                     an unknown component, or a plugin.
         * Pre-condition: `policyScope` is a valid {@link PolicyScope} resolved from the `requestingClass`.
         * Invariant: Exactly one branch of the switch statement is executed, returning a {@link ModuleEntitlements} object.
         */
        switch (policyScope.kind()) {
            case SERVER -> {
                /**
                 * Functional Utility: Retrieves entitlements for server-side modules.
                 * Invariant: Server modules have their entitlements pre-loaded into `serverEntitlements`.
                 */
                return getModuleScopeEntitlements(
                    serverEntitlements,
                    moduleName,
                    SERVER.componentName,
                    getComponentPathsFromClass(requestingClass)
                );
            }
            case APM_AGENT -> {
                /**
                 * Functional Utility: Provides entitlements for the APM agent, which operates outside the standard modular system.
                 * Invariant: APM agent entitlements are hardcoded and treated as a single `ALL_UNNAMED` module.
                 */
                // The APM agent is the only thing running non-modular in the system classloader
                return policyEntitlements(
                    APM_AGENT.componentName,
                    getComponentPathsFromClass(requestingClass),
                    ALL_UNNAMED,
                    apmAgentEntitlements
                );
            }
            case UNKNOWN -> {
                /**
                 * Functional Utility: Assigns default (minimal) entitlements to classes from unknown components.
                 * Invariant: Unknown components receive a base level of access, preventing unauthorized actions.
                 */
                return defaultEntitlements(UNKNOWN.componentName, List.of(), moduleName);
            }
            default -> {
                /**
                 * Functional Utility: Handles entitlements for plugin modules.
                 * Invariant: If a plugin has no specific entitlements defined, it receives default entitlements.
                 */
                assert policyScope.kind() == PLUGIN;
                var pluginEntitlements = pluginsEntitlements.get(componentName);
                Collection<Path> componentPaths = pluginSourcePaths.getOrDefault(componentName, List.of());
                if (pluginEntitlements == null) {
                    return defaultEntitlements(componentName, componentPaths, moduleName);
                } else {
                    return getModuleScopeEntitlements(pluginEntitlements, moduleName, componentName, componentPaths);
                }
            }
        }
    }

    /**
     * @brief Retrieves the file system {@link Path}s from which a given {@link Class} was loaded.
     * @param requestingClass The {@link Class} for which to determine source paths.
     * @return A {@link Collection} of {@link Path} objects representing the source locations of the class.
     *         Returns an empty list if the code source is null or if there's an error resolving the URI.
     * Functional Utility: Essential for granting modules read access to their own code base,
     *                     supporting self-referential entitlements.
     */
    static Collection<Path> getComponentPathsFromClass(Class<?> requestingClass) {
        var codeSource = requestingClass.getProtectionDomain().getCodeSource();
        /**
         * Block Logic: Checks if the `codeSource` for the `requestingClass` is available.
         * Functional Utility: Handles cases where a class might not have a code source (e.g., dynamically generated classes),
         *                     preventing NullPointerExceptions.
         * Invariant: If `codeSource` is null, an empty list is returned immediately.
         */
        if (codeSource == null) {
            return List.of();
        }
        try {
            /**
             * Block Logic: Attempts to convert the `codeSource`'s {@link URL} to a {@link URI} and then to a {@link Path}.
             * Functional Utility: Extracts the physical file system location from which the class was loaded.
             * Pre-condition: `codeSource.getLocation()` returns a valid {@link URL}.
             */
            return List.of(Paths.get(codeSource.getLocation().toURI()));
        } catch (Exception e) {
            /**
             * Block Logic: Catches any exceptions that occur during the conversion of the URL to a Path.
             * Functional Utility: Provides graceful handling for malformed URIs or other unexpected errors,
             *                     logging the issue and returning an empty list rather than failing.
             * Invariant: Logs an informative message if path resolution fails.
             */
            // If we get a URISyntaxException, or any other Exception due to an invalid URI, we return null to safely skip this location
            generalLogger.info(
                "Cannot get component path for [{}]: [{}] cannot be converted to a valid Path",
                requestingClass.getName(),
                codeSource.getLocation().toString()
            );
            return List.of();
        }
    }

    /**
     * @brief Retrieves {@link ModuleEntitlements} for a specific module scope, using either predefined or default entitlements.
     * @param scopeEntitlements A map of module names to their entitlements within a specific policy (e.g., server or plugin policy).
     * @param scopeName The name of the module scope to look up.
     * @param componentName The name of the component (e.g., plugin name, "(server)").
     * @param componentPaths A collection of paths associated with the current component.
     * @return The {@link ModuleEntitlements} for the specified scope.
     * Functional Utility: Provides a flexible way to obtain entitlements, falling back to default entitlements
     *                     if no specific policy is defined for the given scope.
     */
    private ModuleEntitlements getModuleScopeEntitlements(
        Map<String, List<Entitlement>> scopeEntitlements,
        String scopeName,
        String componentName,
        Collection<Path> componentPaths
    ) {
        var entitlements = scopeEntitlements.get(scopeName);
        if (entitlements == null) {
            return defaultEntitlements(componentName, componentPaths, scopeName);
        }
        return policyEntitlements(componentName, componentPaths, scopeName, entitlements);
    }

    /**
     * @brief Determines if a permission request from a given {@link Class} should be "trivially allowed".
     *        Trivially allowed classes do not undergo further entitlement checks.
     * @param requestingClass The {@link Class} that is requesting permission.
     * @return `true` if the class is trivially allowed, `false` otherwise.
     * Functional Utility: Provides an essential optimization and security bypass for trusted code
     *                     (e.g., core JDK classes, internal framework code) that is known to be safe
     *                     or would cause circular dependencies if checked.
     * Pre-condition: `requestingClass` can be null (representing no caller context), {@link Util#NO_CLASS}, or a valid {@link Class}.
     */
    boolean isTriviallyAllowed(Class<?> requestingClass) {
        if (generalLogger.isTraceEnabled()) {
            generalLogger.trace("Stack trace for upcoming trivially-allowed check", new Exception());
        }
        /**
         * Block Logic: Checks if `requestingClass` is null, indicating no caller context.
         * Functional Utility: Allows permissions when the call stack leading to the check is untraceable
         *                     or originates from outside the instrumented application scope.
         * Invariant: Returns `true` for null `requestingClass`.
         */
        if (requestingClass == null) {
            generalLogger.debug("Entitlement trivially allowed: no caller frames outside the entitlement library");
            return true;
        }
        /**
         * Block Logic: Checks if `requestingClass` is {@link Util#NO_CLASS}, a sentinel for the outermost frame.
         * Functional Utility: Handles scenarios where the permission check occurs at the very top of the call stack,
         *                     often indicating a direct call from the application entry point.
         * Invariant: Returns `true` for {@link Util#NO_CLASS}.
         */
        if (requestingClass == NO_CLASS) {
            generalLogger.debug("Entitlement trivially allowed from outermost frame");
            return true;
        }
        /**
         * Block Logic: Checks if the `requestingClass` belongs to a trusted system module.
         * Functional Utility: Determines if the class is part of the JDK or other core trusted components
         *                     that are inherently allowed to perform privileged operations.
         * Invariant: Delegates to {@link #isTrustedSystemClass(Class)} for the actual trust evaluation.
         */
        if (isTrustedSystemClass(requestingClass)) {
            generalLogger.debug("Entitlement trivially allowed from system module [{}]", requestingClass.getModule().getName());
            return true;
        }
        generalLogger.trace("Entitlement not trivially allowed");
        return false;
    }

    /**
     * @brief The main decision point for determining if a {@link Class} belongs to a trusted, built-in JDK module.
     * @param requestingClass The {@link Class} to check for system trust.
     * @return `true` if the class's module is part of the {@link #SYSTEM_LAYER_MODULES}, `false` otherwise.
     * Functional Utility: Centralizes the logic for identifying highly privileged classes that should
     *                     not be subjected to fine-grained entitlement checks, thereby preventing
     *                     unnecessary overhead and potential security policy conflicts with the JDK itself.
     */
    protected boolean isTrustedSystemClass(Class<?> requestingClass) {
        return SYSTEM_LAYER_MODULES.contains(requestingClass.getModule());
    }

    @Override
    /**
     * @brief Provides a string representation of the {@link PolicyManager} instance.
     * @return A {@link String} detailing the server and plugin entitlements managed by this instance.
     * Functional Utility: Aids in debugging and logging by offering a concise overview of the
     *                     configured policies.
     */
    public String toString() {
        return "PolicyManager{" + "serverEntitlements=" + serverEntitlements + ", pluginsEntitlements=" + pluginsEntitlements + '}';
    }
}
