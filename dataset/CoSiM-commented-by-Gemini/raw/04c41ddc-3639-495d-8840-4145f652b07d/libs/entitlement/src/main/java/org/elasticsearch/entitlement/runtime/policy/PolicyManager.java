/**
 * @file PolicyManager.java
 * @brief Manages and enforces entitlement policies at runtime.
 *
 * @details
 * This class is the core of the runtime entitlement enforcement engine. Its primary responsibility
 * is to determine the security context (or "component") of a piece of code that is attempting
 * a privileged action and then check if the policy associated with that component grants the
 * necessary entitlement for the action.
 *
 * The `PolicyManager` operates by:
 * 1.  **Component Identification**: When a check is triggered, it walks the call stack to find the
 *     first frame originating from outside the entitlement library itself. It then uses the module
 *     information of the caller's class to determine which component it belongs to (e.g., system,
 *     server, a specific plugin, or an agent like APM).
 *
 * 2.  **Policy and Entitlement Resolution**: It maintains a cache (`moduleEntitlementsMap`) that maps
 *     a Java `Module` to its calculated `ModuleEntitlements`. This record contains all entitlements
 *     granted to that module, derived from the policies loaded during bootstrap. This caching
 *     avoids the need to re-compute entitlements for every check.
 *
 * 3.  **Entitlement Checking**: For a given action (like `checkExitVM` or `checkFileRead`), it looks
 *     up the entitlements for the calling component. If the required entitlement is present, the
 *     action is allowed to proceed. If not, it throws a `NotEntitledException`, effectively
 *     blocking the operation.
 *
 * 4.  **Trivial Allowance**: Code originating from core Java system modules (as identified by
 *     `SYSTEM_LAYER_MODULES`) is "trivially allowed" to perform any action, bypassing the
 *     entitlement checks. This is a fundamental trust assumption of the security model.
 *
 * The class handles a variety of entitlement types, including file system access (managed via
 * a `FileAccessTree`), network operations, classloader creation, and thread management, providing
 * a comprehensive security layer for the application.
 */
/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
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
import static org.elasticsearch.entitlement.runtime.policy.PolicyManager.ComponentKind.APM_AGENT;
import static org.elasticsearch.entitlement.runtime.policy.PolicyManager.ComponentKind.PLUGIN;
import static org.elasticsearch.entitlement.runtime.policy.PolicyManager.ComponentKind.SERVER;
import static org.elasticsearch.entitlement.runtime.policy.PolicyManager.ComponentKind.UNKNOWN;

/**
 * This class is the central authority for managing and enforcing entitlement policies at runtime.
 * It determines the component identity of a caller and checks its permissions against loaded policies.
 */
public class PolicyManager {
    /**
     * A general-purpose logger for operations where a specific module context is not available.
     */
    private static final Logger generalLogger = LogManager.getLogger(PolicyManager.class);

    static final Class<?> DEFAULT_FILESYSTEM_CLASS = PathUtils.getDefaultFileSystem().getClass();

    static final Set<String> MODULES_EXCLUDED_FROM_SYSTEM_MODULES = Set.of("java.desktop");

    /**
     * Identifies a particular entitlement {@link Scope} within a {@link Policy}, linking it to a
     * specific component and module.
     */
    public record PolicyScope(ComponentKind kind, String componentName, String moduleName) {
        public PolicyScope {
            requireNonNull(kind);
            requireNonNull(componentName);
            requireNonNull(moduleName);
            assert kind.componentName == null || kind.componentName.equals(componentName);
        }

        public static PolicyScope unknown(String moduleName) {
            return new PolicyScope(UNKNOWN, UNKNOWN.componentName, moduleName);
        }

        public static PolicyScope server(String moduleName) {
            return new PolicyScope(SERVER, SERVER.componentName, moduleName);
        }

        public static PolicyScope apmAgent(String moduleName) {
            return new PolicyScope(APM_AGENT, APM_AGENT.componentName, moduleName);
        }

        public static PolicyScope plugin(String componentName, String moduleName) {
            return new PolicyScope(PLUGIN, componentName, moduleName);
        }
    }

    /**
     * Enumerates the different kinds of components recognized by the policy manager.
     */
    public enum ComponentKind {
        UNKNOWN("(unknown)"),
        SERVER("(server)"),
        APM_AGENT("(APM agent)"),
        PLUGIN(null);

        /**
         * A fixed name for singleton components like SERVER, or null for kinds like PLUGIN
         * which can have multiple instances.
         */
        final String componentName;

        ComponentKind(String componentName) {
            this.componentName = componentName;
        }
    }

    /**
     * A container for the resolved entitlements of a specific module.
     * It includes entitlements grouped by type and a specialized `FileAccessTree` for efficient
     * file permission checks. This record is cached to optimize performance.
     *
     * @param componentName The name of the component (e.g., "server", "my-plugin").
     * @param entitlementsByType A map of entitlement classes to a list of granted entitlements of that type.
     * @param fileAccess A tree structure for efficiently checking file path permissions.
     * @param logger A logger specific to the component and module.
     */
    record ModuleEntitlements(
        String componentName,
        Map<Class<? extends Entitlement>, List<Entitlement>> entitlementsByType,
        FileAccessTree fileAccess,
        Logger logger
    ) {

        ModuleEntitlements {
            entitlementsByType = Map.copyOf(entitlementsByType);
        }

        public boolean hasEntitlement(Class<? extends Entitlement> entitlementClass) {
            return entitlementsByType.containsKey(entitlementClass);
        }

        public <E extends Entitlement> Stream<E> getEntitlements(Class<E> entitlementClass) {
            var entitlements = entitlementsByType.get(entitlementClass);
            if (entitlements == null) {
                return Stream.empty();
            }
            return entitlements.stream().map(entitlementClass::cast);
        }
    }

    private FileAccessTree getDefaultFileAccess(Path componentPath) {
        return FileAccessTree.withoutExclusivePaths(FilesEntitlement.EMPTY, pathLookup, componentPath);
    }

    // Creates a default, empty set of entitlements for a component.
    ModuleEntitlements defaultEntitlements(String componentName, Path componentPath, String moduleName) {
        return new ModuleEntitlements(componentName, Map.of(), getDefaultFileAccess(componentPath), getLogger(componentName, moduleName));
    }

    // Creates a set of entitlements from a policy scope.
    ModuleEntitlements policyEntitlements(String componentName, Path componentPath, String moduleName, List<Entitlement> entitlements) {
        FilesEntitlement filesEntitlement = FilesEntitlement.EMPTY;
        for (Entitlement entitlement : entitlements) {
            if (entitlement instanceof FilesEntitlement) {
                filesEntitlement = (FilesEntitlement) entitlement;
            }
        }
        return new ModuleEntitlements(
            componentName,
            entitlements.stream().collect(groupingBy(Entitlement::getClass)),
            FileAccessTree.of(componentName, moduleName, filesEntitlement, pathLookup, componentPath, exclusivePaths),
            getLogger(componentName, moduleName)
        );
    }

    // A concurrent map to cache resolved entitlements for each module.
    final Map<Module, ModuleEntitlements> moduleEntitlementsMap = new ConcurrentHashMap<>();

    private final Map<String, List<Entitlement>> serverEntitlements;
    private final List<Entitlement> apmAgentEntitlements;
    private final Map<String, Map<String, List<Entitlement>>> pluginsEntitlements;
    private final Function<Class<?>, PolicyScope> scopeResolver;
    private final PathLookup pathLookup;
    private final Set<Class<?>> mutedClasses;

    public static final String ALL_UNNAMED = "ALL-UNNAMED";

    // A set of modules considered part of the trusted "system" layer (e.g., core Java modules).
    private static final Set<Module> SYSTEM_LAYER_MODULES = findSystemLayerModules();

    private static Set<Module> findSystemLayerModules() {
        var systemModulesDescriptors = ModuleFinder.ofSystem()
            .findAll()
            .stream()
            .map(ModuleReference::descriptor)
            .collect(Collectors.toUnmodifiableSet());
        return Stream.concat(
            Stream.of(PolicyManager.class.getModule()),
            ModuleLayer.boot()
                .modules()
                .stream()
                .filter(
                    m -> systemModulesDescriptors.contains(m.getDescriptor())
                        && MODULES_EXCLUDED_FROM_SYSTEM_MODULES.contains(m.getName()) == false
                )
        ).collect(Collectors.toUnmodifiableSet());
    }

    // Modules in the boot layer that are not part of the system layer are considered server-level modules.
    public static final Set<Module> SERVER_LAYER_MODULES = ModuleLayer.boot()
        .modules()
        .stream()
        .filter(m -> SYSTEM_LAYER_MODULES.contains(m) == false)
        .collect(Collectors.toUnmodifiableSet());

    private final Map<String, Path> sourcePaths;

    private final Module entitlementsModule;

    private final List<ExclusivePath> exclusivePaths;

    public PolicyManager(
        Policy serverPolicy,
        List<Entitlement> apmAgentEntitlements,
        Map<String, Policy> pluginPolicies,
        Function<Class<?>, PolicyScope> scopeResolver,
        Map<String, Path> sourcePaths,
        Module entitlementsModule,
        PathLookup pathLookup,
        Set<Class<?>> suppressFailureLogClasses
    ) {
        this.serverEntitlements = buildScopeEntitlementsMap(requireNonNull(serverPolicy));
        this.apmAgentEntitlements = apmAgentEntitlements;
        this.pluginsEntitlements = requireNonNull(pluginPolicies).entrySet()
            .stream()
            .collect(toUnmodifiableMap(Map.Entry::getKey, e -> buildScopeEntitlementsMap(e.getValue())));
        this.scopeResolver = scopeResolver;
        this.sourcePaths = sourcePaths;
        this.entitlementsModule = entitlementsModule;
        this.pathLookup = requireNonNull(pathLookup);
        this.mutedClasses = suppressFailureLogClasses;

        List<ExclusiveFileEntitlement> exclusiveFileEntitlements = new ArrayList<>();
        for (var e : serverEntitlements.entrySet()) {
            validateEntitlementsPerModule(SERVER.componentName, e.getKey(), e.getValue(), exclusiveFileEntitlements);
        }
        validateEntitlementsPerModule(APM_AGENT.componentName, ALL_UNNAMED, apmAgentEntitlements, exclusiveFileEntitlements);
        for (var p : pluginsEntitlements.entrySet()) {
            for (var m : p.getValue().entrySet()) {
                validateEntitlementsPerModule(p.getKey(), m.getKey(), m.getValue(), exclusiveFileEntitlements);
            }
        }
        List<ExclusivePath> exclusivePaths = FileAccessTree.buildExclusivePathList(exclusiveFileEntitlements, pathLookup);
        FileAccessTree.validateExclusivePaths(exclusivePaths);
        this.exclusivePaths = exclusivePaths;
    }

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
        for (var e : entitlements) {
            if (found.contains(e.getClass())) {
                throw new IllegalArgumentException(
                    "[" + componentName + "] using module [" + moduleName + "] found duplicate entitlement [" + e.getClass().getName() + "]"
                );
            }
            found.add(e.getClass());
            if (e instanceof FilesEntitlement fe) {
                exclusiveFileEntitlements.add(new ExclusiveFileEntitlement(componentName, moduleName, fe));
            }
        }
    }

    public void checkStartProcess(Class<?> callerClass) {
        neverEntitled(callerClass, () -> "start process");
    }

    public void checkWriteStoreAttributes(Class<?> callerClass) {
        neverEntitled(callerClass, () -> "change file store attributes");
    }

    public void checkReadStoreAttributes(Class<?> callerClass) {
        checkEntitlementPresent(callerClass, ReadStoreAttributesEntitlement.class);
    }

    private void neverEntitled(Class<?> callerClass, Supplier<String> operationDescription) {
        var requestingClass = requestingClass(callerClass);
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

    public void checkExitVM(Class<?> callerClass) {
        checkEntitlementPresent(callerClass, ExitVMEntitlement.class);
    }

    public void checkCreateClassLoader(Class<?> callerClass) {
        checkEntitlementPresent(callerClass, CreateClassLoaderEntitlement.class);
    }

    public void checkSetHttpsConnectionProperties(Class<?> callerClass) {
        checkEntitlementPresent(callerClass, SetHttpsConnectionPropertiesEntitlement.class);
    }

    public void checkChangeJVMGlobalState(Class<?> callerClass) {
        neverEntitled(callerClass, () -> walkStackForCheckMethodName().orElse("change JVM global state"));
    }

    public void checkLoggingFileHandler(Class<?> callerClass) {
        neverEntitled(callerClass, () -> walkStackForCheckMethodName().orElse("create logging file handler"));
    }

    private Optional<String> walkStackForCheckMethodName() {
        return StackWalker.getInstance()
            .walk(
                frames -> frames.map(StackFrame::getMethodName)
                    .dropWhile(not(methodName -> methodName.startsWith(InstrumentationService.CHECK_METHOD_PREFIX)))
                    .findFirst()
            )
            .map(this::operationDescription);
    }

    public void checkChangeNetworkHandling(Class<?> callerClass) {
        checkChangeJVMGlobalState(callerClass);
    }

    public void checkChangeFilesHandling(Class<?> callerClass) {
        checkChangeJVMGlobalState(callerClass);
    }

    @SuppressForbidden(reason = "Explicitly checking File apis")
    public void checkFileRead(Class<?> callerClass, File file) {
        checkFileRead(callerClass, file.toPath());
    }

    private static boolean isPathOnDefaultFilesystem(Path path) {
        var pathFileSystemClass = path.getFileSystem().getClass();
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

    public void checkFileRead(Class<?> callerClass, Path path, boolean followLinks) throws NoSuchFileException {
        if (isPathOnDefaultFilesystem(path) == false) {
            return;
        }
        var requestingClass = requestingClass(callerClass);
        if (isTriviallyAllowed(requestingClass)) {
            return;
        }

        ModuleEntitlements entitlements = getEntitlements(requestingClass);

        Path realPath = null;
        boolean canRead = entitlements.fileAccess().canRead(path);
        if (canRead && followLinks) {
            try {
                realPath = path.toRealPath();
                if (realPath.equals(path) == false) {
                    canRead = entitlements.fileAccess().canRead(realPath);
                }
            } catch (NoSuchFileException e) {
                throw e; // rethrow
            } catch (IOException e) {
                canRead = false;
            }
        }

        if (canRead == false) {
            notEntitled(
                Strings.format(
                    "component [%s], module [%s], class [%s], entitlement [file], operation [read], path [%s]",
                    entitlements.componentName(),
                    getModuleName(requestingClass),
                    requestingClass,
                    realPath == null ? path : Strings.format("%s -> %s", path, realPath)
                ),
                callerClass,
                entitlements
            );
        }
    }

    @SuppressForbidden(reason = "Explicitly checking File apis")
    public void checkFileWrite(Class<?> callerClass, File file) {
        checkFileWrite(callerClass, file.toPath());
    }

    public void checkFileWrite(Class<?> callerClass, Path path) {
        if (isPathOnDefaultFilesystem(path) == false) {
            return;
        }
        var requestingClass = requestingClass(callerClass);
        if (isTriviallyAllowed(requestingClass)) {
            return;
        }

        ModuleEntitlements entitlements = getEntitlements(requestingClass);
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

    public void checkCreateTempFile(Class<?> callerClass) {
        checkFileWrite(callerClass, pathLookup.getBaseDirPaths(TEMP).findFirst().get());
    }

    @SuppressForbidden(reason = "Explicitly checking File apis")
    public void checkFileWithZipMode(Class<?> callerClass, File file, int zipMode) {
        assert zipMode == OPEN_READ || zipMode == (OPEN_READ | OPEN_DELETE);
        if ((zipMode & OPEN_DELETE) == OPEN_DELETE) {
            checkFileWrite(callerClass, file);
        } else {
            checkFileRead(callerClass, file);
        }
    }

    public void checkFileDescriptorRead(Class<?> callerClass) {
        neverEntitled(callerClass, () -> "read file descriptor");
    }

    public void checkFileDescriptorWrite(Class<?> callerClass) {
        neverEntitled(callerClass, () -> "write file descriptor");
    }

    public void checkGetFileAttributeView(Class<?> callerClass) {
        neverEntitled(callerClass, () -> "get file attribute view");
    }

    public void checkLoadingNativeLibraries(Class<?> callerClass) {
        checkEntitlementPresent(callerClass, LoadNativeLibrariesEntitlement.class);
    }

    private String operationDescription(String methodName) {
        return methodName.substring(methodName.indexOf('$'));
    }

    public void checkInboundNetworkAccess(Class<?> callerClass) {
        checkEntitlementPresent(callerClass, InboundNetworkEntitlement.class);
    }

    public void checkOutboundNetworkAccess(Class<?> callerClass) {
        checkEntitlementPresent(callerClass, OutboundNetworkEntitlement.class);
    }

    public void checkAllNetworkAccess(Class<?> callerClass) {
        var requestingClass = requestingClass(callerClass);
        if (isTriviallyAllowed(requestingClass)) {
            return;
        }

        var classEntitlements = getEntitlements(requestingClass);
        checkFlagEntitlement(classEntitlements, InboundNetworkEntitlement.class, requestingClass, callerClass);
        checkFlagEntitlement(classEntitlements, OutboundNetworkEntitlement.class, requestingClass, callerClass);
    }

    public void checkUnsupportedURLProtocolConnection(Class<?> callerClass, String protocol) {
        neverEntitled(callerClass, () -> Strings.format("unsupported URL protocol [%s]", protocol));
    }

    private void checkFlagEntitlement(
        ModuleEntitlements classEntitlements,
        Class<? extends Entitlement> entitlementClass,
        Class<?> requestingClass,
        Class<?> callerClass
    ) {
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

    public void checkWriteProperty(Class<?> callerClass, String property) {
        var requestingClass = requestingClass(callerClass);
        if (isTriviallyAllowed(requestingClass)) {
            return;
        }

        ModuleEntitlements entitlements = getEntitlements(requestingClass);
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

    private void notEntitled(String message, Class<?> callerClass, ModuleEntitlements entitlements) {
        var exception = new NotEntitledException(message);
        if (mutedClasses.contains(callerClass) == false) {
            entitlements.logger().warn("Not entitled: {}", message, exception);
        }
        throw exception;
    }

    private static Logger getLogger(String componentName, String moduleName) {
        var loggerSuffix = "." + componentName + "." + ((moduleName == null) ? ALL_UNNAMED : moduleName);
        return MODULE_LOGGERS.computeIfAbsent(PolicyManager.class.getName() + loggerSuffix, LogManager::getLogger);
    }

    private static final ConcurrentHashMap<String, Logger> MODULE_LOGGERS = new ConcurrentHashMap<>();

    public void checkManageThreadsEntitlement(Class<?> callerClass) {
        checkEntitlementPresent(callerClass, ManageThreadsEntitlement.class);
    }

    private void checkEntitlementPresent(Class<?> callerClass, Class<? extends Entitlement> entitlementClass) {
        var requestingClass = requestingClass(callerClass);
        if (isTriviallyAllowed(requestingClass)) {
            return;
        }
        checkFlagEntitlement(getEntitlements(requestingClass), entitlementClass, requestingClass, callerClass);
    }

    ModuleEntitlements getEntitlements(Class<?> requestingClass) {
        return moduleEntitlementsMap.computeIfAbsent(requestingClass.getModule(), m -> computeEntitlements(requestingClass));
    }

    private ModuleEntitlements computeEntitlements(Class<?> requestingClass) {
        var policyScope = scopeResolver.apply(requestingClass);
        var componentName = policyScope.componentName();
        var moduleName = policyScope.moduleName();

        switch (policyScope.kind()) {
            case SERVER -> {
                return getModuleScopeEntitlements(
                    serverEntitlements,
                    moduleName,
                    SERVER.componentName,
                    getComponentPathFromClass(requestingClass)
                );
            }
            case APM_AGENT -> {
                return policyEntitlements(
                    APM_AGENT.componentName,
                    getComponentPathFromClass(requestingClass),
                    ALL_UNNAMED,
                    apmAgentEntitlements
                );
            }
            case UNKNOWN -> {
                return defaultEntitlements(UNKNOWN.componentName, null, moduleName);
            }
            default -> {
                assert policyScope.kind() == PLUGIN;
                var pluginEntitlements = pluginsEntitlements.get(componentName);
                if (pluginEntitlements == null) {
                    return defaultEntitlements(componentName, sourcePaths.get(componentName), moduleName);
                } else {
                    return getModuleScopeEntitlements(pluginEntitlements, moduleName, componentName, sourcePaths.get(componentName));
                }
            }
        }
    }

    static Path getComponentPathFromClass(Class<?> requestingClass) {
        var codeSource = requestingClass.getProtectionDomain().getCodeSource();
        if (codeSource == null) {
            return null;
        }
        try {
            return Paths.get(codeSource.getLocation().toURI());
        } catch (Exception e) {
            generalLogger.info(
                "Cannot get component path for [{}]: [{}] cannot be converted to a valid Path",
                requestingClass.getName(),
                codeSource.getLocation().toString()
            );
            return null;
        }
    }

    private ModuleEntitlements getModuleScopeEntitlements(
        Map<String, List<Entitlement>> scopeEntitlements,
        String scopeName,
        String componentName,
        Path componentPath
    ) {
        var entitlements = scopeEntitlements.get(scopeName);
        if (entitlements == null) {
            return defaultEntitlements(componentName, componentPath, scopeName);
        }
        return policyEntitlements(componentName, componentPath, scopeName, entitlements);
    }

    /**
     * Walks the stack to identify the first class outside of the entitlement library, which is
     * considered the "requesting class" for an entitlement check.
     *
     * @param callerClass A potential fast-path if the caller is already known.
     * @return The Class that is requesting the privileged operation.
     */
    Class<?> requestingClass(Class<?> callerClass) {
        if (callerClass != null) {
            return callerClass;
        }
        Optional<Class<?>> result = StackWalker.getInstance(RETAIN_CLASS_REFERENCE)
            .walk(frames -> findRequestingFrame(frames).map(StackFrame::getDeclaringClass));
        return result.orElse(null);
    }

    /**
     * Finds the first relevant stack frame for an entitlement check.
     */
    Optional<StackFrame> findRequestingFrame(Stream<StackFrame> frames) {
        return frames.filter(f -> f.getDeclaringClass().getModule() != entitlementsModule) // ignore entitlements library
            .skip(1) // Skip the sensitive caller method
            .findFirst();
    }

    /**
     * Determines if a class is part of the trusted system layer, in which case it is
     * granted all permissions by default.
     * @return true if the class belongs to a system module.
     */
    private static boolean isTriviallyAllowed(Class<?> requestingClass) {
        if (generalLogger.isTraceEnabled()) {
            generalLogger.trace("Stack trace for upcoming trivially-allowed check", new Exception());
        }
        if (requestingClass == null) {
            generalLogger.debug("Entitlement trivially allowed: no caller frames outside the entitlement library");
            return true;
        }
        if (requestingClass == NO_CLASS) {
            generalLogger.debug("Entitlement trivially allowed from outermost frame");
            return true;
        }
        if (SYSTEM_LAYER_MODULES.contains(requestingClass.getModule())) {
            generalLogger.debug("Entitlement trivially allowed from system module [{}]", requestingClass.getModule().getName());
            return true;
        }
        generalLogger.trace("Entitlement not trivially allowed");
        return false;
    }

    private static String getModuleName(Class<?> requestingClass) {
        String name = requestingClass.getModule().getName();
        return (name == null) ? ALL_UNNAMED : name;
    }

    @Override
    public String toString() {
        return "PolicyManager{" + "serverEntitlements=" + serverEntitlements + ", pluginsEntitlements=" + pluginsEntitlements + '}';
    }
}
