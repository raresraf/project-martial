/**
 * @file EntitlementInitialization.java
 * @brief Initializes the Elasticsearch entitlement system by dynamically instrumenting the JVM.
 *
 * @details
 * This class is the core of the entitlement agent's initialization process. It is called
 * reflectively by the agent's `agentmain` method after being attached to the running JVM.
 * Its primary responsibility is to set up and activate the runtime security checks that enforce
 * entitlements based on predefined policies.
 *
 * The initialization process involves several key steps:
 * 1.  **Policy Management**: It creates a `PolicyManager` that consolidates security policies from
 *     the server, plugins, and modules. This includes defining default entitlements for core
 *     Java and Elasticsearch components (e.g., file access, network operations, thread management).
 * 2.  **Instrumentation Setup**: It identifies the set of methods across the JDK and Elasticsearch
 *     that need to be instrumented for security checks. This is a crucial step for intercepting
 *     potentially sensitive operations.
 * 3.  **Dynamic Implementation Lookup**: For platform-dependent APIs (like `FileSystemProvider`), it
 *     dynamically discovers the concrete implementation class at runtime and targets its methods
 *     for instrumentation.
 * 4.  **Bytecode Transformation**: It uses the Java Instrumentation API (`java.lang.instrument`) to
 *     install a `ClassFileTransformer`. This transformer injects bytecode at class load time,
 *     inserting calls to the `EntitlementChecker` before the original method bodies.
 * 5.  **Re-transformation**: It re-transforms classes that were already loaded before the agent was
 *     attached, ensuring that entitlement checks are applied retroactively to the entire application.
 *
 * This mechanism allows for a flexible and powerful security model that can be applied to a running
 * application without modifying its source code, enforcing fine-grained access control based on
 * the entitlements granted to different parts of the system.
 */
/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
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
 * This class is the main entry point for the entitlement agent. It is called via reflection
 * during the `agentmain` phase to configure the entitlement system, instantiate the
 * {@link EntitlementChecker}, and install the necessary bytecode instrumentation.
 */
public class EntitlementInitialization {

    private static final Module ENTITLEMENTS_MODULE = PolicyManager.class.getModule();

    private static ElasticsearchEntitlementChecker manager;

    @FunctionalInterface
    interface InstrumentationInfoFactory {
        InstrumentationService.InstrumentationInfo of(String methodName, Class<?>... parameterTypes) throws ClassNotFoundException,
            NoSuchMethodException;
    }

    /**
     * Provides access to the initialized entitlement checker instance.
     * This method is referenced reflectively by the agent bridge.
     */
    public static EntitlementChecker checker() {
        return manager;
    }

    /**
     * Initializes the entire entitlement system. This method orchestrates the creation of the policy manager,
     * identification of methods to instrument, and the application of bytecode transformations.
     *
     * @param inst The {@link Instrumentation} instance provided by the JVM.
     * @throws Exception if any part of the initialization fails.
     */
    public static void initialize(Instrumentation inst) throws Exception {
        manager = initChecker();

        var latestCheckerInterface = getVersionSpecificCheckerClass(EntitlementChecker.class, Runtime.version().feature());
        var verifyBytecode = Booleans.parseBoolean(System.getProperty("es.entitlements.verify_bytecode", "false"));

        if (verifyBytecode) {
            ensureClassesSensitiveToVerificationAreInitialized();
        }

        // Block Logic: Gathers all methods that require security checks. This includes methods from platform-dependent
        // implementations (like file system providers) which are resolved dynamically at runtime.
        Map<MethodKey, CheckMethod> checkMethods = new HashMap<>(INSTRUMENTATION_SERVICE.lookupMethods(latestCheckerInterface));
        Stream.of(
            fileSystemProviderChecks(),
            fileStoreChecks(),
            pathChecks(),
            Stream.of(
                INSTRUMENTATION_SERVICE.lookupImplementationMethod(
                    SelectorProvider.class,
                    "inheritedChannel",
                    SelectorProvider.provider().getClass(),
                    EntitlementChecker.class,
                    "checkSelectorProviderInheritedChannel"
                )
            )
        )
            .flatMap(Function.identity())
            .forEach(instrumentation -> checkMethods.put(instrumentation.targetMethod(), instrumentation.checkMethod()));

        var classesToTransform = checkMethods.keySet().stream().map(MethodKey::className).collect(Collectors.toSet());

        // Create and install the transformer that will inject the entitlement checks.
        Instrumenter instrumenter = INSTRUMENTATION_SERVICE.newInstrumenter(latestCheckerInterface, checkMethods);
        var transformer = new Transformer(instrumenter, classesToTransform, verifyBytecode);
        inst.addTransformer(transformer, true);

        // Block Logic: Re-transforms all classes that were already loaded before the agent was attached.
        // This ensures that the security policy is applied comprehensively to the entire application.
        var classesToRetransform = findClassesToRetransform(inst.getAllLoadedClasses(), classesToTransform);
        try {
            inst.retransformClasses(classesToRetransform);
        } catch (VerifyError e) {
            // If re-transformation fails, enable detailed verification and retry class by class for diagnostics.
            transformer.enableClassVerification();
            for (var classToRetransform : classesToRetransform) {
                inst.retransformClasses(classToRetransform);
            }
            throw e;
        }
    }

    private static Class<?>[] findClassesToRetransform(Class<?>[] loadedClasses, Set<String> classesToTransform) {
        List<Class<?>> retransform = new ArrayList<>();
        for (Class<?> loadedClass : loadedClasses) {
            if (classesToTransform.contains(loadedClass.getName().replace(".", "/"))) {
                retransform.add(loadedClass);
            }
        }
        return retransform.toArray(new Class<?>[0]);
    }

    /**
     * Creates and configures the `PolicyManager`, which is responsible for loading and managing
     * all security policies from the server, plugins, and modules.
     * @return An initialized `PolicyManager`.
     */
    private static PolicyManager createPolicyManager() {
        EntitlementBootstrap.BootstrapArgs bootstrapArgs = EntitlementBootstrap.bootstrapArgs();
        Map<String, Policy> pluginPolicies = bootstrapArgs.pluginPolicies();
        PathLookup pathLookup = bootstrapArgs.pathLookup();

        // Block Logic: Defines the default security scopes and file access entitlements for core Elasticsearch components.
        List<Scope> serverScopes = new ArrayList<>();
        List<FileData> serverModuleFileDatas = new ArrayList<>();
        Collections.addAll(
            serverModuleFileDatas,
            // Defines base directory permissions
            FileData.ofBaseDirPath(PLUGINS, READ),
            FileData.ofBaseDirPath(MODULES, READ),
            FileData.ofBaseDirPath(CONFIG, READ),
            FileData.ofBaseDirPath(LOGS, READ_WRITE),
            FileData.ofBaseDirPath(LIB, READ),
            FileData.ofBaseDirPath(DATA, READ_WRITE),
            FileData.ofBaseDirPath(SHARED_REPO, READ_WRITE),
            // exclusive settings file
            FileData.ofRelativePath(Path.of("operator/settings.json"), CONFIG, READ_WRITE).withExclusive(true),

            // Platform-specific file permissions for Linux
            FileData.ofPath(Path.of("/etc/os-release"), READ).withPlatform(LINUX),
            FileData.ofPath(Path.of("/etc/system-release"), READ).withPlatform(LINUX),
            FileData.ofPath(Path.of("/usr/lib/os-release"), READ).withPlatform(LINUX),
            FileData.ofPath(Path.of("/proc/sys/vm/max_map_count"), READ).withPlatform(LINUX),
            FileData.ofPath(Path.of("/proc/meminfo"), READ).withPlatform(LINUX),
            FileData.ofPath(Path.of("/proc/loadavg"), READ).withPlatform(LINUX),
            FileData.ofPath(Path.of("/proc/self/cgroup"), READ).withPlatform(LINUX),
            FileData.ofPath(Path.of("/sys/fs/cgroup/"), READ).withPlatform(LINUX),
            FileData.ofPath(Path.of("/proc/self/mountinfo"), READ).withPlatform(LINUX),
            FileData.ofPath(Path.of("/proc/diskstats"), READ).withPlatform(LINUX)
        );
        if (pathLookup.pidFile() != null) {
            serverModuleFileDatas.add(FileData.ofPath(pathLookup.pidFile(), READ_WRITE));
        }

        // Defines default entitlements for core modules.
        Collections.addAll(
            serverScopes,
            new Scope(
                "org.elasticsearch.base",
                List.of(
                    new CreateClassLoaderEntitlement(),
                    new FilesEntitlement(
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
                    new ExitVMEntitlement(),
                    new ReadStoreAttributesEntitlement(),
                    new CreateClassLoaderEntitlement(),
                    new InboundNetworkEntitlement(),
                    new LoadNativeLibrariesEntitlement(),
                    new ManageThreadsEntitlement(),
                    new FilesEntitlement(serverModuleFileDatas)
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
            new Scope("org.apache.lucene.misc", List.of(new FilesEntitlement(List.of(FileData.ofBaseDirPath(DATA, READ_WRITE))))),
            new Scope(
                "org.apache.logging.log4j.core",
                List.of(new ManageThreadsEntitlement(), new FilesEntitlement(List.of(FileData.ofBaseDirPath(LOGS, READ_WRITE))))
            ),
            new Scope(
                "org.elasticsearch.nativeaccess",
                List.of(new LoadNativeLibrariesEntitlement(), new FilesEntitlement(List.of(FileData.ofBaseDirPath(DATA, READ_WRITE))))
            )
        );

        // Conditionally add FIPS-related entitlements if FIPS mode is enabled.
        if (Booleans.parseBoolean(System.getProperty("org.bouncycastle.fips.approved_only"), false)) {
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
                    List.of(new FilesEntitlement(List.of(FileData.ofBaseDirPath(LIB, READ))), new ManageThreadsEntitlement())
                )
            );
        }

        var serverPolicy = new Policy(
            "server",
            bootstrapArgs.serverPolicyPatch() == null
                ? serverScopes
                : PolicyUtils.mergeScopes(serverScopes, bootstrapArgs.serverPolicyPatch().scopes())
        );

        // A special case to grant entitlements to the APM agent, which runs in an unnamed module.
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
                    FileData.ofPath(Path.of("/proc/meminfo"), READ),
                    FileData.ofPath(Path.of("/sys/fs/cgroup/"), READ)
                )
            )
        );

        validateFilesEntitlements(pluginPolicies, pathLookup);

        return new PolicyManager(
            serverPolicy,
            agentEntitlements,
            pluginPolicies,
            EntitlementBootstrap.bootstrapArgs().scopeResolver(),
            EntitlementBootstrap.bootstrapArgs().sourcePaths(),
            ENTITLEMENTS_MODULE,
            pathLookup,
            bootstrapArgs.suppressFailureLogClasses()
        );
    }

    /**
     * Validates that plugin policies do not grant forbidden file access permissions.
     * @param pluginPolicies The policies to validate.
     * @param pathLookup The path lookup utility.
     */
    static void validateFilesEntitlements(Map<String, Policy> pluginPolicies, PathLookup pathLookup) {
        Set<Path> readAccessForbidden = new HashSet<>();
        pathLookup.getBaseDirPaths(PLUGINS).forEach(p -> readAccessForbidden.add(p.toAbsolutePath().normalize()));
        pathLookup.getBaseDirPaths(MODULES).forEach(p -> readAccessForbidden.add(p.toAbsolutePath().normalize()));
        pathLookup.getBaseDirPaths(LIB).forEach(p -> readAccessForbidden.add(p.toAbsolutePath().normalize()));
        Set<Path> writeAccessForbidden = new HashSet<>();
        pathLookup.getBaseDirPaths(CONFIG).forEach(p -> writeAccessForbidden.add(p.toAbsolutePath().normalize()));
        for (var pluginPolicy : pluginPolicies.entrySet()) {
            for (var scope : pluginPolicy.getValue().scopes()) {
                var filesEntitlement = scope.entitlements()
                    .stream()
                    .filter(x -> x instanceof FilesEntitlement)
                    .map(x -> ((FilesEntitlement) x))
                    .findFirst();
                if (filesEntitlement.isPresent()) {
                    var fileAccessTree = FileAccessTree.withoutExclusivePaths(filesEntitlement.get(), pathLookup, null);
                    validateReadFilesEntitlements(pluginPolicy.getKey(), scope.moduleName(), fileAccessTree, readAccessForbidden);
                    validateWriteFilesEntitlements(pluginPolicy.getKey(), scope.moduleName(), fileAccessTree, writeAccessForbidden);
                }
            }
        }
    }

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

    private static void validateReadFilesEntitlements(
        String componentName,
        String moduleName,
        FileAccessTree fileAccessTree,
        Set<Path> readForbiddenPaths
    ) {
        for (Path forbiddenPath : readForbiddenPaths) {
            if (fileAccessTree.canRead(forbiddenPath)) {
                throw buildValidationException(componentName, moduleName, forbiddenPath, READ);
            }
        }
    }

    private static void validateWriteFilesEntitlements(
        String componentName,
        String moduleName,
        FileAccessTree fileAccessTree,
        Set<Path> writeForbiddenPaths
    ) {
        for (Path forbiddenPath : writeForbiddenPaths) {
            if (fileAccessTree.canWrite(forbiddenPath)) {
                throw buildValidationException(componentName, moduleName, forbiddenPath, READ_WRITE);
            }
        }
    }

    private static Path getUserHome() {
        String userHome = System.getProperty("user.home");
        if (userHome == null) {
            throw new IllegalStateException("user.home system property is required");
        }
        return PathUtils.get(userHome);
    }

    /**
     * Dynamically finds and collects methods from the platform-specific `FileSystemProvider` implementation that need to be instrumented.
     * @return A stream of `InstrumentationInfo` objects for file system operations.
     */
    private static Stream<InstrumentationService.InstrumentationInfo> fileSystemProviderChecks() throws ClassNotFoundException,
        NoSuchMethodException {
        var fileSystemProviderClass = FileSystems.getDefault().provider().getClass();

        var instrumentation = new InstrumentationInfoFactory() {
            @Override
            public InstrumentationService.InstrumentationInfo of(String methodName, Class<?>... parameterTypes)
                throws ClassNotFoundException, NoSuchMethodException {
                return INSTRUMENTATION_SERVICE.lookupImplementationMethod(
                    FileSystemProvider.class,
                    methodName,
                    fileSystemProviderClass,
                    EntitlementChecker.class,
                    "check" + Character.toUpperCase(methodName.charAt(0)) + methodName.substring(1),
                    parameterTypes
                );
            }
        };

        var allVersionsMethods = Stream.of(
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
            instrumentation.of("setAttribute", Path.class, String.class, Object.class, LinkOption[].class)
        );

        if (Runtime.version().feature() >= 20) {
            var java20EntitlementCheckerClass = getVersionSpecificCheckerClass(EntitlementChecker.class, 20);
            var java20Methods = Stream.of(
                INSTRUMENTATION_SERVICE.lookupImplementationMethod(
                    FileSystemProvider.class,
                    "readAttributesIfExists",
                    fileSystemProviderClass,
                    java20EntitlementCheckerClass,
                    "checkReadAttributesIfExists",
                    Path.class,
                    Class.class,
                    LinkOption[].class
                ),
                INSTRUMENTATION_SERVICE.lookupImplementationMethod(
                    FileSystemProvider.class,
                    "exists",
                    fileSystemProviderClass,
                    java20EntitlementCheckerClass,
                    "checkExists",
                    Path.class,
                    LinkOption[].class
                )
            );
            return Stream.concat(allVersionsMethods, java20Methods);
        }
        return allVersionsMethods;
    }

    private static Stream<InstrumentationService.InstrumentationInfo> fileStoreChecks() {
        var fileStoreClasses = StreamSupport.stream(FileSystems.getDefault().getFileStores().spliterator(), false)
            .map(FileStore::getClass)
            .distinct();
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
                throw new RuntimeException(e);
            }
        });
    }

    private static Stream<InstrumentationService.InstrumentationInfo> pathChecks() {
        var pathClasses = StreamSupport.stream(FileSystems.getDefault().getRootDirectories().spliterator(), false)
            .map(Path::getClass)
            .distinct();
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
                throw new RuntimeException(e);
            }
        });
    }

    /**
     * Pre-loads certain sensitive classes if bytecode verification is enabled. This helps to avoid
     * circular class loading issues during the transformation and verification process.
     */
    private static void ensureClassesSensitiveToVerificationAreInitialized() {
        var classesToInitialize = Set.of(
            "sun.net.www.protocol.http.HttpURLConnection",
            "sun.nio.ch.SocketChannelImpl",
            "java.net.ProxySelector"
        );
        for (String className : classesToInitialize) {
            try {
                Class.forName(className);
            } catch (ClassNotFoundException unexpected) {
                throw new AssertionError(unexpected);
            }
        }
    }

    /**
     * Selects the appropriate version-specific `EntitlementChecker` class based on the current Java runtime version.
     * This allows for handling API changes across different JDK versions.
     */
    private static Class<?> getVersionSpecificCheckerClass(Class<?> baseClass, int javaVersion) {
        String packageName = baseClass.getPackageName();
        String baseClassName = baseClass.getSimpleName();

        final String classNamePrefix;
        if (javaVersion < 19) {
            classNamePrefix = "";
        } else if (javaVersion < 23) {
            classNamePrefix = "Java" + javaVersion;
        } else {
            classNamePrefix = "Java23";
        }

        final String className = packageName + "." + classNamePrefix + baseClassName;
        Class<?> clazz;
        try {
            clazz = Class.forName(className);
        } catch (ClassNotFoundException e) {
            throw new AssertionError("entitlement lib cannot find entitlement class " + className, e);
        }
        return clazz;
    }

    /**
     * Instantiates the version-specific `ElasticsearchEntitlementChecker` using the created `PolicyManager`.
     * @return An instance of the entitlement checker.
     */
    private static ElasticsearchEntitlementChecker initChecker() {
        final PolicyManager policyManager = createPolicyManager();

        final Class<?> clazz = getVersionSpecificCheckerClass(ElasticsearchEntitlementChecker.class, Runtime.version().feature());

        Constructor<?> constructor;
        try {
            constructor = clazz.getConstructor(PolicyManager.class);
        } catch (NoSuchMethodException e) {
            throw new AssertionError("entitlement impl is missing no arg constructor", e);
        }
        try {
            return (ElasticsearchEntitlementChecker) constructor.newInstance(policyManager);
        } catch (IllegalAccessException | InvocationTargetException | InstantiationException e) {
            throw new AssertionError(e);
        }
    }

    private static final InstrumentationService INSTRUMENTATION_SERVICE = new ProviderLocator<>(
        "entitlement",
        InstrumentationService.class,
        "org.elasticsearch.entitlement.instrumentation",
        Set.of()
    ).get();
}
