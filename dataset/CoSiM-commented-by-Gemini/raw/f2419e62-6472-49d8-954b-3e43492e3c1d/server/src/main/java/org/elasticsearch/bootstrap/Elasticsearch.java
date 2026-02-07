/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.bootstrap;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configurator;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.Build;
import org.elasticsearch.ElasticsearchException;
import org.elasticsearch.ReleaseVersions;
import org.elasticsearch.action.support.SubscribableListener;
import org.elasticsearch.common.ReferenceDocs;
import org.elasticsearch.common.io.stream.InputStreamStreamInput;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.common.network.IfConfig;
import org.elasticsearch.common.settings.SecureSettings;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.transport.BoundTransportAddress;
import org.elasticsearch.common.util.concurrent.RunOnce;
import org.elasticsearch.core.AbstractRefCounted;
import org.elasticsearch.core.CheckedConsumer;
import org.elasticsearch.core.IOUtils;
import org.elasticsearch.core.SuppressForbidden;
import org.elasticsearch.entitlement.bootstrap.EntitlementBootstrap;
import org.elasticsearch.entitlement.runtime.api.NotEntitledException;
import org.elasticsearch.entitlement.runtime.policy.Policy;
import org.elasticsearch.entitlement.runtime.policy.PolicyManager;
import org.elasticsearch.entitlement.runtime.policy.PolicyUtils;
import org.elasticsearch.entitlement.runtime.policy.entitlements.LoadNativeLibrariesEntitlement;
import org.elasticsearch.env.Environment;
import org.elasticsearch.index.IndexVersion;
import org.elasticsearch.jdk.JarHell;
import org.elasticsearch.monitor.jvm.HotThreads;
import org.elasticsearch.monitor.jvm.JvmInfo;
import org.elasticsearch.monitor.os.OsProbe;
import org.elasticsearch.monitor.process.ProcessProbe;
import org.elasticsearch.nativeaccess.NativeAccess;
import org.elasticsearch.node.Node;
import org.elasticsearch.node.NodeValidationException;
import org.elasticsearch.plugins.PluginBundle;
import org.elasticsearch.plugins.PluginsLoader;

import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.Security;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.elasticsearch.nativeaccess.WindowsFunctions.ConsoleCtrlHandler.CTRL_CLOSE_EVENT;

/**
 * This class starts elasticsearch.
 */
/**
 * @brief This class is the main entry point for starting, initializing, and shutting down an Elasticsearch node.
 * It orchestrates the multi-phase bootstrapping process, including security setup, logging configuration,
 * plugin loading, native resource initialization, and node lifecycle management.
 */
class Elasticsearch {

    private static final String POLICY_PATCH_PREFIX = "es.entitlements.policy.";
    private static final String SERVER_POLICY_PATCH_NAME = POLICY_PATCH_PREFIX + "server";
    private static final String APM_AGENT_PACKAGE_NAME = "co.elastic.apm.agent";

    /**
     * @brief Main entry point for starting Elasticsearch.
     * This method initiates the bootstrapping process through three distinct phases:
     * Phase 1: Basic setup and logging initialization.
     * Phase 2: Security manager, environment, and native resource initialization.
     * Phase 3: Node construction, startup, and readiness signaling.
     * @param args Command line arguments passed to the application.
     */
    public static void main(final String[] args) {

        Bootstrap bootstrap = initPhase1();
        assert bootstrap != null;

        try {
            initPhase2(bootstrap);
            initPhase3(bootstrap);
        } catch (NodeValidationException e) {
            // Block Logic: Handles exceptions specifically related to node validation, exiting gracefully.
            bootstrap.exitWithNodeValidationException(e);
        } catch (Throwable t) {
            // Block Logic: Catches any other unexpected exceptions during initialization and exits.
            bootstrap.exitWithUnknownException(t);
        }
    }

    /**
     * @brief Retrieves the standard error stream.
     * This method is marked with `@SuppressForbidden` as it directly accesses `System.err`,
     * which is typically restricted in some security contexts, but is necessary for
     * communication with the server-cli.
     * @return The `PrintStream` representing the standard error stream.
     */
    @SuppressForbidden(reason = "grab stderr for communication with server-cli")
    private static PrintStream getStderr() {
        return System.err;
    }

    /**
     * @brief Retrieves the standard output stream.
     * This method is marked with `@SuppressForbidden` as it directly accesses `System.out`,
     * which is typically restricted in some security contexts, but is necessary for
     * communication with the server-cli (and also temporarily for debugging).
     * @return The `PrintStream` representing the standard output stream.
     */
    // TODO: remove this, just for debugging
    @SuppressForbidden(reason = "grab stdout for communication with server-cli")
    private static PrintStream getStdout() {
        return System.out;
    }

    /**
     * @brief Executes the first phase of the Elasticsearch process initialization.
     * Phase 1 involves static initialization, reading command-line arguments, and configuring logging.
     * Minimal operations are performed here to ensure logging is set up as the final step.
     * @return A `Bootstrap` object containing initial setup information and command-line arguments.
     */
    private static Bootstrap initPhase1() {
        final PrintStream out = getStdout();
        final PrintStream err = getStderr();
        final ServerArgs args;

        try {
            initSecurityProperties(); // Initialize Java security properties
            LogConfigurator.registerErrorListener(); // Register a listener for logging errors

            BootstrapInfo.init(); // Initialize bootstrap information

            // Block Logic: Reads server arguments from standard input. System.in is not closed as it's used later.
            var in = new InputStreamStreamInput(System.in);
            args = new ServerArgs(in);

            // Functional Utility: Creates an initial Environment instance using node settings and config directory.
            Environment nodeEnv = new Environment(args.nodeSettings(), args.configDir());

            BootstrapInfo.setConsole(ConsoleLoader.loadConsole(nodeEnv)); // Set console for bootstrap info

            // DO NOT MOVE THIS
            // Block Logic: Logging configuration must be the very last step of Phase 1.
            // Any initialization requiring logging should occur in subsequent phases.
            LogConfigurator.setNodeName(Node.NODE_NAME_SETTING.get(args.nodeSettings()));
            LogConfigurator.configure(nodeEnv, args.quiet() == false);
        } catch (Throwable t) {
            // Block Logic: Catches any exceptions occurring during this early phase, prints to stderr, and exits the process.
            t.printStackTrace(err);
            err.flush();
            Bootstrap.exit(1); // mimic JDK exit code on exception
            return null; // unreachable, to satisfy compiler
        }

        return new Bootstrap(out, err, args);
    }

    /**
     * @brief Executes the second phase of the Elasticsearch process initialization.
     * Phase 2 focuses on everything leading up to and including the security manager initialization.
     * This involves setting up the environment, PID file, uncaught exception handler,
     * native resource initialization, and preparing for entitlements and plugin loading.
     * @param bootstrap The `Bootstrap` object containing the current bootstrapping state.
     * @throws IOException If an I/O error occurs during file operations.
     */
    private static void initPhase2(Bootstrap bootstrap) throws IOException {
        final ServerArgs args = bootstrap.args();
        final SecureSettings secrets = args.secrets();
        bootstrap.setSecureSettings(secrets);
        Environment nodeEnv = createEnvironment(args.configDir(), args.nodeSettings(), secrets);
        bootstrap.setEnvironment(nodeEnv);

        initPidFile(args.pidFile()); // Initialize PID file

        // Block Logic: Installs a default uncaught exception handler for threads.
        // This is crucial to prevent unhandled exceptions from crashing the process
        // and must be done before security manager initialization.
        Thread.setDefaultUncaughtExceptionHandler(new ElasticsearchUncaughtExceptionHandler());

        bootstrap.spawner().spawnNativeControllers(nodeEnv); // Spawn native controllers

        nodeEnv.validateNativesConfig(); // Validate native configurations (e.g., temporary directories for JNA)
        initializeNatives(
            nodeEnv.tmpDir(),
            BootstrapSettings.MEMORY_LOCK_SETTING.get(args.nodeSettings()),
            true, // always install system call filters, not user-configurable since 8.0.0
            BootstrapSettings.CTRLHANDLER_SETTING.get(args.nodeSettings())
        );

        // Block Logic: Initializes system probes (process, OS, JVM) before the security manager is installed
        // to collect initial system information.
        initializeProbes();

        // Functional Utility: Registers a shutdown hook to gracefully shut down Elasticsearch on JVM exit.
        Runtime.getRuntime().addShutdownHook(new Thread(Elasticsearch::shutdown, "elasticsearch-shutdown"));

        // Block Logic: Checks for "jar hell" (conflicting JAR files) to ensure a clean classpath.
        final Logger logger = LogManager.getLogger(JarHell.class);
        JarHell.checkJarHell(logger::debug);

        // Functional Utility: Logs network interface configuration before the SecurityManager is installed.
        IfConfig.logIfNecessary();

        // Block Logic: Eagerly initializes critical classes to avoid permissions issues or
        // ensure static initialization completes before SecurityManager is fully active.
        ensureInitialized(
            ReleaseVersions.class, // Ensures ReleaseVersions' static initialization
            ReferenceDocs.class,   // Ensures ReferenceDocs' static initialization
            AbstractRefCounted.class, // Uses MethodHandles.lookup during initialization
            SubscribableListener.class, // Uses MethodHandles.lookup during initialization
            RunOnce.class,         // Uses MethodHandles.lookup during initialization
            VectorUtil.class       // Eagerly initializes for log4j permissions & JDK-8309727 workaround
        );

        // Block Logic: Loads plugin Java modules and layers, preparing them for entitlement checks.
        var modulesBundles = PluginsLoader.loadModulesBundles(nodeEnv.modulesDir());
        var pluginsBundles = PluginsLoader.loadPluginsBundles(nodeEnv.pluginsDir());

        final PluginsLoader pluginsLoader;

        LogManager.getLogger(Elasticsearch.class).info("Bootstrapping Entitlements");

        // Functional Utility: Collects plugin data necessary for policy creation.
        var pluginData = Stream.concat(
            modulesBundles.stream()
                .map(bundle -> new PolicyUtils.PluginData(bundle.getDir(), bundle.pluginDescriptor().isModular(), false)),
            pluginsBundles.stream().map(bundle -> new PolicyUtils.PluginData(bundle.getDir(), bundle.pluginDescriptor().isModular(), true))
        ).toList();

        // Functional Utility: Collects and parses policy patches for plugins and the server.
        var pluginPolicyPatches = collectPluginPolicyPatches(modulesBundles, pluginsBundles, logger);
        var pluginPolicies = PolicyUtils.createPluginPolicies(pluginData, pluginPolicyPatches, Build.current().version());
        var serverPolicyPatch = PolicyUtils.parseEncodedPolicyIfExists(
            System.getProperty(SERVER_POLICY_PATCH_NAME),
            Build.current().version(),
            false,
            "server",
            PolicyManager.SERVER_LAYER_MODULES.stream().map(Module::getName).collect(Collectors.toUnmodifiableSet())
        );

        pluginsLoader = PluginsLoader.createPluginsLoader(modulesBundles, pluginsBundles, findPluginsWithNativeAccess(pluginPolicies));

        // Block Logic: Bootstraps the entitlement system, applying policies for various components and plugins.
        var scopeResolver = ScopeResolver.create(pluginsLoader.pluginLayers(), APM_AGENT_PACKAGE_NAME);
        Map<String, Collection<Path>> pluginSourcePaths = Stream.concat(modulesBundles.stream(), pluginsBundles.stream())
            .collect(Collectors.toUnmodifiableMap(bundle -> bundle.pluginDescriptor().getName(), bundle -> List.of(bundle.getDir())));
        EntitlementBootstrap.bootstrap(
            serverPolicyPatch,
            pluginPolicies,
            scopeResolver::resolveClassToScope,
            nodeEnv.settings()::getValues,
            nodeEnv.dataDirs(),
            nodeEnv.repoDirs(),
            nodeEnv.configDir(),
            nodeEnv.libDir(),
            nodeEnv.modulesDir(),
            nodeEnv.pluginsDir(),
            pluginSourcePaths,
            nodeEnv.logsDir(),
            nodeEnv.tmpDir(),
            args.pidFile(),
            Set.of(EntitlementSelfTester.class.getPackage())
        );
        EntitlementSelfTester.entitlementSelfTest(); // Self-test the entitlement system

        bootstrap.setPluginsLoader(pluginsLoader);
    }

    /**
     * @brief Collects and parses policy patches for plugins from system properties.
     * This method scans system properties for keys prefixed with `POLICY_PATCH_PREFIX`
     * (excluding the server's own policy patch) and extracts them. It also logs warnings
     * for any policy patches found for unknown plugins.
     * @param modulesBundles A set of `PluginBundle`s representing loaded modules.
     * @param pluginsBundles A set of `PluginBundle`s representing loaded plugins.
     * @param logger The `Logger` instance for logging warnings.
     * @return A `Map` where keys are plugin names (derived from the policy patch prefix)
     *         and values are the policy patch strings.
     */
    private static Map<String, String> collectPluginPolicyPatches(
        Set<PluginBundle> modulesBundles,
        Set<PluginBundle> pluginsBundles,
        Logger logger
    ) {
        var policyPatches = new HashMap<String, String>();
        var systemProperties = BootstrapInfo.getSystemProperties();
        systemProperties.keys().asIterator().forEachRemaining(key -> {
            var value = systemProperties.get(key);
            // Block Logic: Filters system properties for valid policy patch entries, ensuring they start with the prefix
            // and are not the server's own policy patch.
            if (key instanceof String k
                && value instanceof String v
                && k.startsWith(POLICY_PATCH_PREFIX)
                && k.equals(SERVER_POLICY_PATCH_NAME) == false) {
                policyPatches.put(k.substring(POLICY_PATCH_PREFIX.length()), v);
            }
        });
        // Functional Utility: Collects all known plugin names for validation purposes.
        var pluginNames = Stream.concat(modulesBundles.stream(), pluginsBundles.stream())
            .map(bundle -> bundle.pluginDescriptor().getName())
            .collect(Collectors.toUnmodifiableSet());

        // Block Logic: Warns if a collected policy patch refers to a plugin that is not known to the system.
        for (var patchedPluginName : policyPatches.keySet()) {
            if (pluginNames.contains(patchedPluginName) == false) {
                logger.warn(
                    "Found command-line policy patch for unknown plugin [{}] (available plugins: [{}])",
                    patchedPluginName,
                    String.join(", ", pluginNames)
                );
            }
        }
        return policyPatches;
    }

    /**
     * @brief Internal helper class for self-testing entitlement protection.
     * This class ensures that the entitlement system correctly prevents unauthorized process creation.
     */
    private static class EntitlementSelfTester {
        /**
         * @brief Performs a self-test of the entitlement system by attempting to start processes.
         * It verifies that process creation is correctly denied by the entitlement protection.
         * This method must reside outside the core entitlements library.
         */
        private static void entitlementSelfTest() {
            ensureCannotStartProcess(ProcessBuilder::start); // Test direct process start
            ensureCannotStartProcess(EntitlementSelfTester::reflectiveStartProcess); // Test reflective process start
        }

        /**
         * @brief Verifies that a process cannot be started due to entitlement protection.
         * @param startProcess A `CheckedConsumer` representing a method to attempt process creation.
         * @throws IllegalStateException If the entitlement protection incorrectly permits process creation.
         */
        private static void ensureCannotStartProcess(CheckedConsumer<ProcessBuilder, ?> startProcess) {
            try {
                // The command doesn't matter; it doesn't even need to exist
                startProcess.accept(new ProcessBuilder(""));
            } catch (NotEntitledException e) {
                return; // Expected exception, entitlement worked
            } catch (Exception e) {
                // Block Logic: Catches unexpected exceptions during process start attempt, indicating a test failure.
                throw new IllegalStateException("Failed entitlement protection self-test", e);
            }
            // Block Logic: If no exception was thrown, it means entitlement protection failed to deny process creation.
            throw new IllegalStateException("Entitlement protection self-test was incorrectly permitted");
        }

        /**
         * @brief Attempts to start a process using Java Reflection.
         * This method is used to test if entitlement protection is effective against reflective calls.
         * @param pb The `ProcessBuilder` instance to use for starting the process.
         * @throws Exception If an error occurs during reflection or process invocation.
         */
        private static void reflectiveStartProcess(ProcessBuilder pb) throws Exception {
            try {
                var start = ProcessBuilder.class.getMethod("start");
                start.invoke(pb);
            } catch (InvocationTargetException e) {
                // Functional Utility: Unwraps the cause of `InvocationTargetException` to get the actual exception.
                throw (Exception) e.getCause();
            }
        }
    }

    /**
     * @brief Ensures that a set of classes are eagerly initialized.
     * This is often used to work around Java platform module system permissions
     * or to ensure static initializers run before security restrictions are fully applied.
     * @param classes A variable argument list of `Class<?>` objects to be initialized.
     * @throws AssertionError If an `IllegalAccessException` occurs during initialization,
     *                         which should be unexpected for public classes.
     */
    private static void ensureInitialized(Class<?>... classes) {
        for (final var clazz : classes) {
            try {
                MethodHandles.publicLookup().ensureInitialized(clazz);
            } catch (IllegalAccessException unexpected) {
                // Block Logic: Re-throws unexpected `IllegalAccessException` as an `AssertionError`,
                // as public classes should be accessible for initialization.
                throw new AssertionError(unexpected);
            }
        }
    }

    /**
     * @brief Executes the third and final phase of Elasticsearch initialization.
     * This phase occurs after the security manager is initialized and can involve
     * multithreading, logging, and operations subject to security policies.
     * At the end of this phase, the system is fully ready to accept requests.
     * @param bootstrap The `Bootstrap` object containing the current bootstrapping state.
     * @throws IOException If an I/O error occurs (e.g., filesystem, network).
     * @throws NodeValidationException If the node fails to start due to configuration issues.
     */
    private static void initPhase3(Bootstrap bootstrap) throws IOException, NodeValidationException {
        checkLucene(); // Verify Lucene version compatibility

        // Block Logic: Creates a new Node instance, overriding `validateNodeBeforeAcceptingRequests`
        // to perform bootstrap checks.
        Node node = new Node(bootstrap.environment(), bootstrap.pluginsLoader()) {
            @Override
            protected void validateNodeBeforeAcceptingRequests(
                final BootstrapContext context,
                final BoundTransportAddress boundTransportAddress,
                List<BootstrapCheck> checks
            ) throws NodeValidationException {
                BootstrapChecks.check(context, boundTransportAddress, checks);
            }
        };
        INSTANCE = new Elasticsearch(bootstrap.spawner(), node); // Set the static INSTANCE of Elasticsearch

        // Functional Utility: Secure settings must be read during node construction and then closed.
        IOUtils.close(bootstrap.secureSettings());

        INSTANCE.start(); // Start the Elasticsearch node instance

        // Block Logic: If running in daemonize mode, remove the console appender to prevent console output.
        if (bootstrap.args().daemonize()) {
            LogConfigurator.removeConsoleAppender();
        }

        // DO NOT MOVE THIS
        // Block Logic: Signals to the parent CLI process that the server is ready.
        // This is a critical step and must be the last action before closing streams in daemon mode.
        bootstrap.sendCliMarker(BootstrapInfo.SERVER_READY_MARKER);
        if (bootstrap.args().daemonize()) {
            bootstrap.closeStreams(); // Close streams in daemon mode
        } else {
            startCliMonitorThread(System.in); // Start CLI monitor thread for non-daemonized mode
        }
    }

    /**
     * @brief Initializes native resources and sets up various operating system-level configurations.
     * This includes checking for root privileges, installing system call filters, memory locking,
     * and registering a console control handler for graceful shutdown on Windows.
     * @param tmpFile The temporary directory path.
     * @param mlockAll Whether or not to lock the process's memory to prevent swapping.
     * @param systemCallFilter Whether or not to install system call filters (e.g., seccomp).
     * @param ctrlHandler Whether or not to install the console control handler (Windows specific) for shutdown events.
     */
    static void initializeNatives(final Path tmpFile, final boolean mlockAll, final boolean systemCallFilter, final boolean ctrlHandler) {
        final Logger logger = LogManager.getLogger(Elasticsearch.class);
        var nativeAccess = NativeAccess.instance();

        // Block Logic: Prevents Elasticsearch from running as root for security reasons.
        if (nativeAccess.definitelyRunningAsRoot()) {
            throw new RuntimeException("can not run elasticsearch as root");
        }

        // Block Logic: Installs system call filters (e.g., seccomp sandbox) if requested, enhancing security.
        if (systemCallFilter) {
            /*
             * Try to install system call filters; if they fail to install; a bootstrap check will fail startup in production mode.
             *
             * TODO: should we fail hard here if system call filters fail to install, or remain lenient in non-production environments?
             */
            nativeAccess.tryInstallExecSandbox();
        }

        // Block Logic: Locks the process's memory into RAM if `mlockAll` is true, preventing it from being swapped to disk.
        if (mlockAll) {
            nativeAccess.tryLockMemory();
        }

        // Block Logic: For Windows, registers a console control handler to gracefully shut down Elasticsearch
        // if a close event (e.g., user closing the console window) is received.
        if (ctrlHandler) {
            var windowsFunctions = nativeAccess.getWindowsFunctions();
            if (windowsFunctions != null) {
                windowsFunctions.addConsoleCtrlHandler(code -> {
                    if (CTRL_CLOSE_EVENT == code) {
                        logger.info("running graceful exit on windows");
                        shutdown();
                        return true;
                    }
                    return false;
                });
            }
        }

        // Functional Utility: Initializes Lucene's random seed, potentially using /dev/urandom for higher quality entropy.
        StringHelper.randomId();
    }

    /**
     * @brief Initializes various system probes to collect diagnostic information.
     * This includes probes for process, operating system, and JVM metrics,
     * as well as runtime monitoring for hot threads.
     */
    static void initializeProbes() {
        // Functional Utility: Force initialization of ProcessProbe to collect process-specific metrics.
        ProcessProbe.getInstance();
        // Functional Utility: Force initialization of OsProbe to collect operating system-specific metrics.
        OsProbe.getInstance();
        // Functional Utility: Force initialization of JvmInfo to collect JVM-specific metrics.
        JvmInfo.jvmInfo();
        // Functional Utility: Initializes runtime monitoring for detecting "hot" (busy) threads within the JVM.
        HotThreads.initializeRuntimeMonitoring();
    }

    /**
     * @brief Checks for compatibility between Elasticsearch's required Lucene version and the currently loaded Lucene version.
     * If there is a mismatch, an `AssertionError` is thrown, indicating a critical configuration problem.
     * @throws AssertionError If the Lucene version required by Elasticsearch does not match the Lucene version found at runtime.
     */
    static void checkLucene() {
        if (IndexVersion.current().luceneVersion().equals(org.apache.lucene.util.Version.LATEST) == false) {
            // Block Logic: Throws an AssertionError if the current Lucene version does not match the expected latest version.
            throw new AssertionError(
                "Lucene version mismatch this version of Elasticsearch requires lucene version ["
                    + IndexVersion.current().luceneVersion()
                    + "]  but the current lucene version is ["
                    + org.apache.lucene.util.Version.LATEST
                    + "]"
            );
        }
    }

    /**
     * @brief Starts a daemon thread to monitor `stdin` for a shutdown signal from the parent CLI process.
     * This mechanism enables graceful shutdown of Elasticsearch when managed externally.
     * @param stdin The `InputStream` representing the standard input for this process,
     *              used to receive the shutdown signal.
     */
    private static void startCliMonitorThread(InputStream stdin) {
        new Thread(() -> {
            int msg = -1;
            try {
                msg = stdin.read();
            } catch (IOException e) {
                // Block Logic: Catches IOException but effectively ignores it, as either end-of-stream (-1)
                // or an error will lead to a shutdown decision.
                // ignore, whether we cleanly got end of stream (-1) or an error, we will shut down below
            } finally {
                // Block Logic: Interprets the received message from stdin.
                // A specific marker signifies a clean shutdown request (exit 0),
                // while any other state (e.g., pipe broken, parent died) results in an error exit (exit 1).
                if (msg == BootstrapInfo.SERVER_SHUTDOWN_MARKER) {
                    Bootstrap.exit(0);
                } else {
                    // parent process died or there was an error reading from it
                    Bootstrap.exit(1);
                }
            }
        }, "elasticsearch-cli-monitor-thread").start();
    }

    /**
     * @brief Initializes and manages the PID file for the Elasticsearch process.
     * If a `pidFile` path is provided, the current process ID is written to it.
     * A shutdown hook is registered to ensure the PID file is deleted upon normal system exit.
     * @param pidFile A `Path` to the PID file, or `null` if no PID file should be written.
     * @throws IOException If an I/O error occurs during PID file creation or directory creation.
     */
    private static void initPidFile(Path pidFile) throws IOException {
        if (pidFile == null) {
            return;
        }
        // Block Logic: Registers a shutdown hook to delete the PID file when the JVM exits.
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                Files.deleteIfExists(pidFile);
            } catch (IOException e) {
                throw new ElasticsearchException("Failed to delete pid file " + pidFile, e);
            }
        }, "elasticsearch[pidfile-cleanup]"));

        // Pre-condition: `pidFile` must be an absolute path for `pidFile.getParent()` to work correctly.
        assert pidFile.isAbsolute();

        // Block Logic: Ensures that the parent directories for the PID file exist; creates them if they don't.
        if (Files.exists(pidFile.getParent()) == false) {
            Files.createDirectories(pidFile.getParent());
        }

        // Functional Utility: Writes the current process's PID to the specified PID file.
        Files.writeString(pidFile, Long.toString(ProcessHandle.current().pid()));
    }

    /**
     * @brief Initializes and overrides specific Java security properties from system properties.
     * This method customizes network address caching settings (`networkaddress.cache.ttl` and `networkaddress.cache.negative.ttl`)
     * if corresponding `es.` prefixed system properties are defined.
     * It ensures the override values are valid integers.
     * @throws IllegalArgumentException If an override property's value cannot be parsed as an integer.
     */
    private static void initSecurityProperties() {
        // Block Logic: Iterates over network address caching properties to allow overriding their default values via system properties.
        for (final String property : new String[] { "networkaddress.cache.ttl", "networkaddress.cache.negative.ttl" }) {
            final String overrideProperty = "es." + property;
            final String overrideValue = System.getProperty(overrideProperty);
            if (overrideValue != null) {
                try {
                    // Functional Utility: Converts the override value to an integer and back to a string to validate its format.
                    Security.setProperty(property, Integer.toString(Integer.valueOf(overrideValue)));
                } catch (final NumberFormatException e) {
                    // Block Logic: Throws an IllegalArgumentException if the override value is not a valid integer.
                    throw new IllegalArgumentException("failed to parse [" + overrideProperty + "] with value [" + overrideValue + "]", e);
                }
            }
        }
    }

    /**
     * @brief Creates an Elasticsearch `Environment` instance.
     * This method constructs an `Environment` object using initial settings,
     * a configuration directory, and optional secure settings.
     * @param configDir The configuration directory for the Elasticsearch node.
     * @param initialSettings Initial `Settings` for the environment.
     * @param secureSettings Optional `SecureSettings` to be applied to the environment.
     * @return A new `Environment` instance.
     */
    private static Environment createEnvironment(Path configDir, Settings initialSettings, SecureSettings secureSettings) {
        Settings.Builder builder = Settings.builder();
        builder.put(initialSettings);
        if (secureSettings != null) {
            builder.setSecureSettings(secureSettings);
        }
        return new Environment(builder.build(), configDir);
    }

    /**
     * @brief Identifies which plugins, based on provided policies, require native library access.
     * This method iterates through the entitlements defined in each plugin's policy
     * to determine if the `LoadNativeLibrariesEntitlement` is present.
     * @param policies A map of plugin names to their associated `Policy` objects.
     * @return A `Map` where keys are plugin names and values are a `Set` of module names
     *         within that plugin that require native access.
     */
    static Map<String, Set<String>> findPluginsWithNativeAccess(Map<String, Policy> policies) {
        Map<String, Set<String>> pluginsWithNativeAccess = new HashMap<>();
        // Block Logic: Iterates through each plugin's policy entries.
        for (var kv : policies.entrySet()) {
            // Block Logic: For each policy, examines its scopes and their associated entitlements.
            for (var scope : kv.getValue().scopes()) {
                // Invariant: Checks if any entitlement within the current scope is of type `LoadNativeLibrariesEntitlement`.
                if (scope.entitlements().stream().anyMatch(entitlement -> entitlement instanceof LoadNativeLibrariesEntitlement)) {
                    // Functional Utility: Adds the module name to the set of modules requiring native access for the current plugin.
                    var modulesToEnable = pluginsWithNativeAccess.computeIfAbsent(kv.getKey(), k -> new HashSet<>());
                    modulesToEnable.add(scope.moduleName());
                }
            }
        }
        return pluginsWithNativeAccess;
    }

    // -- instance

    private static volatile Elasticsearch INSTANCE;

    private final Spawner spawner;
    private final Node node;
    private final CountDownLatch keepAliveLatch = new CountDownLatch(1);
    private final Thread keepAliveThread;

    /**
     * @brief Constructs an `Elasticsearch` instance.
     * This private constructor initializes the spawner and node components,
     * and sets up a `keepAliveThread` to prevent the JVM from exiting prematurely.
     * @param spawner The `Spawner` instance for managing native processes.
     * @param node The `Node` instance representing the Elasticsearch node.
     */
    private Elasticsearch(Spawner spawner, Node node) {
        this.spawner = Objects.requireNonNull(spawner);
        this.node = Objects.requireNonNull(node);
        // Functional Utility: The `keepAliveThread` waits on `keepAliveLatch` indefinitely,
        // ensuring the JVM remains active until `keepAliveLatch` is counted down during shutdown.
        this.keepAliveThread = new Thread(() -> {
            try {
                keepAliveLatch.await();
            } catch (InterruptedException e) {
                // Block Logic: If the thread is interrupted, it bails out, allowing the JVM to exit.
                // bail out
            }
        }, "elasticsearch[keepAlive/" + Build.current().version() + "]");
    }

    /**
     * @brief Starts the Elasticsearch node and the associated keep-alive thread.
     * @throws NodeValidationException If the node fails validation during startup.
     */
    private void start() throws NodeValidationException {
        node.start(); // Start the core Elasticsearch node functionality
        keepAliveThread.start(); // Start the thread that keeps the JVM alive
    }

    /**
     * @brief Initiates a graceful shutdown of the Elasticsearch node.
     * This static method marks the process as stopping, prepares the node for closure,
     * attempts to close associated resources, and waits for the node to fully stop.
     * It also shuts down the logging context and releases the keep-alive latch.
     */
    private static void shutdown() {
        ElasticsearchProcess.markStopping(); // Mark the Elasticsearch process as stopping

        // Block Logic: If the instance was never fully initialized, return early.
        if (INSTANCE == null) {
            return; // never got far enough
        }
        var es = INSTANCE;
        try {
            es.node.prepareForClose(); // Prepare the node for closing
            IOUtils.close(es.node, es.spawner); // Close node and spawner resources
            // Block Logic: Waits for the node to close within a specified timeout.
            // If the timeout is exceeded, an `IllegalStateException` is thrown.
            if (es.node.awaitClose(10, TimeUnit.SECONDS) == false) {
                throw new IllegalStateException(
                    "Node didn't stop within 10 seconds. " + "Any outstanding requests or tasks might get killed."
                );
            }
        } catch (IOException ex) {
            // Block Logic: Catches I/O errors during shutdown and re-throws as an ElasticsearchException.
            throw new ElasticsearchException("Failure occurred while shutting down node", ex);
        } catch (InterruptedException e) {
            // Block Logic: Catches InterruptedException, logs a warning, and re-interrupts the current thread.
            LogManager.getLogger(Elasticsearch.class).warn("Thread got interrupted while waiting for the node to shutdown.");
            Thread.currentThread().interrupt();
        } finally {
            // Block Logic: Ensures the logging context is shut down and the keep-alive latch is released.
            LoggerContext context = (LoggerContext) LogManager.getContext(false);
            Configurator.shutdown(context);

            es.keepAliveLatch.countDown(); // Release the keep-alive thread
        }
    }
}
