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
import org.elasticsearch.entitlement.bridge.EntitlementChecker;
import org.elasticsearch.entitlement.runtime.policy.ElasticsearchEntitlementChecker;
import org.elasticsearch.entitlement.runtime.policy.PathLookup;
import org.elasticsearch.entitlement.runtime.policy.PolicyChecker;
import org.elasticsearch.entitlement.runtime.policy.PolicyCheckerImpl;
import org.elasticsearch.entitlement.runtime.policy.PolicyManager;

import java.lang.instrument.Instrumentation;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.Set;

import static java.util.Objects.requireNonNull;

/**
 * @brief Serves as the central class for initializing the Elasticsearch entitlement system via a Java agent.
 *
 * This class is invoked reflectively by the Java agent's `agentmain` method.
 * Its primary responsibilities include:
 * <ol>
 *     <li>Configuring the core entitlement components.</li>
 *     <li>Instantiating and making the {@link EntitlementChecker} accessible.</li>
 *     <li>Installing the {@link org.elasticsearch.entitlement.instrumentation.Instrumenter}
 *         to initiate bytecode instrumentation.</li>
 * </ol>
 *
 * Functional Utility: This is the crucial link between the static bootstrap process
 *                     and the dynamic, runtime enforcement of entitlement policies
 *                     through bytecode manipulation. It ensures that the entitlement
 *                     framework is correctly set up within the JVM.
 * Architecture: Operates in conjunction with a Java agent to dynamically modify
 *               application bytecode, allowing for fine-grained access control
 *               without altering the original source code.
 * Lifecycle: Relies on `initializeArgs` to pass necessary configuration from the
 *            bootstrapping application to the agent's static context.
 */
public class EntitlementInitialization {

    private static final Module ENTITLEMENTS_MODULE = PolicyManager.class.getModule();

    /**
     * @brief Holds the arguments passed during the entitlement system's initialization.
     * Functional Utility: This static field acts as a bridge to pass necessary configuration
     *                     data from the {@link EntitlementBootstrap} (application thread)
     *                     to the static context of the agent's {@link #initialize(Instrumentation)} method.
     * Pre-condition: Must be set by {@link org.elasticsearch.entitlement.bootstrap.EntitlementBootstrap#bootstrap}
     *                before the agent is loaded.
     */
    public static InitializeArgs initializeArgs;
    /**
     * @brief The singleton instance of {@link ElasticsearchEntitlementChecker} used for runtime entitlement checks.
     * Functional Utility: This checker is referenced by all instrumented methods to enforce access policies.
     * Pre-condition: Must be initialized by {@link #initialize(Instrumentation)}.
     */
    private static ElasticsearchEntitlementChecker checker;

    /**
     * @brief Provides the singleton {@link EntitlementChecker} instance.
     * @return The active {@link EntitlementChecker} instance.
     * Functional Utility: This method is referenced reflectively by the instrumented methods
     *                     to obtain the checker and perform entitlement evaluations.
     * Pre-condition: The {@link EntitlementChecker} must have been previously initialized
     *                by calling {@link #initialize(Instrumentation)}.
     */
    // Note: referenced by bridge reflectively
    public static EntitlementChecker checker() {
        return checker;
    }

    /**
     * @brief Initializes the entire entitlement system within the JVM.
     * <ol>
     * <li>Initializes dynamic instrumentation via {@link DynamicInstrumentation#initialize}</li>
     * <li>Creates the {@link PolicyManager}</li>
     * <li>Creates the {@link ElasticsearchEntitlementChecker} instance referenced by the instrumented methods</li>
     * </ol>
     * Functional Utility: This method is the core setup routine for the Java agent, ensuring
     *                     that the {@link EntitlementChecker} is prepared and bytecode instrumentation
     *                     is activated to enforce policies.
     * Pre-condition: {@link #initializeArgs} must be set by {@link org.elasticsearch.entitlement.bootstrap.EntitlementBootstrap#bootstrap}.
     * Post-condition: The {@link EntitlementChecker} is instantiated, and bytecode instrumentation is enabled.
     *
     * @param inst The JVM {@link Instrumentation} class instance, provided by the Java agent.
     * @throws Exception if any error occurs during initialization.
     * <strong>NOTE:</strong> this method is referenced by the agent reflectively.
     */
    public static void initialize(Instrumentation inst) throws Exception {
        /**
         * Block Logic: Initializes the static `checker` instance using the policy manager from `initializeArgs`.
         * Functional Utility: Sets up the central entitlement checking component before any bytecode is instrumented,
         *                     ensuring that the checker is available when instrumented methods are invoked.
         * Pre-condition: `initializeArgs` is not null and contains a valid `policyManager`.
         * Invariant: The `checker` instance is set before `initInstrumentation` is called.
         */
        // the checker _MUST_ be set before _any_ instrumentation is done
        checker = initChecker(initializeArgs.policyManager());
        /**
         * Block Logic: Initializes the dynamic bytecode instrumentation process.
         * Functional Utility: Activates the mechanism that modifies class bytecode to inject calls
         *                     to the {@link EntitlementChecker} at method entry points.
         * Pre-condition: The `checker` instance is already initialized.
         */
        initInstrumentation(inst);
    }

    /**
     * @brief A record to encapsulate the arguments required for {@link #initialize(Instrumentation)}.
     * Functional Utility: This record serves as a container to safely pass critical initialization data
     *                     from the main application bootstrap to the static context of the Java agent's
     *                     initialization method.
     * Invariants: All fields (`pathLookup`, `suppressFailureLogPackages`, `policyManager`) are guaranteed to be non-null.
     *
     * @param pathLookup An instance of {@link PathLookup} for resolving file system paths.
     * @param suppressFailureLogPackages A {@link Set} of packages for which entitlement failure logging should be suppressed.
     * @param policyManager The {@link PolicyManager} instance responsible for managing entitlement policies.
     */
    public record InitializeArgs(PathLookup pathLookup, Set<Package> suppressFailureLogPackages, PolicyManager policyManager) {
        /**
         * Block Logic: Ensures that all required initialization arguments are non-null.
         * Functional Utility: Provides immediate validation of critical inputs during construction,
         *                     preventing NullPointerExceptions in later stages of entitlement setup.
         * Invariant: An {@link NullPointerException} is thrown if any of the core arguments are null.
         */
        public InitializeArgs {
            requireNonNull(pathLookup);
            requireNonNull(suppressFailureLogPackages);
            requireNonNull(policyManager);
        }
    }

    /**
     * @brief Creates a {@link PolicyCheckerImpl} instance.
     * @param policyManager The {@link PolicyManager} responsible for providing entitlement policies.
     * @return A new {@link PolicyCheckerImpl} configured with the necessary suppression packages,
     *         entitlements module, policy manager, and path lookup.
     * Functional Utility: Instantiates the component that actually evaluates entitlement requests
     *                     against the loaded policies, acting as the decision-making engine for access control.
     */
    private static PolicyCheckerImpl createPolicyChecker(PolicyManager policyManager) {
        return new PolicyCheckerImpl(
            initializeArgs.suppressFailureLogPackages(),
            ENTITLEMENTS_MODULE,
            policyManager,
            initializeArgs.pathLookup()
        );
    }

    /**
     * @brief Ensures that specific classes sensitive to bytecode verification are initialized before instrumentation.
     * Functional Utility: Addresses potential {@link ClassCircularityError}s or other verification issues
     *                     that can arise during dynamic bytecode transformation. By forcing early loading
     *                     of these classes, it establishes a partial order, mitigating complex verification
     *                     scenarios.
     * Rationale: When bytecode is transformed, verification might trigger the loading of other classes. If
     *            those classes are also targets for transformation and verification, a circular dependency
     *            or unexpected verification failures can occur. Pre-initializing critical classes helps
     *            to stabilize this process.
     * Pre-condition: Bytecode verification (`es.entitlements.verify_bytecode`) is enabled.
     * Post-condition: The specified classes are loaded, potentially simplifying subsequent bytecode transformations.
     */
    private static void ensureClassesSensitiveToVerificationAreInitialized() {
        var classesToInitialize = Set.of(
            "sun.net.www.protocol.http.HttpURLConnection",
            "sun.nio.ch.SocketChannelImpl",
            "java.net.ProxySelector",
            "sun.nio.ch.DatagramChannelImpl",
            "sun.nio.ch.ServerSocketChannelImpl"
        );
        /**
         * Block Logic: Iterates through a predefined set of class names and attempts to load each class.
         * Functional Utility: Forces the JVM to load and initialize these classes early, thus avoiding
         *                     potential circularity or race conditions during subsequent bytecode verification
         *                     and instrumentation.
         * Pre-condition: `classesToInitialize` contains fully qualified names of classes.
         * Invariant: An {@link AssertionError} is thrown if any of the specified classes cannot be found,
         *            indicating an unexpected environment or configuration.
         */
        for (String className : classesToInitialize) {
            try {
                Class.forName(className);
            } catch (ClassNotFoundException unexpected) {
                throw new AssertionError(unexpected);
            }
        }
    }

    static ElasticsearchEntitlementChecker initChecker(PolicyManager policyManager) {
        /**
         * Block Logic: Creates an instance of {@link PolicyChecker} based on the provided {@link PolicyManager}.
         * Functional Utility: This checker will be used by the {@link ElasticsearchEntitlementChecker} to evaluate
         *                     individual entitlement requests against the loaded policies.
         * Pre-condition: `policyManager` is fully initialized with all relevant policies.
         * Post-condition: A {@link PolicyChecker} instance is available for policy evaluation.
         */
        final PolicyChecker policyChecker = createPolicyChecker(policyManager);
        /**
         * Block Logic: Dynamically determines the version-specific {@link ElasticsearchEntitlementChecker} implementation class.
         * Functional Utility: Allows the entitlement system to adapt to different JVM versions by loading
         *                     the correct checker implementation, ensuring compatibility.
         * Pre-condition: `ElasticsearchEntitlementChecker` is a base class or interface for version-specific implementations.
         * Invariant: The returned `clazz` is a concrete implementation suitable for the current Java runtime.
         */
        final Class<?> clazz = EntitlementCheckerUtils.getVersionSpecificCheckerClass(
            ElasticsearchEntitlementChecker.class,
            Runtime.version().feature()
        );

        Constructor<?> constructor;
        /**
         * Block Logic: Retrieves the constructor of the determined checker class that accepts a {@link PolicyChecker}.
         * Functional Utility: Prepares for the instantiation of the version-specific checker, ensuring
         *                     it can be properly initialized with its policy evaluation logic.
         * Pre-condition: `clazz` is a valid class, and it must have a constructor accepting a single `PolicyChecker` argument.
         * Invariant: An {@link AssertionError} is thrown if the required constructor is not found.
         */
        try {
            constructor = clazz.getConstructor(PolicyChecker.class);
        } catch (NoSuchMethodException e) {
            throw new AssertionError("entitlement impl is missing required constructor: [" + clazz.getName() + "]", e);
        }

        ElasticsearchEntitlementChecker checker;
        /**
         * Block Logic: Instantiates the {@link ElasticsearchEntitlementChecker} using the retrieved constructor and policy checker.
         * Functional Utility: Creates the concrete object that will perform all runtime entitlement checks.
         * Pre-condition: `constructor` is accessible and `policyChecker` is a valid argument.
         * Invariant: An {@link AssertionError} is thrown if instantiation fails for any reflective reason.
         */
        try {
            checker = (ElasticsearchEntitlementChecker) constructor.newInstance(policyChecker);
        } catch (IllegalAccessException | InvocationTargetException | InstantiationException e) {
            throw new AssertionError(e);
        }

        return checker;
    }

    static void initInstrumentation(Instrumentation instrumentation) throws Exception {
        /**
         * Block Logic: Reads the system property `es.entitlements.verify_bytecode` to determine if bytecode verification is enabled.
         * Functional Utility: Allows an administrator to enable or disable post-instrumentation bytecode verification,
         *                     providing a trade-off between strict code integrity checks and potential performance overhead.
         * Pre-condition: The system property may or may not be set; defaults to `false`.
         * Invariant: `verifyBytecode` accurately reflects the user's preference for bytecode verification.
         */
        var verifyBytecode = Booleans.parseBoolean(System.getProperty("es.entitlements.verify_bytecode", "false"));
        /**
         * Block Logic: Conditionally calls `ensureClassesSensitiveToVerificationAreInitialized()` if bytecode verification is enabled.
         * Functional Utility: Addresses potential issues with class loading and verification order during instrumentation
         *                     by preemptively initializing known problematic classes.
         * Pre-condition: `verifyBytecode` is true.
         */
        if (verifyBytecode) {
            ensureClassesSensitiveToVerificationAreInitialized();
        }

        /**
         * Block Logic: Initializes the {@link DynamicInstrumentation} component.
         * Functional Utility: Activates the bytecode transformation process using the provided {@link Instrumentation}
         *                     instance and the version-specific {@link EntitlementChecker}, optionally with bytecode verification.
         * Pre-condition: `instrumentation` is a valid JVM instrumentation instance, and `EntitlementCheckerUtils` is functional.
         * Post-condition: The JVM is set up to apply entitlement checks through bytecode modification.
         */
        DynamicInstrumentation.initialize(
            instrumentation,
            EntitlementCheckerUtils.getVersionSpecificCheckerClass(EntitlementChecker.class, Runtime.version().feature()),
            verifyBytecode
        );

    }
}
