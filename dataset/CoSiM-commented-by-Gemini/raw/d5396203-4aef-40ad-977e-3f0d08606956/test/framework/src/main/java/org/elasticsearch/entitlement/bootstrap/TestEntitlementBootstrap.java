/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

/**
 * @file TestEntitlementBootstrap.java
 * @brief Contains the bootstrap logic for activating the entitlement system
 *        within a testing environment.
 */

package org.elasticsearch.entitlement.bootstrap;

import org.elasticsearch.entitlement.initialization.TestEntitlementInitialization;
import org.elasticsearch.entitlement.runtime.policy.PathLookup;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;

import java.nio.file.Path;
import java.util.stream.Stream;

/**
 * @class TestEntitlementBootstrap
 * @brief A utility class responsible for initializing the entitlement agent
 *        for use in tests.
 *
 * This class provides a static method to load and activate the Java agent
 * that enforces entitlement checks, configuring it with test-specific
 * implementations.
 */
public class TestEntitlementBootstrap {

    private static final Logger logger = LogManager.getLogger(TestEntitlementBootstrap.class);

    /**
     * @brief Activates entitlement checking in tests.
     *
     * This method sets up the necessary arguments for entitlement initialization
     * using a test-specific `PathLookup` and then dynamically loads the
     * entitlement Java agent to instrument the code.
     */
    public static void bootstrap() {
        // Pre-configure the initialization process with a stubbed path lookup
        // suitable for the test environment.
        TestEntitlementInitialization.initializeArgs = new TestEntitlementInitialization.InitializeArgs(new TestPathLookup());
        logger.debug("Loading entitlement agent");
        // Dynamically load the Java agent to enable entitlement enforcement.
        EntitlementBootstrap.loadAgent(EntitlementBootstrap.findAgentJar(), TestEntitlementInitialization.class.getName());
    }

    /**
     * @record TestPathLookup
     * @brief A test-specific stub implementation of the `PathLookup` interface.
     *
     * Functional Utility: This record serves as a null implementation where
     * path resolution is not required for the specific test scenarios. Its
     * methods are designed to fail loudly by throwing an exception if they are
     * ever called, ensuring that tests do not unknowingly rely on them.
     */
    private record TestPathLookup() implements PathLookup {
        @Override
        public Path pidFile() {
            throw notYetImplemented();
        }

        @Override
        public Stream<Path> getBaseDirPaths(BaseDir baseDir) {
            throw notYetImplemented();
        }

        @Override
        public Stream<Path> resolveSettingPaths(BaseDir baseDir, String settingName) {
            throw notYetImplemented();
        }

        private static IllegalStateException notYetImplemented() {
            return new IllegalStateException("not yet implemented");
        }

    }
}
