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
 * @brief Provides a bootstrap mechanism for entitlement checking within test environments.
 *
 * This class facilitates the activation of the entitlement agent specifically for testing purposes.
 * It initializes a test-specific entitlement configuration and loads the agent without
 * requiring actual system paths for PID files or base directories, ensuring a controlled
 * and isolated test execution.
 */
package org.elasticsearch.entitlement.bootstrap;

import org.elasticsearch.entitlement.initialization.TestEntitlementInitialization;
import org.elasticsearch.entitlement.runtime.policy.PathLookup;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;

import java.nio.file.Path;
import java.util.stream.Stream;

public class TestEntitlementBootstrap {

    private static final Logger logger = LogManager.getLogger(TestEntitlementBootstrap.class);

    /**
     * @brief Activates entitlement checking in tests.
     *
     * Functional Utility: This method initializes the entitlement framework with a test-specific
     * PathLookup implementation that returns null or empty streams for path-related queries.
     * It then proceeds to load the entitlement agent, ensuring that entitlement checks are
     * active within the test environment without relying on a real filesystem structure.
     */
    public static void bootstrap() {
        TestEntitlementInitialization.initializeArgs = new TestEntitlementInitialization.InitializeArgs(new TestPathLookup());
        logger.debug("Loading entitlement agent");
        EntitlementBootstrap.loadAgent(EntitlementBootstrap.findAgentJar(), TestEntitlementInitialization.class.getName());
    }

    /**
     * @brief Test-specific implementation of PathLookup that provides empty or null paths.
     *
     * Block Logic: This record acts as a mock for the PathLookup interface,
     * designed to prevent actual filesystem interactions during entitlement tests.
     * All path-related queries return null or empty streams, isolating the test
     * from the underlying file system.
     */
    private record TestPathLookup() implements PathLookup {
        /**
         * @brief Returns null for the PID file path in a test context.
         *
         * Functional Utility: Prevents the test environment from attempting to
         * locate or interact with a process ID file.
         */
        @Override
        public Path pidFile() {
            return null;
        }

        /**
         * @brief Returns an empty stream for base directory paths in a test context.
         *
         * Functional Utility: Ensures that no real base directories are processed
         * or expected during test execution.
         */
        @Override
        public Stream<Path> getBaseDirPaths(BaseDir baseDir) {
            return Stream.empty();
        }

        /**
         * @brief Returns an empty stream for resolved setting paths in a test context.
         *
         * Functional Utility: Prevents the test environment from attempting to
         * resolve configuration paths from the file system.
         */
        @Override
        public Stream<Path> resolveSettingPaths(BaseDir baseDir, String settingName) {
            return Stream.empty();
        }

    }
}
