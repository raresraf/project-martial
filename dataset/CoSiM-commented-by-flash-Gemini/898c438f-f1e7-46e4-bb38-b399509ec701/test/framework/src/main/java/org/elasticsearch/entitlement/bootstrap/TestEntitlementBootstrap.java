/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.entitlement.bootstrap;

import org.elasticsearch.entitlement.initialization.TestEntitlementInitialization;
import org.elasticsearch.entitlement.runtime.policy.PathLookup;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;

import java.nio.file.Path;
import java.util.stream.Stream;

/**
 * @898c438f-f1e7-46e4-bb38-b399509ec701/test/framework/src/main/java/org/elasticsearch/entitlement/bootstrap/TestEntitlementBootstrap.java
 * @brief Utility class to activate entitlement checking specifically for testing environments.
 *
 * This class provides a {@code bootstrap} method to configure and load the entitlement
 * agent within a test context, allowing for the simulation and verification of
 * entitlement-related functionalities without relying on a full production setup.
 */
public class TestEntitlementBootstrap {

    private static final Logger logger = LogManager.getLogger(TestEntitlementBootstrap.class);

    /**
     * @brief Logger instance for recording events and debugging information within this class.
     */

    /**
     * Activates entitlement checking in tests.
     */
    /**
     * @brief Activates entitlement checking in tests.
     *
     * This method initializes the {@code TestEntitlementInitialization} with a test-specific
     * {@code PathLookup} implementation and subsequently loads the entitlement agent.
     * It effectively prepares the test environment for entitlement-aware operations.
     */
    public static void bootstrap() {
        TestEntitlementInitialization.initializeArgs = new TestEntitlementInitialization.InitializeArgs(new TestPathLookup());
        logger.debug("Loading entitlement agent");
        EntitlementBootstrap.loadAgent(EntitlementBootstrap.findAgentJar(), TestEntitlementInitialization.class.getName());
    }

    private record TestPathLookup() implements PathLookup {
        /**
         * @brief Test-specific implementation of {@link PathLookup}.
         *
         * This record serves as a minimal {@code PathLookup} implementation for testing purposes.
         * It provides stubbed methods that return {@code null} or empty streams, as actual
         * file system paths are not relevant for the entitlement tests.
         */
        @Override
        public Path pidFile() {
            /**
             * @brief Returns null as a placeholder for the PID file path in a test context.
             */
            return null;
        }

        @Override
        public Stream<Path> getBaseDirPaths(BaseDir baseDir) {
            /**
             * @brief Returns an empty stream for base directory paths in a test context.
             * @param baseDir The base directory type.
             * @return An empty stream of paths.
             */
            return Stream.empty();
        }

        @Override
        public Stream<Path> resolveSettingPaths(BaseDir baseDir, String settingName) {
            /**
             * @brief Returns an empty stream for resolved setting paths in a test context.
             * @param baseDir The base directory type.
             * @param settingName The name of the setting.
             * @return An empty stream of paths.
             */
            return Stream.empty();
        }

    }
}
