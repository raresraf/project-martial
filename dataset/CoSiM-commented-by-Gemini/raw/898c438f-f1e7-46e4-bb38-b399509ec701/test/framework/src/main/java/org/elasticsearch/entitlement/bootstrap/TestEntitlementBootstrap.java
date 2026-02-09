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
 * A bootstrap class for activating the Elasticsearch entitlement system within a test environment.
 *
 * This class provides a static method to load the entitlement Java agent, which
 * intercepts and controls access to licensed features during tests.
 */
public class TestEntitlementBootstrap {

    private static final Logger logger = LogManager.getLogger(TestEntitlementBootstrap.class);

    /**
     * Activates entitlement checking in tests by loading the entitlement Java agent.
     *
     * This method performs two key actions:
     * 1. It configures the `TestEntitlementInitialization` class with a stubbed
     *    `TestPathLookup`, ensuring that the entitlement system does not perform
     *    any actual file system lookups during tests.
     * 2. It finds and loads the entitlement agent JAR, instructing it to use the
     *    test-specific initialization logic.
     */
    public static void bootstrap() {
        // Functional Utility: Prepare test-specific arguments for the agent's initialization phase.
        TestEntitlementInitialization.initializeArgs = new TestEntitlementInitialization.InitializeArgs(new TestPathLookup());
        logger.debug("Loading entitlement agent");
        // Functional Utility: Dynamically load the Java agent for runtime instrumentation.
        EntitlementBootstrap.loadAgent(EntitlementBootstrap.findAgentJar(), TestEntitlementInitialization.class.getName());
    }

    /**
     * A stub implementation of the `PathLookup` interface for use in tests.
     *
     * This record provides no-op implementations for all path lookup methods,
     * returning `null` or empty streams to prevent any file system interaction
     * when the entitlement system is initialized in a test context.
     */
    private record TestPathLookup() implements PathLookup {
        /**
         * @return Always returns `null`.
         */
        @Override
        public Path pidFile() {
            return null;
        }

        /**
         * @return Always returns an empty stream.
         */
        @Override
        public Stream<Path> getBaseDirPaths(BaseDir baseDir) {
            return Stream.empty();
        }

        /**
         * @return Always returns an empty stream.
         */
        @Override
        public Stream<Path> resolveSettingPaths(BaseDir baseDir, String settingName) {
            return Stream.empty();
        }

    }
}