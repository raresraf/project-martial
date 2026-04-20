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
 * @d5396203-4aef-40ad-977e-3f0d08606956/test/framework/src/main/java/org/elasticsearch/entitlement/bootstrap/TestEntitlementBootstrap.java
 * @brief Bootstrapping utility for entitlement checking in test environments.
 * 
 * Functional Intent: Orchestrates the activation of the entitlement system for 
 * integration testing. It initializes the required test-specific metadata (PathLookup) 
 * and injects the entitlement agent into the running JVM, enabling runtime 
 * enforcement of license and feature policies during test execution.
 */
public class TestEntitlementBootstrap {

    private static final Logger logger = LogManager.getLogger(TestEntitlementBootstrap.class);

    /**
     * @brief Activates entitlement checking in tests.
     * Logic: 
     * 1. Configures the static initialization arguments for the test environment.
     * 2. Dynamically loads the entitlement agent JAR using the standard bootstrap mechanism.
     */
    public static void bootstrap() {
        TestEntitlementInitialization.initializeArgs = new TestEntitlementInitialization.InitializeArgs(new TestPathLookup());
        logger.debug("Loading entitlement agent");
        EntitlementBootstrap.loadAgent(EntitlementBootstrap.findAgentJar(), TestEntitlementInitialization.class.getName());
    }

    /**
     * @brief Minimalist PathLookup implementation for bootstrapping.
     * Logic: Provides a stubbed implementation of the PathLookup interface 
     * required for agent initialization, where all operations return null or 
     * empty streams to ensure a permissive, filesystem-agnostic test environment.
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
