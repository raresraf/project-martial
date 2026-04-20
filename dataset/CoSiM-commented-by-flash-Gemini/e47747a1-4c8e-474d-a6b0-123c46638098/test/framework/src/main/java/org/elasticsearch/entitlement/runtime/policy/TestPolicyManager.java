/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.entitlement.runtime.policy;

import org.elasticsearch.entitlement.runtime.policy.entitlements.Entitlement;

import java.nio.file.Path;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

 /**
  * @e47747a1-4c8e-474d-a6b0-123c46638098/test/framework/src/main/java/org/elasticsearch/entitlement/runtime/policy/TestPolicyManager.java
  * @brief Specialized PolicyManager implementation for test environments.
  * 
  * Functional Intent: Provides a sandbox-aware policy manager that facilitates test isolation 
  * by allowing explicit state resets and broad permissions for testing frameworks and core 
  * entitlement components.
  */
public class TestPolicyManager extends PolicyManager {
    /**
     * @brief Constructs a TestPolicyManager with the specified policy configurations.
     * @param serverPolicy Core server-side entitlement policy.
     * @param apmAgentEntitlements List of entitlements specific to APM agents.
     * @param pluginPolicies Map of plugin-specific entitlement policies.
     * @param scopeResolver Logic to determine the policy scope for a given class.
     * @param pluginSourcePaths File system paths associated with plugin entitlement sources.
     * @param pathLookup Utility for resolving file paths within the policy context.
     */
    public TestPolicyManager(
        Policy serverPolicy,
        List<Entitlement> apmAgentEntitlements,
        Map<String, Policy> pluginPolicies,
        Function<Class<?>, PolicyScope> scopeResolver,
        Map<String, Collection<Path>> pluginSourcePaths,
        PathLookup pathLookup
    ) {
        super(serverPolicy, apmAgentEntitlements, pluginPolicies, scopeResolver, pluginSourcePaths, pathLookup);
    }

    /**
     * Functional Utility: Ensures test atomicity by purging cached entitlement mappings 
     * between execution cycles, preventing state leakage between individual test cases.
     */
    public void reset() {
        super.moduleEntitlementsMap.clear();
    }

    /**
     * Block Logic: Identifies classes that are inherently trusted based on their classloader hierarchy.
     * Pre-condition: requestingClass must be non-null.
     * Invariant: System-level classes (bootstrap/platform) are always considered trusted.
     */
    @Override
    protected boolean isTrustedSystemClass(Class<?> requestingClass) {
        ClassLoader loader = requestingClass.getClassLoader();
        return loader == null || loader == ClassLoader.getPlatformClassLoader();
    }

    /**
     * Block Logic: Extends trivial allowance criteria to include testing infrastructure.
     * Logic: Combines standard policy manager checks with specific exemptions for 
     * the test framework and internal entitlement logic.
     */
    @Override
    boolean isTriviallyAllowed(Class<?> requestingClass) {
        return isTestFrameworkClass(requestingClass) || isEntitlementClass(requestingClass) || super.isTriviallyAllowed(requestingClass);
    }

    /**
     * Block Logic: Determines if a class belongs to the core entitlement implementation.
     * Logic: Identifies internal entitlement packages while excluding test-specific classes 
     * within those packages to maintain strict separation of concerns.
     */
    private boolean isEntitlementClass(Class<?> requestingClass) {
        return requestingClass.getPackageName().startsWith("org.elasticsearch.entitlement")
            && (requestingClass.getName().contains("Test") == false);
    }

    /**
     * Block Logic: Identifies classes belonging to standard testing and build tool packages.
     * Logic: Provides broad access for JUnit and Gradle components to ensure 
     * the test harness can operate without policy interference.
     */
    private boolean isTestFrameworkClass(Class<?> requestingClass) {
        String packageName = requestingClass.getPackageName();
        return packageName.startsWith("org.junit") || packageName.startsWith("org.gradle");
    }
}
