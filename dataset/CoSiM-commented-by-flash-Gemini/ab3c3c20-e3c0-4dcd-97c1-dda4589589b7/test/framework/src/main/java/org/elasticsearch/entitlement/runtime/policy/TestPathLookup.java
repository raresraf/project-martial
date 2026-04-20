/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.entitlement.runtime.policy;

import java.nio.file.Path;
import java.util.stream.Stream;

/**
 * @ab3c3c20-e3c0-4dcd-97c1-dda4589589b7/test/framework/src/main/java/org/elasticsearch/entitlement/runtime/policy/TestPathLookup.java
 * @brief Null-safe stub implementation of PathLookup for entitlement testing.
 * 
 * Functional Intent: Provides a "permissive" path resolution strategy for 
 * entitlement policies in test environments. By returning empty streams or nulls, 
 * it effectively bypasses path-based entitlement constraints that are not relevant 
 * to the current test context, while still fulfilling the interface contract.
 */
public class TestPathLookup implements PathLookup {
    @Override
    public Path pidFile() {
        return null;
    }

    /**
     * @brief Returns an empty stream, indicating no specific base directories are enforced.
     */
    @Override
    public Stream<Path> getBaseDirPaths(BaseDir baseDir) {
        return Stream.empty();
    }

    /**
     * @brief Returns an empty stream, bypassing relative path resolution for entitlements.
     */
    @Override
    public Stream<Path> resolveRelativePaths(BaseDir baseDir, Path relativePath) {
        return Stream.empty();
    }

    /**
     * @brief Returns an empty stream, bypassing setting-based path resolution.
     */
    @Override
    public Stream<Path> resolveSettingPaths(BaseDir baseDir, String settingName) {
        return Stream.empty();
    }

}
