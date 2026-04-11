/**
 * @file VectorSimilarityFunctionsTests.java
 * @brief Unit tests for checking platform support for native vector similarity functions.
 * @author Elasticsearch B.V.
 *
 * @details
 * This class does not test the correctness of the vector similarity calculations.
 * Instead, it verifies that the native library for these functions is loaded
 * correctly and is supported on the current platform (JDK version, OS, and
 * architecture). It acts as a guard to ensure that these native optimizations
- * are only used in environments where they are known to work.
 */
/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.nativeaccess;

import org.elasticsearch.test.ESTestCase;

import java.util.Optional;

import static org.elasticsearch.test.hamcrest.OptionalMatchers.isPresent;
import static org.hamcrest.Matchers.not;

public class VectorSimilarityFunctionsTests extends ESTestCase {

    final Optional<VectorSimilarityFunctions> vectorSimilarityFunctions;

    /**
     * Constructor that initializes the test by attempting to get an instance
     * of the native `VectorSimilarityFunctions` from the `NativeAccess` singleton.
     */
    public VectorSimilarityFunctionsTests() {
        logger.info(platformMsg());
        vectorSimilarityFunctions = NativeAccess.instance().getVectorSimilarityFunctions();
    }

    /**
     * The main test method that triggers the platform support check.
     */
    public void testSupported() {
        supported();
    }

    /**
     * A helper method to get the underlying vector distance functions.
     * Throws an exception if the optional is empty.
     */
    protected VectorSimilarityFunctions getVectorDistance() {
        return vectorSimilarityFunctions.get();
    }

    /**
     * Checks if native vector similarity functions are supported on the current platform
     * and asserts the expected presence or absence of the implementation.
     *
     * @return {@code true} if supported, {@code false} otherwise.
     */
    public boolean supported() {
        var jdkVersion = Runtime.version().feature();
        var arch = System.getProperty("os.arch");
        var osName = System.getProperty("os.name");

        // The logic defines a specific set of supported platforms.
        // Pre-condition: Check if the current platform matches the supported configurations.
        if (jdkVersion >= 21
            && ((arch.equals("aarch64") && (osName.startsWith("Mac") || osName.equals("Linux")))
                || (arch.equals("amd64") && osName.equals("Linux")))) {
            assertThat(vectorSimilarityFunctions, isPresent());
            return true;
        } else {
            assertThat(vectorSimilarityFunctions, not(isPresent()));
            return false;
        }
    }

    /**
     * @return A formatted string indicating that the feature is not supported on the current platform.
     */
    public static String notSupportedMsg() {
        return "Not supported on [" + platformMsg() + "]";
    }

    /**
     * @return A formatted string detailing the current platform (JDK, OS, Arch).
     */
    public static String platformMsg() {
        var jdkVersion = Runtime.version().feature();
        var arch = System.getProperty("os.arch");
        var osName = System.getProperty("os.name");
        return "JDK=" + jdkVersion + ", os=" + osName + ", arch=" + arch;
    }
}
