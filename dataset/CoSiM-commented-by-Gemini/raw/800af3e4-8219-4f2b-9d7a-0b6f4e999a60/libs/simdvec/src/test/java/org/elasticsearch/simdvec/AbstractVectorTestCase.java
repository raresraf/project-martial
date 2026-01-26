/**
 * @file AbstractVectorTestCase.java
 * @brief Base class for vector scoring tests in Elasticsearch.
 * @details This abstract class provides common setup and utility methods for testing vector scorer factories.
 * It initializes the vector scorer factory and includes helper methods for checking platform compatibility
 * and performing data conversions necessary for the tests. It is part of the Elasticsearch test framework.
 */
/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.simdvec;

import org.elasticsearch.test.ESTestCase;
import org.junit.BeforeClass;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Optional;

import static org.elasticsearch.test.hamcrest.OptionalMatchers.isPresent;
import static org.hamcrest.Matchers.not;

public abstract class AbstractVectorTestCase extends ESTestCase {

    /**
     * @brief Holds the singleton instance of the VectorScorerFactory.
     * @details This factory is used to create vector scorer instances, which are responsible for
     * calculating scores between vectors. The Optional wrapper handles cases where the factory
     * might not be available on a given platform.
     */
    static Optional<VectorScorerFactory> factory;

    /**
     * @brief Initializes the VectorScorerFactory before any tests in the class are run.
     * @details This setup method ensures that the factory instance is obtained and ready for use
     * in the test cases. It is annotated with @BeforeClass to run once per test class.
     */
    @BeforeClass
    public static void getVectorScorerFactory() {
        factory = VectorScorerFactory.instance();
    }

    /**
     * @brief Constructor that logs platform information.
     * @details When an instance of a subclass is created, this constructor logs a message
     * containing details about the current platform, which is useful for debugging.
     */
    protected AbstractVectorTestCase() {
        logger.info(platformMsg());
    }

    /**
     * @brief Checks if the vector scoring feature is supported on the current platform.
     * @details It verifies the JDK version, OS, and architecture to determine if the native
     * SIMD optimizations are available. The check is crucial for running tests only on
     * supported environments.
     * @return true if the platform is supported, false otherwise.
     */
    public static boolean supported() {
        var jdkVersion = Runtime.version().feature();
        var arch = System.getProperty("os.arch");
        var osName = System.getProperty("os.name");

        if (jdkVersion >= 21
            && (arch.equals("aarch64") && (osName.startsWith("Mac") || osName.equals("Linux"))
                || arch.equals("amd64") && osName.equals("Linux"))) {
            assertThat(factory, isPresent());
            return true;
        } else {
            assertThat(factory, not(isPresent()));
            return false;
        }
    }

    /**
     * @brief Generates a message indicating that the feature is not supported on the current platform.
     * @return A string with the "Not supported" message, including platform details.
     */
    public static String notSupportedMsg() {
        return "Not supported on [" + platformMsg() + "]";
    }

    /**
     * @brief Constructs a string with detailed platform information.
     * @details This method gathers JDK version, OS name, and architecture to create a
     * comprehensive platform description string, primarily for logging and debugging.
     * @return A string representing the current platform.
     */
    public static String platformMsg() {
        var jdkVersion = Runtime.version().feature();
        var arch = System.getProperty("os.arch");
        var osName = System.getProperty("os.name");
        return "JDK=" + jdkVersion + ", os=" + osName + ", arch=" + arch;
    }

    /**
     * @brief Determines if the JDK version supports passing on-heap arrays to native code.
     * @details This is a feature check for JDK 22 and later, which affects how memory
     * segments are handled in native calls.
     * @return true if on-heap segments are supported, false otherwise.
     */
    // Support for passing on-heap arrays/segments to native
    protected static boolean supportsHeapSegments() {
        return Runtime.version().feature() >= 22;
    }

    /**
     * @brief Converts a float value into a byte array using little-endian byte order.
     * @param value The float value to convert.
     * @return A byte array representing the float.
     */
    public static byte[] floatToByteArray(float value) {
        return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(value).array();
    }

    /**
     * @brief Concatenates multiple byte arrays into a single byte array.
     * @param arrays A variable number of byte arrays to concatenate.
     * @return A new byte array containing all the elements of the input arrays in order.
     * @throws IOException if an I/O error occurs during the stream operations.
     */
    public static byte[] concat(byte[]... arrays) throws IOException {
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            for (var ba : arrays) {
                baos.write(ba);
            }
            return baos.toByteArray();
        }
    }
}
