/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.nativeaccess;

import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.common.logging.NodeNamePatternConverter;
import org.elasticsearch.test.ESTestCase;

import java.lang.foreign.Arena;
import java.util.Arrays;
import java.util.Optional;
import java.util.stream.IntStream;

import static org.elasticsearch.test.hamcrest.OptionalMatchers.isPresent;
import static org.hamcrest.Matchers.not;

/**
 * An abstract base class for testing vector similarity functions provided by the native access library.
 * This class sets up the necessary environment for tests, including logging and memory management,
 * and provides utility methods for checking platform support and creating test parameters.
 */
public abstract class VectorSimilarityFunctionsTests extends ESTestCase {

    static {
        // Initializes logging configuration required for native access components.
        NodeNamePatternConverter.setGlobalNodeName("foo");
        LogConfigurator.loadLog4jPlugins();
        LogConfigurator.configureESLogging(); // native access requires logging to be initialized
    }

    public static final Class<IllegalArgumentException> IAE = IllegalArgumentException.class;
    public static final Class<IndexOutOfBoundsException> IOOBE = IndexOutOfBoundsException.class;

    protected static Arena arena;

    protected final int size;
    protected final Optional<VectorSimilarityFunctions> vectorSimilarityFunctions;

    /**
     * Factory method for creating test parameters, specifically for different vector dimensions.
     *
     * @return An iterable of object arrays, each containing a vector dimension size for a test case.
     */
    protected static Iterable<Object[]> parametersFactory() {
        var dims1 = Arrays.stream(new int[] { 1, 2, 4, 6, 8, 12, 13, 16, 25, 31, 32, 33, 64, 100, 128, 207, 256, 300, 512, 702, 768 });
        var dims2 = Arrays.stream(new int[] { 1000, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097 });
        return () -> IntStream.concat(dims1, dims2).boxed().map(i -> new Object[] { i }).iterator();
    }

    /**
     * Constructor for the test class. Initializes the vector size and obtains the vector similarity functions
     * from the native access library.
     *
     * @param size The dimension of the vectors to be used in the tests.
     */
    protected VectorSimilarityFunctionsTests(int size) {
        logger.info(platformMsg());
        this.size = size;
        vectorSimilarityFunctions = NativeAccess.instance().getVectorSimilarityFunctions();
    }

    /**
     * Sets up the memory arena for native memory allocation before each test.
     */
    public static void setup() {
        arena = Arena.ofConfined();
    }

    /**
     * Cleans up the memory arena after each test to prevent memory leaks.
     */
    public static void cleanup() {
        arena.close();
    }

    /**
     * A test case to verify if the vector similarity functions are supported on the current platform.
     */
    public void testSupported() {
        supported();
    }

    /**
     * Retrieves the implementation of vector similarity functions.
     *
     * @return The VectorSimilarityFunctions instance.
     * @throws java.util.NoSuchElementException if the functions are not available on the current platform.
     */
    protected VectorSimilarityFunctions getVectorDistance() {
        return vectorSimilarityFunctions.get();
    }

    /**
     * Checks if the native vector similarity functions are supported on the current platform.
     * Support is determined by the JDK version, OS, and architecture.
     *
     * @return true if supported, false otherwise.
     */
    public boolean supported() {
        var jdkVersion = Runtime.version().feature();
        var arch = System.getProperty("os.arch");
        var osName = System.getProperty("os.name");

        // Pre-condition: Checks if the runtime environment meets the requirements for native vector similarity functions.
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
     * Generates a message indicating that the feature is not supported on the current platform.
     *
     * @return A formatted string with platform details.
     */
    public static String notSupportedMsg() {
        return "Not supported on [" + platformMsg() + "]";
    }

    /**
     * Gathers and formats platform information (JDK version, OS name, architecture).
     *
     * @return A string containing the platform details.
     */
    public static String platformMsg() {
        var jdkVersion = Runtime.version().feature();
        var arch = System.getProperty("os.arch");
        var osName = System.getProperty("os.name");
        return "JDK=" + jdkVersion + ", os=" + osName + ", arch=" + arch;
    }

    /**
     * Checks if the runtime supports passing on-heap arrays/segments to native memory,
     * a feature available from JDK 22 onwards.
     *
     * @return true if heap segments are supported, false otherwise.
     */
    // Support for passing on-heap arrays/segments to native
    protected static boolean supportsHeapSegments() {
        return Runtime.version().feature() >= 22;
    }
}