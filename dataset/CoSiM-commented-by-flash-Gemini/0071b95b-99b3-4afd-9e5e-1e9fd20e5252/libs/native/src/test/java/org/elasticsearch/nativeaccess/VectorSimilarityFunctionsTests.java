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
 * Abstract base class for testing native vector similarity functions in Elasticsearch.
 * This class provides common test infrastructure, parameter generation, and platform
 * capability checks for {@link VectorSimilarityFunctions}. It ensures that native
 * implementations for vector distance calculations behave as expected and are
 * correctly supported on various JDK, OS, and architecture combinations.
 */
public abstract class VectorSimilarityFunctionsTests extends ESTestCase {

    /**
     * Functional Utility: Configures Elasticsearch logging, which is a prerequisite
     * for the native access functionality to initialize correctly. This static
     * block ensures that logging is set up before any tests are run.
     */
    static {
        NodeNamePatternConverter.setGlobalNodeName("foo");
        LogConfigurator.loadLog4jPlugins();
        LogConfigurator.configureESLogging(); // native access requires logging to be initialized
    }

    /** Reference to {@link IllegalArgumentException} for concise test assertions. */
    public static final Class<IllegalArgumentException> IAE = IllegalArgumentException.class;
    /** Reference to {@link IndexOutOfBoundsException} for concise test assertions. */
    public static final Class<IndexOutOfBoundsException> IOOBE = IndexOutOfBoundsException.class;

    /**
     * A confined native memory {@link Arena} used for allocating native memory segments
     * during tests. This ensures proper memory management and deallocation.
     */
    protected static Arena arena;

    /** The dimension (size) of the vectors being tested. */
    protected final int size;
    /**
     * An {@link Optional} container for the {@link VectorSimilarityFunctions} instance.
     * It will be present if native vector similarity functions are supported on the platform,
     * otherwise it will be empty.
     */
    protected final Optional<VectorSimilarityFunctions> vectorSimilarityFunctions;

    /**
     * Provides a factory for test parameters, generating various vector dimensions.
     * Functional Utility: Creates a stream of diverse vector dimensions (e.g., small,
     * power-of-2, near power-of-2, large) to thoroughly test the native functions
     * across different sizes. This is crucial for verifying how native implementations
     * handle different memory alignments, loop unrolling optimizations, and edge cases.
     *
     * @return An {@link Iterable} of {@code Object[]} containing integer vector dimensions.
     */
    protected static Iterable<Object[]> parametersFactory() {
        var dims1 = Arrays.stream(new int[] { 1, 2, 4, 6, 8, 12, 13, 16, 25, 31, 32, 33, 64, 100, 128, 207, 256, 300, 512, 702, 768 });
        var dims2 = Arrays.stream(new int[] { 1000, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097 });
        return () -> IntStream.concat(dims1, dims2).boxed().map(i -> new Object[] { i }).iterator();
    }

    /**
     * Constructs a new {@link VectorSimilarityFunctionsTests} instance for a given vector size.
     * Attempts to retrieve the platform-specific native vector similarity functions implementation.
     *
     * @param size The dimension of the vectors this test instance will use.
     */
    protected VectorSimilarityFunctionsTests(int size) {
        logger.info(platformMsg());
        this.size = size;
        vectorSimilarityFunctions = NativeAccess.instance().getVectorSimilarityFunctions();
    }

    /**
     * Sets up a confined native memory {@link Arena} before all tests in this class.
     * Functional Utility: Allocates a {@link Arena} for managing native memory segments,
     * ensuring memory is properly managed and automatically deallocated after the test fixture.
     */
    public static void setup() {
        arena = Arena.ofConfined();
    }

    /**
     * Cleans up (closes) the native memory {@link Arena} after all tests in this class.
     * Functional Utility: Releases the {@link Arena} and its associated native memory,
     * preventing resource leaks.
     */
    public static void cleanup() {
        arena.close();
    }

    /**
     * Verifies that the {@link VectorSimilarityFunctions} instance is correctly
     * supported (present) or not supported (absent) on the current platform.
     */
    public void testSupported() {
        supported();
    }

    /**
     * Retrieves the {@link VectorSimilarityFunctions} instance.
     * Precondition: {@link #vectorSimilarityFunctions} must be present.
     *
     * @return The {@link VectorSimilarityFunctions} instance.
     */
    protected VectorSimilarityFunctions getVectorDistance() {
        return vectorSimilarityFunctions.get();
    }

    /**
     * Determines if native vector similarity functions are expected to be supported
     * on the current platform based on JDK version, operating system, and architecture.
     * Functional Utility: Checks system properties to confirm if the current Java
     * runtime environment and operating system architecture meet the requirements
     * for utilizing optimized native vector similarity functions (e.g., JDK 21+,
     * aarch64 on Mac/Linux, amd64 on Linux). It then asserts the presence or
     * absence of the native function implementation accordingly.
     *
     * @return {@code true} if native functions are supported and present, {@code false} otherwise.
     */
    public boolean supported() {
        var jdkVersion = Runtime.version().feature();
        var arch = System.getProperty("os.arch");
        var osName = System.getProperty("os.name");

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
     * Generates a descriptive message indicating why native functions are not supported
     * on the current platform.
     *
     * @return A string message detailing the unsupported platform.
     */
    public static String notSupportedMsg() {
        return "Not supported on [" + platformMsg() + "]";
    }

    /**
     * Constructs a string describing the current Java runtime and operating system platform.
     * Functional Utility: Provides detailed platform information (JDK version, OS name,
     * architecture) for debugging and logging purposes. This is crucial for understanding
     * the environment where native code execution might vary or where specific platform
     * checks are performed.
     *
     * @return A string containing platform details.
     */
    public static String platformMsg() {
        var jdkVersion = Runtime.version().feature();
        var arch = System.getProperty("os.arch");
        var osName = System.getProperty("os.name");
        return "JDK=" + jdkVersion + ", os=" + osName + ", arch=" + arch;
    }

    /**
     * Checks if the current JDK version supports passing on-heap {@link java.lang.foreign.MemorySegment}s
     * directly to native functions without requiring a copy to off-heap memory.
     * Functional Utility: Determines if the Java Foreign Function and Memory API (FFM)
     * allows direct use of Java heap-allocated memory (segments) by native code,
     * which can significantly impact performance and interoperability with native libraries.
     *
     * @return {@code true} if on-heap segments are supported, {@code false} otherwise.
     */
    protected static boolean supportsHeapSegments() {
        return Runtime.version().feature() >= 22;
    }
}
