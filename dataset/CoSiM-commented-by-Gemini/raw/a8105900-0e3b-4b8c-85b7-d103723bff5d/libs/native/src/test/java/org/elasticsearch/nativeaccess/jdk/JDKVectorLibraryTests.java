/**
 * @file JDKVectorLibraryTests.java
 * @brief Unit tests for native vector similarity functions using the JDK Foreign Function & Memory API.
 * @author Elasticsearch B.V.
 *
 * @details
 * This class validates the correctness of native (JNI) implementations of vector
 * similarity functions (dot product, squared distance) for 7-bit integer vectors.
 * It inherits from `VectorSimilarityFunctionsTests` to ensure these tests only run
 * on supported platforms. The core testing strategy is to compare the output of the
 * native functions against equivalent, simple scalar implementations written in pure Java.
 */
/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.nativeaccess.jdk;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;

import org.elasticsearch.nativeaccess.VectorSimilarityFunctionsTests;
import org.junit.AfterClass;
import org.junit.BeforeClass;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.stream.IntStream;

import static org.hamcrest.Matchers.containsString;

public class JDKVectorLibraryTests extends VectorSimilarityFunctionsTests {

    // Defines the valid range for 7-bit integer vector components.
    static final byte MIN_INT7_VALUE = 0;
    static final byte MAX_INT7_VALUE = 127;

    static final Class<IllegalArgumentException> IAE = IllegalArgumentException.class;
    static final Class<IndexOutOfBoundsException> IOOBE = IndexOutOfBoundsException.class;

    // A set of vector dimensions to run the tests against.
    static final int[] VECTOR_DIMS = { 1, 4, 6, 8, 13, 16, 25, 31, 32, 33, 64, 100, 128, 207, 256, 300, 512, 702, 1023, 1024, 1025 };

    final int size;

    // An Arena manages the lifecycle of native memory segments.
    static Arena arena;

    final double delta;

    public JDKVectorLibraryTests(int size) {
        this.size = size;
        this.delta = 1e-5 * size; // scale the delta with the size
    }

    @BeforeClass
    public static void setup() {
        // Create a confined arena for memory allocation, ensuring all segments are freed when the arena is closed.
        arena = Arena.ofConfined();
    }

    @AfterClass
    public static void cleanup() {
        arena.close();
    }

    /**
     * A JUnit `@ParametersFactory` to create a parameterized test suite.
     * This will run all tests in the class for each dimension in `VECTOR_DIMS`.
     */
    @ParametersFactory
    public static Iterable<Object[]> parametersFactory() {
        return () -> IntStream.of(VECTOR_DIMS).boxed().map(i -> new Object[] { i }).iterator();
    }

    /**
     * Tests the correctness of dot product and squared distance for 7-bit integer vectors.
     * It generates random vectors, computes the similarity scores using both the native
     * function and a pure Java scalar implementation, and asserts that the results are equal.
     */
    public void testInt7BinaryVectors() {
        // The test is skipped if the platform is not supported.
        assumeTrue(notSupportedMsg(), supported());
        final int dims = size;
        final int numVecs = randomIntBetween(2, 101);
        var values = new byte[numVecs][dims];
        // Allocate a native memory segment to hold all vectors contiguously.
        var segment = arena.allocate((long) dims * numVecs);
        for (int i = 0; i < numVecs; i++) {
            randomBytesBetween(values[i], MIN_INT7_VALUE, MAX_INT7_VALUE);
            MemorySegment.copy(MemorySegment.ofArray(values[i]), 0L, segment, (long) i * dims, dims);
        }

        final int loopTimes = 1000;
        for (int i = 0; i < loopTimes; i++) {
            int first = randomInt(numVecs - 1);
            int second = randomInt(numVecs - 1);
            var nativeSeg1 = segment.asSlice((long) first * dims, dims);
            var nativeSeg2 = segment.asSlice((long) second * dims, dims);

            // Test dot product
            int expected = dotProductScalar(values[first], values[second]);
            assertEquals(expected, dotProduct7u(nativeSeg1, nativeSeg2, dims));
            // On JDK 22+, also test with heap-based memory segments.
            if (testWithHeapSegments()) {
                var heapSeg1 = MemorySegment.ofArray(values[first]);
                var heapSeg2 = MemorySegment.ofArray(values[second]);
                assertEquals(expected, dotProduct7u(heapSeg1, heapSeg2, dims));
                assertEquals(expected, dotProduct7u(nativeSeg1, heapSeg2, dims));
                assertEquals(expected, dotProduct7u(heapSeg1, nativeSeg2, dims));
            }

            // Test square distance
            expected = squareDistanceScalar(values[first], values[second]);
            assertEquals(expected, squareDistance7u(nativeSeg1, nativeSeg2, dims));
            if (testWithHeapSegments()) {
                var heapSeg1 = MemorySegment.ofArray(values[first]);
                var heapSeg2 = MemorySegment.ofArray(values[second]);
                assertEquals(expected, squareDistance7u(heapSeg1, heapSeg2, dims));
                assertEquals(expected, squareDistance7u(nativeSeg1, heapSeg2, dims));
                assertEquals(expected, squareDistance7u(heapSeg1, nativeSeg2, dims));
            }
        }
    }

    static boolean testWithHeapSegments() {
        return Runtime.version().feature() >= 22;
    }

    /**
     * Tests that the native functions throw appropriate exceptions for invalid arguments,
     * such as mismatched vector dimensions or out-of-bounds access.
     */
    public void testIllegalDims() {
        assumeTrue(notSupportedMsg(), supported());
        var segment = arena.allocate((long) size * 3);

        var e1 = expectThrows(IAE, () -> dotProduct7u(segment.asSlice(0L, size), segment.asSlice(size, size + 1), size));
        assertThat(e1.getMessage(), containsString("dimensions differ"));

        var e2 = expectThrows(IOOBE, () -> dotProduct7u(segment.asSlice(0L, size), segment.asSlice(size, size), size + 1));
        assertThat(e2.getMessage(), containsString("out of bounds for length"));

        var e3 = expectThrows(IOOBE, () -> dotProduct7u(segment.asSlice(0L, size), segment.asSlice(size, size), -1));
        assertThat(e3.getMessage(), containsString("out of bounds for length"));

        var e4 = expectThrows(IAE, () -> squareDistance7u(segment.asSlice(0L, size), segment.asSlice(size, size + 1), size));
        assertThat(e4.getMessage(), containsString("dimensions differ"));

        var e5 = expectThrows(IOOBE, () -> squareDistance7u(segment.asSlice(0L, size), segment.asSlice(size, size), size + 1));
        assertThat(e5.getMessage(), containsString("out of bounds for length"));

        var e6 = expectThrows(IOOBE, () -> squareDistance7u(segment.asSlice(0L, size), segment.asSlice(size, size), -1));
        assertThat(e6.getMessage(), containsString("out of bounds for length"));
    }

    /**
     * Wrapper to invoke the native dot product function via a method handle.
     */
    int dotProduct7u(MemorySegment a, MemorySegment b, int length) {
        try {
            return (int) getVectorDistance().dotProductHandle7u().invokeExact(a, b, length);
        } catch (Throwable e) {
            if (e instanceof Error err) {
                throw err;
            } else if (e instanceof RuntimeException re) {
                throw re;
            } else {
                throw new RuntimeException(e);
            }
        }
    }

    /**
     * Wrapper to invoke the native squared distance function via a method handle.
     */
    int squareDistance7u(MemorySegment a, MemorySegment b, int length) {
        try {
            return (int) getVectorDistance().squareDistanceHandle7u().invokeExact(a, b, length);
        } catch (Throwable e) {
            if (e instanceof Error err) {
                throw err;
            } else if (e instanceof RuntimeException re) {
                throw re;
            } else {
                throw new RuntimeException(e);
            }
        }
    }

    /**
     * A pure Java, scalar implementation of the dot product for byte arrays.
     * This serves as the ground truth for verifying the native implementation.
     */
    static int dotProductScalar(byte[] a, byte[] b) {
        int res = 0;
        // Invariant: Accumulate the product of corresponding vector components.
        for (int i = 0; i < a.length; i++) {
            res += a[i] * b[i];
        }
        return res;
    }

    /**
     * A pure Java, scalar implementation of the squared Euclidean distance for byte arrays.
     * This serves as the ground truth for verifying the native implementation.
     */
    static int squareDistanceScalar(byte[] a, byte[] b) {
        int squareSum = 0;
        // Invariant: Accumulate the square of the difference between corresponding components.
        for (int i = 0; i < a.length; i++) {
            int diff = a[i] - b[i];
            squareSum += diff * diff;
        }
        return squareSum;
    }
}
