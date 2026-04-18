/**
 * @a8105900-0e3b-4b8c-85b7-d103723bff5d/libs/native/src/test/java/org/elasticsearch/nativeaccess/jdk/JDKVectorLibraryTests.java
 * @brief Accuracy and boundary validation for JDK-intrinsic vector operations.
 * Domain: Software Testing, SIMD, Foreign Function & Memory API (Panama).
 * Architecture: Extends VectorSimilarityFunctionsTests; utilizes parameterized testing across various vector dimensions.
 * Functional Utility: Verifies that the native dot product and squared Euclidean distance kernels yield results identical to reference scalar implementations.
 * Performance: Employs Arena-allocated off-heap MemorySegments to simulate high-performance data access patterns.
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

/**
 * @brief Unit test suite for Panama-based vector scoring kernels.
 * Data Range: Specifically targets int7 scalar quantized vectors [0, 127].
 * Dimensions: Tests edge cases and standard SIMD alignment boundaries (1, 4, 31, 32, 1024, etc.).
 */
public class JDKVectorLibraryTests extends VectorSimilarityFunctionsTests {

    // bounds of the range of values that can be seen by int7 scalar quantized vectors
    static final byte MIN_INT7_VALUE = 0;
    static final byte MAX_INT7_VALUE = 127;

    static final Class<IllegalArgumentException> IAE = IllegalArgumentException.class;
    static final Class<IndexOutOfBoundsException> IOOBE = IndexOutOfBoundsException.class;

    static final int[] VECTOR_DIMS = { 1, 4, 6, 8, 13, 16, 25, 31, 32, 33, 64, 100, 128, 207, 256, 300, 512, 702, 1023, 1024, 1025 };

    final int size;

    static Arena arena;

    final double delta;

    public JDKVectorLibraryTests(int size) {
        this.size = size;
        this.delta = 1e-5 * size; // scale the delta with the size
    }

    /**
     * @brief Setup for Foreign Function & Memory access.
     * Memory Management: Uses a confined Arena for deterministic off-heap resource lifecycle.
     */
    @BeforeClass
    public static void setup() {
        arena = Arena.ofConfined();
    }

    @AfterClass
    public static void cleanup() {
        arena.close();
    }

    @ParametersFactory
    public static Iterable<Object[]> parametersFactory() {
        return () -> IntStream.of(VECTOR_DIMS).boxed().map(i -> new Object[] { i }).iterator();
    }

    /**
     * @brief Comprehensive accuracy test for int7 vector scoring.
     * Logic: Performs randomized fuzz testing by comparing native SIMD results with a trusted scalar baseline.
     * Execution: Iterates 1000 times per dimension to ensure robustness across different data distributions.
     */
    public void testInt7BinaryVectors() {
        assumeTrue(notSupportedMsg(), supported());
        final int dims = size;
        final int numVecs = randomIntBetween(2, 101);
        var values = new byte[numVecs][dims];
        var segment = arena.allocate((long) dims * numVecs);
        
        // Initialization: Pre-populates off-heap memory with quantized vector data.
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

            // Block Logic: Cross-verification of Dot Product kernel.
            int expected = dotProductScalar(values[first], values[second]);
            assertEquals(expected, dotProduct7u(nativeSeg1, nativeSeg2, dims));
            
            // Branch Logic: Tests mixed memory segment types (Heap vs Native) if supported by JVM version.
            if (testWithHeapSegments()) {
                var heapSeg1 = MemorySegment.ofArray(values[first]);
                var heapSeg2 = MemorySegment.ofArray(values[second]);
                assertEquals(expected, dotProduct7u(heapSeg1, heapSeg2, dims));
                assertEquals(expected, dotProduct7u(nativeSeg1, heapSeg2, dims));
                assertEquals(expected, dotProduct7u(heapSeg1, nativeSeg2, dims));
            }

            // Block Logic: Cross-verification of Squared Euclidean Distance kernel.
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
     * @brief Boundary condition validation for vector operations.
     * Logic: Intentionally passes incorrect dimensions and offsets to ensure native code correctly triggers expected Java exceptions (IAE, IOOBE).
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
     * @brief Low-level invocation of the native dot product MethodHandle.
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
     * @brief Low-level invocation of the native squared distance MethodHandle.
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
     * @brief Reference scalar implementation of Dot Product. 
     * Functional Utility: Serves as the "Gold Standard" for accuracy comparison.
     */
    static int dotProductScalar(byte[] a, byte[] b) {
        int res = 0;
        for (int i = 0; i < a.length; i++) {
            res += a[i] * b[i];
        }
        return res;
    }

    /** 
     * @brief Reference scalar implementation of Squared Distance. 
     * Complexity: O(N).
     */
    static int squareDistanceScalar(byte[] a, byte[] b) {
        // Note: this will not overflow if dim < 2^18, since max(byte * byte) = 2^14.
        int squareSum = 0;
        for (int i = 0; i < a.length; i++) {
            int diff = a[i] - b[i];
            squareSum += diff * diff;
        }
        return squareSum;
    }
}
