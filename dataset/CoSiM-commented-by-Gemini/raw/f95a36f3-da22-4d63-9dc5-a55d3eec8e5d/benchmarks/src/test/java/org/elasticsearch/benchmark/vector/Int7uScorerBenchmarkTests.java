/**
 * @file Int7uScorerBenchmarkTests.java
 * @brief Unit tests for the `Int7uScorerBenchmark` class.
 *
 * This class validates the correctness of the different vector scoring implementations
 * used in the `Int7uScorerBenchmark`. It ensures that the results produced by the
 * scalar, Lucene, and native (SIMD) implementations are all numerically equivalent
 * within an acceptable margin of error (delta).
 *
 * This "test for a benchmark" is a crucial sanity check to confirm that the performance
 * comparisons are being made on functionally identical calculations. The tests are
 * parameterized to run for all vector dimensions defined in the benchmark class.
 */
package org.elasticsearch.benchmark.vector;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;

import org.apache.lucene.util.Constants;
import org.elasticsearch.test.ESTestCase;
import org.junit.BeforeClass;
import org.openjdk.jmh.annotations.Param;

import java.util.Arrays;

public class Int7uScorerBenchmarkTests extends ESTestCase {

    final double delta = 1e-3;
    final int dims;

    public Int7uScorerBenchmarkTests(int dims) {
        this.dims = dims;
    }

    /**
     * Skips the tests on Windows environments.
     * This is a JUnit lifecycle method that runs once before any tests in the class.
     * It uses an `assumeFalse` clause to halt test execution if the operating system is Windows,
     * likely due to dependencies or features not yet supported on that platform.
     */
    @BeforeClass
    public static void skipWindows() {
        assumeFalse("doesn't work on windows yet", Constants.WINDOWS);
    }

    /**
     * Tests that all dot product implementations produce the same score.
     *
     * This test iterates 100 times, each time setting up a new benchmark instance with
     * random vectors. It calculates the expected score using the simple scalar implementation
     * and then asserts that the scores from the Lucene and native implementations match the
     * expected value. It also validates the query-time scorers against each other.
     *
     * @throws Exception if the benchmark setup or teardown fails.
     */
    public void testDotProduct() throws Exception {
        for (int i = 0; i < 100; i++) {
            var bench = new Int7uScorerBenchmark();
            bench.dims = dims;
            bench.setup();
            try {
                float expected = bench.dotProductScalar();
                assertEquals(expected, bench.dotProductLucene(), delta);
                assertEquals(expected, bench.dotProductNative(), delta);

                expected = bench.dotProductLuceneQuery();
                assertEquals(expected, bench.dotProductNativeQuery(), delta);
            } finally {
                bench.teardown();
            }
        }
    }

    /**
     * Tests that all squared Euclidean distance implementations produce the same score.
     *
     * Similar to `testDotProduct`, this test runs 100 iterations. It uses the scalar
     * implementation as the source of truth and verifies that the Lucene and native
     * implementations produce equivalent results for squared distance. It also validates
     * the query-time scorers.
     *
     * @throws Exception if the benchmark setup or teardown fails.
     */
    public void testSquareDistance() throws Exception {
        for (int i = 0; i < 100; i++) {
            var bench = new Int7uScorerBenchmark();
            bench.dims = dims;
            bench.setup();
            try {
                float expected = bench.squareDistanceScalar();
                assertEquals(expected, bench.squareDistanceLucene(), delta);
                assertEquals(expected, bench.squareDistanceNative(), delta);

                expected = bench.squareDistanceLuceneQuery();
                assertEquals(expected, bench.squareDistanceNativeQuery(), delta);
            } finally {
                bench.teardown();
            }
        }
    }

    /**
     * A JUnit `ParametersFactory` used to parameterize the test runs.
     *
     * This method dynamically reads the `@Param` annotations from the `dims` field
     * in the `Int7uScorerBenchmark` class. This allows the test to automatically
     * run for all vector dimensions that the benchmark is configured to test,
     * ensuring that correctness is maintained across all configurations without
     * hardcoding the dimensions in the test.
     *
     * @return An `Iterable` of object arrays, where each array contains a single
     *         integer representing a vector dimension to be tested.
     */
    @ParametersFactory
    public static Iterable<Object[]> parametersFactory() {
        try {
            var params = Int7uScorerBenchmark.class.getField("dims").getAnnotationsByType(Param.class)[0].value();
            return () -> Arrays.stream(params).map(Integer::parseInt).map(i -> new Object[] { i }).iterator();
        } catch (NoSuchFieldException e) {
            throw new AssertionError(e);
        }
    }
}
