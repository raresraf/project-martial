/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.benchmark.vector;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;

import org.apache.lucene.util.Constants;
import org.elasticsearch.test.ESTestCase;
import org.junit.BeforeClass;
import org.openjdk.jmh.annotations.Param;

import java.util.Arrays;

/**
 * @file Int7uScorerBenchmarkTests.java
 * @brief Regression and correctness testing suite for int7 quantized vector scorers.
 * 
 * Functional Intent: Ensures that optimized native and Lucene vector similarity 
 * implementations yield numerically equivalent results to the reference scalar 
 * implementation within a defined precision delta (1e-3). It uses parameterized 
 * tests to validate correctness across different vector dimensionalities.
 * 
 * Domain: Production Systems, Numeric Validation, Vector Search, SIMD Correctness.
 */
public class Int7uScorerBenchmarkTests extends ESTestCase {

    // Logic: Defines the maximum allowable floating-point drift between implementations.
    final double delta = 1e-3;
    final int dims;

    public Int7uScorerBenchmarkTests(int dims) {
        this.dims = dims;
    }

    @BeforeClass
    public static void skipWindows() {
        // Pre-condition: Native scorers may have platform-specific availability.
        assumeFalse("doesn't work on windows yet", Constants.WINDOWS);
    }

    /**
     * testDotProduct - Verifies parity for Dot Product metrics.
     * 
     * Logic: Iteratively compares the output of scalar reference against Lucene 
     * and Native providers over 100 random samples.
     */
    public void testDotProduct() throws Exception {
        for (int i = 0; i < 100; i++) {
            var bench = new Int7uScorerBenchmark();
            bench.dims = dims;
            bench.setup();
            try {
                float expected = bench.dotProductScalar();
                // Synchronization: Validates Lucene optimized path.
                assertEquals(expected, bench.dotProductLucene(), delta);
                // Synchronization: Validates ES Native optimized path.
                assertEquals(expected, bench.dotProductNative(), delta);

                expected = bench.dotProductLuceneQuery();
                assertEquals(expected, bench.dotProductNativeQuery(), delta);
            } finally {
                bench.teardown();
            }
        }
    }

    /**
     * testSquareDistance - Verifies parity for Euclidean (L2) distance metrics.
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
     * parametersFactory - Parameter discovery for dimensional coverage.
     * 
     * Logic: Reflectively extracts the '@Param' values from the main benchmark 
     * class to ensure the test suite covers all benchmarked scenarios.
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
