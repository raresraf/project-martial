/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.benchmark.vector;

import org.apache.lucene.codecs.lucene99.Lucene99ScalarQuantizedVectorScorer;
import org.apache.lucene.codecs.lucene99.OffHeapQuantizedByteVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.util.quantization.ScalarQuantizer;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.core.IOUtils;
import org.elasticsearch.simdvec.VectorScorerFactory;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

import java.io.IOException;
import java.nio.file.Files;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

import static org.elasticsearch.simdvec.VectorSimilarityType.DOT_PRODUCT;
import static org.elasticsearch.simdvec.VectorSimilarityType.EUCLIDEAN;

/**
 * @file Int7uScorerBenchmark.java
 * @brief JMH-based performance comparison of vector similarity scorers for int7 quantized data.
 * 
 * Functional Intent: Evaluates the throughput of dot-product and euclidean distance 
 * calculations across three distinct implementations:
 * 1. Native Scalar: Standard Java loop for baseline comparison.
 * 2. Lucene Panama: Apache Lucene's implementation utilizing JDK Vector API (Panama).
 * 3. Elasticsearch Native: Custom optimized native scorers likely using platform-specific SIMD instructions.
 * 
 * Domain: Production Systems, Performance Engineering, Vector Search, SIMD.
 */
@Fork(value = 1, jvmArgsPrepend = { "--add-modules=jdk.incubator.vector" })
@Warmup(iterations = 3, time = 3)
@Measurement(iterations = 5, time = 3)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
public class Int7uScorerBenchmark {

    static {
        // Log Configuration: Required for native component initialization.
        LogConfigurator.configureESLogging();
    }

    @Param({ "96", "768", "1024" })
    public int dims;
    final int size = 2; // Fixed set of two vectors for direct comparison.

    Directory dir;
    IndexInput in;
    VectorScorerFactory factory;

    byte[] vec1, vec2;
    float vec1Offset;
    float vec2Offset;
    float scoreCorrectionConstant;

    UpdateableRandomVectorScorer luceneDotScorer;
    UpdateableRandomVectorScorer luceneSqrScorer;
    UpdateableRandomVectorScorer nativeDotScorer;
    UpdateableRandomVectorScorer nativeSqrScorer;

    RandomVectorScorer luceneDotScorerQuery;
    RandomVectorScorer nativeDotScorerQuery;
    RandomVectorScorer luceneSqrScorerQuery;
    RandomVectorScorer nativeSqrScorerQuery;

    /**
     * setup - Prepares temporary data and initializes competitive scorers.
     * 
     * Logic: 
     * 1. Generates random 7-bit quantized bytes.
     * 2. Persists vectors to a temporary MMapDirectory to simulate off-heap production behavior.
     * 3. Instantiates Lucene and Native scorers for both Dot Product and Euclidean metrics.
     */
    @Setup
    public void setup() throws IOException {
        var optionalVectorScorerFactory = VectorScorerFactory.instance();
        // Guard: Verify native provider availability for the current platform/arch.
        if (optionalVectorScorerFactory.isEmpty()) {
            String msg = "JDK=["
                + Runtime.version()
                + "], os.name=["
                + System.getProperty("os.name")
                + "], os.arch=["
                + System.getProperty("os.arch")
                + "]";
            throw new AssertionError("Vector scorer factory not present. Cannot run the benchmark. " + msg);
        }
        factory = optionalVectorScorerFactory.get();
        vec1 = new byte[dims];
        vec2 = new byte[dims];

        randomInt7BytesBetween(vec1);
        randomInt7BytesBetween(vec2);
        vec1Offset = ThreadLocalRandom.current().nextFloat();
        vec2Offset = ThreadLocalRandom.current().nextFloat();

        dir = new MMapDirectory(Files.createTempDirectory("nativeScalarQuantBench"));
        try (IndexOutput out = dir.createOutput("vector.data", IOContext.DEFAULT)) {
            // Serialization: Layout matches quantized byte vector format with trailing floating-point offsets.
            out.writeBytes(vec1, 0, vec1.length);
            out.writeInt(Float.floatToIntBits(vec1Offset));
            out.writeBytes(vec2, 0, vec2.length);
            out.writeInt(Float.floatToIntBits(vec2Offset));
        }
        in = dir.openInput("vector.data", IOContext.DEFAULT);
        var values = vectorValues(dims, 2, in, VectorSimilarityFunction.DOT_PRODUCT);
        scoreCorrectionConstant = values.getScalarQuantizer().getConstantMultiplier();
        
        // Scorer Initialization: Binding scorers to specific ordinals (0 and 1) for the pairwise comparison.
        luceneDotScorer = luceneScoreSupplier(values, VectorSimilarityFunction.DOT_PRODUCT).scorer();
        luceneDotScorer.setScoringOrdinal(0);
        values = vectorValues(dims, 2, in, VectorSimilarityFunction.EUCLIDEAN);
        luceneSqrScorer = luceneScoreSupplier(values, VectorSimilarityFunction.EUCLIDEAN).scorer();
        luceneSqrScorer.setScoringOrdinal(0);

        nativeDotScorer = factory.getInt7SQVectorScorerSupplier(DOT_PRODUCT, in, values, scoreCorrectionConstant).get().scorer();
        nativeDotScorer.setScoringOrdinal(0);
        nativeSqrScorer = factory.getInt7SQVectorScorerSupplier(EUCLIDEAN, in, values, scoreCorrectionConstant).get().scorer();
        nativeSqrScorer.setScoringOrdinal(0);

        // Setup for query-vector scenarios (float query vs quantized index).
        float[] queryVec = new float[dims];
        for (int i = 0; i < dims; i++) {
            queryVec[i] = ThreadLocalRandom.current().nextFloat();
        }
        luceneDotScorerQuery = luceneScorer(values, VectorSimilarityFunction.DOT_PRODUCT, queryVec);
        nativeDotScorerQuery = factory.getInt7SQVectorScorer(VectorSimilarityFunction.DOT_PRODUCT, values, queryVec).get();
        luceneSqrScorerQuery = luceneScorer(values, VectorSimilarityFunction.EUCLIDEAN, queryVec);
        nativeSqrScorerQuery = factory.getInt7SQVectorScorer(VectorSimilarityFunction.EUCLIDEAN, values, queryVec).get();
    }

    @TearDown
    public void teardown() throws IOException {
        IOUtils.close(dir, in);
    }

    /**
     * @benchmark dotProductLucene
     * @brief Measures Lucene's optimized dot product throughput.
     */
    @Benchmark
    public float dotProductLucene() throws IOException {
        return luceneDotScorer.score(1);
    }

    /**
     * @benchmark dotProductNative
     * @brief Measures Elasticsearch's native-accelerated dot product throughput.
     */
    @Benchmark
    public float dotProductNative() throws IOException {
        return nativeDotScorer.score(1);
    }

    /**
     * @benchmark dotProductScalar
     * @brief Measures unoptimized scalar loop dot product.
     * 
     * Logic: Implements the baseline similarity calculation including quantization 
     * correction and offset application.
     */
    @Benchmark
    public float dotProductScalar() {
        int dotProduct = 0;
        for (int i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
        }
        float adjustedDistance = dotProduct * scoreCorrectionConstant + vec1Offset + vec2Offset;
        return (1 + adjustedDistance) / 2;
    }

    @Benchmark
    public float dotProductLuceneQuery() throws IOException {
        return luceneDotScorerQuery.score(1);
    }

    @Benchmark
    public float dotProductNativeQuery() throws IOException {
        return nativeDotScorerQuery.score(1);
    }

    // -- square distance benchmarks

    @Benchmark
    public float squareDistanceLucene() throws IOException {
        return luceneSqrScorer.score(1);
    }

    @Benchmark
    public float squareDistanceNative() throws IOException {
        return nativeSqrScorer.score(1);
    }

    /**
     * @benchmark squareDistanceScalar
     * @brief Baseline Euclidean distance implementation.
     */
    @Benchmark
    public float squareDistanceScalar() {
        int squareDistance = 0;
        for (int i = 0; i < vec1.length; i++) {
            int diff = vec1[i] - vec2[i];
            squareDistance += diff * diff;
        }
        float adjustedDistance = squareDistance * scoreCorrectionConstant;
        return 1 / (1f + adjustedDistance);
    }

    @Benchmark
    public float squareDistanceLuceneQuery() throws IOException {
        return luceneSqrScorerQuery.score(1);
    }

    @Benchmark
    public float squareDistanceNativeQuery() throws IOException {
        return nativeSqrScorerQuery.score(1);
    }

    /**
     * vectorValues - Utility to create an off-heap representation of quantized data.
     */
    QuantizedByteVectorValues vectorValues(int dims, int size, IndexInput in, VectorSimilarityFunction sim) throws IOException {
        var sq = new ScalarQuantizer(0.1f, 0.9f, (byte) 7);
        var slice = in.slice("values", 0, in.length());
        return new OffHeapQuantizedByteVectorValues.DenseOffHeapVectorValues(dims, size, sq, false, sim, null, slice);
    }

    RandomVectorScorerSupplier luceneScoreSupplier(QuantizedByteVectorValues values, VectorSimilarityFunction sim) throws IOException {
        return new Lucene99ScalarQuantizedVectorScorer(null).getRandomVectorScorerSupplier(sim, values);
    }

    RandomVectorScorer luceneScorer(QuantizedByteVectorValues values, VectorSimilarityFunction sim, float[] queryVec) throws IOException {
        return new Lucene99ScalarQuantizedVectorScorer(null).getRandomVectorScorer(sim, values, queryVec);
    }

    // Unsigned int7 byte vectors have values in the range of 0 to 127 (inclusive).
    static final byte MIN_INT7_VALUE = 0;
    static final byte MAX_INT7_VALUE = 127;

    /**
     * randomInt7BytesBetween - populates an array with random 7-bit values.
     */
    static void randomInt7BytesBetween(byte[] bytes) {
        var random = ThreadLocalRandom.current();
        for (int i = 0, len = bytes.length; i < len;) {
            bytes[i++] = (byte) random.nextInt(MIN_INT7_VALUE, MAX_INT7_VALUE + 1);
        }
    }
}
