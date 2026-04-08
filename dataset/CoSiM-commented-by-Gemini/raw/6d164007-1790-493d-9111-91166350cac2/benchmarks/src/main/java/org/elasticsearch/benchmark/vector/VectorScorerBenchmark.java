/**
 * @file VectorScorerBenchmark.java
 * @brief JMH benchmarks for comparing different implementations of scalar quantized vector similarity scoring.
 * @details This file contains benchmarks to evaluate the performance of vector scoring for 7-bit quantized vectors.
 * It compares three different approaches:
 * 1. A pure Java scalar implementation.
 * 2. The Apache Lucene implementation, which may leverage the Java Vector API (Panama).
 * 3. A native JNI implementation provided by Elasticsearch, likely using SIMD instructions.
 * The benchmarks cover both dot product and Euclidean (squared) distance similarities.
 *
 * To run: ./gradlew -p benchmarks run --args 'VectorScorerBenchmark'
 */

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

@Fork(value = 1, jvmArgsPrepend = { "--add-modules=jdk.incubator.vector" })
@Warmup(iterations = 3, time = 3)
@Measurement(iterations = 5, time = 3)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
/**
 * Benchmark that compares various scalar quantized vector similarity function
 * implementations;: scalar, lucene's panama-ized, and Elasticsearch's native.
 */
public class VectorScorerBenchmark {

    static {
        //- Functional Utility: Initialize Elasticsearch logging, a prerequisite for using the native JNI library.
        LogConfigurator.configureESLogging();
    }

    /**
     * @Param The dimensions of the vectors being benchmarked.
     */
    @Param({ "96", "768", "1024" })
    int dims;
    int size = 2; // there are only two vectors to compare

    //- State: Holds data structures needed across benchmarks, such as the Lucene directory and index I/O.
    Directory dir;
    IndexInput in;
    VectorScorerFactory factory;

    //- State: Raw byte vectors and their quantization offsets.
    byte[] vec1;
    byte[] vec2;
    float vec1Offset;
    float vec2Offset;
    float scoreCorrectionConstant;

    //- State: Scorers for comparing two vectors within the index.
    UpdateableRandomVectorScorer luceneDotScorer;
    UpdateableRandomVectorScorer luceneSqrScorer;
    UpdateableRandomVectorScorer nativeDotScorer;
    UpdateableRandomVectorScorer nativeSqrScorer;

    //- State: Scorers for comparing a query vector against a vector in the index.
    RandomVectorScorer luceneDotScorerQuery;
    RandomVectorScorer nativeDotScorerQuery;
    RandomVectorScorer luceneSqrScorerQuery;
    RandomVectorScorer nativeSqrScorerQuery;

    /**
     * @brief Sets up the benchmark state before execution.
     * @details This method prepares all necessary data and objects for the benchmarks. It generates random
     * quantized vectors, writes them to a temporary Lucene directory, and initializes scorer instances
     * for each implementation (Lucene, Native, Scalar) and similarity function being tested. It also
     * performs a sanity check to ensure all implementations produce identical results.
     * @throws IOException If an I/O error occurs during file setup.
     */
    @Setup
    public void setup() throws IOException {
        //- Block Logic: Obtain an instance of the native VectorScorerFactory, asserting its availability.
        var optionalVectorScorerFactory = VectorScorerFactory.instance();
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

        //- Functional Utility: Generate random 7-bit quantized vectors and offsets.
        randomInt7BytesBetween(vec1);
        randomInt7BytesBetween(vec2);
        vec1Offset = ThreadLocalRandom.current().nextFloat();
        vec2Offset = ThreadLocalRandom.current().nextFloat();

        //- Block Logic: Create a temporary MMapDirectory and write the vector data to it to simulate a Lucene index.
        dir = new MMapDirectory(Files.createTempDirectory("nativeScalarQuantBench"));
        try (IndexOutput out = dir.createOutput("vector.data", IOContext.DEFAULT)) {
            out.writeBytes(vec1, 0, vec1.length);
            out.writeInt(Float.floatToIntBits(vec1Offset));
            out.writeBytes(vec2, 0, vec2.length);
            out.writeInt(Float.floatToIntBits(vec2Offset));
        }
        in = dir.openInput("vector.data", IOContext.DEFAULT);

        //- Block Logic: Initialize scorers for comparing two vectors from the index (doc-to-doc scoring).
        var values = vectorValues(dims, 2, in, VectorSimilarityFunction.DOT_PRODUCT);
        scoreCorrectionConstant = values.getScalarQuantizer().getConstantMultiplier();
        luceneDotScorer = luceneScoreSupplier(values, VectorSimilarityFunction.DOT_PRODUCT).scorer();
        luceneDotScorer.setScoringOrdinal(0);
        values = vectorValues(dims, 2, in, VectorSimilarityFunction.EUCLIDEAN);
        luceneSqrScorer = luceneScoreSupplier(values, VectorSimilarityFunction.EUCLIDEAN).scorer();
        luceneSqrScorer.setScoringOrdinal(0);

        nativeDotScorer = factory.getInt7SQVectorScorerSupplier(DOT_PRODUCT, in, values, scoreCorrectionConstant).get().scorer();
        nativeDotScorer.setScoringOrdinal(0);
        nativeSqrScorer = factory.getInt7SQVectorScorerSupplier(EUCLIDEAN, in, values, scoreCorrectionConstant).get().scorer();
        nativeSqrScorer.setScoringOrdinal(0);

        //- Block Logic: Initialize scorers for comparing a new query vector against an indexed vector.
        float[] queryVec = new float[dims];
        for (int i = 0; i < dims; i++) {
            queryVec[i] = ThreadLocalRandom.current().nextFloat();
        }
        luceneDotScorerQuery = luceneScorer(values, VectorSimilarityFunction.DOT_PRODUCT, queryVec);
        nativeDotScorerQuery = factory.getInt7SQVectorScorer(VectorSimilarityFunction.DOT_PRODUCT, values, queryVec).get();
        luceneSqrScorerQuery = luceneScorer(values, VectorSimilarityFunction.EUCLIDEAN, queryVec);
        nativeSqrScorerQuery = factory.getInt7SQVectorScorer(VectorSimilarityFunction.EUCLIDEAN, values, queryVec).get();

        //- Block Logic: Perform a sanity check to ensure all implementations produce the same score before benchmarking.
        var f1 = dotProductLucene();
        var f2 = dotProductNative();
        var f3 = dotProductScalar();
        if (f1 != f2) {
            throw new AssertionError("lucene[" + f1 + "] != " + "native[" + f2 + "]");
        }
        if (f1 != f3) {
            throw new AssertionError("lucene[" + f1 + "] != " + "scalar[" + f3 + "]");
        }
        //- Sanity check for square distance
        f1 = squareDistanceLucene();
        f2 = squareDistanceNative();
        f3 = squareDistanceScalar();
        if (f1 != f2) {
            throw new AssertionError("lucene[" + f1 + "] != " + "native[" + f2 + "]");
        }
        if (f1 != f3) {
            throw new AssertionError("lucene[" + f1 + "] != " + "scalar[" + f3 + "]");
        }
        //- Sanity check for query scoring
        var q1 = dotProductLuceneQuery();
        var q2 = dotProductNativeQuery();
        if (q1 != q2) {
            throw new AssertionError("query: lucene[" + q1 + "] != " + "native[" + q2 + "]");
        }

        var sqr1 = squareDistanceLuceneQuery();
        var sqr2 = squareDistanceNativeQuery();
        if (sqr1 != sqr2) {
            throw new AssertionError("query: lucene[" + q1 + "] != " + "native[" + q2 + "]");
        }
    }

    /**
     * @brief Cleans up resources after the benchmark run.
     * @throws IOException If an I/O error occurs.
     */
    @TearDown
    public void teardown() throws IOException {
        IOUtils.close(dir, in);
    }

    /**
     * @brief Benchmarks dot product using Lucene's implementation.
     */
    @Benchmark
    public float dotProductLucene() throws IOException {
        return luceneDotScorer.score(1);
    }

    /**
     * @brief Benchmarks dot product using Elasticsearch's native JNI/SIMD implementation.
     */
    @Benchmark
    public float dotProductNative() throws IOException {
        return nativeDotScorer.score(1);
    }

    /**
     * @brief Benchmarks dot product using a pure Java scalar loop, as a baseline.
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

    /**
     * @brief Benchmarks dot product with a query vector using Lucene's implementation.
     */
    @Benchmark
    public float dotProductLuceneQuery() throws IOException {
        return luceneDotScorerQuery.score(1);
    }

    /**
     * @brief Benchmarks dot product with a query vector using Elasticsearch's native implementation.
     */
    @Benchmark
    public float dotProductNativeQuery() throws IOException {
        return nativeDotScorerQuery.score(1);
    }


    /**
     * @brief Benchmarks Euclidean squared distance using Lucene's implementation.
     */
    @Benchmark
    public float squareDistanceLucene() throws IOException {
        return luceneSqrScorer.score(1);
    }

    /**
     * @brief Benchmarks Euclidean squared distance using Elasticsearch's native JNI/SIMD implementation.
     */
    @Benchmark
    public float squareDistanceNative() throws IOException {
        return nativeSqrScorer.score(1);
    }

    /**
     * @brief Benchmarks Euclidean squared distance using a pure Java scalar loop, as a baseline.
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

    /**
     * @brief Benchmarks Euclidean squared distance with a query vector using Lucene's implementation.
     */
    @Benchmark
    public float squareDistanceLuceneQuery() throws IOException {
        return luceneSqrScorerQuery.score(1);
    }

    /**
     * @brief Benchmarks Euclidean squared distance with a query vector using Elasticsearch's native implementation.
     */
    @Benchmark
    public float squareDistanceNativeQuery() throws IOException {
        return nativeSqrScorerQuery.score(1);
    }

    /**
     * @brief Helper to create a Lucene `QuantizedByteVectorValues` object from index data.
     */
    QuantizedByteVectorValues vectorValues(int dims, int size, IndexInput in, VectorSimilarityFunction sim) throws IOException {
        var sq = new ScalarQuantizer(0.1f, 0.9f, (byte) 7);
        var slice = in.slice("values", 0, in.length());
        return new OffHeapQuantizedByteVectorValues.DenseOffHeapVectorValues(dims, size, sq, false, sim, null, slice);
    }

    /**
     * @brief Helper to create a Lucene scorer supplier for doc-to-doc scoring.
     */
    RandomVectorScorerSupplier luceneScoreSupplier(QuantizedByteVectorValues values, VectorSimilarityFunction sim) throws IOException {
        return new Lucene99ScalarQuantizedVectorScorer(null).getRandomVectorScorerSupplier(sim, values);
    }

    /**
     * @brief Helper to create a Lucene scorer for query-to-doc scoring.
     */
    RandomVectorScorer luceneScorer(QuantizedByteVectorValues values, VectorSimilarityFunction sim, float[] queryVec) throws IOException {
        return new Lucene99ScalarQuantizedVectorScorer(null).getRandomVectorScorer(sim, values, queryVec);
    }

    // Unsigned int7 byte vectors have values in the range of 0 to 127 (inclusive).
    static final byte MIN_INT7_VALUE = 0;
    static final byte MAX_INT7_VALUE = 127;

    /**
     * @brief Fills a byte array with random values between 0 and 127, simulating 7-bit quantized data.
     * @param bytes The byte array to fill.
     */
    static void randomInt7BytesBetween(byte[] bytes) {
        var random = ThreadLocalRandom.current();
        for (int i = 0, len = bytes.length; i < len;) {
            bytes[i++] = (byte) random.nextInt(MIN_INT7_VALUE, MAX_INT7_VALUE + 1);
        }
    }
}
