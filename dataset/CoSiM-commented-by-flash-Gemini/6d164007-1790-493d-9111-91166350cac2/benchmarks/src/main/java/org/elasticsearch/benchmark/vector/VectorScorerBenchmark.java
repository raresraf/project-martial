/**
 * @6d164007-1790-493d-9111-91166350cac2/benchmarks/src/main/java/org/elasticsearch/benchmark/vector/VectorScorerBenchmark.java
 * @brief JMH microbenchmark suite for evaluating performance across multiple Vector Scoring implementations.
 * Purpose: Compares computational efficiency of scalar, Lucene-optimized (SIMD/Panama), and Elasticsearch-native vector similarity kernels.
 * Domain: Information Retrieval, Vector Search, SIMD Optimization.
 * Metrics: Throughput (ops/µs) under varying dimensional constraints (96, 768, 1024).
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
 * Execution Strategy: 1 fork with incubation of Panama Vector API modules.
 * Invariant: Warmup iterations ensure JIT compilation (C2) stability before data collection.
 */
@Fork(value = 1, jvmArgsPrepend = { "--add-modules=jdk.incubator.vector" })
@Warmup(iterations = 3, time = 3)
@Measurement(iterations = 5, time = 3)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
public class VectorScorerBenchmark {

    static {
        // Functional Utility: Initializes logging to support native access diagnostics.
        LogConfigurator.configureESLogging(); 
    }

    /**
     * @Param dimensions: Tests standard embedding sizes common in vector databases.
     */
    @Param({ "96", "768", "1024" })
    int dims;
    int size = 2; // Fixed cardinality for direct comparison.

    Directory dir;
    IndexInput in;
    VectorScorerFactory factory;

    byte[] vec1;
    byte[] vec2;
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
     * @brief Orchestrates the memory and state initialization for benchmarking.
     * Logic: Pre-calculates quantized vectors and offsets to ensure benchmark iterations measure only scoring logic.
     * Memory Management: Utilizes MMapDirectory for off-heap vector storage, mirroring production access patterns.
     */
    @Setup
    public void setup() throws IOException {
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

        randomInt7BytesBetween(vec1);
        randomInt7BytesBetween(vec2);
        vec1Offset = ThreadLocalRandom.current().nextFloat();
        vec2Offset = ThreadLocalRandom.current().nextFloat();

        dir = new MMapDirectory(Files.createTempDirectory("nativeScalarQuantBench"));
        try (IndexOutput out = dir.createOutput("vector.data", IOContext.DEFAULT)) {
            out.writeBytes(vec1, 0, vec1.length);
            out.writeInt(Float.floatToIntBits(vec1Offset));
            out.writeBytes(vec2, 0, vec2.length);
            out.writeInt(Float.floatToIntBits(vec2Offset));
        }
        in = dir.openInput("vector.data", IOContext.DEFAULT);
        var values = vectorValues(dims, 2, in, VectorSimilarityFunction.DOT_PRODUCT);
        scoreCorrectionConstant = values.getScalarQuantizer().getConstantMultiplier();
        
        // Initialization: Pre-warms Lucene scorers for standard DOT_PRODUCT and EUCLIDEAN comparisons.
        luceneDotScorer = luceneScoreSupplier(values, VectorSimilarityFunction.DOT_PRODUCT).scorer();
        luceneDotScorer.setScoringOrdinal(0);
        values = vectorValues(dims, 2, in, VectorSimilarityFunction.EUCLIDEAN);
        luceneSqrScorer = luceneScoreSupplier(values, VectorSimilarityFunction.EUCLIDEAN).scorer();
        luceneSqrScorer.setScoringOrdinal(0);

        // Initialization: Pre-warms Elasticsearch native scorers utilizing SIMD intrinsics.
        nativeDotScorer = factory.getInt7SQVectorScorerSupplier(DOT_PRODUCT, in, values, scoreCorrectionConstant).get().scorer();
        nativeDotScorer.setScoringOrdinal(0);
        nativeSqrScorer = factory.getInt7SQVectorScorerSupplier(EUCLIDEAN, in, values, scoreCorrectionConstant).get().scorer();
        nativeSqrScorer.setScoringOrdinal(0);

        // Logic: Setup for query vector scoring (on-the-fly quantization vs pre-quantized storage).
        float[] queryVec = new float[dims];
        for (int i = 0; i < dims; i++) {
            queryVec[i] = ThreadLocalRandom.current().nextFloat();
        }
        luceneDotScorerQuery = luceneScorer(values, VectorSimilarityFunction.DOT_PRODUCT, queryVec);
        nativeDotScorerQuery = factory.getInt7SQVectorScorer(VectorSimilarityFunction.DOT_PRODUCT, values, queryVec).get();
        luceneSqrScorerQuery = luceneScorer(values, VectorSimilarityFunction.EUCLIDEAN, queryVec);
        nativeSqrScorerQuery = factory.getInt7SQVectorScorer(VectorSimilarityFunction.EUCLIDEAN, values, queryVec).get();

        // Functional Utility: Cross-validation to ensure all implementations converge on identical score values.
        var f1 = dotProductLucene();
        var f2 = dotProductNative();
        var f3 = dotProductScalar();
        if (f1 != f2) {
            throw new AssertionError("lucene[" + f1 + "] != " + "native[" + f2 + "]");
        }
        if (f1 != f3) {
            throw new AssertionError("lucene[" + f1 + "] != " + "scalar[" + f3 + "]");
        }
        
        f1 = squareDistanceLucene();
        f2 = squareDistanceNative();
        f3 = squareDistanceScalar();
        if (f1 != f2) {
            throw new AssertionError("lucene[" + f1 + "] != " + "native[" + f2 + "]");
        }
        if (f1 != f3) {
            throw new AssertionError("lucene[" + f1 + "] != " + "scalar[" + f3 + "]");
        }

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
     * @brief Resource cleanup ensuring persistent temp directories are removed.
     */
    @TearDown
    public void teardown() throws IOException {
        IOUtils.close(dir, in);
    }

    /**
     * @brief Measures Lucene's internal scalar-quantized dot product scoring.
     */
    @Benchmark
    public float dotProductLucene() throws IOException {
        return luceneDotScorer.score(1);
    }

    /**
     * @brief Measures Elasticsearch's native SIMD-accelerated dot product scoring.
     */
    @Benchmark
    public float dotProductNative() throws IOException {
        return nativeDotScorer.score(1);
    }

    /**
     * @brief Baseline scalar implementation of quantized dot product.
     * Logic: Standard loop-based accumulation; acts as a performance floor.
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
     * @brief Baseline scalar implementation of squared Euclidean distance.
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
     * @brief Helper for generating off-heap vector views.
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

    // Constraints: Unsigned int7 byte vectors define the valid value range [0, 127].
    static final byte MIN_INT7_VALUE = 0;
    static final byte MAX_INT7_VALUE = 127;

    /**
     * @brief Utility for generating randomized test data within quantization bounds.
     */
    static void randomInt7BytesBetween(byte[] bytes) {
        var random = ThreadLocalRandom.current();
        for (int i = 0, len = bytes.length; i < len;) {
            bytes[i++] = (byte) random.nextInt(MIN_INT7_VALUE, MAX_INT7_VALUE + 1);
        }
    }
}
