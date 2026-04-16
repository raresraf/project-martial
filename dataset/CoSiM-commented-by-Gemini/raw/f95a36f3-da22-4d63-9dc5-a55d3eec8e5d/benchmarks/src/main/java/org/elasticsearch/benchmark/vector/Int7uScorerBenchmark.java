/**
 * @file Int7uScorerBenchmark.java
 * @brief JMH benchmark for comparing performance of scalar quantized vector similarity functions.
 *
 * This benchmark evaluates and compares the throughput of different implementations for calculating
 * dot product and squared Euclidean distance on 7-bit unsigned integer quantized vectors.
 *
 * The implementations being compared are:
 * 1.  **Scalar**: A simple, pure Java implementation that iterates through the vectors.
 * 2.  **Lucene**: The implementation provided by Apache Lucene, which may use Panama SIMD optimizations.
 * 3.  **Native**: Elasticsearch's custom SIMD implementation, accessed through JNI.
 *
 * The benchmark is configured to run with different vector dimensions (96, 768, 1024) to
 * assess performance across various common embedding sizes. It measures throughput in
 * microseconds.
 *
 * To run this benchmark: `./gradlew -p benchmarks run --args 'Int7uScorerBenchmark'`
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
 * Run with ./gradlew -p benchmarks run --args 'Int7uScorerBenchmark'
 */
public class Int7uScorerBenchmark {

    static {
        LogConfigurator.configureESLogging(); // native access requires logging to be initialized
    }

    @Param({ "96", "768", "1024" })
    public int dims;
    final int size = 2; // there are only two vectors to compare

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
     * Sets up the benchmark environment before each run.
     *
     * This method initializes two random 7-bit unsigned integer vectors and writes them to a memory-mapped
     * directory to simulate reading from a Lucene index. It prepares various `RandomVectorScorer`
     * instances (Lucene's implementation and Elasticsearch's native one) for both dot product and
     * squared Euclidean distance calculations. This pre-computation ensures that the benchmark methods
     * only measure the scoring time and not the setup overhead.
     *
     * @throws IOException if an I/O error occurs during file setup.
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
        luceneDotScorer = luceneScoreSupplier(values, VectorSimilarityFunction.DOT_PRODUCT).scorer();
        luceneDotScorer.setScoringOrdinal(0);
        values = vectorValues(dims, 2, in, VectorSimilarityFunction.EUCLIDEAN);
        luceneSqrScorer = luceneScoreSupplier(values, VectorSimilarityFunction.EUCLIDEAN).scorer();
        luceneSqrScorer.setScoringOrdinal(0);

        nativeDotScorer = factory.getInt7SQVectorScorerSupplier(DOT_PRODUCT, in, values, scoreCorrectionConstant).get().scorer();
        nativeDotScorer.setScoringOrdinal(0);
        nativeSqrScorer = factory.getInt7SQVectorScorerSupplier(EUCLIDEAN, in, values, scoreCorrectionConstant).get().scorer();
        nativeSqrScorer.setScoringOrdinal(0);

        // setup for getInt7SQVectorScorer / query vector scoring
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

    // --- Dot Product Benchmarks ---

    /**
     * Benchmarks Lucene's implementation of dot product similarity scoring.
     */
    @Benchmark
    public float dotProductLucene() throws IOException {
        return luceneDotScorer.score(1);
    }

    /**
     * Benchmarks Elasticsearch's native SIMD implementation of dot product similarity scoring.
     */
    @Benchmark
    public float dotProductNative() throws IOException {
        return nativeDotScorer.score(1);
    }

    /**
     * Benchmarks a scalar, pure Java implementation of dot product calculation as a baseline.
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
     * Benchmarks Lucene's implementation of dot product similarity scoring against a query vector.
     */
    @Benchmark
    public float dotProductLuceneQuery() throws IOException {
        return luceneDotScorerQuery.score(1);
    }

    /**
     * Benchmarks Elasticsearch's native SIMD implementation of dot product similarity scoring against a query vector.
     */
    @Benchmark
    public float dotProductNativeQuery() throws IOException {
        return nativeDotScorerQuery.score(1);
    }

    // --- Squared Euclidean Distance Benchmarks ---

    /**
     * Benchmarks Lucene's implementation of squared Euclidean distance scoring.
     */
    @Benchmark
    public float squareDistanceLucene() throws IOException {
        return luceneSqrScorer.score(1);
    }

    /**
     * Benchmarks Elasticsearch's native SIMD implementation of squared Euclidean distance scoring.
     */
    @Benchmark
    public float squareDistanceNative() throws IOException {
        return nativeSqrScorer.score(1);
    }

    /**
     * Benchmarks a scalar, pure Java implementation of squared Euclidean distance as a baseline.
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
     * Benchmarks Lucene's implementation of squared Euclidean distance scoring against a query vector.
     */
    @Benchmark
    public float squareDistanceLuceneQuery() throws IOException {
        return luceneSqrScorerQuery.score(1);
    }

    /**
     * Benchmarks Elasticsearch's native SIMD implementation of squared Euclidean distance scoring against a query vector.
     */
    @Benchmark
    public float squareDistanceNativeQuery() throws IOException {
        return nativeSqrScorerQuery.score(1);
    }

    /**
     * Creates a `QuantizedByteVectorValues` instance to simulate reading quantized vectors
     * from a Lucene index.
     */
    QuantizedByteVectorValues vectorValues(int dims, int size, IndexInput in, VectorSimilarityFunction sim) throws IOException {
        var sq = new ScalarQuantizer(0.1f, 0.9f, (byte) 7);
        var slice = in.slice("values", 0, in.length());
        return new OffHeapQuantizedByteVectorValues.DenseOffHeapVectorValues(dims, size, sq, false, sim, null, slice);
    }

    /**
     * Creates a Lucene-based random vector scorer supplier.
     */
    RandomVectorScorerSupplier luceneScoreSupplier(QuantizedByteVectorValues values, VectorSimilarityFunction sim) throws IOException {
        return new Lucene99ScalarQuantizedVectorScorer(null).getRandomVectorScorerSupplier(sim, values);
    }

    /**
     * Creates a Lucene-based random vector scorer for a given query vector.
     */
    RandomVectorScorer luceneScorer(QuantizedByteVectorValues values, VectorSimilarityFunction sim, float[] queryVec) throws IOException {
        return new Lucene99ScalarQuantizedVectorScorer(null).getRandomVectorScorer(sim, values, queryVec);
    }

    // Unsigned int7 byte vectors have values in the range of 0 to 127 (inclusive).
    static final byte MIN_INT7_VALUE = 0;
    static final byte MAX_INT7_VALUE = 127;

    /**
     * Fills a byte array with random values in the 7-bit unsigned integer range [0, 127].
     * @param bytes The byte array to fill.
     */
    static void randomInt7BytesBetween(byte[] bytes) {
        var random = ThreadLocalRandom.current();
        for (int i = 0, len = bytes.length; i < len;) {
            bytes[i++] = (byte) random.nextInt(MIN_INT7_VALUE, MAX_INT7_VALUE + 1);
        }
    }
}
