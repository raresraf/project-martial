/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

/**
 * @file DefaultIVFVectorsWriter.java
 * @brief This file implements the default IVF (Inverted Vector File) vectors writer for Elasticsearch.
 * It leverages Hierarchical K-Means clustering to partition the vector space, enhancing search efficiency
 * for vector similarity.
 * Algorithm: Hierarchical K-Means for vector space partitioning, scalar quantization for compression.
 * Time Complexity: Clustering involves iterative K-Means, which is typically O(I * K * N * D) where I is iterations,
 * K is number of clusters, N is number of vectors, D is vector dimension. Writing posting lists and quantized
 * vectors is O(N * D).
 * Space Complexity: O(N * D) for vectors during clustering, O(K * D) for centroids, plus storage for posting lists.
 */

package org.elasticsearch.index.codec.vectors;

import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.index.codec.vectors.cluster.HierarchicalKMeans;
import org.elasticsearch.index.codec.vectors.cluster.KMeansResult;
import org.elasticsearch.logging.LogManager;
import org.elasticsearch.logging.Logger;
import org.elasticsearch.simdvec.ES91OSQVectorsScorer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Default implementation of {@link IVFVectorsWriter}. It uses {@link HierarchicalKMeans} algorithm to
 * partition the vector space, and then stores the centroids and posting list in a sequential
 * fashion.
 *
 * @brief Manages the writing of Inverted Vector File (IVF) structures for vector fields.
 * This class orchestrates the clustering of vectors, quantization of centroids, and the
 * persistence of both the centroids and their associated posting lists (document IDs
 * grouped by cluster). It is a core component for efficient vector similarity search.
 */
public class DefaultIVFVectorsWriter extends IVFVectorsWriter {
    private static final Logger logger = LogManager.getLogger(DefaultIVFVectorsWriter.class);

    private final int vectorPerCluster;

    /**
     * @brief Constructs a new DefaultIVFVectorsWriter.
     * Initializes the writer with the segment write state and the number of vectors desired per cluster.
     * @param state The current segment write state, providing access to Lucene's I/O context.
     * @param rawVectorDelegate A delegate writer for handling raw vector data, used for fallback or specific storage needs.
     * @param vectorPerCluster The target number of vectors to be assigned to each cluster, influencing clustering granularity.
     */
    public DefaultIVFVectorsWriter(SegmentWriteState state, FlatVectorsWriter rawVectorDelegate, int vectorPerCluster) throws IOException {
        super(state, rawVectorDelegate);
        this.vectorPerCluster = vectorPerCluster;
    }

    /**
     * @brief Builds and writes the inverted posting lists for each centroid.
     * This method iterates through the cluster assignments, quantizes the vectors,
     * and writes the document IDs and quantized vectors to the postings output.
     * @param fieldInfo Information about the vector field.
     * @param centroidSupplier Supplies the centroids for each cluster.
     * @param floatVectorValues The source of all float vector values.
     * @param postingsOutput The IndexOutput to write the posting lists to.
     * @param assignmentsByCluster A 2D array where each inner array contains the ordinal assignments for a specific cluster.
     * @return An array of file offsets, where each offset points to the beginning of a cluster's posting list in `postingsOutput`.
     * @throws IOException If an I/O error occurs during writing.
     */
    @Override
    long[] buildAndWritePostingsLists(
        FieldInfo fieldInfo,
        CentroidSupplier centroidSupplier,
        FloatVectorValues floatVectorValues,
        IndexOutput postingsOutput,
        int[][] assignmentsByCluster
    ) throws IOException {
        // write the posting lists
        final long[] offsets = new long[centroidSupplier.size()];
        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
        DocIdsWriter docIdsWriter = new DocIdsWriter();
        DiskBBQBulkWriter bulkWriter = new DiskBBQBulkWriter.OneBitDiskBBQBulkWriter(
            ES91OSQVectorsScorer.BULK_SIZE,
            quantizer,
            floatVectorValues,
            postingsOutput
        );
        // Block Logic: Iterate through each centroid to build and write its corresponding posting list.
        // Precondition: `centroidSupplier` and `assignmentsByCluster` are initialized and contain valid data.
        // Invariant: For each centroid, its posting list (document IDs and quantized vectors) is written.
        for (int c = 0; c < centroidSupplier.size(); c++) {
            float[] centroid = centroidSupplier.centroid(c);
            // TODO: add back in sorting vectors by distance to centroid
            int[] cluster = assignmentsByCluster[c];
            // TODO align???
            offsets[c] = postingsOutput.getFilePointer();
            int size = cluster.length;
            postingsOutput.writeVInt(size);
            postingsOutput.writeInt(Float.floatToIntBits(VectorUtil.dotProduct(centroid, centroid)));
            // TODO we might want to consider putting the docIds in a separate file
            // to aid with only having to fetch vectors from slower storage when they are required
            // keeping them in the same file indicates we pull the entire file into cache
            docIdsWriter.writeDocIds(j -> floatVectorValues.ordToDoc(cluster[j]), size, postingsOutput);
            bulkWriter.writeOrds(j -> cluster[j], cluster.length, centroid);
        }

        // Block Logic: Log cluster quality statistics for debugging and analysis, if debug logging is enabled.
        // Precondition: `assignmentsByCluster` is populated with cluster assignments.
        // Invariant: Cluster statistics are computed and logged without altering the program state.
        if (logger.isDebugEnabled()) {
            printClusterQualityStatistics(assignmentsByCluster);
        }

        return offsets;
    }

    /**
     * @brief Prints statistics about the quality of the generated clusters.
     * This method calculates and logs the minimum, maximum, mean, standard deviation,
     * and variance of the cluster sizes, providing insights into the distribution
     * and effectiveness of the clustering.
     * @param clusters A 2D array representing the clusters, where each inner array contains the document ordinals assigned to a cluster.
     */
    private static void printClusterQualityStatistics(int[][] clusters) {
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        float mean = 0;
        float m2 = 0;
        // iteratively compute the variance & mean
        int count = 0;
        // Block Logic: Iterate through each cluster to compute aggregate statistics on cluster size.
        // Precondition: `clusters` array is initialized.
        // Invariant: `min`, `max`, `mean`, `m2`, and `count` are updated to reflect the statistics of processed clusters.
        for (int[] cluster : clusters) {
            count += 1;
            if (cluster == null) {
                continue;
            }
            float delta = cluster.length - mean;
            mean += delta / count;
            m2 += delta * (cluster.length - mean);
            min = Math.min(min, cluster.length);
            max = Math.max(max, cluster.length);
        }
        float variance = m2 / (clusters.length - 1);
        logger.debug(
            "Centroid count: {} min: {} max: {} mean: {} stdDev: {} variance: {}",
            clusters.length,
            min,
            max,
            mean,
            Math.sqrt(variance),
            variance
        );
    }

    /**
     * @brief Creates a supplier for centroids that are stored off-heap.
     * This method initializes an `OffHeapCentroidSupplier` which reads centroid data from an `IndexInput`,
     * optimizing memory usage by not loading all centroids into JVM heap space at once.
     * @param centroidsInput The IndexInput from which centroid data can be read.
     * @param numCentroids The total number of centroids.
     * @param fieldInfo Information about the vector field, used to determine vector dimension.
     * @param globalCentroid The global centroid of all vectors (not directly used by this supplier, but part of the signature).
     * @return An instance of `OffHeapCentroidSupplier`.
     */
    @Override
    CentroidSupplier createCentroidSupplier(IndexInput centroidsInput, int numCentroids, FieldInfo fieldInfo, float[] globalCentroid) {
        return new OffHeapCentroidSupplier(centroidsInput, numCentroids, fieldInfo);
    }

    /**
     * @brief Writes the calculated centroids to the centroid output stream.
     * This method quantizes each centroid using `OptimizedScalarQuantizer` and then writes
     * both the quantized byte representation and the raw float values of the centroids
     * to the `centroidOutput`.
     * @param centroids A 2D array of float arrays, where each inner array is a centroid vector.
     * @param fieldInfo Information about the vector field.
     * @param globalCentroid The global centroid of all vectors, used for scalar quantization.
     * @param centroidOutput The IndexOutput to write the centroid data to.
     * @throws IOException If an I/O error occurs during writing.
     */
    static void writeCentroids(float[][] centroids, FieldInfo fieldInfo, float[] globalCentroid, IndexOutput centroidOutput)
        throws IOException {
        final OptimizedScalarQuantizer osq = new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
        byte[] quantizedScratch = new byte[fieldInfo.getVectorDimension()];
        float[] centroidScratch = new float[fieldInfo.getVectorDimension()];
        // TODO do we want to store these distances as well for future use?
        // TODO: sort centroids by global centroid (was doing so previously here)
        // TODO: sorting tanks recall possibly because centroids ordinals no longer are aligned
        // Block Logic: Iterate through each centroid to quantize and write it.
        // Precondition: `centroids` array is populated, `fieldInfo` and `globalCentroid` are valid.
        // Invariant: Each centroid is scalar quantized and written to `centroidOutput` in both quantized and raw forms.
        for (float[] centroid : centroids) {
            System.arraycopy(centroid, 0, centroidScratch, 0, centroid.length);
            OptimizedScalarQuantizer.QuantizationResult result = osq.scalarQuantize(
                centroidScratch,
                quantizedScratch,
                (byte) 4,
                globalCentroid
            );
            writeQuantizedValue(centroidOutput, quantizedScratch, result);
        }
        final ByteBuffer buffer = ByteBuffer.allocate(fieldInfo.getVectorDimension() * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        // Block Logic: Write the raw float values of each centroid.
        // Precondition: `centroids` array is populated.
        // Invariant: Raw float values of each centroid are written to `centroidOutput`.
        for (float[] centroid : centroids) {
            buffer.asFloatBuffer().put(centroid);
            centroidOutput.writeBytes(buffer.array(), buffer.array().length);
        }
    }

    /**
     * @brief Calculates and writes centroids for the given field, considering prior generated clusters from mergeState.
     * This method is an overload that internally calls the more comprehensive `calculateAndWriteCentroids` method.
     * @param fieldInfo Merging field information.
     * @param floatVectorValues The float vector values to merge.
     * @param centroidOutput The IndexOutput to write the centroid data to.
     * @param mergeState The merge state, potentially containing information about previously generated clusters.
     * @param globalCentroid The global centroid, calculated by this method and used to quantize the centroids.
     * @return The vector assignments, soar assignments, and if asked the centroids themselves that were computed.
     * @throws IOException If an I/O error occurs.
     */
    CentroidAssignments calculateAndWriteCentroids(
        FieldInfo fieldInfo,
        FloatVectorValues floatVectorValues,
        IndexOutput centroidOutput,
        MergeState mergeState,
        float[] globalCentroid
    ) throws IOException {
        // TODO: take advantage of prior generated clusters from mergeState in the future
        return calculateAndWriteCentroids(fieldInfo, floatVectorValues, centroidOutput, globalCentroid, false);
    }

    /**
     * @brief Calculates and writes centroids for the given field.
     * This method is an overload that internally calls the more comprehensive `calculateAndWriteCentroids` method
     * with `cacheCentroids` set to true.
     * @param fieldInfo Merging field information.
     * @param floatVectorValues The float vector values to merge.
     * @param centroidOutput The IndexOutput to write the centroid data to.
     * @param globalCentroid The global centroid, calculated by this method and used to quantize the centroids.
     * @return The vector assignments, soar assignments, and the centroids themselves that were computed.
     * @throws IOException If an I/O error occurs.
     */
    CentroidAssignments calculateAndWriteCentroids(
        FieldInfo fieldInfo,
        FloatVectorValues floatVectorValues,
        IndexOutput centroidOutput,
        float[] globalCentroid
    ) throws IOException {
        return calculateAndWriteCentroids(fieldInfo, floatVectorValues, centroidOutput, globalCentroid, true);
    }

    /**
     * Calculate the centroids for the given field and write them to the given centroid output.
     * We use the {@link HierarchicalKMeans} algorithm to partition the space of all vectors across merging segments
     *
     * @param fieldInfo merging field info
     * @param floatVectorValues the float vector values to merge
     * @param centroidOutput the centroid output
     * @param globalCentroid the global centroid, calculated by this method and used to quantize the centroids
     * @param cacheCentroids whether the centroids are kept or discarded once computed
     * @return the vector assignments, soar assignments, and if asked the centroids themselves that were computed
     * @throws IOException if an I/O error occurs
     */
    /**
     * Calculate the centroids for the given field and write them to the given centroid output.
     * We use the {@link HierarchicalKMeans} algorithm to partition the space of all vectors across merging segments
     *
     * @param fieldInfo merging field info
     * @param floatVectorValues the float vector values to merge
     * @param centroidOutput the centroid output
     * @param globalCentroid the global centroid, calculated by this method and used to quantize the centroids
     * @param cacheCentroids whether the centroids are kept or discarded once computed
     * @return the vector assignments, soar assignments, and if asked the centroids themselves that were computed
     * @throws IOException if an I/O error occurs
     */
    CentroidAssignments calculateAndWriteCentroids(
        FieldInfo fieldInfo,
        FloatVectorValues floatVectorValues,
        IndexOutput centroidOutput,
        float[] globalCentroid,
        boolean cacheCentroids
    ) throws IOException {

        long nanoTime = System.nanoTime();

        // Block Logic: Perform Hierarchical K-Means clustering to determine centroids and vector assignments.
        // Precondition: `floatVectorValues` contains the vectors to be clustered, `vectorPerCluster` is configured.
        // Invariant: `kMeansResult` holds the computed centroids and vector-to-cluster assignments.
        // TODO: consider hinting / bootstrapping hierarchical kmeans with the prior segments centroids
        KMeansResult kMeansResult = new HierarchicalKMeans(floatVectorValues.dimension()).cluster(floatVectorValues, vectorPerCluster);
        float[][] centroids = kMeansResult.centroids();
        int[] assignments = kMeansResult.assignments();
        int[] soarAssignments = kMeansResult.soarAssignments();

        // Block Logic: Accumulate centroid values to compute a global centroid.
        // Precondition: `centroids` array is populated.
        // Invariant: `globalCentroid` sums up the component values of all centroids.
        // TODO: for flush we are doing this over the vectors and here centroids which seems duplicative
        // preliminary tests suggest recall is good using only centroids but need to do further evaluation
        // TODO: push this logic into vector util?
        for (float[] centroid : centroids) {
            for (int j = 0; j < centroid.length; j++) {
                globalCentroid[j] += centroid[j];
            }
        }
        // Block Logic: Normalize the accumulated global centroid sum by the number of centroids.
        // Precondition: `globalCentroid` contains the sum of centroid components, `centroids.length` is non-zero.
        // Invariant: `globalCentroid` represents the average centroid.
        for (int j = 0; j < globalCentroid.length; j++) {
            globalCentroid[j] /= centroids.length;
        }

        // write centroids
        writeCentroids(centroids, fieldInfo, globalCentroid, centroidOutput);

        // Block Logic: Log performance and count statistics for centroid calculation and vector assignment.
        // Precondition: `nanoTime` marks the start of the operation, `centroids` is populated.
        // Invariant: Execution time and final centroid count are logged if debug logging is enabled.
        if (logger.isDebugEnabled()) {
            logger.debug("calculate centroids and assign vectors time ms: {}", (System.nanoTime() - nanoTime) / 1000000.0);
            logger.debug("final centroid count: {}", centroids.length);
        }

        // Functional Utility: Initialize an array to count vectors per centroid for efficient memory allocation.
        int[] centroidVectorCount = new int[centroids.length];
        // Block Logic: Populate `centroidVectorCount` by tallying assignments to each centroid.
        // Precondition: `assignments` and `soarAssignments` arrays are populated.
        // Invariant: `centroidVectorCount[c]` accurately reflects the total number of vectors assigned to centroid `c`.
        for (int i = 0; i < assignments.length; i++) {
            centroidVectorCount[assignments[i]]++;
            // if soar assignments are present, count them as well
            if (soarAssignments.length > i && soarAssignments[i] != -1) {
                centroidVectorCount[soarAssignments[i]]++;
            }
        }

        // Functional Utility: Initialize a 2D array to store actual assignments by cluster.
        int[][] assignmentsByCluster = new int[centroids.length][];
        // Block Logic: Allocate inner arrays for `assignmentsByCluster` based on pre-calculated counts.
        // Precondition: `centroidVectorCount` contains accurate counts for each cluster.
        // Invariant: Each inner array `assignmentsByCluster[c]` is sized to hold all vectors assigned to centroid `c`.
        for (int c = 0; c < centroids.length; c++) {
            assignmentsByCluster[c] = new int[centroidVectorCount[c]];
        }
        // Functional Utility: Reset `centroidVectorCount` to zero for use as index pointers in the next loop.
        Arrays.fill(centroidVectorCount, 0);

        // Block Logic: Populate `assignmentsByCluster` with actual vector ordinals.
        // Precondition: `assignmentsByCluster` is initialized with correctly sized inner arrays.
        // Invariant: Each vector ordinal is placed into the appropriate cluster's array.
        for (int i = 0; i < assignments.length; i++) {
            int c = assignments[i];
            assignmentsByCluster[c][centroidVectorCount[c]++] = i;
            // if soar assignments are present, add them to the cluster as well
            if (soarAssignments.length > i) {
                int s = soarAssignments[i];
                if (s != -1) {
                    assignmentsByCluster[s][centroidVectorCount[s]++] = i;
                }
            }
        }

        // Block Logic: Return a CentroidAssignments object, optionally caching the centroids themselves.
        // Precondition: `centroids` and `assignmentsByCluster` are fully populated.
        // Invariant: A `CentroidAssignments` object is returned, facilitating further processing or inspection.
        if (cacheCentroids) {
            return new CentroidAssignments(centroids, assignmentsByCluster);
        } else {
            return new CentroidAssignments(centroids.length, assignmentsByCluster);
        }
    }

    /**
     * @brief Writes a quantized vector value and its associated correction factors to the output stream.
     * This method is used to persist the compressed representation of a vector (typically a centroid)
     * along with metadata required for accurate reconstruction or scoring.
     * @param indexOutput The IndexOutput to write the data to.
     * @param binaryValue The byte array representing the quantized vector.
     * @param corrections The `QuantizationResult` containing correction factors (lower, upper intervals, additional correction, and quantized component sum).
     * @throws IOException If an I/O error occurs during writing.
     */
    static void writeQuantizedValue(IndexOutput indexOutput, byte[] binaryValue, OptimizedScalarQuantizer.QuantizationResult corrections)
        throws IOException {
        indexOutput.writeBytes(binaryValue, binaryValue.length);
        indexOutput.writeInt(Float.floatToIntBits(corrections.lowerInterval()));
        indexOutput.writeInt(Float.floatToIntBits(corrections.upperInterval()));
        indexOutput.writeInt(Float.floatToIntBits(corrections.additionalCorrection()));
        // Inline: Assert that the quantized component sum is within the valid range for a short (0 to 0xffff).
        assert corrections.quantizedComponentSum() >= 0 && corrections.quantizedComponentSum() <= 0xffff;
        indexOutput.writeShort((short) corrections.quantizedComponentSum());
    }

    /**
     * @brief Implements `CentroidSupplier` for centroids stored off-heap in an `IndexInput`.
     * This class provides efficient access to centroids without loading all of them into
     * memory, which is crucial for handling large numbers of centroids. It reads centroid
     * data on demand from the provided `IndexInput`.
     */
    static class OffHeapCentroidSupplier implements CentroidSupplier {
        private final IndexInput centroidsInput;
        private final int numCentroids;
        private final int dimension;
        private final float[] scratch;
        private final long rawCentroidOffset;
        private int currOrd = -1;

        /**
         * @brief Constructs an `OffHeapCentroidSupplier`.
         * Initializes the supplier with the necessary input stream, centroid count,
         * and field information to allow efficient, on-demand reading of centroids.
         * @param centroidsInput The `IndexInput` containing the centroid data.
         * @param numCentroids The total number of centroids available.
         * @param info The `FieldInfo` object, used to retrieve the vector dimension.
         */
        OffHeapCentroidSupplier(IndexInput centroidsInput, int numCentroids, FieldInfo info) {
            this.centroidsInput = centroidsInput;
            this.numCentroids = numCentroids;
            this.dimension = info.getVectorDimension();
            this.scratch = new float[dimension];
            this.rawCentroidOffset = (dimension + 3 * Float.BYTES + Short.BYTES) * numCentroids;
        }

        /**
         * @brief Returns the total number of centroids managed by this supplier.
         * @return The count of centroids.
         */
        @Override
        public int size() {
            return numCentroids;
        }

        /**
         * @brief Retrieves the centroid vector for a given ordinal.
         * This method efficiently reads the centroid data from the underlying `IndexInput`
         * on demand, avoiding the need to load all centroids into memory simultaneously.
         * @param centroidOrdinal The ordinal index of the desired centroid.
         * @return A float array representing the centroid vector.
         * @throws IOException If an I/O error occurs during reading.
         */
        @Override
        public float[] centroid(int centroidOrdinal) throws IOException {
            // Block Logic: Check if the requested centroid is already in the scratch buffer.
            // Precondition: `centroidOrdinal` is a valid index.
            // Invariant: If `centroidOrdinal` matches `currOrd`, return the cached `scratch` array; otherwise, read from disk.
            if (centroidOrdinal == currOrd) {
                return scratch;
            }
            // Block Logic: Seek to the correct position in the `centroidsInput` and read the centroid data.
            // Precondition: `centroidsInput` is open and seekable, `rawCentroidOffset` and `dimension` are correct.
            // Invariant: The `scratch` array is populated with the float values of the requested centroid.
            centroidsInput.seek(rawCentroidOffset + (long) centroidOrdinal * dimension * Float.BYTES);
            centroidsInput.readFloats(scratch, 0, dimension);
            this.currOrd = centroidOrdinal;
            return scratch;
        }
    }
}
