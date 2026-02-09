/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.cluster;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.IntToIntFunction;
import org.elasticsearch.index.codec.vectors.SampleReader;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * k-means implementation specific to the needs of the {@link HierarchicalKMeans} algorithm that deals specifically
 * with finalizing nearby pre-established clusters and generate
 * <a href="https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/">SOAR</a> assignments
 */
class KMeansLocal {

    // the minimum distance that is considered to be "far enough" to a centroid in order to compute the soar distance.
    // For vectors that are closer than this distance to the centroid, we use the squared distance to find the
    // second closest centroid.
    private static final float SOAR_MIN_DISTANCE = 1e-16f;

    /**
     * @brief [Functional description for field sampleSize]: Describe purpose here.
     */
    final int sampleSize;
    /**
     * @brief [Functional description for field maxIterations]: Describe purpose here.
     */
    final int maxIterations;

    KMeansLocal(int sampleSize, int maxIterations) {
        this.sampleSize = sampleSize;
        this.maxIterations = maxIterations;
    }

    /**
     * uses a Reservoir Sampling approach to picking the initial centroids which are subsequently expected
     * to be used by a clustering algorithm
     *
     * @param vectors used to pick an initial set of random centroids
     * @param centroidCount the total number of centroids to pick
     * @return randomly selected centroids that are the min of centroidCount and sampleSize
     * @throws IOException is thrown if vectors is inaccessible
     */
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param vectors: [Description]
     * @param centroidCount: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    static float[][] pickInitialCentroids(FloatVectorValues vectors, int centroidCount) throws IOException {
        Random random = new Random(42L);
        int centroidsSize = Math.min(vectors.size(), centroidCount);
        float[][] centroids = new float[centroidsSize][vectors.dimension()];
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        for (int i = 0; i < vectors.size(); i++) {
    /**
     * @brief [Functional description for field vector]: Describe purpose here.
     */
            float[] vector;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (i < centroidCount) {
                vector = vectors.vectorValue(i);
                System.arraycopy(vector, 0, centroids[i], 0, vector.length);
        // Block Logic: [Describe purpose of this else/else if block]
            } else if (random.nextDouble() < centroidCount * (1.0 / i)) {
                int c = random.nextInt(centroidCount);
                vector = vectors.vectorValue(i);
                System.arraycopy(vector, 0, centroids[c], 0, vector.length);
            }
        }
    /**
     * @brief [Functional description for field centroids]: Describe purpose here.
     */
        return centroids;
    }

    /**
     * @brief [Functional Utility for stepLloyd]: Describe purpose here.
     * @return boolean: [Description]\n     * @throws IOException: [Description]\n     */
    /**
     * @brief [Functional Utility for stepLloyd]: Describe purpose here.
     * @return boolean: [Description]\n     * @throws IOException: [Description]\n     */
    private static boolean stepLloyd(
        FloatVectorValues vectors,
        IntToIntFunction translateOrd,
        float[][] centroids,
        float[][] nextCentroids,
        int[] assignments,
        List<NeighborHood> neighborhoods
    ) throws IOException {
    /**
     * @brief [Functional description for field changed]: Describe purpose here.
     */
        boolean changed = false;
        int dim = vectors.dimension();
    /**
     * @brief [Functional description for field centroidCounts]: Describe purpose here.
     */
        int[] centroidCounts = new int[centroids.length];

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        for (float[] nextCentroid : nextCentroids) {
            Arrays.fill(nextCentroid, 0.0f);
        }

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        for (int idx = 0; idx < vectors.size(); idx++) {
            float[] vector = vectors.vectorValue(idx);
            int vectorOrd = translateOrd.apply(idx);
    /**
     * @brief [Functional description for field assignment]: Describe purpose here.
     */
            final int assignment = assignments[vectorOrd];
    /**
     * @brief [Functional description for field bestCentroidOffset]: Describe purpose here.
     */
            final int bestCentroidOffset;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (neighborhoods != null) {
                bestCentroidOffset = getBestCentroidFromNeighbours(centroids, vector, assignment, neighborhoods.get(assignment));
        // Block Logic: [Describe purpose of this else/else if block]
            } else {
                bestCentroidOffset = getBestCentroid(centroids, vector);
            }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (assignment != bestCentroidOffset) {
                assignments[vectorOrd] = bestCentroidOffset;
                changed = true;
            }
            centroidCounts[bestCentroidOffset]++;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            for (int d = 0; d < dim; d++) {
                nextCentroids[bestCentroidOffset][d] += vector[d];
            }
        }

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        for (int clusterIdx = 0; clusterIdx < centroids.length; clusterIdx++) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (centroidCounts[clusterIdx] > 0) {
                float countF = (float) centroidCounts[clusterIdx];
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                for (int d = 0; d < dim; d++) {
                    centroids[clusterIdx][d] = nextCentroids[clusterIdx][d] / countF;
                }
            }
        }

    /**
     * @brief [Functional description for field changed]: Describe purpose here.
     */
        return changed;
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param centroids: [Description]
     * @param vector: [Description]
     * @param centroidIdx: [Description]
     * @param neighborhood: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    private static int getBestCentroidFromNeighbours(float[][] centroids, float[] vector, int centroidIdx, NeighborHood neighborhood) {
    /**
     * @brief [Functional description for field bestCentroidOffset]: Describe purpose here.
     */
        int bestCentroidOffset = centroidIdx;
        assert centroidIdx >= 0 && centroidIdx < centroids.length;
        float minDsq = VectorUtil.squareDistance(vector, centroids[centroidIdx]);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        for (int i = 0; i < neighborhood.neighbors.length; i++) {
    /**
     * @brief [Functional description for field offset]: Describe purpose here.
     */
            int offset = neighborhood.neighbors[i];
            // float score = neighborhood.scores[i];
            assert offset >= 0 && offset < centroids.length : "Invalid neighbor offset: " + offset;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (minDsq < neighborhood.maxIntraDistance) {
                // if the distance found is smaller than the maximum intra-cluster distance
                // we don't consider it for further re-assignment
    /**
     * @brief [Functional description for field bestCentroidOffset]: Describe purpose here.
     */
                return bestCentroidOffset;
            }
            // compute the distance to the centroid
            float dsq = VectorUtil.squareDistance(vector, centroids[offset]);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (dsq < minDsq) {
                minDsq = dsq;
                bestCentroidOffset = offset;
            }
        }
    /**
     * @brief [Functional description for field bestCentroidOffset]: Describe purpose here.
     */
        return bestCentroidOffset;
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param centroids: [Description]
     * @param vector: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    private static int getBestCentroid(float[][] centroids, float[] vector) {
    /**
     * @brief [Functional description for field bestCentroidOffset]: Describe purpose here.
     */
        int bestCentroidOffset = 0;
    /**
     * @brief [Functional description for field minDsq]: Describe purpose here.
     */
        float minDsq = Float.MAX_VALUE;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        for (int i = 0; i < centroids.length; i++) {
            float dsq = VectorUtil.squareDistance(vector, centroids[i]);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (dsq < minDsq) {
                minDsq = dsq;
                bestCentroidOffset = i;
            }
        }
    /**
     * @brief [Functional description for field bestCentroidOffset]: Describe purpose here.
     */
        return bestCentroidOffset;
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param centers: [Description]
     * @param neighborhoods: [Description]
     * @param clustersPerNeighborhood: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    private void computeNeighborhoods(float[][] centers, List<NeighborHood> neighborhoods, int clustersPerNeighborhood) {
        int k = neighborhoods.size();

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (k == 0 || clustersPerNeighborhood <= 0) {
            return;
        }

        List<NeighborQueue> neighborQueues = new ArrayList<>(k);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        for (int i = 0; i < k; i++) {
            neighborQueues.add(new NeighborQueue(clustersPerNeighborhood, true));
        }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        for (int i = 0; i < k - 1; i++) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            for (int j = i + 1; j < k; j++) {
                float dsq = VectorUtil.squareDistance(centers[i], centers[j]);
                neighborQueues.get(j).insertWithOverflow(i, dsq);
                neighborQueues.get(i).insertWithOverflow(j, dsq);
            }
        }

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        for (int i = 0; i < k; i++) {
            NeighborQueue queue = neighborQueues.get(i);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (queue.size() == 0) {
                // no neighbors, skip
                neighborhoods.set(i, NeighborHood.EMPTY);
                continue;
            }
            // consume the queue into the neighbors array and get the maximum intra-cluster distance
            int[] neighbors = new int[queue.size()];
            float maxIntraDistance = queue.topScore();
    /**
     * @brief [Functional description for field iter]: Describe purpose here.
     */
            int iter = 0;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            while (queue.size() > 0) {
                neighbors[neighbors.length - ++iter] = queue.pop();
            }
            NeighborHood neighborHood = new NeighborHood(neighbors, maxIntraDistance);
            neighborhoods.set(i, neighborHood);
        }
    }

    /**
     * @brief [Functional Utility for assignSpilled]: Describe purpose here.
     * @return [ReturnType]: [Description]\n     * @throws IOException: [Description]\n     */
    /**
     * @brief [Functional Utility for assignSpilled]: Describe purpose here.
     * @return [ReturnType]: [Description]\n     * @throws IOException: [Description]\n     */
    private int[] assignSpilled(
        FloatVectorValues vectors,
        List<NeighborHood> neighborhoods,
        float[][] centroids,
        int[] assignments,
        float soarLambda
    ) throws IOException {
        // SOAR uses an adjusted distance for assigning spilled documents which is
        // given by:
        //
        // soar(x, c) = ||x - c||^2 + lambda * ((x - c_1)^t (x - c))^2 / ||x - c_1||^2
        //
        // Here, x is the document, c is the nearest centroid, and c_1 is the first
        // centroid the document was assigned to. The document is assigned to the
        // cluster with the smallest soar(x, c).

    /**
     * @brief [Functional description for field spilledAssignments]: Describe purpose here.
     */
        int[] spilledAssignments = new int[assignments.length];

        float[] diffs = new float[vectors.dimension()];
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        for (int i = 0; i < vectors.size(); i++) {
            float[] vector = vectors.vectorValue(i);

    /**
     * @brief [Functional description for field currAssignment]: Describe purpose here.
     */
            int currAssignment = assignments[i];
    /**
     * @brief [Functional description for field currentCentroid]: Describe purpose here.
     */
            float[] currentCentroid = centroids[currAssignment];

            // TODO: cache these?
            float vectorCentroidDist = VectorUtil.squareDistance(vector, currentCentroid);

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (vectorCentroidDist > SOAR_MIN_DISTANCE) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                for (int j = 0; j < vectors.dimension(); j++) {
    /**
     * @brief [Functional description for field diff]: Describe purpose here.
     */
                    float diff = vector[j] - currentCentroid[j];
                    diffs[j] = diff;
                }
            }

    /**
     * @brief [Functional description for field bestAssignment]: Describe purpose here.
     */
            int bestAssignment = -1;
    /**
     * @brief [Functional description for field minSoar]: Describe purpose here.
     */
            float minSoar = Float.MAX_VALUE;
    /**
     * @brief [Functional description for field centroidCount]: Describe purpose here.
     */
            int centroidCount = centroids.length;
    /**
     * @brief [Functional description for field centroidOrds]: Describe purpose here.
     */
            IntToIntFunction centroidOrds = c -> c;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (neighborhoods != null) {
                assert neighborhoods.get(currAssignment) != null;
                NeighborHood neighborhood = neighborhoods.get(currAssignment);
                centroidCount = neighborhood.neighbors.length;
                centroidOrds = c -> neighborhood.neighbors[c];
            }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            for (int j = 0; j < centroidCount; j++) {
                int centroidOrd = centroidOrds.apply(j);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (centroidOrd == currAssignment) {
                    continue; // skip the current assignment
                }
    /**
     * @brief [Functional description for field centroid]: Describe purpose here.
     */
                float[] centroid = centroids[centroidOrd];
    /**
     * @brief [Functional description for field soar]: Describe purpose here.
     */
                float soar;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (vectorCentroidDist > SOAR_MIN_DISTANCE) {
                    soar = ESVectorUtil.soarDistance(vector, centroid, diffs, soarLambda, vectorCentroidDist);
        // Block Logic: [Describe purpose of this else/else if block]
                } else {
                    // if the vector is very close to the centroid, we look for the second-nearest centroid
                    soar = VectorUtil.squareDistance(vector, centroid);
                }
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
                 // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n                 // Invariant: [State condition that holds true before and after each iteration/execution]\n                if (soar < minSoar) {
                    minSoar = soar;
                    bestAssignment = centroidOrd;
                }
            }

            assert bestAssignment != -1 : "Failed to assign soar vector to centroid";
            spilledAssignments[i] = bestAssignment;
        }

    /**
     * @brief [Functional description for field spilledAssignments]: Describe purpose here.
     */
        return spilledAssignments;
    }

    /**
     * @brief [Functional Utility for NeighborHood]: Describe purpose here.
     * @param neighbors: [Description]
     * @param maxIntraDistance: [Description]
     * @return [ReturnType]: [Description]
     */
    record NeighborHood(int[] neighbors, float maxIntraDistance) {
        static final NeighborHood EMPTY = new NeighborHood(new int[0], Float.POSITIVE_INFINITY);
    }

    /**
     * cluster using a lloyd k-means algorithm that is not neighbor aware
     *
     * @param vectors the vectors to cluster
     * @param kMeansIntermediate the output object to populate which minimally includes centroids,
     *                     but may include assignments and soar assignments as well; care should be taken in
     *                     passing in a valid output object with a centroids array that is the size of centroids expected
     * @throws IOException is thrown if vectors is inaccessible
     */
    void cluster(FloatVectorValues vectors, KMeansIntermediate kMeansIntermediate) throws IOException {
        doCluster(vectors, kMeansIntermediate, -1, -1);
    }

    /**
     * cluster using a lloyd kmeans algorithm that also considers prior clustered neighborhoods when adjusting centroids
     * this also is used to generate the neighborhood aware additional (SOAR) assignments
     *
     * @param vectors the vectors to cluster
     * @param kMeansIntermediate the output object to populate which minimally includes centroids,
     *                     the prior assignments of the given vectors; care should be taken in
     *                     passing in a valid output object with a centroids array that is the size of centroids expected
     *                     and assignments that are the same size as the vectors.  The SOAR assignments are overwritten by this operation.
     * @param clustersPerNeighborhood number of nearby neighboring centroids to be used to update the centroid positions.
     * @param soarLambda   lambda used for SOAR assignments
     *
     * @throws IOException is thrown if vectors is inaccessible or if the clustersPerNeighborhood is less than 2
     */
    void cluster(FloatVectorValues vectors, KMeansIntermediate kMeansIntermediate, int clustersPerNeighborhood, float soarLambda)
        throws IOException {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (clustersPerNeighborhood < 2) {
            throw new IllegalArgumentException("clustersPerNeighborhood must be at least 2, got [" + clustersPerNeighborhood + "]");
        }
        doCluster(vectors, kMeansIntermediate, clustersPerNeighborhood, soarLambda);
    }

    /**
     * @brief [Functional Utility for doCluster]: Describe purpose here.
     * @param vectors: [Description]
     * @param kMeansIntermediate: [Description]
     * @param clustersPerNeighborhood: [Description]
     * @param soarLambda: [Description]
     * @return [ReturnType]: [Description]
     */
    private void doCluster(FloatVectorValues vectors, KMeansIntermediate kMeansIntermediate, int clustersPerNeighborhood, float soarLambda)
        throws IOException {
        float[][] centroids = kMeansIntermediate.centroids();
    /**
     * @brief [Functional description for field neighborAware]: Describe purpose here.
     */
        boolean neighborAware = clustersPerNeighborhood != -1 && centroids.length > 1;

    /**
     * @brief [Functional description for field neighborhoods]: Describe purpose here.
     */
        List<NeighborHood> neighborhoods = null;
        // if there are very few centroids, don't bother with neighborhoods or neighbor aware clustering
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (neighborAware && centroids.length > clustersPerNeighborhood) {
    /**
     * @brief [Functional description for field k]: Describe purpose here.
     */
            int k = centroids.length;
            neighborhoods = new ArrayList<>(k);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            for (int i = 0; i < k; ++i) {
                neighborhoods.add(null);
            }
            computeNeighborhoods(centroids, neighborhoods, clustersPerNeighborhood);
        }
        cluster(vectors, kMeansIntermediate, neighborhoods);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (neighborAware) {
            int[] assignments = kMeansIntermediate.assignments();
            assert assignments != null;
            assert assignments.length == vectors.size();
            kMeansIntermediate.setSoarAssignments(assignSpilled(vectors, neighborhoods, centroids, assignments, soarLambda));
        }
    }

    /**
     * @brief [Functional Utility for cluster]: Describe purpose here.
     * @param vectors: [Description]
     * @param kMeansIntermediate: [Description]
     * @param neighborhoods: [Description]
     * @return [ReturnType]: [Description]
     */
    private void cluster(FloatVectorValues vectors, KMeansIntermediate kMeansIntermediate, List<NeighborHood> neighborhoods)
        throws IOException {
        float[][] centroids = kMeansIntermediate.centroids();
    /**
     * @brief [Functional description for field k]: Describe purpose here.
     */
        int k = centroids.length;
        int n = vectors.size();
        int[] assignments = kMeansIntermediate.assignments();

        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (k == 1) {
            Arrays.fill(assignments, 0);
            return;
        }
    /**
     * @brief [Functional description for field translateOrd]: Describe purpose here.
     */
        IntToIntFunction translateOrd = i -> i;
    /**
     * @brief [Functional description for field sampledVectors]: Describe purpose here.
     */
        FloatVectorValues sampledVectors = vectors;
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (sampleSize < n) {
            sampledVectors = SampleReader.createSampleReader(vectors, sampleSize, 42L);
            translateOrd = sampledVectors::ordToDoc;
        }

        assert assignments.length == n;
        float[][] nextCentroids = new float[centroids.length][vectors.dimension()];
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        for (int i = 0; i < maxIterations; i++) {
            // This is potentially sampled, so we need to translate ordinals
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            if (stepLloyd(sampledVectors, translateOrd, centroids, nextCentroids, assignments, neighborhoods) == false) {
                break;
            }
        }
        // If we were sampled, do a once over the full set of vectors to finalize the centroids
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (sampleSize < n || maxIterations == 0) {
            // No ordinal translation needed here, we are using the full set of vectors
            stepLloyd(vectors, i -> i, centroids, nextCentroids, assignments, neighborhoods);
        }
    }

    /**
     * helper that calls {@link KMeansLocal#cluster(FloatVectorValues, KMeansIntermediate)} given a set of initialized centroids,
     * this call is not neighbor aware
     *
     * @param vectors the vectors to cluster
     * @param centroids the initialized centroids to be shifted using k-means
     * @param sampleSize the subset of vectors to use when shifting centroids
     * @param maxIterations the max iterations to shift centroids
     */
    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param vectors: [Description]
     * @param centroids: [Description]
     * @param sampleSize: [Description]
     * @param maxIterations: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    public static void cluster(FloatVectorValues vectors, float[][] centroids, int sampleSize, int maxIterations) throws IOException {
        KMeansIntermediate kMeansIntermediate = new KMeansIntermediate(centroids, new int[vectors.size()], vectors::ordToDoc);
        KMeansLocal kMeans = new KMeansLocal(sampleSize, maxIterations);
        kMeans.cluster(vectors, kMeansIntermediate);
    }

}
