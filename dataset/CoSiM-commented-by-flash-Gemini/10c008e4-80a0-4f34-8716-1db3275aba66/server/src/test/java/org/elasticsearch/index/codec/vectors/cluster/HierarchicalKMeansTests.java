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
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @brief Functional description of the HierarchicalKMeansTests class.
 *        This is a placeholder for detailed semantic documentation.
 *        Further analysis will elaborate on its algorithm, complexity, and invariants.
 */
public class HierarchicalKMeansTests extends ESTestCase {

    /**
     * @brief [Functional Utility for testHKmeans]: Describe purpose here.
     * @return [ReturnType]: [Description]
     * @throws IOException: [Description]
     */
    public void testHKmeans() throws IOException {
         /**
          * @brief [Functional description for field nClusters]: Describe purpose here.
          */
        int nClusters = random().nextInt(1, 10);
         /**
          * @brief [Functional description for field nVectors]: Describe purpose here.
          */
        int nVectors = random().nextInt(1, nClusters * 200);
         /**
          * @brief [Functional description for field dims]: Describe purpose here.
          */
        int dims = random().nextInt(2, 20);
         /**
          * @brief [Functional description for field sampleSize]: Describe purpose here.
          */
        int sampleSize = random().nextInt(Math.min(nVectors, 100), nVectors + 1);
         /**
          * @brief [Functional description for field maxIterations]: Describe purpose here.
          */
        int maxIterations = random().nextInt(1, 100);
         /**
          * @brief [Functional description for field clustersPerNeighborhood]: Describe purpose here.
          */
        int clustersPerNeighborhood = random().nextInt(2, 512);
         /**
          * @brief [Functional description for field soarLambda]: Describe purpose here.
          */
        float soarLambda = random().nextFloat(0.5f, 1.5f);
         /**
          * @brief [Functional description for field vectors]: Describe purpose here.
          */
        FloatVectorValues vectors = generateData(nVectors, dims, nClusters);

         /**
          * @brief [Functional description for field targetSize]: Describe purpose here.
          */
        int targetSize = (int) ((float) nVectors / (float) nClusters);
         /**
          * @brief [Functional description for field hkmeans]: Describe purpose here.
          */
        HierarchicalKMeans hkmeans = new HierarchicalKMeans(dims, maxIterations, sampleSize, clustersPerNeighborhood, soarLambda);

         /**
          * @brief [Functional description for field result]: Describe purpose here.
          */
        KMeansResult result = hkmeans.cluster(vectors, targetSize);

         /**
          * @brief [Functional description for field centroids]: Describe purpose here.
          */
        float[][] centroids = result.centroids();
         /**
          * @brief [Functional description for field assignments]: Describe purpose here.
          */
        int[] assignments = result.assignments();
         /**
          * @brief [Functional description for field soarAssignments]: Describe purpose here.
          */
        int[] soarAssignments = result.soarAssignments();

        assertEquals(Math.min(nClusters, nVectors), centroids.length, 8);
        assertEquals(nVectors, assignments.length);
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        if (centroids.length > 1 && centroids.length < nVectors) {
            assertEquals(nVectors, soarAssignments.length);
            // verify no duplicates exist
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            for (int i = 0; i < assignments.length; i++) {
                assert assignments[i] != soarAssignments[i];
            }
        // Block Logic: [Describe purpose of this else/else if block]
        } else {
            assertEquals(0, soarAssignments.length);
        }
    }

    /**
     * @brief [Functional Utility: Describe purpose here]
     * @param nSamples: [Description]
     * @param nDims: [Description]
     * @param nClusters: [Description]
     * @return [ReturnType]: [Description]
     * @throws [ExceptionType]: [Description]
     */
    private static FloatVectorValues generateData(int nSamples, int nDims, int nClusters) {
        List<float[]> vectors = new ArrayList<>(nSamples);
    /**
     * @brief [Functional description for field centroids]: Describe purpose here.
     */
        float[][] centroids = new float[nClusters][nDims];
        // Generate random centroids
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        for (int i = 0; i < nClusters; i++) {
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            for (int j = 0; j < nDims; j++) {
                centroids[i][j] = random().nextFloat() * 100;
            }
        }
        // Generate data points around centroids
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
         // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n         // Invariant: [State condition that holds true before and after each iteration/execution]\n        for (int i = 0; i < nSamples; i++) {
            int cluster = random().nextInt(nClusters);
    /**
     * @brief [Functional description for field vector]: Describe purpose here.
     */
            float[] vector = new float[nDims];
        // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]
        // Invariant: [State condition that holds true before and after each iteration/execution]
             // Block Logic: [Describe purpose of this block, e.g., iteration, conditional execution]\n             // Invariant: [State condition that holds true before and after each iteration/execution]\n            for (int j = 0; j < nDims; j++) {
                vector[j] = centroids[cluster][j] + random().nextFloat() * 10 - 5;
            }
            vectors.add(vector);
        }
        return FloatVectorValues.fromFloats(vectors, nDims);
    }
}
