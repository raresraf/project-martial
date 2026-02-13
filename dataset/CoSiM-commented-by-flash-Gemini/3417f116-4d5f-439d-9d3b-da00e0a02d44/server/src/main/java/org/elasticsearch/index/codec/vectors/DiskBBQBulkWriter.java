/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

/**
 * @file DiskBBQBulkWriter.java
 * @brief This file defines a framework for bulk writing of quantized vectors to disk,
 * primarily using the Product Quantization (PQ) based "BBQ" encoding scheme.
 * It provides abstract mechanisms for writing vectors in optimized bulk operations,
 * supporting various quantization bit configurations.
 * Algorithm: Product Quantization (BBQ encoding) for vector compression and storage.
 * Time Complexity: Writing operations are typically O(N * D) where N is number of vectors and D is vector dimension,
 * but optimized for bulk.
 * Space Complexity: O(B * D) for internal buffers where B is bulk size, plus storage for quantized vectors.
 */

package org.elasticsearch.index.codec.vectors;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.hnsw.IntToIntFunction;

import java.io.IOException;

import static org.elasticsearch.index.codec.vectors.BQVectorUtils.discretize;
import static org.elasticsearch.index.codec.vectors.BQVectorUtils.packAsBinary;

/**
 * @brief Base class for bulk writers that persist quantized vectors to disk using the BBQ encoding.
 * This abstract class provides the fundamental structure for writing vectors in bulk,
 * with specialized concrete implementations handling different bit quantization strategies.
 * It manages common resources such as bulk size, scalar quantizer, and the output stream.
 */
public abstract class DiskBBQBulkWriter {
    protected final int bulkSize;
    protected final OptimizedScalarQuantizer quantizer;
    protected final IndexOutput out;
    protected final FloatVectorValues fvv;

    /**
     * @brief Constructs a new `DiskBBQBulkWriter`.
     * @param bulkSize The number of vectors to process in a single bulk operation.
     * @param quantizer The `OptimizedScalarQuantizer` used to quantize the vectors.
     * @param fvv The `FloatVectorValues` providing the original vector data.
     * @param out The `IndexOutput` stream where the quantized vector data will be written.
     */
    protected DiskBBQBulkWriter(int bulkSize, OptimizedScalarQuantizer quantizer, FloatVectorValues fvv, IndexOutput out) {
        this.bulkSize = bulkSize;
        this.quantizer = quantizer;
        this.out = out;
        this.fvv = fvv;
    }

    /**
     * @brief Abstract method to write a batch of quantized vectors and their associated data.
     * Concrete implementations will define how the vectors are quantized and written
     * based on their specific bit-size strategy.
     * @param ords A function mapping an index to a document ordinal, providing access to the original vector data.
     * @param count The number of vectors to write in this batch.
     * @param centroid The centroid associated with the vectors being written, used for quantization.
     * @throws IOException If an I/O error occurs during writing.
     */
    public abstract void writeOrds(IntToIntFunction ords, int count, float[] centroid) throws IOException;

    /**
     * @brief Writes an array of `QuantizationResult` corrections to the output stream.
     * This method serializes multiple correction factors for a batch of quantized vectors
     * into the `IndexOutput`, enabling proper reconstruction or scoring later.
     * It writes lower intervals, upper intervals, quantized component sums, and additional
     * corrections in distinct blocks.
     * @param corrections An array of `OptimizedScalarQuantizer.QuantizationResult` objects.
     * @param out The `IndexOutput` to write the corrections to.
     * @throws IOException If an I/O error occurs during writing.
     */
    private static void writeCorrections(OptimizedScalarQuantizer.QuantizationResult[] corrections, IndexOutput out) throws IOException {
        // Block Logic: Write all lower interval values for the batch.
        // Precondition: `corrections` array is not null.
        // Invariant: All lower interval floats from `corrections` are converted to int bits and written.
        for (OptimizedScalarQuantizer.QuantizationResult correction : corrections) {
            out.writeInt(Float.floatToIntBits(correction.lowerInterval()));
        }
        // Block Logic: Write all upper interval values for the batch.
        // Precondition: `corrections` array is not null.
        // Invariant: All upper interval floats from `corrections` are converted to int bits and written.
        for (OptimizedScalarQuantizer.QuantizationResult correction : corrections) {
            out.writeInt(Float.floatToIntBits(correction.upperInterval()));
        }
        // Block Logic: Write all quantized component sums for the batch.
        // Precondition: `corrections` array is not null.
        // Invariant: Each quantized component sum is asserted to be within `short` range and written.
        for (OptimizedScalarQuantizer.QuantizationResult correction : corrections) {
            int targetComponentSum = correction.quantizedComponentSum();
            // Inline: Assert that the quantized component sum is within the valid range for a short (0 to 0xffff).
            assert targetComponentSum >= 0 && targetComponentSum <= 0xffff;
            out.writeShort((short) targetComponentSum);
        }
        // Block Logic: Write all additional correction values for the batch.
        // Precondition: `corrections` array is not null.
        // Invariant: All additional correction floats from `corrections` are converted to int bits and written.
        for (OptimizedScalarQuantizer.QuantizationResult correction : corrections) {
            out.writeInt(Float.floatToIntBits(correction.additionalCorrection()));
        }
    }

    /**
     * @brief Writes a single `QuantizationResult` correction to the output stream.
     * This method serializes a single set of correction factors for a quantized vector
     * into the `IndexOutput`, enabling proper reconstruction or scoring later.
     * @param correction A single `OptimizedScalarQuantizer.QuantizationResult` object.
     * @param out The `IndexOutput` to write the correction to.
     * @throws IOException If an I/O error occurs during writing.
     */
    private static void writeCorrection(OptimizedScalarQuantizer.QuantizationResult correction, IndexOutput out) throws IOException {
        out.writeInt(Float.floatToIntBits(correction.lowerInterval()));
        out.writeInt(Float.floatToIntBits(correction.upperInterval()));
        int targetComponentSum = correction.quantizedComponentSum();
        // Inline: Assert that the quantized component sum is within the valid range for a short (0 to 0xffff).
        assert targetComponentSum >= 0 && targetComponentSum <= 0xffff;
        out.writeShort((short) targetComponentSum);
        out.writeInt(Float.floatToIntBits(correction.additionalCorrection()));
    }

    /**
     * @brief Implements `DiskBBQBulkWriter` for 1-bit scalar quantization.
     * This concrete implementation handles the bulk writing of vectors that have been
     * quantized to 1 bit per dimension, using a specific binarization and correction
     * mechanism.
     */
    public static class OneBitDiskBBQBulkWriter extends DiskBBQBulkWriter {
        private final byte[] binarized;
        private final byte[] initQuantized;
        private final OptimizedScalarQuantizer.QuantizationResult[] corrections;

        /**
         * @brief Constructs a new `OneBitDiskBBQBulkWriter`.
         * Initializes the writer with the necessary bulk processing parameters and allocates
         * internal buffers for binarized and quantized vector data, as well as correction results.
         * @param bulkSize The number of vectors to process in a single bulk operation.
         * @param quantizer The `OptimizedScalarQuantizer` configured for 1-bit quantization.
         * @param fvv The `FloatVectorValues` providing the original vector data.
         * @param out The `IndexOutput` stream where the quantized vector data will be written.
         */
        public OneBitDiskBBQBulkWriter(int bulkSize, OptimizedScalarQuantizer quantizer, FloatVectorValues fvv, IndexOutput out) {
            super(bulkSize, quantizer, fvv, out);
            this.binarized = new byte[discretize(fvv.dimension(), 64) / 8];
            this.initQuantized = new byte[fvv.dimension()];
            this.corrections = new OptimizedScalarQuantizer.QuantizationResult[bulkSize];
        }

        /**
         * @brief Writes a batch of vectors, quantized to 1 bit, to the output stream.
         * This method processes vectors in bulk, quantizes them to 1 bit per dimension
         * against a given centroid, binarizes the quantized result, and writes both
         * the binarized vectors and their correction factors to the `IndexOutput`.
         * @param ords A function to retrieve the Lucene ordinal for a given index.
         * @param count The total number of vectors to write.
         * @param centroid The centroid used for quantizing the vectors in this batch.
         * @throws IOException If an I/O error occurs during writing.
         */
        @Override
        public void writeOrds(IntToIntFunction ords, int count, float[] centroid) throws IOException {
            int limit = count - bulkSize + 1;
            int i = 0;
            // Block Logic: Process vectors in full bulkSize chunks.
            // Precondition: `count` and `bulkSize` are positive, `ords` provides valid ordinals.
            // Invariant: `bulkSize` vectors are quantized, binarized, and written in each iteration,
            // along with their collected correction factors.
            for (; i < limit; i += bulkSize) {
                // Block Logic: Quantize and binarize each vector within the current bulk chunk.
                // Precondition: `j` iterates from 0 to `bulkSize - 1`.
                // Invariant: `corrections[j]` stores the quantization result and `binarized` holds the 1-bit vector.
                for (int j = 0; j < bulkSize; j++) {
                    int ord = ords.apply(i + j);
                    float[] fv = fvv.vectorValue(ord);
                    corrections[j] = quantizer.scalarQuantize(fv, initQuantized, (byte) 1, centroid);
                    packAsBinary(initQuantized, binarized);
                    out.writeBytes(binarized, binarized.length);
                }
                writeCorrections(corrections, out);
            }
            // Block Logic: Write any remaining tail vectors that do not form a full bulkSize chunk.
            // Precondition: `i` is the starting index for remaining vectors.
            // Invariant: Each remaining vector is quantized, binarized, and written individually with its correction.
            for (; i < count; ++i) {
                int ord = ords.apply(i);
                float[] fv = fvv.vectorValue(ord);
                OptimizedScalarQuantizer.QuantizationResult correction = quantizer.scalarQuantize(fv, initQuantized, (byte) 1, centroid);
                packAsBinary(initQuantized, binarized);
                out.writeBytes(binarized, binarized.length);
                writeCorrection(correction, out);
            }
        }
    }
}
