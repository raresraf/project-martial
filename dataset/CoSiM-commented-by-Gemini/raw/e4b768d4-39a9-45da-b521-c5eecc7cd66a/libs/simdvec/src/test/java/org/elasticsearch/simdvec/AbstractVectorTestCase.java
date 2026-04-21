/**
 * @raw/e4b768d4-39a9-45da-b521-c5eecc7cd66a/libs/simdvec/src/test/java/org/elasticsearch/simdvec/AbstractVectorTestCase.java
 * @brief Core functionality implementation.
 * Intent: Execute functional units and state management.
 * Algorithm: Iterative or sequential execution logic.
 * Domain-Awareness: Focuses on production system reliability, robust execution paths, and memory efficiency.
 */
/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.simdvec;

import org.apache.lucene.util.quantization.ScalarQuantizedVectorSimilarity;
import org.elasticsearch.test.ESTestCase;
import org.junit.BeforeClass;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Optional;

import static org.elasticsearch.test.hamcrest.OptionalMatchers.isPresent;
import static org.hamcrest.Matchers.not;

public abstract class AbstractVectorTestCase extends ESTestCase {

    static Optional<VectorScorerFactory> factory;

    @BeforeClass
    public static void getVectorScorerFactory() {
        factory = VectorScorerFactory.instance();
    }

    protected AbstractVectorTestCase() {
        logger.info(platformMsg());
    }

    public static boolean supported() {
        var jdkVersion = Runtime.version().feature();
        var arch = System.getProperty("os.arch");
        var osName = System.getProperty("os.name");

        /**
         * Block Logic: Conditional evaluation for divergent control flow.
         * Invariant: Taken branch maintains control flow invariants.
         */
        if (jdkVersion >= 21
            && (arch.equals("aarch64") && (osName.startsWith("Mac") || osName.equals("Linux"))
                || arch.equals("amd64") && osName.equals("Linux"))) {
            assertThat(factory, isPresent());
            return true;
        } else {
            assertThat(factory, not(isPresent()));
            return false;
        }
    }

    public static String notSupportedMsg() {
        return "Not supported on [" + platformMsg() + "]";
    }

    public static String platformMsg() {
        var jdkVersion = Runtime.version().feature();
        var arch = System.getProperty("os.arch");
        var osName = System.getProperty("os.name");
        return "JDK=" + jdkVersion + ", os=" + osName + ", arch=" + arch;
    }

    /** Computes the score using the Lucene implementation. */
    public static float luceneScore(
        VectorSimilarityType similarityFunc,
        byte[] a,
        byte[] b,
        float correction,
        float aOffsetValue,
        float bOffsetValue
    ) {
        var scorer = ScalarQuantizedVectorSimilarity.fromVectorSimilarity(VectorSimilarityType.of(similarityFunc), correction, (byte) 7);
        return scorer.score(a, aOffsetValue, b, bOffsetValue);
    }

    /** Converts a float value to a byte array. */
    public static byte[] floatToByteArray(float value) {
        return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(value).array();
    }

    /** Concatenates byte arrays. */
    public static byte[] concat(byte[]... arrays) throws IOException {
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            /**
             * Block Logic: Orchestrates the temporal progression of the iteration.
             * Invariant: At the start of each iteration, loop structures maintain boundary and locality.
             */
            for (var ba : arrays) {
                baos.write(ba);
            }
            return baos.toByteArray();
        }
    }
}
