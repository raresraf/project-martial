/**
 * @a8105900-0e3b-4b8c-85b7-d103723bff5d/libs/native/src/test/java/org/elasticsearch/nativeaccess/VectorSimilarityFunctionsTests.java
 * @brief Platform compatibility verification for Elasticsearch's Native Access vector similarity kernels.
 * Domain: Software Testing, Native Interoperability, Cross-platform Support.
 * Architecture: Inherits from ESTestCase; utilizes the NativeAccess singleton to verify availability of SIMD-accelerated scoring functions.
 * Functional Utility: Validates that native vector kernels are correctly loaded on supported combinations of JDK versions (>=21), architectures (aarch64, amd64), and operating systems (macOS, Linux).
 */

package org.elasticsearch.nativeaccess;

import org.elasticsearch.test.ESTestCase;

import java.util.Optional;

import static org.elasticsearch.test.hamcrest.OptionalMatchers.isPresent;
import static org.hamcrest.Matchers.not;

/**
 * @brief Core test suite for runtime capability discovery.
 */
public class VectorSimilarityFunctionsTests extends ESTestCase {

    final Optional<VectorSimilarityFunctions> vectorSimilarityFunctions;

    public VectorSimilarityFunctionsTests() {
        // Observability: Emits platform diagnostics (JDK, OS, Arch) during suite initialization.
        logger.info(platformMsg());
        vectorSimilarityFunctions = NativeAccess.instance().getVectorSimilarityFunctions();
    }

    public void testSupported() {
        supported();
    }

    protected VectorSimilarityFunctions getVectorDistance() {
        return vectorSimilarityFunctions.get();
    }

    /**
     * @brief Evaluates whether the current environment satisfies the requirements for native SIMD access.
     * Logic: Enforces strict platform constraints.
     * Requirements: JDK 21+ AND ( (aarch64 on macOS/Linux) OR (amd64 on Linux) ).
     * @return Boolean indicating expected presence of native kernels.
     */
    public boolean supported() {
        var jdkVersion = Runtime.version().feature();
        var arch = System.getProperty("os.arch");
        var osName = System.getProperty("os.name");

        // Block Logic: Multi-factor platform eligibility check.
        if (jdkVersion >= 21
            && ((arch.equals("aarch64") && (osName.startsWith("Mac") || osName.equals("Linux")))
                || (arch.equals("amd64") && osName.equals("Linux")))) {
            // Invariant: On supported platforms, the native access handle MUST be present.
            assertThat(vectorSimilarityFunctions, isPresent());
            return true;
        } else {
            // Invariant: On unsupported platforms, the native access handle MUST NOT be present (fallback to scalar).
            assertThat(vectorSimilarityFunctions, not(isPresent()));
            return false;
        }
    }

    public static String notSupportedMsg() {
        return "Not supported on [" + platformMsg() + "]";
    }

    /**
     * @brief Generates a standardized telemetry string describing the current execution environment.
     */
    public static String platformMsg() {
        var jdkVersion = Runtime.version().feature();
        var arch = System.getProperty("os.arch");
        var osName = System.getProperty("os.name");
        return "JDK=" + jdkVersion + ", os=" + osName + ", arch=" + arch;
    }
}
