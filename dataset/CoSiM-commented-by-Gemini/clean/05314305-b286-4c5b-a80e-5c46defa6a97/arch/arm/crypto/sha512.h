/**
 * @file sha512.h
 * @brief Forward declaration for NEON-accelerated SHA-512/384 algorithms.
 * @details This header provides the forward declaration for the `sha512_neon_algs`
 * array, which contains the `shash_alg` structures defining the NEON-accelerated
 * implementations for SHA-512 and SHA-384. This array is fully defined and
 * initialized in `sha512-neon-glue.c`. The purpose of this header is to allow
 * the main `sha512-glue.c` file to access and conditionally register these
 * NEON-accelerated implementations at runtime, based on CPU feature detection.
 * This architectural approach decouples the main glue logic from the NEON-specific
 * implementation details, allowing for flexible and modular integration of
 * different acceleration layers.
 */
/* SPDX-License-Identifier: GPL-2.0 */

/**
 * @var sha512_neon_algs
 * @brief An array of `shash_alg` structures defining the NEON-accelerated
 *        implementations for SHA-512 and SHA-384.
 * @details This array holds the registration data for the high-priority,
 * NEON-optimized SHA-512 and SHA-384 algorithms. It is explicitly declared
 * `extern` here because its definition and initialization reside in
 * `sha512-neon-glue.c`. This declaration allows `sha512-glue.c` to conditionally
 * reference and register these algorithms with the kernel's cryptographic API
 * when NEON hardware capabilities are detected.
 */
extern struct shash_alg sha512_neon_algs[2];
