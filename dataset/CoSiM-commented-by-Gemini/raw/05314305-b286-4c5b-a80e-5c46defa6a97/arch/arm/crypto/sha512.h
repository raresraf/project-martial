/**
 * @file sha512.h
 * @brief Forward declaration for NEON-accelerated SHA-512/384 algorithms.
 * @details This header provides the declaration for the `sha512_neon_algs`
 * array, which is defined in `sha512-neon-glue.c`. It is included by the main
 * `sha512-glue.c` file to allow it to conditionally register the NEON-accelerated
 * implementations at runtime if the CPU reports NEON support. This decouples the
 * main glue logic from the NEON-specific implementation.
 */
/* SPDX-License-Identifier: GPL-2.0 */

/**
 * @var sha512_neon_algs
 * @brief An array of shash_alg structures defining the NEON-accelerated
 *        implementations for SHA-512 and SHA-384.
 *
 * This array is defined and initialized in `sha512-neon-glue.c`.
 */
extern struct shash_alg sha512_neon_algs[2];
