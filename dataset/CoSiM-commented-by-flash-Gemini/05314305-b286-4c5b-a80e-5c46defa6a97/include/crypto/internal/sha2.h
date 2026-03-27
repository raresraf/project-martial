/**
 * @file sha2.h
 * @brief Internal definitions and function declarations for SHA256 cryptographic hash algorithm.
 *
 * This header file provides internal interfaces for various implementations of
 * the SHA256 algorithm, including generic, architecture-optimized, and SIMD-accelerated
 * versions. It also defines helper functions for processing data blocks and
 * finalizing the hash calculation within the Linux kernel's crypto subsystem.
 *
 * Domain: Linux Kernel, Cryptography, SHA256, Hash Functions, Performance Optimization.
 */

/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef _CRYPTO_INTERNAL_SHA2_H
#define _CRYPTO_INTERNAL_SHA2_H

#include <crypto/internal/simd.h>
#include <crypto/sha2.h>
#include <linux/compiler_attributes.h>
#include <linux/string.h>
#include <linux/types.h>
#include <linux/unaligned.h>

/**
 * @brief Checks if an architecture-optimized SHA256 implementation is available.
 *
 * This function provides a way to query the availability of a highly optimized
 * SHA256 implementation for the current CPU architecture. This is crucial for
 * performance-sensitive cryptographic operations.
 *
 * @return true if an architecture-optimized implementation exists, false otherwise.
 */
#if IS_ENABLED(CONFIG_CRYPTO_ARCH_HAVE_LIB_SHA256)
bool sha256_is_arch_optimized(void);
#else
static inline bool sha256_is_arch_optimized(void)
{
	return false;
}
#endif

/**
 * @brief Processes SHA256 blocks using a generic (software) implementation.
 * @param state The current SHA256 hash state.
 * @param data Pointer to the input data blocks.
 * @param nblocks The number of 512-bit blocks to process.
 */
void sha256_blocks_generic(u32 state[SHA256_STATE_WORDS],
			   const u8 *data, size_t nblocks);

/**
 * @brief Processes SHA256 blocks using an architecture-specific optimized implementation.
 * @param state The current SHA256 hash state.
 * @param data Pointer to the input data blocks.
 * @param nblocks The number of 512-bit blocks to process.
 */
void sha256_blocks_arch(u32 state[SHA256_STATE_WORDS],
			const u8 *data, size_t nblocks);

/**
 * @brief Processes SHA256 blocks using a SIMD-accelerated implementation.
 * @param state The current SHA256 hash state.
 * @param data Pointer to the input data blocks.
 * @param nblocks The number of 512-bit blocks to process.
 */
void sha256_blocks_simd(u32 state[SHA256_STATE_WORDS],
			const u8 *data, size_t nblocks);

/**
 * @brief Selects and executes the most optimal SHA256 block processing function.
 *
 * This function acts as a dispatcher, intelligently choosing between generic,
 * architecture-specific, or SIMD-accelerated SHA256 block processing
 * implementations based on kernel configuration and runtime capabilities (e.g., SIMD availability).
 *
 * @param state The current SHA256 hash state.
 * @param data Pointer to the input data blocks.
 * @param nblocks The number of 512-bit blocks to process.
 * @param force_generic If true, forces the use of the generic implementation.
 * @param force_simd If true, forces the use of the SIMD implementation if available.
 *
 * Performance Optimization: This mechanism ensures that the fastest available
 * SHA256 implementation is used, falling back to a generic software version
 * if hardware acceleration is not present or explicitly disabled.
 */
static __always_inline void sha256_choose_blocks(
	u32 state[SHA256_STATE_WORDS], const u8 *data, size_t nblocks,
	bool force_generic, bool force_simd)
{
	// Block Logic: Conditional execution based on architecture optimization and SIMD capabilities.
	// Prioritizes generic, then SIMD, then architecture-specific implementations.
	if (!IS_ENABLED(CONFIG_CRYPTO_ARCH_HAVE_LIB_SHA256) || force_generic)
		sha256_blocks_generic(state, data, nblocks);
	else if (IS_ENABLED(CONFIG_CRYPTO_ARCH_HAVE_LIB_SHA256_SIMD) &&
		 (force_simd || crypto_simd_usable()))
		sha256_blocks_simd(state, data, nblocks);
	else
		sha256_blocks_arch(state, data, nblocks);
}

/**
 * @brief Finalizes the SHA256 hash calculation after all data has been processed.
 *
 * This function performs the padding required by the SHA256 standard and then
 * processes the final data block(s) to produce the resulting hash digest.
 *
 * @param sctx Pointer to the SHA256 context structure.
 * @param buf Temporary buffer for padding and final block processing.
 * @param len The current length of valid data in the buffer.
 * @param out Output buffer to store the final SHA256 digest.
 * @param digest_size The size of the desired digest (e.g., SHA256_DIGEST_SIZE).
 * @param force_generic If true, forces the use of the generic implementation for final blocks.
 * @param force_simd If true, forces the use of the SIMD implementation for final blocks if available.
 *
 * Pre-condition: All intermediate data blocks have been processed.
 * Post-condition: The 'out' buffer contains the final SHA256 hash digest.
 */
static __always_inline void sha256_finup(
	struct crypto_sha256_state *sctx, u8 buf[SHA256_BLOCK_SIZE],
	size_t len, u8 out[SHA256_DIGEST_SIZE], size_t digest_size,
	bool force_generic, bool force_simd)
{
	const size_t bit_offset = SHA256_BLOCK_SIZE - 8; // Offset for appending message length.
	__be64 *bits = (__be64 *)&buf[bit_offset]; // Pointer to where total bits will be stored.
	int i;

	// Block Logic: Appends the '1' bit and checks if padding requires a new block.
	// SHA256 padding rule: append a '1' bit, then zeros, then 64-bit message length.
	// If appending '1' bit pushes past boundary for message length, process current block.
	buf[len++] = 0x80;
	if (len > bit_offset) {
		// Functional Utility: Zero-pads the remainder of the current block.
		memset(&buf[len], 0, SHA256_BLOCK_SIZE - len);
		// Functional Utility: Processes the padded block.
		sha256_choose_blocks(sctx->state, buf, 1, force_generic,
				     force_simd);
		len = 0; // Reset length for the new padding block.
	}

	// Functional Utility: Zero-pads the buffer up to the point where the message length will be appended.
	memset(&buf[len], 0, bit_offset - len);
	// Functional Utility: Appends the total message length in bits (big-endian).
	*bits = cpu_to_be64(sctx->count << 3);
	// Functional Utility: Processes the final padded block(s).
	sha256_choose_blocks(sctx->state, buf, 1, force_generic, force_simd);

	// Block Logic: Converts the internal hash state to the final digest format.
	// The 32-bit state words are converted to big-endian and written to the output buffer.
	for (i = 0; i < digest_size; i += 4)
		put_unaligned_be32(sctx->state[i / 4], out + i);
}

#endif /* _CRYPTO_INTERNAL_SHA2_H */
