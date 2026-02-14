/* SPDX-License-Identifier: GPL-2.0-only */
/**
 * @file sha2.h
 * @brief Internal functions for optimized SHA256 implementations.
 *
 * This header provides internal helper functions for the SHA256 algorithm.
 * It is not part of the public CryptoAPI and is used to abstract the selection
 * between different SHA256 implementations: a generic C version, an
 * architecture-specific optimized version (e.g., using special CPU instructions),
 * and a SIMD version. This allows for performance improvements while maintaining
 * a consistent internal API.
 */

#ifndef _CRYPTO_INTERNAL_SHA2_H
#define _CRYPTO_INTERNAL_SHA2_H

#include <crypto/internal/simd.h>
#include <crypto/sha2.h>
#include <linux/compiler_attributes.h>
#include <linux/string.h>
#include <linux/types.h>
#include <linux/unaligned.h>

#if IS_ENABLED(CONFIG_CRYPTO_ARCH_HAVE_LIB_SHA256)
/**
 * sha256_is_arch_optimized - Checks if an architecture-specific implementation of SHA256 is available.
 *
 * Return: True if an optimized version exists, false otherwise.
 */
bool sha256_is_arch_optimized(void);
#else
static inline bool sha256_is_arch_optimized(void)
{
	return false;
}
#endif
/*
 * Function prototypes for the different SHA256 block processing implementations.
 */
void sha256_blocks_generic(u32 state[SHA256_STATE_WORDS],
			   const u8 *data, size_t nblocks);
void sha256_blocks_arch(u32 state[SHA256_STATE_WORDS],
			const u8 *data, size_t nblocks);
void sha256_blocks_simd(u32 state[SHA256_STATE_WORDS],
			const u8 *data, size_t nblocks);

/**
 * sha256_choose_blocks - Selects and executes the best available SHA256 block implementation.
 * @state: The current SHA256 state array.
 * @data: Pointer to the input data blocks.
 * @nblocks: The number of SHA256 blocks to process.
 * @force_generic: If true, forces the use of the generic C implementation.
 * @force_simd: If true, forces the use of the SIMD implementation if available.
 *
 * This function acts as a dispatcher, selecting the most efficient SHA256
 * block processing function available at runtime. It prioritizes SIMD if
 * available and requested, falls back to an architecture-specific
 * implementation if one exists, and finally uses the generic C version.
 */
static __always_inline void sha256_choose_blocks(
	u32 state[SHA256_STATE_WORDS], const u8 *data, size_t nblocks,
	bool force_generic, bool force_simd)
{
	/* Prioritize the generic implementation if forced or if no arch-specific version exists. */
	if (!IS_ENABLED(CONFIG_CRYPTO_ARCH_HAVE_LIB_SHA256) || force_generic)
		sha256_blocks_generic(state, data, nblocks);
	/*
	 * If a SIMD implementation is available and either forced or deemed usable
	 * by the crypto layer, use it.
	 */
	else if (IS_ENABLED(CONFIG_CRYPTO_ARCH_HAVE_LIB_SHA256_SIMD) &&
		 (force_simd || crypto_simd_usable()))
		sha256_blocks_simd(state, data, nblocks);
	/* Otherwise, fall back to the general architecture-optimized version. */
	else
		sha256_blocks_arch(state, data, nblocks);
}

/**
 * sha256_finup - Finalizes the SHA256 hashing process.
 * @sctx: The SHA256 state context.
 * @buf: A buffer containing the last partial block of data.
 * @len: The length of the data in the buffer.
 * @out: The output buffer for the final digest.
 * @digest_size: The desired size of the output digest (e.g., 32 for SHA256).
 * @force_generic: If true, forces the use of the generic C implementation.
 * @force_simd: If true, forces the use of the SIMD implementation if available.
 *
 * This function handles the final steps of the SHA256 computation: padding the
 * last block, appending the total message length, processing the final block(s),
 * and producing the final digest.
 */
static __always_inline void sha256_finup(
	struct crypto_sha256_state *sctx, u8 buf[SHA256_BLOCK_SIZE],
	size_t len, u8 out[SHA256_DIGEST_SIZE], size_t digest_size,
	bool force_generic, bool force_simd)
{
	const size_t bit_offset = SHA256_BLOCK_SIZE - 8;
	__be64 *bits = (__be64 *)&buf[bit_offset];
	int i;

	/*
	 * Block Logic: Append the padding bit (0x80). If this overflows the current
	 * block, process the current block and start a new one.
	 */
	buf[len++] = 0x80;
	if (len > bit_offset) {
		memset(&buf[len], 0, SHA256_BLOCK_SIZE - len);
		sha256_choose_blocks(sctx->state, buf, 1, force_generic,
				     force_simd);
		len = 0;
	}

	/*
	 * Block Logic: Pad the rest of the block with zeros and append the 64-bit
	 * message length in big-endian format. Process the final block.
	 */
	memset(&buf[len], 0, bit_offset - len);
	*bits = cpu_to_be64(sctx->count << 3);
	sha256_choose_blocks(sctx->state, buf, 1, force_generic, force_simd);

	/* Copy the final state words to the output digest buffer. */
	for (i = 0; i < digest_size; i += 4)
		put_unaligned_be32(sctx->state[i / 4], out + i);
}

#endif /* _CRYPTO_INTERNAL_SHA2_H */
