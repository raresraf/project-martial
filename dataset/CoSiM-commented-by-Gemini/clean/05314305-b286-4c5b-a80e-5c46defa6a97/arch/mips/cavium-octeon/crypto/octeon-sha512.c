// SPDX-License-Identifier: GPL-2.0-or-later
/**
 * @file octeon-sha512.c
 * @brief Hardware-accelerated SHA-512 and SHA-384 Secure Hash Algorithms using Cavium Octeon's cryptographic co-processor.
 * @details This file provides hardware-accelerated implementations for both
 * SHA-512 and SHA-384 hash functions, specifically adapted for Cavium Octeon
 * processors. It leverages the Octeon's dedicated cryptographic co-processor
 * (COP2) to offload SHA-512/384 compression, significantly improving
 * performance over software-only implementations. The code integrates with
 * the Linux kernel's cryptographic API (`shash`), offering `init`, `update`,
 * and `finup` operations. Critical context-switching for COP2 access is
 * managed through `octeon_crypto_enable` and `octeon_crypto_disable`
 * to ensure safe operation in a multitasking kernel environment.
 */
/*
 * Cryptographic API.
 *
 * SHA-512 and SHA-384 Secure Hash Algorithm.
 *
 * Adapted for OCTEON by Aaro Koskinen <aaro.koskinen@iki.fi>.
 *
 * Based on crypto/sha512_generic.c, which is:
 *
 * Copyright (c) Jean-Luc Cooke <jlcooke@certainkey.com>
 * Copyright (c) Andrew McDonald <andrew@mcdonald.org.uk>
 * Copyright (c) 2003 Kyle McMartin <kyle@debian.org>
 */

#include <asm/octeon/octeon.h>
#include <crypto/internal/hash.h>
#include <crypto/sha2.h>
#include <crypto/sha512_base.h>
#include <linux/kernel.h>
#include <linux/module.h>

#include "octeon-crypto.h"

/*
 * We pass everything as 64-bit. OCTEON can handle misaligned data.
 */

/**
 * @brief Writes the current SHA-512/384 hash state to the Octeon COP2 registers.
 * @details This function transfers the 512-bit (8x 64-bit words) SHA-512/384
 * hash digest from the `sha512_state` context in memory to the appropriate
 * 64-bit registers within the Octeon cryptographic co-processor (COP2).
 * This primes the hardware with the current hash state before processing new data blocks.
 *
 * @param sctx Pointer to the `sha512_state` structure containing the hash digest.
 * Functional Utility: Transfers the software-maintained SHA-512/384 hash state to the Octeon COP2 hardware.
 */
static void octeon_sha512_store_hash(struct sha512_state *sctx)
{
	// Functional Utility: Writes each 64-bit word of the hash digest to the corresponding COP2 register.
	write_octeon_64bit_hash_sha512(sctx->state[0], 0);
	write_octeon_64bit_hash_sha512(sctx->state[1], 1);
	write_octeon_64bit_hash_sha512(sctx->state[2], 2);
	write_octeon_64bit_hash_sha512(sctx->state[3], 3);
	write_octeon_64bit_hash_sha512(sctx->state[4], 4);
	write_octeon_64bit_hash_sha512(sctx->state[5], 5);
	write_octeon_64bit_hash_sha512(sctx->state[6], 6);
	write_octeon_64bit_hash_sha512(sctx->state[7], 7);
}

/**
 * @brief Reads the updated SHA-512/384 hash state from the Octeon COP2 registers.
 * @details This function retrieves the 512-bit SHA-512/384 hash digest from the
 * Octeon cryptographic co-processor (COP2) registers and stores its 64-bit
 * words into the `sha512_state` context in memory. This updates the software-
 * maintained hash state after the hardware has processed a message block.
 *
 * @param sctx Pointer to the `sha512_state` structure where the hash digest will be stored.
 * Functional Utility: Retrieves the hardware-computed SHA-512/384 hash state from the Octeon COP2.
 */
static void octeon_sha512_read_hash(struct sha512_state *sctx)
{
	// Functional Utility: Reads each 64-bit word of the hash digest from the corresponding COP2 register.
	sctx->state[0] = read_octeon_64bit_hash_sha512(0);
	sctx->state[1] = read_octeon_64bit_hash_sha512(1);
	sctx->state[2] = read_octeon_64bit_hash_sha512(2);
	sctx->state[3] = read_octeon_64bit_hash_sha512(3);
	sctx->state[4] = read_octeon_64bit_hash_sha512(4);
	sctx->state[5] = read_octeon_64bit_hash_sha512(5);
	sctx->state[6] = read_octeon_64bit_hash_sha512(6);
	sctx->state[7] = read_octeon_64bit_hash_sha512(7);
}

/**
 * @brief Processes one or more 128-byte SHA-512/384 blocks using the Octeon COP2 hardware.
 * @details This function iteratively takes 128-byte message blocks, loads their
 * 64-bit words into the Octeon cryptographic co-processor (COP2) registers,
 * and then triggers the hardware-accelerated SHA-512/384 compression round for each block.
 * This offloads the intensive SHA-512/384 computation to dedicated hardware, processing
 * multiple blocks in a loop.
 *
 * @param sctx Pointer to the SHA-512 state context (though not directly modified here).
 * @param src Pointer to the input data blocks.
 * @param blocks The number of 128-byte blocks to process.
 * Functional Utility: Performs hardware-accelerated SHA-512/384 compression for multiple 128-byte blocks using Octeon COP2.
 */
static void octeon_sha512_transform(struct sha512_state *sctx,
				    const u8 *src, int blocks)
{
	// Block Logic: Iterates through the given number of 128-byte blocks.
	do {
		const u64 *block = (const u64 *)src;

		// Functional Utility: Loads individual 64-bit words of the current 128-byte message block into the COP2 unit.
		write_octeon_64bit_block_sha512(block[0], 0);
		write_octeon_64bit_block_sha512(block[1], 1);
		write_octeon_64bit_block_sha512(block[2], 2);
		write_octeon_64bit_block_sha512(block[3], 3);
		write_octeon_64bit_block_sha512(block[4], 4);
		write_octeon_64bit_block_sha512(block[5], 5);
		write_octeon_64bit_block_sha512(block[6], 6);
		write_octeon_64bit_block_sha512(block[7], 7);
		write_octeon_64bit_block_sha512(block[8], 8);
		write_octeon_64bit_block_sha512(block[9], 9);
		write_octeon_64bit_block_sha512(block[10], 10);
		write_octeon_64bit_block_sha512(block[11], 11);
		write_octeon_64bit_block_sha512(block[12], 12);
		write_octeon_64bit_block_sha512(block[13], 13);
		write_octeon_64bit_block_sha512(block[14], 14);
		// Functional Utility: Loads the final 64-bit word of the message block and triggers the hardware SHA-512 compression.
		octeon_sha512_start(block[15]);

		// Functional Utility: Advances the source pointer to the next 128-byte block.
		src += SHA512_BLOCK_SIZE;
	} while (--blocks); // Functional Utility: Continues looping until all specified blocks are processed.
}

/**
 * @brief Updates the SHA-512/384 hash with new data.
 * @details This function processes input data incrementally, feeding full
 * 128-byte blocks to the Octeon COP2 for hardware-accelerated SHA-512/384 compression.
 * It carefully manages the COP2 context: enabling it before hardware access
 * and disabling it afterward to prevent conflicts and ensure correct state
 * restoration. It uses the generic `sha512_base_do_update_blocks` to handle
 * data buffering and partial block management.
 *
 * @param desc The `shash_desc` descriptor for the current SHA-512/384 operation.
 * @param data Pointer to the input data.
 * @param len  The length of the input data.
 * @return The number of bytes not processed (due to being a partial block).
 * Functional Utility: Accumulates new data into the SHA-512/384 hash state, utilizing Octeon COP2 for hardware acceleration on full blocks.
 */
static int octeon_sha512_update(struct shash_desc *desc, const u8 *data,
				unsigned int len)
{
	struct sha512_state *sctx = shash_desc_ctx(desc);
	struct octeon_cop2_state state;
	unsigned long flags;
	int remain;

	// Functional Utility: Enables kernel access to the Octeon COP2, saving its previous state.
	flags = octeon_crypto_enable(&state);
	// Functional Utility: Transfers the current SHA-512/384 hash state from software to the COP2 hardware.
	octeon_sha512_store_hash(sctx);

	// Functional Utility: Delegates data buffering and block processing to the base SHA-512 update function,
	// using the Octeon hardware-accelerated `octeon_sha512_transform` as the core block transformation.
	remain = sha512_base_do_update_blocks(desc, data, len,
					      octeon_sha512_transform);

	// Functional Utility: Reads the updated SHA-512/384 hash state from the COP2 hardware back into software.
	octeon_sha512_read_hash(sctx);
	// Functional Utility: Disables kernel access to the Octeon COP2 and restores its previous state.
	octeon_crypto_disable(&state, flags);
	// Functional Utility: Returns the number of bytes remaining from the input data that were not processed as full blocks.
	return remain;
}

/**
 * @brief Finalizes the SHA-512/384 hash computation, including padding and length appending.
 * @details This function handles the final steps of the SHA-512/384 algorithm. It applies
 * the necessary padding to the last data block, appends the total message length,
 * and then processes this final block using the Octeon COP2 for hardware-
 * accelerated compression. It also ensures proper COP2 context management during
 * this critical phase. The final digest is then written to the output buffer.
 *
 * @param desc The `shash_desc` descriptor for the current SHA-512/384 operation.
 * @param src Pointer to any remaining partial input data.
 * @param len The length of the remaining partial input data.
 * @param hash The buffer to store the final SHA-512/384 hash digest.
 * @return 0 on success.
 * Functional Utility: Completes the SHA-512/384 hashing process, including padding and length encoding, using Octeon COP2 hardware.
 */
static int octeon_sha512_finup(struct shash_desc *desc, const u8 *src,
			       unsigned int len, u8 *hash)
{
	struct sha512_state *sctx = shash_desc_ctx(desc);
	struct octeon_cop2_state state;
	unsigned long flags;

	// Functional Utility: Enables kernel access to the Octeon COP2, saving its previous state.
	flags = octeon_crypto_enable(&state);
	// Functional Utility: Transfers the current SHA-512/384 hash state from software to the COP2 hardware.
	octeon_sha512_store_hash(sctx);

	// Functional Utility: Processes any remaining input data and applies SHA-512 padding using the hardware-accelerated transform.
	sha512_base_do_finup(desc, src, len, octeon_sha512_transform);

	// Functional Utility: Reads the final SHA-512/384 hash state from the COP2 hardware back into software.
	octeon_sha512_read_hash(sctx);
	// Functional Utility: Disables kernel access to the Octeon COP2 and restores its previous state.
	octeon_crypto_disable(&state, flags);
	// Functional Utility: Writes the final computed SHA-512/384 hash digest to the output buffer.
	return sha512_base_finish(desc, hash);
}

/**
 * @brief Defines the Octeon hardware-accelerated SHA-512 and SHA-384 algorithms for the crypto API.
 * @details This array of `shash_alg` structures registers both the SHA-512
 * (`octeon-sha512`) and SHA-384 (`octeon-sha384`) algorithm implementations
 * that leverage the Cavium Octeon's cryptographic co-processor (COP2). It
 * specifies the algorithms' properties (digest size, block size), associates
 * the core operations (`init`, `update`, `finup`) with their respective handler
 * functions, and sets a high priority (`OCTEON_CR_OPCODE_PRIORITY`) to ensure
 * these hardware-accelerated versions are preferred over generic software implementations.
 * Functional Utility: Registers Octeon hardware-accelerated SHA-512 and SHA-384 algorithms with the kernel crypto API.
 */
static struct shash_alg octeon_sha512_algs[2] = { {
	.digestsize	=	SHA512_DIGEST_SIZE,
	.init		=	sha512_base_init,
	.update		=	octeon_sha512_update,
	.finup		=	octeon_sha512_finup,
	.descsize	=	SHA512_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha512",
		.cra_driver_name=	"octeon-sha512",
		.cra_priority	=	OCTEON_CR_OPCODE_PRIORITY,
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY |
					CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize	=	SHA512_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
}, {
	.digestsize	=	SHA384_DIGEST_SIZE,
	.init		=	sha384_base_init,
	.update		=	octeon_sha512_update,
	.finup		=	octeon_sha512_finup,
	.descsize	=	SHA512_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha384",
		.cra_driver_name=	"octeon-sha384",
		.cra_priority	=	OCTEON_CR_OPCODE_PRIORITY,
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY |
					CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize	=	SHA384_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
} };

/**
 * @brief Module initialization function for Octeon SHA-512 and SHA-384.
 * @details This function is the entry point when the kernel module is loaded.
 * It first checks if the Octeon CPU supports cryptographic operations via
 * `octeon_has_crypto()`. If hardware crypto is available, it registers both
 * the hardware-accelerated SHA-512 and SHA-384 algorithms (`octeon_sha512_algs`
 * array) with the Linux kernel's cryptographic API. This makes these drivers
 * available to the system, enabling high-performance SHA-512/384 hashing.
 * Functional Utility: Registers the hardware-accelerated Octeon SHA-512 and SHA-384 algorithms with the kernel crypto API after checking for hardware support.
 * @return 0 on successful registration, or `-ENOTSUPP` if Octeon crypto hardware is not available, or other error code on failure.
 */
static int __init octeon_sha512_mod_init(void)
{
	// Pre-condition: Checks if the Octeon CPU has cryptographic hardware support.
	if (!octeon_has_crypto())
		return -ENOTSUPP; // Functional Utility: Returns an error if crypto hardware is not supported.
	// Functional Utility: Registers both the Octeon hardware-accelerated SHA-512 and SHA-384 algorithms with the kernel crypto API.
	return crypto_register_shashes(octeon_sha512_algs,
				       ARRAY_SIZE(octeon_sha512_algs));
}

/**
 * @brief Module cleanup function for Octeon SHA-512 and SHA-384.
 * @details This function is the exit point when the kernel module is unloaded.
 * Functional Utility: Unregisters all hardware-accelerated SHA-512/384 algorithms
 * (`octeon_sha512_algs` array) from the Linux kernel's cryptographic API.
 * This cleanly removes the drivers from the system, releasing associated resources
 * and preventing any lingering references after the module is no longer in use.
 * This ensures proper resource management upon module unload.
 * Functional Utility: Unregisters the hardware-accelerated Octeon SHA-512 and SHA-384 algorithms from the kernel crypto API.
 */
static void __exit octeon_sha512_mod_fini(void)
{
	crypto_unregister_shashes(octeon_sha512_algs,
				  ARRAY_SIZE(octeon_sha512_algs));
}

module_init(octeon_sha512_mod_init);
module_exit(octeon_sha512_mod_fini);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SHA-512 and SHA-384 Secure Hash Algorithms (OCTEON)");
MODULE_AUTHOR("Aaro Koskinen <aaro.koskinen@iki.fi>");
