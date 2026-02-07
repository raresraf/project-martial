// SPDX-License-Identifier: GPL-2.0-or-later
/**
 * @file octeon-sha1.c
 * @brief Hardware-accelerated SHA1 Secure Hash Algorithm using Cavium Octeon's cryptographic co-processor.
 * @details This file provides a hardware-accelerated implementation of the SHA1
 * (Secure Hash Algorithm 1) hash function, specifically adapted for Cavium
 * Octeon processors. It leverages the Octeon's dedicated cryptographic
 * co-processor (COP2) to offload SHA1 compression, significantly improving
 * performance over software-only implementations. The code integrates with
 * the Linux kernel's cryptographic API (`shash`), offering `init`, `update`,
 * `finup` operations. Critical context-switching for COP2 access is managed
 * through `octeon_crypto_enable` and `octeon_crypto_disable` to ensure safe
 * operation in a multitasking kernel environment.
 */
/*
 * Cryptographic API.
 *
 * SHA1 Secure Hash Algorithm.
 *
 * Adapted for OCTEON by Aaro Koskinen <aaro.koskinen@iki.fi>.
 *
 * Based on crypto/sha1_generic.c, which is:
 *
 * Copyright (c) Alan Smithee.
 * Copyright (c) Andrew McDonald <andrew@mcdonald.org.uk>
 * Copyright (c) Jean-Francois Dive <jef@linuxbe.org>
 */

#include <asm/octeon/octeon.h>
#include <crypto/internal/hash.h>
#include <crypto/sha1.h>
#include <crypto/sha1_base.h>
#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/module.h>

#include "octeon-crypto.h"

/*
 * We pass everything as 64-bit. OCTEON can handle misaligned data.
 */

/**
 * @brief Writes the current SHA1 hash state to the Octeon COP2 registers.
 * @details This function takes the 160-bit (5x 32-bit words) SHA1 hash digest
 * from the `sha1_state` context and writes its 32-bit words, packaged as
 * 64-bit quantities, to the appropriate registers within the Octeon
 * cryptographic co-processor (COP2). This primes the hardware with the current
 * hash state before processing new data blocks. Special handling is needed for
 * `sctx->state[4]` as it's a single 32-bit word but the COP2 interface is 64-bit.
 *
 * @param sctx Pointer to the `sha1_state` structure containing the hash digest.
 * Functional Utility: Transfers the software-maintained SHA1 hash state to the Octeon COP2 hardware.
 */
static void octeon_sha1_store_hash(struct sha1_state *sctx)
{
	u64 *hash = (u64 *)sctx->state;
	union {
		u32 word[2];
		u64 dword;
	} hash_tail = { { sctx->state[4], } }; // Functional Utility: Packages the 32-bit H4 into a 64-bit union for COP2 transfer.

	// Functional Utility: Writes the first 64-bit part (H0, H1) of the hash digest to COP2 register 0.
	write_octeon_64bit_hash_dword(hash[0], 0);
	// Functional Utility: Writes the second 64-bit part (H2, H3) of the hash digest to COP2 register 1.
	write_octeon_64bit_hash_dword(hash[1], 1);
	// Functional Utility: Writes the third 64-bit part (H4 and padding) to COP2 register 2.
	write_octeon_64bit_hash_dword(hash_tail.dword, 2);
	// Functional Utility: Securely clears the temporary union buffer for security reasons.
	memzero_explicit(&hash_tail.word[0], sizeof(hash_tail.word[0]));
}

/**
 * @brief Reads the updated SHA1 hash state from the Octeon COP2 registers.
 * @details This function retrieves the 160-bit SHA1 hash digest from the
 * Octeon cryptographic co-processor (COP2) registers and unpacks its 64-bit
 * portions into the `sha1_state` context. This updates the software-maintained
 * hash state after the hardware has processed a message block. It includes
 * logic to correctly extract the 32-bit H4 from the 64-bit COP2 register.
 *
 * @param sctx Pointer to the `sha1_state` structure where the hash digest will be stored.
 * Functional Utility: Retrieves the hardware-computed SHA1 hash state from the Octeon COP2.
 */
static void octeon_sha1_read_hash(struct sha1_state *sctx)
{
	u64 *hash = (u64 *)sctx->state;
	union {
		u32 word[2];
		u64 dword;
	} hash_tail;

	// Functional Utility: Reads the first 64-bit part (H0, H1) of the hash digest from COP2 register 0.
	hash[0]		= read_octeon_64bit_hash_dword(0);
	// Functional Utility: Reads the second 64-bit part (H2, H3) of the hash digest from COP2 register 1.
	hash[1]		= read_octeon_64bit_hash_dword(1);
	// Functional Utility: Reads the third 64-bit part (containing H4) from COP2 register 2.
	hash_tail.dword	= read_octeon_64bit_hash_dword(2);
	// Functional Utility: Extracts the 32-bit H4 from the 64-bit union.
	sctx->state[4]	= hash_tail.word[0];
	// Functional Utility: Securely clears the temporary union buffer for security reasons.
	memzero_explicit(&hash_tail.dword, sizeof(hash_tail.dword));
}

/**
 * @brief Processes one or more 64-byte SHA1 blocks using the Octeon COP2 hardware.
 * @details This function iteratively takes 64-byte message blocks, loads their
 * 64-bit words into the Octeon cryptographic co-processor (COP2) registers,
 * and then triggers the hardware-accelerated SHA1 compression round for each block.
 * This offloads the intensive SHA1 computation to dedicated hardware, processing
 * multiple blocks in a loop.
 *
 * @param sctx Pointer to the SHA1 state context (though not directly modified here).
 * @param src Pointer to the input data blocks.
 * @param blocks The number of 64-byte blocks to process.
 * Functional Utility: Performs hardware-accelerated SHA1 compression for multiple 64-byte blocks using Octeon COP2.
 */
static void octeon_sha1_transform(struct sha1_state *sctx, const u8 *src,
				  int blocks)
{
	// Block Logic: Iterates through the given number of 64-byte blocks.
	do {
		const u64 *block = (const u64 *)src;

		// Functional Utility: Loads individual 64-bit words of the current message block into the COP2 unit.
		write_octeon_64bit_block_dword(block[0], 0);
		write_octeon_64bit_block_dword(block[1], 1);
		write_octeon_64bit_block_dword(block[2], 2);
		write_octeon_64bit_block_dword(block[3], 3);
		write_octeon_64bit_block_dword(block[4], 4);
		write_octeon_64bit_block_dword(block[5], 5);
		write_octeon_64bit_block_dword(block[6], 6);
		// Functional Utility: Loads the final 64-bit word of the message block and triggers the hardware SHA1 compression.
		octeon_sha1_start(block[7]);

		// Functional Utility: Advances the source pointer to the next 64-byte block.
		src += SHA1_BLOCK_SIZE;
	} while (--blocks); // Functional Utility: Continues looping until all specified blocks are processed.
}

/**
 * @brief Updates the SHA1 hash with new data.
 * @details This function processes input data incrementally, feeding full
 * 64-byte blocks to the Octeon COP2 for hardware-accelerated SHA1 compression.
 * It carefully manages the COP2 context: enabling it before hardware access
 * and disabling it afterward to prevent conflicts and ensure correct state
 * restoration. It uses the generic `sha1_base_do_update_blocks` to handle
 * data buffering and partial block management.
 *
 * @param desc The `shash_desc` descriptor for the current SHA1 operation.
 * @param data Pointer to the input data.
 * @param len  The length of the input data.
 * @return The number of bytes not processed (typically 0 on success if all blocks processed).
 * Functional Utility: Accumulates new data into the SHA1 hash state, utilizing Octeon COP2 for hardware acceleration on full blocks.
 */
static int octeon_sha1_update(struct shash_desc *desc, const u8 *data,
			unsigned int len)
{
	struct sha1_state *sctx = shash_desc_ctx(desc);
	struct octeon_cop2_state state;
	unsigned long flags;
	int remain;

	// Functional Utility: Enables kernel access to the Octeon COP2, saving its previous state.
	flags = octeon_crypto_enable(&state);
	// Functional Utility: Transfers the current SHA1 hash state from software to the COP2 hardware.
	octeon_sha1_store_hash(sctx);

	// Functional Utility: Delegates data buffering and block processing to the base SHA1 update function,
	// using the Octeon hardware-accelerated `octeon_sha1_transform` as the core block transformation.
	remain = sha1_base_do_update_blocks(desc, data, len,
					    octeon_sha1_transform);

	// Functional Utility: Reads the updated SHA1 hash state from the COP2 hardware back into software.
	octeon_sha1_read_hash(sctx);
	// Functional Utility: Disables kernel access to the Octeon COP2 and restores its previous state.
	octeon_crypto_disable(&state, flags);
	// Functional Utility: Returns the number of bytes remaining from the input data that were not processed as full blocks.
	return remain;
}

/**
 * @brief Finalizes the SHA1 hash computation, including padding and length appending.
 * @details This function handles the final steps of the SHA1 algorithm. It applies
 * the necessary padding to the last data block, appends the total message length,
 * and then processes this final block using the Octeon COP2 for hardware-
 * accelerated compression. It also ensures proper COP2 context management during
 * this critical phase. The final digest is then written to the output buffer.
 *
 * @param desc The `shash_desc` descriptor for the current SHA1 operation.
 * @param src Pointer to any remaining partial input data.
 * @param len The length of the remaining partial input data.
 * @param out The buffer to store the final 20-byte SHA1 hash digest.
 * @return 0 on success.
 * Functional Utility: Completes the SHA1 hashing process, including padding and length encoding, using Octeon COP2 hardware.
 */
static int octeon_sha1_finup(struct shash_desc *desc, const u8 *src,
			     unsigned int len, u8 *out)
{
	struct sha1_state *sctx = shash_desc_ctx(desc);
	struct octeon_cop2_state state;
	unsigned long flags;

	// Functional Utility: Enables kernel access to the Octeon COP2, saving its previous state.
	flags = octeon_crypto_enable(&state);
	// Functional Utility: Transfers the current SHA1 hash state from software to the COP2 hardware.
	octeon_sha1_store_hash(sctx);

	// Functional Utility: Processes any remaining input data and applies SHA1 padding using the hardware-accelerated transform.
	sha1_base_do_finup(desc, src, len, octeon_sha1_transform);

	// Functional Utility: Reads the final SHA1 hash state from the COP2 hardware back into software.
	octeon_sha1_read_hash(sctx);
	// Functional Utility: Disables kernel access to the Octeon COP2 and restores its previous state.
	octeon_crypto_disable(&state, flags);
	// Functional Utility: Writes the final computed SHA1 hash digest to the output buffer.
	return sha1_base_finish(desc, out);
}

/**
 * @brief Defines the Octeon hardware-accelerated SHA1 algorithm for the crypto API.
 * @details This structure registers the SHA1 algorithm implementation that leverages
 * the Cavium Octeon's cryptographic co-processor (COP2). It specifies the algorithm's
 * properties (digest size, block size), associates the core operations (`init`,
 * `update`, `finup`) with their respective handler functions, and sets a high
 * priority (`OCTEON_CR_OPCODE_PRIORITY`) to ensure this hardware-accelerated version
 * is preferred over generic software implementations.
 * Functional Utility: Registers the Octeon hardware-accelerated SHA1 algorithm with the kernel crypto API.
 */
static struct shash_alg octeon_sha1_alg = {
	.digestsize	=	SHA1_DIGEST_SIZE,
	.init		=	sha1_base_init,
	.update		=	octeon_sha1_update,
	.finup		=	octeon_sha1_finup,
	.descsize	=	SHA1_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha1",
		.cra_driver_name=	"octeon-sha1",
		.cra_priority	=	OCTEON_CR_OPCODE_PRIORITY,
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY,
		.cra_blocksize	=	SHA1_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
};

/**
 * @brief Module initialization function for Octeon SHA1.
 * @details This function is the entry point when the kernel module is loaded.
 * It first checks if the Octeon CPU supports cryptographic operations via
 * `octeon_has_crypto()`. If hardware crypto is available, it registers the
 * hardware-accelerated SHA1 algorithm (`octeon_sha1_alg` structure) with the
 * Linux kernel's cryptographic API. This makes the `octeon-sha1` driver
 * available to the system, enabling high-performance SHA1 hashing.
 * Functional Utility: Registers the hardware-accelerated Octeon SHA1 algorithm with the kernel crypto API after checking for hardware support.
 * @return 0 on successful registration, or `-ENOTSUPP` if Octeon crypto hardware is not available, or other error code on failure.
 */
static int __init octeon_sha1_mod_init(void)
{
	// Pre-condition: Checks if the Octeon CPU has cryptographic hardware support.
	if (!octeon_has_crypto())
		return -ENOTSUPP; // Functional Utility: Returns an error if crypto hardware is not supported.
	// Functional Utility: Registers the Octeon hardware-accelerated SHA1 algorithm with the kernel crypto API.
	return crypto_register_shash(&octeon_sha1_alg);
}

/**
 * @brief Module cleanup function for Octeon SHA1.
 * @details This function is the exit point when the kernel module is unloaded.
 * Functional Utility: Unregisters the hardware-accelerated SHA1 algorithm
 * (`octeon_sha1_alg` structure) from the Linux kernel's cryptographic API.
 * This cleanly removes the `octeon-sha1` driver from the system, releasing
 * associated resources and preventing any lingering references after the
 * module is no longer in use. This ensures proper resource management upon module unload.
 * Functional Utility: Unregisters the hardware-accelerated Octeon SHA1 algorithm from the kernel crypto API.
 */
static void __exit octeon_sha1_mod_fini(void)
{
	crypto_unregister_shash(&octeon_sha1_alg);
}

module_init(octeon_sha1_mod_init);
module_exit(octeon_sha1_mod_fini);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SHA1 Secure Hash Algorithm (OCTEON)");
MODULE_AUTHOR("Aaro Koskinen <aaro.koskinen@iki.fi>");
