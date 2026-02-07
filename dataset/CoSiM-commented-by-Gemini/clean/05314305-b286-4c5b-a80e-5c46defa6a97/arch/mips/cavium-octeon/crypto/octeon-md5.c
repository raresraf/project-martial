/**
 * @file octeon-md5.c
 * @brief Hardware-accelerated MD5 Message Digest Algorithm using Cavium Octeon's cryptographic co-processor.
 * @details This file provides a hardware-accelerated implementation of the MD5
 * (Message Digest Algorithm 5) hash function, specifically adapted for Cavium
 * Octeon processors. It leverages the Octeon's dedicated cryptographic
 * co-processor (COP2) to offload MD5 compression, significantly improving
 * performance over software-only implementations. The code integrates with
 * the Linux kernel's cryptographic API (`shash`), offering `init`, `update`,
 * `finup`, `export`, and `import` operations, all designed to utilize the
 * Octeon hardware for optimal efficiency. Critical context-switching for COP2
 * access is managed through `octeon_crypto_enable` and `octeon_crypto_disable`
 * to ensure safe operation.
 */
/*
 * Cryptographic API.
 *
 * MD5 Message Digest Algorithm (RFC1321).
 *
 * Adapted for OCTEON by Aaro Koskinen <aaro.koskinen@iki.fi>.
 *
 * Based on crypto/md5.c, which is:
 *
 * Derived from cryptoapi implementation, originally based on the
 * public domain implementation written by Colin Plumb in 1993.
 *
 * Copyright (c) Cryptoapi developers.
 * Copyright (c) 2002 James Morris <jmorris@intercode.com.au>
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 */

#include <asm/octeon/octeon.h>
#include <crypto/internal/hash.h>
#include <crypto/md5.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/string.h>
#include <linux/unaligned.h>

#include "octeon-crypto.h"

/**
 * @struct octeon_md5_state
 * @brief Custom state structure for the Octeon MD5 implementation.
 * @details This structure holds the essential state variables for the MD5
 * algorithm when processed by the Octeon hardware. It stores the 128-bit
 * intermediate hash digest (`hash`) and the total number of bytes processed
 * so far (`byte_count`). The hash is stored as an array of `__le32` (little-endian 32-bit integers).
 */
struct octeon_md5_state {
	__le32 hash[MD5_HASH_WORDS];
	u64 byte_count;
};

/*
 * We pass everything as 64-bit. OCTEON can handle misaligned data.
 */

/**
 * @brief Writes the current MD5 hash state to the Octeon COP2 registers.
 * @details This function takes the 128-bit MD5 hash digest stored in the
 * `octeon_md5_state` context and writes its 64-bit portions to the
 * appropriate registers within the Octeon cryptographic co-processor (COP2).
 * This primes the hardware with the current hash state before processing new data.
 *
 * @param ctx Pointer to the `octeon_md5_state` structure containing the hash digest.
 * Functional Utility: Transfers the software-maintained MD5 hash state to the Octeon COP2 hardware.
 */
static void octeon_md5_store_hash(struct octeon_md5_state *ctx)
{
	u64 *hash = (u64 *)ctx->hash;

	// Functional Utility: Writes the first 64-bit part of the hash digest to COP2 register 0.
	write_octeon_64bit_hash_dword(hash[0], 0);
	// Functional Utility: Writes the second 64-bit part of the hash digest to COP2 register 1.
	write_octeon_64bit_hash_dword(hash[1], 1);
}

/**
 * @brief Reads the updated MD5 hash state from the Octeon COP2 registers.
 * @details This function retrieves the 128-bit MD5 hash digest from the
 * Octeon cryptographic co-processor (COP2) registers and stores its 64-bit
 * portions into the `octeon_md5_state` context. This updates the software-
 * maintained hash state after the hardware has processed a message block.
 *
 * @param ctx Pointer to the `octeon_md5_state` structure where the hash digest will be stored.
 * Functional Utility: Retrieves the hardware-computed MD5 hash state from the Octeon COP2.
 */
static void octeon_md5_read_hash(struct octeon_md5_state *ctx)
{
	u64 *hash = (u64 *)ctx->hash;

	// Functional Utility: Reads the first 64-bit part of the hash digest from COP2 register 0.
	hash[0] = read_octeon_64bit_hash_dword(0);
	// Functional Utility: Reads the second 64-bit part of the hash digest from COP2 register 1.
	hash[1] = read_octeon_64bit_hash_dword(1);
}

/**
 * @brief Processes a single 64-byte MD5 block using the Octeon COP2 hardware.
 * @details This function takes a 64-byte message block, loads its 64-bit words
 * into the Octeon cryptographic co-processor (COP2) registers, and then
 * triggers the hardware-accelerated MD5 compression round. This offloads the
 * intensive MD5 computation for a full block to dedicated hardware.
 *
 * @param _block Pointer to the 64-byte (8x u64) message block to be processed.
 * Functional Utility: Performs hardware-accelerated MD5 compression for a single 64-byte block using Octeon COP2.
 */
static void octeon_md5_transform(const void *_block)
{
	const u64 *block = _block;

	// Functional Utility: Loads individual 64-bit words of the message block into the COP2 unit.
	write_octeon_64bit_block_dword(block[0], 0);
	write_octeon_64bit_block_dword(block[1], 1);
	write_octeon_64bit_block_dword(block[2], 2);
	write_octeon_64bit_block_dword(block[3], 3);
	write_octeon_64bit_block_dword(block[4], 4);
	write_octeon_64bit_block_dword(block[5], 5);
	write_octeon_64bit_block_dword(block[6], 6);
	// Functional Utility: Loads the final 64-bit word of the message block and triggers the hardware MD5 compression.
	octeon_md5_start(block[7]);
}

/**
 * @brief Initializes the MD5 hash state for a new computation.
 * @details This function prepares the `octeon_md5_state` context for a new
 * MD5 hashing operation. It initializes the 128-bit hash digest with the
 * standard MD5 Initial Vector (IV) values (`MD5_H0` through `MD5_H3`) and
 * resets the total `byte_count` to zero.
 *
 * @param desc The `shash_desc` descriptor for the current MD5 operation.
 * @return 0 on success.
 * Functional Utility: Sets up the initial state for an MD5 hashing operation.
 */
static int octeon_md5_init(struct shash_desc *desc)
{
	struct octeon_md5_state *mctx = shash_desc_ctx(desc);

	// Functional Utility: Initializes the MD5 hash digest with the standard MD5 Initial Vector (IV) values.
	mctx->hash[0] = cpu_to_le32(MD5_H0);
	mctx->hash[1] = cpu_to_le32(MD5_H1);
	mctx->hash[2] = cpu_to_le32(MD5_H2);
	mctx->hash[3] = cpu_to_le32(MD5_H3);
	// Functional Utility: Resets the total byte count for the new hashing operation.
	mctx->byte_count = 0;

	return 0;
}

/**
 * @brief Updates the MD5 hash with new data.
 * @details This function processes input data incrementally, feeding full
 * 64-byte blocks to the Octeon COP2 for hardware-accelerated MD5 compression.
 * It carefully manages the COP2 context: enabling it before hardware access
 * and disabling it afterward to prevent conflicts and ensure correct state
 * restoration.
 *
 * @param desc The `shash_desc` descriptor for the current MD5 operation.
 * @param data Pointer to the input data.
 * @param len  The length of the input data.
 * @return The number of bytes not processed (due to being a partial block).
 * Functional Utility: Accumulates new data into the MD5 hash state, utilizing Octeon COP2 for hardware acceleration on full blocks.
 */
static int octeon_md5_update(struct shash_desc *desc, const u8 *data,
			     unsigned int len)
{
	struct octeon_md5_state *mctx = shash_desc_ctx(desc);
	struct octeon_cop2_state state;
	unsigned long flags;

	// Functional Utility: Updates the total byte count with the length of the new input data.
	mctx->byte_count += len;
	// Functional Utility: Enables kernel access to the Octeon COP2, saving its previous state.
	flags = octeon_crypto_enable(&state);
	// Functional Utility: Transfers the current MD5 hash state from software to the COP2 hardware.
	octeon_md5_store_hash(mctx);

	// Block Logic: Processes full 64-byte blocks of data using the hardware accelerator.
	do {
		// Functional Utility: Triggers hardware-accelerated MD5 compression for a 64-byte block.
		octeon_md5_transform(data);
		data += MD5_HMAC_BLOCK_SIZE;
		len -= MD5_HMAC_BLOCK_SIZE;
	} while (len >= MD5_HMAC_BLOCK_SIZE);

	// Functional Utility: Reads the updated MD5 hash state from the COP2 hardware back into software.
	octeon_md5_read_hash(mctx);
	// Functional Utility: Disables kernel access to the Octeon COP2 and restores its previous state.
	octeon_crypto_disable(&state, flags);
	// Functional Utility: Adjusts the byte count to exclude any remaining partial block data.
	mctx->byte_count -= len;
	// Functional Utility: Returns the length of any unprocessed partial block data.
	return len;
}

/**
 * @brief Finalizes the MD5 hash computation, including padding and length appending.
 * @details This function handles the final steps of the MD5 algorithm. It applies
 * the necessary padding to the last data block, appends the total message length,
 * and then processes this final block using the Octeon COP2 for hardware-
 * accelerated compression. It also ensures proper COP2 context management during
 * this critical phase.
 *
 * @param desc The `shash_desc` descriptor for the current MD5 operation.
 * @param src Pointer to any remaining partial input data.
 * @param offset The length of the remaining partial input data.
 * @param out The buffer to store the final 16-byte MD5 hash digest.
 * @return 0 on success.
 * Functional Utility: Completes the MD5 hashing process, including padding and length encoding, using Octeon COP2 hardware.
 */
static int octeon_md5_finup(struct shash_desc *desc, const u8 *src,
			    unsigned int offset, u8 *out)
{
	struct octeon_md5_state *mctx = shash_desc_ctx(desc);
	// Functional Utility: Calculates the amount of padding needed for the last block.
	int padding = 56 - (offset + 1);
	struct octeon_cop2_state state;
	u32 block[MD5_BLOCK_WORDS]; // Functional Utility: Buffer to construct the final 64-byte MD5 block.
	unsigned long flags;
	char *p;

	// Functional Utility: Copies any remaining partial data into the beginning of the `block` buffer.
	p = memcpy(block, src, offset);
	// Functional Utility: Advances the pointer `p` past the copied data.
	p += offset;
	// Functional Utility: Appends the mandatory '1' bit (0x80) after the message data.
	*p++ = 0x80;

	// Functional Utility: Enables kernel access to the Octeon COP2, saving its previous state.
	flags = octeon_crypto_enable(&state);
	// Functional Utility: Transfers the current MD5 hash state from software to the COP2 hardware.
	octeon_md5_store_hash(mctx);

	// Block Logic: If the remaining data plus the '1' bit and length requires more than one block,
	// process the current (incomplete) block with zero padding.
	if (padding < 0) {
		// Functional Utility: Fills the rest of the current block with zeros.
		memset(p, 0x00, padding + sizeof(u64));
		// Functional Utility: Processes the incomplete block using hardware acceleration.
		octeon_md5_transform(block);
		p = (char *)block; // Functional Utility: Resets pointer to the beginning of the block.
		padding = 56; // Functional Utility: Resets padding for the next full block.
	}

	// Functional Utility: Fills the remaining part of the block with zeros for padding.
	memset(p, 0, padding);
	// Functional Utility: Updates the total byte count with the length of the remaining partial data.
	mctx->byte_count += offset;
	// Functional Utility: Appends the total message length in bits (big-endian) to the last two 64-bit words of the block.
	block[14] = mctx->byte_count << 3;
	block[15] = mctx->byte_count >> 29;
	// Functional Utility: Ensures the length is in little-endian format (MD5 internal representation).
	cpu_to_le32_array(block + 14, 2);
	// Functional Utility: Processes the final padded block using hardware acceleration.
	octeon_md5_transform(block);

	// Functional Utility: Reads the final MD5 hash state from the COP2 hardware back into software.
	octeon_md5_read_hash(mctx);
	// Functional Utility: Disables kernel access to the Octeon COP2 and restores its previous state.
	octeon_crypto_disable(&state, flags);

	// Functional Utility: Securely clears the temporary block buffer for security reasons.
	memzero_explicit(block, sizeof(block));
	// Functional Utility: Copies the final 16-byte MD5 hash digest to the output buffer.
	memcpy(out, mctx->hash, sizeof(mctx->hash));

	return 0;
}

/**
 * @brief Exports the current MD5 state.
 * @details This function is part of the Linux crypto API's state management.
 * It exports the current MD5 hash context (`octeon_md5_state`) into a
 * contiguous memory buffer, allowing the state to be saved and later restored.
 * The hash digest and byte count are included in the exported state.
 *
 * @param desc The `shash_desc` descriptor.
 * @param out Pointer to the output buffer where the state will be exported.
 * @return 0 on success.
 * Functional Utility: Saves the current MD5 hashing state into an external buffer.
 */
static int octeon_md5_export(struct shash_desc *desc, void *out)
{
	struct octeon_md5_state *ctx = shash_desc_ctx(desc);
	union {
		u8 *u8;
		u32 *u32;
		u64 *u64;
	} p = { .u8 = out };
	int i;

	// Functional Utility: Exports the 128-bit hash digest, converting from little-endian to CPU native format.
	for (i = 0; i < MD5_HASH_WORDS; i++)
		put_unaligned(le32_to_cpu(ctx->hash[i]), p.u32++);
	// Functional Utility: Exports the total byte count.
	put_unaligned(ctx->byte_count, p.u64);
	return 0;
}

/**
 * @brief Imports an MD5 state.
 * @details This function is part of the Linux crypto API's state management.
 * It imports a previously saved MD5 hash context from a memory buffer into
 * the `octeon_md5_state` structure, allowing a hashing operation to resume
 * from a specific point.
 *
 * @param desc The `shash_desc` descriptor.
 * @param in Pointer to the input buffer containing the saved state.
 * @return 0 on success.
 * Functional Utility: Restores a previously saved MD5 hashing state from an external buffer.
 */
static int octeon_md5_import(struct shash_desc *desc, const void *in)
{
	struct octeon_md5_state *ctx = shash_desc_ctx(desc);
	union {
		const u8 *u8;
		const u32 *u32;
		const u64 *u64;
	} p = { .u8 = in };
	int i;

	// Functional Utility: Imports the 128-bit hash digest, converting from CPU native to little-endian format.
	for (i = 0; i < MD5_HASH_WORDS; i++)
		ctx->hash[i] = cpu_to_le32(get_unaligned(p.u32++));
	// Functional Utility: Imports the total byte count.
	ctx->byte_count = get_unaligned(p.u64);
	return 0;
}

/**
 * @brief Defines the Octeon hardware-accelerated MD5 algorithm for the crypto API.
 * @details This structure registers the MD5 algorithm implementation that leverages
 * the Cavium Octeon's cryptographic co-processor. It specifies the algorithm's
 * properties (digest size, block size), associates the core operations (`init`,
 * `update`, `finup`, `export`, `import`) with their respective handler functions,
 * and sets a high priority (`OCTEON_CR_OPCODE_PRIORITY`) to ensure this hardware-
 * accelerated version is preferred over generic software implementations.
 * Functional Utility: Registers the Octeon hardware-accelerated MD5 algorithm with the kernel crypto API.
 */
static struct shash_alg alg = {
	.digestsize	=	MD5_DIGEST_SIZE,
	.init		=	octeon_md5_init,
	.update		=	octeon_md5_update,
	.finup		=	octeon_md5_finup,
	.export		=	octeon_md5_export,
	.import		=	octeon_md5_import,
	.statesize	=	MD5_STATE_SIZE,
	.descsize	=	sizeof(struct octeon_md5_state),
	.base		=	{
		.cra_name	=	"md5",
		.cra_driver_name=	"octeon-md5",
		.cra_priority	=	OCTEON_CR_OPCODE_PRIORITY,
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY,
		.cra_blocksize	=	MD5_HMAC_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
};

/**
 * @brief Module initialization function for Octeon MD5.
 * @details This function is the entry point when the kernel module is loaded.
 * It first checks if the Octeon CPU supports cryptographic operations via
 * `octeon_has_crypto()`. If hardware crypto is available, it registers the
 * hardware-accelerated MD5 algorithm (`alg` structure) with the Linux kernel's
 * cryptographic API. This makes the `octeon-md5` driver available to the system.
 * Functional Utility: Registers the hardware-accelerated Octeon MD5 algorithm with the kernel crypto API after checking for hardware support.
 * @return 0 on successful registration, or `-ENOTSUPP` if Octeon crypto hardware is not available, or other error code on failure.
 */
static int __init md5_mod_init(void)
{
	// Pre-condition: Checks if the Octeon CPU has cryptographic hardware support.
	if (!octeon_has_crypto())
		return -ENOTSUPP; // Functional Utility: Returns an error if crypto hardware is not supported.
	// Functional Utility: Registers the Octeon hardware-accelerated MD5 algorithm with the kernel crypto API.
	return crypto_register_shash(&alg);
}

/**
 * @brief Module cleanup function for Octeon MD5.
 * @details This function is the exit point when the kernel module is unloaded.
 * Functional Utility: Unregisters the hardware-accelerated MD5 algorithm (`alg`
 * structure) from the Linux kernel's cryptographic API. This cleanly removes
 * the `octeon-md5` driver from the system, releasing associated resources
 * and preventing any lingering references after the module is no longer in use.
 * This ensures proper resource management upon module unload.
 * Functional Utility: Unregisters the hardware-accelerated Octeon MD5 algorithm from the kernel crypto API.
 */
static void __exit md5_mod_fini(void)
{
	crypto_unregister_shash(&alg);
}

module_init(md5_mod_init);
module_exit(md5_mod_fini);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("MD5 Message Digest Algorithm (OCTEON)");
MODULE_AUTHOR("Aaro Koskinen <aaro.koskinen@iki.fi>");
