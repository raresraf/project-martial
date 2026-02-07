// SPDX-License-Identifier: GPL-2.0-or-later
/**
 * @file octeon-sha256.c
 * @brief Hardware-accelerated SHA-256 Secure Hash Algorithm using Cavium Octeon's cryptographic co-processor.
 * @details This file provides an architecture-optimized block transformation
 * for the SHA-256 (Secure Hash Algorithm 256) hash function, specifically
 * adapted for Cavium Octeon processors. It leverages the Octeon's dedicated
 * cryptographic co-processor (COP2) to offload SHA-256 compression,
 * significantly improving performance over software-only implementations.
 * This implementation is designed to be used as `sha256_blocks_arch` within
 * the Linux kernel's crypto API, serving as a high-performance replacement
 * for generic SHA-256 block processing functions when Octeon crypto hardware
 * is available. It also provides a mechanism to check for the availability
 * of this hardware optimization.
 */
/*
 * SHA-256 Secure Hash Algorithm.
 *
 * Adapted for OCTEON by Aaro Koskinen <aaro.koskinen@iki.fi>.
 *
 * Based on crypto/sha256_generic.c, which is:
 *
 * Copyright (c) Jean-Luc Cooke <jlcooke@certainkey.com>
 * Copyright (c) Andrew McDonald <andrew@mcdonald.org.uk>
 * Copyright (c) 2002 James Morris <jmorris@intercode.com.au>
 * SHA224 Support Copyright 2007 Intel Corporation <jonathan.lynch@intel.com>
 */

#include <asm/octeon/octeon.h>
#include <crypto/internal/sha2.h>
#include <linux/kernel.h>
#include <linux/module.h>

#include "octeon-crypto.h"

/*
 * We pass everything as 64-bit. OCTEON can handle misaligned data.
 */

/**
 * @brief Performs hardware-accelerated SHA-256 block transformations using the Octeon COP2.
 * @details This function serves as the architecture-optimized block transformation
 * for SHA-256. It takes the current SHA-256 hash state, input data, and the
 * number of blocks, then processes them using the Cavium Octeon's cryptographic
 * co-processor (COP2). This provides a significant performance advantage over
 * software-only implementations. If the Octeon crypto hardware is not detected,
 * it gracefully falls back to the generic software SHA-256 block processing.
 *
 * @param state Pointer to the SHA-256 state array (8x u32 words).
 * @param data Pointer to the input data blocks (64-byte blocks).
 * @param nblocks The number of 64-byte blocks to process.
 * Functional Utility: Accelerates SHA-256 block processing using Octeon hardware, with a fallback to software if hardware is unavailable.
 */
void sha256_blocks_arch(u32 state[SHA256_STATE_WORDS],
			const u8 *data, size_t nblocks)
{
	struct octeon_cop2_state cop2_state;
	u64 *state64 = (u64 *)state; // Functional Utility: Type-casts the 32-bit state array to 64-bit for COP2 interaction.
	unsigned long flags;

	// Pre-condition: Checks if the Octeon CPU has cryptographic hardware support.
	if (!octeon_has_crypto())
		return sha256_blocks_generic(state, data, nblocks); // Functional Utility: Falls back to generic software implementation if crypto hardware is not present.

	// Functional Utility: Enables kernel access to the Octeon COP2, saving its previous state.
	flags = octeon_crypto_enable(&cop2_state);
	// Functional Utility: Transfers the current SHA-256 hash state from software to the COP2 hardware.
	write_octeon_64bit_hash_dword(state64[0], 0);
	write_octeon_64bit_hash_dword(state64[1], 1);
	write_octeon_64bit_hash_dword(state64[2], 2);
	write_octeon_64bit_hash_dword(state64[3], 3);

	// Block Logic: Processes full 64-byte blocks of data using the hardware accelerator.
	do {
		const u64 *block = (const u64 *)data;

		// Functional Utility: Loads individual 64-bit words of the current message block into the COP2 unit.
		write_octeon_64bit_block_dword(block[0], 0);
		write_octeon_64bit_block_dword(block[1], 1);
		write_octeon_64bit_block_dword(block[2], 2);
		write_octeon_64bit_block_dword(block[3], 3);
		write_octeon_64bit_block_dword(block[4], 4);
		write_octeon_64bit_block_dword(block[5], 5);
		write_octeon_64bit_block_dword(block[6], 6);
		// Functional Utility: Loads the final 64-bit word of the message block and triggers the hardware SHA-256 compression.
		octeon_sha256_start(block[7]);

		// Functional Utility: Advances the data pointer to the next 64-byte block.
		data += SHA256_BLOCK_SIZE;
	} while (--nblocks); // Functional Utility: Continues looping until all specified blocks are processed.

	// Functional Utility: Reads the updated SHA-256 hash state from the COP2 hardware back into software.
	state64[0] = read_octeon_64bit_hash_dword(0);
	state64[1] = read_octeon_64bit_hash_dword(1);
	state64[2] = read_octeon_64bit_hash_dword(2);
	state64[3] = read_octeon_64bit_hash_dword(3);
	// Functional Utility: Disables kernel access to the Octeon COP2 and restores its previous state.
	octeon_crypto_disable(&cop2_state, flags);
}
EXPORT_SYMBOL_GPL(sha256_blocks_arch);

/**
 * @brief Checks if SHA-256 processing is optimized for the current architecture (Octeon).
 * @details This function determines whether the current Octeon CPU provides
 * hardware support for cryptographic operations, specifically for SHA-256.
 * It is used by the crypto API to decide whether to utilize the architecture-
 * optimized SHA-256 block transformation or to fall back to a generic
 * software implementation.
 *
 * @return `true` if Octeon crypto hardware is available and optimized SHA-256
 *         is supported, `false` otherwise.
 * Functional Utility: Reports the availability of hardware-optimized SHA-256 support on Octeon.
 */
bool sha256_is_arch_optimized(void)
{
	// Functional Utility: Queries the Octeon hardware to determine if cryptographic capabilities are present.
	return octeon_has_crypto();
}
EXPORT_SYMBOL_GPL(sha256_is_arch_optimized);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SHA-256 Secure Hash Algorithm (OCTEON)");
MODULE_AUTHOR("Aaro Koskinen <aaro.koskinen@iki.fi>");
