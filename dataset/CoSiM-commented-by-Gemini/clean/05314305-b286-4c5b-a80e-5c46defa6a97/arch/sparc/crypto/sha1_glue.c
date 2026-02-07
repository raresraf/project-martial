// SPDX-License-Identifier: GPL-2.0-only
/**
 * @file sha1_glue.c
 * @brief Glue code for SHA-1 hashing optimized for SPARC64 cryptographic opcodes.
 * @details This file provides the C-level interface (glue code) to integrate
 * the highly optimized SHA-1 (Secure Hash Algorithm 1) implementation,
 * which leverages SPARC64 crypto opcodes, with the Linux kernel's cryptographic
 * API (`shash`). Its primary responsibilities include:
 * - **Hardware Capability Detection**: Dynamically checking for the presence of
 *   SPARC64 cryptographic opcode support (specifically for SHA-1) at module
 *   initialization time.
 * - **Integration with Crypto API**: Providing the standard `update` and `finup`
 *   operations required by the `shash` API, delegating the core work to the
 *   SPARC64 assembly-optimized `sha1_sparc64_transform` routine.
 * - **Conditional Registration**: Registering the accelerated SHA-1 algorithm
 *   only if the necessary hardware capabilities are detected, ensuring efficient
 *   resource utilization and stability.
 * This implementation aims to provide a high-performance SHA-1 solution for
 * SPARC64 systems with cryptographic opcode capabilities.
 */
/* Glue code for SHA1 hashing optimized for sparc64 crypto opcodes.
 *
 * This is based largely upon arch/x86/crypto/sha1_ssse3_glue.c
 *
 * Copyright (c) Alan Smithee.
 * Copyright (c) Andrew McDonald <andrew@mcdonald.org.uk>
 * Copyright (c) Jean-Francois Dive <jef@linuxbe.org>
 * Copyright (c) Mathias Krause <minipli@googlemail.com>
 */

#define pr_fmt(fmt)	KBUILD_MODNAME ": " fmt

#include <asm/elf.h>
#include <asm/opcodes.h>
#include <asm/pstate.h>
#include <crypto/internal/hash.h>
#include <crypto/sha1.h>
#include <crypto/sha1_base.h>
#include <linux/kernel.h>
#include <linux/module.h>

/**
 * @brief External declaration for the SPARC64 assembly SHA-1 transform function.
 * @details This function is an `asmlinkage` declaration, indicating that its
 * implementation is provided in a separate SPARC64 assembly file (e.g., `sha1_asm.S`).
 * It performs the core SHA-1 block transformation for a 64-byte message block,
 * leveraging SPARC64 crypto opcodes and VIS (Visual Instruction Set) extensions.
 *
 * @param digest Pointer to the SHA-1 state array (5x u32 words).
 * @param data Pointer to the input data block (64-bytes).
 * @param rounds The number of 64-byte blocks to process.
 * Functional Utility: Dispatches to the SPARC64 assembly routine for hardware-accelerated SHA-1 block processing.
 */
asmlinkage void sha1_sparc64_transform(struct sha1_state *digest,
				       const u8 *data, int rounds);

/**
 * @brief Implements the `shash` 'update' operation for SPARC64 SHA-1.
 * @details This function integrates the SPARC64-accelerated SHA-1 block
 * processing into the Linux kernel's generic `shash` API for incremental updates.
 * It acts as a straightforward wrapper that delegates data buffering and partial
 * block processing to the `sha1_base_do_update_blocks` helper function, providing
 * `sha1_sparc64_transform` as the callback for core block processing.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return 0 on success.
 * Functional Utility: Updates the SHA-1 hash state incrementally, leveraging SPARC64 assembly for core block processing.
 */
static int sha1_sparc64_update(struct shash_desc *desc, const u8 *data,
			       unsigned int len)
{
	// Functional Utility: Delegates data buffering and block processing to the base SHA-1 update function,
	// using the SPARC64-specific `sha1_sparc64_transform` as the core block transformation.
	return sha1_base_do_update_blocks(desc, data, len,
					  sha1_sparc64_transform);
}

/**
 * @brief Finalizes the SHA-1 hash computation for SPARC64.
 * @details This function handles the final steps of the SHA-1 algorithm. It applies
 * the necessary padding to the last data block, appends the total message length,
 * and then processes this final block using the SPARC64 assembly transformation.
 * It utilizes the `sha1_base_do_finup` and `sha1_base_finish` helper functions,
 * providing `sha1_sparc64_transform` for the core block processing.
 *
 * @param desc The `shash_desc` descriptor.
 * @param src Pointer to any remaining partial input data.
 * @param len The length of the remaining partial input data.
 * @param out The buffer to store the final 20-byte SHA-1 hash digest.
 * @return 0 on success.
 * Functional Utility: Completes the SHA-1 hashing process, including padding and length encoding, using SPARC64 assembly.
 */
/* Add padding and return the message digest. */
static int sha1_sparc64_finup(struct shash_desc *desc, const u8 *src,
			      unsigned int len, u8 *out)
{
	// Functional Utility: Processes any remaining input data and applies SHA-1 padding using the SPARC64 assembly transformation.
	sha1_base_do_finup(desc, src, len, sha1_sparc64_transform);
	// Functional Utility: Writes the final computed SHA-1 hash digest to the output buffer.
	return sha1_base_finish(desc, out);
}

/**
 * @brief Defines the SPARC64 hardware-accelerated SHA-1 algorithm for the crypto API.
 * @details This structure registers the SHA-1 algorithm implementation that leverages
 * SPARC64 crypto opcodes. It specifies the algorithm's properties (digest size,
 * block size), associates the core operations (`init`, `update`, `finup`)
 * with their respective handler functions. A high `cra_priority` (`SPARC_CR_OPCODE_PRIORITY`)
 * ensures this hardware-accelerated version is preferred over generic software implementations.
 * Functional Utility: Registers the SPARC64 hardware-accelerated SHA-1 algorithm with the kernel crypto API.
 */
static struct shash_alg alg = {
	.digestsize	=	SHA1_DIGEST_SIZE,
	.init		=	sha1_base_init,
	.update		=	sha1_sparc64_update,
	.finup		=	sha1_sparc64_finup,
	.descsize	=	SHA1_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha1",
		.cra_driver_name=	"sha1-sparc64",
		.cra_priority	=	SPARC_CR_OPCODE_PRIORITY, // Functional Utility: Sets high priority for SPARC64 crypto opcode acceleration.
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY,
		.cra_blocksize	=	SHA1_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
};

/**
 * @brief Checks for the presence of SPARC64 SHA-1 cryptographic opcode support.
 * @details This function determines whether the current SPARC64 CPU provides
 * hardware support for cryptographic operations, specifically for SHA-1.
 * It queries the CPU's hardware capabilities (`sparc64_elf_hwcap`) for
 * `HWCAP_SPARC_CRYPTO` and then checks a specific Control Register (`ASR26`,
 * the Cryptographic Function Register, `CFR`) for the `CFR_SHA1` bit. This
 * ensures that the accelerated implementation is only used on compatible hardware.
 *
 * @return `true` if SPARC64 SHA-1 cryptographic opcode support is available, `false` otherwise.
 * Functional Utility: Detects the presence of dedicated SHA-1 hardware acceleration on SPARC64 CPUs.
 */
static bool __init sparc64_has_sha1_opcode(void)
{
	unsigned long cfr;

	// Pre-condition: Check if the CPU's ELF hardware capabilities include SPARC_CRYPTO.
	if (!(sparc64_elf_hwcap & HWCAP_SPARC_CRYPTO))
		return false;

	// Functional Utility: Reads the Cryptographic Function Register (CFR, ASR26).
	__asm__ __volatile__("rd %%asr26, %0" : "=r" (cfr));
	// Functional Utility: Checks if the CFR indicates support for SHA-1.
	if (!(cfr & CFR_SHA1))
		return false;

	return true;
}

/**
 * @brief Module initialization function for SPARC64 SHA-1.
 * @details This function is the entry point when the kernel module is loaded.
 * It first checks if the SPARC64 CPU supports cryptographic operations (specifically
 * SHA-1 opcodes) via `sparc64_has_sha1_opcode()`. If hardware support is available,
 * it registers the hardware-accelerated SHA-1 algorithm (`alg` structure) with
 * the Linux kernel's cryptographic API. This makes the `sha1-sparc64` driver
 * available to the system, enabling high-performance SHA-1 hashing.
 * Functional Utility: Conditionally registers the hardware-accelerated SPARC64 SHA-1 algorithm with the kernel crypto API, based on hardware support.
 * @return 0 on successful registration, or `-ENODEV` if SPARC64 SHA-1 crypto opcodes are not available.
 */
static int __init sha1_sparc64_mod_init(void)
{
	if (sparc64_has_sha1_opcode()) {
		pr_info("Using sparc64 sha1 opcode optimized SHA-1 implementation
");
		return crypto_register_shash(&alg);
	}
	pr_info("sparc64 sha1 opcode not available.
");
	return -ENODEV;
}

/**
 * @brief Module cleanup function for SPARC64 SHA-1.
 * @details This function is the exit point when the kernel module is unloaded.
 * Functional Utility: Unregisters the hardware-accelerated SHA-1 algorithm
 * (`alg` structure) from the Linux kernel's cryptographic API. This cleanly
 * removes the `sha1-sparc64` driver from the system, releasing associated resources
 * and preventing any lingering references after the module is no longer in use.
 * This ensures proper resource management upon module unload.
 * Functional Utility: Unregisters the hardware-accelerated SPARC64 SHA-1 algorithm from the kernel crypto API.
 */
static void __exit sha1_sparc64_mod_fini(void)
{
	crypto_unregister_shash(&alg);
}

module_init(sha1_sparc64_mod_init);
module_exit(sha1_sparc64_mod_fini);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SHA1 Secure Hash Algorithm, sparc64 sha1 opcode accelerated");

MODULE_ALIAS_CRYPTO("sha1");

// Functional Utility: Includes device ID cropping functionality, which is part of the SPARC crypto subsystem.
#include "crop_devid.c"
