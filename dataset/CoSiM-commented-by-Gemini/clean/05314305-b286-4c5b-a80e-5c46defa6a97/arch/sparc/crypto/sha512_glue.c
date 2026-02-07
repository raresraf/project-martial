// SPDX-License-Identifier: GPL-2.0-only
/**
 * @file sha512_glue.c
 * @brief Glue code for SHA-512 and SHA-384 hashing optimized for SPARC64 crypto opcodes.
 * @details This file provides the C-level interface (glue code) to integrate
 * highly optimized SHA-512 and SHA-384 implementations, which leverage
 * SPARC64 crypto opcodes, with the Linux kernel's cryptographic API (`shash`).
 * Its primary responsibilities include:
 * - **Hardware Capability Detection**: Dynamically checking for the presence of
 *   SPARC64 cryptographic opcode support (specifically for SHA-512) at module
 *   initialization time.
 * - **Integration with Crypto API**: Providing the standard `update` and `finup`
 *   operations required by the `shash` API, delegating the core work to the
 *   SPARC64 assembly-optimized `sha512_sparc64_transform` routine.
 * - **Conditional Registration**: Registering the accelerated SHA-512 and SHA-384
 *   algorithms only if the necessary hardware capabilities are detected, ensuring
 *   efficient resource utilization and stability.
 * This implementation aims to provide a high-performance SHA-512/384 solution
 * for SPARC64 systems with cryptographic opcode capabilities.
 */
/* Glue code for SHA512 hashing optimized for sparc64 crypto opcodes.
 *
 * This is based largely upon crypto/sha512_generic.c
 *
 * Copyright (c) Jean-Luc Cooke <jlcooke@certainkey.com>
 * Copyright (c) Andrew McDonald <andrew@mcdonald.org.uk>
 * Copyright (c) 2003 Kyle McMartin <kyle@debian.org>
 */

#define pr_fmt(fmt)	KBUILD_MODNAME ": " fmt

#include <asm/elf.h>
#include <asm/opcodes.h>
#include <asm/pstate.h>
#include <crypto/internal/hash.h>
#include <crypto/sha2.h>
#include <crypto/sha512_base.h>
#include <linux/kernel.h>
#include <linux/module.h>

/**
 * @brief External declaration for the SPARC64 assembly SHA-512/384 transform function.
 * @details This function is an `asmlinkage` declaration, indicating that its
 * implementation is provided in a separate SPARC64 assembly file (e.g., `sha512_asm.S`).
 * It performs the core SHA-512/384 block transformation for a 128-byte message block,
 * leveraging SPARC64 crypto opcodes and VIS (Visual Instruction Set) extensions.
 *
 * @param digest Pointer to the SHA-512 state array (8x u64 words).
 * @param data Pointer to the 128-byte input data block.
 * @param rounds The number of 128-byte blocks to process.
 * Functional Utility: Dispatches to the SPARC64 assembly routine for hardware-accelerated SHA-512/384 block processing.
 */
asmlinkage void sha512_sparc64_transform(u64 *digest, const char *data,
					 unsigned int rounds);

/**
 * @brief Wrapper for the SPARC64 assembly SHA-512/384 transform function.
 * @details This function provides a simple C-language wrapper that passes the
 * SHA-512/384 state and input data to the `sha512_sparc64_transform` assembly routine.
 * It handles the iterative processing of multiple 128-byte blocks.
 *
 * @param sctx Pointer to the SHA-512 state structure.
 * @param src Pointer to the input data.
 * @param blocks The number of 128-byte blocks to process.
 * Functional Utility: Provides a C-language interface to the SPARC64 assembly SHA-512/384 block transformation.
 */
static void sha512_block(struct sha512_state *sctx, const u8 *src, int blocks)
{
	// Functional Utility: Invokes the SPARC64 assembly routine to transform multiple SHA-512/384 blocks.
	sha512_sparc64_transform(sctx->state, src, blocks);
}

/**
 * @brief Implements the `shash` 'update' operation for SPARC64 SHA-512/384.
 * @details This function integrates the SPARC64-accelerated SHA-512/384 block
 * processing into the Linux kernel's generic `shash` API for incremental updates.
 * It acts as a straightforward wrapper that delegates data buffering and partial
 * block processing to the `sha512_base_do_update_blocks` helper function, providing
 * `sha512_block` as the callback for core block processing.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return 0 on success.
 * Functional Utility: Updates the SHA-512/384 hash state incrementally, leveraging SPARC64 assembly for core block processing.
 */
static int sha512_sparc64_update(struct shash_desc *desc, const u8 *data,
				 unsigned int len)
{
	// Functional Utility: Delegates data buffering and block processing to the base SHA-512 update function,
	// using the SPARC64-specific `sha512_block` as the core block transformation.
	return sha512_base_do_update_blocks(desc, data, len, sha512_block);
}

/**
 * @brief Finalizes the SHA-512/384 hash computation for SPARC64.
 * @details This function handles the final steps of the SHA-512/384 algorithm.
 * It applies the necessary padding to the last data block, appends the total
 * message length, and then processes this final block using the SPARC64 assembly
 * transformation. It utilizes the `sha512_base_do_finup` and `sha512_base_finish`
 * helper functions, providing `sha512_block` for the core block processing.
 *
 * @param desc The `shash_desc` descriptor.
 * @param src Pointer to any remaining partial input data.
 * @param len The length of the remaining partial input data.
 * @param out The buffer to store the final SHA-512/384 hash digest.
 * @return 0 on success.
 * Functional Utility: Completes the SHA-512/384 hashing process, including padding and length encoding, using SPARC64 assembly.
 */
static int sha512_sparc64_finup(struct shash_desc *desc, const u8 *src,
				unsigned int len, u8 *out)
{
	// Functional Utility: Processes any remaining input data and applies SHA-512/384 padding using the SPARC64 assembly transformation.
	sha512_base_do_finup(desc, src, len, sha512_block);
	// Functional Utility: Writes the final computed SHA-512/384 hash digest to the output buffer.
	return sha512_base_finish(desc, out);
}

/**
 * @brief Defines the SPARC64 hardware-accelerated SHA-512 algorithm for the crypto API.
 * @details This structure registers the SHA-512 algorithm implementation that leverages
 * SPARC64 crypto opcodes. It specifies the algorithm's properties (digest size,
 * block size), associates the core operations (`init`, `update`, `finup`)
 * with their respective handler functions. A high `cra_priority` (`SPARC_CR_OPCODE_PRIORITY`)
 * ensures this hardware-accelerated version is preferred over generic software implementations.
 * Functional Utility: Registers the SPARC64 hardware-accelerated SHA-512 algorithm with the kernel crypto API.
 */
static struct shash_alg sha512 = {
	.digestsize	=	SHA512_DIGEST_SIZE,
	.init		=	sha512_base_init,
	.update		=	sha512_sparc64_update,
	.finup		=	sha512_sparc64_finup,
	.descsize	=	SHA512_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha512",
		.cra_driver_name=	"sha512-sparc64",
		.cra_priority	=	SPARC_CR_OPCODE_PRIORITY, // Functional Utility: Sets high priority for SPARC64 crypto opcode acceleration.
		.cra_blocksize	=	SHA512_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
};

/**
 * @brief Defines the SPARC64 hardware-accelerated SHA-384 algorithm for the crypto API.
 * @details This structure registers the SHA-384 algorithm implementation that leverages
 * SPARC64 crypto opcodes. It specifies the algorithm's properties (digest size,
 * block size), associates the core operations (`init`, `update`, `finup`)
 * with their respective handler functions. A high `cra_priority` (`SPARC_CR_OPCODE_PRIORITY`)
 * ensures this hardware-accelerated version is preferred over generic software implementations.
 * Functional Utility: Registers the SPARC64 hardware-accelerated SHA-384 algorithm with the kernel crypto API.
 */
static struct shash_alg sha384 = {
	.digestsize	=	SHA384_DIGEST_SIZE,
	.init		=	sha384_base_init,
	.update		=	sha512_sparc64_update,
	.finup		=	sha512_sparc64_finup,
	.descsize	=	SHA512_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha384",
		.cra_driver_name=	"sha384-sparc64",
		.cra_priority	=	SPARC_CR_OPCODE_PRIORITY, // Functional Utility: Sets high priority for SPARC64 crypto opcode acceleration.
		.cra_blocksize	=	SHA384_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
};

/**
 * @brief Checks for the presence of SPARC64 SHA-512 cryptographic opcode support.
 * @details This function determines whether the current SPARC64 CPU provides
 * hardware support for cryptographic operations, specifically for SHA-512.
 * It queries the CPU's hardware capabilities (`sparc64_elf_hwcap`) for
 * `HWCAP_SPARC_CRYPTO` and then checks a specific Control Register (`ASR26`,
 * the Cryptographic Function Register, `CFR`) for the `CFR_SHA512` bit. This
 * ensures that the accelerated implementation is only used on compatible hardware.
 *
 * @return `true` if SPARC64 SHA-512 cryptographic opcode support is available, `false` otherwise.
 * Functional Utility: Detects the presence of dedicated SHA-512 hardware acceleration on SPARC64 CPUs.
 */
static bool __init sparc64_has_sha512_opcode(void)
{
	unsigned long cfr;

	// Pre-condition: Check if the CPU's ELF hardware capabilities include SPARC_CRYPTO.
	if (!(sparc64_elf_hwcap & HWCAP_SPARC_CRYPTO))
		return false;

	// Functional Utility: Reads the Cryptographic Function Register (CFR, ASR26).
	__asm__ __volatile__("rd %%asr26, %0" : "=r" (cfr));
	// Functional Utility: Checks if the CFR indicates support for SHA-512.
	if (!(cfr & CFR_SHA512))
		return false;

	return true;
}

/**
 * @brief Module initialization function for SPARC64 SHA-512/384.
 * @details This function is the entry point when the kernel module is loaded.
 * It first checks if the SPARC64 CPU supports cryptographic operations (specifically
 * SHA-512 opcodes) via `sparc64_has_sha512_opcode()`. If hardware support is available,
 * it registers both the hardware-accelerated SHA-512 (`sha512` structure) and
 * SHA-384 (`sha384` structure) algorithms with the Linux kernel's cryptographic API.
 * This enables high-performance SHA-512/384 hashing on SPARC64 platforms.
 * Error handling is included to ensure that if SHA-512 registration fails,
 * any previously registered SHA-384 algorithm is unregistered for cleanup.
 * Functional Utility: Conditionally registers hardware-accelerated SPARC64 SHA-512 and SHA-384 algorithms with the kernel crypto API, based on hardware support.
 * @return 0 on successful registration of both algorithms, or `-ENODEV` if SPARC64
 *         SHA-512 crypto opcodes are not available, or another error code on failure.
 */
static int __init sha512_sparc64_mod_init(void)
{
	if (sparc64_has_sha512_opcode()) {
		int ret = crypto_register_shash(&sha384);
		if (ret < 0)
			return ret;

		ret = crypto_register_shash(&sha512);
		if (ret < 0) {
			crypto_unregister_shash(&sha384); // Functional Utility: Unregisters SHA-384 if SHA-512 registration fails.
			return ret;
		}

		pr_info("Using sparc64 sha512 opcode optimized SHA-512/SHA-384 implementation
");
		return 0;
	}
	pr_info("sparc64 sha512 opcode not available.
");
	return -ENODEV;
}

/**
 * @brief Module cleanup function for SPARC64 SHA-512/384.
 * @details This function is the exit point when the kernel module is unloaded.
 * Functional Utility: Unregisters both the hardware-accelerated SHA-512 and
 * SHA-384 algorithms (`sha512` and `sha384` structures) from the Linux kernel's
 * cryptographic API. This cleanly removes the drivers from the system,
 * releasing associated resources and preventing any lingering references
 * after the module is no longer in use. This ensures proper resource
 * management upon module unload.
 * Functional Utility: Unregisters the hardware-accelerated SPARC64 SHA-512 and SHA-384 algorithms from the kernel crypto API.
 */
static void __exit sha512_sparc64_mod_fini(void)
{
	crypto_unregister_shash(&sha384);
	crypto_unregister_shash(&sha512);
}

module_init(sha512_sparc64_mod_init);
module_exit(sha512_sparc64_mod_fini);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SHA-384 and SHA-512 Secure Hash Algorithm, sparc64 sha512 opcode accelerated");

MODULE_ALIAS_CRYPTO("sha384");
MODULE_ALIAS_CRYPTO("sha512");

/**
 * @brief Includes device ID cropping functionality, which is part of the SPARC crypto subsystem.
 * @details This line includes a separate C file that likely provides helper
 * functions or definitions related to device ID cropping, which is an aspect
 * of managing cryptographic devices on SPARC platforms. This modular approach
 * helps keep related functionalities organized.
 * Functional Utility: Incorporates SPARC-specific device ID cropping logic.
 */
#include "crop_devid.c"
