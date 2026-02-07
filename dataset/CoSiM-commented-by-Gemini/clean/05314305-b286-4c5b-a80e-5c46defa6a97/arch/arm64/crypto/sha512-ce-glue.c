/**
 * @file sha512-ce-glue.c
 * @brief Glue for ARMv8 Crypto Extensions SHA-384/SHA-512 for AArch64.
 * @details This file provides the C-level glue code to integrate the ARMv8
 * Cryptography Extensions (CE) accelerated SHA-512 and SHA-384 implementations
 * with the Linux kernel's cryptographic API on AArch64 systems. Its main
 * responsibilities include:
 * - **Registering Hardware-Accelerated Algorithms**: Making the highly optimized
 *   CE implementations available to the kernel.
 * - **Safe NEON/FPU Context Management**: Ensuring the Floating-Point Unit (FPU)
 *   state is correctly saved and restored around calls to the assembly code
 *   to prevent corruption and maintain system stability.
 * - **Delegating Core Transformations**: Wrapping the calls to the dedicated
 *   assembly routines (`sha512-ce-core.S`) that perform the actual hardware-
 *   accelerated SHA-512/384 compression.
 * This file is crucial for enabling high-performance cryptographic operations
 * on ARM64 platforms with Crypto Extensions.
 */
// SPDX-License-Identifier: GPL-2.0
/*
 * sha512-ce-glue.c - SHA-384/SHA-512 using ARMv8 Crypto Extensions
 *
 * Copyright (C) 2018 Linaro Ltd <ard.biesheuvel@linaro.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 */

#include <asm/neon.h>
#include <crypto/internal/hash.h>
#include <crypto/sha2.h>
#include <crypto/sha512_base.h>
#include <linux/cpufeature.h>
#include <linux/kernel.h>
#include <linux/module.h>

MODULE_DESCRIPTION("SHA-384/SHA-512 secure hash using ARMv8 Crypto Extensions");
MODULE_AUTHOR("Ard Biesheuvel <ard.biesheuvel@linaro.org>");
MODULE_LICENSE("GPL v2");
MODULE_ALIAS_CRYPTO("sha384");
MODULE_ALIAS_CRYPTO("sha512");

/**
 * @brief Entry point for the ARMv8 CE assembly implementation of SHA-512/384.
 * @details This function serves as the direct interface to the highly optimized
 * ARMv8-A Cryptography Extensions (CE) assembly implementation of the SHA-512
 * compression function, defined in `sha512-ce-core.S`. It utilizes dedicated
 * hardware instructions to perform the complex 64-bit round updates and message
 * schedule calculations, significantly accelerating the cryptographic process.
 *
 * @param sst    Pointer to the SHA-512 state structure.
 * @param src    Pointer to the source data.
 * @param blocks Number of 128-byte blocks to process.
 * @return       Number of blocks remaining.
 * Functional Utility: Dispatches to the hardware-accelerated ARM64 CE assembly routine for high-performance SHA-512/384 block processing.
 */
asmlinkage int __sha512_ce_transform(struct sha512_state *sst, u8 const *src,
				     int blocks);

/**
 * @brief C wrapper for the ARMv8 CE SHA-512/384 assembly transform function.
 * @details This function provides a safe C-language wrapper for iteratively
 * calling the hardware-accelerated `__sha512_ce_transform` assembly routine.
 * Its primary purpose is to manage the NEON/FPU context around these calls.
 * For each block or set of blocks to be processed, `kernel_neon_begin()` is
 * invoked to save the FPU state, and `kernel_neon_end()` is called afterward
 * to restore it. This iterative and safe context management ensures proper
 * functioning and prevents FPU state corruption during high-performance
 * cryptographic operations within the kernel.
 *
 * @param sst    Pointer to the standard SHA-512 state.
 * @param src    Pointer to the source data.
 * @param blocks Number of blocks to process.
 * Functional Utility: Manages NEON/FPU context around hardware-accelerated SHA-512/384 assembly calls for safe and iterative block processing.
 */
static void sha512_ce_transform(struct sha512_state *sst, u8 const *src,
				int blocks)
{
	// Block Logic: Iterates over the given number of blocks, calling the hardware-accelerated
	// transform for each, while carefully managing the NEON FPU context.
	// Invariant: At the start of each iteration, `blocks` holds the number of remaining blocks,
	// and `src` points to the current data to be processed.
	do {
		int rem;

		// Pre-condition: Saves the current FPU state and enables NEON operations within the kernel context.
		// Invariant: FPU state is preserved across the __sha512_ce_transform call.
		kernel_neon_begin();
		// Functional Utility: Invokes the hardware-accelerated SHA-512/384 block transform,
		// processing a portion of the input data.
		rem = __sha512_ce_transform(sst, src, blocks);
		// Post-condition: Restores the FPU state, making it safe for other kernel components.
		kernel_neon_end();
		// Functional Utility: Adjusts the source pointer and remaining block count based on blocks processed by the assembly.
		src += (blocks - rem) * SHA512_BLOCK_SIZE;
		blocks = rem;
	} while (blocks); // Functional Utility: Loop continues until all blocks have been processed (blocks becomes 0).
}

/**
 * @brief Implements the shash 'update' operation using ARMv8 CE.
 * @details This function integrates the ARMv8 CE accelerated SHA-512/384 into the
 * kernel's generic `shash` API for incremental updates. It acts as a straightforward
 * wrapper that delegates data buffering and partial block processing to the
 * `sha512_base_do_update_blocks` helper function, providing the hardware-accelerated
 * `sha512_ce_transform` as the core block transformation callback.
 *
 * @param desc The shash descriptor.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return 0 on success.
 * Functional Utility: Updates the SHA-512/384 hash state incrementally, leveraging ARMv8 CE for core block processing.
 */
static int sha512_ce_update(struct shash_desc *desc, const u8 *data,
			    unsigned int len)
{
	// Functional Utility: Delegates data buffering and block processing to the base SHA-512 update function,
	// using the CE-accelerated transform.
	return sha512_base_do_update_blocks(desc, data, len,
					    sha512_ce_transform);
}

/**
 * @brief Implements the shash 'finup' (finalize and update) operation using ARMv8 CE.
 * @details This function is responsible for the final stages of the SHA-512/384
 * hashing process. It handles any remaining data in the input stream, applies
 * the necessary padding as per the SHA-512 standard, and then computes the final
 * hash digest. It acts as a straightforward wrapper that delegates these tasks
 * to the `sha512_base_do_finup` and `sha512_base_finish` helper functions,
 * utilizing the hardware-accelerated `sha512_ce_transform` for core block processing.
 *
 * @param desc The shash descriptor.
 * @param data The final data chunk.
 * @param len  The length of the final data.
 * @param out  The buffer for the resulting hash digest.
 * @return 0 on success.
 * Functional Utility: Completes the SHA-512/384 hashing process, including padding and final digest generation, using ARMv8 CE accelerated block transformation.
 */
static int sha512_ce_finup(struct shash_desc *desc, const u8 *data,
			   unsigned int len, u8 *out)
{
	// Functional Utility: Processes any remaining input data and applies SHA-512 padding using the hardware-accelerated transform.
	sha512_base_do_finup(desc, data, len, sha512_ce_transform);
	// Functional Utility: Writes the final computed SHA-512/384 hash digest to the output buffer.
	return sha512_base_finish(desc, out);
}

/**
 * @brief Array of algorithm definitions for the CE-accelerated SHA-384 and SHA-512.
 * @details This array holds the `shash_alg` structures that register the ARMv8
 * Cryptography Extensions (CE) accelerated implementations for both SHA-384
 * (`sha384-ce`) and SHA-512 (`sha512-ce`). Both algorithms share the same
 * underlying `sha512_ce_update` and `sha512_ce_finup` functions, as they
 * both rely on the core SHA-512 block transformation. They differ primarily
 * in their initialization (`init`) functions (calling `sha384_base_init` or
 * `sha512_base_init`) and their `digestsize`. The `cra_priority` of 200 ensures
 * these hardware-accelerated versions are preferred over generic C
 * implementations, providing optimal performance when CE is available.
 * Functional Utility: Defines and registers high-priority, hardware-accelerated SHA-512/384 algorithms for the kernel crypto API.
 */
static struct shash_alg algs[] = { {
	.init			= sha384_base_init, @ Functional Role: Initializes SHA-384 hash state.
	.update			= sha512_ce_update, @ Functional Role: Updates SHA-384 hash state with new data using CE.
	.finup			= sha512_ce_finup, @ Functional Role: Finalizes SHA-384 hash computation and outputs digest using CE.
	.descsize		= SHA512_STATE_SIZE,
	.digestsize		= SHA384_DIGEST_SIZE,
	.base.cra_name		= "sha384",
	.base.cra_driver_name	= "sha384-ce",
	.base.cra_priority	= 200, @ Functional Role: Sets priority higher than generic C implementations.
	.base.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
				  CRYPTO_AHASH_ALG_FINUP_MAX,
	.base.cra_blocksize	= SHA512_BLOCK_SIZE,
	.base.cra_module	= THIS_MODULE,
}, {
	.init			= sha512_base_init, @ Functional Role: Initializes SHA-512 hash state.
	.update			= sha512_ce_update, @ Functional Role: Updates SHA-512 hash state with new data using CE.
	.finup			= sha512_ce_finup, @ Functional Role: Finalizes SHA-512 hash computation and outputs digest using CE.
	.descsize		= SHA512_STATE_SIZE,
	.digestsize		= SHA512_DIGEST_SIZE,
	.base.cra_name		= "sha512",
	.base.cra_driver_name	= "sha512-ce",
	.base.cra_priority	= 200, @ Functional Role: Sets priority higher than generic C implementations.
	.base.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
				  CRYPTO_AHASH_ALG_FINUP_MAX,
	.base.cra_blocksize	= SHA512_BLOCK_SIZE,
	.base.cra_module	= THIS_MODULE,
} };

/**
 * @brief Module initialization function for ARM64 CE SHA-512/384 glue code.
 * @details This function is the entry point when the kernel module is loaded.
 * It registers the hardware-accelerated SHA-512/384 algorithms (`algs` array)
 * with the Linux kernel's cryptographic API. This makes the `sha512-ce` driver
 * available to the system, allowing applications and other kernel components
 * to utilize the ARMv8 Cryptography Extensions for efficient SHA-512 and
 * SHA-384 hashing. The loading is conditionally managed by `module_cpu_feature_match`,
 * ensuring it only occurs on compatible hardware.
 * Functional Utility: Registers hardware-accelerated SHA-512/384 algorithms with the kernel crypto API, making them available for system use.
 * @return 0 on successful registration, or an error code on failure.
 */
static int __init sha512_ce_mod_init(void)
{
	return crypto_register_shashes(algs, ARRAY_SIZE(algs));
}

/**
 * @brief Module cleanup function for ARM64 CE SHA-512/384 glue code.
 * @details This function is the exit point when the kernel module is unloaded.
 * Functional Utility: Unregisters the hardware-accelerated SHA-512/384 algorithms
 * (`algs` array) from the Linux kernel's cryptographic API. This cleanly
 * removes the `sha512-ce` driver from the system, releasing associated resources
 * and preventing any lingering references after the module is no longer in use.
 * This ensures proper resource management upon module unload.
 */
static void __exit sha512_ce_mod_fini(void)
{
	crypto_unregister_shashes(algs, ARRAY_SIZE(algs));
}

/**
 * @brief Macro for conditional module loading based on CPU features.
 * @details This macro plays a crucial role in ensuring that the `sha512-ce` module
 * is only loaded and initialized on ARMv8-A systems that possess the required
 * SHA-512 Cryptography Extensions. It effectively creates a module alias that
 * links the module to the presence of specific CPU features. This mechanism
 * allows the kernel to dynamically load the hardware-accelerated SHA-512/384
 * implementation automatically when compatible hardware is detected, and prevents
 * loading on unsupported systems, thereby enhancing system stability and
 * optimizing resource utilization by avoiding unnecessary module loading.
 * It connects `sha512_ce_mod_init` with the SHA512 CPU feature.
 * Functional Utility: Ensures the SHA-512 CE module is loaded only on ARM64 CPUs with SHA-512 Cryptography Extensions.
 */
module_cpu_feature_match(SHA512, sha512_ce_mod_init);
module_exit(sha512_ce_mod_fini);
