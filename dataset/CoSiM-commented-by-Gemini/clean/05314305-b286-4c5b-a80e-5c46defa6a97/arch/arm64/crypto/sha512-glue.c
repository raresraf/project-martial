/**
 * @file sha512-glue.c
 * @brief Glue for generic AArch64-accelerated SHA-384/SHA-512 implementations.
 * @details This file provides the C-level glue code to integrate a generic,
 * non-CE (Cryptography Extensions) AArch64 assembly implementation of the
 * SHA-512 and SHA-384 algorithms with the Linux kernel's cryptographic API.
 * It serves as a baseline accelerated implementation that is significantly
 * faster than plain C code, and is crucial as a fallback on AArch64 systems
 * that do not have the dedicated SHA-512 hardware instructions (Crypto Extensions).
 * Its priority ensures it is chosen over generic C implementations but is
 * superseded by more highly optimized CE versions when available.
 */
// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Linux/arm64 port of the OpenSSL SHA512 implementation for AArch64
 *
 * Copyright (c) 2016 Linaro Ltd. <ard.biesheuvel@linaro.org>
 */

#include <crypto/internal/hash.h>
#include <crypto/sha2.h>
#include <crypto/sha512_base.h>
#include <linux/kernel.h>
#include <linux/module.h>

MODULE_DESCRIPTION("SHA-384/SHA-512 secure hash for arm64");
MODULE_AUTHOR("Andy Polyakov <appro@openssl.org>");
MODULE_AUTHOR("Ard Biesheuvel <ard.biesheuvel@linaro.org>");
MODULE_LICENSE("GPL v2");
MODULE_ALIAS_CRYPTO("sha384");
MODULE_ALIAS_CRYPTO("sha512");

/**
 * @brief Entry point for the generic AArch64 assembly SHA-512 block transformation.
 * @details This function is an `asmlinkage` declaration, indicating that its
 * implementation is provided in a separate assembly file (e.g., `sha512-armv8.S`
 * or similar). It performs the core SHA-512 block compression logic using
 * optimized AArch64 instructions. This is a scalar assembly version,
 * not utilizing NEON or Crypto Extensions.
 *
 * @param digest Pointer to the SHA-512 state array.
 * @param data   Pointer to the source data.
 * @param num_blks Number of 128-byte blocks to process.
 * Functional Utility: Dispatches to the generic AArch64 assembly routine for efficient SHA-512 block processing.
 */
asmlinkage void sha512_blocks_arch(u64 *digest, const void *data,
				   unsigned int num_blks);

/**
 * @brief C wrapper for the generic AArch64 assembly SHA-512/384 transform function.
 * @details This function provides a straightforward C-language wrapper that
 * calls the `sha512_blocks_arch` assembly routine to perform the core block
 * transformation. For this scalar assembly version, no special NEON or FPU
 * context management (e.g., `kernel_neon_begin()`/`kernel_neon_end()`) is
 * required, as it does not directly utilize NEON instructions.
 *
 * @param sst    Pointer to the standard SHA-512 state.
 * @param src    Pointer to the source data.
 * @param blocks Number of 128-byte blocks to process.
 * Functional Utility: Provides a safe C-language interface to the generic AArch64 SHA-512/384 assembly block transformation.
 */
static void sha512_arm64_transform(struct sha512_state *sst, u8 const *src,
				   int blocks)
{
	// Functional Utility: Directly invokes the generic AArch64 assembly function for SHA-512/384 block transformation.
	sha512_blocks_arch(sst->state, src, blocks);
}

/**
 * @brief Implements the `shash` 'update' operation for the generic AArch64 SHA-512/384.
 * @details This function integrates the generic AArch64 assembly-accelerated
 * SHA-512/384 into the kernel's `shash` API for incremental updates. It acts as
 * a straightforward wrapper that delegates data buffering and partial block
 * processing to the `sha512_base_do_update_blocks` helper function, providing
 * the `sha512_arm64_transform` as the callback for core block processing.
 *
 * @param desc The shash descriptor.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return 0 on success.
 * Functional Utility: Updates the SHA-512/384 hash state incrementally, leveraging generic AArch64 assembly for core block processing.
 */
static int sha512_update(struct shash_desc *desc, const u8 *data,
			 unsigned int len)
{
	// Functional Utility: Delegates data buffering and block processing to the base SHA-512 update function,
	// using the generic AArch64 assembly transform.
	return sha512_base_do_update_blocks(desc, data, len,
					    sha512_arm64_transform);
}

/**
 * @brief Implements the `shash` 'finup' (finalize and update) operation for the generic AArch64 SHA-512/384.
 * @details This function is responsible for the final stages of the SHA-512/384
 * hashing process. It handles any remaining data in the input stream, applies
 * the necessary padding as per the SHA-512 standard, and then computes the final
 * hash digest. It acts as a straightforward wrapper that delegates these tasks
 * to the `sha512_base_do_finup` and `sha512_base_finish` helper functions,
 * utilizing the generic AArch64 assembly `sha512_arm64_transform` for core
 * block processing.
 *
 * @param desc The shash descriptor.
 * @param data The final data chunk.
 * @param len  The length of the final data.
 * @param out  The buffer for the resulting hash digest.
 * @return 0 on success.
 * Functional Utility: Completes the SHA-512/384 hashing process, including padding and final digest generation, using generic AArch64 assembly block transformation.
 */
static int sha512_finup(struct shash_desc *desc, const u8 *data,
			unsigned int len, u8 *out)
{
	// Functional Utility: Processes any remaining input data and applies SHA-512 padding using the generic AArch64 assembly block transformation.
	sha512_base_do_finup(desc, data, len, sha512_arm64_transform);
	// Functional Utility: Writes the final computed SHA-512/384 hash digest to the output buffer.
	return sha512_base_finish(desc, out);
}

/**
 * @brief Array of algorithm definitions for the generic AArch64 SHA-384 and SHA-512.
 * @details This array holds the `shash_alg` structures that register the generic
 * AArch64 assembly implementations for both SHA-384 (`sha384-arm64`) and
 * SHA-512 (`sha512-arm64`). Both algorithms share the same underlying
 * `sha512_update` and `sha512_finup` functions, as they both rely on the core
 * `sha512_arm64_transform` block transformation. They differ primarily
 * in their initialization (`init`) functions (calling `sha384_base_init` or
 * `sha512_base_init`) and their `digestsize`. The `cra_priority` of 150
 * ensures these accelerated versions are preferred over generic C
 * implementations but are superseded by more highly optimized Crypto Extensions
 * (CE) versions when available.
 * Functional Utility: Defines and registers generic AArch64 accelerated SHA-512/384 algorithms for the kernel crypto API.
 */
static struct shash_alg algs[] = { {
	.digestsize		= SHA512_DIGEST_SIZE,
	.init			= sha512_base_init, @ Functional Role: Initializes SHA-512 hash state.
	.update			= sha512_update, @ Functional Role: Updates SHA-512 hash state with new data using generic AArch64 assembly.
	.finup			= sha512_finup, @ Functional Role: Finalizes SHA-512 hash computation and outputs digest using generic AArch64 assembly.
	.descsize		= SHA512_STATE_SIZE,
	.base.cra_name		= "sha512",
	.base.cra_driver_name	= "sha512-arm64",
	.base.cra_priority	= 150, @ Functional Role: Sets priority higher than generic C, lower than CE.
	.base.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
				  CRYPTO_AHASH_ALG_FINUP_MAX,
	.base.cra_blocksize	= SHA512_BLOCK_SIZE,
	.base.cra_module	= THIS_MODULE,
}, {
	.digestsize		= SHA384_DIGEST_SIZE,
	.init			= sha384_base_init, @ Functional Role: Initializes SHA-384 hash state.
	.update			= sha512_update, @ Functional Role: Updates SHA-384 hash state with new data using generic AArch64 assembly.
	.finup			= sha512_finup, @ Functional Role: Finalizes SHA-384 hash computation and outputs digest using generic AArch64 assembly.
	.descsize		= SHA512_STATE_SIZE,
	.base.cra_name		= "sha384",
	.base.cra_driver_name	= "sha384-arm64",
	.base.cra_priority	= 150, @ Functional Role: Sets priority higher than generic C, lower than CE.
	.base.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
				  CRYPTO_AHASH_ALG_FINUP_MAX,
	.base.cra_blocksize	= SHA384_BLOCK_SIZE,
	.base.cra_module	= THIS_MODULE,
} };

/**
 * @brief Module initialization function for generic AArch64 SHA-512/384 glue code.
 * @details This function is the entry point when the kernel module is loaded.
 * It registers the generic AArch64 assembly-accelerated SHA-512/384 algorithms
 * (`algs` array) with the Linux kernel's cryptographic API. This makes these
 * drivers available to the system, providing an optimized baseline for SHA-512/384
 * hashing on AArch64 platforms that may not have dedicated Crypto Extensions,
 * or as a fallback when CE is not utilized.
 * Functional Utility: Registers generic AArch64 accelerated SHA-512/384 algorithms with the kernel crypto API, providing an optimized baseline implementation.
 * @return 0 on successful registration, or an error code on failure.
 */
static int __init sha512_mod_init(void)
{
	return crypto_register_shashes(algs, ARRAY_SIZE(algs));
}

/**
 * @brief Module cleanup function for generic AArch64 SHA-512/384 glue code.
 * @details This function is the exit point when the kernel module is unloaded.
 * Functional Utility: Unregisters all SHA-512/384 algorithm implementations
 * (`algs` array) that were successfully registered during module initialization.
 * This cleanly removes the drivers from the system, releasing associated resources
 * and preventing any lingering references after the module is no longer in use.
 * This ensures proper resource management upon module unload.
 */
static void __exit sha512_mod_fini(void)
{
	crypto_unregister_shashes(algs, ARRAY_SIZE(algs));
}

module_init(sha512_mod_init);
module_exit(sha512_mod_fini);
