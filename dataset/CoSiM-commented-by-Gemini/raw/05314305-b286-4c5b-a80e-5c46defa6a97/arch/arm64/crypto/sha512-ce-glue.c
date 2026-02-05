/**
 * @file sha512-ce-glue.c
 * @brief Glue for ARMv8 Crypto Extensions SHA-384/SHA-512 for AArch64.
 * @details This file provides the C-level glue code to integrate the ARMv8
 * Cryptography Extensions (CE) accelerated SHA-512 and SHA-384 implementations
 * with the Linux kernel's cryptographic API. Its main responsibilities are
 * registering the hardware-accelerated algorithms and safely managing the
 * NEON/FPU context during cryptographic operations.
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
 * @brief Entry point for the ARMv8 CE assembly implementation of SHA-512.
 * @param sst    Pointer to the SHA-512 state structure.
 * @param src    Pointer to the source data.
 * @param blocks Number of 128-byte blocks to process.
 * @return       Number of blocks remaining.
 *
 * This function, defined in sha512-ce-core.S, uses dedicated hardware
 * instructions to perform the SHA-512 compression function.
 */
asmlinkage int __sha512_ce_transform(struct sha512_state *sst, u8 const *src,
				     int blocks);

/**
 * @brief C wrapper for the assembly transform function.
 * @param sst    Pointer to the standard SHA-512 state.
 * @param src    Pointer to the source data.
 * @param blocks Number of blocks to process.
 *
 * This function safely manages the NEON/FPU context by wrapping the call to
 * the core assembly function within kernel_neon_begin() and kernel_neon_end().
 */
static void sha512_ce_transform(struct sha512_state *sst, u8 const *src,
				int blocks)
{
	do {
		int rem;

		kernel_neon_begin();
		rem = __sha512_ce_transform(sst, src, blocks);
		kernel_neon_end();
		src += (blocks - rem) * SHA512_BLOCK_SIZE;
		blocks = rem;
	} while (blocks);
}

/**
 * @brief Implements the shash 'update' operation using ARMv8 CE.
 * @param desc The shash descriptor.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return 0 on success.
 *
 * A straightforward wrapper that delegates to the base helper function,
 * providing the CE transform function as the callback.
 */
static int sha512_ce_update(struct shash_desc *desc, const u8 *data,
			    unsigned int len)
{
	return sha512_base_do_update_blocks(desc, data, len,
					    sha512_ce_transform);
}

/**
 * @brief Implements the shash 'finup' operation using ARMv8 CE.
 * @param desc The shash descriptor.
 * @param data The final data chunk.
 * @param len  The length of the final data.
 * @param out  The buffer for the resulting hash digest.
 * @return 0 on success.
 *
 * A straightforward wrapper that delegates to the base helper functions for
 * finalization, using the CE transform as the core processing callback.
 */
static int sha512_ce_finup(struct shash_desc *desc, const u8 *data,
			   unsigned int len, u8 *out)
{
	sha512_base_do_finup(desc, data, len, sha512_ce_transform);
	return sha512_base_finish(desc, out);
}

/**
 * @brief Algorithm definitions for the CE-accelerated SHA-384 and SHA-512.
 *
 * These structures register the `sha384-ce` and `sha512-ce` implementations.
 * They share the same underlying hardware-accelerated functions, differing only
 * in their digest sizes and initial state vectors (handled by the init functions).
 * The high priority (200) ensures they are preferred over generic implementations.
 */
static struct shash_alg algs[] = { {
	.init			= sha384_base_init,
	.update			= sha512_ce_update,
	.finup			= sha512_ce_finup,
	.descsize		= SHA512_STATE_SIZE,
	.digestsize		= SHA384_DIGEST_SIZE,
	.base.cra_name		= "sha384",
	.base.cra_driver_name	= "sha384-ce",
	.base.cra_priority	= 200,
	.base.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
				  CRYPTO_AHASH_ALG_FINUP_MAX,
	.base.cra_blocksize	= SHA512_BLOCK_SIZE,
	.base.cra_module	= THIS_MODULE,
}, {
	.init			= sha512_base_init,
	.update			= sha512_ce_update,
	.finup			= sha512_ce_finup,
	.descsize		= SHA512_STATE_SIZE,
	.digestsize		= SHA512_DIGEST_SIZE,
	.base.cra_name		= "sha512",
	.base.cra_driver_name	= "sha512-ce",
	.base.cra_priority	= 200,
	.base.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
				  CRYPTO_AHASH_ALG_FINUP_MAX,
	.base.cra_blocksize	= SHA512_BLOCK_SIZE,
	.base.cra_module	= THIS_MODULE,
} };

/**
 * @brief Module initialization function.
 *
 * Registers the CE-accelerated SHA-512/384 algorithms with the kernel crypto API.
 */
static int __init sha512_ce_mod_init(void)
{
	return crypto_register_shashes(algs, ARRAY_SIZE(algs));
}

/**
 * @brief Module cleanup function.
 *
 * Unregisters the SHA-512/384 algorithms when the module is unloaded.
 */
static void __exit sha512_ce_mod_fini(void)
{
	crypto_unregister_shashes(algs, ARRAY_SIZE(algs));
}

/*
 * This macro uses CPU feature detection to ensure this module is only loaded on
 * systems that have the ARMv8 SHA-512 Crypto Extensions, allowing for automatic
 * loading of the most optimal driver available.
 */
module_cpu_feature_match(SHA512, sha512_ce_mod_init);
module_exit(sha512_ce_mod_fini);
