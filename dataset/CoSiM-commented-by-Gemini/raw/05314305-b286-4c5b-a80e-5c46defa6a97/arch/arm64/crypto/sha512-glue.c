/**
 * @file sha512-glue.c
 * @brief Glue for generic AArch64-accelerated SHA-384/SHA-512.
 * @details This file provides the C-level glue code to integrate a generic,
 * non-CE (Crypto Extensions) AArch64 assembly implementation of the SHA-512 and
 * SHA-384 algorithms with the Linux kernel's cryptographic API. It serves as a
 * baseline accelerated implementation that is faster than plain C code and acts
 * as a fallback on systems that do not have the dedicated SHA-512 hardware
 * instructions.
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
 * @brief Entry point for the generic AArch64 assembly SHA-512 implementation.
 * @param digest Pointer to the SHA-512 state array.
 * @param data   Pointer to the source data.
 * @param num_blks Number of 128-byte blocks to process.
 */
asmlinkage void sha512_blocks_arch(u64 *digest, const void *data,
				   unsigned int num_blks);

/**
 * @brief C wrapper for the generic AArch64 assembly transform function.
 * @param sst    Pointer to the standard SHA-512 state.
 * @param src    Pointer to the source data.
 * @param blocks Number of blocks to process.
 *
 * This function simply calls the assembly function to perform the block
 * transformation. No special FPU context management is required for this
 * scalar assembly version.
 */
static void sha512_arm64_transform(struct sha512_state *sst, u8 const *src,
				   int blocks)
{
	sha512_blocks_arch(sst->state, src, blocks);
}

/**
 * @brief Implements the shash 'update' operation for the generic arm64 version.
 * @param desc The shash descriptor.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return 0 on success.
 */
static int sha512_update(struct shash_desc *desc, const u8 *data,
			 unsigned int len)
{
	return sha512_base_do_update_blocks(desc, data, len,
					    sha512_arm64_transform);
}

/**
 * @brief Implements the shash 'finup' operation for the generic arm64 version.
 * @param desc The shash descriptor.
 * @param data The final data chunk.
 * @param len  The length of the final data.
 * @param out  The buffer for the resulting hash digest.
 * @return 0 on success.
 */
static int sha512_finup(struct shash_desc *desc, const u8 *data,
			unsigned int len, u8 *out)
{
	sha512_base_do_finup(desc, data, len, sha512_arm64_transform);
	return sha512_base_finish(desc, out);
}

/**
 * @brief Algorithm definitions for the generic AArch64 SHA-384 and SHA-512.
 *
 * These structures register the `sha384-arm64` and `sha512-arm64`
 * implementations. Their priority (150) is set to be higher than a generic C
 * implementation but lower than the CE-accelerated version, ensuring they are
 * used as a fallback.
 */
static struct shash_alg algs[] = { {
	.digestsize		= SHA512_DIGEST_SIZE,
	.init			= sha512_base_init,
	.update			= sha512_update,
	.finup			= sha512_finup,
	.descsize		= SHA512_STATE_SIZE,
	.base.cra_name		= "sha512",
	.base.cra_driver_name	= "sha512-arm64",
	.base.cra_priority	= 150,
	.base.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
				  CRYPTO_AHASH_ALG_FINUP_MAX,
	.base.cra_blocksize	= SHA512_BLOCK_SIZE,
	.base.cra_module	= THIS_MODULE,
}, {
	.digestsize		= SHA384_DIGEST_SIZE,
	.init			= sha384_base_init,
	.update			= sha512_update,
	.finup			= sha512_finup,
	.descsize		= SHA512_STATE_SIZE,
	.base.cra_name		= "sha384",
	.base.cra_driver_name	= "sha384-arm64",
	.base.cra_priority	= 150,
	.base.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
				  CRYPTO_AHASH_ALG_FINUP_MAX,
	.base.cra_blocksize	= SHA384_BLOCK_SIZE,
	.base.cra_module	= THIS_MODULE,
} };

/**
 * @brief Module initialization function.
 *
 * Registers the generic AArch64 SHA-512/384 algorithms with the kernel
 * crypto API.
 */
static int __init sha512_mod_init(void)
{
	return crypto_register_shashes(algs, ARRAY_SIZE(algs));
}

/**
 * @brief Module cleanup function.
 *
 * Unregisters the SHA-512/384 algorithms when the module is unloaded.
 */
static void __exit sha512_mod_fini(void)
{
	crypto_unregister_shashes(algs, ARRAY_SIZE(algs));
}

module_init(sha512_mod_init);
module_exit(sha512_mod_fini);

