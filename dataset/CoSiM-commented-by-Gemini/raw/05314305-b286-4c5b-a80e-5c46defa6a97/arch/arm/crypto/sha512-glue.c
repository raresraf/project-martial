/**
 * @file sha512-glue.c
 * @brief Glue logic for ARM-accelerated SHA-384/SHA-512 implementations.
 * @details This file registers ARM-accelerated versions of the SHA-512 and
 * SHA-384 hash algorithms with the Linux kernel's cryptographic API. It follows
 * a layered registration strategy:
 * 1. It registers a baseline assembly implementation (sha512-arm) suitable for
 *    ARMv4 and later.
 * 2. If the CPU supports NEON, it also registers a higher-priority NEON-accelerated
 *    version (sha512-neon).
 * This allows the crypto subsystem to automatically select the most performant
 * implementation available on the host CPU.
 */
// SPDX-License-Identifier: GPL-2.0-only
/*
 * sha512-glue.c - accelerated SHA-384/512 for ARM
 *
 * Copyright (C) 2015 Linaro Ltd <ard.biesheuvel@linaro.org>
 */

#include <asm/hwcap.h>
#include <asm/neon.h>
#include <crypto/internal/hash.h>
#include <crypto/sha2.h>
#include <crypto/sha512_base.h>
#include <linux/kernel.h>
#include <linux/module.h>

#include "sha512.h"

MODULE_DESCRIPTION("Accelerated SHA-384/SHA-512 secure hash for ARM");
MODULE_AUTHOR("Ard Biesheuvel <ard.biesheuvel@linaro.org>");
MODULE_LICENSE("GPL v2");

MODULE_ALIAS_CRYPTO("sha384");
MODULE_ALIAS_CRYPTO("sha512");
MODULE_ALIAS_CRYPTO("sha384-arm");
MODULE_ALIAS_CRYPTO("sha512-arm");

/**
 * @brief Entry point for the assembly-based SHA-512 block transformation.
 * @param state  Pointer to the SHA-512 state structure.
 * @param src    Pointer to the source data blocks.
 * @param blocks Number of 128-byte blocks to process.
 *
 * This function is defined in assembly (sha512-armv4.S) and can dispatch to
 * either a scalar ARM or a NEON implementation at runtime based on CPU features.
 */
asmlinkage void sha512_block_data_order(struct sha512_state *state,
					u8 const *src, int blocks);

/**
 * @brief Implements the shash 'update' operation for the scalar ARM version.
 * @param desc The shash descriptor containing the hash state.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return 0 on success.
 *
 * This function is a wrapper that passes data to the core block processing
 * helper, using the ARM assembly implementation as the transform function.
 */
static int sha512_arm_update(struct shash_desc *desc, const u8 *data,
			     unsigned int len)
{
	return sha512_base_do_update_blocks(desc, data, len,
					    sha512_block_data_order);
}

/**
 * @brief Implements the shash 'finup' operation for the scalar ARM version.
 * @param desc The shash descriptor.
 * @param data The final data chunk.
 * @param len  The length of the final data.
 * @param out  The buffer for the resulting hash digest.
 * @return 0 on success.
 *
 * Processes the last data segment and computes the final hash digest, using
 * the ARM assembly implementation.
 */
static int sha512_arm_finup(struct shash_desc *desc, const u8 *data,
			    unsigned int len, u8 *out)
{
	sha512_base_do_finup(desc, data, len, sha512_block_data_order);
	return sha512_base_finish(desc, out);
}

/**
 * @brief Algorithm definitions for the ARM-optimized SHA-384 and SHA-512.
 *
 * These structures register the baseline `sha384-arm` and `sha512-arm`
 * implementations. They share the same underlying update/finup functions,
 * differing only in their names, digest sizes, and initialization vectors.
 * Their priority is set to be preferred over generic C code but lower than
 * NEON or other hardware-specific versions.
 */
static struct shash_alg sha512_arm_algs[] = { {
	.init			= sha384_base_init,
	.update			= sha512_arm_update,
	.finup			= sha512_arm_finup,
	.descsize		= SHA512_STATE_SIZE,
	.digestsize		= SHA384_DIGEST_SIZE,
	.base			= {
		.cra_name		= "sha384",
		.cra_driver_name	= "sha384-arm",
		.cra_priority		= 250,
		.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
					  CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize		= SHA512_BLOCK_SIZE,
		.cra_module		= THIS_MODULE,
	}
},  {
	.init			= sha512_base_init,
	.update			= sha512_arm_update,
	.finup			= sha512_arm_finup,
	.descsize		= SHA512_STATE_SIZE,
	.digestsize		= SHA512_DIGEST_SIZE,
	.base			= {
		.cra_name		= "sha512",
		.cra_driver_name	= "sha512-arm",
		.cra_priority		= 250,
		.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
					  CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize		= SHA512_BLOCK_SIZE,
		.cra_module		= THIS_MODULE,
	}
} };

/**
 * @brief Module initialization function.
 *
 * Registers the available SHA-512/384 implementations. It always registers
 * the baseline ARM assembly version. If the CPU supports NEON, it also
 * registers the higher-priority NEON-accelerated version.
 */
static int __init sha512_arm_mod_init(void)
{
	int err;

	err = crypto_register_shashes(sha512_arm_algs,
				      ARRAY_SIZE(sha512_arm_algs));
	if (err)
		return err;

	// Pre-condition: Check if NEON is supported by the kernel and CPU.
	if (IS_ENABLED(CONFIG_KERNEL_MODE_NEON) && cpu_has_neon()) {
		// If NEON is available, register the higher-priority NEON implementation.
		err = crypto_register_shashes(sha512_neon_algs,
					      ARRAY_SIZE(sha512_neon_algs));
		if (err)
			goto err_unregister;
	}
	return 0;

err_unregister:
	crypto_unregister_shashes(sha512_arm_algs,
				  ARRAY_SIZE(sha512_arm_algs));

	return err;
}

/**
 * @brief Module cleanup function.
 *
 * Unregisters all SHA-512/384 algorithm implementations that were loaded at
 * initialization time.
 */
static void __exit sha512_arm_mod_fini(void)
{
	crypto_unregister_shashes(sha512_arm_algs,
				  ARRAY_SIZE(sha512_arm_algs));
	if (IS_ENABLED(CONFIG_KERNEL_MODE_NEON) && cpu_has_neon())
		crypto_unregister_shashes(sha512_neon_algs,
					  ARRAY_SIZE(sha512_neon_algs));
}

module_init(sha512_arm_mod_init);
module_exit(sha512_arm_mod_fini);
