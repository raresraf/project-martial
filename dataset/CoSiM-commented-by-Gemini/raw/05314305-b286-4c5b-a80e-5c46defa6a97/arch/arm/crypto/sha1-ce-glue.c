/**
 * @file sha1-ce-glue.c
 * @brief Glue code for SHA-1 using ARMv8 Crypto Extensions.
 * @details This file provides the interface layer to integrate the ARMv8
 * Cryptography Extensions (CE) accelerated SHA-1 implementation with the
 * Linux kernel's cryptographic API. It handles the registration of the
 * hardware-accelerated algorithm and ensures safe use of the NEON/FPU context
 * from kernel mode.
 */
// SPDX-License-Identifier: GPL-2.0-only
/*
 * sha1-ce-glue.c - SHA-1 secure hash using ARMv8 Crypto Extensions
 *
 * Copyright (C) 2015 Linaro Ltd <ard.biesheuvel@linaro.org>
 */

#include <asm/neon.h>
#include <crypto/internal/hash.h>
#include <crypto/sha1.h>
#include <crypto/sha1_base.h>
#include <linux/cpufeature.h>
#include <linux/kernel.h>
#include <linux/module.h>

MODULE_DESCRIPTION("SHA1 secure hash using ARMv8 Crypto Extensions");
MODULE_AUTHOR("Ard Biesheuvel <ard.biesheuvel@linaro.org>");
MODULE_LICENSE("GPL v2");

/**
 * @brief Processes SHA-1 blocks using ARMv8 Crypto Extensions.
 * @param sst    Pointer to the SHA-1 state structure.
 * @param src    Pointer to the source data.
 * @param blocks Number of 64-byte blocks to process.
 *
 * This is the entry point to the assembly implementation in sha1-ce-core.S,
 * which uses dedicated hardware instructions to perform the SHA-1 compression.
 */
asmlinkage void sha1_ce_transform(struct sha1_state *sst, u8 const *src,
				  int blocks);

/**
 * @brief Implements the shash 'update' operation using ARMv8 CE.
 * @param desc The shash descriptor containing the hash state.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return The number of bytes not processed (if any).
 *
 * This function wraps the hardware-accelerated block processing. It manages the
 * NEON/FPU context by calling kernel_neon_begin() before and kernel_neon_end()
 * after the core transformation, ensuring the FPU state is preserved.
 */
static int sha1_ce_update(struct shash_desc *desc, const u8 *data,
			  unsigned int len)
{
	int remain;

	// Pre-condition: Save the current FPU state and enable NEON/CE use.
	kernel_neon_begin();
	// Functional Utility: Delegates block processing to the assembly function.
	remain = sha1_base_do_update_blocks(desc, data, len, sha1_ce_transform);
	// Post-condition: Restore the FPU state.
	kernel_neon_end();

	return remain;
}

/**
 * @brief Implements the shash 'finup' (finalize and update) operation using CE.
 * @param desc The shash descriptor.
 * @param data The final data chunk to be hashed.
 * @param len  The length of the final data.
 * @param out  The buffer to store the resulting 20-byte hash.
 * @return 0 on success.
 *
 * This function handles the final data segment and computes the final hash,
 * ensuring safe management of the NEON/FPU context.
 */
static int sha1_ce_finup(struct shash_desc *desc, const u8 *data,
			 unsigned int len, u8 *out)
{
	// Pre-condition: Save FPU state.
	kernel_neon_begin();
	// Functional Utility: Perform the final update and padding via the CE implementation.
	sha1_base_do_finup(desc, data, len, sha1_ce_transform);
	// Post-condition: Restore FPU state.
	kernel_neon_end();

	return sha1_base_finish(desc, out);
}

/**
 * @brief Defines the CE-accelerated SHA-1 algorithm for the crypto API.
 *
 * This structure registers the hardware-accelerated implementation, setting a
 * high priority to ensure it is selected on CPUs with Crypto Extensions.
 */
static struct shash_alg alg = {
	.init			= sha1_base_init,
	.update			= sha1_ce_update,
	.finup			= sha1_ce_finup,
	.descsize		= SHA1_STATE_SIZE,
	.digestsize		= SHA1_DIGEST_SIZE,
	.base			= {
		.cra_name		= "sha1",
		.cra_driver_name	= "sha1-ce",
		.cra_priority		= 200,
		.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY,
		.cra_blocksize		= SHA1_BLOCK_SIZE,
		.cra_module		= THIS_MODULE,
	}
};

/**
 * @brief Module initialization function.
 *
 * Registers the CE-accelerated SHA-1 algorithm with the kernel crypto API.
 */
static int __init sha1_ce_mod_init(void)
{
	return crypto_register_shash(&alg);
}

/**
 * @brief Module cleanup function.
 *
 * Unregisters the SHA-1 algorithm when the module is unloaded.
 */
static void __exit sha1_ce_mod_fini(void)
{
	crypto_unregister_shash(&alg);
}

/*
 * This macro ensures that this module is only loaded on systems that have
 * the ARMv8 SHA-1 crypto extensions. It creates a module alias based on the
 * CPU feature, allowing for automatic loading on compatible hardware.
 */
module_cpu_feature_match(SHA1, sha1_ce_mod_init);
module_exit(sha1_ce_mod_fini);
