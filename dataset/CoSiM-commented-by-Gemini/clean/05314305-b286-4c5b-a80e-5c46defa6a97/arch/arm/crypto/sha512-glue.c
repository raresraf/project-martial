/**
 * @file sha512-glue.c
 * @brief Glue logic for ARM-accelerated SHA-384/SHA-512 implementations.
 * @details This file orchestrates the registration of ARM-accelerated versions
 * of the SHA-512 and SHA-384 hash algorithms with the Linux kernel's
 * cryptographic API. It implements a sophisticated layered registration strategy:
 * 1. A baseline assembly implementation (`sha512-arm`) is always registered, providing
 *    optimized performance for ARMv4 and later architectures.
 * 2. If the CPU and kernel configuration (`CONFIG_KERNEL_MODE_NEON`) support NEON,
 *    a separate, higher-priority NEON-accelerated version (`sha512-neon`) is also
 *    registered.
 * This intelligent dispatch mechanism ensures that the crypto subsystem
 * automatically selects the most performant and hardware-optimized implementation
 * available on the host CPU, significantly boosting cryptographic throughput
 * for SHA-512 and SHA-384 operations.
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
 * @details This function serves as the common entry point for both scalar ARM
 * and potentially NEON-accelerated assembly implementations of the SHA-512
 * block transformation. The actual low-level processing (which may internally
 * dispatch to NEON based on CPU features) is handled by the assembly code
 * defined in files like `sha512-armv4.S`.
 *
 * @param state  Pointer to the SHA-512 state structure.
 * @param src    Pointer to the source data blocks.
 * @param blocks Number of 128-byte blocks to process.
 * Functional Utility: Dispatches to the architecture-specific assembly routine for efficient SHA-512 block processing.
 */
asmlinkage void sha512_block_data_order(struct sha512_state *state,
					u8 const *src, int blocks);

/**
 * @brief Implements the generic `shash` 'update' operation for ARM-optimized SHA-512/384.
 * @details This function acts as a wrapper for the `shash` API's update operation,
 * integrating the ARM-optimized SHA-512/384 block processing into the kernel's
 * cryptographic framework. It internally calls `sha512_base_do_update_blocks`,
 * providing the assembly-based `sha512_block_data_order` as the core transform function.
 * This allows the cryptographic subsystem to handle data buffering and partial
 * block processing, while delegating the performance-critical full-block computations
 * to the assembly implementation.
 *
 * @param desc The shash descriptor containing the hash state.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return 0 on success.
 * Functional Utility: Integrates ARM-optimized SHA-512/384 block processing into the generic kernel cryptographic update flow.
 */
static int sha512_arm_update(struct shash_desc *desc, const u8 *data,
			     unsigned int len)
{
	// Functional Utility: Delegates the data buffering and block-wise update logic to a common SHA-512 base helper,
	// passing the ARM-specific assembly function for the actual block transformation.
	return sha512_base_do_update_blocks(desc, data, len,
					    sha512_block_data_order);
}

/**
 * @brief Implements the generic `shash` 'finup' (finalize and update) operation for ARM-optimized SHA-512/384.
 * @details This function is responsible for the final stages of the SHA-512/384
 * hashing process. It handles any remaining data in the input stream, applies
 * the necessary padding as per the SHA-512 standard, and then computes the final
 * hash digest. The core block processing for these final steps is delegated
 * to the ARM assembly implementation (`sha512_block_data_order`).
 *
 * @param desc The shash descriptor.
 * @param data The final data chunk.
 * @param len  The length of the final data.
 * @param out  The buffer for the resulting hash digest.
 * @return 0 on success.
 * Functional Utility: Completes the SHA-512/384 hashing process, including padding and final digest generation, using the ARM-optimized block transformation.
 */
static int sha512_arm_finup(struct shash_desc *desc, const u8 *data,
			    unsigned int len, u8 *out)
{
	// Functional Utility: Processes any remaining input data and applies SHA-512 padding using the ARM-optimized block transformation.
	sha512_base_do_finup(desc, data, len, sha512_block_data_order);
	// Functional Utility: Writes the final computed SHA-512/384 hash digest to the output buffer.
	return sha512_base_finish(desc, out);
}

/**
 * @brief Array of algorithm definitions for the ARM-optimized SHA-384 and SHA-512.
 * @details This array holds the `shash_alg` structures that register the baseline
 * ARM assembly implementations for both SHA-384 (`sha384-arm`) and SHA-512 (`sha512-arm`).
 * Both algorithms share the same underlying `sha512_arm_update` and `sha512_arm_finup`
 * functions, as they both rely on the core SHA-512 block transformation. They differ
 * primarily in their initialization (`init`) functions (calling `sha384_base_init`
 * or `sha512_base_init`) and their `digestsize`. The `cra_priority` of 250 ensures
 * these optimized versions are preferred over generic C implementations but
 * can be overridden by more hardware-specific accelerators like NEON.
 */
static struct shash_alg sha512_arm_algs[] = { {
	.init			= sha384_base_init, @ Functional Role: Initializes SHA-384 hash state.
	.update			= sha512_arm_update, @ Functional Role: Updates SHA-384 hash state with new data.
	.finup			= sha512_arm_finup, @ Functional Role: Finalizes SHA-384 hash computation and outputs digest.
	.descsize		= SHA512_STATE_SIZE,
	.digestsize		= SHA384_DIGEST_SIZE,
	.base			= {
		.cra_name		= "sha384",
		.cra_driver_name	= "sha384-arm",
		.cra_priority		= 250, @ Functional Role: Sets priority higher than generic C implementations.
		.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
					  CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize		= SHA512_BLOCK_SIZE,
		.cra_module		= THIS_MODULE,
	}
},  {
	.init			= sha512_base_init, @ Functional Role: Initializes SHA-512 hash state.
	.update			= sha512_arm_update, @ Functional Role: Updates SHA-512 hash state with new data.
	.finup			= sha512_arm_finup, @ Functional Role: Finalizes SHA-512 hash computation and outputs digest.
	.descsize		= SHA512_STATE_SIZE,
	.digestsize		= SHA512_DIGEST_SIZE,
	.base			= {
		.cra_name		= "sha512",
		.cra_driver_name	= "sha512-arm",
		.cra_priority		= 250, @ Functional Role: Sets priority higher than generic C implementations.
		.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
					  CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize		= SHA512_BLOCK_SIZE,
		.cra_module		= THIS_MODULE,
	}
} };

/**
 * @brief Module initialization function for ARM-accelerated SHA-512/384.
 * @details This function is the entry point when the kernel module is loaded.
 * It implements a two-tiered registration process to provide the best possible
 * SHA-512/384 performance on ARM systems:
 * 1. It always registers the baseline ARM assembly versions (`sha512_arm_algs`).
 * 2. It then conditionally checks for NEON support (`CONFIG_KERNEL_MODE_NEON`
 *    and `cpu_has_neon()`). If NEON is available, it registers a separate,
 *    higher-priority set of NEON-accelerated algorithms (`sha512_neon_algs`).
 * This ensures that the cryptographic subsystem automatically picks the most
 * optimized implementation suitable for the host CPU. Proper error handling
 * is included to unregister any previously registered algorithms on failure.
 *
 * Functional Utility: Registers available ARM-optimized SHA-512/384 implementations with the kernel crypto API, dynamically enabling NEON acceleration if supported by the CPU.
 * @return 0 on success, or an error code if registration fails.
 */
static int __init sha512_arm_mod_init(void)
{
	int err;

	// Functional Utility: Register the baseline ARM assembly SHA-512/384 algorithms.
	err = crypto_register_shashes(sha512_arm_algs,
				      ARRAY_SIZE(sha512_arm_algs));
	if (err)
		return err;

	// Pre-condition: Checks if kernel NEON mode is enabled in configuration and if the CPU has NEON capabilities.
	if (IS_ENABLED(CONFIG_KERNEL_MODE_NEON) && cpu_has_neon()) {
		// Functional Utility: If NEON is available, register the higher-priority NEON-accelerated SHA-512/384 algorithms.
		err = crypto_register_shashes(sha512_neon_algs,
					      ARRAY_SIZE(sha512_neon_algs));
		if (err)
			goto err_unregister;
	}
	return 0;

err_unregister:
	// Functional Utility: On failure to register NEON algorithms, unregister previously registered ARM algorithms to clean up.
	crypto_unregister_shashes(sha512_arm_algs,
				  ARRAY_SIZE(sha512_arm_algs));

	return err;
}

/**
 * @brief Module cleanup function for ARM-accelerated SHA-512/384.
 * @details This function is the exit point when the kernel module is unloaded.
 * Functional Utility: Unregisters all SHA-512/384 algorithm implementations
 * that were successfully registered during module initialization. This includes
 * the baseline ARM assembly versions and, if applicable, the NEON-accelerated
 * versions, ensuring a clean shutdown and release of kernel resources.
 */
static void __exit sha512_arm_mod_fini(void)
{
	// Functional Utility: Unregisters the baseline ARM assembly SHA-512/384 algorithms.
	crypto_unregister_shashes(sha512_arm_algs,
				  ARRAY_SIZE(sha512_arm_algs));
	// Pre-condition: Checks if NEON algorithms were potentially registered.
	// Functional Utility: If NEON was enabled, unregister the NEON-accelerated SHA-512/384 algorithms.
	if (IS_ENABLED(CONFIG_KERNEL_MODE_NEON) && cpu_has_neon())
		crypto_unregister_shashes(sha512_neon_algs,
					  ARRAY_SIZE(sha512_neon_algs));
}

module_init(sha512_arm_mod_init);
module_exit(sha512_arm_mod_fini);
