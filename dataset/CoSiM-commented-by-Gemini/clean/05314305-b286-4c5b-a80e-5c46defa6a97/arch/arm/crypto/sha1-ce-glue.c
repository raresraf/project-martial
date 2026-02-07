/**
 * @file sha1-ce-glue.c
 * @brief Glue code for SHA-1 using ARMv8 Crypto Extensions.
 * @details This file provides the interface layer to seamlessly integrate the
 * ARMv8 Cryptography Extensions (CE) accelerated SHA-1 implementation with the
 * Linux kernel's generic cryptographic API (`shash`). It handles the crucial
 * registration of the hardware-accelerated algorithm, allowing the kernel to
 * automatically select this high-performance path on compatible hardware.
 * Furthermore, it ensures the safe and proper use of the NEON/FPU context
 * from kernel mode, preventing corruption of Floating-Point Unit state during
 * cryptographic operations.
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
#include <crypto/sha1_base.
#include <linux/cpufeature.h>
#include <linux/kernel.h>
#include <linux/module.h>

MODULE_DESCRIPTION("SHA1 secure hash using ARMv8 Crypto Extensions");
MODULE_AUTHOR("Ard Biesheuvel <ard.biesheuvel@linaro.org>");
MODULE_LICENSE("GPL v2");

/**
 * @brief Processes SHA-1 blocks using ARMv8 Crypto Extensions.
 * @details This function serves as the high-level interface to the underlying
 * assembly implementation (`sha1-ce-core.S`), which directly utilizes ARMv8
 * Cryptography Extension hardware instructions to perform SHA-1 compression.
 * By delegating to this optimized assembly routine, the computation of SHA-1
 * hash blocks is significantly accelerated.
 *
 * @param sst    Pointer to the SHA-1 state structure.
 * @param src    Pointer to the source data.
 * @param blocks Number of 64-byte blocks to process.
 * Functional Utility: Offloads SHA-1 block transformation to hardware-accelerated ARMv8 CE instructions.
 */
asmlinkage void sha1_ce_transform(struct sha1_state *sst, u8 const *src,
				  int blocks);

/**
 * @brief Implements the shash 'update' operation using ARMv8 CE.
 * @details This function integrates the hardware-accelerated SHA-1 block processing
 * into the kernel's generic cryptographic update mechanism. It is critical
 * for safely managing the NEON/FPU context. Before invoking the core
 * hardware-accelerated transformation, `kernel_neon_begin()` is called to
 * save the current FPU state. After the transformation, `kernel_neon_end()`
 * restores the FPU state, ensuring that cryptographic operations do not
 * interfere with other kernel components that might also use the FPU.
 *
 * @param desc The shash descriptor containing the hash state.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return The number of bytes not processed (if any).
 * Functional Utility: Processes data incrementally for SHA-1 hashing, safely integrating ARMv8 CE hardware acceleration with kernel FPU context management.
 */
static int sha1_ce_update(struct shash_desc *desc, const u8 *data,
			  unsigned int len)
{
	int remain;

	// Pre-condition: Saves the current FPU state and enables the kernel for NEON/CE operations.
	kernel_neon_begin();
	// Functional Utility: Delegates the bulk of data processing to the hardware-accelerated SHA-1 transform assembly function.
	remain = sha1_base_do_update_blocks(desc, data, len, sha1_ce_transform);
	// Post-condition: Restores the FPU state, making it available for other kernel tasks.
	kernel_neon_end();

	return remain;
}

/**
 * @brief Implements the shash 'finup' (finalize and update) operation using CE.
 * @details This function handles the processing of the final data segment and
 * the computation of the complete SHA-1 hash digest. It ensures that any
 * remaining data is incorporated into the hash, padding is applied correctly,
 * and the final hash value is produced. Crucially, it manages the NEON/FPU
 * context by calling `kernel_neon_begin()` and `kernel_neon_end()` to maintain
 * system stability during hardware-accelerated operations.
 *
 * @param desc The shash descriptor.
 * @param data The final data chunk to be hashed.
 * @param len  The length of the final data.
 * @param out  The buffer to store the resulting 20-byte hash.
 * @return 0 on success.
 * Functional Utility: Completes the SHA-1 hashing process for the final data block, including padding and final digest generation, while ensuring safe FPU context management.
 */
static int sha1_ce_finup(struct shash_desc *desc, const u8 *data,
			 unsigned int len, u8 *out)
{
	// Pre-condition: Saves the FPU state, enabling safe use of NEON/CE for the final cryptographic operations.
	kernel_neon_begin();
	// Functional Utility: Processes the last data block and applies SHA-1 padding using the hardware-accelerated transform.
	sha1_base_do_finup(desc, data, len, sha1_ce_transform);
	// Post-condition: Restores the FPU state after the hardware-accelerated operations are complete.
	kernel_neon_end();

	return sha1_base_finish(desc, out);
}

/**
 * @brief Defines the CE-accelerated SHA-1 algorithm for the kernel's cryptographic API.
 * @details This structure registers the SHA-1 algorithm implementation that leverages
 * ARMv8 Cryptography Extensions. By assigning a high priority (`cra_priority = 200`),
 * this hardware-accelerated version is preferentially selected over less optimized
 * or generic software implementations on CPUs that support the CE features. This
 * ensures that cryptographic operations automatically benefit from available
 * hardware acceleration.
 * Functional Utility: Registers the hardware-accelerated SHA-1 algorithm implementation with the kernel, making it discoverable and prioritizable by the cryptographic API.
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
 * @brief Module initialization function for the ARMv8 CE SHA-1 glue code.
 * @details This function is invoked when the kernel module is loaded.
 * Functional Utility: Registers the hardware-accelerated SHA-1 algorithm (`alg` structure)
 * with the Linux kernel's cryptographic API. This makes the `sha1-ce` driver available
 * to the system, allowing applications and other kernel components to utilize
 * the ARMv8 Cryptography Extensions for efficient SHA-1 hashing. The registration
 * is contingent on the presence of the necessary CPU features, as determined by `module_cpu_feature_match`.
 * @return 0 on successful registration, or an error code on failure.
 */
static int __init sha1_ce_mod_init(void)
{
	return crypto_register_shash(&alg);
}

/**
 * @brief Module cleanup function for the ARMv8 CE SHA-1 glue code.
 * @details This function is invoked when the kernel module is unloaded.
 * Functional Utility: Unregisters the hardware-accelerated SHA-1 algorithm (`alg` structure)
 * from the Linux kernel's cryptographic API. This cleanly removes the `sha1-ce`
 * driver from the system, releasing associated resources and preventing any
 * lingering references after the module is no longer in use.
 */
static void __exit sha1_ce_mod_fini(void)
{
	crypto_unregister_shash(&alg);
}

/**
 * @brief Macro for conditional module loading based on CPU features.
 * @details This macro plays a crucial role in ensuring that the `sha1-ce` module
 * is only loaded and initialized on ARMv8 systems that possess the required
 * Cryptography Extensions for SHA-1. It effectively creates a module alias
 * that links the module to the presence of specific CPU features. This mechanism
 * allows the kernel to dynamically load the hardware-accelerated SHA-1
 * implementation automatically when compatible hardware is detected, and prevents
 * loading on unsupported systems, enhancing system stability and resource management.
 * It connects `sha1_ce_mod_init` with the SHA1 CPU feature.
 */
module_cpu_feature_match(SHA1, sha1_ce_mod_init);
module_exit(sha1_ce_mod_fini);
