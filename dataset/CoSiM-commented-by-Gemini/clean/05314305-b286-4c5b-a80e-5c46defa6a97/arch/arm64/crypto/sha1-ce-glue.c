/**
 * @file sha1-ce-glue.c
 * @brief Glue for ARMv8 Crypto Extensions SHA-1 for AArch64.
 * @details This file provides the C-level glue code to integrate the ARMv8
 * Cryptography Extensions (CE) accelerated SHA-1 implementation with the Linux
 * kernel's cryptographic API on AArch64 systems. It is responsible for
 * registering the hardware-accelerated algorithm, managing the NEON/FPU context
 * safely, and implementing a key optimization to offload the entire finalization
 * process (including padding and length appending) to the hardware-accelerated
 * assembly code. This approach maximizes the use of ARMv8 CE for SHA-1 hashing,
 * improving performance and reducing CPU overhead.
 */
// SPDX-License-Identifier: GPL-2.0-only
/*
 * sha1-ce-glue.c - SHA-1 secure hash using ARMv8 Crypto Extensions
 *
 * Copyright (C) 2014 - 2017 Linaro Ltd <ard.biesheuvel@linaro.org>
 */

#include <asm/neon.h>
#include <asm/simd.h>
#include <crypto/internal/hash.h>
#include <crypto/internal/simd.h>
#include <crypto/sha1.h>
#include <crypto/sha1_base.h>
#include <linux/cpufeature.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/string.h>

MODULE_DESCRIPTION("SHA1 secure hash using ARMv8 Crypto Extensions");
MODULE_AUTHOR("Ard Biesheuvel <ard.biesheuvel@linaro.org>");
MODULE_LICENSE("GPL v2");
MODULE_ALIAS_CRYPTO("sha1");

/**
 * @struct sha1_ce_state
 * @brief Custom state structure for the SHA-1 CE implementation.
 * @details This structure extends the standard `sha1_state` with a `finalize`
 * flag. This flag is crucial for enabling a performance optimization: it is
 * used to signal to the underlying assembly code (`sha1-ce-core.S`) whether it
 * should perform the final padding and length appending required by the SHA-1
 * standard. By offloading this entire finalization process to the hardware-
 * accelerated assembly, the overall SHA-1 hashing performance is significantly
 * improved.
 */
struct sha1_ce_state {
	struct sha1_state	sst;
	u32			finalize;
};

extern const u32 sha1_ce_offsetof_count;
extern const u32 sha1_ce_offsetof_finalize;

/**
 * @brief Entry point for the ARMv8 CE assembly implementation of SHA-1.
 * @details This function is the primary interface to the highly optimized
 * ARMv8-A Cryptography Extensions (CE) assembly implementation of the SHA-1
 * compression function, defined in `sha1-ce-core.S`. It takes a custom SHA-1
 * state structure (`sha1_ce_state`), which includes a `finalize` flag. This
 * flag allows the assembly code to conditionally handle the final padding
 * and length appending, enabling a hardware-accelerated finalization process.
 *
 * @param sst    Pointer to the custom SHA-1 CE state.
 * @param src    Pointer to the source data.
 * @param blocks Number of blocks to process.
 * @return       Number of blocks remaining to be processed.
 * Functional Utility: Dispatches to the hardware-accelerated ARM64 CE assembly routine for SHA-1 block processing, with optional hardware finalization.
 */
asmlinkage int __sha1_ce_transform(struct sha1_ce_state *sst, u8 const *src,
				   int blocks);

/**
 * @brief C wrapper for the ARMv8 CE SHA-1 assembly transform function.
 * @details This function provides a safe C-language wrapper for calling the
 * hardware-accelerated `__sha1_ce_transform` assembly routine. Its primary
 * purpose is to manage the NEON/FPU context. It iteratively processes
 * data blocks, ensuring that `kernel_neon_begin()` is called before each
 * assembly call to save the FPU state, and `kernel_neon_end()` is called
 * afterward to restore it. This prevents FPU state corruption and ensures
 * safe execution within the kernel environment.
 *
 * @param sst    Pointer to the standard SHA-1 state.
 * @param src    Pointer to the source data.
 * @param blocks Number of blocks to process.
 * Functional Utility: Manages NEON/FPU context around hardware-accelerated SHA-1 assembly calls for safe and iterative block processing.
 */
static void sha1_ce_transform(struct sha1_state *sst, u8 const *src,
			      int blocks)
{
	// Block Logic: Iterates over the given number of blocks, calling the hardware-accelerated
	// transform for each, while carefully managing the NEON FPU context.
	while (blocks) {
		int rem;

		// Pre-condition: Saves the current FPU state and enables NEON operations within the kernel context.
		// Invariant: FPU state is preserved across the __sha1_ce_transform call.
		kernel_neon_begin();
		// Functional Utility: Invokes the hardware-accelerated SHA-1 block transform,
		// processing a portion of the input data.
		rem = __sha1_ce_transform(container_of(sst,
						       struct sha1_ce_state,
						       sst), src, blocks);
		// Post-condition: Restores the FPU state, making it safe for other kernel components.
		kernel_neon_end();
		// Functional Utility: Adjusts the source pointer and remaining block count based on blocks processed by the assembly.
		src += (blocks - rem) * SHA1_BLOCK_SIZE;
		blocks = rem;
	}
}

const u32 sha1_ce_offsetof_count = offsetof(struct sha1_ce_state, sst.count);
const u32 sha1_ce_offsetof_finalize = offsetof(struct sha1_ce_state, finalize);

/**
 * @brief Implements the shash 'update' operation for ARMv8 CE SHA-1.
 * @details This function integrates the ARMv8 CE accelerated SHA-1 into the
 * kernel's `shash` API for incremental updates. Crucially, it sets the
 * `finalize` flag in the custom `sha1_ce_state` to 0. This ensures that
 * during intermediate updates, the underlying assembly code (`__sha1_ce_transform`)
 * does not perform any premature finalization (padding and length appending),
 * reserving that for the explicit `finup` call.
 *
 * @param desc The shash descriptor.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return 0 on success.
 * Functional Utility: Updates the SHA-1 hash state incrementally, explicitly disabling hardware finalization for intermediate steps.
 */
static int sha1_ce_update(struct shash_desc *desc, const u8 *data,
			  unsigned int len)
{
	struct sha1_ce_state *sctx = shash_desc_ctx(desc);

	// Functional Utility: Explicitly sets the finalize flag to 0, ensuring that the assembly routine
	// does not perform finalization (padding/length appending) during intermediate updates.
	sctx->finalize = 0;
	// Functional Utility: Delegates data buffering and block processing to the base SHA-1 update function,
	// using the CE-accelerated transform.
	return sha1_base_do_update_blocks(desc, data, len, sha1_ce_transform);
}

/**
 * @brief Implements the shash 'finup' (finalize and update) operation for ARMv8 CE SHA-1.
 * @details This function provides an optimized finalization path for ARMv8 CE
 * accelerated SHA-1. It attempts to offload the entire finalization process
 * (including padding and length encoding) to the hardware-accelerated assembly
 * code (`__sha1_ce_transform`) if the input data meets specific criteria
 * (i.e., the remaining length is a multiple of the SHA-1 block size, meaning no
 * partial data from previous updates remains). If hardware finalization is not
 * possible due to partial data, it gracefully falls back to a software-based
 * finalization path using `sha1_base_do_finup`.
 *
 * @param desc The shash descriptor.
 * @param data The final data chunk.
 * @param len  The length of the final data.
 * @param out  The buffer for the resulting hash digest.
 * @return 0 on success.
 * Functional Utility: Optimizes the finalization of SHA-1 hashing by conditionally offloading padding and length appending to hardware.
 */
static int sha1_ce_finup(struct shash_desc *desc, const u8 *data,
			 unsigned int len, u8 *out)
{
	struct sha1_ce_state *sctx = shash_desc_ctx(desc);
	bool finalized = false;

	/*
	 * Optimization: Allow the assembly code to perform finalization if the
	 * input is at least one full block and there is no partial data from
	 * previous updates. This maximizes hardware acceleration.
	 */
	// Block Logic: Checks if the remaining data is suitable for hardware finalization.
	// Invariant: `finalized` will be true if hardware finalization is attempted.
	if (len >= SHA1_BLOCK_SIZE) {
		unsigned int remain = len - round_down(len, SHA1_BLOCK_SIZE);

		finalized = !remain; // Functional Utility: Determines if the data length is a perfect multiple of block size.
		sctx->finalize = finalized; // Functional Utility: Sets the flag to inform assembly to perform finalization.
		// Functional Utility: Processes full blocks, potentially including the final block with padding by assembly.
		sha1_base_do_update_blocks(desc, data, len, sha1_ce_transform);
		data += len - remain;
		len = remain;
	}
	// Fallback: If hardware finalization was not possible (due to partial data),
	// use the software-based path to handle padding and final block processing.
	// Invariant: If `finalized` is false, software finalization is performed.
	if (!finalized) {
		sctx->finalize = 0; // Functional Utility: Ensures assembly does not finalize if software path is used.
		// Functional Utility: Performs software-based padding and final update using the CE-accelerated transform.
		sha1_base_do_finup(desc, data, len, sha1_ce_transform);
	}
	// Functional Utility: Writes the final computed SHA-1 hash digest to the output buffer.
	return sha1_base_finish(desc, out);
}

/**
 * @brief Defines the CE-accelerated SHA-1 algorithm for the kernel's cryptographic API.
 * @details This structure registers the SHA-1 algorithm implementation that leverages
 * ARMv8 Cryptography Extensions. The `descsize` is specifically increased to
 * accommodate the custom `sha1_ce_state` structure, which includes the `finalize`
 * flag for optimization. The function pointers for `update` and `finup` are
 * set to the custom handlers (`sha1_ce_update`, `sha1_ce_finup`) that implement
 * the finalization offload logic and NEON/FPU context management. A `cra_priority`
 * of 200 ensures this hardware-accelerated version is preferred over generic
 * software implementations.
 * Functional Utility: Registers the hardware-accelerated SHA-1 algorithm implementation with the kernel, using a custom state structure to enable finalization offload.
 */
static struct shash_alg alg = {
	.init			= sha1_base_init,
	.update			= sha1_ce_update,
	.finup			= sha1_ce_finup,
	.descsize		= sizeof(struct sha1_ce_state),
	.statesize		= SHA1_STATE_SIZE,
	.digestsize		= SHA1_DIGEST_SIZE,
	.base			= {
		.cra_name		= "sha1",
		.cra_driver_name	= "sha1-ce",
		.cra_priority		= 200,
		.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
					  CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize		= SHA1_BLOCK_SIZE,
		.cra_module		= THIS_MODULE,
	}
};

/**
 * @brief Module initialization function for the ARM64 CE SHA-1 glue code.
 * @details This function is the entry point when the kernel module is loaded.
 * It registers the hardware-accelerated SHA-1 algorithm (`alg` structure)
 * with the Linux kernel's cryptographic API. This makes the `sha1-ce` driver
 * available to the system, allowing applications and other kernel components
 * to utilize the ARMv8 Cryptography Extensions for efficient SHA-1 hashing.
 * The loading is conditionally managed by `module_cpu_feature_match`, ensuring
 * it only occurs on compatible hardware.
 * Functional Utility: Registers the hardware-accelerated SHA-1 algorithm with the kernel crypto API, making it available for system use.
 * @return 0 on successful registration, or an error code on failure.
 */
static int __init sha1_ce_mod_init(void)
{
	return crypto_register_shash(&alg);
}

/**
 * @brief Module cleanup function for the ARM64 CE SHA-1 glue code.
 * @details This function is the exit point when the kernel module is unloaded.
 * Functional Utility: Unregisters the hardware-accelerated SHA-1 algorithm
 * (`alg` structure) from the Linux kernel's cryptographic API. This cleanly
 * removes the `sha1-ce` driver from the system, releasing associated resources
 * and preventing any lingering references after the module is no longer in use.
 * This ensures proper resource management upon module unload.
 */
static void __exit sha1_ce_mod_fini(void)
{
	crypto_unregister_shash(&alg);
}

/**
 * @brief Macro for conditional module loading based on CPU features.
 * @details This macro plays a crucial role in ensuring that the `sha1-ce` module
 * is only loaded and initialized on ARM64 systems that possess the required
 * SHA-1 Cryptography Extensions. It effectively creates a module alias that
 * links the module to the presence of specific CPU features. This mechanism
 * allows the kernel to dynamically load the hardware-accelerated SHA-1
 * implementation automatically when compatible hardware is detected, and prevents
 * loading on unsupported systems, thereby enhancing system stability and
 * optimizing resource utilization by avoiding unnecessary module loading.
 * It connects `sha1_ce_mod_init` with the SHA1 CPU feature.
 * Functional Utility: Ensures the SHA-1 CE module is loaded only on ARM64 CPUs with SHA-1 Cryptography Extensions.
 */
module_cpu_feature_match(SHA1, sha1_ce_mod_init);
module_exit(sha1_ce_mod_fini);
