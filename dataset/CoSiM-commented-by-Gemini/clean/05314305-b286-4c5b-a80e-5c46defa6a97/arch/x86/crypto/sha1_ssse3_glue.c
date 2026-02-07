// SPDX-License-Identifier: GPL-2.0-or-later
/**
 * @file sha1_ssse3_glue.c
 * @brief Glue code for x86-accelerated SHA-1 implementations leveraging SSSE3, AVX, AVX2, and SHA-NI.
 * @details This file provides the C-level interface (glue code) to integrate
 * multiple highly optimized SHA-1 (Secure Hash Algorithm 1) implementations,
 * specifically tailored for x86 processors with various SIMD extensions
 * (SSSE3, AVX, AVX2, and SHA-NI). Its primary responsibilities include:
 * - **Dynamic Feature Detection**: Probing the CPU at module initialization
 *   to determine the highest available SHA-1 acceleration technology.
 * - **Adaptive Dispatch**: Providing a common interface that dynamically
 *   dispatches to the most performant assembly routine (SSSE3, AVX, AVX2, or SHA-NI)
 *   based on CPU capabilities and data block characteristics.
 * - **FPU/SIMD Context Management**: Ensuring proper saving and restoring of
 *   FPU/SIMD state around cryptographic operations to comply with kernel
 *   conventions (`kernel_fpu_begin`/`kernel_fpu_end`).
 * - **Integration with Crypto API**: Registering multiple `shash` algorithms
 *   with appropriate priorities, allowing the kernel to automatically select
 *   the best available SHA-1 implementation.
 * This module aims to provide the fastest possible SHA-1 hashing on a wide
 * range of x86 CPUs within the Linux kernel by intelligently utilizing hardware capabilities.
 */
/*
 * Cryptographic API.
 *
 * Glue code for the SHA1 Secure Hash Algorithm assembler implementations
 * using SSSE3, AVX, AVX2, and SHA-NI instructions.
 *
 * This file is based on sha1_generic.c
 *
 * Copyright (c) Alan Smithee.
 * Copyright (c) Andrew McDonald <andrew@mcdonald.org.uk>
 * Copyright (c) Jean-Francois Dive <jef@linuxbe.org>
 * Copyright (c) Mathias Krause <minipli@googlemail.com>
 * Copyright (c) Chandramouli Narayanan <mouli@linux.intel.com>
 */

#define pr_fmt(fmt)	KBUILD_MODNAME ": " fmt

#include <asm/cpu_device_id.h>
#include <asm/simd.h>
#include <crypto/internal/hash.h>
#include <crypto/sha1.h>
#include <crypto/sha1_base.h>
#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/module.

/**
 * @brief List of x86 CPU features supported by this module.
 * @details This array defines the CPU features that this module can leverage
 * for SHA-1 acceleration, ordered by general performance (SHA_NI > AVX2 > AVX > SSSE3).
 * It is used by `MODULE_DEVICE_TABLE` to allow the kernel to automatically
 * load this module when a compatible CPU is detected.
 * Functional Role: Specifies the CPU capabilities required for optimal SHA-1 acceleration by this module.
 */
static const struct x86_cpu_id module_cpu_ids[] = {
	X86_MATCH_FEATURE(X86_FEATURE_SHA_NI, NULL), // Functional Role: Match CPU with SHA New Instructions.
	X86_MATCH_FEATURE(X86_FEATURE_AVX2, NULL), // Functional Role: Match CPU with AVX2.
	X86_MATCH_FEATURE(X86_FEATURE_AVX, NULL), // Functional Role: Match CPU with AVX.
	X86_MATCH_FEATURE(X86_FEATURE_SSSE3, NULL), // Functional Role: Match CPU with SSSE3.
	{} // Functional Role: Terminator for the array.
};
MODULE_DEVICE_TABLE(x86cpu, module_cpu_ids);

/**
 * @brief Common helper function for the `shash` 'update' operation.
 * @details This inline function provides a unified interface for the `shash`
 * 'update' callback across different x86 SHA-1 optimized versions. It ensures
 * proper FPU/SIMD context management by wrapping the `sha1_base_do_update_blocks`
 * call with `kernel_fpu_begin()` and `kernel_fpu_end()`. It also includes a
 * compile-time assertion to ensure that `struct sha1_state` has the expected
 * memory layout for the assembly routines.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The input data to be hashed.
 * @param len The length of the input data.
 * @param sha1_xform A function pointer to the specific SHA-1 block transform assembly routine.
 * @return The number of remaining bytes after processing full blocks.
 * Functional Utility: Handles FPU context and dispatches to the appropriate SHA-1 assembly transform for data updates.
 */
static inline int sha1_update(struct shash_desc *desc, const u8 *data,
			      unsigned int len, sha1_block_fn *sha1_xform)
{
	int remain;

	/*
	 * Functional Utility: Compile-time assertion to ensure that `struct sha1_state`
	 * begins directly with its 160-bit internal state (an array of 5 `u32`),
	 * as this is a strict requirement for the assembly functions to correctly
	 * access and manipulate the hash state. This guarantees memory layout compatibility.
	 */
	BUILD_BUG_ON(offsetof(struct sha1_state, state) != 0);

	kernel_fpu_begin(); // Functional Utility: Saves the FPU/SIMD state and enables its use in kernel mode.
	// Functional Utility: Delegates data buffering and block processing to the base SHA-1 update function,
	// using the provided `sha1_xform` for core block transformation.
	remain = sha1_base_do_update_blocks(desc, data, len, sha1_xform);
	kernel_fpu_end(); // Functional Utility: Restores the FPU/SIMD state and disables its use in kernel mode.

	return remain;
}

/**
 * @brief Common helper function for the `shash` 'finup' operation.
 * @details This inline function provides a unified interface for the `shash`
 * 'finup' callback across different x86 SHA-1 optimized versions. It ensures
 * proper FPU/SIMD context management by wrapping the `sha1_base_do_finup`
 * and `sha1_base_finish` calls with `kernel_fpu_begin()` and `kernel_fpu_end()`.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The final input data to be hashed.
 * @param len The length of the final input data.
 * @param out The buffer to store the final 20-byte SHA-1 hash digest.
 * @param sha1_xform A function pointer to the specific SHA-1 block transform assembly routine.
 * @return 0 on success.
 * Functional Utility: Handles FPU context and dispatches to the appropriate SHA-1 assembly transform for finalization.
 */
static inline int sha1_finup(struct shash_desc *desc, const u8 *data,
			     unsigned int len, u8 *out,
			     sha1_block_fn *sha1_xform)
{
	kernel_fpu_begin(); // Functional Utility: Saves the FPU/SIMD state and enables its use in kernel mode.
	// Functional Utility: Processes any remaining input data and applies SHA-1 padding using the provided `sha1_xform`.
	sha1_base_do_finup(desc, data, len, sha1_xform);
	kernel_fpu_end(); // Functional Utility: Restores the FPU/SIMD state and disables its use in kernel mode.

	// Functional Utility: Writes the final computed SHA-1 hash digest to the output buffer.
	return sha1_base_finish(desc, out);
}

/**
 * @brief External declaration for the SSSE3-optimized SHA-1 transform function.
 * @details This function is an `asmlinkage` declaration, indicating that its
 * implementation is provided in a separate assembly file (`sha1_ssse3_asm.S`).
 * It performs the core SHA-1 block transformation for a 64-byte message block,
 * leveraging Intel SSSE3 instructions.
 *
 * @param state Pointer to the SHA-1 state array.
 * @param data Pointer to the input data blocks.
 * @param blocks The number of 64-byte blocks to process.
 * Functional Utility: Dispatches to the SSSE3 assembly routine for hardware-accelerated SHA-1 block processing.
 */
asmlinkage void sha1_transform_ssse3(struct sha1_state *state,
				     const u8 *data, int blocks);

/**
 * @brief `shash` 'update' callback for SSSE3-accelerated SHA-1.
 * @details This function is the 'update' entry point for the SSSE3-optimized
 * SHA-1 algorithm. It calls the common `sha1_update` helper, passing the
 * `sha1_transform_ssse3` assembly routine as the block transformation function.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The input data to be hashed.
 * @param len The length of the input data.
 * @return The number of remaining bytes.
 * Functional Utility: Updates the SHA-1 hash state using the SSSE3-optimized assembly transform.
 */
static int sha1_ssse3_update(struct shash_desc *desc, const u8 *data,
			     unsigned int len)
{
	return sha1_update(desc, data, len, sha1_transform_ssse3);
}

/**
 * @brief `shash` 'finup' callback for SSSE3-accelerated SHA-1.
 * @details This function is the 'finup' entry point for the SSSE3-optimized
 * SHA-1 algorithm. It calls the common `sha1_finup` helper, passing the
 * `sha1_transform_ssse3` assembly routine as the block transformation function.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The final input data to be hashed.
 * @param len The length of the final input data.
 * @param out The buffer to store the final SHA-1 digest.
 * @return 0 on success.
 * Functional Utility: Finalizes SHA-1 computation using the SSSE3-optimized assembly transform.
 */
static int sha1_ssse3_finup(struct shash_desc *desc, const u8 *data,
			      unsigned int len, u8 *out)
{
	return sha1_finup(desc, data, len, out, sha1_transform_ssse3);
}

/**
 * @brief Defines the SSSE3-accelerated SHA-1 algorithm for the crypto API.
 * @details This structure registers the SHA-1 algorithm implementation that
 * leverages Intel SSSE3 instructions. It specifies the algorithm's properties
 * (digest size, block size), associates the core operations (`init`, `update`, `finup`)
 * with their respective handler functions. A `cra_priority` of 150 indicates
 * a mid-level priority, typically higher than generic software but lower than
 * more advanced SIMD or dedicated SHA-NI implementations.
 * Functional Utility: Registers the SSSE3-optimized SHA-1 algorithm with the kernel crypto API.
 */
static struct shash_alg sha1_ssse3_alg = {
	.digestsize	=	SHA1_DIGEST_SIZE,
	.init		=	sha1_base_init,
	.update		=	sha1_ssse3_update,
	.finup		=	sha1_ssse3_finup,
	.descsize	=	SHA1_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha1",
		.cra_driver_name =	"sha1-ssse3",
		.cra_priority	=	150, // Functional Utility: Priority for SSSE3 implementation.
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY,
		.cra_blocksize	=	SHA1_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
};

/**
 * @brief Conditionally registers the SSSE3-accelerated SHA-1 algorithm.
 * @details This function attempts to register the `sha1-ssse3` algorithm
 * with the crypto API. It first checks if the CPU supports the `X86_FEATURE_SSSE3`
 * feature using `boot_cpu_has()`. The registration proceeds only if the feature
 * is present.
 *
 * @return 0 on success, or a negative error code if registration fails or feature is absent.
 * Functional Utility: Registers the SSSE3 SHA-1 algorithm if the CPU feature is present.
 */
static int register_sha1_ssse3(void)
{
	if (boot_cpu_has(X86_FEATURE_SSSE3))
		return crypto_register_shash(&sha1_ssse3_alg);
	return 0; // Functional Utility: Returns 0 if SSSE3 is not available (no registration attempted).
}

/**
 * @brief Conditionally unregisters the SSSE3-accelerated SHA-1 algorithm.
 * @details This function attempts to unregister the `sha1-ssse3` algorithm
 * from the crypto API. It first checks if the CPU supports the `X86_FEATURE_SSSE3`
 * feature. The unregistration proceeds only if the feature is present, ensuring
 * that only registered algorithms are unregistered.
 * Functional Utility: Unregisters the SSSE3 SHA-1 algorithm if it was potentially registered.
 */
static void unregister_sha1_ssse3(void)
{
	if (boot_cpu_has(X86_FEATURE_SSSE3))
		crypto_unregister_shash(&sha1_ssse3_alg);
}

/**
 * @brief External declaration for the AVX-optimized SHA-1 transform function.
 * @details This function is an `asmlinkage` declaration, indicating that its
 * implementation is provided in a separate assembly file (e.g., `sha1_ssse3_asm.S`
 * or a dedicated AVX file). It performs the core SHA-1 block transformation
 * leveraging Intel AVX instructions.
 *
 * @param state Pointer to the SHA-1 state array.
 * @param data Pointer to the input data blocks.
 * @param blocks The number of 64-byte blocks to process.
 * Functional Utility: Dispatches to the AVX assembly routine for hardware-accelerated SHA-1 block processing.
 */
asmlinkage void sha1_transform_avx(struct sha1_state *state,
				   const u8 *data, int blocks);

/**
 * @brief `shash` 'update' callback for AVX-accelerated SHA-1.
 * @details This function is the 'update' entry point for the AVX-optimized
 * SHA-1 algorithm. It calls the common `sha1_update` helper, passing the
 * `sha1_transform_avx` assembly routine as the block transformation function.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The input data to be hashed.
 * @param len The length of the input data.
 * @return The number of remaining bytes.
 * Functional Utility: Updates the SHA-1 hash state using the AVX-optimized assembly transform.
 */
static int sha1_avx_update(struct shash_desc *desc, const u8 *data,
			     unsigned int len)
{
	return sha1_update(desc, data, len, sha1_transform_avx);
}

/**
 * @brief `shash` 'finup' callback for AVX-accelerated SHA-1.
 * @details This function is the 'finup' entry point for the AVX-optimized
 * SHA-1 algorithm. It calls the common `sha1_finup` helper, passing the
 * `sha1_transform_avx` assembly routine as the block transformation function.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The final input data to be hashed.
 * @param len The length of the final input data.
 * @param out The buffer to store the final SHA-1 digest.
 * @return 0 on success.
 * Functional Utility: Finalizes SHA-1 computation using the AVX-optimized assembly transform.
 */
static int sha1_avx_finup(struct shash_desc *desc, const u8 *data,
			      unsigned int len, u8 *out)
{
	return sha1_finup(desc, data, len, out, sha1_transform_avx);
}

/**
 * @brief Defines the AVX-accelerated SHA-1 algorithm for the crypto API.
 * @details This structure registers the SHA-1 algorithm implementation that
 * leverages Intel AVX instructions. It specifies the algorithm's properties
 * (digest size, block size), associates the core operations (`init`, `update`, `finup`)
 * with their respective handler functions. A `cra_priority` of 160 indicates
 * a higher priority than SSSE3 but lower than AVX2 or SHA-NI.
 * Functional Utility: Registers the AVX-optimized SHA-1 algorithm with the kernel crypto API.
 */
static struct shash_alg sha1_avx_alg = {
	.digestsize	=	SHA1_DIGEST_SIZE,
	.init		=	sha1_base_init,
	.update		=	sha1_avx_update,
	.finup		=	sha1_avx_finup,
	.descsize	=	SHA1_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha1",
		.cra_driver_name =	"sha1-avx",
		.cra_priority	=	160, // Functional Utility: Priority for AVX implementation.
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY,
		.cra_blocksize	=	SHA1_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
};

/**
 * @brief Checks if AVX instructions are usable in the current kernel environment.
 * @details This function determines if AVX instructions can be safely used.
 * It checks if the CPU has the necessary XFEATURE_MASK_SSE and XFEATURE_MASK_YMM
 * features enabled in the CPU's XCR0 register, which dictates the FPU state
 * saving mechanism. If AVX is detected but unusable (e.g., due to missing
 * kernel support for saving/restoring YMM state), a warning message is logged.
 *
 * @return `true` if AVX is usable, `false` otherwise.
 * Functional Utility: Verifies both CPU support and kernel enablement for AVX instruction set usage.
 */
static bool avx_usable(void)
{
	// Functional Utility: Checks if the kernel has set up extended FPU state saving for SSE and YMM registers.
	if (!cpu_has_xfeatures(XFEATURE_MASK_SSE | XFEATURE_MASK_YMM, NULL)) {
		// Functional Utility: Logs a warning if AVX is present on the CPU but cannot be used by the kernel.
		if (boot_cpu_has(X86_FEATURE_AVX))
			pr_info("AVX detected but unusable.
");
		return false;
	}

	return true;
}

/**
 * @brief Conditionally registers the AVX-accelerated SHA-1 algorithm.
 * @details This function attempts to register the `sha1-avx` algorithm
 * with the crypto API. It first checks if AVX instructions are usable
 * (`avx_usable()`). The registration proceeds only if AVX is usable.
 *
 * @return 0 on success, or a negative error code if registration fails or AVX is unusable.
 * Functional Utility: Registers the AVX SHA-1 algorithm if the CPU supports it and it's usable.
 */
static int register_sha1_avx(void)
{
	if (avx_usable())
		return crypto_register_shash(&sha1_avx_alg);
	return 0; // Functional Utility: Returns 0 if AVX is not usable (no registration attempted).
}

/**
 * @brief Conditionally unregisters the AVX-accelerated SHA-1 algorithm.
 * @details This function attempts to unregister the `sha1-avx` algorithm
 * from the crypto API. It checks if AVX was usable, ensuring that only
 * registered algorithms are unregistered.
 * Functional Utility: Unregisters the AVX SHA-1 algorithm if it was potentially registered.
 */
static void unregister_sha1_avx(void)
{
	if (avx_usable())
		crypto_unregister_shash(&sha1_avx_alg);
}

#define SHA1_AVX2_BLOCK_OPTSIZE	4	/* Functional Role: Optimal number of 64-byte SHA-1 blocks to process per AVX2 transform call. */

/**
 * @brief External declaration for the AVX2-optimized SHA-1 transform function.
 * @details This function is an `asmlinkage` declaration, indicating that its
 * implementation is provided in a separate assembly file (`sha1_avx2_x86_64_asm.S`).
 * It performs the core SHA-1 block transformation leveraging Intel AVX2 instructions.
 *
 * @param state Pointer to the SHA-1 state array.
 * @param data Pointer to the input data blocks.
 * @param blocks The number of 64-byte blocks to process.
 * Functional Utility: Dispatches to the AVX2 assembly routine for hardware-accelerated SHA-1 block processing.
 */
asmlinkage void sha1_transform_avx2(struct sha1_state *state,
				    const u8 *data, int blocks);

/**
 * @brief Checks if AVX2 instructions, along with BMI1 and BMI2, are usable.
 * @details This function determines if AVX2 instructions can be safely used
 * for SHA-1 acceleration. It relies on `avx_usable()` for base AVX checks
 * and additionally verifies the presence of `X86_FEATURE_AVX2`, `X86_FEATURE_BMI1`,
 * and `X86_FEATURE_BMI2` via `boot_cpu_has()`.
 *
 * @return `true` if AVX2, BMI1, and BMI2 are all supported and usable, `false` otherwise.
 * Functional Utility: Verifies CPU support and kernel enablement for AVX2, BMI1, and BMI2 instruction sets.
 */
static bool avx2_usable(void)
{
	// Functional Utility: Checks for base AVX usability, plus AVX2, BMI1, and BMI2 features.
	if (avx_usable() && boot_cpu_has(X86_FEATURE_AVX2)
		&& boot_cpu_has(X86_FEATURE_BMI1)
		&& boot_cpu_has(X86_FEATURE_BMI2))
		return true;

	return false;
}

/**
 * @brief Applies the optimal SHA-1 transform (AVX2 or AVX) based on block count.
 * @details This inline function serves as a dispatcher to select between
 * `sha1_transform_avx2` and `sha1_transform_avx`. If the number of blocks
 * (`blocks`) is greater than or equal to `SHA1_AVX2_BLOCK_OPTSIZE`, the AVX2
 * transform is preferred. Otherwise, it falls back to the AVX transform, which
 * might be more efficient for smaller block counts due to AVX2's setup overheads.
 *
 * @param state Pointer to the SHA-1 state array.
 * @param data Pointer to the input data blocks.
 * @param blocks The number of 64-byte blocks to process.
 * Functional Utility: Dynamically selects between AVX2 and AVX optimized transforms based on input block size.
 */
static inline void sha1_apply_transform_avx2(struct sha1_state *state,
					     const u8 *data, int blocks)
{
	/* Functional Utility: Selects the optimal transform based on data block size. */
	if (blocks >= SHA1_AVX2_BLOCK_OPTSIZE)
		sha1_transform_avx2(state, data, blocks);
	else
		sha1_transform_avx(state, data, blocks);
}

/**
 * @brief `shash` 'update' callback for AVX2-accelerated SHA-1.
 * @details This function is the 'update' entry point for the AVX2-optimized
 * SHA-1 algorithm. It calls the common `sha1_update` helper, passing the
 * `sha1_apply_transform_avx2` dispatcher as the block transformation function,
 * allowing adaptive selection between AVX2 and AVX.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The input data to be hashed.
 * @param len The length of the input data.
 * @return The number of remaining bytes.
 * Functional Utility: Updates the SHA-1 hash state using AVX2-optimized transform, falling back to AVX for small blocks.
 */
static int sha1_avx2_update(struct shash_desc *desc, const u8 *data,
			     unsigned int len)
{
	return sha1_update(desc, data, len, sha1_apply_transform_avx2);
}

/**
 * @brief `shash` 'finup' callback for AVX2-accelerated SHA-1.
 * @details This function is the 'finup' entry point for the AVX2-optimized
 * SHA-1 algorithm. It calls the common `sha1_finup` helper, passing the
 * `sha1_apply_transform_avx2` dispatcher as the block transformation function.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The final input data to be hashed.
 * @param len The length of the final input data.
 * @param out The buffer to store the final SHA-1 digest.
 * @return 0 on success.
 * Functional Utility: Finalizes SHA-1 computation using AVX2-optimized transform, falling back to AVX for small blocks.
 */
static int sha1_avx2_finup(struct shash_desc *desc, const u8 *data,
			      unsigned int len, u8 *out)
{
	return sha1_finup(desc, data, len, out, sha1_apply_transform_avx2);
}

/**
 * @brief Defines the AVX2-accelerated SHA-1 algorithm for the crypto API.
 * @details This structure registers the SHA-1 algorithm implementation that
 * leverages Intel AVX2 instructions. It specifies the algorithm's properties
 * (digest size, block size), associates the core operations (`init`, `update`, `finup`)
 * with their respective handler functions. A `cra_priority` of 170 indicates
 * a higher priority than AVX but lower than SHA-NI.
 * Functional Utility: Registers the AVX2-optimized SHA-1 algorithm with the kernel crypto API.
 */
static struct shash_alg sha1_avx2_alg = {
	.digestsize	=	SHA1_DIGEST_SIZE,
	.init		=	sha1_base_init,
	.update		=	sha1_avx2_update,
	.finup		=	sha1_avx2_finup,
	.descsize	=	SHA1_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha1",
		.cra_driver_name =	"sha1-avx2",
		.cra_priority	=	170, // Functional Utility: Priority for AVX2 implementation.
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY,
		.cra_blocksize	=	SHA1_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
};

/**
 * @brief Conditionally registers the AVX2-accelerated SHA-1 algorithm.
 * @details This function attempts to register the `sha1-avx2` algorithm
 * with the crypto API. It first checks if AVX2, BMI1, and BMI2 instructions
 * are usable (`avx2_usable()`). The registration proceeds only if these
 * features are present and usable.
 *
 * @return 0 on success, or a negative error code if registration fails or features are absent.
 * Functional Utility: Registers the AVX2 SHA-1 algorithm if the CPU features are present and usable.
 */
static int register_sha1_avx2(void)
{
	if (avx2_usable())
		return crypto_register_shash(&sha1_avx2_alg);
	return 0; // Functional Utility: Returns 0 if AVX2 is not usable (no registration attempted).
}

/**
 * @brief Conditionally unregisters the AVX2-accelerated SHA-1 algorithm.
 * @details This function attempts to unregister the `sha1-avx2` algorithm
 * from the crypto API. It checks if AVX2 was usable, ensuring that only
 * registered algorithms are unregistered.
 * Functional Utility: Unregisters the AVX2 SHA-1 algorithm if it was potentially registered.
 */
static void unregister_sha1_avx2(void)
{
	if (avx2_usable())
		crypto_unregister_shash(&sha1_avx2_alg);
}

/**
 * @brief External declaration for the SHA-NI-optimized SHA-1 transform function.
 * @details This function is an `asmlinkage` declaration, indicating that its
 * implementation is provided in a separate assembly file (`sha1_ni_asm.S`).
 * It performs the core SHA-1 block transformation leveraging Intel SHA Extensions.
 *
 * @param digest Pointer to the SHA-1 state array.
 * @param data Pointer to the input data blocks.
 * @param rounds The number of 64-byte blocks to process.
 * Functional Utility: Dispatches to the SHA-NI assembly routine for hardware-accelerated SHA-1 block processing.
 */
asmlinkage void sha1_ni_transform(struct sha1_state *digest, const u8 *data,
				  int rounds);

/**
 * @brief `shash` 'update' callback for SHA-NI-accelerated SHA-1.
 * @details This function is the 'update' entry point for the SHA-NI-optimized
 * SHA-1 algorithm. It calls the common `sha1_update` helper, passing the
 * `sha1_ni_transform` assembly routine as the block transformation function.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The input data to be hashed.
 * @param len The length of the input data.
 * @return The number of remaining bytes.
 * Functional Utility: Updates the SHA-1 hash state using the SHA-NI-optimized assembly transform.
 */
static int sha1_ni_update(struct shash_desc *desc, const u8 *data,
			     unsigned int len)
{
	return sha1_update(desc, data, len, sha1_ni_transform);
}

/**
 * @brief `shash` 'finup' callback for SHA-NI-accelerated SHA-1.
 * @details This function is the 'finup' entry point for the SHA-NI-optimized
 * SHA-1 algorithm. It calls the common `sha1_finup` helper, passing the
 * `sha1_ni_transform` assembly routine as the block transformation function.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The final input data to be hashed.
 * @param len The length of the final input data.
 * @param out The buffer to store the final SHA-1 digest.
 * @return 0 on success.
 * Functional Utility: Finalizes SHA-1 computation using the SHA-NI-optimized assembly transform.
 */
static int sha1_ni_finup(struct shash_desc *desc, const u8 *data,
			      unsigned int len, u8 *out)
{
	return sha1_finup(desc, data, len, out, sha1_ni_transform);
}

/**
 * @brief Defines the SHA-NI-accelerated SHA-1 algorithm for the crypto API.
 * @details This structure registers the SHA-1 algorithm implementation that
 * leverages Intel SHA Extensions (SHA-NI). It specifies the algorithm's properties
 * (digest size, block size), associates the core operations (`init`, `update`, `finup`)
 * with their respective handler functions. A `cra_priority` of 250 indicates
 * the highest priority among the x86 SIMD implementations, reflecting the
 * dedicated hardware acceleration provided by SHA-NI.
 * Functional Utility: Registers the SHA-NI-optimized SHA-1 algorithm with the kernel crypto API.
 */
static struct shash_alg sha1_ni_alg = {
	.digestsize	=	SHA1_DIGEST_SIZE,
	.init		=	sha1_base_init,
	.update		=	sha1_ni_update,
	.finup		=	sha1_ni_finup,
	.descsize	=	SHA1_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha1",
		.cra_driver_name =	"sha1-ni",
		.cra_priority	=	250, // Functional Utility: Highest priority for SHA-NI implementation.
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY,
		.cra_blocksize	=	SHA1_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
};

/**
 * @brief Conditionally registers the SHA-NI-accelerated SHA-1 algorithm.
 * @details This function attempts to register the `sha1-ni` algorithm
 * with the crypto API. It first checks if the CPU supports the `X86_FEATURE_SHA_NI`
 * feature using `boot_cpu_has()`. The registration proceeds only if the feature
 * is present.
 *
 * @return 0 on success, or a negative error code if registration fails or feature is absent.
 * Functional Utility: Registers the SHA-NI SHA-1 algorithm if the CPU feature is present.
 */
static int register_sha1_ni(void)
{
	if (boot_cpu_has(X86_FEATURE_SHA_NI))
		return crypto_register_shash(&sha1_ni_alg);
	return 0; // Functional Utility: Returns 0 if SHA-NI is not available (no registration attempted).
}

/**
 * @brief Conditionally unregisters the SHA-NI-accelerated SHA-1 algorithm.
 * @details This function attempts to unregister the `sha1-ni` algorithm
 * from the crypto API. It checks if the CPU supports the `X86_FEATURE_SHA_NI`
 * feature, ensuring that only registered algorithms are unregistered.
 * Functional Utility: Unregisters the SHA-NI SHA-1 algorithm if it was potentially registered.
 */
static void unregister_sha1_ni(void)
{
	if (boot_cpu_has(X86_FEATURE_SHA_NI))
		crypto_unregister_shash(&sha1_ni_alg);
}

/**
 * @brief Module initialization function for x86 SHA-1 glue code.
 * @details This function is the main entry point when the kernel module is loaded.
 * It attempts to register all available SHA-1 optimized versions (SSSE3, AVX, AVX2, SHA-NI)
 * with the Linux kernel's cryptographic API, in a prioritized order.
 * The order of registration is important: it tries to register the lowest
 * priority (SSSE3) first, then AVX, AVX2, and finally SHA-NI. If any registration
 * fails, it performs a clean unregistration of all previously successfully
 * registered algorithms. This ensures that only supported and correctly
 * registered algorithms are active.
 * Functional Utility: Registers available x86 SHA-1 optimized algorithms (SSSE3, AVX, AVX2, SHA-NI) with error handling.
 * @return 0 on successful registration of at least one algorithm, or `-ENODEV` if no algorithms could be registered.
 */
static int __init sha1_ssse3_mod_init(void)
{
	// Pre-condition: Checks if the current CPU matches any of the supported features.
	if (!x86_match_cpu(module_cpu_ids))
		return -ENODEV; // Functional Utility: Returns -ENODEV if no matching CPU features are found.

	// Functional Utility: Attempts to register SSSE3 SHA-1 algorithm. If it fails, jumps to cleanup.
	if (register_sha1_ssse3())
		goto fail;

	// Functional Utility: Attempts to register AVX SHA-1 algorithm. If it fails, unregisters SSSE3 and jumps to cleanup.
	if (register_sha1_avx()) {
		unregister_sha1_ssse3();
		goto fail;
	}

	// Functional Utility: Attempts to register AVX2 SHA-1 algorithm. If it fails, unregisters AVX and SSSE3, then jumps to cleanup.
	if (register_sha1_avx2()) {
		unregister_sha1_avx();
		unregister_sha1_ssse3();
		goto fail;
	}

	// Functional Utility: Attempts to register SHA-NI SHA-1 algorithm. If it fails, unregisters AVX2, AVX, and SSSE3, then jumps to cleanup.
	if (register_sha1_ni()) {
		unregister_sha1_avx2();
		unregister_sha1_avx();
		unregister_sha1_ssse3();
		goto fail;
	}

	return 0; // Functional Utility: Returns 0 if at least one algorithm was successfully registered.
fail:
	return -ENODEV; // Functional Utility: Returns -ENODEV if any registration failed.
}

/**
 * @brief Module cleanup function for x86 SHA-1 glue code.
 * @details This function is the exit point when the kernel module is unloaded.
 * It cleanly unregisters all potentially registered SHA-1 optimized versions
 * (SHA-NI, AVX2, AVX, SSSE3) from the Linux kernel's cryptographic API.
 * The unregistration order is the reverse of registration, ensuring a proper
 * teardown of the cryptographic interfaces.
 * Functional Utility: Unregisters all x86 SHA-1 optimized algorithms from the kernel crypto API.
 */
static void __exit sha1_ssse3_mod_fini(void)
{
	// Functional Utility: Unregisters SHA-NI algorithm.
	unregister_sha1_ni();
	// Functional Utility: Unregisters AVX2 algorithm.
	unregister_sha1_avx2();
	// Functional Utility: Unregisters AVX algorithm.
	unregister_sha1_avx();
	// Functional Utility: Unregisters SSSE3 algorithm.
	unregister_sha1_ssse3();
}

module_init(sha1_ssse3_mod_init);
module_exit(sha1_ssse3_mod_fini);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SHA1 Secure Hash Algorithm, Supplemental SSE3 accelerated");

MODULE_ALIAS_CRYPTO("sha1");
MODULE_ALIAS_CRYPTO("sha1-ssse3");
MODULE_ALIAS_CRYPTO("sha1-avx");
MODULE_ALIAS_CRYPTO("sha1-avx2");
MODULE_ALIAS_CRYPTO("sha1-ni");
