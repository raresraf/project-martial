/*
 * Cryptographic API.
 *
 * Glue code for the SHA512 Secure Hash Algorithm assembler
 * implementation using supplemental SSE3 / AVX / AVX2 instructions.
 *
 * This file is based on sha512_generic.c
 *
 * Copyright (C) 2013 Intel Corporation
 * Author: Tim Chen <tim.c.chen@linux.intel.com>
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

/**
 * @file sha512_ssse3_glue.c
 * @brief Glue code for x86-accelerated SHA-512/384 implementations leveraging SSSE3, AVX, and AVX2.
 * @details This file provides the C-level interface (glue code) to integrate
 * multiple highly optimized SHA-512 and SHA-384 implementations, specifically
 * tailored for x86 processors with various SIMD extensions (SSSE3, AVX, and AVX2).
 * Its primary responsibilities include:
 * - **Dynamic Feature Detection**: Probing the CPU at module initialization
 *   to determine the highest available SHA-512/384 acceleration technology.
 * - **FPU/SIMD Context Management**: Ensuring proper saving and restoring of
 *   FPU/SIMD state around cryptographic operations to comply with kernel
 *   conventions (`kernel_fpu_begin`/`kernel_fpu_end`).
 * - **Integration with Crypto API**: Registering multiple `shash` algorithms
 *   with appropriate priorities, allowing the kernel to automatically select
 *   the best available SHA-512/384 implementation.
 * This module aims to provide the fastest possible SHA-512/384 hashing on a wide
 * range of x86 CPUs within the Linux kernel by intelligently utilizing hardware capabilities.
 */

#define pr_fmt(fmt)	KBUILD_MODNAME ": " fmt

#include <asm/cpu_device_id.h>
#include <asm/simd.h>
#include <crypto/internal/hash.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <crypto/sha2.h>
#include <crypto/sha512_base.h>

/**
 * @brief External declaration for the SSSE3-optimized SHA-512 transform function.
 * @details This function is an `asmlinkage` declaration, indicating that its
 * implementation is provided in a separate assembly file (e.g., `sha512-ssse3-asm.S`).
 * It performs the core SHA-512 block transformation for a 128-byte message block,
 * leveraging Intel SSSE3 instructions.
 *
 * @param state Pointer to the SHA-512 state array.
 * @param data Pointer to the input data blocks.
 * @param blocks The number of 128-byte blocks to process.
 * Functional Utility: Dispatches to the SSSE3 assembly routine for hardware-accelerated SHA-512 block processing.
 */
asmlinkage void sha512_transform_ssse3(struct sha512_state *state,
				       const u8 *data, int blocks);

/**
 * @brief Common helper function for the `shash` 'update' operation.
 * @details This inline function provides a unified interface for the `shash`
 * 'update' callback across different x86 SHA-512/384 optimized versions. It ensures
 * proper FPU/SIMD context management by wrapping the `sha512_base_do_update_blocks`
 * call with `kernel_fpu_begin()` and `kernel_fpu_end()`. It also includes a
 * compile-time assertion to ensure that `struct sha512_state` has the expected
 * memory layout for the assembly routines.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The input data to be hashed.
 * @param len The length of the input data.
 * @param sha512_xform A function pointer to the specific SHA-512 block transform assembly routine.
 * @return The number of remaining bytes after processing full blocks.
 * Functional Utility: Handles FPU context and dispatches to the appropriate SHA-512/384 assembly transform for data updates.
 */
static int sha512_update(struct shash_desc *desc, const u8 *data,
		       unsigned int len, sha512_block_fn *sha512_xform)
{
	int remain;

	/*
	 * Functional Utility: Compile-time assertion to ensure that `struct sha512_state`
	 * begins directly with its 512-bit internal state (an array of 8 `u64`),
	 * as this is a strict requirement for the assembly functions to correctly
	 * access and manipulate the hash state. This guarantees memory layout compatibility.
	 */
	BUILD_BUG_ON(offsetof(struct sha512_state, state) != 0);

	kernel_fpu_begin(); // Functional Utility: Saves the FPU/SIMD state and enables its use in kernel mode.
	// Functional Utility: Delegates data buffering and block processing to the base SHA-512 update function,
	// using the provided `sha512_xform` for core block transformation.
	remain = sha512_base_do_update_blocks(desc, data, len, sha512_xform);
	kernel_fpu_end(); // Functional Utility: Restores the FPU/SIMD state and disables its use in kernel mode.

	return remain;
}

/**
 * @brief Common helper function for the `shash` 'finup' operation.
 * @details This inline function provides a unified interface for the `shash`
 * 'finup' callback across different x86 SHA-512/384 optimized versions. It ensures
 * proper FPU/SIMD context management by wrapping the `sha512_base_do_finup`
 * and `sha512_base_finish` calls with `kernel_fpu_begin()` and `kernel_fpu_end()`.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The final input data to be hashed.
 * @param len The length of the final input data.
 * @param out The buffer to store the final SHA-512/384 hash digest.
 * @param sha512_xform A function pointer to the specific SHA-512 block transform assembly routine.
 * @return 0 on success.
 * Functional Utility: Handles FPU context and dispatches to the appropriate SHA-512/384 assembly transform for finalization.
 */
static int sha512_finup(struct shash_desc *desc, const u8 *data,
	      unsigned int len, u8 *out, sha512_block_fn *sha512_xform)
{
	kernel_fpu_begin(); // Functional Utility: Saves the FPU/SIMD state and enables its use in kernel mode.
	// Functional Utility: Processes any remaining input data and applies SHA-512/384 padding using the provided `sha512_xform`.
	sha512_base_do_finup(desc, data, len, sha512_xform);
	kernel_fpu_end(); // Functional Utility: Restores the FPU/SIMD state and disables its use in kernel mode.

	// Functional Utility: Writes the final computed SHA-512/384 hash digest to the output buffer.
	return sha512_base_finish(desc, out);
}

/**
 * @brief `shash` 'update' callback for SSSE3-accelerated SHA-512/384.
 * @details This function is the 'update' entry point for the SSSE3-optimized
 * SHA-512/384 algorithm. It calls the common `sha512_update` helper, passing the
 * `sha512_transform_ssse3` assembly routine as the block transformation function.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The input data to be hashed.
 * @param len The length of the input data.
 * @return The number of remaining bytes.
 * Functional Utility: Updates the SHA-512/384 hash state using the SSSE3-optimized assembly transform.
 */
static int sha512_ssse3_update(struct shash_desc *desc, const u8 *data,
		       unsigned int len)
{
	return sha512_update(desc, data, len, sha512_transform_ssse3);
}

/**
 * @brief `shash` 'finup' callback for SSSE3-accelerated SHA-512/384.
 * @details This function is the 'finup' entry point for the SSSE3-optimized
 * SHA-512/384 algorithm. It calls the common `sha512_finup` helper, passing the
 * `sha512_transform_ssse3` assembly routine as the block transformation function.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The final input data to be hashed.
 * @param len The length of the final input data.
 * @param out The buffer to store the final SHA-512/384 digest.
 * @return 0 on success.
 * Functional Utility: Finalizes SHA-512/384 computation using the SSSE3-optimized assembly transform.
 */
static int sha512_ssse3_finup(struct shash_desc *desc, const u8 *data,
	      unsigned int len, u8 *out)
{
	return sha512_finup(desc, data, len, out, sha512_transform_ssse3);
}

/**
 * @brief Defines the SSSE3-accelerated SHA-512 and SHA-384 algorithms for the crypto API.
 * @details This array of `shash_alg` structures registers both the SHA-512
 * and SHA-384 algorithm implementations that leverage Intel SSSE3 instructions.
 * It specifies the algorithms' properties (digest size, block size), associates
 * the core operations (`init`, `update`, `finup`) with their respective handler
 * functions. A `cra_priority` of 150 indicates a mid-level priority, typically
 * higher than generic software but lower than more advanced SIMD implementations.
 * Functional Utility: Registers SSSE3-optimized SHA-512 and SHA-384 algorithms with the kernel crypto API.
 */
static struct shash_alg sha512_ssse3_algs[] = { {
	.digestsize	=	SHA512_DIGEST_SIZE,
	.init		=	sha512_base_init,
	.update		=	sha512_ssse3_update,
	.finup		=	sha512_ssse3_finup,
	.descsize	=	SHA512_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha512",
		.cra_driver_name =	"sha512-ssse3",
		.cra_priority	=	150, // Functional Utility: Priority for SSSE3 SHA-512 implementation.
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY |
					CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize	=	SHA512_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
},  {
	.digestsize	=	SHA384_DIGEST_SIZE,
	.init		=	sha384_base_init,
	.update		=	sha512_ssse3_update,
	.finup		=	sha512_ssse3_finup,
	.descsize	=	SHA512_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha384",
		.cra_driver_name =	"sha384-ssse3",
		.cra_priority	=	150, // Functional Utility: Priority for SSSE3 SHA-384 implementation.
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY |
					CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize	=	SHA384_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
} };

/**
 * @brief Conditionally registers the SSSE3-accelerated SHA-512/384 algorithms.
 * @details This function attempts to register the `sha512-ssse3` and `sha384-ssse3`
 * algorithms with the crypto API. It first checks if the CPU supports the `X86_FEATURE_SSSE3`
 * feature using `boot_cpu_has()`. The registration proceeds only if the feature
 * is present.
 *
 * @return 0 on success, or a negative error code if registration fails or feature is absent.
 * Functional Utility: Registers the SSSE3 SHA-512/384 algorithms if the CPU feature is present.
 */
static int register_sha512_ssse3(void)
{
	if (boot_cpu_has(X86_FEATURE_SSSE3))
		return crypto_register_shashes(sha512_ssse3_algs,
			ARRAY_SIZE(sha512_ssse3_algs));
	return 0; // Functional Utility: Returns 0 if SSSE3 is not available (no registration attempted).
}

/**
 * @brief Conditionally unregisters the SSSE3-accelerated SHA-512/384 algorithms.
 * @details This function attempts to unregister the `sha512-ssse3` and `sha384-ssse3`
 * algorithms from the crypto API. It first checks if the CPU supports the `X86_FEATURE_SSSE3`
 * feature. The unregistration proceeds only if the feature is present, ensuring
 * that only registered algorithms are unregistered.
 * Functional Utility: Unregisters the SSSE3 SHA-512/384 algorithms if they were potentially registered.
 */
static void unregister_sha512_ssse3(void)
{
	if (boot_cpu_has(X86_FEATURE_SSSE3))
		crypto_unregister_shashes(sha512_ssse3_algs,
			ARRAY_SIZE(sha512_ssse3_algs));
}

/**
 * @brief External declaration for the AVX-optimized SHA-512 transform function.
 * @details This function is an `asmlinkage` declaration, indicating that its
 * implementation is provided in a separate assembly file (e.g., `sha512-avx-asm.S`).
 * It performs the core SHA-512 block transformation leveraging Intel AVX instructions.
 *
 * @param state Pointer to the SHA-512 state array.
 * @param data Pointer to the input data blocks.
 * @param blocks The number of 128-byte blocks to process.
 * Functional Utility: Dispatches to the AVX assembly routine for hardware-accelerated SHA-512 block processing.
 */
asmlinkage void sha512_transform_avx(struct sha512_state *state,
				     const u8 *data, int blocks);

/**
 * @brief Checks if AVX instructions are usable in the current kernel environment.
 * @details This function determines if AVX instructions can be safely used.
 * It checks if the CPU has the necessary `XFEATURE_MASK_SSE` and `XFEATURE_MASK_YMM`
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
 * @brief `shash` 'update' callback for AVX-accelerated SHA-512/384.
 * @details This function is the 'update' entry point for the AVX-optimized
 * SHA-512/384 algorithm. It calls the common `sha512_update` helper, passing the
 * `sha512_transform_avx` assembly routine as the block transformation function.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The input data to be hashed.
 * @param len The length of the input data.
 * @return The number of remaining bytes.
 * Functional Utility: Updates the SHA-512/384 hash state using the AVX-optimized assembly transform.
 */
static int sha512_avx_update(struct shash_desc *desc, const u8 *data,
		       unsigned int len)
{
	return sha512_update(desc, data, len, sha512_transform_avx);
}

/**
 * @brief `shash` 'finup' callback for AVX-accelerated SHA-512/384.
 * @details This function is the 'finup' entry point for the AVX-optimized
 * SHA-512/384 algorithm. It calls the common `sha512_finup` helper, passing the
 * `sha512_transform_avx` assembly routine as the block transformation function.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The final input data to be hashed.
 * @param len The length of the final input data.
 * @param out The buffer to store the final SHA-512/384 digest.
 * @return 0 on success.
 * Functional Utility: Finalizes SHA-512/384 computation using the AVX-optimized assembly transform.
 */
static int sha512_avx_finup(struct shash_desc *desc, const u8 *data,
	      unsigned int len, u8 *out)
{
	return sha512_finup(desc, data, len, out, sha512_transform_avx);
}

/**
 * @brief Defines the AVX-accelerated SHA-512 and SHA-384 algorithms for the crypto API.
 * @details This array of `shash_alg` structures registers both the SHA-512
 * and SHA-384 algorithm implementations that leverage Intel AVX instructions.
 * It specifies the algorithms' properties (digest size, block size), associates
 * the core operations (`init`, `update`, `finup`) with their respective handler
 * functions. A `cra_priority` of 160 indicates a higher priority than SSSE3
 * but lower than AVX2.
 * Functional Utility: Registers AVX-optimized SHA-512 and SHA-384 algorithms with the kernel crypto API.
 */
static struct shash_alg sha512_avx_algs[] = { {
	.digestsize	=	SHA512_DIGEST_SIZE,
	.init		=	sha512_base_init,
	.update		=	sha512_avx_update,
	.finup		=	sha512_avx_finup,
	.descsize	=	SHA512_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha512",
		.cra_driver_name =	"sha512-avx",
		.cra_priority	=	160, // Functional Utility: Priority for AVX SHA-512 implementation.
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY |
					CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize	=	SHA512_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
},  {
	.digestsize	=	SHA384_DIGEST_SIZE,
	.init		=	sha384_base_init,
	.update		=	sha512_avx_update,
	.finup		=	sha512_avx_finup,
	.descsize	=	SHA512_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha384",
		.cra_driver_name =	"sha384-avx",
		.cra_priority	=	160, // Functional Utility: Priority for AVX SHA-384 implementation.
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY |
					CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize	=	SHA384_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
} };

/**
 * @brief Conditionally registers the AVX-accelerated SHA-512/384 algorithms.
 * @details This function attempts to register the `sha512-avx` and `sha384-avx`
 * algorithms with the crypto API. It first checks if AVX instructions are usable
 * (`avx_usable()`). The registration proceeds only if AVX is usable.
 *
 * @return 0 on success, or a negative error code if registration fails or AVX is unusable.
 * Functional Utility: Registers the AVX SHA-512/384 algorithms if the CPU supports it and it's usable.
 */
static int register_sha512_avx(void)
{
	if (avx_usable())
		return crypto_register_shashes(sha512_avx_algs,
			ARRAY_SIZE(sha512_avx_algs));
	return 0; // Functional Utility: Returns 0 if AVX is not usable (no registration attempted).
}

/**
 * @brief Conditionally unregisters the AVX-accelerated SHA-512/384 algorithms.
 * @details This function attempts to unregister the `sha512-avx` and `sha384-avx`
 * algorithms from the crypto API. It checks if AVX was usable, ensuring that only
 * registered algorithms are unregistered.
 * Functional Utility: Unregisters the AVX SHA-512/384 algorithms if they were potentially registered.
 */
static void unregister_sha512_avx(void)
{
	if (avx_usable())
		crypto_unregister_shashes(sha512_avx_algs,
			ARRAY_SIZE(sha512_avx_algs));
}

/**
 * @brief External declaration for the AVX2-optimized SHA-512 transform function.
 * @details This function is an `asmlinkage` declaration, indicating that its
 * implementation is provided in a separate assembly file (e.g., `sha512-avx2-asm.S`
 * or `sha512_transform_rorx`). It performs the core SHA-512 block transformation
 * leveraging Intel AVX2 instructions. The name `rorx` implies it might also
 * utilize BMI2 (Rotate-with-Extend) instructions for further optimization.
 *
 * @param state Pointer to the SHA-512 state array.
 * @param data Pointer to the input data blocks.
 * @param blocks The number of 128-byte blocks to process.
 * Functional Utility: Dispatches to the AVX2 assembly routine for hardware-accelerated SHA-512 block processing.
 */
asmlinkage void sha512_transform_rorx(struct sha512_state *state,
				      const u8 *data, int blocks);

/**
 * @brief `shash` 'update' callback for AVX2-accelerated SHA-512/384.
 * @details This function is the 'update' entry point for the AVX2-optimized
 * SHA-512/384 algorithm. It calls the common `sha512_update` helper, passing the
 * `sha512_transform_rorx` assembly routine as the block transformation function.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The input data to be hashed.
 * @param len The length of the input data.
 * @return The number of remaining bytes.
 * Functional Utility: Updates the SHA-512/384 hash state using the AVX2-optimized assembly transform.
 */
static int sha512_avx2_update(struct shash_desc *desc, const u8 *data,
		       unsigned int len)
{
	return sha512_update(desc, data, len, sha512_transform_rorx);
}

/**
 * @brief `shash` 'finup' callback for AVX2-accelerated SHA-512/384.
 * @details This function is the 'finup' entry point for the AVX2-optimized
 * SHA-512/384 algorithm. It calls the common `sha512_finup` helper, passing the
 * `sha512_transform_rorx` assembly routine as the block transformation function.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The final input data to be hashed.
 * @param len The length of the final input data.
 * @param out The buffer to store the final SHA-512/384 digest.
 * @return 0 on success.
 * Functional Utility: Finalizes SHA-512/384 computation using the AVX2-optimized assembly transform.
 */
static int sha512_avx2_finup(struct shash_desc *desc, const u8 *data,
	      unsigned int len, u8 *out)
{
	return sha512_finup(desc, data, len, out, sha512_transform_rorx);
}

/**
 * @brief Defines the AVX2-accelerated SHA-512 and SHA-384 algorithms for the crypto API.
 * @details This array of `shash_alg` structures registers both the SHA-512
 * and SHA-384 algorithm implementations that leverage Intel AVX2 instructions.
 * It specifies the algorithms' properties (digest size, block size), associates
 * the core operations (`init`, `update`, `finup`) with their respective handler
 * functions. A `cra_priority` of 170 indicates a higher priority than AVX.
 * Functional Utility: Registers AVX2-optimized SHA-512 and SHA-384 algorithms with the kernel crypto API.
 */
static struct shash_alg sha512_avx2_algs[] = { {
	.digestsize	=	SHA512_DIGEST_SIZE,
	.init		=	sha512_base_init,
	.update		=	sha512_avx2_update,
	.finup		=	sha512_avx2_finup,
	.descsize	=	SHA512_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha512",
		.cra_driver_name =	"sha512-avx2",
		.cra_priority	=	170, // Functional Utility: Priority for AVX2 SHA-512 implementation.
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY |
					CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize	=	SHA512_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
},  {
	.digestsize	=	SHA384_DIGEST_SIZE,
	.init		=	sha384_base_init,
	.update		=	sha512_avx2_update,
	.finup		=	sha512_avx2_finup,
	.descsize	=	SHA512_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha384",
		.cra_driver_name =	"sha384-avx2",
		.cra_priority	=	170, // Functional Utility: Priority for AVX2 SHA-384 implementation.
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY |
					CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize	=	SHA384_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
} };

/**
 * @brief Checks if AVX2 instructions, along with BMI2, are usable.
 * @details This function determines if AVX2 instructions can be safely used
 * for SHA-512/384 acceleration. It relies on `avx_usable()` for base AVX checks
 * and additionally verifies the presence of `X86_FEATURE_AVX2` and `X86_FEATURE_BMI2`
 * via `boot_cpu_has()`. BMI2 (Bit Manipulation Instruction Set 2) often provides
 * instructions like `RORX` that are beneficial for SHA implementations.
 *
 * @return `true` if AVX, AVX2, and BMI2 are all supported and usable, `false` otherwise.
 * Functional Utility: Verifies CPU support and kernel enablement for AVX2 and BMI2 instruction sets.
 */
static bool avx2_usable(void)
{
	// Functional Utility: Checks for base AVX usability, plus AVX2 and BMI2 features.
	if (avx_usable() && boot_cpu_has(X86_FEATURE_AVX2) &&
		    boot_cpu_has(X86_FEATURE_BMI2))
		return true;

	return false;
}

/**
 * @brief Conditionally registers the AVX2-accelerated SHA-512/384 algorithms.
 * @details This function attempts to register the `sha512-avx2` and `sha384-avx2`
 * algorithms with the crypto API. It first checks if AVX2 instructions (and
 * required BMI2) are usable (`avx2_usable()`). The registration proceeds only if
 * these features are present and usable.
 *
 * @return 0 on success, or a negative error code if registration fails or features are absent.
 * Functional Utility: Registers the AVX2 SHA-512/384 algorithms if the CPU features are present and usable.
 */
static int register_sha512_avx2(void)
{
	if (avx2_usable())
		return crypto_register_shashes(sha512_avx2_algs,
			ARRAY_SIZE(sha512_avx2_algs));
	return 0; // Functional Utility: Returns 0 if AVX2 is not usable (no registration attempted).
}
/**
 * @brief List of x86 CPU features supported by this module.
 * @details This array defines the CPU features that this module can leverage
 * for SHA-512/384 acceleration, ordered by general performance (AVX2 > AVX > SSSE3).
 * It is used by `MODULE_DEVICE_TABLE` to allow the kernel to automatically
 * load this module when a compatible CPU is detected.
 * Functional Role: Specifies the CPU capabilities required for optimal SHA-512/384 acceleration by this module.
 */
static const struct x86_cpu_id module_cpu_ids[] = {
	X86_MATCH_FEATURE(X86_FEATURE_AVX2, NULL), // Functional Role: Match CPU with AVX2.
	X86_MATCH_FEATURE(X86_FEATURE_AVX, NULL), // Functional Role: Match CPU with AVX.
	X86_MATCH_FEATURE(X86_FEATURE_SSSE3, NULL), // Functional Role: Match CPU with SSSE3.
	{} // Functional Role: Terminator for the array.
};
MODULE_DEVICE_TABLE(x86cpu, module_cpu_ids);

/**
 * @brief Conditionally unregisters the AVX2-accelerated SHA-512/384 algorithms.
 * @details This function attempts to unregister the `sha512-avx2` and `sha384-avx2`
 * algorithms from the crypto API. It checks if AVX2 was usable, ensuring that only
 * registered algorithms are unregistered.
 * Functional Utility: Unregisters the AVX2 SHA-512/384 algorithms if they were potentially registered.
 */
static void unregister_sha512_avx2(void)
{
	if (avx2_usable())
		crypto_unregister_shashes(sha512_avx2_algs,
			ARRAY_SIZE(sha512_avx2_algs));
}

/**
 * @brief Module initialization function for x86 SHA-512/384 glue code.
 * @details This function is the main entry point when the kernel module is loaded.
 * It attempts to register all available SHA-512/384 optimized versions (SSSE3, AVX, AVX2)
 * with the Linux kernel's cryptographic API, in a prioritized order.
 * The order of registration is important: it tries to register the lowest
 * priority (SSSE3) first, then AVX, and finally AVX2. If any registration
 * fails, it performs a clean unregistration of all previously successfully
 * registered algorithms. This ensures that only supported and correctly
 * registered algorithms are active.
 * Functional Utility: Registers available x86 SHA-512/384 optimized algorithms (SSSE3, AVX, AVX2) with error handling.
 * @return 0 on successful registration of at least one algorithm, or `-ENODEV` if no algorithms could be registered.
 */
static int __init sha512_ssse3_mod_init(void)
{
	// Pre-condition: Checks if the current CPU matches any of the supported features.
	if (!x86_match_cpu(module_cpu_ids))
		return -ENODEV; // Functional Utility: Returns -ENODEV if no matching CPU features are found.

	// Functional Utility: Attempts to register SSSE3 SHA-512/384 algorithms. If it fails, jumps to cleanup.
	if (register_sha512_ssse3())
		goto fail;

	// Functional Utility: Attempts to register AVX SHA-512/384 algorithms. If it fails, unregisters SSSE3 and jumps to cleanup.
	if (register_sha512_avx()) {
		unregister_sha512_ssse3();
		goto fail;
	}

	// Functional Utility: Attempts to register AVX2 SHA-512/384 algorithms. If it fails, unregisters AVX and SSSE3, then jumps to cleanup.
	if (register_sha512_avx2()) {
		unregister_sha512_avx();
		unregister_sha512_ssse3();
		goto fail;
	}

	return 0; // Functional Utility: Returns 0 if at least one algorithm was successfully registered.
fail:
	return -ENODEV; // Functional Utility: Returns -ENODEV if any registration failed.
}

/**
 * @brief Module cleanup function for x86 SHA-512/384 glue code.
 * @details This function is the exit point when the kernel module is unloaded.
 * It cleanly unregisters all potentially registered SHA-512/384 optimized versions
 * (AVX2, AVX, SSSE3) from the Linux kernel's cryptographic API.
 * The unregistration order is the reverse of registration, ensuring a proper
 * teardown of the cryptographic interfaces.
 * Functional Utility: Unregisters all x86 SHA-512/384 optimized algorithms from the kernel crypto API.
 */
static void __exit sha512_ssse3_mod_fini(void)
{
	// Functional Utility: Unregisters AVX2 algorithms.
	unregister_sha512_avx2();
	// Functional Utility: Unregisters AVX algorithms.
	unregister_sha512_avx();
	// Functional Utility: Unregisters SSSE3 algorithms.
	unregister_sha512_ssse3();
}

module_init(sha512_ssse3_mod_init);
module_exit(sha512_ssse3_mod_fini);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SHA512 Secure Hash Algorithm, Supplemental SSE3 accelerated");

MODULE_ALIAS_CRYPTO("sha512");
MODULE_ALIAS_CRYPTO("sha512-ssse3");
MODULE_ALIAS_CRYPTO("sha512-avx");
MODULE_ALIAS_CRYPTO("sha512-avx2");
MODULE_ALIAS_CRYPTO("sha384");
MODULE_ALIAS_CRYPTO("sha384-ssse3");
MODULE_ALIAS_CRYPTO("sha384-avx");
MODULE_ALIAS_CRYPTO("sha384-avx2");
