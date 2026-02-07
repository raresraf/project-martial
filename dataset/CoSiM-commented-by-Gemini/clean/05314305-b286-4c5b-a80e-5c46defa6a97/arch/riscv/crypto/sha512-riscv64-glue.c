// SPDX-License-Identifier: GPL-2.0-or-later
/**
 * @file sha512-riscv64-glue.c
 * @brief Glue code for SHA-512 and SHA-384 using RISC-V vector crypto extensions (ZVKNHB, ZVKB).
 * @details This file provides the C-level interface (glue code) to integrate
 * highly optimized SHA-512 and SHA-384 implementations, which leverage the
 * RISC-V 64-bit vector cryptography extensions (ZVKNHB and ZVKB), with the
 * Linux kernel's cryptographic API (`shash`). Its primary responsibilities include:
 * - **Vector SIMD Context Management**: Ensuring safe and proper saving/restoring
 *   of the RISC-V vector unit's context when entering and exiting
 *   vector-accelerated sections (`kernel_vector_begin()` and `kernel_vector_end()`).
 * - **Conditional Dispatch**: Dynamically choosing between the high-performance
 *   vector assembly implementation and a generic software fallback based on the
 *   availability and usability of the vector unit.
 * - **Integration with Crypto API**: Providing the standard `update`, `finup`,
 *   and `digest` operations required by the `shash` API, delegating the core
 *   work to the vector-accelerated assembly.
 * This implementation aims to provide a high-performance SHA-512/384 solution
 * for RISC-V 64-bit platforms with vector crypto extensions.
 */
/*
 * SHA-512 and SHA-384 using the RISC-V vector crypto extensions
 *
 * Copyright (C) 2023 VRULL GmbH
 * Author: Heiko Stuebner <heiko.stuebner@vrull.eu>
 *
 * Copyright (C) 2023 SiFive, Inc.
 * Author: Jerry Shih <jerry.shih@sifive.com>
 */

#include <asm/simd.h>
#include <asm/vector.h>
#include <crypto/internal/hash.h>
#include <crypto/internal/simd.h>
#include <crypto/sha512_base.h>
#include <linux/kernel.h>
#include <linux/module.h>

/*
 * Note: the asm function only uses the 'state' field of struct sha512_state.
 * It is assumed to be the first field.
 */
/**
 * @brief Entry point for the RISC-V 64-bit vector crypto extensions assembly implementation of SHA-512.
 * @details This function is an `asmlinkage` declaration, indicating that its
 * implementation is provided in a separate RISC-V assembly file (e.g., `sha512-riscv64-zvknhb-zvkb.S`).
 * It performs the core SHA-512 block transformation using specialized RISC-V
 * vector crypto extensions (ZVKNHB and ZVKB) for high performance. The assembly
 * function expects `struct sha512_state` to begin directly with the SHA-512
 * 512-bit internal state.
 *
 * @param state Pointer to the SHA-512 state array.
 * @param data Pointer to the input data blocks.
 * @param num_blocks The number of 128-byte blocks to process.
 * Functional Utility: Dispatches to the RISC-V vector crypto assembly routine for hardware-accelerated SHA-512 block processing.
 */
asmlinkage void sha512_transform_zvknhb_zvkb(
	struct sha512_state *state, const u8 *data, int num_blocks);

/**
 * @brief Conditionally processes SHA-512 blocks using RISC-V vector extensions or a generic fallback.
 * @details This function serves as the primary dispatcher for SHA-512 block
 * processing. It first checks if the RISC-V vector unit is usable (`crypto_simd_usable()`).
 * If available, it manages the vector context using `kernel_vector_begin()` and
 * `kernel_vector_end()` and dispatches to the highly optimized assembly routine
 * (`sha512_transform_zvknhb_zvkb`). Otherwise, it falls back to a generic
 * software implementation (`sha512_generic_block_fn`).
 *
 * @param state Pointer to the SHA-512 state structure.
 * @param data Pointer to the input data blocks.
 * @param num_blocks The number of 128-byte blocks to process.
 * Functional Utility: Selects and executes the most optimal SHA-512 block transformation based on RISC-V vector unit availability.
 */
static void sha512_block(struct sha512_state *state, const u8 *data,
			 int num_blocks)
{
	/*
	 * Functional Utility: Compile-time assertion to ensure that `struct sha512_state`
	 * begins directly with its 512-bit internal state, as this is a strict
	 * requirement for the assembly function to correctly access and manipulate
	 * the hash state. This guarantees memory layout compatibility.
	 */
	BUILD_BUG_ON(offsetof(struct sha512_state, state) != 0);

	// Block Logic: Conditionally use vector extensions if available, otherwise fallback to generic.
	if (crypto_simd_usable()) {
		// Functional Utility: Saves the current vector unit state and enables its use in kernel mode.
		kernel_vector_begin();
		// Functional Utility: Dispatches to the RISC-V vector crypto extensions assembly for SHA-512 block processing.
		sha512_transform_zvknhb_zvkb(state, data, num_blocks);
		// Functional Utility: Restores the vector unit state after processing.
		kernel_vector_end();
	} else {
		// Functional Utility: Falls back to the generic software implementation of SHA-512 block processing.
		sha512_generic_block_fn(state, data, num_blocks);
	}
}

/**
 * @brief Implements the `shash` 'update' operation for RISC-V 64-bit vector-accelerated SHA-512/384.
 * @details This function integrates the RISC-V 64-bit vector-accelerated SHA-512/384
 * block processing into the kernel's generic `shash` API for incremental updates.
 * It acts as a wrapper that delegates data buffering and partial block
 * processing to the `sha512_base_do_update_blocks` helper function, providing
 * the `sha512_block` function as the callback for core block transformation.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return 0 on success.
 * Functional Utility: Updates the SHA-512/384 hash state incrementally, leveraging conditional RISC-V vector acceleration for core block processing.
 */
static int riscv64_sha512_update(struct shash_desc *desc, const u8 *data,
				 unsigned int len)
{
	// Functional Utility: Delegates data buffering and block processing to the base SHA-512 update function,
	// using the `sha512_block` dispatcher (which handles vector acceleration or fallback).
	return sha512_base_do_update_blocks(desc, data, len, sha512_block);
}

/**
 * @brief Finalizes the SHA-512/384 hash computation for RISC-V 64-bit vector-accelerated implementation.
 * @details This function handles the final steps of the SHA-512/384 algorithm. It
 * applies the necessary padding to the last data block, appends the total
 * message length, and then processes this final block using the RISC-V vector-
 * accelerated transformation (or its generic fallback). It utilizes the
 * `sha512_base_do_finup` and `sha512_base_finish` helper functions.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The final data chunk.
 * @param len  The length of the final data.
 * @param out  The buffer for the resulting hash digest.
 * @return 0 on success.
 * Functional Utility: Completes the SHA-512/384 hashing process, including padding and final digest generation, using RISC-V vector acceleration.
 */
static int riscv64_sha512_finup(struct shash_desc *desc, const u8 *data,
				unsigned int len, u8 *out)
{
	// Functional Utility: Processes any remaining input data and applies SHA-512 padding using the `sha512_block` dispatcher.
	sha512_base_do_finup(desc, data, len, sha512_block);
	// Functional Utility: Writes the final computed SHA-512/384 hash digest to the output buffer.
	return sha512_base_finish(desc, out);
}

/**
 * @brief Performs a one-shot SHA-512/384 hash computation for RISC-V 64-bit vector-accelerated implementation.
 * @details This function provides a convenient one-shot interface for hashing an
 * entire message. It initializes the hash state, processes the input data
 * (including padding and finalization), and outputs the digest in a single call.
 * It leverages the RISC-V vector-accelerated `finup` operation.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The entire input data to be hashed.
 * @param len  The total length of the input data.
 * @param out  The buffer for the resulting hash digest.
 * @return 0 on success.
 * Functional Utility: Computes the SHA-512/384 hash of an entire message in a single operation, using RISC-V vector acceleration.
 */
static int riscv64_sha512_digest(struct shash_desc *desc, const u8 *data,
				 unsigned int len, u8 *out)
{
	// Functional Utility: Initializes the SHA-512/384 hash state.
	// Functional Utility: Performs the finalization step with the entire data, returning the digest.
	return sha512_base_init(desc) ?:
	       riscv64_sha512_finup(desc, data, len, out);
}

/**
 * @brief Array of algorithm definitions for RISC-V 64-bit vector-accelerated SHA-512 and SHA-384.
 * @details This array of `shash_alg` structures registers both the SHA-512
 * (`sha512-riscv64-zvknhb-zvkb`) and SHA-384 (`sha384-riscv64-zvknhb-zvkb`)
 * algorithm implementations that leverage the RISC-V 64-bit vector crypto
 * extensions. It specifies the algorithms' properties, associates the core
 * operations (`init`, `update`, `finup`, `digest`) with their handler functions,
 * and sets a high priority (`300`) to ensure these hardware-accelerated versions
 * are preferred over generic software implementations when available.
 * Functional Utility: Defines and registers high-priority, RISC-V vector-accelerated SHA-512/384 algorithms for the kernel crypto API.
 */
static struct shash_alg riscv64_sha512_algs[] = {
	{
		.init = sha512_base_init, // Functional Role: Initializes SHA-512 hash state.
		.update = riscv64_sha512_update, // Functional Role: Updates SHA-512 hash state incrementally.
		.finup = riscv64_sha512_finup, // Functional Role: Finalizes SHA-512 hash computation.
		.digest = riscv64_sha512_digest, // Functional Role: Computes SHA-512 hash in a single operation.
		.descsize = SHA512_STATE_SIZE,
		.digestsize = SHA512_DIGEST_SIZE,
		.base = {
			.cra_blocksize = SHA512_BLOCK_SIZE,
			.cra_priority = 300, // Functional Role: Sets high priority for RISC-V vector acceleration.
			.cra_flags = CRYPTO_AHASH_ALG_BLOCK_ONLY |
				     CRYPTO_AHASH_ALG_FINUP_MAX,
			.cra_name = "sha512",
			.cra_driver_name = "sha512-riscv64-zvknhb-zvkb",
			.cra_module = THIS_MODULE,
		},
	}, {
		.init = sha384_base_init, // Functional Role: Initializes SHA-384 hash state.
		.update = riscv64_sha512_update, // Functional Role: Updates SHA-384 hash state incrementally.
		.finup = riscv64_sha512_finup, // Functional Role: Finalizes SHA-384 hash computation.
		.descsize = SHA512_STATE_SIZE,
		.digestsize = SHA384_DIGEST_SIZE,
		.base = {
			.cra_blocksize = SHA384_BLOCK_SIZE,
			.cra_priority = 300, // Functional Role: Sets high priority for RISC-V vector acceleration.
			.cra_flags = CRYPTO_AHASH_ALG_BLOCK_ONLY |
				     CRYPTO_AHASH_ALG_FINUP_MAX,
			.cra_name = "sha384",
			.cra_driver_name = "sha384-riscv64-zvknhb-zvkb",
			.cra_module = THIS_MODULE,
		},
	},
};

/**
 * @brief Module initialization function for RISC-V 64-bit vector-accelerated SHA-512/384.
 * @details This function is the entry point when the kernel module is loaded.
 * It performs a critical check for the availability of RISC-V vector crypto
 * extensions (ZVKNHB and ZVKB) and a minimum vector length (VLEN >= 128 bits).
 * If these hardware features are present, it registers the highly optimized
 * SHA-512 and SHA-384 algorithms (`riscv64_sha512_algs` array) with the Linux
 * kernel's cryptographic API. This ensures the module is only loaded on
 * compatible hardware, enabling high-performance hashing.
 * Functional Utility: Conditionally registers RISC-V vector-accelerated SHA-512/384 algorithms based on CPU feature availability and vector length.
 * @return 0 on successful registration, or `-ENODEV` if required CPU features are not available.
 */
static int __init riscv64_sha512_mod_init(void)
{
	// Pre-condition: Checks for the availability of specific RISC-V ISA extensions (ZVKNHB, ZVKB) and a minimum vector length.
	if (riscv_isa_extension_available(NULL, ZVKNHB) &&
	    riscv_isa_extension_available(NULL, ZVKB) &&
	    riscv_vector_vlen() >= 128)
		// Functional Utility: Registers the RISC-V vector-accelerated SHA-512/384 algorithms if hardware features are present.
		return crypto_register_shashes(riscv64_sha512_algs,
					       ARRAY_SIZE(riscv64_sha512_algs));

	// Functional Utility: Returns an error if the required RISC-V vector crypto extensions or vector length are not available.
	return -ENODEV;
}

/**
 * @brief Module cleanup function for RISC-V 64-bit vector-accelerated SHA-512/384.
 * @details This function is the exit point when the kernel module is unloaded.
 * Functional Utility: Unregisters all RISC-V vector-accelerated SHA-512/384
 * algorithms (`riscv64_sha512_algs` array) that were successfully registered
 * during module initialization. This cleanly removes the drivers from the system,
 * releasing associated resources and preventing any lingering references after
 * the module is no longer in use. This ensures proper resource management upon module unload.
 * Functional Utility: Unregisters RISC-V vector-accelerated SHA-512/384 algorithms from the kernel crypto API.
 */
static void __exit riscv64_sha512_mod_exit(void)
{
	crypto_unregister_shashes(riscv64_sha512_algs,
				  ARRAY_SIZE(riscv64_sha512_algs));
}

module_init(riscv64_sha512_mod_init);
module_exit(riscv64_sha512_mod_exit);

MODULE_DESCRIPTION("SHA-512 (RISC-V accelerated)");
MODULE_AUTHOR("Heiko Stuebner <heiko.stuebner@vrull.eu>");
MODULE_LICENSE("GPL");
MODULE_ALIAS_CRYPTO("sha512");
MODULE_ALIAS_CRYPTO("sha384");
