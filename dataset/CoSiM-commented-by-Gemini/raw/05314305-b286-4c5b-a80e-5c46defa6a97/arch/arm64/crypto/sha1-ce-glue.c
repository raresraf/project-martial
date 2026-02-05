/**
 * @file sha1-ce-glue.c
 * @brief Glue for ARMv8 Crypto Extensions SHA-1 for AArch64.
 * @details This file provides the C-level glue code to integrate the ARMv8
 * Cryptography Extensions (CE) accelerated SHA-1 implementation with the Linux
 * kernel's cryptographic API on AArch64. It is responsible for registering the
 * hardware-accelerated algorithm, managing the NEON/FPU context, and
 * implementing an optimization to offload the finalization (padding and length
 * appending) to the hardware-accelerated assembly code.
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
 * @details This structure extends the standard SHA-1 state with a `finalize`
 * flag. This flag is used to signal to the assembly code whether it should
 * perform the final padding and length appending, enabling an efficient
 * hardware offload of the entire finalization process.
 */
struct sha1_ce_state {
	struct sha1_state	sst;
	u32			finalize;
};

extern const u32 sha1_ce_offsetof_count;
extern const u32 sha1_ce_offsetof_finalize;

/**
 * @brief Entry point for the ARMv8 CE assembly implementation of SHA-1.
 * @param sst    Pointer to the custom SHA-1 CE state.
 * @param src    Pointer to the source data.
 * @param blocks Number of blocks to process.
 * @return       Number of blocks remaining to be processed.
 *
 * This function is defined in sha1-ce-core.S and executes the SHA-1
 * compression using dedicated hardware instructions.
 */
asmlinkage int __sha1_ce_transform(struct sha1_ce_state *sst, u8 const *src,
				   int blocks);

/**
 * @brief C wrapper for the assembly transform function.
 * @param sst    Pointer to the standard SHA-1 state.
 * @param src    Pointer to the source data.
 * @param blocks Number of blocks to process.
 *
 * This function's primary purpose is to manage the NEON FPU context safely
 * by wrapping the call to the assembly function with kernel_neon_begin() and
 * kernel_neon_end().
 */
static void sha1_ce_transform(struct sha1_state *sst, u8 const *src,
			      int blocks)
{
	while (blocks) {
		int rem;

		kernel_neon_begin();
		rem = __sha1_ce_transform(container_of(sst,
						       struct sha1_ce_state,
						       sst), src, blocks);
		kernel_neon_end();
		src += (blocks - rem) * SHA1_BLOCK_SIZE;
		blocks = rem;
	}
}

const u32 sha1_ce_offsetof_count = offsetof(struct sha1_ce_state, sst.count);
const u32 sha1_ce_offsetof_finalize = offsetof(struct sha1_ce_state, finalize);

/**
 * @brief Implements the shash 'update' operation.
 * @param desc The shash descriptor.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return 0 on success.
 *
 * This function clears the `finalize` flag in the custom state to ensure that
 * intermediate updates are not treated as a finalization step by the assembly
 * code.
 */
static int sha1_ce_update(struct shash_desc *desc, const u8 *data,
			  unsigned int len)
{
	struct sha1_ce_state *sctx = shash_desc_ctx(desc);

	sctx->finalize = 0;
	return sha1_base_do_update_blocks(desc, data, len, sha1_ce_transform);
}

/**
 * @brief Implements the shash 'finup' (finalize and update) operation.
 * @param desc The shash descriptor.
 * @param data The final data chunk.
 * @param len  The length of the final data.
 * @param out  The buffer for the resulting hash digest.
 * @return 0 on success.
 *
 * This function contains an optimization to offload the entire finalization
 * process to the hardware-accelerated assembly code if the input data meets
 * certain criteria (no partial data and length is a multiple of the block size).
 * Otherwise, it falls back to a software-based finalization path.
 */
static int sha1_ce_finup(struct shash_desc *desc, const u8 *data,
			 unsigned int len, u8 *out)
{
	struct sha1_ce_state *sctx = shash_desc_ctx(desc);
	bool finalized = false;

	/*
	 * Optimization: Allow the assembly code to perform finalization if the
	 * input is at least one full block and there is no partial data from
	 * previous updates.
	 */
	if (len >= SHA1_BLOCK_SIZE) {
		unsigned int remain = len - round_down(len, SHA1_BLOCK_SIZE);

		finalized = !remain;
		sctx->finalize = finalized;
		sha1_base_do_update_blocks(desc, data, len, sha1_ce_transform);
		data += len - remain;
		len = remain;
	}
	// Fallback: If hardware finalization was not possible, use the software
	// path to handle padding and final block processing.
	if (!finalized) {
		sctx->finalize = 0;
		sha1_base_do_finup(desc, data, len, sha1_ce_transform);
	}
	return sha1_base_finish(desc, out);
}

/**
 * @brief Defines the CE-accelerated SHA-1 algorithm for the crypto API.
 *
 * This structure registers the hardware-accelerated implementation. The
 * descriptor size is increased to accommodate the custom `sha1_ce_state`, and
 * the function pointers are set to the custom update/finup handlers that
 * implement the finalization offload logic.
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
 * This macro uses CPU feature detection to ensure this module is only loaded on
 * systems that have the ARMv8 SHA-1 Cryptography Extensions, allowing for
 * automatic loading of the best available driver.
 */
module_cpu_feature_match(SHA1, sha1_ce_mod_init);
module_exit(sha1_ce_mod_fini);
