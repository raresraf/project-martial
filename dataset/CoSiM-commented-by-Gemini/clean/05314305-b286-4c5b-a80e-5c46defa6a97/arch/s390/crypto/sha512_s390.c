// SPDX-License-Identifier: GPL-2.0+
/**
 * @file sha512_s390.c
 * @brief Hardware-accelerated SHA-512 and SHA-384 Secure Hash Algorithms for s390 architecture.
 * @details This file provides hardware-accelerated implementations for both
 * SHA-512 and SHA-384 hash functions, specifically adapted for IBM z/Architecture
 * (s390) processors. It leverages the Central Processor Assist Facility (CPACF)
 * to offload SHA-512/384 compression, significantly improving performance over
 * software-only implementations. The code integrates with the Linux kernel's
 * cryptographic API (`shash`), offering `init`, `update`, `finup`, `export`,
 * and `import` operations. This module conditionally registers itself only if
 * the underlying CPACF hardware supports the SHA-512 function (`CPACF_KIMD_SHA_512`).
 */
/*
 * Cryptographic API.
 *
 * s390 implementation of the SHA512 and SHA38 Secure Hash Algorithm.
 *
 * Copyright IBM Corp. 2007
 * Author(s): Jan Glauber (jang@de.ibm.com)
 */
#include <asm/cpacf.h>
#include <crypto/internal/hash.h>
#include <crypto/sha2.h>
#include <linux/cpufeature.h>
#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/module.h>

#include "sha.h"

/**
 * @brief Initializes the SHA-512 hash state for a new computation.
 * @details This function prepares the `s390_sha_ctx` context for a new SHA-512
 * hashing operation. It initializes the 512-bit hash digest with the standard
 * SHA-512 Initial Vector (IV) values (`SHA512_H0` through `SHA512_H7`),
 * resets the total `count` and `count_hi` (for 128-bit byte count), and
 * configures the CPACF function code for SHA-512 (`CPACF_KIMD_SHA_512`).
 *
 * @param desc The `shash_desc` descriptor for the current SHA-512 operation.
 * @return 0 on success.
 * Functional Utility: Sets up the initial state for an SHA-512 hashing operation, including CPACF configuration.
 */
static int sha512_init(struct shash_desc *desc)
{
	struct s390_sha_ctx *ctx = shash_desc_ctx(desc);

	// Functional Utility: Initializes the SHA-512 hash digest with the standard SHA-512 Initial Vector (IV) values.
	ctx->sha512.state[0] = SHA512_H0;
	ctx->sha512.state[1] = SHA512_H1;
	ctx->sha512.state[2] = SHA512_H2;
	ctx->sha512.state[3] = SHA512_H3;
	ctx->sha512.state[4] = SHA512_H4;
	ctx->sha512.state[5] = SHA512_H5;
	ctx->sha512.state[6] = SHA512_H6;
	ctx->sha512.state[7] = SHA512_H7;
	ctx->count = 0; // Functional Utility: Resets the lower 64-bit part of the total byte count.
	ctx->sha512.count_hi = 0; // Functional Utility: Resets the upper 64-bit part of the total byte count.
	ctx->func = CPACF_KIMD_SHA_512; // Functional Utility: Sets the CPACF function code for SHA-512.
	ctx->first_message_part = 0; // Functional Utility: Resets flag for the first message part.

	return 0;
}

/**
 * @brief Exports the current SHA-512/384 hash state to a generic `sha512_state` structure.
 * @details This function is part of the Linux crypto API's state management.
 * It exports the current SHA-512/384 hash context (`s390_sha_ctx`) into a generic
 * `sha512_state` structure, allowing the state to be saved and later restored
 * or passed to other implementations. Both the hash digest and the 128-bit
 * byte count are included in the exported state.
 *
 * @param desc The `shash_desc` descriptor.
 * @param out Pointer to a `sha512_state` structure where the state will be exported.
 * @return 0 on success.
 * Functional Utility: Saves the current SHA-512/384 hashing state into a generic external buffer.
 */
static int sha512_export(struct shash_desc *desc, void *out)
{
	struct s390_sha_ctx *sctx = shash_desc_ctx(desc);
	struct sha512_state *octx = out;

	octx->count[0] = sctx->count; // Functional Utility: Exports the lower 64-bit part of the total byte count.
	octx->count[1] = sctx->sha512.count_hi; // Functional Utility: Exports the upper 64-bit part of the total byte count.
	memcpy(octx->state, sctx->sha512.state, sizeof(octx->state)); // Functional Utility: Exports the SHA-512/384 hash digest.
	return 0;
}

/**
 * @brief Imports a previously saved SHA-512/384 state into the `s390_sha_ctx` context.
 * @details This function is part of the Linux crypto API's state management.
 * It imports a previously saved SHA-512/384 hash context from a generic
 * `sha512_state` structure into the `s390_sha_ctx` context. This allows a
 * hashing operation to resume from a specific point. It also re-configures
 * the CPACF function code and resets the `first_message_part` flag.
 *
 * @param desc The `shash_desc` descriptor.
 * @param in Pointer to a `sha512_state` structure containing the saved state.
 * @return 0 on success.
 * Functional Utility: Restores a previously saved SHA-512/384 hashing state from a generic external buffer.
 */
static int sha512_import(struct shash_desc *desc, const void *in)
{
	struct s390_sha_ctx *sctx = shash_desc_ctx(desc);
	const struct sha512_state *ictx = in;

	sctx->count = ictx->count[0]; // Functional Utility: Imports the lower 64-bit part of the total byte count.
	sctx->sha512.count_hi = ictx->count[1]; // Functional Utility: Imports the upper 64-bit part of the total byte count.

	memcpy(sctx->sha512.state, ictx->state, sizeof(ictx->state)); // Functional Utility: Imports the SHA-512/384 hash digest.
	sctx->func = CPACF_KIMD_SHA_512; // Functional Utility: Sets the CPACF function code for SHA-512.
	sctx->first_message_part = 0; // Functional Utility: Resets flag for the first message part.
	return 0;
}

/**
 * @brief Defines the S390 hardware-accelerated SHA-512 algorithm for the crypto API.
 * @details This structure registers the SHA-512 algorithm implementation that leverages
 * IBM z/Architecture's Central Processor Assist Facility (CPACF). It specifies
 * the algorithm's properties (digest size, block size), associates the core
 * operations (`init`, `update`, `finup`, `export`, `import`) with their
 * respective handler functions. A high `cra_priority` of 300 ensures this
 * hardware-accelerated version is preferred over generic software implementations.
 * Functional Utility: Registers the S390 hardware-accelerated SHA-512 algorithm with the kernel crypto API.
 */
static struct shash_alg sha512_alg = {
	.digestsize	=	SHA512_DIGEST_SIZE,
	.init		=	sha512_init,
	.update		=	s390_sha_update_blocks,
	.finup		=	s390_sha_finup,
	.export		=	sha512_export,
	.import		=	sha512_import,
	.descsize	=	sizeof(struct s390_sha_ctx),
	.statesize	=	SHA512_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha512",
		.cra_driver_name=	"sha512-s390",
		.cra_priority	=	300, // Functional Utility: Sets high priority for CPACF hardware acceleration.
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY |
					CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize	=	SHA512_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
};

MODULE_ALIAS_CRYPTO("sha512");

/**
 * @brief Initializes the SHA-384 hash state for a new computation.
 * @details This function prepares the `s390_sha_ctx` context for a new SHA-384
 * hashing operation. It initializes the 512-bit hash digest with the standard
 * SHA-384 Initial Vector (IV) values, resets the total `count` and `count_hi`,
 * and configures the CPACF function code for SHA-512 (`CPACF_KIMD_SHA_512`),
 * as SHA-384 is a truncated version of SHA-512.
 *
 * @param desc The `shash_desc` descriptor for the current SHA-384 operation.
 * @return 0 on success.
 * Functional Utility: Sets up the initial state for an SHA-384 hashing operation, including CPACF configuration.
 */
static int sha384_init(struct shash_desc *desc)
{
	struct s390_sha_ctx *ctx = shash_desc_ctx(desc);

	// Functional Utility: Initializes the SHA-384 hash digest with the standard SHA-384 Initial Vector (IV) values.
	ctx->sha512.state[0] = SHA384_H0;
	ctx->sha512.state[1] = SHA384_H1;
	ctx->sha512.state[2] = SHA384_H2;
	ctx->sha512.state[3] = SHA384_H3;
	ctx->sha512.state[4] = SHA384_H4;
	ctx->sha512.state[5] = SHA384_H5;
	ctx->sha512.state[6] = SHA384_H6;
	ctx->sha512.state[7] = SHA384_H7;
	ctx->count = 0; // Functional Utility: Resets the lower 64-bit part of the total byte count.
	ctx->sha512.count_hi = 0; // Functional Utility: Resets the upper 64-bit part of the total byte count.
	ctx->func = CPACF_KIMD_SHA_512; // Functional Utility: Sets the CPACF function code for SHA-512 (as SHA-384 uses SHA-512 core).
	ctx->first_message_part = 0; // Functional Utility: Resets flag for the first message part.

	return 0;
}

/**
 * @brief Defines the S390 hardware-accelerated SHA-384 algorithm for the crypto API.
 * @details This structure registers the SHA-384 algorithm implementation that leverages
 * IBM z/Architecture's Central Processor Assist Facility (CPACF). It specifies
 * the algorithm's properties (digest size, block size), associates the core
 * operations (`init`, `update`, `finup`, `export`, `import`) with their
 * respective handler functions. A high `cra_priority` of 300 ensures this
 * hardware-accelerated version is preferred over generic software implementations.
 * Functional Utility: Registers the S390 hardware-accelerated SHA-384 algorithm with the kernel crypto API.
 */
static struct shash_alg sha384_alg = {
	.digestsize	=	SHA384_DIGEST_SIZE,
	.init		=	sha384_init,
	.update		=	s390_sha_update_blocks,
	.finup		=	s390_sha_finup,
	.export		=	sha512_export,
	.import		=	sha512_import,
	.descsize	=	sizeof(struct s390_sha_ctx),
	.statesize	=	SHA512_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha384",
		.cra_driver_name=	"sha384-s390",
		.cra_priority	=	300, // Functional Utility: Sets high priority for CPACF hardware acceleration.
		.cra_blocksize	=	SHA384_BLOCK_SIZE,
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY |
					CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_ctxsize	=	sizeof(struct s390_sha_ctx),
		.cra_module	=	THIS_MODULE,
	}
};

MODULE_ALIAS_CRYPTO("sha384");

/**
 * @brief Module initialization function for S390 SHA-512 and SHA-384.
 * @details This function is the entry point when the kernel module is loaded.
 * It first performs a crucial check using `cpacf_query_func()` to determine
 * if the underlying S390 hardware's CPACF supports the SHA-512 function (`CPACF_KIMD_SHA_512`).
 * If the hardware support is confirmed, it attempts to register both the
 * hardware-accelerated SHA-512 (`sha512_alg`) and SHA-384 (`sha384_alg`)
 * algorithms with the Linux kernel's cryptographic API. It includes robust
 * error handling to unregister any successfully registered algorithm if a
 * subsequent registration fails, ensuring a clean state.
 * Functional Utility: Conditionally registers hardware-accelerated S390 SHA-512 and SHA-384 algorithms based on CPACF hardware support.
 * @return 0 on successful registration of both algorithms, or `-ENODEV` if CPACF
 * SHA-512 support is not available, or another error code on registration failure.
 */
static int __init init(void)
{
	int ret;

	// Pre-condition: Checks if the S390 CPACF (Central Processor Assist Facility) supports the SHA-512 function.
	if (!cpacf_query_func(CPACF_KIMD, CPACF_KIMD_SHA_512))
		return -ENODEV; // Functional Utility: Returns an error if CPACF SHA-512 support is not available.
	// Functional Utility: Registers the S390 hardware-accelerated SHA-512 algorithm.
	if ((ret = crypto_register_shash(&sha512_alg)) < 0)
		goto out; // Functional Utility: Jumps to cleanup if SHA-512 registration fails.
	// Functional Utility: Registers the S390 hardware-accelerated SHA-384 algorithm.
	if ((ret = crypto_register_shash(&sha384_alg)) < 0)
		crypto_unregister_shash(&sha512_alg); // Functional Utility: Unregisters SHA-512 if SHA-384 registration fails.
out:
	return ret;
}

/**
 * @brief Module cleanup function for S390 SHA-512 and SHA-384.
 * @details This function is the exit point when the kernel module is unloaded.
 * Functional Utility: Unregisters both the hardware-accelerated SHA-512 and
 * SHA-384 algorithms (`sha512_alg` and `sha384_alg` structures) from the
 * Linux kernel's cryptographic API. This cleanly removes the drivers from
 * the system, releasing associated resources and preventing any lingering
 * references after the module is no longer in use. This ensures proper
 * resource management upon module unload.
 * Functional Utility: Unregisters the hardware-accelerated S390 SHA-512 and SHA-384 algorithms from the kernel crypto API.
 */
static void __exit fini(void)
{
	// Functional Utility: Unregisters the S390 hardware-accelerated SHA-512 algorithm.
	crypto_unregister_shash(&sha512_alg);
	// Functional Utility: Unregisters the S390 hardware-accelerated SHA-384 algorithm.
	crypto_unregister_shash(&sha384_alg);
}

/**
 * @brief Macro for conditional module loading based on S390 CPU features.
 * @details This macro ensures that the `sha512_s390` module is only loaded and
 * initialized on S390 systems that possess the Message-Security Assist (MSA)
 * CPU feature, which is required for CPACF cryptographic instructions. This
 * mechanism allows the kernel to dynamically load the hardware-accelerated
 * SHA-512/384 implementation automatically when compatible hardware is detected.
 * Functional Utility: Ensures the S390 SHA-512/384 module is loaded only on S390 CPUs with the Message-Security Assist feature.
 */
module_cpu_feature_match(S390_CPU_FEATURE_MSA, init);
module_exit(fini);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SHA512 and SHA-384 Secure Hash Algorithm");
