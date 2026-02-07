// SPDX-License-Identifier: GPL-2.0+
/**
 * @file sha1_s390.c
 * @brief Hardware-accelerated SHA-1 Secure Hash Algorithm for s390 architecture.
 * @details This file provides a hardware-accelerated implementation of the SHA-1
 * (Secure Hash Algorithm 1) hash function, specifically adapted for IBM z/Architecture
 * (s390) processors. It leverages the Central Processor Assist Facility (CPACF)
 * to offload SHA-1 compression, significantly improving performance over
 * software-only implementations. The code integrates with the Linux kernel's
 * cryptographic API (`shash`), offering `init`, `update`, `finup`, `export`,
 * and `import` operations. This module conditionally registers itself only if
 * the underlying CPACF hardware supports the SHA-1 function (`CPACF_KIMD_SHA_1`).
 */
/*
 * Cryptographic API.
 *
 * s390 implementation of the SHA1 Secure Hash Algorithm.
 *
 * Derived from cryptoapi implementation, adapted for in-place
 * scatterlist interface.  Originally based on the public domain
 * implementation written by Steve Reid.
 *
 * s390 Version:
 *   Copyright IBM Corp. 2003, 2007
 *   Author(s): Thomas Spatzier
 *		Jan Glauber (jan.glauber@de.ibm.com)
 *
 * Derived from "crypto/sha1_generic.c"
 *   Copyright (c) Alan Smithee.
 *   Copyright (c) Andrew McDonald <andrew@mcdonald.org.uk>
 *   Copyright (c) Jean-Francois Dive <jef@linuxbe.org>
 */
#include <asm/cpacf.h>
#include <crypto/internal/hash.h>
#include <crypto/sha1.h>
#include <linux/cpufeature.h>
#include <linux/kernel.h>
#include <linux/module.h>

#include "sha.h"

/**
 * @brief Initializes the SHA-1 hash state for a new computation.
 * @details This function prepares the `s390_sha_ctx` context for a new SHA-1
 * hashing operation. It initializes the 160-bit hash digest with the standard
 * SHA-1 Initial Vector (IV) values, resets the total `byte_count`, and
 * configures the CPACF function code for SHA-1 (`CPACF_KIMD_SHA_1`).
 *
 * @param desc The `shash_desc` descriptor for the current SHA-1 operation.
 * @return 0 on success.
 * Functional Utility: Sets up the initial state for an SHA-1 hashing operation, including CPACF configuration.
 */
static int s390_sha1_init(struct shash_desc *desc)
{
	struct s390_sha_ctx *sctx = shash_desc_ctx(desc);

	// Functional Utility: Initializes the SHA-1 hash digest with the standard IV values.
	sctx->state[0] = SHA1_H0;
	sctx->state[1] = SHA1_H1;
	sctx->state[2] = SHA1_H2;
	sctx->state[3] = SHA1_H3;
	sctx->state[4] = SHA1_H4;
	sctx->count = 0; // Functional Utility: Resets the total byte count.
	sctx->func = CPACF_KIMD_SHA_1; // Functional Utility: Sets the CPACF function code for SHA-1.
	sctx->first_message_part = 0; // Functional Utility: Resets flag for the first message part.

	return 0;
}

/**
 * @brief Exports the current SHA-1 hash state to a generic `sha1_state` structure.
 * @details This function is part of the Linux crypto API's state management.
 * It exports the current SHA-1 hash context (`s390_sha_ctx`) into a generic
 * `sha1_state` structure, allowing the state to be saved and later restored
 * or passed to other implementations. Both the hash digest and the byte count
 * are included in the exported state.
 *
 * @param desc The `shash_desc` descriptor.
 * @param out Pointer to a `sha1_state` structure where the state will be exported.
 * @return 0 on success.
 * Functional Utility: Saves the current SHA-1 hashing state into a generic external buffer.
 */
static int s390_sha1_export(struct shash_desc *desc, void *out)
{
	struct s390_sha_ctx *sctx = shash_desc_ctx(desc);
	struct sha1_state *octx = out;

	octx->count = sctx->count; // Functional Utility: Exports the total byte count.
	memcpy(octx->state, sctx->state, sizeof(octx->state)); // Functional Utility: Exports the SHA-1 hash digest.
	return 0;
}

/**
 * @brief Imports a previously saved SHA-1 state into the `s390_sha_ctx` context.
 * @details This function is part of the Linux crypto API's state management.
 * It imports a previously saved SHA-1 hash context from a generic `sha1_state`
 * structure into the `s390_sha_ctx` context. This allows a hashing operation
 * to resume from a specific point. It also re-configures the CPACF function
 * code and resets the `first_message_part` flag.
 *
 * @param desc The `shash_desc` descriptor.
 * @param in Pointer to a `sha1_state` structure containing the saved state.
 * @return 0 on success.
 * Functional Utility: Restores a previously saved SHA-1 hashing state from a generic external buffer.
 */
static int s390_sha1_import(struct shash_desc *desc, const void *in)
{
	struct s390_sha_ctx *sctx = shash_desc_ctx(desc);
	const struct sha1_state *ictx = in;

	sctx->count = ictx->count; // Functional Utility: Imports the total byte count.
	memcpy(sctx->state, ictx->state, sizeof(ictx->state)); // Functional Utility: Imports the SHA-1 hash digest.
	sctx->func = CPACF_KIMD_SHA_1; // Functional Utility: Sets the CPACF function code for SHA-1.
	sctx->first_message_part = 0; // Functional Utility: Resets flag for the first message part.
	return 0;
}

/**
 * @brief Defines the S390 hardware-accelerated SHA-1 algorithm for the crypto API.
 * @details This structure registers the SHA-1 algorithm implementation that leverages
 * IBM z/Architecture's Central Processor Assist Facility (CPACF). It specifies
 * the algorithm's properties (digest size, block size), associates the core
 * operations (`init`, `update`, `finup`, `export`, `import`) with their
 * respective handler functions. A high `cra_priority` of 300 ensures this
 * hardware-accelerated version is preferred over generic software implementations.
 * Functional Utility: Registers the S390 hardware-accelerated SHA-1 algorithm with the kernel crypto API.
 */
static struct shash_alg alg = {
	.digestsize	=	SHA1_DIGEST_SIZE,
	.init		=	s390_sha1_init,
	.update		=	s390_sha_update_blocks,
	.finup		=	s390_sha_finup,
	.export		=	s390_sha1_export,
	.import		=	s390_sha1_import,
	.descsize	=	S390_SHA_CTX_SIZE,
	.statesize	=	SHA1_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha1",
		.cra_driver_name=	"sha1-s390",
		.cra_priority	=	300, // Functional Utility: Sets high priority for CPACF hardware acceleration.
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY |
					CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize	=	SHA1_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
};

/**
 * @brief Module initialization function for S390 SHA-1.
 * @details This function is the entry point when the kernel module is loaded.
 * It first performs a crucial check using `cpacf_query_func()` to determine
 * if the underlying S390 hardware's CPACF supports the SHA-1 function (`CPACF_KIMD_SHA_1`).
 * If the hardware support is confirmed, it registers the hardware-accelerated
 * SHA-1 algorithm (`alg` structure) with the Linux kernel's cryptographic API,
 * making the `sha1-s390` driver available to the system.
 * Functional Utility: Conditionally registers the hardware-accelerated S390 SHA-1 algorithm with the kernel crypto API, based on CPACF hardware support.
 * @return 0 on successful registration, or `-ENODEV` if CPACF SHA-1 support is not available, or other error code on failure.
 */
static int __init sha1_s390_init(void)
{
	// Pre-condition: Checks if the S390 CPACF (Central Processor Assist Facility) supports the SHA-1 function.
	if (!cpacf_query_func(CPACF_KIMD, CPACF_KIMD_SHA_1))
		return -ENODEV; // Functional Utility: Returns an error if CPACF SHA-1 support is not available.
	// Functional Utility: Registers the S390 hardware-accelerated SHA-1 algorithm with the kernel crypto API.
	return crypto_register_shash(&alg);
}

/**
 * @brief Module cleanup function for S390 SHA-1.
 * @details This function is the exit point when the kernel module is unloaded.
 * Functional Utility: Unregisters the hardware-accelerated SHA-1 algorithm
 * (`alg` structure) from the Linux kernel's cryptographic API. This cleanly
 * removes the `sha1-s390` driver from the system, releasing associated resources
 * and preventing any lingering references after the module is no longer in use.
 * This ensures proper resource management upon module unload.
 * Functional Utility: Unregisters the hardware-accelerated S390 SHA-1 algorithm from the kernel crypto API.
 */
static void __exit sha1_s390_fini(void)
{
	crypto_unregister_shash(&alg);
}

/**
 * @brief Macro for conditional module loading based on S390 CPU features.
 * @details This macro ensures that the `sha1-s390` module is only loaded and
 * initialized on S390 systems that possess the Message-Security Assist (MSA)
 * CPU feature, which is required for CPACF cryptographic instructions. This
 * mechanism allows the kernel to dynamically load the hardware-accelerated
 * SHA-1 implementation automatically when compatible hardware is detected.
 * Functional Utility: Ensures the S390 SHA-1 module is loaded only on S390 CPUs with the Message-Security Assist feature.
 */
module_cpu_feature_match(S390_CPU_FEATURE_MSA, sha1_s390_init);
module_exit(sha1_s390_fini);

MODULE_ALIAS_CRYPTO("sha1");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SHA1 Secure Hash Algorithm");
