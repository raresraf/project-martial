// SPDX-License-Identifier: GPL-2.0-or-later
/**
 * @file sha256.c
 * @brief Crypto API wrapper for the SHA-256 and SHA-224 library functions.
 * @details This file provides a comprehensive wrapper that integrates various
 * SHA-256 and SHA-224 implementations (generic, library-optimized, and
 * architecture-specific) into the Linux kernel's cryptographic API (`shash`).
 * It acts as a central dispatcher, allowing the kernel to dynamically select
 * the most appropriate and optimized SHA-2 family hashing function based on
 * CPU capabilities and caller preferences. This modular design ensures both
 * portability and high performance across different hardware platforms.
 */
/*
 * Crypto API wrapper for the SHA-256 and SHA-224 library functions
 *
 * Copyright (c) Jean-Luc Cooke <jlcooke@certainkey.com>
 * Copyright (c) Andrew McDonald <andrew@mcdonald.org.uk>
 * Copyright (c) 2002 James Morris <jmorris@intercode.com.au>
 * SHA224 Support Copyright 2007 Intel Corporation <jonathan.lynch@intel.com>
 */
#include <crypto/internal/hash.h>
#include <crypto/internal/sha2.h>
#include <linux/kernel.h>
#include <linux/module.h>

/**
 * @var sha224_zero_message_hash
 * @brief The pre-computed SHA-224 hash of a zero-length message.
 *
 * @details This constant array contains the well-known 28-byte SHA-224 digest
 * for an empty input string or message. This value is defined by the SHA-2
 * standard for its truncated variant, SHA-224. It is used for verification
 * purposes, self-tests, and can enable minor optimizations for empty inputs.
 * Functional Utility: Provides the canonical hash for an empty message in SHA-224.
 */
const u8 sha224_zero_message_hash[SHA224_DIGEST_SIZE] = {
	0xd1, 0x4a, 0x02, 0x8c, 0x2a, 0x3a, 0x2b, 0xc9, 0x47,
	0x61, 0x02, 0xbb, 0x28, 0x82, 0x34, 0xc4, 0x15, 0xa2,
	0xb0, 0x1f, 0x82, 0x8e, 0xa6, 0x2a, 0xc5, 0xb3, 0xe4,
	0x2f
};
EXPORT_SYMBOL_GPL(sha224_zero_message_hash);

/**
 * @var sha256_zero_message_hash
 * @brief The pre-computed SHA-256 hash of a zero-length message.
 *
 * @details This constant array contains the well-known 32-byte SHA-256 digest
 * for an empty input string or message. This value is defined by the SHA-2
 * standard. It is used for verification purposes, self-tests, and can enable
 * minor optimizations for empty inputs.
 * Functional Utility: Provides the canonical hash for an empty message in SHA-256.
 */
const u8 sha256_zero_message_hash[SHA256_DIGEST_SIZE] = {
	0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
	0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
	0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
	0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55
};
EXPORT_SYMBOL_GPL(sha256_zero_message_hash);

/**
 * @brief Initializes the SHA-256 hash state.
 * @details This function is the `init` callback for SHA-256 algorithms registered
 * with the Crypto API. It sets the initial hash values (IVs) for SHA-256
 * in the provided `crypto_sha256_state` context, preparing it for a new hashing operation.
 *
 * @param desc The `shash_desc` descriptor containing the SHA-256 state context.
 * @return 0 on success.
 * Functional Utility: Sets the initial internal state of the SHA-256 algorithm.
 */
static int crypto_sha256_init(struct shash_desc *desc)
{
	sha256_block_init(shash_desc_ctx(desc));
	return 0;
}

/**
 * @brief Flexible update function for SHA-256, allowing generic or optimized processing.
 * @details This inline function processes input data for SHA-256. It calculates
 * the number of full blocks and updates the `count` in the hash state. It then
 * dispatches the actual block processing to `sha256_choose_blocks`, which can
 * either force a generic software implementation or dynamically select an
 * architecture-optimized version based on the `force_generic` and `use_arch` flags.
 * A compile-time check (`BUILD_BUG_ON`) ensures `crypto_sha256_state` alignment.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data Pointer to the input data.
 * @param len Length of the input data in bytes.
 * @param force_generic If `true`, forces the generic SHA-256 block function.
 * @return The number of remaining bytes in the partial block.
 * Functional Utility: Updates the SHA-256 hash state by processing data blocks, with flexible implementation choice.
 */
static inline int crypto_sha256_update(struct shash_desc *desc, const u8 *data,
				       unsigned int len, bool force_generic)
{
	struct crypto_sha256_state *sctx = shash_desc_ctx(desc);
	int remain = len % SHA256_BLOCK_SIZE;

	/*
	 * Functional Utility: Compile-time assertion to ensure that `struct crypto_sha256_state`
	 * begins directly with its 256-bit internal state, as this is a strict
	 * requirement for the assembly functions to correctly access and manipulate
	 * the hash state. This guarantees memory layout compatibility.
	 */
	BUILD_BUG_ON(offsetof(struct crypto_sha256_state, state) != 0);

	sctx->count += len - remain; // Functional Utility: Updates the total byte count, excluding the partial block.
	// Functional Utility: Dispatches to the appropriate block processing function (generic or architecture-optimized).
	sha256_choose_blocks(sctx->state, data, len / SHA256_BLOCK_SIZE,
			     force_generic, !force_generic);
	return remain;
}

/**
 * @brief `shash` 'update' callback using the generic SHA-256 implementation.
 * @details This function is an 'update' entry point that explicitly forces
 * the use of the generic software implementation of SHA-256 block processing,
 * even if architecture-optimized versions are available. This can be useful
 * for debugging or specific testing scenarios.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data Pointer to the input data.
 * @param len Length of the input data in bytes.
 * @return The number of remaining bytes in the partial block.
 * Functional Utility: Updates SHA-256 state using the generic implementation.
 */
static int crypto_sha256_update_generic(struct shash_desc *desc, const u8 *data,
					unsigned int len)
{
	return crypto_sha256_update(desc, data, len, true);
}

/**
 * @brief `shash` 'update' callback directly calling the library's SHA-256 update function.
 * @details This function is an 'update' entry point that directly calls the
 * underlying SHA-256 library's `sha256_update` function, bypassing the internal
 * `sha256_choose_blocks` mechanism. This is used for algorithms defined with
 * the `sha256-lib` driver name, which might have different buffering or internal
 * logic managed by the library itself.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data Pointer to the input data.
 * @param len Length of the input data in bytes.
 * @return 0 on success.
 * Functional Utility: Updates SHA-256 state by calling the underlying library's update function.
 */
static int crypto_sha256_update_lib(struct shash_desc *desc, const u8 *data,
				    unsigned int len)
{
	sha256_update(shash_desc_ctx(desc), data, len);
	return 0;
}

/**
 * @brief `shash` 'update' callback using the architecture-optimized SHA-256 implementation.
 * @details This function is an 'update' entry point that allows the dynamic
 * selection of the best architecture-optimized SHA-256 block processing function
 * available on the current CPU. It explicitly does not force the generic implementation.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data Pointer to the input data.
 * @param len Length of the input data in bytes.
 * @return The number of remaining bytes in the partial block.
 * Functional Utility: Updates SHA-256 state using the architecture-optimized implementation.
 */
static int crypto_sha256_update_arch(struct shash_desc *desc, const u8 *data,
				     unsigned int len)
{
	return crypto_sha256_update(desc, data, len, false);
}

/**
 * @brief `shash` 'final' callback directly calling the library's SHA-256 finalization.
 * @details This function is a 'final' entry point that directly calls the
 * underlying SHA-256 library's `sha256_final` function. It is used by algorithms
 * defined with the `sha256-lib` driver name.
 *
 * @param desc The `shash_desc` descriptor.
 * @param out Buffer to store the resulting 32-byte SHA-256 digest.
 * @return 0 on success.
 * Functional Utility: Finalizes SHA-256 computation by calling the underlying library's final function.
 */
static int crypto_sha256_final_lib(struct shash_desc *desc, u8 *out)
{
	sha256_final(shash_desc_ctx(desc), out);
	return 0;
}

/**
 * @brief Flexible finup function for SHA-256, handling final data and padding.
 * @details This inline function performs the final update and digest computation
 * for SHA-256. It first processes any full blocks in the remaining data using
 * `crypto_sha256_update`. Then, it handles the remaining partial block by copying
 * it to an internal buffer, updating the `count`, and finally calling `sha256_finup`
 * to apply padding and produce the final digest. It also allows forcing a
 * generic implementation or using an optimized one.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data Pointer to the final input data chunk.
 * @param len Length of the final data chunk in bytes.
 * @param out Buffer to store the resulting 32-byte SHA-256 digest.
 * @param force_generic If `true`, forces the generic SHA-256 block function.
 * @return 0 on success.
 * Functional Utility: Completes SHA-256 hashing, including padding and digest output, with flexible implementation choice.
 */
static __always_inline int crypto_sha256_finup(struct shash_desc *desc,
					       const u8 *data,
					       unsigned int len, u8 *out,
					       bool force_generic)
{
	struct crypto_sha256_state *sctx = shash_desc_ctx(desc);
	unsigned int remain = len;
	u8 *buf;

	// Functional Utility: Processes any full blocks within the final data chunk.
	if (len >= SHA256_BLOCK_SIZE)
		remain = crypto_sha256_update(desc, data, len, force_generic);
	sctx->count += remain; // Functional Utility: Updates the total byte count with the remaining partial block size.
	// Functional Utility: Copies the remaining partial block to an internal buffer.
	buf = memcpy(sctx + 1, data + len - remain, remain);
	// Functional Utility: Applies SHA-256 padding, processes the final block, and produces the digest.
	sha256_finup(sctx, buf, remain, out,
		     crypto_shash_digestsize(desc->tfm), force_generic,
		     !force_generic);
	return 0;
}

/**
 * @brief `shash` 'finup' callback using the generic SHA-256 implementation.
 * @details This function is a 'finup' entry point that explicitly forces
 * the use of the generic software implementation of SHA-256 for finalization,
 * including padding and digest output.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data Pointer to the final input data chunk.
 * @param len Length of the final data chunk in bytes.
 * @param out Buffer to store the resulting 32-byte SHA-256 digest.
 * @return 0 on success.
 * Functional Utility: Finalizes SHA-256 computation using the generic implementation.
 */
static int crypto_sha256_finup_generic(struct shash_desc *desc, const u8 *data,
				       unsigned int len, u8 *out)
{
	return crypto_sha256_finup(desc, data, len, out, true);
}

/**
 * @brief `shash` 'finup' callback using the architecture-optimized SHA-256 implementation.
 * @details This function is a 'finup' entry point that allows the dynamic
 * selection of the best architecture-optimized SHA-256 finalization function
 * available on the current CPU, including padding and digest output.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data Pointer to the final input data chunk.
 * @param len Length of the final data chunk in bytes.
 * @param out Buffer to store the resulting 32-byte SHA-256 digest.
 * @return 0 on success.
 * Functional Utility: Finalizes SHA-256 computation using the architecture-optimized implementation.
 */
static int crypto_sha256_finup_arch(struct shash_desc *desc, const u8 *data,
				    unsigned int len, u8 *out)
{
	return crypto_sha256_finup(desc, data, len, out, false);
}

/**
 * @brief `shash` 'digest' callback using the generic SHA-256 implementation.
 * @details This function provides a one-shot digest computation for SHA-256,
 * explicitly using the generic software implementation. It initializes the
 * hash state and then calls `crypto_sha256_finup_generic` to process all data
 * and produce the final digest.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data Pointer to the entire input data.
 * @param len Length of the input data in bytes.
 * @param out Buffer to store the resulting 32-byte SHA-256 digest.
 * @return 0 on success.
 * Functional Utility: Computes a one-shot SHA-256 digest using the generic implementation.
 */
static int crypto_sha256_digest_generic(struct shash_desc *desc, const u8 *data,
					unsigned int len, u8 *out)
{
	crypto_sha256_init(desc);
	return crypto_sha256_finup_generic(desc, data, len, out);
}

/**
 * @brief `shash` 'digest' callback directly calling the library's SHA-256 one-shot function.
 * @details This function provides a one-shot digest computation for SHA-256
 * by directly calling the underlying SHA-256 library's `sha256` function.
 * It is used for algorithms defined with the `sha256-lib` driver name.
 *
 * @param desc The `shash_desc` descriptor (not used by direct library call).
 * @param data Pointer to the entire input data.
 * @param len Length of the input data in bytes.
 * @param out Buffer to store the resulting 32-byte SHA-256 digest.
 * @return 0 on success.
 * Functional Utility: Computes a one-shot SHA-256 digest by directly calling the underlying library function.
 */
static int crypto_sha256_digest_lib(struct shash_desc *desc, const u8 *data,
				    unsigned int len, u8 *out)
{
	sha256(data, len, out);
	return 0;
}

/**
 * @brief `shash` 'digest' callback using the architecture-optimized SHA-256 implementation.
 * @details This function provides a one-shot digest computation for SHA-256,
 * using the architecture-optimized implementation. It initializes the
 * hash state and then calls `crypto_sha256_finup_arch` to process all data
 * and produce the final digest.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data Pointer to the entire input data.
 * @param len Length of the input data in bytes.
 * @param out Buffer to store the resulting 32-byte SHA-256 digest.
 * @return 0 on success.
 * Functional Utility: Computes a one-shot SHA-256 digest using the architecture-optimized implementation.
 */
static int crypto_sha256_digest_arch(struct shash_desc *desc, const u8 *data,
				     unsigned int len, u8 *out)
{
	crypto_sha256_init(desc);
	return crypto_sha256_finup_arch(desc, data, len, out);
}

/**
 * @brief Initializes the SHA-224 hash state.
 * @details This function is the `init` callback for SHA-224 algorithms registered
 * with the Crypto API. It sets the initial hash values (IVs) for SHA-224
 * in the provided `crypto_sha256_state` context, preparing it for a new hashing operation.
 * Note that SHA-224 uses the same internal state structure as SHA-256.
 *
 * @param desc The `shash_desc` descriptor containing the SHA-224 state context.
 * @return 0 on success.
 * Functional Utility: Sets the initial internal state of the SHA-224 algorithm.
 */
static int crypto_sha224_init(struct shash_desc *desc)
{
	sha224_block_init(shash_desc_ctx(desc));
	return 0;
}

/**
 * @brief `shash` 'final' callback directly calling the library's SHA-224 finalization.
 * @details This function is a 'final' entry point that directly calls the
 * underlying SHA-224 library's `sha224_final` function. It is used by algorithms
 * defined with the `sha224-lib` driver name.
 *
 * @param desc The `shash_desc` descriptor.
 * @param out Buffer to store the resulting 28-byte SHA-224 digest.
 * @return 0 on success.
 * Functional Utility: Finalizes SHA-224 computation by calling the underlying library's final function.
 */
static int crypto_sha224_final_lib(struct shash_desc *desc, u8 *out)
{
	sha224_final(shash_desc_ctx(desc), out);
	return 0;
}

/**
 * @brief Imports SHA-256 state from an external buffer.
 * @details This function is the `import` callback for SHA-256 algorithms.
 * It reconstructs the `crypto_sha256_state` context from a serialized buffer `in`.
 * This typically involves copying the internal hash state and the total byte count.
 * A partial block count is stored separately as a single byte.
 *
 * @param desc The `shash_desc` descriptor.
 * @param in Pointer to the buffer containing the serialized SHA-256 state.
 * @return 0 on success.
 * Functional Utility: Restores a SHA-256 hash state from a previously exported representation.
 */
static int crypto_sha256_import_lib(struct shash_desc *desc, const void *in)
{
	struct sha256_state *sctx = shash_desc_ctx(desc);
	const u8 *p = in;

	memcpy(sctx, p, sizeof(*sctx)); // Functional Utility: Copies the main state structure.
	p += sizeof(*sctx);
	sctx->count += *p; // Functional Utility: Restores the partial block count.
	return 0;
}

/**
 * @brief Exports SHA-256 state to an external buffer.
 * @details This function is the `export` callback for SHA-256 algorithms.
 * It serializes the current `crypto_sha256_state` context into a buffer `out`.
 * This allows the hash state to be saved and later restored, or transferred
 * between different contexts. The total byte count is adjusted to separate
 * the partial block count for storage.
 *
 * @param desc The `shash_desc` descriptor.
 * @param out Pointer to the buffer where the SHA-256 state will be exported.
 * @return 0 on success.
 * Functional Utility: Saves the current SHA-256 hash state into an external buffer for later import.
 */
static int crypto_sha256_export_lib(struct shash_desc *desc, void *out)
{
	struct sha256_state *sctx0 = shash_desc_ctx(desc);
	struct sha256_state sctx = *sctx0;
	unsigned int partial;
	u8 *p = out;

	partial = sctx.count % SHA256_BLOCK_SIZE; // Functional Utility: Calculates the size of the partial block.
	sctx.count -= partial; // Functional Utility: Adjusts count to represent full blocks only.
	memcpy(p, &sctx, sizeof(sctx)); // Functional Utility: Copies the main state structure.
	p += sizeof(sctx);
	*p = partial; // Functional Utility: Stores the partial block count as a separate byte.
	return 0;
}

/**
 * @brief Array of `shash_alg` structures for various SHA-256/224 implementations.
 * @details This array defines and registers multiple SHA-256 and SHA-224
 * algorithm implementations with the Linux kernel's cryptographic API. It includes
 * generic, library-based, and architecture-optimized versions for both hash functions.
 * Each entry specifies the algorithm's properties, associated callbacks (`init`,
 * `update`, `finup`, `digest`, `import`, `export`), `cra_priority`, and flags,
 * allowing the kernel to select the most appropriate implementation.
 * Functional Utility: Defines and registers multiple SHA-256 and SHA-224 algorithms with different optimization levels.
 */
static struct shash_alg algs[] = {
	{ /* Generic SHA-256 implementation */
		.base.cra_name		= "sha256",
		.base.cra_driver_name	= "sha256-generic",
		.base.cra_priority	= 100, // Functional Utility: Low priority ensures optimized versions are preferred.
		.base.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
					  CRYPTO_AHASH_ALG_FINUP_MAX,
		.base.cra_blocksize	= SHA256_BLOCK_SIZE,
		.base.cra_module	= THIS_MODULE,
		.digestsize		= SHA256_DIGEST_SIZE,
		.init			= crypto_sha256_init,
		.update			= crypto_sha256_update_generic,
		.finup			= crypto_sha256_finup_generic,
		.digest			= crypto_sha256_digest_generic,
		.descsize		= sizeof(struct crypto_sha256_state),
	},
	{ /* Generic SHA-224 implementation */
		.base.cra_name		= "sha224",
		.base.cra_driver_name	= "sha224-generic",
		.base.cra_priority	= 100, // Functional Utility: Low priority ensures optimized versions are preferred.
		.base.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
					  CRYPTO_AHASH_ALG_FINUP_MAX,
		.base.cra_blocksize	= SHA224_BLOCK_SIZE,
		.base.cra_module	= THIS_MODULE,
		.digestsize		= SHA224_DIGEST_SIZE,
		.init			= crypto_sha224_init,
		.update			= crypto_sha256_update_generic,
		.finup			= crypto_sha256_finup_generic,
		.descsize		= sizeof(struct crypto_sha256_state),
	},
	{ /* Library-based SHA-256 implementation (direct library calls) */
		.base.cra_name		= "sha256",
		.base.cra_driver_name	= "sha256-lib",
		.base.cra_blocksize	= SHA256_BLOCK_SIZE,
		.base.cra_module	= THIS_MODULE,
		.digestsize		= SHA256_DIGEST_SIZE,
		.init			= crypto_sha256_init,
		.update			= crypto_sha256_update_lib,
		.final			= crypto_sha256_final_lib,
		.digest			= crypto_sha256_digest_lib,
		.descsize		= sizeof(struct sha256_state),
		.statesize		= sizeof(struct crypto_sha256_state) +
					  SHA256_BLOCK_SIZE + 1,
		.import			= crypto_sha256_import_lib,
		.export			= crypto_sha256_export_lib,
	},
	{ /* Library-based SHA-224 implementation (direct library calls) */
		.base.cra_name		= "sha224",
		.base.cra_driver_name	= "sha224-lib",
		.base.cra_blocksize	= SHA224_BLOCK_SIZE,
		.base.cra_module	= THIS_MODULE,
		.digestsize		= SHA224_DIGEST_SIZE,
		.init			= crypto_sha224_init,
		.update			= crypto_sha256_update_lib,
		.final			= crypto_sha224_final_lib,
		.descsize		= sizeof(struct sha256_state),
		.statesize		= sizeof(struct crypto_sha256_state) +
					  SHA256_BLOCK_SIZE + 1,
		.import			= crypto_sha256_import_lib,
		.export			= crypto_sha256_export_lib,
	},
	{ /* Architecture-optimized SHA-256 implementation */
		.base.cra_name		= "sha256",
		.base.cra_driver_name	= "sha256-" __stringify(ARCH), // Functional Utility: Driver name includes architecture for uniqueness.
		.base.cra_priority	= 300, // Functional Utility: High priority for architecture-optimized implementation.
		.base.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
					  CRYPTO_AHASH_ALG_FINUP_MAX,
		.base.cra_blocksize	= SHA256_BLOCK_SIZE,
		.base.cra_module	= THIS_MODULE,
		.digestsize		= SHA256_DIGEST_SIZE,
		.init			= crypto_sha256_init,
		.update			= crypto_sha256_update_arch,
		.finup			= crypto_sha256_finup_arch,
		.digest			= crypto_sha256_digest_arch,
		.descsize		= sizeof(struct crypto_sha256_state),
	},
	{ /* Architecture-optimized SHA-224 implementation */
		.base.cra_name		= "sha224",
		.base.cra_driver_name	= "sha224-" __stringify(ARCH), // Functional Utility: Driver name includes architecture for uniqueness.
		.base.cra_priority	= 300, // Functional Utility: High priority for architecture-optimized implementation.
		.base.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
					  CRYPTO_AHASH_ALG_FINUP_MAX,
		.base.cra_blocksize	= SHA224_BLOCK_SIZE,
		.base.cra_module	= THIS_MODULE,
		.digestsize		= SHA224_DIGEST_SIZE,
		.init			= crypto_sha224_init,
		.update			= crypto_sha256_update_arch,
		.finup			= crypto_sha256_finup_arch,
		.descsize		= sizeof(struct crypto_sha256_state),
	},
};

static unsigned int num_algs; // Functional Role: Tracks the number of algorithms currently registered.

/**
 * @brief Module initialization function for the SHA-256/224 Crypto API wrapper.
 * @details This function is the entry point when the module is loaded. It
 * conditionally registers various SHA-256 and SHA-224 algorithms (generic,
 * library-based, and architecture-optimized) with the kernel's cryptographic API.
 * It dynamically adjusts the number of algorithms to register based on whether
 * architecture-optimized versions are distinct from the generic ones (`sha256_is_arch_optimized()`).
 *
 * @return 0 on successful registration of all relevant algorithms, or an error code otherwise.
 * Functional Utility: Registers all SHA-256/224 algorithms, adapting to CPU capabilities.
 */
static int __init crypto_sha256_mod_init(void)
{
	num_algs = ARRAY_SIZE(algs);
	// Functional Utility: Compile-time check to ensure a minimum number of algorithms are defined.
	BUILD_BUG_ON(ARRAY_SIZE(algs) <= 2);
	// Functional Utility: If architecture-optimized algorithms are not distinct from generic, reduce the count.
	if (!sha256_is_arch_optimized())
		num_algs -= 2;
	// Functional Utility: Registers all applicable SHA-256/224 algorithms with the crypto API.
	return crypto_register_shashes(algs, ARRAY_SIZE(algs));
}
module_init(crypto_sha256_mod_init);

/**
 * @brief Module cleanup function for the SHA-256/224 Crypto API wrapper.
 * @details This function is the exit point when the module is unloaded. It
 * unregisters all previously registered SHA-256 and SHA-224 algorithms from
 * the cryptographic API, ensuring a clean and proper release of resources and
 * preventing any lingering references after the module is no longer in use.
 * Functional Utility: Unregisters all SHA-256/224 algorithms, ensuring a clean module exit.
 */
static void __exit crypto_sha256_mod_exit(void)
{
	crypto_unregister_shashes(algs, num_algs);
}
module_exit(crypto_sha256_mod_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Crypto API wrapper for the SHA-256 and SHA-224 library functions");

MODULE_ALIAS_CRYPTO("sha256");
MODULE_ALIAS_CRYPTO("sha256-generic");
MODULE_ALIAS_CRYPTO("sha256-" __stringify(ARCH));
MODULE_ALIAS_CRYPTO("sha224");
MODULE_ALIAS_CRYPTO("sha224-generic");
MODULE_ALIAS_CRYPTO("sha224-" __stringify(ARCH));
