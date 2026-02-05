/**
 * @file sha1_generic.c
 * @brief Generic, portable implementation of the SHA-1 Secure Hash Algorithm.
 *
 * This file provides a hardware-agnostic implementation of the SHA-1 hash
 * algorithm, conforming to the kernel's cryptographic API. It serves as a
 * fallback or default implementation for platforms that do not have a more
 * optimized or hardware-accelerated version of SHA-1. The implementation is
 * based on the shash (synchronous hash) interface.
 */
// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Cryptographic API.
 *
 * SHA1 Secure Hash Algorithm.
 *
 * Derived from cryptoapi implementation, adapted for in-place
 * scatterlist interface.
 *
 * Copyright (c) Alan Smithee.
 * Copyright (c) Andrew McDonald <andrew@mcdonald.org.uk>
 * Copyright (c) Jean-Francois Dive <jef@linuxbe.org>
 */
#include <crypto/internal/hash.h>
#include <crypto/sha1.h>
#include <crypto/sha1_base.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/string.h>

/**
 * @var sha1_zero_message_hash
 * @brief The pre-computed SHA-1 hash of a zero-length message.
 *
 * Functional Utility: This constant provides the well-known digest for an
 * empty input, as defined by the SHA-1 standard. It can be used for
 * optimizations or self-tests where this specific value is expected.
 */
const u8 sha1_zero_message_hash[SHA1_DIGEST_SIZE] = {
	0xda, 0x39, 0xa3, 0xee, 0x5e, 0x6b, 0x4b, 0x0d,
	0x32, 0x55, 0xbf, 0xef, 0x95, 0x60, 0x18, 0x90,
	0xaf, 0xd8, 0x07, 0x09
};
EXPORT_SYMBOL_GPL(sha1_zero_message_hash);

/**
 * sha1_generic_block_fn - Processes one or more full blocks of SHA-1.
 * @sst: Pointer to the SHA-1 state structure.
 * @src: Pointer to the source data blocks.
 * @blocks: The number of SHA-1 blocks to process.
 *
 * This function is the core processing loop for the generic SHA-1
 * implementation. It repeatedly calls the SHA-1 transform function for each
 * 64-byte block of input data, updating the algorithm's internal state.
 */
static void sha1_generic_block_fn(struct sha1_state *sst, u8 const *src,
				  int blocks)
{
	u32 temp[SHA1_WORKSPACE_WORDS];

	while (blocks--) {
		sha1_transform(sst->state, src, temp);
		src += SHA1_BLOCK_SIZE;
	}
	memzero_explicit(temp, sizeof(temp));
}

/**
 * crypto_sha1_update - Updates the SHA-1 hash state with new data.
 * @desc: The shash descriptor containing the hash state.
 * @data: Pointer to the input data.
 * @len: Length of the input data in bytes.
 *
 * This function serves as the 'update' entry point for the shash API. It
 * feeds a chunk of data into the SHA-1 algorithm, handling any partial
 * blocks from previous updates and processing all new full blocks. It uses
 * the base helper function `sha1_base_do_update_blocks` to manage the state
 * and calls `sha1_generic_block_fn` for the actual block processing.
 *
 * @return 0 on success.
 */
static int crypto_sha1_update(struct shash_desc *desc, const u8 *data,
			      unsigned int len)
{
	return sha1_base_do_update_blocks(desc, data, len,
					  sha1_generic_block_fn);
}

/**
 * crypto_sha1_finup - Processes the final data chunk and computes the digest.
 * @desc: The shash descriptor containing the hash state.
 * @data: Pointer to the final input data chunk.
 * @len: Length of the final data chunk in bytes.
 * @out: Buffer to store the resulting 20-byte SHA-1 digest.
 *
 * This function combines the 'update' and 'final' steps. It processes the
 * last piece of data, applies the necessary SHA-1 padding, and computes the
 * final hash value. It uses the base helper `sha1_base_do_finup` to handle
 * the final data processing and padding.
 *
 * @return 0 on success.
 */
static int crypto_sha1_finup(struct shash_desc *desc, const u8 *data,
			     unsigned int len, u8 *out)
{
	sha1_base_do_finup(desc, data, len, sha1_generic_block_fn);
	return sha1_base_finish(desc, out);
}

/**
 * @struct shash_alg alg
 * @brief Defines and registers the "sha1-generic" algorithm with the Crypto API.
 *
 * Functional Utility: This structure populates the fields required by the
 * cryptographic API to define a synchronous hash algorithm. It specifies the
 * digest size, block size, and function pointers for initialization, update,
 * and finalization. The `cra_driver_name` ("sha1-generic") is the unique
 * identifier for this specific implementation, and the priority is set to a
 * low value (100) so that architecture-specific, more optimized versions are
- * preferred if available.
+ * preferred if available. The `CRYPTO_AHASH_ALG_BLOCK_ONLY` flag indicates that this
+ * implementation can only process full blocks, relying on the base framework for
+ * buffering and padding.
 */
static struct shash_alg alg = {
	.digestsize	=	SHA1_DIGEST_SIZE,
	.init		=	sha1_base_init,
	.update		=	crypto_sha1_update,
	.finup		=	crypto_sha1_finup,
	.descsize	=	SHA1_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha1",
		.cra_driver_name=	"sha1-generic",
		.cra_priority	=	100,
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY,
		.cra_blocksize	=	SHA1_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
};

/**
 * sha1_generic_mod_init - Module initialization function.
 *
 * This function is called when the module is loaded. It registers the generic
 * SHA-1 shash algorithm with the kernel's cryptographic API, making it
 * available for other kernel subsystems to use.
 *
 * @return 0 on successful registration, or an error code otherwise.
 */
static int __init sha1_generic_mod_init(void)
{
	return crypto_register_shash(&alg);
}

/**
 * sha1_generic_mod_fini - Module cleanup function.
 *
 * This function is called when the module is unloaded. It unregisters the
 * generic SHA-1 algorithm from the cryptographic API, ensuring a clean exit.
 */
static void __exit sha1_generic_mod_fini(void)
{
	crypto_unregister_shash(&alg);
}

module_init(sha1_generic_mod_init);
module_exit(sha1_generic_mod_fini);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SHA1 Secure Hash Algorithm");

MODULE_ALIAS_CRYPTO("sha1");
MODULE_ALIAS_CRYPTO("sha1-generic");
