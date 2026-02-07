// SPDX-License-Identifier: GPL-2.0-or-later
/**
 * @file sha512_generic.c
 * @brief Generic, portable implementation of the SHA-512 and SHA-384 Secure Hash Algorithms.
 * @details This file provides a hardware-agnostic and fully portable implementation
 * of the SHA-512 and SHA-384 hash algorithms, conforming to the Linux kernel's
 * cryptographic API (`shash`). It serves as a reliable fallback or default
 * implementation for platforms and architectures that do not have a more
 * optimized or hardware-accelerated version of SHA-512/384 available. The
 * implementation is based on the `shash` (synchronous hash) interface,
 * providing standard `init`, `update`, and `finup` operations.
 */
/* SHA-512 code by Jean-Luc Cooke <jlcooke@certainkey.com>
 *
 * Copyright (c) Jean-Luc Cooke <jlcooke@certainkey.com>
 * Copyright (c) Andrew McDonald <andrew@mcdonald.org.uk>
 * Copyright (c) 2003 Kyle McMartin <kyle@debian.org>
 */
#include <crypto/internal/hash.h>
#include <crypto/sha2.h>
#include <crypto/sha512_base.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/unaligned.h>

/**
 * @var sha384_zero_message_hash
 * @brief The pre-computed SHA-384 hash of a zero-length message.
 *
 * @details This constant array contains the well-known 48-byte SHA-384 digest
 * for an empty input string or message. This value is defined by the SHA-2
 * standard for its truncated variant, SHA-384. It is used for verification
 * purposes, self-tests, and can enable minor optimizations for empty inputs.
 * Functional Utility: Provides the canonical hash for an empty message in SHA-384.
 */
const u8 sha384_zero_message_hash[SHA384_DIGEST_SIZE] = {
	0x38, 0xb0, 0x60, 0xa7, 0x51, 0xac, 0x96, 0x38,
	0x4c, 0xd9, 0x32, 0x7e, 0xb1, 0xb1, 0xe3, 0x6a,
	0x21, 0xfd, 0xb7, 0x11, 0x14, 0xbe, 0x07, 0x43,
	0x4c, 0x0c, 0xc7, 0xbf, 0x63, 0xf6, 0xe1, 0xda,
	0x27, 0x4e, 0xde, 0xbf, 0xe7, 0x6f, 0x65, 0xfb,
	0xd5, 0x1a, 0xd2, 0xf1, 0x48, 0x98, 0xb9, 0x5b
};
EXPORT_SYMBOL_GPL(sha384_zero_message_hash);

/**
 * @var sha512_zero_message_hash
 * @brief The pre-computed SHA-512 hash of a zero-length message.
 *
 * @details This constant array contains the well-known 64-byte SHA-512 digest
 * for an empty input string or message. This value is defined by the SHA-2
 * standard. It is used for verification purposes, self-tests, and can enable
 * minor optimizations for empty inputs.
 * Functional Utility: Provides the canonical hash for an empty message in SHA-512.
 */
const u8 sha512_zero_message_hash[SHA512_DIGEST_SIZE] = {
	0xcf, 0x83, 0xe1, 0x35, 0x7e, 0xef, 0xb8, 0xbd,
	0xf1, 0x54, 0x28, 0x50, 0xd6, 0x6d, 0x80, 0x07,
	0xd6, 0x20, 0xe4, 0x05, 0x0b, 0x57, 0x15, 0xdc,
	0x83, 0xf4, 0xa9, 0x21, 0xd3, 0x6c, 0xe9, 0xce,
	0x47, 0xd0, 0xd1, 0x3c, 0x5d, 0x85, 0xf2, 0xb0,
	0xff, 0x83, 0x18, 0xd2, 0x87, 0x7e, 0xec, 0x2f,
	0x63, 0xb9, 0x31, 0xbd, 0x47, 0x41, 0x7a, 0x81,
	0xa5, 0x38, 0x32, 0x7a, 0xf9, 0x27, 0xda, 0x3e
};
EXPORT_SYMBOL_GPL(sha512_zero_message_hash);

/**
 * @brief Implements the SHA-512 cryptographic Choose function (Ch).
 * @details This inline function computes the SHA-512 logical function `Ch(x, y, z)`,
 * which is `(x AND y) XOR (NOT x AND z)` or equivalently `z XOR (x AND (y XOR z))`.
 * This function is one of the three core bitwise logical functions used in each
 * round of the SHA-512 compression algorithm.
 *
 * @param x Input 64-bit word.
 * @param y Input 64-bit word.
 * @param z Input 64-bit word.
 * @return The 64-bit result of the Choose function.
 * Functional Utility: Computes the SHA-512 Choose function.
 */
static inline u64 Ch(u64 x, u64 y, u64 z)
{
        return z ^ (x & (y ^ z));
}

/**
 * @brief Implements the SHA-512 cryptographic Majority function (Maj).
 * @details This inline function computes the SHA-512 logical function `Maj(x, y, z)`,
 * which is `(x AND y) XOR (x AND z) XOR (y AND z)` or equivalently `(x AND y) | (z AND (x | y))`.
 * This function is one of the three core bitwise logical functions used in each
 * round of the SHA-512 compression algorithm.
 *
 * @param x Input 64-bit word.
 * @param y Input 64-bit word.
 * @param z Input 64-bit word.
 * @return The 64-bit result of the Majority function.
 * Functional Utility: Computes the SHA-512 Majority function.
 */
static inline u64 Maj(u64 x, u64 y, u64 z)
{
        return (x & y) | (z & (x | y));
}

/**
 * @var sha512_K
 * @brief Array of SHA-512 round constants.
 * @details This constant array contains the 80 64-bit integer values (`K_t`)
 * used in the SHA-512 compression algorithm. These constants are derived from
 * the cube roots of the first 80 prime numbers. Each constant is added once
 * per round in the main compression loop.
 * Functional Utility: Provides the immutable SHA-512 round constants for cryptographic operations.
 */
static const u64 sha512_K[80] = {
        0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL,
        0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL,
        0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL,
        0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
        0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL,
        0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL,
        0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL, 0x2de92c6f592b0275ULL,
        0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
        0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL,
        0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
        0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL,
        0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
        0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL,
        0x92722c851482353bULL, 0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL,
        0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL,
        0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
        0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL,
        0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL,
        0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL,
        0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
        0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL,
        0xc67178f2e372532bULL, 0xca273eceea26619cULL, 0xd186b8c721c0c207ULL,
        0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL,
        0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
        0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL,
        0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL,
        0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL,
};

/**
 * @brief Implements SHA-512 Sigma0 (uppercase sigma0) function.
 * @details This macro computes the SHA-512 logical function `Sigma0(x)`,
 * which is `(x ROR 28) XOR (x ROR 34) XOR (x ROR 39)`. This function is
 * applied to the `a` value in each round of the SHA-512 compression.
 * @param x Input 64-bit word.
 * @return The 64-bit result of the Sigma0 function.
 * Functional Utility: Computes the SHA-512 Sigma0 function.
 */
#define e0(x)       (ror64(x,28) ^ ror64(x,34) ^ ror64(x,39))
/**
 * @brief Implements SHA-512 Sigma1 (uppercase sigma1) function.
 * @details This macro computes the SHA-512 logical function `Sigma1(x)`,
 * which is `(x ROR 14) XOR (x ROR 18) XOR (x ROR 41)`. This function is
 * applied to the `e` value in each round of the SHA-512 compression.
 * @param x Input 64-bit word.
 * @return The 64-bit result of the Sigma1 function.
 * Functional Utility: Computes the SHA-512 Sigma1 function.
 */
#define e1(x)       (ror64(x,14) ^ ror64(x,18) ^ ror64(x,41))
/**
 * @brief Implements SHA-512 sigma0 (lowercase sigma0) function.
 * @details This macro computes the SHA-512 logical function `sigma0(x)`,
 * which is `(x ROR 1) XOR (x ROR 8) XOR (x SHR 7)`. This function is
 * used in the message schedule generation to derive new message words.
 * @param x Input 64-bit word.
 * @return The 64-bit result of the sigma0 function.
 * Functional Utility: Computes the SHA-512 sigma0 function for message schedule generation.
 */
#define s0(x)       (ror64(x, 1) ^ ror64(x, 8) ^ (x >> 7))
/**
 * @brief Implements SHA-512 sigma1 (lowercase sigma1) function.
 * @details This macro computes the SHA-512 logical function `sigma1(x)`,
 * which is `(x ROR 19) XOR (x ROR 61) XOR (x SHR 6)`. This function is
 * used in the message schedule generation to derive new message words.
 * @param x Input 64-bit word.
 * @return The 64-bit result of the sigma1 function.
 * Functional Utility: Computes the SHA-512 sigma1 function for message schedule generation.
 */
#define s1(x)       (ror64(x,19) ^ ror64(x,61) ^ (x >> 6))

/**
 * @brief Loads an unaligned 64-bit big-endian word.
 * @details This inline function loads a 64-bit word from the `input` buffer
 * at index `I`, handling potential unaligned memory access and converting it
 * from big-endian format to the host's native endianness. It's a helper for
 * preparing message words.
 *
 * @param I Index into the message word array.
 * @param W Pointer to the 64-bit message word array.
 * @param input Pointer to the input data buffer.
 * Functional Utility: Reads and endian-converts a 64-bit word from the input block.
 */
static inline void LOAD_OP(int I, u64 *W, const u8 *input)
{
	W[I] = get_unaligned_be64((__u64 *)input + I);
}

/**
 * @brief Blends new message words into the message schedule.
 * @details This inline function implements the message schedule generation
 * logic for SHA-512. For `I >= 16`, it computes `W[I] = sigma1(W[I-2]) + W[I-7] + sigma0(W[I-15]) + W[I-16]`.
 * The modulo 16 arithmetic (`I & 15`) is used to index into the `W` array,
 * which acts as a circular buffer for the 16 most recent message words.
 *
 * @param I Current index for message word generation.
 * @param W Pointer to the 64-bit message word array (circular buffer).
 * Functional Utility: Generates new message schedule words based on previous words for SHA-512.
 */
static inline void BLEND_OP(int I, u64 *W)
{
	W[I & 15] += s1(W[(I-2) & 15]) + W[(I-7) & 15] + s0(W[(I-15) & 15]);
}

/**
 * @brief The core SHA-512 compression function for a single 128-byte block.
 * @details This function implements the main SHA-512 compression loop,
 * processing a single 128-byte input block (`input`) and updating the
 * 512-bit intermediate hash state (`state`). It performs 80 rounds of
 * computation. In each round, it updates eight working variables (`a` through `h`)
 * based on the previous round's values, message words, and round constants (`sha512_K`).
 * The message schedule (`W`) is dynamically generated as needed.
 *
 * @param state Pointer to the 8x `u64` array representing the intermediate hash state (H0-H7).
 * @param input Pointer to the 128-byte input data block.
 * Functional Utility: Applies the SHA-512 compression algorithm to a single data block.
 */
static void
sha512_transform(u64 *state, const u8 *input)
{
	u64 a, b, c, d, e, f, g, h, t1, t2;

	int i;
	u64 W[16]; // Functional Role: Circular buffer for 16 message words (W[0] to W[15]).

	/* load the state into our registers */
	// Functional Utility: Loads the intermediate hash values from `state` into local working variables.
	a=state[0];   b=state[1];   c=state[2];   d=state[3];
	e=state[4];   f=state[5];   g=state[6];   h=state[7];

	/* now iterate */
	// Block Logic: Main loop for 80 rounds, processing 8 rounds per iteration (i += 8).
	for (i=0; i<80; i+=8) {
		// Block Logic: Message schedule generation for initial 16 words (i < 16) or blending for subsequent words.
		if (!(i & 8)) { // This condition is effectively true for i=0, 16, 32, ...
			int j;

			if (i < 16) {
				/* load the input */
				// Functional Utility: Loads initial 16 message words from input, handling unaligned big-endian data.
				for (j = 0; j < 16; j++)
					LOAD_OP(i + j, W, input);
			} else {
				// Functional Utility: Blends (generates) subsequent message words based on the SHA-512 schedule.
				for (j = 0; j < 16; j++) {
					BLEND_OP(i + j, W);
				}
			}
		}

		// Block Logic: The 8 rounds of SHA-512 compression, computed per loop iteration.
		// Each t1/t2 calculation represents one step of the round function, updating a/e/d/h.
		// Functional Utility: Round i computations.
		t1 = h + e1(e) + Ch(e,f,g) + sha512_K[i  ] + W[(i & 15)];
		t2 = e0(a) + Maj(a,b,c);    d+=t1;    h=t1+t2;
		// Functional Utility: Round i+1 computations.
		t1 = g + e1(d) + Ch(d,e,f) + sha512_K[i+1] + W[(i & 15) + 1];
		t2 = e0(h) + Maj(h,a,b);    c+=t1;    g=t1+t2;
		// Functional Utility: Round i+2 computations.
		t1 = f + e1(c) + Ch(c,d,e) + sha512_K[i+2] + W[(i & 15) + 2];
		t2 = e0(g) + Maj(g,h,a);    b+=t1;    f=t1+t2;
		// Functional Utility: Round i+3 computations.
		t1 = e + e1(b) + Ch(b,c,d) + sha512_K[i+3] + W[(i & 15) + 3];
		t2 = e0(f) + Maj(f,g,h);    a+=t1;    e=t1+t2;
		// Functional Utility: Round i+4 computations.
		t1 = d + e1(a) + Ch(a,b,c) + sha512_K[i+4] + W[(i & 15) + 4];
		t2 = e0(e) + Maj(e,f,g);    h+=t1;    d=t1+t2;
		// Functional Utility: Round i+5 computations.
		t1 = c + e1(h) + Ch(h,a,b) + sha512_K[i+5] + W[(i & 15) + 5];
		t2 = e0(d) + Maj(d,e,f);    g+=t1;    c=t1+t2;
		// Functional Utility: Round i+6 computations.
		t1 = b + e1(g) + Ch(g,h,a) + sha512_K[i+6] + W[(i & 15) + 6];
		t2 = e0(c) + Maj(c,d,e);    f+=t1;    b=t1+t2;
		// Functional Utility: Round i+7 computations.
		t1 = a + e1(f) + Ch(f,g,h) + sha512_K[i+7] + W[(i & 15) + 7];
		t2 = e0(b) + Maj(b,c,d);    e+=t1;    a=t1+t2;
	}

	// Functional Utility: Accumulates the results of the compression function into the original state array.
	state[0] += a; state[1] += b; state[2] += c; state[3] += d;
	state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

/**
 * @brief Processes one or more full blocks of SHA-512 data.
 * @details This function is the core processing loop for the generic SHA-512
 * and SHA-384 implementations. It repeatedly calls the `sha512_transform`
 * function for each 128-byte block of input data (`src`), updating the
 * algorithm's internal state (`sst->state`).
 *
 * @param sst Pointer to the SHA-512 state structure.
 * @param src Pointer to the source data blocks.
 * @param blocks The number of 128-byte SHA-512 blocks to process.
 * Functional Utility: Iteratively applies the SHA-512 compression function to multiple input blocks.
 */
void sha512_generic_block_fn(struct sha512_state *sst, u8 const *src,
			     int blocks)
{
	// Block Logic: Loops through each 128-byte block.
	do {
		// Functional Utility: Applies the core SHA-512 compression function to a single block.
		sha512_transform(sst->state, src);
		// Functional Utility: Advances the source pointer to the next 128-byte block.
		src += SHA512_BLOCK_SIZE;
	} while (--blocks); // Functional Utility: Continues looping until all blocks are processed.
}
EXPORT_SYMBOL_GPL(sha512_generic_block_fn);

/**
 * @brief Updates the SHA-512/384 hash state with new data.
 * @details This function serves as the 'update' entry point for the `shash`
 * API for the generic SHA-512/384 implementation. It feeds a chunk of data into
 * the SHA-512/384 algorithm, efficiently managing any partial blocks buffered
 * from previous updates and processing all new full blocks. It delegates the
 * complex buffering logic to the base helper function `sha512_base_do_update_blocks`
 * and provides `sha512_generic_block_fn` for the actual 128-byte block processing.
 *
 * @param desc The `shash_desc` descriptor containing the hash state.
 * @param data Pointer to the input data.
 * @param len Length of the input data in bytes.
 * @return 0 on success.
 * Functional Utility: Increments the SHA-512/384 hash state with new data, handling partial blocks and using a generic block processor.
 */
static int crypto_sha512_update(struct shash_desc *desc, const u8 *data,
				unsigned int len)
{
	// Functional Utility: Delegates data buffering, partial block handling, and full block processing to the base SHA-512 update function.
	return sha512_base_do_update_blocks(desc, data, len,
					    sha512_generic_block_fn);
}

/**
 * @brief Processes the final data chunk and computes the digest.
 * @details This function combines the 'update' and 'final' steps for the
 * generic SHA-512/384 algorithm. It takes the last piece of input data, applies
 * the necessary SHA-512 padding scheme (including message length append),
 * and computes the final 64-byte (for SHA-512) or 48-byte (for SHA-384) hash value.
 * It leverages the base helper `sha512_base_do_finup` to manage final data processing
 * and padding, and `sha512_base_finish` to write out the digest.
 *
 * @param desc The `shash_desc` descriptor containing the hash state.
 * @param data Pointer to the final input data chunk.
 * @param len Length of the final data chunk in bytes.
 * @param hash Buffer to store the resulting SHA-512/384 digest.
 * @return 0 on success.
 * Functional Utility: Completes the SHA-512/384 hashing process, including padding and final digest generation, using generic block processing.
 */
static int crypto_sha512_finup(struct shash_desc *desc, const u8 *data,
			       unsigned int len, u8 *hash)
{
	// Functional Utility: Handles final data processing, padding, and delegates block processing to `sha512_generic_block_fn`.
	sha512_base_do_finup(desc, data, len, sha512_generic_block_fn);
	// Functional Utility: Writes the final computed SHA-512/384 hash digest to the output buffer.
	return sha512_base_finish(desc, hash);
}

/**
 * @brief Defines and registers the "sha512-generic" and "sha384-generic" algorithms with the Crypto API.
 * @details This array contains two `shash_alg` structures, one for SHA-512
 * and one for SHA-384, providing generic software implementations for each.
 * They are registered with the Linux cryptographic API. Each specifies the
 * digest size, block size, and function pointers for initialization, update,
 * and finalization. The `cra_driver_name` identifies the implementation,
 * and the `cra_priority` is set to a low value (100) to ensure that
 * architecture-specific, more optimized versions are preferred if available.
 * The `CRYPTO_AHASH_ALG_BLOCK_ONLY` flag indicates that these implementations
 * can only process full blocks, relying on the base framework for buffering and padding.
 * Functional Utility: Defines and registers generic SHA-512 and SHA-384 algorithms with the kernel crypto API.
 */
static struct shash_alg sha512_algs[2] = { {
	.digestsize	=	SHA512_DIGEST_SIZE,
	.init		=	sha512_base_init,
	.update		=	crypto_sha512_update,
	.finup		=	crypto_sha512_finup,
	.descsize	=	SHA512_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha512",
		.cra_driver_name =	"sha512-generic",
		.cra_priority	=	100, // Functional Utility: Low priority ensures optimized versions are preferred.
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY |
					CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize	=	SHA512_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
}, {
	.digestsize	=	SHA384_DIGEST_SIZE,
	.init		=	sha384_base_init,
	.update		=	crypto_sha512_update,
	.finup		=	crypto_sha512_finup,
	.descsize	=	SHA512_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha384",
		.cra_driver_name =	"sha384-generic",
		.cra_priority	=	100, // Functional Utility: Low priority ensures optimized versions are preferred.
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY |
					CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize	=	SHA384_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
} };

/**
 * @brief Module initialization function for the generic SHA-512/384 implementation.
 * @details This function is the entry point when the `sha512_generic` kernel
 * module is loaded. It registers both the generic SHA-512 and SHA-384 `shash`
 * algorithms (`sha512_algs` array) with the kernel's cryptographic API. This
 * action makes these algorithms available for other kernel subsystems that
 * require SHA-512 or SHA-384 hashing, serving as a robust and portable baseline.
 *
 * @return 0 on successful registration, or an error code otherwise.
 * Functional Utility: Registers the generic SHA-512 and SHA-384 algorithms, making them available system-wide.
 */
static int __init sha512_generic_mod_init(void)
{
	return crypto_register_shashes(sha512_algs, ARRAY_SIZE(sha512_algs));
}

/**
 * @brief Module cleanup function for the generic SHA-512/384 implementation.
 * @details This function is the exit point when the `sha512_generic` kernel
 * module is unloaded. It unregisters both the generic SHA-512 and SHA-384
 * algorithms (`sha512_algs` array) from the cryptographic API, ensuring a
 * clean and proper release of resources and preventing any lingering
 * references after the module is no longer in use.
 * Functional Utility: Unregisters the generic SHA-512 and SHA-384 algorithms, ensuring a clean module exit.
 */
static void __exit sha512_generic_mod_fini(void)
{
	crypto_unregister_shashes(sha512_algs, ARRAY_SIZE(sha512_algs));
}

module_init(sha512_generic_mod_init);
module_exit(sha512_generic_mod_fini);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SHA-512 and SHA-384 Secure Hash Algorithms");

MODULE_ALIAS_CRYPTO("sha384");
MODULE_ALIAS_CRYPTO("sha384-generic");
MODULE_ALIAS_CRYPTO("sha512");
MODULE_ALIAS_CRYPTO("sha512-generic");
