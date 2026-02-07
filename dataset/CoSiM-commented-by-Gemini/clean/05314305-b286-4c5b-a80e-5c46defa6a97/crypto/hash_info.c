/**
 * @file hash_info.c
 * @brief Provides static metadata for hash algorithms in the Linux kernel.
 * @details This file contains constant arrays that serve as a central registry
 * for properties of various cryptographic hash algorithms. It allows other parts
 * of the kernel's cryptographic subsystem to retrieve information like the standard
 * algorithm name and digest size based on a fixed enumeration (`hash_algo`). This
 * approach decouples the rest of the crypto subsystem from hardcoded details of
 * each hash function, promoting modularity and maintainability. It is a key
 * component for managing cryptographic algorithm metadata.
 */

// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Hash Info: Hash algorithms information
 *
 * Copyright (c) 2013 Dmitry Kasatkin <d.kasatkin@samsung.com>
 */

#include <linux/export.h>
#include <crypto/hash_info.h>

/**
 * @var hash_algo_name
 * @brief Maps a hash algorithm enumeration to its standard string identifier.
 *
 * @details This array provides a canonical, human-readable string name for each
 * cryptographic hash algorithm identified by the `hash_algo` enum. This mapping
 * is crucial for several aspects of the Linux kernel's cryptographic framework:
 * - **Logging and Debugging**: Provides clear, descriptive names in kernel logs.
 * - **User-Space Interfaces**: Enables user-space applications to refer to hash
 *   algorithms by their standard names (e.g., via `algif_hash` sockets, procfs, or sysfs).
 * - **Module Loading**: Facilitates dynamic loading of cryptographic modules by name.
 * - **Canonical Naming**: Ensures a consistent and standardized naming convention
 *   across the entire kernel for hash algorithms.
 * Functional Utility: This array provides a canonical, human-readable name
 * for each hash algorithm identified by the `hash_algo` enum. It is used
 * throughout the kernel for logging, user-space interfaces (e.g., via procfs
 * or sysfs), and when dynamically loading cryptographic modules by name.
 */
const char *const hash_algo_name[HASH_ALGO__LAST] = {
	[HASH_ALGO_MD4]		= "md4",
	[HASH_ALGO_MD5]		= "md5",
	[HASH_ALGO_SHA1]	= "sha1",
	[HASH_ALGO_RIPE_MD_160]	= "rmd160",
	[HASH_ALGO_SHA256]	= "sha256",
	[HASH_ALGO_SHA384]	= "sha384",
	[HASH_ALGO_SHA512]	= "sha512",
	[HASH_ALGO_SHA224]	= "sha224",
	[HASH_ALGO_RIPE_MD_128]	= "rmd128",
	[HASH_ALGO_RIPE_MD_256]	= "rmd256",
	[HASH_ALGO_RIPE_MD_320]	= "rmd320",
	[HASH_ALGO_WP_256]	= "wp256",
	[HASH_ALGO_WP_384]	= "wp384",
	[HASH_ALGO_WP_512]	= "wp512",
	[HASH_ALGO_TGR_128]	= "tgr128",
	[HASH_ALGO_TGR_160]	= "tgr160",
	[HASH_ALGO_TGR_192]	= "tgr192",
	[HASH_ALGO_SM3_256]	= "sm3",
	[HASH_ALGO_STREEBOG_256] = "streebog256",
	[HASH_ALGO_STREEBOG_512] = "streebog512",
	[HASH_ALGO_SHA3_256]    = "sha3-256",
	[HASH_ALGO_SHA3_384]    = "sha3-384",
	[HASH_ALGO_SHA3_512]    = "sha3-512",
};
EXPORT_SYMBOL_GPL(hash_algo_name);

/**
 * @var hash_digest_size
 * @brief Maps a hash algorithm enumeration to its output digest size in bytes.
 *
 * @details This array provides the precise size in bytes of the message digest
 * (the fixed-size output hash value) for each algorithm defined in the `hash_algo` enum.
 * This information is critically important for various consumers of the
 * cryptographic API within the kernel and in user-space:
 * - **Memory Allocation**: Allows callers to correctly allocate memory buffers
 *   of the appropriate size to store the hash results, preventing overflows or
 *   wasted memory.
 * - **API Consumers**: Essential for functions that expect a hash output to know
 *   how much data to expect or process.
 * - **Validation Checks**: Used by cryptographic frameworks and drivers to validate
 *   that allocated buffers are large enough or that received digests conform to
 *   expected sizes.
 * Functional Utility: This array provides the precise size of the message
 * digest (hash output) for each algorithm in the `hash_algo` enum. This is
 * critical for consumers of the cryptographic API to correctly allocate memory
 * for hash results and to perform validation checks.
 */
const int hash_digest_size[HASH_ALGO__LAST] = {
	[HASH_ALGO_MD4]		= MD5_DIGEST_SIZE,
	[HASH_ALGO_MD5]		= MD5_DIGEST_SIZE,
	[HASH_ALGO_SHA1]	= SHA1_DIGEST_SIZE,
	[HASH_ALGO_RIPE_MD_160]	= RMD160_DIGEST_SIZE,
	[HASH_ALGO_SHA256]	= SHA256_DIGEST_SIZE,
	[HASH_ALGO_SHA384]	= SHA384_DIGEST_SIZE,
	[HASH_ALGO_SHA512]	= SHA512_DIGEST_SIZE,
	[HASH_ALGO_SHA224]	= SHA224_DIGEST_SIZE,
	[HASH_ALGO_RIPE_MD_128]	= RMD128_DIGEST_SIZE,
	[HASH_ALGO_RIPE_MD_256]	= RMD256_DIGEST_SIZE,
	[HASH_ALGO_RIPE_MD_320]	= RMD320_DIGEST_SIZE,
	[HASH_ALGO_WP_256]	= WP256_DIGEST_SIZE,
	[HASH_ALGO_WP_384]	= WP384_DIGEST_SIZE,
	[HASH_ALGO_WP_512]	= WP512_DIGEST_SIZE,
	[HASH_ALGO_TGR_128]	= TGR128_DIGEST_SIZE,
	[HASH_ALGO_TGR_160]	= TGR160_DIGEST_SIZE,
	[HASH_ALGO_TGR_192]	= TGR192_DIGEST_SIZE,
	[HASH_ALGO_SM3_256]	= SM3256_DIGEST_SIZE,
	[HASH_ALGO_STREEBOG_256] = STREEBOG256_DIGEST_SIZE,
	[HASH_ALGO_STREEBOG_512] = STREEBOG512_DIGEST_SIZE,
	[HASH_ALGO_SHA3_256]    = SHA3_256_DIGEST_SIZE,
	[HASH_ALGO_SHA3_384]    = SHA3_384_DIGEST_SIZE,
	[HASH_ALGO_SHA3_512]    = SHA3_512_DIGEST_SIZE,
};
EXPORT_SYMBOL_GPL(hash_digest_size);
