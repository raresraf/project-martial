// SPDX-License-Identifier: GPL-2.0-only
/*
 * @b9377395-0640-42a5-8f73-82ff9c50719f/security/apparmor/crypto.c
 * @brief Policy integrity verification logic for the AppArmor LSM.
 * 
 * Functional Intent: Provides cryptographic hashing utilities to generate 
 * stable fingerprints of loaded security profiles. These hashes allow 
 * userspace tools to verify that the kernel's active policy matches the 
 * intended source, facilitating auditing and detection of unauthorized 
 * policy modifications.
 * 
 * Domain: Kernel Security, AppArmor, Cryptographic Hashing.
 * 
 * Copyright 2013 Canonical Ltd.
 */

#include <crypto/sha2.h>

#include "include/apparmor.h"
#include "include/crypto.h"

/**
 * @brief Returns the byte size of the SHA256 digest used for policy hashing.
 */
unsigned int aa_hash_size(void)
{
	return SHA256_DIGEST_SIZE;
}

/**
 * aa_calc_hash - Generates a standalone SHA256 fingerprint for arbitrary data.
 * @data: Buffer containing policy fragments.
 * @len: Size of the data in bytes.
 * 
 * Logic: Allocates a kernel buffer for the digest and performs a single-pass 
 * SHA256 transformation.
 * @return Pointer to the allocated hash or an error pointer on allocation failure.
 */
char *aa_calc_hash(void *data, size_t len)
{
	char *hash;

	hash = kzalloc(SHA256_DIGEST_SIZE, GFP_KERNEL);
	if (!hash)
		return ERR_PTR(-ENOMEM);

	sha256(data, len, hash);
	return hash;
}

/**
 * aa_calc_profile_hash - Computes the aggregate fingerprint for a security profile.
 * @profile: Target AppArmor profile to associate the hash with.
 * @version: Policy binary format version.
 * @start: Start of the raw profile data.
 * @len: Length of the raw profile data.
 * 
 * Block Logic: Multi-part hashing sequence.
 * Logic: 
 * 1. Checks global 'aa_g_hash_policy' toggle to decide if hashing is required.
 * 2. Initializes a SHA256 context.
 * 3. Incorporates the policy version (little-endian) into the hash.
 * 4. Incorporates the profile body.
 * 5. Finalizes and stores the digest in the profile structure.
 * 
 * @return 0 on success, -ENOMEM on failure.
 */
int aa_calc_profile_hash(struct aa_profile *profile, u32 version, void *start,
			 size_t len)
{
	struct sha256_ctx sctx;
	__le32 le32_version = cpu_to_le32(version);

	if (!aa_g_hash_policy)
		return 0;

	profile->hash = kzalloc(SHA256_DIGEST_SIZE, GFP_KERNEL);
	if (!profile->hash)
		return -ENOMEM;

	sha256_init(&sctx);
	sha256_update(&sctx, (u8 *)&le32_version, 4);
	sha256_update(&sctx, (u8 *)start, len);
	sha256_final(&sctx, profile->hash);
	return 0;
}

/**
 * @brief Late initialization hook to announce hashing status.
 */
static int __init init_profile_hash(void)
{
	if (apparmor_initialized)
		aa_info_message("AppArmor sha256 policy hashing enabled");
	return 0;
}
late_initcall(init_profile_hash);
