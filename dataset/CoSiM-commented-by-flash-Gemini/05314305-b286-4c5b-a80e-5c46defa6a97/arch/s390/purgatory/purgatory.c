/**
 * @file purgatory.c
 * @brief Implements SHA256 digest verification for s390 architecture purgatory.
 *
 * This code operates within the s390 kexec purgatory environment, serving to
 * validate the integrity of kernel memory regions through SHA256 hashing.
 * It's a critical security component, ensuring that the subsequent kernel
 * execution proceeds only with verified code.
 *
 * Domain: Linux Kernel, s390 Architecture, kexec, Security, Cryptography.
 */

// SPDX-License-Identifier: GPL-2.0
/*
 * Purgatory code running between two kernels.
 *
 * Copyright IBM Corp. 2018
 *
 * Author(s): Philipp Rudo <prudo@linux.vnet.ibm.com>
 */

#include <linux/kexec.h>
#include <linux/string.h>
#include <crypto/sha2.h>
#include <asm/purgatory.h> // Provides definitions for purgatory_sha_regions and purgatory_sha256_digest

/**
 * @brief Verifies the SHA256 digest of specified kernel regions.
 *
 * This function is responsible for ensuring the integrity of various
 * kernel memory segments during the kexec boot process for the s390 architecture.
 * It computes the SHA256 hash of these segments and compares it against
 * a pre-calculated digest.
 *
 * Implicit Global Variables Used:
 *   - `purgatory_sha_regions`: An array of `kexec_sha_region` structures,
 *     each describing a memory segment (start address and length) to be hashed.
 *   - `purgatory_sha256_digest`: A byte array containing the expected SHA256
 *     digest for all `purgatory_sha_regions` combined.
 *
 * @return 0 if the computed digest matches the expected digest, 1 otherwise.
 */
int verify_sha256_digest(void)
{
	struct kexec_sha_region *ptr, *end;
	u8 digest[SHA256_DIGEST_SIZE];
	struct sha256_state sctx;

	// Functional Utility: Initializes the SHA256 context.
	sha256_init(&sctx);
	// Invariant: 'end' points one past the last element of the global 'purgatory_sha_regions' array.
	end = purgatory_sha_regions + ARRAY_SIZE(purgatory_sha_regions);

	// Block Logic: Iterates through each kernel memory region specified for verification.
	// For each region, it updates the SHA256 hash calculation based on the region's content.
	for (ptr = purgatory_sha_regions; ptr < end; ptr++)
		sha256_update(&sctx, (uint8_t *)(ptr->start), ptr->len);

	// Functional Utility: Finalizes the SHA256 hash calculation and stores the result.
	sha256_final(&sctx, digest);

	// Block Logic: Compares the calculated digest with the expected digest.
	// If the digests do not match, it signifies a potential compromise or corruption.
	if (memcmp(digest, purgatory_sha256_digest, sizeof(digest)))
		return 1; // Mismatch found

	return 0; // Digests match
}
