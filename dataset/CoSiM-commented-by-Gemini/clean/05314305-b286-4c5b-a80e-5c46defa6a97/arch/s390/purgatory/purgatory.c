// SPDX-License-Identifier: GPL-2.0
/**
 * @file purgatory.c
 * @brief Integrity verification for kexec'd kernel images within the S390 purgatory environment.
 * @details This file implements the critical integrity verification logic that
 * runs in a restricted `purgatory` environment on IBM z/Architecture (s390)
 * systems. Its primary purpose is to calculate the SHA256 hash of specific
 * memory regions belonging to a kexec'd kernel image and compare it against a
 * pre-stored digest. This is a crucial security check to ensure that the kernel
 * image about to be booted has not been tampered with, thereby guaranteeing a
 * secure and reliable handover of control from the initial kernel to the kexec'd kernel.
 */
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
#include <asm/purgatory.h>

/**
 * @brief Verifies the integrity of the kexec'd kernel image using SHA256.
 * @details This function calculates the SHA256 hash of all memory regions
 * specified in the global `purgatory_sha_regions` array (defined elsewhere
 * in the S390 `purgatory` context) and compares the computed digest against
 * the pre-stored `purgatory_sha256_digest`. This is a critical security check
 * performed in the restricted `purgatory` environment to ensure that the
 * kernel image about to be booted has not been tampered with.
 *
 * @return 0 on successful verification (computed and expected digests match),
 *         or 1 if verification fails (digests do not match).
 * Functional Utility: Cryptographically confirms the authenticity and integrity of the target kernel image within the purgatory environment.
 */
int verify_sha256_digest(void)
{
	struct kexec_sha_region *ptr, *end;
	u8 digest[SHA256_DIGEST_SIZE];
	struct sha256_state sctx;

	// Functional Utility: Initializes the SHA256 hash context.
	sha256_init(&sctx);
	// Functional Utility: Calculates the end pointer for the array of SHA regions to be verified.
	end = purgatory_sha_regions + ARRAY_SIZE(purgatory_sha_regions);

	// Block Logic: Iterates through each defined memory region to update the SHA256 hash.
	for (ptr = purgatory_sha_regions; ptr < end; ptr++)
		// Functional Utility: Updates the SHA256 hash with the data from the current region.
		sha256_update(&sctx, (uint8_t *)(ptr->start), ptr->len);

	// Functional Utility: Finalizes the SHA256 hash computation and stores the result in `digest`.
	sha256_final(&sctx, digest);

	// Functional Utility: Compares the computed digest with the pre-stored expected digest.
	if (memcmp(digest, purgatory_sha256_digest, sizeof(digest)))
		return 1; // Functional Utility: Returns 1 (failure) if digests do not match.

	return 0; // Functional Utility: Returns 0 (success) if digests match.
}
