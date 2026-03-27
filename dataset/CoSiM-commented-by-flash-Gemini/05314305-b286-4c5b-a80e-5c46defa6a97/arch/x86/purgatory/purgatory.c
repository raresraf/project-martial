/**
 * @file purgatory.c
 * @brief Implements the purgatory code for x86 architecture, primarily for SHA256 digest verification.
 *
 * This code runs in a transient state, typically as part of the kexec boot process,
 * to ensure the integrity of kernel memory regions by verifying their SHA256 hashes.
 * If the verification fails, the system enters an infinite loop, preventing execution
 * of potentially compromised code. This file also includes a placeholder `warn` function.
 *
 * Domain: Linux Kernel, x86 Architecture, kexec, Security, Cryptography.
 */

// SPDX-License-Identifier: GPL-2.0-only
/*
 * purgatory: Runs between two kernels
 *
 * Copyright (C) 2014 Red Hat Inc.
 *
 * Author:
 *       Vivek Goyal <vgoyal@redhat.com>
 */

#include <linux/bug.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <crypto/sha2.h>
#include <asm/purgatory.h>

#include "../boot/compressed/error.h"
#include "../boot/string.h"

/**
 * @brief Stores the expected SHA256 digest of the kernel image.
 * This global array holds the pre-calculated SHA256 hash that the purgatory code
 * will compare against the runtime-calculated hash of the kernel regions.
 * It is placed in the ".kexec-purgatory" section to ensure it's part of
 * the purgatory executable.
 */
u8 purgatory_sha256_digest[SHA256_DIGEST_SIZE] __section(".kexec-purgatory");

/**
 * @brief Defines the memory regions to be verified.
 * This global array specifies the start address and length of each segment of the
 * kexec kernel image that needs to be included in the SHA256 integrity check.
 * It's also located in the ".kexec-purgatory" section.
 */
struct kexec_sha_region purgatory_sha_regions[KEXEC_SEGMENT_MAX] __section(".kexec-purgatory");

/**
 * @brief Verifies the SHA256 digest of specified kernel regions.
 *
 * This function iterates through the `purgatory_sha_regions` array,
 * calculates the SHA256 hash of each region, and then compares the final
 * digest against the stored `purgatory_sha256_digest`. This is a critical
 * integrity check during the kexec process.
 *
 * @return 0 if the digest matches, 1 if there is a mismatch.
 */
static int verify_sha256_digest(void)
{
	struct kexec_sha_region *ptr, *end;
	u8 digest[SHA256_DIGEST_SIZE];
	struct sha256_state sctx;

	// Functional Utility: Initializes the SHA256 context for a new hash calculation.
	sha256_init(&sctx);
	// Invariant: 'end' points one past the last element of the global 'purgatory_sha_regions' array,
	// defining the boundary for iteration.
	end = purgatory_sha_regions + ARRAY_SIZE(purgatory_sha_regions);

	// Block Logic: Iterates through each kernel memory region specified for verification.
	// For each region, it updates the SHA256 hash calculation based on the region's content
	// and length, accumulating the hash for all segments.
	for (ptr = purgatory_sha_regions; ptr < end; ptr++)
		sha256_update(&sctx, (uint8_t *)(ptr->start), ptr->len);

	// Functional Utility: Finalizes the SHA256 hash calculation and stores the resulting digest.
	sha256_final(&sctx, digest);

	// Block Logic: Compares the calculated digest with the expected digest.
	// A non-zero return value indicates a mismatch, suggesting potential data corruption or tampering.
	if (memcmp(digest, purgatory_sha256_digest, sizeof(digest)))
		return 1; // Mismatch found, integrity check failed.

	return 0; // Digests match, integrity check passed.
}

/**
 * @brief The entry point for the x86 purgatory code.
 *
 * This function serves as the primary execution flow for the purgatory stage.
 * It invokes the `verify_sha256_digest` function to perform an integrity check
 * on the kernel image. If the verification fails, the system deliberately
 * enters an infinite loop to prevent the execution of potentially compromised
 * or corrupted kernel code, thereby enhancing system security.
 */
void purgatory(void)
{
	int ret;

	// Block Logic: Executes the SHA256 digest verification.
	// Pre-condition: Kernel regions and expected digest are properly set up.
	// Post-condition: 'ret' indicates success (0) or failure (1) of verification.
	ret = verify_sha256_digest();
	if (ret) {
		// Block Logic: Enters an infinite loop if digest verification fails.
		// This action prevents further execution of a potentially compromised
		// kernel, acting as a critical security measure.
		for (;;)
			/* loop forever */
			;
	}
}

/*
 * Functional Utility: Placeholder function to prevent compiler warnings.
 * This `warn` function is defined to reuse `memcpy()` and `memset()`
 * from `arch/x86/boot/compressed/string.c` which might call `warn()`
 * in error conditions. In the purgatory, a simple infinite loop is
 * the desired behavior for fatal errors, thus this function is a no-op.
 */
void warn(const char *msg) {}
