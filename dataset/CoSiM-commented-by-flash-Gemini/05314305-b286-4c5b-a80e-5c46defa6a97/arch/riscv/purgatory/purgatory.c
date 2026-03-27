/**
 * @file purgatory.c
 * @brief Implements the purgatory code for RISC-V architecture, primarily for SHA256 digest verification.
 *
 * This code runs in a transient state, typically as part of the kexec boot process,
 * to ensure the integrity of kernel memory regions by verifying their SHA256 hashes.
 * If the verification fails, the system enters an infinite loop, preventing execution
 * of potentially compromised code.
 *
 * Domain: Linux Kernel, RISC-V Architecture, kexec, Security, Cryptography.
 */

// SPDX-License-Identifier: GPL-2.0-only
/*
 * purgatory: Runs between two kernels
 *
 * Copyright (C) 2022 Huawei Technologies Co, Ltd.
 *
 * Author: Li Zhengyu (lizhengyu3@huawei.com)
 *
 */

#include <linux/purgatory.h>
#include <linux/kernel.h>
#include <linux/string.h>
#include <asm/string.h>

/**
 * @brief Stores the expected SHA256 digest of the kernel image.
 * This array holds the pre-calculated SHA256 hash that the purgatory code
 * will compare against the runtime-calculated hash of the kernel regions.
 * It is placed in the ".kexec-purgatory" section to ensure it's part of
 * the purgatory executable.
 */
u8 purgatory_sha256_digest[SHA256_DIGEST_SIZE] __section(".kexec-purgatory");

/**
 * @brief Defines the memory regions to be verified.
 * This array specifies the start address and length of each segment of the
 * kexec kernel image that needs to be included in the SHA256 integrity check.
 * It's also located in the ".kexec-purgatory" section.
 */
struct kexec_sha_region purgatory_sha_regions[KEXEC_SEGMENT_MAX] __section(".kexec-purgatory");

/**
 * @brief Verifies the SHA256 digest of specified kernel regions.
 *
 * This function iterates through the `purgatory_sha_regions` array,
 * calculates the SHA256 hash of each region, and then compares the final
 * digest against the stored `purgatory_sha256_digest`.
 *
 * @return 0 if the digest matches, 1 if there is a mismatch.
 */
static int verify_sha256_digest(void)
{
	struct kexec_sha_region *ptr, *end;
	struct sha256_state ss;
	u8 digest[SHA256_DIGEST_SIZE];

	// Functional Utility: Initializes the SHA256 context.
	sha256_init(&ss);
	// Invariant: 'end' points one past the last element of 'purgatory_sha_regions'.
	end = purgatory_sha_regions + ARRAY_SIZE(purgatory_sha_regions);
	// Block Logic: Iterates through each kernel memory region specified for verification.
	// For each region, it updates the SHA256 hash calculation.
	for (ptr = purgatory_sha_regions; ptr < end; ptr++)
		sha256_update(&ss, (uint8_t *)(ptr->start), ptr->len);
	// Functional Utility: Finalizes the SHA256 hash calculation and stores the result.
	sha256_final(&ss, digest);
	// Block Logic: Compares the calculated digest with the expected digest.
	// If the digests do not match, it indicates a compromise or corruption.
	if (memcmp(digest, purgatory_sha256_digest, sizeof(digest)) != 0)
		return 1; // Mismatch found
	return 0; // Digests match
}

/* workaround for a warning with -Wmissing-prototypes */
void purgatory(void);

/**
 * @brief The entry point for the purgatory code.
 *
 * This function is the main execution flow for the purgatory stage.
 * Its sole purpose is to call `verify_sha256_digest`. If the digest
 * verification fails, it enters an infinite loop, effectively halting
 * system progression to prevent execution of untrusted code.
 */
void purgatory(void)
{
	// Block Logic: Executes the SHA256 digest verification.
	// If verification fails, the system enters an unrecoverable state (infinite loop).
	if (verify_sha256_digest())
		for (;;)
			/* loop forever */
			;
}
