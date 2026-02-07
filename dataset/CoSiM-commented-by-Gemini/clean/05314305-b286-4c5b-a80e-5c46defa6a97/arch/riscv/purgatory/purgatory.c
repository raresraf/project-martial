// SPDX-License-Identifier: GPL-2.0-only
/**
 * @file purgatory.c
 * @brief Integrity verification for kexec'd kernel images within the RISC-V purgatory environment.
 * @details This file implements the critical integrity verification logic that
 * runs in a restricted `purgatory` environment on RISC-V systems. Its primary
 * purpose is to calculate the SHA256 hash of specific memory regions belonging
 * to a kexec'd kernel image and compare it against a pre-stored digest.
 * If the verification fails, the system halts to prevent the booting of a
 * potentially corrupted or malicious kernel. This ensures a secure and reliable
 * handover of control from the initial kernel to the kexec'd kernel.
 */
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
 * @brief Stores the expected SHA256 digest of the kexec'd kernel image.
 * @details This array holds the pre-calculated SHA256 hash that the `purgatory`
 * code will compare against the computed hash of the target kernel image. It is
 * located in the `.kexec-purgatory` section, ensuring it's available in the
 * restricted purgatory environment.
 * Functional Role: Contains the cryptographic fingerprint for kernel image verification.
 */
u8 purgatory_sha256_digest[SHA256_DIGEST_SIZE] __section(".kexec-purgatory");

/**
 * @brief Describes the memory regions of the kexec'd kernel image to be verified.
 * @details This array of `struct kexec_sha_region` defines the start address
 * and length of various segments of the kexec'd kernel image. The `purgatory`
 * code iterates through these regions to compute their collective SHA256 hash.
 * It is located in the `.kexec-purgatory` section.
 * Functional Role: Specifies the memory areas whose integrity must be cryptographically verified.
 */
struct kexec_sha_region purgatory_sha_regions[KEXEC_SEGMENT_MAX] __section(".kexec-purgatory");

/**
 * @brief Verifies the integrity of the kexec'd kernel image using SHA256.
 * @details This function calculates the SHA256 hash of all memory regions
 * specified in `purgatory_sha_regions` and compares the computed digest
 * against the pre-stored `purgatory_sha256_digest`. This is a critical
 * security check to ensure that the kernel image about to be booted has
 * not been tampered with.
 *
 * @return 0 on successful verification (digests match), or 1 if verification fails.
 * Functional Utility: Cryptographically confirms the authenticity and integrity of the target kernel image.
 */
static int verify_sha256_digest(void)
{
	struct kexec_sha_region *ptr, *end;
	struct sha256_state ss;
	u8 digest[SHA256_DIGEST_SIZE];

	// Functional Utility: Initializes the SHA256 hash context.
	sha256_init(&ss);
	// Functional Utility: Calculates the end pointer for the array of SHA regions.
	end = purgatory_sha_regions + ARRAY_SIZE(purgatory_sha_regions);
	// Block Logic: Iterates through each defined memory region.
	for (ptr = purgatory_sha_regions; ptr < end; ptr++)
		// Functional Utility: Updates the SHA256 hash with the data from the current region.
		sha256_update(&ss, (uint8_t *)(ptr->start), ptr->len);
	// Functional Utility: Finalizes the SHA256 hash computation and stores the result in `digest`.
	sha256_final(&ss, digest);
	// Functional Utility: Compares the computed digest with the pre-stored expected digest.
	if (memcmp(digest, purgatory_sha256_digest, sizeof(digest)) != 0)
		return 1; // Functional Utility: Returns 1 (failure) if digests do not match.
	return 0; // Functional Utility: Returns 0 (success) if digests match.
}

/* workaround for a warning with -Wmissing-prototypes */
void purgatory(void);

/**
 * @brief Main entry point for the RISC-V purgatory code.
 * @details This function is the entry point for the purgatory environment
 * on RISC-V. Its sole and critical purpose is to invoke the integrity
 * verification process. If `verify_sha256_digest()` returns a failure,
 * this function enters an infinite loop, effectively halting the system
 * to prevent the booting of an unverified or corrupted kernel image.
 *
 * Functional Utility: Orchestrates the critical integrity check of the kexec'd kernel and halts if verification fails.
 */
void purgatory(void)
{
	// Block Logic: Performs integrity verification.
	if (verify_sha256_digest())
		// Functional Utility: Enters an infinite loop if verification fails, halting system execution.
		for (;;)
			/* loop forever */
			;
}
