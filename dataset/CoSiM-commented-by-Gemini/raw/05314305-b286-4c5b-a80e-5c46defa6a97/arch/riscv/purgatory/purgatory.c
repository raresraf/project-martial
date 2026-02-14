// SPDX-License-Identifier: GPL-2.0-only
/*
 * purgatory: Runs between two kernels
 *
 * Copyright (C) 2022 Huawei Technologies Co, Ltd.
 *
 * Author: Li Zhengyu (lizhengyu3@huawei.com)
 *
 */

/**
 * @file purgatory.c
 * @brief Kernel integrity verification for kexec on RISC-V.
 *
 * The "purgatory" is a small, intermediate program that runs after the first
 * kernel has shut down but before the new kernel (loaded via kexec) starts.
 * Its primary purpose is to provide a secure environment to verify the
 * integrity of the new kernel image before execution, typically by checking
 * a cryptographic hash. If the verification fails, it deliberately enters
 * an infinite loop to halt the boot process.
 */

#include <linux/purgatory.h>
#include <linux/kernel.h>
#include <linux/string.h>
#include <asm/string.h>

/* The expected SHA256 digest of the new kernel image, passed by the first kernel. */
u8 purgatory_sha256_digest[SHA256_DIGEST_SIZE] __section(".kexec-purgatory");

/* An array describing the memory regions (segments) of the new kernel to be hashed. */
struct kexec_sha_region purgatory_sha_regions[KEXEC_SEGMENT_MAX] __section(".kexec-purgatory");

/**
 * verify_sha256_digest - Verifies the SHA256 digest of the loaded kernel image.
 *
 * This function iterates through the memory regions specified in
 * `purgatory_sha_regions`, which describe the segments of the kernel
 * and initramfs image loaded into memory. It computes a cumulative SHA256
 * hash of these regions and compares it against the expected hash stored in
 * `purgatory_sha256_digest`.
 *
 * Return: 0 if the digest matches, 1 on mismatch.
 */
static int verify_sha256_digest(void)
{
	struct kexec_sha_region *ptr, *end;
	struct sha256_state ss;
	u8 digest[SHA256_DIGEST_SIZE];

	sha256_init(&ss);
	end = purgatory_sha_regions + ARRAY_SIZE(purgatory_sha_regions);

	/* Block Logic: Iterate over each kernel segment and update the SHA256 context.
	 * Invariant: `ss` contains the cumulative hash of all segments processed so far.
	 */
	for (ptr = purgatory_sha_regions; ptr < end; ptr++)
		sha256_update(&ss, (uint8_t *)(ptr->start), ptr->len);
	sha256_final(&ss, digest);

	/* Block Logic: Compare the computed digest with the expected digest.
	 * Pre-condition: `digest` holds the computed hash, and `purgatory_sha256_digest`
	 * holds the expected hash.
	 */
	if (memcmp(digest, purgatory_sha256_digest, sizeof(digest)) != 0)
		return 1;
	return 0;
}

/* workaround for a warning with -Wmissing-prototypes */
void purgatory(void);

/**
 * purgatory - The entry point for the kexec purgatory.
 *
 * This function is the main entry point for the purgatory code. It calls
 * the verification function to check the integrity of the next kernel. If
 * verification fails, it enters an infinite loop, effectively halting the
 * system to prevent booting a potentially corrupt or malicious kernel.
 */
void purgatory(void)
{
	/*
	 * Block Logic: If the kernel image verification fails, halt the system.
	 * This is a critical security measure to prevent execution of a
	 * compromised kernel.
	 */
	if (verify_sha256_digest())
		for (;;)
			/* loop forever */
			;
}
