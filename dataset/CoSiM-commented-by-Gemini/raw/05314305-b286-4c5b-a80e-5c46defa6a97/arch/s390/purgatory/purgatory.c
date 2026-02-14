// SPDX-License-Identifier: GPL-2.0
/*
 * Purgatory code running between two kernels for s390.
 *
 * Copyright IBM Corp. 2018
 *
 * Author(s): Philipp Rudo <prudo@linux.vnet.ibm.com>
 */

/**
 * @file purgatory.c
 * @brief Kernel integrity verification for kexec on s390.
 *
 * The "purgatory" is a small, intermediate program that runs after the first
 * kernel has shut down but before the new kernel (loaded via kexec) starts.
 * Its primary purpose on s390 is to provide a secure environment to verify the
 * integrity of the new kernel image before execution, typically by checking
 * a cryptographic hash. If verification fails, the boot process is halted.
 */

#include <linux/kexec.h>
#include <linux/string.h>
#include <crypto/sha2.h>
#include <asm/purgatory.h>

/* The expected SHA256 digest of the new kernel image, defined elsewhere. */
extern u8 purgatory_sha256_digest[SHA256_DIGEST_SIZE];

/* An array describing the memory regions of the new kernel to be hashed. */
extern struct kexec_sha_region purgatory_sha_regions[KEXEC_SEGMENT_MAX];

/**
 * verify_sha256_digest - Verifies the SHA256 digest of the loaded kernel image.
 *
 * This function iterates through the memory regions specified in
 * `purgatory_sha_regions`, which describe the segments of the kernel
 * and initramfs image loaded into memory. It computes a cumulative SHA256
 * hash of these regions and compares it against the expected hash stored in
 * `purgatory_sha256_digest`.
 *
 * The main `purgatory()` entry point (defined in assembly) calls this function.
 * If this function returns an error, the assembly code will halt the machine.
 *
 * Return: 0 if the digest matches, 1 on mismatch.
 */
int verify_sha256_digest(void)
{
	struct kexec_sha_region *ptr, *end;
	u8 digest[SHA256_DIGEST_SIZE];
	struct sha256_state sctx;

	sha256_init(&sctx);
	end = purgatory_sha_regions + ARRAY_SIZE(purgatory_sha_regions);

	/*
	 * Block Logic: Iterate over each kernel segment defined in the SHA regions
	 * and incrementally update the SHA256 hash.
	 * Invariant: `sctx` holds the cumulative hash of all segments processed so far.
	 */
	for (ptr = purgatory_sha_regions; ptr < end; ptr++)
		sha256_update(&sctx, (uint8_t *)(ptr->start), ptr->len);

	sha256_final(&sctx, digest);

	/*
	 * Block Logic: Compare the computed digest with the expected digest passed
	 * from the first kernel. A mismatch indicates a corrupted image.
	 */
	if (memcmp(digest, purgatory_sha256_digest, sizeof(digest)))
		return 1;

	return 0;
}
