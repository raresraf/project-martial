// SPDX-License-Identifier: GPL-2.0-only
/**
 * @file purgatory.c
 * @brief Kernel integrity verification for kexec on x86.
 *
 * The "purgatory" is a small, intermediate program that runs after the first
 * kernel has shut down but before the new kernel (loaded via kexec) starts.
 * Its primary purpose is to provide a secure environment to verify the
 * integrity of the new kernel image before execution, typically by checking
 * a cryptographic hash. If the verification fails, it deliberately enters
 * an infinite loop to halt the boot process, preventing the execution of a
 * potentially corrupt or malicious kernel.
 */
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
	u8 digest[SHA256_DIGEST_SIZE];
	struct sha256_state sctx;

	sha256_init(&sctx);
	end = purgatory_sha_regions + ARRAY_SIZE(purgatory_sha_regions);

	/*
	 * Block Logic: Iterate over each kernel segment and update the SHA256 context.
	 * Invariant: `sctx` contains the cumulative hash of all segments processed so far.
	 */
	for (ptr = purgatory_sha_regions; ptr < end; ptr++)
		sha256_update(&sctx, (uint8_t *)(ptr->start), ptr->len);

	sha256_final(&sctx, digest);

	/*
	 * Block Logic: Compare the computed digest with the expected digest. A mismatch
	 * indicates that the loaded kernel image is corrupt or has been tampered with.
	 */
	if (memcmp(digest, purgatory_sha256_digest, sizeof(digest)))
		return 1;

	return 0;
}

/**
 * purgatory - The entry point for the kexec purgatory on x86.
 *
 * This function is the main entry point called from assembly code. It orchestrates
 * the verification of the next kernel. If the integrity check fails, it enters
 * an infinite loop, effectively halting the system to prevent booting.
 */
void purgatory(void)
{
	int ret;

	ret = verify_sha256_digest();
	if (ret) {
		/*
		 * Block Logic: If the kernel image verification fails, halt the system.
		 * This is a critical security measure.
		 */
		/* loop forever */
		for (;;)
			;
	}
}

/**
 * warn - A stub function to satisfy linker dependencies.
 * @msg: The warning message string (unused).
 *
 * This function is defined as an empty stub to allow the purgatory code to
 * link against and reuse standard string manipulation functions (like memcpy
 * and memset) from the kernel's boot compression library, which may have
 * dependencies on a `warn` function. It does not perform any action.
 */
/*
 * Defined in order to reuse memcpy() and memset() from
 * arch/x86/boot/compressed/string.c
 */
void warn(const char *msg) {}
