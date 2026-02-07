/* SPDX-License-Identifier: GPL-2.0 */
/**
 * @file sha1_asm.S
 * @brief Highly optimized SHA-1 implementation for SPARC64 architecture utilizing VIS extensions.
 * @details This assembly file provides a high-performance implementation
 * of the SHA-1 (Secure Hash Algorithm 1) compression function, specifically
 * tuned for SPARC64 processors. It leverages the SPARC VIS (Visual
 * Instruction Set) extensions to accelerate cryptographic operations.
 * The implementation intelligently handles both aligned and unaligned input data
 * for optimal performance, ensuring efficient loading and processing of message
 * blocks.
 *
 * **Optimization Techniques:**
 * - **SPARC VIS Extensions**: Utilizes VIS instructions like `ldd` (load doubleword)
 *   and `faligndata` (Floating-point Align Data) for efficient data movement and
 *   preparation for SHA-1 rounds.
 * - **Data Alignment Handling**: Contains separate code paths to handle input data
 *   that is either naturally aligned or unaligned to doubleword boundaries,
 *   maximizing `ldd` efficiency.
 * - **Macro-driven Round Execution**: The core SHA-1 rounds are encapsulated within
 *   the `SHA1` macro (assumed to be defined externally), which performs the
 *   80 rounds of hash computation.
 * - **Register Utilization**: Floating-point registers (`%f0` through `%f22` etc.)
 *   are used to hold both the SHA-1 digest state and message words, enabling fast
 *   computation within the VIS unit.
 *
 * This implementation aims to provide maximum SHA-1 hashing performance on
 * SPARC64 systems within the Linux kernel.
 */
#include <linux/linkage.h>
#include <asm/opcodes.h>
#include <asm/visasm.h>

/**
 * @brief SPARC64 optimized SHA-1 transform function.
 * @details This function implements the core SHA-1 compression for a 64-byte
 * message block, highly optimized for the SPARC64 architecture using assembly
 * and VIS (Visual Instruction Set) extensions. It intelligently handles
 * data alignment to ensure efficient loading and processing of message blocks.
 *
 * @param %o0 (digest): Pointer to the SHA-1 state array (5x u32 words).
 * @param %o1 (data): Pointer to the 64-byte input data block.
 * @param %o2 (rounds): Number of 64-byte blocks to process.
 * Functional Utility: Performs hardware-optimized SHA-1 compression on a 64-byte block using SPARC64 assembly with VIS extensions.
 */
ENTRY(sha1_sparc64_transform)
	/* %o0 = digest, %o1 = data, %o2 = rounds */
	// Functional Utility: Enters VIS mode, saving the floating-point and VIS state.
	VISEntryHalf
	// Functional Utility: Loads the initial SHA-1 state (H0-H4) from memory (%o0) into floating-point registers %f0-%f4.
	ld	[%o0 + 0x00], %f0
	ld	[%o0 + 0x04], %f1
	ld	[%o0 + 0x08], %f2
	// Functional Utility: Checks if the input data pointer (%o1) is 8-byte aligned.
	andcc	%o1, 0x7, %g0
	ld	[%o0 + 0x0c], %f3
	// Functional Utility: Branches to `10f` (unaligned path) if data is not 8-byte aligned.
	bne,pn	%xcc, 10f
	ld	[%o0 + 0x10], %f4

1:	/* Aligned Data Path */
	// Functional Utility: Loads 8 doublewords (64 bytes) of message data from the aligned input (%o1) into floating-point registers %f8-%f22.
	ldd	[%o1 + 0x00], %f8
	ldd	[%o1 + 0x08], %f10
	ldd	[%o1 + 0x10], %f12
	ldd	[%o1 + 0x18], %f14
	ldd	[%o1 + 0x20], %f16
	ldd	[%o1 + 0x28], %f18
	ldd	[%o1 + 0x30], %f20
	ldd	[%o1 + 0x38], %f22

	// Functional Utility: Invokes the `SHA1` macro (defined externally) to perform the 80 rounds of SHA-1 compression.
	SHA1

	// Functional Utility: Decrements the block counter (%o2) and sets condition codes.
	subcc	%o2, 1, %o2
	// Functional Utility: Branches back to `1b` if more blocks remain to be processed.
	bne,pt	%xcc, 1b
	// Functional Utility: Advances the input data pointer (%o1) to the next 64-byte block.
	add	%o1, 0x40, %o1

5:	/* Epilogue: Store final state and return. */
	// Functional Utility: Stores the updated SHA-1 state from floating-point registers %f0-%f4 back to memory (%o0).
	st	%f0, [%o0 + 0x00]
	st	%f1, [%o0 + 0x04]
	st	%f2, [%o0 + 0x08]
	st	%f3, [%o0 + 0x0c]
	st	%f4, [%o0 + 0x10]
	retl // Functional Utility: Returns from the function, restoring the Link Register.
	// Functional Utility: Exits VIS mode, restoring the previous floating-point and VIS state.
	VISExitHalf

10:	/* Unaligned Data Path */
	// Functional Utility: Aligns the input data pointer (%o1) to an 8-byte boundary.
	alignaddr %o1, %g0, %o1

	// Functional Utility: Loads the first doubleword from the (now aligned) input.
	ldd	[%o1 + 0x00], %f10
1:
	// Functional Utility: Loads 8 doublewords of message data into floating-point registers %f10-%f26.
	ldd	[%o1 + 0x08], %f12
	ldd	[%o1 + 0x10], %f14
	ldd	[%o1 + 0x18], %f16
	ldd	[%o1 + 0x20], %f18
	ldd	[%o1 + 0x28], %f20
	ldd	[%o1 + 0x30], %f22
	ldd	[%o1 + 0x38], %f24
	ldd	[%o1 + 0x40], %f26

	// Functional Utility: Re-aligns the loaded doublewords into contiguous floating-point registers %f8-%f22.
	// This is necessary because `ldd` on unaligned addresses might produce fragmented data.
	faligndata %f10, %f12, %f8
	faligndata %f12, %f14, %f10
	faligndata %f14, %f16, %f12
	faligndata %f16, %f18, %f14
	faligndata %f18, %f20, %f16
	faligndata %f20, %f22, %f18
	faligndata %f22, %f24, %f20
	faligndata %f24, %f26, %f22

	// Functional Utility: Invokes the `SHA1` macro (defined externally) to perform the 80 rounds of SHA-1 compression.
	SHA1

	// Functional Utility: Decrements the block counter (%o2) and sets condition codes.
	subcc	%o2, 1, %o2
	fsrc2	%f26, %f10 // Functional Utility: Placeholder or part of a VIS optimization (Floating-point Set Register from Context 2).
	// Functional Utility: Branches back to `1b` if more blocks remain to be processed.
	bne,pt	%xcc, 1b
	add	%o1, 0x40, %o1 // Functional Utility: Advances the input data pointer (%o1) to the next 64-byte block.

	// Functional Utility: Branches unconditionally and always predicts taken to the common epilogue `5b`.
	ba,a,pt	%xcc, 5b
ENDPROC(sha1_sparc64_transform)
