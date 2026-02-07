/* SPDX-License-Identifier: GPL-2.0 */
/**
 * @file sha512_asm.S
 * @brief Highly optimized SHA-512/384 implementation for SPARC64 architecture utilizing VIS extensions.
 * @details This assembly file provides a high-performance implementation
 * of the SHA-512 (and SHA-384) compression function, specifically
 * tuned for SPARC64 processors. It leverages the SPARC VIS (Visual
 * Instruction Set) extensions to accelerate cryptographic operations.
 * The implementation intelligently handles both aligned and unaligned input data
 * for optimal performance, ensuring efficient loading and processing of 128-byte
 * message blocks.
 *
 * **Optimization Techniques:**
 * - **SPARC VIS Extensions**: Utilizes VIS instructions like `ldd` (load doubleword)
 *   and `faligndata` (Floating-point Align Data) for efficient data movement and
 *   preparation for SHA-512 rounds.
 * - **Data Alignment Handling**: Contains separate code paths to handle input data
 *   that is either naturally aligned or unaligned to doubleword boundaries,
 *   maximizing `ldd` efficiency and `faligndata` usage.
 * - **Macro-driven Round Execution**: The core SHA-512 rounds are encapsulated within
 *   the `SHA512` macro (assumed to be defined externally), which performs the
 *   80 rounds of hash computation.
 * - **Register Utilization**: Floating-point doubleword registers (`%f0` through
 *   `%f62`) are used extensively to hold both the SHA-512 digest state and
 *   16 message words, enabling fast computation within the VIS unit.
 *
 * This implementation aims to provide maximum SHA-512/384 hashing performance on
 * SPARC64 systems with VIS capabilities within the Linux kernel.
 */
#include <linux/linkage.h>
#include <asm/opcodes.h>
#include <asm/visasm.h>

/**
 * @brief SPARC64 optimized SHA-512/384 transform function.
 * @details This function implements the core SHA-512/384 compression for a 128-byte
 * message block, highly optimized for the SPARC64 architecture using assembly
 * and VIS (Visual Instruction Set) extensions. It intelligently handles
 * data alignment to ensure efficient loading and processing of message blocks.
 *
 * @param %o0 (digest): Pointer to the SHA-512 state array (8x u64 words).
 * @param %o1 (data): Pointer to the 128-byte input data block.
 * @param %o2 (rounds): Number of 128-byte blocks to process.
 * Functional Utility: Performs hardware-optimized SHA-512/384 compression on a 128-byte block using SPARC64 assembly with VIS extensions.
 */
ENTRY(sha512_sparc64_transform)
	/* %o0 = digest, %o1 = data, %o2 = rounds */
	// Functional Utility: Enters VIS mode, saving the floating-point and VIS state.
	VISEntry
	// Functional Utility: Loads the initial SHA-512 state (H0-H7, 8x 64-bit words) from memory (%o0) into floating-point doubleword registers %f0, %f2, ..., %f14.
	ldd	[%o0 + 0x00], %f0
	ldd	[%o0 + 0x08], %f2
	ldd	[%o0 + 0x10], %f4
	ldd	[%o0 + 0x18], %f6
	ldd	[%o0 + 0x20], %f8
	ldd	[%o0 + 0x28], %f10
	// Functional Utility: Checks if the input data pointer (%o1) is 8-byte aligned (doubleword aligned).
	andcc	%o1, 0x7, %g0
	ldd	[%o0 + 0x30], %f12
	// Functional Utility: Branches to `10f` (unaligned path) if data is not 8-byte aligned.
	bne,pn	%xcc, 10f
	ldd	[%o0 + 0x38], %f14

1:	/* Aligned Data Path */
	// Functional Utility: Loads 16 doublewords (128 bytes) of message data from the aligned input (%o1) into floating-point registers %f16-%f46.
	ldd	[%o1 + 0x00], %f16
	ldd	[%o1 + 0x08], %f18
	ldd	[%o1 + 0x10], %f20
	ldd	[%o1 + 0x18], %f22
	ldd	[%o1 + 0x20], %f24
	ldd	[%o1 + 0x28], %f26
	ldd	[%o1 + 0x30], %f28
	ldd	[%o1 + 0x38], %f30
	ldd	[%o1 + 0x40], %f32
	ldd	[%o1 + 0x48], %f34
	ldd	[%o1 + 0x50], %f36
	ldd	[%o1 + 0x58], %f38
	ldd	[%o1 + 0x60], %f40
	ldd	[%o1 + 0x68], %f42
	ldd	[%o1 + 0x70], %f44
	ldd	[%o1 + 0x78], %f46

	// Functional Utility: Invokes the `SHA512` macro (defined externally) to perform the 80 rounds of SHA-512/384 compression.
	SHA512

	// Functional Utility: Decrements the block counter (%o2) and sets condition codes.
	subcc	%o2, 1, %o2
	// Functional Utility: Branches back to `1b` if more blocks remain to be processed.
	bne,pt	%xcc, 1b
	// Functional Utility: Advances the input data pointer (%o1) by 128 bytes to the next block.
	add	%o1, 0x80, %o1

5:	/* Epilogue: Store final state and return. */
	// Functional Utility: Stores the updated SHA-512 state from floating-point registers %f0-%f14 back to memory (%o0).
	std	%f0, [%o0 + 0x00]
	std	%f2, [%o0 + 0x08]
	std	%f4, [%o0 + 0x10]
	std	%f6, [%o0 + 0x18]
	std	%f8, [%o0 + 0x20]
	std	%f10, [%o0 + 0x28]
	std	%f12, [%o0 + 0x30]
	std	%f14, [%o0 + 0x38]
	retl // Functional Utility: Returns from the function, restoring the Link Register.
	// Functional Utility: Exits VIS mode, restoring the previous floating-point and VIS state.
	VISExit

10:	/* Unaligned Data Path */
	// Functional Utility: Aligns the input data pointer (%o1) to an 8-byte (doubleword) boundary.
	alignaddr %o1, %g0, %o1

	// Functional Utility: Loads the first doubleword from the (now aligned) input.
	ldd	[%o1 + 0x00], %f18
1:
	// Functional Utility: Loads 16 doublewords (128 bytes) of message data into floating-point registers %f18-%f50.
	// This sequence accounts for potential unalignment, loading slightly more data than needed to ensure overlap for `faligndata`.
	ldd	[%o1 + 0x08], %f20
	ldd	[%o1 + 0x10], %f22
	ldd	[%o1 + 0x18], %f24
	ldd	[%o1 + 0x20], %f26
	ldd	[%o1 + 0x28], %f28
	ldd	[%o1 + 0x30], %f30
	ldd	[%o1 + 0x38], %f32
	ldd	[%o1 + 0x40], %f34
	ldd	[%o1 + 0x48], %f36
	ldd	[%o1 + 0x50], %f38
	ldd	[%o1 + 0x58], %f40
	ldd	[%o1 + 0x60], %f42
	ldd	[%o1 + 0x68], %f44
	ldd	[%o1 + 0x70], %f46
	ldd	[%o1 + 0x78], %f48
	ldd	[%o1 + 0x80], %f50

	// Functional Utility: Re-aligns the loaded doublewords into contiguous floating-point registers %f16-%f46.
	// `faligndata` combines parts of two registers to form a properly aligned doubleword.
	faligndata %f18, %f20, %f16
	faligndata %f20, %f22, %f18
	faligndata %f22, %f24, %f20
	faligndata %f24, %f26, %f22
	faligndata %f26, %f28, %f24
	faligndata %f28, %f30, %f26
	faligndata %f30, %f32, %f28
	faligndata %f32, %f34, %f30
	faligndata %f34, %f36, %f32
	faligndata %f36, %f38, %f34
	faligndata %f38, %f40, %f36
	faligndata %f40, %f42, %f38
	faligndata %f42, %f44, %f40
	faligndata %f44, %f46, %f42
	faligndata %f46, %f48, %f44
	faligndata %f48, %f50, %f46

	// Functional Utility: Invokes the `SHA512` macro (defined externally) to perform the 80 rounds of SHA-512/384 compression.
	SHA512

	// Functional Utility: Decrements the block counter (%o2) and sets condition codes.
	subcc	%o2, 1, %o2
	fsrc2	%f50, %f18 // Functional Utility: Placeholder or part of a VIS optimization (Floating-point Set Register from Context 2).
	// Functional Utility: Branches back to `1b` if more blocks remain to be processed.
	bne,pt	%xcc, 1b
	// Functional Utility: Advances the input data pointer (%o1) by 128 bytes to the next block.
	add	%o1, 0x80, %o1

	// Functional Utility: Branches unconditionally and always predicts taken to the common epilogue `5b`.
	ba,a,pt	%xcc, 5b
ENDPROC(sha512_sparc64_transform)
