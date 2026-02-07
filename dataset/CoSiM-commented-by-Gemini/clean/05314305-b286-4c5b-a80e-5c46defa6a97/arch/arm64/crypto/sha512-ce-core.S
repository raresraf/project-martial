/* SPDX-License-Identifier: GPL-2.0 */
/**
 * @file sha512-ce-core.S
 * @brief High-performance SHA-512/384 transform function using ARMv8 Cryptography Extensions.
 * @details This assembly file implements a highly optimized SHA-512/384 compression
 * function designed for ARMv8-A architectures. It leverages the dedicated
 * Cryptography Extension (CE) instructions to offload complex 64-bit round
 * updates and message schedule calculations directly to hardware. This results
 * in significantly faster and more power-efficient hash computations compared
 * to software-only implementations, crucial for cryptographic primitives in
 * modern embedded and high-performance computing systems.
 *
 * **Hardware Acceleration Strategy:**
 * The core SHA-512/384 logic is offloaded to hardware. The key ARMv8 CE instructions
 * used are:
 * - `sha512h`: Performs the first part of a SHA-512 round update.
 * - `sha512h2`: Performs the second part of a SHA-512 round update.
 * - `sha512su0`: Computes the first stage of the SHA-512 message schedule update.
 * - `sha512su1`: Computes the second stage of the SHA-512 message schedule update.
 * These instructions enable parallel and single-cycle execution of operations
 * that would typically require many scalar instructions.
 *
 * **Pipelining and Interleaving (`dround` macro):**
 * The `dround` macro embodies a deeply pipelined and interleaved optimization
 * strategy. Within a single invocation, it performs three tasks in parallel:
 * 1. **Message Schedule Generation (Future)**: Calculates the next message words
 *    using `sha512su0` and `sha512su1`.
 * 2. **Round Computation (Current)**: Executes the core hash logic for the
 *    current two rounds using `sha512h` and `sha512h2`.
 * 3. **Constant Loading (Far Future)**: Loads round constants from memory for
 *    upcoming rounds.
 * This parallel execution model maximizes hardware utilization and hides
 * computational and memory latencies. The implementation processes one 128-byte
 * block at a time.
 */
/*
 * sha512-ce-core.S - core SHA-384/SHA-512 transform using v8 Crypto Extensions
 *
 * Copyright (C) 2018 Linaro Ltd <ard.biesheuvel@linaro.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 */

#include <linux/linkage.h>
#include <asm/assembler.h>

	/*
	 * Custom macros to emit the raw instruction encoding for the SHA-512
	 * Crypto Extension instructions. This provides a readable syntax and
	 * avoids reliance on bleeding-edge assembler support.
	 */
	.irp		b,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
	.set		.Lq\b, \b
	.set		.Lv\b\().2d, \b
	.endr

	/**
	 * @brief Macro to emit the `sha512h` instruction encoding.
	 * @details This custom macro generates the raw instruction encoding for
	 * the ARMv8 `sha512h` Cryptography Extension instruction. This instruction
	 * performs part of the SHA-512 round update, specifically handling the
	 * `A` (working variable) update based on the `E` (working variable).
	 *
	 * @param rd Destination register (NEON quad-word register).
	 * @param rn First source register (NEON quad-word register).
	 * @param rm Second source register (NEON quad-word register).
	 * Functional Utility: Provides symbolic representation and direct encoding for the `sha512h` CE instruction.
	 */
	.macro		sha512h, rd, rn, rm
	.inst		0xce608000 | .Ld | (.Ln << 5) | (.Lm << 16)
	.endm

	/**
	 * @brief Macro to emit the `sha512h2` instruction encoding.
	 * @details This custom macro generates the raw instruction encoding for
	 * the ARMv8 `sha512h2` Cryptography Extension instruction. This instruction
	 * performs the second part of a SHA-512 round update.
	 *
	 * @param rd Destination register (NEON quad-word register).
	 * @param rn First source register (NEON quad-word register).
	 * @param rm Second source register (NEON quad-word register).
	 * Functional Utility: Provides symbolic representation and direct encoding for the `sha512h2` CE instruction.
	 */
	.macro		sha512h2, rd, rn, rm
	.inst		0xce608400 | .Ld | (.Ln << 5) | (.Lm << 16)
	.endm

	/**
	 * @brief Macro to emit the `sha512su0` instruction encoding.
	 * @details This custom macro generates the raw instruction encoding for
	 * the ARMv8 `sha512su0` Cryptography Extension instruction. This instruction
	 * computes the first stage of the SHA-512 message schedule update, typically
	 * involving XOR operations on previous message words.
	 *
	 * @param rd Destination register (NEON quad-word register).
	 * @param rn First source register (NEON quad-word register).
	 * Functional Utility: Provides symbolic representation and direct encoding for the `sha512su0` CE instruction, performing part of the message schedule generation.
	 */
	.macro		sha512su0, rd, rn
	.inst		0xcec08000 | .Ld | (.Ln << 5)
	.endm

	/**
	 * @brief Macro to emit the `sha512su1` instruction encoding.
	 * @details This custom macro generates the raw instruction encoding for
	 * the ARMv8 `sha512su1` Cryptography Extension instruction. This instruction
	 * computes the second stage of the SHA-512 message schedule update,
	 * typically involving further XORs and a rotate left operation.
	 *
	 * @param rd Destination register (NEON quad-word register).
	 * @param rn First source register (NEON quad-word register).
	 * @param rm Second source register (NEON quad-word register).
	 * Functional Utility: Provides symbolic representation and direct encoding for the `sha512su1` CE instruction, completing the message schedule generation for a word.
	 */
	.macro		sha512su1, rd, rn, rm
	.inst		0xce608800 | .Ld | (.Ln << 5) | (.Lm << 16)
	.endm

	/*
	 * The SHA-512 round constants (K0-K79)
	 */
	.section	".rodata", "a"
	.align		4
.Lsha512_rcon: @ Functional Role: Data section containing the 80 SHA-512 64-bit round constants.
	.quad		0x428a2f98d728ae22, 0x7137449123ef65cd
	.quad		0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc
	.quad		0x3956c25bf348b538, 0x59f111f1b605d019
	.quad		0x923f82a4af194f9b, 0xab1c5ed5da6d8118
	.quad		0xd807aa98a3030242, 0x12835b0145706fbe
	.quad		0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2
	.quad		0x72be5d74f27b896f, 0x80deb1fe3b1696b1
	.quad		0x9bdc06a725c71235, 0xc19bf174cf692694
	.quad		0xe49b69c19ef14ad2, 0xefbe4786384f25e3
	.quad		0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65
	.quad		0x2de92c6f592b0275, 0x4a7484aa6ea6e483
	.quad		0x5cb0a9dcbd41fbd4, 0x76f988da831153b5
	.quad		0x983e5152ee66dfab, 0xa831c66d2db43210
	.quad		0xb00327c898fb213f, 0xbf597fc7beef0ee4
	.quad		0xc6e00bf33da88fc2, 0xd5a79147930aa725
	.quad		0x06ca6351e003826f, 0x142929670a0e6e70
	.quad		0x27b70a8546d22ffc, 0x2e1b21385c26c926
	.quad		0x4d2c6dfc5ac42aed, 0x53380d139d95b3df
	.quad		0x650a73548baf63de, 0x766a0abb3c77b2a8
	.quad		0x81c2c92e47edaee6, 0x92722c851482353b
	.quad		0xa2bfe8a14cf10364, 0xa81a664bbc423001
	.quad		0xc24b8b70d0f89791, 0xc76c51a30654be30
	.quad		0xd192e819d6ef5218, 0xd69906245565a910
	.quad		0xf40e35855771202a, 0x106aa07032bbd1b8
	.quad		0x19a4c116b8d2d0c8, 0x1e376c085141ab53
	.quad		0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8
	.quad		0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb
	.quad		0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3
	.quad		0x748f82ee5defb2fc, 0x78a5636f43172f60
	.quad		0x84c87814a1f0ab72, 0x8cc702081a6439ec
	.quad		0x90befffa23631e28, 0xa4506cebde82bde9
	.quad		0xbef9a3f7b2c67915, 0xc67178f2e372532b
	.quad		0xca273eceea26619c, 0xd186b8c721c0c207
	.quad		0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178
	.quad		0x06f067aa72176fba, 0x0a637dc5a2c898a6
	.quad		0x113f9804bef90dae, 0x1b710b35131c471b
	.quad		0x28db77f523047d84, 0x32caab7b40c72493
	.quad		0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c
	.quad		0x4cc5d4becb3e42b6, 0x597f299cfc657e2a
	.quad		0x5fcb6fab3ad6faec, 0x6c44198c4a475817

	/**
	 * @brief Macro to generate instructions for two SHA-512 rounds in a deeply pipelined manner.
	 * @details This `dround` macro is the core of the highly optimized SHA-512
	 * Cryptography Extensions implementation. It is designed to perform two
	 * SHA-512 rounds simultaneously, embodying a deeply pipelined and interleaved
	 * optimization strategy. Within a single invocation, it performs three
	 * critical tasks in parallel to maximize hardware utilization and hide
	 * computational and memory latencies:
	 * 1. **Message Schedule Generation (Future)**: Calculates the next message words
	 *    using `sha512su0` and `sha512su1`, preparing data for upcoming rounds.
	 * 2. **Round Computation (Current)**: Executes the core hash logic for the
	 *    current two rounds using `sha512h` and `sha512h2` CE instructions.
	 * 3. **Constant Loading (Far Future)**: Efficiently loads round constants from
	 *    memory (`x4`) for upcoming rounds.
	 *
	 * The register arguments are carefully rotated on each invocation to sequence
	 * the updates of the eight SHA-512 state variables (a,b,c,d,e,f,g,h) correctly.
	 *
	 * @param i0-i4  Register indices representing the rotating SHA-512 state variables.
	 * @param rc0, rc1  Register indices for the round constants being loaded/used.
	 * @param in0-in4 Register indices for the message words and internal state used in `sha512su0`/`sha512su1`.
	 * Functional Utility: Executes two pipelined SHA-512 rounds, concurrently updating the hash state and generating message words.
	 */
	.macro		dround, i0, i1, i2, i3, i4, rc0, rc1, in0, in1, in2, in3, in4
	.ifnb		c1 @ Functional Utility: Check if `rc1` (second round constant) is provided.
	ld1		{vc1\().2d}, [x4], #16 @ Functional Utility: Load the next 128-bit round constant into NEON register `vc1` from `x4`, and increment `x4` by 16 bytes.
	.endif
	add		v5.2d, vc0\().2d, v\in0\().2d @ Functional Utility: Add round constant (`vc0`) to message word (`v\in0`) into `v5`.
	ext		v6.16b, v\i2\().16b, v\i3\().16b, #8 @ Functional Utility: Extract and combine bytes from `v\i2` and `v\i3` into `v6`.
	ext		v5.16b, v5.16b, v5.16b, #8 @ Functional Utility: Extract and combine bytes from `v5` (self-extending).
	ext		v7.16b, v\i1\().16b, v\i2\().16b, #8 @ Functional Utility: Extract and combine bytes from `v\i1` and `v\i2` into `v7`.
	add		v\i3\().2d, v\i3\().2d, v5.2d @ Functional Utility: Update `v\i3` with `v5`.
	.ifnb		\in1 @ Functional Utility: Check if `in1` (second input for sha512su0) is provided.
	ext		v5.16b, v\in3\().16b, v\in4\().16b, #8 @ Functional Utility: Extract and combine bytes from `v\in3` and `v\in4` into `v5`.
	sha512su0	v\in0\().2d, v\in1\().2d @ Functional Utility: ARMv8 CE instruction - computes first stage of message schedule update for `v\in0`.
	.endif
	sha512h		q\i3, q6, v7.2d @ Functional Utility: ARMv8 CE instruction - performs first part of SHA-512 round update.
	.ifnb		\in1 @ Functional Utility: Check if `in1` is provided.
	sha512su1	v\in0\().2d, v\in2\().2d, v5.2d @ Functional Utility: ARMv8 CE instruction - computes second stage of message schedule update for `v\in0`.
	.endif
	add		v\i4\().2d, v\i1\().2d, v\i3\().2d @ Functional Utility: Update `v\i4` with `v\i1` and `v\i3`.
	sha512h2	q\i3, q\i1, v\i0\().2d @ Functional Utility: ARMv8 CE instruction - performs second part of SHA-512 round update.
	.endm

	/**
	 * @brief ARM64 Cryptography Extensions accelerated SHA-512/384 transform.
	 * @details This function performs the core SHA-512 or SHA-384 message compression
	 * using ARMv8-A Cryptography Extension (CE) instructions. It processes a
	 * specified number of 128-byte data blocks, updating the SHA-512 state context.
	 * The implementation is highly optimized for performance by leveraging hardware
	 * acceleration for both round computations and message schedule generation
	 * through deep pipelining enabled by the `dround` macro.
	 *
	 * @param x0 (sst): Pointer to the `sha512_state` context structure (containing a..h state).
	 * @param x1 (src): Pointer to the source data (128-byte aligned blocks).
	 * @param w2 (blocks): Number of 128-byte blocks to process.
	 *
	 * @pre `x0` points to a valid `sha512_state` structure with the current hash values.
	 * @pre `x1` points to the input data, aligned to 128 bytes.
	 * @pre `w2` is a non-negative integer representing the number of blocks.
	 * @post The `sha512_state` structure pointed to by `x0` is updated with the hash
	 *       of all processed blocks.
	 */
	.text
SYM_FUNC_START(__sha512_ce_transform)
	/* Prologue: Load initial state and round constants into NEON registers. */
	@ Functional Utility: Load the initial eight 64-bit SHA-512 state variables (a..h)
	@ from the `sha512_state` structure (pointed to by x0) into NEON vector registers `v8-v11`.
	ld1		{v8.2d-v11.2d}, [x0]

	@ Functional Utility: Load the address of the SHA-512 round constants table (`.Lsha512_rcon`) into `x3`.
	adr_l		x3, .Lsha512_rcon
	@ Functional Utility: Load the first two 128-bit SHA-512 round constants from `x3` into NEON registers `v20-v23`. `x3` is incremented by 64 bytes.
	ld1		{v20.2d-v23.2d}, [x3], #64

0:	/* Main loop: Processes one 128-byte SHA-512 block per iteration. */
	@ Invariant: At the start of each iteration, x0 points to the sha512_state, x1 points to the current
	@ 128-byte data block, and w2 holds the number of remaining blocks.
	/* 1. Load 16x64-bit message words (128 bytes) into NEON registers v12-v19. */
	@ Functional Utility: Load the first eight 64-bit message words from `x1` into NEON registers `v12-v15`. `x1` is incremented by 64 bytes.
	ld1		{v12.2d-v15.2d}, [x1], #64
	@ Functional Utility: Load the next eight 64-bit message words from `x1` into NEON registers `v16-v19`. `x1` is incremented by 64 bytes.
	ld1		{v16.2d-v19.2d}, [x1], #64
	@ Functional Utility: Decrement the block counter (w2) by 1.
	sub		w2, w2, #1

	/* 2. Byte-swap if on a little-endian system to conform to SHA-512's big-endian standard. */
CPU_LE(	rev64		v12.16b, v12.16b	) @ Functional Utility: Byte-reverse `v12` (two 64-bit words) for little-endian systems.
CPU_LE(	rev64		v13.16b, v13.16b	) @ Functional Utility: Byte-reverse `v13`.
CPU_LE(	rev64		v14.16b, v14.16b	) @ Functional Utility: Byte-reverse `v14`.
CPU_LE(	rev64		v15.16b, v15.16b	) @ Functional Utility: Byte-reverse `v15`.
CPU_LE(	rev64		v16.16b, v16.16b	) @ Functional Utility: Byte-reverse `v16`.
CPU_LE(	rev64		v17.16b, v17.16b	) @ Functional Utility: Byte-reverse `v17`.
CPU_LE(	rev64		v18.16b, v18.16b	) @ Functional Utility: Byte-reverse `v18`.
CPU_LE(	rev64		v19.16b, v19.16b	) @ Functional Utility: Byte-reverse `v19`.

	@ Functional Utility: The `x4` register is used as a pointer to the current SHA-512 round constant in `.rodata`.
	mov		x4, x3				// rc pointer

	/* 3. Copy initial state into working registers. */
	@ Functional Utility: Copy the initial SHA-512 state variables (H0-H7 from v8-v11)
	@ into working registers `v0-v3` for computation.
	mov		v0.16b, v8.16b
	mov		v1.16b, v9.16b
	mov		v2.16b, v10.16b
	mov		v3.16b, v11.16b

	/*
	 * 4. Execute all 80 rounds of compression by invoking the 'dround'
	 *    macro 40 times. The register arguments are rotated to sequence
	 *    the state updates (a,b,c,d,e,f,g,h) correctly.
	 * Functional Utility: This section orchestrates the 80 rounds of SHA-512/384 compression.
	 * Each `dround` macro call computes two rounds. The rotating pattern of register
	 * arguments (`i0` to `i4`) ensures that the SHA-512 state variables (a,b,c,d,e,f,g,h)
	 * are correctly updated in sequence, maximizing pipeline efficiency and hardware utilization.
	 */
	// v0  ab  cd  --  ef  gh  ab
	// v1  cd  --  ef  gh  ab  cd
	// v2  ef  gh  ab  cd  --  ef
	// v3  gh  ab  cd  --  ef  gh
	// v4  --  ef  gh  ab  cd  --

	dround		0, 1, 2, 3, 4, 20, 24, 12, 13, 19, 16, 17
	dround		3, 0, 4, 2, 1, 21, 25, 13, 14, 12, 17, 18
	dround		2, 3, 1, 4, 0, 22, 26, 14, 15, 13, 18, 19
	dround		4, 2, 0, 1, 3, 23, 27, 15, 16, 14, 19, 12
	dround		1, 4, 3, 0, 2, 24, 28, 16, 17, 15, 12, 13

	dround		0, 1, 2, 3, 4, 25, 29, 17, 18, 16, 13, 14
	dround		3, 0, 4, 2, 1, 26, 30, 18, 19, 17, 14, 15
	dround		2, 3, 1, 4, 0, 27, 31, 19, 12, 18, 15, 16
	dround		4, 2, 0, 1, 3, 28, 24, 12, 13, 19, 16, 17
	dround		1, 4, 3, 0, 2, 29, 25, 13, 14, 12, 17, 18

	dround		0, 1, 2, 3, 4, 30, 26, 14, 15, 13, 18, 19
	dround		3, 0, 4, 2, 1, 31, 27, 15, 16, 14, 19, 12
	dround		2, 3, 1, 4, 0, 24, 28, 16, 17, 15, 12, 13
	dround		4, 2, 0, 1, 3, 25, 29, 17, 18, 16, 13, 14
	dround		1, 4, 3, 0, 2, 26, 30, 18, 19, 17, 14, 15

	dround		0, 1, 2, 3, 4, 27, 31, 19, 12, 18, 15, 16
	dround		3, 0, 4, 2, 1, 28, 24, 12, 13, 19, 16, 17
	dround		2, 3, 1, 4, 0, 29, 25, 13, 14, 12, 17, 18
	dround		4, 2, 0, 1, 3, 30, 26, 14, 15, 13, 18, 19
	dround		1, 4, 3, 0, 2, 31, 27, 15, 16, 14, 19, 12

	dround		0, 1, 2, 3, 4, 24, 28, 16, 17, 15, 12, 13
	dround		3, 0, 4, 2, 1, 25, 29, 17, 18, 16, 13, 14
	dround		2, 3, 1, 4, 0, 26, 30, 18, 19, 17, 14, 15
	dround		4, 2, 0, 1, 3, 27, 31, 19, 12, 18, 15, 16
	dround		1, 4, 3, 0, 2, 28, 24, 12, 13, 19, 16, 17

	dround		0, 1, 2, 3, 4, 29, 25, 13, 14, 12, 17, 18
	dround		3, 0, 4, 2, 1, 30, 26, 14, 15, 13, 18, 19
	dround		2, 3, 1, 4, 0, 31, 27, 15, 16, 14, 19, 12
	dround		4, 2, 0, 1, 3, 24, 28, 16, 17, 15, 12, 13
	dround		1, 4, 3, 0, 2, 25, 29, 17, 18, 16, 13, 14

	dround		0, 1, 2, 3, 4, 26, 30, 18, 19, 17, 14, 15
	dround		3, 0, 4, 2, 1, 27, 31, 19, 12, 18, 15, 16
	dround		2, 3, 1, 4, 0, 28, 24, 12
	dround		4, 2, 0, 1, 3, 29, 25, 13
	dround		1, 4, 3, 0, 2, 30, 26, 14

	dround		0, 1, 2, 3, 4, 31, 27, 15
	dround		3, 0, 4, 2, 1, 24,   , 16
	dround		2, 3, 1, 4, 0, 25,   , 17
	dround		4, 2, 0, 1, 3, 26,   , 18
	dround		1, 4, 3, 0, 2, 27,   , 19

	/* Epilogue: Update the initial state with the results of the compression. */
	@ Functional Utility: Add the results of the 80 rounds (v0-v3) to the initial SHA-512 state (v8-v11)
	@ to produce the new cumulative hash state.
	add		v8.2d, v8.2d, v0.2d
	add		v9.2d, v9.2d, v1.2d
	add		v10.2d, v10.2d, v2.2d
	add		v11.2d, v11.2d, v3.2d

	@ Functional Utility: If there are more blocks to process (`w2 != 0`), perform a conditional yield
	@ to allow the scheduler to potentially switch tasks.
	cond_yield	3f, x4, x5
	/* handled all input blocks? */
	@ Functional Utility: Check if all input blocks have been processed (`w2` is zero).
	@ If not, branch back to the beginning of the main loop (`0b`) to process the next block.
	cbnz		w2, 0b

	/* Store the final updated state back to memory and return. */
3:	@ Functional Utility: Store the final updated SHA-512 state (v8-v11)
	@ back into the `sha512_state` structure pointed to by x0.
	st1		{v8.2d-v11.2d}, [x0]
	@ Functional Utility: Move the content of w2 into w0 (return value, typically 0 on success).
	mov		w0, w2
	@ Functional Utility: Return from the function.
	ret
SYM_FUNC_END(__sha512_ce_transform)
