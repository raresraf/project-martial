/* SPDX-License-Identifier: GPL-2.0-only */
/**
 * @file sha1-ce-core.S
 * @brief High-performance SHA-1 compression function using ARMv8 Cryptography Extensions.
 * @details This assembly file implements a highly optimized SHA-1 compression
 * function designed for ARMv8-A architectures. It leverages the dedicated
 * Cryptography Extension (CE) instructions to offload complex SHA-1 operations
 * directly to hardware, resulting in significantly faster and more power-efficient
 * hash computations compared to software-only implementations.
 *
 * **Hardware Acceleration Strategy:**
 * The core SHA-1 logic, including round updates and message schedule calculations,
 * is executed directly by ARMv8 CE instructions such as:
 * - `sha1c` (choose function)
 * - `sha1p` (parity function)
 * - `sha1m` (majority function)
 * - `sha1h` (update A with E, rotate)
 * - `sha1su0` (message schedule update 0)
 * - `sha1su1` (message schedule update 1)
 * These instructions provide parallel and single-cycle execution of operations
 * that would typically require many scalar instructions, greatly boosting throughput.
 * The implementation processes one 64-byte block at a time.
 *
 * **Key Optimization Techniques:**
 * - **Direct Hardware Instruction Use**: Utilizes dedicated SHA-1 instructions from ARMv8 CE.
 * - **Pipelined Message Schedule**: `sha1su0` and `sha1su1` instructions efficiently
 *   compute the next message words (`W[i]`) in parallel with the current round computations.
 * - **Register Allocation**: Extensive use of NEON registers (`v0-v3`, `v4-v5`, `v6-v7`, `v8-v11`, `v12-v15`) for round constants,
 *   intermediate digest values, and message words to minimize memory accesses.
 * - **Macro-driven structure**: Macros (`add_only`, `add_update`) encapsulate and
 *   orchestrate the complex sequence of CE instructions to manage the SHA-1 rounds
 *   and message schedule generation efficiently.
 */
/*
 * sha1-ce-core.S - SHA-1 secure hash using ARMv8 Crypto Extensions
 *
 * Copyright (C) 2014 Linaro Ltd <ard.biesheuvel@linaro.org>
 */

#include <linux/linkage.h>
#include <asm/assembler.h>

/*
 * This file provides __sha1_ce_transform, a high-performance implementation
 * of the SHA-1 compression function for ARMv8-A architectures, accelerated
 * using the dedicated Cryptography Extension (CE) instructions.
 *
 * Hardware Acceleration Strategy:
 * The core SHA-1 logic is executed directly in hardware. The instructions
 * `sha1c`, `sha1p`, `sha1m`, `sha1h`, `sha1su0`, and `sha1su1` perform the
 * complex round updates and message schedule calculations, providing
 * significant performance and power efficiency benefits.
 */

	.text
	.arch		armv8-a+crypto

/*
 * Register Aliases for Readability
 * Functional Role: These aliases provide meaningful names to the NEON and general-purpose
 * registers, clarifying their use in the SHA-1 computation process.
 */
	k0		.req	v0 @ Functional Role: NEON vector register holding SHA1 round constant K_00_19 (for rounds 0-19).
	k1		.req	v1 @ Functional Role: NEON vector register holding SHA1 round constant K_20_39 (for rounds 20-39).
	k2		.req	v2 @ Functional Role: NEON vector register holding SHA1 round constant K_40_59 (for rounds 40-59).
	k3		.req	v3 @ Functional Role: NEON vector register holding SHA1 round constant K_60_79 (for rounds 60-79).

	t0		.req	v4 @ Functional Role: NEON temporary register for intermediate round calculation (even).
	t1		.req	v5 @ Functional Role: NEON temporary register for intermediate round calculation (odd).

	dga		.req	q6 @ Functional Role: NEON quad-word register holding SHA1 state variables A, B, C, D (32-bit words).
	dgav		.req	v6 @ Functional Role: NEON vector register alias for dga.
	dgb		.req	s7 @ Functional Role: NEON single-precision floating-point register alias for SHA1 state variable E (32-bit word).
	dgbv		.req	v7 @ Functional Role: NEON vector register alias for dgb.

	dg0q		.req	q12 @ Functional Role: NEON quad-word register for current digest values (H0-H3 of SHA1 state).
	dg0s		.req	s12 @ Functional Role: NEON single-precision register alias for one component of dg0q.
	dg0v		.req	v12 @ Functional Role: NEON vector register alias for dg0q.
	dg1s		.req	s13 @ Functional Role: NEON single-precision register alias for one component of dg1q.
	dg1v		.req	v13 @ Functional Role: NEON vector register alias for dg1q (dgb).
	dg2s		.req	s14 @ Functional Role: NEON single-precision register alias for one component of dg2q.

	/**
	 * @brief Performs a single SHA-1 round update using Cryptography Extensions.
	 * @details This macro orchestrates a single round of the SHA-1 algorithm
	 * by combining the `sha1h` instruction (to update the A state from E, and rotate A by 5 bits)
	 * with one of the `sha1c` (choose), `sha1p` (parity), or `sha1m` (majority) instructions,
	 * which implement the logical function for the round. This directly utilizes
	 * hardware acceleration for the core SHA-1 computations. The `ev` parameter
	 * is used to alternate between temporary registers (`t0`, `t1`) to facilitate
	 * instruction scheduling and pipelining.
	 *
	 * @param op The specific SHA-1 logical function instruction (`c`, `p`, or `m`).
	 * @param ev Even/odd temporary register selector (`ev` or `od`).
	 * @param rc Round constant register (`k0`-`k3`).
	 * @param s0 Message schedule register (e.g., `8` for `v8`).
	 * @param dg1 Previous digest state register (`dg1s`, `dg2s`, or `dgb`).
	 * Functional Utility: Executes a single SHA1 round, including updating A from E
	 * and applying the round's logical function using ARMv8 Crypto Extensions.
	 */
	.macro		add_only, op, ev, rc, s0, dg1
	.ifc		\ev, ev @ Functional Utility: If 'ev' (even) selector is used for temporary registers.
	add		t1.4s, v\s0\().4s, c\().4s @ Functional Utility: Add current message word (v\s0) and round constant (c) into temporary register t1.
	sha1h		dg2s, dg0s @ Functional Utility: ARMv8 CE instruction - updates A from E (dg0s).
	.ifnb		\dg1 @ Conditional assembly: if dg1 is not blank.
	sha1\op		dg0q, \dg1, t0.4s @ Functional Utility: ARMv8 CE instruction - computes round logical function.
	.else
	sha1\op		dg0q, dg1s, t0.4s @ Functional Utility: ARMv8 CE instruction - computes round logical function.
	.endif
	.else @ Functional Utility: If 'od' (odd) selector is used for temporary registers.
	.ifnb		\s0
	add		t0.4s, v\s0\().4s, c\().4s @ Functional Utility: Add current message word (v\s0) and round constant (c) into temporary register t0.
	.endif
	sha1h		dg1s, dg0s @ Functional Utility: ARMv8 CE instruction - updates A from E (dg0s).
	sha1\op		dg0q, dg2s, t1.4s @ Functional Utility: ARMv8 CE instruction - computes round logical function.
	.endif
	.endm

	/**
	 * @brief Performs a SHA-1 round update and concurrently calculates next message words.
	 * @details This macro is the core engine of the SHA-1 compression loop
	 * when using ARMv8 Cryptography Extensions. It executes one round by
	 * calling `add_only` to perform the hash update, and simultaneously
	 * calculates the next four words of the message schedule using the
	 * `sha1su0` and `sha1su1` hardware instructions. This interleaving
	 * of round computation and message schedule generation is crucial for
	 * achieving maximum efficiency and pipeline utilization in the hardware
	 * accelerated SHA-1 implementation.
	 *
	 * @param op The specific SHA-1 logical function instruction (`c`, `p`, or `m`).
	 * @param ev Even/odd temporary register selector (`ev` or `od`).
	 * @param rc Round constant register (`k0`-`k3`).
	 * @param s0, s1, s2, s3 Message schedule registers for `sha1su0`/`sha1su1`.
	 * @param dg1 Previous digest state.
	 * Functional Utility: Executes one SHA1 round via hardware acceleration while
	 * concurrently generating the next set of message words, critical for pipelined
	 * execution.
	 */
	.macro		add_update, op, ev, rc, s0, s1, s2, s3, dg1
	sha1su0		v\s0\().4s, v\s1\().4s, v\s2\().4s @ Functional Utility: ARMv8 CE instruction - computes first stage of message schedule update (W[i] = W[i-3] ^ W[i-8] ^ W[i-14]).
	add_only	\op, \ev, c, \s1, \dg1 @ Functional Utility: Perform the core SHA1 round computation.
	sha1su1		v\s0\().4s, v\s3\().4s @ Functional Utility: ARMv8 CE instruction - computes second stage of message schedule update (W[i] ^= W[i-16], then ROL(W[i], 1)).
	.endm

	/**
	 * @brief Loads a 32-bit constant value and duplicates it across a NEON vector register.
	 * @details This macro simplifies the process of loading a specific 32-bit
	 * constant value into a scalar register (`tmp`) and then replicating it
	 * across all lanes of a NEON quad-word register (`k`). This is used
	 * for initializing the SHA-1 round constants (`k0`-`k3`) into NEON
	 * registers for vectorized operations.
	 *
	 * @param k NEON vector register (e.g., `k0`, `k1`).
	 * @param val The 32-bit constant value to load.
	 * @param tmp Temporary general-purpose register (`w6`).
	 * Functional Utility: Initializes NEON vector registers with SHA-1 round constants efficiently.
	 */
	.macro		loadrc, k, val, tmp
	movz			mp, :abs_g0_nc:\val @ Functional Utility: Move immediate (zero extended) to tmp, lower 16 bits of val.
	movk			mp, :abs_g1:\val @ Functional Utility: Move immediate (keep) to tmp, upper 16 bits of val.
	dup		\k, 	mp @ Functional Utility: Duplicate the 32-bit value from tmp across all four lanes of NEON register k.
	.endm

	/**
	 * @brief ARM64 Cryptography Extensions accelerated SHA-1 transform.
	 * @details This function performs the core SHA-1 message compression using
	 * ARMv8-A Cryptography Extension (CE) instructions. It processes a specified
	 * number of 64-byte data blocks, updating the SHA-1 hash context. The
	 * implementation is highly optimized for performance by leveraging hardware
	 * acceleration for both round computations and message schedule generation.
	 *
	 * @param x0 (sst): Pointer to the `sha1_ce_state` context structure (containing H0-H4).
	 * @param x1 (src): Pointer to the source data (64-byte aligned blocks).
	 * @param w2 (blocks): Number of 64-byte blocks to process.
	 *
	 * @pre `x0` points to a valid `sha1_ce_state` structure with the current hash values.
	 * @pre `x1` points to the input data, aligned to 64 bytes.
	 * @pre `w2` is a non-negative integer representing the number of blocks.
	 * @post The `sha1_ce_state` structure pointed to by `x0` is updated with the hash
	 *       of all processed blocks.
	 */
SYM_FUNC_START(__sha1_ce_transform)
	/* Prologue: Load round constants and initial hash state into registers. */
	@ Functional Utility: Load SHA-1 round constants (K_00_19, K_20_39, K_40_59, K_60_79)
	@ into NEON vector registers k0-k3. Each constant is duplicated across all
	@ 32-bit lanes of its respective vector register for efficient vectorized addition.
	loadrc		k0.4s, 0x5a827999, w6
	loadrc		k1.4s, 0x6ed9eba1, w6
	loadrc		k2.4s, 0x8f1bbcdc, w6
	loadrc		k3.4s, 0xca62c1d6, w6

	@ Functional Utility: Load the first four 32-bit SHA-1 chaining variables (H0-H3)
	@ from the `sha1_ce_state` structure (pointed to by x0) into NEON vector register `dgav`.
	ld1		{dgav.4s}, [x0]
	@ Functional Utility: Load the fifth 32-bit SHA-1 chaining variable (H4)
	@ from `sha1_ce_state` (x0 + 16 bytes offset) into NEON scalar register `dgb`.
	ldr		dgb, [x0, #16]

	@ Functional Utility: Load the offset of the finalize flag within the `sha1_ce_state` structure.
	ldr_l		w4, sha1_ce_offsetof_finalize, x4
	@ Functional Utility: Load the actual value of the finalize flag into w4.
	ldr		w4, [x0, x4]

	/* Main loop: Processes one 64-byte block per iteration. */
	@ Invariant: At the start of each iteration, x0 points to the sha1_ce_state, x1 points to the current
	@ 64-byte data block, and w2 holds the number of remaining blocks.
0:	@ 1. Load 16 words (64 bytes) of message data into NEON registers v8-v11.
	@ Functional Utility: Load 4 quad-words (16 words, 64 bytes) of input data
	@ from the source address (x1) into NEON registers v8-v11. The source pointer
	@ x1 is incremented by 64 bytes after the load.
	ld1		{v8.4s-v11.4s}, [x1], #64
	@ Functional Utility: Decrement the block counter (w2) by 1.
	subs		w2, w2, #1

CPU_LE(	rev32		v8.16b, v8.16b		) @ Functional Utility: Byte-reverse v8 (four 32-bit words) for little-endian systems to conform to SHA-1's big-endian requirement.
CPU_LE(	rev32		v9.16b, v9.16b		) @ Functional Utility: Byte-reverse v9.
CPU_LE(	rev32		v10.16b, v10.16b	) @ Functional Utility: Byte-reverse v10.
CPU_LE(	rev32		v11.16b, v11.16b	) @ Functional Utility: Byte-reverse v11.

1:	@ 2. Execute the 80 rounds of SHA-1 compression using hardware instructions,
	@    interleaving message schedule updates with round calculations.
	@ Functional Utility: Add the first message word (v8) and round constant (k0) to t0.
	add		t0.4s, v8.4s, k0.4s
	@ Functional Utility: Move the current SHA-1 state (H0-H3 from dg0v) into dg0v for processing.
	mov		dg0v.16b, dgav.16b

	@ Rounds 0-19 (Choose function: F(B,C,D) = (B & C) | (~B & D))
	@ Functional Utility: These `add_update` macros execute 5 rounds each, for a total of 20 rounds.
	@ They use the `sha1c` (choose) instruction for the logical function and
	@ `sha1su0`/`sha1su1` to concurrently calculate the next message words using hardware acceleration.
	add_update	c, ev, k0,  8,  9, 10, 11, dgb
	add_update	c, od, k0,  9, 10, 11,  8
	add_update	c, ev, k0, 10, 11,  8,  9
	add_update	c, od, k0, 11,  8,  9, 10
	add_update	c, ev, k1,  8,  9, 10, 11

	@ Rounds 20-39 (Parity function: F(B,C,D) = B ^ C ^ D)
	@ Functional Utility: These `add_update` macros execute 5 rounds each, for a total of 20 rounds.
	@ They use the `sha1p` (parity) instruction for the logical function and
	@ `sha1su0`/`sha1su1` to concurrently calculate the next message words using hardware acceleration.
	add_update	p, od, k1,  9, 10, 11,  8
	add_update	p, ev, k1, 10, 11,  8,  9
	add_update	p, od, k1, 11,  8,  9, 10
	add_update	p, ev, k1,  8,  9, 10, 11
	add_update	p, od, k2,  9, 10, 11,  8

	@ Rounds 40-59 (Majority function: F(B,C,D) = (B & C) | (B & D) | (C & D))
	@ Functional Utility: These `add_update` macros execute 5 rounds each, for a total of 20 rounds.
	@ They use the `sha1m` (majority) instruction for the logical function and
	@ `sha1su0`/`sha1su1` to concurrently calculate the next message words using hardware acceleration.
	add_update	m, ev, k2, 10, 11,  8,  9
	add_update	m, od, k2, 11,  8,  9, 10
	add_update	m, ev, k2,  8,  9, 10, 11
	add_update	m, od, k2,  9, 10, 11,  8
	add_update	m, ev, k3, 10, 11,  8,  9

	@ Rounds 60-79 (Parity function: F(B,C,D) = B ^ C ^ D)
	@ Functional Utility: These `add_update` and `add_only` macros execute 5 rounds each, for a total of 20 rounds.
	@ They use the `sha1p` (parity) instruction for the logical function. For the last few rounds,
	@ `add_only` is used as message word generation is no longer necessary, simplifying the macro call.
	add_update	p, od, k3, 11,  8,  9, 10
	add_only	p, ev, k3,  9
	add_only	p, od, k3, 10
	add_only	p, ev, k3, 11
	add_only	p, od

	@ 3. Update the digest state with the result of the compression.
	@ Functional Utility: Add current `dgbv` (H4) to `dg1v` (intermediate H4 result).
	add		dgbv.2s, dgbv.2s, dg1v.2s
	@ Functional Utility: Add current `dgav` (H0-H3) to `dg0v` (intermediate H0-H3 result).
	add		dgav.4s, dgav.4s, dg0v.4s

	@ Functional Utility: Check if there are no more full blocks to process (w2 == 0).
	@ If true, branch to label 2f for final block handling.
	cbz		w2, 2f			@ If no more full blocks, check for final block.
	@ Functional Utility: Conditional yield instruction, likely for task preemption or cooperative multitasking.
	cond_yield	3f, x5, x6
	@ Functional Utility: Branch back to the beginning of the loop (`0:`) to process the next 64-byte block.
	b		0b			@ Loop for next block.

	/*
	 * Final block handling: If the 'finalize' flag is set, this section
	 * constructs the padding and length encoding required by the SHA-1
	 * standard and processes it as the last block.
	 */
2:	@ Functional Utility: Check if the finalize flag (w4) is zero. If true, branch to 3f (epilogue), indicating no finalization is needed.
	cbz		w4, 3f
	@ Functional Utility: Load the offset of the count field within the `sha1_ce_state` structure into w4.
	ldr_l		w4, sha1_ce_offsetof_count, x4
	@ Functional Utility: Load the 64-bit message length count from the `sha1_ce_state` structure (at offset x4 from x0) into x4.
	ldr		x4, [x0, x4]
	@ Functional Utility: Clear NEON registers v9.2d (double-word) with zeros, used for padding.
	movi		v9.2d, #0
	@ Functional Utility: Load 0x80000000 into x8, representing the padding bit (1 followed by zeros).
	mov		x8, #0x80000000
	@ Functional Utility: Clear NEON registers v10.2d (double-word) with zeros, used for padding.
	movi		v10.2d, #0
	@ Functional Utility: Rotate the message length (x4) by 29 bits to the right. This prepares the 64-bit length for big-endian encoding in the SHA-1 padding block.
	ror		x7, x4, #29		// ror(lsl(x4, 3), 32)
	@ Functional Utility: Move the padding bit (x8) into the lower double-word of NEON register v8.
	fmov		d8, x8
	@ Functional Utility: Clear x4.
	mov		x4, #0
	@ Functional Utility: Set the lower double-word of NEON register v11 to zero.
	mov		v11.d[0], xzr
	@ Functional Utility: Set the upper double-word of NEON register v11 with the rotated message length (x7).
	mov		v11.d[1], x7
	@ Functional Utility: Branch back to label 1b to re-run the compression on the newly constructed padded block.
	b		1b			@ Re-run the compression on the padded block.

	/* Epilogue: Store the final state and return. */
3:	@ Functional Utility: Store the final updated SHA-1 chaining variables (H0-H3 from `dgav`)
	@ back into the `sha1_ce_state` structure pointed to by x0.
	st1		{dgav.4s}, [x0]
	@ Functional Utility: Store the final updated SHA-1 chaining variable (H4 from `dgb`)
	@ back into the `sha1_ce_state` structure (x0 + 16 bytes offset).
	str		dgb, [x0, #16]
	@ Functional Utility: Move the content of w2 into w0 (return value, typically 0 on success).
	mov		w0, w2
	@ Functional Utility: Return from the function.
	ret
SYM_FUNC_END(__sha1_ce_transform)
