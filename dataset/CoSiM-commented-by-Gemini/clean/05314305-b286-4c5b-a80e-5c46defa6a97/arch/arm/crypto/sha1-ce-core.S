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
 * is executed directly by ARMv8 CE instructions such as `sha1c` (choose), `sha1p` (parity),
 * `sha1m` (majority), `sha1h` (update A), `sha1su0` (message schedule update 0),
 * and `sha1su1` (message schedule update 1). This allows for parallel and
 * single-cycle execution of operations that would typically require many scalar
 * instructions. The implementation processes one 64-byte block at a time.
 *
 * **Key Optimization Techniques:**
 * - **Direct Hardware Instruction Use**: Utilizes dedicated SHA-1 instructions from ARMv8 CE.
 * - **Pipelined Message Schedule**: The `sha1su0` and `sha1su1` instructions efficiently
 *   compute the next message words (`W[i]`) in parallel with the current round computations.
 * - **Register Allocation**: Efficient use of NEON registers (`q0-q15`) for round constants,
 *   intermediate digest values, and message words.
 * - **Macro-driven structure**: Macros (`add_only`, `add_update`) encapsulate and
 *   orchestrate the complex sequence of CE instructions to manage the SHA-1 rounds
 *   and message schedule generation.
 */
/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * sha1-ce-core.S - SHA-1 secure hash using ARMv8 Crypto Extensions
 *
 * Copyright (C) 2015 Linaro Ltd.
 * Author: Ard Biesheuvel <ard.biesheuvel@linaro.org>
 */

#include <linux/linkage.h>
#include <asm/assembler.h>

@
@ This file provides sha1_ce_transform, a high-performance implementation of the
@ SHA-1 compression function for ARMv8-A architectures, accelerated using the
@ dedicated Cryptography Extension (CE) instructions.
@
@ Hardware Acceleration Strategy:
@ The core SHA-1 logic is executed directly in hardware. The CE instructions
@ (sha1c, sha1p, sha1m, sha1h, sha1su0, sha1su1) perform the complex round
@ updates and message schedule calculations, making this implementation
@ significantly faster and more power-efficient than software-only versions.
@ The implementation processes one 64-byte block at a time.
@

	.text
	.arch		armv8-a
	.fpu		crypto-neon-fp-armv8

@ Register aliases for clarity
@ Functional Role: These aliases provide meaningful names to the NEON and general-purpose
@ registers, clarifying their use in the SHA-1 computation process.

@ k0-k3: NEON registers for the four SHA-1 round constants.
	k0		.req	q0 @ Functional Role: NEON vector register holding SHA1 round constant K_00_19.
	k1		.req	q1 @ Functional Role: NEON vector register holding SHA1 round constant K_20_39.
	k2		.req	q2 @ Functional Role: NEON vector register holding SHA1 round constant K_40_59.
	k3		.req	q3 @ Functional Role: NEON vector register holding SHA1 round constant K_60_79.

@ ta0/tb0, ta1/tb1: NEON registers for temporary round calculations.
	ta0		.req	q4 @ Functional Role: Temporary NEON register for intermediate round calculation.
	ta1		.req	q5 @ Functional Role: Temporary NEON register for intermediate round calculation.
	tb0		.req	q5 @ Functional Role: Temporary NEON register for intermediate round calculation.
	tb1		.req	q4 @ Functional Role: Temporary NEON register for intermediate round calculation.

@ dga, dgb, dgbs: NEON registers holding the SHA-1 state (A,B,C,D,E).
@ dga contains A,B,C,D. dgb/dgbs contains E.
	dga		.req	q6 @ Functional Role: NEON vector register holding SHA1 state variables A, B, C, D.
	dgb		.req	q7 @ Functional Role: NEON vector register holding SHA1 state variable E.
	dgbs		.req	s28 @ Functional Role: Scalar alias for the E component within dgb (q7).

@ dg0, dg1*: NEON registers for intermediate digest values during rounds.
	dg0		.req	q12 @ Functional Role: NEON vector register storing current digest values.
	dg1a0		.req	q13 @ Functional Role: NEON vector register for intermediate digest state.
	dg1a1		.req	q14 @ Functional Role: NEON vector register for intermediate digest state.
	dg1b0		.req	q14 @ Functional Role: NEON vector register for intermediate digest state.
	dg1b1		.req	q13 @ Functional Role: NEON vector register for intermediate digest state.

	/**
	 * @brief Performs a single SHA-1 round update using Cryptography Extensions.
	 * @details This macro orchestrates a single round of the SHA-1 algorithm
	 * by combining the `sha1h` instruction (to update the A state with E)
	 * with one of the `sha1c`, `sha1p`, or `sha1m` instructions, which
	 * implement the logical function for the round. This directly utilizes
	 * hardware acceleration for the core SHA-1 computations.
	 *
	 * @param op The specific SHA-1 logical function instruction (c for choose,
	 *           p for parity, or m for majority).
	 * @param ev Even/odd temporary register selector (0 or 1), used to
	 *           alternate between `ta0`/`ta1` for temporary calculations.
	 * @param rc Round constant register (e.g., `k0`, `k1`).
	 * @param s0 Message schedule register (or empty if not used).
	 * @param dg1 Previous digest state register.
	 * Functional Utility: Executes a single SHA1 round, including updating A from E
	 * and applying the round's logical function using ARMv8 Crypto Extensions.
	 */
	.macro		add_only, op, ev, rc, s0, dg1
	.ifnb		\s0 @ Conditional assembly: if s0 is not blank (i.e., s0 is provided).
	vadd.u32	tb\ev, q\s0, c @ Functional Utility: Add the message word (q\s0) and round constant (c).
	.endif
	sha1h.32	dg1b\ev, dg0 @ Functional Utility: ARMv8 CE instruction - updates A from E (dg0).
	.ifb		\dg1 @ Conditional assembly: if dg1 is blank (i.e., dg1 is not provided).
	sha1\op\().32	dg0, dg1a\ev, ta\ev @ Functional Utility: ARMv8 CE instruction - computes round logical function.
	.else
	sha1\op\().32	dg0, \dg1, ta\ev @ Functional Utility: ARMv8 CE instruction - computes round logical function.
	.endif
	.endm

	/**
	 * @brief Performs a SHA-1 round update and concurrently calculates next message words.
	 * @details This macro is central to the pipelined SHA-1 compression loop
	 * when using ARMv8 Cryptography Extensions. It executes a single SHA-1
	 * round by invoking `add_only`, and simultaneously uses `sha1su0` and `sha1su1`
	 * instructions to calculate the next four words of the message schedule
	 * (W[i] to W[i+3]). This interleaving of message scheduling and round
	 * computation significantly enhances the hardware implementation's efficiency.
	 *
	 * @param op The specific SHA-1 logical function instruction (c, p, or m).
	 * @param ev Even/odd temporary register selector (0 or 1).
	 * @param rc Round constant register.
	 * @param s0, s1, s2, s3 Message schedule registers for `sha1su0`.
	 * @param dg1 Previous digest state.
	 * Functional Utility: Executes one SHA1 round via hardware acceleration while
	 * concurrently generating the next set of message words, critical for pipelined
	 * execution.
	 */
	.macro		add_update, op, ev, rc, s0, s1, s2, s3, dg1
	sha1su0.32	q\s0, q\s1, q\s2 @ Functional Utility: ARMv8 CE instruction - computes first stage of message schedule update (W[i] = W[i-3] ^ W[i-8] ^ W[i-14]).
	add_only	\op, \ev, c, \s1, \dg1 @ Functional Utility: Perform the core SHA1 round computation.
	sha1su1.32	q\s0, q\s3 @ Functional Utility: ARMv8 CE instruction - computes second stage of message schedule update (W[i] ^= W[i-16], then ROL(W[i], 1)).
	.endm

	.align		6
.Lsha1_rcon:
	.word		0x5a827999, 0x5a827999, 0x5a827999, 0x5a827999 @ Functional Role: SHA1 round constant K_00_19 for rounds 0-19.
	.word		0x6ed9eba1, 0x6ed9eba1, 0x6ed9eba1, 0x6ed9eba1 @ Functional Role: SHA1 round constant K_20_39 for rounds 20-39.
	.word		0x8f1bbcdc, 0x8f1bbcdc, 0x8f1bbcdc, 0x8f1bbcdc @ Functional Role: SHA1 round constant K_40_59 for rounds 40-59.
	.word		0xca62c1d6, 0xca62c1d6, 0xca62c1d6, 0xca62c1d6 @ Functional Role: SHA1 round constant K_60_79 for rounds 60-79.

	/**
	 * @brief ARMv8 Cryptography Extensions accelerated SHA-1 transform.
	 * @details This function processes `blocks` number of 64-byte data blocks
	 * using the SHA-1 compression algorithm, leveraging the ARMv8 Cryptography
	 * Extensions for hardware acceleration. It takes the current SHA-1 hash state
	 * and updates it based on the input data.
	 *
	 * @param r0 (sst): Pointer to the `sha1_state` context structure (containing H0-H4).
	 * @param r1 (src): Pointer to the source data (64-byte aligned blocks).
	 * @param r2 (blocks): Number of 64-byte blocks to process.
	 *
	 * @pre `sst` points to a valid `sha1_state` structure with the current hash values.
	 * @pre `src` points to the input data, aligned to 64 bytes.
	 * @pre `blocks` is a non-negative integer indicating the number of blocks.
	 * @post The `sha1_state` structure pointed to by `sst` is updated with the hash
	 *       of all processed blocks.
	 */
ENTRY(sha1_ce_transform)
	@ Prologue: Load round constants and initial hash state into registers.
	@ Functional Utility: Load the address of the SHA-1 round constants table (.Lsha1_rcon) into `ip`.
	adr		ip, .Lsha1_rcon
	@ Functional Utility: Load the first two NEON quad-word registers (`k0-k1`) with K_00_19 and K_20_39.
	@ The `[ip, :128]!` syntax indicates loading 128 bits from `ip` and then incrementing `ip` by 128 bits.
	vld1.32		{k0-k1}, [ip, :128]!
	@ Functional Utility: Load the next two NEON quad-word registers (`k2-k3`) with K_40_59 and K_60_79.
	vld1.32		{k2-k3}, [ip, :128]

	@ Functional Utility: Load the first four SHA-1 chaining variables (H0-H3) from the `sha1_state` struct (r0) into NEON register `dga`.
	vld1.32		{dga}, [r0]
	@ Functional Utility: Load the fifth SHA-1 chaining variable (H4) from the `sha1_state` struct (r0 + 16 bytes offset) into NEON scalar register `dgbs`.
	vldr		dgbs, [r0, #16]

	@ Main loop: Processes one 64-byte block per iteration.
	@ Invariant: At the start of each iteration, r0 points to the sha1_state, r1 points to the current
	@ 64-byte data block, and r2 holds the number of remaining blocks.
0:	@ 1. Load 16 words (64 bytes) of message data into NEON registers `q8-q11`.
	@ Functional Utility: Load the first two quad-word (8 words) of message data from `r1` into `q8` and `q9`. `r1` is then incremented.
	vld1.32		{q8-q9}, [r1]!
	@ Functional Utility: Load the next two quad-word (8 words) of message data from `r1` into `q10` and `q11`. `r1` is then incremented.
	vld1.32		{q10-q11}, [r1]!
	@ Functional Utility: Decrement the block counter (`r2`). If zero, this is the last block.
	subs		r2, r2, #1

#ifndef CONFIG_CPU_BIG_ENDIAN
	@ 2. On little-endian systems, reverse byte order to match SHA-1's big-endian standard.
	@ Functional Utility: These `vrev32.8` instructions perform byte reversal on the loaded
	@ 32-bit words within the NEON registers. This ensures that the message data
	@ conforms to the big-endian format required by the SHA-1 algorithm, regardless
	@ of the system's endianness.
	vrev32.8	q8, q8
	vrev32.8	q9, q9
	vrev32.8	q10, q10
	vrev32.8	q11, q11
#endif

	@ 3. Execute the 80 rounds of SHA-1 compression using ARMv8 Cryptography Extensions.
	@    The `add_update` and `add_only` macros orchestrate the hardware instructions
	@    (`sha1c`, `sha1p`, `sha1m`, `sha1h`, `sha1su0`, `sha1su1`) for each round.
	@ Functional Utility: Initialize `ta0` by adding message word `q8` and round constant `k0`. This prepares for the first round.
	vadd.u32	ta0, q8, k0
	@ Functional Utility: Move current SHA1 digest state (`dga`) into `dg0` for processing.
	vmov		dg0, dga

	@ Rounds 0-19 (Choose function: F(B,C,D) = (B & C) | (~B & D))
	@ Functional Utility: These `add_update` macros execute 5 rounds each, for a total of 20 rounds.
	@ They use the `sha1c` (choose) instruction for the logical function and
	@ `sha1su0`/`sha1su1` to concurrently calculate the next message words.
	add_update	c, 0, k0,  8,  9, 10, 11, dgb
	add_update	c, 1, k0,  9, 10, 11,  8
	add_update	c, 0, k0, 10, 11,  8,  9
	add_update	c, 1, k0, 11,  8,  9, 10
	add_update	c, 0, k1,  8,  9, 10, 11

	@ Rounds 20-39 (Parity function: F(B,C,D) = B ^ C ^ D)
	@ Functional Utility: These `add_update` macros execute 5 rounds each, for a total of 20 rounds.
	@ They use the `sha1p` (parity) instruction for the logical function and
	@ `sha1su0`/`sha1su1` to concurrently calculate the next message words.
	add_update	p, 1, k1,  9, 10, 11,  8
	add_update	p, 0, k1, 10, 11,  8,  9
	add_update	p, 1, k1, 11,  8,  9, 10
	add_update	p, 0, k1,  8,  9, 10, 11
	add_update	p, 1, k2,  9, 10, 11,  8

	@ Rounds 40-59 (Majority function: F(B,C,D) = (B & C) | (B & D) | (C & D))
	@ Functional Utility: These `add_update` macros execute 5 rounds each, for a total of 20 rounds.
	@ They use the `sha1m` (majority) instruction for the logical function and
	@ `sha1su0`/`sha1su1` to concurrently calculate the next message words.
	add_update	m, 0, k2, 10, 11,  8,  9
	add_update	m, 1, k2, 11,  8,  9, 10
	add_update	m, 0, k2,  8,  9, 10, 11
	add_update	m, 1, k2,  9, 10, 11,  8
	add_update	m, 0, k3, 10, 11,  8,  9

	@ Rounds 60-79 (Parity function: F(B,C,D) = B ^ C ^ D)
	@ Functional Utility: These `add_update` and `add_only` macros execute 5 rounds each, for a total of 20 rounds.
	@ They use the `sha1p` (parity) instruction for the logical function. For the last few rounds,
	@ `add_only` is used as message word generation is no longer necessary.
	add_update	p, 1, k3, 11,  8,  9, 10
	add_only	p, 0, k3,  9 @ Functional Utility: Performs a single round update without message schedule generation.
	add_only	p, 1, k3, 10
	add_only	p, 0, k3, 11
	add_only	p, 1

	@ Epilogue for the current block.
	@ Functional Utility: Update the digest state with the result of the compression.
	@ This involves adding the initial hash values to the results of the 80 rounds,
	@ accumulating the current block's hash into the overall digest.
	vadd.u32	dga, dga, dg0 @ Functional Utility: Add previous H0-H3 to current round results (dg0)
	vadd.u32	dgb, dgb, dg1a0 @ Functional Utility: Add previous H4 to current round results (dg1a0)
	@ Functional Utility: Branch to the beginning of the loop (`0:`) if there are more blocks (`r2 > 0`) to process.
	bne		0b

	@ Functional Utility: Store the final updated SHA1 state back to memory.
	vst1.32		{dga}, [r0] @ Functional Utility: Store the updated H0-H3 from `dga` back into the `sha1_state` struct.
	vstr		dgbs, [r0, #16] @ Functional Utility: Store the updated H4 from `dgbs` back into the `sha1_state` struct (offset 16 bytes).
	@ Functional Utility: Return from the function.
	bx		lr
ENDPROC(sha1_ce_transform)
