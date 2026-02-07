/*
 *	Implement fast SHA-1 with AVX2 instructions. (x86_64)
 *
 * This file is provided under a dual BSD/GPLv2 license.  When using or
 * redistributing this file, you may do so under either license.
 *
 * GPL LICENSE SUMMARY
 *
 * Copyright(c) 2014 Intel Corporation.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of version 2 of the GNU General Public License as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * Contact Information:
 * Ilya Albrekht <ilya.albrekht@intel.com>
 * Maxim Locktyukhin <maxim.locktyukhin@intel.com>
 * Ronen Zohar <ronen.zohar@intel.com>
 * Chandramouli Narayanan <mouli@linux.intel.com>
 *
 * BSD LICENSE
 *
 * Copyright(c) 2014 Intel Corporation.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in
 * the documentation and/or other materials provided with the
 * distribution.
 * Neither the name of Intel Corporation nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 * @file sha1_avx2_x86_64_asm.S
 * @brief High-performance SHA-1 implementation leveraging Intel AVX2 instructions for x86_64.
 * @details This assembly file provides a highly optimized implementation of the
 * SHA-1 (Secure Hash Algorithm 1) compression function, specifically tailored
 * for x86_64 processors equipped with the Intel AVX2 (Advanced Vector Extensions 2)
 * instruction set. It heavily utilizes AVX2's 256-bit vector capabilities to
 * process data in parallel, significantly accelerating the cryptographic process.
 *
 * **Optimization Techniques:**
 * - **AVX2 Vectorization**: Extensive use of AVX2 instructions (`vmovups`, `vpshufb`,
 *   `vpaddd`, `vpxor`, `vpalignr`, etc.) for vectorized operations on message words
 *   and hash values.
 * - **Software Pipelining**: The implementation employs software pipelining to
 *   process multiple blocks concurrently, hiding latency and maximizing throughput.
 * - **Message Schedule Pre-calculation**: Message schedule words and `K+W` values
 *   are pre-calculated and stored to optimize round computations.
 * - **Blended AVX2 and ALU Instructions**: Combines vector (AVX2) and scalar (ALU)
 *   instructions to efficiently handle data dependencies and optimize execution flow.
 * - **Register Rotation**: Clever rotation of YMM registers (`WY0` to `WY28`) for
 *   efficient message schedule generation without excessive data movement.
 * This implementation is designed to provide maximum SHA-1 hashing performance
 * on AVX2-enabled x86_64 systems within the Linux kernel.
 */

/*
 * SHA-1 implementation with Intel(R) AVX2 instruction set extensions.
 *
 *This implementation is based on the previous SSSE3 release:
 *Visit http://software.intel.com/en-us/articles/
 *and refer to improving-the-performance-of-the-secure-hash-algorithm-1/
 *
 *Updates 20-byte SHA-1 record at start of 'state', from 'input', for
 *even number of 'blocks' consecutive 64-byte blocks.
 *
 *extern "C" void sha1_transform_avx2(
 *	struct sha1_state *state, const u8* input, int blocks );
 */

#include <linux/linkage.h>

#define	CTX	%rdi	/* Functional Role: Pointer to `struct sha1_state` (argument 1). */
#define BUF	%rsi	/* Functional Role: Pointer to input data buffer (argument 2). */
#define CNT	%rdx	/* Functional Role: Number of 64-byte blocks to process (argument 3). */

#define	REG_A	%ecx	/* Functional Role: General-purpose register (GPR) for SHA-1 state A (32-bit). */
#define	REG_B	%esi	/* Functional Role: GPR for SHA-1 state B (32-bit). */
#define	REG_C	%edi	/* Functional Role: GPR for SHA-1 state C (32-bit). */
#define	REG_D	%eax	/* Functional Role: GPR for SHA-1 state D (32-bit). */
#define	REG_E	%edx	/* Functional Role: GPR for SHA-1 state E (32-bit). */
#define	REG_TB	%ebx	/* Functional Role: GPR for temporary storage of B, used for state rotation (32-bit). */
#define	REG_TA	%r12d	/* Functional Role: GPR for temporary storage of A, used for state rotation (32-bit). */
#define	REG_RA	%rcx	/* Functional Role: GPR for SHA-1 state A (64-bit, full register). */
#define	REG_RB	%rsi	/* Functional Role: GPR for SHA-1 state B (64-bit, full register). */
#define	REG_RC	%rdi	/* Functional Role: GPR for SHA-1 state C (64-bit, full register). */
#define	REG_RD	%rax	/* Functional Role: GPR for SHA-1 state D (64-bit, full register). */
#define	REG_RE	%rdx	/* Functional Role: GPR for SHA-1 state E (64-bit, full register). */
#define	REG_RTA	%r12	/* Functional Role: GPR for temporary storage of A (64-bit, full register). */
#define	REG_RTB	%rbx	/* Functional Role: GPR for temporary storage of B (64-bit, full register). */
#define	REG_T1	%r11d	/* Functional Role: GPR for general temporary use (32-bit). */
#define	xmm_mov	vmovups	/* Functional Role: AVX2 instruction alias for moving unaligned packed single-precision floating-point values or packed doubleword integers. */
#define	avx2_zeroupper	vzeroupper /* Functional Role: AVX2 instruction alias to zero the upper 128 bits of all YMM registers. */
#define	RND_F1	1	/* Functional Role: Identifier for SHA-1 round function F1 (Ch/Choose). */
#define	RND_F2	2	/* Functional Role: Identifier for SHA-1 round function F2 (Parity/XOR). */
#define	RND_F3	3	/* Functional Role: Identifier for SHA-1 round function F3 (Maj/Majority). */

/**
 * @brief Macro to set up aliases for general-purpose registers used in SHA-1 computations.
 * @details This macro defines symbolic aliases for a set of general-purpose
 * registers, mapping them to specific functional roles (A, B, C, D, E for SHA-1
 * state, TB, TA for temporaries). This improves readability and maintainability
 * of the assembly code by using descriptive names. It defines both 32-bit and
 * 64-bit register aliases.
 * Functional Utility: Simplifies register management and improves code readability by assigning functional names.
 */
.macro REGALLOC
	.set A, REG_A
	.set B, REG_B
	.set C, REG_C
	.set D, REG_D
	.set E, REG_E
	.set TB, REG_TB
	.set TA, REG_TA

	.set RA, REG_RA
	.set RB, REG_RB
	.set RC, REG_RC
	.set RD, REG_RD
	.set RE, REG_RE

	.set RTA, REG_RTA
	.set RTB, REG_RTB

	.set T1, REG_T1
.endm

#define HASH_PTR	%r9	/* Functional Role: GPR holding pointer to the SHA-1 hash context. */
#define BLOCKS_CTR	%r8	/* Functional Role: GPR holding the counter for remaining blocks to process. */
#define BUFFER_PTR	%r10	/* Functional Role: GPR holding pointer to the current input data buffer for first pipeline stage. */
#define BUFFER_PTR2	%r13	/* Functional Role: GPR holding pointer to the current input data buffer for second pipeline stage. */

#define PRECALC_BUF	%r14	/* Functional Role: GPR holding pointer to the pre-calculation buffer for message words. */
#define WK_BUF		%r15	/* Functional Role: GPR holding pointer to the working buffer for K+W values. */

#define W_TMP		%xmm0	/* Functional Role: XMM register for temporary message word storage. */
#define WY_TMP		%ymm0	/* Functional Role: YMM register for temporary message word storage (256-bit). */
#define WY_TMP2		%ymm9	/* Functional Role: YMM register for temporary message word storage (256-bit). */

/* AVX2 variables */
#define WY0		%ymm3	/* Functional Role: YMM register for message schedule words W[0]-W[7] (or rotating alias). */
#define WY4		%ymm5	/* Functional Role: YMM register for message schedule words W[8]-W[15] (or rotating alias). */
#define WY08		%ymm7	/* Functional Role: YMM register for message schedule words W[16]-W[23] (or rotating alias). */
#define WY12		%ymm8	/* Functional Role: YMM register for message schedule words W[24]-W[31] (or rotating alias). */
#define WY16		%ymm12	/* Functional Role: YMM register for message schedule words W[32]-W[39] (or rotating alias). */
#define WY20		%ymm13	/* Functional Role: YMM register for message schedule words W[40]-W[47] (or rotating alias). */
#define WY24		%ymm14	/* Functional Role: YMM register for message schedule words W[48]-W[55] (or rotating alias). */
#define WY28		%ymm15	/* Functional Role: YMM register for message schedule words W[56]-W[63] (or rotating alias). */

#define YMM_SHUFB_BSWAP	%ymm10	/* Functional Role: YMM register holding the shuffle control mask for byte-swapping. */

/*
 * Keep 2 iterations precalculated at a time:
 *    - 80 DWORDs per iteration * 2
 */
#define W_SIZE		(80*2*2 +16) /* Functional Role: Size of the message word buffer for pre-calculation, accommodating 2 full 80-round SHA-1 iterations (160 DWORDS) plus padding. */

#define WK(t)	((((t) % 80) / 4)*32 + ( (t) % 4)*4 + ((t)/80)*16 )(WK_BUF) /* Functional Role: Macro to calculate offset within `WK_BUF` for `K+W[t]`. */
#define PRECALC_WK(t)	((t)*2*2)(PRECALC_BUF) /* Functional Role: Macro to calculate offset within `PRECALC_BUF` for `W[t]`. */


/**
 * @brief Macro to update a hash component by adding a new value to it.
 * @details This macro performs an addition operation, adding `val` to `hash`,
 * and then moves the result back into `hash`. It's a simple helper to
 * accumulate the results of SHA-1 rounds into the main hash state registers.
 * @param hash The destination register (e.g., A, B, C, D, E).
 * @param val The value to add to `hash`.
 * Functional Utility: Accumulates the output of SHA-1 rounds into the hash state.
 */
.macro UPDATE_HASH  hash, val
	add	\hash, \val /* Functional Utility: Adds `val` to the content of `hash`. */
	mov	\val, \hash /* Functional Utility: Moves the result from `val` back into `hash`. */
.endm

/**
 * @brief Macro to reset aliases for YMM registers to their initial configuration for message scheduling.
 * @details This macro defines aliases `WY_00` through `WY_32` to specific
 * YMM registers (`WY0` through `WY28`). This is used to logically reset the
 * message word "window" before starting a new sequence of message schedule
 * generation, ensuring consistent register assignment.
 * Functional Utility: Initializes the logical mapping of YMM registers for message schedule generation.
 */
.macro PRECALC_RESET_WY
	.set WY_00, WY0
	.set WY_04, WY4
	.set WY_08, WY08
	.set WY_12, WY12
	.set WY_16, WY16
	.set WY_20, WY20
	.set WY_24, WY24
	.set WY_28, WY28
	.set WY_32, WY_00 /* Functional Role: Alias for `WY0` as the "next" register in the rotating scheme. */
.endm

/**
 * @brief Macro to rotate the aliases for YMM registers used in message scheduling.
 * @details This macro performs a logical rotation of the YMM register aliases.
 * This effectively shifts the "window" of message words (`WY_00` to `WY_32`)
 * for the next set of computations without physically moving data between
 * registers. This is a key part of the pipelined message schedule generation.
 * Functional Utility: Logically shifts the YMM register aliases to process subsequent message schedule words.
 */
.macro PRECALC_ROTATE_WY
	/* Rotate macros */
	.set WY_32, WY_28
	.set WY_28, WY_24
	.set WY_24, WY_20
	.set WY_20, WY_16
	.set WY_16, WY_12
	.set WY_12, WY_08
	.set WY_08, WY_04
	.set WY_04, WY_00
	.set WY_00, WY_32

	/* Define register aliases */
	.set WY, WY_00
	.set WY_minus_04, WY_04
	.set WY_minus_08, WY_08
	.set WY_minus_12, WY_12
	.set WY_minus_16, WY_16
	.set WY_minus_20, WY_20
	.set WY_minus_24, WY_24
	.set WY_minus_28, WY_28
	.set WY_minus_32, WY
.endm

/**
 * @brief Pre-calculates SHA-1 message schedule words and K+W values for rounds 0-15 using AVX2.
 * @details This macro implements the message schedule generation (`W[i]`) and
 * the `K+W[i]` values for rounds 0-15 of SHA-1. It processes 8 rounds per
 * vector iteration, blending AVX2 and ALU instructions for efficiency.
 * It loads input data, performs byte-swapping, adds round constants, and
 * stores the result in a pre-calculation buffer.
 * Functional Utility: Generates SHA-1 message schedule words and K+W values for initial rounds with AVX2.
 */
.macro PRECALC_00_15
	.if (i == 0) /* Functional Utility: Initializes and rotates registers only for the very first iteration (round 0). */
		PRECALC_RESET_WY
		PRECALC_ROTATE_WY
	.endif

	/* message scheduling pre-compute for rounds 0-15 */
	.if   ((i & 7) == 0) /* Functional Utility: Logic for every 8th round (0, 8, 16, etc.) to load initial data. */
		/*
		 * blended AVX2 and ALU instruction scheduling
		 * 1 vector iteration per 8 rounds
		 */
		// Functional Utility: Loads 256 bits (8 DWORDS) of message data from `BUFFER_PTR` into `W_TMP`.
		vmovdqu (i * 2)(BUFFER_PTR), W_TMP
	.elseif ((i & 7) == 1) /* Functional Utility: Logic for every (8k+1)th round. */
		// Functional Utility: Inserts the second 128-bit lane from `BUFFER_PTR2` into `WY_TMP`.
		vinsertf128 $1, ((i-1) * 2)(BUFFER_PTR2),
			 WY_TMP, WY_TMP
	.elseif ((i & 7) == 2) /* Functional Utility: Logic for every (8k+2)th round. */
		// Functional Utility: Performs byte-swapping on `WY_TMP` using `YMM_SHUFB_BSWAP` and stores result in `WY`.
		vpshufb YMM_SHUFB_BSWAP, WY_TMP, WY
	.elseif ((i & 7) == 4) /* Functional Utility: Logic for every (8k+4)th round. */
		// Functional Utility: Adds the constant `K_XMM` to `WY` and stores the result in `WY_TMP`.
		vpaddd  K_XMM + K_XMM_AR(%rip), WY, WY_TMP
	.elseif ((i & 7) == 7) /* Functional Utility: Logic for every (8k+7)th round (last in the 8-round block). */
		// Functional Utility: Stores the pre-calculated `K+W` value (`WY_TMP`) into `PRECALC_WK` buffer.
		vmovdqu  WY_TMP, PRECALC_WK(i&~7)

		PRECALC_ROTATE_WY /* Functional Utility: Rotates the YMM register aliases for the next block of message words. */
	.endif
.endm

/**
 * @brief Pre-calculates SHA-1 message schedule words and K+W values for rounds 16-31 using AVX2.
 * @details This macro generates message schedule words (`W[i]`) and `K+W[i]` values
 * for rounds 16-31. It handles the specific SHA-1 message expansion formula for
 * these rounds, including the `W[i-3]` dependency. It uses a "brute force"
 * vectorization approach due to the data dependencies, but still achieves
 * efficiency through AVX2 instructions.
 * Functional Utility: Generates SHA-1 message schedule words and K+W values for middle rounds (16-31) with AVX2.
 */
.macro PRECALC_16_31
	/*
	 * message scheduling pre-compute for rounds 16-31
	 * calculating last 32 w[i] values in 8 XMM registers
	 * pre-calculate K+w[i] values and store to mem
	 * for later load by ALU add instruction
	 *
	 * "brute force" vectorization for rounds 16-31 only
	 * due to w[i]->w[i-3] dependency
	 */
	.if   ((i & 7) == 0) /* Functional Utility: Logic for every 8th round (16, 24, etc.). */
		/*
		 * blended AVX2 and ALU instruction scheduling
		 * 1 vector iteration per 8 rounds
		 */
		// Functional Utility: Vector right-aligns 8 bytes, effectively getting W[i-14].
		vpalignr	$8, WY_minus_16, WY_minus_12, WY
		// Functional Utility: Vector right logical shift by 4 bytes, effectively getting W[i-3].
		vpsrldq	$4, WY_minus_04, WY_TMP
	.elseif ((i & 7) == 1) /* Functional Utility: Logic for every (8k+1)th round. */
		// Functional Utility: Vector XOR operation.
		vpxor	WY_minus_08, WY, WY
		vpxor	WY_minus_16, WY_TMP, WY_TMP
	.elseif ((i & 7) == 2) /* Functional Utility: Logic for every (8k+2)th round. */
		// Functional Utility: Vector XOR operation.
		vpxor	WY_TMP, WY, WY
		// Functional Utility: Vector left logical shift by 12 bytes.
		vpslldq	$12, WY, WY_TMP2
	.elseif ((i & 7) == 3) /* Functional Utility: Logic for every (8k+3)th round. */
		// Functional Utility: Vector left logical shift by 1 bit.
		vpslld	$1, WY, WY_TMP
		// Functional Utility: Vector right logical shift by 31 bits.
		vpsrld	$31, WY, WY
	.elseif ((i & 7) == 4) /* Functional Utility: Logic for every (8k+4)th round. */
		// Functional Utility: Vector OR operation.
		vpor	WY, WY_TMP, WY_TMP
		// Functional Utility: Vector left logical shift by 2 bits.
		vpslld	$2, WY_TMP2, WY
	.elseif ((i & 7) == 5) /* Functional Utility: Logic for every (8k+5)th round. */
		// Functional Utility: Vector right logical shift by 30 bits.
		vpsrld	$30, WY_TMP2, WY_TMP2
		// Functional Utility: Vector XOR operation.
		vpxor	WY, WY_TMP, WY_TMP
	.elseif ((i & 7) == 7) /* Functional Utility: Logic for every (8k+7)th round (last in the 8-round block). */
		// Functional Utility: Vector XOR operation.
		vpxor	WY_TMP2, WY_TMP, WY
		// Functional Utility: Adds the constant `K_XMM` to `WY` and stores the result in `WY_TMP`.
		vpaddd  K_XMM + K_XMM_AR(%rip), WY, WY_TMP
		// Functional Utility: Stores the pre-calculated `K+W` value (`WY_TMP`) into `PRECALC_WK` buffer.
		vmovdqu	WY_TMP, PRECALC_WK(i&~7)

		PRECALC_ROTATE_WY /* Functional Utility: Rotates the YMM register aliases for the next block of message words. */
	.endif
.endm

/**
 * @brief Pre-calculates SHA-1 message schedule words and K+W values for rounds 32-79 using AVX2.
 * @details This macro generates message schedule words (`W[i]`) and `K+W[i]` values
 * for rounds 32-79. It uses an alternative message expansion formula to break
 * direct dependencies (`W[i] = (W[i-6] ^ W[i-16] ^ W[i-28] ^ W[i-32]) rol 2`)
 * compared to the standard SHA-1 formula. This allows for more efficient
 * vectorization, crucial for high performance.
 * Functional Utility: Generates SHA-1 message schedule words and K+W values for later rounds (32-79) with AVX2.
 */
.macro PRECALC_32_79
	/*
	 * in SHA-1 specification:
	 * w[i] = (w[i-3] ^ w[i-8]  ^ w[i-14] ^ w[i-16]) rol 1
	 * instead we do equal:
	 * w[i] = (w[i-6] ^ w[i-16] ^ w[i-28] ^ w[i-32]) rol 2
	 * allows more efficient vectorization
	 * since w[i]=>w[i-3] dependency is broken
	 */

	.if   ((i & 7) == 0) /* Functional Utility: Logic for every 8th round (32, 40, etc.). */
	/*
	 * blended AVX2 and ALU instruction scheduling
	 * 1 vector iteration per 8 rounds
	 */
		// Functional Utility: Vector right-aligns 8 bytes, effectively getting W[i-6].
		vpalignr	$8, WY_minus_08, WY_minus_04, WY_TMP
	.elseif ((i & 7) == 1) /* Functional Utility: Logic for every (8k+1)th round. */
		/* W is W_minus_32 before xor */
		// Functional Utility: Vector XOR operation with W[i-32].
		vpxor	WY_minus_28, WY, WY
	.elseif ((i & 7) == 2) /* Functional Utility: Logic for every (8k+2)th round. */
		// Functional Utility: Vector XOR operation with W[i-16].
		vpxor	WY_minus_16, WY_TMP, WY_TMP
	.elseif ((i & 7) == 3) /* Functional Utility: Logic for every (8k+3)th round. */
		// Functional Utility: Vector XOR operation with W[i-6].
		vpxor	WY_TMP, WY, WY
	.elseif ((i & 7) == 4) /* Functional Utility: Logic for every (8k+4)th round. */
		// Functional Utility: Vector left logical shift by 2 bits.
		vpslld	$2, WY, WY_TMP
	.elseif ((i & 7) == 5) /* Functional Utility: Logic for every (8k+5)th round. */
		// Functional Utility: Vector right logical shift by 30 bits.
		vpsrld	$30, WY, WY
		// Functional Utility: Vector OR operation to combine shifted parts for 2-bit rotation.
		vpor	WY, WY_TMP, WY
	.elseif ((i & 7) == 7) /* Functional Utility: Logic for every (8k+7)th round (last in the 8-round block). */
		// Functional Utility: Adds the constant `K_XMM` to `WY` and stores the result in `WY_TMP`.
		vpaddd  K_XMM + K_XMM_AR(%rip), WY, WY_TMP
		// Functional Utility: Stores the pre-calculated `K+W` value (`WY_TMP`) into `PRECALC_WK` buffer.
		vmovdqu	WY_TMP, PRECALC_WK(i&~7)

		PRECALC_ROTATE_WY /* Functional Utility: Rotates the YMM register aliases for the next block of message words. */
	.endif
.endm

/**
 * @brief Dispatcher macro for message schedule pre-calculation based on round number.
 * @details This macro selects the appropriate `PRECALC_00_15`, `PRECALC_16_31`,
 * or `PRECALC_32_79` macro based on the current round number (`r`). It also
 * sets the base offset for `K_XMM` to load the correct SHA-1 round constant.
 *
 * @param r The current round number (0-79).
 * @param s A parameter for the underlying PRECALC macros (unused here).
 * Functional Utility: Directs message schedule pre-calculation to the correct specialized macro based on the SHA-1 round phase.
 */
.macro PRECALC r, s
	.set i, 

	.if (i < 40) /* Functional Utility: Sets `K_XMM` offset for rounds 0-39 (K1 constant). */
		.set K_XMM, 32*0
	.elseif (i < 80) /* Functional Utility: Sets `K_XMM` offset for rounds 40-79 (K2 constant). */
		.set K_XMM, 32*1
	.elseif (i < 120) /* Functional Utility: Sets `K_XMM` offset for rounds 80-119 (K3 constant, for pipelined 2nd block). */
		.set K_XMM, 32*2
	.else /* Functional Utility: Sets `K_XMM` offset for rounds 120-159 (K4 constant, for pipelined 2nd block). */
		.set K_XMM, 32*3
	.endif

	.if (i<32) /* Functional Utility: Calls `PRECALC_00_15` for initial rounds. */
		PRECALC_00_15	\s
	.elseif (i<64) /* Functional Utility: Calls `PRECALC_16_31` for middle rounds. */
		PRECALC_16_31	\s
	.elseif (i < 160) /* Functional Utility: Calls `PRECALC_32_79` for later rounds (up to 159 for pipelining). */
		PRECALC_32_79	\s
	.endif
.endm

/**
 * @brief Macro to logically rotate the SHA-1 state registers (A, B, C, D, E).
 * @details This macro swaps the values between a set of general-purpose
 * registers that hold the SHA-1 hash state (A, B, C, D, E). This rotation
 * is a fundamental part of the SHA-1 algorithm, where the output of one
 * round becomes the input for the next, with specific shifting of the state variables.
 * Functional Utility: Shifts the SHA-1 hash state registers (A, B, C, D, E) for the next round.
 */
.macro ROTATE_STATE
	.set T_REG, E
	.set E, D
	.set D, C
	.set C, B
	.set B, TB
	.set TB, A
	.set A, T_REG

	.set T_REG, RE
	.set RE, RD
	.set RD, RC
	.set RC, RB
	.set RB, RTB
	.set RTB, RA
	.set RA, T_REG
.endm

/* Macro relies on saved ROUND_Fx */

/**
 * @brief Dispatcher macro for SHA-1 round functions.
 * @details This macro acts as a switch to call the appropriate SHA-1 round
 * function macro (`ROUND_F1`, `ROUND_F2`, or `ROUND_F3`) based on the
 * current round's functional type (`f`). This allows the main loop to
 * abstract the specifics of the round function.
 * @param f The functional type of the round (RND_F1, RND_F2, or RND_F3).
 * @param r The current round number.
 * Functional Utility: Dispatches to the correct SHA-1 round function macro.
 */
.macro RND_FUN f, r
	.if (\f == RND_F1)
		ROUND_F1	
	.elseif (\f == RND_F2)
		ROUND_F2	
	.elseif (\f == RND_F3)
		ROUND_F3	
	.endif
.endm

/**
 * @brief Macro to execute two SHA-1 rounds and rotate state.
 * @details This macro executes two consecutive rounds of the SHA-1 algorithm.
 * It first determines the current round function, calls the `RND_FUN`
 * dispatcher, and then rotates the SHA-1 hash state registers (`ROTATE_STATE`).
 * It also updates the round function type (`ROUND_FUNC`) dynamically based
 * on the round number.
 * @param r The starting round number for this pair of rounds.
 * Functional Utility: Executes a pair of SHA-1 rounds and prepares the state for subsequent rounds.
 */
.macro RR r
	.set round_id, ( % 80)

	.if (round_id == 0)        /* Precalculate F for first round */
		.set ROUND_FUNC, RND_F1 /* Functional Utility: Sets the round function type to F1 for rounds 0-19. */
		mov	B, TB /* Functional Utility: Saves current B value for next round calculation. */

		rorx	$(32-30), B, B    /* b>>>2 */ /* Functional Utility: Rotates B right by 2 bits. */
		andn	D, TB, T1 /* Functional Utility: Computes (~B & D). */
		and	C, TB /* Functional Utility: Computes (B & C). */
		xor	T1, TB /* Functional Utility: Combines results to get F1. */
	.endif

	RND_FUN ROUND_FUNC,  /* Functional Utility: Executes the specific round function for round `r`. */
	ROTATE_STATE /* Functional Utility: Rotates the SHA-1 state registers. */

	.if   (round_id == 18) /* Functional Utility: Changes round function to F2 for rounds 20-39. */
		.set ROUND_FUNC, RND_F2
	.elseif (round_id == 38) /* Functional Utility: Changes round function to F3 for rounds 40-59. */
		.set ROUND_FUNC, RND_F3
	.elseif (round_id == 58) /* Functional Utility: Changes round function to F2 for rounds 60-79. */
		.set ROUND_FUNC, RND_F2
	.endif

	.set round_id, ( (+1) % 80) /* Functional Utility: Updates round_id for the next round. */

	RND_FUN ROUND_FUNC, (+1) /* Functional Utility: Executes the specific round function for round `r+1`. */
	ROTATE_STATE /* Functional Utility: Rotates the SHA-1 state registers. */
.endm

/**
 * @brief Implements SHA-1 round function F1 (Ch/Choose) and related calculations.
 * @details This macro performs the core computations for rounds 0-19 of SHA-1,
 * which use the Choose function `F(B,C,D) = (B & C) | (~B & D)`. It updates
 * the hash state (A, B, C, D, E), performs bitwise operations for `F1`,
 * and includes `PRECALC` for message schedule generation for subsequent blocks.
 * @param r The current round number.
 * Functional Utility: Executes SHA-1 round computations using the F1 (Choose) function.
 */
.macro ROUND_F1 r
	add	WK(), E /* Functional Utility: Adds the pre-calculated `K+W` for the current round to E. */

	andn	C, A, T1			/* ~b&d */ /* Functional Utility: Computes (NOT C) AND A, storing in T1. */
	lea	(RE,RTB), E		/* Add F from the previous round */ /* Functional Utility: Adds result of previous F to E. */

	rorx	$(32-5), A, TA		/* T2 = A >>> 5 */ /* Functional Utility: Rotates A left by 5 bits, storing in TA. */
	rorx	$(32-30),A, TB		/* b>>>2 for next round */ /* Functional Utility: Rotates A right by 2 bits, storing in TB (for next round B). */

	PRECALC	()			/* msg scheduling for next 2 blocks */ /* Functional Utility: Dispatches to pre-calculate message words for pipelining. */

	/*
	 * Calculate F for the next round
	 * (b & c) ^ andn[b, d]
	 */
	and	B, A			/* b&c */ /* Functional Utility: Computes B AND A, storing in A. */
	xor	T1, A			/* F1 = (b&c) ^ (~b&d) */ /* Functional Utility: Computes XOR with T1 (which held ~B&D) to complete F1. */

	lea	(RE,RTA), E		/* E += A >>> 5 */ /* Functional Utility: Adds rotated A (TA) to E. */
.endm

/**
 * @brief Implements SHA-1 round function F2 (Parity/XOR) and related calculations.
 * @details This macro performs the core computations for rounds 20-39 and 60-79
 * of SHA-1, which use the Parity function `F(B,C,D) = B ^ C ^ D`. It updates
 * the hash state, performs bitwise operations for `F2`, and includes `PRECALC`
 * for message schedule generation.
 * @param r The current round number.
 * Functional Utility: Executes SHA-1 round computations using the F2 (Parity) function.
 */
.macro ROUND_F2 r
	add	WK(), E /* Functional Utility: Adds the pre-calculated `K+W` for the current round to E. */
	lea	(RE,RTB), E		/* Add F from the previous round */ /* Functional Utility: Adds result of previous F to E. */

	/* Calculate F for the next round */
	rorx	$(32-5), A, TA		/* T2 = A >>> 5 */ /* Functional Utility: Rotates A left by 5 bits, storing in TA. */
	.if ((round_id) < 79) /* Functional Utility: Conditionally rotates A for B in next round, only if not the final round. */
		rorx	$(32-30), A, TB	/* b>>>2 for next round */
	.endif
	PRECALC	()			/* msg scheduling for next 2 blocks */ /* Functional Utility: Dispatches to pre-calculate message words for pipelining. */

	.if ((round_id) < 79) /* Functional Utility: Conditionally performs XOR operations for F2, only if not the final round. */
		xor	B, A
	.endif

	add	TA, E			/* E += A >>> 5 */ /* Functional Utility: Adds rotated A (TA) to E. */

	.if ((round_id) < 79) /* Functional Utility: Conditionally performs XOR operation for F2, only if not the final round. */
		xor	C, A
	.endif
.endm

/**
 * @brief Implements SHA-1 round function F3 (Maj/Majority) and related calculations.
 * @details This macro performs the core computations for rounds 40-59 of SHA-1,
 * which use the Majority function `F(B,C,D) = (B & C) | (B & D) | (C & D)`.
 * It updates the hash state, performs bitwise operations for `F3`, and includes
 * `PRECALC` for message schedule generation.
 * @param r The current round number.
 * Functional Utility: Executes SHA-1 round computations using the F3 (Majority) function.
 */
.macro ROUND_F3 r
	add	WK(), E /* Functional Utility: Adds the pre-calculated `K+W` for the current round to E. */
	PRECALC	()			/* msg scheduling for next 2 blocks */ /* Functional Utility: Dispatches to pre-calculate message words for pipelining. */

	lea	(RE,RTB), E		/* Add F from the previous round */ /* Functional Utility: Adds result of previous F to E. */

	mov	B, T1 /* Functional Utility: Copies B to T1. */
	or	A, T1 /* Functional Utility: Computes A OR T1 (B OR A). */

	rorx	$(32-5), A, TA		/* T2 = A >>> 5 */ /* Functional Utility: Rotates A left by 5 bits, storing in TA. */
	rorx	$(32-30), A, TB		/* b>>>2 for next round */ /* Functional Utility: Rotates A right by 2 bits, storing in TB (for next round B). */

	/* Calculate F for the next round
	 * (b and c) or (d and (b or c))
	 */
	and	C, T1 /* Functional Utility: Computes C AND T1 (C AND (B OR A)). */
	and	B, A /* Functional Utility: Computes B AND A, storing in A. */
	or	T1, A /* Functional Utility: Computes A OR T1 to combine parts of Majority function. */

	add	TA, E			/* E += A >>> 5 */ /* Functional Utility: Adds rotated A (TA) to E. */
.endm

/* Add constant only if (%2 > %3) condition met (uses RTA as temp)
 * %1 + %2 >= %3 ? %4 : 0
 */
/**
 * @brief Conditionally adds a value to a register based on a comparison.
 * @details This macro attempts to add a constant `d` to register `a`. However,
 * the addition only takes effect if the comparison `b >= c` is true. This is
 * typically used for control flow in pipelined loops, such as advancing
 * buffer pointers only when sufficient blocks remain to be processed.
 * @param a The register to which `d` might be added.
 * @param b The register holding the value for comparison (e.g., block counter).
 * @param c The value to compare `b` against.
 * @param d The constant value to add to `a` if the condition is met.
 * Functional Utility: Controls conditional updates of registers based on a comparison, e.g., advancing buffer pointers in a pipelined loop.
 */
.macro ADD_IF_GE a, b, c, d
	mov     \a, RTA /* Functional Utility: Moves `a` into temporary register RTA. */
	add     $\d, RTA /* Functional Utility: Adds `d` to RTA. */
	cmp     $\c, \b /* Functional Utility: Compares `b` with `c`. */
	cmovge  RTA, \a /* Functional Utility: Conditionally moves RTA (a+d) to `a` if `b >= c`. */
.endm

/*
 * macro implements 80 rounds of SHA-1, for multiple blocks with s/w pipelining
 */
/**
 * @brief Main body macro for pipelined SHA-1 processing of multiple blocks.
 * @details This macro encapsulates the core logic for processing multiple 64-byte
 * SHA-1 blocks using software pipelining. It manages the SHA-1 hash state (A, B, C, D, E),
 * message schedule generation, round computations (via `RR` macro), and
 * accumulation of results. The pipelining allows for concurrent processing
 * of two blocks, significantly improving throughput.
 * Functional Utility: Orchestrates the pipelined execution of SHA-1 compression for multiple blocks using AVX2.
 */
.macro SHA1_PIPELINED_MAIN_BODY

	REGALLOC /* Functional Utility: Initializes register aliases for working registers. */

	// Functional Utility: Loads the initial SHA-1 state (H0-H4) from `HASH_PTR` into GPRs A, B, C, D, E.
	mov	(HASH_PTR), A
	mov	4(HASH_PTR), B
	mov	8(HASH_PTR), C
	mov	12(HASH_PTR), D
	mov	16(HASH_PTR), E

	// Functional Utility: Sets up `PRECALC_BUF` and `WK_BUF` pointers on the stack for message word pre-calculation.
	mov	%rsp, PRECALC_BUF
	lea	(2*4*80+32)(%rsp), WK_BUF

	# Precalc WK for first 2 blocks
	// Functional Utility: Conditionally advances `BUFFER_PTR2` if there are at least 2 blocks.
	ADD_IF_GE BUFFER_PTR2, BLOCKS_CTR, 2, 64
	.set i, 0
	.rept    160 /* Functional Utility: Iterates 160 times (2 * 80 rounds) to pre-calculate message words for two blocks. */
		PRECALC i /* Functional Utility: Dispatches to message schedule pre-calculation. */
		.set i, i + 1
	.endr

	/* Go to next block if needed */
	// Functional Utility: Conditionally advances `BUFFER_PTR` if there are at least 3 blocks.
	ADD_IF_GE BUFFER_PTR, BLOCKS_CTR, 3, 128
	// Functional Utility: Conditionally advances `BUFFER_PTR2` if there are at least 4 blocks.
	ADD_IF_GE BUFFER_PTR2, BLOCKS_CTR, 4, 128
	// Functional Utility: Swaps `WK_BUF` and `PRECALC_BUF` to prepare for the next pipeline stage.
	xchg	WK_BUF, PRECALC_BUF

	.align 32
.L_loop: /* Functional Role: Main loop for processing blocks in a pipelined fashion. */
	/*
	 * code loops through more than one block
	 * we use K_BASE value as a signal of a last block,
	 * it is set below by: cmovae BUFFER_PTR, K_BASE
	 */
	// Functional Utility: Checks if `BLOCKS_CTR` is zero.
	test BLOCKS_CTR, BLOCKS_CTR
	// Functional Utility: Jumps to `.L_end` if `BLOCKS_CTR` is zero (no more blocks).
	jnz .L_begin
	.align 32
	jmp	.L_end
	.align 32
.L_begin:

	/*
	 * Do first block
	 * rounds: 0,2,4,6,8
	 */
	.set j, 0
	.rept 5 /* Functional Utility: Executes 5 pairs of rounds (0-9). */
		RR	j /* Functional Utility: Executes two SHA-1 rounds and rotates state. */
		.set j, j+2
	.endr

	/*
	 * rounds:
	 * 10,12,14,16,18
	 * 20,22,24,26,28
	 * 30,32,34,36,38
	 * 40,42,44,46,48
	 * 50,52,54,56,58
	 */
	.rept 25 /* Functional Utility: Executes 25 pairs of rounds (10-59). */
		RR	j
		.set j, j+2
	.endr

	/* Update Counter */
	sub $1, BLOCKS_CTR /* Functional Utility: Decrements the blocks counter. */
	/* Move to the next block only if needed*/
	// Functional Utility: Conditionally advances `BUFFER_PTR` for the next block.
	ADD_IF_GE BUFFER_PTR, BLOCKS_CTR, 4, 128
	/*
	 * rounds
	 * 60,62,64,66,68
	 * 70,72,74,76,78
	 */
	.rept 10 /* Functional Utility: Executes 10 pairs of rounds (60-79). */
		RR	j
		.set j, j+2
	.endr

	// Functional Utility: Updates the main hash state registers in memory (`HASH_PTR`) with the results from the first block.
	UPDATE_HASH	(HASH_PTR), A
	UPDATE_HASH	4(HASH_PTR), TB
	UPDATE_HASH	8(HASH_PTR), C
	UPDATE_HASH	12(HASH_PTR), D
	UPDATE_HASH	16(HASH_PTR), E

	// Functional Utility: Checks if `BLOCKS_CTR` is zero.
	test	BLOCKS_CTR, BLOCKS_CTR
	// Functional Utility: Jumps to `.L_loop` if `BLOCKS_CTR` is zero (finished).
	jz	.L_loop

	mov	TB, B /* Functional Utility: Moves the saved B from `TB` back to `B` for the second block. */

	/* Process second block */
	/*
	 * rounds
	 *  0+80, 2+80, 4+80, 6+80, 8+80
	 * 10+80,12+80,14+80,16+80,18+80
	 */

	.set j, 0
	.rept 10 /* Functional Utility: Executes 10 pairs of rounds (80-99) for the second block in the pipeline. */
		RR	j+80
		.set j, j+2
	.endr

	/*
	 * rounds
	 * 20+80,22+80,24+80,26+80,28+80
	 * 30+80,32+80,34+80,36+80,38+80
	 */
	.rept 10 /* Functional Utility: Executes 10 pairs of rounds (100-119) for the second block. */
		RR	j+80
		.set j, j+2
	.endr

	/*
	 * rounds
	 * 40+80,42+80,44+80,46+80,48+80
	 * 50+80,52+80,54+80,56+80,58+80
	 */
	.rept 10 /* Functional Utility: Executes 10 pairs of rounds (120-139) for the second block. */
		RR	j+80
		.set j, j+2
	.endr

	/* update counter */
	sub     $1, BLOCKS_CTR /* Functional Utility: Decrements the blocks counter. */
	/* Move to the next block only if needed*/
	// Functional Utility: Conditionally advances `BUFFER_PTR2` for the next block.
	ADD_IF_GE BUFFER_PTR2, BLOCKS_CTR, 4, 128

	/*
	 * rounds
	 * 60+80,62+80,64+80,66+80,68+80
	 * 70+80,72+80,74+80,76+80,78+80
	 */
	.rept 10 /* Functional Utility: Executes 10 pairs of rounds (140-159) for the second block. */
		RR	j+80
		.set j, j+2
	.endr

	// Functional Utility: Updates the main hash state registers in memory (`HASH_PTR`) with the results from the second block.
	UPDATE_HASH	(HASH_PTR), A
	UPDATE_HASH	4(HASH_PTR), TB
	UPDATE_HASH	8(HASH_PTR), C
	UPDATE_HASH	12(HASH_PTR), D
	UPDATE_HASH	16(HASH_PTR), E

	/* Reset state for AVX2 reg permutation */
	// Functional Utility: Performs a series of register moves to effectively rotate/permute the SHA-1 state
	// (A, B, C, D, E) for the next iteration of the pipeline. This prepares the registers for the next block processing.
	mov	A, TA
	mov	TB, A
	mov	C, TB
	mov	E, C
	mov	D, B
	mov	TA, D

	REGALLOC /* Functional Utility: Re-initializes register aliases, which now reflect the permuted state. */

	// Functional Utility: Swaps `WK_BUF` and `PRECALC_BUF` to prepare for the next pipeline stage.
	xchg	WK_BUF, PRECALC_BUF

	jmp	.L_loop /* Functional Utility: Jumps back to the start of the loop for the next block. */

	.align 32
.L_end: /* Functional Role: End of the pipelined processing loop. */

.endm
/*
 * macro implements SHA-1 function's body for several 64-byte blocks
 * param: function's name
 */
/**
 * @brief Macro to define the `sha1_transform_avx2` function with necessary boilerplate.
 * @details This macro wraps the `SHA1_PIPELINED_MAIN_BODY` with standard
 * function prologue and epilogue, including saving/restoring registers,
 * stack alignment, and clearing the upper bits of YMM registers (`vzeroupper`)
 * to avoid performance penalties.
 * @param name The name of the function to define (e.g., `sha1_transform_avx2`).
 * Functional Utility: Defines the AVX2-optimized SHA-1 transform function, handling function call conventions.
 */
.macro SHA1_VECTOR_ASM  name
	SYM_FUNC_START(
ame)

	// Functional Utility: Saves callee-saved registers as per x86_64 ABI.
	push	%rbx
	push	%r12
	push	%r13
	push	%r14
	push	%r15

	RESERVE_STACK  = (W_SIZE*4 + 8+24) /* Functional Role: Calculates the stack space required for local variables and alignment. */

	/* Align stack */
	// Functional Utility: Saves the current base pointer.
	push	%rbp
	// Functional Utility: Sets the base pointer to the current stack pointer.
	mov	%rsp, %rbp
	// Functional Utility: Aligns the stack pointer to a 32-byte boundary (for AVX2).
	and	$~(0x20-1), %rsp
	// Functional Utility: Allocates stack space for local variables.
	sub	$RESERVE_STACK, %rsp

	// Functional Utility: Clears the upper 128-bits of all YMM registers. This is crucial for performance when mixing AVX and SSE instructions.
	avx2_zeroupper

	/* Setup initial values */
	// Functional Utility: Moves argument registers to internal aliases.
	mov	CTX, HASH_PTR
	mov	BUF, BUFFER_PTR

	mov	BUF, BUFFER_PTR2
	mov	CNT, BLOCKS_CTR

	// Functional Utility: Loads the byte-swap shuffle control mask into a YMM register.
	xmm_mov	BSWAP_SHUFB_CTL(%rip), YMM_SHUFB_BSWAP

	SHA1_PIPELINED_MAIN_BODY /* Functional Utility: Executes the main pipelined SHA-1 processing logic. */

	// Functional Utility: Clears the upper 128-bits of all YMM registers before function return.
	avx2_zeroupper

	// Functional Utility: Restores the stack pointer.
	mov	%rbp, %rsp
	// Functional Utility: Restores the base pointer.
	pop	%rbp

	// Functional Utility: Restores callee-saved registers.
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbx

	RET /* Functional Utility: Returns from the function. */

	SYM_FUNC_END(
ame)
.endm

.section .rodata

#define K1 0x5a827999 /* Functional Role: SHA-1 round constant for rounds 0-19. */
#define K2 0x6ed9eba1 /* Functional Role: SHA-1 round constant for rounds 20-39. */
#define K3 0x8f1bbcdc /* Functional Role: SHA-1 round constant for rounds 40-59. */
#define K4 0xca62c1d6 /* Functional Role: SHA-1 round constant for rounds 60-79. */

.align 128
/**
 * @brief Array of SHA-1 round constants for AVX2 processing.
 * @details This data section stores the SHA-1 round constants (K1, K2, K3, K4)
 * replicated across 128-byte aligned memory to facilitate efficient loading
 * into YMM registers (as 8 DWORDS). This allows the `vpaddd` instruction
 * to add the correct constant value to multiple message words simultaneously.
 * Functional Role: Provides aligned SHA-1 round constants for vectorized addition.
 */
K_XMM_AR:
	.long K1, K1, K1, K1
	.long K1, K1, K1, K1
	.long K2, K2, K2, K2
	.long K2, K2, K2, K2
	.long K3, K3, K3, K3
	.long K3, K3, K3, K3
	.long K4, K4, K4, K4
	.long K4, K4, K4, K4

/**
 * @brief Shuffle control mask for byte-swapping with AVX2 `vpshufb` instruction.
 * @details This data section provides the constant mask required by the AVX2
 * `vpshufb` instruction to perform byte-swapping on 32-bit words within a
 * 256-bit YMM register. This is essential for converting input data from
 * little-endian to big-endian, as required by the SHA-1 algorithm.
 * Functional Role: Provides a shuffle mask for efficient byte-swapping of DWORDS.
 */
BSWAP_SHUFB_CTL:
	.long 0x00010203
	.long 0x04050607
	.long 0x08090a0b
	.long 0x0c0d0e0f
	.long 0x00010203
	.long 0x04050607
	.long 0x08090a0b
	.long 0x0c0d0e0f
.text

SHA1_VECTOR_ASM     sha1_transform_avx2
