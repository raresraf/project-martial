/**
 * @file sha1-armv7-neon.S
 * @brief ARMv7 NEON-accelerated SHA-1 transform function for high performance.
 * @details This assembly file implements a highly optimized SHA1 compression
 * function utilizing ARMv7 NEON SIMD instructions. The design focuses on
 * maximizing instruction-level parallelism and hiding memory latency through
 * aggressive pipelining and interleaved message schedule calculations,
 * specifically targeting the capabilities of NEON-enabled ARM processors.
 *
 * **SIMD Strategy:**
 * The core optimization leverages NEON's 128-bit vector registers (q0-q15)
 * to process multiple 32-bit words concurrently. While the overall process
 * handles one 64-byte data block at a time, the computations within each
 * of the 80 SHA1 rounds are heavily vectorized, particularly for message
 * schedule generation and the arithmetic logic.
 *
 * **Pipelining and Interleaving:**
 * A deep pipeline is implemented where the calculation of future message
 * schedule words (`WPRECALC_*` macros) is interleaved with the execution of
 * the current SHA1 rounds (`_R_F*` macros). This strategy effectively hides
 * the latency of memory accesses and complex message word computations,
 * ensuring that the NEON and ARM scalar execution units are continuously
 * fed with data and operations.
 *
 * **Optimization Techniques:**
 * - **Vectorization**: NEON registers are extensively used for parallel
 *   operations on message words (W[i]), including XORs, shifts, and additions.
 * - **Instruction Scheduling**: Instructions are carefully ordered to prevent
 *   pipeline stalls and fully utilize the dual-issue capabilities of modern
 *   ARM cores.
 * - **Macro-driven structure**: Complex logic is encapsulated in macros
 *   (`_R_F*`, `WPRECALC_*`) to manage the pipelining and unrolling effectively.
 * - **Register Allocation**: Scalar registers are efficiently allocated for
 *   SHA1 state variables (A, B, C, D, E) and temporary calculations, while
 *   NEON registers handle the message schedule and vectorized intermediate results.
 */
/* SPDX-License-Identifier: GPL-2.0-or-later */
/* sha1-armv7-neon.S - ARM/NEON accelerated SHA-1 transform function
 *
 * Copyright © 2013-2014 Jussi Kivilinna <jussi.kivilinna@iki.fi>
 */

#include <linux/linkage.h>
#include <asm/assembler.h>

@
@ This file provides sha1_transform_neon, a high-performance implementation of
@ the SHA1 compression function for ARMv7 and later architectures, accelerated
@ using NEON SIMD instructions.
@
@ SIMD Strategy:
@ The core optimization is the use of NEON's 128-bit vector registers to
@ perform operations on multiple 32-bit words simultaneously. While the overall
@ structure processes one 64-byte block at a time, the computations within
@ each of the 80 rounds are heavily vectorized.
@
@ Optimization Techniques:
@ - Vectorization: The SHA1 state (A,B,C,D,E) is processed using scalar ARM
@   instructions, but the message schedule (W words) and the round logic
@   leverage NEON registers (q0-q15) for parallel computations.
@ - Instruction Scheduling & Pipelining: The round computations (_R_F* macros)
@   are deeply pipelined. The calculation of the message schedule for future
@   rounds (WPRECALC_* macros) is interleaved with the execution of current
@   rounds to hide memory and computational latencies, maximizing the use of
@   the ARM/NEON execution units.
@

.syntax unified
.fpu neon

.text


/* Context structure offsets */

#define state_h0 0
#define state_h1 4
#define state_h2 8
#define state_h3 12
#define state_h4 16


/* SHA-1 round constants */

#define K1  0x5A827999
#define K2  0x6ED9EBA1
#define K3  0x8F1BBCDC
#define K4  0xCA62C1D6
.align 4
.LK_VEC:
.LK1:	.long K1, K1, K1, K1 @ NEON vector constant for rounds 0-19
.LK2:	.long K2, K2, K2, K2 @ NEON vector constant for rounds 20-39
.LK3:	.long K3, K3, K3, K3 @ NEON vector constant for rounds 40-59
.LK4:	.long K4, K4, K4, K4 @ NEON vector constant for rounds 60-79


/* Register Aliases for clarity */
@ Core registers for state and pointers
#define RSTATE r0 @ Functional Role: Pointer to the SHA1 context structure (sha1_state).
#define RDATA r1  @ Functional Role: Pointer to the input data block (64-byte aligned).
#define RNBLKS r2 @ Functional Role: Number of 64-byte blocks remaining to process.
#define ROLDSTACK r3 @ Functional Role: Stores the stack pointer from function entry.
#define RWK lr    @ Functional Role: Link Register (used as a temporary register for message word pointer).

@ SHA-1 state variables (A, B, C, D, E)
#define _a r4 @ Functional Role: SHA1 working variable A.
#define _b r5 @ Functional Role: SHA1 working variable B.
#define _c r6 @ Functional Role: SHA1 working variable C.
#define _d r7 @ Functional Role: SHA1 working variable D.
#define _e r8 @ Functional Role: SHA1 working variable E.

@ Temporary scalar registers
#define RT0 r9  @ Functional Role: Temporary general-purpose register 0.
#define RT1 r10 @ Functional Role: Temporary general-purpose register 1.
#define RT2 r11 @ Functional Role: Temporary general-purpose register 2.
#define RT3 r12 @ Functional Role: Temporary general-purpose register 3.

@ NEON registers for message schedule words (W)
#define W0 q0 @ Functional Role: NEON vector register for message word W0 (containing W[0]-W[3]).
#define W1 q7 @ Functional Role: NEON vector register for message word W1 (containing W[4]-W[7]).
#define W2 q2 @ Functional Role: NEON vector register for message word W2 (containing W[8]-W[11]).
#define W3 q3 @ Functional Role: NEON vector register for message word W3 (containing W[12]-W[15]).
#define W4 q4 @ Functional Role: NEON vector register for message word W4.
#define W5 q6 @ Functional Role: NEON vector register for message word W5.
#define W6 q5 @ Functional Role: NEON vector register for message word W6.
#define W7 q1 @ Functional Role: NEON vector register for message word W7.

@ Temporary NEON registers
#define tmp0 q8  @ Functional Role: Temporary NEON vector register 0.
#define tmp1 q9  @ Functional Role: Temporary NEON vector register 1.
#define tmp2 q10 @ Functional Role: Temporary NEON vector register 2.
#define tmp3 q11 @ Functional Role: Temporary NEON vector register 3.

@ NEON registers for round constants (K)
#define qK1 q12 @ Functional Role: NEON vector register holding K1 constant (for rounds 0-19).
#define qK2 q13 @ Functional Role: NEON vector register holding K2 constant (for rounds 20-39).
#define qK3 q14 @ Functional Role: NEON vector register holding K3 constant (for rounds 40-59).
#define qK4 q15 @ Functional Role: NEON vector register holding K4 constant (for rounds 60-79).

#ifdef CONFIG_CPU_BIG_ENDIAN
#define ARM_LE(code...)
#else
@ Macro to handle byte-swapping on Little Endian systems
@ Functional Utility: Provides conditional compilation for byte-swapping
@ instructions, ensuring that message words are correctly ordered for SHA1
@ processing on little-endian ARM systems.
#define ARM_LE(code...)		code
#endif

/*
 * SHA-1 Round Function Macros (_R_F1, _R_F2, _R_F3, _R_F4)
 *
 * These macros implement the four distinct logical functions used in the
 * 80 rounds of SHA-1. They are designed to be deeply pipelined, where
 * message schedule calculations (`pre1`, `pre2`, `pre3` parameters) are
 * interleaved with the arithmetic operations of the current round.
 */

#define WK_offs(i) (((i) & 15) * 4) @ Functional Role: Calculates the byte offset for accessing W[i] on the stack.

@ Round function for rounds 0-19: (B&C) | (~B&D)
#define _R_F1(a,b,c,d,e,i,pre1,pre2,pre3,i16,
	      W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	ldr RT3, [sp, WK_offs(i)]; /* Functional Utility: Load message word W[i] from stack */ 
		pre1(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); /* Functional Utility: Pipelined message schedule calculation (stage 1) */ 
	bic RT0, d, b; /* Functional Utility: Calculate ~B & D */ 
	add e, e, a, ror #(32 - 5); /* Functional Utility: Update E with rotated A (A = ROL(A, 5)) */ 
	and RT1, c, b; /* Functional Utility: Calculate B & C */ 
		pre2(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); /* Functional Utility: Pipelined message schedule calculation (stage 2) */ 
	add RT0, RT0, RT3; /* Functional Utility: Add W[i] */ 
	add e, e, RT1; /* Functional Utility: Add B & C to E */ 
	ror b, #(32 - 30); /* Functional Utility: Rotate B (B = ROL(B, 30)) */ 
		pre3(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); /* Functional Utility: Pipelined message schedule calculation (stage 3) */ 
	add e, e, RT0; /* Functional Utility: Add (~B & D) + W[i] to E */

@ Round function for rounds 20-39 and 60-79: B ^ C ^ D
#define _R_F2(a,b,c,d,e,i,pre1,pre2,pre3,i16,
	      W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	ldr RT3, [sp, WK_offs(i)]; /* Functional Utility: Load message word W[i] from stack */ 
		pre1(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); /* Functional Utility: Pipelined message schedule calculation (stage 1) */ 
	eor RT0, d, b; /* Functional Utility: Calculate B ^ D */ 
	add e, e, a, ror #(32 - 5); /* Functional Utility: Update E with rotated A (A = ROL(A, 5)) */ 
	eor RT0, RT0, c; /* Functional Utility: Calculate B ^ D ^ C (B XOR C XOR D) */ 
		pre2(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); /* Functional Utility: Pipelined message schedule calculation (stage 2) */ 
	add e, e, RT3; /* Functional Utility: Add W[i] */ 
	ror b, #(32 - 30); /* Functional Utility: Rotate B (B = ROL(B, 30)) */ 
		pre3(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); /* Functional Utility: Pipelined message schedule calculation (stage 3) */ 
	add e, e, RT0; /* Functional Utility: Add B XOR C XOR D to E */ 

@ Round function for rounds 40-59: (B&C) | (B&D) | (C&D) (Majority Function)
#define _R_F3(a,b,c,d,e,i,pre1,pre2,pre3,i16,
	      W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	ldr RT3, [sp, WK_offs(i)]; /* Functional Utility: Load message word W[i] from stack */ 
		pre1(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); /* Functional Utility: Pipelined message schedule calculation (stage 1) */ 
	eor RT0, b, c; /* Functional Utility: Calculate B ^ C (intermediate for majority) */ 
	and RT1, b, c; /* Functional Utility: Calculate B & C (part of majority) */ 
	add e, e, a, ror #(32 - 5); /* Functional Utility: Update E with rotated A (A = ROL(A, 5)) */ 
		pre2(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); /* Functional Utility: Pipelined message schedule calculation (stage 2) */ 
	and RT0, RT0, d; /* Functional Utility: Calculate (B^C) & D = (B&D)|(C&D) (part of majority) */ 
	add RT1, RT1, RT3; /* Functional Utility: Add W[i] to B & C */ 
	add e, e, RT0; /* Functional Utility: Add (B^C)&D to E */ 
	ror b, #(32 - 30); /* Functional Utility: Rotate B (B = ROL(B, 30)) */ 
		pre3(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); /* Functional Utility: Pipelined message schedule calculation (stage 3) */ 
	add e, e, RT1; /* Functional Utility: Add (B&C) + W[i] to E */

#define _R_F4(a,b,c,d,e,i,pre1,pre2,pre3,i16,
	      W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	_R_F2(a,b,c,d,e,i,pre1,pre2,pre3,i16,
	      W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) @ Functional Utility: F4 is identical to F2, implementing B XOR C XOR D.

#define _R(a,b,c,d,e,f,i,pre1,pre2,pre3,i16,
           W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	_R_##f(a,b,c,d,e,i,pre1,pre2,pre3,i16,
	       W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) @ Functional Utility: Dispatches to the correct round function macro (_R_F1, _R_F2, _R_F3, _R_F4) based on the round type 'f'.

#define R(a,b,c,d,e,f,i) 
	_R_##f(a,b,c,d,e,i,dummy,dummy,dummy,i16,
	       W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) @ Functional Utility: Simplified round macro for non-pipelined sections, using dummy placeholders for message schedule pre-calculations.

#define dummy(...) @ Functional Role: Placeholder macro used in R() where no pipelined message schedule calculation is needed.


/*
 * Message Schedule Expansion Macros (WPRECALC_*)
 *
 * These macros implement the SHA-1 message schedule W[i], where for i >= 16,
 * W[i] = ROL((W[i-16] ^ W[i-14] ^ W[i-8] ^ W[i-3]), 1).
 * The logic is broken into fine-grained sub-macros to be interleaved with
 * the round computations for maximum instruction-level parallelism using NEON.
 */

/********* Precalc macros for rounds 0-15 *************************************/
@ Functional Utility: This macro orchestrates the loading of the initial 16 message words (W[0]-W[15])
@ from the input data (RDATA) into NEON vector registers. It also handles necessary byte-swapping
@ for little-endian systems and adds the SHA1 round constant K1 (curK) before storing these
@ pre-processed words onto the stack. This prepares the message schedule for the first rounds.
#define W_PRECALC_00_15() 
	add       RWK, sp, #(WK_offs(0));			
	
	vld1.32   {W0, W7}, [RDATA]!;				
 ARM_LE(vrev32.8  W0, W0;	)	/* big => little */	
	vld1.32   {W6, W5}, [RDATA]!;				
	vadd.u32  tmp0, W0, curK;				
 ARM_LE(vrev32.8  W7, W7;	)	/* big => little */	
 ARM_LE(vrev32.8  W6, W6;	)	/* big => little */	
	vadd.u32  tmp1, W7, curK;				
 ARM_LE(vrev32.8  W5, W5;	)	/* big => little */	
	vadd.u32  tmp2, W6, curK;				
	vst1.32   {tmp0, tmp1}, [RWK]!;				
	vadd.u32  tmp3, W5, curK;				
	vst1.32   {tmp2, tmp3}, [RWK];				

@ Functional Role: Individual stages of loading initial message words from RDATA into NEON registers W0, W7.
#define WPRECALC_00_15_0(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vld1.32   {W0, W7}, [RDATA]!;				

@ Functional Role: Sets RWK to the base address on stack for storing pre-calculated W words.
#define WPRECALC_00_15_1(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	add       RWK, sp, #(WK_offs(0));			

@ Functional Role: Byte-swaps W0 for little-endian systems (big => little).
#define WPRECALC_00_15_2(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
 ARM_LE(vrev32.8  W0, W0;	)	/* big => little */	

@ Functional Role: Loads next set of initial message words into NEON registers W6, W5.
#define WPRECALC_00_15_3(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vld1.32   {W6, W5}, [RDATA]!;				

@ Functional Role: Adds round constant curK to W0 and stores in tmp0.
#define WPRECALC_00_15_4(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vadd.u32  tmp0, W0, curK;				

@ Functional Role: Byte-swaps W7 for little-endian systems.
#define WPRECALC_00_15_5(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
 ARM_LE(vrev32.8  W7, W7;	)	/* big => little */	

@ Functional Role: Byte-swaps W6 for little-endian systems.
#define WPRECALC_00_15_6(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
 ARM_LE(vrev32.8  W6, W6;	)	/* big => little */	

@ Functional Role: Adds round constant curK to W7 and stores in tmp1.
#define WPRECALC_00_15_7(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vadd.u32  tmp1, W7, curK;				

@ Functional Role: Byte-swaps W5 for little-endian systems.
#define WPRECALC_00_15_8(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
 ARM_LE(vrev32.8  W5, W5;	)	/* big => little */	

@ Functional Role: Adds round constant curK to W6 and stores in tmp2.
#define WPRECALC_00_15_9(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vadd.u32  tmp2, W6, curK;				

@ Functional Role: Stores pre-processed W0 and W7 (from tmp0, tmp1) onto the stack.
#define WPRECALC_00_15_10(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vst1.32   {tmp0, tmp1}, [RWK]!;				

@ Functional Role: Adds round constant curK to W5 and stores in tmp3.
#define WPRECALC_00_15_11(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vadd.u32  tmp3, W5, curK;				

@ Functional Role: Stores pre-processed W6 and W5 (from tmp2, tmp3) onto the stack.
#define WPRECALC_00_15_12(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vst1.32   {tmp2, tmp3}, [RWK];				


/********* Precalc macros for rounds 16-31 ************************************/
@ Functional Utility: These macros collectively implement the SHA-1 message schedule expansion
@ for W[16] through W[31]. They compute W[i] = ROL((W[i-16]^W[i-14]^W[i-8]^W[i-3]), 1)
@ using NEON vector operations (veor, vshl, vshr, vext, vorr) across multiple pipeline stages.
@
@ Parameters:
@   i: Current round index.
@   W: Current message word NEON register (e.g., W0, W1).
@   W_m04 to W_m28: NEON registers holding previously computed message words at specified offsets.
#define WPRECALC_16_31_0(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	veor      tmp0, tmp0; /* Functional Role: Clear tmp0 (set to all zeros) */ 
	vext.8    W, W_m16, W_m12, #8; /* Functional Role: Extract and combine bytes from W_m16 and W_m12 to form part of W. This is part of the efficient ROL(X,1) calculation. */ 

#define WPRECALC_16_31_1(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	add       RWK, sp, #(WK_offs(i));	/* Functional Role: Set RWK to the stack address for storing W[i]. */ 
	vext.8    tmp0, W_m04, tmp0, #4;	/* Functional Role: Extract and combine bytes from W_m04 and tmp0. */ 

#define WPRECALC_16_31_2(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	veor      tmp0, tmp0, W_m16;	/* Functional Role: XOR tmp0 with W_m16. */ 
	veor.32   W, W, W_m08;	/* Functional Role: XOR W with W_m08 (part of message schedule formula). */ 

#define WPRECALC_16_31_3(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	veor      tmp1, tmp1; /* Functional Role: Clear tmp1. */ 
	veor      W, W, tmp0;	/* Functional Role: XOR W with tmp0. */ 

#define WPRECALC_16_31_4(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vshl.u32  tmp0, W, #1;	/* Functional Role: Shift W left by 1 bit (part of ROL(W,1)). */ 

#define WPRECALC_16_31_5(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vext.8    tmp1, W_m04, tmp1, #(16-12);	/* Functional Role: Extract and combine bytes from W_m04 and tmp1. */ 
	vshr.u32  W, W, #31;	/* Functional Role: Shift W right by 31 bits (other part of ROL(W,1)). */ 

#define WPRECALC_16_31_6(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vorr      tmp0, tmp0, W;	/* Functional Role: Combine shifted parts to complete ROL(W,1). */ 
	vshr.u32  W, tmp1, #30;	/* Functional Role: Shift tmp1 right by 30 bits. */ 

#define WPRECALC_16_31_7(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vshl.u32  tmp1, tmp1, #2;	/* Functional Role: Shift tmp1 left by 2 bits. */ 

#define WPRECALC_16_31_8(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	veor      tmp0, tmp0, W;	/* Functional Role: XOR tmp0 with W. */ 

#define WPRECALC_16_31_9(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	veor      W, tmp0, tmp1;	/* Functional Role: Final XOR to compute W[i] after ROL. */ 

#define WPRECALC_16_31_10(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vadd.u32  tmp0, W, curK;	/* Functional Role: Add round constant curK to the computed W[i]. */ 

#define WPRECALC_16_31_11(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vst1.32   {tmp0}, [RWK]; /* Functional Role: Store the final pre-calculated W[i] onto the stack. */


/********* Precalc macros for rounds 32-79 ************************************/
@ Functional Utility: These macros continue the pipelined SHA-1 message schedule expansion
@ for W[32] through W[79]. They implement the same formula, W[i] = ROL((W[i-16]^W[i-14]^W[i-8]^W[i-3]), 1),
@ by orchestrating NEON vector operations across multiple pipeline stages. This ensures
@ that message words are prepared efficiently and on time for the subsequent SHA1 rounds.
#define WPRECALC_32_79_0(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	veor W, W_m28; /* Functional Role: XOR W with W_m28 (W[i-16] component) */ 

#define WPRECALC_32_79_1(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vext.8 tmp0, W_m08, W_m04, #8; /* Functional Role: Extract and combine bytes from W_m08 and W_m04 (W[i-14] component) */ 

#define WPRECALC_32_79_2(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	veor W, W_m16; /* Functional Role: XOR W with W_m16 (W[i-8] component) */ 

#define WPRECALC_32_79_3(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	veor W, tmp0; /* Functional Role: XOR W with tmp0 (W[i-3] component) */ 

#define WPRECALC_32_79_4(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	add RWK, sp, #(WK_offs(i&~3)); /* Functional Role: Set RWK to the stack address for storing W[i]. */ 

#define WPRECALC_32_79_5(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vshl.u32 tmp1, W, #2; /* Functional Role: Shift W left by 2 bits (part of ROL(W,1)) */ 

#define WPRECALC_32_79_6(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vshr.u32 tmp0, W, #30; /* Functional Role: Shift W right by 30 bits (other part of ROL(W,1)) */ 

#define WPRECALC_32_79_7(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vorr W, tmp0, tmp1; /* Functional Role: Combine shifted parts to complete ROL(W,1) */ 

#define WPRECALC_32_79_8(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vadd.u32 tmp0, W, curK; /* Functional Role: Add round constant curK to the computed W[i] */ 

#define WPRECALC_32_79_9(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) 
	vst1.32 {tmp0}, [RWK]; /* Functional Role: Store the final pre-calculated W[i] onto the stack. */


/**
 * @brief SHA1 compression function accelerated by ARM NEON.
 * @details This function performs the core SHA1 message compression using ARM NEON
 * SIMD instructions. It processes `nblks` number of 64-byte data blocks, updating
 * the SHA1 hash context. The function employs a deeply pipelined approach to
 * interleave message schedule computations with round calculations for maximum throughput.
 *
 * @param r0 (RSTATE): Pointer to the `sha1_state` context structure (A,B,C,D,E).
 * @param r1 (RDATA): Pointer to the input data block (64-byte aligned).
 * @param r2 (RNBLKS): Number of 64-byte blocks to process.
 *
 * @pre `RSTATE` points to a valid `sha1_state` structure with the current hash values.
 * @pre `RDATA` points to the input data, aligned to 64 bytes.
 * @pre `RNBLKS` is a non-negative integer representing the number of blocks.
 * @post The `sha1_state` structure pointed to by `RSTATE` is updated with the hash
 *       of all processed blocks.
 */
.align 3
ENTRY(sha1_transform_neon)
  @ Pre-condition: Check if there are any blocks to process. If RNBLKS is 0, skip the main processing loop.
  cmp RNBLKS, #0;
  beq .Ldo_nothing;

  @ Prologue: Save registers, align stack, and load initial state and constants.
  @ Functional Utility: Save callee-saved general-purpose registers (r4-r12) and the Link Register (lr)
  @ onto the stack, adhering to the ARM Procedure Call Standard (APCS).
  push {r4-r12, lr};
  @ Functional Utility: Load the address of the NEON round constants table (.LK_VEC) into RT3.
  adr RT3, .LK_VEC;
  @ Functional Utility: Store the initial stack pointer (sp) for later restoration in the epilogue.
  mov ROLDSTACK, sp;
  @ Functional Utility: Calculate and align the new stack pointer (sp) to ensure 16-byte alignment,
  @ which is beneficial for NEON operations. This allocates space for message words W[0]-W[15].
  sub RT0, sp, #(16*4);
  and RT0, #(~(16-1));
  mov sp, RT0;
  @ Functional Utility: Load SHA1 round constants K1 and K2 into NEON registers qK1 and qK2 from memory pointed to by RT3.
  vld1.32 {qK1-qK2}, [RT3]!; /* Load K1,K2 into NEON registers */
  @ Functional Utility: Load the initial five SHA1 hash state variables (A, B, C, D, E)
  @ from the `sha1_state` context (RSTATE) into scalar registers _a through _e.
  ldm RSTATE, {_a-_e}; /* Load A,B,C,D,E from context */
  @ Functional Utility: Load SHA1 round constants K3 and K4 into NEON registers qK3 and qK4.
  vld1.32 {qK3-qK4}, [RT3]; /* Load K3,K4 */

#undef curK
#define curK qK1 @ Functional Role: Alias for the current NEON round constant (qK1 for rounds 0-19).
  @ Functional Utility: Pre-calculate and store the first 16 message words (W[0]-W[15])
  @ for the first block. This is done before entering the main round loop to prime the pipeline.
  W_PRECALC_00_15();

.Loop:
@
@ Main processing loop for a single 64-byte block.
@ Block Logic: This loop iteratively processes a 64-byte block of input data
@ through all 80 SHA1 rounds. It uses a deeply pipelined approach, where the
@ message schedule calculation for future rounds is performed concurrently
@ with the current round's hash computation. This maximizes instruction-level
@ parallelism, leveraging both ARM scalar and NEON vector units.
@ Invariant: At the beginning of each iteration, _a through _e hold the current
@ SHA1 hash state, RDATA points to the start of the current 64-byte data block,
@ and RNBLKS indicates remaining blocks.
@

  /* Rounds 0-15 (using F1 function) / Pre-calculate W[16-31] */
  @ Functional Utility: These macro calls execute 4 rounds of SHA1 using the F1 logical function.
  @ Simultaneously, they interleave stages of the WPRECALC_16_31 message schedule calculation
  @ to prepare W words for upcoming rounds, hiding computation latency.
  _R( _a, _b, _c, _d, _e, F1,  0,
      WPRECALC_16_31_0, WPRECALC_16_31_1, WPRECALC_16_31_2, 16,
      W4, W5, W6, W7, W0, _, _, _ );
  _R( _e, _a, _b, _c, _d, F1,  1,
      WPRECALC_16_31_3, WPRECALC_16_31_4, WPRECALC_16_31_5, 16,
      W4, W5, W6, W7, W0, _, _, _ );
  _R( _d, _e, _a, _b, _c, F1,  2,
      WPRECALC_16_31_6, WPRECALC_16_31_7, WPRECALC_16_31_8, 16,
      W4, W5, W6, W7, W0, _, _, _ );
  _R( _c, _d, _e, _a, _b, F1,  3,
      WPRECALC_16_31_9, WPRECALC_16_31_10,WPRECALC_16_31_11,16,
      W4, W5, W6, W7, W0, _, _, _ );

#undef curK
#define curK qK2 @ Functional Role: Alias for the current NEON round constant (qK2 for rounds 20-39).
  _R( _b, _c, _d, _e, _a, F1,  4,
      WPRECALC_16_31_0, WPRECALC_16_31_1, WPRECALC_16_31_2, 20,
      W3, W4, W5, W6, W7, _, _, _ );
  _R( _a, _b, _c, _d, _e, F1,  5,
      WPRECALC_16_31_3, WPRECALC_16_31_4, WPRECALC_16_31_5, 20,
      W3, W4, W5, W6, W7, _, _, _ );
  _R( _e, _a, _b, _c, _d, F1,  6,
      WPRECALC_16_31_6, WPRECALC_16_31_7, WPRECALC_16_31_8, 20,
      W3, W4, W5, W6, W7, _, _, _ );
  _R( _d, _e, _a, _b, _c, F1,  7,
      WPRECALC_16_31_9, WPRECALC_16_31_10,WPRECALC_16_31_11,20,
      W3, W4, W5, W6, W7, _, _, _ );

  _R( _c, _d, _e, _a, _b, F1,  8,
      WPRECALC_16_31_0, WPRECALC_16_31_1, WPRECALC_16_31_2, 24,
      W2, W3, W4, W5, W6, _, _, _ );
  _R( _b, _c, _d, _e, _a, F1,  9,
      WPRECALC_16_31_3, WPRECALC_16_31_4, WPRECALC_16_31_5, 24,
      W2, W3, W4, W5, W6, _, _, _ );
  _R( _a, _b, _c, _d, _e, F1, 10,
      WPRECALC_16_31_6, WPRECALC_16_31_7, WPRECALC_16_31_8, 24,
      W2, W3, W4, W5, W6, _, _, _ );
  _R( _e, _a, _b, _c, _d, F1, 11,
      WPRECALC_16_31_9, WPRECALC_16_31_10,WPRECALC_16_31_11,24,
      W2, W3, W4, W5, W6, _, _, _ );

  _R( _d, _e, _a, _b, _c, F1, 12,
      WPRECALC_16_31_0, WPRECALC_16_31_1, WPRECALC_16_31_2, 28,
      W1, W2, W3, W4, W5, _, _, _ );
  _R( _c, _d, _e, _a, _b, F1, 13,
      WPRECALC_16_31_3, WPRECALC_16_31_4, WPRECALC_16_31_5, 28,
      W1, W2, W3, W4, W5, _, _, _ );
  _R( _b, _c, _d, _e, _a, F1, 14,
      WPRECALC_16_31_6, WPRECALC_16_31_7, WPRECALC_16_31_8, 28,
      W1, W2, W3, W4, W5, _, _, _ );
  _R( _a, _b, _c, _d, _e, F1, 15,
      WPRECALC_16_31_9, WPRECALC_16_31_10,WPRECALC_16_31_11,28,
      W1, W2, W3, W4, W5, _, _, _ );

  /* Rounds 16-63 (Logical Functions F1, F2, F3) / Pre-calculate W[32-79] */
  @ Functional Utility: These macro calls execute 4 rounds of SHA1. The logical function used for these
  @ rounds depends on the specific round (F1 for 16-19, F2 for 20-39, F3 for 40-59, F4 for 60-79).
  @ Concurrently, they interleave stages of the WPRECALC_32_79 message schedule calculation
  @ to prepare W words for upcoming rounds, maintaining pipeline efficiency.
  _R( _e, _a, _b, _c, _d, F1, 16,
      WPRECALC_32_79_0, WPRECALC_32_79_1, WPRECALC_32_79_2, 32,
      W0, W1, W2, W3, W4, W5, W6, W7);
  _R( _d, _e, _a, _b, _c, F1, 17,
      WPRECALC_32_79_3, WPRECALC_32_79_4, WPRECALC_32_79_5, 32,
      W0, W1, W2, W3, W4, W5, W6, W7);
  _R( _c, _d, _e, _a, _b, F1, 18,
      WPRECALC_32_79_6, dummy,            WPRECALC_32_79_7, 32,
      W0, W1, W2, W3, W4, W5, W6, W7);
  _R( _b, _c, _d, _e, _a, F1, 19,
      WPRECALC_32_79_8, dummy,            WPRECALC_32_79_9, 32,
      W0, W1, W2, W3, W4, W5, W6, W7);

  _R( _a, _b, _c, _d, _e, F2, 20,
      WPRECALC_32_79_0, WPRECALC_32_79_1, WPRECALC_32_79_2, 36,
      W7, W0, W1, W2, W3, W4, W5, W6);
  _R( _e, _a, _b, _c, _d, F2, 21,
      WPRECALC_32_79_3, WPRECALC_32_79_4, WPRECALC_32_79_5, 36,
      W7, W0, W1, W2, W3, W4, W5, W6);
  _R( _d, _e, _a, _b, _c, F2, 22,
      WPRECALC_32_79_6, dummy,            WPRECALC_32_79_7, 36,
      W7, W0, W1, W2, W3, W4, W5, W6);
  _R( _c, _d, _e, _a, _b, F2, 23,
      WPRECALC_32_79_8, dummy,            WPRECALC_32_79_9, 36,
      W7, W0, W1, W2, W3, W4, W5, W6);

#undef curK
#define curK qK3 @ Functional Role: Alias for the current NEON round constant (qK3 for rounds 40-59).
  _R( _b, _c, _d, _e, _a, F2, 24,
      WPRECALC_32_79_0, WPRECALC_32_79_1, WPRECALC_32_79_2, 40,
      W6, W7, W0, W1, W2, W3, W4, W5);
  _R( _a, _b, _c, _d, _e, F2, 25,
      WPRECALC_32_79_3, WPRECALC_32_79_4, WPRECALC_32_79_5, 40,
      W6, W7, W0, W1, W2, W3, W4, W5);
  _R( _e, _a, _b, _c, _d, F2, 26,
      WPRECALC_32_79_6, dummy,            WPRECALC_32_79_7, 40,
      W6, W7, W0, W1, W2, W3, W4, W5);
  _R( _d, _e, _a, _b, _c, F2, 27,
      WPRECALC_32_79_8, dummy,            WPRECALC_32_79_9, 40,
      W6, W7, W0, W1, W2, W3, W4, W5);

  _R( _c, _d, _e, _a, _b, F2, 28,
      WPRECALC_32_79_0, WPRECALC_32_79_1, WPRECALC_32_79_2, 44,
      W5, W6, W7, W0, W1, W2, W3, W4);
  _R( _b, _c, _d, _e, _a, F2, 29,
      WPRECALC_32_79_3, WPRECALC_32_79_4, WPRECALC_32_79_5, 44,
      W5, W6, W7, W0, W1, W2, W3, W4);
  _R( _a, _b, _c, _d, _e, F2, 30,
      WPRECALC_32_79_6, dummy,            WPRECALC_32_79_7, 44,
      W5, W6, W7, W0, W1, W2, W3, W4);
  _R( _e, _a, _b, _c, _d, F2, 31,
      WPRECALC_32_79_8, dummy,            WPRECALC_32_79_9, 44,
      W5, W6, W7, W0, W1, W2, W3, W4);

  _R( _d, _e, _a, _b, _c, F2, 32,
      WPRECALC_32_79_0, WPRECALC_32_79_1, WPRECALC_32_79_2, 48,
      W4, W5, W6, W7, W0, W1, W2, W3);
  _R( _c, _d, _e, _a, _b, F2, 33,
      WPRECALC_32_79_3, WPRECALC_32_79_4, WPRECALC_32_79_5, 48,
      W4, W5, W6, W7, W0, W1, W2, W3);
  _R( _b, _c, _d, _e, _a, F2, 34,
      WPRECALC_32_79_6, dummy,            WPRECALC_32_79_7, 48,
      W4, W5, W6, W7, W0, W1, W2, W3);
  _R( _a, _b, _c, _d, _e, F2, 35,
      WPRECALC_32_79_8, dummy,            WPRECALC_32_79_9, 48,
      W4, W5, W6, W7, W0, W1, W2, W3);

  _R( _e, _a, _b, _c, _d, F2, 36,
      WPRECALC_32_79_0, WPRECALC_32_79_1, WPRECALC_32_79_2, 52,
      W3, W4, W5, W6, W7, W0, W1, W2);
  _R( _d, _e, _a, _b, _c, F2, 37,
      WPRECALC_32_79_3, WPRECALC_32_79_4, WPRECALC_32_79_5, 52,
      W3, W4, W5, W6, W7, W0, W1, W2);
  _R( _c, _d, _e, _a, _b, F2, 38,
      WPRECALC_32_79_6, dummy,            WPRECALC_32_79_7, 52,
      W3, W4, W5, W6, W7, W0, W1, W2);
  _R( _b, _c, _d, _e, _a, F2, 39,
      WPRECALC_32_79_8, dummy,            WPRECALC_32_79_9, 52,
      W3, W4, W5, W6, W7, W0, W1, W2);

  _R( _a, _b, _c, _d, _e, F3, 40,
      WPRECALC_32_79_0, WPRECALC_32_79_1, WPRECALC_32_79_2, 56,
      W2, W3, W4, W5, W6, W7, W0, W1);
  _R( _e, _a, _b, _c, _d, F3, 41,
      WPRECALC_32_79_3, WPRECALC_32_79_4, WPRECALC_32_79_5, 56,
      W2, W3, W4, W5, W6, W7, W0, W1);
  _R( _d, _e, _a, _b, _c, F3, 42,
      WPRECALC_32_79_6, dummy,            WPRECALC_32_79_7, 56,
      W2, W3, W4, W5, W6, W7, W0, W1);
  _R( _c, _d, _e, _a, _b, F3, 43,
      WPRECALC_32_79_8, dummy,            WPRECALC_32_79_9, 56,
      W2, W3, W4, W5, W6, W7, W0, W1);

#undef curK
#define curK qK4 @ Functional Role: Alias for the current NEON round constant (qK4 for rounds 60-79).
  _R( _b, _c, _d, _e, _a, F3, 44,
      WPRECALC_32_79_0, WPRECALC_32_79_1, WPRECALC_32_79_2, 60,
      W1, W2, W3, W4, W5, W6, W7, W0);
  _R( _a, _b, _c, _d, _e, F3, 45,
      WPRECALC_32_79_3, WPRECALC_32_79_4, WPRECALC_32_79_5, 60,
      W1, W2, W3, W4, W5, W6, W7, W0);
  _R( _e, _a, _b, _c, _d, F3, 46,
      WPRECALC_32_79_6, dummy,            WPRECALC_32_79_7, 60,
      W1, W2, W3, W4, W5, W6, W7, W0);
  _R( _d, _e, _a, _b, _c, F3, 47,
      WPRECALC_32_79_8, dummy,            WPRECALC_32_79_9, 60,
      W1, W2, W3, W4, W5, W6, W7, W0);

  _R( _c, _d, _e, _a, _b, F3, 48,
      WPRECALC_32_79_0, WPRECALC_32_79_1, WPRECALC_32_79_2, 64,
      W0, W1, W2, W3, W4, W5, W6, W7);
  _R( _b, _c, _d, _e, _a, F3, 49,
      WPRECALC_32_79_3, WPRECALC_32_79_4, WPRECALC_32_79_5, 64,
      W0, W1, W2, W3, W4, W5, W6, W7);
  _R( _a, _b, _c, _d, _e, F3, 50,
      WPRECALC_32_79_6, dummy,            WPRECALC_32_79_7, 64,
      W0, W1, W2, W3, W4, W5, W6, W7);
  _R( _e, _a, _b, _c, _d, F3, 51,
      WPRECALC_32_79_8, dummy,            WPRECALC_32_79_9, 64,
      W0, W1, W2, W3, W4, W5, W6, W7);

  _R( _d, _e, _a, _b, _c, F3, 52,
      WPRECALC_32_79_0, WPRECALC_32_79_1, WPRECALC_32_79_2, 68,
      W7, W0, W1, W2, W3, W4, W5, W6);
  _R( _c, _d, _e, _a, _b, F3, 53,
      WPRECALC_32_79_3, WPRECALC_32_79_4, WPRECALC_32_79_5, 68,
      W7, W0, W1, W2, W3, W4, W5, W6);
  _R( _b, _c, _d, _e, _a, F3, 54,
      WPRECALC_32_79_6, dummy,            WPRECALC_32_79_7, 68,
      W7, W0, W1, W2, W3, W4, W5, W6);
  _R( _a, _b, _c, _d, _e, F3, 55,
      WPRECALC_32_79_8, dummy,            WPRECALC_32_79_9, 68,
      W7, W0, W1, W2, W3, W4, W5, W6);

  _R( _e, _a, _b, _c, _d, F3, 56,
      WPRECALC_32_79_0, WPRECALC_32_79_1, WPRECALC_32_79_2, 72,
      W6, W7, W0, W1, W2, W3, W4, W5);
  _R( _d, _e, _a, _b, _c, F3, 57,
      WPRECALC_32_79_3, WPRECALC_32_79_4, WPRECALC_32_79_5, 72,
      W6, W7, W0, W1, W2, W3, W4, W5);
  _R( _c, _d, _e, _a, _b, F3, 58,
      WPRECALC_32_79_6, dummy,            WPRECALC_32_79_7, 72,
      W6, W7, W0, W1, W2, W3, W4, W5);
  _R( _b, _c, _d, _e, _a, F3, 59,
      WPRECALC_32_79_8, dummy,            WPRECALC_32_79_9, 72,
      W6, W7, W0, W1, W2, W3, W4, W5);

  @ Functional Utility: Decrement the block counter.
  subs RNBLKS, #1;

  _R( _a, _b, _c, _d, _e, F4, 60,
      WPRECALC_32_79_0, WPRECALC_32_79_1, WPRECALC_32_79_2, 76,
      W5, W6, W7, W0, W1, W2, W3, W4);
  _R( _e, _a, _b, _c, _d, F4, 61,
      WPRECALC_32_79_3, WPRECALC_32_79_4, WPRECALC_32_79_5, 76,
      W5, W6, W7, W0, W1, W2, W3, W4);
  _R( _d, _e, _a, _b, _c, F4, 62,
      WPRECALC_32_79_6, dummy,            WPRECALC_32_79_7, 76,
      W5, W6, W7, W0, W1, W2, W3, W4);
  _R( _c, _d, _e, _a, _b, F4, 63,
      WPRECALC_32_79_8, dummy,            WPRECALC_32_79_9, 76,
      W5, W6, W7, W0, W1, W2, W3, W4);

  @ Pre-condition: If RNBLKS is zero, branch to the .Lend section, indicating all blocks are processed.
  beq .Lend;

  /* Pipelined Epilogue: Process rounds 64-79 of the current block while
   * pre-calculating W[0]-W[15] for the *next* block. */
#undef curK
#define curK qK1 @ Functional Role: Alias for the NEON round constant (qK1) for the next block's initial rounds.
  _R( _b, _c, _d, _e, _a, F4, 64,
      WPRECALC_00_15_0, dummy, dummy, _, _, _, _, _, _, _, _, _ );
  _R( _a, _b, _c, _d, _e, F4, 65,
      WPRECALC_00_15_1, dummy, dummy, _, _, _, _, _, _, _, _, _ );
  _R( _e, _a, _b, _c, _d, F4, 66,
      WPRECALC_00_15_2, dummy, dummy, _, _, _, _, _, _, _, _, _ );
  _R( _d, _e, _a, _b, _c, F4, 67,
      WPRECALC_00_15_3, dummy, dummy, _, _, _, _, _, _, _, _, _ );

  _R( _c, _d, _e, _a, _b, F4, 68,
      dummy,            dummy, dummy, _, _, _, _, _, _, _, _, _ );
  _R( _b, _c, _d, _e, _a, F4, 69,
      dummy,            dummy, dummy, _, _, _, _, _, _, _, _, _ );
  _R( _a, _b, _c, _d, _e, F4, 70,
      WPRECALC_00_15_4, dummy, dummy, _, _, _, _, _, _, _, _, _ );
  _R( _e, _a, _b, _c, _d, F4, 71,
      WPRECALC_00_15_5, dummy, dummy, _, _, _, _, _, _, _, _, _ );

  _R( _d, _e, _a, _b, _c, F4, 72,
      dummy,            dummy, dummy, _, _, _, _, _, _, _, _, _ );
  _R( _c, _d, _e, _a, _b, F4, 73,
      dummy,            dummy, dummy, _, _, _, _, _, _, _, _, _ );
  _R( _b, _c, _d, _e, _a, F4, 74,
      WPRECALC_00_15_6, dummy, dummy, _, _, _, _, _, _, _, _, _ );
  _R( _a, _b, _c, _d, _e, F4, 75,
      WPRECALC_00_15_7, dummy, dummy, _, _, _, _, _, _, _, _, _ );

  _R( _e, _a, _b, _c, _d, F4, 76,
      WPRECALC_00_15_8, dummy, dummy, _, _, _, _, _, _, _, _, _ );
  _R( _d, _e, _a, _b, _c, F4, 77,
      WPRECALC_00_15_9, dummy, dummy, _, _, _, _, _, _, _, _, _ );
  _R( _c, _d, _e, _a, _b, F4, 78,
      WPRECALC_00_15_10, dummy, dummy, _, _, _, _, _, _, _, _, _ );
  _R( _b, _c, _d, _e, _a, F4, 79,
      WPRECALC_00_15_11, dummy, WPRECALC_00_15_12, _, _, _, _, _, _, _, _, _ );

  /* Functional Utility: Update the chaining variables in the SHA1 state context
   * with the results of the completed block. This involves adding the current
   * working variables (_a-_e) to the initial hash values loaded at the start
   * of the block.
   */
  ldm RSTATE, {RT0-RT3}; @ Functional Utility: Load the first four chaining variables from RSTATE into RT0-RT3.
  add _a, RT0; @ Functional Utility: Add original H0 to _a.
  ldr RT0, [RSTATE, #state_h4]; @ Functional Utility: Load the fifth chaining variable (H4) into RT0.
  add _b, RT1; @ Functional Utility: Add original H1 to _b.
  add _c, RT2; @ Functional Utility: Add original H2 to _c.
  add _d, RT3; @ Functional Utility: Add original H3 to _d.
  add _e, RT0; @ Functional Utility: Add original H4 to _e.
  stm RSTATE, {_a-_e}; @ Functional Utility: Store the updated chaining variables back into RSTATE.

  b .Loop; @ Functional Utility: Branch back to the .Loop to process the next 64-byte block.

.Lend:
  /* Functional Utility: Final rounds for the last block (not pipelined).
   * This section processes the remaining 16 rounds (64-79) of the last block
   * if the main loop was exited early due to only one block remaining.
   * These rounds are executed without pipelining the message schedule pre-calculation.
   */
  R( _b, _c, _d, _e, _a, F4, 64 );
  R( _a, _b, _c, _d, _e, F4, 65 );
  R( _e, _a, _b, _c, _d, F4, 66 );
  R( _d, _e, _a, _b, _c, F4, 67 );
  R( _c, _d, _e, _a, _b, F4, 68 );
  R( _b, _c, _d, _e, _a, F4, 69 );
  R( _a, _b, _c, _d, _e, F4, 70 );
  R( _e, _a, _b, _c, _d, F4, 71 );
  R( _d, _e, _a, _b, _c, F4, 72 );
  R( _c, _d, _e, _a, _b, F4, 73 );
  R( _b, _c, _d, _e, _a, F4, 74 );
  R( _a, _b, _c, _d, _e, F4, 75 );
  R( _e, _a, _b, _c, _d, F4, 76 );
  R( _d, _e, _a, _b, _c, F4, 77 );
  R( _c, _d, _e, _a, _b, F4, 78 );
  R( _b, _c, _d, _e, _a, F4, 79 );

  mov sp, ROLDSTACK; @ Functional Utility: Restore the stack pointer to its value at function entry.

  /* Functional Utility: Final state update for the last block.
   * This section performs the final update of the SHA1 chaining variables
   * after all 80 rounds of the last block are completed.
   */
  ldm RSTATE, {RT0-RT3}; @ Functional Utility: Load the first four chaining variables from RSTATE into RT0-RT3.
  add _a, RT0; @ Functional Utility: Add original H0 to _a.
  ldr RT0, [RSTATE, #state_h4]; @ Functional Utility: Load the fifth chaining variable (H4) into RT0.
  add _b, RT1; @ Functional Utility: Add original H1 to _b.
  add _c, RT2; @ Functional Utility: Add original H2 to _c.
  add _d, RT3; @ Functional Utility: Add original H3 to _d.
  add _e, RT0; @ Functional Utility: Add original H4 to _e.
  stm RSTATE, {_a-_e}; @ Functional Utility: Store the updated chaining variables back into RSTATE.

  @ Epilogue: Restore saved registers and return.
  @ Functional Utility: Pop saved registers (r4-r12) and the Program Counter (pc) from the stack.
  @ This restores the caller's context and returns control to the calling function.
  pop {r4-r12, pc};

.Ldo_nothing:
  @ Functional Utility: If RNBLKS was initially 0, this path is taken.
  @ It simply returns from the function without performing any SHA1 processing.
  bx lr
ENDPROC(sha1_transform_neon)
