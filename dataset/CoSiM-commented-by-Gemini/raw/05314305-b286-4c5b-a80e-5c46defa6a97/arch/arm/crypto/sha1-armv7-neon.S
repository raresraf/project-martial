/* SPDX-License-Identifier: GPL-2.0-or-later */
/**
 * @file sha1-armv7-neon.S
 * @brief ARM/NEON accelerated SHA-1 transform function.
 *
 * @details This file provides sha1_transform_neon, a high-performance
 * implementation of the SHA-1 compression function for ARMv7 and later
 * architectures, accelerated using NEON SIMD instructions.
 *
 * SIMD Strategy:
 * The core optimization is the use of NEON's 128-bit vector registers to
 * perform operations on multiple 32-bit words simultaneously. While the overall
 * structure processes one 64-byte block at a time, the computations within
 * each of the 80 rounds are heavily vectorized.
 *
 * Optimization Techniques:
 * - Vectorization: The SHA-1 state (A,B,C,D,E) is processed using scalar ARM
 *   instructions, but the message schedule (W words) and the round logic
 *   leverage NEON registers (q0-q15) for parallel computations.
 * - Instruction Scheduling & Pipelining: The round computations (_R_F* macros)
 *   are deeply pipelined. The calculation of the message schedule for future
 *   rounds (WPRECALC_* macros) is interleaved with the execution of current
 *   rounds to hide memory and computational latencies, maximizing the use of
 *   the ARM/NEON execution units.
 * - Pipelined Main Loop: When processing multiple blocks, the epilogue of the
 *   current block (rounds 64-79) is overlapped with the prologue of the next
 *   block (calculating W[0]-W[15]), further enhancing throughput.
 */

#include <linux/linkage.h>
#include <asm/assembler.h>

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
.LK1:	.long K1, K1, K1, K1
.LK2:	.long K2, K2, K2, K2
.LK3:	.long K3, K3, K3, K3
.LK4:	.long K4, K4, K4, K4


/* Register Aliases for clarity */
@ Core registers for state and pointers
#define RSTATE r0
#define RDATA r1
#define RNBLKS r2
#define ROLDSTACK r3
#define RWK lr

@ SHA-1 state variables (A, B, C, D, E)
#define _a r4
#define _b r5
#define _c r6
#define _d r7
#define _e r8

@ Temporary scalar registers
#define RT0 r9
#define RT1 r10
#define RT2 r11
#define RT3 r12

@ NEON registers for message schedule words (W)
#define W0 q0
#define W1 q7
#define W2 q2
#define W3 q3
#define W4 q4
#define W5 q6
#define W6 q5
#define W7 q1

@ Temporary NEON registers
#define tmp0 q8
#define tmp1 q9
#define tmp2 q10
#define tmp3 q11

@ NEON registers for round constants (K)
#define qK1 q12
#define qK2 q13
#define qK3 q14
#define qK4 q15

#ifdef CONFIG_CPU_BIG_ENDIAN
#define ARM_LE(code...)
#else
@ Macro to handle byte-swapping on Little Endian systems
#define ARM_LE(code...)		code
#endif

/*
 * SHA-1 Round Function Macros (_R_F1, _R_F2, _R_F3, _R_F4)
 *
 * These macros implement the four distinct logical functions used in the
 * 80 rounds of SHA-1.
 *
 * Pipelining Strategy:
 * The 'pre1', 'pre2', and 'pre3' parameters are placeholders where
 * instructions for the message schedule calculation (from WPRECALC_* macros)
 * are inserted. This interleaving hides the latency of preparing the next
 * message word (W[i]) by executing it in parallel with the arithmetic
 * operations of the current round.
 */

#define WK_offs(i) (((i) & 15) * 4)

@ Round function for rounds 0-19: (b&c) | (~b&d)
#define _R_F1(a,b,c,d,e,i,pre1,pre2,pre3,i16,\
	      W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	ldr RT3, [sp, WK_offs(i)]; \
		pre1(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); \
	bic RT0, d, b; \
	add e, e, a, ror #(32 - 5); \
	and RT1, c, b; \
		pre2(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); \
	add RT0, RT0, RT3; \
	add e, e, RT1; \
	ror b, #(32 - 30); \
		pre3(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); \
	add e, e, RT0;

@ Round function for rounds 20-39 and 60-79: b ^ c ^ d
#define _R_F2(a,b,c,d,e,i,pre1,pre2,pre3,i16,\
	      W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	ldr RT3, [sp, WK_offs(i)]; \
		pre1(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); \
	eor RT0, d, b; \
	add e, e, a, ror #(32 - 5); \
	eor RT0, RT0, c; \
		pre2(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); \
	add e, e, RT3; \
	ror b, #(32 - 30); \
		pre3(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); \
	add e, e, RT0; \

@ Round function for rounds 40-59: (b&c) | (b&d) | (c&d)
#define _R_F3(a,b,c,d,e,i,pre1,pre2,pre3,i16,\
	      W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	ldr RT3, [sp, WK_offs(i)]; \
		pre1(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); \
	eor RT0, b, c; \
	and RT1, b, c; \
	add e, e, a, ror #(32 - 5); \
		pre2(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); \
	and RT0, RT0, d; \
	add RT1, RT1, RT3; \
	add e, e, RT0; \
	ror b, #(32 - 30); \
		pre3(i16,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28); \
	add e, e, RT1;

#define _R_F4(a,b,c,d,e,i,pre1,pre2,pre3,i16,\
	      W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	_R_F2(a,b,c,d,e,i,pre1,pre2,pre3,i16,\
	      W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28)

#define _R(a,b,c,d,e,f,i,pre1,pre2,pre3,i16,\
           W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	_R_##f(a,b,c,d,e,i,pre1,pre2,pre3,i16,\
	       W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28)

#define R(a,b,c,d,e,f,i) \
	_R_##f(a,b,c,d,e,i,dummy,dummy,dummy,i16,\
	       W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28)

#define dummy(...)


/*
 * Message Schedule Expansion Macros (WPRECALC_*)
 *
 * These macros implement the SHA-1 message schedule W[i], where for i >= 16,
 * W[i] = ROL((W[i-16] ^ W[i-14] ^ W[i-8] ^ W[i-3]), 1).
 * The logic is broken into fine-grained sub-macros to be interleaved with
 * the round computations for maximum instruction-level parallelism.
 */

/********* Precalc macros for rounds 0-15 *************************************/
@ This macro loads the first 16 words (W[0]-W[15]) from the input data,
@ performs necessary byte-swapping on little-endian machines, adds the
@ round constant K1, and stores the result on the stack.
#define W_PRECALC_00_15() \
	add       RWK, sp, #(WK_offs(0));			\
	\
	vld1.32   {W0, W7}, [RDATA]!;				\
 ARM_LE(vrev32.8  W0, W0;	)	/* big => little */	\
	vld1.32   {W6, W5}, [RDATA]!;				\
	vadd.u32  tmp0, W0, curK;				\
 ARM_LE(vrev32.8  W7, W7;	)	/* big => little */	\
 ARM_LE(vrev32.8  W6, W6;	)	/* big => little */	\
	vadd.u32  tmp1, W7, curK;				\
 ARM_LE(vrev32.8  W5, W5;	)	/* big => little */	\
	vadd.u32  tmp2, W6, curK;				\
	vst1.32   {tmp0, tmp1}, [RWK]!;				\
	vadd.u32  tmp3, W5, curK;				\
	vst1.32   {tmp2, tmp3}, [RWK];				\

#define WPRECALC_00_15_0(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vld1.32   {W0, W7}, [RDATA]!;				\

#define WPRECALC_00_15_1(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	add       RWK, sp, #(WK_offs(0));			\

#define WPRECALC_00_15_2(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
 ARM_LE(vrev32.8  W0, W0;	)	/* big => little */	\

#define WPRECALC_00_15_3(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vld1.32   {W6, W5}, [RDATA]!;				\

#define WPRECALC_00_15_4(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vadd.u32  tmp0, W0, curK;				\

#define WPRECALC_00_15_5(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
 ARM_LE(vrev32.8  W7, W7;	)	/* big => little */	\

#define WPRECALC_00_15_6(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
 ARM_LE(vrev32.8  W6, W6;	)	/* big => little */	\

#define WPRECALC_00_15_7(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vadd.u32  tmp1, W7, curK;				\

#define WPRECALC_00_15_8(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
 ARM_LE(vrev32.8  W5, W5;	)	/* big => little */	\

#define WPRECALC_00_15_9(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vadd.u32  tmp2, W6, curK;				\

#define WPRECALC_00_15_10(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vst1.32   {tmp0, tmp1}, [RWK]!;				\

#define WPRECALC_00_15_11(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vadd.u32  tmp3, W5, curK;				\

#define WPRECALC_00_15_12(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vst1.32   {tmp2, tmp3}, [RWK];				\


/********* Precalc macros for rounds 16-31 ************************************/
@ These macros calculate W[16]-W[31] using NEON vector operations (veor, vshl, etc.)
@ to implement the formula W[i] = ROL(W[i-16]^W[i-14]^W[i-8]^W[i-3], 1).
#define WPRECALC_16_31_0(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	veor      tmp0, tmp0;			\
	vext.8    W, W_m16, W_m12, #8;		\

#define WPRECALC_16_31_1(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	add       RWK, sp, #(WK_offs(i));	\
	vext.8    tmp0, W_m04, tmp0, #4;	\

#define WPRECALC_16_31_2(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	veor      tmp0, tmp0, W_m16;		\
	veor.32   W, W, W_m08;			\

#define WPRECALC_16_31_3(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	veor      tmp1, tmp1;			\
	veor      W, W, tmp0;			\

#define WPRECALC_16_31_4(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vshl.u32  tmp0, W, #1;			\

#define WPRECALC_16_31_5(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vext.8    tmp1, tmp1, W, #(16-12);	\
	vshr.u32  W, W, #31;			\

#define WPRECALC_16_31_6(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vorr      tmp0, tmp0, W;		\
	vshr.u32  W, tmp1, #30;			\

#define WPRECALC_16_31_7(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vshl.u32  tmp1, tmp1, #2;		\

#define WPRECALC_16_31_8(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	veor      tmp0, tmp0, W;		\

#define WPRECALC_16_31_9(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	veor      W, tmp0, tmp1;		\

#define WPRECALC_16_31_10(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vadd.u32  tmp0, W, curK;		\

#define WPRECALC_16_31_11(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vst1.32   {tmp0}, [RWK];


/********* Precalc macros for rounds 32-79 ************************************/
@ These macros calculate W[32]-W[79] using the same formula as above,
@ continuing the pipelined message schedule expansion.
#define WPRECALC_32_79_0(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	veor W, W_m28; \

#define WPRECALC_32_79_1(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vext.8 tmp0, W_m08, W_m04, #8; \

#define WPRECALC_32_79_2(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	veor W, W_m16; \

#define WPRECALC_32_79_3(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	veor W, tmp0; \

#define WPRECALC_32_79_4(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	add RWK, sp, #(WK_offs(i&~3)); \

#define WPRECALC_32_79_5(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vshl.u32 tmp1, W, #2; \

#define WPRECALC_32_79_6(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vshr.u32 tmp0, W, #30; \

#define WPRECALC_32_79_7(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vorr W, tmp0, tmp1; \

#define WPRECALC_32_79_8(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vadd.u32 tmp0, W, curK; \

#define WPRECALC_32_79_9(i,W,W_m04,W_m08,W_m12,W_m16,W_m20,W_m24,W_m28) \
	vst1.32 {tmp0}, [RWK];


/**
 * @brief Performs the SHA-1 compression function using ARM NEON instructions.
 * @param r0 (RSTATE): Pointer to the sha1_state context (H0-H4).
 * @param r1 (RDATA):  Pointer to the input data blocks.
 * @param r2 (RNBLKS): Number of 64-byte blocks to process.
 */
.align 3
ENTRY(sha1_transform_neon)
  cmp RNBLKS, #0;
  beq .Ldo_nothing;

  @ Function Prologue:
  @ - Save callee-saved registers.
  @ - Align stack to 16 bytes for NEON.
  @ - Load initial hash state (A,B,C,D,E) into scalar registers.
  @ - Load round constants (K1-K4) into NEON vector registers.
  push {r4-r12, lr};
  adr RT3, .LK_VEC;
  mov ROLDSTACK, sp;
  sub RT0, sp, #(16*4);
  and RT0, #(~(16-1));
  mov sp, RT0;
  vld1.32 {qK1-qK2}, [RT3]!; /* Load K1,K2 into NEON registers */
  ldm RSTATE, {_a-_e}; /* Load A,B,C,D,E from context */
  vld1.32 {qK3-qK4}, [RT3]; /* Load K3,K4 */

#undef curK
#define curK qK1
  @ Pre-calculate W[0]-W[15] for the first block.
  W_PRECALC_00_15();

.Loop:
/**
 * @brief Main processing loop for a single 64-byte block.
 *
 * @details The structure is a deep pipeline where the message schedule
 * calculation for future rounds is performed concurrently with the current
 * round's computation. This is achieved by passing the WPRECALC_* macros
 * as arguments to the _R_F* round macros, which interleave the instructions.
 * The state variables (a,b,c,d,e) are cyclically permuted in each round.
 */
  /* Rounds 0-15 (F1) / Pre-calculate W[16-31] */
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
#define curK qK2
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

  /* Rounds 16-63 / Pre-calculate W[32-79] */
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
#define curK qK3
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
#define curK qK4
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

  beq .Lend;

  /**
   * @brief Pipelined Main Loop and Epilogue.
   *
   * This section executes rounds 64-79 of the *current* block while
   * simultaneously pre-calculating the message schedule (W[0]-W[15]) for the
   * *next* block. This overlapping of the epilogue of one block with the
   * prologue of the next is a key performance optimization that maximizes
   * hardware utilization. After these rounds, the hash state for the just-
   * finished block is updated.
   */
#undef curK
#define curK qK1
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

  @ Block Update: Add the result of the compression to the intermediate hash state.
  ldm RSTATE, {RT0-RT3};
  add _a, RT0;
  ldr RT0, [RSTATE, #state_h4];
  add _b, RT1;
  add _c, RT2;
  add _d, RT3;
  add _e, RT0;
  stm RSTATE, {_a-_e};

  b .Loop;

.Lend:
  @ Final rounds for the last block (not pipelined).
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

  mov sp, ROLDSTACK;

  @ Final State Update: Add the results of the final block's compression
  @ to the chaining variables (H0-H4).
  ldm RSTATE, {RT0-RT3};
  add _a, RT0;
  ldr RT0, [RSTATE, #state_h4];
  add _b, RT1;
  add _c, RT2;
  add _d, RT3;
  add _e, RT0;
  stm RSTATE, {_a-_e};

  @ Function Epilogue: Restore saved registers from the stack and return.
  pop {r4-r12, pc};

.Ldo_nothing:
  bx lr
ENDPROC(sha1_transform_neon)
