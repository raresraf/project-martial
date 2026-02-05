#define __ARM_ARCH__ __LINUX_ARM_ARCH__
@ SPDX-License-Identifier: GPL-2.0

@ This code is taken from the OpenSSL project but the author (Andy Polyakov)
@ has relicensed it under the GPLv2. Therefore this program is free software;
@ you can redistribute it and/or modify it under the terms of the GNU General
@ Public License version 2 as published by the Free Software Foundation.
@
@ The original headers, including the original license headers, are
@ included below for completeness.

@ ====================================================================
@ Written by Andy Polyakov <appro@fy.chalmers.se> for the OpenSSL
@ project. The module is, however, dual licensed under OpenSSL and
@ CRYPTOGAMS licenses depending on where you obtain it. For further
@ details see https://www.openssl.org/~appro/cryptogams/.
@ ====================================================================

@ sha1_block procedure for ARMv4.
@
@ January 2007.

@ Size/performance trade-off
@ ====================================================================
@ impl		size in bytes	comp cycles[*]	measured performance
@ ====================================================================
@ thumb		304		3212		4420
@ armv4-small	392/+29%	1958/+64%	2250/+96%
@ armv4-compact	740/+89%	1552/+26%	1840/+22%
@ armv4-large	1420/+92%	1307/+19%	1370/+34%[***]
@ full unroll	~5100/+260%	~1260/+4%	~1300/+5%
@ ====================================================================
@ thumb		= same as 'small' but in Thumb instructions[**] and
@		  with recurring code in two private functions;
@ small		= detached Xload/update, loops are folded;
@ compact	= detached Xload/update, 5x unroll;
@ large		= interleaved Xload/update, 5x unroll;
@ full unroll	= interleaved Xload/update, full unroll, estimated[!];
@
@ [*]	Manually counted instructions in "grand" loop body. Measured
@	performance is affected by prologue and epilogue overhead,
@	i-cache availability, branch penalties, etc.
@ [**]	While each Thumb instruction is twice smaller, they are not as
@	diverse as ARM ones: e.g., there are only two arithmetic
@	instructions with 3 arguments, no [fixed] rotate, addressing
@	modes are limited. As result it takes more instructions to do
@	the same job in Thumb, therefore the code is never twice as
@	small and always slower.
@ [***]	which is also ~35% better than compiler generated code. Dual-
@	issue Cortex A8 core was measured to process input block in
@	~990 cycles.

@ August 2010.
@
@ Rescheduling for dual-issue pipeline resulted in 13% improvement on
@ Cortex A8 core and in absolute terms ~870 cycles per input block
@ [or 13.6 cycles per byte].

@ February 2011.
@
@ Profiler-assisted and platform-specific optimization resulted in 10%
@ improvement on Cortex A8 core and 12.2 cycles per byte.

#include <linux/linkage.h>
@
@ This file provides sha1_block_data_order, a performance-optimized
@ implementation of the SHA1 compression function for ARMv4 and later
@ architectures.
@
@ Optimization Strategies:
@ - 5x Loop Unrolling: The main loop processes 5 rounds of the SHA1
@   algorithm per iteration to reduce loop overhead.
@ - Instruction Scheduling: Instructions are carefully ordered to maximize
@   throughput on dual-issue ARM cores (e.g., Cortex-A8) by avoiding
@   pipeline stalls.
@ - Interleaved Load/Update: The message schedule calculation (W[i]) is
@   interleaved with the round updates to hide memory latency.
@ - Register-based State: The five SHA1 state variables (A, B, C, D, E)
@   are held in registers (r3-r7) for fast access.
@

.text

.align	2
@ sha1_block_data_order (struct sha1_state *digest, const u8 *data,
@                        int blocks)
@ r0: pointer to sha1_state structure (A,B,C,D,E)
@ r1: pointer to input data
@ r2: number of 64-byte blocks to process
ENTRY(sha1_block_data_order)
	@ Function Prologue: Save callee-saved registers.
	stmdb	sp!,{r4-r12,lr}
	add	r2,r1,r2,lsl#6	@ r2 to point at the end of r1
	@ Load initial hash state (A,B,C,D,E) into registers.
	ldmia	r0,{r3,r4,r5,r6,r7}
.Lloop:
	@ This is the main loop that processes each 64-byte input block.
	ldr	r8,.LK_00_19
	mov	r14,sp
	sub	sp,sp,#15*4
	@ Rotate state variables B, C, D in preparation for the rounds.
	@ B, C, D are rotated left by 30 bits (ror #2).
	mov	r5,r5,ror#30
	mov	r6,r6,ror#30
	mov	r7,r7,ror#30		@ [6]
.L_00_15:
@
@ SHA1 Rounds 0-15
@
@ This section implements the first 16 rounds of the SHA1 compression.
@ The logical function F(B,C,D) = (B & C) | (~B & D) is used.
@ The round constant K_00_19 (0x5a827999) is loaded into r8.
@ Message words W[0]-W[15] are loaded directly from the input buffer.
@ The loop is unrolled 5 times.
@
#if __ARM_ARCH__<7
	ldrb	r10,[r1,#2]
	ldrb	r9,[r1,#3]
	ldrb	r11,[r1,#1]
	add	r7,r8,r7,ror#2			@ E+=K_00_19
	ldrb	r12,[r1],#4
	orr	r9,r9,r10,lsl#8
	eor	r10,r5,r6			@ F_xx_xx
	orr	r9,r9,r11,lsl#16
	add	r7,r7,r3,ror#27			@ E+=ROR(A,27)
	orr	r9,r9,r12,lsl#24
#else
	@ Load W[i] from memory.
	ldr	r9,[r1],#4			@ handles unaligned
	add	r7,r8,r7,ror#2			@ E+=K_00_19
	eor	r10,r5,r6			@ F_xx_xx
	add	r7,r7,r3,ror#27			@ E+=ROR(A,27)
#ifdef __ARMEL__
	@ Byte swap for little-endian systems.
	rev	r9,r9
#endif
#endif
	and	r10,r4,r10,ror#2
	add	r7,r7,r9			@ E+=X[i]
	eor	r10,r10,r6,ror#2		@ F_00_19(B,C,D)
	str	r9,[r14,#-4]!
	add	r7,r7,r10			@ E+=F_00_19(B,C,D)
#if __ARM_ARCH__<7
	ldrb	r10,[r1,#2]
	ldrb	r9,[r1,#3]
	ldrb	r11,[r1,#1]
	add	r6,r8,r6,ror#2			@ E+=K_00_19
	ldrb	r12,[r1],#4
	orr	r9,r9,r10,lsl#8
	eor	r10,r4,r5			@ F_xx_xx
	orr	r9,r9,r11,lsl#16
	add	r6,r6,r7,ror#27			@ E+=ROR(A,27)
	orr	r9,r9,r12,lsl#24
#else
	ldr	r9,[r1],#4
	add	r6,r8,r6,ror#2
	eor	r10,r4,r5
	add	r6,r6,r7,ror#27
#ifdef __ARMEL__
	rev	r9,r9
#endif
#endif
	and	r10,r3,r10,ror#2
	add	r6,r6,r9
	eor	r10,r10,r5,ror#2
	str	r9,[r14,#-4]!
	add	r6,r6,r10
#if __ARM_ARCH__<7
	ldrb	r10,[r1,#2]
	ldrb	r9,[r1,#3]
	ldrb	r11,[r1,#1]
	add	r5,r8,r5,ror#2			@ E+=K_00_19
	ldrb	r12,[r1],#4
	orr	r9,r9,r10,lsl#8
	eor	r10,r3,r4			@ F_xx_xx
	orr	r9,r9,r11,lsl#16
	add	r5,r5,r6,ror#27			@ E+=ROR(A,27)
	orr	r9,r9,r12,lsl#24
#else
	ldr	r9,[r1],#4
	add	r5,r8,r5,ror#2
	eor	r10,r3,r4
	add	r5,r5,r6,ror#27
#ifdef __ARMEL__
	rev	r9,r9
#endif
#endif
	and	r10,r7,r10,ror#2
	add	r5,r5,r9
	eor	r10,r10,r4,ror#2
	str	r9,[r14,#-4]!
	add	r5,r5,r10
#if __ARM_ARCH__<7
	ldrb	r10,[r1,#2]
	ldrb	r9,[r1,#3]
	ldrb	r11,[r1,#1]
	add	r4,r8,r4,ror#2			@ E+=K_00_19
	ldrb	r12,[r1],#4
	orr	r9,r9,r10,lsl#8
	eor	r10,r7,r3			@ F_xx_xx
	orr	r9,r9,r11,lsl#16
	add	r4,r4,r5,ror#27			@ E+=ROR(A,27)
	orr	r9,r9,r12,lsl#24
#else
	ldr	r9,[r1],#4
	add	r4,r8,r4,ror#2
	eor	r10,r7,r3
	add	r4,r4,r5,ror#27
#ifdef __ARMEL__
	rev	r9,r9
#endif
#endif
	and	r10,r6,r10,ror#2
	add	r4,r4,r9
	eor	r10,r10,r3,ror#2
	str	r9,[r14,#-4]!
	add	r4,r4,r10
#if __ARM_ARCH__<7
	ldrb	r10,[r1,#2]
	ldrb	r9,[r1,#3]
	ldrb	r11,[r1,#1]
	add	r3,r8,r3,ror#2			@ E+=K_00_19
	ldrb	r12,[r1],#4
	orr	r9,r9,r10,lsl#8
	eor	r10,r6,r7			@ F_xx_xx
	orr	r9,r9,r11,lsl#16
	add	r3,r3,r4,ror#27			@ E+=ROR(A,27)
	orr	r9,r9,r12,lsl#24
#else
	ldr	r9,[r1],#4
	add	r3,r8,r3,ror#2
	eor	r10,r6,r7
	add	r3,r3,r4,ror#27
#ifdef __ARMEL__
	rev	r9,r9
#endif
#endif
	and	r10,r5,r10,ror#2
	add	r3,r3,r9
	eor	r10,r10,r7,ror#2
	str	r9,[r14,#-4]!
	add	r3,r3,r10
	cmp	r14,sp
	bne	.L_00_15
	sub	sp,sp,#25*4

@
@ SHA1 Rounds 16-19
@
@ This part continues the first 20 rounds, but now interleaves the
@ message schedule calculation: W[i] = ROL((W[i-3]^W[i-8]^W[i-14]^W[i-16]), 1)
@
#if __ARM_ARCH__<7
	ldrb	r10,[r1,#2]
	ldrb	r9,[r1,#3]
	ldrb	r11,[r1,#1]
	add	r7,r8,r7,ror#2			@ E+=K_00_19
	ldrb	r12,[r1],#4
	orr	r9,r9,r10,lsl#8
	eor	r10,r5,r6			@ F_xx_xx
	orr	r9,r9,r11,lsl#16
	add	r7,r7,r3,ror#27			@ E+=ROR(A,27)
	orr	r9,r9,r12,lsl#24
#else
	ldr	r9,[r1],#4
	add	r7,r8,r7,ror#2
	eor	r10,r5,r6
	add	r7,r7,r3,ror#27
#ifdef __ARMEL__
	rev	r9,r9
#endif
#endif
	and	r10,r4,r10,ror#2
	add	r7,r7,r9			@ E+=X[i]
	eor	r10,r10,r6,ror#2		@ F_00_19(B,C,D)
	str	r9,[r14,#-4]!
	add	r7,r7,r10			@ E+=F_00_19(B,C,D)

	@ Message schedule calculation for W[16]
	ldr	r9,[r14,#15*4]
	ldr	r10,[r14,#13*4]
	ldr	r11,[r14,#7*4]
	add	r6,r8,r6,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4]
	eor	r9,r9,r10
	eor	r11,r11,r12
	eor	r10,r4,r5			@ F_xx_xx
	@ ROL(..., 1) is implemented as ROR(..., 31)
	mov	r9,r9,ror#31
	add	r6,r6,r7,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31
	str	r9,[r14,#-4]!
	and r10,r3,r10,ror#2
	add	r6,r6,r9			@ E+=X[i]
	eor	r10,r10,r5,ror#2		@ F_00_19(B,C,D)
	add	r6,r6,r10			@ E+=F_00_19(B,C,D)
	ldr	r9,[r14,#15*4]
	ldr	r10,[r14,#13*4]
	ldr	r11,[r14,#7*4]
	add	r5,r8,r5,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4]
	eor	r9,r9,r10
	eor	r11,r11,r12
	eor	r10,r3,r4			@ F_xx_xx
	mov	r9,r9,ror#31
	add	r5,r5,r6,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31
	str	r9,[r14,#-4]!
	and r10,r7,r10,ror#2
	add	r5,r5,r9			@ E+=X[i]
	eor	r10,r10,r4,ror#2		@ F_00_19(B,C,D)
	add	r5,r5,r10			@ E+=F_00_19(B,C,D)
	ldr	r9,[r14,#15*4]
	ldr	r10,[r14,#13*4]
	ldr	r11,[r14,#7*4]
	add	r4,r8,r4,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4]
	eor	r9,r9,r10
	eor	r11,r11,r12
	eor	r10,r7,r3			@ F_xx_xx
	mov	r9,r9,ror#31
	add	r4,r4,r5,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31
	str	r9,[r14,#-4]!
	and r10,r6,r10,ror#2
	add	r4,r4,r9			@ E+=X[i]
	eor	r10,r10,r3,ror#2		@ F_00_19(B,C,D)
	add	r4,r4,r10			@ E+=F_00_19(B,C,D)
	ldr	r9,[r14,#15*4]
	ldr	r10,[r14,#13*4]
	ldr	r11,[r14,#7*4]
	add	r3,r8,r3,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4]
	eor	r9,r9,r10
	eor	r11,r11,r12
	eor	r10,r6,r7			@ F_xx_xx
	mov	r9,r9,ror#31
	add	r3,r3,r4,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31
	str	r9,[r14,#-4]!
	and r10,r5,r10,ror#2
	add	r3,r3,r9			@ E+=X[i]
	eor	r10,r10,r7,ror#2		@ F_00_19(B,C,D)
	add	r3,r3,r10			@ E+=F_00_19(B,C,D)

	ldr	r8,.LK_20_39		@ [+15+16*4]
	cmn	sp,#0			@ [+3], clear carry to denote 20_39
.L_20_39_or_60_79:
@
@ SHA1 Rounds 20-39 and 60-79
@
@ This section implements rounds where the logical function is
@ F(B,C,D) = B ^ C ^ D.
@ The round constant K_20_39 (0x6ed9eba1) or K_60_79 (0xca62c1d6) is in r8.
@
	ldr	r9,[r14,#15*4]
	ldr	r10,[r14,#13*4]
	ldr	r11,[r14,#7*4]
	add	r7,r8,r7,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4]
	eor	r9,r9,r10
	eor	r11,r11,r12
	eor	r10,r5,r6			@ F_xx_xx
	mov	r9,r9,ror#31
	add	r7,r7,r3,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31
	str	r9,[r14,#-4]!
	eor r10,r4,r10,ror#2
	add	r7,r7,r9			@ E+=X[i]
	add	r7,r7,r10			@ E+=F_20_39(B,C,D)
	ldr	r9,[r14,#15*4]
	ldr	r10,[r14,#13*4]
	ldr	r11,[r14,#7*4]
	add	r6,r8,r6,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4]
	eor	r9,r9,r10
	eor	r11,r11,r12
	eor	r10,r4,r5			@ F_xx_xx
	mov	r9,r9,ror#31
	add	r6,r6,r7,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31
	str	r9,[r14,#-4]!
	eor r10,r3,r10,ror#2
	add	r6,r6,r9			@ E+=X[i]
	add	r6,r6,r10			@ E+=F_20_39(B,C,D)
	ldr	r9,[r14,#15*4]
	ldr	r10,[r14,#13*4]
	ldr	r11,[r14,#7*4]
	add	r5,r8,r5,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4]
	eor	r9,r9,r10
	eor	r11,r11,r12
	eor	r10,r3,r4			@ F_xx_xx
	mov	r9,r9,ror#31
	add	r5,r5,r6,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31
	str	r9,[r14,#-4]!
	eor r10,r7,r10,ror#2
	add	r5,r5,r9			@ E+=X[i]
	add	r5,r5,r10			@ E+=F_20_39(B,C,D)
	ldr	r9,[r14,#15*4]
	ldr	r10,[r14,#13*4]
	ldr	r11,[r14,#7*4]
	add	r4,r8,r4,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4]
	eor	r9,r9,r10
	eor	r11,r11,r12
	eor	r10,r7,r3			@ F_xx_xx
	mov	r9,r9,ror#31
	add	r4,r4,r5,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31
	str	r9,[r14,#-4]!
	eor r10,r6,r10,ror#2
	add	r4,r4,r9			@ E+=X[i]
	add	r4,r4,r10			@ E+=F_20_39(B,C,D)
	ldr	r9,[r14,#15*4]
	ldr	r10,[r14,#13*4]
	ldr	r11,[r14,#7*4]
	add	r3,r8,r3,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4]
	eor	r9,r9,r10
	eor	r11,r11,r12
	eor	r10,r6,r7			@ F_xx_xx
	mov	r9,r9,ror#31
	add	r3,r3,r4,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31
	str	r9,[r14,#-4]!
	eor r10,r5,r10,ror#2
	add	r3,r3,r9			@ E+=X[i]
	add	r3,r3,r10			@ E+=F_20_39(B,C,D)
 ARM(	teq	r14,sp		)	@ preserve carry
 THUMB(	mov	r11,sp		)
 THUMB(	teq	r14,r11		)	@ preserve carry
	bne	.L_20_39_or_60_79
	bcs	.L_done			@ Finished all 80 rounds, jump to epilogue.

	ldr	r8,.LK_40_59
	sub	sp,sp,#20*4		@ [+2]
.L_40_59:
@
@ SHA1 Rounds 40-59
@
@ This section implements rounds where the logical function is the majority
@ function: F(B,C,D) = (B & C) | (B & D) | (C & D).
@ The round constant K_40_59 (0x8f1bbcdc) is in r8.
@
	ldr	r9,[r14,#15*4]
	ldr	r10,[r14,#13*4]
	ldr	r11,[r14,#7*4]
	add	r7,r8,r7,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4]
	eor	r9,r9,r10
	eor	r11,r11,r12
	eor	r10,r5,r6			@ F_xx_xx
	mov	r9,r9,ror#31
	add	r7,r7,r3,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31
	str	r9,[r14,#-4]!
	and r10,r4,r10,ror#2
	and r11,r5,r6
	add	r7,r7,r9			@ E+=X[i]
	add	r7,r7,r10			@ E+=F_40_59(B,C,D)
	add	r7,r7,r11,ror#2
	ldr	r9,[r14,#15*4]
	ldr	r10,[r14,#13*4]
	ldr	r11,[r14,#7*4]
	add	r6,r8,r6,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4]
	eor	r9,r9,r10
	eor	r11,r11,r12
	eor	r10,r4,r5			@ F_xx_xx
	mov	r9,r9,ror#31
	add	r6,r6,r7,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31
	str	r9,[r14,#-4]!
	and r10,r3,r10,ror#2
	and r11,r4,r5
	add	r6,r6,r9			@ E+=X[i]
	add	r6,r6,r10			@ E+=F_40_59(B,C,D)
	add	r6,r6,r11,ror#2
	ldr	r9,[r14,#15*4]
	ldr	r10,[r14,#13*4]
	ldr	r11,[r14,#7*4]
	add	r5,r8,r5,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4]
	eor	r9,r9,r10
	eor	r11,r11,r12
	eor	r10,r3,r4			@ F_xx_xx
	mov	r9,r9,ror#31
	add	r5,r5,r6,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31
	str	r9,[r14,#-4]!
	and r10,r7,r10,ror#2
	and r11,r3,r4
	add	r5,r5,r9			@ E+=X[i]
	add	r5,r5,r10			@ E+=F_40_59(B,C,D)
	add	r5,r5,r11,ror#2
	ldr	r9,[r14,#15*4]
	ldr	r10,[r14,#13*4]
	ldr	r11,[r14,#7*4]
	add	r4,r8,r4,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4]
	eor	r9,r9,r10
	eor	r11,r11,r12
	eor	r10,r7,r3			@ F_xx_xx
	mov	r9,r9,ror#31
	add	r4,r4,r5,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31
	str	r9,[r14,#-4]!
	and r10,r6,r10,ror#2
	and r11,r7,r3
	add	r4,r4,r9			@ E+=X[i]
	add	r4,r4,r10			@ E+=F_40_59(B,C,D)
	add	r4,r4,r11,ror#2
	ldr	r9,[r14,#15*4]
	ldr	r10,[r14,#13*4]
	ldr	r11,[r14,#7*4]
	add	r3,r8,r3,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4]
	eor	r9,r9,r10
	eor	r11,r11,r12
	eor	r10,r6,r7			@ F_xx_xx
	mov	r9,r9,ror#31
	add	r3,r3,r4,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31
	str	r9,[r14,#-4]!
	and r10,r5,r10,ror#2
	and r11,r6,r7
	add	r3,r3,r9			@ E+=X[i]
	add	r3,r3,r10			@ E+=F_40_59(B,C,D)
	add	r3,r3,r11,ror#2
	cmp	r14,sp
	bne	.L_40_59

	ldr	r8,.LK_60_79
	sub	sp,sp,#20*4
	cmp	sp,#0			@ set carry to denote 60_79
	b	.L_20_39_or_60_79	@ Re-use logic for rounds 20-39
.L_done:
@
@ Epilogue
@
@ Update the final hash state and prepare for the next block or return.
@
	add	sp,sp,#80*4		@ "deallocate" stack frame
	@ Load original hash values.
	ldmia	r0,{r8,r9,r10,r11,r12}
	@ Add the results of this block's compression to the state.
	add	r3,r8,r3
	add	r4,r9,r4
	add	r5,r10,r5,ror#2
	add	r6,r11,r6,ror#2
	add	r7,r12,r7,ror#2
	@ Store the updated hash state.
	stmia	r0,{r3,r4,r5,r6,r7}
	@ Check if all blocks have been processed.
	teq	r1,r2
	bne	.Lloop			@ If not, loop to the next block.

	@ Restore saved registers and return.
	ldmia	sp!,{r4-r12,pc}
.align	2
.LK_00_19:	.word	0x5a827999
.LK_20_39:	.word	0x6ed9eba1
.LK_40_59:	.word	0x8f1bbcdc
.LK_60_79:	.word	0xca62c1d6
ENDPROC(sha1_block_data_order)
.asciz	"SHA1 block transform for ARMv4, CRYPTOGAMS by <appro@openssl.org>"
.align	2
