/**
 * @file sha1-armv4-large.S
 * @brief Performance-optimized SHA1 compression function for ARMv4 architecture.
 * @details This assembly file provides a highly optimized implementation of the
 * SHA1 compression function specifically tailored for ARMv4 and later architectures.
 * It employs several performance optimization strategies crucial for cryptographic
 * primitives in embedded and high-performance computing contexts.
 *
 * Key Optimization Strategies:
 * - **5x Loop Unrolling**: The core SHA1 rounds are unrolled by a factor of five
 *   to significantly reduce loop overhead and improve instruction-level parallelism.
 * - **Instruction Scheduling**: Instructions are meticulously ordered to maximize
 *   pipeline utilization and minimize stalls on ARM processors, particularly
 *   benefiting dual-issue cores like Cortex-A8.
 * - **Interleaved Load/Update**: Message word loading and the subsequent SHA1
 *   state updates are interleaved. This technique aims to hide memory latency
 *   by performing computations while data is being fetched from memory.
 * - **Register-based State**: The five core SHA1 state variables (A, B, C, D, E)
 *   are maintained entirely within general-purpose registers (r3-r7) throughout
 *   the compression process, ensuring fast access and minimizing memory operations.
 * - **Byte Swapping**: Conditional byte-swapping (`rev`) is applied for little-endian
 *   systems to correctly handle message word byte order as required by SHA1.
 *
 * This implementation is derived from the OpenSSL project and re-licensed under GPLv2.
 * It serves as a critical component for accelerating SHA1 hash calculations within
 * the Linux kernel on ARM-based systems.
 */
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
@ details see https://www.opens,org/~appro/cryptogams/.
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
/**
 * @brief SHA1 compression function optimized for ARMv4 and later.
 * @details This function processes a block of data using the SHA1 algorithm's
 * compression function. It takes the current SHA1 state, input data, and the
 * number of blocks to process. Optimized for speed using assembly-level techniques.
 *
 * @param r0 Pointer to the `sha1_state` structure (containing A, B, C, D, E registers).
 * @param r1 Pointer to the input data (64-byte blocks).
 * @param r2 Number of 64-byte blocks to process.
 *
 * @pre `r0` points to a valid `sha1_state` structure.
 * @pre `r1` points to valid input data, aligned appropriately.
 * @pre `r2` contains a positive integer indicating the number of blocks.
 * @post The `sha1_state` structure pointed to by `r0` is updated with the hash
 *       of the processed blocks.
 */
ENTRY(sha1_block_data_order)
	@ Functional Utility: Save callee-saved registers (r4-r12) and the Link Register (lr)
	@ onto the stack. This adheres to ARM Procedure Call Standard (APCS) for function calls.
	stmdb	sp!,{r4-r12,lr}
	@ Pre-condition: r1 contains the start address of the input data.
	@ Functional Utility: Calculate the end address of the input data stream.
	@ r2 = r1 + (r2 << 6) where r2 initially holds the number of blocks, and each block is 64 bytes.
	@ This loop invariant simplifies the loop termination check.
	add	r2,r1,r2,lsl#6	@ r2 to point at the end of r1
	@ Functional Utility: Load the initial five 32-bit SHA1 hash state variables (A, B, C, D, E)
	@ from the `sha1_state` structure pointed to by r0 into registers r3-r7.
	@ These registers will hold the working hash variables throughout the compression.
	ldmia	r0,{r3,r4,r5,r6,r7}
.Lloop:
	@ This is the main loop that processes each 64-byte input block.
	@ Invariant: At the start of this loop, r3-r7 hold the current SHA1 state,
	@ r1 points to the current input block, and r2 points to the end of the input data.
	@ Functional Utility: Load the SHA1 round constant K_00_19 (0x5a827999) into r8.
	ldr	r8,.LK_00_19
	@ Functional Utility: Save the current stack pointer (sp) to r14 (Message Schedule Pointer)
	@ and allocate 15*4 bytes (15 32-bit words) on the stack for the message schedule (W[t]).
	mov	r14,sp
	sub	sp,sp,#15*4
	@ Functional Utility: Apply initial rotations to state variables B, C, D (r5, r6, r7).
	@ This is part of the SHA1 algorithm's mixing function.
	@ Specifically, B, C, D are rotated right by 30 bits (which is equivalent to left by 2 bits).
	mov	r5,r5,ror#30
	mov	r6,r6,ror#30
	mov	r7,r7,ror#30		@ [6]
.L_00_15:
@
@ SHA1 Rounds 0-15 (Initial Rounds)
@ Block Logic: This section computes the first 16 SHA1 rounds.
@ The core logical function F(B,C,D) = (B & C) | ((~B) & D) is used for these rounds.
@ The round constant K_00_19 (0x5a827999) is added, along with the current message word W[t]
@ and rotated values of the hash state.
@
#if __ARM_ARCH__<7
	@ Functional Utility: Load bytes from the input data (r1) to form a 32-bit word W[t].
	@ This sequence is for ARM architectures prior to ARMv7, handling byte-wise loading.
	ldrb	r10,[r1,#2]
	ldrb	r9,[r1,#3]
	ldrb	r11,[r1,#1]
	@ Functional Utility: Update E (r7) by adding K_00_19 (r8) and A (r3) rotated.
	@ This is part of the core SHA1 round computation.
	add	r7,r8,r7,ror#2			@ E+=K_00_19
	ldrb	r12,[r1],#4
	orr	r9,r9,r10,lsl#8
	@ Functional Utility: Compute F(B,C,D) = (B & C) | ((~B) & D). In ARM assembly, this is optimized as (C XOR D) AND B XOR D.
	eor	r10,r5,r6			@ F_xx_xx: r10 = B ^ C
	orr	r9,r9,r11,lsl#16
	add	r7,r7,r3,ror#27			@ E+=ROR(A,27) ; A is rotated right by 27 bits (left by 5 bits).
	orr	r9,r9,r12,lsl#24
#else
	@ Functional Utility: Load 32-bit word W[t] from input data (r1) directly.
	@ This is more efficient for ARMv7 and later architectures.
	ldr	r9,[r1],#4			@ handles unaligned
	@ Functional Utility: Update E (r7) with round constant and rotated A.
	add	r7,r8,r7,ror#2			@ E+=K_00_19 (E rotated right by 2 bits)
	@ Functional Utility: Compute F(B,C,D) for rounds 0-19: (B & C) | ((~B) & D).
	eor	r10,r5,r6			@ F_xx_xx: r10 = B ^ C
	add	r7,r7,r3,ror#27			@ E+=ROR(A,27) (A rotated right by 27 bits)
#ifdef __ARMEL__
	@ Functional Utility: Byte-swap the loaded word for little-endian systems.
	@ SHA1 operates on big-endian data, so this ensures correct byte ordering.
	rev	r9,r9
#endif
#endif
	@ Functional Utility: Complete the F function calculation and add to E.
	and	r10,r4,r10,ror#2        @ r10 = (C^D) & B' ; (B^C) & D
	add	r7,r7,r9			@ E+=W[t]
	eor	r10,r10,r6,ror#2		@ F_00_19(B,C,D) ; r10 = r10 ^ D (r6)
	@ Functional Utility: Store the current message word W[t] onto the stack for later use in message schedule.
	str	r9,[r14,#-4]!
	add	r7,r7,r10			@ E+=F_00_19(B,C,D)

	@ The following 4 blocks repeat the SHA1 round calculations for
	@ subsequent steps within the 0-15 range, adjusting state variables.
	@ (Similar functional utility as above, applied to different state variables)
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
	@ Functional Utility: Check if 15 words of the message schedule (W[t]) have been stored on the stack.
	@ The comparison is between the current stack pointer (r14, which is decremented) and the stack pointer
	@ at the beginning of the function (sp).
	cmp	r14,sp
	bne	.L_00_15 @ Loop back if not all 15 words are processed in the current block.
	@ Functional Utility: Deallocate additional stack space, preparing for the next set of rounds.
	sub	sp,sp,#25*4

@
@ SHA1 Rounds 16-19 (Message Schedule Interleaving)
@ Block Logic: These rounds continue the computation for the first 20 SHA1 rounds.
@ Critically, the message schedule generation is now interleaved with the hash updates.
@ The message word W[t] is calculated using the formula: W[t] = ROL((W[t-3]^W[t-8]^W[t-14]^W[t-16]), 1).
@ The logical function F(B,C,D) = (B & C) | ((~B) & D) and round constant K_00_19 are still used.
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

	@ Functional Utility: Calculate W[16] for the message schedule.
	@ W[16] = ROL((W[t-3]^W[t-8]^W[t-14]^W[t-16]), 1)
	@ Message words are retrieved from the stack relative to r14.
	ldr	r9,[r14,#15*4] @ W[t-3] (W[13])
	ldr	r10,[r14,#13*4] @ W[t-8] (W[8])
	ldr	r11,[r14,#7*4] @ W[t-14] (W[2])
	add	r6,r8,r6,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4] @ W[t-16] (W[0])
	eor	r9,r9,r10 @ W[13] ^ W[8]
	eor	r11,r11,r12 @ W[2] ^ W[0]
	eor	r10,r4,r5			@ F_xx_xx
	@ Functional Utility: Implement Rotate Left (ROL) by 1 bit as Rotate Right (ROR) by 31 bits.
	mov	r9,r9,ror#31 @ ROL(W[13]^W[8], 1)
	add	r6,r6,r7,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31 @ ROL((W[13]^W[8])^(W[2]^W[0]), 1)
	@ Functional Utility: Store the newly calculated message word W[t] onto the stack.
	str	r9,[r14,#-4]!
	and r10,r3,r10,ror#2
	add	r6,r6,r9			@ E+=X[i]
	eor	r10,r10,r5,ror#2		@ F_00_19(B,C,D)
	add	r6,r6,r10			@ E+=F_00_19(B,C,D)

	@ (Further blocks for rounds 17-19, with similar message schedule calculations and hash updates)
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

	@ Functional Utility: Load the round constant K_20_39 (0x6ed9eba1) into r8.
	ldr	r8,.LK_20_39		@ [+15+16*4]
	@ Functional Utility: Clear the carry flag by comparing sp with 0. This is an idiom
	@ used to distinguish between rounds 20-39 and 60-79 when re-using the same code block.
	cmn	sp,#0			@ [+3], clear carry to denote 20_39
.L_20_39_or_60_79:
@
@ SHA1 Rounds 20-39 or 60-79
@ Block Logic: This section implements SHA1 rounds where the logical function is
@ F(B,C,D) = B ^ C ^ D (XOR majority function).
@ The round constant K_20_39 (0x6ed9eba1) or K_60_79 (0xca62c1d6) is in r8,
@ determined by the carry flag set earlier. The message schedule calculation
@ continues with W[t] = ROL((W[t-3]^W[t-8]^W[t-14]^W[t-16]), 1).
@
	@ Functional Utility: Calculate the next message word W[t] and update the hash state.
	@ (Similar message schedule calculations and hash updates as in rounds 16-19,
	@ but with the XOR logical function and corresponding round constant.)
	ldr	r9,[r14,#15*4]
	ldr	r10,[r14,#13*4]
	ldr	r11,[r14,#7*4]
	add	r7,r8,r7,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4]
	eor	r9,r9,r10
	eor	r11,r11,r12
	eor	r10,r5,r6			@ F_xx_xx (here, B^C^D simplified as B^C)
	mov	r9,r9,ror#31
	add	r7,r7,r3,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31
	str	r9,[r14,#-4]!
	eor r10,r4,r10,ror#2 @ r10 = B^C^D
	add	r7,r7,r9			@ E+=X[i]
	add	r7,r7,r10			@ E+=F_20_39(B,C,D)

	@ (Further blocks for rounds within this range)
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

	@ Functional Utility: Load the round constant K_40_59 (0x8f1bbcdc) into r8.
	ldr	r8,.LK_40_59
	@ Functional Utility: Allocate more stack space for message schedule words.
	sub	sp,sp,#20*4		@ [+2]
.L_40_59:
@
@ SHA1 Rounds 40-59
@ Block Logic: This section implements SHA1 rounds using the majority function:
@ F(B,C,D) = (B & C) | (B & D) | (C & D).
@ The round constant K_40_59 (0x8f1bbcdc) is loaded into r8. The message schedule
@ calculation continues with W[t] = ROL((W[t-3]^W[t-8]^W[t-14]^W[t-16]), 1).
@
	@ Functional Utility: Calculate the next message word W[t] and update the hash state.
	@ (Similar message schedule calculations and hash updates as in previous rounds,
	@ but with the majority logical function and corresponding round constant.)
	ldr	r9,[r14,#15*4]
	ldr	r10,[r14,#13*4]
	ldr	r11,[r14,#7*4]
	add	r7,r8,r7,ror#2			@ E+=K_xx_xx
	ldr	r12,[r14,#2*4]
	eor	r9,r9,r10
	eor	r11,r11,r12
	eor	r10,r5,r6			@ F_xx_xx (here, B^C, part of majority function)
	mov	r9,r9,ror#31
	add	r7,r7,r3,ror#27			@ E+=ROR(A,27)
	eor	r9,r9,r11,ror#31
	str	r9,[r14,#-4]!
	and r10,r4,r10,ror#2
	and r11,r5,r6
	add	r7,r7,r9			@ E+=X[i]
	add	r7,r7,r10			@ E+=F_40_59(B,C,D) (intermediate add for majority)
	add	r7,r7,r11,ror#2 @ Final add for F_40_59 (B & C) | (B & D) | (C & D)
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

	@ Functional Utility: Load the round constant K_60_79 (0xca62c1d6) into r8.
	ldr	r8,.LK_60_79
	@ Functional Utility: Allocate more stack space for message schedule words.
	sub	sp,sp,#20*4
	@ Functional Utility: Set the carry flag by comparing sp with 0. This prepares
	@ for re-using the .L_20_39_or_60_79 code block for rounds 60-79.
	cmp	sp,#0			@ set carry to denote 60_79
	b	.L_20_39_or_60_79	@ Re-use logic for rounds 20-39
.L_done:
@
@ Epilogue: Final state update and function exit.
@ Functional Utility: This section concludes the SHA1 compression for the current
@ block, updates the cumulative hash state, and restores the processor state
@ before returning to the caller.
@
	@ Functional Utility: Deallocate the stack frame used for message schedule words.
	add	sp,sp,#80*4		@ "deallocate" stack frame
	@ Functional Utility: Load the initial hash values (A, B, C, D, E) for this block
	@ from the `sha1_state` structure (pointed to by r0) into registers r8-r12.
	ldmia	r0,{r8,r9,r10,r11,r12}
	@ Functional Utility: Add the results of this block's compression (r3-r7)
	@ to the initial hash state values (r8-r12) to produce the new cumulative hash state.
	add	r3,r8,r3
	add	r4,r9,r4
	add	r5,r10,r5,ror#2
	add	r6,r11,r6,ror#2
	add	r7,r12,r7,ror#2
	@ Functional Utility: Store the updated cumulative hash state (A, B, C, D, E)
	@ back into the `sha1_state` structure pointed to by r0.
	stmia	r0,{r3,r4,r5,r6,r7}
	@ Functional Utility: Check if all input data blocks have been processed.
	@ If r1 (current data pointer) has not reached r2 (end data pointer),
	@ loop back to process the next block.
	teq	r1,r2
	bne	.Lloop			@ If not, loop to the next block.

	@ Functional Utility: Restore the callee-saved registers (r4-r12) and the
	@ Program Counter (pc) from the stack, effectively returning from the function.
	ldmia	sp!,{r4-r12,pc}
.align	2
.LK_00_19:	.word	0x5a827999 @ Round constant for SHA1 rounds 0-19
.LK_20_39:	.word	0x6ed9eba1 @ Round constant for SHA1 rounds 20-39
.LK_40_59:	.word	0x8f1bbcdc @ Round constant for SHA1 rounds 40-59
.LK_60_79:	.word	0xca62c1d6 @ Round constant for SHA1 rounds 60-79
ENDPROC(sha1_block_data_order)
.asciz	"SHA1 block transform for ARMv4, CRYPTOGAMS by <appro@openssl.org>"
.align	2
