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
@ k0-k3: NEON registers for the four SHA-1 round constants.
	k0		.req	q0
	k1		.req	q1
	k2		.req	q2
	k3		.req	q3

@ ta0/tb0, ta1/tb1: NEON registers for temporary round calculations.
	ta0		.req	q4
	ta1		.req	q5
	tb0		.req	q5
	tb1		.req	q4

@ dga, dgb, dgbs: NEON registers holding the SHA-1 state (A,B,C,D,E).
@ dga contains A,B,C,D. dgb/dgbs contains E.
	dga		.req	q6
	dgb		.req	q7
	dgbs		.req	s28

@ dg0, dg1*: NEON registers for intermediate digest values during rounds.
	dg0		.req	q12
	dg1a0		.req	q13
	dg1a1		.req	q14
	dg1b0		.req	q14
	dg1b1		.req	q13

	@
	@ add_only - Performs a single SHA-1 round update.
	@ op: Round operation (c, p, or m for choose, parity, or majority).
	@ ev: Even/odd temporary register selector (0 or 1).
	@ rc: Round constant register.
	@ s0: Message schedule register.
	@ dg1: Previous digest state.
	@
	@ This macro uses sha1h (update A from E) and one of sha1c, sha1p, or
	@ sha1m to execute the core logic for a single SHA-1 round in hardware.
	@
	.macro		add_only, op, ev, rc, s0, dg1
	.ifnb		\s0
	vadd.u32	tb\ev, q\s0, \rc
	.endif
	sha1h.32	dg1b\ev, dg0
	.ifb		\dg1
	sha1\op\().32	dg0, dg1a\ev, ta\ev
	.else
	sha1\op\().32	dg0, \dg1, ta\ev
	.endif
	.endm

	@
	@ add_update - Performs a round update and calculates the next message words.
	@
	@ This macro is the core of the compression loop. It executes one round by
	@ calling add_only, while simultaneously using sha1su0 and sha1su1 to
	@ calculate the next four words of the message schedule (W[i] to W[i+3]).
	@ This interleaving of message scheduling and round computation is key to
	@ the hardware implementation's efficiency.
	@
	.macro		add_update, op, ev, rc, s0, s1, s2, s3, dg1
	sha1su0.32	q\s0, q\s1, q\s2
	add_only	\op, \ev, \rc, \s1, \dg1
	sha1su1.32	q\s0, q\s3
	.endm

	.align		6
.Lsha1_rcon:
	.word		0x5a827999, 0x5a827999, 0x5a827999, 0x5a827999
	.word		0x6ed9eba1, 0x6ed9eba1, 0x6ed9eba1, 0x6ed9eba1
	.word		0x8f1bbcdc, 0x8f1bbcdc, 0x8f1bbcdc, 0x8f1bbcdc
	.word		0xca62c1d6, 0xca62c1d6, 0xca62c1d6, 0xca62c1d6

	/*
	 * void sha1_ce_transform(struct sha1_state *sst, u8 const *src,
	 *			  int blocks);
	 * r0: Pointer to sha1_state struct.
	 * r1: Pointer to source data.
	 * r2: Number of 64-byte blocks to process.
	 */
ENTRY(sha1_ce_transform)
	@ Prologue: Load round constants and initial hash state into registers.
	adr		ip, .Lsha1_rcon
	vld1.32		{k0-k1}, [ip, :128]!
	vld1.32		{k2-k3}, [ip, :128]

	vld1.32		{dga}, [r0]
	vldr		dgbs, [r0, #16]

	@ Main loop: processes one 64-byte block per iteration.
0:	@ 1. Load 16 words (64 bytes) of message data into q8-q11.
	vld1.32		{q8-q9}, [r1]!
	vld1.32		{q10-q11}, [r1]!
	subs		r2, r2, #1

#ifndef CONFIG_CPU_BIG_ENDIAN
	@ 2. On little-endian systems, reverse byte order to match SHA-1's big-endian standard.
	vrev32.8	q8, q8
	vrev32.8	q9, q9
	vrev32.8	q10, q10
	vrev32.8	q11, q11
#endif

	@ 3. Execute the 80 rounds of SHA-1 by invoking the hardware instructions
	@    via the add_update and add_only macros.
	vadd.u32	ta0, q8, k0
	vmov		dg0, dga

	@ Rounds 0-19 (choose function)
	add_update	c, 0, k0,  8,  9, 10, 11, dgb
	add_update	c, 1, k0,  9, 10, 11,  8
	add_update	c, 0, k0, 10, 11,  8,  9
	add_update	c, 1, k0, 11,  8,  9, 10
	add_update	c, 0, k1,  8,  9, 10, 11

	@ Rounds 20-39 (parity function)
	add_update	p, 1, k1,  9, 10, 11,  8
	add_update	p, 0, k1, 10, 11,  8,  9
	add_update	p, 1, k1, 11,  8,  9, 10
	add_update	p, 0, k1,  8,  9, 10, 11
	add_update	p, 1, k2,  9, 10, 11,  8

	@ Rounds 40-59 (majority function)
	add_update	m, 0, k2, 10, 11,  8,  9
	add_update	m, 1, k2, 11,  8,  9, 10
	add_update	m, 0, k2,  8,  9, 10, 11
	add_update	m, 1, k2,  9, 10, 11,  8
	add_update	m, 0, k3, 10, 11,  8,  9

	@ Rounds 60-79 (parity function)
	add_update	p, 1, k3, 11,  8,  9, 10
	add_only	p, 0, k3,  9
	add_only	p, 1, k3, 10
	add_only	p, 0, k3, 11
	add_only	p, 1

	@ Epilogue for the current block.
	@ Update the digest state with the result of the compression.
	vadd.u32	dga, dga, dg0
	vadd.u32	dgb, dgb, dg1a0
	@ Loop if there are more blocks to process.
	bne		0b

	@ Store the final updated state back to memory.
	vst1.32		{dga}, [r0]
	vstr		dgbs, [r0, #16]
	bx		lr
ENDPROC(sha1_ce_transform)
