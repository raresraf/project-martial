/* SPDX-License-Identifier: GPL-2.0-only */
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
 * k0-k3: NEON registers holding the four SHA-1 round constants.
 * t0-t1: NEON temporary registers.
 * dga,dgb,dgbs: NEON registers holding the five 32-bit words of the SHA-1 state (A,B,C,D,E).
 * dg0,dg1*: Intermediate digest state registers.
 * q8-q11 (v8-v11): NEON registers for the 16-word message block.
 */
	k0		.req	v0
	k1		.req	v1
	k2		.req	v2
	k3		.req	v3

	t0		.req	v4
	t1		.req	v5

	dga		.req	q6
	dgav		.req	v6
	dgb		.req	s7
	dgbv		.req	v7

	dg0q		.req	q12
	dg0s		.req	s12
	dg0v		.req	v12
	dg1s		.req	s13
	dg1v		.req	v13
	dg2s		.req	s14

	/*
	 * add_only - Performs a single SHA-1 round update.
	 * op:  Round operation (c, p, or m for choose, parity, or majority).
	 * ev:  Even/odd temporary register selector.
	 * rc:  Round constant register.
	 * s0:  Message schedule register.
	 * dg1: Previous digest state.
	 *
	 * This macro uses sha1h (update A) and one of sha1c, sha1p, or sha1m
	 * to execute the core logic for a single SHA-1 round in hardware.
	 */
	.macro		add_only, op, ev, rc, s0, dg1
	.ifc		\ev, ev
	add		t1.4s, v\s0\().4s, \rc\().4s
	sha1h		dg2s, dg0s
	.ifnb		\dg1
	sha1\op		dg0q, \dg1, t0.4s
	.else
	sha1\op		dg0q, dg1s, t0.4s
	.endif
	.else
	.ifnb		\s0
	add		t0.4s, v\s0\().4s, \rc\().4s
	.endif
	sha1h		dg1s, dg0s
	sha1\op		dg0q, dg2s, t1.4s
	.endif
	.endm

	/*
	 * add_update - Performs a round update and calculates the next message words.
	 *
	 * This macro is the core engine of the compression loop. It executes one
	 * round by calling add_only, while simultaneously calculating the next
	 * four words of the message schedule using the sha1su0 and sha1su1
	 * hardware instructions. This interleaving is the key to efficiency.
	 */
	.macro		add_update, op, ev, rc, s0, s1, s2, s3, dg1
	sha1su0		v\s0\().4s, v\s1\().4s, v\s2\().4s
	add_only	\op, \ev, \rc, \s1, \dg1
	sha1su1		v\s0\().4s, v\s3\().4s
	.endm

	.macro		loadrc, k, val, tmp
	movz		\tmp, :abs_g0_nc:\val
	movk		\tmp, :abs_g1:\val
	dup		\k, \tmp
	.endm

	/*
	 * int __sha1_ce_transform(struct sha1_ce_state *sst, u8 const *src,
	 *			   int blocks);
	 * x0: Pointer to sha1_ce_state struct (A,B,C,D,E state).
	 * x1: Pointer to source data.
	 * w2: Number of 64-byte blocks to process.
	 */
SYM_FUNC_START(__sha1_ce_transform)
	/* Prologue: Load round constants and initial hash state into registers. */
	loadrc		k0.4s, 0x5a827999, w6
	loadrc		k1.4s, 0x6ed9eba1, w6
	loadrc		k2.4s, 0x8f1bbcdc, w6
	loadrc		k3.4s, 0xca62c1d6, w6

	ld1		{dgav.4s}, [x0]
	ldr		dgb, [x0, #16]

	ldr_l		w4, sha1_ce_offsetof_finalize, x4
	ldr		w4, [x0, x4]

	/* Main loop: processes one 64-byte block per iteration. */
0:	@ 1. Load 16 words (64 bytes) of message data into v8-v11.
	ld1		{v8.4s-v11.4s}, [x1], #64
	sub		w2, w2, #1

CPU_LE(	rev32		v8.16b, v8.16b		)
CPU_LE(	rev32		v9.16b, v9.16b		)
CPU_LE(	rev32		v10.16b, v10.16b	)
CPU_LE(	rev32		v11.16b, v11.16b	)

1:	@ 2. Execute the 80 rounds of SHA-1 using hardware instructions,
	@    interleaving message schedule updates with round calculations.
	add		t0.4s, v8.4s, k0.4s
	mov		dg0v.16b, dgav.16b

	@ Rounds 0-19 (choose function)
	add_update	c, ev, k0,  8,  9, 10, 11, dgb
	add_update	c, od, k0,  9, 10, 11,  8
	add_update	c, ev, k0, 10, 11,  8,  9
	add_update	c, od, k0, 11,  8,  9, 10
	add_update	c, ev, k1,  8,  9, 10, 11

	@ Rounds 20-39 (parity function)
	add_update	p, od, k1,  9, 10, 11,  8
	add_update	p, ev, k1, 10, 11,  8,  9
	add_update	p, od, k1, 11,  8,  9, 10
	add_update	p, ev, k1,  8,  9, 10, 11
	add_update	p, od, k2,  9, 10, 11,  8

	@ Rounds 40-59 (majority function)
	add_update	m, ev, k2, 10, 11,  8,  9
	add_update	m, od, k2, 11,  8,  9, 10
	add_update	m, ev, k2,  8,  9, 10, 11
	add_update	m, od, k2,  9, 10, 11,  8
	add_update	m, ev, k3, 10, 11,  8,  9

	@ Rounds 60-79 (parity function)
	add_update	p, od, k3, 11,  8,  9, 10
	add_only	p, ev, k3,  9
	add_only	p, od, k3, 10
	add_only	p, ev, k3, 11
	add_only	p, od

	@ 3. Update the digest state with the result of the compression.
	add		dgbv.2s, dgbv.2s, dg1v.2s
	add		dgav.4s, dgav.4s, dg0v.4s

	cbz		w2, 2f			@ If no more full blocks, check for final block.
	cond_yield	3f, x5, x6
	b		0b			@ Loop for next block.

	/*
	 * Final block handling: If the 'finalize' flag is set, this section
	 * constructs the padding and length encoding required by the SHA-1
	 * standard and processes it as the last block.
	 */
2:	cbz		x4, 3f
	ldr_l		w4, sha1_ce_offsetof_count, x4
	ldr		x4, [x0, x4]
	movi		v9.2d, #0
	mov		x8, #0x80000000
	movi		v10.2d, #0
	ror		x7, x4, #29		// ror(lsl(x4, 3), 32)
	fmov		d8, x8
	mov		x4, #0
	mov		v11.d[0], xzr
	mov		v11.d[1], x7
	b		1b			@ Re-run the compression on the padded block.

	/* Epilogue: Store the final state and return. */
3:	st1		{dgav.4s}, [x0]
	str		dgb, [x0, #16]
	mov		w0, w2
	ret
SYM_FUNC_END(__sha1_ce_transform)
