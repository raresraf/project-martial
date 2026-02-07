/* SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause */
/**
 * @file sha512-riscv64-zvknhb-zvkb.S
 * @brief High-performance SHA-512/384 implementation using RISC-V Vector Cryptography Extensions.
 * @details This assembly file provides a highly optimized implementation of the
 * SHA-512 (and implicitly SHA-384) compression function, specifically tuned
 * for RISC-V 64-bit architectures equipped with the Vector (V), Vector SHA-2
 * Secure Hash (Zvknhb), and Vector Cryptography Bit-manipulation (Zvkb) extensions.
 * It leverages the RISC-V vector registers and instructions to perform
 * cryptographic operations in parallel, significantly accelerating the hashing process.
 *
 * **Hardware Acceleration Strategy:**
 * The core SHA-512 logic is executed directly using RISC-V vector instructions. Key
 * instructions from the Zvknhb extension (`vsha2cl.vv`, `vsha2ch.vv`) perform
 * SHA-2 compression loop operations, while `vsha2ms.vv` handles message schedule
 * generation. The Zvkb extension provides bit-manipulation capabilities.
 *
 * **Optimization Techniques:**
 * - **Vectorization**: Operations are performed on multiple 64-bit words simultaneously
 *   using vector registers (e.g., `v10-v19` for message schedule, `v22`, `v24` for SHA-512 state).
 * - **Message Schedule Acceleration**: `vsha2ms.vv` efficiently computes message schedule words.
 * - **Endianness Handling**: Message words are byte-reversed (`vrev8.v`) to ensure big-endian
 *   format as required by SHA-512.
 * - **Register Mapping**: Dedicated GPRs (`a0`, `a1`, `a2`, `a3`, `a4`) and vector registers
 *   (`v0`, `v1`, `v10`-`v28`) are used for SHA-512 state, message words, and constants.
 * This implementation is crucial for high-performance cryptographic operations on
 * RISC-V 64-bit platforms with the specified vector crypto extensions.
 */
//
// This file is dual-licensed, meaning that you can use it under your
// choice of either of the following two licenses:
//
// Copyright 2023 The OpenSSL Project Authors. All Rights Reserved.
//
// Licensed under the Apache License 2.0 (the "License"). You can obtain
// a copy in the file LICENSE in the source distribution or at
// https://www.openssl.org/source/license.html
//
// or
//
// Copyright (c) 2023, Christoph Müllner <christoph.muellner@vrull.eu>
// Copyright (c) 2023, Phoebe Chen <phoebe.chen@sifive.com>
// Copyright 2024 Google LLC
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// The generated code of this file depends on the following RISC-V extensions:
// - RV64I
// - RISC-V Vector ('V') with VLEN >= 128
// - RISC-V Vector SHA-2 Secure Hash extension ('Zvknhb')
// - RISC-V Vector Cryptography Bit-manipulation extension ('Zvkb')

#include <linux/linkage.h>

.text
.option arch, +zvknhb, +zvkb

#define STATEP		a0 /* Functional Role: General-purpose register (GPR) holding pointer to SHA-512 state (u64 state[8]). */
#define DATA		a1 /* Functional Role: GPR holding pointer to input data. */
#define NUM_BLOCKS	a2 /* Functional Role: GPR holding the number of 128-byte blocks to process. */

#define STATEP_C	a3 /* Functional Role: GPR holding pointer to a calculated offset of SHA-512 state for loading. */
#define K		a4 /* Functional Role: GPR holding pointer to SHA-512 round constants table (K512). */

#define MASK		v0 /* Functional Role: Vector register holding a mask for `vmerge.vvm` instruction. */
#define INDICES		v1 /* Functional Role: Vector register holding indices for scattered memory access. */
#define W0		v10 /* Functional Role: Vector register (LMUL=2) holding message schedule words (W[t] to W[t+3]). */
#define W1		v12 /* Functional Role: Vector register (LMUL=2) holding message schedule words (W[t+4] to W[t+7]). */
#define W2		v14 /* Functional Role: Vector register (LMUL=2) holding message schedule words (W[t+8] to W[t+11]). */
#define W3		v16 /* Functional Role: Vector register (LMUL=2) holding message schedule words (W[t+12] to W[t+15]). */
#define VTMP		v20 /* Functional Role: Vector register (LMUL=2) for temporary computations. */
#define FEBA		v22 /* Functional Role: Vector register (LMUL=2) holding SHA-512 state variables F, E, B, A (64-bit words). */
#define HGDC		v24 /* Functional Role: Vector register (LMUL=2) holding SHA-512 state variables H, G, D, C (64-bit words). */
#define PREV_FEBA	v26 /* Functional Role: Vector register (LMUL=2) holding previous FEBA state for accumulation. */
#define PREV_HGDC	v28 /* Functional Role: Vector register (LMUL=2) holding previous HGDC state for accumulation. */

/**
 * @brief Executes 4 rounds of SHA-512 compression with optional message schedule update.
 * @details This macro performs four rounds of the SHA-512 algorithm using RISC-V
 * vector crypto extensions (Zvknhb). It updates the SHA-512 state (represented
 * by `FEBA` and `HGDC`) based on the current message schedule words (`w0`) and
 * round constants (`K`). If `last` is not set, it also computes the *next* 4
 * message schedule words, implementing a pipelined message expansion.
 *
 * @param last Boolean flag: if set (1), this is the last call in a 16-round block,
 *             and message schedule words are not computed for the next iteration.
 * @param w0 Vector register holding current 4 message schedule words.
 * @param w1 Vector register holding next 4 message schedule words.
 * @param w2 Vector register holding next 4 message schedule words after `w1`.
 * @param w3 Vector register holding next 4 message schedule words after `w2`.
 * Functional Utility: Performs a vectorized block of SHA-512 rounds, optionally computing the next message schedule.
 */
.macro	sha512_4rounds	last, w0, w1, w2, w3
	vle64.v		VTMP, (K) /* Functional Utility: Loads 4 SHA-512 round constants from `K` into `VTMP`. */
	addi		K, K, 32 /* Functional Utility: Increments `K` pointer to the next set of round constants. */
	vadd.vv		VTMP, VTMP, \w0 /* Functional Utility: Adds the round constants (VTMP) to the current message schedule words (\w0). */
	vsha2cl.vv	HGDC, FEBA, VTMP /* Functional Utility: RISC-V Zvknhb instruction - Computes the SHA-512 compression logic for CL (Choose) function. */
	vsha2ch.vv	FEBA, HGDC, VTMP /* Functional Utility: RISC-V Zvknhb instruction - Computes the SHA-512 compression logic for CH (Ch) function. */
.if !\last /* Functional Utility: Conditional compilation: if this is NOT the last 4 rounds in a 16-round block. */
	vmerge.vvm	VTMP, \w2, \w1, MASK /* Functional Utility: Merges vector registers for message schedule update based on `MASK`. */
	vsha2ms.vv	\w0, VTMP, \w3 /* Functional Utility: RISC-V Zvknhb instruction - Computes message schedule words for the next iteration. */
.endif
.endm

/**
 * @brief Executes 16 rounds of SHA-512 compression.
 * @details This macro orchestrates 16 consecutive rounds of SHA-512 compression
 * by making four sequential calls to `sha512_4rounds`. It manages the cycling
 * of message schedule word registers (`W0`, `W1`, `W2`, `W3`) to efficiently
 * feed the `sha512_4rounds` macro.
 *
 * @param last Boolean flag: if set (1), this is the last 16-round block,
 *             and message schedule words are not computed for the next iteration.
 * Functional Utility: Orchestrates 16 rounds of SHA-512 computation using vectorized operations.
 */
.macro	sha512_16rounds	last
	sha512_4rounds	\last, W0, W1, W2, W3 /* Functional Utility: Processes 4 rounds, W0 is current. */
	sha512_4rounds	\last, W1, W2, W3, W0 /* Functional Utility: Processes next 4 rounds, W1 is current. */
	sha512_4rounds	\last, W2, W3, W0, W1 /* Functional Utility: Processes next 4 rounds, W2 is current. */
	sha512_4rounds	\last, W3, W0, W1, W2 /* Functional Utility: Processes next 4 rounds, W3 is current. */
.endm

/**
 * @brief RISC-V 64-bit vector crypto accelerated SHA-512/384 transform function.
 * @details This function performs the core SHA-512 or SHA-384 message compression
 * using RISC-V 64-bit Vector (V) extension instructions, specifically leveraging
 * the Zvknhb (Vector SHA-2 Secure Hash) and Zvkb (Vector Cryptography Bit-manipulation)
 * extensions. It processes a specified number of 128-byte data blocks, updating
 * the SHA-512 state context in vector registers. The implementation is highly
 * optimized for performance through extensive vectorization.
 *
 * @param STATEP (a0): Pointer to the `sha512_state` context structure (u64 state[8]).
 * @param DATA (a1): Pointer to the input data blocks (128-byte aligned).
 * @param NUM_BLOCKS (a2): Number of 128-byte blocks to process.
 * Functional Utility: Executes hardware-accelerated SHA-512/384 compression using RISC-V vector crypto extensions.
 */
SYM_FUNC_START(sha512_transform_zvknhb_zvkb)

	/* Setup mask for the vmerge to replace the first word (idx==0) in
	 * message scheduling.  There are 4 words, so an 8-bit mask suffices.
	 */
	// Functional Utility: Configures the Vector Length and element width for subsequent vector instructions.
	// Sets VLEN to 1 element, e8 (8-bit elements), m1 (LMUL=1), ta (tail-agnostic), ma (mask-agnostic).
	vsetivli	zero, 1, e8, m1, ta, ma
	// Functional Utility: Moves the immediate value 0x01 into the vector register `MASK`. This creates a mask where the first element is set, used for selective merging in message scheduling.
	vmv.v.i		MASK, 0x01

	/* Load the state.  The state is stored as {a,b,c,d,e,f,g,h}, but we
	 * need {f,e,b,a},{h,g,d,c}.  The dst vtype is e64m2 and the index vtype
	 * is e8mf4.  We use index-load with the i8 indices {40, 32, 8, 0},
	 * loaded using the 32-bit little endian value 0x00082028.
	 */
	// Functional Utility: Loads a 32-bit immediate value (0x00082028) into GPR `t0`, which represents scattered indices.
	li		t0, 0x00082028
	// Functional Utility: Configures the Vector Length and element width for subsequent vector instructions.
	// Sets VLEN to 1 element, e32 (32-bit elements), m1 (LMUL=1), ta (tail-agnostic), ma (mask-agnostic).
	vsetivli	zero, 1, e32, m1, ta, ma
	// Functional Utility: Moves the GPR value from `t0` into the vector register `INDICES`.
	vmv.v.x		INDICES, t0
	// Functional Utility: Calculates an offset pointer to load a different part of the state.
	addi		STATEP_C, STATEP, 16
	// Functional Utility: Configures the Vector Length and element width for subsequent vector instructions.
	// Sets VLEN to 4 elements, e64 (64-bit elements), m2 (LMUL=2), ta (tail-agnostic), ma (mask-agnostic).
	vsetivli	zero, 4, e64, m2, ta, ma
	// Functional Utility: Loads SHA-512 state (F, E, B, A) from scattered memory locations pointed by `STATEP` using `INDICES`.
	vluxei8.v	FEBA, (STATEP), INDICES
	// Functional Utility: Loads SHA-512 state (H, G, D, C) from scattered memory locations pointed by `STATEP_C` using `INDICES`.
	vluxei8.v	HGDC, (STATEP_C), INDICES

.Lnext_block:
	// Functional Utility: Loads the base address of the SHA-512 round constants table (K512) into `K`.
	la		K, K512
	// Functional Utility: Decrements the block counter.
	addi		NUM_BLOCKS, NUM_BLOCKS, -1

	// Save the previous state, as it's needed later for accumulation.
	vmv.v.v		PREV_FEBA, FEBA /* Functional Utility: Saves current FEBA state into PREV_FEBA. */
	vmv.v.v		PREV_HGDC, HGDC /* Functional Utility: Saves current HGDC state into PREV_HGDC. */

	// Load the next 1024-bit message block and endian-swap each 64-bit word
	// Functional Utility: Loads 4x 64-bit message words into `W0` and then byte-reverses them for big-endian.
	vle64.v		W0, (DATA)
	vrev8.v		W0, W0
	addi		DATA, DATA, 32
	// Functional Utility: Loads 4x 64-bit message words into `W1` and then byte-reverses them for big-endian.
	vle64.v		W1, (DATA)
	vrev8.v		W1, W1
	addi		DATA, DATA, 32
	// Functional Utility: Loads 4x 64-bit message words into `W2` and then byte-reverses them for big-endian.
	vle64.v		W2, (DATA)
	vrev8.v		W2, W2
	addi		DATA, DATA, 32
	// Functional Utility: Loads 4x 64-bit message words into `W3` and then byte-reverses them for big-endian.
	vle64.v		W3, (DATA)
	vrev8.v		W3, W3
	addi		DATA, DATA, 32

	// Do the 80 rounds of SHA-512.
	// Functional Utility: Executes 16 SHA-512 rounds, with message schedule update enabled.
	sha512_16rounds 0
	// Functional Utility: Executes next 16 SHA-512 rounds, with message schedule update enabled.
	sha512_16rounds 0
	// Functional Utility: Executes next 16 SHA-512 rounds, with message schedule update enabled.
	sha512_16rounds 0
	// Functional Utility: Executes next 16 SHA-512 rounds, with message schedule update enabled.
	sha512_16rounds 0
	// Functional Utility: Executes final 16 SHA-512 rounds, with message schedule update disabled (last round block).
	sha512_16rounds 1

	// Add the previous state for accumulation.
	vadd.vv		FEBA, FEBA, PREV_FEBA /* Functional Utility: Adds previous FEBA state to current FEBA result. */
	vadd.vv		HGDC, HGDC, PREV_HGDC /* Functional Utility: Adds previous HGDC state to current HGDC result. */

	// Repeat if more blocks remain.
	bnez		NUM_BLOCKS, .Lnext_block /* Functional Utility: If NUM_BLOCKS is not zero, branch to process the next block. */

	// Store the new state and return.
	vsuxei8.v	FEBA, (STATEP), INDICES /* Functional Utility: Stores updated FEBA state back to memory using scattered-store with `INDICES`. */
	vsuxei8.v	HGDC, (STATEP_C), INDICES /* Functional Utility: Stores updated HGDC state back to memory using scattered-store with `INDICES`. */
	ret /* Functional Utility: Returns from the function. */
SYM_FUNC_END(sha512_transform_zvknhb_zvkb)

.section ".rodata"
.p2align 3
.type K512, @object
K512:
	.dword		0x428a2f98d728ae22, 0x7137449123ef65cd
	.dword		0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc
	.dword		0x3956c25bf348b538, 0x59f111f1b605d019
	.dword		0x923f82a4af194f9b, 0xab1c5ed5da6d8118
	.dword		0xd807aa98a3030242, 0x12835b0145706fbe
	.dword		0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2
	.dword		0x72be5d74f27b896f, 0x80deb1fe3b1696b1
	.dword		0x9bdc06a725c71235, 0xc19bf174cf692694
	.dword		0xe49b69c19ef14ad2, 0xefbe4786384f25e3
	.dword		0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65
	.dword		0x2de92c6f592b0275, 0x4a7484aa6ea6e483
	.dword		0x5cb0a9dcbd41fbd4, 0x76f988da831153b5
	.dword		0x983e5152ee66dfab, 0xa831c66d2db43210
	.dword		0xb00327c898fb213f, 0xbf597fc7beef0ee4
	.dword		0xc6e00bf33da88fc2, 0xd5a79147930aa725
	.dword		0x06ca6351e003826f, 0x142929670a0e6e70
	.dword		0x27b70a8546d22ffc, 0x2e1b21385c26c926
	.dword		0x4d2c6dfc5ac42aed, 0x53380d139d95b3df
	.dword		0x650a73548baf63de, 0x766a0abb3c77b2a8
	.dword		0x81c2c92e47edaee6, 0x92722c851482353b
	.dword		0xa2bfe8a14cf10364, 0xa81a664bbc423001
	.dword		0xc24b8b70d0f89791, 0xc76c51a30654be30
	.dword		0xd192e819d6ef5218, 0xd69906245565a910
	.dword		0xf40e35855771202a, 0x106aa07032bbd1b8
	.dword		0x19a4c116b8d2d0c8, 0x1e376c085141ab53
	.dword		0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8
	.dword		0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb
	.dword		0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3
	.dword		0x748f82ee5defb2fc, 0x78a5636f43172f60
	.dword		0x84c87814a1f0ab72, 0x8cc702081a6439ec
	.dword		0x90befffa23631e28, 0xa4506cebde82bde9
	.dword		0xbef9a3f7b2c67915, 0xc67178f2e372532b
	.dword		0xca273eceea26619c, 0xd186b8c721c0c207
	.dword		0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178
	.dword		0x06f067aa72176fba, 0x0a637dc5a2c898a6
	.dword		0x113f9804bef90dae, 0x1b710b35131c471b
	.dword		0x28db77f523047d84, 0x32caab7b40c72493
	.dword		0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c
	.dword		0x4cc5d4becb3e42b6, 0x597f299cfc657e2a
	.dword		0x5fcb6fab3ad6faec, 0x6c44198c4a475817
.size K512, . - K512
