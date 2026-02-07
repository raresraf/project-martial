/*
 * Intel SHA Extensions optimized implementation of a SHA-1 update function
 *
 * This file is provided under a dual BSD/GPLv2 license.  When using or
 * redistributing this file, you may do so under either license.
 *
 * GPL LICENSE SUMMARY
 *
 * Copyright(c) 2015 Intel Corporation.
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
 * 	Sean Gulley <sean.m.gulley@intel.com>
 * 	Tim Chen <tim.c.chen@linux.intel.com>
 *
 * BSD LICENSE
 *
 * Copyright(c) 2015 Intel Corporation.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 	* Redistributions of source code must retain the above copyright
 * 	  notice, this list of conditions and the following disclaimer.
 * 	* Redistributions in binary form must reproduce the above copyright
 * 	  notice, this list of conditions and the following disclaimer in
 * 	  the documentation and/or other materials provided with the
 * 	  distribution.
 * 	* Neither the name of Intel Corporation nor the names of its
 * 	  contributors may be used to endorse or promote products derived
 * 	  from this software without specific prior written permission.
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
 * @file sha1_ni_asm.S
 * @brief High-performance SHA-1 implementation leveraging Intel SHA Extensions (SHA-NI) for x86_64.
 * @details This assembly file provides a highly optimized implementation of the
 * SHA-1 (Secure Hash Algorithm 1) compression function, specifically tuned
 * for x86 processors equipped with Intel SHA Extensions (SHA-NI). It utilizes
 * dedicated SHA-NI instructions to accelerate the cryptographic process,
 * leading to significant performance improvements over software-only implementations.
 *
 * **Optimization Techniques:**
 * - **SHA-NI Instructions**: Extensive use of specialized instructions like
 *   `sha1rnds4`, `sha1nexte`, `sha1msg1`, and `sha1msg2` directly perform
 *   SHA-1 round computations and message schedule generation.
 * - **SIMD Processing**: XMM registers are used to process multiple SHA-1
 *   state components and message words in parallel.
 * - **Message Schedule Acceleration**: `sha1msg1` and `sha1msg2` instructions
 *   efficiently generate the message schedule words with minimal overhead.
 * - **Software Pipelining**: The loop structure implicitly supports a form of
 *   pipelining where message schedule generation overlaps with hash computations.
 *
 * This implementation is designed to provide maximum SHA-1 hashing performance
 * on SHA-NI-enabled x86 systems within the Linux kernel.
 */

#include <linux/linkage.h>
#include <linux/cfi_types.h>

#define DIGEST_PTR	%rdi	/* Functional Role: Pointer to `uint32_t *digest` (argument 1). */
#define DATA_PTR	%rsi	/* Functional Role: Pointer to `const void *data` (argument 2). */
#define NUM_BLKS	%rdx	/* Functional Role: `uint32_t numBlocks` (argument 3), number of 64-byte blocks. */

/* gcc conversion */
#define FRAME_SIZE	32	/* Functional Role: Size of the stack frame for local variables (e.g., saving XMM registers). */

#define ABCD		%xmm0	/* Functional Role: XMM register holding SHA-1 state A, B, C, D (packed 32-bit words). */
#define E0		%xmm1	/* Functional Role: XMM register holding SHA-1 state E for current round (packed 32-bit words). */
#define E1		%xmm2	/* Functional Role: XMM register holding SHA-1 state E for next round, used for ping-ponging E. */
#define MSG0		%xmm3	/* Functional Role: XMM register for current message block words. */
#define MSG1		%xmm4	/* Functional Role: XMM register for current message block words. */
#define MSG2		%xmm5	/* Functional Role: XMM register for current message block words. */
#define MSG3		%xmm6	/* Functional Role: XMM register for current message block words. */
#define SHUF_MASK	%xmm7	/* Functional Role: XMM register holding the shuffle mask for byte-swapping. */


/*
 * Intel SHA Extensions optimized implementation of a SHA-1 update function
 *
 * The function takes a pointer to the current hash values, a pointer to the
 * input data, and a number of 64 byte blocks to process.  Once all blocks have
 * been processed, the digest pointer is  updated with the resulting hash value.
 * The function only processes complete blocks, there is no functionality to
 * store partial blocks. All message padding and hash value initialization must
 * be done outside the update function.
 *
 * The indented lines in the loop are instructions related to rounds processing.
 * The non-indented lines are instructions related to the message schedule.
 *
 * void sha1_ni_transform(uint32_t *digest, const void *data,
		uint32_t numBlocks)
 * digest : pointer to digest
 * data: pointer to input data
 * numBlocks: Number of blocks to process
 */
.text
/**
 * @brief SHA-1 transform function optimized with Intel SHA Extensions (SHA-NI).
 * @details This function processes `numBlocks` of 64-byte input data using
 * Intel SHA-NI instructions to update an existing SHA-1 hash digest. It is
 * designed for high performance, leveraging specialized hardware instructions
 * for both hash computation (`sha1rnds4`, `sha1nexte`) and message schedule
 * generation (`sha1msg1`, `sha1msg2`). The function operates on complete 64-byte
 * blocks; padding and initial hash value setup must be handled externally.
 *
 * @param DIGEST_PTR (rdi): Pointer to the 5x `uint32_t` SHA-1 hash digest array.
 * @param DATA_PTR (rsi): Pointer to the `const void` input data.
 * @param NUM_BLKS (rdx): Number of 64-byte blocks to process.
 * Functional Utility: Accelerates SHA-1 compression using Intel SHA-NI instructions for multiple data blocks.
 */
SYM_TYPED_FUNC_START(sha1_ni_transform)
	// Functional Utility: Standard function prologue: saves %rbp, sets up new frame.
	push		%rbp
	mov		%rsp, %rbp
	// Functional Utility: Allocates space on the stack for local variables (e.g., temporary storage of XMM registers).
	sub		$FRAME_SIZE, %rsp
	// Functional Utility: Aligns the stack pointer to a 16-byte boundary, required for XMM operations.
	and		$~0xF, %rsp

	// Functional Utility: Converts the number of blocks from count to total bytes (numBlocks * 64).
	shl		$6, NUM_BLKS
	// Functional Utility: Checks if NUM_BLKS is zero; if so, skips directly to done.
	jz		.Ldone_hash
	// Functional Utility: Calculates the pointer to the end of the input data buffer.
	add		DATA_PTR, NUM_BLKS

	/* load initial hash values */
	// Functional Utility: Extracts the E component of the SHA-1 digest from `DIGEST_PTR + 16` into E0.
	pinsrd		$3, 1*16(DIGEST_PTR), E0
	// Functional Utility: Loads the A, B, C, D components of the SHA-1 digest from `DIGEST_PTR` into ABCD.
	movdqu		0*16(DIGEST_PTR), ABCD
	// Functional Utility: Masks off the upper words of E0, leaving only the required 32-bit E value.
	pand		UPPER_WORD_MASK(%rip), E0
	// Functional Utility: Permutes the ABCD register to align with the SHA-1 internal state representation.
	pshufd		$0x1B, ABCD, ABCD

	// Functional Utility: Loads the byte-flip shuffle mask into SHUF_MASK register, used for endian conversion of message words.
	movdqa		PSHUFFLE_BYTE_FLIP_MASK(%rip), SHUF_MASK

.Lloop0: /* Functional Role: Main loop for processing each 64-byte data block. */
	/* Save hash values for addition after rounds */
	// Functional Utility: Saves the initial hash values (E0, ABCD) to the stack for later accumulation.
	movdqa		E0, (0*16)(%rsp)
	movdqa		ABCD, (1*16)(%rsp)

	/* Rounds 0-3 */
	// Functional Utility: Loads the first 16 bytes (4 DWORDS) of the current message block into MSG0.
	movdqu		0*16(DATA_PTR), MSG0
	// Functional Utility: Byte-swaps the DWORDS in MSG0 using SHUF_MASK for big-endian format.
	pshufb		SHUF_MASK, MSG0
		// Functional Utility: Adds MSG0 to E0.
		paddd		MSG0, E0
		// Functional Utility: Copies ABCD to E1.
		movdqa		ABCD, E1
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 0-3) using initial constants ($0).
		sha1rnds4	$0, E0, ABCD

	/* Rounds 4-7 */
	// Functional Utility: Loads the next 16 bytes (4 DWORDS) of the current message block into MSG1.
	movdqu		1*16(DATA_PTR), MSG1
	// Functional Utility: Byte-swaps the DWORDS in MSG1.
	pshufb		SHUF_MASK, MSG1
		// Functional Utility: Prepares for the next E value, part of E0/E1 ping-pong.
		sha1nexte	MSG1, E1
		// Functional Utility: Copies ABCD to E0.
		movdqa		ABCD, E0
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 4-7) using initial constants ($0).
		sha1rnds4	$0, E1, ABCD
	// Functional Utility: Generates the next set of message words (W[16]-W[19]) based on previous words.
	sha1msg1	MSG1, MSG0

	/* Rounds 8-11 */
	// Functional Utility: Loads the next 16 bytes (4 DWORDS) of the current message block into MSG2.
	movdqu		2*16(DATA_PTR), MSG2
	// Functional Utility: Byte-swaps the DWORDS in MSG2.
	pshufb		SHUF_MASK, MSG2
		// Functional Utility: Prepares for the next E value.
		sha1nexte	MSG2, E0
		// Functional Utility: Copies ABCD to E1.
		movdqa		ABCD, E1
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 8-11) using initial constants ($0).
		sha1rnds4	$0, E0, ABCD
	// Functional Utility: Generates message words.
	sha1msg1	MSG2, MSG1
	// Functional Utility: XORs MSG2 with MSG0 for message schedule W[16]-W[19].
	pxor		MSG2, MSG0

	/* Rounds 12-15 */
	// Functional Utility: Loads the next 16 bytes (4 DWORDS) of the current message block into MSG3.
	movdqu		3*16(DATA_PTR), MSG3
	// Functional Utility: Byte-swaps the DWORDS in MSG3.
	pshufb		SHUF_MASK, MSG3
		// Functional Utility: Prepares for the next E value.
		sha1nexte	MSG3, E1
		// Functional Utility: Copies ABCD to E0.
		movdqa		ABCD, E0
	// Functional Utility: Generates message words for W[16]-W[19] and W[20]-W[23].
	sha1msg2	MSG3, MSG0
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 12-15) using initial constants ($0).
		sha1rnds4	$0, E1, ABCD
	// Functional Utility: Generates message words.
	sha1msg1	MSG3, MSG2
	// Functional Utility: XORs MSG3 with MSG1 for message schedule W[20]-W[23].
	pxor		MSG3, MSG1

	/* Rounds 16-19 */
		// Functional Utility: Prepares for the next E value, now using MSG0 which holds W[16]-W[19].
		sha1nexte	MSG0, E0
		// Functional Utility: Copies ABCD to E1.
		movdqa		ABCD, E1
	// Functional Utility: Generates message words for W[24]-W[27].
	sha1msg2	MSG0, MSG1
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 16-19) using initial constants ($0).
		sha1rnds4	$0, E0, ABCD
	// Functional Utility: Generates message words.
	sha1msg1	MSG0, MSG3
	// Functional Utility: XORs MSG0 with MSG2 for message schedule W[28]-W[31].
	pxor		MSG0, MSG2

	/* Rounds 20-23 */
		// Functional Utility: Prepares for the next E value, now using MSG1 which holds W[20]-W[23].
		sha1nexte	MSG1, E1
		// Functional Utility: Copies ABCD to E0.
		movdqa		ABCD, E0
	// Functional Utility: Generates message words for W[28]-W[31].
	sha1msg2	MSG1, MSG2
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 20-23) using constants for rounds 20-39 ($1).
		sha1rnds4	$1, E1, ABCD
	// Functional Utility: Generates message words.
	sha1msg1	MSG1, MSG0
	// Functional Utility: XORs MSG1 with MSG3 for message schedule W[32]-W[35].
	pxor		MSG1, MSG3

	/* Rounds 24-27 */
		// Functional Utility: Prepares for the next E value, now using MSG2 which holds W[24]-W[27].
		sha1nexte	MSG2, E0
		// Functional Utility: Copies ABCD to E1.
		movdqa		ABCD, E1
	// Functional Utility: Generates message words for W[32]-W[35].
	sha1msg2	MSG2, MSG3
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 24-27) using constants ($1).
		sha1rnds4	$1, E0, ABCD
	// Functional Utility: Generates message words.
	sha1msg1	MSG2, MSG1
	// Functional Utility: XORs MSG2 with MSG0 for message schedule W[36]-W[39].
	pxor		MSG2, MSG0

	/* Rounds 28-31 */
		// Functional Utility: Prepares for the next E value, now using MSG3 which holds W[28]-W[31].
		sha1nexte	MSG3, E1
		// Functional Utility: Copies ABCD to E0.
		movdqa		ABCD, E0
	// Functional Utility: Generates message words for W[36]-W[39].
	sha1msg2	MSG3, MSG0
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 28-31) using constants ($1).
		sha1rnds4	$1, E1, ABCD
	// Functional Utility: Generates message words.
	sha1msg1	MSG3, MSG2
	// Functional Utility: XORs MSG3 with MSG1 for message schedule W[40]-W[43].
	pxor		MSG3, MSG1

	/* Rounds 32-35 */
		// Functional Utility: Prepares for the next E value, now using MSG0 which holds W[32]-W[35].
		sha1nexte	MSG0, E0
		// Functional Utility: Copies ABCD to E1.
		movdqa		ABCD, E1
	// Functional Utility: Generates message words for W[40]-W[43].
	sha1msg2	MSG0, MSG1
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 32-35) using constants ($1).
		sha1rnds4	$1, E0, ABCD
	// Functional Utility: Generates message words.
	sha1msg1	MSG0, MSG3
	// Functional Utility: XORs MSG0 with MSG2 for message schedule W[44]-W[47].
	pxor		MSG0, MSG2

	/* Rounds 36-39 */
		// Functional Utility: Prepares for the next E value, now using MSG1 which holds W[36]-W[39].
		sha1nexte	MSG1, E1
		// Functional Utility: Copies ABCD to E0.
		movdqa		ABCD, E0
	// Functional Utility: Generates message words for W[44]-W[47].
	sha1msg2	MSG1, MSG2
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 36-39) using constants ($1).
		sha1rnds4	$1, E1, ABCD
	// Functional Utility: Generates message words.
	sha1msg1	MSG1, MSG0
	// Functional Utility: XORs MSG1 with MSG3 for message schedule W[48]-W[51].
	pxor		MSG1, MSG3

	/* Rounds 40-43 */
		// Functional Utility: Prepares for the next E value, now using MSG2 which holds W[40]-W[43].
		sha1nexte	MSG2, E0
		// Functional Utility: Copies ABCD to E1.
		movdqa		ABCD, E1
	// Functional Utility: Generates message words for W[48]-W[51].
	sha1msg2	MSG2, MSG3
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 40-43) using constants for rounds 40-59 ($2).
		sha1rnds4	$2, E0, ABCD
	// Functional Utility: Generates message words.
	sha1msg1	MSG2, MSG1
	// Functional Utility: XORs MSG2 with MSG0 for message schedule W[52]-W[55].
	pxor		MSG2, MSG0

	/* Rounds 44-47 */
		// Functional Utility: Prepares for the next E value, now using MSG3 which holds W[44]-W[47].
		sha1nexte	MSG3, E1
		// Functional Utility: Copies ABCD to E0.
		movdqa		ABCD, E0
	// Functional Utility: Generates message words for W[52]-W[55].
	sha1msg2	MSG3, MSG0
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 44-47) using constants ($2).
		sha1rnds4	$2, E1, ABCD
	// Functional Utility: Generates message words.
	sha1msg1	MSG3, MSG2
	// Functional Utility: XORs MSG3 with MSG1 for message schedule W[56]-W[59].
	pxor		MSG3, MSG1

	/* Rounds 48-51 */
		// Functional Utility: Prepares for the next E value, now using MSG0 which holds W[48]-W[51].
		sha1nexte	MSG0, E0
		// Functional Utility: Copies ABCD to E1.
		movdqa		ABCD, E1
	// Functional Utility: Generates message words for W[56]-W[59].
	sha1msg2	MSG0, MSG1
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 48-51) using constants ($2).
		sha1rnds4	$2, E0, ABCD
	// Functional Utility: Generates message words.
	sha1msg1	MSG0, MSG3
	// Functional Utility: XORs MSG0 with MSG2 for message schedule W[60]-W[63].
	pxor		MSG0, MSG2

	/* Rounds 52-55 */
		// Functional Utility: Prepares for the next E value, now using MSG1 which holds W[52]-W[55].
		sha1nexte	MSG1, E1
		// Functional Utility: Copies ABCD to E0.
		movdqa		ABCD, E0
	// Functional Utility: Generates message words for W[60]-W[63].
	sha1msg2	MSG1, MSG2
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 52-55) using constants ($2).
		sha1rnds4	$2, E1, ABCD
	// Functional Utility: Generates message words.
	sha1msg1	MSG1, MSG0
	// Functional Utility: XORs MSG1 with MSG3 for message schedule W[64]-W[67].
	pxor		MSG1, MSG3

	/* Rounds 56-59 */
		// Functional Utility: Prepares for the next E value, now using MSG2 which holds W[56]-W[59].
		sha1nexte	MSG2, E0
		// Functional Utility: Copies ABCD to E1.
		movdqa		ABCD, E1
	// Functional Utility: Generates message words for W[64]-W[67].
	sha1msg2	MSG2, MSG3
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 56-59) using constants ($2).
		sha1rnds4	$2, E0, ABCD
	// Functional Utility: Generates message words.
	sha1msg1	MSG2, MSG1
	// Functional Utility: XORs MSG2 with MSG0 for message schedule W[68]-W[71].
	pxor		MSG2, MSG0

	/* Rounds 60-63 */
		// Functional Utility: Prepares for the next E value, now using MSG3 which holds W[60]-W[63].
		sha1nexte	MSG3, E1
		// Functional Utility: Copies ABCD to E0.
		movdqa		ABCD, E0
	// Functional Utility: Generates message words for W[68]-W[71].
	sha1msg2	MSG3, MSG0
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 60-63) using constants for rounds 60-79 ($3).
		sha1rnds4	$3, E1, ABCD
	// Functional Utility: Generates message words.
	sha1msg1	MSG3, MSG2
	// Functional Utility: XORs MSG3 with MSG1 for message schedule W[72]-W[75].
	pxor		MSG3, MSG1

	/* Rounds 64-67 */
		// Functional Utility: Prepares for the next E value, now using MSG0 which holds W[64]-W[67].
		sha1nexte	MSG0, E0
		// Functional Utility: Copies ABCD to E1.
		movdqa		ABCD, E1
	// Functional Utility: Generates message words for W[72]-W[75].
	sha1msg2	MSG0, MSG1
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 64-67) using constants ($3).
		sha1rnds4	$3, E0, ABCD
	// Functional Utility: Generates message words.
	sha1msg1	MSG0, MSG3
	// Functional Utility: XORs MSG0 with MSG2 for message schedule W[76]-W[79].
	pxor		MSG0, MSG2

	/* Rounds 68-71 */
		// Functional Utility: Prepares for the next E value, now using MSG1 which holds W[68]-W[71].
		sha1nexte	MSG1, E1
		// Functional Utility: Copies ABCD to E0.
		movdqa		ABCD, E0
	// Functional Utility: Generates message words for W[76]-W[79].
	sha1msg2	MSG1, MSG2
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 68-71) using constants ($3).
		sha1rnds4	$3, E1, ABCD
	// Functional Utility: XORs MSG1 with MSG3 for message schedule W[80]-W[83].
	pxor		MSG1, MSG3

	/* Rounds 72-75 */
		// Functional Utility: Prepares for the next E value, now using MSG2 which holds W[72]-W[75].
		sha1nexte	MSG2, E0
		// Functional Utility: Copies ABCD to E1.
		movdqa		ABCD, E1
	// Functional Utility: Generates message words for W[80]-W[83].
	sha1msg2	MSG2, MSG3
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 72-75) using constants ($3).
		sha1rnds4	$3, E0, ABCD

	/* Rounds 76-79 */
		// Functional Utility: Prepares for the next E value, now using MSG3 which holds W[76]-W[79].
		sha1nexte	MSG3, E1
		// Functional Utility: Copies ABCD to E0.
		movdqa		ABCD, E0
		// Functional Utility: Performs 4 rounds of SHA-1 compression (rounds 76-79) using constants ($3).
		sha1rnds4	$3, E1, ABCD

	/* Add current hash values with previously saved */
	// Functional Utility: Adds the final E0 hash component to the initial E value saved on the stack.
	sha1nexte	(0*16)(%rsp), E0
	// Functional Utility: Adds the final ABCD hash components to the initial ABCD values saved on the stack.
	paddd		(1*16)(%rsp), ABCD

	/* Increment data pointer and loop if more to process */
	// Functional Utility: Advances the DATA_PTR by 64 bytes to the next block.
	add		$64, DATA_PTR
	// Functional Utility: Compares the current DATA_PTR with the end-of-data pointer (NUM_BLKS).
	cmp		NUM_BLKS, DATA_PTR
	// Functional Utility: Jumps back to the start of the loop (`.Lloop0`) if more blocks are to be processed.
	jne		.Lloop0

	/* Write hash values back in the correct order */
	// Functional Utility: Permutes the ABCD register back to its original order for storing.
	pshufd		$0x1B, ABCD, ABCD
	// Functional Utility: Stores the final ABCD hash components back to `DIGEST_PTR`.
	movdqu		ABCD, 0*16(DIGEST_PTR)
	// Functional Utility: Extracts the final E component from E0 and stores it to `DIGEST_PTR + 16`.
	pextrd		$3, E0, 1*16(DIGEST_PTR)

.Ldone_hash: /* Functional Role: Epilogue for the function. */
	// Functional Utility: Restores the stack pointer.
	mov		%rbp, %rsp
	// Functional Utility: Restores the base pointer.
	pop		%rbp

	RET /* Functional Utility: Returns from the function. */
SYM_FUNC_END(sha1_ni_transform)

.section	.rodata.cst16.PSHUFFLE_BYTE_FLIP_MASK, "aM", @progbits, 16
.align 16
/**
 * @brief Shuffle mask for byte-swapping DWORDS in XMM registers.
 * @details This 16-byte constant provides the control mask for the `pshufb`
 * instruction to perform byte-swapping on four 32-bit words within an XMM
 * register. This is used to convert input data from little-endian to
 * big-endian, as required by the SHA-1 algorithm.
 * Functional Role: Defines a pattern for byte-reversal of 32-bit words within an XMM register.
 */
PSHUFFLE_BYTE_FLIP_MASK:
	.octa 0x000102030405060708090a0b0c0d0e0f

.section	.rodata.cst16.UPPER_WORD_MASK, "aM", @progbits, 16
.align 16
/**
 * @brief Mask to zero out the upper words of an XMM register.
 * @details This 16-byte constant contains a pattern of `0xFFFFFFFF000000000000000000000000`
 * which, when used with bitwise AND operations (e.g., `pand`), effectively
 * masks off the upper 96 bits of an XMM register, isolating the lower 32-bit
 * word. This is specifically used to initialize or manipulate the SHA-1 E state
 * component when it's stored in an XMM register.
 * Functional Role: Provides a mask to isolate the lower 32-bit word in an XMM register.
 */
UPPER_WORD_MASK:
	.octa 0xFFFFFFFF000000000000000000000000
