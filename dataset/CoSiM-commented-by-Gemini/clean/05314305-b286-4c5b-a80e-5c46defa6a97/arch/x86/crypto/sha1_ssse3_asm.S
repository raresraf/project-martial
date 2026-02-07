/* SPDX-License-Identifier: GPL-2.0-or-later */
/**
 * @file sha1_ssse3_asm.S
 * @brief High-performance SHA-1 implementation leveraging Intel SSSE3 and AVX instructions for x86.
 * @details This assembly file provides a highly optimized implementation of the
 * SHA-1 (Secure Hash Algorithm 1) compression function, specifically tailored
 * for x86 processors equipped with Intel Supplemental SSE3 (SSSE3) and
 * Advanced Vector Extensions (AVX/AVX2) instruction sets. It employs
 * advanced vectorization techniques for message scheduling and round
 * computations, significantly accelerating the hashing process.
 *
 * **Optimization Techniques:**
 * - **SSSE3/AVX Vectorization**: Utilizes instructions like `pshufb` (SSSE3),
 *   `vmovdqu`, `vpshufb`, `vpaddd`, `vpxor`, `vpalignr`, `vpslld`, `vpsrld`, `vpor` (AVX)
 *   for parallel processing of data.
 * - **Pipelined Message Scheduling**: Message schedule words are pre-calculated
 *   and stored in a circular buffer (`WK`, `W_PRECALC_AHEAD`) to minimize
 *   dependencies and maximize throughput.
 * - **Alternative Message Expansion**: For later rounds (32-79), an alternative
 *   message word expansion formula (`w[i] = (w[i-6] ^ w[i-16] ^ w[i-28] ^ w[i-32]) rol 2`)
 *   is used to break dependencies and enable more efficient vectorization.
 * - **Blended Scalar and Vector Operations**: Carefully combines scalar (ALU)
 *   and vector (XMM/YMM) instructions to optimize various stages of the SHA-1 algorithm.
 * - **Register Renaming/Swapping**: Macros (`INIT_REGALLOC`, `RESTORE_RENAMED_REGS`,
 *   `SWAP_REG_NAMES`) manage register aliases to facilitate state rotation without
 *   physical data movement.
 * This implementation is designed for maximum SHA-1 hashing performance on
 * SSSE3/AVX-enabled x86 systems within the Linux kernel, building upon previous
 * vectorized approaches.
 */
/*
 * This is a SIMD SHA-1 implementation. It requires the Intel(R) Supplemental
 * SSE3 instruction set extensions introduced in Intel Core Microarchitecture
 * processors. CPUs supporting Intel(R) AVX extensions will get an additional
 * boost.
 *
 * This work was inspired by the vectorized implementation of Dean Gaudet.
 * Additional information on it can be found at:
 *    http://www.arctic.org/~dean/crypto/sha1.html
 *
 * It was improved upon with more efficient vectorization of the message
 * scheduling. This implementation has also been optimized for all current and
 * several future generations of Intel CPUs.
 *
 * See this article for more information about the implementation details:
 *   http://software.intel.com/en-us/articles/improving-the-performance-of-the-secure-hash-algorithm-1/
 *
 * Copyright (C) 2010, Intel Corp.
 *   Authors: Maxim Locktyukhin <maxim.locktyukhin@intel.com>
 *            Ronen Zohar <ronen.zohar@intel.com>
 *
 * Converted to AT&T syntax and adapted for inclusion in the Linux kernel:
 *   Author: Mathias Krause <minipli@googlemail.com>
 */

#include <linux/linkage.h>
#include <linux/cfi_types.h>

#define CTX	%rdi	// Functional Role: Pointer to `struct sha1_state` (argument 1).
#define BUF	%rsi	// Functional Role: Pointer to input data buffer (argument 2).
#define CNT	%rdx	// Functional Role: Number of 64-byte blocks to process (argument 3).

#define REG_A	%ecx	// Functional Role: General-purpose register (GPR) for SHA-1 state A (32-bit).
#define REG_B	%esi	// Functional Role: GPR for SHA-1 state B (32-bit).
#define REG_C	%edi	// Functional Role: GPR for SHA-1 state C (32-bit).
#define REG_D	%r12d	// Functional Role: GPR for SHA-1 state D (32-bit).
#define REG_E	%edx	// Functional Role: GPR for SHA-1 state E (32-bit).

#define REG_T1	%eax	// Functional Role: GPR for general temporary use (32-bit).
#define REG_T2	%ebx	// Functional Role: GPR for general temporary use (32-bit).

#define K_BASE		%r8	// Functional Role: GPR holding pointer to the SHA-1 round constants table (`K_XMM_AR`).
#define HASH_PTR	%r9	// Functional Role: GPR holding pointer to the SHA-1 hash context.
#define BUFFER_PTR	%r10	// Functional Role: GPR holding pointer to the current input data buffer.
#define BUFFER_END	%r11	// Functional Role: GPR holding pointer to the end of the input data buffer.

#define W_TMP1	%xmm0	// Functional Role: XMM register for temporary message word storage.
#define W_TMP2	%xmm9	// Functional Role: XMM register for temporary message word storage.

#define W0	%xmm1	// Functional Role: XMM register for message schedule words W[0]-W[3] (or rotating alias).
#define W4	%xmm2	// Functional Role: XMM register for message schedule words W[4]-W[7] (or rotating alias).
#define W8	%xmm3	// Functional Role: XMM register for message schedule words W[8]-W[11] (or rotating alias).
#define W12	%xmm4	// Functional Role: XMM register for message schedule words W[12]-W[15] (or rotating alias).
#define W16	%xmm5	// Functional Role: XMM register for message schedule words W[16]-W[19] (or rotating alias).
#define W20	%xmm6	// Functional Role: XMM register for message schedule words W[20]-W[23] (or rotating alias).
#define W24	%xmm7	// Functional Role: XMM register for message schedule words W[24]-W[27] (or rotating alias).
#define W28	%xmm8	// Functional Role: XMM register for message schedule words W[28]-W[31] (or rotating alias).

#define XMM_SHUFB_BSWAP	%xmm10	// Functional Role: XMM register holding the shuffle control mask for byte-swapping.

/* we keep window of 64 w[i]+K pre-calculated values in a circular buffer */
#define WK(t)	(((t) & 15) * 4)(%rsp) /* Functional Role: Macro to calculate offset within the stack-based circular buffer for `W[t]` or `K+W[t]`. */
#define W_PRECALC_AHEAD	16 /* Functional Role: Number of message words to pre-calculate ahead of the current round to enable pipelining. */

/*
 * This macro implements the SHA-1 function's body for single 64-byte block
 * param: function's name
 */
/**
 * @brief Macro to define the `sha1_transform_ssse3` or `sha1_transform_avx` function.
 * @details This macro wraps the `SHA1_PIPELINED_MAIN_BODY` with standard
 * function prologue and epilogue, including saving/restoring registers,
 * stack alignment, and setting up initial values for the SHA-1 computation.
 * It is used for both SSSE3 and AVX optimized implementations.
 * @param name The name of the function to define (e.g., `sha1_transform_ssse3`).
 * Functional Utility: Defines the SSSE3/AVX-optimized SHA-1 transform function, handling function call conventions.
 */
.macro SHA1_VECTOR_ASM  name
	SYM_TYPED_FUNC_START(
ame)

	// Functional Utility: Saves callee-saved registers as per x86_64 ABI.
	push	%rbx
	push	%r12
	push	%rbp
	// Functional Utility: Saves the current stack pointer to %rbp for stack frame management.
	mov	%rsp, %rbp

	// Functional Utility: Allocates 64 bytes of workspace on the stack.
	sub	$64, %rsp
	// Functional Utility: Aligns the stack pointer to a 16-byte boundary, crucial for XMM operations.
	and	$~15, %rsp

	// Functional Utility: Moves argument registers to internal aliases.
	mov	CTX, HASH_PTR
	mov	BUF, BUFFER_PTR

	// Functional Utility: Converts the number of blocks (`CNT`) to total bytes by multiplying by 64.
	shl	$6, CNT
	// Functional Utility: Calculates the absolute address of the end of the input buffer.
	add	BUF, CNT
	mov	CNT, BUFFER_END

	// Functional Utility: Loads the base address of the SHA-1 round constants table into `K_BASE`.
	lea	K_XMM_AR(%rip), K_BASE
	// Functional Utility: Loads the byte-flip shuffle mask into `XMM_SHUFB_BSWAP` for endian conversion.
	xmm_mov	BSWAP_SHUFB_CTL(%rip), XMM_SHUFB_BSWAP

	SHA1_PIPELINED_MAIN_BODY /* Functional Utility: Executes the main pipelined SHA-1 processing logic. */

	# cleanup workspace
	mov	$8, %ecx /* Functional Utility: Sets loop counter for clearing stack workspace. */
	mov	%rsp, %rdi /* Functional Utility: Sets destination pointer to the start of the workspace. */
	xor	%eax, %eax /* Functional Utility: Zeros %eax, which will be written to memory. */
	rep stosq /* Functional Utility: Clears the allocated stack workspace by writing zeros. */

	// Functional Utility: Restores the stack pointer.
	mov	%rbp, %rsp
	// Functional Utility: Restores the base pointer.
	pop	%rbp
	// Functional Utility: Restores callee-saved registers.
	pop	%r12
	pop	%rbx
	RET /* Functional Utility: Returns from the function. */

	SYM_FUNC_END(
ame)
.endm

/*
 * This macro implements 80 rounds of SHA-1 for one 64-byte block
 */
/**
 * @brief Main body macro for pipelined SHA-1 processing of a single 64-byte block.
 * @details This macro encapsulates the core logic for processing one 64-byte
 * SHA-1 block using software pipelining. It manages the SHA-1 hash state (A, B, C, D, E),
 * message schedule generation (via `W_PRECALC`), and round computations (via `RR` macro).
 * The processing is broken down into distinct blocks of rounds, and hash
 * values are updated at the end.
 * Functional Utility: Orchestrates the pipelined execution of SHA-1 compression for a single block using SSSE3/AVX.
 */
.macro SHA1_PIPELINED_MAIN_BODY
	INIT_REGALLOC /* Functional Utility: Initializes register aliases for working registers. */

	// Functional Utility: Loads the initial SHA-1 state (H0-H4) from `HASH_PTR` into GPRs A, B, C, D, E.
	mov	  (HASH_PTR), A
	mov	 4(HASH_PTR), B
	mov	 8(HASH_PTR), C
	mov	12(HASH_PTR), D
	mov	16(HASH_PTR), E

  .set i, 0
  .rept W_PRECALC_AHEAD /* Functional Utility: Loop to pre-calculate `W_PRECALC_AHEAD` message words. */
	W_PRECALC i /* Functional Utility: Dispatches to message schedule pre-calculation. */
    .set i, (i+1)
  .endr

.align 4
1: /* Functional Role: Main loop for processing each 64-byte block. */
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F1 and rotates state (rounds 0-1).
	RR F1,A,B,C,D,E,0
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F1 and rotates state (rounds 2-3).
	RR F1,D,E,A,B,C,2
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F1 and rotates state (rounds 4-5).
	RR F1,B,C,D,E,A,4
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F1 and rotates state (rounds 6-7).
	RR F1,E,A,B,C,D,6
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F1 and rotates state (rounds 8-9).
	RR F1,C,D,E,A,B,8

	// Functional Utility: Executes 2 rounds of SHA-1 using round function F1 and rotates state (rounds 10-11).
	RR F1,A,B,C,D,E,10
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F1 and rotates state (rounds 12-13).
	RR F1,D,E,A,B,C,12
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F1 and rotates state (rounds 14-15).
	RR F1,B,C,D,E,A,14
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F1 and rotates state (rounds 16-17).
	RR F1,E,A,B,C,D,16
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F1 and rotates state (rounds 18-19).
	RR F1,C,D,E,A,B,18

	// Functional Utility: Executes 2 rounds of SHA-1 using round function F2 and rotates state (rounds 20-21).
	RR F2,A,B,C,D,E,20
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F2 and rotates state (rounds 22-23).
	RR F2,D,E,A,B,C,22
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F2 and rotates state (rounds 24-25).
	RR F2,B,C,D,E,A,24
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F2 and rotates state (rounds 26-27).
	RR F2,E,A,B,C,D,26
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F2 and rotates state (rounds 28-29).
	RR F2,C,D,E,A,B,28

	// Functional Utility: Executes 2 rounds of SHA-1 using round function F2 and rotates state (rounds 30-31).
	RR F2,A,B,C,D,E,30
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F2 and rotates state (rounds 32-33).
	RR F2,D,E,A,B,C,32
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F2 and rotates state (rounds 34-35).
	RR F2,B,C,D,E,A,34
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F2 and rotates state (rounds 36-37).
	RR F2,E,A,B,C,D,36
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F2 and rotates state (rounds 38-39).
	RR F2,C,D,E,A,B,38

	// Functional Utility: Executes 2 rounds of SHA-1 using round function F3 and rotates state (rounds 40-41).
	RR F3,A,B,C,D,E,40
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F3 and rotates state (rounds 42-43).
	RR F3,D,E,A,B,C,42
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F3 and rotates state (rounds 44-45).
	RR F3,B,C,D,E,A,44
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F3 and rotates state (rounds 46-47).
	RR F3,E,A,B,C,D,46
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F3 and rotates state (rounds 48-49).
	RR F3,C,D,E,A,B,48

	// Functional Utility: Executes 2 rounds of SHA-1 using round function F3 and rotates state (rounds 50-51).
	RR F3,A,B,C,D,E,50
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F3 and rotates state (rounds 52-53).
	RR F3,D,E,A,B,C,52
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F3 and rotates state (rounds 54-55).
	RR F3,B,C,D,E,A,54
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F3 and rotates state (rounds 56-57).
	RR F3,E,A,B,C,D,56
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F3 and rotates state (rounds 58-59).
	RR F3,C,D,E,A,B,58

	add	$64, BUFFER_PTR		# move to the next 64-byte block
	// Functional Utility: Compares `BUFFER_PTR` with `BUFFER_END` to check if the current block is the last one.
	cmp	BUFFER_END, BUFFER_PTR
	// Functional Utility: If `BUFFER_PTR` is greater than or equal to `BUFFER_END` (last block or beyond),
	// it uses a dummy source (K_BASE) to avoid buffer overrun. This is a trick to signal the end of blocks.
	cmovae	K_BASE, BUFFER_PTR

	// Functional Utility: Executes 2 rounds of SHA-1 using round function F4 (which is F2) and rotates state (rounds 60-61).
	RR F4,A,B,C,D,E,60
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F4 and rotates state (rounds 62-63).
	RR F4,D,E,A,B,C,62
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F4 and rotates state (rounds 64-65).
	RR F4,B,C,D,E,A,64
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F4 and rotates state (rounds 66-67).
	RR F4,E,A,B,C,D,66
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F4 and rotates state (rounds 68-69).
	RR F4,C,D,E,A,B,68

	// Functional Utility: Executes 2 rounds of SHA-1 using round function F4 and rotates state (rounds 70-71).
	RR F4,A,B,C,D,E,70
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F4 and rotates state (rounds 72-73).
	RR F4,D,E,A,B,C,72
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F4 and rotates state (rounds 74-75).
	RR F4,B,C,D,E,A,74
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F4 and rotates state (rounds 76-77).
	RR F4,E,A,B,C,D,76
	// Functional Utility: Executes 2 rounds of SHA-1 using round function F4 and rotates state (rounds 78-79).
	RR F4,C,D,E,A,B,78

	// Functional Utility: Updates the main hash state registers in memory (`HASH_PTR`) with the accumulated results.
	UPDATE_HASH   (HASH_PTR), A
	UPDATE_HASH  4(HASH_PTR), B
	UPDATE_HASH  8(HASH_PTR), C
	UPDATE_HASH 12(HASH_PTR), D
	UPDATE_HASH 16(HASH_PTR), E

	RESTORE_RENAMED_REGS // Functional Utility: Restores the original register names after state rotation.
	// Functional Utility: Compares `K_BASE` with `BUFFER_PTR`. If they are not equal, it means there are more blocks to process.
	cmp	K_BASE, BUFFER_PTR
	// Functional Utility: Jumps back to `1b` if more blocks remain to be processed.
	jne	1b
.endm

/**
 * @brief Initializes register aliases for working registers.
 * @details This macro sets up aliases for the general-purpose registers
 * used to hold the SHA-1 state variables (A, B, C, D, E) and temporary
 * values (T1, T2). This ensures consistency and readability throughout
 * the SHA-1 computation logic.
 * Functional Utility: Assigns symbolic names to general-purpose registers used in SHA-1 processing.
 */
.macro INIT_REGALLOC
  .set A, REG_A
  .set B, REG_B
  .set C, REG_C
  .set D, REG_D
  .set E, REG_E
  .set T1, REG_T1
  .set T2, REG_T2
.endm

/**
 * @brief Restores original register names after permutation.
 * @details This macro is used in the epilogue of a processing block to map
 * the dynamically rotated/renamed registers back to their canonical A,B,C,D,E
 * roles before the final hash update. The order of moves is important to
 * correctly re-establish the original assignment.
 * Functional Utility: Re-establishes the standard naming convention for SHA-1 state registers after internal permutations.
 */
.macro RESTORE_RENAMED_REGS
	# order is important (REG_C is where it should be)
	mov	B, REG_B // Functional Utility: Moves current B to REG_B (original B).
	mov	D, REG_D // Functional Utility: Moves current D to REG_D (original D).
	mov	A, REG_A // Functional Utility: Moves current A to REG_A (original A).
	mov	E, REG_E // Functional Utility: Moves current E to REG_E (original E).
	mov	C, REG_C // Functional Utility: Moves current C to REG_C (original C).
.endm

/**
 * @brief Swaps the aliases of two registers.
 * @details This macro redefines the symbolic names of two registers,
 * effectively swapping their aliases. This is a purely symbolic operation
 * at assembly time, used for logical register rotation without generating
 * any actual move instructions.
 * @param a The first register alias.
 * @param b The second register alias.
 * Functional Utility: Facilitates logical register rotation by swapping register aliases during assembly.
 */
.macro SWAP_REG_NAMES  a, b
  .set _T, \a
  .set \a, \b
  .set \b, _T
.endm

/**
 * @brief Computes SHA-1 round function F1: `(B & C) | (~B & D)`.
 * @details This macro implements the SHA-1 logical function F1 (Choose) used
 * in rounds 0-19. It uses bitwise operations (`mov`, `xor`, `and`) to
 * compute `(B & C) | (~B & D)` and stores the result in T1.
 * @param b Register holding SHA-1 state B.
 * @param c Register holding SHA-1 state C.
 * @param d Register holding SHA-1 state D.
 * Functional Utility: Calculates the SHA-1 F1 (Choose) logical function.
 */
.macro F1  b, c, d
	mov	\c, T1 // Functional Utility: Copies C to T1.
	SWAP_REG_NAMES \c, T1 // Functional Utility: Swaps alias of C and T1 (symbolic, T1 now refers to original C).
	xor	\d, T1 // Functional Utility: Computes C XOR D.
	and	\b, T1 // Functional Utility: Computes B AND (C XOR D).
	xor	\d, T1 // Functional Utility: Computes D XOR (B AND (C XOR D)), which is (B & C) | (~B & D).
.endm

/**
 * @brief Computes SHA-1 round function F2: `B ^ C ^ D`.
 * @details This macro implements the SHA-1 logical function F2 (Parity) used
 * in rounds 20-39 and 60-79. It uses bitwise XOR operations to compute `B ^ C ^ D`
 * and stores the result in T1.
 * @param b Register holding SHA-1 state B.
 * @param c Register holding SHA-1 state C.
 * @param d Register holding SHA-1 state D.
 * Functional Utility: Calculates the SHA-1 F2 (Parity) logical function.
 */
.macro F2  b, c, d
	mov	\d, T1 // Functional Utility: Copies D to T1.
	SWAP_REG_NAMES \d, T1 // Functional Utility: Swaps alias of D and T1 (symbolic).
	xor	\c, T1 // Functional Utility: Computes D XOR C.
	xor	\b, T1 // Functional Utility: Computes (D XOR C) XOR B.
.endm

/**
 * @brief Computes SHA-1 round function F3: `(B & C) | (B & D) | (C & D)`.
 * @details This macro implements the SHA-1 logical function F3 (Majority) used
 * in rounds 40-59. It uses bitwise operations (`mov`, `or`, `and`) to
 * compute `(B & C) | (B & D) | (C & D)` and stores the result in T1.
 * @param b Register holding SHA-1 state B.
 * @param c Register holding SHA-1 state C.
 * @param d Register holding SHA-1 state D.
 * Functional Utility: Calculates the SHA-1 F3 (Majority) logical function.
 */
.macro F3  b, c ,d
	mov	\c, T1 // Functional Utility: Copies C to T1.
	SWAP_REG_NAMES \c, T1 // Functional Utility: Swaps alias of C and T1 (symbolic).
	mov	\b, T2 // Functional Utility: Copies B to T2.
	or	\b, T1 // Functional Utility: Computes C OR B.
	and	\c, T2 // Functional Utility: Computes B AND C.
	and	\d, T1 // Functional Utility: Computes D AND (C OR B).
	or	T2, T1 // Functional Utility: Computes (B AND C) OR (D AND (C OR B)), which is (B & C) | (B & D) | (C & D).
.endm

/**
 * @brief Computes SHA-1 round function F4, which is equivalent to F2.
 * @details This macro simply calls the F2 macro, as SHA-1 round function F4
 * (used in rounds 60-79) is identical to F2.
 * @param b Register holding SHA-1 state B.
 * @param c Register holding SHA-1 state C.
 * @param d Register holding SHA-1 state D.
 * Functional Utility: Delegates to F2 as the logical function for SHA-1 rounds 60-79.
 */
.macro F4  b, c, d
	F2 \b, \c, \d
.endm

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
	add	\hash, \val // Functional Utility: Adds `val` to the content of `hash`.
	mov	\val, \hash // Functional Utility: Moves the result from `val` back into `hash`.
.endm

/*
 * RR does two rounds of SHA-1 back to back with W[] pre-calc
 *   t1 = F(b, c, d);   e += w(i)
 *   e += t1;           b <<= 30;   d  += w(i+1);
 *   t1 = F(a, b, c);
 *   d += t1;           a <<= 5;
 *   e += a;
 *   t1 = e;            a >>= 7;
 *   t1 <<= 5;
 *   d += t1;
 */
/**
 * @brief Executes two SHA-1 rounds and manages state rotation.
 * @details This macro performs two consecutive rounds of the SHA-1 algorithm,
 * including loading message words (`WK`), computing the round function (`F`),
 * performing bitwise rotations (`rol`, `ror`), and updating the hash state
 * registers (`a`, `b`, `c`, `d`, `e`). It also handles pre-calculation of
 * message words for pipelining.
 * @param F The round function macro to use (F1, F2, F3, F4).
 * @param a,b,c,d,e Registers holding SHA-1 state variables.
 * @param round The current round number.
 * Functional Utility: Executes a pair of SHA-1 rounds, updates the hash state, and prepares for future rounds.
 */
.macro RR  F, a, b, c, d, e, round
	add	WK(ound), \e // Functional Utility: Adds the pre-calculated `K+W` for the current round to E.
	\F   \b, \c, \d		// t1 = F(b, c, d); Functional Utility: Computes the round function F(b, c, d), storing result in T1.
	W_PRECALC (ound + W_PRECALC_AHEAD) // Functional Utility: Dispatches to pre-calculate message words for pipelining.
	rol	$30, \b // Functional Utility: Rotates B left by 30 bits.
	add	T1, \e // Functional Utility: Adds T1 (result of F) to E.
	add	WK(ound + 1), \d // Functional Utility: Adds the pre-calculated `K+W` for the next round (round+1) to D.

	\F   \a, \b, \c // Functional Utility: Computes the round function F(a, b, c), storing result in T1.
	W_PRECALC (ound + W_PRECALC_AHEAD + 1) // Functional Utility: Dispatches to pre-calculate message words for pipelining.
	rol	$5, \a // Functional Utility: Rotates A left by 5 bits.
	add	\a, \e // Functional Utility: Adds rotated A to E.
	add	T1, \d // Functional Utility: Adds T1 (result of F) to D.
	ror	$7, \a		# (a <<r 5) >>r 7) => a <<r 30) // Functional Utility: Rotates A right by 7 bits, effectively performing a 30-bit left rotation.

	mov	\e, T1 // Functional Utility: Copies E to T1.
	SWAP_REG_NAMES \e, T1 // Functional Utility: Swaps alias of E and T1 (symbolic).

	rol	$5, T1 // Functional Utility: Rotates T1 (which held E) left by 5 bits.
	add	T1, \d // Functional Utility: Adds rotated T1 to D.

	# write:  \a, \b
	# rotate: \a<=\d, \b<=\e, \c<=\a, \d<=\b, \e<=\c
.endm

/**
 * @brief Dispatcher macro for message schedule pre-calculation based on round number and instruction set.
 * @details This macro selects the appropriate `W_PRECALC_*` macro based on the
 * current round number (`i`) and implicitly the chosen instruction set (SSSE3 or AVX).
 * It also sets the base offset for `K_XMM` to load the correct SHA-1 round constant.
 * @param r The current round number (0-79).
 * Functional Utility: Directs message schedule pre-calculation to the correct specialized macro.
 */
.macro W_PRECALC  r
  .set i, 

  .if (i < 20) /* Functional Utility: Sets `K_XMM` offset for rounds 0-19. */
    .set K_XMM, 0
  .elseif (i < 40) /* Functional Utility: Sets `K_XMM` offset for rounds 20-39. */
    .set K_XMM, 16
  .elseif (i < 60) /* Functional Utility: Sets `K_XMM` offset for rounds 40-59. */
    .set K_XMM, 32
  .elseif (i < 80) /* Functional Utility: Sets `K_XMM` offset for rounds 60-79. */
    .set K_XMM, 48
  .endif

  .if ((i < 16) || ((i >= 80) && (i < (80 + W_PRECALC_AHEAD)))) // Functional Utility: For initial rounds (0-15) and pipelined future rounds.
    .set i, (() % 80)	    # pre-compute for the next iteration
    .if (i == 0) // Functional Utility: Resets circular buffer pointers at the start of each 80-round block.
	W_PRECALC_RESET
    .endif
	W_PRECALC_00_15 // Functional Utility: Calls SSSE3/AVX specific macro for rounds 0-15.
  .elseif (i<32) // Functional Utility: For rounds 16-31.
	W_PRECALC_16_31 // Functional Utility: Calls SSSE3/AVX specific macro for rounds 16-31.
  .elseif (i < 80)   // rounds 32-79 // Functional Utility: For rounds 32-79.
	W_PRECALC_32_79 // Functional Utility: Calls SSSE3/AVX specific macro for rounds 32-79.
  .endif
.endm

/**
 * @brief Resets aliases for XMM registers to their initial configuration for message scheduling.
 * @details This macro defines aliases `W` through `W_minus_32` to specific
 * XMM registers (`W0` through `W28`). This is used to logically reset the
 * message word "window" before starting a new sequence of message schedule
 * generation, ensuring consistent register assignment for the circular buffer.
 * Functional Utility: Initializes the logical mapping of XMM registers for message schedule generation.
 */
.macro W_PRECALC_RESET
  .set W,          W0
  .set W_minus_04, W4
  .set W_minus_08, W8
  .set W_minus_12, W12
  .set W_minus_16, W16
  .set W_minus_20, W20
  .set W_minus_24, W24
  .set W_minus_28, W28
  .set W_minus_32, W
.endm

/**
 * @brief Rotates the aliases for XMM registers used in message scheduling.
 * @details This macro performs a logical rotation of the XMM register aliases.
 * This effectively shifts the "window" of message words (`W` to `W_minus_32`)
 * for the next set of computations without physically moving data between
 * registers. This is a key part of the pipelined message schedule generation.
 * Functional Utility: Logically shifts the XMM register aliases to process subsequent message schedule words.
 */
.macro W_PRECALC_ROTATE
  .set W_minus_32, W_minus_28
  .set W_minus_28, W_minus_24
  .set W_minus_24, W_minus_20
  .set W_minus_20, W_minus_16
  .set W_minus_16, W_minus_12
  .set W_minus_12, W_minus_08
  .set W_minus_08, W_minus_04
  .set W_minus_04, W
  .set W,          W_minus_32
.endm

/**
 * @brief Umbrella macro for SSSE3-specific message schedule pre-calculation macros.
 * @details This macro defines the specific `W_PRECALC_*` macros to be used
 * when compiling for SSSE3 instruction set extensions. It provides the SSSE3-optimized
 * implementations for generating message schedule words and combining them with
 * round constants.
 * Functional Utility: Groups SSSE3-optimized message schedule pre-calculation macros.
 */
.macro W_PRECALC_SSSE3

.macro W_PRECALC_00_15
	W_PRECALC_00_15_SSSE3
.endm
.macro W_PRECALC_16_31
	W_PRECALC_16_31_SSSE3
.endm
.macro W_PRECALC_32_79
	W_PRECALC_32_79_SSSE3
.endm

/* message scheduling pre-compute for rounds 0-15 */
/**
 * @brief Pre-calculates SHA-1 message schedule words and K+W for rounds 0-15 (SSSE3).
 * @details This macro implements the message schedule generation (`W[i]`) and
 * the `K+W[i]` values for rounds 0-15 of SHA-1, optimized for SSSE3. It loads
 * input data, performs byte-swapping (`pshufb`), adds round constants, and
 * stores the result.
 * Functional Utility: Generates SHA-1 message schedule words and K+W values for initial rounds with SSSE3.
 */
.macro W_PRECALC_00_15_SSSE3
  .if ((i & 3) == 0) // Functional Utility: For every 4th round (0, 4, 8, 12) - loads 16 bytes.
	movdqu	(i*4)(BUFFER_PTR), W_TMP1 // Functional Utility: Loads 16 bytes (4 DWORDS) of message data into W_TMP1.
  .elseif ((i & 3) == 1) // Functional Utility: For every (4k+1)th round.
	pshufb	XMM_SHUFB_BSWAP, W_TMP1 // Functional Utility: Performs byte-swapping on W_TMP1 for big-endian.
	movdqa	W_TMP1, W // Functional Utility: Moves byte-swapped data to W.
  .elseif ((i & 3) == 2) // Functional Utility: For every (4k+2)th round.
	paddd	(K_BASE), W_TMP1 // Functional Utility: Adds round constant from K_BASE to W_TMP1.
  .elseif ((i & 3) == 3) // Functional Utility: For every (4k+3)th round.
	movdqa  W_TMP1, WK(i&~3) // Functional Utility: Stores the pre-calculated K+W value.
	W_PRECALC_ROTATE // Functional Utility: Rotates XMM register aliases.
  .endif
.endm

/* message scheduling pre-compute for rounds 16-31
 *
 * - calculating last 32 w[i] values in 8 XMM registers
 * - pre-calculate K+w[i] values and store to mem, for later load by ALU add
 *   instruction
 *
 * some "heavy-lifting" vectorization for rounds 16-31 due to w[i]->w[i-3]
 * dependency, but improves for 32-79
 */
/**
 * @brief Pre-calculates SHA-1 message schedule words and K+W for rounds 16-31 (SSSE3).
 * @details This macro generates message schedule words (`W[i]`) and `K+W[i]` values
 * for rounds 16-31, optimized for SSSE3. It handles the specific SHA-1 message
 * expansion formula for these rounds, including the `W[i-3]` dependency,
 * using instructions like `palignr`, `psrldq`, `pxor`, `pslldq`, `pslld`, `psrld`, `por`.
 * Functional Utility: Generates SHA-1 message schedule words and K+W values for middle rounds (16-31) with SSSE3.
 */
.macro W_PRECALC_16_31_SSSE3
  # blended scheduling of vector and scalar instruction streams, one 4-wide
  # vector iteration / 4 scalar rounds
  .if ((i & 3) == 0) // Functional Utility: For every 4th round (16, 20, etc.) - first step in message expansion.
	movdqa	W_minus_12, W // Functional Utility: Moves W[i-12] into W.
	palignr	$8, W_minus_16, W	# w[i-14] // Functional Utility: Aligns and combines W[i-16] and W[i-12] to get W[i-14].
	movdqa	W_minus_04, W_TMP1 // Functional Utility: Moves W[i-4] into W_TMP1.
	psrldq	$4, W_TMP1		# w[i-3] // Functional Utility: Shifts W_TMP1 to get W[i-3].
	pxor	W_minus_08, W // Functional Utility: XORs W[i-8] with W (W[i-14]).
  .elseif ((i & 3) == 1) // Functional Utility: For every (4k+1)th round.
	pxor	W_minus_16, W_TMP1 // Functional Utility: XORs W[i-16] with W_TMP1 (W[i-3]).
	pxor	W_TMP1, W // Functional Utility: XORs W_TMP1 with W.
	movdqa	W, W_TMP2 // Functional Utility: Copies W to W_TMP2.
	movdqa	W, W_TMP1 // Functional Utility: Copies W to W_TMP1.
	pslldq	$12, W_TMP2 // Functional Utility: Shifts W_TMP2 for rotation.
  .elseif ((i & 3) == 2) // Functional Utility: For every (4k+2)th round.
	psrld	$31, W // Functional Utility: Shifts W right by 31 bits.
	pslld	$1, W_TMP1 // Functional Utility: Shifts W_TMP1 left by 1 bit.
	por	W, W_TMP1 // Functional Utility: ORs W and W_TMP1 for rotation.
	movdqa	W_TMP2, W // Functional Utility: Moves W_TMP2 to W.
	psrld	$30, W_TMP2 // Functional Utility: Shifts W_TMP2 right by 30 bits.
	pslld	$2, W // Functional Utility: Shifts W left by 2 bits.
  .elseif ((i & 3) == 3) // Functional Utility: For every (4k+3)th round.
	pxor	W, W_TMP1 // Functional Utility: XORs W and W_TMP1.
	pxor	W_TMP2, W_TMP1 // Functional Utility: XORs W_TMP2 and W_TMP1.
	movdqa	W_TMP1, W // Functional Utility: Moves W_TMP1 to W.
	paddd	K_XMM(K_BASE), W_TMP1 // Functional Utility: Adds round constant to W_TMP1.
	movdqa	W_TMP1, WK(i&~3) // Functional Utility: Stores pre-calculated K+W.
	W_PRECALC_ROTATE // Functional Utility: Rotates XMM register aliases.
  .endif
.endm

/* message scheduling pre-compute for rounds 32-79
 *
 * in SHA-1 specification: w[i] = (w[i-3] ^ w[i-8]  ^ w[i-14] ^ w[i-16]) rol 1
 * instead we do equal:    w[i] = (w[i-6] ^ w[i-16] ^ w[i-28] ^ w[i-32]) rol 2
 * allows more efficient vectorization since w[i]=>w[i-3] dependency is broken
 */
/**
 * @brief Pre-calculates SHA-1 message schedule words and K+W for rounds 32-79 (SSSE3).
 * @details This macro generates message schedule words (`W[i]`) and `K+W[i]` values
 * for rounds 32-79, optimized for SSSE3. It uses an alternative message
 * expansion formula to break direct dependencies, allowing for more efficient
 * vectorization, using instructions like `pxor`, `palignr`, `psrld`, `pslld`, `por`.
 * Functional Utility: Generates SHA-1 message schedule words and K+W values for later rounds (32-79) with SSSE3.
 */
.macro W_PRECALC_32_79_SSSE3
  .if ((i & 3) == 0) // Functional Utility: For every 4th round (32, 36, etc.).
	movdqa	W_minus_04, W_TMP1 // Functional Utility: Moves W[i-4] to W_TMP1.
	pxor	W_minus_28, W		# W is W_minus_32 before xor // Functional Utility: XORs W[i-28] with W (W[i-32]).
	palignr	$8, W_minus_08, W_TMP1 // Functional Utility: Aligns and combines W[i-8] and W[i-4] to get W[i-6].
  .elseif ((i & 3) == 1) // Functional Utility: For every (4k+1)th round.
	pxor	W_minus_16, W // Functional Utility: XORs W[i-16] with W.
	pxor	W_TMP1, W // Functional Utility: XORs W_TMP1 (W[i-6]) with W.
	movdqa	W, W_TMP1 // Functional Utility: Copies W to W_TMP1.
  .elseif ((i & 3) == 2) // Functional Utility: For every (4k+2)th round.
	psrld	$30, W // Functional Utility: Shifts W right by 30 bits.
	pslld	$2, W_TMP1 // Functional Utility: Shifts W_TMP1 left by 2 bits.
	por	W, W_TMP1 // Functional Utility: ORs W and W_TMP1 for 2-bit rotation.
  .elseif ((i & 3) == 3) // Functional Utility: For every (4k+3)th round.
	movdqa	W_TMP1, W // Functional Utility: Moves W_TMP1 to W.
	paddd	K_XMM(K_BASE), W_TMP1 // Functional Utility: Adds round constant to W_TMP1.
	movdqa	W_TMP1, WK(i&~3) // Functional Utility: Stores pre-calculated K+W.
	W_PRECALC_ROTATE // Functional Utility: Rotates XMM register aliases.
  .endif
.endm

.endm		// W_PRECALC_SSSE3


#define K1	0x5a827999 // Functional Role: SHA-1 round constant for rounds 0-19.
#define K2	0x6ed9eba1 // Functional Role: SHA-1 round constant for rounds 20-39.
#define K3	0x8f1bbcdc // Functional Role: SHA-1 round constant for rounds 40-59.
#define K4	0xca62c1d6 // Functional Role: SHA-1 round constant for rounds 60-79.

.section .rodata
.align 16

/**
 * @brief Array of SHA-1 round constants for XMM processing.
 * @details This data section stores the SHA-1 round constants (K1, K2, K3, K4)
 * replicated across 16-byte aligned memory to facilitate efficient loading
 * into XMM registers (as 4 DWORDS). This allows the `paddd` instruction
 * to add the correct constant value to multiple message words simultaneously.
 * Functional Role: Provides aligned SHA-1 round constants for vectorized addition.
 */
K_XMM_AR:
	.long K1, K1, K1, K1
	.long K2, K2, K2, K2
	.long K3, K3, K3, K3
	.long K4, K4, K4, K4

/**
 * @brief Shuffle control mask for byte-swapping with SSSE3 `pshufb` instruction.
 * @details This 16-byte constant provides the control mask required by the SSSE3
 * `pshufb` instruction to perform byte-swapping on 32-bit words within an XMM
 * register. This is essential for converting input data from little-endian to
 * big-endian, as required by the SHA-1 algorithm.
 * Functional Role: Defines a pattern for byte-reversal of 32-bit words within an XMM register.
 */
BSWAP_SHUFB_CTL:
	.long 0x00010203
	.long 0x04050607
	.long 0x08090a0b
	.long 0x0c0d0e0f


.section .text

W_PRECALC_SSSE3 // Functional Utility: Invokes the SSSE3-specific message schedule pre-calculation macros.
.macro xmm_mov a, b
	movdqu	\a,\b
.endm

/*
 * SSSE3 optimized implementation:
 *
 * extern "C" void sha1_transform_ssse3(struct sha1_state *state,
 *					const u8 *data, int blocks);
 *
 * Note that struct sha1_state is assumed to begin with u32 state[5].
 */
/**
 * @brief SHA-1 transform function optimized with Intel SSSE3 instructions.
 * @details This function processes `blocks` of 64-byte input data using
 * Intel SSSE3 instructions to update an existing SHA-1 hash digest. It is
 * designed for high performance on CPUs supporting SSSE3, leveraging its
 * vector capabilities for parallel processing of message words and hash state.
 * @param state Pointer to the `struct sha1_state` hash digest.
 * @param data Pointer to the `const u8` input data.
 * @param blocks Number of 64-byte blocks to process.
 * Functional Utility: Accelerates SHA-1 compression using SSSE3 instructions for multiple data blocks.
 */
SHA1_VECTOR_ASM     sha1_transform_ssse3

/**
 * @brief Umbrella macro for AVX-specific message schedule pre-calculation macros.
 * @details This macro defines the specific `W_PRECALC_*` macros to be used
 * when compiling for AVX instruction set extensions. It provides the AVX-optimized
 * implementations for generating message schedule words and combining them with
 * round constants, often utilizing VEX-encoded instructions.
 * Functional Utility: Groups AVX-optimized message schedule pre-calculation macros.
 */
.macro W_PRECALC_AVX

.purgem W_PRECALC_00_15 // Functional Utility: Purges the SSSE3 version of W_PRECALC_00_15.
.macro  W_PRECALC_00_15 // Functional Utility: Redefines W_PRECALC_00_15 for AVX.
    W_PRECALC_00_15_AVX
.endm
.purgem W_PRECALC_16_31 // Functional Utility: Purges the SSSE3 version of W_PRECALC_16_31.
.macro  W_PRECALC_16_31 // Functional Utility: Redefines W_PRECALC_16_31 for AVX.
    W_PRECALC_16_31_AVX
.endm
.purgem W_PRECALC_32_79 // Functional Utility: Purges the SSSE3 version of W_PRECALC_32_79.
.macro  W_PRECALC_32_79 // Functional Utility: Redefines W_PRECALC_32_79 for AVX.
    W_PRECALC_32_79_AVX
.endm

/**
 * @brief Pre-calculates SHA-1 message schedule words and K+W for rounds 0-15 (AVX).
 * @details This macro implements the message schedule generation (`W[i]`) and
 * the `K+W[i]` values for rounds 0-15 of SHA-1, optimized for AVX. It loads
 * input data using `vmovdqu`, performs byte-swapping (`vpshufb`), and combines
 * with round constants (`vpaddd`).
 * Functional Utility: Generates SHA-1 message schedule words and K+W values for initial rounds with AVX.
 */
.macro W_PRECALC_00_15_AVX
  .if ((i & 3) == 0) // Functional Utility: For every 4th round (0, 4, 8, 12) - loads 16 bytes.
	vmovdqu	(i*4)(BUFFER_PTR), W_TMP1 // Functional Utility: Loads 16 bytes (4 DWORDS) of message data into W_TMP1 using VEX-encoded instruction.
  .elseif ((i & 3) == 1) // Functional Utility: For every (4k+1)th round.
	vpshufb	XMM_SHUFB_BSWAP, W_TMP1, W // Functional Utility: Performs byte-swapping on W_TMP1 for big-endian, stores result in W.
  .elseif ((i & 3) == 2) // Functional Utility: For every (4k+2)th round.
	vpaddd	(K_BASE), W, W_TMP1 // Functional Utility: Adds round constant from K_BASE to W, stores result in W_TMP1.
  .elseif ((i & 3) == 3) // Functional Utility: For every (4k+3)th round.
	vmovdqa	W_TMP1, WK(i&~3) // Functional Utility: Stores the pre-calculated K+W value.
	W_PRECALC_ROTATE // Functional Utility: Rotates XMM register aliases.
  .endif
.endm

/**
 * @brief Pre-calculates SHA-1 message schedule words and K+W for rounds 16-31 (AVX).
 * @details This macro generates message schedule words (`W[i]`) and `K+W[i]` values
 * for rounds 16-31, optimized for AVX. It handles the specific SHA-1 message
 * expansion formula for these rounds using AVX instructions like `vpalignr`,
 * `vpsrldq`, `vpxor`, `vpslldq`, `vpslld`, `vpsrld`, `vpor`.
 * Functional Utility: Generates SHA-1 message schedule words and K+W values for middle rounds (16-31) with AVX.
 */
.macro W_PRECALC_16_31_AVX
  .if ((i & 3) == 0) // Functional Utility: For every 4th round (16, 20, etc.).
	vpalignr $8, W_minus_16, W_minus_12, W	# w[i-14] // Functional Utility: Aligns and combines W[i-16] and W[i-12] to get W[i-14].
	vpsrldq	$4, W_minus_04, W_TMP1		# w[i-3] // Functional Utility: Shifts W[i-4] to get W[i-3].
	vpxor	W_minus_08, W, W // Functional Utility: XORs W[i-8] with W (W[i-14]).
	vpxor	W_minus_16, W_TMP1, W_TMP1 // Functional Utility: XORs W[i-16] with W_TMP1 (W[i-3]).
  .elseif ((i & 3) == 1) // Functional Utility: For every (4k+1)th round.
	vpxor	W_TMP1, W, W // Functional Utility: XORs W_TMP1 with W.
	vpslldq	$12, W, W_TMP2 // Functional Utility: Shifts W for rotation.
	vpslld	$1, W, W_TMP1 // Functional Utility: Shifts W left by 1 bit.
  .elseif ((i & 3) == 2) // Functional Utility: For every (4k+2)th round.
	vpsrld	$31, W, W // Functional Utility: Shifts W right by 31 bits.
	vpor	W, W_TMP1, W_TMP1 // Functional Utility: ORs W and W_TMP1 for rotation.
	vpslld	$2, W_TMP2, W // Functional Utility: Shifts W_TMP2 left by 2 bits.
	vpsrld	$30, W_TMP2, W_TMP2 // Functional Utility: Shifts W_TMP2 right by 30 bits.
  .elseif ((i & 3) == 3) // Functional Utility: For every (4k+3)th round.
	vpxor	W, W_TMP1, W_TMP1 // Functional Utility: XORs W and W_TMP1.
	vpxor	W_TMP2, W_TMP1, W // Functional Utility: XORs W_TMP2 and W_TMP1, stores result in W.
	vpaddd	K_XMM(K_BASE), W, W_TMP1 // Functional Utility: Adds round constant to W, stores result in W_TMP1.
	vmovdqu	W_TMP1, WK(i&~3) // Functional Utility: Stores pre-calculated K+W.
	W_PRECALC_ROTATE // Functional Utility: Rotates XMM register aliases.
  .endif
.endm

/**
 * @brief Pre-calculates SHA-1 message schedule words and K+W for rounds 32-79 (AVX).
 * @details This macro generates message schedule words (`W[i]`) and `K+W[i]` values
 * for rounds 32-79, optimized for AVX. It uses an alternative message
 * expansion formula to break direct dependencies, allowing for more efficient
 * vectorization, using instructions like `vpalignr`, `vpxor`, `vpsrld`, `vpslld`, `vpor`.
 * Functional Utility: Generates SHA-1 message schedule words and K+W values for later rounds (32-79) with AVX.
 */
.macro W_PRECALC_32_79_AVX
  .if ((i & 3) == 0) // Functional Utility: For every 4th round (32, 36, etc.).
	vpalignr $8, W_minus_08, W_minus_04, W_TMP1 // Functional Utility: Aligns and combines W[i-8] and W[i-4] to get W[i-6].
	vpxor	W_minus_28, W, W		# W is W_minus_32 before xor // Functional Utility: XORs W[i-28] with W (W[i-32]).
  .elseif ((i & 3) == 1) // Functional Utility: For every (4k+1)th round.
	vpxor	W_minus_16, W_TMP1, W_TMP1 // Functional Utility: XORs W[i-16] with W_TMP1 (W[i-6]).
	vpxor	W_TMP1, W, W // Functional Utility: XORs W_TMP1 with W.
  .elseif ((i & 3) == 2) // Functional Utility: For every (4k+2)th round.
	vpslld	$2, W, W_TMP1 // Functional Utility: Shifts W left by 2 bits.
	vpsrld	$30, W, W // Functional Utility: Shifts W right by 30 bits.
	vpor	W, W_TMP1, W // Functional Utility: ORs W and W_TMP1 for 2-bit rotation.
  .elseif ((i & 3) == 3) // Functional Utility: For every (4k+3)th round.
	vpaddd	K_XMM(K_BASE), W, W_TMP1 // Functional Utility: Adds round constant to W, stores result in W_TMP1.
	vmovdqu	W_TMP1, WK(i&~3) // Functional Utility: Stores pre-calculated K+W.
	W_PRECALC_ROTATE // Functional Utility: Rotates XMM register aliases.
  .endif
.endm

.endm    // W_PRECALC_AVX

W_PRECALC_AVX // Functional Utility: Invokes the AVX-specific message schedule pre-calculation macros.
.purgem xmm_mov // Functional Utility: Purges the SSSE3 version of xmm_mov.
.macro xmm_mov a, b // Functional Utility: Redefines xmm_mov for AVX using VEX-encoded `vmovdqu`.
	vmovdqu	\a,\b
.endm


/* AVX optimized implementation:
 *  extern "C" void sha1_transform_avx(struct sha1_state *state,
 *				       const u8 *data, int blocks);
 */
/**
 * @brief SHA-1 transform function optimized with Intel AVX instructions.
 * @details This function processes `blocks` of 64-byte input data using
 * Intel AVX instructions to update an existing SHA-1 hash digest. It is
 * designed for high performance on CPUs supporting AVX, leveraging its
 * vector capabilities for parallel processing of message words and hash state.
 * @param state Pointer to the `struct sha1_state` hash digest.
 * @param data Pointer to the `const u8` input data.
 * @param blocks Number of 64-byte blocks to process.
 * Functional Utility: Accelerates SHA-1 compression using AVX instructions for multiple data blocks.
 */
SHA1_VECTOR_ASM     sha1_transform_avx
