/* SPDX-License-Identifier: GPL-2.0-or-later */
/**
 * @file sha1-spe-asm.S
 * @brief Fast SHA-1 implementation for PowerPC SPE (Signal Processing Engine) instruction set.
 * @details This assembly file provides a highly optimized implementation of the
 * SHA-1 (Secure Hash Algorithm 1) compression function, specifically tailored
 * for the PowerPC architecture utilizing the Signal Processing Engine (SPE)
 * SIMD instruction set. It leverages the SPE's 128-bit vector capabilities
 * to perform operations on multiple 32-bit words simultaneously,
 * significantly accelerating the cryptographic process.
 *
 * **Optimization Techniques:**
 * - **SPE SIMD Instructions**: Extensive use of SPE instructions (`evlwwsplat`,
 *   `evxor`, `evand`, `evor`, `evrlwi`, `evaddw`, `evmergelohi/hi`) for
 *   vectorized operations on hash values and message words.
 * - **Register Mapping**: Dedicated GPRs are used to represent SHA-1 hash values (H0-H4)
 *   and message words (W0-W7), optimizing data flow.
 * - **Pipelined Message Schedule**: Message schedule expansion is often interleaved
 *   with round computations to hide latency.
 * - **Macro-driven Round Execution**: Complex macros encapsulate the logic for
 *   different SHA-1 round phases (0-15, 16-19, 20-39, 40-59, 60-79), enabling
 *   efficient code generation and unrolling.
 * - **Endianness Handling**: The `LOAD_DATA` macro (in `PPC_SPE_SHA1_K` context)
 *   ensures correct byte ordering for SHA-1 processing.
 * This implementation is designed to provide maximum SHA-1 hashing performance
 * on PowerPC SPE-enabled systems within the Linux kernel.
 */
/*
 * Fast SHA-1 implementation for SPE instruction set (PPC)
 *
 * This code makes use of the SPE SIMD instruction set as defined in
 * http://cache.freescale.com/files/32bit/doc/ref_manual/SPEPIM.pdf
 * Implementation is based on optimization guide notes from
 * http://cache.freescale.com/files/32bit/doc/app_note/AN2665.pdf
 *
 * Copyright (c) 2015 Markus Stockhausen <stockhausen@collogia.de>
 */

#include <asm/ppc_asm.h>
#include <asm/asm-offsets.h>

#define rHP	r3	/* Functional Role: Pointer to hash value (SHA-1 state). */
#define rWP	r4	/* Functional Role: Pointer to input message block. */
#define rKP	r5	/* Functional Role: Pointer to SHA-1 round constants. */

#define rW0	r14	/* Functional Role: 64-bit message word register W0. */
#define rW1	r15 /* Functional Role: 64-bit message word register W1. */
#define rW2	r16 /* Functional Role: 64-bit message word register W2. */
#define rW3	r17 /* Functional Role: 64-bit message word register W3. */
#define rW4	r18 /* Functional Role: 64-bit message word register W4. */
#define rW5	r19 /* Functional Role: 64-bit message word register W5. */
#define rW6	r20 /* Functional Role: 64-bit message word register W6. */
#define rW7	r21 /* Functional Role: 64-bit message word register W7. */

#define rH0	r6	/* Functional Role: 32-bit SHA-1 hash value H0. */
#define rH1	r7	/* Functional Role: 32-bit SHA-1 hash value H1. */
#define rH2	r8	/* Functional Role: 32-bit SHA-1 hash value H2. */
#define rH3	r9	/* Functional Role: 32-bit SHA-1 hash value H3. */
#define rH4	r10 /* Functional Role: 32-bit SHA-1 hash value H4. */

#define rT0	r22	/* Functional Role: 64-bit temporary register T0. */
#define rT1	r0	/* Functional Role: 32-bit temporary register T1. */
#define rT2	r11 /* Functional Role: 32-bit temporary register T2. */
#define rT3	r12 /* Functional Role: 32-bit temporary register T3. */

#define rK	r23	/* Functional Role: 64-bit volatile register for SHA-1 round constant. */

/**
 * @brief Macro to load SHA-1 round constant K01.
 * Functional Utility: This macro is used to load the SHA-1 round constant
 * for rounds 0-19 into the volatile register `rK`.
 */
#define LOAD_K01

/**
 * @brief Macro to load SHA-1 round constant K11.
 * @details This macro uses the SPE instruction `evlwwsplat` (load word with
 * word splat) to load the first 32-bit word of the K20-39 round constant
 * from memory into `rK` and replicate it across the 128-bit SPE register.
 * This prepares `rK` for vectorized addition in SHA-1 rounds.
 * Functional Utility: Loads and splats a 32-bit SHA-1 round constant from rKP into rK for vectorized operations.
 */
#define LOAD_K11 
	evlwwsplat	rK,0(rKP);

/**
 * @brief Macro to load SHA-1 round constant K21.
 * @details This macro uses the SPE instruction `evlwwsplat` (load word with
 * word splat) to load the first 32-bit word of the K40-59 round constant
 * from memory into `rK` and replicate it across the 128-bit SPE register.
 * Functional Utility: Loads and splats a 32-bit SHA-1 round constant from rKP+4 into rK for vectorized operations.
 */
#define LOAD_K21 
	evlwwsplat	rK,4(rKP);

/**
 * @brief Macro to load SHA-1 round constant K31.
 * @details This macro uses the SPE instruction `evlwwsplat` (load word with
 * word splat) to load the first 32-bit word of the K60-79 round constant
 * from memory into `rK` and replicate it across the 128-bit SPE register.
 * Functional Utility: Loads and splats a 32-bit SHA-1 round constant from rKP+8 into rK for vectorized operations.
 */
#define LOAD_K31 
	evlwwsplat	rK,8(rKP);

/**
 * @brief Macro to load SHA-1 round constant K41.
 * @details This macro uses the SPE instruction `evlwwsplat` (load word with
 * word splat) to load the first 32-bit word of the K0-15 round constant
 * from memory into `rK` and replicate it across the 128-bit SPE register.
 * Functional Utility: Loads and splats a 32-bit SHA-1 round constant from rKP+12 into rK for vectorized operations.
 */
#define LOAD_K41 
	evlwwsplat	rK,12(rKP);

/**
 * @brief Macro to initialize the stack frame and save non-volatile SPE registers.
 * @details This macro sets up the stack frame for the function call and saves
 * a set of non-volatile SPE (Signal Processing Engine) registers (r14-r23)
 * onto the newly created stack frame. This adheres to the PowerPC ABI for
 * function calls, ensuring that these registers are preserved across the function's
 * execution.
 * Functional Utility: Sets up function stack and saves callee-saved SPE registers.
 */
#define INITIALIZE 
	stwu		r1,-128(r1);	/* Functional Utility: Create stack frame of 128 bytes. */ 
	evstdw		r14,8(r1);	/* Functional Utility: Save SPE register r14 onto the stack. */ 
	evstdw		r15,16(r1);	/* Functional Utility: Save SPE register r15 onto the stack. */ 
	evstdw		r16,24(r1);	/* Functional Utility: Save SPE register r16 onto the stack. */ 
	evstdw		r17,32(r1);					   
	evstdw		r18,40(r1);					   
	evstdw		r19,48(r1);					   
	evstdw		r20,56(r1);					   
	evstdw		r21,64(r1);					   
	evstdw		r22,72(r1);					   
	evstdw		r23,80(r1);


/**
 * @brief Macro to restore non-volatile SPE registers, clear sensitive data, and clean up the stack frame.
 * @details This macro performs the inverse operations of `INITIALIZE`. It restores
 * the saved non-volatile SPE registers from the stack, securely clears the
 * stack area where sensitive data might have been temporarily stored (for
 * security reasons), and then cleans up the stack frame. This ensures proper
 * function exit and resource management.
 * Functional Utility: Restores callee-saved SPE registers, clears sensitive stack data, and cleans up the stack frame.
 */
#define FINALIZE 
	evldw		r14,8(r1);	/* Functional Utility: Restore SPE register r14 from the stack. */ 
	evldw		r15,16(r1);					   
	evldw		r16,24(r1);					   
	evldw		r17,32(r1);					   
	evldw		r18,40(r1);					   
	evldw		r19,48(r1);					   
	evldw		r20,56(r1);					   
	evldw		r21,64(r1);					   
	evldw		r22,72(r1);					   
	evldw		r23,80(r1);					   
	xor		r0,r0,r0;					   
	stw		r0,8(r1);	/* Functional Utility: Zeroes out stack memory. */ 
	stw		r0,16(r1);	/* Functional Utility: Zeroes out stack memory. */ 
	stw		r0,24(r1);					   
	stw		r0,32(r1);					   
	stw		r0,40(r1);					   
	stw		r0,48(r1);					   
	stw		r0,56(r1);					   
	stw		r0,64(r1);					   
	stw		r0,72(r1);					   
	stw		r0,80(r1);					   
	addi		r1,r1,128;	/* Functional Utility: Clean up the stack frame by adjusting stack pointer. */

/**
 * @brief Macro to load a 32-bit data word from the input block.
 * @details This macro loads a 32-bit word from the input message block (`rWP`)
 * into the specified register (`reg`). It handles endianness: on big-endian
 * systems (`__BIG_ENDIAN__` defined), it uses `lwz` (load word zero-extended).
 * On little-endian systems, it uses `lwbrx` (load word byte-reverse indexed)
 * to perform byte-reversal, ensuring the word is in big-endian format as
 * required by SHA-1.
 *
 * @param reg Destination register.
 * @param off Offset from `rWP` for `lwz` on big-endian.
 * Functional Utility: Loads a 32-bit word from the input data, handling endianness.
 */
#ifdef __BIG_ENDIAN__
#define LOAD_DATA(reg, off) 
	lwz		reg,off(rWP);	/* load data			*/
#define NEXT_BLOCK 
	addi		rWP,rWP,64;	/* increment per block		*/
#else
#define LOAD_DATA(reg, off) 
	lwbrx		reg,0,rWP;	/* load data			*/ 
	addi		rWP,rWP,4;	/* increment per word		*/
#define NEXT_BLOCK			/* nothing to do		*/
#endif

/**
 * @brief Executes two SHA-1 rounds (0-15) using SPE SIMD instructions.
 * @details This macro performs two consecutive rounds of the SHA-1 algorithm
 * for rounds 0-15. It utilizes SPE SIMD instructions to efficiently compute
 * the logical function `F(B,C,D) = (B & C) | (~B & D)`, update the hash
 * values (A-E), and add the message word (W) and round constant (K).
 * The parameters `a,b,c,d,e` represent the SHA-1 hash values, `w0,w1` are
 * message words, and `k` is the round constant. `off` is for data loading offset.
 *
 * @param a,b,c,d,e Registers holding SHA-1 state variables.
 * @param w0,w1 Registers holding message words.
 * @param k Round constant identifier.
 * @param off Offset for loading data.
 * Functional Utility: Executes two pipelined SHA-1 rounds (0-15) with the Choose logical function.
 */
#define	R_00_15(a, b, c, d, e, w0, w1, k, off) 
	LOAD_DATA(w0, off)		/* Functional Utility: Loads message word W0 from input. */ 
	and		rT2,b,c;	/* Functional Utility: F' = B and C. */ 
	LOAD_K##k##1							   
	andc		rT1,d,b;	/* Functional Utility: F" = ~B and D. */ 
	rotrwi		rT0,a,27;	/* Functional Utility: A' = A rotl 5. */ 
	or		rT2,rT2,rT1;	/* Functional Utility: F = F' or F". */ 
	add		e,e,rT0;	/* Functional Utility: E = E + A'. */ 
	rotrwi		b,b,2;		/* Functional Utility: B = B rotl 30. */ 
	add		e,e,w0;		/* Functional Utility: E = E + W. */ 
	LOAD_DATA(w1, off+4)		/* Functional Utility: Loads message word W1 from input. */ 
	add		e,e,rT2;	/* Functional Utility: E = E + F. */ 
	and		rT1,a,b;	/* Functional Utility: F' = B and C. */ 
	add		e,e,rK;		/* Functional Utility: E = E + K. */ 
	andc		rT2,c,a;	/* Functional Utility: F" = ~B and D. */ 
	add		d,d,rK;		/* Functional Utility: E = E + K. */ 
	or		rT2,rT2,rT1;	/* Functional Utility: F = F' or F". */ 
	rotrwi		rT0,e,27;	/* Functional Utility: A' = A rotl 5. */ 
	add		d,d,w1;		/* Functional Utility: E = E + W. */ 
	rotrwi		a,a,2;		/* Functional Utility: B = B rotl 30. */ 
	add		d,d,rT0;	/* Functional Utility: E = E + A'. */ 
	evmergelo	w1,w1,w0;	/* Functional Utility: Mix W[0]/W[1] for message schedule. */ 
	add		d,d,rT2		/* Functional Utility: E = E + F. */

/**
 * @brief Executes two SHA-1 rounds (16-19) with message schedule update.
 * @details This macro performs two consecutive rounds of the SHA-1 algorithm
 * for rounds 16-19. It uses the logical function `F(B,C,D) = (B & C) | (~B & D)`
 * and simultaneously updates the message schedule using `evxor` and `evrlwi`
 * (rotate left word immediate) instructions. This is crucial for hiding the
 * latency of message word generation.
 *
 * @param a,b,c,d,e Registers holding SHA-1 state variables.
 * @param w0,w1,w4,w6,w7 Registers holding message words for schedule expansion.
 * @param k Round constant identifier.
 * Functional Utility: Executes two pipelined SHA-1 rounds (16-19), performing message schedule updates.
 */
#define R_16_19(a, b, c, d, e, w0, w1, w4, w6, w7, k) 
	and		rT2,b,c;	/* Functional Utility: F' = B and C. */ 
	evmergelohi	rT0,w7,w6;	/* Functional Utility: W[-3] component for schedule expansion. */ 
	andc		rT1,d,b;	/* Functional Utility: F" = ~B and D. */ 
	evxor		w0,w0,rT0;	/* Functional Utility: W = W[-16] xor W[-3]. */ 
	or		rT1,rT1,rT2;	/* Functional Utility: F = F' or F". */ 
	evxor		w0,w0,w4;	/* Functional Utility: W = W xor W[-8]. */ 
	add		e,e,rT1;	/* Functional Utility: E = E + F. */ 
	evxor		w0,w0,w1;	/* Functional Utility: W = W xor W[-14]. */ 
	rotrwi		rT2,a,27;	/* Functional Utility: A' = A rotl 5. */ 
	evrlwi		w0,w0,1;	/* Functional Utility: W = W rotl 1. */ 
	add		e,e,rT2;	/* Functional Utility: E = E + A'. */ 
	evaddw		rT0,w0,rK;	/* Functional Utility: WK = W + K. */ 
	rotrwi		b,b,2;		/* Functional Utility: B = B rotl 30. */ 
	LOAD_K##k##1							   
	evmergehi	rT1,rT1,rT0;	/* Functional Utility: Mix WK1/WK2 for SPE. */ 
	add		e,e,rT0;	/* Functional Utility: E = E + WK. */ 
	add		d,d,rT1;	/* Functional Utility: E = E + WK. */ 
	and		rT2,a,b;	/* Functional Utility: F' = B and C. */ 
	andc		rT1,c,a;	/* Functional Utility: F" = ~B and D. */ 
	rotrwi		rT0,e,27;	/* Functional Utility: A' = A rotl 5. */ 
	or		rT1,rT1,rT2;	/* Functional Utility: F = F' or F". */ 
	add		d,d,rT0;	/* Functional Utility: E = E + A'. */ 
	rotrwi		a,a,2;		/* Functional Utility: B = B rotl 30. */ 
	add		d,d,rT1		/* Functional Utility: E = E + F. */

/**
 * @brief Executes two SHA-1 rounds (20-39) with message schedule update.
 * @details This macro performs two consecutive rounds of the SHA-1 algorithm
 * for rounds 20-39. It uses the logical function `F(B,C,D) = B ^ C ^ D` (XOR
 * majority) and simultaneously updates the message schedule using SPE SIMD
 * instructions.
 *
 * @param a,b,c,d,e Registers holding SHA-1 state variables.
 * @param w0,w1,w4,w6,w7 Registers holding message words for schedule expansion.
 * @param k Round constant identifier.
 * Functional Utility: Executes two pipelined SHA-1 rounds (20-39), performing message schedule updates.
 */
#define R_20_39(a, b, c, d, e, w0, w1, w4, w6, w7, k) 
	evmergelohi	rT0,w7,w6;	/* Functional Utility: W[-3] component for schedule expansion. */ 
	xor		rT2,b,c;	/* Functional Utility: F' = B xor C. */ 
	evxor		w0,w0,rT0;	/* Functional Utility: W = W[-16] xor W[-3]. */ 
	xor		rT2,rT2,d;	/* Functional Utility: F = F' xor D. */ 
	evxor		w0,w0,w4;	/* Functional Utility: W = W xor W[-8]. */ 
	add		e,e,rT2;	/* Functional Utility: E = E + F. */ 
	evxor		w0,w0,w1;	/* Functional Utility: W = W xor W[-14]. */ 
	rotrwi		rT2,a,27;	/* Functional Utility: A' = A rotl 5. */ 
	evrlwi		w0,w0,1;	/* Functional Utility: W = W rotl 1. */ 
	add		e,e,rT2;	/* Functional Utility: E = E + A'. */ 
	evaddw		rT0,w0,rK;	/* Functional Utility: WK = W + K. */ 
	rotrwi		b,b,2;		/* Functional Utility: B = B rotl 30. */ 
	LOAD_K##k##1							   
	evmergehi	rT1,rT1,rT0;	/* Functional Utility: Mix WK1/WK2 for SPE. */ 
	add		e,e,rT0;	/* Functional Utility: E = E + WK. */ 
	xor		rT2,a,b;	/* Functional Utility: F' = B xor C. */ 
	add		d,d,rT1;	/* Functional Utility: E = E + WK. */ 
	xor		rT2,rT2,c;	/* Functional Utility: F = F' xor D. */ 
	rotrwi		rT0,e,27;	/* Functional Utility: A' = A rotl 5. */ 
	add		d,d,rT2;	/* Functional Utility: E = E + F. */ 
	rotrwi		a,a,2;		/* Functional Utility: B = B rotl 30. */ 
	add		d,d,rT0		/* Functional Utility: E = E + A'. */

/**
 * @brief Executes two SHA-1 rounds (40-59) with message schedule update.
 * @details This macro performs two consecutive rounds of the SHA-1 algorithm
 * for rounds 40-59. It uses the logical function `F(B,C,D) = (B & C) | (B & D) | (C & D)`
 * (Majority Function) and simultaneously updates the message schedule using
 * SPE SIMD instructions.
 *
 * @param a,b,c,d,e Registers holding SHA-1 state variables.
 * @param w0,w1,w4,w6,w7 Registers holding message words for schedule expansion.
 * @param k Round constant identifier.
 * Functional Utility: Executes two pipelined SHA-1 rounds (40-59), performing message schedule updates.
 */
#define R_40_59(a, b, c, d, e, w0, w1, w4, w6, w7, k) 
	and		rT2,b,c;	/* Functional Utility: F' = B and C. */ 
	evmergelohi	rT0,w7,w6;	/* Functional Utility: W[-3] component for schedule expansion. */ 
	or		rT1,b,c;	/* Functional Utility: F" = B or C. */ 
	evxor		w0,w0,rT0;	/* Functional Utility: W = W[-16] xor W[-3]. */ 
	and		rT1,d,rT1;	/* Functional Utility: F" = F" and D (B or C) and D. */ 
	evxor		w0,w0,w4;	/* Functional Utility: W = W xor W[-8]. */ 
	or		rT2,rT2,rT1;	/* Functional Utility: F = F' or F". */ 
	evxor		w0,w0,w1;	/* Functional Utility: W = W xor W[-14]. */ 
	add		e,e,rT2;	/* Functional Utility: E = E + F. */ 
	evrlwi		w0,w0,1;	/* Functional Utility: W = W rotl 1. */ 
	rotrwi		rT2,a,27;	/* Functional Utility: A' = A rotl 5. */ 
	evaddw		rT0,w0,rK;	/* Functional Utility: WK = W + K. */ 
	add		e,e,rT2;	/* Functional Utility: E = E + A'. */ 
	LOAD_K##k##1							   
	evmergehi	rT1,rT1,rT0;	/* Functional Utility: Mix WK1/WK2 for SPE. */ 
	rotrwi		b,b,2;		/* Functional Utility: B = B rotl 30. */ 
	add		e,e,rT0;	/* Functional Utility: E = E + WK. */ 
	and		rT2,a,b;	/* Functional Utility: F' = B and C. */ 
	or		rT0,a,b;	/* Functional Utility: F" = B or C. */ 
	add		d,d,rT1;	/* Functional Utility: E = E + WK. */ 
	and		rT0,c,rT0;	/* Functional Utility: F" = F" and D. */ 
	rotrwi		a,a,2;		/* Functional Utility: B = B rotl 30. */ 
	or		rT2,rT2,rT0;	/* Functional Utility: F = F' or F". */ 
	rotrwi		rT0,e,27;	/* Functional Utility: A' = A rotl 5. */ 
	add		d,d,rT2;	/* Functional Utility: E = E + F. */ 
	add		d,d,rT0		/* Functional Utility: E = E + A'. */

/**
 * @brief Executes two SHA-1 rounds (60-79) with message schedule update.
 * @details This macro performs two consecutive rounds of the SHA-1 algorithm
 * for rounds 60-79. It uses the logical function `F(B,C,D) = B ^ C ^ D` (XOR
 * majority) and simultaneously updates the message schedule using SPE SIMD
 * instructions. It reuses the logic from `R_20_39` as the logical function is the same.
 *
 * @param a,b,c,d,e Registers holding SHA-1 state variables.
 * @param w0,w1,w4,w6,w7 Registers holding message words for schedule expansion.
 * @param k Round constant identifier.
 * Functional Utility: Executes two pipelined SHA-1 rounds (60-79), performing message schedule updates.
 */
#define R_60_79(a, b, c, d, e, w0, w1, w4, w6, w7, k) 
	R_20_39(a, b, c, d, e, w0, w1, w4, w6, w7, k)

/**
 * @brief PowerPC SPE optimized SHA-1 transform function.
 * @details This function implements the core SHA-1 compression for a 64-byte
 * message block, highly optimized for the PowerPC SPE (Signal Processing Engine)
 * architecture using assembly instructions. It utilizes SPE SIMD capabilities
 * for vectorized operations, a rotating register scheme for SHA-1 state variables,
 * and macro-driven loop unrolling to maximize performance.
 *
 * @param r3 (rHP) Pointer to the SHA-1 state array (H0-H4).
 * @param r4 (rWP) Pointer to the 64-byte input data block.
 * @param r5 (rKP) Pointer to the SHA-1 round constants array.
 * Functional Utility: Performs hardware-optimized SHA-1 compression on a 64-byte block using PowerPC SPE assembly.
 */
_GLOBAL(ppc_spe_sha1_transform)
	INITIALIZE /* Functional Utility: Sets up the stack frame and saves non-volatile SPE registers. */

	/* Load initial SHA-1 state (H0-H4) from rHP into working registers. */
	lwz		rH0,0(rHP) /* Functional Utility: Loads H0 into rH0. */
	lwz		rH1,4(rHP) /* Functional Utility: Loads H1 into rH1. */
	mtctr		r5 /* Functional Utility: Initializes loop counter with number of blocks in r5. */
	lwz		rH2,8(rHP) /* Functional Utility: Loads H2 into rH2. */
	lis		rKP,PPC_SPE_SHA1_K@h /* Functional Utility: Loads upper 16 bits of the address of SHA1 constants table into rKP. */
	lwz		rH3,12(rHP) /* Functional Utility: Loads H3 into rH3. */
	ori		rKP,rKP,PPC_SPE_SHA1_K@l /* Functional Utility: Loads lower 16 bits of the address of SHA1 constants table into rKP. */
	lwz		rH4,16(rHP) /* Functional Utility: Loads H4 into rH4. */

ppc_spe_sha1_main:
	/* Rounds 0-15: Initial processing with logical function F(B,C,D) = (B&C) | (~B&D) */
	/* Functional Utility: These macro calls execute 8 rounds each, for a total of 16 rounds. */
	R_00_15(rH0, rH1, rH2, rH3, rH4, rW1, rW0, 1, 0) /* Functional Utility: Processes rounds 0-1. */
	R_00_15(rH3, rH4, rH0, rH1, rH2, rW2, rW1, 0, 8) /* Functional Utility: Processes rounds 2-3. */
	R_00_15(rH1, rH2, rH3, rH4, rH0, rW3, rW2, 0, 16) /* Functional Utility: Processes rounds 4-5. */
	R_00_15(rH4, rH0, rH1, rH2, rH3, rW4, rW3, 0, 24) /* Functional Utility: Processes rounds 6-7. */
	R_00_15(rH2, rH3, rH4, rH0, rH1, rW5, rW4, 0, 32) /* Functional Utility: Processes rounds 8-9. */
	R_00_15(rH0, rH1, rH2, rH3, rH4, rW6, rW5, 0, 40) /* Functional Utility: Processes rounds 10-11. */
	R_00_15(rH3, rH4, rH0, rH1, rH2, rT3, rW6, 0, 48) /* Functional Utility: Processes rounds 12-13. */
	R_00_15(rH1, rH2, rH3, rH4, rH0, rT3, rW7, 0, 56) /* Functional Utility: Processes rounds 14-15. */

	/* Rounds 16-19: Logical function F(B,C,D) = (B&C) | (~B&D) with message schedule update. */
	R_16_19(rH4, rH0, rH1, rH2, rH3, rW0, rW1, rW4, rW6, rW7, 0) /* Functional Utility: Processes rounds 16-17. */
	R_16_19(rH2, rH3, rH4, rH0, rH1, rW1, rW2, rW5, rW7, rW0, 2) /* Functional Utility: Processes rounds 18-19. */

	/* Rounds 20-39: Logical function F(B,C,D) = B ^ C ^ D (Parity Function) with message schedule update. */
	R_20_39(rH0, rH1, rH2, rH3, rH4, rW2, rW3, rW6, rW0, rW1, 0) /* Functional Utility: Processes rounds 20-21. */
	R_20_39(rH3, rH4, rH0, rH1, rH2, rW3, rW4, rW7, rW1, rW2, 0) /* Functional Utility: Processes rounds 22-23. */
	R_20_39(rH1, rH2, rH3, rH4, rH0, rW4, rW5, rW0, rW2, rW3, 0) /* Functional Utility: Processes rounds 24-25. */
	R_20_39(rH4, rH0, rH1, rH2, rH3, rW5, rW6, rW1, rW3, rW4, 0) /* Functional Utility: Processes rounds 26-27. */
	R_20_39(rH2, rH3, rH4, rH0, rH1, rW6, rW7, rW2, rW4, rW5, 0) /* Functional Utility: Processes rounds 28-29. */
	R_20_39(rH0, rH1, rH2, rH3, rH4, rW7, rW0, rW3, rW5, rW6, 0) /* Functional Utility: Processes rounds 30-31. */
	R_20_39(rH3, rH4, rH0, rH1, rH2, rW0, rW1, rW4, rW6, rW7, 0) /* Functional Utility: Processes rounds 32-33. */
	R_20_39(rH1, rH2, rH3, rH4, rH0, rW1, rW2, rW5, rW7, rW0, 0) /* Functional Utility: Processes rounds 34-35. */
	R_20_39(rH4, rH0, rH1, rH2, rH3, rW2, rW3, rW6, rW0, rW1, 0) /* Functional Utility: Processes rounds 36-37. */
	R_20_39(rH2, rH3, rH4, rH0, rH1, rW3, rW4, rW7, rW1, rW2, 3) /* Functional Utility: Processes rounds 38-39. */

	/* Rounds 40-59: Logical function F(B,C,D) = (B&C) | (B&D) | (C&D) (Majority Function) with message schedule update. */
	R_40_59(rH0, rH1, rH2, rH3, rH4, rW4, rW5, rW0, rW2, rW3, 0) /* Functional Utility: Processes rounds 40-41. */
	R_40_59(rH3, rH4, rH0, rH1, rH2, rW5, rW6, rW1, rW3, rW4, 0) /* Functional Utility: Processes rounds 42-43. */
	R_40_59(rH1, rH2, rH3, rH4, rH0, rW6, rW7, rW2, rW4, rW5, 0) /* Functional Utility: Processes rounds 44-45. */
	R_40_59(rH4, rH0, rH1, rH2, rH3, rW7, rW0, rW3, rW5, rW6, 0) /* Functional Utility: Processes rounds 46-47. */
	R_40_59(rH2, rH3, rH4, rH0, rH1, rW0, rW1, rW4, rW6, rW7, 0) /* Functional Utility: Processes rounds 48-49. */
	R_40_59(rH0, rH1, rH2, rH3, rH4, rW1, rW2, rW5, rW7, rW0, 0) /* Functional Utility: Processes rounds 50-51. */
	R_40_59(rH3, rH4, rH0, rH1, rH2, rW2, rW3, rW6, rW0, rW1, 0) /* Functional Utility: Processes rounds 52-53. */
	R_40_59(rH1, rH2, rH3, rH4, rH0, rW3, rW4, rW7, rW1, rW2, 0) /* Functional Utility: Processes rounds 54-55. */
	R_40_59(rH4, rH0, rH1, rH2, rH3, rW4, rW5, rW0, rW2, rW3, 0) /* Functional Utility: Processes rounds 56-57. */
	R_40_59(rH2, rH3, rH4, rH0, rH1, rW5, rW6, rW1, rW3, rW4, 4) /* Functional Utility: Processes rounds 58-59. */

	/* Rounds 60-79: Logical function F(B,C,D) = B ^ C ^ D (Parity Function) with message schedule update. */
	R_60_79(rH0, rH1, rH2, rH3, rH4, rW6, rW7, rW2, rW4, rW5, 0) /* Functional Utility: Processes rounds 60-61. */
	R_60_79(rH3, rH4, rH0, rH1, rH2, rW7, rW0, rW3, rW5, rW6, 0) /* Functional Utility: Processes rounds 62-63. */
	R_60_79(rH1, rH2, rH3, rH4, rH0, rW0, rW1, rW4, rW6, rW7, 0) /* Functional Utility: Processes rounds 64-65. */
	R_60_79(rH4, rH0, rH1, rH2, rH3, rW1, rW2, rW5, rW7, rW0, 0) /* Functional Utility: Processes rounds 66-67. */
	R_60_79(rH2, rH3, rH4, rH0, rH1, rW2, rW3, rW6, rW0, rW1, 0) /* Functional Utility: Processes rounds 68-69. */
	R_60_79(rH0, rH1, rH2, rH3, rH4, rW3, rW4, rW7, rW1, rW2, 0) /* Functional Utility: Processes rounds 70-71. */
	R_60_79(rH3, rH4, rH0, rH1, rH2, rW4, rW5, rW0, rW2, rW3, 0) /* Functional Utility: Processes rounds 72-73. */
	lwz		rT3,0(rHP) /* Functional Utility: Loads original H0 into rT3 for final state update. */
	R_60_79(rH1, rH2, rH3, rH4, rH0, rW5, rW6, rW1, rW3, rW4, 0) /* Functional Utility: Processes rounds 74-75. */
	lwz		rW1,4(rHP) /* Functional Utility: Loads original H1 into rW1 for final state update. */
	R_60_79(rH4, rH0, rH1, rH2, rH3, rW6, rW7, rW2, rW4, rW5, 0) /* Functional Utility: Processes rounds 76-77. */
	lwz		rW2,8(rHP) /* Functional Utility: Loads original H2 into rW2 for final state update. */
	R_60_79(rH2, rH3, rH4, rH0, rH1, rW7, rW0, rW3, rW5, rW6, 0) /* Functional Utility: Processes rounds 78-79. */
	lwz		rW3,12(rHP) /* Functional Utility: Loads original H3 into rW3 for final state update. */
	NEXT_BLOCK /* Functional Utility: Advances rWP to the next 64-byte block. */
	lwz		rW4,16(rHP) /* Functional Utility: Loads original H4 into rW4 for final state update. */

	/* Epilogue: Update the initial state with the results of the compression and loop control. */
	add		rH0,rH0,rT3 /* Functional Utility: Adds the result of H0 (rH0) to original H0 (rT3). */
	stw		rH0,0(rHP) /* Functional Utility: Stores updated H0 back to memory. */
	add		rH1,rH1,rW1 /* Functional Utility: Adds the result of H1 (rH1) to original H1 (rW1). */
	stw		rH1,4(rHP) /* Functional Utility: Stores updated H1 back to memory. */
	add		rH2,rH2,rW2 /* Functional Utility: Adds the result of H2 (rH2) to original H2 (rW2). */
	stw		rH2,8(rHP) /* Functional Utility: Stores updated H2 back to memory. */
	add		rH3,rH3,rW3 /* Functional Utility: Adds the result of H3 (rH3) to original H3 (rW3). */
	stw		rH3,12(rHP) /* Functional Utility: Stores updated H3 back to memory. */
	add		rH4,rH4,rW4 /* Functional Utility: Adds the result of H4 (rH4) to original H4 (rW4). */
	stw		rH4,16(rHP) /* Functional Utility: Stores updated H4 back to memory. */

	bdnz		ppc_spe_sha1_main /* Functional Utility: Decrements CTR and branches back to `ppc_spe_sha1_main` if CTR is not zero, processing the next block. */

	FINALIZE /* Functional Utility: Restores saved SPE registers, clears sensitive stack data, and cleans up the stack frame. */
	blr /* Functional Utility: Returns from the function. */

.data
.align 4
PPC_SPE_SHA1_K:
	.long 0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xCA62C1D6 /* Functional Role: SHA-1 round constants for K0-19, K20-39, K40-59, K60-79. */
