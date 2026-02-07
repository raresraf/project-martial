/* SPDX-License-Identifier: GPL-2.0 */
/**
 * @file sha1-powerpc-asm.S
 * @brief Highly optimized SHA-1 implementation for PowerPC architecture.
 * @details This assembly file provides a performance-critical implementation
 * of the SHA-1 (Secure Hash Algorithm 1) compression function, specifically
 * tuned for the PowerPC architecture. It employs a number of low-level
 * optimization techniques to maximize throughput, including:
 * - **Register Rolling/Rotation**: SHA-1 state variables (A, B, C, D, E) are
 *   mapped to a rotating set of physical registers (r7-r12) to simplify the
 *   algorithm's iterative nature and reduce register moves.
 * - **Macro-driven Loop Unrolling**: Extensive use of macros (`STEPD*`, `STEPUP*`)
 *   to unroll multiple SHA-1 rounds per iteration, thereby reducing loop overhead
 *   and increasing instruction-level parallelism.
 * - **Instruction Scheduling**: Instructions are carefully ordered within macros
 *   to utilize the PowerPC's pipeline efficiently.
 * - **Endianness Handling**: The `LWZ` macro conditionally handles byte-reversal
 *   for little-endian systems to ensure correct message word ordering for SHA-1.
 * This implementation is designed to provide maximum SHA-1 hashing performance
 * on PowerPC-based systems within the Linux kernel.
 */
/*
 * SHA-1 implementation for PowerPC.
 *
 * Copyright (C) 2005 Paul Mackerras <paulus@samba.org>
 */

#include <asm/ppc_asm.h>
#include <asm/asm-offsets.h>
#include <asm/asm-compat.h>

/**
 * @brief Loads a 32-bit word from memory, conditionally byte-swapping for endianness.
 * @details This macro provides an endian-aware way to load a 32-bit word (`lwz`)
 * from memory. On little-endian systems (`__BIG_ENDIAN__` not defined), it uses
 * `lwbrx` (Load Word Byte-Reverse Indexed) to automatically perform byte-reversal,
 * ensuring that the loaded word is in big-endian format, as required by the SHA-1
 * algorithm, regardless of the CPU's native endianness.
 *
 * @param rt Destination register.
 * @param d  Offset from base register.
 * @param ra Base register.
 * Functional Utility: Loads a 32-bit word from memory into a register, handling endianness for SHA-1 processing.
 */
#ifdef __BIG_ENDIAN__
#define LWZ(rt, d, ra)	
	lwz	rt,d(ra)
#else
#define LWZ(rt, d, ra)	
	li	rt,d;	
	lwbrx	rt,rt,ra
#endif

/*
 * We roll the registers for T, A, B, C, D, E around on each
 * iteration; T on iteration t is A on iteration t+1, and so on.
 * We use registers 7 - 12 for this.
 */
/**
 * @brief Macro for dynamic register assignment for SHA-1 state variables.
 * @details These macros (`RT`, `RA`, `RB`, `RC`, `RD`, `RE`) implement a
 * register rolling/rotation scheme. They map the conceptual SHA-1 state
 * variables (A, B, C, D, E, and T for temporary) to a fixed set of physical
 * PowerPC registers (r7-r12). By adjusting the modulo 6 offset based on the
 * current `t` (iteration index), they ensure that the A-E variables "rotate"
 * through these physical registers on each SHA-1 round. This is a key
 * optimization to reduce explicit register moves.
 *
 * @param t Current iteration index (0-79).
 * Functional Utility: Provides rotating register aliases for SHA-1 state variables (A-E) to optimize data flow.
 */
#define RT(t)	((((t)+5)%6)+7) /* Functional Role: Aliases a rotating register for the temporary T value. */
#define RA(t)	((((t)+4)%6)+7) /* Functional Role: Aliases a rotating register for SHA-1 state variable A. */
#define RB(t)	((((t)+3)%6)+7) /* Functional Role: Aliases a rotating register for SHA-1 state variable B. */
#define RC(t)	((((t)+2)%6)+7) /* Functional Role: Aliases a rotating register for SHA-1 state variable C. */
#define RD(t)	((((t)+1)%6)+7) /* Functional Role: Aliases a rotating register for SHA-1 state variable D. */
#define RE(t)	((((t)+0)%6)+7) /* Functional Role: Aliases a rotating register for SHA-1 state variable E. */

/* We use registers 16 - 31 for the W values */
/**
 * @brief Macro for dynamic register assignment for SHA-1 message words W.
 * @details This macro maps the SHA-1 message schedule words (W[t]) to a set
 * of physical registers (r16-r31). The modulo 16 ensures that only 16
 * physical registers are used to store the 16 working message words needed
 * for the message schedule expansion.
 *
 * @param t Current message word index (0-15).
 * Functional Utility: Provides a rotating register alias for SHA-1 message words W[t].
 */
#define W(t)	(((t)%16)+16)

/**
 * @brief Loads a message word W[t] from memory.
 * @details This macro uses the `LWZ` macro to load a 32-bit message word
 * from the input data buffer (r4) at a specific offset (`(t)*4`) into the
 * register allocated for W(t). It handles endianness conversion if necessary.
 *
 * @param t Message word index.
 * Functional Utility: Loads a 32-bit message word into a W register, handling endianness.
 */
#define LOADW(t)				
	LWZ(W(t),(t)*4,r4)

/**
 * @brief Performs one SHA-1 round (0-19) with simultaneous message word loading.
 * @details This macro implements one round of the SHA-1 algorithm for rounds
 * 0-19, using the logical function `(B & C) | (~B & D)`. It is designed to
 * interleave message word loading (`LWZ(W((t)+4),((t)+4)*4,r4)`) from the input
 * buffer with the current round's computation, thereby hiding memory latency.
 *
 * @param t Current round index.
 * Functional Utility: Executes a SHA-1 round (0-19) and loads the next message word for future rounds.
 */
#define STEPD0_LOAD(t)				
	andc	r0,RD(t),RB(t);		/* Functional Utility: Computes (~D) & B part of F(B,C,D) */ 
	and	r6,RB(t),RC(t);		/* Functional Utility: Computes B & C part of F(B,C,D) */ 
	rotlwi	RT(t),RA(t),5;			/* Functional Utility: Rotates A by 5 bits left (A = ROL(A, 5)) */ 
	or	r6,r6,r0;			/* Functional Utility: Completes F(B,C,D) = (B & C) | ((~D) & B) */ 
	add	r0,RE(t),r15;			/* Functional Utility: Adds E and the round constant K */ 
	add	RT(t),RT(t),r6;		/* Functional Utility: Adds F(B,C,D) to A */ 
	add	r14,r0,W(t);			/* Functional Utility: Adds W(t) to E + K */ 
	LWZ(W((t)+4),((t)+4)*4,r4);	/* Functional Utility: Loads the next message word W[t+4] from input, pipelined. */ 
	rotlwi	RB(t),RB(t),30;			/* Functional Utility: Rotates B by 30 bits left (B = ROL(B, 30)) */ 
	add	RT(t),RT(t),r14 /* Functional Utility: Adds E + K + W(t) to A to get new T (next A) */

/**
 * @brief Performs one SHA-1 round (0-19) with simultaneous message schedule update.
 * @details This macro implements one round of the SHA-1 algorithm for rounds
 * 0-19, using the logical function `(B & C) | (~B & D)`. Crucially, it
 * simultaneously updates the message schedule (`W((t)+4)`) using the SHA-1
 * message expansion formula `W[i] = ROL((W[i-3]^W[i-8]^W[i-14]^W[i-16]), 1)`,
 * thereby hiding the latency of message word generation.
 *
 * @param t Current round index.
 * Functional Utility: Executes a SHA-1 round (0-19) and updates the message schedule for future rounds.
 */
#define STEPD0_UPDATE(t)			
	and	r6,RB(t),RC(t);		/* Functional Utility: Computes B & C part of F(B,C,D) */ 
	andc	r0,RD(t),RB(t);		/* Functional Utility: Computes (~B) & D part of F(B,C,D) */ 
	rotlwi	RT(t),RA(t),5;			/* Functional Utility: Rotates A by 5 bits left (A = ROL(A, 5)) */ 
	rotlwi	RB(t),RB(t),30;			/* Functional Utility: Rotates B by 30 bits left (B = ROL(B, 30)) */ 
	or	r6,r6,r0;			/* Functional Utility: Completes F(B,C,D) */ 
	add	r0,RE(t),r15;			/* Functional Utility: Adds E and the round constant K */ 
	xor	r5,W((t)+4-3),W((t)+4-8);		/* Functional Utility: Computes W[i-3] ^ W[i-8] for message schedule expansion. */ 
	add	RT(t),RT(t),r6;		/* Functional Utility: Adds F(B,C,D) to A */ 
	xor	W((t)+4),W((t)+4-16),W((t)+4-14);	/* Functional Utility: Computes W[i-16] ^ W[i-14] for message schedule expansion. */ 
	add	r0,r0,W(t);			/* Functional Utility: Adds W(t) to E + K */ 
	xor	W((t)+4),W((t)+4),r5;			/* Functional Utility: Completes W[i] = W[i-16] ^ W[i-14] ^ W[i-8] ^ W[i-3]. */ 
	add	RT(t),RT(t),r0;		/* Functional Utility: Adds E + K + W(t) to A to get new T (next A) */ 
	rotlwi	W((t)+4),W((t)+4),1 /* Functional Utility: Rotates the newly computed W by 1 bit left (ROL(W,1)) for message schedule. */

/**
 * @brief Performs one SHA-1 round (20-39 or 60-79) without message schedule update.
 * @details This macro implements one round of the SHA-1 algorithm for rounds
 * 20-39 or 60-79, using the logical function `(B ^ C ^ D)`. It focuses solely
 * on the hash state update without modifying the message schedule (which is
 * already computed by this stage).
 *
 * @param t Current round index.
 * Functional Utility: Executes a SHA-1 round (20-39 or 60-79) with the XOR logical function.
 */
#define STEPD1(t)				
	xor	r6,RB(t),RC(t);		/* Functional Utility: Computes B ^ C part of F(B,C,D) */ 
	rotlwi	RT(t),RA(t),5;			/* Functional Utility: Rotates A by 5 bits left (A = ROL(A, 5)) */ 
	rotlwi	RB(t),RB(t),30;			/* Functional Utility: Rotates B by 30 bits left (B = ROL(B, 30)) */ 
	xor	r6,r6,RD(t);			/* Functional Utility: Completes F(B,C,D) = B ^ C ^ D */ 
	add	r0,RE(t),r15;			/* Functional Utility: Adds E and the round constant K */ 
	add	RT(t),RT(t),r6;		/* Functional Utility: Adds F(B,C,D) to A */ 
	add	r0,r0,W(t);			/* Functional Utility: Adds W(t) to E + K */ 
	add	RT(t),RT(t),r0 /* Functional Utility: Adds E + K + W(t) to A to get new T (next A) */

/**
 * @brief Performs one SHA-1 round (20-39 or 60-79) with simultaneous message schedule update.
 * @details This macro implements one round of the SHA-1 algorithm for rounds
 * 20-39 or 60-79, using the logical function `(B ^ C ^ D)`. It simultaneously
 * updates the message schedule (`W((t)+4)`) using the SHA-1 message expansion
 * formula, similar to `STEPD0_UPDATE`.
 *
 * @param t Current round index.
 * Functional Utility: Executes a SHA-1 round (20-39 or 60-79) and updates the message schedule for future rounds.
 */
#define STEPD1_UPDATE(t)				
	xor	r6,RB(t),RC(t);		/* Functional Utility: Computes B ^ C part of F(B,C,D) */ 
	rotlwi	RT(t),RA(t),5;			/* Functional Utility: Rotates A by 5 bits left (A = ROL(A, 5)) */ 
	rotlwi	RB(t),RB(t),30;			/* Functional Utility: Rotates B by 30 bits left (B = ROL(B, 30)) */ 
	xor	r6,r6,RD(t);			/* Functional Utility: Completes F(B,C,D) = B ^ C ^ D */ 
	add	r0,RE(t),r15;			/* Functional Utility: Adds E and the round constant K */ 
	xor	r5,W((t)+4-3),W((t)+4-8);		/* Functional Utility: Computes W[i-3] ^ W[i-8] for message schedule expansion. */ 
	add	RT(t),RT(t),r6;		/* Functional Utility: Adds F(B,C,D) to A */ 
	xor	W((t)+4),W((t)+4-16),W((t)+4-14);	/* Functional Utility: Computes W[i-16] ^ W[i-14] for message schedule expansion. */ 
	add	r0,r0,W(t);			/* Functional Utility: Adds W(t) to E + K */ 
	xor	W((t)+4),W((t)+4),r5;			/* Functional Utility: Completes W[i] = W[i-16] ^ W[i-14] ^ W[i-8] ^ W[i-3]. */ 
	add	RT(t),RT(t),r0;		/* Functional Utility: Adds E + K + W(t) to A to get new T (next A) */ 
	rotlwi	W((t)+4),W((t)+4),1 /* Functional Utility: Rotates the newly computed W by 1 bit left (ROL(W,1)) for message schedule. */

/**
 * @brief Performs one SHA-1 round (40-59) with simultaneous message schedule update.
 * @details This macro implements one round of the SHA-1 algorithm for rounds
 * 40-59, using the logical function `(B & C) | (B & D) | (C & D)` (Majority Function).
 * It simultaneously updates the message schedule (`W((t)+4)`) using the SHA-1
 * message expansion formula.
 *
 * @param t Current round index.
 * Functional Utility: Executes a SHA-1 round (40-59) and updates the message schedule for future rounds.
 */
#define STEPD2_UPDATE(t)			
	and	r6,RB(t),RC(t);		/* Functional Utility: Computes B & C part of F(B,C,D) */ 
	and	r0,RB(t),RD(t);		/* Functional Utility: Computes B & D part of F(B,C,D) */ 
	rotlwi	RT(t),RA(t),5;			/* Functional Utility: Rotates A by 5 bits left (A = ROL(A, 5)) */ 
	or	r6,r6,r0;			/* Functional Utility: Combines B & C and B & D. */ 
	rotlwi	RB(t),RB(t),30;			/* Functional Utility: Rotates B by 30 bits left (B = ROL(B, 30)) */ 
	and	r0,RC(t),RD(t);		/* Functional Utility: Computes C & D part of F(B,C,D) */ 
	xor	r5,W((t)+4-3),W((t)+4-8);	/* Functional Utility: Computes W[i-3] ^ W[i-8] for message schedule expansion. */ 
	or	r6,r6,r0;			/* Functional Utility: Completes F(B,C,D) = (B & C) | (B & D) | (C & D). */ 
	xor	W((t)+4),W((t)+4-16),W((t)+4-14);	/* Functional Utility: Computes W[i-16] ^ W[i-14] for message schedule expansion. */ 
	add	r0,RE(t),r15;			/* Functional Utility: Adds E and the round constant K */ 
	add	RT(t),RT(t),r6;		/* Functional Utility: Adds F(B,C,D) to A */ 
	add	r0,r0,W(t);			/* Functional Utility: Adds W(t) to E + K */ 
	xor	W((t)+4),W((t)+4),r5;		/* Functional Utility: Completes W[i] = W[i-16] ^ W[i-14] ^ W[i-8] ^ W[i-3]. */ 
	add	RT(t),RT(t),r0;		/* Functional Utility: Adds E + K + W(t) to A to get new T (next A) */ 
	rotlwi	W((t)+4),W((t)+4),1 /* Functional Utility: Rotates the newly computed W by 1 bit left (ROL(W,1)) for message schedule. */

/**
 * @brief Unrolls 4 consecutive SHA-1 rounds, loading message words from input.
 * @details This macro orchestrates four consecutive calls to `STEPD0_LOAD`,
 * effectively processing 4 rounds of SHA-1 while simultaneously loading the
 * corresponding message words from the input buffer. This reduces loop
 * overhead for the initial rounds where message words are directly consumed.
 *
 * @param t Starting round index.
 * Functional Utility: Unrolls 4 SHA-1 rounds (0-19), loading message words from input.
 */
#define STEP0LD4(t)				
	STEPD0_LOAD(t);				
	STEPD0_LOAD((t)+1);			
	STEPD0_LOAD((t)+2);			
	STEPD0_LOAD((t)+3)

/**
 * @brief Unrolls 4 consecutive SHA-1 rounds using a specified update function.
 * @details This macro orchestrates four consecutive calls to a specified `STEPD*`
 * update macro (e.g., `STEPD0_UPDATE`, `STEPD1_UPDATE`, `STEPD2_UPDATE`). This
 * effectively processes 4 rounds of SHA-1, including message schedule updates
 * if applicable, thereby reducing loop overhead and increasing throughput.
 *
 * @param t Starting round index.
 * @param fn The specific `STEPD*` update function macro to use (e.g., `D0`, `D1`, `D2`).
 * Functional Utility: Unrolls 4 SHA-1 rounds using a specified message schedule and round logic.
 */
#define STEPUP4(t, fn)				
	STEP##fn##_UPDATE(t);			
	STEP##fn##_UPDATE((t)+1);		
	STEP##fn##_UPDATE((t)+2);		
	STEP##fn##_UPDATE((t)+3)

/**
 * @brief Unrolls 20 consecutive SHA-1 rounds using a specified update function.
 * @details This macro orchestrates five consecutive calls to `STEPUP4`,
 * effectively processing 20 rounds of SHA-1 at a time. This provides a high
 * degree of loop unrolling for the main bulk of the SHA-1 rounds (e.g., 20-39, 40-59, 60-79),
 * significantly reducing loop overhead and maximizing instruction-level parallelism.
 *
 * @param t Starting round index.
 * @param fn The specific `STEPD*` update function macro to use.
 * Functional Utility: Unrolls 20 SHA-1 rounds, optimizing for a large block of operations.
 */
#define STEPUP20(t, fn)				
	STEPUP4(t, fn);				
	STEPUP4((t)+4, fn);			
	STEPUP4((t)+8, fn);			
	STEPUP4((t)+12, fn);			
	STEPUP4((t)+16, fn)

/**
 * @brief PowerPC optimized SHA-1 transform function.
 * @details This function implements the core SHA-1 compression for a 64-byte
 * message block, highly optimized for the PowerPC architecture using assembly
 * instructions. It utilizes a rotating register scheme for SHA-1 state variables,
 * extensive loop unrolling via macros, and interleaved message schedule updates
 * to maximize performance.
 *
 * @param r3 Pointer to the SHA-1 state array (H0-H4).
 * @param r4 Pointer to the 64-byte input data block.
 * Functional Utility: Performs hardware-optimized SHA-1 compression on a 64-byte block using PowerPC assembly.
 */
_GLOBAL(powerpc_sha_transform)
	/* Prologue: Save registers and set up the stack frame. */
	PPC_STLU r1,-INT_FRAME_SIZE(r1) /* Functional Utility: Adjusts stack pointer and stores a register on stack. */
	SAVE_GPRS(14, 31, r1) /* Functional Utility: Saves general-purpose registers (r14-r31) onto the stack. */

	/* Load up initial SHA-1 state (A - E) into rotating registers. */
	lwz	RA(0),0(r3)	/* A */
	lwz	RB(0),4(r3)	/* B */
	lwz	RC(0),8(r3)	/* C */
	lwz	RD(0),12(r3)	/* D */
	lwz	RE(0),16(r3)	/* E */

	/* Initial loading of message words W[0]-W[3] from input. */
	LOADW(0)
	LOADW(1)
	LOADW(2)
	LOADW(3)

	/* Rounds 0-19: Initial processing with logical function F(B,C,D) = (B&C) | (~B&D) */
	lis	r15,0x5a82	/* Functional Utility: Loads upper half of round constant K0-19 (0x5a827999) into r15. */
	ori	r15,r15,0x7999 /* Functional Utility: Loads lower half of round constant K0-19 (0x5a827999) into r15. */
	STEP0LD4(0) /* Functional Utility: Processes rounds 0-3, loading message words. */
	STEP0LD4(4) /* Functional Utility: Processes rounds 4-7, loading message words. */
	STEP0LD4(8) /* Functional Utility: Processes rounds 8-11, loading message words. */
	STEPUP4(12, D0) /* Functional Utility: Processes rounds 12-15, updating message schedule. */
	STEPUP4(16, D0) /* Functional Utility: Processes rounds 16-19, updating message schedule. */

	/* Rounds 20-39: Logical function F(B,C,D) = B ^ C ^ D (Parity Function) */
	lis	r15,0x6ed9	/* Functional Utility: Loads upper half of round constant K20-39 (0x6ed9eba1) into r15. */
	ori	r15,r15,0xeba1 /* Functional Utility: Loads lower half of round constant K20-39 (0x6ed9eba1) into r15. */
	STEPUP20(20, D1) /* Functional Utility: Processes rounds 20-39, updating message schedule. */

	/* Rounds 40-59: Logical function F(B,C,D) = (B&C) | (B&D) | (C&D) (Majority Function) */
	lis	r15,0x8f1b	/* Functional Utility: Loads upper half of round constant K40-59 (0x8f1bbcdc) into r15. */
	ori	r15,r15,0xbcdc /* Functional Utility: Loads lower half of round constant K40-59 (0x8f1bbcdc) into r15. */
	STEPUP20(40, D2) /* Functional Utility: Processes rounds 40-59, updating message schedule. */

	/* Rounds 60-79: Logical function F(B,C,D) = B ^ C ^ D (Parity Function) */
	lis	r15,0xca62	/* Functional Utility: Loads upper half of round constant K60-79 (0xca62c1d6) into r15. */
	ori	r15,r15,0xc1d6 /* Functional Utility: Loads lower half of round constant K60-79 (0xca62c1d6) into r15. */
	STEPUP4(60, D1) /* Functional Utility: Processes rounds 60-63, updating message schedule. */
	STEPUP4(64, D1) /* Functional Utility: Processes rounds 64-67, updating message schedule. */
	STEPUP4(68, D1) /* Functional Utility: Processes rounds 68-71, updating message schedule. */
	STEPUP4(72, D1) /* Functional Utility: Processes rounds 72-75, updating message schedule. */
	lwz	r20,16(r3) /* Functional Utility: Loads original H4 (E) into r20 for final state update. */
	STEPD1(76) /* Functional Utility: Processes round 76. */
	lwz	r19,12(r3) /* Functional Utility: Loads original H3 (D) into r19 for final state update. */
	STEPD1(77) /* Functional Utility: Processes round 77. */
	lwz	r18,8(r3) /* Functional Utility: Loads original H2 (C) into r18 for final state update. */
	STEPD1(78) /* Functional Utility: Processes round 78. */
	lwz	r17,4(r3) /* Functional Utility: Loads original H1 (B) into r17 for final state update. */
	STEPD1(79) /* Functional Utility: Processes round 79. */

	/* Epilogue: Update the initial state with the results of the compression. */
	lwz	r16,0(r3) /* Functional Utility: Loads original H0 (A) into r16 for final state update. */
	add	r20,RE(80),r20 /* Functional Utility: Adds the result of E (RE(80)) to original H4 (r20). */
	add	RD(0),RD(80),r19 /* Functional Utility: Adds the result of D (RD(80)) to original H3 (r19). */
	add	RC(0),RC(80),r18 /* Functional Utility: Adds the result of C (RC(80)) to original H2 (r18). */
	add	RB(0),RB(80),r17 /* Functional Utility: Adds the result of B (RB(80)) to original H1 (r17). */
	add	RA(0),RA(80),r16 /* Functional Utility: Adds the result of A (RA(80)) to original H0 (r16). */
	mr	RE(0),r20 /* Functional Utility: Moves updated H4 into RE(0) for storing. */
	stw	RA(0),0(r3) /* Functional Utility: Stores updated H0 back to memory. */
	stw	RB(0),4(r3) /* Functional Utility: Stores updated H1 back to memory. */
	stw	RC(0),8(r3) /* Functional Utility: Stores updated H2 back to memory. */
	stw	RD(0),12(r3) /* Functional Utility: Stores updated H3 back to memory. */
	stw	RE(0),16(r3) /* Functional Utility: Stores updated H4 back to memory. */

	REST_GPRS(14, 31, r1) /* Functional Utility: Restores general-purpose registers (r14-r31) from the stack. */
	addi	r1,r1,INT_FRAME_SIZE /* Functional Utility: Adjusts the stack pointer to deallocate the stack frame. */
	blr /* Functional Utility: Returns from the function. */
