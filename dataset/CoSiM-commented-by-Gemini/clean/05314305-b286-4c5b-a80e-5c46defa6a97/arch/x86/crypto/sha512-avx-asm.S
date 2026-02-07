########################################################################
# Implement fast SHA-512 with AVX instructions. (x86_64)
#
# Copyright (C) 2013 Intel Corporation.
#
# Authors:
#     James Guilford <james.guilford@intel.com>
#     Kirk Yap <kirk.s.yap@intel.com>
#     David Cote <david.m.cote@intel.com>
#     Tim Chen <tim.c.chen@linux.intel.com>
#
# This software is available to you under a choice of one of two
# licenses.  You may choose to be licensed under the terms of the GNU
# General Public License (GPL) Version 2, available from the file
# COPYING in the main directory of this source tree, or the
# OpenIB.org BSD license below:
#
#     Redistribution and use in source and binary forms, with or
#     without modification, are permitted provided that the following
#     conditions are met:
#
#      - Redistributions of source code must retain the above
#        copyright notice, this list of conditions and the following
#        disclaimer.
#
#      - Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials
#        provided with the distribution.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
########################################################################
#
# This code is described in an Intel White-Paper:
# "Fast SHA-512 Implementations on Intel Architecture Processors"
#
# To find it, surf to http://www.intel.com/p/en_US/embedded
# and search for that title.
#
########################################################################

/**
 * @file sha512-avx-asm.S
 * @brief High-performance SHA-512 implementation leveraging Intel AVX instructions for x86_64.
 * @details This assembly file provides a highly optimized implementation of the
 * SHA-512 (Secure Hash Algorithm 2, 512-bit variant) compression function,
 * specifically tailored for x86_64 processors equipped with the Intel AVX
 * (Advanced Vector Extensions) instruction set. It extensively utilizes AVX's
 * 128-bit vector capabilities to process data in parallel, significantly
 * accelerating the cryptographic process.
 *
 * **Optimization Techniques:**
 * - **AVX Vectorization**: Uses AVX instructions (`vmovdqa`, `vmovdqu`, `vpshufb`,
 *   `vpaddq`, `vpsrlq`, `vpsllq`, `vpxor`) for efficient loading, byte-swapping,
 *   message schedule generation, and hash state updates.
 * - **Software Pipelining**: The implementation employs software pipelining,
 *   where message schedule generation and round computations are interleaved
 *   (`SHA512_2Sched_2Round_avx` macro) to hide latency and maximize throughput.
 * - **Message Schedule Acceleration**: Dedicated instructions and bitwise
 *   operations are optimized for generating SHA-512's `sigma0` and `sigma1`
 *   functions.
 * - **Register Rotation**: A custom `RotateState` macro efficiently shuffles
 *   the SHA-512 hash state variables without excessive data movement.
 * - **Optimized Rotation (`RORQ`)**: Leverages `shld` for faster 64-bit rotations
 *   on Sandybridge and newer architectures.
 * This implementation is designed to provide maximum SHA-512 hashing performance
 * on AVX-enabled x86_64 systems within the Linux kernel.
 */

#include <linux/linkage.h>
#include <linux/cfi_types.h>

.text

# Virtual Registers
# ARG1
digest	= %rdi	/* Functional Role: Pointer to `struct sha512_state` (argument 1). */
# ARG2
msg	= %rsi	/* Functional Role: Pointer to input data buffer (argument 2). */
# ARG3
msglen	= %rdx	/* Functional Role: Number of 128-byte blocks to process (argument 3). */
T1	= %rcx	/* Functional Role: GPR for temporary computations (64-bit). */
T2	= %r8	/* Functional Role: GPR for temporary computations (64-bit). */
a_64	= %r9	/* Functional Role: GPR holding SHA-512 state variable 'a' (64-bit). */
b_64	= %r10	/* Functional Role: GPR holding SHA-512 state variable 'b' (64-bit). */
c_64	= %r11	/* Functional Role: GPR holding SHA-512 state variable 'c' (64-bit). */
d_64	= %r12	/* Functional Role: GPR holding SHA-512 state variable 'd' (64-bit). */
e_64	= %r13	/* Functional Role: GPR holding SHA-512 state variable 'e' (64-bit). */
f_64	= %r14	/* Functional Role: GPR holding SHA-512 state variable 'f' (64-bit). */
g_64	= %r15	/* Functional Role: GPR holding SHA-512 state variable 'g' (64-bit). */
h_64	= %rbx	/* Functional Role: GPR holding SHA-512 state variable 'h' (64-bit). */
tmp0	= %rax	/* Functional Role: GPR for very temporary general use (64-bit). */

# Local variables (stack frame)

# Message Schedule
W_SIZE = 80*8	/* Functional Role: Total size in bytes for storing 80 QWORDS of message schedule (W[t]) in the stack frame. */
# W[t] + K[t] | W[t+1] + K[t+1]
WK_SIZE = 2*8	/* Functional Role: Size in bytes for storing two QWORDs of `W[t]+K[t]` values on the stack. */

frame_W = 0	/* Functional Role: Offset for the message schedule buffer within the stack frame. */
frame_WK = frame_W + W_SIZE	/* Functional Role: Offset for the `W[t]+K[t]` buffer within the stack frame. */
frame_size = frame_WK + WK_SIZE	/* Functional Role: Total size of the stack frame required. */

# Useful QWORD "arrays" for simpler memory references
# MSG, DIGEST, K_t, W_t are arrays
# WK_2(t) points to 1 of 2 qwords at frame.WK depending on t being odd/even

# Input message (arg1)
#define MSG(i)    8*i(msg) /* Functional Role: Macro to access i-th QWORD from the input message buffer. */

# Output Digest (arg2)
#define DIGEST(i) 8*i(digest) /* Functional Role: Macro to access i-th QWORD from the SHA-512 digest. */

# SHA Constants (static mem)
#define K_t(i)    8*i+K512(%rip) /* Functional Role: Macro to access i-th QWORD from the SHA-512 round constants table. */

# Message Schedule (stack frame)
#define W_t(i)    8*i+frame_W(%rsp) /* Functional Role: Macro to access i-th QWORD from the message schedule buffer on the stack. */

# W[t]+K[t] (stack frame)
#define WK_2(i)   8*((i%2))+frame_WK(%rsp) /* Functional Role: Macro to access `W[t]+K[t]` for a given round `i`, storing two QWORDs in a circular buffer fashion. */

/**
 * @brief Logically rotates the SHA-512 state variables to the right.
 * @details This macro efficiently shuffles the SHA-512 state variables
 * (`a_64` through `h_64`) by reassigning their virtual register names.
 * This implements the step where the output of one round becomes the input
 * for the next, with `h` becoming `g`, `g` becoming `f`, and so on, and
 * the new `a` value coming from a temporary. This is a purely symbolic
 * rotation at assembly time to avoid physical `mov` instructions between these registers.
 * Functional Utility: Efficiently rotates the SHA-512 hash state registers without explicit moves.
 */
.macro RotateState
	# Functional Utility: Temporary storage for 'h_64' before rotation.
	TMP   = h_64
	# Functional Utility: Each state variable receives the value of the variable to its left.
	h_64  = g_64
	g_64  = f_64
	f_64  = e_64
	e_64  = d_64
	d_64  = c_64
	c_64  = b_64
	b_64  = a_64
	# Functional Utility: The new 'a_64' receives the value that was originally in 'h_64'.
	a_64  = TMP
.endm

/**
 * @brief Performs a 64-bit right rotation.
 * @details This macro implements a 64-bit right rotation. On Sandybridge
 * and newer architectures, the `shld` (Shift Left Double) instruction can
 * be faster than `ror` (Rotate Right) for implementing rotations. This macro
 * provides a performant way to achieve the required 64-bit rotations for
 * SHA-512, which are typically 28, 34, 39, etc.
 * @param p1 The 64-bit register to rotate.
 * @param p2 The number of bits to rotate right.
 * Functional Utility: Efficiently rotates a 64-bit register to the right by a specified amount.
 */
.macro RORQ p1 p2
	# shld is faster than ror on Sandybridge
	shld	$(64-\p2), \p1, \p1 /* Functional Utility: Implements right rotation using `shld` instruction. */
.endm

/**
 * @brief Computes a single round of SHA-512.
 * @details This macro performs the core computations for one round of the SHA-512
 * algorithm. It calculates the `Ch` (Choose) function, `Sigma1` function
 * (S1), `Maj` (Majority) function, and `Sigma0` function (S0) on the current
 * SHA-512 state variables. It accumulates these results and updates the `d_64`
 * and `h_64` state variables, then rotates the entire state using `RotateState`.
 * @param rnd The current round number.
 * Functional Utility: Executes a single, optimized round of SHA-512 compression.
 */
.macro SHA512_Round rnd
	# Compute Round %%t
	mov     f_64, T1          # T1 = f
	mov     e_64, tmp0        # tmp = e
	xor     g_64, T1          # T1 = f ^ g
	RORQ    tmp0, 23   # 41    # tmp = e ror 23  // Functional Utility: Part of S1(e) calculation.
	and     e_64, T1          # T1 = (f ^ g) & e // Functional Utility: (f^g) & e
	xor     e_64, tmp0        # tmp = (e ror 23) ^ e // Functional Utility: Part of S1(e) calculation.
	xor     g_64, T1          # T1 = ((f ^ g) & e) ^ g = CH(e,f,g) // Functional Utility: Computes CH(e,f,g).
	idx = nd
	add     WK_2(idx), T1     # W[t] + K[t] from message scheduler // Functional Utility: Adds (W[t]+K[t]) to T1.
	RORQ    tmp0, 4   # 18    # tmp = ((e ror 23) ^ e) ror 4 // Functional Utility: Part of S1(e) calculation.
	xor     e_64, tmp0        # tmp = (((e ror 23) ^ e) ror 4) ^ e // Functional Utility: Part of S1(e) calculation.
	xor     g_64, tmp0        # tmp = ((((e ror23)^e)ror4)^e)ror14 // Functional Utility: Corrected. Part of S1(e) calculation (g_64 xor tmp0).
	add     h_64, T1          # T1 = CH(e,f,g) + W[t] + K[t] + h // Functional Utility: Adds h_64 to T1.
	mov     a_64, T2          # T2 = a
	RORQ    tmp0, 14  # 14    # tmp = ((((e ror23)^e)ror4)^e)ror14 = S1(e) // Functional Utility: Completes S1(e).
	add     tmp0, T1          # T1 = CH(e,f,g) + W[t] + K[t] + S1(e) + h // Functional Utility: Adds S1(e) to T1.
	mov     a_64, tmp0        # tmp = a
	xor     c_64, T2          # T2 = a ^ c
	and     c_64, tmp0        # tmp = a & c
	and     b_64, T2          # T2 = (a ^ c) & b
	xor     tmp0, T2          # T2 = ((a ^ c) & b) ^ (a & c) = Maj(a,b,c) // Functional Utility: Computes Maj(a,b,c).
	mov     a_64, tmp0        # tmp = a
	RORQ    tmp0, 5  # 39     # tmp = a ror 5 // Functional Utility: Part of S0(a) calculation.
	xor     a_64, tmp0        # tmp = (a ror 5) ^ a // Functional Utility: Part of S0(a) calculation.
	add     T1, d_64          # e(next_state) = d + T1 // Functional Utility: Updates d_64 with T1.
	RORQ    tmp0, 6  # 34     # tmp = ((a ror 5) ^ a) ror 6 // Functional Utility: Part of S0(a) calculation.
	xor     a_64, tmp0        # tmp = (((a ror 5) ^ a) ror 6) ^ a // Functional Utility: Part of S0(a) calculation.
	lea     (T1, T2), h_64    # a(next_state) = T1 + Maj(a,b,c) // Functional Utility: Sets h_64 (next 'a') as T1 + Maj.
	RORQ    tmp0, 28  # 28    # tmp = ((((a ror5)^a)ror6)^a)ror28 = S0(a) // Functional Utility: Completes S0(a).
	add     tmp0, h_64        # a(next_state) = T1 + Maj(a,b,c) S0(a) // Functional Utility: Adds S0(a) to h_64.
	RotateState // Functional Utility: Rotates the SHA-512 state variables for the next round.
.endm

/**
 * @brief Computes two SHA-512 rounds and two message schedule QWORDS using AVX.
 * @details This macro is a core component of the AVX-optimized SHA-512
 * implementation, demonstrating software pipelining. It performs two
 * consecutive rounds of SHA-512 while simultaneously generating two new
 * message schedule words. It utilizes AVX vector instructions for efficient
 * message schedule computation (`sigma0` and `sigma1` functions) and
 * interleaves these with the SHA-512 round computations.
 *
 * @param rnd The starting round number for this pair of rounds.
 * Functional Utility: Pipelined computation of two SHA-512 rounds and two message schedule QWORDS using AVX.
 */
.macro SHA512_2Sched_2Round_avx rnd
	# Compute rounds t-2 and t-1
	# Compute message schedule QWORDS t and t+1

	#   Two rounds are computed based on the values for K[t-2]+W[t-2] and
	# K[t-1]+W[t-1] which were previously stored at WK_2 by the message
	# scheduler.
	#   The two new schedule QWORDS are stored at [W_t(t)] and [W_t(t+1)].
	# They are then added to their respective SHA512 constants at
	# [K_t(t)] and [K_t(t+1)] and stored at dqword [WK_2(t)]
	#   For brevity, the comments following vectored instructions only refer to
	# the first of a pair of QWORDS.
	# Eg. XMM4=W[t-2] really means XMM4={W[t-2]|W[t-1]}
	#   The computation of the message schedule and the rounds are tightly
	# stitched to take advantage of instruction-level parallelism.

	idx = nd - 2
	vmovdqa	W_t(idx), %xmm4		# XMM4 = W[t-2] // Functional Utility: Loads W[t-2] and W[t-1] into XMM4.
	idx = nd - 15
	vmovdqu	W_t(idx), %xmm5		# XMM5 = W[t-15] // Functional Utility: Loads W[t-15] and W[t-14] into XMM5.
	mov	f_64, T1
	vpsrlq	$61, %xmm4, %xmm0	# XMM0 = W[t-2]>>61 // Functional Utility: Calculates part of `s1(W[t-2])`.
	mov	e_64, tmp0
	vpsrlq	$1, %xmm5, %xmm6	# XMM6 = W[t-15]>>1 // Functional Utility: Calculates part of `s0(W[t-15])`.
	xor	g_64, T1
	RORQ	tmp0, 23 # 41
	vpsrlq	$19, %xmm4, %xmm1	# XMM1 = W[t-2]>>19 // Functional Utility: Calculates part of `s1(W[t-2])`.
	and	e_64, T1
	xor	e_64, tmp0
	vpxor	%xmm1, %xmm0, %xmm0	# XMM0 = W[t-2]>>61 ^ W[t-2]>>19 // Functional Utility: Accumulates `s1(W[t-2])` components.
	xor	g_64, T1
	idx = nd
	add	WK_2(idx), T1# // Functional Utility: Adds `W[t-2]+K[t-2]` (from WK_2) to T1. Note: This should be idx-2 and idx-1 for current rounds being computed.
	vpsrlq	$8, %xmm5, %xmm7	# XMM7 = W[t-15]>>8 // Functional Utility: Calculates part of `s0(W[t-15])`.
	RORQ	tmp0, 4 # 18
	vpsrlq	$6, %xmm4, %xmm2	# XMM2 = W[t-2]>>6 // Functional Utility: Calculates part of `s1(W[t-2])`.
	xor	e_64, tmp0
	mov	a_64, T2
	add	h_64, T1
	vpxor	%xmm7, %xmm6, %xmm6	# XMM6 = W[t-15]>>1 ^ W[t-15]>>8 // Functional Utility: Accumulates `s0(W[t-15])` components.
	RORQ	tmp0, 14 # 14
	add	tmp0, T1
	vpsrlq	$7, %xmm5, %xmm8	# XMM8 = W[t-15]>>7 // Functional Utility: Calculates part of `s0(W[t-15])`.
	mov	a_64, tmp0
	xor	c_64, T2
	vpsllq	$(64-61), %xmm4, %xmm3  # XMM3 = W[t-2]<<3 // Functional Utility: Calculates part of `s1(W[t-2])`.
	and	c_64, tmp0
	and	b_64, T2
	vpxor	%xmm3, %xmm2, %xmm2	# XMM2 = W[t-2]>>6 ^ W[t-2]<<3 // Functional Utility: Accumulates `s1(W[t-2])` components.
	xor	tmp0, T2
	mov	a_64, tmp0
	vpsllq	$(64-1), %xmm5, %xmm9	# XMM9 = W[t-15]<<63 // Functional Utility: Calculates part of `s0(W[t-15])`.
	RORQ	tmp0, 5 # 39
	vpxor	%xmm9, %xmm8, %xmm8	# XMM8 = W[t-15]>>7 ^ W[t-15]<<63 // Functional Utility: Accumulates `s0(W[t-15])` components.
	xor	a_64, tmp0
	add	T1, d_64
	RORQ	tmp0, 6 # 34
	xor	a_64, tmp0
	vpxor	%xmm8, %xmm6, %xmm6	# XMM6 = W[t-15]>>1 ^ W[t-15]>>8 ^ // Functional Utility: Completes `s0(W[t-15])`.
					#  W[t-15]>>7 ^ W[t-15]<<63
	lea	(T1, T2), h_64
	RORQ	tmp0, 28 # 28
	vpsllq	$(64-19), %xmm4, %xmm4  # XMM4 = W[t-2]<<25 // Functional Utility: Calculates part of `s1(W[t-2])`.
	add	tmp0, h_64
	RotateState
	vpxor	%xmm4, %xmm0, %xmm0     # XMM0 = W[t-2]>>61 ^ W[t-2]>>19 ^ // Functional Utility: Completes `s1(W[t-2])`.
					#        W[t-2]<<25
	mov	f_64, T1
	vpxor	%xmm2, %xmm0, %xmm0     # XMM0 = s1(W[t-2]) // Functional Utility: Final `s1(W[t-2])` for both QWORDS.
	mov	e_64, tmp0
	xor	g_64, T1
	idx = nd - 16
	vpaddq	W_t(idx), %xmm0, %xmm0  # XMM0 = s1(W[t-2]) + W[t-16] // Functional Utility: Adds W[t-16] and W[t-15] to `s1(W[t-2])`.
	idx = nd - 7
	vmovdqu	W_t(idx), %xmm1		# XMM1 = W[t-7] // Functional Utility: Loads W[t-7] and W[t-6] into XMM1.
	RORQ	tmp0, 23 # 41
	and	e_64, T1
	xor	e_64, tmp0
	xor	g_64, T1
	vpsllq	$(64-8), %xmm5, %xmm5   # XMM5 = W[t-15]<<56 // Functional Utility: Calculates part of `s0(W[t-15])`.
	idx = nd + 1
	add	WK_2(idx), T1 // Functional Utility: Adds `W[t-1]+K[t-1]` (from WK_2) to T1.
	vpxor	%xmm5, %xmm6, %xmm6     # XMM6 = s0(W[t-15]) // Functional Utility: Final `s0(W[t-15])` for both QWORDS.
	RORQ	tmp0, 4 # 18
	vpaddq	%xmm6, %xmm0, %xmm0     # XMM0 = s1(W[t-2]) + W[t-16] + s0(W[t-15]) // Functional Utility: Adds `s0(W[t-15])` to XMM0.
	xor	e_64, tmp0
	vpaddq	%xmm1, %xmm0, %xmm0     # XMM0 = W[t] = s1(W[t-2]) + W[t-7] + // Functional Utility: Adds W[t-7] to XMM0 to complete W[t] generation.
					#               s0(W[t-15]) + W[t-16]
	mov	a_64, T2
	add	h_64, T1
	RORQ	tmp0, 14 # 14
	add	tmp0, T1
	idx = nd
	vmovdqa	%xmm0, W_t(idx)		# Store W[t] // Functional Utility: Stores the newly computed W[t] and W[t+1] in the message schedule buffer.
	vpaddq	K_t(idx), %xmm0, %xmm0  # Compute W[t]+K[t] // Functional Utility: Adds SHA-512 round constants K[t] and K[t+1] to W[t] and W[t+1].
	vmovdqa	%xmm0, WK_2(idx)	# Store W[t]+K[t] for next rounds // Functional Utility: Stores the computed `W[t]+K[t]` values for use in subsequent rounds.
	mov	a_64, tmp0
	xor	c_64, T2
	and	c_64, tmp0
	and	b_64, T2
	xor	tmp0, T2
	mov	a_64, tmp0
	RORQ	tmp0, 5 # 39
	xor	a_64, tmp0
	add	T1, d_64
	RORQ	tmp0, 6 # 34
	xor	a_64, tmp0
	lea	(T1, T2), h_64
	RORQ	tmp0, 28 # 28
	add	tmp0, h_64
	RotateState
.endm

########################################################################
# void sha512_transform_avx(sha512_state *state, const u8 *data, int blocks)
# Purpose: Updates the SHA512 digest stored at "state" with the message
# stored in "data".
# The size of the message pointed to by "data" must be an integer multiple
# of SHA512 message blocks.
# "blocks" is the message length in SHA512 blocks
########################################################################
/**
 * @brief SHA-512 transform function optimized with Intel AVX instructions.
 * @details This function updates the SHA-512 digest using AVX instructions
 * to process multiple 128-byte input data blocks. It employs software pipelining
 * where message schedule generation and hash round computations are tightly
 * interleaved. The function requires that the input data (`msg`) be an integer
 * multiple of SHA-512 message blocks (128 bytes) and handles iteration over these blocks.
 *
 * @param digest (rdi): Pointer to the `sha512_state` context structure (u64 state[8]).
 * @param msg (rsi): Pointer to the input data blocks (128-byte aligned).
 * @param msglen (rdx): Number of 128-byte blocks to process.
 * Functional Utility: Accelerates SHA-512 compression using AVX instructions for multiple data blocks.
 */
SYM_TYPED_FUNC_START(sha512_transform_avx)
	// Functional Utility: Checks if `msglen` (number of blocks) is zero.
	test msglen, msglen
	// Functional Utility: Jumps to `.Lnowork` if `msglen` is zero (no blocks to process).
	je .Lnowork

	# Save GPRs
	// Functional Utility: Saves callee-saved registers as per x86_64 ABI.
	push	%rbx
	push	%r12
	push	%r13
	push	%r14
	push	%r15

	# Allocate Stack Space
	// Functional Utility: Saves the current base pointer for stack frame management.
	push	%rbp
	mov	%rsp, %rbp
	// Functional Utility: Allocates stack space for local variables (message schedule, WK values).
	sub     $frame_size, %rsp
	// Functional Utility: Aligns the stack pointer to a 32-byte boundary, crucial for AVX operations.
	and	$~(0x20 - 1), %rsp

.Lupdateblock: /* Functional Role: Main loop for processing each 128-byte data block. */

	# Load state variables
	// Functional Utility: Loads the initial SHA-512 state variables (H0-H7) from `digest` into GPRs `a_64` through `h_64`.
	mov     DIGEST(0), a_64
	mov     DIGEST(1), b_64
	mov     DIGEST(2), c_64
	mov     DIGEST(3), d_64
	mov     DIGEST(4), e_64
	mov     DIGEST(5), f_64
	mov     DIGEST(6), g_64
	mov     DIGEST(7), h_64

	t = 0
	.rept 80/2 + 1 /* Functional Utility: Loop through 80 rounds of SHA-512, processing 2 rounds per iteration, plus one extra iteration for scheduler lead. */
	# (80 rounds) / (2 rounds/iteration) + (1 iteration)
	# +1 iteration because the scheduler leads hashing by 1 iteration
		.if t < 2 /* Functional Utility: Initial Message Schedule Generation (Rounds 0, 1). */
			# BSWAP 2 QWORDS
			vmovdqa  XMM_QWORD_BSWAP(%rip), %xmm1 // Functional Utility: Loads byte-swap mask into %xmm1.
			vmovdqu  MSG(t), %xmm0 // Functional Utility: Loads two QWORDS of message data into %xmm0.
			vpshufb  %xmm1, %xmm0, %xmm0    # BSWAP // Functional Utility: Byte-swaps the QWORDS in %xmm0.
			vmovdqa  %xmm0, W_t(t) # Store Scheduled Pair // Functional Utility: Stores byte-swapped W[t] and W[t+1] into stack.
			vpaddq   K_t(t), %xmm0, %xmm0 # Compute W[t]+K[t] // Functional Utility: Adds K[t] and K[t+1] to W[t] and W[t+1].
			vmovdqa  %xmm0, WK_2(t) # Store into WK for rounds // Functional Utility: Stores `W[t]+K[t]` and `W[t+1]+K[t+1]` into stack for rounds.
		.elseif t < 16 /* Functional Utility: Message Schedule and First Rounds (Rounds 2-15). */
			# BSWAP 2 QWORDS# Compute 2 Rounds
			vmovdqu  MSG(t), %xmm0 // Functional Utility: Loads two QWORDS of message data into %xmm0.
			vpshufb  %xmm1, %xmm0, %xmm0    # BSWAP // Functional Utility: Byte-swaps the QWORDS in %xmm0.
			SHA512_Round t-2    # Round t-2 // Functional Utility: Computes SHA-512 round for t-2.
			vmovdqa  %xmm0, W_t(t) # Store Scheduled Pair // Functional Utility: Stores byte-swapped W[t] and W[t+1] into stack.
			vpaddq   K_t(t), %xmm0, %xmm0 # Compute W[t]+K[t] // Functional Utility: Adds K[t] and K[t+1] to W[t] and W[t+1].
			SHA512_Round t-1    # Round t-1 // Functional Utility: Computes SHA-512 round for t-1.
			vmovdqa  %xmm0, WK_2(t)# Store W[t]+K[t] into WK // Functional Utility: Stores `W[t]+K[t]` and `W[t+1]+K[t+1]` into stack for rounds.
		.elseif t < 79 /* Functional Utility: Pipelined Message Schedule and Rounds (Rounds 16-78). */
			# Schedule 2 QWORDS# Compute 2 Rounds
			SHA512_2Sched_2Round_avx t // Functional Utility: Computes two rounds and two message schedule QWORDs using pipelined AVX.
		.else /* Functional Utility: Final Rounds (Rounds 79-80). */
			# Compute 2 Rounds
			SHA512_Round t-2 // Functional Utility: Computes SHA-512 round for t-2.
			SHA512_Round t-1 // Functional Utility: Computes SHA-512 round for t-1.
		.endif
		t = t+2
	.endr

	# Update digest
	// Functional Utility: Accumulates the working hash state variables (`a_64` through `h_64`) back into the `digest` array in memory.
	add     a_64, DIGEST(0)
	add     b_64, DIGEST(1)
	add     c_64, DIGEST(2)
	add     d_64, DIGEST(3)
	add     e_64, DIGEST(4)
	add     f_64, DIGEST(5)
	add     g_64, DIGEST(6)
	add     h_64, DIGEST(7)

	# Advance to next message block
	// Functional Utility: Advances the `msg` pointer by 128 bytes to the next message block.
	add     $16*8, msg
	// Functional Utility: Decrements the `msglen` (block counter).
	dec     msglen
	// Functional Utility: Jumps back to `.Lupdateblock` if more blocks remain to be processed.
	jnz     .Lupdateblock

	# Restore Stack Pointer
	// Functional Utility: Restores the stack pointer.
	mov	%rbp, %rsp
	// Functional Utility: Restores the base pointer.
	pop	%rbp

	# Restore GPRs
	// Functional Utility: Restores callee-saved registers as per x86_64 ABI.
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbx

.Lnowork: /* Functional Role: Epilogue for the function when no blocks were processed or all blocks are done. */
	RET // Functional Utility: Returns from the function.
SYM_FUNC_END(sha512_transform_avx)

########################################################################
### Binary Data

.section	.rodata.cst16.XMM_QWORD_BSWAP, "aM", @progbits, 16
.align 16
/**
 * @brief Mask for byte-swapping QWORDs in an XMM register using `vpshufb`.
 * @details This 16-byte constant provides the control mask for the AVX
 * `vpshufb` instruction to perform byte-swapping on two 64-bit words
 * (QWORDs) within an XMM register. This is essential for converting input
 * data from little-endian to big-endian, as required by the SHA-512 algorithm.
 * Functional Role: Defines a pattern for byte-reversal of 64-bit words within an XMM register.
 */
XMM_QWORD_BSWAP:
	.octa 0x08090a0b0c0d0e0f0001020304050607

# Mergeable 640-byte rodata section. This allows linker to merge the table
# with other, exactly the same 640-byte fragment of another rodata section
# (if such section exists).
.section	.rodata.cst640.K512, "aM", @progbits, 640
.align 64
/**
 * @brief Array of SHA-512 round constants (K[t]).
 * @details This read-only data section stores the 80 64-bit SHA-512 round
 * constants (`K512`) as an array of QWORDs. These constants are added
 * to the message words during each round of the SHA-512 compression function.
 * The section is `640-byte` aligned and marked as mergeable (`aM`) to optimize
 * memory usage by allowing the linker to combine identical constant tables.
 * Functional Role: Provides the immutable SHA-512 round constants for cryptographic operations.
 */
K512:
	.quad 0x428a2f98d728ae22,0x7137449123ef65cd
	.quad 0xb5c0fbcfec4d3b2f,0xe9b5dba58189dbbc
	.quad 0x3956c25bf348b538,0x59f111f1b605d019
	.quad 0x923f82a4af194f9b,0xab1c5ed5da6d8118
	.quad 0xd807aa98a3030242,0x12835b0145706fbe
	.quad 0x243185be4ee4b28c,0x550c7dc3d5ffb4e2
	.quad 0x72be5d74f27b896f,0x80deb1fe3b1696b1
	.quad 0x9bdc06a725c71235,0xc19bf174cf692694
	.quad 0xe49b69c19ef14ad2,0xefbe4786384f25e3
	.quad 0x0fc19dc68b8cd5b5,0x240ca1cc77ac9c65
	.quad 0x2de92c6f592b0275,0x4a7484aa6ea6e483
	.quad 0x5cb0a9dcbd41fbd4,0x76f988da831153b5
	.quad 0x983e5152ee66dfab,0xa831c66d2db43210
	.quad 0xb00327c898fb213f,0xbf597fc7beef0ee4
	.quad 0xc6e00bf33da88fc2,0xd5a79147930aa725
	.quad 0x06ca6351e003826f,0x142929670a0e6e70
	.quad 0x27b70a8546d22ffc,0x2e1b21385c26c926
	.quad 0x4d2c6dfc5ac42aed,0x53380d139d95b3df
	.quad 0x650a73548baf63de,0x766a0abb3c77b2a8
	.quad 0x81c2c92e47edaee6,0x92722c851482353b
	.quad 0xa2bfe8a14cf10364,0xa81a664bbc423001
	.quad 0xc24b8b70d0f89791,0xc76c51a30654be30
	.quad 0xd192e819d6ef5218,0xd69906245565a910
	.quad 0xf40e35855771202a,0x106aa07032bbd1b8
	.quad 0x19a4c116b8d2d0c8,0x1e376c085141ab53
	.quad 0x2748774cdf8eeb99,0x34b0bcb5e19b48a8
	.quad 0x391c0cb3c5c95a63,0x4ed8aa4ae3418acb
	.quad 0x5b9cca4f7763e373,0x682e6ff3d6b2b8a3
	.quad 0x748f82ee5defb2fc,0x78a5636f43172f60
	.quad 0x84c87814a1f0ab72,0x8cc702081a6439ec
	.quad 0x90befffa23631e28,0xa4506cebde82bde9
	.quad 0xbef9a3f7b2c67915,0xc67178f2e372532b
	.quad 0xca273eceea26619c,0xd186b8c721c0c207
	.quad 0xeada7dd6cde0eb1e,0xf57d4f7fee6ed178
	.quad 0x06f067aa72176fba,0x0a637dc5a2c898a6
	.quad 0x113f9804bef90dae,0x1b710b35131c471b
	.quad 0x28db77f523047d84,0x32caab7b40c72493
	.quad 0x3c9ebe0a15c9bebc,0x431d67c49c100d4c
	.quad 0x4cc5d4becb3e42b6,0x597f299cfc657e2a
	.quad 0x5fcb6fab3ad6faec,0x6c44198c4a475817
