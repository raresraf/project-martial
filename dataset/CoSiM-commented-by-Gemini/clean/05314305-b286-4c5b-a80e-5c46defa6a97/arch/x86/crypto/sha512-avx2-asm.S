########################################################################
# Implement fast SHA-512 with AVX2 instructions. (x86_64)
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
# This code schedules 1 blocks at a time, with 4 lanes per block
########################################################################

/**
 * @file sha512-avx2-asm.S
 * @brief High-performance SHA-512 implementation leveraging Intel AVX2 instructions for x86_64.
 * @details This assembly file provides a highly optimized implementation of the
 * SHA-512 (Secure Hash Algorithm 2, 512-bit variant) compression function,
 * specifically tailored for x86_64 processors equipped with the Intel AVX2
 * (Advanced Vector Extensions 2) instruction set. It extensively utilizes AVX2's
 * 256-bit vector capabilities (YMM registers) to process data in parallel,
 * significantly accelerating the cryptographic process.
 *
 * **Optimization Techniques:**
 * - **AVX2 Vectorization**: Uses AVX2 instructions (`vmovdqu`, `vpshufb`, `vpaddq`,
 *   `vpsrlq`, `vpsllq`, `vpor`, `vpxor`, `vperm2f128`, `vpblendd`) for efficient
 *   loading, byte-swapping, message schedule generation, and hash state updates
 *   across multiple processing lanes.
 * - **Software Pipelining with Lanes**: The implementation processes a single
 *   128-byte block but utilizes four "lanes" of YMM registers (Y_0 to Y_3) to
 *   interleave message schedule generation and round computations.
 * - **Optimized Rotations**: Leverages `rorx` (Rotate with eXtend) instruction,
 *   which is part of the BMI2 (Bit Manipulation Instruction Set 2) and is
 *   highly efficient for fixed-count rotations, crucial for SHA-512's `sigma` functions.
 * - **Message Schedule Acceleration**: Dedicated vector instructions and macros
 *   are optimized for generating SHA-512's `sigma0` and `sigma1` functions.
 * - **Register Rotation**: A custom `RotateState` macro efficiently shuffles
 *   the SHA-512 hash state variables without excessive data movement between GPRs.
 * This implementation is designed to provide maximum SHA-512 hashing performance
 * on AVX2-enabled x86_64 systems within the Linux kernel.
 */

#include <linux/linkage.h>
#include <linux/cfi_types.h>

.text

# Virtual Registers
Y_0 = %ymm4	/* Functional Role: YMM register for message schedule words (lane 0). */
Y_1 = %ymm5	/* Functional Role: YMM register for message schedule words (lane 1). */
Y_2 = %ymm6	/* Functional Role: YMM register for message schedule words (lane 2). */
Y_3 = %ymm7	/* Functional Role: YMM register for message schedule words (lane 3). */

YTMP0 = %ymm0	/* Functional Role: YMM register for temporary vector computations. */
YTMP1 = %ymm1	/* Functional Role: YMM register for temporary vector computations. */
YTMP2 = %ymm2	/* Functional Role: YMM register for temporary vector computations. */
YTMP3 = %ymm3	/* Functional Role: YMM register for temporary vector computations. */
YTMP4 = %ymm8	/* Functional Role: YMM register for temporary vector computations. */
XFER  = YTMP0	/* Functional Role: Alias for YTMP0, typically used for data transfer. */

BYTE_FLIP_MASK  = %ymm9	/* Functional Role: YMM register holding the shuffle control mask for byte-swapping. */

# 1st arg is %rdi, which is saved to the stack and accessed later via %r12
CTX1        = %rdi	/* Functional Role: Pointer to `struct sha512_state` (argument 1, initially in RDI). */
CTX2        = %r12	/* Functional Role: Alias for R12, used to store CTX1 (digest pointer) after it's saved. */
# 2nd arg
INP         = %rsi	/* Functional Role: Pointer to input data buffer (argument 2). */
# 3rd arg
NUM_BLKS    = %rdx	/* Functional Role: Number of 128-byte blocks to process (argument 3). */

c           = %rcx	/* Functional Role: GPR for SHA-512 state variable 'c' (64-bit). */
d           = %r8	/* Functional Role: GPR for SHA-512 state variable 'd' (64-bit). */
e           = %rdx	/* Functional Role: GPR for SHA-512 state variable 'e' (64-bit). (Note: Overlaps with NUM_BLKS, but used after NUM_BLKS is processed). */
y3          = %rsi	/* Functional Role: GPR for temporary computations (64-bit). (Note: Overlaps with INP, but used after INP is processed). */

TBL   = %rdi /* clobbers CTX1 */	/* Functional Role: Alias for RDI, used to hold pointer to SHA-512 round constants table (K512). */

a     = %rax	/* Functional Role: GPR for SHA-512 state variable 'a' (64-bit). */
b     = %rbx	/* Functional Role: GPR for SHA-512 state variable 'b' (64-bit). */

f     = %r9	/* Functional Role: GPR for SHA-512 state variable 'f' (64-bit). */
g     = %r10	/* Functional Role: GPR for SHA-512 state variable 'g' (64-bit). */
h     = %r11	/* Functional Role: GPR for SHA-512 state variable 'h' (64-bit). */
old_h = %r11	/* Functional Role: Alias for R11, used to save 'h' before it's updated. */

T1    = %r12 /* clobbers CTX2 */	/* Functional Role: GPR for temporary computations (64-bit). (Note: Overlaps with CTX2, but used after CTX2 is saved to stack). */
y0    = %r13	/* Functional Role: GPR for temporary computations (64-bit). */
y1    = %r14	/* Functional Role: GPR for temporary computations (64-bit). */
y2    = %r15	/* Functional Role: GPR for temporary computations (64-bit). */

# Local variables (stack frame)
XFER_SIZE = 4*8	/* Functional Role: Size in bytes for transferring 4 QWORDS (2 XMM registers) of data. */
SRND_SIZE = 1*8	/* Functional Role: Size in bytes for storing a round counter (1 QWORD). */
INP_SIZE = 1*8	/* Functional Role: Size in bytes for storing original INP (1 QWORD). */
INPEND_SIZE = 1*8	/* Functional Role: Size in bytes for storing calculated end of input (1 QWORD). */
CTX_SIZE = 1*8	/* Functional Role: Size in bytes for storing original CTX1 (1 QWORD). */

frame_XFER = 0	/* Functional Role: Offset for XFER buffer within the stack frame. */
frame_SRND = frame_XFER + XFER_SIZE	/* Functional Role: Offset for round counter within the stack frame. */
frame_INP = frame_SRND + SRND_SIZE	/* Functional Role: Offset for saved INP within the stack frame. */
frame_INPEND = frame_INP + INP_SIZE	/* Functional Role: Offset for saved INPEND within the stack frame. */
frame_CTX = frame_INPEND + INPEND_SIZE	/* Functional Role: Offset for saved CTX1 within the stack frame. */
frame_size = frame_CTX + CTX_SIZE	/* Functional Role: Total size of the stack frame required. */

## assume buffers not aligned
#define	VMOVDQ vmovdqu /* Functional Role: Alias for `vmovdqu`, indicating usage for unaligned vector moves. */

# addm [mem], reg
# Add reg to mem using reg-mem add and store
/**
 * @brief Adds a register value to a memory location and stores the result back in memory.
 * @details This macro performs an atomic addition where the content of `p2`
 * is added to the 64-bit value at memory address `p1`, and the result is
 * written back to `p1`. It's a convenient shorthand for `add` and `mov`.
 * @param p1 Memory operand (e.g., `8*0(CTX2)`).
 * @param p2 Register operand (e.g., `a`).
 * Functional Utility: Accumulates a register value into a memory location, typically for updating hash digest.
 */
.macro addm p1 p2
	add	\p1, \p2 /* Functional Utility: Adds the value in `p2` to the value at memory address `p1`. */
	mov	\p2, \p1 /* Functional Utility: Moves the result back to memory address `p1`. */
.endm


# COPY_YMM_AND_BSWAP ymm, [mem], byte_flip_mask
# Load ymm with mem and byte swap each dword
/**
 * @brief Loads a YMM register from memory and byte-swaps each 32-bit word.
 * @details This macro efficiently loads 256 bits (8 DWORDS) from a memory
 * location (`p2`) into a YMM register (`p1`) and then performs a byte-swap
 * operation on each 32-bit word within that YMM register using a shuffle mask (`p3`).
 * This is crucial for converting little-endian input data to big-endian,
 * as required by SHA-512.
 * @param p1 The destination YMM register.
 * @param p2 The memory source operand.
 * @param p3 The YMM register holding the byte-flip shuffle mask.
 * Functional Utility: Loads and endian-converts 256 bits of data for SHA-512 message processing.
 */
.macro COPY_YMM_AND_BSWAP p1 p2 p3
	VMOVDQ \p2, \p1 /* Functional Utility: Loads 256 bits from memory `p2` into YMM register `p1`. */
	vpshufb \p3, \p1, \p1 /* Functional Utility: Byte-swaps each 32-bit word in `p1` using `p3` as a mask. */
.endm
# rotate_Ys
# Rotate values of symbols Y0...Y3
/**
 * @brief Logically rotates the YMM register aliases Y_0 through Y_3.
 * @details This macro performs a symbolic rotation of the YMM register aliases
 * that hold message schedule words for different processing lanes. This allows
 * for efficient pipelining of message schedule generation, effectively shifting
 * the "window" of processed words without actual data movement between YMM registers.
 * Functional Utility: Facilitates software pipelining by rotating YMM register aliases.
 */
.macro rotate_Ys
	Y_ = Y_0
	Y_0 = Y_1
	Y_1 = Y_2
	Y_2 = Y_3
	Y_3 = Y_
.endm

# RotateState
/**
 * @brief Logically rotates the SHA-512 state variables (a-h) to the right.
 * @details This macro efficiently shuffles the SHA-512 state variables
 * (`a` through `h`) by reassigning their virtual register names.
 * This implements the step where the output of one round becomes the input
 * for the next, with `h` becoming `g`, `g` becoming `f`, and so on, and
 * the new `a` value coming from a temporary. This is a purely symbolic
 * rotation at assembly time to avoid physical `mov` instructions between these registers.
 * Functional Utility: Efficiently rotates the SHA-512 hash state registers without explicit moves.
 */
.macro RotateState
	# Functional Utility: Saves the current value of 'h' for later use (accumulation).
	old_h  = h
	# Functional Utility: Temporary storage for 'h' before rotation.
	TMP_   = h
	# Functional Utility: Each state variable receives the value of the variable to its right.
	h      = g
	g      = f
	f      = e
	e      = d
	d      = c
	c      = b
	b      = a
	# Functional Utility: The new 'a' receives the value that was originally in 'h' (stored in TMP_).
	a      = TMP_
.endm

# macro MY_VPALIGNR	YDST, YSRC1, YSRC2, RVAL
# YDST = {YSRC1, YSRC2} >> RVAL*8
/**
 * @brief Performs a byte-granular right shift/alignment across concatenated YMM registers.
 * @details This macro is a custom implementation for efficient message schedule
 * word generation. It effectively combines parts of `YSRC1` and `YSRC2` and
 * then performs a byte-granular right shift (`RVAL` * 8 bits) to produce the
 * result in `YDST`. This is crucial for calculating shifted versions of message
 * words, as required by SHA-512's `sigma` functions.
 * @param YDST Destination YMM register.
 * @param YSRC1 Source YMM register 1.
 * @param YSRC2 Source YMM register 2.
 * @param RVAL Number of bytes to shift right.
 * Functional Utility: Efficiently calculates byte-shifted message words for SHA-512 message scheduling.
 */
.macro MY_VPALIGNR YDST YSRC1 YSRC2 RVAL
	# Functional Utility: Permutes elements across YSRC2 and YSRC1 to create an interleaved YMM register in YDST.
	vperm2f128      $0x3, \YSRC2, \YSRC1, \YDST     # YDST = {YS1_LO, YS2_HI}
	# Functional Utility: Performs a byte-granular right shift on YDST, effectively aligning data.
	vpalignr        $\RVAL, \YSRC2, \YDST, \YDST    # YDST = {YDS1, YS2} >> RVAL*8
.endm

/**
 * @brief Computes four rounds of SHA-512, tightly interleaved with message schedule generation.
 * @details This macro describes the complex interplay of scalar (GPR) and vector (YMM)
 * instructions to perform four rounds of SHA-512 hashing for a single processing
 * lane, along with generating new message schedule words. It heavily utilizes
 * AVX2 instructions for parallel data processing and efficient calculation of
 * SHA-512's `Ch`, `Maj`, `Sigma0`, and `Sigma1` functions.
 * Functional Utility: Optimizes SHA-512 core computation by interleaving hash rounds and message schedule generation.
 */
.macro FOUR_ROUNDS_AND_SCHED
################################### RND N + 0 #########################################

	# Extract w[t-7]
	MY_VPALIGNR	YTMP0, Y_3, Y_2, 8		# YTMP0 = W[-7] // Functional Utility: Extracts message word W[t-7] from Y_3/Y_2.
	# Calculate w[t-16] + w[t-7]
	vpaddq		Y_0, YTMP0, YTMP0		# YTMP0 = W[-7] + W[-16] // Functional Utility: Computes W[t-7] + W[t-16].
	# Extract w[t-15]
	MY_VPALIGNR	YTMP1, Y_1, Y_0, 8		# YTMP1 = W[-15] // Functional Utility: Extracts message word W[t-15] from Y_1/Y_0.

	# Calculate sigma0

	# Calculate w[t-15] ror 1
	vpsrlq		$1, YTMP1, YTMP2 // Functional Utility: Logical right shift by 1 bit.
	vpsllq		$(64-1), YTMP1, YTMP3 // Functional Utility: Logical left shift by 63 bits.
	vpor		YTMP2, YTMP3, YTMP3		# YTMP3 = W[-15] ror 1 // Functional Utility: Rotates W[t-15] right by 1 bit.
	# Calculate w[t-15] shr 7
	vpsrlq		$7, YTMP1, YTMP4		# YTMP4 = W[-15] >> 7 // Functional Utility: Logical right shift by 7 bits.

	mov	a, y3		# y3 = a                                # MAJA // Functional Utility: Part of Maj(a,b,c).
	rorx	$41, e, y0	# y0 = e >> 41				# S1A // Functional Utility: Part of S1(e).
	rorx	$18, e, y1	# y1 = e >> 18				# S1B // Functional Utility: Part of S1(e).
	add	frame_XFER(%rsp),h		# h = k + w + h         # -- // Functional Utility: Adds pre-calculated K+W to h.
	or	c, y3		# y3 = a|c                              # MAJA // Functional Utility: Part of Maj(a,b,c).
	mov	f, y2		# y2 = f                                # CH // Functional Utility: Part of CH(e,f,g).
	rorx	$34, a, T1	# T1 = a >> 34				# S0B // Functional Utility: Part of S0(a).

	xor	y1, y0		# y0 = (e>>41) ^ (e>>18)		# S1 // Functional Utility: Part of S1(e).
	xor	g, y2		# y2 = f^g                              # CH // Functional Utility: Part of CH(e,f,g).
	rorx	$14, e, y1	# y1 = (e >> 14)			# S1 // Functional Utility: Part of S1(e).

	and	e, y2		# y2 = (f^g)&e                          # CH // Functional Utility: Part of CH(e,f,g).
	xor	y1, y0		# y0 = (e>>41) ^ (e>>18) ^ (e>>14)	# S1 // Functional Utility: Completes S1(e).
	rorx	$39, a, y1	# y1 = a >> 39				# S0A // Functional Utility: Part of S0(a).
	add	h, d		# d = k + w + h + d                     # -- // Functional Utility: Adds h to d.

	and	b, y3		# y3 = (a|c)&b                          # MAJA // Functional Utility: Part of Maj(a,b,c).
	xor	T1, y1		# y1 = (a>>39) ^ (a>>34)		# S0 // Functional Utility: Part of S0(a).
	rorx	$28, a, T1	# T1 = (a >> 28)			# S0 // Functional Utility: Part of S0(a).

	xor	g, y2		# y2 = CH = ((f^g)&e)^g                 # CH // Functional Utility: Completes CH(e,f,g).
	xor	T1, y1		# y1 = (a>>39) ^ (a>>34) ^ (a>>28)	# S0 // Functional Utility: Completes S0(a).
	mov	a, T1		# T1 = a                                # MAJB // Functional Utility: Part of Maj(a,b,c).
	and	c, T1		# T1 = a&c                              # MAJB // Functional Utility: Part of Maj(a,b,c).

	add	y0, y2		# y2 = S1 + CH                          # -- // Functional Utility: Adds S1(e) to CH(e,f,g).
	or	T1, y3		# y3 = MAJ = (a|c)&b)|(a&c)             # MAJ // Functional Utility: Completes Maj(a,b,c).
	add	y1, h		# h = k + w + h + S0                    # -- // Functional Utility: Adds S0(a) to h.

	add	y2, d		# d = k + w + h + d + S1 + CH = d + t1  # -- // Functional Utility: Updates d.

	add	y2, h		# h = k + w + h + S0 + S1 + CH = t1 + S0# -- // Functional Utility: Adds S1+CH to h.
	add	y3, h		# h = t1 + S0 + MAJ                     # -- // Functional Utility: Adds MAJ to h.

	RotateState // Functional Utility: Rotates the SHA-512 state variables.

################################### RND N + 1 #########################################

	# Calculate w[t-15] ror 8
	vpsrlq		$8, YTMP1, YTMP2 // Functional Utility: Logical right shift by 8 bits.
	vpsllq		$(64-8), YTMP1, YTMP1 // Functional Utility: Logical left shift by 56 bits.
	vpor		YTMP2, YTMP1, YTMP1		# YTMP1 = W[-15] ror 8 // Functional Utility: Rotates W[t-15] right by 8 bits.
	# XOR the three components
	vpxor		YTMP4, YTMP3, YTMP3		# YTMP3 = W[-15] ror 1 ^ W[-15] >> 7 // Functional Utility: Accumulates components for `s0(W[t-15])`.
	vpxor		YTMP1, YTMP3, YTMP1		# YTMP1 = s0 // Functional Utility: Completes `s0(W[t-15])`.

	# Add three components, w[t-16], w[t-7] and sigma0
	vpaddq		YTMP1, YTMP0, YTMP0		# YTMP0 = W[-16] + W[-7] + s0 // Functional Utility: Computes W[t-16] + W[t-7] + s0(W[t-15]).
	# Move to appropriate lanes for calculating w[16] and w[17]
	vperm2f128	$0x0, YTMP0, YTMP0, Y_0		# Y_0 = W[-16] + W[-7] + s0 {BABA} // Functional Utility: Transfers data to Y_0 (next message schedule words).
	# Move to appropriate lanes for calculating w[18] and w[19]
	vpand		MASK_YMM_LO(%rip), YTMP0, YTMP0	# YTMP0 = W[-16] + W[-7] + s0 {DC00} // Functional Utility: Masks lower 128-bit lane for further processing.

	# Calculate w[16] and w[17] in both 128 bit lanes

	# Calculate sigma1 for w[16] and w[17] on both 128 bit lanes
	vperm2f128	$0x11, Y_3, Y_3, YTMP2		# YTMP2 = W[-2] {BABA} // Functional Utility: Permutes Y_3 to get W[t-2] for both lanes.
	vpsrlq		$6, YTMP2, YTMP4		# YTMP4 = W[-2] >> 6 {BABA} // Functional Utility: Calculates part of `s1(W[t-2])`.

	mov	a, y3		# y3 = a                                # MAJA
	rorx	$41, e, y0	# y0 = e >> 41				# S1A
	rorx	$18, e, y1	# y1 = e >> 18				# S1B
	add	1*8+frame_XFER(%rsp), h		# h = k + w + h         # --
	or	c, y3		# y3 = a|c                              # MAJA


	mov	f, y2		# y2 = f                                # CH
	rorx	$34, a, T1	# T1 = a >> 34				# S0B
	xor	y1, y0		# y0 = (e>>41) ^ (e>>18)		# S1
	xor	g, y2		# y2 = f^g                              # CH


	rorx	$14, e, y1	# y1 = (e >> 14)			# S1
	xor	y1, y0		# y0 = (e>>41) ^ (e>>18) ^ (e>>14)	# S1
	rorx	$39, a, y1	# y1 = a >> 39				# S0A
	and	e, y2		# y2 = (f^g)&e                          # CH
	add	h, d		# d = k + w + h + d                     # --

	and	b, y3		# y3 = (a|c)&b                          # MAJA
	xor	T1, y1		# y1 = (a>>39) ^ (a>>34)		# S0

	rorx	$28, a, T1	# T1 = (a >> 28)			# S0
	xor	g, y2		# y2 = CH = ((f^g)&e)^g                 # CH

	xor	T1, y1		# y1 = (a>>39) ^ (a>>34) ^ (a>>28)	# S0
	mov	a, T1		# T1 = a                                # MAJB
	and	c, T1		# T1 = a&c                              # MAJB
	add	y0, y2		# y2 = S1 + CH                          # --

	or	T1, y3		# y3 = MAJ = (a|c)&b)|(a&c)             # MAJ
	add	y1, h		# h = k + w + h + S0                    # --

	add	y2, d		# d = k + w + h + d + S1 + CH = d + t1  # --

	RotateState


################################### RND N + 2 #########################################

	add	y2, old_h	# h = k + w + h + S0 + S1 + CH = t1 + S0# --
	mov	f, y2		# y2 = f                                # CH
	rorx	$41, e, y0	# y0 = e >> 41				# S1A
	rorx	$18, e, y1	# y1 = e >> 18				# S1B
	xor	g, y2		# y2 = f^g                              # CH

	xor	y1, y0		# y0 = (e>>41) ^ (e>>18)		# S1
	rorx	$14, e, y1	# y1 = (e >> 14)			# S1
	and	e, y2		# y2 = (f^g)&e                          # CH
	add	y3, old_h	# h = t1 + S0 + MAJ                     # --

	xor	y1, y0		# y0 = (e>>41) ^ (e>>18) ^ (e>>14)	# S1
	rorx	$34, a, T1	# T1 = a >> 34				# S0B
	xor	g, y2		# y2 = CH = ((f^g)&e)^g                 # CH
	rorx	$39, a, y1	# y1 = a >> 39				# S0A
	mov	a, y3		# y3 = a                                # MAJA

	xor	T1, y1		# y1 = (a>>39) ^ (a>>34)		# S0
	rorx	$28, a, T1	# T1 = (a >> 28)			# S0
	add	8*2+frame_XFER(%rsp), h		# h = k + w + h         # --
	or	c, y3		# y3 = a|c                              # MAJA

	xor	T1, y1		# y1 = (a>>39) ^ (a>>34) ^ (a>>28)	# S0
	mov	a, T1		# T1 = a                                # MAJB
	and	b, y3		# y3 = (a|c)&b                          # MAJA
	and	c, T1		# T1 = a&c                              # MAJB
	add	y0, y2		# y2 = S1 + CH                          # --

	add	h, d		# d = k + w + h + d                     # --
	or	T1, y3		# y3 = MAJ = (a|c)&b)|(a&c)             # MAJ
	add	y1, h		# h = k + w + h + S0                    # --

	add	y2, d		# d = k + w + h + d + S1 + CH = d + t1  # --

	RotateState

################################### RND N + 3 #########################################

	add	y2, old_h	# h = k + w + h + S0 + S1 + CH = t1 + S0# --
	mov	f, y2		# y2 = f                                # CH
	rorx	$41, e, y0	# y0 = e >> 41				# S1A
	rorx	$18, e, y1	# y1 = e >> 18				# S1B
	xor	g, y2		# y2 = f^g                              # CH

	xor	y1, y0		# y0 = (e>>41) ^ (e>>18)		# S1
	rorx	$14, e, y1	# y1 = (e >> 14)			# S1
	and	e, y2		# y2 = (f^g)&e                          # CH
	add	y3, old_h	# h = t1 + S0 + MAJ                     # --

	xor	y1, y0		# y0 = (e>>41) ^ (e>>18) ^ (e>>14)	# S1
	rorx	$34, a, T1	# T1 = a >> 34				# S0B
	xor	g, y2		# y2 = CH = ((f^g)&e)^g                 # CH
	rorx	$39, a, y1	# y1 = a >> 39				# S0A
	mov	a, y3		# y3 = a                                # MAJA

	xor	T1, y1		# y1 = (a>>39) ^ (a>>34)		# S0
	rorx	$28, a, T1	# T1 = (a >> 28)			# S0
	add	3*8+frame_XFER(%rsp), h		# h = k + w + h         # --
	or	c, y3		# y3 = a|c                              # MAJA

	xor	T1, y1		# y1 = (a>>39) ^ (a>>34) ^ (a>>28)	# S0
	mov	a, T1		# T1 = a                                # MAJB
	and	b, y3		# y3 = (a|c)&b                          # MAJA
	and	c, T1		# T1 = a&c                              # MAJB
	add	y0, y2		# y2 = S1 + CH                          # --


	add	h, d		# d = k + w + h + d                     # --
	or	T1, y3		# y3 = MAJ = (a|c)&b)|(a&c)             # MAJ
	add	y1, h		# h = k + w + h + S0                    # --

	add	y2, d		# d = k + w + h + d + S1 + CH = d + t1  # --

	add	y2, h		# h = k + w + h + S0 + S1 + CH = t1 + S0# --

	add	y3, h		# h = t1 + S0 + MAJ                     # --

	RotateState

	rotate_Ys // Functional Utility: Rotates the YMM message schedule register aliases for the next block.
.endm

/**
 * @brief Computes four rounds of SHA-512 without message schedule generation.
 * @details This macro implements the computation for four rounds of SHA-512
 * hash algorithm. Unlike `FOUR_ROUNDS_AND_SCHED`, this macro assumes the message
 * schedule words are already prepared in the YMM registers and focuses solely
 * on updating the scalar SHA-512 state variables (a-h) based on the provided
 * (K+W) values from the stack.
 * Functional Utility: Executes four SHA-512 rounds, updating hash state variables using pre-generated message words.
 */
.macro DO_4ROUNDS

################################### RND N + 0 #########################################

	mov	f, y2		# y2 = f                                # CH
	rorx	$41, e, y0	# y0 = e >> 41				# S1A
	rorx	$18, e, y1	# y1 = e >> 18				# S1B
	xor	g, y2		# y2 = f^g                              # CH

	xor	y1, y0		# y0 = (e>>41) ^ (e>>18)		# S1
	rorx	$14, e, y1	# y1 = (e >> 14)			# S1
	and	e, y2		# y2 = (f^g)&e                          # CH

	xor	y1, y0		# y0 = (e>>41) ^ (e>>18) ^ (e>>14)	# S1
	rorx	$34, a, T1	# T1 = a >> 34				# S0B
	xor	g, y2		# y2 = CH = ((f^g)&e)^g                 # CH
	rorx	$39, a, y1	# y1 = a >> 39				# S0A
	mov	a, y3		# y3 = a                                # MAJA

	xor	T1, y1		# y1 = (a>>39) ^ (a>>34)		# S0
	rorx	$28, a, T1	# T1 = (a >> 28)			# S0
	add	frame_XFER(%rsp), h		# h = k + w + h         # --
	or	c, y3		# y3 = a|c                              # MAJA

	xor	T1, y1		# y1 = (a>>39) ^ (a>>34) ^ (a>>28)	# S0
	mov	a, T1		# T1 = a                                # MAJB
	and	b, y3		# y3 = (a|c)&b                          # MAJA
	and	c, T1		# T1 = a&c                              # MAJB
	add	y0, y2		# y2 = S1 + CH                          # --

	add	h, d		# d = k + w + h + d                     # --
	or	T1, y3		# y3 = MAJ = (a|c)&b)|(a&c)             # MAJ
	add	y1, h		# h = k + w + h + S0                    # --

	add	y2, d		# d = k + w + h + d + S1 + CH = d + t1  # --

	RotateState

################################### RND N + 1 #########################################

	add	y2, old_h	# h = k + w + h + S0 + S1 + CH = t1 + S0# --
	mov	f, y2		# y2 = f                                # CH
	rorx	$41, e, y0	# y0 = e >> 41				# S1A
	rorx	$18, e, y1	# y1 = e >> 18				# S1B
	xor	g, y2		# y2 = f^g                              # CH

	xor	y1, y0		# y0 = (e>>41) ^ (e>>18)		# S1
	rorx	$14, e, y1	# y1 = (e >> 14)			# S1
	and	e, y2		# y2 = (f^g)&e                          # CH
	add	y3, old_h	# h = t1 + S0 + MAJ                     # --

	xor	y1, y0		# y0 = (e>>41) ^ (e>>18) ^ (e>>14)	# S1
	rorx	$34, a, T1	# T1 = a >> 34				# S0B
	xor	g, y2		# y2 = CH = ((f^g)&e)^g                 # CH
	rorx	$39, a, y1	# y1 = a >> 39				# S0A
	mov	a, y3		# y3 = a                                # MAJA

	xor	T1, y1		# y1 = (a>>39) ^ (a>>34)		# S0
	rorx	$28, a, T1	# T1 = (a >> 28]			# S0
	add	8*1+frame_XFER(%rsp), h		# h = k + w + h         # --
	or	c, y3		# y3 = a|c                              # MAJA

	xor	T1, y1		# y1 = (a>>39) ^ (a>>34) ^ (a>>28)	# S0
	mov	a, T1		# T1 = a                                # MAJB
	and	b, y3		# y3 = (a|c)&b                          # MAJA
	and	c, T1		# T1 = a&c                              # MAJB
	add	y0, y2		# y2 = S1 + CH                          # --

	add	h, d		# d = k + w + h + d                     # --
	or	T1, y3		# y3 = MAJ = (a|c)&b)|(a&c)             # MAJ
	add	y1, h		# h = k + w + h + S0                    # --

	add	y2, d		# d = k + w + h + d + S1 + CH = d + t1  # --

	RotateState

################################### RND N + 2 #########################################

	add	y2, old_h	# h = k + w + h + S0 + S1 + CH = t1 + S0# --
	mov	f, y2		# y2 = f                                # CH
	rorx	$41, e, y0	# y0 = e >> 41				# S1A
	rorx	$18, e, y1	# y1 = e >> 18				# S1B
	xor	g, y2		# y2 = f^g                              # CH

	xor	y1, y0		# y0 = (e>>41) ^ (e>>18)		# S1
	rorx	$14, e, y1	# y1 = (e >> 14)			# S1
	and	e, y2		# y2 = (f^g)&e                          # CH
	add	y3, old_h	# h = t1 + S0 + MAJ                     # --

	xor	y1, y0		# y0 = (e>>41) ^ (e>>18) ^ (e>>14)	# S1
	rorx	$34, a, T1	# T1 = a >> 34				# S0B
	xor	g, y2		# y2 = CH = ((f^g)&e)^g                 # CH
	rorx	$39, a, y1	# y1 = a >> 39				# S0A
	mov	a, y3		# y3 = a                                # MAJA

	xor	T1, y1		# y1 = (a>>39) ^ (a>>34)		# S0
	rorx	$28, a, T1	# T1 = (a >> 28)			# S0
	add	8*2+frame_XFER(%rsp), h		# h = k + w + h         # --
	or	c, y3		# y3 = a|c                              # MAJA

	xor	T1, y1		# y1 = (a>>39) ^ (a>>34) ^ (a>>28)	# S0
	mov	a, T1		# T1 = a                                # MAJB
	and	b, y3		# y3 = (a|c)&b                          # MAJA
	and	c, T1		# T1 = a&c                              # MAJB
	add	y0, y2		# y2 = S1 + CH                          # --

	add	h, d		# d = k + w + h + d                     # --
	or	T1, y3		# y3 = MAJ = (a|c)&b)|(a&c)             # MAJ
	add	y1, h		# h = k + w + h + S0                    # --

	add	y2, d		# d = k + w + h + d + S1 + CH = d + t1  # --

	RotateState

################################### RND N + 3 #########################################

	add	y2, old_h	# h = k + w + h + S0 + S1 + CH = t1 + S0# --
	mov	f, y2		# y2 = f                                # CH
	rorx	$41, e, y0	# y0 = e >> 41				# S1A
	rorx	$18, e, y1	# y1 = e >> 18				# S1B
	xor	g, y2		# y2 = f^g                              # CH

	xor	y1, y0		# y0 = (e>>41) ^ (e>>18)		# S1
	rorx	$14, e, y1	# y1 = (e >> 14)			# S1
	and	e, y2		# y2 = (f^g)&e                          # CH
	add	y3, old_h	# h = t1 + S0 + MAJ                     # --

	xor	y1, y0		# y0 = (e>>41) ^ (e>>18) ^ (e>>14)	# S1
	rorx	$34, a, T1	# T1 = a >> 34				# S0B
	xor	g, y2		# y2 = CH = ((f^g)&e)^g                 # CH
	rorx	$39, a, y1	# y1 = a >> 39				# S0A
	mov	a, y3		# y3 = a                                # MAJA

	xor	T1, y1		# y1 = (a>>39) ^ (a>>34)		# S0
	rorx	$28, a, T1	# T1 = (a >> 28)			# S0
	add	8*3+frame_XFER(%rsp), h		# h = k + w + h         # --
	or	c, y3		# y3 = a|c                              # MAJA

	xor	T1, y1		# y1 = (a>>39) ^ (a>>34) ^ (a>>28)	# S0
	mov	a, T1		# T1 = a                                # MAJB
	and	b, y3		# y3 = (a|c)&b                          # MAJA
	and	c, T1		# T1 = a&c                              # MAJB
	add	y0, y2		# y2 = S1 + CH                          # --


	add	h, d		# d = k + w + h + d                     # --
	or	T1, y3		# y3 = MAJ = (a|c)&b)|(a&c)             # MAJ
	add	y1, h		# h = k + w + h + S0                    # --

	add	y2, d		# d = k + w + h + d + S1 + CH = d + t1  # --

	add	y2, h		# h = k + w + h + S0 + S1 + CH = t1 + S0# --

	add	y3, h		# h = t1 + S0 + MAJ                     # --

	RotateState

	rotate_Ys // Functional Utility: Rotates the YMM message schedule register aliases for the next block.
.endm

/**
 * @brief SHA-512 transform function optimized with Intel AVX2 and BMI2 instructions.
 * @details This function updates the SHA-512 digest using AVX2 and BMI2 (`rorx`)
 * instructions to process multiple 128-byte input data blocks. It employs
 * software pipelining where message schedule generation and hash round
 * computations are tightly interleaved. The code is structured to process
 * 4 "lanes" concurrently within a block, maximizing parallel execution.
 *
 * @param CTX1 (rdi): Pointer to the `sha512_state` context structure (u64 state[8]).
 * @param INP (rsi): Pointer to the input data blocks (128-byte aligned).
 * @param NUM_BLKS (rdx): Number of 128-byte blocks to process.
 * Functional Utility: Accelerates SHA-512 compression using AVX2 and BMI2 instructions for multiple data blocks.
 */
SYM_TYPED_FUNC_START(sha512_transform_rorx)
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
	// Functional Utility: Allocates stack space for local variables (XFER, SRND, INP, INPEND, CTX).
	sub	$frame_size, %rsp
	// Functional Utility: Aligns the stack pointer to a 32-byte boundary, crucial for AVX2 operations.
	and	$~(0x20 - 1), %rsp

	// Functional Utility: Converts the number of blocks (`NUM_BLKS`) from count to total bytes (`NUM_BLKS` * 128).
	shl	$7, NUM_BLKS
	// Functional Utility: Checks if `NUM_BLKS` is zero.
	jz	.Ldone_hash
	// Functional Utility: Calculates the pointer to the end of the input data buffer.
	add	INP, NUM_BLKS
	mov	NUM_BLKS, frame_INPEND(%rsp) // Functional Utility: Stores the end-of-data pointer on the stack.

	## load initial digest
	// Functional Utility: Loads the initial SHA-512 state variables (H0-H7) from `CTX1` into GPRs `a` through `h`.
	mov	8*0(CTX1), a
	mov	8*1(CTX1), b
	mov	8*2(CTX1), c
	mov	8*3(CTX1), d
	mov	8*4(CTX1), e
	mov	8*5(CTX1), f
	mov	8*6(CTX1), g
	mov	8*7(CTX1), h

	# save %rdi (CTX) before it gets clobbered
	mov	%rdi, frame_CTX(%rsp) // Functional Utility: Saves the original `CTX1` (digest pointer) to stack.

	// Functional Utility: Loads the byte-flip shuffle mask into `BYTE_FLIP_MASK` (YMM9) for endian conversion.
	vmovdqa	PSHUFFLE_BYTE_FLIP_MASK(%rip), BYTE_FLIP_MASK

.Lloop0: /* Functional Role: Main loop for processing each 128-byte data block. */
	lea	K512(%rip), TBL // Functional Utility: Loads the base address of the SHA-512 round constants table into `TBL`.

	## byte swap first 16 dwords
	// Functional Utility: Loads 32 bytes (4 QWORDS) from `INP`, byte-swaps them, and stores in `Y_0`.
	COPY_YMM_AND_BSWAP	Y_0, (INP), BYTE_FLIP_MASK
	// Functional Utility: Loads 32 bytes from `INP + 32`, byte-swaps, and stores in `Y_1`.
	COPY_YMM_AND_BSWAP	Y_1, 1*32(INP), BYTE_FLIP_MASK
	// Functional Utility: Loads 32 bytes from `INP + 64`, byte-swaps, and stores in `Y_2`.
	COPY_YMM_AND_BSWAP	Y_2, 2*32(INP), BYTE_FLIP_MASK
	// Functional Utility: Loads 32 bytes from `INP + 96`, byte-swaps, and stores in `Y_3`.
	COPY_YMM_AND_BSWAP	Y_3, 3*32(INP), BYTE_FLIP_MASK

	mov	INP, frame_INP(%rsp) // Functional Utility: Saves the current input pointer (`INP`) to the stack.

	## schedule 64 input dwords, by doing 12 rounds of 4 each
	movq	$4, frame_SRND(%rsp) // Functional Utility: Initializes round counter (4 iterations of 4 rounds = 16 rounds).

.align 16
.Lloop1: /* Functional Role: Loop for initial 64 rounds, tightly coupled with message schedule generation. */
	// Functional Utility: Adds 32 bytes of K constants from `TBL` to `Y_0` and stores in `XFER`.
	vpaddq	(TBL), Y_0, XFER
	// Functional Utility: Stores `XFER` (K+W) to a stack-based buffer (`frame_XFER`) for current rounds.
	vmovdqa XFER, frame_XFER(%rsp)
	// Functional Utility: Executes four SHA-512 rounds, interleaving with message schedule generation.
	FOUR_ROUNDS_AND_SCHED

	// Functional Utility: Adds K constants from `TBL + 32` to `Y_0` and stores in `XFER`.
	vpaddq	1*32(TBL), Y_0, XFER
	// Functional Utility: Stores `XFER` (K+W) to stack.
	vmovdqa XFER, frame_XFER(%rsp)
	// Functional Utility: Executes four SHA-512 rounds.
	FOUR_ROUNDS_AND_SCHED

	// Functional Utility: Adds K constants from `TBL + 64` to `Y_0` and stores in `XFER`.
	vpaddq	2*32(TBL), Y_0, XFER
	// Functional Utility: Stores `XFER` (K+W) to stack.
	vmovdqa XFER, frame_XFER(%rsp)
	// Functional Utility: Executes four SHA-512 rounds.
	FOUR_ROUNDS_AND_SCHED

	// Functional Utility: Adds K constants from `TBL + 96` to `Y_0` and stores in `XFER`.
	vpaddq	3*32(TBL), Y_0, XFER
	// Functional Utility: Stores `XFER` (K+W) to stack.
	vmovdqa XFER, frame_XFER(%rsp)
	add	$(4*32), TBL // Functional Utility: Advances the K constants table pointer.
	// Functional Utility: Executes four SHA-512 rounds.
	FOUR_ROUNDS_AND_SCHED

	subq	$1, frame_SRND(%rsp) // Functional Utility: Decrements the round counter.
	// Functional Utility: Jumps back to `.Lloop1` if more rounds are needed (for the initial 64 rounds).
	jne	.Lloop1

	movq	$2, frame_SRND(%rsp) // Functional Utility: Initializes round counter for the final 16 rounds.
.Lloop2: /* Functional Role: Loop for final 16 rounds, using pre-generated message schedule. */
	// Functional Utility: Adds K constants from `TBL` to `Y_0` and stores in `XFER`.
	vpaddq	(TBL), Y_0, XFER
	// Functional Utility: Stores `XFER` (K+W) to stack.
	vmovdqa XFER, frame_XFER(%rsp)
	// Functional Utility: Executes four SHA-512 rounds using pre-calculated message words.
	DO_4ROUNDS
	// Functional Utility: Adds K constants from `TBL + 32` to `Y_1` and stores in `XFER`.
	vpaddq	1*32(TBL), Y_1, XFER
	// Functional Utility: Stores `XFER` (K+W) to stack.
	vmovdqa XFER, frame_XFER(%rsp)
	add	$(2*32), TBL // Functional Utility: Advances the K constants table pointer.
	// Functional Utility: Executes four SHA-512 rounds using pre-calculated message words.
	DO_4ROUNDS

	// Functional Utility: Rotates YMM registers for message schedule (`Y_0 = Y_2`, `Y_1 = Y_3`).
	vmovdqa	Y_2, Y_0
	vmovdqa	Y_3, Y_1

	subq	$1, frame_SRND(%rsp) // Functional Utility: Decrements the round counter.
	// Functional Utility: Jumps back to `.Lloop2` if more rounds are needed (for the final 16 rounds).
	jne	.Lloop2

	mov	frame_CTX(%rsp), CTX2 // Functional Utility: Retrieves the original digest pointer from the stack.
	// Functional Utility: Accumulates the final hash state from GPRs `a` through `h` into the `digest` array in memory.
	addm	8*0(CTX2), a
	addm	8*1(CTX2), b
	addm	8*2(CTX2), c
	addm	8*3(CTX2), d
	addm	8*4(CTX2), e
	addm	8*5(CTX2), f
	addm	8*6(CTX2), g
	addm	8*7(CTX2), h

	mov	frame_INP(%rsp), INP // Functional Utility: Restores the original input pointer (`INP`) from the stack.
	add	$128, INP // Functional Utility: Advances the input pointer by 128 bytes to the next block.
	cmp	frame_INPEND(%rsp), INP // Functional Utility: Compares the current input pointer with the end-of-data pointer.
	// Functional Utility: Jumps back to `.Lloop0` if more blocks remain to be processed.
	jne	.Lloop0

.Ldone_hash: /* Functional Role: Epilogue for the function when all blocks are processed. */

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

	vzeroupper // Functional Utility: Zeros the upper 128-bits of all YMM registers to avoid AVX-SSE transition penalties.
	RET // Functional Utility: Returns from the function.
SYM_FUNC_END(sha512_transform_rorx)

########################################################################
### Binary Data


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
	.quad	0x428a2f98d728ae22,0x7137449123ef65cd
	.quad	0xb5c0fbcfec4d3b2f,0xe9b5dba58189dbbc
	.quad	0x3956c25bf348b538,0x59f111f1b605d019
	.quad	0x923f82a4af194f9b,0xab1c5ed5da6d8118
	.quad	0xd807aa98a3030242,0x12835b0145706fbe
	.quad	0x243185be4ee4b28c,0x550c7dc3d5ffb4e2
	.quad	0x72be5d74f27b896f,0x80deb1fe3b1696b1
	.quad	0x9bdc06a725c71235,0xc19bf174cf692694
	.quad	0xe49b69c19ef14ad2,0xefbe4786384f25e3
	.quad	0x0fc19dc68b8cd5b5,0x240ca1cc77ac9c65
	.quad	0x2de92c6f592b0275,0x4a7484aa6ea6e483
	.quad	0x5cb0a9dcbd41fbd4,0x76f988da831153b5
	.quad	0x983e5152ee66dfab,0xa831c66d2db43210
	.quad	0xb00327c898fb213f,0xbf597fc7beef0ee4
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

.section	.rodata.cst32.PSHUFFLE_BYTE_FLIP_MASK, "aM", @progbits, 32
.align 32
/**
 * @brief Shuffle mask for byte-swapping QWORDs in a YMM register.
 * @details This 32-byte constant provides the control mask for the AVX2
 * `vpshufb` instruction to perform byte-swapping on four 64-bit words
 * (QWORDs) within a YMM register. This is essential for converting input
 * data from little-endian to big-endian, as required by the SHA-512 algorithm.
 * Functional Role: Defines a pattern for byte-reversal of 64-bit words within a YMM register.
 */
PSHUFFLE_BYTE_FLIP_MASK:
	.octa 0x08090a0b0c0d0e0f0001020304050607 // Functional Role: Mask for the lower 128-bit lane.
	.octa 0x18191a1b1c1d1e1f1011121314151617 // Functional Role: Mask for the upper 128-bit lane.

.section	.rodata.cst32.MASK_YMM_LO, "aM", @progbits, 32
.align 32
/**
 * @brief Mask to isolate the lower 128-bit lane of a YMM register.
 * @details This 32-byte constant contains a pattern where the lower 16 bytes
 * are all zeros (`0x00...00`) and the upper 16 bytes are all ones (`0xFF...FF`).
 * When used with bitwise AND operations (e.g., `vpand`), it effectively
 * masks off the lower 128-bit lane of a YMM register.
 * Functional Role: Provides a mask to selectively operate on the upper 128-bit lane of a YMM register.
 */
MASK_YMM_LO:
	.octa 0x00000000000000000000000000000000 // Functional Role: Zeroes out the lower 128-bit lane.
	.octa 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF // Functional Role: Keeps the upper 128-bit lane.
