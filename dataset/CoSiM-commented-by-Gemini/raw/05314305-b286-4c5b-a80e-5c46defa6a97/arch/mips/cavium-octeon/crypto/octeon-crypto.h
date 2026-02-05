/**
 * @file octeon-crypto.h
 * @brief Low-level hardware interface for the Cavium Octeon cryptographic co-processor (COP2).
 * @details This header file provides a set of low-level definitions and inline
 * assembly macros that map directly to the hardware operations of the Octeon's
 * crypto unit. These macros wrap MIPS `dmtc2` (Doubleword Move to Coprocessor 2)
 * and `dmfc2` (Doubleword Move from Coprocessor 2) instructions, enabling kernel
 * drivers to perform hardware-accelerated cryptographic hashing for algorithms
 * like MD5, SHA1, SHA256, and SHA512.
 */
/*
 * This file is subject to the terms and conditions of the GNU General Public
 * License. See the file "COPYING" in the main directory of this archive
 * for more details.
 *
 * Copyright (C) 2012-2013 Cavium Inc., All Rights Reserved.
 *
 * MD5/SHA1/SHA256/SHA512 instruction definitions added by
 * Aaro Koskinen <aaro.koskinen@iki.fi>.
 *
 */
#ifndef __LINUX_OCTEON_CRYPTO_H
#define __LINUX_OCTEON_CRYPTO_H

#include <linux/sched.h>
#include <asm/mipsregs.h>

#define OCTEON_CR_OPCODE_PRIORITY 300

/**
 * @brief Enables kernel access to the Octeon crypto co-processor (COP2).
 * @see octeon_crypto_enable in octeon-crypto.c for detailed documentation.
 */
extern unsigned long octeon_crypto_enable(struct octeon_cop2_state *state);
/**
 * @brief Disables kernel access to COP2 and restores the previous context.
 * @see octeon_crypto_disable in octeon-crypto.c for detailed documentation.
 */
extern void octeon_crypto_disable(struct octeon_cop2_state *state,
				  unsigned long flags);

/*
 * Macros for MD5/SHA1/SHA256 Hardware Acceleration
 */

/**
 * @brief Writes a 64-bit portion of the hash state (digest) to a COP2 register.
 * @param value The 64-bit value to write.
 * @param index The target register index (0-1 for MD5, 0-2 for SHA1, 0-3 for SHA256).
 */
#define write_octeon_64bit_hash_dword(value, index)	\
do {							\
	__asm__ __volatile__ (				\
	"dmtc2 %[rt],0x0048+" STR(index)		\
	:						\
	: [rt] "d" (cpu_to_be64(value)));		\
} while (0)

/**
 * @brief Reads a 64-bit portion of the hash state from a COP2 register.
 * @param index The source register index.
 * @return The 64-bit value read from the hardware.
 */
#define read_octeon_64bit_hash_dword(index)		\
({							\
	__be64 __value;					\
							\
	__asm__ __volatile__ (				\
	"dmfc2 %[rt],0x0048+" STR(index)		\
	: [rt] "=d" (__value)				\
	: );						\
							\
	be64_to_cpu(__value);				\
})

/**
 * @brief Loads a 64-bit word of the message block into the COP2 unit.
 * @param value The 64-bit message data word.
 * @param index The target block register index (0-6).
 */
#define write_octeon_64bit_block_dword(value, index)	\
do {							\
	__asm__ __volatile__ (				\
	"dmtc2 %[rt],0x0040+" STR(index)		\
	:						\
	: [rt] "d" (cpu_to_be64(value)));		\
} while (0)

/**
 * @brief Writes the final 64-bit word of a message block and triggers the MD5
 *        compression round in hardware.
 * @param value The final 64-bit message data word.
 */
#define octeon_md5_start(value)				\
do {							\
	__asm__ __volatile__ (				\
	"dmtc2 %[rt],0x4047"				\
	:						\
	: [rt] "d" (cpu_to_be64(value)));		\
} while (0)

/**
 * @brief Writes the final 64-bit word of a message block and triggers the SHA1
 *        compression round in hardware.
 * @param value The final 64-bit message data word.
 */
#define octeon_sha1_start(value)			\
do {							\
	__asm__ __volatile__ (				\
	"dmtc2 %[rt],0x4057"				\
	:						\
	: [rt] "d" (value));				\
} while (0)

/**
 * @brief Writes the final 64-bit word of a message block and triggers the SHA256
 *        compression round in hardware.
 * @param value The final 64-bit message data word.
 */
#define octeon_sha256_start(value)			\
do {							\
	__asm__ __volatile__ (				\
	"dmtc2 %[rt],0x404f"				\
	:						\
	: [rt] "d" (value));				\
} while (0)

/*
 * Macros for SHA512 Hardware Acceleration
 */

/**
 * @brief Writes a 64-bit portion of the SHA-512 hash state to a COP2 register.
 * @param value The 64-bit value to write.
 * @param index The target register index (0-7).
 */
#define write_octeon_64bit_hash_sha512(value, index)	\
do {							\
	__asm__ __volatile__ (				\
	"dmtc2 %[rt],0x0250+" STR(index)		\
	:						\
	: [rt] "d" (value));				\
} while (0)

/**
 * @brief Reads a 64-bit portion of the SHA-512 hash state from a COP2 register.
 * @param index The source register index (0-7).
 * @return The 64-bit value read from the hardware.
 */
#define read_octeon_64bit_hash_sha512(index)		\
({							\
	u64 __value;					\
							\
	__asm__ __volatile__ (				\
	"dmfc2 %[rt],0x0250+" STR(index)		\
	: [rt] "=d" (__value)				\
	: );						\
							\
	__value;					\
})

/**
 * @brief Loads a 64-bit word of the 1024-bit SHA-512 message block into the COP2 unit.
 * @param value The 64-bit message data word.
 * @param index The target block register index (0-14).
 */
#define write_octeon_64bit_block_sha512(value, index)	\
do {							\
	__asm__ __volatile__ (				\
	"dmtc2 %[rt],0x0240+" STR(index)		\
	:						\
	: [rt] "d" (value));				\
} while (0)

/**
 * @brief Writes the final 64-bit word of a message block and triggers the SHA512
 *        compression round in hardware.
 * @param value The final 64-bit message data word.
 */
#define octeon_sha512_start(value)			\
do {							\
	__asm__ __volatile__ (				\
	"dmtc2 %[rt],0x424f"				\
	:						\
	: [rt] "d" (value));				\
} while (0)

/*
 * The value is the final block dword (64-bit).
 */
#define octeon_sha1_start(value)			\
do {							\
	__asm__ __volatile__ (				\
	"dmtc2 %[rt],0x4057"				\
	:						\
	: [rt] "d" (value));				\
} while (0)

/*
 * The value is the final block dword (64-bit).
 */
#define octeon_sha256_start(value)			\
do {							\
	__asm__ __volatile__ (				\
	"dmtc2 %[rt],0x404f"				\
	:						\
	: [rt] "d" (value));				\
} while (0)

/*
 * Macros needed to implement SHA512:
 */

/*
 * The index can be 0-7.
 */
#define write_octeon_64bit_hash_sha512(value, index)	\
do {							\
	__asm__ __volatile__ (				\
	"dmtc2 %[rt],0x0250+" STR(index)		\
	:						\
	: [rt] "d" (value));				\
} while (0)

/*
 * The index can be 0-7.
 */
#define read_octeon_64bit_hash_sha512(index)		\
({							\
	u64 __value;					\
							\
	__asm__ __volatile__ (				\
	"dmfc2 %[rt],0x0250+" STR(index)		\
	: [rt] "=d" (__value)				\
	: );						\
							\
	__value;					\
})

/*
 * The index can be 0-14.
 */
#define write_octeon_64bit_block_sha512(value, index)	\
do {							\
	__asm__ __volatile__ (				\
	"dmtc2 %[rt],0x0240+" STR(index)		\
	:						\
	: [rt] "d" (value));				\
} while (0)

/*
 * The value is the final block word (64-bit).
 */
#define octeon_sha512_start(value)			\
do {							\
	__asm__ __volatile__ (				\
	"dmtc2 %[rt],0x424f"				\
	:						\
	: [rt] "d" (value));				\
} while (0)

#endif /* __LINUX_OCTEON_CRYPTO_H */
