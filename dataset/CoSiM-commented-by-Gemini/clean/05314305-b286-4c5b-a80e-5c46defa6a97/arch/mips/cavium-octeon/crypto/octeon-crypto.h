/**
 * @file octeon-crypto.h
 * @brief Low-level hardware interface for the Cavium Octeon cryptographic co-processor (COP2).
 * @details This header file provides a set of low-level definitions and inline
 * assembly macros that map directly to the hardware operations of the Octeon's
 * crypto unit (COP2). These macros wrap MIPS `dmtc2` (Doubleword Move to Coprocessor 2)
 * and `dmfc2` (Doubleword Move from Coprocessor 2) instructions, enabling kernel
 * drivers to perform hardware-accelerated cryptographic hashing for algorithms
 * like MD5, SHA1, SHA256, and SHA512. The interface is designed for direct,
 * high-performance interaction with the dedicated cryptographic hardware, offloading
 * intensive computations from the main CPU.
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

/**
 * @brief Priority constant for Octeon cryptographic operations.
 * @details This constant defines the priority level at which Octeon's
 * cryptographic operations are registered within the Linux kernel's
 * crypto API. A higher value indicates a higher priority, meaning this
 * hardware-accelerated implementation will be preferred over software-only
 * or less optimized versions.
 */
#define OCTEON_CR_OPCODE_PRIORITY 300

/**
 * @brief Enables kernel access to the Octeon crypto co-processor (COP2).
 * @details This function is declared externally and defined in `octeon-crypto.c`.
 * It prepares the hardware for use by a kernel driver, disabling preemption,
 * enabling COP2 access, and saving the previous COP2 context if it was in use.
 * @see octeon_crypto_enable in octeon-crypto.c for detailed documentation.
 */
extern unsigned long octeon_crypto_enable(struct octeon_cop2_state *state);
/**
 * @brief Disables kernel access to COP2 and restores the previous context.
 * @details This function is declared externally and defined in `octeon-crypto.c`.
 * It must be called after `octeon_crypto_enable()` to complete the critical section
 * and restore the hardware to its original state.
 * @see octeon_crypto_disable in octeon-crypto.c for detailed documentation.
 */
extern void octeon_crypto_disable(struct octeon_cop2_state *state,
				  unsigned long flags);

/*
 * Macros for MD5/SHA1/SHA256 Hardware Acceleration
 * Functional Utility: These macros provide a direct interface to the Octeon COP2
 * for cryptographic operations specific to MD5, SHA1, and SHA256, leveraging MIPS
 * `dmtc2` and `dmfc2` instructions for hardware acceleration.
 */

/**
 * @brief Writes a 64-bit portion of the hash state (digest) to a COP2 register.
 * @details This macro uses the MIPS `dmtc2` instruction to move a 64-bit value
 * from a CPU general-purpose register to a specific register within the COP2
 * unit. This is used to load intermediate hash digest values into the hardware
 * accelerator for MD5, SHA1, or SHA256. The value is converted to big-endian
 * format (`cpu_to_be64`) before writing to the hardware.
 *
 * @param value The 64-bit value to write (part of the hash state).
 * @param index The target COP2 register index (0-1 for MD5, 0-2 for SHA1, 0-3 for SHA256).
 * Functional Utility: Loads a 64-bit hash digest word into the Octeon COP2 hardware accelerator.
 */
#define write_octeon_64bit_hash_dword(value, index)	
do {							
	__asm__ __volatile__ (				
	"dmtc2 %[rt],0x0048+" STR(index)		
	:						
	: [rt] "d" (cpu_to_be64(value)));		
} while (0)

/**
 * @brief Reads a 64-bit portion of the hash state from a COP2 register.
 * @details This macro uses the MIPS `dmfc2` instruction to move a 64-bit value
 * from a specific register within the COP2 unit to a CPU general-purpose register.
 * This is used to retrieve intermediate or final hash digest values from the
 * hardware accelerator for MD5, SHA1, or SHA256. The value is converted from
 * big-endian format (`be64_to_cpu`) after reading from the hardware.
 *
 * @param index The source COP2 register index.
 * @return The 64-bit value read from the hardware.
 * Functional Utility: Retrieves a 64-bit hash digest word from the Octeon COP2 hardware accelerator.
 */
#define read_octeon_64bit_hash_dword(index)		
({							
	__be64 __value;					
							
	__asm__ __volatile__ (				
	"dmfc2 %[rt],0x0048+" STR(index)		
	: [rt] "=d" (__value)				
	: );						
							
	be64_to_cpu(__value);				
})

/**
 * @brief Loads a 64-bit word of the message block into the COP2 unit.
 * @details This macro uses the MIPS `dmtc2` instruction to move a 64-bit message
 * data word from a CPU general-purpose register to a specific message block
 * register within the COP2 unit. This is part of feeding the input data to the
 * hardware accelerator for cryptographic processing.
 *
 * @param value The 64-bit message data word.
 * @param index The target COP2 message block register index (0-6).
 * Functional Utility: Loads a 64-bit message block word into the Octeon COP2 hardware accelerator.
 */
#define write_octeon_64bit_block_dword(value, index)	
do {							
	__asm__ __volatile__ (				
	"dmtc2 %[rt],0x0040+" STR(index)		
	:						
	: [rt] "d" (cpu_to_be64(value)));		
} while (0)

/**
 * @brief Writes the final 64-bit word of a message block and triggers the MD5
 *        compression round in hardware.
 * @details This macro uses the MIPS `dmtc2` instruction with a specific COP2
 * register address (`0x4047`) to indicate the final 64-bit message word for
 * an MD5 block and simultaneously initiate the hardware-accelerated MD5
 * compression process. The value is converted to big-endian before writing.
 *
 * @param value The final 64-bit message data word.
 * Functional Utility: Triggers hardware-accelerated MD5 compression for a message block.
 */
#define octeon_md5_start(value)				
do {							
	__asm__ __volatile__ (				
	"dmtc2 %[rt],0x4047"				
	:						
	: [rt] "d" (cpu_to_be64(value)));		
} while (0)

/**
 * @brief Writes the final 64-bit word of a message block and triggers the SHA1
 *        compression round in hardware.
 * @details This macro uses the MIPS `dmtc2` instruction with a specific COP2
 * register address (`0x4057`) to indicate the final 64-bit message word for
 * a SHA1 block and simultaneously initiate the hardware-accelerated SHA1
 * compression process.
 *
 * @param value The final 64-bit message data word.
 * Functional Utility: Triggers hardware-accelerated SHA1 compression for a message block.
 */
#define octeon_sha1_start(value)			
do {							
	__asm__ __volatile__ (				
	"dmtc2 %[rt],0x4057"				
	:						
	: [rt] "d" (value));				
} while (0)

/**
 * @brief Writes the final 64-bit word of a message block and triggers the SHA256
 *        compression round in hardware.
 * @details This macro uses the MIPS `dmtc2` instruction with a specific COP2
 * register address (`0x404f`) to indicate the final 64-bit message word for
 * a SHA256 block and simultaneously initiate the hardware-accelerated SHA256
 * compression process.
 *
 * @param value The final 64-bit message data word.
 * Functional Utility: Triggers hardware-accelerated SHA256 compression for a message block.
 */
#define octeon_sha256_start(value)			
do {							
	__asm__ __volatile__ (				
	"dmtc2 %[rt],0x404f"				
	:						
	: [rt] "d" (value));				
} while (0)

/*
 * Macros for SHA512 Hardware Acceleration
 * Functional Utility: These macros provide a direct interface to the Octeon COP2
 * for cryptographic operations specific to SHA512/384, leveraging MIPS
 * `dmtc2` and `dmfc2` instructions for hardware acceleration.
 */

/**
 * @brief Writes a 64-bit portion of the SHA-512 hash state to a COP2 register.
 * @details This macro uses the MIPS `dmtc2` instruction to move a 64-bit value
 * from a CPU general-purpose register to a specific register within the COP2
 * unit. This is used to load intermediate hash digest values into the hardware
 * accelerator for SHA-512/384.
 *
 * @param value The 64-bit value to write (part of the SHA-512 hash state).
 * @param index The target COP2 register index (0-7).
 * Functional Utility: Loads a 64-bit SHA-512 hash digest word into the Octeon COP2 hardware accelerator.
 */
#define write_octeon_64bit_hash_sha512(value, index)	
do {							
	__asm__ __volatile__ (				
	"dmtc2 %[rt],0x0250+" STR(index)		
	:						
	: [rt] "d" (value));				
} while (0)

/**
 * @brief Reads a 64-bit portion of the SHA-512 hash state from a COP2 register.
 * @details This macro uses the MIPS `dmfc2` instruction to move a 64-bit value
 * from a specific register within the COP2 unit to a CPU general-purpose register.
 * This is used to retrieve intermediate or final hash digest values from the
 * hardware accelerator for SHA-512/384.
 *
 * @param index The source COP2 register index (0-7).
 * @return The 64-bit value read from the hardware.
 * Functional Utility: Retrieves a 64-bit SHA-512 hash digest word from the Octeon COP2 hardware accelerator.
 */
#define read_octeon_64bit_hash_sha512(index)		
({							
	u64 __value;					
							
	__asm__ __volatile__ (				
	"dmfc2 %[rt],0x0250+" STR(index)		
	: [rt] "=d" (__value)				
	: );						
							
	__value;					
})

/**
 * @brief Loads a 64-bit word of the 1024-bit SHA-512 message block into the COP2 unit.
 * @details This macro uses the MIPS `dmtc2` instruction to move a 64-bit message
 * data word from a CPU general-purpose register to a specific message block
 * register within the COP2 unit. This is part of feeding the input data to the
 * hardware accelerator for SHA-512/384 cryptographic processing.
 *
 * @param value The 64-bit message data word.
 * @param index The target COP2 message block register index (0-14).
 * Functional Utility: Loads a 64-bit SHA-512 message block word into the Octeon COP2 hardware accelerator.
 */
#define write_octeon_64bit_block_sha512(value, index)	
do {							
	__asm__ __volatile__ (				
	"dmtc2 %[rt],0x0240+" STR(index)		
	:						
	: [rt] "d" (value));				
} while (0)

/**
 * @brief Writes the final 64-bit word of a message block and triggers the SHA512
 *        compression round in hardware.
 * @details This macro uses the MIPS `dmtc2` instruction with a specific COP2
 * register address (`0x424f`) to indicate the final 64-bit message word for
 * a SHA-512 block and simultaneously initiate the hardware-accelerated SHA-512
 * compression process.
 *
 * @param value The final 64-bit message data word.
 * Functional Utility: Triggers hardware-accelerated SHA-512 compression for a message block.
 */
#define octeon_sha512_start(value)			
do {							
	__asm__ __volatile__ (				
	"dmtc2 %[rt],0x424f"				
	:						
	: [rt] "d" (value));				
} while (0)

#endif /* __LINUX_OCTEON_CRYPTO_H */
