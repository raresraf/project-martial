/**
 * @file octeon-crypto.c
 * @brief Core context-switching logic for Cavium Octeon's crypto co-processor (COP2).
 * @details This file implements the essential functions (`octeon_crypto_enable`
 * and `octeon_crypto_disable`) required to manage safe, shared, and exclusive
 * access to the Octeon's hardware cryptographic accelerator (COP2) by kernel-level
 * drivers. Its primary purpose is to allow kernel drivers to utilize the COP2
 * without corrupting the cryptographic context of a user-space process (or another
 * kernel context) that might also be using it. This is achieved through a meticulous
 * saving and restoring of the COP2 context, analogous to how the FPU context
 * is managed in multitasking environments. These functions form a critical
 * pair that must encapsulate any kernel code interacting with the COP2 hardware.
 */
/*
 * This file is subject to the terms and conditions of the GNU General Public
 * License. See the file "COPYING" in the main directory of this archive
 * for more details.
 *
 * Copyright (C) 2004-2012 Cavium Networks
 */

#include <asm/cop2.h>
#include <linux/export.h>
#include <linux/interrupt.h>
#include <linux/sched/task_stack.h>

#include "octeon-crypto.h"

/**
 * @brief Enables kernel access to the Octeon crypto co-processor (COP2) and saves previous context.
 * @details This function meticulously prepares the Octeon's COP2 hardware for
 * exclusive use by a kernel driver. It executes a critical sequence of operations
 * to ensure context integrity:
 * 1. Disables preemption and saves interrupt flags to ensure atomicity.
 * 2. Grants the current CPU core access to the COP2 unit by setting the `ST0_CU2` bit
 *    in the MIPS `CP0 Status` register.
 * 3. Conditionally saves the *previous* COP2 state:
 *    - If the current task was already using COP2 (e.g., a user-space process),
 *      its COP2 state is saved into the task structure's `thread.cp2` field.
 *    - If COP2 was active for another reason (e.g., another kernel context),
 *      its state is saved to the caller-provided `state` parameter on the stack.
 * This function forms the opening part of a critical section for COP2 access
 * and must be paired with `octeon_crypto_disable()`.
 *
 * @param[out] state Pointer to a `struct octeon_cop2_state` on the caller's stack.
 *                   This buffer will receive the previous COP2 state if it was
 *                   active for a non-task-specific reason.
 * @return      The original state of the `ST0_CU2` flag *before* this function's
 *              modifications. This return value is crucial and *must* be passed
 *              to `octeon_crypto_disable()` to correctly restore the context.
 * @pre         Called from a kernel context that intends to use the Octeon COP2.
 * @post        Kernel access to COP2 is enabled; previous COP2 context is saved.
 *              Preemption is disabled, and local interrupts are masked.
 */
unsigned long octeon_crypto_enable(struct octeon_cop2_state *state)
{
	int status;
	unsigned long flags;

	// Pre-condition: Disables preemption to ensure atomicity of the context switch operation.
	preempt_disable();
	// Functional Utility: Saves the current interrupt flags and disables local interrupts to prevent interference.
	local_irq_save(flags);
	// Functional Utility: Reads the current value of the MIPS CP0 Status register.
	status = read_c0_status();
	// Functional Utility: Writes a new value to the CP0 Status register, setting the ST0_CU2 bit.
	// This grants the current kernel context access to the COP2 unit.
	write_c0_status(status | ST0_CU2);

	// Block Logic: Checks if the current task (e.g., a user process) had COP2 access enabled.
	if (KSTK_STATUS(current) & ST0_CU2) {
		// Functional Utility: Saves the current task's COP2 context into its thread information structure.
		octeon_cop2_save(&(current->thread.cp2));
		// Functional Utility: Clears the ST0_CU2 bit in the task's saved status, indicating COP2 is no longer owned by the task.
		KSTK_STATUS(current) &= ~ST0_CU2;
		// Functional Utility: Clears the ST0_CU2 bit from the local 'status' variable, reflecting the task's relinquishment.
		status &= ~ST0_CU2;
	// Block Logic: If COP2 was active but not for the current task (e.g., another kernel component),
	// save its current state to the caller-provided stack variable.
	} else if (status & ST0_CU2) {
		// Functional Utility: Saves the generic COP2 context to the provided 'state' buffer.
		octeon_cop2_save(state);
	}
	// Functional Utility: Restores the local interrupt flags, re-enabling interrupts if they were enabled before.
	local_irq_restore(flags);
	// Functional Utility: Returns the original ST0_CU2 status bit, indicating whether COP2 was active before this function.
	return status & ST0_CU2;
}
EXPORT_SYMBOL_GPL(octeon_crypto_enable);

/**
 * @brief Disables kernel access to COP2 and restores the previous context.
 * @details This function is the complementary part to `octeon_crypto_enable()`
 * and must be called after the critical section of COP2 usage is complete.
 * It is responsible for restoring the COP2 hardware to its original state
 * prior to any potential context switch or return to user-space.
 *
 * @param[in] state A pointer to the `struct octeon_cop2_state` which holds
 *                  the COP2 state that needs to be restored if it was saved
 *                  by the corresponding `octeon_crypto_enable()` call.
 * @param[in] crypto_flags The return value from `octeon_crypto_enable()`,
 *                          which indicates the original `ST0_CU2` status and
 *                          guides whether a restore operation is necessary.
 * @pre                 `octeon_crypto_enable()` has been called, and the kernel
 *                      has finished its COP2 operations.
 * @post                COP2 access for the kernel is disabled, and the previous
 *                      COP2 context is restored. Preemption is re-enabled, and
 *                      local interrupts are restored.
 */
void octeon_crypto_disable(struct octeon_cop2_state *state,
			   unsigned long crypto_flags)
{
	unsigned long flags;

	// Functional Utility: Saves the current interrupt flags and disables local interrupts to ensure atomicity.
	local_irq_save(flags);
	// Block Logic: If the 'crypto_flags' indicate that COP2 was originally enabled (meaning its state was saved),
	// restore that saved state. Otherwise, simply disable COP2 access for the kernel.
	if (crypto_flags & ST0_CU2)
		// Functional Utility: Restores the COP2 context from the provided 'state' buffer.
		octeon_cop2_restore(state);
	else
		// Functional Utility: Clears the ST0_CU2 bit in the CP0 Status register, effectively disabling
		// kernel access to the COP2 unit.
		write_c0_status(read_c0_status() & ~ST0_CU2);
	// Functional Utility: Restores the local interrupt flags.
	local_irq_restore(flags);
	// Post-condition: Re-enables preemption, allowing the scheduler to operate normally.
	preempt_enable();
}
EXPORT_SYMBOL_GPL(octeon_crypto_disable);
