/**
 * @file octeon-crypto.c
 * @brief Core context-switching logic for Cavium Octeon's crypto co-processor (COP2).
 * @details This file implements the essential functions to manage safe, shared
 * access to the Octeon's cryptographic hardware. It provides a mechanism
 * (`octeon_crypto_enable` and `octeon_crypto_disable`) that allows kernel-level
 * drivers to use the crypto co-processor without corrupting the state of a
 * user-space process that might also be using it. This is achieved by saving
 * and restoring the COP2 context, analogous to how FPU context is managed.
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
 * @brief Enables kernel access to the Octeon crypto co-processor (COP2).
 * @details This function prepares the hardware for use by a kernel driver.
 * It disables preemption, enables COP2 access for the kernel by setting the
 * ST0_CU2 bit, and performs the necessary context switch.
 *
 * If the current task was already using COP2 (indicating the kernel
 * interrupted a user-space crypto operation), the user-space COP2 state is
 * saved into the task structure. Otherwise, if COP2 was active for any other
 * reason, its state is saved to the stack-allocated `state` parameter.
 *
 * This function and octeon_crypto_disable() form a critical pair that must
 * wrap any kernel code using the COP2 hardware.
 *
 * @param[out] state Pointer to a structure on the caller's stack where the
 *                   co-processor's state will be saved if it was previously in use.
 * @return      The original state of the ST0_CU2 flag, which must be passed
 *              to octeon_crypto_disable() to correctly restore the context.
 */
unsigned long octeon_crypto_enable(struct octeon_cop2_state *state)
{
	int status;
	unsigned long flags;

	// Pre-condition: Disable preemption to ensure atomicity of the context switch.
	preempt_disable();
	local_irq_save(flags);
	status = read_c0_status();
	// Logic: Grant the kernel access to the COP2 unit.
	write_c0_status(status | ST0_CU2);

	// Block Logic: If the current task was using COP2 (e.g., a user process),
	// save its state to the task struct so it can be restored later.
	if (KSTK_STATUS(current) & ST0_CU2) {
		octeon_cop2_save(&(current->thread.cp2));
		KSTK_STATUS(current) &= ~ST0_CU2;
		status &= ~ST0_CU2;
	// Block Logic: If COP2 was active for another reason, save its current
	// state to the provided stack variable as a precaution.
	} else if (status & ST0_CU2) {
		octeon_cop2_save(state);
	}
	local_irq_restore(flags);
	return status & ST0_CU2;
}
EXPORT_SYMBOL_GPL(octeon_crypto_enable);

/**
 * @brief Disables kernel access to COP2 and restores the previous context.
 * @details This function must be called after octeon_crypto_enable() to
 * complete the critical section and restore the hardware to its original
 * state before any potential context switch or return to userspace.
 *
 * @param[in] state A pointer to the structure containing the COP2 state to be
 *                  restored if it was saved by the enable function.
 * @param[in] crypto_flags The flags returned by the corresponding
 *                         octeon_crypto_enable() call, used to determine
 *                         whether a restore is necessary.
 */
void octeon_crypto_disable(struct octeon_cop2_state *state,
			   unsigned long crypto_flags)
{
	unsigned long flags;

	local_irq_save(flags);
	// Block Logic: If the original context had COP2 enabled, restore the saved
	// state. Otherwise, simply disable kernel access to COP2.
	if (crypto_flags & ST0_CU2)
		octeon_cop2_restore(state);
	else
		write_c0_status(read_c0_status() & ~ST0_CU2);
	local_irq_restore(flags);
	// Post-condition: Re-enable preemption.
	preempt_enable();
}
EXPORT_SYMBOL_GPL(octeon_crypto_disable);
