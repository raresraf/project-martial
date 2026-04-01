/**
 * @file rtmutex.h
 * @brief Real-Time Mutex (RT Mutex) data structures and API definitions for the Linux kernel.
 *
 * This header provides the necessary definitions for implementing real-time
 * mutual exclusion locks with Priority Inheritance (PI) support. RT Mutexes
 * are designed to prevent priority inversion, a critical issue in real-time
 * operating systems, by ensuring that a task holding a lock temporarily
 * inherits the priority of the highest-priority task waiting for that lock.
 *
 * Domain: Linux Kernel, Real-Time Systems, Concurrency, Synchronization, Priority Inheritance.
 */
/* SPDX-License-Identifier: GPL-2.0 */
/*
 * RT Mutexes: blocking mutual exclusion locks with PI support
 *
 * started by Ingo Molnar and Thomas Gleixner:
 *
 *  Copyright (C) 2004-2006 Red Hat, Inc., Ingo Molnar <mingo@redhat.com>
 *  Copyright (C) 2006, Timesys Corp., Thomas Gleixner <tglx@timesys.com>
 *
 * This file contains the public data structure and API definitions.
 */

#ifndef __LINUX_RT_MUTEX_H
#define __LINUX_RT_MUTEX_H

#include <linux/compiler.h>
#include <linux/linkage.h>
#include <linux/rbtree_types.h>
#include <linux/spinlock_types_raw.h>

extern int max_lock_depth; /* Defines the maximum nesting level for RT mutexes to prevent deadlock. */

/**
 * @brief Core structure for a real-time mutex.
 *
 * This fundamental structure encapsulates the core state of a real-time mutex,
 * providing the essential components for managing mutual exclusion with
 * priority inheritance capabilities.
 */
struct rt_mutex_base {
	raw_spinlock_t		wait_lock; /**< Spinlock protecting the mutex's internal state. */
	struct rb_root_cached   waiters;   /**< Red-black tree storing tasks waiting for the mutex, ordered by priority. */
	struct task_struct	*owner;    /**< Pointer to the task currently holding the mutex. */
};

/**
 * @brief Statically initializes an `rt_mutex_base` structure.
 *
 * This macro provides a convenient way to statically initialize an
 * `rt_mutex_base` structure, setting up its spinlock as unlocked,
 * its waiter queue as empty, and its owner to NULL.
 */
#define __RT_MUTEX_BASE_INITIALIZER(rtbasename)				\
{									\
	.wait_lock = __RAW_SPIN_LOCK_UNLOCKED(rtbasename.wait_lock),	\
	.waiters = RB_ROOT_CACHED,					\
	.owner = NULL							\
}

/**
 * @brief Checks if an RT mutex is currently locked.
 * @param lock A pointer to the `rt_mutex_base` structure to query.
 * @return Returns `true` if the mutex's owner field is not NULL (i.e., locked), `false` otherwise.
 *
 * Functional Utility: This function provides a quick and atomic check for the
 * lock status of a real-time mutex, essential for conditional operations or
 * debugging in a concurrency-aware environment.
 */
static inline bool rt_mutex_base_is_locked(struct rt_mutex_base *lock)
{
	return READ_ONCE(lock->owner) != NULL;
}

/**
 * @brief Initializes an `rt_mutex_base` structure at runtime.
 * @param rtb A pointer to the `rt_mutex_base` structure to initialize.
 *
 * Functional Utility: This function dynamically initializes the given
 * `rt_mutex_base` structure, setting its internal state to an unlocked
 * and empty condition. It prepares the mutex for use in synchronization
 * within the kernel.
 */
extern void rt_mutex_base_init(struct rt_mutex_base *rtb);

/**
 * @brief The main Real-Time Mutex structure.
 *
 * This structure represents a complete real-time mutex, building upon the
 * `rt_mutex_base` to provide a robust synchronization primitive for kernel
 * tasks. It includes priority inheritance mechanisms to prevent priority
 * inversions and can optionally incorporate lock debugging capabilities
 * if `CONFIG_DEBUG_LOCK_ALLOC` is enabled.
 */
struct rt_mutex {
	struct rt_mutex_base	rtmutex; /**< The core real-time mutex functionality. */
#ifdef CONFIG_DEBUG_LOCK_ALLOC
	struct lockdep_map	dep_map; /**< Lock dependency map for debugging and deadlock detection. */
#endif
};

struct rt_mutex_waiter;
struct hrtimer_sleeper;

#ifdef CONFIG_DEBUG_RT_MUTEXES
/**
 * @brief Frees RT mutex debugging information associated with a task.
 * @param tsk A pointer to the `task_struct` whose debugging information is to be freed.
 *
 * Functional Utility: When `CONFIG_DEBUG_RT_MUTEXES` is enabled, this function
 * ensures that any debugging data structures related to RT mutexes held or
 * waited upon by the specified task are properly cleaned up upon task exit or
 * deallocation. This prevents memory leaks and spurious debugging reports.
 */
extern void rt_mutex_debug_task_free(struct task_struct *tsk);
#else
/**
 * @brief Placeholder for `rt_mutex_debug_task_free` when debugging is disabled.
 * @param tsk A pointer to the `task_struct`.
 *
 * Functional Utility: Provides a no-operation (no-op) stub when
 * `CONFIG_DEBUG_RT_MUTEXES` is not enabled, ensuring that calls to
 * `rt_mutex_debug_task_free` compile without errors but have no runtime effect.
 */
static inline void rt_mutex_debug_task_free(struct task_struct *tsk) { }
#endif

/**
 * @brief Initializes a real-time mutex.
 * @param mutex A pointer to the `rt_mutex` structure to initialize.
 *
 * Functional Utility: This macro provides a safe and proper way to initialize
 * an `rt_mutex` structure, particularly for dynamically allocated mutexes.
 * It sets up the underlying `rt_mutex_base` and, if lock debugging is enabled,
 * registers a unique lock class key to allow for lock dependency tracking
 * and deadlock detection.
 */
#define rt_mutex_init(mutex) \
do { \
	static struct lock_class_key __key; \
	__rt_mutex_init(mutex, __func__, &__key); \
} while (0)

#ifdef CONFIG_DEBUG_LOCK_ALLOC
/**
 * @brief Initializes the lock dependency map for an RT mutex.
 *
 * This macro is used when `CONFIG_DEBUG_LOCK_ALLOC` is enabled to set up
 * the `lockdep_map` within an `rt_mutex` structure. It assigns a name
 * to the lock for debugging and specifies its waiting type, facilitating
 * deadlock detection and lock order validation.
 */
#define __DEP_MAP_RT_MUTEX_INITIALIZER(mutexname)	\
	.dep_map = {					\
		.name = #mutexname,			\
		.wait_type_inner = LD_WAIT_SLEEP,	\
	}
#else
/**
 * @brief Placeholder for `__DEP_MAP_RT_MUTEX_INITIALIZER` when debugging is disabled.
 *
 * Functional Utility: Provides an empty definition when lock debugging is not
 * enabled, ensuring that `__DEP_MAP_RT_MUTEX_INITIALIZER` expands to nothing
 * and thus does not introduce any overhead or compilation errors.
 */
#define __DEP_MAP_RT_MUTEX_INITIALIZER(mutexname)
#endif

/**
 * @brief Statically initializes an `rt_mutex` structure.
 *
 * This macro provides a comprehensive static initializer for an `rt_mutex`
 * structure. It combines the initialization of the core `rt_mutex_base`
 * with the conditional initialization of the `lockdep_map` for debugging,
 * ensuring the mutex is properly set up from its declaration.
 */
#define __RT_MUTEX_INITIALIZER(mutexname)				\
{									\
	.rtmutex = __RT_MUTEX_BASE_INITIALIZER(mutexname.rtmutex),	\
	__DEP_MAP_RT_MUTEX_INITIALIZER(mutexname)			\
}

/**
 * @brief Declares and statically initializes a real-time mutex.
 * @param mutexname The name of the `rt_mutex` variable to declare and initialize.
 *
 * Functional Utility: This macro simplifies the declaration and static
 * initialization of an `rt_mutex` variable, ensuring it is ready for use
 * at compile time with all its internal components correctly set up.
 */
#define DEFINE_RT_MUTEX(mutexname) \
	struct rt_mutex mutexname = __RT_MUTEX_INITIALIZER(mutexname)

/**
 * @brief Internal function for initializing a real-time mutex.
 * @param lock A pointer to the `rt_mutex` structure to initialize.
 * @param name The name of the mutex, used for debugging.
 * @param key A pointer to a `lock_class_key` for lock dependency tracking.
 *
 * Functional Utility: This function performs the core initialization of an
 * `rt_mutex` structure, setting up its base components and, if lock debugging
 * is enabled, associating it with a unique name and lock class key for
 * tracking and analysis of locking behavior.
 */
extern void __rt_mutex_init(struct rt_mutex *lock, const char *name, struct lock_class_key *key);

#ifdef CONFIG_DEBUG_LOCK_ALLOC
/**
 * @brief Acquires a real-time mutex with nested lock tracking.
 * @param lock A pointer to the `rt_mutex` to acquire.
 * @param subclass A numerical identifier for the lock's subclass, used by lockdep.
 *
 * Functional Utility: This function is used when `CONFIG_DEBUG_LOCK_ALLOC`
 * is enabled to acquire an `rt_mutex`. It performs the standard mutex
 * acquisition while also informing the lock dependency validator (lockdep)
 * about the nested locking context, helping to detect potential deadlocks
 * and enforce lock ordering rules.
 */
extern void rt_mutex_lock_nested(struct rt_mutex *lock, unsigned int subclass);
/**
 * @brief Acquires a real-time mutex with an explicit nested lock map.
 * @param lock A pointer to the `rt_mutex` to acquire.
 * @param nest_lock A pointer to a `lockdep_map` representing the nesting lock.
 *
 * Functional Utility: This variant is used for acquiring an `rt_mutex` within
 * a nested locking scenario where the nesting lock's `lockdep_map` is
 * explicitly provided. It ensures that lockdep correctly tracks the lock
 * acquisition order and nesting relationships for debugging purposes.
 */
extern void _rt_mutex_lock_nest_lock(struct rt_mutex *lock, struct lockdep_map *nest_lock);
/**
 * @brief Acquires a real-time mutex (debug-enabled version).
 * @param lock A pointer to the `rt_mutex` to acquire.
 *
 * Functional Utility: This macro expands to `rt_mutex_lock_nested` when
 * lock debugging is enabled, providing the default entry point for acquiring
 * an `rt_mutex` with lockdep tracking.
 */
#define rt_mutex_lock(lock) rt_mutex_lock_nested(lock, 0)
/**
 * @brief Acquires a real-time mutex with an explicit nested lock map (debug-enabled version).
 * @param lock A pointer to the `rt_mutex` to acquire.
 * @param nest_lock The nesting lock object (struct type) whose `dep_map` will be used.
 *
 * Functional Utility: This macro is a type-checked wrapper around
 * `_rt_mutex_lock_nest_lock`, ensuring that a valid `lockdep_map` is passed
 * for nested lock tracking when lock debugging is enabled.
 */
#define rt_mutex_lock_nest_lock(lock, nest_lock)			\
	do {								\
		typecheck(struct lockdep_map *, &(nest_lock)->dep_map);	\
		_rt_mutex_lock_nest_lock(lock, &(nest_lock)->dep_map);	\
	} while (0)

#else
/**
 * @brief Acquires a real-time mutex (non-debug version).
 * @param lock A pointer to the `rt_mutex` to acquire.
 *
 * Functional Utility: This function attempts to acquire the given
 * `rt_mutex`. If the mutex is already held, the calling task will
 * block until the mutex becomes available. This is the standard,
 * non-debug entry point for acquiring a real-time mutex.
 */
extern void rt_mutex_lock(struct rt_mutex *lock);
/**
 * @brief Acquires a real-time mutex with a subclass (non-debug version).
 * @param lock A pointer to the `rt_mutex` to acquire.
 * @param subclass This parameter is ignored in the non-debug build.
 *
 * Functional Utility: In non-debug builds, this macro simply redirects
 * to `rt_mutex_lock`, as lock dependency tracking and subclassing are
 * not performed.
 */
#define rt_mutex_lock_nested(lock, subclass) rt_mutex_lock(lock)
/**
 * @brief Acquires a real-time mutex with a nested lock map (non-debug version).
 * @param lock A pointer to the `rt_mutex` to acquire.
 * @param nest_lock This parameter is ignored in the non-debug build.
 *
 * Functional Utility: In non-debug builds, this macro simply redirects
 * to `rt_mutex_lock`, as explicit nested lock map tracking is not performed.
 */
#define rt_mutex_lock_nest_lock(lock, nest_lock) rt_mutex_lock(lock)
#endif

/**
 * @brief Acquires a real-time mutex, allowing interruption by signals.
 * @param lock A pointer to the `rt_mutex` to acquire.
 * @return 0 if the mutex was acquired, -EINTR if interrupted by a signal.
 *
 * Functional Utility: This function attempts to acquire the `rt_mutex`. If
 * the mutex is unavailable, the calling task will block, but it can be
 * woken up and return prematurely if a signal is received. This is useful
 * for operations that need to be responsive to user input or system events.
 */
extern int rt_mutex_lock_interruptible(struct rt_mutex *lock);
/**
 * @brief Acquires a real-time mutex, allowing termination by fatal signals.
 * @param lock A pointer to the `rt_mutex` to acquire.
 * @return 0 if the mutex was acquired, -EINTR if interrupted by a fatal signal.
 *
 * Functional Utility: Similar to `rt_mutex_lock_interruptible`, this function
 * attempts to acquire the mutex but specifically allows the task to be killed
 * (terminated) by a fatal signal while waiting. This is crucial for resource
 * acquisition in scenarios where blocking indefinitely could lead to system
 * unresponsiveness or deadlock in the face of process termination requests.
 */
extern int rt_mutex_lock_killable(struct rt_mutex *lock);
/**
 * @brief Attempts to acquire a real-time mutex without blocking.
 * @param lock A pointer to the `rt_mutex` to try acquiring.
 * @return 1 if the mutex was successfully acquired, 0 otherwise.
 *
 * Functional Utility: This function provides a non-blocking attempt to acquire
 * the `rt_mutex`. If the mutex is already held, the function immediately
 * returns without putting the calling task to sleep. This is suitable for
 * scenarios where the task can perform alternative work if the resource is
 * not immediately available.
 */
extern int rt_mutex_trylock(struct rt_mutex *lock);

/**
 * @brief Releases a previously acquired real-time mutex.
 * @param lock A pointer to the `rt_mutex` to release.
 *
 * Functional Utility: This function releases the `rt_mutex`, making it
 * available for other waiting tasks. It also handles priority adjustments
 * if priority inheritance was activated during the lock's tenure, ensuring
 * that waiting higher-priority tasks are woken up and assigned the CPU.
 */
extern void rt_mutex_unlock(struct rt_mutex *lock);

#endif
