/* SPDX-License-Identifier: GPL-2.0 */
/**
 * @file rtmutex.h
 * @brief Public API for Real-Time Mutexes with Priority Inheritance.
 *
 * This header file defines the public data structures and API for Real-Time (RT)
 * mutexes. RT-mutexes are a special type of mutex designed for real-time
 * systems. Their key feature is Priority Inheritance (PI), a mechanism that
 * temporarily boosts the priority of a lock-holding task to the priority of the
 * highest-priority task waiting for that lock. This mechanism is critical for
 * preventing "priority inversion," a condition where a high-priority task is
 * forced to wait for a lower-priority task, which can lead to missed deadlines
 * in a real-time environment.
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

extern int max_lock_depth;

/**
 * struct rt_mutex_base - The core fields of an RT-mutex, shared with other locking structures.
 * @wait_lock: A raw spinlock to protect the rt_mutex's internal data structures (waiters tree, owner field).
 * @waiters:   An rbtree that stores the tasks waiting for the lock, ordered by priority.
 *             This allows for efficient retrieval of the highest-priority waiter.
 * @owner:     A pointer to the task_struct of the current lock owner. This is also used
 *             as the primary indicator of the lock's state (NULL means unlocked).
 */
struct rt_mutex_base {
	raw_spinlock_t		wait_lock;
	struct rb_root_cached   waiters;
	struct task_struct	*owner;
};

#define __RT_MUTEX_BASE_INITIALIZER(rtbasename)				\
{									\
	.wait_lock = __RAW_SPIN_LOCK_UNLOCKED(rtbasename.wait_lock),	\
	.waiters = RB_ROOT_CACHED,					\
	.owner = NULL							\
}

/**
 * rt_mutex_base_is_locked - Checks if the rtmutex is currently locked.
 * @lock: The mutex base to be queried.
 *
 * This function provides a quick, non-locking way to check if the mutex is
 * owned by any task.
 *
 * Return: True if the mutex is locked, false if unlocked.
 */
static inline bool rt_mutex_base_is_locked(struct rt_mutex_base *lock)
{
	return READ_ONCE(lock->owner) != NULL;
}

extern void rt_mutex_base_init(struct rt_mutex_base *rtb);

/**
 * struct rt_mutex - The user-visible RT-mutex structure.
 * @rtmutex: The core, shared fields of the RT-mutex.
 * @dep_map: The lock dependency validator map, used for debugging lock contention issues.
 */
struct rt_mutex {
	struct rt_mutex_base	rtmutex;
#ifdef CONFIG_DEBUG_LOCK_ALLOC
	struct lockdep_map	dep_map;
#endif
};

struct rt_mutex_waiter;
struct hrtimer_sleeper;

#ifdef CONFIG_DEBUG_RT_MUTEXES
extern void rt_mutex_debug_task_free(struct task_struct *tsk);
#else
static inline void rt_mutex_debug_task_free(struct task_struct *tsk) { }
#endif

/**
 * rt_mutex_init - Initializes a dynamically allocated rt_mutex.
 * @mutex: The rt_mutex to be initialized.
 */
#define rt_mutex_init(mutex) \
do { \
	static struct lock_class_key __key; \
	__rt_mutex_init(mutex, __func__, &__key); \
} while (0)

#ifdef CONFIG_DEBUG_LOCK_ALLOC
#define __DEP_MAP_RT_MUTEX_INITIALIZER(mutexname)	\
	.dep_map = {					\
		.name = #mutexname,			\
		.wait_type_inner = LD_WAIT_SLEEP,	\
	}
#else
#define __DEP_MAP_RT_MUTEX_INITIALIZER(mutexname)
#endif

/**
 * __RT_MUTEX_INITIALIZER - Statically initializes an rt_mutex.
 * @mutexname: The variable name of the rt_mutex.
 */
#define __RT_MUTEX_INITIALIZER(mutexname)				\
{									\
	.rtmutex = __RT_MUTEX_BASE_INITIALIZER(mutexname.rtmutex),	\
	__DEP_MAP_RT_MUTEX_INITIALIZER(mutexname)			\
}

/**
 * DEFINE_RT_MUTEX - Defines and statically initializes an rt_mutex.
 * @mutexname: The variable name of the rt_mutex to be defined.
 */
#define DEFINE_RT_MUTEX(mutexname) \
	struct rt_mutex mutexname = __RT_MUTEX_INITIALIZER(mutexname)

extern void __rt_mutex_init(struct rt_mutex *lock, const char *name, struct lock_class_key *key);

#ifdef CONFIG_DEBUG_LOCK_ALLOC
extern void rt_mutex_lock_nested(struct rt_mutex *lock, unsigned int subclass);
extern void _rt_mutex_lock_nest_lock(struct rt_mutex *lock, struct lockdep_map *nest_lock);
/**
 * rt_mutex_lock - Acquires the rt_mutex.
 * @lock: The rt_mutex to be locked.
 *
 * A task calling this function will block if the mutex is already held. If the
 * calling task has a higher priority than the current owner, the owner's
 * priority will be elevated to match the caller's priority (Priority Inheritance).
 */
#define rt_mutex_lock(lock) rt_mutex_lock_nested(lock, 0)
#define rt_mutex_lock_nest_lock(lock, nest_lock)			\
	do {								\
		typecheck(struct lockdep_map *, &(nest_lock)->dep_map);	\
		_rt_mutex_lock_nest_lock(lock, &(nest_lock)->dep_map);	\
	} while (0)

#else
/**
 * rt_mutex_lock - Acquires the rt_mutex.
 * @lock: The rt_mutex to be locked.
 *
 * A task calling this function will block if the mutex is already held. If the
 * calling task has a higher priority than the current owner, the owner's
 * priority will be elevated to match the caller's priority (Priority Inheritance).
 */
extern void rt_mutex_lock(struct rt_mutex *lock);
#define rt_mutex_lock_nested(lock, subclass) rt_mutex_lock(lock)
#define rt_mutex_lock_nest_lock(lock, nest_lock) rt_mutex_lock(lock)
#endif

/**
 * rt_mutex_lock_interruptible - Acquires the rt_mutex, but can be interrupted by a signal.
 * @lock: The rt_mutex to be locked.
 *
 * Return: 0 on success, -EINTR if interrupted by a signal.
 */
extern int rt_mutex_lock_interruptible(struct rt_mutex *lock);

/**
 * rt_mutex_lock_killable - Acquires the rt_mutex, but can be interrupted by a fatal signal.
 * @lock: The rt_mutex to be locked.
 *
 * Return: 0 on success, -EINTR if interrupted by a kill signal.
 */
extern int rt_mutex_lock_killable(struct rt_mutex *lock);

/**
 * rt_mutex_trylock - Tries to acquire the rt_mutex without blocking.
 * @lock: The rt_mutex to be tried.
 *
 * Return: 1 if the lock was acquired successfully, 0 if it was already held.
 */
extern int rt_mutex_trylock(struct rt_mutex *lock);

/**
 * rt_mutex_unlock - Releases the rt_mutex.
 * @lock: The rt_mutex to be unlocked.
 *
 * If the lock owner's priority was boosted due to Priority Inheritance, its
 * original priority is restored. The highest-priority waiting task, if any,
 * is then woken up to acquire the lock.
 */
extern void rt_mutex_unlock(struct rt_mutex *lock);

#endif
