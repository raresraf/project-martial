/* SPDX-License-Identifier: GPL-2.0 */
/*
 * @25a65e34-59dd-45fa-a291-115057d5e1f8/include/linux/rtmutex.h
 * @brief Public API and data structures for Real-Time (RT) Mutexes with Priority Inheritance (PI).
 *
 * RT Mutexes are designed to mitigate priority inversion in real-time systems by 
 * implementing a priority inheritance protocol. When a high-priority task blocks on 
 * a mutex held by a lower-priority task, the holder's priority is temporarily 
 * elevated to match the high-priority task.
 *
 * Functional Intent: Provides deterministic blocking mutual exclusion for kernel-level 
 * synchronization where latency guarantees are critical.
 */

#ifndef __LINUX_RT_MUTEX_H
#define __LINUX_RT_MUTEX_H

#include <linux/compiler.h>
#include <linux/linkage.h>
#include <linux/rbtree_types.h>
#include <linux/spinlock_types_raw.h>

extern int max_lock_depth; /* for sysctl */

/**
 * struct rt_mutex_base - Foundation for RT-aware mutual exclusion.
 * @wait_lock: Raw spinlock ensuring atomic access to the mutex state and waiter list.
 * @waiters: Augmented red-black tree (rb_root_cached) storing blocked tasks 
 *           indexed by their priority to facilitate O(1) top-waiter retrieval.
 * @owner: Pointer to the task_struct currently holding the lock; encodes PI state.
 */
struct rt_mutex_base {
	raw_spinlock_t		wait_lock;
	struct rb_root_cached   waiters;
	struct task_struct	*owner;
};

/**
 * Functional Utility: Static initializer for the base RT mutex structure.
 * Logic: Disables the wait_lock by default, initializes an empty rbtree for waiters, 
 * and sets the initial ownership to NULL.
 */
#define __RT_MUTEX_BASE_INITIALIZER(rtbasename)				\
{									\
	.wait_lock = __RAW_SPIN_LOCK_UNLOCKED(rtbasename.wait_lock),	\
	.waiters = RB_ROOT_CACHED,					\
	.owner = NULL							\
}

/**
 * rt_mutex_base_is_locked - Predicate to check the current locking state.
 * @lock: Pointer to the base mutex structure.
 * Returns: true if the owner field is non-NULL, indicating active acquisition.
 */
static inline bool rt_mutex_base_is_locked(struct rt_mutex_base *lock)
{
	return READ_ONCE(lock->owner) != NULL;
}

extern void rt_mutex_base_init(struct rt_mutex_base *rtb);

/**
 * struct rt_mutex - High-level RT mutex container.
 * @rtmutex: Internal base structure containing core PI logic and state.
 * @dep_map: Lock dependency metadata for kernel debugging (lockdep).
 * 
 * Functional Intent: Wraps the base PI-capable mutex with additional 
 * instrumentation for debugging and validation.
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
 * Functional Utility: Runtime initializer for RT mutexes.
 * Logic: Employs a unique lock_class_key per call site to enable fine-grained 
 * lockdep tracking and deadlock detection.
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
 * Block Logic: Aggregate initializer for RT mutex instances.
 * Logic: Chains base state initialization with optional lockdep metadata.
 */
#define __RT_MUTEX_INITIALIZER(mutexname)				\
{									\
	.rtmutex = __RT_MUTEX_BASE_INITIALIZER(mutexname.rtmutex),	\
	__DEP_MAP_RT_MUTEX_INITIALIZER(mutexname)			\
}

#define DEFINE_RT_MUTEX(mutexname) \
	struct rt_mutex mutexname = __RT_MUTEX_INITIALIZER(mutexname)

extern void __rt_mutex_init(struct rt_mutex *lock, const char *name, struct lock_class_key *key);

#ifdef CONFIG_DEBUG_LOCK_ALLOC
extern void rt_mutex_lock_nested(struct rt_mutex *lock, unsigned int subclass);
extern void _rt_mutex_lock_nest_lock(struct rt_mutex *lock, struct lockdep_map *nest_lock);
/**
 * Block Logic: Locking primitives with lockdep nesting support.
 * Logic: Prevents false positive deadlock reports by explicitly defining 
 * acquisition hierarchies (subclasses) or nesting relationships.
 */
#define rt_mutex_lock(lock) rt_mutex_lock_nested(lock, 0)
#define rt_mutex_lock_nest_lock(lock, nest_lock)			\
	do {								\
		typecheck(struct lockdep_map *, &(nest_lock)->dep_map);	\
		_rt_mutex_lock_nest_lock(lock, &(nest_lock)->dep_map);	\
	} while (0)

#else
extern void rt_mutex_lock(struct rt_mutex *lock);
#define rt_mutex_lock_nested(lock, subclass) rt_mutex_lock(lock)
#define rt_mutex_lock_nest_lock(lock, nest_lock) rt_mutex_lock(lock)
#endif

/**
 * External API: Variants of the acquisition protocol.
 * Logic: Supports interruptible and killable wait states, as well as 
 * non-blocking 'trylock' attempts.
 */
extern int rt_mutex_lock_interruptible(struct rt_mutex *lock);
extern int rt_mutex_lock_killable(struct rt_mutex *lock);
extern int rt_mutex_trylock(struct rt_mutex *lock);

/**
 * External API: Release protocol.
 * Logic: Transfers ownership and triggers PI priority de-boosting for the caller.
 */
extern void rt_mutex_unlock(struct rt_mutex *lock);

#endif
