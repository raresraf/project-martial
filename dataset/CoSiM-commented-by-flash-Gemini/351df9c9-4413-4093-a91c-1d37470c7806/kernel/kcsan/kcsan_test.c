/**
 * @file kcsan_test.c
 * @brief KUnit test suite for KCSAN (Kernel Concurrency Sanitizer) runtime behavior.
 *
 * This file implements a comprehensive set of KUnit tests to verify KCSAN's
 * ability to detect various data race scenarios and report them correctly.
 * It leverages the kernel's `console` tracepoint to capture KCSAN reports
 * and the `Torture` framework for precise control over test threads.
 * The tests focus on confirming the presence or absence of KCSAN reports
 * for different types of memory accesses, synchronization primitives, and
 * memory ordering semantics.
 *
 * @author Marco Elver <elver@google.com>
 * @copyright Copyright (C) 2020, Google LLC.
 * @license SPDX-License-Identifier: GPL-2.0
 */

// SPDX-License-Identifier: GPL-2.0
/*
 * KCSAN test with various race scenarious to test runtime behaviour. Since the
 * interface with which KCSAN's reports are obtained is via the console, this is
 * the output we should verify. For each test case checks the presence (or
 * absence) of generated reports. Relies on 'console' tracepoint to capture
 * reports as they appear in the kernel log.
 *
 * Makes use of KUnit for test organization, and the Torture framework for test
 * thread control.
 *
 * Copyright (C) 2020, Google LLC.
 * Author: Marco Elver <elver@google.com>
 */

#define pr_fmt(fmt) "kcsan_test: " fmt

#include <kunit/test.h>
#include <linux/atomic.h>
#include <linux/bitops.h>
#include <linux/jiffies.h>
#include <linux/kcsan-checks.h>
#include <linux/kernel.h>
#include <linux/mutex.h>
#include <linux/sched.h>
#include <linux/seqlock.h>
#include <linux/spinlock.h>
#include <linux/string.h>
#include <linux/timer.h>
#include <linux/torture.h>
#include <linux/tracepoint.h>
#include <linux/types.h>
#include <trace/events/printk.h>

/**
 * @brief Macro to define a test dependency, skipping the test if the condition is not met.
 * @param test The KUnit test context.
 * @param cond The condition that must be true for the test to run.
 */
#define KCSAN_TEST_REQUIRES(test, cond) do {			\
	if (!(cond))						\
		kunit_skip((test), "Test requires: " #cond);	\
} while (0)

/**
 * @brief Helper macro to define the KCSAN access type for a Read-Write (RW) compound access.
 *
 * This macro adjusts its definition based on whether `CONFIG_CC_HAS_TSAN_COMPOUND_READ_BEFORE_WRITE`
 * is defined, which indicates specific compiler support for ThreadSanitizer-like compound access
 * instrumentation.
 *
 * @param alt An alternative access type to use if compound read-before-write is not supported.
 * @return `KCSAN_ACCESS_COMPOUND | KCSAN_ACCESS_WRITE` if compound read-before-write is supported,
 *         otherwise returns `alt`.
 */
#ifdef CONFIG_CC_HAS_TSAN_COMPOUND_READ_BEFORE_WRITE
#define __KCSAN_ACCESS_RW(alt) (KCSAN_ACCESS_COMPOUND | KCSAN_ACCESS_WRITE)
#else
#define __KCSAN_ACCESS_RW(alt) (alt)
#endif

/**
 * @brief Array of function pointers to current test-case memory access "kernels".
 *
 * This array holds two function pointers, `access_kernels[0]` and `access_kernels[1]`,
 * which are dynamically set by `begin_test_checks` to control the specific memory
 * access patterns used in a given KCSAN test scenario.
 */
static void (*access_kernels[2])(void);

/**
 * @brief Pointer to an array of `task_struct` pointers, representing the worker threads.
 *
 * This variable stores references to the `torture` framework's worker threads
 * that concurrently execute memory access "kernels" during KCSAN tests.
 * @warning This array is dynamically allocated and must be freed.
 */
static struct task_struct **threads;

/**
 * @brief Stores the jiffies value indicating the end time of the current test.
 *
 * Used in conjunction with `time_before(jiffies, end_time)` to control the
 * duration of test loops and ensure that KCSAN has sufficient time to report races.
 */
static unsigned long end_time;

/**
 * @brief Structure to store information about KCSAN reports observed from the console.
 *
 * This structure acts as a buffer to capture KCSAN reports, which are printed
 * to the kernel console. It includes a spinlock for safe concurrent access,
 * a counter for the number of lines captured, and a buffer for the report lines themselves.
 */
static struct {
	spinlock_t lock; /**< Spinlock to protect concurrent access to `observed` data. */
	int nlines;      /**< Number of lines currently captured in `lines`. */
	char lines[3][512]; /**< Buffer to store captured report lines (max 3 lines, 512 chars each). */
} observed = {
	.lock = __SPIN_LOCK_UNLOCKED(observed.lock),
};

/**
 * @brief Sets up the test checking loop by configuring the memory access "kernels" and test duration.
 *
 * This function disables KCSAN instrumentation for the current thread to prevent interference
 * with the test setup itself. It calculates the `end_time` to ensure the test runs long enough
 * for KCSAN to report races, and then publishes the function pointers for the memory access
 * patterns to be executed by worker threads.
 *
 * @param func1 Function pointer to the first memory access "kernel" for worker threads.
 * @param func2 Function pointer to the second memory access "kernel" for worker threads.
 * @pre KCSAN is expected to be enabled globally for race detection.
 * @post `access_kernels` array is populated with `func1` and `func2`, and `end_time` is set.
 */
static __no_kcsan inline void
begin_test_checks(void (*func1)(void), void (*func2)(void))
{
	kcsan_disable_current(); // Disable KCSAN for this thread to avoid self-reporting.

	/*
	 * Block Logic: Calculate the end time for the test. The duration is set to be at
	 * least as long as `CONFIG_KCSAN_REPORT_ONCE_IN_MS` plus a buffer, to ensure
	 * that if a race occurs, KCSAN has enough time to detect and report it.
	 */
	end_time = jiffies + msecs_to_jiffies(CONFIG_KCSAN_REPORT_ONCE_IN_MS + 500);

	/*
	 * Block Logic: Publish the function pointers to `access_kernels` using a store-release
	 * barrier. This ensures that the worker threads (which will load these pointers
	 * with an acquire barrier) see the fully initialized function pointers, preventing
	 * data races on the `access_kernels` array itself.
	 */
	smp_store_release(&access_kernels[0], func1);
	smp_store_release(&access_kernels[1], func2);
}

/**
 * @brief Manages the termination of the test checking loop.
 *
 * This function is called periodically within the test loop to determine if the
 * test should continue or stop. It re-enables KCSAN for the current thread upon
 * test completion.
 *
 * @param stop A boolean flag indicating if an expected report has been found,
 *             signaling an early stop for the test.
 * @return `true` if the test loop should terminate, `false` if it should continue.
 * @pre `begin_test_checks` must have been called previously.
 * @post KCSAN is re-enabled for the current thread if the test is stopping.
 */
static __no_kcsan inline bool
end_test_checks(bool stop)
{
	// Block Logic: If not explicitly stopping and the test duration has not elapsed, continue the test.
	if (!stop && time_before(jiffies, end_time)) {
		/* Continue checking */
		might_sleep(); // Allow other tasks/threads to run.
		return false;
	}

	kcsan_enable_current(); // Re-enable KCSAN for this thread.
	return true; // Signal to stop the test loop.
}

/**
 * @brief Tracepoint callback function to probe kernel console output for KCSAN reports.
 *
 * This function is registered with the `console` tracepoint. It intercepts kernel log messages
 * and processes them to identify and capture KCSAN data race reports related to the tests.
 * It stores relevant lines of the report in the `observed` global structure.
 *
 * @param ignore Unused context pointer.
 * @param buf Pointer to the console output buffer.
 * @param len Length of the console output buffer.
 * @pre KCSAN reports are expected to be formatted in a specific way ("BUG: KCSAN: ").
 * @post The `observed` structure may be updated with lines from a KCSAN report.
 * @invariant KCSAN reports are serialized under a global lock, preventing interleaved reports.
 */
__no_kcsan
static void probe_console(void *ignore, const char *buf, size_t len)
{
	unsigned long flags;
	int nlines;

	/*
	 * Invariant: KCSAN reports are generated under a global lock. This design ensures that
	 * individual reports are not interleaved in the console output, simplifying parsing and
	 * avoiding potential test failures due to fragmented reports.
	 */

	spin_lock_irqsave(&observed.lock, flags); // Protect access to the `observed` global structure.
	nlines = observed.nlines;

	// Block Logic: Identify the start of a KCSAN report related to the test.
	if (strnstr(buf, "BUG: KCSAN: ", len) && strnstr(buf, "test_", len)) {
		/*
		 * KCSAN report and related to the test.
		 *
		 * The provided @buf is not NUL-terminated; copy no more than
		 * @len bytes and let strscpy() add the missing NUL-terminator.
		 */
		// Block Logic: Copy the first line of the report.
		strscpy(observed.lines[0], buf, min(len + 1, sizeof(observed.lines[0])));
		nlines = 1;
	}
	// Block Logic: Capture subsequent lines of interest (e.g., lines describing memory accesses).
	else if ((nlines == 1 || nlines == 2) && strnstr(buf, "bytes by", len)) {
		strscpy(observed.lines[nlines++], buf, min(len + 1, sizeof(observed.lines[0])));

		// Special case for "race at unknown origin" reports, which might not have a second access line.
		if (strnstr(buf, "race at unknown origin", len)) {
			if (WARN_ON(nlines != 2)) // Expecting to be on the second line.
				goto out;

			/* No second line of interest. */
			// Block Logic: Fill the third line with a placeholder if no second access line exists.
			strcpy(observed.lines[nlines++], "<none>");
		}
	}

out:
	// Publish the updated number of lines using WRITE_ONCE to ensure visibility across CPUs.
	WRITE_ONCE(observed.nlines, nlines);
	spin_unlock_irqrestore(&observed.lock, flags); // Release the lock.
}

/**
 * @brief Checks if a complete KCSAN report related to the current test has been captured.
 *
 * This function determines if all expected lines of a KCSAN report have been observed
 * in the console output by checking the `nlines` field of the `observed` global structure.
 *
 * @return `true` if a full report is available, `false` otherwise.
 * @post The state of `observed` is read, but not modified.
 */
__no_kcsan
static bool report_available(void)
{
	// Block Logic: Compare the number of observed lines with the expected total number of lines in a report.
	return READ_ONCE(observed.nlines) == ARRAY_SIZE(observed.lines);
}

/**
 * @brief Structure defining the expected content of a KCSAN report.
 *
 * This structure is used to specify the characteristics of the memory accesses
 * that KCSAN is expected to report, allowing for programmatic verification
 * of the report against actual console output.
 */
struct expect_report {
	/**
	 * @brief Array of two access information structures, representing the two racing accesses.
	 *
	 * Each element describes one of the two memory accesses involved in a data race.
	 * The order of accesses in the report is not guaranteed, so this structure
	 * facilitates matching regardless of order.
	 */
	struct {
		void *fn;    /**< Function pointer to the expected function of the top stack frame performing the access. */
		void *addr;  /**< Address of the memory access; if NULL, the address is not checked. */
		size_t size; /**< Size of the memory access in bytes; unchecked if `addr` is NULL. */
		int type;    /**< Type of access, using `KCSAN_ACCESS` definitions (e.g., read, write, atomic). */
	} access[2];
};

/**
 * @brief Matches an observed KCSAN report with a given expected report structure.
 *
 * This function constructs an expected report string based on the `expect_report` structure
 * and compares it against the lines captured in the `observed` global variable. It handles
 * variations in report formatting, such as the order of access descriptions and optional
 * address information.
 *
 * @param r Pointer to the `expect_report` structure containing the expected report details.
 * @return `true` if the observed report matches the expected information, `false` otherwise.
 * @pre `observed` global structure must contain a complete KCSAN report.
 * @post The function performs a deep comparison of string content to match reports.
 */
__no_kcsan
static bool __report_matches(const struct expect_report *r)
{
	const bool is_assert = (r->access[0].type | r->access[1].type) & KCSAN_ACCESS_ASSERT;
	bool ret = false;
	unsigned long flags;
	typeof(*observed.lines) *expect;
	const char *end;
	char *cur;
	int i;

	/* Doubled-checked locking. */
	if (!report_available())
		return false;

	expect = kmalloc(sizeof(observed.lines), GFP_KERNEL);
	if (WARN_ON(!expect))
		return false;

	/* Block Logic: Generate expected report contents. */

	/* Block Logic: Construct the expected title line of the KCSAN report. */
	cur = expect[0];
	end = &expect[0][sizeof(expect[0]) - 1];
	cur += scnprintf(cur, end - cur, "BUG: KCSAN: %s in ",
			 is_assert ? "assert: race" : "data-race");
	if (r->access[1].fn) {
		char tmp[2][64];
		int cmp;

		/* Invariant: Expected lexographically sorted function names in title. */
		scnprintf(tmp[0], sizeof(tmp[0]), "%pS", r->access[0].fn);
		scnprintf(tmp[1], sizeof(tmp[1]), "%pS", r->access[1].fn);
		cmp = strcmp(tmp[0], tmp[1]);
		cur += scnprintf(cur, end - cur, "%ps / %ps",
				 cmp < 0 ? r->access[0].fn : r->access[1].fn,
				 cmp < 0 ? r->access[1].fn : r->access[0].fn);
	} else {
		scnprintf(cur, end - cur, "%pS", r->access[0].fn);
		/* The exact offset won't match, remove it. */
		cur = strchr(expect[0], '+');
		if (cur)
			*cur = '\0';
	}

	/* Block Logic: Prepare for Access 1 description. */
	cur = expect[1];
	end = &expect[1][sizeof(expect[1]) - 1];
	if (!r->access[1].fn)
		cur += scnprintf(cur, end - cur, "race at unknown origin, with ");

	/* Block Logic: Construct expected access description for both accesses. */
	for (i = 0; i < 2; ++i) {
		const int ty = r->access[i].type;
		const char *const access_type =
			(ty & KCSAN_ACCESS_ASSERT) ?
				      ((ty & KCSAN_ACCESS_WRITE) ?
					       "assert no accesses" :
					       "assert no writes") :
				      ((ty & KCSAN_ACCESS_WRITE) ?
					       ((ty & KCSAN_ACCESS_COMPOUND) ?
							"read-write" :
							"write") :
					       "read");
		const bool is_atomic = (ty & KCSAN_ACCESS_ATOMIC);
		const bool is_scoped = (ty & KCSAN_ACCESS_SCOPED);
		const char *const access_type_aux =
				(is_atomic && is_scoped)	? " (marked, reordered)"
				: (is_atomic			? " (marked)"
				   : (is_scoped			? " (reordered)" : ""));

		if (i == 1) {
			/* Block Logic: Prepare for Access 2 description. */
			cur = expect[2];
			end = &expect[2][sizeof(expect[2]) - 1];

			if (!r->access[1].fn) {
				/* Dummy string if no second access is available. */
				strcpy(cur, "<none>");
				break;
			}
		}

		cur += scnprintf(cur, end - cur, "%s%s to ", access_type,
				 access_type_aux);

		if (r->access[i].addr) /* Address is optional. */
			cur += scnprintf(cur, end - cur, "0x%px of %zu bytes",
					 r->access[i].addr, r->access[i].size);
	}

	spin_lock_irqsave(&observed.lock, flags);
	if (!report_available())
		goto out; /* A new report is being captured. */

	/* Block Logic: Finally match expected output to what we actually observed. */
	// Check title line.
	ret = strstr(observed.lines[0], expect[0]) &&
	      /* Access info may appear in any order in lines 1 and 2. */
	      ((strstr(observed.lines[1], expect[1]) &&
		strstr(observed.lines[2], expect[2])) ||
	       (strstr(observed.lines[1], expect[2]) &&
		strstr(observed.lines[2], expect[1])));
out:
	spin_unlock_irqrestore(&observed.lock, flags);
	kfree(expect);
	return ret;
}

/**
 * @brief Helper function to set or clear the `KCSAN_ACCESS_SCOPED` flag for accesses in an `expect_report`.
 *
 * This is used to test scenarios involving scoped assertions, where the `(reordered)` annotation
 * might appear in the KCSAN report depending on memory reordering.
 *
 * @param r Pointer to the `expect_report` structure to modify.
 * @param accesses Bitmask indicating which accesses (0 for first, 1 for second) should have the SCOPED flag set.
 *                 Values: 0 (none), 1 (first), 2 (second), 3 (both).
 * @return A constant pointer to the modified `expect_report` structure.
 */
static __always_inline const struct expect_report *
__report_set_scoped(struct expect_report *r, int accesses)
{
	BUILD_BUG_ON(accesses > 3); // Ensure accesses is a valid bitmask for 2 accesses.

	if (accesses & 1) // Check if the first access should be scoped.
		r->access[0].type |= KCSAN_ACCESS_SCOPED;
	else
		r->access[0].type &= ~KCSAN_ACCESS_SCOPED;

	if (accesses & 2) // Check if the second access should be scoped.
		r->access[1].type |= KCSAN_ACCESS_SCOPED;
	else
		r->access[1].type &= ~KCSAN_ACCESS_SCOPED;

	return r;
}

/**
 * @brief Checks if an observed KCSAN report matches the expected report, considering all possible memory reordering annotations.
 *
 * This function wraps `__report_matches` and attempts to match the report against variations
 * where either none, the first, the second, or both accesses are marked as `(reordered)`
 * due to compiler/CPU memory reordering. This is relevant when `CONFIG_KCSAN_WEAK_MEMORY` is enabled.
 *
 * @param r Pointer to the `expect_report` structure with expected report details.
 * @return `true` if any of the reordered report variations match, `false` otherwise.
 */
__no_kcsan
static bool report_matches_any_reordered(struct expect_report *r)
{
	return __report_matches(__report_set_scoped(r, 0)) || // No accesses reordered
	       __report_matches(__report_set_scoped(r, 1)) || // First access reordered
	       __report_matches(__report_set_scoped(r, 2)) || // Second access reordered
	       __report_matches(__report_set_scoped(r, 3));   // Both accesses reordered
}

/**
 * @brief Macro to determine the appropriate report matching function based on `CONFIG_KCSAN_WEAK_MEMORY`.
 *
 * If `CONFIG_KCSAN_WEAK_MEMORY` is enabled, compiler/CPU reordering is possible, and thus
 * the report matching should account for `(reordered)` annotations in the report.
 * Otherwise, a direct match is sufficient.
 */
#ifdef CONFIG_KCSAN_WEAK_MEMORY
#define report_matches report_matches_any_reordered
#else
#define report_matches __report_matches
#endif

/* ===== Test kernels ===== */

/**
 * @brief A volatile variable used as a sink for values to prevent compiler optimizations.
 *
 * `test_sink` is primarily used in `sink_value()` to consume values
 * from reads or computations, ensuring that the compiler does not optimize away
 * these operations, which is crucial for testing KCSAN's instrumentation.
 */
static long test_sink;

/**
 * @brief The primary test variable for various data race scenarios.
 *
 * This variable is the target of most memory accesses within the test "kernels"
 * and is subject to concurrent reads and writes to expose data races.
 */
static long test_var;

/**
 * @brief A large array used for testing KCSAN's handling of accesses spanning multiple watchpoint slots.
 *
 * The size of `test_array` is chosen to be large enough (3 times `PAGE_SIZE`)
 * to ensure that memory accesses to different parts of the array potentially trigger
 * distinct KCSAN watchpoints, verifying KCSAN's scalability and accuracy for larger data structures.
 */
static long test_array[3 * PAGE_SIZE / sizeof(long)];

/**
 * @brief A test structure containing multiple long values, used for testing struct-level accesses.
 *
 * This structure allows for testing how KCSAN instruments and reports races
 * on parts of a structure, or on the entire structure as a single access.
 */
static struct {
	long val[8]; /**< Array of long values within the test structure. */
} test_struct;

/**
 * @brief A variable specifically marked with `__data_racy` to test KCSAN's handling of such annotations.
 *
 * The `__data_racy` attribute indicates that the variable is intentionally accessed
 * in a racy manner, and KCSAN should generally ignore such races unless specific
 * conditions (like non-atomic writes to it) are met.
 */
static long __data_racy test_data_racy;

/**
 * @brief A seqlock instance used for testing KCSAN's interaction with seqlocks.
 *
 * Seqlocks are synchronization primitives that allow for concurrent readers
 * and a single writer without requiring a lock for readers, but require readers
 * to re-check a sequence number. KCSAN should not report races within properly
 * guarded seqlock critical sections.
 */
static DEFINE_SEQLOCK(test_seqlock);

/**
 * @brief A spinlock instance used for testing KCSAN's interaction with spinlocks.
 *
 * Spinlocks are basic locking primitives that prevent multiple threads from
 * simultaneously entering a critical section. KCSAN should recognize and
 * not report races on data protected by spinlocks.
 */
static DEFINE_SPINLOCK(test_spinlock);

/**
 * @brief A mutex instance used for testing KCSAN's interaction with mutexes.
 *
 * Mutexes are synchronization primitives that provide mutual exclusion,
 * allowing only one thread to hold the lock at a time. KCSAN should respect
 * mutex-protected critical sections and not report races within them.
 */
static DEFINE_MUTEX(test_mutex);

/**
 * @brief Helper function to prevent compiler optimizations from removing reads and writes.
 *
 * This function acts as a "sink" for a given value, writing it to `test_sink`.
 * By writing to a `WRITE_ONCE` variable, it ensures that the compiler
 * does not optimize out prior reads or subsequent writes that feed into `v`,
 * which is essential for accurate KCSAN instrumentation testing.
 *
 * @param v The value to "sink" (write to `test_sink`).
 */
__no_kcsan
static noinline void sink_value(long v) { WRITE_ONCE(test_sink, v); }

/**
 * @brief Generates a controlled delay and performs non-racy memory accesses.
 *
 * This function introduces a delay by looping a specified number of times,
 * performing benign memory accesses (`READ_ONCE(test_sink)`) that interact
 * with the KCSAN runtime but are not expected to cause data races.
 * It's useful for consuming CPU cycles without introducing false positives.
 *
 * @param iter The number of iterations to loop, controlling the delay duration.
 */
static noinline void test_delay(int iter)
{
	while (iter--)
		sink_value(READ_ONCE(test_sink));
}

/**
 * @brief Performs a non-atomic read operation on `test_var`.
 *
 * This function reads the value of `test_var` and passes it to `sink_value`
 * to prevent the compiler from optimizing away the read. It represents a plain
 * read access that KCSAN should instrument and detect races on.
 */
static noinline void test_kernel_read(void) { sink_value(test_var); }

/**
 * @brief Performs a non-atomic write operation on `test_var`.
 *
 * This function writes a new value to `test_var`, derived from `test_sink`.
 * It represents a plain write access that KCSAN should instrument and
 * detect races on when executed concurrently with other conflicting accesses.
 */
static noinline void test_kernel_write(void)
{
	test_var = READ_ONCE_NOCHECK(test_sink) + 1;
}

/**
 * @brief Performs a non-atomic write to `test_var` that does not change its value (if it's already 42).
 *
 * This function is used to test KCSAN's behavior when `CONFIG_KCSAN_REPORT_VALUE_CHANGE_ONLY`
 * is enabled. If the value of `test_var` is already 42, this write will not
 * trigger a value change, and KCSAN might (or might not, depending on configuration)
 * suppress a report for a race involving this access.
 */
static noinline void test_kernel_write_nochange(void) { test_var = 42; }

/**
 * @brief Performs a non-atomic write to `test_var` that does not change its value,
 * and is intended to be excluded by a value-change exception filter (e.g., for RCU).
 *
 * This function is specifically designed to test that certain write operations,
 * even if they don't change the value, are still reported by KCSAN when
 * `CONFIG_KCSAN_REPORT_VALUE_CHANGE_ONLY` is active, because they are
 * explicitly excluded from the value-change suppression filter. This is common
 * for RCU-related writes that logically imply a modification even if the value stays the same.
 */
static noinline void test_kernel_write_nochange_rcu(void) { test_var = 42; }

/**
 * @brief Performs an atomic read operation on `test_var`.
 *
 * This function uses `READ_ONCE` to perform an atomic read of `test_var`,
 * passing the value to `sink_value` to prevent optimization. This is
 * intended to represent an atomic access that KCSAN should not report
 * as a data race when paired with other atomic accesses.
 */
static noinline void test_kernel_read_atomic(void)
{
	sink_value(READ_ONCE(test_var));
}

/**
 * @brief Performs an atomic write operation on `test_var`.
 *
 * This function uses `WRITE_ONCE` to perform an atomic write to `test_var`,
 * setting its value based on `test_sink`. This is intended to represent an
 * atomic access that KCSAN should not report as a data race when paired
 * with other atomic accesses.
 */
static noinline void test_kernel_write_atomic(void)
{
	WRITE_ONCE(test_var, READ_ONCE_NOCHECK(test_sink) + 1);
}

/**
 * @brief Performs an atomic Read-Modify-Write (RMW) operation on `test_var` using a GCC builtin.
 *
 * This function uses `__atomic_fetch_add` to atomically increment `test_var`.
 * It is used to test KCSAN's handling of atomic RMW operations, particularly
 * in scenarios where it might race with plain (non-atomic) accesses.
 *
 * @remark The use of `__atomic_fetch_add` with `__ATOMIC_RELAXED` provides a basic
 *         atomic operation, suitable for testing KCSAN's recognition of atomicity.
 */
static noinline void test_kernel_atomic_rmw(void)
{
	/* Use builtin, so we can set up the "bad" atomic/non-atomic scenario. */
	__atomic_fetch_add(&test_var, 1, __ATOMIC_RELAXED);
}

/**
 * @brief Performs a write to `test_var` that is explicitly uninstrumented by KCSAN.
 *
 * This function increments `test_var` but is marked with `__no_kcsan`,
 * meaning KCSAN will not instrument this access. When this function races
 * with an instrumented access, KCSAN should report a "race at unknown origin"
 * if `CONFIG_KCSAN_REPORT_RACE_UNKNOWN_ORIGIN` is enabled.
 */
__no_kcsan
static noinline void test_kernel_write_uninstrumented(void) { test_var++; }

/**
 * @brief Performs a write to `test_var` explicitly marked as a data race using `data_race()`.
 *
 * This function uses the `data_race()` macro, which signals to KCSAN that the
 * access is an intended data race and should typically not be reported,
 * allowing tests to verify KCSAN's suppression of designated races.
 */
static noinline void test_kernel_data_race(void) { data_race(test_var++); }

/**
 * @brief Performs a write to a `__data_racy` variable.
 *
 * This function increments `test_data_racy`, a variable marked with the
 * `__data_racy` attribute. This attribute indicates that races on this variable
 * are expected and typically ignored by KCSAN, allowing for testing specific
 * suppression mechanisms.
 */
static noinline void test_kernel_data_racy_qualifier(void) { test_data_racy++; }

/**
 * @brief Performs an assertion that `test_var` has an exclusive writer.
 *
 * This function uses `ASSERT_EXCLUSIVE_WRITER` to dynamically assert that
 * no other concurrent accesses are writing to or reading from `test_var`
 * while this assertion holds. It is used to test KCSAN's ability to
 * detect violations of such assertions.
 */
static noinline void test_kernel_assert_writer(void)
{
	ASSERT_EXCLUSIVE_WRITER(test_var);
}

/**
 * @brief Performs an assertion that `test_var` has exclusive access (no concurrent reads or writes).
 *
 * This function uses `ASSERT_EXCLUSIVE_ACCESS` to dynamically assert that
 * no other concurrent accesses (reads or writes) are occurring on `test_var`.
 * It is used to test KCSAN's ability to detect violations of strict exclusive access requirements.
 */
static noinline void test_kernel_assert_access(void)
{
	ASSERT_EXCLUSIVE_ACCESS(test_var);
}

/**
 * @brief Bitmask used to define specific bits for testing bit-level assertions.
 */
#define TEST_CHANGE_BITS 0xff00ff00

/**
 * @brief Performs a bitwise XOR operation on `test_var` with `TEST_CHANGE_BITS`.
 *
 * This function modifies `test_var` by XORing it with a predefined bitmask.
 * It's designed to either perform an explicitly atomic-like operation (if
 * `CONFIG_KCSAN_IGNORE_ATOMICS` is enabled) or a regular non-atomic write
 * (via `WRITE_ONCE`) that changes specific bits of `test_var`. This is crucial
 * for testing KCSAN's bit-level assertion capabilities.
 */
static noinline void test_kernel_change_bits(void)
{
	if (IS_ENABLED(CONFIG_KCSAN_IGNORE_ATOMICS)) {
		/*
		 * Block Logic: If KCSAN is configured to ignore atomics, this simulates an atomic
		 * bitwise XOR to avoid false positives for "race at unknown origin."
		 */
		kcsan_nestable_atomic_begin(); // Temporarily disable KCSAN reporting for nested atomic-like operations.
		test_var ^= TEST_CHANGE_BITS;
		kcsan_nestable_atomic_end();   // Re-enable KCSAN reporting.
	} else
		// Perform a non-atomic bitwise XOR, instrumented by KCSAN.
		WRITE_ONCE(test_var, READ_ONCE(test_var) ^ TEST_CHANGE_BITS);
}

/**
 * @brief Asserts that only `TEST_CHANGE_BITS` bits of `test_var` are being modified by concurrent writes.
 *
 * This function uses `ASSERT_EXCLUSIVE_BITS` to assert that any concurrent
 * writes to `test_var` only affect the bits specified by `TEST_CHANGE_BITS`.
 * It's used to test KCSAN's ability to detect bit-level data races and
 * assertion violations.
 */
static noinline void test_kernel_assert_bits_change(void)
{
	ASSERT_EXCLUSIVE_BITS(test_var, TEST_CHANGE_BITS);
}

/**
 * @brief Asserts that no bits other than `TEST_CHANGE_BITS` of `test_var` are being modified.
 *
 * This function uses `ASSERT_EXCLUSIVE_BITS` to assert that any concurrent
 * writes to `test_var` do not affect the bits *not* specified by `TEST_CHANGE_BITS`
 * (i.e., `~TEST_CHANGE_BITS`). It's used to test KCSAN's precise bit-level
 * data race detection and assertion capabilities.
 */
static noinline void test_kernel_assert_bits_nochange(void)
{
	ASSERT_EXCLUSIVE_BITS(test_var, ~TEST_CHANGE_BITS);
}

/**
 * @brief Helper function to perform some unrelated accesses within a test scope.
 *
 * This function introduces a local variable `x` and performs some non-racy
 * accesses to it, intended to simulate operations that might occur within
 * the scope of a KCSAN scoped assertion without directly causing races
 * on the asserted variable. The goal is to verify that KCSAN reports point
 * to the start of the scoped assertion, not these unrelated accesses.
 */
static noinline void test_enter_scope(void)
{
	int x = 0;

	/* Unrelated accesses to scoped assert. */
	READ_ONCE(test_sink);       // Read to a sink to prevent optimization.
	kcsan_check_read(&x, sizeof(x)); // KCSAN check on a local variable.
}

/**
 * @brief Performs a scoped assertion that `test_var` has an exclusive writer within a defined scope.
 *
 * This function uses `ASSERT_EXCLUSIVE_WRITER_SCOPED` to establish a scope
 * within which `test_var` is expected to have an exclusive writer.
 * It then calls `test_enter_scope()` to introduce some activity within
 * this asserted scope. KCSAN reports for violations within this scope should
 * point to the `ASSERT_EXCLUSIVE_WRITER_SCOPED` call site.
 */
static noinline void test_kernel_assert_writer_scoped(void)
{
	ASSERT_EXCLUSIVE_WRITER_SCOPED(test_var); // Start of the scoped assertion.
	test_enter_scope();                       // Activity within the asserted scope.
}

/**
 * @brief Performs a scoped assertion that `test_var` has exclusive access within a defined scope.
 *
 * Similar to `test_kernel_assert_writer_scoped`, this function uses
 * `ASSERT_EXCLUSIVE_ACCESS_SCOPED` to establish a scope for exclusive access
 * (no concurrent reads or writes) on `test_var`. It also calls
 * `test_enter_scope()` to perform activity within the asserted scope,
 * verifying that KCSAN reports correctly attribute violations to the
 * start of the scoped assertion.
 */
static noinline void test_kernel_assert_access_scoped(void)
{
	ASSERT_EXCLUSIVE_ACCESS_SCOPED(test_var); // Start of the scoped assertion.
	test_enter_scope();                       // Activity within the asserted scope.
}

/**
 * @brief Performs Read-Modify-Write (RMW) operations across a large array.
 *
 * This function iterates over `test_array` and increments each element.
 * It is designed to create a large number of concurrent RMW operations
 * on different memory locations, stressing KCSAN's ability to track
 * and report races across a wide range of addresses, particularly when
 * `test_array` spans multiple KCSAN watchpoint slots.
 */
static noinline void test_kernel_rmw_array(void)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(test_array); ++i)
		test_array[i]++;
}

/**
 * @brief Performs a write to an entire structure, then a sub-part.
 *
 * This function first performs an instrumented write to the entire `test_struct`.
 * It then temporarily disables KCSAN to induce a value change on a member
 * (`test_struct.val[3]`) without KCSAN reporting a race on this specific
 * modification, simulating a scenario where a large access might cover a smaller,
 * intentionally uninstrumented change.
 */
static noinline void test_kernel_write_struct(void)
{
	kcsan_check_write(&test_struct, sizeof(test_struct)); // Instrumented write to the entire structure.
	kcsan_disable_current();                              // Disable KCSAN to avoid reporting on the next line.
	test_struct.val[3]++; /* induce value change */       // Intentional value change on a sub-part.
	kcsan_enable_current();                               // Re-enable KCSAN.
}

/**
 * @brief Performs a write to a specific part of `test_struct`.
 *
 * This function writes a constant value to `test_struct.val[3]`.
 * It is used to create a race scenario with `test_kernel_write_struct`,
 * where one access is to the entire structure and another is to a sub-part,
 * testing KCSAN's precision in reporting overlapping accesses.
 */
static noinline void test_kernel_write_struct_part(void)
{
	test_struct.val[3] = 42;
}

/**
 * @brief Performs a KCSAN check on a zero-sized read access to a struct member.
 *
 * This function calls `kcsan_check_read` with a size of 0 for `test_struct.val[3]`.
 * Zero-sized accesses should generally not cause data race reports, and this
 * function is used to verify KCSAN's behavior in such edge cases.
 */
static noinline void test_kernel_read_struct_zero_size(void)
{
	kcsan_check_read(&test_struct.val[3], 0);
}

/**
 * @brief Reads the `jiffies` global variable.
 *
 * This function reads the `jiffies` variable, which is declared as `volatile`.
 * This test is used to verify that KCSAN correctly handles accesses to `volatile`
 * variables and does not generate false positives, as `volatile` accesses
 * have specific memory ordering semantics.
 */
static noinline void test_kernel_jiffies_reader(void)
{
	sink_value((long)jiffies);
}

/**
 * @brief Performs a read operation within a seqlock-protected critical section.
 *
 * This function demonstrates the typical reader-side pattern for a `seqlock`,
 * ensuring consistency by re-checking the sequence number if a write occurred
 * during the read. KCSAN should not report data races for accesses to
 * `test_var` when properly guarded by `test_seqlock`.
 */
static noinline void test_kernel_seqlock_reader(void)
{
	unsigned int seq;

	do {
		seq = read_seqbegin(&test_seqlock); // Start of seqlock read-side critical section.
		sink_value(test_var);               // Access to `test_var` protected by seqlock.
	} while (read_seqretry(&test_seqlock, seq)); // Check if a write interfered during the read.
}

/**
 * @brief Performs a write operation within a seqlock-protected critical section.
 *
 * This function demonstrates the typical writer-side pattern for a `seqlock`,
 * acquiring the write lock, modifying `test_var`, and then releasing the lock.
 * KCSAN should recognize `seqlock` as a synchronization primitive and not
 * report data races when accesses are correctly protected.
 */
static noinline void test_kernel_seqlock_writer(void)
{
	unsigned long flags;

	write_seqlock_irqsave(&test_seqlock, flags); // Acquire seqlock write lock.
	test_var++;                                  // Modify `test_var` within the critical section.
	write_sequnlock_irqrestore(&test_seqlock, flags); // Release seqlock write lock.
}

/**
 * @brief Performs an atomic load using a GCC atomic builtin.
 *
 * This function uses `__atomic_load_n` to atomically read `test_var`.
 * It is intended to test that KCSAN correctly recognizes GCC's atomic builtins
 * as atomic operations and does not report data races when these are used
 * concurrently.
 *
 * @remark This test explicitly confirms that KCSAN treats builtin atomics as truly atomic,
 *         preventing false positives for well-synchronized code.
 */
static noinline void test_kernel_atomic_builtins(void)
{
	/*
	 * Generate concurrent accesses, expecting no reports, ensuring KCSAN
	 * treats builtin atomics as actually atomic.
	 */
	__atomic_load_n(&test_var, __ATOMIC_RELAXED);
}

/**
 * @brief Performs a bitwise XOR operation on `test_var` within a KCSAN-nestable atomic section.
 *
 * This function modifies `test_var` using a bitwise XOR operation. It is enclosed
 * within `kcsan_nestable_atomic_begin()` and `kcsan_nestable_atomic_end()`,
 * signaling to KCSAN that this sequence of operations should be treated as a
 * single atomic-like unit, preventing data race reports even if the underlying
 * operation is not a hardware atomic instruction. This is used to test KCSAN's
 * ability to suppress races in designated "atomic" blocks.
 */
static noinline void test_kernel_xor_1bit(void)
{
	/* Do not report data races between the read-writes. */
	kcsan_nestable_atomic_begin(); // Marks the start of a KCSAN-defined atomic section.
	test_var ^= 0x10000;          // Non-atomic read-modify-write operation.
	kcsan_nestable_atomic_end();    // Marks the end of the atomic section.
}

/**
 * @brief Macro to generate test kernel functions that simulate locked access patterns.
 *
 * This macro creates a function `test_kernel_##name` that attempts to acquire a lock
 * (represented by `flag`), performs a series of operations (incrementing `test_var`),
 * and then releases the lock. It's designed to test KCSAN's ability to recognize
 * synchronization primitives and memory ordering semantics.
 *
 * @param name The suffix for the generated function name (e.g., `with_memorder`).
 * @param acquire The code snippet for acquiring the lock.
 * @param release The code snippet for releasing the lock.
 * @remark The `flag` variable (`test_struct.val[0]`) acts as a simple lock mechanism.
 */
#define TEST_KERNEL_LOCKED(name, acquire, release)		\
	static noinline void test_kernel_##name(void)		\
	{							\
		long *flag = &test_struct.val[0];		\
		long v = 0;					\
		if (!(acquire))					\
			return;					\
		while (v++ < 100) {				\
			test_var++;				\
			barrier();				\
		}						\
		release;					\
		test_delay(10);					\
	}

/**
 * @brief Test kernel function simulating locked access with correct memory ordering.
 *
 * This function uses `cmpxchg_acquire` for acquiring the lock and `smp_store_release`
 * for releasing it, ensuring proper memory ordering. KCSAN should not report races
 * for accesses to `test_var` within this critical section.
 */
TEST_KERNEL_LOCKED(with_memorder,
		   cmpxchg_acquire(flag, 0, 1) == 0,
		   smp_store_release(flag, 0));

/**
 * @brief Test kernel function simulating locked access with incorrect (relaxed) memory ordering.
 *
 * This function uses `cmpxchg_relaxed` for acquiring the lock and `WRITE_ONCE`
 * for releasing it, which typically implies weaker memory ordering. KCSAN is expected
 * to report races in this scenario if `CONFIG_KCSAN_WEAK_MEMORY` is enabled.
 */
TEST_KERNEL_LOCKED(wrong_memorder,
		   cmpxchg_relaxed(flag, 0, 1) == 0,
		   WRITE_ONCE(*flag, 0));

/**
 * @brief Test kernel function simulating locked access with atomic builtins and correct memory ordering.
 *
 * This function uses `__atomic_compare_exchange_n` with `__ATOMIC_ACQUIRE` and
 * `__atomic_store_n` with `__ATOMIC_RELEASE` for acquiring and releasing the lock,
 * demonstrating correct memory ordering with atomic builtins. KCSAN should not
 * report races here.
 */
TEST_KERNEL_LOCKED(atomic_builtin_with_memorder,
		   __atomic_compare_exchange_n(flag, &v, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED),
		   __atomic_store_n(flag, 0, __ATOMIC_RELEASE));

/**
 * @brief Test kernel function simulating locked access with atomic builtins and incorrect (relaxed) memory ordering.
 *
 * This function uses `__atomic_compare_exchange_n` and `__atomic_store_n`
 * both with `__ATOMIC_RELAXED` ordering. KCSAN is expected to report races in
 * this scenario if `CONFIG_KCSAN_WEAK_MEMORY` is enabled, as the relaxed
 * ordering might not guarantee sufficient synchronization.
 */
TEST_KERNEL_LOCKED(atomic_builtin_wrong_memorder,
		   __atomic_compare_exchange_n(flag, &v, 1, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED),
		   __atomic_store_n(flag, 0, __ATOMIC_RELAXED));

/* ===== Test cases ===== */

/*
 * Tests that various barriers have the expected effect on internal state. Not
 * exhaustive on atomic_t operations. Unlike the selftest, also checks for
 * too-strict barrier instrumentation; these can be tolerated, because it does
 * not cause false positives, but at least we should be aware of such cases.
 */
static void test_barrier_nothreads(struct kunit *test)
{
#ifdef CONFIG_KCSAN_WEAK_MEMORY
	struct kcsan_scoped_access *reorder_access = &current->kcsan_ctx.reorder_access;
#else
	struct kcsan_scoped_access *reorder_access = NULL;
#endif
	arch_spinlock_t arch_spinlock = __ARCH_SPIN_LOCK_UNLOCKED;
	atomic_t dummy = ATOMIC_INIT(0);

	KCSAN_TEST_REQUIRES(test, reorder_access != NULL);
	KCSAN_TEST_REQUIRES(test, IS_ENABLED(CONFIG_SMP));

#define __KCSAN_EXPECT_BARRIER(access_type, barrier, order_before, name)			\
	do {											\
		reorder_access->type = (access_type) | KCSAN_ACCESS_SCOPED;			\
		reorder_access->size = sizeof(test_var);					\
		barrier;									\
		KUNIT_EXPECT_EQ_MSG(test, reorder_access->size,					\
				    order_before ? 0 : sizeof(test_var),			\
				    "improperly instrumented type=(" #access_type "): " name);	\
	} while (0)
#define KCSAN_EXPECT_READ_BARRIER(b, o)  __KCSAN_EXPECT_BARRIER(0, b, o, #b)
#define KCSAN_EXPECT_WRITE_BARRIER(b, o) __KCSAN_EXPECT_BARRIER(KCSAN_ACCESS_WRITE, b, o, #b)
#define KCSAN_EXPECT_RW_BARRIER(b, o)    __KCSAN_EXPECT_BARRIER(KCSAN_ACCESS_COMPOUND | KCSAN_ACCESS_WRITE, b, o, #b)

	/*
	 * Lockdep initialization can strengthen certain locking operations due
	 * to calling into instrumented files; "warm up" our locks.
	 */
	spin_lock(&test_spinlock);
	spin_unlock(&test_spinlock);
	mutex_lock(&test_mutex);
	mutex_unlock(&test_mutex);

	/* Force creating a valid entry in reorder_access first. */
	test_var = 0;
	while (test_var++ < 1000000 && reorder_access->size != sizeof(test_var))
		__kcsan_check_read(&test_var, sizeof(test_var));
	KUNIT_ASSERT_EQ(test, reorder_access->size, sizeof(test_var));

	kcsan_nestable_atomic_begin(); /* No watchpoints in called functions. */

	KCSAN_EXPECT_READ_BARRIER(mb(), true);
	KCSAN_EXPECT_READ_BARRIER(wmb(), false);
	KCSAN_EXPECT_READ_BARRIER(rmb(), true);
	KCSAN_EXPECT_READ_BARRIER(smp_mb(), true);
	KCSAN_EXPECT_READ_BARRIER(smp_wmb(), false);
	KCSAN_EXPECT_READ_BARRIER(smp_rmb(), true);
	KCSAN_EXPECT_READ_BARRIER(dma_wmb(), false);
	KCSAN_EXPECT_READ_BARRIER(dma_rmb(), true);
	KCSAN_EXPECT_READ_BARRIER(smp_mb__before_atomic(), true);
	KCSAN_EXPECT_READ_BARRIER(smp_mb__after_atomic(), true);
	KCSAN_EXPECT_READ_BARRIER(smp_mb__after_spinlock(), true);
	KCSAN_EXPECT_READ_BARRIER(smp_store_mb(test_var, 0), true);
	KCSAN_EXPECT_READ_BARRIER(smp_load_acquire(&test_var), false);
	KCSAN_EXPECT_READ_BARRIER(smp_store_release(&test_var, 0), true);
	KCSAN_EXPECT_READ_BARRIER(xchg(&test_var, 0), true);
	KCSAN_EXPECT_READ_BARRIER(xchg_release(&test_var, 0), true);
	KCSAN_EXPECT_READ_BARRIER(xchg_relaxed(&test_var, 0), false);
	KCSAN_EXPECT_READ_BARRIER(cmpxchg(&test_var, 0,  0), true);
	KCSAN_EXPECT_READ_BARRIER(cmpxchg_release(&test_var, 0,  0), true);
	KCSAN_EXPECT_READ_BARRIER(cmpxchg_relaxed(&test_var, 0,  0), false);
	KCSAN_EXPECT_READ_BARRIER(atomic_read(&dummy), false);
	KCSAN_EXPECT_READ_BARRIER(atomic_read_acquire(&dummy), false);
	KCSAN_EXPECT_READ_BARRIER(atomic_set(&dummy, 0), false);
	KCSAN_EXPECT_READ_BARRIER(atomic_set_release(&dummy, 0), true);
	KCSAN_EXPECT_READ_BARRIER(atomic_add(1, &dummy), false);
	KCSAN_EXPECT_READ_BARRIER(atomic_add_return(1, &dummy), true);
	KCSAN_EXPECT_READ_BARRIER(atomic_add_return_acquire(1, &dummy), false);
	KCSAN_EXPECT_READ_BARRIER(atomic_add_return_release(1, &dummy), true);
	KCSAN_EXPECT_READ_BARRIER(atomic_add_return_relaxed(1, &dummy), false);
	KCSAN_EXPECT_READ_BARRIER(atomic_fetch_add(1, &dummy), true);
	KCSAN_EXPECT_READ_BARRIER(atomic_fetch_add_acquire(1, &dummy), false);
	KCSAN_EXPECT_READ_BARRIER(atomic_fetch_add_release(1, &dummy), true);
	KCSAN_EXPECT_READ_BARRIER(atomic_fetch_add_relaxed(1, &dummy), false);
	KCSAN_EXPECT_READ_BARRIER(test_and_set_bit(0, &test_var), true);
	KCSAN_EXPECT_READ_BARRIER(test_and_clear_bit(0, &test_var), true);
	KCSAN_EXPECT_READ_BARRIER(test_and_change_bit(0, &test_var), true);
	KCSAN_EXPECT_READ_BARRIER(clear_bit_unlock(0, &test_var), true);
	KCSAN_EXPECT_READ_BARRIER(__clear_bit_unlock(0, &test_var), true);
	KCSAN_EXPECT_READ_BARRIER(arch_spin_lock(&arch_spinlock), false);
	KCSAN_EXPECT_READ_BARRIER(arch_spin_unlock(&arch_spinlock), true);
	KCSAN_EXPECT_READ_BARRIER(spin_lock(&test_spinlock), false);
	KCSAN_EXPECT_READ_BARRIER(spin_unlock(&test_spinlock), true);
	KCSAN_EXPECT_READ_BARRIER(mutex_lock(&test_mutex), false);
	KCSAN_EXPECT_READ_BARRIER(mutex_unlock(&test_mutex), true);

	KCSAN_EXPECT_WRITE_BARRIER(mb(), true);
	KCSAN_EXPECT_WRITE_BARRIER(wmb(), true);
	KCSAN_EXPECT_WRITE_BARRIER(rmb(), false);
	KCSAN_EXPECT_WRITE_BARRIER(smp_mb(), true);
	KCSAN_EXPECT_WRITE_BARRIER(smp_wmb(), true);
	KCSAN_EXPECT_WRITE_BARRIER(smp_rmb(), false);
	KCSAN_EXPECT_WRITE_BARRIER(dma_wmb(), true);
	KCSAN_EXPECT_WRITE_BARRIER(dma_rmb(), false);
	KCSAN_EXPECT_WRITE_BARRIER(smp_mb__before_atomic(), true);
	KCSAN_EXPECT_WRITE_BARRIER(smp_mb__after_atomic(), true);
	KCSAN_EXPECT_WRITE_BARRIER(smp_mb__after_spinlock(), true);
	KCSAN_EXPECT_WRITE_BARRIER(smp_store_mb(test_var, 0), true);
	KCSAN_EXPECT_WRITE_BARRIER(smp_load_acquire(&test_var), false);
	KCSAN_EXPECT_WRITE_BARRIER(smp_store_release(&test_var, 0), true);
	KCSAN_EXPECT_WRITE_BARRIER(xchg(&test_var, 0), true);
	KCSAN_EXPECT_WRITE_BARRIER(xchg_release(&test_var, 0), true);
	KCSAN_EXPECT_WRITE_BARRIER(xchg_relaxed(&test_var, 0), false);
	KCSAN_EXPECT_WRITE_BARRIER(cmpxchg(&test_var, 0,  0), true);
	KCSAN_EXPECT_WRITE_BARRIER(cmpxchg_release(&test_var, 0,  0), true);
	KCSAN_EXPECT_WRITE_BARRIER(cmpxchg_relaxed(&test_var, 0,  0), false);
	KCSAN_EXPECT_WRITE_BARRIER(atomic_read(&dummy), false);
	KCSAN_EXPECT_WRITE_BARRIER(atomic_read_acquire(&dummy), false);
	KCSAN_EXPECT_WRITE_BARRIER(atomic_set(&dummy, 0), false);
	KCSAN_EXPECT_WRITE_BARRIER(atomic_set_release(&dummy, 0), true);
	KCSAN_EXPECT_WRITE_BARRIER(atomic_add(1, &dummy), false);
	KCSAN_EXPECT_WRITE_BARRIER(atomic_add_return(1, &dummy), true);
	KCSAN_EXPECT_WRITE_BARRIER(atomic_add_return_acquire(1, &dummy), false);
	KCSAN_EXPECT_WRITE_BARRIER(atomic_add_return_release(1, &dummy), true);
	KCSAN_EXPECT_WRITE_BARRIER(atomic_add_return_relaxed(1, &dummy), false);
	KCSAN_EXPECT_WRITE_BARRIER(atomic_fetch_add(1, &dummy), true);
	KCSAN_EXPECT_WRITE_BARRIER(atomic_fetch_add_acquire(1, &dummy), false);
	KCSAN_EXPECT_WRITE_BARRIER(atomic_fetch_add_release(1, &dummy), true);
	KCSAN_EXPECT_WRITE_BARRIER(atomic_fetch_add_relaxed(1, &dummy), false);
	KCSAN_EXPECT_WRITE_BARRIER(test_and_set_bit(0, &test_var), true);
	KCSAN_EXPECT_WRITE_BARRIER(test_and_clear_bit(0, &test_var), true);
	KCSAN_EXPECT_WRITE_BARRIER(test_and_change_bit(0, &test_var), true);
	KCSAN_EXPECT_WRITE_BARRIER(clear_bit_unlock(0, &test_var), true);
	KCSAN_EXPECT_WRITE_BARRIER(__clear_bit_unlock(0, &test_var), true);
	KCSAN_EXPECT_WRITE_BARRIER(arch_spin_lock(&arch_spinlock), false);
	KCSAN_EXPECT_WRITE_BARRIER(arch_spin_unlock(&arch_spinlock), true);
	KCSAN_EXPECT_WRITE_BARRIER(spin_lock(&test_spinlock), false);
	KCSAN_EXPECT_WRITE_BARRIER(spin_unlock(&test_spinlock), true);
	KCSAN_EXPECT_WRITE_BARRIER(mutex_lock(&test_mutex), false);
	KCSAN_EXPECT_WRITE_BARRIER(mutex_unlock(&test_mutex), true);

	KCSAN_EXPECT_RW_BARRIER(mb(), true);
	KCSAN_EXPECT_RW_BARRIER(wmb(), true);
	KCSAN_EXPECT_RW_BARRIER(rmb(), true);
	KCSAN_EXPECT_RW_BARRIER(smp_mb(), true);
	KCSAN_EXPECT_RW_BARRIER(smp_wmb(), true);
	KCSAN_EXPECT_RW_BARRIER(smp_rmb(), true);
	KCSAN_EXPECT_RW_BARRIER(dma_wmb(), true);
	KCSAN_EXPECT_RW_BARRIER(dma_rmb(), true);
	KCSAN_EXPECT_RW_BARRIER(smp_mb__before_atomic(), true);
	KCSAN_EXPECT_RW_BARRIER(smp_mb__after_atomic(), true);
	KCSAN_EXPECT_RW_BARRIER(smp_mb__after_spinlock(), true);
	KCSAN_EXPECT_RW_BARRIER(smp_store_mb(test_var, 0), true);
	KCSAN_EXPECT_RW_BARRIER(smp_load_acquire(&test_var), false);
	KCSAN_EXPECT_RW_BARRIER(smp_store_release(&test_var, 0), true);
	KCSAN_EXPECT_RW_BARRIER(xchg(&test_var, 0), true);
	KCSAN_EXPECT_RW_BARRIER(xchg_release(&test_var, 0), true);
	KCSAN_EXPECT_RW_BARRIER(xchg_relaxed(&test_var, 0), false);
	KCSAN_EXPECT_RW_BARRIER(cmpxchg(&test_var, 0,  0), true);
	KCSAN_EXPECT_RW_BARRIER(cmpxchg_release(&test_var, 0,  0), true);
	KCSAN_EXPECT_RW_BARRIER(cmpxchg_relaxed(&test_var, 0,  0), false);
	KCSAN_EXPECT_RW_BARRIER(atomic_read(&dummy), false);
	KCSAN_EXPECT_RW_BARRIER(atomic_read_acquire(&dummy), false);
	KCSAN_EXPECT_RW_BARRIER(atomic_set(&dummy, 0), false);
	KCSAN_EXPECT_RW_BARRIER(atomic_set_release(&dummy, 0), true);
	KCSAN_EXPECT_RW_BARRIER(atomic_add(1, &dummy), false);
	KCSAN_EXPECT_RW_BARRIER(atomic_add_return(1, &dummy), true);
	KCSAN_EXPECT_RW_BARRIER(atomic_add_return_acquire(1, &dummy), false);
	KCSAN_EXPECT_RW_BARRIER(atomic_add_return_release(1, &dummy), true);
	KCSAN_EXPECT_RW_BARRIER(atomic_add_return_relaxed(1, &dummy), false);
	KCSAN_EXPECT_RW_BARRIER(atomic_fetch_add(1, &dummy), true);
	KCSAN_EXPECT_RW_BARRIER(atomic_fetch_add_acquire(1, &dummy), false);
	KCSAN_EXPECT_RW_BARRIER(atomic_fetch_add_release(1, &dummy), true);
	KCSAN_EXPECT_RW_BARRIER(atomic_fetch_add_relaxed(1, &dummy), false);
	KCSAN_EXPECT_RW_BARRIER(test_and_set_bit(0, &test_var), true);
	KCSAN_EXPECT_RW_BARRIER(test_and_clear_bit(0, &test_var), true);
	KCSAN_EXPECT_RW_BARRIER(test_and_change_bit(0, &test_var), true);
	KCSAN_EXPECT_RW_BARRIER(clear_bit_unlock(0, &test_var), true);
	KCSAN_EXPECT_RW_BARRIER(__clear_bit_unlock(0, &test_var), true);
	KCSAN_EXPECT_RW_BARRIER(arch_spin_lock(&arch_spinlock), false);
	KCSAN_EXPECT_RW_BARRIER(arch_spin_unlock(&arch_spinlock), true);
	KCSAN_EXPECT_RW_BARRIER(spin_lock(&test_spinlock), false);
	KCSAN_EXPECT_RW_BARRIER(spin_unlock(&test_spinlock), true);
	KCSAN_EXPECT_RW_BARRIER(mutex_lock(&test_mutex), false);
	KCSAN_EXPECT_RW_BARRIER(mutex_unlock(&test_mutex), true);
	KCSAN_EXPECT_READ_BARRIER(xor_unlock_is_negative_byte(1, &test_var), true);
	KCSAN_EXPECT_WRITE_BARRIER(xor_unlock_is_negative_byte(1, &test_var), true);
	KCSAN_EXPECT_RW_BARRIER(xor_unlock_is_negative_byte(1, &test_var), true);
	kcsan_nestable_atomic_end();
}

/**
 * @brief Tests KCSAN's basic data race detection for simple read/write conflicts.
 *
 * This test sets up a scenario where one thread performs a non-atomic write
 * (`test_kernel_write`) and another performs a non-atomic read (`test_kernel_read`)
 * to the same shared variable (`test_var`). KCSAN is expected to report a data race.
 * It also asserts that no false positive reports are generated for read-read races.
 *
 * @param test The KUnit test context.
 * @pre `test_kernel_write` and `test_kernel_read` modify/access `test_var`.
 * @post KCSAN detects and reports the expected read-write data race.
 */
__no_kcsan
static void test_basic(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			{ test_kernel_write, &test_var, sizeof(test_var), KCSAN_ACCESS_WRITE },
			{ test_kernel_read, &test_var, sizeof(test_var), 0 }, // Plain read, no special flags.
		},
	};
	struct expect_report never = {
		.access = {
			{ test_kernel_read, &test_var, sizeof(test_var), 0 },
			{ test_kernel_read, &test_var, sizeof(test_var), 0 },
		},
	};
	bool match_expect = false;
	bool match_never = false;

	// Set up concurrent execution of a write kernel and a read kernel.
	begin_test_checks(test_kernel_write, test_kernel_read);
	do {
		match_expect |= report_matches(&expect); // Check if the expected read-write race is reported.
		match_never = report_matches(&never);    // Check for unexpected read-read race reports.
	} while (!end_test_checks(match_never)); // Continue until expected or unexpected reports found, or timeout.
	KUNIT_EXPECT_TRUE(test, match_expect);    // Assert that the expected race was reported.
	KUNIT_EXPECT_FALSE(test, match_never);   // Assert that no unexpected race (read-read) was reported.
}

/**
 * @brief Stresses KCSAN with numerous concurrent Read-Modify-Write (RMW) races on different addresses.
 *
 * This test uses `test_kernel_rmw_array` concurrently from multiple threads. Each thread
 * performs RMW operations on potentially different elements of `test_array`.
 * The purpose is to stress KCSAN's ability to handle and report a high volume
 * of concurrent data races across a large memory region without crashing or missing reports.
 *
 * @param test The KUnit test context.
 * @pre `test_kernel_rmw_array` performs RMW operations on `test_array`.
 * @post KCSAN detects and reports a data race for at least one of the concurrent RMW operations.
 */
__no_kcsan
static void test_concurrent_races(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			/* NULL for address will match any address in the report. */
			{ test_kernel_rmw_array, NULL, 0, __KCSAN_ACCESS_RW(KCSAN_ACCESS_WRITE) }, // Expected write access type.
			{ test_kernel_rmw_array, NULL, 0, __KCSAN_ACCESS_RW(0) }, // Expected read access type.
		},
	};
	struct expect_report never = {
		.access = {
			{ test_kernel_rmw_array, NULL, 0, 0 }, // Plain read
			{ test_kernel_rmw_array, NULL, 0, 0 }, // Plain read
		},
	};
	bool match_expect = false;
	bool match_never = false;

	// Set up concurrent execution of `test_kernel_rmw_array` from both threads.
	begin_test_checks(test_kernel_rmw_array, test_kernel_rmw_array);
	do {
		match_expect |= report_matches(&expect); // Check for expected RMW races.
		match_never |= report_matches(&never);   // Check for unexpected plain read races.
	} while (!end_test_checks(false));          // Continue until timeout or significant reports are found.
	KUNIT_EXPECT_TRUE(test, match_expect);       // Assert that at least one expected race was reported.
	KUNIT_EXPECT_FALSE(test, match_never);      // Assert that no unexpected plain read races were reported.
}

/**
 * @brief Tests KCSAN's behavior with the `KCSAN_REPORT_VALUE_CHANGE_ONLY` option.
 *
 * This test uses `test_kernel_write_nochange` (which writes the same value to `test_var`)
 * and `test_kernel_read` concurrently. When `KCSAN_REPORT_VALUE_CHANGE_ONLY`
 * is enabled, KCSAN should suppress reports for races where the value does not
 * actually change. This test verifies this suppression behavior.
 *
 * @param test The KUnit test context.
 * @pre `test_kernel_write_nochange` performs a write that might not change `test_var`'s value.
 * @post KCSAN reports depend on `CONFIG_KCSAN_REPORT_VALUE_CHANGE_ONLY`.
 */
__no_kcsan
static void test_novalue_change(struct kunit *test)
{
	struct expect_report expect_rw = {
		.access = {
			{ test_kernel_write_nochange, &test_var, sizeof(test_var), KCSAN_ACCESS_WRITE },
			{ test_kernel_read, &test_var, sizeof(test_var), 0 },
		},
	};
	struct expect_report expect_ww = {
		.access = {
			{ test_kernel_write_nochange, &test_var, sizeof(test_var), KCSAN_ACCESS_WRITE },
			{ test_kernel_write_nochange, &test_var, sizeof(test_var), KCSAN_ACCESS_WRITE },
		},
	};
	bool match_expect = false;

	test_kernel_write_nochange(); /* Reset value. */ // Initialize `test_var` to a known state.
	begin_test_checks(test_kernel_write_nochange, test_kernel_read); // Run concurrent writes and reads.
	do {
		// Check for reports from both read-write and write-write scenarios.
		match_expect = report_matches(&expect_rw) || report_matches(&expect_ww);
	} while (!end_test_checks(match_expect));
	// Assert based on whether `CONFIG_KCSAN_REPORT_VALUE_CHANGE_ONLY` is enabled.
	if (IS_ENABLED(CONFIG_KCSAN_REPORT_VALUE_CHANGE_ONLY))
		KUNIT_EXPECT_FALSE(test, match_expect); // Expect no reports if option is enabled.
	else
		KUNIT_EXPECT_TRUE(test, match_expect);  // Expect reports if option is disabled.
}

/**
 * @brief Tests KCSAN's `KCSAN_REPORT_VALUE_CHANGE_ONLY` exceptions (RCU).
 *
 * This test verifies that certain types of writes, specifically those marked
 * as RCU-related (`test_kernel_write_nochange_rcu`), are always reported
 * as data races, even if `KCSAN_REPORT_VALUE_CHANGE_ONLY` is enabled.
 * This ensures that critical updates, even if value-preserving, are not suppressed.
 *
 * @param test The KUnit test context.
 * @pre `test_kernel_write_nochange_rcu` is intended to bypass value-change suppression.
 * @post KCSAN always reports the data race, regardless of `KCSAN_REPORT_VALUE_CHANGE_ONLY`.
 */
__no_kcsan
static void test_novalue_change_exception(struct kunit *test)
{
	struct expect_report expect_rw = {
		.access = {
			{ test_kernel_write_nochange_rcu, &test_var, sizeof(test_var), KCSAN_ACCESS_WRITE },
			{ test_kernel_read, &test_var, sizeof(test_var), 0 },
		},
	};
	struct expect_report expect_ww = {
		.access = {
			{ test_kernel_write_nochange_rcu, &test_var, sizeof(test_var), KCSAN_ACCESS_WRITE },
			{ test_kernel_write_nochange_rcu, &test_var, sizeof(test_var), KCSAN_ACCESS_WRITE },
		},
	};
	bool match_expect = false;

	test_kernel_write_nochange_rcu(); /* Reset value. */ // Initialize `test_var` to a known state.
	begin_test_checks(test_kernel_write_nochange_rcu, test_kernel_read); // Run concurrent writes and reads.
	do {
		match_expect = report_matches(&expect_rw) || report_matches(&expect_ww); // Check for reports.
	} while (!end_test_checks(match_expect));
	KUNIT_EXPECT_TRUE(test, match_expect); // Assert that the race was reported, even with value-change suppression.
}

/**
 * @brief Tests KCSAN's ability to report data races where one access has an unknown origin.
 *
 * This test sets up a scenario where an instrumented read (`test_kernel_read`)
 * races with an explicitly uninstrumented write (`test_kernel_write_uninstrumented`).
 * KCSAN is expected to report a data race where one of the accesses has an
 * "unknown origin," but only if `CONFIG_KCSAN_REPORT_RACE_UNKNOWN_ORIGIN` is enabled.
 *
 * @param test The KUnit test context.
 * @pre `test_kernel_write_uninstrumented` is marked with `__no_kcsan`.
 * @post KCSAN reports a race with unknown origin if configured to do so.
 */
__no_kcsan
static void test_unknown_origin(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			{ test_kernel_read, &test_var, sizeof(test_var), 0 },
			{ NULL }, // Represents the unknown origin access.
		},
	};
	bool match_expect = false;

	begin_test_checks(test_kernel_write_uninstrumented, test_kernel_read);
	do {
		match_expect = report_matches(&expect);
	} while (!end_test_checks(match_expect));
	// Assert based on whether `CONFIG_KCSAN_REPORT_RACE_UNKNOWN_ORIGIN` is enabled.
	if (IS_ENABLED(CONFIG_KCSAN_REPORT_RACE_UNKNOWN_ORIGIN))
		KUNIT_EXPECT_TRUE(test, match_expect);
	else
		KUNIT_EXPECT_FALSE(test, match_expect);
}

/**
 * @brief Tests KCSAN's `KCSAN_ASSUME_PLAIN_WRITES_ATOMIC` option for write-write races.
 *
 * This test involves two concurrent non-atomic writes to `test_var`. If
 * `KCSAN_ASSUME_PLAIN_WRITES_ATOMIC` is enabled, KCSAN is expected to suppress
 * reporting this write-write race, treating plain writes as atomic.
 * Otherwise, it should report the race.
 *
 * @param test The KUnit test context.
 * @post KCSAN reports depend on `CONFIG_KCSAN_ASSUME_PLAIN_WRITES_ATOMIC`.
 */
__no_kcsan
static void test_write_write_assume_atomic(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			{ test_kernel_write, &test_var, sizeof(test_var), KCSAN_ACCESS_WRITE },
			{ test_kernel_write, &test_var, sizeof(test_var), KCSAN_ACCESS_WRITE },
		},
	};
	bool match_expect = false;

	begin_test_checks(test_kernel_write, test_kernel_write); // Two concurrent non-atomic writes.
	do {
		sink_value(READ_ONCE(test_var)); /* induce value-change */ // Ensure `test_var` value changes to make races detectable.
		match_expect = report_matches(&expect);
	} while (!end_test_checks(match_expect));
	// Assert based on whether `CONFIG_KCSAN_ASSUME_PLAIN_WRITES_ATOMIC` is enabled.
	if (IS_ENABLED(CONFIG_KCSAN_ASSUME_PLAIN_WRITES_ATOMIC))
		KUNIT_EXPECT_FALSE(test, match_expect); // Expect no reports if option enabled.
	else
		KUNIT_EXPECT_TRUE(test, match_expect);  // Expect reports if option disabled.
}

/**
 * @brief Tests that KCSAN always reports races involving writes larger than word-size,
 * even with `KCSAN_ASSUME_PLAIN_WRITES_ATOMIC`.
 *
 * This test involves two concurrent writes to `test_struct`, a data structure
 * larger than a word. KCSAN is expected to always report this race, regardless
 * of `KCSAN_ASSUME_PLAIN_WRITES_ATOMIC`, because the assumption only applies
 * to single-word writes.
 *
 * @param test The KUnit test context.
 * @pre `test_kernel_write_struct` performs a write to `test_struct`.
 * @post KCSAN always reports the write-write race on the larger structure.
 */
__no_kcsan
static void test_write_write_struct(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			{ test_kernel_write_struct, &test_struct, sizeof(test_struct), KCSAN_ACCESS_WRITE },
			{ test_kernel_write_struct, &test_struct, sizeof(test_struct), KCSAN_ACCESS_WRITE },
		},
	};
	bool match_expect = false;

	begin_test_checks(test_kernel_write_struct, test_kernel_write_struct); // Two concurrent writes to the structure.
	do {
		match_expect = report_matches(&expect);
	} while (!end_test_checks(match_expect));
	KUNIT_EXPECT_TRUE(test, match_expect); // Always expect a report.
}

/**
 * @brief Tests that KCSAN always reports races when one write is larger than word-size,
 * even with `KCSAN_ASSUME_PLAIN_WRITES_ATOMIC`.
 *
 * This test involves one concurrent write to the entire `test_struct` and another
 * to a sub-part of it (`test_struct.val[3]`). KCSAN is expected to always report
 * this race, regardless of `KCSAN_ASSUME_PLAIN_WRITES_ATOMIC`, due to the
 * presence of a word-sized or larger write to the structure.
 *
 * @param test The KUnit test context.
 * @pre `test_kernel_write_struct` writes to the whole structure.
 * @pre `test_kernel_write_struct_part` writes to a member of the structure.
 * @post KCSAN always reports the overlapping write race.
 */
__no_kcsan
static void test_write_write_struct_part(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			{ test_kernel_write_struct, &test_struct, sizeof(test_struct), KCSAN_ACCESS_WRITE },
			{ test_kernel_write_struct_part, &test_struct.val[3], sizeof(test_struct.val[3]), KCSAN_ACCESS_WRITE },
		},
	};
	bool match_expect = false;

	begin_test_checks(test_kernel_write_struct, test_kernel_write_struct_part); // Concurrent writes to whole struct and part.
	do {
		match_expect = report_matches(&expect);
	} while (!end_test_checks(match_expect));
	KUNIT_EXPECT_TRUE(test, match_expect); // Always expect a report.
}

/**
 * @brief Tests that races between atomic accesses are never reported by KCSAN.
 *
 * This test sets up concurrent atomic read (`test_kernel_read_atomic`) and
 * atomic write (`test_kernel_write_atomic`) operations on `test_var`.
 * KCSAN is expected to recognize these as properly synchronized atomic accesses
 * and, therefore, should not generate any data race reports.
 *
 * @param test The KUnit test context.
 * @pre `test_kernel_read_atomic` and `test_kernel_write_atomic` use atomic primitives.
 * @post KCSAN does not report any data races.
 */
__no_kcsan
static void test_read_atomic_write_atomic(struct kunit *test)
{
	bool match_never = false;

	begin_test_checks(test_kernel_read_atomic, test_kernel_write_atomic);
	do {
		match_never = report_available(); // Check if any report is available.
	} while (!end_test_checks(match_never)); // Continue until report found or timeout.
	KUNIT_EXPECT_FALSE(test, match_never); // Assert that no reports were generated.
}

/**
 * @brief Tests that a race between a plain access and an atomic access results in a report.
 *
 * This test involves a plain read (`test_kernel_read`) racing with an
 * atomic write (`test_kernel_write_atomic`) on `test_var`. KCSAN is
 * expected to report this as a data race, unless atomic accesses are
 * explicitly ignored via `CONFIG_KCSAN_IGNORE_ATOMICS`.
 *
 * @param test The KUnit test context.
 * @pre `CONFIG_KCSAN_IGNORE_ATOMICS` must not be enabled for this test to be meaningful.
 * @post KCSAN detects and reports the expected data race.
 */
__no_kcsan
static void test_read_plain_atomic_write(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			{ test_kernel_read, &test_var, sizeof(test_var), 0 }, // Plain read.
			{ test_kernel_write_atomic, &test_var, sizeof(test_var), KCSAN_ACCESS_WRITE | KCSAN_ACCESS_ATOMIC }, // Atomic write.
		},
	};
	bool match_expect = false;

	// Require that KCSAN is not ignoring atomics, as that would suppress the expected report.
	KCSAN_TEST_REQUIRES(test, !IS_ENABLED(CONFIG_KCSAN_IGNORE_ATOMICS));

	begin_test_checks(test_kernel_read, test_kernel_write_atomic);
	do {
		match_expect = report_matches(&expect); // Check if the expected race is reported.
	} while (!end_test_checks(match_expect));
	KUNIT_EXPECT_TRUE(test, match_expect); // Assert that the expected race was reported.
}

/**
 * @brief Tests that atomic Read-Modify-Write (RMW) operations generate correct reports
 * when racing with plain accesses.
 *
 * This test involves a plain read (`test_kernel_read`) racing with an atomic
 * RMW operation (`test_kernel_atomic_rmw`) on `test_var`. KCSAN is expected
 * to report this as a data race, treating the atomic RMW as a compound read-write.
 *
 * @param test The KUnit test context.
 * @pre `CONFIG_KCSAN_IGNORE_ATOMICS` must not be enabled.
 * @post KCSAN detects and reports the expected data race for the atomic RMW.
 */
__no_kcsan
static void test_read_plain_atomic_rmw(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			{ test_kernel_read, &test_var, sizeof(test_var), 0 }, // Plain read.
			{ test_kernel_atomic_rmw, &test_var, sizeof(test_var),
				KCSAN_ACCESS_COMPOUND | KCSAN_ACCESS_WRITE | KCSAN_ACCESS_ATOMIC }, // Atomic RMW is a compound write.
		},
	};
	bool match_expect = false;

	// Require that KCSAN is not ignoring atomics.
	KCSAN_TEST_REQUIRES(test, !IS_ENABLED(CONFIG_KCSAN_IGNORE_ATOMICS));

	begin_test_checks(test_kernel_read, test_kernel_atomic_rmw);
	do {
		match_expect = report_matches(&expect); // Check if the expected race is reported.
	} while (!end_test_checks(match_expect));
	KUNIT_EXPECT_TRUE(test, match_expect); // Assert that the expected race was reported.
}

/**
 * @brief Tests that zero-sized memory accesses do not cause data race reports.
 *
 * This test sets up a race between a regular write to a structure (`test_kernel_write_struct`)
 * and a zero-sized read access to a member of that structure (`test_kernel_read_struct_zero_size`).
 * KCSAN is expected to report the race involving the regular write, but should *not*
 * report any race related to the zero-sized access, as such accesses are typically
 * ignored for race detection purposes.
 *
 * @param test The KUnit test context.
 * @pre `test_kernel_read_struct_zero_size` performs a zero-sized access.
 * @post KCSAN reports races for normal accesses but ignores zero-sized ones.
 */
__no_kcsan
static void test_zero_size_access(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			{ test_kernel_write_struct, &test_struct, sizeof(test_struct), KCSAN_ACCESS_WRITE },
			{ test_kernel_write_struct, &test_struct, sizeof(test_struct), KCSAN_ACCESS_WRITE }, // Expect two writes to race.
		},
	};
	struct expect_report never = {
		.access = {
			{ test_kernel_write_struct, &test_struct, sizeof(test_struct), KCSAN_ACCESS_WRITE },
			{ test_kernel_read_struct_zero_size, &test_struct.val[3], 0, 0 }, // Should not match on zero-sized read.
		},
	};
	bool match_expect = false;
	bool match_never = false;

	// Set up concurrent execution of writing to the struct and a zero-sized read.
	begin_test_checks(test_kernel_write_struct, test_kernel_read_struct_zero_size);
	do {
		match_expect |= report_matches(&expect); // Check for expected races (write-write on struct).
		match_never = report_matches(&never);    // Check for unexpected reports involving zero-sized access.
	} while (!end_test_checks(match_never));
	KUNIT_EXPECT_TRUE(test, match_expect);    // Sanity check: ensure the expected write-write race is reported.
	KUNIT_EXPECT_FALSE(test, match_never);   // Assert that no race was reported for the zero-sized access.
}

/**
 * @brief Tests the `data_race()` macro to ensure it suppresses KCSAN reports.
 *
 * This test involves two concurrent accesses to `test_var` using the `data_race()`
 * macro (`test_kernel_data_race`). KCSAN is explicitly instructed *not* to
 * report data races for accesses marked with `data_race()`. This test verifies
 * that KCSAN correctly honors this suppression and does not generate reports.
 *
 * @param test The KUnit test context.
 * @pre `test_kernel_data_race` uses the `data_race()` macro.
 * @post KCSAN does not report any data races.
 */
__no_kcsan
static void test_data_race(struct kunit *test)
{
	bool match_never = false;

	// Set up concurrent execution of `test_kernel_data_race` from both threads.
	begin_test_checks(test_kernel_data_race, test_kernel_data_race);
	do {
		match_never = report_available(); // Check if any report is available.
	} while (!end_test_checks(match_never)); // Continue until report found or timeout.
	KUNIT_EXPECT_FALSE(test, match_never); // Assert that no reports were generated.
}

/**
 * @brief Tests the `__data_racy` type qualifier to ensure it suppresses KCSAN reports.
 *
 * This test involves two concurrent accesses to `test_data_racy`, a variable
 * marked with the `__data_racy` type qualifier. KCSAN is explicitly
 * instructed *not* to report data races for accesses to such variables.
 * This test verifies that KCSAN correctly honors this suppression.
 *
 * @param test The KUnit test context.
 * @pre `test_data_racy` is declared with `__data_racy` qualifier.
 * @post KCSAN does not report any data races.
 */
__no_kcsan
static void test_data_racy_qualifier(struct kunit *test)
{
	bool match_never = false;

	// Set up concurrent execution of `test_kernel_data_racy_qualifier` from both threads.
	begin_test_checks(test_kernel_data_racy_qualifier, test_kernel_data_racy_qualifier);
	do {
		match_never = report_available(); // Check if any report is available.
	} while (!end_test_checks(match_never)); // Continue until report found or timeout.
	KUNIT_EXPECT_FALSE(test, match_never); // Assert that no reports were generated.
}

/**
 * @brief Tests KCSAN's detection of a write race against an `ASSERT_EXCLUSIVE_WRITER`.
 *
 * This test establishes an `ASSERT_EXCLUSIVE_WRITER` on `test_var` and concurrently
 * performs a non-atomic write (`test_kernel_write_nochange`). KCSAN is expected
 * to report a violation of the exclusive writer assertion.
 *
 * @param test The KUnit test context.
 * @post KCSAN reports a violation of the exclusive writer assertion.
 */
__no_kcsan
static void test_assert_exclusive_writer(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			{ test_kernel_assert_writer, &test_var, sizeof(test_var), KCSAN_ACCESS_ASSERT }, // The assertion itself.
			{ test_kernel_write_nochange, &test_var, sizeof(test_var), KCSAN_ACCESS_WRITE }, // The racing write.
		},
	};
	bool match_expect = false;

	begin_test_checks(test_kernel_assert_writer, test_kernel_write_nochange);
	do {
		match_expect = report_matches(&expect);
	} while (!end_test_checks(match_expect));
	KUNIT_EXPECT_TRUE(test, match_expect); // Assert that the expected assertion violation is reported.
}

/**
 * @brief Tests KCSAN's detection of a read race against an `ASSERT_EXCLUSIVE_ACCESS`.
 *
 * This test establishes an `ASSERT_EXCLUSIVE_ACCESS` (which implies no reads or writes)
 * on `test_var` and concurrently performs a non-atomic read (`test_kernel_read`).
 * KCSAN is expected to report a violation of the exclusive access assertion.
 *
 * @param test The KUnit test context.
 * @post KCSAN reports a violation of the exclusive access assertion.
 */
__no_kcsan
static void test_assert_exclusive_access(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			{ test_kernel_assert_access, &test_var, sizeof(test_var), KCSAN_ACCESS_ASSERT | KCSAN_ACCESS_WRITE }, // The assertion itself implies a write internally.
			{ test_kernel_read, &test_var, sizeof(test_var), 0 }, // The racing read.
		},
	};
	bool match_expect = false;

	begin_test_checks(test_kernel_assert_access, test_kernel_read);
	do {
		match_expect = report_matches(&expect);
	} while (!end_test_checks(match_expect));
	KUNIT_EXPECT_TRUE(test, match_expect); // Assert that the expected assertion violation is reported.
}

/**
 * @brief Tests interactions between `ASSERT_EXCLUSIVE_ACCESS` and `ASSERT_EXCLUSIVE_WRITER`.
 *
 * This test runs `ASSERT_EXCLUSIVE_ACCESS` and `ASSERT_EXCLUSIVE_WRITER` concurrently
 * on `test_var`. It verifies that KCSAN correctly identifies conflicts between
 * these assertions, as `EXCLUSIVE_ACCESS` is stricter than `EXCLUSIVE_WRITER`.
 *
 * @param test The KUnit test context.
 * @post KCSAN reports conflicts between the exclusive access and exclusive writer assertions.
 */
__no_kcsan
static void test_assert_exclusive_access_writer(struct kunit *test)
{
	struct expect_report expect_access_writer = {
		.access = {
			{ test_kernel_assert_access, &test_var, sizeof(test_var), KCSAN_ACCESS_ASSERT | KCSAN_ACCESS_WRITE },
			{ test_kernel_assert_writer, &test_var, sizeof(test_var), KCSAN_ACCESS_ASSERT },
		},
	};
	struct expect_report expect_access_access = {
		.access = {
			{ test_kernel_assert_access, &test_var, sizeof(test_var), KCSAN_ACCESS_ASSERT | KCSAN_ACCESS_WRITE },
			{ test_kernel_assert_access, &test_var, sizeof(test_var), KCSAN_ACCESS_ASSERT | KCSAN_ACCESS_WRITE },
		},
	};
	struct expect_report never = {
		.access = {
			{ test_kernel_assert_writer, &test_var, sizeof(test_var), KCSAN_ACCESS_ASSERT },
			{ test_kernel_assert_writer, &test_var, sizeof(test_var), KCSAN_ACCESS_ASSERT },
		},
	};
	bool match_expect_access_writer = false;
	bool match_expect_access_access = false;
	bool match_never = false;

	begin_test_checks(test_kernel_assert_access, test_kernel_assert_writer); // Run access and writer assertions concurrently.
	do {
		match_expect_access_writer |= report_matches(&expect_access_writer); // Check for access-writer conflict.
		match_expect_access_access |= report_matches(&expect_access_access); // Check for access-access conflict.
		match_never |= report_matches(&never);                              // Check for writer-writer false positive.
	} while (!end_test_checks(match_never));
	KUNIT_EXPECT_TRUE(test, match_expect_access_writer); // Expect conflict between access and writer.
	KUNIT_EXPECT_TRUE(test, match_expect_access_access); // Expect conflict between two access assertions.
	KUNIT_EXPECT_FALSE(test, match_never);               // Assert no false positive for writer-writer.
}

/**
 * @brief Tests KCSAN's detection of bit-level changes conflicting with `ASSERT_EXCLUSIVE_BITS`.
 *
 * This test establishes an `ASSERT_EXCLUSIVE_BITS` for `TEST_CHANGE_BITS` on `test_var`
 * and concurrently performs a write that modifies these bits (`test_kernel_change_bits`).
 * KCSAN is expected to report a violation, confirming its ability to detect
 * fine-grained bit-level data races and assertion violations.
 *
 * @param test The KUnit test context.
 * @post KCSAN reports a violation of the bit-level exclusive assertion.
 */
__no_kcsan
static void test_assert_exclusive_bits_change(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			{ test_kernel_assert_bits_change, &test_var, sizeof(test_var), KCSAN_ACCESS_ASSERT }, // The bit-level assertion.
			{ test_kernel_change_bits, &test_var, sizeof(test_var),
				KCSAN_ACCESS_WRITE | (IS_ENABLED(CONFIG_KCSAN_IGNORE_ATOMICS) ? 0 : KCSAN_ACCESS_ATOMIC) }, // The racing write.
		},
	};
	bool match_expect = false;

	begin_test_checks(test_kernel_assert_bits_change, test_kernel_change_bits);
	do {
		match_expect = report_matches(&expect);
	} while (!end_test_checks(match_expect));
	KUNIT_EXPECT_TRUE(test, match_expect); // Assert that the expected assertion violation is reported.
}

/**
 * @brief Tests that `ASSERT_EXCLUSIVE_BITS` does not report races when only unasserted bits change.
 *
 * This test establishes an `ASSERT_EXCLUSIVE_BITS` for bits *not* in `TEST_CHANGE_BITS` (`~TEST_CHANGE_BITS`)
 * and concurrently performs a write that *only* modifies `TEST_CHANGE_BITS` (`test_kernel_change_bits`).
 * KCSAN is expected to *not* report a violation, verifying its precision in bit-level assertions.
 *
 * @param test The KUnit test context.
 * @post KCSAN does not report a violation, as only unasserted bits are changed.
 */
__no_kcsan
static void test_assert_exclusive_bits_nochange(struct kunit *test)
{
	bool match_never = false;

	// Set up concurrent execution of an assertion on some bits and a write that changes other bits.
	begin_test_checks(test_kernel_assert_bits_nochange, test_kernel_change_bits);
	do {
		match_never = report_available(); // Check if any report is available.
	} while (!end_test_checks(match_never));
	KUNIT_EXPECT_FALSE(test, match_never); // Assert that no reports were generated.
}

/**
 * @brief Tests KCSAN's reporting for scoped exclusive writer assertions.
 *
 * This test uses `test_kernel_assert_writer_scoped` to define a scope where
 * `test_var` is expected to have an exclusive writer, and concurrently races it
 * with a non-atomic write (`test_kernel_write_nochange`). It verifies that
 * KCSAN reports violations pointing to the start of the `ASSERT_EXCLUSIVE_WRITER_SCOPED`
 * call site, not to accesses within the scope (like `test_enter_scope`).
 *
 * @param test The KUnit test context.
 * @post KCSAN reports violations attributed to the `ASSERT_EXCLUSIVE_WRITER_SCOPED` call.
 */
__no_kcsan
static void test_assert_exclusive_writer_scoped(struct kunit *test)
{
	struct expect_report expect_start = {
		.access = {
			{ test_kernel_assert_writer_scoped, &test_var, sizeof(test_var), KCSAN_ACCESS_ASSERT | KCSAN_ACCESS_SCOPED },
			{ test_kernel_write_nochange, &test_var, sizeof(test_var), KCSAN_ACCESS_WRITE },
		},
	};
	struct expect_report expect_inscope = {
		.access = {
			{ test_enter_scope, &test_var, sizeof(test_var), KCSAN_ACCESS_ASSERT | KCSAN_ACCESS_SCOPED },
			{ test_kernel_write_nochange, &test_var, sizeof(test_var), KCSAN_ACCESS_WRITE },
		},
	};
	bool match_expect_start = false;
	bool match_expect_inscope = false;

	begin_test_checks(test_kernel_assert_writer_scoped, test_kernel_write_nochange);
	do {
		match_expect_start |= report_matches(&expect_start);     // Check for report at assertion start.
		match_expect_inscope |= report_matches(&expect_inscope); // Check for report within scope (should not happen).
	} while (!end_test_checks(match_expect_inscope));
	KUNIT_EXPECT_TRUE(test, match_expect_start);     // Assert report at assertion start.
	KUNIT_EXPECT_FALSE(test, match_expect_inscope); // Assert no report from within scope.
}

/**
 * @brief Tests KCSAN's reporting for scoped exclusive access assertions.
 *
 * This test uses `test_kernel_assert_access_scoped` to define a scope where
 * `test_var` is expected to have exclusive access, and concurrently races it
 * with a non-atomic read (`test_kernel_read`). It verifies that KCSAN reports
 * violations pointing to the start of the `ASSERT_EXCLUSIVE_ACCESS_SCOPED`
 * call site, not to accesses within the scope. The test duration is extended
 * to increase the likelihood of race detection.
 *
 * @param test The KUnit test context.
 * @post KCSAN reports violations attributed to the `ASSERT_EXCLUSIVE_ACCESS_SCOPED` call.
 */
__no_kcsan
static void test_assert_exclusive_access_scoped(struct kunit *test)
{
	struct expect_report expect_start1 = {
		.access = {
			{ test_kernel_assert_access_scoped, &test_var, sizeof(test_var), KCSAN_ACCESS_ASSERT | KCSAN_ACCESS_WRITE | KCSAN_ACCESS_SCOPED },
			{ test_kernel_read, &test_var, sizeof(test_var), 0 },
		},
	};
	struct expect_report expect_start2 = {
		.access = { expect_start1.access[0], expect_start1.access[0] }, // Symmetric case.
	};
	struct expect_report expect_inscope = {
		.access = {
			{ test_enter_scope, &test_var, sizeof(test_var), KCSAN_ACCESS_ASSERT | KCSAN_ACCESS_WRITE | KCSAN_ACCESS_SCOPED },
			{ test_kernel_read, &test_var, sizeof(test_var), 0 },
		},
	};
	bool match_expect_start = false;
	bool match_expect_inscope = false;

	begin_test_checks(test_kernel_assert_access_scoped, test_kernel_read);
	end_time += msecs_to_jiffies(1000); /* This test requires a bit more time. */ // Extend test duration.
	do {
		match_expect_start |= report_matches(&expect_start1) || report_matches(&expect_start2); // Check for report at assertion start.
		match_expect_inscope |= report_matches(&expect_inscope);                            // Check for report within scope (should not happen).
	} while (!end_test_checks(match_expect_inscope));
	KUNIT_EXPECT_TRUE(test, match_expect_start);     // Assert report at assertion start.
	KUNIT_EXPECT_FALSE(test, match_expect_inscope); // Assert no report from within scope.
}

/**
 * @brief Tests that accesses to `jiffies` do not cause KCSAN reports.
 *
 * `jiffies` is a kernel global variable declared as `volatile`, and accesses
 * to it are typically not instrumented for race detection. This test verifies
 * that KCSAN does not report false positives when `test_kernel_jiffies_reader`
 * is executed concurrently.
 *
 * @param test The KUnit test context.
 * @post KCSAN does not report any data races related to `jiffies`.
 */
__no_kcsan
static void test_jiffies_noreport(struct kunit *test)
{
	bool match_never = false;

	begin_test_checks(test_kernel_jiffies_reader, test_kernel_jiffies_reader);
	do {
		match_never = report_available(); // Check if any report is available.
	} while (!end_test_checks(match_never));
	KUNIT_EXPECT_FALSE(test, match_never); // Assert that no reports were generated.
}

/**
 * @brief Tests that racing accesses within seqlock critical sections are not reported.
 *
 * This test sets up concurrent execution of a `seqlock` reader (`test_kernel_seqlock_reader`)
 * and a `seqlock` writer (`test_kernel_seqlock_writer`). KCSAN is expected to recognize
 * `seqlock` as a synchronization primitive and, therefore, should not report
 * data races for accesses properly guarded by `test_seqlock`.
 *
 * @param test The KUnit test context.
 * @post KCSAN does not report any data races related to seqlock-protected accesses.
 */
__no_kcsan
static void test_seqlock_noreport(struct kunit *test)
{
	bool match_never = false;

	begin_test_checks(test_kernel_seqlock_reader, test_kernel_seqlock_writer);
	do {
		match_never = report_available(); // Check if any report is available.
	} while (!end_test_checks(match_never));
	KUNIT_EXPECT_FALSE(test, match_never); // Assert that no reports were generated.
}

/**
 * @brief Tests KCSAN's recognition of atomic builtins and basic atomic operations.
 *
 * This test verifies that KCSAN correctly treats GCC's atomic builtins (like
 * `__atomic_store_n`, `__atomic_load_n`, `__atomic_compare_exchange_n`, etc.)
 * as atomic operations, and thus does not report data races when these are
 * used concurrently. It also includes various assertions to confirm the
 * correctness of these atomic operations.
 *
 * @param test The KUnit test context.
 * @post KCSAN does not report any data races for properly used atomic builtins.
 * @remark The `__atomic` builtins are generally not recommended for normal kernel code.
 */
static void test_atomic_builtins(struct kunit *test)
{
	bool match_never = false;

	// Run concurrent atomic accesses; KCSAN should not report races.
	begin_test_checks(test_kernel_atomic_builtins, test_kernel_atomic_builtins);
	do {
		long tmp;

		kcsan_enable_current(); // Re-enable KCSAN for the current thread for testing atomic operations.

		// Assertions to verify the behavior of various atomic builtins.
		__atomic_store_n(&test_var, 42L, __ATOMIC_RELAXED);
		KUNIT_EXPECT_EQ(test, 42L, __atomic_load_n(&test_var, __ATOMIC_RELAXED));

		KUNIT_EXPECT_EQ(test, 42L, __atomic_exchange_n(&test_var, 20, __ATOMIC_RELAXED));
		KUNIT_EXPECT_EQ(test, 20L, test_var);

		tmp = 20L;
		KUNIT_EXPECT_TRUE(test, __atomic_compare_exchange_n(&test_var, &tmp, 30L,
								    0, __ATOMIC_RELAXED,
								    __ATOMIC_RELAXED));
		KUNIT_EXPECT_EQ(test, tmp, 20L);
		KUNIT_EXPECT_EQ(test, test_var, 30L);
		KUNIT_EXPECT_FALSE(test, __atomic_compare_exchange_n(&test_var, &tmp, 40L,
								     1, __ATOMIC_RELAXED,
								     __ATOMIC_RELAXED));
		KUNIT_EXPECT_EQ(test, tmp, 30L);
		KUNIT_EXPECT_EQ(test, test_var, 30L);

		KUNIT_EXPECT_EQ(test, 30L, __atomic_fetch_add(&test_var, 1, __ATOMIC_RELAXED));
		KUNIT_EXPECT_EQ(test, 31L, __atomic_fetch_sub(&test_var, 1, __ATOMIC_RELAXED));
		KUNIT_EXPECT_EQ(test, 30L, __atomic_fetch_and(&test_var, 0xf, __ATOMIC_RELAXED));
		KUNIT_EXPECT_EQ(test, 14L, __atomic_fetch_xor(&test_var, 0xf, __ATOMIC_RELAXED));
		KUNIT_EXPECT_EQ(test, 1L, __atomic_fetch_or(&test_var, 0xf0, __ATOMIC_RELAXED));
		KUNIT_EXPECT_EQ(test, 241L, __atomic_fetch_nand(&test_var, 0xf, __ATOMIC_RELAXED));
		KUNIT_EXPECT_EQ(test, -2L, test_var);

		// Memory fences to ensure memory ordering for subsequent operations.
		__atomic_thread_fence(__ATOMIC_SEQ_CST);
		__atomic_signal_fence(__ATOMIC_SEQ_CST);

		kcsan_disable_current(); // Disable KCSAN again before `end_test_checks`.

		match_never = report_available();
	} while (!end_test_checks(match_never));
	KUNIT_EXPECT_FALSE(test, match_never); // Assert that no reports were generated.
}

/**
 * @brief Tests KCSAN's handling of 1-bit value changes within atomic sections.
 *
 * This test creates a race between a plain read (`test_kernel_read`) and a
 * specific bitwise XOR operation (`test_kernel_xor_1bit`) on `test_var`, where
 * the XOR operation is enclosed in `kcsan_nestable_atomic_begin/end`.
 * It verifies that KCSAN reports this race unless `CONFIG_KCSAN_PERMISSIVE`
 * is enabled, in which case it should be suppressed.
 *
 * @param test The KUnit test context.
 * @post KCSAN reports depend on `CONFIG_KCSAN_PERMISSIVE`.
 */
__no_kcsan
static void test_1bit_value_change(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			{ test_kernel_read, &test_var, sizeof(test_var), 0 },
			{ test_kernel_xor_1bit, &test_var, sizeof(test_var), __KCSAN_ACCESS_RW(KCSAN_ACCESS_WRITE) },
		},
	};
	bool match = false;

	begin_test_checks(test_kernel_read, test_kernel_xor_1bit);
	do {
		match = IS_ENABLED(CONFIG_KCSAN_PERMISSIVE)
				? report_available() // If permissive, check if *any* report is available (expect none).
				: report_matches(&expect); // Otherwise, check for specific expected report.
	} while (!end_test_checks(match));
	if (IS_ENABLED(CONFIG_KCSAN_PERMISSIVE))
		KUNIT_EXPECT_FALSE(test, match); // If permissive, expect no reports.
	else
		KUNIT_EXPECT_TRUE(test, match);  // Otherwise, expect a report.
}

/**
 * @brief Tests KCSAN's detection of correct memory barriers.
 *
 * This test uses `test_kernel_with_memorder` concurrently, where the critical
 * section is protected by `cmpxchg_acquire` and `smp_store_release`, ensuring
 * correct memory ordering. KCSAN is expected to *not* report any data races
 * because the synchronization primitives are used correctly.
 *
 * @param test The KUnit test context.
 * @post KCSAN does not report any data races.
 */
__no_kcsan
static void test_correct_barrier(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			{ test_kernel_with_memorder, &test_var, sizeof(test_var), __KCSAN_ACCESS_RW(KCSAN_ACCESS_WRITE) },
			{ test_kernel_with_memorder, &test_var, sizeof(test_var), __KCSAN_ACCESS_RW(0) },
		},
	};
	bool match_expect = false;

	test_struct.val[0] = 0; /* init unlocked */ // Initialize the lock flag.
	begin_test_checks(test_kernel_with_memorder, test_kernel_with_memorder);
	do {
		match_expect = report_matches_any_reordered(&expect); // Check for expected reports (should be none).
	} while (!end_test_checks(match_expect));
	KUNIT_EXPECT_FALSE(test, match_expect); // Assert that no reports were generated.
}

/**
 * @brief Tests KCSAN's detection of missing or incorrect memory barriers.
 *
 * This test uses `test_kernel_wrong_memorder` concurrently, where the critical
 * section uses `cmpxchg_relaxed` and `WRITE_ONCE` for synchronization, which
 * provides weak memory ordering. KCSAN is expected to report data races
 * if `CONFIG_KCSAN_WEAK_MEMORY` is enabled, as the weak ordering might not
 * be sufficient to prevent races.
 *
 * @param test The KUnit test context.
 * @post KCSAN reports data races if `CONFIG_KCSAN_WEAK_MEMORY` is enabled.
 */
__no_kcsan
static void test_missing_barrier(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			{ test_kernel_wrong_memorder, &test_var, sizeof(test_var), __KCSAN_ACCESS_RW(KCSAN_ACCESS_WRITE) },
			{ test_kernel_wrong_memorder, &test_var, sizeof(test_var), __KCSAN_ACCESS_RW(0) },
		},
	};
	bool match_expect = false;

	test_struct.val[0] = 0; /* init unlocked */ // Initialize the lock flag.
	begin_test_checks(test_kernel_wrong_memorder, test_kernel_wrong_memorder);
	do {
		match_expect = report_matches_any_reordered(&expect);
	} while (!end_test_checks(match_expect));
	if (IS_ENABLED(CONFIG_KCSAN_WEAK_MEMORY))
		KUNIT_EXPECT_TRUE(test, match_expect);  // Expect reports if weak memory model is assumed.
	else
		KUNIT_EXPECT_FALSE(test, match_expect); // Expect no reports if strong memory model (or KCSAN disabled for weak mem) is assumed.
}

/**
 * @brief Tests KCSAN's detection of correct memory barriers when using atomic builtins.
 *
 * This test uses `test_kernel_atomic_builtin_with_memorder` concurrently, where
 * the critical section is protected by atomic builtins with `__ATOMIC_ACQUIRE`
 * and `__ATOMIC_RELEASE` memory orders. KCSAN is expected to *not* report any
 * data races, as these builtins provide correct synchronization.
 *
 * @param test The KUnit test context.
 * @post KCSAN does not report any data races.
 */
__no_kcsan
static void test_atomic_builtins_correct_barrier(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			{ test_kernel_atomic_builtin_with_memorder, &test_var, sizeof(test_var), __KCSAN_ACCESS_RW(KCSAN_ACCESS_WRITE) },
			{ test_kernel_atomic_builtin_with_memorder, &test_var, sizeof(test_var), __KCSAN_ACCESS_RW(0) },
		},
	};
	bool match_expect = false;

	test_struct.val[0] = 0; /* init unlocked */ // Initialize the lock flag.
	begin_test_checks(test_kernel_atomic_builtin_with_memorder,
			  test_kernel_atomic_builtin_with_memorder);
	do {
		match_expect = report_matches_any_reordered(&expect);
	} while (!end_test_checks(match_expect));
	KUNIT_EXPECT_FALSE(test, match_expect); // Assert that no reports were generated.
}

/**
 * @brief Tests KCSAN's detection of missing or incorrect memory barriers with atomic builtins.
 *
 * This test uses `test_kernel_atomic_builtin_wrong_memorder` concurrently, where
 * the critical section uses atomic builtins with `__ATOMIC_RELAXED` memory ordering.
 * KCSAN is expected to report data races if `CONFIG_KCSAN_WEAK_MEMORY` is enabled,
 * as relaxed ordering might not provide sufficient synchronization.
 *
 * @param test The KUnit test context.
 * @post KCSAN reports data races if `CONFIG_KCSAN_WEAK_MEMORY` is enabled.
 */
__no_kcsan
static void test_atomic_builtins_missing_barrier(struct kunit *test)
{
	struct expect_report expect = {
		.access = {
			{ test_kernel_atomic_builtin_wrong_memorder, &test_var, sizeof(test_var), __KCSAN_ACCESS_RW(KCSAN_ACCESS_WRITE) },
			{ test_kernel_atomic_builtin_wrong_memorder, &test_var, sizeof(test_var), __KCSAN_ACCESS_RW(0) },
		},
	};
	bool match_expect = false;

	test_struct.val[0] = 0; /* init unlocked */ // Initialize the lock flag.
	begin_test_checks(test_kernel_atomic_builtin_wrong_memorder,
			  test_kernel_atomic_builtin_wrong_memorder);
	do {
		match_expect = report_matches_any_reordered(&expect);
	} while (!end_test_checks(match_expect));
	if (IS_ENABLED(CONFIG_KCSAN_WEAK_MEMORY))
		KUNIT_EXPECT_TRUE(test, match_expect);  // Expect reports if weak memory model is assumed.
	else
		KUNIT_EXPECT_FALSE(test, match_expect); // Expect no reports if strong memory model (or KCSAN disabled for weak mem) is assumed.
}

/**
 * @brief Parameter generator for `nthreads` (number of worker threads) for KUnit test cases.
 *
 * This function generates a sequence of thread counts for parameterized KUnit tests.
 * The sequence covers small thread counts (2-5) for boundary cases and then
 * exponentially increasing counts (8 to 32) to stress the system.
 * It also incorporates logic to adjust thread counts based on available online CPUs
 * and system preemption model to ensure test feasibility.
 *
 * @param prev The previous parameter value (thread count) generated; 0 for the first call.
 * @param desc Character buffer to store a human-readable description of the generated parameter.
 * @return The next `nthreads` value (as `void *`), or 0 to signal the end of parameters.
 */
static const void *nthreads_gen_params(const void *prev, char *desc)
{
	long nthreads = (long)prev;

	// Block Logic: Generate thread counts in a specific sequence.
	if (nthreads < 0 || nthreads >= 32)
		nthreads = 0; /* stop: Signal end of parameter generation. */
	else if (!nthreads)
		nthreads = 2; /* initial value: Start with 2 threads. */
	else if (nthreads < 5)
		nthreads++;   /* Increment for values 2, 3, 4, 5. */
	else if (nthreads == 5)
		nthreads = 8; /* Jump to 8 after 5. */
	else
		nthreads *= 2; /* Exponentially increase for values 8, 16, 32. */

	// Block Logic: Adjust thread count based on system capabilities.
	if (!preempt_model_preemptible() ||
	    !IS_ENABLED(CONFIG_KCSAN_INTERRUPT_WATCHER)) {
		/*
		 * Without any preemption, keep 2 CPUs free for other tasks, one
		 * of which is the main test case function checking for
		 * completion or failure.
		 */
		const long min_unused_cpus = preempt_model_none() ? 2 : 0;
		const long min_required_cpus = 2 + min_unused_cpus;

		// If insufficient CPUs are online, print an error and stop parameter generation.
		if (num_online_cpus() < min_required_cpus) {
			pr_err_once("Too few online CPUs (%u < %ld) for test\n",
				    num_online_cpus(), min_required_cpus);
			nthreads = 0;
		}
		// If the generated thread count exceeds available CPUs, limit it.
		else if (nthreads >= num_online_cpus() - min_unused_cpus) {
			/* Use negative value to indicate last param. */
			nthreads = -(num_online_cpus() - min_unused_cpus);
			pr_warn_once("Limiting number of threads to %ld (only %d online CPUs)\n",
				     -nthreads, num_online_cpus());
		}
	}

	// Format a descriptive string for the current parameter value.
	snprintf(desc, KUNIT_PARAM_DESC_SIZE, "threads=%ld", abs(nthreads));
	return (void *)nthreads;
}

/**
 * @brief Macro to define a KUnit test case that is parameterized by thread counts.
 *
 * This macro uses `KUNIT_CASE_PARAM` to associate a test function with the
 * `nthreads_gen_params` parameter generator, allowing the test to be run
 * with varying numbers of concurrent threads.
 *
 * @param test_name The name of the test function (e.g., `test_basic`).
 */
#define KCSAN_KUNIT_CASE(test_name) KUNIT_CASE_PARAM(test_name, nthreads_gen_params)
static struct kunit_case kcsan_test_cases[] = {
	/**
	 * @brief Test case for verifying KCSAN's barrier instrumentation without creating worker threads.
	 *
	 * This test directly probes KCSAN's internal `reorder_access` state
	 * after performing various memory barrier operations. It checks whether
	 * KCSAN correctly instruments these barriers by setting `reorder_access->size`
	 * to 0, indicating that any previously recorded access has been ordered.
	 * It also verifies that lockdep warm-up doesn't interfere.
	 *
	 * @param test The KUnit test context.
	 * @pre Requires `CONFIG_KCSAN_WEAK_MEMORY` for `reorder_access` to be non-NULL.
	 * @pre Requires `CONFIG_SMP` to test synchronization primitives.
	 */
	KUNIT_CASE(test_barrier_nothreads),
	KCSAN_KUNIT_CASE(test_basic),
	KCSAN_KUNIT_CASE(test_concurrent_races),
	KCSAN_KUNIT_CASE(test_novalue_change),
	KCSAN_KUNIT_CASE(test_novalue_change_exception),
	KCSAN_KUNIT_CASE(test_unknown_origin),
	KCSAN_KUNIT_CASE(test_write_write_assume_atomic),
	KCSAN_KUNIT_CASE(test_write_write_struct),
	KCSAN_KUNIT_CASE(test_write_write_struct_part),
	KCSAN_KUNIT_CASE(test_read_atomic_write_atomic),
	KCSAN_KUNIT_CASE(test_read_plain_atomic_write),
	KCSAN_KUNIT_CASE(test_read_plain_atomic_rmw),
	KCSAN_KUNIT_CASE(test_zero_size_access),
	KCSAN_KUNIT_CASE(test_data_race),
	KCSAN_KUNIT_CASE(test_data_racy_qualifier),
	KCSAN_KUNIT_CASE(test_assert_exclusive_writer),
	KCSAN_KUNIT_CASE(test_assert_exclusive_access),
	KCSAN_KUNIT_CASE(test_assert_exclusive_access_writer),
	KCSAN_KUNIT_CASE(test_assert_exclusive_bits_change),
	KCSAN_KUNIT_CASE(test_assert_exclusive_bits_nochange),
	KCSAN_KUNIT_CASE(test_assert_exclusive_writer_scoped),
	KCSAN_KUNIT_CASE(test_assert_exclusive_access_scoped),
	KCSAN_KUNIT_CASE(test_jiffies_noreport),
	KCSAN_KUNIT_CASE(test_seqlock_noreport),
	KCSAN_KUNIT_CASE(test_atomic_builtins),
	KCSAN_KUNIT_CASE(test_1bit_value_change),
	KCSAN_KUNIT_CASE(test_correct_barrier),
	KCSAN_KUNIT_CASE(test_missing_barrier),
	KCSAN_KUNIT_CASE(test_atomic_builtins_correct_barrier),
	KCSAN_KUNIT_CASE(test_atomic_builtins_missing_barrier),
	{},
};

/* ===== End test cases ===== */

/**
 * @brief Timer callback function for concurrent accesses, simulating interrupt-driven access.
 *
 * This function is invoked periodically by a timer. It dynamically selects one of the
 * `access_kernels` functions (randomly using an atomic counter) and executes it.
 * This simulates concurrent memory accesses that might originate from interrupt
 * handlers, providing an important test vector for KCSAN's interrupt watcher.
 *
 * @param timer Pointer to the `timer_list` structure that invoked this callback.
 * @pre `access_kernels` array is populated with valid function pointers.
 * @post One of the `access_kernels` functions may be executed.
 */
__no_kcsan
static void access_thread_timer(struct timer_list *timer)
{
	static atomic_t cnt = ATOMIC_INIT(0); // Atomic counter to select kernel.
	unsigned int idx;
	void (*func)(void);

	// Block Logic: Select a kernel function from `access_kernels` using an atomic counter.
	idx = (unsigned int)atomic_inc_return(&cnt) % ARRAY_SIZE(access_kernels);
	/* Block Logic: Acquire potential initialization. */
	// Load the function pointer with an acquire barrier to ensure proper synchronization
	// with `begin_test_checks` (which uses a release store).
	func = smp_load_acquire(&access_kernels[idx]);
	if (func)
		func(); // Execute the selected kernel function.
}

/**
 * @brief Main loop for each worker thread in the KCSAN test suite.
 *
 * Each `access_thread` continuously runs memory access "kernels" or schedules
 * a timer to do so, until signaled to stop by the `torture` framework.
 * This thread serves to generate the concurrent memory access patterns
 * that KCSAN monitors for data races.
 *
 * @param arg Unused argument.
 * @return 0 upon successful termination.
 * @pre `access_kernels` array is populated with valid function pointers.
 * @post The thread terminates cleanly.
 */
__no_kcsan
static int access_thread(void *arg)
{
	struct timer_list timer;
	unsigned int cnt = 0;
	unsigned int idx;
	void (*func)(void);

	timer_setup_on_stack(&timer, access_thread_timer, 0); // Setup a timer to call `access_thread_timer`.
	do {
		might_sleep(); // Indicate that this thread may sleep, allowing scheduling.

		// Block Logic: Periodically schedule the timer or execute a kernel directly.
		if (!timer_pending(&timer))
			mod_timer(&timer, jiffies + 1); // Schedule timer for the next jiffy.
		else {
			/* Iterate through all kernels. */
			// Block Logic: Select and execute a kernel function directly.
			idx = cnt++ % ARRAY_SIZE(access_kernels);
			/* Acquire potential initialization. */
			// Load the function pointer with an acquire barrier.
			func = smp_load_acquire(&access_kernels[idx]);
			if (func)
				func(); // Execute the selected kernel.
		}
	} while (!torture_must_stop()); // Continue until the `torture` framework signals to stop.
	timer_delete_sync(&timer);      // Delete the timer synchronously.
	timer_destroy_on_stack(&timer); // Destroy the timer on the stack.

	torture_kthread_stopping("access_thread"); // Signal thread stopping to the `torture` framework.
	return 0;
}

/**
 * @brief Initializes a KUnit test case, setting up worker threads and clearing previous reports.
 *
 * This function is invoked at the beginning of each KUnit test case. It clears
 * the `observed` report buffer, and for multithreaded tests, it initializes
 * the `torture` framework and creates `nthreads` worker `access_thread`s
 * to execute the test kernels concurrently.
 *
 * @param test The KUnit test context.
 * @return 0 on success, or a negative error code if thread creation or initialization fails.
 * @post `observed` report buffer is cleared. Worker threads are created for multithreaded tests.
 */
__no_kcsan
static int test_init(struct kunit *test)
{
	unsigned long flags;
	int nthreads;
	int i;

	// Block Logic: Clear any previously observed KCSAN reports.
	spin_lock_irqsave(&observed.lock, flags);
	for (i = 0; i < ARRAY_SIZE(observed.lines); ++i)
		observed.lines[i][0] = '\0'; // Clear content of each line.
	observed.nlines = 0;               // Reset line counter.
	spin_unlock_irqrestore(&observed.lock, flags);

	// For tests not involving threads (e.g., test_barrier_nothreads), skip thread setup.
	if (strstr(test->name, "nothreads"))
		return 0;

	// Initialize the torture framework for the current test.
	if (!torture_init_begin((char *)test->name, 1))
		return -EBUSY;

	// Assert that `threads` and `access_kernels` are initially null/empty.
	if (WARN_ON(threads))
		goto err;

	for (i = 0; i < ARRAY_SIZE(access_kernels); ++i) {
		if (WARN_ON(access_kernels[i]))
			goto err;
	}

	nthreads = abs((long)test->param_value); // Get the number of threads for this test case.
	if (WARN_ON(!nthreads))
		goto err;

	// Allocate memory for thread pointers.
	threads = kcalloc(nthreads + 1, sizeof(struct task_struct *), GFP_KERNEL);
	if (WARN_ON(!threads))
		goto err;

	threads[nthreads] = NULL; // Null-terminate the array of thread pointers.
	// Create and start worker threads.
	for (i = 0; i < nthreads; ++i) {
		if (torture_create_kthread(access_thread, NULL, threads[i]))
			goto err;
	}

	torture_init_end(); // Finalize torture framework initialization.

	return 0;

err:
	// Error handling: Clean up allocated resources if thread creation fails.
	kfree(threads);
	threads = NULL;
	torture_init_end();
	return -EINVAL;
}

/**
 * @brief Cleans up resources after a KUnit test case completes.
 *
 * This function is invoked at the end of each KUnit test case. It signals
 * worker threads to stop, waits for their termination, releases allocated
 * memory, and cleans up the `torture` framework.
 *
 * @param test The KUnit test context.
 * @post Worker threads are stopped and joined. Dynamically allocated memory is freed.
 */
__no_kcsan
static void test_exit(struct kunit *test)
{
	struct task_struct **stop_thread;
	int i;

	// For tests not involving threads, skip thread cleanup.
	if (strstr(test->name, "nothreads"))
		return;

	// Begin torture framework cleanup.
	if (torture_cleanup_begin())
		return;

	// Block Logic: Signal all access kernels to stop.
	for (i = 0; i < ARRAY_SIZE(access_kernels); ++i)
		WRITE_ONCE(access_kernels[i], NULL); // Set function pointers to NULL to stop execution.

	// If worker threads were created, stop and join them.
	if (threads) {
		for (stop_thread = threads; *stop_thread; stop_thread++)
			torture_stop_kthread(reader_thread, *stop_thread); // Stop individual worker threads.

		kfree(threads); // Free memory allocated for thread pointers.
		threads = NULL;
	}

	torture_cleanup_end(); // Finalize torture framework cleanup.
}

/**
 * @brief Registers the `probe_console` function as a tracepoint handler for console output.
 *
 * This function sets up the mechanism for intercepting kernel console messages
 * to monitor for KCSAN reports.
 */
__no_kcsan
static void register_tracepoints(void)
{
	register_trace_console(probe_console, NULL); // Register `probe_console` to handle `console` tracepoint events.
}

/**
 * @brief Unregisters the `probe_console` tracepoint handler.
 *
 * This function cleans up the tracepoint registration, stopping the interception
 * of kernel console messages.
 */
__no_kcsan
static void unregister_tracepoints(void)
{
	unregister_trace_console(probe_console, NULL); // Unregister `probe_console` handler.
}

/**
 * @brief Initializes the KCSAN KUnit test suite.
 *
 * This function is called once before any test cases in the suite are run.
 * Its primary responsibility is to register the tracepoint handler for
 * console output, enabling the test suite to capture KCSAN reports.
 *
 * @param suite Pointer to the `kunit_suite` structure for the KCSAN tests.
 * @return 0 on success.
 * @post `probe_console` is registered as a tracepoint handler.
 */
static int kcsan_suite_init(struct kunit_suite *suite)
{
	register_tracepoints();
	return 0;
}

/**
 * @brief Cleans up resources after the KCSAN KUnit test suite has completed.
 *
 * This function is called once after all test cases in the suite have run.
 * It unregisters the tracepoint handler and synchronizes tracepoint
 * unregistration to ensure all resources are properly released.
 *
 * @param suite Pointer to the `kunit_suite` structure for the KCSAN tests.
 * @post `probe_console` is unregistered, and tracepoint unregistration is synchronized.
 */
static void kcsan_suite_exit(struct kunit_suite *suite)
{
	unregister_tracepoints();
	tracepoint_synchronize_unregister();
}

/**
 * @brief The KUnit test suite definition for KCSAN.
 *
 * This structure defines the metadata and functions for the KCSAN test suite,
 * including its name, the array of test cases, and the initialization/exit
 * functions for individual test cases and the entire suite.
 */
static struct kunit_suite kcsan_test_suite = {
	.name = "kcsan",             /**< Name of the KUnit test suite. */
	.test_cases = kcsan_test_cases, /**< Array of individual KUnit test cases. */
	.init = test_init,           /**< Initialization function for each test case. */
	.exit = test_exit,           /**< Exit/cleanup function for each test case. */
	.suite_init = kcsan_suite_init, /**< Initialization function for the entire test suite. */
	.suite_exit = kcsan_suite_exit, /**< Exit/cleanup function for the entire test suite. */
};

// Registers the KCSAN test suite with the KUnit framework.
kunit_test_suites(&kcsan_test_suite);

// Module metadata for the Linux kernel.
MODULE_DESCRIPTION("KCSAN test suite"); /**< Description of the kernel module. */
MODULE_LICENSE("GPL v2");              /**< License under which the module is distributed. */
MODULE_AUTHOR("Marco Elver <elver@google.com>"); /**< Author of the module. */
