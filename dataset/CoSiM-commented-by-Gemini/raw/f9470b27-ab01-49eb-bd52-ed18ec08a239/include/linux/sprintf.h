/* SPDX-License-Identifier: GPL-2.0 */
/**
 * @file sprintf.h
 * @brief Kernel-space string formatting and parsing utilities.
 *
 * This header provides the kernel's internal implementations of standard
 * string manipulation functions like sprintf, snprintf, and sscanf. It also
 * includes kernel-specific variants that handle memory allocation (kasprintf)
 * and other security features like pointer hashing. The function declarations
 * are annotated with compiler attributes (`__printf`, `__scanf`) to enable
 * compile-time checking of format strings and arguments, enhancing type safety.
 */
#ifndef _LINUX_KERNEL_SPRINTF_H_
#define _LINUX_KERNEL_SPRINTF_H_

#include <linux/compiler_attributes.h>
#include <linux/types.h>

/**
 * @brief Converts a number to its string representation.
 * @param buf   The buffer to place the result into.
 * @param size  The size of the buffer.
 * @param num   The number to convert.
 * @param width The minimum width of the output string.
 * @return The number of characters written to the buffer.
 */
int num_to_str(char *buf, int size, unsigned long long num, unsigned int width);

/**
 * @brief Formats a string and stores it in a buffer.
 * @note This function is unsafe as it does not check for buffer overflows.
 *       Use snprintf or scnprintf instead.
 */
__printf(2, 3) int sprintf(char *buf, const char * fmt, ...);

/**
 * @brief Formats a string from a va_list and stores it in a buffer.
 * @note This function is unsafe as it does not check for buffer overflows.
 */
__printf(2, 0) int vsprintf(char *buf, const char *, va_list);

/**
 * @brief Formats a string and safely stores it in a sized buffer.
 * @return The number of characters that would have been written, excluding the null terminator.
 */
__printf(3, 4) int snprintf(char *buf, size_t size, const char *fmt, ...);

/**
 * @brief Formats a string from a va_list and safely stores it in a sized buffer.
 */
__printf(3, 0) int vsnprintf(char *buf, size_t size, const char *fmt, va_list args);

/**
 * @brief A variant of snprintf that returns the number of characters actually written.
 * @return The number of characters written to the buffer, excluding the null terminator.
 */
__printf(3, 4) int scnprintf(char *buf, size_t size, const char *fmt, ...);

/**
 * @brief A variant of vsnprintf that returns the number of characters actually written.
 */
__printf(3, 0) int vscnprintf(char *buf, size_t size, const char *fmt, va_list args);

/**
 * @brief Formats a string into a dynamically allocated buffer.
 * @param gfp The GFP flags for memory allocation.
 * @return A pointer to the newly allocated buffer, or NULL on failure.
 */
__printf(2, 3) __malloc char *kasprintf(gfp_t gfp, const char *fmt, ...);

/**
 * @brief Formats a string from a va_list into a dynamically allocated buffer.
 */
__printf(2, 0) __malloc char *kvasprintf(gfp_t gfp, const char *fmt, va_list args);

/**
 * @brief A variant of kvasprintf that returns a const char pointer.
 */
__printf(2, 0) const char *kvasprintf_const(gfp_t gfp, const char *fmt, va_list args);

/**
 * @brief Parses a string according to a format string.
 */
__scanf(2, 3) int sscanf(const char *, const char *, ...);

/**
 * @brief Parses a string using a va_list.
 */
__scanf(2, 0) int vsscanf(const char *, const char *, va_list);

/* These are for specific cases, do not use without real need */
/**
 * @brief A global flag to disable pointer hashing in printk.
 * Functional Utility: Used for debugging purposes where actual pointer values are needed.
 */
extern bool no_hash_pointers;
int no_hash_pointers_enable(char *str);

/* Used for Rust formatting ('%pA') */
/**
 * @brief Formatter for Rust's '%pA' format specifier.
 * @param buf The buffer to write to.
 * @param end The end of the buffer.
 * @param ptr The pointer to format.
 * @return A pointer to the end of the written string.
 */
char *rust_fmt_argument(char *buf, char *end, const void *ptr);

#endif	/* _LINUX_KERNEL_SPRINTF_H */
