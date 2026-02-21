/* SPDX-License-Identifier: GPL-2.0 */
/**
 * @file sprintf.h
 * @brief Kernel-level string formatting and parsing utilities.
 *
 * This header provides declarations for various `snprintf`, `sprintf`, `sscanf`,
 * and related functions adapted for use within the Linux kernel. These functions
 * offer robust string manipulation capabilities, including type-safe format
 * string checking via GCC/Clang `__printf` and `__scanf` attributes, and
 * memory allocation variants (`kasprintf`, `kvasprintf`).
 */
#ifndef _LINUX_KERNEL_SPRINTF_H_
#define _LINUX_KERNEL_SPRINTF_H_

#include <linux/compiler_attributes.h>
#include <linux/types.h>
#include <linux/stdarg.h>

/**
 * @brief Converts an unsigned long long number to a string.
 *
 * Functional Utility: This function converts a numeric value into its string
 * representation, writing it into a provided buffer with a specified maximum size
 * and ensuring a minimum width by padding.
 *
 * @param buf Pointer to the output character buffer.
 * @param size Maximum size of the output buffer.
 * @param num The unsigned long long number to convert.
 * @param width Minimum width of the output string (padded with spaces if shorter).
 * @return The number of characters written to the buffer, excluding the null terminator.
 */
int num_to_str(char *buf, int size, unsigned long long num, unsigned int width);

/**
 * @brief Formats and prints data to a string buffer.
 *
 * Functional Utility: This function composes a formatted string and writes
 * it to the buffer `buf`. It behaves similarly to the standard library `sprintf`,
 * but is intended for kernel use.
 *
 * @param buf Pointer to the output character buffer.
 * @param fmt The format string.
 * @param ... Variable arguments matching the format specifiers in `fmt`.
 * @return The number of characters written to the buffer, excluding the null terminator.
 * @annotation __printf(2, 3) Indicates that `fmt` is a format string (like printf)
 * and the arguments start from the 3rd parameter.
 */
__printf(2, 3) int sprintf(char *buf, const char * fmt, ...);
/**
 * @brief Formats and prints data to a string buffer using a va_list.
 *
 * Functional Utility: This function is the `va_list` variant of `sprintf`,
 * allowing for dynamic argument lists.
 *
 * @param buf Pointer to the output character buffer.
 * @param fmt The format string.
 * @param args The `va_list` of arguments.
 * @return The number of characters written to the buffer, excluding the null terminator.
 * @annotation __printf(2, 0) Indicates that `fmt` is a format string (like printf)
 * and the arguments are passed via `va_list` (position 0 means no further args).
 */
__printf(2, 0) int vsprintf(char *buf, const char *, va_list);
/**
 * @brief Formats and prints data to a sized string buffer.
 *
 * Functional Utility: This function composes a formatted string and writes
 * it to the buffer `buf`, ensuring that no more than `size - 1` characters
 * are written, thus preventing buffer overflows. It behaves similarly to
 * the standard library `snprintf`.
 *
 * @param buf Pointer to the output character buffer.
 * @param size Maximum size of the output buffer (including null terminator).
 * @param fmt The format string.
 * @param ... Variable arguments matching the format specifiers in `fmt`.
 * @return The number of characters that would have been written if `size`
 *         was sufficiently large, or a negative value on error.
 * @annotation __printf(3, 4) Indicates that `fmt` is a format string (like printf)
 * and the arguments start from the 4th parameter.
 */
__printf(3, 4) int snprintf(char *buf, size_t size, const char *fmt, ...);
/**
 * @brief Formats and prints data to a sized string buffer using a va_list.
 *
 * Functional Utility: This function is the `va_list` variant of `snprintf`,
 * preventing buffer overflows with dynamic argument lists.
 *
 * @param buf Pointer to the output character buffer.
 * @param size Maximum size of the output buffer (including null terminator).
 * @param fmt The format string.
 * @param args The `va_list` of arguments.
 * @return The number of characters that would have been written if `size`
 *         was sufficiently large, or a negative value on error.
 * @annotation __printf(3, 0) Indicates that `fmt` is a format string (like printf)
 * and the arguments are passed via `va_list` (position 0 means no further args).
 */
__printf(3, 0) int vsnprintf(char *buf, size_t size, const char *fmt, va_list args);
/**
 * @brief Formats and prints data to a sized string buffer, returning actual characters written.
 *
 * Functional Utility: Similar to `snprintf`, but returns the number of
 * characters *actually* written to `buf` (excluding the null terminator),
 * which is useful for situations where the exact written length is needed.
 *
 * @param buf Pointer to the output character buffer.
 * @param size Maximum size of the output buffer (including null terminator).
 * @param fmt The format string.
 * @param ... Variable arguments matching the format specifiers in `fmt`.
 * @return The number of characters written to the buffer, excluding the null terminator.
 * @annotation __printf(3, 4) Indicates that `fmt` is a format string (like printf)
 * and the arguments start from the 4th parameter.
 */
__printf(3, 4) int scnprintf(char *buf, size_t size, const char *fmt, ...);
/**
 * @brief Formats and prints data to a sized string buffer using a va_list, returning actual characters written.
 *
 * Functional Utility: This function is the `va_list` variant of `scnprintf`,
 * providing the actual count of characters written to `buf`.
 *
 * @param buf Pointer to the output character buffer.
 * @param size Maximum size of the output buffer (including null terminator).
 * @param fmt The format string.
 * @param args The `va_list` of arguments.
 * @return The number of characters written to the buffer, excluding the null terminator.
 * @annotation __printf(3, 0) Indicates that `fmt` is a format string (like printf)
 * and the arguments are passed via `va_list` (position 0 means no further args).
 */
__printf(3, 0) int vscnprintf(char *buf, size_t size, const char *fmt, va_list args);
/**
 * @brief Allocates a new string and formats data into it.
 *
 * Functional Utility: This function combines memory allocation (`kmalloc`)
 * and string formatting, returning a newly allocated buffer containing the
 * formatted string. The caller is responsible for freeing the memory.
 *
 * @param gfp `gfp_t` flags for memory allocation (e.g., `GFP_KERNEL`).
 * @param fmt The format string.
 * @param ... Variable arguments matching the format specifiers in `fmt`.
 * @return A pointer to the newly allocated formatted string, or `NULL` on error.
 * @annotation __printf(2, 3) Indicates that `fmt` is a format string (like printf)
 * and the arguments start from the 3rd parameter.
 * @annotation __malloc Indicates that this function returns newly allocated memory.
 */
__printf(2, 3) __malloc char *kasprintf(gfp_t gfp, const char *fmt, ...);
/**
 * @brief Allocates a new string and formats data into it using a va_list.
 *
 * Functional Utility: This function is the `va_list` variant of `kasprintf`,
 * combining memory allocation and string formatting with dynamic argument lists.
 *
 * @param gfp `gfp_t` flags for memory allocation (e.g., `GFP_KERNEL`).
 * @param fmt The format string.
 * @param args The `va_list` of arguments.
 * @return A pointer to the newly allocated formatted string, or `NULL` on error.
 * @annotation __printf(2, 0) Indicates that `fmt` is a format string (like printf)
 * and the arguments are passed via `va_list` (position 0 means no further args).
 * @annotation __malloc Indicates that this function returns newly allocated memory.
 */
__printf(2, 0) __malloc char *kvasprintf(gfp_t gfp, const char *fmt, va_list args);
/**
 * @brief Allocates a new constant string and formats data into it using a va_list.
 *
 * Functional Utility: Similar to `kvasprintf`, but returns a `const char *`,
 * implying the string content should not be modified by the caller.
 *
 * @param gfp `gfp_t` flags for memory allocation.
 * @param fmt The format string.
 * @param args The `va_list` of arguments.
 * @return A pointer to the newly allocated formatted constant string, or `NULL` on error.
 * @annotation __printf(2, 0) Indicates that `fmt` is a format string (like printf)
 * and the arguments are passed via `va_list` (position 0 means no further args).
 * @annotation __malloc Indicates that this function returns newly allocated memory.
 */
__printf(2, 0) __malloc const char *kvasprintf_const(gfp_t gfp, const char *fmt, va_list args);

/**
 * @brief Parses formatted input from a string.
 *
 * Functional Utility: This function reads data from a string `str` according
 * to the format string `fmt` and stores the results in the provided arguments.
 * It behaves similarly to the standard library `sscanf`.
 *
 * @param str The input string to parse.
 * @param fmt The format string.
 * @param ... Variable arguments where parsed values will be stored.
 * @return The number of input items successfully matched and assigned, or `EOF` on input failure.
 * @annotation __scanf(2, 3) Indicates that `fmt` is a format string (like scanf)
 * and the arguments start from the 3rd parameter.
 */
__scanf(2, 3) int sscanf(const char *, const char *, ...);
/**
 * @brief Parses formatted input from a string using a va_list.
 *
 * Functional Utility: This function is the `va_list` variant of `sscanf`,
 * allowing for dynamic argument lists during string parsing.
 *
 * @param str The input string to parse.
 * @param fmt The format string.
 * @param args The `va_list` of arguments.
 * @return The number of input items successfully matched and assigned, or `EOF` on input failure.
 * @annotation __scanf(2, 0) Indicates that `fmt` is a format string (like scanf)
 * and the arguments are passed via `va_list` (position 0 means no further args).
 */
__scanf(2, 0) int vsscanf(const char *, const char *, va_list);

/**
 * @brief Global flag to control hashing of pointers in kernel output.
 *
 * Functional Utility: When `true`, pointers printed in certain kernel
 * contexts might be hashed for security or obfuscation purposes, preventing
 * direct disclosure of memory addresses.
 */
extern bool no_hash_pointers;
/**
 * @brief Enables or disables hashing of pointers based on a string argument.
 *
 * Functional Utility: This function provides a way to control the `no_hash_pointers`
 * flag, typically used via kernel command line parameters.
 *
 * @param str A string argument that determines whether to enable pointer hashing.
 * @return An integer indicating success or failure.
 */
int no_hash_pointers_enable(char *str);

/**
 * @brief Formats a pointer argument for Rust-specific output.
 *
 * Functional Utility: This function is specifically designed to handle
 * Rust formatting (`%pA`) for pointer arguments, ensuring compatibility
 * when printing Rust-managed pointers in kernel contexts.
 *
 * @param buf Pointer to the output buffer.
 * @param end Pointer to the end of the output buffer.
 * @param ptr The pointer value to format.
 * @return A pointer to the next available position in the output buffer.
 */
char *rust_fmt_argument(char *buf, char *end, const void *ptr);

#endif	/* _LINUX_KERNEL_SPRINTF_H */