/**
 * @file sprintf.h
 * @brief This header file provides declarations for kernel-internal string formatting and parsing functions.
 * @details It includes `printf`-like functions (`sprintf`, `snprintf`, `vsnprintf`, `kasprintf`, `kvasprintf`)
 * for generating formatted output into strings, and `scanf`-like functions (`sscanf`, `vsscanf`) for parsing
 * input from strings. These functions are optimized and adapted for use within the Linux kernel environment,
 * often providing buffer-size safety and memory allocation capabilities.
 */
/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _LINUX_KERNEL_SPRINTF_H_
#define _LINUX_KERNEL_SPRINTF_H_

#include <linux/compiler_attributes.h>
#include <linux/types.h>


/**
 * @file sprintf.h
 * @brief This header file provides declarations for kernel-internal string formatting and parsing functions.
 * @details It includes `printf`-like functions (`sprintf`, `snprintf`, `vsnprintf`, `kasprintf`, `kvasprintf`)
 * for generating formatted output into strings, and `scanf`-like functions (`sscanf`, `vsscanf`) for parsing
 * input from strings. These functions are optimized and adapted for use within the Linux kernel environment,
 * often providing buffer-size safety and memory allocation capabilities.
 */
/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _LINUX_KERNEL_SPRINTF_H_
#define _LINUX_KERNEL_SPRINTF_H_

#include <linux/compiler_attributes.h>
#include <linux/types.h>

/**
 * @brief Converts an unsigned long long number to its string representation.
 * @param buf Pointer to the buffer where the string will be written.
 * @param size The maximum size of the buffer (including null terminator).
 * @param num The unsigned long long number to convert.
 * @param width The minimum field width. The output string will be padded with spaces if shorter.
 * @return The number of characters written to the buffer, excluding the null terminator.
 */
int num_to_str(char *buf, int size, unsigned long long num, unsigned int width);

/**
 * @brief Formats and stores a series of characters and values into a string buffer.
 * @details This function is similar to the standard C library's `sprintf`, but is intended for kernel use.
 * It does not provide buffer overflow protection, so `snprintf` or `scnprintf` should be preferred for safety.
 * @param buf Pointer to the buffer where the formatted string will be written.
 * @param fmt The format string (similar to printf).
 * @param ... Variable arguments according to the format string.
 * @return The number of characters written to the buffer, excluding the null terminator.
 */
__printf(2, 3) int sprintf(char *buf, const char * fmt, ...);

/**
 * @brief Formats and stores a series of characters and values into a string buffer using a va_list.
 * @details This function is the `va_list` version of `sprintf`, intended for kernel use without
 * buffer overflow protection.
 * @param buf Pointer to the buffer where the formatted string will be written.
 * @param fmt The format string (similar to printf).
 * @param args The `va_list` of arguments.
 * @return The number of characters written to the buffer, excluding the null terminator.
 */
__printf(2, 0) int vsprintf(char *buf, const char *, va_list);

/**
 * @brief Formats and stores a series of characters and values into a sized string buffer.
 * @details This function is a buffer-size-safe version of `sprintf`, preventing buffer overflows
 * by limiting the number of characters written to `size - 1` (plus null terminator).
 * @param buf Pointer to the buffer where the formatted string will be written.
 * @param size The maximum size of the buffer (including null terminator).
 * @param fmt The format string (similar to printf).
 * @param ... Variable arguments according to the format string.
 * @return The number of characters that *would have been written* if `size` were large enough,
 * excluding the null terminator. If the return value is greater than or equal to `size`,
 * the output was truncated.
 */
__printf(3, 4) int snprintf(char *buf, size_t size, const char *fmt, ...);

/**
 * @brief Formats and stores a series of characters and values into a sized string buffer using a va_list.
 * @details This function is the `va_list` version of `snprintf`, providing buffer overflow protection.
 * @param buf Pointer to the buffer where the formatted string will be written.
 * @param size The maximum size of the buffer (including null terminator).
 * @param fmt The format string (similar to printf).
 * @param args The `va_list` of arguments.
 * @return The number of characters that *would have been written* if `size` were large enough,
 * excluding the null terminator. If the return value is greater than or equal to `size`,
 * the output was truncated.
 */
__printf(3, 0) int vsnprintf(char *buf, size_t size, const char *fmt, va_list args);

/**
 * @brief Formats and stores a series of characters and values into a sized string buffer, returning characters written.
 * @details This function is similar to `snprintf` but returns the number of characters *actually written* to the buffer
 * (excluding the null terminator), which can be `size - 1` at most.
 * @param buf Pointer to the buffer where the formatted string will be written.
 * @param size The maximum size of the buffer (including null terminator).
 * @param fmt The format string (similar to printf).
 * @param ... Variable arguments according to the format string.
 * @return The number of characters actually written to the buffer, excluding the null terminator.
 * The buffer will always be null-terminated if `size > 0`.
 */
__printf(3, 4) int scnprintf(char *buf, size_t size, const char *fmt, ...);

/**
 * @brief Formats and stores a series of characters and values into a sized string buffer using a va_list,
 * returning characters written.
 * @details This function is the `va_list` version of `scnprintf`.
 * @param buf Pointer to the buffer where the formatted string will be written.
 * @param size The maximum size of the buffer (including null terminator).
 * @param fmt The format string (similar to printf).
 * @param args The `va_list` of arguments.
 * @return The number of characters actually written to the buffer, excluding the null terminator.
 * The buffer will always be null-terminated if `size > 0`.
 */
__printf(3, 0) int vscnprintf(char *buf, size_t size, const char *fmt, va_list args);

/**
 * @brief Allocates memory and formats a string into it.
 * @details This function allocates memory using `kmalloc` (or similar, depending on `gfp`)
 * and then formats a string into the newly allocated buffer. The caller is responsible for freeing the memory.
 * @param gfp `gfp_t` flags for memory allocation (e.g., `GFP_KERNEL`, `GFP_ATOMIC`).
 * @param fmt The format string (similar to printf).
 * @param ... Variable arguments according to the format string.
 * @return On success, a pointer to the newly allocated and formatted string. On failure, returns NULL.
 */
__printf(2, 3) __malloc char *kasprintf(gfp_t gfp, const char *fmt, ...);

/**
 * @brief Allocates memory and formats a string into it using a va_list.
 * @details This function is the `va_list` version of `kasprintf`.
 * @param gfp `gfp_t` flags for memory allocation.
 * @param fmt The format string (similar to printf).
 * @param args The `va_list` of arguments.
 * @return On success, a pointer to the newly allocated and formatted string. On failure, returns NULL.
 */
__printf(2, 0) __malloc char *kvasprintf(gfp_t gfp, const char *fmt, va_list args);

/**
 * @brief Allocates memory and formats a string into it using a va_list, returning a const pointer.
 * @details Similar to `kvasprintf` but returns a `const char *`, implying the content should not be modified
 * directly. The caller is responsible for freeing the memory.
 * @param gfp `gfp_t` flags for memory allocation.
 * @param fmt The format string (similar to printf).
 * @param args The `va_list` of arguments.
 * @return On success, a const pointer to the newly allocated and formatted string. On failure, returns NULL.
 */
__printf(2, 0) const char *kvasprintf_const(gfp_t gfp, const char *fmt, va_list args);

/**
 * @brief Parses input from a string according to a format.
 * @details This function is similar to the standard C library's `sscanf`, but is intended for kernel use.
 * @param buf The input string to parse.
 * @param fmt The format string (similar to scanf).
 * @param ... Variable arguments to store the parsed values.
 * @return The number of input items successfully matched and assigned.
 */
__scanf(2, 3) int sscanf(const char *, const char *, ...);

/**
 * @brief Parses input from a string according to a format using a va_list.
 * @details This function is the `va_list` version of `sscanf`, intended for kernel use.
 * @param buf The input string to parse.
 * @param fmt The format string (similar to scanf).
 * @param args The `va_list` of arguments to store the parsed values.
 * @return The number of input items successfully matched and assigned.
 */
__scanf(2, 0) int vsscanf(const char *, const char *, va_list);

/* These are for specific cases, do not use without real need */
/**
 * @brief Global flag to control the hashing of pointers in kernel output.
 * @details When `true`, pointer values printed with `%p` format specifier
 * might be obfuscated (hashed) for security or debugging purposes.
 * @warning Use `no_hash_pointers_enable` to modify this for specific cases.
 */
extern bool no_hash_pointers;

/**
 * @brief Enables or disables the hashing of pointers in kernel output based on a string.
 * @param str A string argument that can control the state of pointer hashing.
 *            (Specific parsing logic for `str` is in implementation, e.g., "on"/"off").
 * @return An integer indicating success or specific configuration applied.
 */
int no_hash_pointers_enable(char *str);

/* Used for Rust formatting ('%pA') */
/**
 * @brief Formats an argument for Rust-style pointer formatting ('%pA').
 * @details This function is specific to handling Rust's `%pA` format specifier
 * within the kernel's printf infrastructure. It formats a pointer into a buffer.
 * @param buf Pointer to the character buffer to write into.
 * @param end Pointer to the end of the buffer (limit).
 * @param ptr The pointer value to format.
 * @return A pointer to the next character after the formatted output in the buffer.
 */
char *rust_fmt_argument(char *buf, char *end, const void *ptr);

#endif	/* _LINUX_KERNEL_SPRINTF_H */

