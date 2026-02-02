/*
   BlueZ - Bluetooth protocol stack for Linux
   Copyright (C) 2000-2001 Qualcomm Incorporated

   Written 2000,2001 by Maxim Krasnyansky <maxk@qualcomm.com>

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License version 2 as
   published by the Free Software Foundation;

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD PARTY RIGHTS.
   IN NO EVENT SHALL THE COPYRIGHT HOLDER(S) AND AUTHOR(S) BE LIABLE FOR ANY
   CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES
   WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
   ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
   OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

   ALL LIABILITY, INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PATENTS,
   COPYRIGHTS, TRADEMARKS OR OTHER RIGHTS, RELATING TO USE OF THIS
   SOFTWARE IS DISCLAIMED.
*/

/**
 * @file
 * @brief Bluetooth core utility functions and logging infrastructure.
 *
 * This file provides a set of common utility functions for the Bluetooth
 * stack, including byte swapping for Bluetooth addresses, error code
 * translation between the Bluetooth specification and standard Linux errnos,
 * and a logging framework with different severity levels (info, warn, err, dbg).
 */

#define pr_fmt(fmt) "Bluetooth: " fmt

#include <linux/export.h>

#include <net/bluetooth/bluetooth.h>

/**
 * @brief Reverses the byte order of a Bluetooth device address (BD_ADDR).
 *
 * This function is essential for converting between the host byte order and
 * the little-endian format used in Bluetooth specifications.
 *
 * @param dst Pointer to the destination bdaddr_t structure to store the swapped address.
 * @param src Pointer to the source bdaddr_t structure containing the address to be swapped.
 */
void baswap(bdaddr_t *dst, const bdaddr_t *src)
{
	const unsigned char *s = (const unsigned char *)src;
	unsigned char *d = (unsigned char *)dst;
	unsigned int i;

	for (i = 0; i < 6; i++)
		d[i] = s[5 - i];
}
EXPORT_SYMBOL(baswap);

/**
 * @brief Translates a Bluetooth error code to a standard Linux errno value.
 *
 * This function maps error codes defined in the Bluetooth specification to their
 * corresponding standard Linux error numbers, facilitating consistent error handling
 * throughout the kernel.
 *
 * @param code The Bluetooth error code to be converted.
 * @return The corresponding standard Linux errno value. Returns ENOSYS if the
 *         code is not recognized.
 */
int bt_to_errno(__u16 code)
{
	switch (code) {
	case 0:
		return 0;

	case 0x01:
		return EBADRQC;

	case 0x02:
		return ENOTCONN;

	case 0x03:
		return EIO;

	case 0x04:
	case 0x3c:
		return EHOSTDOWN;

	case 0x05:
		return EACCES;

	case 0x06:
		return EBADE;

	case 0x07:
		return ENOMEM;

	case 0x08:
		return ETIMEDOUT;

	case 0x09:
		return EMLINK;

	case 0x0a:
		return EMLINK;

	case 0x0b:
		return EALREADY;

	case 0x0c:
		return EBUSY;

	case 0x0d:
	case 0x0e:
	case 0x0f:
		return ECONNREFUSED;

	case 0x10:
		return ETIMEDOUT;

	case 0x11:
	case 0x27:
	case 0x29:
	case 0x20:
		return EOPNOTSUPP;

	case 0x12:
		return EINVAL;

	case 0x13:
	case 0x14:
	case 0x15:
		return ECONNRESET;

	case 0x16:
		return ECONNABORTED;

	case 0x17:
		return ELOOP;

	case 0x18:
		return EACCES;

	case 0x1a:
		return EPROTONOSUPPORT;

	case 0x1b:
		return ECONNREFUSED;

	case 0x19:
	case 0x1e:
	case 0x23:
	case 0x24:
	case 0x25:
		return EPROTO;

	default:
		return ENOSYS;
	}
}
EXPORT_SYMBOL(bt_to_errno);

/**
 * @brief Translates a standard Linux errno value to a Bluetooth error code.
 *
 * This function maps standard Linux error numbers to their corresponding error codes
 * as defined in the Bluetooth specification, facilitating consistent error reporting
 * to remote devices.
 *
 * @param err The standard Linux errno value to be converted.
 * @return The corresponding Bluetooth error code. Returns 0x1f (unspecified error)
 *         if the errno is not recognized.
 */
__u8 bt_status(int err)
{
	if (err >= 0)
		return err;

	switch (err) {
	case -EBADRQC:
		return 0x01;

	case -ENOTCONN:
		return 0x02;

	case -EIO:
		return 0x03;

	case -EHOSTDOWN:
		return 0x04;

	case -EACCES:
		return 0x05;

	case -EBADE:
		return 0x06;

	case -ENOMEM:
		return 0x07;

	case -ETIMEDOUT:
		return 0x08;

	case -EMLINK:
		return 0x09;

	case -EALREADY:
		return 0x0b;

	case -EBUSY:
		return 0x0c;

	case -ECONNREFUSED:
		return 0x0d;

	case -EOPNOTSUPP:
		return 0x11;

	case -EINVAL:
		return 0x12;

	case -ECONNRESET:
		return 0x13;

	case -ECONNABORTED:
		return 0x16;

	case -ELOOP:
		return 0x17;

	case -EPROTONOSUPPORT:
		return 0x1a;

	case -EPROTO:
		return 0x19;

	default:
		return 0x1f;
	}
}
EXPORT_SYMBOL(bt_status);

/**
 * @brief Logs an informational message for the Bluetooth subsystem.
 *
 * This function provides a standardized way to log informational messages,
 * ensuring consistent formatting and identification of Bluetooth-related logs.
 *
 * @param format The format string for the message, similar to printk.
 * @param ... Variable arguments for the format string.
 */
void bt_info(const char *format, ...)
{
	struct va_format vaf;
	va_list args;

	va_start(args, format);

	vaf.fmt = format;
	vaf.va = &args;

	pr_info("%pV", &vaf);

	va_end(args);
}
EXPORT_SYMBOL(bt_info);

/**
 * @brief Logs a warning message for the Bluetooth subsystem.
 *
 * This function provides a standardized way to log warning messages,
 * ensuring consistent formatting and identification of Bluetooth-related logs.
 *
 * @param format The format string for the message, similar to printk.
 * @param ... Variable arguments for the format string.
 */
void bt_warn(const char *format, ...)
{
	struct va_format vaf;
	va_list args;

	va_start(args, format);

	vaf.fmt = format;
	vaf.va = &args;

	pr_warn("%pV", &vaf);

	va_end(args);
}
EXPORT_SYMBOL(bt_warn);

/**
 * @brief Logs an error message for the Bluetooth subsystem.
 *
 * This function provides a standardized way to log error messages,
 * ensuring consistent formatting and identification of Bluetooth-related logs.
 *
 * @param format The format string for the message, similar to printk.
 * @param ... Variable arguments for the format string.
 */
void bt_err(const char *format, ...)
{
	struct va_format vaf;
	va_list args;

	va_start(args, format);

	vaf.fmt = format;
	vaf.va = &args;

	pr_err("%pV", &vaf);

	va_end(args);
}
EXPORT_SYMBOL(bt_err);

#ifdef CONFIG_BT_FEATURE_DEBUG
static bool debug_enable;

/**
 * @brief Enables or disables Bluetooth debug logging globally.
 *
 * This function provides a runtime switch to control the verbosity of
 * debug messages from the Bluetooth subsystem, which is useful for
 * dynamic debugging without recompiling the kernel.
 *
 * @param enable A boolean flag; true to enable debug logs, false to disable.
 */
void bt_dbg_set(bool enable)
{
	debug_enable = enable;
}

/**
 * @brief Retrieves the current state of the global Bluetooth debug logging flag.
 *
 * @return A boolean value indicating whether debug logging is currently enabled.
 */
bool bt_dbg_get(void)
{
	return debug_enable;
}

/**
 * @brief Logs a debug message for the Bluetooth subsystem, if enabled.
 *
 * This function provides a standardized way to log debug messages. The log
 * is only produced if the debug_enable flag is true, minimizing performance
 * impact in production environments.
 *
 * @param format The format string for the message, similar to printk.
 * @param ... Variable arguments for the format string.
 */
void bt_dbg(const char *format, ...)
{
	struct va_format vaf;
	va_list args;

	if (likely(!debug_enable))
		return;

	va_start(args, format);

	vaf.fmt = format;
	vaf.va = &args;

	printk(KERN_DEBUG pr_fmt("%pV"), &vaf);

	va_end(args);
}
EXPORT_SYMBOL(bt_dbg);
#endif

/**
 * @brief Logs a rate-limited warning message for the Bluetooth subsystem.
 *
 * This function works like bt_warn but prevents flooding the logs by
 * limiting the rate at which identical messages are printed.
 *
 * @param format The format string for the message, similar to printk.
 * @param ... Variable arguments for the format string.
 */
void bt_warn_ratelimited(const char *format, ...)
{
	struct va_format vaf;
	va_list args;

	va_start(args, format);

	vaf.fmt = format;
	vaf.va = &args;

	pr_warn_ratelimited("%pV", &vaf);

	va_end(args);
}
EXPORT_SYMBOL(bt_warn_ratelimited);

/**
 * @brief Logs a rate-limited error message for the Bluetooth subsystem.
 *
 * This function works like bt_err but prevents flooding the logs by
 * limiting the rate at which identical messages are printed.
 *
 * @param format The format string for the message, similar to printk.
 * @param ... Variable arguments for the format string.
 */
void bt_err_ratelimited(const char *format, ...)
{
	struct va_format vaf;
	va_list args;

	va_start(args, format);

	vaf.fmt = format;
	vaf.va = &args;

	pr_err_ratelimited("%pV", &vaf);

	va_end(args);
}
EXPORT_SYMBOL(bt_err_ratelimited);