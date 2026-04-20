/* SPDX-License-Identifier: GPL-2.0 */
/*
 * @bb28a994-deef-40d4-b995-31a749a438d8/include/linux/gpio.h
 * @brief Legacy public API for the global GPIO numberspace in the Linux kernel.
 *
 * Functional Intent: Provides a backwards-compatible interface for GPIO drivers 
 * and consumers still using the integer-based global numberspace. This header 
 * facilitates the transition to the descriptor-based GPIO API (gpiod) by 
 * providing wrappers around newer primitives while maintaining support for 
 * legacy call sites.
 *
 * NOTE: This header is DEPRECATED. New implementations MUST use <linux/gpio/consumer.h> 
 * or <linux/gpio/driver.h> for descriptor-based access.
 *
 * Domain: Kernel Infrastructure, Hardware Abstraction, Legacy Support.
 */
#ifndef __LINUX_GPIO_H
#define __LINUX_GPIO_H

#include <linux/types.h>
#ifdef CONFIG_GPIOLIB
#include <linux/gpio/consumer.h>
#endif

#ifdef CONFIG_GPIOLIB_LEGACY

struct device;

/**
 * Functional Utility: Initialization flags for legacy GPIO requests.
 */
#define GPIOF_IN		((1 << 0))
#define GPIOF_OUT_INIT_LOW	((0 << 0) | (0 << 1))
#define GPIOF_OUT_INIT_HIGH	((0 << 0) | (1 << 1))

#ifdef CONFIG_GPIOLIB
/**
 * gpio_is_valid - Predicate to check if a GPIO index is within the legal range.
 * Logic: Valid indices are strictly non-negative.
 */
static inline bool gpio_is_valid(int number)
{
	return number >= 0;
}

/**
 * External API: Resource Management.
 */
int gpio_request(unsigned gpio, const char *label);
void gpio_free(unsigned gpio);

/**
 * Block Logic: Directional configuration wrappers.
 * Logic: Maps integer-based GPIO IDs to internal descriptors and invokes 
 * the modern gpiod configuration logic.
 */
static inline int gpio_direction_input(unsigned gpio)
{
	return gpiod_direction_input(gpio_to_desc(gpio));
}
static inline int gpio_direction_output(unsigned gpio, int value)
{
	return gpiod_direction_output_raw(gpio_to_desc(gpio), value);
}

/**
 * Block Logic: Value accessors (Sleep-safe variants).
 */
static inline int gpio_get_value_cansleep(unsigned gpio)
{
	return gpiod_get_raw_value_cansleep(gpio_to_desc(gpio));
}
static inline void gpio_set_value_cansleep(unsigned gpio, int value)
{
	gpiod_set_raw_value_cansleep(gpio_to_desc(gpio), value);
}

/**
 * Block Logic: Value accessors (Atomic/Spinlock-safe variants).
 */
static inline int gpio_get_value(unsigned gpio)
{
	return gpiod_get_raw_value(gpio_to_desc(gpio));
}
static inline void gpio_set_value(unsigned gpio, int value)
{
	gpiod_set_raw_value(gpio_to_desc(gpio), value);
}

/**
 * Functional Utility: Maps a GPIO line to its associated IRQ domain number.
 */
static inline int gpio_to_irq(unsigned gpio)
{
	return gpiod_to_irq(gpio_to_desc(gpio));
}

int gpio_request_one(unsigned gpio, unsigned long flags, const char *label);

/**
 * Functional Utility: Device-managed GPIO acquisition.
 */
int devm_gpio_request_one(struct device *dev, unsigned gpio,
			  unsigned long flags, const char *label);

#else /* ! CONFIG_GPIOLIB */

#include <linux/kernel.h>
#include <asm/bug.h>
#include <asm/errno.h>

/**
 * Block Logic: Stub implementations for non-GPIO enabled kernels.
 */
static inline bool gpio_is_valid(int number)
{
	return false;
}

static inline int gpio_request(unsigned gpio, const char *label)
{
	return -ENOSYS;
}

static inline int gpio_request_one(unsigned gpio,
					unsigned long flags, const char *label)
{
	return -ENOSYS;
}

static inline void gpio_free(unsigned gpio)
{
	might_sleep();
	WARN_ON(1);
}

static inline int gpio_direction_input(unsigned gpio)
{
	return -ENOSYS;
}

static inline int gpio_direction_output(unsigned gpio, int value)
{
	return -ENOSYS;
}

static inline int gpio_get_value(unsigned gpio)
{
	WARN_ON(1);
	return 0;
}

static inline void gpio_set_value(unsigned gpio, int value)
{
	WARN_ON(1);
}

static inline int gpio_get_value_cansleep(unsigned gpio)
{
	WARN_ON(1);
	return 0;
}

static inline void gpio_set_value_cansleep(unsigned gpio, int value)
{
	WARN_ON(1);
}

static inline int gpio_to_irq(unsigned gpio)
{
	WARN_ON(1);
	return -EINVAL;
}

static inline int devm_gpio_request_one(struct device *dev, unsigned gpio,
					unsigned long flags, const char *label)
{
	WARN_ON(1);
	return -EINVAL;
}

#endif /* ! CONFIG_GPIOLIB */
#endif /* CONFIG_GPIOLIB_LEGACY */
#endif /* __LINUX_GPIO_H */
