/* SPDX-License-Identifier: GPL-2.0-or-later */
/**
 * @file
 * @brief MTD (Memory Technology Device) map driver interface
 *
 * This header file provides a generic interface for memory-mapped MTD (flash)
 * devices. It defines the structures and functions necessary to abstract the
 * hardware mapping details from the MTD chip drivers (e.g., CFI, JEDEC).
 *
 * A board driver provides a `struct map_info` describing the physical
 * mapping of the flash device(s), and this map driver layer uses that
 * information to present a consistent interface to the MTD chip drivers.
 * This allows chip drivers to focus on the flash command set without
 * worrying about bus width, interleaving, or virtual memory mapping.
 *
 * Copyright © 2000-2010 David Woodhouse <dwmw2@infradead.org> et al.
 */

/* Overhauled routines for dealing with different mmap regions of flash */

#ifndef __LINUX_MTD_MAP_H__
#define __LINUX_MTD_MAP_H__

#include <linux/types.h>
#include <linux/list.h>
#include <linux/string.h>
#include <linux/bug.h>
#include <linux/kernel.h>
#include <linux/io.h>

#include <linux/unaligned.h>
#include <asm/barrier.h>

/*
 * The following preprocessor macros are used to handle different bus widths
 * (bank widths) of the flash memory mapping. Depending on the kernel
 * configuration (e.g., CONFIG_MTD_MAP_BANK_WIDTH_1), these macros resolve
 * to either constants or expressions that access the `bankwidth` field of
 * `struct map_info`. This allows for compile-time optimization when only a
 * single bank width is supported, while retaining flexibility for runtime
 * configuration.
 */
#ifdef CONFIG_MTD_MAP_BANK_WIDTH_1
#define map_bankwidth(map) 1
#define map_bankwidth_is_1(map) (map_bankwidth(map) == 1)
#define map_bankwidth_is_large(map) (0)
#define map_words(map) (1)
#define MAX_MAP_BANKWIDTH 1
#else
#define map_bankwidth_is_1(map) (0)
#endif

#ifdef CONFIG_MTD_MAP_BANK_WIDTH_2
# ifdef map_bankwidth
#  undef map_bankwidth
#  define map_bankwidth(map) ((map)->bankwidth)
# else
#  define map_bankwidth(map) 2
#  define map_bankwidth_is_large(map) (0)
#  define map_words(map) (1)
# endif
#define map_bankwidth_is_2(map) (map_bankwidth(map) == 2)
#undef MAX_MAP_BANKWIDTH
#define MAX_MAP_BANKWIDTH 2
#else
#define map_bankwidth_is_2(map) (0)
#endif

#ifdef CONFIG_MTD_MAP_BANK_WIDTH_4
# ifdef map_bankwidth
#  undef map_bankwidth
#  define map_bankwidth(map) ((map)->bankwidth)
# else
#  define map_bankwidth(map) 4
#  define map_bankwidth_is_large(map) (0)
#  define map_words(map) (1)
# endif
#define map_bankwidth_is_4(map) (map_bankwidth(map) == 4)
#undef MAX_MAP_BANKWIDTH
#define MAX_MAP_BANKWIDTH 4
#else
#define map_bankwidth_is_4(map) (0)
#endif

/* ensure we never evaluate anything shorted than an unsigned long
 * to zero, and ensure we'll never miss the end of an comparison (bjd) */

#define map_calc_words(map) ((map_bankwidth(map) + (sizeof(unsigned long)-1)) / sizeof(unsigned long))

#ifdef CONFIG_MTD_MAP_BANK_WIDTH_8
# ifdef map_bankwidth
#  undef map_bankwidth
#  define map_bankwidth(map) ((map)->bankwidth)
#  if BITS_PER_LONG < 64
#   undef map_bankwidth_is_large
#   define map_bankwidth_is_large(map) (map_bankwidth(map) > BITS_PER_LONG/8)
#   undef map_words
#   define map_words(map) map_calc_words(map)
#  endif
# else
#  define map_bankwidth(map) 8
#  define map_bankwidth_is_large(map) (BITS_PER_LONG < 64)
#  define map_words(map) map_calc_words(map)
# endif
#define map_bankwidth_is_8(map) (map_bankwidth(map) == 8)
#undef MAX_MAP_BANKWIDTH
#define MAX_MAP_BANKWIDTH 8
#else
#define map_bankwidth_is_8(map) (0)
#endif

#ifdef CONFIG_MTD_MAP_BANK_WIDTH_16
# ifdef map_bankwidth
#  undef map_bankwidth
#  define map_bankwidth(map) ((map)->bankwidth)
#  undef map_bankwidth_is_large
#  define map_bankwidth_is_large(map) (map_bankwidth(map) > BITS_PER_LONG/8)
#  undef map_words
#  define map_words(map) map_calc_words(map)
# else
#  define map_bankwidth(map) 16
#  define map_bankwidth_is_large(map) (1)
#  define map_words(map) map_calc_words(map)
# endif
#define map_bankwidth_is_16(map) (map_bankwidth(map) == 16)
#undef MAX_MAP_BANKWIDTH
#define MAX_MAP_BANKWIDTH 16
#else
#define map_bankwidth_is_16(map) (0)
#endif

#ifdef CONFIG_MTD_MAP_BANK_WIDTH_32
/* always use indirect access for 256-bit to preserve kernel stack */
# undef map_bankwidth
# define map_bankwidth(map) ((map)->bankwidth)
# undef map_bankwidth_is_large
# define map_bankwidth_is_large(map) (map_bankwidth(map) > BITS_PER_LONG/8)
# undef map_words
# define map_words(map) map_calc_words(map)
#define map_bankwidth_is_32(map) (map_bankwidth(map) == 32)
#undef MAX_MAP_BANKWIDTH
#define MAX_MAP_BANKWIDTH 32
#else
#define map_bankwidth_is_32(map) (0)
#endif

#ifndef map_bankwidth
#ifdef CONFIG_MTD
#warning "No CONFIG_MTD_MAP_BANK_WIDTH_xx selected. No NOR chip support can work"
#endif
static inline int map_bankwidth(void *map)
{
	BUG();
	return 0;
}
#define map_bankwidth_is_large(map) (0)
#define map_words(map) (0)
#define MAX_MAP_BANKWIDTH 1
#endif

static inline int map_bankwidth_supported(int w)
{
	switch (w) {
#ifdef CONFIG_MTD_MAP_BANK_WIDTH_1
	case 1:
#endif
#ifdef CONFIG_MTD_MAP_BANK_WIDTH_2
	case 2:
#endif
#ifdef CONFIG_MTD_MAP_BANK_WIDTH_4
	case 4:
#endif
#ifdef CONFIG_MTD_MAP_BANK_WIDTH_8
	case 8:
#endif
#ifdef CONFIG_MTD_MAP_BANK_WIDTH_16
	case 16:
#endif
#ifdef CONFIG_MTD_MAP_BANK_WIDTH_32
	case 32:
#endif
		return 1;

	default:
		return 0;
	}
}

#define MAX_MAP_LONGS (((MAX_MAP_BANKWIDTH * 8) + BITS_PER_LONG - 1) / BITS_PER_LONG)

/**
 * @typedef map_word
 * @brief A data type for holding a single word read from or written to the flash.
 *
 * This union is sized to hold the largest possible bus width, allowing a single
 * data type to be used for flash access regardless of the underlying hardware
 * configuration. The actual number of `unsigned long` elements used depends on
 * the configured `MAX_MAP_BANKWIDTH`.
 */
typedef union {
	unsigned long x[MAX_MAP_LONGS];
} map_word;

/**
 * struct map_info - describes a memory-mapped device
 *
 * @name: A string identifier for this mapping.
 * @size: The total size of the mapped region.
 * @phys: The physical address of the mapping. If set to NO_XIP, it indicates
 *        that the mapping is not linear and cannot be used for eXecute-In-Place.
 * @virt: The kernel virtual address of the mapped region, obtained via ioremap().
 * @cached: A pointer to a cached copy of the flash content in system RAM. This
 *          is used by some drivers to improve read performance.
 * @swap: Specifies the byte-swapping behavior required for the mapping.
 * @bankwidth: The interleave of the flash chips, in bytes. This defines the
 *             stride before addressing wraps around to the first chip again.
 *             It is not necessarily the same as the bus width.
 * @read: A function pointer for reading data from the flash. Used for complex
 *        (non-linear) mappings where a direct `memcpy_fromio` is insufficient.
 * @copy_from: A function pointer for copying a block of data from the flash.
 * @write: A function pointer for writing data to the flash. Used for complex mappings.
 * @copy_to: A function pointer for copying a block of data to the flash.
 * @inval_cache: A function pointer to a routine that invalidates a region of the
 *               cached memory (`@cached`). This is called by chip drivers when
 *               the flash content has been modified.
 * @set_vpp: A function pointer to control the Vpp (programming voltage). The map
 *           core calls this with `1` to enable Vpp and `0` to disable it.
 * @pfow_base: Physical address for "Power-Fail-On-Write" operations.
 * @map_priv_1: Private data for the map driver's use.
 * @map_priv_2: More private data for the map driver.
 * @device_node: Pointer to the device tree node for this mapping.
 * @fldrv_priv: Private data used by the flash chip driver (`fldrv`).
 * @fldrv: A pointer to the `mtd_chip_driver` that has successfully probed this map.
 *
 * This structure is the central point of information for a memory-mapped MTD
 * device. A board-specific driver populates this structure and passes it to
 * the map probe functions (`do_map_probe`) to detect and initialize the
 * underlying flash chip(s).
 */
struct map_info {
	const char *name;
	unsigned long size;
	resource_size_t phys;
#define NO_XIP (-1UL)

	void __iomem *virt;
	void *cached;

	int swap;
	int bankwidth;

#ifdef CONFIG_MTD_COMPLEX_MAPPINGS
	map_word (*read)(struct map_info *, unsigned long);
	void (*copy_from)(struct map_info *, void *, unsigned long, ssize_t);

	void (*write)(struct map_info *, const map_word, unsigned long);
	void (*copy_to)(struct map_info *, unsigned long, const void *, ssize_t);
#endif
	void (*inval_cache)(struct map_info *, unsigned long, ssize_t);
	void (*set_vpp)(struct map_info *, int);

	unsigned long pfow_base;
	unsigned long map_priv_1;
	unsigned long map_priv_2;
	struct device_node *device_node;
	void *fldrv_priv;
	struct mtd_chip_driver *fldrv;
};

/**
 * struct mtd_chip_driver - describes a driver for a particular family of MTD chips
 *
 * @probe: A function pointer that attempts to probe for a supported chip on a
 *         given `map_info`. If successful, it returns a new `mtd_info` struct.
 * @destroy: A function pointer to clean up and free resources allocated by the
 *           probe function when the MTD device is destroyed.
 * @module: A pointer to the kernel module that owns this driver.
 * @name: A string identifier for this chip driver.
 * @list: A list head for linking this driver into the kernel's list of
 *        registered MTD chip drivers.
 */
struct mtd_chip_driver {
	struct mtd_info *(*probe)(struct map_info *map);
	void (*destroy)(struct mtd_info *);
	struct module *module;
	char *name;
	struct list_head list;
};

void register_mtd_chip_driver(struct mtd_chip_driver *);
void unregister_mtd_chip_driver(struct mtd_chip_driver *);

struct mtd_info *do_map_probe(const char *name, struct map_info *map);
void map_destroy(struct mtd_info *mtd);

#define ENABLE_VPP(map) do { if (map->set_vpp) map->set_vpp(map, 1); } while (0)
#define DISABLE_VPP(map) do { if (map->set_vpp) map->set_vpp(map, 0); } while (0)

#define INVALIDATE_CACHED_RANGE(map, from, size) \
	do { if (map->inval_cache) map->inval_cache(map, from, size); } while (0)

#define map_word_equal(map, val1, val2)					\
({									\
	int i, ret = 1;							\
	for (i = 0; i < map_words(map); i++)				\
		if ((val1).x[i] != (val2).x[i]) {			\
			ret = 0;					\
			break;						\
		}							\
	ret;								\
})

#define map_word_and(map, val1, val2)					\
({									\
	map_word r;							\
	int i;								\
	for (i = 0; i < map_words(map); i++)				\
		r.x[i] = (val1).x[i] & (val2).x[i];			\
	r;								\
})

#define map_word_clr(map, val1, val2)					\
({									\
	map_word r;							\
	int i;								\
	for (i = 0; i < map_words(map); i++)				\
		r.x[i] = (val1).x[i] & ~(val2).x[i];			\
	r;								\
})

#define map_word_or(map, val1, val2)					\
({									\
	map_word r;							\
	int i;								\
	for (i = 0; i < map_words(map); i++)				\
		r.x[i] = (val1).x[i] | (val2).x[i];			\
	r;								\
})

#define map_word_andequal(map, val1, val2, val3)			\
({									\
	int i, ret = 1;							\
	for (i = 0; i < map_words(map); i++) {				\
		if (((val1).x[i] & (val2).x[i]) != (val3).x[i]) {	\
			ret = 0;					\
			break;						\
		}							\
	}								\
	ret;								\
})

#define map_word_bitsset(map, val1, val2)				\
({									\
	int i, ret = 0;							\
	for (i = 0; i < map_words(map); i++) {				\
		if ((val1).x[i] & (val2).x[i]) {			\
			ret = 1;					\
			break;						\
		}							\
	}								\
	ret;								\
})

static inline map_word map_word_load(struct map_info *map, const void *ptr)
{
	map_word r;

	if (map_bankwidth_is_1(map))
		r.x[0] = *(unsigned char *)ptr;
	else if (map_bankwidth_is_2(map))
		r.x[0] = get_unaligned((uint16_t *)ptr);
	else if (map_bankwidth_is_4(map))
		r.x[0] = get_unaligned((uint32_t *)ptr);
#if BITS_PER_LONG >= 64
	else if (map_bankwidth_is_8(map))
		r.x[0] = get_unaligned((uint64_t *)ptr);
#endif
	else if (map_bankwidth_is_large(map))
		memcpy(r.x, ptr, map->bankwidth);
	else
		BUG();

	return r;
}

static inline map_word map_word_load_partial(struct map_info *map, map_word orig, const unsigned char *buf, int start, int len)
{
	int i;

	if (map_bankwidth_is_large(map)) {
		char *dest = (char *)&orig;

		memcpy(dest+start, buf, len);
	} else {
		for (i = start; i < start+len; i++) {
			int bitpos;

#ifdef __LITTLE_ENDIAN
			bitpos = i * 8;
#else /* __BIG_ENDIAN */
			bitpos = (map_bankwidth(map) - 1 - i) * 8;
#endif
			orig.x[0] &= ~(0xff << bitpos);
			orig.x[0] |= (unsigned long)buf[i-start] << bitpos;
		}
	}
	return orig;
}

#if BITS_PER_LONG < 64
#define MAP_FF_LIMIT 4
#else
#define MAP_FF_LIMIT 8
#endif

static inline map_word map_word_ff(struct map_info *map)
{
	map_word r;
	int i;

	if (map_bankwidth(map) < MAP_FF_LIMIT) {
		int bw = 8 * map_bankwidth(map);

		r.x[0] = (1UL << bw) - 1;
	} else {
		for (i = 0; i < map_words(map); i++)
			r.x[i] = ~0UL;
	}
	return r;
}

/**
 * The following `inline_map_*` functions provide default implementations
 * for accessing the flash memory on simple, linearly mapped devices.
 * They are used when `CONFIG_MTD_COMPLEX_MAPPINGS` is not defined.
 * For complex mappings (e.g., non-contiguous memory), a board driver
 * must provide its own implementations of the `read`, `write`, `copy_from`,
 * and `copy_to` methods in its `map_info` struct.
 */
static inline map_word inline_map_read(struct map_info *map, unsigned long ofs)
{
	map_word r;

	if (map_bankwidth_is_1(map))
		r.x[0] = __raw_readb(map->virt + ofs);
	else if (map_bankwidth_is_2(map))
		r.x[0] = __raw_readw(map->virt + ofs);
	else if (map_bankwidth_is_4(map))
		r.x[0] = __raw_readl(map->virt + ofs);
#if BITS_PER_LONG >= 64
	else if (map_bankwidth_is_8(map))
		r.x[0] = __raw_readq(map->virt + ofs);
#endif
	else if (map_bankwidth_is_large(map))
		memcpy_fromio(r.x, map->virt + ofs, map->bankwidth);
	else
		BUG();

	return r;
}

static inline void inline_map_write(struct map_info *map, const map_word datum, unsigned long ofs)
{
	if (map_bankwidth_is_1(map))
		__raw_writeb(datum.x[0], map->virt + ofs);
	else if (map_bankwidth_is_2(map))
		__raw_writew(datum.x[0], map->virt + ofs);
	else if (map_bankwidth_is_4(map))
		__raw_writel(datum.x[0], map->virt + ofs);
#if BITS_PER_LONG >= 64
	else if (map_bankwidth_is_8(map))
		__raw_writeq(datum.x[0], map->virt + ofs);
#endif
	else if (map_bankwidth_is_large(map))
		memcpy_toio(map->virt+ofs, datum.x, map->bankwidth);
	else
		BUG();
	mb();
}

static inline void inline_map_copy_from(struct map_info *map, void *to, unsigned long from, ssize_t len)
{
	if (map->cached)
		memcpy(to, (char *)map->cached + from, len);
	else
		memcpy_fromio(to, map->virt + from, len);
}

static inline void inline_map_copy_to(struct map_info *map, unsigned long to, const void *from, ssize_t len)
{
	memcpy_toio(map->virt + to, from, len);
}

#ifdef CONFIG_MTD_COMPLEX_MAPPINGS
#define map_read(map, ofs) (map)->read(map, ofs)
#define map_copy_from(map, to, from, len) (map)->copy_from(map, to, from, len)
#define map_write(map, datum, ofs) (map)->write(map, datum, ofs)
#define map_copy_to(map, to, from, len) (map)->copy_to(map, to, from, len)

extern void simple_map_init(struct map_info *);
#define map_is_linear(map) (map->phys != NO_XIP)

#else
#define map_read(map, ofs) inline_map_read(map, ofs)
#define map_copy_from(map, to, from, len) inline_map_copy_from(map, to, from, len)
#define map_write(map, datum, ofs) inline_map_write(map, datum, ofs)
#define map_copy_to(map, to, from, len) inline_map_copy_to(map, to, from, len)


#define simple_map_init(map) BUG_ON(!map_bankwidth_supported((map)->bankwidth))
#define map_is_linear(map) ({ (void)(map); 1; })

#endif /* !CONFIG_MTD_COMPLEX_MAPPINGS */

#endif /* __LINUX_MTD_MAP_H__ */
