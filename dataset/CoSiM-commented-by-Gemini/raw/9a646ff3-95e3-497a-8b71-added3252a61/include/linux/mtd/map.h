/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * Copyright © 2000-2010 David Woodhouse <dwmw2@infradead.org> et al.
 */

/**
 * @file map.h
 * @brief MTD abstraction for memory-mapped NOR flash devices.
 *
 * This header provides the core data structures and helper functions for generic
 * memory-mapped flash access within the Linux MTD subsystem. It defines the
 * `struct map_info` which describes the mapping properties (address, size, bus
 * width) and provides function pointers for read/write operations.
 *
 * The file uses extensive pre-processor macros to optimize for different bus
 * widths (bankwidth) at compile time, allowing for efficient access to NOR
 * flash chips on various hardware platforms. It supports both simple linear
 * mappings (suitable for XIP) and complex, non-linear mappings.
 */

#ifndef __LINUX_MTD_MAP_H__
#define __LINUX_MTD_MAP_H__

#include <linux/bug.h>
#include <linux/io.h>
#include <linux/ioport.h>
#include <linux/string.h>
#include <linux/types.h>
#include <linux/unaligned.h>

struct device_node;
struct module;

/*
 * Pre-processor hell: The following macros are used to define the 'bankwidth'
 * of the flash mapping at compile time. This allows the compiler to generate
 * optimised code for accessing the flash, without runtime checks.
 * For any given build, only one of these options is likely to be selected.
 * If more than one are selected, it's a 'complex mapping' and the bankwidth
 * is determined at runtime from the 'bankwidth' field in struct map_info.
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

/*
 * For bankwidths greater than the native long size, we need to use an array
 * of 'unsigned long's to store the data. The 'map_words' macro calculates
 * how many 'unsigned long's are required.
 */
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

/**
 * map_bankwidth_supported() - Check if a given bankwidth is supported by the build
 * @w: The bankwidth in bytes.
 *
 * This function returns true if the kernel was compiled with support for the
 * specified bankwidth, and false otherwise.
 */
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

/**
 * @brief The maximum number of `unsigned long`s needed to represent a read/write
 * from/to the flash bus.
 *
 * This is used to size the `map_word` union.
 */
#define MAX_MAP_LONGS (((MAX_MAP_BANKWIDTH * 8) + BITS_PER_LONG - 1) / BITS_PER_LONG)

/**
 * @brief A union to hold data of the flash's bus width.
 *
 * This union is used to handle reads and writes to the flash. For bus widths
 * up to the processor's word size, the data is held in a single `unsigned long`.
 * For wider buses, an array of `unsigned long`s is used. This abstracts away
 * the bus width from the chip driver logic.
 */
typedef union {
	unsigned long x[MAX_MAP_LONGS];
} map_word;

struct mtd_chip_driver;

/**
 * @brief Describes a memory-mapped MTD device.
 *
 * An instance of this structure is created by a map driver (e.g., physmap)
 * to describe the memory mapping of a NOR flash device. This structure is then
 * passed to a chip driver (e.g., cfi_probe) to discover and initialize the
 * actual MTD device.
 */
struct map_info {
	/** @brief The name of this mapping. */
	const char *name;
	/** @brief The total size of the mapping in bytes. */
	unsigned long size;
	/**
	 * @brief The physical address of the mapping.
	 *
	 * If the mapping is linear and suitable for eXecute-In-Place (XIP),
	 * this holds the physical address. Otherwise, it should be set to
	 * `NO_XIP`.
	 */
	resource_size_t phys;
#define NO_XIP (-1UL)

	/** @brief The virtual address for ioremap'ped flash access. */
	void __iomem *virt;
	/**
	 * @brief A pointer to a cached, CPU-accessible copy of the flash contents.
	 *
	 * This is used on platforms where direct I/O access is slow, allowing
	 * `copy_from` operations to be satisfied from a RAM cache. The
	 * `inval_cache` callback must be used to keep this consistent.
	 */
	void *cached;

	/** @brief The byte-swapping behavior of the mapping. */
	int swap;
	/**
	 * @brief The width of the flash bus in bytes (octets).
	 *
	 * This determines the size of each read/write operation and the repeat
	 * interval before accessing the same chip in an interleaved setup.
	 */
	int bankwidth;

#ifdef CONFIG_MTD_COMPLEX_MAPPINGS
	/**
	 * @brief A function to read a `map_word` from a given offset.
	 *
	 * This is required for non-linear or complex mappings where a simple
	 * `memcpy_fromio` is not sufficient.
	 */
	map_word (*read)(struct map_info *, unsigned long);
	/**
	 * @brief A function to copy a block of data from the flash.
	 *
	 * Required for complex mappings.
	 */
	void (*copy_from)(struct map_info *, void *, unsigned long, ssize_t);

	/**
	 * @brief A function to write a `map_word` to a given offset.
	 *
	 * Required for non-linear or complex mappings.
	 */
	void (*write)(struct map_info *, const map_word, unsigned long);
	/**
	 * @brief A function to copy a block of data to the flash.
	 *
	 * Required for complex mappings.
	 */
	void (*copy_to)(struct map_info *, unsigned long, const void *, ssize_t);
#endif
	/**
	 * @brief A function to invalidate a region of the CPU cache.
	 *
	 * If the map driver uses a RAM cache (`map_info->cached`), this
	 * function must be provided to invalidate parts of that cache when
	 * the flash content is modified by a chip-level operation (e.g., erase
	 * or program).
	 */
	void (*inval_cache)(struct map_info *, unsigned long, ssize_t);

	/**
	 * @brief A function to enable or disable the programming voltage (Vpp).
	 *
	 * This is called by the MTD core when write or erase operations are
	 * about to start or have finished. The map driver should implement this
	 * if the board requires explicit Vpp control.
	 */
	void (*set_vpp)(struct map_info *, int);

	/** @brief Platform-specific data fields. */
	unsigned long pfow_base;
	unsigned long map_priv_1;
	unsigned long map_priv_2;
	/** @brief The device tree node associated with this mapping. */
	struct device_node *device_node;
	/** @brief Private data for the flash chip driver. */
	void *fldrv_priv;
	/** @brief The flash chip driver associated with this map. */
	struct mtd_chip_driver *fldrv;
};

/**
 * @brief Represents a driver for a family of flash chips (e.g., CFI, JEDEC).
 *
 * Chip drivers are registered with the map core and are responsible for
 * probing for specific flash chips on a given `map_info` and creating an
 * `mtd_info` structure if a compatible chip is found.
 */
struct mtd_chip_driver {
	/**
	 * @brief Probes for a flash chip on the given map.
	 * @param map The memory map to probe.
	 * @return A pointer to a new `mtd_info` structure on success, or NULL.
	 */
	struct mtd_info *(*probe)(struct map_info *map);
	/**
	 * @brief Destroys an `mtd_info` structure created by this driver.
	 * @param mtd The MTD info structure to destroy.
	 */
	void (*destroy)(struct mtd_info *);
	/** @brief The module that owns this chip driver. */
	struct module *module;
	/** @brief The name of the chip driver. */
	char *name;
	/** @brief The list head for linking into the core's driver list. */
	struct list_head list;
};

void register_mtd_chip_driver(struct mtd_chip_driver *);
void unregister_mtd_chip_driver(struct mtd_chip_driver *);

struct mtd_info *do_map_probe(const char *name, struct map_info *map);
void map_destroy(struct mtd_info *mtd);

/** @brief Helper macro to enable Vpp if the map supports it. */
#define ENABLE_VPP(map) do { if (map->set_vpp) map->set_vpp(map, 1); } while (0)
/** @brief Helper macro to disable Vpp if the map supports it. */
#define DISABLE_VPP(map) do { if (map->set_vpp) map->set_vpp(map, 0); } while (0)

/** @brief Helper macro to invalidate the cache for a given range. */
#define INVALIDATE_CACHED_RANGE(map, from, size) 
	do { if (map->inval_cache) map->inval_cache(map, from, size); } while (0)

/**
 * @defgroup map_word_ops Bitwise operations on map_word
 * @{
 * These macros provide a bus-width-agnostic way to perform bitwise logic on
 * `map_word` objects. They iterate over the `unsigned long` array within the
 * `map_word` union, handling bus widths larger than the native word size.
 */

/** @brief Compare two map_words for equality. */
#define map_word_equal(map, val1, val2)					
({									
	int i, ret = 1;							
	for (i = 0; i < map_words(map); i++)				
		if ((val1).x[i] != (val2).x[i]) {			
			ret = 0;					
			break;						
		}							
	ret;								
})

/** @brief Perform a bitwise AND on two map_words. */
#define map_word_and(map, val1, val2)					
({									
	map_word r;							
	int i;								
	for (i = 0; i < map_words(map); i++)				
		r.x[i] = (val1).x[i] & (val2).x[i];			
	r;								
})

/** @brief Perform a bitwise clear (AND with NOT) on two map_words. */
#define map_word_clr(map, val1, val2)					
({									
	map_word r;							
	int i;								
	for (i = 0; i < map_words(map); i++)				
		r.x[i] = (val1).x[i] & ~(val2).x[i];			
	r;								
})

/** @brief Perform a bitwise OR on two map_words. */
#define map_word_or(map, val1, val2)					
({									
	map_word r;							
	int i;								
	for (i = 0; i < map_words(map); i++)				
		r.x[i] = (val1).x[i] | (val2).x[i];			
	r;								
})

/** @brief Check if `(val1 & val2) == val3`. */
#define map_word_andequal(map, val1, val2, val3)			
({									
	int i, ret = 1;							
	for (i = 0; i < map_words(map); i++) {				
		if (((val1).x[i] & (val2).x[i]) != (val3).x[i]) {	
			ret = 0;					
			break;						
		}							
	}								
	ret;								
})

/** @brief Check if any bits are set in `(val1 & val2)`. */
#define map_word_bitsset(map, val1, val2)				
({									
	int i, ret = 0;							
	for (i = 0; i < map_words(map); i++) {				
		if ((val1).x[i] & (val2).x[i]) {			
			ret = 1;					
			break;						
		}							
	}								
	ret;								
})
/** @} */

/**
 * @brief Load a `map_word` from a buffer in memory.
 * @param map The map_info structure.
 * @param ptr The source buffer.
 * @return The loaded `map_word`.
 *
 * This function handles unaligned access and different bankwidths correctly.
 */
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

/**
 * @brief Overwrite a portion of a `map_word` with data from a buffer.
 * @param map The map_info structure.
 * @param orig The original `map_word` to modify.
 * @param buf The source buffer.
 * @param start The starting byte offset within the `map_word`.
 * @param len The number of bytes to copy.
 * @return The modified `map_word`.
 *
 * Functional Utility: This is used for partial writes, for example, when a
 * user write does not align to the flash bus width. It performs a
 * read-modify-write cycle at the `map_word` level.
 */
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

/**
 * @brief Create a `map_word` with all bits set to 1.
 * @param map The map_info structure.
 * @return A `map_word` full of 0xFF bytes.
 *
 * This is used to check for erased flash content.
 */
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
 * @brief Read a `map_word` from a linearly mapped flash device.
 * @param map The map_info structure.
 * @param ofs The offset within the flash mapping.
 * @return The `map_word` read from flash.
 *
 * This is the inline implementation for simple, linear mappings. It uses
 * raw I/O accessors (`__raw_readb/w/l/q`).
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

/**
 * @brief Write a `map_word` to a linearly mapped flash device.
 * @param map The map_info structure.
 * @param datum The `map_word` to write.
 * @param ofs The offset within the flash mapping.
 *
 * This is the inline implementation for simple, linear mappings. It uses
 * raw I/O accessors and issues a memory barrier.
 */
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

/**
 * @brief Copy a block of data from a linearly mapped flash device.
 * @param map The map_info structure.
 * @param to The destination buffer in RAM.
 * @param from The source offset within the flash mapping.
 * @param len The number of bytes to copy.
 *
 * This function will use the `cached` RAM buffer if available, otherwise it
 * will copy directly from I/O memory.
 */
static inline void inline_map_copy_from(struct map_info *map, void *to, unsigned long from, ssize_t len)
{
	if (map->cached)
		memcpy(to, (char *)map->cached + from, len);
	else
		memcpy_fromio(to, map->virt + from, len);
}

/**
 * @brief Copy a block of data to a linearly mapped flash device.
 * @param map The map_info structure.
 * @param to The destination offset within the flash mapping.
 * @param from The source buffer in RAM.
 * @param len The number of bytes to copy.
 */
static inline void inline_map_copy_to(struct map_info *map, unsigned long to, const void *from, ssize_t len)
{
	memcpy_toio(map->virt + to, from, len);
}

/*
 * If complex mappings are enabled, the map_{read,write,copy_*} macros
 * are defined to call the function pointers in struct map_info. Otherwise,
 * they are defined to call the inline_* versions directly for performance.
 */
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
