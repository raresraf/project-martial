/**
 * @file module.h
 * @brief Core definitions for Linux kernel modules.
 *
 * This header file provides the fundamental data structures, macros, and API
 * for implementing dynamic loadable kernel modules in Linux. It defines
 * how modules interact with the kernel, including their initialization,
 * cleanup, parameter handling, symbol export/import, and integration
 * with various kernel subsystems like sysfs and kallsyms.
 *
 * Functional Utility:
 * - Defines the `module` structure, representing a loaded kernel module.
 * - Provides `module_init` and `module_exit` macros for module entry points.
 * - Supports module parameters (`module_param`).
 * - Enables symbol export (`EXPORT_SYMBOL`) and import (`symbol_get`).
 * - Integrates with the kobject and sysfs infrastructure for user-space interaction.
 * - Handles module dependencies and versioning.
 * - Facilitates module unloading and reference counting.
 *
 * Architectural Intent:
 * - To allow kernel functionality to be added or removed at runtime, reducing
 *   kernel size and enabling flexible system configurations.
 * - To provide a robust and secure framework for extending kernel capabilities
 *   while maintaining system stability.
 *
 * Rewritten by Richard Henderson <rth@tamu.edu> Dec 1996
 * Rewritten again by Rusty Russell, 2002
 */
/* SPDX-License-Identifier: GPL-2.0-only */
#ifndef _LINUX_MODULE_H
#define _LINUX_MODULE_H

#include <linux/list.h>
#include <linux/stat.h>
#include <linux/buildid.h>
#include <linux/compiler.h>
#include <linux/cache.h>
#include <linux/kmod.h>
#include <linux/init.h>
#include <linux/elf.h>
#include <linux/stringify.h>
#include <linux/kobject.h>
#include <linux/moduleparam.h>
#include <linux/jump_label.h>
#include <linux/export.h>
#include <linux/rbtree_latch.h>
#include <linux/error-injection.h>
#include <linux/tracepoint-defs.h>
#include <linux/srcu.h>
#include <linux/static_call_types.h>
#include <linux/dynamic_debug.h>

#include <linux/percpu.h>
#include <asm/module.h>

/**
 * @def MODULE_NAME_LEN
 * @brief Maximum length of a module's name.
 *
 * This constant defines the buffer size for storing module names,
 * derived from `MAX_PARAM_PREFIX_LEN`.
 */
#define MODULE_NAME_LEN MAX_PARAM_PREFIX_LEN

/**
 * @struct modversion_info
 * @brief Stores module versioning information for symbol checking.
 *
 * This structure is used to verify symbol compatibility between modules
 * and the kernel, or between different modules.
 */
struct modversion_info {
	unsigned long crc;		/**< @brief CRC (Cyclic Redundancy Check) of the symbol. */
	char name[MODULE_NAME_LEN];	/**< @brief Name of the symbol. */
};

struct module;
struct exception_table_entry;

/**
 * @struct module_kobject
 * @brief Represents a kobject for a kernel module in sysfs.
 *
 * This structure embeds a `kobject` and provides module-specific fields
 * for sysfs representation and module parameter management.
 */
struct module_kobject {
	struct kobject kobj;		/**< @brief The embedded kobject. */
	struct module *mod;		/**< @brief Pointer to the associated module. */
	struct kobject *drivers_dir;	/**< @brief Pointer to the 'drivers' kobject directory. */
	struct module_param_attrs *mp;	/**< @brief Module parameter attributes. */
	struct completion *kobj_completion; /**< @brief Completion for kobject release. */
} __randomize_layout;

/**
 * @struct module_attribute
 * @brief Generic attribute for a module.
 *
 * This structure defines a generic attribute that can be exposed via sysfs
 * for a kernel module, including show, store, setup, test, and free operations.
 */
struct module_attribute {
	struct attribute attr;		/**< @brief The base attribute. */
	ssize_t (*show)(const struct module_attribute *, struct module_kobject *,
			char *);	/**< @brief Callback to show attribute value. */
	ssize_t (*store)(const struct module_attribute *, struct module_kobject *,
			 const char *, size_t count); /**< @brief Callback to store attribute value. */
	void (*setup)(struct module *, const char *); /**< @brief Callback for attribute setup. */
	int (*test)(struct module *);	/**< @brief Callback for attribute test. */
	void (*free)(struct module *);	/**< @brief Callback to free attribute resources. */
};

/**
 * @struct module_version_attribute
 * @brief Module attribute for version information.
 *
 * A specialized `module_attribute` for exposing module version strings
 * in sysfs.
 */
struct module_version_attribute {
	struct module_attribute mattr;	/**< @brief The base module attribute. */
	const char *module_name;	/**< @brief Name of the module. */
	const char *version;		/**< @brief Version string of the module. */
};

/**
 * @brief Function to display the module version.
 * @param att The module attribute.
 * @param kobj The module kobject.
 * @param buf Output buffer.
 * @return Number of bytes written to buffer.
 */
extern ssize_t __modver_version_show(const struct module_attribute *,
				     struct module_kobject *, char *);

/**
 * @brief Module attribute for uevent handling.
 */
extern const struct module_attribute module_uevent;

/* These are either module local, or the kernel's dummy ones. */
/**
 * @brief Module initialization function prototype.
 *
 * This is the standard entry point for a kernel module.
 */
extern int init_module(void);
/**
 * @brief Module cleanup function prototype.
 *
 * This is the standard exit point for a kernel module.
 */
extern void cleanup_module(void);

#ifndef MODULE
/**
 * @def module_init(x)
 * @brief driver initialization entry point
 * @param x: function to be run at kernel boot time or module insertion
 *
 * `module_init()` will either be called during `do_initcalls()` (if
 * builtin) or at module insertion time (if a module). There can only
 * be one per module.
 */
#define module_init(x)	__initcall(x);

/**
 * @def module_exit(x)
 * @brief driver exit entry point
 * @param x: function to be run when driver is removed
 *
 * `module_exit()` will wrap the driver clean-up code
 * with `cleanup_module()` when used with `rmmod` when
 * the driver is a module. If the driver is statically
 * compiled into the kernel, `module_exit()` has no effect.
 * There can only be one per module.
 */
#define module_exit(x)	__exitcall(x);

#else /* MODULE */

/*
 * In most cases loadable modules do not need custom
 * initcall levels. There are still some valid cases where
 * a driver may be needed early if built in, and does not
 * matter when built as a loadable module. Like bus
 * snooping debug drivers.
 */
/**
 * @def early_initcall(fn)
 * @brief Macro for early initialization call, maps to `module_init` for modules.
 */
#define early_initcall(fn)		module_init(fn)
/**
 * @def core_initcall(fn)
 * @brief Macro for core initialization call, maps to `module_init` for modules.
 */
#define core_initcall(fn)		module_init(fn)
/**
 * @def core_initcall_sync(fn)
 * @brief Macro for synchronous core initialization call, maps to `module_init` for modules.
 */
#define core_initcall_sync(fn)		module_init(fn)
/**
 * @def postcore_initcall(fn)
 * @brief Macro for post-core initialization call, maps to `module_init` for modules.
 */
#define postcore_initcall(fn)		module_init(fn)
/**
 * @def postcore_initcall_sync(fn)
 * @brief Macro for synchronous post-core initialization call, maps to `module_init` for modules.
 */
#define postcore_initcall_sync(fn)	module_init(fn)
/**
 * @def arch_initcall(fn)
 * @brief Macro for architecture-specific initialization call, maps to `module_init` for modules.
 */
#define arch_initcall(fn)		module_init(fn)
/**
 * @def subsys_initcall(fn)
 * @brief Macro for subsystem initialization call, maps to `module_init` for modules.
 */
#define subsys_initcall(fn)		module_init(fn)
/**
 * @def subsys_initcall_sync(fn)
 * @brief Macro for synchronous subsystem initialization call, maps to `module_init` for modules.
 */
#define subsys_initcall_sync(fn)	module_init(fn)
/**
 * @def fs_initcall(fn)
 * @brief Macro for filesystem initialization call, maps to `module_init` for modules.
 */
#define fs_initcall(fn)			module_init(fn)
/**
 * @def fs_initcall_sync(fn)
 * @brief Macro for synchronous filesystem initialization call, maps to `module_init` for modules.
 */
#define fs_initcall_sync(fn)		module_init(fn)
/**
 * @def rootfs_initcall(fn)
 * @brief Macro for root filesystem initialization call, maps to `module_init` for modules.
 */
#define rootfs_initcall(fn)		module_init(fn)
/**
 * @def device_initcall(fn)
 * @brief Macro for device initialization call, maps to `module_init` for modules.
 */
#define device_initcall(fn)		module_init(fn)
/**
 * @def device_initcall_sync(fn)
 * @brief Macro for synchronous device initialization call, maps to `module_init` for modules.
 */
#define device_initcall_sync(fn)	module_init(fn)
/**
 * @def late_initcall(fn)
 * @brief Macro for late initialization call, maps to `module_init` for modules.
 */
#define late_initcall(fn)		module_init(fn)
/**
 * @def late_initcall_sync(fn)
 * @brief Macro for synchronous late initialization call, maps to `module_init` for modules.
 */
#define late_initcall_sync(fn)		module_init(fn)

/**
 * @def console_initcall(fn)
 * @brief Macro for console initialization call, maps to `module_init` for modules.
 */
#define console_initcall(fn)		module_init(fn)

/* Each module must use one module_init(). */
/**
 * @def module_init(initfn)
 * @brief Macro to define a module's initialization function.
 * @param initfn The function to be called when the module is loaded.
 *
 * For loadable modules, this macro sets up `init_module` as an alias
 * to `initfn`.
 */
#define module_init(initfn)					\
	static inline initcall_t __maybe_unused __inittest(void)		\
	{ return initfn; }					\
	int init_module(void) __copy(initfn)			\
		__attribute__((alias(#initfn)));		\
	___ADDRESSABLE(init_module, __initdata);

/* This is only required if you want to be unloadable. */
/**
 * @def module_exit(exitfn)
 * @brief Macro to define a module's cleanup function.
 * @param exitfn The function to be called when the module is unloaded.
 *
 * For loadable modules, this macro sets up `cleanup_module` as an alias
 * to `exitfn`.
 */
#define module_exit(exitfn)					\
	static inline exitcall_t __maybe_unused __exittest(void)		\
	{ return exitfn; }					\
	void cleanup_module(void) __copy(exitfn)		\
		__attribute__((alias(#exitfn)));		\
	___ADDRESSABLE(cleanup_module, __exitdata);

#endif

/* This means "can be init if no module support, otherwise module load
   may call it." */
#ifdef CONFIG_MODULES
/**
 * @def __init_or_module
 * @brief Denotes code sections that are part of init code or module code.
 */
#define __init_or_module
/**
 * @def __initdata_or_module
 * @brief Denotes data sections that are part of init data or module data.
 */
#define __initdata_or_module
/**
 * @def __initconst_or_module
 * @brief Denotes constant data sections that are part of init data or module data.
 */
#define __initconst_or_module
/**
 * @def __INIT_OR_MODULE
 * @brief Specifies the text section for init/module code.
 */
#define __INIT_OR_MODULE	.text
/**
 * @def __INITDATA_OR_MODULE
 * @brief Specifies the data section for init/module data.
 */
#define __INITDATA_OR_MODULE	.data
/**
 * @def __INITRODATA_OR_MODULE
 * @brief Specifies the read-only data section for init/module data.
 */
#define __INITRODATA_OR_MODULE	.section ".rodata","a",%progbits
#else
// Block Logic: If modules are not configured, these macros map to standard init sections.
#define __init_or_module __init
#define __initdata_or_module __initdata
#define __initconst_or_module __initconst
#define __INIT_OR_MODULE __INIT
#define __INITDATA_OR_MODULE __INITDATA
#define __INITRODATA_OR_MODULE __INITRODATA
#endif /*CONFIG_MODULES*/

/**
 * @brief Looks up or creates a `module_kobject` for a given module name.
 * @param name The name of the module.
 * @return Pointer to the `module_kobject` or NULL on failure.
 */
struct module_kobject *lookup_or_create_module_kobject(const char *name);

/* Generic info of form tag = "info" */
/**
 * @def MODULE_INFO(tag, info)
 * @brief Macro to add generic module information.
 * @param tag The tag for the information (e.g., "alias", "license").
 * @param info The information string.
 */
#define MODULE_INFO(tag, info) __MODULE_INFO(tag, tag, info)

/* For userspace: you can also call me... */
/**
 * @def MODULE_ALIAS(_alias)
 * @brief Macro to add a module alias.
 * @param _alias The alias string.
 *
 * Module aliases are used by `modprobe` to load modules based on device IDs
 * or other identifiers.
 */
#define MODULE_ALIAS(_alias) MODULE_INFO(alias, _alias)

/* Soft module dependencies. See man modprobe.d for details.
 * Example: MODULE_SOFTDEP("pre: module-foo module-bar post: module-baz")
 */
/**
 * @def MODULE_SOFTDEP(_softdep)
 * @brief Macro to declare soft module dependencies.
 * @param _softdep The dependency string, e.g., "pre: module-foo".
 *
 * Soft dependencies are hints to the module loader but are not strictly enforced.
 */
#define MODULE_SOFTDEP(_softdep) MODULE_INFO(softdep, _softdep)

/*
 * Weak module dependencies. See man modprobe.d for details.
 * Example: MODULE_WEAKDEP("module-foo")
 */
/**
 * @def MODULE_WEAKDEP(_weakdep)
 * @brief Macro to declare weak module dependencies.
 * @param _weakdep The weak dependency string.
 *
 * Weak dependencies are similar to soft dependencies but indicate a less
 * stringent requirement.
 */
#define MODULE_WEAKDEP(_weakdep) MODULE_INFO(weakdep, _weakdep)

/*
 * MODULE_FILE is used for generating modules.builtin
 * So, make it no-op when this is being built as a module
 */
#ifdef MODULE
/**
 * @def MODULE_FILE
 * @brief Macro to record the module's source file for built-in modules.
 *
 * This macro is primarily used when generating the `modules.builtin` file
 * and is a no-op when compiled as a loadable module.
 */
#define MODULE_FILE
#else
// Block Logic: When not building a module, this records the module file.
#define MODULE_FILE	MODULE_INFO(file, KBUILD_MODFILE);
#endif

/*
 * The following license idents are currently accepted as indicating free
 * software modules
 *
 *	"GPL"				[GNU Public License v2]
 *	"GPL v2"			[GNU Public License v2]
 *	"GPL and additional rights"	[GNU Public License v2 rights and more]
 *	"Dual BSD/GPL"			[GNU Public License v2
 *					 or BSD license choice]
 *	"Dual MIT/GPL"			[GNU Public License v2
 *					 or MIT license choice]
 *	"Dual MPL/GPL"			[GNU Public License v2
 *					 or Mozilla license choice]
 *
 * The following other idents are available
 *
 *	"Proprietary"			[Non free products]
 *
 * Both "GPL v2" and "GPL" (the latter also in dual licensed strings) are
 * merely stating that the module is licensed under the GPL v2, but are not
 * telling whether "GPL v2 only" or "GPL v2 or later". The reason why there
 * are two variants is a historic and failed attempt to convey more
 * information in the MODULE_LICENSE string. For module loading the
 * "only/or later" distinction is completely irrelevant and does neither
 * replace the proper license identifiers in the corresponding source file
 * nor amends them in any way. The sole purpose is to make the
 * 'Proprietary' flagging work and to refuse to bind symbols which are
 * exported with EXPORT_SYMBOL_GPL when a non free module is loaded.
 *
 * In the same way "BSD" is not a clear license information. It merely
 * states, that the module is licensed under one of the compatible BSD
 * license variants. The detailed and correct license information is again
 * to be found in the corresponding source files.
 *
 * There are dual licensed components, but when running with Linux it is the
 * GPL that is relevant so this is a non issue. Similarly LGPL linked with GPL
 * is a GPL combined work.
 *
 * This exists for several reasons
 * 1.	So modinfo can show license info for users wanting to vet their setup
 *	is free
 * 2.	So the community can ignore bug reports including proprietary modules
 * 3.	So vendors can do likewise based on their own policies
 */
/**
 * @def MODULE_LICENSE(_license)
 * @brief Macro to declare the module's license.
 * @param _license The license string (e.g., "GPL", "Dual BSD/GPL").
 *
 * This information is exposed via `modinfo` and is used by the kernel
 * to determine symbol visibility and enforce GPL-only symbol rules.
 */
#define MODULE_LICENSE(_license) MODULE_FILE MODULE_INFO(license, _license)

/*
 * Author(s), use "Name <email>" or just "Name", for multiple
 * authors use multiple MODULE_AUTHOR() statements/lines.
 */
/**
 * @def MODULE_AUTHOR(_author)
 * @brief Macro to declare a module's author.
 * @param _author The author's name and optional email (e.g., "John Doe <john.doe@example.com>").
 */
#define MODULE_AUTHOR(_author) MODULE_INFO(author, _author)

/* What your module does. */
/**
 * @def MODULE_DESCRIPTION(_description)
 * @brief Macro to provide a brief description of the module's functionality.
 * @param _description The description string.
 */
#define MODULE_DESCRIPTION(_description) MODULE_INFO(description, _description)

#ifdef MODULE
/* Creates an alias so file2alias.c can find device table. */
/**
 * @def MODULE_DEVICE_TABLE(type, name)
 * @brief Macro to export a module's device table.
 * @param type The type of the device table (e.g., `of`, `pci`).
 * @param name The name of the device table array.
 *
 * This macro is used to make a module's device table discoverable by the
 * kernel's module loading mechanisms, typically for hotplugging or auto-loading.
 */
#define MODULE_DEVICE_TABLE(type, name)					\
static typeof(name) __mod_device_table__##type##__##name		\
  __attribute__ ((used, alias(__stringify(name))))
#else  /* !MODULE */
// Block Logic: When not building a module, this macro is a no-op.
#define MODULE_DEVICE_TABLE(type, name)
#endif

/* Version of form [<epoch>:]<version>[-<extra-version>].
 * Or for CVS/RCS ID version, everything but the number is stripped.
 * <epoch>: A (small) unsigned integer which allows you to start versions
 * anew. If not mentioned, it's zero.  eg. "2:1.0" is after
 * "1:2.0".

 * <version>: The <version> may contain only alphanumerics and the
 * character `.'.  Ordered by numeric sort for numeric parts,
 * ascii sort for ascii parts (as per RPM or DEB algorithm).

 * <extraversion>: Like <version>, but inserted for local
 * customizations, eg "rh3" or "rusty1".

 * Using this automatically adds a checksum of the .c files and the
 * local headers in "srcversion".
 */

#if defined(MODULE) || !defined(CONFIG_SYSFS)
/**
 * @def MODULE_VERSION(_version)
 * @brief Macro to declare the module's version string.
 * @param _version The version string.
 *
 * This version information is used for module dependency checking
 * and is exposed via `modinfo`.
 */
#define MODULE_VERSION(_version) MODULE_INFO(version, _version)
#else
// Block Logic: When not a module or sysfs is enabled, a `module_version_attribute` is created.
#define MODULE_VERSION(_version)					\
	MODULE_INFO(version, _version);					\
	static const struct module_version_attribute __modver_attr	\
		__used __section("__modver")				\
		__aligned(__alignof__(struct module_version_attribute)) \
		= {							\
			.mattr	= {					\
				.attr	= {				\
					.name	= "version",		\
					.mode	= S_IRUGO,		\
				},					\
				.show	= __modver_version_show,	\
			},						\
			.module_name	= KBUILD_MODNAME,		\
			.version	= _version,			\
		}
#endif

/* Optional firmware file (or files) needed by the module
 * format is simply firmware file name.  Multiple firmware
 * files require multiple MODULE_FIRMWARE() specifiers */
/**
 * @def MODULE_FIRMWARE(_firmware)
 * @brief Macro to declare required firmware files for a module.
 * @param _firmware The path to the firmware file.
 *
 * The kernel's firmware loader uses this information to find and load
 * necessary firmware before the module is initialized.
 */
#define MODULE_FIRMWARE(_firmware) MODULE_INFO(firmware, _firmware)

/**
 * @def MODULE_IMPORT_NS(ns)
 * @brief Macro to import a kernel symbol namespace.
 * @param ns The name of the namespace to import.
 *
 * This allows a module to access symbols exported under a specific namespace.
 */
#define MODULE_IMPORT_NS(ns)	MODULE_INFO(import_ns, ns)

struct notifier_block;

#ifdef CONFIG_MODULES

/**
 * @brief Retrieves the address of a kernel symbol.
 * @param symbol The name of the symbol.
 * @return Pointer to the symbol's address.
 *
 * This function increments the reference count of the module exporting
 * the symbol. It is primarily for internal kernel use; `symbol_get()`
 * is the preferred user-facing macro.
 */
void *__symbol_get(const char *symbol);
/**
 * @brief Retrieves the address of a GPL-only kernel symbol.
 * @param symbol The name of the GPL-only symbol.
 * @return Pointer to the symbol's address.
 *
 * Similar to `__symbol_get`, but specifically for symbols
 * exported with `EXPORT_SYMBOL_GPL`.
 */
void *__symbol_get_gpl(const char *symbol);
/**
 * @def symbol_get(x)
 * @brief Macro to retrieve a kernel symbol's address and increment its module's reference count.
 * @param x The symbol name.
 * @return The address of the symbol.
 *
 * Calls to `symbol_get()` must be balanced with `symbol_put()`.
 */
#define symbol_get(x)	({ \
	static const char __notrim[] \
		__used __section(".no_trim_symbol") = __stringify(x); \
	(typeof(&x))(__symbol_get(__stringify(x))); })

/* modules using other modules: kdb wants to see this. */
/**
 * @struct module_use
 * @brief Tracks module dependencies.
 *
 * This structure records which modules depend on other modules.
 */
struct module_use {
	struct list_head source_list;	/**< @brief List head for modules that use this module. */
	struct list_head target_list;	/**< @brief List head for modules this module uses. */
	struct module *source, *target;	/**< @brief Pointers to the source and target modules. */
};

/**
 * @enum module_state
 * @brief Enumeration of possible states for a kernel module.
 */
enum module_state {
	MODULE_STATE_LIVE,	/**< @brief Normal state: module is fully loaded and operational. */
	MODULE_STATE_COMING,	/**< @brief Module is fully formed, running module_init. */
	MODULE_STATE_GOING,	/**< @brief Going away. */
	MODULE_STATE_UNFORMED,	/**< @brief Still setting it up. */
};

/**
 * @struct mod_tree_node
 * @brief Node for a module in the module address tree.
 *
 * This structure is used for efficient lookup of modules by address
 * within an rbtree (red-black tree) or similar data structure.
 */
struct mod_tree_node {
	struct module *mod;		/**< @brief Pointer to the module. */
	struct latch_tree_node node;	/**< @brief The latch tree node. */
};

/**
 * @enum mod_mem_type
 * @brief Enumeration of different memory sections within a module.
 */
enum mod_mem_type {
	MOD_TEXT = 0,		/**< @brief Executable code section. */
	MOD_DATA,		/**< @brief Initialized data section. */
	MOD_RODATA,		/**< @brief Read-only data section. */
	MOD_RO_AFTER_INIT,	/**< @brief Read-only data that persists after init. */
	MOD_INIT_TEXT,		/**< @brief Executable code section for initialization. */
	MOD_INIT_DATA,		/**< @brief Initialized data section for initialization. */
	MOD_INIT_RODATA,	/**< @brief Read-only data section for initialization. */

	MOD_MEM_NUM_TYPES,	/**< @brief Total number of memory types. */
	MOD_INVALID = -1,	/**< @brief Invalid memory type. */
};

/**
 * @def mod_mem_type_is_init(type)
 * @brief Checks if a module memory type is part of the initialization sections.
 * @param type The `mod_mem_type` to check.
 */
#define mod_mem_type_is_init(type)	\
	((type) == MOD_INIT_TEXT ||	\
	 (type) == MOD_INIT_DATA ||	\
	 (type) == MOD_INIT_RODATA)

/**
 * @def mod_mem_type_is_core(type)
 * @brief Checks if a module memory type is part of the core (non-init) sections.
 * @param type The `mod_mem_type` to check.
 */
#define mod_mem_type_is_core(type) (!mod_mem_type_is_init(type))

/**
 * @def mod_mem_type_is_text(type)
 * @brief Checks if a module memory type is a text (code) section.
 * @param type The `mod_mem_type` to check.
 */
#define mod_mem_type_is_text(type)	\
	 ((type) == MOD_TEXT ||		\
	  (type) == MOD_INIT_TEXT)

/**
 * @def mod_mem_type_is_data(type)
 * @brief Checks if a module memory type is a data section.
 * @param type The `mod_mem_type` to check.
 */
#define mod_mem_type_is_data(type) (!mod_mem_type_is_text(type))

/**
 * @def mod_mem_type_is_core_data(type)
 * @brief Checks if a module memory type is a core data section.
 * @param type The `mod_mem_type` to check.
 */
#define mod_mem_type_is_core_data(type)	\
	(mod_mem_type_is_core(type) &&	\
	 mod_mem_type_is_data(type))

/**
 * @def for_each_mod_mem_type(type)
 * @brief Macro to iterate through all module memory types.
 * @param type The variable name to use for the current `mod_mem_type`.
 */
#define for_each_mod_mem_type(type)			\
	for (enum mod_mem_type (type) = 0;		\
	     (type) < MOD_MEM_NUM_TYPES; (type)++)

/**
 * @def for_class_mod_mem_type(type, class)
 * @brief Macro to iterate through module memory types of a specific class (e.g., core, init).
 * @param type The variable name for the current `mod_mem_type`.
 * @param class The class of memory types to iterate (e.g., `core`, `init`).
 */
#define for_class_mod_mem_type(type, class)		\
	for_each_mod_mem_type(type)			\
		if (mod_mem_type_is_##class(type))

/**
 * @struct module_memory
 * @brief Describes a memory region within a module.
 *
 * This structure holds information about the base address, size, and
 * read-only/executable status of a module's memory section.
 */
struct module_memory {
	void *base;			/**< @brief Base address of the memory region. */
	bool is_rox;			/**< @brief True if the region is Read-Only eXecutable. */
	unsigned int size;		/**< @brief Size of the memory region in bytes. */

#ifdef CONFIG_MODULES_TREE_LOOKUP
	struct mod_tree_node mtn;	/**< @brief Module tree node for address lookup. */
#endif
};

#ifdef CONFIG_MODULES_TREE_LOOKUP
/* Only touch one cacheline for common rbtree-for-core-layout case. */
/**
 * @def __module_memory_align
 * @brief Cacheline alignment for module memory structures.
 */
#define __module_memory_align ____cacheline_aligned
#else
#define __module_memory_align
#endif

/**
 * @struct mod_kallsyms
 * @brief Stores information for kallsyms (kernel symbol lookup) for a module.
 *
 * This structure contains the symbol table, number of symbols, string table,
 * and type table used by kallsyms to resolve symbol names to addresses.
 */
struct mod_kallsyms {
	Elf_Sym *symtab;		/**< @brief Pointer to the ELF symbol table. */
	unsigned int num_symtab;	/**< @brief Number of entries in the symbol table. */
	char *strtab;			/**< @brief Pointer to the string table for symbol names. */
	char *typetab;			/**< @brief Pointer to the type table. */
};

#ifdef CONFIG_LIVEPATCH
/**
 * @struct klp_modinfo - ELF information preserved from the livepatch module
 *
 * @param hdr: ELF header
 * @param sechdrs: Section header table
 * @param secstrings: String table for the section headers
 * @param symndx: The symbol table section index
 */
struct klp_modinfo {
	Elf_Ehdr hdr;			/**< @brief ELF header. */
	Elf_Shdr *sechdrs;		/**< @brief Section header table. */
	char *secstrings;		/**< @brief String table for the section headers. */
	unsigned int symndx;		/**< @brief The symbol table section index. */
};
#endif

/**
 * @struct module
 * @brief Main structure representing a loaded kernel module.
 *
 * This comprehensive structure contains all information about a kernel module,
 * including its state, name, kobject, exported symbols, parameters, memory
 * layout, and various debugging and tracing information.
 */
struct module {
	enum module_state state;	/**< @brief Current state of the module. */

	/* Member of list of modules */
	struct list_head list;		/**< @brief List head for the global module list. */

	/* Unique handle for this module */
	char name[MODULE_NAME_LEN];	/**< @brief Unique name of the module. */

#ifdef CONFIG_STACKTRACE_BUILD_ID
	/* Module build ID */
	unsigned char build_id[BUILD_ID_SIZE_MAX]; /**< @brief Build ID for stack traces. */
#endif

	/* Sysfs stuff. */
	struct module_kobject mkobj;	/**< @brief Kobject for sysfs representation. */
	struct module_attribute *modinfo_attrs; /**< @brief Module info attributes. */
	const char *version;		/**< @brief Module version string. */
	const char *srcversion;		/**< @brief Source version string. */
	struct kobject *holders_dir;	/**< @brief Directory for module holders. */

	/* Exported symbols */
	const struct kernel_symbol *syms;	/**< @brief Array of exported symbols. */
	const u32 *crcs;			/**< @brief Array of CRC checksums for exported symbols. */
	unsigned int num_syms;			/**< @brief Number of exported symbols. */

#ifdef CONFIG_ARCH_USES_CFI_TRAPS
	s32 *kcfi_traps;		/**< @brief Kernel Control-Flow Integrity traps start. */
	s32 *kcfi_traps_end;		/**< @brief Kernel Control-Flow Integrity traps end. */
#endif

	/* Kernel parameters. */
#ifdef CONFIG_SYSFS
	struct mutex param_lock;	/**< @brief Mutex for protecting module parameters. */
#endif
	struct kernel_param *kp;	/**< @brief Array of module parameters. */
	unsigned int num_kp;		/**< @brief Number of module parameters. */

	/* GPL-only exported symbols. */
	unsigned int num_gpl_syms;	/**< @brief Number of GPL-only exported symbols. */
	const struct kernel_symbol *gpl_syms;	/**< @brief Array of GPL-only exported symbols. */
	const u32 *gpl_crcs;		/**< @brief Array of CRC checksums for GPL-only symbols. */
	bool using_gplonly_symbols;	/**< @brief True if this module uses GPL-only symbols. */

#ifdef CONFIG_MODULE_SIG
	/* Signature was verified. */
	bool sig_ok;			/**< @brief True if module signature was verified. */
#endif

	bool async_probe_requested;	/**< @brief True if asynchronous probing is requested for this module. */

	/* Exception table */
	unsigned int num_exentries;	/**< @brief Number of entries in the exception table. */
	struct exception_table_entry *extable; /**< @brief Pointer to the exception table. */

	/* Startup function. */
	int (*init)(void);		/**< @brief Pointer to the module initialization function. */

	struct module_memory mem[MOD_MEM_NUM_TYPES] __module_memory_align; /**< @brief Array describing module memory regions. */

	/* Arch-specific module values */
	struct mod_arch_specific arch;	/**< @brief Architecture-specific module data. */

	unsigned long taints;		/**< @brief Module taint flags (same bits as kernel:taint_flags). */

#ifdef CONFIG_GENERIC_BUG
	/* Support for BUG */
	unsigned num_bugs;		/**< @brief Number of BUG entries. */
	struct list_head bug_list;	/**< @brief List of BUG entries. */
	struct bug_entry *bug_table;	/**< @brief Pointer to the BUG table. */
#endif

#ifdef CONFIG_KALLSYMS
	/* Protected by RCU and/or module_mutex: use rcu_dereference() */
	struct mod_kallsyms __rcu *kallsyms;	/**< @brief Kernel symbol lookup information. */
	struct mod_kallsyms core_kallsyms;	/**< @brief Core kernel symbol lookup information. */

	/* Section attributes */
	struct module_sect_attrs *sect_attrs;	/**< @brief Section attributes. */

	/* Notes attributes */
	struct module_notes_attrs *notes_attrs;	/**< @brief Notes attributes. */
#endif

	/* The command line arguments (may be mangled).  People like
	   keeping pointers to this stuff */
	char *args;			/**< @brief Command line arguments passed to the module. */

#ifdef CONFIG_SMP
	/* Per-cpu data. */
	void __percpu *percpu;		/**< @brief Per-CPU data allocated by the module. */
	unsigned int percpu_size;	/**< @brief Size of per-CPU data. */
#endif
	void *noinstr_text_start;	/**< @brief Start address of non-instrumented text. */
	unsigned int noinstr_text_size;	/**< @brief Size of non-instrumented text. */

#ifdef CONFIG_TRACEPOINTS
	unsigned int num_tracepoints;	/**< @brief Number of tracepoints in the module. */
	tracepoint_ptr_t *tracepoints_ptrs; /**< @brief Pointers to tracepoints. */
#endif
#ifdef CONFIG_TREE_SRCU
	unsigned int num_srcu_structs;	/**< @brief Number of SRCU structures in the module. */
	struct srcu_struct **srcu_struct_ptrs; /**< @brief Pointers to SRCU structures. */
#endif
#ifdef CONFIG_BPF_EVENTS
	unsigned int num_bpf_raw_events;/**< @brief Number of BPF raw events. */
	struct bpf_raw_event_map *bpf_raw_events; /**< @brief BPF raw event map. */
#endif
#ifdef CONFIG_DEBUG_INFO_BTF_MODULES
	unsigned int btf_data_size;	/**< @brief Size of BTF (BPF Type Format) data. */
	unsigned int btf_base_data_size;/**< @brief Size of base BTF data. */
	void *btf_data;			/**< @brief Pointer to BTF data. */
	void *btf_base_data;		/**< @brief Pointer to base BTF data. */
#endif
#ifdef CONFIG_JUMP_LABEL
	struct jump_entry *jump_entries;/**< @brief Jump label entries. */
	unsigned int num_jump_entries;	/**< @brief Number of jump label entries. */
#endif
#ifdef CONFIG_TRACING
	unsigned int num_trace_bprintk_fmt; /**< @brief Number of trace_bprintk format strings. */
	const char **trace_bprintk_fmt_start; /**< @brief Start of trace_bprintk format strings. */
#endif
#ifdef CONFIG_EVENT_TRACING
	struct trace_event_call **trace_events; /**< @brief Trace event calls. */
	unsigned int num_trace_events;	/**< @brief Number of trace events. */
	struct trace_eval_map **trace_evals;	/**< @brief Trace evaluation maps. */
	unsigned int num_trace_evals;	/**< @brief Number of trace evaluation maps. */
#endif
#ifdef CONFIG_FTRACE_MCOUNT_RECORD
	unsigned int num_ftrace_callsites; /**< @brief Number of ftrace callsites. */
	unsigned long *ftrace_callsites;/**< @brief Ftrace callsites. */
#endif
#ifdef CONFIG_KPROBES
	void *kprobes_text_start;	/**< @brief Start address of kprobes text. */
	unsigned int kprobes_text_size;	/**< @brief Size of kprobes text. */
	unsigned long *kprobe_blacklist;/**< @brief Kprobe blacklist. */
	unsigned int num_kprobe_blacklist; /**< @brief Number of kprobe blacklist entries. */
#endif
#ifdef CONFIG_HAVE_STATIC_CALL_INLINE
	int num_static_call_sites;	/**< @brief Number of static call sites. */
	struct static_call_site *static_call_sites; /**< @brief Static call sites. */
#endif
#if IS_ENABLED(CONFIG_KUNIT)
	int num_kunit_init_suites;	/**< @brief Number of KUnit init suites. */
	struct kunit_suite **kunit_init_suites;	/**< @brief KUnit init suites. */
	int num_kunit_suites;		/**< @brief Number of KUnit suites. */
	struct kunit_suite **kunit_suites;	/**< @brief KUnit suites. */
#endif


#ifdef CONFIG_LIVEPATCH
	bool klp; /* Is this a livepatch module? */	/**< @brief True if this is a livepatch module. */
	bool klp_alive;					/**< @brief True if the livepatch is alive. */

	/* ELF information */
	struct klp_modinfo *klp_info;		/**< @brief Livepatch module ELF information. */
#endif

#ifdef CONFIG_PRINTK_INDEX
	unsigned int printk_index_size;		/**< @brief Size of printk index. */
	struct pi_entry **printk_index_start;	/**< @brief Start of printk index. */
#endif

#ifdef CONFIG_MODULE_UNLOAD
	/* What modules depend on me? */
	struct list_head source_list;		/**< @brief List of modules that depend on this one. */
	/* What modules do I depend on? */
	struct list_head target_list;		/**< @brief List of modules this one depends on. */

	/* Destruction function. */
	void (*exit)(void);			/**< @brief Pointer to the module exit function. */

	atomic_t refcnt;			/**< @brief Atomic reference count for the module. */
#endif

#ifdef CONFIG_CONSTRUCTORS
	/* Constructor functions. */
	ctor_fn_t *ctors;			/**< @brief Constructor functions. */
	unsigned int num_ctors;			/**< @brief Number of constructor functions. */
#endif

#ifdef CONFIG_FUNCTION_ERROR_INJECTION
	struct error_injection_entry *ei_funcs;	/**< @brief Error injection functions. */
	unsigned int num_ei_funcs;		/**< @brief Number of error injection functions. */
#endif
#ifdef CONFIG_DYNAMIC_DEBUG_CORE
	struct _ddebug_info dyndbg_info;	/**< @brief Dynamic debug information. */
#endif
} ____cacheline_aligned __randomize_layout;
#ifndef MODULE_ARCH_INIT
/**
 * @def MODULE_ARCH_INIT
 * @brief Macro for architecture-specific module initialization.
 */
#define MODULE_ARCH_INIT {}
#endif

#ifndef HAVE_ARCH_KALLSYMS_SYMBOL_VALUE
/**
 * @brief Retrieves the value (address) of an ELF symbol.
 * @param sym Pointer to the ELF symbol.
 * @return The value of the symbol.
 */
static inline unsigned long kallsyms_symbol_value(const Elf_Sym *sym)
{
	return sym->st_value;
}
#endif

/* FIXME: It'd be nice to isolate modules during init, too, so they
   aren't used before they (may) fail.  But presently too much code
   (IDE & SCSI) require entry into the module during init.*/
/**
 * @brief Checks if a module is currently in a "live" state.
 * @param mod Pointer to the `module` structure.
 * @return True if the module is live, false otherwise.
 */
static inline bool module_is_live(struct module *mod)
{
	return mod->state != MODULE_STATE_GOING;
}

/**
 * @brief Checks if a module is currently in a "coming" state.
 * @param mod Pointer to the `module` structure.
 * @return True if the module is coming, false otherwise.
 */
static inline bool module_is_coming(struct module *mod)
{
        return mod->state == MODULE_STATE_COMING;
}

/**
 * @brief Finds the module that owns a given text address.
 * @param addr The address to look up.
 * @return Pointer to the `module` structure, or NULL if not found.
 */
struct module *__module_text_address(unsigned long addr);
/**
 * @brief Finds the module that owns a given address (text or data).
 * @param addr The address to look up.
 * @return Pointer to the `module` structure, or NULL if not found.
 */
struct module *__module_address(unsigned long addr);
/**
 * @brief Checks if an address belongs to any loaded module.
 * @param addr The address to check.
 * @return True if the address is within a module, false otherwise.
 */
bool is_module_address(unsigned long addr);
/**
 * @brief Checks if an address is within a module's per-CPU data section.
 * @param addr The address to check.
 * @param can_addr Output parameter for canonical address.
 * @return True if the address is within per-CPU data, false otherwise.
 */
bool __is_module_percpu_address(unsigned long addr, unsigned long *can_addr);
/**
 * @brief Checks if an address is within a module's per-CPU data section.
 * @param addr The address to check.
 * @return True if the address is within per-CPU data, false otherwise.
 */
bool is_module_percpu_address(unsigned long addr);
/**
 * @brief Checks if an address is within a module's text (code) section.
 * @param addr The address to check.
 * @return True if the address is within text section, false otherwise.
 */
bool is_module_text_address(unsigned long addr);

/**
 * @brief Checks if an address is within a specific memory type of a module.
 * @param addr The address to check.
 * @param mod Pointer to the `module` structure.
 * @param type The `mod_mem_type` to check against.
 * @return True if the address is within the specified memory region, false otherwise.
 */
static inline bool within_module_mem_type(unsigned long addr,
					  const struct module *mod,
					  enum mod_mem_type type)
{
	unsigned long base, size;

	base = (unsigned long)mod->mem[type].base;
	size = mod->mem[type].size;
	return addr - base < size;
}

/**
 * @brief Checks if an address is within the core (non-init) sections of a module.
 * @param addr The address to check.
 * @param mod Pointer to the `module` structure.
 * @return True if the address is within a core section, false otherwise.
 */
static inline bool within_module_core(unsigned long addr,
				      const struct module *mod)
{
	for_class_mod_mem_type(type, core) {
		if (within_module_mem_type(addr, mod, type))
			return true;
	}
	return false;
}

/**
 * @brief Checks if an address is within the initialization sections of a module.
 * @param addr The address to check.
 * @param mod Pointer to the `module` structure.
 * @return True if the address is within an init section, false otherwise.
 */
static inline bool within_module_init(unsigned long addr,
				      const struct module *mod)
{
	for_class_mod_mem_type(type, init) {
		if (within_module_mem_type(addr, mod, type))
			return true;
	}
	return false;
}

/**
 * @brief Checks if an address is within any memory section of a module.
 * @param addr The address to check.
 * @param mod Pointer to the `module` structure.
 * @return True if the address is within the module, false otherwise.
 */
static inline bool within_module(unsigned long addr, const struct module *mod)
{
	return within_module_init(addr, mod) || within_module_core(addr, mod);
}

/**
 * @brief Searches for a loaded module by its name.
 * @param name The name of the module to find.
 * @return Pointer to the `module` structure, or NULL if not found.
 *
 * This function must be called within an RCU (Read-Copy Update) critical section.
 */
struct module *find_module(const char *name);

/**
 * @brief Puts a module reference and exits the kthread.
 * @param mod Pointer to the module to put.
 * @param code The exit code for the kthread.
 *
 * This function handles the final `module_put` and `kthread_exit` sequence.
 */
extern void __noreturn __module_put_and_kthread_exit(struct module *mod,
			long code);
/**
 * @def module_put_and_kthread_exit(code)
 * @brief Macro to put a module reference and exit the current kthread.
 * @param code The exit code for the kthread.
 *
 * This macro is used by kernel threads that hold a reference to `THIS_MODULE`.
 */
#define module_put_and_kthread_exit(code) __module_put_and_kthread_exit(THIS_MODULE, code)

#ifdef CONFIG_MODULE_UNLOAD
/**
 * @brief Returns the current reference count of a module.
 * @param mod Pointer to the `module` structure.
 * @return The module's reference count.
 */
int module_refcount(struct module *mod);
/**
 * @brief Decrements the reference count of the module that exports a symbol.
 * @param symbol The name of the symbol.
 *
 * This function is primarily for internal kernel use; `symbol_put()`
 * is the preferred user-facing macro.
 */
void __symbol_put(const char *symbol);
/**
 * @def symbol_put(x)
 * @brief Macro to decrement the reference count of the module exporting symbol `x`.
 * @param x The symbol name.
 *
 * Calls to `symbol_put()` must balance calls to `symbol_get()`.
 */
#define symbol_put(x) __symbol_put(__stringify(x))
/**
 * @brief Decrements the reference count of the module that owns the given address.
 * @param addr The address within the module.
 */
void symbol_put_addr(void *addr);

/* Sometimes we know we already have a refcount, and it's easier not
   to handle the error case (which only happens with rmmod --wait). */
/**
 * @brief Increments a module's reference count without checking its state.
 * @param module Pointer to the `module` structure.
 *
 * This function is used when it's guaranteed that the module is not being
 * removed.
 */
extern void __module_get(struct module *module);

/**
 * @brief Tries to increment a module's reference count.
 * @param module: the module we should check for
 *
 * Only try to get a module reference count if the module is not being removed.
 * This call will fail if the module is in the process of being removed.
 *
 * Care must also be taken to ensure the module exists and is alive prior to
 * usage of this call. This can be gauranteed through two means:
 *
 * 1) Direct protection: you know an earlier caller must have increased the
 *    module reference through __module_get(). This can typically be achieved
 *    by having another entity other than the module itself increment the
 *    module reference count.
 *
 * 2) Implied protection: there is an implied protection against module
 *    removal. An example of this is the implied protection used by kernfs /
 *    sysfs. The sysfs store / read file operations are guaranteed to exist
 *    through the use of kernfs's active reference (see kernfs_active()) and a
 *    sysfs / kernfs file removal cannot happen unless the same file is not
 *    active. Therefore, if a sysfs file is being read or written to the module
 *    which created it must still exist. It is therefore safe to use
 *    try_module_get() on module sysfs store / read ops.
 *
 * One of the real values to try_module_get() is the module_is_live() check
 * which ensures that the caller of try_module_get() can yield to userspace
 * module removal requests and gracefully fail if the module is on its way out.
 *
 * Returns true if the reference count was successfully incremented.
 */
extern bool try_module_get(struct module *module);

/**
 * @brief Decrements a module's reference count.
 * @param module: the module we should release a reference count for
 *
 * If you successfully bump a reference count to a module with try_module_get(),
 * when you are finished you must call module_put() to release that reference
 * count.
 */
extern void module_put(struct module *module);

#else /*!CONFIG_MODULE_UNLOAD*/
// Block Logic: If module unloading is not configured, these are stub implementations.
static inline bool try_module_get(struct module *module)
{
	return !module || module_is_live(module);
}
static inline void module_put(struct module *module)
{
}
static inline void __module_get(struct module *module)
{
}
#define symbol_put(x) do { } while (0)
#define symbol_put_addr(p) do { } while (0)

#endif /* CONFIG_MODULE_UNLOAD */

/* This is a #define so the string doesn't get put in every .o file */
/**
 * @def module_name(mod)
 * @brief Macro to get the name of a module.
 * @param mod Pointer to the `module` structure.
 * @return The module's name string, or "kernel" if `mod` is NULL.
 */
#define module_name(mod)			\
({						\
	struct module *__mod = (mod);		\
	__mod ? __mod->name : "kernel";		\
})

/**
 * @brief Dereferences a module function descriptor.
 * @param mod Pointer to the `module` structure.
 * @param ptr The function descriptor pointer.
 * @return The actual function pointer.
 */
void *dereference_module_function_descriptor(struct module *mod, void *ptr);

/**
 * @brief Registers a module notifier.
 * @param nb Pointer to the `notifier_block`.
 * @return 0 on success, or a negative errno on failure.
 *
 * Module notifiers are used to be informed about module load/unload events.
 */
int register_module_notifier(struct notifier_block *nb);
/**
 * @brief Unregisters a module notifier.
 * @param nb Pointer to the `notifier_block`.
 * @return 0 on success, or a negative errno on failure.
 */
int unregister_module_notifier(struct notifier_block *nb);

/**
 * @brief Prints information about all loaded modules to the kernel log.
 */
extern void print_modules(void);

/**
 * @brief Checks if a module has requested asynchronous probing.
 * @param module Pointer to the `module` structure.
 * @return True if async probing is requested, false otherwise.
 */
static inline bool module_requested_async_probing(struct module *module)
{
	return module && module->async_probe_requested;
}

/**
 * @brief Checks if a module is a livepatch module.
 * @param mod Pointer to the `module` structure.
 * @return True if it's a livepatch module, false otherwise.
 */
static inline bool is_livepatch_module(struct module *mod)
{
#ifdef CONFIG_LIVEPATCH
	return mod->klp;
#else
	return false;
#endif
}

/**
 * @brief Sets the module signature enforcement policy.
 *
 * This function is used to enable or disable the enforcement of module
 * signatures.
 */
void set_module_sig_enforced(void);

/**
 * @brief Iterates over all loaded modules and calls a function for each.
 * @param func The callback function to execute for each module.
 * @param data User data to pass to the callback function.
 */
void module_for_each_mod(int(*func)(struct module *mod, void *data), void *data);

#else /* !CONFIG_MODULES... */

// Block Logic: If modules are not configured, these functions are stubbed out.

static inline struct module *__module_address(unsigned long addr)
{
	return NULL;
}

static inline struct module *__module_text_address(unsigned long addr)
{
	return NULL;
}

static inline bool is_module_address(unsigned long addr)
{
	return false;
}

static inline bool is_module_percpu_address(unsigned long addr)
{
	return false;
}

static inline bool __is_module_percpu_address(unsigned long addr, unsigned long *can_addr)
{
	return false;
}

static inline bool is_module_text_address(unsigned long addr)
{
	return false;
}

static inline bool within_module_core(unsigned long addr,
				      const struct module *mod)
{
	return false;
}

static inline bool within_module_init(unsigned long addr,
				      const struct module *mod)
{
	return false;
}

static inline bool within_module(unsigned long addr, const struct module *mod)
{
	return false;
}

/* Get/put a kernel symbol (calls should be symmetric) */
/**
 * @def symbol_get(x)
 * @brief Stub macro for `symbol_get` when modules are disabled.
 */
#define symbol_get(x) ({ extern typeof(x) x __attribute__((weak,visibility("hidden"))); &(x); })
/**
 * @def symbol_put(x)
 * @brief Stub macro for `symbol_put` when modules are disabled.
 */
#define symbol_put(x) do { } while (0)
/**
 * @def symbol_put_addr(p)
 * @brief Stub macro for `symbol_put_addr` when modules are disabled.
 */
#define symbol_put_addr(p) do { } while (0)

static inline void __module_get(struct module *module)
{
}

static inline bool try_module_get(struct module *module)
{
	return true;
}

static inline void module_put(struct module *module)
{
}

/**
 * @def module_name(mod)
 * @brief Stub macro for `module_name` when modules are disabled, returns "kernel".
 */
#define module_name(mod) "kernel"

static inline int register_module_notifier(struct notifier_block *nb)
{
	/* no events will happen anyway, so this can always succeed */
	return 0;
}

static inline int unregister_module_notifier(struct notifier_block *nb)
{
	return 0;
}

/**
 * @def module_put_and_kthread_exit(code)
 * @brief Stub macro for `module_put_and_kthread_exit` when modules are disabled.
 */
#define module_put_and_kthread_exit(code) kthread_exit(code)

static inline void print_modules(void)
{
}

static inline bool module_requested_async_probing(struct module *module)
{
	return false;
}


static inline void set_module_sig_enforced(void)
{
}

/* Dereference module function descriptor */
static inline
void *dereference_module_function_descriptor(struct module *mod, void *ptr)
{
	return ptr;
}

static inline bool module_is_coming(struct module *mod)
{
	return false;
}

static inline void module_for_each_mod(int(*func)(struct module *mod, void *data), void *data)
{
}
#endif /* CONFIG_MODULES */

#ifdef CONFIG_SYSFS
extern struct kset *module_kset;	/**< @brief Kset for all modules. */
extern const struct kobj_type module_ktype; /**< @brief Kobject type for modules. */
#endif /* CONFIG_SYSFS */

/**
 * @def symbol_request(x)
 * @brief Macro to request a kernel symbol, potentially loading the module if needed.
 * @param x The symbol name.
 */
#define symbol_request(x) try_then_request_module(symbol_get(x), "symbol:" #x)

/* BELOW HERE ALL THESE ARE OBSOLETE AND WILL VANISH */

/**
 * @def __MODULE_STRING(x)
 * @brief Macro to stringify a value.
 */
#define __MODULE_STRING(x) __stringify(x)

#ifdef CONFIG_GENERIC_BUG
/**
 * @brief Finalizes bug table information for a module.
 * @param hdr Pointer to the ELF header.
 * @param sechdrs Pointer to the section headers.
 * @param mod Pointer to the module structure.
 */
void module_bug_finalize(const Elf_Ehdr *, const Elf_Shdr *,
			 struct module *);
/**
 * @brief Cleans up bug table information for a module.
 * @param mod Pointer to the module structure.
 */
void module_bug_cleanup(struct module *);

#else	/* !CONFIG_GENERIC_BUG */

static inline void module_bug_finalize(const Elf_Ehdr *hdr,
					const Elf_Shdr *sechdrs,
					struct module *mod)
{
}
static inline void module_bug_cleanup(struct module *mod) {}
#endif	/* CONFIG_GENERIC_BUG */

#ifdef CONFIG_MITIGATION_RETPOLINE
/**
 * @brief Checks if a module is compatible with Retpoline mitigation.
 * @param has_retpoline True if the module was compiled with Retpoline.
 * @return True if the module is okay, false otherwise.
 */
extern bool retpoline_module_ok(bool has_retpoline);
#else
static inline bool retpoline_module_ok(bool has_retpoline)
{
	return true;
}
#endif

#ifdef CONFIG_MODULE_SIG
/**
 * @brief Checks if module signature enforcement is enabled.
 * @return True if signature enforcement is on, false otherwise.
 */
bool is_module_sig_enforced(void);

/**
 * @brief Checks if a module's signature has been verified as OK.
 * @param module Pointer to the `module` structure.
 * @return True if the signature is OK, false otherwise.
 */
static inline bool module_sig_ok(struct module *module)
{
	return module->sig_ok;
}
#else	/* !CONFIG_MODULE_SIG */
static inline bool is_module_sig_enforced(void)
{
	return false;
}

static inline bool module_sig_ok(struct module *module)
{
	return true;
}
#endif	/* CONFIG_MODULE_SIG */

#if defined(CONFIG_MODULES) && defined(CONFIG_KALLSYMS)
/**
 * @brief Executes a callback function for each symbol in a specified module's kallsyms.
 * @param modname The name of the module.
 * @param fn The callback function.
 * @param data User data to pass to the callback.
 * @return 0 on success, or a negative errno on failure.
 */
int module_kallsyms_on_each_symbol(const char *modname,
				   int (*fn)(void *, const char *, unsigned long),
				   void *data);

/* For kallsyms to ask for address resolution.  namebuf should be at
 * least KSYM_NAME_LEN long: a pointer to namebuf is returned if
 * found, otherwise NULL.
 */
/**
 * @brief Looks up module address information for a given address.
 * @param addr The address to look up.
 * @param symbolsize Output parameter for symbol size.
 * @param offset Output parameter for offset within symbol.
 * @param modname Output parameter for module name.
 * @param modbuildid Output parameter for module build ID.
 * @param namebuf Buffer for symbol name.
 * @return 0 on success, or a negative errno on failure.
 */
int module_address_lookup(unsigned long addr,
			  unsigned long *symbolsize,
			  unsigned long *offset,
			  char **modname, const unsigned char **modbuildid,
			  char *namebuf);
/**
 * @brief Looks up a module symbol name for a given address.
 * @param addr The address to look up.
 * @param symname Buffer for symbol name.
 * @return 0 on success, or a negative errno on failure.
 */
int lookup_module_symbol_name(unsigned long addr, char *symname);
/**
 * @brief Looks up module symbol attributes for a given address.
 * @param addr The address to look up.
 * @param size Output parameter for symbol size.
 * @param offset Output parameter for offset within symbol.
 * @param modname Buffer for module name.
 * @param name Buffer for symbol name.
 * @return 0 on success, or a negative errno on failure.
 */
int lookup_module_symbol_attrs(unsigned long addr,
			       unsigned long *size,
			       unsigned long *offset,
			       char *modname,
			       char *name);

/* Returns 0 and fills in value, defined and namebuf, or -ERANGE if
 * symnum out of range.
 */
/**
 * @brief Retrieves information about a kallsyms symbol from a module.
 * @param symnum The index of the symbol.
 * @param value Output parameter for symbol value (address).
 * @param type Output parameter for symbol type.
 * @param name Output parameter for symbol name.
 * @param module_name Output parameter for module name.
 * @param exported Output parameter for export status.
 * @return 0 on success, or -ERANGE if `symnum` is out of range.
 */
int module_get_kallsym(unsigned int symnum, unsigned long *value, char *type,
		       char *name, char *module_name, int *exported);

/* Look for this name: can be of form module:name. */
/**
 * @brief Looks up the value (address) of a kallsyms symbol by name.
 * @param name The symbol name (can be "module:symbol_name").
 * @return The address of the symbol, or 0 if not found.
 */
unsigned long module_kallsyms_lookup_name(const char *name);

/**
 * @brief Finds the value (address) of a kallsyms symbol within a specific module.
 * @param mod Pointer to the `module` structure.
 * @param name The symbol name.
 * @return The address of the symbol, or 0 if not found.
 */
unsigned long find_kallsyms_symbol_value(struct module *mod, const char *name);

#else	/* CONFIG_MODULES && CONFIG_KALLSYMS */

// Block Logic: If modules and kallsyms are not configured, these functions are stubbed out.

static inline int module_kallsyms_on_each_symbol(const char *modname,
						 int (*fn)(void *, const char *, unsigned long),
						 void *data)
{
	return -EOPNOTSUPP;
}

/* For kallsyms to ask for address resolution.  NULL means not found. */
static inline int module_address_lookup(unsigned long addr,
						unsigned long *symbolsize,
						unsigned long *offset,
						char **modname,
						const unsigned char **modbuildid,
						char *namebuf)
{
	return 0;
}

static inline int lookup_module_symbol_name(unsigned long addr, char *symname)
{
	return -ERANGE;
}

static inline int module_get_kallsym(unsigned int symnum, unsigned long *value,
				     char *type, char *name,
				     char *module_name, int *exported)
{
	return -ERANGE;
}

static inline unsigned long module_kallsyms_lookup_name(const char *name)
{
	return 0;
}

static inline unsigned long find_kallsyms_symbol_value(struct module *mod,
						       const char *name)
{
	return 0;
}

#endif  /* CONFIG_MODULES && CONFIG_KALLSYMS */

#endif /* _LINUX_MODULE_H */
