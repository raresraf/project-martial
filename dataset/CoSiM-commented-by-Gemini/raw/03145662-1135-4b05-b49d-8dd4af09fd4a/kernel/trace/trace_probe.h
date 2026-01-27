/**
 * @file trace_probe.h
 * @brief Common header file for probe-based Dynamic events in the Linux kernel.
 *
 * This header provides shared definitions for various dynamic tracing
 * mechanisms within the Linux kernel, including kprobes, uprobes, eprobes,
 * and fprobes. It defines common data structures, macros, and API prototypes
 * necessary for parsing probe arguments, generating fetch instructions to
 * extract data from different kernel contexts (registers, stack, memory),
 * formatting output, and managing the lifecycle of these dynamic events.
 *
 * Functional Utility:
 * - Defines generic fetch operations and instruction formats for data extraction.
 * - Provides structures for representing probe arguments and their associated types.
 * - Supports dynamic event creation, registration, and cleanup.
 * - Facilitates argument parsing with support for symbol resolution, offsets,
 *   and advanced dereferencing.
 * - Integrates with BTF (BPF Type Format) for type-aware argument handling.
 * - Centralizes error logging for probe parsing.
 *
 * Architectural Intent:
 * - To unify the infrastructure for different types of dynamic probes, reducing
 *   code duplication and promoting consistency.
 * - To enable a powerful and flexible user-space interface for creating and
 *   managing custom trace events.
 *
 * This code was copied from kernel/trace/trace_kprobe.h written by
 * Masami Hiramatsu <masami.hiramatsu.pt@hitachi.com>
 *
 * Updates to make this generic:
 * Copyright (C) IBM Corporation, 2010-2011
 * Author:     Srikar Dronamraju
 */
// SPDX-License-Identifier: GPL-2.0
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/smp.h>
#include <linux/tracefs.h>
#include <linux/types.h>
#include <linux/string.h>
#include <linux/ptrace.h>
#include <linux/perf_event.h>
#include <linux/kprobes.h>
#include <linux/stringify.h>
#include <linux/limits.h>
#include <linux/uaccess.h>
#include <linux/bitops.h>
#include <linux/btf.h>
#include <asm/bitsperlong.h>

#include "trace.h"
#include "trace_output.h"

/**
 * @def MAX_TRACE_ARGS
 * @brief Maximum number of arguments supported for a single trace probe.
 */
#define MAX_TRACE_ARGS		128
/**
 * @def MAX_ARGSTR_LEN
 * @brief Maximum length of an argument string.
 */
#define MAX_ARGSTR_LEN		63
/**
 * @def MAX_ARRAY_LEN
 * @brief Maximum length of an array that can be specified in probe arguments.
 */
#define MAX_ARRAY_LEN		64
/**
 * @def MAX_ARG_NAME_LEN
 * @brief Maximum length of a probe argument's name.
 */
#define MAX_ARG_NAME_LEN	32
/**
 * @def MAX_BTF_ARGS_LEN
 * @brief Maximum length for BTF-expanded arguments.
 */
#define MAX_BTF_ARGS_LEN	128
/**
 * @def MAX_DENTRY_ARGS_LEN
 * @brief Maximum length for dentry-expanded arguments.
 */
#define MAX_DENTRY_ARGS_LEN	256
/**
 * @def MAX_STRING_SIZE
 * @brief Maximum size of a string that can be captured, typically `PATH_MAX`.
 */
#define MAX_STRING_SIZE		PATH_MAX

/* Reserved field names */
/**
 * @def FIELD_STRING_IP
 * @brief Reserved field name for instruction pointer.
 */
#define FIELD_STRING_IP		"__probe_ip"
/**
 * @def FIELD_STRING_RETIP
 * @brief Reserved field name for return instruction pointer.
 */
#define FIELD_STRING_RETIP	"__probe_ret_ip"
/**
 * @def FIELD_STRING_FUNC
 * @brief Reserved field name for function entry instruction pointer.
 */
#define FIELD_STRING_FUNC	"__probe_func"

#undef DEFINE_FIELD
/**
 * @def DEFINE_FIELD(type, item, name, is_signed)
 * @brief Macro to define a field within a trace event structure.
 * @param type The C type of the field (e.g., `unsigned long`).
 * @param item The name of the member in the trace entry head structure.
 * @param name The name of the field as exposed in tracefs.
 * @param is_signed Boolean indicating if the field is signed.
 *
 * This macro simplifies the process of defining fields for dynamic trace events,
 * calculating offsets and sizes.
 */
#define DEFINE_FIELD(type, item, name, is_signed)			\
	do {								\
		ret = trace_define_field(event_call, #type, name,	\
					 offsetof(typeof(field), item),	\
					 sizeof(field.item), is_signed, \
					 FILTER_OTHER);			\
		if (ret)						\
			return ret;					\
	} while (0)


/* Flags for trace_probe */
/**
 * @def TP_FLAG_TRACE
 * @brief Flag indicating that the trace probe is enabled for tracing.
 */
#define TP_FLAG_TRACE		1
/**
 * @def TP_FLAG_PROFILE
 * @brief Flag indicating that the trace probe is enabled for profiling (perf events).
 */
#define TP_FLAG_PROFILE		2

/* data_loc: data location, compatible with u32 */
/**
 * @def make_data_loc(len, offs)
 * @brief Encodes length and offset into a `u32` for data location.
 * @param len The length of the data.
 * @param offs The offset of the data.
 *
 * This macro packs data length into the upper 16 bits and offset into the
 * lower 16 bits of a `u32`.
 */
#define make_data_loc(len, offs)	\
	(((u32)(len) << 16) | ((u32)(offs) & 0xffff))
/**
 * @def get_loc_len(dl)
 * @brief Extracts data length from a `u32` data location.
 * @param dl The `u32` data location.
 */
#define get_loc_len(dl)		((u32)(dl) >> 16)
/**
 * @def get_loc_offs(dl)
 * @brief Extracts data offset from a `u32` data location.
 * @param dl The `u32` data location.
 */
#define get_loc_offs(dl)	((u32)(dl) & 0xffff)

/**
 * @brief Retrieves the actual data pointer from a data location and event entry.
 * @param dl Pointer to the `u32` data location.
 * @param ent Pointer to the raw event entry.
 * @return Pointer to the data.
 */
static nokprobe_inline void *get_loc_data(u32 *dl, void *ent)
{
	return (u8 *)ent + get_loc_offs(*dl);
}

/**
 * @brief Updates a `u32` data location after consuming some data.
 * @param loc The `u32` data location.
 * @param consumed The amount of data consumed.
 * @return The updated `u32` data location.
 */
static nokprobe_inline u32 update_data_loc(u32 loc, int consumed)
{
	u32 maxlen = get_loc_len(loc);
	u32 offset = get_loc_offs(loc);

	return make_data_loc(maxlen - consumed, offset + consumed);
}

/* Printing function type */
/**
 * @typedef print_type_func_t
 * @brief Function pointer type for printing various data types.
 */
typedef int (*print_type_func_t)(struct trace_seq *, void *, void *);

/**
 * @enum fetch_op
 * @brief Enumeration of fetch operations for probe arguments.
 *
 * These operations represent the steps involved in extracting data from
 * the kernel's context (e.g., registers, stack, memory) and processing it.
 */
enum fetch_op {
	FETCH_OP_NOP = 0,	/**< @brief No operation. */
	// Stage 1 (load) ops
	FETCH_OP_REG,		/**< @brief Load from a register (`.param = offset`). */
	FETCH_OP_STACK,		/**< @brief Load from stack (`.param = index`). */
	FETCH_OP_STACKP,	/**< @brief Load stack pointer. */
	FETCH_OP_RETVAL,	/**< @brief Load function return value. */
	FETCH_OP_IMM,		/**< @brief Load immediate value (`.immediate`). */
	FETCH_OP_COMM,		/**< @brief Load current process's `comm` string. */
	FETCH_OP_ARG,		/**< @brief Load function argument (`.param`). */
	FETCH_OP_FOFFS,		/**< @brief Load from file offset (`.immediate`). */
	FETCH_OP_DATA,		/**< @brief Load allocated data (`.data`). */
	FETCH_OP_EDATA,		/**< @brief Load from entry data buffer (`.offset`). */
	// Stage 2 (dereference) op
	FETCH_OP_DEREF,		/**< @brief Dereference memory (`.offset`). */
	FETCH_OP_UDEREF,	/**< @brief User-space Dereference: .offset. */
	// Stage 3 (store) ops
	FETCH_OP_ST_RAW,	/**< @brief Store raw data (`.size`). */
	FETCH_OP_ST_MEM,	/**< @brief Store memory (`.offset`, `.size`). */
	FETCH_OP_ST_UMEM,	/**< @brief Store user-space memory (`.offset`, `.size`). */
	FETCH_OP_ST_STRING,	/**< @brief Store string (`.offset`, `.size`). */
	FETCH_OP_ST_USTRING,	/**< @brief Store user-space string (`.offset`, `.size`). */
	FETCH_OP_ST_SYMSTR,	/**< @brief Store kernel symbol string (`.offset`, `.size`). */
	FETCH_OP_ST_EDATA,	/**< @brief Store entry data (`.offset`). */
	// Stage 4 (modify) op
	FETCH_OP_MOD_BF,	/**< @brief Modify bitfield (`.basesize`, `.lshift`, `.rshift`). */
	// Stage 5 (loop) op
	FETCH_OP_LP_ARRAY,	/**< @brief Loop for array processing (`.param = loop count`). */
	FETCH_OP_TP_ARG,	/**< @brief Trace Point argument. */
	FETCH_OP_END,		/**< @brief End of fetch instruction sequence. */
	FETCH_NOP_SYMBOL,	/**< @brief Unresolved symbol placeholder. */
};

/**
 * @struct fetch_insn
 * @brief Represents a single fetch instruction in a probe argument.
 *
 * This structure defines an operation (`op`) and its associated parameters
 * (e.g., `param`, `offset`, `immediate`, `data`) for extracting data.
 */
struct fetch_insn {
	enum fetch_op op;	/**< @brief The fetch operation. */
	union {
		unsigned int param;	/**< @brief Generic parameter (e.g., register offset, stack index). */
		struct {
			unsigned int size;	/**< @brief Size of data to fetch/store. */
			int offset;		/**< @brief Offset for memory access or entry data. */
		};
		struct {
			unsigned char basesize;	/**< @brief Base size for bitfield operations. */
			unsigned char lshift;	/**< @brief Left shift for bitfield operations. */
			unsigned char rshift;	/**< @brief Right shift for bitfield operations. */
		};
		unsigned long immediate;	/**< @brief Immediate value. */
		void *data;			/**< @brief Pointer to allocated data (e.g., string literals). */
	};
};

/* fetch + deref*N + store + mod + end <= 16, this allows N=12, enough */
/**
 * @def FETCH_INSN_MAX
 * @brief Maximum number of fetch instructions in an argument.
 *
 * This limits the complexity of argument parsing to prevent excessive resource
 * consumption.
 */
#define FETCH_INSN_MAX	16
/**
 * @def FETCH_TOKEN_COMM
 * @brief Token for `comm` string, used internally.
 */
#define FETCH_TOKEN_COMM	(-ECOMM)

/* Fetch type information table */
/**
 * @struct fetch_type
 * @brief Describes a data type for fetching and printing.
 *
 * This structure associates a type name with its size, signedness, string
 * property, and corresponding print functions and format strings.
 */
struct fetch_type {
	const char		*name;		/**< @brief Name of type (e.g., "u32", "string"). */
	size_t			size;		/**< @brief Byte size of the type. */
	bool			is_signed;	/**< @brief True if the type is signed. */
	bool			is_string;	/**< @brief True if the type is a string. */
	print_type_func_t	print;		/**< @brief Print function for this type. */
	const char		*fmt;		/**< @brief Format string for `trace_seq_printf`. */
	const char		*fmttype;	/**< @brief Name used in the format file. */
};

/* For defining macros, define string/string_size types */
/**
 * @typedef string
 * @brief Alias for `u32` representing a string data location.
 */
typedef u32 string;
/**
 * @typedef string_size
 * @brief Alias for `u32` representing string size information.
 */
typedef u32 string_size;

/**
 * @def PRINT_TYPE_FUNC_NAME(type)
 * @brief Helper macro to construct print function names.
 */
#define PRINT_TYPE_FUNC_NAME(type)	print_type_##type
/**
 * @def PRINT_TYPE_FMT_NAME(type)
 * @brief Helper macro to construct print format name.
 */
#define PRINT_TYPE_FMT_NAME(type)	print_type_format_##type

/* Printing  in basic type function template */
/**
 * @def DECLARE_BASIC_PRINT_TYPE_FUNC(type)
 * @brief Macro to declare basic print functions for various data types.
 */
#define DECLARE_BASIC_PRINT_TYPE_FUNC(type)				\
int PRINT_TYPE_FUNC_NAME(type)(struct trace_seq *s, void *data, void *ent);\
extern const char PRINT_TYPE_FMT_NAME(type)[]

DECLARE_BASIC_PRINT_TYPE_FUNC(u8);
DECLARE_BASIC_PRINT_TYPE_FUNC(u16);
DECLARE_BASIC_PRINT_TYPE_FUNC(u32);
DECLARE_BASIC_PRINT_TYPE_FUNC(u64);
DECLARE_BASIC_PRINT_TYPE_FUNC(s8);
DECLARE_BASIC_PRINT_TYPE_FUNC(s16);
DECLARE_BASIC_PRINT_TYPE_FUNC(s32);
DECLARE_BASIC_PRINT_TYPE_FUNC(s64);
DECLARE_BASIC_PRINT_TYPE_FUNC(x8);
DECLARE_BASIC_PRINT_TYPE_FUNC(x16);
DECLARE_BASIC_PRINT_TYPE_FUNC(x32);
DECLARE_BASIC_PRINT_TYPE_FUNC(x64);

DECLARE_BASIC_PRINT_TYPE_FUNC(char);
DECLARE_BASIC_PRINT_TYPE_FUNC(string);
DECLARE_BASIC_PRINT_TYPE_FUNC(symbol);

/* Default (unsigned long) fetch type */
#define __DEFAULT_FETCH_TYPE(t) x##t
#define _DEFAULT_FETCH_TYPE(t) __DEFAULT_FETCH_TYPE(t)
/**
 * @def DEFAULT_FETCH_TYPE
 * @brief Default fetch type, typically `x32` or `x64` based on `BITS_PER_LONG`.
 */
#define DEFAULT_FETCH_TYPE _DEFAULT_FETCH_TYPE(BITS_PER_LONG)
/**
 * @def DEFAULT_FETCH_TYPE_STR
 * @brief String representation of the default fetch type.
 */
#define DEFAULT_FETCH_TYPE_STR __stringify(DEFAULT_FETCH_TYPE)

#define __ADDR_FETCH_TYPE(t) u##t
#define _ADDR_FETCH_TYPE(t) __ADDR_FETCH_TYPE(t)
/**
 * @def ADDR_FETCH_TYPE
 * @brief Address fetch type, typically `u32` or `u64` based on `BITS_PER_LONG`.
 */
#define ADDR_FETCH_TYPE _ADDR_FETCH_TYPE(BITS_PER_LONG)

/**
 * @def __ASSIGN_FETCH_TYPE(_name, ptype, ftype, _size, sign, str, _fmttype)
 * @brief Helper macro to assign properties for a new fetch type.
 */
#define __ASSIGN_FETCH_TYPE(_name, ptype, ftype, _size, sign, str, _fmttype)	\
	{.name = _name,					\
	 .size = _size,					\
	 .is_signed = (bool)sign,			\
	 .is_string = (bool)str,			\
	 .print = PRINT_TYPE_FUNC_NAME(ptype),		\
	 .fmt = PRINT_TYPE_FMT_NAME(ptype),		\
	 .fmttype = _fmttype,				\
	}

/* Non string types can use these macros */
/**
 * @def _ASSIGN_FETCH_TYPE(_name, ptype, ftype, _size, sign, _fmttype)
 * @brief Helper macro for assigning non-string fetch types.
 */
#define _ASSIGN_FETCH_TYPE(_name, ptype, ftype, _size, sign, _fmttype)	\
	__ASSIGN_FETCH_TYPE(_name, ptype, ftype, _size, sign, 0, #_fmttype)
/**
 * @def ASSIGN_FETCH_TYPE(ptype, ftype, sign)
 * @brief Macro to assign basic fetch type properties.
 */
#define ASSIGN_FETCH_TYPE(ptype, ftype, sign)			\
	_ASSIGN_FETCH_TYPE(#ptype, ptype, ftype, sizeof(ftype), sign, ptype)

/* If ptype is an alias of atype, use this macro (show atype in format) */
/**
 * @def ASSIGN_FETCH_TYPE_ALIAS(ptype, atype, ftype, sign)
 * @brief Macro to assign alias fetch type properties.
 */
#define ASSIGN_FETCH_TYPE_ALIAS(ptype, atype, ftype, sign)		\
	_ASSIGN_FETCH_TYPE(#ptype, ptype, ftype, sizeof(ftype), sign, atype)

/**
 * @def ASSIGN_FETCH_TYPE_END
 * @brief Marker for the end of the fetch type table.
 */
#define ASSIGN_FETCH_TYPE_END {}
#undef MAX_ARRAY_LEN	// Functional Utility: Undefine to redefine.
/**
 * @def MAX_ARRAY_LEN
 * @brief Maximum length of an array that can be specified in probe arguments.
 */
#define MAX_ARRAY_LEN	64

#ifdef CONFIG_KPROBE_EVENTS
/**
 * @brief Checks if a kprobe is at a function entry point.
 * @param call Pointer to `trace_event_call`.
 * @return True if at function entry, false otherwise.
 */
bool trace_kprobe_on_func_entry(struct trace_event_call *call);
/**
 * @brief Checks if a kprobe is error-injectable.
 * @param call Pointer to `trace_event_call`.
 * @return True if error-injectable, false otherwise.
 */
bool trace_kprobe_error_injectable(struct trace_event_call *call);
#else
static inline bool trace_kprobe_on_func_entry(struct trace_event_call *call)
{
	return false;
}

static inline bool trace_kprobe_error_injectable(struct trace_event_call *call)
{
	return false;
}
#endif /* CONFIG_KPROBE_EVENTS */

/**
 * @struct probe_arg
 * @brief Represents a single argument of a trace probe.
 *
 * This structure stores the parsed information about a probe argument,
 * including its name, type, and the sequence of fetch instructions.
 */
struct probe_arg {
	struct fetch_insn	*code;		/**< @brief Array of fetch instructions for this argument. */
	bool			dynamic;	/**< @brief True if it's a dynamic array (e.g., string). */
	unsigned int		offset;		/**< @brief Offset from the argument entry in the trace record. */
	unsigned int		count;		/**< @brief Array count (0 if not an array). */
	const char		*name;		/**< @brief Name of this argument. */
	const char		*comm;		/**< @brief Comment or original command string for this argument. */
	char			*fmt;		/**< @brief Custom format string if needed. */
	const struct fetch_type	*type;		/**< @brief Type of this argument. */
};

/**
 * @struct probe_entry_arg
 * @brief Stores information about arguments to be fetched at function entry for kretprobes.
 *
 * This is used to capture arguments when a function is entered, so they can
 * be used later by a kretprobe's exit handler.
 */
struct probe_entry_arg {
	struct fetch_insn	*code;		/**< @brief Array of fetch instructions for entry data. */
	unsigned int		size;		/**< @brief The total size of entry data. */
};

/**
 * @struct trace_uprobe_filter
 * @brief Filter for uprobe events.
 *
 * This structure manages the system-wide and per-event filters for uprobes.
 */
struct trace_uprobe_filter {
	rwlock_t		rwlock;		/**< @brief Read-write lock for filter access. */
	int			nr_systemwide;	/**< @brief Number of system-wide uprobes. */
	struct list_head	perf_events;	/**< @brief List of perf events. */
};

/**
 * @struct trace_probe_event
 * @brief Aggregates information for a trace probe event.
 *
 * This structure represents a logical trace event, potentially composed of
 * multiple `trace_probe` instances, and includes its class, call, and
 * associated files.
 */
struct trace_probe_event {
	unsigned int			flags;	/**< @brief Flags for the event (e.g., `TP_FLAG_TRACE`). */
	struct trace_event_class	class;	/**< @brief The trace event class. */
	struct trace_event_call		call;	/**< @brief The trace event call structure. */
	struct list_head 		files;	/**< @brief List of `trace_event_file` links. */
	struct list_head		probes;	/**< @brief List of associated `trace_probe` instances. */
	struct trace_uprobe_filter	filter[]; /**< @brief Uprobe filter (if applicable). */
};

/**
 * @struct trace_probe
 * @brief Main structure for a trace probe instance.
 *
 * This structure defines a single dynamic probe, linking it to its parent
 * `trace_probe_event`, holding its size, arguments, and entry data.
 */
struct trace_probe {
	struct list_head		list;		/**< @brief List head for linking probes. */
	struct trace_probe_event	*event;		/**< @brief Pointer to the parent trace probe event. */
	ssize_t				size;		/**< @brief Size of the trace entry data. */
	unsigned int			nr_args;	/**< @brief Number of arguments for this probe. */
	struct probe_entry_arg		*entry_arg;	/**< @brief Entry data arguments (for return probes). */
	struct probe_arg		args[];		/**< @brief Array of probe arguments. */
};

/**
 * @struct event_file_link
 * @brief Links a `trace_probe` to a `trace_event_file`.
 *
 * This structure is used to manage the association between a trace probe
 * and the trace event files it is enabled on.
 */
struct event_file_link {
	struct trace_event_file		*file;	/**< @brief Pointer to the linked `trace_event_file`. */
	struct list_head		list;	/**< @brief List head for event file links. */
};

/**
 * @brief Tests if a specific flag is set for a `trace_probe`.
 * @param tp Pointer to the `trace_probe`.
 * @param flag The flag to test.
 * @return True if the flag is set, false otherwise.
 */
static inline bool trace_probe_test_flag(struct trace_probe *tp,
					 unsigned int flag)
{
	return !!(tp->event->flags & flag);
}

/**
 * @brief Sets a specific flag for a `trace_probe`.
 * @param tp Pointer to the `trace_probe`.
 * @param flag The flag to set.
 */
static inline void trace_probe_set_flag(struct trace_probe *tp,
					unsigned int flag)
{
	tp->event->flags |= flag;
}

/**
 * @brief Clears a specific flag for a `trace_probe`.
 * @param tp Pointer to the `trace_probe`.
 * @param flag The flag to clear.
 */
static inline void trace_probe_clear_flag(struct trace_probe *tp,
					  unsigned int flag)
{
	tp->event->flags &= ~flag;
}

/**
 * @brief Checks if a `trace_probe` is enabled (either for tracing or profiling).
 * @param tp Pointer to the `trace_probe`.
 * @return True if enabled, false otherwise.
 */
static inline bool trace_probe_is_enabled(struct trace_probe *tp)
{
	return trace_probe_test_flag(tp, TP_FLAG_TRACE | TP_FLAG_PROFILE);
}

/**
 * @brief Retrieves the name of a `trace_probe` event.
 * @param tp Pointer to the `trace_probe`.
 * @return The event name as a C string.
 */
static inline const char *trace_probe_name(struct trace_probe *tp)
{
	return trace_event_name(&tp->event->call);
}

/**
 * @brief Retrieves the group name of a `trace_probe` event.
 * @param tp Pointer to the `trace_probe`.
 * @return The group name as a C string.
 */
static inline const char *trace_probe_group_name(struct trace_probe *tp)
{
	return tp->event->call.class->system;
}

/**
 * @brief Retrieves the `trace_event_call` associated with a `trace_probe`.
 * @param tp Pointer to the `trace_probe`.
 * @return Pointer to the `trace_event_call`.
 */
static inline struct trace_event_call *
	trace_probe_event_call(struct trace_probe *tp)
{
	return &tp->event->call;
}

/**
 * @brief Retrieves the `trace_probe_event` from a `trace_event_call`.
 * @param event_call Pointer to the `trace_event_call`.
 * @return Pointer to the `trace_probe_event`.
 */
static inline struct trace_probe_event *
trace_probe_event_from_call(struct trace_event_call *event_call)
{
	return container_of(event_call, struct trace_probe_event, call);
}

/**
 * @brief Retrieves the primary `trace_probe` from a `trace_event_call`.
 * @param call Pointer to the `trace_event_call`.
 * @return Pointer to the primary `trace_probe` instance.
 *
 * The primary probe is the first probe in the list of probes associated
 * with an event.
 */
static inline struct trace_probe *
trace_probe_primary_from_call(struct trace_event_call *call)
{
	struct trace_probe_event *tpe = trace_probe_event_from_call(call);

	return list_first_entry_or_null(&tpe->probes, struct trace_probe, list);
}

/**
 * @brief Retrieves the list head of probes associated with a `trace_probe`.
 * @param tp Pointer to the `trace_probe`.
 * @return Pointer to the `list_head` of probes.
 */
static inline struct list_head *trace_probe_probe_list(struct trace_probe *tp)
{
	return &tp->event->probes;
}

/**
 * @brief Checks if a `trace_probe` has siblings (other probes in the same event).
 * @param tp Pointer to the `trace_probe`.
 * @return True if it has siblings, false otherwise.
 */
static inline bool trace_probe_has_sibling(struct trace_probe *tp)
{
	struct list_head *list = trace_probe_probe_list(tp);

	return !list_empty(list) && !list_is_singular(list);
}

/**
 * @brief Unregisters the `trace_event_call` associated with a `trace_probe`.
 * @param tp Pointer to the `trace_probe`.
 * @return 0 on success, or a negative errno on failure.
 */
static inline int trace_probe_unregister_event_call(struct trace_probe *tp)
{
	/* tp->event is unregistered in trace_remove_event_call() */
	return trace_remove_event_call(&tp->event->call);
}

/**
 * @brief Checks if a `trace_probe` has only a single associated file.
 * @param tp Pointer to the `trace_probe`.
 * @return True if only one file, false otherwise.
 */
static inline bool trace_probe_has_single_file(struct trace_probe *tp)
{
	return !!list_is_singular(&tp->event->files);
}

/**
 * @brief Initializes a `trace_probe` structure.
 * @param tp Pointer to `trace_probe`.
 * @param event Event name.
 * @param group Group name.
 * @param alloc_filter True if filter should be allocated.
 * @param nargs Number of arguments.
 * @return 0 on success, or a negative errno on failure.
 */
int trace_probe_init(struct trace_probe *tp, const char *event,
		     const char *group, bool alloc_filter, int nargs);
/**
 * @brief Cleans up resources associated with a `trace_probe`.
 * @param tp Pointer to `trace_probe`.
 */
void trace_probe_cleanup(struct trace_probe *tp);
/**
 * @brief Appends a `trace_probe` to an existing one.
 * @param tp Pointer to the `trace_probe` to append.
 * @param to Pointer to the `trace_probe` to append to.
 * @return 0 on success, or -EBUSY if `tp` has siblings.
 */
int trace_probe_append(struct trace_probe *tp, struct trace_probe *to);
/**
 * @brief Unlinks a `trace_probe` from its event list.
 * @param tp Pointer to `trace_probe`.
 */
void trace_probe_unlink(struct trace_probe *tp);
/**
 * @brief Registers the `trace_event_call` for a `trace_probe`.
 * @param tp Pointer to `trace_probe`.
 * @return 0 on success, or a negative errno on failure.
 */
int trace_probe_register_event_call(struct trace_probe *tp);
/**
 * @brief Adds a `trace_event_file` to a `trace_probe`.
 * @param tp Pointer to `trace_probe`.
 * @param file Pointer to `trace_event_file`.
 * @return 0 on success, or a negative errno on failure.
 */
int trace_probe_add_file(struct trace_probe *tp, struct trace_event_file *file);
/**
 * @brief Removes a `trace_event_file` from a `trace_probe`.
 * @param tp Pointer to `trace_probe`.
 * @param file Pointer to `trace_event_file`.
 * @return 0 on success, or a negative errno on failure.
 */
int trace_probe_remove_file(struct trace_probe *tp,
			    struct trace_event_file *file);
/**
 * @brief Retrieves the link between a `trace_probe` and a `trace_event_file`.
 * @param tp Pointer to `trace_probe`.
 * @param file Pointer to `trace_event_file`.
 * @return Pointer to `event_file_link`, or NULL if not found.
 */
struct event_file_link *trace_probe_get_file_link(struct trace_probe *tp,
						struct trace_event_file *file);
/**
 * @brief Compares argument types between two `trace_probe`s.
 * @param a Pointer to the first `trace_probe`.
 * @param b Pointer to the second `trace_probe`.
 * @return 0 if types match, or an index indicating the mismatch.
 */
int trace_probe_compare_arg_type(struct trace_probe *a, struct trace_probe *b);
/**
 * @brief Matches command arguments against a `trace_probe`.
 * @param tp Pointer to `trace_probe`.
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @return True if arguments match, false otherwise.
 */
bool trace_probe_match_command_args(struct trace_probe *tp,
				    int argc, const char **argv);
/**
 * @brief Creates a trace probe from a raw command string.
 * @param raw_command The raw command string.
 * @param createfn Function to create the probe.
 * @return 0 on success, or a negative errno on failure.
 */
int trace_probe_create(const char *raw_command, int (*createfn)(int, const char **));
/**
 * @brief Prints probe arguments into a `trace_seq`.
 * @param s Pointer to `trace_seq`.
 * @param args Array of `probe_arg`.
 * @param nr_args Number of arguments.
 * @param data Pointer to raw data.
 * @param field Pointer to the event field.
 * @return Number of characters printed.
 */
int trace_probe_print_args(struct trace_seq *s, struct probe_arg *args, int nr_args,
		 u8 *data, void *field);

#ifdef CONFIG_HAVE_FUNCTION_ARG_ACCESS_API
/**
 * @brief Calculates the size of the entry data buffer required for a trace probe.
 * @param tp Pointer to `trace_probe`.
 * @return The size of the entry data buffer in bytes.
 */
int traceprobe_get_entry_data_size(struct trace_probe *tp);
/* This is a runtime function to store entry data */
/**
 * @brief Stores trace entry data into the `entry_data` buffer.
 * @param edata Pointer to the `entry_data` buffer.
 * @param tp Pointer to `trace_probe`.
 * @param regs Pointer to `pt_regs`.
 */
void store_trace_entry_data(void *edata, struct trace_probe *tp, struct pt_regs *regs);
#else /* !CONFIG_HAVE_FUNCTION_ARG_ACCESS_API */
static inline int traceprobe_get_entry_data_size(struct trace_probe *tp)
{
	return 0;
}
#define store_trace_entry_data(edata, tp, regs) do { } while (0)
#endif

/**
 * @def trace_probe_for_each_link(pos, tp)
 * @brief Macro to iterate over `event_file_link`s associated with a `trace_probe`.
 */
#define trace_probe_for_each_link(pos, tp)	\
	list_for_each_entry(pos, &(tp)->event->files, list)
/**
 * @def trace_probe_for_each_link_rcu(pos, tp)
 * @brief RCU-safe macro to iterate over `event_file_link`s.
 */
#define trace_probe_for_each_link_rcu(pos, tp)	\
	list_for_each_entry_rcu(pos, &(tp)->event->files, list)

/*
 * The flags used for parsing trace_probe arguments.
 * TPARG_FL_RETURN, TPARG_FL_FENTRY and TPARG_FL_TEVENT are mutually exclusive.
 * TPARG_FL_KERNEL and TPARG_FL_USER are also mutually exclusive.
 * TPARG_FL_FPROBE and TPARG_FL_TPOINT are optional but it should be with
 * TPARG_FL_KERNEL.
 */
/**
 * @def TPARG_FL_RETURN
 * @brief Flag indicating a return probe argument.
 */
#define TPARG_FL_RETURN BIT(0)
/**
 * @def TPARG_FL_KERNEL
 * @brief Flag indicating a kernel probe argument.
 */
#define TPARG_FL_KERNEL BIT(1)
/**
 * @def TPARG_FL_FENTRY
 * @brief Flag indicating a function entry probe argument.
 */
#define TPARG_FL_FENTRY BIT(2)
/**
 * @def TPARG_FL_TEVENT
 * @brief Flag indicating a trace event argument.
 */
#define TPARG_FL_TEVENT BIT(3)
/**
 * @def TPARG_FL_USER
 * @brief Flag indicating a user-space probe argument.
 */
#define TPARG_FL_USER   BIT(4)
/**
 * @def TPARG_FL_FPROBE
 * @brief Flag indicating an fprobe argument.
 */
#define TPARG_FL_FPROBE BIT(5)
/**
 * @def TPARG_FL_TPOINT
 * @brief Flag indicating a tracepoint argument.
 */
#define TPARG_FL_TPOINT BIT(6)
/**
 * @def TPARG_FL_LOC_MASK
 * @brief Mask for location-related flags.
 */
#define TPARG_FL_LOC_MASK	GENMASK(4, 0)

/**
 * @brief Checks if trace probe arguments are for a function entry in the kernel.
 * @param flags The `TPARG_FL_*` flags.
 * @return True if function entry in kernel, false otherwise.
 */
static inline bool tparg_is_function_entry(unsigned int flags)
{
	return (flags & TPARG_FL_LOC_MASK) == (TPARG_FL_KERNEL | TPARG_FL_FENTRY);
}

/**
 * @brief Checks if trace probe arguments are for a function return in the kernel.
 * @param flags The `TPARG_FL_*` flags.
 * @return True if function return in kernel, false otherwise.
 */
static inline bool tparg_is_function_return(unsigned int flags)
{
	return (flags & TPARG_FL_LOC_MASK) == (TPARG_FL_KERNEL | TPARG_FL_RETURN);
}

/**
 * @struct traceprobe_parse_context
 * @brief Context structure for parsing trace probe arguments.
 *
 * This structure holds state and intermediate results during the parsing
 * of probe arguments, including BTF information and error reporting context.
 */
struct traceprobe_parse_context {
	struct trace_event_call *event;	/**< @brief The target `trace_event_call`. */
	/* BTF related parameters */
	const char *funcname;		/**< @brief Function name in BTF. */
	const struct btf_type  *proto;	/**< @brief Prototype of the function (from BTF). */
	const struct btf_param *params;	/**< @brief Parameters of the function (from BTF). */
	s32 nr_params;			/**< @brief The number of the parameters. */
	struct btf *btf;		/**< @brief The BTF instance to be used. */
	const struct btf_type *last_type;	/**< @brief Saved last BTF type during parsing. */
	u32 last_bitoffs;		/**< @brief Saved last bit offset. */
	u32 last_bitsize;		/**< @brief Saved last bit size. */
	struct trace_probe *tp;		/**< @brief Current `trace_probe` being parsed. */
	unsigned int flags;		/**< @brief `TPARG_FL_*` flags. */
	int offset;			/**< @brief Current offset in the argument string for error reporting. */
};

extern int traceprobe_parse_probe_arg(struct trace_probe *tp, int i,
				      const char *argv,
				      struct traceprobe_parse_context *ctx);
/**
 * @brief Expands meta arguments (like `$arg*`, `$argN`) in an argument list.
 * @param argc Number of original arguments.
 * @param argv Array of original argument strings.
 * @param new_argc Output parameter for the new argument count.
 * @param buf Buffer to store expanded arguments.
 * @param bufsize Size of the buffer.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return A newly allocated array of expanded argument strings on success,
 *         or an `ERR_PTR` on failure. The caller must free this array.
 */
const char **traceprobe_expand_meta_args(int argc, const char *argv[],
					 int *new_argc, char *buf, int bufsize,
					 struct traceprobe_parse_context *ctx);
/**
 * @brief Expands dentry arguments (e.g., `%pD`) in an argument list.
 * @param argc Number of original arguments.
 * @param argv Array of original argument strings.
 * @param buf Output parameter for the newly allocated buffer containing expanded arguments.
 * @return 0 on success, -EINVAL on error, -ENOMEM on memory allocation failure.
 */
extern int traceprobe_expand_dentry_args(int argc, const char *argv[], char **buf);

extern int traceprobe_update_arg(struct probe_arg *arg);
extern void traceprobe_free_probe_arg(struct probe_arg *arg);

/*
 * If either traceprobe_parse_probe_arg() or traceprobe_expand_meta_args() is called,
 * this MUST be called for clean up the context and return a resource.
 */
/**
 * @brief Cleans up the `traceprobe_parse_context` after parsing.
 * @param ctx Pointer to `traceprobe_parse_context`.
 */
void traceprobe_finish_parse(struct traceprobe_parse_context *ctx);

extern int traceprobe_split_symbol_offset(char *symbol, long *offset);
/**
 * @brief Parses an event name and group from a string.
 * @param pevent Input/Output: Pointer to the event string.
 * @param pgroup Input/Output: Pointer to the group string.
 * @param buf Buffer to store the group name.
 * @param offset Starting offset in the original command string for error reporting.
 * @return 0 on success, or a negative errno on failure.
 */
int traceprobe_parse_event_name(const char **pevent, const char **pgroup,
				char *buf, int offset);

/**
 * @enum probe_print_type
 * @brief Enumeration of different print types for trace probes.
 */
enum probe_print_type {
	PROBE_PRINT_NORMAL,	/**< @brief Normal (function entry) print format. */
	PROBE_PRINT_RETURN,	/**< @brief Function return print format. */
	PROBE_PRINT_EVENT,	/**< @brief Event-specific print format. */
};

extern int traceprobe_set_print_fmt(struct trace_probe *tp, enum probe_print_type ptype);

#ifdef CONFIG_PERF_EVENTS
extern struct trace_event_call *
create_local_trace_kprobe(char *func, void *addr, unsigned long offs,
			  bool is_return);
extern void destroy_local_trace_kprobe(struct trace_event_call *event_call);

extern struct trace_event_call *
create_local_trace_uprobe(char *name, unsigned long offs,
			  unsigned long ref_ctr_offset, bool is_return);
extern void destroy_local_trace_uprobe(struct trace_event_call *event_call);
#endif
extern int traceprobe_define_arg_fields(struct trace_event_call *event_call,
					size_t offset, struct trace_probe *tp);

#undef ERRORS
#define ERRORS	\
	C(FILE_NOT_FOUND,	"Failed to find the given file"),	\
	C(NO_REGULAR_FILE,	"Not a regular file"),			\
	C(BAD_REFCNT,		"Invalid reference counter offset"),	\
	C(REFCNT_OPEN_BRACE,	"Reference counter brace is not closed"), \
	C(BAD_REFCNT_SUFFIX,	"Reference counter has wrong suffix"),	\
	C(BAD_UPROBE_OFFS,	"Invalid uprobe offset"),		\
	C(BAD_MAXACT_TYPE,	"Maxactive is only for function exit"),	\
	C(BAD_MAXACT,		"Invalid maxactive number"),		\
	C(MAXACT_TOO_BIG,	"Maxactive is too big"),		\
	C(BAD_PROBE_ADDR,	"Invalid probed address or symbol"),	\
	C(NON_UNIQ_SYMBOL,	"The symbol is not unique"),		\
	C(BAD_RETPROBE,		"Retprobe address must be an function entry"), \
	C(NO_TRACEPOINT,	"Tracepoint is not found"),		\
	C(BAD_TP_NAME,		"Invalid character in tracepoint name"),\
	C(BAD_ADDR_SUFFIX,	"Invalid probed address suffix"), \
	C(NO_GROUP_NAME,	"Group name is not specified"),		\
	C(GROUP_TOO_LONG,	"Group name is too long"),		\
	C(BAD_GROUP_NAME,	"Group name must follow the same rules as C identifiers"), \
	C(NO_EVENT_NAME,	"Event name is not specified"),		\
	C(EVENT_TOO_LONG,	"Event name is too long"),		\
	C(BAD_EVENT_NAME,	"Event name must follow the same rules as C identifiers"), \
	C(EVENT_EXIST,		"Given group/event name is already used by another event"), \
	C(RETVAL_ON_PROBE,	"$retval is not available on probe"),	\
	C(NO_RETVAL,		"This function returns 'void' type"),	\
	C(BAD_STACK_NUM,	"Invalid stack number"),		\
	C(BAD_ARG_NUM,		"Invalid argument number"),		\
	C(BAD_VAR,		"Invalid $-valiable specified"),	\
	C(BAD_REG_NAME,		"Invalid register name"),		\
	C(BAD_MEM_ADDR,		"Invalid memory address"),		\
	C(BAD_IMM,		"Invalid immediate value"),		\
	C(IMMSTR_NO_CLOSE,	"String is not closed with '\"'"),	\
	C(FILE_ON_KPROBE,	"File offset is not available with kprobe"), \
	C(BAD_FILE_OFFS,	"Invalid file offset value"),		\
	C(SYM_ON_UPROBE,	"Symbol is not available with uprobe"),	\
	C(TOO_MANY_OPS,		"Dereference is too much nested"), 	\
	C(DEREF_NEED_BRACE,	"Dereference needs a brace"),		\
	C(BAD_DEREF_OFFS,	"Invalid dereference offset"),		\
	C(DEREF_OPEN_BRACE,	"Dereference brace is not closed"),	\
	C(COMM_CANT_DEREF,	"$comm can not be dereferenced"),	\
	C(BAD_FETCH_ARG,	"Invalid fetch argument"),		\
	C(ARRAY_NO_CLOSE,	"Array is not closed"),			\
	C(BAD_ARRAY_SUFFIX,	"Array has wrong suffix"),		\
	C(BAD_ARRAY_NUM,	"Invalid array size"),			\
	C(ARRAY_TOO_BIG,	"Array number is too big"),		\
	C(BAD_TYPE,		"Unknown type is specified"),		\
	C(BAD_STRING,		"String accepts only memory argument"),	\
	C(BAD_SYMSTRING,	"Symbol String doesn't accept data/userdata"),	\
	C(BAD_BITFIELD,		"Invalid bitfield"),			\
	C(ARG_NAME_TOO_LONG,	"Argument name is too long"),		\
	C(NO_ARG_NAME,		"Argument name is not specified"),	\
	C(BAD_ARG_NAME,		"Argument name must follow the same rules as C identifiers"), \
	C(USED_ARG_NAME,	"This argument name is already used"),	\
	C(ARG_TOO_LONG,		"Argument expression is too long"),	\
	C(NO_ARG_BODY,		"No argument expression"),		\
	C(BAD_INSN_BNDRY,	"Probe point is not an instruction boundary"),\
	C(FAIL_REG_PROBE,	"Failed to register probe event"),\
	C(DIFF_PROBE_TYPE,	"Probe type is different from existing probe"),\
	C(DIFF_ARG_TYPE,	"Argument type or name is different from existing probe"),\
	C(SAME_PROBE,		"There is already the exact same probe event"),\
	C(NO_EVENT_INFO,	"This requires both group and event name to attach"),\
	C(BAD_ATTACH_EVENT,	"Attached event does not exist"),\
	C(BAD_ATTACH_ARG,	"Attached event does not have this field"),\
	C(NO_EP_FILTER,		"No filter rule after 'if'"),		\
	C(NOSUP_BTFARG,		"BTF is not available or not supported"),	\
	C(NO_BTFARG,		"This variable is not found at this probe point"),\
	C(NO_BTF_ENTRY,		"No BTF entry for this probe point"),	\
	C(BAD_VAR_ARGS,		"$arg* must be an independent parameter without name etc."),\
	C(NOFENTRY_ARGS,	"$arg* can be used only on function entry or exit"),	\
	C(DOUBLE_ARGS,		"$arg* can be used only once in the parameters"),	\
	C(ARGS_2LONG,		"$arg* failed because the argument list is too long"),	\
	C(ARGIDX_2BIG,		"$argN index is too big"),		\
	C(NO_PTR_STRCT,		"This is not a pointer to union/structure."),	\
	C(NOSUP_DAT_ARG,	"Non pointer structure/union argument is not supported."),\
	C(BAD_HYPHEN,		"Failed to parse single hyphen. Forgot '>''?"),	\
	C(NO_BTF_FIELD,		"This field is not found."),	\
	C(BAD_BTF_TID,		"Failed to get BTF type info."),\
	C(BAD_TYPE4STR,		"This type does not fit for string."),\
	C(NEED_STRING_TYPE,	"$comm and immediate-string only accepts string type"),\
	C(TOO_MANY_ARGS,	"Too many arguments are specified"),	\
	C(TOO_MANY_EARGS,	"Too many entry arguments specified"),

#undef C
/**
 * @def C(a, b)
 * @brief Helper macro for constructing enum error codes.
 */
#define C(a, b)		TP_ERR_##a

/* Define TP_ERR_ */
/**
 * @enum (anonymous)
 * @brief Enumeration of trace probe error codes.
 *
 * These error codes are used internally for detailed error reporting during
 * probe parsing and registration.
 */
enum { ERRORS };

/* Error text is defined in trace_probe.c */

/**
 * @struct trace_probe_log
 * @brief Context structure for trace probe error logging.
 *
 * Stores information about the subsystem, command-line arguments,
 * and current argument index for precise error reporting.
 */
struct trace_probe_log {
	const char	*subsystem;	/**< @brief Subsystem name (e.g., "kprobes"). */
	const char	**argv;		/**< @brief Array of command-line arguments. */
	int		argc;		/**< @brief Number of arguments. */
	int		index;		/**< @brief Index of the argument currently being parsed. */
};

/**
 * @brief Initializes the trace probe error logging context.
 * @param subsystem The name of the subsystem using trace probes (e.g., "kprobes").
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 */
void trace_probe_log_init(const char *subsystem, int argc, const char **argv);
/**
 * @brief Sets the argument index for trace probe error logging.
 * @param index The index of the argument where the error occurred.
 */
void trace_probe_log_set_index(int index);
/**
 * @brief Clears the trace probe error logging context.
 */
void trace_probe_log_clear(void);
/**
 * @brief Logs a trace probe parsing error.
 * @param offset The offset within the current argument where the error occurred.
 * @param err The error code (`TP_ERR_*`).
 */
void __trace_probe_log_err(int offset, int err);

/**
 * @def trace_probe_log_err(offs, err)
 * @brief Macro for logging trace probe errors with specific offset and error type.
 */
#define trace_probe_log_err(offs, err)	\
	__trace_probe_log_err(offs, TP_ERR_##err)

/**
 * @struct uprobe_dispatch_data
 * @brief Data structure for uprobe dispatching.
 *
 * Contains information about the `trace_uprobe` and the breakpoint address.
 */
struct uprobe_dispatch_data {
	struct trace_uprobe	*tu;		/**< @brief Pointer to `trace_uprobe` instance. */
	unsigned long		bp_addr;	/**< @brief Breakpoint address. */
};