/**
 * @file trace_probe.c
 * @brief Provides common code for probe-based Dynamic events in the Linux kernel.
 *
 * This file implements generic functionalities shared across various dynamic
 * tracing mechanisms like kprobes, uprobes, eprobes, and fprobes. It handles
 * the parsing of probe arguments, generation of fetch instructions to extract
 * data from kernel contexts, formatting of output, and integration with
 * BTF (BPF Type Format) for type-aware argument parsing.
 *
 * Functional Utility:
 * - **Argument Parsing**: Interprets user-provided probe arguments, including
 *   variables, registers, stack offsets, memory dereferences, and immediate values.
 * - **Fetch Instruction Generation**: Translates parsed arguments into a sequence
 *   of internal fetch operations to efficiently retrieve data from the probed context.
 * - **Output Formatting**: Generates `print_fmt` strings for ftrace events,
 *   allowing flexible output of captured data.
 * - **BTF Integration**: Uses BTF information for more accurate and type-aware
 *   argument parsing, especially for function arguments and structure members.
 * - **Error Reporting**: Provides detailed error logging for parsing and
 *   semantic issues.
 *
 * Algorithms:
 * - Recursive parsing of probe arguments to build a chain of `fetch_insn`
 *   structures.
 * - String manipulation and tokenization for command and argument parsing.
 * - Integration with kernel symbol lookup (`kallsyms`) and ftrace event
 *   subsystem.
 *
 * Architectural Intent:
 * - To create a unified and extensible framework for dynamic kernel tracing.
 * - To minimize code duplication across different probe types.
 * - To provide a powerful, yet user-friendly, command-line interface for
 *   defining complex trace events.
 *
 * This code was copied from kernel/trace/trace_kprobe.c written by
 * Masami Hiramatsu <masami.hiramatsu.pt@hitachi.com>
 *
 * Updates to make this generic:
 * Copyright (C) IBM Corporation, 2010-2011
 * Author:     Srikar Dronamraju
 */
// SPDX-License-Identifier: GPL-2.0
#define pr_fmt(fmt)	"trace_probe: " fmt

#include <linux/bpf.h>
#include <linux/fs.h>
#include "trace_btf.h"

#include "trace_probe.h"

#undef C
#define C(a, b)		b

/**
 * @var trace_probe_err_text
 * @brief Array of error messages corresponding to `TP_ERR_*` codes.
 */
static const char *trace_probe_err_text[] = { ERRORS };

/**
 * @var reserved_field_names
 * @brief Array of field names reserved by the tracing system.
 *
 * These names cannot be used as argument names in probes to avoid conflicts
 * with common event fields.
 */
static const char *reserved_field_names[] = {
	"common_type",
	"common_flags",
	"common_preempt_count",
	"common_pid",
	"common_tgid",
	FIELD_STRING_IP,
	FIELD_STRING_RETIP,
	FIELD_STRING_FUNC,
};

/* Printing  in basic type function template */
/**
 * @def DEFINE_BASIC_PRINT_TYPE_FUNC(tname, type, fmt)
 * @brief Macro to define basic print functions for various data types.
 *
 * This macro generates a print function and its corresponding format string
 * for basic integer types, simplifying the creation of trace output formatters.
 */
#define DEFINE_BASIC_PRINT_TYPE_FUNC(tname, type, fmt)			\
int PRINT_TYPE_FUNC_NAME(tname)(struct trace_seq *s, void *data, void *ent)\
{									\
	trace_seq_printf(s, fmt, *(type *)data);			\
	return !trace_seq_has_overflowed(s);				\
}									\
const char PRINT_TYPE_FMT_NAME(tname)[] = fmt;

DEFINE_BASIC_PRINT_TYPE_FUNC(u8,  u8,  "%u")
DEFINE_BASIC_PRINT_TYPE_FUNC(u16, u16, "%u")
DEFINE_BASIC_PRINT_TYPE_FUNC(u32, u32, "%u")
DEFINE_BASIC_PRINT_TYPE_FUNC(u64, u64, "%Lu")
DEFINE_BASIC_PRINT_TYPE_FUNC(s8,  s8,  "%d")
DEFINE_BASIC_PRINT_TYPE_FUNC(s16, s16, "%d")
DEFINE_BASIC_PRINT_TYPE_FUNC(s32, s32, "%d")
DEFINE_BASIC_PRINT_TYPE_FUNC(s64, s64, "%Ld")
DEFINE_BASIC_PRINT_TYPE_FUNC(x8,  u8,  "0x%x")
DEFINE_BASIC_PRINT_TYPE_FUNC(x16, u16, "0x%x")
DEFINE_BASIC_PRINT_TYPE_FUNC(x32, u32, "0x%x")
DEFINE_BASIC_PRINT_TYPE_FUNC(x64, u64, "0x%Lx")
DEFINE_BASIC_PRINT_TYPE_FUNC(char, u8, "'%c'")

/**
 * @brief Print function for symbol addresses.
 * @param s Pointer to `trace_seq`.
 * @param data Pointer to the symbol address (`unsigned long`).
 * @param ent Not used.
 * @return True on success, false if `trace_seq` overflowed.
 *
 * Formats and prints a symbol address using `%pS` for symbol resolution.
 */
int PRINT_TYPE_FUNC_NAME(symbol)(struct trace_seq *s, void *data, void *ent)
{
	trace_seq_printf(s, "%pS", (void *)*(unsigned long *)data);
	return !trace_seq_has_overflowed(s);
}
/**
 * @var PRINT_TYPE_FMT_NAME(symbol)
 * @brief Format string for symbol addresses.
 */
const char PRINT_TYPE_FMT_NAME(symbol)[] = "%pS";

/* Print type function for string type */
/**
 * @brief Print function for string types.
 * @param s Pointer to `trace_seq`.
 * @param data Pointer to the `u32` containing length and location data.
 * @param ent Pointer to the raw event entry.
 * @return True on success, false if `trace_seq` overflowed.
 *
 * Prints a string, handling potential fault cases and dynamic string data.
 */
int PRINT_TYPE_FUNC_NAME(string)(struct trace_seq *s, void *data, void *ent)
{
	int len = *(u32 *)data >> 16;

	if (!len)
		trace_seq_puts(s, FAULT_STRING);
	else
		trace_seq_printf(s, "\"%s\"",
				 (const char *)get_loc_data(data, ent));
	return !trace_seq_has_overflowed(s);
}

/**
 * @var PRINT_TYPE_FMT_NAME(string)
 * @brief Format string for string types.
 */
const char PRINT_TYPE_FMT_NAME(string)[] = "\\\"%s\\\"";

/* Fetch type information table */
/**
 * @var probe_fetch_types
 * @brief Array of supported fetch types for probe arguments.
 *
 * This table defines how different data types are fetched and formatted,
 * including special handling for strings and aliases for common types.
 */
static const struct fetch_type probe_fetch_types[] = {
	/* Special types */
	__ASSIGN_FETCH_TYPE("string", string, string, sizeof(u32), 1, 1,
			    "__data_loc char[]"),
	__ASSIGN_FETCH_TYPE("ustring", string, string, sizeof(u32), 1, 1,
			    "__data_loc char[]"),
	__ASSIGN_FETCH_TYPE("symstr", string, string, sizeof(u32), 1, 1,
			    "__data_loc char[]"),
	/* Basic types */
	ASSIGN_FETCH_TYPE(u8,  u8,  0),
	ASSIGN_FETCH_TYPE(u16, u16, 0),
	ASSIGN_FETCH_TYPE(u32, u32, 0),
	ASSIGN_FETCH_TYPE(u64, u64, 0),
	ASSIGN_FETCH_TYPE(s8,  u8,  1),
	ASSIGN_FETCH_TYPE(s16, s16, 1),
	ASSIGN_FETCH_TYPE(s32, u32, 1),
	ASSIGN_FETCH_TYPE(s64, u64, 1),
	ASSIGN_FETCH_TYPE_ALIAS(x8,  u8,  u8,  0),
	ASSIGN_FETCH_TYPE_ALIAS(x16, u16, u16, 0),
	ASSIGN_FETCH_TYPE_ALIAS(x32, u32, u32, 0),
	ASSIGN_FETCH_TYPE_ALIAS(x64, u64, u64, 0),
	ASSIGN_FETCH_TYPE_ALIAS(char, u8, u8,  0),
	ASSIGN_FETCH_TYPE_ALIAS(symbol, ADDR_FETCH_TYPE, ADDR_FETCH_TYPE, 0),

	ASSIGN_FETCH_TYPE_END
};

/**
 * @brief Finds a `fetch_type` by its name.
 * @param type The name of the type to find (e.g., "u32", "string").
 * @param flags Context flags for parsing (e.g., `TPARG_FL_USER`).
 * @return Pointer to the `fetch_type` structure, or NULL if not found.
 *
 * This function searches the `probe_fetch_types` table. It also handles
 * special cases like bitfield types (`b<bitsize>@<bitoffset>/<basesize>`).
 */
static const struct fetch_type *find_fetch_type(const char *type, unsigned long flags)
{
	int i;

	/* Reject the symbol/symstr for uprobes */
	// Block Logic: Uprobes do not support "symbol" or "symstr" types.
	if (type && (flags & TPARG_FL_USER) &&
	    (!strcmp(type, "symbol") || !strcmp(type, "symstr")))
		return NULL;

	// Block Logic: Uses default type if none specified.
	if (!type)
		type = DEFAULT_FETCH_TYPE_STR;

	/* Special case: bitfield */
	// Block Logic: Handles bitfield type parsing (e.g., "b<size>@<offset>/<basesize>").
	if (*type == 'b') {
		unsigned long bs;

		type = strchr(type, '/');
		if (!type)
			goto fail;

		type++;
		if (kstrtoul(type, 0, &bs))
			goto fail;

		switch (bs) {
		case 8:
			return find_fetch_type("u8", flags);
		case 16:
			return find_fetch_type("u16", flags);
		case 32:
			return find_fetch_type("u32", flags);
		case 64:
			return find_fetch_type("u64", flags);
		default:
			goto fail;
		}
	}

	// Block Logic: Iterates through `probe_fetch_types` to find a match.
	for (i = 0; probe_fetch_types[i].name; i++) {
		if (strcmp(type, probe_fetch_types[i].name) == 0)
			return &probe_fetch_types[i];
	}

fail:
	return NULL;
}

/**
 * @var trace_probe_log
 * @brief Global structure for logging trace probe parsing errors.
 */
static struct trace_probe_log trace_probe_log;
extern struct mutex dyn_event_ops_mutex;

/**
 * @brief Initializes the trace probe error logging context.
 * @param subsystem The name of the subsystem using trace probes (e.g., "kprobes").
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 *
 * This function sets up the context for logging parsing errors, including
 * the command-line arguments that are being processed.
 */
void trace_probe_log_init(const char *subsystem, int argc, const char **argv)
{
	lockdep_assert_held(&dyn_event_ops_mutex);

	trace_probe_log.subsystem = subsystem;
	trace_probe_log.argc = argc;
	trace_probe_log.argv = argv;
	trace_probe_log.index = 0;
}

/**
 * @brief Clears the trace probe error logging context.
 */
void trace_probe_log_clear(void)
{
	lockdep_assert_held(&dyn_event_ops_mutex);

	memset(&trace_probe_log, 0, sizeof(trace_probe_log));
}

/**
 * @brief Sets the argument index for trace probe error logging.
 * @param index The index of the argument where the error occurred.
 */
void trace_probe_log_set_index(int index)
{
	lockdep_assert_held(&dyn_event_ops_mutex);

	trace_probe_log.index = index;
}

/**
 * @brief Logs a trace probe parsing error.
 * @param offset The offset within the current argument where the error occurred.
 * @param err_type The type of the error (from `TP_ERR_*`).
 *
 * This function formats a detailed error message, including the command string
 * and the exact position of the error, and logs it using `tracing_log_err`.
 */
void __trace_probe_log_err(int offset, int err_type)
{
	char *command, *p;
	int i, len = 0, pos = 0;

	lockdep_assert_held(&dyn_event_ops_mutex);

	// Block Logic: Returns if no arguments are set in the log context.
	if (!trace_probe_log.argv)
		return;

	/* Recalculate the length and allocate buffer */
	// Block Logic: Calculates the total length of the command string and allocates memory.
	for (i = 0; i < trace_probe_log.argc; i++) {
		if (i == trace_probe_log.index)
			pos = len; // Functional Utility: Marks the position of the error argument.
		len += strlen(trace_probe_log.argv[i]) + 1;
	}
	command = kzalloc(len, GFP_KERNEL);
	if (!command)
		return;

	// Block Logic: Adjusts error position if index is out of bounds.
	if (trace_probe_log.index >= trace_probe_log.argc) {
		/**
		 * Set the error position is next to the last arg + space.
		 * Note that len includes the terminal null and the cursor
		 * appears at pos + 1.
		 */
		pos = len;
		offset = 0;
	}

	/* And make a command string from argv array */
	// Block Logic: Reconstructs the full command string from `argv`.
	p = command;
	for (i = 0; i < trace_probe_log.argc; i++) {
		len = strlen(trace_probe_log.argv[i]);
		strcpy(p, trace_probe_log.argv[i]);
		p[len] = ' '; // Functional Utility: Adds space between arguments.
		p += len + 1;
	}
	*(p - 1) = '\0'; // Functional Utility: Null-terminates the command string.

	tracing_log_err(NULL, trace_probe_log.subsystem, command,
			trace_probe_err_text, err_type, pos + offset);

	kfree(command); // Functional Utility: Frees allocated command string buffer.
}

/**
 * @brief Splits a symbol string into symbol name and offset.
 * @param symbol The input string (e.g., "my_func+0x10", "my_func-5").
 * @param offset Output parameter for the parsed offset.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function modifies the `symbol` string in place by null-terminating
 * it at the `+` or `-` character, and parses the offset.
 */
int traceprobe_split_symbol_offset(char *symbol, long *offset)
{
	char *tmp;
	int ret;

	// Block Logic: Checks for valid offset pointer.
	if (!offset)
		return -EINVAL;

	tmp = strpbrk(symbol, "+-"); // Functional Utility: Finds the first '+' or '-'.
	// Block Logic: If an offset separator is found.
	if (tmp) {
		ret = kstrtol(tmp, 0, offset); // Functional Utility: Parses the offset.
		if (ret)
			return ret;
		*tmp = '\0'; // Functional Utility: Null-terminates the symbol name.
	} else
		*offset = 0; // Functional Utility: No offset if no separator found.

	return 0;
}

/**
 * @brief Parses an event name and group from a string.
 * @param pevent Input/Output: Pointer to the event string.
 * @param pgroup Input/Output: Pointer to the group string.
 * @param buf Buffer to store the group name.
 * @param offset Starting offset in the original command string for error reporting.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function handles event names in the format `GROUP/EVENT` or `GROUP.EVENT`,
 * extracting the group and event components and validating their names.
 */
int traceprobe_parse_event_name(const char **pevent, const char **pgroup,
				char *buf, int offset)
{
	const char *slash, *event = *pevent;
	int len;

	slash = strchr(event, '/'); // Functional Utility: Checks for '/' separator.
	if (!slash)
		slash = strchr(event, '.'); // Functional Utility: Checks for '.' separator.

	// Block Logic: If a separator is found.
	if (slash) {
		if (slash == event) { // Block Logic: Group name cannot be empty.
			trace_probe_log_err(offset, NO_GROUP_NAME);
			return -EINVAL;
		}
		if (slash - event + 1 > MAX_EVENT_NAME_LEN) { // Block Logic: Group name too long.
			trace_probe_log_err(offset, GROUP_TOO_LONG);
			return -EINVAL;
		}
		strscpy(buf, event, slash - event + 1); // Functional Utility: Copies group name to buffer.
		if (!is_good_system_name(buf)) { // Functional Utility: Validates group name.
			trace_probe_log_err(offset, BAD_GROUP_NAME);
			return -EINVAL;
		}
		*pgroup = buf; // Functional Utility: Updates group pointer.
		*pevent = slash + 1; // Functional Utility: Updates event pointer to after separator.
		offset += slash - event + 1;
		event = *pevent;
	}
	len = strlen(event);
	// Block Logic: Validates event name length and characters.
	if (len == 0) {
		if (slash) {
			*pevent = NULL; // Functional Utility: If separator found, event can be NULL.
			return 0;
		}
		trace_probe_log_err(offset, NO_EVENT_NAME);
		return -EINVAL;
	} else if (len >= MAX_EVENT_NAME_LEN) {
		trace_probe_log_err(offset, EVENT_TOO_LONG);
		return -EINVAL;
	}
	if (!is_good_name(event)) { // Functional Utility: Validates event name.
		trace_probe_log_err(offset, BAD_EVENT_NAME);
		return -EINVAL;
	}
	return 0;
}

/**
 * @brief Parses a trace event argument as an existing field.
 * @param arg The argument string.
 * @param code Pointer to `fetch_insn` to store the parsed instruction.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return 0 on success, -ENOENT if field not found.
 *
 * This function attempts to match the argument string against existing fields
 * of the target trace event (`ctx->event`).
 */
static int parse_trace_event_arg(char *arg, struct fetch_insn *code,
				 struct traceprobe_parse_context *ctx)
{
	struct ftrace_event_field *field;
	struct list_head *head;

	head = trace_get_fields(ctx->event); // Functional Utility: Gets the list of event fields.
	list_for_each_entry(field, head, link) { // Functional Utility: Iterates through event fields.
		if (!strcmp(arg, field->name)) { // Block Logic: If field name matches.
			code->op = FETCH_OP_TP_ARG; // Functional Utility: Sets operation to tracepoint argument.
			code->data = field; // Functional Utility: Stores field information.
			return 0;
		}
	}
	return -ENOENT;
}

#ifdef CONFIG_PROBE_EVENTS_BTF_ARGS

/**
 * @brief Retrieves the integer data from a BTF integer type.
 * @param t Pointer to the BTF type structure.
 * @return The integer data as a `u32`.
 */
static u32 btf_type_int(const struct btf_type *t)
{
	return *(u32 *)(t + 1);
}

/**
 * @brief Checks if a BTF type is a pointer to a `char`.
 * @param btf Pointer to the BTF instance.
 * @param type Pointer to the BTF type to check.
 * @return True if it's a `char *`, false otherwise.
 */
static bool btf_type_is_char_ptr(struct btf *btf, const struct btf_type *type)
{
	const struct btf_type *real_type;
	u32 intdata;
	s32 tid;

	real_type = btf_type_skip_modifiers(btf, type->type, &tid); // Functional Utility: Skips modifiers to get real type.
	if (!real_type)
		return false;

	if (BTF_INFO_KIND(real_type->info) != BTF_KIND_INT) // Block Logic: Checks if it's an integer type.
		return false;

	intdata = btf_type_int(real_type); // Functional Utility: Retrieves integer data.
	return !(BTF_INT_ENCODING(intdata) & BTF_INT_SIGNED) // Block Logic: Checks for unsigned 8-bit integer.
		&& BTF_INT_BITS(intdata) == 8;
}

/**
 * @brief Checks if a BTF type is a character array.
 * @param btf Pointer to the BTF instance.
 * @param type Pointer to the BTF type to check.
 * @return True if it's a `char[]`, false otherwise.
 */
static bool btf_type_is_char_array(struct btf *btf, const struct btf_type *type)
{
	const struct btf_type *real_type;
	const struct btf_array *array;
	u32 intdata;
	s32 tid;

	if (BTF_INFO_KIND(type->info) != BTF_KIND_ARRAY) // Block Logic: Checks if it's an array type.
		return false;

	array = (const struct btf_array *)(type + 1); // Functional Utility: Gets array information.

	real_type = btf_type_skip_modifiers(btf, array->type, &tid); // Functional Utility: Skips modifiers to get real type.

	intdata = btf_type_int(real_type); // Functional Utility: Retrieves integer data.
	return !(BTF_INT_ENCODING(intdata) & BTF_INT_SIGNED) // Block Logic: Checks for unsigned 8-bit integer.
		&& BTF_INT_BITS(intdata) == 8;
}

/**
 * @brief Checks and prepares BTF-based string fetching.
 * @param typename The name of the type.
 * @param pcode Input/Output: Pointer to the current `fetch_insn` pointer.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return 0 on success, -E2BIG if too many operations, -EINVAL for bad type.
 *
 * This function determines if a BTF type represents a string (char array or
 * char pointer) and inserts appropriate dereference operations into the
 * `fetch_insn` chain.
 */
static int check_prepare_btf_string_fetch(char *typename,
				struct fetch_insn **pcode,
				struct traceprobe_parse_context *ctx)
{
	struct btf *btf = ctx->btf;

	// Block Logic: Returns if BTF or last type is not available.
	if (!btf || !ctx->last_type)
		return 0;

	/* char [] does not need any change. */
	if (btf_type_is_char_array(btf, ctx->last_type))
		return 0;

	/* char * requires dereference the pointer. */
	// Block Logic: If it's a char pointer, inserts a dereference operation.
	if (btf_type_is_char_ptr(btf, ctx->last_type)) {
		struct fetch_insn *code = *pcode + 1;

		if (code->op == FETCH_OP_END) {
			trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
			return -E2BIG;
		}
		if (typename[0] == 'u') // Functional Utility: Checks for unsigned dereference.
			code->op = FETCH_OP_UDEREF;
		else
			code->op = FETCH_OP_DEREF;
		code->offset = 0;
		*pcode = code;
		return 0;
	}
	/* Other types are not available for string */
	trace_probe_log_err(ctx->offset, BAD_TYPE4STR); // Block Logic: Logs error for unsupported string type.
	return -EINVAL;
}

/**
 * @brief Converts a BTF type to a fetch type string.
 * @param btf Pointer to the BTF instance.
 * @param type Pointer to the BTF type.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return A string representation of the fetch type, or NULL if not supported.
 *
 * This function maps BTF types (like `INT`, `PTR`, `ENUM`) to corresponding
 * fetch type strings (e.g., "s32", "x64", "u8").
 */
static const char *fetch_type_from_btf_type(struct btf *btf,
					const struct btf_type *type,
					struct traceprobe_parse_context *ctx)
{
	u32 intdata;

	/* TODO: const char * could be converted as a string */
	switch (BTF_INFO_KIND(type->info)) {
	case BTF_KIND_ENUM:
		/* enum is "int", so convert to "s32" */
		return "s32";
	case BTF_KIND_ENUM64:
		return "s64";
	case BTF_KIND_PTR:
		/* pointer will be converted to "x??" */
		// Block Logic: Converts pointer type to architecture-dependent hexadecimal type.
		if (IS_ENABLED(CONFIG_64BIT))
			return "x64";
		else
			return "x32";
	case BTF_KIND_INT:
		intdata = btf_type_int(type);
		// Block Logic: Handles signed integer types.
		if (BTF_INT_ENCODING(intdata) & BTF_INT_SIGNED) {
			switch (BTF_INT_BITS(intdata)) {
			case 8:
				return "s8";
			case 16:
				return "s16";
			case 32:
				return "s32";
			case 64:
				return "s64";
			}
		} else {	/* unsigned */
			// Block Logic: Handles unsigned integer types.
			switch (BTF_INT_BITS(intdata)) {
			case 8:
				return "u8";
			case 16:
				return "u16";
			case 32:
				return "u32";
			case 64:
				return "u64";
			}
			/* bitfield, size is encoded in the type */
			ctx->last_bitsize = BTF_INT_BITS(intdata); // Functional Utility: Stores bitfield size.
			ctx->last_bitoffs += BTF_INT_OFFSET(intdata); // Functional Utility: Accumulates bitfield offset.
			return "u64";
		}
	}
	/* TODO: support other types */

	return NULL;
}

/**
 * @brief Queries BTF context for a function.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function retrieves BTF (BPF Type Format) information for the
 * target function (`ctx->funcname`), including its prototype and parameters.
 */
static int query_btf_context(struct traceprobe_parse_context *ctx)
{
	const struct btf_param *param;
	const struct btf_type *type;
	struct btf *btf;
	s32 nr;

	// Block Logic: Returns if BTF context already initialized.
	if (ctx->btf)
		return 0;

	// Block Logic: Returns error if function name is not set.
	if (!ctx->funcname)
		return -EINVAL;

	type = btf_find_func_proto(ctx->funcname, &btf); // Functional Utility: Finds function prototype from BTF.
	if (!type)
		return -ENOENT;

	ctx->btf = btf;
	ctx->proto = type;

	/* ctx->params is optional, since func(void) will not have params. */
	nr = 0;
	param = btf_get_func_param(type, &nr); // Functional Utility: Gets function parameters.
	if (!IS_ERR_OR_NULL(param)) {
		/* Hide the first 'data' argument of tracepoint */
		// Block Logic: Adjusts for tracepoint's dummy first argument.
		if (ctx->flags & TPARG_FL_TPOINT) {
			nr--;
			param++;
		}
	}

	// Block Logic: Stores parameters if any.
	if (nr > 0) {
		ctx->nr_params = nr;
		ctx->params = param;
	} else {
		ctx->nr_params = 0;
		ctx->params = NULL;
	}

	return 0;
}

/**
 * @brief Clears the BTF context.
 * @param ctx Pointer to `traceprobe_parse_context`.
 *
 * This function releases the BTF reference and resets BTF-related fields.
 */
static void clear_btf_context(struct traceprobe_parse_context *ctx)
{
	if (ctx->btf) {
		btf_put(ctx->btf); // Functional Utility: Releases BTF reference.
		ctx->btf = NULL;
		ctx->proto = NULL;
		ctx->params = NULL;
		ctx->nr_params = 0;
	}
}

/* Return 1 if the field separater is arrow operator ('->') */
/**
 * @brief Splits a variable name into the current field and the next field.
 * @param varname The variable name string.
 * @param next_field Output parameter for the pointer to the next field.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return 1 if arrow operator (`->`) was used, 0 if dot operator (`.`) was used, -EINVAL on error.
 *
 * This function handles both dot (`.`) and arrow (`->`) operators for
 * accessing structure members, modifying `varname` in place.
 */
static int split_next_field(char *varname, char **next_field,
			    struct traceprobe_parse_context *ctx)
{
	char *field;
	int ret = 0;

	field = strpbrk(varname, ".-"); // Functional Utility: Searches for '.' or '-'.
	// Block Logic: If a separator is found.
	if (field) {
		if (field[0] == '-' && field[1] == '>') { // Block Logic: Handles arrow operator.
			field[0] = '\0';
			field += 2;
			ret = 1; // Functional Utility: Indicates arrow operator.
		} else if (field[0] == '.') { // Block Logic: Handles dot operator.
			field[0] = '\0';
			field += 1;
		} else { // Block Logic: Invalid separator.
			trace_probe_log_err(ctx->offset + field - varname, BAD_HYPHEN);
			return -EINVAL;
		}
		*next_field = field; // Functional Utility: Updates pointer to the next field.
	}

	return ret;
}

/*
 * Parse the field of data structure. The @type must be a pointer type
 * pointing the target data structure type.
 */
/**
 * @brief Parses a BTF field from a data structure.
 * @param fieldname The name of the field to parse.
 * @param type Pointer to the BTF type of the current structure.
 * @param pcode Input/Output: Pointer to the current `fetch_insn` pointer.
 * @param end Pointer to the end of the `fetch_insn` array.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function recursively parses structure members, handling dot and arrow
 * operators, and inserts `FETCH_OP_DEREF` instructions.
 */
static int parse_btf_field(char *fieldname, const struct btf_type *type,
			   struct fetch_insn **pcode, struct fetch_insn *end,
			   struct traceprobe_parse_context *ctx)
{
	struct fetch_insn *code = *pcode;
	const struct btf_member *field;
	u32 bitoffs, anon_offs;
	char *next;
	int is_ptr;
	s32 tid;

	do {
		/* Outer loop for solving arrow operator ('->') */
		// Block Logic: If not a pointer type, returns error.
		if (BTF_INFO_KIND(type->info) != BTF_KIND_PTR) {
			trace_probe_log_err(ctx->offset, NO_PTR_STRCT);
			return -EINVAL;
		}
		/* Convert a struct pointer type to a struct type */
		type = btf_type_skip_modifiers(ctx->btf, type->type, &tid); // Functional Utility: Skips modifiers to get real type.
		if (!type) {
			trace_probe_log_err(ctx->offset, BAD_BTF_TID);
			return -EINVAL;
		}

		bitoffs = 0;
		do {
			/* Inner loop for solving dot operator ('.') */
			next = NULL;
			is_ptr = split_next_field(fieldname, &next, ctx); // Functional Utility: Splits fieldname.
			if (is_ptr < 0)
				return is_ptr;

			anon_offs = 0;
			field = btf_find_struct_member(ctx->btf, type, fieldname,
						       &anon_offs); // Functional Utility: Finds struct member.
			if (IS_ERR(field)) {
				trace_probe_log_err(ctx->offset, BAD_BTF_TID);
				return PTR_ERR(field);
			}
			if (!field) {
				trace_probe_log_err(ctx->offset, NO_BTF_FIELD);
				return -ENOENT;
			}
			/* Add anonymous structure/union offset */
			bitoffs += anon_offs; // Functional Utility: Accumulates anonymous struct/union offset.

			/* Accumulate the bit-offsets of the dot-connected fields */
			// Block Logic: Accumulates bit offsets for field access.
			if (btf_type_kflag(type)) {
				bitoffs += BTF_MEMBER_BIT_OFFSET(field->offset);
				ctx->last_bitsize = BTF_MEMBER_BITFIELD_SIZE(field->offset);
			} else {
				bitoffs += field->offset;
				ctx->last_bitsize = 0;
			}

			type = btf_type_skip_modifiers(ctx->btf, field->type, &tid); // Functional Utility: Updates type to the field's type.
			if (!type) {
				trace_probe_log_err(ctx->offset, BAD_BTF_TID);
				return -EINVAL;
			}

			ctx->offset += next - fieldname; // Functional Utility: Updates offset for error reporting.
			fieldname = next; // Functional Utility: Moves to the next field.
		} while (!is_ptr && fieldname); // Block Logic: Continues while dot operator and more fields exist.

		if (++code == end) { // Block Logic: Checks for end of instruction array.
			trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
			return -EINVAL;
		}
		code->op = FETCH_OP_DEREF;	/* TODO: user deref support */
		code->offset = bitoffs / 8; // Functional Utility: Sets dereference offset.
		*pcode = code;

		ctx->last_bitoffs = bitoffs % 8; // Functional Utility: Stores last bit offset.
		ctx->last_type = type; // Functional Utility: Stores last BTF type.
	} while (fieldname); // Block Logic: Continues while arrow operator and more fields exist.

	return 0;
}

static int __store_entry_arg(struct trace_probe *tp, int argnum); // Forward declaration.

/**
 * @brief Parses a BTF-based argument.
 * @param varname The variable name string.
 * @param pcode Input/Output: Pointer to the current `fetch_insn` pointer.
 * @param end Pointer to the end of the `fetch_insn` array.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function parses a variable name, resolving it against BTF function
 * parameters, and inserts appropriate fetch instructions.
 */
static int parse_btf_arg(char *varname,
			 struct fetch_insn **pcode, struct fetch_insn *end,
			 struct traceprobe_parse_context *ctx)
{
	struct fetch_insn *code = *pcode;
	const struct btf_param *params;
	const struct btf_type *type;
	char *field = NULL;
	int i, is_ptr, ret;
	u32 tid;

	// Block Logic: Ensures function name is set in context.
	if (WARN_ON_ONCE(!ctx->funcname))
		return -EINVAL;

	is_ptr = split_next_field(varname, &field, ctx); // Functional Utility: Splits varname for field access.
	if (is_ptr < 0)
		return is_ptr;
	// Block Logic: Dot-connected fields on arguments are not supported directly.
	if (!is_ptr && field) {
		/* dot-connected field on an argument is not supported. */
		trace_probe_log_err(ctx->offset + field - varname,
				    NOSUP_DAT_ARG);
		return -EOPNOTSUPP;
	}

	// Block Logic: Handles `$retval` for return probes.
	if (ctx->flags & TPARG_FL_RETURN && !strcmp(varname, "$retval")) {
		code->op = FETCH_OP_RETVAL; // Functional Utility: Sets operation to fetch return value.
		/* Check whether the function return type is not void */
		if (query_btf_context(ctx) == 0) { // Functional Utility: Queries BTF context.
			if (ctx->proto->type == 0) { // Block Logic: Checks for void return type.
				trace_probe_log_err(ctx->offset, NO_RETVAL);
				return -ENOENT;
			}
			tid = ctx->proto->type; // Functional Utility: Gets return type ID.
			goto found;
		}
		if (field) { // Block Logic: Field access on `$retval` not supported without BTF.
			trace_probe_log_err(ctx->offset + field - varname,
					    NO_BTF_ENTRY);
			return -ENOENT;
		}
		return 0;
	}

	// Block Logic: Queries BTF context if not already set.
	if (!ctx->btf) {
		ret = query_btf_context(ctx);
		if (ret < 0 || ctx->nr_params == 0) {
			trace_probe_log_err(ctx->offset, NO_BTF_ENTRY);
			return -ENOENT;
		}
	}
	params = ctx->params;

	// Block Logic: Iterates through function parameters to find a match.
	for (i = 0; i < ctx->nr_params; i++) {
		const char *name = btf_name_by_offset(ctx->btf, params[i].name_off);

		if (name && !strcmp(name, varname)) {
			// Block Logic: If at function entry, fetches argument.
			if (tparg_is_function_entry(ctx->flags)) {
				code->op = FETCH_OP_ARG;
				// Block Logic: Adjusts parameter index for tracepoints.
				if (ctx->flags & TPARG_FL_TPOINT)
					code->param++;
				else
					code->param = i;
			} else if (tparg_is_function_return(ctx->flags)) { // Block Logic: If at function return, stores entry argument.
				code->op = FETCH_OP_EDATA;
				ret = __store_entry_arg(ctx->tp, i); // Functional Utility: Stores entry argument.
				if (ret < 0) {
					/* internal error */
					return ret;
				}
				code->offset = ret;
			}
			tid = params[i].type; // Functional Utility: Gets parameter type ID.
			goto found;
		}
	}
	trace_probe_log_err(ctx->offset, NO_BTFARG); // Block Logic: Logs error if argument not found.
	return -ENOENT;

found:
	type = btf_type_skip_modifiers(ctx->btf, tid, &tid); // Functional Utility: Skips modifiers to get real type.
	if (!type) {
		trace_probe_log_err(ctx->offset, BAD_BTF_TID);
		return -EINVAL;
	}
	/* Initialize the last type information */
	ctx->last_type = type;
	ctx->last_bitoffs = 0;
	ctx->last_bitsize = 0;
	// Block Logic: If a field is present, parses it.
	if (field) {
		ctx->offset += field - varname;
		return parse_btf_field(field, type, pcode, end, ctx);
	}
	return 0;
}

/**
 * @brief Finds the fetch type corresponding to the last parsed BTF type.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return Pointer to the `fetch_type` structure, or NULL if not found.
 *
 * This function uses `fetch_type_from_btf_type` to convert the last BTF type
 * into a fetch type string, then uses `find_fetch_type` to get the
 * corresponding fetch type structure.
 */
static const struct fetch_type *find_fetch_type_from_btf_type(
					struct traceprobe_parse_context *ctx)
{
	struct btf *btf = ctx->btf;
	const char *typestr = NULL;

	// Block Logic: If BTF and last type are available, converts BTF type to string.
	if (btf && ctx->last_type)
		typestr = fetch_type_from_btf_type(btf, ctx->last_type, ctx);

	return find_fetch_type(typestr, ctx->flags); // Functional Utility: Finds fetch type from string.
}

/**
 * @brief Parses a BTF bitfield and inserts appropriate fetch instructions.
 * @param pcode Input/Output: Pointer to the current `fetch_insn` pointer.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return 0 on success, -EINVAL on parsing error.
 *
 * This function inserts a `FETCH_OP_MOD_BF` instruction if the last BTF type
 * represents a bitfield.
 */
static int parse_btf_bitfield(struct fetch_insn **pcode,
			      struct traceprobe_parse_context *ctx)
{
	struct fetch_insn *code = *pcode;

	// Block Logic: Returns if not a bitfield or already aligned.
	if ((ctx->last_bitsize % 8 == 0) && ctx->last_bitoffs == 0)
		return 0;

	code++;
	// Block Logic: Checks for overflow of instruction array.
	if (code->op != FETCH_OP_NOP) {
		trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
		return -EINVAL;
	}
	*pcode = code;

	code->op = FETCH_OP_MOD_BF; // Functional Utility: Sets operation to bitfield modification.
	code->lshift = 64 - (ctx->last_bitsize + ctx->last_bitoffs); // Functional Utility: Calculates left shift.
	code->rshift = 64 - ctx->last_bitsize; // Functional Utility: Calculates right shift.
	code->basesize = 64 / 8; // Functional Utility: Sets base size to 8 bytes.
	// Block Logic: Checks for invalid bitfield range.
	return (BYTES_TO_BITS(t->size) < (bw + bo)) ? -EINVAL : 0;
}

#else
// Block Logic: Stubs for CONFIG_PROBE_EVENTS_BTF_ARGS disabled builds.
static void clear_btf_context(struct traceprobe_parse_context *ctx)
{
	ctx->btf = NULL;
}

static int query_btf_context(struct traceprobe_parse_context *ctx)
{
	return -EOPNOTSUPP;
}

static int parse_btf_arg(char *varname,
			 struct fetch_insn **pcode, struct fetch_insn *end,
			 struct traceprobe_parse_context *ctx)
{
	trace_probe_log_err(ctx->offset, NOSUP_BTFARG);
	return -EOPNOTSUPP;
}

static int parse_btf_bitfield(struct fetch_insn **pcode,
			      struct traceprobe_parse_context *ctx)
{
	trace_probe_log_err(ctx->offset, NOSUP_BTFARG);
	return -EOPNOTSUPP;
}

#define find_fetch_type_from_btf_type(ctx)		\
	find_fetch_type(NULL, ctx->flags)

static int check_prepare_btf_string_fetch(char *typename,
				struct fetch_insn **pcode,
				struct traceprobe_parse_context *ctx)
{
	return 0;
}

#endif

#ifdef CONFIG_HAVE_FUNCTION_ARG_ACCESS_API

/*
 * Add the entry code to store the 'argnum'th parameter and return the offset
 * in the entry data buffer where the data will be stored.
 */
/**
 * @brief Stores an entry argument for kretprobes.
 * @param tp Pointer to the `trace_probe`.
 * @param argnum The argument number to store.
 * @return The offset in the entry data buffer where the data is stored,
 *         or a negative errno on failure.
 *
 * This function builds an array of `fetch_insn` structures to capture
 * function arguments at entry and store them in the kretprobe's `entry_data`
 * buffer.
 */
static int __store_entry_arg(struct trace_probe *tp, int argnum)
{
	struct probe_entry_arg *earg = tp->entry_arg;
	bool match = false;
	int i, offset;

	// Block Logic: If `entry_arg` is not yet allocated, initializes it.
	if (!earg) {
		earg = kzalloc(sizeof(*tp->entry_arg), GFP_KERNEL);
		if (!earg)
			return -ENOMEM;
		earg->size = 2 * tp->nr_args + 1; // Functional Utility: Calculates size for instruction array.
		earg->code = kcalloc(earg->size, sizeof(struct fetch_insn),
				     GFP_KERNEL);
		if (!earg->code) {
			kfree(earg);
			return -ENOMEM;
		}
		/* Fill the code buffer with 'end' to simplify it */
		// Functional Utility: Initializes code buffer with `FETCH_OP_END`.
		for (i = 0; i < earg->size; i++)
			earg->code[i].op = FETCH_OP_END;
		tp->entry_arg = earg;
	}

	/*
	 * The entry code array is repeating the pair of
	 * [FETCH_OP_ARG(argnum)][FETCH_OP_ST_EDATA(offset of entry data buffer)]
	 * and the rest of entries are filled with [FETCH_OP_END].
	 *
	 * To reduce the redundant function parameter fetching, we scan the entry
	 * code array to find the FETCH_OP_ARG which already fetches the 'argnum'
	 * parameter. If it doesn't match, update 'offset' to find the last
	 * offset.
	 * If we find the FETCH_OP_END without matching FETCH_OP_ARG entry, we
	 * will save the entry with FETCH_OP_ARG and FETCH_OP_ST_EDATA, and
	 * return data offset so that caller can find the data offset in the entry
	 * data buffer.
	 */
	offset = 0;
	// Block Logic: Scans existing entry code to avoid redundant fetching.
	for (i = 0; i < earg->size - 1; i++) {
		switch (earg->code[i].op) {
		case FETCH_OP_END: // Block Logic: If `FETCH_OP_END` is found, inserts new instructions.
			earg->code[i].op = FETCH_OP_ARG;
			earg->code[i].param = argnum;
			earg->code[i + 1].op = FETCH_OP_ST_EDATA;
			earg->code[i + 1].offset = offset;
			return offset;
		case FETCH_OP_ARG:
			match = (earg->code[i].param == argnum); // Functional Utility: Checks for matching argument number.
			break;
		case FETCH_OP_ST_EDATA:
			offset = earg->code[i].offset; // Functional Utility: Updates offset.
			if (match)
				return offset; // Functional Utility: If argument already fetched, returns its offset.
			offset += sizeof(unsigned long); // Functional Utility: Increments offset for next data.
			break;
		default:
			break;
		}
	}
	return -ENOSPC; // Block Logic: Returns error if no space available.
}

/**
 * @brief Calculates the size of the entry data buffer required for a trace probe.
 * @param tp Pointer to the `trace_probe`.
 * @return The size of the entry data buffer in bytes.
 *
 * This function iterates through the `entry_arg` code array to determine
 * the maximum offset used for storing entry data.
 */
int traceprobe_get_entry_data_size(struct trace_probe *tp)
{
	struct probe_entry_arg *earg = tp->entry_arg;
	int i, size = 0;

	// Block Logic: Returns 0 if no entry arguments.
	if (!earg)
		return 0;

	/*
	 * earg->code[] array has an operation sequence which is run in
	 * the entry handler.
	 * The sequence stopped by FETCH_OP_END and each data stored in
	 * the entry data buffer by FETCH_OP_ST_EDATA. The FETCH_OP_ST_EDATA
	 * stores the data at the data buffer + its offset, and all data are
	 * "unsigned long" size. The offset must be increased when a data is
	 * stored. Thus we need to find the last FETCH_OP_ST_EDATA in the
	 * code array.
	 */
	// Block Logic: Iterates through the entry code array.
	for (i = 0; i < earg->size; i++) {
		switch (earg->code[i].op) {
		case FETCH_OP_END: // Block Logic: Ends loop if `FETCH_OP_END` is found.
			goto out;
		case FETCH_OP_ST_EDATA:
			size = earg->code[i].offset + sizeof(unsigned long); // Functional Utility: Updates max size.
			break;
		default:
			break;
		}
	}
out:
	return size;
}

/**
 * @brief Stores trace entry data into the `entry_data` buffer.
 * @param edata Pointer to the `entry_data` buffer.
 * @param tp Pointer to the `trace_probe`.
 * @param regs Pointer to `pt_regs`.
 *
 * This function executes the `entry_arg` code array to fetch and store
 * function arguments into the `entry_data` buffer.
 */
void store_trace_entry_data(void *edata, struct trace_probe *tp, struct pt_regs *regs)
{
	struct probe_entry_arg *earg = tp->entry_arg;
	unsigned long val = 0;
	int i;

	// Block Logic: Returns if no entry arguments.
	if (!earg)
		return;

	// Block Logic: Executes fetch instructions to store data.
	for (i = 0; i < earg->size; i++) {
		struct fetch_insn *code = &earg->code[i];

		switch (code->op) {
		case FETCH_OP_ARG:
			val = regs_get_kernel_argument(regs, code->param); // Functional Utility: Gets kernel argument.
			break;
		case FETCH_OP_ST_EDATA:
			*(unsigned long *)((unsigned long)edata + code->offset) = val; // Functional Utility: Stores data at offset.
			break;
		case FETCH_OP_END:
			goto end; // Functional Utility: Ends loop.
		default:
			break;
		}
	}
end:
	return;
}
NOKPROBE_SYMBOL(store_trace_entry_data)
#endif

#define PARAM_MAX_STACK (THREAD_SIZE / sizeof(unsigned long))

/* Parse $vars. @orig_arg points '$', which syncs to @ctx->offset */
/**
 * @brief Parses probe variables (`$var`).
 * @param orig_arg The original argument string starting with `$`.
 * @param t Pointer to the `fetch_type` (can be NULL).
 * @param pcode Input/Output: Pointer to the current `fetch_insn` pointer.
 * @param end Pointer to the end of the `fetch_insn` array.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return 0 on success, -EINVAL on parsing error, -ENOENT if symbol not found.
 *
 * This function parses special variables like `$retval`, `$stack`, `$argN`,
 * and `$comm`, inserting appropriate fetch instructions. It handles BTF
 * integration for type-aware parsing.
 */
static int parse_probe_vars(char *orig_arg, const struct fetch_type *t,
			    struct fetch_insn **pcode,
			    struct fetch_insn *end,
			    struct traceprobe_parse_context *ctx)
{
	struct fetch_insn *code = *pcode;
	int err = TP_ERR_BAD_VAR;
	char *arg = orig_arg + 1; // Functional Utility: Skips '$'.
	unsigned long param;
	int ret = 0;
	int len;

	// Block Logic: Handles trace event arguments (`TPARG_FL_TEVENT`).
	if (ctx->flags & TPARG_FL_TEVENT) {
		if (code->data)
			return -EFAULT;
		ret = parse_trace_event_arg(arg, code, ctx);
		if (!ret)
			return 0;
		if (strcmp(arg, "comm") == 0 || strcmp(arg, "COMM") == 0) {
			code->op = FETCH_OP_COMM;
			return 0;
		}
		/* backward compatibility */
		ctx->offset = 0;
		goto inval;
	}

	// Block Logic: Handles `$retval`.
	if (str_has_prefix(arg, "retval")) {
		if (!(ctx->flags & TPARG_FL_RETURN)) { // Block Logic: `$retval` only valid for return probes.
			err = TP_ERR_RETVAL_ON_PROBE;
			goto inval;
		}
		// Block Logic: If BTF is available, uses BTF parsing for `$retval`.
		if (!(ctx->flags & TPARG_FL_KERNEL) ||
		    !IS_ENABLED(CONFIG_PROBE_EVENTS_BTF_ARGS)) {
			code->op = FETCH_OP_RETVAL; // Functional Utility: Sets operation to fetch return value.
			return 0;
		}
		return parse_btf_arg(orig_arg, pcode, end, ctx);
	}

	// Block Logic: Handles `$stack` and `$stackN`.
	len = str_has_prefix(arg, "stack");
	if (len) {
		// Block Logic: If just `$stack`, fetches stack pointer.
		if (arg[len] == '\0') {
			code->op = FETCH_OP_STACKP;
			return 0;
		}

		// Block Logic: If `$stackN`, fetches Nth stack entry.
		if (isdigit(arg[len])) {
			ret = kstrtoul(arg + len, 10, &param);
			if (ret)
				goto inval;

			if ((ctx->flags & TPARG_FL_KERNEL) &&
			    param > PARAM_MAX_STACK) { // Block Logic: Checks stack bounds for kernel.
				err = TP_ERR_BAD_STACK_NUM;
				goto inval;
			}
			code->op = FETCH_OP_STACK; // Functional Utility: Sets operation to fetch stack entry.
			code->param = (unsigned int)param; // Functional Utility: Stores stack index.
			return 0;
		}
		goto inval;
	}

	// Block Logic: Handles `$comm`.
	if (strcmp(arg, "comm") == 0 || strcmp(arg, "COMM") == 0) {
		code->op = FETCH_OP_COMM; // Functional Utility: Sets operation to fetch comm.
		return 0;
	}

#ifdef CONFIG_HAVE_FUNCTION_ARG_ACCESS_API
	// Block Logic: Handles `$argN`.
	len = str_has_prefix(arg, "arg");
	if (len) {
		ret = kstrtoul(arg + len, 10, &param);
		if (ret)
			goto inval;

		if (!param || param > PARAM_MAX_STACK) { // Block Logic: Checks arg bounds.
			err = TP_ERR_BAD_ARG_NUM;
			goto inval;
		}
		param--; /* argN starts from 1, but internal arg[N] starts from 0 */

		// Block Logic: Handles function entry arguments.
		if (tparg_is_function_entry(ctx->flags)) {
			code->op = FETCH_OP_ARG;
			// Block Logic: Adjusts parameter index for tracepoints.
			if (ctx->flags & TPARG_FL_TPOINT)
				code->param++;
			else
				code->param = (unsigned int)param;
		} else if (tparg_is_function_return(ctx->flags)) { // Block Logic: Handles function return arguments.
			/* function entry argument access from return probe */
			ret = __store_entry_arg(ctx->tp, param); // Functional Utility: Stores entry argument.
			if (ret < 0)	/* This error should be an internal error */
				return ret;

			code->op = FETCH_OP_EDATA; // Functional Utility: Sets operation to fetch entry data.
			code->offset = ret; // Functional Utility: Stores offset to entry data.
		} else {
			err = TP_ERR_NOFENTRY_ARGS;
			goto inval;
		}
		return 0;
	}
#endif

inval:
	__trace_probe_log_err(ctx->offset, err); // Functional Utility: Logs detailed error.
	return -EINVAL;
}

/**
 * @brief Converts a string to an immediate unsigned long value.
 * @param str The string to convert.
 * @param imm Output parameter for the immediate value.
 * @return 0 on success, -EINVAL on parsing error.
 *
 * This function handles decimal and hexadecimal immediate values,
 * including those with `+` or `-` prefixes.
 */
static int str_to_immediate(char *str, unsigned long *imm)
{
	if (isdigit(str[0]))
		return kstrtoul(str, 0, imm);
	else if (str[0] == '-')
		return kstrtol(str, 0, (long *)imm); // Functional Utility: Handles negative values.
	else if (str[0] == '+')
		return kstrtol(str + 1, 0, (long *)imm); // Functional Utility: Handles positive values with explicit '+'.
	return -EINVAL;
}

/**
 * @brief Parses an immediate string literal.
 * @param str The input string (e.g., `"my_string"`).
 * @param pbuf Output parameter for the duplicated string buffer.
 * @param offs Offset in the original command for error reporting.
 * @return 0 on success, -EINVAL if string not closed, -ENOMEM on memory allocation failure.
 *
 * This function duplicates an immediate string literal (enclosed in double quotes)
 * and stores it.
 */
static int __parse_imm_string(char *str, char **pbuf, int offs)
{
	size_t len = strlen(str);

	// Block Logic: Checks for closing double quote.
	if (str[len - 1] != '"') {
		trace_probe_log_err(offs + len, IMMSTR_NO_CLOSE);
		return -EINVAL;
	}
	*pbuf = kstrndup(str, len - 1, GFP_KERNEL); // Functional Utility: Duplicates string, excluding quotes.
	if (!*pbuf)
		return -ENOMEM;
	return 0;
}

/* Recursive argument parser */
/**
 * @brief Recursively parses a probe argument and generates fetch instructions.
 * @param arg The argument string.
 * @param type Pointer to the expected `fetch_type` (can be NULL).
 * @param pcode Input/Output: Pointer to the current `fetch_insn` pointer.
 * @param end Pointer to the end of the `fetch_insn` array.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return 0 on success, or a negative errno on parsing error.
 *
 * This is the main recursive parser for probe arguments, handling different
 * argument types like variables, registers, memory addresses, and immediate values.
 */
static int
parse_probe_arg(char *arg, const struct fetch_type *type,
		struct fetch_insn **pcode, struct fetch_insn *end,
		struct traceprobe_parse_context *ctx)
{
	struct fetch_insn *code = *pcode;
	unsigned long param;
	int deref = FETCH_OP_DEREF; // Functional Utility: Default dereference operation.
	long offset = 0;
	char *tmp;
	int ret = 0;

	// Block Logic: Dispatches based on the first character of the argument.
	switch (arg[0]) {
	case '$': // Block Logic: Handles variables starting with '$'.
		ret = parse_probe_vars(arg, type, pcode, end, ctx);
		break;

	case '%':	/* named register */
		// Block Logic: Handles named registers starting with '%'.
		if (ctx->flags & (TPARG_FL_TEVENT | TPARG_FL_FPROBE)) {
			/* eprobe and fprobe do not handle registers */
			trace_probe_log_err(ctx->offset, BAD_VAR);
			break;
		}
		ret = regs_query_register_offset(arg + 1); // Functional Utility: Queries register offset.
		if (ret >= 0) {
			code->op = FETCH_OP_REG; // Functional Utility: Sets operation to fetch register.
			code->param = (unsigned int)ret;
			ret = 0;
		} else
			trace_probe_log_err(ctx->offset, BAD_REG_NAME);
		break;

	case '@':	/* memory, file-offset or symbol */
		// Block Logic: Handles memory addresses, file offsets, or symbols starting with '@'.
		if (isdigit(arg[1])) { // Block Logic: If it's a numeric address.
			ret = kstrtoul(arg + 1, 0, &param);
			if (ret) {
				trace_probe_log_err(ctx->offset, BAD_MEM_ADDR);
				break;
			}
			/* load address */
			code->op = FETCH_OP_IMM; // Functional Utility: Sets operation to load immediate value.
			code->immediate = param;
		} else if (arg[1] == '+') { // Block Logic: If it's a file offset (e.g., "@+100").
			/* kprobes don't support file offsets */
			if (ctx->flags & TPARG_FL_KERNEL) { // Block Logic: Kernel probes do not support file offsets.
				trace_probe_log_err(ctx->offset, FILE_ON_KPROBE);
				return -EINVAL;
			}
			ret = kstrtol(arg + 2, 0, &offset);
			if (ret) {
				trace_probe_log_err(ctx->offset, BAD_FILE_OFFS);
				break;
			}

			code->op = FETCH_OP_FOFFS; // Functional Utility: Sets operation to fetch file offset.
			code->immediate = (unsigned long)offset;  // imm64?
		} else { // Block Logic: If it's a symbol (e.g., "@my_symbol").
			/* uprobes don't support symbols */
			if (!(ctx->flags & TPARG_FL_KERNEL)) { // Block Logic: Uprobes do not support symbols.
				trace_probe_log_err(ctx->offset, SYM_ON_UPROBE);
				return -EINVAL;
			}
			/* Preserve symbol for updating */
			code->op = FETCH_NOP_SYMBOL; // Functional Utility: Sets operation to no-op for symbol.
			code->data = kstrdup(arg + 1, GFP_KERNEL); // Functional Utility: Duplicates symbol string.
			if (!code->data)
				return -ENOMEM;
			if (++code == end) { // Block Logic: Checks for end of instruction array.
				trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
				return -EINVAL;
			}
			code->op = FETCH_OP_IMM; // Functional Utility: Sets operation to load immediate value.
			code->immediate = 0;
		}
		/* These are fetching from memory */
		if (++code == end) { // Block Logic: Checks for end of instruction array.
			trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
			return -EINVAL;
		}
		*pcode = code;
		code->op = FETCH_OP_DEREF; // Functional Utility: Sets operation to dereference.
		code->offset = offset;
		break;

	case '+':	/* deref memory */
	case '-':
		// Block Logic: Handles memory dereferencing (e.g., "+0x10(arg)").
		if (arg[1] == 'u') { // Functional Utility: Checks for unsigned dereference.
			deref = FETCH_OP_UDEREF;
			arg[1] = arg[0]; // Functional Utility: Corrects argument pointer.
			arg++;
		}
		if (arg[0] == '+')
			arg++;	/* Skip '+', because kstrtol() rejects it. */
		tmp = strchr(arg, '('); // Functional Utility: Finds '(' for argument start.
		if (!tmp) {
			trace_probe_log_err(ctx->offset, DEREF_NEED_BRACE);
			return -EINVAL;
		}
		*tmp = '\0'; // Functional Utility: Null-terminates offset string.
		ret = kstrtol(arg, 0, &offset); // Functional Utility: Parses offset.
		if (ret) {
			trace_probe_log_err(ctx->offset, BAD_DEREF_OFFS);
			break;
		}
		ctx->offset += (tmp + 1 - arg) + (arg[0] != '-' ? 1 : 0); // Functional Utility: Updates offset for error reporting.
		arg = tmp + 1; // Functional Utility: Moves to argument part.
		tmp = strrchr(arg, ')'); // Functional Utility: Finds ')' for argument end.
		if (!tmp) {
			trace_probe_log_err(ctx->offset + strlen(arg),
					    DEREF_OPEN_BRACE);
			return -EINVAL;
		} else {
			const struct fetch_type *t2 = find_fetch_type(NULL, ctx->flags); // Functional Utility: Finds default fetch type.
			int cur_offs = ctx->offset;

			*tmp = '\0'; // Functional Utility: Null-terminates argument string.
			ret = parse_probe_arg(arg, t2, &code, end, ctx); // Functional Utility: Recursively parses argument.
			if (ret)
				break;
			ctx->offset = cur_offs;
			// Block Logic: Checks for invalid dereference combinations.
			if (code->op == FETCH_OP_COMM ||
			    code->op == FETCH_OP_DATA) {
				trace_probe_log_err(ctx->offset, COMM_CANT_DEREF);
				return -EINVAL;
			}
			if (++code == end) { // Block Logic: Checks for end of instruction array.
				trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
				return -EINVAL;
			}
			*pcode = code;

			code->op = deref; // Functional Utility: Sets dereference operation.
			code->offset = offset; // Functional Utility: Sets dereference offset.
			/* Reset the last type if used */
			ctx->last_type = NULL; // Functional Utility: Resets last BTF type.
		}
		break;
	case '\\':	/* Immediate value */
		// Block Logic: Handles immediate values and string literals.
		if (arg[1] == '"') {	/* Immediate string */
			ret = __parse_imm_string(arg + 2, &tmp, ctx->offset + 2); // Functional Utility: Parses immediate string.
			if (ret)
				break;
			code->op = FETCH_OP_DATA; // Functional Utility: Sets operation to load data.
			code->data = tmp; // Functional Utility: Stores duplicated string.
		} else {
			ret = str_to_immediate(arg + 1, &code->immediate); // Functional Utility: Parses immediate numeric value.
			if (ret)
				trace_probe_log_err(ctx->offset + 1, BAD_IMM);
			else
				code->op = FETCH_OP_IMM; // Functional Utility: Sets operation to load immediate value.
		}
		break;
	default:
		// Block Logic: Handles BTF variables.
		if (isalpha(arg[0]) || arg[0] == '_') {	/* BTF variable */
			if (!tparg_is_function_entry(ctx->flags) &&
			    !tparg_is_function_return(ctx->flags)) { // Block Logic: BTF variables only valid for function entry/return.
				trace_probe_log_err(ctx->offset, NOSUP_BTFARG);
				return -EINVAL;
			}
			ret = parse_btf_arg(arg, pcode, end, ctx); // Functional Utility: Parses BTF argument.
			break;
		}
	}
	// Block Logic: If no fetch method found.
	if (!ret && code->op == FETCH_OP_NOP) {
		/* Parsed, but do not find fetch method */
		trace_probe_log_err(ctx->offset, BAD_FETCH_ARG);
		ret = -EINVAL;
	}
	return ret;
}

/* Bitfield type needs to be parsed into a fetch function */
/**
 * @brief Parses a bitfield type and inserts appropriate fetch instructions.
 * @param bf The bitfield string (e.g., "b<bitsize>@<bitoffset>").
 * @param t Pointer to the base `fetch_type`.
 * @param pcode Input/Output: Pointer to the current `fetch_insn` pointer.
 * @return 0 on success, -EINVAL on parsing error.
 *
 * This function inserts a `FETCH_OP_MOD_BF` instruction to handle bitfield
 * extraction.
 */
static int __parse_bitfield_probe_arg(const char *bf,
				      const struct fetch_type *t,
				      struct fetch_insn **pcode)
{
	struct fetch_insn *code = *pcode;
	unsigned long bw, bo;
	char *tail;

	// Block Logic: Returns if not a bitfield.
	if (*bf != 'b')
		return 0;

	bw = simple_strtoul(bf + 1, &tail, 0);	/* Use simple one */ // Functional Utility: Parses bit width.

	if (bw == 0 || *tail != '@')
		return -EINVAL;

	bf = tail + 1;
	bo = simple_strtoul(bf, &tail, 0); // Functional Utility: Parses bit offset.

	if (tail == bf || *tail != '/')
		return -EINVAL;
	code++;
	if (code->op != FETCH_OP_NOP) // Block Logic: Checks for overflow of instruction array.
		return -EINVAL;
	*pcode = code;

	code->op = FETCH_OP_MOD_BF; // Functional Utility: Sets operation to bitfield modification.
	code->lshift = BYTES_TO_BITS(t->size) - (bw + bo); // Functional Utility: Calculates left shift.
	code->rshift = BYTES_TO_BITS(t->size) - bw; // Functional Utility: Calculates right shift.
	code->basesize = t->size; // Functional Utility: Sets base size.

	return (BYTES_TO_BITS(t->size) < (bw + bo)) ? -EINVAL : 0; // Block Logic: Checks for invalid bitfield range.
}

/* Split type part from @arg and return it. */
/**
 * @brief Parses the type information from an argument string.
 * @param arg The argument string.
 * @param parg Pointer to `probe_arg`.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return Pointer to the type string, or an `ERR_PTR` on parsing error.
 *
 * This function extracts the type and array count (e.g., `s32[4]`) from
 * an argument string and finds the corresponding `fetch_type`.
 */
static char *parse_probe_arg_type(char *arg, struct probe_arg *parg,
				  struct traceprobe_parse_context *ctx)
{
	char *t = NULL, *t2, *t3;
	int offs;

	t = strchr(arg, ':'); // Functional Utility: Finds ':' separator for type.
	// Block Logic: If type separator found.
	if (t) {
		*t++ = '\0'; // Functional Utility: Null-terminates argument part.
		t2 = strchr(t, '['); // Functional Utility: Finds '[' for array.
		// Block Logic: If array syntax found.
		if (t2) {
			*t2++ = '\0'; // Functional Utility: Null-terminates type part.
			t3 = strchr(t2, ']'); // Functional Utility: Finds ']' for array end.
			// Block Logic: Validates array syntax.
			if (!t3) {
				offs = t2 + strlen(t2) - arg;

				trace_probe_log_err(ctx->offset + offs,
						    ARRAY_NO_CLOSE);
				return ERR_PTR(-EINVAL);
			} else if (t3[1] != '\0') {
				trace_probe_log_err(ctx->offset + t3 + 1 - arg,
						    BAD_ARRAY_SUFFIX);
				return ERR_PTR(-EINVAL);
			}
			*t3 = '\0'; // Functional Utility: Null-terminates array count.
			if (kstrtouint(t2, 0, &parg->count) || !parg->count) { // Functional Utility: Parses array count.
				trace_probe_log_err(ctx->offset + t2 - arg,
						    BAD_ARRAY_NUM);
				return ERR_PTR(-EINVAL);
			}
			if (parg->count > MAX_ARRAY_LEN) { // Block Logic: Checks array length against maximum.
				trace_probe_log_err(ctx->offset + t2 - arg,
						    ARRAY_TOO_BIG);
				return ERR_PTR(-EINVAL);
			}
		}
	}
	offs = t ? t - arg : 0; // Functional Utility: Calculates offset for error reporting.

	/*
	 * Since $comm and immediate string can not be dereferenced,
	 * we can find those by strcmp. But ignore for eprobes.
	 */
	// Block Logic: Handles special cases like `$comm` and immediate strings.
	if (!(ctx->flags & TPARG_FL_TEVENT) &&
	    (strcmp(arg, "$comm") == 0 || strcmp(arg, "$COMM") == 0 ||
	     strncmp(arg, "\\\"", 2) == 0)) {
		/* The type of $comm must be "string", and not an array type. */
		if (parg->count || (t && strcmp(t, "string"))) { // Block Logic: Validates type for `$comm`.
			trace_probe_log_err(ctx->offset + offs, NEED_STRING_TYPE);
			return ERR_PTR(-EINVAL);
		}
		parg->type = find_fetch_type("string", ctx->flags); // Functional Utility: Sets type to "string".
	} else
		parg->type = find_fetch_type(t, ctx->flags); // Functional Utility: Finds fetch type.

	// Block Logic: Returns error if fetch type not found.
	if (!parg->type) {
		trace_probe_log_err(ctx->offset + offs, BAD_TYPE);
		return ERR_PTR(-EINVAL);
	}

	return t;
}

/* After parsing, adjust the fetch_insn according to the probe_arg */
/**
 * @brief Finalizes the `fetch_insn` array based on the `probe_arg` type.
 * @param code Pointer to the current `fetch_insn` pointer.
 * @param parg Pointer to `probe_arg`.
 * @param type The type string.
 * @param type_offset Offset within the argument for error reporting.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function inserts store operations (`FETCH_OP_ST_STRING`, `FETCH_OP_ST_MEM`, etc.),
 * bitfield modifications, and array loop operations as needed.
 */
static int finalize_fetch_insn(struct fetch_insn *code,
			       struct probe_arg *parg,
			       char *type,
			       int type_offset,
			       struct traceprobe_parse_context *ctx)
{
	struct fetch_insn *scode;
	int ret;

	/* Store operation */
	// Block Logic: Handles string types.
	if (parg->type->is_string) {
		/* Check bad combination of the type and the last fetch_insn. */
		// Block Logic: Validates fetch instruction for string types.
		if (!strcmp(parg->type->name, "symstr")) {
			if (code->op != FETCH_OP_REG && code->op != FETCH_OP_STACK &&
			    code->op != FETCH_OP_RETVAL && code->op != FETCH_OP_ARG &&
			    code->op != FETCH_OP_DEREF && code->op != FETCH_OP_TP_ARG) {
				trace_probe_log_err(ctx->offset + type_offset,
						    BAD_SYMSTRING);
				return -EINVAL;
			}
		} else {
			if (code->op != FETCH_OP_DEREF && code->op != FETCH_OP_UDEREF &&
			    code->op != FETCH_OP_IMM && code->op != FETCH_OP_COMM &&
			    code->op != FETCH_OP_DATA && code->op != FETCH_OP_TP_ARG) {
				trace_probe_log_err(ctx->offset + type_offset,
						    BAD_STRING);
				return -EINVAL;
			}
		}

		// Block Logic: Handles `symstr`, immediate, data, and comm operations.
		if (!strcmp(parg->type->name, "symstr") ||
		    (code->op == FETCH_OP_IMM || code->op == FETCH_OP_COMM ||
		     code->op == FETCH_OP_DATA) || code->op == FETCH_OP_TP_ARG ||
		     parg->count) {
			/*
			 * IMM, DATA and COMM is pointing actual address, those
			 * must be kept, and if parg->count != 0, this is an
			 * array of string pointers instead of string address
			 * itself.
			 * For the symstr, it doesn't need to dereference, thus
			 * it just get the value.
			 */
			code++;
			if (code->op != FETCH_OP_NOP) { // Block Logic: Checks for overflow.
				trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
				return -EINVAL;
			}
		}

		/* If op == DEREF, replace it with STRING */
		// Block Logic: Sets appropriate string store operation.
		if (!strcmp(parg->type->name, "ustring") ||
		    code->op == FETCH_OP_UDEREF)
			code->op = FETCH_OP_ST_USTRING;
		else if (!strcmp(parg->type->name, "symstr"))
			code->op = FETCH_OP_ST_SYMSTR;
		else
			code->op = FETCH_OP_ST_STRING;
		code->size = parg->type->size; // Functional Utility: Sets size.
		parg->dynamic = true; // Functional Utility: Marks as dynamic.
	} else if (code->op == FETCH_OP_DEREF) { // Block Logic: Handles dereference operations.
		code->op = FETCH_OP_ST_MEM; // Functional Utility: Sets operation to store memory.
		code->size = parg->type->size; // Functional Utility: Sets size.
	} else if (code->op == FETCH_OP_UDEREF) { // Block Logic: Handles unsigned dereference operations.
		code->op = FETCH_OP_ST_UMEM; // Functional Utility: Sets operation to store unsigned memory.
		code->size = parg->type->size; // Functional Utility: Sets size.
	} else {
		code++;
		if (code->op != FETCH_OP_NOP) { // Block Logic: Checks for overflow.
			trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
			return -E2BIG;
		}
		code->op = FETCH_OP_ST_RAW; // Functional Utility: Sets operation to store raw data.
		code->size = parg->type->size; // Functional Utility: Sets size.
	}

	/* Save storing fetch_insn. */
	scode = code;

	/* Modify operation */
	// Block Logic: Handles bitfield modifications.
	if (type != NULL) {
		/* Bitfield needs a special fetch_insn. */
		ret = __parse_bitfield_probe_arg(type, parg->type, &code);
		if (ret) {
			trace_probe_log_err(ctx->offset + type_offset, BAD_BITFIELD);
			return ret;
		}
	} else if (IS_ENABLED(CONFIG_PROBE_EVENTS_BTF_ARGS) &&
		   ctx->last_type) {
		/* If user not specified the type, try parsing BTF bitfield. */
		ret = parse_btf_bitfield(&code, ctx); // Functional Utility: Parses BTF bitfield.
		if (ret)
			return ret;
	}

	/* Loop(Array) operation */
	// Block Logic: Handles array loop operations.
	if (parg->count) {
		if (scode->op != FETCH_OP_ST_MEM &&
		    scode->op != FETCH_OP_ST_STRING &&
		    scode->op != FETCH_OP_ST_USTRING) { // Block Logic: Invalid store operation for array.
			trace_probe_log_err(ctx->offset + type_offset, BAD_STRING);
			return -EINVAL;
		}
		code++;
		if (code->op != FETCH_OP_NOP) { // Block Logic: Checks for overflow.
			trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
			return -E2BIG;
		}
		code->op = FETCH_OP_LP_ARRAY; // Functional Utility: Sets operation to loop array.
		code->param = parg->count; // Functional Utility: Stores array count.
	}

	/* Finalize the fetch_insn array. */
	code++;
	code->op = FETCH_OP_END; // Functional Utility: Marks end of instructions.

	return 0;
}

/**
 * @brief Parses the body of a probe argument.
 * @param argv The argument string.
 * @param size Input/Output: Pointer to the total size of the probe.
 * @param parg Pointer to `probe_arg`.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function parses the core part of a probe argument, including its
 * fetch instructions, type, and array information.
 */
static int traceprobe_parse_probe_arg_body(const char *argv, ssize_t *size,
					   struct probe_arg *parg,
					   struct traceprobe_parse_context *ctx)
{
	struct fetch_insn *code, *tmp = NULL;
	char *type, *arg __free(kfree) = NULL;
	int ret, len;

	len = strlen(argv);
	// Block Logic: Validates argument string length.
	if (len > MAX_ARGSTR_LEN) {
		trace_probe_log_err(ctx->offset, ARG_TOO_LONG);
		return -E2BIG;
	} else if (len == 0) {
		trace_probe_log_err(ctx->offset, NO_ARG_BODY);
		return -EINVAL;
	}

	arg = kstrdup(argv, GFP_KERNEL); // Functional Utility: Duplicates argument string.
	if (!arg)
		return -ENOMEM;

	parg->comm = kstrdup(arg, GFP_KERNEL); // Functional Utility: Duplicates comment string.
	if (!parg->comm)
		return -ENOMEM;

	type = parse_probe_arg_type(arg, parg, ctx); // Functional Utility: Parses argument type.
	if (IS_ERR(type))
		return PTR_ERR(type);

	code = tmp = kcalloc(FETCH_INSN_MAX, sizeof(*code), GFP_KERNEL); // Functional Utility: Allocates fetch instruction array.
	if (!code)
		return -ENOMEM;
	code[FETCH_INSN_MAX - 1].op = FETCH_OP_END; // Functional Utility: Marks end of array.

	ctx->last_type = NULL; // Functional Utility: Resets last BTF type.
	ret = parse_probe_arg(arg, parg->type, &code, &code[FETCH_INSN_MAX - 1],
			      ctx); // Functional Utility: Parses argument and generates fetch instructions.
	if (ret < 0)
		goto fail;

	/* Update storing type if BTF is available */
	// Block Logic: If BTF is enabled and type is not specified, uses BTF type.
	if (IS_ENABLED(CONFIG_PROBE_EVENTS_BTF_ARGS) &&
	    ctx->last_type) {
		if (!type) {
			parg->type = find_fetch_type_from_btf_type(ctx); // Functional Utility: Finds fetch type from BTF.
		} else if (strstr(type, "string")) { // Block Logic: Checks for string type and prepares BTF string fetch.
			ret = check_prepare_btf_string_fetch(type, &code, ctx);
			if (ret)
				goto fail;
		}
	}
	parg->offset = *size; // Functional Utility: Stores argument offset.
	*size += parg->type->size * (parg->count ?: 1); // Functional Utility: Updates total size.

	// Block Logic: If array count is specified, formats the print string.
	if (parg->count) {
		len = strlen(parg->type->fmttype) + 6;
		parg->fmt = kmalloc(len, GFP_KERNEL);
		if (!parg->fmt) {
			ret = -ENOMEM;
			goto fail;
		}
		snprintf(parg->fmt, len, "%s[%d]", parg->type->fmttype,
			 parg->count);
	}

	ret = finalize_fetch_insn(code, parg, type, type ? type - arg : 0, ctx); // Functional Utility: Finalizes fetch instructions.
	if (ret < 0)
		goto fail;

	// Block Logic: Finds the end of the generated code.
	for (; code < tmp + FETCH_INSN_MAX; code++)
		if (code->op == FETCH_OP_END)
			break;
	/* Shrink down the code buffer */
	parg->code = kcalloc(code - tmp + 1, sizeof(*code), GFP_KERNEL); // Functional Utility: Shrinks code buffer.
	if (!parg->code)
		ret = -ENOMEM;
	else
		memcpy(parg->code, tmp, sizeof(*code) * (code - tmp + 1)); // Functional Utility: Copies code.

fail:
	// Block Logic: Frees allocated data on failure.
	if (ret < 0) {
		for (code = tmp; code < tmp + FETCH_INSN_MAX; code++)
			if (code->op == FETCH_NOP_SYMBOL ||
			    code->op == FETCH_OP_DATA)
				kfree(code->data);
	}
	kfree(tmp); // Functional Utility: Frees temporary buffer.

	return ret;
}

/* Return 1 if name is reserved or already used by another argument */
/**
 * @brief Checks for conflicts in probe argument field names.
 * @param name The argument name to check.
 * @param args Array of existing `probe_arg` structures.
 * @param narg Number of existing arguments.
 * @return 1 if the name is reserved or already used, 0 otherwise.
 */
static int traceprobe_conflict_field_name(const char *name,
					  struct probe_arg *args, int narg)
{
	int i;

	// Block Logic: Checks against reserved field names.
	for (i = 0; i < ARRAY_SIZE(reserved_field_names); i++)
		if (strcmp(reserved_field_names[i], name) == 0)
			return 1;

	// Block Logic: Checks against already used argument names.
	for (i = 0; i < narg; i++)
		if (strcmp(args[i].name, name) == 0)
			return 1;

	return 0;
}

/**
 * @brief Generates a name for a probe argument.
 * @param arg The argument string.
 * @param idx The index of the argument.
 * @return A newly allocated string for the argument name, or NULL on memory allocation failure.
 *
 * This function attempts to use the argument itself as a name (if BTF is
 * enabled) or falls back to a generic "argN" name.
 */
static char *generate_probe_arg_name(const char *arg, int idx)
{
	char *name = NULL;
	const char *end;

	/*
	 * If argument name is omitted, try arg as a name (BTF variable)
	 * or "argN".
	 */
	// Block Logic: If BTF is enabled, tries to extract name from argument.
	if (IS_ENABLED(CONFIG_PROBE_EVENTS_BTF_ARGS)) {
		end = strchr(arg, ':');
		if (!end)
			end = arg + strlen(arg);

		name = kmemdup_nul(arg, end - arg, GFP_KERNEL); // Functional Utility: Duplicates potential name.
		if (!name || !is_good_name(name)) { // Block Logic: Validates generated name.
			kfree(name);
			name = NULL;
		}
	}

	// Block Logic: If no name generated, falls back to "argN".
	if (!name)
		name = kasprintf(GFP_KERNEL, "arg%d", idx + 1);

	return name;
}

/**
 * @brief Parses a probe argument from a string.
 * @param tp Pointer to `trace_probe`.
 * @param i Index of the argument.
 * @param arg The argument string.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function parses a single probe argument, extracting its name,
 * and then calls `traceprobe_parse_probe_arg_body` to parse the argument's
 * body and generate fetch instructions.
 */
int traceprobe_parse_probe_arg(struct trace_probe *tp, int i, const char *arg,
			       struct traceprobe_parse_context *ctx)
{
	struct probe_arg *parg = &tp->args[i];
	const char *body;

	ctx->tp = tp; // Functional Utility: Sets current trace probe in context.
	body = strchr(arg, '='); // Functional Utility: Checks for '=' separator for name.
	// Block Logic: If name is provided (e.g., `my_arg=...`).
	if (body) {
		if (body - arg > MAX_ARG_NAME_LEN) { // Block Logic: Checks name length.
			trace_probe_log_err(0, ARG_NAME_TOO_LONG);
			return -EINVAL;
		} else if (body == arg) { // Block Logic: Empty name.
			trace_probe_log_err(0, NO_ARG_NAME);
			return -EINVAL;
		}
		parg->name = kmemdup_nul(arg, body - arg, GFP_KERNEL); // Functional Utility: Duplicates name.
		body++; // Functional Utility: Moves to argument body.
	} else { // Block Logic: If name is omitted.
		parg->name = generate_probe_arg_name(arg, i); // Functional Utility: Generates default name.
		body = arg; // Functional Utility: Argument body is the whole string.
	}
	if (!parg->name) // Block Logic: Handles name allocation failure.
		return -ENOMEM;

	// Block Logic: Validates argument name and checks for conflicts.
	if (!is_good_name(parg->name)) {
		trace_probe_log_err(0, BAD_ARG_NAME);
		return -EINVAL;
	}
	if (traceprobe_conflict_field_name(parg->name, tp->args, i)) {
		trace_probe_log_err(0, USED_ARG_NAME);
		return -EINVAL;
	}
	ctx->offset = body - arg; // Functional Utility: Sets offset for error reporting.
	/* Parse fetch argument */
	return traceprobe_parse_probe_arg_body(body, &tp->size, parg, ctx); // Functional Utility: Parses argument body.
}

/**
 * @brief Frees resources associated with a `probe_arg` structure.
 * @param arg Pointer to `probe_arg`.
 *
 * This function frees the `fetch_insn` code array, name, comment,
 * and format string associated with the argument.
 */
void traceprobe_free_probe_arg(struct probe_arg *arg)
{
	struct fetch_insn *code = arg->code;

	// Block Logic: Iterates through fetch instructions and frees data.
	while (code && code->op != FETCH_OP_END) {
		if (code->op == FETCH_NOP_SYMBOL ||
		    code->op == FETCH_OP_DATA)
				kfree(code->data);
		code++;
	}
	kfree(arg->code); // Functional Utility: Frees instruction array.
	kfree(arg->name); // Functional Utility: Frees name.
	kfree(arg->comm); // Functional Utility: Frees comment.
	kfree(arg->fmt); // Functional Utility: Frees format string.
}

/**
 * @brief Checks if an argument list contains a variable argument (`$arg*`).
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @param args_idx Output parameter for the index of the variable argument.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return 1 if variable argument found, 0 otherwise, or -EINVAL on error.
 */
static int argv_has_var_arg(int argc, const char *argv[], int *args_idx,
			    struct traceprobe_parse_context *ctx)
{
	int i, found = 0;

	// Block Logic: Iterates through arguments to find `$arg*`.
	for (i = 0; i < argc; i++)
		if (str_has_prefix(argv[i], "$arg")) {
			trace_probe_log_set_index(i + 2);

			// Block Logic: `$arg*` only valid for function entry/return.
			if (!tparg_is_function_entry(ctx->flags) &&
			    !tparg_is_function_return(ctx->flags)) {
				trace_probe_log_err(0, NOFENTRY_ARGS);
				return -EINVAL;
			}

			// Block Logic: If `$argN` (specific number), just marks as found.
			if (isdigit(argv[i][4])) {
				found = 1;
				continue;
			}

			// Block Logic: If `$arg*` (wildcard), checks for conflicts.
			if (argv[i][4] != '*') {
				trace_probe_log_err(0, BAD_VAR);
				return -EINVAL;
			}

			if (*args_idx >= 0 && *args_idx < argc) { // Block Logic: Only one `$arg*` allowed.
				trace_probe_log_err(0, DOUBLE_ARGS);
				return -EINVAL;
			}
			found = 1;
			*args_idx = i; // Functional Utility: Stores index of `$arg*`.
		}

	return found;
}

/**
 * @brief Prints the Nth BTF argument into a buffer.
 * @param idx The index of the argument.
 * @param type The type string (e.g., ":s32").
 * @param buf Output buffer.
 * @param bufsize Size of the buffer.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return Number of bytes written on success, or a negative errno on failure.
 */
static int sprint_nth_btf_arg(int idx, const char *type,
			      char *buf, int bufsize,
			      struct traceprobe_parse_context *ctx)
{
	const char *name;
	int ret;

	// Block Logic: Checks for valid argument index.
	if (idx >= ctx->nr_params) {
		trace_probe_log_err(0, NO_BTFARG);
		return -ENOENT;
	}
	name = btf_name_by_offset(ctx->btf, ctx->params[idx].name_off); // Functional Utility: Gets parameter name from BTF.
	if (!name) {
		trace_probe_log_err(0, NO_BTF_ENTRY);
		return -ENOENT;
	}
	ret = snprintf(buf, bufsize, "%s%s", name, type); // Functional Utility: Formats name and type.
	if (ret >= bufsize) {
		trace_probe_log_err(0, ARGS_2LONG);
		return -E2BIG;
	}
	return ret;
}

/* Return new_argv which must be freed after use */
/**
 * @brief Expands meta arguments (like `$arg*` and `$argN`) in an argument list.
 * @param argc Number of original arguments.
 * @param argv Array of original argument strings.
 * @param new_argc Output parameter for the new argument count.
 * @param buf Buffer to store expanded arguments.
 * @param bufsize Size of the buffer.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return A newly allocated array of expanded argument strings on success,
 *         or an `ERR_PTR` on failure. The caller must free this array.
 *
 * This function handles `meta-arguments` that expand into multiple actual
 * arguments, primarily by using BTF information to resolve function parameters.
 */
const char **traceprobe_expand_meta_args(int argc, const char *argv[],
					 int *new_argc, char *buf, int bufsize,
					 struct traceprobe_parse_context *ctx)
{
	const struct btf_param *params = NULL;
	int i, j, n, used, ret, args_idx = -1;
	const char **new_argv __free(kfree) = NULL;

	ret = argv_has_var_arg(argc, argv, &args_idx, ctx); // Functional Utility: Checks for variable arguments.
	if (ret < 0)
		return ERR_PTR(ret);

	// Block Logic: If no variable arguments, returns original argument count.
	if (!ret) {
		*new_argc = argc;
		return NULL;
	}

	// Functional Utility: Queries BTF context if not already set.
	ret = query_btf_context(ctx);
	if (ret < 0 || ctx->nr_params == 0) {
		if (args_idx != -1) { // Block Logic: `$arg*` requires BTF information.
			/* $arg* requires BTF info */
			trace_probe_log_err(0, NOSUP_BTFARG);
			return (const char **)params;
		}
		*new_argc = argc;
		return NULL;
	}

	// Block Logic: Calculates new argument count.
	if (args_idx >= 0)
		*new_argc = argc + ctx->nr_params - 1;
	else
		*new_argc = argc;

	new_argv = kcalloc(*new_argc, sizeof(char *), GFP_KERNEL); // Functional Utility: Allocates new argument array.
	if (!new_argv)
		return ERR_PTR(-ENOMEM);

	used = 0;
	// Block Logic: Iterates through original arguments, expanding meta arguments.
	for (i = 0, j = 0; i < argc; i++) {
		trace_probe_log_set_index(i + 2);
		// Block Logic: If it's the `$arg*` position, expands all BTF parameters.
		if (i == args_idx) {
			for (n = 0; n < ctx->nr_params; n++) {
				ret = sprint_nth_btf_arg(n, "", buf + used,
							 bufsize - used, ctx);
				if (ret < 0)
					return ERR_PTR(ret);

				new_argv[j++] = buf + used;
				used += ret + 1;
			}
			continue;
		}

		// Block Logic: If it's `$argN`, expands to Nth BTF parameter.
		if (str_has_prefix(argv[i], "$arg")) {
			char *type = NULL;

			n = simple_strtoul(argv[i] + 4, &type, 10);
			if (type && !(*type == ':' || *type == '\0')) {
				trace_probe_log_err(0, BAD_VAR);
				return ERR_PTR(-ENOENT);
			}
			/* Note: $argN starts from $arg1 */
			ret = sprint_nth_btf_arg(n - 1, type, buf + used,
						 bufsize - used, ctx);
			if (ret < 0)
				return ERR_PTR(ret);
			new_argv[j++] = buf + used;
			used += ret + 1;
		} else
			new_argv[j++] = argv[i]; // Functional Utility: Copies non-meta arguments.
	}

	return_ptr(new_argv);
}

/* @buf: *buf must be equal to NULL. Caller must to free *buf */
/**
 * @brief Expands dentry arguments (e.g., `%pD`) in an argument list.
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @param buf Output parameter for the newly allocated buffer containing expanded arguments.
 * @return 0 on success, -EINVAL on error, -ENOMEM on memory allocation failure.
 *
 * This function recognizes `%p[dD]` format specifiers in arguments and
 * expands them into explicit dereference operations to access dentry
 * names or file paths.
 */
int traceprobe_expand_dentry_args(int argc, const char *argv[], char **buf)
{
	int i, used, ret;
	const int bufsize = MAX_DENTRY_ARGS_LEN;
	char *tmpbuf __free(kfree) = NULL;

	// Block Logic: Ensures buffer is initially NULL.
	if (*buf)
		return -EINVAL;

	used = 0;
	// Block Logic: Iterates through arguments.
	for (i = 0; i < argc; i++) {
		char *tmp __free(kfree) = NULL;
		char *equal;
		size_t arg_len;

		// Block Logic: Checks for dentry format specifier.
		if (!glob_match("*:%p[dD]", argv[i]))
			continue;

		// Block Logic: Allocates buffer if not already.
		if (!tmpbuf) {
			tmpbuf = kmalloc(bufsize, GFP_KERNEL);
			if (!tmpbuf)
				return -ENOMEM;
		}

		tmp = kstrdup(argv[i], GFP_KERNEL); // Functional Utility: Duplicates argument.
		if (!tmp)
			return -ENOMEM;

		equal = strchr(tmp, '='); // Functional Utility: Checks for alias.
		if (equal)
			*equal = '\0'; // Functional Utility: Null-terminates alias part.
		arg_len = strlen(argv[i]);
		tmp[arg_len - 4] = '\0'; // Functional Utility: Removes `:%p[dD]` suffix.
		// Block Logic: Formats based on `d` or `D` specifier.
		if (argv[i][arg_len - 1] == 'd')
			ret = snprintf(tmpbuf + used, bufsize - used,
				       "%s%s+0x0(+0x%zx(%s)):string",
				       equal ? tmp : "", equal ? "=" : "",
				       offsetof(struct dentry, d_name.name),
				       equal ? equal + 1 : tmp);
		else
			ret = snprintf(tmpbuf + used, bufsize - used,
				       "%s%s+0x0(+0x%zx(+0x%zx(%s))):string",
				       equal ? tmp : "", equal ? "=" : "",
				       offsetof(struct dentry, d_name.name),
				       offsetof(struct file, f_path.dentry),
				       equal ? equal + 1 : tmp);

		if (ret >= bufsize - used)
			return -ENOMEM;
		argv[i] = tmpbuf + used; // Functional Utility: Updates argument pointer.
		used += ret + 1; // Functional Utility: Updates used size.
	}

	*buf = no_free_ptr(tmpbuf); // Functional Utility: Assigns allocated buffer.
	return 0;
}

/**
 * @brief Finishes parsing context, including clearing BTF context.
 * @param ctx Pointer to `traceprobe_parse_context`.
 */
void traceprobe_finish_parse(struct traceprobe_parse_context *ctx)
{
	clear_btf_context(ctx); // Functional Utility: Clears BTF context.
}

/**
 * @brief Updates a `probe_arg` by resolving symbolic addresses in its fetch instructions.
 * @param arg Pointer to `probe_arg`.
 * @return 0 on success, -EINVAL if instruction array invalid, -ENOENT if symbol not found.
 *
 * This function iterates through the fetch instructions of a `probe_arg` and
 * resolves `FETCH_NOP_SYMBOL` operations into direct addresses using `kallsyms_lookup_name`.
 */
int traceprobe_update_arg(struct probe_arg *arg)
{
	struct fetch_insn *code = arg->code;
	long offset;
	char *tmp;
	char c;
	int ret = 0;

	// Block Logic: Iterates through fetch instructions.
	while (code && code->op != FETCH_OP_END) {
		// Block Logic: If a symbol needs to be resolved.
		if (code->op == FETCH_NOP_SYMBOL) {
			if (code[1].op != FETCH_OP_IMM)
				return -EINVAL;

			tmp = strpbrk(code->data, "+-"); // Functional Utility: Checks for offset.
			if (tmp)
				c = *tmp;
			ret = traceprobe_split_symbol_offset(code->data,
							     &offset); // Functional Utility: Splits symbol and offset.
			if (ret)
				return ret;

			code[1].immediate =
				(unsigned long)kallsyms_lookup_name(code->data); // Functional Utility: Looks up symbol address.
			if (tmp)
				*tmp = c; // Functional Utility: Restores original string.
			if (!code[1].immediate)
				return -ENOENT;
			code[1].immediate += offset; // Functional Utility: Applies offset.
		}
		code++;
	}
	return 0;
}

/* When len=0, we just calculate the needed length */
#define LEN_OR_ZERO (len ? len - pos : 0)
/**
 * @brief Generates the `print_fmt` string for a trace probe.
 * @param tp Pointer to `trace_probe`.
 * @param buf Output buffer for the format string.
 * @param len Length of the buffer.
 * @param ptype Print type (normal, return, event).
 * @return The length of the generated format string.
 *
 * This function constructs a `print_fmt` string dynamically based on
 * the probe's arguments and print type, suitable for `ftrace_event_call`.
 */
static int __set_print_fmt(struct trace_probe *tp, char *buf, int len,
			   enum probe_print_type ptype)
{
	struct probe_arg *parg;
	int i, j;
	int pos = 0;
	const char *fmt, *arg;

	// Block Logic: Sets base format and arguments based on print type.
	switch (ptype) {
	case PROBE_PRINT_NORMAL:
		fmt = "(%lx)";
		arg = ", REC->" FIELD_STRING_IP;
		break;
	case PROBE_PRINT_RETURN:
		fmt = "(%lx <- %lx)";
		arg = ", REC->" FIELD_STRING_FUNC ", REC->" FIELD_STRING_RETIP;
		break;
	case PROBE_PRINT_EVENT:
		fmt = "";
		arg = "";
		break;
	default:
		WARN_ON_ONCE(1);
		return 0;
	}

	pos += snprintf(buf + pos, LEN_OR_ZERO, "\"%s", fmt);

	// Block Logic: Appends argument names and formats.
	for (i = 0; i < tp->nr_args; i++) {
		parg = tp->args + i;
		pos += snprintf(buf + pos, LEN_OR_ZERO, " %s=", parg->name);
		// Block Logic: Handles array formatting.
		if (parg->count) {
			pos += snprintf(buf + pos, LEN_OR_ZERO, "{%s",
					parg->type->fmttype);
			for (j = 1; j < parg->count; j++)
				pos += snprintf(buf + pos, LEN_OR_ZERO, ",%s",
						parg->type->fmttype);
			pos += snprintf(buf + pos, LEN_OR_ZERO, "}");
		} else
			pos += snprintf(buf + pos, LEN_OR_ZERO, "%s",
					parg->type->fmttype);
	}

	pos += snprintf(buf + pos, LEN_OR_ZERO, "\"%s", arg);

	// Block Logic: Appends argument values for the print format.
	for (i = 0; i < tp->nr_args; i++) {
		parg = tp->args + i;
		if (parg->count) {
			if (parg->type->is_string)
				fmt = ", __get_str(%s[%d])";
			else
				fmt = ", REC->%s[%d]";
			for (j = 0; j < parg->count; j++)
				pos += snprintf(buf + pos, LEN_OR_ZERO,
						fmt, parg->name, j);
		} else {
			if (parg->type->is_string)
				fmt = ", __get_str(%s)";
			else
				fmt = ", REC->%s";
			pos += snprintf(buf + pos, LEN_OR_ZERO,
					fmt, parg->name);
		}
	}

	/* return the length of print_fmt */
	return pos;
}
#undef LEN_OR_ZERO

/**
 * @brief Sets the `print_fmt` field of a `trace_event_call` for a trace probe.
 * @param tp Pointer to `trace_probe`.
 * @param ptype Print type (normal, return, event).
 * @return 0 on success, -ENOMEM on memory allocation failure.
 *
 * This function calculates the required buffer size, allocates memory,
 * and generates the `print_fmt` string for the trace probe's event call.
 */
int traceprobe_set_print_fmt(struct trace_probe *tp, enum probe_print_type ptype)
{
	struct trace_event_call *call = trace_probe_event_call(tp);
	int len;
	char *print_fmt;

	/* First: called with 0 length to calculate the needed length */
	len = __set_print_fmt(tp, NULL, 0, ptype); // Functional Utility: Calculates required length.
	print_fmt = kmalloc(len + 1, GFP_KERNEL); // Functional Utility: Allocates memory for format string.
	if (!print_fmt)
		return -ENOMEM;

	/* Second: actually write the @print_fmt */
	__set_print_fmt(tp, print_fmt, len + 1, ptype); // Functional Utility: Writes format string.
	call->print_fmt = print_fmt; // Functional Utility: Assigns to event call.

	return 0;
}

/**
 * @brief Defines argument fields for a `trace_event_call`.
 * @param event_call Pointer to `trace_event_call`.
 * @param offset Base offset for argument fields.
 * @param tp Pointer to `trace_probe`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function defines the fields of a trace event based on the arguments
 * (`probe_arg`) of the `trace_probe`.
 */
int traceprobe_define_arg_fields(struct trace_event_call *event_call,
				 size_t offset, struct trace_probe *tp)
{
	int ret, i;

	/* Set argument names as fields */
	// Block Logic: Iterates through probe arguments.
	for (i = 0; i < tp->nr_args; i++) {
		struct probe_arg *parg = &tp->args[i];
		const char *fmt = parg->type->fmttype;
		int size = parg->type->size;

		if (parg->fmt) // Functional Utility: Uses custom format if available.
			fmt = parg->fmt;
		if (parg->count) // Functional Utility: Adjusts size for arrays.
			size *= parg->count;
		ret = trace_define_field(event_call, fmt, parg->name,
					 offset + parg->offset, size,
					 parg->type->is_signed,
					 FILTER_OTHER); // Functional Utility: Defines the field.
		if (ret)
			return ret;
	}
	return 0;
}

/**
 * @brief Frees resources associated with a `trace_probe_event` structure.
 * @param tpe Pointer to `trace_probe_event`.
 *
 * This function frees the system name, event name, and print format string.
 */
static void trace_probe_event_free(struct trace_probe_event *tpe)
{
	kfree(tpe->class.system);
	kfree(tpe->call.name);
	kfree(tpe->call.print_fmt);
	kfree(tpe);
}

/**
 * @brief Appends a `trace_probe` to an existing `trace_probe_event`.
 * @param tp Pointer to the `trace_probe` to append.
 * @param to Pointer to the `trace_probe` to append to.
 * @return 0 on success, -EBUSy if the `trace_probe` has siblings.
 *
 * This function moves the `tp` to the list of probes managed by `to->event`.
 */
int trace_probe_append(struct trace_probe *tp, struct trace_probe *to)
{
	// Block Logic: Returns busy if `tp` already has siblings.
	if (trace_probe_has_sibling(tp))
		return -EBUSY;

	list_del_init(&tp->list); // Functional Utility: Removes from current list.
	trace_probe_event_free(tp->event); // Functional Utility: Frees old event.

	tp->event = to->event; // Functional Utility: Assigns new event.
	list_add_tail(&tp->list, trace_probe_probe_list(to)); // Functional Utility: Adds to new event's probe list.
	return 0;
}
