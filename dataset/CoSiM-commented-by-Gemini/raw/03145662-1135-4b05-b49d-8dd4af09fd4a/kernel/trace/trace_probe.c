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
	if (type && (flags & TPARG_FL_USER) &&
	    (!strcmp(type, "symbol") || !strcmp(type, "symstr")))
		return NULL;

	if (!type)
		type = DEFAULT_FETCH_TYPE_STR;

	/* Special case: bitfield */
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

	if (!trace_probe_log.argv)
		return;

	/* Recalculate the length and allocate buffer */
	for (i = 0; i < trace_probe_log.argc; i++) {
		if (i == trace_probe_log.index)
			pos = len;
		len += strlen(trace_probe_log.argv[i]) + 1;
	}
	command = kzalloc(len, GFP_KERNEL);
	if (!command)
		return;

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
	p = command;
	for (i = 0; i < trace_probe_log.argc; i++) {
		len = strlen(trace_probe_log.argv[i]);
		strcpy(p, trace_probe_log.argv[i]);
		p[len] = ' ';
		p += len + 1;
	}
	*(p - 1) = '\0';

	tracing_log_err(NULL, trace_probe_log.subsystem, command,
			trace_probe_err_text, err_type, pos + offset);

	kfree(command);
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

	if (!offset)
		return -EINVAL;

	tmp = strpbrk(symbol, "+-");
	if (tmp) {
		ret = kstrtol(tmp, 0, offset);
		if (ret)
			return ret;
		*tmp = '\0';
	} else
		*offset = 0;

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

	slash = strchr(event, '/');
	if (!slash)
		slash = strchr(event, '.');

	if (slash) {
		if (slash == event) {
			trace_probe_log_err(offset, NO_GROUP_NAME);
			return -EINVAL;
		}
		if (slash - event + 1 > MAX_EVENT_NAME_LEN) {
			trace_probe_log_err(offset, GROUP_TOO_LONG);
			return -EINVAL;
		}
		strscpy(buf, event, slash - event + 1);
		if (!is_good_system_name(buf)) {
			trace_probe_log_err(offset, BAD_GROUP_NAME);
			return -EINVAL;
		}
		*pgroup = buf;
		*pevent = slash + 1;
		offset += slash - event + 1;
		event = *pevent;
	}
	len = strlen(event);
	if (len == 0) {
		if (slash) {
			*pevent = NULL;
			return 0;
		}
		trace_probe_log_err(offset, NO_EVENT_NAME);
		return -EINVAL;
	} else if (len >= MAX_EVENT_NAME_LEN) {
		trace_probe_log_err(offset, EVENT_TOO_LONG);
		return -EINVAL;
	}
	if (!is_good_name(event)) {
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

	head = trace_get_fields(ctx->event);
	list_for_each_entry(field, head, link) {
		if (!strcmp(arg, field->name)) {
			code->op = FETCH_OP_TP_ARG;
			code->data = field;
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

	real_type = btf_type_skip_modifiers(btf, type->type, &tid);
	if (!real_type)
		return false;

	if (BTF_INFO_KIND(real_type->info) != BTF_KIND_INT)
		return false;

	intdata = btf_type_int(real_type);
	return !(BTF_INT_ENCODING(intdata) & BTF_INT_SIGNED)
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

	if (BTF_INFO_KIND(type->info) != BTF_KIND_ARRAY)
		return false;

	array = (const struct btf_array *)(type + 1);

	real_type = btf_type_skip_modifiers(btf, array->type, &tid);

	intdata = btf_type_int(real_type);
	return !(BTF_INT_ENCODING(intdata) & BTF_INT_SIGNED)
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

	if (!btf || !ctx->last_type)
		return 0;

	/* char [] does not need any change. */
	if (btf_type_is_char_array(btf, ctx->last_type))
		return 0;

	/* char * requires dereference the pointer. */
	if (btf_type_is_char_ptr(btf, ctx->last_type)) {
		struct fetch_insn *code = *pcode + 1;

		if (code->op == FETCH_OP_END) {
			trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
			return -E2BIG;
		}
		if (typename[0] == 'u')
			code->op = FETCH_OP_UDEREF;
		else
			code->op = FETCH_OP_DEREF;
		code->offset = 0;
		*pcode = code;
		return 0;
	}
	/* Other types are not available for string */
	trace_probe_log_err(ctx->offset, BAD_TYPE4STR);
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
		if (IS_ENABLED(CONFIG_64BIT))
			return "x64";
		else
			return "x32";
	case BTF_KIND_INT:
		intdata = btf_type_int(type);
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
			ctx->last_bitsize = BTF_INT_BITS(intdata);
			ctx->last_bitoffs += BTF_INT_OFFSET(intdata);
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

	if (ctx->btf)
		return 0;

	if (!ctx->funcname)
		return -EINVAL;

	type = btf_find_func_proto(ctx->funcname, &btf);
	if (!type)
		return -ENOENT;

	ctx->btf = btf;
	ctx->proto = type;

	/* ctx->params is optional, since func(void) will not have params. */
	nr = 0;
	param = btf_get_func_param(type, &nr);
	if (!IS_ERR_OR_NULL(param)) {
		/* Hide the first 'data' argument of tracepoint */
		if (ctx->flags & TPARG_FL_TPOINT) {
			nr--;
			param++;
		}
	}

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
		btf_put(ctx->btf);
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

	field = strpbrk(varname, ".-");
	if (field) {
		if (field[0] == '-' && field[1] == '>') {
			field[0] = '\0';
			field += 2;
			ret = 1;
		} else if (field[0] == '.') {
			field[0] = '\0';
			field += 1;
		} else {
			trace_probe_log_err(ctx->offset + field - varname, BAD_HYPHEN);
			return -EINVAL;
		}
		*next_field = field;
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
		if (BTF_INFO_KIND(type->info) != BTF_KIND_PTR) {
			trace_probe_log_err(ctx->offset, NO_PTR_STRCT);
			return -EINVAL;
		}
		/* Convert a struct pointer type to a struct type */
		type = btf_type_skip_modifiers(ctx->btf, type->type, &tid);
		if (!type) {
			trace_probe_log_err(ctx->offset, BAD_BTF_TID);
			return -EINVAL;
		}

		bitoffs = 0;
		do {
			/* Inner loop for solving dot operator ('.') */
			next = NULL;
			is_ptr = split_next_field(fieldname, &next, ctx);
			if (is_ptr < 0)
				return is_ptr;

			anon_offs = 0;
			field = btf_find_struct_member(ctx->btf, type, fieldname,
						       &anon_offs);
			if (IS_ERR(field)) {
				trace_probe_log_err(ctx->offset, BAD_BTF_TID);
				return PTR_ERR(field);
			}
			if (!field) {
				trace_probe_log_err(ctx->offset, NO_BTF_FIELD);
				return -ENOENT;
			}
			/* Add anonymous structure/union offset */
			bitoffs += anon_offs;

			/* Accumulate the bit-offsets of the dot-connected fields */
			if (btf_type_kflag(type)) {
				bitoffs += BTF_MEMBER_BIT_OFFSET(field->offset);
				ctx->last_bitsize = BTF_MEMBER_BITFIELD_SIZE(field->offset);
			} else {
				bitoffs += field->offset;
				ctx->last_bitsize = 0;
			}

			type = btf_type_skip_modifiers(ctx->btf, field->type, &tid);
			if (!type) {
				trace_probe_log_err(ctx->offset, BAD_BTF_TID);
				return -EINVAL;
			}

			ctx->offset += next - fieldname;
			fieldname = next;
		} while (!is_ptr && fieldname);

		if (++code == end) {
			trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
			return -EINVAL;
		}
		code->op = FETCH_OP_DEREF;	/* TODO: user deref support */
		code->offset = bitoffs / 8;
		*pcode = code;

		ctx->last_bitoffs = bitoffs % 8;
		ctx->last_type = type;
	} while (fieldname);

	return 0;
}

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

	if (WARN_ON_ONCE(!ctx->funcname))
		return -EINVAL;

	is_ptr = split_next_field(varname, &field, ctx);
	if (is_ptr < 0)
		return is_ptr;
	if (!is_ptr && field) {
		/* dot-connected field on an argument is not supported. */
		trace_probe_log_err(ctx->offset + field - varname,
				    NOSUP_DAT_ARG);
		return -EOPNOTSUPP;
	}

	if (ctx->flags & TPARG_FL_RETURN && !strcmp(varname, "$retval")) {
		code->op = FETCH_OP_RETVAL;
		/* Check whether the function return type is not void */
		if (query_btf_context(ctx) == 0) {
			if (ctx->proto->type == 0) {
				trace_probe_log_err(ctx->offset, NO_RETVAL);
				return -ENOENT;
			}
			tid = ctx->proto->type;
			goto found;
		}
		if (field) {
			trace_probe_log_err(ctx->offset + field - varname,
					    NO_BTF_ENTRY);
			return -ENOENT;
		}
		return 0;
	}

	if (!ctx->btf) {
		ret = query_btf_context(ctx);
		if (ret < 0 || ctx->nr_params == 0) {
			trace_probe_log_err(ctx->offset, NO_BTF_ENTRY);
			return -ENOENT;
		}
	}
	params = ctx->params;

	for (i = 0; i < ctx->nr_params; i++) {
		const char *name = btf_name_by_offset(ctx->btf, params[i].name_off);

		if (name && !strcmp(name, varname)) {
			if (tparg_is_function_entry(ctx->flags)) {
				code->op = FETCH_OP_ARG;
				if (ctx->flags & TPARG_FL_TPOINT)
					code->param++;
				else
					code->param = i;
			} else if (tparg_is_function_return(ctx->flags)) {
				/* function entry argument access from return probe */
				ret = __store_entry_arg(ctx->tp, i);
				if (ret < 0)	/* internal error */
					return ret;

				code->op = FETCH_OP_EDATA;
				code->offset = ret;
			}
			tid = params[i].type;
			goto found;
		}
	}
	trace_probe_log_err(ctx->offset, NO_BTFARG);
	return -ENOENT;

found:
	type = btf_type_skip_modifiers(ctx->btf, tid, &tid);
	if (!type) {
		trace_probe_log_err(ctx->offset, BAD_BTF_TID);
		return -EINVAL;
	}
	/* Initialize the last type information */
	ctx->last_type = type;
	ctx->last_bitoffs = 0;
	ctx->last_bitsize = 0;
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

	if (btf && ctx->last_type)
		typestr = fetch_type_from_btf_type(btf, ctx->last_type, ctx);

	return find_fetch_type(typestr, ctx->flags);
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

	if ((ctx->last_bitsize % 8 == 0) && ctx->last_bitoffs == 0)
		return 0;

	code++;
	if (code->op != FETCH_OP_NOP) {
		trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
		return -EINVAL;
	}
	*pcode = code;

	code->op = FETCH_OP_MOD_BF;
	code->lshift = 64 - (ctx->last_bitsize + ctx->last_bitoffs);
	code->rshift = 64 - ctx->last_bitsize;
	code->basesize = 64 / 8;
	return (BYTES_TO_BITS(t->size) < (bw + bo)) ? -EINVAL : 0;
}

#else
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

	if (!earg) {
		earg = kzalloc(sizeof(*tp->entry_arg), GFP_KERNEL);
		if (!earg)
			return -ENOMEM;
		earg->size = 2 * tp->nr_args + 1;
		earg->code = kcalloc(earg->size, sizeof(struct fetch_insn),
				     GFP_KERNEL);
		if (!earg->code) {
			kfree(earg);
			return -ENOMEM;
		}
		/* Fill the code buffer with 'end' to simplify it */
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
	for (i = 0; i < earg->size - 1; i++) {
		switch (earg->code[i].op) {
		case FETCH_OP_END:
			earg->code[i].op = FETCH_OP_ARG;
			earg->code[i].param = argnum;
			earg->code[i + 1].op = FETCH_OP_ST_EDATA;
			earg->code[i + 1].offset = offset;
			return offset;
		case FETCH_OP_ARG:
			match = (earg->code[i].param == argnum);
			break;
		case FETCH_OP_ST_EDATA:
			offset = earg->code[i].offset;
			if (match)
				return offset;
			offset += sizeof(unsigned long);
			break;
		default:
			break;
		}
	}
	return -ENOSPC;
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
	for (i = 0; i < earg->size; i++) {
		switch (earg->code[i].op) {
		case FETCH_OP_END:
			goto out;
		case FETCH_OP_ST_EDATA:
			size = earg->code[i].offset + sizeof(unsigned long);
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

	if (!earg)
		return;

	for (i = 0; i < earg->size; i++) {
		struct fetch_insn *code = &earg->code[i];

		switch (code->op) {
		case FETCH_OP_ARG:
			val = regs_get_kernel_argument(regs, code->param);
			break;
		case FETCH_OP_ST_EDATA:
			*(unsigned long *)((unsigned long)edata + code->offset) = val;
			break;
		case FETCH_OP_END:
			goto end;
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
	char *arg = orig_arg + 1;
	unsigned long param;
	int ret = 0;
	int len;

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

	if (str_has_prefix(arg, "retval")) {
		if (!(ctx->flags & TPARG_FL_RETURN)) {
			err = TP_ERR_RETVAL_ON_PROBE;
			goto inval;
		}
		if (!(ctx->flags & TPARG_FL_KERNEL) ||
		    !IS_ENABLED(CONFIG_PROBE_EVENTS_BTF_ARGS)) {
			code->op = FETCH_OP_RETVAL;
			return 0;
		}
		return parse_btf_arg(orig_arg, pcode, end, ctx);
	}

	len = str_has_prefix(arg, "stack");
	if (len) {
		if (arg[len] == '\0') {
			code->op = FETCH_OP_STACKP;
			return 0;
		}

		if (isdigit(arg[len])) {
			ret = kstrtoul(arg + len, 10, &param);
			if (ret)
				goto inval;

			if ((ctx->flags & TPARG_FL_KERNEL) &&
			    param > PARAM_MAX_STACK) {
				err = TP_ERR_BAD_STACK_NUM;
				goto inval;
			}
			code->op = FETCH_OP_STACK;
			code->param = (unsigned int)param;
			return 0;
		}
		goto inval;
	}

	if (strcmp(arg, "comm") == 0 || strcmp(arg, "COMM") == 0) {
		code->op = FETCH_OP_COMM;
		return 0;
	}

#ifdef CONFIG_HAVE_FUNCTION_ARG_ACCESS_API
	len = str_has_prefix(arg, "arg");
	if (len) {
		ret = kstrtoul(arg + len, 10, &param);
		if (ret)
			goto inval;

		if (!param || param > PARAM_MAX_STACK) {
			err = TP_ERR_BAD_ARG_NUM;
			goto inval;
		}
		param--; /* argN starts from 1, but internal arg[N] starts from 0 */

		if (tparg_is_function_entry(ctx->flags)) {
			code->op = FETCH_OP_ARG;
			if (ctx->flags & TPARG_FL_TPOINT)
				code->param++;
			else
				code->param = (unsigned int)param;
		} else if (tparg_is_function_return(ctx->flags)) {
			/* function entry argument access from return probe */
			ret = __store_entry_arg(ctx->tp, param);
			if (ret < 0)	/* This error should be an internal error */
				return ret;

			code->op = FETCH_OP_EDATA;
			code->offset = ret;
		} else {
			err = TP_ERR_NOFENTRY_ARGS;
			goto inval;
		}
		return 0;
	}
#endif

inval:
	__trace_probe_log_err(ctx->offset, err);
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
		return kstrtol(str, 0, (long *)imm);
	else if (str[0] == '+')
		return kstrtol(str + 1, 0, (long *)imm);
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

	if (str[len - 1] != '"') {
		trace_probe_log_err(offs + len, IMMSTR_NO_CLOSE);
		return -EINVAL;
	}
	*pbuf = kstrndup(str, len - 1, GFP_KERNEL);
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
	int deref = FETCH_OP_DEREF;
	long offset = 0;
	char *tmp;
	int ret = 0;

	switch (arg[0]) {
	case '$':
		ret = parse_probe_vars(arg, type, pcode, end, ctx);
		break;

	case '%':	/* named register */
		if (ctx->flags & (TPARG_FL_TEVENT | TPARG_FL_FPROBE)) {
			/* eprobe and fprobe do not handle registers */
			trace_probe_log_err(ctx->offset, BAD_VAR);
			break;
		}
		ret = regs_query_register_offset(arg + 1);
		if (ret >= 0) {
			code->op = FETCH_OP_REG;
			code->param = (unsigned int)ret;
			ret = 0;
		} else
			trace_probe_log_err(ctx->offset, BAD_REG_NAME);
		break;

	case '@':	/* memory, file-offset or symbol */
		if (isdigit(arg[1])) {
			ret = kstrtoul(arg + 1, 0, &param);
			if (ret) {
				trace_probe_log_err(ctx->offset, BAD_MEM_ADDR);
				break;
			}
			/* load address */
			code->op = FETCH_OP_IMM;
			code->immediate = param;
		} else if (arg[1] == '+') {
			/* kprobes don't support file offsets */
			if (ctx->flags & TPARG_FL_KERNEL) {
				trace_probe_log_err(ctx->offset, FILE_ON_KPROBE);
				return -EINVAL;
			}
			ret = kstrtol(arg + 2, 0, &offset);
			if (ret) {
				trace_probe_log_err(ctx->offset, BAD_FILE_OFFS);
				break;
			}

			code->op = FETCH_OP_FOFFS;
			code->immediate = (unsigned long)offset;  // imm64?
		} else {
			/* uprobes don't support symbols */
			if (!(ctx->flags & TPARG_FL_KERNEL)) {
				trace_probe_log_err(ctx->offset, SYM_ON_UPROBE);
				return -EINVAL;
			}
			/* Preserve symbol for updating */
			code->op = FETCH_NOP_SYMBOL;
			code->data = kstrdup(arg + 1, GFP_KERNEL);
			if (!code->data)
				return -ENOMEM;
			if (++code == end) {
				trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
				return -EINVAL;
			}
			code->op = FETCH_OP_IMM;
			code->immediate = 0;
		}
		/* These are fetching from memory */
		if (++code == end) {
			trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
			return -EINVAL;
		}
		*pcode = code;
		code->op = FETCH_OP_DEREF;
		code->offset = offset;
		break;

	case '+':	/* deref memory */
	case '-':
		if (arg[1] == 'u') {
			deref = FETCH_OP_UDEREF;
			arg[1] = arg[0];
			arg++;
		}
		if (arg[0] == '+')
			arg++;	/* Skip '+', because kstrtol() rejects it. */
		tmp = strchr(arg, '(');
		if (!tmp) {
			trace_probe_log_err(ctx->offset, DEREF_NEED_BRACE);
			return -EINVAL;
		}
		*tmp = '\0';
		ret = kstrtol(arg, 0, &offset);
		if (ret) {
			trace_probe_log_err(ctx->offset, BAD_DEREF_OFFS);
			break;
		}
		ctx->offset += (tmp + 1 - arg) + (arg[0] != '-' ? 1 : 0);
		arg = tmp + 1;
		tmp = strrchr(arg, ')');
		if (!tmp) {
			trace_probe_log_err(ctx->offset + strlen(arg),
					    DEREF_OPEN_BRACE);
			return -EINVAL;
		} else {
			const struct fetch_type *t2 = find_fetch_type(NULL, ctx->flags);
			int cur_offs = ctx->offset;

			*tmp = '\0';
			ret = parse_probe_arg(arg, t2, &code, end, ctx);
			if (ret)
				break;
			ctx->offset = cur_offs;
			if (code->op == FETCH_OP_COMM ||
			    code->op == FETCH_OP_DATA) {
				trace_probe_log_err(ctx->offset, COMM_CANT_DEREF);
				return -EINVAL;
			}
			if (++code == end) {
				trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
				return -EINVAL;
			}
			*pcode = code;

			code->op = deref;
			code->offset = offset;
			/* Reset the last type if used */
			ctx->last_type = NULL;
		}
		break;
	case '\\':	/* Immediate value */
		if (arg[1] == '"') {	/* Immediate string */
			ret = __parse_imm_string(arg + 2, &tmp, ctx->offset + 2);
			if (ret)
				break;
			code->op = FETCH_OP_DATA;
			code->data = tmp;
		} else {
			ret = str_to_immediate(arg + 1, &code->immediate);
			if (ret)
				trace_probe_log_err(ctx->offset + 1, BAD_IMM);
			else
				code->op = FETCH_OP_IMM;
		}
		break;
	default:
		if (isalpha(arg[0]) || arg[0] == '_') {	/* BTF variable */
			if (!tparg_is_function_entry(ctx->flags) &&
			    !tparg_is_function_return(ctx->flags)) {
				trace_probe_log_err(ctx->offset, NOSUP_BTFARG);
				return -EINVAL;
			}
			ret = parse_btf_arg(arg, pcode, end, ctx);
			break;
		}
	}
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

	if (*bf != 'b')
		return 0;

	bw = simple_strtoul(bf + 1, &tail, 0);	/* Use simple one */

	if (bw == 0 || *tail != '@')
		return -EINVAL;

	bf = tail + 1;
	bo = simple_strtoul(bf, &tail, 0);

	if (tail == bf || *tail != '/')
		return -EINVAL;
	code++;
	if (code->op != FETCH_OP_NOP)
		return -EINVAL;
	*pcode = code;

	code->op = FETCH_OP_MOD_BF;
	code->lshift = BYTES_TO_BITS(t->size) - (bw + bo);
	code->rshift = BYTES_TO_BITS(t->size) - bw;
	code->basesize = t->size;

	return (BYTES_TO_BITS(t->size) < (bw + bo)) ? -EINVAL : 0;
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

	t = strchr(arg, ':');
	if (t) {
		*t++ = '\0';
		t2 = strchr(t, '[');
		if (t2) {
			*t2++ = '\0';
			t3 = strchr(t2, ']');
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
			*t3 = '\0';
			if (kstrtouint(t2, 0, &parg->count) || !parg->count) {
				trace_probe_log_err(ctx->offset + t2 - arg,
						    BAD_ARRAY_NUM);
				return ERR_PTR(-EINVAL);
			}
			if (parg->count > MAX_ARRAY_LEN) {
				trace_probe_log_err(ctx->offset + t2 - arg,
						    ARRAY_TOO_BIG);
				return ERR_PTR(-EINVAL);
			}
		}
	}
	offs = t ? t - arg : 0;

	/*
	 * Since $comm and immediate string can not be dereferenced,
	 * we can find those by strcmp. But ignore for eprobes.
	 */
	if (!(ctx->flags & TPARG_FL_TEVENT) &&
	    (strcmp(arg, "$comm") == 0 || strcmp(arg, "$COMM") == 0 ||
	     strncmp(arg, "\\\"", 2) == 0)) {
		/* The type of $comm must be "string", and not an array type. */
		if (parg->count || (t && strcmp(t, "string"))) {
			trace_probe_log_err(ctx->offset + offs, NEED_STRING_TYPE);
			return ERR_PTR(-EINVAL);
		}
		parg->type = find_fetch_type("string", ctx->flags);
	} else
		parg->type = find_fetch_type(t, ctx->flags);

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
	if (parg->type->is_string) {
		/* Check bad combination of the type and the last fetch_insn. */
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
			if (code->op != FETCH_OP_NOP) {
				trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
				return -EINVAL;
			}
		}

		/* If op == DEREF, replace it with STRING */
		if (!strcmp(parg->type->name, "ustring") ||
		    code->op == FETCH_OP_UDEREF)
			code->op = FETCH_OP_ST_USTRING;
		else if (!strcmp(parg->type->name, "symstr"))
			code->op = FETCH_OP_ST_SYMSTR;
		else
			code->op = FETCH_OP_ST_STRING;
		code->size = parg->type->size;
		parg->dynamic = true;
	} else if (code->op == FETCH_OP_DEREF) {
		code->op = FETCH_OP_ST_MEM;
		code->size = parg->type->size;
	} else if (code->op == FETCH_OP_UDEREF) {
		code->op = FETCH_OP_ST_UMEM;
		code->size = parg->type->size;
	} else {
		code++;
		if (code->op != FETCH_OP_NOP) {
			trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
			return -E2BIG;
		}
		code->op = FETCH_OP_ST_RAW;
		code->size = parg->type->size;
	}

	/* Save storing fetch_insn. */
	scode = code;

	/* Modify operation */
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
		ret = parse_btf_bitfield(&code, ctx);
		if (ret)
			return ret;
	}

	/* Loop(Array) operation */
	if (parg->count) {
		if (scode->op != FETCH_OP_ST_MEM &&
		    scode->op != FETCH_OP_ST_STRING &&
		    scode->op != FETCH_OP_ST_USTRING) {
			trace_probe_log_err(ctx->offset + type_offset, BAD_STRING);
			return -EINVAL;
		}
		code++;
		if (code->op != FETCH_OP_NOP) {
			trace_probe_log_err(ctx->offset, TOO_MANY_OPS);
			return -E2BIG;
		}
		code->op = FETCH_OP_LP_ARRAY;
		code->param = parg->count;
	}

	/* Finalize the fetch_insn array. */
	code++;
	code->op = FETCH_OP_END;

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
	if (len > MAX_ARGSTR_LEN) {
		trace_probe_log_err(ctx->offset, ARG_TOO_LONG);
		return -E2BIG;
	} else if (len == 0) {
		trace_probe_log_err(ctx->offset, NO_ARG_BODY);
		return -EINVAL;
	}

	arg = kstrdup(argv, GFP_KERNEL);
	if (!arg)
		return -ENOMEM;

	parg->comm = kstrdup(arg, GFP_KERNEL);
	if (!parg->comm)
		return -ENOMEM;

	type = parse_probe_arg_type(arg, parg, ctx);
	if (IS_ERR(type))
		return PTR_ERR(type);

	code = tmp = kcalloc(FETCH_INSN_MAX, sizeof(*code), GFP_KERNEL);
	if (!code)
		return -ENOMEM;
	code[FETCH_INSN_MAX - 1].op = FETCH_OP_END;

	ctx->last_type = NULL;
	ret = parse_probe_arg(arg, parg->type, &code, &code[FETCH_INSN_MAX - 1],
			      ctx);
	if (ret < 0)
		goto fail;

	/* Update storing type if BTF is available */
	if (IS_ENABLED(CONFIG_PROBE_EVENTS_BTF_ARGS) &&
	    ctx->last_type) {
		if (!type) {
			parg->type = find_fetch_type_from_btf_type(ctx);
		} else if (strstr(type, "string")) {
			ret = check_prepare_btf_string_fetch(type, &code, ctx);
			if (ret)
				goto fail;
		}
	}
	parg->offset = *size;
	*size += parg->type->size * (parg->count ?: 1);

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

	ret = finalize_fetch_insn(code, parg, type, type ? type - arg : 0, ctx);
	if (ret < 0)
		goto fail;

	for (; code < tmp + FETCH_INSN_MAX; code++)
		if (code->op == FETCH_OP_END)
			break;
	/* Shrink down the code buffer */
	parg->code = kcalloc(code - tmp + 1, sizeof(*code), GFP_KERNEL);
	if (!parg->code)
		ret = -ENOMEM;
	else
		memcpy(parg->code, tmp, sizeof(*code) * (code - tmp + 1));

fail:
	if (ret < 0) {
		for (code = tmp; code < tmp + FETCH_INSN_MAX; code++)
			if (code->op == FETCH_NOP_SYMBOL ||
			    code->op == FETCH_OP_DATA)
				kfree(code->data);
	}
	kfree(tmp);

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

	for (i = 0; i < ARRAY_SIZE(reserved_field_names); i++)
		if (strcmp(reserved_field_names[i], name) == 0)
			return 1;

	for (i = 0; i < narg; i++)
		if (strcmp(args[i].name, name) == 0)
			return 1;

	return 0;
}

/**
 * @brief Parses a single probe argument string.
 * @param tp Pointer to the `trace_probe`.
 * @param i Index of the argument.
 * @param arg The argument string.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function parses an argument of the form `NAME=BODY`, extracts the name
 * and body, and then calls `traceprobe_parse_probe_arg_body` to handle the
 * body parsing.
 */
int traceprobe_parse_probe_arg(struct trace_probe *tp, int i,
			       const char *arg,
			       struct traceprobe_parse_context *ctx)
{
	char *body, *name;
	int ret;

	if (!arg)
		return -EINVAL;

	name = strchr(arg, '=');
	if (name) {
		if (name == arg) {
			trace_probe_log_err(0, NO_ARG_NAME);
			return -EINVAL;
		}
		body = name + 1;
		if (*body == '\0') {
			trace_probe_log_err(body - arg, NO_ARG_BODY);
			return -EINVAL;
		}
		*name = '\0';
		name = (char *)arg;
	} else {
		body = (char *)arg;
		name = body;
		/* find the variable name from body */
		if (*name == '$' || *name == '%')
			name++;
		else if (*name == '@' || *name == '\\')
			name = NULL;
		else {
			name = strpbrk(name, ".(+-");
			if (name)
				*name = '\0';
			name = body;
		}
	}

	if (name) {
		if (!is_good_name(name)) {
			trace_probe_log_err(0, BAD_ARG_NAME);
			return -EINVAL;
		}
		if (traceprobe_conflict_field_name(name, tp->args, i)) {
			trace_probe_log_err(0, DUPLICATE_ARG_NAME);
			return -EINVAL;
		}
		tp->args[i].name = kstrdup(name, GFP_KERNEL);
		if (!tp->args[i].name)
			return -ENOMEM;
	}

	if (!tp->args[i].name) {
		tp->args[i].name = kasprintf(GFP_KERNEL, "arg%d", i + 1);
		if (!tp->args[i].name)
			return -ENOMEM;
	}
	ctx->tp = tp;
	ctx->offset = body - arg;
	ret = traceprobe_parse_probe_arg_body(body, &tp->size, &tp->args[i], ctx);
	ctx->offset = 0;
	ctx->tp = NULL;

	return ret;
}

/**
 * @brief Helper macro for kfreeing an array of strings.
 */
static void str_array_free(char **array)
{
	int i = 0;

	if (!array)
		return;

	while (array[i])
		kfree(array[i++]);
	kfree(array);
}

/**
 * @brief Expands BTF meta-arguments (`%function(...)` or `%return(...)`).
 * @param argc Original number of arguments.
 * @param argv Original argument array.
 * @param new_argc Output parameter for the new argument count.
 * @param buf Buffer to store the expanded arguments.
 * @param len Length of the buffer.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return A newly allocated array of argument strings on success, or an `ERR_PTR` on failure.
 *
 * This function handles special meta-arguments that automatically expand
 * to all parameters of a function or its return type, using BTF information.
 */
static const char **
traceprobe_expand_meta_args(int argc, const char *argv[], int *new_argc,
				   char *buf, int len,
				   struct traceprobe_parse_context *ctx)
{
	char **__argv;
	int i, j;
	const struct btf_param *param;
	const struct btf_type *type;
	const char *arg_name, *type_name;

	*new_argc = 0;
	if (argc != 1)
		goto out;

	if (!strcmp(argv[0], "%function")) {
		/* Expand to all function parameters */
		if (query_btf_context(ctx) < 0)
			return ERR_PTR(-ENOENT);
		if (!ctx->params)
			return NULL;
		__argv = kcalloc(ctx->nr_params + 1, sizeof(char *), GFP_KERNEL);
		if (!__argv)
			return ERR_PTR(-ENOMEM);
		for (i = 0; i < ctx->nr_params; i++) {
			param = ctx->params + i;
			arg_name = btf_name_by_offset(ctx->btf, param->name_off);
			type = btf_type_skip_modifiers(ctx->btf, param->type, NULL);
			if (!arg_name || !type) {
				str_array_free(__argv);
				return ERR_PTR(-EINVAL);
			}
			/*
			 * Generate an argument string for each parameter.
			 * Note that we can not use BTF argument syntax (e.g. varname),
			 * because it does not support space in it. Use $argN syntax
			 * instead.
			 */
			type_name = fetch_type_from_btf_type(ctx->btf, type, ctx);
			if (!type_name)
				j = snprintf(buf, len, "%s=$arg%d", arg_name, i + 1);
			else
				j = snprintf(buf, len, "%s=$arg%d:%s", arg_name,
					     i + 1, type_name);
			if (j >= len) {
				str_array_free(__argv);
				return ERR_PTR(-E2BIG);
			}
			__argv[i] = kstrdup(buf, GFP_KERNEL);
			if (!__argv[i]) {
				str_array_free(__argv);
				return ERR_PTR(-ENOMEM);
			}
		}
		*new_argc = ctx->nr_params;
	} else if (!strcmp(argv[0], "%return")) {
		/* Expand to $retval */
		if (query_btf_context(ctx) < 0)
			return ERR_PTR(-ENOENT);
		/* If there is no return value, return NULL */
		if (ctx->proto->type == 0)
			return NULL;
		type = btf_type_skip_modifiers(ctx->btf, ctx->proto->type, NULL);
		type_name = fetch_type_from_btf_type(ctx->btf, type, ctx);
		if (!type_name)
			j = snprintf(buf, len, "ret=$retval");
		else
			j = snprintf(buf, len, "ret=$retval:%s", type_name);
		if (j >= len)
			return ERR_PTR(-E2BIG);
		__argv = kcalloc(2, sizeof(char *), GFP_KERNEL);
		if (!__argv)
			return ERR_PTR(-ENOMEM);
		__argv[0] = kstrdup(buf, GFP_KERNEL);
		if (!__argv[0]) {
			kfree(__argv);
			return ERR_PTR(-ENOMEM);
		}
		*new_argc = 1;
	} else
		goto out;

	return __argv;
out:
	/* if no meta args, return NULL */
	return NULL;
}

/**
 * @brief Expands `dentry` argument types (`dentry->d_name.name`).
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @param dbuf_p Output parameter for the expanded argument string buffer.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function replaces any `dentry` type arguments with `+0(%arg):string`,
 * which fetches the `d_name.name` field from the `dentry` structure.
 */
static int traceprobe_expand_dentry_args(int argc, const char **argv, char **dbuf_p)
{
	int i, j = 0, n = 0, len, diff;
	char *dbuf;

	if (argc == 0)
		return 0;

	len = argc * (MAX_ARGSTR_LEN + 1); // Functional Utility: Calculates total buffer size.
	dbuf = kmalloc(len, GFP_KERNEL); // Functional Utility: Allocates memory for the new argument strings.
	if (!dbuf)
		return -ENOMEM;
	*dbuf_p = dbuf;

	// Block Logic: Iterates through arguments to find `dentry` types.
	for (i = 0; i < argc; i++) {
		diff = strlen(argv[i]) - strlen(":dentry");
		if (diff > 0 && strcmp(argv[i] + diff, ":dentry") == 0) {
			n = i + 1;
			break;
		}
	}
	if (!n)
		return 0;

	// Block Logic: Rebuilds argument array with expanded `dentry` arguments.
	for (i = 0; i < argc; i++) {
		diff = strlen(argv[i]) - strlen(":dentry");
		if (diff > 0 && strcmp(argv[i] + diff, ":dentry") == 0) {
			/* dentry is a pointer to struct dentry. */
			len = snprintf(dbuf, MAX_ARGSTR_LEN + 1,
				       "d_name=+0(%%arg%d):string", n); // Functional Utility: Creates expanded argument string.
			argv[j++] = dbuf; // Functional Utility: Replaces original argument.
			dbuf += len + 1;
		} else {
			argv[j++] = argv[i]; // Functional Utility: Copies non-dentry arguments.
		}
	}
	argv[j] = NULL; // Functional Utility: Null-terminates new argument array.

	return 0;
}

/**
 * @brief Finalizes the parsing context for trace probes.
 * @param ctx Pointer to `traceprobe_parse_context`.
 */
void traceprobe_finish_parse(struct traceprobe_parse_context *ctx)
{
	clear_btf_context(ctx); // Functional Utility: Clears the BTF context.
}

/**
 * @brief Cleans up a `probe_arg` structure.
 * @param parg Pointer to `probe_arg` to clean up.
 *
 * This function frees the name, comment string, format string, and fetch
 * instruction code associated with a probe argument.
 */
void traceprobe_cleanup_probe_arg(struct probe_arg *parg)
{
	int i = 0;

	if (!parg)
		return;

	kfree(parg->name);
	kfree(parg->comm);
	kfree(parg->fmt);

	if (parg->code) {
		while (parg->code[i].op != FETCH_OP_END) {
			if (parg->code[i].op == FETCH_NOP_SYMBOL ||
			    parg->code[i].op == FETCH_OP_DATA)
				kfree(parg->code[i].data);
			i++;
		}
		kfree(parg->code);
	}
}

/*
 * Note: This must be called after unlinking from event->probes, because
 * another probe might be using this primary probe's data.
 */
/**
 * @brief Cleans up a `trace_probe_event` structure.
 * @param tpe Pointer to `trace_probe_event` to clean up.
 */
void trace_probe_event_cleanup(struct trace_probe_event *tpe)
{
	int i;

	for (i = 0; i < tpe->nr_probes; i++)
		traceprobe_cleanup_probe_arg(tpe->probes[i].entry_arg);
	kfree(tpe->probes);
	kfree(tpe);
}

/**
 * @brief Initializes a `trace_probe` structure.
 * @param tp Pointer to `trace_probe` to initialize.
 * @param event The event name.
 * @param group The group name.
 * @param is_return True if it's a return probe.
 * @param nargs Number of arguments.
 * @return 0 on success, or a negative errno on failure.
 */
int trace_probe_init(struct trace_probe *tp, const char *event,
		     const char *group, bool is_return, int nargs)
{
	tp->event = NULL;
	tp->name = kstrdup(event, GFP_KERNEL);
	if (!tp->name)
		return -ENOMEM;
	tp->group = kstrdup(group, GFP_KERNEL);
	if (!tp->group) {
		kfree(tp->name);
		tp->name = NULL;
		return -ENOMEM;
	}
	tp->nr_args = nargs;
	tp->size = 0;
	return 0;
}

/**
 * @brief Cleans up a `trace_probe` structure.
 * @param tp Pointer to `trace_probe` to clean up.
 *
 * This function frees the event name, group name, print format string,
 * entry arguments, and all individual arguments of the trace probe.
 */
void trace_probe_cleanup(struct trace_probe *tp)
{
	int i;

	kfree(tp->name);
	kfree(tp->group);
	kfree(tp->print_fmt);
	if (tp->entry_arg) {
		traceprobe_cleanup_probe_arg(tp->entry_arg);
		kfree(tp->entry_arg);
	}

	if (tp->args) {
		for (i = 0; i < tp->nr_args; i++)
			traceprobe_cleanup_probe_arg(&tp->args[i]);
	}
}

/*
 * Create new trace_probe_event for new event.
 */
/**
 * @brief Allocates and initializes a new `trace_probe_event`.
 * @param event_name The name of the event.
 * @param group The group name of the event.
 * @return Pointer to the new `trace_probe_event` on success, or an `ERR_PTR` on failure.
 *
 * This function allocates memory for the event, its `probes` array,
 * and sets up the list of probes.
 */
static struct trace_probe_event *
alloc_trace_probe_event(const char *event_name, const char *group)
{
	struct trace_probe_event *tpe;

	tpe = kzalloc(sizeof(*tpe), GFP_KERNEL);
	if (!tpe)
		return ERR_PTR(-ENOMEM);
	tpe->probes = kcalloc(1, sizeof(struct trace_probe_event_probe), GFP_KERNEL);
	if (!tpe->probes) {
		kfree(tpe);
		return ERR_PTR(-ENOMEM);
	}
	tpe->probes[0].call = __trace_create_class_event(event_name, group, NULL, NULL);
	if (IS_ERR(tpe->probes[0].call)) {
		kfree(tpe->probes);
		kfree(tpe);
		return ERR_CAST(tpe->probes[0].call);
	}
	tpe->probes[0].call->flags |= TRACE_EVENT_FL_DYNAMIC;
	INIT_LIST_HEAD(&tpe->probes[0].files);
	INIT_LIST_HEAD(&tpe->list);
	tpe->nr_probes = 1;

	return tpe;
}

/**
 * @brief Registers a `trace_event_call` for a trace probe.
 * @param tp Pointer to the `trace_probe` to register.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function finds or creates a `trace_probe_event`, links the
 * `trace_probe` to it, and registers the event with the ftrace subsystem.
 */
int trace_probe_register_event_call(struct trace_probe *tp)
{
	struct trace_probe_event *tpe;
	int ret = 0;

	tpe = trace_find_probe_event(tp->name, tp->group);
	if (tpe) {
		pr_debug("Event %s/%s already registered.\n",
			 tp->group, tp->name);
		return -EEXIST;
	}

	tpe = alloc_trace_probe_event(tp->name, tp->group);
	if (IS_ERR(tpe))
		return PTR_ERR(tpe);

	tpe->probes[0].call->tp = tp;
	ret = register_trace_event(&tpe->probes[0].call->event);
	if (ret) {
		trace_probe_event_cleanup(tpe);
		return ret;
	}

	tp->event = tpe;
	list_add_tail_rcu(&tp->list, &tpe->probes[0].probes);

	return 0;
}

/**
 * @brief Unregisters a `trace_event_call` for a trace probe.
 * @param tp Pointer to the `trace_probe` to unregister.
 * @return 0 on success, -EINVAL if the trace probe is not registered.
 *
 * This function unregisters the event from the ftrace subsystem,
 * and if it's the last probe for the event, frees the `trace_probe_event`.
 */
int trace_probe_unregister_event_call(struct trace_probe *tp)
{
	struct trace_probe_event *tpe;

	if (!tp->event)
		return -EINVAL;

	tpe = tp->event;

	/*
	 * We need to synchronize any callers to the event before
	 * freeing the data.
	 */
	if (trace_remove_event_call(tpe->probes[0].call)) {
		pr_warn("trace_probe: Can't remove event %s/%s. It is still in use.\n",
			tp->group, tp->name);
		return -EBUSY;
	}

	trace_probe_event_cleanup(tpe);

	return 0;
}

/**
 * @brief Appends a trace probe to an existing `trace_probe_event`.
 * @param tp The trace probe to append.
 * @param orig The existing trace probe to append to.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function links `tp` to the same `trace_probe_event` as `orig`,
 * expanding the `probes` array if necessary.
 */
int trace_probe_append(struct trace_probe *tp, struct trace_probe *orig)
{
	struct trace_probe_event *tpe = orig->event;
	struct trace_probe_event_probe *probes;
	int cnt, i;

	for (i = 0; i < tpe->nr_probes; i++) {
		if (tpe->probes[i].tp == orig)
			break;
	}
	if (WARN_ON_ONCE(i == tpe->nr_probes))
		return -ENOENT;

	/* If another probe is appended to a primary probe, expand it */
	cnt = tpe->probes[i].nr_probes + 1;
	probes = kmemdup(tpe->probes, sizeof(*probes) * cnt, GFP_KERNEL);
	if (!probes)
		return -ENOMEM;

	for (i = 0; i < tpe->nr_probes; i++) {
		if (tpe->probes[i].tp == orig)
			break;
	}
	tpe->probes[i].nr_probes++;
	kfree(tpe->probes);
	tpe->probes = probes;

	tp->event = tpe;
	list_add_tail_rcu(&tp->list, &orig->list);

	return 0;
}

/**
 * @brief Adds a `trace_event_file` to a `trace_probe_event`.
 * @param tp Pointer to the `trace_probe`.
 * @param file Pointer to the `trace_event_file`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function links a trace event file to the probe, enabling tracing
 * output to that file.
 */
int trace_probe_add_file(struct trace_probe *tp, struct trace_event_file *file)
{
	struct trace_probe_event *tpe = tp->event;
	struct event_file_link *link;
	int i;

	for (i = 0; i < tpe->nr_probes; i++) {
		if (tpe->probes[i].tp == tp)
			break;
	}
	if (WARN_ON_ONCE(i == tpe->nr_probes))
		return -ENOENT;

	link = kmalloc(sizeof(*link), GFP_KERNEL);
	if (!link)
		return -ENOMEM;
	link->file = file;
	list_add_rcu(&link->list, &tpe->probes[i].files);
	trace_probe_set_flag(tp, TP_FLAG_TRACE);

	return 0;
}

/**
 * @brief Removes a `trace_event_file` from a `trace_probe_event`.
 * @param tp Pointer to the `trace_probe`.
 * @param file Pointer to the `trace_event_file`.
 */
void trace_probe_remove_file(struct trace_probe *tp,
			      struct trace_event_file *file)
{
	struct trace_probe_event *tpe = tp->event;
	struct event_file_link *link;
	bool found = false;
	int i;

	for (i = 0; i < tpe->nr_probes; i++) {
		if (tpe->probes[i].tp == tp)
			break;
	}
	if (WARN_ON_ONCE(i == tpe->nr_probes))
		return;

	list_for_each_entry_rcu(link, &tpe->probes[i].files, list) {
		if (link->file == file) {
			list_del_rcu(&link->list);
			found = true;
			break;
		}
	}
	if (found)
		kfree_rcu(link, rcu);
}