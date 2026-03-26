/**
 * @file dynptr.c
 * @brief Test suite for BPF dynamic pointers (dynptr).
 * @description This file contains a series of self-tests for the BPF `dynptr` feature.
 * It verifies the correct behavior of dynptrs in various contexts, including reading,
 * writing, and manipulating data from user space, SKBs (socket buffers), and XDP
 * (Xpress Data Path). It tests both success and failure scenarios.
 * The tests are structured to load and run BPF programs defined in external skeleton
 * files (`dynptr_success.skel.h` and `dynptr_fail.skel.h`).
 */

// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2022 Facebook */

#include <test_progs.h>
#include <network_helpers.h>
#include "dynptr_fail.skel.h"
#include "dynptr_success.skel.h"

/**
 * @enum test_setup_type
 * @brief Defines the execution context for a BPF dynptr test.
 * @description This enumeration is used to select the appropriate environment for
 * running a specific BPF program test, such as a simple syscall context,
 * a network packet context (SKB or XDP), or a tracepoint.
 */
enum test_setup_type {
	/**
	 * @brief A simple syscall context. The BPF program is attached and triggered,
	 * then the test sleeps briefly to allow it to execute. Useful for basic
	 * memory manipulation and logic tests.
	 */
	SETUP_SYSCALL_SLEEP,
	/**
	 * @brief A network packet (sk_buff) processing context.
	 * The BPF program is triggered via `bpf_prog_test_run_opts` on a dummy packet.
	 */
	SETUP_SKB_PROG,
	/**
	 * @brief A tracepoint context triggered by a network packet event.
	 * Specifically used for testing dynptr on kfree_skb tracepoint.
	 */
	SETUP_SKB_PROG_TP,
	/**
	 * @brief An XDP (Xpress Data Path) context.
	 * The BPF program is run as if it were an XDP program handling an incoming packet.
	 */
	SETUP_XDP_PROG,
};

/**
 * @var success_tests
 * @brief An array defining the suite of successful-case BPF dynptr tests.
 * @description Each entry specifies the name of the BPF program to run (which corresponds
 * to a function in the BPF source) and the context type required to set up the test.
 */
static struct {
	const char *prog_name;
	enum test_setup_type type;
} success_tests[] = {
	{"test_read_write", SETUP_SYSCALL_SLEEP},
	{"test_dynptr_data", SETUP_SYSCALL_SLEEP},
	{"test_dynptr_copy", SETUP_SYSCALL_SLEEP},
	{"test_dynptr_copy_xdp", SETUP_XDP_PROG},
	{"test_dynptr_memset_zero", SETUP_SYSCALL_SLEEP},
	{"test_dynptr_memset_notzero", SETUP_SYSCALL_SLEEP},
	{"test_dynptr_memset_zero_offset", SETUP_SYSCALL_SLEEP},
	{"test_dynptr_memset_zero_adjusted", SETUP_SYSCALL_SLEEP},
	{"test_dynptr_memset_overflow", SETUP_SYSCALL_SLEEP},
	{"test_dynptr_memset_overflow_offset", SETUP_SYSCALL_SLEEP},
	{"test_dynptr_memset_readonly", SETUP_SKB_PROG},
	{"test_dynptr_memset_xdp_chunks", SETUP_XDP_PROG},
	{"test_ringbuf", SETUP_SYSCALL_SLEEP},
	{"test_skb_readonly", SETUP_SKB_PROG},
	{"test_dynptr_skb_data", SETUP_SKB_PROG},
	{"test_adjust", SETUP_SYSCALL_SLEEP},
	{"test_adjust_err", SETUP_SYSCALL_SLEEP},
	{"test_zero_size_dynptr", SETUP_SYSCALL_SLEEP},
	{"test_dynptr_is_null", SETUP_SYSCALL_SLEEP},
	{"test_dynptr_is_rdonly", SETUP_SKB_PROG},
	{"test_dynptr_clone", SETUP_SKB_PROG},
	{"test_dynptr_skb_no_buff", SETUP_SKB_PROG},
	{"test_dynptr_skb_strcmp", SETUP_SKB_PROG},
	{"test_dynptr_skb_tp_btf", SETUP_SKB_PROG_TP},
	{"test_probe_read_user_dynptr", SETUP_XDP_PROG},
	{"test_probe_read_kernel_dynptr", SETUP_XDP_PROG},
	{"test_probe_read_user_str_dynptr", SETUP_XDP_PROG},
	{"test_probe_read_kernel_str_dynptr", SETUP_XDP_PROG},
	{"test_copy_from_user_dynptr", SETUP_SYSCALL_SLEEP},
	{"test_copy_from_user_str_dynptr", SETUP_SYSCALL_SLEEP},
	{"test_copy_from_user_task_dynptr", SETUP_SYSCALL_SLEEP},
	{"test_copy_from_user_task_str_dynptr", SETUP_SYSCALL_SLEEP},
};

/**
 * @brief Executes a single BPF dynptr success-case test.
 * @param prog_name The name of the BPF program to execute.
 * @param setup_type The context in which to run the test.
 * @description This function orchestrates the setup, execution, and validation of a
 * single BPF test. It opens the BPF object skeleton, loads the specified program,
 * sets up the necessary context (e.g., attaches to a syscall or prepares a test packet),
 * triggers the BPF program, and finally asserts that no errors occurred in the BPF-side
 * logic.
 */
static void verify_success(const char *prog_name, enum test_setup_type setup_type)
{
	char user_data[384] = {[0 ... 382] = 'a', '\0'};
	struct dynptr_success *skel;
	struct bpf_program *prog;
	struct bpf_link *link;
	int err;

	/* Functional Utility: Opens the BPF object file and prepares it for loading. */
	skel = dynptr_success__open();
	if (!ASSERT_OK_PTR(skel, "dynptr_success__open"))
		return;

	/* Pre-condition: Set the process ID for tests that need to reference the current task. */
	skel->bss->pid = getpid();

	/* Block Logic: Find the specific BPF program within the object file by its name. */
	prog = bpf_object__find_program_by_name(skel->obj, prog_name);
	if (!ASSERT_OK_PTR(prog, "bpf_object__find_program_by_name"))
		goto cleanup;

	/* Functional Utility: Mark the program for auto-loading. */
	bpf_program__set_autoload(prog, true);

	/* Block Logic: Load the BPF program and maps into the kernel. */
	err = dynptr_success__load(skel);
	if (!ASSERT_OK(err, "dynptr_success__load"))
		goto cleanup;

	/* Pre-condition: Prepare user-space data that the BPF program will interact with. */
	skel->bss->user_ptr = user_data;
	skel->data->test_len[0] = sizeof(user_data);
	memcpy(skel->bss->expected_str, user_data, sizeof(user_data));

	/* Block Logic: Set up the execution environment based on the test type and trigger the BPF program. */
	switch (setup_type) {
	case SETUP_SYSCALL_SLEEP:
		/* Attach to a syscall and sleep, letting the BPF program run. */
		link = bpf_program__attach(prog);
		if (!ASSERT_OK_PTR(link, "bpf_program__attach"))
			goto cleanup;

		usleep(1);

		bpf_link__destroy(link);
		break;
	case SETUP_SKB_PROG:
	{
		int prog_fd;
		char buf[64];

		/* Invariant: Prepares a dummy network packet and options for the test run. */
		LIBBPF_OPTS(bpf_test_run_opts, topts,
			    .data_in = &pkt_v4,
			    .data_size_in = sizeof(pkt_v4),
			    .data_out = buf,
			    .data_size_out = sizeof(buf),
			    .repeat = 1,
		);

		prog_fd = bpf_program__fd(prog);
		if (!ASSERT_GE(prog_fd, 0, "prog_fd"))
			goto cleanup;

		/* Trigger the BPF program with the test packet. */
		err = bpf_prog_test_run_opts(prog_fd, &topts);

		if (!ASSERT_OK(err, "test_run"))
			goto cleanup;

		break;
	}
	case SETUP_SKB_PROG_TP:
	{
		struct __sk_buff skb = {};
		struct bpf_object *obj;
		int aux_prog_fd;

		/* Functional Utility: Load an auxiliary BPF program whose sole purpose is to trigger the kfree_skb tracepoint. */
		err = bpf_prog_test_load("./test_pkt_access.bpf.o", BPF_PROG_TYPE_SCHED_CLS,
					 &obj, &aux_prog_fd);
		if (!ASSERT_OK(err, "prog_load sched cls"))
			goto cleanup;

		LIBBPF_OPTS(bpf_test_run_opts, topts,
			    .data_in = &pkt_v4,
			    .data_size_in = sizeof(pkt_v4),
			    .ctx_in = &skb,
			    .ctx_size_in = sizeof(skb),
		);

		/* Attach the main test program to the tracepoint. */
		link = bpf_program__attach(prog);
		if (!ASSERT_OK_PTR(link, "bpf_program__attach"))
			goto cleanup;

		/* Run the auxiliary program to trigger the tracepoint. */
		err = bpf_prog_test_run_opts(aux_prog_fd, &topts);
		bpf_link__destroy(link);

		if (!ASSERT_OK(err, "test_run"))
			goto cleanup;

		break;
	}
	case SETUP_XDP_PROG:
	{
		char data[5000];
		int err, prog_fd;
		/* Invariant: Prepare dummy packet data for the XDP test run. */
		LIBBPF_OPTS(bpf_test_run_opts, opts,
			    .data_in = &data,
			    .data_size_in = sizeof(data),
			    .repeat = 1,
		);

		prog_fd = bpf_program__fd(prog);
		/* Trigger the BPF program in an XDP context. */
		err = bpf_prog_test_run_opts(prog_fd, &opts);

		if (!ASSERT_OK(err, "test_run"))
			goto cleanup;

		break;
	}
	}

	/* Invariant: After execution, check the BPF map for a success code (0). */
	ASSERT_EQ(skel->bss->err, 0, "err");

cleanup:
	/* Functional Utility: Frees all resources associated with the BPF object. */
	dynptr_success__destroy(skel);
}

/**
 * @brief Main entry point for the BPF dynptr test suite.
 * @description This function iterates through all defined dynptr tests. It runs the
 * suite of success-case tests defined in the `success_tests` array and then
 * executes the failure-case tests defined in the `dynptr_fail` BPF object.
 */
void test_dynptr(void)
{
	int i;

	/* Block Logic: Iterate through and execute all success-case tests. */
	for (i = 0; i < ARRAY_SIZE(success_tests); i++) {
		if (!test__start_subtest(success_tests[i].prog_name))
			continue;

		verify_success(success_tests[i].prog_name, success_tests[i].type);
	}

	/* Functional Utility: Run all failure-case tests defined in the dynptr_fail skeleton. */
	RUN_TESTS(dynptr_fail);
}
