// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2022 Facebook */

/**
 * @file dynptr.c
 * @brief User-space test runner for BPF dynamic pointers (dynptr).
 *
 * This file is the user-space component of the BPF selftest suite for the
 * 'dynptr' feature. It is responsible for orchestrating a series of tests
 * that validate the functionality of dynamic pointers in various BPF program
 * contexts. It works in conjunction with BPF programs defined in external
 * files (dynptr_success.bpf.c, dynptr_fail.bpf.c), which are compiled into
 * BPF skeletons.
 *
 * The tests cover creating dynptrs, reading/writing data, passing them to
 * helpers, and ensuring the verifier correctly accepts valid usage and rejects
 * invalid usage.
 */

#include <test_progs.h>
#include <network_helpers.h>
#include "dynptr_fail.skel.h"
#include "dynptr_success.skel.h"

/* Defines the different BPF program contexts in which dynptrs are tested. */
enum test_setup_type {
	SETUP_SYSCALL_SLEEP, /* Attach to a syscall tracepoint */
	SETUP_SKB_PROG,      /* Run as a networking program on an skb */
	SETUP_SKB_PROG_TP,   /* Run as a tracepoint on skb free */
	SETUP_XDP_PROG,      /* Run as an XDP program */
};

/**
 * @var success_tests
 * @brief An array defining the suite of tests that are expected to pass.
 *
 * Each entry maps a specific BPF program name from the 'dynptr_success' skeleton
 * to a test setup type, which determines how the BPF program will be attached
 * and triggered.
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

#define PAGE_SIZE_64K 65536

/**
 * verify_success - Executes a single successful-path test case.
 * @prog_name: The name of the BPF program to test.
 * @setup_type: The context in which to run the test.
 *
 * This function handles the full lifecycle of a single test:
 * 1. Opens and loads the BPF skeleton.
 * 2. Sets up any required user-space data to be passed to the BPF program.
 * 3. Attaches and triggers the BPF program based on the setup type.
 * 4. Verifies that the BPF program completed without setting an error flag.
 * 5. Cleans up all resources.
 */
static void verify_success(const char *prog_name, enum test_setup_type setup_type)
{
	char user_data[384] = {[0 ... 382] = 'a', '\0'};
	struct dynptr_success *skel;
	struct bpf_program *prog;
	struct bpf_link *link;
	int err;

	skel = dynptr_success__open();
	if (!ASSERT_OK_PTR(skel, "dynptr_success__open"))
		return;

	skel->bss->pid = getpid();

	prog = bpf_object__find_program_by_name(skel->obj, prog_name);
	if (!ASSERT_OK_PTR(prog, "bpf_object__find_program_by_name"))
		goto cleanup;

	bpf_program__set_autoload(prog, true);

	err = dynptr_success__load(skel);
	if (!ASSERT_OK(err, "dynptr_success__load"))
		goto cleanup;

	skel->bss->user_ptr = user_data;
	skel->data->test_len[0] = sizeof(user_data);
	memcpy(skel->bss->expected_str, user_data, sizeof(user_data));

	/* Attach and trigger the BPF program based on its type */
	switch (setup_type) {
	case SETUP_SYSCALL_SLEEP:
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

		/* Just use its test_run to trigger kfree_skb tracepoint */
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

		link = bpf_program__attach(prog);
		if (!ASSERT_OK_PTR(link, "bpf_program__attach"))
			goto cleanup;

		err = bpf_prog_test_run_opts(aux_prog_fd, &topts);
		bpf_link__destroy(link);

		if (!ASSERT_OK(err, "test_run"))
			goto cleanup;

		break;
	}
	case SETUP_XDP_PROG:
	{
		char data[90000];
		int err, prog_fd;
		LIBBPF_OPTS(bpf_test_run_opts, opts,
			    .data_in = &data,
			    .repeat = 1,
		);

		if (getpagesize() == PAGE_SIZE_64K)
			opts.data_size_in = sizeof(data);
		else
			opts.data_size_in = 5000;

		prog_fd = bpf_program__fd(prog);
		err = bpf_prog_test_run_opts(prog_fd, &opts);

		if (!ASSERT_OK(err, "test_run"))
			goto cleanup;

		break;
	}
	}

	/* Verify BPF program ran and reported no errors */
	ASSERT_EQ(skel->bss->err, 0, "err");

cleanup:
	dynptr_success__destroy(skel);
}

/**
 * test_dynptr - The main entry point for the dynptr selftest suite.
 *
 * This function iterates through all defined success cases and runs them.
 * It also invokes the RUN_TESTS macro to execute the suite of tests that are
 * expected to fail, ensuring the BPF verifier correctly rejects invalid programs.
 */
void test_dynptr(void)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(success_tests); i++) {
		if (!test__start_subtest(success_tests[i].prog_name))
			continue;

		verify_success(success_tests[i].prog_name, success_tests[i].type);
	}

	RUN_TESTS(dynptr_fail);
}