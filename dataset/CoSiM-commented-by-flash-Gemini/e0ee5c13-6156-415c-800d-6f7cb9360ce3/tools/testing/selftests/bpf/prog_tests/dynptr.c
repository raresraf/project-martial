/**
 * @e0ee5c13-6156-415c-800d-6f7cb9360ce3/tools/testing/selftests/bpf/prog_tests/dynptr.c
 * @brief User-space test harness for validating BPF dynamic pointer (dynptr) functionality.
 * * Domain: Linux Kernel BPF Selftests.
 * * Functional Utility: Orchestrates the loading, attachment, and execution of BPF programs 
 *   that exercise various dynptr APIs (read, write, copy, memset, etc.) across different 
 *   program types (XDP, SKB, Syscall).
 */

// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2022 Facebook */

#include <test_progs.h>
#include <network_helpers.h>
#include "dynptr_fail.skel.h"
#include "dynptr_success.skel.h"

/**
 * @enum test_setup_type
 * @brief Defines the execution environment required for a specific BPF dynptr test.
 */
enum test_setup_type {
	SETUP_SYSCALL_SLEEP, // Intent: Run via syscall attachment and usleep trigger.
	SETUP_SKB_PROG,      // Intent: Run via BPF_PROG_TEST_RUN with socket buffer context.
	SETUP_SKB_PROG_TP,   // Intent: Run via tracepoint attachment triggered by dummy network activity.
	SETUP_XDP_PROG,      // Intent: Run via BPF_PROG_TEST_RUN with XDP packet context.
};

/**
 * @struct success_tests
 * @brief Registry of dynptr functional tests and their associated setup environments.
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
 * @brief Executes a single BPF dynptr "success" test case.
 * Logic: Loads the BPF skeleton, identifies the target program, configures its input data,
 *        and triggers execution based on the specified setup type.
 */
static void verify_success(const char *prog_name, enum test_setup_type setup_type)
{
	char user_data[384] = {[0 ... 382] = 'a', '\0'};
	struct dynptr_success *skel;
	struct bpf_program *prog;
	struct bpf_link *link;
	int err;

	// Logic: Skeleton lifecycle management - Open.
	skel = dynptr_success__open();
	if (!ASSERT_OK_PTR(skel, "dynptr_success__open"))
		return;

	skel->bss->pid = getpid();

	prog = bpf_object__find_program_by_name(skel->obj, prog_name);
	if (!ASSERT_OK_PTR(prog, "bpf_object__find_program_by_name"))
		goto cleanup;

	bpf_program__set_autoload(prog, true);

	// Logic: Skeleton lifecycle management - Load.
	err = dynptr_success__load(skel);
	if (!ASSERT_OK(err, "dynptr_success__load"))
		goto cleanup;

	// Block Logic: Input data preparation.
	// Functional Utility: Populates BPF maps with pointers and lengths used by probe/copy helpers.
	skel->bss->user_ptr = user_data;
	skel->data->test_len[0] = sizeof(user_data);
	memcpy(skel->bss->expected_str, user_data, sizeof(user_data));

	/**
	 * Block Logic: Triggering Mechanism.
	 * Dispatches based on the setup_type to invoke the BPF program in the correct kernel context.
	 */
	switch (setup_type) {
	case SETUP_SYSCALL_SLEEP:
		// Logic: Triggers BPF programs attached to syscalls (e.g., sys_enter_nanosleep).
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

		// Logic: Simulates a network packet processing environment for __sk_buff based programs.
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

		// Logic: Indirect trigger - uses a secondary program to induce tracepoint events.
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
		char data[5000];
		int err, prog_fd;
		// Logic: Simulates XDP processing with packet context.
		LIBBPF_OPTS(bpf_test_run_opts, opts,
			    .data_in = &data,
			    .data_size_in = sizeof(data),
			    .repeat = 1,
		);

		prog_fd = bpf_program__fd(prog);
		err = bpf_prog_test_run_opts(prog_fd, &opts);

		if (!ASSERT_OK(err, "test_run"))
			goto cleanup;

		break;
	}
	}

	// Invariant: The BPF program must not set the internal error flag in its .bss section.
	ASSERT_EQ(skel->bss->err, 0, "err");

cleanup:
	dynptr_success__destroy(skel);
}

/**
 * @brief Entry point for the dynptr selftest suite.
 * Logic: Iterates through all registered functional tests and subsequently runs the negative (failure) test suite.
 */
void test_dynptr(void)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(success_tests); i++) {
		if (!test__start_subtest(success_tests[i].prog_name))
			continue;

		verify_success(success_tests[i].prog_name, success_tests[i].type);
	}

	// Logic: Executes tests that are expected to fail during verification (e.g., out-of-bounds access).
	RUN_TESTS(dynptr_fail);
}
