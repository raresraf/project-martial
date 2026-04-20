# SPDX-License-Identifier: GPL-2.0
# @cc911f60-0e8f-4a0b-9ccb-2ca193af3003/tools/testing/selftests/net/mptcp/Makefile
# @brief Build configuration for MultiPath TCP (MPTCP) kernel selftests.
#
# Functional Intent: Orchestrates the compilation of binary helpers and the 
# execution of shell-based test suites for MPTCP. It defines the set of 
# integration tests covering connection establishment, path management via 
# netlink, socket options, and diagnostic interfaces.
#
# Domain: Kernel Networking, MPTCP Protocol, Build System.

top_srcdir = ../../../../..

# Functional Utility: Compilation flags for networking test utilities.
# Includes UAPI headers from the kernel tree to ensure alignment with the tested subsystem.
CFLAGS += -Wall -Wl,--no-as-needed -O2 -g -I$(top_srcdir)/usr/include $(KHDR_INCLUDES)

# Block Logic: Definition of test artifacts.
# TEST_PROGS: High-level integration scripts triggered by the kselftest runner.
# TEST_GEN_FILES: Compiled C binaries used as path management or connection helpers.
TEST_PROGS := mptcp_connect.sh pm_netlink.sh mptcp_join.sh diag.sh \
	      simult_flows.sh mptcp_sockopt.sh userspace_pm.sh

TEST_GEN_FILES = mptcp_connect pm_nl_ctl mptcp_sockopt mptcp_inq mptcp_diag

TEST_FILES := mptcp_lib.sh settings

TEST_INCLUDES := ../lib.sh $(wildcard ../lib/sh/*.sh)

# Functional Utility: Cleanup rules for packet capture artifacts generated during test runs.
EXTRA_CLEAN := *.pcap

# Block Logic: Integration with the global kselftest framework.
include ../../lib.mk
