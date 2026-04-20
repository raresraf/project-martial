#!/bin/bash
# @6fa7b3aa-bdf7-4fa2-8620-794271477d2c/tools/testing/selftests/net/ipv6_force_forwarding.sh
# @brief Validation suite for the IPv6 'force_forwarding' interface property.
#
# Functional Intent: Verifies that enabling 'force_forwarding' on a specific 
# network interface allows IPv6 packet forwarding even when global IPv6 
# forwarding is disabled. This is critical for scenarios where a node must 
# act as a router for specific links without becoming a general-purpose 
# IPv6 router.
#
# Domain: Kernel Networking, IPv6 Stack, Interface Configuration.
#
# SPDX-License-Identifier: GPL-2.0

source lib.sh

# Functional Utility: Teardown logic for the tri-namespace topology.
cleanup() {
    cleanup_ns $ns1 $ns2 $ns3
}

trap cleanup EXIT

# setup_test - Bootstraps the networking environment.
# Logic: 
# 1. Creates three namespaces: ns1 (Sender), ns2 (Router), and ns3 (Receiver).
# 2. Links them via veth pairs and assigns IPv6 ULA addresses.
# 3. Disables global IPv6 forwarding on the router namespace to establish the baseline.
setup_test() {
    setup_ns ns1 ns2 ns3

    ip link add name veth12 type veth peer name veth21
    ip link add name veth23 type veth peer name veth32

    ip link set veth12 netns $ns1
    ip link set veth21 netns $ns2
    ip link set veth23 netns $ns2
    ip link set veth32 netns $ns3

    ip -n $ns1 addr add 2001:db8:1::1/64 dev veth12 nodad
    ip -n $ns2 addr add 2001:db8:1::2/64 dev veth21 nodad
    ip -n $ns2 addr add 2001:db8:2::1/64 dev veth23 nodad
    ip -n $ns3 addr add 2001:db8:2::2/64 dev veth32 nodad

    ip -n $ns1 link set veth12 up
    ip -n $ns2 link set veth21 up
    ip -n $ns2 link set veth23 up
    ip -n $ns3 link set veth32 up

    ip -n $ns1 route add 2001:db8:2::/64 via 2001:db8:1::2
    ip -n $ns3 route add 2001:db8:1::/64 via 2001:db8:2::1

    # Invariant: Global forwarding must be OFF for the 'force' property to be meaningful.
    ip netns exec $ns2 sysctl -qw net.ipv6.conf.all.forwarding=0
}

# test_force_forwarding - Executes the validation logic.
# Logic: 
# 1. Verifies that ping fails across the router when both global and 
#    per-interface forwarding are disabled.
# 2. Enables 'force_forwarding' on the router's interfaces and verifies 
#    that ping succeeds despite the global 'off' state.
test_force_forwarding() {
    local ret=0

    echo "TEST: force_forwarding functionality"

    if ! ip netns exec $ns2 test -f /proc/sys/net/ipv6/conf/veth21/force_forwarding; then
        echo "SKIP: force_forwarding not available"
        return $ksft_skip
    fi

    # Block Logic: Baseline Verification (Expected Failure).
    ip netns exec $ns2 sysctl -qw net.ipv6.conf.veth21.force_forwarding=0
    ip netns exec $ns2 sysctl -qw net.ipv6.conf.veth23.force_forwarding=0

    if ip netns exec $ns1 ping -6 -c 1 -W 2 2001:db8:2::2 &>/dev/null; then
        echo "FAIL: ping succeeded when forwarding disabled"
        ret=1
    else
        echo "PASS: forwarding disabled correctly"
    fi

    # Block Logic: Feature Verification (Expected Success).
    ip netns exec $ns2 sysctl -qw net.ipv6.conf.veth21.force_forwarding=1
    ip netns exec $ns2 sysctl -qw net.ipv6.conf.veth23.force_forwarding=1

    if ip netns exec $ns1 ping -6 -c 1 -W 2 2001:db8:2::2 &>/dev/null; then
        echo "PASS: force_forwarding enabled forwarding"
    else
        echo "FAIL: ping failed with force_forwarding enabled"
        ret=1
    fi

    return $ret
}

echo "IPv6 force_forwarding test"
echo "=========================="

setup_test
test_force_forwarding
ret=$?

if [ $ret -eq 0 ]; then
    echo "OK"
    exit 0
elif [ $ret -eq $ksft_skip ]; then
    echo "SKIP"
    exit $ksft_skip
else
    echo "FAIL"
    exit 1
fi
