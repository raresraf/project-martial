#!/bin/bash
# @cc911f60-0e8f-4a0b-9ccb-2ca193af3003/tools/testing/selftests/net/netfilter/conntrack_clash.sh
# @brief Validation suite for netfilter conntrack clash resolution logic.
#
# Functional Intent: Verifies the kernel's ability to resolve race conditions 
# (clashes) during simultaneous connection tracking insertion. It simulates 
# high-concurrency UDP traffic across network namespaces with and without NAT 
# and validates that the 'clash_resolve' mechanism correctly handles duplicate 
# flow entries.
#
# Domain: Kernel Networking, Netfilter, Conntrack, Race Condition Testing.
#
# SPDX-License-Identifier: GPL-2.0

source lib.sh

clash_resolution_active=0
dport=22111
ret=0

# Functional Utility: Teardown logic for network namespaces and background servers.
cleanup()
{
	# netns cleanup also zaps any remaining socat echo server.
	cleanup_all_ns
}

# Block Logic: Toolchain validation.
# Pre-condition: nftables, conntrack-tools, and socat must be installed and in PATH.
checktool "nft --version" "run test without nft"
checktool "conntrack --version" "run test without conntrack"
checktool "socat -h" "run test without socat"

trap cleanup EXIT

# Block Logic: Topology and NAT configuration.
# Logic: Creates two clients and one router namespace. Configures the router 
# with a DNAT load-balancing rule that randomly maps traffic to three UDP ports, 
# intentionally creating potential tracking collisions.
setup_ns nsclient1 nsclient2 nsrouter

ip netns exec "$nsrouter" nft -f -<<EOF
table ip t {
	chain lb {
		meta l4proto udp dnat to numgen random mod 3 map { 0 : 10.0.2.1 . 9000, 1 : 10.0.2.1 . 9001, 2 : 10.0.2.1 . 9002 }
	}

	chain prerouting {
		type nat hook prerouting priority dstnat

		udp dport $dport counter jump lb
	}

	chain output {
		type nat hook output priority dstnat

		udp dport $dport counter jump lb
	}
}
EOF

# Functional Utility: Injects a simple filter rule to trigger conntrack state machine.
load_simple_ruleset()
{
ip netns exec "$1" nft -f -<<EOF
table ip t {
	chain forward {
		type filter hook forward priority 0

		ct state new counter
	}
}
EOF
}

# Block Logic: Server instantiation.
# Logic: Spawns multiple socat UDP listeners to act as DNAT targets.
spawn_servers()
{
	local ns="$1"
	local ports="9000 9001 9002"

	for port in $ports; do
		ip netns exec "$ns" socat UDP-RECVFROM:$port,fork PIPE 2>/dev/null &
	done

	for port in $ports; do
		wait_local_port_listen "$ns" $port udp
	done
}

# Functional Utility: Configures network interface addresses and brings links up.
add_addr()
{
	local ns="$1"
	local dev="$2"
	local i="$3"
	local j="$4"

	ip -net "$ns" link set "$dev" up
	ip -net "$ns" addr add "10.0.$i.$j/24" dev "$dev"
}

# Functional Utility: Basic connectivity verification.
ping_test()
{
	local ns="$1"
	local daddr="$2"

	if ! ip netns exec "$ns" ping -q -c 1 $daddr > /dev/null;then
		echo "FAIL: ping from $ns to $daddr"
		exit 1
	fi
}

# run_one_clash_test - Executes a single high-concurrency UDP burst.
# Logic: Uses the 'udpclash' helper to flood the target with simultaneous packets. 
# Inspects conntrack statistics (clash_resolve counter) to verify if the 
# kernel's collision resolution path was triggered.
run_one_clash_test()
{
	local ns="$1"
	local daddr="$2"
	local dport="$3"
	local entries
	local cre

	if ! ip netns exec "$ns" ./udpclash $daddr $dport;then
		echo "FAIL: did not receive expected number of replies for $daddr:$dport"
		ret=1
		return 1
	fi

	entries=$(conntrack -S | wc -l)
	cre=$(conntrack -S | grep -v "clash_resolve=0" | wc -l)

	# Invariant: If cre != entries, it means at least one CPU saw a resolution event.
	if [ "$cre" -ne "$entries" ] ;then
		clash_resolution_active=1
		return 0
	fi

	# 1 cpu -> parallel insertion impossible
	if [ "$entries" -eq 1 ]; then
		return 0
	fi

	# skip if timing didn't allow for a clash.
	return $ksft_skip
}

# run_clash_test - Retries the burst test to overcome timing variations.
run_clash_test()
{
	local ns="$1"
	local daddr="$2"
	local dport="$3"

	for i in $(seq 1 10);do
		run_one_clash_test "$ns" "$daddr" "$dport"
		local rv=$?
		if [ $rv -eq 0 ];then
			echo "PASS: clash resolution test for $daddr:$dport on attempt $i"
			return 0
		elif [ $rv -eq 1 ];then
			echo "FAIL: clash resolution test for $daddr:$dport on attempt $i"
			return 1
		fi
	done
}

# Block Logic: Infrastructure Initialization.
# Logic: Links namespaces via veth pairs and establishes routing.
ip link add veth0 netns "$nsclient1" type veth peer name veth0 netns "$nsrouter"
ip link add veth0 netns "$nsclient2" type veth peer name veth1 netns "$nsrouter"
add_addr "$nsclient1" veth0 1 1
add_addr "$nsclient2" veth0 2 1
add_addr "$nsrouter" veth0 1 99
add_addr "$nsrouter" veth1 2 99

ip -net "$nsclient1" route add default via 10.0.1.99
ip -net "$nsclient2" route add default via 10.0.2.99
ip netns exec "$nsrouter" sysctl -q net.ipv4.ip_forward=1

ping_test "$nsclient1" 10.0.1.99
ping_test "$nsclient1" 10.0.2.1
ping_test "$nsclient2" 10.0.1.1

spawn_servers "$nsclient2"

# Block Logic: Test Scenarios.
# 1. Exercise clash resolution with NAT (Router DNAT).
run_clash_test "$nsclient1" 10.0.1.99 "$dport"

# 2. Exercise clash resolution without NAT (Local traffic).
load_simple_ruleset "$nsclient2"
run_clash_test "$nsclient2" 127.0.0.1 9001

# Final Validation: Reports if the clash logic was actually exercised.
if [ $clash_resolution_active -eq 0 ];then
	[ "$ret" -eq 0 ] && ret=$ksft_skip
	echo "SKIP: Clash resolution did not trigger"
fi

exit $ret
