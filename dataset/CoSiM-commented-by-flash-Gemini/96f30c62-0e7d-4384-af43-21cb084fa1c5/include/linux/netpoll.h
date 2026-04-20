/* SPDX-License-Identifier: GPL-2.0 */
/**
 * @96f30c62-0e7d-4384-af43-21cb084fa1c5/include/linux/netpoll.h
 * @brief Low-level network polling API for critical system communication.
 * 
 * Functional Intent: Facilitates robust network I/O (primarily UDP) in contexts 
 * where the standard networking stack is unavailable or unsafe (e.g., kernel 
 * panics, early boot, or kgdb-over-ethernet). It bypasses standard interrupt-driven 
 * processing by manually polling network devices, ensuring diagnostic data (netconsole) 
 * can be transmitted even when the scheduler or IRQ handlers are compromised.
 * 
 * Domain: Kernel Infrastructure, Remote Debugging, RAS (Reliability, Availability, Serviceability).
 */

#ifndef _LINUX_NETPOLL_H
#define _LINUX_NETPOLL_H

#include <linux/netdevice.h>
#include <linux/interrupt.h>
#include <linux/rcupdate.h>
#include <linux/list.h>
#include <linux/refcount.h>

/**
 * union inet_addr - Family-agnostic IP address container.
 * Logic: Provides a polymorphic view of network addresses, allowing the netpoll 
 * core to handle IPv4 and IPv6 without redundant structure branches.
 */
union inet_addr {
	__u32		all[4];
	__be32		ip;
	__be32		ip6[4];
	struct in_addr	in;
	struct in6_addr	in6;
};

/**
 * struct netpoll - Configuration and context for a polling network instance.
 * @dev: Targeted network interface.
 * @local_ip/remote_ip: Endpoint addresses.
 * @skb_pool: Pre-allocated packet buffer pool to avoid memory allocation during panics.
 * 
 * Functional Intent: Maintains all static and dynamic state required to 
 * forge and transmit packets in an out-of-band manner.
 */
struct netpoll {
	struct net_device *dev;
	netdevice_tracker dev_tracker;
	/*
	 * either dev_name or dev_mac can be used to specify the local
	 * interface - dev_name is used if it is a nonempty string, else
	 * dev_mac is used.
	 */
	char dev_name[IFNAMSIZ];
	u8 dev_mac[ETH_ALEN];
	const char *name;

	union inet_addr local_ip, remote_ip;
	bool ipv6;
	u16 local_port, remote_port;
	u8 remote_mac[ETH_ALEN];
	struct sk_buff_head skb_pool;
	struct work_struct refill_wq;
};

/**
 * Functional Utility: Standardized logging macros for netpoll modules.
 */
#define np_info(np, fmt, ...)				\
	pr_info("%s: " fmt, np->name, ##__VA_ARGS__)
#define np_err(np, fmt, ...)				\
	pr_err("%s: " fmt, np->name, ##__VA_ARGS__)
#define np_notice(np, fmt, ...)				\
	pr_notice("%s: " fmt, np->name, ##__VA_ARGS__)

/**
 * struct netpoll_info - Per-device netpoll orchestration state.
 * Logic: Anchors multiple netpoll instances to a single network device, 
 * managing the low-level transmit queue and hardware access serialization.
 */
struct netpoll_info {
	refcount_t refcnt;

	struct semaphore dev_lock;

	struct sk_buff_head txq;

	struct delayed_work tx_work;

	struct netpoll *netpoll;
	struct rcu_head rcu;
};

#ifdef CONFIG_NETPOLL
/**
 * External API: Polling Control.
 * Logic: netpoll_poll_dev triggers a manual RX/TX cycle on the hardware. 
 * netpoll_poll_disable ensures standard stack processing doesn't race 
 * with a netpoll session.
 */
void netpoll_poll_dev(struct net_device *dev);
void netpoll_poll_disable(struct net_device *dev);
void netpoll_poll_enable(struct net_device *dev);
#else
static inline void netpoll_poll_disable(struct net_device *dev) { return; }
static inline void netpoll_poll_enable(struct net_device *dev) { return; }
#endif

/**
 * External API: Transmission Primitives.
 * Logic: netpoll_send_udp handles protocol encapsulation, while 
 * netpoll_send_skb provides direct low-level packet submission.
 */
int netpoll_send_udp(struct netpoll *np, const char *msg, int len);
int __netpoll_setup(struct netpoll *np, struct net_device *ndev);
int netpoll_setup(struct netpoll *np);
void __netpoll_free(struct netpoll *np);
void netpoll_cleanup(struct netpoll *np);
void do_netpoll_cleanup(struct netpoll *np);
netdev_tx_t netpoll_send_skb(struct netpoll *np, struct sk_buff *skb);

#ifdef CONFIG_NETPOLL
/**
 * netpoll_poll_lock - Optimistic NAPI context acquisition.
 * 
 * Block Logic: Atomic owner tracking.
 * Logic: 
 * 1. Checks for active netpoll info on the device.
 * 2. Uses cmpxchg to atomically claim the NAPI polling slot for the current CPU.
 * 3. Spins with cpu_relax() to wait for lock release, preventing concurrent 
 *    access to NAPI structures from different cores.
 */
static inline void *netpoll_poll_lock(struct napi_struct *napi)
{
	struct net_device *dev = napi->dev;

	if (dev && rcu_access_pointer(dev->npinfo)) {
		int owner = smp_processor_id();

		while (cmpxchg(&napi->poll_owner, -1, owner) != -1)
			cpu_relax();

		return napi;
	}
	return NULL;
}

/**
 * netpoll_poll_unlock - Atomic release of NAPI context.
 */
static inline void netpoll_poll_unlock(void *have)
{
	struct napi_struct *napi;

	if (have) {
		napi = have;
		// Functional Utility: Finalizes visibility of NAPI updates before 
		// allowing other processors to claim the lock.
		smp_store_release(&napi->poll_owner, -1);
	}
}

/**
 * netpoll_tx_running - Predicate to detect active netpoll transmission.
 */
static inline bool netpoll_tx_running(struct net_device *dev)
{
	return irqs_disabled();
}

#else
static inline void *netpoll_poll_lock(struct napi_struct *napi)
{
	return NULL;
}
static inline void netpoll_poll_unlock(void *have)
{
}
static inline bool netpoll_tx_running(struct net_device *dev)
{
	return false;
}
#endif

#endif
