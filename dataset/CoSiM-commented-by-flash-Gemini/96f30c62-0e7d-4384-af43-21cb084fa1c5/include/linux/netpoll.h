/* SPDX-License-Identifier: GPL-2.0 */
/*
 * @file netpoll.h
 * @brief Common definitions for low-level network debugging and logging facilities.
 *
 * This header provides the necessary structures and function prototypes for
 * kernel netpoll facilities, which enable network console, network dump,
 * and network debugger functionalities without relying on the full network
 * stack. It is designed for early boot or critical error scenarios where
 * the standard network drivers might not be fully operational or safe to use.
 *
 * Derived from netconsole, kgdb-over-ethernet, and netdump patches.
 */

#ifndef _LINUX_NETPOLL_H
#define _LINUX_NETPOLL_H

#include <linux/netdevice.h>
#include <linux/interrupt.h>
#include <linux/rcupdate.h>
#include <linux/list.h>
#include <linux/refcount.h>

union inet_addr {
	__u32		all[4];
	__be32		ip;
	__be32		ip6[4];
	struct in_addr	in;
	struct in6_addr	in6;
};

struct netpoll {
	struct net_device *dev; /**< @brief Pointer to the network device. */
	netdevice_tracker dev_tracker; /**< @brief Tracks the net_device lifetime. */
	/*
	 * @brief Specifies the local interface.
	 *
	 * Either dev_name or dev_mac can be used to specify the local
	 * interface - dev_name is used if it is a nonempty string, else
	 * dev_mac is used.
	 */
	char dev_name[IFNAMSIZ]; /**< @brief Name of the network device. */
	u8 dev_mac[ETH_ALEN]; /**< @brief MAC address of the network device. */
	const char *name; /**< @brief Name of the netpoll instance (e.g., "netconsole"). */

	union inet_addr local_ip, remote_ip; /**< @brief Local and remote IP addresses. */
	bool ipv6; /**< @brief Flag indicating if IPv6 is used. */
	u16 local_port, remote_port; /**< @brief Local and remote UDP ports. */
	u8 remote_mac[ETH_ALEN]; /**< @brief MAC address of the remote host. */
	struct sk_buff_head skb_pool; /**< @brief Head of the socket buffer pool for outgoing packets. */
	struct work_struct refill_wq; /**< @brief Workqueue for refilling the SKB pool. */
};

/**
 * @brief Macro for printing informational messages from a netpoll instance.
 * @param np The netpoll instance.
 * @param fmt Format string.
 * @param ... Variable arguments for the format string.
 */
#define np_info(np, fmt, ...) \
	pr_info("%s: " fmt, np->name, ##__VA_ARGS__)
/**
 * @brief Macro for printing error messages from a netpoll instance.
 * @param np The netpoll instance.
 * @param fmt Format string.
 * @param ... Variable arguments for the format string.
 */
#define np_err(np, fmt, ...) \
	pr_err("%s: " fmt, np->name, ##__VA_ARGS__)
/**
 * @brief Macro for printing notice-level messages from a netpoll instance.
 * @param np The netpoll instance.
 * @param fmt Format string.
 * @param ... Variable arguments for the format string.
 */
#define np_notice(np, fmt, ...) \
	pr_notice("%s: " fmt, np->name, ##__VA_ARGS__)

/**
 * @struct netpoll_info
 * @brief Supplementary information and state for a netpoll instance, managed via RCU.
 *
 * This structure holds reference counts, a device lock, and a transmit queue
 * for managing netpoll operations, particularly in an RCU-protected context.
 */
struct netpoll_info {
	refcount_t refcnt; /**< @brief Reference count for the netpoll_info structure. */

	struct semaphore dev_lock; /**< @brief Semaphore for protecting access to the network device. */

	struct sk_buff_head txq; /**< @brief Transmit queue for outgoing packets. */

	struct delayed_work tx_work; /**< @brief Delayed work for processing the transmit queue. */

	struct netpoll *netpoll; /**< @brief Back-pointer to the main netpoll structure. */
	struct rcu_head rcu; /**< @brief RCU head for safe deferred freeing. */
};

/**
 * @brief Conditional compilation block for CONFIG_NETPOLL.
 *
 * This block contains functions that are only available when the NETPOLL
 * configuration option is enabled in the kernel. These functions provide
 * mechanisms for polling the network device and enabling/disabling polling.
 */
#ifdef CONFIG_NETPOLL
/**
 * @brief Forces a poll of the specified network device.
 *
 * This function is used to manually trigger the polling mechanism for a
 * network device, bypassing the normal interrupt-driven receive path.
 *
 * @param dev Pointer to the network device to poll.
 */
void netpoll_poll_dev(struct net_device *dev);
/**
 * @brief Disables netpoll polling for the specified network device.
 *
 * Prevents the netpoll mechanism from actively polling the network device
 * for incoming packets.
 *
 * @param dev Pointer to the network device.
 */
void netpoll_poll_disable(struct net_device *dev);
/**
 * @brief Enables netpoll polling for the specified network device.
 *
 * Activates the netpoll mechanism to actively poll the network device
 * for incoming packets, typically used in situations where interrupts
 * are disabled or unreliable.
 *
 * @param dev Pointer to the network device.
 */
void netpoll_poll_enable(struct net_device *dev);
#else /* !CONFIG_NETPOLL */
/**
 * @brief Statically inline function to disable netpoll polling (no-op when CONFIG_NETPOLL is off).
 * @param dev Pointer to the network device.
 */
static inline void netpoll_poll_disable(struct net_device *dev) { return; }
/**
 * @brief Statically inline function to enable netpoll polling (no-op when CONFIG_NETPOLL is off).
 * @param dev Pointer to the network device.
 */
static inline void netpoll_poll_enable(struct net_device *dev) { return; }
#endif /* CONFIG_NETPOLL */

/**
 * @brief Sends a UDP message using the netpoll mechanism.
 *
 * This function allows sending a raw UDP message through the netpoll-configured
 * network interface, bypassing the normal network stack.
 *
 * @param np Pointer to the netpoll instance.
 * @param msg Pointer to the message data.
 * @param len Length of the message data.
 * @return 0 on success, or a negative error code on failure.
 */
int netpoll_send_udp(struct netpoll *np, const char *msg, int len);
/**
 * @brief Internal function to set up a netpoll instance.
 *
 * This function performs the core setup for a netpoll instance, associating
 * it with a network device.
 *
 * @param np Pointer to the netpoll instance to set up.
 * @param ndev Pointer to the net_device to associate with.
 * @return 0 on success, or a negative error code on failure.
 */
int __netpoll_setup(struct netpoll *np, struct net_device *ndev);
/**
 * @brief Sets up a netpoll instance.
 *
 * This function is the primary entry point for configuring and activating
 * a netpoll instance.
 *
 * @param np Pointer to the netpoll instance to set up.
 * @return 0 on success, or a negative error code on failure.
 */
int netpoll_setup(struct netpoll *np);
/**
 * @brief Internal function to free resources associated with a netpoll instance.
 *
 * This function performs the core cleanup and deallocation of resources
 * for a netpoll instance.
 *
 * @param np Pointer to the netpoll instance to free.
 */
void __netpoll_free(struct netpoll *np);
/**
 * @brief Cleans up and deactivates a netpoll instance.
 *
 * This function is the primary entry point for deconfiguring and cleaning up
 * a netpoll instance.
 *
 * @param np Pointer to the netpoll instance to clean up.
 */
void netpoll_cleanup(struct netpoll *np);
/**
 * @brief Executes the actual netpoll cleanup operations.
 *
 * This function is called to perform the final cleanup of a netpoll instance.
 *
 * @param np Pointer to the netpoll instance.
 */
void do_netpoll_cleanup(struct netpoll *np);
/**
 * @brief Sends an SKB using the netpoll mechanism.
 *
 * This function transmits a pre-allocated socket buffer (SKB) through the
 * netpoll-configured interface.
 *
 * @param np Pointer to the netpoll instance.
 * @param skb Pointer to the socket buffer to send.
 * @return `NETDEV_TX_OK` on success, or other `netdev_tx_t` error codes.
 */
netdev_tx_t netpoll_send_skb(struct netpoll *np, struct sk_buff *skb);

/**
 * @brief Conditional compilation block for CONFIG_NETPOLL, related to polling locks.
 *
 * This block contains functions for acquiring and releasing polling locks,
 * as well as checking if netpoll transmission is currently active. These are
 * only available when CONFIG_NETPOLL is enabled.
 */
#ifdef CONFIG_NETPOLL
/**
 * @brief Acquires the netpoll polling lock for a NAPI instance.
 *
 * This function attempts to acquire a lock to allow netpoll to process
 * packets on a given NAPI instance. It spins until the lock is obtained.
 *
 * @param napi Pointer to the NAPI structure.
 * @return Pointer to the NAPI structure on success (as a token for unlock), or NULL if netpoll is not configured for the device.
 */
static inline void *netpoll_poll_lock(struct napi_struct *napi)
{
	struct net_device *dev = napi->dev;

	if (dev && rcu_access_pointer(dev->npinfo)) {
		int owner = smp_processor_id(); // Get the current CPU ID.

		while (cmpxchg(&napi->poll_owner, -1, owner) != -1)
			cpu_relax(); // Yield CPU to avoid busy-waiting excessively.

		return napi;
	}
	return NULL;
}

/**
 * @brief Releases the netpoll polling lock.
 *
 * Releases the lock previously acquired by `netpoll_poll_lock`,
 * allowing other CPUs or contexts to acquire it.
 *
 * @param have The token returned by `netpoll_poll_lock`.
 */
static inline void netpoll_poll_unlock(void *have)
{
	struct napi_struct *napi;

	if (have) { // Block Logic: Ensures 'have' is a valid pointer before proceeding.
		napi = have;
		// Functional Utility: Atomically releases the poll_owner lock, ensuring
		// proper memory ordering and allowing other CPUs to acquire the lock.
		smp_store_release(&napi->poll_owner, -1);
	}
}

/**
 * @brief Checks if netpoll transmission is currently running on the device.
 *
 * This is typically true when interrupts are disabled, indicating a critical
 * section where netpoll might be actively transmitting.
 *
 * @param dev Pointer to the network device.
 * @return True if netpoll transmission is active, false otherwise.
 */
static inline bool netpoll_tx_running(struct net_device *dev)
{
	// Functional Utility: Checks if interrupts are disabled, which often
	// implies that netpoll is in a critical transmission phase.
	return irqs_disabled();
}

#else /* !CONFIG_NETPOLL */
/**
 * @brief Statically inline function to acquire netpoll polling lock (no-op when CONFIG_NETPOLL is off).
 * @param napi Pointer to the NAPI structure.
 * @return NULL (lock cannot be acquired if netpoll is disabled).
 */
static inline void *netpoll_poll_lock(struct napi_struct *napi)
{
	return NULL;
}
/**
 * @brief Statically inline function to release netpoll polling lock (no-op when CONFIG_NETPOLL is off).
 * @param have The token.
 */
static inline void netpoll_poll_unlock(void *have)
{
}
/**
 * @brief Statically inline function to check netpoll transmission status (always false when CONFIG_NETPOLL is off).
 * @param dev Pointer to the network device.
 * @return False (netpoll transmission is never active if netpoll is disabled).
 */
static inline bool netpoll_tx_running(struct net_device *dev)
{
	return false;
}
#endif /* CONFIG_NETPOLL */

#endif // _LINUX_NETPOLL_H

