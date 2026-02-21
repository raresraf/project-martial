/* SPDX-License-Identifier: GPL-2.0 */
/**
 * @file netpoll.h
 * @brief Common definitions for low-level network polling mechanisms in the Linux kernel.
 *
 * This header provides the fundamental structures and function prototypes used by
 * low-level network utilities such as network consoles, network dump tools, and
 * network debuggers. It centralizes common code originally derived from projects
 * like netconsole, kgdb-over-ethernet, and netdump, facilitating robust network
 * communication even in critical system states.
 *
 * @96f30c62-0e7d-4384-af43-21cb084fa1c5/include/linux/netpoll.h
 */

#ifndef _LINUX_NETPOLL_H
#define _LINUX_NETPOLL_H

#include <linux/netdevice.h>
#include <linux/interrupt.h>
#include <linux/rcupdate.h>
#include <linux/list.h>
#include <linux/refcount.h>

/**
 * @brief Union to hold either IPv4 or IPv6 addresses.
 *
 * This union provides a flexible way to store network addresses,
 * accommodating both IPv4 ({@code in_addr}) and IPv6 ({@code in6_addr}) formats,
 * as well as raw 32-bit or 128-bit representations.
 */
union inet_addr {
	__u32		all[4];
	__be32		ip;
	__be32		ip6[4];
	struct in_addr	in;
	struct in6_addr	in6;
};

/**
 * @brief Structure to hold network polling configuration and state.
 *
 * This structure encapsulates all necessary information for a network polling
 * instance, including device details, IP/port configurations, MAC addresses,
 * and an SKB (Socket Buffer) pool for packet transmission.
 */
struct netpoll {
	struct net_device *dev; /**< @brief The network device to be used for polling. */
	netdevice_tracker dev_tracker; /**< @brief Tracks the net_device for RCU protection. */
	/*
	 * Either dev_name or dev_mac can be used to specify the local
	 * interface - dev_name is used if it is a nonempty string, else
	 * dev_mac is used.
	 */
	char dev_name[IFNAMSIZ]; /**< @brief Name of the local network device. */
	u8 dev_mac[ETH_ALEN]; /**< @brief MAC address of the local network device. */
	const char *name; /**< @brief A descriptive name for this netpoll instance. */

	union inet_addr local_ip, remote_ip; /**< @brief Local and remote IP addresses. */
	bool ipv6; /**< @brief True if IPv6 is used, false for IPv4. */
	u16 local_port, remote_port; /**< @brief Local and remote UDP port numbers. */
	u8 remote_mac[ETH_ALEN]; /**< @brief MAC address of the remote host. */
	struct sk_buff_head skb_pool; /**< @brief SKB pool for allocating packets. */
	struct work_struct refill_wq; /**< @brief Work queue for refilling the SKB pool. */
};

/**
 * @brief Logs an informational message related to a netpoll instance.
 * @param np The netpoll instance.
 * @param fmt Format string for the message.
 * @param ... Variable arguments for the format string.
 */
#define np_info(np, fmt, ...)				\
	pr_info("%s: " fmt, np->name, ##__VA_ARGS__)
/**
 * @brief Logs an error message related to a netpoll instance.
 * @param np The netpoll instance.
 * @param fmt Format string for the message.
 * @param ... Variable arguments for the format string.
 */
#define np_err(np, fmt, ...)				\
	pr_err("%s: " fmt, np->name, ##__VA_ARGS__)
/**
 * @brief Logs a notice message related to a netpoll instance.
 * @param np The netpoll instance.
 * @param fmt Format string for the message.
 * @param ... Variable arguments for the format string.
 */
#define np_notice(np, fmt, ...)				\
	pr_notice("%s: " fmt, np->name, ##__VA_ARGS__)

/**
 * @brief Internal structure to manage netpoll instance lifecycle and transmission.
 *
 * This structure holds runtime information for a netpoll instance,
 * primarily managing reference counting, device locking, and a transmit queue.
 */
struct netpoll_info {
	refcount_t refcnt; /**< @brief Reference count for the netpoll_info structure. */

	struct semaphore dev_lock; /**< @brief Semaphore to protect net_device access. */

	struct sk_buff_head txq; /**< @brief Transmit queue for SKBs. */

	struct delayed_work tx_work; /**< @brief Delayed work structure for transmit operations. */

	struct netpoll *netpoll; /**< @brief Pointer to the associated netpoll instance. */
	struct rcu_head rcu; /**< @brief RCU head for safe cleanup. */
};

#ifdef CONFIG_NETPOLL
/**
 * @brief Forces a poll on the specified network device.
 * @param dev The network device to poll.
 */
void netpoll_poll_dev(struct net_device *dev);
/**
 * @brief Disables polling on a network device for netpoll operations.
 * @param dev The network device to disable polling on.
 */
void netpoll_poll_disable(struct net_device *dev);
/**
 * @brief Enables polling on a network device for netpoll operations.
 * @param dev The network device to enable polling on.
 */
void netpoll_poll_enable(struct net_device *dev);
#else
static inline void netpoll_poll_disable(struct net_device *dev) { return; }
static inline void netpoll_poll_enable(struct net_device *dev) { return; }
#endif

/**
 * @brief Sends a UDP message using the netpoll mechanism.
 * @param np The netpoll instance to use for sending.
 * @param msg The message to send.
 * @param len The length of the message.
 * @return 0 on success, or a negative error code on failure.
 */
int netpoll_send_udp(struct netpoll *np, const char *msg, int len);
/**
 * @brief Internal setup function for a netpoll instance, using a specific net_device.
 * @param np The netpoll instance to set up.
 * @param ndev The net_device to associate with the netpoll instance.
 * @return 0 on success, or a negative error code on failure.
 */
int __netpoll_setup(struct netpoll *np, struct net_device *ndev);
/**
 * @brief Sets up a netpoll instance.
 * @param np The netpoll instance to set up.
 * @return 0 on success, or a negative error code on failure.
 */
int netpoll_setup(struct netpoll *np);
/**
 * @brief Internal function to free resources associated with a netpoll instance.
 * @param np The netpoll instance to free.
 */
void __netpoll_free(struct netpoll *np);
/**
 * @brief Cleans up and releases resources for a netpoll instance.
 * @param np The netpoll instance to clean up.
 */
void netpoll_cleanup(struct netpoll *np);
/**
 * @brief Performs the actual cleanup operations for a netpoll instance.
 * @param np The netpoll instance to perform cleanup on.
 */
void do_netpoll_cleanup(struct netpoll *np);
/**
 * @brief Sends a pre-allocated SKB using the netpoll mechanism.
 * @param np The netpoll instance to use for sending.
 * @param skb The socket buffer to send.
 * @return netdev_tx_t status code (e.g., NETDEV_TX_OK).
 */
netdev_tx_t netpoll_send_skb(struct netpoll *np, struct sk_buff *skb);

#ifdef CONFIG_NETPOLL
/**
 * @brief Acquires a lock for netpoll polling operations.
 *
 * This function attempts to acquire a polling lock for the given NAPI structure.
 * It is used to protect NAPI context during netpoll operations, especially
 * when potentially conflicting operations might occur.
 *
 * @param napi The NAPI structure associated with the network device.
 * @return The NAPI structure if the lock was acquired, or NULL if not.
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
 * @brief Releases the lock for netpoll polling operations.
 *
 * This function releases the polling lock previously acquired by `netpoll_poll_lock`.
 * It is crucial for ensuring proper synchronization and preventing deadlocks.
 *
 * @param have The value returned by `netpoll_poll_lock` (the NAPI structure).
 */
static inline void netpoll_poll_unlock(void *have)
{
	struct napi_struct *napi = have;

	if (napi)
		smp_store_release(&napi->poll_owner, -1);
}

/**
 * @brief Checks if netpoll transmit operations are currently running.
 *
 * This function indicates whether transmit operations are active, typically
 * by checking if IRQs are disabled, which is a common state during netpoll TX.
 *
 * @param dev The network device to check.
 * @return True if transmit operations are running (IRQs disabled), false otherwise.
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
