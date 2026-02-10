/* SPDX-License-Identifier: GPL-2.0 */
/**
 * @file netpoll.h
 * @brief Header for the Netpoll subsystem.
 * @details The Netpoll API provides a simplified, polling-based I/O interface over network devices.
 * It is designed for use in atomic contexts and situations where the full network stack is
 * unavailable or undesirable, such as for kernel debuggers (kgdb), network consoles (netconsole),
 * and crash dump mechanisms (netdump). It bypasses the standard queueing disciplines and
 * operates directly on the network device driver.
 */

#ifndef _LINUX_NETPOLL_H
#define _LINUX_NETPOLL_H

#include <linux/netdevice.h>
#include <linux/interrupt.h>
#include <linux/rcupdate.h>
#include <linux/list.h>
#include <linux/refcount.h>

/**
 * @union inet_addr
 * @brief A union to hold an IPv4 or IPv6 address.
 * @details This structure allows network address information to be handled in a protocol-agnostic
 *          manner throughout the Netpoll subsystem.
 */
union inet_addr {
	__u32		all[4];	/**< Raw 128-bit storage for IPv6 addresses */
	__be32		ip;	/**< Big-endian IPv4 address */
	__be32		ip6[4];	/**< Big-endian IPv6 address as an array */
	struct in_addr	in;	/**< Standard IPv4 address structure */
	struct in6_addr	in6;	/**< Standard IPv6 address structure */
};

/**
 * @struct netpoll
 * @brief Represents a single instance of a Netpoll client.
 * @details This structure contains all the configuration required to send and receive packets
 *          on a specific network interface, including addressing information and a private
 *          pool of SKBs.
 */
struct netpoll {
	struct net_device *dev;		/**< The network device to use. */
	netdevice_tracker dev_tracker;	/**< Tracker for device lifetime management. */
	/*
	 * Either dev_name or dev_mac can be used to specify the local
	 * interface - dev_name is used if it is a nonempty string, else
	 * dev_mac is used.
	 */
	char dev_name[IFNAMSIZ];	/**< Name of the network device (e.g., "eth0"). */
	u8 dev_mac[ETH_ALEN];		/**< MAC address of the local device. */
	const char *name;		/**< Name of the Netpoll client (e.g., "netconsole"). */

	union inet_addr local_ip, remote_ip; /**< Local and remote IP addresses. */
	bool ipv6;			/**< Flag indicating if IPv6 is used. */
	u16 local_port, remote_port;	/**< Local and remote UDP ports. */
	u8 remote_mac[ETH_ALEN];	/**< MAC address of the remote target. */
	struct sk_buff_head skb_pool;	/**< A pre-allocated pool of SKBs for sending packets. */
	struct work_struct refill_wq;	/**< Work queue for replenishing the SKB pool. */
};

/* Logging macros with Netpoll client name prefix */
#define np_info(np, fmt, ...)				\
	pr_info("%s: " fmt, np->name, ##__VA_ARGS__)
#define np_err(np, fmt, ...)				\
	pr_err("%s: " fmt, np->name, ##__VA_ARGS__)
#define np_notice(np, fmt, ...)				\
	pr_notice("%s: " fmt, np->name, ##__VA_ARGS__)

/**
 * @struct netpoll_info
 * @brief Per-device state for the Netpoll subsystem.
 * @details This structure is attached to a net_device that is being used by Netpoll.
 *          It holds a transmit queue and state needed to manage polling and cleanup.
 */
struct netpoll_info {
	refcount_t refcnt;		/**< Reference count for multiple Netpoll clients on one device. */

	struct semaphore dev_lock;	/**< Semaphore to protect against device state changes. */

	struct sk_buff_head txq;	/**< Queue for packets to be transmitted during polling. */

	struct delayed_work tx_work;	/**< Work structure for delayed transmission processing. */

	struct netpoll *netpoll;	/**< Back-pointer to the associated netpoll structure. */
	struct rcu_head rcu;		/**< RCU head for safe deferred cleanup. */
};

#ifdef CONFIG_NETPOLL
/**
 * @brief Force a polling cycle on a network device.
 * @details This function is called by a Netpoll client to manually trigger the NAPI poll loop
 *          for the specified device, allowing for packet reception and transmission in an atomic context.
 * @param dev The network device to poll.
 */
void netpoll_poll_dev(struct net_device *dev);
void netpoll_poll_disable(struct net_device *dev);
void netpoll_poll_enable(struct net_device *dev);
#else
static inline void netpoll_poll_disable(struct net_device *dev) { return; }
static inline void netpoll_poll_enable(struct net_device *dev) { return; }
#endif

int netpoll_send_udp(struct netpoll *np, const char *msg, int len);
int __netpoll_setup(struct netpoll *np, struct net_device *ndev);
/**
 * @brief Initialize a Netpoll client instance.
 * @details Sets up the netpoll structure, finds the specified network device, and allocates
 *          initial resources like the SKB pool.
 * @param np Pointer to the netpoll structure to be initialized.
 * @return 0 on success, or a negative error code on failure.
 */
int netpoll_setup(struct netpoll *np);
void __netpoll_free(struct netpoll *np);
void netpoll_cleanup(struct netpoll *np);
void do_netpoll_cleanup(struct netpoll *np);
netdev_tx_t netpoll_send_skb(struct netpoll *np, struct sk_buff *skb);

#ifdef CONFIG_NETPOLL
/**
 * @brief Acquire a lock on the NAPI poll context.
 * @details This function attempts to take ownership of the NAPI poll loop for a device.
 *          It uses a cmpxchg loop to atomically set the `poll_owner`, preventing contention
 *          between the Netpoll client and the normal interrupt-driven network stack. This is
 *          essential for safe polling from an atomic context.
 * @param napi The NAPI structure of the device to be locked.
 * @return A non-NULL pointer (the napi struct) on successful lock acquisition, NULL otherwise.
 */
static inline void *netpoll_poll_lock(struct napi_struct *napi)
{
	struct net_device *dev = napi->dev;

	if (dev && rcu_access_pointer(dev->npinfo)) {
		int owner = smp_processor_id();

		/* Non-blocking attempt to acquire the poll lock. */
		while (cmpxchg(&napi->poll_owner, -1, owner) != -1)
			cpu_relax();

		return napi;
	}
	return NULL;
}

/**
 * @brief Release the lock on the NAPI poll context.
 * @details Releases the ownership of the NAPI poll loop acquired by `netpoll_poll_lock`.
 *          The memory barrier (`smp_store_release`) ensures that all memory operations
 *          are completed before the lock is released.
 * @param have The pointer returned by a successful call to `netpoll_poll_lock`.
 */
static inline void netpoll_poll_unlock(void *have)
{
	struct napi_struct *napi = have;

	if (napi)
		smp_store_release(&napi->poll_owner, -1);
}

/**
 * @brief Check if Netpoll is currently active for transmission.
 * @details Netpoll is considered to be in a transmit context if interrupts are disabled,
 *          as this is a prerequisite for safe, non-blocking transmission from critical code paths.
 * @param dev The network device.
 * @return True if running in a Netpoll transmit context, false otherwise.
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