/**
 * @file netpoll.h
 * @brief Core definitions for the Netpoll API.
 *
 * Overview:
 * This file provides the central data structures and function prototypes for the Linux
 * Netpoll subsystem. Netpoll is a lightweight, polled-mode networking interface
 * designed for critical, low-level kernel services like network-based consoles
 * (netconsole), kernel debuggers (kgdb-over-ethernet), and crash dump mechanisms
 * (netdump).
 *
 * Functional Utility:
 * The key characteristic of Netpoll is its ability to send and receive network
 * packets without relying on interrupts. This makes it viable in contexts where
 * the standard interrupt-driven network stack may be non-operational or explicitly
 * disabled, such as during a system panic, early boot, or a debugging session.
 * It operates by directly polling the network interface controller (NIC) driver.
 *
 * Key Structures:
 * - struct netpoll: Represents a single Netpoll client instance, containing all
 *   configuration details like IP addresses, ports, and MAC addresses for a
 *   communication channel.
 * - struct netpoll_info: A companion structure attached to a `net_device`,
 *   managing the transmit queue and state needed for polled-mode I/O on that
 *   device.
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
 * @brief A generic container for either an IPv4 or IPv6 address.
 * @note This union allows Netpoll to handle both IP protocol versions with a
 *       common data structure, abstracting away address-family-specific details.
 */
union inet_addr {
	__be32		ip;    // Storage for an IPv4 address in network byte order.
	struct in6_addr	in6;   // Storage for an IPv6 address.
};

/**
 * @struct netpoll
 * @brief Defines a Netpoll client instance and its communication parameters.
 *
 * This structure holds all the configuration required to establish a network
 * communication channel for a low-level service. It specifies the local and
 * remote endpoints, including network device, IP addresses, and MAC addresses.
 */
struct netpoll {
	struct net_device *dev;		// The network device this Netpoll instance is bound to.
	netdevice_tracker dev_tracker;	// Tracker to safely handle device unbinding.

	/*
	 * Either dev_name or dev_mac can be used to specify the local
	 * interface - dev_name is used if it is a nonempty string, else
	 * dev_mac is used.
	 */
	char dev_name[IFNAMSIZ]; // Name of the network device (e.g., "eth0").
	u8 dev_mac[ETH_ALEN];    // MAC address of the local network device.
	const char *name;		// Name of the service using this Netpoll instance (e.g., "netconsole").

	union inet_addr local_ip, remote_ip; // Local and remote IP addresses.
	bool ipv6;				// Flag indicating if the connection is IPv6.
	u16 local_port, remote_port;	// Local and remote UDP ports.
	u8 remote_mac[ETH_ALEN];		// MAC address of the remote target.
	struct sk_buff_head skb_pool;	// A pre-allocated pool of SKBs for sending data.
	struct work_struct refill_wq;	// Work queue item for replenishing the SKB pool.
};

/*
 * @def np_info, np_err, np_notice
 * @brief Logging macros for Netpoll instances.
 *
 * Functional Utility: These macros provide a standardized logging format for Netpoll,
 * automatically prefixing each message with the name of the service (e.g., "netconsole")
 * to simplify debugging.
 */
#define np_info(np, fmt, ...)																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				<br>