/*
 * BlueZ - Bluetooth protocol stack for Linux
 *
 * Copyright (C) 2010  Nokia Corporation
 * Copyright (C) 2011-2012 Intel Corporation
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD PARTY RIGHTS.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER(S) AND AUTHOR(S) BE LIABLE FOR ANY
 * CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * ALL LIABILITY, INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PATENTS,
 * COPYRIGHTS, TRADEMARKS OR OTHER RIGHTS, RELATING TO USE OF THIS
 * SOFTWARE IS DISCLAIMED.
 */
/**
 * @file
 * @brief This file implements the Bluetooth Management (mgmt) interface, providing a unified way to manage Bluetooth controllers and their settings from user space.
 *
 * It handles various management commands such as powering controllers on/off,
 * managing discoverability, pairing, and handling different types of keys.
 * It also provides functionalities for managing LE advertising and mesh networking.
 * The interface is designed to be extensible and supports both standard and
 * vendor-specific Bluetooth features.
 */

#include <linux/module.h>
#include <linux/unaligned.h>

#include <net/bluetooth/bluetooth.h>
#include <net/bluetooth/hci_core.h>
#include <net/bluetooth/hci_sock.h>
#include <net/bluetooth/l2cap.h>
#include <net/bluetooth/mgmt.h>

#include "smp.h"
#include "mgmt_util.h"
#include "mgmt_config.h"
#include "msft.h"
#include "eir.h"
#include "aosp.h"

#define MGMT_VERSION	1
#define MGMT_REVISION	23

static const u16 mgmt_commands[] = {
	MGMT_OP_READ_INDEX_LIST,
	MGMT_OP_READ_INFO,
	MGMT_OP_SET_POWERED,
	MGMT_OP_SET_DISCOVERABLE,
	MGMT_OP_SET_CONNECTABLE,
	MGMT_OP_SET_FAST_CONNECTABLE,
	MGMT_OP_SET_BONDABLE,
	MGMT_OP_SET_LINK_SECURITY,
	MGMT_OP_SET_SSP,
	MGMT_OP_SET_HS,
	MGMT_OP_SET_LE,
	MGMT_OP_SET_DEV_CLASS,
	MGMT_OP_SET_LOCAL_NAME,
	MGMT_OP_ADD_UUID,
	MGMT_OP_REMOVE_UUID,
	MGMT_OP_LOAD_LINK_KEYS,
	MGMT_OP_LOAD_LONG_TERM_KEYS,
	MGMT_OP_DISCONNECT,
	MGMT_OP_GET_CONNECTIONS,
	MGMT_OP_PIN_CODE_REPLY,
	MGMT_OP_PIN_CODE_NEG_REPLY,
	MGMT_OP_SET_IO_CAPABILITY,
	MGMT_OP_PAIR_DEVICE,
	MGMT_OP_CANCEL_PAIR_DEVICE,
	MGMT_OP_UNPAIR_DEVICE,
	MGMT_OP_USER_CONFIRM_REPLY,
	MGMT_OP_USER_CONFIRM_NEG_REPLY,
	MGMT_OP_USER_PASSKEY_REPLY,
	MGMT_OP_USER_PASSKEY_NEG_REPLY,
	MGMT_OP_READ_LOCAL_OOB_DATA,
	MGMT_OP_ADD_REMOTE_OOB_DATA,
	MGMT_OP_REMOVE_REMOTE_OOB_DATA,
	MGMT_OP_START_DISCOVERY,
	MGMT_OP_STOP_DISCOVERY,
	MGMT_OP_CONFIRM_NAME,
	MGMT_OP_BLOCK_DEVICE,
	MGMT_OP_UNBLOCK_DEVICE,
	MGMT_OP_SET_DEVICE_ID,
	MGMT_OP_SET_ADVERTISING,
	MGMT_OP_SET_BREDR,
	MGMT_OP_SET_STATIC_ADDRESS,
	MGMT_OP_SET_SCAN_PARAMS,
	MGMT_OP_SET_SECURE_CONN,
	MGMT_OP_SET_DEBUG_KEYS,
	MGMT_OP_SET_PRIVACY,
	MGMT_OP_LOAD_IRKS,
	MGMT_OP_GET_CONN_INFO,
	MGMT_OP_GET_CLOCK_INFO,
	MGMT_OP_ADD_DEVICE,
	MGMT_OP_REMOVE_DEVICE,
	MGMT_OP_LOAD_CONN_PARAM,
	MGMT_OP_READ_UNCONF_INDEX_LIST,
	MGMT_OP_READ_CONFIG_INFO,
	MGMT_OP_SET_EXTERNAL_CONFIG,
	MGMT_OP_SET_PUBLIC_ADDRESS,
	MGMT_OP_START_SERVICE_DISCOVERY,
	MGMT_OP_READ_LOCAL_OOB_EXT_DATA,
	MGMT_OP_READ_EXT_INDEX_LIST,
	MGMT_OP_READ_ADV_FEATURES,
	MGMT_OP_ADD_ADVERTISING,
	MGMT_OP_REMOVE_ADVERTISING,
	MGMT_OP_GET_ADV_SIZE_INFO,
	MGMT_OP_START_LIMITED_DISCOVERY,
	MGMT_OP_READ_EXT_INFO,
	MGMT_OP_SET_APPEARANCE,
	MGMT_OP_GET_PHY_CONFIGURATION,
	MGMT_OP_SET_PHY_CONFIGURATION,
	MGMT_OP_SET_BLOCKED_KEYS,
	MGMT_OP_SET_WIDEBAND_SPEECH,
	MGMT_OP_READ_CONTROLLER_CAP,
	MGMT_OP_READ_EXP_FEATURES_INFO,
	MGMT_OP_SET_EXP_FEATURE,
	MGMT_OP_READ_DEF_SYSTEM_CONFIG,
	MGMT_OP_SET_DEF_SYSTEM_CONFIG,
	MGMT_OP_READ_DEF_RUNTIME_CONFIG,
	MGMT_OP_SET_DEF_RUNTIME_CONFIG,
	MGMT_OP_GET_DEVICE_FLAGS,
	MGMT_OP_SET_DEVICE_FLAGS,
	MGMT_OP_READ_ADV_MONITOR_FEATURES,
	MGMT_OP_ADD_ADV_PATTERNS_MONITOR,
	MGMT_OP_REMOVE_ADV_MONITOR,
	MGMT_OP_ADD_EXT_ADV_PARAMS,
	MGMT_OP_ADD_EXT_ADV_DATA,
	MGMT_OP_ADD_ADV_PATTERNS_MONITOR_RSSI,
	MGMT_OP_SET_MESH_RECEIVER,
	MGMT_OP_MESH_READ_FEATURES,
	MGMT_OP_MESH_SEND,
	MGMT_OP_MESH_SEND_CANCEL,
	MGMT_OP_HCI_CMD_SYNC,
};

static const u16 mgmt_events[] = {
	MGMT_EV_CONTROLLER_ERROR,
	MGMT_EV_INDEX_ADDED,
	MGMT_EV_INDEX_REMOVED,
	MGMT_EV_NEW_SETTINGS,
	MGMT_EV_CLASS_OF_DEV_CHANGED,
	MGMT_EV_LOCAL_NAME_CHANGED,
	MGMT_EV_NEW_LINK_KEY,
	MGMT_EV_NEW_LONG_TERM_KEY,
	MGMT_EV_DEVICE_CONNECTED,
	MGMT_EV_DEVICE_DISCONNECTED,
	MGMT_EV_CONNECT_FAILED,
	MGMT_EV_PIN_CODE_REQUEST,
	MGMT_EV_USER_CONFIRM_REQUEST,
	MGMT_EV_USER_PASSKEY_REQUEST,
	MGMT_EV_AUTH_FAILED,
	MGMT_EV_DEVICE_FOUND,
	MGMT_EV_DISCOVERING,
	MGMT_EV_DEVICE_BLOCKED,
	MGMT_EV_DEVICE_UNBLOCKED,
	MGMT_EV_DEVICE_UNPAIRED,
	MGMT_EV_PASSKEY_NOTIFY,
	MGMT_EV_NEW_IRK,
	MGMT_EV_NEW_CSRK,
	MGMT_EV_DEVICE_ADDED,
	MGMT_EV_DEVICE_REMOVED,
	MGMT_EV_NEW_CONN_PARAM,
	MGMT_EV_UNCONF_INDEX_ADDED,
	MGMT_EV_UNCONF_INDEX_REMOVED,
	MGMT_EV_NEW_CONFIG_OPTIONS,
	MGMT_EV_EXT_INDEX_ADDED,
	MGMT_EV_EXT_INDEX_REMOVED,
	MGMT_EV_LOCAL_OOB_DATA_UPDATED,
	MGMT_EV_ADVERTISING_ADDED,
	MGMT_EV_ADVERTISING_REMOVED,
	MGMT_EV_EXT_INFO_CHANGED,
	MGMT_EV_PHY_CONFIGURATION_CHANGED,
	MGMT_EV_EXP_FEATURE_CHANGED,
	MGMT_EV_DEVICE_FLAGS_CHANGED,
	MGMT_EV_ADV_MONITOR_ADDED,
	MGMT_EV_ADV_MONITOR_REMOVED,
	MGMT_EV_CONTROLLER_SUSPEND,
	MGMT_EV_CONTROLLER_RESUME,
	MGMT_EV_ADV_MONITOR_DEVICE_FOUND,
	MGMT_EV_ADV_MONITOR_DEVICE_LOST,
};

static const u16 mgmt_untrusted_commands[] = {
	MGMT_OP_READ_INDEX_LIST,
	MGMT_OP_READ_INFO,
	MGMT_OP_READ_UNCONF_INDEX_LIST,
	MGMT_OP_READ_CONFIG_INFO,
	MGMT_OP_READ_EXT_INDEX_LIST,
	MGMT_OP_READ_EXT_INFO,
	MGMT_OP_READ_CONTROLLER_CAP,
	MGMT_OP_READ_EXP_FEATURES_INFO,
	MGMT_OP_READ_DEF_SYSTEM_CONFIG,
	MGMT_OP_READ_DEF_RUNTIME_CONFIG,
};

static const u16 mgmt_untrusted_events[] = {
	MGMT_EV_INDEX_ADDED,
	MGMT_EV_INDEX_REMOVED,
	MGMT_EV_NEW_SETTINGS,
	MGMT_EV_CLASS_OF_DEV_CHANGED,
	MGMT_EV_LOCAL_NAME_CHANGED,
	MGMT_EV_UNCONF_INDEX_ADDED,
	MGMT_EV_UNCONF_INDEX_REMOVED,
	MGMT_EV_NEW_CONFIG_OPTIONS,
	MGMT_EV_EXT_INDEX_ADDED,
	MGMT_EV_EXT_INDEX_REMOVED,
	MGMT_EV_EXT_INFO_CHANGED,
	MGMT_EV_EXP_FEATURE_CHANGED,
};

#define CACHE_TIMEOUT	secs_to_jiffies(2)

#define ZERO_KEY "\x00\x00\x00\x00\x00\x00\x00\x00" \
		 "\x00\x00\x00\x00\x00\x00\x00\x00"

/* HCI to MGMT error code conversion table */
static const u8 mgmt_status_table[] = {
	MGMT_STATUS_SUCCESS,
	MGMT_STATUS_UNKNOWN_COMMAND,	/* Unknown Command */
	MGMT_STATUS_NOT_CONNECTED,	/* No Connection */
	MGMT_STATUS_FAILED,		/* Hardware Failure */
	MGMT_STATUS_CONNECT_FAILED,	/* Page Timeout */
	MGMT_STATUS_AUTH_FAILED,	/* Authentication Failed */
	MGMT_STATUS_AUTH_FAILED,	/* PIN or Key Missing */
	MGMT_STATUS_NO_RESOURCES,	/* Memory Full */
	MGMT_STATUS_TIMEOUT,		/* Connection Timeout */
	MGMT_STATUS_NO_RESOURCES,	/* Max Number of Connections */
	MGMT_STATUS_NO_RESOURCES,	/* Max Number of SCO Connections */
	MGMT_STATUS_ALREADY_CONNECTED,	/* ACL Connection Exists */
	MGMT_STATUS_BUSY,		/* Command Disallowed */
	MGMT_STATUS_NO_RESOURCES,	/* Rejected Limited Resources */
	MGMT_STATUS_REJECTED,		/* Rejected Security */
	MGMT_STATUS_REJECTED,		/* Rejected Personal */
	MGMT_STATUS_TIMEOUT,		/* Host Timeout */
	MGMT_STATUS_NOT_SUPPORTED,	/* Unsupported Feature */
	MGMT_STATUS_INVALID_PARAMS,	/* Invalid Parameters */
	MGMT_STATUS_DISCONNECTED,	/* OE User Ended Connection */
	MGMT_STATUS_NO_RESOURCES,	/* OE Low Resources */
	MGMT_STATUS_DISCONNECTED,	/* OE Power Off */
	MGMT_STATUS_DISCONNECTED,	/* Connection Terminated */
	MGMT_STATUS_BUSY,		/* Repeated Attempts */
	MGMT_STATUS_REJECTED,		/* Pairing Not Allowed */
	MGMT_STATUS_FAILED,		/* Unknown LMP PDU */
	MGMT_STATUS_NOT_SUPPORTED,	/* Unsupported Remote Feature */
	MGMT_STATUS_REJECTED,		/* SCO Offset Rejected */
	MGMT_STATUS_REJECTED,		/* SCO Interval Rejected */
	MGMT_STATUS_REJECTED,		/* Air Mode Rejected */
	MGMT_STATUS_INVALID_PARAMS,	/* Invalid LMP Parameters */
	MGMT_STATUS_FAILED,		/* Unspecified Error */
	MGMT_STATUS_NOT_SUPPORTED,	/* Unsupported LMP Parameter Value */
	MGMT_STATUS_FAILED,		/* Role Change Not Allowed */
	MGMT_STATUS_TIMEOUT,		/* LMP Response Timeout */
	MGMT_STATUS_FAILED,		/* LMP Error Transaction Collision */
	MGMT_STATUS_FAILED,		/* LMP PDU Not Allowed */
	MGMT_STATUS_REJECTED,		/* Encryption Mode Not Accepted */
	MGMT_STATUS_FAILED,		/* Unit Link Key Used */
	MGMT_STATUS_NOT_SUPPORTED,	/* QoS Not Supported */
	MGMT_STATUS_TIMEOUT,		/* Instant Passed */
	MGMT_STATUS_NOT_SUPPORTED,	/* Pairing Not Supported */
	MGMT_STATUS_FAILED,		/* Transaction Collision */
	MGMT_STATUS_FAILED,		/* Reserved for future use */
	MGMT_STATUS_INVALID_PARAMS,	/* Unacceptable Parameter */
	MGMT_STATUS_REJECTED,		/* QoS Rejected */
	MGMT_STATUS_NOT_SUPPORTED,	/* Classification Not Supported */
	MGMT_STATUS_REJECTED,		/* Insufficient Security */
	MGMT_STATUS_INVALID_PARAMS,	/* Parameter Out Of Range */
	MGMT_STATUS_FAILED,		/* Reserved for future use */
	MGMT_STATUS_BUSY,		/* Role Switch Pending */
	MGMT_STATUS_FAILED,		/* Reserved for future use */
	MGMT_STATUS_FAILED,		/* Slot Violation */
	MGMT_STATUS_FAILED,		/* Role Switch Failed */
	MGMT_STATUS_INVALID_PARAMS,	/* EIR Too Large */
	MGMT_STATUS_NOT_SUPPORTED,	/* Simple Pairing Not Supported */
	MGMT_STATUS_BUSY,		/* Host Busy Pairing */
	MGMT_STATUS_REJECTED,		/* Rejected, No Suitable Channel */
	MGMT_STATUS_BUSY,		/* Controller Busy */
	MGMT_STATUS_INVALID_PARAMS,	/* Unsuitable Connection Interval */
	MGMT_STATUS_TIMEOUT,		/* Directed Advertising Timeout */
	MGMT_STATUS_AUTH_FAILED,	/* Terminated Due to MIC Failure */
	MGMT_STATUS_CONNECT_FAILED,	/* Connection Establishment Failed */
	MGMT_STATUS_CONNECT_FAILED,	/* MAC Connection Failed */
};

/**
 * @brief Converts a kernel error code to a management status code.
 * @param err The kernel error code.
 * @return The corresponding management status code.
 */
static u8 mgmt_errno_status(int err)
{
	switch (err) {
	case 0:
		return MGMT_STATUS_SUCCESS;
	case -EPERM:
		return MGMT_STATUS_REJECTED;
	case -EINVAL:
		return MGMT_STATUS_INVALID_PARAMS;
	case -EOPNOTSUPP:
		return MGMT_STATUS_NOT_SUPPORTED;
	case -EBUSY:
		return MGMT_STATUS_BUSY;
	case -ETIMEDOUT:
		return MGMT_STATUS_AUTH_FAILED;
	case -ENOMEM:
		return MGMT_STATUS_NO_RESOURCES;
	case -EISCONN:
		return MGMT_STATUS_ALREADY_CONNECTED;
	case -ENOTCONN:
		return MGMT_STATUS_DISCONNECTED;
	}

	return MGMT_STATUS_FAILED;
}

/**
 * @brief Converts an error code to a management status code.
 * @param err The error code.
 * @return The corresponding management status code.
 *
 * This function converts both kernel error codes and HCI error codes to
 * management status codes.
 */
static u8 mgmt_status(int err)
{
	if (err < 0)
		return mgmt_errno_status(err);

	if (err < ARRAY_SIZE(mgmt_status_table))
		return mgmt_status_table[err];

	return MGMT_STATUS_FAILED;
}

/**
 * @brief Sends an index-related management event.
 * @param event The event code.
 * @param hdev The HCI device.
 * @param data The event data.
 * @param len The length of the event data.
 * @param flag The socket flag to check.
 * @return 0 on success, or a negative error code on failure.
 */
static int mgmt_index_event(u16 event, struct hci_dev *hdev, void *data,
			    u16 len, int flag)
{
	return mgmt_send_event(event, hdev, HCI_CHANNEL_CONTROL, data, len,
			       flag, NULL);
}

/**
 * @brief Sends a management event to a limited set of sockets.
 * @param event The event code.
 * @param hdev The HCI device.
 * @param data The event data.
 * @param len The length of the event data.
 * @param flag The socket flag to check.
 * @param skip_sk The socket to skip.
 * @return 0 on success, or a negative error code on failure.
 */
static int mgmt_limited_event(u16 event, struct hci_dev *hdev, void *data,
			      u16 len, int flag, struct sock *skip_sk)
{
	return mgmt_send_event(event, hdev, HCI_CHANNEL_CONTROL, data, len,
			       flag, skip_sk);
}

/**
 * @brief Sends a management event to all trusted sockets.
 * @param event The event code.
 * @param hdev The HCI device.
 * @param data The event data.
 * @param len The length of the event data.
 * @param skip_sk The socket to skip.
 * @return 0 on success, or a negative error code on failure.
 */
static int mgmt_event(u16 event, struct hci_dev *hdev, void *data, u16 len,
		      struct sock *skip_sk)
{
	return mgmt_send_event(event, hdev, HCI_CHANNEL_CONTROL, data, len,
			       HCI_SOCK_TRUSTED, skip_sk);
}

/**
 * @brief Sends a management event from a socket buffer.
 * @param skb The socket buffer containing the event.
 * @param skip_sk The socket to skip.
 * @return 0 on success, or a negative error code on failure.
 */
static int mgmt_event_skb(struct sk_buff *skb, struct sock *skip_sk)
{
	return mgmt_send_event_skb(HCI_CHANNEL_CONTROL, skb, HCI_SOCK_TRUSTED,
				   skip_sk);
}

/**
 * @brief Converts a management address type to an LE address type.
 * @param mgmt_addr_type The management address type.
 * @return The corresponding LE address type.
 */
static u8 le_addr_type(u8 mgmt_addr_type)
{
	if (mgmt_addr_type == BDADDR_LE_PUBLIC)
		return ADDR_LE_DEV_PUBLIC;
	else
		return ADDR_LE_DEV_RANDOM;
}

/**
 * @brief Fills a version info structure with the current management interface version.
 * @param ver A pointer to the mgmt_rp_read_version structure to fill.
 */
void mgmt_fill_version_info(void *ver)
{
	struct mgmt_rp_read_version *rp = ver;

	rp->version = MGMT_VERSION;
	rp->revision = cpu_to_le16(MGMT_REVISION);
}
/**
 * @brief Handles the MGMT_OP_READ_VERSION command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device (unused).
 * @param data The command data (unused).
 * @param data_len The length of the command data (unused).
 * @return 0 on success, or a negative error code on failure.
 */
static int read_version(struct sock *sk, struct hci_dev *hdev, void *data,
			u16 data_len)
{
	struct mgmt_rp_read_version rp;

	bt_dev_dbg(hdev, "sock %p", sk);

	mgmt_fill_version_info(&rp);

	return mgmt_cmd_complete(sk, MGMT_INDEX_NONE, MGMT_OP_READ_VERSION, 0,
				 &rp, sizeof(rp));
}
/**
 * @brief Handles the MGMT_OP_READ_COMMANDS command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device (unused).
 * @param data The command data (unused).
 * @param data_len The length of the command data (unused).
 * @return 0 on success, or a negative error code on failure.
 *
 * This function returns a list of supported management commands and events.
 * The list depends on whether the socket is trusted.
 */
static int read_commands(struct sock *sk, struct hci_dev *hdev, void *data,
			 u16 data_len)
{
	struct mgmt_rp_read_commands *rp;
	u16 num_commands, num_events;
	size_t rp_size;
	int i, err;

	bt_dev_dbg(hdev, "sock %p", sk);

	if (hci_sock_test_flag(sk, HCI_SOCK_TRUSTED)) {
		num_commands = ARRAY_SIZE(mgmt_commands);
		num_events = ARRAY_SIZE(mgmt_events);
	} else {
		num_commands = ARRAY_SIZE(mgmt_untrusted_commands);
		num_events = ARRAY_SIZE(mgmt_untrusted_events);
	}

	rp_size = sizeof(*rp) + ((num_commands + num_events) * sizeof(u16));

	rp = kmalloc(rp_size, GFP_KERNEL);
	if (!rp)
		return -ENOMEM;

	rp->num_commands = cpu_to_le16(num_commands);
	rp->num_events = cpu_to_le16(num_events);

	if (hci_sock_test_flag(sk, HCI_SOCK_TRUSTED)) {
		__le16 *opcode = rp->opcodes;

		for (i = 0; i < num_commands; i++, opcode++)
			put_unaligned_le16(mgmt_commands[i], opcode);

		for (i = 0; i < num_events; i++, opcode++)
			put_unaligned_le16(mgmt_events[i], opcode);
	} else {
		__le16 *opcode = rp->opcodes;

		for (i = 0; i < num_commands; i++, opcode++)
			put_unaligned_le16(mgmt_untrusted_commands[i], opcode);

		for (i = 0; i < num_events; i++, opcode++)
			put_unaligned_le16(mgmt_untrusted_events[i], opcode);
	}

	err = mgmt_cmd_complete(sk, MGMT_INDEX_NONE, MGMT_OP_READ_COMMANDS, 0,
				rp, rp_size);
	kfree(rp);

	return err;
}

/**
 * @brief Handles the MGMT_OP_READ_INDEX_LIST command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device (unused).
 * @param data The command data (unused).
 * @param data_len The length of the command data (unused).
 * @return 0 on success, or a negative error code on failure.
 *
 * This function returns a list of configured Bluetooth controller indices.
 */
static int read_index_list(struct sock *sk, struct hci_dev *hdev, void *data,
			   u16 data_len)
{
	struct mgmt_rp_read_index_list *rp;
	struct hci_dev *d;
	size_t rp_len;
	u16 count;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	read_lock(&hci_dev_list_lock);

	count = 0;
	list_for_each_entry(d, &hci_dev_list, list) {
		if (!hci_dev_test_flag(d, HCI_UNCONFIGURED))
			count++;
	}

	rp_len = sizeof(*rp) + (2 * count);
	rp = kmalloc(rp_len, GFP_ATOMIC);
	if (!rp) {
		read_unlock(&hci_dev_list_lock);
		return -ENOMEM;
	}

	count = 0;
	list_for_each_entry(d, &hci_dev_list, list) {
		if (hci_dev_test_flag(d, HCI_SETUP) ||
		    hci_dev_test_flag(d, HCI_CONFIG) ||
		    hci_dev_test_flag(d, HCI_USER_CHANNEL))
			continue;

		/* Devices marked as raw-only are neither configured
		 * nor unconfigured controllers.
		 */
		if (hci_test_quirk(d, HCI_QUIRK_RAW_DEVICE))
			continue;

		if (!hci_dev_test_flag(d, HCI_UNCONFIGURED)) {
			rp->index[count++] = cpu_to_le16(d->id);
			bt_dev_dbg(hdev, "Added hci%u", d->id);
		}
	}

	rp->num_controllers = cpu_to_le16(count);
	rp_len = sizeof(*rp) + (2 * count);

	read_unlock(&hci_dev_list_lock);

	err = mgmt_cmd_complete(sk, MGMT_INDEX_NONE, MGMT_OP_READ_INDEX_LIST,
				0, rp, rp_len);

	kfree(rp);

	return err;
}
/**
 * @brief Handles the MGMT_OP_READ_UNCONF_INDEX_LIST command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device (unused).
 * @param data The command data (unused).
 * @param data_len The length of the command data (unused).
 * @return 0 on success, or a negative error code on failure.
 *
 * This function returns a list of unconfigured Bluetooth controller indices.
 */
static int read_unconf_index_list(struct sock *sk, struct hci_dev *hdev,
				  void *data, u16 data_len)
{
	struct mgmt_rp_read_unconf_index_list *rp;
	struct hci_dev *d;
	size_t rp_len;
	u16 count;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	read_lock(&hci_dev_list_lock);

	count = 0;
	list_for_each_entry(d, &hci_dev_list, list) {
		if (hci_dev_test_flag(d, HCI_UNCONFIGURED))
			count++;
	}

	rp_len = sizeof(*rp) + (2 * count);
	rp = kmalloc(rp_len, GFP_ATOMIC);
	if (!rp) {
		read_unlock(&hci_dev_list_lock);
		return -ENOMEM;
	}

	count = 0;
	list_for_each_entry(d, &hci_dev_list, list) {
		if (hci_dev_test_flag(d, HCI_SETUP) ||
		    hci_dev_test_flag(d, HCI_CONFIG) ||
		    hci_dev_test_flag(d, HCI_USER_CHANNEL))
			continue;

		/* Devices marked as raw-only are neither configured
		 * nor unconfigured controllers.
		 */
		if (hci_test_quirk(d, HCI_QUIRK_RAW_DEVICE))
			continue;

		if (hci_dev_test_flag(d, HCI_UNCONFIGURED)) {
			rp->index[count++] = cpu_to_le16(d->id);
			bt_dev_dbg(hdev, "Added hci%u", d->id);
		}
	}

	rp->num_controllers = cpu_to_le16(count);
	rp_len = sizeof(*rp) + (2 * count);

	read_unlock(&hci_dev_list_lock);

	err = mgmt_cmd_complete(sk, MGMT_INDEX_NONE,
				MGMT_OP_READ_UNCONF_INDEX_LIST, 0, rp, rp_len);

	kfree(rp);

	return err;
}
/**
 * @brief Handles the MGMT_OP_READ_EXT_INDEX_LIST command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device (unused).
 * @param data The command data (unused).
 * @param data_len The length of the command data (unused).
 * @return 0 on success, or a negative error code on failure.
 *
 * This function returns a list of all Bluetooth controller indices,
 * including their type (configured or unconfigured) and bus.
 */
static int read_ext_index_list(struct sock *sk, struct hci_dev *hdev,
			       void *data, u16 data_len)
{
	struct mgmt_rp_read_ext_index_list *rp;
	struct hci_dev *d;
	u16 count;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	read_lock(&hci_dev_list_lock);

	count = 0;
	list_for_each_entry(d, &hci_dev_list, list)
		count++;

	rp = kmalloc(struct_size(rp, entry, count), GFP_ATOMIC);
	if (!rp) {
		read_unlock(&hci_dev_list_lock);
		return -ENOMEM;
	}

	count = 0;
	list_for_each_entry(d, &hci_dev_list, list) {
		if (hci_dev_test_flag(d, HCI_SETUP) ||
		    hci_dev_test_flag(d, HCI_CONFIG) ||
		    hci_dev_test_flag(d, HCI_USER_CHANNEL))
			continue;

		/* Devices marked as raw-only are neither configured
		 * nor unconfigured controllers.
		 */
		if (hci_test_quirk(d, HCI_QUIRK_RAW_DEVICE))
			continue;

		if (hci_dev_test_flag(d, HCI_UNCONFIGURED))
			rp->entry[count].type = 0x01;
		else
			rp->entry[count].type = 0x00;

		rp->entry[count].bus = d->bus;
		rp->entry[count++].index = cpu_to_le16(d->id);
		bt_dev_dbg(hdev, "Added hci%u", d->id);
	}

	rp->num_controllers = cpu_to_le16(count);

	read_unlock(&hci_dev_list_lock);

	/* If this command is called at least once, then all the
	 * default index and unconfigured index events are disabled
	 * and from now on only extended index events are used.
	 */
	hci_sock_set_flag(sk, HCI_MGMT_EXT_INDEX_EVENTS);
	hci_sock_clear_flag(sk, HCI_MGMT_INDEX_EVENTS);
	hci_sock_clear_flag(sk, HCI_MGMT_UNCONF_INDEX_EVENTS);

	err = mgmt_cmd_complete(sk, MGMT_INDEX_NONE,
				MGMT_OP_READ_EXT_INDEX_LIST, 0, rp,
				struct_size(rp, entry, count));

	kfree(rp);

	return err;
}
/**
 * @brief Checks if a device is fully configured.
 * @param hdev The HCI device.
 * @return True if the device is configured, false otherwise.
 *
 * A device is considered configured if it does not require external
 * configuration or a public address, or if those have been provided.
 */
static bool is_configured(struct hci_dev *hdev)
{
	if (hci_test_quirk(hdev, HCI_QUIRK_EXTERNAL_CONFIG) &&
	    !hci_dev_test_flag(hdev, HCI_EXT_CONFIGURED))
		return false;

	if ((hci_test_quirk(hdev, HCI_QUIRK_INVALID_BDADDR) ||
	     hci_test_quirk(hdev, HCI_QUIRK_USE_BDADDR_PROPERTY)) &&
	    !bacmp(&hdev->public_addr, BDADDR_ANY))
		return false;

	return true;
}
/**
 * @brief Gets a bitmask of missing configuration options for a device.
 * @param hdev The HCI device.
 * @return A bitmask of missing MGMT_OPTION flags.
 */
static __le32 get_missing_options(struct hci_dev *hdev)
{
	u32 options = 0;

	if (hci_test_quirk(hdev, HCI_QUIRK_EXTERNAL_CONFIG) &&
	    !hci_dev_test_flag(hdev, HCI_EXT_CONFIGURED))
		options |= MGMT_OPTION_EXTERNAL_CONFIG;

	if ((hci_test_quirk(hdev, HCI_QUIRK_INVALID_BDADDR) ||
	     hci_test_quirk(hdev, HCI_QUIRK_USE_BDADDR_PROPERTY)) &&
	    !bacmp(&hdev->public_addr, BDADDR_ANY))
		options |= MGMT_OPTION_PUBLIC_ADDRESS;

	return cpu_to_le32(options);
}
/**
 * @brief Sends a MGMT_EV_NEW_CONFIG_OPTIONS event.
 * @param hdev The HCI device.
 * @param skip The socket to skip.
 * @return 0 on success, or a negative error code on failure.
 */
static int new_options(struct hci_dev *hdev, struct sock *skip)
{
	__le32 options = get_missing_options(hdev);

	return mgmt_limited_event(MGMT_EV_NEW_CONFIG_OPTIONS, hdev, &options,
				  sizeof(options), HCI_MGMT_OPTION_EVENTS, skip);
}
/**
 * @brief Sends a command complete response with missing configuration options.
 * @param sk The socket to send the response to.
 * @param opcode The opcode of the command.
 * @param hdev The HCI device.
 * @return 0 on success, or a negative error code on failure.
 */
static int send_options_rsp(struct sock *sk, u16 opcode, struct hci_dev *hdev)
{
	__le32 options = get_missing_options(hdev);

	return mgmt_cmd_complete(sk, hdev->id, opcode, 0, &options,
				 sizeof(options));
}

/**
 * @brief Handles the MGMT_OP_READ_CONFIG_INFO command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data (unused).
 * @param data_len The length of the command data (unused).
 * @return 0 on success, or a negative error code on failure.
 *
 * This function returns information about the configuration options of a device,
 * including supported and missing options.
 */
static int read_config_info(struct sock *sk, struct hci_dev *hdev,
			    void *data, u16 data_len)
{
	struct mgmt_rp_read_config_info rp;
	u32 options = 0;

	bt_dev_dbg(hdev, "sock %p", sk);

	hci_dev_lock(hdev);

	memset(&rp, 0, sizeof(rp));
	rp.manufacturer = cpu_to_le16(hdev->manufacturer);

	if (hci_test_quirk(hdev, HCI_QUIRK_EXTERNAL_CONFIG))
		options |= MGMT_OPTION_EXTERNAL_CONFIG;

	if (hdev->set_bdaddr)
		options |= MGMT_OPTION_PUBLIC_ADDRESS;

	rp.supported_options = cpu_to_le32(options);
	rp.missing_options = get_missing_options(hdev);

	hci_dev_unlock(hdev);

	return mgmt_cmd_complete(sk, hdev->id, MGMT_OP_READ_CONFIG_INFO, 0,
				 &rp, sizeof(rp));
}
/**
 * @brief Gets a bitmask of supported PHYs for a device.
 * @param hdev The HCI device.
 * @return A bitmask of supported MGMT_PHY flags.
 */
static u32 get_supported_phys(struct hci_dev *hdev)
{
	u32 supported_phys = 0;

	if (lmp_bredr_capable(hdev)) {
		supported_phys |= MGMT_PHY_BR_1M_1SLOT;

		if (hdev->features[0][0] & LMP_3SLOT)
			supported_phys |= MGMT_PHY_BR_1M_3SLOT;

		if (hdev->features[0][0] & LMP_5SLOT)
			supported_phys |= MGMT_PHY_BR_1M_5SLOT;

		if (lmp_edr_2m_capable(hdev)) {
			supported_phys |= MGMT_PHY_EDR_2M_1SLOT;

			if (lmp_edr_3slot_capable(hdev))
				supported_phys |= MGMT_PHY_EDR_2M_3SLOT;

			if (lmp_edr_5slot_capable(hdev))
				supported_phys |= MGMT_PHY_EDR_2M_5SLOT;

			if (lmp_edr_3m_capable(hdev)) {
				supported_phys |= MGMT_PHY_EDR_3M_1SLOT;

				if (lmp_edr_3slot_capable(hdev))
					supported_phys |= MGMT_PHY_EDR_3M_3SLOT;

				if (lmp_edr_5slot_capable(hdev))
					supported_phys |= MGMT_PHY_EDR_3M_5SLOT;
			}
		}
	}

	if (lmp_le_capable(hdev)) {
		supported_phys |= MGMT_PHY_LE_1M_TX;
		supported_phys |= MGMT_PHY_LE_1M_RX;

		if (hdev->le_features[1] & HCI_LE_PHY_2M) {
			supported_phys |= MGMT_PHY_LE_2M_TX;
			supported_phys |= MGMT_PHY_LE_2M_RX;
		}

		if (hdev->le_features[1] & HCI_LE_PHY_CODED) {
			supported_phys |= MGMT_PHY_LE_CODED_TX;
			supported_phys |= MGMT_PHY_LE_CODED_RX;
		}
	}

	return supported_phys;
}
/**
 * @brief Gets a bitmask of currently selected PHYs for a device.
 * @param hdev The HCI device.
 * @return A bitmask of selected MGMT_PHY flags.
 */
static u32 get_selected_phys(struct hci_dev *hdev)
{
	u32 selected_phys = 0;

	if (lmp_bredr_capable(hdev)) {
		selected_phys |= MGMT_PHY_BR_1M_1SLOT;

		if (hdev->pkt_type & (HCI_DM3 | HCI_DH3))
			selected_phys |= MGMT_PHY_BR_1M_3SLOT;

		if (hdev->pkt_type & (HCI_DM5 | HCI_DH5))
			selected_phys |= MGMT_PHY_BR_1M_5SLOT;

		if (lmp_edr_2m_capable(hdev)) {
			if (!(hdev->pkt_type & HCI_2DH1))
				selected_phys |= MGMT_PHY_EDR_2M_1SLOT;

			if (lmp_edr_3slot_capable(hdev) &&
			    !(hdev->pkt_type & HCI_2DH3))
				selected_phys |= MGMT_PHY_EDR_2M_3SLOT;

			if (lmp_edr_5slot_capable(hdev) &&
			    !(hdev->pkt_type & HCI_2DH5))
				selected_phys |= MGMT_PHY_EDR_2M_5SLOT;

			if (lmp_edr_3m_capable(hdev)) {
				if (!(hdev->pkt_type & HCI_3DH1))
					selected_phys |= MGMT_PHY_EDR_3M_1SLOT;

				if (lmp_edr_3slot_capable(hdev) &&
				    !(hdev->pkt_type & HCI_3DH3))
					selected_phys |= MGMT_PHY_EDR_3M_3SLOT;

				if (lmp_edr_5slot_capable(hdev) &&
				    !(hdev->pkt_type & HCI_3DH5))
					selected_phys |= MGMT_PHY_EDR_3M_5SLOT;
			}
		}
	}

	if (lmp_le_capable(hdev)) {
		if (hdev->le_tx_def_phys & HCI_LE_SET_PHY_1M)
			selected_phys |= MGMT_PHY_LE_1M_TX;

		if (hdev->le_rx_def_phys & HCI_LE_SET_PHY_1M)
			selected_phys |= MGMT_PHY_LE_1M_RX;

		if (hdev->le_tx_def_phys & HCI_LE_SET_PHY_2M)
			selected_phys |= MGMT_PHY_LE_2M_TX;

		if (hdev->le_rx_def_phys & HCI_LE_SET_PHY_2M)
			selected_phys |= MGMT_PHY_LE_2M_RX;

		if (hdev->le_tx_def_phys & HCI_LE_SET_PHY_CODED)
			selected_phys |= MGMT_PHY_LE_CODED_TX;

		if (hdev->le_rx_def_phys & HCI_LE_SET_PHY_CODED)
			selected_phys |= MGMT_PHY_LE_CODED_RX;
	}

	return selected_phys;
}
/**
 * @brief Gets a bitmask of configurable PHYs for a device.
 * @param hdev The HCI device.
 * @return A bitmask of configurable MGMT_PHY flags.
 */
static u32 get_configurable_phys(struct hci_dev *hdev)
{
	return (get_supported_phys(hdev) & ~MGMT_PHY_BR_1M_1SLOT &
		~MGMT_PHY_LE_1M_TX & ~MGMT_PHY_LE_1M_RX);
}

/**
 * @brief Gets a bitmask of supported settings for a device.
 * @param hdev The HCI device.
 * @return A bitmask of supported MGMT_SETTING flags.
 */
static u32 get_supported_settings(struct hci_dev *hdev)
{
	u32 settings = 0;

	settings |= MGMT_SETTING_POWERED;
	settings |= MGMT_SETTING_BONDABLE;
	settings |= MGMT_SETTING_DEBUG_KEYS;
	settings |= MGMT_SETTING_CONNECTABLE;
	settings |= MGMT_SETTING_DISCOVERABLE;

	if (lmp_bredr_capable(hdev)) {
		if (hdev->hci_ver >= BLUETOOTH_VER_1_2)
			settings |= MGMT_SETTING_FAST_CONNECTABLE;
		settings |= MGMT_SETTING_BREDR;
		settings |= MGMT_SETTING_LINK_SECURITY;

		if (lmp_ssp_capable(hdev)) {
			settings |= MGMT_SETTING_SSP;
		}

		if (lmp_sc_capable(hdev))
			settings |= MGMT_SETTING_SECURE_CONN;

		if (hci_test_quirk(hdev, HCI_QUIRK_WIDEBAND_SPEECH_SUPPORTED))
			settings |= MGMT_SETTING_WIDEBAND_SPEECH;
	}

	if (lmp_le_capable(hdev)) {
		settings |= MGMT_SETTING_LE;
		settings |= MGMT_SETTING_SECURE_CONN;
		settings |= MGMT_SETTING_PRIVACY;
		settings |= MGMT_SETTING_STATIC_ADDRESS;
		settings |= MGMT_SETTING_ADVERTISING;
	}

	if (hci_test_quirk(hdev, HCI_QUIRK_EXTERNAL_CONFIG) || hdev->set_bdaddr)
		settings |= MGMT_SETTING_CONFIGURATION;

	if (cis_central_capable(hdev))
		settings |= MGMT_SETTING_CIS_CENTRAL;

	if (cis_peripheral_capable(hdev))
		settings |= MGMT_SETTING_CIS_PERIPHERAL;

	if (ll_privacy_capable(hdev))
		settings |= MGMT_SETTING_LL_PRIVACY;

	settings |= MGMT_SETTING_PHY_CONFIGURATION;

	return settings;
}
/**
 * @brief Gets a bitmask of current settings for a device.
 * @param hdev The HCI device.
 * @return A bitmask of current MGMT_SETTING flags.
 */
static u32 get_current_settings(struct hci_dev *hdev)
{
	u32 settings = 0;

	if (hdev_is_powered(hdev))
		settings |= MGMT_SETTING_POWERED;

	if (hci_dev_test_flag(hdev, HCI_CONNECTABLE))
		settings |= MGMT_SETTING_CONNECTABLE;

	if (hci_dev_test_flag(hdev, HCI_FAST_CONNECTABLE))
		settings |= MGMT_SETTING_FAST_CONNECTABLE;

	if (hci_dev_test_flag(hdev, HCI_DISCOVERABLE))
		settings |= MGMT_SETTING_DISCOVERABLE;

	if (hci_dev_test_flag(hdev, HCI_BONDABLE))
		settings |= MGMT_SETTING_BONDABLE;

	if (hci_dev_test_flag(hdev, HCI_BREDR_ENABLED))
		settings |= MGMT_SETTING_BREDR;

	if (hci_dev_test_flag(hdev, HCI_LE_ENABLED))
		settings |= MGMT_SETTING_LE;

	if (hci_dev_test_flag(hdev, HCI_LINK_SECURITY))
		settings |= MGMT_SETTING_LINK_SECURITY;

	if (hci_dev_test_flag(hdev, HCI_SSP_ENABLED))
		settings |= MGMT_SETTING_SSP;

	if (hci_dev_test_flag(hdev, HCI_ADVERTISING))
		settings |= MGMT_SETTING_ADVERTISING;

	if (hci_dev_test_flag(hdev, HCI_SC_ENABLED))
		settings |= MGMT_SETTING_SECURE_CONN;

	if (hci_dev_test_flag(hdev, HCI_KEEP_DEBUG_KEYS))
		settings |= MGMT_SETTING_DEBUG_KEYS;

	if (hci_dev_test_flag(hdev, HCI_PRIVACY))
		settings |= MGMT_SETTING_PRIVACY;

	/* The current setting for static address has two purposes. The
	 * first is to indicate if the static address will be used and
	 * the second is to indicate if it is actually set.
	 *
	 * This means if the static address is not configured, this flag
	 * will never be set. If the address is configured, then if the
	 * address is actually used decides if the flag is set or not.
	 *
	 * For single mode LE only controllers and dual-mode controllers
	 * with BR/EDR disabled, the existence of the static address will
	 * be evaluated.
	 */
	if (hci_dev_test_flag(hdev, HCI_FORCE_STATIC_ADDR) ||
	    !hci_dev_test_flag(hdev, HCI_BREDR_ENABLED) ||
	    !bacmp(&hdev->bdaddr, BDADDR_ANY)) {
		if (bacmp(&hdev->static_addr, BDADDR_ANY))
			settings |= MGMT_SETTING_STATIC_ADDRESS;
	}

	if (hci_dev_test_flag(hdev, HCI_WIDEBAND_SPEECH_ENABLED))
		settings |= MGMT_SETTING_WIDEBAND_SPEECH;

	if (cis_central_capable(hdev))
		settings |= MGMT_SETTING_CIS_CENTRAL;

	if (cis_peripheral_capable(hdev))
		settings |= MGMT_SETTING_CIS_PERIPHERAL;

	if (bis_capable(hdev))
		settings |= MGMT_SETTING_ISO_BROADCASTER;

	if (sync_recv_capable(hdev))
		settings |= MGMT_SETTING_ISO_SYNC_RECEIVER;

	if (ll_privacy_capable(hdev))
		settings |= MGMT_SETTING_LL_PRIVACY;

	return settings;
}
/**
 * @brief Finds a pending management command for a given HCI device and opcode.
 * @param opcode The opcode of the command to find.
 * @param hdev The HCI device.
 * @return A pointer to the pending command, or NULL if not found.
 */
static struct mgmt_pending_cmd *pending_find(u16 opcode, struct hci_dev *hdev)
{
	return mgmt_pending_find(HCI_CHANNEL_CONTROL, opcode, hdev);
}

/**
 * @brief Gets the advertising discovery flags for a device.
 * @param hdev The HCI device.
 * @return The LE advertising flags.
 */
u8 mgmt_get_adv_discov_flags(struct hci_dev *hdev)
{
	struct mgmt_pending_cmd *cmd;

	/* If there's a pending mgmt command the flags will not yet have
	 * their final values, so check for this first.
	 */
	cmd = pending_find(MGMT_OP_SET_DISCOVERABLE, hdev);
	if (cmd) {
		struct mgmt_mode *cp = cmd->param;
		if (cp->val == 0x01)
			return LE_AD_GENERAL;
		else if (cp->val == 0x02)
			return LE_AD_LIMITED;
	} else {
		if (hci_dev_test_flag(hdev, HCI_LIMITED_DISCOVERABLE))
			return LE_AD_LIMITED;
		else if (hci_dev_test_flag(hdev, HCI_DISCOVERABLE))
			return LE_AD_GENERAL;
	}

	return 0;
}
/**
 * @brief Gets the connectable state of a device.
 * @param hdev The HCI device.
 * @return True if the device is connectable, false otherwise.
 */
bool mgmt_get_connectable(struct hci_dev *hdev)
{
	struct mgmt_pending_cmd *cmd;

	/* If there's a pending mgmt command the flag will not yet have
	 * it's final value, so check for this first.
	 */
	cmd = pending_find(MGMT_OP_SET_CONNECTABLE, hdev);
	if (cmd) {
		struct mgmt_mode *cp = cmd->param;

		return cp->val;
	}

	return hci_dev_test_flag(hdev, HCI_CONNECTABLE);
}

/**
 * @brief Synchronizes the service cache by updating EIR and class of device.
 * @param hdev The HCI device.
 * @param data Unused.
 * @return 0 on success.
 */
static int service_cache_sync(struct hci_dev *hdev, void *data)
{
	hci_update_eir_sync(hdev);
	hci_update_class_sync(hdev);

	return 0;
}
/**
 * @brief Work function to turn off the service cache.
 * @param work The work struct.
 */
static void service_cache_off(struct work_struct *work)
{
	struct hci_dev *hdev = container_of(work, struct hci_dev,
					    service_cache.work);

	if (!hci_dev_test_and_clear_flag(hdev, HCI_SERVICE_CACHE))
		return;

	hci_cmd_sync_queue(hdev, service_cache_sync, NULL, NULL);
}

/**
 * @brief Synchronizes advertising when the RPA expires.
 * @param hdev The HCI device.
 * @param data Unused.
 * @return 0 on success.
 */
static int rpa_expired_sync(struct hci_dev *hdev, void *data)
{
	/* The generation of a new RPA and programming it into the
	 * controller happens in the hci_req_enable_advertising()
	 * function.
	 */
	if (ext_adv_capable(hdev))
		return hci_start_ext_adv_sync(hdev, hdev->cur_adv_instance);
	else
		return hci_enable_advertising_sync(hdev);
}
/**
 * @brief Work function to handle RPA expiration.
 * @param work The work struct.
 */
static void rpa_expired(struct work_struct *work)
{
	struct hci_dev *hdev = container_of(work, struct hci_dev,
					    rpa_expired.work);

	bt_dev_dbg(hdev, "");

	hci_dev_set_flag(hdev, HCI_RPA_EXPIRED);

	if (!hci_dev_test_flag(hdev, HCI_ADVERTISING))
		return;

	hci_cmd_sync_queue(hdev, rpa_expired_sync, NULL, NULL);
}

static int set_discoverable_sync(struct hci_dev *hdev, void *data);

/**
 * @brief Work function to turn off discoverability.
 * @param work The work struct.
 */
static void discov_off(struct work_struct *work)
{
	struct hci_dev *hdev = container_of(work, struct hci_dev,
					    discov_off.work);

	bt_dev_dbg(hdev, "");

	hci_dev_lock(hdev);

	/* When discoverable timeout triggers, then just make sure
	 * the limited discoverable flag is cleared. Even in the case
	 * of a timeout triggered from general discoverable, it is
	 * safe to unconditionally clear the flag.
	 */
	hci_dev_clear_flag(hdev, HCI_LIMITED_DISCOVERABLE);
	hci_dev_clear_flag(hdev, HCI_DISCOVERABLE);
	hdev->discov_timeout = 0;

	hci_cmd_sync_queue(hdev, set_discoverable_sync, NULL, NULL);

	mgmt_new_settings(hdev);

	hci_dev_unlock(hdev);
}

static int send_settings_rsp(struct sock *sk, u16 opcode, struct hci_dev *hdev);

/**
 * @brief Completes a mesh send operation.
 * @param hdev The HCI device.
 * @param mesh_tx The mesh transmission structure.
 * @param silent True to suppress the completion event.
 */
static void mesh_send_complete(struct hci_dev *hdev,
			       struct mgmt_mesh_tx *mesh_tx, bool silent)
{
	u8 handle = mesh_tx->handle;

	if (!silent)
		mgmt_event(MGMT_EV_MESH_PACKET_CMPLT, hdev, &handle,
			   sizeof(handle), NULL);

	mgmt_mesh_remove(mesh_tx);
}

/**
 * @brief Synchronizes the mesh send done state.
 * @param hdev The HCI device.
 * @param data Unused.
 * @return 0 on success.
 */
static int mesh_send_done_sync(struct hci_dev *hdev, void *data)
{
	struct mgmt_mesh_tx *mesh_tx;

	hci_dev_clear_flag(hdev, HCI_MESH_SENDING);
	if (list_empty(&hdev->adv_instances))
		hci_disable_advertising_sync(hdev);
	mesh_tx = mgmt_mesh_next(hdev, NULL);

	if (mesh_tx)
		mesh_send_complete(hdev, mesh_tx, false);

	return 0;
}

static int mesh_send_sync(struct hci_dev *hdev, void *data);
static void mesh_send_start_complete(struct hci_dev *hdev, void *data, int err);
/**
 * @brief Handles the next mesh transmission.
 * @param hdev The HCI device.
 * @param data Unused.
 * @param err The error code from the previous operation.
 */
static void mesh_next(struct hci_dev *hdev, void *data, int err)
{
	struct mgmt_mesh_tx *mesh_tx = mgmt_mesh_next(hdev, NULL);

	if (!mesh_tx)
		return;

	err = hci_cmd_sync_queue(hdev, mesh_send_sync, mesh_tx,
				 mesh_send_start_complete);

	if (err < 0)
		mesh_send_complete(hdev, mesh_tx, false);
	else
		hci_dev_set_flag(hdev, HCI_MESH_SENDING);
}

/**
 * @brief Work function to handle mesh send completion.
 * @param work The work struct.
 */
static void mesh_send_done(struct work_struct *work)
{
	struct hci_dev *hdev = container_of(work, struct hci_dev,
					    mesh_send_done.work);

	if (!hci_dev_test_flag(hdev, HCI_MESH_SENDING))
		return;

	hci_cmd_sync_queue(hdev, mesh_send_done_sync, NULL, mesh_next);
}

/**
 * @brief Initializes management-specific data for an HCI device.
 * @param sk The socket associated with the initialization.
 * @param hdev The HCI device to initialize.
 */
static void mgmt_init_hdev(struct sock *sk, struct hci_dev *hdev)
{
	if (hci_dev_test_flag(hdev, HCI_MGMT))
		return;

	BT_INFO("MGMT ver %d.%d", MGMT_VERSION, MGMT_REVISION);

	INIT_DELAYED_WORK(&hdev->discov_off, discov_off);
	INIT_DELAYED_WORK(&hdev->service_cache, service_cache_off);
	INIT_DELAYED_WORK(&hdev->rpa_expired, rpa_expired);
	INIT_DELAYED_WORK(&hdev->mesh_send_done, mesh_send_done);

	/* Non-mgmt controlled devices get this bit set
	 * implicitly so that pairing works for them, however
	 * for mgmt we require user-space to explicitly enable
	 * it
	 */
	hci_dev_clear_flag(hdev, HCI_BONDABLE);

	hci_dev_set_flag(hdev, HCI_MGMT);
}

/**
 * @brief Handles the MGMT_OP_READ_INFO command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data (unused).
 * @param data_len The length of the command data (unused).
 * @return 0 on success, or a negative error code on failure.
 *
 * This function returns information about the controller, including its
 * address, version, manufacturer, and current settings.
 */
static int read_controller_info(struct sock *sk, struct hci_dev *hdev,
				void *data, u16 data_len)
{
	struct mgmt_rp_read_info rp;

	bt_dev_dbg(hdev, "sock %p", sk);

	hci_dev_lock(hdev);

	memset(&rp, 0, sizeof(rp));

	bacpy(&rp.bdaddr, &hdev->bdaddr);

	rp.version = hdev->hci_ver;
	rp.manufacturer = cpu_to_le16(hdev->manufacturer);

	rp.supported_settings = cpu_to_le32(get_supported_settings(hdev));
	rp.current_settings = cpu_to_le32(get_current_settings(hdev));

	memcpy(rp.dev_class, hdev->dev_class, 3);

	memcpy(rp.name, hdev->dev_name, sizeof(hdev->dev_name));
	memcpy(rp.short_name, hdev->short_name, sizeof(hdev->short_name));

	hci_dev_unlock(hdev);

	return mgmt_cmd_complete(sk, hdev->id, MGMT_OP_READ_INFO, 0, &rp,
				 sizeof(rp));
}
/**
 * @brief Appends EIR data to a buffer.
 * @param hdev The HCI device.
 * @param eir The buffer to append to.
 * @return The length of the appended data.
 */
static u16 append_eir_data_to_buf(struct hci_dev *hdev, u8 *eir)
{
	u16 eir_len = 0;
	size_t name_len;

	if (hci_dev_test_flag(hdev, HCI_BREDR_ENABLED))
		eir_len = eir_append_data(eir, eir_len, EIR_CLASS_OF_DEV,
					  hdev->dev_class, 3);

	if (hci_dev_test_flag(hdev, HCI_LE_ENABLED))
		eir_len = eir_append_le16(eir, eir_len, EIR_APPEARANCE,
					  hdev->appearance);

	name_len = strnlen(hdev->dev_name, sizeof(hdev->dev_name));
	eir_len = eir_append_data(eir, eir_len, EIR_NAME_COMPLETE,
				  hdev->dev_name, name_len);

	name_len = strnlen(hdev->short_name, sizeof(hdev->short_name));
	eir_len = eir_append_data(eir, eir_len, EIR_NAME_SHORT,
				  hdev->short_name, name_len);

	return eir_len;
}
/**
 * @brief Handles the MGMT_OP_READ_EXT_INFO command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data (unused).
 * @param data_len The length of the command data (unused).
 * @return 0 on success, or a negative error code on failure.
 *
 * This function returns extended information about the controller, including
 * EIR data.
 */
static int read_ext_controller_info(struct sock *sk, struct hci_dev *hdev,
				    void *data, u16 data_len)
{
	char buf[512];
	struct mgmt_rp_read_ext_info *rp = (void *)buf;
	u16 eir_len;

	bt_dev_dbg(hdev, "sock %p", sk);

	memset(&buf, 0, sizeof(buf));

	hci_dev_lock(hdev);

	bacpy(&rp->bdaddr, &hdev->bdaddr);

	rp->version = hdev->hci_ver;
	rp->manufacturer = cpu_to_le16(hdev->manufacturer);

	rp->supported_settings = cpu_to_le32(get_supported_settings(hdev));
	rp->current_settings = cpu_to_le32(get_current_settings(hdev));


	eir_len = append_eir_data_to_buf(hdev, rp->eir);
	rp->eir_len = cpu_to_le16(eir_len);

	hci_dev_unlock(hdev);

	/* If this command is called at least once, then the events
	 * for class of device and local name changes are disabled
	 * and only the new extended controller information event
	 * is used.
	 */
	hci_sock_set_flag(sk, HCI_MGMT_EXT_INFO_EVENTS);
	hci_sock_clear_flag(sk, HCI_MGMT_DEV_CLASS_EVENTS);
	hci_sock_clear_flag(sk, HCI_MGMT_LOCAL_NAME_EVENTS);

	return mgmt_cmd_complete(sk, hdev->id, MGMT_OP_READ_EXT_INFO, 0, rp,
				 sizeof(*rp) + eir_len);
}
/**
 * @brief Sends a MGMT_EV_EXT_INFO_CHANGED event.
 * @param hdev The HCI device.
 * @param skip The socket to skip.
 * @return 0 on success, or a negative error code on failure.
 */
static int ext_info_changed(struct hci_dev *hdev, struct sock *skip)
{
	char buf[512];
	struct mgmt_ev_ext_info_changed *ev = (void *)buf;
	u16 eir_len;

	memset(buf, 0, sizeof(buf));

	eir_len = append_eir_data_to_buf(hdev, ev->eir);
	ev->eir_len = cpu_to_le16(eir_len);

	return mgmt_limited_event(MGMT_EV_EXT_INFO_CHANGED, hdev, ev,
				  sizeof(*ev) + eir_len,
				  HCI_MGMT_EXT_INFO_EVENTS, skip);
}
/**
 * @brief Sends a command complete response with current settings.
 * @param sk The socket to send the response to.
 * @param opcode The opcode of the command.
 * @param hdev The HCI device.
 * @return 0 on success, or a negative error code on failure.
 */
static int send_settings_rsp(struct sock *sk, u16 opcode, struct hci_dev *hdev)
{
	__le32 settings = cpu_to_le32(get_current_settings(hdev));

	return mgmt_cmd_complete(sk, hdev->id, opcode, 0, &settings,
				 sizeof(settings));
}
/**
 * @brief Sends a MGMT_EV_ADVERTISING_ADDED event.
 * @param sk The socket that initiated the command.
 * @param hdev The HCI device.
 * @param instance The advertising instance that was added.
 */
void mgmt_advertising_added(struct sock *sk, struct hci_dev *hdev, u8 instance)
{
	struct mgmt_ev_advertising_added ev;

	ev.instance = instance;

	mgmt_event(MGMT_EV_ADVERTISING_ADDED, hdev, &ev, sizeof(ev), sk);
}
/**
 * @brief Sends a MGMT_EV_ADVERTISING_REMOVED event.
 * @param sk The socket that initiated the command.
 * @param hdev The HCI device.
 * @param instance The advertising instance that was removed.
 */
void mgmt_advertising_removed(struct sock *sk, struct hci_dev *hdev,
			      u8 instance)
{
	struct mgmt_ev_advertising_removed ev;

	ev.instance = instance;

	mgmt_event(MGMT_EV_ADVERTISING_REMOVED, hdev, &ev, sizeof(ev), sk);
}
/**
 * @brief Cancels the advertising timeout work.
 * @param hdev The HCI device.
 */
static void cancel_adv_timeout(struct hci_dev *hdev)
{
	if (hdev->adv_instance_timeout) {
		hdev->adv_instance_timeout = 0;
		cancel_delayed_work(&hdev->adv_instance_expire);
	}
}

/**
 * @brief Restarts LE actions for a device.
 * @param hdev The HCI device.
 *
 * This function is called after powering on a device to re-initiate
 * pending LE connections and reports.
 */
static void restart_le_actions(struct hci_dev *hdev)
{
	struct hci_conn_params *p;

	list_for_each_entry(p, &hdev->le_conn_params, list) {
		/* Needed for AUTO_OFF case where might not "really"
		 * have been powered off.
		 */
		hci_pend_le_list_del_init(p);

		switch (p->auto_connect) {
		case HCI_AUTO_CONN_DIRECT:
		case HCI_AUTO_CONN_ALWAYS:
			hci_pend_le_list_add(p, &hdev->pend_le_conns);
			break;
		case HCI_AUTO_CONN_REPORT:
			hci_pend_le_list_add(p, &hdev->pend_le_reports);
			break;
		default:
			break;
		}
	}
}
/**
 * @brief Sends a MGMT_EV_NEW_SETTINGS event.
 * @param hdev The HCI device.
 * @param skip The socket to skip.
 * @return 0 on success, or a negative error code on failure.
 */
int mgmt_new_settings(struct hci_dev *hdev, struct sock *skip)
{
	__le32 ev = cpu_to_le32(get_current_settings(hdev));

	return mgmt_limited_event(MGMT_EV_NEW_SETTINGS, hdev, &ev,
				  sizeof(ev), HCI_MGMT_SETTING_EVENTS, skip);
}

/**
 * @brief Handles the completion of the set_powered command.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @param err The error code from the HCI command.
 */
static void mgmt_set_powered_complete(struct hci_dev *hdev, void *data, int err)
{
	struct mgmt_pending_cmd *cmd = data;
	struct mgmt_mode *cp;

	/* Make sure cmd still outstanding. */
	if (err == -ECANCELED ||
	    cmd != pending_find(MGMT_OP_SET_POWERED, hdev))
		return;

	cp = cmd->param;

	bt_dev_dbg(hdev, "err %d", err);

	if (!err) {
		if (cp->val) {
			hci_dev_lock(hdev);
			restart_le_actions(hdev);
			hci_update_passive_scan(hdev);
			hci_dev_unlock(hdev);
		}

		send_settings_rsp(cmd->sk, cmd->opcode, hdev);

		/* Only call new_setting for power on as power off is deferred
		 * to hdev->power_off work which does call hci_dev_do_close.
		 */
		if (cp->val)
			new_settings(hdev, cmd->sk);
	} else {
		mgmt_cmd_status(cmd->sk, hdev->id, MGMT_OP_SET_POWERED,
				mgmt_status(err));
	}

	mgmt_pending_remove(cmd);
}

/**
 * @brief Synchronously sets the powered state of a device.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_powered_sync(struct hci_dev *hdev, void *data)
{
	struct mgmt_pending_cmd *cmd = data;
	struct mgmt_mode *cp;

	/* Make sure cmd still outstanding. */
	if (cmd != pending_find(MGMT_OP_SET_POWERED, hdev))
		return -ECANCELED;

	cp = cmd->param;

	BT_DBG("%s", hdev->name);

	return hci_set_powered_sync(hdev, cp->val);
}

/**
 * @brief Handles the MGMT_OP_SET_POWERED command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_powered(struct sock *sk, struct hci_dev *hdev, void *data,
		       u16 len)
{
	struct mgmt_mode *cp = data;
	struct mgmt_pending_cmd *cmd;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	if (cp->val != 0x00 && cp->val != 0x01)
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_POWERED,
				       MGMT_STATUS_INVALID_PARAMS);

	hci_dev_lock(hdev);

	if (!cp->val) {
		if (hci_dev_test_flag(hdev, HCI_POWERING_DOWN)) {
			err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_POWERED,
					      MGMT_STATUS_BUSY);
			goto failed;
		}
	}

	if (pending_find(MGMT_OP_SET_POWERED, hdev)) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_POWERED,
				      MGMT_STATUS_BUSY);
		goto failed;
	}

	if (!!cp->val == hdev_is_powered(hdev)) {
		err = send_settings_rsp(sk, MGMT_OP_SET_POWERED, hdev);
		goto failed;
	}

	cmd = mgmt_pending_add(sk, MGMT_OP_SET_POWERED, hdev, data, len);
	if (!cmd) {
		err = -ENOMEM;
		goto failed;
	}

	/* Cancel potentially blocking sync operation before power off */
	if (cp->val == 0x00) {
		hci_cmd_sync_cancel_sync(hdev, -EHOSTDOWN);
		err = hci_cmd_sync_queue(hdev, set_powered_sync, cmd,
					 mgmt_set_powered_complete);
	} else {
		/* Use hci_cmd_sync_submit since hdev might not be running */
		err = hci_cmd_sync_submit(hdev, set_powered_sync, cmd,
					  mgmt_set_powered_complete);
	}

	if (err < 0)
		mgmt_pending_remove(cmd);

failed:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Handles the completion of the set_discoverable command.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @param err The error code from the HCI command.
 */
static void mgmt_set_discoverable_complete(struct hci_dev *hdev, void *data,
					   int err)
{
	struct mgmt_pending_cmd *cmd = data;

	bt_dev_dbg(hdev, "err %d", err);

	/* Make sure cmd still outstanding. */
	if (err == -ECANCELED ||
	    cmd != pending_find(MGMT_OP_SET_DISCOVERABLE, hdev))
		return;

	hci_dev_lock(hdev);

	if (err) {
		u8 mgmt_err = mgmt_status(err);
		mgmt_cmd_status(cmd->sk, cmd->hdev->id, cmd->opcode, mgmt_err);
		hci_dev_clear_flag(hdev, HCI_LIMITED_DISCOVERABLE);
		goto done;
	}

	if (hci_dev_test_flag(hdev, HCI_DISCOVERABLE) &&
	    hdev->discov_timeout > 0) {
		int to = secs_to_jiffies(hdev->discov_timeout);
		queue_delayed_work(hdev->req_workqueue, &hdev->discov_off, to);
	}

	send_settings_rsp(cmd->sk, MGMT_OP_SET_DISCOVERABLE, hdev);
	new_settings(hdev, cmd->sk);

done:
	mgmt_pending_remove(cmd);
	hci_dev_unlock(hdev);
}
/**
 * @brief Synchronously sets the discoverable state of a device.
 * @param hdev The HCI device.
 * @param data Unused.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_discoverable_sync(struct hci_dev *hdev, void *data)
{
	BT_DBG("%s", hdev->name);

	return hci_update_discoverable_sync(hdev);
}

/**
 * @brief Handles the MGMT_OP_SET_DISCOVERABLE command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_discoverable(struct sock *sk, struct hci_dev *hdev, void *data,
			    u16 len)
{
	struct mgmt_cp_set_discoverable *cp = data;
	struct mgmt_pending_cmd *cmd;
	u16 timeout;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	if (!hci_dev_test_flag(hdev, HCI_LE_ENABLED) &&
	    !hci_dev_test_flag(hdev, HCI_BREDR_ENABLED))
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_DISCOVERABLE,
				       MGMT_STATUS_REJECTED);

	if (cp->val != 0x00 && cp->val != 0x01 && cp->val != 0x02)
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_DISCOVERABLE,
				       MGMT_STATUS_INVALID_PARAMS);

	timeout = __le16_to_cpu(cp->timeout);

	/* Disabling discoverable requires that no timeout is set,
	 * and enabling limited discoverable requires a timeout.
	 */
	if ((cp->val == 0x00 && timeout > 0) ||
	    (cp->val == 0x02 && timeout == 0))
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_DISCOVERABLE,
				       MGMT_STATUS_INVALID_PARAMS);

	hci_dev_lock(hdev);

	if (!hdev_is_powered(hdev) && timeout > 0) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_DISCOVERABLE,
				      MGMT_STATUS_NOT_POWERED);
		goto failed;
	}

	if (pending_find(MGMT_OP_SET_DISCOVERABLE, hdev) ||
	    pending_find(MGMT_OP_SET_CONNECTABLE, hdev)) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_DISCOVERABLE,
				      MGMT_STATUS_BUSY);
		goto failed;
	}

	if (!hci_dev_test_flag(hdev, HCI_CONNECTABLE)) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_DISCOVERABLE,
				      MGMT_STATUS_REJECTED);
		goto failed;
	}

	if (hdev->advertising_paused) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_DISCOVERABLE,
				      MGMT_STATUS_BUSY);
		goto failed;
	}

	if (!hdev_is_powered(hdev)) {
		bool changed = false;

		/* Setting limited discoverable when powered off is
		 * not a valid operation since it requires a timeout
		 * and so no need to check HCI_LIMITED_DISCOVERABLE.
		 */
		if (!!cp->val != hci_dev_test_flag(hdev, HCI_DISCOVERABLE)) {
			hci_dev_change_flag(hdev, HCI_DISCOVERABLE);
			changed = true;
		}

		err = send_settings_rsp(sk, MGMT_OP_SET_DISCOVERABLE, hdev);
		if (err < 0)
			goto failed;

		if (changed)
			err = new_settings(hdev, sk);

		goto failed;
	}

	/* If the current mode is the same, then just update the timeout
	 * value with the new value. And if only the timeout gets updated,
	 * then no need for any HCI transactions.
	 */
	if (!!cp->val == hci_dev_test_flag(hdev, HCI_DISCOVERABLE) &&
	    (cp->val == 0x02) == hci_dev_test_flag(hdev,
						   HCI_LIMITED_DISCOVERABLE)) {
		cancel_delayed_work(&hdev->discov_off);
		hdev->discov_timeout = timeout;

		if (cp->val && hdev->discov_timeout > 0) {
			int to = secs_to_jiffies(hdev->discov_timeout);
			queue_delayed_work(hdev->req_workqueue,
					   &hdev->discov_off, to);
		}

		err = send_settings_rsp(sk, MGMT_OP_SET_DISCOVERABLE, hdev);
		goto failed;
	}

	cmd = mgmt_pending_add(sk, MGMT_OP_SET_DISCOVERABLE, hdev, data, len);
	if (!cmd) {
		err = -ENOMEM;
		goto failed;
	}

	/* Cancel any potential discoverable timeout that might be
	 * still active and store new timeout value. The arming of
	 * the timeout happens in the complete handler.
	 */
	cancel_delayed_work(&hdev->discov_off);
	hdev->discov_timeout = timeout;

	if (cp->val)
		hci_dev_set_flag(hdev, HCI_DISCOVERABLE);
	else
		hci_dev_clear_flag(hdev, HCI_DISCOVERABLE);

	/* Limited discoverable mode */
	if (cp->val == 0x02)
		hci_dev_set_flag(hdev, HCI_LIMITED_DISCOVERABLE);
	else
		hci_dev_clear_flag(hdev, HCI_LIMITED_DISCOVERABLE);

	err = hci_cmd_sync_queue(hdev, set_discoverable_sync, cmd,
				 mgmt_set_discoverable_complete);

	if (err < 0)
		mgmt_pending_remove(cmd);

failed:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Handles the completion of the set_connectable command.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @param err The error code from the HCI command.
 */
static void mgmt_set_connectable_complete(struct hci_dev *hdev, void *data,
					  int err)
{
	struct mgmt_pending_cmd *cmd = data;

	bt_dev_dbg(hdev, "err %d", err);

	/* Make sure cmd still outstanding. */
	if (err == -ECANCELED ||
	    cmd != pending_find(MGMT_OP_SET_CONNECTABLE, hdev))
		return;

	hci_dev_lock(hdev);

	if (err) {
		u8 mgmt_err = mgmt_status(err);
		mgmt_cmd_status(cmd->sk, cmd->hdev->id, cmd->opcode, mgmt_err);
		goto done;
	}

	send_settings_rsp(cmd->sk, MGMT_OP_SET_CONNECTABLE, hdev);
	new_settings(hdev, cmd->sk);

done:
	mgmt_pending_remove(cmd);

	hci_dev_unlock(hdev);
}
/**
 * @brief Updates the connectable setting and notifies listeners.
 * @param hdev The HCI device.
 * @param sk The socket that initiated the change.
 * @param val The new connectable value.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_connectable_update_settings(struct hci_dev *hdev,
					   struct sock *sk, u8 val)
{
	bool changed = false;
	int err;

	if (!!val != hci_dev_test_flag(hdev, HCI_CONNECTABLE))
		changed = true;

	if (val) {
		hci_dev_set_flag(hdev, HCI_CONNECTABLE);
	} else {
		hci_dev_clear_flag(hdev, HCI_CONNECTABLE);
		hci_dev_clear_flag(hdev, HCI_DISCOVERABLE);
	}

	err = send_settings_rsp(sk, MGMT_OP_SET_CONNECTABLE, hdev);
	if (err < 0)
		return err;

	if (changed) {
		hci_update_scan(hdev);
		hci_update_passive_scan(hdev);
		return new_settings(hdev, sk);
	}

	return 0;
}
/**
 * @brief Synchronously sets the connectable state of a device.
 * @param hdev The HCI device.
 * @param data Unused.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_connectable_sync(struct hci_dev *hdev, void *data)
{
	BT_DBG("%s", hdev->name);

	return hci_update_connectable_sync(hdev);
}
/**
 * @brief Handles the MGMT_OP_SET_CONNECTABLE command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_connectable(struct sock *sk, struct hci_dev *hdev, void *data,
			   u16 len)
{
	struct mgmt_mode *cp = data;
	struct mgmt_pending_cmd *cmd;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	if (!hci_dev_test_flag(hdev, HCI_LE_ENABLED) &&
	    !hci_dev_test_flag(hdev, HCI_BREDR_ENABLED))
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_CONNECTABLE,
				       MGMT_STATUS_REJECTED);

	if (cp->val != 0x00 && cp->val != 0x01)
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_CONNECTABLE,
				       MGMT_STATUS_INVALID_PARAMS);

	hci_dev_lock(hdev);

	if (!hdev_is_powered(hdev)) {
		err = set_connectable_update_settings(hdev, sk, cp->val);
		goto failed;
	}

	if (pending_find(MGMT_OP_SET_DISCOVERABLE, hdev) ||
	    pending_find(MGMT_OP_SET_CONNECTABLE, hdev)) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_CONNECTABLE,
				      MGMT_STATUS_BUSY);
		goto failed;
	}

	cmd = mgmt_pending_add(sk, MGMT_OP_SET_CONNECTABLE, hdev, data, len);
	if (!cmd) {
		err = -ENOMEM;
		goto failed;
	}

	if (cp->val) {
		hci_dev_set_flag(hdev, HCI_CONNECTABLE);
	} else {
		if (hdev->discov_timeout > 0)
			cancel_delayed_work(&hdev->discov_off);

		hci_dev_clear_flag(hdev, HCI_LIMITED_DISCOVERABLE);
		hci_dev_clear_flag(hdev, HCI_DISCOVERABLE);
		hci_dev_clear_flag(hdev, HCI_CONNECTABLE);
	}

	err = hci_cmd_sync_queue(hdev, set_connectable_sync, cmd,
				 mgmt_set_connectable_complete);

	if (err < 0)
		mgmt_pending_remove(cmd);

failed:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Handles the MGMT_OP_SET_BONDABLE command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_bondable(struct sock *sk, struct hci_dev *hdev, void *data,
			u16 len)
{
	struct mgmt_mode *cp = data;
	bool changed;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	if (cp->val != 0x00 && cp->val != 0x01)
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_BONDABLE,
				       MGMT_STATUS_INVALID_PARAMS);

	hci_dev_lock(hdev);

	if (cp->val)
		changed = !hci_dev_test_and_set_flag(hdev, HCI_BONDABLE);
	else
		changed = hci_dev_test_and_clear_flag(hdev, HCI_BONDABLE);

	err = send_settings_rsp(sk, MGMT_OP_SET_BONDABLE, hdev);
	if (err < 0)
		goto unlock;

	if (changed) {
		/* In limited privacy mode the change of bondable mode
		 * may affect the local advertising address.
		 */
		hci_update_discoverable(hdev);

		err = new_settings(hdev, sk);
	}

unlock:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Handles the MGMT_OP_SET_LINK_SECURITY command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_link_security(struct sock *sk, struct hci_dev *hdev, void *data,
			     u16 len)
{
	struct mgmt_mode *cp = data;
	struct mgmt_pending_cmd *cmd;
	u8 val, status;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	status = mgmt_bredr_support(hdev);
	if (status)
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_LINK_SECURITY,
				       status);

	if (cp->val != 0x00 && cp->val != 0x01)
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_LINK_SECURITY,
				       MGMT_STATUS_INVALID_PARAMS);

	hci_dev_lock(hdev);

	if (!hdev_is_powered(hdev)) {
		bool changed = false;

		if (!!cp->val != hci_dev_test_flag(hdev, HCI_LINK_SECURITY)) {
			hci_dev_change_flag(hdev, HCI_LINK_SECURITY);
			changed = true;
		}

		err = send_settings_rsp(sk, MGMT_OP_SET_LINK_SECURITY, hdev);
		if (err < 0)
			goto failed;

		if (changed)
			err = new_settings(hdev, sk);

		goto failed;
	}

	if (pending_find(MGMT_OP_SET_LINK_SECURITY, hdev)) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_LINK_SECURITY,
				      MGMT_STATUS_BUSY);
		goto failed;
	}

	val = !!cp->val;

	if (test_bit(HCI_AUTH, &hdev->flags) == val) {
		err = send_settings_rsp(sk, MGMT_OP_SET_LINK_SECURITY, hdev);
		goto failed;
	}

	cmd = mgmt_pending_add(sk, MGMT_OP_SET_LINK_SECURITY, hdev, data, len);
	if (!cmd) {
		err = -ENOMEM;
		goto failed;
	}

	err = hci_send_cmd(hdev, HCI_OP_WRITE_AUTH_ENABLE, sizeof(val), &val);
	if (err < 0) {
		mgmt_pending_remove(cmd);
		goto failed;
	}

failed:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Handles the completion of the set_ssp command.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @param err The error code from the HCI command.
 */
static void set_ssp_complete(struct hci_dev *hdev, void *data, int err)
{
	struct cmd_lookup match = { NULL, hdev };
	struct mgmt_pending_cmd *cmd = data;
	struct mgmt_mode *cp = cmd->param;
	u8 enable = cp->val;
	bool changed;

	/* Make sure cmd still outstanding. */
	if (err == -ECANCELED || cmd != pending_find(MGMT_OP_SET_SSP, hdev))
		return;

	if (err) {
		u8 mgmt_err = mgmt_status(err);

		if (enable && hci_dev_test_and_clear_flag(hdev,
							  HCI_SSP_ENABLED)) {
			new_settings(hdev, NULL);
		}

		mgmt_pending_foreach(MGMT_OP_SET_SSP, hdev, true,
				     cmd_status_rsp, &mgmt_err);
		return;
	}

	if (enable) {
		changed = !hci_dev_test_and_set_flag(hdev, HCI_SSP_ENABLED);
	} else {
		changed = hci_dev_test_and_clear_flag(hdev, HCI_SSP_ENABLED);
	}

	mgmt_pending_foreach(MGMT_OP_SET_SSP, hdev, true, settings_rsp, &match);

	if (changed)
		new_settings(hdev, match.sk);

	if (match.sk)
		sock_put(match.sk);

	hci_update_eir_sync(hdev);
}
/**
 * @brief Synchronously sets the SSP mode of a device.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_ssp_sync(struct hci_dev *hdev, void *data)
{
	struct mgmt_pending_cmd *cmd = data;
	struct mgmt_mode *cp = cmd->param;
	bool changed = false;
	int err;

	if (cp->val)
		changed = !hci_dev_test_and_set_flag(hdev, HCI_SSP_ENABLED);

	err = hci_write_ssp_mode_sync(hdev, cp->val);

	if (!err && changed)
		hci_dev_clear_flag(hdev, HCI_SSP_ENABLED);

	return err;
}
/**
 * @brief Handles the MGMT_OP_SET_SSP command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_ssp(struct sock *sk, struct hci_dev *hdev, void *data, u16 len)
{
	struct mgmt_mode *cp = data;
	struct mgmt_pending_cmd *cmd;
	u8 status;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	status = mgmt_bredr_support(hdev);
	if (status)
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_SSP, status);

	if (!lmp_ssp_capable(hdev))
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_SSP,
				       MGMT_STATUS_NOT_SUPPORTED);

	if (cp->val != 0x00 && cp->val != 0x01)
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_SSP,
				       MGMT_STATUS_INVALID_PARAMS);

	hci_dev_lock(hdev);

	if (!hdev_is_powered(hdev)) {
		bool changed;

		if (cp->val) {
			changed = !hci_dev_test_and_set_flag(hdev,
							     HCI_SSP_ENABLED);
		} else {
			changed = hci_dev_test_and_clear_flag(hdev,
							      HCI_SSP_ENABLED);
		}

		err = send_settings_rsp(sk, MGMT_OP_SET_SSP, hdev);
		if (err < 0)
			goto failed;

		if (changed)
			err = new_settings(hdev, sk);

		goto failed;
	}

	if (pending_find(MGMT_OP_SET_SSP, hdev)) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_SSP,
				      MGMT_STATUS_BUSY);
		goto failed;
	}

	if (!!cp->val == hci_dev_test_flag(hdev, HCI_SSP_ENABLED)) {
		err = send_settings_rsp(sk, MGMT_OP_SET_SSP, hdev);
		goto failed;
	}

	cmd = mgmt_pending_add(sk, MGMT_OP_SET_SSP, hdev, data, len);
	if (!cmd)
		err = -ENOMEM;
	else
		err = hci_cmd_sync_queue(hdev, set_ssp_sync, cmd,
					 set_ssp_complete);

	if (err < 0) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_SSP,
				      MGMT_STATUS_FAILED);

		if (cmd)
			mgmt_pending_remove(cmd);
	}

failed:
	hci_dev_unlock(hdev);
	return err;
}

/**
 * @brief Handles the MGMT_OP_SET_HS command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return MGMT_STATUS_NOT_SUPPORTED as High Speed is not supported.
 */
static int set_hs(struct sock *sk, struct hci_dev *hdev, void *data, u16 len)
{
	bt_dev_dbg(hdev, "sock %p", sk);

	return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_HS,
				       MGMT_STATUS_NOT_SUPPORTED);
}
/**
 * @brief Handles the completion of the set_le command.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @param err The error code from the HCI command.
 */
static void set_le_complete(struct hci_dev *hdev, void *data, int err)
{
	struct cmd_lookup match = { NULL, hdev };
	u8 status = mgmt_status(err);

	bt_dev_dbg(hdev, "err %d", err);

	if (status) {
		mgmt_pending_foreach(MGMT_OP_SET_LE, hdev, true, cmd_status_rsp,
				     &status);
		return;
	}

	mgmt_pending_foreach(MGMT_OP_SET_LE, hdev, true, settings_rsp, &match);

	new_settings(hdev, match.sk);

	if (match.sk)
		sock_put(match.sk);
}
/**
 * @brief Synchronously sets the LE state of a device.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_le_sync(struct hci_dev *hdev, void *data)
{
	struct mgmt_pending_cmd *cmd = data;
	struct mgmt_mode *cp = cmd->param;
	u8 val = !!cp->val;
	int err;

	if (!val) {
		hci_clear_adv_instance_sync(hdev, NULL, 0x00, true);

		if (hci_dev_test_flag(hdev, HCI_LE_ADV))
			hci_disable_advertising_sync(hdev);

		if (ext_adv_capable(hdev))
			hci_remove_ext_adv_instance_sync(hdev, 0, cmd->sk);
	} else {
		hci_dev_set_flag(hdev, HCI_LE_ENABLED);
	}

	err = hci_write_le_host_supported_sync(hdev, val, 0);

	/* Make sure the controller has a good default for
	 * advertising data. Restrict the update to when LE
	 * has actually been enabled. During power on, the
	 * update in powered_update_hci will take care of it.
	 */
	if (!err && hci_dev_test_flag(hdev, HCI_LE_ENABLED)) {
		if (ext_adv_capable(hdev)) {
			int status;

			status = hci_setup_ext_adv_instance_sync(hdev, 0x00);
			if (!status)
				hci_update_scan_rsp_data_sync(hdev, 0x00);
		} else {
			hci_update_adv_data_sync(hdev, 0x00);
			hci_update_scan_rsp_data_sync(hdev, 0x00);
		}

		hci_update_passive_scan(hdev);
	}

	return err;
}
/**
 * @brief Handles the completion of the set_mesh command.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @param err The error code from the HCI command.
 */
static void set_mesh_complete(struct hci_dev *hdev, void *data, int err)
{
	struct mgmt_pending_cmd *cmd = data;
	u8 status = mgmt_status(err);
	struct sock *sk = cmd->sk;

	if (status) {
		mgmt_pending_foreach(MGMT_OP_SET_MESH_RECEIVER, hdev, true,
				     cmd_status_rsp, &status);
		return;
	}

	mgmt_pending_remove(cmd);
	mgmt_cmd_complete(sk, hdev->id, MGMT_OP_SET_MESH_RECEIVER, 0, NULL, 0);
}
/**
 * @brief Synchronously sets the mesh state of a device.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_mesh_sync(struct hci_dev *hdev, void *data)
{
	struct mgmt_pending_cmd *cmd = data;
	struct mgmt_cp_set_mesh *cp = cmd->param;
	size_t len = cmd->param_len;

	memset(hdev->mesh_ad_types, 0, sizeof(hdev->mesh_ad_types));

	if (cp->enable)
		hci_dev_set_flag(hdev, HCI_MESH);
	else
		hci_dev_clear_flag(hdev, HCI_MESH);

	hdev->le_scan_interval = __le16_to_cpu(cp->period);
	hdev->le_scan_window = __le16_to_cpu(cp->window);

	len -= sizeof(*cp);

	/* If filters don't fit, forward all adv pkts */
	if (len <= sizeof(hdev->mesh_ad_types))
		memcpy(hdev->mesh_ad_types, cp->ad_types, len);

	hci_update_passive_scan_sync(hdev);
	return 0;
}
/**
 * @brief Handles the MGMT_OP_SET_MESH_RECEIVER command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_mesh(struct sock *sk, struct hci_dev *hdev, void *data, u16 len)
{
	struct mgmt_cp_set_mesh *cp = data;
	struct mgmt_pending_cmd *cmd;
	__u16 period, window;
	int err = 0;

	bt_dev_dbg(hdev, "sock %p", sk);

	if (!lmp_le_capable(hdev) ||
	    !hci_dev_test_flag(hdev, HCI_MESH_EXPERIMENTAL))
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_MESH_RECEIVER,
				       MGMT_STATUS_NOT_SUPPORTED);

	if (cp->enable != 0x00 && cp->enable != 0x01)
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_MESH_RECEIVER,
				       MGMT_STATUS_INVALID_PARAMS);

	/* Keep allowed ranges in sync with set_scan_params() */
	period = __le16_to_cpu(cp->period);

	if (period < 0x0004 || period > 0x4000)
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_MESH_RECEIVER,
				       MGMT_STATUS_INVALID_PARAMS);

	window = __le16_to_cpu(cp->window);

	if (window < 0x0004 || window > 0x4000)
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_MESH_RECEIVER,
				       MGMT_STATUS_INVALID_PARAMS);

	if (window > period)
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_MESH_RECEIVER,
				       MGMT_STATUS_INVALID_PARAMS);

	hci_dev_lock(hdev);

	cmd = mgmt_pending_add(sk, MGMT_OP_SET_MESH_RECEIVER, hdev, data, len);
	if (!cmd)
		err = -ENOMEM;
	else
		err = hci_cmd_sync_queue(hdev, set_mesh_sync, cmd,
					 set_mesh_complete);

	if (err < 0) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_MESH_RECEIVER,
				      MGMT_STATUS_FAILED);

		if (cmd)
			mgmt_pending_remove(cmd);
	}

	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Handles the completion of a mesh send operation.
 * @param hdev The HCI device.
 * @param data The mesh transmission structure.
 * @param err The error code from the HCI command.
 */
static void mesh_send_start_complete(struct hci_dev *hdev, void *data, int err)
{
	struct mgmt_mesh_tx *mesh_tx = data;
	struct mgmt_cp_mesh_send *send = (void *)mesh_tx->param;
	unsigned long mesh_send_interval;
	u8 mgmt_err = mgmt_status(err);

	/* Report any errors here, but don't report completion */

	if (mgmt_err) {
		hci_dev_clear_flag(hdev, HCI_MESH_SENDING);
		/* Send Complete Error Code for handle */
		mesh_send_complete(hdev, mesh_tx, false);
		return;
	}

	mesh_send_interval = msecs_to_jiffies((send->cnt) * 25);
	queue_delayed_work(hdev->req_workqueue, &hdev->mesh_send_done,
			   mesh_send_interval);
}

/**
 * @brief Synchronously sends a mesh packet.
 * @param hdev The HCI device.
 * @param data The mesh transmission structure.
 * @return 0 on success, or a negative error code on failure.
 */
static int mesh_send_sync(struct hci_dev *hdev, void *data)
{
	struct mgmt_mesh_tx *mesh_tx = data;
	struct mgmt_cp_mesh_send *send = (void *)mesh_tx->param;
	struct adv_info *adv, *next_instance;
	u8 instance = hdev->le_num_of_adv_sets + 1;
	u16 timeout, duration;
	int err = 0;

	if (hdev->le_num_of_adv_sets <= hdev->adv_instance_cnt)
		return MGMT_STATUS_BUSY;

	timeout = 1000;
	duration = send->cnt * INTERVAL_TO_MS(hdev->le_adv_max_interval);
	adv = hci_add_adv_instance(hdev, instance, 0,
				   send->adv_data_len, send->adv_data,
				   0, NULL,
				   timeout, duration,
				   HCI_ADV_TX_POWER_NO_PREFERENCE,
				   hdev->le_adv_min_interval,
				   hdev->le_adv_max_interval,
				   mesh_tx->handle);

	if (!IS_ERR(adv))
		mesh_tx->instance = instance;
	else
		err = PTR_ERR(adv);

	if (hdev->cur_adv_instance == instance) {
		/* If the currently advertised instance is being changed then
		 * cancel the current advertising and schedule the next
		 * instance. If there is only one instance then the overridden
		 * advertising data will be visible right away.
		 */
		cancel_adv_timeout(hdev);

		next_instance = hci_get_next_instance(hdev, instance);
		if (next_instance)
			instance = next_instance->instance;
		else
			instance = 0;
	} else if (hdev->adv_instance_timeout) {
		/* Immediately advertise the new instance if no other, or
		 * let it go naturally from queue if ADV is already happening
		 */
		instance = 0;
	}

	if (instance)
		return hci_schedule_adv_instance_sync(hdev, instance, true);

	return err;
}
/**
 * @brief Counts the number of active mesh transmissions.
 * @param mesh_tx The mesh transmission structure.
 * @param data A pointer to a mgmt_rp_mesh_read_features structure.
 */
static void send_count(struct mgmt_mesh_tx *mesh_tx, void *data)
{
	struct mgmt_rp_mesh_read_features *rp = data;

	if (rp->used_handles >= rp->max_handles)
		return;

	rp->handles[rp->used_handles++] = mesh_tx->handle;
}
/**
 * @brief Handles the MGMT_OP_MESH_READ_FEATURES command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data (unused).
 * @param len The length of the command data (unused).
 * @return 0 on success, or a negative error code on failure.
 */
static int mesh_features(struct sock *sk, struct hci_dev *hdev,
			 void *data, u16 len)
{
	struct mgmt_rp_mesh_read_features rp;

	if (!lmp_le_capable(hdev) ||
	    !hci_dev_test_flag(hdev, HCI_MESH_EXPERIMENTAL))
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_MESH_READ_FEATURES,
				       MGMT_STATUS_NOT_SUPPORTED);

	memset(&rp, 0, sizeof(rp));
	rp.index = cpu_to_le16(hdev->id);
	if (hci_dev_test_flag(hdev, HCI_LE_ENABLED))
		rp.max_handles = MESH_HANDLES_MAX;

	hci_dev_lock(hdev);

	if (rp.max_handles)
		mgmt_mesh_foreach(hdev, send_count, &rp, sk);

	mgmt_cmd_complete(sk, hdev->id, MGMT_OP_MESH_READ_FEATURES, 0, &rp,
			  rp.used_handles + sizeof(rp) - MESH_HANDLES_MAX);

	hci_dev_unlock(hdev);
	return 0;
}
/**
 * @brief Cancels a mesh transmission.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int send_cancel(struct hci_dev *hdev, void *data)
{
	struct mgmt_pending_cmd *cmd = data;
	struct mgmt_cp_mesh_send_cancel *cancel = (void *)cmd->param;
	struct mgmt_mesh_tx *mesh_tx;

	if (!cancel->handle) {
		do {
			mesh_tx = mgmt_mesh_next(hdev, cmd->sk);

			if (mesh_tx)
				mesh_send_complete(hdev, mesh_tx, false);
		} while (mesh_tx);
	} else {
		mesh_tx = mgmt_mesh_find(hdev, cancel->handle);

		if (mesh_tx && mesh_tx->sk == cmd->sk)
			mesh_send_complete(hdev, mesh_tx, false);
	}

	mgmt_cmd_complete(cmd->sk, hdev->id, MGMT_OP_MESH_SEND_CANCEL,
			  0, NULL, 0);
	mgmt_pending_free(cmd);

	return 0;
}
/**
 * @brief Handles the MGMT_OP_MESH_SEND_CANCEL command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int mesh_send_cancel(struct sock *sk, struct hci_dev *hdev,
			    void *data, u16 len)
{
	struct mgmt_pending_cmd *cmd;
	int err;

	if (!lmp_le_capable(hdev) ||
	    !hci_dev_test_flag(hdev, HCI_MESH_EXPERIMENTAL))
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_MESH_SEND_CANCEL,
				       MGMT_STATUS_NOT_SUPPORTED);

	if (!hci_dev_test_flag(hdev, HCI_LE_ENABLED))
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_MESH_SEND_CANCEL,
				       MGMT_STATUS_REJECTED);

	hci_dev_lock(hdev);
	cmd = mgmt_pending_new(sk, MGMT_OP_MESH_SEND_CANCEL, hdev, data, len);
	if (!cmd)
		err = -ENOMEM;
	else
		err = hci_cmd_sync_queue(hdev, send_cancel, cmd, NULL);

	if (err < 0) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_MESH_SEND_CANCEL,
				      MGMT_STATUS_FAILED);

		if (cmd)
			mgmt_pending_free(cmd);
	}

	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Handles the MGMT_OP_MESH_SEND command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int mesh_send(struct sock *sk, struct hci_dev *hdev, void *data, u16 len)
{
	struct mgmt_mesh_tx *mesh_tx;
	struct mgmt_cp_mesh_send *send = data;
	struct mgmt_rp_mesh_read_features rp;
	bool sending;
	int err = 0;

	if (!lmp_le_capable(hdev) ||
	    !hci_dev_test_flag(hdev, HCI_MESH_EXPERIMENTAL))
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_MESH_SEND,
				       MGMT_STATUS_NOT_SUPPORTED);
	if (!hci_dev_test_flag(hdev, HCI_LE_ENABLED) ||
	    len <= MGMT_MESH_SEND_SIZE ||
	    len > (MGMT_MESH_SEND_SIZE + 31))
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_MESH_SEND,
				       MGMT_STATUS_REJECTED);

	hci_dev_lock(hdev);

	memset(&rp, 0, sizeof(rp));
	rp.max_handles = MESH_HANDLES_MAX;

	mgmt_mesh_foreach(hdev, send_count, &rp, sk);

	if (rp.max_handles <= rp.used_handles) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_MESH_SEND,
				      MGMT_STATUS_BUSY);
		goto done;
	}

	sending = hci_dev_test_flag(hdev, HCI_MESH_SENDING);
	mesh_tx = mgmt_mesh_add(sk, hdev, send, len);

	if (!mesh_tx)
		err = -ENOMEM;
	else if (!sending)
		err = hci_cmd_sync_queue(hdev, mesh_send_sync, mesh_tx,
					 mesh_send_start_complete);

	if (err < 0) {
		bt_dev_err(hdev, "Send Mesh Failed %d", err);
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_MESH_SEND,
				      MGMT_STATUS_FAILED);

		if (mesh_tx) {
			if (sending)
				mgmt_mesh_remove(mesh_tx);
		}
	} else {
		hci_dev_set_flag(hdev, HCI_MESH_SENDING);

		mgmt_cmd_complete(sk, hdev->id, MGMT_OP_MESH_SEND, 0,
				  &mesh_tx->handle, 1);
	}

done:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Handles the MGMT_OP_SET_LE command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_le(struct sock *sk, struct hci_dev *hdev, void *data, u16 len)
{
	struct mgmt_mode *cp = data;
	struct mgmt_pending_cmd *cmd;
	int err;
	u8 val, enabled;

	bt_dev_dbg(hdev, "sock %p", sk);

	if (!lmp_le_capable(hdev))
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_LE,
				       MGMT_STATUS_NOT_SUPPORTED);

	if (cp->val != 0x00 && cp->val != 0x01)
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_LE,
				       MGMT_STATUS_INVALID_PARAMS);

	/* Bluetooth single mode LE only controllers or dual-mode
	 * controllers configured as LE only devices, do not allow
	 * switching LE off. These have either LE enabled explicitly
	 * or BR/EDR has been previously switched off.
	 *
	 * When trying to enable an already enabled LE, then gracefully
	 * send a positive response. Trying to disable it however will
	 * result into rejection.
	 */
	if (!hci_dev_test_flag(hdev, HCI_BREDR_ENABLED)) {
		if (cp->val == 0x01)
			return send_settings_rsp(sk, MGMT_OP_SET_LE, hdev);

		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_LE,
				       MGMT_STATUS_REJECTED);
	}

	hci_dev_lock(hdev);

	val = !!cp->val;
	enabled = lmp_host_le_capable(hdev);

	if (!hdev_is_powered(hdev) || val == enabled) {
		bool changed = false;

		if (val != hci_dev_test_flag(hdev, HCI_LE_ENABLED)) {
			hci_dev_change_flag(hdev, HCI_LE_ENABLED);
			changed = true;
		}

		if (!val && hci_dev_test_flag(hdev, HCI_ADVERTISING)) {
			hci_dev_clear_flag(hdev, HCI_ADVERTISING);
			changed = true;
		}

		err = send_settings_rsp(sk, MGMT_OP_SET_LE, hdev);
		if (err < 0)
			goto unlock;

		if (changed)
			err = new_settings(hdev, sk);

		goto unlock;
	}

	if (pending_find(MGMT_OP_SET_LE, hdev) ||
	    pending_find(MGMT_OP_SET_ADVERTISING, hdev)) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_LE,
				      MGMT_STATUS_BUSY);
		goto unlock;
	}

	cmd = mgmt_pending_add(sk, MGMT_OP_SET_LE, hdev, data, len);
	if (!cmd)
		err = -ENOMEM;
	else
		err = hci_cmd_sync_queue(hdev, set_le_sync, cmd,
					 set_le_complete);

	if (err < 0) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_LE,
				      MGMT_STATUS_FAILED);

		if (cmd)
			mgmt_pending_remove(cmd);
	}

unlock:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Synchronously sends an HCI command.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int send_hci_cmd_sync(struct hci_dev *hdev, void *data)
{
	struct mgmt_pending_cmd *cmd = data;
	struct mgmt_cp_hci_cmd_sync *cp = cmd->param;
	struct sk_buff *skb;

	skb = __hci_cmd_sync_ev(hdev, le16_to_cpu(cp->opcode),
				le16_to_cpu(cp->params_len), cp->params,
				cp->event, cp->timeout ?
				secs_to_jiffies(cp->timeout) :
				HCI_CMD_TIMEOUT);
	if (IS_ERR(skb)) {
		mgmt_cmd_status(cmd->sk, hdev->id, MGMT_OP_HCI_CMD_SYNC,
				mgmt_status(PTR_ERR(skb)));
		goto done;
	}

	mgmt_cmd_complete(cmd->sk, hdev->id, MGMT_OP_HCI_CMD_SYNC, 0,
			  skb->data, skb->len);

	kfree_skb(skb);

done:
	mgmt_pending_free(cmd);

	return 0;
}
/**
 * @brief Handles the MGMT_OP_HCI_CMD_SYNC command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int mgmt_hci_cmd_sync(struct sock *sk, struct hci_dev *hdev,
			     void *data, u16 len)
{
	struct mgmt_cp_hci_cmd_sync *cp = data;
	struct mgmt_pending_cmd *cmd;
	int err;

	if (len != (offsetof(struct mgmt_cp_hci_cmd_sync, params) +
		    le16_to_cpu(cp->params_len)))
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_HCI_CMD_SYNC,
				       MGMT_STATUS_INVALID_PARAMS);

	hci_dev_lock(hdev);
	cmd = mgmt_pending_new(sk, MGMT_OP_HCI_CMD_SYNC, hdev, data, len);
	if (!cmd)
		err = -ENOMEM;
	else
		err = hci_cmd_sync_queue(hdev, send_hci_cmd_sync, cmd, NULL);

	if (err < 0) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_HCI_CMD_SYNC,
				      MGMT_STATUS_FAILED);

		if (cmd)
			mgmt_pending_free(cmd);
	}

	hci_dev_unlock(hdev);
	return err;
}

/**
 * @brief Checks if there are any pending management commands that could affect EIR or class of device.
 * @param hdev The HCI device.
 * @return True if there are pending commands, false otherwise.
 */
static bool pending_eir_or_class(struct hci_dev *hdev)
{
	struct mgmt_pending_cmd *cmd;

	list_for_each_entry(cmd, &hdev->mgmt_pending, list) {
		switch (cmd->opcode) {
		case MGMT_OP_ADD_UUID:
		case MGMT_OP_REMOVE_UUID:
		case MGMT_OP_SET_DEV_CLASS:
		case MGMT_OP_SET_POWERED:
			return true;
		}
	}

	return false;
}

static const u8 bluetooth_base_uuid[] = {
			0xfb, 0x34, 0x9b, 0x5f, 0x80, 0x00, 0x00, 0x80,
			0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

/**
 * @brief Gets the size of a UUID.
 * @param uuid The UUID.
 * @return The size of the UUID in bits (16, 32, or 128).
 */
static u8 get_uuid_size(const u8 *uuid)
{
	u32 val;

	if (memcmp(uuid, bluetooth_base_uuid, 12))
		return 128;

	val = get_unaligned_le32(&uuid[12]);
	if (val > 0xffff)
		return 32;

	return 16;
}
/**
 * @brief Handles the completion of class of device related commands.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @param err The error code from the HCI command.
 */
static void mgmt_class_complete(struct hci_dev *hdev, void *data, int err)
{
	struct mgmt_pending_cmd *cmd = data;

	bt_dev_dbg(hdev, "err %d", err);

	mgmt_cmd_complete(cmd->sk, cmd->hdev->id, cmd->opcode,
			  mgmt_status(err), hdev->dev_class, 3);

	mgmt_pending_free(cmd);
}
/**
 * @brief Synchronously adds a UUID and updates EIR and class of device.
 * @param hdev The HCI device.
 * @param data Unused.
 * @return 0 on success, or a negative error code on failure.
 */
static int add_uuid_sync(struct hci_dev *hdev, void *data)
{
	int err;

	err = hci_update_class_sync(hdev);
	if (err)
		return err;

	return hci_update_eir_sync(hdev);
}
/**
 * @brief Handles the MGMT_OP_ADD_UUID command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int add_uuid(struct sock *sk, struct hci_dev *hdev, void *data, u16 len)
{
	struct mgmt_cp_add_uuid *cp = data;
	struct mgmt_pending_cmd *cmd;
	struct bt_uuid *uuid;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	hci_dev_lock(hdev);

	if (pending_eir_or_class(hdev)) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_ADD_UUID,
				      MGMT_STATUS_BUSY);
		goto failed;
	}

	uuid = kmalloc(sizeof(*uuid), GFP_KERNEL);
	if (!uuid) {
		err = -ENOMEM;
		goto failed;
	}

	memcpy(uuid->uuid, cp->uuid, 16);
	uuid->svc_hint = cp->svc_hint;
	uuid->size = get_uuid_size(cp->uuid);

	list_add_tail(&uuid->list, &hdev->uuids);

	cmd = mgmt_pending_new(sk, MGMT_OP_ADD_UUID, hdev, data, len);
	if (!cmd) {
		err = -ENOMEM;
		goto failed;
	}

	/* MGMT_OP_ADD_UUID don't require adapter the UP/Running so use
	 * hci_cmd_sync_submit instead of hci_cmd_sync_queue.
	 */
	err = hci_cmd_sync_submit(hdev, add_uuid_sync, cmd,
				  mgmt_class_complete);
	if (err < 0) {
		mgmt_pending_free(cmd);
		goto failed;
	}

failed:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Enables the service cache if not already enabled.
 * @param hdev The HCI device.
 * @return True if the service cache was enabled, false otherwise.
 */
static bool enable_service_cache(struct hci_dev *hdev)
{
	if (!hdev_is_powered(hdev))
		return false;

	if (!hci_dev_test_and_set_flag(hdev, HCI_SERVICE_CACHE)) {
		queue_delayed_work(hdev->workqueue, &hdev->service_cache,
				   CACHE_TIMEOUT);
		return true;
	}

	return false;
}
/**
 * @brief Synchronously removes a UUID and updates EIR and class of device.
 * @param hdev The HCI device.
 * @param data Unused.
 * @return 0 on success, or a negative error code on failure.
 */
static int remove_uuid_sync(struct hci_dev *hdev, void *data)
{
	int err;

	err = hci_update_class_sync(hdev);
	if (err)
		return err;

	return hci_update_eir_sync(hdev);
}
/**
 * @brief Handles the MGMT_OP_REMOVE_UUID command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int remove_uuid(struct sock *sk, struct hci_dev *hdev, void *data,
		       u16 len)
{
	struct mgmt_cp_remove_uuid *cp = data;
	struct mgmt_pending_cmd *cmd;
	struct bt_uuid *match, *tmp;
	static const u8 bt_uuid_any[] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	};
	int err, found;

	bt_dev_dbg(hdev, "sock %p", sk);

	hci_dev_lock(hdev);

	if (pending_eir_or_class(hdev)) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_REMOVE_UUID,
				      MGMT_STATUS_BUSY);
		goto unlock;
	}

	if (memcmp(cp->uuid, bt_uuid_any, 16) == 0) {
		hci_uuids_clear(hdev);

		if (enable_service_cache(hdev)) {
			err = mgmt_cmd_complete(sk, hdev->id,
						MGMT_OP_REMOVE_UUID,
						0, hdev->dev_class, 3);
			goto unlock;
		}

		goto update_class;
	}

	found = 0;

	list_for_each_entry_safe(match, tmp, &hdev->uuids, list) {
		if (memcmp(match->uuid, cp->uuid, 16) != 0)
			continue;

		list_del(&match->list);
		kfree(match);
		found++;
	}

	if (found == 0) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_REMOVE_UUID,
				      MGMT_STATUS_INVALID_PARAMS);
		goto unlock;
	}

update_class:
	cmd = mgmt_pending_new(sk, MGMT_OP_REMOVE_UUID, hdev, data, len);
	if (!cmd) {
		err = -ENOMEM;
		goto unlock;
	}

	/* MGMT_OP_REMOVE_UUID don't require adapter the UP/Running so use
	 * hci_cmd_sync_submit instead of hci_cmd_sync_queue.
	 */
	err = hci_cmd_sync_submit(hdev, remove_uuid_sync, cmd,
				  mgmt_class_complete);
	if (err < 0)
		mgmt_pending_free(cmd);

unlock:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Synchronously sets the class of device.
 * @param hdev The HCI device.
 * @param data Unused.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_class_sync(struct hci_dev *hdev, void *data)
{
	int err = 0;

	if (hci_dev_test_and_clear_flag(hdev, HCI_SERVICE_CACHE)) {
		cancel_delayed_work_sync(&hdev->service_cache);
		err = hci_update_eir_sync(hdev);
	}

	if (err)
		return err;

	return hci_update_class_sync(hdev);
}
/**
 * @brief Handles the MGMT_OP_SET_DEV_CLASS command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_dev_class(struct sock *sk, struct hci_dev *hdev, void *data,
			 u16 len)
{
	struct mgmt_cp_set_dev_class *cp = data;
	struct mgmt_pending_cmd *cmd;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	if (!lmp_bredr_capable(hdev))
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_DEV_CLASS,
				       MGMT_STATUS_NOT_SUPPORTED);

	hci_dev_lock(hdev);

	if (pending_eir_or_class(hdev)) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_DEV_CLASS,
				      MGMT_STATUS_BUSY);
		goto unlock;
	}

	if ((cp->minor & 0x03) != 0 || (cp->major & 0xe0) != 0) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_DEV_CLASS,
				      MGMT_STATUS_INVALID_PARAMS);
		goto unlock;
	}

	hdev->major_class = cp->major;
	hdev->minor_class = cp->minor;

	if (!hdev_is_powered(hdev)) {
		err = mgmt_cmd_complete(sk, hdev->id, MGMT_OP_SET_DEV_CLASS, 0,
					hdev->dev_class, 3);
		goto unlock;
	}

	cmd = mgmt_pending_new(sk, MGMT_OP_SET_DEV_CLASS, hdev, data, len);
	if (!cmd) {
		err = -ENOMEM;
		goto unlock;
	}

	/* MGMT_OP_SET_DEV_CLASS don't require adapter the UP/Running so use
	 * hci_cmd_sync_submit instead of hci_cmd_sync_queue.
	 */
	err = hci_cmd_sync_submit(hdev, set_class_sync, cmd,
				  mgmt_class_complete);
	if (err < 0)
		mgmt_pending_free(cmd);

unlock:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Handles the MGMT_OP_LOAD_LINK_KEYS command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int load_link_keys(struct sock *sk, struct hci_dev *hdev, void *data,
			  u16 len)
{
	struct mgmt_cp_load_link_keys *cp = data;
	const u16 max_key_count = ((U16_MAX - sizeof(*cp)) /
				   sizeof(struct mgmt_link_key_info));
	u16 key_count, expected_len;
	bool changed;
	int i;

	bt_dev_dbg(hdev, "sock %p", sk);

	if (!lmp_bredr_capable(hdev))
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_LOAD_LINK_KEYS,
				       MGMT_STATUS_NOT_SUPPORTED);

	key_count = __le16_to_cpu(cp->key_count);
	if (key_count > max_key_count) {
		bt_dev_err(hdev, "load_link_keys: too big key_count value %u",
			   key_count);
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_LOAD_LINK_KEYS,
				       MGMT_STATUS_INVALID_PARAMS);
	}

	expected_len = struct_size(cp, keys, key_count);
	if (expected_len != len) {
		bt_dev_err(hdev, "load_link_keys: expected %u bytes, got %u bytes",
			   expected_len, len);
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_LOAD_LINK_KEYS,
				       MGMT_STATUS_INVALID_PARAMS);
	}

	if (cp->debug_keys != 0x00 && cp->debug_keys != 0x01)
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_LOAD_LINK_KEYS,
				       MGMT_STATUS_INVALID_PARAMS);

	bt_dev_dbg(hdev, "debug_keys %u key_count %u", cp->debug_keys,
		   key_count);

	hci_dev_lock(hdev);

	hci_link_keys_clear(hdev);

	if (cp->debug_keys)
		changed = !hci_dev_test_and_set_flag(hdev, HCI_KEEP_DEBUG_KEYS);
	else
		changed = hci_dev_test_and_clear_flag(hdev,
						      HCI_KEEP_DEBUG_KEYS);

	if (changed)
		new_settings(hdev, NULL);

	for (i = 0; i < key_count; i++) {
		struct mgmt_link_key_info *key = &cp->keys[i];

		if (hci_is_blocked_key(hdev,
				       HCI_BLOCKED_KEY_TYPE_LINKKEY,
				       key->val)) {
			bt_dev_warn(hdev, "Skipping blocked link key for %pMR",
				    &key->addr.bdaddr);
			continue;
		}

		if (key->addr.type != BDADDR_BREDR) {
			bt_dev_warn(hdev,
				    "Invalid link address type %u for %pMR",
				    key->addr.type, &key->addr.bdaddr);
			continue;
		}

		if (key->type > 0x08) {
			bt_dev_warn(hdev, "Invalid link key type %u for %pMR",
				    key->type, &key->addr.bdaddr);
			continue;
		}

		/* Always ignore debug keys and require a new pairing if
		 * the user wants to use them.
		 */
		if (key->type == HCI_LK_DEBUG_COMBINATION)
			continue;

		hci_add_link_key(hdev, NULL, &key->addr.bdaddr, key->val,
				 key->type, key->pin_len, NULL);
	}

	mgmt_cmd_complete(sk, hdev->id, MGMT_OP_LOAD_LINK_KEYS, 0, NULL, 0);

	hci_dev_unlock(hdev);

	return 0;
}
/**
 * @brief Sends a MGMT_EV_DEVICE_UNPAIRED event.
 * @param hdev The HCI device.
 * @param bdaddr The address of the unpaired device.
 * @param addr_type The address type of the unpaired device.
 * @param skip_sk The socket to skip.
 * @return 0 on success, or a negative error code on failure.
 */
static int device_unpaired(struct hci_dev *hdev, bdaddr_t *bdaddr,
			   u8 addr_type, struct sock *skip_sk)
{
	struct mgmt_ev_device_unpaired ev;

	bacpy(&ev.addr.bdaddr, bdaddr);
	ev.addr.type = addr_type;

	return mgmt_event(MGMT_EV_DEVICE_UNPAIRED, hdev, &ev, sizeof(ev),
			  skip_sk);
}
/**
 * @brief Handles the completion of the unpair_device command.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @param err The error code from the HCI command.
 */
static void unpair_device_complete(struct hci_dev *hdev, void *data, int err)
{
	struct mgmt_pending_cmd *cmd = data;
	struct mgmt_cp_unpair_device *cp = cmd->param;

	if (!err)
		device_unpaired(hdev, &cp->addr.bdaddr, cp->addr.type, cmd->sk);

	cmd->cmd_complete(cmd, err);
	mgmt_pending_free(cmd);
}
/**
 * @brief Synchronously unpairs a device.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int unpair_device_sync(struct hci_dev *hdev, void *data)
{
	struct mgmt_pending_cmd *cmd = data;
	struct mgmt_cp_unpair_device *cp = cmd->param;
	struct hci_conn *conn;

	if (cp->addr.type == BDADDR_BREDR)
		conn = hci_conn_hash_lookup_ba(hdev, ACL_LINK,
					       &cp->addr.bdaddr);
	else
		conn = hci_conn_hash_lookup_le(hdev, &cp->addr.bdaddr,
					       le_addr_type(cp->addr.type));

	if (!conn)
		return 0;

	/* Disregard any possible error since the likes of hci_abort_conn_sync
	 * will clean up the connection no matter the error.
	 */
	hci_abort_conn(conn, HCI_ERROR_REMOTE_USER_TERM);

	return 0;
}
/**
 * @brief Handles the MGMT_OP_UNPAIR_DEVICE command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int unpair_device(struct sock *sk, struct hci_dev *hdev, void *data,
			 u16 len)
{
	struct mgmt_cp_unpair_device *cp = data;
	struct mgmt_rp_unpair_device rp;
	struct hci_conn_params *params;
	struct mgmt_pending_cmd *cmd;
	struct hci_conn *conn;
	u8 addr_type;
	int err;

	memset(&rp, 0, sizeof(rp));
	bacpy(&rp.addr.bdaddr, &cp->addr.bdaddr);
	rp.addr.type = cp->addr.type;

	if (!bdaddr_type_is_valid(cp->addr.type))
		return mgmt_cmd_complete(sk, hdev->id, MGMT_OP_UNPAIR_DEVICE,
					 MGMT_STATUS_INVALID_PARAMS,
					 &rp, sizeof(rp));

	if (cp->disconnect != 0x00 && cp->disconnect != 0x01)
		return mgmt_cmd_complete(sk, hdev->id, MGMT_OP_UNPAIR_DEVICE,
					 MGMT_STATUS_INVALID_PARAMS,
					 &rp, sizeof(rp));

	hci_dev_lock(hdev);

	if (!hdev_is_powered(hdev)) {
		err = mgmt_cmd_complete(sk, hdev->id, MGMT_OP_UNPAIR_DEVICE,
					MGMT_STATUS_NOT_POWERED, &rp,
					sizeof(rp));
		goto unlock;
	}

	if (cp->addr.type == BDADDR_BREDR) {
		/* If disconnection is requested, then look up the
		 * connection. If the remote device is connected, it
		 * will be later used to terminate the link.
		 *
		 * Setting it to NULL explicitly will cause no
		 * termination of the link.
		 */
		if (cp->disconnect)
			conn = hci_conn_hash_lookup_ba(hdev, ACL_LINK,
						       &cp->addr.bdaddr);
		else
			conn = NULL;

		err = hci_remove_link_key(hdev, &cp->addr.bdaddr);
		if (err < 0) {
			err = mgmt_cmd_complete(sk, hdev->id,
						MGMT_OP_UNPAIR_DEVICE,
						MGMT_STATUS_NOT_PAIRED, &rp,
						sizeof(rp));
			goto unlock;
		}

		goto done;
	}

	/* LE address type */
	addr_type = le_addr_type(cp->addr.type);

	/* Abort any ongoing SMP pairing. Removes ltk and irk if they exist. */
	err = smp_cancel_and_remove_pairing(hdev, &cp->addr.bdaddr, addr_type);
	if (err < 0) {
		err = mgmt_cmd_complete(sk, hdev->id, MGMT_OP_UNPAIR_DEVICE,
					MGMT_STATUS_NOT_PAIRED, &rp,
					sizeof(rp));
		goto unlock;
	}

	conn = hci_conn_hash_lookup_le(hdev, &cp->addr.bdaddr, addr_type);
	if (!conn) {
		hci_conn_params_del(hdev, &cp->addr.bdaddr, addr_type);
		goto done;
	}


	/* Defer clearing up the connection parameters until closing to
	 * give a chance of keeping them if a repairing happens.
	 */
	set_bit(HCI_CONN_PARAM_REMOVAL_PEND, &conn->flags);

	/* Disable auto-connection parameters if present */
	params = hci_conn_params_lookup(hdev, &cp->addr.bdaddr, addr_type);
	if (params) {
		if (params->explicit_connect)
			params->auto_connect = HCI_AUTO_CONN_EXPLICIT;
		else
			params->auto_connect = HCI_AUTO_CONN_DISABLED;
	}

	/* If disconnection is not requested, then clear the connection
	 * variable so that the link is not terminated.
	 */
	if (!cp->disconnect)
		conn = NULL;

done:
	/* If the connection variable is set, then termination of the
	 * link is requested.
	 */
	if (!conn) {
		err = mgmt_cmd_complete(sk, hdev->id, MGMT_OP_UNPAIR_DEVICE, 0,
					&rp, sizeof(rp));
		device_unpaired(hdev, &cp->addr.bdaddr, cp->addr.type, sk);
		goto unlock;
	}

	cmd = mgmt_pending_new(sk, MGMT_OP_UNPAIR_DEVICE, hdev, cp,
			       sizeof(*cp));
	if (!cmd) {
		err = -ENOMEM;
		goto unlock;
	}

	cmd->cmd_complete = addr_cmd_complete;

	err = hci_cmd_sync_queue(hdev, unpair_device_sync, cmd,
				 unpair_device_complete);
	if (err < 0)
		mgmt_pending_free(cmd);

unlock:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Handles the completion of the disconnect command.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @param err The error code from the HCI command.
 */
static void disconnect_complete(struct hci_dev *hdev, void *data, int err)
{
	struct mgmt_pending_cmd *cmd = data;

	cmd->cmd_complete(cmd, mgmt_status(err));
	mgmt_pending_free(cmd);
}
/**
 * @brief Synchronously disconnects a device.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int disconnect_sync(struct hci_dev *hdev, void *data)
{
	struct mgmt_pending_cmd *cmd = data;
	struct mgmt_cp_disconnect *cp = cmd->param;
	struct hci_conn *conn;

	if (cp->addr.type == BDADDR_BREDR)
		conn = hci_conn_hash_lookup_ba(hdev, ACL_LINK,
					       &cp->addr.bdaddr);
	else
		conn = hci_conn_hash_lookup_le(hdev, &cp->addr.bdaddr,
					       le_addr_type(cp->addr.type));

	if (!conn)
		return -ENOTCONN;

	/* Disregard any possible error since the likes of hci_abort_conn_sync
	 * will clean up the connection no matter the error.
	 */
	hci_abort_conn(conn, HCI_ERROR_REMOTE_USER_TERM);

	return 0;
}
/**
 * @brief Handles the MGMT_OP_DISCONNECT command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int disconnect(struct sock *sk, struct hci_dev *hdev, void *data,
		      u16 len)
{
	struct mgmt_cp_disconnect *cp = data;
	struct mgmt_rp_disconnect rp;
	struct mgmt_pending_cmd *cmd;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	memset(&rp, 0, sizeof(rp));
	bacpy(&rp.addr.bdaddr, &cp->addr.bdaddr);
	rp.addr.type = cp->addr.type;

	if (!bdaddr_type_is_valid(cp->addr.type))
		return mgmt_cmd_complete(sk, hdev->id, MGMT_OP_DISCONNECT,
					 MGMT_STATUS_INVALID_PARAMS,
					 &rp, sizeof(rp));

	hci_dev_lock(hdev);

	if (!test_bit(HCI_UP, &hdev->flags)) {
		err = mgmt_cmd_complete(sk, hdev->id, MGMT_OP_DISCONNECT,
					MGMT_STATUS_NOT_POWERED, &rp,
					sizeof(rp));
		goto failed;
	}

	cmd = mgmt_pending_new(sk, MGMT_OP_DISCONNECT, hdev, data, len);
	if (!cmd) {
		err = -ENOMEM;
		goto failed;
	}

	cmd->cmd_complete = generic_cmd_complete;

	err = hci_cmd_sync_queue(hdev, disconnect_sync, cmd,
				 disconnect_complete);
	if (err < 0)
		mgmt_pending_free(cmd);

failed:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Converts a link type and address type to a BD address type.
 * @param link_type The link type.
 * @param addr_type The address type.
 * @return The corresponding BD address type.
 */
static u8 link_to_bdaddr(u8 link_type, u8 addr_type)
{
	switch (link_type) {
	case CIS_LINK:
	case BIS_LINK:
	case LE_LINK:
		switch (addr_type) {
		case ADDR_LE_DEV_PUBLIC:
			return BDADDR_LE_PUBLIC;

		default:
			/* Fallback to LE Random address type */
			return BDADDR_LE_RANDOM;
		}

	default:
		/* Fallback to BR/EDR type */
		return BDADDR_BREDR;
	}
}
/**
 * @brief Handles the MGMT_OP_GET_CONNECTIONS command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data (unused).
 * @param data_len The length of the command data (unused).
 * @return 0 on success, or a negative error code on failure.
 */
static int get_connections(struct sock *sk, struct hci_dev *hdev, void *data,
			   u16 data_len)
{
	struct mgmt_rp_get_connections *rp;
	struct hci_conn *c;
	int err;
	u16 i;

	bt_dev_dbg(hdev, "sock %p", sk);

	hci_dev_lock(hdev);

	if (!hdev_is_powered(hdev)) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_GET_CONNECTIONS,
				      MGMT_STATUS_NOT_POWERED);
		goto unlock;
	}

	i = 0;
	list_for_each_entry(c, &hdev->conn_hash.list, list) {
		if (test_bit(HCI_CONN_MGMT_CONNECTED, &c->flags))
			i++;
	}

	rp = kmalloc(struct_size(rp, addr, i), GFP_KERNEL);
	if (!rp) {
		err = -ENOMEM;
		goto unlock;
	}

	i = 0;
	list_for_each_entry(c, &hdev->conn_hash.list, list) {
		if (!test_bit(HCI_CONN_MGMT_CONNECTED, &c->flags))
			continue;
		bacpy(&rp->addr[i].bdaddr, &c->dst);
		rp->addr[i].type = link_to_bdaddr(c->type, c->dst_type);
		if (c->type == SCO_LINK || c->type == ESCO_LINK)
			continue;
		i++;
	}

	rp->conn_count = cpu_to_le16(i);

	/* Recalculate length in case of filtered SCO connections, etc */
	err = mgmt_cmd_complete(sk, hdev->id, MGMT_OP_GET_CONNECTIONS, 0, rp,
				struct_size(rp, addr, i));

	kfree(rp);

unlock:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Sends a PIN code negative reply.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param cp The PIN code negative reply parameters.
 * @return 0 on success, or a negative error code on failure.
 */
static int send_pin_code_neg_reply(struct sock *sk, struct hci_dev *hdev,
				   struct mgmt_cp_pin_code_neg_reply *cp)
{
	struct mgmt_pending_cmd *cmd;
	int err;

	cmd = mgmt_pending_add(sk, MGMT_OP_PIN_CODE_NEG_REPLY, hdev, cp,
			       sizeof(*cp));
	if (!cmd)
		return -ENOMEM;

	cmd->cmd_complete = addr_cmd_complete;

	err = hci_send_cmd(hdev, HCI_OP_PIN_CODE_NEG_REPLY,
			   sizeof(cp->addr.bdaddr), &cp->addr.bdaddr);
	if (err < 0)
		mgmt_pending_remove(cmd);

	return err;
}
/**
 * @brief Handles the MGMT_OP_PIN_CODE_REPLY command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int pin_code_reply(struct sock *sk, struct hci_dev *hdev, void *data,
			  u16 len)
{
	struct hci_conn *conn;
	struct mgmt_cp_pin_code_reply *cp = data;
	struct hci_cp_pin_code_reply reply;
	struct mgmt_pending_cmd *cmd;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	hci_dev_lock(hdev);

	if (!hdev_is_powered(hdev)) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_PIN_CODE_REPLY,
				      MGMT_STATUS_NOT_POWERED);
		goto failed;
	}

	conn = hci_conn_hash_lookup_ba(hdev, ACL_LINK, &cp->addr.bdaddr);
	if (!conn) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_PIN_CODE_REPLY,
				      MGMT_STATUS_NOT_CONNECTED);
		goto failed;
	}

	if (conn->pending_sec_level == BT_SECURITY_HIGH && cp->pin_len != 16) {
		struct mgmt_cp_pin_code_neg_reply ncp;

		memcpy(&ncp.addr, &cp->addr, sizeof(ncp.addr));

		bt_dev_err(hdev, "PIN code is not 16 bytes long");

		err = send_pin_code_neg_reply(sk, hdev, &ncp);
		if (err >= 0)
			err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_PIN_CODE_REPLY,
					      MGMT_STATUS_INVALID_PARAMS);

		goto failed;
	}

	cmd = mgmt_pending_add(sk, MGMT_OP_PIN_CODE_REPLY, hdev, data, len);
	if (!cmd) {
		err = -ENOMEM;
		goto failed;
	}

	cmd->cmd_complete = addr_cmd_complete;

	bacpy(&reply.bdaddr, &cp->addr.bdaddr);
	reply.pin_len = cp->pin_len;
	memcpy(reply.pin_code, cp->pin_code, sizeof(reply.pin_code));

	err = hci_send_cmd(hdev, HCI_OP_PIN_CODE_REPLY, sizeof(reply), &reply);
	if (err < 0)
		mgmt_pending_remove(cmd);

failed:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Handles the MGMT_OP_SET_IO_CAPABILITY command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_io_capability(struct sock *sk, struct hci_dev *hdev, void *data,
			     u16 len)
{
	struct mgmt_cp_set_io_capability *cp = data;

	bt_dev_dbg(hdev, "sock %p", sk);

	if (cp->io_capability > SMP_IO_KEYBOARD_DISPLAY)
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_IO_CAPABILITY,
				       MGMT_STATUS_INVALID_PARAMS);

	hci_dev_lock(hdev);

	hdev->io_capability = cp->io_capability;

	bt_dev_dbg(hdev, "IO capability set to 0x%02x", hdev->io_capability);

	hci_dev_unlock(hdev);

	return mgmt_cmd_complete(sk, hdev->id, MGMT_OP_SET_IO_CAPABILITY, 0,
				 NULL, 0);
}

/**
 * @brief Finds a pending pairing command for a given HCI connection.
 * @param conn The HCI connection.
 * @return A pointer to the pending command, or NULL if not found.
 */
static struct mgmt_pending_cmd *find_pairing(struct hci_conn *conn)
{
	struct hci_dev *hdev = conn->hdev;
	struct mgmt_pending_cmd *cmd;

	list_for_each_entry(cmd, &hdev->mgmt_pending, list) {
		if (cmd->opcode != MGMT_OP_PAIR_DEVICE)
			continue;

		if (cmd->user_data != conn)
			continue;

		return cmd;
	}

	return NULL;
}
/**
 * @brief Handles the completion of the pair_device command.
 * @param cmd The pending command.
 * @param status The status of the pairing operation.
 * @return 0 on success, or a negative error code on failure.
 */
static int pairing_complete(struct mgmt_pending_cmd *cmd, u8 status)
{
	struct mgmt_rp_pair_device rp;
	struct hci_conn *conn = cmd->user_data;
	int err;

	bacpy(&rp.addr.bdaddr, &conn->dst);
	rp.addr.type = link_to_bdaddr(conn->type, conn->dst_type);

	err = mgmt_cmd_complete(cmd->sk, cmd->hdev->id, MGMT_OP_PAIR_DEVICE,
				status, &rp, sizeof(rp));

	/* So we don't get further callbacks for this connection */
	conn->connect_cfm_cb = NULL;
	conn->security_cfm_cb = NULL;
	conn->disconn_cfm_cb = NULL;

	hci_conn_drop(conn);

	/* The device is paired so there is no need to remove
	 * its connection parameters anymore.
	 */
	clear_bit(HCI_CONN_PARAM_REMOVAL_PEND, &conn->flags);

	hci_conn_put(conn);

	return err;
}
/**
 * @brief Notifies the management layer about the completion of an SMP operation.
 * @param conn The HCI connection.
 * @param complete True if the operation was successful, false otherwise.
 */
void mgmt_smp_complete(struct hci_conn *conn, bool complete)
{
	u8 status = complete ? MGMT_STATUS_SUCCESS : MGMT_STATUS_FAILED;
	struct mgmt_pending_cmd *cmd;

	cmd = find_pairing(conn);
	if (cmd) {
		cmd->cmd_complete(cmd, status);
		mgmt_pending_remove(cmd);
	}
}

/**
 * @brief Callback function for pairing completion.
 * @param conn The HCI connection.
 * @param status The status of the pairing operation.
 */
static void pairing_complete_cb(struct hci_conn *conn, u8 status)
{
	struct mgmt_pending_cmd *cmd;

	BT_DBG("status %u", status);

	cmd = find_pairing(conn);
	if (!cmd) {
		BT_DBG("Unable to find a pending command");
		return;
	}

	cmd->cmd_complete(cmd, mgmt_status(status));
	mgmt_pending_remove(cmd);
}
/**
 * @brief Callback function for LE pairing completion.
 * @param conn The HCI connection.
 * @param status The status of the pairing operation.
 */
static void le_pairing_complete_cb(struct hci_conn *conn, u8 status)
{
	struct mgmt_pending_cmd *cmd;

	BT_DBG("status %u", status);

	if (!status)
		return;

	cmd = find_pairing(conn);
	if (!cmd) {
		BT_DBG("Unable to find a pending command");
		return;
	}

	cmd->cmd_complete(cmd, mgmt_status(status));
	mgmt_pending_remove(cmd);
}
/**
 * @brief Handles the MGMT_OP_PAIR_DEVICE command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int pair_device(struct sock *sk, struct hci_dev *hdev, void *data,
		       u16 len)
{
	struct mgmt_cp_pair_device *cp = data;
	struct mgmt_rp_pair_device rp;
	struct mgmt_pending_cmd *cmd;
	u8 sec_level, auth_type;
	struct hci_conn *conn;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	memset(&rp, 0, sizeof(rp));
	bacpy(&rp.addr.bdaddr, &cp->addr.bdaddr);
	rp.addr.type = cp->addr.type;

	if (!bdaddr_type_is_valid(cp->addr.type))
		return mgmt_cmd_complete(sk, hdev->id, MGMT_OP_PAIR_DEVICE,
					 MGMT_STATUS_INVALID_PARAMS,
					 &rp, sizeof(rp));

	if (cp->io_cap > SMP_IO_KEYBOARD_DISPLAY)
		return mgmt_cmd_complete(sk, hdev->id, MGMT_OP_PAIR_DEVICE,
					 MGMT_STATUS_INVALID_PARAMS,
					 &rp, sizeof(rp));

	hci_dev_lock(hdev);

	if (!hdev_is_powered(hdev)) {
		err = mgmt_cmd_complete(sk, hdev->id, MGMT_OP_PAIR_DEVICE,
					MGMT_STATUS_NOT_POWERED, &rp,
					sizeof(rp));
		goto unlock;
	}

	if (hci_bdaddr_is_paired(hdev, &cp->addr.bdaddr, cp->addr.type)) {
		err = mgmt_cmd_complete(sk, hdev->id, MGMT_OP_PAIR_DEVICE,
					MGMT_STATUS_ALREADY_PAIRED, &rp,
					sizeof(rp));
		goto unlock;
	}

	sec_level = BT_SECURITY_MEDIUM;
	auth_type = HCI_AT_DEDICATED_BONDING;

	if (cp->addr.type == BDADDR_BREDR) {
		conn = hci_connect_acl(hdev, &cp->addr.bdaddr, sec_level,
				       auth_type, CONN_REASON_PAIR_DEVICE,
				       HCI_ACL_CONN_TIMEOUT);
	} else {
		u8 addr_type = le_addr_type(cp->addr.type);
		struct hci_conn_params *p;

		/* When pairing a new device, it is expected to remember
		 * this device for future connections. Adding the connection
		 * parameter information ahead of time allows tracking
		 * of the peripheral preferred values and will speed up any
		 * further connection establishment.
		 *
		 * If connection parameters already exist, then they
		 * will be kept and this function does nothing.
		 */
		p = hci_conn_params_add(hdev, &cp->addr.bdaddr, addr_type);
		if (!p) {
			err = -EIO;
			goto unlock;
		}

		if (p->auto_connect == HCI_AUTO_CONN_EXPLICIT)
			p->auto_connect = HCI_AUTO_CONN_DISABLED;

		conn = hci_connect_le_scan(hdev, &cp->addr.bdaddr, addr_type,
					   sec_level, HCI_LE_CONN_TIMEOUT,
					   CONN_REASON_PAIR_DEVICE);
	}

	if (IS_ERR(conn)) {
		int status;

		if (PTR_ERR(conn) == -EBUSY)
			status = MGMT_STATUS_BUSY;
		else if (PTR_ERR(conn) == -EOPNOTSUPP)
			status = MGMT_STATUS_NOT_SUPPORTED;
		else if (PTR_ERR(conn) == -ECONNREFUSED)
			status = MGMT_STATUS_REJECTED;
		else
			status = MGMT_STATUS_CONNECT_FAILED;

		err = mgmt_cmd_complete(sk, hdev->id, MGMT_OP_PAIR_DEVICE,
					status, &rp, sizeof(rp));
		goto unlock;
	}

	if (conn->connect_cfm_cb) {
		hci_conn_drop(conn);
		err = mgmt_cmd_complete(sk, hdev->id, MGMT_OP_PAIR_DEVICE,
					MGMT_STATUS_BUSY, &rp, sizeof(rp));
		goto unlock;
	}

	cmd = mgmt_pending_add(sk, MGMT_OP_PAIR_DEVICE, hdev, data, len);
	if (!cmd) {
		err = -ENOMEM;
		hci_conn_drop(conn);
		goto unlock;
	}

	cmd->cmd_complete = pairing_complete;

	/* For LE, just connecting isn't a proof that the pairing finished */
	if (cp->addr.type == BDADDR_BREDR) {
		conn->connect_cfm_cb = pairing_complete_cb;
		conn->security_cfm_cb = pairing_complete_cb;
		conn->disconn_cfm_cb = pairing_complete_cb;
	} else {
		conn->connect_cfm_cb = le_pairing_complete_cb;
		conn->security_cfm_cb = le_pairing_complete_cb;
		conn->disconn_cfm_cb = le_pairing_complete_cb;
	}

	conn->io_capability = cp->io_cap;
	cmd->user_data = hci_conn_get(conn);

	if ((conn->state == BT_CONNECTED || conn->state == BT_CONFIG) &&
	    hci_conn_security(conn, sec_level, auth_type, true)) {
		cmd->cmd_complete(cmd, 0);
		mgmt_pending_remove(cmd);
	}

	err = 0;

unlock:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Handles the MGMT_OP_CANCEL_PAIR_DEVICE command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int cancel_pair_device(struct sock *sk, struct hci_dev *hdev, void *data,
			      u16 len)
{
	struct mgmt_addr_info *addr = data;
	struct mgmt_pending_cmd *cmd;
	struct hci_conn *conn;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	hci_dev_lock(hdev);

	if (!hdev_is_powered(hdev)) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_CANCEL_PAIR_DEVICE,
				      MGMT_STATUS_NOT_POWERED);
		goto unlock;
	}

	cmd = pending_find(MGMT_OP_PAIR_DEVICE, hdev);
	if (!cmd) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_CANCEL_PAIR_DEVICE,
				      MGMT_STATUS_INVALID_PARAMS);
		goto unlock;
	}

	conn = cmd->user_data;

	if (bacmp(&addr->bdaddr, &conn->dst) != 0) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_CANCEL_PAIR_DEVICE,
				      MGMT_STATUS_INVALID_PARAMS);
		goto unlock;
	}

	cmd->cmd_complete(cmd, MGMT_STATUS_CANCELLED);
	mgmt_pending_remove(cmd);

	err = mgmt_cmd_complete(sk, hdev->id, MGMT_OP_CANCEL_PAIR_DEVICE, 0,
				addr, sizeof(*addr));

	/* Since user doesn't want to proceed with the connection, abort any
	 * ongoing pairing and then terminate the link if it was created
	 * because of the pair device action.
	 */
	if (addr->type == BDADDR_BREDR)
		hci_remove_link_key(hdev, &addr->bdaddr);
	else
		smp_cancel_and_remove_pairing(hdev, &addr->bdaddr,
					      le_addr_type(addr->type));

	if (conn->conn_reason == CONN_REASON_PAIR_DEVICE)
		hci_abort_conn(conn, HCI_ERROR_REMOTE_USER_TERM);

unlock:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Handles a user pairing response.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param addr The address of the device being paired.
 * @param mgmt_op The management opcode.
 * @param hci_op The HCI opcode.
 * @param passkey The passkey (if any).
 * @return 0 on success, or a negative error code on failure.
 */
static int user_pairing_resp(struct sock *sk, struct hci_dev *hdev,
			     struct mgmt_addr_info *addr, u16 mgmt_op,
			     u16 hci_op, __le32 passkey)
{
	struct mgmt_pending_cmd *cmd;
	struct hci_conn *conn;
	int err;

	hci_dev_lock(hdev);

	if (!hdev_is_powered(hdev)) {
		err = mgmt_cmd_complete(sk, hdev->id, mgmt_op,
					MGMT_STATUS_NOT_POWERED, addr,
					sizeof(*addr));
		goto done;
	}

	if (addr->type == BDADDR_BREDR)
		conn = hci_conn_hash_lookup_ba(hdev, ACL_LINK, &addr->bdaddr);
	else
		conn = hci_conn_hash_lookup_le(hdev, &addr->bdaddr,
					       le_addr_type(addr->type));

	if (!conn) {
		err = mgmt_cmd_complete(sk, hdev->id, mgmt_op,
					MGMT_STATUS_NOT_CONNECTED, addr,
					sizeof(*addr));
		goto done;
	}

	if (addr->type == BDADDR_LE_PUBLIC || addr->type == BDADDR_LE_RANDOM) {
		err = smp_user_confirm_reply(conn, mgmt_op, passkey);
		if (!err)
			err = mgmt_cmd_complete(sk, hdev->id, mgmt_op,
						MGMT_STATUS_SUCCESS, addr,
						sizeof(*addr));
		else
			err = mgmt_cmd_complete(sk, hdev->id, mgmt_op,
						MGMT_STATUS_FAILED, addr,
						sizeof(*addr));

		goto done;
	}

	cmd = mgmt_pending_add(sk, mgmt_op, hdev, addr, sizeof(*addr));
	if (!cmd) {
		err = -ENOMEM;
		goto done;
	}

	cmd->cmd_complete = addr_cmd_complete;

	/* Continue with pairing via HCI */
	if (hci_op == HCI_OP_USER_PASSKEY_REPLY) {
		struct hci_cp_user_passkey_reply cp;

		bacpy(&cp.bdaddr, &addr->bdaddr);
		cp.passkey = passkey;
		err = hci_send_cmd(hdev, hci_op, sizeof(cp), &cp);
	} else
		err = hci_send_cmd(hdev, hci_op, sizeof(addr->bdaddr),
				   &addr->bdaddr);

	if (err < 0)
		mgmt_pending_remove(cmd);

done:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Handles the MGMT_OP_PIN_CODE_NEG_REPLY command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int pin_code_neg_reply(struct sock *sk, struct hci_dev *hdev,
			      void *data, u16 len)
{
	struct mgmt_cp_pin_code_neg_reply *cp = data;

	bt_dev_dbg(hdev, "sock %p", sk);

	return user_pairing_resp(sk, hdev, &cp->addr,
				MGMT_OP_PIN_CODE_NEG_REPLY,
				HCI_OP_PIN_CODE_NEG_REPLY, 0);
}
/**
 * @brief Handles the MGMT_OP_USER_CONFIRM_REPLY command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int user_confirm_reply(struct sock *sk, struct hci_dev *hdev, void *data,
			      u16 len)
{
	struct mgmt_cp_user_confirm_reply *cp = data;

	bt_dev_dbg(hdev, "sock %p", sk);

	if (len != sizeof(*cp))
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_USER_CONFIRM_REPLY,
				       MGMT_STATUS_INVALID_PARAMS);

	return user_pairing_resp(sk, hdev, &cp->addr,
				 MGMT_OP_USER_CONFIRM_REPLY,
				 HCI_OP_USER_CONFIRM_REPLY, 0);
}
/**
 * @brief Handles the MGMT_OP_USER_CONFIRM_NEG_REPLY command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int user_confirm_neg_reply(struct sock *sk, struct hci_dev *hdev,
				  void *data, u16 len)
{
	struct mgmt_cp_user_confirm_neg_reply *cp = data;

	bt_dev_dbg(hdev, "sock %p", sk);

	return user_pairing_resp(sk, hdev, &cp->addr,
				 MGMT_OP_USER_CONFIRM_NEG_REPLY,
				 HCI_OP_USER_CONFIRM_NEG_REPLY, 0);
}
/**
 * @brief Handles the MGMT_OP_USER_PASSKEY_REPLY command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int user_passkey_reply(struct sock *sk, struct hci_dev *hdev, void *data,
			      u16 len)
{
	struct mgmt_cp_user_passkey_reply *cp = data;

	bt_dev_dbg(hdev, "sock %p", sk);

	return user_pairing_resp(sk, hdev, &cp->addr,
				 MGMT_OP_USER_PASSKEY_REPLY,
				 HCI_OP_USER_PASSKEY_REPLY, cp->passkey);
}
/**
 * @brief Handles the MGMT_OP_USER_PASSKEY_NEG_REPLY command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int user_passkey_neg_reply(struct sock *sk, struct hci_dev *hdev,
				  void *data, u16 len)
{
	struct mgmt_cp_user_passkey_neg_reply *cp = data;

	bt_dev_dbg(hdev, "sock %p", sk);

	return user_pairing_resp(sk, hdev, &cp->addr,
				 MGMT_OP_USER_PASSKEY_NEG_REPLY,
				 HCI_OP_USER_PASSKEY_NEG_REPLY, 0);
}
/**
 * @brief Synchronously updates advertising when the local name changes.
 * @param hdev The HCI device.
 * @param data Unused.
 * @return 0 on success, or a negative error code on failure.
 */
static int name_changed_sync(struct hci_dev *hdev, void *data)
{
	return adv_expire_sync(hdev, MGMT_ADV_FLAG_LOCAL_NAME);
}

/**
 * @brief Handles the completion of the set_local_name command.
 * @param hdev The HCI device.
 * @param data The pending command data.
 * @param err The error code from the HCI command.
 */
static void set_name_complete(struct hci_dev *hdev, void *data, int err)
{
	struct mgmt_pending_cmd *cmd = data;
	struct mgmt_cp_set_local_name *cp = cmd->param;
	u8 status = mgmt_status(err);

	bt_dev_dbg(hdev, "err %d", err);

	if (err == -ECANCELED ||
	    cmd != pending_find(MGMT_OP_SET_LOCAL_NAME, hdev))
		return;

	if (status) {
		mgmt_cmd_status(cmd->sk, hdev->id, MGMT_OP_SET_LOCAL_NAME,
				status);
	} else {
		mgmt_cmd_complete(cmd->sk, hdev->id, MGMT_OP_SET_LOCAL_NAME, 0,
				  cp, sizeof(*cp));

		if (hci_dev_test_flag(hdev, HCI_LE_ADV))
			hci_cmd_sync_queue(hdev, name_changed_sync, NULL, NULL);
	}

	mgmt_pending_remove(cmd);
}
/**
 * @brief Synchronously updates the local name and EIR data.
 * @param hdev The HCI device.
 * @param data Unused.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_name_sync(struct hci_dev *hdev, void *data)
{
	if (lmp_bredr_capable(hdev)) {
		hci_update_name_sync(hdev);
		hci_update_eir_sync(hdev);
	}

	/* The name is stored in the scan response data and so
	 * no need to update the advertising data here.
	 */
	if (lmp_le_capable(hdev) && hci_dev_test_flag(hdev, HCI_ADVERTISING))
		hci_update_scan_rsp_data_sync(hdev, hdev->cur_adv_instance);

	return 0;
}
/**
 * @brief Handles the MGMT_OP_SET_LOCAL_NAME command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_local_name(struct sock *sk, struct hci_dev *hdev, void *data,
			  u16 len)
{
	struct mgmt_cp_set_local_name *cp = data;
	struct mgmt_pending_cmd *cmd;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	hci_dev_lock(hdev);

	/* If the old values are the same as the new ones just return a
	 * direct command complete event.
	 */
	if (!memcmp(hdev->dev_name, cp->name, sizeof(hdev->dev_name)) &&
	    !memcmp(hdev->short_name, cp->short_name,
		    sizeof(hdev->short_name))) {
		err = mgmt_cmd_complete(sk, hdev->id, MGMT_OP_SET_LOCAL_NAME, 0,
					data, len);
		goto failed;
	}

	memcpy(hdev->short_name, cp->short_name, sizeof(hdev->short_name));

	if (!hdev_is_powered(hdev)) {
		memcpy(hdev->dev_name, cp->name, sizeof(hdev->dev_name));

		err = mgmt_cmd_complete(sk, hdev->id, MGMT_OP_SET_LOCAL_NAME, 0,
					data, len);
		if (err < 0)
			goto failed;

		err = mgmt_limited_event(MGMT_EV_LOCAL_NAME_CHANGED, hdev, data,
					 len, HCI_MGMT_LOCAL_NAME_EVENTS, sk);
		ext_info_changed(hdev, sk);

		goto failed;
	}

	cmd = mgmt_pending_add(sk, MGMT_OP_SET_LOCAL_NAME, hdev, data, len);
	if (!cmd)
		err = -ENOMEM;
	else
		err = hci_cmd_sync_queue(hdev, set_name_sync, cmd,
					 set_name_complete);

	if (err < 0) {
		err = mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_LOCAL_NAME,
				      MGMT_STATUS_FAILED);

		if (cmd)
			mgmt_pending_remove(cmd);

		goto failed;
	}

	memcpy(hdev->dev_name, cp->name, sizeof(hdev->dev_name));

failed:
	hci_dev_unlock(hdev);
	return err;
}
/**
 * @brief Synchronously updates advertising when the appearance changes.
 * @param hdev The HCI device.
 * @param data Unused.
 * @return 0 on success, or a negative error code on failure.
 */
static int appearance_changed_sync(struct hci_dev *hdev, void *data)
{
	return adv_expire_sync(hdev, MGMT_ADV_FLAG_APPEARANCE);
}
/**
 * @brief Handles the MGMT_OP_SET_APPEARANCE command.
 * @param sk The socket from which the command was received.
 * @param hdev The HCI device.
 * @param data The command data.
 * @param len The length of the command data.
 * @return 0 on success, or a negative error code on failure.
 */
static int set_appearance(struct sock *sk, struct hci_dev *hdev, void *data,
			  u16 len)
{
	struct mgmt_cp_set_appearance *cp = data;
	u16 appearance;
	int err;

	bt_dev_dbg(hdev, "sock %p", sk);

	if (!lmp_le_capable(hdev))
		return mgmt_cmd_status(sk, hdev->id, MGMT_OP_SET_APPEARANCE,
				       MGMT_STATUS_NOT_SUPPORTED);

	appearance = le16_to_cpu(cp->appearance);

	hci_dev_lock(hdev);

	if (hdev->appearance != appearance) {
		hdev->appearance = appearance;

		if (hci_dev_test_flag(hdev, HCI_LE_ADV))
			hci_cmd_sync_queue(hdev, appearance_changed_sync, NULL,
					   NULL);

		ext_info_changed(hdev, sk);
	}

	err = mgmt_cmd_complete(sk, hdev->id, MGMT_OP_SET_APPEARANCE, 0, NULL,
				0);

	hci_dev_unlock(hdev);

	return err;
}
/**
 * @brief Initializes the management channel.
 * @return 0 on success, or a negative error code on failure.
 *
 * This function registers the management channel with the HCI layer.
 */
int mgmt_init(void)
{
	return hci_mgmt_chan_register(&chan);
}
/**
 * @brief Exits the management channel.
 *
 * This function unregisters the management channel from the HCI layer.
 */
void mgmt_exit(void)
{
	hci_mgmt_chan_unregister(&chan);
}

/**
 * @brief Cleans up management resources for a closing socket.
 * @param sk The socket that is closing.
 *
 * This function iterates through all HCI devices and removes any pending
 * mesh transmissions associated with the closing socket.
 */
void mgmt_cleanup(struct sock *sk)
{
	struct mgmt_mesh_tx *mesh_tx;
	struct hci_dev *hdev;

	read_lock(&hci_dev_list_lock);

	list_for_each_entry(hdev, &hci_dev_list, list) {
		do {
			mesh_tx = mgmt_mesh_next(hdev, sk);

			if (mesh_tx)
				mesh_send_complete(hdev, mesh_tx, true);
		} while (mesh_tx);
	}

	read_unlock(&hci_dev_list_lock);
}