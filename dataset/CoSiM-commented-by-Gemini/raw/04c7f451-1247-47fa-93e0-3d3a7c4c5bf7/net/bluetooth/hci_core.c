/*
   BlueZ - Bluetooth protocol stack for Linux
   Copyright (C) 2000-2001 Qualcomm Incorporated
   Copyright (C) 2011 ProFUSION Embedded Systems

   Written 2000,2001 by Maxim Krasnyansky <maxk@qualcomm.com>

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License version 2 as
   published by the Free Software Foundation;

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD PARTY RIGHTS.
   IN NO EVENT SHALL THE COPYRIGHT HOLDER(S) AND AUTHOR(S) BE LIABLE FOR ANY
   CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES
   WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
   ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
   OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

   ALL LIABILITY, INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PATENTS,
   COPYRIGHTS, TRADEMARKS OR OTHER RIGHTS, RELATING TO USE OF THIS
   SOFTWARE IS DISCLAIMED.
*/

/**
 * @file
 * @brief This file is the core of the HCI layer in the Bluetooth stack.
 *
 * It manages HCI devices, including their registration, removal, and state
 * transitions. It handles HCI commands, events, and data packets, and
 * provides an interface for upper layers to interact with Bluetooth
 * controllers. It also implements key functionalities such as device
 * discovery (inquiry), connection management, and security features like
 * pairing and encryption.
 */

#include <linux/export.h>
#include <linux/rfkill.h>
#include <linux/debugfs.h>
#include <linux/crypto.h>
#include <linux/kcov.h>
#include <linux/property.h>
#include <linux/suspend.h>
#include <linux/wait.h>
#include <linux/unaligned.h>

#include <net/bluetooth/bluetooth.h>
#include <net/bluetooth/hci_core.h>
#include <net/bluetooth/l2cap.h>
#include <net/bluetooth/mgmt.h>

#include "hci_debugfs.h"
#include "smp.h"
#include "leds.h"
#include "msft.h"
#include "aosp.h"
#include "hci_codec.h"

static void hci_rx_work(struct work_struct *work);
static void hci_cmd_work(struct work_struct *work);
static void hci_tx_work(struct work_struct *work);

/* HCI device list */
LIST_HEAD(hci_dev_list);
DEFINE_RWLOCK(hci_dev_list_lock);

/* HCI callback list */
LIST_HEAD(hci_cb_list);
DEFINE_MUTEX(hci_cb_list_lock);

/* HCI ID Numbering */
static DEFINE_IDA(hci_index_ida);

/**
 * @brief Retrieves an HCI device by its index.
 *
 * This function looks up an HCI device in the global list by its index.
 * It acquires a reference to the device, which must be released with
 * hci_dev_put() when no longer needed.
 *
 * @param index The index of the HCI device to retrieve.
 * @param srcu_index Optional pointer to store the SRCU index for read-side critical sections.
 * @return A pointer to the HCI device, or NULL if not found.
 */
static struct hci_dev *__hci_dev_get(int index, int *srcu_index)
{
	struct hci_dev *hdev = NULL, *d;

	BT_DBG("%d", index);

	if (index < 0)
		return NULL;

	read_lock(&hci_dev_list_lock);
	list_for_each_entry(d, &hci_dev_list, list) {
		if (d->id == index) {
			hdev = hci_dev_hold(d);
			if (srcu_index)
				*srcu_index = srcu_read_lock(&d->srcu);
			break;
		}
	}
	read_unlock(&hci_dev_list_lock);
	return hdev;
}

struct hci_dev *hci_dev_get(int index)
{
	return __hci_dev_get(index, NULL);
}

/**
 * @brief Retrieves an HCI device and locks it for SRCU-based read-side critical sections.
 * @param index The index of the HCI device.
 * @param srcu_index Pointer to an integer where the SRCU index will be stored.
 * @return A pointer to the HCI device, or NULL if not found.
 */
static struct hci_dev *hci_dev_get_srcu(int index, int *srcu_index)
{
	return __hci_dev_get(index, srcu_index);
}

/**
 * @brief Releases the SRCU lock and the reference to an HCI device.
 * @param hdev The HCI device.
 * @param srcu_index The SRCU index returned by hci_dev_get_srcu.
 */
static void hci_dev_put_srcu(struct hci_dev *hdev, int srcu_index)
{
	srcu_read_unlock(&hdev->srcu, srcu_index);
	hci_dev_put(hdev);
}

/* ---- Inquiry support ---- */

/**
 * @brief Checks if the device discovery process is active.
 * @param hdev The HCI device.
 * @return True if discovery is active, false otherwise.
 */
bool hci_discovery_active(struct hci_dev *hdev)
{
	struct discovery_state *discov = &hdev->discovery;

	switch (discov->state) {
	case DISCOVERY_FINDING:
	case DISCOVERY_RESOLVING:
		return true;

	default:
		return false;
	}
}

/**
 * @brief Sets the state of the device discovery process.
 * @param hdev The HCI device.
 * @param state The new discovery state.
 *
 * This function manages the state machine for device discovery and sends
 * appropriate management events to user space.
 */
void hci_discovery_set_state(struct hci_dev *hdev, int state)
{
	int old_state = hdev->discovery.state;

	if (old_state == state)
		return;

	hdev->discovery.state = state;

	switch (state) {
	case DISCOVERY_STOPPED:
		hci_update_passive_scan(hdev);

		if (old_state != DISCOVERY_STARTING)
			mgmt_discovering(hdev, 0);
		break;
	case DISCOVERY_STARTING:
		break;
	case DISCOVERY_FINDING:
		mgmt_discovering(hdev, 1);
		break;
	case DISCOVERY_RESOLVING:
		break;
	case DISCOVERY_STOPPING:
		break;
	}

	bt_dev_dbg(hdev, "state %u -> %u", old_state, state);
}

/**
 * @brief Flushes the inquiry cache of an HCI device.
 * @param hdev The HCI device.
 *
 * This function removes all entries from the inquiry cache, which stores
 * information about discovered devices.
 */
void hci_inquiry_cache_flush(struct hci_dev *hdev)
{
	struct discovery_state *cache = &hdev->discovery;
	struct inquiry_entry *p, *n;

	list_for_each_entry_safe(p, n, &cache->all, all) {
		list_del(&p->all);
		kfree(p);
	}

	INIT_LIST_HEAD(&cache->unknown);
	INIT_LIST_HEAD(&cache->resolve);
}
/**
 * @brief Looks up an entry in the inquiry cache by Bluetooth address.
 * @param hdev The HCI device.
 * @param bdaddr The Bluetooth address of the device to look up.
 * @return A pointer to the inquiry entry, or NULL if not found.
 */
struct inquiry_entry *hci_inquiry_cache_lookup(struct hci_dev *hdev,
					       bdaddr_t *bdaddr)
{
	struct discovery_state *cache = &hdev->discovery;
	struct inquiry_entry *e;

	BT_DBG("cache %p, %pMR", cache, bdaddr);

	list_for_each_entry(e, &cache->all, all) {
		if (!bacmp(&e->data.bdaddr, bdaddr))
			return e;
	}

	return NULL;
}

/**
 * @brief Looks up an entry in the inquiry cache for devices with an unknown name.
 * @param hdev The HCI device.
 * @param bdaddr The Bluetooth address of the device to look up.
 * @return A pointer to the inquiry entry, or NULL if not found.
 */
struct inquiry_entry *hci_inquiry_cache_lookup_unknown(struct hci_dev *hdev,
						       bdaddr_t *bdaddr)
{
	struct discovery_state *cache = &hdev->discovery;
	struct inquiry_entry *e;

	BT_DBG("cache %p, %pMR", cache, bdaddr);

	list_for_each_entry(e, &cache->unknown, list) {
		if (!bacmp(&e->data.bdaddr, bdaddr))
			return e;
	}

	return NULL;
}
/**
 * @brief Looks up an entry in the inquiry cache that needs name resolution.
 * @param hdev The HCI device.
 * @param bdaddr The Bluetooth address of the device to look up. If BDADDR_ANY, finds the first entry with the given state.
 * @param state The desired name resolution state.
 * @return A pointer to the inquiry entry, or NULL if not found.
 */
struct inquiry_entry *hci_inquiry_cache_lookup_resolve(struct hci_dev *hdev,
						       bdaddr_t *bdaddr,
						       int state)
{
	struct discovery_state *cache = &hdev->discovery;
	struct inquiry_entry *e;

	BT_DBG("cache %p bdaddr %pMR state %d", cache, bdaddr, state);

	list_for_each_entry(e, &cache->resolve, list) {
		if (!bacmp(bdaddr, BDADDR_ANY) && e->name_state == state)
			return e;
		if (!bacmp(&e->data.bdaddr, bdaddr))
			return e;
	}

	return NULL;
}

/**
 * @brief Updates the position of an inquiry entry in the resolve list based on RSSI.
 * @param hdev The HCI device.
 * @param ie The inquiry entry to update.
 *
 * This function reorders the list of devices needing name resolution, prioritizing
 * those with stronger signals.
 */
void hci_inquiry_cache_update_resolve(struct hci_dev *hdev,
				      struct inquiry_entry *ie)
{
	struct discovery_state *cache = &hdev->discovery;
	struct list_head *pos = &cache->resolve;
	struct inquiry_entry *p;

	list_del(&ie->list);

	list_for_each_entry(p, &cache->resolve, list) {
		if (p->name_state != NAME_PENDING &&
		    abs(p->data.rssi) >= abs(ie->data.rssi))
			break;
		pos = &p->list;
	}

	list_add(&ie->list, pos);
}

/**
 * @brief Updates the inquiry cache with new data from a discovered device.
 * @param hdev The HCI device.
 * @param data The inquiry data for the device.
 * @param name_known True if the device's name is known, false otherwise.
 * @return A bitmask of flags indicating the status of the update.
 */
u32 hci_inquiry_cache_update(struct hci_dev *hdev, struct inquiry_data *data,
			     bool name_known)
{
	struct discovery_state *cache = &hdev->discovery;
	struct inquiry_entry *ie;
	u32 flags = 0;

	BT_DBG("cache %p, %pMR", cache, &data->bdaddr);

	hci_remove_remote_oob_data(hdev, &data->bdaddr, BDADDR_BREDR);

	if (!data->ssp_mode)
		flags |= MGMT_DEV_FOUND_LEGACY_PAIRING;

	ie = hci_inquiry_cache_lookup(hdev, &data->bdaddr);
	if (ie) {
		if (!ie->data.ssp_mode)
			flags |= MGMT_DEV_FOUND_LEGACY_PAIRING;

		if (ie->name_state == NAME_NEEDED &&
		    data->rssi != ie->data.rssi) {
			ie->data.rssi = data->rssi;
			hci_inquiry_cache_update_resolve(hdev, ie);
		}

		goto update;
	}

	/* Entry not in the cache. Add new one. */
	ie = kzalloc(sizeof(*ie), GFP_KERNEL);
	if (!ie) {
		flags |= MGMT_DEV_FOUND_CONFIRM_NAME;
		goto done;
	}

	list_add(&ie->all, &cache->all);

	if (name_known) {
		ie->name_state = NAME_KNOWN;
	} else {
		ie->name_state = NAME_NOT_KNOWN;
		list_add(&ie->list, &cache->unknown);
	}

update:
	if (name_known && ie->name_state != NAME_KNOWN &&
	    ie->name_state != NAME_PENDING) {
		ie->name_state = NAME_KNOWN;
		list_del(&ie->list);
	}

	memcpy(&ie->data, data, sizeof(*data));
	ie->timestamp = jiffies;
	cache->timestamp = jiffies;

	if (ie->name_state == NAME_NOT_KNOWN)
		flags |= MGMT_DEV_FOUND_CONFIRM_NAME;

done:
	return flags;
}
/**
 * @brief Dumps the contents of the inquiry cache into a buffer.
 * @param hdev The HCI device.
 * @param num The maximum number of entries to dump.
 * @param buf The buffer to dump the entries into.
 * @return The number of entries copied to the buffer.
 */
static int inquiry_cache_dump(struct hci_dev *hdev, int num, __u8 *buf)
{
	struct discovery_state *cache = &hdev->discovery;
	struct inquiry_info *info = (struct inquiry_info *) buf;
	struct inquiry_entry *e;
	int copied = 0;

	list_for_each_entry(e, &cache->all, all) {
		struct inquiry_data *data = &e->data;

		if (copied >= num)
			break;

		bacpy(&info->bdaddr, &data->bdaddr);
		info->pscan_rep_mode	= data->pscan_rep_mode;
		info->pscan_period_mode	= data->pscan_period_mode;
		info->pscan_mode	= data->pscan_mode;
		memcpy(info->dev_class, data->dev_class, 3);
		info->clock_offset	= data->clock_offset;

		info++;
		copied++;
	}

	BT_DBG("cache %p, copied %d", cache, copied);
	return copied;
}

/**
 * @brief Handles the HCI inquiry ioctl.
 * @param arg A pointer to a user-space struct hci_inquiry_req.
 * @return 0 on success, or a negative error code on failure.
 *
 * This function initiates a Bluetooth device discovery process and returns
 * the results to user space.
 */
int hci_inquiry(void __user *arg)
{
	__u8 __user *ptr = arg;
	struct hci_inquiry_req ir;
	struct hci_dev *hdev;
	int err = 0, do_inquiry = 0, max_rsp;
	__u8 *buf;

	if (copy_from_user(&ir, ptr, sizeof(ir)))
		return -EFAULT;

	hdev = hci_dev_get(ir.dev_id);
	if (!hdev)
		return -ENODEV;

	if (hci_dev_test_flag(hdev, HCI_USER_CHANNEL)) {
		err = -EBUSY;
		goto done;
	}

	if (hci_dev_test_flag(hdev, HCI_UNCONFIGURED)) {
		err = -EOPNOTSUPP;
		goto done;
	}

	if (!hci_dev_test_flag(hdev, HCI_BREDR_ENABLED)) {
		err = -EOPNOTSUPP;
		goto done;
	}

	/* Restrict maximum inquiry length to 60 seconds */
	if (ir.length > 60) {
		err = -EINVAL;
		goto done;
	}

	hci_dev_lock(hdev);
	if (inquiry_cache_age(hdev) > INQUIRY_CACHE_AGE_MAX ||
	    inquiry_cache_empty(hdev) || ir.flags & IREQ_CACHE_FLUSH) {
		hci_inquiry_cache_flush(hdev);
		do_inquiry = 1;
	}
	hci_dev_unlock(hdev);

	if (do_inquiry) {
		hci_req_sync_lock(hdev);
		err = hci_inquiry_sync(hdev, ir.length, ir.num_rsp);
		hci_req_sync_unlock(hdev);

		if (err < 0)
			goto done;

		/* Wait until Inquiry procedure finishes (HCI_INQUIRY flag is
		 * cleared). If it is interrupted by a signal, return -EINTR.
		 */
		if (wait_on_bit(&hdev->flags, HCI_INQUIRY,
				TASK_INTERRUPTIBLE)) {
			err = -EINTR;
			goto done;
		}
	}

	/* for unlimited number of responses we will use buffer with
	 * 255 entries
	 */
	max_rsp = (ir.num_rsp == 0) ? 255 : ir.num_rsp;

	/* cache_dump can't sleep. Therefore we allocate temp buffer and then
	 * copy it to the user space.
	 */
	buf = kmalloc_array(max_rsp, sizeof(struct inquiry_info), GFP_KERNEL);
	if (!buf) {
		err = -ENOMEM;
		goto done;
	}

	hci_dev_lock(hdev);
	ir.num_rsp = inquiry_cache_dump(hdev, max_rsp, buf);
	hci_dev_unlock(hdev);

	BT_DBG("num_rsp %d", ir.num_rsp);

	if (!copy_to_user(ptr, &ir, sizeof(ir))) {
		ptr += sizeof(ir);
		if (copy_to_user(ptr, buf, sizeof(struct inquiry_info) *
				 ir.num_rsp))
			err = -EFAULT;
	} else
		err = -EFAULT;

	kfree(buf);

done:
	hci_dev_put(hdev);
	return err;
}
/**
 * @brief Opens and initializes an HCI device.
 * @param hdev The HCI device to open.
 * @return 0 on success, or a negative error code on failure.
 */
static int hci_dev_do_open(struct hci_dev *hdev)
{
	int ret = 0;

	BT_DBG("%s %p", hdev->name, hdev);

	hci_req_sync_lock(hdev);

	ret = hci_dev_open_sync(hdev);

	hci_req_sync_unlock(hdev);
	return ret;
}

/* ---- HCI ioctl helpers ---- */
/**
 * @brief Opens an HCI device by its index.
 * @param dev The index of the HCI device to open.
 * @return 0 on success, or a negative error code on failure.
 */
int hci_dev_open(__u16 dev)
{
	struct hci_dev *hdev;
	int err;

	hdev = hci_dev_get(dev);
	if (!hdev)
		return -ENODEV;

	/* Devices that are marked as unconfigured can only be powered
	 * up as user channel. Trying to bring them up as normal devices
	 * will result into a failure. Only user channel operation is
	 * possible.
	 *
	 * When this function is called for a user channel, the flag
	 * HCI_USER_CHANNEL will be set first before attempting to
	 * open the device.
	 */
	if (hci_dev_test_flag(hdev, HCI_UNCONFIGURED) &&
	    !hci_dev_test_flag(hdev, HCI_USER_CHANNEL)) {
		err = -EOPNOTSUPP;
		goto done;
	}

	/* We need to ensure that no other power on/off work is pending
	 * before proceeding to call hci_dev_do_open. This is
	 * particularly important if the setup procedure has not yet
	 * completed.
	 */
	if (hci_dev_test_and_clear_flag(hdev, HCI_AUTO_OFF))
		cancel_delayed_work(&hdev->power_off);

	/* After this call it is guaranteed that the setup procedure
	 * has finished. This means that error conditions like RFKILL
	 * or no valid public or static random address apply.
	 */
	flush_workqueue(hdev->req_workqueue);

	/* For controllers not using the management interface and that
	 * are brought up using legacy ioctl, set the HCI_BONDABLE bit
	 * so that pairing works for them. Once the management interface
	 * is in use this bit will be cleared again and userspace has
	 * to explicitly enable it.
	 */
	if (!hci_dev_test_flag(hdev, HCI_USER_CHANNEL) &&
	    !hci_dev_test_flag(hdev, HCI_MGMT))
		hci_dev_set_flag(hdev, HCI_BONDABLE);

	err = hci_dev_do_open(hdev);

done:
	hci_dev_put(hdev);
	return err;
}
/**
 * @brief Closes an HCI device.
 * @param hdev The HCI device to close.
 * @return 0 on success, or a negative error code on failure.
 */
int hci_dev_do_close(struct hci_dev *hdev)
{
	int err;

	BT_DBG("%s %p", hdev->name, hdev);

	hci_req_sync_lock(hdev);

	err = hci_dev_close_sync(hdev);

	hci_req_sync_unlock(hdev);

	return err;
}
/**
 * @brief Closes an HCI device by its index.
 * @param dev The index of the HCI device to close.
 * @return 0 on success, or a negative error code on failure.
 */
int hci_dev_close(__u16 dev)
{
	struct hci_dev *hdev;
	int err;

	hdev = hci_dev_get(dev);
	if (!hdev)
		return -ENODEV;

	if (hci_dev_test_flag(hdev, HCI_USER_CHANNEL)) {
		err = -EBUSY;
		goto done;
	}

	cancel_work_sync(&hdev->power_on);
	if (hci_dev_test_and_clear_flag(hdev, HCI_AUTO_OFF))
		cancel_delayed_work(&hdev->power_off);

	err = hci_dev_do_close(hdev);

done:
	hci_dev_put(hdev);
	return err;
}
/**
 * @brief Resets an HCI device.
 * @param hdev The HCI device to reset.
 * @return 0 on success, or a negative error code on failure.
 *
 * This function resets the device by flushing queues, clearing caches,
 * and sending a reset command to the controller.
 */
static int hci_dev_do_reset(struct hci_dev *hdev)
{
	int ret;

	BT_DBG("%s %p", hdev->name, hdev);

	hci_req_sync_lock(hdev);

	/* Drop queues */
	skb_queue_purge(&hdev->rx_q);
	skb_queue_purge(&hdev->cmd_q);

	/* Cancel these to avoid queueing non-chained pending work */
	hci_dev_set_flag(hdev, HCI_CMD_DRAIN_WORKQUEUE);
	/* Wait for
	 *
	 *    if (!hci_dev_test_flag(hdev, HCI_CMD_DRAIN_WORKQUEUE))
	 *        queue_delayed_work(&hdev->{cmd,ncmd}_timer)
	 *
	 * inside RCU section to see the flag or complete scheduling.
	 */
	synchronize_rcu();
	/* Explicitly cancel works in case scheduled after setting the flag. */
	cancel_delayed_work(&hdev->cmd_timer);
	cancel_delayed_work(&hdev->ncmd_timer);

	/* Avoid potential lockdep warnings from the *_flush() calls by
	 * ensuring the workqueue is empty up front.
	 */
	drain_workqueue(hdev->workqueue);

	hci_dev_lock(hdev);
	hci_inquiry_cache_flush(hdev);
	hci_conn_hash_flush(hdev);
	hci_dev_unlock(hdev);

	if (hdev->flush)
		hdev->flush(hdev);

	hci_dev_clear_flag(hdev, HCI_CMD_DRAIN_WORKQUEUE);

	atomic_set(&hdev->cmd_cnt, 1);
	hdev->acl_cnt = 0;
	hdev->sco_cnt = 0;
	hdev->le_cnt = 0;
	hdev->iso_cnt = 0;

	ret = hci_reset_sync(hdev);

	hci_req_sync_unlock(hdev);
	return ret;
}
/**
 * @brief Resets an HCI device by its index.
 * @param dev The index of the HCI device to reset.
 * @return 0 on success, or a negative error code on failure.
 */
int hci_dev_reset(__u16 dev)
{
	struct hci_dev *hdev;
	int err, srcu_index;

	hdev = hci_dev_get_srcu(dev, &srcu_index);
	if (!hdev)
		return -ENODEV;

	if (!test_bit(HCI_UP, &hdev->flags)) {
		err = -ENETDOWN;
		goto done;
	}

	if (hci_dev_test_flag(hdev, HCI_USER_CHANNEL)) {
		err = -EBUSY;
		goto done;
	}

	if (hci_dev_test_flag(hdev, HCI_UNCONFIGURED)) {
		err = -EOPNOTSUPP;
		goto done;
	}

	err = hci_dev_do_reset(hdev);

done:
	hci_dev_put_srcu(hdev, srcu_index);
	return err;
}
/**
 * @brief Resets the statistics of an HCI device.
 * @param dev The index of the HCI device.
 * @return 0 on success, or a negative error code on failure.
 */
int hci_dev_reset_stat(__u16 dev)
{
	struct hci_dev *hdev;
	int ret = 0;

	hdev = hci_dev_get(dev);
	if (!hdev)
		return -ENODEV;

	if (hci_dev_test_flag(hdev, HCI_USER_CHANNEL)) {
		ret = -EBUSY;
		goto done;
	}

	if (hci_dev_test_flag(hdev, HCI_UNCONFIGURED)) {
		ret = -EOPNOTSUPP;
		goto done;
	}

	memset(&hdev->stat, 0, sizeof(struct hci_dev_stats));

done:
	hci_dev_put(hdev);
	return ret;
}
/**
 * @brief Updates the passive scan state of an HCI device based on scan flags.
 * @param hdev The HCI device.
 * @param scan The scan flags.
 */
static void hci_update_passive_scan_state(struct hci_dev *hdev, u8 scan)
{
	bool conn_changed, discov_changed;

	BT_DBG("%s scan 0x%02x", hdev->name, scan);

	if ((scan & SCAN_PAGE))
		conn_changed = !hci_dev_test_and_set_flag(hdev,
							  HCI_CONNECTABLE);
	else
		conn_changed = hci_dev_test_and_clear_flag(hdev,
							   HCI_CONNECTABLE);

	if ((scan & SCAN_INQUIRY)) {
		discov_changed = !hci_dev_test_and_set_flag(hdev,
							    HCI_DISCOVERABLE);
	} else {
		hci_dev_clear_flag(hdev, HCI_LIMITED_DISCOVERABLE);
		discov_changed = hci_dev_test_and_clear_flag(hdev,
							     HCI_DISCOVERABLE);
	}

	if (!hci_dev_test_flag(hdev, HCI_MGMT))
		return;

	if (conn_changed || discov_changed) {
		/* In case this was disabled through mgmt */
		hci_dev_set_flag(hdev, HCI_BREDR_ENABLED);

		if (hci_dev_test_flag(hdev, HCI_LE_ENABLED))
			hci_update_adv_data(hdev, hdev->cur_adv_instance);

		mgmt_new_settings(hdev);
	}
}

/**
 * @brief Handles various HCI device ioctls.
 * @param cmd The ioctl command.
 * @param arg A pointer to user-space data.
 * @return 0 on success, or a negative error code on failure.
 */
int hci_dev_cmd(unsigned int cmd, void __user *arg)
{
	struct hci_dev *hdev;
	struct hci_dev_req dr;
	__le16 policy;
	int err = 0;

	if (copy_from_user(&dr, arg, sizeof(dr)))
		return -EFAULT;

	hdev = hci_dev_get(dr.dev_id);
	if (!hdev)
		return -ENODEV;

	if (hci_dev_test_flag(hdev, HCI_USER_CHANNEL)) {
		err = -EBUSY;
		goto done;
	}

	if (hci_dev_test_flag(hdev, HCI_UNCONFIGURED)) {
		err = -EOPNOTSUPP;
		goto done;
	}

	if (!hci_dev_test_flag(hdev, HCI_BREDR_ENABLED)) {
		err = -EOPNOTSUPP;
		goto done;
	}

	switch (cmd) {
	case HCISETAUTH:
		err = hci_cmd_sync_status(hdev, HCI_OP_WRITE_AUTH_ENABLE,
					  1, &dr.dev_opt, HCI_CMD_TIMEOUT);
		break;

	case HCISETENCRYPT:
		if (!lmp_encrypt_capable(hdev)) {
			err = -EOPNOTSUPP;
			break;
		}

		if (!test_bit(HCI_AUTH, &hdev->flags)) {
			/* Auth must be enabled first */
			err = hci_cmd_sync_status(hdev,
						  HCI_OP_WRITE_AUTH_ENABLE,
						  1, &dr.dev_opt,
						  HCI_CMD_TIMEOUT);
			if (err)
				break;
		}

		err = hci_cmd_sync_status(hdev, HCI_OP_WRITE_ENCRYPT_MODE,
					  1, &dr.dev_opt, HCI_CMD_TIMEOUT);
		break;

	case HCISETSCAN:
		err = hci_cmd_sync_status(hdev, HCI_OP_WRITE_SCAN_ENABLE,
					  1, &dr.dev_opt, HCI_CMD_TIMEOUT);

		/* Ensure that the connectable and discoverable states
		 * get correctly modified as this was a non-mgmt change.
		 */
		if (!err)
			hci_update_passive_scan_state(hdev, dr.dev_opt);
		break;

	case HCISETLINKPOL:
		policy = cpu_to_le16(dr.dev_opt);

		err = hci_cmd_sync_status(hdev, HCI_OP_WRITE_DEF_LINK_POLICY,
					  2, &policy, HCI_CMD_TIMEOUT);
		break;

	case HCISETLINKMODE:
		hdev->link_mode = ((__u16) dr.dev_opt) &
					(HCI_LM_MASTER | HCI_LM_ACCEPT);
		break;

	case HCISETPTYPE:
		if (hdev->pkt_type == (__u16) dr.dev_opt)
			break;

		hdev->pkt_type = (__u16) dr.dev_opt;
		mgmt_phy_configuration_changed(hdev, NULL);
		break;

	case HCISETACLMTU:
		hdev->acl_mtu  = *((__u16 *) &dr.dev_opt + 1);
		hdev->acl_pkts = *((__u16 *) &dr.dev_opt + 0);
		break;

	case HCISETSCOMTU:
		hdev->sco_mtu  = *((__u16 *) &dr.dev_opt + 1);
		hdev->sco_pkts = *((__u16 *) &dr.dev_opt + 0);
		break;

	default:
		err = -EINVAL;
		break;
	}

done:
	hci_dev_put(hdev);
	return err;
}
/**
 * @brief Handles the ioctl to get the list of HCI devices.
 * @param arg A pointer to a user-space struct hci_dev_list_req.
 * @return 0 on success, or a negative error code on failure.
 */
int hci_get_dev_list(void __user *arg)
{
	struct hci_dev *hdev;
	struct hci_dev_list_req *dl;
	struct hci_dev_req *dr;
	int n = 0, err;
	__u16 dev_num;

	if (get_user(dev_num, (__u16 __user *) arg))
		return -EFAULT;

	if (!dev_num || dev_num > (PAGE_SIZE * 2) / sizeof(*dr))
		return -EINVAL;

	dl = kzalloc(struct_size(dl, dev_req, dev_num), GFP_KERNEL);
	if (!dl)
		return -ENOMEM;

	dl->dev_num = dev_num;
	dr = dl->dev_req;

	read_lock(&hci_dev_list_lock);
	list_for_each_entry(hdev, &hci_dev_list, list) {
		unsigned long flags = hdev->flags;

		/* When the auto-off is configured it means the transport
		 * is running, but in that case still indicate that the
		 * device is actually down.
		 */
		if (hci_dev_test_flag(hdev, HCI_AUTO_OFF))
			flags &= ~BIT(HCI_UP);

		dr[n].dev_id  = hdev->id;
		dr[n].dev_opt = flags;

		if (++n >= dev_num)
			break;
	}
	read_unlock(&hci_dev_list_lock);

	dl->dev_num = n;
	err = copy_to_user(arg, dl, struct_size(dl, dev_req, n));
	kfree(dl);

	return err ? -EFAULT : 0;
}
/**
 * @brief Handles the ioctl to get information about a specific HCI device.
 * @param arg A pointer to a user-space struct hci_dev_info.
 * @return 0 on success, or a negative error code on failure.
 */
int hci_get_dev_info(void __user *arg)
{
	struct hci_dev *hdev;
	struct hci_dev_info di;
	unsigned long flags;
	int err = 0;

	if (copy_from_user(&di, arg, sizeof(di)))
		return -EFAULT;

	hdev = hci_dev_get(di.dev_id);
	if (!hdev)
		return -ENODEV;

	/* When the auto-off is configured it means the transport
	 * is running, but in that case still indicate that the
	 * device is actually down.
	 */
	if (hci_dev_test_flag(hdev, HCI_AUTO_OFF))
		flags = hdev->flags & ~BIT(HCI_UP);
	else
		flags = hdev->flags;

	strscpy(di.name, hdev->name, sizeof(di.name));
	di.bdaddr   = hdev->bdaddr;
	di.type     = (hdev->bus & 0x0f);
	di.flags    = flags;
	di.pkt_type = hdev->pkt_type;
	if (lmp_bredr_capable(hdev)) {
		di.acl_mtu  = hdev->acl_mtu;
		di.acl_pkts = hdev->acl_pkts;
		di.sco_mtu  = hdev->sco_mtu;
		di.sco_pkts = hdev->sco_pkts;
	} else {
		di.acl_mtu  = hdev->le_mtu;
		di.acl_pkts = hdev->le_pkts;
		di.sco_mtu  = 0;
		di.sco_pkts = 0;
	}
	di.link_policy = hdev->link_policy;
	di.link_mode   = hdev->link_mode;

	memcpy(&di.stat, &hdev->stat, sizeof(di.stat));
	memcpy(&di.features, &hdev->features, sizeof(di.features));

	if (copy_to_user(arg, &di, sizeof(di)))
		err = -EFAULT;

	hci_dev_put(hdev);

	return err;
}

/* ---- Interface to HCI drivers ---- */

/**
 * @brief Powers off an HCI device.
 * @param hdev The HCI device to power off.
 * @return 0 on success, or a negative error code on failure.
 */
static int hci_dev_do_poweroff(struct hci_dev *hdev)
{
	int err;

	BT_DBG("%s %p", hdev->name, hdev);

	hci_req_sync_lock(hdev);

	err = hci_set_powered_sync(hdev, false);

	hci_req_sync_unlock(hdev);

	return err;
}
/**
 * @brief RFKILL set block callback.
 * @param data A pointer to the HCI device.
 * @param blocked True if the device is to be blocked, false otherwise.
 * @return 0 on success, or a negative error code on failure.
 */
static int hci_rfkill_set_block(void *data, bool blocked)
{
	struct hci_dev *hdev = data;
	int err;

	BT_DBG("%p name %s blocked %d", hdev, hdev->name, blocked);

	if (hci_dev_test_flag(hdev, HCI_USER_CHANNEL))
		return -EBUSY;

	if (blocked == hci_dev_test_flag(hdev, HCI_RFKILLED))
		return 0;

	if (blocked) {
		hci_dev_set_flag(hdev, HCI_RFKILLED);

		if (!hci_dev_test_flag(hdev, HCI_SETUP) &&
		    !hci_dev_test_flag(hdev, HCI_CONFIG)) {
			err = hci_dev_do_poweroff(hdev);
			if (err) {
				bt_dev_err(hdev, "Error when powering off device on rfkill (%d)",
					   err);

				/* Make sure the device is still closed even if
				 * anything during power off sequence (eg.
				 * disconnecting devices) failed.
				 */
				hci_dev_do_close(hdev);
			}
		}
	} else {
		hci_dev_clear_flag(hdev, HCI_RFKILLED);
	}

	return 0;
}

static const struct rfkill_ops hci_rfkill_ops = {
	.set_block = hci_rfkill_set_block,
};
/**
 * @brief Work function to power on an HCI device.
 * @param work The work struct.
 */
static void hci_power_on(struct work_struct *work)
{
	struct hci_dev *hdev = container_of(work, struct hci_dev, power_on);
	int err;

	BT_DBG("%s", hdev->name);

	if (test_bit(HCI_UP, &hdev->flags) &&
	    hci_dev_test_flag(hdev, HCI_MGMT) &&
	    hci_dev_test_and_clear_flag(hdev, HCI_AUTO_OFF)) {
		cancel_delayed_work(&hdev->power_off);
		err = hci_powered_update_sync(hdev);
		mgmt_power_on(hdev, err);
		return;
	}

	err = hci_dev_do_open(hdev);
	if (err < 0) {
		hci_dev_lock(hdev);
		mgmt_set_powered_failed(hdev, err);
		hci_dev_unlock(hdev);
		return;
	}

	/* During the HCI setup phase, a few error conditions are
	 * ignored and they need to be checked now. If they are still
	 * valid, it is important to turn the device back off.
	 */
	if (hci_dev_test_flag(hdev, HCI_RFKILLED) ||
	    hci_dev_test_flag(hdev, HCI_UNCONFIGURED) ||
	    (!bacmp(&hdev->bdaddr, BDADDR_ANY) &&
	     !bacmp(&hdev->static_addr, BDADDR_ANY))) {
		hci_dev_clear_flag(hdev, HCI_AUTO_OFF);
		hci_dev_do_close(hdev);
	} else if (hci_dev_test_flag(hdev, HCI_AUTO_OFF)) {
		queue_delayed_work(hdev->req_workqueue, &hdev->power_off,
				   HCI_AUTO_OFF_TIMEOUT);
	}

	if (hci_dev_test_and_clear_flag(hdev, HCI_SETUP)) {
		/* For unconfigured devices, set the HCI_RAW flag
		 * so that userspace can easily identify them.
		 */
		if (hci_dev_test_flag(hdev, HCI_UNCONFIGURED))
			set_bit(HCI_RAW, &hdev->flags);

		/* For fully configured devices, this will send
		 * the Index Added event. For unconfigured devices,
		 * it will send Unconfigued Index Added event.
		 *
		 * Devices with HCI_QUIRK_RAW_DEVICE are ignored
		 * and no event will be send.
		 */
		mgmt_index_added(hdev);
	} else if (hci_dev_test_and_clear_flag(hdev, HCI_CONFIG)) {
		/* When the controller is now configured, then it
		 * is important to clear the HCI_RAW flag.
		 */
		if (!hci_dev_test_flag(hdev, HCI_UNCONFIGURED))
			clear_bit(HCI_RAW, &hdev->flags);

		/* Powering on the controller with HCI_CONFIG set only
		 * happens with the transition from unconfigured to
		 * configured. This will send the Index Added event.
		 */
		mgmt_index_added(hdev);
	}
}
/**
 * @brief Work function to power off an HCI device.
 * @param work The work struct.
 */
static void hci_power_off(struct work_struct *work)
{
	struct hci_dev *hdev = container_of(work, struct hci_dev,
					    power_off.work);

	BT_DBG("%s", hdev->name);

	hci_dev_do_close(hdev);
}
/**
 * @brief Work function to reset an HCI device after an error.
 * @param work The work struct.
 */
static void hci_error_reset(struct work_struct *work)
{
	struct hci_dev *hdev = container_of(work, struct hci_dev, error_reset);

	hci_dev_hold(hdev);
	BT_DBG("%s", hdev->name);

	if (hdev->hw_error)
		hdev->hw_error(hdev, hdev->hw_error_code);
	else
		bt_dev_err(hdev, "hardware error 0x%2.2x", hdev->hw_error_code);

	if (!hci_dev_do_close(hdev))
		hci_dev_do_open(hdev);

	hci_dev_put(hdev);
}
/**
 * @brief Clears all UUIDs from an HCI device.
 * @param hdev The HCI device.
 */
void hci_uuids_clear(struct hci_dev *hdev)
{
	struct bt_uuid *uuid, *tmp;

	list_for_each_entry_safe(uuid, tmp, &hdev->uuids, list) {
		list_del(&uuid->list);
		kfree(uuid);
	}
}
/**
 * @brief Clears all link keys from an HCI device.
 * @param hdev The HCI device.
 */
void hci_link_keys_clear(struct hci_dev *hdev)
{
	struct link_key *key, *tmp;

	list_for_each_entry_safe(key, tmp, &hdev->link_keys, list) {
		list_del_rcu(&key->list);
		kfree_rcu(key, rcu);
	}
}
/**
 * @brief Clears all Long Term Keys (LTKs) from an HCI device.
 * @param hdev The HCI device.
 */
void hci_smp_ltks_clear(struct hci_dev *hdev)
{
	struct smp_ltk *k, *tmp;

	list_for_each_entry_safe(k, tmp, &hdev->long_term_keys, list) {
		list_del_rcu(&k->list);
		kfree_rcu(k, rcu);
	}
}
/**
 * @brief Clears all Identity Resolving Keys (IRKs) from an HCI device.
 * @param hdev The HCI device.
 */
void hci_smp_irks_clear(struct hci_dev *hdev)
{
	struct smp_irk *k, *tmp;

	list_for_each_entry_safe(k, tmp, &hdev->identity_resolving_keys, list) {
		list_del_rcu(&k->list);
		kfree_rcu(k, rcu);
	}
}
/**
 * @brief Clears all blocked keys from an HCI device.
 * @param hdev The HCI device.
 */
void hci_blocked_keys_clear(struct hci_dev *hdev)
{
	struct blocked_key *b, *tmp;

	list_for_each_entry_safe(b, tmp, &hdev->blocked_keys, list) {
		list_del_rcu(&b->list);
		kfree_rcu(b, rcu);
	}
}
/**
 * @brief Checks if a given key is blocked.
 * @param hdev The HCI device.
 * @param type The type of the key.
 * @param val The value of the key.
 * @return True if the key is blocked, false otherwise.
 */
bool hci_is_blocked_key(struct hci_dev *hdev, u8 type, u8 val[16])
{
	bool blocked = false;
	struct blocked_key *b;

	rcu_read_lock();
	list_for_each_entry_rcu(b, &hdev->blocked_keys, list) {
		if (b->type == type && !memcmp(b->val, val, sizeof(b->val))) {
			blocked = true;
			break;
		}
	}

	rcu_read_unlock();
	return blocked;
}
/**
 * @brief Finds a link key for a given Bluetooth address.
 * @param hdev The HCI device.
 * @param bdaddr The Bluetooth address.
 * @return A pointer to the link key, or NULL if not found.
 */
struct link_key *hci_find_link_key(struct hci_dev *hdev, bdaddr_t *bdaddr)
{
	struct link_key *k;

	rcu_read_lock();
	list_for_each_entry_rcu(k, &hdev->link_keys, list) {
		if (bacmp(bdaddr, &k->bdaddr) == 0) {
			rcu_read_unlock();

			if (hci_is_blocked_key(hdev,
					       HCI_BLOCKED_KEY_TYPE_LINKKEY,
					       k->val)) {
				bt_dev_warn_ratelimited(hdev,
							"Link key blocked for %pMR",
							&k->bdaddr);
				return NULL;
			}

			return k;
		}
	}
	rcu_read_unlock();

	return NULL;
}
/**
 * @brief Determines if a link key should be stored persistently.
 * @param hdev The HCI device.
 * @param conn The HCI connection.
 * @param key_type The type of the new key.
 * @param old_key_type The type of the old key.
 * @return True if the key should be stored persistently, false otherwise.
 */
static bool hci_persistent_key(struct hci_dev *hdev, struct hci_conn *conn,
			       u8 key_type, u8 old_key_type)
{
	/* Legacy key */
	if (key_type < 0x03)
		return true;

	/* Debug keys are insecure so don't store them persistently */
	if (key_type == HCI_LK_DEBUG_COMBINATION)
		return false;

	/* Changed combination key and there's no previous one */
	if (key_type == HCI_LK_CHANGED_COMBINATION && old_key_type == 0xff)
		return false;

	/* Security mode 3 case */
	if (!conn)
		return true;

	/* BR/EDR key derived using SC from an LE link */
	if (conn->type == LE_LINK)
		return true;

	/* Neither local nor remote side had no-bonding as requirement */
	if (conn->auth_type > 0x01 && conn->remote_auth > 0x01)
		return true;

	/* Local side had dedicated bonding as requirement */
	if (conn->auth_type == 0x02 || conn->auth_type == 0x03)
		return true;

	/* Remote side had dedicated bonding as requirement */
	if (conn->remote_auth == 0x02 || conn->remote_auth == 0x03)
		return true;

	/* If none of the above criteria match, then don't store the key
	 * persistently */
	return false;
}
/**
 * @brief Determines the role (master/slave) based on the LTK type.
 * @param type The LTK type.
 * @return HCI_ROLE_MASTER or HCI_ROLE_SLAVE.
 */
static u8 ltk_role(u8 type)
{
	if (type == SMP_LTK)
		return HCI_ROLE_MASTER;

	return HCI_ROLE_SLAVE;
}

/**
 * @brief Finds a Long Term Key (LTK) for a given Bluetooth address and role.
 * @param hdev The HCI device.
 * @param bdaddr The Bluetooth address.
 * @param addr_type The address type.
 * @param role The role (master or slave).
 * @return A pointer to the LTK, or NULL if not found.
 */
struct smp_ltk *hci_find_ltk(struct hci_dev *hdev, bdaddr_t *bdaddr,
			     u8 addr_type, u8 role)
{
	struct smp_ltk *k;

	rcu_read_lock();
	list_for_each_entry_rcu(k, &hdev->long_term_keys, list) {
		if (addr_type != k->bdaddr_type || bacmp(bdaddr, &k->bdaddr))
			continue;

		if (smp_ltk_is_sc(k) || ltk_role(k->type) == role) {
			rcu_read_unlock();

			if (hci_is_blocked_key(hdev, HCI_BLOCKED_KEY_TYPE_LTK,
					       k->val)) {
				bt_dev_warn_ratelimited(hdev,
							"LTK blocked for %pMR",
							&k->bdaddr);
				return NULL;
			}

			return k;
		}
	}
	rcu_read_unlock();

	return NULL;
}
/**
 * @brief Finds an Identity Resolving Key (IRK) by its Resolvable Private Address (RPA).
 * @param hdev The HCI device.
 * @param rpa The RPA to look up.
 * @return A pointer to the IRK, or NULL if not found.
 */
struct smp_irk *hci_find_irk_by_rpa(struct hci_dev *hdev, bdaddr_t *rpa)
{
	struct smp_irk *irk_to_return = NULL;
	struct smp_irk *irk;

	rcu_read_lock();
	list_for_each_entry_rcu(irk, &hdev->identity_resolving_keys, list) {
		if (!bacmp(&irk->rpa, rpa)) {
			irk_to_return = irk;
			goto done;
		}
	}

	list_for_each_entry_rcu(irk, &hdev->identity_resolving_keys, list) {
		if (smp_irk_matches(hdev, irk->val, rpa)) {
			bacpy(&irk->rpa, rpa);
			irk_to_return = irk;
			goto done;
		}
	}

done:
	if (irk_to_return && hci_is_blocked_key(hdev, HCI_BLOCKED_KEY_TYPE_IRK,
						irk_to_return->val)) {
		bt_dev_warn_ratelimited(hdev, "Identity key blocked for %pMR",
					&irk_to_return->bdaddr);
		irk_to_return = NULL;
	}

	rcu_read_unlock();

	return irk_to_return;
}
/**
 * @brief Finds an Identity Resolving Key (IRK) by its identity address.
 * @param hdev The HCI device.
 * @param bdaddr The identity address.
 * @param addr_type The address type.
 * @return A pointer to the IRK, or NULL if not found.
 */
struct smp_irk *hci_find_irk_by_addr(struct hci_dev *hdev, bdaddr_t *bdaddr,
				     u8 addr_type)
{
	struct smp_irk *irk_to_return = NULL;
	struct smp_irk *irk;

	/* Identity Address must be public or static random */
	if (addr_type == ADDR_LE_DEV_RANDOM && (bdaddr->b[5] & 0xc0) != 0xc0)
		return NULL;

	rcu_read_lock();
	list_for_each_entry_rcu(irk, &hdev->identity_resolving_keys, list) {
		if (addr_type == irk->addr_type &&
		    bacmp(bdaddr, &irk->bdaddr) == 0) {
			irk_to_return = irk;
			goto done;
		}
	}

done:

	if (irk_to_return && hci_is_blocked_key(hdev, HCI_BLOCKED_KEY_TYPE_IRK,
						irk_to_return->val)) {
		bt_dev_warn_ratelimited(hdev, "Identity key blocked for %pMR",
					&irk_to_return->bdaddr);
		irk_to_return = NULL;
	}

	rcu_read_unlock();

	return irk_to_return;
}
/**
 * @brief Adds or updates a link key for a device.
 * @param hdev The HCI device.
 * @param conn The HCI connection (can be NULL).
 * @param bdaddr The Bluetooth address.
 * @param val The link key value.
 * @param type The link key type.
 * @param pin_len The PIN length.
 * @param persistent Pointer to a boolean that will be set to indicate if the key should be stored persistently.
 * @return A pointer to the new or updated link key, or NULL on failure.
 */
struct link_key *hci_add_link_key(struct hci_dev *hdev, struct hci_conn *conn,
				  bdaddr_t *bdaddr, u8 *val, u8 type,
				  u8 pin_len, bool *persistent)
{
	struct link_key *key, *old_key;
	u8 old_key_type;

	old_key = hci_find_link_key(hdev, bdaddr);
	if (old_key) {
		old_key_type = old_key->type;
		key = old_key;
	} else {
		old_key_type = conn ? conn->key_type : 0xff;
		key = kzalloc(sizeof(*key), GFP_KERNEL);
		if (!key)
			return NULL;
		list_add_rcu(&key->list, &hdev->link_keys);
	}

	BT_DBG("%s key for %pMR type %u", hdev->name, bdaddr, type);

	/* Some buggy controller combinations generate a changed
	 * combination key for legacy pairing even when there's no
	 * previous key */
	if (type == HCI_LK_CHANGED_COMBINATION &&
	    (!conn || conn->remote_auth == 0xff) && old_key_type == 0xff) {
		type = HCI_LK_COMBINATION;
		if (conn)
			conn->key_type = type;
	}

	bacpy(&key->bdaddr, bdaddr);
	memcpy(key->val, val, HCI_LINK_KEY_SIZE);
	key->pin_len = pin_len;

	if (type == HCI_LK_CHANGED_COMBINATION)
		key->type = old_key_type;
	else
		key->type = type;

	if (persistent)
		*persistent = hci_persistent_key(hdev, conn, type,
						 old_key_type);

	return key;
}
/**
 * @brief Adds or updates a Long Term Key (LTK) for a device.
 * @param hdev The HCI device.
 * @param bdaddr The Bluetooth address.
 * @param addr_type The address type.
 * @param type The LTK type.
 * @param authenticated The authentication level of the key.
 * @param tk The LTK value.
 * @param enc_size The encryption key size.
 * @param ediv The encrypted diversifier.
 * @param rand The random number.
 * @return A pointer to the new or updated LTK, or NULL on failure.
 */
struct smp_ltk *hci_add_ltk(struct hci_dev *hdev, bdaddr_t *bdaddr,
			    u8 addr_type, u8 type, u8 authenticated,
			    u8 tk[16], u8 enc_size, __le16 ediv, __le64 rand)
{
	struct smp_ltk *key, *old_key;
	u8 role = ltk_role(type);

	old_key = hci_find_ltk(hdev, bdaddr, addr_type, role);
	if (old_key)
		key = old_key;
	else {
		key = kzalloc(sizeof(*key), GFP_KERNEL);
		if (!key)
			return NULL;
		list_add_rcu(&key->list, &hdev->long_term_keys);
	}

	bacpy(&key->bdaddr, bdaddr);
	key->bdaddr_type = addr_type;
	memcpy(key->val, tk, sizeof(key->val));
	key->authenticated = authenticated;
	key->ediv = ediv;
	key->rand = rand;
	key->enc_size = enc_size;
	key->type = type;

	return key;
}
/**
 * @brief Adds or updates an Identity Resolving Key (IRK) for a device.
 * @param hdev The HCI device.
 * @param bdaddr The identity address.
 * @param addr_type The address type.
 * @param val The IRK value.
 * @param rpa The Resolvable Private Address.
 * @return A pointer to the new or updated IRK, or NULL on failure.
 */
struct smp_irk *hci_add_irk(struct hci_dev *hdev, bdaddr_t *bdaddr,
			    u8 addr_type, u8 val[16], bdaddr_t *rpa)
{
	struct smp_irk *irk;

	irk = hci_find_irk_by_addr(hdev, bdaddr, addr_type);
	if (!irk) {
		irk = kzalloc(sizeof(*irk), GFP_KERNEL);
		if (!irk)
			return NULL;

		bacpy(&irk->bdaddr, bdaddr);
		irk->addr_type = addr_type;

		list_add_rcu(&irk->list, &hdev->identity_resolving_keys);
	}

	memcpy(irk->val, val, 16);
	bacpy(&irk->rpa, rpa);

	return irk;
}
/**
 * @brief Removes a link key for a given Bluetooth address.
 * @param hdev The HCI device.
 * @param bdaddr The Bluetooth address.
 * @return 0 on success, or -ENOENT if the key was not found.
 */
int hci_remove_link_key(struct hci_dev *hdev, bdaddr_t *bdaddr)
{
	struct link_key *key;

	key = hci_find_link_key(hdev, bdaddr);
	if (!key)
		return -ENOENT;

	BT_DBG("%s removing %pMR", hdev->name, bdaddr);

	list_del_rcu(&key->list);
	kfree_rcu(key, rcu);

	return 0;
}
/**
 * @brief Removes all Long Term Keys (LTKs) for a given Bluetooth address.
 * @param hdev The HCI device.
 * @param bdaddr The Bluetooth address.
 * @param bdaddr_type The address type.
 * @return 0 on success, or -ENOENT if no keys were found.
 */
int hci_remove_ltk(struct hci_dev *hdev, bdaddr_t *bdaddr, u8 bdaddr_type)
{
	struct smp_ltk *k, *tmp;
	int removed = 0;

	list_for_each_entry_safe(k, tmp, &hdev->long_term_keys, list) {
		if (bacmp(bdaddr, &k->bdaddr) || k->bdaddr_type != bdaddr_type)
			continue;

		BT_DBG("%s removing %pMR", hdev->name, bdaddr);

		list_del_rcu(&k->list);
		kfree_rcu(k, rcu);
		removed++;
	}

	return removed ? 0 : -ENOENT;
}
/**
 * @brief Removes the Identity Resolving Key (IRK) for a given Bluetooth address.
 * @param hdev The HCI device.
 * @param bdaddr The identity address.
 * @param addr_type The address type.
 */
void hci_remove_irk(struct hci_dev *hdev, bdaddr_t *bdaddr, u8 addr_type)
{
	struct smp_irk *k, *tmp;

	list_for_each_entry_safe(k, tmp, &hdev->identity_resolving_keys, list) {
		if (bacmp(bdaddr, &k->bdaddr) || k->addr_type != addr_type)
			continue;

		BT_DBG("%s removing %pMR", hdev->name, bdaddr);

		list_del_rcu(&k->list);
		kfree_rcu(k, rcu);
	}
}
/**
 * @brief Checks if a device is paired.
 * @param hdev The HCI device.
 * @param bdaddr The Bluetooth address.
 * @param type The address type.
 * @return True if the device is paired, false otherwise.
 */
bool hci_bdaddr_is_paired(struct hci_dev *hdev, bdaddr_t *bdaddr, u8 type)
{
	struct smp_ltk *k;
	struct smp_irk *irk;
	u8 addr_type;

	if (type == BDADDR_BREDR) {
		if (hci_find_link_key(hdev, bdaddr))
			return true;
		return false;
	}

	/* Convert to HCI addr type which struct smp_ltk uses */
	if (type == BDADDR_LE_PUBLIC)
		addr_type = ADDR_LE_DEV_PUBLIC;
	else
		addr_type = ADDR_LE_DEV_RANDOM;

	irk = hci_get_irk(hdev, bdaddr, addr_type);
	if (irk) {
		bdaddr = &irk->bdaddr;
		addr_type = irk->addr_type;
	}

	rcu_read_lock();
	list_for_each_entry_rcu(k, &hdev->long_term_keys, list) {
		if (k->bdaddr_type == addr_type && !bacmp(bdaddr, &k->bdaddr)) {
			rcu_read_unlock();
			return true;
		}
	}
	rcu_read_unlock();

	return false;
}

/**
 * @brief HCI command timeout handler.
 * @param work The work struct.
 *
 * This function is called when an HCI command times out. It logs an error,
 * cancels the command, and may trigger a device reset.
 */
static void hci_cmd_timeout(struct work_struct *work)
{
	struct hci_dev *hdev = container_of(work, struct hci_dev,
					    cmd_timer.work);

	if (hdev->req_skb) {
		u16 opcode = hci_skb_opcode(hdev->req_skb);

		bt_dev_err(hdev, "command 0x%4.4x tx timeout", opcode);

		hci_cmd_sync_cancel_sync(hdev, ETIMEDOUT);
	} else {
		bt_dev_err(hdev, "command tx timeout");
	}

	if (hdev->reset)
		hdev->reset(hdev);

	atomic_set(&hdev->cmd_cnt, 1);
	queue_work(hdev->workqueue, &hdev->cmd_work);
}
/**
 * @brief HCI ncmd timeout handler.
 * @param work The work struct.
 *
 * This function is called when the controller stops accepting new commands.
 * It logs an error and may trigger a device reset.
 */
static void hci_ncmd_timeout(struct work_struct *work)
{
	struct hci_dev *hdev = container_of(work, struct hci_dev,
					    ncmd_timer.work);

	bt_dev_err(hdev, "Controller not accepting commands anymore: ncmd = 0");

	/* During HCI_INIT phase no events can be injected if the ncmd timer
	 * triggers since the procedure has its own timeout handling.
	 */
	if (test_bit(HCI_INIT, &hdev->flags))
		return;

	/* This is an irrecoverable state, inject hardware error event */
	hci_reset_dev(hdev);
}
/**
 * @brief Finds remote Out-Of-Band (OOB) data for a device.
 * @param hdev The HCI device.
 * @param bdaddr The Bluetooth address.
 * @param bdaddr_type The address type.
 * @return A pointer to the OOB data, or NULL if not found.
 */
struct oob_data *hci_find_remote_oob_data(struct hci_dev *hdev,
					  bdaddr_t *bdaddr, u8 bdaddr_type)
{
	struct oob_data *data;

	list_for_each_entry(data, &hdev->remote_oob_data, list) {
		if (bacmp(bdaddr, &data->bdaddr) != 0)
			continue;
		if (data->bdaddr_type != bdaddr_type)
			continue;
		return data;
	}

	return NULL;
}
/**
 * @brief Removes remote Out-Of-Band (OOB) data for a device.
 * @param hdev The HCI device.
 * @param bdaddr The Bluetooth address.
 * @param bdaddr_type The address type.
 * @return 0 on success, or -ENOENT if not found.
 */
int hci_remove_remote_oob_data(struct hci_dev *hdev, bdaddr_t *bdaddr,
			       u8 bdaddr_type)
{
	struct oob_data *data;

	data = hci_find_remote_oob_data(hdev, bdaddr, bdaddr_type);
	if (!data)
		return -ENOENT;

	BT_DBG("%s removing %pMR (%u)", hdev->name, bdaddr, bdaddr_type);

	list_del(&data->list);
	kfree(data);

	return 0;
}
/**
 * @brief Clears all remote Out-Of-Band (OOB) data from an HCI device.
 * @param hdev The HCI device.
 */
void hci_remote_oob_data_clear(struct hci_dev *hdev)
{
	struct oob_data *data, *n;

	list_for_each_entry_safe(data, n, &hdev->remote_oob_data, list) {
		list_del(&data->list);
		kfree(data);
	}
}
/**
 * @brief Adds or updates remote Out-Of-Band (OOB) data for a device.
 * @param hdev The HCI device.
 * @param bdaddr The Bluetooth address.
 * @param bdaddr_type The address type.
 * @param hash192 The P-192 hash.
 * @param rand192 The P-192 randomizer.
 * @param hash256 The P-256 hash.
 * @param rand256 The P-256 randomizer.
 * @return 0 on success, or -ENOMEM on failure.
 */
int hci_add_remote_oob_data(struct hci_dev *hdev, bdaddr_t *bdaddr,
			    u8 bdaddr_type, u8 *hash192, u8 *rand192,
			    u8 *hash256, u8 *rand256)
{
	struct oob_data *data;

	data = hci_find_remote_oob_data(hdev, bdaddr, bdaddr_type);
	if (!data) {
		data = kmalloc(sizeof(*data), GFP_KERNEL);
		if (!data)
			return -ENOMEM;

		bacpy(&data->bdaddr, bdaddr);
		data->bdaddr_type = bdaddr_type;
		list_add(&data->list, &hdev->remote_oob_data);
	}

	if (hash192 && rand192) {
		memcpy(data->hash192, hash192, sizeof(data->hash192));
		memcpy(data->rand192, rand192, sizeof(data->rand192));
		if (hash256 && rand256)
			data->present = 0x03;
	} else {
		memset(data->hash192, 0, sizeof(data->hash192));
		memset(data->rand192, 0, sizeof(data->rand192));
		if (hash256 && rand256)
			data->present = 0x02;
		else
			data->present = 0x00;
	}

	if (hash256 && rand256) {
		memcpy(data->hash256, hash256, sizeof(data->hash256));
		memcpy(data->rand256, rand256, sizeof(data->rand256));
	} else {
		memset(data->hash256, 0, sizeof(data->hash256));
		memset(data->rand256, 0, sizeof(data->rand256));
		if (hash192 && rand192)
			data->present = 0x01;
	}

	BT_DBG("%s for %pMR", hdev->name, bdaddr);

	return 0;
}
/**
 * @brief Finds an advertising instance by its ID.
 * @param hdev The HCI device.
 * @param instance The instance ID.
 * @return A pointer to the advertising instance, or NULL if not found.
 */
struct adv_info *hci_find_adv_instance(struct hci_dev *hdev, u8 instance)
{
	struct adv_info *adv_instance;

	list_for_each_entry(adv_instance, &hdev->adv_instances, list) {
		if (adv_instance->instance == instance)
			return adv_instance;
	}

	return NULL;
}
/**
 * @brief Finds an advertising instance by its advertising set ID (SID).
 * @param hdev The HCI device.
 * @param sid The advertising set ID.
 * @return A pointer to the advertising instance, or NULL if not found.
 */
struct adv_info *hci_find_adv_sid(struct hci_dev *hdev, u8 sid)
{
	struct adv_info *adv;

	list_for_each_entry(adv, &hdev->adv_instances, list) {
		if (adv->sid == sid)
			return adv;
	}

	return NULL;
}
/**
 * @brief Gets the next advertising instance in the rotation.
 * @param hdev The HCI device.
 * @param instance The current instance ID.
 * @return A pointer to the next advertising instance, or NULL if the current instance is not found.
 */
struct adv_info *hci_get_next_instance(struct hci_dev *hdev, u8 instance)
{
	struct adv_info *cur_instance;

	cur_instance = hci_find_adv_instance(hdev, instance);
	if (!cur_instance)
		return NULL;

	if (cur_instance == list_last_entry(&hdev->adv_instances,
					    struct adv_info, list))
		return list_first_entry(&hdev->adv_instances,
						 struct adv_info, list);
	else
		return list_next_entry(cur_instance, list);
}
/**
 * @brief Removes an advertising instance.
 * @param hdev The HCI device.
 * @param instance The instance ID to remove.
 * @return 0 on success, or -ENOENT if the instance is not found.
 */
int hci_remove_adv_instance(struct hci_dev *hdev, u8 instance)
{
	struct adv_info *adv_instance;

	adv_instance = hci_find_adv_instance(hdev, instance);
	if (!adv_instance)
		return -ENOENT;

	BT_DBG("%s removing %dMR", hdev->name, instance);

	if (hdev->cur_adv_instance == instance) {
		if (hdev->adv_instance_timeout) {
			cancel_delayed_work(&hdev->adv_instance_expire);
			hdev->adv_instance_timeout = 0;
		}
		hdev->cur_adv_instance = 0x00;
	}

	cancel_delayed_work_sync(&adv_instance->rpa_expired_cb);

	list_del(&adv_instance->list);
	kfree(adv_instance);

	hdev->adv_instance_cnt--;

	return 0;
}
/**
 * @brief Sets the RPA expired flag for all advertising instances.
 * @param hdev The HCI device.
 * @param rpa_expired The new value of the flag.
 */
void hci_adv_instances_set_rpa_expired(struct hci_dev *hdev, bool rpa_expired)
{
	struct adv_info *adv_instance, *n;

	list_for_each_entry_safe(adv_instance, n, &hdev->adv_instances, list)
		adv_instance->rpa_expired = rpa_expired;
}
/**
 * @brief Clears all advertising instances from an HCI device.
 * @param hdev The HCI device.
 */
void hci_adv_instances_clear(struct hci_dev *hdev)
{
	struct adv_info *adv_instance, *n;

	if (hdev->adv_instance_timeout) {
		disable_delayed_work(&hdev->adv_instance_expire);
		hdev->adv_instance_timeout = 0;
	}

	list_for_each_entry_safe(adv_instance, n, &hdev->adv_instances, list) {
		disable_delayed_work_sync(&adv_instance->rpa_expired_cb);
		list_del(&adv_instance->list);
		kfree(adv_instance);
	}

	hdev->adv_instance_cnt = 0;
	hdev->cur_adv_instance = 0x00;
}
/**
 * @brief Work function to handle RPA expiration for an advertising instance.
 * @param work The work struct.
 */
static void adv_instance_rpa_expired(struct work_struct *work)
{
	struct adv_info *adv_instance = container_of(work, struct adv_info,
						     rpa_expired_cb.work);

	BT_DBG("");

	adv_instance->rpa_expired = true;
}
/**
 * @brief Adds or updates an advertising instance.
 * @param hdev The HCI device.
 * @param instance The instance ID.
 * @param flags The advertising flags.
 * @param adv_data_len The length of the advertising data.
 * @param adv_data The advertising data.
 * @param scan_rsp_len The length of the scan response data.
 * @param scan_rsp_data The scan response data.
 * @param timeout The advertising timeout.
 * @param duration The advertising duration.
 * @param tx_power The advertising TX power.
 * @param min_interval The minimum advertising interval.
 * @param max_interval The maximum advertising interval.
 * @param mesh_handle The mesh handle.
 * @return A pointer to the new or updated advertising instance, or an error pointer on failure.
 */
struct adv_info *hci_add_adv_instance(struct hci_dev *hdev, u8 instance,
				      u32 flags, u16 adv_data_len, u8 *adv_data,
				      u16 scan_rsp_len, u8 *scan_rsp_data,
				      u16 timeout, u16 duration, s8 tx_power,
				      u32 min_interval, u32 max_interval,
				      u8 mesh_handle)
{
	struct adv_info *adv;

	adv = hci_find_adv_instance(hdev, instance);
	if (adv) {
		memset(adv->adv_data, 0, sizeof(adv->adv_data));
		memset(adv->scan_rsp_data, 0, sizeof(adv->scan_rsp_data));
		memset(adv->per_adv_data, 0, sizeof(adv->per_adv_data));
	} else {
		if (hdev->adv_instance_cnt >= hdev->le_num_of_adv_sets ||
		    instance < 1 || instance > hdev->le_num_of_adv_sets + 1)
			return ERR_PTR(-EOVERFLOW);

		adv = kzalloc(sizeof(*adv), GFP_KERNEL);
		if (!adv)
			return ERR_PTR(-ENOMEM);

		adv->pending = true;
		adv->instance = instance;

		/* If controller support only one set and the instance is set to
		 * 1 then there is no option other than using handle 0x00.
		 */
		if (hdev->le_num_of_adv_sets == 1 && instance == 1)
			adv->handle = 0x00;
		else
			adv->handle = instance;

		list_add(&adv->list, &hdev->adv_instances);
		hdev->adv_instance_cnt++;
	}

	adv->flags = flags;
	adv->min_interval = min_interval;
	adv->max_interval = max_interval;
	adv->tx_power = tx_power;
	/* Defining a mesh_handle changes the timing units to ms,
	 * rather than seconds, and ties the instance to the requested
	 * mesh_tx queue.
	 */
	adv->mesh = mesh_handle;

	hci_set_adv_instance_data(hdev, instance, adv_data_len, adv_data,
				  scan_rsp_len, scan_rsp_data);

	adv->timeout = timeout;
	adv->remaining_time = timeout;

	if (duration == 0)
		adv->duration = hdev->def_multi_adv_rotation_duration;
	else
		adv->duration = duration;

	INIT_DELAYED_WORK(&adv->rpa_expired_cb, adv_instance_rpa_expired);

	BT_DBG("%s for %dMR", hdev->name, instance);

	return adv;
}
/**
 * @brief Adds or updates a periodic advertising instance.
 * @param hdev The HCI device.
 * @param instance The instance ID.
 * @param sid The advertising set ID.
 * @param flags The advertising flags.
 * @param data_len The length of the periodic advertising data.
 * @param data The periodic advertising data.
 * @param min_interval The minimum advertising interval.
 * @param max_interval The maximum advertising interval.
 * @return A pointer to the new or updated periodic advertising instance, or an error pointer on failure.
 */
struct adv_info *hci_add_per_instance(struct hci_dev *hdev, u8 instance, u8 sid,
				      u32 flags, u8 data_len, u8 *data,
				      u32 min_interval, u32 max_interval)
{
	struct adv_info *adv;

	adv = hci_add_adv_instance(hdev, instance, flags, 0, NULL, 0, NULL,
				   0, 0, HCI_ADV_TX_POWER_NO_PREFERENCE,
				   min_interval, max_interval, 0);
	if (IS_ERR(adv))
		return adv;

	adv->sid = sid;
	adv->periodic = true;
	adv->per_adv_data_len = data_len;

	if (data)
		memcpy(adv->per_adv_data, data, data_len);

	return adv;
}
/**
 * @brief Sets the data for an advertising instance.
 * @param hdev The HCI device.
 * @param instance The instance ID.
 * @param adv_data_len The length of the advertising data.
 * @param adv_data The advertising data.
 * @param scan_rsp_len The length of the scan response data.
 * @param scan_rsp_data The scan response data.
 * @return 0 on success, or -ENOENT if the instance is not found.
 */
int hci_set_adv_instance_data(struct hci_dev *hdev, u8 instance,
			      u16 adv_data_len, u8 *adv_data,
			      u16 scan_rsp_len, u8 *scan_rsp_data)
{
	struct adv_info *adv;

	adv = hci_find_adv_instance(hdev, instance);

	/* If advertisement doesn't exist, we can't modify its data */
	if (!adv)
		return -ENOENT;

	if (adv_data_len && ADV_DATA_CMP(adv, adv_data, adv_data_len)) {
		memset(adv->adv_data, 0, sizeof(adv->adv_data));
		memcpy(adv->adv_data, adv_data, adv_data_len);
		adv->adv_data_len = adv_data_len;
		adv->adv_data_changed = true;
	}

	if (scan_rsp_len && SCAN_RSP_CMP(adv, scan_rsp_data, scan_rsp_len)) {
		memset(adv->scan_rsp_data, 0, sizeof(adv->scan_rsp_data));
		memcpy(adv->scan_rsp_data, scan_rsp_data, scan_rsp_len);
		adv->scan_rsp_len = scan_rsp_len;
		adv->scan_rsp_changed = true;
	}

	/* Mark as changed if there are flags which would affect it */
	if (((adv->flags & MGMT_ADV_FLAG_APPEARANCE) && hdev->appearance) ||
	    adv->flags & MGMT_ADV_FLAG_LOCAL_NAME)
		adv->scan_rsp_changed = true;

	return 0;
}
/**
 * @brief Gets the advertising flags for a given instance.
 * @param hdev The HCI device.
 * @param instance The instance ID.
 * @return The advertising flags.
 */
u32 hci_adv_instance_flags(struct hci_dev *hdev, u8 instance)
{
	u32 flags;
	struct adv_info *adv;

	if (instance == 0x00) {
		/* Instance 0 always manages the "Tx Power" and "Flags"
		 * fields
		 */
		flags = MGMT_ADV_FLAG_TX_POWER | MGMT_ADV_FLAG_MANAGED_FLAGS;

		/* For instance 0, the HCI_ADVERTISING_CONNECTABLE setting
		 * corresponds to the "connectable" instance flag.
		 */
		if (hci_dev_test_flag(hdev, HCI_ADVERTISING_CONNECTABLE))
			flags |= MGMT_ADV_FLAG_CONNECTABLE;

		if (hci_dev_test_flag(hdev, HCI_LIMITED_DISCOVERABLE))
			flags |= MGMT_ADV_FLAG_LIMITED_DISCOV;
		else if (hci_dev_test_flag(hdev, HCI_DISCOVERABLE))
			flags |= MGMT_ADV_FLAG_DISCOV;

		return flags;
	}

	adv = hci_find_adv_instance(hdev, instance);

	/* Return 0 when we got an invalid instance identifier. */
	if (!adv)
		return 0;

	return adv->flags;
}
/**
 * @brief Checks if an advertising instance is scannable.
 * @param hdev The HCI device.
 * @param instance The instance ID.
 * @return True if the instance is scannable, false otherwise.
 */
bool hci_adv_instance_is_scannable(struct hci_dev *hdev, u8 instance)
{
	struct adv_info *adv;

	/* Instance 0x00 always set local name */
	if (instance == 0x00)
		return true;

	adv = hci_find_adv_instance(hdev, instance);
	if (!adv)
		return false;

	if (adv->flags & MGMT_ADV_FLAG_APPEARANCE ||
	    adv->flags & MGMT_ADV_FLAG_LOCAL_NAME)
		return true;

	return adv->scan_rsp_len ? true : false;
}
/**
 * @brief Clears all advertising monitors from an HCI device.
 * @param hdev The HCI device.
 */
void hci_adv_monitors_clear(struct hci_dev *hdev)
{
	struct adv_monitor *monitor;
	int handle;

	idr_for_each_entry(&hdev->adv_monitors_idr, monitor, handle)
		hci_free_adv_monitor(hdev, monitor);

	idr_destroy(&hdev->adv_monitors_idr);
}
/**
 * @brief Frees an advertising monitor.
 * @param hdev The HCI device.
 * @param monitor The advertising monitor to free.
 */
void hci_free_adv_monitor(struct hci_dev *hdev, struct adv_monitor *monitor)
{
	struct adv_pattern *pattern;
	struct adv_pattern *tmp;

	if (!monitor)
		return;

	list_for_each_entry_safe(pattern, tmp, &monitor->patterns, list) {
		list_del(&pattern->list);
		kfree(pattern);
	}

	if (monitor->handle)
		idr_remove(&hdev->adv_monitors_idr, monitor->handle);

	if (monitor->state != ADV_MONITOR_STATE_NOT_REGISTERED)
		hdev->adv_monitors_cnt--;

	kfree(monitor);
}
/**
 * @brief Adds an advertising monitor to an HCI device.
 * @param hdev The HCI device.
 * @param monitor The advertising monitor to add.
 * @return 0 on success, or a negative error code on failure.
 */
int hci_add_adv_monitor(struct hci_dev *hdev, struct adv_monitor *monitor)
{
	int min, max, handle;
	int status = 0;

	if (!monitor)
		return -EINVAL;

	hci_dev_lock(hdev);

	min = HCI_MIN_ADV_MONITOR_HANDLE;
	max = HCI_MIN_ADV_MONITOR_HANDLE + HCI_MAX_ADV_MONITOR_NUM_HANDLES;
	handle = idr_alloc(&hdev->adv_monitors_idr, monitor, min, max,
			   GFP_KERNEL);

	hci_dev_unlock(hdev);

	if (handle < 0)
		return handle;

	monitor->handle = handle;

	if (!hdev_is_powered(hdev))
		return status;

	switch (hci_get_adv_monitor_offload_ext(hdev)) {
	case HCI_ADV_MONITOR_EXT_NONE:
		bt_dev_dbg(hdev, "add monitor %d status %d",
			   monitor->handle, status);
		/* Message was not forwarded to controller - not an error */
		break;

	case HCI_ADV_MONITOR_EXT_MSFT:
		status = msft_add_monitor_pattern(hdev, monitor);
		bt_dev_dbg(hdev, "add monitor %d msft status %d",
			   handle, status);
		break;
	}

	return status;
}

/**
 * @brief Removes an advertising monitor from an HCI device.
 * @param hdev The HCI device.
 * @param monitor The advertising monitor to remove.
 * @return 0 on success, or a negative error code on failure.
 */
static int hci_remove_adv_monitor(struct hci_dev *hdev,
				  struct adv_monitor *monitor)
{
	int status = 0;
	int handle;

	switch (hci_get_adv_monitor_offload_ext(hdev)) {
	case HCI_ADV_MONITOR_EXT_NONE: /* also goes here when powered off */
		bt_dev_dbg(hdev, "remove monitor %d status %d",
			   monitor->handle, status);
		goto free_monitor;

	case HCI_ADV_MONITOR_EXT_MSFT:
		handle = monitor->handle;
		status = msft_remove_monitor(hdev, monitor);
		bt_dev_dbg(hdev, "remove monitor %d msft status %d",
			   handle, status);
		break;
	}

	/* In case no matching handle registered, just free the monitor */
	if (status == -ENOENT)
		goto free_monitor;

	return status;

free_monitor:
	if (status == -ENOENT)
		bt_dev_warn(hdev, "Removing monitor with no matching handle %d",
			    monitor->handle);
	hci_free_adv_monitor(hdev, monitor);

	return status;
}