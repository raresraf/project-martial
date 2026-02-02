/*
   RFCOMM implementation for Linux Bluetooth stack (BlueZ).
   Copyright (C) 2002 Maxim Krasnyansky <maxk@qualcomm.com>
   Copyright (C) 2002 Marcel Holtmann <marcel@holtmann.org>

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
 * @brief Core implementation of the Bluetooth RFCOMM protocol.
 *
 * This file provides the core functionality for the RFCOMM protocol, which
 * emulates serial ports over the L2CAP protocol. It manages sessions, Data
 * Link Connections (DLCs), and the flow of data between the host and remote
 * devices. This includes handling connection setup, teardown, flow control,
 * and error handling. It also provides an interface for upper layers to
 * interact with RFCOMM DLCs as if they were serial ports.
 */

#include <linux/module.h>
#include <linux/debugfs.h>
#include <linux/kthread.h>
#include <linux/unaligned.h>

#include <net/bluetooth/bluetooth.h>
#include <net/bluetooth/hci_core.h>
#include <net/bluetooth/l2cap.h>
#include <net/bluetooth/rfcomm.h>

#include <trace/events/sock.h>

#define VERSION "1.11"

static bool disable_cfc;
static bool l2cap_ertm;
static int channel_mtu = -1;

static struct task_struct *rfcomm_thread;

static DEFINE_MUTEX(rfcomm_mutex);
#define rfcomm_lock()	mutex_lock(&rfcomm_mutex)
#define rfcomm_unlock()	mutex_unlock(&rfcomm_mutex)


static LIST_HEAD(session_list);

static int rfcomm_send_frame(struct rfcomm_session *s, u8 *data, int len);
static int rfcomm_send_sabm(struct rfcomm_session *s, u8 dlci);
static int rfcomm_send_disc(struct rfcomm_session *s, u8 dlci);
static int rfcomm_queue_disc(struct rfcomm_dlc *d);
static int rfcomm_send_nsc(struct rfcomm_session *s, int cr, u8 type);
static int rfcomm_send_pn(struct rfcomm_session *s, int cr, struct rfcomm_dlc *d);
static int rfcomm_send_msc(struct rfcomm_session *s, int cr, u8 dlci, u8 v24_sig);
static int rfcomm_send_test(struct rfcomm_session *s, int cr, u8 *pattern, int len);
static int rfcomm_send_credits(struct rfcomm_session *s, u8 addr, u8 credits);
static void rfcomm_make_uih(struct sk_buff *skb, u8 addr);

static void rfcomm_process_connect(struct rfcomm_session *s);

static struct rfcomm_session *rfcomm_session_create(bdaddr_t *src,
							bdaddr_t *dst,
							u8 sec_level,
							int *err);
static struct rfcomm_session *rfcomm_session_get(bdaddr_t *src, bdaddr_t *dst);
static struct rfcomm_session *rfcomm_session_del(struct rfcomm_session *s);

/* ---- RFCOMM frame parsing macros ---- */
#define __get_dlci(b)     ((b & 0xfc) >> 2)
#define __get_type(b)     ((b & 0xef))

#define __test_ea(b)      ((b & 0x01))
#define __test_cr(b)      (!!(b & 0x02))
#define __test_pf(b)      (!!(b & 0x10))

#define __session_dir(s)  ((s)->initiator ? 0x00 : 0x01)

#define __addr(cr, dlci)       (((dlci & 0x3f) << 2) | (cr << 1) | 0x01)
#define __ctrl(type, pf)       (((type & 0xef) | (pf << 4)))
#define __dlci(dir, chn)       (((chn & 0x1f) << 1) | dir)
#define __srv_channel(dlci)    (dlci >> 1)

#define __len8(len)       (((len) << 1) | 1)
#define __len16(len)      ((len) << 1)

/* MCC macros */
#define __mcc_type(cr, type)   (((type << 2) | (cr << 1) | 0x01))
#define __get_mcc_type(b) ((b & 0xfc) >> 2)
#define __get_mcc_len(b)  ((b & 0xfe) >> 1)

/* RPN macros */
#define __rpn_line_settings(data, stop, parity)  ((data & 0x3) | ((stop & 0x1) << 2) | ((parity & 0x7) << 3))
#define __get_rpn_data_bits(line) ((line) & 0x3)
#define __get_rpn_stop_bits(line) (((line) >> 2) & 0x1)
#define __get_rpn_parity(line)    (((line) >> 3) & 0x7)

static DECLARE_WAIT_QUEUE_HEAD(rfcomm_wq);

/**
 * @brief Schedules the RFCOMM worker thread to run.
 *
 * This function wakes up the RFCOMM kthread, allowing it to process pending
 * events and data.
 */
static void rfcomm_schedule(void)
{
	wake_up_all(&rfcomm_wq);
}

/* ---- RFCOMM FCS computation ---- */

/* reversed, 8-bit, poly=0x07 */
static unsigned char rfcomm_crc_table[256] = {
	0x00, 0x91, 0xe3, 0x72, 0x07, 0x96, 0xe4, 0x75,
	0x0e, 0x9f, 0xed, 0x7c, 0x09, 0x98, 0xea, 0x7b,
	0x1c, 0x8d, 0xff, 0x6e, 0x1b, 0x8a, 0xf8, 0x69,
	0x12, 0x83, 0xf1, 0x60, 0x15, 0x84, 0xf6, 0x67,

	0x38, 0xa9, 0xdb, 0x4a, 0x3f, 0xae, 0xdc, 0x4d,
	0x36, 0xa7, 0xd5, 0x44, 0x31, 0xa0, 0xd2, 0x43,
	0x24, 0xb5, 0xc7, 0x56, 0x23, 0xb2, 0xc0, 0x51,
	0x2a, 0xbb, 0xc9, 0x58, 0x2d, 0xbc, 0xce, 0x5f,

	0x70, 0xe1, 0x93, 0x02, 0x77, 0xe6, 0x94, 0x05,
	0x7e, 0xef, 0x9d, 0x0c, 0x79, 0xe8, 0x9a, 0x0b,
	0x6c, 0xfd, 0x8f, 0x1e, 0x6b, 0xfa, 0x88, 0x19,
	0x62, 0xf3, 0x81, 0x10, 0x65, 0xf4, 0x86, 0x17,

	0x48, 0xd9, 0xab, 0x3a, 0x4f, 0xde, 0xac, 0x3d,
	0x46, 0xd7, 0xa5, 0x34, 0x41, 0xd0, 0xa2, 0x33,
	0x54, 0xc5, 0xb7, 0x26, 0x53, 0xc2, 0xb0, 0x21,
	0x5a, 0xcb, 0xb9, 0x28, 0x5d, 0xcc, 0xbe, 0x2f,

	0xe0, 0x71, 0x03, 0x92, 0xe7, 0x76, 0x04, 0x95,
	0xee, 0x7f, 0x0d, 0x9c, 0xe9, 0x78, 0x0a, 0x9b,
	0xfc, 0x6d, 0x1f, 0x8e, 0xfb, 0x6a, 0x18, 0x89,
	0xf2, 0x63, 0x11, 0x80, 0xf5, 0x64, 0x16, 0x87,

	0xd8, 0x49, 0x3b, 0xaa, 0xdf, 0x4e, 0x3c, 0xad,
	0xd6, 0x47, 0x35, 0xa4, 0xd1, 0x40, 0x32, 0xa3,
	0xc4, 0x55, 0x27, 0xb6, 0xc3, 0x52, 0x20, 0xb1,
	0xca, 0x5b, 0x29, 0xb8, 0xcd, 0x5c, 0x2e, 0xbf,

	0x90, 0x01, 0x73, 0xe2, 0x97, 0x06, 0x74, 0xe5,
	0x9e, 0x0f, 0x7d, 0xec, 0x99, 0x08, 0x7a, 0xeb,
	0x8c, 0x1d, 0x6f, 0xfe, 0x8b, 0x1a, 0x68, 0xf9,
	0x82, 0x13, 0x61, 0xf0, 0x85, 0x14, 0x66, 0xf7,

	0xa8, 0x39, 0x4b, 0xda, 0xaf, 0x3e, 0x4c, 0xdd,
	0xa6, 0x37, 0x45, 0xd4, 0xa1, 0x30, 0x42, 0xd3,
	0xb4, 0x25, 0x57, 0xc6, 0xb3, 0x22, 0x50, 0xc1,
	0xba, 0x2b, 0x59, 0xc8, 0xbd, 0x2c, 0x5e, 0xcf
};

/* CRC on 2 bytes */
#define __crc(data) (rfcomm_crc_table[rfcomm_crc_table[0xff ^ data[0]] ^ data[1]])

/* FCS on 2 bytes */
static inline u8 __fcs(u8 *data)
{
	return 0xff - __crc(data);
}

/* FCS on 3 bytes */
static inline u8 __fcs2(u8 *data)
{
	return 0xff - rfcomm_crc_table[__crc(data) ^ data[2]];
}

/* Check FCS */
/**
 * @brief Verifies the Frame Check Sequence (FCS) of a received RFCOMM frame.
 * @param data Pointer to the start of the frame data used for FCS calculation.
 * @param type The type of the RFCOMM frame (e.g., UIH, SABM).
 * @param fcs The received FCS value from the frame.
 * @return 0 if the FCS is valid, non-zero otherwise.
 *
 * This function calculates the expected FCS for the given frame data and
 * compares it with the received FCS to ensure data integrity.
 */
static inline int __check_fcs(u8 *data, int type, u8 fcs)
{
	u8 f = __crc(data);

	if (type != RFCOMM_UIH)
		f = rfcomm_crc_table[f ^ data[2]];

	return rfcomm_crc_table[f ^ fcs] != 0xcf;
}

/* ---- L2CAP callbacks ---- */
/**
 * @brief Callback function for L2CAP socket state changes.
 * @param sk The L2CAP socket.
 *
 * This function is invoked by the L2CAP layer when the state of the underlying
 * L2CAP connection changes. It schedules the RFCOMM worker thread to handle
 * the state change.
 */
static void rfcomm_l2state_change(struct sock *sk)
{
	BT_DBG("%p state %d", sk, sk->sk_state);
	rfcomm_schedule();
}

/**
 * @brief Callback function for L2CAP data ready events.
 * @param sk The L2CAP socket.
 *
 * This function is invoked by the L2CAP layer when new data is available on
 * the L2CAP connection. It schedules the RFCOMM worker thread to process
 * the incoming data.
 */
static void rfcomm_l2data_ready(struct sock *sk)
{
	trace_sk_data_ready(sk);

	BT_DBG("%p", sk);
	rfcomm_schedule();
}
/**
 * @brief Creates an L2CAP socket for an RFCOMM session.
 * @param sock A pointer to a socket pointer, which will be filled with the
 *             created socket.
 * @return 0 on success, or a negative error code on failure.
 */
static int rfcomm_l2sock_create(struct socket **sock)
{
	int err;

	BT_DBG("");

	err = sock_create_kern(&init_net, PF_BLUETOOTH, SOCK_SEQPACKET, BTPROTO_L2CAP, sock);
	if (!err) {
		struct sock *sk = (*sock)->sk;
		sk->sk_data_ready   = rfcomm_l2data_ready;
		sk->sk_state_change = rfcomm_l2state_change;
	}
	return err;
}
/**
 * @brief Checks if the security level of an RFCOMM DLC is sufficient.
 * @param d The RFCOMM DLC.
 * @return 0 if the security level is sufficient, or a negative error code
 *         if authentication or encryption is required.
 */
static int rfcomm_check_security(struct rfcomm_dlc *d)
{
	struct sock *sk = d->session->sock->sk;
	struct l2cap_conn *conn = l2cap_pi(sk)->chan->conn;

	__u8 auth_type;

	switch (d->sec_level) {
	case BT_SECURITY_HIGH:
	case BT_SECURITY_FIPS:
		auth_type = HCI_AT_GENERAL_BONDING_MITM;
		break;
	case BT_SECURITY_MEDIUM:
		auth_type = HCI_AT_GENERAL_BONDING;
		break;
	default:
		auth_type = HCI_AT_NO_BONDING;
		break;
	}

	return hci_conn_security(conn->hcon, d->sec_level, auth_type,
				 d->out);
}
/**
 * @brief Callback function for RFCOMM session timeouts.
 * @param t The timer list structure.
 */
static void rfcomm_session_timeout(struct timer_list *t)
{
	struct rfcomm_session *s = timer_container_of(s, t, timer);

	BT_DBG("session %p state %ld", s, s->state);

	set_bit(RFCOMM_TIMED_OUT, &s->flags);
	rfcomm_schedule();
}
/**
 * @brief Sets a timer for an RFCOMM session.
 * @param s The RFCOMM session.
 * @param timeout The timeout value in jiffies.
 */
static void rfcomm_session_set_timer(struct rfcomm_session *s, long timeout)
{
	BT_DBG("session %p state %ld timeout %ld", s, s->state, timeout);

	mod_timer(&s->timer, jiffies + timeout);
}
/**
 * @brief Clears the timer for an RFCOMM session.
 * @param s The RFCOMM session.
 */
static void rfcomm_session_clear_timer(struct rfcomm_session *s)
{
	BT_DBG("session %p state %ld", s, s->state);

	timer_delete_sync(&s->timer);
}

/* ---- RFCOMM DLCs ---- */
/**
 * @brief Callback function for RFCOMM DLC timeouts.
 * @param t The timer list structure.
 */
static void rfcomm_dlc_timeout(struct timer_list *t)
{
	struct rfcomm_dlc *d = timer_container_of(d, t, timer);

	BT_DBG("dlc %p state %ld", d, d->state);

	set_bit(RFCOMM_TIMED_OUT, &d->flags);
	rfcomm_dlc_put(d);
	rfcomm_schedule();
}
/**
 * @brief Sets a timer for an RFCOMM DLC.
 * @param d The RFCOMM DLC.
 * @param timeout The timeout value in jiffies.
 */
static void rfcomm_dlc_set_timer(struct rfcomm_dlc *d, long timeout)
{
	BT_DBG("dlc %p state %ld timeout %ld", d, d->state, timeout);

	if (!mod_timer(&d->timer, jiffies + timeout))
		rfcomm_dlc_hold(d);
}

/**
 * @brief Clears the timer for an RFCOMM DLC.
 * @param d The RFCOMM DLC.
 */
static void rfcomm_dlc_clear_timer(struct rfcomm_dlc *d)
{
	BT_DBG("dlc %p state %ld", d, d->state);

	if (timer_delete(&d->timer))
		rfcomm_dlc_put(d);
}
/**
 * @brief Resets the state of an RFCOMM DLC.
 * @param d The RFCOMM DLC.
 */
static void rfcomm_dlc_clear_state(struct rfcomm_dlc *d)
{
	BT_DBG("%p", d);

	d->state      = BT_OPEN;
	d->flags      = 0;
	d->mscex      = 0;
	d->sec_level  = BT_SECURITY_LOW;
	d->mtu        = RFCOMM_DEFAULT_MTU;
	d->v24_sig    = RFCOMM_V24_RTC | RFCOMM_V24_RTR | RFCOMM_V24_DV;

	d->cfc        = RFCOMM_CFC_DISABLED;
	d->rx_credits = RFCOMM_DEFAULT_CREDITS;
}
/**
 * @brief Allocates and initializes a new RFCOMM DLC.
 * @param prio The GFP allocation flags.
 * @return A pointer to the new DLC, or NULL on failure.
 */
struct rfcomm_dlc *rfcomm_dlc_alloc(gfp_t prio)
{
	struct rfcomm_dlc *d = kzalloc(sizeof(*d), prio);

	if (!d)
		return NULL;

	timer_setup(&d->timer, rfcomm_dlc_timeout, 0);

	skb_queue_head_init(&d->tx_queue);
	mutex_init(&d->lock);
	refcount_set(&d->refcnt, 1);

	rfcomm_dlc_clear_state(d);

	BT_DBG("%p", d);

	return d;
}
/**
 * @brief Frees the resources associated with an RFCOMM DLC.
 * @param d The RFCOMM DLC.
 */
void rfcomm_dlc_free(struct rfcomm_dlc *d)
{
	BT_DBG("%p", d);

	skb_queue_purge(&d->tx_queue);
	kfree(d);
}
/**
 * @brief Links an RFCOMM DLC to a session.
 * @param s The RFCOMM session.
 * @param d The RFCOMM DLC.
 */
static void rfcomm_dlc_link(struct rfcomm_session *s, struct rfcomm_dlc *d)
{
	BT_DBG("dlc %p session %p", d, s);

	rfcomm_session_clear_timer(s);
	rfcomm_dlc_hold(d);
	list_add(&d->list, &s->dlcs);
	d->session = s;
}
/**
 * @brief Unlinks an RFCOMM DLC from a session.
 * @param d The RFCOMM DLC.
 */
static void rfcomm_dlc_unlink(struct rfcomm_dlc *d)
{
	struct rfcomm_session *s = d->session;

	BT_DBG("dlc %p refcnt %d session %p", d, refcount_read(&d->refcnt), s);

	list_del(&d->list);
	d->session = NULL;
	rfcomm_dlc_put(d);

	if (list_empty(&s->dlcs))
		rfcomm_session_set_timer(s, RFCOMM_IDLE_TIMEOUT);
}
/**
 * @brief Gets an RFCOMM DLC from a session by its DLCI.
 * @param s The RFCOMM session.
 * @param dlci The DLCI of the DLC to retrieve.
 * @return A pointer to the DLC, or NULL if not found.
 */
static struct rfcomm_dlc *rfcomm_dlc_get(struct rfcomm_session *s, u8 dlci)
{
	struct rfcomm_dlc *d;

	list_for_each_entry(d, &s->dlcs, list)
		if (d->dlci == dlci)
			return d;

	return NULL;
}
/**
 * @brief Checks if a channel number is valid for RFCOMM.
 * @param channel The channel number to check.
 * @return 0 if valid, non-zero otherwise.
 */
static int rfcomm_check_channel(u8 channel)
{
	return channel < 1 || channel > 30;
}
/**
 * @brief Opens an RFCOMM DLC.
 * @param d The RFCOMM DLC.
 * @param src The source Bluetooth address.
 * @param dst The destination Bluetooth address.
 * @param channel The RFCOMM channel number.
 * @return 0 on success, or a negative error code on failure.
 *
 * This function initiates the opening of an RFCOMM DLC, which may involve
 * creating a new RFCOMM session if one does not already exist.
 */
static int __rfcomm_dlc_open(struct rfcomm_dlc *d, bdaddr_t *src, bdaddr_t *dst, u8 channel)
{
	struct rfcomm_session *s;
	int err = 0;
	u8 dlci;

	BT_DBG("dlc %p state %ld %pMR -> %pMR channel %d",
	       d, d->state, src, dst, channel);

	if (rfcomm_check_channel(channel))
		return -EINVAL;

	if (d->state != BT_OPEN && d->state != BT_CLOSED)
		return 0;

	s = rfcomm_session_get(src, dst);
	if (!s) {
		s = rfcomm_session_create(src, dst, d->sec_level, &err);
		if (!s)
			return err;
	}

	dlci = __dlci(__session_dir(s), channel);

	/* Check if DLCI already exists */
	if (rfcomm_dlc_get(s, dlci))
		return -EBUSY;

	rfcomm_dlc_clear_state(d);

	d->dlci     = dlci;
	d->addr     = __addr(s->initiator, dlci);
	d->priority = 7;

	d->state = BT_CONFIG;
	rfcomm_dlc_link(s, d);

	d->out = 1;

	d->mtu = s->mtu;
	d->cfc = (s->cfc == RFCOMM_CFC_UNKNOWN) ? 0 : s->cfc;

	if (s->state == BT_CONNECTED) {
		if (rfcomm_check_security(d))
			rfcomm_send_pn(s, 1, d);
		else
			set_bit(RFCOMM_AUTH_PENDING, &d->flags);
	}

	rfcomm_dlc_set_timer(d, RFCOMM_CONN_TIMEOUT);

	return 0;
}
/**
 * @brief Disconnects an RFCOMM DLC.
 * @param d The RFCOMM DLC.
 *
 * This function initiates the disconnection of an RFCOMM DLC by sending a
 * DISC frame.
 */
static void __rfcomm_dlc_disconn(struct rfcomm_dlc *d)
{
	struct rfcomm_session *s = d->session;

	d->state = BT_DISCONN;
	if (skb_queue_empty(&d->tx_queue)) {
		rfcomm_send_disc(s, d->dlci);
		rfcomm_dlc_set_timer(d, RFCOMM_DISC_TIMEOUT);
	} else {
		rfcomm_queue_disc(d);
		rfcomm_dlc_set_timer(d, RFCOMM_DISC_TIMEOUT * 2);
	}
}
/**
 * @brief Closes an RFCOMM DLC.
 * @param d The RFCOMM DLC.
 * @param err The error code to report to the upper layer.
 * @return 0 on success.
 */
static int __rfcomm_dlc_close(struct rfcomm_dlc *d, int err)
{
	struct rfcomm_session *s = d->session;
	if (!s)
		return 0;

	BT_DBG("dlc %p state %ld dlci %d err %d session %p",
			d, d->state, d->dlci, err, s);

	switch (d->state) {
	case BT_CONNECT:
	case BT_CONFIG:
	case BT_OPEN:
	case BT_CONNECT2:
		if (test_and_clear_bit(RFCOMM_DEFER_SETUP, &d->flags)) {
			set_bit(RFCOMM_AUTH_REJECT, &d->flags);
			rfcomm_schedule();
			return 0;
		}
	}

	switch (d->state) {
	case BT_CONNECT:
	case BT_CONNECTED:
		__rfcomm_dlc_disconn(d);
		break;

	case BT_CONFIG:
		if (s->state != BT_BOUND) {
			__rfcomm_dlc_disconn(d);
			break;
		}
		/* if closing a dlc in a session that hasn't been started,
		 * just close and unlink the dlc
		 */
		fallthrough;

	default:
		rfcomm_dlc_clear_timer(d);

		rfcomm_dlc_lock(d);
		d->state = BT_CLOSED;
		d->state_change(d, err);
		rfcomm_dlc_unlock(d);

		skb_queue_purge(&d->tx_queue);
		rfcomm_dlc_unlink(d);
	}

	return 0;
}
/**
 * @brief Checks if an RFCOMM DLC exists for a given connection and channel.
 * @param src The source Bluetooth address.
 * @param dst The destination Bluetooth address.
 * @param channel The RFCOMM channel number.
 * @return A pointer to the DLC if it exists, otherwise an error pointer.
 */
struct rfcomm_dlc *rfcomm_dlc_exists(bdaddr_t *src, bdaddr_t *dst, u8 channel)
{
	struct rfcomm_session *s;
	struct rfcomm_dlc *dlc = NULL;
	u8 dlci;

	if (rfcomm_check_channel(channel))
		return ERR_PTR(-EINVAL);

	rfcomm_lock();
	s = rfcomm_session_get(src, dst);
	if (s) {
		dlci = __dlci(__session_dir(s), channel);
		dlc = rfcomm_dlc_get(s, dlci);
	}
	rfcomm_unlock();
	return dlc;
}
/**
 * @brief Sends a data fragment on an RFCOMM DLC.
 * @param d The RFCOMM DLC.
 * @param frag The socket buffer containing the fragment.
 * @return The number of bytes sent, or a negative error code on failure.
 */
static int rfcomm_dlc_send_frag(struct rfcomm_dlc *d, struct sk_buff *frag)
{
	int len = frag->len;

	BT_DBG("dlc %p mtu %d len %d", d, d->mtu, len);

	if (len > d->mtu)
		return -EINVAL;

	rfcomm_make_uih(frag, d->addr);
	__skb_queue_tail(&d->tx_queue, frag);

	return len;
}
/**
 * @brief Throttles an RFCOMM DLC, preventing it from sending more data.
 * @param d The RFCOMM DLC.
 *
 * This is used for flow control.
 */
void __rfcomm_dlc_throttle(struct rfcomm_dlc *d)
{
	BT_DBG("dlc %p state %ld", d, d->state);

	if (!d->cfc) {
		d->v24_sig |= RFCOMM_V24_FC;
		set_bit(RFCOMM_MSC_PENDING, &d->flags);
	}
	rfcomm_schedule();
}
/**
 * @brief Unthrottles an RFCOMM DLC, allowing it to send more data.
 * @param d The RFCOMM DLC.
 *
 * This is used for flow control.
 */
void __rfcomm_dlc_unthrottle(struct rfcomm_dlc *d)
{
	BT_DBG("dlc %p state %ld", d, d->state);

	if (!d->cfc) {
		d->v24_sig &= ~RFCOMM_V24_FC;
		set_bit(RFCOMM_MSC_PENDING, &d->flags);
	}
	rfcomm_schedule();
}

/**
 * @brief Sets the modem status for an RFCOMM DLC.
 * @param d The RFCOMM DLC.
 * @param v24_sig The new modem status signals.
 * @return 0 on success.
 */
int rfcomm_dlc_set_modem_status(struct rfcomm_dlc *d, u8 v24_sig)
{
	BT_DBG("dlc %p state %ld v24_sig 0x%x",
			d, d->state, v24_sig);

	if (test_bit(RFCOMM_RX_THROTTLED, &d->flags))
		v24_sig |= RFCOMM_V24_FC;
	else
		v24_sig &= ~RFCOMM_V24_FC;

	d->v24_sig = v24_sig;

	if (!test_and_set_bit(RFCOMM_MSC_PENDING, &d->flags))
		rfcomm_schedule();

	return 0;
}
/**
 * @brief Gets the modem status for an RFCOMM DLC.
 * @param d The RFCOMM DLC.
 * @param v24_sig A pointer to store the modem status signals.
 * @return 0 on success.
 */
int rfcomm_dlc_get_modem_status(struct rfcomm_dlc *d, u8 *v24_sig)
{
	BT_DBG("dlc %p state %ld v24_sig 0x%x",
			d, d->state, d->v24_sig);

	*v24_sig = d->v24_sig;
	return 0;
}

/* ---- RFCOMM sessions ---- */
/**
 * @brief Adds a new RFCOMM session.
 * @param sock The L2CAP socket for the session.
 * @param state The initial state of the session.
 * @return A pointer to the new session, or NULL on failure.
 */
static struct rfcomm_session *rfcomm_session_add(struct socket *sock, int state)
{
	struct rfcomm_session *s = kzalloc(sizeof(*s), GFP_KERNEL);

	if (!s)
		return NULL;

	BT_DBG("session %p sock %p", s, sock);

	timer_setup(&s->timer, rfcomm_session_timeout, 0);

	INIT_LIST_HEAD(&s->dlcs);
	s->state = state;
	s->sock  = sock;

	s->mtu = RFCOMM_DEFAULT_MTU;
	s->cfc = disable_cfc ? RFCOMM_CFC_DISABLED : RFCOMM_CFC_UNKNOWN;

	/* Do not increment module usage count for listening sessions.
	 * Otherwise we won't be able to unload the module. */
	if (state != BT_LISTEN)
		if (!try_module_get(THIS_MODULE)) {
			kfree(s);
			return NULL;
		}

	list_add(&s->list, &session_list);

	return s;
}
/**
 * @brief Deletes an RFCOMM session.
 * @param s The RFCOMM session to delete.
 * @return NULL.
 */
static struct rfcomm_session *rfcomm_session_del(struct rfcomm_session *s)
{
	int state = s->state;

	BT_DBG("session %p state %ld", s, s->state);

	list_del(&s->list);

	rfcomm_session_clear_timer(s);
	sock_release(s->sock);
	kfree(s);

	if (state != BT_LISTEN)
		module_put(THIS_MODULE);

	return NULL;
}
/**
 * @brief Gets an RFCOMM session by source and destination addresses.
 * @param src The source Bluetooth address.
 * @param dst The destination Bluetooth address.
 * @return A pointer to the session, or NULL if not found.
 */
static struct rfcomm_session *rfcomm_session_get(bdaddr_t *src, bdaddr_t *dst)
{
	struct rfcomm_session *s, *n;
	struct l2cap_chan *chan;
	list_for_each_entry_safe(s, n, &session_list, list) {
		chan = l2cap_pi(s->sock->sk)->chan;

		if ((!bacmp(src, BDADDR_ANY) || !bacmp(&chan->src, src)) &&
		    !bacmp(&chan->dst, dst))
			return s;
	}
	return NULL;
}
/**
 * @brief Closes an RFCOMM session.
 * @param s The RFCOMM session.
 * @param err The error code to report to the DLCs.
 * @return NULL.
 */
static struct rfcomm_session *rfcomm_session_close(struct rfcomm_session *s,
						   int err)
{
	struct rfcomm_dlc *d, *n;

	s->state = BT_CLOSED;

	BT_DBG("session %p state %ld err %d", s, s->state, err);

	/* Close all dlcs */
	list_for_each_entry_safe(d, n, &s->dlcs, list) {
		d->state = BT_CLOSED;
		__rfcomm_dlc_close(d, err);
	}

	rfcomm_session_clear_timer(s);
	return rfcomm_session_del(s);
}
/**
 * @brief Creates a new RFCOMM session.
 * @param src The source Bluetooth address.
 * @param dst The destination Bluetooth address.
 * @param sec_level The security level for the connection.
 * @param err A pointer to an integer to store the error code.
 * @return A pointer to the new session, or NULL on failure.
 */
static struct rfcomm_session *rfcomm_session_create(bdaddr_t *src,
							bdaddr_t *dst,
							u8 sec_level,
							int *err)
{
	struct rfcomm_session *s = NULL;
	struct sockaddr_l2 addr;
	struct socket *sock;
	struct sock *sk;

	BT_DBG("%pMR -> %pMR", src, dst);

	*err = rfcomm_l2sock_create(&sock);
	if (*err < 0)
		return NULL;

	bacpy(&addr.l2_bdaddr, src);
	addr.l2_family = AF_BLUETOOTH;
	addr.l2_psm    = 0;
	addr.l2_cid    = 0;
	addr.l2_bdaddr_type = BDADDR_BREDR;
	*err = kernel_bind(sock, (struct sockaddr *) &addr, sizeof(addr));
	if (*err < 0)
		goto failed;

	/* Set L2CAP options */
	sk = sock->sk;
	lock_sock(sk);
	/* Set MTU to 0 so L2CAP can auto select the MTU */
	l2cap_pi(sk)->chan->imtu = 0;
	l2cap_pi(sk)->chan->sec_level = sec_level;
	if (l2cap_ertm)
		l2cap_pi(sk)->chan->mode = L2CAP_MODE_ERTM;
	release_sock(sk);

	s = rfcomm_session_add(sock, BT_BOUND);
	if (!s) {
		*err = -ENOMEM;
		goto failed;
	}

	s->initiator = 1;

	bacpy(&addr.l2_bdaddr, dst);
	addr.l2_family = AF_BLUETOOTH;
	addr.l2_psm    = cpu_to_le16(L2CAP_PSM_RFCOMM);
	addr.l2_cid    = 0;
	addr.l2_bdaddr_type = BDADDR_BREDR;
	*err = kernel_connect(sock, (struct sockaddr *) &addr, sizeof(addr), O_NONBLOCK);
	if (*err == 0 || *err == -EINPROGRESS)
		return s;

	return rfcomm_session_del(s);

failed:
	sock_release(sock);
	return NULL;
}
/**
 * @brief Gets the source and destination addresses of an RFCOMM session.
 * @param s The RFCOMM session.
 * @param src A pointer to store the source address.
 * @param dst A pointer to store the destination address.
 */
void rfcomm_session_getaddr(struct rfcomm_session *s, bdaddr_t *src, bdaddr_t *dst)
{
	struct l2cap_chan *chan = l2cap_pi(s->sock->sk)->chan;
	if (src)
		bacpy(src, &chan->src);
	if (dst)
		bacpy(dst, &chan->dst);
}

/* ---- RFCOMM frame sending ---- */
/**
 * @brief Sends a raw RFCOMM frame.
 * @param s The RFCOMM session.
 * @param data The data to send.
 * @param len The length of the data.
 * @return The number of bytes sent, or a negative error code on failure.
 */
static int rfcomm_send_frame(struct rfcomm_session *s, u8 *data, int len)
{
	struct kvec iv = { data, len };
	struct msghdr msg;

	BT_DBG("session %p len %d", s, len);

	memset(&msg, 0, sizeof(msg));

	return kernel_sendmsg(s->sock, &msg, &iv, 1, len);
}

/**
 * @brief Sends an RFCOMM command frame.
 * @param s The RFCOMM session.
 * @param cmd The command to send.
 * @return The number of bytes sent, or a negative error code on failure.
 */
static int rfcomm_send_cmd(struct rfcomm_session *s, struct rfcomm_cmd *cmd)
{
	BT_DBG("%p cmd %u", s, cmd->ctrl);

	return rfcomm_send_frame(s, (void *) cmd, sizeof(*cmd));
}
/**
 * @brief Sends a Set Asynchronous Balanced Mode (SABM) frame.
 * @param s The RFCOMM session.
 * @param dlci The DLCI for the frame.
 * @return 0 on success, or a negative error code on failure.
 */
static int rfcomm_send_sabm(struct rfcomm_session *s, u8 dlci)
{
	struct rfcomm_cmd cmd;

	BT_DBG("%p dlci %d", s, dlci);

	cmd.addr = __addr(s->initiator, dlci);
	cmd.ctrl = __ctrl(RFCOMM_SABM, 1);
	cmd.len  = __len8(0);
	cmd.fcs  = __fcs2((u8 *) &cmd);

	return rfcomm_send_cmd(s, &cmd);
}
/**
 * @brief Sends an Unnumbered Acknowledgment (UA) frame.
 * @param s The RFCOMM session.
 * @param dlci The DLCI for the frame.
 * @return 0 on success, or a negative error code on failure.
 */
static int rfcomm_send_ua(struct rfcomm_session *s, u8 dlci)
{
	struct rfcomm_cmd cmd;

	BT_DBG("%p dlci %d", s, dlci);

	cmd.addr = __addr(!s->initiator, dlci);
	cmd.ctrl = __ctrl(RFCOMM_UA, 1);
	cmd.len  = __len8(0);
	cmd.fcs  = __fcs2((u8 *) &cmd);

	return rfcomm_send_cmd(s, &cmd);
}
/**
 * @brief Sends a Disconnect (DISC) frame.
 * @param s The RFCOMM session.
 * @param dlci The DLCI for the frame.
 * @return 0 on success, or a negative error code on failure.
 */
static int rfcomm_send_disc(struct rfcomm_session *s, u8 dlci)
{
	struct rfcomm_cmd cmd;

	BT_DBG("%p dlci %d", s, dlci);

	cmd.addr = __addr(s->initiator, dlci);
	cmd.ctrl = __ctrl(RFCOMM_DISC, 1);
	cmd.len  = __len8(0);
	cmd.fcs  = __fcs2((u8 *) &cmd);

	return rfcomm_send_cmd(s, &cmd);
}

/**
 * @brief Queues a Disconnect (DISC) frame for sending.
 * @param d The RFCOMM DLC.
 * @return 0 on success, or -ENOMEM on failure.
 */
static int rfcomm_queue_disc(struct rfcomm_dlc *d)
{
	struct rfcomm_cmd *cmd;
	struct sk_buff *skb;

	BT_DBG("dlc %p dlci %d", d, d->dlci);

	skb = alloc_skb(sizeof(*cmd), GFP_KERNEL);
	if (!skb)
		return -ENOMEM;

	cmd = __skb_put(skb, sizeof(*cmd));
	cmd->addr = d->addr;
	cmd->ctrl = __ctrl(RFCOMM_DISC, 1);
	cmd->len  = __len8(0);
	cmd->fcs  = __fcs2((u8 *) cmd);

	skb_queue_tail(&d->tx_queue, skb);
	rfcomm_schedule();
	return 0;
}
/**
 * @brief Sends a Disconnected Mode (DM) frame.
 * @param s The RFCOMM session.
 * @param dlci The DLCI for the frame.
 * @return 0 on success, or a negative error code on failure.
 */
static int rfcomm_send_dm(struct rfcomm_session *s, u8 dlci)
{
	struct rfcomm_cmd cmd;

	BT_DBG("%p dlci %d", s, dlci);

	cmd.addr = __addr(!s->initiator, dlci);
	cmd.ctrl = __ctrl(RFCOMM_DM, 1);
	cmd.len  = __len8(0);
	cmd.fcs  = __fcs2((u8 *) &cmd);

	return rfcomm_send_cmd(s, &cmd);
}
/**
 * @brief Sends a Non Supported Command (NSC) frame.
 * @param s The RFCOMM session.
 * @param cr The command/response flag.
 * @param type The type of the unsupported command.
 * @return 0 on success, or a negative error code on failure.
 */
static int rfcomm_send_nsc(struct rfcomm_session *s, int cr, u8 type)
{
	struct rfcomm_hdr *hdr;
	struct rfcomm_mcc *mcc;
	u8 buf[16], *ptr = buf;

	BT_DBG("%p cr %d type %d", s, cr, type);

	hdr = (void *) ptr; ptr += sizeof(*hdr);
	hdr->addr = __addr(s->initiator, 0);
	hdr->ctrl = __ctrl(RFCOMM_UIH, 0);
	hdr->len  = __len8(sizeof(*mcc) + 1);

	mcc = (void *) ptr; ptr += sizeof(*mcc);
	mcc->type = __mcc_type(0, RFCOMM_NSC);
	mcc->len  = __len8(1);

	/* Type that we didn't like */
	*ptr = __mcc_type(cr, type); ptr++;

	*ptr = __fcs(buf); ptr++;

	return rfcomm_send_frame(s, buf, ptr - buf);
}
/**
 * @brief Sends a Parameter Negotiation (PN) frame.
 * @param s The RFCOMM session.
 * @param cr The command/response flag.
 * @param d The RFCOMM DLC.
 * @return 0 on success, or a negative error code on failure.
 */
static int rfcomm_send_pn(struct rfcomm_session *s, int cr, struct rfcomm_dlc *d)
{
	struct rfcomm_hdr *hdr;
	struct rfcomm_mcc *mcc;
	struct rfcomm_pn  *pn;
	u8 buf[16], *ptr = buf;

	BT_DBG("%p cr %d dlci %d mtu %d", s, cr, d->dlci, d->mtu);

	hdr = (void *) ptr; ptr += sizeof(*hdr);
	hdr->addr = __addr(s->initiator, 0);
	hdr->ctrl = __ctrl(RFCOMM_UIH, 0);
	hdr->len  = __len8(sizeof(*mcc) + sizeof(*pn));

	mcc = (void *) ptr; ptr += sizeof(*mcc);
	mcc->type = __mcc_type(cr, RFCOMM_PN);
	mcc->len  = __len8(sizeof(*pn));

	pn = (void *) ptr; ptr += sizeof(*pn);
	pn->dlci        = d->dlci;
	pn->priority    = d->priority;
	pn->ack_timer   = 0;
	pn->max_retrans = 0;

	if (s->cfc) {
		pn->flow_ctrl = cr ? 0xf0 : 0xe0;
		pn->credits = RFCOMM_DEFAULT_CREDITS;
	} else {
		pn->flow_ctrl = 0;
		pn->credits   = 0;
	}

	if (cr && channel_mtu >= 0)
		pn->mtu = cpu_to_le16(channel_mtu);
	else
		pn->mtu = cpu_to_le16(d->mtu);

	*ptr = __fcs(buf); ptr++;

	return rfcomm_send_frame(s, buf, ptr - buf);
}
/**
 * @brief Sends a Remote Port Negotiation (RPN) frame.
 * @param s The RFCOMM session.
 * @param cr The command/response flag.
 * @param dlci The DLCI.
 * @param bit_rate The bit rate.
 * @param data_bits The number of data bits.
 * @param stop_bits The number of stop bits.
 * @param parity The parity setting.
 * @param flow_ctrl_settings The flow control settings.
 * @param xon_char The XON character.
 * @param xoff_char The XOFF character.
 * @param param_mask A bitmask of which parameters are being set.
 * @return 0 on success, or a negative error code on failure.
 */
int rfcomm_send_rpn(struct rfcomm_session *s, int cr, u8 dlci,
			u8 bit_rate, u8 data_bits, u8 stop_bits,
			u8 parity, u8 flow_ctrl_settings,
			u8 xon_char, u8 xoff_char, u16 param_mask)
{
	struct rfcomm_hdr *hdr;
	struct rfcomm_mcc *mcc;
	struct rfcomm_rpn *rpn;
	u8 buf[16], *ptr = buf;

	BT_DBG("%p cr %d dlci %d bit_r 0x%x data_b 0x%x stop_b 0x%x parity 0x%x"
			" flwc_s 0x%x xon_c 0x%x xoff_c 0x%x p_mask 0x%x",
		s, cr, dlci, bit_rate, data_bits, stop_bits, parity,
		flow_ctrl_settings, xon_char, xoff_char, param_mask);

	hdr = (void *) ptr; ptr += sizeof(*hdr);
	hdr->addr = __addr(s->initiator, 0);
	hdr->ctrl = __ctrl(RFCOMM_UIH, 0);
	hdr->len  = __len8(sizeof(*mcc) + sizeof(*rpn));

	mcc = (void *) ptr; ptr += sizeof(*mcc);
	mcc->type = __mcc_type(cr, RFCOMM_RPN);
	mcc->len  = __len8(sizeof(*rpn));

	rpn = (void *) ptr; ptr += sizeof(*rpn);
	rpn->dlci          = __addr(1, dlci);
	rpn->bit_rate      = bit_rate;
	rpn->line_settings = __rpn_line_settings(data_bits, stop_bits, parity);
	rpn->flow_ctrl     = flow_ctrl_settings;
	rpn->xon_char      = xon_char;
	rpn->xoff_char     = xoff_char;
	rpn->param_mask    = cpu_to_le16(param_mask);

	*ptr = __fcs(buf); ptr++;

	return rfcomm_send_frame(s, buf, ptr - buf);
}
/**
 * @brief Sends a Remote Line Status (RLS) frame.
 * @param s The RFCOMM session.
 * @param cr The command/response flag.
 * @param dlci The DLCI.
 * @param status The line status.
 * @return 0 on success, or a negative error code on failure.
 */
static int rfcomm_send_rls(struct rfcomm_session *s, int cr, u8 dlci, u8 status)
{
	struct rfcomm_hdr *hdr;
	struct rfcomm_mcc *mcc;
	struct rfcomm_rls *rls;
	u8 buf[16], *ptr = buf;

	BT_DBG("%p cr %d status 0x%x", s, cr, status);

	hdr = (void *) ptr; ptr += sizeof(*hdr);
	hdr->addr = __addr(s->initiator, 0);
	hdr->ctrl = __ctrl(RFCOMM_UIH, 0);
	hdr->len  = __len8(sizeof(*mcc) + sizeof(*rls));

	mcc = (void *) ptr; ptr += sizeof(*mcc);
	mcc->type = __mcc_type(cr, RFCOMM_RLS);
	mcc->len  = __len8(sizeof(*rls));

	rls = (void *) ptr; ptr += sizeof(*rls);
	rls->dlci   = __addr(1, dlci);
	rls->status = status;

	*ptr = __fcs(buf); ptr++;

	return rfcomm_send_frame(s, buf, ptr - buf);
}
/**
 * @brief Sends a Modem Status Command (MSC) frame.
 * @param s The RFCOMM session.
 * @param cr The command/response flag.
 * @param dlci The DLCI.
 * @param v24_sig The V.24 signals.
 * @return 0 on success, or a negative error code on failure.
 */
static int rfcomm_send_msc(struct rfcomm_session *s, int cr, u8 dlci, u8 v24_sig)
{
	struct rfcomm_hdr *hdr;
	struct rfcomm_mcc *mcc;
	struct rfcomm_msc *msc;
	u8 buf[16], *ptr = buf;

	BT_DBG("%p cr %d v24 0x%x", s, cr, v24_sig);

	hdr = (void *) ptr; ptr += sizeof(*hdr);
	hdr->addr = __addr(s->initiator, 0);
	hdr->ctrl = __ctrl(RFCOMM_UIH, 0);
	hdr->len  = __len8(sizeof(*mcc) + sizeof(*msc));

	mcc = (void *) ptr; ptr += sizeof(*mcc);
	mcc->type = __mcc_type(cr, RFCOMM_MSC);
	mcc->len  = __len8(sizeof(*msc));

	msc = (void *) ptr; ptr += sizeof(*msc);
	msc->dlci    = __addr(1, dlci);
	msc->v24_sig = v24_sig | 0x01;

	*ptr = __fcs(buf); ptr++;

	return rfcomm_send_frame(s, buf, ptr - buf);
}
/**
 * @brief Sends a Flow Control Off (FCOFF) frame.
 * @param s The RFCOMM session.
 * @param cr The command/response flag.
 * @return 0 on success, or a negative error code on failure.
 */
static int rfcomm_send_fcoff(struct rfcomm_session *s, int cr)
{
	struct rfcomm_hdr *hdr;
	struct rfcomm_mcc *mcc;
	u8 buf[16], *ptr = buf;

	BT_DBG("%p cr %d", s, cr);

	hdr = (void *) ptr; ptr += sizeof(*hdr);
	hdr->addr = __addr(s->initiator, 0);
	hdr->ctrl = __ctrl(RFCOMM_UIH, 0);
	hdr->len  = __len8(sizeof(*mcc));

	mcc = (void *) ptr; ptr += sizeof(*mcc);
	mcc->type = __mcc_type(cr, RFCOMM_FCOFF);
	mcc->len  = __len8(0);

	*ptr = __fcs(buf); ptr++;

	return rfcomm_send_frame(s, buf, ptr - buf);
}
/**
 * @brief Sends a Flow Control On (FCON) frame.
 * @param s The RFCOMM session.
 * @param cr The command/response flag.
 * @return 0 on success, or a negative error code on failure.
 */
static int rfcomm_send_fcon(struct rfcomm_session *s, int cr)
{
	struct rfcomm_hdr *hdr;
	struct rfcomm_mcc *mcc;
	u8 buf[16], *ptr = buf;

	BT_DBG("%p cr %d", s, cr);

	hdr = (void *) ptr; ptr += sizeof(*hdr);
	hdr->addr = __addr(s->initiator, 0);
	hdr->ctrl = __ctrl(RFCOMM_UIH, 0);
	hdr->len  = __len8(sizeof(*mcc));

	mcc = (void *) ptr; ptr += sizeof(*mcc);
	mcc->type = __mcc_type(cr, RFCOMM_FCON);
	mcc->len  = __len8(0);

	*ptr = __fcs(buf); ptr++;

	return rfcomm_send_frame(s, buf, ptr - buf);
}
/**
 * @brief Sends a Test (TEST) frame.
 * @param s The RFCOMM session.
 * @param cr The command/response flag.
 * @param pattern The test pattern.
 * @param len The length of the test pattern.
 * @return The number of bytes sent, or a negative error code on failure.
 */
static int rfcomm_send_test(struct rfcomm_session *s, int cr, u8 *pattern, int len)
{
	struct socket *sock = s->sock;
	struct kvec iv[3];
	struct msghdr msg;
	unsigned char hdr[5], crc[1];

	if (len > 125)
		return -EINVAL;

	BT_DBG("%p cr %d", s, cr);

	hdr[0] = __addr(s->initiator, 0);
	hdr[1] = __ctrl(RFCOMM_UIH, 0);
	hdr[2] = 0x01 | ((len + 2) << 1);
	hdr[3] = 0x01 | ((cr & 0x01) << 1) | (RFCOMM_TEST << 2);
	hdr[4] = 0x01 | (len << 1);

	crc[0] = __fcs(hdr);

	iv[0].iov_base = hdr;
	iv[0].iov_len  = 5;
	iv[1].iov_base = pattern;
	iv[1].iov_len  = len;
	iv[2].iov_base = crc;
	iv[2].iov_len  = 1;

	memset(&msg, 0, sizeof(msg));

	return kernel_sendmsg(sock, &msg, iv, 3, 6 + len);
}
/**
 * @brief Sends credits for flow control.
 * @param s The RFCOMM session.
 * @param addr The address field for the frame.
 * @param credits The number of credits to send.
 * @return 0 on success, or a negative error code on failure.
 */
static int rfcomm_send_credits(struct rfcomm_session *s, u8 addr, u8 credits)
{
	struct rfcomm_hdr *hdr;
	u8 buf[16], *ptr = buf;

	BT_DBG("%p addr %d credits %d", s, addr, credits);

	hdr = (void *) ptr; ptr += sizeof(*hdr);
	hdr->addr = addr;
	hdr->ctrl = __ctrl(RFCOMM_UIH, 1);
	hdr->len  = __len8(0);

	*ptr = credits; ptr++;

	*ptr = __fcs(buf); ptr++;

	return rfcomm_send_frame(s, buf, ptr - buf);
}
/**
 * @brief Creates a UIH frame header.
 * @param skb The socket buffer to prepend the header to.
 * @param addr The address field for the frame.
 */
static void rfcomm_make_uih(struct sk_buff *skb, u8 addr)
{
	struct rfcomm_hdr *hdr;
	int len = skb->len;
	u8 *crc;

	if (len > 127) {
		hdr = skb_push(skb, 4);
		put_unaligned(cpu_to_le16(__len16(len)), (__le16 *) &hdr->len);
	} else {
		hdr = skb_push(skb, 3);
		hdr->len = __len8(len);
	}
	hdr->addr = addr;
	hdr->ctrl = __ctrl(RFCOMM_UIH, 0);

	crc = skb_put(skb, 1);
	*crc = __fcs((void *) hdr);
}

/* ---- RFCOMM frame reception ---- */
/**
 * @brief Handles a received Unnumbered Acknowledgment (UA) frame.
 * @param s The RFCOMM session.
 * @param dlci The DLCI of the frame.
 * @return The session pointer, which may have changed if the session was closed.
 */
static struct rfcomm_session *rfcomm_recv_ua(struct rfcomm_session *s, u8 dlci)
{
	BT_DBG("session %p state %ld dlci %d", s, s->state, dlci);

	if (dlci) {
		/* Data channel */
		struct rfcomm_dlc *d = rfcomm_dlc_get(s, dlci);
		if (!d) {
			rfcomm_send_dm(s, dlci);
			return s;
		}

		switch (d->state) {
		case BT_CONNECT:
			rfcomm_dlc_clear_timer(d);

			rfcomm_dlc_lock(d);
			d->state = BT_CONNECTED;
			d->state_change(d, 0);
			rfcomm_dlc_unlock(d);

			rfcomm_send_msc(s, 1, dlci, d->v24_sig);
			break;

		case BT_DISCONN:
			d->state = BT_CLOSED;
			__rfcomm_dlc_close(d, 0);

			if (list_empty(&s->dlcs)) {
				s->state = BT_DISCONN;
				rfcomm_send_disc(s, 0);
				rfcomm_session_clear_timer(s);
			}

			break;
		}
	} else {
		/* Control channel */
		switch (s->state) {
		case BT_CONNECT:
			s->state = BT_CONNECTED;
			rfcomm_process_connect(s);
			break;

		case BT_DISCONN:
			s = rfcomm_session_close(s, ECONNRESET);
			break;
		}
	}
	return s;
}
/**
 * @brief Handles a received Disconnected Mode (DM) frame.
 * @param s The RFCOMM session.
 * @param dlci The DLCI of the frame.
 * @return The session pointer, which may have changed if the session was closed.
 */
static struct rfcomm_session *rfcomm_recv_dm(struct rfcomm_session *s, u8 dlci)
{
	int err = 0;

	BT_DBG("session %p state %ld dlci %d", s, s->state, dlci);

	if (dlci) {
		/* Data DLC */
		struct rfcomm_dlc *d = rfcomm_dlc_get(s, dlci);
		if (d) {
			if (d->state == BT_CONNECT || d->state == BT_CONFIG)
				err = ECONNREFUSED;
			else
				err = ECONNRESET;

			d->state = BT_CLOSED;
			__rfcomm_dlc_close(d, err);
		}
	} else {
		if (s->state == BT_CONNECT)
			err = ECONNREFUSED;
		else
			err = ECONNRESET;

		s = rfcomm_session_close(s, err);
	}
	return s;
}
/**
 * @brief Handles a received Disconnect (DISC) frame.
 * @param s The RFCOMM session.
 * @param dlci The DLCI of the frame.
 * @return The session pointer, which may have changed if the session was closed.
 */
static struct rfcomm_session *rfcomm_recv_disc(struct rfcomm_session *s,
					       u8 dlci)
{
	int err = 0;

	BT_DBG("session %p state %ld dlci %d", s, s->state, dlci);

	if (dlci) {
		struct rfcomm_dlc *d = rfcomm_dlc_get(s, dlci);
		if (d) {
			rfcomm_send_ua(s, dlci);

			if (d->state == BT_CONNECT || d->state == BT_CONFIG)
				err = ECONNREFUSED;
			else
				err = ECONNRESET;

			d->state = BT_CLOSED;
			__rfcomm_dlc_close(d, err);
		} else
			rfcomm_send_dm(s, dlci);

	} else {
		rfcomm_send_ua(s, 0);

		if (s->state == BT_CONNECT)
			err = ECONNREFUSED;
		else
			err = ECONNRESET;

		s = rfcomm_session_close(s, err);
	}
	return s;
}
/**
 * @brief Accepts an incoming DLC connection.
 * @param d The RFCOMM DLC.
 */
void rfcomm_dlc_accept(struct rfcomm_dlc *d)
{
	struct sock *sk = d->session->sock->sk;
	struct l2cap_conn *conn = l2cap_pi(sk)->chan->conn;

	BT_DBG("dlc %p", d);

	rfcomm_send_ua(d->session, d->dlci);

	rfcomm_dlc_clear_timer(d);

	rfcomm_dlc_lock(d);
	d->state = BT_CONNECTED;
	d->state_change(d, 0);
	rfcomm_dlc_unlock(d);

	if (d->role_switch)
		hci_conn_switch_role(conn->hcon, 0x00);

	rfcomm_send_msc(d->session, 1, d->dlci, d->v24_sig);
}

/**
 * @brief Checks if an incoming connection can be accepted.
 * @param d The RFCOMM DLC.
 *
 * This function checks the security level of the connection and, if
 * sufficient, accepts it. If security is pending, it defers the setup.
 */
static void rfcomm_check_accept(struct rfcomm_dlc *d)
{
	if (rfcomm_check_security(d)) {
		if (d->defer_setup) {
			set_bit(RFCOMM_DEFER_SETUP, &d->flags);
			rfcomm_dlc_set_timer(d, RFCOMM_AUTH_TIMEOUT);

			rfcomm_dlc_lock(d);
			d->state = BT_CONNECT2;
			d->state_change(d, 0);
			rfcomm_dlc_unlock(d);
		} else
			rfcomm_dlc_accept(d);
	} else {
		set_bit(RFCOMM_AUTH_PENDING, &d->flags);
		rfcomm_dlc_set_timer(d, RFCOMM_AUTH_TIMEOUT);
	}
}
/**
 * @brief Handles a received Set Asynchronous Balanced Mode (SABM) frame.
 * @param s The RFCOMM session.
 * @param dlci The DLCI of the frame.
 * @return 0 on success.
 */
static int rfcomm_recv_sabm(struct rfcomm_session *s, u8 dlci)
{
	struct rfcomm_dlc *d;
	u8 channel;

	BT_DBG("session %p state %ld dlci %d", s, s->state, dlci);

	if (!dlci) {
		rfcomm_send_ua(s, 0);

		if (s->state == BT_OPEN) {
			s->state = BT_CONNECTED;
			rfcomm_process_connect(s);
		}
		return 0;
	}

	/* Check if DLC exists */
	d = rfcomm_dlc_get(s, dlci);
	if (d) {
		if (d->state == BT_OPEN) {
			/* DLC was previously opened by PN request */
			rfcomm_check_accept(d);
		}
		return 0;
	}

	/* Notify socket layer about incoming connection */
	channel = __srv_channel(dlci);
	if (rfcomm_connect_ind(s, channel, &d)) {
		d->dlci = dlci;
		d->addr = __addr(s->initiator, dlci);
		rfcomm_dlc_link(s, d);

		rfcomm_check_accept(d);
	} else {
		rfcomm_send_dm(s, dlci);
	}

	return 0;
}

/**
 * @brief Applies the parameters from a received Parameter Negotiation (PN) frame.
 * @param d The RFCOMM DLC.
 * @param cr The command/response flag.
 * @param pn The PN frame data.
 * @return 0 on success.
 */
static int rfcomm_apply_pn(struct rfcomm_dlc *d, int cr, struct rfcomm_pn *pn)
{
	struct rfcomm_session *s = d->session;

	BT_DBG("dlc %p state %ld dlci %d mtu %d fc 0x%x credits %d",
			d, d->state, d->dlci, pn->mtu, pn->flow_ctrl, pn->credits);

	if ((pn->flow_ctrl == 0xf0 && s->cfc != RFCOMM_CFC_DISABLED) ||
						pn->flow_ctrl == 0xe0) {
		d->cfc = RFCOMM_CFC_ENABLED;
		d->tx_credits = pn->credits;
	} else {
		d->cfc = RFCOMM_CFC_DISABLED;
		set_bit(RFCOMM_TX_THROTTLED, &d->flags);
	}

	if (s->cfc == RFCOMM_CFC_UNKNOWN)
		s->cfc = d->cfc;

	d->priority = pn->priority;

	d->mtu = __le16_to_cpu(pn->mtu);

	if (cr && d->mtu > s->mtu)
		d->mtu = s->mtu;

	return 0;
}
/**
 * @brief Handles a received Parameter Negotiation (PN) frame.
 * @param s The RFCOMM session.
 * @param cr The command/response flag.
 * @param skb The socket buffer containing the PN frame.
 * @return 0 on success.
 */
static int rfcomm_recv_pn(struct rfcomm_session *s, int cr, struct sk_buff *skb)
{
	struct rfcomm_pn *pn = (void *) skb->data;
	struct rfcomm_dlc *d;
	u8 dlci = pn->dlci;

	BT_DBG("session %p state %ld dlci %d", s, s->state, dlci);

	if (!dlci)
		return 0;

	d = rfcomm_dlc_get(s, dlci);
	if (d) {
		if (cr) {
			/* PN request */
			rfcomm_apply_pn(d, cr, pn);
			rfcomm_send_pn(s, 0, d);
		} else {
			/* PN response */
			switch (d->state) {
			case BT_CONFIG:
				rfcomm_apply_pn(d, cr, pn);

				d->state = BT_CONNECT;
				rfcomm_send_sabm(s, d->dlci);
				break;
			}
		}
	} else {
		u8 channel = __srv_channel(dlci);

		if (!cr)
			return 0;

		/* PN request for non existing DLC.
		 * Assume incoming connection. */
		if (rfcomm_connect_ind(s, channel, &d)) {
			d->dlci = dlci;
			d->addr = __addr(s->initiator, dlci);
			rfcomm_dlc_link(s, d);

			rfcomm_apply_pn(d, cr, pn);

			d->state = BT_OPEN;
			rfcomm_send_pn(s, 0, d);
		} else {
			rfcomm_send_dm(s, dlci);
		}
	}
	return 0;
}
/**
 * @brief Handles a received Remote Port Negotiation (RPN) frame.
 * @param s The RFCOMM session.
 * @param cr The command/response flag.
 * @param len The length of the RPN frame.
 * @param skb The socket buffer containing the RPN frame.
 * @return 0 on success.
 */
static int rfcomm_recv_rpn(struct rfcomm_session *s, int cr, int len, struct sk_buff *skb)
{
	struct rfcomm_rpn *rpn = (void *) skb->data;
	u8 dlci = __get_dlci(rpn->dlci);

	u8 bit_rate  = 0;
	u8 data_bits = 0;
	u8 stop_bits = 0;
	u8 parity    = 0;
	u8 flow_ctrl = 0;
	u8 xon_char  = 0;
	u8 xoff_char = 0;
	u16 rpn_mask = RFCOMM_RPN_PM_ALL;

	BT_DBG("dlci %d cr %d len 0x%x bitr 0x%x line 0x%x flow 0x%x xonc 0x%x xoffc 0x%x pm 0x%x",
		dlci, cr, len, rpn->bit_rate, rpn->line_settings, rpn->flow_ctrl,
		rpn->xon_char, rpn->xoff_char, rpn->param_mask);

	if (!cr)
		return 0;

	if (len == 1) {
		/* This is a request, return default (according to ETSI TS 07.10) settings */
		bit_rate  = RFCOMM_RPN_BR_9600;
		data_bits = RFCOMM_RPN_DATA_8;
		stop_bits = RFCOMM_RPN_STOP_1;
		parity    = RFCOMM_RPN_PARITY_NONE;
		flow_ctrl = RFCOMM_RPN_FLOW_NONE;
		xon_char  = RFCOMM_RPN_XON_CHAR;
		xoff_char = RFCOMM_RPN_XOFF_CHAR;
		goto rpn_out;
	}

	/* Check for sane values, ignore/accept bit_rate, 8 bits, 1 stop bit,
	 * no parity, no flow control lines, normal XON/XOFF chars */

	if (rpn->param_mask & cpu_to_le16(RFCOMM_RPN_PM_BITRATE)) {
		bit_rate = rpn->bit_rate;
		if (bit_rate > RFCOMM_RPN_BR_230400) {
			BT_DBG("RPN bit rate mismatch 0x%x", bit_rate);
			bit_rate = RFCOMM_RPN_BR_9600;
			rpn_mask ^= RFCOMM_RPN_PM_BITRATE;
		}
	}

	if (rpn->param_mask & cpu_to_le16(RFCOMM_RPN_PM_DATA)) {
		data_bits = __get_rpn_data_bits(rpn->line_settings);
		if (data_bits != RFCOMM_RPN_DATA_8) {
			BT_DBG("RPN data bits mismatch 0x%x", data_bits);
			data_bits = RFCOMM_RPN_DATA_8;
			rpn_mask ^= RFCOMM_RPN_PM_DATA;
		}
	}

	if (rpn->param_mask & cpu_to_le16(RFCOMM_RPN_PM_STOP)) {
		stop_bits = __get_rpn_stop_bits(rpn->line_settings);
		if (stop_bits != RFCOMM_RPN_STOP_1) {
			BT_DBG("RPN stop bits mismatch 0x%x", stop_bits);
			stop_bits = RFCOMM_RPN_STOP_1;
			rpn_mask ^= RFCOMM_RPN_PM_STOP;
		}
	}

	if (rpn->param_mask & cpu_to_le16(RFCOMM_RPN_PM_PARITY)) {
		parity = __get_rpn_parity(rpn->line_settings);
		if (parity != RFCOMM_RPN_PARITY_NONE) {
			BT_DBG("RPN parity mismatch 0x%x", parity);
			parity = RFCOMM_RPN_PARITY_NONE;
			rpn_mask ^= RFCOMM_RPN_PM_PARITY;
		}
	}

	if (rpn->param_mask & cpu_to_le16(RFCOMM_RPN_PM_FLOW)) {
		flow_ctrl = rpn->flow_ctrl;
		if (flow_ctrl != RFCOMM_RPN_FLOW_NONE) {
			BT_DBG("RPN flow ctrl mismatch 0x%x", flow_ctrl);
			flow_ctrl = RFCOMM_RPN_FLOW_NONE;
			rpn_mask ^= RFCOMM_RPN_PM_FLOW;
		}
	}

	if (rpn->param_mask & cpu_to_le16(RFCOMM_RPN_PM_XON)) {
		xon_char = rpn->xon_char;
		if (xon_char != RFCOMM_RPN_XON_CHAR) {
			BT_DBG("RPN XON char mismatch 0x%x", xon_char);
			xon_char = RFCOMM_RPN_XON_CHAR;
			rpn_mask ^= RFCOMM_RPN_PM_XON;
		}
	}

	if (rpn->param_mask & cpu_to_le16(RFCOMM_RPN_PM_XOFF)) {
		xoff_char = rpn->xoff_char;
		if (xoff_char != RFCOMM_RPN_XOFF_CHAR) {
			BT_DBG("RPN XOFF char mismatch 0x%x", xoff_char);
			xoff_char = RFCOMM_RPN_XOFF_CHAR;
			rpn_mask ^= RFCOMM_RPN_PM_XOFF;
		}
	}

rpn_out:
	rfcomm_send_rpn(s, 0, dlci, bit_rate, data_bits, stop_bits,
			parity, flow_ctrl, xon_char, xoff_char, rpn_mask);

	return 0;
}
/**
 * @brief Handles a received Remote Line Status (RLS) frame.
 * @param s The RFCOMM session.
 * @param cr The command/response flag.
 * @param skb The socket buffer containing the RLS frame.
 * @return 0 on success.
 */
static int rfcomm_recv_rls(struct rfcomm_session *s, int cr, struct sk_buff *skb)
{
	struct rfcomm_rls *rls = (void *) skb->data;
	u8 dlci = __get_dlci(rls->dlci);

	BT_DBG("dlci %d cr %d status 0x%x", dlci, cr, rls->status);

	if (!cr)
		return 0;

	/* We should probably do something with this information here. But
	 * for now it's sufficient just to reply -- Bluetooth 1.1 says it's
	 * mandatory to recognise and respond to RLS */

	rfcomm_send_rls(s, 0, dlci, rls->status);

	return 0;
}
/**
 * @brief Handles a received Modem Status Command (MSC) frame.
 * @param s The RFCOMM session.
 * @param cr The command/response flag.
 * @param skb The socket buffer containing the MSC frame.
 * @return 0 on success.
 */
static int rfcomm_recv_msc(struct rfcomm_session *s, int cr, struct sk_buff *skb)
{
	struct rfcomm_msc *msc = (void *) skb->data;
	struct rfcomm_dlc *d;
	u8 dlci = __get_dlci(msc->dlci);

	BT_DBG("dlci %d cr %d v24 0x%x", dlci, cr, msc->v24_sig);

	d = rfcomm_dlc_get(s, dlci);
	if (!d)
		return 0;

	if (cr) {
		if (msc->v24_sig & RFCOMM_V24_FC && !d->cfc)
			set_bit(RFCOMM_TX_THROTTLED, &d->flags);
		else
			clear_bit(RFCOMM_TX_THROTTLED, &d->flags);

		rfcomm_dlc_lock(d);

		d->remote_v24_sig = msc->v24_sig;

		if (d->modem_status)
			d->modem_status(d, msc->v24_sig);

		rfcomm_dlc_unlock(d);

		rfcomm_send_msc(s, 0, dlci, msc->v24_sig);

		d->mscex |= RFCOMM_MSCEX_RX;
	} else
		d->mscex |= RFCOMM_MSCEX_TX;

	return 0;
}
/**
 * @brief Handles a received Modem Control Command (MCC) frame.
 * @param s The RFCOMM session.
 * @param skb The socket buffer containing the MCC frame.
 * @return 0 on success.
 */
static int rfcomm_recv_mcc(struct rfcomm_session *s, struct sk_buff *skb)
{
	struct rfcomm_mcc *mcc = (void *) skb->data;
	u8 type, cr, len;

	cr   = __test_cr(mcc->type);
	type = __get_mcc_type(mcc->type);
	len  = __get_mcc_len(mcc->len);

	BT_DBG("%p type 0x%x cr %d", s, type, cr);

	skb_pull(skb, 2);

	switch (type) {
	case RFCOMM_PN:
		rfcomm_recv_pn(s, cr, skb);
		break;

	case RFCOMM_RPN:
		rfcomm_recv_rpn(s, cr, len, skb);
		break;

	case RFCOMM_RLS:
		rfcomm_recv_rls(s, cr, skb);
		break;

	case RFCOMM_MSC:
		rfcomm_recv_msc(s, cr, skb);
		break;

	case RFCOMM_FCOFF:
		if (cr) {
			set_bit(RFCOMM_TX_THROTTLED, &s->flags);
			rfcomm_send_fcoff(s, 0);
		}
		break;

	case RFCOMM_FCON:
		if (cr) {
			clear_bit(RFCOMM_TX_THROTTLED, &s->flags);
			rfcomm_send_fcon(s, 0);
		}
		break;

	case RFCOMM_TEST:
		if (cr)
			rfcomm_send_test(s, 0, skb->data, skb->len);
		break;

	case RFCOMM_NSC:
		break;

	default:
		BT_ERR("Unknown control type 0x%02x", type);
		rfcomm_send_nsc(s, cr, type);
		break;
	}
	return 0;
}
/**
 * @brief Handles received data on an RFCOMM DLC.
 * @param s The RFCOMM session.
 * @param dlci The DLCI of the DLC.
 * @param pf The poll/final bit.
 * @param skb The socket buffer containing the data.
 * @return 0 on success.
 */
static int rfcomm_recv_data(struct rfcomm_session *s, u8 dlci, int pf, struct sk_buff *skb)
{
	struct rfcomm_dlc *d;

	BT_DBG("session %p state %ld dlci %d pf %d", s, s->state, dlci, pf);

	d = rfcomm_dlc_get(s, dlci);
	if (!d) {
		rfcomm_send_dm(s, dlci);
		goto drop;
	}

	if (pf && d->cfc) {
		u8 credits = *(u8 *) skb->data; skb_pull(skb, 1);

		d->tx_credits += credits;
		if (d->tx_credits)
			clear_bit(RFCOMM_TX_THROTTLED, &d->flags);
	}

	if (skb->len && d->state == BT_CONNECTED) {
		rfcomm_dlc_lock(d);
		d->rx_credits--;
		d->data_ready(d, skb);
		rfcomm_dlc_unlock(d);
		return 0;
	}

drop:
	kfree_skb(skb);
	return 0;
}
/**
 * @brief Processes a received RFCOMM frame.
 * @param s The RFCOMM session.
 * @param skb The socket buffer containing the frame.
 * @return The session pointer, which may have changed if the session was closed.
 */
static struct rfcomm_session *rfcomm_recv_frame(struct rfcomm_session *s,
						struct sk_buff *skb)
{
	struct rfcomm_hdr *hdr = (void *) skb->data;
	u8 type, dlci, fcs;

	if (!s) {
		/* no session, so free socket data */
		kfree_skb(skb);
		return s;
	}

	dlci = __get_dlci(hdr->addr);
	type = __get_type(hdr->ctrl);

	/* Trim FCS */
	skb->len--; skb->tail--;
	fcs = *(u8 *)skb_tail_pointer(skb);

	if (__check_fcs(skb->data, type, fcs)) {
		BT_ERR("bad checksum in packet");
		kfree_skb(skb);
		return s;
	}

	if (__test_ea(hdr->len))
		skb_pull(skb, 3);
	else
		skb_pull(skb, 4);

	switch (type) {
	case RFCOMM_SABM:
		if (__test_pf(hdr->ctrl))
			rfcomm_recv_sabm(s, dlci);
		break;

	case RFCOMM_DISC:
		if (__test_pf(hdr->ctrl))
			s = rfcomm_recv_disc(s, dlci);
		break;

	case RFCOMM_UA:
		if (__test_pf(hdr->ctrl))
			s = rfcomm_recv_ua(s, dlci);
		break;

	case RFCOMM_DM:
		s = rfcomm_recv_dm(s, dlci);
		break;

	case RFCOMM_UIH:
		if (dlci) {
			rfcomm_recv_data(s, dlci, __test_pf(hdr->ctrl), skb);
			return s;
		}
		rfcomm_recv_mcc(s, skb);
		break;

	default:
		BT_ERR("Unknown packet type 0x%02x", type);
		break;
	}
	kfree_skb(skb);
	return s;
}

/* ---- Connection and data processing ---- */

/**
 * @brief Processes the connection of an RFCOMM session.
 * @param s The RFCOMM session.
 *
 * This function is called when the underlying L2CAP connection is established.
 * It iterates through the DLCs in the session and initiates parameter
 * negotiation for those that are in the BT_CONFIG state.
 */
static void rfcomm_process_connect(struct rfcomm_session *s)
{
	struct rfcomm_dlc *d, *n;

	BT_DBG("session %p state %ld", s, s->state);

	list_for_each_entry_safe(d, n, &s->dlcs, list) {
		if (d->state == BT_CONFIG) {
			d->mtu = s->mtu;
			if (rfcomm_check_security(d)) {
				rfcomm_send_pn(s, 1, d);
			} else {
				set_bit(RFCOMM_AUTH_PENDING, &d->flags);
				rfcomm_dlc_set_timer(d, RFCOMM_AUTH_TIMEOUT);
			}
		}
	}
}
/**
 * @brief Processes the transmit queue for an RFCOMM DLC.
 * @param d The RFCOMM DLC.
 * @return The number of frames left in the transmit queue.
 */
static int rfcomm_process_tx(struct rfcomm_dlc *d)
{
	struct sk_buff *skb;
	int err;

	BT_DBG("dlc %p state %ld cfc %d rx_credits %d tx_credits %d",
			d, d->state, d->cfc, d->rx_credits, d->tx_credits);

	/* Send pending MSC */
	if (test_and_clear_bit(RFCOMM_MSC_PENDING, &d->flags))
		rfcomm_send_msc(d->session, 1, d->dlci, d->v24_sig);

	if (d->cfc) {
		/* CFC enabled.
		 * Give them some credits */
		if (!test_bit(RFCOMM_RX_THROTTLED, &d->flags) &&
				d->rx_credits <= (d->cfc >> 2)) {
			rfcomm_send_credits(d->session, d->addr, d->cfc - d->rx_credits);
			d->rx_credits = d->cfc;
		}
	} else {
		/* CFC disabled.
		 * Give ourselves some credits */
		d->tx_credits = 5;
	}

	if (test_bit(RFCOMM_TX_THROTTLED, &d->flags))
		return skb_queue_len(&d->tx_queue);

	while (d->tx_credits && (skb = skb_dequeue(&d->tx_queue))) {
		err = rfcomm_send_frame(d->session, skb->data, skb->len);
		if (err < 0) {
			skb_queue_head(&d->tx_queue, skb);
			break;
		}
		kfree_skb(skb);
		d->tx_credits--;
	}

	if (d->cfc && !d->tx_credits) {
		/* We're out of TX credits.
		 * Set TX_THROTTLED flag to avoid unnesary wakeups by dlc_send. */
		set_bit(RFCOMM_TX_THROTTLED, &d->flags);
	}

	return skb_queue_len(&d->tx_queue);
}
/**
 * @brief Processes all DLCs in an RFCOMM session.
 * @param s The RFCOMM session.
 */
static void rfcomm_process_dlcs(struct rfcomm_session *s)
{
	struct rfcomm_dlc *d, *n;

	BT_DBG("session %p state %ld", s, s->state);

	list_for_each_entry_safe(d, n, &s->dlcs, list) {
		if (test_bit(RFCOMM_TIMED_OUT, &d->flags)) {
			__rfcomm_dlc_close(d, ETIMEDOUT);
			continue;
		}

		if (test_bit(RFCOMM_ENC_DROP, &d->flags)) {
			__rfcomm_dlc_close(d, ECONNREFUSED);
			continue;
		}

		if (test_and_clear_bit(RFCOMM_AUTH_ACCEPT, &d->flags)) {
			rfcomm_dlc_clear_timer(d);
			if (d->out) {
				rfcomm_send_pn(s, 1, d);
				rfcomm_dlc_set_timer(d, RFCOMM_CONN_TIMEOUT);
			} else {
				if (d->defer_setup) {
					set_bit(RFCOMM_DEFER_SETUP, &d->flags);
					rfcomm_dlc_set_timer(d, RFCOMM_AUTH_TIMEOUT);

					rfcomm_dlc_lock(d);
					d->state = BT_CONNECT2;
					d->state_change(d, 0);
					rfcomm_dlc_unlock(d);
				} else
					rfcomm_dlc_accept(d);
			}
			continue;
		} else if (test_and_clear_bit(RFCOMM_AUTH_REJECT, &d->flags)) {
			rfcomm_dlc_clear_timer(d);
			if (!d->out)
				rfcomm_send_dm(s, d->dlci);
			else
				d->state = BT_CLOSED;
			__rfcomm_dlc_close(d, ECONNREFUSED);
			continue;
		}

		if (test_bit(RFCOMM_SEC_PENDING, &d->flags))
			continue;

		if (test_bit(RFCOMM_TX_THROTTLED, &s->flags))
			continue;

		if ((d->state == BT_CONNECTED || d->state == BT_DISCONN) &&
						d->mscex == RFCOMM_MSCEX_OK)
			rfcomm_process_tx(d);
	}
}
/**
 * @brief Processes the receive queue for an RFCOMM session.
 * @param s The RFCOMM session.
 * @return The session pointer, which may have changed if the session was closed.
 */
static struct rfcomm_session *rfcomm_process_rx(struct rfcomm_session *s)
{
	struct socket *sock = s->sock;
	struct sock *sk = sock->sk;
	struct sk_buff *skb;

	BT_DBG("session %p state %ld qlen %d", s, s->state, skb_queue_len(&sk->sk_receive_queue));

	/* Get data directly from socket receive queue without copying it. */
	while ((skb = skb_dequeue(&sk->sk_receive_queue))) {
		skb_orphan(skb);
		if (!skb_linearize(skb) && sk->sk_state != BT_CLOSED) {
			s = rfcomm_recv_frame(s, skb);
			if (!s)
				break;
		} else {
			kfree_skb(skb);
		}
	}

	if (s && (sk->sk_state == BT_CLOSED))
		s = rfcomm_session_close(s, sk->sk_err);

	return s;
}

/**
 * @brief Accepts a new incoming L2CAP connection for RFCOMM.
 * @param s The listening RFCOMM session.
 */
static void rfcomm_accept_connection(struct rfcomm_session *s)
{
	struct socket *sock = s->sock, *nsock;
	int err;

	/* Fast check for a new connection.
	 * Avoids unnesesary socket allocations. */
	if (list_empty(&bt_sk(sock->sk)->accept_q))
		return;

	BT_DBG("session %p", s);

	err = kernel_accept(sock, &nsock, O_NONBLOCK);
	if (err < 0)
		return;

	/* Set our callbacks */
	nsock->sk->sk_data_ready   = rfcomm_l2data_ready;
	nsock->sk->sk_state_change = rfcomm_l2state_change;

	s = rfcomm_session_add(nsock, BT_OPEN);
	if (s) {
		/* We should adjust MTU on incoming sessions.
		 * L2CAP MTU minus UIH header and FCS. */
		s->mtu = min(l2cap_pi(nsock->sk)->chan->omtu,
				l2cap_pi(nsock->sk)->chan->imtu) - 5;

		rfcomm_schedule();
	} else
		sock_release(nsock);
}
/**
 * @brief Checks the state of the underlying L2CAP connection.
 * @param s The RFCOMM session.
 * @return The session pointer, which may have changed if the session was closed.
 */
static struct rfcomm_session *rfcomm_check_connection(struct rfcomm_session *s)
{
	struct sock *sk = s->sock->sk;

	BT_DBG("%p state %ld", s, s->state);

	switch (sk->sk_state) {
	case BT_CONNECTED:
		s->state = BT_CONNECT;
		rfcomm_schedule();
		break;
	case BT_CLOSED:
		s = rfcomm_session_close(s, sk->sk_err);
		break;
	}
	return s;
}

/**
 * @brief The main loop for the RFCOMM worker thread.
 * @param data Unused.
 * @return 0 on exit.
 */
static int rfcomm_run(void *data)
{
	struct rfcomm_session *s, *n;

	BT_DBG("thread started");
	set_user_nice(current, -15);

	while (!kthread_should_stop()) {
		rfcomm_lock();

		list_for_each_entry_safe(s, n, &session_list, list) {
			if (test_and_clear_bit(RFCOMM_TIMED_OUT, &s->flags)) {
				rfcomm_session_close(s, ETIMEDOUT);
				continue;
			}

			switch (s->state) {
			case BT_LISTEN:
				rfcomm_accept_connection(s);
				break;
			case BT_BOUND:
				s = rfcomm_check_connection(s);
				break;
			case BT_CONNECT:
			case BT_CONNECTED:
				s = rfcomm_process_rx(s);
				if (s)
					rfcomm_process_dlcs(s);
				break;
			}
		}

		rfcomm_unlock();

		wait_event_interruptible(rfcomm_wq,
				kthread_should_stop() || !list_empty(&session_list));
	}

	BT_DBG("thread stopped");
	return 0;
}
/**
 * @brief Initializes the RFCOMM core.
 * @return 0 on success, or a negative error code on failure.
 */
int __init rfcomm_init(void)
{
	int err;

	rfcomm_thread = kthread_run(rfcomm_run, NULL, "krfcommd");
	if (IS_ERR(rfcomm_thread))
		return PTR_ERR(rfcomm_thread);

	err = rfcomm_init_ttys();
	if (err < 0) {
		kthread_stop(rfcomm_thread);
		return err;
	}

	err = rfcomm_init_sockets();
	if (err < 0) {
		rfcomm_cleanup_ttys();
		kthread_stop(rfcomm_thread);
		return err;
	}

	BT_INFO("RFCOMM ver %s", VERSION);
	return 0;
}
/**
 * @brief Cleans up the RFCOMM core.
 */
void rfcomm_exit(void)
{
	kthread_stop(rfcomm_thread);
	rfcomm_cleanup_sockets();
	rfcomm_cleanup_ttys();
}

MODULE_AUTHOR("Maxim Krasnyansky <maxk@qualcomm.com>");
MODULE_DESCRIPTION("Bluetooth RFCOMM ver " VERSION);
MODULE_VERSION(VERSION);
MODULE_LICENSE("GPL");
MODULE_ALIAS("bt-proto-3");
module_param(disable_cfc, bool, 0644);
MODULE_PARM_DESC(disable_cfc, "Disable credit based flow control");
module_param(l2cap_ertm, bool, 0644);
MODULE_PARM_DESC(l2cap_ertm, "Enable L2CAP ERTM");
module_param(channel_mtu, int, 0644);
MODULE_PARM_DESC(channel_mtu, "Set initial channel MTU");