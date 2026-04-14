// SPDX-License-Identifier: GPL-2.0+
// Copyright (c) 2021 Hisilicon Limited.

/**
 * @file hclge_ptp.c
 * @brief Precision Time Protocol (PTP) hardware clock support for Hisilicon
 * hns3 network driver.
 *
 * This file implements the PTP Hardware Clock (PHC) interface, allowing the
 * hns3 driver to interact with the kernel's PTP subsystem. It provides
 * functionalities for reading and adjusting the hardware clock, managing
 * hardware timestamps for transmitted and received packets, and handling
 * PTP configuration via ethtool and ioctl.
 */

#include <linux/skbuff.h>
#include <linux/string_choices.h>
#include "hclge_main.h"
#include "hnae3.h"

/**
 * @brief Reads the hardware clock cycle parameters from registers.
 * @param hdev The HCLGE device structure.
 * @return 0 on success, or -EINVAL if the cycle denominator is invalid.
 *
 * Functional Utility: This function retrieves the hardware-specific clock
 * cycle configuration, which is composed of a quotient, numerator, and
 * denominator. This configuration is essential for frequency adjustments.
 */
static int hclge_ptp_get_cycle(struct hclge_dev *hdev)
{
	struct hclge_ptp *ptp = hdev->ptp;

	/*
	 * Hardware Interaction: Reads the cycle quotient, numerator, and
	 * denominator from their respective PTP registers. These values define
	 * the base frequency of the hardware clock.
	 */
	ptp->cycle.quo = readl(hdev->ptp->io_base + HCLGE_PTP_CYCLE_QUO_REG) &
			 HCLGE_PTP_CYCLE_QUO_MASK;
	ptp->cycle.numer = readl(hdev->ptp->io_base + HCLGE_PTP_CYCLE_NUM_REG);
	ptp->cycle.den = readl(hdev->ptp->io_base + HCLGE_PTP_CYCLE_DEN_REG);

	/* Pre-condition: The denominator must not be zero to avoid division errors. */
	if (ptp->cycle.den == 0) {
		dev_err(&hdev->pdev->dev, "invalid ptp cycle denominator!
");
		return -EINVAL;
	}

	return 0;
}

/**
 * @brief Adjusts the hardware clock frequency.
 * @param ptp The PTP clock info structure.
 * @param scaled_ppm The frequency adjustment in scaled parts-per-million.
 * @return 0 on success.
 *
 * Algorithm: This function implements the adjfine callback for the PTP clock.
 * It calculates a new cycle value based on the requested scaled_ppm and
 * reprograms the hardware clock cycle registers to adjust its frequency.
 */
static int hclge_ptp_adjfine(struct ptp_clock_info *ptp, long scaled_ppm)
{
	struct hclge_dev *hdev = hclge_ptp_get_hdev(ptp);
	struct hclge_ptp_cycle *cycle = &hdev->ptp->cycle;
	u64 adj_val, adj_base;
	unsigned long flags;
	u32 quo, numerator;

	/*
	 * Block Logic: The base adjustment value is calculated from the current
	 * cycle configuration. This value is then adjusted using the scaled_ppm
	 * provided by the PTP core.
	 */
	adj_base = (u64)cycle->quo * (u64)cycle->den + (u64)cycle->numer;
	adj_val = adjust_by_scaled_ppm(adj_base, scaled_ppm);

	/*
	 * Block Logic: The adjusted value is converted back into the hardware's
	 * quotient and numerator format based on the fixed denominator. This
	 * represents the new clock cycle duration.
	 * For example, for 2.5ns, quotient is 2, denominator is fixed, and
	 * numerator is 0.5 * ptp->cycle.den.
	 */
	quo = div_u64_rem(adj_val, cycle->den, &numerator);

	/*
	 * Hardware Interaction: The new cycle values are written to hardware
	 * registers under a spinlock to ensure atomicity. The cycle adjustment
	 * enable bit is then set to apply the new frequency.
	 */
	spin_lock_irqsave(&hdev->ptp->lock, flags);
	writel(quo & HCLGE_PTP_CYCLE_QUO_MASK,
	       hdev->ptp->io_base + HCLGE_PTP_CYCLE_QUO_REG);
	writel(numerator, hdev->ptp->io_base + HCLGE_PTP_CYCLE_NUM_REG);
	writel(cycle->den, hdev->ptp->io_base + HCLGE_PTP_CYCLE_DEN_REG);
	writel(HCLGE_PTP_CYCLE_ADJ_EN,
	       hdev->ptp->io_base + HCLGE_PTP_CYCLE_CFG_REG);
	spin_unlock_irqrestore(&hdev->ptp->lock, flags);

	return 0;
}

/**
 * @brief Prepares for transmitting a PTP packet and capturing its timestamp.
 * @param handle The HNAE3 handle.
 * @param skb The socket buffer being transmitted.
 * @return true if the packet is accepted for TX timestamping, false otherwise.
 *
 * Functional Utility: This function is called by the transmit path for packets
 * requiring a hardware timestamp. It acts as a gatekeeper, ensuring that TX
 * timestamping is enabled and that only one packet is being handled at a time.
 */
bool hclge_ptp_set_tx_info(struct hnae3_handle *handle, struct sk_buff *skb)
{
	struct hclge_vport *vport = hclge_get_vport(handle);
	struct hclge_dev *hdev = vport->back;
	struct hclge_ptp *ptp = hdev->ptp;

	if (!ptp)
		return false;

	/*
	 * Pre-condition: Check if PTP TX timestamping is enabled and if another
	 * packet is already pending a timestamp. If either condition fails,
	 * the packet is skipped. This prevents race conditions and resource
	 * conflicts.
	 */
	if (!test_bit(HCLGE_PTP_FLAG_TX_EN, &ptp->flags) ||
	    test_and_set_bit(HCLGE_STATE_PTP_TX_HANDLING, &hdev->state)) {
		ptp->tx_skipped++;
		return false;
	}

	/*
	 * Block Logic: If the packet is accepted, store a reference to the skb.
	 * This skb will be used later in hclge_ptp_clean_tx_hwts to deliver the
	 * captured timestamp to the networking stack.
	 */
	ptp->tx_start = jiffies;
	ptp->tx_skb = skb_get(skb);
	ptp->tx_cnt++;

	return true;
}

/**
 * @brief Retrieves the TX timestamp from hardware and completes the TX process.
 * @param hdev The HCLGE device structure.
 *
 * Functional Utility: This function is typically called from an interrupt
 * context after a PTP packet has been transmitted. It reads the timestamp
 * from the hardware registers, applies it to the saved skb, and releases
 * the resources.
 */
void hclge_ptp_clean_tx_hwts(struct hclge_dev *hdev)
{
	struct sk_buff *skb = hdev->ptp->tx_skb;
	struct skb_shared_hwtstamps hwts;
	u32 hi, lo;
	u64 ns;

	/*
	 * Hardware Interaction: Reads the high/low seconds and nanoseconds
	 * of the captured transmit timestamp from the hardware registers.
	 */
	ns = readl(hdev->ptp->io_base + HCLGE_PTP_TX_TS_NSEC_REG) &
	     HCLGE_PTP_TX_TS_NSEC_MASK;
	lo = readl(hdev->ptp->io_base + HCLGE_PTP_TX_TS_SEC_L_REG);
	hi = readl(hdev->ptp->io_base + HCLGE_PTP_TX_TS_SEC_H_REG) &
	     HCLGE_PTP_TX_TS_SEC_H_MASK;
	hdev->ptp->last_tx_seqid = readl(hdev->ptp->io_base +
		HCLGE_PTP_TX_TS_SEQID_REG);

	/* Block Logic: If there is a pending skb, process the timestamp. */
	if (skb) {
		hdev->ptp->tx_skb = NULL;
		hdev->ptp->tx_cleaned++;

		/*
		 * Block Logic: Combine the seconds and nanoseconds parts into a
		 * single 64-bit nanosecond value, convert it to ktime, and pass
		 * it to the networking stack using skb_tstamp_tx.
		 */
		ns += (((u64)hi) << 32 | lo) * NSEC_PER_SEC;
		hwts.hwtstamp = ns_to_ktime(ns);
		skb_tstamp_tx(skb, &hwts);
		dev_kfree_skb_any(skb);
	}

	/* Invariant: Clear the handling flag to allow the next TX timestamp. */
	clear_bit(HCLGE_STATE_PTP_TX_HANDLING, &hdev->state);
}

/**
 * @brief Retrieves the RX timestamp and attaches it to the skb.
 * @param handle The HNAE3 handle.
 * @param skb The received socket buffer.
 * @param nsec The nanoseconds part of the timestamp (from descriptor).
 * @param sec The lower 32 bits of the seconds part (from descriptor).
 *
 * Functional Utility: This function is called from the receive path when a
 * PTP packet is received. It reconstructs the full 64-bit timestamp from the
 * partial information in the Rx descriptor and the upper bits from a hardware
 * register.
 */
void hclge_ptp_get_rx_hwts(struct hnae3_handle *handle, struct sk_buff *skb,
			   u32 nsec, u32 sec)
{
	struct hclge_vport *vport = hclge_get_vport(handle);
	struct hclge_dev *hdev = vport->back;
	unsigned long flags;
	u64 ns = nsec;
	u32 sec_h;

	/* Pre-condition: Check if PTP and RX timestamping are enabled. */
	if (!hdev->ptp || !test_bit(HCLGE_PTP_FLAG_RX_EN, &hdev->ptp->flags))
		return;

	/*
	 * Block Logic: The Rx descriptor contains only the lower bits of the
	 * seconds counter. The upper bits, which change infrequently, are read
	 * from a shared hardware register to reconstruct the full timestamp.
	 * A spinlock protects access to this shared register.
	 */
	spin_lock_irqsave(&hdev->ptp->lock, flags);
	sec_h = readl(hdev->ptp->io_base + HCLGE_PTP_CUR_TIME_SEC_H_REG);
	spin_unlock_irqrestore(&hdev->ptp->lock, flags);

	/*
	 * Block Logic: Combine the high/low seconds and nanoseconds into a
	 * single 64-bit nanosecond value and attach it to the skb.
	 */
	ns += (((u64)sec_h) << HCLGE_PTP_SEC_H_OFFSET | sec) * NSEC_PER_SEC;
	skb_hwtstamps(skb)->hwtstamp = ns_to_ktime(ns);
	hdev->ptp->last_rx = jiffies;
	hdev->ptp->rx_cnt++;
}

/**
 * @brief Reads the current time from the hardware clock.
 * @param ptp The PTP clock info structure.
 * @param ts Pointer to a timespec64 to store the current time.
 * @param sts Pointer to a ptp_system_timestamp for system/PHC correlation.
 * @return 0 on success.
 *
 * Functional Utility: Implements the gettimex64 callback. It reads the current
 * nanoseconds and seconds from the hardware registers atomically and converts
 * them into a timespec64.
 */
static int hclge_ptp_gettimex(struct ptp_clock_info *ptp, struct timespec64 *ts,
			      struct ptp_system_timestamp *sts)
{
	struct hclge_dev *hdev = hclge_ptp_get_hdev(ptp);
	unsigned long flags;
	u32 hi, lo;
	u64 ns;

	/*
	 * Hardware Interaction: Reads the current time registers under a spinlock
	 * to ensure a consistent snapshot of the time.
	 */
	spin_lock_irqsave(&hdev->ptp->lock, flags);
	ns = readl(hdev->ptp->io_base + HCLGE_PTP_CUR_TIME_NSEC_REG);
	hi = readl(hdev->ptp->io_base + HCLGE_PTP_CUR_TIME_SEC_H_REG);
	lo = readl(hdev->ptp->io_base + HCLGE_PTP_CUR_TIME_SEC_L_REG);
	spin_unlock_irqrestore(&hdev->ptp->lock, flags);

	/* Block Logic: Combine register values into nanoseconds and convert to timespec64. */
	ns += (((u64)hi) << HCLGE_PTP_SEC_H_OFFSET | lo) * NSEC_PER_SEC;
	*ts = ns_to_timespec64(ns);

	return 0;
}

/**
 * @brief Sets the hardware clock to a specific time.
 * @param ptp The PTP clock info structure.
 * @param ts The new time to set.
 * @return 0 on success.
 *
 * Functional Utility: Implements the settime64 callback. It writes the given
 * time into the hardware's time registers to perform a hard reset of the clock.
 */
static int hclge_ptp_settime(struct ptp_clock_info *ptp,
			     const struct timespec64 *ts)
{
	struct hclge_dev *hdev = hclge_ptp_get_hdev(ptp);
	unsigned long flags;

	/*
	 * Hardware Interaction: Programs the hardware clock with the new time value
	 * and issues a time synchronization command to apply it. This is done
	 * atomically under a spinlock.
	 */
	spin_lock_irqsave(&hdev->ptp->lock, flags);
	writel(ts->tv_nsec, hdev->ptp->io_base + HCLGE_PTP_TIME_NSEC_REG);
	writel(ts->tv_sec >> HCLGE_PTP_SEC_H_OFFSET,
	       hdev->ptp->io_base + HCLGE_PTP_TIME_SEC_H_REG);
	writel(ts->tv_sec & HCLGE_PTP_SEC_L_MASK,
	       hdev->ptp->io_base + HCLGE_PTP_TIME_SEC_L_REG);
	/* synchronize the time of phc */
	writel(HCLGE_PTP_TIME_SYNC_EN,
	       hdev->ptp->io_base + HCLGE_PTP_TIME_SYNC_REG);
	spin_unlock_irqrestore(&hdev->ptp->lock, flags);

	return 0;
}

/**
 * @brief Adjusts the hardware clock by a specific delta (phase adjustment).
 * @param ptp The PTP clock info structure.
 * @param delta The time adjustment in nanoseconds.
 * @return 0 on success.
 *
 * Functional Utility: Implements the adjtime callback. For small adjustments,
 * it uses a hardware feature to smoothly adjust the clock phase. For large
 * adjustments, it falls back to a hard reset of the clock time (settime).
 */
static int hclge_ptp_adjtime(struct ptp_clock_info *ptp, s64 delta)
{
	struct hclge_dev *hdev = hclge_ptp_get_hdev(ptp);
	unsigned long flags;
	bool is_neg = false;
	u32 adj_val = 0;

	/* Block Logic: Determine if the adjustment is positive or negative. */
	if (delta < 0) {
		adj_val |= HCLGE_PTP_TIME_NSEC_NEG;
		delta = -delta;
		is_neg = true;
	}

	/*
	 * Block Logic: If the delta is too large for the hardware's fine-grained
	 * adjustment register, perform a full time reset by reading the current
	 * time, adding the delta, and setting the new time.
	 */
	if (delta > HCLGE_PTP_TIME_NSEC_MASK) {
		struct timespec64 ts;
		s64 ns;

		hclge_ptp_gettimex(ptp, &ts, NULL);
		ns = timespec64_to_ns(&ts);
		ns = is_neg ? ns - delta : ns + delta;
		ts = ns_to_timespec64(ns);
		return hclge_ptp_settime(ptp, &ts);
	}

	adj_val |= delta & HCLGE_PTP_TIME_NSEC_MASK;

	/*
	 * Hardware Interaction: For small deltas, write the adjustment value
	 * to the hardware and trigger the time adjustment mechanism.
	 */
	spin_lock_irqsave(&hdev->ptp->lock, flags);
	writel(adj_val, hdev->ptp->io_base + HCLGE_PTP_TIME_NSEC_REG);
	writel(HCLGE_PTP_TIME_ADJ_EN,
	       hdev->ptp->io_base + HCLGE_PTP_TIME_ADJ_REG);
	spin_unlock_irqrestore(&hdev->ptp->lock, flags);

	return 0;
}

/**
 * @brief Retrieves the current hardware timestamping configuration.
 * @param hdev The HCLGE device structure.
 * @param ifr The interface request structure from the ioctl.
 * @return 0 on success, or a negative error code on failure.
 *
 * Functional Utility: Handles the SIOCGHWTSTAMP ioctl by copying the currently
 * active hwtstamp_config to userspace.
 */
int hclge_ptp_get_cfg(struct hclge_dev *hdev, struct ifreq *ifr)
{
	/* Pre-condition: PTP must be supported and enabled. */
	if (!test_bit(HCLGE_STATE_PTP_EN, &hdev->state))
		return -EOPNOTSUPP;

	return copy_to_user(ifr->ifr_data, &hdev->ptp->ts_cfg,
		sizeof(struct hwtstamp_config)) ? -EFAULT : 0;
}

/**
 * @brief Enables or disables PTP-related interrupts in the hardware.
 * @param hdev The HCLGE device structure.
 * @param en True to enable, false to disable.
 * @return 0 on success, or a negative error code on command failure.
 *
 * Functional Utility: Sends a command to the device firmware to control the
 * generation of PTP interrupts (e.g., for TX timestamp events).
 */
static int hclge_ptp_int_en(struct hclge_dev *hdev, bool en)
{
	struct hclge_ptp_int_cmd *req;
	struct hclge_desc desc;
	int ret;

	req = (struct hclge_ptp_int_cmd *)desc.data;

	/* Block Logic: Construct a firmware command to enable/disable PTP interrupts. */
	hclge_cmd_setup_basic_desc(&desc, HCLGE_OPC_PTP_INT_EN, false);
	req->int_en = en ? 1 : 0;

	ret = hclge_cmd_send(&hdev->hw, &desc, 1);
	if (ret)
		dev_err(&hdev->pdev->dev,
			"failed to %s ptp interrupt, ret = %d
",
			str_enable_disable(en), ret);

	return ret;
}

/**
 * @brief Queries the current PTP mode configuration from the firmware.
 * @param hdev The HCLGE device structure.
 * @param cfg Pointer to store the returned configuration bitmap.
 * @return 0 on success, or a negative error code on failure.
 *
 * Functional Utility: Sends a command to the firmware to read the current
 * PTP configuration, which includes settings like TX/RX enable and filtering modes.
 */
int hclge_ptp_cfg_qry(struct hclge_dev *hdev, u32 *cfg)
{
	struct hclge_ptp_cfg_cmd *req;
	struct hclge_desc desc;
	int ret;

	req = (struct hclge_ptp_cfg_cmd *)desc.data;

	/* Block Logic: Construct and send a command to query PTP configuration. */
	hclge_cmd_setup_basic_desc(&desc, HCLGE_OPC_PTP_MODE_CFG, true);
	ret = hclge_cmd_send(&hdev->hw, &desc, 1);
	if (ret) {
		dev_err(&hdev->pdev->dev,
			"failed to query ptp config, ret = %d
", ret);
		return ret;
	}

	*cfg = le32_to_cpu(req->cfg);

	return 0;
}

/**
 * @brief Writes a new PTP mode configuration to the firmware.
 * @param hdev The HCLGE device structure.
 * @param cfg The configuration bitmap to write.
 * @return 0 on success, or a negative error code on failure.
 *
 * Functional Utility: Sends a command to the firmware to apply a new PTP
 * configuration.
 */
static int hclge_ptp_cfg(struct hclge_dev *hdev, u32 cfg)
{
	struct hclge_ptp_cfg_cmd *req;
	struct hclge_desc desc;
	int ret;

	req = (struct hclge_ptp_cfg_cmd *)desc.data;
	hclge_cmd_setup_basic_desc(&desc, HCLGE_OPC_PTP_MODE_CFG, false);
	req->cfg = cpu_to_le32(cfg);
	ret = hclge_cmd_send(&hdev->hw, &desc, 1);
	if (ret)
		dev_err(&hdev->pdev->dev,
			"failed to config ptp, ret = %d
", ret);

	return ret;
}

/**
 * @brief Parses and applies the transmit timestamping mode from a config struct.
 * @param cfg The user-provided hwtstamp_config.
 * @param flags Pointer to the driver's internal PTP flags.
 * @param ptp_cfg Pointer to the hardware configuration bitmap being built.
 * @return 0 on success, or -ERANGE for an unsupported mode.
 */
static int hclge_ptp_set_tx_mode(struct hwtstamp_config *cfg,
				 unsigned long *flags, u32 *ptp_cfg)
{
	switch (cfg->tx_type) {
	case HWTSTAMP_TX_OFF:
		/* Block Logic: Disable transmit timestamping. */
		clear_bit(HCLGE_PTP_FLAG_TX_EN, flags);
		break;
	case HWTSTAMP_TX_ON:
		/* Block Logic: Enable transmit timestamping for all packets. */
		set_bit(HCLGE_PTP_FLAG_TX_EN, flags);
		*ptp_cfg |= HCLGE_PTP_TX_EN_B;
		break;
	default:
		return -ERANGE;
	}

	return 0;
}

/**
 * @brief Parses and applies the receive timestamping mode from a config struct.
 * @param cfg The user-provided hwtstamp_config.
 * @param flags Pointer to the driver's internal PTP flags.
 * @param ptp_cfg Pointer to the hardware configuration bitmap being built.
 * @return 0 on success, or -ERANGE for an unsupported mode.
 */
static int hclge_ptp_set_rx_mode(struct hwtstamp_config *cfg,
				 unsigned long *flags, u32 *ptp_cfg)
{
	int rx_filter = cfg->rx_filter;

	switch (cfg->rx_filter) {
	case HWTSTAMP_FILTER_NONE:
		/* Block Logic: Disable receive timestamping. */
		clear_bit(HCLGE_PTP_FLAG_RX_EN, flags);
		break;
	/* Block Logic: Cases for PTPv1 events. */
	case HWTSTAMP_FILTER_PTP_V1_L4_SYNC:
	case HWTSTAMP_FILTER_PTP_V1_L4_DELAY_REQ:
	case HWTSTAMP_FILTER_PTP_V1_L4_EVENT:
		set_bit(HCLGE_PTP_FLAG_RX_EN, flags);
		*ptp_cfg |= HCLGE_PTP_RX_EN_B;
		*ptp_cfg |= HCLGE_PTP_UDP_FULL_TYPE << HCLGE_PTP_UDP_EN_SHIFT;
		rx_filter = HWTSTAMP_FILTER_PTP_V1_L4_EVENT;
		break;
	/* Block Logic: Cases for PTPv2 events over L2 and L4. */
	case HWTSTAMP_FILTER_PTP_V2_EVENT:
	case HWTSTAMP_FILTER_PTP_V2_L4_EVENT:
	case HWTSTAMP_FILTER_PTP_V2_SYNC:
	case HWTSTAMP_FILTER_PTP_V2_L4_SYNC:
	case HWTSTAMP_FILTER_PTP_V2_DELAY_REQ:
	case HWTSTAMP_FILTER_PTP_V2_L4_DELAY_REQ:
	case HWTSTAMP_FILTER_PTP_V2_L2_EVENT:
	case HWTSTAMP_FILTER_PTP_V2_L2_SYNC:
	case HWTSTAMP_FILTER_PTP_V2_L2_DELAY_REQ:
		set_bit(HCLGE_PTP_FLAG_RX_EN, flags);
		*ptp_cfg |= HCLGE_PTP_RX_EN_B;
		*ptp_cfg |= HCLGE_PTP_UDP_FULL_TYPE << HCLGE_PTP_UDP_EN_SHIFT;
		*ptp_cfg |= HCLGE_PTP_MSG1_V2_DEFAULT << HCLGE_PTP_MSG1_SHIFT;
		*ptp_cfg |= HCLGE_PTP_MSG0_V2_EVENT << HCLGE_PTP_MSG0_SHIFT;
		*ptp_cfg |= HCLGE_PTP_MSG_TYPE_V2 << HCLGE_PTP_MSG_TYPE_SHIFT;
		/* Invariant: Normalize specific filters to the generic event filter. */
		rx_filter = HWTSTAMP_FILTER_PTP_V2_EVENT;
		break;
	case HWTSTAMP_FILTER_ALL:
	default:
		return -ERANGE;
	}

	cfg->rx_filter = rx_filter;

	return 0;
}

/**
 * @brief Configures the hardware for a specific timestamping mode.
 * @param hdev The HCLGE device structure.
 * @param cfg The desired timestamping configuration.
 * @return 0 on success, or a negative error code on failure.
 *
 * Functional Utility: This is the main function for setting the timestamping
 * mode. It parses both TX and RX settings, constructs a hardware configuration
 * value, and applies it by sending a command to the firmware.
 */
static int hclge_ptp_set_ts_mode(struct hclge_dev *hdev,
				 struct hwtstamp_config *cfg)
{
	unsigned long flags = hdev->ptp->flags;
	u32 ptp_cfg = 0;
	int ret;

	if (test_bit(HCLGE_PTP_FLAG_EN, &hdev->ptp->flags))
		ptp_cfg |= HCLGE_PTP_EN_B;

	/* Block Logic: Set TX and RX modes based on the config. */
	ret = hclge_ptp_set_tx_mode(cfg, &flags, &ptp_cfg);
	if (ret)
		return ret;

	ret = hclge_ptp_set_rx_mode(cfg, &flags, &ptp_cfg);
	if (ret)
		return ret;

	/* Hardware Interaction: Apply the new configuration to the hardware. */
	ret = hclge_ptp_cfg(hdev, ptp_cfg);
	if (ret)
		return ret;

	hdev->ptp->flags = flags;
	hdev->ptp->ptp_cfg = ptp_cfg;

	return 0;
}

/**
 * @brief Sets the hardware timestamping configuration from an ioctl.
 * @param hdev The HCLGE device structure.
 * @param ifr The interface request structure from the ioctl.
 * @return 0 on success, or a negative error code on failure.
 *
 * Functional Utility: Handles the SIOCSHWTSTAMP ioctl. It copies the desired
 * config from userspace, applies it to the hardware, and copies the
 * (potentially modified) config back to userspace.
 */
int hclge_ptp_set_cfg(struct hclge_dev *hdev, struct ifreq *ifr)
{
	struct hwtstamp_config cfg;
	int ret;

	if (!test_bit(HCLGE_STATE_PTP_EN, &hdev->state)) {
		dev_err(&hdev->pdev->dev, "phc is unsupported
");
		return -EOPNOTSUPP;
	}

	if (copy_from_user(&cfg, ifr->ifr_data, sizeof(cfg)))
		return -EFAULT;

	ret = hclge_ptp_set_ts_mode(hdev, &cfg);
	if (ret)
		return ret;

	hdev->ptp->ts_cfg = cfg;

	return copy_to_user(ifr->ifr_data, &cfg, sizeof(cfg)) ? -EFAULT : 0;
}

/**
 * @brief Reports timestamping capabilities to ethtool.
 * @param handle The HNAE3 handle.
 * @param info The ethtool timestamp info structure to be filled.
 * @return 0 on success, or -EOPNOTSUPP if PTP is not supported.
 */
int hclge_ptp_get_ts_info(struct hnae3_handle *handle,
			  struct kernel_ethtool_ts_info *info)
{
	struct hclge_vport *vport = hclge_get_vport(handle);
	struct hclge_dev *hdev = vport->back;

	if (!test_bit(HCLGE_STATE_PTP_EN, &hdev->state)) {
		dev_err(&hdev->pdev->dev, "phc is unsupported
");
		return -EOPNOTSUPP;
	}

	/* Block Logic: Populate the info struct with supported capabilities. */
	info->so_timestamping = SOF_TIMESTAMPING_TX_SOFTWARE |
				SOF_TIMESTAMPING_TX_HARDWARE |
				SOF_TIMESTAMPING_RX_HARDWARE |
				SOF_TIMESTAMPING_RAW_HARDWARE;

	if (hdev->ptp->clock)
		info->phc_index = ptp_clock_index(hdev->ptp->clock);

	info->tx_types = BIT(HWTSTAMP_TX_OFF) | BIT(HWTSTAMP_TX_ON);

	/* Block Logic: Announce supported hardware RX filters for various PTP versions and layers. */
	info->rx_filters = BIT(HWTSTAMP_FILTER_NONE) |
			   BIT(HWTSTAMP_FILTER_PTP_V2_L2_EVENT) |
			   BIT(HWTSTAMP_FILTER_PTP_V2_L2_SYNC) |
			   BIT(HWTSTAMP_FILTER_PTP_V2_L2_DELAY_REQ);

	info->rx_filters |= BIT(HWTSTAMP_FILTER_PTP_V1_L4_SYNC) |
			    BIT(HWTSTAMP_FILTER_PTP_V1_L4_DELAY_REQ) |
			    BIT(HWTSTAMP_FILTER_PTP_V2_EVENT) |
			    BIT(HWTSTAMP_FILTER_PTP_V2_L4_EVENT) |
			    BIT(HWTSTAMP_FILTER_PTP_V2_SYNC) |
			    BIT(HWTSTAMP_FILTER_PTP_V2_L4_SYNC) |
			    BIT(HWTSTAMP_FILTER_PTP_V2_DELAY_REQ) |
			    BIT(HWTSTAMP_FILTER_PTP_V2_L4_DELAY_REQ);

	return 0;
}

/**
 * @brief Allocates and initializes the PTP clock structure.
 * @param hdev The HCLGE device structure.
 * @return 0 on success, or a negative error code on failure.
 *
 * Functional Utility: This function sets up the driver's internal PTP tracking
 * structure and registers the PTP clock with the kernel's PTP subsystem.
 */
static int hclge_ptp_create_clock(struct hclge_dev *hdev)
{
	struct hclge_ptp *ptp;

	ptp = devm_kzalloc(&hdev->pdev->dev, sizeof(*ptp), GFP_KERNEL);
	if (!ptp)
		return -ENOMEM;

	ptp->hdev = hdev;

	/* Block Logic: Initialize the ptp_clock_info structure with callbacks and capabilities. */
	snprintf(ptp->info.name, sizeof(ptp->info.name), "%s",
		 HCLGE_DRIVER_NAME);
	ptp->info.owner = THIS_MODULE;
	ptp->info.max_adj = HCLGE_PTP_CYCLE_ADJ_MAX;
	ptp->info.n_ext_ts = 0;
	ptp->info.pps = 0;
	ptp->info.adjfine = hclge_ptp_adjfine;
	ptp->info.adjtime = hclge_ptp_adjtime;
	ptp->info.gettimex64 = hclge_ptp_gettimex;
	ptp->info.settime64 = hclge_ptp_settime;

	ptp->info.n_alarm = 0;

	spin_lock_init(&ptp->lock);
	ptp->io_base = hdev->hw.hw.io_base + HCLGE_PTP_REG_OFFSET;
	ptp->ts_cfg.rx_filter = HWTSTAMP_FILTER_NONE;
	ptp->ts_cfg.tx_type = HWTSTAMP_TX_OFF;
	hdev->ptp = ptp;

	/* Block Logic: Register this clock with the PTP subsystem. */
	ptp->clock = ptp_clock_register(&ptp->info, &hdev->pdev->dev);
	if (IS_ERR(ptp->clock)) {
		dev_err(&hdev->pdev->dev,
			"%d failed to register ptp clock, ret = %ld
",
			ptp->info.n_alarm, PTR_ERR(ptp->clock));
		return -ENODEV;
	} else if (!ptp->clock) {
		dev_err(&hdev->pdev->dev, "failed to register ptp clock
");
		return -ENODEV;
	}

	return 0;
}

/**
 * @brief Unregisters and destroys the PTP clock structure.
 * @param hdev The HCLGE device structure.
 */
static void hclge_ptp_destroy_clock(struct hclge_dev *hdev)
{
	ptp_clock_unregister(hdev->ptp->clock);
	hdev->ptp->clock = NULL;
	devm_kfree(&hdev->pdev->dev, hdev->ptp);
	hdev->ptp = NULL;
}

/**
 * @brief Initializes the PTP functionality for the device.
 * @param hdev The HCLGE device structure.
 * @return 0 on success, or a negative error code on failure.
 *
 * Functional Utility: This is the main entry point for PTP initialization.
 * It checks for hardware support, creates the PTP clock, enables interrupts,
 * and synchronizes the hardware clock with the system's real time.
 */
int hclge_ptp_init(struct hclge_dev *hdev)
{
	struct hnae3_ae_dev *ae_dev = pci_get_drvdata(hdev->pdev);
	struct timespec64 ts;
	int ret;

	/* Pre-condition: Check if the hardware reports PTP support. */
	if (!test_bit(HNAE3_DEV_SUPPORT_PTP_B, ae_dev->caps))
		return 0;

	if (!hdev->ptp) {
		ret = hclge_ptp_create_clock(hdev);
		if (ret)
			return ret;

		ret = hclge_ptp_get_cycle(hdev);
		if (ret)
			goto out;
	}

	ret = hclge_ptp_int_en(hdev, true);
	if (ret)
		goto out;

	set_bit(HCLGE_PTP_FLAG_EN, &hdev->ptp->flags);

	/* Block Logic: Perform an initial frequency adjustment of zero to stabilize the clock. */
	ret = hclge_ptp_adjfine(&hdev->ptp->info, 0);
	if (ret) {
		dev_err(&hdev->pdev->dev,
			"failed to init freq, ret = %d
", ret);
		goto out_clear_int;
	}

	/* Block Logic: Apply the default (disabled) timestamping mode. */
	ret = hclge_ptp_set_ts_mode(hdev, &hdev->ptp->ts_cfg);
	if (ret) {
		dev_err(&hdev->pdev->dev,
			"failed to init ts mode, ret = %d
", ret);
		goto out_clear_int;
	}

	/* Block Logic: Synchronize the PHC with the current system time. */
	ktime_get_real_ts64(&ts);
	ret = hclge_ptp_settime(&hdev->ptp->info, &ts);
	if (ret) {
		dev_err(&hdev->pdev->dev,
			"failed to init ts time, ret = %d
", ret);
		goto out_clear_int;
	}

	set_bit(HCLGE_STATE_PTP_EN, &hdev->state);
	dev_info(&hdev->pdev->dev, "phc initializes ok!
");

	return 0;

out_clear_int:
	/* Block Logic: Error handling path to disable interrupts on failure. */
	clear_bit(HCLGE_PTP_FLAG_EN, &hdev->ptp->flags);
	hclge_ptp_int_en(hdev, false);
out:
	/* Block Logic: Fully destroy the clock on initialization failure. */
	hclge_ptp_destroy_clock(hdev);

	return ret;
}

/**
 * @brief Uninitializes the PTP functionality for the device.
 * @param hdev The HCLGE device structure.
 *
 * Functional Utility: This is the main entry point for PTP uninitialization.
 * It disables PTP features in hardware, cleans up any pending TX skbs, and
 * destroys the PTP clock instance.
 */
void hclge_ptp_uninit(struct hclge_dev *hdev)
{
	struct hclge_ptp *ptp = hdev->ptp;

	if (!ptp)
		return;

	/* Hardware Interaction: Disable PTP interrupts and reset PTP mode. */
	hclge_ptp_int_en(hdev, false);
	clear_bit(HCLGE_STATE_PTP_EN, &hdev->state);
	clear_bit(HCLGE_PTP_FLAG_EN, &ptp->flags);
	ptp->ts_cfg.rx_filter = HWTSTAMP_FILTER_NONE;
	ptp->ts_cfg.tx_type = HWTSTAMP_TX_OFF;

	if (hclge_ptp_set_ts_mode(hdev, &ptp->ts_cfg))
		dev_err(&hdev->pdev->dev, "failed to disable phc
");

	/* Block Logic: Clean up any TX skb that was waiting for a timestamp. */
	if (ptp->tx_skb) {
		struct sk_buff *skb = ptp->tx_skb;

		ptp->tx_skb = NULL;
		dev_kfree_skb_any(skb);
	}

	hclge_ptp_destroy_clock(hdev);
}
