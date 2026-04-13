/**
 * @file hclge_ptp.c
 * @brief PTP (Precision Time Protocol) hardware clock support for Hisilicon HNS3 series NICs.
 * @details This file implements the necessary functions to integrate the hardware
 *          PTP clock (PHC) of the HNS3 network controller with the Linux PTP
 *          subsystem. It handles time synchronization, frequency adjustment,
 *          and hardware timestamping for both transmit (Tx) and receive (Rx) packets.
 *          The driver communicates with the hardware through memory-mapped I/O
 *          to configure PTP features and retrieve timestamps.
 */
// SPDX-License-Identifier: GPL-2.0+
// Copyright (c) 2021 Hisilicon Limited.

#include <linux/skbuff.h>
#include <linux/string_choices.h>
#include "hclge_main.h"
#include "hnae3.h"

/**
 * @brief Reads the hardware clock cycle parameters from registers.
 * @param hdev Pointer to the hclge_dev structure.
 * @return 0 on success, or -EINVAL if the denominator is invalid.
 * @note This function accesses hardware registers to fetch the quotient,
 *       numerator, and denominator that define the hardware clock's cycle period.
 *       These values are fundamental for frequency adjustment calculations.
 */
static int hclge_ptp_get_cycle(struct hclge_dev *hdev)
{
	struct hclge_ptp *ptp = hdev->ptp;

	ptp->cycle.quo = readl(hdev->ptp->io_base + HCLGE_PTP_CYCLE_QUO_REG) &
			 HCLGE_PTP_CYCLE_QUO_MASK;
	ptp->cycle.numer = readl(hdev->ptp->io_base + HCLGE_PTP_CYCLE_NUM_REG);
	ptp->cycle.den = readl(hdev->ptp->io_base + HCLGE_PTP_CYCLE_DEN_REG);

	if (ptp->cycle.den == 0) {
		dev_err(&hdev->pdev->dev, "invalid ptp cycle denominator!
");
		return -EINVAL;
	}

	return 0;
}

/**
 * @brief Adjusts the hardware clock frequency.
 * @param ptp Pointer to the PTP clock info structure.
 * @param scaled_ppm The desired frequency adjustment in scaled parts-per-million.
 * @return 0 on success.
 * @note This is a callback for the Linux PTP subsystem. It calculates a new
 *       clock cycle value based on the requested `scaled_ppm` and writes it
 *       to the hardware registers to perform the frequency adjustment.
 *       The adjustment is applied atomically by enabling the cycle adjustment register.
 */
static int hclge_ptp_adjfine(struct ptp_clock_info *ptp, long scaled_ppm)
{
	struct hclge_dev *hdev = hclge_ptp_get_hdev(ptp);
	struct hclge_ptp_cycle *cycle = &hdev->ptp->cycle;
	u64 adj_val, adj_base;
	unsigned long flags;
	u32 quo, numerator;

	/*
	 * Block Logic: Calculate the base clock value and the adjusted value.
	 * The `adj_base` represents the current cycle period in a fixed-point format.
	 * `adjust_by_scaled_ppm` computes the new cycle period based on the `scaled_ppm` offset.
	 */
	adj_base = (u64)cycle->quo * (u64)cycle->den + (u64)cycle->numer;
	adj_val = adjust_by_scaled_ppm(adj_base, scaled_ppm);

	/*
	 * Block Logic: Decompose the adjusted value back into hardware-specific
	 * quotient and numerator parts.
	 * This clock cycle is defined by three part: quotient, numerator
	 * and denominator. For example, 2.5ns, the quotient is 2,
	 * denominator is fixed to ptp->cycle.den, and numerator
	 * is 0.5 * ptp->cycle.den.
	 */
	quo = div_u64_rem(adj_val, cycle->den, &numerator);

	/*
	 * Block Logic: Write the new cycle parameters to the hardware registers.
	 * Pre-condition: A spinlock must be held to ensure atomic update of the
	 * multiple registers involved in the frequency adjustment.
	 * Invariant: After this block, the hardware is triggered to use the new
	 * cycle values for its internal clock.
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
 * @brief Prepares for a PTP transmit timestamp operation.
 * @param handle Pointer to the hnae3_handle structure.
 * @param skb The socket buffer being transmitted.
 * @return true if the packet is accepted for timestamping, false otherwise.
 * @note This function is called from the transmit path for packets requiring
 *       hardware timestamping. It performs checks to ensure that PTP Tx is
 *       enabled and that no other Tx timestamping operation is already in progress.
 *       It saves a reference to the skb, which will be used later to report the timestamp.
 */
bool hclge_ptp_set_tx_info(struct hnae3_handle *handle, struct sk_buff *skb)
{
	struct hclge_vport *vport = hclge_get_vport(handle);
	struct hclge_dev *hdev = vport->back;
	struct hclge_ptp *ptp = hdev->ptp;

	if (!ptp)
		return false;

	/*
	 * Block Logic: Check if Tx timestamping is enabled and if another one is already pending.
	 * If conditions are not met, the packet is skipped for timestamping to avoid races
	 * or unnecessary work.
	 */
	if (!test_bit(HCLGE_PTP_FLAG_TX_EN, &ptp->flags) ||
	    test_and_set_bit(HCLGE_STATE_PTP_TX_HANDLING, &hdev->state)) {
		ptp->tx_skipped++;
		return false;
	}

	ptp->tx_start = jiffies;
	ptp->tx_skb = skb_get(skb);
	ptp->tx_cnt++;

	return true;
}

/**
 * @brief Retrieves the Tx hardware timestamp and completes the Tx timestamping process.
 * @param hdev Pointer to the hclge_dev structure.
 * @note This function is called after the hardware has captured a timestamp for
 *       a transmitted packet. It reads the timestamp from hardware registers,
 *       applies it to the saved skb using `skb_tstamp_tx`, and then frees the skb.
 *       Finally, it clears the `HCLGE_STATE_PTP_TX_HANDLING` state bit.
 */
void hclge_ptp_clean_tx_hwts(struct hclge_dev *hdev)
{
	struct sk_buff *skb = hdev->ptp->tx_skb;
	struct skb_shared_hwtstamps hwts;
	u32 hi, lo;
	u64 ns;

	/*
	 * Block Logic: Read the multipart timestamp from hardware registers.
	 * The timestamp is composed of nanoseconds and a split seconds value (high/low parts).
	 */
	ns = readl(hdev->ptp->io_base + HCLGE_PTP_TX_TS_NSEC_REG) &
	     HCLGE_PTP_TX_TS_NSEC_MASK;
	lo = readl(hdev->ptp->io_base + HCLGE_PTP_TX_TS_SEC_L_REG);
	hi = readl(hdev->ptp->io_base + HCLGE_PTP_TX_TS_SEC_H_REG) &
	     HCLGE_PTP_TX_TS_SEC_H_MASK;
	hdev->ptp->last_tx_seqid = readl(hdev->ptp->io_base +
		HCLGE_PTP_TX_TS_SEQID_REG);

	/*
	 * Block Logic: If there is a pending skb, associate the timestamp with it.
	 * Pre-condition: hdev->ptp->tx_skb must point to a valid skb awaiting a timestamp.
	 * Invariant: The skb is timestamped, freed, and the tx_skb pointer is cleared.
	 */
	if (skb) {
		hdev->ptp->tx_skb = NULL;
		hdev->ptp->tx_cleaned++;

		ns += (((u64)hi) << 32 | lo) * NSEC_PER_SEC;
		hwts.hwtstamp = ns_to_ktime(ns);
		skb_tstamp_tx(skb, &hwts);
		dev_kfree_skb_any(skb);
	}

	clear_bit(HCLGE_STATE_PTP_TX_HANDLING, &hdev->state);
}

/**
 * @brief Retrieves and applies the Rx hardware timestamp to a received skb.
 * @param handle Pointer to the hnae3_handle structure.
 * @param skb The received socket buffer.
 * @param nsec The nanoseconds part of the timestamp from the Rx descriptor.
 * @param sec The lower 32 bits of the seconds part of the timestamp from the Rx descriptor.
 * @note The Rx descriptor provides only the lower part of the seconds timestamp.
 *       This function reads the upper 16 bits from a hardware register to reconstruct
 *       the full 48-bit seconds value, then combines it with the nanoseconds part
 *       and applies it to the skb.
 */
void hclge_ptp_get_rx_hwts(struct hnae3_handle *handle, struct sk_buff *skb,
			   u32 nsec, u32 sec)
{
	struct hclge_vport *vport = hclge_get_vport(handle);
	struct hclge_dev *hdev = vport->back;
	unsigned long flags;
	u64 ns = nsec;
	u32 sec_h;

	if (!hdev->ptp || !test_bit(HCLGE_PTP_FLAG_RX_EN, &hdev->ptp->flags))
		return;

	/*
	 * Block Logic: Reconstruct the full timestamp.
	 * Since the BD does not have enough space for the higher 16 bits of
	 * second, and this part will not change frequently, so read it
	 * from register. A spinlock is used to protect the register read.
	 */
	spin_lock_irqsave(&hdev->ptp->lock, flags);
	sec_h = readl(hdev->ptp->io_base + HCLGE_PTP_CUR_TIME_SEC_H_REG);
	spin_unlock_irqrestore(&hdev->ptp->lock, flags);

	ns += (((u64)sec_h) << HCLGE_PTP_SEC_H_OFFSET | sec) * NSEC_PER_SEC;
	skb_hwtstamps(skb)->hwtstamp = ns_to_ktime(ns);
	hdev->ptp->last_rx = jiffies;
	hdev->ptp->rx_cnt++;
}

/**
 * @brief Reads the current time from the hardware clock.
 * @param ptp Pointer to the PTP clock info structure.
 * @param ts Pointer to a timespec64 structure to store the time.
 * @param sts Pointer to a ptp_system_timestamp structure (unused).
 * @return 0 on success.
 * @note This is a callback for the Linux PTP subsystem. It reads the current
 *       nanoseconds and seconds from the hardware registers atomically
 *       (protected by a spinlock) and converts the result to a timespec64.
 */
static int hclge_ptp_gettimex(struct ptp_clock_info *ptp, struct timespec64 *ts,
			      struct ptp_system_timestamp *sts)
{
	struct hclge_dev *hdev = hclge_ptp_get_hdev(ptp);
	unsigned long flags;
	u32 hi, lo;
	u64 ns;

	spin_lock_irqsave(&hdev->ptp->lock, flags);
	ns = readl(hdev->ptp->io_base + HCLGE_PTP_CUR_TIME_NSEC_REG);
	hi = readl(hdev->ptp->io_base + HCLGE_PTP_CUR_TIME_SEC_H_REG);
	lo = readl(hdev->ptp->io_base + HCLGE_PTP_CUR_TIME_SEC_L_REG);
	spin_unlock_irqrestore(&hdev->ptp->lock, flags);

	ns += (((u64)hi) << HCLGE_PTP_SEC_H_OFFSET | lo) * NSEC_PER_SEC;
	*ts = ns_to_timespec64(ns);

	return 0;
}

/**
 * @brief Sets the hardware clock to a specific time.
 * @param ptp Pointer to the PTP clock info structure.
 * @param ts Pointer to the timespec64 structure containing the time to set.
 * @return 0 on success.
 * @note This is a callback for the Linux PTP subsystem. It writes the given
 *       time into the hardware registers and then triggers a synchronization
 *       event to atomically update the hardware clock.
 */
static int hclge_ptp_settime(struct ptp_clock_info *ptp,
			     const struct timespec64 *ts)
{
	struct hclge_dev *hdev = hclge_ptp_get_hdev(ptp);
	unsigned long flags;

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
 * @param ptp Pointer to the PTP clock info structure.
 * @param delta The time adjustment in nanoseconds.
 * @return 0 on success.
 * @note This is a callback for the Linux PTP subsystem. For small adjustments,
 *       it writes the delta to the hardware adjustment register. For large
 *       adjustments, it falls back to a full `hclge_ptp_settime` call to
 *       prevent overflowing the hardware's adjustment range.
 */
static int hclge_ptp_adjtime(struct ptp_clock_info *ptp, s64 delta)
{
	struct hclge_dev *hdev = hclge_ptp_get_hdev(ptp);
	unsigned long flags;
	bool is_neg = false;
	u32 adj_val = 0;

	if (delta < 0) {
		adj_val |= HCLGE_PTP_TIME_NSEC_NEG;
		delta = -delta;
		is_neg = true;
	}

	/*
	 * Block Logic: Handle large deltas by reading, adjusting, and writing back
	 * the full time value, as the hardware adjustment register has a limited range.
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
	 * Block Logic: For small deltas, use the hardware's time adjustment feature
	 * for a fine-grained phase shift.
	 */
	spin_lock_irqsave(&hdev->ptp->lock, flags);
	writel(adj_val, hdev->ptp->io_base + HCLGE_PTP_TIME_NSEC_REG);
	writel(HCLGE_PTP_TIME_ADJ_EN,
	       hdev->ptp->io_base + HCLGE_PTP_TIME_ADJ_REG);
	spin_unlock_irqrestore(&hdev->ptp->lock, flags);

	return 0;
}

/**
 * @brief Gets the current timestamping configuration.
 * @param hdev Pointer to the hclge_dev structure.
 * @param ifr Pointer to the ifreq structure from the user.
 * @return 0 on success, -EOPNOTSUPP if PTP is not enabled, or -EFAULT on copy error.
 * @note This function handles the SIOCGHWTSTAMP ioctl.
 */
int hclge_ptp_get_cfg(struct hclge_dev *hdev, struct ifreq *ifr)
{
	if (!test_bit(HCLGE_STATE_PTP_EN, &hdev->state))
		return -EOPNOTSUPP;

	return copy_to_user(ifr->ifr_data, &hdev->ptp->ts_cfg,
		sizeof(struct hwtstamp_config)) ? -EFAULT : 0;
}

/**
 * @brief Enables or disables the PTP interrupt in hardware.
 * @param hdev Pointer to the hclge_dev structure.
 * @param en Boolean flag to enable (true) or disable (false) the interrupt.
 * @return 0 on success, or an error code from the command send function.
 * @note This uses an asynchronous command to the firmware to control the PTP interrupt.
 */
static int hclge_ptp_int_en(struct hclge_dev *hdev, bool en)
{
	struct hclge_ptp_int_cmd *req;
	struct hclge_desc desc;
	int ret;

	req = (struct hclge_ptp_int_cmd *)desc.data;
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
 * @brief Queries the current PTP hardware configuration from the firmware.
 * @param hdev Pointer to the hclge_dev structure.
 * @param cfg Pointer to a u32 to store the configuration bitmap.
 * @return 0 on success, or an error code on failure.
 * @note This sends a command to the firmware to read the current PTP mode settings.
 */
int hclge_ptp_cfg_qry(struct hclge_dev *hdev, u32 *cfg)
{
	struct hclge_ptp_cfg_cmd *req;
	struct hclge_desc desc;
	int ret;

	req = (struct hclge_ptp_cfg_cmd *)desc.data;
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
 * @brief Writes a new PTP hardware configuration to the firmware.
 * @param hdev Pointer to the hclge_dev structure.
 * @param cfg The configuration bitmap to write.
 * @return 0 on success, or an error code on failure.
 * @note This sends a command to the firmware to update the PTP mode settings.
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
 * @brief Configures the Tx timestamping mode based on user request.
 * @param cfg Pointer to the user's hwtstamp_config.
 * @param flags Pointer to the driver's internal PTP flags.
 * @param ptp_cfg Pointer to the hardware configuration bitmap to be updated.
 * @return 0 on success, or -ERANGE for an unsupported mode.
 */
static int hclge_ptp_set_tx_mode(struct hwtstamp_config *cfg,
				 unsigned long *flags, u32 *ptp_cfg)
{
	switch (cfg->tx_type) {
	case HWTSTAMP_TX_OFF:
		clear_bit(HCLGE_PTP_FLAG_TX_EN, flags);
		break;
	case HWTSTAMP_TX_ON:
		set_bit(HCLGE_PTP_FLAG_TX_EN, flags);
		*ptp_cfg |= HCLGE_PTP_TX_EN_B;
		break;
	default:
		return -ERANGE;
	}

	return 0;
}

/**
 * @brief Configures the Rx timestamping filter mode based on user request.
 * @param cfg Pointer to the user's hwtstamp_config, `rx_filter` may be updated.
 * @param flags Pointer to the driver's internal PTP flags.
 * @param ptp_cfg Pointer to the hardware configuration bitmap to be updated.
 * @return 0 on success, or -ERANGE for an unsupported mode.
 * @note This function translates the generic Linux `HWTSTAMP_FILTER_*` values
 *       into hardware-specific configuration bits for PTP message type filtering.
 *       It also normalizes the user's filter request to a canonical value.
 */
static int hclge_ptp_set_rx_mode(struct hwtstamp_config *cfg,
				 unsigned long *flags, u32 *ptp_cfg)
{
	int rx_filter = cfg->rx_filter;

	/*
	 * Block Logic: This switch statement translates the user-requested Rx filter
	 * into hardware configuration bits. It handles various PTPv1 and PTPv2
	 * filter types across Layer 2 and Layer 4.
	 */
	switch (cfg->rx_filter) {
	case HWTSTAMP_FILTER_NONE:
		clear_bit(HCLGE_PTP_FLAG_RX_EN, flags);
		break;
	case HWTSTAMP_FILTER_PTP_V1_L4_SYNC:
	case HWTSTAMP_FILTER_PTP_V1_L4_DELAY_REQ:
	case HWTSTAMP_FILTER_PTP_V1_L4_EVENT:
		set_bit(HCLGE_PTP_FLAG_RX_EN, flags);
		*ptp_cfg |= HCLGE_PTP_RX_EN_B;
		*ptp_cfg |= HCLGE_PTP_UDP_FULL_TYPE << HCLGE_PTP_UDP_EN_SHIFT;
		rx_filter = HWTSTAMP_FILTER_PTP_V1_L4_EVENT;
		break;
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
 * @brief Sets the overall timestamping mode (Tx and Rx).
 * @param hdev Pointer to the hclge_dev structure.
 * @param cfg Pointer to the requested hwtstamp_config.
 * @return 0 on success, or an error code on failure.
 * @note This function orchestrates the configuration of both Tx and Rx timestamping
 *       by calling helper functions and then writing the final configuration to hardware.
 */
static int hclge_ptp_set_ts_mode(struct hclge_dev *hdev,
				 struct hwtstamp_config *cfg)
{
	unsigned long flags = hdev->ptp->flags;
	u32 ptp_cfg = 0;
	int ret;

	if (test_bit(HCLGE_PTP_FLAG_EN, &hdev->ptp->flags))
		ptp_cfg |= HCLGE_PTP_EN_B;

	ret = hclge_ptp_set_tx_mode(cfg, &flags, &ptp_cfg);
	if (ret)
		return ret;

	ret = hclge_ptp_set_rx_mode(cfg, &flags, &ptp_cfg);
	if (ret)
		return ret;

	ret = hclge_ptp_cfg(hdev, ptp_cfg);
	if (ret)
		return ret;

	hdev->ptp->flags = flags;
	hdev->ptp->ptp_cfg = ptp_cfg;

	return 0;
}

/**
 * @brief Sets the timestamping configuration based on a user request.
 * @param hdev Pointer to the hclge_dev structure.
 * @param ifr Pointer to the ifreq structure from the user.
 * @return 0 on success, or an error code on failure.
 * @note This function handles the SIOCSHWTSTAMP ioctl. It copies the user's
 *       configuration, applies it to the hardware, and copies the (potentially
 *       modified) configuration back to the user.
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
 * @brief Provides timestamping capabilities to ethtool.
 * @param handle Pointer to the hnae3_handle structure.
 * @param info Pointer to the kernel_ethtool_ts_info structure to be filled.
 * @return 0 on success, or -EOPNOTSUPP if PTP is not supported.
 * @note This function reports the device's PTP capabilities, including supported
 *       timestamping modes and filters, for use by tools like ethtool.
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

	info->so_timestamping = SOF_TIMESTAMPING_TX_SOFTWARE |
				SOF_TIMESTAMPING_TX_HARDWARE |
				SOF_TIMESTAMPING_RX_HARDWARE |
				SOF_TIMESTAMPING_RAW_HARDWARE;

	if (hdev->ptp->clock)
		info->phc_index = ptp_clock_index(hdev->ptp->clock);

	info->tx_types = BIT(HWTSTAMP_TX_OFF) | BIT(HWTSTAMP_TX_ON);

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
 * @param hdev Pointer to the hclge_dev structure.
 * @return 0 on success, or an error code on failure.
 * @note This function allocates memory for the PTP control structure, populates
 *       the `ptp_clock_info` with capabilities and callbacks, and registers
 *       the clock with the Linux PTP subsystem.
 */
static int hclge_ptp_create_clock(struct hclge_dev *hdev)
{
	struct hclge_ptp *ptp;

	ptp = devm_kzalloc(&hdev->pdev->dev, sizeof(*ptp), GFP_KERNEL);
	if (!ptp)
		return -ENOMEM;

	ptp->hdev = hdev;
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
 * @brief Unregisters and destroys the PTP clock.
 * @param hdev Pointer to the hclge_dev structure.
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
 * @param hdev Pointer to the hclge_dev structure.
 * @return 0 on success, or an error code on failure.
 * @note This is the main entry point for PTP initialization. It checks for
 *       hardware support, creates the PTP clock, enables interrupts, sets
 *       initial frequency and time, and enables the overall PTP state.
 */
int hclge_ptp_init(struct hclge_dev *hdev)
{
	struct hnae3_ae_dev *ae_dev = pci_get_drvdata(hdev->pdev);
	struct timespec64 ts;
	int ret;

	/*
	 * Block Logic: Check if the hardware reports PTP support.
	 * If not, silently exit as PTP is not available on this device.
	 */
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

	/*
	 * Block Logic: Configure the initial state of the PTP hardware.
	 * This includes setting the initial frequency adjustment to zero,
	 * applying the default timestamping mode, and synchronizing the
	 * hardware clock with the current system (wall) time.
	 */
	set_bit(HCLGE_PTP_FLAG_EN, &hdev->ptp->flags);
	ret = hclge_ptp_adjfine(&hdev->ptp->info, 0);
	if (ret) {
		dev_err(&hdev->pdev->dev,
			"failed to init freq, ret = %d
", ret);
		goto out;
	}

	ret = hclge_ptp_set_ts_mode(hdev, &hdev->ptp->ts_cfg);
	if (ret) {
		dev_err(&hdev->pdev->dev,
			"failed to init ts mode, ret = %d
", ret);
		goto out;
	}

	ktime_get_real_ts64(&ts);
	ret = hclge_ptp_settime(&hdev->ptp->info, &ts);
	if (ret) {
		dev_err(&hdev->pdev->dev,
			"failed to init ts time, ret = %d
", ret);
		goto out;
	}

	set_bit(HCLGE_STATE_PTP_EN, &hdev->state);
	dev_info(&hdev->pdev->dev, "phc initializes ok!
");

	return 0;

out:
	hclge_ptp_destroy_clock(hdev);

	return ret;
}

/**
 * @brief Uninitializes the PTP functionality for the device.
 * @param hdev Pointer to the hclge_dev structure.
 * @note This function disables PTP interrupts, resets the timestamping
 *       configuration to 'off', cleans up any pending Tx skbs, and
 *       destroys the PTP clock instance.
 */
void hclge_ptp_uninit(struct hclge_dev *hdev)
{
	struct hclge_ptp *ptp = hdev->ptp;

	if (!ptp)
		return;

	hclge_ptp_int_en(hdev, false);
	clear_bit(HCLGE_STATE_PTP_EN, &hdev->state);
	clear_bit(HCLGE_PTP_FLAG_EN, &ptp->flags);
	ptp->ts_cfg.rx_filter = HWTSTAMP_FILTER_NONE;
	ptp->ts_cfg.tx_type = HWTSTAMP_TX_OFF;

	if (hclge_ptp_set_ts_mode(hdev, &ptp->ts_cfg))
		dev_err(&hdev->pdev->dev, "failed to disable phc
");

	if (ptp->tx_skb) {
		struct sk_buff *skb = ptp->tx_skb;

		ptp->tx_skb = NULL;
		dev_kfree_skb_any(skb);
	}

	hclge_ptp_destroy_clock(hdev);
}
