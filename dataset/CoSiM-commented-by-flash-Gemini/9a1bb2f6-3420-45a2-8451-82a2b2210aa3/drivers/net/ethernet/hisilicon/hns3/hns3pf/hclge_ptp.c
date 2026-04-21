// SPDX-License-Identifier: GPL-2.0+
// Copyright (c) 2021 Hisilicon Limited.

/**
 * @file hclge_ptp.c
 * @brief Precision Time Protocol (PTP) support for the Hisilicon HNS3 Ethernet controller.
 * 
 * Functional Intent: Provides the core logic for high-accuracy time synchronization 
 * between the hardware clock (PHC) and external references. Implements PTP clock 
 * adjustment, time-of-day management, and hardware-assisted packet timestamping 
 * for both RX and TX paths to minimize synchronization jitter in distributed systems.
 * 
 * Domain: Production Systems, Network Device Drivers, High-Precision Timing.
 */

#include <linux/skbuff.h>
#include <linux/string_choices.h>
#include "hclge_main.h"
#include "hnae3.h"

/**
 * hclge_ptp_get_cycle - Retrieves the hardware clock cycle configuration.
 * 
 * Logic: Reads the quotient, numerator, and denominator from hardware registers 
 * that define the granularity of the PTP clock increments.
 */
static int hclge_ptp_get_cycle(struct hclge_dev *hdev)
{
	struct hclge_ptp *ptp = hdev->ptp;

	ptp->cycle.quo = readl(hdev->ptp->io_base + HCLGE_PTP_CYCLE_QUO_REG) &
			 HCLGE_PTP_CYCLE_QUO_MASK;
	ptp->cycle.numer = readl(hdev->ptp->io_base + HCLGE_PTP_CYCLE_NUM_REG);
	ptp->cycle.den = readl(hdev->ptp->io_base + HCLGE_PTP_CYCLE_DEN_REG);

	// Pre-condition: Hardware must provide a non-zero denominator to avoid division errors.
	if (ptp->cycle.den == 0) {
		dev_err(&hdev->pdev->dev, "invalid ptp cycle denominator!\n");
		return -EINVAL;
	}

	return 0;
}

/**
 * hclge_ptp_adjfine - Adjusts the hardware clock frequency with fine granularity.
 * 
 * Algorithm: Proportional frequency adjustment using scaled PPM (parts per million).
 * Logic: Calculates a new clock increment value (quo + numer/den) based on 
 * the drift feedback, allowing the PHC to track the master clock's frequency.
 */
static int hclge_ptp_adjfine(struct ptp_clock_info *ptp, long scaled_ppm)
{
	struct hclge_dev *hdev = hclge_ptp_get_hdev(ptp);
	struct hclge_ptp_cycle *cycle = &hdev->ptp->cycle;
	u64 adj_val, adj_base;
	unsigned long flags;
	u32 quo, numerator;

	adj_base = (u64)cycle->quo * (u64)cycle->den + (u64)cycle->numer;
	adj_val = adjust_by_scaled_ppm(adj_base, scaled_ppm);

	/* Block Logic: Fractional cycle decomposition.
	 * This clock cycle is defined by three part: quotient, numerator
	 * and denominator. For example, 2.5ns, the quotient is 2,
	 * denominator is fixed to ptp->cycle.den, and numerator
	 * is 0.5 * ptp->cycle.den.
	 */
	quo = div_u64_rem(adj_val, cycle->den, &numerator);

	// Synchronization: Disables interrupts to ensure atomic register updates.
	spin_lock_irqsave(&hdev->ptp->lock, flags);
	writel(quo & HCLGE_PTP_CYCLE_QUO_MASK,
	       hdev->ptp->io_base + HCLGE_PTP_CYCLE_QUO_REG);
	writel(numerator, hdev->ptp->io_base + HCLGE_PTP_CYCLE_NUM_REG);
	writel(cycle->den, hdev->ptp->io_base + HCLGE_PTP_CYCLE_DEN_REG);
	// Functional Utility: Activates the hardware adjustment logic.
	writel(HCLGE_PTP_CYCLE_ADJ_EN,
	       hdev->ptp->io_base + HCLGE_PTP_CYCLE_CFG_REG);
	spin_unlock_irqrestore(&hdev->ptp->lock, flags);

	return 0;
}

/**
 * hclge_ptp_set_tx_info - Prepares a packet for hardware TX timestamping.
 * 
 * Logic: Tags the socket buffer for PTP processing and sets the internal 
 * handling state to prevent concurrent TX timestamp requests.
 */
bool hclge_ptp_set_tx_info(struct hnae3_handle *handle, struct sk_buff *skb)
{
	struct hclge_vport *vport = hclge_get_vport(handle);
	struct hclge_dev *hdev = vport->back;
	struct hclge_ptp *ptp = hdev->ptp;

	if (!ptp)
		return false;

	// Optimization: Fast-path rejection if PTP TX is disabled or already in progress.
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
 * hclge_ptp_clean_tx_hwts - Retrieves and reports a hardware TX timestamp.
 * 
 * Logic: Reads the egress timestamp from hardware registers and notifies 
 * the network stack via skb_tstamp_tx.
 */
void hclge_ptp_clean_tx_hwts(struct hclge_dev *hdev)
{
	struct sk_buff *skb = hdev->ptp->tx_skb;
	struct skb_shared_hwtstamps hwts;
	u32 hi, lo;
	u64 ns;

	// Block Logic: Hardware timestamp reconstruction.
	ns = readl(hdev->ptp->io_base + HCLGE_PTP_TX_TS_NSEC_REG) &
	     HCLGE_PTP_TX_TS_NSEC_MASK;
	lo = readl(hdev->ptp->io_base + HCLGE_PTP_TX_TS_SEC_L_REG);
	hi = readl(hdev->ptp->io_base + HCLGE_PTP_TX_TS_SEC_H_REG) &
	     HCLGE_PTP_TX_TS_SEC_H_MASK;
	hdev->ptp->last_tx_seqid = readl(hdev->ptp->io_base +
		HCLGE_PTP_TX_TS_SEQID_REG);

	if (skb) {
		hdev->ptp->tx_skb = NULL;
		hdev->ptp->tx_cleaned++;

		// Conversion: Translates multi-register time format into absolute nanoseconds.
		ns += (((u64)hi) << 32 | lo) * NSEC_PER_SEC;
		hwts.hwtstamp = ns_to_ktime(ns);
		skb_tstamp_tx(skb, &hwts);
		dev_kfree_skb_any(skb);
	}

	// Synchronization: Releases the PTP TX handling lock.
	clear_bit(HCLGE_STATE_PTP_TX_HANDLING, &hdev->state);
}

/**
 * hclge_ptp_get_rx_hwts - Attaches hardware timestamps to an incoming packet.
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

	/* Block Logic: High-bit seconds retrieval.
	 * Since the BD does not have enough space for the higher 16 bits of
	 * second, and this part will not change frequently, so read it
	 * from register.
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
 * hclge_ptp_gettimex - Atomically retrieves the current hardware clock time.
 */
static int hclge_ptp_gettimex(struct ptp_clock_info *ptp, struct timespec64 *ts,
			      struct ptp_system_timestamp *sts)
{
	struct hclge_dev *hdev = hclge_ptp_get_hdev(ptp);
	unsigned long flags;
	u32 hi, lo;
	u64 ns;

	// Synchronization: Snapshot PHC time registers to prevent inconsistent mixed reads.
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
 * hclge_ptp_settime - Manually sets the hardware clock to a specific time.
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
	/* Functional Utility: Forces a hardware time synchronization pulse. */
	writel(HCLGE_PTP_TIME_SYNC_EN,
	       hdev->ptp->io_base + HCLGE_PTP_TIME_SYNC_REG);
	spin_unlock_irqrestore(&hdev->ptp->lock, flags);

	return 0;
}

/**
 * hclge_ptp_adjtime - Offsets the hardware clock by a specific delta.
 * 
 * Logic: Efficiently handles small nanosecond adjustments vs large multi-second shifts 
 * by choosing between register-based offsets or full clock re-setting.
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

	// Logic: If delta exceeds hardware register limits, fallback to manual get-adjust-set.
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

	spin_lock_irqsave(&hdev->ptp->lock, flags);
	writel(adj_val, hdev->ptp->io_base + HCLGE_PTP_TIME_NSEC_REG);
	// Functional Utility: Triggers the hardware nanosecond accumulator adjustment.
	writel(HCLGE_PTP_TIME_ADJ_EN,
	       hdev->ptp->io_base + HCLGE_PTP_TIME_ADJ_REG);
	spin_unlock_irqrestore(&hdev->ptp->lock, flags);

	return 0;
}

int hclge_ptp_get_cfg(struct hclge_dev *hdev, struct ifreq *ifr)
{
	if (!test_bit(HCLGE_STATE_PTP_EN, &hdev->state))
		return -EOPNOTSUPP;

	return copy_to_user(ifr->ifr_data, &hdev->ptp->ts_cfg,
		sizeof(struct hwtstamp_config)) ? -EFAULT : 0;
}

/**
 * hclge_ptp_int_en - Enables or disables hardware PTP-related interrupts.
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
			"failed to %s ptp interrupt, ret = %d\n",
			str_enable_disable(en), ret);

	return ret;
}

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
			"failed to query ptp config, ret = %d\n", ret);
		return ret;
	}

	*cfg = le32_to_cpu(req->cfg);

	return 0;
}

/**
 * hclge_ptp_cfg - Configures the hardware PTP operating mode.
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
			"failed to config ptp, ret = %d\n", ret);

	return ret;
}

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

static int hclge_ptp_set_rx_mode(struct hwtstamp_config *cfg,
				 unsigned long *flags, u32 *ptp_cfg)
{
	int rx_filter = cfg->rx_filter;

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
 * hclge_ptp_set_ts_mode - Reconfigures PHC based on kernel timestamping request.
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

int hclge_ptp_set_cfg(struct hclge_dev *hdev, struct ifreq *ifr)
{
	struct hwtstamp_config cfg;
	int ret;

	if (!test_bit(HCLGE_STATE_PTP_EN, &hdev->state)) {
		dev_err(&hdev->pdev->dev, "phc is unsupported\n");
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
 * hclge_ptp_get_ts_info - Reports supported timestamping capabilities to ethtool.
 */
int hclge_ptp_get_ts_info(struct hnae3_handle *handle,
			  struct kernel_ethtool_ts_info *info)
{
	struct hclge_vport *vport = hclge_get_vport(handle);
	struct hclge_dev *hdev = vport->back;

	if (!test_bit(HCLGE_STATE_PTP_EN, &hdev->state)) {
		dev_err(&hdev->pdev->dev, "phc is unsupported\n");
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
 * hclge_ptp_create_clock - Allocates and registers the Linux PTP hardware clock.
 * 
 * Functional Utility: Initializes the ptp_clock_info structure with driver 
 * callbacks for frequency/time management and binds the PHC to the kernel infrastructure.
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
			"%d failed to register ptp clock, ret = %ld\n",
			ptp->info.n_alarm, PTR_ERR(ptp->clock));
		return -ENODEV;
	} else if (!ptp->clock) {
		dev_err(&hdev->pdev->dev, "failed to register ptp clock\n");
		return -ENODEV;
	}

	return 0;
}

static void hclge_ptp_destroy_clock(struct hclge_dev *hdev)
{
	ptp_clock_unregister(hdev->ptp->clock);
	hdev->ptp->clock = NULL;
	devm_kfree(&hdev->pdev->dev, hdev->ptp);
	hdev->ptp = NULL;
}

/**
 * hclge_ptp_init - Entry point for PTP subsystem initialization during device probe.
 * 
 * Logic: Checks hardware capabilities, registers the PHC, enables interrupts, 
 * and synchronizes the PHC with system real-time to provide a consistent baseline.
 */
int hclge_ptp_init(struct hclge_dev *hdev)
{
	struct hnae3_ae_dev *ae_dev = pci_get_drvdata(hdev->pdev);
	struct timespec64 ts;
	int ret;

	// Pre-condition: Device must report PTP support in its capability flags.
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
	// Logic: Resets frequency adjustment to baseline (0 PPM).
	ret = hclge_ptp_adjfine(&hdev->ptp->info, 0);
	if (ret) {
		dev_err(&hdev->pdev->dev,
			"failed to init freq, ret = %d\n", ret);
		goto out_clear_int;
	}

	ret = hclge_ptp_set_ts_mode(hdev, &hdev->ptp->ts_cfg);
	if (ret) {
		dev_err(&hdev->pdev->dev,
			"failed to init ts mode, ret = %d\n", ret);
		goto out_clear_int;
	}

	// Initialization: Sets PHC to current system time to avoid large initial drifts.
	ktime_get_real_ts64(&ts);
	ret = hclge_ptp_settime(&hdev->ptp->info, &ts);
	if (ret) {
		dev_err(&hdev->pdev->dev,
			"failed to init ts time, ret = %d\n", ret);
		goto out_clear_int;
	}

	set_bit(HCLGE_STATE_PTP_EN, &hdev->state);
	dev_info(&hdev->pdev->dev, "phc initializes ok!\n");

	return 0;

out_clear_int:
	clear_bit(HCLGE_PTP_FLAG_EN, &hdev->ptp->flags);
	hclge_ptp_int_en(hdev, false);
out:
	hclge_ptp_destroy_clock(hdev);

	return ret;
}

/**
 * hclge_ptp_uninit - Teardown logic for the PTP subsystem during device removal.
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
		dev_err(&hdev->pdev->dev, "failed to disable phc\n");

	// Cleanup: Releases any socket buffer currently pending a TX timestamp.
	if (ptp->tx_skb) {
		struct sk_buff *skb = ptp->tx_skb;

		ptp->tx_skb = NULL;
		dev_kfree_skb_any(skb);
	}

	hclge_ptp_destroy_clock(hdev);
}
