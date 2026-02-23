/**
 * @file i2c-imx-lpi2c.c
 * @brief i.MX low power I2C controller driver for Linux.
 *
 * This driver provides support for the i.MX Low Power I2C (LPI2C) controller,
 * enabling I2C master and slave functionalities. It integrates with the Linux
 * kernel's I2C subsystem, supporting both PIO (Programmed I/O) and DMA
 * (Direct Memory Access) transfers for efficient data handling.
 *
 * Architecture:
 * - Direct register access for LPI2C hardware control.
 * - Interrupt-driven operation for both master and slave modes.
 * - Optional DMA support for read and write transfers to offload CPU.
 * - Runtime Power Management (RPM) integration for power efficiency.
 * - I2C bus recovery mechanisms.
 *
 * Functional Utility:
 * - Manages clock configurations, FIFO operations, and transfer control.
 * - Implements I2C message transfer (`i2c_adapter.xfer`).
 * - Supports I2C slave registration and unregistration.
 * - Handles arbitration loss, NACKs, and other I2C bus events.
 *
 * Domain-Specific Awareness (HPC & Parallelism, Performance Optimization):
 * - DMA is utilized for higher throughput transfers, especially for messages
 *   larger than a defined threshold, reducing CPU overhead.
 * - FIFO management (`txfifosize`, `rxfifosize`, watermarks) optimizes data flow
 *   between CPU/DMA and LPI2C hardware.
 * - `readl_poll_timeout` is used for polling hardware registers with a timeout
 *   to prevent hangs and improve robustness.
 * - PM runtime calls ensure the I2C controller is powered efficiently, only active
 *   when needed.
 *
 * SPDX-License-Identifier: GPL-2.0+
 * Copyright 2016 Freescale Semiconductor, Inc.
 */

#include <linux/clk.h>
#include <linux/completion.h>
#include <linux/delay.h>
#include <linux/dma-mapping.h>
#include <linux/dmaengine.h>
#include <linux/err.h>
#include <linux/errno.h>
#include <linux/i2c.h>
#include <linux/init.h>
#include <linux/interrupt.h>
#include <linux/io.h>
#include <linux/iopoll.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/pinctrl/consumer.h>
#include <linux/platform_device.h>
#include <linux/pm_runtime.h>
#include <linux/sched.h>
#include <linux/slab.h>

#define DRIVER_NAME "imx-lpi2c" // Device driver name.

// LPI2C Master Register Offsets
#define LPI2C_PARAM	0x04	/* i2c RX/TX FIFO size register */
#define LPI2C_MCR	0x10	/* i2c Master Control Register */
#define LPI2C_MSR	0x14	/* i2c Master Status Register */
#define LPI2C_MIER	0x18	/* i2c Master Interrupt Enable Register */
#define LPI2C_MDER	0x1C	/* i2c Master DMA Enable Register */
#define LPI2C_MCFGR0	0x20	/* i2c Master Configuration Register 0 */
#define LPI2C_MCFGR1	0x24	/* i2c Master Configuration Register 1 */
#define LPI2C_MCFGR2	0x28	/* i2c Master Configuration Register 2 */
#define LPI2C_MCFGR3	0x2C	/* i2c Master Configuration Register 3 */
#define LPI2C_MCCR0	0x48	/* i2c Master Clock Configuration Register 0 */
#define LPI2C_MCCR1	0x50	/* i2c Master Clock Configuration Register 1 (for high-speed mode) */
#define LPI2C_MFCR	0x58	/* i2c Master FIFO Control Register */
#define LPI2C_MFSR	0x5C	/* i2c Master FIFO Status Register */
#define LPI2C_MTDR	0x60	/* i2c Master Transmit Data Register (also for commands) */
#define LPI2C_MRDR	0x70	/* i2c Master Receive Data Register */

// LPI2C Slave Register Offsets
#define LPI2C_SCR	0x110	/* i2c Slave Control Register */
#define LPI2C_SSR	0x114	/* i2c Slave Status Register */
#define LPI2C_SIER	0x118	/* i2c Slave Interrupt Enable Register */
#define LPI2C_SDER	0x11C	/* i2c Slave DMA Enable Register */
#define LPI2C_SCFGR0	0x120	/* i2c Slave Configuration Register 0 */
#define LPI2C_SCFGR1	0x124	/* i2c Slave Configuration Register 1 */
#define LPI2C_SCFGR2	0x128	/* i2c Slave Configuration Register 2 */
#define LPI2C_SAMR	0x140	/* i2c Slave Address Match Register */
#define LPI2C_SASR	0x150	/* i2c Slave Address Status Register */
#define LPI2C_STAR	0x154	/* i2c Slave Transmit ACK Register */
#define LPI2C_STDR	0x160	/* i2c Slave Transmit Data Register */
#define LPI2C_SRDR	0x170	/* i2c Slave Receive Data Register */
#define LPI2C_SRDROR	0x178	/* i2c Slave Receive Data Read Only Register */

/* i2c command codes for MTDR */
#define TRAN_DATA	0X00 // Transmit Data
#define RECV_DATA	0X01 // Receive Data
#define GEN_STOP	0X02 // Generate STOP condition
#define RECV_DISCARD	0X03 // Receive and Discard Data
#define GEN_START	0X04 // Generate START condition
#define START_NACK	0X05 // Generate START and expect NACK
#define START_HIGH	0X06 // Generate START, High-speed mode
#define START_HIGH_NACK	0X07 // Generate START, High-speed mode, expect NACK

// Bit definitions for Master Control Register (LPI2C_MCR)
#define MCR_MEN		BIT(0) // Master Enable
#define MCR_RST		BIT(1) // Software Reset
#define MCR_DOZEN	BIT(2) // Doze Mode Enable
#define MCR_DBGEN	BIT(3) // Debug Mode Enable
#define MCR_RTF		BIT(8) // Receive FIFO Clear
#define MCR_RRF		BIT(9) // Transmit FIFO Clear
// Bit definitions for Master Status Register (LPI2C_MSR)
#define MSR_TDF		BIT(0) // Transmit Data Flag
#define MSR_RDF		BIT(1) // Receive Data Flag
#define MSR_SDF		BIT(9) // STOP Detect Flag
#define MSR_NDF		BIT(10) // NACK Detect Flag
#define MSR_ALF		BIT(11) // Arbitration Lost Flag
#define MSR_MBF		BIT(24) // Master Busy Flag
#define MSR_BBF		BIT(25) // Bus Busy Flag
// Bit definitions for Master Interrupt Enable Register (LPI2C_MIER)
#define MIER_TDIE	BIT(0) // Transmit Data Interrupt Enable
#define MIER_RDIE	BIT(1) // Receive Data Interrupt Enable
#define MIER_SDIE	BIT(9) // STOP Detect Interrupt Enable
#define MIER_NDIE	BIT(10) // NACK Detect Interrupt Enable
// Bit definitions for Master Configuration Register 1 (LPI2C_MCFGR1)
#define MCFGR1_AUTOSTOP	BIT(8) // Auto Stop Enable
#define MCFGR1_IGNACK	BIT(9) // Ignore NACK
// Bit definitions for Master Receive Data Register (LPI2C_MRDR)
#define MRDR_RXEMPTY	BIT(14) // RX Empty Flag
// Bit definitions for Master DMA Enable Register (LPI2C_MDER)
#define MDER_TDDE	BIT(0) // Transmit Data DMA Enable
#define MDER_RDDE	BIT(1) // Receive Data DMA Enable

// Bit definitions for Slave Control Register (LPI2C_SCR)
#define SCR_SEN		BIT(0) // Slave Enable
#define SCR_RST		BIT(1) // Software Reset
#define SCR_FILTEN	BIT(4) // Filter Enable
// Bit definitions for Slave Status Register (LPI2C_SSR)
#define SSR_TDF		BIT(0) // Transmit Data Flag
#define SSR_RDF		BIT(1) // Receive Data Flag
#define SSR_AVF		BIT(2) // Address Valid Flag
#define SSR_TAF		BIT(3) // Transmit ACK Flag
#define SSR_RSF		BIT(8) // Repeated START Flag
#define SSR_SDF		BIT(9) // STOP Detect Flag
#define SSR_BEF		BIT(10) // Bus Error Flag
#define SSR_FEF		BIT(11) // FIFO Error Flag
#define SSR_SBF		BIT(24) // Slave Busy Flag
#define SSR_BBF		BIT(25) // Bus Busy Flag
#define SSR_CLEAR_BITS	(SSR_RSF | SSR_SDF | SSR_BEF | SSR_FEF) // Bits to clear in SSR
// Bit definitions for Slave Interrupt Enable Register (LPI2C_SIER)
#define SIER_TDIE	BIT(0) // Transmit Data Interrupt Enable
#define SIER_RDIE	BIT(1) // Receive Data Interrupt Enable
#define SIER_AVIE	BIT(2) // Address Valid Interrupt Enable
#define SIER_TAIE	BIT(3) // Transmit ACK Interrupt Enable
#define SIER_RSIE	BIT(8) // Repeated START Interrupt Enable
#define SIER_SDIE	BIT(9) // STOP Detect Interrupt Enable
#define SIER_BEIE	BIT(10) // Bus Error Interrupt Enable
#define SIER_FEIE	BIT(11) // FIFO Error Interrupt Enable
#define SIER_AM0F	BIT(12) // Address Match 0 Flag
// Bit definitions for Slave Configuration Register 1 (LPI2C_SCFGR1)
#define SCFGR1_RXSTALL	BIT(1) // Receive Stall
#define SCFGR1_TXDSTALL	BIT(2) // Transmit Data Stall
// Bit definitions for Slave Configuration Register 2 (LPI2C_SCFGR2)
#define SCFGR2_FILTSDA_SHIFT	24 // Shift for SDA Filter
#define SCFGR2_FILTSCL_SHIFT	16 // Shift for SCL Filter
#define SCFGR2_CLKHOLD(x)	(x) // Clock Hold value
#define SCFGR2_FILTSDA(x)	((x) << SCFGR2_FILTSDA_SHIFT) // Set SDA Filter
#define SCFGR2_FILTSCL(x)	((x) << SCFGR2_FILTSCL_SHIFT) // Set SCL Filter
// Bit definitions for Slave Address Status Register (LPI2C_SASR)
#define SASR_READ_REQ	0x1 // Read Request
// Interrupt flags for slave mode
#define SLAVE_INT_FLAG	(SIER_TDIE | SIER_RDIE | SIER_AVIE | \
			 SIER_SDIE | SIER_BEIE) // Combined slave interrupt flags

#define I2C_CLK_RATIO	2 // Ratio between CLKLO and CLKHI.
#define CHUNK_DATA	256 // Maximum data chunk size for transfers.

#define I2C_PM_TIMEOUT		10 /* ms, for runtime power management autosuspend delay */
#define I2C_DMA_THRESHOLD	8 /* bytes, minimum message length to use DMA */

/**
 * @brief Defines the operating modes (speed grades) of the LPI2C controller.
 */
enum lpi2c_imx_mode {
	STANDARD,	/* 100+ Kbps */
	FAST,		/* 400+ Kbps */
	FAST_PLUS,	/* 1.0+ Mbps */
	HS,		/* 3.4+ Mbps (High Speed) */
	ULTRA_FAST,	/* 5.0+ Mbps */
};

/**
 * @brief Defines the pin configuration types for the LPI2C controller.
 */
enum lpi2c_imx_pincfg {
	TWO_PIN_OD,  // Two pin, open drain
	TWO_PIN_OO,  // Two pin, open output
	TWO_PIN_PP,  // Two pin, push-pull
	FOUR_PIN_PP, // Four pin, push-pull (SDA/SCL separated for TX/RX)
};

/**
 * @brief Structure to hold DMA related information for the LPI2C driver.
 * This includes DMA channel pointers, buffer details, and transfer parameters.
 */
struct lpi2c_imx_dma {
	bool		using_pio_mode; // Flag to indicate if PIO fallback is used after DMA failure.
	u8		rx_cmd_buf_len; // Length of the RX command buffer.
	u8		*dma_buf; // Pointer to the DMA buffer (I2C message data).
	u16		*rx_cmd_buf; // Buffer for RX command words in DMA mode.
	unsigned int	dma_len; // Length of data for current DMA transfer.
	unsigned int	tx_burst_num; // Transmit DMA burst number.
	unsigned int	rx_burst_num; // Receive DMA burst number.
	unsigned long	dma_msg_flag; // Flags of the current I2C message (e.g., I2C_M_RD).
	resource_size_t	phy_addr; // Physical base address of the LPI2C controller.
	dma_addr_t	dma_tx_addr; // DMA address for transmit (used for RX command buffer).
	dma_addr_t	dma_addr; // DMA address for the data buffer.
	enum dma_data_direction dma_data_dir; // Direction of DMA data transfer (TO_DEVICE/FROM_DEVICE).
	enum dma_transfer_direction dma_transfer_dir; // Transfer direction (MEM_TO_DEV/DEV_TO_MEM).
	struct dma_chan	*chan_tx; // DMA channel for transmit.
	struct dma_chan	*chan_rx; // DMA channel for receive.
};

/**
 * @brief Main driver structure for the i.MX LPI2C controller.
 * Holds all relevant hardware and software state for an LPI2C instance.
 */
struct lpi2c_imx_struct {
	struct i2c_adapter	adapter; // Linux I2C adapter structure.
	int			num_clks; // Number of clocks associated with the device.
	struct clk_bulk_data	*clks; // Array of clock data for the device.
	void __iomem		*base; // Base address of the LPI2C registers (memory-mapped).
	__u8			*rx_buf; // Pointer to the current receive buffer.
	__u8			*tx_buf; // Pointer to the current transmit buffer.
	struct completion	complete; // Completion object for PIO transfers.
	unsigned long		rate_per; // Peripheral clock rate.
	unsigned int		msglen; // Total length of the current I2C message.
	unsigned int		delivered; // Number of bytes delivered/processed so far.
	unsigned int		block_data; // Flag for SMBus block data read.
	unsigned int		bitrate; // Desired I2C bus bitrate.
	unsigned int		txfifosize; // Size of the transmit FIFO.
	unsigned int		rxfifosize; // Size of the receive FIFO.
	enum lpi2c_imx_mode	mode; // Current operating mode (speed).
	struct i2c_bus_recovery_info rinfo; // Bus recovery information.
	bool			can_use_dma; // Flag to indicate if DMA is available/enabled.
	struct lpi2c_imx_dma	*dma; // Pointer to DMA specific information.
	struct i2c_client	*target; // The registered I2C slave client, if any.
};

/**
 * @brief Macro to poll the LPI2C Master Status Register (MSR) with a timeout.
 * Provides a robust way to wait for certain status bits to be set.
 * @param val Variable to store the MSR value.
 * @param cond Condition to check against the MSR value.
 */
#define lpi2c_imx_read_msr_poll_timeout(val, cond)                            \
		  readl_poll_timeout(lpi2c_imx->base + LPI2C_MSR, val, cond,  \
				     0, 500000) // Poll for 500ms.

/**
 * @brief Enables or disables LPI2C master interrupts.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @param enable Bitmask of interrupts to enable in LPI2C_MIER.
 */
static void lpi2c_imx_intctrl(struct lpi2c_imx_struct *lpi2c_imx,
			      unsigned int enable)
{
	writel(enable, lpi2c_imx->base + LPI2C_MIER);
}

/**
 * @brief Checks if the I2C bus is busy and handles arbitration loss.
 * Polls the Master Status Register (MSR) for bus busy flags or arbitration lost.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @return 0 on success (bus not busy and no arbitration lost), -EAGAIN on arbitration loss, -ETIMEDOUT on timeout.
 */
static int lpi2c_imx_bus_busy(struct lpi2c_imx_struct *lpi2c_imx)
{
	unsigned int temp;
	int err;

	// Poll MSR until bus busy or arbitration lost flags are cleared.
	err = lpi2c_imx_read_msr_poll_timeout(temp,
					      temp & (MSR_ALF | MSR_BBF | MSR_MBF));

	// Pre-condition: Check for arbitration lost.
	if (temp & MSR_ALF) {
		// Invariant: Clear arbitration lost flag and return error.
		writel(temp, lpi2c_imx->base + LPI2C_MSR);
		return -EAGAIN; // Arbitration lost.
	}

	// Pre-condition: Check if poll timed out (bus still busy).
	if (err) {
		// Invariant: Log error, attempt bus recovery if supported, and return timeout error.
		dev_dbg(&lpi2c_imx->adapter.dev, "bus not work\n");
		if (lpi2c_imx->adapter.bus_recovery_info)
			i2c_recover_bus(&lpi2c_imx->adapter); // Attempt I2C bus recovery.
		return -ETIMEDOUT; // Bus timeout.
	}

	return 0; // Bus is not busy.
}

/**
 * @brief Reads the current number of entries in the transmit FIFO.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @return The number of entries in the transmit FIFO.
 */
static u32 lpi2c_imx_txfifo_cnt(struct lpi2c_imx_struct *lpi2c_imx)
{
	// Read Master FIFO Status Register (MFSR) and mask for TX FIFO count.
	return readl(lpi2c_imx->base + LPI2C_MFSR) & 0xff;
}

/**
 * @brief Determines the LPI2C operating mode (speed grade) based on the desired bitrate.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 */
static void lpi2c_imx_set_mode(struct lpi2c_imx_struct *lpi2c_imx)
{
	unsigned int bitrate = lpi2c_imx->bitrate;
	enum lpi2c_imx_mode mode;

	// Block Logic: Determine the I2C mode based on standard I2C speed thresholds.
	if (bitrate < I2C_MAX_FAST_MODE_FREQ)
		mode = STANDARD;
	else if (bitrate < I2C_MAX_FAST_MODE_PLUS_FREQ)
		mode = FAST;
	else if (bitrate < I2C_MAX_HIGH_SPEED_MODE_FREQ)
		mode = FAST_PLUS;
	else if (bitrate < I2C_MAX_ULTRA_FAST_MODE_FREQ)
		mode = HS;
	else
		mode = ULTRA_FAST; // Fallback to Ultra Fast if higher than HS.

	lpi2c_imx->mode = mode; // Store the determined mode.
}

/**
 * @brief Initiates an I2C transfer by generating a START condition and sending the slave address.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @param msgs Pointer to the first I2C message (used to get slave address).
 * @return 0 on success, or an error code from `lpi2c_imx_bus_busy`.
 */
static int lpi2c_imx_start(struct lpi2c_imx_struct *lpi2c_imx,
			   struct i2c_msg *msgs)
{
	unsigned int temp;

	// Clear RX and TX FIFOs.
	temp = readl(lpi2c_imx->base + LPI2C_MCR);
	temp |= MCR_RRF | MCR_RTF;
	writel(temp, lpi2c_imx->base + LPI2C_MCR);
	// Clear all status flags by writing 0x7f00.
	writel(0x7f00, lpi2c_imx->base + LPI2C_MSR);

	// Invariant: Construct the command to generate START and send 8-bit slave address.
	temp = i2c_8bit_addr_from_msg(msgs) | (GEN_START << 8);
	writel(temp, lpi2c_imx->base + LPI2C_MTDR); // Write command to MTDR.

	return lpi2c_imx_bus_busy(lpi2c_imx); // Wait for bus to become free.
}

/**
 * @brief Generates an I2C STOP condition.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 */
static void lpi2c_imx_stop(struct lpi2c_imx_struct *lpi2c_imx)
{
	unsigned int temp;
	int err;

	// Write command to generate STOP condition to MTDR.
	writel(GEN_STOP << 8, lpi2c_imx->base + LPI2C_MTDR);

	// Poll MSR for STOP Detect Flag (MSR_SDF) to confirm STOP.
	err = lpi2c_imx_read_msr_poll_timeout(temp, temp & MSR_SDF);

	// Pre-condition: Check if stop generation timed out.
	if (err) {
		dev_dbg(&lpi2c_imx->adapter.dev, "stop timeout\n");
		if (lpi2c_imx->adapter.bus_recovery_info)
			i2c_recover_bus(&lpi2c_imx->adapter); // Attempt bus recovery.
	}
}

/*
 * @brief Configures the LPI2C master for a specific bitrate.
 * This involves calculating prescalers, clock high/low periods, and filter settings.
 * CLKLO = I2C_CLK_RATIO * CLKHI, SETHOLD = CLKHI, DATAVD = CLKHI/2
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @return 0 on success, -EINVAL if configuration fails.
 */
static int lpi2c_imx_config(struct lpi2c_imx_struct *lpi2c_imx)
{
	u8 prescale, filt, sethold, datavd;
	unsigned int clk_rate, clk_cycle, clkhi, clklo;
	enum lpi2c_imx_pincfg pincfg;
	unsigned int temp;

	lpi2c_imx_set_mode(lpi2c_imx); // Determine the operating mode (speed).

	clk_rate = lpi2c_imx->rate_per; // Get peripheral clock rate.

	// Set filter value based on the I2C mode.
	if (lpi2c_imx->mode == HS || lpi2c_imx->mode == ULTRA_FAST)
		filt = 0; // No filter for high-speed modes.
	else
		filt = 2; // Default filter for standard/fast modes.

	// Block Logic: Calculate prescaler, CLKHI, and CLKLO values.
	// Invariant: Iterate to find a suitable prescaler that results in CLKLO < 64.
	for (prescale = 0; prescale <= 7; prescale++) {
		clk_cycle = clk_rate / ((1 << prescale) * lpi2c_imx->bitrate)
			    - 3 - (filt >> 1); // Calculate clock cycle.
		clkhi = DIV_ROUND_UP(clk_cycle, I2C_CLK_RATIO + 1); // Calculate CLKHI.
		clklo = clk_cycle - clkhi; // Calculate CLKLO.
		if (clklo < 64) // Break if CLKLO is within acceptable range.
			break;
	}

	// Pre-condition: If no suitable prescaler found.
	if (prescale > 7)
		return -EINVAL; // Invalid configuration.

	/* Block Logic: Set MCFGR1: PINCFG, PRESCALE, IGNACK */
	if (lpi2c_imx->mode == ULTRA_FAST)
		pincfg = TWO_PIN_OO; // Open output for ultra-fast mode.
	else
		pincfg = TWO_PIN_OD; // Open drain for other modes.
	temp = prescale | pincfg << 24; // Combine prescaler and pincfg.

	if (lpi2c_imx->mode == ULTRA_FAST)
		temp |= MCFGR1_IGNACK; // Ignore NACK for ultra-fast mode.

	writel(temp, lpi2c_imx->base + LPI2C_MCFGR1); // Write to MCFGR1.

	/* Block Logic: Set MCFGR2: FILTSDA, FILTSCL */
	temp = (filt << 16) | (filt << 24); // Configure SCL and SDA filters.
	writel(temp, lpi2c_imx->base + LPI2C_MCFGR2); // Write to MCFGR2.

	/* Block Logic: Set MCCR: DATAVD, SETHOLD, CLKHI, CLKLO */
	sethold = clkhi; // Set hold time.
	datavd = clkhi >> 1; // Set data valid time.
	temp = datavd << 24 | sethold << 16 | clkhi << 8 | clklo; // Combine clock timing values.

	// Write to appropriate MCCR based on I2C mode.
	if (lpi2c_imx->mode == HS)
		writel(temp, lpi2c_imx->base + LPI2C_MCCR1); // Use MCCR1 for high-speed.
	else
		writel(temp, lpi2c_imx->base + LPI2C_MCCR0); // Use MCCR0 for other modes.

	return 0; // Configuration successful.
}

/**
 * @brief Enables the LPI2C master controller.
 * Resets the controller, configures it, and then enables the master module.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @return 0 on success, or an error code if power management or configuration fails.
 */
static int lpi2c_imx_master_enable(struct lpi2c_imx_struct *lpi2c_imx)
{
	unsigned int temp;
	int ret;

	// Runtime PM: Resume and get a usage count.
	ret = pm_runtime_resume_and_get(lpi2c_imx->adapter.dev.parent);
	if (ret < 0)
		return ret;

	// Block Logic: Reset the LPI2C master module.
	temp = MCR_RST; // Assert reset bit.
	writel(temp, lpi2c_imx->base + LPI2C_MCR);
	writel(0, lpi2c_imx->base + LPI2C_MCR); // De-assert reset by writing 0.

	// Configure the I2C timing and pin settings.
	ret = lpi2c_imx_config(lpi2c_imx);
	if (ret)
		goto rpm_put; // Error in configuration, jump to RPM cleanup.

	// Enable the LPI2C master module.
	temp = readl(lpi2c_imx->base + LPI2C_MCR);
	temp |= MCR_MEN; // Set Master Enable bit.
	writel(temp, lpi2c_imx->base + LPI2C_MCR);

	return 0;

rpm_put: // Error handling for PM resume.
	pm_runtime_mark_last_busy(lpi2c_imx->adapter.dev.parent);
	pm_runtime_put_autosuspend(lpi2c_imx->adapter.dev.parent);

	return ret;
}

/**
 * @brief Disables the LPI2C master controller.
 * Clears the master enable bit and then suspends runtime power management.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @return 0 on success.
 */
static int lpi2c_imx_master_disable(struct lpi2c_imx_struct *lpi2c_imx)
{
	u32 temp;

	// Disable the LPI2C master module.
	temp = readl(lpi2c_imx->base + LPI2C_MCR);
	temp &= ~MCR_MEN; // Clear Master Enable bit.
	writel(temp, lpi2c_imx->base + LPI2C_MCR);

	// Runtime PM: Mark last busy and put for autosuspend.
	pm_runtime_mark_last_busy(lpi2c_imx->adapter.dev.parent);
	pm_runtime_put_autosuspend(lpi2c_imx->adapter.dev.parent);

	return 0;
}

/**
 * @brief Waits for the completion of a PIO-based I2C message transfer.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @return 0 on completion, -ETIMEDOUT on timeout.
 */
static int lpi2c_imx_pio_msg_complete(struct lpi2c_imx_struct *lpi2c_imx)
{
	unsigned long time_left;

	// Wait for the completion object to be signaled, with a timeout of HZ (1 second).
	time_left = wait_for_completion_timeout(&lpi2c_imx->complete, HZ);

	return time_left ? 0 : -ETIMEDOUT; // Return 0 if completed, -ETIMEDOUT otherwise.
}

/**
 * @brief Checks if the transmit FIFO is empty and handles NACK detection.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @return 0 on success (FIFO empty), -EIO on NACK detection, -ETIMEDOUT on timeout.
 */
static int lpi2c_imx_txfifo_empty(struct lpi2c_imx_struct *lpi2c_imx)
{
	unsigned int temp;
	int err;

	// Poll MSR until NACK Detect Flag (MSR_NDF) is set or TX FIFO is empty.
	err = lpi2c_imx_read_msr_poll_timeout(temp,
					      (temp & MSR_NDF) || !lpi2c_imx_txfifo_cnt(lpi2c_imx));

	// Pre-condition: Check for NACK detection.
	if (temp & MSR_NDF) {
		dev_dbg(&lpi2c_imx->adapter.dev, "NDF detected\n");
		return -EIO; // NACK detected.
	}

	// Pre-condition: Check if poll timed out (TX FIFO not empty).
	if (err) {
		dev_dbg(&lpi2c_imx->adapter.dev, "txfifo empty timeout\n");
		if (lpi2c_imx->adapter.bus_recovery_info)
			i2c_recover_bus(&lpi2c_imx->adapter); // Attempt bus recovery.
		return -ETIMEDOUT; // Timeout.
	}

	return 0; // Transmit FIFO is empty.
}

/**
 * @brief Sets the transmit FIFO watermark.
 * This determines when the LPI2C will generate a Transmit Data Flag (TDF) interrupt.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 */
static void lpi2c_imx_set_tx_watermark(struct lpi2c_imx_struct *lpi2c_imx)
{
	// Set TX watermark to half the FIFO size in MFCR.
	writel(lpi2c_imx->txfifosize >> 1, lpi2c_imx->base + LPI2C_MFCR);
}

/**
 * @brief Sets the receive FIFO watermark.
 * This determines when the LPI2C will generate a Receive Data Flag (RDF) interrupt.
 * The watermark is adjusted based on remaining message length.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 */
static void lpi2c_imx_set_rx_watermark(struct lpi2c_imx_struct *lpi2c_imx)
{
	unsigned int temp, remaining;

	remaining = lpi2c_imx->msglen - lpi2c_imx->delivered; // Calculate remaining bytes.

	// Set watermark to half RX FIFO size, or 0 if remaining is less.
	if (remaining > (lpi2c_imx->rxfifosize >> 1))
		temp = lpi2c_imx->rxfifosize >> 1;
	else
		temp = 0; // Set to 0 if less than half FIFO size.

	// Write RX watermark to MFCR (shifted by 16 bits).
	writel(temp << 16, lpi2c_imx->base + LPI2C_MFCR);
}

/**
 * @brief Writes data to the LPI2C transmit FIFO.
 * Continuously writes bytes from the transmit buffer until the message is sent
 * or the FIFO is full.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 */
static void lpi2c_imx_write_txfifo(struct lpi2c_imx_struct *lpi2c_imx)
{
	unsigned int data, txcnt;

	// Get current TX FIFO count.
	txcnt = readl(lpi2c_imx->base + LPI2C_MFSR) & 0xff;

	// Block Logic: Fill the TX FIFO until it's full or all data is sent.
	while (txcnt < lpi2c_imx->txfifosize) {
		// Pre-condition: Check if all bytes of the message have been delivered.
		if (lpi2c_imx->delivered == lpi2c_imx->msglen)
			break; // All data sent.

		// Invariant: Write next byte from buffer to MTDR.
		data = lpi2c_imx->tx_buf[lpi2c_imx->delivered++];
		writel(data, lpi2c_imx->base + LPI2C_MTDR);
		txcnt++; // Increment TX FIFO count.
	}

	// Pre-condition: If not all data has been delivered.
	if (lpi2c_imx->delivered < lpi2c_imx->msglen)
		// Invariant: Enable TDF and NDF interrupts to continue writing later.
		lpi2c_imx_intctrl(lpi2c_imx, MIER_TDIE | MIER_NDIE);
	else
		// Invariant: All data delivered, complete the transfer.
		complete(&lpi2c_imx->complete);
}

/**
 * @brief Reads data from the LPI2C receive FIFO.
 * Continuously reads bytes from the receive FIFO until it's empty.
 * Handles SMBus block data length.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 */
static void lpi2c_imx_read_rxfifo(struct lpi2c_imx_struct *lpi2c_imx)
{
	unsigned int blocklen, remaining;
	unsigned int temp, data;

	// Block Logic: Read from RX FIFO until it's empty.
	do {
		data = readl(lpi2c_imx->base + LPI2C_MRDR);
		// Pre-condition: If RXEMPTY flag is set, FIFO is empty.
		if (data & MRDR_RXEMPTY)
			break;

		// Invariant: Store received byte in RX buffer.
		lpi2c_imx->rx_buf[lpi2c_imx->delivered++] = data & 0xff;
	} while (1);

	/*
	 * Block Logic: For SMBus block data read, the first byte received is the length of the remaining packet.
	 * This length needs to be added to the total message length.
	 */
	if (lpi2c_imx->block_data) {
		blocklen = lpi2c_imx->rx_buf[0];
		lpi2c_imx->msglen += blocklen; // Adjust total message length.
	}

	remaining = lpi2c_imx->msglen - lpi2c_imx->delivered; // Calculate remaining bytes to receive.

	// Pre-condition: If no remaining bytes.
	if (!remaining) {
		complete(&lpi2c_imx->complete); // All data received, complete transfer.
		return;
	}

	/* Invariant: Not finished, still waiting for more RX data. */
	lpi2c_imx_set_rx_watermark(lpi2c_imx); // Reset RX watermark.

	/*
	 * Block Logic: Send multiple RECV_DATA commands if block data or chunk-based reception.
	 */
	if (lpi2c_imx->block_data) {
		lpi2c_imx->block_data = 0; // Clear block data flag after processing first byte.
		temp = remaining;
		temp |= (RECV_DATA << 8); // Command to receive remaining data.
		writel(temp, lpi2c_imx->base + LPI2C_MTDR);
	} else if (!(lpi2c_imx->delivered & 0xff)) { // Heuristic: if delivered bytes is a multiple of 256.
		temp = (remaining > CHUNK_DATA ? CHUNK_DATA : remaining) - 1; // Receive up to CHUNK_DATA.
		temp |= (RECV_DATA << 8);
		writel(temp, lpi2c_imx->base + LPI2C_MTDR);
	}

	// Enable RDIE interrupt to continue receiving data.
	lpi2c_imx_intctrl(lpi2c_imx, MIER_RDIE);
}

/**
 * @brief Prepares for an LPI2C master write transfer.
 * Sets the transmit buffer, watermark, and starts writing to the FIFO.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @param msgs Pointer to the I2C message to write.
 */
static void lpi2c_imx_write(struct lpi2c_imx_struct *lpi2c_imx,
			    struct i2c_msg *msgs)
{
	lpi2c_imx->tx_buf = msgs->buf; // Set transmit buffer.
	lpi2c_imx_set_tx_watermark(lpi2c_imx); // Set TX watermark.
	lpi2c_imx_write_txfifo(lpi2c_imx); // Start writing data.
}

/**
 * @brief Prepares for an LPI2C master read transfer.
 * Sets the receive buffer, watermark, and sends the initial RECV_DATA command.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @param msgs Pointer to the I2C message to read.
 */
static void lpi2c_imx_read(struct lpi2c_imx_struct *lpi2c_imx,
			   struct i2c_msg *msgs)
{
	unsigned int temp;

	lpi2c_imx->rx_buf = msgs->buf; // Set receive buffer.
	lpi2c_imx->block_data = msgs->flags & I2C_M_RECV_LEN; // Check for SMBus block read.

	lpi2c_imx_set_rx_watermark(lpi2c_imx); // Set RX watermark.
	// Invariant: Send the initial RECV_DATA command, specifying number of bytes to receive.
	temp = msgs->len > CHUNK_DATA ? CHUNK_DATA - 1 : msgs->len - 1; // Receive up to CHUNK_DATA.
	temp |= (RECV_DATA << 8);
	writel(temp, lpi2c_imx->base + LPI2C_MTDR); // Write command to MTDR.

	// Enable RDIE and NDIE interrupts for receiving data and NACK detection.
	lpi2c_imx_intctrl(lpi2c_imx, MIER_RDIE | MIER_NDIE);
}

/**
 * @brief Determines if DMA should be used for the current I2C message.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @param msg Pointer to the I2C message.
 * @return True if DMA should be used, false otherwise.
 */
static bool is_use_dma(struct lpi2c_imx_struct *lpi2c_imx, struct i2c_msg *msg)
{
	// Pre-condition: Check if DMA is generally enabled for the controller.
	if (!lpi2c_imx->can_use_dma)
		return false; // Cannot use DMA if not enabled.

	/*
	 * Block Logic: When the length of data is less than I2C_DMA_THRESHOLD,
	 * PIO mode is used directly to avoid low performance due to DMA overhead.
	 */
	return !(msg->len < I2C_DMA_THRESHOLD); // Use DMA only if message length exceeds threshold.
}

/**
 * @brief Executes an I2C message transfer using PIO mode.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @param msg Pointer to the I2C message.
 * @return 0 on success, or an error code on timeout.
 */
static int lpi2c_imx_pio_xfer(struct lpi2c_imx_struct *lpi2c_imx,
			      struct i2c_msg *msg)
{
	reinit_completion(&lpi2c_imx->complete); // Reinitialize completion object.

	// Block Logic: Call appropriate read/write function based on message flags.
	if (msg->flags & I2C_M_RD)
		lpi2c_imx_read(lpi2c_imx, msg); // Start I2C read.
	else
		lpi2c_imx_write(lpi2c_imx, msg); // Start I2C write.

	return lpi2c_imx_pio_msg_complete(lpi2c_imx); // Wait for PIO transfer to complete.
}

/**
 * @brief Calculates a timeout value for DMA transfers based on data length and bitrate.
 * Adds extra time for scheduler-related activities.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @return Timeout value in jiffies.
 */
static int lpi2c_imx_dma_timeout_calculate(struct lpi2c_imx_struct *lpi2c_imx)
{
	unsigned long time = 0;

	// Calculate estimated transfer time: (bits * 1000ms/s) / (bits/s).
	time = 8 * lpi2c_imx->dma->dma_len * 1000 / lpi2c_imx->bitrate;

	/* Add extra second for scheduler related activities */
	time += 1;

	/* Double calculated time to be safe. */
	return secs_to_jiffies(time); // Convert milliseconds to jiffies.
}

/**
 * @brief Allocates and builds the RX command buffer for DMA read transfers.
 * This buffer contains the `RECV_DATA` commands needed to instruct the LPI2C
 * controller how much data to receive.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @return 0 on success, -ENOMEM on memory allocation failure.
 */
static int lpi2c_imx_alloc_rx_cmd_buf(struct lpi2c_imx_struct *lpi2c_imx)
{
	struct lpi2c_imx_dma *dma = lpi2c_imx->dma;
	u16 rx_remain = dma->dma_len; // Remaining bytes to receive.
	int cmd_num; // Number of RX command words.
	u16 temp;

	/*
	 * Block Logic: Calculate the number of rx command words needed, based on
	 * `CHUNK_DATA` (256 bytes per command) and the total `dma_len`.
	 * Then allocate and populate the `rx_cmd_buf`.
	 */
	cmd_num = DIV_ROUND_UP(rx_remain, CHUNK_DATA); // Round up to get total commands.
	// Allocate memory for the RX command buffer.
	dma->rx_cmd_buf = kcalloc(cmd_num, sizeof(u16), GFP_KERNEL);
	dma->rx_cmd_buf_len = cmd_num * sizeof(u16); // Store the buffer length.

	// Pre-condition: Check for memory allocation failure.
	if (!dma->rx_cmd_buf) {
		dev_err(&lpi2c_imx->adapter.dev, "Alloc RX cmd buffer failed\n");
		return -ENOMEM;
	}

	// Block Logic: Populate the RX command buffer with RECV_DATA commands.
	for (int i = 0; i < cmd_num ; i++) {
		// Invariant: Each command requests up to CHUNK_DATA bytes.
		temp = rx_remain > CHUNK_DATA ? CHUNK_DATA - 1 : rx_remain - 1; // Bytes to receive for this command.
		temp |= (RECV_DATA << 8); // Add RECV_DATA command.
		rx_remain -= CHUNK_DATA; // Decrement remaining bytes.
		dma->rx_cmd_buf[i] = temp; // Store the command word.
	}

	return 0; // Allocation and population successful.
}

/**
 * @brief Waits for the completion of a DMA-based I2C message transfer.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @return 0 on completion, -ETIMEDOUT on timeout.
 */
static int lpi2c_imx_dma_msg_complete(struct lpi2c_imx_struct *lpi2c_imx)
{
	unsigned long time_left, time;

	time = lpi2c_imx_dma_timeout_calculate(lpi2c_imx); // Calculate dynamic timeout.
	// Wait for the completion object to be signaled, with calculated timeout.
	time_left = wait_for_completion_timeout(&lpi2c_imx->complete, time);
	// Pre-condition: Check if timeout occurred.
	if (time_left == 0) {
		dev_err(&lpi2c_imx->adapter.dev, "I/O Error in DMA Data Transfer\n");
		return -ETIMEDOUT; // Timeout.
	}

	return 0; // DMA transfer completed.
}

/**
 * @brief Unmaps a DMA buffer from the DMA controller.
 * @param dma Pointer to the DMA information structure.
 */
static void lpi2c_dma_unmap(struct lpi2c_imx_dma *dma)
{
	struct dma_chan *chan = dma->dma_data_dir == DMA_FROM_DEVICE
				? dma->chan_rx : dma->chan_tx; // Select appropriate channel.

	// Unmap the DMA buffer.
	dma_unmap_single(chan->device->dev, dma->dma_addr,
			 dma->dma_len, dma->dma_data_dir);

	dma->dma_data_dir = DMA_NONE; // Reset data direction flag.
}

/**
 * @brief Cleans up (terminates and unmaps) the RX command DMA transfer.
 * Used when an RX DMA transfer needs to be aborted or has completed.
 * @param dma Pointer to the DMA information structure.
 */
static void lpi2c_cleanup_rx_cmd_dma(struct lpi2c_imx_dma *dma)
{
	dmaengine_terminate_sync(dma->chan_tx); // Terminate DMA transfer on TX channel.
	// Unmap the RX command buffer.
	dma_unmap_single(dma->chan_tx->device->dev, dma->dma_tx_addr,
			 dma->rx_cmd_buf_len, DMA_TO_DEVICE);
}

/**
 * @brief Cleans up (terminates and unmaps) the main data DMA transfer (TX or RX).
 * @param dma Pointer to the DMA information structure.
 */
static void lpi2c_cleanup_dma(struct lpi2c_imx_dma *dma)
{
	// Terminate DMA transfer on the relevant channel based on data direction.
	if (dma->dma_data_dir == DMA_FROM_DEVICE)
		dmaengine_terminate_sync(dma->chan_rx);
	else if (dma->dma_data_dir == DMA_TO_DEVICE)
		dmaengine_terminate_sync(dma->chan_tx);

	lpi2c_dma_unmap(dma); // Unmap the main data DMA buffer.
}

/**
 * @brief DMA callback function executed upon completion of a DMA transfer.
 * Signals the completion object to wake up waiting threads.
 * @param data Pointer to the `lpi2c_imx_struct` for the associated I2C controller.
 */
static void lpi2c_dma_callback(void *data)
{
	struct lpi2c_imx_struct *lpi2c_imx = (struct lpi2c_imx_struct *)data;

	complete(&lpi2c_imx->complete); // Signal completion.
}

/**
 * @brief Submits the RX command DMA transfer.
 * This sets up a DMA transfer to write the RX command words (from `rx_cmd_buf`)
 * to the LPI2C_MTDR register.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @return 0 on success, -EINVAL on various DMA submission failures.
 */
static int lpi2c_dma_rx_cmd_submit(struct lpi2c_imx_struct *lpi2c_imx)
{
	struct dma_async_tx_descriptor *rx_cmd_desc;
	struct lpi2c_imx_dma *dma = lpi2c_imx->dma;
	struct dma_chan *txchan = dma->chan_tx; // Use TX channel for writing commands.
	dma_cookie_t cookie;

	// Map the RX command buffer for DMA transfer.
	dma->dma_tx_addr = dma_map_single(txchan->device->dev,
					  dma->rx_cmd_buf, dma->rx_cmd_buf_len,
					  DMA_TO_DEVICE);
	// Pre-condition: Check for DMA mapping error.
	if (dma_mapping_error(txchan->device->dev, dma->dma_tx_addr)) {
		dev_err(&lpi2c_imx->adapter.dev, "DMA map failed, use pio\n");
		return -EINVAL; // Fallback to PIO.
	}

	// Prepare a single-shot DMA transfer descriptor for slave mode (memory to device).
	rx_cmd_desc = dmaengine_prep_slave_single(txchan, dma->dma_tx_addr,
						  dma->rx_cmd_buf_len, DMA_MEM_TO_DEV,
						  DMA_PREP_INTERRUPT | DMA_CTRL_ACK);
	// Pre-condition: Check for descriptor preparation failure.
	if (!rx_cmd_desc) {
		dev_err(&lpi2c_imx->adapter.dev, "DMA prep slave sg failed, use pio\n");
		goto desc_prepare_err_exit; // Clean up and exit.
	}

	cookie = dmaengine_submit(rx_cmd_desc); // Submit the DMA descriptor.
	// Pre-condition: Check for DMA submission failure.
	if (dma_submit_error(cookie)) {
		dev_err(&lpi2c_imx->adapter.dev, "submitting DMA failed, use pio\n");
		goto submit_err_exit; // Clean up and exit.
	}

	dma_async_issue_pending(txchan); // Issue pending DMA requests.

	return 0; // Submission successful.

desc_prepare_err_exit: // Error path for descriptor preparation.
	dma_unmap_single(txchan->device->dev, dma->dma_tx_addr,
			 dma->rx_cmd_buf_len, DMA_TO_DEVICE);
	return -EINVAL;

submit_err_exit: // Error path for DMA submission.
	dma_unmap_single(txchan->device->dev, dma->dma_tx_addr,
			 dma->rx_cmd_buf_len, DMA_TO_DEVICE);
	dmaengine_desc_free(rx_cmd_desc); // Free descriptor.
	return -EINVAL;
}

/**
 * @brief Submits the main data DMA transfer (TX or RX).
 * Maps the data buffer, prepares the DMA descriptor, sets the callback,
 * and submits the transfer.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @return 0 on success, -EINVAL on various DMA submission failures.
 */
static int lpi2c_dma_submit(struct lpi2c_imx_struct *lpi2c_imx)
{
	struct lpi2c_imx_dma *dma = lpi2c_imx->dma;
	struct dma_async_tx_descriptor *desc;
	struct dma_chan *chan;
	dma_cookie_t cookie;

	// Block Logic: Determine DMA channel and data direction based on message flag.
	if (dma->dma_msg_flag & I2C_M_RD) {
		chan = dma->chan_rx;
		dma->dma_data_dir = DMA_FROM_DEVICE; // Receiving data from device.
		dma->dma_transfer_dir = DMA_DEV_TO_MEM;
	} else {
		chan = dma->chan_tx;
		dma->dma_data_dir = DMA_TO_DEVICE; // Transmitting data to device.
		dma->dma_transfer_dir = DMA_MEM_TO_DEV;
	}

	// Map the main data buffer for DMA transfer.
	dma->dma_addr = dma_map_single(chan->device->dev,
				       dma->dma_buf, dma->dma_len, dma->dma_data_dir);
	// Pre-condition: Check for DMA mapping error.
	if (dma_mapping_error(chan->device->dev, dma->dma_addr)) {
		dev_err(&lpi2c_imx->adapter.dev, "DMA map failed, use pio\n");
		return -EINVAL; // Fallback to PIO.
	}

	// Prepare a single-shot DMA transfer descriptor for slave mode.
	desc = dmaengine_prep_slave_single(chan, dma->dma_addr,
					   dma->dma_len, dma->dma_transfer_dir,
					   DMA_PREP_INTERRUPT | DMA_CTRL_ACK);
	// Pre-condition: Check for descriptor preparation failure.
	if (!desc) {
		dev_err(&lpi2c_imx->adapter.dev, "DMA prep slave sg failed, use pio\n");
		goto desc_prepare_err_exit; // Clean up and exit.
	}

	reinit_completion(&lpi2c_imx->complete); // Reinitialize completion object for this transfer.
	desc->callback = lpi2c_dma_callback; // Set DMA completion callback.
	desc->callback_param = lpi2c_imx;

	cookie = dmaengine_submit(desc); // Submit the DMA descriptor.
	// Pre-condition: Check for DMA submission failure.
	if (dma_submit_error(cookie)) {
		dev_err(&lpi2c_imx->adapter.dev, "submitting DMA failed, use pio\n");
		goto submit_err_exit; // Clean up and exit.
	}

	/* Invariant: DMA transfer has started, so cannot switch to PIO mode. */
	dma->using_pio_mode = false;

	dma_async_issue_pending(chan); // Issue pending DMA requests.

	return 0; // Submission successful.

desc_prepare_err_exit: // Error path for descriptor preparation.
	lpi2c_dma_unmap(dma); // Unmap DMA buffer.
	return -EINVAL;

submit_err_exit: // Error path for DMA submission.
	lpi2c_dma_unmap(dma); // Unmap DMA buffer.
	dmaengine_desc_free(desc); // Free descriptor.
	return -EINVAL;
}

/**
 * @brief Finds the maximum burst number for DMA efficiency based on FIFO size and data length.
 * The burst number should be a divisor of the data length.
 * @param fifosize Size of the FIFO.
 * @param len Total length of the data.
 * @return The calculated maximum burst number.
 */
static int lpi2c_imx_find_max_burst_num(unsigned int fifosize, unsigned int len)
{
	unsigned int i;

	// Invariant: Iterate downwards from half the FIFO size to find the largest divisor of `len`.
	for (i = fifosize / 2; i > 0; i--)
		if (!(len % i)) // Check if `i` divides `len` evenly.
			break;

	return i;
}

/*
 * @brief Calculates the transmit and receive DMA burst numbers for optimal FIFO usage.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 */
static void lpi2c_imx_dma_burst_num_calculate(struct lpi2c_imx_struct *lpi2c_imx)
{
	struct lpi2c_imx_dma *dma = lpi2c_imx->dma;
	unsigned int cmd_num;

	// Block Logic: If it's a read message, calculate burst numbers for both TX (commands) and RX (data).
	if (dma->dma_msg_flag & I2C_M_RD) {
		/*
		 * Invariant: One RX command word can trigger DMA receive no more than 256 bytes.
		 * The number of RX cmd words should be calculated based on the data length.
		 */
		cmd_num = DIV_ROUND_UP(dma->dma_len, CHUNK_DATA); // Number of RX commands needed.
		// Calculate TX burst number for RX commands (written to MTDR).
		dma->tx_burst_num = lpi2c_imx_find_max_burst_num(lpi2c_imx->txfifosize,
								 cmd_num);
		// Calculate RX burst number for actual data (read from MRDR).
		dma->rx_burst_num = lpi2c_imx_find_max_burst_num(lpi2c_imx->rxfifosize,
								 dma->dma_len);
	} else { // Block Logic: If it's a write message, only TX burst number for data is needed.
		// Calculate TX burst number for data (written to MTDR).
		dma->tx_burst_num = lpi2c_imx_find_max_burst_num(lpi2c_imx->txfifosize,
								 dma->dma_len);
	}
}

/**
 * @brief Configures the DMA engine slave parameters for I2C transfers.
 * Sets up source/destination addresses, data widths, and burst sizes for DMA channels.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @return 0 on success, or an error code on configuration failure.
 */
static int lpi2c_dma_config(struct lpi2c_imx_struct *lpi2c_imx)
{
	struct lpi2c_imx_dma *dma = lpi2c_imx->dma;
	struct dma_slave_config rx = {}, tx = {}; // DMA slave configuration structures.
	int ret;

	lpi2c_imx_dma_burst_num_calculate(lpi2c_imx); // Calculate optimal burst numbers.

	// Block Logic: Configure DMA channels based on read or write message.
	if (dma->dma_msg_flag & I2C_M_RD) { // If it's an I2C read operation.
		// Configure TX channel for sending RX commands to LPI2C_MTDR.
		tx.dst_addr = dma->phy_addr + LPI2C_MTDR;
		tx.dst_addr_width = DMA_SLAVE_BUSWIDTH_2_BYTES; // Commands are 2 bytes (16-bit).
		tx.dst_maxburst = dma->tx_burst_num;
		tx.direction = DMA_MEM_TO_DEV;
		ret = dmaengine_slave_config(dma->chan_tx, &tx);
		if (ret < 0)
			return ret;

		// Configure RX channel for receiving data from LPI2C_MRDR.
		rx.src_addr = dma->phy_addr + LPI2C_MRDR;
		rx.src_addr_width = DMA_SLAVE_BUSWIDTH_1_BYTE; // Data is 1 byte (8-bit).
		rx.src_maxburst = dma->rx_burst_num;
		rx.direction = DMA_DEV_TO_MEM;
		ret = dmaengine_slave_config(dma->chan_rx, &rx);
		if (ret < 0)
			return ret;
	} else { // If it's an I2C write operation.
		// Configure TX channel for sending data to LPI2C_MTDR.
		tx.dst_addr = dma->phy_addr + LPI2C_MTDR;
		tx.dst_addr_width = DMA_SLAVE_BUSWIDTH_1_BYTE; // Data is 1 byte (8-bit).
		tx.dst_maxburst = dma->tx_burst_num;
		tx.direction = DMA_MEM_TO_DEV;
		ret = dmaengine_slave_config(dma->chan_tx, &tx);
		if (ret < 0)
			return ret;
	}

	return 0; // Configuration successful.
}

/**
 * @brief Enables DMA functionality in the LPI2C controller and sets FIFO watermarks
 * according to calculated DMA burst numbers.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 */
static void lpi2c_dma_enable(struct lpi2c_imx_struct *lpi2c_imx)
{
	struct lpi2c_imx_dma *dma = lpi2c_imx->dma;
	/*
	 * Block Logic: Set TX/RX watermarks to optimize DMA transfers.
	 * TX interrupt will be triggered when the number of words in
	 * the transmit FIFO is equal or less than TX watermark.
	 * RX interrupt will be triggered when the number of words in
	 * the receive FIFO is greater than RX watermark.
	 * In order to trigger the DMA interrupt, TX watermark should be
	 * set equal to the DMA TX burst number but RX watermark should
	 * be set less than the DMA RX burst number.
	 */
	if (dma->dma_msg_flag & I2C_M_RD) { // If I2C read operation.
		/* Set I2C TX/RX watermark */
		// TX watermark = tx_burst_num, RX watermark = rx_burst_num - 1.
		writel(dma->tx_burst_num | (dma->rx_burst_num - 1) << 16,
		       lpi2c_imx->base + LPI2C_MFCR);
		/* Enable I2C DMA TX/RX function */
		writel(MDER_TDDE | MDER_RDDE, lpi2c_imx->base + LPI2C_MDER); // Enable both TX and RX DMA.
	} else { // If I2C write operation.
		/* Set I2C TX watermark */
		// TX watermark = tx_burst_num.
		writel(dma->tx_burst_num, lpi2c_imx->base + LPI2C_MFCR);
		/* Enable I2C DMA TX function */
		writel(MDER_TDDE, lpi2c_imx->base + LPI2C_MDER); // Enable only TX DMA.
	}

	/* Enable NACK detected interrupt. */
	lpi2c_imx_intctrl(lpi2c_imx, MIER_NDIE);
};

/*
 * @brief Handles an I2C message transfer using DMA mode.
 * This is a critical function for DMA-based I2C communication. It sets up
 * DMA channels, buffers, and submits the transfers. For RX, it also handles
 * the submission of RX command words via a TX DMA channel.
 *
 * Background: When lpi2c is in TX DMA mode we can use one DMA TX channel to write
 * data word into TXFIFO, but in RX DMA mode it is different.
 * The LPI2C MTDR register is a command data and transmit data register.
 * Bits 8-10 are the command data field and Bits 0-7 are the transmit
 * data field. When the LPI2C master needs to read data, the number of
 * bytes to read should be set in the command field and RECV_DATA should
 * be set into the command data field to receive (DATA[7:0] + 1) bytes.
 * The recv data command word is made of RECV_DATA in the command data
 * field and the number of bytes to read in transmit data field. When the
 * length of data to be read exceeds 256 bytes, recv data command word
 * needs to be written to TXFIFO multiple times.
 * So when in RX DMA mode, the TX channel also must to be configured to
 * send RX command words and the RX command word must be set in advance
 * before transmitting.
 *
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @param msg Pointer to the I2C message to transfer.
 * @return 0 on success, or an error code on failure.
 */
static int lpi2c_imx_dma_xfer(struct lpi2c_imx_struct *lpi2c_imx,
			      struct i2c_msg *msg)
{
	struct lpi2c_imx_dma *dma = lpi2c_imx->dma;
	int ret;

	/* Invariant: Assume PIO fallback if DMA fails. */
	dma->using_pio_mode = true;

	dma->dma_len = msg->len; // Set DMA data length.
	dma->dma_msg_flag = msg->flags; // Set DMA message flags.
	// Get a DMA-safe buffer for the message data.
	dma->dma_buf = i2c_get_dma_safe_msg_buf(msg, I2C_DMA_THRESHOLD);
	// Pre-condition: Check for buffer allocation failure.
	if (!dma->dma_buf)
		return -ENOMEM;

	// Configure DMA channels with burst numbers.
	ret = lpi2c_dma_config(lpi2c_imx);
	if (ret) {
		dev_err(&lpi2c_imx->adapter.dev, "Failed to configure DMA (%d)\n", ret);
		goto disable_dma; // Clean up and disable DMA.
	}

	lpi2c_dma_enable(lpi2c_imx); // Enable LPI2C DMA functionality.

	// Submit the main data DMA transfer (TX or RX).
	ret = lpi2c_dma_submit(lpi2c_imx);
	if (ret) {
		dev_err(&lpi2c_imx->adapter.dev, "DMA submission failed (%d)\n", ret);
		goto disable_dma; // Clean up and disable DMA.
	}

	// Block Logic: If it's an I2C read, also submit the RX command DMA transfer.
	if (dma->dma_msg_flag & I2C_M_RD) {
		ret = lpi2c_imx_alloc_rx_cmd_buf(lpi2c_imx); // Allocate RX command buffer.
		if (ret)
			goto disable_cleanup_data_dma; // Clean up data DMA.

		ret = lpi2c_dma_rx_cmd_submit(lpi2c_imx); // Submit RX command DMA.
		if (ret)
			goto disable_cleanup_data_dma; // Clean up data DMA.
	}

	// Wait for the DMA transfer to complete.
	ret = lpi2c_imx_dma_msg_complete(lpi2c_imx);
	if (ret)
		goto disable_cleanup_all_dma; // Clean up all DMA.

	/*
	 * Pre-condition: Check for NACK detected during transfer.
	 * If NACK occurs and no other error, mark as I/O error.
	 */
	if ((readl(lpi2c_imx->base + LPI2C_MSR) & MSR_NDF) && !ret) {
		ret = -EIO;
		goto disable_cleanup_all_dma; // Clean up all DMA.
	}

	// Block Logic: Unmap DMA buffers after successful transfer.
	if (dma->dma_msg_flag & I2C_M_RD)
		dma_unmap_single(dma->chan_tx->device->dev, dma->dma_tx_addr,
				 dma->rx_cmd_buf_len, DMA_TO_DEVICE); // Unmap RX command buffer.
	lpi2c_dma_unmap(dma); // Unmap main data buffer.

	goto disable_dma; // Jump to common DMA disable and cleanup.

disable_cleanup_all_dma: // Error path for cleaning up both data and command DMA.
	if (dma->dma_msg_flag & I2C_M_RD)
		lpi2c_cleanup_rx_cmd_dma(dma);
disable_cleanup_data_dma: // Error path for cleaning up only data DMA.
	lpi2c_cleanup_dma(dma);
disable_dma: // Common DMA disable and buffer cleanup.
	/* Disable I2C DMA function */
	writel(0, lpi2c_imx->base + LPI2C_MDER); // Disable DMA in hardware.

	if (dma->dma_msg_flag & I2C_M_RD)
		kfree(dma->rx_cmd_buf); // Free RX command buffer.

	// Release DMA-safe message buffer, copying back if successful.
	if (ret)
		i2c_put_dma_safe_msg_buf(dma->dma_buf, msg, false); // No data copy.
	else
		i2c_put_dma_safe_msg_buf(dma->dma_buf, msg, true); // Copy data back.

	return ret;
}

/**
 * @brief Main I2C master transfer function, handles a sequence of I2C messages.
 * This function orchestrates the entire I2C transaction, including starting,
 * stopping, and choosing between PIO and DMA modes.
 * @param adapter Pointer to the I2C adapter structure.
 * @param msgs Array of I2C messages to transfer.
 * @param num Number of messages in the array.
 * @return Number of messages successfully transferred, or a negative error code.
 */
static int lpi2c_imx_xfer(struct i2c_adapter *adapter,
			  struct i2c_msg *msgs, int num)
{
	struct lpi2c_imx_struct *lpi2c_imx = i2c_get_adapdata(adapter);
	unsigned int temp;
	int i, result;

	// Enable the LPI2C master controller.
	result = lpi2c_imx_master_enable(lpi2c_imx);
	if (result)
		return result; // Return error if master cannot be enabled.

	// Block Logic: Iterate through each message in the transfer.
	for (i = 0; i < num; i++) {
		// Generate START condition and send slave address for the current message.
		result = lpi2c_imx_start(lpi2c_imx, &msgs[i]);
		if (result)
			goto disable; // Error starting, disable master and exit.

		/* Invariant: Handle quick SMBus (zero-length message). */
		if (num == 1 && msgs[0].len == 0)
			goto stop; // For quick SMBus, only START/STOP without data.

		// Initialize transfer-specific variables.
		lpi2c_imx->rx_buf = NULL;
		lpi2c_imx->tx_buf = NULL;
		lpi2c_imx->delivered = 0;
		lpi2c_imx->msglen = msgs[i].len;
		init_completion(&lpi2c_imx->complete); // Reinitialize completion for current message.

		// Block Logic: Choose between DMA and PIO for data transfer.
		if (is_use_dma(lpi2c_imx, &msgs[i])) {
			result = lpi2c_imx_dma_xfer(lpi2c_imx, &msgs[i]); // Attempt DMA transfer.
			// Pre-condition: If DMA failed but can fallback to PIO.
			if (result && lpi2c_imx->dma->using_pio_mode)
				result = lpi2c_imx_pio_xfer(lpi2c_imx, &msgs[i]); // Fallback to PIO.
		} else {
			result = lpi2c_imx_pio_xfer(lpi2c_imx, &msgs[i]); // Use PIO transfer directly.
		}

		if (result)
			goto stop; // Error during transfer, go to stop.

		// Block Logic: For write transfers, ensure TX FIFO is empty.
		if (!(msgs[i].flags & I2C_M_RD)) {
			result = lpi2c_imx_txfifo_empty(lpi2c_imx);
			if (result)
				goto stop; // Error, TX FIFO not empty.
		}
	}

stop: // Label to generate STOP condition.
	lpi2c_imx_stop(lpi2c_imx); // Generate I2C STOP.

	// Pre-condition: Check for NACK detected after STOP, if no other error.
	temp = readl(lpi2c_imx->base + LPI2C_MSR);
	if ((temp & MSR_NDF) && !result)
		result = -EIO; // NACK detected.

disable: // Label to disable master.
	lpi2c_imx_master_disable(lpi2c_imx); // Disable the LPI2C master.

	// Log exit status.
	dev_dbg(&lpi2c_imx->adapter.dev, "<%s> exit with: %s: %d\n", __func__,
		(result < 0) ? "error" : "success msg",
		(result < 0) ? result : num);

	// Return number of messages transferred or error.
	return (result < 0) ? result : num;
}

/**
 * @brief Interrupt service routine (ISR) for LPI2C slave mode.
 * Handles various slave events such as address match, read/write requests,
 * transmit/receive data, and stop conditions.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @param ssr Current Slave Status Register value.
 * @param sier_filter Filtered slave interrupt enable bits.
 * @return IRQ_HANDLED to indicate the interrupt was processed.
 */
static irqreturn_t lpi2c_imx_target_isr(struct lpi2c_imx_struct *lpi2c_imx,
					u32 ssr, u32 sier_filter)
{
	u8 value;
	u32 sasr;

	/* Pre-condition: Check for Arbitration lost (Bus Error Flag). */
	if (sier_filter & SSR_BEF) {
		writel(0, lpi2c_imx->base + LPI2C_SIER); // Disable all slave interrupts.
		return IRQ_HANDLED;
	}

	/* Pre-condition: Check for Address Valid Flag (slave address matched). */
	if (sier_filter & SSR_AVF) {
		sasr = readl(lpi2c_imx->base + LPI2C_SASR); // Read Slave Address Status Register.
		if (SASR_READ_REQ & sasr) { // Pre-condition: If master is requesting a read.
			// Invariant: Trigger I2C_SLAVE_READ_REQUESTED event and write data to STDR.
			i2c_slave_event(lpi2c_imx->target, I2C_SLAVE_READ_REQUESTED, &value);
			writel(value, lpi2c_imx->base + LPI2C_STDR);
			goto ret; // Jump to clear SSR.
		} else { // Pre-condition: Master is requesting a write.
			// Invariant: Trigger I2C_SLAVE_WRITE_REQUESTED event.
			i2c_slave_event(lpi2c_imx->target, I2C_SLAVE_WRITE_REQUESTED, &value);
		}
	}

	// Pre-condition: Check for STOP Detect Flag.
	if (sier_filter & SSR_SDF)
		// Invariant: Trigger I2C_SLAVE_STOP event.
		i2c_slave_event(lpi2c_imx->target, I2C_SLAVE_STOP, &value);

	// Pre-condition: Check for Transmit Data Flag (master read, slave sending data).
	if (sier_filter & SSR_TDF) {
		// Invariant: Trigger I2C_SLAVE_READ_PROCESSED event and write next data to STDR.
		i2c_slave_event(lpi2c_imx->target, I2C_SLAVE_READ_PROCESSED, &value);
		writel(value, lpi2c_imx->base + LPI2C_STDR);
	}

	// Pre-condition: Check for Receive Data Flag (master write, slave receiving data).
	if (sier_filter & SSR_RDF) {
		// Invariant: Read received byte from SRDR and trigger I2C_SLAVE_WRITE_RECEIVED event.
		value = readl(lpi2c_imx->base + LPI2C_SRDR);
		i2c_slave_event(lpi2c_imx->target, I2C_SLAVE_WRITE_RECEIVED, &value);
	}

ret: // Label to clear status bits.
	/* Clear relevant SSR bits by writing them back. */
	writel(ssr & SSR_CLEAR_BITS, lpi2c_imx->base + LPI2C_SSR);
	return IRQ_HANDLED;
}

/**
 * @brief Interrupt service routine (ISR) for LPI2C master mode.
 * Handles various master events such as NACK detection, receive data,
 * and transmit data.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @return IRQ_HANDLED to indicate the interrupt was processed.
 */
static irqreturn_t lpi2c_imx_master_isr(struct lpi2c_imx_struct *lpi2c_imx)
{
	unsigned int enabled;
	unsigned int temp;

	enabled = readl(lpi2c_imx->base + LPI2C_MIER); // Read enabled interrupts.

	lpi2c_imx_intctrl(lpi2c_imx, 0); // Disable all master interrupts initially.
	temp = readl(lpi2c_imx->base + LPI2C_MSR); // Read Master Status Register.
	temp &= enabled; // Filter status bits by enabled interrupts.

	// Block Logic: Handle specific master interrupt events.
	if (temp & MSR_NDF) // Pre-condition: NACK detected.
		complete(&lpi2c_imx->complete); // Invariant: Complete the transfer (indicates NACK occurred).
	else if (temp & MSR_RDF) // Pre-condition: Receive Data Flag (data in RX FIFO).
		lpi2c_imx_read_rxfifo(lpi2c_imx); // Invariant: Read data from RX FIFO.
	else if (temp & MSR_TDF) // Pre-condition: Transmit Data Flag (TX FIFO ready for more data).
		lpi2c_imx_write_txfifo(lpi2c_imx); // Invariant: Write more data to TX FIFO.

	return IRQ_HANDLED;
}

/**
 * @brief Combined interrupt service routine (ISR) for LPI2C, dispatching to
 * master or slave specific handlers.
 * @param irq The interrupt number.
 * @param dev_id Pointer to the `lpi2c_imx_struct` for the associated I2C controller.
 * @return IRQ_HANDLED if the interrupt was processed, IRQ_NONE otherwise.
 */
static irqreturn_t lpi2c_imx_isr(int irq, void *dev_id)
{
	struct lpi2c_imx_struct *lpi2c_imx = dev_id;

	// Block Logic: If an I2C slave is registered and enabled, check for slave interrupts.
	if (lpi2c_imx->target) {
		u32 scr = readl(lpi2c_imx->base + LPI2C_SCR); // Read Slave Control Register.
		u32 ssr = readl(lpi2c_imx->base + LPI2C_SSR); // Read Slave Status Register.
		// Filter SSR by enabled slave interrupts.
		u32 sier_filter = ssr & readl(lpi2c_imx->base + LPI2C_SIER);

		/*
		 * Pre-condition: The target is enabled (SCR_SEN) and an interrupt has been triggered.
		 * Invariant: Enter the target's irq handler.
		 */
		if ((scr & SCR_SEN) && sier_filter)
			return lpi2c_imx_target_isr(lpi2c_imx, ssr, sier_filter);
	}

	/*
	 * Invariant: Otherwise, the interrupt has been triggered by the master.
	 * Enter the master's irq handler.
	 */
	return lpi2c_imx_master_isr(lpi2c_imx);
}

/**
 * @brief Initializes the LPI2C controller to operate in slave mode.
 * Resets the slave module, sets the slave address, configures filters,
 * and enables the slave module and its interrupts.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 */
static void lpi2c_imx_target_init(struct lpi2c_imx_struct *lpi2c_imx)
{
	u32 temp;

	/* Block Logic: Reset target module. */
	writel(SCR_RST, lpi2c_imx->base + LPI2C_SCR); // Assert reset.
	writel(0, lpi2c_imx->base + LPI2C_SCR); // De-assert reset.

	/* Set target address from the registered client. */
	writel((lpi2c_imx->target->addr << 1), lpi2c_imx->base + LPI2C_SAMR); // Slave Address Match Register.

	// Configure SCFGR1 for RX and TX data stalls.
	writel(SCFGR1_RXSTALL | SCFGR1_TXDSTALL, lpi2c_imx->base + LPI2C_SCFGR1);

	/*
	 * Block Logic: Set SCFGR2: FILTSDA, FILTSCL and CLKHOLD.
	 * FILTSCL/FILTSDA can eliminate signal skew. It should generally be
	 * set to the same value and should be set >= 50ns.
	 * CLKHOLD is only used when clock stretching is enabled, but it will
	 * extend the clock stretching to ensure there is an additional delay
	 * between the target driving SDA and the target releasing the SCL pin.
	 * CLKHOLD setting is crucial for lpi2c target. When master read data
	 * from target, if there is a delay caused by cpu idle, excessive load,
	 * or other delays between two bytes in one message transmission, it
	 * will cause a short interval time between the driving SDA signal and
	 * releasing SCL signal. The lpi2c master will mistakenly think it is a stop
	 * signal resulting in an arbitration failure. This issue can be avoided
	 * by setting CLKHOLD.
	 * In order to ensure lpi2c function normally when the lpi2c speed is as
	 * low as 100kHz, CLKHOLD should be set to 3 and it is also compatible with
	 * higher clock frequency like 400kHz and 1MHz.
	 */
	temp = SCFGR2_FILTSDA(2) | SCFGR2_FILTSCL(2) | SCFGR2_CLKHOLD(3); // Configure filters and clock hold.
	writel(temp, lpi2c_imx->base + LPI2C_SCFGR2);

	/*
	 * Block Logic: Enable module: SCR_FILTEN can enable digital filter and
	 * output delay counter for LPI2C target mode. So SCR_FILTEN need be
	 * asserted when enable SDA/SCL FILTER and CLKHOLD.
	 */
	writel(SCR_SEN | SCR_FILTEN, lpi2c_imx->base + LPI2C_SCR); // Enable slave and filter.

	/* Enable relevant interrupts from i2c module for slave operation. */
	writel(SLAVE_INT_FLAG, lpi2c_imx->base + LPI2C_SIER);
}

/**
 * @brief Registers an I2C client as a slave with this LPI2C adapter.
 * Sets the `target` client and initializes the hardware for slave mode.
 * @param client Pointer to the I2C client structure to register.
 * @return 0 on success, -EBUSY if a target is already registered, or other error codes.
 */
static int lpi2c_imx_register_target(struct i2c_client *client)
{
	struct lpi2c_imx_struct *lpi2c_imx = i2c_get_adapdata(client->adapter);
	int ret;

	// Pre-condition: Check if an I2C target is already registered.
	if (lpi2c_imx->target)
		return -EBUSY; // Only one target allowed.

	lpi2c_imx->target = client; // Assign the client as the current target.

	// Runtime PM: Resume and get usage count for the parent device.
	ret = pm_runtime_resume_and_get(lpi2c_imx->adapter.dev.parent);
	if (ret < 0) {
		dev_err(&lpi2c_imx->adapter.dev, "failed to resume i2c controller");
		return ret;
	}

	lpi2c_imx_target_init(lpi2c_imx); // Initialize the hardware for slave mode.

	return 0;
}

/**
 * @brief Unregisters an I2C client from being a slave of this LPI2C adapter.
 * Resets the slave hardware and clears the `target` client.
 * @param client Pointer to the I2C client structure to unregister.
 * @return 0 on success, -EINVAL if no target is registered, or other error codes.
 */
static int lpi2c_imx_unregister_target(struct i2c_client *client)
{
	struct lpi2c_imx_struct *lpi2c_imx = i2c_get_adapdata(client->adapter);
	int ret;

	// Pre-condition: Check if any I2C target is registered.
	if (!lpi2c_imx->target)
		return -EINVAL; // No target to unregister.

	/* Reset target address. */
	writel(0, lpi2c_imx->base + LPI2C_SAMR); // Clear slave address.

	// Reset the slave module.
	writel(SCR_RST, lpi2c_imx->base + LPI2C_SCR);
	writel(0, lpi2c_imx->base + LPI2C_SCR);

	lpi2c_imx->target = NULL; // Clear the target client.

	// Runtime PM: Put usage count for the parent device.
	ret = pm_runtime_put_sync(lpi2c_imx->adapter.dev.parent);
	if (ret < 0)
		dev_err(&lpi2c_imx->adapter.dev, "failed to suspend i2c controller");

	return ret;
}

/**
 * @brief Initializes I2C bus recovery information.
 * Retrieves pinctrl handles for bus recovery.
 * @param lpi2c_imx Pointer to the driver's private data structure.
 * @param pdev Pointer to the platform device.
 * @return 0 on success, or an error code if pinctrl cannot be acquired.
 */
static int lpi2c_imx_init_recovery_info(struct lpi2c_imx_struct *lpi2c_imx,
				  struct platform_device *pdev)
{
	struct i2c_bus_recovery_info *bri = &lpi2c_imx->rinfo; // Pointer to recovery info.

	// Get pinctrl handle for bus recovery.
	bri->pinctrl = devm_pinctrl_get(&pdev->dev);
	// Pre-condition: Check if pinctrl acquisition failed.
	if (IS_ERR(bri->pinctrl))
		return PTR_ERR(bri->pinctrl); // Return error code.

	lpi2c_imx->adapter.bus_recovery_info = bri; // Assign recovery info to the I2C adapter.

	return 0;
}

/**
 * @brief Releases DMA channels.
 * @param dev Pointer to the device structure.
 * @param dma Pointer to the DMA information structure.
 */
static void dma_exit(struct device *dev, struct lpi2c_imx_dma *dma)
{
	// Release RX DMA channel if it was requested.
	if (dma->chan_rx)
		dma_release_channel(dma->chan_rx);

	// Release TX DMA channel if it was requested.
	if (dma->chan_tx)
		dma_release_channel(dma->chan_tx);

	// Free DMA structure memory.
	devm_kfree(dev, dma);
}

/**
 * @brief Initializes DMA for the LPI2C controller.
 * Requests DMA channels for TX and RX, and allocates memory for the DMA structure.
 * @param dev Pointer to the device structure.
 * @param phy_addr Physical base address of the LPI2C controller.
 * @return 0 on success, -ENOMEM on memory allocation failure, or other error codes.
 */
static int lpi2c_dma_init(struct device *dev, dma_addr_t phy_addr)
{
	struct lpi2c_imx_struct *lpi2c_imx = dev_get_drvdata(dev);
	struct lpi2c_imx_dma *dma;
	int ret;

	// Allocate memory for DMA structure.
	dma = devm_kzalloc(dev, sizeof(*dma), GFP_KERNEL);
	// Pre-condition: Check for memory allocation failure.
	if (!dma)
		return -ENOMEM;

	dma->phy_addr = phy_addr; // Store physical base address.

	/* Prepare for TX DMA: */
	dma->chan_tx = dma_request_chan(dev, "tx"); // Request TX DMA channel.
	// Pre-condition: Check for TX DMA channel request failure.
	if (IS_ERR(dma->chan_tx)) {
		ret = PTR_ERR(dma->chan_tx);
		// Log error if not -ENODEV or -EPROBE_DEFER (expected deferral).
		if (ret != -ENODEV && ret != -EPROBE_DEFER)
			dev_err(dev, "can't request DMA tx channel (%d)\n", ret);
		dma->chan_tx = NULL; // Clear channel pointer.
		goto dma_exit; // Exit, potentially releasing RX channel.
	}

	/* Prepare for RX DMA: */
	dma->chan_rx = dma_request_chan(dev, "rx"); // Request RX DMA channel.
	// Pre-condition: Check for RX DMA channel request failure.
	if (IS_ERR(dma->chan_rx)) {
		ret = PTR_ERR(dma->chan_rx);
		// Log error if not -ENODEV or -EPROBE_DEFER.
		if (ret != -ENODEV && ret != -EPROBE_DEFER)
			dev_err(dev, "can't request DMA rx channel (%d)\n", ret);
		dma->chan_rx = NULL; // Clear channel pointer.
		goto dma_exit; // Exit and release TX channel.
	}

	lpi2c_imx->can_use_dma = true; // Mark DMA as usable.
	lpi2c_imx->dma = dma; // Assign DMA structure to the driver.
	return 0; // DMA initialization successful.

dma_exit: // Error path for DMA initialization.
	dma_exit(dev, dma); // Release any requested DMA channels.
	return ret;
}

/**
 * @brief Returns the I2C functionality flags supported by this adapter.
 * @param adapter Pointer to the I2C adapter structure.
 * @return Bitmask of supported I2C functionalities.
 */
static u32 lpi2c_imx_func(struct i2c_adapter *adapter)
{
	return I2C_FUNC_I2C | I2C_FUNC_SMBUS_EMUL |
		I2C_FUNC_SMBUS_READ_BLOCK_DATA; // Supported functionalities.
}

/**
 * @brief I2C algorithm structure, defining callback functions for I2C operations.
 */
static const struct i2c_algorithm lpi2c_imx_algo = {
	.xfer = lpi2c_imx_xfer, // Master transfer function.
	.functionality = lpi2c_imx_func, // Functionality query.
	.reg_target = lpi2c_imx_register_target, // Slave registration.
	.unreg_target = lpi2c_imx_unregister_target, // Slave unregistration.
};

/**
 * @brief Device tree compatible match table for the LPI2C driver.
 * Used by the kernel to match devices with this driver.
 */
static const struct of_device_id lpi2c_imx_of_match[] = {
	{ .compatible = "fsl,imx7ulp-lpi2c" }, // Compatible string for i.MX7ULP LPI2C.
	{ } // Terminator.
};
MODULE_DEVICE_TABLE(of, lpi2c_imx_of_match); // Exports the match table for module auto-loading.

/**
 * @brief Probe function for the LPI2C platform driver.
 * This function is called when a device matching `of_match_table` is found.
 * It initializes the hardware, clocks, interrupts, and registers the I2C adapter.
 * @param pdev Pointer to the platform device.
 * @return 0 on success, or a negative error code on failure.
 */
static int lpi2c_imx_probe(struct platform_device *pdev)
{
	struct lpi2c_imx_struct *lpi2c_imx;
	struct resource *res; // Resource structure for device memory region.
	dma_addr_t phy_addr; // Physical address of the device.
	unsigned int temp;
	int irq, ret;

	// Allocate and zero-initialize driver private data structure.
	lpi2c_imx = devm_kzalloc(&pdev->dev, sizeof(*lpi2c_imx), GFP_KERNEL);
	// Pre-condition: Check for memory allocation failure.
	if (!lpi2c_imx)
		return -ENOMEM;

	// Map device memory region (registers) into kernel virtual address space.
	lpi2c_imx->base = devm_platform_get_and_ioremap_resource(pdev, 0, &res);
	// Pre-condition: Check for ioremap failure.
	if (IS_ERR(lpi2c_imx->base))
		return PTR_ERR(lpi2c_imx->base);

	irq = platform_get_irq(pdev, 0); // Get interrupt number.
	// Pre-condition: Check for IRQ retrieval failure.
	if (irq < 0)
		return irq;

	// Block Logic: Initialize I2C adapter structure.
	lpi2c_imx->adapter.owner	= THIS_MODULE;
	lpi2c_imx->adapter.algo		= &lpi2c_imx_algo;
	lpi2c_imx->adapter.dev.parent	= &pdev->dev;
	lpi2c_imx->adapter.dev.of_node	= pdev->dev.of_node;
	strscpy(lpi2c_imx->adapter.name, pdev->name,
		sizeof(lpi2c_imx->adapter.name));
	phy_addr = (dma_addr_t)res->start; // Get physical address from resource.

	// Get all clocks associated with the device.
	ret = devm_clk_bulk_get_all(&pdev->dev, &lpi2c_imx->clks);
	// Pre-condition: Check for clock acquisition failure.
	if (ret < 0)
		return dev_err_probe(&pdev->dev, ret, "can't get I2C peripheral clock\n");
	lpi2c_imx->num_clks = ret; // Store number of clocks.

	// Read clock-frequency from device tree, default to standard mode if not found.
	ret = of_property_read_u32(pdev->dev.of_node,
				   "clock-frequency", &lpi2c_imx->bitrate);
	if (ret)
		lpi2c_imx->bitrate = I2C_MAX_STANDARD_MODE_FREQ;

	// Request and enable interrupt handler.
	ret = devm_request_irq(&pdev->dev, irq, lpi2c_imx_isr, IRQF_NO_SUSPEND,
			       pdev->name, lpi2c_imx);
	// Pre-condition: Check for IRQ request failure.
	if (ret)
		return dev_err_probe(&pdev->dev, ret, "can't claim irq %d\n", irq);

	i2c_set_adapdata(&lpi2c_imx->adapter, lpi2c_imx); // Set private data for adapter.
	platform_set_drvdata(pdev, lpi2c_imx); // Set private data for platform device.

	// Prepare and enable all clocks.
	ret = clk_bulk_prepare_enable(lpi2c_imx->num_clks, lpi2c_imx->clks);
	if (ret)
		return ret;

	/*
	 * Lock the parent clock rate to avoid getting parent clock upon
	 * each transfer.
	 */
	ret = devm_clk_rate_exclusive_get(&pdev->dev, lpi2c_imx->clks[0].clk);
	if (ret)
		return dev_err_probe(&pdev->dev, ret,
				     "can't lock I2C peripheral clock rate\n");

	// Get the actual clock rate.
	lpi2c_imx->rate_per = clk_get_rate(lpi2c_imx->clks[0].clk);
	if (!lpi2c_imx->rate_per)
		return dev_err_probe(&pdev->dev, -EINVAL,
				     "can't get I2C peripheral clock rate\n");

	// Block Logic: Configure runtime power management.
	pm_runtime_set_autosuspend_delay(&pdev->dev, I2C_PM_TIMEOUT);
	pm_runtime_use_autosuspend(&pdev->dev);
	pm_runtime_get_noresume(&pdev->dev);
	pm_runtime_set_active(&pdev->dev);
	pm_runtime_enable(&pdev->dev);

	// Read FIFO sizes from hardware.
	temp = readl(lpi2c_imx->base + LPI2C_PARAM);
	lpi2c_imx->txfifosize = 1 << (temp & 0x0f); // Transmit FIFO size.
	lpi2c_imx->rxfifosize = 1 << ((temp >> 8) & 0x0f); // Receive FIFO size.

	/* Init optional bus recovery function. */
	ret = lpi2c_imx_init_recovery_info(lpi2c_imx, pdev);
	/* Invariant: Give it another chance if pinctrl used is not ready yet. */
	if (ret == -EPROBE_DEFER)
		goto rpm_disable; // Defer probe if pinctrl not ready.

	/* Init DMA (optional). */
	ret = lpi2c_dma_init(&pdev->dev, phy_addr);
	if (ret) {
		if (ret == -EPROBE_DEFER)
			goto rpm_disable; // Defer probe if DMA not ready.
		dev_info(&pdev->dev, "use pio mode\n"); // Fallback to PIO if DMA init fails.
	}

	ret = i2c_add_adapter(&lpi2c_imx->adapter); // Register the I2C adapter with the kernel.
	if (ret)
		goto rpm_disable; // Error adding adapter.

	pm_runtime_mark_last_busy(&pdev->dev);
	pm_runtime_put_autosuspend(&pdev->dev); // Put initial usage count for autosuspend.

	dev_info(&lpi2c_imx->adapter.dev, "LPI2C adapter registered\n");

	return 0;

rpm_disable: // Error path for runtime PM disable.
	pm_runtime_dont_use_autosuspend(&pdev->dev);
	pm_runtime_put_sync(&pdev->dev);
	pm_runtime_disable(&pdev->dev);

	return ret;
}

/**
 * @brief Remove function for the LPI2C platform driver.
 * Called when the device is removed. Unregisters the I2C adapter and
 * cleans up runtime power management.
 * @param pdev Pointer to the platform device.
 */
static void lpi2c_imx_remove(struct platform_device *pdev)
{
	struct lpi2c_imx_struct *lpi2c_imx = platform_get_drvdata(pdev);

	i2c_del_adapter(&lpi2c_imx->adapter); // Unregister I2C adapter.

	// Clean up runtime power management.
	pm_runtime_disable(&pdev->dev);
	pm_runtime_dont_use_autosuspend(&pdev->dev);
}

/**
 * @brief Runtime suspend callback for the LPI2C driver.
 * Disables clocks and selects sleep pinctrl state.
 * @param dev Pointer to the device structure.
 * @return 0 on success.
 */
static int __maybe_unused lpi2c_runtime_suspend(struct device *dev)
{
	struct lpi2c_imx_struct *lpi2c_imx = dev_get_drvdata(dev);

	clk_bulk_disable(lpi2c_imx->num_clks, lpi2c_imx->clks); // Disable clocks.
	pinctrl_pm_select_sleep_state(dev); // Select sleep pinctrl state.

	return 0;
}

/**
 * @brief Runtime resume callback for the LPI2C driver.
 * Selects default pinctrl state and enables clocks.
 * @param dev Pointer to the device structure.
 * @return 0 on success, or an error code on clock enable failure.
 */
static int __maybe_unused lpi2c_runtime_resume(struct device *dev)
{
	struct lpi2c_imx_struct *lpi2c_imx = dev_get_drvdata(dev);
	int ret;

	pinctrl_pm_select_default_state(dev); // Select default pinctrl state.
	ret = clk_bulk_enable(lpi2c_imx->num_clks, lpi2c_imx->clks); // Enable clocks.
	// Pre-condition: Check for clock enable failure.
	if (ret) {
		dev_err(dev, "failed to enable I2C clock, ret=%d\n", ret);
		return ret;
	}

	return 0;
}

/**
 * @brief System suspend (no-IRQ) callback for the LPI2C driver.
 * Forces runtime suspend of the device.
 * @param dev Pointer to the device structure.
 * @return 0 on success.
 */
static int __maybe_unused lpi2c_suspend_noirq(struct device *dev)
{
	return pm_runtime_force_suspend(dev);
}

/**
 * @brief System resume (no-IRQ) callback for the LPI2C driver.
 * Forces runtime resume of the device and reinitializes slave if active.
 * @param dev Pointer to the device structure.
 * @return 0 on success.
 */
static int __maybe_unused lpi2c_resume_noirq(struct device *dev)
{
	struct lpi2c_imx_struct *lpi2c_imx = dev_get_drvdata(dev);
	int ret;

	ret = pm_runtime_force_resume(dev); // Force runtime resume.
	if (ret)
		return ret;

	/*
	 * Block Logic: If the I2C module powers down during system suspend,
	 * the register values will be lost. Therefore, reinitialize
	 * the target when the system resumes if it was active.
	 */
	if (lpi2c_imx->target)
		lpi2c_imx_target_init(lpi2c_imx);

	return 0;
}

/**
 * @brief System suspend callback for the LPI2C driver.
 * Ensures the controller remains active during the suspend process if needed.
 * @param dev Pointer to the device structure.
 * @return 0 on success.
 */
static int lpi2c_suspend(struct device *dev)
{
	/*
	 * Block Logic: Some I2C devices may need the I2C controller to remain active
	 * during resume_noirq() or suspend_noirq(). This function ensures the controller
	 * is woken up and stays active while runtime PM is still enabled, allowing
	 * it to be used during critical resume phases.
	 */
	return pm_runtime_resume_and_get(dev);
}

/**
 * @brief System resume callback for the LPI2C driver.
 * Resets the runtime PM busy state and allows autosuspend.
 * @param dev Pointer to the device structure.
 * @return 0 on success.
 */
static int lpi2c_resume(struct device *dev)
{
	pm_runtime_mark_last_busy(dev); // Mark device as busy.
	pm_runtime_put_autosuspend(dev); // Put usage count, allowing autosuspend.

	return 0;
}

/**
 * @brief Device Power Management Operations structure for the LPI2C driver.
 * Defines callbacks for system sleep and runtime power management states.
 */
static const struct dev_pm_ops lpi2c_pm_ops = {
	SET_NOIRQ_SYSTEM_SLEEP_PM_OPS(lpi2c_suspend_noirq, // No-IRQ suspend/resume.
				      lpi2c_resume_noirq)
	SYSTEM_SLEEP_PM_OPS(lpi2c_suspend, lpi2c_resume) // System sleep suspend/resume.
	SET_RUNTIME_PM_OPS(lpi2c_runtime_suspend, // Runtime suspend/resume.
			   lpi2c_runtime_resume, NULL)
};

/**
 * @brief Platform driver structure for the LPI2C controller.
 * Binds the probe, remove, and PM operations to the compatible device tree entry.
 */
static struct platform_driver lpi2c_imx_driver = {
	.probe = lpi2c_imx_probe, // Called when device is found.
	.remove = lpi2c_imx_remove, // Called when device is removed.
	.driver = {
		.name = DRIVER_NAME, // Driver name.
		.of_match_table = lpi2c_imx_of_match, // Device tree match table.
		.pm = &lpi2c_pm_ops, // Power management operations.
	},
};

module_platform_driver(lpi2c_imx_driver); // Registers the platform driver.

MODULE_AUTHOR("Gao Pan <pandy.gao@nxp.com>"); // Module author.
MODULE_DESCRIPTION("I2C adapter driver for LPI2C bus"); // Module description.
MODULE_LICENSE("GPL"); // Module license.