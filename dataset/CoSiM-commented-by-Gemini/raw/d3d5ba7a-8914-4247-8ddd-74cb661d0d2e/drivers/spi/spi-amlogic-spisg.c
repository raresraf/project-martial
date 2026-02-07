/**
 * @file spi-amlogic-spisg.c
 * @brief Linux kernel driver for the Amlogic SPI Scatter-Gather Controller.
 * This driver provides support for Amlogic's SPI controller with Scatter-Gather (SG) capabilities,
 * enabling efficient data transfer for SPI devices by managing DMA operations and interrupt handling.
 * It integrates with the Linux SPI subsystem and handles device-specific clocking, power management,
 * and transfer setups.
 */
// SPDX-License-Identifier: GPL-2.0+
/*
 * Driver for Amlogic SPI communication Scatter-Gather Controller
 *
 * Copyright (C) 2025 Amlogic, Inc. All rights reserved
 *
 * Author: Sunny Luo <sunny.luo@amlogic.com>
 * Author: Xianwei Zhao <xianwei.zhao@amlogic.com>
 */

#include <linux/bitfield.h>
#include <linux/device.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/clk.h>
#include <linux/clk-provider.h>
#include <linux/dma-mapping.h>
#include <linux/platform_device.h>
#include <linux/pinctrl/consumer.h>
#include <linux/pm_runtime.h>
#include <linux/spi/spi.h>
#include <linux/types.h>
#include <linux/interrupt.h>
#include <linux/reset.h>
#include <linux/regmap.h>

/* Register Map */
/** @brief SPISG_REG_CFG_READY Register: Indicates the readiness status of the SPI controller. */
#define SPISG_REG_CFG_READY		0x00

/** @brief SPISG_REG_CFG_SPI Register: Configures general SPI operation modes. */
#define SPISG_REG_CFG_SPI		0x04
/** @brief CFG_BUS64_EN Bit: Enables 64-bit bus mode. */
#define CFG_BUS64_EN			BIT(0)
/** @brief CFG_SLAVE_EN Bit: Enables slave mode operation. */
#define CFG_SLAVE_EN			BIT(1)
/** @brief CFG_SLAVE_SELECT Bitfield: Selects the active slave device. */
#define CFG_SLAVE_SELECT		GENMASK(3, 2)
/** @brief CFG_SFLASH_WP Bit: SPI flash write protect enable. */
#define CFG_SFLASH_WP			BIT(4)
/** @brief CFG_SFLASH_HD Bit: SPI flash hold enable. */
#define CFG_SFLASH_HD			BIT(5)
/** @brief CFG_HW_POS Bit: Initiates transfer on VSYNC rising edge. */
/* start on vsync rising */
#define CFG_HW_POS			BIT(6)
/** @brief CFG_HW_NEG Bit: Initiates transfer on VSYNC falling edge. */
/* start on vsync falling */
#define CFG_HW_NEG			BIT(7)

/** @brief SPISG_REG_CFG_START Register: Configures transfer start parameters and block properties. */
#define SPISG_REG_CFG_START		0x08
/** @brief CFG_BLOCK_NUM Bitfield: Specifies the number of blocks to transfer. */
#define CFG_BLOCK_NUM			GENMASK(19, 0)
/** @brief CFG_BLOCK_SIZE Bitfield: Defines the size of each transfer block. */
#define CFG_BLOCK_SIZE			GENMASK(22, 20)
/** @brief CFG_DATA_COMMAND Bit: Differentiates between data and command phases. */
#define CFG_DATA_COMMAND		BIT(23)
/** @brief CFG_OP_MODE Bitfield: Selects the operation mode (e.g., read, write, write command). */
#define CFG_OP_MODE			GENMASK(25, 24)
/** @brief CFG_RXD_MODE Bitfield: Configures receive data mode (e.g., PIO, MEM, SG). */
#define CFG_RXD_MODE			GENMASK(27, 26)
/** @brief CFG_TXD_MODE Bitfield: Configures transmit data mode (e.g., PIO, MEM, SG). */
#define CFG_TXD_MODE			GENMASK(29, 28)
/** @brief CFG_EOC Bit: End of chain indicator. */
#define CFG_EOC				BIT(30)
/** @brief CFG_PEND Bit: Pending transfer indication. */
#define CFG_PEND			BIT(31)

/** @brief SPISG_REG_CFG_BUS Register: Configures SPI bus parameters. */
#define SPISG_REG_CFG_BUS		0x0C
/** @brief CFG_CLK_DIV Bitfield: Sets the clock divider for SPI bus speed. */
#define CFG_CLK_DIV			GENMASK(7, 0)
/** @brief CLK_DIV_WIDTH Macro: Width of the clock divider bitfield. */
#define CLK_DIV_WIDTH			8
/** @brief CFG_RX_TUNING Bitfield: Configures RX data tuning. */
#define CFG_RX_TUNING			GENMASK(11, 8)
/** @brief CFG_TX_TUNING Bitfield: Configures TX data tuning. */
#define CFG_TX_TUNING			GENMASK(15, 12)
/** @brief CFG_CS_SETUP Bitfield: Chip select setup time. */
#define CFG_CS_SETUP			GENMASK(19, 16)
/** @brief CFG_LANE Bitfield: Configures the number of data lanes (single, dual, quad SPI). */
#define CFG_LANE			GENMASK(21, 20)
/** @brief CFG_HALF_DUPLEX Bit: Enables half-duplex mode. */
#define CFG_HALF_DUPLEX			BIT(22)
/** @brief CFG_B_L_ENDIAN Bit: Selects Big/Little Endian mode. */
#define CFG_B_L_ENDIAN			BIT(23)
/** @brief CFG_DC_MODE Bit: Data/Command mode enable. */
#define CFG_DC_MODE			BIT(24)
/** @brief CFG_NULL_CTL Bit: Null cycle control enable. */
#define CFG_NULL_CTL			BIT(25)
/** @brief CFG_DUMMY_CTL Bit: Dummy cycle control enable. */
#define CFG_DUMMY_CTL			BIT(26)
/** @brief CFG_READ_TURN Bitfield: Configures read turnaround cycles. */
#define CFG_READ_TURN			GENMASK(28, 27)
/** @brief CFG_KEEP_SS Bit: Keeps chip select active between transfers in a message. */
#define CFG_KEEP_SS			BIT(29)
/** @brief CFG_CPHA Bit: Clock Phase configuration. */
#define CFG_CPHA			BIT(30)
/** @brief CFG_CPOL Bit: Clock Polarity configuration. */
#define CFG_CPOL			BIT(31)

/** @brief SPISG_REG_PIO_TX_DATA_L Register: PIO transmit data low word. */
#define SPISG_REG_PIO_TX_DATA_L		0x10
/** @brief SPISG_REG_PIO_TX_DATA_H Register: PIO transmit data high word. */
#define SPISG_REG_PIO_TX_DATA_H		0x14
/** @brief SPISG_REG_PIO_RX_DATA_L Register: PIO receive data low word. */
#define SPISG_REG_PIO_RX_DATA_L		0x18
/** @brief SPISG_REG_PIO_RX_DATA_H Register: PIO receive data high word. */
#define SPISG_REG_PIO_RX_DATA_H		0x1C
/** @brief SPISG_REG_MEM_TX_ADDR_L Register: Memory transmit address low word. */
#define SPISG_REG_MEM_TX_ADDR_L		0x10
/** @brief SPISG_REG_MEM_TX_ADDR_H Register: Memory transmit address high word. */
#define SPISG_REG_MEM_TX_ADDR_H		0x14
/** @brief SPISG_REG_MEM_RX_ADDR_L Register: Memory receive address low word. */
#define SPISG_REG_MEM_RX_ADDR_L		0x18
/** @brief SPISG_REG_MEM_RX_ADDR_H Register: Memory receive address high word. */
#define SPISG_REG_MEM_RX_ADDR_H		0x1C
/** @brief SPISG_REG_DESC_LIST_L Register: Descriptor list base address low word. */
#define SPISG_REG_DESC_LIST_L		0x20
/** @brief SPISG_REG_DESC_LIST_H Register: Descriptor list base address high word. */
#define SPISG_REG_DESC_LIST_H		0x24
/** @brief LIST_DESC_PENDING Bit: Indicates descriptor list is pending. */
#define LIST_DESC_PENDING		BIT(31)
/** @brief SPISG_REG_DESC_CURRENT_L Register: Current descriptor address low word. */
#define SPISG_REG_DESC_CURRENT_L	0x28
/** @brief SPISG_REG_DESC_CURRENT_H Register: Current descriptor address high word. */
#define SPISG_REG_DESC_CURRENT_H	0x2c
/** @brief SPISG_REG_IRQ_STS Register: Interrupt Status Register. */
#define SPISG_REG_IRQ_STS		0x30
/** @brief SPISG_REG_IRQ_ENABLE Register: Interrupt Enable Register. */
#define SPISG_REG_IRQ_ENABLE		0x34
/** @brief IRQ_RCH_DESC_EOC Bit: Receive Channel Descriptor End-Of-Chain interrupt. */
#define IRQ_RCH_DESC_EOC		BIT(0)
/** @brief IRQ_RCH_DESC_INVALID Bit: Receive Channel Descriptor Invalid interrupt. */
#define IRQ_RCH_DESC_INVALID		BIT(1)
/** @brief IRQ_RCH_DESC_RESP Bit: Receive Channel Descriptor Response interrupt. */
#define IRQ_RCH_DESC_RESP		BIT(2)
/** @brief IRQ_RCH_DATA_RESP Bit: Receive Channel Data Response interrupt. */
#define IRQ_RCH_DATA_RESP		BIT(3)
/** @brief IRQ_WCH_DESC_EOC Bit: Write Channel Descriptor End-Of-Chain interrupt. */
#define IRQ_WCH_DESC_EOC		BIT(4)
/** @brief IRQ_WCH_DESC_INVALID Bit: Write Channel Descriptor Invalid interrupt. */
#define IRQ_WCH_DESC_INVALID		BIT(5)
/** @brief IRQ_WCH_DESC_RESP Bit: Write Channel Descriptor Response interrupt. */
#define IRQ_WCH_DESC_RESP		BIT(6)
/** @brief IRQ_WCH_DATA_RESP Bit: Write Channel Data Response interrupt. */
#define IRQ_WCH_DATA_RESP		BIT(7)
/** @brief IRQ_DESC_ERR Bit: Descriptor Error interrupt. */
#define IRQ_DESC_ERR			BIT(8)
/** @brief IRQ_SPI_READY Bit: SPI Ready interrupt. */
#define IRQ_SPI_READY			BIT(9)
/** @brief IRQ_DESC_DONE Bit: Descriptor processing done interrupt. */
#define IRQ_DESC_DONE			BIT(10)
/** @brief IRQ_DESC_CHAIN_DONE Bit: Descriptor chain processing done interrupt. */
#define IRQ_DESC_CHAIN_DONE		BIT(11)

/** @brief SPISG_MAX_REG Macro: Maximum register address for the SPISG controller. */
#define SPISG_MAX_REG			0x40

/** @brief SPISG_BLOCK_MAX Macro: Maximum block size for Scatter-Gather transfers. */
#define SPISG_BLOCK_MAX			0x100000

/** @brief SPISG_OP_MODE_WRITE_CMD Macro: Operation mode for writing commands. */
#define SPISG_OP_MODE_WRITE_CMD		0
/** @brief SPISG_OP_MODE_READ_STS Macro: Operation mode for reading status. */
#define SPISG_OP_MODE_READ_STS		1
/** @brief SPISG_OP_MODE_WRITE Macro: Operation mode for writing data. */
#define SPISG_OP_MODE_WRITE		2
/** @brief SPISG_OP_MODE_READ Macro: Operation mode for reading data. */
#define SPISG_OP_MODE_READ		3

/** @brief SPISG_DATA_MODE_NONE Macro: Data mode - None. */
#define SPISG_DATA_MODE_NONE		0
/** @brief SPISG_DATA_MODE_PIO Macro: Data mode - Programmed I/O. */
#define SPISG_DATA_MODE_PIO		1
/** @brief SPISG_DATA_MODE_MEM Macro: Data mode - Memory. */
#define SPISG_DATA_MODE_MEM		2
/** @brief SPISG_DATA_MODE_SG Macro: Data mode - Scatter-Gather. */
#define SPISG_DATA_MODE_SG		3

/** @brief SPISG_CLK_DIV_MAX Macro: Maximum clock divider value. */
#define SPISG_CLK_DIV_MAX		256
/** @brief SPISG_CLK_DIV_MIN Macro: Minimum clock divider value (recommended by specification). */

#define SPISG_CLK_DIV_MIN		4
/** @brief DIV_NUM Macro: Number of possible clock divider values. */
#define DIV_NUM (SPISG_CLK_DIV_MAX - SPISG_CLK_DIV_MIN + 1)

/** @brief SPISG_PCLK_RATE_MIN Macro: Minimum PCLK rate for SPISG operation. */
#define SPISG_PCLK_RATE_MIN		24000000

/** @brief SPISG_SINGLE_SPI Macro: Single SPI lane mode. */
#define SPISG_SINGLE_SPI		0
/** @brief SPISG_DUAL_SPI Macro: Dual SPI lane mode. */
#define SPISG_DUAL_SPI			1
/** @brief SPISG_QUAD_SPI Macro: Quad SPI lane mode. */
#define SPISG_QUAD_SPI			2

/**
 * @brief Represents a Scatter-Gather (SG) link descriptor for DMA operations.
 * This structure defines the properties of a single SG entry, including its address,
 * length, and various flags controlling its behavior in a chain.
 */
struct spisg_sg_link {
/** @brief LINK_ADDR_VALID Bit: Indicates if the address in this link descriptor is valid. */
#define LINK_ADDR_VALID		BIT(0)
/** @brief LINK_ADDR_EOC Bit: Indicates if this is the End-Of-Chain for the descriptor list. */
#define LINK_ADDR_EOC		BIT(1)
/** @brief LINK_ADDR_IRQ Bit: Generates an interrupt upon processing this link descriptor. */
#define LINK_ADDR_IRQ		BIT(2)
/** @brief LINK_ADDR_ACT Bitfield: Action to perform for this descriptor. */
#define LINK_ADDR_ACT		GENMASK(5, 3)
/** @brief LINK_ADDR_RING Bit: Indicates if the descriptor should loop back to the beginning of the list (ring mode). */
#define LINK_ADDR_RING		BIT(6)
/** @brief LINK_ADDR_LEN Bitfield: Length of the data block for this descriptor. */
#define LINK_ADDR_LEN		GENMASK(31, 8)
	u32			addr; /**< @brief Address and control flags for the SG entry. */
	u32			addr1; /**< @brief Physical address of the data buffer. */
};

/**
 * @brief Represents a single SPI Scatter-Gather descriptor.
 * This structure holds the configuration for a specific SPI transfer,
 * including bus and start configurations, and physical addresses for TX/RX data.
 */
struct spisg_descriptor {
	u32				cfg_start; /**< @brief Value for SPISG_REG_CFG_START register. */
	u32				cfg_bus; /**< @brief Value for SPISG_REG_CFG_BUS register. */
	u64				tx_paddr; /**< @brief Physical address of the transmit data buffer. */
	u64				rx_paddr; /**< @brief Physical address of the receive data buffer. */
};

/**
 * @brief Additional descriptor information for Scatter-Gather lists.
 * This structure is used to store pointers to the Scatter-Gather link lists (ccsg)
 * and their respective lengths for both transmit and receive paths.
 */
struct spisg_descriptor_extra {
	struct spisg_sg_link		*tx_ccsg; /**< @brief Pointer to the transmit Scatter-Gather link list. */
	struct spisg_sg_link		*rx_ccsg; /**< @brief Pointer to the receive Scatter-Gather link list. */
	int				tx_ccsg_len; /**< @brief Length of the transmit Scatter-Gather link list. */
	int				rx_ccsg_len; /**< @brief Length of the receive Scatter-Gather link list. */
};

/**
 * @brief Represents the Amlogic SPI Scatter-Gather device context.
 * This comprehensive structure holds all relevant information about the SPISG controller,
 * including its associated SPI controller, platform device, register map, clocks,
 * completion mechanism, and various configuration parameters.
 */
struct spisg_device {
	struct spi_controller		*controller; /**< @brief Pointer to the generic Linux SPI controller. */
	struct platform_device		*pdev; /**< @brief Pointer to the platform device. */
	struct regmap			*map; /**< @brief Register map interface for accessing SPISG registers. */
	struct clk			*core; /**< @brief Core clock for the SPISG controller. */
	struct clk			*pclk; /**< @brief Peripheral clock for the SPISG controller. */
	struct clk			*sclk; /**< @brief SPI clock used for transfers. */
	struct clk_div_table		*tbl; /**< @brief Clock divider table for SPI clock generation. */
	struct completion		completion; /**< @brief Completion object for synchronizing IRQ handling. */
	u32				status; /**< @brief Stores the interrupt status from SPISG_REG_IRQ_STS. */
	u32				speed_hz; /**< @brief Desired SPI bus speed in Hz. */
	u32				effective_speed_hz; /**< @brief Actual effective SPI bus speed in Hz. */
	u32				bytes_per_word; /**< @brief Number of bytes per SPI word. */
	u32				cfg_spi; /**< @brief Cached value for SPISG_REG_CFG_SPI register. */
	u32				cfg_start; /**< @brief Cached value for SPISG_REG_CFG_START register. */
	u32				cfg_bus; /**< @brief Cached value for SPISG_REG_CFG_BUS register. */
};

/**
 * @brief Converts a SPI delay value to the equivalent number of SPI clock cycles.
 * @param slck_speed_hz The speed of the SPI clock in Hz.
 * @param delay Pointer to a `spi_delay` structure containing the delay unit and value.
 * @return The delay in terms of SPI clock cycles, or 0 if no delay structure is provided
 *         or the delay unit is not supported.
 */
/**
 * @brief Converts a SPI delay value to the equivalent number of SPI clock cycles.
 * @param slck_speed_hz The speed of the SPI clock in Hz.
 * @param delay Pointer to a `spi_delay` structure containing the delay unit and value.
 * @return The delay in terms of SPI clock cycles, or 0 if no delay structure is provided
 *         or the delay unit is not supported.
 */
static int spi_delay_to_sclk(u32 slck_speed_hz, struct spi_delay *delay)
{
	s32 ns;

	if (!delay)
		return 0;

	if (delay->unit == SPI_DELAY_UNIT_SCK)
		return delay->value;

	ns = spi_delay_to_ns(delay, NULL);
	if (ns < 0)
		return 0;

	return DIV_ROUND_UP_ULL(slck_speed_hz * ns, NSEC_PER_SEC);
}

/**
 * @brief Reads the SPISG_REG_CFG_READY register and clears it if set.
 * This acts as a semaphore to check if the controller is ready and to acknowledge it.
 * @param spisg Pointer to the `spisg_device` structure.
 * @return The value read from the SPISG_REG_CFG_READY register (0 or 1).
 */
/**
 * @brief Reads the SPISG_REG_CFG_READY register and clears it if set.
 * This acts as a semaphore to check if the controller is ready and to acknowledge it.
 * @param spisg Pointer to the `spisg_device` structure.
 * @return The value read from the SPISG_REG_CFG_READY register (0 or 1).
 */
static inline u32 aml_spisg_sem_down_read(struct spisg_device *spisg)
{
	u32 ret;

	regmap_read(spisg->map, SPISG_REG_CFG_READY, &ret);
	if (ret)
		regmap_write(spisg->map, SPISG_REG_CFG_READY, 0);

	return ret;
}

/**
 * @brief Sets the SPISG_REG_CFG_READY register to indicate the controller is available.
 * This acts as a semaphore to signal readiness.
 * @param spisg Pointer to the `spisg_device` structure.
 */
/**
 * @brief Sets the SPISG_REG_CFG_READY register to indicate the controller is available.
 * This acts as a semaphore to signal readiness.
 * @param spisg Pointer to the `spisg_device` structure.
 */
static inline void aml_spisg_sem_up_write(struct spisg_device *spisg)
{
	regmap_write(spisg->map, SPISG_REG_CFG_READY, 1);
}

/**
 * @brief Sets the SPI clock speed for the SPISG device.
 * Attempts to set the `sclk` to the desired speed and updates the internal
 * `cfg_bus` register value with the corresponding clock divider.
 * @param spisg Pointer to the `spisg_device` structure.
 * @param speed_hz The desired SPI bus speed in Hz.
 * @return `0` on success.
 */
/**
 * @brief Sets the SPI clock speed for the SPISG device.
 * Attempts to set the `sclk` to the desired speed and updates the internal
 * `cfg_bus` register value with the corresponding clock divider.
 * @param spisg Pointer to the `spisg_device` structure.
 * @param speed_hz The desired SPI bus speed in Hz.
 * @return `0` on success.
 */
static int aml_spisg_set_speed(struct spisg_device *spisg, uint speed_hz)
{
	u32 cfg_bus;

	if (!speed_hz || speed_hz == spisg->speed_hz)
		return 0;

	spisg->speed_hz = speed_hz;
	clk_set_rate(spisg->sclk, speed_hz);
	/* Store the div for the descriptor mode */
	regmap_read(spisg->map, SPISG_REG_CFG_BUS, &cfg_bus);
	spisg->cfg_bus &= ~CFG_CLK_DIV;
	spisg->cfg_bus |= cfg_bus & CFG_CLK_DIV;
	spisg->effective_speed_hz = clk_get_rate(spisg->sclk);
	dev_dbg(&spisg->pdev->dev,
		"desired speed %dHz, effective speed %dHz\n",
		speed_hz, spisg->effective_speed_hz);

	return 0;
}

/**
 * @brief Determines if the SPISG controller can handle DMA for a given SPI transfer.
 * Always returns true, indicating that the controller supports DMA for all transfers.
 * @param ctlr Pointer to the `spi_controller` structure.
 * @param spi Pointer to the `spi_device` structure.
 * @param xfer Pointer to the `spi_transfer` structure.
 * @return `true` if DMA can be used, `false` otherwise (though currently always true).
 */
/**
 * @brief Determines if the SPISG controller can handle DMA for a given SPI transfer.
 * Always returns true, indicating that the controller supports DMA for all transfers.
 * @param ctlr Pointer to the `spi_controller` structure.
 * @param spi Pointer to the `spi_device` structure.
 * @param xfer Pointer to the `spi_transfer` structure.
 * @return `true` if DMA can be used, `false` otherwise (though currently always true).
 */
static bool aml_spisg_can_dma(struct spi_controller *ctlr,
			      struct spi_device *spi,
			      struct spi_transfer *xfer)
{
	return true;
}

/**
 * @brief Translates a scatterlist (sg_table) into a series of SPISG Scatter-Gather link descriptors.
 * This function converts the kernel's scatter-gather list format into the hardware-specific
 * `spisg_sg_link` format, preparing it for DMA.
 * @param sgt Pointer to the `sg_table` to translate.
 * @param ccsg Pointer to the array of `spisg_sg_link` descriptors to fill.
 */
static void aml_spisg_sg_xlate(struct sg_table *sgt, struct spisg_sg_link *ccsg)
{
	struct scatterlist *sg;
	int i;

	for_each_sg(sgt->sgl, sg, sgt->nents, i) {
		// Block Logic: Populates the `spisg_sg_link` fields from the scatterlist entry.
		// This includes setting address validity, end-of-chain, and data length.
		ccsg->addr = FIELD_PREP(LINK_ADDR_VALID, 1) |
			     FIELD_PREP(LINK_ADDR_RING, 0) |
			     FIELD_PREP(LINK_ADDR_EOC, sg_is_last(sg)) |
			     FIELD_PREP(LINK_ADDR_LEN, sg_dma_len(sg));
		ccsg->addr1 = (u32)sg_dma_address(sg); // Physical DMA address
		ccsg++;
	}
}



/**
 * @brief Sets up an Amlogic SPISG descriptor for a given SPI transfer.
 * This function configures a `spisg_descriptor` and `spisg_descriptor_extra`
 * based on the parameters of an `spi_transfer`, including DMA mapping of buffers
 * and scatter-gather lists.
 * @param spisg Pointer to the `spisg_device` structure.
 * @param xfer Pointer to the `spi_transfer` structure to configure.
 * @param desc Pointer to the `spisg_descriptor` to be populated.
 * @param exdesc Pointer to the `spisg_descriptor_extra` to be populated for SG links.
 * @return `0` on success, or a negative error code on failure (e.g., memory allocation, DMA mapping error).
 */
static int aml_spisg_setup_transfer(struct spisg_device *spisg,
				    struct spi_transfer *xfer,
				    struct spisg_descriptor *desc,
				    struct spisg_descriptor_extra *exdesc)
{
	int block_size, blocks;
	struct device *dev = &spisg->pdev->dev;
	struct spisg_sg_link *ccsg;
	int ccsg_len;
	dma_addr_t paddr;
	int ret;

	memset(desc, 0, sizeof(*desc));
	if (exdesc)
		memset(exdesc, 0, sizeof(*exdesc));
	aml_spisg_set_speed(spisg, xfer->speed_hz);
	xfer->effective_speed_hz = spisg->effective_speed_hz;

	desc->cfg_start = spisg->cfg_start;
	desc->cfg_bus = spisg->cfg_bus;

	block_size = xfer->bits_per_word >> 3;
	blocks = xfer->len / block_size;

	desc->cfg_start |= FIELD_PREP(CFG_EOC, 0);
	desc->cfg_bus |= FIELD_PREP(CFG_KEEP_SS, !xfer->cs_change);
	desc->cfg_bus |= FIELD_PREP(CFG_NULL_CTL, 0);

	// Block Logic: Configure transmit and receive operation modes based on `xfer` buffers.
	if (xfer->tx_buf || xfer->tx_dma) {
		desc->cfg_bus |= FIELD_PREP(CFG_LANE, nbits_to_lane[xfer->tx_nbits]);
		desc->cfg_start |= FIELD_PREP(CFG_OP_MODE, SPISG_OP_MODE_WRITE);
	}
	if (xfer->rx_buf || xfer->rx_dma) {
		desc->cfg_bus |= FIELD_PREP(CFG_LANE, nbits_to_lane[xfer->rx_nbits]);
		desc->cfg_start |= FIELD_PREP(CFG_OP_MODE, SPISG_OP_MODE_READ);
	}

	// Block Logic: Adjust block size and number for read status mode or general data transfer.
	if (FIELD_GET(CFG_OP_MODE, desc->cfg_start) == SPISG_OP_MODE_READ_STS) {
		desc->cfg_start |= FIELD_PREP(CFG_BLOCK_SIZE, blocks) |
				   FIELD_PREP(CFG_BLOCK_NUM, 1);
	} else {
		blocks = min_t(int, blocks, SPISG_BLOCK_MAX);
		desc->cfg_start |= FIELD_PREP(CFG_BLOCK_SIZE, block_size & 0x7) |
				   FIELD_PREP(CFG_BLOCK_NUM, blocks);
	}

	// Block Logic: Handle Scatter-Gather for TX buffer. Allocates, translates, and maps SG links for DMA.
	if (xfer->tx_sg.nents && xfer->tx_sg.sgl) {
		ccsg_len = xfer->tx_sg.nents * sizeof(struct spisg_sg_link);
		ccsg = kzalloc(ccsg_len, GFP_KERNEL | GFP_DMA);
		if (!ccsg) {
			dev_err(dev, "alloc tx_ccsg failed\n");
			return -ENOMEM;
		}

		aml_spisg_sg_xlate(&xfer->tx_sg, ccsg);
		paddr = dma_map_single(dev, (void *)ccsg,
				       ccsg_len, DMA_TO_DEVICE);
		ret = dma_mapping_error(dev, paddr);
		if (ret) {
			kfree(ccsg);
			dev_err(dev, "tx ccsg map failed\n");
			return ret;
		}

		desc->tx_paddr = paddr;
		desc->cfg_start |= FIELD_PREP(CFG_TXD_MODE, SPISG_DATA_MODE_SG);
		exdesc->tx_ccsg = ccsg;
		exdesc->tx_ccsg_len = ccsg_len;
		dma_sync_sgtable_for_device(spisg->controller->cur_tx_dma_dev,
					    &xfer->tx_sg, DMA_TO_DEVICE);
	} else if (xfer->tx_buf || xfer->tx_dma) {
		paddr = xfer->tx_dma;
		if (!paddr) {
			paddr = dma_map_single(dev, (void *)xfer->tx_buf,
					       xfer->len, DMA_TO_DEVICE);
			ret = dma_mapping_error(dev, paddr);
			if (ret) {
				dev_err(dev, "tx buf map failed\n");
				return ret;
			}
		}
		desc->tx_paddr = paddr;
		desc->cfg_start |= FIELD_PREP(CFG_TXD_MODE, SPISG_DATA_MODE_MEM);
	}

	// Block Logic: Handle Scatter-Gather for RX buffer. Allocates, translates, and maps SG links for DMA.
	if (xfer->rx_sg.nents && xfer->rx_sg.sgl) {
		ccsg_len = xfer->rx_sg.nents * sizeof(struct spisg_sg_link);
		ccsg = kzalloc(ccsg_len, GFP_KERNEL | GFP_DMA);
		if (!ccsg) {
			dev_err(dev, "alloc rx_ccsg failed\n");
			return -ENOMEM;
		}

		aml_spisg_sg_xlate(&xfer->rx_sg, ccsg);
		paddr = dma_map_single(dev, (void *)ccsg,
				       ccsg_len, DMA_TO_DEVICE);
		ret = dma_mapping_error(dev, paddr);
		if (ret) {
			kfree(ccsg);
			dev_err(dev, "rx ccsg map failed\n");
			return ret;
		}

		desc->rx_paddr = paddr;
		desc->cfg_start |= FIELD_PREP(CFG_RXD_MODE, SPISG_DATA_MODE_SG);
		exdesc->rx_ccsg = ccsg;
		exdesc->rx_ccsg_len = ccsg_len;
		dma_sync_sgtable_for_device(spisg->controller->cur_rx_dma_dev,
					    &xfer->rx_sg, DMA_FROM_DEVICE);
	} else if (xfer->rx_buf || xfer->rx_dma) {
		paddr = xfer->rx_dma;
		if (!paddr) {
			paddr = dma_map_single(dev, xfer->rx_buf,
					       xfer->len, DMA_FROM_DEVICE);
			ret = dma_mapping_error(dev, paddr);
			if (ret) {
				dev_err(dev, "rx buf map failed\n");
				return ret;
			}
		}

		desc->rx_paddr = paddr;
		desc->cfg_start |= FIELD_PREP(CFG_RXD_MODE, SPISG_DATA_MODE_MEM);
	}

	return 0;
}

static void aml_spisg_cleanup_transfer(struct spisg_device *spisg,
				       struct spi_transfer *xfer,
				       struct spisg_descriptor *desc,
				       struct spisg_descriptor_extra *exdesc)
{
	struct device *dev = &spisg->pdev->dev;

	if (desc->tx_paddr) {
		if (FIELD_GET(CFG_TXD_MODE, desc->cfg_start) == SPISG_DATA_MODE_SG) {
			dma_unmap_single(dev, (dma_addr_t)desc->tx_paddr,
					 exdesc->tx_ccsg_len, DMA_TO_DEVICE);
			kfree(exdesc->tx_ccsg);
			dma_sync_sgtable_for_cpu(spisg->controller->cur_tx_dma_dev,
						 &xfer->tx_sg, DMA_TO_DEVICE);
		} else if (!xfer->tx_dma) {
			dma_unmap_single(dev, (dma_addr_t)desc->tx_paddr,
					 xfer->len, DMA_TO_DEVICE);
		}
	}

	if (desc->rx_paddr) {
		if (FIELD_GET(CFG_RXD_MODE, desc->cfg_start) == SPISG_DATA_MODE_SG) {
			dma_unmap_single(dev, (dma_addr_t)desc->rx_paddr,
					 exdesc->rx_ccsg_len, DMA_TO_DEVICE);
			kfree(exdesc->rx_ccsg);
			dma_sync_sgtable_for_cpu(spisg->controller->cur_rx_dma_dev,
						 &xfer->rx_sg, DMA_FROM_DEVICE);
		} else if (!xfer->rx_dma) {
			dma_unmap_single(dev, (dma_addr_t)desc->rx_paddr,
					 xfer->len, DMA_FROM_DEVICE);
		}
	}
}

/**
 * @brief Sets up a null descriptor to introduce a delay for chip select hold time.
 * This is used to ensure the chip select remains active for a specified duration
 * after a transfer completes.
 * @param spisg Pointer to the `spisg_device` structure.
 * @param desc Pointer to the `spisg_descriptor` to be configured as a null descriptor.
 * @param n_sclk The number of SPI clock cycles for which to hold the chip select.
 */
static void aml_spisg_setup_null_desc(struct spisg_device *spisg,
				      struct spisg_descriptor *desc,
				      u32 n_sclk)
{
	/* unit is the last xfer sclk */
	desc->cfg_start = spisg->cfg_start;
	desc->cfg_bus = spisg->cfg_bus;

	desc->cfg_start |= FIELD_PREP(CFG_OP_MODE, SPISG_OP_MODE_WRITE) |
			   FIELD_PREP(CFG_BLOCK_SIZE, 1) |
			   FIELD_PREP(CFG_BLOCK_NUM, DIV_ROUND_UP(n_sclk, 8));

	desc->cfg_bus |= FIELD_PREP(CFG_NULL_CTL, 1);
}

/**
 * @brief Submits a descriptor list to the SPISG hardware for processing.
 * Configures the hardware with the physical address of the descriptor list,
 * enables interrupts, and optionally triggers the transfer.
 * @param spisg Pointer to the `spisg_device` structure.
 * @param desc_paddr Physical address of the descriptor list.
 * @param trig If `true`, triggers the transfer immediately (hardware position mode).
 * @param irq_en If `true`, enables SPISG interrupts for descriptor processing.
 */
static void aml_spisg_pending(struct spisg_device *spisg,
			      dma_addr_t desc_paddr,
			      bool trig,
			      bool irq_en)
{
	u32 desc_l, desc_h, cfg_spi, irq_enable;

#ifdef	CONFIG_ARCH_DMA_ADDR_T_64BIT
	desc_l = (u64)desc_paddr & 0xffffffff;
	desc_h = (u64)desc_paddr >> 32;
#else
	desc_l = desc_paddr & 0xffffffff;
	desc_h = 0;
#endif

	cfg_spi = spisg->cfg_spi;
	if (trig)
		cfg_spi |= CFG_HW_POS;
	else
		desc_h |= LIST_DESC_PENDING;

	// Block Logic: Configure which interrupts to enable for descriptor processing.
	irq_enable = IRQ_RCH_DESC_INVALID | IRQ_RCH_DESC_RESP |
		     IRQ_RCH_DATA_RESP | IRQ_WCH_DESC_INVALID |
		     IRQ_WCH_DESC_RESP | IRQ_WCH_DATA_RESP |
		     IRQ_DESC_ERR | IRQ_DESC_CHAIN_DONE;
	regmap_write(spisg->map, SPISG_REG_IRQ_ENABLE, irq_en ? irq_enable : 0);
	regmap_write(spisg->map, SPISG_REG_CFG_SPI, cfg_spi);
	regmap_write(spisg->map, SPISG_REG_DESC_LIST_L, desc_l);
	regmap_write(spisg->map, SPISG_REG_DESC_LIST_H, desc_h);
}

/**
 * @brief Interrupt handler for the Amlogic SPISG controller.
 * Reads the interrupt status register, clears the interrupts, and signals
 * completion if a relevant interrupt has occurred.
 * @param irq The interrupt number.
 * @param data Pointer to the `spisg_device` structure (passed as `void *`).
 * @return `IRQ_HANDLED` if the interrupt was handled by this driver, `IRQ_NONE` otherwise.
 */
static irqreturn_t aml_spisg_irq(int irq, void *data)
{
	struct spisg_device *spisg = (void *)data;
	u32 sts;

	spisg->status = 0;
	regmap_read(spisg->map, SPISG_REG_IRQ_STS, &sts);
	regmap_write(spisg->map, SPISG_REG_IRQ_STS, sts); // Clear pending interrupts
	// Block Logic: Check for various descriptor and data response errors or chain completion.
	if (sts & (IRQ_RCH_DESC_INVALID |
		   IRQ_RCH_DESC_RESP |
		   IRQ_RCH_DATA_RESP |
		   IRQ_WCH_DESC_INVALID |
		   IRQ_WCH_DESC_RESP |
		   IRQ_WCH_DATA_RESP |
		   IRQ_DESC_ERR))
		spisg->status = sts; // Store error status
	else if (sts & IRQ_DESC_CHAIN_DONE)
		spisg->status = 0; // Clear status on successful chain completion
	else
		return IRQ_NONE; // Not our interrupt

	complete(&spisg->completion); // Signal completion to waiting threads

	return IRQ_HANDLED;
}

/**
 * @brief Transfers a single SPI message using the Amlogic SPISG controller.
 * This is the main transfer function called by the SPI core. It prepares
 * descriptor lists for all transfers in the message, initiates the DMA transfer,
 * and waits for completion.
 * @param ctlr Pointer to the `spi_controller` structure.
 * @param msg Pointer to the `spi_message` to transfer.
 * @return `0` on success, or a negative error code on failure (e.g., busy, timeout, DMA error).
 */
static int aml_spisg_transfer_one_message(struct spi_controller *ctlr,
					  struct spi_message *msg)
{
	struct spisg_device *spisg = spi_controller_get_devdata(ctlr);
	struct device *dev = &spisg->pdev->dev;
	unsigned long long ms = 0;
	struct spi_transfer *xfer;
	struct spisg_descriptor *descs, *desc;
	struct spisg_descriptor_extra *exdescs, *exdesc;
	dma_addr_t descs_paddr;
	int desc_num = 1, descs_len; // Start with 1 for potential null-descriptor
	u32 cs_hold_in_sclk = 0;
	int ret = -EIO;

	// Pre-condition: Check if the controller is available (semaphore check).
	if (!aml_spisg_sem_down_read(spisg)) {
		spi_finalize_current_message(ctlr);
		dev_err(dev, "controller busy\n");
		return -EBUSY;
	}

	// Block Logic: Calculate the total number of descriptors needed for all transfers and any additional delays.
	list_for_each_entry(xfer, &msg->transfers, transfer_list)
		desc_num++;

	// Block Logic: Allocate memory for the descriptors and extra descriptor information (for SG links).
	descs = kcalloc(desc_num, sizeof(*desc) + sizeof(*exdesc),
			GFP_KERNEL | GFP_DMA);
	if (!descs) {
		spi_finalize_current_message(ctlr);
		aml_spisg_sem_up_write(spisg);
		return -ENOMEM;
	}
	descs_len = sizeof(*desc) * desc_num;
	exdescs = (struct spisg_descriptor_extra *)(descs + desc_num);

	// Block Logic: Iterate through each SPI transfer to set up its corresponding hardware descriptor.
	desc = descs;
	exdesc = exdescs;
	list_for_each_entry(xfer, &msg->transfers, transfer_list) {
		ret = aml_spisg_setup_transfer(spisg, xfer, desc, exdesc);
		if (ret) {
			dev_err(dev, "config descriptor failed\n");
			goto end;
		}

		// Block Logic: Configure CS setup delay for the first transfer and CS hold delay for the last transfer.
		if (list_is_first(&xfer->transfer_list, &msg->transfers))
			desc->cfg_bus |= FIELD_PREP(CFG_CS_SETUP,
				spi_delay_to_sclk(xfer->effective_speed_hz, &msg->spi->cs_setup));

		if (list_is_last(&xfer->transfer_list, &msg->transfers))
			cs_hold_in_sclk =
				spi_delay_to_sclk(xfer->effective_speed_hz, &msg->spi->cs_hold);

		desc++;
		exdesc++;
		// Functional Utility: Estimate message transfer time to set a reasonable timeout.
		ms += DIV_ROUND_UP_ULL(8LL * MSEC_PER_SEC * xfer->len,
				       xfer->effective_speed_hz);
	}

	// Block Logic: If a CS hold delay is required, add an additional null-descriptor to achieve it.
	if (cs_hold_in_sclk)
		/* additional null-descriptor to achieve the cs-hold delay */
		aml_spisg_setup_null_desc(spisg, desc, cs_hold_in_sclk);
	else
		desc--; // Adjust `desc` to point to the last actual transfer descriptor

	desc->cfg_bus |= FIELD_PREP(CFG_KEEP_SS, 0); // Release CS after last transfer
	desc->cfg_start |= FIELD_PREP(CFG_EOC, 1); // Mark last descriptor as end of chain

	/* some tolerances */
	ms += ms + 20; // Add some tolerance to the timeout
	if (ms > UINT_MAX)
		ms = UINT_MAX;

	// Block Logic: Map the descriptor table for DMA and handle any mapping errors.
	descs_paddr = dma_map_single(dev, (void *)descs,
				     descs_len, DMA_TO_DEVICE);
	ret = dma_mapping_error(dev, descs_paddr);
	if (ret) {
		dev_err(dev, "desc table map failed\n");
		goto end;
	}

	// Block Logic: Reinitialize completion, arm the hardware with the descriptor list, and wait for transfer completion.
	reinit_completion(&spisg->completion);
	aml_spisg_pending(spisg, descs_paddr, false, true); // Submit descriptors and enable IRQ
	if (wait_for_completion_timeout(&spisg->completion,
					spi_controller_is_target(spisg->controller) ?
					MAX_SCHEDULE_TIMEOUT : msecs_to_jiffies(ms)))
		ret = spisg->status ? -EIO : 0; // Check for IRQ status errors or success
	else
		ret = -ETIMEDOUT; // Transfer timed out

	dma_unmap_single(dev, descs_paddr, descs_len, DMA_TO_DEVICE); // Unmap descriptor table from DMA
end:
	// Block Logic: Clean up resources for each transfer after completion (unmap buffers, free SG links).
	desc = descs;
	exdesc = exdescs;
	list_for_each_entry(xfer, &msg->transfers, transfer_list)
		aml_spisg_cleanup_transfer(spisg, xfer, desc++, exdesc++);
	kfree(descs); // Free allocated descriptor memory

	if (!ret)
		msg->actual_length = msg->frame_length; // Update actual length if transfer was successful
	msg->status = ret; // Set message status
	spi_finalize_current_message(ctlr); // Finalize SPI message
	aml_spisg_sem_up_write(spisg); // Release semaphore

	return ret;
}

/**
 * @brief Prepares the SPISG controller for a new SPI message.
 * Configures the controller's mode (CPOL, CPHA, LSB_FIRST, 3WIRE), chip select,
 * and bytes per word based on the `spi_device` settings in the message.
 * @param ctlr Pointer to the `spi_controller` structure.
 * @param message Pointer to the `spi_message` to prepare.
 * @return `0` on success, or a negative error code if the word length is invalid.
 */
static int aml_spisg_prepare_message(struct spi_controller *ctlr,
				     struct spi_message *message)
{
	struct spisg_device *spisg = spi_controller_get_devdata(ctlr);
	struct spi_device *spi = message->spi;

	if (!spi->bits_per_word || spi->bits_per_word % 8) {
		dev_err(&spisg->pdev->dev, "invalid wordlen %d\n", spi->bits_per_word);
		return -EINVAL;
	}

	spisg->bytes_per_word = spi->bits_per_word >> 3;

	spisg->cfg_spi &= ~CFG_SLAVE_SELECT;
	spisg->cfg_spi |= FIELD_PREP(CFG_SLAVE_SELECT, spi_get_chipselect(spi, 0));

	spisg->cfg_bus &= ~(CFG_CPOL | CFG_CPHA | CFG_B_L_ENDIAN | CFG_HALF_DUPLEX);
	spisg->cfg_bus |= FIELD_PREP(CFG_CPOL, !!(spi->mode & SPI_CPOL)) |
			  FIELD_PREP(CFG_CPHA, !!(spi->mode & SPI_CPHA)) |
			  FIELD_PREP(CFG_B_L_ENDIAN, !!(spi->mode & SPI_LSB_FIRST)) |
			  FIELD_PREP(CFG_HALF_DUPLEX, !!(spi->mode & SPI_3WIRE));

	return 0;
}

/**
 * @brief SPI controller setup callback.
 * Used to store a pointer to the `spisg_device` in the `spi_device`'s controller state.
 * @param spi Pointer to the `spi_device` structure.
 * @return `0` always.
 */
static int aml_spisg_setup(struct spi_device *spi)
{
	if (!spi->controller_state)
		spi->controller_state = spi_controller_get_devdata(spi->controller);

	return 0;
}

/**
 * @brief SPI controller cleanup callback.
 * Clears the `spi_device`'s controller state.
 * @param spi Pointer to the `spi_device` structure.
 */
static void aml_spisg_cleanup(struct spi_device *spi)
{
	spi->controller_state = NULL;
}

/**
 * @brief SPI controller target abort callback.
 * Aborts any ongoing transfer by clearing the descriptor list and signaling completion.
 * @param ctlr Pointer to the `spi_controller` structure.
 * @return `0` always.
 */
static int aml_spisg_target_abort(struct spi_controller *ctlr)
{
	struct spisg_device *spisg = spi_controller_get_devdata(ctlr);

	spisg->status = 0;
	regmap_write(spisg->map, SPISG_REG_DESC_LIST_H, 0); // Clear descriptor list high address
	complete(&spisg->completion); // Signal completion

	return 0;
}

/**
 * @brief Initializes the clocking for the Amlogic SPISG controller.
 * Retrieves and configures core, peripheral (pclk), and SPI (sclk) clocks.
 * It also sets up a clock divider table for flexible SPI clock generation.
 * @param spisg Pointer to the `spisg_device` structure.
 * @param base Base address of the SPISG registers.
 * @return `0` on success, or a negative error code on failure.
 */
static int aml_spisg_clk_init(struct spisg_device *spisg, void __iomem *base)
{
	struct device *dev = &spisg->pdev->dev;
	struct clk_init_data init;
	struct clk_divider *div;
	struct clk_div_table *tbl;
	char name[32];
	int ret, i;

	// Block Logic: Get and enable core and peripheral clocks.
	spisg->core = devm_clk_get_enabled(dev, "core");
	if (IS_ERR_OR_NULL(spisg->core)) {
		dev_err(dev, "core clock request failed\n");
		return PTR_ERR(spisg->core);
	}

	spisg->pclk = devm_clk_get_enabled(dev, "pclk");
	if (IS_ERR_OR_NULL(spisg->pclk)) {
		dev_err(dev, "pclk clock request failed\n");
		return PTR_ERR(spisg->pclk);
	}

	clk_set_min_rate(spisg->pclk, SPISG_PCLK_RATE_MIN);

	clk_disable_unprepare(spisg->pclk);

	// Block Logic: Allocate and populate a clock divider table for dynamic SPI clock adjustment.
	tbl = devm_kzalloc(dev, sizeof(struct clk_div_table) * (DIV_NUM + 1), GFP_KERNEL);
	if (!tbl)
		return -ENOMEM;

	for (i = 0; i < DIV_NUM; i++) {
		tbl[i].val = i + SPISG_CLK_DIV_MIN - 1;
		tbl[i].div = i + SPISG_CLK_DIV_MIN;
	}
	spisg->tbl = tbl;

	// Block Logic: Allocate and configure a clock divider structure.
	div = devm_kzalloc(dev, sizeof(*div), GFP_KERNEL);
	if (!div)
		return -ENOMEM;

	div->flags = CLK_DIVIDER_ROUND_CLOSEST;
	div->reg = base + SPISG_REG_CFG_BUS;
	div->shift = __bf_shf(CFG_CLK_DIV);
	div->width = CLK_DIV_WIDTH;
	div->table = tbl;

	/* Register value should not be outside of the table */
	regmap_update_bits(spisg->map, SPISG_REG_CFG_BUS, CFG_CLK_DIV,
			   FIELD_PREP(CFG_CLK_DIV, SPISG_CLK_DIV_MIN - 1));

	// Block Logic: Register the clock divider and obtain the SPI clock (sclk).
	snprintf(name, sizeof(name), "%s_div", dev_name(dev));
	init.name = name;
	init.ops = &clk_divider_ops;
	init.flags = CLK_SET_RATE_PARENT;
	init.parent_data = &(const struct clk_parent_data) {
				.fw_name = "pclk",
			   };
	init.num_parents = 1;
	div->hw.init = &init;
	ret = devm_clk_hw_register(dev, &div->hw);
	if (ret) {
		dev_err(dev, "clock registration failed\n");
		return ret;
	}

	spisg->sclk = devm_clk_hw_get_clk(dev, &div->hw, NULL);
	if (IS_ERR_OR_NULL(spisg->sclk)) {
		dev_err(dev, "get clock failed\n");
		return PTR_ERR(spisg->sclk);
	}

	clk_prepare_enable(spisg->sclk);

	return 0;
}
/**
 * @brief Probe function for the Amlogic SPISG platform driver.
 * This function is called when the kernel finds a device that matches this driver.
 * It initializes the SPI controller, allocates resources, sets up clocks,
 * configures initial SPISG registers, and registers the SPI controller with the kernel.
 * @param pdev Pointer to the `platform_device` structure.
 * @return `0` on success, or a negative error code on failure.
 */
static int aml_spisg_probe(struct platform_device *pdev)
{
	struct spi_controller *ctlr;
	struct spisg_device *spisg;
	struct device *dev = &pdev->dev;
	void __iomem *base;
	int ret, irq;

	const struct regmap_config aml_regmap_config = {
		.reg_bits = 32,
		.val_bits = 32,
		.reg_stride = 4,
		.max_register = SPISG_MAX_REG,
	};

	// Block Logic: Allocate SPI controller (host or target) based on device tree property.
	if (of_property_read_bool(dev->of_node, "spi-slave"))
		ctlr = spi_alloc_target(dev, sizeof(*spisg));
	else
		ctlr = spi_alloc_host(dev, sizeof(*spisg));
	if (!ctlr)
		return dev_err_probe(dev, -ENOMEM, "controller allocation failed\n");

	spisg = spi_controller_get_devdata(ctlr);
	spisg->controller = ctlr;

	spisg->pdev = pdev;
	platform_set_drvdata(pdev, spisg);

	// Block Logic: Map device memory regions to virtual addresses.
	base = devm_platform_ioremap_resource(pdev, 0);
	if (IS_ERR(base))
		return dev_err_probe(dev, PTR_ERR(base), "resource ioremap failed\n");

	// Block Logic: Initialize the regmap interface for register access.
	spisg->map = devm_regmap_init_mmio(dev, base, &aml_regmap_config);
	if (IS_ERR(spisg->map))
		return dev_err_probe(dev, PTR_ERR(spisg->map), "regmap init failed\n");

	irq = platform_get_irq(pdev, 0); // Get interrupt number
	if (irq < 0) {
		ret = irq;
		goto out_controller;
	}

	// Block Logic: Perform device reset if available and configured.
	ret = device_reset_optional(dev);
	if (ret)
		return dev_err_probe(dev, ret, "reset dev failed\n");

	// Block Logic: Initialize and configure device clocks.
	ret = aml_spisg_clk_init(spisg, base);
	if (ret)
		return dev_err_probe(dev, ret, "clock init failed\n");

	// Block Logic: Initialize cached register values and set default configurations.
	spisg->cfg_spi = 0;
	spisg->cfg_start = 0;
	spisg->cfg_bus = 0;

	spisg->cfg_spi = FIELD_PREP(CFG_SFLASH_WP, 1) |
			 FIELD_PREP(CFG_SFLASH_HD, 1);
	if (spi_controller_is_target(ctlr)) {
		spisg->cfg_spi |= FIELD_PREP(CFG_SLAVE_EN, 1);
		spisg->cfg_bus = FIELD_PREP(CFG_TX_TUNING, 0xf);
	}
	/* default pending */
	spisg->cfg_start = FIELD_PREP(CFG_PEND, 1);

	// Block Logic: Enable runtime PM and set initial active state.
	pm_runtime_set_active(&spisg->pdev->dev);
	pm_runtime_enable(&spisg->pdev->dev);
	pm_runtime_resume_and_get(&spisg->pdev->dev);

	// Block Logic: Configure generic SPI controller parameters.
	ctlr->num_chipselect = 4;
	ctlr->dev.of_node = pdev->dev.of_node;
	ctlr->mode_bits = SPI_CPHA | SPI_CPOL | SPI_LSB_FIRST |
			  SPI_3WIRE | SPI_TX_QUAD | SPI_RX_QUAD;
	ctlr->max_speed_hz = 1000 * 1000 * 100;
	ctlr->min_speed_hz = 1000 * 10;
	ctlr->setup = aml_spisg_setup;
	ctlr->cleanup = aml_spisg_cleanup;
	ctlr->prepare_message = aml_spisg_prepare_message;
	ctlr->transfer_one_message = aml_spisg_transfer_one_message;
	ctlr->target_abort = aml_spisg_target_abort;
	ctlr->can_dma = aml_spisg_can_dma;
	ctlr->max_dma_len = SPISG_BLOCK_MAX;
	ctlr->auto_runtime_pm = true;

	dma_set_max_seg_size(&pdev->dev, SPISG_BLOCK_MAX);

	// Block Logic: Request and register the interrupt handler.
	ret = devm_request_irq(&pdev->dev, irq, aml_spisg_irq, 0, NULL, spisg);
	if (ret) {
		dev_err(&pdev->dev, "irq request failed\n");
		goto out_clk;
	}

	// Block Logic: Register the SPI controller with the kernel's SPI subsystem.
	ret = devm_spi_register_controller(dev, ctlr);
	if (ret) {
		dev_err(&pdev->dev, "spi controller registration failed\n");
		goto out_clk;
	}

	init_completion(&spisg->completion); // Initialize completion object

	pm_runtime_put(&spisg->pdev->dev); // Balance runtime PM reference

	return 0;

out_clk:
	// Block Logic: Clean up clocks on error during probe.
	if (spisg->core)
		clk_disable_unprepare(spisg->core);
	clk_disable_unprepare(spisg->pclk);
out_controller:
	spi_controller_put(ctlr); // Release SPI controller

	return ret;
}

/**
 * @brief Remove function for the Amlogic SPISG platform driver.
 * This function is called when the device is unbound from the driver.
 * It handles runtime PM cleanup, disables clocks, and unregisters the device.
 * @param pdev Pointer to the `platform_device` structure.
 */
static void aml_spisg_remove(struct platform_device *pdev)
{
	struct spisg_device *spisg = platform_get_drvdata(pdev);

	// Block Logic: If not runtime suspended, select sleep state and disable clocks.
	if (!pm_runtime_suspended(&pdev->dev)) {
		pinctrl_pm_select_sleep_state(&spisg->pdev->dev);
		clk_disable_unprepare(spisg->core);
		clk_disable_unprepare(spisg->pclk);
	}
}

/**
 * @brief Runtime suspend callback for the Amlogic SPISG device.
 * Puts the device into a low-power state by selecting sleep pin-control state
 * and disabling clocks.
 * @param dev Pointer to the `device` structure.
 * @return `0` always.
 */
static int spisg_suspend_runtime(struct device *dev)
{
	struct spisg_device *spisg = dev_get_drvdata(dev);

	pinctrl_pm_select_sleep_state(&spisg->pdev->dev); // Select sleep pin-control state
	clk_disable_unprepare(spisg->sclk); // Disable SPI clock
	clk_disable_unprepare(spisg->core); // Disable core clock

	return 0;
}

/**
 * @brief Runtime resume callback for the Amlogic SPISG device.
 * Restores the device from a low-power state by enabling clocks
 * and selecting the default pin-control state.
 * @param dev Pointer to the `device` structure.
 * @return `0` always.
 */
static int spisg_resume_runtime(struct device *dev)
{
	struct spisg_device *spisg = dev_get_drvdata(dev);

	clk_prepare_enable(spisg->core); // Enable core clock
	clk_prepare_enable(spisg->sclk); // Enable SPI clock
	pinctrl_pm_select_default_state(&spisg->pdev->dev); // Select default pin-control state

	return 0;
}

/**
 * @brief Power management operations structure for the Amlogic SPISG driver.
 * Defines the runtime suspend and resume callbacks.
 */
static const struct dev_pm_ops amlogic_spisg_pm_ops = {
	.runtime_suspend	= spisg_suspend_runtime,
	.runtime_resume		= spisg_resume_runtime,
};

/**
 * @brief Device tree match table for the Amlogic SPISG driver.
 * Used to identify compatible devices in the device tree.
 */
static const struct of_device_id amlogic_spisg_of_match[] = {
	{
		.compatible = "amlogic,a4-spisg",
	},

	{ /* sentinel */ }
};
// Functional Utility: Exports the device table for module auto-loading.
MODULE_DEVICE_TABLE(of, amlogic_spisg_of_match);

/**
 * @brief Platform driver structure for the Amlogic SPISG controller.
 * Defines the probe, remove, and power management operations for the driver.
 */
static struct platform_driver amlogic_spisg_driver = {
	.probe = aml_spisg_probe,
	.remove = aml_spisg_remove,
	.driver  = {
		.name = "amlogic-spisg",
		.of_match_table = amlogic_spisg_of_match,
		.pm = &amlogic_spisg_pm_ops,
	},
};

// Functional Utility: Registers the platform driver with the Linux kernel.
module_platform_driver(amlogic_spisg_driver);

// Functional Utility: Module description string.
MODULE_DESCRIPTION("Amlogic SPI Scatter-Gather Controller driver");
// Functional Utility: Module author string.
MODULE_AUTHOR("Sunny Luo <sunny.luo@amlogic.com>");
// Functional Utility: Module license information.
MODULE_LICENSE("GPL");
