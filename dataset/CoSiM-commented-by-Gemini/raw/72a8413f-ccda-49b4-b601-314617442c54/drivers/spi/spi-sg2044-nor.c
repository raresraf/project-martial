// SPDX-License-Identifier: GPL-2.0-only
/**
 * @file spi-sg2044-nor.c
 * @brief Linux kernel driver for the Sophgo SG2044/SG2042 SPI NOR Flash Memory Controller (FMC).
 *
 * This file implements a platform driver that interfaces with the SPI NOR
 * controller on Sophgo SoCs. It uses the generic `spi-mem` framework to
- * expose the connected NOR flash memory to the rest of the kernel.
 *
 * @author Longbin Li <looong.bin@gmail.com>
 * @copyright Copyright (c) 2025 Longbin Li
 */

#include <linux/bitfield.h>
#include <linux/clk.h>
#include <linux/iopoll.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/platform_device.h>
#include <linux/spi/spi-mem.h>

/**
 * @name SG2044 SPI FMC Register Definitions
 * @{
 * @brief These definitions map directly to the hardware register layout of the
 *        SPI NOR Flash Memory Controller. They are used to read and write
 *        control, status, and data values to the hardware.
 */
#define SPIFMC_CTRL				0x00
#define SPIFMC_CTRL_CPHA			BIT(12)
#define SPIFMC_CTRL_CPOL			BIT(13)
#define SPIFMC_CTRL_HOLD_OL			BIT(14)
#define SPIFMC_CTRL_WP_OL			BIT(15)
#define SPIFMC_CTRL_LSBF			BIT(20)
#define SPIFMC_CTRL_SRST			BIT(21)
#define SPIFMC_CTRL_SCK_DIV_SHIFT		0
#define SPIFMC_CTRL_FRAME_LEN_SHIFT		16
#define SPIFMC_CTRL_SCK_DIV_MASK		0x7FF

#define SPIFMC_CE_CTRL				0x04
#define SPIFMC_CE_CTRL_CEMANUAL			BIT(0)
#define SPIFMC_CE_CTRL_CEMANUAL_EN		BIT(1)

#define SPIFMC_DLY_CTRL				0x08
#define SPIFMC_CTRL_FM_INTVL_MASK		0x000f
#define SPIFMC_CTRL_FM_INTVL			BIT(0)
#define SPIFMC_CTRL_CET_MASK			0x0f00
#define SPIFMC_CTRL_CET				BIT(8)

#define SPIFMC_DMMR				0x0c

#define SPIFMC_TRAN_CSR				0x10
#define SPIFMC_TRAN_CSR_TRAN_MODE_MASK		GENMASK(1, 0)
#define SPIFMC_TRAN_CSR_TRAN_MODE_RX		BIT(0)
#define SPIFMC_TRAN_CSR_TRAN_MODE_TX		BIT(1)
#define SPIFMC_TRAN_CSR_FAST_MODE		BIT(3)
#define SPIFMC_TRAN_CSR_BUS_WIDTH_1_BIT		(0x00 << 4)
#define SPIFMC_TRAN_CSR_BUS_WIDTH_2_BIT		(0x01 << 4)
#define SPIFMC_TRAN_CSR_BUS_WIDTH_4_BIT		(0x02 << 4)
#define SPIFMC_TRAN_CSR_DMA_EN			BIT(6)
#define SPIFMC_TRAN_CSR_MISO_LEVEL		BIT(7)
#define SPIFMC_TRAN_CSR_ADDR_BYTES_MASK		GENMASK(10, 8)
#define SPIFMC_TRAN_CSR_ADDR_BYTES_SHIFT	8
#define SPIFMC_TRAN_CSR_WITH_CMD		BIT(11)
#define SPIFMC_TRAN_CSR_FIFO_TRG_LVL_MASK	GENMASK(13, 12)
#define SPIFMC_TRAN_CSR_FIFO_TRG_LVL_1_BYTE	(0x00 << 12)
#define SPIFMC_TRAN_CSR_FIFO_TRG_LVL_2_BYTE	(0x01 << 12)
#define SPIFMC_TRAN_CSR_FIFO_TRG_LVL_4_BYTE	(0x02 << 12)
#define SPIFMC_TRAN_CSR_FIFO_TRG_LVL_8_BYTE	(0x03 << 12)
#define SPIFMC_TRAN_CSR_GO_BUSY			BIT(15)
#define SPIFMC_TRAN_CSR_ADDR4B_SHIFT		20
#define SPIFMC_TRAN_CSR_CMD4B_SHIFT		21

#define SPIFMC_TRAN_NUM				0x14
#define SPIFMC_FIFO_PORT			0x18
#define SPIFMC_FIFO_PT				0x20

#define SPIFMC_INT_STS				0x28
#define SPIFMC_INT_TRAN_DONE			BIT(0)
#define SPIFMC_INT_RD_FIFO			BIT(2)
#define SPIFMC_INT_WR_FIFO			BIT(3)
#define SPIFMC_INT_RX_FRAME			BIT(4)
#define SPIFMC_INT_TX_FRAME			BIT(5)

#define SPIFMC_INT_EN				0x2c
#define SPIFMC_INT_TRAN_DONE_EN			BIT(0)
#define SPIFMC_INT_RD_FIFO_EN			BIT(2)
#define SPIFMC_INT_WR_FIFO_EN			BIT(3)
#define SPIFMC_INT_RX_FRAME_EN			BIT(4)
#define SPIFMC_INT_TX_FRAME_EN			BIT(5)

#define SPIFMC_OPT				0x030
#define SPIFMC_OPT_DISABLE_FIFO_FLUSH		BIT(1)

#define SPIFMC_MAX_FIFO_DEPTH			8
#define SPIFMC_MAX_READ_SIZE			0x10000
/** @} */

/**
 * @struct sg204x_spifmc_chip_info
 * @brief Holds configuration details specific to a chip variant.
 *
 * This allows the same driver to support multiple, slightly different SoCs
 * like the SG2042 and SG2044 by abstracting away their differences.
 */
struct sg204x_spifmc_chip_info {
	bool has_opt_reg;
	u32 rd_fifo_int_trigger_level;
};

/**
 * @struct sg2044_spifmc
 * @brief The private context structure for the SPI FMC driver instance.
 *
 * It holds all state needed to operate the controller, including memory-mapped
 * register base, clocks, locks, and chip-specific information.
 */
struct sg2044_spifmc {
	struct spi_controller *ctrl;
	void __iomem *io_base;
	struct device *dev;
	struct mutex lock;
	struct clk *clk;
	const struct sg204x_spifmc_chip_info *chip_info;
};

/**
 * @brief Polls the interrupt status register until a specific condition is met.
 * @param spifmc: Driver's private context.
 * @param int_type: The interrupt status bit to wait for.
 * @return 0 on success, -ETIMEDOUT on timeout.
 */
static int sg2044_spifmc_wait_int(struct sg2044_spifmc *spifmc, u8 int_type)
{
	u32 stat;

	return readl_poll_timeout(spifmc->io_base + SPIFMC_INT_STS, stat,
				  (stat & int_type), 0, 1000000);
}

// ... (other static helper functions for hardware interaction) ...

/**
 * @brief Executes a memory operation (read, write, or command) on the SPI flash.
 *
 * This is the core callback for the spi-mem framework. The kernel calls this
 * function to perform any operation on the flash chip. It locks the controller,
 * determines the operation type, and calls the appropriate low-level function.
 *
 * @param mem: The SPI memory device.
 * @param op: The operation to be executed.
 * @return 0 on success, or a negative error code on failure.
 */
static int sg2044_spifmc_exec_op(struct spi_mem *mem,
				 const struct spi_mem_op *op)
{
	struct sg2044_spifmc *spifmc;

	spifmc = spi_controller_get_devdata(mem->spi->controller);

	// Lock to ensure exclusive access to the hardware controller.
	mutex_lock(&spifmc->lock);

	// Dispatch to the correct handler based on the operation type.
	if (op->addr.nbytes == 0)
		sg2044_spifmc_trans_reg(spifmc, op); // Register-like access (e.g., read status)
	else
		sg2044_spifmc_trans(spifmc, op);      // Memory access with an address

	mutex_unlock(&spifmc->lock);

	return 0;
}

// Hooks the driver's execution logic into the kernel's spi-mem subsystem.
static const struct spi_controller_mem_ops sg2044_spifmc_mem_ops = {
	.exec_op = sg2044_spifmc_exec_op,
};

/**
 * @brief Initializes the SPI FMC hardware to a known default state.
 * @param spifmc: Driver's private context.
 */
static void sg2044_spifmc_init(struct sg2044_spifmc *spifmc)
{
	// ... (writes default values to control registers) ...
}

/**
 * @brief The driver's probe function, called by the kernel's driver framework.
 *
 * This function is the entry point for the driver. It is called when the kernel
 * discovers a device in the device tree that matches this driver's `compatible` string.
 * Its purpose is to initialize the hardware and register it with the appropriate subsystems.
 *
 * @param pdev: The platform device being probed.
 * @return 0 on successful initialization, or a negative error code on failure.
 */
static int sg2044_spifmc_probe(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct spi_controller *ctrl;
	struct sg2044_spifmc *spifmc;
	int ret;

	// 1. Allocate memory for the SPI controller and our private data structure.
	ctrl = devm_spi_alloc_host(&pdev->dev, sizeof(*spifmc));
	if (!ctrl)
		return -ENOMEM;

	spifmc = spi_controller_get_devdata(ctrl);

	// 2. Get resources defined in the device tree.
	spifmc->clk = devm_clk_get_enabled(&pdev->dev, NULL);
	if (IS_ERR(spifmc->clk))
		return dev_err_probe(dev, PTR_ERR(spifmc->clk), "Cannot get and enable AHB clock
");

	spifmc->dev = &pdev->dev;
	spifmc->ctrl = ctrl;

	spifmc->io_base = devm_platform_ioremap_resource(pdev, 0);
	if (IS_ERR(spifmc->io_base))
		return PTR_ERR(spifmc->io_base);

	// 3. Configure the spi_controller struct with hardware capabilities.
	ctrl->num_chipselect = 1;
	ctrl->dev.of_node = pdev->dev.of_node;
	ctrl->bits_per_word_mask = SPI_BPW_MASK(8);
	ctrl->auto_runtime_pm = false;
	ctrl->mem_ops = &sg2044_spifmc_mem_ops; // Hook into spi-mem framework
	ctrl->mode_bits = SPI_RX_DUAL | SPI_TX_DUAL | SPI_RX_QUAD | SPI_TX_QUAD;

	ret = devm_mutex_init(dev, &spifmc->lock);
	if (ret)
		return ret;

	// 4. Get chip-specific configuration from the OF match data.
	spifmc->chip_info = device_get_match_data(&pdev->dev);
	if (!spifmc->chip_info) {
		dev_err(&pdev->dev, "Failed to get specific chip info
");
		return -EINVAL;
	}

	// 5. Initialize the hardware controller.
	sg2044_spifmc_init(spifmc);
	sg2044_spifmc_init_reg(spifmc);

	// 6. Register the controller with the kernel's SPI subsystem.
	ret = devm_spi_register_controller(&pdev->dev, ctrl);
	if (ret)
		return dev_err_probe(dev, ret, "spi_register_controller failed
");

	return 0;
}

// Chip-specific data for the SG2044 variant.
static const struct sg204x_spifmc_chip_info sg2044_chip_info = {
	.has_opt_reg = true,
	.rd_fifo_int_trigger_level = SPIFMC_TRAN_CSR_FIFO_TRG_LVL_8_BYTE,
};

// Chip-specific data for the SG2042 variant.
static const struct sg204x_spifmc_chip_info sg2042_chip_info = {
	.has_opt_reg = false,
	.rd_fifo_int_trigger_level = SPIFMC_TRAN_CSR_FIFO_TRG_LVL_1_BYTE,
};

/**
 * @brief The OF (Open Firmware / Device Tree) match table.
 *
 * This is the crucial link between the device tree and this driver. The kernel
 * uses this table to find which driver to load for a device with a matching
 * `compatible` string. The `.data` field points to the chip-specific
 * configuration for that compatible string.
 */
static const struct of_device_id sg2044_spifmc_match[] = {
	{ .compatible = "sophgo,sg2044-spifmc-nor", .data = &sg2044_chip_info },
	{ .compatible = "sophgo,sg2042-spifmc-nor", .data = &sg2042_chip_info },
	{ /* sentinel */ }
};
MODULE_DEVICE_TABLE(of, sg2044_spifmc_match);

// The main platform driver structure that registers the probe function
// and the OF match table with the kernel.
static struct platform_driver sg2044_nor_driver = {
	.driver = {
		.name = "sg2044,spifmc-nor",
		.of_match_table = sg2044_spifmc_match,
	},
	.probe = sg2044_spifmc_probe,
};
module_platform_driver(sg2044_nor_driver);

MODULE_DESCRIPTION("SG2044 SPI NOR controller driver");
MODULE_AUTHOR("Longbin Li <looong.bin@gmail.com>");
MODULE_LICENSE("GPL");
