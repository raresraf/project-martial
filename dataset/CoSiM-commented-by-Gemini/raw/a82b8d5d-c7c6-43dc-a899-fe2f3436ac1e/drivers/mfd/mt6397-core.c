/**
 * @file mt6397-core.c
 * @brief Core driver for MediaTek Multi-Function Devices (MFDs).
 * @author Flora Fu, MediaTek
 *
 * @details
 * This file implements a generic MFD core driver for a family of MediaTek
 * Power Management ICs (PMICs), including MT6323, MT6328, MT6357, MT6397, and others.
 * As an MFD driver, its primary responsibility is not to control the end-functionality
 * itself, but to initialize the main chip and register the various sub-devices
 * (e.g., RTC, regulator, audio codec) that reside on it. These sub-devices are
 * then handled by their own respective drivers.
 *
 * The driver uses the device tree's "compatible" property to identify the specific
 * PMIC model and load the appropriate configuration for its child devices and
 * interrupt controller.
 */
// SPDX-License-Identifier: GPL-2.0-only
/*
 * Copyright (c) 2014 MediaTek Inc.
 * Author: Flora Fu, MediaTek
 */

#include <linux/interrupt.h>
#include <linux/ioport.h>
#include <linux/irqdomain.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/platform_device.h>
#include <linux/regmap.h>
#include <linux/mfd/core.h>
#include <linux/mfd/mt6323/core.h>
#include <linux/mfd/mt6328/core.h>
#include <linux/mfd/mt6331/core.h>
#include <linux/mfd/mt6357/core.h>
#include <linux/mfd/mt6358/core.h>
#include <linux/mfd/mt6359/core.h>
#include <linux/mfd/mt6397/core.h>
#include <linux/mfd/mt6323/registers.h>
#include <linux/mfd/mt6328/registers.h>
#include <linux/mfd/mt6331/registers.h>
#include <linux/mfd/mt6357/registers.h>
#include <linux/mfd/mt6358/registers.h>
#include <linux/mfd/mt6359/registers.h>
#include <linux/mfd/mt6397/registers.h>

/*
 * The following resource definitions specify the memory-mapped register regions
 * and the interrupt lines for the various child devices (like RTC, keys) on
 * different PMIC models. These resources are passed to the child drivers when
 * they are probed.
 */
#define MT6323_RTC_BASE		0x8000
#define MT6323_RTC_SIZE		0x40

#define MT6357_RTC_BASE		0x0588
#define MT6357_RTC_SIZE		0x3c

#define MT6331_RTC_BASE		0x4000
#define MT6331_RTC_SIZE		0x40

#define MT6358_RTC_BASE		0x0588
#define MT6358_RTC_SIZE		0x3c

#define MT6397_RTC_BASE		0xe000
#define MT6397_RTC_SIZE		0x3e

#define MT6323_PWRC_BASE	0x8000
#define MT6323_PWRC_SIZE	0x40

static const struct resource mt6323_rtc_resources[] = {
	DEFINE_RES_MEM(MT6323_RTC_BASE, MT6323_RTC_SIZE),
	DEFINE_RES_IRQ(MT6323_IRQ_STATUS_RTC),
};

static const struct resource mt6357_rtc_resources[] = {
	DEFINE_RES_MEM(MT6357_RTC_BASE, MT6357_RTC_SIZE),
	DEFINE_RES_IRQ(MT6357_IRQ_RTC),
};

static const struct resource mt6331_rtc_resources[] = {
	DEFINE_RES_MEM(MT6331_RTC_BASE, MT6331_RTC_SIZE),
	DEFINE_RES_IRQ(MT6331_IRQ_STATUS_RTC),
};

static const struct resource mt6358_rtc_resources[] = {
	DEFINE_RES_MEM(MT6358_RTC_BASE, MT6358_RTC_SIZE),
	DEFINE_RES_IRQ(MT6358_IRQ_RTC),
};

static const struct resource mt6397_rtc_resources[] = {
	DEFINE_RES_MEM(MT6397_RTC_BASE, MT6397_RTC_SIZE),
	DEFINE_RES_IRQ(MT6397_IRQ_RTC),
};

static const struct resource mt6358_keys_resources[] = {
	DEFINE_RES_IRQ_NAMED(MT6358_IRQ_PWRKEY, "powerkey"),
	DEFINE_RES_IRQ_NAMED(MT6358_IRQ_HOMEKEY, "homekey"),
	DEFINE_RES_IRQ_NAMED(MT6358_IRQ_PWRKEY_R, "powerkey_r"),
	DEFINE_RES_IRQ_NAMED(MT6358_IRQ_HOMEKEY_R, "homekey_r"),
};

static const struct resource mt6359_keys_resources[] = {
	DEFINE_RES_IRQ_NAMED(MT6359_IRQ_PWRKEY, "powerkey"),
	DEFINE_RES_IRQ_NAMED(MT6359_IRQ_HOMEKEY, "homekey"),
	DEFINE_RES_IRQ_NAMED(MT6359_IRQ_PWRKEY_R, "powerkey_r"),
	DEFINE_RES_IRQ_NAMED(MT6359_IRQ_HOMEKEY_R, "homekey_r"),
};

static const struct resource mt6359_accdet_resources[] = {
	DEFINE_RES_IRQ_NAMED(MT6359_IRQ_ACCDET, "accdet_irq"),
	DEFINE_RES_IRQ_NAMED(MT6359_IRQ_ACCDET_EINT0, "accdet_eint0"),
	DEFINE_RES_IRQ_NAMED(MT6359_IRQ_ACCDET_EINT1, "accdet_eint1"),
};

static const struct resource mt6323_keys_resources[] = {
	DEFINE_RES_IRQ_NAMED(MT6323_IRQ_STATUS_PWRKEY, "powerkey"),
	DEFINE_RES_IRQ_NAMED(MT6323_IRQ_STATUS_FCHRKEY, "homekey"),
};

static const struct resource mt6328_keys_resources[] = {
	DEFINE_RES_IRQ_NAMED(MT6328_IRQ_STATUS_PWRKEY, "powerkey"),
	DEFINE_RES_IRQ_NAMED(MT6328_IRQ_STATUS_HOMEKEY, "homekey"),
	DEFINE_RES_IRQ_NAMED(MT6328_IRQ_STATUS_PWRKEY_R, "powerkey_r"),
	DEFINE_RES_IRQ_NAMED(MT6328_IRQ_STATUS_HOMEKEY_R, "homekey_r"),
};

static const struct resource mt6357_keys_resources[] = {
	DEFINE_RES_IRQ_NAMED(MT6357_IRQ_PWRKEY, "powerkey"),
	DEFINE_RES_IRQ_NAMED(MT6357_IRQ_HOMEKEY, "homekey"),
	DEFINE_RES_IRQ_NAMED(MT6357_IRQ_PWRKEY_R, "powerkey_r"),
	DEFINE_RES_IRQ_NAMED(MT6357_IRQ_HOMEKEY_R, "homekey_r"),
};

static const struct resource mt6331_keys_resources[] = {
	DEFINE_RES_IRQ_NAMED(MT6331_IRQ_STATUS_PWRKEY, "powerkey"),
	DEFINE_RES_IRQ_NAMED(MT6331_IRQ_STATUS_HOMEKEY, "homekey"),
};

static const struct resource mt6397_keys_resources[] = {
	DEFINE_RES_IRQ_NAMED(MT6397_IRQ_PWRKEY, "powerkey"),
	DEFINE_RES_IRQ_NAMED(MT6397_IRQ_HOMEKEY, "homekey"),
};

static const struct resource mt6323_pwrc_resources[] = {
	DEFINE_RES_MEM(MT6323_PWRC_BASE, MT6323_PWRC_SIZE),
};


/**
 * @brief An array of `mfd_cell` structs defines the child devices for a specific
 * MFD. Each cell describes a single function on the chip, such as an RTC,
 * a regulator controller, or a keypad interface.
 */
static const struct mfd_cell mt6323_devs[] = {
	{
		.name = "mt6323-rtc",
		.num_resources = ARRAY_SIZE(mt6323_rtc_resources),
		.resources = mt6323_rtc_resources,
		.of_compatible = "mediatek,mt6323-rtc",
	}, {
		.name = "mt6323-regulator",
		.of_compatible = "mediatek,mt6323-regulator"
	}, {
		.name = "mt6323-led",
		.of_compatible = "mediatek,mt6323-led"
	}, {
		.name = "mt6323-keys",
		.num_resources = ARRAY_SIZE(mt6323_keys_resources),
		.resources = mt6323_keys_resources,
		.of_compatible = "mediatek,mt6323-keys"
	}, {
		.name = "mt6323-pwrc",
		.num_resources = ARRAY_SIZE(mt6323_pwrc_resources),
		.resources = mt6323_pwrc_resources,
		.of_compatible = "mediatek,mt6323-pwrc"
	},
};

static const struct mfd_cell mt6328_devs[] = {
	{
		.name = "mt6328-regulator",
		.of_compatible = "mediatek,mt6328-regulator"
	}, {
		.name = "mt6328-keys",
		.num_resources = ARRAY_SIZE(mt6328_keys_resources),
		.resources = mt6328_keys_resources,
		.of_compatible = "mediatek,mt6328-keys"
	},
};

static const struct mfd_cell mt6357_devs[] = {
	{
		.name = "mt6359-auxadc",
		.of_compatible = "mediatek,mt6357-auxadc"
	}, {
		.name = "mt6357-regulator",
	}, {
		.name = "mt6357-rtc",
		.num_resources = ARRAY_SIZE(mt6357_rtc_resources),
		.resources = mt6357_rtc_resources,
		.of_compatible = "mediatek,mt6357-rtc",
	}, {
		.name = "mt6357-sound",
		.of_compatible = "mediatek,mt6357-sound"
	}, {
		.name = "mt6357-keys",
		.num_resources = ARRAY_SIZE(mt6357_keys_resources),
		.resources = mt6357_keys_resources,
		.of_compatible = "mediatek,mt6357-keys"
	},
};

/* MT6331 is always used in combination with MT6332 */
static const struct mfd_cell mt6331_mt6332_devs[] = {
	{
		.name = "mt6331-rtc",
		.num_resources = ARRAY_SIZE(mt6331_rtc_resources),
		.resources = mt6331_rtc_resources,
		.of_compatible = "mediatek,mt6331-rtc",
	}, {
		.name = "mt6331-regulator",
		.of_compatible = "mediatek,mt6331-regulator"
	}, {
		.name = "mt6332-regulator",
		.of_compatible = "mediatek,mt6332-regulator"
	}, {
		.name = "mt6331-keys",
		.num_resources = ARRAY_SIZE(mt6331_keys_resources),
		.resources = mt6331_keys_resources,
		.of_compatible = "mediatek,mt6331-keys"
	},
};

static const struct mfd_cell mt6358_devs[] = {
	{
		.name = "mt6359-auxadc",
		.of_compatible = "mediatek,mt6358-auxadc"
	}, {
		.name = "mt6358-regulator",
		.of_compatible = "mediatek,mt6358-regulator"
	}, {
		.name = "mt6358-rtc",
		.num_resources = ARRAY_SIZE(mt6358_rtc_resources),
		.resources = mt6358_rtc_resources,
		.of_compatible = "mediatek,mt6358-rtc",
	}, {
		.name = "mt6358-sound",
		.of_compatible = "mediatek,mt6358-sound"
	}, {
		.name = "mt6358-keys",
		.num_resources = ARRAY_SIZE(mt6358_keys_resources),
		.resources = mt6358_keys_resources,
		.of_compatible = "mediatek,mt6358-keys"
	},
};

static const struct mfd_cell mt6359_devs[] = {
	{
		.name = "mt6359-auxadc",
		.of_compatible = "mediatek,mt6359-auxadc"
	},
	{ .name = "mt6359-regulator", },
	{
		.name = "mt6359-rtc",
		.num_resources = ARRAY_SIZE(mt6358_rtc_resources),
		.resources = mt6358_rtc_resources,
		.of_compatible = "mediatek,mt6358-rtc",
	},
	{ .name = "mt6359-sound", },
	{
		.name = "mt6359-keys",
		.num_resources = ARRAY_SIZE(mt6359_keys_resources),
		.resources = mt6359_keys_resources,
		.of_compatible = "mediatek,mt6359-keys"
	},
	{
		.name = "mt6359-accdet",
		.of_compatible = "mediatek,mt6359-accdet",
		.num_resources = ARRAY_SIZE(mt6359_accdet_resources),
		.resources = mt6359_accdet_resources,
	},
};

static const struct mfd_cell mt6397_devs[] = {
	{
		.name = "mt6397-rtc",
		.num_resources = ARRAY_SIZE(mt6397_rtc_resources),
		.resources = mt6397_rtc_resources,
		.of_compatible = "mediatek,mt6397-rtc",
	}, {
		.name = "mt6397-regulator",
		.of_compatible = "mediatek,mt6397-regulator",
	}, {
		.name = "mt6397-codec",
		.of_compatible = "mediatek,mt6397-codec",
	}, {
		.name = "mt6397-clk",
		.of_compatible = "mediatek,mt6397-clk",
	}, {
		.name = "mt6397-pinctrl",
		.of_compatible = "mediatek,mt6397-pinctrl",
	}, {
		.name = "mt6397-keys",
		.num_resources = ARRAY_SIZE(mt6397_keys_resources),
		.resources = mt6397_keys_resources,
		.of_compatible = "mediatek,mt6397-keys"
	}
};

/**
 * @struct chip_data
 * @brief A structure to hold chip-specific data.
 *
 * This allows the driver to be generic by abstracting the differences
 * between the various supported PMIC models. An instance of this struct
 * is associated with each "compatible" string.
 */
struct chip_data {
	u32 cid_addr;     /**< Register address for the chip ID. */
	u32 cid_shift;    /**< Bit shift needed to extract the chip ID from the register. */
	const struct mfd_cell *cells; /**< Pointer to the array of child devices for this chip. */
	int cell_size;    /**< The number of child devices in the `cells` array. */
	int (*irq_init)(struct mt6397_chip *chip); /**< Function pointer to the chip-specific IRQ initializer. */
};

/* Data structures defining the specific properties for each PMIC variant. */

static const struct chip_data mt6323_core = {
	.cid_addr = MT6323_CID,
	.cid_shift = 0,
	.cells = mt6323_devs,
	.cell_size = ARRAY_SIZE(mt6323_devs),
	.irq_init = mt6397_irq_init,
};

static const struct chip_data mt6328_core = {
	.cid_addr = MT6328_HWCID,
	.cid_shift = 0,
	.cells = mt6328_devs,
	.cell_size = ARRAY_SIZE(mt6328_devs),
	.irq_init = mt6397_irq_init,
};

static const struct chip_data mt6357_core = {
	.cid_addr = MT6357_SWCID,
	.cid_shift = 8,
	.cells = mt6357_devs,
	.cell_size = ARRAY_SIZE(mt6357_devs),
	.irq_init = mt6358_irq_init,
};

static const struct chip_data mt6331_mt6332_core = {
	.cid_addr = MT6331_HWCID,
	.cid_shift = 0,
	.cells = mt6331_mt6332_devs,
	.cell_size = ARRAY_SIZE(mt6331_mt6332_devs),
	.irq_init = mt6397_irq_init,
};

static const struct chip_data mt6358_core = {
	.cid_addr = MT6358_SWCID,
	.cid_shift = 8,
	.cells = mt6358_devs,
	.cell_size = ARRAY_SIZE(mt6358_devs),
	.irq_init = mt6358_irq_init,
};

static const struct chip_data mt6359_core = {
	.cid_addr = MT6359_SWCID,
	.cid_shift = 8,
	.cells = mt6359_devs,
	.cell_size = ARRAY_SIZE(mt6359_devs),
	.irq_init = mt6358_irq_init,
};

static const struct chip_data mt6397_core = {
	.cid_addr = MT6397_CID,
	.cid_shift = 0,
	.cells = mt6397_devs,
	.cell_size = ARRAY_SIZE(mt6397_devs),
	.irq_init = mt6397_irq_init,
};

/**
 * @brief The probe function, called when a matching device is found in the device tree.
 *
 * @param pdev The platform device structure.
 * @return 0 on success, or a negative error code on failure.
 *
 * @details This function is the main entry point for the driver. It performs
 * the following steps:
 * 1. Allocates a private data structure for the device.
 * 2. Gets the `regmap` from the parent bus device (e.g., I2C).
 * 3. Matches the device's "compatible" string to get the correct `chip_data`.
 * 4. Reads the chip ID from the hardware to verify the model.
 * 5. Initializes the chip's interrupt controller via the `irq_init` function pointer.
 * 6. Calls `devm_mfd_add_devices` to register all the child devices with the kernel.
 */
static int mt6397_probe(struct platform_device *pdev)
{
	int ret;
	unsigned int id = 0;
	struct mt6397_chip *pmic;
	const struct chip_data *pmic_core;

	pmic = devm_kzalloc(&pdev->dev, sizeof(*pmic), GFP_KERNEL);
	if (!pmic)
		return -ENOMEM;

	pmic->dev = &pdev->dev;

	// The `regmap` provides a unified way to access device registers, abstracting
	// the underlying bus (e.g., I2C or SPI).
	pmic->regmap = dev_get_regmap(pdev->dev.parent, NULL);
	if (!pmic->regmap)
		return -ENODEV;

	// Get the chip-specific data based on the device tree "compatible" string.
	pmic_core = of_device_get_match_data(&pdev->dev);
	if (!pmic_core)
		return -ENODEV;

	// Read the chip ID register to confirm we are talking to the correct hardware.
	ret = regmap_read(pmic->regmap, pmic_core->cid_addr, &id);
	if (ret) {
		dev_err(&pdev->dev, "Failed to read chip id: %d
", ret);
		return ret;
	}
	pmic->chip_id = (id >> pmic_core->cid_shift) & 0xff;

	platform_set_drvdata(pdev, pmic);

	pmic->irq = platform_get_irq(pdev, 0);
	if (pmic->irq <= 0)
		return pmic->irq;

	// Call the chip-specific IRQ initialization function.
	ret = pmic_core->irq_init(pmic);
	if (ret)
		return ret;

	// Register all the child devices defined in the `mfd_cell` array.
	ret = devm_mfd_add_devices(&pdev->dev, PLATFORM_DEVID_NONE,
				   pmic_core->cells, pmic_core->cell_size,
				   NULL, 0, pmic->irq_domain);
	if (ret) {
		irq_domain_remove(pmic->irq_domain);
		dev_err(&pdev->dev, "failed to add child devices: %d
", ret);
	}

	return ret;
}

/**
 * @brief An array that maps device tree "compatible" strings to the `chip_data`
 * for each supported PMIC. This is how the kernel's driver model knows which
 * data to pass to the probe function for a given device.
 */
static const struct of_device_id mt6397_of_match[] = {
	{
		.compatible = "mediatek,mt6323",
		.data = &mt6323_core,
	}, {
		.compatible = "mediatek,mt6328",
		.data = &mt6328_core,
	}, {
		.compatible = "mediatek,mt6331",
		.data = &mt6331_mt6332_core,
	}, {
		.compatible = "mediatek,mt6357",
		.data = &mt6357_core,
	}, {
		.compatible = "mediatek,mt6358",
		.data = &mt6358_core,
	}, {
		.compatible = "mediatek,mt6359",
		.data = &mt6359_core,
	}, {
		.compatible = "mediatek,mt6397",
		.data = &mt6397_core,
	}, {
		/* sentinel */
	}
};
MODULE_DEVICE_TABLE(of, mt6397_of_match);

static const struct platform_device_id mt6397_id[] = {
	{ "mt6397", 0 },
	{ },
};
MODULE_DEVICE_TABLE(platform, mt6397_id);

static struct platform_driver mt6397_driver = {
	.probe = mt6397_probe,
	.driver = {
		.name = "mt6397",
		.of_match_table = mt6397_of_match,
	},
	.id_table = mt6397_id,
};

module_platform_driver(mt6397_driver);

MODULE_AUTHOR("Flora Fu, MediaTek");
MODULE_DESCRIPTION("Driver for MediaTek MT6397 PMIC");
MODULE_LICENSE("GPL");
