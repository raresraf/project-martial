/**
 * @file mt6397-core.c
 * @brief Core driver for MediaTek Multi-Function PMICs (Power Management ICs).
 *
 * This driver acts as a parent for a variety of MediaTek MFDs, such as the
 * MT6323, MT6358, and MT6397. Its primary responsibility is to initialize the
 * main chip, identify it, set up its interrupt controller, and then register
 * its various sub-functions (like RTC, audio codec, regulator, etc.) as
 * separate platform devices. This allows dedicated drivers for each function
 * to bind to their respective hardware blocks. The driver uses a data-driven
 * approach to support multiple chip variants with a single codebase.
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
 * These resource definitions describe the hardware resources (memory-mapped
 * register regions and IRQ lines) for the sub-devices within each PMIC.
 * These resources will be assigned to the child platform_devices when they
 * are created.
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

/* Resources for the MT6323 real-time clock */
static const struct resource mt6323_rtc_resources[] = {
	DEFINE_RES_MEM(MT6323_RTC_BASE, MT6323_RTC_SIZE),
	DEFINE_RES_IRQ(MT6323_IRQ_STATUS_RTC),
};

/* Resources for the MT6357 real-time clock */
static const struct resource mt6357_rtc_resources[] = {
	DEFINE_RES_MEM(MT6357_RTC_BASE, MT6357_RTC_SIZE),
	DEFINE_RES_IRQ(MT6357_IRQ_RTC),
};

/* Resources for the MT6331 real-time clock */
static const struct resource mt6331_rtc_resources[] = {
	DEFINE_RES_MEM(MT6331_RTC_BASE, MT6331_RTC_SIZE),
	DEFINE_RES_IRQ(MT6331_IRQ_STATUS_RTC),
};

/* Resources for the MT6358 real-time clock */
static const struct resource mt6358_rtc_resources[] = {
	DEFINE_RES_MEM(MT6358_RTC_BASE, MT6358_RTC_SIZE),
	DEFINE_RES_IRQ(MT6358_IRQ_RTC),
};

/* Resources for the MT6397 real-time clock */
static const struct resource mt6397_rtc_resources[] = {
	DEFINE_RES_MEM(MT6397_RTC_BASE, MT6397_RTC_SIZE),
	DEFINE_RES_IRQ(MT6397_IRQ_RTC),
};

/* Resources for the MT6358 power/home keys */
static const struct resource mt6358_keys_resources[] = {
	DEFINE_RES_IRQ_NAMED(MT6358_IRQ_PWRKEY, "powerkey"),
	DEFINE_RES_IRQ_NAMED(MT6358_IRQ_HOMEKEY, "homekey"),
	DEFINE_RES_IRQ_NAMED(MT6358_IRQ_PWRKEY_R, "powerkey_r"),
	DEFINE_RES_IRQ_NAMED(MT6358_IRQ_HOMEKEY_R, "homekey_r"),
};

/* Resources for the MT6359 power/home keys */
static const struct resource mt6359_keys_resources[] = {
	DEFINE_RES_IRQ_NAMED(MT6359_IRQ_PWRKEY, "powerkey"),
	DEFINE_RES_IRQ_NAMED(MT6359_IRQ_HOMEKEY, "homekey"),
	DEFINE_RES_IRQ_NAMED(MT6359_IRQ_PWRKEY_R, "powerkey_r"),
	DEFINE_RES_IRQ_NAMED(MT6359_IRQ_HOMEKEY_R, "homekey_r"),
};

/* Resources for the MT6359 accessory detection block */
static const struct resource mt6359_accdet_resources[] = {
	DEFINE_RES_IRQ_NAMED(MT6359_IRQ_ACCDET, "accdet_irq"),
	DEFINE_RES_IRQ_NAMED(MT6359_IRQ_ACCDET_EINT0, "accdet_eint0"),
	DEFINE_RES_IRQ_NAMED(MT6359_IRQ_ACCDET_EINT1, "accdet_eint1"),
};

/* Resources for the MT6323 power/home keys */
static const struct resource mt6323_keys_resources[] = {
	DEFINE_RES_IRQ_NAMED(MT6323_IRQ_STATUS_PWRKEY, "powerkey"),
	DEFINE_RES_IRQ_NAMED(MT6323_IRQ_STATUS_FCHRKEY, "homekey"),
};

/* Resources for the MT6328 power/home keys */
static const struct resource mt6328_keys_resources[] = {
	DEFINE_RES_IRQ_NAMED(MT6328_IRQ_STATUS_PWRKEY, "powerkey"),
	DEFINE_RES_IRQ_NAMED(MT6328_IRQ_STATUS_HOMEKEY, "homekey"),
	DEFINE_RES_IRQ_NAMED(MT6328_IRQ_STATUS_PWRKEY_R, "powerkey_r"),
	DEFINE_RES_IRQ_NAMED(MT6328_IRQ_STATUS_HOMEKEY_R, "homekey_r"),
};

/* Resources for the MT6357 power/home keys */
static const struct resource mt6357_keys_resources[] = {
	DEFINE_RES_IRQ_NAMED(MT6357_IRQ_PWRKEY, "powerkey"),
	DEFINE_RES_IRQ_NAMED(MT6357_IRQ_HOMEKEY, "homekey"),
	DEFINE_RES_IRQ_NAMED(MT6357_IRQ_PWRKEY_R, "powerkey_r"),
	DEFINE_RES_IRQ_NAMED(MT6357_IRQ_HOMEKEY_R, "homekey_r"),
};

/* Resources for the MT6331 power/home keys */
static const struct resource mt6331_keys_resources[] = {
	DEFINE_RES_IRQ_NAMED(MT6331_IRQ_STATUS_PWRKEY, "powerkey"),
	DEFINE_RES_IRQ_NAMED(MT6331_IRQ_STATUS_HOMEKEY, "homekey"),
};

/* Resources for the MT6397 power/home keys */
static const struct resource mt6397_keys_resources[] = {
	DEFINE_RES_IRQ_NAMED(MT6397_IRQ_PWRKEY, "powerkey"),
	DEFINE_RES_IRQ_NAMED(MT6397_IRQ_HOMEKEY, "homekey"),
};

static const struct resource mt6323_pwrc_resources[] = {
	DEFINE_RES_MEM(MT6323_PWRC_BASE, MT6323_PWRC_SIZE),
};

/**
 * @brief An array of `mfd_cell` structures that defines the sub-devices
 * for a specific PMIC model. This array acts as a blueprint for creating the
 * child platform devices.
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
		.name = "mtk-pmic-keys",
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
		.name = "mtk-pmic-keys",
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
		.name = "mtk-pmic-keys",
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
		.name = "mtk-pmic-keys",
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
		.name = "mtk-pmic-keys",
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
		.name = "mtk-pmic-keys",
		.num_resources = ARRAY_SIZE(mt6397_keys_resources),
		.resources = mt6397_keys_resources,
		.of_compatible = "mediatek,mt6397-keys"
	}
};

/**
 * @brief Consolidates chip-specific data to enable a generic probe function.
 *
 * This struct holds pointers to chip-specific cells, register addresses,
 * and initialization functions, allowing the probe function to be data-driven.
 */
struct chip_data {
	u32 cid_addr; /* Chip ID register address */
	u32 cid_shift; /* Bit shift for extracting the chip ID */
	const struct mfd_cell *cells; /* Array of sub-devices */
	int cell_size; /* Number of sub-devices */
	int (*irq_init)(struct mt6397_chip *chip); /* IRQ initialization function */
};

/* Static definitions of chip-specific data, one for each supported PMIC. */

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
 * @brief The main probe function for the MFD driver.
 *
 * This function is called by the kernel when a device matching one of the
 * `of_device_id` entries is found in the device tree. It identifies the specific
 * PMIC, initializes it, and registers all of its child devices.
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

	/*
	 * The PMIC is a child of another device (e.g., an I2C controller).
	 * We get the regmap from this parent device to communicate with the PMIC.
	 */
	pmic->regmap = dev_get_regmap(pdev->dev.parent, NULL);
	if (!pmic->regmap)
		return -ENODEV;

	/* Get the chip-specific data based on the device tree "compatible" string. */
	pmic_core = of_device_get_match_data(&pdev->dev);
	if (!pmic_core)
		return -ENODEV;

	/* Read the hardware Chip ID register to verify the device. */
	ret = regmap_read(pmic->regmap, pmic_core->cid_addr, &id);
	if (ret) {
		dev_err(&pdev->dev, "Failed to read chip id: %d\n", ret);
		return ret;
	}

	pmic->chip_id = (id >> pmic_core->cid_shift) & 0xff;

	platform_set_drvdata(pdev, pmic);

	/* Get the main interrupt line for the PMIC from the platform device. */
	pmic->irq = platform_get_irq(pdev, 0);
	if (pmic->irq <= 0)
		return pmic->irq;

	/* Call the chip-specific interrupt initialization function. */
	ret = pmic_core->irq_init(pmic);
	if (ret)
		return ret;

	/*
	 * Core MFD Logic: Add all the child devices defined in the `cells`
	 * array. The kernel will then probe for drivers for these new devices.
	 */
	ret = devm_mfd_add_devices(&pdev->dev, PLATFORM_DEVID_NONE,
				   pmic_core->cells, pmic_core->cell_size,
				   NULL, 0, pmic->irq_domain);
	if (ret) {
		irq_domain_remove(pmic->irq_domain);
		dev_err(&pdev->dev, "failed to add child devices: %d\n", ret);
	}

	return ret;
}

/**
 * @brief This structure maps the device tree "compatible" strings to their
 * corresponding `chip_data`. This is how the driver knows which chip-specific
 * data to use for a given device.
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

/*
 * @brief The platform_driver structure, which registers the probe function
 * and the device tree match table with the kernel's platform bus.
 */
static struct platform_driver mt6397_driver = {
	.probe = mt6397_probe,
	.driver = {
		.name = "mt6397",
		.of_match_table = mt6397_of_match,
	},
	.id_table = mt6397_id,
};

/*
 * This macro registers the platform driver with the kernel. It is a wrapper
 * around module_init() for platform drivers.
 */
module_platform_driver(mt6397_driver);

MODULE_AUTHOR("Flora Fu, MediaTek");
MODULE_DESCRIPTION("Driver for MediaTek MT6397 PMIC");
MODULE_LICENSE("GPL");
