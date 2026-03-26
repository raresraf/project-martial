/**
 * @file core.h
 * @brief Core driver definitions for the NXP PCF50633 Power Management IC (PMIC).
 * @description This header file provides the central data structures, register definitions,
 * and function prototypes for the PCF50633 Multi-Function Device (MFD) driver. The
 * PCF50633 is a highly integrated PMIC designed for mobile devices, providing power
 * regulation, battery charging, RTC, and interrupt management.
 *
 * (C) 2006-2008 by Openmoko, Inc.
 * All rights reserved.
 */

/* SPDX-License-Identifier: GPL-2.0-or-later */

#ifndef __LINUX_MFD_PCF50633_CORE_H
#define __LINUX_MFD_PCF50633_CORE_H

#include <linux/i2c.h>
#include <linux/workqueue.h>
#include <linux/regulator/driver.h>
#include <linux/regulator/machine.h>
#include <linux/pm.h>
#include <linux/power_supply.h>

struct pcf50633;
struct regmap;

/**
 * @def PCF50633_NUM_REGULATORS
 * @brief Total number of voltage regulators provided by the PCF50633 chip.
 */
#define PCF50633_NUM_REGULATORS	11

/**
 * @struct pcf50633_platform_data
 * @brief Board-specific configuration data for the PCF50633 driver.
 * @description This structure allows platform (board) code to provide essential
 * configuration details to the generic PCF50633 driver, such as regulator constraints,
 * battery parameters, and callbacks for system-specific actions.
 */
struct pcf50633_platform_data {
	/** @brief Initialization data for each of the chip's voltage regulators. */
	struct regulator_init_data reg_init_data[PCF50633_NUM_REGULATORS];

	/** @brief Names of power supplies (batteries) managed by this PMIC. */
	char **batteries;
	/** @brief Number of batteries listed. */
	int num_batteries;

	/**
	 * @brief The charger's reference current in milliamps.
	 * @description This value must be set according to the reference resistor
	 * used on the board to ensure correct battery charging behavior.
	 */
	int charger_reference_current_ma;

	/* Callbacks for platform-specific handling */
	/** @brief Called after the core driver has successfully probed. */
	void (*probe_done)(struct pcf50633 *);
	/** @brief Called on a Mobile Battery Charger (MBC) event. */
	void (*mbc_event_callback)(struct pcf50633 *, int);
	/** @brief Called after a regulator sub-driver has been registered. */
	void (*regulator_registered)(struct pcf50633 *, int);
	/** @brief A platform-specific function to force a system shutdown. */
	void (*force_shutdown)(struct pcf50633 *);

	/** @brief Defines which IRQs can resume the system from suspend. */
	u8 resumers[5];
};

/**
 * @struct pcf50633_irq
 * @brief Represents a registered interrupt handler for a PCF50633 IRQ.
 */
struct pcf50633_irq {
	void (*handler) (int, void *);
	void *data;
};

/* Public Functions for IRQ management */
int pcf50633_register_irq(struct pcf50633 *pcf, int irq,
			void (*handler) (int, void *), void *data);
int pcf50633_free_irq(struct pcf50633 *pcf, int irq);
int pcf50633_irq_mask(struct pcf50633 *pcf, int irq);
int pcf50633_irq_unmask(struct pcf50633 *pcf, int irq);
int pcf50633_irq_mask_get(struct pcf50633 *pcf, int irq);

/* Public Functions for register access */
int pcf50633_read_block(struct pcf50633 *, u8 reg,
					int nr_regs, u8 *data);
int pcf50633_write_block(struct pcf50633 *pcf, u8 reg,
					int nr_regs, u8 *data);
u8 pcf50633_reg_read(struct pcf50633 *, u8 reg);
int pcf50633_reg_write(struct pcf50633 *pcf, u8 reg, u8 val);
int pcf50633_reg_set_bit_mask(struct pcf50633 *pcf, u8 reg, u8 mask, u8 val);
int pcf50633_reg_clear_bits(struct pcf50633 *pcf, u8 reg, u8 bits);

/*
 * Register Map: Interrupt Status Registers
 * These registers hold the status flags for various interrupt sources.
 * Reading them clears the respective IRQ status bits.
 */
#define PCF50633_REG_INT1	0x02
#define PCF50633_REG_INT2	0x03
#define PCF50633_REG_INT3	0x04
#define PCF50633_REG_INT4	0x05
#define PCF50633_REG_INT5	0x06

/*
 * Register Map: Interrupt Mask Registers
 * These registers are used to enable or disable individual interrupt sources.
 */
#define PCF50633_REG_INT1M	0x07
#define PCF50633_REG_INT2M	0x08
#define PCF50633_REG_INT3M	0x09
#define PCF50633_REG_INT4M	0x0a
#define PCF50633_REG_INT5M	0x0b

/**
 * @enum pcf50633_irqs
 * @brief Enumeration of all interrupt sources available on the PCF50633.
 * @description Provides a symbolic name for each hardware interrupt, used for
 * registration and handling.
 */
enum {
	/* Chip IRQs */
	PCF50633_IRQ_ADPINS,
	PCF50633_IRQ_ADPREM,
	PCF50633_IRQ_USBINS,
	PCF50633_IRQ_USBREM,
	PCF50633_IRQ_RESERVED1,
	PCF50633_IRQ_RESERVED2,
	PCF50633_IRQ_ALARM,
	PCF50633_IRQ_SECOND,
	PCF50633_IRQ_ONKEYR,
	PCF50633_IRQ_ONKEYF,
	PCF50633_IRQ_EXTON1R,
	PCF50633_IRQ_EXTON1F,
	PCF50633_IRQ_EXTON2R,
	PCF50633_IRQ_EXTON2F,
	PCF50633_IRQ_EXTON3R,
	PCF50633_IRQ_EXTON3F,
	PCF50633_IRQ_BATFULL,
	PCF50633_IRQ_CHGHALT,
	PCF50633_IRQ_THLIMON,
	PCF50633_IRQ_THLIMOFF,
	PCF50633_IRQ_USBLIMON,
	PCF50633_IRQ_USBLIMOFF,
	PCF50633_IRQ_ADCRDY,
	PCF50633_IRQ_ONKEY1S,
	PCF50633_IRQ_LOWSYS,
	PCF50633_IRQ_LOWBAT,
	PCF50633_IRQ_HIGHTMP,
	PCF50633_IRQ_AUTOPWRFAIL,
	PCF50633_IRQ_DWN1PWRFAIL,
	PCF50633_IRQ_DWN2PWRFAIL,
	PCF50633_IRQ_LEDPWRFAIL,
	PCF50633_IRQ_LEDOVP,
	PCF50633_IRQ_LDO1PWRFAIL,
	PCF50633_IRQ_LDO2PWRFAIL,
	PCF50633_IRQ_LDO3PWRFAIL,
	PCF50633_IRQ_LDO4PWRFAIL,
	PCF50633_IRQ_LDO5PWRFAIL,
	PCF50633_IRQ_LDO6PWRFAIL,
	PCF50633_IRQ_HCLDOPWRFAIL,
	PCF50633_IRQ_HCLDOOVL,

	/** @brief Total number of interrupts. Must always be the last entry. */
	PCF50633_NUM_IRQ,
};

/**
 * @struct pcf50633
 * @brief The core device structure for the PCF50633 driver.
 * @description This structure holds all the state associated with a single
 * PCF50633 device instance, including device pointers, interrupt handling
 * infrastructure, and pointers to its sub-devices (RTC, ADC, etc.).
 */
struct pcf50633 {
	struct device *dev;
	struct regmap *regmap;

	struct pcf50633_platform_data *pdata;
	int irq; /* The physical IRQ line connected to the host processor */
	struct pcf50633_irq irq_handler[PCF50633_NUM_IRQ];
	struct work_struct irq_work;
	struct workqueue_struct *work_queue;
	struct mutex lock;

	u8 mask_regs[5]; /* Cached copy of interrupt mask registers */

	u8 suspend_irq_masks[5];
	u8 resume_reason[5];
	int is_suspended;

	int onkey1s_held;

	/* Pointers to platform devices for sub-functions */
	struct platform_device *rtc_pdev;
	struct platform_device *mbc_pdev;
	struct platform_device *adc_pdev;
	struct platform_device *input_pdev;
	struct platform_device *bl_pdev;
	struct platform_device *regulator_pdev[PCF50633_NUM_REGULATORS];
};

/* Bit definitions for Interrupt Register 1 (PCF50633_REG_INT1) */
enum pcf50633_reg_int1 {
	PCF50633_INT1_ADPINS	= 0x01,	/* Adapter inserted */
	PCF50633_INT1_ADPREM	= 0x02,	/* Adapter removed */
	PCF50633_INT1_USBINS	= 0x04,	/* USB inserted */
	PCF50633_INT1_USBREM	= 0x08,	/* USB removed */
	/* reserved */
	PCF50633_INT1_ALARM	= 0x40, /* RTC alarm time is reached */
	PCF50633_INT1_SECOND	= 0x80,	/* RTC periodic second interrupt */
};

/* Bit definitions for Interrupt Register 2 (PCF50633_REG_INT2) */
enum pcf50633_reg_int2 {
	PCF50633_INT2_ONKEYR	= 0x01, /* ONKEY rising edge */
	PCF50633_INT2_ONKEYF	= 0x02, /* ONKEY falling edge */
	PCF50633_INT2_EXTON1R	= 0x04, /* EXTON1 rising edge */
	PCF50633_INT2_EXTON1F	= 0x08, /* EXTON1 falling edge */
	PCF50633_INT2_EXTON2R	= 0x10, /* EXTON2 rising edge */
	PCF50633_INT2_EXTON2F	= 0x20, /* EXTON2 falling edge */
	PCF50633_INT2_EXTON3R	= 0x40, /* EXTON3 rising edge */
	PCF50633_INT2_EXTON3F	= 0x80, /* EXTON3 falling edge */
};

/* Bit definitions for Interrupt Register 3 (PCF50633_REG_INT3) */
enum pcf50633_reg_int3 {
	PCF50633_INT3_BATFULL	= 0x01, /* Battery full */
	PCF50633_INT3_CHGHALT	= 0x02,	/* Charger halt */
	PCF50633_INT3_THLIMON	= 0x04, /* Thermal limit on */
	PCF50633_INT3_THLIMOFF	= 0x08, /* Thermal limit off */
	PCF50633_INT3_USBLIMON	= 0x10, /* USB current limit on */
	PCF50633_INT3_USBLIMOFF	= 0x20, /* USB current limit off */
	PCF50633_INT3_ADCRDY	= 0x40, /* ADC result ready */
	PCF50633_INT3_ONKEY1S	= 0x80,	/* ONKEY pressed for 1 second */
};

/* Bit definitions for Interrupt Register 4 (PCF50633_REG_INT4) */
enum pcf50633_reg_int4 {
	PCF50633_INT4_LOWSYS		= 0x01, /* System voltage is low */
	PCF50633_INT4_LOWBAT		= 0x02, /* Battery voltage is low */
	PCF50633_INT4_HIGHTMP		= 0x04, /* High temperature detected */
	PCF50633_INT4_AUTOPWRFAIL	= 0x08, /* Auto power-down regulator fail */
	PCF50633_INT4_DWN1PWRFAIL	= 0x10, /* Down regulator 1 fail */
	PCF50633_INT4_DWN2PWRFAIL	= 0x20, /* Down regulator 2 fail */
	PCF50633_INT4_LEDPWRFAIL	= 0x40, /* LED driver fail */
	PCF50633_INT4_LEDOVP		= 0x80, /* LED over-voltage protection */
};

/* Bit definitions for Interrupt Register 5 (PCF50633_REG_INT5) */
enum pcf50633_reg_int5 {
	PCF50633_INT5_LDO1PWRFAIL	= 0x01, /* LDO1 power fail */
	PCF50633_INT5_LDO2PWRFAIL	= 0x02, /* LDO2 power fail */
	PCF50633_INT5_LDO3PWRFAIL	= 0x04, /* LDO3 power fail */
	PCF50633_INT5_LDO4PWRFAIL	= 0x08, /* LDO4 power fail */
	PCF50633_INT5_LDO5PWRFAIL	= 0x10, /* LDO5 power fail */
	PCF50633_INT5_LDO6PWRFAIL	= 0x20, /* LDO6 power fail */
	PCF50633_INT5_HCLDOPWRFAIL	= 0x40, /* HC-LDO power fail */
	PCF50633_INT5_HCLDOOVL		= 0x80, /* HC-LDO overload */
};

/* misc. registers */
#define PCF50633_REG_OOCSHDWN	0x0c /* Out-of-charge shutdown register */

/* LED registers */
#define PCF50633_REG_LEDOUT 0x28
#define PCF50633_REG_LEDENA 0x29
#define PCF50633_REG_LEDCTL 0x2a
#define PCF50633_REG_LEDDIM 0x2b

/**
 * @brief Helper function to retrieve the pcf50633 main struct from a device struct.
 * @param dev Pointer to the device struct.
 * @return Pointer to the containing pcf50633 struct.
 */
static inline struct pcf50633 *dev_to_pcf50633(struct device *dev)
{
	return dev_get_drvdata(dev);
}

/* Core driver lifecycle functions */
int pcf50633_irq_init(struct pcf50633 *pcf, int irq);
void pcf50633_irq_free(struct pcf50633 *pcf);
extern const struct dev_pm_ops pcf50633_pm;

#endif
