/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * core.h  -- Core driver for NXP PCF50633
 *
 * (C) 2006-2008 by Openmoko, Inc.
 * All rights reserved.
 */

#ifndef __LINUX_MFD_PCF50633_CORE_H
#define __LINUX_MFD_PCF50633_CORE_H

#include <linux/i2c.h>
#include <linux/workqueue.h>
#include <linux/regulator/driver.h>
#include <linux/regulator/machine.h>
#include <linux/pm.h>
#include <linux/power_supply.h>

/**
 * @e10293e5-8b0b-4355-8f63-9abc9244e5b2/include/linux/mfd/pcf50633/core.h
 * @brief Core interface definitions for the NXP PCF50633 Power Management Integrated Circuit (PMIC).
 * * Domain: Kernel Drivers, Power Management, Multi-Function Devices (MFD).
 * * Functional Utility: Provides the architectural schema and API for managing the PCF50633 chip, 
 *   including regulator control, battery charging, interrupt handling, and register access.
 */

struct pcf50633;
struct regmap;

// Domain: Hardware Constraints - The chip supports exactly 11 independent voltage regulators.
#define PCF50633_NUM_REGULATORS	11

/**
 * @struct pcf50633_platform_data
 * @brief Configuration data provided by the board support package (BSP).
 */
struct pcf50633_platform_data {
	struct regulator_init_data reg_init_data[PCF50633_NUM_REGULATORS];

	char **batteries;
	int num_batteries;

	/*
	 * Logic: Calibration constant for the charging circuit.
	 * Target: Sets the baseline current for the MBC (Main Battery Charger) subsystem.
	 */
	int charger_reference_current_ma;

	/* Callbacks: Lifecycle and event notifications for subordinate drivers. */
	void (*probe_done)(struct pcf50633 *);
	void (*mbc_event_callback)(struct pcf50633 *, int);
	void (*regulator_registered)(struct pcf50633 *, int);
	void (*force_shutdown)(struct pcf50633 *);

	u8 resumers[5]; // Intent: Defines which interrupt sources trigger a system wakeup.
};

/**
 * @struct pcf50633_irq
 * @brief Internal registry for chip-level interrupt handlers.
 */
struct pcf50633_irq {
	void (*handler) (int, void *);
	void *data;
};

/* 
 * Functional Utility: Low-level interrupt management API.
 * Facilitates the dispatching of sub-module events (e.g. Battery Full, USB Inserted) 
 * from the main chip interrupt line.
 */
int pcf50633_register_irq(struct pcf50633 *pcf, int irq,
			void (*handler) (int, void *), void *data);
int pcf50633_free_irq(struct pcf50633 *pcf, int irq);

int pcf50633_irq_mask(struct pcf50633 *pcf, int irq);
int pcf50633_irq_unmask(struct pcf50633 *pcf, int irq);
int pcf50633_irq_mask_get(struct pcf50633 *pcf, int irq);

/* 
 * Block Logic: Register I/O primitives.
 * Implements standard I2C block and single-byte operations with error propagation.
 */
int pcf50633_read_block(struct pcf50633 *, u8 reg,
					int nr_regs, u8 *data);
int pcf50633_write_block(struct pcf50633 *pcf, u8 reg,
					int nr_regs, u8 *data);
u8 pcf50633_reg_read(struct pcf50633 *, u8 reg);
int pcf50633_reg_write(struct pcf50633 *pcf, u8 reg, u8 val);

int pcf50633_reg_set_bit_mask(struct pcf50633 *pcf, u8 reg, u8 mask, u8 val);
int pcf50633_reg_clear_bits(struct pcf50633 *pcf, u8 reg, u8 bits);

/* Interrupt Register Map: Offsets for status and mask registers. */
#define PCF50633_REG_INT1	0x02
#define PCF50633_REG_INT2	0x03
#define PCF50633_REG_INT3	0x04
#define PCF50633_REG_INT4	0x05
#define PCF50633_REG_INT5	0x06

#define PCF50633_REG_INT1M	0x07
#define PCF50633_REG_INT2M	0x08
#define PCF50633_REG_INT3M	0x09
#define PCF50633_REG_INT4M	0x0a
#define PCF50633_REG_INT5M	0x0b

/**
 * @enum pcf50633_irq_index
 * @brief Logical indices for all hardware-level interrupts supported by the PMIC.
 */
enum {
	/* Power Supply Events */
	PCF50633_IRQ_ADPINS, // Logic: Wall adapter connected.
	PCF50633_IRQ_ADPREM,
	PCF50633_IRQ_USBINS, // Logic: USB power source connected.
	PCF50633_IRQ_USBREM,
	PCF50633_IRQ_RESERVED1,
	PCF50633_IRQ_RESERVED2,
	/* Timekeeping */
	PCF50633_IRQ_ALARM,
	PCF50633_IRQ_SECOND,
	/* User Input */
	PCF50633_IRQ_ONKEYR, // Logic: Power button rising edge.
	PCF50633_IRQ_ONKEYF, // Logic: Power button falling edge.
	/* External Triggering */
	PCF50633_IRQ_EXTON1R,
	PCF50633_IRQ_EXTON1F,
	PCF50633_IRQ_EXTON2R,
	PCF50633_IRQ_EXTON2F,
	PCF50633_IRQ_EXTON3R,
	PCF50633_IRQ_EXTON3F,
	/* Charging & Thermal */
	PCF50633_IRQ_BATFULL,
	PCF50633_IRQ_CHGHALT,
	PCF50633_IRQ_THLIMON,
	PCF50633_IRQ_THLIMOFF,
	PCF50633_IRQ_USBLIMON,
	PCF50633_IRQ_USBLIMOFF,
	PCF50633_IRQ_ADCRDY,
	PCF50633_IRQ_ONKEY1S,
	/* System Safety */
	PCF50633_IRQ_LOWSYS,
	PCF50633_IRQ_LOWBAT,
	PCF50633_IRQ_HIGHTMP,
	PCF50633_IRQ_AUTOPWRFAIL,
	PCF50633_IRQ_DWN1PWRFAIL,
	PCF50633_IRQ_DWN2PWRFAIL,
	PCF50633_IRQ_LEDPWRFAIL,
	PCF50633_IRQ_LEDOVP,
	/* LDO Faults */
	PCF50633_IRQ_LDO1PWRFAIL,
	PCF50633_IRQ_LDO2PWRFAIL,
	PCF50633_IRQ_LDO3PWRFAIL,
	PCF50633_IRQ_LDO4PWRFAIL,
	PCF50633_IRQ_LDO5PWRFAIL,
	PCF50633_IRQ_LDO6PWRFAIL,
	PCF50633_IRQ_HCLDOPWRFAIL,
	PCF50633_IRQ_HCLDOOVL,

	/* Sentinel for array sizing */
	PCF50633_NUM_IRQ,
};

/**
 * @struct pcf50633
 * @brief Master handle for the PCF50633 device instance.
 * * Algorithm: Aggregates sub-device handles and coordinates shared state (mutex, IRQ workqueue).
 */
struct pcf50633 {
	struct device *dev;
	struct regmap *regmap; // Intent: Abstraction layer for register access.

	struct pcf50633_platform_data *pdata;
	int irq; // Domain: Host processor IRQ line connected to PMIC.
	struct pcf50633_irq irq_handler[PCF50633_NUM_IRQ];
	struct work_struct irq_work;
	struct workqueue_struct *work_queue; // Intent: Bottom-half processing for PMIC interrupts.
	struct mutex lock;

	u8 mask_regs[5];

	u8 suspend_irq_masks[5];
	u8 resume_reason[5]; // Domain: Diagnostic state for post-suspend analysis.
	int is_suspended;

	int onkey1s_held;

	/* Logical sub-devices managed by this core driver. */
	struct platform_device *rtc_pdev;
	struct platform_device *mbc_pdev;
	struct platform_device *adc_pdev;
	struct platform_device *input_pdev;
	struct platform_device *bl_pdev;
	struct platform_device *regulator_pdev[PCF50633_NUM_REGULATORS];
};

/* Bitmask definitions for interrupt status registers. */

enum pcf50633_reg_int1 {
	PCF50633_INT1_ADPINS	= 0x01,	/* Adapter inserted */
	PCF50633_INT1_ADPREM	= 0x02,	/* Adapter removed */
	PCF50633_INT1_USBINS	= 0x04,	/* USB inserted */
	PCF50633_INT1_USBREM	= 0x08,	/* USB removed */
	/* reserved */
	PCF50633_INT1_ALARM	= 0x40, /* RTC alarm time is reached */
	PCF50633_INT1_SECOND	= 0x80,	/* RTC periodic second interrupt */
};

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

enum pcf50633_reg_int3 {
	PCF50633_INT3_BATFULL	= 0x01, /* Battery full */
	PCF50633_INT3_CHGHALT	= 0x02,	/* Charger halt */
	PCF50633_INT3_THLIMON	= 0x04,
	PCF50633_INT3_THLIMOFF	= 0x08,
	PCF50633_INT3_USBLIMON	= 0x10,
	PCF50633_INT3_USBLIMOFF	= 0x20,
	PCF50633_INT3_ADCRDY	= 0x40, /* ADC result ready */
	PCF50633_INT3_ONKEY1S	= 0x80,	/* ONKEY pressed 1 second */
};

enum pcf50633_reg_int4 {
	PCF50633_INT4_LOWSYS		= 0x01,
	PCF50633_INT4_LOWBAT		= 0x02,
	PCF50633_INT4_HIGHTMP		= 0x04,
	PCF50633_INT4_AUTOPWRFAIL	= 0x08,
	PCF50633_INT4_DWN1PWRFAIL	= 0x10,
	PCF50633_INT4_DWN2PWRFAIL	= 0x20,
	PCF50633_INT4_LEDPWRFAIL	= 0x40,
	PCF50633_INT4_LEDOVP		= 0x80,
};

enum pcf50633_reg_int5 {
	PCF50633_INT5_LDO1PWRFAIL	= 0x01,
	PCF50633_INT5_LDO2PWRFAIL	= 0x02,
	PCF50633_INT5_LDO3PWRFAIL	= 0x04,
	PCF50633_INT5_LDO4PWRFAIL	= 0x08,
	PCF50633_INT5_LDO5PWRFAIL	= 0x10,
	PCF50633_INT5_LDO6PWRFAIL	= 0x20,
	PCF50633_INT5_HCLDOPWRFAIL	= 0x40,
	PCF50633_INT5_HCLDOOVL		= 0x80,
};

/* MISC Registers */
#define PCF50633_REG_OOCSHDWN	0x0c

/* LED registers */
#define PCF50633_REG_LEDOUT 0x28
#define PCF50633_REG_LEDENA 0x29
#define PCF50633_REG_LEDCTL 0x2a
#define PCF50633_REG_LEDDIM 0x2b

/**
 * @brief Container-of style macro for converting a generic device pointer to PCF50633 structure.
 */
static inline struct pcf50633 *dev_to_pcf50633(struct device *dev)
{
	return dev_get_drvdata(dev);
}

int pcf50633_irq_init(struct pcf50633 *pcf, int irq);
void pcf50633_irq_free(struct pcf50633 *pcf);
extern const struct dev_pm_ops pcf50633_pm;

#endif
