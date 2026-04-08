// SPDX-License-Identifier: GPL-2.0
/**
 * @file nvec_power.c
 * @brief Power supply and battery driver for NVIDIA Embedded Controller (NVEC).
 * @details This driver acts as a bridge between the low-level NVIDIA Embedded
 * Controller (NVEC) interface and the standard Linux power supply subsystem. It
 * registers two power supply devices, 'ac' and 'battery', and reports their
 * status to the kernel. Data is received from the EC asynchronously via notifiers,
 * and a polling mechanism is used to periodically request updated status information.
 *
 * Copyright (C) 2011 The AC100 Kernel Team <ac100@lists.launchpad.net>
 *
 * Authors:  Ilya Petrov <ilya.muromec@gmail.com>
 *           Marc Dietrich <marvin24@gmx.de>
 */

#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/err.h>
#include <linux/power_supply.h>
#include <linux/slab.h>
#include <linux/workqueue.h>
#include <linux/delay.h>

#include "nvec.h"

#define GET_SYSTEM_STATUS 0x00

/**
 * @struct nvec_power
 * @brief Holds the state of the power supply and battery devices.
 * @details This structure contains cached data received from the NVEC. The
 * `power_supply` callbacks read from this struct to report status to userspace.
 */
struct nvec_power {
	struct notifier_block notifier;      /**< For receiving async events from NVEC */
	struct delayed_work poller;          /**< For periodically polling the EC */
	struct nvec_chip *nvec;              /**< Pointer to the underlying NVEC device */

	/* AC adapter state */
	int on;                              /**< AC adapter online status */

	/* Battery state */
	int bat_present;                     /**< Whether the battery is present */
	int bat_status;                      /**< e.g., CHARGING, DISCHARGING */
	int bat_voltage_now;                 /**< Instantaneous voltage in microvolts */
	int bat_current_now;                 /**< Instantaneous current in microamps */
	int bat_current_avg;                 /**< Average current in microamps */
	int time_remain;                     /**< Remaining time until empty in seconds */
	int charge_full_design;              /**< Designed full charge capacity in microamp-hours */
	int charge_last_full;                /**< Last measured full charge capacity */
	int critical_capacity;               /**< Critical low capacity level */
	int capacity_remain;                 /**< Remaining capacity in microamp-hours */
	int bat_temperature;                 /**< Battery temperature in tenths of a degree Celsius */
	int bat_cap;                         /**< Remaining capacity as a percentage (0-100) */
	int bat_type_enum;                   /**< Battery technology (e.g., LION) */
	char bat_manu[30];                   /**< Manufacturer name string */
	char bat_model[30];                  /**< Model name string */
	char bat_type[30];                   /**< Type name string */
};

/* EC command subtypes for requesting battery information */
enum {
	SLOT_STATUS,
	VOLTAGE,
	TIME_REMAINING,
	CURRENT,
	AVERAGE_CURRENT,
	AVERAGING_TIME_INTERVAL,
	CAPACITY_REMAINING,
	LAST_FULL_CHARGE_CAPACITY,
	DESIGN_CAPACITY,
	CRITICAL_CAPACITY,
	TEMPERATURE,
	MANUFACTURER,
	MODEL,
	TYPE,
};

/* Device IDs for AC adapter and Battery */
enum {
	AC,
	BAT,
};

/**
 * @struct bat_response
 * @brief Represents the data structure of a response from the NVEC.
 */
struct bat_response {
	u8 event_type;
	u8 length;
	u8 sub_type;
	u8 status;
	/* payload */
	union {
		char plc[30];
		u16 plu;
		s16 pls;
	};
};

static struct power_supply *nvec_bat_psy;
static struct power_supply *nvec_psy;

/**
 * @brief Notifier callback for system-level power events (e.g., AC adapter status).
 * @return NOTIFY_STOP if the event is handled, NOTIFY_DONE otherwise.
 */
static int nvec_power_notifier(struct notifier_block *nb,
			       unsigned long event_type, void *data)
{
	struct nvec_power *power =
	    container_of(nb, struct nvec_power, notifier);
	struct bat_response *res = data;

	if (event_type != NVEC_SYS)
		return NOTIFY_DONE;

	// sub_type 0 corresponds to AC adapter online status
	if (res->sub_type == 0) {
		if (power->on != res->plu) {
			power->on = res->plu;
			power_supply_changed(nvec_psy);
		}
		return NOTIFY_STOP; // Event handled
	}
	return NOTIFY_OK;
}

// List of one-time battery information to fetch upon battery insertion.
static const int bat_init[] = {
	LAST_FULL_CHARGE_CAPACITY, DESIGN_CAPACITY, CRITICAL_CAPACITY,
	MANUFACTURER, MODEL, TYPE,
};

/**
 * @brief Sends requests to the NVEC to fetch static battery manufacturing data.
 */
static void get_bat_mfg_data(struct nvec_power *power)
{
	int i;
	char buf[] = { NVEC_BAT, SLOT_STATUS };

	for (i = 0; i < ARRAY_SIZE(bat_init); i++) {
		buf[1] = bat_init[i];
		nvec_write_async(power->nvec, buf, 2);
	}
}

/**
 * @brief Notifier callback for battery-specific events.
 * @details This function is called when the NVEC sends an asynchronous message
 * about the battery. It parses the message and updates the corresponding fields
 * in the `nvec_power` struct.
 * @return NOTIFY_STOP if the event is handled.
 */
static int nvec_power_bat_notifier(struct notifier_block *nb,
				   unsigned long event_type, void *data)
{
	struct nvec_power *power =
	    container_of(nb, struct nvec_power, notifier);
	struct bat_response *res = data;
	int status_changed = 0;

	if (event_type != NVEC_BAT)
		return NOTIFY_DONE;

	switch (res->sub_type) {
	case SLOT_STATUS:
		if (res->plc[0] & 1) { // Check if battery is present
			if (power->bat_present == 0) {
				status_changed = 1;
				// If battery was just inserted, fetch static data like manufacturer.
				get_bat_mfg_data(power);
			}

			power->bat_present = 1;

			// Decode battery status (Charging, Discharging, etc.)
			switch ((res->plc[0] >> 1) & 3) {
			case 0:
				power->bat_status =
				    POWER_SUPPLY_STATUS_NOT_CHARGING;
				break;
			case 1:
				power->bat_status =
				    POWER_SUPPLY_STATUS_CHARGING;
				break;
			case 2:
				power->bat_status =
				    POWER_SUPPLY_STATUS_DISCHARGING;
				break;
			default:
				power->bat_status = POWER_SUPPLY_STATUS_UNKNOWN;
			}
		} else { // Battery is not present
			if (power->bat_present == 1)
				status_changed = 1;

			power->bat_present = 0;
			power->bat_status = POWER_SUPPLY_STATUS_UNKNOWN;
		}
		power->bat_cap = res->plc[1]; // Update battery capacity percentage
		if (status_changed)
			power_supply_changed(nvec_bat_psy);
		break;
	case VOLTAGE:
		power->bat_voltage_now = res->plu * 1000;
		break;
	case TIME_REMAINING:
		power->time_remain = res->plu * 3600;
		break;
	case CURRENT:
		power->bat_current_now = res->pls * 1000;
		break;
	case AVERAGE_CURRENT:
		power->bat_current_avg = res->pls * 1000;
		break;
	case CAPACITY_REMAINING:
		power->capacity_remain = res->plu * 1000;
		break;
	case LAST_FULL_CHARGE_CAPACITY:
		power->charge_last_full = res->plu * 1000;
		break;
	case DESIGN_CAPACITY:
		power->charge_full_design = res->plu * 1000;
		break;
	case CRITICAL_CAPACITY:
		power->critical_capacity = res->plu * 1000;
		break;
	case TEMPERATURE:
		// Convert from tenths of Kelvin to tenths of Celsius
		power->bat_temperature = res->plu - 2732;
		break;
	case MANUFACTURER:
		memcpy(power->bat_manu, &res->plc, res->length - 2);
		power->bat_manu[res->length - 2] = '\0';
		break;
	case MODEL:
		memcpy(power->bat_model, &res->plc, res->length - 2);
		power->bat_model[res->length - 2] = '\0';
		break;
	case TYPE:
		memcpy(power->bat_type, &res->plc, res->length - 2);
		power->bat_type[res->length - 2] = '\0';
		// Attempt to parse technology from the type string
		if (!strncmp(power->bat_type, "Li", 30))
			power->bat_type_enum = POWER_SUPPLY_TECHNOLOGY_LION;
		else
			power->bat_type_enum = POWER_SUPPLY_TECHNOLOGY_UNKNOWN;
		break;
	default:
		return NOTIFY_STOP;
	}

	return NOTIFY_STOP;
}

/**
 * @brief Callback for the power_supply framework to get AC adapter properties.
 * @details This function is called by the kernel when userspace requests
 * information about the AC adapter. It reads the cached value from the driver's state.
 */
static int nvec_power_get_property(struct power_supply *psy,
				   enum power_supply_property psp,
				   union power_supply_propval *val)
{
	struct nvec_power *power = dev_get_drvdata(psy->dev.parent);

	switch (psp) {
	case POWER_SUPPLY_PROP_ONLINE:
		val->intval = power->on;
		break;
	default:
		return -EINVAL;
	}
	return 0;
}

/**
 * @brief Callback for the power_supply framework to get battery properties.
 * @details This function is called by the kernel when userspace requests
 * information about the battery. It reads the cached values from the driver's state.
 */
static int nvec_battery_get_property(struct power_supply *psy,
				     enum power_supply_property psp,
				     union power_supply_propval *val)
{
	struct nvec_power *power = dev_get_drvdata(psy->dev.parent);

	switch (psp) {
	case POWER_SUPPLY_PROP_STATUS:
		val->intval = power->bat_status;
		break;
	case POWER_SUPPLY_PROP_CAPACITY:
		val->intval = power->bat_cap;
		break;
	case POWER_SUPPLY_PROP_PRESENT:
		val->intval = power->bat_present;
		break;
	case POWER_SUPPLY_PROP_VOLTAGE_NOW:
		val->intval = power->bat_voltage_now;
		break;
	case POWER_SUPPLY_PROP_CURRENT_NOW:
		val->intval = power->bat_current_now;
		break;
	case POWER_SUPPLY_PROP_CURRENT_AVG:
		val->intval = power->bat_current_avg;
		break;
	case POWER_SUPPLY_PROP_TIME_TO_EMPTY_NOW:
		val->intval = power->time_remain;
		break;
	case POWER_SUPPLY_PROP_CHARGE_FULL_DESIGN:
		val->intval = power->charge_full_design;
		break;
	case POWER_SUPPLY_PROP_CHARGE_FULL:
		val->intval = power->charge_last_full;
		break;
	case POWER_SUPPLY_PROP_CHARGE_EMPTY:
		val->intval = power->critical_capacity;
		break;
	case POWER_SUPPLY_PROP_CHARGE_NOW:
		val->intval = power->capacity_remain;
		break;
	case POWER_SUPPLY_PROP_TEMP:
		val->intval = power->bat_temperature;
		break;
	case POWER_SUPPLY_PROP_MANUFACTURER:
		val->strval = power->bat_manu;
		break;
	case POWER_SUPPLY_PROP_MODEL_NAME:
		val->strval = power->bat_model;
		break;
	case POWER_SUPPLY_PROP_TECHNOLOGY:
		val->intval = power->bat_type_enum;
		break;
	default:
		return -EINVAL;
	}
	return 0;
}

// Properties exposed by the AC adapter power supply device.
static enum power_supply_property nvec_power_props[] = {
	POWER_SUPPLY_PROP_ONLINE,
};

// Properties exposed by the battery power supply device.
static enum power_supply_property nvec_battery_props[] = {
	POWER_SUPPLY_PROP_STATUS,
	POWER_SUPPLY_PROP_PRESENT,
	POWER_SUPPLY_PROP_CAPACITY,
	POWER_SUPPLY_PROP_VOLTAGE_NOW,
	POWER_SUPPLY_PROP_CURRENT_NOW,
#ifdef EC_FULL_DIAG
	POWER_SUPPLY_PROP_CURRENT_AVG,
	POWER_SUPPLY_PROP_TEMP,
	POWER_SUPPLY_PROP_TIME_TO_EMPTY_NOW,
#endif
	POWER_SUPPLY_PROP_CHARGE_FULL_DESIGN,
	POWER_SUPPLY_PROP_CHARGE_FULL,
	POWER_SUPPLY_PROP_CHARGE_EMPTY,
	POWER_SUPPLY_PROP_CHARGE_NOW,
	POWER_SUPPLY_PROP_MANUFACTURER,
	POWER_SUPPLY_PROP_MODEL_NAME,
	POWER_SUPPLY_PROP_TECHNOLOGY,
};

static char *nvec_power_supplied_to[] = {
	"battery",
};

// Descriptor for the battery power supply.
static const struct power_supply_desc nvec_bat_psy_desc = {
	.name = "battery",
	.type = POWER_SUPPLY_TYPE_BATTERY,
	.properties = nvec_battery_props,
	.num_properties = ARRAY_SIZE(nvec_battery_props),
	.get_property = nvec_battery_get_property,
};

// Descriptor for the AC adapter power supply.
static const struct power_supply_desc nvec_psy_desc = {
	.name = "ac",
	.type = POWER_SUPPLY_TYPE_MAINS,
	.properties = nvec_power_props,
	.num_properties = ARRAY_SIZE(nvec_power_props),
	.get_property = nvec_power_get_property,
};

static int counter;
// List of battery properties to poll in a round-robin fashion.
static const int bat_iter[] = {
	SLOT_STATUS, VOLTAGE, CURRENT, CAPACITY_REMAINING,
#ifdef EC_FULL_DIAG
	AVERAGE_CURRENT, TEMPERATURE, TIME_REMAINING,
#endif
};

/**
 * @brief Work function for periodically polling power status.
 * @details This function is scheduled to run periodically (every 5 seconds). It sends
 * requests to the NVEC to get the latest AC status and one of the battery
 * properties in a round-robin fashion.
 */
static void nvec_power_poll(struct work_struct *work)
{
	char buf[] = { NVEC_SYS, GET_SYSTEM_STATUS };
	struct nvec_power *power = container_of(work, struct nvec_power,
						poller.work);

	if (counter >= ARRAY_SIZE(bat_iter))
		counter = 0;

	/* AC status via sys req */
	nvec_write_async(power->nvec, buf, 2);
	msleep(100);

	/*
	 * Select a battery request function via round robin doing it all at
	 * once seems to overload the power supply.
	 */
	buf[0] = NVEC_BAT;
	buf[1] = bat_iter[counter++];
	nvec_write_async(power->nvec, buf, 2);

	schedule_delayed_work(to_delayed_work(work), msecs_to_jiffies(5000));
};

/**
 * @brief Probe function called when a matching device is found.
 * @details This function initializes the driver instance, allocates memory, registers
 * notifiers for asynchronous events, sets up the periodic polling work, and
 * registers the device with the Linux power supply subsystem.
 * @param pdev The platform device being probed.
 * @return 0 on success, or a negative error code on failure.
 */
static int nvec_power_probe(struct platform_device *pdev)
{
	struct power_supply **psy;
	const struct power_supply_desc *psy_desc;
	struct nvec_power *power;
	struct nvec_chip *nvec = dev_get_drvdata(pdev->dev.parent);
	struct power_supply_config psy_cfg = {};

	power = devm_kzalloc(&pdev->dev, sizeof(struct nvec_power), GFP_NOWAIT);
	if (!power)
		return -ENOMEM;

	dev_set_drvdata(&pdev->dev, power);
	power->nvec = nvec;

	// Block Logic: Differentiate initialization based on whether this is the AC or BAT device.
	switch (pdev->id) {
	case AC:
		psy = &nvec_psy;
		psy_desc = &nvec_psy_desc;
		psy_cfg.supplied_to = nvec_power_supplied_to;
		psy_cfg.num_supplicants = ARRAY_SIZE(nvec_power_supplied_to);

		power->notifier.notifier_call = nvec_power_notifier;

		// Initialize and start the periodic polling work.
		INIT_DELAYED_WORK(&power->poller, nvec_power_poll);
		schedule_delayed_work(&power->poller, msecs_to_jiffies(5000));
		break;
	case BAT:
		psy = &nvec_bat_psy;
		psy_desc = &nvec_bat_psy_desc;

		power->notifier.notifier_call = nvec_power_bat_notifier;
		break;
	default:
		return -ENODEV;
	}

	nvec_register_notifier(nvec, &power->notifier, NVEC_SYS | NVEC_BAT);

	if (pdev->id == BAT)
		get_bat_mfg_data(power);

	*psy = power_supply_register(&pdev->dev, psy_desc, &psy_cfg);

	return PTR_ERR_OR_ZERO(*psy);
}

/**
 * @brief Remove function called when the device is removed or driver is unloaded.
 * @details This function cleans up all resources allocated by the probe function,
 * including unregistering the power supply devices, canceling polling work, and
 * unregistering the notifier.
 */
static void nvec_power_remove(struct platform_device *pdev)
{
	struct nvec_power *power = platform_get_drvdata(pdev);

	cancel_delayed_work_sync(&power->poller);
	nvec_unregister_notifier(power->nvec, &power->notifier);
	switch (pdev->id) {
	case AC:
		power_supply_unregister(nvec_psy);
		break;
	case BAT:
		power_supply_unregister(nvec_bat_psy);
	}
}

static struct platform_driver nvec_power_driver = {
	.probe = nvec_power_probe,
	.remove = nvec_power_remove,
	.driver = {
		   .name = "nvec-power",
	}
};

module_platform_driver(nvec_power_driver);

MODULE_AUTHOR("Ilya Petrov <ilya.muromec@gmail.com>");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("NVEC battery and AC driver");
MODULE_ALIAS("platform:nvec-power");
