/**
 * @file nvec_power.c
 * @brief Power supply driver for NVIDIA embedded controllers (NVEC),
 *        managing AC power and battery status.
 *
 * This driver interfaces with the NVEC to provide power management capabilities
 * to the Linux kernel. It registers two power supply devices: one for AC power
 * and another for battery. It uses a notifier mechanism to receive events from
 * the NVEC and a polling mechanism to periodically update battery status.
 *
 * Functional Utility: Enables the operating system to monitor and react to
 *                     power-related events (e.g., AC connected/disconnected,
 *                     battery charge level, battery status) on devices equipped
 *                     with an NVIDIA embedded controller.
 */

// SPDX-License-Identifier: GPL-2.0
/*
 * nvec_power: power supply driver for a NVIDIA compliant embedded controller
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

/** @brief NVEC command to get system status. */
#define GET_SYSTEM_STATUS 0x00

/**
 * @struct nvec_power
 * @brief Represents the state and associated resources for an NVEC power supply device.
 *
 * This structure holds all relevant information for both AC and battery power
 * supplies managed by the NVEC driver. It includes power status, battery
 * parameters, notifier block for events, and a delayed work for polling.
 */
struct nvec_power {
	/** @brief Notifier block for receiving events from the NVEC. */
	struct notifier_block notifier;
	/** @brief Delayed work structure for polling power supply status. */
	struct delayed_work poller;
	/** @brief Pointer to the NVEC chip instance. */
	struct nvec_chip *nvec;
	/** @brief Current AC online status (1 if on, 0 if off). */
	int on;
	/** @brief Battery present status (1 if present, 0 if not). */
	int bat_present;
	/** @brief Current battery charge status (e.g., charging, discharging). */
	int bat_status;
	/** @brief Current battery voltage in microvolts. */
	int bat_voltage_now;
	/** @brief Current battery current flow in microamperes. */
	int bat_current_now;
	/** @brief Average battery current flow in microamperes. */
	int bat_current_avg;
	/** @brief Estimated time remaining until empty in seconds. */
	int time_remain;
	/** @brief Designed full charge capacity of the battery in microamp-hours. */
	int charge_full_design;
	/** @brief Last full charge capacity of the battery in microamp-hours. */
	int charge_last_full;
	/** @brief Critical capacity level of the battery in microamp-hours. */
	int critical_capacity;
	/** @brief Remaining charge capacity of the battery in microamp-hours. */
	int capacity_remain;
	/** @brief Battery temperature in tenths of degrees Celsius. */
	int bat_temperature;
	/** @brief Battery percentage capacity. */
	int bat_cap;
	/** @brief Enumerated battery technology type. */
	int bat_type_enum;
	/** @brief Battery manufacturer string. */
	char bat_manu[30];
	/** @brief Battery model string. */
	char bat_model[30];
	/** @brief Battery type string. */
	char bat_type[30];
};

/**
 * @enum BatteryPropertySubType
 * @brief Sub-types for battery-related commands/properties exchanged with the NVEC.
 */
enum {
	SLOT_STATUS, /**< Battery slot status. */
	VOLTAGE, /**< Battery voltage. */
	TIME_REMAINING, /**< Battery time remaining. */
	CURRENT, /**< Battery current. */
	AVERAGE_CURRENT, /**< Battery average current. */
	AVERAGING_TIME_INTERVAL, /**< Battery averaging time interval. */
	CAPACITY_REMAINING, /**< Battery remaining capacity. */
	LAST_FULL_CHARGE_CAPACITY, /**< Battery last full charge capacity. */
	DESIGN_CAPACITY, /**< Battery design capacity. */
	CRITICAL_CAPACITY, /**< Battery critical capacity. */
	TEMPERATURE, /**< Battery temperature. */
	MANUFACTURER, /**< Battery manufacturer information. */
	MODEL, /**< Battery model information. */
	TYPE, /**< Battery type information. */
};

/**
 * @enum PowerSupplyType
 * @brief Identifiers for different power supply types handled by the driver.
 */
enum {
	AC,  /**< AC power supply. */
	BAT, /**< Battery power supply. */
};

/**
 * @struct bat_response
 * @brief Structure to parse responses from the NVEC regarding battery information.
 *
 * This structure is used to interpret the raw byte data received from the NVEC
 * into a more usable format, including event type, length, sub-type, status,
 * and a payload that can be interpreted as different data types using a union.
 */
struct bat_response {
	u8 event_type; /**< Type of the event (e.g., NVEC_SYS, NVEC_BAT). */
	u8 length;     /**< Length of the payload. */
	u8 sub_type;   /**< Specific command/property sub-type. */
	u8 status;     /**< Status byte. */
	/** @brief Payload for the response, unioned to handle different data types. */
	union {
		char plc[30]; /**< Payload as a character array (e.g., for strings). */
		u16 plu;      /**< Payload as an unsigned 16-bit integer. */
		s16 pls;      /**< Payload as a signed 16-bit integer. */
	};
};

/** @brief Global pointer to the registered battery power supply device. */
static struct power_supply *nvec_bat_psy;
/** @brief Global pointer to the registered AC power supply device. */
static struct power_supply *nvec_psy;

/**
 * @brief Notifier callback for NVEC system events.
 * Functional Utility: Handles system-level events from the NVEC, particularly
 *                     changes in the AC power status (plugged in/unplugged).
 * @param nb Pointer to the notifier block.
 * @param event_type Type of the NVEC event.
 * @param data Pointer to the `bat_response` containing event details.
 * @return NOTIFY_STOP if the event is handled, NOTIFY_DONE or NOTIFY_OK otherwise.
 */
static int nvec_power_notifier(struct notifier_block *nb,
			       unsigned long event_type, void *data)
{
	struct nvec_power *power =
	    container_of(nb, struct nvec_power, notifier);
	struct bat_response *res = data;

	if (event_type != NVEC_SYS)
		return NOTIFY_DONE;

	// Block Logic: Checks if the sub_type is 0, indicating a system status update.
	if (res->sub_type == 0) {
		// Invariant: Updates the `power->on` status only if it has changed.
		if (power->on != res->plu) {
			power->on = res->plu;
			power_supply_changed(nvec_psy); // Notify power supply framework of change.
		}
		return NOTIFY_STOP;
	}
	return NOTIFY_OK;
}

/**
 * @brief Array of battery initialization properties to fetch on battery presence detection.
 * Functional Utility: Specifies the sequence of battery properties to request from the NVEC
 *                     when a battery is newly detected, primarily for static manufacturing data.
 */
static const int bat_init[] = {
	LAST_FULL_CHARGE_CAPACITY, DESIGN_CAPACITY, CRITICAL_CAPACITY,
	MANUFACTURER, MODEL, TYPE,
};

/**
 * @brief Requests battery manufacturing data from the NVEC.
 * Functional Utility: Sends asynchronous NVEC commands to query static battery
 *                     information (e.g., design capacity, manufacturer, model).
 * @param power Pointer to the `nvec_power` structure.
 */
static void get_bat_mfg_data(struct nvec_power *power)
{
	int i;
	char buf[] = { NVEC_BAT, SLOT_STATUS };

	// Block Logic: Iterates through `bat_init` array to send individual requests.
	for (i = 0; i < ARRAY_SIZE(bat_init); i++) {
		buf[1] = bat_init[i]; // Set the sub-type to the current battery property.
		nvec_write_async(power->nvec, buf, 2); // Send the asynchronous request to NVEC.
	}
}

/**
 * @brief Notifier callback for NVEC battery events.
 * Functional Utility: Processes incoming battery-specific responses from the NVEC,
 *                     updating the `nvec_power` structure's battery state and
 *                     notifying the power supply framework of changes.
 * @param nb Pointer to the notifier block.
 * @param event_type Type of the NVEC event.
 * @param data Pointer to the `bat_response` containing event details.
 * @return NOTIFY_STOP if the event is handled, NOTIFY_DONE otherwise.
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

	// Block Logic: Uses a switch statement to handle different battery sub-types.
	switch (res->sub_type) {
	case SLOT_STATUS:
		// Block Logic: Checks battery presence and updates status accordingly.
		if (res->plc[0] & 1) { // Bit 0 indicates battery present.
			// Invariant: If battery was not present before, mark status as changed and fetch manufacturing data.
			if (power->bat_present == 0) {
				status_changed = 1;
				get_bat_mfg_data(power);
			}

			power->bat_present = 1; // Mark battery as present.

			// Block Logic: Interprets bits 1 and 2 of plc[0] for charging status.
			switch ((res->plc[0] >> 1) & 3) {
			case 0: // 00 - Not charging
				power->bat_status =
				    POWER_SUPPLY_STATUS_NOT_CHARGING;
				break;
			case 1: // 01 - Charging
				power->bat_status =
				    POWER_SUPPLY_STATUS_CHARGING;
				break;
			case 2: // 10 - Discharging
				power->bat_status =
				    POWER_SUPPLY_STATUS_DISCHARGING;
				break;
			default: // 11 - Unknown
				power->bat_status = POWER_SUPPLY_STATUS_UNKNOWN;
			}
		} else {
			// Invariant: If battery was present before but now is not, mark status as changed.
			if (power->bat_present == 1)
				status_changed = 1;

			power->bat_present = 0; // Mark battery as not present.
			power->bat_status = POWER_SUPPLY_STATUS_UNKNOWN;
		}
		power->bat_cap = res->plc[1]; // Update battery capacity percentage.
		if (status_changed)
			power_supply_changed(nvec_bat_psy); // Notify power supply framework.
		break;
	case VOLTAGE:
		power->bat_voltage_now = res->plu * 1000; // Convert to microvolts.
		break;
	case TIME_REMAINING:
		power->time_remain = res->plu * 3600; // Convert to seconds.
		break;
	case CURRENT:
		power->bat_current_now = res->pls * 1000; // Convert to microamperes.
		break;
	case AVERAGE_CURRENT:
		power->bat_current_avg = res->pls * 1000; // Convert to microamperes.
		break;
	case CAPACITY_REMAINING:
		power->capacity_remain = res->plu * 1000; // Convert to microamp-hours.
		break;
	case LAST_FULL_CHARGE_CAPACITY:
		power->charge_last_full = res->plu * 1000; // Convert to microamp-hours.
		break;
	case DESIGN_CAPACITY:
		power->charge_full_design = res->plu * 1000; // Convert to microamp-hours.
		break;
	case CRITICAL_CAPACITY:
		power->critical_capacity = res->plu * 1000; // Convert to microamp-hours.
		break;
	case TEMPERATURE:
		power->bat_temperature = res->plu - 2732; // Convert to tenths of degrees Celsius.
		break;
	case MANUFACTURER:
		// Block Logic: Copies manufacturer string from payload.
		memcpy(power->bat_manu, &res->plc, res->length - 2);
		power->bat_model[res->length - 2] = '\0'; // Null-terminate the string.
		break;
	case MODEL:
		// Block Logic: Copies model string from payload.
		memcpy(power->bat_model, &res->plc, res->length - 2);
		power->bat_model[res->length - 2] = '\0'; // Null-terminate the string.
		break;
	case TYPE:
		// Block Logic: Copies type string and determines technology enum.
		memcpy(power->bat_type, &res->plc, res->length - 2);
		power->bat_type[res->length - 2] = '\0'; // Null-terminate the string.
		/*
		 * This differs a little from the spec fill in more if you find
		 * some.
		 */
		// Heuristic: Check if the type string starts with "Li" for Lithium-Ion.
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
 * @brief Retrieves properties for the AC power supply.
 * Functional Utility: Implements the `get_property` callback for the AC power
 *                     supply, providing its online status to the power supply framework.
 * @param psy Pointer to the power_supply structure.
 * @param psp The specific power supply property being requested.
 * @param val Union to store the retrieved property value.
 * @return 0 on success, -EINVAL if the property is not supported.
 */
static int nvec_power_get_property(struct power_supply *psy,
				   enum power_supply_property psp,
				   union power_supply_propval *val)
{
	struct nvec_power *power = dev_get_drvdata(psy->dev.parent);

	// Block Logic: Switches on the requested power supply property.
	switch (psp) {
	case POWER_SUPPLY_PROP_ONLINE:
		val->intval = power->on; // Return the AC online status.
		break;
	default:
		return -EINVAL; // Unsupported property.
	}
	return 0;
}

/**
 * @brief Retrieves properties for the battery power supply.
 * Functional Utility: Implements the `get_property` callback for the battery
 *                     power supply, providing various battery-related metrics
 *                     to the power supply framework.
 * @param psy Pointer to the power_supply structure.
 * @param psp The specific power supply property being requested.
 * @param val Union to store the retrieved property value.
 * @return 0 on success, -EINVAL if the property is not supported.
 */
static int nvec_battery_get_property(struct power_supply *psy,
				     enum power_supply_property psp,
				     union power_supply_propval *val)
{
	struct nvec_power *power = dev_get_drvdata(psy->dev.parent);

	// Block Logic: Switches on the requested power supply property.
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
		return -EINVAL; // Unsupported property.
	}
	return 0;
}

/**
 * @brief Array of power supply properties supported by the AC power supply device.
 */
static enum power_supply_property nvec_power_props[] = {
	POWER_SUPPLY_PROP_ONLINE,
};

/**
 * @brief Array of power supply properties supported by the battery power supply device.
 * Functional Utility: Conditionally includes `POWER_SUPPLY_PROP_CURRENT_AVG` and
 *                     `POWER_SUPPLY_PROP_TEMP` if `EC_FULL_DIAG` is defined,
 *                     allowing for more detailed diagnostics.
 */
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

/**
 * @brief Array indicating which power supply devices the AC adapter supplies power to.
 * Functional Utility: Used by the power supply framework to establish relationships
 *                     between different power supply entities.
 */
static char *nvec_power_supplied_to[] = {
	"battery",
};

/**
 * @brief Descriptor for the battery power supply device.
 * Functional Utility: Defines the characteristics and callbacks for the battery device
 *                     when it's registered with the power supply framework.
 */
static const struct power_supply_desc nvec_bat_psy_desc = {
	.name = "battery",
	.type = POWER_SUPPLY_TYPE_BATTERY,
	.properties = nvec_battery_props,
	.num_properties = ARRAY_SIZE(nvec_battery_props),
	.get_property = nvec_battery_get_property,
};

/**
 * @brief Descriptor for the AC power supply device.
 * Functional Utility: Defines the characteristics and callbacks for the AC device
 *                     when it's registered with the power supply framework.
 */
static const struct power_supply_desc nvec_psy_desc = {
	.name = "ac",
	.type = POWER_SUPPLY_TYPE_MAINS,
	.properties = nvec_power_props,
	.num_properties = ARRAY_SIZE(nvec_power_props),
	.get_property = nvec_power_get_property,
};

/** @brief Counter for round-robin polling of battery properties. */
static int counter;
/**
 * @brief Array of battery properties to iterate through during polling.
 * Functional Utility: Specifies which battery metrics are periodically queried
 *                     from the NVEC, allowing for staggered updates to avoid
 *                     overloading the embedded controller.
 */
static const int bat_iter[] = {
	SLOT_STATUS, VOLTAGE, CURRENT, CAPACITY_REMAINING,
#ifdef EC_FULL_DIAG
	AVERAGE_CURRENT, TEMPERATURE, TIME_REMAINING,
#endif
};

/**
 * @brief Workqueue callback function for polling power supply status.
 * Functional Utility: Periodically requests AC status and a rotating set of
 *                     battery properties from the NVEC to keep the power supply
 *                     framework updated. It uses a round-robin approach for battery
 *                     requests to avoid overwhelming the embedded controller.
 * @param work Pointer to the `work_struct` embedded in `nvec_power.poller`.
 */
static void nvec_power_poll(struct work_struct *work)
{
	char buf[] = { NVEC_SYS, GET_SYSTEM_STATUS };
	struct nvec_power *power = container_of(work, struct nvec_power,
						poller.work);

	// Block Logic: Resets the counter if all battery iteration properties have been requested.
	if (counter >= ARRAY_SIZE(bat_iter))
		counter = 0;

	/* AC status via sys req */
	// Functional Utility: Sends an asynchronous request to the NVEC for the overall system status, which includes AC online state.
	nvec_write_async(power->nvec, buf, 2);
	msleep(100); // Delay to allow NVEC to process the request.

	/*
	 * Select a battery request function via round robin doing it all at
	 * once seems to overload the power supply.
	 */
	// Functional Utility: Implements a round-robin polling strategy for battery properties.
	// Invariant: Only one battery property is requested per poll cycle to prevent NVEC overload.
	buf[0] = NVEC_BAT;
	buf[1] = bat_iter[counter++]; // Get the next battery property to request.
	nvec_write_async(power->nvec, buf, 2); // Send the asynchronous request.

	// Schedule the next poll after a delay.
	schedule_delayed_work(to_delayed_work(work), msecs_to_jiffies(5000));
};

/**
 * @brief Probe function for the NVEC power driver.
 * Functional Utility: Initializes and registers AC and battery power supply
 *                     devices with the kernel's power supply framework. It allocates
 *                     resources, sets up notifier callbacks, and initiates polling.
 * @param pdev Pointer to the platform device representing the NVEC power component.
 * @return 0 on success, or a negative errno on failure.
 */
static int nvec_power_probe(struct platform_device *pdev)
{
	struct power_supply **psy;
	const struct power_supply_desc *psy_desc;
	struct nvec_power *power;
	struct nvec_chip *nvec = dev_get_drvdata(pdev->dev.parent);
	struct power_supply_config psy_cfg = {};

	// Allocate and zero-initialize the `nvec_power` structure.
	power = devm_kzalloc(&pdev->dev, sizeof(struct nvec_power), GFP_NOWAIT);
	if (!power)
		return -ENOMEM;

	dev_set_drvdata(&pdev->dev, power); // Store `power` structure in device data.
	power->nvec = nvec; // Link to the parent NVEC chip.

	// Block Logic: Configures the power supply based on the platform device ID (AC or BAT).
	switch (pdev->id) {
	case AC:
		psy = &nvec_psy; // Point to the global AC power supply pointer.
		psy_desc = &nvec_psy_desc; // Use the AC power supply descriptor.
		psy_cfg.supplied_to = nvec_power_supplied_to; // Indicate it supplies power to the battery.
		psy_cfg.num_supplicants = ARRAY_SIZE(nvec_power_supplied_to);

		power->notifier.notifier_call = nvec_power_notifier; // Set AC notifier callback.

		INIT_DELAYED_WORK(&power->poller, nvec_power_poll); // Initialize polling work.
		schedule_delayed_work(&power->poller, msecs_to_jiffies(5000)); // Start polling.
		break;
	case BAT:
		psy = &nvec_bat_psy; // Point to the global battery power supply pointer.
		psy_desc = &nvec_bat_psy_desc; // Use the battery power supply descriptor.

		power->notifier.notifier_call = nvec_power_bat_notifier; // Set battery notifier callback.
		break;
	default:
		return -ENODEV; // Unknown device ID.
	}

	// Register the NVEC notifier for system events.
	nvec_register_notifier(nvec, &power->notifier, NVEC_SYS);

	// If it's the battery device, fetch initial manufacturing data.
	if (pdev->id == BAT)
		get_bat_mfg_data(power);

	// Register the power supply device with the kernel framework.
	*psy = power_supply_register(&pdev->dev, psy_desc, &psy_cfg);

	return PTR_ERR_OR_ZERO(*psy); // Return result of registration.
}

/**
 * @brief Remove function for the NVEC power driver.
 * Functional Utility: Cleans up resources allocated during the probe phase,
 *                     including canceling pending work, unregistering notifiers,
 *                     and unregistering power supply devices.
 * @param pdev Pointer to the platform device.
 */
static void nvec_power_remove(struct platform_device *pdev)
{
	struct nvec_power *power = platform_get_drvdata(pdev);

	// Cancel any scheduled polling work.
	cancel_delayed_work_sync(&power->poller);
	// Unregister the NVEC notifier.
	nvec_unregister_notifier(power->nvec, &power->notifier);
	// Block Logic: Unregisters the specific power supply device based on its ID.
	switch (pdev->id) {
	case AC:
		power_supply_unregister(nvec_psy);
		break;
	case BAT:
		power_supply_unregister(nvec_bat_psy);
	}
}

/**
 * @brief Platform driver structure for the NVEC power management.
 * Functional Utility: Defines the core operations (probe, remove) and name
 *                     for the platform driver, allowing the kernel to bind it
 *                     to matching platform devices.
 */
static struct platform_driver nvec_power_driver = {
	.probe = nvec_power_probe,
	.remove = nvec_power_remove,
	.driver = {
		   .name = "nvec-power", // Name used for device matching.
	}
};

/**
 * @brief Macro to register the platform driver with the kernel.
 * Functional Utility: This macro is the entry point for the module,
 *                     registering `nvec_power_driver` upon module load.
 */
module_platform_driver(nvec_power_driver);

/** @brief Module author information. */
MODULE_AUTHOR("Ilya Petrov <ilya.muromec@gmail.com>");
/** @brief Module license information (GPL-2.0). */
MODULE_LICENSE("GPL");
/** @brief Short description of the module. */
MODULE_DESCRIPTION("NVEC battery and AC driver");
/** @brief Alias for the platform device, enabling alternative device matching. */
MODULE_ALIAS("platform:nvec-power");
