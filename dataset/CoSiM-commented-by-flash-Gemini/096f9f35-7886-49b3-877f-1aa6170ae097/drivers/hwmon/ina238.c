// SPDX-License-Identifier: GPL-2.0-only
/*
 * Driver for Texas Instruments INA238 power monitor chip
 * Datasheet: https://www.ti.com/product/ina238
 *
 * Copyright (C) 2021 Nathan Rossi <nathan.rossi@digi.com>
 */

/**
 * @file ina238.c
 * @brief Linux kernel driver for the Texas Instruments INA238 power monitor IC and compatible devices.
 * This driver provides hwmon (hardware monitoring) support for reading shunt voltage, bus voltage,
 * current, power, and die temperature, as well as configuring alert limits.
 *
 * The driver abstracts hardware register access via regmap and exposes sensor data
 * through the sysfs interface in standard hwmon formats. It handles specific
 * calibration and scaling for different device variants like INA237 and SQ52206.
 */

#include <linux/err.h>
#include <linux/hwmon.h>
#include <linux/i2c.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/regmap.h>

#include <linux/platform_data/ina2xx.h> // Common platform data for INA2xx series.

/*
 * @name INA238 Register Definitions
 * @brief These macros define the memory-mapped registers for the INA238 device.
 * Functional Utility: Provides direct addressing for device configuration, data acquisition, and alert settings.
 */
#define INA238_CONFIG			0x0  //!< Configuration Register
#define INA238_ADC_CONFIG		0x1  //!< ADC Configuration Register
#define INA238_SHUNT_CALIBRATION	0x2  //!< Shunt Calibration Register
#define SQ52206_SHUNT_TEMPCO		0x3  //!< SQ52206 specific Shunt Temperature Coefficient Register
#define INA238_SHUNT_VOLTAGE		0x4  //!< Shunt Voltage Register
#define INA238_BUS_VOLTAGE		0x5  //!< Bus Voltage Register
#define INA238_DIE_TEMP			0x6  //!< Die Temperature Register
#define INA238_CURRENT			0x7  //!< Current Register
#define INA238_POWER			0x8  //!< Power Register
#define SQ52206_ENERGY			0x9  //!< SQ52206 specific Energy Register
#define SQ52206_CHARGE			0xa  //!< SQ52206 specific Charge Register
#define INA238_DIAG_ALERT		0xb  //!< Diagnostic and Alert Register
#define INA238_SHUNT_OVER_VOLTAGE	0xc  //!< Shunt Overvoltage Limit Register
#define INA238_SHUNT_UNDER_VOLTAGE	0xd  //!< Shunt Undervoltage Limit Register
#define INA238_BUS_OVER_VOLTAGE		0xe  //!< Bus Overvoltage Limit Register
#define INA238_BUS_UNDER_VOLTAGE	0xf  //!< Bus Undervoltage Limit Register
#define INA238_TEMP_LIMIT		0x10 //!< Temperature Limit Register
#define INA238_POWER_LIMIT		0x11 //!< Power Limit Register
#define SQ52206_POWER_PEAK		0x20 //!< SQ52206 specific Power Peak Register
#define INA238_DEVICE_ID		0x3f //!< Device ID Register (not available on INA237)

/*
 * @name Configuration Register Bit Definitions
 * @brief These macros define specific bits within the INA238_CONFIG register.
 * Functional Utility: Used to configure ADC range and other operational parameters.
 */
#define INA238_CONFIG_ADCRANGE		BIT(4) //!< ADC Range selection bit for INA238
#define SQ52206_CONFIG_ADCRANGE_HIGH	BIT(4) //!< High ADC Range selection bit for SQ52206
#define SQ52206_CONFIG_ADCRANGE_LOW	BIT(3) //!< Low ADC Range selection bit for SQ52206

/*
 * @name Diagnostic and Alert Register Bit Definitions
 * @brief These macros define specific bits within the INA238_DIAG_ALERT register.
 * Functional Utility: Used to identify active alerts and configure alert behavior.
 */
#define INA238_DIAG_ALERT_TMPOL		BIT(7) //!< Temperature Over-Limit Alert Flag
#define INA238_DIAG_ALERT_SHNTOL	BIT(6) //!< Shunt Overvoltage Limit Alert Flag
#define INA238_DIAG_ALERT_SHNTUL	BIT(5) //!< Shunt Undervoltage Limit Alert Flag
#define INA238_DIAG_ALERT_BUSOL		BIT(4) //!< Bus Overvoltage Limit Alert Flag
#define INA238_DIAG_ALERT_BUSUL		BIT(3) //!< Bus Undervoltage Limit Alert Flag
#define INA238_DIAG_ALERT_POL		BIT(2) //!< Power Over-Limit Alert Flag

#define INA238_REGISTERS		0x20 //!< Maximum accessible register address.

#define INA238_RSHUNT_DEFAULT		10000 /* uOhm */ //!< Default shunt resistor value in micro-Ohms.

/*
 * @name Default Device Configuration Values
 * @brief These macros define the default power-on configurations for various registers.
 * Functional Utility: Ensures a known good state or typical operating conditions upon initialization.
 */
#define INA238_CONFIG_DEFAULT		0    //!< Default value for INA238_CONFIG.
#define SQ52206_CONFIG_DEFAULT		0x0005 //!< Default value for SQ52206 specific CONFIG.
#define INA238_ADC_CONFIG_DEFAULT	0xfb6a //!< Default ADC configuration: 16 samples, 1052us conversion, continuous mode.
#define INA238_DIAG_ALERT_DEFAULT	0x2000 //!< Default alert configuration: alerts based on averaged values (SLOWALERT).

/*
 * @brief Calibration and Scaling Constants
 * Functional Utility: These constants are crucial for converting raw register values
 * into meaningful physical units (Amps, Watts) by accounting for shunt resistance
 * and internal device scaling factors. The driver uses a fixed internal calibration
 * value and then scales the output based on the actual shunt resistor.
 *
 * The value of the Current register is calculated given the following:
 * Current (A) = (shunt voltage register * 5) * calibration / 81920
 *
 * The maximum shunt voltage is 163.835 mV (0x7fff, ADC_RANGE = 0, gain = 4).
 * With the maximum current value of 0x7fff and a fixed shunt value results in
 * a calibration value of 16384 (0x4000).
 *
 * 0x7fff = (0x7fff * 5) * calibration / 81920
 * calibration = 0x4000
 *
 * Equivalent calibration is applied for the Power register (maximum value for
 * bus voltage is 102396.875 mV, 0x7fff), where the maximum power that can
 * occur is ~16776192 uW (register value 0x147a8):
 *
 * This scaling means the resulting values for Current and Power registers need
 * to be scaled by the difference between the fixed shunt resistor and the
 * actual shunt resistor:
 *
 * Current (mA) = register value * INA238_FIXED_SHUNT * data->gain / (data->rshunt * 4)
 * Power (mW) = 0.2 * register value * INA238_FIXED_SHUNT * data->gain / (data->rshunt * 4)
 * (Specific for SQ52206)
 * Power (mW) = 0.24 * register value * INA238_FIXED_SHUNT * data->gain / (data->rshunt * 4)
 * Energy (uJ) = 16 * 0.24 * register value * INA238_FIXED_SHUNT * data->gain * 1000 / (data->rshunt * 4)
 */
#define INA238_CALIBRATION_VALUE	16384 //!< Fixed internal calibration value used by the device.
#define INA238_FIXED_SHUNT		20000 //!< Fixed shunt resistance (in uOhms) assumed for internal calculations.

/*
 * @name LSB (Least Significant Bit) Values
 * @brief Define the physical unit value represented by one LSB for various registers.
 * Functional Utility: Essential for converting raw digital readings to real-world measurements.
 */
#define INA238_SHUNT_VOLTAGE_LSB	5       //!< Shunt Voltage LSB: 5 uV/lsb
#define INA238_BUS_VOLTAGE_LSB		3125    //!< Bus Voltage LSB: 3.125 mV/lsb (x1000 for uV)
#define INA238_DIE_TEMP_LSB		1250000 //!< Die Temperature LSB: 125.0000 mC/lsb (x1000 for uC)
#define SQ52206_BUS_VOLTAGE_LSB		3750    //!< SQ52206 Bus Voltage LSB: 3.75 mV/lsb
#define SQ52206_DIE_TEMP_LSB		78125   //!< SQ52206 Die Temperature LSB: 7.8125 mC/lsb

/**
 * @brief Configuration for the regmap library.
 * Functional Utility: Defines how register access should be performed for the INA238 device,
 * including the address range and bit widths.
 */
static const struct regmap_config ina238_regmap_config = {
	.max_register = INA238_REGISTERS, //!< Maximum register address.
	.reg_bits = 8,                    //!< Register address bit width.
	.val_bits = 16,                   //!< Register value bit width.
};

/**
 * @enum ina238_ids
 * @brief Enumeration of supported INA238 family device IDs.
 * Functional Utility: Used to differentiate between various chip variants and apply
 * device-specific configurations and scaling factors.
 */
enum ina238_ids { ina238, ina237, sq52206 };

/**
 * @struct ina238_config
 * @brief Device-specific configuration parameters for INA238 family chips.
 * Functional Utility: Stores variant-specific properties and calculation factors,
 * allowing the driver to support multiple INA238-compatible devices with a single codebase.
 */
struct ina238_config {
	bool has_power_highest;		  //!< Indicates if the chip supports power peak measurement.
	bool has_energy;		  //!< Indicates if the chip supports energy measurement.
	u8 temp_shift;			  //!< Temperature calculation shift factor.
	u32 power_calculate_factor;	  //!< Factor used in power calculations.
	u16 config_default;		  //!< Default value for the CONFIG register for this chip.
	int bus_voltage_lsb;		  //!< Bus voltage LSB in uV/lsb for this chip.
	int temp_lsb;			  //!< Temperature LSB for this chip.
};

/**
 * @struct ina238_data
 * @brief Private driver data structure for an INA238 device instance.
 * Functional Utility: Holds all runtime-specific data for a given INA238 device,
 * including device configuration, I2C client, mutex for thread safety, regmap instance,
 * shunt resistance, and gain settings.
 */
struct ina238_data {
	const struct ina238_config *config; //!< Pointer to the device's specific configuration.
	struct i2c_client *client;          //!< I2C client device.
	struct mutex config_lock;           //!< Mutex to protect configuration writes.
	struct regmap *regmap;              //!< Regmap instance for register access.
	u32 rshunt;                         //!< Shunt resistance in micro-Ohms.
	int gain;                           //!< Configured gain.
};

/**
 * @brief Reads a 24-bit register value from the INA238 device via I2C.
 * Functional Utility: Handles specific I2C communication for registers that span 3 bytes.
 *
 * @param client Pointer to the I2C client structure.
 * @param reg The register address to read.
 * @param val Pointer to a u32 where the 24-bit value will be stored.
 * @return 0 on success, or a negative errno on failure.
 */
static int ina238_read_reg24(const struct i2c_client *client, u8 reg, u32 *val)
{
	u8 data[3]; // Buffer to hold 3 bytes of data.
	int err;

	/* Block Logic: Read 3 bytes of block data from the specified register. */
	err = i2c_smbus_read_i2c_block_data(client, reg, 3, data);
	if (err < 0)
		return err; // Returns I2C communication error.
	if (err != 3)
		return -EIO; // Returns I/O error if not exactly 3 bytes were read.
	/* Block Logic: Assembles the 24-bit value from the 3 bytes (big-endian). */
	*val = (data[0] << 16) | (data[1] << 8) | data[2];

	return 0;
}

/**
 * @brief Reads a 40-bit register value from the INA238 device via I2C.
 * Functional Utility: Handles specific I2C communication for registers that span 5 bytes,
 * used for larger values like energy.
 *
 * @param client Pointer to the I2C client structure.
 * @param reg The register address to read.
 * @param val Pointer to a u64 where the 40-bit value will be stored.
 * @return 0 on success, or a negative errno on failure.
 */
static int ina238_read_reg40(const struct i2c_client *client, u8 reg, u64 *val)
{
	u8 data[5]; // Buffer to hold 5 bytes of data.
	u32 low;    // Temporary storage for the lower 32 bits.
	int err;

	/* Block Logic: Reads 5 bytes of block data from the specified register. */
	err = i2c_smbus_read_i2c_block_data(client, reg, 5, data);
	if (err < 0)
		return err; // Returns I2C communication error.
	if (err != 5)
		return -EIO; // Returns I/O error if not exactly 5 bytes were read.

	/* Block Logic: Assembles the 40-bit value from the 5 bytes (big-endian). */
	// The lowest 4 bytes form the lower 32 bits.
	low = (data[1] << 24) | (data[2] << 16) | (data[3] << 8) | data[4];
	// The highest byte forms the upper 8 bits, shifted into a 64-bit value.
	*val = ((long long)data[0] << 32) | low;

	return 0;
}

/**
 * @brief Reads input (voltage/shunt voltage) related sensor data from the INA238.
 * Functional Utility: Maps hwmon attributes (input, max, min, alarms) to corresponding
 * INA238 registers and performs necessary scaling.
 *
 * @param dev Pointer to the device structure.
 * @param attr The hwmon attribute being requested (e.g., hwmon_in_input).
 * @param channel The channel index (0 for shunt voltage, 1 for bus voltage).
 * @param val Pointer to a long where the result will be stored (in microvolts or millivolts).
 * @return 0 on success, or a negative errno on failure.
 */
static int ina238_read_in(struct device *dev, u32 attr, int channel,
			  long *val)
{
	struct ina238_data *data = dev_get_drvdata(dev); // Retrieves driver private data.
	int reg, mask;   // 'reg' for register address, 'mask' for alert bits.
	int regval;      // Raw value read from register.
	int err;

	/* Block Logic: Determines the register to read based on channel and attribute. */
	switch (channel) {
	case 0: /* Shunt Voltage */
		switch (attr) {
		case hwmon_in_input:        reg = INA238_SHUNT_VOLTAGE; break;
		case hwmon_in_max:          reg = INA238_SHUNT_OVER_VOLTAGE; break;
		case hwmon_in_min:          reg = INA238_SHUNT_UNDER_VOLTAGE; break;
		case hwmon_in_max_alarm:    reg = INA238_DIAG_ALERT; mask = INA238_DIAG_ALERT_SHNTOL; break;
		case hwmon_in_min_alarm:    reg = INA238_DIAG_ALERT; mask = INA238_DIAG_ALERT_SHNTUL; break;
		default: return -EOPNOTSUPP; // Attribute not supported for this channel.
		}
		break;
	case 1: /* Bus Voltage */
		switch (attr) {
		case hwmon_in_input:        reg = INA238_BUS_VOLTAGE; break;
		case hwmon_in_max:          reg = INA238_BUS_OVER_VOLTAGE; break;
		case hwmon_in_min:          reg = INA238_BUS_UNDER_VOLTAGE; break;
		case hwmon_in_max_alarm:    reg = INA238_DIAG_ALERT; mask = INA238_DIAG_ALERT_BUSOL; break;
		case hwmon_in_min_alarm:    reg = INA238_DIAG_ALERT; mask = INA238_DIAG_ALERT_BUSUL; break;
		default: return -EOPNOTSUPP; // Attribute not supported for this channel.
		}
		break;
	default:
		return -EOPNOTSUPP; // Channel not supported.
	}

	/* Block Logic: Reads the raw register value. */
	err = regmap_read(data->regmap, reg, &regval);
	if (err < 0)
		return err; // Returns error from regmap_read.

	/* Block Logic: Scales the raw register value into the appropriate hwmon output format. */
	switch (attr) {
	case hwmon_in_input:
	case hwmon_in_max:
	case hwmon_in_min:
		// These registers store signed 16-bit values.
		regval = (s16)regval;
		if (channel == 0)
			/* Scale for shunt voltage (channel 0) by LSB and gain. Result in mV.
			 * LSB 5uV/lsb, gain typically 4. Division by 1000 for mV conversion,
			 * and another 4 due to internal device gain scaling.
			 */
			*val = (regval * INA238_SHUNT_VOLTAGE_LSB) *
				data->gain / (1000 * 4);
		else
			/* Scale for bus voltage (channel 1) by LSB. Result in mV. */
			*val = (regval * data->config->bus_voltage_lsb) / 1000;
		break;
	case hwmon_in_max_alarm:
	case hwmon_in_min_alarm:
		// For alarm attributes, return 1 if the corresponding bit in DIAG_ALERT is set, else 0.
		*val = !!(regval & mask);
		break;
	}

	return 0;
}

/**
 * @brief Writes input (voltage/shunt voltage) limit values to the INA238.
 * Functional Utility: Maps hwmon attributes (max, min) to corresponding INA238 registers
 * and performs necessary inverse scaling for writing.
 *
 * @param dev Pointer to the device structure.
 * @param attr The hwmon attribute being written (e.g., hwmon_in_max).
 * @param channel The channel index (0 for shunt voltage, 1 for bus voltage).
 * @param val The long value to write (in millivolts).
 * @return 0 on success, or a negative errno on failure.
 */
static int ina238_write_in(struct device *dev, u32 attr, int channel,
			   long val)
{
	struct ina238_data *data = dev_get_drvdata(dev); // Retrieves driver private data.
	int regval; // Value to be written to register.

	/* Block Logic: Only hwmon_in_max and hwmon_in_min attributes are supported for writing. */
	if (attr != hwmon_in_max && attr != hwmon_in_min)
		return -EOPNOTSUPP; // Attribute not supported for writing.

	/* Block Logic: Converts the decimal input value to the device's register format. */
	switch (channel) {
	case 0: /* Shunt Voltage */
		// Clamps the input value to the valid shunt voltage range (-163 mV to 163 mV).
		regval = clamp_val(val, -163, 163);
		// Inverse scaling to convert mV to raw register value.
		regval = (regval * 1000 * 4) /
			 (INA238_SHUNT_VOLTAGE_LSB * data->gain);
		// Clamps the raw register value to fit within a signed 16-bit integer.
		regval = clamp_val(regval, S16_MIN, S16_MAX);

		switch (attr) {
		case hwmon_in_max:
			// Writes the calculated value to the Shunt Overvoltage Limit register.
			return regmap_write(data->regmap,
					    INA238_SHUNT_OVER_VOLTAGE, regval);
		case hwmon_in_min:
			// Writes the calculated value to the Shunt Undervoltage Limit register.
			return regmap_write(data->regmap,
					    INA238_SHUNT_UNDER_VOLTAGE, regval);
		default:
			return -EOPNOTSUPP;
		}
	case 1: /* Bus Voltage */
		// Clamps the input value to the valid bus voltage range (0 mV to 102396 mV).
		regval = clamp_val(val, 0, 102396);
		// Inverse scaling to convert mV to raw register value.
		regval = (regval * 1000) / data->config->bus_voltage_lsb;
		// Clamps the raw register value to fit within an unsigned 16-bit integer (max for S16_MAX).
		regval = clamp_val(regval, 0, S16_MAX);

		switch (attr) {
		case hwmon_in_max:
			// Writes the calculated value to the Bus Overvoltage Limit register.
			return regmap_write(data->regmap,
					    INA238_BUS_OVER_VOLTAGE, regval);
		case hwmon_in_min:
			// Writes the calculated value to the Bus Undervoltage Limit register.
			return regmap_write(data->regmap,
					    INA238_BUS_UNDER_VOLTAGE, regval);
		default:
			return -EOPNOTSUPP;
		}
	default:
		return -EOPNOTSUPP;
	}
}

/**
 * @brief Reads current sensor data from the INA238.
 * Functional Utility: Retrieves the current value and applies scaling based on
 * fixed shunt, gain, and actual shunt resistance.
 *
 * @param dev Pointer to the device structure.
 * @param attr The hwmon attribute being requested (hwmon_curr_input).
 * @param val Pointer to a long where the result will be stored (in milliamperes).
 * @return 0 on success, or a negative errno on failure.
 */
static int ina238_read_current(struct device *dev, u32 attr, long *val)
{
	struct ina238_data *data = dev_get_drvdata(dev); // Retrieves driver private data.
	int regval; // Raw value read from register.
	int err;

	/* Block Logic: Only hwmon_curr_input attribute is supported for reading. */
	switch (attr) {
	case hwmon_curr_input:
		// Reads the raw current register value.
		err = regmap_read(data->regmap, INA238_CURRENT, &regval);
		if (err < 0)
			return err; // Returns error from regmap_read.

		/*
		 * Block Logic: Scales the raw current register value to milliamperes.
		 * The register value is signed 16-bit. The result is scaled by
		 * INA238_FIXED_SHUNT, device gain, and inversely by actual shunt
		 * resistance (data->rshunt) and a factor of 4.
		 *
		 * Current (mA) = (regval * INA238_FIXED_SHUNT * data->gain) / (data->rshunt * 4)
		 */
		*val = div_s64((s16)regval * INA238_FIXED_SHUNT * data->gain,
			       data->rshunt * 4);
		break;
	default:
		return -EOPNOTSUPP; // Attribute not supported.
	}

	return 0;
}

/**
 * @brief Reads power related sensor data from the INA238.
 * Functional Utility: Retrieves power values (input, peak, limit, alarm) and applies
 * appropriate scaling to obtain values in microwatts (uW).
 *
 * @param dev Pointer to the device structure.
 * @param attr The hwmon attribute being requested.
 * @param val Pointer to a long where the result will be stored (in microwatts).
 * @return 0 on success, or a negative errno on failure.
 */
static int ina238_read_power(struct device *dev, u32 attr, long *val)
{
	struct ina238_data *data = dev_get_drvdata(dev); // Retrieves driver private data.
	long long power; // Intermediate calculation for power, using 64-bit to prevent overflow.
	int regval;      // Raw value read from register.
	int err;

	switch (attr) {
	case hwmon_power_input:
		// Reads the 24-bit power register.
		err = ina238_read_reg24(data->client, INA238_POWER, &regval);
		if (err)
			return err; // Returns error from register read.

		/*
		 * Block Logic: Scales the raw power register value to microwatts (uW).
		 * The formula considers fixed shunt, gain, actual shunt, and a device-specific
		 * power calculation factor. Multiplied by 1000ULL for uW.
		 *
		 * Power (uW) = (regval * 1000ULL * INA238_FIXED_SHUNT * data->gain * data->config->power_calculate_factor) / (4 * 100 * data->rshunt)
		 */
		power = div_u64(regval * 1000ULL * INA238_FIXED_SHUNT * data->gain *
				data->config->power_calculate_factor, 4 * 100 * data->rshunt);
		// Clamps the calculated power value to fit within a long integer.
		*val = clamp_val(power, 0, LONG_MAX);
		break;
	case hwmon_power_input_highest:
		// Reads the 24-bit power peak register (SQ52206 specific).
		err = ina238_read_reg24(data->client, SQ52206_POWER_PEAK, &regval);
		if (err)
			return err; // Returns error from register read.

		/*
		 * Block Logic: Scales the raw power peak register value to microwatts (uW).
		 * Uses the same scaling logic as for hwmon_power_input.
		 */
		power = div_u64(regval * 1000ULL * INA238_FIXED_SHUNT * data->gain *
				data->config->power_calculate_factor, 4 * 100 * data->rshunt);
		// Clamps the calculated power value to fit within a long integer.
		*val = clamp_val(power, 0, LONG_MAX);
		break;
	case hwmon_power_max:
		// Reads the power limit register.
		err = regmap_read(data->regmap, INA238_POWER_LIMIT, &regval);
		if (err)
			return err; // Returns error from regmap_read.

		/*
		 * Block Logic: Scales the raw power limit register value to microwatts (uW).
		 * The register value is a truncated 24-bit value (lower 8 bits are implicit 0).
		 * It's shifted by 8 bits to reflect its true value before scaling.
		 * Uses the same scaling logic as for power input.
		 */
		power = div_u64((regval << 8) * 1000ULL * INA238_FIXED_SHUNT * data->gain *
				data->config->power_calculate_factor, 4 * 100 * data->rshunt);
		// Clamps the calculated power value to fit within a long integer.
		*val = clamp_val(power, 0, LONG_MAX);
		break;
	case hwmon_power_max_alarm:
		// Reads the diagnostic and alert register to check the power over-limit alarm.
		err = regmap_read(data->regmap, INA238_DIAG_ALERT, &regval);
		if (err)
			return err; // Returns error from regmap_read.

		// Returns 1 if the power over-limit alert bit is set, else 0.
		*val = !!(regval & INA238_DIAG_ALERT_POL);
		break;
	default:
		return -EOPNOTSUPP; // Attribute not supported.
	}

	return 0;
}

/**
 * @brief Writes power limit values to the INA238.
 * Functional Utility: Allows setting the maximum power limit and performing
 * inverse scaling for writing to the device register.
 *
 * @param dev Pointer to the device structure.
 * @param attr The hwmon attribute being written (hwmon_power_max).
 * @param val The long value to write (in microwatts).
 * @return 0 on success, or a negative errno on failure.
 */
static int ina238_write_power(struct device *dev, u32 attr, long val)
{
	struct ina238_data *data = dev_get_drvdata(dev); // Retrieves driver private data.
	long regval; // Value to be written to register.

	/* Block Logic: Only hwmon_power_max attribute is supported for writing. */
	if (attr != hwmon_power_max)
		return -EOPNOTSUPP; // Attribute not supported for writing.

	/*
	 * Block Logic: Converts the decimal input value to the device's register format.
	 * The value is unsigned and positive. Inverse scaling is applied, similar to reading,
	 * but adjusted for writing to a truncated 24-bit register (lower 8 bits are truncated).
	 */
	regval = clamp_val(val, 0, LONG_MAX); // Clamps the input value to valid range.
	// Inverse scaling to convert microwatts to raw register value.
	regval = div_u64(val * 4 * 100 * data->rshunt, data->config->power_calculate_factor *
			1000ULL * INA238_FIXED_SHUNT * data->gain);
	// Truncates the lower 8 bits (equivalent to right shift by 8) and clamps to U16_MAX.
	regval = clamp_val(regval >> 8, 0, U16_MAX);

	// Writes the calculated value to the Power Limit register.
	return regmap_write(data->regmap, INA238_POWER_LIMIT, regval);
}

/**
 * @brief Reads temperature sensor data from the INA238.
 * Functional Utility: Retrieves die temperature values (input, max, alarm) and applies
 * appropriate scaling to obtain values in millicelsius (mC).
 *
 * @param dev Pointer to the device structure.
 * @param attr The hwmon attribute being requested.
 * @param val Pointer to a long where the result will be stored (in millicelsius).
 * @return 0 on success, or a negative errno on failure.
 */
static int ina238_read_temp(struct device *dev, u32 attr, long *val)
{
	struct ina238_data *data = dev_get_drvdata(dev); // Retrieves driver private data.
	int regval; // Raw value read from register.
	int err;

	switch (attr) {
	case hwmon_temp_input:
		// Reads the raw die temperature register value.
		err = regmap_read(data->regmap, INA238_DIE_TEMP, &regval);
		if (err)
			return err; // Returns error from regmap_read.
		/*
		 * Block Logic: Scales the raw temperature register value to millicelsius.
		 * The register value is signed 16-bit. It's shifted by a device-specific
		 * 'temp_shift' factor and then scaled by 'temp_lsb'. Result is in mC.
		 */
		*val = div_s64(((s64)((s16)regval) >> data->config->temp_shift) *
			       (s64)data->config->temp_lsb, 10000); // Divided by 10000 for mC.
		break;
	case hwmon_temp_max:
		// Reads the temperature limit register.
		err = regmap_read(data->regmap, INA238_TEMP_LIMIT, &regval);
		if (err)
			return err; // Returns error from regmap_read.
		/*
		 * Block Logic: Scales the raw temperature limit register value to millicelsius.
		 * Uses the same scaling logic as for hwmon_temp_input.
		 */
		*val = div_s64(((s64)((s16)regval) >> data->config->temp_shift) *
			       (s64)data->config->temp_lsb, 10000); // Divided by 10000 for mC.
		break;
	case hwmon_temp_max_alarm:
		// Reads the diagnostic and alert register to check the temperature over-limit alarm.
		err = regmap_read(data->regmap, INA238_DIAG_ALERT, &regval);
		if (err)
			return err; // Returns error from regmap_read.

		// Returns 1 if the temperature over-limit alert bit is set, else 0.
		*val = !!(regval & INA238_DIAG_ALERT_TMPOL);
		break;
	default:
		return -EOPNOTSUPP; // Attribute not supported.
	}

	return 0;
}

/**
 * @brief Writes temperature limit values to the INA238.
 * Functional Utility: Allows setting the maximum temperature limit and performing
 * inverse scaling for writing to the device register.
 *
 * @param dev Pointer to the device structure.
 * @param attr The hwmon attribute being written (hwmon_temp_max).
 * @param val The long value to write (in millicelsius).
 * @return 0 on success, or a negative errno on failure.
 */
static int ina238_write_temp(struct device *dev, u32 attr, long val)
{
	struct ina238_data *data = dev_get_drvdata(dev); // Retrieves driver private data.
	int regval; // Value to be written to register.

	/* Block Logic: Only hwmon_temp_max attribute is supported for writing. */
	if (attr != hwmon_temp_max)
		return -EOPNOTSUPP; // Attribute not supported for writing.

	/*
	 * Block Logic: Converts the decimal input value to the device's register format.
	 * Clamps the input value to the valid temperature range (-40000 mC to 125000 mC).
	 * Inverse scaling involves dividing by temp_lsb, multiplying by 10000, and then
	 * applying the temp_shift factor.
	 */
	regval = clamp_val(val, -40000, 125000); // Clamps the input value to valid range.
	regval = div_s64(val * 10000, data->config->temp_lsb) << data->config->temp_shift;
	// Clamps the raw register value to fit within a signed 16-bit integer and applies a mask.
	regval = clamp_val(regval, S16_MIN, S16_MAX) & (0xffff << data->config->temp_shift);

	// Writes the calculated value to the Temperature Limit register.
	return regmap_write(data->regmap, INA238_TEMP_LIMIT, regval);
}

/**
 * @brief Reads energy input data for SQ52206 devices.
 * Functional Utility: Retrieves the 40-bit energy register value and scales it
 * to microwatts-seconds (uJ) for display via sysfs.
 *
 * @param dev Pointer to the device structure.
 * @param da Pointer to the device attribute structure.
 * @param buf Character buffer to store the output string.
 * @return Number of bytes written to buffer on success, or a negative errno on failure.
 */
static ssize_t energy1_input_show(struct device *dev,
				  struct device_attribute *da, char *buf)
{
	struct ina238_data *data = dev_get_drvdata(dev); // Retrieves driver private data.
	int ret;
	u64 regval; // Raw 40-bit value read from register.
	u64 energy; // Calculated energy value.

	// Reads the 40-bit energy register.
	ret = ina238_read_reg40(data->client, SQ52206_ENERGY, &regval);
	if (ret)
		return ret; // Returns error from register read.

	/*
	 * Block Logic: Scales the raw 40-bit energy register value to microjoules (uJ).
	 * This calculation involves INA238_FIXED_SHUNT, gain, a factor of 16, a factor of 10,
	 * and the device-specific power_calculate_factor, inversely scaled by actual shunt.
	 *
	 * Energy (uJ) = (regval * INA238_FIXED_SHUNT * data->gain * 16 * 10 * data->config->power_calculate_factor) / (4 * data->rshunt)
	 */
	energy = div_u64(regval * INA238_FIXED_SHUNT * data->gain * 16 * 10 *
			 data->config->power_calculate_factor, 4 * data->rshunt);

	// Formats and outputs the calculated energy value to the sysfs buffer.
	return sysfs_emit(buf, "%llu\n", energy);
}

/**
 * @brief Generic read function for all hwmon sensor types.
 * Functional Utility: Acts as a dispatcher, calling the appropriate
 * sensor-specific read function based on the requested sensor type.
 *
 * @param dev Pointer to the device structure.
 * @param type The hwmon sensor type (e.g., hwmon_in, hwmon_curr).
 * @param attr The specific attribute of the sensor type.
 * @param channel The channel index of the sensor.
 * @param val Pointer to a long where the result will be stored.
 * @return 0 on success, or a negative errno on failure.
 */
static int ina238_read(struct device *dev, enum hwmon_sensor_types type,
		       u32 attr, int channel, long *val)
{
	switch (type) {
	case hwmon_in:
		return ina238_read_in(dev, attr, channel, val); // Delegates to ina238_read_in for voltage inputs.
	case hwmon_curr:
		return ina238_read_current(dev, attr, val);     // Delegates to ina238_read_current for current.
	case hwmon_power:
		return ina238_read_power(dev, attr, val);       // Delegates to ina238_read_power for power.
	case hwmon_temp:
		return ina238_read_temp(dev, attr, val);        // Delegates to ina238_read_temp for temperature.
	default:
		return -EOPNOTSUPP; // Sensor type not supported.
	}
	return 0; // Should not be reached.
}

/**
 * @brief Generic write function for all hwmon sensor types.
 * Functional Utility: Acts as a dispatcher, calling the appropriate
 * sensor-specific write function based on the requested sensor type,
 * and ensures thread-safe access to configuration.
 *
 * @param dev Pointer to the device structure.
 * @param type The hwmon sensor type.
 * @param attr The specific attribute of the sensor type.
 * @param channel The channel index of the sensor.
 * @param val The long value to write.
 * @return 0 on success, or a negative errno on failure.
 */
static int ina238_write(struct device *dev, enum hwmon_sensor_types type,
			u32 attr, int channel, long val)
{
	struct ina238_data *data = dev_get_drvdata(dev); // Retrieves driver private data.
	int err;

	mutex_lock(&data->config_lock); // Acquires mutex to protect configuration registers during write.

	switch (type) {
	case hwmon_in:
		err = ina238_write_in(dev, attr, channel, val); // Delegates to ina238_write_in for voltage inputs.
		break;
	case hwmon_power:
		err = ina238_write_power(dev, attr, val);       // Delegates to ina238_write_power for power.
		break;
	case hwmon_temp:
		err = ina238_write_temp(dev, attr, val);        // Delegates to ina238_write_temp for temperature.
		break;
	default:
		err = -EOPNOTSUPP; // Sensor type not supported.
		break;
	}

	mutex_unlock(&data->config_lock); // Releases mutex.
	return err;
}

/**
 * @brief Determines the visibility (read/write permissions) of hwmon attributes.
 * Functional Utility: Controls which sysfs attributes are exposed and their access modes
 * based on device capabilities (e.g., if it supports power peak) and attribute type.
 *
 * @param drvdata Pointer to the driver private data (ina238_data).
 * @param type The hwmon sensor type.
 * @param attr The specific attribute of the sensor type.
 * @param channel The channel index.
 * @return File mode (e.g., 0444 for read-only, 0644 for read/write) or 0 if not visible.
 */
static umode_t ina238_is_visible(const void *drvdata,
				 enum hwmon_sensor_types type,
				 u32 attr, int channel)
{
	const struct ina238_data *data = drvdata; // Casts drvdata to ina238_data.
	bool has_power_highest = data->config->has_power_highest; // Checks if the chip supports power peak.

	switch (type) {
	case hwmon_in:
		switch (attr) {
		case hwmon_in_input:
		case hwmon_in_max_alarm:
		case hwmon_in_min_alarm:
			return 0444; // Read-only attributes.
		case hwmon_in_max:
		case hwmon_in_min:
			return 0644; // Read/write attributes.
		default:
			return 0; // Not visible.
		}
	case hwmon_curr:
		switch (attr) {
		case hwmon_curr_input:
			return 0444; // Read-only current input.
		default:
			return 0; // Not visible.
		}
	case hwmon_power:
		switch (attr) {
		case hwmon_power_input:
		case hwmon_power_max_alarm:
			return 0444; // Read-only attributes.
		case hwmon_power_max:
			return 0644; // Read/write power max limit.
		case hwmon_power_input_highest:
			// Visible only if the chip supports power highest (peak) measurement.
			if (has_power_highest)
				return 0444; // Read-only power peak.
			return 0; // Not visible.
		default:
			return 0; // Not visible.
		}
	case hwmon_temp:
		switch (attr) {
		case hwmon_temp_input:
		case hwmon_temp_max_alarm:
			return 0444; // Read-only attributes.
		case hwmon_temp_max:
			return 0644; // Read/write temperature max limit.
		default:
			return 0; // Not visible.
		}
	default:
		return 0; // Not visible.
	}
}

/**
 * @name HWMON Input Channel Configuration
 * @brief Bitmask defining the set of hwmon attributes supported for input channels (shunt and bus voltage).
 * Functional Utility: Simplifies the definition of supported attributes for hwmon channels.
 */
#define INA238_HWMON_IN_CONFIG (HWMON_I_INPUT | \
				HWMON_I_MAX | HWMON_I_MAX_ALARM | \
				HWMON_I_MIN | HWMON_I_MIN_ALARM)

/**
 * @brief Array of hwmon_channel_info structures.
 * Functional Utility: Defines the specific sensor channels and their supported attributes
 * that this driver exposes to the hwmon subsystem.
 */
static const struct hwmon_channel_info * const ina238_info[] = {
	HWMON_CHANNEL_INFO(in,
			   /* 0: shunt voltage */
			   INA238_HWMON_IN_CONFIG,
			   /* 1: bus voltage */
			   INA238_HWMON_IN_CONFIG),
	HWMON_CHANNEL_INFO(curr,
			   /* 0: current through shunt */
			   HWMON_C_INPUT),
	HWMON_CHANNEL_INFO(power,
			   /* 0: power */
			   HWMON_P_INPUT | HWMON_P_MAX |
			   HWMON_P_MAX_ALARM | HWMON_P_INPUT_HIGHEST),
	HWMON_CHANNEL_INFO(temp,
			   /* 0: die temperature */
			   HWMON_T_INPUT | HWMON_T_MAX | HWMON_T_MAX_ALARM),
	NULL // Marks the end of the array.
};

/**
 * @brief Hardware monitoring operations structure.
 * Functional Utility: Provides pointers to the driver's functions for checking
 * visibility, reading, and writing hwmon attributes.
 */
static const struct hwmon_ops ina238_hwmon_ops = {
	.is_visible = ina238_is_visible, //!< Function to determine attribute visibility.
	.read = ina238_read,             //!< Generic read function.
	.write = ina238_write,           //!< Generic write function.
};

/**
 * @brief Hardware monitoring chip information structure.
 * Functional Utility: Encapsulates the operations and channel information for the INA238 chip.
 */
static const struct hwmon_chip_info ina238_chip_info = {
	.ops = &ina238_hwmon_ops, //!< Pointer to the hwmon operations.
	.info = ina238_info,      //!< Pointer to the channel information.
};

/*
 * @name Sysfs Attributes
 * @brief Defines a sysfs attribute for energy input.
 * Functional Utility: Exposes the energy measurement of SQ52206 devices via sysfs.
 */
static DEVICE_ATTR_RO(energy1_input); //!< Read-only sysfs attribute for energy1_input.

/**
 * @brief Array of device attributes.
 * Functional Utility: Lists all custom sysfs attributes provided by this driver.
 */
static struct attribute *ina238_attrs[] = {
	&dev_attr_energy1_input.attr, // Includes the energy1_input attribute.
	NULL, // Marks the end of the array.
};
ATTRIBUTE_GROUPS(ina238); // Macro to generate attribute groups for the hwmon device.

/**
 * @brief Probe function for the INA238 I2C driver.
 * Functional Utility: Initializes the INA238 device upon discovery, allocates resources,
 * configures registers, and registers the device with the hwmon subsystem.
 *
 * @param client Pointer to the I2C client structure.
 * @return 0 on success, or a negative errno on failure.
 */
static int ina238_probe(struct i2c_client *client)
{
	struct ina2xx_platform_data *pdata = dev_get_platdata(&client->dev); // Retrieves platform data if available.
	struct device *dev = &client->dev;         // Pointer to the device structure.
	struct device *hwmon_dev;                  // Pointer to the registered hwmon device.
	struct ina238_data *data;                  // Driver private data structure.
	enum ina238_ids chip;                      // Detected chip ID.
	int config;                                // Temporary variable for configuration register value.
	int ret;                                   // Return value for function calls.

	// Block Logic: Determine the chip type based on the I2C device match data.
	chip = (uintptr_t)i2c_get_match_data(client);

	// Block Logic: Allocate and initialize driver private data.
	data = devm_kzalloc(dev, sizeof(*data), GFP_KERNEL); // Allocate memory for ina238_data.
	if (!data)
		return -ENOMEM; // Returns if memory allocation fails.

	data->client = client; // Stores a pointer to the I2C client.
	data->config = &ina238_config[chip]; // Assigns chip-specific configuration.

	mutex_init(&data->config_lock); // Initializes the mutex for configuration lock.

	// Block Logic: Initialize regmap for I2C register access.
	data->regmap = devm_regmap_init_i2c(client, &ina238_regmap_config);
	if (IS_ERR(data->regmap)) {
		dev_err(dev, "failed to allocate register map\n"); // Logs error if regmap initialization fails.
		return PTR_ERR(data->regmap); // Returns error from regmap_init_i2c.
	}

	// Block Logic: Load shunt resistor value from device tree or platform data.
	data->rshunt = INA238_RSHUNT_DEFAULT; // Sets default shunt resistance.
	// Attempts to read "shunt-resistor" from device tree, falls back to platform data.
	if (device_property_read_u32(dev, "shunt-resistor", &data->rshunt) < 0 && pdata)
		data->rshunt = pdata->shunt_uohms;
	if (data->rshunt == 0) {
		dev_err(dev, "invalid shunt resister value %u\n", data->rshunt); // Logs error for invalid shunt.
		return -EINVAL; // Returns invalid argument error.
	}

	// Block Logic: Load shunt gain value from device tree or use default.
	// Attempts to read "ti,shunt-gain" from device tree.
	if (device_property_read_u32(dev, "ti,shunt-gain", &data->gain) < 0)
		data->gain = 4; /* Default of ADCRANGE = 0 */
	if (data->gain != 1 && data->gain != 2 && data->gain != 4) {
		dev_err(dev, "invalid shunt gain value %u\n", data->gain); // Logs error for invalid gain.
		return -EINVAL; // Returns invalid argument error.
	}

	// Block Logic: Setup INA238_CONFIG register based on chip and gain.
	config = data->config->config_default; // Starts with chip's default config.
	if (chip == sq52206) {
		if (data->gain == 1)
			// For SQ52206, if gain is 1, set high ADC range.
			config |= SQ52206_CONFIG_ADCRANGE_HIGH;
		else if (data->gain == 2)
			// For SQ52206, if gain is 2, set low ADC range.
			config |= SQ52206_CONFIG_ADCRANGE_LOW;
	} else if (data->gain == 1) {
		// For other INA238 variants, if gain is 1, set ADCRANGE bit.
		config |= INA238_CONFIG_ADCRANGE;
	}
	// Writes the configured value to the CONFIG register.
	ret = regmap_write(data->regmap, INA238_CONFIG, config);
	if (ret < 0) {
		dev_err(dev, "error configuring the device: %d\n", ret); // Logs error if write fails.
		return -ENODEV; // Returns no such device error.
	}

	// Block Logic: Setup ADC_CONFIG register with default value.
	ret = regmap_write(data->regmap, INA238_ADC_CONFIG,
			   INA238_ADC_CONFIG_DEFAULT);
	if (ret < 0) {
		dev_err(dev, "error configuring the device: %d\n", ret); // Logs error if write fails.
		return -ENODEV; // Returns no such device error.
	}

	// Block Logic: Setup SHUNT_CALIBRATION register with fixed value.
	ret = regmap_write(data->regmap, INA238_SHUNT_CALIBRATION,
			   INA238_CALIBRATION_VALUE);
	if (ret < 0) {
		dev_err(dev, "error configuring the device: %d\n", ret); // Logs error if write fails.
		return -ENODEV; // Returns no such device error.
	}

	// Block Logic: Setup DIAG_ALERT register with default value for alert/alarm configuration.
	ret = regmap_write(data->regmap, INA238_DIAG_ALERT,
			   INA238_DIAG_ALERT_DEFAULT);
	if (ret < 0) {
		dev_err(dev, "error configuring the device: %d\n", ret); // Logs error if write fails.
		return -ENODEV; // Returns no such device error.
	}

	// Block Logic: Register the hwmon device with the kernel.
	// Uses ina238_chip_info for operations and channel info.
	// If the chip has energy support, it also registers ina238_groups for energy attributes.
	hwmon_dev = devm_hwmon_device_register_with_info(dev, client->name, data,
							 &ina238_chip_info,
							 data->config->has_energy ?
								ina238_groups : NULL);
	if (IS_ERR(hwmon_dev))
		return PTR_ERR(hwmon_dev); // Returns error from hwmon_device_register_with_info.

	// Logs successful probe information.
	dev_info(dev, "power monitor %s (Rshunt = %u uOhm, gain = %u)\n",
		 client->name, data->rshunt, data->gain);

	return 0; // Probe successful.
}

/**
 * @brief I2C device ID table.
 * Functional Utility: Used by the kernel's I2C subsystem to match I2C devices to this driver.
 */
static const struct i2c_device_id ina238_id[] = {
	{ "ina237", ina237 },   // Matches "ina237" device, maps to ina237 enum.
	{ "ina238", ina238 },   // Matches "ina238" device, maps to ina238 enum.
	{ "sq52206", sq52206 }, // Matches "sq52206" device, maps to sq52206 enum.
	{ } // Sentinel indicating the end of the table.
};
MODULE_DEVICE_TABLE(i2c, ina238_id); // Informs the kernel about this I2C device table.

/**
 * @brief Open Firmware (OF) device ID table.
 * Functional Utility: Used by the kernel to match devices defined in device tree
 * to this driver based on compatible strings.
 */
static const struct of_device_id __maybe_unused ina238_of_match[] = {
	{
		.compatible = "ti,ina237", // Matches "ti,ina237" compatible string.
		.data = (void *)ina237     // Maps to ina237 enum.
	},
	{
		.compatible = "ti,ina238", // Matches "ti,ina238" compatible string.
		.data = (void *)ina238     // Maps to ina238 enum.
	},
	{
		.compatible = "silergy,sq52206", // Matches "silergy,sq52206" compatible string.
		.data = (void *)sq52206          // Maps to sq52206 enum.
	},
	{ } // Sentinel indicating the end of the table.
};
MODULE_DEVICE_TABLE(of, ina238_of_match); // Informs the kernel about this OF device table.

/**
 * @brief I2C driver structure for the INA238.
 * Functional Utility: Defines the core properties of the I2C driver, including
 * its name, OF match table, probe function, and ID table.
 */
static struct i2c_driver ina238_driver = {
	.driver = {
		.name	= "ina238",                      //!< Driver name.
		.of_match_table = of_match_ptr(ina238_of_match), //!< OF match table.
	},
	.probe		= ina238_probe,                  //!< Probe function called when device is found.
	.id_table	= ina238_id,                     //!< I2C device ID table.
};

module_i2c_driver(ina238_driver); // Macro to register the I2C driver with the kernel.

MODULE_AUTHOR("Nathan Rossi <nathan.rossi@digi.com>"); //!< Module author.
MODULE_DESCRIPTION("ina238 driver");                   //!< Module description.
MODULE_LICENSE("GPL");                                 //!< Module license.