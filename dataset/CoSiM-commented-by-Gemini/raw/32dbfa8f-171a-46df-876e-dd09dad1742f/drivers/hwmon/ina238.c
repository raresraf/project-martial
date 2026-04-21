/**
 * @file ina238.c
 * @brief Linux kernel driver for the Texas Instruments INA238/INA237/INA228 power monitor chips.
 * 
 * Architectural Intent: Provides a hardware monitoring (hwmon) interface to expose 
 * shunt voltage, bus voltage, current, power, energy, and die temperature measurements 
 * over the I2C bus.
 * 
 * Scaling Strategy: Uses a fixed device-side calibration to maintain precision across 
 * various shunt resistor values by performing relative scaling within the driver logic.
 */

#include <linux/bitops.h>
#include <linux/err.h>
#include <linux/hwmon.h>
#include <linux/i2c.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/regmap.h>

#include <linux/platform_data/ina2xx.h>

/* Register address space definitions for the INA238 family */
#define INA238_CONFIG			0x0
#define INA238_ADC_CONFIG		0x1
#define INA238_SHUNT_CALIBRATION	0x2
#define SQ52206_SHUNT_TEMPCO		0x3
#define INA238_SHUNT_VOLTAGE		0x4
#define INA238_BUS_VOLTAGE		0x5
#define INA238_DIE_TEMP			0x6
#define INA238_CURRENT			0x7
#define INA238_POWER			0x8
#define SQ52206_ENERGY			0x9
#define SQ52206_CHARGE			0xa
#define INA238_DIAG_ALERT		0xb
#define INA238_SHUNT_OVER_VOLTAGE	0xc
#define INA238_SHUNT_UNDER_VOLTAGE	0xd
#define INA238_BUS_OVER_VOLTAGE		0xe
#define INA238_BUS_UNDER_VOLTAGE	0xf
#define INA238_TEMP_LIMIT		0x10
#define INA238_POWER_LIMIT		0x11
#define SQ52206_POWER_PEAK		0x20
#define INA238_DEVICE_ID		0x3f

#define INA238_CONFIG_ADCRANGE		BIT(4)
#define SQ52206_CONFIG_ADCRANGE_HIGH	BIT(4)
#define SQ52206_CONFIG_ADCRANGE_LOW	BIT(3)

#define INA238_DIAG_ALERT_TMPOL		BIT(7)
#define INA238_DIAG_ALERT_SHNTOL	BIT(6)
#define INA238_DIAG_ALERT_SHNTUL	BIT(5)
#define INA238_DIAG_ALERT_BUSOL		BIT(4)
#define INA238_DIAG_ALERT_BUSUL		BIT(3)
#define INA238_DIAG_ALERT_POL		BIT(2)

#define INA238_REGISTERS		0x20
#define INA238_RSHUNT_DEFAULT		10000 /* uOhm */
#define INA238_CONFIG_DEFAULT		0
#define SQ52206_CONFIG_DEFAULT		0x0005
#define INA238_ADC_CONFIG_DEFAULT	0xfb6a
#define INA238_DIAG_ALERT_DEFAULT	0x2000

#define INA238_CALIBRATION_VALUE	16384
#define INA238_FIXED_SHUNT		20000

#define INA238_SHUNT_VOLTAGE_LSB	5
#define INA238_BUS_VOLTAGE_LSB		3125
#define INA238_DIE_TEMP_LSB		1250000
#define SQ52206_BUS_VOLTAGE_LSB		3750
#define SQ52206_DIE_TEMP_LSB		78125
#define INA228_DIE_TEMP_LSB		78125

static const struct regmap_config ina238_regmap_config = {
	.max_register = INA238_REGISTERS,
	.reg_bits = 8,
	.val_bits = 16,
};

enum ina238_ids { ina238, ina237, sq52206, ina228 };

/**
 * @brief Per-variant static configuration parameters.
 */
struct ina238_config {
	bool has_20bit_voltage_current;
	bool has_power_highest;
	bool has_energy;
	u8 temp_shift;
	u32 power_calculate_factor;
	u16 config_default;
	int bus_voltage_lsb;
	int temp_lsb;
};

/**
 * @brief Internal state for an INA238 instance.
 */
struct ina238_data {
	const struct ina238_config *config;
	struct i2c_client *client;
	struct mutex config_lock;
	struct regmap *regmap;
	u32 rshunt;
	int gain;
};

static const struct ina238_config ina238_config[] = {
	[ina238] = {
		.has_20bit_voltage_current = false,
		.has_energy = false,
		.has_power_highest = false,
		.temp_shift = 4,
		.power_calculate_factor = 20,
		.config_default = INA238_CONFIG_DEFAULT,
		.bus_voltage_lsb = INA238_BUS_VOLTAGE_LSB,
		.temp_lsb = INA238_DIE_TEMP_LSB,
	},
	[ina237] = {
		.has_20bit_voltage_current = false,
		.has_energy = false,
		.has_power_highest = false,
		.temp_shift = 4,
		.power_calculate_factor = 20,
		.config_default = INA238_CONFIG_DEFAULT,
		.bus_voltage_lsb = INA238_BUS_VOLTAGE_LSB,
		.temp_lsb = INA238_DIE_TEMP_LSB,
	},
	[sq52206] = {
		.has_20bit_voltage_current = false,
		.has_energy = true,
		.has_power_highest = true,
		.temp_shift = 0,
		.power_calculate_factor = 24,
		.config_default = SQ52206_CONFIG_DEFAULT,
		.bus_voltage_lsb = SQ52206_BUS_VOLTAGE_LSB,
		.temp_lsb = SQ52206_DIE_TEMP_LSB,
	},
	[ina228] = {
		.has_20bit_voltage_current = true,
		.has_energy = true,
		.has_power_highest = false,
		.temp_shift = 0,
		.power_calculate_factor = 20,
		.config_default = INA238_CONFIG_DEFAULT,
		.bus_voltage_lsb = INA238_BUS_VOLTAGE_LSB,
		.temp_lsb = INA228_DIE_TEMP_LSB,
	},
};

/**
 * @brief Reads a 24-bit value from the device using SMBus block read.
 */
static int ina238_read_reg24(const struct i2c_client *client, u8 reg, u32 *val)
{
	u8 data[3];
	int err;

	err = i2c_smbus_read_i2c_block_data(client, reg, 3, data);
	if (err < 0)
		return err;
	/**
	 * Block Logic: Big-endian to CPU-endian conversion.
	 * Pre-condition: SMBus read succeeded.
	 */
	if (err != 3)
		return -EIO;
	*val = (data[0] << 16) | (data[1] << 8) | data[2];

	return 0;
}

/**
 * @brief Reads a 40-bit value (e.g., energy/charge) from the device.
 */
static int ina238_read_reg40(const struct i2c_client *client, u8 reg, u64 *val)
{
	u8 data[5];
	u32 low;
	int err;

	err = i2c_smbus_read_i2c_block_data(client, reg, 5, data);
	if (err < 0)
		return err;
	if (err != 5)
		return -EIO;
	low = (data[1] << 24) | (data[2] << 16) | (data[3] << 8) | data[4];
	*val = ((long long)data[0] << 32) | low;

	return 0;
}

/**
 * @brief Reads and sign-extends a 20-bit measurement field.
 */
static int ina238_read_field_s20(const struct i2c_client *client, u8 reg, s32 *val)
{
	u32 regval;
	int err;

	err = ina238_read_reg24(client, reg, &regval);
	if (err)
		return err;

	// Optimization: Measurement is left-aligned in the 24-bit window; discarding reserved bits.
	regval >>= 4;
	*val = sign_extend32(regval, 19);

	return 0;
}

/**
 * @brief Converts raw 20-bit shunt voltage to millivolts.
 */
static int ina228_read_shunt_voltage(struct device *dev, u32 attr, int channel,
				     long *val)
{
	struct ina238_data *data = dev_get_drvdata(dev);
	int regval;
	int err;

	err = ina238_read_field_s20(data->client, INA238_SHUNT_VOLTAGE, &regval);
	if (err)
		return err;

	/**
	 * Functional Utility: Scaling compensation.
	 * Logic: Applies LSB factor and gain adjustments, then normalizes for 20-bit precision.
	 */
	*val = (regval * INA238_SHUNT_VOLTAGE_LSB) * data->gain / (1000 * 4) / 16;
	return 0;
}

static int ina228_read_bus_voltage(struct device *dev, u32 attr, int channel,
				   long *val)
{
	struct ina238_data *data = dev_get_drvdata(dev);
	int regval;
	int err;

	err = ina238_read_field_s20(data->client, INA238_BUS_VOLTAGE, &regval);
	if (err)
		return err;

	*val = (regval * data->config->bus_voltage_lsb) / 1000 / 16;
	return 0;
}

/**
 * @brief Dispatcher for reading voltage-related hwmon attributes.
 */
static int ina238_read_in(struct device *dev, u32 attr, int channel,
			  long *val)
{
	struct ina238_data *data = dev_get_drvdata(dev);
	int reg, mask;
	int regval;
	int err;

	/**
	 * Block Logic: Routing based on sensor channel.
	 * Invariant: Channel 0 maps to shunt (current-based), Channel 1 to bus (voltage-based).
	 */
	switch (channel) {
	case 0:
		switch (attr) {
		case hwmon_in_input:
			if (data->config->has_20bit_voltage_current)
				return ina228_read_shunt_voltage(dev, attr, channel, val);
			reg = INA238_SHUNT_VOLTAGE;
			break;
		case hwmon_in_max:
			reg = INA238_SHUNT_OVER_VOLTAGE;
			break;
		case hwmon_in_min:
			reg = INA238_SHUNT_UNDER_VOLTAGE;
			break;
		case hwmon_in_max_alarm:
			reg = INA238_DIAG_ALERT;
			mask = INA238_DIAG_ALERT_SHNTOL;
			break;
		case hwmon_in_min_alarm:
			reg = INA238_DIAG_ALERT;
			mask = INA238_DIAG_ALERT_SHNTUL;
			break;
		default:
			return -EOPNOTSUPP;
		}
		break;
	case 1:
		switch (attr) {
		case hwmon_in_input:
			if (data->config->has_20bit_voltage_current)
				return ina228_read_bus_voltage(dev, attr, channel, val);
			reg = INA238_BUS_VOLTAGE;
			break;
		case hwmon_in_max:
			reg = INA238_BUS_OVER_VOLTAGE;
			break;
		case hwmon_in_min:
			reg = INA238_BUS_UNDER_VOLTAGE;
			break;
		case hwmon_in_max_alarm:
			reg = INA238_DIAG_ALERT;
			mask = INA238_DIAG_ALERT_BUSOL;
			break;
		case hwmon_in_min_alarm:
			reg = INA238_DIAG_ALERT;
			mask = INA238_DIAG_ALERT_BUSUL;
			break;
		default:
			return -EOPNOTSUPP;
		}
		break;
	default:
		return -EOPNOTSUPP;
	}

	err = regmap_read(data->regmap, reg, &regval);
	if (err < 0)
		return err;

	/**
	 * Block Logic: Unit conversion for userspace presentation.
	 * Pre-condition: Register data retrieved from hardware.
	 */
	switch (attr) {
	case hwmon_in_input:
	case hwmon_in_max:
	case hwmon_in_min:
		regval = (s16)regval;
		if (channel == 0)
			*val = (regval * INA238_SHUNT_VOLTAGE_LSB) *
				data->gain / (1000 * 4);
		else
			*val = (regval * data->config->bus_voltage_lsb) / 1000;
		break;
	case hwmon_in_max_alarm:
	case hwmon_in_min_alarm:
		*val = !!(regval & mask);
		break;
	}

	return 0;
}

/**
 * @brief Configures voltage threshold registers from user input.
 */
static int ina238_write_in(struct device *dev, u32 attr, int channel,
			   long val)
{
	struct ina238_data *data = dev_get_drvdata(dev);
	int regval;

	if (attr != hwmon_in_max && attr != hwmon_in_min)
		return -EOPNOTSUPP;

	/**
	 * Block Logic: Inverse scaling for threshold programming.
	 * Invariant: Values are clamped to device limits before transmission.
	 */
	switch (channel) {
	case 0:
		regval = clamp_val(val, -163, 163);
		regval = (regval * 1000 * 4) /
			 (INA238_SHUNT_VOLTAGE_LSB * data->gain);
		regval = clamp_val(regval, S16_MIN, S16_MAX);

		switch (attr) {
		case hwmon_in_max:
			return regmap_write(data->regmap,
					    INA238_SHUNT_OVER_VOLTAGE, regval);
		case hwmon_in_min:
			return regmap_write(data->regmap,
					    INA238_SHUNT_UNDER_VOLTAGE, regval);
		default:
			return -EOPNOTSUPP;
		}
	case 1:
		regval = clamp_val(val, 0, 102396);
		regval = (regval * 1000) / data->config->bus_voltage_lsb;
		regval = clamp_val(regval, 0, S16_MAX);

		switch (attr) {
		case hwmon_in_max:
			return regmap_write(data->regmap,
					    INA238_BUS_OVER_VOLTAGE, regval);
		case hwmon_in_min:
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
 * @brief Calculates current through the shunt resistor in milliamps.
 */
static int ina238_read_current(struct device *dev, u32 attr, long *val)
{
	struct ina238_data *data = dev_get_drvdata(dev);
	int regval;
	int err;

	switch (attr) {
	case hwmon_curr_input:
		/**
		 * Block Logic: Precision-aware register retrieval.
		 * Invariant: Reads either 20-bit or 16-bit current fields based on chip variant.
		 */
		if (data->config->has_20bit_voltage_current) {
			err = ina238_read_field_s20(data->client, INA238_CURRENT, &regval);
			if (err)
				return err;
		} else {
			err = regmap_read(data->regmap, INA238_CURRENT, &regval);
			if (err < 0)
				return err;
			regval = (s16)regval;
		}

		/**
		 * Functional Utility: Scaling by shunt resistance ratio.
		 * Invariant: Current is derived from voltage drop relative to configured shunt resistor.
		 */
		*val = div_s64((s64)regval * INA238_FIXED_SHUNT * data->gain,
			       data->rshunt * 4);

		if (data->config->has_20bit_voltage_current)
			*val /= 16;
		break;
	default:
		return -EOPNOTSUPP;
	}

	return 0;
}

/**
 * @brief Retrieves power consumption in microwatts.
 */
static int ina238_read_power(struct device *dev, u32 attr, long *val)
{
	struct ina238_data *data = dev_get_drvdata(dev);
	long long power;
	int regval;
	int err;

	switch (attr) {
	case hwmon_power_input:
		err = ina238_read_reg24(data->client, INA238_POWER, &regval);
		if (err)
			return err;

		power = div_u64(regval * 1000ULL * INA238_FIXED_SHUNT * data->gain *
				data->config->power_calculate_factor, 4 * 100 * data->rshunt);
		*val = clamp_val(power, 0, LONG_MAX);
		break;
	case hwmon_power_input_highest:
		err = ina238_read_reg24(data->client, SQ52206_POWER_PEAK, &regval);
		if (err)
			return err;

		power = div_u64(regval * 1000ULL * INA238_FIXED_SHUNT * data->gain *
				data->config->power_calculate_factor, 4 * 100 * data->rshunt);
		*val = clamp_val(power, 0, LONG_MAX);
		break;
	case hwmon_power_max:
		err = regmap_read(data->regmap, INA238_POWER_LIMIT, &regval);
		if (err)
			return err;

		power = div_u64((regval << 8) * 1000ULL * INA238_FIXED_SHUNT * data->gain *
				data->config->power_calculate_factor, 4 * 100 * data->rshunt);
		*val = clamp_val(power, 0, LONG_MAX);
		break;
	case hwmon_power_max_alarm:
		err = regmap_read(data->regmap, INA238_DIAG_ALERT, &regval);
		if (err)
			return err;

		*val = !!(regval & INA238_DIAG_ALERT_POL);
		break;
	default:
		return -EOPNOTSUPP;
	}

	return 0;
}

static int ina238_write_power(struct device *dev, u32 attr, long val)
{
	struct ina238_data *data = dev_get_drvdata(dev);
	long regval;

	if (attr != hwmon_power_max)
		return -EOPNOTSUPP;

	regval = clamp_val(val, 0, LONG_MAX);
	regval = div_u64(val * 4 * 100 * data->rshunt, data->config->power_calculate_factor *
			1000ULL * INA238_FIXED_SHUNT * data->gain);
	regval = clamp_val(regval >> 8, 0, U16_MAX);

	return regmap_write(data->regmap, INA238_POWER_LIMIT, regval);
}

/**
 * @brief Reads internal die temperature in millidegrees Celsius.
 */
static int ina238_read_temp(struct device *dev, u32 attr, long *val)
{
	struct ina238_data *data = dev_get_drvdata(dev);
	int regval;
	int err;

	switch (attr) {
	case hwmon_temp_input:
		err = regmap_read(data->regmap, INA238_DIE_TEMP, &regval);
		if (err)
			return err;
		*val = div_s64(((s64)((s16)regval) >> data->config->temp_shift) *
			       (s64)data->config->temp_lsb, 10000);
		break;
	case hwmon_temp_max:
		err = regmap_read(data->regmap, INA238_TEMP_LIMIT, &regval);
		if (err)
			return err;
		*val = div_s64(((s64)((s16)regval) >> data->config->temp_shift) *
			       (s64)data->config->temp_lsb, 10000);
		break;
	case hwmon_temp_max_alarm:
		err = regmap_read(data->regmap, INA238_DIAG_ALERT, &regval);
		if (err)
			return err;

		*val = !!(regval & INA238_DIAG_ALERT_TMPOL);
		break;
	default:
		return -EOPNOTSUPP;
	}

	return 0;
}

static int ina238_write_temp(struct device *dev, u32 attr, long val)
{
	struct ina238_data *data = dev_get_drvdata(dev);
	int regval;

	if (attr != hwmon_temp_max)
		return -EOPNOTSUPP;

	regval = clamp_val(val, -40000, 125000);
	regval = div_s64(val * 10000, data->config->temp_lsb) << data->config->temp_shift;
	regval = clamp_val(regval, S16_MIN, S16_MAX) & (0xffff << data->config->temp_shift);

	return regmap_write(data->regmap, INA238_TEMP_LIMIT, regval);
}

/**
 * @brief Sysfs interface to retrieve accumulated energy in microjoules.
 */
static ssize_t energy1_input_show(struct device *dev,
				  struct device_attribute *da, char *buf)
{
	struct ina238_data *data = dev_get_drvdata(dev);
	int ret;
	u64 regval;
	u64 energy;

	ret = ina238_read_reg40(data->client, SQ52206_ENERGY, &regval);
	if (ret)
		return ret;

	energy = div_u64(regval * INA238_FIXED_SHUNT * data->gain * 16 * 10 *
			 data->config->power_calculate_factor, 4 * data->rshunt);

	return sysfs_emit(buf, "%llu\n", energy);
}

static int ina238_read(struct device *dev, enum hwmon_sensor_types type,
		       u32 attr, int channel, long *val)
{
	switch (type) {
	case hwmon_in:
		return ina238_read_in(dev, attr, channel, val);
	case hwmon_curr:
		return ina238_read_current(dev, attr, val);
	case hwmon_power:
		return ina238_read_power(dev, attr, val);
	case hwmon_temp:
		return ina238_read_temp(dev, attr, val);
	default:
		return -EOPNOTSUPP;
	}
	return 0;
}

static int ina238_write(struct device *dev, enum hwmon_sensor_types type,
			u32 attr, int channel, long val)
{
	struct ina238_data *data = dev_get_drvdata(dev);
	int err;

	mutex_lock(&data->config_lock);

	switch (type) {
	case hwmon_in:
		err = ina238_write_in(dev, attr, channel, val);
		break;
	case hwmon_power:
		err = ina238_write_power(dev, attr, val);
		break;
	case hwmon_temp:
		err = ina238_write_temp(dev, attr, val);
		break;
	default:
		err = -EOPNOTSUPP;
		break;
	}

	mutex_unlock(&data->config_lock);
	return err;
}

/**
 * @brief Defines attribute permissions and visibility based on chip capabilities.
 */
static umode_t ina238_is_visible(const void *drvdata,
				 enum hwmon_sensor_types type,
				 u32 attr, int channel)
{
	const struct ina238_data *data = drvdata;
	bool has_power_highest = data->config->has_power_highest;

	switch (type) {
	case hwmon_in:
		switch (attr) {
		case hwmon_in_input:
		case hwmon_in_max_alarm:
		case hwmon_in_min_alarm:
			return 0444;
		case hwmon_in_max:
		case hwmon_in_min:
			return 0644;
		default:
			return 0;
		}
	case hwmon_curr:
		switch (attr) {
		case hwmon_curr_input:
			return 0444;
		default:
			return 0;
		}
	case hwmon_power:
		switch (attr) {
		case hwmon_power_input:
		case hwmon_power_max_alarm:
			return 0444;
		case hwmon_power_max:
			return 0644;
		case hwmon_power_input_highest:
			if (has_power_highest)
				return 0444;
			return 0;
		default:
			return 0;
		}
	case hwmon_temp:
		switch (attr) {
		case hwmon_temp_input:
		case hwmon_temp_max_alarm:
			return 0444;
		case hwmon_temp_max:
			return 0644;
		default:
			return 0;
		}
	default:
		return 0;
	}
}

#define INA238_HWMON_IN_CONFIG (HWMON_I_INPUT | \
				HWMON_I_MAX | HWMON_I_MAX_ALARM | \
				HWMON_I_MIN | HWMON_I_MIN_ALARM)

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
	NULL
};

static const struct hwmon_ops ina238_hwmon_ops = {
	.is_visible = ina238_is_visible,
	.read = ina238_read,
	.write = ina238_write,
};

static const struct hwmon_chip_info ina238_chip_info = {
	.ops = &ina238_hwmon_ops,
	.info = ina238_info,
};

static DEVICE_ATTR_RO(energy1_input);

static struct attribute *ina238_attrs[] = {
	&dev_attr_energy1_input.attr,
	NULL,
};
ATTRIBUTE_GROUPS(ina238);

/**
 * @brief Probes the I2C device and initializes hardware monitoring.
 * Architectural Intent: Validates hardware variant, configures internal scaling factors, 
 * and registers the device with the Linux hwmon subsystem.
 */
static int ina238_probe(struct i2c_client *client)
{
	struct ina2xx_platform_data *pdata = dev_get_platdata(&client->dev);
	struct device *dev = &client->dev;
	struct device *hwmon_dev;
	struct ina238_data *data;
	enum ina238_ids chip;
	int config;
	int ret;

	chip = (uintptr_t)i2c_get_match_data(client);

	data = devm_kzalloc(dev, sizeof(*data), GFP_KERNEL);
	if (!data)
		return -ENOMEM;

	data->client = client;
	data->config = &ina238_config[chip];

	mutex_init(&data->config_lock);

	data->regmap = devm_regmap_init_i2c(client, &ina238_regmap_config);
	if (IS_ERR(data->regmap)) {
		dev_err(dev, "failed to allocate register map\n");
		return PTR_ERR(data->regmap);
	}

	/* load shunt value */
	data->rshunt = INA238_RSHUNT_DEFAULT;
	if (device_property_read_u32(dev, "shunt-resistor", &data->rshunt) < 0 && pdata)
		data->rshunt = pdata->shunt_uohms;
	if (data->rshunt == 0) {
		dev_err(dev, "invalid shunt resister value %u\n", data->rshunt);
		return -EINVAL;
	}

	/* load shunt gain value */
	if (device_property_read_u32(dev, "ti,shunt-gain", &data->gain) < 0)
		data->gain = 4; /* Default of ADCRANGE = 0 */
	if (data->gain != 1 && data->gain != 2 && data->gain != 4) {
		dev_err(dev, "invalid shunt gain value %u\n", data->gain);
		return -EINVAL;
	}

	/* Setup CONFIG register */
	config = data->config->config_default;
	if (chip == sq52206) {
		if (data->gain == 1)
			config |= SQ52206_CONFIG_ADCRANGE_HIGH;
		else if (data->gain == 2)
			config |= SQ52206_CONFIG_ADCRANGE_LOW;
	} else if (data->gain == 1) {
		config |= INA238_CONFIG_ADCRANGE;
	}
	ret = regmap_write(data->regmap, INA238_CONFIG, config);
	if (ret < 0) {
		dev_err(dev, "error configuring the device: %d\n", ret);
		return -ENODEV;
	}

	/* Setup ADC_CONFIG register */
	ret = regmap_write(data->regmap, INA238_ADC_CONFIG,
			   INA238_ADC_CONFIG_DEFAULT);
	if (ret < 0) {
		dev_err(dev, "error configuring the device: %d\n", ret);
		return -ENODEV;
	}

	/* Setup SHUNT_CALIBRATION register with fixed value */
	ret = regmap_write(data->regmap, INA238_SHUNT_CALIBRATION,
			   INA238_CALIBRATION_VALUE);
	if (ret < 0) {
		dev_err(dev, "error configuring the device: %d\n", ret);
		return -ENODEV;
	}

	/* Setup alert/alarm configuration */
	ret = regmap_write(data->regmap, INA238_DIAG_ALERT,
			   INA238_DIAG_ALERT_DEFAULT);
	if (ret < 0) {
		dev_err(dev, "error configuring the device: %d\n", ret);
		return -ENODEV;
	}

	hwmon_dev = devm_hwmon_device_register_with_info(dev, client->name, data,
							 &ina238_chip_info,
							 data->config->has_energy ?
								ina238_groups : NULL);
	if (IS_ERR(hwmon_dev))
		return PTR_ERR(hwmon_dev);

	dev_info(dev, "power monitor %s (Rshunt = %u uOhm, gain = %u)\n",
		 client->name, data->rshunt, data->gain);

	return 0;
}

static const struct i2c_device_id ina238_id[] = {
	{ "ina228", ina228 },
	{ "ina237", ina237 },
	{ "ina238", ina238 },
	{ "sq52206", sq52206 },
	{ }
};
MODULE_DEVICE_TABLE(i2c, ina238_id);

static const struct of_device_id __maybe_unused ina238_of_match[] = {
	{ .compatible = "ti,ina228", .data = (void *)ina228 },
	{ .compatible = "ti,ina237", .data = (void *)ina237 },
	{ .compatible = "ti,ina238", .data = (void *)ina238 },
	{ .compatible = "silergy,sq52206", .data = (void *)sq52206 },
	{ }
};
MODULE_DEVICE_TABLE(of, ina238_of_match);

static struct i2c_driver ina238_driver = {
	.driver = {
		.name	= "ina238",
		.of_match_table = of_match_ptr(ina238_of_match),
	},
	.probe		= ina238_probe,
	.id_table	= ina238_id,
};

module_i2c_driver(ina238_driver);

MODULE_AUTHOR("Nathan Rossi <nathan.rossi@digi.com>");
MODULE_DESCRIPTION("ina238 driver");
MODULE_LICENSE("GPL");
