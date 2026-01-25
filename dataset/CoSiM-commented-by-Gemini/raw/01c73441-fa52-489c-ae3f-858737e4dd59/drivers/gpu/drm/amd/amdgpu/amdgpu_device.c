/**
 * @file amdgpu_device.c
 * @brief Core AMD GPU device management
 *
 * This file contains the top-level device management functions for the amdgpu
 * driver. It is responsible for the entire lifecycle of an `amdgpu_device`
 * instance, from initialization to teardown. This includes:
 *
 * - Probing and identifying the ASIC.
 * - Initializing all the hardware IP blocks (GFX, VCN, etc.).
 * - Managing device power states (suspend, resume, runtime PM).
 * - Handling GPU reset and recovery from hangs.
 * - Providing low-level register and VRAM access helpers.
 * - Setting up and managing the scheduler for command submission.
 * - Interfacing with other kernel subsystems like ACPI, VGA switcheroo, and IOMMU.
 */
/*
 * Copyright 2008 Advanced Micro Devices, Inc.
 * Copyright 2008 Red Hat Inc.
 * Copyright 2009 Jerome Glisse.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * Authors: Dave Airlie
 *          Alex Deucher
 *          Jerome Glisse
 */

#include <linux/aperture.h>
#include <linux/power_supply.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/console.h>
#include <linux/slab.h>
#include <linux/iommu.h>
#include <linux/pci.h>
#include <linux/pci-p2pdma.h>
#include <linux/apple-gmux.h>

#include <drm/drm_atomic_helper.h>
#include <drm/drm_client_event.h>
#include <drm/drm_crtc_helper.h>
#include <drm/drm_probe_helper.h>
#include <drm/amdgpu_drm.h>
#include <linux/device.h>
#include <linux/vgaarb.h>
#include <linux/vga_switcheroo.h>
#include <linux/efi.h>
#include "amdgpu.h"
#include "amdgpu_trace.h"
#include "amdgpu_i2c.h"
#include "atom.h"
#include "amdgpu_atombios.h"
#include "amdgpu_atomfirmware.h"
#include "amd_pcie.h"
#ifdef CONFIG_DRM_AMDGPU_SI
#include "si.h"
#endif
#ifdef CONFIG_DRM_AMDGPU_CIK
#include "cik.h"
#endif
#include "vi.h"
#include "soc15.h"
#include "nv.h"
#include "bif/bif_4_1_d.h"
#include <linux/firmware.h>
#include "amdgpu_vf_error.h"

#include "amdgpu_amdkfd.h"
#include "amdgpu_pm.h"

#include "amdgpu_xgmi.h"
#include "amdgpu_ras.h"
#include "amdgpu_pmu.h"
#include "amdgpu_fru_eeprom.h"
#include "amdgpu_reset.h"
#include "amdgpu_virt.h"
#include "amdgpu_dev_coredump.h"

#include <linux/suspend.h>
#include <drm/task_barrier.h>
#include <linux/pm_runtime.h>

#include <drm/drm_drv.h>

#if IS_ENABLED(CONFIG_X86)
#include <asm/intel-family.h>
#include <asm/cpu_device_id.h>
#endif

MODULE_FIRMWARE("amdgpu/vega10_gpu_info.bin");
MODULE_FIRMWARE("amdgpu/vega12_gpu_info.bin");
MODULE_FIRMWARE("amdgpu/raven_gpu_info.bin");
MODULE_FIRMWARE("amdgpu/picasso_gpu_info.bin");
MODULE_FIRMWARE("amdgpu/raven2_gpu_info.bin");
MODULE_FIRMWARE("amdgpu/arcturus_gpu_info.bin");
MODULE_FIRMWARE("amdgpu/navi12_gpu_info.bin");

#define AMDGPU_RESUME_MS		2000
#define AMDGPU_MAX_RETRY_LIMIT		2
#define AMDGPU_RETRY_SRIOV_RESET(r) ((r) == -EBUSY || (r) == -ETIMEDOUT || (r) == -EINVAL)
#define AMDGPU_PCIE_INDEX_FALLBACK (0x38 >> 2)
#define AMDGPU_PCIE_INDEX_HI_FALLBACK (0x44 >> 2)
#define AMDGPU_PCIE_DATA_FALLBACK (0x3C >> 2)

#define AMDGPU_VBIOS_SKIP (1U << 0)
#define AMDGPU_VBIOS_OPTIONAL (1U << 1)

static const struct drm_driver amdgpu_kms_driver;

const char *amdgpu_asic_name[] = {
	"TAHITI",
	"PITCAIRN",
	"VERDE",
	"OLAND",
	"HAINAN",
	"BONAIRE",
	"KAVERI",
	"KABINI",
	"HAWAII",
	"MULLINS",
	"TOPAZ",
	"TONGA",
	"FIJI",
	"CARRIZO",
	"STONEY",
	"POLARIS10",
	"POLARIS11",
	"POLARIS12",
	"VEGAM",
	"VEGA10",
	"VEGA12",
	"VEGA20",
	"RAVEN",
	"ARCTURUS",
	"RENOIR",
	"ALDEBARAN",
	"NAVI10",
	"CYAN_SKILLFISH",
	"NAVI14",
	"NAVI12",
	"SIENNA_CICHLID",
	"NAVY_FLOUNDER",
	"VANGOGH",
	"DIMGREY_CAVEFISH",
	"BEIGE_GOBY",
	"YELLOW_CARP",
	"IP DISCOVERY",
	"LAST",
};

#define AMDGPU_IP_BLK_MASK_ALL GENMASK(AMD_IP_BLOCK_TYPE_NUM  - 1, 0)
/*
 * Default init level where all blocks are expected to be initialized. This is
 * the level of initialization expected by default and also after a full reset
 * of the device.
 */
struct amdgpu_init_level amdgpu_init_default = {
	.level = AMDGPU_INIT_LEVEL_DEFAULT,
	.hwini_ip_block_mask = AMDGPU_IP_BLK_MASK_ALL,
};

struct amdgpu_init_level amdgpu_init_recovery = {
	.level = AMDGPU_INIT_LEVEL_RESET_RECOVERY,
	.hwini_ip_block_mask = AMDGPU_IP_BLK_MASK_ALL,
};

/*
 * Minimal blocks needed to be initialized before a XGMI hive can be reset. This
 * is used for cases like reset on initialization where the entire hive needs to
 * be reset before first use.
 */
struct amdgpu_init_level amdgpu_init_minimal_xgmi = {
	.level = AMDGPU_INIT_LEVEL_MINIMAL_XGMI,
	.hwini_ip_block_mask =
		BIT(AMD_IP_BLOCK_TYPE_GMC) | BIT(AMD_IP_BLOCK_TYPE_SMC) |
		BIT(AMD_IP_BLOCK_TYPE_COMMON) | BIT(AMD_IP_BLOCK_TYPE_IH) |
		BIT(AMD_IP_BLOCK_TYPE_PSP)
};

static inline bool amdgpu_ip_member_of_hwini(struct amdgpu_device *adev,
					     enum amd_ip_block_type block)
{
	return (adev->init_lvl->hwini_ip_block_mask & (1U << block)) != 0;
}

void amdgpu_set_init_level(struct amdgpu_device *adev,
			   enum amdgpu_init_lvl_id lvl)
{
	switch (lvl) {
	case AMDGPU_INIT_LEVEL_MINIMAL_XGMI:
		adev->init_lvl = &amdgpu_init_minimal_xgmi;
		break;
	case AMDGPU_INIT_LEVEL_RESET_RECOVERY:
		adev->init_lvl = &amdgpu_init_recovery;
		break;
	case AMDGPU_INIT_LEVEL_DEFAULT:
		fallthrough;
	default:
		adev->init_lvl = &amdgpu_init_default;
		break;
	}
}

static inline void amdgpu_device_stop_pending_resets(struct amdgpu_device *adev);
static int amdgpu_device_pm_notifier(struct notifier_block *nb, unsigned long mode,
				     void *data);

/**
 * DOC: pcie_replay_count
 *
 * The amdgpu driver provides a sysfs API for reporting the total number
 * of PCIe replays (NAKs).
 * The file pcie_replay_count is used for this and returns the total
 * number of replays as a sum of the NAKs generated and NAKs received.
 */

static ssize_t amdgpu_device_get_pcie_replay_count(struct device *dev,
		struct device_attribute *attr, char *buf)
{
	struct drm_device *ddev = dev_get_drvdata(dev);
	struct amdgpu_device *adev = drm_to_adev(ddev);
	uint64_t cnt = amdgpu_asic_get_pcie_replay_count(adev);

	return sysfs_emit(buf, "%llu\n", cnt);
}

static DEVICE_ATTR(pcie_replay_count, 0444,
		amdgpu_device_get_pcie_replay_count, NULL);

static int amdgpu_device_attr_sysfs_init(struct amdgpu_device *adev)
{
	int ret = 0;

	if (!amdgpu_sriov_vf(adev))
		ret = sysfs_create_file(&adev->dev->kobj,
					&dev_attr_pcie_replay_count.attr);

	return ret;
}

static void amdgpu_device_attr_sysfs_fini(struct amdgpu_device *adev)
{
	if (!amdgpu_sriov_vf(adev))
		sysfs_remove_file(&adev->dev->kobj,
				  &dev_attr_pcie_replay_count.attr);
}

static ssize_t amdgpu_sysfs_reg_state_get(struct file *f, struct kobject *kobj,
					  const struct bin_attribute *attr, char *buf,
					  loff_t ppos, size_t count)
{
	struct device *dev = kobj_to_dev(kobj);
	struct drm_device *ddev = dev_get_drvdata(dev);
	struct amdgpu_device *adev = drm_to_adev(ddev);
	ssize_t bytes_read;

	switch (ppos) {
	case AMDGPU_SYS_REG_STATE_XGMI:
		bytes_read = amdgpu_asic_get_reg_state(
			adev, AMDGPU_REG_STATE_TYPE_XGMI, buf, count);
		break;
	case AMDGPU_SYS_REG_STATE_WAFL:
		bytes_read = amdgpu_asic_get_reg_state(
			adev, AMDGPU_REG_STATE_TYPE_WAFL, buf, count);
		break;
	case AMDGPU_SYS_REG_STATE_PCIE:
		bytes_read = amdgpu_asic_get_reg_state(
			adev, AMDGPU_REG_STATE_TYPE_PCIE, buf, count);
		break;
	case AMDGPU_SYS_REG_STATE_USR:
		bytes_read = amdgpu_asic_get_reg_state(
			adev, AMDGPU_REG_STATE_TYPE_USR, buf, count);
		break;
	case AMDGPU_SYS_REG_STATE_USR_1:
		bytes_read = amdgpu_asic_get_reg_state(
			adev, AMDGPU_REG_STATE_TYPE_USR_1, buf, count);
		break;
	default:
		return -EINVAL;
	}

	return bytes_read;
}

static const BIN_ATTR(reg_state, 0444, amdgpu_sysfs_reg_state_get, NULL,
		      AMDGPU_SYS_REG_STATE_END);

int amdgpu_reg_state_sysfs_init(struct amdgpu_device *adev)
{
	int ret;

	if (!amdgpu_asic_get_reg_state_supported(adev))
		return 0;

	ret = sysfs_create_bin_file(&adev->dev->kobj, &bin_attr_reg_state);

	return ret;
}

void amdgpu_reg_state_sysfs_fini(struct amdgpu_device *adev)
{
	if (!amdgpu_asic_get_reg_state_supported(adev))
		return;
	sysfs_remove_bin_file(&adev->dev->kobj, &bin_attr_reg_state);
}

int amdgpu_ip_block_suspend(struct amdgpu_ip_block *ip_block)
{
	int r;

	if (ip_block->version->funcs->suspend) {
		r = ip_block->version->funcs->suspend(ip_block);
		if (r) {
			dev_err(ip_block->adev->dev,
				"suspend of IP block <%s> failed %d\n",
				ip_block->version->funcs->name, r);
			return r;
		}
	}

	ip_block->status.hw = false;
	return 0;
}

int amdgpu_ip_block_resume(struct amdgpu_ip_block *ip_block)
{
	int r;

	if (ip_block->version->funcs->resume) {
		r = ip_block->version->funcs->resume(ip_block);
		if (r) {
			dev_err(ip_block->adev->dev,
				"resume of IP block <%s> failed %d\n",
				ip_block->version->funcs->name, r);
			return r;
		}
	}

	ip_block->status.hw = true;
	return 0;
}

/**
 * DOC: board_info
 *
 * The amdgpu driver provides a sysfs API for giving board related information.
 * It provides the form factor information in the format
 *
 *   type : form factor
 *
 * Possible form factor values
 *
 * - "cem"		- PCIE CEM card
 * - "oam"		- Open Compute Accelerator Module
 * - "unknown"	- Not known
 *
 */

static ssize_t amdgpu_device_get_board_info(struct device *dev,
					    struct device_attribute *attr,
					    char *buf)
{
	struct drm_device *ddev = dev_get_drvdata(dev);
	struct amdgpu_device *adev = drm_to_adev(ddev);
	enum amdgpu_pkg_type pkg_type = AMDGPU_PKG_TYPE_CEM;
	const char *pkg;

	if (adev->smuio.funcs && adev->smuio.funcs->get_pkg_type)
		pkg_type = adev->smuio.funcs->get_pkg_type(adev);

	switch (pkg_type) {
	case AMDGPU_PKG_TYPE_CEM:
		pkg = "cem";
		break;
	case AMDGPU_PKG_TYPE_OAM:
		pkg = "oam";
		break;
	default:
		pkg = "unknown";
		break;
	}

	return sysfs_emit(buf, "%s : %s\n", "type", pkg);
}

static DEVICE_ATTR(board_info, 0444, amdgpu_device_get_board_info, NULL);

static struct attribute *amdgpu_board_attrs[] = {
	&dev_attr_board_info.attr,
	NULL,
};

static umode_t amdgpu_board_attrs_is_visible(struct kobject *kobj,
					     struct attribute *attr, int n)
{
	struct device *dev = kobj_to_dev(kobj);
	struct drm_device *ddev = dev_get_drvdata(dev);
	struct amdgpu_device *adev = drm_to_adev(ddev);

	if (adev->flags & AMD_IS_APU)
		return 0;

	return attr->mode;
}

static const struct attribute_group amdgpu_board_attrs_group = {
	.attrs = amdgpu_board_attrs,
	.is_visible = amdgpu_board_attrs_is_visible
};

static void amdgpu_device_get_pcie_info(struct amdgpu_device *adev);


/**
 * amdgpu_device_supports_px - Is the device a dGPU with ATPX power control
 *
 * @dev: drm_device pointer
 *
 * Returns true if the device is a dGPU with ATPX power control,
 * otherwise return false.
 */
bool amdgpu_device_supports_px(struct drm_device *dev)
{
	struct amdgpu_device *adev = drm_to_adev(dev);

	if ((adev->flags & AMD_IS_PX) && !amdgpu_is_atpx_hybrid())
		return true;
	return false;
}

/**
 * amdgpu_device_supports_boco - Is the device a dGPU with ACPI power resources
 *
 * @dev: drm_device pointer
 *
 * Returns true if the device is a dGPU with ACPI power control,
 * otherwise return false.
 */
bool amdgpu_device_supports_boco(struct drm_device *dev)
{
	struct amdgpu_device *adev = drm_to_adev(dev);

	if (!IS_ENABLED(CONFIG_HOTPLUG_PCI_PCIE))
		return false;

	if (adev->has_pr3 ||
	    ((adev->flags & AMD_IS_PX) && amdgpu_is_atpx_hybrid()))
		return true;
	return false;
}

/**
 * amdgpu_device_supports_baco - Does the device support BACO
 *
 * @dev: drm_device pointer
 *
 * Return:
 * 1 if the device supports BACO;
 * 3 if the device supports MACO (only works if BACO is supported)
 * otherwise return 0.
 */
int amdgpu_device_supports_baco(struct drm_device *dev)
{
	struct amdgpu_device *adev = drm_to_adev(dev);

	return amdgpu_asic_supports_baco(adev);
}

void amdgpu_device_detect_runtime_pm_mode(struct amdgpu_device *adev)
{
	struct drm_device *dev;
	int bamaco_support;

	dev = adev_to_drm(adev);

	adev->pm.rpm_mode = AMDGPU_RUNPM_NONE;
	bamaco_support = amdgpu_device_supports_baco(dev);

	switch (amdgpu_runtime_pm) {
	case 2:
		if (bamaco_support & MACO_SUPPORT) {
			adev->pm.rpm_mode = AMDGPU_RUNPM_BAMACO;
			dev_info(adev->dev, "Forcing BAMACO for runtime pm\n");
		} else if (bamaco_support == BACO_SUPPORT) {
			adev->pm.rpm_mode = AMDGPU_RUNPM_BACO;
			dev_info(adev->dev, "Requested mode BAMACO not available,fallback to use BACO\n");
		}
		break;
	case 1:
		if (bamaco_support & BACO_SUPPORT) {
			adev->pm.rpm_mode = AMDGPU_RUNPM_BACO;
			dev_info(adev->dev, "Forcing BACO for runtime pm\n");
		}
		break;
	case -1:
	case -2:
		if (amdgpu_device_supports_px(dev)) { /* enable PX as runtime mode */
			adev->pm.rpm_mode = AMDGPU_RUNPM_PX;
			dev_info(adev->dev, "Using ATPX for runtime pm\n");
		} else if (amdgpu_device_supports_boco(dev)) { /* enable boco as runtime mode */
			adev->pm.rpm_mode = AMDGPU_RUNPM_BOCO;
			dev_info(adev->dev, "Using BOCO for runtime pm\n");
		} else {
			if (!bamaco_support)
				goto no_runtime_pm;

			switch (adev->asic_type) {
			case CHIP_VEGA20:
			case CHIP_ARCTURUS:
				/* BACO are not supported on vega20 and arctrus */
				break;
			case CHIP_VEGA10:
				/* enable BACO as runpm mode if noretry=0 */
				if (!adev->gmc.noretry && !amdgpu_passthrough(adev))
					adev->pm.rpm_mode = AMDGPU_RUNPM_BACO;
				break;
			default:
				/* enable BACO as runpm mode on CI+ */
				if (!amdgpu_passthrough(adev))
					adev->pm.rpm_mode = AMDGPU_RUNPM_BACO;
				break;
			}

			if (adev->pm.rpm_mode == AMDGPU_RUNPM_BACO) {
				if (bamaco_support & MACO_SUPPORT) {
					adev->pm.rpm_mode = AMDGPU_RUNPM_BAMACO;
					dev_info(adev->dev, "Using BAMACO for runtime pm\n");
				} else {
					dev_info(adev->dev, "Using BACO for runtime pm\n");
				}
			}
		}
		break;
	case 0:
		dev_info(adev->dev, "runtime pm is manually disabled\n");
		break;
	default:
		break;
	}

no_runtime_pm:
	if (adev->pm.rpm_mode == AMDGPU_RUNPM_NONE)
		dev_info(adev->dev, "Runtime PM not available\n");
}
/**
 * amdgpu_device_supports_smart_shift - Is the device dGPU with
 * smart shift support
 *
 * @dev: drm_device pointer
 *
 * Returns true if the device is a dGPU with Smart Shift support,
 * otherwise returns false.
 */
bool amdgpu_device_supports_smart_shift(struct drm_device *dev)
{
	return (amdgpu_device_supports_boco(dev) &&
		amdgpu_acpi_is_power_shift_control_supported());
}

/*
 * VRAM access helper functions
 */

/**
 * amdgpu_device_mm_access - access vram by MM_INDEX/MM_DATA
 *
 * @adev: amdgpu_device pointer
 * @pos: offset of the buffer in vram
 * @buf: virtual address of the buffer in system memory
 * @size: read/write size, sizeof(@buf) must > @size
 * @write: true - write to vram, otherwise - read from vram
 */
void amdgpu_device_mm_access(struct amdgpu_device *adev, loff_t pos,
			     void *buf, size_t size, bool write)
{
	unsigned long flags;
	uint32_t hi = ~0, tmp = 0;
	uint32_t *data = buf;
	uint64_t last;
	int idx;

	if (!drm_dev_enter(adev_to_drm(adev), &idx))
		return;

	BUG_ON(!IS_ALIGNED(pos, 4) || !IS_ALIGNED(size, 4));

	spin_lock_irqsave(&adev->mmio_idx_lock, flags);
	for (last = pos + size; pos < last; pos += 4) {
		tmp = pos >> 31;

		WREG32_NO_KIQ(mmMM_INDEX, ((uint32_t)pos) | 0x80000000);
		if (tmp != hi) {
			WREG32_NO_KIQ(mmMM_INDEX_HI, tmp);
			hi = tmp;
		}
		if (write)
			WREG32_NO_KIQ(mmMM_DATA, *data++);
		else
			*data++ = RREG32_NO_KIQ(mmMM_DATA);
	}

	spin_unlock_irqrestore(&adev->mmio_idx_lock, flags);
	drm_dev_exit(idx);
}

/**
 * amdgpu_device_aper_access - access vram by vram aperture
 *
 * @adev: amdgpu_device pointer
 * @pos: offset of the buffer in vram
 * @buf: virtual address of the buffer in system memory
 * @size: read/write size, sizeof(@buf) must > @size
 * @write: true - write to vram, otherwise - read from vram
 *
 * The return value means how many bytes have been transferred.
 */
size_t amdgpu_device_aper_access(struct amdgpu_device *adev, loff_t pos,
				 void *buf, size_t size, bool write)
{
#ifdef CONFIG_64BIT
	void __iomem *addr;
	size_t count = 0;
	uint64_t last;

	if (!adev->mman.aper_base_kaddr)
		return 0;

	last = min(pos + size, adev->gmc.visible_vram_size);
	if (last > pos) {
		addr = adev->mman.aper_base_kaddr + pos;
		count = last - pos;

		if (write) {
			memcpy_toio(addr, buf, count);
			/* Make sure HDP write cache flush happens without any reordering
			 * after the system memory contents are sent over PCIe device
			 */
			mb();
			amdgpu_device_flush_hdp(adev, NULL);
		} else {
			amdgpu_device_invalidate_hdp(adev, NULL);
			/* Make sure HDP read cache is invalidated before issuing a read
			 * to the PCIe device
			 */
			mb();
			memcpy_fromio(buf, addr, count);
		}

	}

	return count;
#else
	return 0;
#endif
}

/**
 * amdgpu_device_vram_access - read/write a buffer in vram
 *
 * @adev: amdgpu_device pointer
 * @pos: offset of the buffer in vram
 * @buf: virtual address of the buffer in system memory
 * @size: read/write size, sizeof(@buf) must > @size
 * @write: true - write to vram, otherwise - read from vram
 */
void amdgpu_device_vram_access(struct amdgpu_device *adev, loff_t pos,
			       void *buf, size_t size, bool write)
{
	size_t count;

	/* try to using vram apreature to access vram first */
	count = amdgpu_device_aper_access(adev, pos, buf, size, write);
	size -= count;
	if (size) {
		/* using MM to access rest vram */
		pos += count;
		buf += count;
		amdgpu_device_mm_access(adev, pos, buf, size, write);
	}
}

/*
 * register access helper functions.
 */

/* Check if hw access should be skipped because of hotplug or device error */
bool amdgpu_device_skip_hw_access(struct amdgpu_device *adev)
{
	if (adev->no_hw_access)
		return true;

#ifdef CONFIG_LOCKDEP
	/*
	 * This is a bit complicated to understand, so worth a comment. What we assert
	 * here is that the GPU reset is not running on another thread in parallel.
	 *
	 * For this we trylock the read side of the reset semaphore, if that succeeds
	 * we know that the reset is not running in parallel.
	 *
	 * If the trylock fails we assert that we are either already holding the read
	 * side of the lock or are the reset thread itself and hold the write side of
	 * the lock.
	 */
	if (in_task()) {
		if (down_read_trylock(&adev->reset_domain->sem))
			up_read(&adev->reset_domain->sem);
		else
			lockdep_assert_held(&adev->reset_domain->sem);
	}
#endif
	return false;
}

/**
 * amdgpu_device_rreg - read a memory mapped IO or indirect register
 *
 * @adev: amdgpu_device pointer
 * @reg: dword aligned register offset
 * @acc_flags: access flags which require special behavior
 *
 * Returns the 32 bit value from the offset specified.
 */
uint32_t amdgpu_device_rreg(struct amdgpu_device *adev,
			    uint32_t reg, uint32_t acc_flags)
{
	uint32_t ret;

	if (amdgpu_device_skip_hw_access(adev))
		return 0;

	if ((reg * 4) < adev->rmmio_size) {
		if (!(acc_flags & AMDGPU_REGS_NO_KIQ) &&
		    amdgpu_sriov_runtime(adev) &&
		    down_read_trylock(&adev->reset_domain->sem)) {
			ret = amdgpu_kiq_rreg(adev, reg, 0);
			up_read(&adev->reset_domain->sem);
		} else {
			ret = readl(((void __iomem *)adev->rmmio) + (reg * 4));
		}
	} else {
		ret = adev->pcie_rreg(adev, reg * 4);
	}

	trace_amdgpu_device_rreg(adev->pdev->device, reg, ret);

	return ret;
}

/*
 * MMIO register read with bytes helper functions
 * @offset:bytes offset from MMIO start
 */

/**
 * amdgpu_mm_rreg8 - read a memory mapped IO register
 *
 * @adev: amdgpu_device pointer
 * @offset: byte aligned register offset
 *
 * Returns the 8 bit value from the offset specified.
 */
uint8_t amdgpu_mm_rreg8(struct amdgpu_device *adev, uint32_t offset)
{
	if (amdgpu_device_skip_hw_access(adev))
		return 0;

	if (offset < adev->rmmio_size)
		return (readb(adev->rmmio + offset));
	BUG();
}


/**
 * amdgpu_device_xcc_rreg - read a memory mapped IO or indirect register with specific XCC
 *
 * @adev: amdgpu_device pointer
 * @reg: dword aligned register offset
 * @acc_flags: access flags which require special behavior
 * @xcc_id: xcc accelerated compute core id
 *
 * Returns the 32 bit value from the offset specified.
 */
uint32_t amdgpu_device_xcc_rreg(struct amdgpu_device *adev,
				uint32_t reg, uint32_t acc_flags,
				uint32_t xcc_id)
{
	uint32_t ret, rlcg_flag;

	if (amdgpu_device_skip_hw_access(adev))
		return 0;

	if ((reg * 4) < adev->rmmio_size) {
		if (amdgpu_sriov_vf(adev) &&
		    !amdgpu_sriov_runtime(adev) &&
		    adev->gfx.rlc.rlcg_reg_access_supported &&
		    amdgpu_virt_get_rlcg_reg_access_flag(adev, acc_flags,
							 GC_HWIP, false,
							 &rlcg_flag)) {
			ret = amdgpu_virt_rlcg_reg_rw(adev, reg, 0, rlcg_flag, GET_INST(GC, xcc_id));
		} else if (!(acc_flags & AMDGPU_REGS_NO_KIQ) &&
		    amdgpu_sriov_runtime(adev) &&
		    down_read_trylock(&adev->reset_domain->sem)) {
			ret = amdgpu_kiq_rreg(adev, reg, xcc_id);
			up_read(&adev->reset_domain->sem);
		} else {
			ret = readl(((void __iomem *)adev->rmmio) + (reg * 4));
		}
	} else {
		ret = adev->pcie_rreg(adev, reg * 4);
	}

	return ret;
}

/*
 * MMIO register write with bytes helper functions
 * @offset:bytes offset from MMIO start
 * @value: the value want to be written to the register
 */

/**
 * amdgpu_mm_wreg8 - read a memory mapped IO register
 *
 * @adev: amdgpu_device pointer
 * @offset: byte aligned register offset
 * @value: 8 bit value to write
 *
 * Writes the value specified to the offset specified.
 */
void amdgpu_mm_wreg8(struct amdgpu_device *adev, uint32_t offset, uint8_t value)
{
	if (amdgpu_device_skip_hw_access(adev))
		return;

	if (offset < adev->rmmio_size)
		writeb(value, adev->rmmio + offset);
	else
		BUG();
}

/**
 * amdgpu_device_wreg - write to a memory mapped IO or indirect register
 *
 * @adev: amdgpu_device pointer
 * @reg: dword aligned register offset
 * @v: 32 bit value to write to the register
 * @acc_flags: access flags which require special behavior
 *
 * Writes the value specified to the offset specified.
 */
void amdgpu_device_wreg(struct amdgpu_device *adev,
			uint32_t reg, uint32_t v,
			uint32_t acc_flags)
{
	if (amdgpu_device_skip_hw_access(adev))
		return;

	if ((reg * 4) < adev->rmmio_size) {
		if (!(acc_flags & AMDGPU_REGS_NO_KIQ) &&
		    amdgpu_sriov_runtime(adev) &&
		    down_read_trylock(&adev->reset_domain->sem)) {
			amdgpu_kiq_wreg(adev, reg, v, 0);
			up_read(&adev->reset_domain->sem);
		} else {
			writel(v, ((void __iomem *)adev->rmmio) + (reg * 4));
		}
	} else {
		adev->pcie_wreg(adev, reg * 4, v);
	}

	trace_amdgpu_device_wreg(adev->pdev->device, reg, v);
}

/**
 * amdgpu_mm_wreg_mmio_rlc -  write register either with direct/indirect mmio or with RLC path if in range
 *
 * @adev: amdgpu_device pointer
 * @reg: mmio/rlc register
 * @v: value to write
 * @xcc_id: xcc accelerated compute core id
 *
 * this function is invoked only for the debugfs register access
 */
void amdgpu_mm_wreg_mmio_rlc(struct amdgpu_device *adev,
			     uint32_t reg, uint32_t v,
			     uint32_t xcc_id)
{
	if (amdgpu_device_skip_hw_access(adev))
		return;

	if (amdgpu_sriov_fullaccess(adev) &&
	    adev->gfx.rlc.funcs &&
	    adev->gfx.rlc.funcs->is_rlcg_access_range) {
		if (adev->gfx.rlc.funcs->is_rlcg_access_range(adev, reg))
			return amdgpu_sriov_wreg(adev, reg, v, 0, 0, xcc_id);
	} else if ((reg * 4) >= adev->rmmio_size) {
		adev->pcie_wreg(adev, reg * 4, v);
	} else {
		writel(v, ((void __iomem *)adev->rmmio) + (reg * 4));
	}
}

/**
 * amdgpu_device_xcc_wreg - write to a memory mapped IO or indirect register with specific XCC
 *
 * @adev: amdgpu_device pointer
 * @reg: dword aligned register offset
 * @v: 32 bit value to write to the register
 * @acc_flags: access flags which require special behavior
 * @xcc_id: xcc accelerated compute core id
 *
 * Writes the value specified to the offset specified.
 */
void amdgpu_device_xcc_wreg(struct amdgpu_device *adev,
			uint32_t reg, uint32_t v,
			uint32_t acc_flags, uint32_t xcc_id)
{
	uint32_t rlcg_flag;

	if (amdgpu_device_skip_hw_access(adev))
		return;

	if ((reg * 4) < adev->rmmio_size) {
		if (amdgpu_sriov_vf(adev) &&
		    !amdgpu_sriov_runtime(adev) &&
		    adev->gfx.rlc.rlcg_reg_access_supported &&
		    amdgpu_virt_get_rlcg_reg_access_flag(adev, acc_flags,
							 GC_HWIP, true,
							 &rlcg_flag)) {
			amdgpu_virt_rlcg_reg_rw(adev, reg, v, rlcg_flag, GET_INST(GC, xcc_id));
		} else if (!(acc_flags & AMDGPU_REGS_NO_KIQ) &&
		    amdgpu_sriov_runtime(adev) &&
		    down_read_trylock(&adev->reset_domain->sem)) {
			amdgpu_kiq_wreg(adev, reg, v, xcc_id);
			up_read(&adev->reset_domain->sem);
		} else {
			writel(v, ((void __iomem *)adev->rmmio) + (reg * 4));
		}
	} else {
		adev->pcie_wreg(adev, reg * 4, v);
	}
}

/**
 * amdgpu_device_indirect_rreg - read an indirect register
 *
 * @adev: amdgpu_device pointer
 * @reg_addr: indirect register address to read from
 *
 * Returns the value of indirect register @reg_addr
 */
u32 amdgpu_device_indirect_rreg(struct amdgpu_device *adev,
				u32 reg_addr)
{
	unsigned long flags, pcie_index, pcie_data;
	void __iomem *pcie_index_offset;
	void __iomem *pcie_data_offset;
	u32 r;

	pcie_index = adev->nbio.funcs->get_pcie_index_offset(adev);
	pcie_data = adev->nbio.funcs->get_pcie_data_offset(adev);

	spin_lock_irqsave(&adev->pcie_idx_lock, flags);
	pcie_index_offset = (void __iomem *)adev->rmmio + pcie_index * 4;
	pcie_data_offset = (void __iomem *)adev->rmmio + pcie_data * 4;

	writel(reg_addr, pcie_index_offset);
	readl(pcie_index_offset);
	r = readl(pcie_data_offset);
	spin_unlock_irqrestore(&adev->pcie_idx_lock, flags);

	return r;
}

u32 amdgpu_device_indirect_rreg_ext(struct amdgpu_device *adev,
				    u64 reg_addr)
{
	unsigned long flags, pcie_index, pcie_index_hi, pcie_data;
	u32 r;
	void __iomem *pcie_index_offset;
	void __iomem *pcie_index_hi_offset;
	void __iomem *pcie_data_offset;

	if (unlikely(!adev->nbio.funcs)) {
		pcie_index = AMDGPU_PCIE_INDEX_FALLBACK;
		pcie_data = AMDGPU_PCIE_DATA_FALLBACK;
	} else {
		pcie_index = adev->nbio.funcs->get_pcie_index_offset(adev);
		pcie_data = adev->nbio.funcs->get_pcie_data_offset(adev);
	}

	if (reg_addr >> 32) {
		if (unlikely(!adev->nbio.funcs))
			pcie_index_hi = AMDGPU_PCIE_INDEX_HI_FALLBACK;
		else
			pcie_index_hi = adev->nbio.funcs->get_pcie_index_hi_offset(adev);
	} else {
		pcie_index_hi = 0;
	}

	spin_lock_irqsave(&adev->pcie_idx_lock, flags);
	pcie_index_offset = (void __iomem *)adev->rmmio + pcie_index * 4;
	pcie_data_offset = (void __iomem *)adev->rmmio + pcie_data * 4;
	if (pcie_index_hi != 0)
		pcie_index_hi_offset = (void __iomem *)adev->rmmio +
				pcie_index_hi * 4;

	writel(reg_addr, pcie_index_offset);
	readl(pcie_index_offset);
	if (pcie_index_hi != 0) {
		writel((reg_addr >> 32) & 0xff, pcie_index_hi_offset);
		readl(pcie_index_hi_offset);
	}
	r = readl(pcie_data_offset);

	/* clear the high bits */
	if (pcie_index_hi != 0) {
		writel(0, pcie_index_hi_offset);
		readl(pcie_index_hi_offset);
	}

	spin_unlock_irqrestore(&adev->pcie_idx_lock, flags);

	return r;
}

/**
 * amdgpu_device_indirect_rreg64 - read a 64bits indirect register
 *
 * @adev: amdgpu_device pointer
 * @reg_addr: indirect register address to read from
 *
 * Returns the value of indirect register @reg_addr
 */
u64 amdgpu_device_indirect_rreg64(struct amdgpu_device *adev,
				  u32 reg_addr)
{
	unsigned long flags, pcie_index, pcie_data;
	void __iomem *pcie_index_offset;
	void __iomem *pcie_data_offset;
	u64 r;

	pcie_index = adev->nbio.funcs->get_pcie_index_offset(adev);
	pcie_data = adev->nbio.funcs->get_pcie_data_offset(adev);

	spin_lock_irqsave(&adev->pcie_idx_lock, flags);
	pcie_index_offset = (void __iomem *)adev->rmmio + pcie_index * 4;
	pcie_data_offset = (void __iomem *)adev->rmmio + pcie_data * 4;

	/* read low 32 bits */
	writel(reg_addr, pcie_index_offset);
	readl(pcie_index_offset);
	r = readl(pcie_data_offset);
	/* read high 32 bits */
	writel(reg_addr + 4, pcie_index_offset);
	readl(pcie_index_offset);
	r |= ((u64)readl(pcie_data_offset) << 32);
	spin_unlock_irqrestore(&adev->pcie_idx_lock, flags);

	return r;
}

u64 amdgpu_device_indirect_rreg64_ext(struct amdgpu_device *adev,
				  u64 reg_addr)
{
	unsigned long flags, pcie_index, pcie_data;
	unsigned long pcie_index_hi = 0;
	void __iomem *pcie_index_offset;
	void __iomem *pcie_index_hi_offset;
	void __iomem *pcie_data_offset;
	u64 r;

	pcie_index = adev->nbio.funcs->get_pcie_index_offset(adev);
	pcie_data = adev->nbio.funcs->get_pcie_data_offset(adev);
	if ((reg_addr >> 32) && (adev->nbio.funcs->get_pcie_index_hi_offset))
		pcie_index_hi = adev->nbio.funcs->get_pcie_index_hi_offset(adev);

	spin_lock_irqsave(&adev->pcie_idx_lock, flags);
	pcie_index_offset = (void __iomem *)adev->rmmio + pcie_index * 4;
	pcie_data_offset = (void __iomem *)adev->rmmio + pcie_data * 4;
	if (pcie_index_hi != 0)
		pcie_index_hi_offset = (void __iomem *)adev->rmmio +
			pcie_index_hi * 4;

	/* read low 32 bits */
	writel(reg_addr, pcie_index_offset);
	readl(pcie_index_offset);
	if (pcie_index_hi != 0) {
		writel((reg_addr >> 32) & 0xff, pcie_index_hi_offset);
		readl(pcie_index_hi_offset);
	}
	r = readl(pcie_data_offset);
	/* read high 32 bits */
	writel(reg_addr + 4, pcie_index_offset);
	readl(pcie_index_offset);
	if (pcie_index_hi != 0) {
		writel((reg_addr >> 32) & 0xff, pcie_index_hi_offset);
		readl(pcie_index_hi_offset);
	}
	r |= ((u64)readl(pcie_data_offset) << 32);

	/* clear the high bits */
	if (pcie_index_hi != 0) {
		writel(0, pcie_index_hi_offset);
		readl(pcie_index_hi_offset);
	}

	spin_unlock_irqrestore(&adev->pcie_idx_lock, flags);

	return r;
}

/**
 * amdgpu_device_indirect_wreg - write an indirect register address
 *
 * @adev: amdgpu_device pointer
 * @reg_addr: indirect register offset
 * @reg_data: indirect register data
 *
 */
void amdgpu_device_indirect_wreg(struct amdgpu_device *adev,
				 u32 reg_addr, u32 reg_data)
{
	unsigned long flags, pcie_index, pcie_data;
	void __iomem *pcie_index_offset;
	void __iomem *pcie_data_offset;

	pcie_index = adev->nbio.funcs->get_pcie_index_offset(adev);
	pcie_data = adev->nbio.funcs->get_pcie_data_offset(adev);

	spin_lock_irqsave(&adev->pcie_idx_lock, flags);
	pcie_index_offset = (void __iomem *)adev->rmmio + pcie_index * 4;
	pcie_data_offset = (void __iomem *)adev->rmmio + pcie_data * 4;

	writel(reg_addr, pcie_index_offset);
	readl(pcie_index_offset);
	writel(reg_data, pcie_data_offset);
	readl(pcie_data_offset);
	spin_unlock_irqrestore(&adev->pcie_idx_lock, flags);
}

void amdgpu_device_indirect_wreg_ext(struct amdgpu_device *adev,
				     u64 reg_addr, u32 reg_data)
{
	unsigned long flags, pcie_index, pcie_index_hi, pcie_data;
	void __iomem *pcie_index_offset;
	void __iomem *pcie_index_hi_offset;
	void __iomem *pcie_data_offset;

	pcie_index = adev->nbio.funcs->get_pcie_index_offset(adev);
	pcie_data = adev->nbio.funcs->get_pcie_data_offset(adev);
	if ((reg_addr >> 32) && (adev->nbio.funcs->get_pcie_index_hi_offset))
		pcie_index_hi = adev->nbio.funcs->get_pcie_index_hi_offset(adev);
	else
		pcie_index_hi = 0;

	spin_lock_irqsave(&adev->pcie_idx_lock, flags);
	pcie_index_offset = (void __iomem *)adev->rmmio + pcie_index * 4;
	pcie_data_offset = (void __iomem *)adev->rmmio + pcie_data * 4;
	if (pcie_index_hi != 0)
		pcie_index_hi_offset = (void __iomem *)adev->rmmio +
				pcie_index_hi * 4;

	writel(reg_addr, pcie_index_offset);
	readl(pcie_index_offset);
	if (pcie_index_hi != 0) {
		writel((reg_addr >> 32) & 0xff, pcie_index_hi_offset);
		readl(pcie_index_hi_offset);
	}
	writel(reg_data, pcie_data_offset);
	readl(pcie_data_offset);

	/* clear the high bits */
	if (pcie_index_hi != 0) {
		writel(0, pcie_index_hi_offset);
		readl(pcie_index_hi_offset);
	}

	spin_unlock_irqrestore(&adev->pcie_idx_lock, flags);
}

/**
 * amdgpu_device_indirect_wreg64 - write a 64bits indirect register address
 *
 * @adev: amdgpu_device pointer
 * @reg_addr: indirect register offset
 * @reg_data: indirect register data
 *
 */
void amdgpu_device_indirect_wreg64(struct amdgpu_device *adev,
				   u32 reg_addr, u64 reg_data)
{
	unsigned long flags, pcie_index, pcie_data;
	void __iomem *pcie_index_offset;
	void __iomem *pcie_data_offset;

	pcie_index = adev->nbio.funcs->get_pcie_index_offset(adev);
	pcie_data = adev->nbio.funcs->get_pcie_data_offset(adev);

	spin_lock_irqsave(&adev->pcie_idx_lock, flags);
	pcie_index_offset = (void __iomem *)adev->rmmio + pcie_index * 4;
	pcie_data_offset = (void __iomem *)adev->rmmio + pcie_data * 4;

	/* write low 32 bits */
	writel(reg_addr, pcie_index_offset);
	readl(pcie_index_offset);
	writel((u32)(reg_data & 0xffffffffULL), pcie_data_offset);
	readl(pcie_data_offset);
	/* write high 32 bits */
	writel(reg_addr + 4, pcie_index_offset);
	readl(pcie_index_offset);
	writel((u32)(reg_data >> 32), pcie_data_offset);
	readl(pcie_data_offset);
	spin_unlock_irqrestore(&adev->pcie_idx_lock, flags);
}

void amdgpu_device_indirect_wreg64_ext(struct amdgpu_device *adev,
				   u64 reg_addr, u64 reg_data)
{
	unsigned long flags, pcie_index, pcie_data;
	unsigned long pcie_index_hi = 0;
	void __iomem *pcie_index_offset;
	void __iomem *pcie_index_hi_offset;
	void __iomem *pcie_data_offset;

	pcie_index = adev->nbio.funcs->get_pcie_index_offset(adev);
	pcie_data = adev->nbio.funcs->get_pcie_data_offset(adev);
	if ((reg_addr >> 32) && (adev->nbio.funcs->get_pcie_index_hi_offset))
		pcie_index_hi = adev->nbio.funcs->get_pcie_index_hi_offset(adev);

	spin_lock_irqsave(&adev->pcie_idx_lock, flags);
	pcie_index_offset = (void __iomem *)adev->rmmio + pcie_index * 4;
	pcie_data_offset = (void __iomem *)adev->rmmio + pcie_data * 4;
	if (pcie_index_hi != 0)
		pcie_index_hi_offset = (void __iomem *)adev->rmmio +
				pcie_index_hi * 4;

	/* write low 32 bits */
	writel(reg_addr, pcie_index_offset);
	readl(pcie_index_offset);
	if (pcie_index_hi != 0) {
		writel((reg_addr >> 32) & 0xff, pcie_index_hi_offset);
		readl(pcie_index_hi_offset);
	}
	writel((u32)(reg_data & 0xffffffffULL), pcie_data_offset);
	readl(pcie_data_offset);
	/* write high 32 bits */
	writel(reg_addr + 4, pcie_index_offset);
	readl(pcie_index_offset);
	if (pcie_index_hi != 0) {
		writel((reg_addr >> 32) & 0xff, pcie_index_hi_offset);
		readl(pcie_index_hi_offset);
	}
	writel((u32)(reg_data >> 32), pcie_data_offset);
	readl(pcie_data_offset);

	/* clear the high bits */
	if (pcie_index_hi != 0) {
		writel(0, pcie_index_hi_offset);
		readl(pcie_index_hi_offset);
	}

	spin_unlock_irqrestore(&adev->pcie_idx_lock, flags);
}

/**
 * amdgpu_device_get_rev_id - query device rev_id
 *
 * @adev: amdgpu_device pointer
 *
 * Return device rev_id
 */
u32 amdgpu_device_get_rev_id(struct amdgpu_device *adev)
{
	return adev->nbio.funcs->get_rev_id(adev);
}

/**
 * amdgpu_invalid_rreg - dummy reg read function
 *
 * @adev: amdgpu_device pointer
 * @reg: offset of register
 *
 * Dummy register read function.  Used for register blocks
 * that certain asics don't have (all asics).
 * Returns the value in the register.
 */
static uint32_t amdgpu_invalid_rreg(struct amdgpu_device *adev, uint32_t reg)
{
	DRM_ERROR("Invalid callback to read register 0x%04X\n", reg);
	BUG();
	return 0;
}

static uint32_t amdgpu_invalid_rreg_ext(struct amdgpu_device *adev, uint64_t reg)
{
	DRM_ERROR("Invalid callback to read register 0x%llX\n", reg);
	BUG();
	return 0;
}

/**
 * amdgpu_invalid_wreg - dummy reg write function
 *
 * @adev: amdgpu_device pointer
 * @reg: offset of register
 * @v: value to write to the register
 *
 * Dummy register read function.  Used for register blocks
 * that certain asics don't have (all asics).
 */
static void amdgpu_invalid_wreg(struct amdgpu_device *adev, uint32_t reg, uint32_t v)
{
	DRM_ERROR("Invalid callback to write register 0x%04X with 0x%08X\n",
		  reg, v);
	BUG();
}

static void amdgpu_invalid_wreg_ext(struct amdgpu_device *adev, uint64_t reg, uint32_t v)
{
	DRM_ERROR("Invalid callback to write register 0x%llX with 0x%08X\n",
		  reg, v);
	BUG();
}

/**
 * amdgpu_invalid_rreg64 - dummy 64 bit reg read function
 *
 * @adev: amdgpu_device pointer
 * @reg: offset of register
 *
 * Dummy register read function.  Used for register blocks
 * that certain asics don't have (all asics).
 * Returns the value in the register.
 */
static uint64_t amdgpu_invalid_rreg64(struct amdgpu_device *adev, uint32_t reg)
{
	DRM_ERROR("Invalid callback to read 64 bit register 0x%04X\n", reg);
	BUG();
	return 0;
}

static uint64_t amdgpu_invalid_rreg64_ext(struct amdgpu_device *adev, uint64_t reg)
{
	DRM_ERROR("Invalid callback to read register 0x%llX\n", reg);
	BUG();
	return 0;
}

/**
 * amdgpu_invalid_wreg64 - dummy reg write function
 *
 * @adev: amdgpu_device pointer
 * @reg: offset of register
 * @v: value to write to the register
 *
 * Dummy register read function.  Used for register blocks
 * that certain asics don't have (all asics).
 */
static void amdgpu_invalid_wreg64(struct amdgpu_device *adev, uint32_t reg, uint64_t v)
{
	DRM_ERROR("Invalid callback to write 64 bit register 0x%04X with 0x%08llX\n",
		  reg, v);
	BUG();
}

static void amdgpu_invalid_wreg64_ext(struct amdgpu_device *adev, uint64_t reg, uint64_t v)
{
	DRM_ERROR("Invalid callback to write 64 bit register 0x%llX with 0x%08llX\n",
		  reg, v);
	BUG();
}

/**
 * amdgpu_block_invalid_rreg - dummy reg read function
 *
 * @adev: amdgpu_device pointer
 * @block: offset of instance
 * @reg: offset of register
 *
 * Dummy register read function.  Used for register blocks
 * that certain asics don't have (all asics).
 * Returns the value in the register.
 */
static uint32_t amdgpu_block_invalid_rreg(struct amdgpu_device *adev,
					  uint32_t block, uint32_t reg)
{
	DRM_ERROR("Invalid callback to read register 0x%04X in block 0x%04X\n",
		  reg, block);
	BUG();
	return 0;
}

/**
 * amdgpu_block_invalid_wreg - dummy reg write function
 *
 * @adev: amdgpu_device pointer
 * @block: offset of instance
 * @reg: offset of register
 * @v: value to write to the register
 *
 * Dummy register read function.  Used for register blocks
 * that certain asics don't have (all asics).
 */
static void amdgpu_block_invalid_wreg(struct amdgpu_device *adev,
				      uint32_t block,
				      uint32_t reg, uint32_t v)
{
	DRM_ERROR("Invalid block callback to write register 0x%04X in block 0x%04X with 0x%08X\n",
		  reg, block, v);
	BUG();
}

static uint32_t amdgpu_device_get_vbios_flags(struct amdgpu_device *adev)
{
	if (hweight32(adev->aid_mask) && (adev->flags & AMD_IS_APU))
		return AMDGPU_VBIOS_SKIP;

	if (hweight32(adev->aid_mask) && amdgpu_passthrough(adev))
		return AMDGPU_VBIOS_OPTIONAL;

	return 0;
}

/**
 * amdgpu_device_asic_init - Wrapper for atom asic_init
 *
 * @adev: amdgpu_device pointer
 *
 * Does any asic specific work and then calls atom asic init.
 */
static int amdgpu_device_asic_init(struct amdgpu_device *adev)
{
	uint32_t flags;
	bool optional;
	int ret;

	amdgpu_asic_pre_asic_init(adev);
	flags = amdgpu_device_get_vbios_flags(adev);
	optional = !!(flags & (AMDGPU_VBIOS_OPTIONAL | AMDGPU_VBIOS_SKIP));

	if (amdgpu_ip_version(adev, GC_HWIP, 0) == IP_VERSION(9, 4, 3) ||
	    amdgpu_ip_version(adev, GC_HWIP, 0) == IP_VERSION(9, 4, 4) ||
	    amdgpu_ip_version(adev, GC_HWIP, 0) == IP_VERSION(9, 5, 0) ||
	    amdgpu_ip_version(adev, GC_HWIP, 0) >= IP_VERSION(11, 0, 0)) {
		amdgpu_psp_wait_for_bootloader(adev);
		if (optional && !adev->bios)
			return 0;

		ret = amdgpu_atomfirmware_asic_init(adev, true);
		return ret;
	} else {
		if (optional && !adev->bios)
			return 0;

		return amdgpu_atom_asic_init(adev->mode_info.atom_context);
	}

	return 0;
}

/**
 * amdgpu_device_mem_scratch_init - allocate the VRAM scratch page
 *
 * @adev: amdgpu_device pointer
 *
 * Allocates a scratch page of VRAM for use by various things in the
 * driver.
 */
static int amdgpu_device_mem_scratch_init(struct amdgpu_device *adev)
{
	return amdgpu_bo_create_kernel(adev, AMDGPU_GPU_PAGE_SIZE, PAGE_SIZE,
				       AMDGPU_GEM_DOMAIN_VRAM |
				       AMDGPU_GEM_DOMAIN_GTT,
				       &adev->mem_scratch.robj,
				       &adev->mem_scratch.gpu_addr,
				       (void **)&adev->mem_scratch.ptr);
}

/**
 * amdgpu_device_mem_scratch_fini - Free the VRAM scratch page
 *
 * @adev: amdgpu_device pointer
 *
 * Frees the VRAM scratch page.
 */
static void amdgpu_device_mem_scratch_fini(struct amdgpu_device *adev)
{
	amdgpu_bo_free_kernel(&adev->mem_scratch.robj, NULL, NULL);
}

/**
 * amdgpu_device_program_register_sequence - program an array of registers.
 *
 * @adev: amdgpu_device pointer
 * @registers: pointer to the register array
 * @array_size: size of the register array
 *
 * Programs an array or registers with and or masks.
 * This is a helper for setting golden registers.
 */
void amdgpu_device_program_register_sequence(struct amdgpu_device *adev,
					     const u32 *registers,
					     const u32 array_size)
{
	u32 tmp, reg, and_mask, or_mask;
	int i;

	if (array_size % 3)
		return;

	for (i = 0; i < array_size; i += 3) {
		reg = registers[i + 0];
		and_mask = registers[i + 1];
		or_mask = registers[i + 2];

		if (and_mask == 0xffffffff) {
			tmp = or_mask;
		} else {
			tmp = RREG32(reg);
			tmp &= ~and_mask;
			if (adev->family >= AMDGPU_FAMILY_AI)
				tmp |= (or_mask & and_mask);
			else
				tmp |= or_mask;
		}
		WREG32(reg, tmp);
	}
}

/**
 * amdgpu_device_pci_config_reset - reset the GPU
 *
 * @adev: amdgpu_device pointer
 *
 * Resets the GPU using the pci config reset sequence.
 * Only applicable to asics prior to vega10.
 */
void amdgpu_device_pci_config_reset(struct amdgpu_device *adev)
{
	pci_write_config_dword(adev->pdev, 0x7c, AMDGPU_ASIC_RESET_DATA);
}

/**
 * amdgpu_device_pci_reset - reset the GPU using generic PCI means
 *
 * @adev: amdgpu_device pointer
 *
 * Resets the GPU using generic pci reset interfaces (FLR, SBR, etc.).
 */
int amdgpu_device_pci_reset(struct amdgpu_device *adev)
{
	return pci_reset_function(adev->pdev);
}

/*
 * amdgpu_device_wb_*()
 * Writeback is the method by which the GPU updates special pages in memory
 * with the status of certain GPU events (fences, ring pointers,etc.).
 */

/**
 * amdgpu_device_wb_fini - Disable Writeback and free memory
 *
 * @adev: amdgpu_device pointer
 *
 * Disables Writeback and frees the Writeback memory (all asics).
 * Used at driver shutdown.
 */
static void amdgpu_device_wb_fini(struct amdgpu_device *adev)
{
	if (adev->wb.wb_obj) {
		amdgpu_bo_free_kernel(&adev->wb.wb_obj,
				      &adev->wb.gpu_addr,
				      (void **)&adev->wb.wb);
		adev->wb.wb_obj = NULL;
	}
}

/**
 * amdgpu_device_wb_init - Init Writeback driver info and allocate memory
 *
 * @adev: amdgpu_device pointer
 *
 * Initializes writeback and allocates writeback memory (all asics).
 * Used at driver startup.
 * Returns 0 on success or an -error on failure.
 */
static int amdgpu_device_wb_init(struct amdgpu_device *adev)
{
	int r;

	if (adev->wb.wb_obj == NULL) {
		/* AMDGPU_MAX_WB * sizeof(uint32_t) * 8 = AMDGPU_MAX_WB 256bit slots */
		r = amdgpu_bo_create_kernel(adev, AMDGPU_MAX_WB * sizeof(uint32_t) * 8,
					    PAGE_SIZE, AMDGPU_GEM_DOMAIN_GTT,
					    &adev->wb.wb_obj, &adev->wb.gpu_addr,
					    (void **)&adev->wb.wb);
		if (r) {
			dev_warn(adev->dev, "(%d) create WB bo failed\n", r);
			return r;
		}

		adev->wb.num_wb = AMDGPU_MAX_WB;
		memset(&adev->wb.used, 0, sizeof(adev->wb.used));

		/* clear wb memory */
		memset((char *)adev->wb.wb, 0, AMDGPU_MAX_WB * sizeof(uint32_t) * 8);
	}

	return 0;
}

/**
 * amdgpu_device_wb_get - Allocate a wb entry
 *
 * @adev: amdgpu_device pointer
 * @wb: wb index
 *
 * Allocate a wb slot for use by the driver (all asics).
 * Returns 0 on success or -EINVAL on failure.
 */
int amdgpu_device_wb_get(struct amdgpu_device *adev, u32 *wb)
{
	unsigned long flags, offset;

	spin_lock_irqsave(&adev->wb.lock, flags);
	offset = find_first_zero_bit(adev->wb.used, adev->wb.num_wb);
	if (offset < adev->wb.num_wb) {
		__set_bit(offset, adev->wb.used);
		spin_unlock_irqrestore(&adev->wb.lock, flags);
		*wb = offset << 3; /* convert to dw offset */
		return 0;
	} else {
		spin_unlock_irqrestore(&adev->wb.lock, flags);
		return -EINVAL;
	}
}

/**
 * amdgpu_device_wb_free - Free a wb entry
 *
 * @adev: amdgpu_device pointer
 * @wb: wb index
 *
 * Free a wb slot allocated for use by the driver (all asics)
 */
void amdgpu_device_wb_free(struct amdgpu_device *adev, u32 wb)
{
	unsigned long flags;

	wb >>= 3;
	spin_lock_irqsave(&adev->wb.lock, flags);
	if (wb < adev->wb.num_wb)
		__clear_bit(wb, adev->wb.used);
	spin_unlock_irqrestore(&adev->wb.lock, flags);
}

/**
 * amdgpu_device_resize_fb_bar - try to resize FB BAR
 *
 * @adev: amdgpu_device pointer
 *
 * Try to resize FB BAR to make all VRAM CPU accessible. We try very hard not
 * to fail, but if any of the BARs is not accessible after the size we abort
 * driver loading by returning -ENODEV.
 */
int amdgpu_device_resize_fb_bar(struct amdgpu_device *adev)
{
	int rbar_size = pci_rebar_bytes_to_size(adev->gmc.real_vram_size);
	struct pci_bus *root;
	struct resource *res;
	unsigned int i;
	u16 cmd;
	int r;

	if (!IS_ENABLED(CONFIG_PHYS_ADDR_T_64BIT))
		return 0;

	/* Bypass for VF */
	if (amdgpu_sriov_vf(adev))
		return 0;

	if (!amdgpu_rebar)
		return 0;

	/* resizing on Dell G5 SE platforms causes problems with runtime pm */
	if ((amdgpu_runtime_pm != 0) &&
	    adev->pdev->vendor == PCI_VENDOR_ID_ATI &&
	    adev->pdev->device == 0x731f &&
	    adev->pdev->subsystem_vendor == PCI_VENDOR_ID_DELL)
		return 0;

	/* PCI_EXT_CAP_ID_VNDR extended capability is located at 0x100 */
	if (!pci_find_ext_capability(adev->pdev, PCI_EXT_CAP_ID_VNDR))
		DRM_WARN("System can't access extended configuration space, please check!!\n");

	/* skip if the bios has already enabled large BAR */
	if (adev->gmc.real_vram_size &&
	    (pci_resource_len(adev->pdev, 0) >= adev->gmc.real_vram_size))
		return 0;

	/* Check if the root BUS has 64bit memory resources */
	root = adev->pdev->bus;
	while (root->parent)
		root = root->parent;

	pci_bus_for_each_resource(root, res, i) {
		if (res && res->flags & (IORESOURCE_MEM | IORESOURCE_MEM_64) &&
		    res->start > 0x100000000ull)
			break;
	}

	/* Trying to resize is pointless without a root hub window above 4GB */
	if (!res)
		return 0;

	/* Limit the BAR size to what is available */
	rbar_size = min(fls(pci_rebar_get_possible_sizes(adev->pdev, 0)) - 1,
			rbar_size);

	/* Disable memory decoding while we change the BAR addresses and size */
	pci_read_config_word(adev->pdev, PCI_COMMAND, &cmd);
	pci_write_config_word(adev->pdev, PCI_COMMAND,
			      cmd & ~PCI_COMMAND_MEMORY);

	/* Free the VRAM and doorbell BAR, we most likely need to move both. */
	amdgpu_doorbell_fini(adev);
	if (adev->asic_type >= CHIP_BONAIRE)
		pci_release_resource(adev->pdev, 2);

	pci_release_resource(adev->pdev, 0);

	r = pci_resize_resource(adev->pdev, 0, rbar_size);
	if (r == -ENOSPC)
		DRM_INFO("Not enough PCI address space for a large BAR.");
	else if (r && r != -ENOTSUPP)
		DRM_ERROR("Problem resizing BAR0 (%d).", r);

	pci_assign_unassigned_bus_resources(adev->pdev->bus);

	/* When the doorbell or fb BAR isn't available we have no chance of
	 * using the device.
	 */
	r = amdgpu_doorbell_init(adev);
	if (r || (pci_resource_flags(adev->pdev, 0) & IORESOURCE_UNSET))
		return -ENODEV;

	pci_write_config_word(adev->pdev, PCI_COMMAND, cmd);

	return 0;
}

/*
 * GPU helpers function.
 */
/**
 * amdgpu_device_need_post - check if the hw need post or not
 *
 * @adev: amdgpu_device pointer
 *
 * Check if the asic has been initialized (all asics) at driver startup
 * or post is needed if  hw reset is performed.
 * Returns true if need or false if not.
 */
bool amdgpu_device_need_post(struct amdgpu_device *adev)
{
	uint32_t reg, flags;

	if (amdgpu_sriov_vf(adev))
		return false;

	flags = amdgpu_device_get_vbios_flags(adev);
	if (flags & AMDGPU_VBIOS_SKIP)
		return false;
	if ((flags & AMDGPU_VBIOS_OPTIONAL) && !adev->bios)
		return false;

	if (amdgpu_passthrough(adev)) {
		/* for FIJI: In whole GPU pass-through virtualization case, after VM reboot
		 * some old smc fw still need driver do vPost otherwise gpu hang, while
		 * those smc fw version above 22.15 doesn't have this flaw, so we force
		 * vpost executed for smc version below 22.15
		 */
		if (adev->asic_type == CHIP_FIJI) {
			int err;
			uint32_t fw_ver;

			err = request_firmware(&adev->pm.fw, "amdgpu/fiji_smc.bin", adev->dev);
			/* force vPost if error occurred */
			if (err)
				return true;

			fw_ver = *((uint32_t *)adev->pm.fw->data + 69);
			release_firmware(adev->pm.fw);
			if (fw_ver < 0x00160e00)
				return true;
		}
	}

	/* Don't post if we need to reset whole hive on init */
	if (adev->init_lvl->level == AMDGPU_INIT_LEVEL_MINIMAL_XGMI)
		return false;

	if (adev->has_hw_reset) {
		adev->has_hw_reset = false;
		return true;
	}

	/* bios scratch used on CIK+ */
	if (adev->asic_type >= CHIP_BONAIRE)
		return amdgpu_atombios_scratch_need_asic_init(adev);

	/* check MEM_SIZE for older asics */
	reg = amdgpu_asic_get_config_memsize(adev);

	if ((reg != 0) && (reg != 0xffffffff))
		return false;

	return true;
}

/*
 * Check whether seamless boot is supported.
 *
 * So far we only support seamless boot on DCE 3.0 or later.
 * If users report that it works on older ASICS as well, we may
 * loosen this.
 */
bool amdgpu_device_seamless_boot_supported(struct amdgpu_device *adev)
{
	switch (amdgpu_seamless) {
	case -1:
		break;
	case 1:
		return true;
	case 0:
		return false;
	default:
		DRM_ERROR("Invalid value for amdgpu.seamless: %d\n",
			  amdgpu_seamless);
		return false;
	}

	if (!(adev->flags & AMD_IS_APU))
		return false;

	if (adev->mman.keep_stolen_vga_memory)
		return false;

	return amdgpu_ip_version(adev, DCE_HWIP, 0) >= IP_VERSION(3, 0, 0);
}

/*
 * Intel hosts such as Rocket Lake, Alder Lake, Raptor Lake and Sapphire Rapids
 * don't support dynamic speed switching. Until we have confirmation from Intel
 * that a specific host supports it, it's safer that we keep it disabled for all.
 *
 * https://edc.intel.com/content/www/us/en/design/products/platforms/details/raptor-lake-s/13th-generation-core-processors-datasheet-volume-1-of-2/005/pci-express-support/
 * https://gitlab.freedesktop.org/drm/amd/-/issues/2663
 */
static bool amdgpu_device_pcie_dynamic_switching_supported(struct amdgpu_device *adev)
{
#if IS_ENABLED(CONFIG_X86)
	struct cpuinfo_x86 *c = &cpu_data(0);

	/* eGPU change speeds based on USB4 fabric conditions */
	if (dev_is_removable(adev->dev))
		return true;

	if (c->x86_vendor == X86_VENDOR_INTEL)
		return false;
#endif
	return true;
}

static bool amdgpu_device_aspm_support_quirk(struct amdgpu_device *adev)
{
#if IS_ENABLED(CONFIG_X86)
	struct cpuinfo_x86 *c = &cpu_data(0);

	if (!(amdgpu_ip_version(adev, GC_HWIP, 0) == IP_VERSION(12, 0, 0) ||
		  amdgpu_ip_version(adev, GC_HWIP, 0) == IP_VERSION(12, 0, 1)))
		return false;

	if (c->x86 == 6 &&
		adev->pm.pcie_gen_mask & CAIL_PCIE_LINK_SPEED_SUPPORT_GEN5) {
		switch (c->x86_model) {
		case VFM_MODEL(INTEL_ALDERLAKE):
		case VFM_MODEL(INTEL_ALDERLAKE_L):
		case VFM_MODEL(INTEL_RAPTORLAKE):
		case VFM_MODEL(INTEL_RAPTORLAKE_P):
		case VFM_MODEL(INTEL_RAPTORLAKE_S):
			return true;
		default:
			return false;
		}
	} else {
		return false;
	}
#else
	return false;
#endif
}

/**
 * amdgpu_device_should_use_aspm - check if the device should program ASPM
 *
 * @adev: amdgpu_device pointer
 *
 * Confirm whether the module parameter and pcie bridge agree that ASPM should
 * be set for this device.
 *
 * Returns true if it should be used or false if not.
 */
bool amdgpu_device_should_use_aspm(struct amdgpu_device *adev)
{
	switch (amdgpu_aspm) {
	case -1:
		break;
	case 0:
		return false;
	case 1:
		return true;
	default:
		return false;
	}
	if (adev->flags & AMD_IS_APU)
		return false;
	if (amdgpu_device_aspm_support_quirk(adev))
		return false;
	return pcie_aspm_enabled(adev->pdev);
}

/* if we get transitioned to only one device, take VGA back */
/**
 * amdgpu_device_vga_set_decode - enable/disable vga decode
 *
 * @pdev: PCI device pointer
 * @state: enable/disable vga decode
 *
 * Enable/disable vga decode (all asics).
 * Returns VGA resource flags.
 */
static unsigned int amdgpu_device_vga_set_decode(struct pci_dev *pdev,
		bool state)
{
	struct amdgpu_device *adev = drm_to_adev(pci_get_drvdata(pdev));

	amdgpu_asic_set_vga_state(adev, state);
	if (state)
		return VGA_RSRC_LEGACY_IO | VGA_RSRC_LEGACY_MEM |
		       VGA_RSRC_NORMAL_IO | VGA_RSRC_NORMAL_MEM;
	else
		return VGA_RSRC_NORMAL_IO | VGA_RSRC_NORMAL_MEM;
}

/**
 * amdgpu_device_check_block_size - validate the vm block size
 *
 * @adev: amdgpu_device pointer
 *
 * Validates the vm block size specified via module parameter.
 * The vm block size defines number of bits in page table versus page directory,
 * a page is 4KB so we have 12 bits offset, minimum 9 bits in the
 * page table and the remaining bits are in the page directory.
 */
static void amdgpu_device_check_block_size(struct amdgpu_device *adev)
{
	/* defines number of bits in page table versus page directory,
	 * a page is 4KB so we have 12 bits offset, minimum 9 bits in the
	 * page table and the remaining bits are in the page directory
	 */
	if (amdgpu_vm_block_size == -1)
		return;

	if (amdgpu_vm_block_size < 9) {
		dev_warn(adev->dev, "VM page table size (%d) too small\n",
			 amdgpu_vm_block_size);
		amdgpu_vm_block_size = -1;
	}
}

/**
 * amdgpu_device_check_vm_size - validate the vm size
 *
 * @adev: amdgpu_device pointer
 *
 * Validates the vm size in GB specified via module parameter.
 * The VM size is the size of the GPU virtual memory space in GB.
 */
static void amdgpu_device_check_vm_size(struct amdgpu_device *adev)
{
	/* no need to check the default value */
	if (amdgpu_vm_size == -1)
		return;

	if (amdgpu_vm_size < 1) {
		dev_warn(adev->dev, "VM size (%d) too small, min is 1GB\n",