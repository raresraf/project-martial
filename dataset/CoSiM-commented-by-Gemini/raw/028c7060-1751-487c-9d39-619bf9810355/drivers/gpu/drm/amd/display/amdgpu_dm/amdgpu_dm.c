/**
 * @file amdgpu_dm.c
 * @brief AMD Display Manager (DM) for AMDGPU DRM driver.
 *
 * This file implements the interface between the Linux DRM (Direct Rendering Manager)
 * subsystem and AMD's internal Display Core (DC) library. It translates DRM display
 * requests into DC operations and vice-versa, managing various display functionalities
 * such as mode setting, hotplug detection (HPD), backlight control, audio, and
 * interaction with the Display Management Unit (DMUB) firmware.
 *
 * Domain: Kernel-mode Display Driver (DRM/KMS)
 *
 * Key Components:
 * - `amdgpu_dm_init`, `amdgpu_dm_fini`: Basic initialization and finalization.
 * - `amdgpu_dm_initialize_drm_device`, `amdgpu_dm_destroy_drm_device`: DRM device setup.
 * - HPD/IRQ handlers: Manage hotplug events and display-related interrupts.
 * - DMUB (Display Microcontroller Unit Bypass) integration: Handles firmware communication.
 * - Power management: Clock gating, power gating, stutter mode control.
 * - Audio component: Integrates with the kernel audio subsystem.
 *
 * Optimization Techniques:
 * - DMUB firmware for offloading display management tasks to a dedicated microcontroller.
 * - Stutter mode control for power saving.
 *
 * Architectural Intent:
 * - Provide a robust, performant, and feature-rich display driver for AMD GPUs
 *   within the Linux kernel.
 * - Decouple DRM-specific logic from hardware-specific display control logic (DC library).
 */

/* The caprices of the preprocessor require that this be declared right here */
#define CREATE_TRACE_POINTS

#include "dm_services_types.h"
#include "dc.h"
#include "link_enc_cfg.h"
#include "dc/inc/core_types.h"
#include "dal_asic_id.h"
#include "dmub/dmub_srv.h"
#include "dc/inc/hw/dmcu.h"
#include "dc/inc/hw/abm.h"
#include "dc/dc_dmub_srv.h"
#include "dc/dc_edid_parser.h"
#include "dc/dc_stat.h"
#include "dc/dc_state.h"
#include "amdgpu_dm_trace.h"
#include "dpcd_defs.h"
#include "link/protocols/link_dpcd.h"
#include "link_service_types.h"
#include "link/protocols/link_dp_capability.h"
#include "link/protocols/link_ddc.h"

#include "vid.h"
#include "amdgpu.h"
#include "amdgpu_display.h"
#include "amdgpu_ucode.h"
#include "atom.h"
#include "amdgpu_dm.h"
#include "amdgpu_dm_plane.h"
#include "amdgpu_dm_crtc.h"
#include "amdgpu_dm_hdcp.h"
#include <drm/display/drm_hdcp_helper.h>
#include "amdgpu_dm_wb.h"
#include "amdgpu_pm.h"
#include "amdgpu_atombios.h"

#include "amd_shared.h"
#include "amdgpu_dm_irq.h"
#include "dm_helpers.h"
#include "amdgpu_dm_mst_types.h"
#if defined(CONFIG_DEBUG_FS)
#include "amdgpu_dm_debugfs.h"
#endif
#include "amdgpu_dm_psr.h"
#include "amdgpu_dm_replay.h"

#include "ivsrcid/ivsrcid_vislands30.h"

#include <linux/backlight.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/types.h>
#include <linux/pm_runtime.h>
#include <linux/pci.h>
#include <linux/power_supply.h>
#include <linux/firmware.h>
#include <linux/component.h>
#include <linux/sort.h>

#include <drm/display/drm_dp_mst_helper.h>
#include <drm/display/drm_hdmi_helper.h>
#include <drm/drm_atomic.h>
#include <drm/drm_atomic_uapi.h>
#include <drm/drm_atomic_helper.h>
#include <drm/drm_blend.h>
#include <drm/drm_fixed.h>
#include <drm/drm_fourcc.h>
#include <drm/drm_edid.h>
#include <drm/drm_eld.h>
#include <drm/drm_utils.h>
#include <drm/drm_vblank.h>
#include <drm/drm_audio_component.h>
#include <drm/drm_gem_atomic_helper.h>

#include <media/cec-notifier.h>
#include <acpi/video.h>

#include "ivsrcid/dcn/irqsrcs_dcn_1_0.h"

#include "dcn/dcn_1_0_offset.h"
#include "dcn/dcn_1_0_sh_mask.h"
#include "soc15_hw_ip.h"
#include "soc15_common.h"
#include "vega10_ip_offset.h"

#include "gc/gc_11_0_0_offset.h"
#include "gc/gc_11_0_0_sh_mask.h"

#include "modules/inc/mod_freesync.h"
#include "modules/power/power_helpers.h"

// Functional Utility: Statically asserts that two constants are equal, ensuring API compatibility.
static_assert(AMDGPU_DMUB_NOTIFICATION_MAX == DMUB_NOTIFICATION_MAX, "AMDGPU_DMUB_NOTIFICATION_MAX mismatch");

// Functional Utility: Defines firmware paths for various DMUB versions and associates them with the module.
#define FIRMWARE_RENOIR_DMUB "amdgpu/renoir_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_RENOIR_DMUB);
#define FIRMWARE_SIENNA_CICHLID_DMUB "amdgpu/sienna_cichlid_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_SIENNA_CICHLID_DMUB);
#define FIRMWARE_NAVY_FLOUNDER_DMUB "amdgpu/navy_flounder_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_NAVY_FLOUNDER_DMUB);
#define FIRMWARE_GREEN_SARDINE_DMUB "amdgpu/green_sardine_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_GREEN_SARDINE_DMUB);
#define FIRMWARE_VANGOGH_DMUB "amdgpu/vangogh_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_VANGOGH_DMUB);
#define FIRMWARE_DIMGREY_CAVEFISH_DMUB "amdgpu/dimgrey_cavefish_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_DIMGREY_CAVEFISH_DMUB);
#define FIRMWARE_BEIGE_GOBY_DMUB "amdgpu/beige_goby_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_BEIGE_GOBY_DMUB);
#define FIRMWARE_YELLOW_CARP_DMUB "amdgpu/yellow_carp_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_YELLOW_CARP_DMUB);
#define FIRMWARE_DCN_314_DMUB "amdgpu/dcn_3_1_4_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_DCN_314_DMUB);
#define FIRMWARE_DCN_315_DMUB "amdgpu/dcn_3_1_5_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_DCN_315_DMUB);
#define FIRMWARE_DCN316_DMUB "amdgpu/dcn_3_1_6_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_DCN316_DMUB);

#define FIRMWARE_DCN_V3_2_0_DMCUB "amdgpu/dcn_3_2_0_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_DCN_V3_2_0_DMCUB);
#define FIRMWARE_DCN_V3_2_1_DMCUB "amdgpu/dcn_3_2_1_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_DCN_V3_2_1_DMCUB);

#define FIRMWARE_RAVEN_DMCU		"amdgpu/raven_dmcu.bin"
MODULE_FIRMWARE(FIRMWARE_RAVEN_DMCU);

#define FIRMWARE_NAVI12_DMCU            "amdgpu/navi12_dmcu.bin"
MODULE_FIRMWARE(FIRMWARE_NAVI12_DMCU);

#define FIRMWARE_DCN_35_DMUB "amdgpu/dcn_3_5_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_DCN_35_DMUB);

#define FIRMWARE_DCN_351_DMUB "amdgpu/dcn_3_5_1_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_DCN_351_DMUB);

#define FIRMWARE_DCN_36_DMUB "amdgpu/dcn_3_6_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_DCN_36_DMUB);

#define FIRMWARE_DCN_401_DMUB "amdgpu/dcn_4_0_1_dmcub.bin"
MODULE_FIRMWARE(FIRMWARE_DCN_401_DMUB);

// Functional Utility: Defines the size of the PSP header for firmware images.
#define PSP_HEADER_BYTES 0x100

// Functional Utility: Defines the size of the PSP footer for firmware images.
#define PSP_FOOTER_BYTES 0x100

/**
 * DOC: overview
 *
 * The AMDgpu display manager, **amdgpu_dm** (or even simpler,
 * **dm**) sits between DRM and DC. It acts as a liaison, converting DRM
 * requests into DC requests, and DC responses into DRM responses.
 *
 * The root control structure is &struct amdgpu_display_manager.
 */

/* basic init/fini API */
// Functional Utility: Initializes the AMDGPU Display Manager.
static int amdgpu_dm_init(struct amdgpu_device *adev);
// Functional Utility: Finalizes and cleans up the AMDGPU Display Manager.
static void amdgpu_dm_fini(struct amdgpu_device *adev);
// Functional Utility: Checks if a given DRM display mode is compatible with FreeSync video mode.
static bool is_freesync_video_mode(const struct drm_display_mode *mode, struct amdgpu_dm_connector *aconnector);
// Functional Utility: Resets the FreeSync configuration for a given CRTC state.
static void reset_freesync_config_for_crtc(struct dm_crtc_state *new_crtc_state);
// Functional Utility: Creates an I2C adapter for DDC communication.
static struct amdgpu_i2c_adapter *
create_i2c(struct ddc_service *ddc_service, bool oem);

/**
 * @brief Retrieves the subconnector type based on the DisplayPort dongle type.
 * @param link Pointer to the DC link object.
 * @return An enum `drm_mode_subconnector` representing the detected dongle type.
 *
 * This function translates the DPCD (DisplayPort Configuration Data) dongle type
 * information from the DC link into a DRM-specific subconnector type.
 */
static enum drm_mode_subconnector get_subconnector_type(struct dc_link *link)
{
	switch (link->dpcd_caps.dongle_type) {
	// Block Logic: Maps various DisplayPort dongle types to corresponding DRM subconnector types.
	case DISPLAY_DONGLE_NONE:
		return DRM_MODE_SUBCONNECTOR_Native;
	case DISPLAY_DONGLE_DP_VGA_CONVERTER:
		return DRM_MODE_SUBCONNECTOR_VGA;
	case DISPLAY_DONGLE_DP_DVI_CONVERTER:
	case DISPLAY_DONGLE_DP_DVI_DONGLE:
		return DRM_MODE_SUBCONNECTOR_DVID;
	case DISPLAY_DONGLE_DP_HDMI_CONVERTER:
	case DISPLAY_DONGLE_DP_HDMI_DONGLE:
		return DRM_MODE_SUBCONNECTOR_HDMIA;
	case DISPLAY_DONGLE_DP_HDMI_MISMATCHED_DONGLE:
	default:
		return DRM_MODE_SUBCONNECTOR_Unknown;
	}
}

/**
 * @brief Updates the subconnector property of a DRM connector.
 * @param aconnector Pointer to the AMDGPU DM connector object.
 *
 * This function updates the `dp_subconnector_property` for DisplayPort
 * connectors based on the detected dongle type in the DC link.
 */
static void update_subconnector_property(struct amdgpu_dm_connector *aconnector)
{
	struct dc_link *link = aconnector->dc_link;
	struct drm_connector *connector = &aconnector->base;
	enum drm_mode_subconnector subconnector = DRM_MODE_SUBCONNECTOR_Unknown;

	// Block Logic: Only applies to DisplayPort connectors.
	if (connector->connector_type != DRM_MODE_CONNECTOR_DisplayPort)
		return;

	// Block Logic: If a sink is detected, determine the subconnector type.
	if (aconnector->dc_sink)
		subconnector = get_subconnector_type(link);

	// Functional Utility: Sets the DRM object property for the subconnector.
	drm_object_property_set_value(&connector->base,
			connector->dev->mode_config.dp_subconnector_property,
			subconnector);
}

/*
 * initializes drm_device display related structures, based on the information
 * provided by DAL. The drm strcutures are: drm_crtc, drm_connector,
 * drm_encoder, drm_mode_config
 *
 * Returns 0 on success
 */
// Functional Utility: Initializes DRM device display-related structures.
static int amdgpu_dm_initialize_drm_device(struct amdgpu_device *adev);
/* removes and deallocates the drm structures, created by the above function */
// Functional Utility: Destroys and deallocates DRM display-related structures.
static void amdgpu_dm_destroy_drm_device(struct amdgpu_display_manager *dm);

// Functional Utility: Initializes an AMDGPU DM connector.
static int amdgpu_dm_connector_init(struct amdgpu_display_manager *dm,
				    struct amdgpu_dm_connector *amdgpu_dm_connector,
				    u32 link_index,
				    struct amdgpu_encoder *amdgpu_encoder);
// Functional Utility: Initializes an AMDGPU encoder.
static int amdgpu_dm_encoder_init(struct drm_device *dev,
				  struct amdgpu_encoder *aencoder,
				  uint32_t link_index);

// Functional Utility: Retrieves display modes supported by a DRM connector.
static int amdgpu_dm_connector_get_modes(struct drm_connector *connector);

// Functional Utility: Handles the tail end of an atomic commit operation for DRM.
static void amdgpu_dm_atomic_commit_tail(struct drm_atomic_state *state);

// Functional Utility: Performs an atomic check for a DRM atomic state.
static int amdgpu_dm_atomic_check(struct drm_device *dev,
				  struct drm_atomic_state *state);

// Functional Utility: Helper function to handle HPD (Hot Plug Detect) IRQ.
static void handle_hpd_irq_helper(struct amdgpu_dm_connector *aconnector);
// Functional Utility: Handles HPD RX (Receiver) IRQ.
static void handle_hpd_rx_irq(void *param);

// Functional Utility: Sets the backlight level for a display.
static void amdgpu_dm_backlight_set_level(struct amdgpu_display_manager *dm,
					 int bl_idx,
					 u32 user_brightness);

// Functional Utility: Checks if display timing has remained unchanged for FreeSync.
static bool
is_timing_unchanged_for_freesync(struct drm_crtc_state *old_crtc_state,
				 struct drm_crtc_state *new_crtc_state);
/*
 * dm_vblank_get_counter
 *
 * @brief
 * Get counter for number of vertical blanks
 *
 * @param
 * struct amdgpu_device *adev - [in] desired amdgpu device
 * int disp_idx - [in] which CRTC to get the counter from
 *
 * @return
 * Counter for vertical blanks
 */
/**
 * @brief Retrieves the vertical blank counter for a specified CRTC.
 * @param adev Pointer to the AMDGPU device structure.
 * @param crtc The index of the CRTC.
 * @return The current vertical blank counter value, or 0 on error.
 *
 * This function queries the Display Core (DC) to get the VBLANK counter
 * for a given CRTC. It includes error checks for invalid CRTC index
 * and missing stream state.
 */
static u32 dm_vblank_get_counter(struct amdgpu_device *adev, int crtc)
{
	struct amdgpu_crtc *acrtc = NULL;

	// Block Logic: Validates the CRTC index to prevent out-of-bounds access.
	if (crtc >= adev->mode_info.num_crtc)
		return 0;

	// Functional Utility: Retrieves the AMDGPU CRTC object.
	acrtc = adev->mode_info.crtcs[crtc];

	// Block Logic: Checks if the DC stream state is valid for the CRTC.
	if (!acrtc->dm_irq_params.stream) {
		drm_err(adev_to_drm(adev), "dc_stream_state is NULL for crtc '%d'!\n",
			  crtc);
		return 0;
	}

	// Functional Utility: Calls the DC library function to get the VBLANK counter.
	return dc_stream_get_vblank_counter(acrtc->dm_irq_params.stream);
}

/**
 * @brief Retrieves the scanout position and vertical blank information for a CRTC.
 * @param adev Pointer to the AMDGPU device structure.
 * @param crtc The index of the CRTC.
 * @param vbl Output parameter for vertical blank start/end positions.
 * @param position Output parameter for current horizontal/vertical scanout position.
 * @return 0 on success, -EINVAL on invalid CRTC index.
 *
 * This function gets detailed scanout position data from the Display Core,
 * including vertical blank timing and current cursor position.
 */
static int dm_crtc_get_scanoutpos(struct amdgpu_device *adev, int crtc,
				  u32 *vbl, u32 *position)
{
	u32 v_blank_start = 0, v_blank_end = 0, h_position = 0, v_position = 0;
	struct amdgpu_crtc *acrtc = NULL;
	struct dc *dc = adev->dm.dc;

	// Block Logic: Validates the CRTC index.
	if ((crtc < 0) || (crtc >= adev->mode_info.num_crtc))
		return -EINVAL;

	// Functional Utility: Retrieves the AMDGPU CRTC object.
	acrtc = adev->mode_info.crtcs[crtc];

	// Block Logic: Checks if the DC stream state is valid.
	if (!acrtc->dm_irq_params.stream) {
		drm_err(adev_to_drm(adev), "dc_stream_state is NULL for crtc '%d'!\n",
			  crtc);
		return 0;
	}

	// Block Logic: Temporarily disables idle optimizations if IPS is supported and enabled.
	if (dc && dc->caps.ips_support && dc->idle_optimizations_allowed)
		dc_allow_idle_optimizations(dc, false);

	/*
	 * TODO rework base driver to use values directly.
	 * for now parse it back into reg-format
	 */
	// Functional Utility: Calls the DC library to get raw scanout position data.
	dc_stream_get_scanoutpos(acrtc->dm_irq_params.stream,
				 &v_blank_start,
				 &v_blank_end,
				 &h_position,
				 &v_position);

	// Functional Utility: Packs the horizontal and vertical positions into a single U32.
	*position = v_position | (h_position << 16);
	// Functional Utility: Packs the vertical blank start and end into a single U32.
	*vbl = v_blank_start | (v_blank_end << 16);

	return 0;
}

/**
 * @brief Checks if the AMDGPU IP block is idle.
 * @param ip_block Pointer to the AMDGPU IP block structure.
 * @return Always returns true (TODO: implement actual idle check).
 */
static bool dm_is_idle(struct amdgpu_ip_block *ip_block)
{
	/* XXX todo */
	return true;
}

/**
 * @brief Waits for the AMDGPU IP block to become idle.
 * @param ip_block Pointer to the AMDGPU IP block structure.
 * @return Always returns 0 (TODO: implement actual wait).
 */
static int dm_wait_for_idle(struct amdgpu_ip_block *ip_block)
{
	/* XXX todo */
	return 0;
}

/**
 * @brief Checks for a soft reset condition in the AMDGPU IP block.
 * @param ip_block Pointer to the AMDGPU IP block structure.
 * @return Always returns false (TODO: implement actual soft reset check).
 */
static bool dm_check_soft_reset(struct amdgpu_ip_block *ip_block)
{
	return false;
}

/**
 * @brief Initiates a soft reset for the AMDGPU IP block.
 * @param ip_block Pointer to the AMDGPU IP block structure.
 * @return Always returns 0 (TODO: implement actual soft reset).
 */
static int dm_soft_reset(struct amdgpu_ip_block *ip_block)
{
	/* XXX todo */
	return 0;
}

/**
 * @brief Retrieves an AMDGPU CRTC object by its OTG (Output Timing Generator) instance.
 * @param adev Pointer to the AMDGPU device structure.
 * @param otg_inst The OTG instance number.
 * @return Pointer to the AMDGPU CRTC object, or NULL if not found.
 *
 * This function iterates through all registered CRTCs to find the one
 * associated with the given OTG instance.
 */
static struct amdgpu_crtc *
get_crtc_by_otg_inst(struct amdgpu_device *adev,
		     int otg_inst)
{
	struct drm_device *dev = adev_to_drm(adev);
	struct drm_crtc *crtc;
	struct amdgpu_crtc *amdgpu_crtc;

	// Block Logic: Handles a warning condition for invalid OTG instance.
	if (WARN_ON(otg_inst == -1))
		return adev->mode_info.crtcs[0];

	// Block Logic: Iterates through the list of DRM CRTCs.
	list_for_each_entry(crtc, &dev->mode_config.crtc_list, head) {
		amdgpu_crtc = to_amdgpu_crtc(crtc);

		// Block Logic: Compares the CRTC's OTG instance with the target instance.
		if (amdgpu_crtc->otg_inst == otg_inst)
			return amdgpu_crtc;
	}

	return NULL;
}

/**
 * @brief Checks if a DC timing adjustment is needed for FreeSync.
 * @param old_crtc_state Previous CRTC state.
 * @param new_crtc_state New CRTC state.
 * @return True if timing adjustment is needed, false otherwise.
 *
 * This function determines if the Display Core (DC) needs to perform
 * a timing adjustment, especially in the context of FreeSync/VRR modes.
 */
static inline bool is_dc_timing_adjust_needed(struct dm_crtc_state *old_crtc_state,
					      struct dm_crtc_state *new_crtc_state)
{
	// Block Logic: Checks if a timing adjustment is pending in the new stream state.
	if (new_crtc_state->stream->adjust.timing_adjust_pending)
		return true;
	// Block Logic: Checks if the FreeSync state is active fixed.
	if (new_crtc_state->freesync_config.state ==  VRR_STATE_ACTIVE_FIXED)
		return true;
	// Block Logic: Checks if the VRR (Variable Refresh Rate) active state has changed.
	else if (amdgpu_dm_crtc_vrr_active(old_crtc_state) != amdgpu_dm_crtc_vrr_active(new_crtc_state))
		return true;
	else
		return false;
}

/*
 * DC will program planes with their z-order determined by their ordering
 * in the dc_surface_updates array. This comparator is used to sort them
 * by descending zpos.
 */
/**
 * @brief Comparator function to sort `dc_surface_update` structures by layer index.
 * @param a Pointer to the first `dc_surface_update` structure.
 * @param b Pointer to the second `dc_surface_update` structure.
 * @return An integer indicating the relative order for sorting in descending layer index.
 *
 * This function is used to sort display planes based on their Z-order (layer_index)
 * in descending order, ensuring that DC programs them correctly.
 */
static int dm_plane_layer_index_cmp(const void *a, const void *b)
{
	const struct dc_surface_update *sa = (const struct dc_surface_update *)a;
	const struct dc_surface_update *sb = (const struct dc_surface_update *)b;

	/* Sort by descending dc_plane layer_index (i.e. normalized_zpos) */
	return sb->surface->layer_index - sa->surface->layer_index;
}

/**
 * @brief Wrapper function to send plane and stream updates to Display Core.
 * @param dc Display Core control structure.
 * @param update_type Type of update (FULL, MEDIUM, FAST).
 * @param planes_count Number of planes to update.
 * @param stream Pointer to the DC stream state.
 * @param stream_update Pointer to the DC stream update information.
 * @param array_of_surface_update Array of DC surface update structures.
 * @return True if the update was successful, false otherwise.
 *
 * This function prepares plane updates for the DC by sorting them by Z-order
 * and then calls the `dc_update_planes_and_stream` function.
 * It also handles post-update actions for fast updates.
 */
static inline bool update_planes_and_stream_adapter(struct dc *dc,
						    int update_type,
						    int planes_count,
						    struct dc_stream_state *stream,
						    struct dc_stream_update *stream_update,
						    struct dc_surface_update *array_of_surface_update)
{
	// Functional Utility: Sorts the array of surface updates based on their layer index (Z-order).
	sort(array_of_surface_update, planes_count,
	     sizeof(*array_of_surface_update), dm_plane_layer_index_cmp, NULL);

	/*
	 * Previous frame finished and HW is ready for optimization.
	 */
	// Block Logic: If it's a fast update, signals the DC that surfaces have been updated.
	if (update_type == UPDATE_TYPE_FAST)
		dc_post_update_surfaces_to_stream(dc);

	// Functional Utility: Calls the DC function to apply the plane and stream updates.
	return dc_update_planes_and_stream(dc,
					   array_of_surface_update,
					   planes_count,
					   stream,
					   stream_update);
}

/**
 * @brief Handles the pageflip interrupt (high priority).
 * @param interrupt_params Pointer to common IRQ parameters.
 *
 * This function is invoked when a pageflip completion interrupt occurs.
 * It notifies DRM of the completion, handles vblank events, and manages
 * the pageflip status, especially considering VRR (Variable Refresh Rate) scenarios.
 */
static void dm_pflip_high_irq(void *interrupt_params)
{
	struct amdgpu_crtc *amdgpu_crtc;
	struct common_irq_params *irq_params = interrupt_params;
	struct amdgpu_device *adev = irq_params->adev;
	struct drm_device *dev = adev_to_drm(adev);
	unsigned long flags;
	struct drm_pending_vblank_event *e;
	u32 vpos, hpos, v_blank_start, v_blank_end;
	bool vrr_active;

	// Functional Utility: Retrieves the AMDGPU CRTC associated with the IRQ source.
	amdgpu_crtc = get_crtc_by_otg_inst(adev, irq_params->irq_src - IRQ_TYPE_PFLIP);

	/* IRQ could occur when in initial stage */
	/* TODO work and BO cleanup */
	// Block Logic: Handles cases where CRTC is null, potentially during early initialization.
	if (amdgpu_crtc == NULL) {
		drm_dbg_state(dev, "CRTC is null, returning.\n");
		return;
	}

	// Block Logic: Acquires a spinlock to protect DRM event list manipulation.
	spin_lock_irqsave(&adev_to_drm(adev)->event_lock, flags);

	// Block Logic: Checks if a pageflip was actually submitted.
	if (amdgpu_crtc->pflip_status != AMDGPU_FLIP_SUBMITTED) {
		drm_dbg_state(dev,
			      "amdgpu_crtc->pflip_status = %d != AMDGPU_FLIP_SUBMITTED(%d) on crtc:%d[%p]\n",
			      amdgpu_crtc->pflip_status, AMDGPU_FLIP_SUBMITTED,
			      amdgpu_crtc->crtc_id, amdgpu_crtc);
		spin_unlock_irqrestore(&adev_to_drm(adev)->event_lock, flags);
		return;
	}

	/* page flip completed. */
	// Functional Utility: Retrieves the pending vblank event for the completed pageflip.
	e = amdgpu_crtc->event;
	amdgpu_crtc->event = NULL;

	// Block Logic: Asserts that a pending event exists.
	WARN_ON(!e);

	// Functional Utility: Checks if VRR is active for the CRTC.
	vrr_active = amdgpu_dm_crtc_vrr_active_irq(amdgpu_crtc);

	/* Fixed refresh rate, or VRR scanout position outside front-porch? */
	/**
	 * Block Logic: Differentiates vblank handling based on VRR state and scanout position.
	 * For fixed refresh rates or when VRR is active but scanout is outside the front-porch,
	 * it updates the vblank count and sends the pageflip event directly.
	 */
	if (!vrr_active ||
	    !dc_stream_get_scanoutpos(amdgpu_crtc->dm_irq_params.stream, &v_blank_start,
				      &v_blank_end, &hpos, &vpos) ||
	    (vpos < v_blank_start)) {
		/* Update to correct count and vblank timestamp if racing with
		 * vblank irq. This also updates to the correct vblank timestamp
		 * even in VRR mode, as scanout is past the front-porch atm.
		 */
		drm_crtc_accurate_vblank_count(&amdgpu_crtc->base);

		/* Wake up userspace by sending the pageflip event with proper
		 * count and timestamp of vblank of flip completion.
		 */
		if (e) {
			drm_crtc_send_vblank_event(&amdgpu_crtc->base, e);

			/* Event sent, so done with vblank for this flip */
			drm_crtc_vblank_put(&amdgpu_crtc->base);
		}
	} else if (e) {
		/* VRR active and inside front-porch: vblank count and
		 * timestamp for pageflip event will only be up to date after
		 * drm_crtc_handle_vblank() has been executed from late vblank
		 * irq handler after start of back-porch (vline 0). We queue the
		 * pageflip event for send-out by drm_crtc_handle_vblank() with
		 * updated timestamp and count, once it runs after us.
		 *
		 * We need to open-code this instead of using the helper
		 * drm_crtc_arm_vblank_event(), as that helper would
		 * call drm_crtc_accurate_vblank_count(), which we must
		 * not call in VRR mode while we are in front-porch!
		 */

		/* sequence will be replaced by real count during send-out. */
		// Functional Utility: Prepares the event for later delivery by the vblank handler.
		e->sequence = drm_crtc_vblank_count(&amdgpu_crtc->base);
		e->pipe = amdgpu_crtc->crtc_id;

		list_add_tail(&e->base.link, &adev_to_drm(adev)->vblank_event_list);
		e = NULL;
	}

	/* Keep track of vblank of this flip for flip throttling. We use the
	 * cooked hw counter, as that one incremented at start of this vblank
	 * of pageflip completion, so last_flip_vblank is the forbidden count
	 * for queueing new pageflips if vsync + VRR is enabled.
	 */
	// Functional Utility: Records the vblank counter at the time of flip completion for throttling.
	amdgpu_crtc->dm_irq_params.last_flip_vblank =
		amdgpu_get_vblank_counter_kms(&amdgpu_crtc->base);

	// Functional Utility: Resets the pageflip status.
	amdgpu_crtc->pflip_status = AMDGPU_FLIP_NONE;
	spin_unlock_irqrestore(&adev_to_drm(adev)->event_lock, flags);

	drm_dbg_state(dev,
		      "crtc:%d[%p], pflip_stat:AMDGPU_FLIP_NONE, vrr[%d]-fp %d\n",
		      amdgpu_crtc->crtc_id, amdgpu_crtc, vrr_active, (int)!e);
}

/**
 * @brief Handles the vupdate (vertical update) high priority interrupt.
 * @param interrupt_params Pointer to common IRQ parameters.
 *
 * This function is triggered by the vertical update interrupt, primarily
 * used for Variable Refresh Rate (VRR) tracking and related timing adjustments.
 * It tracks frame durations and updates FreeSync parameters.
 */
static void dm_vupdate_high_irq(void *interrupt_params)
{
	struct common_irq_params *irq_params = interrupt_params;
	struct amdgpu_device *adev = irq_params->adev;
	struct amdgpu_crtc *acrtc;
	struct drm_device *drm_dev;
	struct drm_vblank_crtc *vblank;
	ktime_t frame_duration_ns, previous_timestamp;
	unsigned long flags;
	int vrr_active;

	// Functional Utility: Retrieves the AMDGPU CRTC associated with the IRQ source.
	acrtc = get_crtc_by_otg_inst(adev, irq_params->irq_src - IRQ_TYPE_VUPDATE);

	// Block Logic: If a valid CRTC is found, proceed with vupdate processing.
	if (acrtc) {
		// Functional Utility: Checks if VRR is active.
		vrr_active = amdgpu_dm_crtc_vrr_active_irq(acrtc);
		drm_dev = acrtc->base.dev;
		vblank = drm_crtc_vblank_crtc(&acrtc->base);
		previous_timestamp = atomic64_read(&irq_params->previous_timestamp);
		frame_duration_ns = vblank->time - previous_timestamp;

		// Block Logic: Tracks and traces refresh rate if frame duration is positive.
		if (frame_duration_ns > 0) {
			trace_amdgpu_refresh_rate_track(acrtc->base.index,
						frame_duration_ns,
						ktime_divns(NSEC_PER_SEC, frame_duration_ns));
			atomic64_set(&irq_params->previous_timestamp, vblank->time);
		}

		drm_dbg_vbl(drm_dev,
			    "crtc:%d, vupdate-vrr:%d\n", acrtc->crtc_id,
			    vrr_active);

		/* Core vblank handling is done here after end of front-porch in
		 * vrr mode, as vblank timestamping will give valid results
		 * while now done after front-porch. This will also deliver
		 * page-flip completion events that have been queued to us
		 * if a pageflip happened inside front-porch.
		 */
		// Block Logic: If VRR is active, handle vblank and FreeSync adjustments.
		if (vrr_active) {
			amdgpu_dm_crtc_handle_vblank(acrtc);

			/* BTR processing for pre-DCE12 ASICs */
			// Block Logic: Applies FreeSync adjustments for older ASICs.
			if (acrtc->dm_irq_params.stream &&
			    adev->family < AMDGPU_FAMILY_AI) {
				spin_lock_irqsave(&adev_to_drm(adev)->event_lock, flags);
				mod_freesync_handle_v_update(
				    adev->dm.freesync_module,
				    acrtc->dm_irq_params.stream,
				    &acrtc->dm_irq_params.vrr_params);

				dc_stream_adjust_vmin_vmax(
				    adev->dm.dc,
				    acrtc->dm_irq_params.stream,
				    &acrtc->dm_irq_params.vrr_params.adjust);
				spin_unlock_irqrestore(&adev_to_drm(adev)->event_lock, flags);
			}
		}
	}
}

/**
 * @brief Handles CRTC (Cathode Ray Tube Controller) interrupt, typically VBLANK.
 * @param interrupt_params Parameters for the interrupt, used to identify the CRTC.
 *
 * This function processes CRTC interrupts, which can signal VBLANK events.
 * It handles writeback job completion, VRR (Variable Refresh Rate) logic,
 * CRC (Cyclic Redundancy Check) calculation, and pageflip event delivery.
 */
static void dm_crtc_high_irq(void *interrupt_params)
{
	struct common_irq_params *irq_params = interrupt_params;
	struct amdgpu_device *adev = irq_params->adev;
	struct drm_writeback_job *job;
	struct amdgpu_crtc *acrtc;
	unsigned long flags;
	int vrr_active;

	// Functional Utility: Retrieves the AMDGPU CRTC associated with the IRQ source.
	acrtc = get_crtc_by_otg_inst(adev, irq_params->irq_src - IRQ_TYPE_VBLANK);
	// Block Logic: Returns if no valid CRTC is found.
	if (!acrtc)
		return;

	// Block Logic: Handles writeback job completion if a writeback connector is present and a job is pending.
	if (acrtc->wb_conn) {
		spin_lock_irqsave(&acrtc->wb_conn->job_lock, flags);

		if (acrtc->wb_pending) {
			job = list_first_entry_or_null(&acrtc->wb_conn->job_queue,
						       struct drm_writeback_job,
						       list_entry);
			acrtc->wb_pending = false;
			spin_unlock_irqrestore(&acrtc->wb_conn->job_lock, flags);

			if (job) {
				unsigned int v_total, refresh_hz;
				struct dc_stream_state *stream = acrtc->dm_irq_params.stream;

				// Functional Utility: Calculates vertical total and refresh rate.
				v_total = stream->adjust.v_total_max ?
					  stream->adjust.v_total_max : stream->timing.v_total;
				refresh_hz = div_u64((uint64_t) stream->timing.pix_clk_100hz *
					     100LL, (v_total * stream->timing.h_total));
				// Functional Utility: Introduces a delay based on refresh rate.
				mdelay(1000 / refresh_hz);

				// Functional Utility: Signals writeback completion and disables FC writeback.
				drm_writeback_signal_completion(acrtc->wb_conn, 0);
				dc_stream_fc_disable_writeback(adev->dm.dc,
							       acrtc->dm_irq_params.stream, 0);
			}
		} else
			spin_unlock_irqrestore(&acrtc->wb_conn->job_lock, flags);
	}

	// Functional Utility: Checks if VRR is active for the CRTC.
	vrr_active = amdgpu_dm_crtc_vrr_active_irq(acrtc);

	drm_dbg_vbl(adev_to_drm(adev),
		    "crtc:%d, vupdate-vrr:%d, planes:%d\n", acrtc->crtc_id,
		    vrr_active, acrtc->dm_irq_params.active_planes);

	/**
	 * Block Logic: Handles core vblank processing.
	 * If VRR is not active, `amdgpu_dm_crtc_handle_vblank` is called directly.
	 * Otherwise, it's deferred to `dm_vupdate_high_irq`.
	 */
	if (!vrr_active)
		amdgpu_dm_crtc_handle_vblank(acrtc);

	/**
	 * Following stuff must happen at start of vblank, for crc
	 * computation and below-the-range btr support in vrr mode.
	 */
	// Functional Utility: Handles CRC IRQ for the CRTC.
	amdgpu_dm_crtc_handle_crc_irq(&acrtc->base);

	/* BTR updates need to happen before VUPDATE on Vega and above. */
	// Block Logic: Applies FreeSync adjustments for ASICs newer than Carrizo.
	if (adev->family < AMDGPU_FAMILY_AI)
		return;

	spin_lock_irqsave(&adev_to_drm(adev)->event_lock, flags);

	// Block Logic: If VRR is supported and active variable, updates FreeSync and stream vmin/vmax.
	if (acrtc->dm_irq_params.stream &&
	    acrtc->dm_irq_params.vrr_params.supported &&
	    acrtc->dm_irq_params.freesync_config.state ==
		    VRR_STATE_ACTIVE_VARIABLE) {
		mod_freesync_handle_v_update(adev->dm.freesync_module,
					     acrtc->dm_irq_params.stream,
					     &acrtc->dm_irq_params.vrr_params);

		dc_stream_adjust_vmin_vmax(adev->dm.dc, acrtc->dm_irq_params.stream,
					   &acrtc->dm_irq_params.vrr_params.adjust);
	}

	/*
	 * If there aren't any active_planes then DCH HUBP may be clock-gated.
	 * In that case, pageflip completion interrupts won't fire and pageflip
	 * completion events won't get delivered. Prevent this by sending
	 * pending pageflip events from here if a flip is still pending.
	 *
	 * If any planes are enabled, use dm_pflip_high_irq() instead, to
	 * avoid race conditions between flip programming and completion,
	 * which could cause too early flip completion events.
	 */
	// Block Logic: For newer ASICs and pending pageflips with no active planes, forces pageflip event delivery.
	if (adev->family >= AMDGPU_FAMILY_RV &&
	    acrtc->pflip_status == AMDGPU_FLIP_SUBMITTED &&
	    acrtc->dm_irq_params.active_planes == 0) {
		if (acrtc->event) {
			drm_crtc_send_vblank_event(&acrtc->base, acrtc->event);
			acrtc->event = NULL;
			drm_crtc_vblank_put(&acrtc->base);
		}
		acrtc->pflip_status = AMDGPU_FLIP_NONE;
	}

	spin_unlock_irqrestore(&adev_to_drm(adev)->event_lock, flags);
}

#if defined(CONFIG_DRM_AMD_SECURE_DISPLAY)
/**
 * @brief Handles OTG Vertical interrupt0 for DCN generation ASICs for secure display.
 * @param interrupt_params Parameters for the interrupt.
 *
 * This function is used to set the CRC window or read out CRC values at the vertical line 0 position
 * for DCN (Display Core Next) generation ASICs, specifically for secure display functionalities.
 */
static void dm_dcn_vertical_interrupt0_high_irq(void *interrupt_params)
{
	struct common_irq_params *irq_params = interrupt_params;
	struct amdgpu_device *adev = irq_params->adev;
	struct amdgpu_crtc *acrtc;

	// Functional Utility: Retrieves the AMDGPU CRTC associated with the IRQ source.
	acrtc = get_crtc_by_otg_inst(adev, irq_params->irq_src - IRQ_TYPE_VLINE0);

	// Block Logic: Returns if no valid CRTC is found.
	if (!acrtc)
		return;

	// Functional Utility: Handles the CRC window IRQ for the CRTC.
	amdgpu_dm_crtc_handle_crc_window_irq(&acrtc->base);
}
#endif /* CONFIG_DRM_AMD_SECURE_DISPLAY */

/**
 * @brief Callback for AUX (Auxiliary) or SET_CONFIG (Set Configuration) commands from DMUB.
 * @param adev Pointer to the AMDGPU device structure.
 * @param notify Pointer to the DMUB notification structure.
 *
 * This function processes completion notifications for AUX or SET_CONFIG commands
 * sent to the DMUB. It copies the notification data and signals completion to the
 * waiting thread.
 */
static void dmub_aux_setconfig_callback(struct amdgpu_device *adev,
					struct dmub_notification *notify)
{
	// Block Logic: Copies the DMUB notification data if a valid target is available.
	if (adev->dm.dmub_notify)
		memcpy(adev->dm.dmub_notify, notify, sizeof(struct dmub_notification));
	// Block Logic: If the notification is an AUX reply, signals the completion event.
	if (notify->type == DMUB_NOTIFICATION_AUX_REPLY)
		complete(&adev->dm.dmub_aux_transfer_done);
}

/**
 * @brief Callback for Fused I/O notifications from DMUB.
 * @param adev Pointer to the AMDGPU device structure.
 * @param notify Pointer to the DMUB notification structure.
 *
 * This function handles fused I/O notifications from the DMUB, copying the
 * reply data and signaling completion for the relevant DDC line.
 */
static void dmub_aux_fused_io_callback(struct amdgpu_device *adev,
					struct dmub_notification *notify)
{
	// Block Logic: Basic validation of input pointers.
	if (!adev || !notify) {
		ASSERT(false);
		return;
	}

	const struct dmub_cmd_fused_request *req = &notify->fused_request;
	const uint8_t ddc_line = req->u.aux.ddc_line;

	// Block Logic: Validates the DDC line index.
	if (ddc_line >= ARRAY_SIZE(adev->dm.fused_io)) {
		ASSERT(false);
		return;
	}

	struct fused_io_sync *sync = &adev->dm.fused_io[ddc_line];

	// Functional Utility: Static assertion for size consistency.
	static_assert(sizeof(*req) <= sizeof(sync->reply_data), "Size mismatch");
	// Functional Utility: Copies the reply data to the synchronization structure.
	memcpy(sync->reply_data, req, sizeof(*req));
	// Functional Utility: Signals completion to wake up waiting threads.
	complete(&sync->replied);
}

/**
 * @brief DMUB HPD (Hot Plug Detect) interrupt processing callback.
 * @param adev Pointer to the AMDGPU device structure.
 * @param notify Pointer to the DMUB notification structure.
 *
 * This function handles HPD notifications from the DMUB, identifying the
 * affected display link and calling appropriate handlers for HPD or HPD RX IRQ.
 */
static void dmub_hpd_callback(struct amdgpu_device *adev,
			      struct dmub_notification *notify)
{
	struct amdgpu_dm_connector *aconnector;
	struct amdgpu_dm_connector *hpd_aconnector = NULL;
	struct drm_connector *connector;
	struct drm_connector_list_iter iter;
	struct dc_link *link;
	u8 link_index = 0;
	struct drm_device *dev;

	// Block Logic: Basic validation of `adev` pointer.
	if (adev == NULL)
		return;

	// Block Logic: Basic validation of `notify` pointer.
	if (notify == NULL) {
		drm_err(adev_to_drm(adev), "DMUB HPD callback notification was NULL");
		return;
	}

	// Block Logic: Validates the link index from the notification.
	if (notify->link_index > adev->dm.dc->link_count) {
		drm_err(adev_to_drm(adev), "DMUB HPD index (%u)is abnormal", notify->link_index);
		return;
	}

	/* Skip DMUB HPD IRQ in suspend/resume. We will probe them later. */
	// Block Logic: Skips HPD IRQ processing during suspend/resume cycles.
	if (notify->type == DMUB_NOTIFICATION_HPD && adev->in_suspend) {
		drm_info(adev_to_drm(adev), "Skip DMUB HPD IRQ callback in suspend/resume\n");
		return;
	}

	link_index = notify->link_index;
	link = adev->dm.dc->links[link_index];
	dev = adev->dm.ddev;

	// Block Logic: Iterates through DRM connectors to find the matching AMDGPU DM connector.
	drm_connector_list_iter_begin(dev, &iter);
	drm_for_each_connector_iter(connector, &iter) {

		// Block Logic: Skips writeback connectors.
		if (connector->connector_type == DRM_MODE_CONNECTOR_WRITEBACK)
			continue;

		aconnector = to_amdgpu_dm_connector(connector);
		// Block Logic: Checks if the DC link matches and logs the notification type.
		if (link && aconnector->dc_link == link) {
			if (notify->type == DMUB_NOTIFICATION_HPD)
				drm_info(adev_to_drm(adev), "DMUB HPD IRQ callback: link_index=%u\n", link_index);
			else if (notify->type == DMUB_NOTIFICATION_HPD_IRQ)
				drm_info(adev_to_drm(adev), "DMUB HPD RX IRQ callback: link_index=%u\n", link_index);
			else
				drm_warn(adev_to_drm(adev), "DMUB Unknown HPD callback type %d, link_index=%u\n",
						notify->type, link_index);

			hpd_aconnector = aconnector;
			break;
		}
	}
	drm_connector_list_iter_end(&iter);

	// Block Logic: If a matching connector is found, handles HPD or HPD RX IRQ.
	if (hpd_aconnector) {
		if (notify->type == DMUB_NOTIFICATION_HPD) {
			// Block Logic: Warns if HPD status is reported unchanged.
			if (hpd_aconnector->dc_link->hpd_status == (notify->hpd_status == DP_HPD_PLUG))
				drm_warn(adev_to_drm(adev), "DMUB reported hpd status unchanged. link_index=%u\n", link_index);
			handle_hpd_irq_helper(hpd_aconnector);
		} else if (notify->type == DMUB_NOTIFICATION_HPD_IRQ) {
			handle_hpd_rx_irq(hpd_aconnector);
		}
	}
}

/**
 * @brief DMUB HPD sense processing callback.
 * @param adev Pointer to the AMDGPU device structure.
 * @param notify Pointer to the DMUB notification structure.
 *
 * This callback is invoked when HPD (Hot Plug Detect) sense changes occur,
 * especially during low power states, to notify the driver from the firmware.
 */
static void dmub_hpd_sense_callback(struct amdgpu_device *adev,
			      struct dmub_notification *notify)
{
	drm_dbg_driver(adev_to_drm(adev), "DMUB HPD SENSE callback.\n");
}

/**
 * @brief Registers a callback function for a specific DMUB notification type.
 * @param adev Pointer to the AMDGPU device structure.
 * @param type Type of DMUB notification.
 * @param callback The callback function to register.
 * @param dmub_int_thread_offload Boolean indicating if callback processing should be offloaded to a dedicated thread.
 * @return True if successfully registered, false if there is an existing registration for the type.
 *
 * This API allows registering a handler for DMUB notifications, with an option
 * to offload the processing to a separate interrupt handling thread.
 */
static bool register_dmub_notify_callback(struct amdgpu_device *adev,
					  enum dmub_notification_type type,
					  dmub_notify_interrupt_callback_t callback,
					  bool dmub_int_thread_offload)
{
	// Block Logic: Validates callback and type, then assigns the callback and offload preference.
	if (callback != NULL && type < ARRAY_SIZE(adev->dm.dmub_thread_offload)) {
		adev->dm.dmub_callback[type] = callback;
		adev->dm.dmub_thread_offload[type] = dmub_int_thread_offload;
	} else
		return false;

	return true;
}

/**
 * @brief Work function for handling HPD events offloaded to a workqueue.
 * @param work Pointer to the `work_struct` associated with the HPD event.
 *
 * This function retrieves the DMUB HPD work item, calls the appropriate
 * DMUB notification callback, and then frees the allocated memory.
 */
static void dm_handle_hpd_work(struct work_struct *work)
{
	struct dmub_hpd_work *dmub_hpd_wrk;

	// Functional Utility: Casts the generic `work_struct` to the specific `dmub_hpd_work`.
	dmub_hpd_wrk = container_of(work, struct dmub_hpd_work, handle_hpd_work);

	// Block Logic: Validates the DMUB notification pointer.
	if (!dmub_hpd_wrk->dmub_notify) {
		drm_err(adev_to_drm(dmub_hpd_wrk->adev), "dmub_hpd_wrk dmub_notify is NULL");
		return;
	}

	// Block Logic: Calls the registered DMUB callback for the notification type.
	if (dmub_hpd_wrk->dmub_notify->type < ARRAY_SIZE(dmub_hpd_wrk->adev->dm.dmub_callback)) {
		dmub_hpd_wrk->adev->dm.dmub_callback[dmub_hpd_wrk->dmub_notify->type](dmub_hpd_wrk->adev,
		dmub_hpd_wrk->dmub_notify);
	}

	// Functional Utility: Frees memory allocated for the DMUB notification and work item.
	kfree(dmub_hpd_wrk->dmub_notify);
	kfree(dmub_hpd_wrk);

}

/**
 * @brief Converts a DMUB notification type enum to its string representation.
 * @param e The `dmub_notification_type` enum value.
 * @return A constant string representing the notification type.
 */
static const char *dmub_notification_type_str(enum dmub_notification_type e)
{
	switch (e) {
	// Block Logic: Maps each DMUB notification type to a descriptive string.
	case DMUB_NOTIFICATION_NO_DATA:
		return "NO_DATA";
	case DMUB_NOTIFICATION_AUX_REPLY:
		return "AUX_REPLY";
	case DMUB_NOTIFICATION_HPD:
		return "HPD";
	case DMUB_NOTIFICATION_HPD_IRQ:
		return "HPD_IRQ";
	case DMUB_NOTIFICATION_SET_CONFIG_REPLY:
		return "SET_CONFIG_REPLY";
	case DMUB_NOTIFICATION_DPIA_NOTIFICATION:
		return "DPIA_NOTIFICATION";
	case DMUB_NOTIFICATION_HPD_SENSE_NOTIFY:
		return "HPD_SENSE_NOTIFY";
	case DMUB_NOTIFICATION_FUSED_IO:
		return "FUSED_IO";
	default:
		return "<unknown>";
	}
}

// Functional Utility: Defines the maximum number of DMUB trace entries to read at once.
#define DMUB_TRACE_MAX_READ 64
/**
 * @brief Handles the DMUB (Display Microcontroller Unit Bypass) Outbox1 low priority interrupt.
 * @param interrupt_params Parameters for the interrupt.
 *
 * This function processes messages from the DMUB Outbox1, including trace entries
 * and various notifications (HPD, AUX replies, etc.). It logs trace data and
 * dispatches notifications to appropriate handlers, potentially offloading them
 * to a workqueue for asynchronous processing.
 */
static void dm_dmub_outbox1_low_irq(void *interrupt_params)
{
	struct dmub_notification notify = {0};
	struct common_irq_params *irq_params = interrupt_params;
	struct amdgpu_device *adev = irq_params->adev;
	struct amdgpu_display_manager *dm = &adev->dm;
	struct dmcub_trace_buf_entry entry = { 0 };
	u32 count = 0;
	struct dmub_hpd_work *dmub_hpd_wrk;

	// Block Logic: Reads DMUB trace entries from the Outbox0 until no more entries or max read limit is reached.
	do {
		if (dc_dmub_srv_get_dmub_outbox0_msg(dm->dc, &entry)) {
			// Functional Utility: Traces DMUB messages for debugging.
			trace_amdgpu_dmub_trace_high_irq(entry.trace_code, entry.tick_count,
							entry.param0, entry.param1);

			drm_dbg_driver(adev_to_drm(adev), "trace_code:%u, tick_count:%u, param0:%u, param1:%u\n",
				 entry.trace_code, entry.tick_count, entry.param0, entry.param1);
		} else
			break;

		count++;

	} while (count <= DMUB_TRACE_MAX_READ);

	// Block Logic: Logs a warning if more trace entries were available than the read limit.
	if (count > DMUB_TRACE_MAX_READ)
		drm_dbg_driver(adev_to_drm(adev), "Warning : count > DMUB_TRACE_MAX_READ");

	// Block Logic: Processes DMUB notifications if enabled and the IRQ source is from DMCUB Outbox.
	if (dc_enable_dmub_notifications(adev->dm.dc) &&
		irq_params->irq_src == DC_IRQ_SOURCE_DMCUB_OUTBOX) {

		// Block Logic: Continues processing notifications as long as there are pending notifications.
		do {
			dc_stat_get_dmub_notification(adev->dm.dc, &notify);
			// Block Logic: Validates the notification type.
			if (notify.type >= ARRAY_SIZE(dm->dmub_thread_offload)) {
				drm_err(adev_to_drm(adev), "DM: notify type %d invalid!", notify.type);
				continue;
			}
			// Block Logic: Warns if no handler is registered for the notification type.
			if (!dm->dmub_callback[notify.type]) {
				drm_warn(adev_to_drm(adev), "DMUB notification skipped due to no handler: type=%s\n",
					dmub_notification_type_str(notify.type));
				continue;
			}
			// Block Logic: If offloading is enabled, allocates work and queues it for asynchronous processing.
			if (dm->dmub_thread_offload[notify.type] == true) {
				dmub_hpd_wrk = kzalloc(sizeof(*dmub_hpd_wrk), GFP_ATOMIC);
				if (!dmub_hpd_wrk) {
					drm_err(adev_to_drm(adev), "Failed to allocate dmub_hpd_wrk");
					return;
				}
				dmub_hpd_wrk->dmub_notify = kmemdup(&notify, sizeof(struct dmub_notification),
								    GFP_ATOMIC);
				if (!dmub_hpd_wrk->dmub_notify) {
					kfree(dmub_hpd_wrk);
					drm_err(adev_to_drm(adev), "Failed to allocate dmub_hpd_wrk->dmub_notify");
					return;
				}
				INIT_WORK(&dmub_hpd_wrk->handle_hpd_work, dm_handle_hpd_work);
				dmub_hpd_wrk->adev = adev;
				queue_work(adev->dm.delayed_hpd_wq, &dmub_hpd_wrk->handle_hpd_work);
			} else {
				// Block Logic: Otherwise, calls the registered callback directly.
				dm->dmub_callback[notify.type](adev, &notify);
			}
		} while (notify.pending_notification);
	}
}

/**
 * @brief Sets the clockgating state for the AMDGPU IP block.
 * @param ip_block Pointer to the AMDGPU IP block structure.
 * @param state The desired clockgating state.
 * @return Always returns 0 (TODO: implement actual clockgating).
 */
static int dm_set_clockgating_state(struct amdgpu_ip_block *ip_block,
		  enum amd_clockgating_state state)
{
	return 0;
}

/**
 * @brief Sets the powergating state for the AMDGPU IP block.
 * @param ip_block Pointer to the AMDGPU IP block structure.
 * @param state The desired powergating state.
 * @return Always returns 0 (TODO: implement actual powergating).
 */
static int dm_set_powergating_state(struct amdgpu_ip_block *ip_block,
		  enum amd_powergating_state state)
{
	return 0;
}

/* Prototypes of private functions */
/**
 * @brief Performs early initialization for the AMDGPU IP block.
 * @param ip_block Pointer to the AMDGPU IP block structure.
 * @return An integer status code.
 */
static int dm_early_init(struct amdgpu_ip_block *ip_block);

/**
 * @brief Initializes FBC (Frame Buffer Compression) for a DRM connector.
 * @param connector Pointer to the DRM connector.
 *
 * This function allocates GPU memory for FBC compressed data if the FBC
 * compressor is available and the connector is an eDP (embedded DisplayPort)
 * type. It calculates the maximum required size based on supported modes.
 */
static void amdgpu_dm_fbc_init(struct drm_connector *connector)
{
	struct amdgpu_device *adev = drm_to_adev(connector->dev);
	struct dm_compressor_info *compressor = &adev->dm.compressor;
	struct amdgpu_dm_connector *aconn = to_amdgpu_dm_connector(connector);
	struct drm_display_mode *mode;
	unsigned long max_size = 0;

	// Block Logic: Returns if FBC compressor is not available.
	if (adev->dm.dc->fbc_compressor == NULL)
		return;

	// Block Logic: Only applies FBC initialization to eDP connectors.
	if (aconn->dc_link->connector_signal != SIGNAL_TYPE_EDP)
		return;

	// Block Logic: Returns if FBC buffer is already allocated.
	if (compressor->bo_ptr)
		return;

	// Block Logic: Iterates through display modes to find the maximum resolution for FBC allocation.
	list_for_each_entry(mode, &connector->modes, head) {
		if (max_size < (unsigned long) mode->htotal * mode->vtotal)
			max_size = (unsigned long) mode->htotal * mode->vtotal;
	}

	// Block Logic: If a maximum size is determined, allocates kernel memory for FBC.
	if (max_size) {
		int r = amdgpu_bo_create_kernel(adev, max_size * 4, PAGE_SIZE,
			    AMDGPU_GEM_DOMAIN_GTT, &compressor->bo_ptr,
			    &compressor->gpu_addr, &compressor->cpu_addr);

		// Block Logic: Handles errors during FBC memory allocation.
		if (r)
			drm_err(adev_to_drm(adev), "DM: Failed to initialize FBC\n");
		else {
			adev->dm.dc->ctx->fbc_gpu_addr = compressor->gpu_addr;
			drm_info(adev_to_drm(adev), "DM: FBC alloc %lu\n", max_size*4);
		}

	}

}

/**
 * @brief Retrieves ELD (EDID-Like Data) for an audio component.
 * @param kdev Pointer to the kernel device.
 * @param port The audio port number.
 * @param pipe The audio pipe number (unused).
 * @param enabled Output parameter indicating if audio is enabled.
 * @param buf Buffer to store the ELD.
 * @param max_bytes Maximum number of bytes to copy to the buffer.
 * @return The size of the copied ELD, or 0 if disabled/error.
 *
 * This function is part of the `drm_audio_component_ops` and provides
 * ELD data to the kernel audio subsystem. It locks relevant mutexes
 * to ensure thread safety during ELD access.
 */
static int amdgpu_dm_audio_component_get_eld(struct device *kdev, int port,
					  int pipe, bool *enabled,
					  unsigned char *buf, int max_bytes)
{
	struct drm_device *dev = dev_get_drvdata(kdev);
	struct amdgpu_device *adev = drm_to_adev(dev);
	struct drm_connector *connector;
	struct drm_connector_list_iter conn_iter;
	struct amdgpu_dm_connector *aconnector;
	int ret = 0;

	// Functional Utility: Initializes `enabled` to false.
	*enabled = false;

	// Block Logic: Acquires a mutex to protect audio-related data.
	mutex_lock(&adev->dm.audio_lock);

	// Block Logic: Iterates through DRM connectors to find the one matching the audio port.
	drm_connector_list_iter_begin(dev, &conn_iter);
	drm_for_each_connector_iter(connector, &conn_iter) {

		// Block Logic: Skips writeback connectors.
		if (connector->connector_type == DRM_MODE_CONNECTOR_WRITEBACK)
			continue;

		aconnector = to_amdgpu_dm_connector(connector);
		// Block Logic: Checks if the connector's audio instance matches the target port.
		if (aconnector->audio_inst != port)
			continue;

		// Block Logic: If a match is found, marks audio as enabled, locks ELD mutex, copies ELD.
		*enabled = true;
		mutex_lock(&connector->eld_mutex);
		ret = drm_eld_size(connector->eld);
		memcpy(buf, connector->eld, min(max_bytes, ret));
		mutex_unlock(&connector->eld_mutex);

		break;
	}
	drm_connector_list_iter_end(&conn_iter);

	mutex_unlock(&adev->dm.audio_lock);

	DRM_DEBUG_KMS("Get ELD : idx=%d ret=%d en=%d\n", port, ret, *enabled);

	return ret;
}

// Functional Utility: Defines the DRM audio component operations for AMDGPU DM.
static const struct drm_audio_component_ops amdgpu_dm_audio_component_ops = {
	.get_eld = amdgpu_dm_audio_component_get_eld,
};

/**
 * @brief Binds the AMDGPU DM audio component to a kernel device.
 * @param kdev Pointer to the kernel device.
 * @param hda_kdev Pointer to the HDA (High Definition Audio) kernel device (unused).
 * @param data Pointer to the DRM audio component data.
 * @return 0 on success.
 *
 * This function registers the audio component operations with the DRM
 * audio component for the AMDGPU device.
 */
static int amdgpu_dm_audio_component_bind(struct device *kdev,
				       struct device *hda_kdev, void *data)
{
	struct drm_device *dev = dev_get_drvdata(kdev);
	struct amdgpu_device *adev = drm_to_adev(dev);
	struct drm_audio_component *acomp = data;

	// Functional Utility: Assigns the audio component operations.
	acomp->ops = &amdgpu_dm_audio_component_ops;
	acomp->dev = kdev;
	adev->dm.audio_component = acomp;

	return 0;
}

/**
 * @brief Unbinds the AMDGPU DM audio component from a kernel device.
 * @param kdev Pointer to the kernel device.
 * @param hda_kdev Pointer to the HDA kernel device (unused).
 * @param data Pointer to the DRM audio component data.
 *
 * This function unregisters the audio component operations from the DRM
 * audio component.
 */
static void amdgpu_dm_audio_component_unbind(struct device *kdev,
					  struct device *hda_kdev, void *data)
{
	struct amdgpu_device *adev = drm_to_adev(dev_get_drvdata(kdev));
	struct drm_audio_component *acomp = data;

	// Functional Utility: Clears the audio component operations.
	acomp->ops = NULL;
	acomp->dev = NULL;
	adev->dm.audio_component = NULL;
}

// Functional Utility: Defines the component operations for binding/unbinding the audio component.
static const struct component_ops amdgpu_dm_audio_component_bind_ops = {
	.bind	= amdgpu_dm_audio_component_bind,
	.unbind	= amdgpu_dm_audio_component_unbind,
};

/**
 * @brief Initializes the AMDGPU DM audio subsystem.
 * @param adev Pointer to the AMDGPU device structure.
 * @return 0 on success, error code on failure.
 *
 * This function sets up the audio infrastructure for the AMDGPU DM,
 * including initializing audio pins and registering the audio component.
 */
static int amdgpu_dm_audio_init(struct amdgpu_device *adev)
{
	int i, ret;

	// Block Logic: Returns if AMDGPU audio is globally disabled.
	if (!amdgpu_audio)
		return 0;

	// Functional Utility: Enables audio and sets the number of audio pins from DC resources.
	adev->mode_info.audio.enabled = true;
	adev->mode_info.audio.num_pins = adev->dm.dc->res_pool->audio_count;

	// Block Logic: Initializes each audio pin's state.
	for (i = 0; i < adev->mode_info.audio.num_pins; i++) {
		adev->mode_info.audio.pin[i].channels = -1;
		adev->mode_info.audio.pin[i].rate = -1;
		adev->mode_info.audio.pin[i].bits_per_sample = -1;
		adev->mode_info.audio.pin[i].status_bits = 0;
		adev->mode_info.audio.pin[i].category_code = 0;
		adev->mode_info.audio.pin[i].connected = false;
		adev->mode_info.audio.pin[i].id =
			adev->dm.dc->res_pool->audios[i]->inst;
		adev->mode_info.audio.pin[i].offset = 0;
	}

	// Functional Utility: Adds the audio component to the device.
	ret = component_add(adev->dev, &amdgpu_dm_audio_component_bind_ops);
	// Block Logic: Handles errors during component addition.
	if (ret < 0)
		return ret;

	// Functional Utility: Marks audio component as registered.
	adev->dm.audio_registered = true;

	return 0;
}

/**
 * @brief Finalizes the AMDGPU DM audio subsystem.
 * @param adev Pointer to the AMDGPU device structure.
 *
 * This function cleans up the audio infrastructure, unregistering
 * the audio component and disabling audio.
 */
static void amdgpu_dm_audio_fini(struct amdgpu_device *adev)
{
	// Block Logic: Returns if AMDGPU audio is globally disabled.
	if (!amdgpu_audio)
		return;

	// Block Logic: Returns if audio is not enabled.
	if (!adev->mode_info.audio.enabled)
		return;

	// Block Logic: If audio component is registered, removes it.
	if (adev->dm.audio_registered) {
		component_del(adev->dev, &amdgpu_dm_audio_component_bind_ops);
		adev->dm.audio_registered = false;
	}

	/* TODO: Disable audio? */

	// Functional Utility: Disables audio.
	adev->mode_info.audio.enabled = false;
}

/**
 * @brief Notifies the audio component about ELD (EDID-Like Data) changes.
 * @param adev Pointer to the AMDGPU device structure.
 * @param pin The audio pin index.
 *
 * This function triggers an ELD notification to the registered DRM audio
 * component, indicating that ELD data for a specific pin has changed.
 */
static  void amdgpu_dm_audio_eld_notify(struct amdgpu_device *adev, int pin)
{
	struct drm_audio_component *acomp = adev->dm.audio_component;

	// Block Logic: If an audio component is available and has pin ELD notify operations, call it.
	if (acomp && acomp->audio_ops && acomp->audio_ops->pin_eld_notify) {
		DRM_DEBUG_KMS("Notify ELD: %d\n", pin);

		acomp->audio_ops->pin_eld_notify(acomp->audio_ops->audio_ptr,
						 pin, -1);
	}
}

/**
 * @brief Initializes the DMUB (Display Microcontroller Unit Bypass) hardware.
 * @param adev Pointer to the AMDGPU device structure.
 * @return 0 on success, error code on failure.
 *
 * This function loads the DMUB firmware, initializes the DMUB service,
 * and sets up its hardware parameters. It handles different firmware
 * loading types (PSP or backdoor) and initializes related display components.
 */
static int dm_dmub_hw_init(struct amdgpu_device *adev)
{
	const struct dmcub_firmware_header_v1_0 *hdr;
	struct dmub_srv *dmub_srv = adev->dm.dmub_srv;
	struct dmub_srv_fb_info *fb_info = adev->dm.dmub_fb_info;
	const struct firmware *dmub_fw = adev->dm.dmub_fw;
	struct dmcu *dmcu = adev->dm.dc->res_pool->dmcu;
	struct abm *abm = adev->dm.dc->res_pool->abm;
	struct dc_context *ctx = adev->dm.dc->ctx;
	struct dmub_srv_hw_params hw_params;
	enum dmub_status status;
	const unsigned char *fw_inst_const, *fw_bss_data;
	u32 i, fw_inst_const_size, fw_bss_data_size;
	bool has_hw_support;

	// Block Logic: Returns early if DMUB is not supported on the ASIC.
	if (!dmub_srv)
		/* DMUB isn't supported on the ASIC. */
		return 0;

	// Block Logic: Returns error if framebuffer info for DMUB service is missing.
	if (!fb_info) {
		drm_err(adev_to_drm(adev), "No framebuffer info for DMUB service.\n");
		return -EINVAL;
	}

	// Block Logic: Returns error if DMUB firmware is not provided.
	if (!dmub_fw) {
		/* Firmware required for DMUB support. */
		drm_err(adev_to_drm(adev), "No firmware provided for DMUB.\n");
		return -EINVAL;
	}

	/* initialize register offsets for ASICs with runtime initialization available */
	// Block Logic: Initializes DMUB register offsets if runtime initialization is available.
	if (dmub_srv->hw_funcs.init_reg_offsets)
		dmub_srv->hw_funcs.init_reg_offsets(dmub_srv, ctx);

	// Functional Utility: Checks for hardware support for DMUB.
	status = dmub_srv_has_hw_support(dmub_srv, &has_hw_support);
	// Block Logic: Handles errors during hardware support check.
	if (status != DMUB_STATUS_OK) {
		drm_err(adev_to_drm(adev), "Error checking HW support for DMUB: %d\n", status);
		return -EINVAL;
	}

	// Block Logic: Returns if DMUB is explicitly unsupported on the ASIC.
	if (!has_hw_support) {
		drm_info(adev_to_drm(adev), "DMUB unsupported on ASIC\n");
		return 0;
	}

	/* Reset DMCUB if it was previously running - before we overwrite its memory. */
	// Functional Utility: Resets the DMUB hardware if it was running.
	status = dmub_srv_hw_reset(dmub_srv);
	// Block Logic: Logs a warning if DMUB hardware reset fails.
	if (status != DMUB_STATUS_OK)
		drm_warn(adev_to_drm(adev), "Error resetting DMUB HW: %d\n", status);

	// Functional Utility: Retrieves the DMUB firmware header.
	hdr = (const struct dmcub_firmware_header_v1_0 *)dmub_fw->data;

	fw_inst_const = dmub_fw->data +
			le32_to_cpu(hdr->header.ucode_array_offset_bytes) +
			PSP_HEADER_BYTES;

	fw_bss_data = dmub_fw->data +
		      le32_to_cpu(hdr->header.ucode_array_offset_bytes) +
		      le32_to_cpu(hdr->inst_const_bytes);

	/* Copy firmware and bios info into FB memory. */
	fw_inst_const_size = le32_to_cpu(hdr->inst_const_bytes) -
			     PSP_HEADER_BYTES - PSP_FOOTER_BYTES;

	fw_bss_data_size = le32_to_cpu(hdr->bss_data_bytes);

	/* if adev->firmware.load_type == AMDGPU_FW_LOAD_PSP,
	 * amdgpu_ucode_init_single_fw will load dmub firmware
	 * fw_inst_const part to cw0; otherwise, the firmware back door load
	 * will be done by dm_dmub_hw_init
	 */
	// Block Logic: Copies firmware instruction and constant data to framebuffer memory, depending on load type.
	if (adev->firmware.load_type != AMDGPU_FW_LOAD_PSP) {
		memcpy(fb_info->fb[DMUB_WINDOW_0_INST_CONST].cpu_addr, fw_inst_const,
				fw_inst_const_size);
	}

	// Block Logic: Copies BSS data if available.
	if (fw_bss_data_size)
		memcpy(fb_info->fb[DMUB_WINDOW_2_BSS_DATA].cpu_addr,
		       fw_bss_data, fw_bss_data_size);

	/* Copy firmware bios info into FB memory. */
	// Functional Utility: Copies VBIOS data to framebuffer memory.
	memcpy(fb_info->fb[DMUB_WINDOW_3_VBIOS].cpu_addr, adev->bios,
	       adev->bios_size);

	/* Reset regions that need to be reset. */
	// Block Logic: Resets various DMUB framebuffer memory regions to zero.
	memset(fb_info->fb[DMUB_WINDOW_4_MAILBOX].cpu_addr, 0,
	fb_info->fb[DMUB_WINDOW_4_MAILBOX].size);

	memset(fb_info->fb[DMUB_WINDOW_5_TRACEBUFF].cpu_addr, 0,
	       fb_info->fb[DMUB_WINDOW_5_TRACEBUFF].size);

	memset(fb_info->fb[DMUB_WINDOW_6_FW_STATE].cpu_addr, 0,
	       fb_info->fb[DMUB_WINDOW_6_FW_STATE].size);

	memset(fb_info->fb[DMUB_WINDOW_SHARED_STATE].cpu_addr, 0,
	       fb_info->fb[DMUB_WINDOW_SHARED_STATE].size);

	/* Initialize hardware. */
	memset(&hw_params, 0, sizeof(hw_params));
	hw_params.fb_base = adev->gmc.fb_start;
	hw_params.fb_offset = adev->vm_manager.vram_base_offset;

	/* backdoor load firmware and trigger dmub running */
	// Block Logic: Sets flag for backdoor firmware load if not PSP.
	if (adev->firmware.load_type != AMDGPU_FW_LOAD_PSP)
		hw_params.load_inst_const = true;

	// Block Logic: Sets PSP version if DMCU is present.
	if (dmcu)
		hw_params.psp_version = dmcu->psp_version;

	// Block Logic: Populates framebuffer information for DMUB hardware parameters.
	for (i = 0; i < fb_info->num_fb; ++i)
		hw_params.fb[i] = &fb_info->fb[i];

	// Block Logic: Sets DPIA (DisplayPort Intermediate Adaptor) support based on IP version.
	switch (amdgpu_ip_version(adev, DCE_HWIP, 0)) {
	case IP_VERSION(3, 1, 3):
	case IP_VERSION(3, 1, 4):
	case IP_VERSION(3, 5, 0):
	case IP_VERSION(3, 5, 1):
	case IP_VERSION(3, 6, 0):
	case IP_VERSION(4, 0, 1):
		hw_params.dpia_supported = true;
		hw_params.disable_dpia = adev->dm.dc->debug.dpia_debug.bits.disable_dpia;
		break;
	default:
		break;
	}

	// Block Logic: Configures IPS (Idle Power Saving) parameters and PHY SSC (Spread Spectrum Clocking) for specific IP versions.
	switch (amdgpu_ip_version(adev, DCE_HWIP, 0)) {
	case IP_VERSION(3, 5, 0):
	case IP_VERSION(3, 5, 1):
	case IP_VERSION(3, 6, 0):
		hw_params.ips_sequential_ono = adev->external_rev_id > 0x10;
		hw_params.lower_hbr3_phy_ssc = true;
		break;
	default:
		break;
	}

	// Functional Utility: Initializes the DMUB hardware.
	status = dmub_srv_hw_init(dmub_srv, &hw_params);
	// Block Logic: Handles errors during DMUB hardware initialization.
	if (status != DMUB_STATUS_OK) {
		drm_err(adev_to_drm(adev), "Error initializing DMUB HW: %d\n", status);
		return -EINVAL;
	}

	/* Wait for firmware load to finish. */
	// Functional Utility: Waits for DMUB firmware to auto-load.
	status = dmub_srv_wait_for_auto_load(dmub_srv, 100000);
	// Block Logic: Logs a warning if DMUB auto-load times out.
	if (status != DMUB_STATUS_OK)
		drm_warn(adev_to_drm(adev), "Wait for DMUB auto-load failed: %d\n", status);

	/* Init DMCU and ABM if available. */
	// Block Logic: Initializes DMCU and ABM (Adaptive Backlight Management) if they are available.
	if (dmcu && abm) {
		dmcu->funcs->dmcu_init(dmcu);
		abm->dmcu_is_running = dmcu->funcs->is_dmcu_initialized(dmcu);
	}

	// Block Logic: Creates and assigns the DC DMUB service if not already present.
	if (!adev->dm.dc->ctx->dmub_srv)
		adev->dm.dc->ctx->dmub_srv = dc_dmub_srv_create(adev->dm.dc, dmub_srv);
	// Block Logic: Handles errors if DC DMUB service allocation fails.
	if (!adev->dm.dc->ctx->dmub_srv) {
		drm_err(adev_to_drm(adev), "Couldn't allocate DC DMUB server!\n");
		return -ENOMEM;
	}

	drm_info(adev_to_drm(adev), "DMUB hardware initialized: version=0x%08X\n",
		 adev->dm.dmcub_fw_version);

	/* Keeping sanity checks off if
	 * DCN31 >= 4.0.59.0
	 * DCN314 >= 8.0.16.0
	 * Otherwise, turn on sanity checks
	 */
	// Block Logic: Configures debug sanity checks based on DMUB firmware version and IP version.
	switch (amdgpu_ip_version(adev, DCE_HWIP, 0)) {
	case IP_VERSION(3, 1, 2):
	case IP_VERSION(3, 1, 3):
		if (adev->dm.dmcub_fw_version &&
			adev->dm.dmcub_fw_version >= DMUB_FW_VERSION(4, 0, 0) &&
			adev->dm.dmcub_fw_version < DMUB_FW_VERSION(4, 0, 59))
				adev->dm.dc->debug.sanity_checks = true;
		break;
	case IP_VERSION(3, 1, 4):
		if (adev->dm.dmcub_fw_version &&
			adev->dm.dmcub_fw_version >= DMUB_FW_VERSION(4, 0, 0) &&
			adev->dm.dmcub_fw_version < DMUB_FW_VERSION(8, 0, 16))
				adev->dm.dc->debug.sanity_checks = true;
		break;
	default:
		break;
	}

	return 0;
}

/**
 * @brief Resumes the DMUB hardware after suspend.
 * @param adev Pointer to the AMDGPU device structure.
 *
 * This function handles the resumption of DMUB hardware, either by waiting
 * for auto-load completion if already initialized, or by performing a full
 * hardware initialization if not.
 */
static void dm_dmub_hw_resume(struct amdgpu_device *adev)
{
	struct dmub_srv *dmub_srv = adev->dm.dmub_srv;
	enum dmub_status status;
	bool init;
	int r;

	// Block Logic: Returns early if DMUB is not supported.
	if (!dmub_srv) {
		/* DMUB isn't supported on the ASIC. */
		return;
	}

	// Functional Utility: Checks if DMUB hardware has been initialized.
	status = dmub_srv_is_hw_init(dmub_srv, &init);
	// Block Logic: Logs a warning if the hardware initialization check fails.
	if (status != DMUB_STATUS_OK)
		drm_warn(adev_to_drm(adev), "DMUB hardware init check failed: %d\n", status);

	// Block Logic: If already initialized, waits for firmware auto-load. Otherwise, re-initializes.
	if (status == DMUB_STATUS_OK && init) {
		/* Wait for firmware load to finish. */
		status = dmub_srv_wait_for_auto_load(dmub_srv, 100000);
		// Block Logic: Logs a warning if auto-load times out.
		if (status != DMUB_STATUS_OK)
			drm_warn(adev_to_drm(adev), "Wait for DMUB auto-load failed: %d\n", status);
	} else {
		/* Perform the full hardware initialization. */
		r = dm_dmub_hw_init(adev);
		// Block Logic: Logs an error if full hardware initialization fails.
		if (r)
			drm_err(adev_to_drm(adev), "DMUB interface failed to initialize: status=%d\n", r);
	}
}

/**
 * @brief Reads system context information from MMHUB for DC physical address configuration.
 * @param adev Pointer to the AMDGPU device structure.
 * @param pa_config Pointer to the `dc_phy_addr_space_config` structure to populate.
 *
 * This function retrieves and configures physical address space settings
 * from the MMHUB (Memory Management Hub) for the Display Core (DC),
 * including AGP aperture, framebuffer addresses, and GART page table configurations.
 */
static void mmhub_read_system_context(struct amdgpu_device *adev, struct dc_phy_addr_space_config *pa_config)
{
	u64 pt_base;
	u32 logical_addr_low;
	u32 logical_addr_high;
	u32 agp_base, agp_bot, agp_top;
	PHYSICAL_ADDRESS_LOC page_table_start, page_table_end, page_table_base;

	// Functional Utility: Initializes the physical address configuration structure to zero.
	memset(pa_config, 0, sizeof(*pa_config));

	agp_base = 0;
	agp_bot = adev->gmc.agp_start >> 24;
	agp_top = adev->gmc.agp_end >> 24;

	/* AGP aperture is disabled */
	// Block Logic: Determines logical address range based on whether AGP aperture is enabled.
	if (agp_bot > agp_top) {
		logical_addr_low = adev->gmc.fb_start >> 18;
		// Block Logic: Adjusts high logical address for specific APU families due to hardware quirks.
		if (adev->apu_flags & (AMD_APU_IS_RAVEN2 |
				       AMD_APU_IS_RENOIR |
				       AMD_APU_IS_GREEN_SARDINE))
			/*
			 * Raven2 has a HW issue that it is unable to use the vram which
			 * is out of MC_VM_SYSTEM_APERTURE_HIGH_ADDR. So here is the
			 * workaround that increase system aperture high address (add 1)
			 * to get rid of the VM fault and hardware hang.
			 */
			logical_addr_high = (adev->gmc.fb_end >> 18) + 0x1;
		else
			logical_addr_high = adev->gmc.fb_end >> 18;
	} else {
		logical_addr_low = min(adev->gmc.fb_start, adev->gmc.agp_start) >> 18;
		// Block Logic: Adjusts high logical address for specific APU families when AGP is active.
		if (adev->apu_flags & (AMD_APU_IS_RAVEN2 |
				       AMD_APU_IS_RENOIR |
				       AMD_APU_IS_GREEN_SARDINE))
			/*
			 * Raven2 has a HW issue that it is unable to use the vram which
			 * is out of MC_VM_SYSTEM_APERTURE_HIGH_ADDR. So here is the
			 * workaround that increase system aperture high address (add 1)
			 * to get rid of the VM fault and hardware hang.
			 */
			logical_addr_high = max((adev->gmc.fb_end >> 18) + 0x1, adev->gmc.agp_end >> 18);
		else
			logical_addr_high = max(adev->gmc.fb_end, adev->gmc.agp_end) >> 18;
	}

	// Functional Utility: Retrieves the page table base address.
	pt_base = amdgpu_gmc_pd_addr(adev->gart.bo);

	// Functional Utility: Populates GART page table start, end, and base addresses.
	page_table_start.high_part = upper_32_bits(adev->gmc.gart_start >>
						   AMDGPU_GPU_PAGE_SHIFT);
	page_table_start.low_part = lower_32_bits(adev->gmc.gart_start >>
						  AMDGPU_GPU_PAGE_SHIFT);
	page_table_end.high_part = upper_32_bits(adev->gmc.gart_end >>
						 AMDGPU_GPU_PAGE_SHIFT);
	page_table_end.low_part = lower_32_bits(adev->gmc.gart_end >>
						AMDGPU_GPU_PAGE_SHIFT);
	page_table_base.high_part = upper_32_bits(pt_base);
	page_table_base.low_part = lower_32_bits(pt_base);

	// Functional Utility: Populates system aperture configuration.
	pa_config->system_aperture.start_addr = (uint64_t)logical_addr_low << 18;
	pa_config->system_aperture.end_addr = (uint64_t)logical_addr_high << 18;

	pa_config->system_aperture.agp_base = (uint64_t)agp_base << 24;
	pa_config->system_aperture.agp_bot = (uint64_t)agp_bot << 24;
	pa_config->system_aperture.agp_top = (uint64_t)agp_top << 24;

	pa_config->system_aperture.fb_base = adev->gmc.fb_start;
	pa_config->system_aperture.fb_offset = adev->vm_manager.vram_base_offset;
	pa_config->system_aperture.fb_top = adev->gmc.fb_end;

	pa_config->gart_config.page_table_start_addr = page_table_start.quad_part << 12;
	pa_config->gart_config.page_table_end_addr = page_table_end.quad_part << 12;
	pa_config->gart_config.page_table_base_addr = page_table_base.quad_part;

	// Functional Utility: Sets HVM (Hardware Virtual Machine) enabled status.
	pa_config->is_hvm_enabled = adev->mode_info.gpu_vm_support;

}

/**
 * @brief Forces a specific state on a DRM connector.
 * @param aconnector Pointer to the AMDGPU DM connector object.
 * @param force_state The desired force state for the connector.
 *
 * This function is used to programmatically change the force state of a
 * connector and trigger a hotplug event, for example, during automated testing.
 */
static void force_connector_state(
	struct amdgpu_dm_connector *aconnector,
	enum drm_connector_force force_state)
{
	struct drm_connector *connector = &aconnector->base;

	// Block Logic: Acquires mode_config mutex to safely update connector force state.
	mutex_lock(&connector->dev->mode_config.mutex);
	aconnector->base.force = force_state;
	mutex_unlock(&connector->dev->mode_config.mutex);

	// Block Logic: Acquires HPD lock and triggers a KMS helper hotplug event.
	mutex_lock(&aconnector->hpd_lock);
	drm_kms_helper_connector_hotplug_event(connector);
	mutex_unlock(&aconnector->hpd_lock);
}

/**
 * @brief Work function for handling HPD RX IRQ (Receiver Interrupt Request) offload.
 * @param work Pointer to the `work_struct` associated with the offload task.
 *
 * This function processes HPD RX IRQ events that have been offloaded to a workqueue.
 * It detects connection type changes, handles MST (Multi-Stream Transport) sideband
 * messages, and manages automated display tests, including link loss detection.
 */
static void dm_handle_hpd_rx_offload_work(struct work_struct *work)
{
	struct hpd_rx_irq_offload_work *offload_work;
	struct amdgpu_dm_connector *aconnector;
	struct dc_link *dc_link;
	struct amdgpu_device *adev;
	enum dc_connection_type new_connection_type = dc_connection_none;
	unsigned long flags;
	union test_response test_response;

	// Functional Utility: Initializes test_response to zero.
	memset(&test_response, 0, sizeof(test_response));

	// Functional Utility: Casts the generic `work_struct` to the specific `hpd_rx_irq_offload_work`.
	offload_work = container_of(work, struct hpd_rx_irq_offload_work, work);
	aconnector = offload_work->offload_wq->aconnector;
	adev = offload_work->adev;

	// Block Logic: Handles null `aconnector` by logging an error and skipping.
	if (!aconnector) {
		drm_err(adev_to_drm(adev), "Can't retrieve aconnector in hpd_rx_irq_offload_work");
		goto skip;
	}

	dc_link = aconnector->dc_link;

	// Block Logic: Detects new connection type by locking HPD mutex.
	mutex_lock(&aconnector->hpd_lock);
	if (!dc_link_detect_connection_type(dc_link, &new_connection_type))
		drm_err(adev_to_drm(adev), "KMS: Failed to detect connector\n");
	mutex_unlock(&aconnector->hpd_lock);

	// Block Logic: If no new connection is detected, skips further processing.
	if (new_connection_type == dc_connection_none)
		goto skip;

	// Block Logic: Skips if AMDGPU device is in reset state.
	if (amdgpu_in_reset(adev))
		goto skip;

	// Block Logic: Handles MST sideband message ready event.
	if (offload_work->data.bytes.device_service_irq.bits.UP_REQ_MSG_RDY ||
		offload_work->data.bytes.device_service_irq.bits.DOWN_REP_MSG_RDY) {
		dm_handle_mst_sideband_msg_ready_event(&aconnector->mst_mgr, DOWN_OR_UP_MSG_RDY_EVENT);
		spin_lock_irqsave(&offload_work->offload_wq->offload_lock, flags);
		offload_work->offload_wq->is_handling_mst_msg_rdy_event = false;
		spin_unlock_irqrestore(&offload_work->offload_wq->offload_lock, flags);
		goto skip;
	}

	// Block Logic: Acquires DC lock for Display Core operations.
	mutex_lock(&adev->dm.dc_lock);
	// Block Logic: If automated test is indicated, handles the test.
	if (offload_work->data.bytes.device_service_irq.bits.AUTOMATED_TEST) {
		dc_link_dp_handle_automated_test(dc_link);

		// Block Logic: If timing changed during automated test, forces connector disconnect/reconnect.
		if (aconnector->timing_changed) {
			/* force connector disconnect and reconnect */
			force_connector_state(aconnector, DRM_FORCE_OFF);
			msleep(100);
			force_connector_state(aconnector, DRM_FORCE_UNSPECIFIED);
		}

		// Functional Utility: Sets ACK bit in test response.
		test_response.bits.ACK = 1;

		// Functional Utility: Writes test response to DPCD (DisplayPort Configuration Data).
		core_link_write_dpcd(
		dc_link,
		DP_TEST_RESPONSE,
		&test_response.raw,
		sizeof(test_response));
	} else if ((dc_link->connector_signal != SIGNAL_TYPE_EDP) &&
			dc_link_check_link_loss_status(dc_link, &offload_work->data) &&
			dc_link_dp_allow_hpd_rx_irq(dc_link)) {
		/* offload_work->data is from handle_hpd_rx_irq->
		 * schedule_hpd_rx_offload_work.this is defer handle
		 * for hpd short pulse. upon here, link status may be
		 * changed, need get latest link status from dpcd
		 * registers. if link status is good, skip run link
		 * training again.
		 */
		union hpd_irq_data irq_data;

		// Functional Utility: Initializes IRQ data to zero.
		memset(&irq_data, 0, sizeof(irq_data));

		/* before dc_link_dp_handle_link_loss, allow new link lost handle
		 * request be added to work queue if link lost at end of dc_link_
		 * dp_handle_link_loss
		 */
		// Block Logic: Resets link loss handling flag before checking link status.
		spin_lock_irqsave(&offload_work->offload_wq->offload_lock, flags);
		offload_work->offload_wq->is_handling_link_loss = false;
		spin_unlock_irqrestore(&offload_work->offload_wq->offload_lock, flags);

		// Block Logic: If link loss is still detected after reading latest DPCD data, handles it.
		if ((dc_link_dp_read_hpd_rx_irq_data(dc_link, &irq_data) == DC_OK) &&
			dc_link_check_link_loss_status(dc_link, &irq_data))
			dc_link_dp_handle_link_loss(dc_link);
	}
	mutex_unlock(&adev->dm.dc_lock);

skip:
	// Functional Utility: Frees the offload work structure.
	kfree(offload_work);

}

/**
 * @brief Creates a workqueue for HPD RX IRQ offloading.
 * @param adev Pointer to the AMDGPU device structure.
 * @return Pointer to an array of `hpd_rx_irq_offload_work_queue` structures, or NULL on failure.
 *
 * This function allocates and initializes a single-threaded workqueue for each
 * possible display link, used for asynchronous processing of HPD RX IRQ events.
 */
static struct hpd_rx_irq_offload_work_queue *hpd_rx_irq_create_workqueue(struct amdgpu_device *adev)
{
	struct dc *dc = adev->dm.dc;
	int max_caps = dc->caps.max_links;
	int i = 0;
	struct hpd_rx_irq_offload_work_queue *hpd_rx_offload_wq = NULL;

	// Functional Utility: Allocates memory for an array of workqueue structures.
	hpd_rx_offload_wq = kcalloc(max_caps, sizeof(*hpd_rx_offload_wq), GFP_KERNEL);

	// Block Logic: Handles memory allocation failure.
	if (!hpd_rx_offload_wq)
		return NULL;

	// Block Logic: Initializes a single-threaded workqueue for each display link.
	for (i = 0; i < max_caps; i++) {
		hpd_rx_offload_wq[i].wq =
				    create_singlethread_workqueue("amdgpu_dm_hpd_rx_offload_wq");

		// Block Logic: Handles workqueue creation failure and cleans up.
		if (hpd_rx_offload_wq[i].wq == NULL) {
			drm_err(adev_to_drm(adev), "create amdgpu_dm_hpd_rx_offload_wq fail!");
			goto out_err;
		}

		// Functional Utility: Initializes a spinlock for each workqueue.
		spin_lock_init(&hpd_rx_offload_wq[i].offload_lock);
	}

	return hpd_rx_offload_wq;

out_err:
	// Block Logic: Cleans up allocated workqueues in case of an error.
	for (i = 0; i < max_caps; i++) {
		if (hpd_rx_offload_wq[i].wq)
			destroy_workqueue(hpd_rx_offload_wq[i].wq);
	}
	kfree(hpd_rx_offload_wq);
	return NULL;
}

/**
 * @brief Structure defining a quirk for stutter mode on specific AMDGPU devices.
 * @var amdgpu_stutter_quirk::chip_vendor
 *   PCI vendor ID of the GPU.
 * @var amdgpu_stutter_quirk::chip_device
 *   PCI device ID of the GPU.
 * @var amdgpu_stutter_quirk::subsys_vendor
 *   PCI subsystem vendor ID.
 * @var amdgpu_stutter_quirk::subsys_device
 *   PCI subsystem device ID.
 * @var amdgpu_stutter_quirk::revision
 *   PCI revision ID.
 */
struct amdgpu_stutter_quirk {
	u16 chip_vendor;
	u16 chip_device;
	u16 subsys_vendor;
	u16 subsys_device;
	u8 revision;
};

// Functional Utility: Array of known AMDGPU stutter quirks.
static const struct amdgpu_stutter_quirk amdgpu_stutter_quirk_list[] = {
	/* https://bugzilla.kernel.org/show_bug.cgi?id=214417 */
	{ 0x1002, 0x15dd, 0x1002, 0x15dd, 0xc8 },
	{ 0, 0, 0, 0, 0 }, // Sentinel for end of list
};

/**
 * @brief Determines if stutter mode should be disabled for a given PCI device.
 * @param pdev Pointer to the PCI device structure.
 * @return True if stutter mode should be disabled due to a known quirk, false otherwise.
 *
 * This function checks the PCI device's vendor, device, subsystem, and revision IDs
 * against a predefined list of quirks that require stutter mode to be disabled.
 */
static bool dm_should_disable_stutter(struct pci_dev *pdev)
{
	const struct amdgpu_stutter_quirk *p = amdgpu_stutter_quirk_list;

	// Block Logic: Iterates through the list of stutter quirks.
	while (p && p->chip_device != 0) {
		// Block Logic: Compares PCI device identification fields with the quirk entry.
		if (pdev->vendor == p->chip_vendor &&
		    pdev->device == p->chip_device &&
		    pdev->subsystem_vendor == p->subsys_vendor &&
		    pdev->subsystem_device == p->subsys_device &&
		    pdev->revision == p->revision) {
			return true;
		}
		// Functional Utility: Moves to the next quirk entry.
		++p;
	}
	return false;
}

/**
 * @brief Allocates GPU memory for the Display Core.
 * @param adev Pointer to the AMDGPU device structure.
 * @param type Type of GPU memory allocation (GART or VRAM).
 * @param size The size of memory to allocate in bytes.
 * @param addr Output parameter for the GPU physical address of the allocated memory.
 * @return A kernel virtual pointer to the allocated memory, or NULL on failure.
 *
 * This function allocates GPU memory (either GART-backed or VRAM-backed)
 * and associates it with a `dal_allocation` structure for tracking.
 */
void*
dm_allocate_gpu_mem(
		struct amdgpu_device *adev,
		enum dc_gpu_mem_alloc_type type,
		size_t size,
		long long *addr)
{
	struct dal_allocation *da;
	u32 domain = (type == DC_MEM_ALLOC_TYPE_GART) ?
		AMDGPU_GEM_DOMAIN_GTT : AMDGPU_GEM_DOMAIN_VRAM;
	int ret;

	// Functional Utility: Allocates memory for a `dal_allocation` structure.
	da = kzalloc(sizeof(struct dal_allocation), GFP_KERNEL);
	// Block Logic: Handles allocation failure.
	if (!da)
		return NULL;

	// Functional Utility: Creates a kernel-managed AMDGPU buffer object.
	ret = amdgpu_bo_create_kernel(adev, size, PAGE_SIZE,
				      domain, &da->bo,
				      &da->gpu_addr, &da->cpu_ptr);

	// Functional Utility: Stores the GPU address.
	*addr = da->gpu_addr;

	// Block Logic: Handles buffer object creation failure and cleans up.
	if (ret) {
		kfree(da);
		return NULL;
	}

	/* add da to list in dm */
	// Functional Utility: Adds the new allocation to the DM's list of allocations.
	list_add(&da->list, &adev->dm.da_list);

	return da->cpu_ptr;
}

/**
 * @brief Frees GPU memory previously allocated for the Display Core.
 * @param adev Pointer to the AMDGPU device structure.
 * @param type Type of GPU memory allocation (unused).
 * @param pvMem Kernel virtual pointer to the memory to free.
 *
 * This function searches for the `dal_allocation` associated with `pvMem`
 * and frees the corresponding AMDGPU buffer object and the allocation structure.
 */
void
dm_free_gpu_mem(
		struct amdgpu_device *adev,
		enum dc_gpu_mem_alloc_type type,
		void *pvMem)
{
	struct dal_allocation *da;

	/* walk the da list in DM */
	// Block Logic: Iterates through the list of DM allocations to find a match for `pvMem`.
	list_for_each_entry(da, &adev->dm.da_list, list) {
		// Block Logic: If a matching CPU pointer is found, frees the buffer object and removes it from the list.
		if (pvMem == da->cpu_ptr) {
			amdgpu_bo_free_kernel(&da->bo, &da->gpu_addr, &da->cpu_ptr);
			list_del(&da->list);
			kfree(da);
			break;
		}
	}

}

/**
 * @brief Sends a VBIOS (Video BIOS) GPINT (General Purpose Interrupt) command to DMUB.
 * @param adev Pointer to the AMDGPU device structure.
 * @param command_code The GPINT command code.
 * @param param A 16-bit parameter for the command.
 * @param timeout_us Timeout in microseconds for command acknowledgment.
 * @return `DMUB_STATUS_OK` on success, `DMUB_STATUS_TIMEOUT` if the command times out.
 *
 * This function sends a command to the DMUB via a general-purpose interrupt
 * and waits for its acknowledgment, with a specified timeout.
 */
static enum dmub_status
dm_dmub_send_vbios_gpint_command(struct amdgpu_device *adev,
				 enum dmub_gpint_command command_code,
				 uint16_t param,
				 uint32_t timeout_us)
{
	union dmub_gpint_data_register reg, test;
	uint32_t i;

	/* Assume that VBIOS DMUB is ready to take commands */

	// Functional Utility: Sets up the GPINT data register with command, parameter, and status.
	reg.bits.status = 1;
	reg.bits.command_code = command_code;
	reg.bits.param = param;

	// Functional Utility: Writes the command to a CGS (Core Graphics System) register.
	cgs_write_register(adev->dm.cgs_device, 0x34c0 + 0x01f8, reg.all);

	// Block Logic: Waits for acknowledgment from DMUB by polling a register.
	for (i = 0; i < timeout_us; ++i) {
		udelay(1); // Delay for 1 microsecond.

		/* Check if our GPINT got acked */
		reg.bits.status = 0; // Clear status for comparison.
		test = (union dmub_gpint_data_register)
			cgs_read_register(adev->dm.cgs_device, 0x34c0 + 0x01f8);

		// Block Logic: If the register matches the expected acknowledgment, command is successful.
		if (test.all == reg.all)
			return DMUB_STATUS_OK;
	}

	// Functional Utility: Returns timeout status if acknowledgment is not received within the timeout.
	return DMUB_STATUS_TIMEOUT;
}

/**
 * @brief Retrieves the VBIOS bounding box from DMUB.
 * @param adev Pointer to the AMDGPU device structure.
 * @return A kernel virtual pointer to the VBIOS bounding box data, or NULL on failure.
 *
 * This function allocates GPU memory for the VBIOS bounding box, sends its
 * address to DMUB, and then requests DMUB to copy the bounding box data.
 */
static void *dm_dmub_get_vbios_bounding_box(struct amdgpu_device *adev)
{
	void *bb;
	long long addr;
	unsigned int bb_size;
	int i = 0;
	uint16_t chunk;
	enum dmub_gpint_command send_addrs[] = {
		DMUB_GPINT__SET_BB_ADDR_WORD0,
		DMUB_GPINT__SET_BB_ADDR_WORD1,
		DMUB_GPINT__SET_BB_ADDR_WORD2,
		DMUB_GPINT__SET_BB_ADDR_WORD3,
	};
	enum dmub_status ret;

	// Block Logic: Determines the bounding box size based on the IP version.
	switch (amdgpu_ip_version(adev, DCE_HWIP, 0)) {
	case IP_VERSION(4, 0, 1):
		bb_size = sizeof(struct dml2_soc_bb);
		break;
	default:
		return NULL;
	}

	// Functional Utility: Allocates GPU memory for the bounding box.
	bb =  dm_allocate_gpu_mem(adev,
				  DC_MEM_ALLOC_TYPE_GART,
				  bb_size,
				  &addr);
	// Block Logic: Handles allocation failure.
	if (!bb)
		return NULL;

	// Block Logic: Sends chunks of the bounding box address to DMUB via GPINT commands.
	for (i = 0; i < 4; i++) {
		/* Extract 16-bit chunk */
		chunk = ((uint64_t) addr >> (i * 16)) & 0xFFFF;
		/* Send the chunk */
		ret = dm_dmub_send_vbios_gpint_command(adev, send_addrs[i], chunk, 30000);
		// Block Logic: Handles GPINT command failure.
		if (ret != DMUB_STATUS_OK)
			goto free_bb;
	}

	/* Now ask DMUB to copy the bb */
	// Functional Utility: Sends a GPINT command to DMUB to copy the bounding box data.
	ret = dm_dmub_send_vbios_gpint_command(adev, DMUB_GPINT__BB_COPY, 1, 200000);
	// Block Logic: Handles DMUB copy command failure.
	if (ret != DMUB_STATUS_OK)
		goto free_bb;

	return bb;

free_bb:
	// Functional Utility: Frees allocated GPU memory in case of error.
	dm_free_gpu_mem(adev, DC_MEM_ALLOC_TYPE_GART, (void *) bb);
	return NULL;

}

/**
 * @brief Retrieves the default IPS (Idle Power Saving) mode for a given AMDGPU device.
 * @param adev Pointer to the AMDGPU device structure.
 * @return The default `dmub_ips_disable_type`.
 *
 * This function determines the appropriate IPS mode based on the IP version
 * of the AMDGPU device.
 */
static enum dmub_ips_disable_type dm_get_default_ips_mode(
	struct amdgpu_device *adev)
{
	enum dmub_ips_disable_type ret = DMUB_IPS_ENABLE;

	// Block Logic: Sets IPS mode based on IP version.
	switch (amdgpu_ip_version(adev, DCE_HWIP, 0)) {
	case IP_VERSION(3, 5, 0):
	case IP_VERSION(3, 6, 0):
	case IP_VERSION(3, 5, 1):
		ret =  DMUB_IPS_RCG_IN_ACTIVE_IPS2_IN_OFF;
		break;
	default:
		/* ASICs older than DCN35 do not have IPSs */
		// Block Logic: Disables IPS for older ASICs.
		if (amdgpu_ip_version(adev, DCE_HWIP, 0) < IP_VERSION(3, 5, 0))
			ret = DMUB_IPS_DISABLE_ALL;
		break;
	}

	return ret;
}

/**
 * @brief Initializes the AMDGPU Display Manager subsystem.
 * @param adev Pointer to the AMDGPU device structure.
 * @return 0 on success, error code on failure.
 *
 * This is a core initialization function that sets up the `amdgpu_display_manager`
 * structure, initializes mutexes, IRQ support, and populates `dc_init_data`
 * with ASIC-specific information before creating the Display Core (DC) instance.
 */
static int amdgpu_dm_init(struct amdgpu_device *adev)
{
	struct dc_init_data init_data;
	struct dc_callback_init init_params;
	int r;

	adev->dm.ddev = adev_to_drm(adev);
	adev->dm.adev = adev;

	/* Zero all the fields */
	// Functional Utility: Initializes structures to zero.
	memset(&init_data, 0, sizeof(init_data));
	memset(&init_params, 0, sizeof(init_params));

	// Functional Utility: Initializes mutexes for thread-safe access.
	mutex_init(&adev->dm.dpia_aux_lock);
	mutex_init(&adev->dm.dc_lock);
	mutex_init(&adev->dm.audio_lock);

	// Functional Utility: Initializes DM IRQ support.
	if (amdgpu_dm_irq_init(adev)) {
		drm_err(adev_to_drm(adev), "failed to initialize DM IRQ support.\n");
		goto error;
	}

	// Functional Utility: Populates ASIC ID information in `init_data`.
	init_data.asic_id.chip_family = adev->family;
	init_data.asic_id.pci_revision_id = adev->pdev->revision;
	init_data.asic_id.hw_internal_rev = adev->external_rev_id;
	init_data.asic_id.chip_id = adev->pdev->device;
	init_data.asic_id.vram_width = adev->gmc.vram_width;
	/* TODO: initialize init_data.asic_id.vram_type here!!!! */
	init_data.asic_id.atombios_base_address =
		adev->mode_info.atom_context->bios;

	init_data.driver = adev;

	/* cgs_device was created in dm_sw_init() */
	init_data.cgs_device = adev->dm.cgs_device;

	init_data.dce_environment = DCE_ENV_PRODUCTION_DRV;

	// Block Logic: Conditionally disables DMCU (Display Microcontroller Unit) based on IP version and firmware version.
	switch (amdgpu_ip_version(adev, DCE_HWIP, 0)) {
	case IP_VERSION(2, 1, 0):
		switch (adev->dm.dmcub_fw_version) {
		case 0: /* development */
		case 0x1: /* linux-firmware.git hash 6d9f399 */
		case 0x01000000: /* linux-firmware.git hash 9a0b0f4 */
			init_data.flags.disable_dmcu = false;
			break;
		default:
			init_data.flags.disable_dmcu = true;
		}
		break;
	case IP_VERSION(2, 0, 3):
		init_data.flags.disable_dmcu = true;
		break;
	default:
		break;
	}

	/* APU support S/G display by default except:
	 * ASICs before Carrizo,
	 * RAVEN1 (Users reported stability issue)
	 */

	// Block Logic: Configures GPU VM (Virtual Memory) support based on ASIC type and `amdgpu_sg_display` parameter.
	if (adev->asic_type < CHIP_CARRIZO) {
		init_data.flags.gpu_vm_support = false;
	} else if (adev->asic_type == CHIP_RAVEN) {
		if (adev->apu_flags & AMD_APU_IS_RAVEN)
			init_data.flags.gpu_vm_support = false;
		else
			init_data.flags.gpu_vm_support = (amdgpu_sg_display != 0);
	} else {
		if (amdgpu_ip_version(adev, DCE_HWIP, 0) == IP_VERSION(2, 0, 3))
			init_data.flags.gpu_vm_support = (amdgpu_sg_display == 1);
		else
			init_data.flags.gpu_vm_support =
				(amdgpu_sg_display != 0) && (adev->flags & AMD_IS_APU);
	}

	adev->mode_info.gpu_vm_support = init_data.flags.gpu_vm_support;

	// Block Logic: Enables FBC (Frame Buffer Compression) support if specified by feature mask.
	if (amdgpu_dc_feature_mask & DC_FBC_MASK)
		init_data.flags.fbc_support = true;

	// Block Logic: Enables multi-monitor PP (PowerPlay) MCLK (Memory Clock) switch if specified.
	if (amdgpu_dc_feature_mask & DC_MULTI_MON_PP_MCLK_SWITCH_MASK)
		init_data.flags.multi_mon_pp_mclk_switch = true;

	// Block Logic: Disables fractional PWM (Pulse Width Modulation) if specified.
	if (amdgpu_dc_feature_mask & DC_DISABLE_FRACTIONAL_PWM_MASK)
		init_data.flags.disable_fractional_pwm = true;

	// Block Logic: Disables eDP (embedded DisplayPort) power sequencing if specified.
	if (amdgpu_dc_feature_mask & DC_EDP_NO_POWER_SEQUENCING)
		init_data.flags.edp_no_power_sequencing = true;

	// Block Logic: Disables LTTPR (Link Training Tunable PHY Repeater) non-transparent mode for DP1.4A and DP2.0 if specified.
	if (amdgpu_dc_feature_mask & DC_DISABLE_LTTPR_DP1_4A)
		init_data.flags.allow_lttpr_non_transparent_mode.bits.DP1_4A = true;
	if (amdgpu_dc_feature_mask & DC_DISABLE_LTTPR_DP2_0)
		init_data.flags.allow_lttpr_non_transparent_mode.bits.DP2_0 = true;

	init_data.flags.seamless_boot_edp_requested = false;

	// Block Logic: Enables seamless boot optimizations if supported by the device.
	if (amdgpu_device_seamless_boot_supported(adev)) {
		init_data.flags.seamless_boot_edp_requested = true;
		init_data.flags.allow_seamless_boot_optimization = true;
		drm_dbg(adev->dm.ddev, "Seamless boot requested\n");
	}

	init_data.flags.enable_mipi_converter_optimization = true;

	// Functional Utility: Populates DCN (Display Core Next), NBIO (North Bridge I/O), and CLK (Clock) register offsets.
	init_data.dcn_reg_offsets = adev->reg_offset[DCE_HWIP][0];
	init_data.nbio_reg_offsets = adev->reg_offset[NBIO_HWIP][0];
	init_data.clk_reg_offsets = adev->reg_offset[CLK_HWIP][0];

	// Block Logic: Configures IPS (Idle Power Saving) disable type based on debug mask.
	if (amdgpu_dc_debug_mask & DC_DISABLE_IPS)
		init_data.flags.disable_ips = DMUB_IPS_DISABLE_ALL;
	else if (amdgpu_dc_debug_mask & DC_DISABLE_IPS_DYNAMIC)
		init_data.flags.disable_ips = DMUB_IPS_DISABLE_DYNAMIC;
	else if (amdgpu_dc_debug_mask & DC_DISABLE_IPS2_DYNAMIC)
		init_data.flags.disable_ips = DMUB_IPS_RCG_IN_ACTIVE_IPS2_IN_OFF;
	else if (amdgpu_dc_debug_mask & DC_FORCE_IPS_ENABLE)
		init_data.flags.disable_ips = DMUB_IPS_ENABLE;
	else
		init_data.flags.disable_ips = dm_get_default_ips_mode(adev);

	init_data.flags.disable_ips_in_vpb = 0;

	/* Enable DWB for tested platforms only */
	// Block Logic: Enables DWB (Display Write Back) and sets number of virtual links for DCN3.0+.
	if (amdgpu_ip_version(adev, DCE_HWIP, 0) >= IP_VERSION(3, 0, 0))
		init_data.num_virtual_links = 1;

	// Functional Utility: Retrieves DMI (Desktop Management Interface) information.
	retrieve_dmi_info(&adev->dm);
	// Block Logic: Sets flag for eDP0 on DP1 quirk if detected.
	if (adev->dm.edp0_on_dp1_quirk)
		init_data.flags.support_edp0_on_dp1 = true;

	// Block Logic: Assigns DMUB bounding box information if available.
	if (adev->dm.bb_from_dmub)
		init_data.bb_from_dmub = adev->dm.bb_from_dmub;
	else
		init_data.bb_from_dmub = NULL;

	/* Display Core create. */
	// Functional Utility: Creates the Display Core (DC) instance.
	adev->dm.dc = dc_create(&init_data);

	// Block Logic: Checks if DC initialization was successful and logs version information.
	if (adev->dm.dc) {
		drm_info(adev_to_drm(adev), "Display Core v%s initialized on %s\n", DC_VER,
			 dce_version_to_string(adev->dm.dc->ctx->dce_version));
	} else {
		drm_info(adev_to_drm(adev), "Display Core failed to initialize with v%s!\n", DC_VER);
		goto error;
	}

	// Block Logic: Disables pipe split if specified by debug mask.
	if (amdgpu_dc_debug_mask & DC_DISABLE_PIPE_SPLIT) {
		adev->dm.dc->debug.force_single_disp_pipe_split = false;
		adev->dm.dc->debug.pipe_split_policy = MPC_SPLIT_AVOID;
	}

	// Block Logic: Disables stutter mode based on ASIC type, stutter quirks, or debug mask.
	if (adev->asic_type != CHIP_CARRIZO && adev->asic_type != CHIP_STONEY)
		adev->dm.dc->debug.disable_stutter = amdgpu_pp_feature_mask & PP_STUTTER_MODE ? false : true;
	if (dm_should_disable_stutter(adev->pdev))
		adev->dm.dc->debug.disable_stutter = true;

	if (amdgpu_dc_debug_mask & DC_DISABLE_STUTTER)
		adev->dm.dc->debug.disable_stutter = true;

	// Block Logic: Disables DSC (Display Stream Compression) if specified by debug mask.
	if (amdgpu_dc_debug_mask & DC_DISABLE_DSC)
		adev->dm.dc->debug.disable_dsc = true;

	// Block Logic: Disables clock gating if specified by debug mask.
	if (amdgpu_dc_debug_mask & DC_DISABLE_CLOCK_GATING)
		adev->dm.dc->debug.disable_clock_gate = true;
	// Functional Utility: Jumps to this label in case of error during initialization.
error:
	return r;
}

static void amdgpu_dm_fini(struct amdgpu_device *adev)
{
	if (!adev->dm.dc)
		return;

	// Block Logic: Unregisters the audio component if it was registered.
	if (adev->dm.audio_registered) {
		component_del(adev->dev, &amdgpu_dm_audio_component_bind_ops);
		adev->dm.audio_registered = false;
	}

	amdgpu_dm_destroy_drm_device(&adev->dm);
	dc_destroy(adev->dm.dc);
	adev->dm.dc = NULL;

	amdgpu_dm_irq_fini(adev);
	mutex_destroy(&adev->dm.dpia_aux_lock);
	mutex_destroy(&adev->dm.dc_lock);
	mutex_destroy(&adev->dm.audio_lock);
}

static void amdgpu_dm_detect(struct amdgpu_device *adev,
			     struct drm_connector *connector)
{
	struct amdgpu_dm_connector *aconnector = to_amdgpu_dm_connector(connector);
	struct dc_link *dc_link = aconnector->dc_link;

	enum dc_connection_type new_connection_type = dc_connection_none;

	/* If link is disabled, do not perform detect */
	if (!dc_link_is_link_enabled(dc_link))
		return;

	/* This is a temporary solution for the detection problem by some MST branch device.
	 * We need to always allow detection on first MST branch device.
	 */
	if (aconnector->mst_mgr.mst_state == MST_DETECTED_BRANCH_DEVICE) {
		dc_link_detect_connection_type(dc_link, &new_connection_type);
		aconnector->detected_type = new_connection_type;
		return;
	}

	/* Force detect if link is in non-transparent mode */
	if (!aconnector->force_hpd_poll && !dc_link->lttpr_mode == LTTPR_MODE_NON_TRANSPARENT)
		/* Optimization for detecting disconnects without polling */
		if (!aconnector->fake_enable_hpd)
			if (!dc_link->hpd_status && dc_link->hpd_init_data.hpd_always_on_connected) {
				aconnector->detected_type = dc_connection_single;
				return;
			}


	if (amdgpu_device_should_use_asvm(adev)) {
		mutex_lock(&adev->dm.dpia_aux_lock);
		dc_link_detect_connection_type(dc_link, &new_connection_type);
		mutex_unlock(&adev->dm.dpia_aux_lock);
	} else {
		dc_link_detect_connection_type(dc_link, &new_connection_type);
	}
	aconnector->detected_type = new_connection_type;

	if (new_connection_type != dc_connection_none &&
	    (aconnector->dc_link->connector_signal == SIGNAL_TYPE_DP ||
	     aconnector->dc_link->connector_signal == SIGNAL_TYPE_EDP)) {
		aconnector->is_mst_supported = dc_link_is_mst_supported(dc_link);

		/* When dc detects an mst branch device without a connected sink,
		 * it will set the connection type as single. We need to override
		 * this and treat it as mst detected branch device.
		 */
		if (aconnector->is_mst_supported && !aconnector->dc_sink) {
			aconnector->mst_mgr.mst_state = MST_DETECTED_BRANCH_DEVICE;
			aconnector->detected_type = dc_connection_mst_branch;
		} else if (aconnector->is_mst_supported) {
			if (aconnector->mst_mgr.mst_state == MST_UNKNOWN)
				aconnector->mst_mgr.mst_state = MST_NO_BRANCH_DEVICE;
		} else {
			aconnector->mst_mgr.mst_state = MST_NOT_SUPPORTED;
		}
	}
}

static const struct drm_encoder_funcs amdgpu_dm_encoder_funcs = {
	.destroy = drm_encoder_cleanup,
};

static void amdgpu_dm_encoder_destroy(struct drm_encoder *encoder)
{
	struct amdgpu_encoder *amdgpu_encoder = to_amdgpu_encoder(encoder);

	/* DP MST encoders are dynamically created and destroyed */
	if (amdgpu_encoder->is_mst_encoder) {
		drm_encoder_cleanup(encoder);
		kfree(amdgpu_encoder);
	}
}

static struct drm_encoder *
amdgpu_dm_encoder_create(struct drm_device *dev,
			 struct amdgpu_encoder *aencoder)
{
	struct drm_encoder *encoder = &aencoder->base;

	encoder->possible_crtcs = amdgpu_dm_get_valid_crtc_masks(dev);
	return encoder;
}

static struct amdgpu_encoder *
amdgpu_dm_get_amdgpu_encoder(struct drm_encoder *encoder)
{
	return to_amdgpu_encoder(encoder);
}

static int amdgpu_dm_connector_late_register(struct drm_connector *connector)
{
	struct amdgpu_dm_connector *amdgpu_dm_connector = to_amdgpu_dm_connector(connector);

	if (amdgpu_dm_connector->mst_mgr.mst_state == MST_IS_TOP_CONNECTOR)
		drm_dp_mst_connector_late_register(connector);

	return 0;
}

static void amdgpu_dm_connector_destroy(struct drm_connector *connector)
{
	struct amdgpu_dm_connector *amdgpu_dm_connector = to_amdgpu_dm_connector(connector);

	if (amdgpu_dm_connector->mst_mgr.mst_state == MST_IS_TOP_CONNECTOR)
		drm_dp_mst_connector_destroy(connector);

	amdgpu_dm_connector_fini(connector);
	drm_connector_cleanup(connector);
	kfree(amdgpu_dm_connector);
}

static const struct drm_connector_funcs amdgpu_dm_connector_funcs = {
	.fill_modes = amdgpu_dm_connector_get_modes,
	.destroy = amdgpu_dm_connector_destroy,
	.late_register = amdgpu_dm_connector_late_register,
	.dp_mst_get_port_id = drm_dp_mst_get_port_id,
	.dp_mst_get_vcpi = drm_dp_mst_get_vcpi,
	.atomic_destroy_state = drm_atomic_helper_connector_destroy_state,
	.atomic_duplicate_state = drm_atomic_helper_connector_duplicate_state,
};

static int amdgpu_dm_mode_valid(struct drm_connector *connector,
				const struct drm_display_mode *mode)
{
	struct amdgpu_dm_connector *aconnector = to_amdgpu_dm_connector(connector);
	struct dc_link *dc_link = aconnector->dc_link;
	enum dc_status result;

	if (mode->clock == 0)
		return DRM_MODE_ERROR;

	result = dc_link_validate_mode(dc_link, mode);

	if (result == DC_OK)
		return MODE_OK;
	else if (result == DC_FAIL_DSC_VALIDATION)
		return MODE_DSC_UNSUPPORTED;
	else if (result == DC_FAIL_HRZ_VALIDATION)
		return MODE_HRZ_UNSUPPORTED;
	else
		return MODE_BAD;
}

static struct drm_encoder *amdgpu_dm_best_encoder(struct drm_connector *connector)
{
	struct amdgpu_dm_connector *amdgpu_dm_connector = to_amdgpu_dm_connector(connector);

	return &amdgpu_dm_connector->encoder->base;
}

static const struct drm_connector_helper_funcs amdgpu_dm_connector_helper_funcs = {
	.get_modes = amdgpu_dm_connector_get_modes,
	.mode_valid = amdgpu_dm_mode_valid,
	.best_encoder = amdgpu_dm_best_encoder,
};

static const struct drm_crtc_funcs amdgpu_dm_crtc_funcs = {
	.cursor_set2 = amdgpu_dm_crtc_cursor_set2,
	.cursor_move = amdgpu_dm_crtc_cursor_move,
	.set_config = drm_atomic_helper_set_config,
	.destroy = amdgpu_dm_crtc_destroy,
	.page_flip = drm_atomic_helper_page_flip,
	.atomic_duplicate_state = amdgpu_dm_crtc_duplicate_state,
	.atomic_destroy_state = amdgpu_dm_crtc_destroy_state,
	.set_crc_source = amdgpu_dm_crtc_set_crc_source,
	.verify_crc_source = amdgpu_dm_crtc_verify_crc_source,
	.gamma_set = amdgpu_dm_crtc_gamma_set,
	.get_vblank_counter = dm_vblank_get_counter,
	.get_vblank_timestamp = amdgpu_dm_crtc_get_vblank_timestamp,
	.get_vscanoutpos = dm_crtc_get_scanoutpos,
};

static const struct drm_plane_funcs amdgpu_dm_plane_funcs = {
	.update_plane = drm_atomic_helper_update_plane,
	.disable_plane = drm_atomic_helper_disable_plane,
	.destroy = amdgpu_dm_plane_destroy,
	.atomic_duplicate_state = amdgpu_dm_plane_duplicate_state,
	.atomic_destroy_state = amdgpu_dm_plane_destroy_state,
};

/**
 * amdgpu_dm_irq_init() - Register IRQ handlers for DM
 * @adev: amdgpu_device pointer
 *
 * Register IRQ handlers for HPD, PFLIP and VBLANK
 *
 * Return: 0 on success, error code otherwise
 */
int amdgpu_dm_irq_init(struct amdgpu_device *adev)
{
	int r;
	struct dc *dc = adev->dm.dc;

	if (dc_is_dmub_outbox_supported(dc)) {
		r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_DMCUB_OUTBOX,
						  dm_dmub_outbox1_low_irq, NULL);
		if (r) {
			DRM_ERROR("amdgpu_dm_irq_init: failed to register dmub outbox1 irq handler\n");
			return r;
		}

		if (amdgpu_dm_set_irq_params(
			adev, &adev->dm.dmub_outbox_irq_params,
			DC_IRQ_SOURCE_DMCUB_OUTBOX,
			DM_DMUB_OUTBOX_IRQ_HIGH_IRQ_REG_OFFSET,
			dm_dmub_outbox1_low_irq) != 0) {
			DRM_ERROR("amdgpu_dm_irq_init: failed to set irq params for dmub outbox1 irq handler\n");
			return -EINVAL;
		}

		// Register default DMUB callbacks:
		register_dmub_notify_callback(adev, DMUB_NOTIFICATION_AUX_REPLY,
			dmub_aux_setconfig_callback, false);
		register_dmub_notify_callback(adev, DMUB_NOTIFICATION_SET_CONFIG_REPLY,
			dmub_aux_setconfig_callback, false);
		register_dmub_notify_callback(adev, DMUB_NOTIFICATION_HPD,
			dmub_hpd_callback, true);
		register_dmub_notify_callback(adev, DMUB_NOTIFICATION_HPD_IRQ,
			dmub_hpd_callback, true);
		register_dmub_notify_callback(adev, DMUB_NOTIFICATION_FUSED_IO,
			dmub_aux_fused_io_callback, false);
		register_dmub_notify_callback(adev, DMUB_NOTIFICATION_HPD_SENSE_NOTIFY,
			dmub_hpd_sense_callback, false);
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_HPD1,
					  handle_hpd_irq, &adev->dm.hpd_irq_params[0]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register hpd irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_HPD2,
					  handle_hpd_irq, &adev->dm.hpd_irq_params[1]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register hpd irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_HPD3,
					  handle_hpd_irq, &adev->dm.hpd_irq_params[2]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register hpd irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_HPD4,
					  handle_hpd_irq, &adev->dm.hpd_irq_params[3]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register hpd irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_HPD5,
					  handle_hpd_irq, &adev->dm.hpd_irq_params[4]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register hpd irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_HPD6,
					  handle_hpd_irq, &adev->dm.hpd_irq_params[5]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register hpd irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_HPD_RX1,
					  handle_hpd_rx_irq, &adev->dm.hpd_irq_params[0]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register hpd rx irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_HPD_RX2,
					  handle_hpd_rx_irq, &adev->dm.hpd_irq_params[1]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register hpd rx irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_HPD_RX3,
					  handle_hpd_rx_irq, &adev->dm.hpd_irq_params[2]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register hpd rx irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_HPD_RX4,
					  handle_hpd_rx_irq, &adev->dm.hpd_irq_params[3]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register hpd rx irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_HPD_RX5,
					  handle_hpd_rx_irq, &adev->dm.hpd_irq_params[4]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register hpd rx irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_HPD_RX6,
					  handle_hpd_rx_irq, &adev->dm.hpd_irq_params[5]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register hpd rx irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_PFLIP1,
					  dm_pflip_high_irq, &adev->dm.pflip_irq_params[0]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register pflip irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_PFLIP2,
					  dm_pflip_high_irq, &adev->dm.pflip_irq_params[1]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register pflip irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_PFLIP3,
					  dm_pflip_high_irq, &adev->dm.pflip_irq_params[2]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register pflip irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_PFLIP4,
					  dm_pflip_high_irq, &adev->dm.pflip_irq_params[3]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register pflip irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_PFLIP5,
					  dm_pflip_high_irq, &adev->dm.pflip_irq_params[4]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register pflip irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_PFLIP6,
					  dm_pflip_high_irq, &adev->dm.pflip_irq_params[5]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register pflip irq handler\n");
		return r;
	}
#if defined(CONFIG_DRM_AMD_SECURE_DISPLAY)
	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_VLINE0,
					  dm_dcn_vertical_interrupt0_high_irq,
					  &adev->dm.vline0_irq_params[0]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register vline0 irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_VLINE0_1,
					  dm_dcn_vertical_interrupt0_high_irq,
					  &adev->dm.vline0_irq_params[1]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register vline0_1 irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_VLINE0_2,
					  dm_dcn_vertical_interrupt0_high_irq,
					  &adev->dm.vline0_irq_params[2]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register vline0_2 irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_VLINE0_3,
					  dm_dcn_vertical_interrupt0_high_irq,
					  &adev->dm.vline0_irq_params[3]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register vline0_3 irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_VLINE0_4,
					  dm_dcn_vertical_interrupt0_high_irq,
					  &adev->dm.vline0_irq_params[4]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register vline0_4 irq handler\n");
		return r;
	}

	r = amdgpu_irq_add_domain_handler(adev, DC_IRQ_SOURCE_VLINE0_5,
					  dm_dcn_vertical_interrupt0_high_irq,
					  &adev->dm.vline0_irq_params[5]);
	if (r) {
		DRM_ERROR("amdgpu_dm_irq_init: failed to register vline0_5 irq handler\n");
		return r;
	}
#endif
	for (i = 0; i < adev->mode_info.num_crtc; i++) {
		if (amdgpu_dm_set_irq_params(
			adev, &adev->dm.vblank_irq_params[i],
			DC_IRQ_SOURCE_VBLANK1 + i,
			DM_CRTC_IRQ_HIGH_IRQ_REG_OFFSET,
			dm_crtc_high_irq) != 0) {
			DRM_ERROR("amdgpu_dm_irq_init: failed to set irq params for vblank irq handler\n");
			return -EINVAL;
		}

		if (amdgpu_dm_set_irq_params(
			adev, &adev->dm.vupdate_irq_params[i],
			DC_IRQ_SOURCE_VUPDATE1 + i,
			DM_VUPDATE_IRQ_HIGH_IRQ_REG_OFFSET,
			dm_vupdate_high_irq) != 0) {
			DRM_ERROR("amdgpu_dm_irq_init: failed to set irq params for vupdate irq handler\n");
			return -EINVAL;
		}
	}

	adev->dm.delayed_hpd_wq = create_singlethread_workqueue("amdgpu_dm_delayed_hpd_wq");
	if (adev->dm.delayed_hpd_wq == NULL) {
		DRM_ERROR("create amdgpu_dm_delayed_hpd_wq fail!");
		return -EINVAL;
	}

	adev->dm.hpd_rx_offload_wq = hpd_rx_irq_create_workqueue(adev);

	if (adev->dm.hpd_rx_offload_wq == NULL) {
		DRM_ERROR("create hpd_rx_offload_wq fail!");
		return -EINVAL;
	}
	return 0;
}

/**
 * amdgpu_dm_irq_fini() - Unregister IRQ handlers for DM
 * @adev: amdgpu_device pointer
 *
 * Unregister IRQ handlers for HPD, PFLIP and VBLANK
 */
void amdgpu_dm_irq_fini(struct amdgpu_device *adev)
{
	int i;
	struct dc *dc = adev->dm.dc;

	if (dc_is_dmub_outbox_supported(dc)) {
		amdgpu_irq_remove_domain_handler(adev, DC_IRQ_SOURCE_DMCUB_OUTBOX,
						 dm_dmub_outbox1_low_irq, NULL);

		// Clear default DMUB callbacks:
		register_dmub_notify_callback(adev, DMUB_NOTIFICATION_AUX_REPLY,
			NULL, false);
		register_dmub_notify_callback(adev, DMUB_NOTIFICATION_SET_CONFIG_REPLY,
			NULL, false);
		register_dmub_notify_callback(adev, DMUB_NOTIFICATION_HPD,
			NULL, false);
		register_dmub_notify_callback(adev, DMUB_NOTIFICATION_HPD_IRQ,
			NULL, false);
		register_dmub_notify_callback(adev, DMUB_NOTIFICATION_FUSED_IO,
			NULL, false);
		register_dmub_notify_callback(adev, DMUB_NOTIFICATION_HPD_SENSE_NOTIFY,
			NULL, false);
	}

	for (i = 0; i < MAX_HPD_PINS; i++) {
		amdgpu_irq_remove_domain_handler(adev, DC_IRQ_SOURCE_HPD1 + i,
						 handle_hpd_irq,
						 &adev->dm.hpd_irq_params[i]);
		amdgpu_irq_remove_domain_handler(adev, DC_IRQ_SOURCE_HPD_RX1 + i,
						 handle_hpd_rx_irq,
						 &adev->dm.hpd_irq_params[i]);
	}

	for (i = 0; i < adev->mode_info.num_crtc; i++) {
		amdgpu_irq_remove_domain_handler(adev, DC_IRQ_SOURCE_PFLIP1 + i,
						 dm_pflip_high_irq,
						 &adev->dm.pflip_irq_params[i]);
		amdgpu_irq_remove_domain_handler(adev, DC_IRQ_SOURCE_VBLANK1 + i,
						 dm_crtc_high_irq,
						 &adev->dm.vblank_irq_params[i]);
		amdgpu_irq_remove_domain_handler(adev, DC_IRQ_SOURCE_VUPDATE1 + i,
						 dm_vupdate_high_irq,
						 &adev->dm.vupdate_irq_params[i]);
#if defined(CONFIG_DRM_AMD_SECURE_DISPLAY)
		amdgpu_irq_remove_domain_handler(adev, DC_IRQ_SOURCE_VLINE0 + i,
						 dm_dcn_vertical_interrupt0_high_irq,
						 &adev->dm.vline0_irq_params[i]);
#endif
	}
	if (adev->dm.delayed_hpd_wq) {
		destroy_workqueue(adev->dm.delayed_hpd_wq);
		adev->dm.delayed_hpd_wq = NULL;
	}
	if (adev->dm.hpd_rx_offload_wq) {
		int max_caps = dc->caps.max_links;

		for (i = 0; i < max_caps; i++) {
			if (adev->dm.hpd_rx_offload_wq[i].wq)
				destroy_workqueue(adev->dm.hpd_rx_offload_wq[i].wq);
		}
		kfree(adev->dm.hpd_rx_offload_wq);
		adev->dm.hpd_rx_offload_wq = NULL;
	}
}

static int amdgpu_dm_early_init(void *handle)
{
	struct amdgpu_device *adev = (struct amdgpu_device *)handle;
	int r;

	r = amdgpu_dm_irq_params_init(adev);
	if (r) {
		DRM_ERROR("amdgpu_dm_early_init: failed to init irq parameters\n");
		return r;
	}
	adev->dm.cgs_device = cgs_create_device(adev->dev);
	if (!adev->dm.cgs_device) {
		DRM_ERROR("amdgpu_dm_early_init: failed to create cgs device\n");
		return -EINVAL;
	}
	return 0;
}

static int amdgpu_dm_late_init(void *handle)
{
	struct amdgpu_device *adev = (struct amdgpu_device *)handle;
	struct dc_bios *dcbios = adev->dm.dc->res_pool->bios;

	if (dcbios->funcs->init_bonaire_power_gating)
		dcbios->funcs->init_bonaire_power_gating(dcbios, true);

	return 0;
}

static int amdgpu_dm_sw_init(void *handle)
{
	int r;
	struct amdgpu_device *adev = (struct amdgpu_device *)handle;
	struct dc_firmware_info dmub_fw_info;

	r = amdgpu_dm_early_init(adev);
	if (r)
		goto error;

	r = amdgpu_ucode_load_dmcub(adev, &dmub_fw_info);
	if (!r) {
		adev->dm.dmcub_fw_version = dmub_fw_info.dmcub_fw_version;
		adev->dm.dmub_fw = dmub_fw_info.dmub_fw;
	} else if (r != -ENOENT) {
		DRM_ERROR("amdgpu_ucode_load_dmcub: Failed to load DMCUB firmware: %d\n", r);
		goto error;
	}

	r = amdgpu_dm_init(adev);
	if (r) {
		amdgpu_dm_fini(adev);
		DRM_ERROR("amdgpu_dm_sw_init: amdgpu_dm_init failed. %d\n", r);
	}

error:
	return r;
}

static int amdgpu_dm_sw_fini(void *handle)
{
	struct amdgpu_device *adev = (struct amdgpu_device *)handle;

	amdgpu_dm_fini(adev);
	cgs_destroy_device(&adev->dm.cgs_device);
	amdgpu_dm_irq_params_fini(adev);
	return 0;
}

static int amdgpu_dm_hw_init(void *handle)
{
	struct amdgpu_device *adev = (struct amdgpu_device *)handle;
	int r;

	r = dm_dmub_hw_init(adev);
	if (r) {
		drm_err(adev_to_drm(adev), "DMUB interface failed to initialize: status=%d\n", r);
		return r;
	}

	r = amdgpu_dm_initialize_drm_device(adev);
	if (r) {
		drm_err(adev_to_drm(adev), "amdgpu_dm_hw_init: amdgpu_dm_initialize_drm_device failed. %d\n", r);
		amdgpu_dm_fini(adev);
	}

	return r;
}

static int amdgpu_dm_hw_fini(void *handle)
{
	/* amdgpu_dm_hw_fini handled in amdgpu_dm_sw_fini */
	return 0;
}

static int amdgpu_dm_suspend(void *handle)
{
	struct amdgpu_device *adev = (struct amdgpu_device *)handle;
	int i;

	if (adev->dm.dc) {
		/* Disable HPD/HPD_RX interrupts before dm suspend. */
		for (i = 0; i < MAX_HPD_PINS; i++) {
			amdgpu_irq_put(adev, DC_IRQ_SOURCE_HPD1 + i, handle_hpd_irq, &adev->dm.hpd_irq_params[i]);
			amdgpu_irq_put(adev, DC_IRQ_SOURCE_HPD_RX1 + i, handle_hpd_rx_irq, &adev->dm.hpd_irq_params[i]);
		}
		dc_suspend_dmub_srv(adev->dm.dc);
		dc_suspend(adev->dm.dc);
	}
	adev->in_suspend = true;

	return 0;
}

static int amdgpu_dm_resume(void *handle)
{
	struct amdgpu_device *adev = (struct amdgpu_device *)handle;
	int i;

	adev->in_suspend = false;

	if (adev->dm.dc) {
		dc_resume(adev->dm.dc);
		dm_dmub_hw_resume(adev);
		/* Enable HPD/HPD_RX interrupts after dm resume. */
		for (i = 0; i < MAX_HPD_PINS; i++) {
			amdgpu_irq_get(adev, DC_IRQ_SOURCE_HPD1 + i, handle_hpd_irq, &adev->dm.hpd_irq_params[i]);
			amdgpu_irq_get(adev, DC_IRQ_SOURCE_HPD_RX1 + i, handle_hpd_rx_irq, &adev->dm.hpd_irq_params[i]);
		}
	}

	return 0;
}

static int amdgpu_dm_set_clockgating_state(void *handle,
					  enum amd_clockgating_state state)
{
	struct amdgpu_device *adev = (struct amdgpu_device *)handle;
	struct dc_state *ctx;
	int r = 0;

	if (adev->dm.dc) {
		if (state == AMD_CG_STATE_GATE) {
			/* DC needs to handle clock gating on suspend but only when
			 * suspend was initiated from an S-state (e.g. S3, S4).
			 * If it was initiated from D-state (e.g. D1, D2, D3hot) then
			 * all clocks should already be off.
			 */
			if (adev->in_suspend) {
				dc_allow_idle_optimizations(adev->dm.dc, true);
				dc_set_power_state(adev->dm.dc, DC_ACPI_CM_POWER_STATE_D3);
			} else {
				dc_set_clock_gating_state(adev->dm.dc, true);
			}
		} else {
			if (adev->in_suspend)
				dc_set_power_state(adev->dm.dc, DC_ACPI_CM_POWER_STATE_D0);

			dc_set_clock_gating_state(adev->dm.dc, false);
			dc_allow_idle_optimizations(adev->dm.dc, false);
		}
	}

	ctx = dc_create_state(adev->dm.dc);
	if (ctx) {
		if (state == AMD_CG_STATE_GATE)
			ctx->stream_count = 0;
		else
			dc_set_power_state(adev->dm.dc, DC_ACPI_CM_POWER_STATE_D0);

		dc_commit_state(adev->dm.dc, ctx);
		dc_release_state(ctx);
	}

	return r;
}

static int amdgpu_dm_set_powergating_state(void *handle,
					   enum amd_powergating_state state)
{
	/* amdgpu_dm_set_powergating_state - no op*/
	return 0;
}

static const struct amd_ip_funcs amdgpu_dm_ip_funcs = {
	.name = "dm",
	.early_init = amdgpu_dm_early_init,
	.late_init = amdgpu_dm_late_init,
	.sw_init = amdgpu_dm_sw_init,
	.sw_fini = amdgpu_dm_sw_fini,
	.hw_init = amdgpu_dm_hw_init,
	.hw_fini = amdgpu_dm_hw_fini,
	.suspend = amdgpu_dm_suspend,
	.resume = amdgpu_dm_resume,
	.set_clockgating_state = amdgpu_dm_set_clockgating_state,
	.set_powergating_state = amdgpu_dm_set_powergating_state,
};

const struct amdgpu_ip_block_version dm_ip_block = {
	.type = AMD_IP_BLOCK_TYPE_DCE,
	.funcs = &amdgpu_dm_ip_funcs,
	.version_major = 1,
	.version_minor = 0,
	.rev = 0,
};

const struct drm_mode_config_funcs amdgpu_dm_mode_funcs = {
	.fb_create = amdgpu_dm_framebuffer_init,
	.output_poll_changed = drm_fb_helper_output_poll_changed,
	.atomic_check = amdgpu_dm_atomic_check,
	.atomic_commit = amdgpu_dm_atomic_commit_tail,
};

static void amdgpu_dm_mode_config_init(struct amdgpu_device *adev)
{
	int i;

	adev->mode_info.mode_config_initialized = true;

	/*
	 * TODO: the default limit of 'max_cursor_width' and
	 * 'max_cursor_height' is 64, but Windows supports 256
	 * with A+ command table, so increase max to 256 for now.
	 * In the future, this value will come from DAL.
	 */
	adev->dm.ddev->mode_config.max_cursor_width = 256;
	adev->dm.ddev->mode_config.max_cursor_height = 256;

	adev->dm.ddev->mode_config.funcs = &amdgpu_dm_mode_funcs;

	/*
	 * Enforce maximum number of CRTC's, this is a hard-coded limit
	 * for now but in the future it will come from the DC library
	 */
	adev->dm.ddev->mode_config.max_crtc = 6;

	drm_mode_config_init(adev->dm.ddev);

	adev->dm.dc->caps.max_display_planes = adev->dm.ddev->mode_config.max_crtc;
	adev->dm.dc->caps.max_planes = adev->dm.ddev->mode_config.max_crtc * 2;
	if (adev->dm.dc->caps.max_planes > amdgpu_dm_max_hdr_display_num)
		adev->dm.dc->caps.max_planes = amdgpu_dm_max_hdr_display_num;

	for (i = 0; i < adev->dm.ddev->mode_config.max_crtc; ++i) {
		struct amdgpu_crtc *acrtc = amdgpu_crtc_at(adev, i);

		acrtc->max_otg_inst = adev->dm.ddev->mode_config.max_crtc;
	}
	adev->dm.ddev->mode_config.async_page_flip = true;

	adev->dm.ddev->mode_config.preferred_depth = 24;
	adev->dm.ddev->mode_config.prefer_shadow_fb = true;

	/*
	 * To support YCbCr color, we need at least 10 bits per color,
	 * so increase the default to 10
	 */
	if (!amdgpu_dm_need_slice_mode(adev->dm.dc))
		adev->dm.ddev->mode_config.funcs->check_image = drm_atomic_helper_check_planes;

	/*
	 * For now, only 8, 10, and 12-bit per color are supported.
	 * The driver can expose support for 6-bit per color if required,
	 * but since no panel supports 6-bit input, this is not needed.
	 */
	drm_mode_create_scaling_filter_property(adev->dm.ddev);
	drm_mode_create_aspect_ratio_property(adev->dm.ddev);
	drm_mode_create_suggested_offset_property(adev->dm.ddev);
	drm_mode_create_dithering_property(adev->dm.ddev);
	drm_mode_create_color_encoding_property(adev->dm.ddev, 0);
	drm_mode_create_color_range_property(adev->dm.ddev, 0);
	drm_mode_create_freesync_property(adev->dm.ddev);

	/* DPMS property for writeback */
	drm_mode_create_dpms_property(adev->dm.ddev);
	drm_mode_create_writeback_cmd_property(adev->dm.ddev);
	drm_mode_create_writeback_out_size_property(adev->dm.ddev);
	drm_mode_create_writeback_pixel_format_property(adev->dm.ddev);
}

static const struct drm_framebuffer_funcs amdgpu_dm_fb_funcs = {
	.destroy = drm_gem_fb_destroy,
	.create_handle = drm_gem_fb_create_handle,
};

static int amdgpu_dm_framebuffer_init(struct drm_device *dev,
				      struct drm_framebuffer *fb,
				      struct drm_modeset_acquire_ctx *ctx)
{
	struct amdgpu_framebuffer *amdgpu_fb = to_amdgpu_framebuffer(fb);
	struct amdgpu_bo *rbo;
	int ret;

	rbo = gem_to_amdgpu_bo(amdgpu_fb->base.obj[0]);
	ret = amdgpu_bo_reserve(rbo, false);
	if (ret)
		return ret;

	ret = amdgpu_bo_pin(rbo, AMDGPU_GEM_DOMAIN_VRAM, &amdgpu_fb->gpu_addr);
	if (ret) {
		amdgpu_bo_unreserve(rbo);
		return ret;
	}
	amdgpu_bo_unreserve(rbo);

	drm_framebuffer_init(dev, fb, &amdgpu_dm_fb_funcs);

	return 0;
}

static void amdgpu_dm_framebuffer_destroy(struct drm_framebuffer *fb)
{
	struct amdgpu_framebuffer *amdgpu_fb = to_amdgpu_framebuffer(fb);
	struct amdgpu_bo *rbo = gem_to_amdgpu_bo(amdgpu_fb->base.obj[0]);
	int ret;

	ret = amdgpu_bo_reserve(rbo, false);
	if (ret) {
		drm_err(fb->dev, "failed to reserve rbo to unpin\n");
		return;
	}

	amdgpu_bo_unpin(rbo);
	amdgpu_bo_unreserve(rbo);

	drm_framebuffer_cleanup(fb);
	drm_framebuffer_free(fb);
}

const struct drm_crtc_helper_funcs amdgpu_dm_crtc_helper_funcs = {
	.mode_set_base_atomic = amdgpu_dm_crtc_mode_set_base_atomic,
	.atomic_check = amdgpu_dm_crtc_atomic_check,
	.atomic_begin = amdgpu_dm_crtc_atomic_begin,
	.atomic_flush = amdgpu_dm_crtc_atomic_flush,
};

const struct drm_plane_helper_funcs amdgpu_dm_plane_helper_funcs = {
	.atomic_check = amdgpu_dm_plane_atomic_check,
	.atomic_update = amdgpu_dm_plane_atomic_update,
	.atomic_disable = amdgpu_dm_plane_atomic_disable,
};

static int amdgpu_dm_crtc_set_property(struct drm_crtc *crtc,
					struct drm_property *property,
					u64 val)
{
	int ret = 0;

	// Functional Utility: Handles setting the CRTC's "freesync_video_mode" property.
	ret = amdgpu_dm_crtc_set_freesync_video_mode_property(crtc, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the CRTC's "freesync_active" property.
	ret = amdgpu_dm_crtc_set_freesync_active_property(crtc, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the CRTC's "freesync_capable" property.
	ret = amdgpu_dm_crtc_set_freesync_capable_property(crtc, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the CRTC's "freesync_force_static" property.
	ret = amdgpu_dm_crtc_set_freesync_force_static_property(crtc, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the CRTC's "freesync_vertical_range_min" property.
	ret = amdgpu_dm_crtc_set_freesync_vertical_range_min_property(crtc, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the CRTC's "freesync_vertical_range_max" property.
	ret = amdgpu_dm_crtc_set_freesync_vertical_range_max_property(crtc, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the CRTC's "freesync_h_range_in_pixels" property.
	ret = amdgpu_dm_crtc_set_freesync_h_range_in_pixels_property(crtc, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the CRTC's "freesync_h_range_out_pixels" property.
	ret = amdgpu_dm_crtc_set_freesync_h_range_out_pixels_property(crtc, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the CRTC's "scaling_filter" property.
	ret = amdgpu_dm_crtc_set_scaling_filter_property(crtc, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the CRTC's "dithering" property.
	ret = amdgpu_dm_crtc_set_dithering_property(crtc, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the CRTC's "color_encoding" property.
	ret = amdgpu_dm_crtc_set_color_encoding_property(crtc, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the CRTC's "color_range" property.
	ret = amdgpu_dm_crtc_set_color_range_property(crtc, property, val);
	if (ret < 0)
		return ret;

	return 0;
}

static int amdgpu_dm_plane_set_property(struct drm_plane *plane,
					struct drm_property *property,
					u64 val)
{
	int ret = 0;

	// Functional Utility: Handles setting the plane's "scaling_filter" property.
	ret = amdgpu_dm_plane_set_scaling_filter_property(plane, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the plane's "pixel_blend_mode" property.
	ret = amdgpu_dm_plane_set_pixel_blend_mode_property(plane, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the plane's "rotation" property.
	ret = amdgpu_dm_plane_set_rotation_property(plane, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the plane's "zpos" property.
	ret = amdgpu_dm_plane_set_zpos_property(plane, property, val);
	if (ret < 0)
		return ret;

	return 0;
}

static int amdgpu_dm_connector_set_property(struct drm_connector *connector,
					    struct drm_property *property,
					    u64 val)
{
	int ret = 0;

	// Functional Utility: Handles setting the connector's "underscan" property.
	ret = amdgpu_dm_connector_set_underscan_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "underscan_hborder" property.
	ret = amdgpu_dm_connector_set_underscan_hborder_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "underscan_vborder" property.
	ret = amdgpu_dm_connector_set_underscan_vborder_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "link_status" property.
	ret = amdgpu_dm_connector_set_link_status_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "panel_orientation" property.
	ret = amdgpu_dm_connector_set_panel_orientation_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "dpms" property.
	ret = amdgpu_dm_connector_set_dpms_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "scaling_filter" property.
	ret = amdgpu_dm_connector_set_scaling_filter_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "dithering" property.
	ret = amdgpu_dm_connector_set_dithering_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "content_type" property.
	ret = amdgpu_dm_connector_set_content_type_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "hdcp_content_type" property.
	ret = amdgpu_dm_connector_set_hdcp_content_type_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "max_bpc" property.
	ret = amdgpu_dm_connector_set_max_bpc_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "dsc_enable" property.
	ret = amdgpu_dm_connector_set_dsc_enable_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "abm_level" property.
	ret = amdgpu_dm_connector_set_abm_level_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "hdr_metadata" property.
	ret = amdgpu_dm_connector_set_hdr_metadata_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "psr_cap" property.
	ret = amdgpu_dm_connector_set_psr_cap_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "psr_state" property.
	ret = amdgpu_dm_connector_set_psr_state_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "replay_cap" property.
	ret = amdgpu_dm_connector_set_replay_cap_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "replay_state" property.
	ret = amdgpu_dm_connector_set_replay_state_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "replay_mode" property.
	ret = amdgpu_dm_connector_set_replay_mode_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "writeback_src_h" property.
	ret = amdgpu_dm_connector_set_writeback_src_h_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "writeback_src_w" property.
	ret = amdgpu_dm_connector_set_writeback_src_w_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "writeback_src_x" property.
	ret = amdgpu_dm_connector_set_writeback_src_x_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "writeback_src_y" property.
	ret = amdgpu_dm_connector_set_writeback_src_y_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "writeback_dst_x" property.
	ret = amdgpu_dm_connector_set_writeback_dst_x_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "writeback_dst_y" property.
	ret = amdgpu_dm_connector_set_writeback_dst_y_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "writeback_dst_w" property.
	ret = amdgpu_dm_connector_set_writeback_dst_w_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "writeback_dst_h" property.
	ret = amdgpu_dm_connector_set_writeback_dst_h_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "writeback_target_id" property.
	ret = amdgpu_dm_connector_set_writeback_target_id_property(connector, property, val);
	if (ret < 0)
		return ret;

	// Functional Utility: Handles setting the connector's "audio" property.
	ret = amdgpu_dm_connector_set_audio_property(connector, property, val);
	if (ret < 0)
		return ret;

	return 0;
}

static const struct drm_crtc_helper_funcs amdgpu_dm_atomic_crtc_helper_funcs = {
	.mode_set_base = amdgpu_dm_crtc_mode_set_base,
	.atomic_check = amdgpu_dm_crtc_atomic_check,
	.atomic_begin = amdgpu_dm_crtc_atomic_begin,
	.atomic_flush = amdgpu_dm_crtc_atomic_flush,
};

const struct drm_plane_helper_funcs amdgpu_dm_atomic_plane_helper_funcs = {
	.atomic_check = amdgpu_dm_plane_atomic_check,
	.atomic_update = amdgpu_dm_plane_atomic_update,
	.atomic_disable = amdgpu_dm_plane_atomic_disable,
};

const struct amdgpu_display_funcs amdgpu_dm_display_funcs = {
	.set_backlight_level = amdgpu_dm_backlight_set_level,
	.backlight_init = amdgpu_dm_backlight_init,
	.hpd_init = amdgpu_dm_hpd_init,
	.hpd_fini = amdgpu_dm_hpd_fini,
	.audio_init = amdgpu_dm_audio_init,
	.audio_fini = amdgpu_dm_audio_fini,
	.audio_eld_notify = amdgpu_dm_audio_eld_notify,
	.get_dig_monitor_crc_info = amdgpu_dm_get_dig_monitor_crc_info,
	.get_total_active_displays = amdgpu_dm_get_total_active_displays,
	.add_sink_irq_params = amdgpu_dm_add_sink_irq_params,
	.remove_sink_irq_params = amdgpu_dm_remove_sink_irq_params,
	.dm_dp_aux_transfer = amdgpu_dm_dp_aux_transfer,
	.enable_dmub_notifications = amdgpu_dm_enable_dmub_notifications,
	.dm_dmub_outbox1_low_irq = dm_dmub_outbox1_low_irq,
	.dm_dmub_aux_setconfig_callback = dmub_aux_setconfig_callback,
	.dm_dmub_hpd_callback = dmub_hpd_callback,
	.dm_dmub_hpd_sense_callback = dmub_hpd_sense_callback,
	.dm_dmub_aux_fused_io_callback = dmub_aux_fused_io_callback,
	.is_freesync_video_mode = is_freesync_video_mode,
	.reset_freesync_config_for_crtc = reset_freesync_config_for_crtc,
	.update_freesync_video_mode = amdgpu_dm_crtc_update_freesync_video_mode,
	.is_timing_unchanged_for_freesync = is_timing_unchanged_for_freesync,
	.set_freesync_config_for_pipe = amdgpu_dm_crtc_set_freesync_config_for_pipe,
	.dm_dmub_get_vbios_bounding_box = dm_dmub_get_vbios_bounding_box,
	.get_num_active_crtc = amdgpu_dm_get_num_active_crtc,
	.set_ps_connector_property = amdgpu_dm_set_ps_connector_property,
	.get_pa_config = mmhub_read_system_context,
	.alloc_gpu_mem = dm_allocate_gpu_mem,
	.free_gpu_mem = dm_free_gpu_mem,
	.trigger_hotplug_cpu = amdgpu_dm_trigger_hotplug_cpu,
	.wait_for_reset = amdgpu_dm_wait_for_reset,
	.power_gate_all_dcn_pipes = amdgpu_dm_power_gate_all_dcn_pipes,
	.get_current_freesync_caps = amdgpu_dm_crtc_get_current_freesync_caps,
	.get_current_freesync_info = amdgpu_dm_crtc_get_current_freesync_info,
	.log_hw_state = amdgpu_dm_log_hw_state,
};

static int amdgpu_dm_encoder_init(struct drm_device *dev,
				  struct amdgpu_encoder *aencoder,
				  uint32_t link_index)
{
	struct amdgpu_device *adev = dev->dev_private;
	struct dc_link *link = adev->dm.dc->links[link_index];
	struct drm_encoder *encoder = &aencoder->base;
	int ret;

	// Functional Utility: Clears the encoder structure.
	memset(aencoder, 0, sizeof(struct amdgpu_encoder));

	// Functional Utility: Initializes the DRM encoder.
	drm_encoder_init(dev, encoder, &amdgpu_dm_encoder_funcs,
			 drm_dp_mst_encoder_is_mst(encoder) ?
			 DRM_MODE_ENCODER_DPMST : DRM_MODE_ENCODER_NONE, NULL);

	encoder->possible_crtcs = amdgpu_dm_get_valid_crtc_masks(dev);

	aencoder->link_index = link_index;
	aencoder->dc_link = link;
	aencoder->is_mst_encoder = drm_dp_mst_encoder_is_mst(encoder);
	aencoder->base.helper_private = &amdgpu_dm_encoder_helper_funcs;

	// Block Logic: If the link supports MST, it initializes DP MST.
	if (dc_link_is_dp_mst_supported(link)) {
		ret = drm_dp_mst_encoder_init(encoder,
					      &adev->dm.mst_mgr.aux_mgr[link_index],
					      adev->dm.ddev->mode_config.max_crtc - 1,
					      1);
		if (ret < 0) {
			DRM_ERROR("Failed to initialize mst encoder\n");
			drm_encoder_cleanup(encoder);
			return ret;
		}
		aencoder->is_mst_encoder = true;
	}

	return 0;
}

static void amdgpu_dm_connector_fini(struct drm_connector *connector)
{
	struct amdgpu_dm_connector *amdgpu_dm_connector = to_amdgpu_dm_connector(connector);
	struct amdgpu_device *adev = connector->dev->dev_private;
	struct dc_link *dc_link = amdgpu_dm_connector->dc_link;

	// Block Logic: Decrements the usage count for the IRQ source.
	amdgpu_irq_put(adev, DC_IRQ_SOURCE_HPD1 + dc_link->link_index,
		       handle_hpd_irq, &adev->dm.hpd_irq_params[dc_link->link_index]);
	amdgpu_irq_put(adev, DC_IRQ_SOURCE_HPD_RX1 + dc_link->link_index,
		       handle_hpd_rx_irq, &adev->dm.hpd_irq_params[dc_link->link_index]);

	// Block Logic: If it's the top connector for MST, destroys MST manager.
	if (amdgpu_dm_connector->mst_mgr.mst_state == MST_IS_TOP_CONNECTOR)
		dm_destroy_mst_mgr(&amdgpu_dm_connector->mst_mgr);

	// Functional Utility: Frees the DC sink associated with the connector.
	dc_sink_release(amdgpu_dm_connector->dc_sink);

	// Functional Utility: Frees the FBC (Frame Buffer Compression) resources if initialized.
	if (amdgpu_dm_connector->dm_fbc_bo_ptr) {
		amdgpu_bo_free_kernel(&amdgpu_dm_connector->dm_fbc_bo_ptr,
				      &amdgpu_dm_connector->dm_fbc_gpu_addr,
				      &amdgpu_dm_connector->dm_fbc_cpu_addr);
		amdgpu_dm_connector->dm_fbc_bo_ptr = NULL;
	}
}

/**
 * amdgpu_dm_hpd_init() - Init HPD (Hot Plug Detect)
 * @adev: amdgpu_device pointer
 *
 * Initialize HPD (Hot Plug Detect) pins by enabling the interrupts
 *
 * Return: void
 */
static void amdgpu_dm_hpd_init(struct amdgpu_device *adev)
{
	int i;

	// Block Logic: Enables HPD and HPD RX interrupts for all HPD pins.
	for (i = 0; i < MAX_HPD_PINS; i++) {
		if (adev->dm.hpd_irq_params[i].irq_src != DC_IRQ_SOURCE_INVALID) {
			amdgpu_irq_get(adev, DC_IRQ_SOURCE_HPD1 + i, handle_hpd_irq, &adev->dm.hpd_irq_params[i]);
			amdgpu_irq_get(adev, DC_IRQ_SOURCE_HPD_RX1 + i, handle_hpd_rx_irq, &adev->dm.hpd_irq_params[i]);
		}
	}
}

/**
 * amdgpu_dm_hpd_fini() - Fini HPD (Hot Plug Detect)
 * @adev: amdgpu_device pointer
 *
 * Fini HPD (Hot Plug Detect) pins by disabling the interrupts
 *
 * Return: void
 */
static void amdgpu_dm_hpd_fini(struct amdgpu_device *adev)
{
	int i;

	// Block Logic: Disables HPD and HPD RX interrupts for all HPD pins.
	for (i = 0; i < MAX_HPD_PINS; i++) {
		if (adev->dm.hpd_irq_params[i].irq_src != DC_IRQ_SOURCE_INVALID) {
			amdgpu_irq_put(adev, DC_IRQ_SOURCE_HPD1 + i, handle_hpd_irq, &adev->dm.hpd_irq_params[i]);
			amdgpu_irq_put(adev, DC_IRQ_SOURCE_HPD_RX1 + i, handle_hpd_rx_irq, &adev->dm.hpd_irq_params[i]);
		}
	}
}

static int amdgpu_dm_initialize_drm_device(struct amdgpu_device *adev)
{
	int i;
	int r;
	struct dc_version info;

	// Functional Utility: Initializes the mode configuration for the DRM device.
	amdgpu_dm_mode_config_init(adev);

	// Block Logic: Iterates through each display link to initialize DRM connectors and encoders.
	for (i = 0; i < adev->dm.dc->link_count; i++) {
		struct amdgpu_dm_connector *amdgpu_dm_connector =
			kzalloc(sizeof(struct amdgpu_dm_connector), GFP_KERNEL);

		struct amdgpu_encoder *amdgpu_encoder =
			kzalloc(sizeof(struct amdgpu_encoder), GFP_KERNEL);

		if (!amdgpu_dm_connector || !amdgpu_encoder) {
			DRM_ERROR("amdgpu_dm_initialize_drm_device: failed to allocate connector/encoder\n");
			r = -ENOMEM;
			goto error;
		}

		// Functional Utility: Initializes the DRM encoder.
		r = amdgpu_dm_encoder_init(adev_to_drm(adev), amdgpu_encoder, i);
		if (r) {
			DRM_ERROR("amdgpu_dm_initialize_drm_device: failed to initialize encoder\n");
			kfree(amdgpu_encoder);
			kfree(amdgpu_dm_connector);
			goto error;
		}

		// Functional Utility: Initializes the DRM connector.
		r = amdgpu_dm_connector_init(
			&adev->dm,
			amdgpu_dm_connector,
			i,
			amdgpu_encoder);
		if (r) {
			DRM_ERROR("amdgpu_dm_initialize_drm_device: failed to initialize connector\n");
			amdgpu_dm_encoder_destroy(&amdgpu_encoder->base);
			kfree(amdgpu_dm_connector);
			goto error;
		}
	}
	// Functional Utility: Creates DRM CRTCs and planes.
	r = amdgpu_dm_crtc_init(adev);
	if (r) {
		DRM_ERROR("amdgpu_dm_initialize_drm_device: failed to create crtc\n");
		goto error;
	}

	dc_get_version(adev->dm.dc, &info);
	drm_info(adev_to_drm(adev), "DC: %d.%d.%d\n",
			info.version_major,
			info.version_minor,
			info.version_patch);

	return 0;

error:
	amdgpu_dm_destroy_drm_device(&adev->dm);
	return r;
}

static void amdgpu_dm_destroy_drm_device(struct amdgpu_display_manager *dm)
{
	struct drm_device *dev = dm->ddev;
	struct drm_connector *connector;
	struct drm_connector_list_iter conn_iter;

	// Block Logic: Iterates through all registered DRM connectors and destroys them.
	drm_connector_list_iter_begin(dev, &conn_iter);
	drm_for_each_connector_iter(connector, &conn_iter) {
		drm_connector_unregister(connector);
		amdgpu_dm_connector_destroy(connector);
	}
	drm_connector_list_iter_end(&conn_iter);

	// Functional Utility: Cleans up the DRM mode configuration.
	drm_mode_config_cleanup(dev);
	dm->mode_config_initialized = false;
}

static int amdgpu_dm_connector_init(struct amdgpu_display_manager *dm,
				    struct amdgpu_dm_connector *amdgpu_dm_connector,
				    u32 link_index,
				    struct amdgpu_encoder *amdgpu_encoder)
{
	struct drm_device *dev = dm->ddev;
	struct dc_link *link = dm->dc->links[link_index];
	struct drm_connector *connector = &amdgpu_dm_connector->base;
	enum drm_connector_type connector_type;
	int id;

	// Functional Utility: Maps DC connector type to DRM connector type.
	connector_type = amdgpu_dm_get_connector_type(link);

	if (connector_type == DRM_MODE_CONNECTOR_Unknown)
		return -EINVAL;

	// Functional Utility: Clears the connector structure.
	memset(amdgpu_dm_connector, 0, sizeof(struct amdgpu_dm_connector));

	amdgpu_dm_connector->dc_link = link;
	amdgpu_dm_connector->encoder = amdgpu_encoder;
	amdgpu_dm_connector->connector_id = link_index;

	// Functional Utility: Initializes the DRM connector.
	drm_connector_init(dev, connector, &amdgpu_dm_connector_funcs,
			   connector_type);

	drm_connector_helper_add(connector, &amdgpu_dm_connector_helper_funcs);

	// Block Logic: Initializes DP MST (Multi-Stream Transport) manager if the link supports it.
	if (dc_link_is_dp_mst_supported(link)) {
		dm_init_mst_mgr(amdgpu_dm_connector, link_index);
		connector->funcs = &amdgpu_dm_connector_funcs;
	} else {
		// Functional Utility: Attaches the connector to the encoder for non-MST links.
		drm_connector_attach_encoder(connector, &amdgpu_encoder->base);
	}

	// Functional Utility: Registers the connector.
	drm_connector_register(connector);

	// Functional Utility: Creates connector properties.
	amdgpu_dm_connector_create_properties(amdgpu_dm_connector);

	// Functional Utility: Creates an I2C adapter for DDC (Display Data Channel) communication.
	id = amdgpu_i2c_create(dm->adev, &amdgpu_dm_connector->dc_link->ddc, "DM_DDC");

	if (id != -1)
		amdgpu_dm_connector->i2c =
			amdgpu_i2c_lookup(dm->adev, &amdgpu_dm_connector->dc_link->ddc, "DM_DDC");

	// Functional Utility: Creates an I2C adapter for OEM DDC communication.
	id = amdgpu_i2c_create(dm->adev, &amdgpu_dm_connector->dc_link->edid_lpt_ddc, "DM_DDC_OEM");
	if (id != -1)
		amdgpu_dm_connector->oem_i2c =
			amdgpu_i2c_lookup(dm->adev, &amdgpu_dm_connector->dc_link->edid_lpt_ddc, "DM_DDC_OEM");

	// Functional Utility: Initializes HPD (Hot Plug Detect) IRQ parameters.
	amdgpu_dm_hpd_irq_params_init(amdgpu_dm_connector->dc_link->link_index,
					&dm->hpd_irq_params[amdgpu_dm_connector->dc_link->link_index],
					dm_hpd_irq_high_irq);
	amdgpu_dm_hpd_irq_params_init(amdgpu_dm_connector->dc_link->link_index,
					&dm->hpd_irq_params[amdgpu_dm_connector->dc_link->link_index],
					dm_hpd_rx_irq_high_irq);
	return 0;
}

static int amdgpu_dm_connector_get_modes(struct drm_connector *connector)
{
	struct amdgpu_dm_connector *amdgpu_dm_connector = to_amdgpu_dm_connector(connector);
	struct dc_link *dc_link = amdgpu_dm_connector->dc_link;
	struct dc_sink *dc_sink;
	struct drm_display_mode *mode;
	int count = 0;

	// Block Logic: Returns if connector is an MST top connector and MST acquisition has failed.
	if (amdgpu_dm_connector->mst_mgr.mst_state == MST_IS_TOP_CONNECTOR &&
			amdgpu_dm_connector->mst_mgr.mst_root->cbs.get_mst_state_info(&amdgpu_dm_connector->mst_mgr.mst_root->aux, true)->mst_state_flags.mst_acq_fail)
		return 0;

	// Block Logic: If the connection type is not MST branch, proceeds with mode retrieval.
	if (amdgpu_dm_connector->detected_type != dc_connection_mst_branch) {
		dc_sink = dc_link_detect_sink(dc_link, &amdgpu_dm_connector->prev_sink_count);
		if (amdgpu_dm_connector->dc_sink != dc_sink) {
			if (amdgpu_dm_connector->dc_sink)
				dc_sink_release(amdgpu_dm_connector->dc_sink);
			amdgpu_dm_connector->dc_sink = dc_sink;
		}

		// Block Logic: If a sink is detected, adds modes from EDID.
		if (amdgpu_dm_connector->dc_sink) {
			struct edid *edid;

			edid = (struct edid *)amdgpu_dm_connector->dc_sink->dc_edid.raw_edid;

			if (edid)
				// Functional Utility: Adds modes from EDID to the DRM connector.
				count = drm_add_edid_modes(connector, edid);
			else
				DRM_INFO("amdgpu_dm_connector_get_modes: dc_sink has no edid\n");

			// Functional Utility: Updates subconnector property based on detected type.
			update_subconnector_property(amdgpu_dm_connector);
		}
	} else {
		// Block Logic: For MST branch devices, adds a default 640x480 mode if no sinks.
		if (!amdgpu_dm_connector->dc_sink) {
			mode = drm_mode_duplicate(connector->dev, &amdgpu_dm_mst_fake_mode);
			drm_mode_probed_add(connector, mode);
			count = 1;
		} else {
			struct edid *edid;

			edid = (struct edid *)amdgpu_dm_connector->dc_sink->dc_edid.raw_edid;

			if (edid)
				count = drm_add_edid_modes(connector, edid);
			else
				DRM_INFO("amdgpu_dm_connector_get_modes: dc_sink has no edid\n");
		}
	}

	// Functional Utility: Initializes FBC (Frame Buffer Compression) if applicable.
	amdgpu_dm_fbc_init(connector);

	return count;
}

static void amdgpu_dm_atomic_commit_tail(struct drm_atomic_state *state)
{
	struct drm_device *dev = state->dev;
	struct amdgpu_device *adev = dev->dev_private;
	struct dc_state *context;

	// Functional Utility: Retrieves the Display Core state from the atomic state.
	context = (struct dc_state *)state->private_data;

	// Block Logic: Commits the Display Core state and then releases it.
	if (context) {
		dc_commit_state(adev->dm.dc, context);
		dc_release_state(context);
	}

	// Functional Utility: Calls the atomic helper commit tail function.
	drm_atomic_helper_commit_tail(state);
}

static int amdgpu_dm_atomic_check(struct drm_device *dev,
				  struct drm_atomic_state *state)
{
	struct amdgpu_device *adev = dev->dev_private;
	struct dc_state *context = dc_create_state(adev->dm.dc);
	int res;

	// Block Logic: Handles Display Core state creation failure.
	if (!context)
		return -ENOMEM;

	// Functional Utility: Stores the DC context in the atomic state.
	state->private_data = context;

	// Functional Utility: Calls the atomic helper to perform checks.
	res = drm_atomic_helper_check(dev, state);
	// Block Logic: If checks fail, releases the DC state.
	if (res)
		dc_release_state(context);

	return res;
}