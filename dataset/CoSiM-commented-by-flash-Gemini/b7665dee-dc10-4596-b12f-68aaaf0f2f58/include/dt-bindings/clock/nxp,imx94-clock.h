/* SPDX-License-Identifier: (GPL-2.0-only OR BSD-2-Clause) */
/*
 * @b7665dee-dc10-4596-b12f-68aaaf0f2f58/include/dt-bindings/clock/nxp,imx94-clock.h
 * @brief Devicetree binding constants for the NXP i.MX94 clock controller.
 *
 * Functional Intent: Provides symbolic constants for clock indices and gates, 
 * ensuring consistency between the Linux kernel clock driver and the 
 * hardware description in Devicetree files.
 *
 * Copyright 2025 NXP
 */

#ifndef __DT_BINDINGS_CLOCK_IMX94_H
#define __DT_BINDINGS_CLOCK_IMX94_H

/**
 * Functional Utility: Selection indices for multiplexed clock sources.
 */
#define IMX94_CLK_DISPMIX_CLK_SEL	0

/**
 * Functional Utility: Gating indices for the Display Mix LVDS clock.
 */
#define IMX94_CLK_DISPMIX_LVDS_CLK_GATE	0

#endif /* __DT_BINDINGS_CLOCK_IMX94_H */
