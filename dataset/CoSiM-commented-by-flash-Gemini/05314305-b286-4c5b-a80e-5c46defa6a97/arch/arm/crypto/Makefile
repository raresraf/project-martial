#
# @brief Makefile for compiling architecture-specific CryptoAPI modules for ARM.
# This Makefile orchestrates the compilation of various cryptographic algorithms
# for the ARM architecture, including AES, SHA1, SHA512, BLAKE2B, NHPOLY1305,
# and CURVE25519, based on kernel configuration options.
#
# Domain: Kernel Cryptography, ARM Architecture, Build System.
#

# SPDX-License-Identifier: GPL-2.0
#
# Arch-specific CryptoAPI modules.
#

# Block Logic: Conditional inclusion of standard cryptographic modules.
# These assignments dynamically include object files for various cryptographic
# algorithms (AES, SHA1, SHA512, BLAKE2B, NHPOLY1305, CURVE25519) into the build
# based on whether their corresponding kernel configuration options are enabled.
# This mechanism allows for modular and configurable cryptographic support.
obj-$(CONFIG_CRYPTO_AES_ARM) += aes-arm.o
obj-$(CONFIG_CRYPTO_AES_ARM_BS) += aes-arm-bs.o
obj-$(CONFIG_CRYPTO_SHA1_ARM) += sha1-arm.o
obj-$(CONFIG_CRYPTO_SHA1_ARM_NEON) += sha1-arm-neon.o
obj-$(CONFIG_CRYPTO_SHA512_ARM) += sha512-arm.o
obj-$(CONFIG_CRYPTO_BLAKE2B_NEON) += blake2b-neon.o
obj-$(CONFIG_CRYPTO_NHPOLY1305_NEON) += nhpoly1305-neon.o
obj-$(CONFIG_CRYPTO_CURVE25519_NEON) += curve25519-neon.o

# Block Logic: Conditional inclusion of Cryptography Extensions (CE) modules.
# Similar to the standard modules, these lines conditionally include object files
# for ARM Cryptography Extensions (AES-CE, SHA1-CE, GHASH-CE) when the
# corresponding kernel configuration flags are set, leveraging hardware acceleration.
obj-$(CONFIG_CRYPTO_AES_ARM_CE) += aes-arm-ce.o
obj-$(CONFIG_CRYPTO_SHA1_ARM_CE) += sha1-arm-ce.o
obj-$(CONFIG_CRYPTO_GHASH_ARM_CE) += ghash-arm-ce.o

# Block Logic: Grouping of source files for cryptographic algorithms.
# These variables define the object files that constitute each cryptographic
# module. They typically separate core algorithm implementations from
# architecture-specific glue code or Neon optimizations.
aes-arm-y	:= aes-cipher-core.o aes-cipher-glue.o
aes-arm-bs-y	:= aes-neonbs-core.o aes-neonbs-glue.o
sha1-arm-y	:= sha1-armv4-large.o sha1_glue.o
sha1-arm-neon-y	:= sha1-armv7-neon.o sha1_neon_glue.o
sha512-arm-neon-$(CONFIG_KERNEL_MODE_NEON) := sha512-neon-glue.o
sha512-arm-y	:= sha512-core.o sha512-glue.o $(sha512-arm-neon-y)
blake2b-neon-y  := blake2b-neon-core.o blake2b-neon-glue.o
sha1-arm-ce-y	:= sha1-ce-core.o sha1-ce-glue.o
aes-arm-ce-y	:= aes-ce-core.o aes-ce-glue.o
ghash-arm-ce-y	:= ghash-ce-core.o ghash-ce-glue.o
nhpoly1305-neon-y := nh-neon-core.o nhpoly1305-neon-glue.o
curve25519-neon-y := curve25519-core.o curve25519-glue.o

# Block Logic: Defines commands for generating assembly source files from Perl scripts.
# The 'quiet_cmd_perl' and 'cmd_perl' variables set up the execution of Perl
# scripts, often used for generating highly optimized assembly code for
# cryptographic primitives based on specific ARM instruction sets.
quiet_cmd_perl = PERL    $@
      cmd_perl = $(PERL) $(<) > $(@)

# This rule specifies that any file matching the pattern `$(obj)/%-core.S`
# depends on a corresponding Perl script `$(src)/%-armv4.pl`.
# The `call cmd,perl` executes the Perl script to generate the assembly file.
$(obj)/%-core.S: $(src)/%-armv4.pl
	$(call cmd,perl)

# Functional Utility: Specifies files to be removed during a 'make clean' operation.
# This ensures that generated files, such as the sha512-core.S assembly source,
# are properly cleaned from the build directory.
clean-files += sha512-core.S

# Block Logic: Defines architecture-specific assembly flags.
# The 'aflags-thumb2-y' variable sets compiler flags for Thumb-2 mode
# on ARM, used when CONFIG_THUMB2_KERNEL is enabled.
# These flags are then applied to the compilation of sha512-core.o,
# ensuring correct assembly and linking for the target architecture.
aflags-thumb2-$(CONFIG_THUMB2_KERNEL)  := -U__thumb2__ -D__thumb2__=1

AFLAGS_sha512-core.o += $(aflags-thumb2-y)
