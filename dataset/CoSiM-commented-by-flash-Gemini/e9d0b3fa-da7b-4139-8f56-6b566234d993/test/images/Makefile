# @e9d0b3fa-da7b-4139-8f56-6b566234d993/test/images/Makefile
# @brief Build orchestration for Kubernetes End-to-End (E2E) containerized test artifacts.
#
# Functional Intent: Automates the lifecycle of multi-architecture container images 
# required for Kubernetes integration testing. It provides a standardized interface 
# to build and distribute specialized test components across various registries.
#
# Domain: Kubernetes CI/CD, Containerization, Build Systems.
#
# Copyright 2017 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Block Logic: Includes shared manifest rules for image versioning and metadata management.
include  ../../hack/make-rules/Makefile.manifest

# Functional Utility: Configurable parameters for image distribution and compilation targets.
# REGISTRY: Target repository for artifact storage.
# GOARM: Instruction set version for ARM-based cross-compilation.
REGISTRY ?= gcr.io/kubernetes-e2e-test-images
GOARM ?= 7
QEMUVERSION=v2.9.1
GOLANG_VERSION=1.13.6
export # Ensures environment consistency for the image-util.sh sub-shell.

# Block Logic: Validation of mandatory input parameters.
# Pre-condition: The 'WHAT' variable must specify one or more target image directories.
# Invariant: Build processes will not initiate without a defined target scope.
ifndef WHAT
$(error WHAT is a required variable, ex: make all WHAT=net)
endif

# Functional Utility: Aggregates all build-related tasks into a single entry point.
all: all-container

# Block Logic: Orchestrates the image construction phase.
# Logic: Delegates the low-level container assembly to an external utility script, 
# passing the targeted components as arguments.
all-container:
	./image-util.sh build $(WHAT)

# Block Logic: Orchestrates the artifact distribution phase.
# Logic: Sequences the build process (dependency) before initiating the registry push operation.
all-push: all-container
	./image-util.sh push $(WHAT)

# .PHONY: Ensures that targets are treated as actions rather than filesystem lookups.
.PHONY: all all-push all-container
