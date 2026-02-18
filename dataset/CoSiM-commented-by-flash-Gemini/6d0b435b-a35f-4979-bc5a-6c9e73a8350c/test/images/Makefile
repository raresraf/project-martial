# @file Makefile
# @brief Manages the building and pushing of Kubernetes End-to-End (E2E) test container images.
#
# This Makefile automates the process of creating container images for various
# test components and optionally pushing them to a specified container registry.
# It relies on an external `image-util.sh` script to perform the core build
# and push operations.
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

# @brief The container registry to use for pulling and pushing images.
# Defaults to `gcr.io/kubernetes-e2e-test-images`.
REGISTRY ?= gcr.io/kubernetes-e2e-test-images
# @brief The ARM version for Go cross-compilation, defaults to 7.
GOARM ?= 7
# @brief The QEMU version used for multi-architecture builds.
QEMUVERSION=v2.9.1
# @brief The GoLang version used for building.
GOLANG_VERSION=1.13.6
export # Export all variables to sub-processes.

# Block Logic: Checks if the WHAT variable is set, which is mandatory for build operations.
ifndef WHAT
$(error WHAT is a required variable, ex: make all WHAT=net)
endif

# @brief Default target: Builds the container image(s) specified by the WHAT variable.
# Example: `make all WHAT=clusterapi-tester`
all: all-container

# @brief Builds the container image(s) specified by the WHAT variable.
# This target invokes the `image-util.sh` script with the 'build' command.
# Args:
#   WHAT: Directory names corresponding to the images to build.
all-container:
	./image-util.sh build $(WHAT)

# @brief Builds and then pushes the container image(s) specified by the WHAT variable.
# This target first calls `all-container` to build, then invokes `image-util.sh` with the 'push' command.
# Args:
#   WHAT: Directory names corresponding to the images to push.
all-push: all-container
	./image-util.sh push $(WHAT)

# .PHONY: Declares targets that do not correspond to actual files, ensuring they are always executed.
.PHONY: all all-push all-container
