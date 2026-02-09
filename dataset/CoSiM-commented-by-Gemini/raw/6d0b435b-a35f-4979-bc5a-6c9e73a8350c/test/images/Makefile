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

#
# @file Makefile
# @brief Makefile for building and pushing container images for Kubernetes e2e tests.
#
# This Makefile serves as a wrapper around an `image-util.sh` script to simplify
# the process of building and publishing container images.
#

# Defines the default container registry. Can be overridden from the command line.
REGISTRY ?= gcr.io/kubernetes-e2e-test-images
# Defines the ARM architecture version for Go builds.
GOARM ?= 7
# Specifies the version of QEMU to be used, likely for multi-architecture builds.
QEMUVERSION=v2.9.1
# Specifies the version of Go to be used for compilation.
GOLANG_VERSION=1.13.6
export

# Pre-condition: Ensures that the 'WHAT' variable is defined. 'WHAT' specifies
# the name of the image subdirectory to be built.
ifndef WHAT
$(error WHAT is a required variable, ex: make all WHAT=net)
endif

# Build code.
#
# Args:
#   WHAT: Directory names to build.
#
# Example:
#   make all WHAT=clusterapi-tester

# Default target: builds the container images.
all: all-container

# Target to build the container image(s).
# It invokes the `image-util.sh` script with the 'build' command.
all-container:
	./image-util.sh build $(WHAT)

# Target to build and then push the container image(s).
# It depends on `all-container` to ensure the image is built before pushing.
all-push: all-container
	./image-util.sh push $(WHAT)

# Declares targets that are not actual files to prevent `make` from getting
# confused if a file with the same name exists.
.PHONY: all all-push all-container