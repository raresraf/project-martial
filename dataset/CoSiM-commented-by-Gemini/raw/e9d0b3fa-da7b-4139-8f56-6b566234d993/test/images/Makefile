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

# Include shared rules and variables from a common manifest file.
# Functional Utility: This promotes modularity and reuse of build logic.
include  ../../hack/make-rules/Makefile.manifest

# --- Variable Definitions ---

# The container registry to push images to. Can be overridden from the command line.
REGISTRY ?= gcr.io/kubernetes-e2e-test-images
# The ARM architecture version to target for Go builds.
GOARM ?= 7
# The version of QEMU to use for emulation.
QEMUVERSION=v2.9.1
# The version of the Go compiler to use.
GOLANG_VERSION=1.13.6
# Export these variables so they are available to sub-shells, like the one
# running the image-util.sh script.
export

# --- Build Guard ---

# Enforce that the 'WHAT' variable is set. This variable specifies which
# container image (i.e., which directory) to build.
# Invariant: The build will fail immediately if WHAT is not provided.
ifndef WHAT
$(error WHAT is a required variable, ex: make all WHAT=net)
endif

# --- Build Targets ---

# Build code.
#
# Args:
#   WHAT: Directory names to build.
#
# Example:
#   make all WHAT=clusterapi-tester

# Default target. Builds the container image but does not push it.
all: all-container

# Core build target. Executes the build script for the specified component.
all-container:
	./image-util.sh build $(WHAT)

# Build and push target. Depends on `all-container` to ensure the image is
# built before being pushed.
all-push: all-container
	./image-util.sh push $(WHAT)

# Declare targets that are not actual files.
# Functional Utility: This prevents `make` from getting confused if a file with
# the same name as a target (e.g., a file named 'all') exists in the directory.
.PHONY: all all-push all-container
