/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package app provides the main entry point for the kubectl command-line tool.
// Its primary function is to initialize and execute the core kubectl command,
// enabling users to interact with Kubernetes clusters.
package app

import (
	"os" // Imports the operating system package for accessing standard I/O streams.

	// Imports Kubernetes-specific packages for kubectl command implementation and utilities.
	"k8s.io/kubernetes/pkg/kubectl/cmd"        // Provides the core kubectl command definition.
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util" // Provides utility functions for command creation, including a factory for dependencies.
)

/*
WARNING: this logic is duplicated, with minor changes, in cmd/hyperkube/kubectl.go
Any salient changes here will need to be manually reflected in that file.
*/
// Run initializes and executes the main Kubectl command.
// This function serves as the entry point for the kubectl CLI application.
// It sets up the command structure, configures I/O streams, and handles command execution.
// Returns an error if the command execution fails.
func Run() error {
	// NewKubectlCommand creates the root command for kubectl,
	// taking a factory for dependencies (like Kubernetes clients) and standard I/O streams.
	cmd := cmd.NewKubectlCommand(cmdutil.NewFactory(nil), os.Stdin, os.Stdout, os.Stderr)
	// Execute the command, which parses arguments and runs the appropriate subcommands.
	return cmd.Execute()
}
