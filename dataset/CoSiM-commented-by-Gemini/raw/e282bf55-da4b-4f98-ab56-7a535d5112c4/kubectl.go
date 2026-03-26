/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT' WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package app is the main entry point for the kubectl binary.
package app

import (
	"os"

	"k8s.io/kubernetes/pkg/kubectl/cmd"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

// Run creates and executes the `kubectl` command.
// It returns an error if the command fails to execute.
// Its primary function is to bootstrap the command-line utility by constructing
// the root command and triggering its execution.
func Run() error {
	// A factory provides a standard way of creating and configuring clients
	// and other required components for kubectl commands. Passing nil results
	// in a default factory that will look for a standard kubeconfig file.
	factory := cmdutil.NewFactory(nil)

	// NewKubectlCommand builds the root command 'kubectl' and attaches all its
	// subcommands. It is configured with the factory for creating clients and
	// with the standard OS input/output streams.
	cmd := cmd.NewKubectlCommand(factory, os.Stdin, os.Stdout, os.Stderr)

	// Execute is the main entry point of the cobra-based command. It parses the
	// command-line arguments and runs the appropriate subcommand.
	return cmd.Execute()
}
