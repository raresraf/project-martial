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

// Package app implements a standalone entrypoint for the kubectl command.
package app

import (
	"os"

	"k8s.io/kubernetes/pkg/kubectl/cmd"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

/*
WARNING: this logic is duplicated, with minor changes, in cmd/hyperkube/kubectl.go
Any salient changes here will need to be manually reflected in that file.
*/
// Run creates and executes the default kubectl command.
// It is the main entrypoint for the kubectl executable.
//
// Note: This function's logic is intentionally duplicated in
// cmd/hyperkube/kubectl.go. Any significant changes made here should be
// mirrored in that file.
func Run() error {
	// Create a new factory for creating clients and other dependencies.
	// Passing nil results in a default factory that infers configuration
	// from standard locations (e.g., ~/.kube/config).
	// The factory is then used to construct the root kubectl command.
	cmd := cmd.NewKubectlCommand(cmdutil.NewFactory(nil), os.Stdin, os.Stdout, os.Stderr)
	// Execute the command, which will parse command-line arguments and run
	// the appropriate sub-command.
	return cmd.Execute()
}
