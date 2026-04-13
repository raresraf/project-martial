/*
Copyright 2014 The Kubernetes Authors.

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

// Package main provides the entry point for the Kubernetes kube-proxy command.
// It initializes and executes the kube-proxy command-line application,
// handling metric registration and command execution.
package main

import (
	"os"

	"k8s.io/component-base/cli"
	// Blank import for client metric registration:
	// This import ensures that the `restclient` package's `init()` function is called,
	// registering Prometheus metrics related to client operations.
	_ "k8s.io/component-base/metrics/prometheus/restclient"
	// Blank import for version metric registration:
	// This import ensures that the `version` package's `init()` function is called,
	// registering Prometheus metrics related to the application's version.
	_ "k8s.io/component-base/metrics/prometheus/version"
	"k8s.io/kubernetes/cmd/kube-proxy/app"
)

// main is the entry point for the kube-proxy command-line application.
// It sets up the command, runs it, and exits with the appropriate status code.
func main() {
	command := app.NewProxyCommand() // Creates a new kube-proxy command instance.
	code := cli.Run(command)         // Executes the command and captures its exit code.
	os.Exit(code)                    // Exits the application with the captured code.
}
