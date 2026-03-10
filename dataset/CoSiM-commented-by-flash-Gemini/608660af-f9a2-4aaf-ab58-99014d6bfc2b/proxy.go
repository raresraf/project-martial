/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// Package main provides the entry point for the kube-proxy application.
// This component of Kubernetes maintains network rules on nodes and performs connection forwarding
// for services to ensure that requests to a Service are properly routed to its backend Pods.
package main

import (
	"fmt"
	"os"
	"runtime"

	"k8s.io/kubernetes/cmd/kube-proxy/app"
	"k8s.io/kubernetes/cmd/kube-proxy/app/options"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/version/verflag"

	"github.com/spf13/pflag"
)

// init function is executed before the main function.
// Functional Utility: Registers the default health check endpoint.
func init() {
	healthz.DefaultHealthz()
}

// main function serves as the primary entry point for the kube-proxy process.
// Architectural Intent: Initializes and runs the kube-proxy server, handling configuration,
// logging, version information, and error management.
func main() {
	// Block Logic: Sets the maximum number of CPUs that can be executing simultaneously.
	// This optimizes Go runtime scheduling for multi-core processors.
	runtime.GOMAXPROCS(runtime.NumCPU())
	// Block Logic: Creates a new, default configuration object for the kube-proxy.
	config := options.NewProxyConfig()
	// Block Logic: Registers command-line flags from the configuration object with the global flag set.
	config.AddFlags(pflag.CommandLine)

	// Block Logic: Initializes command-line flags, parsing arguments provided at startup.
	util.InitFlags()
	// Block Logic: Initializes the logging system for the application.
	util.InitLogs()
	// Functional Utility: Ensures that all buffered logs are flushed to their destination before the application exits.
	defer util.FlushLogs()

	// Block Logic: Checks if a version flag was requested (e.g., --version) and prints version info, then exits if so.
	verflag.PrintAndExitIfRequested()

	// Block Logic: Attempts to create a new kube-proxy server instance with the parsed configuration.
	// Error Handling: If server creation fails, an error message is printed to stderr and the process exits with status 1.
	s, err := app.NewProxyServerDefault(config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	// Block Logic: Starts the kube-proxy server, which then begins to watch Kubernetes API server for changes
	// and updates network rules accordingly.
	// Error Handling: If the server encounters a runtime error, an error message is printed to stderr and the process exits with status 1.
	if err = s.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
