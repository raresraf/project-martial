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

// Package main is the entry point for the Kubernetes kube-proxy daemon.
// kube-proxy is a network proxy that runs on each node in the cluster,
// implementing part of the Kubernetes Service concept. It maintains network rules
// on nodes and performs connection forwarding.
package main

import (
	"fmt"
	"os"
	"runtime"

	"k8s.io/kubernetes/cmd/kube-proxy/app"
	"k8s.io/kubernetes/cmd/kube-proxy/app/options"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/flag"
	"k8s.io/kubernetes/pkg/version/verflag"

	"github.com/spf13/pflag"
)

// init is called automatically at program startup.
// It is used here to register the default health check endpoint.
func init() {
	healthz.DefaultHealthz()
}

// main is the entry point for the kube-proxy application.
// It initializes and configures the proxy server, parses command-line flags,
// sets up logging, handles version information requests, and starts the proxy server.
//
// Time Complexity: O(1) for initialization and startup.
// Space Complexity: O(1) for initial configuration structures.
func main() {
	// Block Logic: Sets the number of operating system threads that can
	// execute user-level Go code simultaneously. This is typically set
	// to the number of available CPU cores for optimal performance.
	runtime.GOMAXPROCS(runtime.NumCPU())
	
	// Inline: Creates a new default configuration object for the kube-proxy server.
	config := options.NewProxyConfig()
	// Inline: Adds command-line flags defined in the proxy configuration to the
	// global pflag set, allowing them to be parsed from the command line.
	config.AddFlags(pflag.CommandLine)

	// Block Logic: Parses the command-line flags that were added to pflag.CommandLine.
	flag.InitFlags()
	// Block Logic: Initializes the application's logging system.
	util.InitLogs()
	// Inline: Ensures that all buffered log entries are flushed to their destination
	// before the program exits, even if an unexpected panic occurs.
	defer util.FlushLogs()

	// Block Logic: Checks if the user requested version information via command-line flags.
	// If so, it prints the version details and exits the application.
	verflag.PrintAndExitIfRequested()

	// Block Logic: Creates a new instance of the kube-proxy server with the
	// parsed configuration. This involves setting up various components
	// like network proxies and watchers.
	s, err := app.NewProxyServerDefault(config)
	if err != nil {
		// Inline: If server creation fails, print the error to standard error and exit with a non-zero status.
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	// Block Logic: Starts the kube-proxy server, initiating its main operational loop.
	// This function typically blocks indefinitely, handling network traffic and Kubernetes events.
	if err = s.Run(); err != nil {
		// Inline: If the server encounters a runtime error, print it to standard error and exit.
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
