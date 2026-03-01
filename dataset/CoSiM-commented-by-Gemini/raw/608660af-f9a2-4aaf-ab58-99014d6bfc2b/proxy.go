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

// The main package for the Kubernetes proxy server.
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

// init runs before main and is used to set up default healthz checks.
func init() {
	// Functional-Utility: Registers the default healthz HTTP handlers.
	// This allows the kube-proxy to be monitored for liveness.
	healthz.DefaultHealthz()
}

// main is the entry point for the kube-proxy application.
// Its primary responsibility is to parse configuration, initialize the
// environment, and start the proxy server.
func main() {
	// Performance-Optimization: Set the number of operating system threads that can
	// execute user-level Go code simultaneously to the number of available CPUs.
	runtime.GOMAXPROCS(runtime.NumCPU())

	// Create a new configuration object for the proxy server.
	config := options.NewProxyConfig()
	// Register command-line flags for the proxy configuration.
	config.AddFlags(pflag.CommandLine)

	// Initialize and parse command-line flags.
	util.InitFlags()
	// Initialize logging components.
	util.InitLogs()
	// Defer a log flush to ensure all buffered logs are written before exit.
	defer util.FlushLogs()

	// Handle the --version flag, printing version information and exiting if present.
	verflag.PrintAndExitIfRequested()

	// Create a new ProxyServer instance using the parsed configuration.
	// This is where the application's core logic is instantiated.
	s, err := app.NewProxyServerDefault(config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	// Start the proxy server's main run loop.
	// This call is blocking and will run until the server is shut down.
	if err = s.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
