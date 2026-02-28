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

	"k8s.ioio/kubernetes/cmd/kube-proxy/app"
	"k8s.io/kubernetes/cmd/kube-proxy/app/options"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/flag"
	"k8s.io/kubernetes/pkg/version/verflag"

	"github.com/spf13/pflag"
)

// init is a standard Go function that is called before main().
// It is used here to set up the default health checking endpoints.
func init() {
	healthz.DefaultHealthz()
}

// main is the entry point for the kube-proxy application.
func main() {
	// Set the number of CPUs to use to the number of available logical CPUs.
	// This is a common optimization to maximize performance.
	runtime.GOMAXPROCS(runtime.NumCPU())

	// Create a new ProxyConfig object with default settings.
	config := options.NewProxyConfig()
	// Add command-line flags for the proxy configuration.
	config.AddFlags(pflag.CommandLine)

	// Initialize flag parsing.
	flag.InitFlags()

	// Initialize logging and defer flushing of logs until the program exits.
	util.InitLogs()
	defer util.FlushLogs()

	// Check if a version flag (`--version`) has been passed and, if so,
	// print the version and exit.
	verflag.PrintAndExitIfRequested()

	// Create a new ProxyServer instance from the default configuration.
	// This is where the application object is instantiated.
	s, err := app.NewProxyServerDefault(config)
	if err != nil {
		// If server creation fails, report the error and exit.
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	// Run the proxy server. This is a blocking call that starts the main
	// application loop and will not return until the server is stopped or
	// an unrecoverable error occurs.
	if err = s.Run(); err != nil {
		// If the server fails to run, report the error and exit.
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
