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

// @653bb700-147c-4eb2-a6bc-2cee143f96bc/gen_kube_docs.go
// @brief This package `main` provides a utility to generate documentation
// for various Kubernetes command-line interface (CLI) tools.
// It uses the `cobra/doc` library to create markdown-formatted documentation
// for specified Kubernetes components like `kube-apiserver`, `kube-controller-manager`, etc.
package main

import (
	"fmt" // @brief Package `fmt` provides functions for formatted I/O (e.g., printing error messages).
	"os"  // @brief Package `os` provides a platform-independent interface to operating system functionality (e.g., command-line arguments, exit).

	"github.com/spf13/cobra/doc" // @brief Package `doc` from Cobra library is used to generate documentation for Cobra commands.
	"k8s.io/kubernetes/cmd/genutils" // @brief Internal Kubernetes utility package for generating output directories.
	apiservapp "k8s.io/kubernetes/cmd/kube-apiserver/app" // @brief Application package for `kube-apiserver` command, providing its Cobra command structure.
	cmapp "k8s.io/kubernetes/cmd/kube-controller-manager/app" // @brief Application package for `kube-controller-manager` command.
	proxyapp "k8s.io/kubernetes/cmd/kube-proxy/app" // @brief Application package for `kube-proxy` command.
	kubeletapp "k8s.io/kubernetes/cmd/kubelet/app" // @brief Application package for `kubelet` command.
	schapp "k8s.io/kubernetes/plugin/cmd/kube-scheduler/app" // @brief Application package for `kube-scheduler` command.
)

func main() {
	// Inline: Note to use os.Args directly rather than a flags package to avoid
	// interference with documentation generation for CLI tools that use flags themselves.
	// use os.Args instead of "flags" because "flags" will mess up the man pages!
	
	path := ""   // Declares `path` to store the output directory for generated documentation.
	module := "" // Declares `module` to store the name of the Kubernetes component to document.
	
	// Block Logic: Parses command-line arguments. Expects exactly two arguments:
	// the output directory and the module name. If not, prints usage and exits.
	if len(os.Args) == 3 {
		path = os.Args[1] // The first argument is the output directory.
		module = os.Args[2] // The second argument is the module name.
	} else {
		// Prints usage information to standard error.
		fmt.Fprintf(os.Stderr, "usage: %s [output directory] [module] \n", os.Args[0])
		os.Exit(1) // Exits with a non-zero status code indicating an error.
	}

	// Block Logic: Validates and creates the output directory if it doesn't exist.
	outDir, err := genutils.OutDir(path)
	if err != nil {
		// Prints an error message to standard error if directory creation fails.
		fmt.Fprintf(os.Stderr, "failed to get output directory: %v\n", err)
		os.Exit(1) // Exits with an error status code.
	}

	// Block Logic: Uses a switch statement to identify the specified Kubernetes module
	// and generate its documentation using the `cobra/doc` library.
	switch module {
	case "kube-apiserver":
		// generate docs for kube-apiserver
		apiserver := apiservapp.NewAPIServerCommand() // Creates the Cobra command for kube-apiserver.
		doc.GenMarkdownTree(apiserver, outDir)         // Generates markdown documentation tree.
	case "kube-controller-manager":
		// generate docs for kube-controller-manager
		controllermanager := cmapp.NewControllerManagerCommand() // Creates the Cobra command for kube-controller-manager.
		doc.GenMarkdownTree(controllermanager, outDir)           // Generates markdown documentation tree.
	case "kube-proxy":
		// generate docs for kube-proxy
		proxy := proxyapp.NewProxyCommand() // Creates the Cobra command for kube-proxy.
		doc.GenMarkdownTree(proxy, outDir)   // Generates markdown documentation tree.
	case "kube-scheduler":
		// generate docs for kube-scheduler
		scheduler := schapp.NewSchedulerCommand() // Creates the Cobra command for kube-scheduler.
		doc.GenMarkdownTree(scheduler, outDir)     // Generates markdown documentation tree.
	case "kubelet":
		// generate docs for kubelet
		kubelet := kubeletapp.NewKubeletCommand() // Creates the Cobra command for kubelet.
		doc.GenMarkdownTree(kubelet, outDir)       // Generates markdown documentation tree.
	default:
		// If an unsupported module is specified, prints an error and exits.
		fmt.Fprintf(os.Stderr, "Module %s is not supported", module)
		os.Exit(1) // Exits with an error status code.
	}
}
