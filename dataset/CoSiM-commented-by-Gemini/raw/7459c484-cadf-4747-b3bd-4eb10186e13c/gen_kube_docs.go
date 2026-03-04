/**
 * @file gen_kube_docs.go
 * @brief A command-line tool for auto-generating Markdown documentation for Kubernetes components.
 *
 * This program leverages the `cobra` library to generate documentation for the command-line
 * interfaces of various Kubernetes services like kube-apiserver, kube-controller-manager,
 * kube-proxy, kube-scheduler, and kubelet. It takes an output directory and a module
 * name as arguments and creates a tree of Markdown files in that directory.
 */
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

package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/cmd/genutils"
	apiservapp "k8s.io/kubernetes/cmd/kube-apiserver/app"
	cmapp "k8s.io/kubernetes/cmd/kube-controller-manager/app"
	proxyapp "k8s.io/kubernetes/cmd/kube-proxy/app"
	kubeletapp "k8s.io/kubernetes/cmd/kubelet/app"
	schapp "k8s.io/kubernetes/plugin/cmd/kube-scheduler/app"
)

func main() {
	// use os.Args instead of "flags" because "flags" will mess up the man pages!
	path := ""
	module := ""
	// Basic command-line argument parsing for output directory and module name.
	if len(os.Args) == 3 {
		path = os.Args[1]
		module = os.Args[2]
	} else {
		fmt.Fprintf(os.Stderr, "usage: %s [output directory] [module] \n", os.Args[0])
		os.Exit(1)
	}

	outDir, err := genutils.OutDir(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to get output directory: %v\n", err)
		os.Exit(1)
	}

	/**
	 * @brief Selects the appropriate Kubernetes module and generates its documentation.
	 * This switch statement uses the 'module' argument to determine which Kubernetes
	 * component's documentation to generate. For each case, it creates the root
	 * cobra command for that component and then calls `cobra.GenMarkdownTree` to
	 * generate the documentation.
	 */
	switch module {
	case "kube-apiserver":
		// generate docs for kube-apiserver
		apiserver := apiservapp.NewAPIServerCommand()
		cobra.GenMarkdownTree(apiserver, outDir)
	case "kube-controller-manager":
		// generate docs for kube-controller-manager
		controllermanager := cmapp.NewControllerManagerCommand()
		cobra.GenMarkdownTree(controllermanager, outDir)
	case "kube-proxy":
		// generate docs for kube-proxy
		proxy := proxyapp.NewProxyCommand()
		cobra.GenMarkdownTree(proxy, outDir)
	case "kube-scheduler":
		// generate docs for kube-scheduler
		scheduler := schapp.NewSchedulerCommand()
		cobra.GenMarkdownTree(scheduler, outDir)
	case "kubelet":
		// generate docs for kubelet
		kubelet := kubeletapp.NewKubeletCommand()
		cobra.GenMarkdownTree(kubelet, outDir)
	default:
		fmt.Fprintf(os.Stderr, "Module %s is not supported", module)
		os.Exit(1)
	}
}