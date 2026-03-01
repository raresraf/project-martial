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

// gen_kube_docs is a command-line utility that auto-generates Markdown
// documentation for the main Kubernetes components (kube-apiserver, kubelet, etc.).
package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra/doc"
	"k8s.io/kubernetes/cmd/genutils"
	apiservapp "k8s.io/kubernetes/cmd/kube-apiserver/app"
	cmapp "k8s.io/kubernetes/cmd/kube-controller-manager/app"
	proxyapp "k8s.io/kubernetes/cmd/kube-proxy/app"
	kubeletapp "k8s.io/kubernetes/cmd/kubelet/app"
	schapp "k8s.io/kubernetes/plugin/cmd/kube-scheduler/app"
)

func main() {
	// This tool expects exactly two arguments:
	// 1. The output directory for the generated documentation.
	// 2. The name of the Kubernetes module to document.
	// It uses os.Args directly to avoid interfering with the flag parsing of the
	// Kubernetes commands themselves, which is necessary for doc generation.
	path := ""
	module := ""
	if len(os.Args) == 3 {
		path = os.Args[1]
		module = os.Args[2]
	} else {
		fmt.Fprintf(os.Stderr, "usage: %s [output directory] [module] \n", os.Args[0])
		os.Exit(1)
	}

	// Prepare the output directory.
	outDir, err := genutils.OutDir(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to get output directory: %v\n", err)
		os.Exit(1)
	}

	// Select the appropriate module and generate its documentation.
	switch module {
	case "kube-apiserver":
		// Create a new command object for the kube-apiserver.
		apiserver := apiservapp.NewAPIServerCommand()
		// Generate a Markdown documentation tree for the command.
		doc.GenMarkdownTree(apiserver, outDir)
	case "kube-controller-manager":
		// generate docs for kube-controller-manager
		controllermanager := cmapp.NewControllerManagerCommand()
		doc.GenMarkdownTree(controllermanager, outDir)
	case "kube-proxy":
		// generate docs for kube-proxy
		proxy := proxyapp.NewProxyCommand()
		doc.GenMarkdownTree(proxy, outDir)
	case "kube-scheduler":
		// generate docs for kube-scheduler
		scheduler := schapp.NewSchedulerCommand()
		doc.GenMarkdownTree(scheduler, outDir)
	case "kubelet":
		// generate docs for kubelet
		kubelet := kubeletapp.NewKubeletCommand()
		doc.GenMarkdownTree(kubelet, outDir)
	default:
		fmt.Fprintf(os.Stderr, "Module %s is not supported", module)
		os.Exit(1)
	}
}
