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

/**
 * @file gen_kube_docs.go
 * @brief Command-line utility to generate Markdown documentation for Kubernetes components.
 * @details This program acts as a documentation generation tool for specific Kubernetes binaries.
 * It takes an output directory and the name of a Kubernetes component (e.g., "kube-apiserver", "kubelet")
 * as command-line arguments. It then uses the Cobra library's `GenMarkdownTree` function to generate
 * comprehensive Markdown documentation for the specified component's command-line interface.
 * This is crucial for maintaining up-to-date and accessible documentation for complex CLI tools.
 * @command_line_usage `gen_kube_docs [output directory] [module]`
 * @architecture Tooling/Automation script.
 * Time Complexity: Dependent on the complexity of the Cobra command trees for each component, typically fast.
 * Space Complexity: Dependent on the size of the generated documentation, typically small.
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

/**
 * @file gen_kube_docs.go
 * @brief Command-line utility to generate Markdown documentation for Kubernetes components.
 * @details This program acts as a documentation generation tool for specific Kubernetes binaries.
 * It takes an output directory and the name of a Kubernetes component (e.g., "kube-apiserver", "kubelet")
 * as command-line arguments. It then uses the Cobra library's `GenMarkdownTree` function to generate
 * comprehensive Markdown documentation for the specified component's command-line interface.
 * This is crucial for maintaining up-to-date and accessible documentation for complex CLI tools.
 * @command_line_usage `gen_kube_docs [output directory] [module]`
 * @architecture Tooling/Automation script.
 * Time Complexity: Dependent on the complexity of the Cobra command trees for each component, typically fast.
 * Space Complexity: Dependent on the size of the generated documentation, typically small.
 */
package main // Declares the package as 'main', indicating an executable program.

import (
	"fmt"  // Required for formatted I/O (e.g., printing error messages).
	"os"   // Required for interacting with the operating system (e.g., accessing command-line arguments, exiting).

	"github.com/spf13/cobra"      // Cobra library for building powerful command-line interfaces, used for documentation generation.
	"k8s.io/kubernetes/cmd/genutils" // Utility functions for documentation generation (e.g., OutDir).
	apiservapp "k8s.io/kubernetes/cmd/kube-apiserver/app"     // Imports the kube-apiserver application.
	cmapp "k8s.io/kubernetes/cmd/kube-controller-manager/app" // Imports the kube-controller-manager application.
	proxyapp "k8s.io/kubernetes/cmd/kube-proxy/app"           // Imports the kube-proxy application.
	kubeletapp "k8s.io/kubernetes/cmd/kubelet/app"             // Imports the kubelet application.
	schapp "k8s.io/kubernetes/plugin/cmd/kube-scheduler/app" // Imports the kube-scheduler application.
)

func main() {
	/**
	 * @brief Main entry point of the documentation generation program.
	 * @details This function parses command-line arguments for the output directory and the target Kubernetes module.
	 * It then dispatches to the appropriate Cobra command generator based on the module,
	 * generating Markdown documentation files.
	 */

	// Functional Utility: Parses command-line arguments.
	// Note: `os.Args` is used directly to avoid interference with Cobra's internal flag parsing
	// when generating man pages or documentation, as per the comment in the original code.
	path := ""   // @var path: Stores the output directory for the generated documentation.
	module := "" // @var module: Stores the name of the Kubernetes component for which to generate documentation.

	/**
	 * @brief Validates the number of command-line arguments.
	 * @block_logic Ensures exactly two arguments (output directory and module name) are provided.
	 * @invariant If arguments are incorrect, prints usage message to stderr and exits with status 1.
	 */
	if len(os.Args) == 3 {
		path = os.Args[1]   // Assigns the first argument as the output directory.
		module = os.Args[2] // Assigns the second argument as the module name.
	} else {
		fmt.Fprintf(os.Stderr, "usage: %s [output directory] [module] \n", os.Args[0]) // Prints usage.
		os.Exit(1)                                                                 // Exits with error status.
	}

	// Functional Utility: Resolves and creates the output directory if necessary.
	outDir, err := genutils.OutDir(path) // @var outDir: The resolved output directory path. @var err: Any error encountered.
	/**
	 * @brief Handles errors during output directory resolution.
	 * @block_logic If `genutils.OutDir` returns an error, prints the error to stderr and exits.
	 * @invariant Exits with status 1 if output directory cannot be determined or created.
	 */
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to get output directory: %v\n", err) // Prints error message.
		os.Exit(1)                                                       // Exits with error status.
	}

	/**
	 * @brief Dispatches to the appropriate documentation generation function based on the specified module.
	 * @block_logic A `switch` statement handles different Kubernetes component names.
	 * @invariant For a recognized module, `cobra.GenMarkdownTree` is called to generate docs.
	 *            For an unsupported module, an error message is printed, and the program exits.
	 */
	switch module {
	case "kube-apiserver":
		// Functional Utility: Generates documentation for kube-apiserver.
		apiserver := apiservapp.NewAPIServerCommand() // Creates the root command for kube-apiserver.
		cobra.GenMarkdownTree(apiserver, outDir)      // Generates Markdown docs from the Cobra command tree.
	case "kube-controller-manager":
		// Functional Utility: Generates documentation for kube-controller-manager.
		controllermanager := cmapp.NewControllerManagerCommand() // Creates the root command.
		cobra.GenMarkdownTree(controllermanager, outDir)        // Generates Markdown docs.
	case "kube-proxy":
		// Functional Utility: Generates documentation for kube-proxy.
		proxy := proxyapp.NewProxyCommand() // Creates the root command.
		cobra.GenMarkdownTree(proxy, outDir) // Generates Markdown docs.
	case "kube-scheduler":
		// Functional Utility: Generates documentation for kube-scheduler.
		scheduler := schapp.NewSchedulerCommand() // Creates the root command.
		cobra.GenMarkdownTree(scheduler, outDir)  // Generates Markdown docs.
	case "kubelet":
		// Functional Utility: Generates documentation for kubelet.
		kubelet := kubeletapp.NewKubeletCommand() // Creates the root command.
		cobra.GenMarkdownTree(kubelet, outDir)    // Generates Markdown docs.
	default:
		// Error Handling: If the module is not recognized.
		fmt.Fprintf(os.Stderr, "Module %s is not supported\n", module) // Prints error message.
		os.Exit(1)                                                    // Exits with error status.
	}
}
