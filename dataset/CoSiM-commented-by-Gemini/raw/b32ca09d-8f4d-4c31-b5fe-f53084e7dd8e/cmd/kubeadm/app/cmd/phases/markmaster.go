/**
 * @file cmd/kubeadm/app/cmd/phases/markmaster.go
 * @brief Defines the Cobra command-line interface for the 'mark-master' phase of kubeadm.
 *        This command is used to apply the master label and taint to a node, officially
 *        designating it as a master node in the Kubernetes cluster.
 *
 * Algorithm: The command takes a single argument, the node name. It uses a kubeconfig file
 *            to establish communication with the Kubernetes cluster, then invokes the
 *            `MarkMaster` function from the `markmasterphase` package to apply the necessary
 *            Kubernetes object modifications (labels and taints).
 *
 * Time Complexity: O(1) from the perspective of this CLI tool. The actual complexity
 *                  depends on the Kubernetes API server's performance for a single
 *                  node update operation, which is typically constant time.
 * Space Complexity: O(1), as the memory usage does not scale with any input size.
 */
/*
Copyright 2017 The Kubernetes Authors.

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

package phases

import (
	"fmt"

	"github.com/spf13/cobra"

	markmasterphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/markmaster"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

// NewCmdMarkMaster returns the Cobra command for running the mark-master phase
func NewCmdMarkMaster() *cobra.Command {
	var kubeConfigFile string
	// Initializes the 'mark-master' command.
	cmd := &cobra.Command{
		Use:     "mark-master <node-name>",
		Short:   "Mark a node as master.",
		Aliases: []string{"markmaster"},
		// RunE is the function that executes when the command is called.
		RunE: func(_ *cobra.Command, args []string) error {
			// Pre-condition: Validate that exactly one argument (the node name) is provided.
			err := validateExactArgNumber(args, []string{"node-name"})
			kubeadmutil.CheckErr(err)

			// Create a Kubernetes clientset from the specified kubeconfig file.
			// This client is used to interact with the Kubernetes API server.
			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			kubeadmutil.CheckErr(err)

			nodeName := args[0]
			fmt.Printf("[markmaster] Will mark node %s as master by adding a label and a taint
", nodeName)

			// Functional Utility: Call the MarkMaster function to apply the master label and taint to the specified node.
			// This is the core logic of the command.
			return markmasterphase.MarkMaster(client, nodeName)
		},
	}

	// Add the --kubeconfig flag to specify the path to the kubeconfig file.
	cmd.Flags().StringVar(&kubeConfigFile, "kubeconfig", "/etc/kubernetes/admin.conf", "The KubeConfig file to use for talking to the cluster")
	return cmd
}
