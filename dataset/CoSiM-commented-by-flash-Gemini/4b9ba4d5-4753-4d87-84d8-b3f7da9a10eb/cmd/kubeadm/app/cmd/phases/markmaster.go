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

/**
 * @file markmaster.go
 * @brief Command definition for the kubeadm 'mark-master' bootstrap phase.
 * 
 * Functional Intent: Provides the CLI interface to explicitly designate a node 
 * as a master (control-plane) member. It orchestrates the application of 
 * specific Kubernetes labels and taints to the target node, ensuring that 
 * the cluster scheduler correctly identifies master nodes and prevents 
 * user workloads from being scheduled on them by default.
 * 
 * Domain: Production Systems, Cluster Orchestration (Kubernetes), Node Lifecycle.
 */

package phases

import (
	"fmt"

	"github.com/spf13/cobra"

	markmasterphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/markmaster"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

/**
 * NewCmdMarkMaster - Factory for the 'kubeadm alpha phase mark-master' command.
 * 
 * Logic: Validates input arguments, establishes an authenticated API client 
 * from the provided KubeConfig, and delegates the node-mutation logic to the 
 * markmasterphase implementation.
 */
func NewCmdMarkMaster() *cobra.Command {
	var kubeConfigFile string
	cmd := &cobra.Command{
		Use:     "mark-master <node-name>",
		Short:   "Create KubeConfig files from given credentials.",
		Aliases: []string{"markmaster"},
		RunE: func(_ *cobra.Command, args []string) error {
			// Pre-condition: Exactly one argument (the node name) must be provided.
			err := validateExactArgNumber(args, []string{"node-name"})
			kubeadmutil.CheckErr(err)

			// Logic: Bootstraps an API client to communicate with the target cluster.
			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			kubeadmutil.CheckErr(err)

			nodeName := args[0]
			fmt.Printf("[markmaster] Will mark node %s as master by adding a label and a taint\n", nodeName)

			// Functional Utility: Applies the master designation labels and taints.
			return markmasterphase.MarkMaster(client, nodeName)
		},
	}

	// Logic: Standard administrative KubeConfig path used as default.
	cmd.Flags().StringVar(&kubeConfigFile, "kubeconfig", "/etc/kubernetes/admin.conf", "The KubeConfig file to use for talking to the cluster")
	return cmd
}
