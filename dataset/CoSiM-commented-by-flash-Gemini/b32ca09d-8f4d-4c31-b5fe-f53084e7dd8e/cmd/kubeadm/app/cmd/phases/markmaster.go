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
 * @brief Command definition for the 'mark-master' phase of kubeadm initialization.
 * 
 * Functional Intent: Provides the CLI entry point for designating a specific node 
 * as a control plane master. This process involves mutating node metadata 
 * (adding the master label) and applying taints to prevent standard workloads 
 * from being scheduled on the control plane unless explicitly tolerated.
 * 
 * Domain: Production Systems, Cluster Orchestration, Metadata Management.
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
 * NewCmdMarkMaster - Factory for the 'mark-master' command.
 * 
 * Logic: Validates input arguments, establishes a secure connection to the API 
 * server via the provided kubeconfig, and delegates the node labeling/tainting 
 * logic to the 'markmaster' phase package.
 */
func NewCmdMarkMaster() *cobra.Command {
	var kubeConfigFile string
	cmd := &cobra.Command{
		Use:     "mark-master <node-name>",
		Short:   "Mark a node as master.",
		Aliases: []string{"markmaster"},
		// Synchronization: Executes the phase logic within the CLI run context.
		RunE: func(_ *cobra.Command, args []string) error {
			// Pre-condition: Exactly one node name must be provided as an argument.
			err := validateExactArgNumber(args, []string{"node-name"})
			kubeadmutil.CheckErr(err)

			// Initialization: Loads client credentials for API interaction.
			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			kubeadmutil.CheckErr(err)

			nodeName := args[0]
			fmt.Printf("[markmaster] Will mark node %s as master by adding a label and a taint\n", nodeName)

			// Functional Utility: Performs the remote API call to update Node object state.
			return markmasterphase.MarkMaster(client, nodeName)
		},
	}

	// Flags: Specifies the location of the administrative configuration required for cluster access.
	cmd.Flags().StringVar(&kubeConfigFile, "kubeconfig", "/etc/kubernetes/admin.conf", "The KubeConfig file to use for talking to the cluster")
	return cmd
}
