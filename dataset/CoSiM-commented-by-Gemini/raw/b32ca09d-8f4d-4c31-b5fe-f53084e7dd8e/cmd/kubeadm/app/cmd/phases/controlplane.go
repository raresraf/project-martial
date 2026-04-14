/**
 * @file cmd/kubeadm/app/cmd/phases/controlplane.go
 * @brief Defines the Cobra command-line interface for the 'controlplane' phase of kubeadm.
 *        This phase is responsible for generating the static pod manifests required to set up
 *        the Kubernetes control plane components (API Server, Controller Manager, Scheduler).
 *
 * Algorithm: This file uses a data-driven approach to construct a command-line interface.
 *            A slice of structs (`subCmdProperties`) defines the properties for each subcommand,
 *            which are then created in a loop. This avoids repetitive code and makes adding new
 *            subcommands straightforward. The command execution is delegated to functions
 *            from the `controlplanephase` package.
 *
 * Time Complexity: O(N), where N is the number of control plane subcommands defined.
 *                  The setup time is linear with respect to the number of components.
 * Space Complexity: O(N), for storing the command objects and their properties in memory.
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
	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	controlplanephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	"k8s.io/kubernetes/pkg/api"
)

// NewCmdControlplane returns the main command for the 'controlplane' phase.
// This command acts as a container for subcommands that generate manifests for
// individual control plane components.
func NewCmdControlplane() *cobra.Command {
	// Initializes the base 'controlplane' command.
	cmd := &cobra.Command{
		Use:   "controlplane",
		Short: "Generate all static pod manifest files necessary to establish the control plane.",
		RunE:  subCmdRunE("controlplane"),
	}

	// Determine the output directory for the manifest files.
	manifestPath := kubeadmconstants.GetStaticPodDirectory()
	// Register all the subcommands for the 'controlplane' phase.
	cmd.AddCommand(getControlPlaneSubCommands(manifestPath)...)
	return cmd
}

// getControlPlaneSubCommands creates and returns the subcommands for the 'controlplane' phase.
// It takes an output directory `outDir` where the generated manifests will be stored.
func getControlPlaneSubCommands(outDir string) []*cobra.Command {

	// cfg holds the master configuration that will be populated by command-line flags.
	cfg := &kubeadmapiext.MasterConfiguration{}
	// Apply default values to the configuration to be shown in the help text.
	api.Scheme.Default(cfg)

	var cfgPath string
	var subCmds []*cobra.Command

	// subCmdProperties defines the metadata and implementation for each subcommand.
	// This data-driven approach simplifies the creation of similar subcommands.
	subCmdProperties := []struct {
		use     string
		short   string
		cmdFunc func(outDir string, cfg *kubeadmapi.MasterConfiguration) error
	}{
		{
			use:     "all",
			short:   "Generate all static pod manifest files necessary to establish the control plane.",
			cmdFunc: controlplanephase.CreateInitStaticPodManifestFiles,
		},
		{
			use:     "apiserver",
			short:   "Generate apiserver static pod manifest.",
			cmdFunc: controlplanephase.CreateAPIServerStaticPodManifestFile,
		},
		{
			use:     "controller-manager",
			short:   "Generate controller-manager static pod manifest.",
			cmdFunc: controlplanephase.CreateControllerManagerStaticPodManifestFile,
		},
		{
			use:     "scheduler",
			short:   "Generate scheduler static pod manifest.",
			cmdFunc: controlplanephase.CreateSchedulerStaticPodManifestFile,
		},
	}

	// Pre-condition: Iterate over the defined subcommand properties to construct each command.
	// Invariant: Each iteration creates and configures one subcommand ('all', 'apiserver', etc.).
	for _, properties := range subCmdProperties {
		// Creates a new Cobra command for a specific control plane component.
		cmd := &cobra.Command{
			Use:   properties.use,
			Short: properties.short,
			// The Run function is a wrapper that executes the corresponding command function
			// with the provided configuration.
			Run:   runCmdPhase(properties.cmdFunc, &outDir, &cfgPath, cfg),
		}

		// Add flags common to all subcommands.
		cmd.Flags().StringVar(&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir, `The path where certificates are stored.`)
		cmd.Flags().StringVar(&cfg.KubernetesVersion, "kubernetes-version", cfg.KubernetesVersion, `Choose a specific Kubernetes version for the control plane.`)

		// Block Logic: Conditionally add flags based on the subcommand's purpose.
		// Pre-condition: Only 'all' and 'apiserver' commands need API server-specific settings.
		if properties.use == "all" || properties.use == "apiserver" {
			cmd.Flags().StringVar(&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress, "The IP address or DNS name the API Server is accessible on.")
			cmd.Flags().Int32Var(&cfg.API.BindPort, "apiserver-bind-port", cfg.API.BindPort, "The port the API Server is accessible on.")
			cmd.Flags().StringVar(&cfg.Networking.ServiceSubnet, "service-cidr", cfg.Networking.ServiceSubnet, "The range of IP address used for service VIPs.")
		}

		// Pre-condition: Only 'all' and 'controller-manager' commands require network configuration for pods.
		if properties.use == "all" || properties.use == "controller-manager" {
			cmd.Flags().StringVar(&cfg.Networking.PodSubnet, "pod-network-cidr", cfg.Networking.PodSubnet, "The range of IP addresses used for the pod network.")
		}

		// Add the --config flag, which is common to all subcommands, to specify a kubeadm config file.
		cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")

		// Append the newly created command to the list of subcommands.
		subCmds = append(subCmds, cmd)
	}

	return subCmds
}
