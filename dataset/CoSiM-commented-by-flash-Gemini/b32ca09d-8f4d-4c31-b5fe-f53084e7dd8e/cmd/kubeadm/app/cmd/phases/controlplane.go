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
 * @file controlplane.go
 * @brief Command definitions for the 'controlplane' phase of the kubeadm bootstrap process.
 * 
 * Functional Intent: Provides the CLI entry points for generating static pod manifest 
 * files for core Kubernetes components (API Server, Controller Manager, Scheduler). 
 * It manages the UX layer by exposing commands that translate user inputs and 
 * configuration files into concrete filesystem artifacts required to establish 
 * the cluster's control plane.
 * 
 * Domain: Production Systems, Cloud Orchestration, CLI Infrastructure.
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

/**
 * NewCmdControlplane - Factory for the parent 'controlplane' command.
 * 
 * Logic: Serves as a grouping command for granular component manifest generation.
 */
func NewCmdControlplane() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "controlplane",
		Short: "Generate all static pod manifest files necessary to establish the control plane.",
		RunE:  subCmdRunE("controlplane"),
	}

	manifestPath := kubeadmconstants.GetStaticPodDirectory()
	// Functional Utility: Dynamically aggregates sub-commands for individual control plane components.
	cmd.AddCommand(getControlPlaneSubCommands(manifestPath)...)
	return cmd
}

/**
 * getControlPlaneSubCommands - Instantiates the set of sub-commands supported by this phase.
 * 
 * Algorithm: Table-driven command registration.
 * Logic: Maps specific 'use' strings to their corresponding generation logic in the 
 * 'controlplane' package, while decorating each with appropriate CLI flags.
 */
func getControlPlaneSubCommands(outDir string) []*cobra.Command {

	cfg := &kubeadmapiext.MasterConfiguration{}
	// Initialization: Populates the configuration with schema-defined defaults for help text rendering.
	api.Scheme.Default(cfg)

	var cfgPath string
	var subCmds []*cobra.Command

	// Block Logic: Command mapping table.
	// Invariant: Each entry corresponds to a distinct static pod manifest or the complete set.
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

	/**
	 * Block Logic: CLI command construction and flag binding.
	 * Logic: Iteratively builds cobra commands and conditionalizes flag availability 
	 * based on the component type to reduce UI clutter for specialized tasks.
	 */
	for _, properties := range subCmdProperties {
		// Creates the UX Command
		cmd := &cobra.Command{
			Use:   properties.use,
			Short: properties.short,
			// Synchronization: Bridges the CLI execution context to the underlying business logic.
			Run:   runCmdPhase(properties.cmdFunc, &outDir, &cfgPath, cfg),
		}

		// Common Flags: Parameters relevant to all control plane components.
		cmd.Flags().StringVar(&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir, `The path where certificates are stored.`)
		cmd.Flags().StringVar(&cfg.KubernetesVersion, "kubernetes-version", cfg.KubernetesVersion, `Choose a specific Kubernetes version for the control plane.`)

		// Specialized Flags: Networking parameters required primarily by the API Server.
		if properties.use == "all" || properties.use == "apiserver" {
			cmd.Flags().StringVar(&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress, "The IP address or DNS name the API Server is accessible on.")
			cmd.Flags().Int32Var(&cfg.API.BindPort, "apiserver-bind-port", cfg.API.BindPort, "The port the API Server is accessible on.")
			cmd.Flags().StringVar(&cfg.Networking.ServiceSubnet, "service-cidr", cfg.Networking.ServiceSubnet, "The range of IP address used for service VIPs.")
		}

		// Specialized Flags: Pod networking parameters for the Controller Manager.
		if properties.use == "all" || properties.use == "controller-manager" {
			cmd.Flags().StringVar(&cfg.Networking.PodSubnet, "pod-network-cidr", cfg.Networking.PodSubnet, "The range of IP addresses used for the pod network.")
		}

		// Global Configuration Override.
		cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")

		subCmds = append(subCmds, cmd)
	}

	return subCmds
}
