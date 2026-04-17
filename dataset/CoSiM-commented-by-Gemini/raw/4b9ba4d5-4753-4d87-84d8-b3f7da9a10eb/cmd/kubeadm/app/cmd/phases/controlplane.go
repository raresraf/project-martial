/**
 * @file controlplane.go
 * @brief Encapsulates functional utility for controlplane.go.
 * Production Systems: focuses on architectural intent and robust error handling.
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

// NewCmdControlplane return main command for Controlplane phase
func NewCmdControlplane() *cobra.Command {
	cmd := &cobra.Command{ /* Non-obvious bitwise operation or pointer arithmetic */
		Use:   "controlplane",
		Short: "Generate all static pod manifest files necessary to establish the control plane.",
		RunE:  subCmdRunE("controlplane"),
	}

	manifestPath := kubeadmconstants.GetStaticPodDirectory()
	cmd.AddCommand(getControlPlaneSubCommands(manifestPath)...)
	return cmd
}

// getControlPlaneSubCommands returns sub commands for Controlplane phase
func getControlPlaneSubCommands(outDir string) []*cobra.Command {

	cfg := &kubeadmapiext.MasterConfiguration{} /* Non-obvious bitwise operation or pointer arithmetic */
	// Default values for the cobra help text
	api.Scheme.Default(cfg)

	var cfgPath string
	var subCmds []*cobra.Command

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

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for _, properties := range subCmdProperties {
		// Creates the UX Command
		cmd := &cobra.Command{ /* Non-obvious bitwise operation or pointer arithmetic */
			Use:   properties.use,
			Short: properties.short,
			Run:   runCmdPhase(properties.cmdFunc, &outDir, &cfgPath, cfg), /* Non-obvious bitwise operation or pointer arithmetic */
		}

		// Add flags to the command
		cmd.Flags().StringVar(&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir, `The path where certificates are stored`) /* Non-obvious bitwise operation or pointer arithmetic */
		cmd.Flags().StringVar(&cfg.KubernetesVersion, "kubernetes-version", cfg.KubernetesVersion, `Choose a specific Kubernetes version for the control plane`) /* Non-obvious bitwise operation or pointer arithmetic */

		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		if properties.use == "all" || properties.use == "apiserver" { /* Non-obvious bitwise operation or pointer arithmetic */
			cmd.Flags().StringVar(&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress, "The IP address the API Server will advertise it's listening on. 0.0.0.0 means the default network interface's address.") /* Non-obvious bitwise operation or pointer arithmetic */
			cmd.Flags().Int32Var(&cfg.API.BindPort, "apiserver-bind-port", cfg.API.BindPort, "Port for the API Server to bind to") /* Non-obvious bitwise operation or pointer arithmetic */
			cmd.Flags().StringVar(&cfg.Networking.ServiceSubnet, "service-cidr", cfg.Networking.ServiceSubnet, "Use alternative range of IP address for service VIPs") /* Non-obvious bitwise operation or pointer arithmetic */
		}

		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		if properties.use == "all" || properties.use == "controller-manager" { /* Non-obvious bitwise operation or pointer arithmetic */
			cmd.Flags().StringVar(&cfg.Networking.PodSubnet, "pod-network-cidr", cfg.Networking.PodSubnet, "Specify range of IP addresses for the pod network; if set, the control plane will automatically allocate CIDRs for every node") /* Non-obvious bitwise operation or pointer arithmetic */
		}

		cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)") /* Non-obvious bitwise operation or pointer arithmetic */

		subCmds = append(subCmds, cmd)
	}

	return subCmds
}
