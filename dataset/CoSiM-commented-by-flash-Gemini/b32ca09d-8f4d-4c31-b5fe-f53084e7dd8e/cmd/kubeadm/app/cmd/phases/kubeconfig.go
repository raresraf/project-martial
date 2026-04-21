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
 * @file kubeconfig.go
 * @brief Command definitions for the 'kubeconfig' phase of the kubeadm bootstrap process.
 * 
 * Functional Intent: Provides the CLI interface for generating authentication 
 * configuration files (kubeconfigs) for core cluster components and administrative 
 * users. It abstracts the complexities of credential generation (certificates/tokens) 
 * and cluster connection details, ensuring each component has the necessary 
 * identity to interact with the Kubernetes API server securely.
 * 
 * Domain: Production Systems, Identity Management, Security Orchestration.
 */

package phases

import (
	"fmt"
	"io"

	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	"k8s.io/kubernetes/pkg/api"
)

/**
 * NewCmdKubeConfig - Factory for the parent 'kubeconfig' command.
 * 
 * Logic: Serves as a grouping command for granular authentication setup tasks.
 */
func NewCmdKubeConfig(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "kubeconfig",
		Short: "Generate all kubeconfig files necessary to establish the control plane and the admin kubeconfig file.",
		RunE:  subCmdRunE("kubeconfig"),
	}

	// Functional Utility: Anchors the output to the standard Kubernetes configuration directory.
	cmd.AddCommand(getKubeConfigSubCommands(out, kubeadmconstants.KubernetesDir)...)
	return cmd
}

/**
 * getKubeConfigSubCommands - Instantiates the sub-commands for identity provisioning.
 * 
 * Algorithm: Table-driven command registration with specialized closure handlers.
 * Logic: Maps component names to their respective kubeconfig generation logic. 
 * Includes a custom handler for the 'user' command to support both token and certificate auth.
 */
func getKubeConfigSubCommands(out io.Writer, outDir string) []*cobra.Command {

	cfg := &kubeadmapiext.MasterConfiguration{}
	// Initialization: Applies baseline configuration defaults for CLI help output.
	api.Scheme.Default(cfg)

	var cfgPath, token, clientName string
	var subCmds []*cobra.Command

	// Block Logic: Provisioning task table.
	// Invariant: Covers system components (Controller Manager, Scheduler) and actors (Admin, User).
	subCmdProperties := []struct {
		use     string
		short   string
		cmdFunc func(outDir string, cfg *kubeadmapi.MasterConfiguration) error
	}{
		{
			use:     "all",
			short:   "Generate all kubeconfig files necessary to establish the control plane and the admin kubeconfig file.",
			cmdFunc: kubeconfigphase.CreateInitKubeConfigFiles,
		},
		{
			use:     "admin",
			short:   "Generate a kubeconfig file for the admin to use and for kubeadm itself.",
			cmdFunc: kubeconfigphase.CreateAdminKubeConfigFile,
		},
		{
			use:     "kubelet",
			short:   "Generate a kubeconfig file for the Kubelet to use. Please note that this should *only* be used for bootstrapping purposes. After your control plane is up, you should request all kubelet credentials from the CSR API.",
			cmdFunc: kubeconfigphase.CreateKubeletKubeConfigFile,
		},
		{
			use:     "controller-manager",
			short:   "Generate a kubeconfig file for the Controller Manager to use.",
			cmdFunc: kubeconfigphase.CreateControllerManagerKubeConfigFile,
		},
		{
			use:     "scheduler",
			short:   "Generate a kubeconfig file for the Scheduler to use.",
			cmdFunc: kubeconfigphase.CreateSchedulerKubeConfigFile,
		},
		{
			use:   "user",
			short: "Outputs a kubeconfig file for an additional user.",
			// Block Logic: Dynamic authentication strategy selection.
			cmdFunc: func(outDir string, cfg *kubeadmapi.MasterConfiguration) error {
				if clientName == "" {
					return fmt.Errorf("missing required argument client-name")
				}

				// Logic: Prefers static token auth if provided, otherwise defaults to client certificate generation.
				if token != "" {
					return kubeconfigphase.WriteKubeConfigWithToken(out, cfg, clientName, token)
				}

				return kubeconfigphase.WriteKubeConfigWithClientCert(out, cfg, clientName)
			},
		},
	}

	/**
	 * Block Logic: CLI construction and parameter binding.
	 */
	for _, properties := range subCmdProperties {
		// Creates the UX Command
		cmd := &cobra.Command{
			Use:   properties.use,
			Short: properties.short,
			// Synchronization: Bridges CLI inputs to backend phase logic.
			Run:   runCmdPhase(properties.cmdFunc, &outDir, &cfgPath, cfg),
		}

		// Flags: Configures cluster connectivity and certificate source paths.
		if properties.use != "user" {
			cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")
		}
		cmd.Flags().StringVar(&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir, "The path where certificates are stored.")
		cmd.Flags().StringVar(&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress, "The IP address or DNS name the API Server is accessible on.")
		cmd.Flags().Int32Var(&cfg.API.BindPort, "apiserver-bind-port", cfg.API.BindPort, "The port the API Server is accessible on.")
		
		// Specialized Flags: Node identification for kubelet credentials.
		if properties.use == "all" || properties.use == "kubelet" {
			cmd.Flags().StringVar(&cfg.NodeName, "node-name", cfg.NodeName, `The node name that the kubelet client cert should use.`)
		}
		
		// User-specific Flags: Identity and token parameters.
		if properties.use == "user" {
			cmd.Flags().StringVar(&token, "token", token, "The token that should be used as the authentication mechanism for this kubeconfig.")
			cmd.Flags().StringVar(&clientName, "client-name", clientName, "The name of the KubeConfig user that will be created. Will also be used as the CN if client certs are created.")
		}

		subCmds = append(subCmds, cmd)
	}

	return subCmds
}
