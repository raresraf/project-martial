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
 * @brief Command definitions for the kubeadm kubeconfig generation phase.
 * 
 * Functional Intent: Manages the generation of KubeConfig files required for 
 * cluster components (Admin, Kubelet, Controller Manager, Scheduler) to 
 * authenticate with the Kubernetes API server. It provides a modular CLI 
 * for bootstrapping initial credentials, supporting both certificate-based 
 * and token-based authentication strategies for system and additional users.
 * 
 * Domain: Production Systems, Security, Cluster Identity (Kubernetes).
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
 * NewCmdKubeConfig - Main entry point for the 'kubeconfig' phase command group.
 */
func NewCmdKubeConfig(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "kubeconfig",
		Short: "Generate all kubeconfig files necessary to establish the control plane and the admin kubeconfig file.",
		RunE:  subCmdRunE("kubeconfig"),
	}

	// Logic: Hooks up subcommands for individual and batch configuration generation.
	cmd.AddCommand(getKubeConfigSubCommands(out, kubeadmconstants.KubernetesDir)...)
	return cmd
}

/**
 * getKubeConfigSubCommands - Factory for generating component-specific credential subcommands.
 * 
 * Algorithm: Property-driven command construction.
 * Logic: Maps component roles (admin, kubelet, etc.) to their respective 
 * configuration implementation functions, handling specific flag requirements 
 * for node identities and user names.
 */
func getKubeConfigSubCommands(out io.Writer, outDir string) []*cobra.Command {

	cfg := &kubeadmapiext.MasterConfiguration{}
	// Pre-condition: Set default configuration for schema validation and help-text parity.
	api.Scheme.Default(cfg)

	var cfgPath, token, clientName string
	var subCmds []*cobra.Command

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
			cmdFunc: func(outDir string, cfg *kubeadmapi.MasterConfiguration) error {
				if clientName == "" {
					return fmt.Errorf("missing required argument client-name")
				}

				// Logic: Polymorphic credential generation (Token vs Cert).
				if token != "" {
					return kubeconfigphase.WriteKubeConfigWithToken(out, cfg, clientName, token)
				}

				return kubeconfigphase.WriteKubeConfigWithClientCert(out, cfg, clientName)
			},
		},
	}

	for _, properties := range subCmdProperties {
		// Logic: Initialization of the Cobra UX command.
		cmd := &cobra.Command{
			Use:   properties.use,
			Short: properties.short,
			Run:   runCmdPhase(properties.cmdFunc, &outDir, &cfgPath, cfg),
		}

		// Block Logic: Standard flag bindings.
		if properties.use != "user" {
			cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")
		}
		cmd.Flags().StringVar(&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir, "The path where to save and store the certificates")
		cmd.Flags().StringVar(&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress, "The IP address the API Server will advertise it's listening on. 0.0.0.0 means the default network interface's address.")
		cmd.Flags().Int32Var(&cfg.API.BindPort, "apiserver-bind-port", cfg.API.BindPort, "Port for the API Server to bind to")
		
		// Block Logic: Identity flags.
		if properties.use == "all" || properties.use == "kubelet" {
			cmd.Flags().StringVar(&cfg.NodeName, "node-name", cfg.NodeName, `Specify the node name`)
		}
		
		// Block Logic: User-specific authentication flags.
		if properties.use == "user" {
			cmd.Flags().StringVar(&token, "token", token, "The path to the directory where the certificates are.")
			cmd.Flags().StringVar(&clientName, "client-name", clientName, "The name of the client for which the KubeConfig file will be generated.")
		}

		subCmds = append(subCmds, cmd)
	}

	return subCmds
}
