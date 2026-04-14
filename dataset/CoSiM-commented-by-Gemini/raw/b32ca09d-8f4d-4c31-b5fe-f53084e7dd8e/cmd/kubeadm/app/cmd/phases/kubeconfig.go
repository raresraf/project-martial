/**
 * @file cmd/kubeadm/app/cmd/phases/kubeconfig.go
 * @brief Defines the Cobra command-line interface for the 'kubeconfig' phase of kubeadm.
 *        This phase is responsible for generating kubeconfig files for various components
 *        (admin, kubelet, controller-manager, scheduler) and additional users.
 *
 * Algorithm: Similar to controlplane.go, this file uses a data-driven approach. A slice
 *            of structs (`subCmdProperties`) defines the properties for each subcommand,
 *            which are then instantiated in a loop. This design pattern centralizes command
 *            definitions, making the CLI structure easy to manage and extend.
 *
 * Time Complexity: O(N), where N is the number of kubeconfig subcommands.
 *                  The setup time is linear with respect to the number of defined kubeconfigs.
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
	"fmt"
	"io"

	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	"k8s.io/kubernetes/pkg/api"
)

// NewCmdKubeConfig return main command for kubeconfig phase
func NewCmdKubeConfig(out io.Writer) *cobra.Command {
	// Initializes the base 'kubeconfig' command.
	cmd := &cobra.Command{
		Use:   "kubeconfig",
		Short: "Generate all kubeconfig files necessary to establish the control plane and the admin kubeconfig file.",
		RunE:  subCmdRunE("kubeconfig"),
	}

	// Register all the subcommands for the 'kubeconfig' phase.
	cmd.AddCommand(getKubeConfigSubCommands(out, kubeadmconstants.KubernetesDir)...)
	return cmd
}

// getKubeConfigSubCommands returns sub commands for kubeconfig phase
func getKubeConfigSubCommands(out io.Writer, outDir string) []*cobra.Command {

	// cfg holds the master configuration that will be populated by command-line flags.
	cfg := &kubeadmapiext.MasterConfiguration{}
	// Default values for the cobra help text
	api.Scheme.Default(cfg)

	var cfgPath, token, clientName string
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
			// cmdFunc is an inline function to handle the specific logic for the 'user' subcommand.
			cmdFunc: func(outDir string, cfg *kubeadmapi.MasterConfiguration) error {
				// Pre-condition: The client-name argument is mandatory for user kubeconfig generation.
				if clientName == "" {
					return fmt.Errorf("missing required argument client-name")
				}

				// Block Logic: If a token is provided, generate a token-based kubeconfig.
				// This is one of two authentication methods supported for user-specific files.
				if token != "" {
					return kubeconfigphase.WriteKubeConfigWithToken(out, cfg, clientName, token)
				}

				// Otherwise, generate a kubeconfig that uses a client certificate for authentication.
				return kubeconfigphase.WriteKubeConfigWithClientCert(out, cfg, clientName)
			},
		},
	}

	// Pre-condition: Iterate over the defined subcommand properties to construct each command.
	// Invariant: Each iteration creates and configures one subcommand.
	for _, properties := range subCmdProperties {
		// Creates the UX Command
		cmd := &cobra.Command{
			Use:   properties.use,
			Short: properties.short,
			// The Run function is a wrapper that executes the corresponding command function
			// with the provided configuration.
			Run:   runCmdPhase(properties.cmdFunc, &outDir, &cfgPath, cfg),
		}

		// Add flags to the command
		// Pre-condition: The 'user' command has a different set of flags and does not use a config file directly.
		if properties.use != "user" {
			cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")
		}
		// These flags are common to most subcommands for specifying API server endpoint and certificate location.
		cmd.Flags().StringVar(&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir, "The path where certificates are stored.")
		cmd.Flags().StringVar(&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress, "The IP address or DNS name the API Server is accessible on.")
		cmd.Flags().Int32Var(&cfg.API.BindPort, "apiserver-bind-port", cfg.API.BindPort, "The port the API Server is accessible on.")
		// Pre-condition: The 'kubelet' subcommand (and 'all') requires a node name for the client certificate.
		if properties.use == "all" || properties.use == "kubelet" {
			cmd.Flags().StringVar(&cfg.NodeName, "node-name", cfg.NodeName, `The node name that the kubelet client cert should use.`)
		}
		// Pre-condition: The 'user' subcommand has its own specific flags for token and client name.
		if properties.use == "user" {
			cmd.Flags().StringVar(&token, "token", token, "The token that should be used as the authentication mechanism for this kubeconfig.")
			cmd.Flags().StringVar(&clientName, "client-name", clientName, "The name of the KubeConfig user that will be created. Will also be used as the CN if client certs are created.")
		}

		subCmds = append(subCmds, cmd)
	}

	return subCmds
}
