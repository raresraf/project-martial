/**
 * @file kubeconfig.go
 * @brief Encapsulates functional utility for kubeconfig.go.
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
	cmd := &cobra.Command{ /* Non-obvious bitwise operation or pointer arithmetic */
		Use:   "kubeconfig",
		Short: "Generate all kubeconfig files necessary to establish the control plane and the admin kubeconfig file.",
		RunE:  subCmdRunE("kubeconfig"),
	}

	cmd.AddCommand(getKubeConfigSubCommands(out, kubeadmconstants.KubernetesDir)...)
	return cmd
}

// getKubeConfigSubCommands returns sub commands for kubeconfig phase
func getKubeConfigSubCommands(out io.Writer, outDir string) []*cobra.Command {

	cfg := &kubeadmapiext.MasterConfiguration{} /* Non-obvious bitwise operation or pointer arithmetic */
	// Default values for the cobra help text
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
				/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
				if clientName == "" {
					return fmt.Errorf("missing required argument client-name")
				}

				// if the kubeconfig file for an additional user has to use a token, use it
				/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
				if token != "" {
					return kubeconfigphase.WriteKubeConfigWithToken(out, cfg, clientName, token)
				}

				// Otherwise, write a kubeconfig file with a generate client cert
				return kubeconfigphase.WriteKubeConfigWithClientCert(out, cfg, clientName)
			},
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
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		if properties.use != "user" {
			cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)") /* Non-obvious bitwise operation or pointer arithmetic */
		}
		cmd.Flags().StringVar(&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir, "The path where to save and store the certificates") /* Non-obvious bitwise operation or pointer arithmetic */
		cmd.Flags().StringVar(&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress, "The IP address the API Server will advertise it's listening on. 0.0.0.0 means the default network interface's address.") /* Non-obvious bitwise operation or pointer arithmetic */
		cmd.Flags().Int32Var(&cfg.API.BindPort, "apiserver-bind-port", cfg.API.BindPort, "Port for the API Server to bind to") /* Non-obvious bitwise operation or pointer arithmetic */
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		if properties.use == "all" || properties.use == "kubelet" { /* Non-obvious bitwise operation or pointer arithmetic */
			cmd.Flags().StringVar(&cfg.NodeName, "node-name", cfg.NodeName, `Specify the node name`) /* Non-obvious bitwise operation or pointer arithmetic */
		}
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		if properties.use == "user" {
			cmd.Flags().StringVar(&token, "token", token, "The path to the directory where the certificates are.") /* Non-obvious bitwise operation or pointer arithmetic */
			cmd.Flags().StringVar(&clientName, "client-name", clientName, "The name of the client for which the KubeConfig file will be generated.") /* Non-obvious bitwise operation or pointer arithmetic */
		}

		subCmds = append(subCmds, cmd)
	}

	return subCmds
}
