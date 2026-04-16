/*
Copyright 2014 Google Inc. All rights reserved.

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

// Package config implements the kubectl config command and its subcommands.
// It provides functionality to view, set, and unset various configuration
// parameters within a kubeconfig file, which is used to define clusters,
// users, and contexts for accessing Kubernetes clusters.
package config

import (
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
)

// pathOptions holds the various options for specifying a kubeconfig file path.
// These options are mutually exclusive and determine which kubeconfig file
// is loaded or modified.
type pathOptions struct {
	local         bool
	global        bool
	envvar        bool
	specifiedFile string
}

// NewCmdConfig creates the `config` command and its subcommands.
// This is the primary entry point for managing kubeconfig files via kubectl.
func NewCmdConfig(out io.Writer) *cobra.Command {
	pathOptions := &pathOptions{}

	cmd := &cobra.Command{
		Use:   "config SUBCOMMAND",
		Short: "config modifies .kubeconfig files",
		Long:  `config modifies .kubeconfig files using subcommands like "kubectl config set current-context my-context"`,
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
	}

	// Functional Utility: Persistent flags are common to all subcommands of 'config'.
	// These flags allow users to specify the location of the kubeconfig file to operate on.
	cmd.PersistentFlags().BoolVar(&pathOptions.local, "local", false, "use the .kubeconfig in the current directory")
	cmd.PersistentFlags().BoolVar(&pathOptions.global, "global", false, "use the .kubeconfig from "+os.Getenv("HOME"))
	cmd.PersistentFlags().BoolVar(&pathOptions.envvar, "envvar", false, "use the .kubeconfig from $KUBECONFIG")
	cmd.PersistentFlags().StringVar(&pathOptions.specifiedFile, "kubeconfig", "", "use a particular .kubeconfig file")

	// Functional Utility: Add all subcommands for 'kubectl config'.
	// Each subcommand (`set-cluster`, `set-auth-info`, etc.) performs a specific operation
	// on the kubeconfig file determined by the persistent flags.
	cmd.AddCommand(NewCmdConfigView(out, pathOptions))
	cmd.AddCommand(NewCmdConfigSetCluster(out, pathOptions))
	cmd.AddCommand(NewCmdConfigSetAuthInfo(out, pathOptions))
	cmd.AddCommand(NewCmdConfigSetContext(out, pathOptions))
	cmd.AddCommand(NewCmdConfigSet(out, pathOptions))
	cmd.AddCommand(NewCmdConfigUnset(out, pathOptions))
	cmd.AddCommand(NewCmdConfigUseContext(out, pathOptions))

	return cmd
}

// getStartingConfig determines the kubeconfig file to load based on the provided pathOptions.
// It checks flags in a specific precedence order: specifiedFile, global, envvar, local.
// If multiple exclusive flags are set, it returns an error. If no specific flag is set,
// it defaults to using the KUBECONFIG environment variable, then a local .kubeconfig file.
// It returns the loaded clientcmdapi.Config, the determined filename, and any error encountered.
func (o *pathOptions) getStartingConfig() (*clientcmdapi.Config, string, error) {
	filename := ""
	config := clientcmdapi.NewConfig()

	// Block Logic: If a specific kubeconfig file is provided, use it directly.
	if len(o.specifiedFile) > 0 {
		filename = o.specifiedFile
		config = getConfigFromFileOrDie(filename)
	}

	// Block Logic: If the 'global' flag is set, load from the user's HOME directory.
	// Pre-condition: Cannot be combined with a specified file.
	if o.global {
		if len(filename) > 0 {
			return nil, "", fmt.Errorf("already loading from %v, cannot specify global as well", filename)
		}

		filename = os.Getenv("HOME") + "/.kube/.kubeconfig"
		config = getConfigFromFileOrDie(filename)
	}

	// Block Logic: If the 'envvar' flag is set, load from the KUBECONFIG environment variable.
	// Pre-condition: Cannot be combined with a specified file or global flag.
	if o.envvar {
		if len(filename) > 0 {
			return nil, "", fmt.Errorf("already loading from %v, cannot specify global as well", filename)
		}

		filename = os.Getenv(clientcmd.RecommendedConfigPathEnvVar)
		if len(filename) == 0 {
			return nil, "", fmt.Errorf("environment variable %v does not have a value", clientcmd.RecommendedConfigPathEnvVar)
		}

		config = getConfigFromFileOrDie(filename)
	}

	// Block Logic: If the 'local' flag is set, load from the current directory's .kubeconfig.
	// Pre-condition: Cannot be combined with specified file, global, or envvar flags.
	if o.local {
		if len(filename) > 0 {
			return nil, "", fmt.Errorf("already loading from %v, cannot specify global as well", filename)
		}

		filename = ".kubeconfig"
		config = getConfigFromFileOrDie(filename)

	}

	// Block Logic: If no specific flag was set, use default precedence: KUBECONFIG env var, then local .kubeconfig.
	if len(filename) == 0 {
		if len(os.Getenv(clientcmd.RecommendedConfigPathEnvVar)) > 0 {
			filename = os.Getenv(clientcmd.RecommendedConfigPathEnvVar)
			config = getConfigFromFileOrDie(filename)
		} else {
			filename = ".kubeconfig"
			config = getConfigFromFileOrDie(filename)
		}
	}

	return config, filename, nil
}

// getConfigFromFileOrDie attempts to load a kubeconfig file from the given filename.
// If the file is missing, it returns an empty config (not an error).
// If any other error occurs during loading, it logs a fatal error and exits.
func getConfigFromFileOrDie(filename string) *clientcmdapi.Config {
	var err error
	config, err = clientcmd.LoadFromFile(filename)
	// Block Logic: Handles file loading errors. Only a non-existence error is tolerated; others are fatal.
	if err != nil && !os.IsNotExist(err) {
		// Functional Utility: Logs a fatal error and exits the program.
		glog.FatalDepth(1, err)
	}

	// Block Logic: If no config was loaded (e.g., file didn't exist), return a new empty config.
	if config == nil {
		config = clientcmdapi.NewConfig()
	}

	return config
}

// toBool converts a string property value to a boolean.
// It returns an error if the string cannot be parsed as a boolean.
func toBool(propertyValue string) (bool, error) {
	boolValue := false
	// Block Logic: Only attempts parsing if the propertyValue string is not empty.
	if len(propertyValue) != 0 {
		var err error
		boolValue, err = strconv.ParseBool(propertyValue)
		if err != nil {
			return false, err
		}
	}

	return boolValue, nil
}
