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

// Package config contains the implementation of the `kubectl config` command.
// This command is the primary entrypoint for all operations that view or
// modify kubeconfig files.
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

// pathOptions holds the command-line options for specifying which kubeconfig
// file to use. These options are persistent across all `config` subcommands.
type pathOptions struct {
	local         bool
	global        bool
	envvar        bool
	specifiedFile string
}

// NewCmdConfig creates the root `kubectl config` command and adds all its subcommands.
// It orchestrates the entire command structure for config management.
func NewCmdConfig(out io.Writer) *cobra.Command {
	// Initialize the shared path options.
	pathOptions := &pathOptions{}

	// Define the top-level 'config' command.
	cmd := &cobra.Command{
		Use:   "config SUBCOMMAND",
		Short: "config modifies .kubeconfig files",
		Long:  `config modifies .kubeconfig files using subcommands like "kubectl config set current-context my-context"`,
		// The base command does nothing but show help text.
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
	}

	// Register persistent flags, which will be available to all subcommands.
	// These flags determine which kubeconfig file is targeted by the command.
	cmd.PersistentFlags().BoolVar(&pathOptions.local, "local", false, "use the .kubeconfig in the current directory")
	cmd.PersistentFlags().BoolVar(&pathOptions.global, "global", false, "use the .kubeconfig from "+os.Getenv("HOME"))
	cmd.PersistentFlags().BoolVar(&pathOptions.envvar, "envvar", false, "use the .kubeconfig from $KUBECONFIG")
	cmd.PersistentFlags().StringVar(&pathOptions.specifiedFile, "kubeconfig", "", "use a particular .kubeconfig file")

	// Attach all the individual subcommands to the parent `config` command.
	cmd.AddCommand(NewCmdConfigView(out, pathOptions))
	cmd.AddCommand(NewCmdConfigSetCluster(out, pathOptions))
	cmd.AddCommand(NewCmdConfigSetAuthInfo(out, pathOptions))
	cmd.AddCommand(NewCmdConfigSetContext(out, pathOptions))
	cmd.AddCommand(NewCmdConfigSet(out, pathOptions))
	cmd.AddCommand(NewCmdConfigUnset(out, pathOptions))
	cmd.AddCommand(NewCmdConfigUseContext(out, pathOptions))

	return cmd
}

// getStartingConfig determines which kubeconfig file to load based on the provided
// pathOptions and returns the loaded configuration object and its path.
func (o *pathOptions) getStartingConfig() (*clientcmdapi.Config, string, error) {
	filename := ""
	config := clientcmdapi.NewConfig()

	// Precedence order:
	// 1. --kubeconfig flag
	if len(o.specifiedFile) > 0 {
		filename = o.specifiedFile
		config = getConfigFromFileOrDie(filename)
	}

	// 2. --global flag
	if o.global {
		if len(filename) > 0 {
			return nil, "", fmt.Errorf("already loading from %v, cannot specify global as well", filename)
		}
		filename = os.Getenv("HOME") + "/.kube/.kubeconfig"
		config = getConfigFromFileOrDie(filename)
	}

	// 3. --envvar flag
	if o.envvar {
		if len(filename) > 0 {
			return nil, "", fmt.Errorf("already loading from %v, cannot specify global as well", filename)
		}
		envVar := os.Getenv(clientcmd.RecommendedConfigPathEnvVar)
		if len(envVar) == 0 {
			return nil, "", fmt.Errorf("environment variable %v does not have a value", clientcmd.RecommendedConfigPathEnvVar)
		}
		filename = envVar
		config = getConfigFromFileOrDie(filename)
	}

	// 4. --local flag
	if o.local {
		if len(filename) > 0 {
			return nil, "", fmt.Errorf("already loading from %v, cannot specify global as well", filename)
		}
		filename = ".kubeconfig"
		config = getConfigFromFileOrDie(filename)
	}

	// 5. Default behavior if no flags are specified.
	// Try the KUBECONFIG environment variable first, then fall back to a local file.
	if len(filename) == 0 {
		if len(os.Getenv(clientcmd.RecommendedConfigPathEnvVar)) > 0 {
			filename = os.Getenv(clientcmd.RecommendedConfigPathEnvVar)
		} else {
			filename = ".kubeconfig"
		}
		config = getConfigFromFileOrDie(filename)
	}

	return config, filename, nil
}

// getConfigFromFileOrDie is a helper that reads a kubeconfig file.
// It terminates the program with a fatal error for any issue other than
// a non-existent file. If the file does not exist, it returns a new empty config.
func getConfigFromFileOrDie(filename string) *clientcmdapi.Config {
	config, err := clientcmd.LoadFromFile(filename)
	// If an error occurred, but it's not a "file not found" error, exit fatally.
	if err != nil && !os.IsNotExist(err) {
		glog.FatalDepth(1, err)
	}

	// If the file was not found or was empty, return a fresh config object.
	if config == nil {
		return clientcmdapi.NewConfig()
	}

	return config
}

// toBool is a utility function to parse a string property value into a boolean.
// It is likely used by the `set` subcommand.
func toBool(propertyValue string) (bool, error) {
	boolValue := false
	if len(propertyValue) != 0 {
		var err error
		boolValue, err = strconv.ParseBool(propertyValue)
		if err != nil {
			return false, err
		}
	}

	return boolValue, nil
}
