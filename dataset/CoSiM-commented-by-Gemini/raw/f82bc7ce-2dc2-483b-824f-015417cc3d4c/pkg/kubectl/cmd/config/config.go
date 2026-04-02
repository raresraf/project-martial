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

// Package config provides the implementation of the `kubectl config` command.
// This command is the entry point for all subcommands that modify or view kubeconfig files.
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

// pathOptions holds the command-line options for specifying which kubeconfig file to use.
// These options are persistent across all 'config' subcommands.
type pathOptions struct {
	// local, if true, indicates that the .kubeconfig file in the current directory should be used.
	local bool
	// global, if true, indicates that the .kubeconfig file in the user's home directory should be used.
	global bool
	// envvar, if true, indicates that the .kubeconfig file pointed to by the KUBECONFIG environment variable should be used.
	envvar bool
	// specifiedFile is the particular kubeconfig file path provided by the --kubeconfig flag.
	specifiedFile string
}

// NewCmdConfig creates the `kubectl config` command and all its subcommands.
func NewCmdConfig(out io.Writer) *cobra.Command {
	pathOptions := &pathOptions{}

	cmd := &cobra.Command{
		Use:   "config SUBCOMMAND",
		Short: "config modifies .kubeconfig files",
		Long:  `config modifies .kubeconfig files using subcommands like "kubectl config set current-context my-context"`,
		Run: func(cmd *cobra.Command, args []string) {
			// By default, if no subcommand is given, print the help information.
			cmd.Help()
		},
	}

	// Register persistent flags that are common to all subcommands.
	// These flags determine which kubeconfig file is loaded.
	cmd.PersistentFlags().BoolVar(&pathOptions.local, "local", false, "use the .kubeconfig in the current directory")
	cmd.PersistentFlags().BoolVar(&pathOptions.global, "global", false, "use the .kubeconfig from "+os.Getenv("HOME"))
	cmd.PersistentFlags().BoolVar(&pathOptions.envvar, "envvar", false, "use the .kubeconfig from $KUBECONFIG")
	cmd.PersistentFlags().StringVar(&pathOptions.specifiedFile, "kubeconfig", "", "use a particular .kubeconfig file")

	// Add all the subcommands to the parent 'config' command.
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
// pathOptions flags. It enforces that only one path-determining flag can be used
// at a time and defines a default loading order if no flag is specified.
func (o *pathOptions) getStartingConfig() (*clientcmdapi.Config, string, error) {
	filename := ""
	config := clientcmdapi.NewConfig()

	// --kubeconfig has the highest precedence.
	if len(o.specifiedFile) > 0 {
		filename = o.specifiedFile
		config = getConfigFromFileOrDie(filename)
	}

	// --global loads from the user's home directory.
	if o.global {
		if len(filename) > 0 {
			return nil, "", fmt.Errorf("already loading from %v, cannot specify global as well", filename)
		}

		filename = os.Getenv("HOME") + "/.kube/.kubeconfig"
		config = getConfigFromFileOrDie(filename)
	}

	// --envvar loads from the KUBECONFIG environment variable.
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

	// --local loads from the current directory.
	if o.local {
		if len(filename) > 0 {
			return nil, "", fmt.Errorf("already loading from %v, cannot specify global as well", filename)
		}

		filename = ".kubeconfig"
		config = getConfigFromFileOrDie(filename)

	}

	// If no specific flag was set, follow the default loading rules:
	// 1. Try the KUBECONFIG environment variable.
	// 2. Fall back to the local .kubeconfig in the current directory.
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

// getConfigFromFileOrDie attempts to read a kubeconfig file from the given path.
// It handles a non-existent file by returning an empty config object, which allows
// commands to create a new file. For any other file loading error, it logs the
// error and terminates the program.
func getConfigFromFileOrDie(filename string) *clientcmdapi.Config {
	var err error
	config, err := clientcmd.LoadFromFile(filename)
	if err != nil && !os.IsNotExist(err) {
		glog.FatalDepth(1, err)
	}

	// If the file did not exist, or was empty, return a new empty config.
	if config == nil {
		config = clientcmdapi.NewConfig()
	}

	return config
}

// toBool is a utility function to parse a string into a boolean value.
// It is likely used by subcommands that set boolean properties.
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
