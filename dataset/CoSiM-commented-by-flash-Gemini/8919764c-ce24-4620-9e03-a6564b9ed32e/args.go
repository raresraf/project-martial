/**
 * @8919764c-ce24-4620-9e03-a6564b9ed32e/args.go
 * @brief Common command-line argument handling and orchestration for Kubernetes go2idl generators.
 * Domain: Build Tooling, Code Generation, Go Metaprogramming.
 * Architecture: Provides a reusable 'GeneratorArgs' structure and 'Execute' workflow for specialized generators (client-gen, informer-gen, etc.).
 * Functional Utility: Handles directory parsing, boilerplate injection, and multi-package generation lifecycles.
 */

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// Package args has common command-line flags for generation programs.
package args

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/cmd/libs/go2idl/generator"
	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/parser"
	"k8s.io/kubernetes/cmd/libs/go2idl/types"

	"github.com/spf13/pflag"
)

/**
 * @brief Factory function for initializing a default set of generator arguments.
 * Logic: Configures default source paths and boilerplate header locations based on the current environment.
 * @return Pointer to a pre-populated GeneratorArgs instance.
 */
func Default() *GeneratorArgs {
	generatorArgs := &GeneratorArgs{
		OutputBase:       DefaultSourceTree(),
		GoHeaderFilePath: filepath.Join(DefaultSourceTree(), "k8s.io/kubernetes/hack/boilerplate/boilerplate.go.txt"),
	}
	generatorArgs.AddFlags(pflag.CommandLine)
	return generatorArgs
}

/**
 * @brief Data container for parameters passed to various code generators.
 */
type GeneratorArgs struct {
	// Which directories to parse.
	InputDirs []string

	// If true, recurse into all children of InputDirs
	Recursive bool

	// Source tree to write results to.
	OutputBase string

	// Package path within the source tree.
	OutputPackagePath string

	// Where to get copyright header text.
	GoHeaderFilePath string

	// If true, only verify, don't write anything.
	VerifyOnly bool

	// Any custom arguments go here
	CustomArgs interface{}
}

/**
 * @brief Binds structure fields to command-line flags using the pflag library.
 */
func (g *GeneratorArgs) AddFlags(fs *pflag.FlagSet) {
	fs.StringSliceVarP(&g.InputDirs, "input-dirs", "i", g.InputDirs, "Comma-separated list of import paths to get input types from.")
	fs.StringVarP(&g.OutputBase, "output-base", "o", g.OutputBase, "Output base; defaults to $GOPATH/src/ or ./ if $GOPATH is not set.")
	fs.StringVarP(&g.OutputPackagePath, "output-package", "p", g.OutputPackagePath, "Base package path.")
	fs.StringVarP(&g.GoHeaderFilePath, "go-header-file", "h", g.GoHeaderFilePath, "File containing boilerplate header text. The string YEAR will be replaced with the current 4-digit year.")
	fs.BoolVar(&g.VerifyOnly, "verify-only", g.VerifyOnly, "If true, only verify existing output, do not write anything.")
	fs.BoolVar(&g.Recursive, "recursive", g.VerifyOnly, "If true, recurse into all children of input directories.")
}

/**
 * @brief Reads and preprocesses the copyright boilerplate file.
 * Logic: Replaces the 'YEAR' placeholder with the current calendar year.
 */
func (g *GeneratorArgs) LoadGoBoilerplate() ([]byte, error) {
	b, err := ioutil.ReadFile(g.GoHeaderFilePath)
	if err != nil {
		return nil, err
	}
	b = bytes.Replace(b, []byte("YEAR"), []byte(strconv.Itoa(time.Now().Year())), -1)
	return b, nil
}

/**
 * @brief Constructs and initializes a parser Builder with the specified input directories.
 * Flow: Iterates through InputDirs and performs either flat or recursive directory additions.
 */
func (g *GeneratorArgs) NewBuilder() (*parser.Builder, error) {
	b := parser.New()
	for _, d := range g.InputDirs {
		if g.Recursive {
			if err := b.AddDirRecursive(d); err != nil {
				return nil, fmt.Errorf("unable to add directory %q: %v", d, err)
			}
		} else {
			if err := b.AddDir(d); err != nil {
				return nil, fmt.Errorf("unable to add directory %q: %v", d, err)
			}
		}
	}
	return b, nil
}

/**
 * @brief Predicate to check if a package is within the scope of the generator's input.
 */
func (g *GeneratorArgs) InputIncludes(p *types.Package) bool {
	for _, dir := range g.InputDirs {
		if strings.HasPrefix(p.Path, dir) {
			return true
		}
	}
	return false
}

/**
 * @brief Heuristic to determine the default output location based on GOPATH.
 * Invariant: Falls back to current directory if GOPATH is undefined.
 */
func DefaultSourceTree() string {
	paths := strings.Split(os.Getenv("GOPATH"), string(filepath.ListSeparator))
	if len(paths) > 0 && len(paths[0]) > 0 {
		return filepath.Join(paths[0], "src")
	}
	return "./"
}

/**
 * @brief Main execution entry point for code generation.
 * Architecture: Orchestrates the Parsing -> Context Creation -> Package Generation workflow.
 * @param nameSystems Strategy for naming generated types.
 * @param pkgs Callback function that defines the specific packages to be generated.
 */
func (g *GeneratorArgs) Execute(nameSystems namer.NameSystems, defaultSystem string, pkgs func(*generator.Context, *GeneratorArgs) generator.Packages) error {
	pflag.Parse()

	b, err := g.NewBuilder()
	if err != nil {
		return fmt.Errorf("Failed making a parser: %v", err)
	}

	c, err := generator.NewContext(b, nameSystems, defaultSystem)
	if err != nil {
		return fmt.Errorf("Failed making a context: %v", err)
	}

	c.Verify = g.VerifyOnly
	packages := pkgs(c, g)
	// Execution: Dispatches to the core go2idl generator engine.
	if err := c.ExecutePackages(g.OutputBase, packages); err != nil {
		return fmt.Errorf("Failed executing generator: %v", err)
	}

	return nil
}
