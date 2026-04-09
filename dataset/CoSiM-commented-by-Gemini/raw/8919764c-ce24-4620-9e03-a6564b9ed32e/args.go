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
// It defines a standard set of arguments that can be used by various code
// generators within the go2idl framework.
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

// Default returns a defaulted GeneratorArgs. You may change the defaults
// before calling AddFlags. This is useful for Kubernetes code generators
// that want to use shared defaults.
func Default() *GeneratorArgs {
	generatorArgs := &GeneratorArgs{
		OutputBase:       DefaultSourceTree(),
		GoHeaderFilePath: filepath.Join(DefaultSourceTree(), "k8s.io/kubernetes/hack/boilerplate/boilerplate.go.txt"),
	}
	generatorArgs.AddFlags(pflag.CommandLine)
	return generatorArgs
}

// GeneratorArgs has arguments that are passed to generators.
type GeneratorArgs struct {
	// InputDirs is a list of package paths to parse.
	InputDirs []string

	// If true, recurse into all children of InputDirs.
	Recursive bool

	// OutputBase is the base directory for all generated output.
	// Defaults to $GOPATH/src or "." if GOPATH is not set.
	OutputBase string

	// OutputPackagePath is the base Go package path for the generated code.
	OutputPackagePath string

	// GoHeaderFilePath is the path to a file containing boilerplate header text.
	// This text will be inserted at the top of all generated files.
	GoHeaderFilePath string

	// If true, the generator will only verify that the existing output is correct.
	// It will not write any files. This is useful for CI checks.
	VerifyOnly bool

	// CustomArgs is a generic field for generator-specific arguments.
	CustomArgs interface{}
}

// AddFlags binds the fields of GeneratorArgs to command-line flags.
func (g *GeneratorArgs) AddFlags(fs *pflag.FlagSet) {
	fs.StringSliceVarP(&g.InputDirs, "input-dirs", "i", g.InputDirs, "Comma-separated list of import paths to get input types from.")
	fs.StringVarP(&g.OutputBase, "output-base", "o", g.OutputBase, "Output base; defaults to $GOPATH/src/ or ./ if $GOPATH is not set.")
	fs.StringVarP(&g.OutputPackagePath, "output-package", "p", g.OutputPackagePath, "Base package path.")
	fs.StringVarP(&g.GoHeaderFilePath, "go-header-file", "h", g.GoHeaderFilePath, "File containing boilerplate header text. The string YEAR will be replaced with the current 4-digit year.")
	fs.BoolVar(&g.VerifyOnly, "verify-only", g.VerifyOnly, "If true, only verify existing output, do not write anything.")
	fs.BoolVar(&g.Recursive, "recursive", g.VerifyOnly, "If true, recurse into all children of input directories.")
}

// LoadGoBoilerplate reads the header file specified by --go-header-file.
// It replaces the string "YEAR" with the current 4-digit year.
func (g *GeneratorArgs) LoadGoBoilerplate() ([]byte, error) {
	b, err := ioutil.ReadFile(g.GoHeaderFilePath)
	if err != nil {
		return nil, err
	}
	b = bytes.Replace(b, []byte("YEAR"), []byte(strconv.Itoa(time.Now().Year())), -1)
	return b, nil
}

// NewBuilder creates a new parser.Builder and populates it with the
// input directories specified in GeneratorArgs.
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

// InputIncludes returns true if the given package is a subpackage of one of
// the InputDirs.
func (g *GeneratorArgs) InputIncludes(p *types.Package) bool {
	for _, dir := range g.InputDirs {
		if strings.HasPrefix(p.Path, dir) {
			return true
		}
	}
	return false
}

// DefaultSourceTree returns the /src directory of the first entry in $GOPATH.
// If $GOPATH is empty, it returns "./". This is used as a default output location.
func DefaultSourceTree() string {
	paths := strings.Split(os.Getenv("GOPATH"), string(filepath.ListSeparator))
	if len(paths) > 0 && len(paths[0]) > 0 {
		return filepath.Join(paths[0], "src")
	}
	return "./"
}

// Execute is a generic driver for go2idl generators.
//
// It performs the following steps:
// 1. Parses command-line flags.
// 2. Creates a parser.Builder and populates it with input directories.
// 3. Creates a generator.Context.
// 4. Calls the `pkgs` function, which is responsible for defining the specific
//    set of packages to generate.
// 5. Executes the generation logic for the defined packages.
//
// To use it, create a GeneratorArgs object (e.g., via `args.Default()`) and
// then call this method, providing the generator-specific logic in the `pkgs`
// function.
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
	if err := c.ExecutePackages(g.OutputBase, packages); err != nil {
		return fmt.Errorf("Failed executing generator: %v", err)
	}

	return nil
}
