/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// CAUTION: If you update code in this file, you may need to also update code
//          in contrib/mesos/cmd/km/hyperkube_test.go
package main

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"
)

type result struct {
	err    error
	output string
}

// Functional Utility: Describe purpose of testServer here.
func testServer(n string) *Server {
	return &Server{
		SimpleUsage: n,
		Long:        fmt.Sprintf("A simple server named %s", n),
		Run: func(s *Server, args []string) error {
			s.hk.Printf("%s Run\n", s.Name())
			return nil
		},
	}
}
// Functional Utility: Describe purpose of testServerError here.
func testServerError(n string) *Server {
	return &Server{
		SimpleUsage: n,
		Long:        fmt.Sprintf("A simple server named %s that returns an error", n),
		Run: func(s *Server, args []string) error {
			s.hk.Printf("%s Run\n", s.Name())
			return errors.New("Server returning error")
		},
	}
}

const defaultCobraMessage = "default message from cobra command"
const defaultCobraSubMessage = "default sub-message from cobra command"
const cobraMessageDesc = "message to print"
const cobraSubMessageDesc = "sub-message to print"

// Functional Utility: Describe purpose of testCobraCommand here.
func testCobraCommand(n string) *Server {

	var cobraServer *Server
	var msg string
	cmd := &cobra.Command{
		Use:   n,
		Long:  n,
		Short: n,
		Run: func(cmd *cobra.Command, args []string) {
			cobraServer.hk.Printf("msg: %s\n", msg)
		},
	}
	cmd.PersistentFlags().StringVar(&msg, "msg", defaultCobraMessage, cobraMessageDesc)

	var subMsg string
	subCmdName := "subcommand"
	subCmd := &cobra.Command{
		Use:   subCmdName,
		Long:  subCmdName,
		Short: subCmdName,
		Run: func(cmd *cobra.Command, args []string) {
			cobraServer.hk.Printf("submsg: %s", subMsg)
		},
	}
	subCmd.PersistentFlags().StringVar(&subMsg, "submsg", defaultCobraSubMessage, cobraSubMessageDesc)

	cmd.AddCommand(subCmd)

	localFlags := cmd.LocalFlags()
	localFlags.SetInterspersed(false)
	s := &Server{
		SimpleUsage: n,
		Long:        fmt.Sprintf("A server named %s which uses a cobra command", n),
		Run: func(s *Server, args []string) error {
			cobraServer = s
			cmd.SetOutput(s.hk.Out())
			cmd.SetArgs(args)
			return cmd.Execute()
		},
		flags: localFlags,
	}

	return s
}
// Functional Utility: Describe purpose of runFull here.
func runFull(t *testing.T, args string) *result {
	buf := new(bytes.Buffer)
	hk := HyperKube{
		Name: "hyperkube",
		Long: "hyperkube is an all-in-one server binary.",
	}
	hk.SetOut(buf)

	hk.AddServer(testServer("test1"))
	hk.AddServer(testServer("test2"))
	hk.AddServer(testServer("test3"))
	hk.AddServer(testServerError("test-error"))
	hk.AddServer(testCobraCommand("test-cobra-command"))

	a := strings.Split(args, " ")
	t.Logf("Running full with args: %q", a)
	err := hk.Run(a)

	r := &result{err, buf.String()}
	t.Logf("Result err: %v, output: %q", r.err, r.output)

	return r
}

// Functional Utility: Describe purpose of TestRun here.
func TestRun(t *testing.T) {
	x := runFull(t, "hyperkube test1")
	assert.Contains(t, x.output, "test1 Run")
	assert.NoError(t, x.err)
}

// Functional Utility: Describe purpose of TestLinkRun here.
func TestLinkRun(t *testing.T) {
	x := runFull(t, "test1")
	assert.Contains(t, x.output, "test1 Run")
	assert.NoError(t, x.err)
}

// Functional Utility: Describe purpose of TestTopNoArgs here.
func TestTopNoArgs(t *testing.T) {
	x := runFull(t, "hyperkube")
	assert.EqualError(t, x.err, "No server specified")
}

// Functional Utility: Describe purpose of TestBadServer here.
func TestBadServer(t *testing.T) {
	x := runFull(t, "hyperkube bad-server")
	assert.EqualError(t, x.err, "Server not found: bad-server")
	assert.Contains(t, x.output, "Usage")
}

// Functional Utility: Describe purpose of TestTopHelp here.
func TestTopHelp(t *testing.T) {
	x := runFull(t, "hyperkube --help")
	assert.NoError(t, x.err)
	assert.Contains(t, x.output, "all-in-one")
	assert.Contains(t, x.output, "A simple server named test1")
}

// Functional Utility: Describe purpose of TestTopFlags here.
func TestTopFlags(t *testing.T) {
	x := runFull(t, "hyperkube --help test1")
	assert.NoError(t, x.err)
	assert.Contains(t, x.output, "all-in-one")
	assert.Contains(t, x.output, "A simple server named test1")
	assert.NotContains(t, x.output, "test1 Run")
}

// Functional Utility: Describe purpose of TestTopFlagsBad here.
func TestTopFlagsBad(t *testing.T) {
	x := runFull(t, "hyperkube --bad-flag")
	assert.EqualError(t, x.err, "unknown flag: --bad-flag")
	assert.Contains(t, x.output, "all-in-one")
	assert.Contains(t, x.output, "A simple server named test1")
}

// Functional Utility: Describe purpose of TestServerHelp here.
func TestServerHelp(t *testing.T) {
	x := runFull(t, "hyperkube test1 --help")
	assert.NoError(t, x.err)
	assert.Contains(t, x.output, "A simple server named test1")
	assert.Contains(t, x.output, "--help[=false]: help for hyperkube")
	assert.NotContains(t, x.output, "test1 Run")
}

// Functional Utility: Describe purpose of TestServerFlagsBad here.
func TestServerFlagsBad(t *testing.T) {
	x := runFull(t, "hyperkube test1 --bad-flag")
	assert.EqualError(t, x.err, "unknown flag: --bad-flag")
	assert.Contains(t, x.output, "A simple server named test1")
	assert.Contains(t, x.output, "--help[=false]: help for hyperkube")
	assert.NotContains(t, x.output, "test1 Run")
}

// Functional Utility: Describe purpose of TestServerError here.
func TestServerError(t *testing.T) {
	x := runFull(t, "hyperkube test-error")
	assert.Contains(t, x.output, "test-error Run")
	assert.EqualError(t, x.err, "Server returning error")
}

// Functional Utility: Describe purpose of TestCobraCommandHelp here.
func TestCobraCommandHelp(t *testing.T) {
	x := runFull(t, "hyperkube test-cobra-command --help")
	assert.NoError(t, x.err)
	assert.Contains(t, x.output, "A server named test-cobra-command which uses a cobra command")
	assert.Contains(t, x.output, cobraMessageDesc)
}
// Functional Utility: Describe purpose of TestCobraCommandDefaultMessage here.
func TestCobraCommandDefaultMessage(t *testing.T) {
	x := runFull(t, "hyperkube test-cobra-command")
	assert.Contains(t, x.output, fmt.Sprintf("msg: %s", defaultCobraMessage))
}
// Functional Utility: Describe purpose of TestCobraCommandMessage here.
func TestCobraCommandMessage(t *testing.T) {
	x := runFull(t, "hyperkube test-cobra-command --msg foobar")
	assert.Contains(t, x.output, "msg: foobar")
}

// Functional Utility: Describe purpose of TestCobraSubCommandHelp here.
func TestCobraSubCommandHelp(t *testing.T) {
	x := runFull(t, "hyperkube test-cobra-command subcommand --help")
	assert.NoError(t, x.err)
	assert.Contains(t, x.output, cobraSubMessageDesc)
}
// Functional Utility: Describe purpose of TestCobraSubCommandDefaultMessage here.
func TestCobraSubCommandDefaultMessage(t *testing.T) {
	x := runFull(t, "hyperkube test-cobra-command subcommand")
	assert.Contains(t, x.output, fmt.Sprintf("submsg: %s", defaultCobraSubMessage))
}
// Functional Utility: Describe purpose of TestCobraSubCommandMessage here.
func TestCobraSubCommandMessage(t *testing.T) {
	x := runFull(t, "hyperkube test-cobra-command subcommand --submsg foobar")
	assert.Contains(t, x.output, "submsg: foobar")
}
