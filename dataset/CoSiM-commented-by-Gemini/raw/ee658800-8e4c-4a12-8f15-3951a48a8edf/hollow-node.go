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

// The hollow-node binary is a proxy to a set of node components (kubelet and
// kube-proxy). It acts as a fake kubelet/kube-proxy for testing purposes.
// It is used for running kubernetes performance tests.
package main

import (
	"fmt"
	"runtime"

	docker "github.com/fsouza/go-dockerclient"

	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	cadvisortest "k8s.io/kubernetes/pkg/kubelet/cadvisor/testing"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubemark"
	proxyconfig "k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/util"
	fakeiptables "k8s.io/kubernetes/pkg/util/iptables/testing"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

// HollowNodeConfig contains the configuration for a hollow node.
type HollowNodeConfig struct {
	// Path to the kubeconfig file.
	KubeconfigPath string
	// Port on which the hollow kubelet should listen.
	KubeletPort int
	// Read-only port on which the hollow kubelet should listen.
	KubeletReadOnlyPort int
	// The component to morph into (e.g., "kubelet" or "proxy").
	Morph string
	// Name of this hollow node.
	NodeName string
	// Port on which the API server is listening.
	ServerPort int
}

const (
	// maxPods is the maximum number of pods that can be scheduled on a hollow node.
	maxPods = 110
)

// knownMorphs is a set of allowed values for the --morph flag.
var knownMorphs = sets.NewString("kubelet", "proxy")

// addFlags adds the command-line flags for the hollow node configuration.
func (c *HollowNodeConfig) addFlags(fs *pflag.FlagSet) {
	fs.StringVar(&c.KubeconfigPath, "kubeconfig", "/kubeconfig/kubeconfig", "Path to kubeconfig file.")
	fs.IntVar(&c.KubeletPort, "kubelet-port", 10250, "Port on which HollowKubelet should be listening.")
	fs.IntVar(&c.KubeletReadOnlyPort, "kubelet-read-only-port", 10255, "Read-only port on which Kubelet is listening.")
	fs.StringVar(&c.NodeName, "name", "fake-node", "Name of this Hollow Node.")
	fs.IntVar(&c.ServerPort, "api-server-port", 443, "Port on which API server is listening.")
	fs.StringVar(&c.Morph, "morph", "", fmt.Sprintf("Specifies into which Hollow component this binary should morph. Allowed values: %v", knownMorphs.List()))
}

// createClientFromFile creates a Kubernetes client from a kubeconfig file.
func createClientFromFile(path string) (*client.Client, error) {
	c, err := clientcmd.LoadFromFile(path)
	if err != nil {
		return nil, fmt.Errorf("error while loading kubeconfig from file %v: %v", path, err)
	}
	config, err := clientcmd.NewDefaultClientConfig(*c, &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("error while creating kubeconfig: %v", err)
	}
	client, err := client.New(config)
	if err != nil {
		return nil, fmt.Errorf("error while creating client: %v", err)
	}
	return client, nil
}

func main() {
	// Set the number of operating system threads to the number of available CPUs.
	runtime.GOMAXPROCS(runtime.NumCPU())

	config := HollowNodeConfig{}
	config.addFlags(pflag.CommandLine)
	util.InitFlags()

	// Validate the --morph flag value.
	if !knownMorphs.Has(config.Morph) {
		glog.Fatalf("Unknown morph: %v. Allowed values: %v", config.Morph, knownMorphs.List())
	}

	// Create a client to communicate with the API server.
	cl, err := createClientFromFile(config.KubeconfigPath)
	clientset := clientset.FromUnversionedClient(cl)
	if err != nil {
		glog.Fatal("Failed to create a Client. Exiting.")
	}

	// Morph into a hollow kubelet if specified.
	if config.Morph == "kubelet" {
		// Create fake/stub implementations for Kubelet dependencies.
		cadvisorInterface := new(cadvisortest.Fake)
		containerManager := cm.NewStubContainerManager()

		fakeDockerClient := dockertools.NewFakeDockerClient()
		fakeDockerClient.VersionInfo = docker.Env{"Version=1.1.3", "ApiVersion=1.18"}
		fakeDockerClient.EnableSleep = true

		// Create and run the hollow kubelet.
		hollowKubelet := kubemark.NewHollowKubelet(
			config.NodeName,
			clientset,
			cadvisorInterface,
			fakeDockerClient,
			config.KubeletPort,
			config.KubeletReadOnlyPort,
			containerManager,
			maxPods,
		)
		hollowKubelet.Run()
	}

	// Morph into a hollow kube-proxy if specified.
	if config.Morph == "proxy" {
		// Set up dependencies for the hollow proxy.
		eventBroadcaster := record.NewBroadcaster()
		recorder := eventBroadcaster.NewRecorder(api.EventSource{Component: "kube-proxy", Host: config.NodeName})

		iptInterface := fakeiptables.NewFake()

		serviceConfig := proxyconfig.NewServiceConfig()
		serviceConfig.RegisterHandler(&kubemark.FakeProxyHandler{})

		endpointsConfig := proxyconfig.NewEndpointsConfig()
		endpointsConfig.RegisterHandler(&kubemark.FakeProxyHandler{})

		// Create and run the hollow proxy.
		hollowProxy := kubemark.NewHollowProxyOrDie(config.NodeName, cl, endpointsConfig, serviceConfig, iptInterface, eventBroadcaster, recorder)
		hollowProxy.Run()
	}
}
