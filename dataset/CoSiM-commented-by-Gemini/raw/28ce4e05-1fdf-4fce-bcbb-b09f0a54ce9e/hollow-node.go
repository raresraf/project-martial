// @file hollow-node.go
// @brief A key component of the Kubernetes 'kubemark' scalability testing framework.
// This program simulates the behavior of a Kubernetes node's main components,
// the Kubelet and the kube-proxy, without the overhead of running actual containers
// or managing real network rules. These "hollow" components interact with the
// Kubernetes API server, allowing for large-scale cluster simulations to test
// the performance and scalability of the Kubernetes control plane. The program
// can "morph" into either a hollow Kubelet or a hollow kube-proxy based on a
// command-line argument.
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

package main

import (
	"fmt"
	"runtime"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	clientset "k8s.io/kubernetes/pkg/client/unversioned/adapters/internalclientset"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	cadvisortest "k8s.io/kubernetes/pkg/kubelet/cadvisor/testing"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubemark"
	proxyconfig "k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/util/flag"
	fakeiptables "k8s.io/kubernetes/pkg/util/iptables/testing"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

// HollowNodeConfig holds the configuration parameters for a hollow node instance.
type HollowNodeConfig struct {
	// KubeconfigPath is the path to the kubeconfig file used to communicate with the API server.
	KubeconfigPath string
	// KubeletPort is the port for the hollow Kubelet's main API.
	KubeletPort int
	// KubeletReadOnlyPort is the port for the hollow Kubelet's read-only API.
	KubeletReadOnlyPort int
	// Morph determines which component this instance will simulate ('kubelet' or 'proxy').
	Morph string
	// NodeName is the name used to register this hollow node with the master.
	NodeName string
	// ServerPort is the port of the Kubernetes API server.
	ServerPort int
	// ContentType is the content type of requests sent to the API server.
	ContentType string
}

const (
	// maxPods is the maximum number of pods that can be scheduled on this hollow node.
	maxPods = 110
	// podsPerCore is a setting for pod density, not actively used in this hollow implementation.
	podsPerCore = 0
)

// knownMorphs is a set of allowed values for the --morph flag.
var knownMorphs = sets.NewString("kubelet", "proxy")

// addFlags registers the command-line flags for the HollowNodeConfig.
func (c *HollowNodeConfig) addFlags(fs *pflag.FlagSet) {
	fs.StringVar(&c.KubeconfigPath, "kubeconfig", "/kubeconfig/kubeconfig", "Path to kubeconfig file.")
	fs.IntVar(&c.KubeletPort, "kubelet-port", 10250, "Port on which HollowKubelet should be listening.")
	fs.IntVar(&c.KubeletReadOnlyPort, "kubelet-read-only-port", 10255, "Read-only port on which Kubelet is listening.")
	fs.StringVar(&c.NodeName, "name", "fake-node", "Name of this Hollow Node.")
	fs.IntVar(&c.ServerPort, "api-server-port", 443, "Port on which API server is listening.")
	fs.StringVar(&c.Morph, "morph", "", fmt.Sprintf("Specifies into which Hollow component this binary should morph. Allowed values: %v", knownMorphs.List()))
	fs.StringVar(&c.ContentType, "kube-api-content-type", "application/vnd.kubernetes.protobuf", "ContentType of requests sent to apiserver.")
}

// createClientFromFile creates a Kubernetes API client using the provided kubeconfig file.
// This client is used by the hollow components to communicate with the master.
func (c *HollowNodeConfig) createClientFromFile() (*client.Client, error) {
	clientConfig, err := clientcmd.LoadFromFile(c.KubeconfigPath)
	if err != nil {
		return nil, fmt.Errorf("error while loading kubeconfig from file %v: %v", c.KubeconfigPath, err)
	}
	config, err := clientcmd.NewDefaultClientConfig(*clientConfig, &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("error while creating kubeconfig: %v", err)
	}
	config.ContentType = c.ContentType
	client, err := client.New(config)
	if err != nil {
		return nil, fmt.Errorf("error while creating client: %v", err)
	}
	return client, nil
}

func main() {
	// Optimizes for multi-core processors by setting the number of OS threads to match the number of CPUs.
	runtime.GOMAXPROCS(runtime.NumCPU())

	config := HollowNodeConfig{}
	config.addFlags(pflag.CommandLine)
	flag.InitFlags()

	// Block Logic: Validate the 'morph' parameter to ensure it's a known component type.
	if !knownMorphs.Has(config.Morph) {
		glog.Fatalf("Unknown morph: %v. Allowed values: %v", config.Morph, knownMorphs.List())
	}

	// create a client to communicate with API server.
	cl, err := config.createClientFromFile()
	clientset := clientset.FromUnversionedClient(cl)
	if err != nil {
		glog.Fatal("Failed to create a Client. Exiting.")
	}

	// Block Logic: If morph is 'kubelet', start a hollow Kubelet instance.
	if config.Morph == "kubelet" {
		// Use fake/stub implementations for Kubelet's dependencies to minimize resource usage.
		cadvisorInterface := new(cadvisortest.Fake)
		containerManager := cm.NewStubContainerManager()
		
		// The fake Docker client simulates Docker operations but does not actually run containers.
		fakeDockerClient := dockertools.NewFakeDockerClient()
		fakeDockerClient.EnableSleep = true

		hollowKubelet := kubemark.NewHollowKubelet(
			config.NodeName,
			clientset,
			cadvisorInterface,
			fakeDockerClient,
			config.KubeletPort,
			config.KubeletReadOnlyPort,
			containerManager,
			maxPods,
			podsPerCore,
		)
		hollowKubelet.Run()
	}

	// Block Logic: If morph is 'proxy', start a hollow kube-proxy instance.
	if config.Morph == "proxy" {
		// Create event broadcaster and recorder for reporting events to the API server.
		eventBroadcaster := record.NewBroadcaster()
		recorder := eventBroadcaster.NewRecorder(api.EventSource{Component: "kube-proxy", Host: config.NodeName})
		
		// Use a fake iptables interface to avoid manipulating the host's actual network rules.
		iptInterface := fakeiptables.NewFake()
		
		// Register fake handlers for service and endpoint configurations. These handlers
		// receive updates but do not perform any real network proxying.
		serviceConfig := proxyconfig.NewServiceConfig()
		serviceConfig.RegisterHandler(&kubemark.FakeProxyHandler{})

		endpointsConfig := proxyconfig.NewEndpointsConfig()
		endpointsConfig.RegisterHandler(&kubemark.FakeProxyHandler{})
		
		// Create and run the hollow proxy.
		hollowProxy := kubemark.NewHollowProxyOrDie(config.NodeName, cl, endpointsConfig, serviceConfig, iptInterface, eventBroadcaster, recorder)
		hollowProxy.Run()
	}
}
