/*
Copyright 2016 The Kubernetes Authors.

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

// Package app implements the entrypoint for the cloud-controller-manager.
package app

import (
	"math/rand"
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	goruntime "runtime"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/cmd/cloud-controller-manager/app/options"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	nodecontroller "k8s.io/kubernetes/pkg/controller/cloud"
	routecontroller "k8s.io/kubernetes/pkg/controller/route"
	servicecontroller "k8s.io/kubernetes/pkg/controller/service"
	"k8s.io/kubernetes/pkg/util/configz"
)

const (
	// ControllerStartJitter is used when starting controller managers to avoid
	// contending on the same node at the same time.
	ControllerStartJitter = 1.0
)

// NewCloudControllerManagerCommand creates a *cobra.Command object with default parameters
func NewCloudControllerManagerCommand() *cobra.Command {
	s := options.NewCloudControllerManagerServer()
	s.AddFlags(pflag.CommandLine)
	cmd := &cobra.Command{
		Use: "cloud-controller-manager",
		Long: `The Cloud controller manager is a daemon that embeds
the cloud specific control loops shipped with Kubernetes.`,
		Run: func(cmd *cobra.Command, args []string) {
		},
	}

	return cmd
}

// resyncPeriod computes the time interval a shared informer waits before resyncing with the api server.
func resyncPeriod(s *options.CloudControllerManagerServer) func() time.Duration {
	return func() time.Duration {
		factor := rand.Float64() + 1
		return time.Duration(float64(s.MinResyncPeriod.Nanoseconds()) * factor)
	}
}

// Run runs the cloud-controller-manager server. This should never exit.
func Run(s *options.CloudControllerManagerServer, cloud cloudprovider.Interface) error {
	// Functional Utility: Set up a configz endpoint for displaying component configuration.
	if c, err := configz.New("componentconfig"); err == nil {
		c.Set(s.KubeControllerManagerConfiguration)
	} else {
		glog.Errorf("unable to register configz: %s", err)
	}

	// Functional Utility: Build the kubeconfig from master URL or kubeconfig file.
	kubeconfig, err := clientcmd.BuildConfigFromFlags(s.Master, s.Kubeconfig)
	if err != nil {
		return err
	}
	kubeconfig.ContentConfig.ContentType = s.ContentType
	kubeconfig.QPS = s.KubeAPIQPS
	kubeconfig.Burst = int(s.KubeAPIBurst)

	// Functional Utility: Create Kubernetes clientsets for general use and for leader election.
	kubeClient, err := clientset.NewForConfig(restclient.AddUserAgent(kubeconfig, "cloud-controller-manager"))
	if err != nil {
		glog.Fatalf("Invalid API configuration: %v", err)
	}
	leaderElectionClient := kubernetes.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "leader-election"))

	// Functional Utility: Start a background HTTP server for healthz, metrics, and profiling.
	go startHTTP(s)

	recorder := createRecorder(kubeClient)

	// The main run function for the controller manager logic.
	run := func(stop <-chan struct{}) {
		// Architectural Pattern: The client builder pattern allows controllers to get clients with specific
		// service accounts. The cloud-controller-manager uses a single root client builder.
		rootClientBuilder := controller.SimpleControllerClientBuilder{
			ClientConfig: kubeconfig,
		}
		var clientBuilder controller.ControllerClientBuilder
		if len(s.ServiceAccountKeyFile) > 0 && s.UseServiceAccountCredentials {
			clientBuilder = controller.SAControllerClientBuilder{
				ClientConfig:         restclient.AnonymousClientConfig(kubeconfig),
				CoreClient:           kubeClient.CoreV1(),
				AuthenticationClient: kubeClient.Authentication(),
				Namespace:            "kube-system",
			}
		} else {
			clientBuilder = rootClientBuilder
		}

		// Block Logic: Start the actual controller loops.
		err := StartControllers(s, kubeconfig, rootClientBuilder, clientBuilder, stop, recorder, cloud)
		glog.Fatalf("error running controllers: %v", err)
		panic("unreachable")
	}

	// Architectural Pattern: Leader Election. For high availability, multiple instances of the
	// cloud-controller-manager can be run, but only one (the leader) should be active at a time.
	if !s.LeaderElection.LeaderElect {
		run(nil)
		panic("unreachable")
	}

	// Identity used to distinguish between multiple cloud controller manager instances.
	id, err := os.Hostname()
	if err != nil {
		return err
	}

	// The resource lock is used for leader election.
	rl := resourcelock.EndpointsLock{
		EndpointsMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "cloud-controller-manager",
		},
		Client: leaderElectionClient.CoreV1(),
		LockConfig: resourcelock.ResourceLockConfig{
			Identity:      id + "-external-cloud-controller",
			EventRecorder: recorder,
		},
	}

	// Block Logic: Start the leader election loop.
	// RunOrDie blocks until leadership is acquired, then calls the `run` function.
	// If leadership is lost, the `OnStoppedLeading` callback is invoked, causing a fatal exit.
	leaderelection.RunOrDie(leaderelection.LeaderElectionConfig{
		Lock:          &rl,
		LeaseDuration: s.LeaderElection.LeaseDuration.Duration,
		RenewDeadline: s.LeaderElection.RenewDeadline.Duration,
		RetryPeriod:   s.LeaderElection.RetryPeriod.Duration,
		Callbacks: leaderelection.LeaderCallbacks{
			OnStartedLeading: run,
			OnStoppedLeading: func() {
				glog.Fatalf("leaderelection lost")
			},
		},
	})
	panic("unreachable")
}

// StartControllers starts the cloud-specific controller loops.
func StartControllers(s *options.CloudControllerManagerServer, kubeconfig *restclient.Config, rootClientBuilder, clientBuilder controller.ControllerClientBuilder, stop <-chan struct{}, recorder record.EventRecorder, cloud cloudprovider.Interface) error {
	client := func(serviceAccountName string) clientset.Interface {
		return rootClientBuilder.ClientOrDie(serviceAccountName)
	}

	// Functional Utility: Initialize the cloud provider interface, passing it a client builder.
	if cloud != nil {
		cloud.Initialize(clientBuilder)
	}

	// Architectural Pattern: Shared Informers provide a cached, event-driven mechanism
	// for controllers to react to changes in the Kubernetes API.
	versionedClient := client("shared-informers")
	sharedInformers := informers.NewSharedInformerFactory(versionedClient, resyncPeriod(s)())

	// Start the CloudNodeController. This controller is responsible for tasks like
	// updating node addresses and deleting nodes that have been removed from the cloud provider.
	nodeController := nodecontroller.NewCloudNodeController(
		sharedInformers.Core().V1().Nodes(),
		client("cloud-node-controller"), cloud,
		s.NodeMonitorPeriod.Duration,
		s.NodeStatusUpdateFrequency.Duration)

	nodeController.Run()
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	// Start the ServiceController. This controller is responsible for creating, updating,
	// and deleting cloud load balancers that correspond to Kubernetes Services of type LoadBalancer.
	serviceController, err := servicecontroller.New(
		cloud,
		client("service-controller"),
		sharedInformers.Core().V1().Services(),
		sharedInformers.Core().V1().Nodes(),
s.ClusterName,
	)
	if err != nil {
		glog.Errorf("Failed to start service controller: %v", err)
	} else {
		go serviceController.Run(stop, int(s.ConcurrentServiceSyncs))
	}
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	// Pre-condition: Start the RouteController only if node CIDR allocation and cloud route
	// configuration are both enabled. This controller manages network routes on the cloud provider.
	if s.AllocateNodeCIDRs && s.ConfigureCloudRoutes {
		if routes, ok := cloud.Routes(); !ok {
			glog.Warning("configure-cloud-routes is set, but cloud provider does not support routes. Will not configure cloud provider routes.")
		} else {
			var clusterCIDR *net.IPNet
			if len(strings.TrimSpace(s.ClusterCIDR)) != 0 {
				_, clusterCIDR, err = net.ParseCIDR(s.ClusterCIDR)
				if err != nil {
					glog.Warningf("Unsuccessful parsing of cluster CIDR %v: %v", s.ClusterCIDR, err)
				}
			}

			routeController := routecontroller.New(routes, client("route-controller"), sharedInformers.Core().V1().Nodes(), s.ClusterName, clusterCIDR)
			go routeController.Run(stop, s.RouteReconciliationPeriod.Duration)
			time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))
		}
	} else {
		glog.Infof("Will not configure cloud provider routes for allocate-node-cidrs: %v, configure-cloud-routes: %v.", s.AllocateNodeCIDRs, s.ConfigureCloudRoutes)
	}

	// Block Logic: Wait for the API server to be available before starting informers.
	err = wait.PollImmediate(time.Second, 10*time.Second, func() (bool, error) {
		if _, err = restclient.ServerAPIVersions(kubeconfig); err == nil {
			return true, nil
		}
		glog.Errorf("Failed to get api versions from server: %v", err)
		return false, nil
	})
	if err != nil {
		glog.Fatalf("Failed to get api versions from server: %v", err)
	}

	// Start all the shared informers, which will begin watching the API server for changes.
	sharedInformers.Start(stop)

	select {}
}

// startHTTP starts an HTTP server for healthz, metrics, and pprof profiling endpoints.
func startHTTP(s *options.CloudControllerManagerServer) {
	mux := http.NewServeMux()
	healthz.InstallHandler(mux)
	if s.EnableProfiling {
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
		if s.EnableContentionProfiling {
			goruntime.SetBlockProfileRate(1)
		}
	}
	configz.InstallHandler(mux)
	mux.Handle("/metrics", prometheus.Handler())

	server := &http.Server{
		Addr:    net.JoinHostPort(s.Address, strconv.Itoa(int(s.Port))),
		Handler: mux,
	}
	glog.Fatal(server.ListenAndServe())
}

// createRecorder creates an event recorder for sending events to the API server.
func createRecorder(kubeClient *clientset.Clientset) record.EventRecorder {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(kubeClient.CoreV1().RESTClient()).Events("")})
	return eventBroadcaster.NewRecorder(api.Scheme, v1.EventSource{Component: "cloud-controller-manager"})
}
