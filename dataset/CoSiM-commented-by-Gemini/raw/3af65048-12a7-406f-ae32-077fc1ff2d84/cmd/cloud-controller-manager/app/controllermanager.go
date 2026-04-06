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

// The cloud-controller-manager is a daemon that embeds the cloud specific
// control loops shipped with Kubernetes.
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

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

const (
	// ControllerStartJitter is the Jitter used when starting controller managers.
	// This helps prevent all controllers from trying to start at the same time.
	ControllerStartJitter = 1.0
)

// NewCloudControllerManagerCommand creates a *cobra.Command object with default parameters.
// This is the main entrypoint for the cloud-controller-manager command.
func NewCloudControllerManagerCommand() *cobra.Command {
	s := options.NewCloudControllerManagerServer()
	s.AddFlags(pflag.CommandLine)
	cmd := &cobra.Command{
		Use: "cloud-controller-manager",
		Long: `The Cloud controller manager is a daemon that embeds
the cloud specific control loops shipped with Kubernetes.`,
		Run: func(cmd *cobra.Command, args []string) {
			// Work is done in the Run function.
		},
	}

	return cmd
}

// resyncPeriod computes the time interval a shared informer waits before resyncing with the api server.
// It adds jitter to the MinResyncPeriod to avoid all informers syncing at the same time.
func resyncPeriod(s *options.CloudControllerManagerServer) func() time.Duration {
	return func() time.Duration {
		factor := rand.Float64() + 1
		return time.Duration(float64(s.MinResyncPeriod.Nanoseconds()) * factor)
	}
}

// Run runs the CloudControllerManagerServer. This should never exit.
func Run(s *options.CloudControllerManagerServer, cloud cloudprovider.Interface) error {
	// Register the controller manager's configuration for viewing via the /configz endpoint.
	if c, err := configz.New("componentconfig"); err == nil {
		c.Set(s.KubeControllerManagerConfiguration)
	} else {
		glog.Errorf("unable to register configz: %s", err)
	}

	// Build the Kubernetes client configuration from flags.
	kubeconfig, err := clientcmd.BuildConfigFromFlags(s.Master, s.Kubeconfig)
	if err != nil {
		return err
	}

	// Configure the client to send requests with a specific content type and QPS settings.
	kubeconfig.ContentConfig.ContentType = s.ContentType
	kubeconfig.QPS = s.KubeAPIQPS
	kubeconfig.Burst = int(s.KubeAPIBurst)

	// Create a Kubernetes clientset from the configuration.
	kubeClient, err := clientset.NewForConfig(restclient.AddUserAgent(kubeconfig, "cloud-controller-manager"))
	if err != nil {
		glog.Fatalf("Invalid API configuration: %v", err)
	}
	// Create a separate client for leader election purposes.
	leaderElectionClient := kubernetes.NewForConfigOrDie(restclient.AddUserAgent(kubeconfig, "leader-election"))

	// Start a separate goroutine to serve HTTP requests for healthz, metrics, and profiling.
	go func() {
		mux := http.NewServeMux()
		healthz.InstallHandler(mux)
		// If profiling is enabled, add the pprof endpoints.
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
	}()

	// Create an event broadcaster and a recorder to record events to the API server.
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(kubeClient.CoreV1().RESTClient()).Events("")})
	recorder := eventBroadcaster.NewRecorder(api.Scheme, v1.EventSource{Component: "cloud-controller-manager"})

	// `run` is a closure that contains the core logic for starting controllers.
	// It's passed to the leader election mechanism.
	run := func(stop <-chan struct{}) {
		// Build client objects for controllers to use.
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

		// Start the actual controller loops.
		err := StartControllers(s, kubeconfig, rootClientBuilder, clientBuilder, stop, recorder, cloud)
		glog.Fatalf("error running controllers: %v", err)
		panic("unreachable")
	}

	// If leader election is disabled, run the controllers directly.
	if !s.LeaderElection.LeaderElect {
		run(nil)
		panic("unreachable")
	}

	// Identity is used to distinguish between multiple cloud controller manager instances.
	id, err := os.Hostname()
	if err != nil {
		return err
	}

	// Create a resource lock for leader election.
	// In a high-availability setup, this lock ensures that only one instance
	// of the cloud-controller-manager is active at a time.
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

	// Try to become the leader and run the controller loops.
	// `RunOrDie` will block until the leader election is won, then call the `OnStartedLeading` callback.
	// If leadership is lost, the `OnStoppedLeading` callback is invoked, which terminates the process.
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

// StartControllers starts the cloud specific controller loops.
func StartControllers(s *options.CloudControllerManagerServer, kubeconfig *restclient.Config, rootClientBuilder, clientBuilder controller.ControllerClientBuilder, stop <-chan struct{}, recorder record.EventRecorder, cloud cloudprovider.Interface) error {
	// client is a helper function to create a clientset for a given service account.
	client := func(serviceAccountName string) clientset.Interface {
		return rootClientBuilder.ClientOrDie(serviceAccountName)
	}

	// Initialize the cloud provider if it exists.
	if cloud != nil {
		// This provides the cloud provider with a way to create its own Kubernetes clients.
		cloud.Initialize(clientBuilder)
	}

	// Create a shared informer factory for all controllers.
	// Informers provide a cached, thread-safe view of resources from the API server.
	versionedClient := client("shared-informers")
	sharedInformers := informers.NewSharedInformerFactory(versionedClient, resyncPeriod(s)())

	// Start the CloudNodeController. This controller is responsible for updating node
	// status based on the cloud provider and deleting nodes from Kubernetes that have
	// been deleted from the cloud provider.
	nodeController := nodecontroller.NewCloudNodeController(
		sharedInformers.Core().V1().Nodes(),
		client("cloud-node-controller"), cloud,
		s.NodeMonitorPeriod.Duration,
		s.NodeStatusUpdateFrequency.Duration)

	nodeController.Run()
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	// Start the service controller. This controller is responsible for creating, updating,
	// and deleting cloud load balancers that correspond to Kubernetes Services of
	// type=LoadBalancer.
	serviceController, err := servicecontroller.New(
		cloud,
		client("service-controller"),
		sharedInformers.Core().V1().Services(),
		sharedInformers.Core().V1().Nodes(),
s.ClusterName,
	)
	if err != nil {
		// Log the error but don't crash. The other controllers might still be functional.
		glog.Errorf("Failed to start service controller: %v", err)
	} else {
		go serviceController.Run(stop, int(s.ConcurrentServiceSyncs))
	}
	time.Sleep(wait.Jitter(s.ControllerStartInterval.Duration, ControllerStartJitter))

	// Pre-condition: Check if the route controller should be started.
	// This is necessary for some cloud providers to configure network routes
	// for pod-to-pod communication across nodes.
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

	// Before starting informers, wait for the API server to be reachable.
	// This is crucial in environments where the controller-manager and apiserver start simultaneously.
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

	// Start all the shared informers. This will begin the process of watching and
	// caching resources from the API server.
	sharedInformers.Start(stop)

	// Block forever, keeping the controllers running until the stop channel is closed.
	select {}
}
