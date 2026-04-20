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

/**
 * @606dbf3d-7ca2-4c7c-8b41-ddcca133ce9a/staging/src/k8s.io/cloud-provider/options/options.go
 * @brief Configuration schema and lifecycle management for the Cloud Controller Manager (CCM).
 * 
 * Functional Intent: Defines the complete set of parameters required to initialize 
 * and run the CCM. It aggregates generic controller settings with cloud-specific 
 * shared options, security (authentication/authorization) configurations, and 
 * networking parameters. It provides the 'ApplyTo' logic to hydrate a runtime 
 * configuration object from CLI flags and environment variables.
 * 
 * Domain: Kubernetes Infrastructure, Configuration Management, Cloud Providers.
 */

package options

import (
	"fmt"
	"math/rand"
	"net"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/record"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/cloud-provider/app/config"
	ccmconfig "k8s.io/cloud-provider/config"
	ccmconfigscheme "k8s.io/cloud-provider/config/install"
	ccmconfigv1alpha1 "k8s.io/cloud-provider/config/v1alpha1"
	cliflag "k8s.io/component-base/cli/flag"
	cmoptions "k8s.io/controller-manager/options"
	"k8s.io/controller-manager/pkg/clientbuilder"
	netutils "k8s.io/utils/net"

	// add the related feature gates
	_ "k8s.io/controller-manager/pkg/features/register"
)

const (
	// CloudControllerManagerUserAgent is the userAgent name when starting cloud-controller managers.
	CloudControllerManagerUserAgent = "cloud-controller-manager"
)

/**
 * CloudControllerManagerOptions - Root context for CCM daemon configuration.
 * @Generic: Standard controller-manager parameters (e.g., leader election, resync).
 * @SecureServing: Configuration for the administrative HTTPS server.
 * @NodeStatusUpdateFrequency: Temporal cadence for heartbeating node state to the cloud.
 */
type CloudControllerManagerOptions struct {
	Generic           *cmoptions.GenericControllerManagerConfigurationOptions
	KubeCloudShared   *KubeCloudSharedOptions
	ServiceController *ServiceControllerOptions

	SecureServing  *apiserveroptions.SecureServingOptionsWithLoopback
	Authentication *apiserveroptions.DelegatingAuthenticationOptions
	Authorization  *apiserveroptions.DelegatingAuthorizationOptions

	Master     string
	Kubeconfig string

	NodeStatusUpdateFrequency metav1.Duration
}

/**
 * NewCloudControllerManagerOptions - Constructs a default configuration baseline.
 * Logic: Merges internal component defaults with standard Kubernetes system defaults 
 * (e.g., system namespaces, well-known port 10258).
 */
func NewCloudControllerManagerOptions() (*CloudControllerManagerOptions, error) {
	componentConfig, err := NewDefaultComponentConfig()
	if err != nil {
		return nil, err
	}

	s := CloudControllerManagerOptions{
		Generic:         cmoptions.NewGenericControllerManagerConfigurationOptions(&componentConfig.Generic),
		KubeCloudShared: NewKubeCloudSharedOptions(&componentConfig.KubeCloudShared),
		ServiceController: &ServiceControllerOptions{
			ServiceControllerConfiguration: &componentConfig.ServiceController,
		},
		SecureServing:             apiserveroptions.NewSecureServingOptions().WithLoopback(),
		Authentication:            apiserveroptions.NewDelegatingAuthenticationOptions(),
		Authorization:             apiserveroptions.NewDelegatingAuthorizationOptions(),
		NodeStatusUpdateFrequency: componentConfig.NodeStatusUpdateFrequency,
	}

	s.Authentication.RemoteKubeConfigFileOptional = true
	s.Authorization.RemoteKubeConfigFileOptional = true

	s.SecureServing.ServerCert.CertDirectory = ""
	s.SecureServing.ServerCert.PairName = "cloud-controller-manager"
	s.SecureServing.BindPort = cloudprovider.CloudControllerManagerPort

	s.Generic.LeaderElection.ResourceName = "cloud-controller-manager"
	s.Generic.LeaderElection.ResourceNamespace = "kube-system"

	return &s, nil
}

/**
 * NewDefaultComponentConfig - Bootstraps the internal versioned configuration object.
 */
func NewDefaultComponentConfig() (*ccmconfig.CloudControllerManagerConfiguration, error) {
	versioned := &ccmconfigv1alpha1.CloudControllerManagerConfiguration{}
	ccmconfigscheme.Scheme.Default(versioned)

	internal := &ccmconfig.CloudControllerManagerConfiguration{}
	if err := ccmconfigscheme.Scheme.Convert(versioned, internal, nil); err != nil {
		return nil, err
	}
	return internal, nil
}

/**
 * Flags - Defines and binds CLI flags to the options structure.
 * Logic: Sections the flags into groups (Generic, Security, Misc) for a clean help output.
 */
func (o *CloudControllerManagerOptions) Flags(allControllers, disabledByDefaultControllers []string) cliflag.NamedFlagSets {
	fss := cliflag.NamedFlagSets{}
	o.Generic.AddFlags(&fss, allControllers, disabledByDefaultControllers)
	o.KubeCloudShared.AddFlags(fss.FlagSet("generic"))
	o.ServiceController.AddFlags(fss.FlagSet("service controller"))

	o.SecureServing.AddFlags(fss.FlagSet("secure serving"))
	o.Authentication.AddFlags(fss.FlagSet("authentication"))
	o.Authorization.AddFlags(fss.FlagSet("authorization"))

	fs := fss.FlagSet("misc")
	fs.StringVar(&o.Master, "master", o.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig).")
	fs.StringVar(&o.Kubeconfig, "kubeconfig", o.Kubeconfig, "Path to kubeconfig file with authorization and master location information.")
	fs.DurationVar(&o.NodeStatusUpdateFrequency.Duration, "node-status-update-frequency", o.NodeStatusUpdateFrequency.Duration, "Specifies how often the controller updates nodes' status.")

	utilfeature.DefaultMutableFeatureGate.AddFlag(fss.FlagSet("generic"))

	return fss
}

/**
 * ApplyTo - Materializes the runtime Config from validated options.
 * 
 * Block Logic: Infrastructure bootstrapping.
 * Logic: 
 * 1. Early-evaluates kubeconfig to prevent leaking goroutines on initialization failure.
 * 2. Populates internal component configurations.
 * 3. Initializes the high-throughput Kubernetes client and event broadcaster.
 * 4. Configures the shared informer factory with randomized jitter to spread API load.
 */
func (o *CloudControllerManagerOptions) ApplyTo(c *config.Config, userAgent string) error {
	var err error

	c.Kubeconfig, err = clientcmd.BuildConfigFromFlags(o.Master, o.Kubeconfig)
	if err != nil {
		return err
	}
	c.Kubeconfig.DisableCompression = true
	c.Kubeconfig.ContentConfig.AcceptContentTypes = o.Generic.ClientConnection.AcceptContentTypes
	c.Kubeconfig.ContentConfig.ContentType = o.Generic.ClientConnection.ContentType
	c.Kubeconfig.QPS = o.Generic.ClientConnection.QPS
	c.Kubeconfig.Burst = int(o.Generic.ClientConnection.Burst)

	if err = o.Generic.ApplyTo(&c.ComponentConfig.Generic); err != nil {
		return err
	}
	if err = o.KubeCloudShared.ApplyTo(&c.ComponentConfig.KubeCloudShared); err != nil {
		return err
	}
	if err = o.ServiceController.ApplyTo(&c.ComponentConfig.ServiceController); err != nil {
		return err
	}
	if err = o.SecureServing.ApplyTo(&c.SecureServing, &c.LoopbackClientConfig); err != nil {
		return err
	}
	if o.SecureServing.BindPort != 0 || o.SecureServing.Listener != nil {
		if err = o.Authentication.ApplyTo(&c.Authentication, c.SecureServing, nil); err != nil {
			return err
		}
		if err = o.Authorization.ApplyTo(&c.Authorization); err != nil {
			return err
		}
	}

	c.Client, err = clientset.NewForConfig(restclient.AddUserAgent(c.Kubeconfig, userAgent))
	if err != nil {
		return err
	}

	c.EventBroadcaster = record.NewBroadcaster()
	c.EventRecorder = c.EventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: userAgent})

	rootClientBuilder := clientbuilder.SimpleControllerClientBuilder{
		ClientConfig: c.Kubeconfig,
	}
	if c.ComponentConfig.KubeCloudShared.UseServiceAccountCredentials {
		c.ClientBuilder = clientbuilder.NewDynamicClientBuilder(
			restclient.AnonymousClientConfig(c.Kubeconfig),
			c.Client.CoreV1(),
			metav1.NamespaceSystem)
	} else {
		c.ClientBuilder = rootClientBuilder
	}
	c.VersionedClient = rootClientBuilder.ClientOrDie("shared-informers")
	c.SharedInformers = informers.NewSharedInformerFactory(c.VersionedClient, resyncPeriod(c)())

	c.ComponentConfig.NodeStatusUpdateFrequency = o.NodeStatusUpdateFrequency

	return nil
}

/**
 * @brief Validates the logical consistency of aggregated options.
 */
func (o *CloudControllerManagerOptions) Validate(allControllers, disabledByDefaultControllers []string) error {
	errors := []error{}

	errors = append(errors, o.Generic.Validate(allControllers, disabledByDefaultControllers)...)
	errors = append(errors, o.KubeCloudShared.Validate()...)
	errors = append(errors, o.ServiceController.Validate()...)
	errors = append(errors, o.SecureServing.Validate()...)
	errors = append(errors, o.Authentication.Validate()...)
	errors = append(errors, o.Authorization.Validate()...)

	if len(o.KubeCloudShared.CloudProvider.Name) == 0 {
		errors = append(errors, fmt.Errorf("--cloud-provider cannot be empty"))
	}

	return utilerrors.NewAggregate(errors)
}

/**
 * @brief Logic: Injects randomized jitter into the resync cycle for API server fairness.
 */
func resyncPeriod(c *config.Config) func() time.Duration {
	return func() time.Duration {
		factor := rand.Float64() + 1
		return time.Duration(float64(c.ComponentConfig.Generic.MinResyncPeriod.Nanoseconds()) * factor)
	}
}

/**
 * Config - Finalizes the configuration and prepares self-signed certificates for secure serving.
 * Logic: Performs validation, generates temporary in-memory certs if needed, 
 * and materializes the finalized runtime configuration.
 */
func (o *CloudControllerManagerOptions) Config(allControllers, disabledByDefaultControllers []string) (*config.Config, error) {
	if err := o.Validate(allControllers, disabledByDefaultControllers); err != nil {
		return nil, err
	}

	if err := o.SecureServing.MaybeDefaultWithSelfSignedCerts("localhost", nil, []net.IP{netutils.ParseIPSloppy("127.0.0.1")}); err != nil {
		return nil, fmt.Errorf("error creating self-signed certificates: %v", err)
	}

	c := &config.Config{}
	if err := o.ApplyTo(c, CloudControllerManagerUserAgent); err != nil {
		return nil, err
	}

	return c, nil
}
