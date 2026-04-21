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
 * @package options
 * @brief Manages configuration and CLI parameters for the Kubernetes Cloud Controller Manager (CCM).
 * 
 * Architectural Intent: Provides a unified context for initializing the Cloud Controller Manager, 
 * which is responsible for cloud-specific control loops (e.g., node, route, and service controllers).
 * It abstracts the complexity of cloud provider integration behind a standard K8s options pattern.
 * 
 * Production System Patterns:
 * - Decouples cloud-specific logic via the `cloudprovider` interface.
 * - Implements secure serving with auto-generated self-signed certificates for local development.
 * - Uses shared informers with randomized resync periods to prevent "thundering herd" API load.
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
 * @struct CloudControllerManagerOptions
 * @brief The primary context object for CCM configuration.
 * 
 * Functional Utility: Aggregates generic manager options, cloud-specific settings, 
 * and infrastructure plumbing (auth, network, logs).
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

	// NodeStatusUpdateFrequency is the frequency at which the controller updates nodes' status
	NodeStatusUpdateFrequency metav1.Duration
}

/**
 * @brief NewCloudControllerManagerOptions creates a new controller manager option set with defaults.
 * Logic: Bootstraps the CCM with standard ports and default leader election parameters.
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

	// Set the PairName but leave certificate directory blank to generate in-memory by default
	s.SecureServing.ServerCert.CertDirectory = ""
	s.SecureServing.ServerCert.PairName = "cloud-controller-manager"
	s.SecureServing.BindPort = cloudprovider.CloudControllerManagerPort

	s.Generic.LeaderElection.ResourceName = "cloud-controller-manager"
	s.Generic.LeaderElection.ResourceNamespace = "kube-system"

	return &s, nil
}

/**
 * @brief NewDefaultComponentConfig returns the internal cloud-controller manager configuration object.
 * Algorithm: Uses the K8s scheme system to apply default values to the versioned API struct.
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
 * @brief Flags returns flags organized by section name for CLI presentation.
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
 * @brief ApplyTo populates the CCM runtime config with the validated options.
 * Pre-condition: Options must be parsed and verified.
 * Logic: Builds the final client sets, event recorders, and shared informers needed for execution.
 */
func (o *CloudControllerManagerOptions) ApplyTo(c *config.Config, userAgent string) error {
	var err error
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

	c.Kubeconfig, err = clientcmd.BuildConfigFromFlags(o.Master, o.Kubeconfig)
	if err != nil {
		return err
	}
	c.Kubeconfig.DisableCompression = true
	c.Kubeconfig.ContentConfig.AcceptContentTypes = o.Generic.ClientConnection.AcceptContentTypes
	c.Kubeconfig.ContentConfig.ContentType = o.Generic.ClientConnection.ContentType
	c.Kubeconfig.QPS = o.Generic.ClientConnection.QPS
	c.Kubeconfig.Burst = int(o.Generic.ClientConnection.Burst)

	c.Client, err = clientset.NewForConfig(restclient.AddUserAgent(c.Kubeconfig, userAgent))
	if err != nil {
		return err
	}

	c.EventBroadcaster = record.NewBroadcaster()
	c.EventRecorder = c.EventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: userAgent})

	rootClientBuilder := clientbuilder.SimpleControllerClientBuilder{
		ClientConfig: c.Kubeconfig,
	}
	
	/**
	 * Block Logic: Client initialization strategy.
	 * Invariant: Uses service account credentials if configured, otherwise falls back to the root client.
	 */
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

	// sync back to component config
	// TODO: find more elegant way than syncing back the values.
	c.ComponentConfig.NodeStatusUpdateFrequency = o.NodeStatusUpdateFrequency

	return nil
}

/**
 * @brief Validate performs integrity checks on the CCM configuration.
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
 * @brief resyncPeriod computes the time interval for informer resyncing.
 * Algorithm: Applies a jitter factor (1.0 - 2.0) to the base resync period to avoid synchronized API peaks.
 */
func resyncPeriod(c *config.Config) func() time.Duration {
	return func() time.Duration {
		factor := rand.Float64() + 1
		return time.Duration(float64(c.ComponentConfig.Generic.MinResyncPeriod.Nanoseconds()) * factor)
	}
}

/**
 * @brief Config returns a fully initialized Cloud Controller Manager configuration.
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
