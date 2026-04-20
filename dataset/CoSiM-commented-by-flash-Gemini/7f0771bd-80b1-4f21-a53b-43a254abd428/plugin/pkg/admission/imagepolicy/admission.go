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
 * @7f0771bd-80b1-4f21-a53b-43a254abd428/plugin/pkg/admission/imagepolicy/admission.go
 * @brief Admission controller for external container image policy enforcement.
 * 
 * Functional Intent: Implements an ImagePolicyWebhook that delegates pod admission 
 * decisions to a remote backend. It validates that all container images within 
 * a pod request comply with external security or governance policies. It features 
 * an LRU cache with TTL to minimize webhook latency and configurable "fail-open" 
 * behavior for backend connectivity issues.
 * 
 * Domain: Kubernetes Security, Admission Control, Webhook Integration.
 */

package imagepolicy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"k8s.io/klog"

	"k8s.io/api/imagepolicy/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/cache"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"

	// install the clientgo image policy API for use with api registry
	_ "k8s.io/kubernetes/pkg/apis/imagepolicy/install"
)

// PluginName indicates name of admission plugin.
const PluginName = "ImagePolicyWebhook"

// AuditKeyPrefix identifies annotations handled by this specific linter.
var AuditKeyPrefix = strings.ToLower(PluginName) + ".image-policy.k8s.io/"

const (
	ImagePolicyFailedOpenKeySuffix string = "failed-open"
	ImagePolicyAuditRequiredKeySuffix string = "audit-required"
)

var (
	groupVersions = []schema.GroupVersion{v1alpha1.SchemeGroupVersion}
)

/**
 * Register - Global entry point for plugin activation.
 */
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		newImagePolicyWebhook, err := NewImagePolicyWebhook(config)
		if err != nil {
			return nil, err
		}
		return newImagePolicyWebhook, nil
	})
}

/**
 * Plugin - State container for the ImagePolicy admission handler.
 * @webhook: Generic client for remote policy evaluation.
 * @responseCache: Concurrent-safe storage for approved/denied image sets.
 * @allowTTL/denyTTL: Expiration intervals for positive and negative cache entries.
 */
type Plugin struct {
	*admission.Handler
	webhook       *webhook.GenericWebhook
	responseCache *cache.LRUExpireCache
	allowTTL      time.Duration
	denyTTL       time.Duration
	defaultAllow  bool
}

var _ admission.ValidationInterface = &Plugin{}

/**
 * @brief Logic: Selects the appropriate TTL based on the backend's allowed state.
 */
func (a *Plugin) statusTTL(status v1alpha1.ImageReviewStatus) time.Duration {
	if status.Allowed {
		return a.allowTTL
	}
	return a.denyTTL
}

/**
 * @brief Functional Utility: Sanitizes metadata by retaining only image-policy relevant annotations.
 */
func (a *Plugin) filterAnnotations(allAnnotations map[string]string) map[string]string {
	annotations := make(map[string]string)
	for k, v := range allAnnotations {
		if strings.Contains(k, ".image-policy.k8s.io/") {
			annotations[k] = v
		}
	}
	return annotations
}

/**
 * webhookError - Failure handler for backend communication.
 * Logic: Implements the 'fail-open' policy. If defaultAllow is true, the request 
 * proceeds with an 'audit-required' annotation; otherwise, it is rejected with 403 Forbidden.
 */
func (a *Plugin) webhookError(pod *api.Pod, attributes admission.Attributes, err error) error {
	if err != nil {
		klog.V(2).Infof("error contacting webhook backend: %s", err)
		if a.defaultAllow {
			attributes.AddAnnotation(AuditKeyPrefix+ImagePolicyFailedOpenKeySuffix, "true")
			annotations := pod.GetAnnotations()
			if annotations == nil {
				annotations = make(map[string]string)
			}
			annotations[api.ImagePolicyFailedOpenKey] = "true"
			pod.ObjectMeta.SetAnnotations(annotations)

			klog.V(2).Infof("resource allowed in spite of webhook backend failure")
			return nil
		}
		klog.V(2).Infof("resource not allowed due to webhook backend failure ")
		return admission.NewForbidden(attributes, err)
	}
	return nil
}

/**
 * Validate - Primary hook for intercepting Pod creation/updates.
 * 
 * Block Logic: Target extraction and request assembly.
 * Logic: 
 * 1. Filters for non-subresource Pod operations.
 * 2. Aggregates all images from InitContainers and standard Containers into 
 *    a unified ImageReview specification.
 * 3. Delegates the decision to admitPod.
 */
func (a *Plugin) Validate(ctx context.Context, attributes admission.Attributes, o admission.ObjectInterfaces) (err error) {
	if attributes.GetSubresource() != "" || attributes.GetResource().GroupResource() != api.Resource("pods") {
		return nil
	}

	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	var imageReviewContainerSpecs []v1alpha1.ImageReviewContainerSpec
	containers := make([]api.Container, 0, len(pod.Spec.Containers)+len(pod.Spec.InitContainers))
	containers = append(containers, pod.Spec.Containers...)
	containers = append(containers, pod.Spec.InitContainers...)
	for _, c := range containers {
		imageReviewContainerSpecs = append(imageReviewContainerSpecs, v1alpha1.ImageReviewContainerSpec{
			Image: c.Image,
		})
	}
	imageReview := v1alpha1.ImageReview{
		Spec: v1alpha1.ImageReviewSpec{
			Containers:  imageReviewContainerSpecs,
			Annotations: a.filterAnnotations(pod.Annotations),
			Namespace:   attributes.GetNamespace(),
		},
	}
	if err := a.admitPod(ctx, pod, attributes, &imageReview); err != nil {
		return admission.NewForbidden(attributes, err)
	}
	return nil
}

/**
 * admitPod - Orchestrates the synchronous validation via remote webhook.
 * 
 * Block Logic: Cache-First evaluation loop.
 * Logic: 
 * 1. Computes a hash of the image request spec for cache lookup.
 * 2. On miss: Executes a POST request to the remote endpoint using exponential backoff.
 * 3. Populates the local cache with the backend response and defined TTL.
 * 4. Merges returned audit annotations into the Kubernetes request metadata.
 */
func (a *Plugin) admitPod(ctx context.Context, pod *api.Pod, attributes admission.Attributes, review *v1alpha1.ImageReview) error {
	cacheKey, err := json.Marshal(review.Spec)
	if err != nil {
		return err
	}
	if entry, ok := a.responseCache.Get(string(cacheKey)); ok {
		review.Status = entry.(v1alpha1.ImageReviewStatus)
	} else {
		result := a.webhook.WithExponentialBackoff(ctx, func() rest.Result {
			return a.webhook.RestClient.Post().Context(ctx).Body(review).Do(context.TODO())
		})

		if err := result.Error(); err != nil {
			return a.webhookError(pod, attributes, err)
		}
		var statusCode int
		if result.StatusCode(&statusCode); statusCode < 200 || statusCode >= 300 {
			return a.webhookError(pod, attributes, fmt.Errorf("Error contacting webhook: %d", statusCode))
		}

		if err := result.Into(review); err != nil {
			return a.webhookError(pod, attributes, err)
		}

		a.responseCache.Add(string(cacheKey), review.Status, a.statusTTL(review.Status))
	}

	for k, v := range review.Status.AuditAnnotations {
		if err := attributes.AddAnnotation(AuditKeyPrefix+k, v); err != nil {
			klog.Warningf("failed to set admission audit annotation %s to %s: %v", AuditKeyPrefix+k, v, err)
		}
	}
	if !review.Status.Allowed {
		if len(review.Status.Reason) > 0 {
			return fmt.Errorf("image policy webhook backend denied one or more images: %s", review.Status.Reason)
		}
		return errors.New("one or more images rejected by webhook backend")
	}
	return nil
}

/**
 * NewImagePolicyWebhook - Factory for plugin instantiation from external config.
 * Logic: Deserializes the admission config (JSON or YAML), normalizes webhook 
 * parameters, and initializes the generic webhook infrastructure with the 
 * provided kubeconfig file.
 */
func NewImagePolicyWebhook(configFile io.Reader) (*Plugin, error) {
	if configFile == nil {
		return nil, fmt.Errorf("no config specified")
	}

	var config AdmissionConfig
	d := yaml.NewYAMLOrJSONDecoder(configFile, 4096)
	err := d.Decode(&config)
	if err != nil {
		return nil, err
	}

	whConfig := config.ImagePolicyWebhook
	if err := normalizeWebhookConfig(&whConfig); err != nil {
		return nil, err
	}

	gw, err := webhook.NewGenericWebhook(legacyscheme.Scheme, legacyscheme.Codecs, whConfig.KubeConfigFile, groupVersions, whConfig.RetryBackoff)
	if err != nil {
		return nil, err
	}
	return &Plugin{
		Handler:       admission.NewHandler(admission.Create, admission.Update),
		webhook:       gw,
		responseCache: cache.NewLRUExpireCache(1024),
		allowTTL:      whConfig.AllowTTL,
		denyTTL:       whConfig.DenyTTL,
		defaultAllow:  whConfig.DefaultAllow,
	}, nil
}
