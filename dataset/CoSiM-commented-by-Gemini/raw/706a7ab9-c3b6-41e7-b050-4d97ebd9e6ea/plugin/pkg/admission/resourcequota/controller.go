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
 * @file controller.go
 * @brief Implementation of the ResourceQuota admission evaluator.
 * 
 * Architectural Intent: Enforces resource consumption limits within a Kubernetes namespace. 
 * Intercepts mutation requests (Create, Update) and validates the delta impact against 
 * defined quota constraints.
 * 
 * Scalability Optimization: Implements a batched evaluation strategy using a namespace-keyed 
 * work queue. This reduces lock contention on shared resource quota documents by bundling 
 * multiple concurrent admission requests for the same namespace into a single atomic update.
 */

package resourcequota

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/util/workqueue"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/generic"
	_ "k8s.io/kubernetes/pkg/util/reflector/prometheus" // for reflector metric registration
	_ "k8s.io/kubernetes/pkg/util/workqueue/prometheus" // for workqueue metric registration
	resourcequotaapi "k8s.io/kubernetes/plugin/pkg/admission/resourcequota/apis/resourcequota"
)

// Evaluator is used to see if quota constraints are satisfied.
type Evaluator interface {
	// Evaluate takes an operation and checks to see if quota constraints are satisfied.  It returns an error if they are not.
	// The default implementation process related operations in chunks when possible.
	Evaluate(a admission.Attributes) error
}

/**
 * @struct quotaEvaluator
 * @brief Logic orchestrator for quota enforcement.
 */
type quotaEvaluator struct {
	quotaAccessor QuotaAccessor
	// lockAcquisitionFunc acquires any required locks and returns a cleanup method to defer
	lockAcquisitionFunc func([]api.ResourceQuota) func()

	ignoredResources map[schema.GroupResource]struct{}

	// registry that knows how to measure usage for objects
	registry quota.Registry

	// queue is indexed by namespace, so that we bundle up on a per-namespace basis
	queue      *workqueue.Type
	workLock   sync.Mutex
	work       map[string][]*admissionWaiter
	dirtyWork  map[string][]*admissionWaiter
	inProgress sets.String

	// controls the run method so that we can cleanly conform to the Evaluator interface
	workers int
	stopCh  <-chan struct{}
	init    sync.Once

	// lets us know what resources are limited by default
	config *resourcequotaapi.Configuration
}

/**
 * @struct admissionWaiter
 * @brief Represents a thread-safe barrier for a single admission request.
 * Logic: Encapsulates request attributes and a signal channel to notify completion.
 */
type admissionWaiter struct {
	attributes admission.Attributes
	finished   chan struct{}
	result     error
}

type defaultDeny struct{}

func (defaultDeny) Error() string {
	return "DEFAULT DENY"
}

// IsDefaultDeny returns true if the error is defaultDeny
func IsDefaultDeny(err error) bool {
	if err == nil {
		return false
	}

	_, ok := err.(defaultDeny)
	return ok
}

func newAdmissionWaiter(a admission.Attributes) *admissionWaiter {
	return &admissionWaiter{
		attributes: a,
		finished:   make(chan struct{}),
		result:     defaultDeny{},
	}
}

// NewQuotaEvaluator configures an admission controller that can enforce quota constraints
// using the provided registry.
func NewQuotaEvaluator(quotaAccessor QuotaAccessor, ignoredResources map[schema.GroupResource]struct{}, quotaRegistry quota.Registry, lockAcquisitionFunc func([]api.ResourceQuota) func(), config *resourcequotaapi.Configuration, workers int, stopCh <-chan struct{}) Evaluator {
	if config == nil {
		config = &resourcequotaapi.Configuration{}
	}

	return &quotaEvaluator{
		quotaAccessor:       quotaAccessor,
		lockAcquisitionFunc: lockAcquisitionFunc,

		ignoredResources: ignoredResources,
		registry:         quotaRegistry,

		queue:      workqueue.NewNamed("admission_quota_controller"),
		work:       map[string][]*admissionWaiter{},
		dirtyWork:  map[string][]*admissionWaiter{},
		inProgress: sets.String{},

		workers: workers,
		stopCh:  stopCh,
		config:  config,
	}
}

// Run begins watching and syncing.
func (e *quotaEvaluator) run() {
	defer utilruntime.HandleCrash()

	for i := 0; i < e.workers; i++ {
		go wait.Until(e.doWork, time.Second, e.stopCh)
	}
	<-e.stopCh
	glog.Infof("Shutting down quota evaluator")
	e.queue.ShutDown()
}

/**
 * @brief Worker loop for processing batched admission requests.
 */
func (e *quotaEvaluator) doWork() {
	workFunc := func() bool {
		ns, admissionAttributes, quit := e.getWork()
		if quit {
			return true
		}
		defer e.completeWork(ns)
		if len(admissionAttributes) == 0 {
			return false
		}
		e.checkAttributes(ns, admissionAttributes)
		return false
	}
	for {
		if quit := workFunc(); quit {
			glog.Infof("quota evaluator worker shutdown")
			return
		}
	}
}

// checkAttributes evaluates all the waiting admissionAttributes.
func (e *quotaEvaluator) checkAttributes(ns string, admissionAttributes []*admissionWaiter) {
	// notify all on exit
	defer func() {
		for _, admissionAttribute := range admissionAttributes {
			close(admissionAttribute.finished)
		}
	}()

	quotas, err := e.quotaAccessor.GetQuotas(ns)
	if err != nil {
		for _, admissionAttribute := range admissionAttributes {
			admissionAttribute.result = err
		}
		return
	}
	// if limited resources are disabled, we can just return safely when there are no quotas.
	limitedResourcesDisabled := len(e.config.LimitedResources) == 0
	if len(quotas) == 0 && limitedResourcesDisabled {
		for _, admissionAttribute := range admissionAttributes {
			admissionAttribute.result = nil
		}
		return
	}

	if e.lockAcquisitionFunc != nil {
		releaseLocks := e.lockAcquisitionFunc(quotas)
		defer releaseLocks()
	}

	e.checkQuotas(quotas, admissionAttributes, 3)
}

/**
 * @brief Core evaluation logic for verifying and committing quota updates.
 * Algorithm: 
 * 1. Simulation: Iteratively apply each request to a local copy of the quotas.
 * 2. Verification: If any request exceeds limits, mark it failed.
 * 3. Committal: Atomically update the persistent quota status for all successful requests.
 * 4. Conflict Resolution: Retries with fresh data if concurrent updates cause version mismatches.
 */
func (e *quotaEvaluator) checkQuotas(quotas []api.ResourceQuota, admissionAttributes []*admissionWaiter, remainingRetries int) {
	// yet another copy to compare against originals to see if we actually have deltas
	originalQuotas, err := copyQuotas(quotas)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}

	atLeastOneChanged := false
	/**
	 * Block Logic: Batch simulation pass.
	 * Invariant: Requests are processed sequentially within the batch. 
	 * Successfully simulated requests advance the 'quotas' state for subsequent items.
	 */
	for i := range admissionAttributes {
		admissionAttribute := admissionAttributes[i]
		newQuotas, err := e.checkRequest(quotas, admissionAttribute.attributes)
		if err != nil {
			admissionAttribute.result = err
			continue
		}

		atLeastOneChangeForThisWaiter := false
		for j := range newQuotas {
			if !quota.Equals(quotas[j].Status.Used, newQuotas[j].Status.Used) {
				atLeastOneChanged = true
				atLeastOneChangeForThisWaiter = true
				break
			}
		}

		if !atLeastOneChangeForThisWaiter {
			admissionAttribute.result = nil
		}

		quotas = newQuotas
	}

	if !atLeastOneChanged {
		return
	}

	var updatedFailedQuotas []api.ResourceQuota
	var lastErr error
	/**
	 * Block Logic: Persistence phase.
	 * Logic: Triggers status updates for any quota document that observed a usage increase.
	 */
	for i := range quotas {
		newQuota := quotas[i]

		if quota.Equals(originalQuotas[i].Status.Used, newQuota.Status.Used) {
			continue
		}

		if err := e.quotaAccessor.UpdateQuotaStatus(&newQuota); err != nil {
			updatedFailedQuotas = append(updatedFailedQuotas, newQuota)
			lastErr = err
		}
	}

	if len(updatedFailedQuotas) == 0 {
		for _, admissionAttribute := range admissionAttributes {
			if IsDefaultDeny(admissionAttribute.result) {
				admissionAttribute.result = nil
			}
		}
		return
	}

	if remainingRetries <= 0 {
		for _, admissionAttribute := range admissionAttributes {
			if IsDefaultDeny(admissionAttribute.result) {
				admissionAttribute.result = lastErr
			}
		}
		return
	}

	newQuotas, err := e.quotaAccessor.GetQuotas(quotas[0].Namespace)
	if err != nil {
		for _, admissionAttribute := range admissionAttributes {
			if IsDefaultDeny(admissionAttribute.result) {
				admissionAttribute.result = lastErr
			}
		}
		return
	}

	quotasToCheck := []api.ResourceQuota{}
	for _, newQuota := range newQuotas {
		for _, oldQuota := range updatedFailedQuotas {
			if newQuota.Name == oldQuota.Name {
				quotasToCheck = append(quotasToCheck, newQuota)
				break
			}
		}
	}
	e.checkQuotas(quotasToCheck, admissionAttributes, remainingRetries-1)
}

func copyQuotas(in []api.ResourceQuota) ([]api.ResourceQuota, error) {
	out := make([]api.ResourceQuota, 0, len(in))
	for _, quota := range in {
		out = append(out, *quota.DeepCopy())
	}

	return out, nil
}

// ... (Helper filters and scope matchers) ...

/**
 * @brief Evaluates a single admission request against the active quota set.
 * Functional Utility: Computes the delta usage of the incoming object and checks 
 * if it fits within the 'hard' limits of all matching ResourceQuotas.
 */
func (e *quotaEvaluator) checkRequest(quotas []api.ResourceQuota, a admission.Attributes) ([]api.ResourceQuota, error) {
	namespace := a.GetNamespace()
	evaluator := e.registry.Get(a.GetResource().GroupResource())
	if evaluator == nil || !evaluator.Handles(a) {
		return quotas, nil
	}

	inputObject := a.GetObject()
	// ... (Lookup limited scopes and resource names) ...

	interestingQuotaIndexes := []int{}
	restrictedResourcesSet := sets.String{}
	restrictedScopes := []api.ScopedResourceSelectorRequirement{}
	/**
	 * Block Logic: Constraint evaluation.
	 * Invariant: The request must satisfy every ResourceQuota that targets its ResourceType/Scope.
	 */
	for i := range quotas {
		resourceQuota := quotas[i]
		// ... (Check scope selectors and matches) ...

		hardResources := quota.ResourceNames(resourceQuota.Status.Hard)
		restrictedResources := evaluator.MatchingResources(hardResources)
		if err := evaluator.Constraints(restrictedResources, inputObject); err != nil {
			return nil, admission.NewForbidden(a, fmt.Errorf("failed quota: %s: %v", resourceQuota.Name, err))
		}
		if !hasUsageStats(&resourceQuota) {
			return nil, admission.NewForbidden(a, fmt.Errorf("status unknown for quota: %s", resourceQuota.Name))
		}
		interestingQuotaIndexes = append(interestingQuotaIndexes, i)
		localRestrictedResourcesSet := quota.ToSet(restrictedResources)
		restrictedResourcesSet.Insert(localRestrictedResourcesSet.List()...)
	}

	// ... (Verify covering quotas) ...

	deltaUsage, err := evaluator.Usage(inputObject)
	if err != nil {
		return quotas, err
	}

	if admission.Update == a.GetOperation() {
		prevItem := a.GetOldObject()
		if prevItem == nil {
			return nil, admission.NewForbidden(a, fmt.Errorf("unable to get previous usage since prior version of object was not found"))
		}

		metadata, err := meta.Accessor(prevItem)
		if err == nil && len(metadata.GetResourceVersion()) > 0 {
			prevUsage, innerErr := evaluator.Usage(prevItem)
			if innerErr != nil {
				return quotas, innerErr
			}
			deltaUsage = quota.SubtractWithNonNegativeResult(deltaUsage, prevUsage)
		}
	}

	if quota.IsZero(deltaUsage) {
		return quotas, nil
	}

	outQuotas, err := copyQuotas(quotas)
	if err != nil {
		return nil, err
	}

	/**
	 * Block Logic: Usage accumulation.
	 * Logic: For each relevant quota, adds the delta usage to the current status and 
	 * checks if the resulting value exceeds the hard limit.
	 */
	for _, index := range interestingQuotaIndexes {
		resourceQuota := outQuotas[index]

		hardResources := quota.ResourceNames(resourceQuota.Status.Hard)
		requestedUsage := quota.Mask(deltaUsage, hardResources)
		newUsage := quota.Add(resourceQuota.Status.Used, requestedUsage)
		maskedNewUsage := quota.Mask(newUsage, quota.ResourceNames(requestedUsage))

		if allowed, exceeded := quota.LessThanOrEqual(maskedNewUsage, resourceQuota.Status.Hard); !allowed {
			failedRequestedUsage := quota.Mask(requestedUsage, exceeded)
			failedUsed := quota.Mask(resourceQuota.Status.Used, exceeded)
			failedHard := quota.Mask(resourceQuota.Status.Hard, exceeded)
			return nil, admission.NewForbidden(a,
				fmt.Errorf("exceeded quota: %s, requested: %s, used: %s, limited: %s",
					resourceQuota.Name,
					prettyPrint(failedRequestedUsage),
					prettyPrint(failedUsed),
					prettyPrint(failedHard)))
		}

		outQuotas[index].Status.Used = newUsage
	}

	return outQuotas, nil
}

// ... (Helper functions for Evaluate, addWork, etc.) ...
