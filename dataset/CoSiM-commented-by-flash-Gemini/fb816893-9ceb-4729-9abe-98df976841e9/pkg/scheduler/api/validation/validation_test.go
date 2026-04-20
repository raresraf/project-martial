/*
Copyright 2015 The Kubernetes Authors.

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

// @fb816893-9ceb-4729-9abe-98df976841e9/pkg/scheduler/api/validation/validation_test.go
// @brief Unit tests for Kubernetes scheduler policy validation logic.
//
// Functional Intent: Ensures the structural and semantic integrity of scheduler 
// policies. Validates constraints on priority weights, extender configurations, 
// binding implementations, and managed resource naming conventions.
package validation

import (
	"errors"
	"fmt"
	"testing"

	"k8s.io/kubernetes/pkg/scheduler/api"
)

/**
 * TestValidatePolicy - Entry point for policy validation test suite.
 * Logic: Employs table-driven testing to verify various edge cases and failure 
 * modes in scheduler policies, including weight overflows and duplicate configurations.
 */
func TestValidatePolicy(t *testing.T) {
	// Block Logic: Definition of test vectors for policy validation.
	// Invariant: Each test case maps a policy configuration to its expected error outcome.
	tests := []struct {
		policy   api.Policy
		expected error
	}{
		{
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "NoWeightPriority"}}},
			expected: errors.New("Priority NoWeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "NoWeightPriority", Weight: 0}}},
			expected: errors.New("Priority NoWeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "WeightPriority", Weight: 2}}},
			expected: nil,
		},
		{
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "WeightPriority", Weight: -2}}},
			expected: errors.New("Priority WeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "WeightPriority", Weight: api.MaxWeight}}},
			expected: errors.New("Priority WeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			policy:   api.Policy{ExtenderConfigs: []api.ExtenderConfig{{URLPrefix: "http://127.0.0.1:8081/extender", PrioritizeVerb: "prioritize", Weight: 2}}},
			expected: nil,
		},
		{
			policy:   api.Policy{ExtenderConfigs: []api.ExtenderConfig{{URLPrefix: "http://127.0.0.1:8081/extender", PrioritizeVerb: "prioritize", Weight: -2}}},
			expected: errors.New("Priority for extender http://127.0.0.1:8081/extender should have a positive weight applied to it"),
		},
		{
			policy:   api.Policy{ExtenderConfigs: []api.ExtenderConfig{{URLPrefix: "http://127.0.0.1:8081/extender", FilterVerb: "filter"}}},
			expected: nil,
		},
		{
			policy:   api.Policy{ExtenderConfigs: []api.ExtenderConfig{{URLPrefix: "http://127.0.0.1:8081/extender", PreemptVerb: "preempt"}}},
			expected: nil,
		},
		{
			policy: api.Policy{
				ExtenderConfigs: []api.ExtenderConfig{
					{URLPrefix: "http://127.0.0.1:8081/extender", BindVerb: "bind"},
					{URLPrefix: "http://127.0.0.1:8082/extender", BindVerb: "bind"},
				}},
			expected: errors.New("Only one extender can implement bind, found 2"),
		},
		{
			policy: api.Policy{
				ExtenderConfigs: []api.ExtenderConfig{
					{URLPrefix: "http://127.0.0.1:8081/extender", ManagedResources: []api.ExtenderManagedResource{{Name: "foo.com/bar"}}},
					{URLPrefix: "http://127.0.0.1:8082/extender", BindVerb: "bind", ManagedResources: []api.ExtenderManagedResource{{Name: "foo.com/bar"}}},
				}},
			expected: errors.New("Duplicate extender managed resource name foo.com/bar"),
		},
		{
			policy: api.Policy{
				ExtenderConfigs: []api.ExtenderConfig{
					{URLPrefix: "http://127.0.0.1:8081/extender", ManagedResources: []api.ExtenderManagedResource{{Name: "kubernetes.io/foo"}}},
				}},
			expected: errors.New("kubernetes.io/foo is an invalid extended resource name"),
		},
	}

	// Block Logic: Execution of the test battery.
	// Logic: Iterates through each test case, invokes the validator, and performs 
	// string-based comparison of error messages to confirm behavioral correctness.
	for _, test := range tests {
		actual := ValidatePolicy(test.policy)
		if fmt.Sprint(test.expected) != fmt.Sprint(actual) {
			t.Errorf("expected: %s, actual: %s", test.expected, actual)
		}
	}
}
