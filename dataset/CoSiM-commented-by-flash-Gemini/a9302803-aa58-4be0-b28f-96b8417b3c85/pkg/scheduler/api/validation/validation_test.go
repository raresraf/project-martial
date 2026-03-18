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

// Package validation_test contains unit tests for the scheduler API validation logic.
// These tests ensure that the Policy and ExtenderConfig structures adhere to expected rules
// and handle various valid and invalid configurations correctly.
package validation

import (
	"errors"
	"fmt"
	"testing"

	"k8s.io/kubernetes/pkg/scheduler/api"
)

func TestValidatePolicy(t *testing.T) {
	// Defines a series of test cases for `ValidatePolicy` to ensure various policy configurations
	// are correctly validated, including priority weights, extender configurations, and resource names.
	tests := []struct {
		policy   api.Policy
		expected error
		name     string
	}{
		// Test case: Verifies that a priority policy without a defined weight (implicitly zero or negative)
		// results in an error, as weights must be positive.
		{
			name:     "no weight defined in policy",
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "NoWeightPriority"}}},
			expected: errors.New("Priority NoWeightPriority should have a positive weight applied to it or it has overflown"),
		},
		// Test case: Ensures that a priority policy with an explicitly set zero weight
		// generates an error, reinforcing the requirement for positive weights.
		{
			name:     "policy weight is not positive",
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "NoWeightPriority", Weight: 0}}},
			expected: errors.New("Priority NoWeightPriority should have a positive weight applied to it or it has overflown"),
		},
		// Test case: Confirms that a priority policy with a valid positive weight
		// passes validation without any errors.
		{
			name:     "valid weight priority",
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "WeightPriority", Weight: 2}}},
			expected: nil,
		},
		// Test case: Validates that a priority policy with a negative weight
		// correctly triggers an error, as negative weights are not allowed.
		{
			name:     "invalid negative weight policy",
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "WeightPriority", Weight: -2}}},
			expected: errors.New("Priority WeightPriority should have a positive weight applied to it or it has overflown"),
		},
		// Test case: Checks that a priority policy with a weight exceeding the maximum allowed value
		// results in an error, ensuring weight constraints are enforced.
		{
			name:     "policy weight exceeds maximum",
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "WeightPriority", Weight: api.MaxWeight}}},
			expected: errors.New("Priority WeightPriority should have a positive weight applied to it or it has overflown"),
		},
		// Test case: Verifies that an extender configuration with a valid positive weight
		// for prioritization passes validation.
		{
			name:     "valid weight in policy extender config",
			policy:   api.Policy{ExtenderConfigs: []api.ExtenderConfig{{URLPrefix: "http://127.0.0.1:8081/extender", PrioritizeVerb: "prioritize", Weight: 2}}},
			expected: nil,
		},
		// Test case: Ensures that an extender configuration with a negative weight for prioritization
		// correctly triggers an error.
		{
			name:     "invalid negative weight in policy extender config",
			policy:   api.Policy{ExtenderConfigs: []api.ExtenderConfig{{URLPrefix: "http://127.0.0.1:8081/extender", PrioritizeVerb: "prioritize", Weight: -2}}},
			expected: errors.New("Priority for extender http://127.0.0.1:8081/extender should have a positive weight applied to it"),
		},
		// Test case: Confirms that an extender configuration with a valid filter verb and URL prefix
		// passes validation.
		{
			name:     "valid filter verb and url prefix",
			policy:   api.Policy{ExtenderConfigs: []api.ExtenderConfig{{URLPrefix: "http://127.0.0.1:8081/extender", FilterVerb: "filter"}}},
			expected: nil,
		},
		// Test case: Validates that an extender configuration with a valid preempt verb and URL prefix
		// passes validation.
		{
			name:     "valid preemt verb and urlprefix",
			policy:   api.Policy{ExtenderConfigs: []api.ExtenderConfig{{URLPrefix: "http://127.0.0.1:8081/extender", PreemptVerb: "preempt"}}},
			expected: nil,
		},
		// Test case: Ensures that defining multiple extenders with the "bind" verb
		// results in an error, as only one extender can handle binding.
		{
			name: "invalid multiple extenders",
			policy: api.Policy{
				ExtenderConfigs: []api.ExtenderConfig{
					{URLPrefix: "http://127.0.0.1:8081/extender", BindVerb: "bind"},
					{URLPrefix: "http://127.0.0.1:8082/extender", BindVerb: "bind"},
				}},
			expected: errors.New("Only one extender can implement bind, found 2"),
		},
		// Test case: Checks that duplicate managed resource names across different extenders
		// correctly trigger an error.
		{
			name: "invalid duplicate extender resource name",
			policy: api.Policy{
				ExtenderConfigs: []api.ExtenderConfig{
					{URLPrefix: "http://127.0.0.1:8081/extender", ManagedResources: []api.ExtenderManagedResource{{Name: "foo.com/bar"}}},
					{URLPrefix: "http://127.0.0.1:8082/extender", BindVerb: "bind", ManagedResources: []api.ExtenderManagedResource{{Name: "foo.com/bar"}}},
				}},
			expected: errors.New("Duplicate extender managed resource name foo.com/bar"),
		},
		// Test case: Verifies that an invalid extended resource name (e.g., using "kubernetes.io" prefix)
		// results in an error.
		{
			name: "invalid extended resource name",
			policy: api.Policy{
				ExtenderConfigs: []api.ExtenderConfig{
					{URLPrefix: "http://127.0.0.1:8081/extender", ManagedResources: []api.ExtenderManagedResource{{Name: "kubernetes.io/foo"}}},
				}},
			expected: errors.New("kubernetes.io/foo is an invalid extended resource name"),
		},
	}

	// Iterates through each defined test case, executes the `ValidatePolicy` function,
	// and compares the actual result with the expected error.
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := ValidatePolicy(test.policy)
			if fmt.Sprint(test.expected) != fmt.Sprint(actual) {
				t.Errorf("expected: %s, actual: %s", test.expected, actual)
			}
		})
	}
}
