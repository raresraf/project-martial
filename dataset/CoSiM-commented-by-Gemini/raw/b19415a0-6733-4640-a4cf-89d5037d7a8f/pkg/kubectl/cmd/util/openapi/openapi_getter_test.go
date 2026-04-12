/*
Copyright 2017 The Kubernetes Authors.

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

// Package openapi_test validates the behavior of the OpenAPI client components.
// It focuses on ensuring that schema retrieval is efficient, idempotent, and 
// correctly handles both transient and terminal server states.
package openapi_test

import (
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
	tst "k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi/testing"
)

// Describe block for testing the resource retrieval logic.
// Focus: Validation of the lazy-loading and memoization pattern implemented in openapi.Getter.
var _ = Describe("Getting the Resources", func() {
	var client *tst.FakeClient
	var expectedData openapi.Resources
	var instance openapi.Getter

	// Setup: Orchestrates the test environment before each specification.
	// Logic: Pre-parses a fake schema and wraps a stateful mock client with the target getter.
	BeforeEach(func() {
		client = tst.NewFakeClient(&fakeSchema)
		d, err := fakeSchema.OpenAPISchema()
		Expect(err).To(BeNil())

		expectedData, err = openapi.NewOpenAPIData(d)
		Expect(err).To(BeNil())

		instance = openapi.NewOpenAPIGetter(client)
	})

	// Context: Happy-path scenario for schema acquisition.
	Context("when the server returns a successful result", func() {
		// Specification: Verifies the internal memoization/caching mechanism.
		// Invariant: The backend client should only be invoked once, regardless of 
		// how many times the getter's public Get() method is called.
		It("should return the same data for multiple calls", func() {
			// Pre-condition: No calls made yet.
			Expect(client.Calls).To(Equal(0))

			// Action: First retrieval triggers network simulation.
			result, err := instance.Get()
			Expect(err).To(BeNil())
			Expect(result).To(Equal(expectedData))
			Expect(client.Calls).To(Equal(1))

			// Action: Second retrieval should hit the internal cache.
			result, err = instance.Get()
			Expect(err).To(BeNil())
			Expect(result).To(Equal(expectedData))
			
			// Verification: Client call count remains 1, confirming memoization.
			Expect(client.Calls).To(Equal(1))
		})
	})

	// Context: Error-handling scenario for schema acquisition.
	Context("when the server returns an unsuccessful result", func() {
		// Specification: Verifies consistency in error propagation and caching.
		// Logic: If the first call fails, subsequent calls should not re-attempt 
		// the request if the implementation caches the failure state.
		It("should return the same instance for multiple calls.", func() {
			Expect(client.Calls).To(Equal(0))

			// Fault Injection: Simulate a server-side or connection error.
			client.Err = fmt.Errorf("expected error")
			_, err := instance.Get()
			Expect(err).To(Equal(client.Err))
			Expect(client.Calls).To(Equal(1))

			// Action: Repeat call.
			_, err = instance.Get()
			Expect(err).To(Equal(client.Err))
			
			// Verification: Ensures the error state is also handled idempotently.
			Expect(client.Calls).To(Equal(1))
		})
	})
})
