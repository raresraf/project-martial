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

// Package openapi_test provides unit tests for the openapi package.
// This file specifically tests the OpenAPI getter functionality.
// Domain: Kubernetes, OpenAPI, Testing.
package openapi_test

import (
	"fmt"

	openapi_v2 "github.com/googleapis/gnostic/OpenAPIv2"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
)

// FakeCounter returns a "null" document and the specified error. It
// also counts how many times the OpenAPISchema method has been called.
// FakeCounter is a mock implementation of a client that returns a "null" OpenAPI document
// and the specified error. It tracks the number of times its OpenAPISchema method has been called.
type FakeCounter struct {
	// Calls tracks the number of times OpenAPISchema has been invoked.
	Calls int
	// Err is the error to be returned by OpenAPISchema.
	Err   error
}

// OpenAPISchema simulates fetching an OpenAPI schema.
// Functional Utility: Increments the call counter and returns a nil document with a predefined error.
// Post-condition: `f.Calls` is incremented by 1.
func (f *FakeCounter) OpenAPISchema() (*openapi_v2.Document, error) {
	f.Calls = f.Calls + 1
	return nil, f.Err
}

// Test Suite: Getting the Resources
// Functional Utility: Defines a test suite for the OpenAPI getter, ensuring its caching
// behavior and error handling are correct regardless of server response.
var _ = Describe("Getting the Resources", func() {
	var client FakeCounter
	var instance openapi.Getter
	var expectedData openapi.Resources

	// Block Logic: Initializes a new FakeCounter client and OpenAPI getter instance
	//              before each test case. Also initializes expectedData to a nil OpenAPI data.
	// Functional Utility: Sets up a clean state for each test to ensure isolation and reproducibility.
	BeforeEach(func() {
		client = FakeCounter{}
		instance = openapi.NewOpenAPIGetter(&client)
		var err error
		expectedData, err = openapi.NewOpenAPIData(nil)
		Expect(err).To(BeNil())
	})

	// Context: Tests the behavior when the underlying server call is successful.
	Context("when the server returns a successful result", func() {
		// Test Case: Verifies that the getter returns the same cached data for multiple calls
		//            after an initial successful fetch.
		// Functional Utility: Confirms the caching mechanism of the OpenAPI getter.
		It("should return the same data for multiple calls", func() {
			// Pre-condition: The mock client's OpenAPISchema method has not been called yet.
			Expect(client.Calls).To(Equal(0))

			// Block Logic: First call to Get, which should trigger a call to the underlying client.
			result, err := instance.Get()
			Expect(err).To(BeNil())
			Expect(result).To(Equal(expectedData))
			// Invariant: The client's OpenAPISchema method should have been called exactly once.
			Expect(client.Calls).To(Equal(1))

			// Block Logic: Second call to Get, which should return cached data without calling the client again.
			result, err = instance.Get()
			Expect(err).To(BeNil())
			Expect(result).To(Equal(expectedData))
			// Invariant: The client's OpenAPISchema method should still have been called exactly once,
			//            demonstrating caching behavior.
			Expect(client.Calls).To(Equal(1))
		})
	})

	// Context: Tests the behavior when the underlying server call returns an unsuccessful result.
	Context("when the server returns an unsuccessful result", func() {
		// Test Case: Verifies that the getter returns the same error and does not make
		//            additional calls to the client after an initial failed fetch.
		// Functional Utility: Confirms that errors are cached and subsequent calls do not retry the fetch.
		It("should return the same instance for multiple calls.", func() {
			// Pre-condition: The mock client's OpenAPISchema method has not been called yet.
			Expect(client.Calls).To(Equal(0))

			// Block Logic: First call to Get, simulating a server error.
			client.Err = fmt.Errorf("expected error")
			_, err := instance.Get()
			Expect(err).To(Equal(client.Err))
			// Invariant: The client's OpenAPISchema method should have been called exactly once.
			Expect(client.Calls).To(Equal(1))

			// Block Logic: Second call to Get, which should return the cached error without calling the client again.
			_, err = instance.Get()
			Expect(err).To(Equal(client.Err))
			// Invariant: The client's OpenAPISchema method should still have been called exactly once,
			//            demonstrating error caching behavior.
			Expect(client.Calls).To(Equal(1))
		})
	})
})
