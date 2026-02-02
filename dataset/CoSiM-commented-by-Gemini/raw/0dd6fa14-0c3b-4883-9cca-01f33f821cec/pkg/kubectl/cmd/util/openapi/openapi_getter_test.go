/**
 * @file openapi_getter_test.go
 * @brief Tests for the OpenAPI getter utility.
 * @details This file contains unit tests for the OpenAPI getter, which is
 * responsible for fetching and caching OpenAPI schema data. The tests use the
 * Ginkgo BDD framework to verify that the getter correctly caches both
 * successful and unsuccessful results, preventing redundant calls to the
 * underlying schema source.
 */
/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUTHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package openapi_test

import (
	"fmt"

	openapi_v2 "github.com/googleapis/gnostic/OpenAPIv2"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
)

// FakeCounter is a mock implementation of the openapi.Getter interface
// used for testing. It simulates fetching an OpenAPI schema.
type FakeCounter struct {
	Calls int   // Counts the number of times OpenAPISchema is called.
	Err   error // The error to be returned by OpenAPISchema.
}

// OpenAPISchema simulates a call to fetch the OpenAPI schema. It increments
// the call counter and returns a predefined error, if any.
func (f *FakeCounter) OpenAPISchema() (*openapi_v2.Document, error) {
	f.Calls = f.Calls + 1
	return nil, f.Err
}

var _ = Describe("Getting the Resources", func() {
	var client FakeCounter
	var instance openapi.Getter
	var expectedData openapi.Resources

	// BeforeEach sets up a new test environment for each test case.
	BeforeEach(func() {
		client = FakeCounter{}
		instance = openapi.NewOpenAPIGetter(&client)
		var err error
		// expectedData is initialized with a nil document, simulating an empty schema.
		expectedData, err = openapi.NewOpenAPIData(nil)
		Expect(err).To(BeNil())
	})

	// Context for tests where the server is expected to return a successful result.
	Context("when the server returns a successful result", func() {
		// It block to test the caching of successful results.
		It("should return the same data for multiple calls", func() {
			Expect(client.Calls).To(Equal(0))

			// First call to Get() should fetch the schema.
			result, err := instance.Get()
			Expect(err).To(BeNil())
			Expect(result).To(Equal(expectedData))
			Expect(client.Calls).To(Equal(1))

			// Second call to Get() should return the cached result.
			result, err = instance.Get()
			Expect(err).To(BeNil())
			Expect(result).To(Equal(expectedData))
			// The client should not be called again.
			Expect(client.Calls).To(Equal(1))
		})
	})

	// Context for tests where the server is expected to return an error.
	Context("when the server returns an unsuccessful result", func() {
		// It block to test the caching of errors.
		It("should return the same instance for multiple calls.", func() {
			Expect(client.Calls).To(Equal(0))

			// Set up the mock to return an error.
			client.Err = fmt.Errorf("expected error")
			// First call to Get() should result in an error.
			_, err := instance.Get()
			Expect(err).To(Equal(client.Err))
			Expect(client.Calls).To(Equal(1))

			// Second call to Get() should return the cached error.
			_, err = instance.Get()
			Expect(err).To(Equal(client.Err))
			// The client should not be called again.
			Expect(client.Calls).To(Equal(1))
		})
	})
})
