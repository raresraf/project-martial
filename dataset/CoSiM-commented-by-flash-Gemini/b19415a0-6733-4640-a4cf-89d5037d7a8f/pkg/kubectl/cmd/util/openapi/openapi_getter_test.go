/**
 * @b19415a0-6733-4640-a4cf-89d5037d7a8f/pkg/kubectl/cmd/util/openapi/openapi_getter_test.go
 * @brief Behavior-driven tests for the OpenAPI Getter lifecycle.
 * Domain: Software Testing, API Metadata, Memoization Patterns.
 * Architecture: Employs Ginkgo BDD framework to validate that the OpenAPI getter correctly implements memoization (caching) for both success and error states.
 * Functional Utility: Ensures that the expensive OpenAPI schema retrieval and parsing operations are executed exactly once per getter instance.
 */

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

package openapi_test

import (
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
	tst "k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi/testing"
)

/**
 * @brief Test suite orchestrating the verification of resource retrieval logic.
 */
var _ = Describe("Getting the Resources", func() {
	var client *tst.FakeClient
	var expectedData openapi.Resources
	var instance openapi.Getter

	/**
	 * @brief Pre-test setup to initialize the mock client and pre-parse the schema.
	 */
	BeforeEach(func() {
		client = tst.NewFakeClient(&fakeSchema)
		d, err := fakeSchema.OpenAPISchema()
		Expect(err).To(BeNil())

		expectedData, err = openapi.NewOpenAPIData(d)
		Expect(err).To(BeNil())

		instance = openapi.NewOpenAPIGetter(client)
	})

	/**
	 * @brief Context validating caching behavior on successful schema retrieval.
	 * Invariant: Subsequent calls to Get() must return cached data without additional client invocations.
	 */
	Context("when the server returns a successful result", func() {
		It("should return the same data for multiple calls", func() {
			Expect(client.Calls).To(Equal(0))

			result, err := instance.Get()
			Expect(err).To(BeNil())
			Expect(result).To(Equal(expectedData))
			Expect(client.Calls).To(Equal(1))

			result, err = instance.Get()
			Expect(err).To(BeNil())
			Expect(result).To(Equal(expectedData))
			// Functional Utility: Verification of internal memoization (cached state).
			// No additional client calls expected
			Expect(client.Calls).To(Equal(1))
		})
	})

	/**
	 * @brief Context validating caching behavior when the server returns an error.
	 * Invariant: Errors are also memoized to prevent redundant failed network calls.
	 */
	Context("when the server returns an unsuccessful result", func() {
		It("should return the same instance for multiple calls.", func() {
			Expect(client.Calls).To(Equal(0))

			client.Err = fmt.Errorf("expected error")
			_, err := instance.Get()
			Expect(err).To(Equal(client.Err))
			Expect(client.Calls).To(Equal(1))

			_, err = instance.Get()
			Expect(err).To(Equal(client.Err))
			// Functional Utility: Ensures error states are consistently propagated from the cache.
			// No additional client calls expected
			Expect(client.Calls).To(Equal(1))
		})
	})
})
