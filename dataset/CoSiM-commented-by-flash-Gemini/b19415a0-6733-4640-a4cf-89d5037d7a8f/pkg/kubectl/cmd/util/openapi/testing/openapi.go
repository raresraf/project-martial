/**
 * @b19415a0-6733-4640-a4cf-89d5037d7a8f/pkg/kubectl/cmd/util/openapi/testing/openapi.go
 * @brief Test fakes and mocks for Kubernetes OpenAPI schema orchestration.
 * Domain: Software Testing, API Metadata, OpenAPI/Swagger.
 * Architecture: Implements the 'Fake' and 'FakeClient' types to simulate server-side OpenAPI discovery without network overhead.
 * Functional Utility: Facilitates idempotent loading of schema documents from the local filesystem and provides call-tracking for validation of client-side caching logic.
 * Synchronization: Uses sync.Once for thread-safe lazy-loading of the schema document, ensuring consistent state across concurrent test workers.
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

package testing

import (
	"io/ioutil"
	"os"
	"sync"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-openapi/pkg/util/proto"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"

	yaml "gopkg.in/yaml.v2"

	"github.com/googleapis/gnostic/OpenAPIv2"
	"github.com/googleapis/gnostic/compiler"
)

/**
 * @brief Thread-safe provider for a mocked OpenAPI document.
 * Strategy: Parses a filesystem-based swagger spec exactly once to minimize test latency.
 */
// Fake opens and returns a openapi swagger from a file Path. It will
// parse only once and then return the same copy everytime.
type Fake struct {
	Path string

	once     sync.Once
	document *openapi_v2.Document
	err      error
}

/**
 * @brief Retrieves the cached OpenAPI document or parses it if first invocation.
 * Synchronization: Guarded by sync.Once to ensure atomic initialization in concurrent environments.
 */
// OpenAPISchema returns the openapi document and a potential error.
func (f *Fake) OpenAPISchema() (*openapi_v2.Document, error) {
	f.once.Do(func() {
		_, err := os.Stat(f.Path)
		if err != nil {
			f.err = err
			return
		}
		spec, err := ioutil.ReadFile(f.Path)
		if err != nil {
			f.err = err
			return
		}
		var info yaml.MapSlice
		err = yaml.Unmarshal(spec, &info)
		if err != nil {
			f.err = err
			return
		}
		f.document, f.err = openapi_v2.NewDocument(info, compiler.NewContext("$root", nil))
	})
	return f.document, f.err
}

/**
 * @brief Instrumented client for tracking schema retrieval invocations.
 * Functional Utility: Enables verification that client logic correctly caches the schema result.
 */
// FakeClient implements a dummy OpenAPISchemaInterface that uses the
// fake OpenAPI schema given as a parameter, and count the number of
// call to the function.
type FakeClient struct {
	Calls int
	Err   error

	fake *Fake
}

// NewFakeClient creates a new FakeClient from the given Fake.
func NewFakeClient(f *Fake) *FakeClient {
	return &FakeClient{fake: f}
}

/**
 * @brief Mocked schema retrieval with call-counting side effect.
 */
// OpenAPISchema returns a OpenAPI Document as returned by the fake, but
// it also counts the number of calls.
func (f *FakeClient) OpenAPISchema() (*openapi_v2.Document, error) {
	f.Calls = f.Calls + 1

	if f.Err != nil {
		return nil, f.Err
	}

	return f.fake.OpenAPISchema()
}

/**
 * @brief Simplified test wrapper for GVK-based resource lookup.
 */
// FakeResources is a wrapper to directly load the openapi schema from a
// file, and get the schema for given GVK. This is only for test since
// it's assuming that the file is there and everything will go fine.
type FakeResources struct {
	fake Fake
}

var _ openapi.Resources = &FakeResources{}

// NewFakeResources creates a new FakeResources.
func NewFakeResources(path string) *FakeResources {
	return &FakeResources{
		fake: Fake{Path: path},
	}
}

/**
 * @brief Performs a resource lookup based on GroupVersionKind.
 * Invariant: Panics on IO or parsing failure as this is strictly for controlled test environments.
 */
// LookupResource will read the schema, parse it and return the
// resources. It doesn't return errors and will panic instead.
func (f *FakeResources) LookupResource(gvk schema.GroupVersionKind) proto.Schema {
	s, err := f.fake.OpenAPISchema()
	if err != nil {
		panic(err)
	}
	resources, err := openapi.NewOpenAPIData(s)
	if err != nil {
		panic(err)
	}
	return resources.LookupResource(gvk)
}

/**
 * @brief Stub implementation of Resources that always returns nil.
 */
// EmptyResources implement a Resources that just doesn't have any resources.
type EmptyResources struct{}

var _ openapi.Resources = EmptyResources{}

// LookupResource will always return nil. It doesn't have any resources.
func (f EmptyResources) LookupResource(gvk schema.GroupVersionKind) proto.Schema {
	return nil
}

/**
 * @brief Closure factory for dynamic OpenAPI resource discovery in tests.
 */
// CreateOpenAPISchemaFunc returns a function useful for the TestFactory.
func CreateOpenAPISchemaFunc(path string) func() (openapi.Resources, error) {
	return func() (openapi.Resources, error) {
		return NewFakeResources(path), nil
	}
}
