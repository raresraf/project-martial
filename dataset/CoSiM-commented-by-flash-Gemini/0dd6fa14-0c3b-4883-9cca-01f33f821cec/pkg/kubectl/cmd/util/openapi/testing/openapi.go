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

// Package testing provides fake implementations for OpenAPI schema loading.
// These fakes are designed for testing purposes within the kubectl utilities,
// allowing for simulation of OpenAPI data without requiring a live Kubernetes API server.
// Domain: Kubernetes, OpenAPI, Testing, Mocking.
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

// Fake opens and returns a openapi swagger from a file Path. It will
// parse only once and then return the same copy everytime.
// Fake is a mock implementation of the openapi.Getter interface for testing.
// It loads an OpenAPI Swagger document from a specified file path once and
// then consistently returns that parsed document or any error encountered during parsing.
type Fake struct {
	// Path to the OpenAPI Swagger file to load.
	Path string

	// once ensures the OpenAPI document is parsed only a single time.
	once     sync.Once
	// document stores the parsed OpenAPIv2.Document.
	document *openapi_v2.Document
	// err stores any error encountered during document parsing.
	err      error
}

// OpenAPISchema returns the openapi document and a potential error.
// OpenAPISchema loads and parses the OpenAPI document from the configured path.
// Functional Utility: This method ensures that the OpenAPI document is loaded and parsed
//                     only once, caching the result for subsequent calls. It handles
//                     file reading, YAML unmarshalling, and OpenAPI document creation.
// Returns: The parsed OpenAPIv2.Document and any error encountered during the process.
func (f *Fake) OpenAPISchema() (*openapi_v2.Document, error) {
	// Block Logic: Ensures that the expensive OpenAPI document parsing operation is performed only once.
	f.once.Do(func() {
		// Block Logic: Checks if the specified file path exists.
		if _, err := os.Stat(f.Path); err != nil {
			f.err = err
			return // Pre-condition: File must exist.
		}
		// Block Logic: Reads the content of the OpenAPI specification file.
		spec, err := ioutil.ReadFile(f.Path)
		if err != nil {
			f.err = err
			return // Pre-condition: File must be readable.
		}
		// Block Logic: Unmarshals the file content (expected to be YAML) into a generic map structure.
		var info yaml.MapSlice
		if err = yaml.Unmarshal(spec, &info); err != nil {
			f.err = err
			return // Pre-condition: File content must be valid YAML.
		}
		// Block Logic: Creates an OpenAPIv2.Document from the unmarshaled data.
		f.document, f.err = openapi_v2.NewDocument(info, compiler.NewContext("$root", nil))
		// Post-condition: `f.document` contains the parsed OpenAPI schema, and `f.err` holds any parsing errors.
	})
	return f.document, f.err
}

// FakeResources is a wrapper to directly load the openapi schema from a
// file, and get the schema for given GVK. This is only for test since
// it's assuming that the file is there and everything will go fine.
// FakeResources is a mock implementation of openapi.Resources for testing.
// It wraps a Fake OpenAPI getter to provide resource lookup capabilities.
// FakeResources is a mock implementation of openapi.Resources for testing.
// It wraps a Fake OpenAPI getter to provide resource lookup capabilities.
type FakeResources struct {
	fake Fake
}

var _ openapi.Resources = &FakeResources{}

// NewFakeResources creates a new FakeResources.
// NewFakeResources creates a new FakeResources instance, initializing it with the path to an OpenAPI spec file.
// Functional Utility: Provides a convenient way to create a testable OpenAPI resource provider.
// Parameters:
//   path: The file path to the OpenAPI specification.
// Returns: A pointer to a new FakeResources instance.
// NewFakeResources creates a new FakeResources instance, initializing it with the path to an OpenAPI spec file.
// Functional Utility: Provides a convenient way to create a testable OpenAPI resource provider.
// Parameters:
//   path: The file path to the OpenAPI specification.
// Returns: A pointer to a new FakeResources instance.
func NewFakeResources(path string) *FakeResources {
	return &FakeResources{
		fake: Fake{Path: path},
	}
}

// LookupResource attempts to find the schema for a given GroupVersionKind.
// Functional Utility: This method loads the OpenAPI schema (if not already loaded)
//                     and then delegates to the real openapi.Resources to look up
//                     the specified resource schema. For testing purposes, it will
//                     panic on any errors during schema loading or data conversion.
// Parameters:
//   gvk: The GroupVersionKind of the resource to look up.
// Returns: A proto.Schema representing the requested resource, or nil if not found.
func (f *FakeResources) LookupResource(gvk schema.GroupVersionKind) proto.Schema {
	// Block Logic: Loads the OpenAPI schema; will use the cached schema if already loaded.
	s, err := f.fake.OpenAPISchema()
	if err != nil {
		panic(err) // Functional Utility: Panics on error for simplicity in testing context.
	}
	// Block Logic: Converts the loaded OpenAPI document into an openapi.Resources object.
	resources, err := openapi.NewOpenAPIData(s)
	if err != nil {
		panic(err) // Functional Utility: Panics on error for simplicity in testing context.
	}
	// Block Logic: Delegates the actual resource lookup to the created openapi.Resources instance.
	return resources.LookupResource(gvk)
}

// EmptyResources implement a Resources that just doesn't have any resources.
// EmptyResources implements openapi.Resources but contains no resources.
// Functional Utility: Useful for testing scenarios where no OpenAPI resources are expected.
type EmptyResources struct{}

var _ openapi.Resources = EmptyResources{}

// LookupResource will always return nil. It doesn't have any resources.
// LookupResource always returns nil, indicating no resource schema is found.
// Functional Utility: Provides a predictable "not found" response for any resource lookup.
// Parameters:
//   gvk: The GroupVersionKind of the resource to look up (ignored).
// Returns: Always returns nil.
func (f EmptyResources) LookupResource(gvk schema.GroupVersionKind) proto.Schema {
	return nil
}

// CreateOpenAPISchemaFunc returns a function useful for the TestFactory.
// CreateOpenAPISchemaFunc returns a function that can be used to create a FakeResources instance.
// Functional Utility: This factory function simplifies the creation of a function suitable for
//                     passing to components that expect an OpenAPI schema provider.
// Parameters:
//   path: The file path to the OpenAPI specification.
// Returns: A function that, when called, returns a new FakeResources instance and a nil error.
func CreateOpenAPISchemaFunc(path string) func() (openapi.Resources, error) {
	return func() (openapi.Resources, error) {
		return NewFakeResources(path), nil
	}
}
