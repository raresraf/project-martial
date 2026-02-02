/**
 * @file openapi.go
 * @brief Provides fake implementations of OpenAPI schema-related interfaces for testing.
 * @details This file is crucial for testing components that depend on OpenAPI schemas
 * without requiring a live API server. It includes utilities to load schemas from
 * files and mock resource lookups.
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

// Fake implements a mock for fetching an OpenAPI schema from a local file path.
// It uses sync.Once to ensure the file is read and parsed only once.
type Fake struct {
	Path string // Path to the OpenAPI swagger file.

	once     sync.Once
	document *openapi_v2.Document
	err      error
}

// OpenAPISchema returns the OpenAPI document. It reads and parses the document
// from the file path on the first call and caches the result for subsequent calls.
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

// FakeResources provides an implementation of openapi.Resources that uses a
// local OpenAPI schema file for resource lookups.
type FakeResources struct {
	fake Fake
}

var _ openapi.Resources = &FakeResources{}

// NewFakeResources creates a new FakeResources instance for a given file path.
func NewFakeResources(path string) *FakeResources {
	return &FakeResources{
		fake: Fake{Path: path},
	}
}

// LookupResource finds the schema for a given GroupVersionKind from the
// OpenAPI document loaded from the file. It will panic if any error occurs
// during schema loading or parsing.
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

// EmptyResources implements the openapi.Resources interface but contains no resources.
type EmptyResources struct{}

var _ openapi.Resources = EmptyResources{}

// LookupResource for EmptyResources always returns nil.
func (f EmptyResources) LookupResource(gvk schema.GroupVersionKind) proto.Schema {
	return nil
}

// CreateOpenAPISchemaFunc is a helper function that returns a factory function
// for creating openapi.Resources. This is useful for dependency injection in tests.
func CreateOpenAPISchemaFunc(path string) func() (openapi.Resources, error) {
	return func() (openapi.Resources, error) {
		return NewFakeResources(path), nil
	}
}