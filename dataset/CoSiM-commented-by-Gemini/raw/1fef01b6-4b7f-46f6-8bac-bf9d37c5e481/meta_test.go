/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// Package api_test contains tests for the API helpers.
package api_test

import (
	"reflect"
	"testing"

	"github.com/google/gofuzz"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/meta/metatypes"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

// Verifies that ObjectMeta implements the meta.Object interface.
// This is a compile-time check.
var _ meta.Object = &api.ObjectMeta{}

// TestFillObjectMetaSystemFields validates that system-populated fields (UID, CreationTimestamp)
// are correctly set on an ObjectMeta instance. This is crucial for ensuring that objects
// receive unique and consistent system identifiers upon creation.
func TestFillObjectMetaSystemFields(t *testing.T) {
	ctx := api.NewDefaultContext()
	resource := api.ObjectMeta{}
	api.FillObjectMetaSystemFields(ctx, &resource)
	if resource.CreationTimestamp.Time.IsZero() {
		t.Errorf("resource.CreationTimestamp is zero")
	} else if len(resource.UID) == 0 {
		t.Errorf("resource.UID missing")
	}
}

// TestHasObjectMetaSystemFieldValues validates that the helper function correctly
// identifies whether system fields have been populated. This is a check for the inverse
// logic of TestFillObjectMetaSystemFields.
func TestHasObjectMetaSystemFieldValues(t *testing.T) {
	ctx := api.NewDefaultContext()
	resource := api.ObjectMeta{}
	// An empty resource should report that it does not have system values.
	if api.HasObjectMetaSystemFieldValues(&resource) {
		t.Errorf("the resource does not have all fields yet populated, but incorrectly reports it does")
	}
	// After filling, it should report that it has the values.
	api.FillObjectMetaSystemFields(ctx, &resource)
	if !api.HasObjectMetaSystemFieldValues(&resource) {
		t.Errorf("the resource does have all fields populated, but incorrectly reports it does not")
	}
}

// getObjectMetaAndOwnerReferences is a helper function for fuzz testing. It creates a
// randomized ObjectMeta instance and a corresponding slice of the more generic
// metatypes.OwnerReference, which is used as the expected result in tests.
func getObjectMetaAndOwnerReferences() (objectMeta api.ObjectMeta, metaOwnerReferences []metatypes.OwnerReference) {
	// Fuzz-populate an ObjectMeta struct.
	fuzz.New().NilChance(.5).NumElements(1, 5).Fuzz(&objectMeta)
	references := objectMeta.OwnerReferences
	metaOwnerReferences = make([]metatypes.OwnerReference, 0)
	// Manually convert the fuzzed api.OwnerReference to metatypes.OwnerReference.
	for i := 0; i < len(references); i++ {
		metaOwnerReferences = append(metaOwnerReferences, metatypes.OwnerReference{
			Kind:       references[i].Kind,
			Name:       references[i].Name,
			UID:        references[i].UID,
			APIVersion: references[i].APIVersion,
			Controller: references[i].Controller,
		})
	}
	// Ensure OwnerReferences is not nil, which can cause issues with DeepEqual.
	if len(references) == 0 {
		objectMeta.OwnerReferences = make([]api.OwnerReference, 0)
	}
	return objectMeta, metaOwnerReferences
}

// testGetOwnerReferences tests the GetOwnerReferences method.
func testGetOwnerReferences(t *testing.T) {
	meta, expected := getObjectMetaAndOwnerReferences()
	refs := meta.GetOwnerReferences()
	if !reflect.DeepEqual(refs, expected) {
		t.Errorf("expect %v
 got %v", expected, refs)
	}
}

// testSetOwnerReferences tests the SetOwnerReferences method.
func testSetOwnerReferences(t *testing.T) {
	expected, newRefs := getObjectMetaAndOwnerReferences()
	objectMeta := &api.ObjectMeta{}
	objectMeta.SetOwnerReferences(newRefs)
	if !reflect.DeepEqual(expected.OwnerReferences, objectMeta.OwnerReferences) {
		t.Errorf("expect: %#v
 got: %#v", expected.OwnerReferences, objectMeta.OwnerReferences)
	}
}

// TestAccessOwnerReferences orchestrates fuzz testing for both Get and Set
// methods for OwnerReferences to ensure they are symmetric and handle various
// randomized inputs correctly.
func TestAccessOwnerReferences(t *testing.T) {
	fuzzIter := 5
	for i := 0; i < fuzzIter; i++ {
		testGetOwnerReferences(t)
		testSetOwnerReferences(t)
	}
}

// TestAccessorImplementations is a conformance test that uses reflection to iterate
// over all known API types. It ensures that any type with an `ObjectMeta` or `ListMeta`
// field correctly implements the corresponding `ObjectMetaAccessor` or `ListMetaAccessor`
// interface. This is critical for maintaining the integrity and consistency of the
// Kubernetes API machinery, which relies on these interfaces for generic object manipulation.
func TestAccessorImplementations(t *testing.T) {
	// Iterate through all registered API groups and versions.
	for _, group := range testapi.Groups {
		for _, gv := range []unversioned.GroupVersion{*group.GroupVersion(), group.InternalGroupVersion()} {
			for kind, knownType := range api.Scheme.KnownTypes(gv) {
				value := reflect.New(knownType)
				obj := value.Interface()
				// All API objects must implement runtime.Object.
				if _, ok := obj.(runtime.Object); !ok {
					t.Errorf("%v (%v) does not implement runtime.Object", gv.WithKind(kind), knownType)
				}
				lm, isLM := obj.(meta.ListMetaAccessor)
				om, isOM := obj.(meta.ObjectMetaAccessor)

				// An object should not be both a List and a single Object.
				switch {
				case isLM && isOM:
					t.Errorf("%v (%v) implements ListMetaAccessor and ObjectMetaAccessor", gv.WithKind(kind), knownType)
					continue
				case isLM:
					// Verify the ListMetaAccessor works as expected.
					m := lm.GetListMeta()
					if m == nil {
						t.Errorf("%v (%v) returns nil ListMeta", gv.WithKind(kind), knownType)
						continue
					}
					m.SetResourceVersion("102030")
					if m.GetResourceVersion() != "102030" {
						t.Errorf("%v (%v) did not preserve resource version", gv.WithKind(kind), knownType)
						continue
					}
					m.SetSelfLink("102030")
					if m.GetSelfLink() != "102030" {
						t.Errorf("%v (%v) did not preserve self link", gv.WithKind(kind), knownType)
						continue
					}
				case isOM:
					// Verify the ObjectMetaAccessor works as expected.
					m := om.GetObjectMeta()
					if m == nil {
						t.Errorf("%v (%v) returns nil ObjectMeta", gv.WithKind(kind), knownType)
						continue
					}
					m.SetResourceVersion("102030")
					if m.GetResourceVersion() != "102030" {
						t.Errorf("%v (%v) did not preserve resource version", gv.WithKind(kind), knownType)
						continue
					}
					m.SetSelfLink("102030")
					if m.GetSelfLink() != "102030" {
						t.Errorf("%v (%v) did not preserve self link", gv.WithKind(kind), knownType)
						continue
					}
					labels := map[string]string{"a": "b"}
					m.SetLabels(labels)
					if !reflect.DeepEqual(m.GetLabels(), labels) {
						t.Errorf("%v (%v) did not preserve labels", gv.WithKind(kind), knownType)
						continue
					}
				default:
					// If it's not a List or Object, check if it *should* have been.
					if _, ok := obj.(unversioned.ListMetaAccessor); ok {
						continue
					}
					if _, ok := value.Elem().Type().FieldByName("ObjectMeta"); ok {
						t.Errorf("%v (%v) has ObjectMeta but does not implement ObjectMetaAccessor", gv.WithKind(kind), knownType)
						continue
					}
					if _, ok := value.Elem().Type().FieldByName("ListMeta"); ok {
						t.Errorf("%v (%v) has ListMeta but does not implement ListMetaAccessor", gv.WithKind(kind), knownType)
						continue
					}
					t.Logf("%v (%v) does not implement ListMetaAccessor or ObjectMetaAccessor", gv.WithKind(kind), knownType)
				}
			}
		}
	}
}
