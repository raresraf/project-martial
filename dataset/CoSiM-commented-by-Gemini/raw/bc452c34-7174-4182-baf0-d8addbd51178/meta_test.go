/**
 * @file meta_test.go
 * @brief Unit tests for the metadata helper functions.
 * @author The Kubernetes Authors
 *
 * @details
 * This file contains unit tests for the functions in `meta.go`. It includes a
 * compile-time check to ensure `api.ObjectMeta` implements the `meta.Object`
 * interface and provides test cases for the system field manipulation functions.
 */
/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package api_test

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
)

// This is a compile-time check to ensure that api.ObjectMeta satisfies the
// meta.Object interface. If the Get/Set methods on api.ObjectMeta were to
// change or be removed, this line would fail to compile, providing an early
// warning that the contract has been broken.
var _ meta.Object = &api.ObjectMeta{}

// TestFillObjectMetaSystemFields validates that system-populated fields
// (CreationTimestamp, UID) are correctly set on an ObjectMeta instance.
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

// TestHasObjectMetaSystemFieldValues validates that the function correctly
// reports whether system fields have been populated.
func TestHasObjectMetaSystemFieldValues(t *testing.T) {
	ctx := api.NewDefaultContext()
	resource := api.ObjectMeta{}
	// Pre-condition: Before filling, the function should return false.
	if api.HasObjectMetaSystemFieldValues(&resource) {
		t.Errorf("the resource does not have all fields yet populated, but incorrectly reports it does")
	}
	api.FillObjectMetaSystemFields(ctx, &resource)
	// Post-condition: After filling, the function should return true.
	if !api.HasObjectMetaSystemFieldValues(&resource) {
		t.Errorf("the resource does have all fields populated, but incorrectly reports it does not")
	}
}
