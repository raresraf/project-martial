/*
Copyright 2014 The Kubernetes Authors.

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

/**
 * @file api_object_versioner_test.go
 * @brief Unit tests for API object resource version management.
 * 
 * Functional Intent: Validates the logic for extracting, parsing, and updating 
 * the 'ResourceVersion' field in Kubernetes API objects. Ensures compatibility 
 * with various version formats and maintains consistency during concurrent 
 * state transitions.
 * 
 * Domain: Production Systems, Cloud Infrastructure, Distributed State Management.
 */

package storage

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
)

/**
 * TestObjectVersioner - Validates resource version extraction and updates on live objects.
 * 
 * Logic:
 * 1. Verifies numeric parsing from string-based resource versions.
 * 2. Checks error handling for non-numeric version strings.
 * 3. Validates in-place object updates with new version numbers.
 */
func TestObjectVersioner(t *testing.T) {
	v := APIObjectVersioner{}
	
	// Block Logic: Positive version extraction.
	if ver, err := v.ObjectResourceVersion(&storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "5"}}); err != nil || ver != 5 {
		t.Errorf("unexpected version: %d %v", ver, err)
	}
	
	// Block Logic: Negative parsing (invalid format).
	if ver, err := v.ObjectResourceVersion(&storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "a"}}); err == nil || ver != 0 {
		t.Errorf("unexpected version: %d %v", ver, err)
	}
	
	// Block Logic: In-place state mutation.
	obj := &storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "a"}}
	if err := v.UpdateObject(obj, 5); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	
	// Invariant: Object metadata must reflect the new version without side effects on other fields.
	if obj.ResourceVersion != "5" || obj.DeletionTimestamp != nil {
		t.Errorf("unexpected resource version: %#v", obj)
	}
}

/**
 * TestEtcdParseResourceVersion - Exhaustive validation of version string parsing logic.
 * 
 * Algorithm: Table-driven testing for edge cases and boundary conditions.
 */
func TestEtcdParseResourceVersion(t *testing.T) {
	testCases := []struct {
		Version       string
		ExpectVersion uint64
		Err           bool
	}{
		{Version: "", ExpectVersion: 0},
		{Version: "a", Err: true},
		{Version: " ", Err: true},
		{Version: "1", ExpectVersion: 1},
		{Version: "10", ExpectVersion: 10},
	}

	v := APIObjectVersioner{}
	testFuncs := []func(string) (uint64, error){
		v.ParseResourceVersion,
	}

	for _, testCase := range testCases {
		for i, f := range testFuncs {
			version, err := f(testCase.Version)
			
			// Block Logic: Error state validation.
			switch {
			case testCase.Err && err == nil:
				t.Errorf("%s[%v]: unexpected non-error", testCase.Version, i)
			case testCase.Err && !IsInvalidError(err):
				t.Errorf("%s[%v]: unexpected error: %v", testCase.Version, i, err)
			case !testCase.Err && err != nil:
				t.Errorf("%s[%v]: unexpected error: %v", testCase.Version, i, err)
			}
			
			// Invariant: Parsed numeric value must match expected baseline.
			if version != testCase.ExpectVersion {
				t.Errorf("%s[%v]: expected version %d but was %d", testCase.Version, i, testCase.ExpectVersion, version)
			}
		}
	}
}

/**
 * TestCompareResourceVersion - Validates the ordering logic for resource versions.
 * 
 * Functional Utility: Crucial for resolving write conflicts and determining the 
 * latest state in distributed storage.
 */
func TestCompareResourceVersion(t *testing.T) {
	five := &storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "5"}}
	six := &storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "6"}}

	versioner := APIObjectVersioner{}

	// Block Logic: Sequential comparison checks.
	if e, a := -1, versioner.CompareResourceVersion(five, six); e != a {
		t.Errorf("expected %v got %v", e, a)
	}
	if e, a := 1, versioner.CompareResourceVersion(six, five); e != a {
		t.Errorf("expected %v got %v", e, a)
	}
	if e, a := 0, versioner.CompareResourceVersion(six, six); e != a {
		t.Errorf("expected %v got %v", e, a)
	}
}
