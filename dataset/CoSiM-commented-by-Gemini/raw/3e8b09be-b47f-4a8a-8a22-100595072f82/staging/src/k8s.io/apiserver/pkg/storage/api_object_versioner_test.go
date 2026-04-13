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

package storage

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
)

// TestObjectVersioner tests the basic functions of the APIObjectVersioner.
// It verifies that the versioner can correctly get a numeric version from an object,
// handle parsing errors for invalid versions, and set a numeric version back
// onto an object.
func TestObjectVersioner(t *testing.T) {
	v := APIObjectVersioner{}

	// Test getting a valid resource version.
	if ver, err := v.ObjectResourceVersion(&storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "5"}}); err != nil || ver != 5 {
		t.Errorf("unexpected version: %d %v", ver, err)
	}

	// Test getting an invalid resource version.
	if ver, err := v.ObjectResourceVersion(&storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "a"}}); err == nil || ver != 0 {
		t.Errorf("unexpected version: %d %v", ver, err)
	}

	// Test updating an object's resource version.
	obj := &storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "a"}}
	if err := v.UpdateObject(obj, 5); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if obj.ResourceVersion != "5" || obj.DeletionTimestamp != nil {
		t.Errorf("unexpected resource version: %#v", obj)
	}
}

// TestEtcdParseResourceVersion provides a table-driven test for the
// ParseResourceVersion method. It ensures that various string inputs (valid,
// invalid, empty) are parsed into the correct uint64 representation or return
// an appropriate error. This is crucial for interpreting the resourceVersion
// provided by etcd.
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
			switch {
			case testCase.Err && err == nil:
				t.Errorf("%s[%v]: unexpected non-error", testCase.Version, i)
			case testCase.Err && !IsInvalidError(err):
				t.Errorf("%s[%v]: unexpected error: %v", testCase.Version, i, err)
			case !testCase.Err && err != nil:
				t.Errorf("%s[%v]: unexpected error: %v", testCase.Version, i, err)
			}
			if version != testCase.ExpectVersion {
				t.Errorf("%s[%v]: expected version %d but was %d", testCase.Version, i, testCase.ExpectVersion, version)
			}
		}
	}
}

// TestCompareResourceVersion verifies the comparison logic of the versioner.
// This is essential for optimistic concurrency control, as the storage layer
// needs to determine if an object has been updated since it was last read.
func TestCompareResourceVersion(t *testing.T) {
	five := &storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "5"}}
	six := &storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "6"}}

	versioner := APIObjectVersioner{}

	// five < six, expect -1
	if e, a := -1, versioner.CompareResourceVersion(five, six); e != a {
		t.Errorf("expected %v got %v", e, a)
	}
	// six > five, expect 1
	if e, a := 1, versioner.CompareResourceVersion(six, five); e != a {
		t.Errorf("expected %v got %v", e, a)
	}
	// six == six, expect 0
	if e, a := 0, versioner.CompareResourceVersion(six, six); e != a {
		t.Errorf("expected %v got %v", e, a)
	}
}
