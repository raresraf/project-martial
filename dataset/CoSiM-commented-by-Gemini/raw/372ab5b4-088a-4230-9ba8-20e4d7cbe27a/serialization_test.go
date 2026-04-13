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

// This file contains a suite of tests designed to verify the serialization and
// deserialization (round-tripping) of Kubernetes API objects. It ensures that
// objects can be encoded and decoded without data loss or corruption, which is
// fundamental to the functioning of the Kubernetes API server.
package api_test

import (
	"encoding/json"
	"math/rand"
	"reflect"
	"testing"

	"github.com/davecgh/go-spew/spew"
	flag "github.com/spf13/pflag"
	"github.com/ugorji/go/codec"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
)

// fuzzIters controls the number of fuzzing iterations for each API object.
// Fuzzing populates objects with random data to test edge cases.
var fuzzIters = flag.Int("fuzz-iters", 20, "How many fuzzing iterations to do.")

var codecsToTest = []func(version unversioned.GroupVersion, item runtime.Object) (runtime.Codec, error){
	func(version unversioned.GroupVersion, item runtime.Object) (runtime.Codec, error) {
		return testapi.GetCodecForObject(item)
	},
}

// fuzzInternalObject uses a custom fuzzer to fill a given runtime.Object with random data.
// It also clears the Kind and APIVersion fields to ensure that the object represents a
// "pure" internal state before being encoded into a versioned representation.
func fuzzInternalObject(t *testing.T, forVersion unversioned.GroupVersion, item runtime.Object, seed int64) runtime.Object {
	apitesting.FuzzerFor(t, forVersion, rand.NewSource(seed)).Fuzz(item)

	j, err := meta.TypeAccessor(item)
	if err != nil {
		t.Fatalf("Unexpected error %v for %#v", err, item)
	}
	j.SetKind("")
	j.SetAPIVersion("")

	return item
}

// roundTrip performs a serialization (encode) and deserialization (decode) cycle
// on a runtime.Object and verifies that the resulting object is semantically
// identical to the original.
func roundTrip(t *testing.T, codec runtime.Codec, item runtime.Object) {
	printer := spew.ConfigState{DisableMethods: true}

	name := reflect.TypeOf(item).Elem().Name()
	data, err := runtime.Encode(codec, item)
	if err != nil {
		t.Errorf("%v: %v (%s)", name, err, printer.Sprintf("%#v", item))
		return
	}

	// 1. Decode into a new object.
	obj2, err := runtime.Decode(codec, data)
	if err != nil {
		t.Errorf("0: %v: %v
Codec: %v
Data: %s
Source: %#v", name, err, codec, string(data), printer.Sprintf("%#v", item))
		return
	}
	if !api.Semantic.DeepEqual(item, obj2) {
		t.Errorf("
1: %v: diff: %v
Codec: %v
Source:

%#v

Encoded:

%s

Final:

%#v", name, util.ObjectGoPrintDiff(item, obj2), codec, printer.Sprintf("%#v", item), string(data), printer.Sprintf("%#v", obj2))
		return
	}

	// 2. Decode into a pre-allocated object.
	obj3 := reflect.New(reflect.TypeOf(item).Elem()).Interface().(runtime.Object)
	if err := runtime.DecodeInto(codec, data, obj3); err != nil {
		t.Errorf("2: %v: %v", name, err)
		return
	}
	if !api.Semantic.DeepEqual(item, obj3) {
		t.Errorf("3: %v: diff: %v
Codec: %v", name, util.ObjectDiff(item, obj3), codec)
		return
	}
}

// roundTripSame verifies that a fuzzed object can be round-tripped correctly
// through the codecs of its associated API group version.
func roundTripSame(t *testing.T, group testapi.TestGroup, item runtime.Object, except ...string) {
	set := sets.NewString(except...)
	// Use a consistent seed for fuzzing to ensure reproducibility.
	seed := rand.Int63()
	fuzzInternalObject(t, group.InternalGroupVersion(), item, seed)

	version := *group.GroupVersion()
	codecs := []runtime.Codec{}
	for _, fn := range codecsToTest {
		codec, err := fn(version, item)
		if err != nil {
			t.Errorf("unable to get codec: %v", err)
			return
		}
		codecs = append(codecs, codec)
	}

	if !set.Has(version.String()) {
		// Fuzz the object again with the same seed to ensure it has the same state
		// before being passed to the roundTrip function.
		fuzzInternalObject(t, version, item, seed)
		for _, codec := range codecs {
			roundTrip(t, codec, item)
		}
	}
}

// TestSpecificKind is a helper for debugging serialization issues with a single API Kind.
// Developers can change the "kind" variable to focus on a specific object type.
func TestSpecificKind(t *testing.T) {
	kind := "DaemonSet"
	for i := 0; i < *fuzzIters; i++ {
		doRoundTripTest(testapi.Groups["extensions"], kind, t)
		if t.Failed() {
			break
		}
	}
}

// TestList specifically tests the round-tripping of the generic `List` object.
func TestList(t *testing.T) {
	kind := "List"
	item, err := api.Scheme.New(api.SchemeGroupVersion.WithKind(kind))
	if err != nil {
		t.Errorf("Couldn't make a %v? %v", kind, err)
		return
	}
	roundTripSame(t, testapi.Default, item)
}

// nonRoundTrippableTypes is a set of kinds that are not expected to round-trip correctly.
// These are often transient types used for actions, not for storage (e.g., ExportOptions).
var nonRoundTrippableTypes = sets.NewString("ExportOptions")
var nonInternalRoundTrippableTypes = sets.NewString("List", "ListOptions", "ExportOptions")
var nonRoundTrippableTypesByVersion = map[string][]string{}

// TestRoundTripTypes is the main test function that iterates through all registered
// API groups and kinds, performing a round-trip test for each one.
func TestRoundTripTypes(t *testing.T) {
	for groupKey, group := range testapi.Groups {
		for kind := range group.InternalTypes() {
			t.Logf("working on %v in %v", kind, groupKey)
			// Skip types that are known to not be round-trippable.
			if nonRoundTrippableTypes.Has(kind) {
				continue
			}
			// Run the test multiple times with different random data.
			for i := 0; i < *fuzzIters; i++ {
				doRoundTripTest(group, kind, t)
				if t.Failed() {
					break
				}
			}
		}
	}
}

// doRoundTripTest is a helper that creates a new object of a given kind,
// gets its type accessor, and runs the round-trip tests.
func doRoundTripTest(group testapi.TestGroup, kind string, t *testing.T) {
	// Create a new instance of the object using the scheme.
	item, err := api.Scheme.New(group.InternalGroupVersion().WithKind(kind))
	if err != nil {
		t.Fatalf("Couldn't make a %v? %v", kind, err)
	}
	if _, err := meta.TypeAccessor(item); err != nil {
		t.Fatalf("%q is not a TypeMeta and cannot be tested - add it to nonRoundTrippableTypes: %v", kind, err)
	}
	if api.Scheme.Recognizes(group.GroupVersion().WithKind(kind)) {
		roundTripSame(t, group, item, nonRoundTrippableTypesByVersion[kind]...)
	}
	// Also test round-tripping to the internal version if applicable.
	if !nonInternalRoundTrippableTypes.Has(kind) && api.Scheme.Recognizes(group.GroupVersion().WithKind(kind)) {
		roundTrip(t, group.Codec(), fuzzInternalObject(t, group.InternalGroupVersion(), item, rand.Int63()))
	}
}

// TestEncode_Ptr verifies that fields which are pointers to primitive types
// (e.g., *int64) are correctly serialized and deserialized.
func TestEncode_Ptr(t *testing.T) {
	grace := int64(30)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels: map[string]string{"name": "foo"},
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
			// This field is a pointer and is the main subject of this test.
			TerminationGracePeriodSeconds: &grace,
			SecurityContext:               &api.PodSecurityContext{},
		},
	}
	obj := runtime.Object(pod)
	data, err := runtime.Encode(testapi.Default.Codec(), obj)
	obj2, err2 := runtime.Decode(testapi.Default.Codec(), data)
	if err != nil || err2 != nil {
		t.Fatalf("Failure: '%v' '%v'", err, err2)
	}
	if _, ok := obj2.(*api.Pod); !ok {
		t.Fatalf("Got wrong type")
	}
	if !api.Semantic.DeepEqual(obj2, pod) {
		t.Errorf("
Expected:

 %#v,

Got:

 %#vDiff: %v

", pod, obj2, util.ObjectDiff(obj2, pod))
	}
}

// TestBadJSONRejection ensures that the decoder correctly fails when given
// invalid or incomplete JSON data.
func TestBadJSONRejection(t *testing.T) {
	// An object must have a 'kind' field.
	badJSONMissingKind := []byte(`{ }`)
	if _, err := runtime.Decode(testapi.Default.Codec(), badJSONMissingKind); err == nil {
		t.Errorf("Did not reject despite lack of kind field: %s", badJSONMissingKind)
	}
	// The 'kind' must be known to the scheme.
	badJSONUnknownType := []byte(`{"kind": "bar"}`)
	if _, err1 := runtime.Decode(testapi.Default.Codec(), badJSONUnknownType); err1 == nil {
		t.Errorf("Did not reject despite use of unknown type: %s", badJSONUnknownType)
	}
}

// TestUnversionedTypes checks that types without a version (like Status) can be
// encoded and decoded correctly by versioned codecs.
func TestUnversionedTypes(t *testing.T) {
	testcases := []runtime.Object{
		&unversioned.Status{Status: "Failure", Message: "something went wrong"},
		&unversioned.APIVersions{Versions: []string{"A", "B", "C"}},
		&unversioned.APIGroupList{Groups: []unversioned.APIGroup{{Name: "mygroup"}}},
		&unversioned.APIGroup{Name: "mygroup"},
		&unversioned.APIResourceList{GroupVersion: "mygroup/myversion"},
	}

	for _, obj := range testcases {
		// Make sure the unversioned codec can encode
		unversionedJSON, err := runtime.Encode(testapi.Default.Codec(), obj)
		if err != nil {
			t.Errorf("%v: unexpected error: %v", obj, err)
			continue
		}

		// Make sure the versioned codec under test can decode the unversioned type
		versionDecodedObject, err := runtime.Decode(testapi.Default.Codec(), unversionedJSON)
		if err != nil {
			t.Errorf("%v: unexpected error: %v", obj, err)
			continue
		}
		// Make sure it decodes correctly
		if !reflect.DeepEqual(obj, versionDecodedObject) {
			t.Errorf("%v: expected %#v, got %#v", obj, obj, versionDecodedObject)
			continue
		}
	}
}

const benchmarkSeed = 100

// benchmarkItems creates a set of fuzzed Pod objects for use in benchmarks.
func benchmarkItems() []v1.Pod {
	apiObjectFuzzer := apitesting.FuzzerFor(nil, api.SchemeGroupVersion, rand.NewSource(benchmarkSeed))
	items := make([]v1.Pod, 2)
	for i := range items {
		apiObjectFuzzer.Fuzz(&items[i])
	}
	return items
}

// BenchmarkEncodeCodec measures the performance of the standard Kubernetes runtime codec for encoding.
func BenchmarkEncodeCodec(b *testing.B) {
	items := benchmarkItems()
	width := len(items)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := runtime.Encode(testapi.Default.Codec(), &items[i%width]); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkEncodeJSONMarshal provides a baseline for encoding performance using the standard Go `encoding/json` library.
func BenchmarkEncodeJSONMarshal(b *testing.B) {
	items := benchmarkItems()
	width := len(items)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := json.Marshal(&items[i%width]); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkDecodeCodec measures the performance of the standard Kubernetes runtime codec for decoding.
func BenchmarkDecodeCodec(b *testing.B) {
	codec := testapi.Default.Codec()
	items := benchmarkItems()
	width := len(items)
	encoded := make([][]byte, width)
	for i := range items {
		data, err := runtime.Encode(codec, &items[i])
		if err != nil {
			b.Fatal(err)
		}
		encoded[i] = data
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := runtime.Decode(codec, encoded[i%width]); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkDecodeIntoExternalCodec measures decoding into a versioned (external) object.
func BenchmarkDecodeIntoExternalCodec(b *testing.B) {
	codec := testapi.Default.Codec()
	items := benchmarkItems()
	width := len(items)
	encoded := make([][]byte, width)
	for i := range items {
		data, err := runtime.Encode(codec, &items[i])
		if err != nil {
			b.Fatal(err)
		}
		encoded[i] = data
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		obj := v1.Pod{}
		if err := runtime.DecodeInto(codec, encoded[i%width], &obj); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkDecodeIntoInternalCodec measures decoding into an unversioned (internal) object.
func BenchmarkDecodeIntoInternalCodec(b *testing.B) {
	codec := testapi.Default.Codec()
	items := benchmarkItems()
	width := len(items)
	encoded := make([][]byte, width)
	for i := range items {
		data, err := runtime.Encode(codec, &items[i])
		if err != nil {
			b.Fatal(err)
		}
		encoded[i] = data
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		obj := api.Pod{}
		if err := runtime.DecodeInto(codec, encoded[i%width], &obj); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkDecodeIntoJSON provides a baseline for decoding performance using the standard Go `encoding/json` library.
func BenchmarkDecodeIntoJSON(b *testing.B) {
	codec := testapi.Default.Codec()
	items := benchmarkItems()
	width := len(items)
	encoded := make([][]byte, width)
	for i := range items {
		data, err := runtime.Encode(codec, &items[i])
		if err != nil {
			b.Fatal(err)
		}
		encoded[i] = data
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		obj := v1.Pod{}
		if err := json.Unmarshal(encoded[i%width], &obj); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkDecodeIntoJSONCodecGen provides a baseline for decoding using the high-performance `ugorji/go/codec` library.
func BenchmarkDecodeIntoJSONCodecGen(b *testing.B) {
	kcodec := testapi.Default.Codec()
	items := benchmarkItems()
	width := len(items)
	encoded := make([][]byte, width)
	for i := range items {
		data, err := runtime.Encode(kcodec, &items[i])
		if err != nil {
			b.Fatal(err)
		}
		encoded[i] = data
	}
	handler := &codec.JsonHandle{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		obj := v1.Pod{}
		if err := codec.NewDecoderBytes(encoded[i%width], handler).Decode(&obj); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}
