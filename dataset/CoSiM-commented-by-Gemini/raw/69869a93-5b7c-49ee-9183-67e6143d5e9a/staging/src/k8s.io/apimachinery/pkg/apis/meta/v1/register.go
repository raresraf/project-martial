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

// Package v1 contains the v1 version of the API Machinery's meta-data interfaces.
//
// Architectural Intent: This package is a cornerstone of the Kubernetes API structure.
// It defines common metadata and options types (e.g., `ObjectMeta`, `ListOptions`, `Status`)
// that are embedded or used by almost all other versioned API objects in the system.
// The `register.go` file, specifically, is responsible for making these types known to the
// `runtime.Scheme`, which is the core registry for API types. This registration enables
// serialization, deserialization, and conversion, allowing clients and servers to
// understand and translate these fundamental objects.
package v1

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

// GroupName is the group name for this API.
const GroupName = "meta.k8s.io"

// SchemeGroupVersion is group version used to register these objects.
// It represents the unique identifier for the `meta/v1` API group.
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v1"}

// Unversioned is group version for unversioned API objects.
// These are types that are not specific to a particular API version, like Status.
// TODO: this should be v1 probably
var Unversioned = schema.GroupVersion{Group: "", Version: "v1"}

// WatchEventKind is name reserved for serializing watch events.
const WatchEventKind = "WatchEvent"

// Kind takes an unqualified kind and returns a Group qualified GroupKind.
// This helper function is used to construct a fully-qualified kind identifier
// for objects within this API group.
func Kind(kind string) schema.GroupKind {
	return SchemeGroupVersion.WithKind(kind).GroupKind()
}

// scheme is the registry for the common types that adhere to the meta v1 API spec.
// It is a local scheme builder used within this package.
var scheme = runtime.NewScheme()

// ParameterCodec knows about query parameters used with the meta v1 API spec.
// It is used for encoding and decoding API options (like GetOptions, ListOptions)
// from URL query parameters.
var ParameterCodec = runtime.NewParameterCodec(scheme)

// addEventConversionFuncs registers conversion functions for WatchEvent types.
// This is necessary to handle transformations between internal and versioned
// representations of watch events.
func addEventConversionFuncs(scheme *runtime.Scheme) error {
	return scheme.AddConversionFuncs(
		Convert_v1_WatchEvent_To_watch_Event,
		Convert_v1_InternalEvent_To_v1_WatchEvent,
		Convert_watch_Event_To_v1_WatchEvent,
		Convert_v1_WatchEvent_To_v1_InternalEvent,
	)
}

var optionsTypes = []runtime.Object{
	&ListOptions{},
	&ExportOptions{},
	&GetOptions{},
	&DeleteOptions{},
	&CreateOptions{},
	&UpdateOptions{},
	&PatchOptions{},
}

// AddToGroupVersion registers common meta types into a scheme.
// This function is the primary mechanism by which other API groups (e.g., `core/v1`, `apps/v1`)
// can incorporate the standard meta types into their own schemes, ensuring that they
// can handle objects like `Status` and `ListOptions` correctly.
func AddToGroupVersion(scheme *runtime.Scheme, groupVersion schema.GroupVersion) {
	// Pre-condition: `scheme` is a valid runtime.Scheme instance.
	// `groupVersion` is the identity of the target group where meta types will be registered.
	scheme.AddKnownTypeWithName(groupVersion.WithKind(WatchEventKind), &WatchEvent{})
	scheme.AddKnownTypeWithName(
		schema.GroupVersion{Group: groupVersion.Group, Version: runtime.APIVersionInternal}.WithKind(WatchEventKind),
		&InternalEvent{},
	)
	// Supports legacy code paths, most callers should use metav1.ParameterCodec for now
	scheme.AddKnownTypes(groupVersion, optionsTypes...)
	// Register Unversioned types under their own special group.
	// These types are considered fundamental and not tied to a specific API version.
	scheme.AddUnversionedTypes(Unversioned,
		&Status{},
		&APIVersions{},
		&APIGroupList{},
		&APIGroup{},
		&APIResourceList{},
	)

	utilruntime.Must(addEventConversionFuncs(scheme))

	// register manually. This usually goes through the SchemeBuilder, which we cannot use here.
	utilruntime.Must(AddConversionFuncs(scheme))
	utilruntime.Must(RegisterDefaults(scheme))
}

// AddMetaToScheme registers the API types that are specific to this meta group.
// This includes types like `Table` for server-side printing and `PartialObjectMetadata`
// for efficient metadata-only fetches.
func AddMetaToScheme(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&Table{},
		&TableOptions{},
		&PartialObjectMetadata{},
		&PartialObjectMetadataList{},
	)

	return scheme.AddConversionFuncs(
		Convert_Slice_string_To_v1_IncludeObjectPolicy,
	)
}

// init performs the registration of meta types into the package-local scheme.
// This ensures that the types are available for use as soon as this package is imported.
func init() {
	scheme.AddUnversionedTypes(SchemeGroupVersion, optionsTypes...)

	utilruntime.Must(AddMetaToScheme(scheme))

	// register manually. This usually goes through the SchemeBuilder, which we cannot use here.
	utilruntime.Must(RegisterDefaults(scheme))
}
