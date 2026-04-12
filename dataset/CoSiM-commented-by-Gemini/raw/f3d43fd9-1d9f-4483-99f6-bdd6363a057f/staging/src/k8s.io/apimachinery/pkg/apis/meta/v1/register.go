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

package v1

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

// GroupName is the group name for this API.
const GroupName = "meta.k8s.io"

// SchemeGroupVersion is the group version used to register these objects.
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v1"}

// Unversioned is the group version for unversioned API objects.
// TODO: this should be v1 probably.
var Unversioned = schema.GroupVersion{Group: "", Version: "v1"}

// WatchEventKind is a name reserved for serializing watch events.
const WatchEventKind = "WatchEvent"

// Kind takes an unqualified kind and returns a Group qualified GroupKind.
func Kind(kind string) schema.GroupKind {
	return SchemeGroupVersion.WithKind(kind).GroupKind()
}

// AddToGroupVersion registers common meta types into a scheme.
func AddToGroupVersion(scheme *runtime.Scheme, groupVersion schema.GroupVersion) {
	// Adds a known type for watch events, which are fundamental to the Kubernetes watch mechanism.
	scheme.AddKnownTypeWithName(groupVersion.WithKind(WatchEventKind), &WatchEvent{})
	// Adds an internal version of the watch event for internal-to-external conversions.
	scheme.AddKnownTypeWithName(
		schema.GroupVersion{Group: groupVersion.Group, Version: runtime.APIVersionInternal}.WithKind(WatchEventKind),
		&InternalEvent{},
	)
	// Registers common option types used across different API operations (list, get, delete, etc.).
	// This ensures that they can be correctly handled by the API server.
	scheme.AddKnownTypes(groupVersion,
		&ListOptions{},
		&ExportOptions{},
		&GetOptions{},
		&DeleteOptions{},
		&CreateOptions{},
		&UpdateOptions{},
		&PatchOptions{},
	)
	// Registers conversion functions for watch events between different versions (internal and v1).
	utilruntime.Must(scheme.AddConversionFuncs(
		Convert_v1_WatchEvent_To_watch_Event,
		Convert_v1_InternalEvent_To_v1_WatchEvent,
		Convert_watch_Event_To_v1_WatchEvent,
		Convert_v1_WatchEvent_To_v1_InternalEvent,
	))
	// Registers unversioned types (like Status and APIGroup) under a special group.
	// These types are used across different API groups and versions.
	scheme.AddUnversionedTypes(Unversioned,
		&Status{},
		&APIVersions{},
		&APIGroupList{},
		&APIGroup{},
		&APIResourceList{},
	)

	// Manually register conversion and default functions. This is typically handled
	// by a SchemeBuilder, but is done directly here for this core package.
	utilruntime.Must(AddConversionFuncs(scheme))
	utilruntime.Must(RegisterDefaults(scheme))
}

// scheme is the registry for the common types that adhere to the meta v1 API spec.
var scheme = runtime.NewScheme()

// ParameterCodec knows about query parameters used with the meta v1 API spec.
// It is used for encoding and decoding query parameters into the option types.
var ParameterCodec = runtime.NewParameterCodec(scheme)

// The init function is automatically called when the package is loaded.
// It registers the meta v1 types with the local scheme.
func init() {
	scheme.AddUnversionedTypes(SchemeGroupVersion,
		&ListOptions{},
		&ExportOptions{},
		&GetOptions{},
		&DeleteOptions{},
		&CreateOptions{},
		&UpdateOptions{},
		&PatchOptions{},
	)

	if err := AddMetaToScheme(scheme); err != nil {
		panic(err)
	}

	// Manually register conversion and default functions for the local scheme.
	utilruntime.Must(RegisterDefaults(scheme))
}

// AddMetaToScheme adds the meta-related types to the given scheme.
func AddMetaToScheme(scheme *runtime.Scheme) error {
	// Registers types used for tabular output from the API server.
	scheme.AddKnownTypes(SchemeGroupVersion,
		&Table{},
		&TableOptions{},
		&PartialObjectMetadata{},
		&PartialObjectMetadataList{},
	)

	// Registers a conversion function for IncludeObjectPolicy.
	return scheme.AddConversionFuncs(
		Convert_Slice_string_To_v1_IncludeObjectPolicy,
	)
}
