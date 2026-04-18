
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
 * @69869a93-5b7c-49ee-9183-67e6143d5e9a/staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/register.go
 * @brief API registration and scheme management for Kubernetes Meta V1.
 * This module defines the core architectural identity for Meta V1 API objects. 
 * It handles the registration of types into the runtime scheme, configures 
 * parameter codecs for URL query parsing, and orchestrates the transition 
 * between versioned and internal object representations.
 * 
 * Domain: Kubernetes API Machinery, Type Registration, Serialization.
 */

package v1

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

// GroupName is the group name for this API.
const GroupName = "meta.k8s.io"

/// Functional Utility: Identifies the GVK (Group Version Kind) mapping for V1 Meta objects.
// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v1"}

/// Functional Utility: Special group version for objects that lack explicit versioning.
// Unversioned is group version for unversioned API objects
// TODO: this should be v1 probably
var Unversioned = schema.GroupVersion{Group: "", Version: "v1"}

// WatchEventKind is name reserved for serializing watch events.
const WatchEventKind = "WatchEvent"

/**
 * Functional Utility: Helper to qualify a string kind with the Meta V1 group.
 */
// Kind takes an unqualified kind and returns a Group qualified GroupKind
func Kind(kind string) schema.GroupKind {
	return SchemeGroupVersion.WithKind(kind).GroupKind()
}

// scheme is the registry for the common types that adhere to the meta v1 API spec.
var scheme = runtime.NewScheme()

// ParameterCodec knows about query parameters used with the meta v1 API spec.
var ParameterCodec = runtime.NewParameterCodec(scheme)

/**
 * Functional Utility: Registers event-specific conversion functions.
 * Logic: Enables the transformation of watch events between versioned 
 * and internal formats.
 */
func addEventConversionFuncs(scheme *runtime.Scheme) error {
	return scheme.AddConversionFuncs(
		Convert_v1_WatchEvent_To_watch_Event,
		Convert_v1_InternalEvent_To_v1_WatchEvent,
		Convert_watch_Event_To_v1_WatchEvent,
		Convert_v1_WatchEvent_To_v1_InternalEvent,
	)
}

/// Functional Utility: Registry of common option objects (Get, List, Delete, etc.).
var optionsTypes = []runtime.Object{
	&ListOptions{},
	&ExportOptions{},
	&GetOptions{},
	&DeleteOptions{},
	&CreateOptions{},
	&UpdateOptions{},
	&PatchOptions{},
}

/**
 * Functional Utility: Primary entry point for registering types into a specific scheme.
 * Logic: Adds known types, unversioned types, and triggers the registration 
 * of conversion functions and default values.
 */
// AddToGroupVersion registers common meta types into schemas.
func AddToGroupVersion(scheme *runtime.Scheme, groupVersion schema.GroupVersion) {
	scheme.AddKnownTypeWithName(groupVersion.WithKind(WatchEventKind), &WatchEvent{})
	scheme.AddKnownTypeWithName(
		schema.GroupVersion{Group: groupVersion.Group, Version: runtime.APIVersionInternal}.WithKind(WatchEventKind),
		&InternalEvent{},
	)
	// Supports legacy code paths, most callers should use metav1.ParameterCodec for now
	scheme.AddKnownTypes(groupVersion, optionsTypes...)
	// Register Unversioned types under their own special group
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

/**
 * Functional Utility: Registers foundational Meta types into the scheme.
 */
// AddMetaToScheme registers base meta types into schemas.
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

/**
 * Functional Utility: Bootstrap logic executed on package import.
 * Logic: Self-registers the package's types into its own private scheme.
 */
func init() {
	scheme.AddUnversionedTypes(SchemeGroupVersion, optionsTypes...)

	utilruntime.Must(AddMetaToScheme(scheme))

	// register manually. This usually goes through the SchemeBuilder, which we cannot use here.
	utilruntime.Must(RegisterDefaults(scheme))
}
