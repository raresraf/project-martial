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
 * @file register.go
 * @brief Canonical type registration for Kubernetes Meta V1 API.
 * 
 * This module defines the schema registration logic for the fundamental Kubernetes 
 * metadata types. It orchestrates the mapping of Group-Version-Kind (GVK) 
 * identifiers to internal Go structures (e.g., ListOptions, Status), ensuring 
 * consistent serialization and versioning across the entire orchestration plane.
 * 
 * Domain: Production Systems, API Orchestration, Resource Lifecycle.
 */

package v1

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

// GroupName is the group name for this API.
const GroupName = "meta.k8s.io"

// SchemeGroupVersion is the canonical identifier for the meta v1 API group.
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v1"}

// Unversioned is the group version for unversioned API objects.
// Implementation Detail: Frequently used for backward compatibility and core status types.
var Unversioned = schema.GroupVersion{Group: "", Version: "v1"}

// WatchEventKind is the reserved GVK name for streaming event serialization.
const WatchEventKind = "WatchEvent"

// Kind resolves an unqualified kind string into a Group-qualified GroupKind.
func Kind(kind string) schema.GroupKind {
	return SchemeGroupVersion.WithKind(kind).GroupKind()
}

// AddToGroupVersion registers core metadata types into a provided runtime schema.
func AddToGroupVersion(scheme *runtime.Scheme, groupVersion schema.GroupVersion) {
	// Serialization Mapping: Registers event types for both external and internal API versions.
	scheme.AddKnownTypeWithName(groupVersion.WithKind(WatchEventKind), &WatchEvent{})
	scheme.AddKnownTypeWithName(
		schema.GroupVersion{Group: groupVersion.Group, Version: runtime.APIVersionInternal}.WithKind(WatchEventKind),
		&InternalEvent{},
	)
	
	/**
	 * Execution Block: Option Registration.
	 * Logic: Registers common operational parameters (e.g., GET, PATCH, LIST) to 
	 * ensure they can be decoded from request parameters.
	 */
	scheme.AddKnownTypes(groupVersion,
		&ListOptions{},
		&ExportOptions{},
		&GetOptions{},
		&DeleteOptions{},
		&CreateOptions{},
		&UpdateOptions{},
		&PatchOptions{},
	)
	
	// Conversion Logic: Registers transformation functions between versioned internal types.
	utilruntime.Must(scheme.AddConversionFuncs(
		Convert_v1_WatchEvent_To_watch_Event,
		Convert_v1_InternalEvent_To_v1_WatchEvent,
		Convert_watch_Event_To_v1_WatchEvent,
		Convert_v1_WatchEvent_To_v1_InternalEvent,
	))

	/**
	 * Block Logic: Unversioned type registration.
	 * Invariant: Registers status and metadata types that do not belong to a 
	 * specific versioned resource group.
	 */
	scheme.AddUnversionedTypes(Unversioned,
		&Status{},
		&APIVersions{},
		&APIGroupList{},
		&APIGroup{},
		&APIResourceList{},
	)

	// register manually. This usually goes through the SchemeBuilder, which we cannot use here.
	utilruntime.Must(AddConversionFuncs(scheme))
	utilruntime.Must(RegisterDefaults(scheme))
}

// scheme is the private registry for common types adhering to the meta v1 spec.
var scheme = runtime.NewScheme()

// ParameterCodec provides standardized query parameter decoding for meta v1 types.
var ParameterCodec = runtime.NewParameterCodec(scheme)

func init() {
	/**
	 * Initialization Logic: Bootstraps the default meta scheme.
	 * Invariant: Configures the schema with common options and metadata types 
	 * during package loading to prevent runtime registration race conditions.
	 */
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

	// register manually. This usually goes through the SchemeBuilder, which we cannot use here.
	utilruntime.Must(RegisterDefaults(scheme))
}

// AddMetaToScheme populates a scheme with advanced metadata types (Tables, PartialMetadata).
func AddMetaToScheme(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&Table{},
		&TableOptions{},
		&PartialObjectMetadata{},
		&PartialObjectMetadataList{},
	)

	// Logic: Injects custom conversion behavior for IncludeObjectPolicy strings.
	return scheme.AddConversionFuncs(
		Convert_Slice_string_To_v1_IncludeObjectPolicy,
	)
}
