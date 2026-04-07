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

// Package v1beta1 contains the v1beta1 version of the API Machinery's meta-data interfaces.
//
// Architectural Intent: This package provides the beta version of common metadata
// types used across the Kubernetes API. Its primary function is to register
// these beta types, such as `Table` and `PartialObjectMetadata`, with the
// `runtime.Scheme`. This registration allows the Kubernetes API server and clients
// to serialize, deserialize, and otherwise handle these objects correctly when they
// are used in API requests and responses, particularly for features that are
// being introduced or evolved.
package v1beta1

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

// GroupName is the group name for this API.
const GroupName = "meta.k8s.io"

// SchemeGroupVersion is group version used to register these objects.
// It uniquely identifies the `meta.k8s.io/v1beta1` API group version.
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v1beta1"}

// Kind takes an unqualified kind and returns a Group qualified GroupKind.
// This is a helper function to construct a fully-qualified kind identifier
// for an object within this specific API group version.
func Kind(kind string) schema.GroupKind {
	return SchemeGroupVersion.WithKind(kind).GroupKind()
}

// scheme is the registry for the common types that adhere to the meta v1beta1 API spec.
// This is a local scheme instance used for registration within this package.
var scheme = runtime.NewScheme()

// ParameterCodec knows about query parameters used with the meta v1beta1 API spec.
// Its role is to handle the encoding and decoding of API options from URL query parameters.
var ParameterCodec = runtime.NewParameterCodec(scheme)

// AddMetaToScheme registers the v1beta1 meta types into a given scheme.
// This function is the primary entry point for other packages to incorporate
// the `meta/v1beta1` types into their own scheme, enabling them to recognize
// and manage types like `Table` and `PartialObjectMetadata`.
func AddMetaToScheme(scheme *runtime.Scheme) error {
	// Pre-condition: The provided `scheme` is a valid, initialized runtime.Scheme.
	// This block registers the v1beta1 versions of key meta-types.
	scheme.AddKnownTypes(SchemeGroupVersion,
		&Table{},
		&TableOptions{},
		&PartialObjectMetadata{},
		&PartialObjectMetadataList{},
	)

	return scheme.AddConversionFuncs(
		Convert_Slice_string_To_v1beta1_IncludeObjectPolicy,
	)
}

// init ensures that the meta types are registered automatically when this package is imported.
// It calls AddMetaToScheme to add the v1beta1 types to the local `scheme` instance.
func init() {
	utilruntime.Must(AddMetaToScheme(scheme))

	// register manually. This usually goes through the SchemeBuilder, which we cannot use here.
	// This step adds default-value-setting functions for the registered types.
	utilruntime.Must(RegisterDefaults(scheme))
}
