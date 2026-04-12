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

// Package v1beta1 contains the v1beta1 version of the API machinery meta types.
package v1beta1

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// GroupName is the group name for this API.
const GroupName = "meta.k8s.io"

// SchemeGroupVersion is group version used to register these objects.
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v1beta1"}

// Kind takes an unqualified kind and returns a Group qualified GroupKind.
func Kind(kind string) schema.GroupKind {
	return SchemeGroupVersion.WithKind(kind).GroupKind()
}

// scheme is the registry for the common types that adhere to the meta v1beta1 API spec.
var scheme = runtime.NewScheme()

// ParameterCodec knows about query parameters used with the meta v1beta1 API spec.
var ParameterCodec = runtime.NewParameterCodec(scheme)

// init is called automatically when the package is loaded. It registers the
// meta v1beta1 types with the local scheme.
func init() {
	if err := AddMetaToScheme(scheme); err != nil {
		panic(err)
	}
}

// AddMetaToScheme adds the meta-related types to the given scheme.
func AddMetaToScheme(scheme *runtime.Scheme) error {
	// Registers the Table and PartialObjectMetadata types, which are often used
	// for server-side printing and efficient object retrieval.
	scheme.AddKnownTypes(SchemeGroupVersion,
		&Table{},
		&TableOptions{},
		&PartialObjectMetadata{},
		&PartialObjectMetadataList{},
	)

	// Registers conversion functions for the types in this package.
	return scheme.AddConversionFuncs(
		Convert_Slice_string_To_v1beta1_IncludeObjectPolicy,
	)

	// register manually. This usually goes through the SchemeBuilder, which we cannot use here.
	//scheme.AddGeneratedDeepCopyFuncs(GetGeneratedDeepCopyFuncs()...)
}
