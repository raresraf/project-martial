/*
Copyright 2015 The Kubernetes Authors.

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

package testing

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// TestResource is a minimalist API object type that is used for testing
// the storage layer. It embeds the standard Kubernetes object metadata and
// includes a single integer value for test purposes.
type TestResource struct {
	// TypeMeta provides the Kind and APIVersion fields, which are fundamental
	// for Kubernetes object serialization.
	metav1.TypeMeta `json:",inline"`

	// ObjectMeta contains the standard metadata for any Kubernetes object,
	// such as Name, Namespace, and ResourceVersion.
	metav1.ObjectMeta `json:"metadata"`

	// Value is a simple integer field used for testing data storage and retrieval.
	Value int `json:"value"`
}
