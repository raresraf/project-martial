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

// Package api contains helper methods for working with API objects, which are centralized representations
// of resources. This file provides utilities for interacting with the metadata of these objects,
// such as setting system-managed fields and accessing common metadata attributes through a standard
// interface.
package api

import (
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/meta/metatypes"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
)

// FillObjectMetaSystemFields populates fields that are managed by the system on ObjectMeta.
// This ensures that objects created by clients have system-level identifiers and timestamps
// applied consistently by the server. It is called during the creation process of a resource.
func FillObjectMetaSystemFields(ctx Context, meta *ObjectMeta) {
	meta.CreationTimestamp = unversioned.Now()
	meta.UID = util.NewUUID()
	// SelfLink is currently cleared by default, to be populated by the server.
	meta.SelfLink = ""
}

// HasObjectMetaSystemFieldValues returns true if fields that are managed by the system on ObjectMeta have values.
// This is used to determine if an object has already been persisted to storage, as these fields are
// only populated by the system upon creation.
func HasObjectMetaSystemFieldValues(meta *ObjectMeta) bool {
	return !meta.CreationTimestamp.Time.IsZero() ||
		len(meta.UID) != 0
}

// ObjectMetaFor returns a pointer to a provided object's ObjectMeta.
// This is a legacy helper function that uses reflection to access the `ObjectMeta` field.
// Its usage is discouraged in favor of the more efficient and type-safe `meta.Accessor()`.
// TODO: allow runtime.Unknown to extract this object
// TODO: Remove this function and use meta.Accessor() instead.
func ObjectMetaFor(obj runtime.Object) (*ObjectMeta, error) {
	v, err := conversion.EnforcePtr(obj)
	if err != nil {
		return nil, err
	}
	var meta *ObjectMeta
	err = runtime.FieldPtr(v, "ObjectMeta", &meta)
	return meta, err
}

// ListMetaFor returns a pointer to a provided object's ListMeta,
// or an error if the object does not have that pointer.
// Like ObjectMetaFor, this is a legacy helper function that relies on reflection.
// TODO: allow runtime.Unknown to extract this object
func ListMetaFor(obj runtime.Object) (*unversioned.ListMeta, error) {
	v, err := conversion.EnforcePtr(obj)
	if err != nil {
		return nil, err
	}
	var meta *unversioned.ListMeta
	err = runtime.FieldPtr(v, "ListMeta", &meta)
	return meta, err
}

// GetObjectMeta implements the ObjectMetaAccessor interface, which allows
// retrieving the ObjectMeta field from any resource that embeds it.
func (obj *ObjectMeta) GetObjectMeta() meta.Object { return obj }

// GetObjectKind implements the ObjectKind interface, returning a pointer to itself.
// This allows an ObjectReference to be used where an ObjectKind is expected.
func (obj *ObjectReference) GetObjectKind() unversioned.ObjectKind { return obj }

// The following methods implement the meta.Object interface for ObjectMeta.
// This allows generic, polymorphic treatment of metadata across all API objects.
// Any object that embeds `api.ObjectMeta` can be cast to `meta.Object` and manipulated
// through these methods.

// GetNamespace returns the namespace of the object.
func (meta *ObjectMeta) GetNamespace() string { return meta.Namespace }

// SetNamespace sets the namespace of the object.
func (meta *ObjectMeta) SetNamespace(namespace string) { meta.Namespace = namespace }

// GetName returns the name of the object.
func (meta *ObjectMeta) GetName() string { return meta.Name }

// SetName sets the name of the object.
func (meta *ObjectMeta) SetName(name string) { meta.Name = name }

// GetGenerateName returns the base name for generated names.
func (meta *ObjectMeta) GetGenerateName() string { return meta.GenerateName }

// SetGenerateName sets the base name for generated names.
func (meta *ObjectMeta) SetGenerateName(generateName string) { meta.GenerateName = generateName }

// GetUID returns the unique identifier of the object.
func (meta *ObjectMeta) GetUID() types.UID { return meta.UID }

// SetUID sets the unique identifier of the object.
func (meta *ObjectMeta) SetUID(uid types.UID) { meta.UID = uid }

// GetResourceVersion returns the resource version of the object.
func (meta *ObjectMeta) GetResourceVersion() string { return meta.ResourceVersion }

// SetResourceVersion sets the resource version of the object.
func (meta *ObjectMeta) SetResourceVersion(version string) { meta.ResourceVersion = version }

// GetSelfLink returns the self link of the object.
func (meta *ObjectMeta) GetSelfLink() string { return meta.SelfLink }

// SetSelfLink sets the self link of the object.
func (meta *ObjectMeta) SetSelfLink(selfLink string) { meta.SelfLink = selfLink }

// GetCreationTimestamp returns the creation timestamp of the object.
func (meta *ObjectMeta) GetCreationTimestamp() unversioned.Time { return meta.CreationTimestamp }

// SetCreationTimestamp sets the creation timestamp of the object.
func (meta *ObjectMeta) SetCreationTimestamp(creationTimestamp unversioned.Time) {
	meta.CreationTimestamp = creationTimestamp
}

// GetDeletionTimestamp returns the deletion timestamp of the object.
func (meta *ObjectMeta) GetDeletionTimestamp() *unversioned.Time { return meta.DeletionTimestamp }

// SetDeletionTimestamp sets the deletion timestamp of the object.
func (meta *ObjectMeta) SetDeletionTimestamp(deletionTimestamp *unversioned.Time) {
	meta.DeletionTimestamp = deletionTimestamp
}

// GetLabels returns the labels of the object.
func (meta *ObjectMeta) GetLabels() map[string]string { return meta.Labels }

// SetLabels sets the labels of the object.
func (meta *ObjectMeta) SetLabels(labels map[string]string) { meta.Labels = labels }

// GetAnnotations returns the annotations of the object.
func (meta *ObjectMeta) GetAnnotations() map[string]string { return meta.Annotations }

// SetAnnotations sets the annotations of the object.
func (meta *ObjectMeta) SetAnnotations(annotations map[string]string) { meta.Annotations = annotations }

// GetFinalizers returns the finalizers of the object.
func (meta *ObjectMeta) GetFinalizers() []string { return meta.Finalizers }

// SetFinalizers sets the finalizers of the object.
func (meta *ObjectMeta) SetFinalizers(finalizers []string) { meta.Finalizers = finalizers }

// GetOwnerReferences returns the owner references of the object.
// It converts the internal `api.OwnerReference` type to the more general
// `metatypes.OwnerReference` type for use in generic contexts.
func (meta *ObjectMeta) GetOwnerReferences() []metatypes.OwnerReference {
	ret := make([]metatypes.OwnerReference, len(meta.OwnerReferences))
	for i := 0; i < len(meta.OwnerReferences); i++ {
		ret[i].Kind = meta.OwnerReferences[i].Kind
		ret[i].Name = meta.OwnerReferences[i].Name
		ret[i].UID = meta.OwnerReferences[i].UID
		ret[i].APIVersion = meta.OwnerReferences[i].APIVersion
		if meta.OwnerReferences[i].Controller != nil {
			value := *meta.OwnerReferences[i].Controller
			ret[i].Controller = &value
		}
	}
	return ret
}

// SetOwnerReferences sets the owner references of the object.
// It converts the general `metatypes.OwnerReference` type to the internal
// `api.OwnerReference` type.
func (meta *ObjectMeta) SetOwnerReferences(references []metatypes.OwnerReference) {
	newReferences := make([]OwnerReference, len(references))
	for i := 0; i < len(references); i++ {
		newReferences[i].Kind = references[i].Kind
		newReferences[i].Name = references[i].Name
		newReferences[i].UID = references[i].UID
		newReferences[i].APIVersion = references[i].APIVersion
		if references[i].Controller != nil {
			value := *references[i].Controller
			newReferences[i].Controller = &value
		}
	}
	meta.OwnerReferences = newReferences
}
