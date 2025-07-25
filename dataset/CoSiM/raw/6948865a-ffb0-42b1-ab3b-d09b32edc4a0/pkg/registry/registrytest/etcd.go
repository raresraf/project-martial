/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package registrytest

import (
	"github.com/coreos/go-etcd/etcd"

	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/tools"
)

func SetResourceVersion(fakeClient *tools.FakeEtcdClient, resourceVersion uint64) {
	fakeClient.ChangeIndex = resourceVersion
}

func SetObjectsForKey(fakeClient *tools.FakeEtcdClient, key string, objects []runtime.Object) []runtime.Object {
	result := make([]runtime.Object, len(objects))
	if len(objects) > 0 {
		nodes := make([]*etcd.Node, len(objects))
		for i, obj := range objects {
			encoded := runtime.EncodeOrDie(latest.Codec, obj)
			decoded, _ := latest.Codec.Decode([]byte(encoded))
			nodes[i] = &etcd.Node{Value: encoded}
			result[i] = decoded
		}
		fakeClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{
				Node: &etcd.Node{
					Nodes: nodes,
				},
			},
			E: nil,
		}
	} else {
		fakeClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{},
			E: fakeClient.NewError(tools.EtcdErrorCodeNotFound),
		}
	}
	return result
}
