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

package etcd

import (
	"net/http"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/rest/resttest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"

	"github.com/coreos/go-etcd/etcd"
)

const (
	PASS = iota
	FAIL
)

type fakeConnectionInfoGetter struct {
}

func (fakeConnectionInfoGetter) GetConnectionInfo(host string) (string, uint, http.RoundTripper, error) {
	return "http", 12345, nil, nil
}

func newEtcdStorage(t *testing.T) (*tools.FakeEtcdClient, storage.Interface) {
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	fakeEtcdClient.TestIndex = true
	etcdStorage := etcdstorage.NewEtcdStorage(fakeEtcdClient, latest.Codec, etcdtest.PathPrefix())
	return fakeEtcdClient, etcdStorage
}

func newStorage(t *testing.T) (*REST, *tools.FakeEtcdClient) {
	fakeEtcdClient, s := newEtcdStorage(t)
	storage, _ := NewStorage(s, fakeConnectionInfoGetter{})
	return storage, fakeEtcdClient
}

func validNewNode() *api.Node {
	return &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
			Labels: map[string]string{
				"name": "foo",
			},
		},
		Spec: api.NodeSpec{
			ExternalID: "external",
		},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
				api.ResourceName(api.ResourceMemory): resource.MustParse("0"),
			},
		},
	}
}

func validChangedNode() *api.Node {
	node := validNewNode()
	node.ResourceVersion = "1"
	return node
}

func TestCreate(t *testing.T) {
	storage, fakeEtcdClient := newStorage(t)
	test := resttest.New(t, storage, fakeEtcdClient.SetError).ClusterScope()
	node := validNewNode()
	node.ObjectMeta = api.ObjectMeta{GenerateName: "foo"}
	test.TestCreate(
		// valid
		node,
		// invalid
		&api.Node{
			ObjectMeta: api.ObjectMeta{Name: "_-a123-a_"},
		},
	)
}

func TestDelete(t *testing.T) {
	ctx := api.NewContext()
	storage, fakeEtcdClient := newStorage(t)
	test := resttest.New(t, storage, fakeEtcdClient.SetError).ClusterScope()

	node := validChangedNode()
	key, _ := storage.KeyFunc(ctx, node.Name)
	key = etcdtest.AddPrefix(key)
	createFn := func() runtime.Object {
		fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{
				Node: &etcd.Node{
					Value:         runtime.EncodeOrDie(latest.Codec, node),
					ModifiedIndex: 1,
				},
			},
		}
		return node
	}
	gracefulSetFn := func() bool {
		if fakeEtcdClient.Data[key].R.Node == nil {
			return false
		}
		return fakeEtcdClient.Data[key].R.Node.TTL == 30
	}
	test.TestDelete(createFn, gracefulSetFn)
}

func TestEtcdGetNode(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError).ClusterScope()
	node := validNewNode()
	test.TestGet(node)
}

func TestEtcdListNodes(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError).ClusterScope()
	key := etcdtest.AddPrefix(storage.KeyRootFunc(test.TestContext()))
	node := validNewNode()
	test.TestList(
		node,
		func(objects []runtime.Object) []runtime.Object {
			return registrytest.SetObjectsForKey(fakeClient, key, objects)
		},
		func(resourceVersion uint64) {
			registrytest.SetResourceVersion(fakeClient, resourceVersion)
		})
}

func TestEtcdUpdateEndpoints(t *testing.T) {
	ctx := api.NewContext()
	storage, fakeClient := newStorage(t)
	node := validChangedNode()

	key, _ := storage.KeyFunc(ctx, node.Name)
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, validNewNode()), 0)

	_, _, err := storage.Update(ctx, node)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	response, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var nodeOut api.Node
	err = latest.Codec.DecodeInto([]byte(response.Node.Value), &nodeOut)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	node.ObjectMeta.ResourceVersion = nodeOut.ObjectMeta.ResourceVersion
	if !api.Semantic.DeepEqual(node, &nodeOut) {
		t.Errorf("Unexpected node: %#v, expected %#v", &nodeOut, node)
	}
}

func TestEtcdDeleteNode(t *testing.T) {
	ctx := api.NewContext()
	storage, fakeClient := newStorage(t)
	node := validNewNode()
	key, _ := storage.KeyFunc(ctx, node.Name)
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, node), 0)
	_, err := storage.Delete(ctx, node.Name, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(fakeClient.DeletedKeys) != 1 {
		t.Errorf("Expected 1 delete, found %#v", fakeClient.DeletedKeys)
	}
	if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
}

func TestEtcdWatchNode(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	watching, err := storage.Watch(ctx,
		labels.Everything(),
		fields.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	default:
	}
	fakeClient.WatchInjectError <- nil
	if _, ok := <-watching.ResultChan(); ok {
		t.Errorf("watching channel should be closed")
	}
	watching.Stop()
}

func TestEtcdWatchNodesMatch(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	node := validNewNode()

	watching, err := storage.Watch(ctx,
		labels.SelectorFromSet(labels.Set{"name": node.Name}),
		fields.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	nodeBytes, _ := latest.Codec.Encode(node)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(nodeBytes),
		},
	}
	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	case <-time.After(time.Millisecond * 100):
		t.Error("unexpected timeout from result channel")
	}
	watching.Stop()
}

func TestEtcdWatchNodesNotMatch(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	node := validNewNode()

	watching, err := storage.Watch(ctx,
		labels.SelectorFromSet(labels.Set{"name": "bar"}),
		fields.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	nodeBytes, _ := latest.Codec.Encode(node)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(nodeBytes),
		},
	}

	select {
	case <-watching.ResultChan():
		t.Error("unexpected result from result channel")
	case <-time.After(time.Millisecond * 100):
		// expected case
	}
}
