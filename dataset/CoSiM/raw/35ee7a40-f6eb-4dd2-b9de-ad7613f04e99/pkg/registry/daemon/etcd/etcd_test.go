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
	"testing"
	"time"

	"github.com/coreos/go-etcd/etcd"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/rest/resttest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
)

const (
	PASS = iota
	FAIL
)

func newStorage(t *testing.T) (*REST, *tools.FakeEtcdClient) {
	etcdStorage, fakeClient := registrytest.NewEtcdStorage(t)
	return NewREST(etcdStorage), fakeClient
}

// createController is a helper function that returns a controller with the updated resource version.
func createController(storage *REST, dc api.Daemon, t *testing.T) (api.Daemon, error) {
	ctx := api.WithNamespace(api.NewContext(), dc.Namespace)
	obj, err := storage.Create(ctx, &dc)
	if err != nil {
		t.Errorf("Failed to create controller, %v", err)
	}
	newDc := obj.(*api.Daemon)
	return *newDc, nil
}

var validPodTemplate = api.PodTemplate{
	Template: api.PodTemplateSpec{
		ObjectMeta: api.ObjectMeta{
			Labels: map[string]string{"a": "b"},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:            "test",
					Image:           "test_image",
					ImagePullPolicy: api.PullIfNotPresent,
				},
			},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
	},
}

var validControllerSpec = api.DaemonSpec{
	Selector: validPodTemplate.Template.Labels,
	Template: &validPodTemplate.Template,
}

var validController = api.Daemon{
	ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "default"},
	Spec:       validControllerSpec,
}

func TestCreate(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	test.TestCreate(
		// valid
		&api.Daemon{
			Spec: api.DaemonSpec{
				Selector: map[string]string{"a": "b"},
				Template: &validPodTemplate.Template,
			},
		},
		func(ctx api.Context, obj runtime.Object) error {
			return registrytest.SetObject(fakeClient, storage.KeyFunc, ctx, obj)
		},
		func(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
			return registrytest.GetObject(fakeClient, storage.KeyFunc, storage.NewFunc, ctx, obj)
		},
		// invalid
		&api.Daemon{
			Spec: api.DaemonSpec{
				Selector: map[string]string{},
				Template: &validPodTemplate.Template,
			},
		},
	)
}

// makeControllerKey constructs etcd paths to controller items enforcing namespace rules.
func makeControllerKey(ctx api.Context, id string) (string, error) {
	return etcdgeneric.NamespaceKeyFunc(ctx, daemonPrefix, id)
}

// makeControllerListKey constructs etcd paths to the root of the resource,
// not a specific controller resource
func makeControllerListKey(ctx api.Context) string {
	return etcdgeneric.NamespaceKeyRootFunc(ctx, daemonPrefix)
}

func TestEtcdGetController(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	key, _ := makeControllerKey(ctx, validController.Name)
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &validController), 0)
	ctrl, err := storage.Get(ctx, validController.Name)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	controller, ok := ctrl.(*api.Daemon)
	if !ok {
		t.Errorf("Expected a controller, got %#v", ctrl)
	}
	if controller.Name != validController.Name {
		t.Errorf("Unexpected controller: %#v", controller)
	}
}

func TestEtcdControllerValidatesUpdate(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, _ := newStorage(t)

	updateController, err := createController(storage, validController, t)
	if err != nil {
		t.Errorf("Failed to create controller, cannot proceed with test.")
	}

	updaters := []func(dc api.Daemon) (runtime.Object, bool, error){
		func(dc api.Daemon) (runtime.Object, bool, error) {
			dc.UID = "newUID"
			return storage.Update(ctx, &dc)
		},
		func(dc api.Daemon) (runtime.Object, bool, error) {
			dc.Name = ""
			return storage.Update(ctx, &dc)
		},
		func(dc api.Daemon) (runtime.Object, bool, error) {
			dc.Spec.Template.Spec.RestartPolicy = api.RestartPolicyOnFailure
			return storage.Update(ctx, &dc)
		},
		func(dc api.Daemon) (runtime.Object, bool, error) {
			dc.Spec.Selector = map[string]string{}
			return storage.Update(ctx, &dc)
		},
	}
	for _, u := range updaters {
		c, updated, err := u(updateController)
		if c != nil || updated {
			t.Errorf("Expected nil object and not created")
		}
		if !errors.IsInvalid(err) && !errors.IsBadRequest(err) {
			t.Errorf("Expected invalid or bad request error, got %v of type %T", err, err)
		}
	}
}

func TestEtcdControllerValidatesNamespaceOnUpdate(t *testing.T) {
	storage, _ := newStorage(t)
	ns := "newnamespace"

	// The update should fail if the namespace on the controller is set to something
	// other than the namespace on the given context, even if the namespace on the
	// controller is valid.
	updateController, err := createController(storage, validController, t)

	newNamespaceController := validController
	newNamespaceController.Namespace = ns
	_, err = createController(storage, newNamespaceController, t)

	c, updated, err := storage.Update(api.WithNamespace(api.NewContext(), ns), &updateController)
	if c != nil || updated {
		t.Errorf("Expected nil object and not created")
	}
	// TODO: Be more paranoid about the type of error and make sure it has the substring
	// "namespace" in it, once #5684 is fixed. Ideally this would be a NewBadRequest.
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	}
}

// TestEtcdGetControllerDifferentNamespace ensures same-name controllers in different namespaces do not clash
func TestEtcdGetControllerDifferentNamespace(t *testing.T) {
	storage, fakeClient := newStorage(t)

	otherNs := "other"
	ctx1 := api.NewDefaultContext()
	ctx2 := api.WithNamespace(api.NewContext(), otherNs)

	key1, _ := makeControllerKey(ctx1, validController.Name)
	key2, _ := makeControllerKey(ctx2, validController.Name)

	key1 = etcdtest.AddPrefix(key1)
	key2 = etcdtest.AddPrefix(key2)

	fakeClient.Set(key1, runtime.EncodeOrDie(latest.Codec, &validController), 0)
	otherNsController := validController
	otherNsController.Namespace = otherNs
	fakeClient.Set(key2, runtime.EncodeOrDie(latest.Codec, &otherNsController), 0)

	obj, err := storage.Get(ctx1, validController.Name)
	ctrl1, _ := obj.(*api.Daemon)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if ctrl1.Name != "foo" {
		t.Errorf("Unexpected controller: %#v", ctrl1)
	}
	if ctrl1.Namespace != "default" {
		t.Errorf("Unexpected controller: %#v", ctrl1)
	}

	obj, err = storage.Get(ctx2, validController.Name)
	ctrl2, _ := obj.(*api.Daemon)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if ctrl2.Name != "foo" {
		t.Errorf("Unexpected controller: %#v", ctrl2)
	}
	if ctrl2.Namespace != "other" {
		t.Errorf("Unexpected controller: %#v", ctrl2)
	}

}

func TestEtcdGetControllerNotFound(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	key, _ := makeControllerKey(ctx, validController.Name)
	key = etcdtest.AddPrefix(key)

	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	ctrl, err := storage.Get(ctx, validController.Name)
	if ctrl != nil {
		t.Errorf("Unexpected non-nil controller: %#v", ctrl)
	}
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error returned: %#v", err)
	}
}

func TestEtcdDeleteController(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	key, _ := makeControllerKey(ctx, validController.Name)
	key = etcdtest.AddPrefix(key)

	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &validController), 0)
	obj, err := storage.Delete(ctx, validController.Name, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if status, ok := obj.(*api.Status); !ok {
		t.Errorf("Expected status of delete, got %#v", status)
	} else if status.Status != api.StatusSuccess {
		t.Errorf("Expected success, got %#v", status.Status)
	}
	if len(fakeClient.DeletedKeys) != 1 {
		t.Errorf("Expected 1 delete, found %#v", fakeClient.DeletedKeys)
	}
	if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
}

func TestEtcdListControllers(t *testing.T) {
	storage, fakeClient := newStorage(t)
	ctx := api.NewDefaultContext()
	key := makeControllerListKey(ctx)
	key = etcdtest.AddPrefix(key)
	controller := validController
	controller.Name = "bar"
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &validController),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &controller),
					},
				},
			},
		},
		E: nil,
	}
	objList, err := storage.List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	controllers, _ := objList.(*api.DaemonList)
	if len(controllers.Items) != 2 || controllers.Items[0].Name != validController.Name || controllers.Items[1].Name != controller.Name {
		t.Errorf("Unexpected controller list: %#v", controllers)
	}
}

func TestEtcdListControllersNotFound(t *testing.T) {
	storage, fakeClient := newStorage(t)
	ctx := api.NewDefaultContext()
	key := makeControllerListKey(ctx)
	key = etcdtest.AddPrefix(key)

	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}
	objList, err := storage.List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	controllers, _ := objList.(*api.DaemonList)
	if len(controllers.Items) != 0 {
		t.Errorf("Unexpected controller list: %#v", controllers)
	}
}

func TestEtcdListControllersLabelsMatch(t *testing.T) {
	storage, fakeClient := newStorage(t)
	ctx := api.NewDefaultContext()
	key := makeControllerListKey(ctx)
	key = etcdtest.AddPrefix(key)

	controller := validController
	controller.Labels = map[string]string{"k": "v"}
	controller.Name = "bar"

	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &validController),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &controller),
					},
				},
			},
		},
		E: nil,
	}
	testLabels := labels.SelectorFromSet(labels.Set(controller.Labels))
	objList, err := storage.List(ctx, testLabels, fields.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	controllers, _ := objList.(*api.DaemonList)
	if len(controllers.Items) != 1 || controllers.Items[0].Name != controller.Name ||
		!testLabels.Matches(labels.Set(controllers.Items[0].Labels)) {
		t.Errorf("Unexpected controller list: %#v for query with labels %#v",
			controllers, testLabels)
	}
}

func TestEtcdWatchController(t *testing.T) {
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

// Tests that we can watch for the creation of daemon controllers with specified labels.
func TestEtcdWatchControllersMatch(t *testing.T) {
	ctx := api.WithNamespace(api.NewDefaultContext(), validController.Namespace)
	storage, fakeClient := newStorage(t)
	fakeClient.ExpectNotFoundGet(etcdgeneric.NamespaceKeyRootFunc(ctx, "/registry/pods"))

	watching, err := storage.Watch(ctx,
		labels.SelectorFromSet(validController.Spec.Selector),
		fields.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	// The watcher above is waiting for these Labels, on receiving them it should
	// apply the ControllerStatus decorator, which lists pods, causing a query against
	// the /registry/pods endpoint of the etcd client.
	controller := &api.Daemon{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Labels:    validController.Spec.Selector,
			Namespace: "default",
		},
	}
	controllerBytes, _ := latest.Codec.Encode(controller)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(controllerBytes),
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

// Tests that we can watch for daemon controllers with specified fields.
func TestEtcdWatchControllersFields(t *testing.T) {
	ctx := api.WithNamespace(api.NewDefaultContext(), validController.Namespace)
	storage, fakeClient := newStorage(t)
	fakeClient.ExpectNotFoundGet(etcdgeneric.NamespaceKeyRootFunc(ctx, "/registry/pods"))

	testFieldMap := map[int][]fields.Set{
		PASS: {
			{"metadata.name": "foo"},
		},
		FAIL: {
			{"metadata.name": "bar"},
			{"name": "foo"},
		},
	}
	testEtcdActions := []string{
		etcdstorage.EtcdCreate,
		etcdstorage.EtcdSet,
		etcdstorage.EtcdCAS,
		etcdstorage.EtcdDelete}

	controller := &api.Daemon{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Labels:    validController.Spec.Selector,
			Namespace: "default",
		},
		Status: api.DaemonStatus{
			CurrentNumberScheduled: 2,
			NumberMisscheduled:     1,
			DesiredNumberScheduled: 4,
		},
	}
	controllerBytes, _ := latest.Codec.Encode(controller)

	for expectedResult, fieldSet := range testFieldMap {
		for _, field := range fieldSet {
			for _, action := range testEtcdActions {
				watching, err := storage.Watch(ctx,
					labels.Everything(),
					field.AsSelector(),
					"1",
				)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				var prevNode *etcd.Node = nil
				node := &etcd.Node{
					Value: string(controllerBytes),
				}
				if action == etcdstorage.EtcdDelete {
					prevNode = node
				}
				fakeClient.WaitForWatchCompletion()
				fakeClient.WatchResponse <- &etcd.Response{
					Action:   action,
					Node:     node,
					PrevNode: prevNode,
				}

				select {
				case r, ok := <-watching.ResultChan():
					if expectedResult == FAIL {
						t.Errorf("Unexpected result from channel %#v", r)
					}
					if !ok {
						t.Errorf("watching channel should be open")
					}
				case <-time.After(time.Millisecond * 100):
					if expectedResult == PASS {
						t.Error("unexpected timeout from result channel")
					}
				}
				watching.Stop()
			}
		}
	}
}

func TestEtcdWatchControllersNotMatch(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	fakeClient.ExpectNotFoundGet(etcdgeneric.NamespaceKeyRootFunc(ctx, "/registry/pods"))

	watching, err := storage.Watch(ctx,
		labels.SelectorFromSet(labels.Set{"name": "foo"}),
		fields.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	controller := &api.Daemon{
		ObjectMeta: api.ObjectMeta{
			Name: "bar",
			Labels: map[string]string{
				"name": "bar",
			},
		},
	}
	controllerBytes, _ := latest.Codec.Encode(controller)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(controllerBytes),
		},
	}

	select {
	case <-watching.ResultChan():
		t.Error("unexpected result from result channel")
	case <-time.After(time.Millisecond * 100):
		// expected case
	}
}

func TestDelete(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	key, _ := makeControllerKey(ctx, validController.Name)
	key = etcdtest.AddPrefix(key)

	createFn := func() runtime.Object {
		dc := validController
		dc.ResourceVersion = "1"
		fakeClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{
				Node: &etcd.Node{
					Value:         runtime.EncodeOrDie(latest.Codec, &dc),
					ModifiedIndex: 1,
				},
			},
		}
		return &dc
	}
	gracefulSetFn := func() bool {
		// If the controller is still around after trying to delete either the delete
		// failed, or we're deleting it gracefully.
		if fakeClient.Data[key].R.Node != nil {
			return true
		}
		return false
	}

	test.TestDelete(createFn, gracefulSetFn)
}
