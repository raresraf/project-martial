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

package cm

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/api/resource"
	internalapi "k8s.io/cri-api/pkg/apis"
	podresourcesapi "k8s.io/kubernetes/pkg/kubelet/apis/podresources/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
	"k8s.io/kubernetes/pkg/kubelet/status"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

type containerManagerStub struct {
	shouldResetExtendedResourceCapacity bool
}

var _ ContainerManager = &containerManagerStub{}

func (cm *containerManagerStub) Start(_ *v1.Node, _ ActivePodsFunc, _ config.SourcesReady, _ status.PodStatusProvider, _ internalapi.RuntimeService) error {
	klog.V(2).Infof("Starting stub container manager")
	return nil
}

func (cm *containerManagerStub) SystemCgroupsLimit() v1.ResourceList {
	return v1.ResourceList{}
}

func (cm *containerManagerStub) GetNodeConfig() NodeConfig {
	return NodeConfig{}
}

func (cm *containerManagerStub) GetMountedSubsystems() *CgroupSubsystems {
	return &CgroupSubsystems{}
}

func (cm *containerManagerStub) GetQOSContainersInfo() QOSContainersInfo {
	return QOSContainersInfo{}
}

func (cm *containerManagerStub) UpdateQOSCgroups() error {
	return nil
}

func (cm *containerManagerStub) Status() Status {
	return Status{}
}

func (cm *containerManagerStub) GetNodeAllocatableReservation() v1.ResourceList {
	return nil
}

func (cm *containerManagerStub) GetCapacity() v1.ResourceList {
	c := v1.ResourceList{
		v1.ResourceEphemeralStorage: *resource.NewQuantity(
			int64(0),
			resource.BinarySI),
	}
	return c
}

func (cm *containerManagerStub) GetPluginRegistrationHandler() cache.PluginHandler {
	return nil
}

func (cm *containerManagerStub) GetDevicePluginResourceCapacity() (v1.ResourceList, v1.ResourceList, []string) {
	return nil, nil, []string{}
}

func (cm *containerManagerStub) NewPodContainerManager() PodContainerManager {
	return &podContainerManagerStub{}
}

func (cm *containerManagerStub) GetResources(pod *v1.Pod, container *v1.Container) (*kubecontainer.RunContainerOptions, error) {
	return &kubecontainer.RunContainerOptions{}, nil
}

func (cm *containerManagerStub) UpdatePluginResources(*framework.NodeInfo, *lifecycle.PodAdmitAttributes) error {
	return nil
}

func (cm *containerManagerStub) InternalContainerLifecycle() InternalContainerLifecycle {
	return &internalContainerLifecycleImpl{cpumanager.NewFakeManager(), topologymanager.NewFakeManager()}
}

func (cm *containerManagerStub) GetPodCgroupRoot() string {
	return ""
}

func (cm *containerManagerStub) GetDevices(_, _ string) []*podresourcesapi.ContainerDevices {
	return nil
}

func (cm *containerManagerStub) ShouldResetExtendedResourceCapacity() bool {
	return cm.shouldResetExtendedResourceCapacity
}

func (cm *containerManagerStub) GetAllocateResourcesPodAdmitHandler() lifecycle.PodAdmitHandler {
	return topologymanager.NewFakeManager()
}

func (cm *containerManagerStub) UpdateAllocatedDevices() {
	return
}

func NewStubContainerManager() ContainerManager {
	return &containerManagerStub{shouldResetExtendedResourceCapacity: false}
}

func NewStubContainerManagerWithExtendedResource(shouldResetExtendedResourceCapacity bool) ContainerManager {
	return &containerManagerStub{shouldResetExtendedResourceCapacity: shouldResetExtendedResourceCapacity}
}
