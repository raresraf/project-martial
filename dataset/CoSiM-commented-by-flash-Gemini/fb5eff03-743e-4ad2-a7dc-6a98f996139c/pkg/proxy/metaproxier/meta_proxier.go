/*
@file meta_proxier.go
@brief Dual-stack network proxy dispatcher for Kubernetes.
This module implements a "Meta Proxier" using the Composite design pattern. It 
wraps two independent proxy providers (IPv4 and IPv6) and orchestrates the 
dispatching of cluster events (Services, Nodes, EndpointSlices) to both stacks. 
It ensures that dual-stack networking state is consistently maintained across 
different address families within the Kubernetes data plane.

Domain: Cloud Orchestration, Kubernetes Networking, Dual-Stack Proxies.
*/

/*
Copyright 2019 The Kubernetes Authors.

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

package metaproxier

import (
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"
)

/**
 * @type metaProxier
 * @brief Internal implementation of the composite proxy provider.
 * Functional Utility: Decouples high-level event notification from family-specific 
 * implementation details by wrapping dual Provider instances.
 */
type metaProxier struct {
	ipv4Proxier proxy.Provider // Backend provider for IPv4 traffic rules.
	ipv6Proxier proxy.Provider // Backend provider for IPv6 traffic rules.
}

/**
 * @brief Factory function for a dual-stack meta-proxier.
 * Architectural Intent: Returns an opaque interface that handles multi-stack 
 * rule synchronization transparently to the caller.
 * @param ipv4Proxier The provider instance for IPv4 addresses.
 * @param ipv6Proxier The provider instance for IPv6 addresses.
 * @return Integrated proxy.Provider implementation.
 */
func NewMetaProxier(ipv4Proxier, ipv6Proxier proxy.Provider) proxy.Provider {
	return proxy.Provider(&metaProxier{
		ipv4Proxier: ipv4Proxier,
		ipv6Proxier: ipv6Proxier,
	})
}

/**
 * @brief Triggers an immediate state-to-rule synchronization.
 * Logic: Sequentially executes flush/synchronization for both protocol families.
 */
func (proxier *metaProxier) Sync() {
	proxier.ipv4Proxier.Sync()
	proxier.ipv6Proxier.Sync()
}

/**
 * @brief Orchestrates periodic rule maintenance.
 * Logic: Launches the IPv6 maintenance loop in a dedicated goroutine while 
 * blocking the current thread on the IPv4 loop. This ensures both stacks 
 * are continuously serviced.
 */
func (proxier *metaProxier) SyncLoop() {
	go proxier.ipv6Proxier.SyncLoop() // Asynchronous background loop for IPv6.
	proxier.ipv4Proxier.SyncLoop()    // Primary blocking loop for IPv4.
}

/* --- Service Event Handlers --- */

// Logic for Service handlers: Propagates all service-level metadata changes 
// (creation, modification, deletion) to both stacks, as service definitions 
// may contain both address families.

func (proxier *metaProxier) OnServiceAdd(service *v1.Service) {
	proxier.ipv4Proxier.OnServiceAdd(service)
	proxier.ipv6Proxier.OnServiceAdd(service)
}

func (proxier *metaProxier) OnServiceUpdate(oldService, service *v1.Service) {
	proxier.ipv4Proxier.OnServiceUpdate(oldService, service)
	proxier.ipv6Proxier.OnServiceUpdate(oldService, service)
}

func (proxier *metaProxier) OnServiceDelete(service *v1.Service) {
	proxier.ipv4Proxier.OnServiceDelete(service)
	proxier.ipv6Proxier.OnServiceDelete(service)
}

func (proxier *metaProxier) OnServiceSynced() {
	proxier.ipv4Proxier.OnServiceSynced()
	proxier.ipv6Proxier.OnServiceSynced()
}

/* --- EndpointSlice Event Handlers --- */

// Logic for EndpointSlice handlers: Unlike Services, EndpointSlices are 
// scoped to a specific AddressType. The meta-proxier performs a family-check 
// (IPv4 vs IPv6) to route the update to the correct underlying provider.

func (proxier *metaProxier) OnEndpointSliceAdd(endpointSlice *discovery.EndpointSlice) {
	switch endpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		proxier.ipv4Proxier.OnEndpointSliceAdd(endpointSlice)
	case discovery.AddressTypeIPv6:
		proxier.ipv6Proxier.OnEndpointSliceAdd(endpointSlice)
	default:
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", endpointSlice.AddressType)
	}
}

func (proxier *metaProxier) OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice *discovery.EndpointSlice) {
	switch newEndpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		proxier.ipv4Proxier.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
	case discovery.AddressTypeIPv6:
		proxier.ipv6Proxier.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
	default:
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", newEndpointSlice.AddressType)
	}
}

func (proxier *metaProxier) OnEndpointSliceDelete(endpointSlice *discovery.EndpointSlice) {
	switch endpointSlice.AddressType {
	case discovery.AddressTypeIPv4:
		proxier.ipv4Proxier.OnEndpointSliceDelete(endpointSlice)
	case discovery.AddressTypeIPv6:
		proxier.ipv6Proxier.OnEndpointSliceDelete(endpointSlice)
	default:
		klog.ErrorS(nil, "EndpointSlice address type not supported", "addressType", endpointSlice.AddressType)
	}
}

func (proxier *metaProxier) OnEndpointSlicesSynced() {
	proxier.ipv4Proxier.OnEndpointSlicesSynced()
	proxier.ipv6Proxier.OnEndpointSlicesSynced()
}

/* --- Node Event Handlers --- */

// Logic for Node handlers: Nodes participate in the dual-stack fabric. 
// Membership changes are broadcast to both providers to maintain reachability 
// information for cluster egress and ingress.

func (proxier *metaProxier) OnNodeAdd(node *v1.Node) {
	proxier.ipv4Proxier.OnNodeAdd(node)
	proxier.ipv6Proxier.OnNodeAdd(node)
}

func (proxier *metaProxier) OnNodeUpdate(oldNode, node *v1.Node) {
	proxier.ipv4Proxier.OnNodeUpdate(oldNode, node)
	proxier.ipv6Proxier.OnNodeUpdate(oldNode, node)
}

func (proxier *metaProxier) OnNodeDelete(node *v1.Node) {
	proxier.ipv4Proxier.OnNodeDelete(node)
	proxier.ipv6Proxier.OnNodeDelete(node)
}

func (proxier *metaProxier) OnNodeSynced() {
	proxier.ipv4Proxier.OnNodeSynced()
	proxier.ipv6Proxier.OnNodeSynced()
}

/**
 * @brief Notifies providers of cluster-wide CIDR changes.
 */
func (proxier *metaProxier) OnServiceCIDRsChanged(cidrs []string) {
	proxier.ipv4Proxier.OnServiceCIDRsChanged(cidrs)
	proxier.ipv6Proxier.OnServiceCIDRsChanged(cidrs)
}
