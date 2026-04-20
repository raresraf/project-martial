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

/**
 * @ab98584e-337f-4eef-9fbe-bf8a97532bc3/pkg/proxy/metaproxier/meta_proxier.go
 * @brief Dual-stack network proxy dispatcher utilizing the Composite pattern.
 * 
 * Functional Intent: Orchestrates the synchronization of networking rules across 
 * independent IPv4 and IPv6 stacks. It acts as a transparent router for cluster 
 * events (Services, Nodes, EndpointSlices), ensuring that dual-stack configurations 
 * are consistently realized in the underlying data plane by delegating to family-specific 
 * providers.
 * 
 * Domain: Kubernetes Networking, Dual-Stack Proxies, Composite Design Pattern.
 */

package metaproxier

import (
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/config"
)

/**
 * @brief Internal container for the dual backend proxy providers.
 */
type metaProxier struct {
	ipv4Proxier proxy.Provider
	ipv6Proxier proxy.Provider
	config.NoopNodeHandler
}

/**
 * NewMetaProxier - Factory for a unified dual-stack proxy provider.
 * Logic: Aggregates family-specific providers into a single interface compliant 
 * with the standard Kubernetes proxy.Provider.
 */
func NewMetaProxier(ipv4Proxier, ipv6Proxier proxy.Provider) proxy.Provider {
	return proxy.Provider(&metaProxier{
		ipv4Proxier: ipv4Proxier,
		ipv6Proxier: ipv6Proxier,
	})
}

/**
 * Sync - Triggers immediate rule reconciliation for both stacks.
 */
func (proxier *metaProxier) Sync() {
	proxier.ipv4Proxier.Sync()
	proxier.ipv6Proxier.Sync()
}

/**
 * SyncLoop - Orchestrates continuous background rule maintenance.
 * Logic: Parallelizes stack maintenance by launching the IPv6 loop in a 
 * separate goroutine while the IPv4 loop occupies the primary thread.
 */
func (proxier *metaProxier) SyncLoop() {
	go proxier.ipv6Proxier.SyncLoop()
	proxier.ipv4Proxier.SyncLoop()
}

/* --- Service Event Delegation Logic --- */

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

/* --- EndpointSlice Event Routing Logic --- */

/**
 * Block Logic: Type-aware event dispatching.
 * Logic: Routes EndpointSlice updates based on their AddressType (IPv4 vs IPv6), 
 * as EndpointSlices are inherently single-family resources.
 */
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

/* --- Node Event Delegation Logic --- */

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
 * OnServiceCIDRsChanged - Broadcasts subnet reconfigurations to both providers.
 */
func (proxier *metaProxier) OnServiceCIDRsChanged(cidrs []string) {
	proxier.ipv4Proxier.OnServiceCIDRsChanged(cidrs)
	proxier.ipv6Proxier.OnServiceCIDRsChanged(cidrs)
}
