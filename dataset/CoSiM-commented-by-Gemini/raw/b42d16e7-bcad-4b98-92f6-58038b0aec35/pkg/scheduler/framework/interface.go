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

// Package framework defines the extensible scheduling framework for Kubernetes.
// It provides a plugin-based architecture where various scheduling stages
// (extension points) can be augmented with custom logic.
//
// The framework handles the orchestration of these plugins, managing state
// across the scheduling cycle (CycleState) and providing access to cluster
// snapshots and clients (Handle).
package framework

import (
	"context"
	"errors"
	"math"
	"strings"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/events"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/internal/parallelize"
)

// NodeScoreList represents a collection of scores assigned to nodes by a Score plugin.
// It is used to rank nodes during the scoring phase of scheduling.
type NodeScoreList []NodeScore

// NodeScore encapsulates the quantitative fitness of a specific node for a given pod.
type NodeScore struct {
	Name  string
	Score int64
}

// PluginToNodeScores maps individual plugin names to their respective scoring outputs.
// This allows the framework to aggregate and weight scores from multiple plugins.
type PluginToNodeScores map[string]NodeScoreList

// NodeToStatusMap correlates node names with their scheduling status,
// typically used to track why nodes were filtered out.
type NodeToStatusMap map[string]*Status

// Code represents the semantic result of a plugin execution.
// It guides the framework on how to proceed with the scheduling cycle.
type Code int

// Predefined status codes that define the operational flow control.
const (
	// Success indicates the plugin executed without error and the pod is
	// considered compatible with the current context.
	Success Code = iota

	// Error denotes an internal failure within the plugin (e.g., failed I/O,
	// invariant violation) that prevents a deterministic scheduling decision.
	Error

	// Unschedulable signals that the pod cannot fit on the node(s) given
	// the current cluster state, but might fit if other pods are preempted.
	Unschedulable

	// UnschedulableAndUnresolvable indicates a hard constraint violation
	// where even preemption would not make the pod schedulable.
	UnschedulableAndUnresolvable

	// Wait is returned by Permit plugins to suspend the scheduling cycle
	// for a pod until a specific condition is met or a timeout occurs.
	Wait

	// Skip is used by Bind plugins to indicate they decline to handle the
	// binding for this pod, allowing subsequent Bind plugins to try.
	Skip
)

// codes provides a human-readable mapping for status codes.
var codes = []string{"Success", "Error", "Unschedulable", "UnschedulableAndUnresolvable", "Wait", "Skip"}

// statusPrecedence determines the "strongest" status when merging multiple results.
// Higher values take priority (e.g., Error overrides Unschedulable).
var statusPrecedence = map[Code]int{
	Error:                        3,
	UnschedulableAndUnresolvable: 2,
	Unschedulable:                1,
	Success:                      -1,
}

func (c Code) String() string {
	return codes[c]
}

const (
	// MaxNodeScore is the normalized ceiling for individual plugin scores.
	MaxNodeScore int64 = 100

	// MinNodeScore is the normalized floor for individual plugin scores.
	MinNodeScore int64 = 0

	// MaxTotalScore is the theoretical maximum for aggregated weighted scores.
	MaxTotalScore int64 = math.MaxInt64
)

// Status represents the result of a plugin's evaluation, including metadata
// about why a particular decision was made.
type Status struct {
	code    Code
	reasons []string
	err     error
	// failedPlugin identifies which plugin originated a non-success status.
	failedPlugin string
}

// Code returns the semantic result code.
func (s *Status) Code() Code {
	if s == nil {
		return Success
	}
	return s.code
}

// Message returns a human-readable explanation of the status reasons.
func (s *Status) Message() string {
	if s == nil {
		return ""
	}
	return strings.Join(s.reasons, ", ")
}

// SetFailedPlugin marks the plugin responsible for this status.
func (s *Status) SetFailedPlugin(plugin string) {
	s.failedPlugin = plugin
}

// WithFailedPlugin is a fluent API to set the failed plugin.
func (s *Status) WithFailedPlugin(plugin string) *Status {
	s.SetFailedPlugin(plugin)
	return s
}

// FailedPlugin retrieves the name of the plugin that failed.
func (s *Status) FailedPlugin() string {
	return s.failedPlugin
}

// Reasons returns the list of diagnostic messages.
func (s *Status) Reasons() []string {
	return s.reasons
}

// AppendReason adds a diagnostic message to the status.
func (s *Status) AppendReason(reason string) {
	s.reasons = append(s.reasons, reason)
}

// IsSuccess checks if the status represents a successful operation.
func (s *Status) IsSuccess() bool {
	return s.Code() == Success
}

// IsUnschedulable checks if the status indicates the pod cannot be scheduled.
func (s *Status) IsUnschedulable() bool {
	code := s.Code()
	return code == Unschedulable || code == UnschedulableAndUnresolvable
}

// AsError converts a non-success status into a standard Go error.
func (s *Status) AsError() error {
	if s.IsSuccess() {
		return nil
	}
	if s.err != nil {
		return s.err
	}
	return errors.New(s.Message())
}

// Equal implements a deep equality check for Status objects, used in testing.
func (s *Status) Equal(x *Status) bool {
	if s == nil || x == nil {
		return s.IsSuccess() && x.IsSuccess()
	}
	if s.code != x.code {
		return false
	}
	if s.code == Error {
		return cmp.Equal(s.err, x.err, cmpopts.EquateErrors())
	}
	return cmp.Equal(s.reasons, x.reasons)
}

// NewStatus constructs a Status with a specific code and optional diagnostic reasons.
func NewStatus(code Code, reasons ...string) *Status {
	s := &Status{
		code:    code,
		reasons: reasons,
	}
	if code == Error {
		s.err = errors.New(s.Message())
	}
	return s
}

// AsStatus creates an Error status from a Go error.
func AsStatus(err error) *Status {
	return &Status{
		code:    Error,
		reasons: []string{err.Error()},
		err:     err,
	}
}

// PluginToStatus maps plugin names to their respective execution outcomes.
type PluginToStatus map[string]*Status

// Merge aggregates multiple statuses into a single representative Status.
// It follows established precedence rules: Error > UnschedulableAndUnresolvable > Unschedulable.
func (p PluginToStatus) Merge() *Status {
	if len(p) == 0 {
		return nil
	}

	finalStatus := NewStatus(Success)
	for _, s := range p {
		if s.Code() == Error {
			finalStatus.err = s.AsError()
		}
		if statusPrecedence[s.Code()] > statusPrecedence[finalStatus.code] {
			finalStatus.code = s.Code()
			finalStatus.failedPlugin = s.FailedPlugin()
		}

		for _, r := range s.reasons {
			finalStatus.AppendReason(r)
		}
	}

	return finalStatus
}

// WaitingPod represents a pod that has been suspended in the Permit phase.
// It allows plugins to asynchronously allow or reject a pod's scheduling decision.
type WaitingPod interface {
	// GetPod returns the underlying Pod object.
	GetPod() *v1.Pod
	// GetPendingPlugins lists the Permit plugins that have yet to signal approval for this pod.
	GetPendingPlugins() []string
	// Allow marks the pod as approved by a specific plugin. If all pending plugins
	// have called Allow, the pod is unblocked and proceeds to the Bind phase.
	Allow(pluginName string)
	// Reject terminates the pod's scheduling attempt, marking it as unschedulable.
	Reject(pluginName, msg string)
}

// Plugin is the foundational interface for all scheduling framework components.
// Every custom scheduling logic must implement this interface to be registered.
type Plugin interface {
	// Name returns the unique identifier of the plugin.
	Name() string
}

// LessFunc defines the signature for custom pod prioritization logic within the scheduling queue.
type LessFunc func(podInfo1, podInfo2 *QueuedPodInfo) bool

// QueueSortPlugin defines the contract for ordering pods in the scheduling queue.
// Only one QueueSort plugin can be active in a given profile.
type QueueSortPlugin interface {
	Plugin
	// Less determines the relative priority of two pods. Return true if podInfo1 should
	// be processed before podInfo2.
	Less(*QueuedPodInfo, *QueuedPodInfo) bool
}

// EnqueueExtensions allows plugins to inform the scheduler about which cluster events
// should trigger a retry for unschedulable pods.
type EnqueueExtensions interface {
	// EventsToRegister returns the set of cluster-level changes (e.g., Node added, Pod deleted)
	// that this plugin cares about.
	EventsToRegister() []ClusterEvent
}

// PreFilterExtensions provides hooks for maintaining incremental state during
// high-frequency pod additions/removals (e.g., during preemption analysis).
type PreFilterExtensions interface {
	// AddPod updates the plugin's internal pre-calculated state with a new pod.
	AddPod(ctx context.Context, state *CycleState, podToSchedule *v1.Pod, podInfoToAdd *PodInfo, nodeInfo *NodeInfo) *Status
	// RemovePod subtracts a pod's impact from the plugin's pre-calculated state.
	RemovePod(ctx context.Context, state *CycleState, podToSchedule *v1.Pod, podInfoToRemove *PodInfo, nodeInfo *NodeInfo) *Status
}

// PreFilterPlugin is invoked at the start of the scheduling cycle to perform
// expensive computations or state setup that can be shared across multiple Filter calls.
type PreFilterPlugin interface {
	Plugin
	// PreFilter performs initial validation or pre-computation. A non-success status
	// here aborts the entire scheduling cycle for the pod.
	PreFilter(ctx context.Context, state *CycleState, p *v1.Pod) *Status
	// PreFilterExtensions returns optional incremental state update handlers.
	PreFilterExtensions() PreFilterExtensions
}

// FilterPlugin implements the "Predicate" logic, determining if a node can
// physically or logically host a specific pod.
type FilterPlugin interface {
	Plugin
	// Filter evaluates a single node for compatibility with the pod.
	// It should be side-effect free and ideally read-only against the NodeInfo.
	Filter(ctx context.Context, state *CycleState, pod *v1.Pod, nodeInfo *NodeInfo) *Status
}

// PostFilterPlugin is a fallback mechanism called when no nodes pass the Filter phase.
// It is primarily used for preemption logic to make room for the pod.
type PostFilterPlugin interface {
	Plugin
	// PostFilter attempts to resolve the unschedulable state of a pod.
	// It may suggest a node for preemption (via PostFilterResult).
	PostFilter(ctx context.Context, state *CycleState, pod *v1.Pod, filteredNodeStatusMap NodeToStatusMap) (*PostFilterResult, *Status)
}

// PreScorePlugin provides an extension point for plugins to perform data
// gathering before the scoring phase begins for the set of filtered nodes.
type PreScorePlugin interface {
	Plugin
	// PreScore initializes scoring state or prepares metrics for the upcoming Score calls.
	PreScore(ctx context.Context, state *CycleState, pod *v1.Pod, nodes []*v1.Node) *Status
}

// ScoreExtensions enables normalization of scores across different plugins,
// ensuring that diverse scoring metrics can be combined fairly.
type ScoreExtensions interface {
	// NormalizeScore transforms raw scores into a standard range (0-100).
	NormalizeScore(ctx context.Context, state *CycleState, p *v1.Pod, scores NodeScoreList) *Status
}

// ScorePlugin assigns a numerical rank to a node, indicating its fitness for a pod.
type ScorePlugin interface {
	Plugin
	// Score calculates the fitness level of a specific node.
	// High scores indicate better suitability.
	Score(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string) (int64, *Status)

	// ScoreExtensions provides access to score normalization logic.
	ScoreExtensions() ScoreExtensions
}

// ReservePlugin manages the "Assume" state, allowing plugins to tentatively
// claim resources on a node before the binding is confirmed.
type ReservePlugin interface {
	Plugin
	// Reserve is called when a pod is assigned to a node in the internal cache.
	// It allows plugins to update local resource accounting.
	Reserve(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string) *Status
	// Unreserve rolls back the reservation if a subsequent step fails.
	// Must be idempotent.
	Unreserve(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string)
}

// PreBindPlugin performs pre-flight checks or setup immediately before
// the pod is committed to a node (e.g., provisioning volumes).
type PreBindPlugin interface {
	Plugin
	// PreBind executes binding-specific preparations. Failure here prevents the bind.
	PreBind(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string) *Status
}

// PostBindPlugin is an informational hook called after a pod is successfully bound.
// Useful for observability and cleanup.
type PostBindPlugin interface {
	Plugin
	// PostBind is called after the Bind operation succeeds.
	PostBind(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string)
}

// PermitPlugin allows for "late-binding" decisions, enabling pods to wait
// for conditions (like other pods being scheduled) before proceeding.
type PermitPlugin interface {
	Plugin
	// Permit can Allow, Reject, or Wait. If it waits, it returns a timeout.
	Permit(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string) (*Status, time.Duration)
}

// BindPlugin implements the actual API call to associate a pod with a node.
// Custom Bind plugins can override the default Kubernetes binding mechanism.
type BindPlugin interface {
	Plugin
	// Bind performs the final association. The first plugin to return Success
	// "wins" and subsequent Bind plugins are skipped.
	Bind(ctx context.Context, state *CycleState, p *v1.Pod, nodeName string) *Status
}

// Framework orchestrates the execution of plugins across the scheduling lifecycle.
// It acts as the central dispatcher, ensuring plugins are called at the correct
// extension points with the appropriate context.
type Framework interface {
	Handle
	// QueueSortFunc returns the active sorting logic for the scheduling queue.
	QueueSortFunc() LessFunc

	// RunPreFilterPlugins executes all registered PreFilter logic.
	RunPreFilterPlugins(ctx context.Context, state *CycleState, pod *v1.Pod) *Status

	// RunFilterPlugins evaluates a node against all registered Filter plugins.
	// It returns a map of outcomes, allowing for detailed failure analysis.
	RunFilterPlugins(ctx context.Context, state *CycleState, pod *v1.Pod, nodeInfo *NodeInfo) PluginToStatus

	// RunPostFilterPlugins executes fallback logic when primary scheduling fails.
	RunPostFilterPlugins(ctx context.Context, state *CycleState, pod *v1.Pod, filteredNodeStatusMap NodeToStatusMap) (*PostFilterResult, *Status)

	// RunPreFilterExtensionAddPod notifies plugins of a pod addition during state-sensitive analysis.
	RunPreFilterExtensionAddPod(ctx context.Context, state *CycleState, podToSchedule *v1.Pod, podInfoToAdd *PodInfo, nodeInfo *NodeInfo) *Status

	// RunPreFilterExtensionRemovePod notifies plugins of a pod removal during state-sensitive analysis.
	RunPreFilterExtensionRemovePod(ctx context.Context, state *CycleState, podToSchedule *v1.Pod, podInfoToRemove *PodInfo, nodeInfo *NodeInfo) *Status

	// RunPreScorePlugins prepares the scoring environment.
	RunPreScorePlugins(ctx context.Context, state *CycleState, pod *v1.Pod, nodes []*v1.Node) *Status

	// RunScorePlugins gathers scores from all registered plugins for the set of candidate nodes.
	RunScorePlugins(ctx context.Context, state *CycleState, pod *v1.Pod, nodes []*v1.Node) (PluginToNodeScores, *Status)

	// RunPreBindPlugins executes final checks before binding.
	RunPreBindPlugins(ctx context.Context, state *CycleState, pod *v1.Pod, nodeName string) *Status

	// RunPostBindPlugins triggers informational hooks after a successful bind.
	RunPostBindPlugins(ctx context.Context, state *CycleState, pod *v1.Pod, nodeName string)

	// RunReservePluginsReserve tentatively claims resources for a pod on a node.
	RunReservePluginsReserve(ctx context.Context, state *CycleState, pod *v1.Pod, nodeName string) *Status

	// RunReservePluginsUnreserve releases tentative resource claims.
	RunReservePluginsUnreserve(ctx context.Context, state *CycleState, pod *v1.Pod, nodeName string)

	// RunPermitPlugins manages pod suspension and conditional scheduling.
	RunPermitPlugins(ctx context.Context, state *CycleState, pod *v1.Pod, nodeName string) *Status

	// WaitOnPermit blocks the scheduling cycle for a "waiting" pod until it is released or times out.
	WaitOnPermit(ctx context.Context, pod *v1.Pod) *Status

	// RunBindPlugins executes the actual binding of a pod to a node.
	RunBindPlugins(ctx context.Context, state *CycleState, pod *v1.Pod, nodeName string) *Status

	// HasFilterPlugins indicates if the framework has any active filter logic.
	HasFilterPlugins() bool

	// HasPostFilterPlugins indicates if the framework has any active fallback logic.
	HasPostFilterPlugins() bool

	// HasScorePlugins indicates if the framework has any active scoring logic.
	HasScorePlugins() bool

	// ListPlugins provides a detailed manifest of all enabled plugins per extension point.
	ListPlugins() map[string][]config.Plugin

	// ProfileName returns the scheduling profile this framework instance represents.
	ProfileName() string
}

// Handle provides plugins with access to cluster state and scheduling utilities.
// It is the primary interface through which plugins interact with the scheduler core.
type Handle interface {
	// PodNominator provides access to nominated pod tracking (for preemption).
	PodNominator
	// PluginsRunner allows executing specific plugin sets (used in complex logic like preemption).
	PluginsRunner
	// SnapshotSharedLister returns a read-only view of the cluster state (Nodes/Pods)
	// that is consistent for the duration of a scheduling cycle.
	SnapshotSharedLister() SharedLister

	// IterateOverWaitingPods allows inspecting all pods currently in the "Wait" state.
	IterateOverWaitingPods(callback func(WaitingPod))

	// GetWaitingPod retrieves a specific waiting pod by UID.
	GetWaitingPod(uid types.UID) WaitingPod

	// RejectWaitingPod manually rejects a pod currently in the "Wait" state.
	RejectWaitingPod(uid types.UID)

	// ClientSet provides a standard Kubernetes API client.
	ClientSet() clientset.Interface

	// KubeConfig returns the configuration used to connect to the Kubernetes API.
	KubeConfig() *restclient.Config

	// EventRecorder provides a mechanism to publish Kubernetes Events for observability.
	EventRecorder() events.EventRecorder

	// SharedInformerFactory provides access to cached cluster data.
	SharedInformerFactory() informers.SharedInformerFactory

	// RunFilterPluginsWithNominatedPods evaluates node fit while accounting for pods
	// that are expected to be scheduled there soon.
	RunFilterPluginsWithNominatedPods(ctx context.Context, state *CycleState, pod *v1.Pod, info *NodeInfo) *Status

	// Extenders returns legacy scheduler extenders, if configured.
	Extenders() []Extender

	// Parallelizer provides a managed pool for concurrent execution of scheduling tasks.
	Parallelizer() parallelize.Parallelizer
}

// PostFilterResult encapsulates the outcome of a PostFilter plugin execution.
// It typically contains suggestions for making a pod schedulable, such as a nominated node.
type PostFilterResult struct {
	NominatedNodeName string
}

// PodNominator manages pods that have been "nominated" to run on specific nodes,
// usually as a result of a preemption decision.
type PodNominator interface {
	// AddNominatedPod registers a pod's nomination for a specific node.
	AddNominatedPod(pod *PodInfo, nodeName string)
	// DeleteNominatedPodIfExists removes a pod's nomination from the internal tracking.
	DeleteNominatedPodIfExists(pod *v1.Pod)
	// UpdateNominatedPod refreshes nomination details when a pod object is updated.
	UpdateNominatedPod(oldPod *v1.Pod, newPodInfo *PodInfo)
	// NominatedPodsForNode retrieves all pods nominated to run on a given node.
	NominatedPodsForNode(nodeName string) []*PodInfo
}

// PluginsRunner provides a subset of framework capabilities focused on executing
// specific plugin sets. This is vital for complex scheduling logic (like preemption)
// that needs to "simulate" scheduling cycles for multiple nodes.
type PluginsRunner interface {
	// RunPreScorePlugins runs the PreScore phase for a specific subset of nodes.
	RunPreScorePlugins(context.Context, *CycleState, *v1.Pod, []*v1.Node) *Status
	// RunScorePlugins runs the Score phase for a specific subset of nodes.
	RunScorePlugins(context.Context, *CycleState, *v1.Pod, []*v1.Node) (PluginToNodeScores, *Status)
	// RunFilterPlugins runs the Filter phase for a specific node.
	RunFilterPlugins(context.Context, *CycleState, *v1.Pod, *NodeInfo) PluginToStatus
	// RunPreFilterExtensionAddPod simulates adding a pod to a node's state.
	RunPreFilterExtensionAddPod(ctx context.Context, state *CycleState, podToSchedule *v1.Pod, podInfoToAdd *PodInfo, nodeInfo *NodeInfo) *Status
	// RunPreFilterExtensionRemovePod simulates removing a pod from a node's state.
	RunPreFilterExtensionRemovePod(ctx context.Context, state *CycleState, podToSchedule *v1.Pod, podInfoToRemove *PodInfo, nodeInfo *NodeInfo) *Status
}
