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

/**
 * @file instances.go
 * @brief EC2 instance identifier resolution and lifecycle caching for AWS cloud provider.
 * 
 * Functional Intent: Manages the mapping between Kubernetes Node objects and 
 * native AWS EC2 instances. It handles identifier parsing from ProviderID URIs 
 * and implements a thread-safe snapshot-based cache for 'DescribeInstances' 
 * metadata to reduce API latency and throttle impact.
 * 
 * Domain: Production Systems, Cloud Infrastructure, Cache Consistency.
 */

package aws

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"regexp"
	"sync"
	"time"
)

// awsInstanceRegMatch - Regex for validating EC2 instance IDs (i-*).
var awsInstanceRegMatch = regexp.MustCompile("^i-[^/]*$")

/**
 * @type awsInstanceID
 * @brief Strong type for native AWS instance identifiers.
 */
type awsInstanceID string

func (i awsInstanceID) awsString() *string {
	return aws.String(string(i))
}

/**
 * @type kubernetesInstanceID
 * @brief Identifier as stored in Kubernetes Node Spec (ProviderID).
 */
type kubernetesInstanceID string

/**
 * mapToAWSInstanceID - Normalizes Kubernetes ProviderIDs to AWS instance strings.
 * 
 * Algorithm: URI tokenization and sanitization.
 * 1. Coerces input into a standard URI scheme if missing.
 * 2. Parses path tokens to resolve optional AZ prefix (e.g. /us-east-1a/i-123).
 * 3. Validates the extracted ID against known EC2 naming patterns.
 */
func (name kubernetesInstanceID) mapToAWSInstanceID() (awsInstanceID, error) {
	s := string(name)

	// Block Logic: URI Normalization.
	if !strings.HasPrefix(s, "aws://") {
		s = "aws://" + "/" + "/" + s
	}
	url, err := url.Parse(s)
	if err != nil {
		return "", fmt.Errorf("Invalid instance name (%s): %v", name, err)
	}
	if url.Scheme != "aws" {
		return "", fmt.Errorf("Invalid scheme for AWS instance (%s)", name)
	}

	// Block Logic: Token extraction.
	awsID := ""
	tokens := strings.Split(strings.Trim(url.Path, "/"), "/")
	if len(tokens) == 1 {
		awsID = tokens[0]
	} else if len(tokens) == 2 {
		awsID = tokens[1]
	}

	// Invariant: Result must be a valid non-empty EC2 instance ID.
	if awsID == "" || !awsInstanceRegMatch.MatchString(awsID) {
		return "", fmt.Errorf("Invalid format for AWS instance (%s)", name)
	}

	return awsInstanceID(awsID), nil
}

/**
 * mapToAWSInstanceIDs - Batch conversion of Nodes to AWS IDs with strict error handling.
 */
func mapToAWSInstanceIDs(nodes []*v1.Node) ([]awsInstanceID, error) {
	var instanceIDs []awsInstanceID
	for _, node := range nodes {
		if node.Spec.ProviderID == "" {
			return nil, fmt.Errorf("node %q did not have ProviderID set", node.Name)
		}
		instanceID, err := kubernetesInstanceID(node.Spec.ProviderID).mapToAWSInstanceID()
		if err != nil {
			return nil, fmt.Errorf("unable to parse ProviderID %q for node %q", node.Spec.ProviderID, node.Name)
		}
		instanceIDs = append(instanceIDs, instanceID)
	}

	return instanceIDs, nil
}

/**
 * mapToAWSInstanceIDsTolerant - Batch conversion that skips nodes with invalid ProviderIDs.
 */
func mapToAWSInstanceIDsTolerant(nodes []*v1.Node) []awsInstanceID {
	var instanceIDs []awsInstanceID
	for _, node := range nodes {
		if node.Spec.ProviderID == "" {
			glog.Warningf("node %q did not have ProviderID set", node.Name)
			continue
		}
		instanceID, err := kubernetesInstanceID(node.Spec.ProviderID).mapToAWSInstanceID()
		if err != nil {
			glog.Warningf("unable to parse ProviderID %q for node %q", node.Spec.ProviderID, node.Name)
			continue
		}
		instanceIDs = append(instanceIDs, instanceID)
	}

	return instanceIDs
}

/**
 * describeInstance - Synchronous high-level wrapper for EC2 DescribeInstances API.
 */
func describeInstance(ec2Client EC2, instanceID awsInstanceID) (*ec2.Instance, error) {
	request := &ec2.DescribeInstancesInput{
		InstanceIds: []*string{instanceID.awsString()},
	}

	instances, err := ec2Client.DescribeInstances(request)
	if err != nil {
		return nil, err
	}
	if len(instances) == 0 {
		return nil, fmt.Errorf("no instances found for instance: %s", instanceID)
	}
	if len(instances) > 1 {
		return nil, fmt.Errorf("multiple instances found for instance: %s", instanceID)
	}
	return instances[0], nil
}

/**
 * @struct instanceCache
 * @brief Thread-safe manager for EC2 metadata snapshots.
 * 
 * Logic: Employs a mutex-protected snapshot to avoid race conditions during 
 * concurrent cache refreshes.
 */
type instanceCache struct {
	cloud *Cloud

	mutex    sync.Mutex
	snapshot *allInstancesSnapshot
}

/**
 * describeAllInstancesUncached - Performs a full sweep of the AWS region to refresh the cache.
 */
func (c *instanceCache) describeAllInstancesUncached() (*allInstancesSnapshot, error) {
	now := time.Now()

	glog.V(4).Infof("EC2 DescribeInstances - fetching all instances")

	filters := []*ec2.Filter{}
	instances, err := c.cloud.describeInstances(filters)
	if err != nil {
		return nil, err
	}

	m := make(map[awsInstanceID]*ec2.Instance)
	for _, i := range instances {
		id := awsInstanceID(aws.StringValue(i.InstanceId))
		m[id] = i
	}

	snapshot := &allInstancesSnapshot{now, m}

	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Logic: Implements a "newest-wins" policy for concurrent cache updates.
	if c.snapshot != nil && snapshot.olderThan(c.snapshot) {
		glog.Infof("Not caching concurrent AWS DescribeInstances results")
	} else {
		c.snapshot = snapshot
	}

	return snapshot, nil
}

/**
 * @struct cacheCriteria
 * @brief Policies for determining the validity of a cached metadata snapshot.
 */
type cacheCriteria struct {
	// MaxAge - Maximum allowed time delta since snapshot creation.
	MaxAge time.Duration

	// HasInstances - Mandatory set of IDs that MUST exist in the snapshot to avoid a forced refresh.
	HasInstances []awsInstanceID
}

/**
 * describeAllInstancesCached - Retrieves metadata using criteria-aware caching.
 * 
 * Logic: Validates the existing snapshot against age and member constraints 
 * before deciding whether to trigger a heavy API call.
 */
func (c *instanceCache) describeAllInstancesCached(criteria cacheCriteria) (*allInstancesSnapshot, error) {
	var err error
	snapshot := c.getSnapshot()
	if snapshot != nil && !snapshot.MeetsCriteria(criteria) {
		snapshot = nil
	}

	if snapshot == nil {
		// Logic: Forced refresh on cache miss or invalidation.
		snapshot, err = c.describeAllInstancesUncached()
		if err != nil {
			return nil, err
		}
	} else {
		glog.V(6).Infof("EC2 DescribeInstances - using cached results")
	}

	return snapshot, nil
}

func (c *instanceCache) getSnapshot() *allInstancesSnapshot {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	return c.snapshot
}

func (s *allInstancesSnapshot) olderThan(other *allInstancesSnapshot) bool {
	return other.timestamp.After(s.timestamp)
}

/**
 * MeetsCriteria - Evaluates snapshot validity against operational constraints.
 * 
 * Invariant: Returns false if the snapshot is expired or missing required data points.
 */
func (s *allInstancesSnapshot) MeetsCriteria(criteria cacheCriteria) bool {
	if criteria.MaxAge > 0 {
		now := time.Now()
		if now.Sub(s.timestamp) > criteria.MaxAge {
			glog.V(6).Infof("instanceCache snapshot cannot be used as is older than MaxAge=%s", criteria.MaxAge)
			return false
		}
	}

	// Block Logic: Membership check.
	// Logic: Prevents stale reads for newly created nodes by ensuring they exist in the snapshot.
	if len(criteria.HasInstances) != 0 {
		for _, id := range criteria.HasInstances {
			if nil == s.instances[id] {
				glog.V(6).Infof("instanceCache snapshot cannot be used as does not contain instance %s", id)
				return false
			}
		}
	}

	return true
}

/**
 * @struct allInstancesSnapshot
 * @brief Immutable (point-in-time) map of AWS instance metadata.
 */
type allInstancesSnapshot struct {
	timestamp time.Time
	instances map[awsInstanceID]*ec2.Instance
}

/**
 * FindInstances - Batch lookup of metadata from the snapshot.
 */
func (s *allInstancesSnapshot) FindInstances(ids []awsInstanceID) map[awsInstanceID]*ec2.Instance {
	m := make(map[awsInstanceID]*ec2.Instance)
	for _, id := range ids {
		instance := s.instances[id]
		if instance != nil {
			m[id] = instance
		}
	}
	return m
}
