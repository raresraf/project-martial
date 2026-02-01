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

// Package aws provides functionalities for interacting with AWS EC2 instances within the Kubernetes cloud provider context.
// This includes parsing various instance ID formats, describing instances using the EC2 API, and caching instance details
// to optimize API calls and reduce the load on the AWS API.
// Architectural Intent: To abstract AWS-specific EC2 instance management details, offering a consistent interface
// for Kubernetes components (like the cloud controller manager) that need to identify, query, or manage worker nodes,
// which are typically EC2 instances in an AWS environment. It emphasizes efficient API usage through a caching mechanism.
package aws

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"sync"
	"time"
)

// awsInstanceID represents the ID of the instance in the AWS API, e.g. i-12345678
// The "traditional" format is "i-12345678"
// A new longer format is also being introduced: "i-12345678abcdef01"
// We should not assume anything about the length or format, though it seems
// reasonable to assume that instances will continue to start with "i-".
type awsInstanceID string

func (i awsInstanceID) awsString() *string {
	return aws.String(string(i))
}

// kubernetesInstanceID represents the id for an instance in the kubernetes API;
// the following form
//  * aws:///<zone>/<awsInstanceId>
//  * aws:////<awsInstanceId>
//  * <awsInstanceId>
type kubernetesInstanceID string

// mapToAWSInstanceID extracts the awsInstanceID from the kubernetesInstanceID
func (name kubernetesInstanceID) mapToAWSInstanceID() (awsInstanceID, error) {
	s := string(name)

	if !strings.HasPrefix(s, "aws://") {
		// Assume a bare aws volume id (vol-1234...)
		// Build a URL with an empty host (AZ)
		s = "aws://" + "/" + "/" + s
	}
	url, err := url.Parse(s)
	if err != nil {
		return "", fmt.Errorf("Invalid instance name (%s): %v", name, err)
	}
	if url.Scheme != "aws" {
		return "", fmt.Errorf("Invalid scheme for AWS instance (%s)", name)
	}

	awsID := ""
	tokens := strings.Split(strings.Trim(url.Path, "/"), "/")
	if len(tokens) == 1 {
		// instanceId
		awsID = tokens[0]
	} else if len(tokens) == 2 {
		// az/instanceId
		awsID = tokens[1]
	}

	// We sanity check the resulting volume; the two known formats are
	// i-12345678 and i-12345678abcdef01
	// TODO: Regex match?
	if awsID == "" || strings.Contains(awsID, "/") || !strings.HasPrefix(awsID, "i-") {
		return "", fmt.Errorf("Invalid format for AWS instance (%s)", name)
	}

	return awsInstanceID(awsID), nil
}

// mapToAWSInstanceID extracts the awsInstanceIDs from the Nodes, returning an error if a Node cannot be mapped
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

// mapToAWSInstanceIDsTolerant extracts the awsInstanceIDs from the Nodes, skipping Nodes that cannot be mapped
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

// describeInstance fetches the full information about a specific EC2 instance from the AWS API.
// @param ec2Client The EC2 service client for making API calls.
// @param instanceID The AWS instance ID to describe.
// @return A pointer to the ec2.Instance object or an error.
func describeInstance(ec2Client EC2, instanceID awsInstanceID) (*ec2.Instance, error) {
	request := &ec2.DescribeInstancesInput{
		InstanceIds: []*string{instanceID.awsString()},
	}

	// Functional Utility: Make the DescribeInstances API call.
	instances, err := ec2Client.DescribeInstances(request)
	// Conditional Logic: Handle API call errors.
	if err != nil {
		return nil, err
	}
	// Conditional Logic: Check if no instances were found within the reservations.
	// Precondition: Expects exactly one instance matching the provided ID.
	if len(instances.Reservations) == 0 || len(instances.Reservations[0].Instances) == 0 {
        return nil, fmt.Errorf("no instances found for instance: %s", instanceID)
    }
	// Conditional Logic: Check if multiple instances were found for a single ID, which is unexpected.
	if len(instances.Reservations[0].Instances) > 1 {
		return nil, fmt.Errorf("multiple instances found for instance: %s", instanceID)
	}
	// Functional Utility: Return the single found instance.
	return instances.Reservations[0].Instances[0], nil
}

// instanceCache manages a cache of EC2 instance descriptions to reduce redundant AWS API calls.
// Architectural Intent: Improves performance and reduces API rate limit issues by storing
// recently fetched instance information, acting as a local repository for instance details.
type instanceCache struct {
	// TODO: Get rid of this field, send all calls through the instanceCache
	// cloud holds a reference to the main Cloud provider implementation, used for making actual AWS API calls.
	cloud *Cloud

	// mutex protects concurrent access to the cache snapshot, ensuring thread safety.
	mutex    sync.Mutex
	// snapshot stores the last fetched allInstancesSnapshot, which contains instance data and a timestamp.
	snapshot *allInstancesSnapshot
}

// describeAllInstancesUncached fetches full information about all relevant EC2 instances directly from the AWS API,
// bypassing any cache. It then creates and stores a new snapshot in the cache, under mutex protection.
// @return A pointer to an allInstancesSnapshot or an error.
func (c *instanceCache) describeAllInstancesUncached() (*allInstancesSnapshot, error) {
	now := time.Now()

	glog.V(4).Infof("EC2 DescribeInstances - fetching all instances")

	filters := []*ec2.Filter{} // Block Logic: Currently no filters are applied, meaning all instances are fetched.
	// Functional Utility: Call the cloud provider's describeInstances method to fetch all instances.
	instances, err := c.cloud.describeInstances(filters)
	if err != nil {
		return nil, err
	}

	m := make(map[awsInstanceID]*ec2.Instance)
	// Block Logic: Populate a map with instance IDs as keys and EC2 instance objects as values.
	// Invariant: Each instance in the AWS response is added to the map.
	for _, i := range instances {
		id := awsInstanceID(aws.StringValue(i.InstanceId))
		m[id] = i
	}

	snapshot := &allInstancesSnapshot{now, m}

	// Block Logic: Protect concurrent access to the cache snapshot.
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Conditional Logic: Check if the newly fetched snapshot is older than the current cached snapshot.
	// This can happen if another goroutine has updated the cache more recently.
	if c.snapshot != nil && snapshot.olderThan(c.snapshot) {
		// If this happens a lot, we could run this function in a mutex and only return one result
		glog.Infof("Not caching concurrent AWS DescribeInstances results")
	} else {
		// Block Logic: Update the cache with the new, fresher snapshot.
		c.snapshot = snapshot
	}

	return snapshot, nil
}

// cacheCriteria defines the conditions under which a cached instance snapshot is considered valid and usable.
// Architectural Intent: Provides flexibility in cache usage by allowing callers to specify freshness and content requirements.
type cacheCriteria struct {
	// MaxAge indicates the maximum allowable age for a cached snapshot.
	// A value of 0 means the age constraint is ignored, effectively making the cache never expire due to age.
	MaxAge time.Duration

	// HasInstances is a list of awsInstanceIDs that *must* be present in the cached snapshot
	// for it to be considered valid. If any of these instances are missing from the snapshot,
	// the entire cached snapshot is deemed invalid and a re-fetch is triggered.
	HasInstances []awsInstanceID
}

// describeAllInstancesCached returns all instances, utilizing cached results if they meet the specified criteria.
// If the cache is stale or does not meet the criteria, it triggers an uncached fetch to refresh the data.
// @param criteria The cacheCriteria that must be satisfied for the cached data to be used.
// @return A pointer to an allInstancesSnapshot or an error.
func (c *instanceCache) describeAllInstancesCached(criteria cacheCriteria) (*allInstancesSnapshot, error) {
	var err error
	snapshot := c.getSnapshot() // Functional Utility: Attempt to retrieve the current snapshot from cache.
	// Conditional Logic: Check if an existing snapshot is available and if it meets the specified criteria.
	// If criteria are not met, the snapshot is invalidated.
	if snapshot != nil && !snapshot.MeetsCriteria(criteria) {
		snapshot = nil // Invalidate snapshot if criteria are not met.
	}

	// Conditional Logic: If no valid snapshot is available (either initially nil or invalidated),
	// perform an uncached fetch to get fresh data.
	if snapshot == nil {
		snapshot, err = c.describeAllInstancesUncached()
		if err != nil {
			return nil, err
		}
	} else {
		// Invariant: A valid cached snapshot was found and used.
		glog.V(6).Infof("EC2 DescribeInstances - using cached results")
	}

	return snapshot, nil
}

// getSnapshot safely retrieves the current cached allInstancesSnapshot.
// Access to the snapshot is protected by a mutex to ensure thread safety.
// @return A pointer to the allInstancesSnapshot, or nil if no snapshot is currently cached.
func (c *instanceCache) getSnapshot() *allInstancesSnapshot {
	// Block Logic: Protect concurrent access to the cache snapshot.
	c.mutex.Lock()
	defer c.mutex.Unlock()

	return c.snapshot
}

// olderThan is a helper method that compares the timestamp of the current snapshot with another snapshot.
// @param other The other allInstancesSnapshot to compare against.
// @return True if the current snapshot's timestamp is strictly older than the other snapshot's timestamp, false otherwise.
func (s *allInstancesSnapshot) olderThan(other *allInstancesSnapshot) bool {
	// After() is technically broken by time changes until we have monotonic time
	return other.timestamp.After(s.timestamp)
}

// MeetsCriteria checks if the allInstancesSnapshot satisfies the given cacheCriteria.
// This method evaluates both MaxAge and HasInstances criteria to determine snapshot validity.
// @param criteria The cacheCriteria to evaluate against.
// @return True if all criteria are met, false otherwise.
func (s *allInstancesSnapshot) MeetsCriteria(criteria cacheCriteria) bool {
	// Conditional Logic: Check MaxAge criteria if specified (greater than 0).
	if criteria.MaxAge > 0 {
		// Sub() is technically broken by time changes until we have monotonic time
		now := time.Now()
		// If the snapshot's age exceeds the MaxAge, it fails the criteria.
		if now.Sub(s.timestamp) > criteria.MaxAge {
			glog.V(6).Infof("instanceCache snapshot cannot be used as is older than MaxAge=%s", criteria.MaxAge)
			return false
		}
	}

	// Conditional Logic: Check HasInstances criteria if specified (list is not empty).
	if len(criteria.HasInstances) != 0 {
		// Invariant: All required instances must be present in the snapshot.
		for _, id := range criteria.HasInstances {
			// If any required instance is missing, the snapshot fails the criteria.
			if nil == s.instances[id] {
				glog.V(6).Infof("instanceCache snapshot cannot be used as does not contain instance %s", id)
				return false
			}
		}
	}

	return true
}

// allInstancesSnapshot holds the results from querying for all instances,
// along with the timestamp for cache-invalidation purposes.
// Architectural Intent: Provides an immutable view of EC2 instances at a specific point in time,
// facilitating efficient caching strategies by bundling data with its freshness metadata.
type allInstancesSnapshot struct {
	timestamp time.Time
	instances map[awsInstanceID]*ec2.Instance
}

// FindInstances returns a new map containing only the instances corresponding to the specified IDs from the snapshot.
// If an ID is not found in the snapshot, it is ignored in the returned map, ensuring no nil entries.
// @param ids A slice of awsInstanceID to find.
// @return A map of awsInstanceID to *ec2.Instance for found instances.
func (s *allInstancesSnapshot) FindInstances(ids []awsInstanceID) map[awsInstanceID]*ec2.Instance {
	m := make(map[awsInstanceID]*ec2.Instance)
	// Block Logic: Iterate over the provided IDs and retrieve corresponding instances from the snapshot.
	for _, id := range ids {
		instance := s.instances[id]
		// Conditional Logic: Add instance to the result map only if it exists in the snapshot.
		if instance != nil {
			m[id] = instance
		}
	}
	return m
}
