/**
 * @file volumes.go
 * @brief Provides utilities for parsing and validating AWS EBS volume identifiers.
 *
 * @details This file is part of the AWS cloud provider implementation for Kubernetes.
 * It contains helper types and functions to handle the different ways an AWS Elastic
 * Block Store (EBS) volume can be identified within the Kubernetes ecosystem.
 *
 * Problem Statement:
 * A volume can be represented in multiple formats:
 * 1. A raw AWS Volume ID (e.g., `vol-12345678abcdef01`).
 * 2. A Kubernetes-specific URI format (e.g., `aws://us-east-1a/vol-1234...`).
 *
 * This file provides the logic to reliably parse these formats and extract the canonical
 * AWS Volume ID, which is essential for making correct API calls to AWS.
 *
 * Production Systems (Go/TypeScript):
 * In a production Kubernetes cluster on AWS, correctly identifying EBS volumes is fundamental
 * for persistent storage. This code ensures that PersistentVolume claims, pod scheduling,
 * and volume attachment/detachment operations all refer to the correct underlying AWS resource,
 * regardless of the ID format used in the Kubernetes object specification.
 */
/*
Copyright 2016 The Kubernetes Authors.

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

package aws

import (
	"fmt"
	"net/url"
	"regexp"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
)

// awsVolumeRegMatch is a regular expression used to perform a basic sanity check
// on a string to see if it looks like a valid AWS volume ID.
var awsVolumeRegMatch = regexp.MustCompile("^vol-[^/]*$")

// awsVolumeID represents the canonical ID of a volume in the AWS API (e.g., "vol-12345678").
// The format has evolved, so we avoid assumptions about its length.
type awsVolumeID string

// awsString converts the awsVolumeID to a string pointer, which is the format
// expected by the AWS SDK for Go.
func (i awsVolumeID) awsString() *string {
	return aws.String(string(i))
}

// KubernetesVolumeID represents the identifier for a volume as used within the Kubernetes API.
// It can appear in several forms:
//  * aws://<zone>/<awsVolumeId>
//  * aws:///<awsVolumeId>
//  * <awsVolumeId> (a bare AWS volume ID)
type KubernetesVolumeID string

// mapToAWSVolumeID extracts the canonical awsVolumeID from the potentially complex KubernetesVolumeID.
//
// Algorithm:
// 1. Checks if the ID string has the "aws://" prefix.
// 2. If not, it assumes a bare AWS volume ID and normalizes it into a URL format
//    (e.g., "vol-123" becomes "aws:///vol-123").
// 3. It then parses the string as a URL.
// 4. The path component of the URL is extracted, which should contain the volume ID.
// 5. A regex is used to validate that the extracted ID matches the expected "vol-..." format.
func (name KubernetesVolumeID) mapToAWSVolumeID() (awsVolumeID, error) {
	s := string(name)

	// Block Logic: Handle the case of a bare volume ID by normalizing it to a URL.
	if !strings.HasPrefix(s, "aws://") {
		// Assume a bare aws volume id (vol-1234...)
		s = "aws://" + "" + "/" + s
	}
	url, err := url.Parse(s)
	if err != nil {
		return "", fmt.Errorf("Invalid disk name (%s): %v", name, err)
	}
	if url.Scheme != "aws" {
		return "", fmt.Errorf("Invalid scheme for AWS volume (%s)", name)
	}

	awsID := url.Path
	awsID = strings.Trim(awsID, "/")

	// Pre-condition: Sanity check the extracted ID against the expected format.
	if !awsVolumeRegMatch.MatchString(awsID) {
		return "", fmt.Errorf("Invalid format for AWS volume (%s)", name)
	}

	return awsVolumeID(awsID), nil
}
