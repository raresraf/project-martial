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

/**
 * @file volumes.go
 * @brief Utility functions for AWS EBS volume identifier resolution within Kubernetes.
 * 
 * Functional Intent: Provides a bridge between Kubernetes internal volume references 
 * and native AWS EBS identifiers. It handles URI-style identifiers (which may 
 * encode availability zone data) and sanitizes them into the "vol-*" format 
 * required by the AWS SDK.
 * 
 * Domain: Production Systems, Cloud Providers, Infrastructure Orchestration.
 */

package aws

import (
	"fmt"
	"net/url"
	"regexp"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
)

// awsVolumeRegMatch - Regular expression for validating native EBS volume IDs.
// Supports both 8-character and 17-character hexadecimal formats.
var awsVolumeRegMatch = regexp.MustCompile("^vol-[^/]*$")

/**
 * @type awsVolumeID
 * @brief Native representation of an AWS EBS volume ID.
 */
type awsVolumeID string

func (i awsVolumeID) awsString() *string {
	return aws.String(string(i))
}

/**
 * @type KubernetesVolumeID
 * @brief Abstract representation of a volume within the Kubernetes API.
 * 
 * Supported Formats:
 * 1. Fully qualified: aws://<zone>/<awsVolumeId>
 * 2. Unqualified URI: aws:///<awsVolumeId>
 * 3. Bare ID: <awsVolumeId>
 */
type KubernetesVolumeID string

/**
 * mapToAWSVolumeID - Extracts and validates the native AWS ID from a Kubernetes string.
 * 
 * Algorithm: URI-based identifier normalization.
 * 1. Normalizes input to a pseudo-URL format.
 * 2. Parses the URL to extract the path component.
 * 3. Sanitizes and validates the extracted string against EBS naming conventions.
 */
func (name KubernetesVolumeID) mapToAWSVolumeID() (awsVolumeID, error) {
	s := string(name)

	// Block Logic: Normalization to URI scheme.
	if !strings.HasPrefix(s, "aws://") {
		// Logic: Implicitly convert bare IDs to null-zone URIs for consistent parsing.
		s = "aws://" + "" + "/" + s
	}
	
	url, err := url.Parse(s)
	if err != nil {
		return "", fmt.Errorf("Invalid disk name (%s): %v", name, err)
	}
	
	// Pre-condition: Input must belong to the AWS cloud provider scheme.
	if url.Scheme != "aws" {
		return "", fmt.Errorf("Invalid scheme for AWS volume (%s)", name)
	}

	// Logic: Path extraction from the URI, removing leading/trailing separators.
	awsID := url.Path
	awsID = strings.Trim(awsID, "/")

	// Block Logic: Validation.
	// Invariant: Resulting ID must start with "vol-" to be considered a valid EBS reference.
	if !awsVolumeRegMatch.MatchString(awsID) {
		return "", fmt.Errorf("Invalid format for AWS volume (%s)", name)
	}

	return awsVolumeID(awsID), nil
}
