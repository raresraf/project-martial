/**
 * @6f3cc88f-6a72-4bbe-b77c-633afe293601/cmd/kubeadm/app/util/error_test.go
 * @brief Unit testing suite for kubeadm error handling and formatting utilities.
 * Functional Utility: Ensures process exit codes and error reports are generated 
 * correctly according to the established error taxonomy.
 * Domain: Kubernetes, Production Software Verification.
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

package util

import (
	"fmt"
	"testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
)

/**
 * @brief Validates the exit code assignment for different error types.
 * Logic: Table-driven test evaluating generic errors, preflight errors, and success states.
 */
func TestCheckErr(t *testing.T) {
	var codeReturned int
	// Functional Utility: Mock error handler to capture exit codes without terminating the test process.
	errHandle := func(err string, code int) {
		codeReturned = code
	}

	var tokenTest = []struct {
		e        error
		expected int
	}{
		{nil, 0},
		{fmt.Errorf(""), DefaultErrorExitCode},
		{&preflight.Error{}, PreFlightExitCode},
	}

	for _, rt := range tokenTest {
		codeReturned = 0
		checkErr("", rt.e, errHandle)
		if codeReturned != rt.expected {
			t.Errorf(
				"failed checkErr:\n\texpected: %d\n\t  actual: %d",
				rt.expected,
				codeReturned,
			)
		}
	}
}

/**
 * @brief Validates the multi-error formatting logic.
 * Logic: Ensures that error slices are correctly transformed into a tab-separated multi-line string.
 */
func TestFormatErrMsg(t *testing.T) {
	errMsg1 := "specified version to upgrade to v1.9.0-alpha.3 is equal to or lower than the cluster version v1.10.0-alpha.0.69+638add6ddfb6d2. Downgrades are not supported yet"
	errMsg2 := "specified version to upgrade to v1.9.0-alpha.3 is higher than the kubeadm version v1.9.0-alpha.1.3121+84178212527295-dirty. Upgrade kubeadm first using the tool you used to install kubeadm"

	testCases := []struct {
		errs   []error
		expect string
	}{
		{
			errs: []error{
				fmt.Errorf(errMsg1),
				fmt.Errorf(errMsg2),
			},
			expect: "\t-" + errMsg1 + "\n" + "\t-" + errMsg2 + "\n",
		},
		{
			errs: []error{
				fmt.Errorf(errMsg1),
			},
			expect: "\t-" + errMsg1 + "\n",
		},
	}

	for _, testCase := range testCases {
		got := FormatErrMsg(testCase.errs)
		if got != testCase.expect {
			t.Errorf("FormatErrMsg error, expect: %v, got: %v", testCase.expect, got)
		}
	}
}
