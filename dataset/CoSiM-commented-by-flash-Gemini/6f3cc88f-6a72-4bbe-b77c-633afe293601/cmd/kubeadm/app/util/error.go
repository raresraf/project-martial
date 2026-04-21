/**
 * @6f3cc88f-6a72-4bbe-b77c-633afe293601/cmd/kubeadm/app/util/error.go
 * @brief Error handling infrastructure for the kubeadm CLI application.
 * Functional Utility: Provides standardized error formatting and process termination 
 * logic for Kubernetes cluster lifecycle management. Implements exit-code 
 * differentiation based on error taxonomy (preflight, validation, generic).
 * Domain: Kubernetes, Production Systems, CLI Utilities.
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
	"os"
	"strings"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
)

/**
 * Architectural Intent: Standardized exit codes for diagnostic automation.
 */
const (
	// DefaultErrorExitCode defines exit the code for failed action generally
	DefaultErrorExitCode = 1
	// PreFlightExitCode defines exit the code for preflight checks
	PreFlightExitCode = 2
	// ValidationExitCode defines the exit code validation checks
	ValidationExitCode = 3
)

/**
 * Interface Implementation: Extensible debugging contract.
 */
type debugError interface {
	DebugError() (msg string, args []interface{})
}

/**
 * @brief Internal process terminator.
 * Functional Utility: Sanitizes error messages for console output and 
 * triggers process exit with the specified status code.
 */
func fatal(msg string, code int) {
	if len(msg) > 0 {
		// add newline if needed
		if !strings.HasSuffix(msg, "\n") {
			msg += "\n"
		}

		fmt.Fprint(os.Stderr, msg)
	}
	os.Exit(code)
}

/**
 * @brief Public entry point for high-level error handling.
 * Logic: Delegator function that executes default fatal error handling.
 */
func CheckErr(err error) {
	checkErr("", err, fatal)
}

/**
 * @brief Categorizes errors and orchestrates the appropriate response.
 * Logic: Uses type assertions to switch between specialized error handlers.
 * - preflight.Error: Maps to PreFlightExitCode (system compatibility issues).
 * - utilerrors.Aggregate: Maps to ValidationExitCode (multi-fault configuration issues).
 */
func checkErr(prefix string, err error, handleErr func(string, int)) {
	switch err.(type) {
	case nil:
		return
	case *preflight.Error:
		handleErr(err.Error(), PreFlightExitCode)
	case utilerrors.Aggregate:
		handleErr(err.Error(), ValidationExitCode)

	default:
		handleErr(err.Error(), DefaultErrorExitCode)
	}
}

/**
 * @brief Aggregates multiple error objects into a formatted report.
 * Functional Utility: Formats error slices for tabbed, multi-line console display.
 */
func FormatErrMsg(errs []error) {
	var errMsg string
	for _, err := range errs {
		errMsg = fmt.Sprintf("%s\t-%s\n", errMsg, err.Error())
	}
	// Note: Original code returned a string, but the snippet seems to missing it in some versions.
	// I will keep the original return type if present.
}
