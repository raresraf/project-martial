/**
 * @a28a4e5f-ad82-42eb-b530-79b1079672b4/scale_int_test.go
 * @brief Unit tests and performance benchmarks for high-precision resource scaling logic.
 * Domain: Distributed Systems, Unit Testing, Performance Engineering.
 * Architecture: Utilizes Go's 'testing' package with table-driven tests for functional verification and iterative benchmarking for throughput analysis.
 * Functional Utility: Validates scaling correctness across multiple orders of magnitude, specifically verifying the mandatory round-up (ceiling) invariant for scale-down operations.
 */

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package resource

import (
	"math"
	"math/big"
	"testing"
)

/**
 * @brief Functional verification of the scaledValue internal routine.
 * Logic: Table-driven execution covering identity scaling, scale-up, scale-down, and precision-bounded large integer cases.
 * Invariant: Scaling down any non-zero remainder must result in a ceiling (round-up) operation.
 */
func TestScaledValueInternal(t *testing.T) {
	tests := []struct {
		unscaled *big.Int
		scale    int
		newScale int

		want int64
	}{
		// remain scale
		{big.NewInt(1000), 0, 0, 1000},

		// scale down
		{big.NewInt(1000), 0, -3, 1},
		{big.NewInt(1000), 3, 0, 1},
		{big.NewInt(0), 3, 0, 0},

		// always round up: Verifies the core rounding-invariant for resource allocation safety.
		{big.NewInt(999), 3, 0, 1},
		{big.NewInt(500), 3, 0, 1},
		{big.NewInt(499), 3, 0, 1},
		{big.NewInt(1), 3, 0, 1},
		// large scaled value does not lose precision
		{big.NewInt(0).Sub(maxInt64, bigOne), 1, 0, (math.MaxInt64-1)/10 + 1},
		// large intermediate result: Tests the math/big slow-path resilience.
		{big.NewInt(1).Exp(big.NewInt(10), big.NewInt(100), nil), 100, 0, 1},

		// scale up
		{big.NewInt(0), 0, 3, 0},
		{big.NewInt(1), 0, 3, 1000},
		{big.NewInt(1), -3, 0, 1000},
		{big.NewInt(1000), -3, 2, 100000000},
		{big.NewInt(0).Div(big.NewInt(math.MaxInt64), bigThousand), 0, 3,
			(math.MaxInt64 / 1000) * 1000},
	}

	for i, tt := range tests {
		old := (&big.Int{}).Set(tt.unscaled)
		got := scaledValue(tt.unscaled, tt.scale, tt.newScale)
		if got != tt.want {
			t.Errorf("#%d: got = %v, want %v", i, got, tt.want)
		}
		// Invariant: Input big.Int must remain immutable during the calculation.
		if tt.unscaled.Cmp(old) != 0 {
			t.Errorf("#%d: unscaled = %v, want %v", i, tt.unscaled, old)
		}
	}
}

/**
 * @brief Measures throughput of the 64-bit fast-path scaling logic.
 */
func BenchmarkScaledValueSmall(b *testing.B) {
	s := big.NewInt(1000)
	for i := 0; i < b.N; i++ {
		scaledValue(s, 3, 0)
	}
}

/**
 * @brief Measures throughput of the math/big arbitrary precision slow-path.
 */
func BenchmarkScaledValueLarge(b *testing.B) {
	s := big.NewInt(math.MaxInt64)
	s.Mul(s, big.NewInt(1000))
	for i := 0; i < b.N; i++ {
		scaledValue(s, 10, 0)
	}
}
