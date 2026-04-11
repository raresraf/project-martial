/**
 * @file scale_int_test.go
 * @brief Unit and benchmark tests for the scaled integer functionality.
 * @author The Kubernetes Authors
 *
 * @details
 * This file contains tests for the `scaledValue` function defined in `scale_int.go`.
 * It includes a table-driven unit test (`TestScaledValueInternal`) to verify correctness
 * across various scenarios and benchmark tests (`BenchmarkScaledValueSmall`,
 * `BenchmarkScaledValueLarge`) to measure performance.
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

// TestScaledValueInternal is a table-driven test that verifies the correctness
// of the scaledValue function across a variety of scenarios.
func TestScaledValueInternal(t *testing.T) {
	tests := []struct {
		unscaled *big.Int
		scale    int
		newScale int
		want     int64
	}{
		// Test case: No change in scale.
		{big.NewInt(1000), 0, 0, 1000},

		// Test cases: Scaling down, where precision is lost.
		{big.NewInt(1000), 0, -3, 1},
		{big.NewInt(1000), 3, 0, 1},
		{big.NewInt(0), 3, 0, 0},

		// Test cases: Verification of the "always round up" behavior.
		// Any non-zero remainder during a scale-down operation should result in rounding up.
		{big.NewInt(999), 3, 0, 1}, // 0.999 becomes 1
		{big.NewInt(500), 3, 0, 1}, // 0.500 becomes 1
		{big.NewInt(499), 3, 0, 1}, // 0.499 becomes 1
		{big.NewInt(1), 3, 0, 1},   // 0.001 becomes 1
		// Test with a large value to ensure the rounding logic holds and doesn't lose precision.
		{big.NewInt(0).Sub(maxInt64, bigOne), 1, 0, (math.MaxInt64-1)/10 + 1},
		// Test with a very large intermediate result to exercise the `big.Int` path.
		{big.NewInt(1).Exp(big.NewInt(10), big.NewInt(100), nil), 100, 0, 1},

		// Test cases: Scaling up, where precision is gained.
		{big.NewInt(0), 0, 3, 0},
		{big.NewInt(1), 0, 3, 1000},
		{big.NewInt(1), -3, 0, 1000},
		{big.NewInt(1000), -3, 2, 100000000},
		// Test scaling up a large number.
		{big.NewInt(0).Div(big.NewInt(math.MaxInt64), bigThousand), 0, 3,
			(math.MaxInt64 / 1000) * 1000},
	}

	for i, tt := range tests {
		// Create a copy of the unscaled value to ensure the original is not mutated by the function.
		old := (&big.Int{}).Set(tt.unscaled)
		got := scaledValue(tt.unscaled, tt.scale, tt.newScale)
		if got != tt.want {
			t.Errorf("#%d: got = %v, want %v", i, got, tt.want)
		}
		// Verify that the input `unscaled` value was not modified.
		if tt.unscaled.Cmp(old) != 0 {
			t.Errorf("#%d: unscaled = %v, want %v", i, tt.unscaled, old)
		}
	}
}

// BenchmarkScaledValueSmall measures the performance of the fast path in scaledValue,
// where the numbers are small enough to be handled with standard int64 arithmetic.
func BenchmarkScaledValueSmall(b *testing.B) {
	s := big.NewInt(1000)
	for i := 0; i < b.N; i++ {
		scaledValue(s, 3, 0)
	}
}

// BenchmarkScaledValueLarge measures the performance of the slow path in scaledValue,
// which requires arbitrary-precision arithmetic using `big.Int` objects.
func BenchmarkScaledValueLarge(b *testing.B) {
	s := big.NewInt(math.MaxInt64)
	s.Mul(s, big.NewInt(1000))
	for i := 0; i < b.N; i++ {
		scaledValue(s, 10, 0)
	}
}
