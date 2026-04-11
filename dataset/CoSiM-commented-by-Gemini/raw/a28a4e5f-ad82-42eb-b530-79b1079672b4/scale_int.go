/**
 * @file scale_int.go
 * @brief Provides functionality for scaled integer arithmetic.
 * @author The Kubernetes Authors
 *
 * @details
 * This file contains utility functions for handling arithmetic on scaled integers,
 * a technique used to represent decimal values with high precision without using
 * floating-point numbers. This is particularly useful for representing resource
 * quantities in systems like Kubernetes where precision is critical.
 *
 * The core logic revolves around a `scaledValue` function that converts a value
 * from one decimal scale to another, using `math/big` for arbitrary-precision
 * arithmetic to prevent overflow and a `sync.Pool` to reduce memory allocations.
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
	"sync"
)

var (
	// intPool is a sync.Pool used to reduce memory allocations by reusing
	// `big.Int` objects. This is a performance optimization for a high-throughput system.
	intPool  sync.Pool
	maxInt64 = big.NewInt(math.MaxInt64)
)

func init() {
	intPool.New = func() interface{} {
		return &big.Int{}
	}
}

// scaledValue scales a given unscaled value from its current scale to a new scale
// and returns the result as an int64. The mathematical value of the decimal is
// represented as `unscaled * 10**(-scale)`.
//
// Key Behaviors:
// - Rounding: It ALWAYS rounds up the result when scaling down (i.e., when precision is lost).
//   For example, scaling 999 with a scale of 3 down to a scale of 0 results in 1 (0.999 rounded up).
// - Overflow: The final result might overflow an int64 if the scaled value is too large.
//
// Parameters:
//   unscaled: A `*big.Int` representing the value without its decimal scaling.
//   scale: The original exponent of the decimal value (e.g., 3 for milli).
//   newScale: The target exponent for the new value.
//
// Returns:
//   An int64 representing the value at the new scale, rounded up if necessary.
func scaledValue(unscaled *big.Int, scale, newScale int) int64 {
	// The difference in scale determines whether we are scaling up or down.
	dif := scale - newScale
	if dif == 0 {
		return unscaled.Int64()
	}

	// Handle scale up (e.g., from milli to nano).
	// This is a simple multiplication. Overflow is possible but handled by standard integer overflow.
	if dif < 0 {
		return unscaled.Int64() * int64(math.Pow10(-dif))
	}

	// Handle scale down (e.g., from nano to milli).
	// This requires careful handling of division and rounding.

	// A fast path for common cases where the numbers fit within standard int64 and
	// the divisor (10^dif) is within pre-calculated math.Pow10 limits. This avoids
	// the overhead of `big.Int` arithmetic.
	const log10MaxInt64 = 18 // math.Pow10 is pre-calculated up to 10^18
	if unscaled.Cmp(maxInt64) < 0 && dif < log10MaxInt64 {
		divide := int64(math.Pow10(dif))
		result := unscaled.Int64() / divide
		mod := unscaled.Int64() % divide
		// Invariant: If there is any remainder, always round up.
		if mod != 0 {
			return result + 1
		}
		return result
	}

	// Slow path for large numbers that require arbitrary-precision arithmetic.
	// We use a pool of `big.Int` objects to avoid allocations.
	divisor := intPool.Get().(*big.Int)
	exp := intPool.Get().(*big.Int)
	result := intPool.Get().(*big.Int)
	defer func() {
		intPool.Put(divisor)
		intPool.Put(exp)
		intPool.Put(result)
	}()

	// divisor = 10^(dif)
	divisor.Exp(bigTen, exp.SetInt64(int64(dif)), nil)
	remainder := exp // Reuse the 'exp' object to store the remainder.

	// result = unscaled / divisor
	// remainder = unscaled % divisor
	result.DivMod(unscaled, divisor, remainder)
	// Invariant: If the remainder is not zero, round up by adding 1.
	if remainder.Sign() != 0 {
		return result.Int64() + 1
	}

	return result.Int64()
}
