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
	// intPool is a sync.Pool to reuse *big.Int objects, reducing memory allocations.
	intPool sync.Pool
	// maxInt64 is a *big.Int representing the maximum value of an int64.
	maxInt64 = big.NewInt(math.MaxInt64)
)

// init initializes the sync.Pool for big.Int objects.
func init() {
	intPool.New = func() interface{} {
		return &big.Int{}
	}
}

// scaledValue scales a given unscaled value from its current scale to a new
// scale and returns the result as an int64. It ALWAYS rounds up the result
// when scaling down (losing precision). The final result might overflow an int64.
//
// The parameters 'scale' and 'newScale' represent the exponent of the decimal.
// The mathematical value of the number is unscaled * 10**(-scale).
func scaledValue(unscaled *big.Int, scale, newScale int) int64 {
	// The difference in scale determines whether we are scaling up or down.
	dif := scale - newScale
	if dif == 0 {
		return unscaled.Int64()
	}

	// Handle scale up.
	// This is a multiplication, which is an easy case. We do not need to
	// care about rounding. If any intermediate operation causes overflow,
	// the final result will also overflow.
	if dif < 0 {
		return unscaled.Int64() * int64(math.Pow10(-dif))
	}

	// Handle scale down.
	// This is a division, so we have to be careful about rounding and
	// potential overflow in intermediate operations.

	// A fast path for when the unscaled value fits in an int64 and the
	// divisor (10^dif) is also within int64 range. This avoids the overhead
	// of big.Int arithmetic for common cases.
	const log10MaxInt64 = 19
	if unscaled.Cmp(maxInt64) < 0 && dif < log10MaxInt64 {
		divide := int64(math.Pow10(dif))
		result := unscaled.Int64() / divide
		mod := unscaled.Int64() % divide
		// If there is a remainder, round up.
		if mod != 0 {
			return result + 1
		}
		return result
	}

	// The general path for large numbers, using the big.Int library to avoid overflow.
	// Objects are retrieved from the pool to reduce allocations.
	divisor := intPool.Get().(*big.Int)
	exp := intPool.Get().(*big.Int)
	result := intPool.Get().(*big.Int)
	defer func() {
		intPool.Put(divisor)
		intPool.Put(exp)
		intPool.Put(result)
	}()

	// divisor = 10^(dif)
	// TODO: Consider a lookup table if exp is too costly.
	divisor.Exp(bigTen, exp.SetInt64(int64(dif)), nil)
	// reuse exp to store the remainder
	remainder := exp

	// result = unscaled / divisor
	// remainder = unscaled % divisor
	result.DivMod(unscaled, divisor, remainder)
	// If there is a non-zero remainder, round up by adding 1.
	if remainder.Sign() != 0 {
		return result.Int64() + 1
	}

	return result.Int64()
}
