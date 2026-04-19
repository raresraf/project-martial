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

/**
 * @file scale_int.go
 * @brief High-precision scaling utilities for resource quantification.
 * 
 * This module provides optimized functions for scaling arbitrary-precision 
 * decimal values represented as unscaled big.Ints. It is specifically designed 
 * for resource quantification (e.g., CPU, Memory) within Kubernetes, ensuring 
 * accuracy through ceiling rounding during down-scaling while optimizing 
 * performance via object pooling and fast-path logic for standard integer types.
 * 
 * Domain: Production Systems, Resource Management, Precision Arithmetic.
 */

package resource

import (
	"math"
	"math/big"
	"sync"
)

var (
	// Performance Optimization: A sync pool to minimize garbage collection pressure 
	// by reusing big.Int allocations during high-frequency scaling operations.
	intPool  sync.Pool
	maxInt64 = big.NewInt(math.MaxInt64)
)

func init() {
	intPool.New = func() interface{} {
		return &big.Int{}
	}
}

/**
 * @brief Scales an unscaled big.Int to a new decimal scale.
 * 
 * functional Utility: Computes (unscaled * 10^(-scale)) and converts the result 
 * to 10^(-newScale) precision, returning it as an int64. 
 * 
 * Rounding: Implements a ceiling policy (round-up) for all down-scaling operations.
 * 
 * @param unscaled The base arbitrary-precision integer.
 * @param scale The current decimal exponent of the unscaled value.
 * @param newScale The target decimal exponent for the return value.
 * @return The scaled value represented as an int64.
 */
func scaledValue(unscaled *big.Int, scale, newScale int) int64 {
	dif := scale - newScale
	if dif == 0 {
		return unscaled.Int64()
	}

	/**
	 * Execution Block: Scale Up (e.g., from Millicores to NanoCores).
	 * Logic: Multiplies by the required power of 10. Does not require rounding.
	 */
	if dif < 0 {
		return unscaled.Int64() * int64(math.Pow10(-dif))
	}

	/**
	 * Execution Block: Scale Down (e.g., from Bytes to MegaBytes).
	 * Logic: Divides by the power of 10 and applies ceiling rounding.
	 */

	/**
	 * Fast Path: Optimized path for values that fit within standard 64-bit integers.
	 * Avoids the overhead of arbitrary-precision arithmetic.
	 */
	const log10MaxInt64 = 19
	if unscaled.Cmp(maxInt64) < 0 && dif < log10MaxInt64 {
		divide := int64(math.Pow10(dif))
		result := unscaled.Int64() / divide
		mod := unscaled.Int64() % divide
		// Logic: Ceiling rounding check for remainders.
		if mod != 0 {
			return result + 1
		}
		return result
	}

	/**
	 * Slow Path: Arbitrary-precision fallback.
	 * Utilizes object pooling to perform the division and modulo operations.
	 */
	divisor := intPool.Get().(*big.Int)
	exp := intPool.Get().(*big.Int)
	result := intPool.Get().(*big.Int)
	defer func() {
		// Cleanup: Returns big.Int objects to the pool for reuse.
		intPool.Put(divisor)
		intPool.Put(exp)
		intPool.Put(result)
	}()

	// Logic: divisor = 10^(dif).
	divisor.Exp(bigTen, exp.SetInt64(int64(dif)), nil)
	remainder := exp

	// Logic: result = unscaled / divisor; remainder = unscaled % divisor.
	result.DivMod(unscaled, divisor, remainder)
	
	// Rounding: Checks for a non-zero remainder to increment the result (ceil).
	if remainder.Sign() != 0 {
		return result.Int64() + 1
	}

	return result.Int64()
}
