/**
 * @file 1426C.CPP
 * @brief Solution to Codeforces Problem 1426C: "Increase and Copy".
 *        The problem asks for the minimum number of operations to obtain a number `N`
 *        starting from 1. The allowed operations are:
 *        1. Increase the current number by 1 (cost 1).
 *        2. Copy the current number (cost 1).
 *
 * Algorithm: The optimal strategy involves reaching some number `x` by incrementing,
 *            and then copying it `k` times such that `x * 2^k` is close to `N`.
 *            More generally, it involves reaching a value `i` (by `i-1` increments)
 *            and then performing `N/i - 1` copies to reach `i * (N/i) = N` if `N` is divisible by `i`.
 *            If `N` is not divisible by `i`, it means we obtain `i * (N/i)` and then
 *            need one more copy operation for the remaining `N % i` if `rem > 0`.
 *            The approach iterates through possible values for `i` up to `sqrt(N)`
 *            to find the minimum operations.
 * Time Complexity: O(sqrt(N)) due to iterating up to the square root of N.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
/**
 * @brief Macro to optimize C++ standard stream I/O operations.
 * Functional Utility: Unties C++ standard streams from C standard streams (`ios_base::sync_with_stdio(false)`)
 * and unties `cin` from `cout` (`cin.tie(0)`), leading to faster input/output,
 * which is crucial for competitive programming performance.
 */
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);

typedef long long int ll;

int main() {
  optimizar_io /**
   * Functional Utility: Declaration of variables.
   * `tc`: Stores the number of test cases.
   * `n`: Stores the target number for each test case.
   * `ans`: Stores the minimum operations found so far, initialized to a maximum possible integer value.
   */
  int tc, n;

  cin >> tc; // Input: Read the number of test cases.
  /**
   * Block Logic: Processes each test case.
   * Pre-condition: `tc` is the number of test cases.
   * Invariant: `n` contains the current target number for which to find minimum operations.
   */
  while (tc--) {
    cin >> n; // Input: Read the target number `n` for the current test case.
    int ans = numeric_limits<int>::max(); // Initialize `ans` to a very large value for minimization.
    /**
     * Block Logic: Iterates through possible intermediate values `i` up to `sqrt(n)`.
     * This loop explores different strategies of reaching `n` by first incrementing to `i`,
     * and then copying `i` multiple times.
     * Pre-condition: `n` is the target number.
     * Invariant: `i` represents a potential intermediate number to be formed by incrementing.
     */
    for (int i = 1; i * i <= n; i++) {
      int rem = n % i; // Calculate the remainder when `n` is divided by `i`.
      int steps = i - 1; // Operations to reach `i` by incrementing from 1.
      steps += (n / i) - 1; // Operations to reach `n` from `i` by copying (n/i - 1) times.
      /**
       * Conditional Logic: If there's a remainder, it means we need one more copy operation
       * to cover the remaining part after reaching `i * (n/i)`. This effectively means
       * reaching `i` and then having `n/i` copies, plus one more copy to increase the last `i`
       * to cover the `rem`.
       */
      if (rem)
        steps++; // Add one more step if `n` is not perfectly divisible by `i`.
      ans = min(ans, steps); // Update the minimum steps found so far.
    }
    cout << ans << "\n"; // Output the minimum operations for the current test case.
  }
  return 0;
}
