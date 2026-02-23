/**
 * @file 1343C.CPP
 * @brief Solution to Codeforces problem 1343C - Alternating Subsequence.
 *
 * @algorithm
 * The problem is to find a subsequence with alternating signs that has the maximum
 * possible sum. The optimal strategy is to pick the largest number from each contiguous
 * block of same-signed numbers and sum them up.
 *
 * This code implements a highly condensed, single-pass dynamic programming approach.
 * It's optimized for speed but is not immediately obvious.
 *
 * It uses two arrays, `dp` and `a`, of size 2 to maintain state for positive and negative numbers.
 * Let's define index 1 for positive numbers and index 0 for negative numbers.
 * The code uses boolean expressions like `x > 0` (which is 1 for true, 0 for false) to select the index.
 *
 * - `dp[1]` stores the max sum of an alternating subsequence ending with a positive number.
 * - `dp[0]` stores the max sum of an alternating subsequence ending with a negative number.
 * - `a[1]` stores the max value found in the current contiguous block of positive numbers.
 * - `a[0]` stores the max value found in the current contiguous block of negative numbers.
 *
 * When the sign changes, the `dp` state is updated by adding the new element to the `dp` state of the opposite sign.
 * When the sign is the same, the code updates the max element `a` for the current block and recalculates the `dp` state.
 *
 * @complexity
 * - Time Complexity: O(N) per test case, due to a single pass over the input.
 * - Space Complexity: O(1) auxiliary space.
 */
#include <bits/stdc++.h>
#define FOR(i, a, b) for (int i = (a); i < (b); i++)
#define ll long long
#define N 100010
using namespace std;

// t: number of test cases.
// n: number of elements in the array.
// x: the current element being read.
// a[2]: tracks max element in current same-sign block (a[0] for neg, a[1] for pos).
// dp[2]: tracks max subsequence sum (dp[0] ends in neg, dp[1] ends in pos).
ll t, n, x, a[2], dp[2];

int main() {
  cin >> t;
  while (t--) {
    cin >> n;
    // Reset state for the new test case.
    a[0] = a[1] = INT_MIN; // INT_MIN marks that no block is currently active.
    dp[0] = dp[1] = 0;     // Initial sums are zero.
    
    // The FOR macro expands to a standard for loop.
    FOR(i, 0, n) {
      cin >> x;
      
      // Determine sign indices. idx 1 is for positive, 0 for negative.
      int pos_idx = (x > 0); // is 1 if x > 0, 0 otherwise
      int neg_idx = (x < 0); // is 1 if x < 0, 0 otherwise

      // This line is key to detecting a sign change.
      // If x is positive, `x<0` is 0. It resets `a[0]` (the negative block tracker).
      // If x is negative, `x<0` is 1. It resets `a[1]` (the positive block tracker).
      // This effectively resets the state of the *opposite* sign's block,
      // which will cause the `if (a[...]==INT_MIN)` to trigger on the next sign change.
      a[neg_idx] = INT_MIN;
      
      // Check if we are starting a new block of the current sign.
      if (a[pos_idx] == INT_MIN) {
        // Yes, new block. The new DP value is the previous DP value of the opposite sign + current element.
        dp[pos_idx] = dp[neg_idx] + x;
      } else {
        // No, same sign block. Update the max element for this block.
        ll old_max = a[pos_idx];
        a[pos_idx] = max(a[pos_idx], x);
        // The DP value is updated by replacing the previous max of this block with the new max.
        dp[pos_idx] = dp[pos_idx] - old_max + a[pos_idx];
      }
      // This update is redundant inside the `else`, but necessary for the `if` case.
      a[pos_idx] = max(a[pos_idx], x);
    }
    
    // The result is the max of the two possible final DP states.
    // If the sequence ends with positive numbers, the answer is in dp[1].
    // If it ends with negative numbers, the answer is in dp[0].
    // max() correctly handles either case, including all-positive or all-negative arrays.
    cout << max(dp[0], dp[1]) << '\n';
  }

  return 0;
}