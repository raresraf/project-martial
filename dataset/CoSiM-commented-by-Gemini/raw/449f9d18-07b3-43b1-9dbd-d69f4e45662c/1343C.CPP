/**
 * @file raw/449f9d18-07b3-43b1-9dbd-d69f4e45662c/1343C.CPP
 * @brief Solution to Codeforces problem 1343C - Alternating Subsequence.
 *
 * @algorithm
 * The problem is to find the maximum sum of a subsequence with alternating signs.
 * The correct approach is a greedy one: from each contiguous block of numbers
 * with the same sign, pick the largest one and add it to the total sum.
 *
 * This code attempts to solve the problem in a single pass using a dynamic
 * programming-like approach with state variables. However, the logic
 * is extremely condensed, appears to be buggy, and is difficult to follow.
 * It uses two arrays, `a` and `dp`, of size 2, to track states.
 *
 * Due to the confusing and likely incorrect logic, it's hard to describe its
 * exact behavior. A more readable and correct solution would iterate, find blocks,
 * find the max in each, and sum them up.
 *
 * @complexity
 * - Time Complexity: O(N) for each test case.
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
      // This logic is extremely condensed and appears to be incorrect.
      // It attempts a greedy strategy in an online fashion.
      a[x < 0] = INT_MIN;
      if (a[x > 0] == INT_MIN)
        dp[x > 0] = dp[x < 0] + x;
      else {
        a[x > 0] = max(a[x > 0], x);
        dp[x > 0] = max(dp[x > 0], dp[x < 0] + a[x > 0]);
      }
      // This update is redundant inside the `else`, but necessary for the `if` case.
      a[pos_idx] = max(a[pos_idx], x);
    }
    // The final result is taken from the dp array based on the last element's sign.
    // This is likely incorrect as the logic within the loop is flawed.
    cout << dp[x > 0] << '\n';
  }

  return 0;
}
