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

ll t, n, x, a[2], dp[2];

int main() {
  cin >> t;
  while (t--) {
    cin >> n;
    a[0] = a[1] = INT_MIN;
    dp[0] = dp[1] = 0;
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
      a[x > 0] = max(a[x > 0], x);
    }
    // The final result is taken from the dp array based on the last element's sign.
    // This is likely incorrect as the logic within the loop is flawed.
    cout << dp[x > 0] << '\n';
  }

  return 0;
}
