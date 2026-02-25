/**
 * @file raw/45aaa46e-5c60-497e-b77d-8412712ba4b9/1221C.CPP
 * @brief Solution to Codeforces problem 1221C - Perfect Team.
 *
 * @algorithm
 * The problem asks for the maximum number of "perfect teams" that can be formed
 * from `c` coders, `m` mathematicians, and `x` individuals with no specialty.
 * A perfect team requires one of each: 1 coder, 1 mathematician, and 1 other member.
 * The "other member" slot can be filled by one of the `x` individuals, or by
 * any leftover coders or mathematicians.
 *
 * Let `k` be the number of teams we want to form. To form `k` teams, we must satisfy
 * the following constraints:
 *
 * 1.  We need `k` coders, so the number of teams cannot exceed the number of available
 *     coders: `k <= c`.
 *
 * 2.  We need `k` mathematicians, so `k <= m`.
 *
 * 3.  We need `k` members for the third "specialist" slot. These can be the `x`
 *     unspecialized people, plus any remaining coders (`c - k`) and mathematicians
 *     (`m - k`). So, the total pool for this third slot is `x + (c - k) + (m - k)`.
 *     Therefore, `k <= x + c - k + m - k`.
 *     Rearranging this gives: `3k <= c + m + x`, which simplifies to
 *     `k <= (c + m + x) / 3`.
 *
 * To find the maximum possible value of `k`, we must satisfy all three constraints
 * simultaneously. The maximum `k` is therefore the minimum of the three upper bounds:
 *   `k_max = min(c, m, (c + m + x) / 3)`
 *
 * The code implements this formula directly.
 *
 * @complexity
 * - Time Complexity: O(1) for each test case, as it only involves a few
 *   arithmetic operations and comparisons.
 * - Space Complexity: O(1), using only a few variables.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

// This is a common preprocessor directive for competitive programming
// to redirect input from "input.txt" when compiling with a `d` flag.
#ifdef d
  freopen("input.txt", "r", stdin);
#endif

  int t; // Number of test cases.
  cin >> t;
  while (t--) {
    int c, m, x; // c: coders, m: mathematicians, x: no specialty.
    cin >> c >> m >> x;

    // The number of teams is limited by the minimum of coders and mathematicians.
    int y = min(c, m);
    
    // The number of teams is also limited by the total number of people divided by 3.
    int z = min(y, (c + m + x) / 3);

    // The result is the minimum of these three constraints.
    cout << z << "\n";
  }

  return 0;
}
