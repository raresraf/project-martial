/**
 * @file 1221C.CPP
 * @brief Solution to Codeforces problem 1221C - Perfect Team.
 *
 * @algorithm
 * The problem asks for the maximum number of teams that can be formed from a
 * group of `c` coders, `m` mathematicians, and `x` people with no specialty.
 * Each team must consist of one coder, one mathematician, and one other person.
 *
 * The number of teams `k` is constrained by two main factors:
 * 1. The number of specialists: You need one coder and one mathematician for each
 *    team. Therefore, the number of teams cannot exceed the number of coders (`c`)
 *    or the number of mathematicians (`m`). This gives the constraint: `k <= min(c, m)`.
 *
 * 2. The total number of people: Each team requires 3 people. The total pool of
 *    people is `c + m + x`. Therefore, the number of teams cannot exceed the
 *    total number of people divided by 3. This gives the constraint: `k <= (c + m + x) / 3`.
 *
 * To find the maximum possible number of teams, we must satisfy both constraints.
 * The answer is therefore the minimum of these two upper bounds:
 * `k = min(min(c, m), (c + m + x) / 3)`.
 *
 * The code implements this logic directly.
 *
 * @complexity
 * - Time Complexity: O(1) for each test case, as it involves a few
 *   arithmetic and comparison operations.
 * - Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  // This is a common competitive programming setup for local testing.
  // If the macro 'd' is defined during compilation (e.g., g++ -D d ...),
  // it redirects standard input from "input.txt".
#ifdef d
  freopen("input.txt", "r", stdin);
#endif

  int t;
  cin >> t; // Read the number of test cases.
  while (t--) {
    // c: number of coders
    // m: number of mathematicians
    // x: number of people with no specialization
    int c, m, x;
    cin >> c >> m >> x;

    // Constraint 1: The number of teams is limited by the minimum number of
    // available specialists (coders or mathematicians).
    int y = min(c, m);
    
    // The final answer is the minimum of the two constraints:
    // 1. The number of specialist pairs (`y`).
    // 2. The total number of groups of three that can be formed (`(c+m+x)/3`).
    int z = min(y, (c + m + x) / 3);
    cout << z << "\n";
  }

  return 0;
}
