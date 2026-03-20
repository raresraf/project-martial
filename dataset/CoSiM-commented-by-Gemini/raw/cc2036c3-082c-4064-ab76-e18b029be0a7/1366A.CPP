/**
 * @file 1366A.CPP
 * @brief Solution for a resource allocation problem.
 *
 * Problem Statement: Given two types of resources, 'a' and 'b', determine the maximum
 * number of teams that can be formed. A team can be formed using either 1 unit of 'a'
 * and 2 units of 'b', or 2 units of 'a' and 1 unit of 'b'.
 *
 * Algorithm:
 * The problem can be modeled with a system of linear inequalities. Let x be the number of
 * teams of type (1a, 2b) and y be the number of teams of type (2a, 1b).
 * The constraints are:
 * 1. x + 2y <= a
 * 2. 2x + y <= b
 * 3. x >= 0, y >= 0
 * We want to maximize k = x + y.
 *
 * By summing the first two inequalities, we get:
 * (x + 2y) + (2x + y) <= a + b
 * 3x + 3y <= a + b
 * 3(x + y) <= a + b
 * k <= (a + b) / 3
 *
 * Additionally, since x >= 0 and y >= 0, the total number of teams 'k' cannot exceed
 * the available quantity of either resource if we only made teams of the less-intensive type.
 * For example, if we only make teams of type (1a, 2b), we are limited by 'b'.
 * The total number of teams k=x+y is also bounded by 'a' and 'b' because to form k teams,
 * we need at least k units from 'a' and at least k units from 'b' in total across all team types.
 * (e.g., k = x+y; a >= x+2y = k+y >= k; b >= 2x+y = k+x >= k).
 *
 * Combining these constraints, the maximum number of teams 'k' is:
 * k_max = min(a, b, (a + b) / 3)
 *
 * Time Complexity: O(1) for each test case.
 * Space Complexity: O(1) for each test case.
 */

#include <bits/stdc++.h>
using namespace std;
int t, a, b, res;
main() {
  // Read the number of test cases.
  cin >> t;
  /**
   * Block Logic: Process each test case independently.
   * Invariant: The loop runs 't' times.
   */
  while (t--) {
    res = 0; // This variable is initialized but not used in the final calculation.
    // Read the two resource quantities for the current test case.
    cin >> a >> b;
    /**
     * Core Logic: Calculate and print the maximum number of teams.
     * The result is the minimum of the available quantities of each resource and the
     * total resource constraint (a+b)/3, which represents the bottleneck.
     */
    cout << min(min(a, b), (a + b) / 3) << '
';
  }
}
