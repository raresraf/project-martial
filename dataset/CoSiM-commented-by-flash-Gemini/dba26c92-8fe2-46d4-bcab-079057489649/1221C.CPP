/**
 * @file 1221C.CPP
 * @brief This program solves a resource allocation problem. Given quantities 'c', 'm', and 'x',
 * it aims to maximize the number of "teams" that can be formed, where each team requires
 * specific combinations of 'c' and 'm' resources, and 'x' acts as a flexible resource.
 * Algorithm: Greedy approach to balance 'c' and 'm' resources first, then distribute 'x'.
 * Time Complexity: O(T), where T is the number of test cases.
 * Space Complexity: O(1) as only a few variables are used per test case.
 */

#include <bits/stdc++.h> // Includes most standard C++ library headers for competitive programming.

using namespace std;

int main() {
  // Optimizes C++ standard streams for faster input/output operations.
  ios_base::sync_with_stdio(false);

  int t; // Stores the number of test cases.
  // Reads the number of test cases.
  cin >> t;

  int c, m, x; // 'c' and 'm' represent two types of primary resources, 'x' is a third, flexible resource.
  int res; // Variable to hold the result for each test case (though 'x' is repurposed for the result).
  int maxi, mini; // Variables to store the maximum and minimum of 'c' and 'm'.

  /**
   * Block Logic: Iterates through each test case provided by the input 't'.
   * Each iteration processes a new set of 'c', 'm', and 'x' values.
   */
  for (int i = 0; i < t; i++) {
    res = 0; // Resets 'res' for each test case (though not explicitly used for final output).
    // Reads the quantities of the three resources for the current test case.
    cin >> c >> m >> x;

    // Identifies the larger and smaller quantities between 'c' and 'm'.
    maxi = max(c, m);
    mini = min(c, m);

    /**
     * Block Logic: Uses resource 'x' to balance 'c' and 'm'.
     * The difference between 'maxi' and 'mini' is "covered" by 'x'.
     * This effectively reduces the larger resource 'maxi' down to 'mini',
     * and the 'x' resource is reduced by the amount used for balancing.
     */
    x += (maxi - mini); // 'x' absorbs the difference between 'c' and 'm'.

    /**
     * Conditional Logic: Checks if 'x' (after balancing) is sufficient to form teams
     * where the number of teams is limited by the minimum of the initial 'c' and 'm'.
     * If 'x' is greater than or equal to either 'c' or 'm' (which are now conceptually balanced, or one is fully consumed),
     * the maximum number of teams is simply the minimum of the original 'c' and 'm'.
     */
    if (x >= c || x >= m) {
      // If 'x' is abundant, the maximum number of teams is limited by the smallest original resource.
      cout << min(c, m) << "\n";
      continue; // Move to the next test case.
    }

    /**
     * Block Logic: If 'x' was not sufficient to cover the difference completely or if more teams can be formed,
     * it further processes the remaining resources. 'x' is now seen as additional capacity.
     * It reduces 'maxi' (which is now equal to 'mini') by 'x', and then adds a third of the remaining total
     * (maxi + mini) to 'x'. This implies that each team formed after initial balancing requires 3 units from
     * the combined 'c' and 'm' pool, which 'x' can help to facilitate.
     */
    maxi = mini; // Both 'c' and 'm' are now effectively equal to 'mini' after balancing with 'x'.
    maxi -= x;   // Deduct the 'x' capacity from one of the balanced resources.
    mini -= x;   // Deduct the 'x' capacity from the other balanced resource.
    // The final result 'x' is incremented by a third of the combined remaining 'maxi' and 'mini'.
    // This implies that 'x' now holds the result, and remaining 'c' and 'm' are split into groups of 3 for more teams.
    x += (maxi + mini) / 3;

    // Prints the final calculated maximum number of teams for the current test case.
    cout << x << "\n";
  }
  // Standard program exit.
  return 0;
}
