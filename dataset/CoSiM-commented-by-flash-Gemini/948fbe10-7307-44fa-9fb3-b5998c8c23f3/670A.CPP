/**
 * @file 670A.CPP
 * @brief Calculates the minimum and maximum possible weekend days within a given number of days.
 *
 * This program reads an integer 'n' representing a duration in days. It then
 * determines the minimum and maximum number of weekend days (Saturday and Sunday)
 * that can occur within this 'n' day period, considering that the starting day
 * of the period can vary.
 *
 * Algorithm: Mathematical calculation based on integer division and modulo operations.
 *            The logic relies on how many full weeks are in 'n' days and how the
 *            remaining days can be arranged to maximize or minimize weekend days.
 * Time Complexity: O(1) as it involves a constant number of arithmetic operations.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h> // Includes most standard library headers.
using namespace std;     // Uses the standard namespace.
int main() {
  /**
   * @brief Main function: Calculates and outputs min/max weekend days for a given duration 'n'.
   * @return 0 for successful execution.
   */
  long long i, n, m, r, mn, mx; // Declare variables: 'n': total days, 'm': full weeks, 'r': remaining days,
                               // 'mn': minimum weekend days, 'mx': maximum weekend days. 'i' is unused.
  /**
   * @brief Loop to process multiple test cases, reading 'n' in each iteration.
   * @pre Input stream provides 'n' (number of days) for each test case.
   * @invariant For each iteration, a new test case is processed.
   */
  while (cin >> n) { // Read the number of days 'n'.
    r = n % 7;       // Calculate the number of remaining days after full weeks.
    m = n / 7;       // Calculate the number of full weeks in 'n' days.

    // Minimum weekend days calculation:
    // Each full week (m) contributes 2 weekend days.
    // If the remaining days (r) include a Friday (r=6), then one more weekend day is added (Saturday).
    mn = m * 2;
    if (r == 6) // Pre-condition: If remaining days include a Friday (day 6 of the week, starting from Monday=0).
      mn++;     // Add one more weekend day (Saturday).

    // Maximum weekend days calculation:
    // Each full week (m) contributes 2 weekend days.
    // The remaining days (r) can contribute at most 2 additional weekend days (Saturday and Sunday),
    // depending on where the 'r' days fall in the week.
    mx = m * 2;
    if (r <= 2) // Pre-condition: If remaining days are 0, 1, or 2 (e.g., Mon, Tue).
      mx += r;  // These days can include at most 'r' weekend days (if starting near weekend).
    else        // Pre-condition: If remaining days are 3 or more (e.g., Wed, Thu, Fri, Sat, Sun).
      mx += 2;  // These days can always include 2 weekend days (Saturday and Sunday).
    
    cout << mn << " " << mx << endl; // Output the minimum and maximum weekend days.
  }
  return 0; // Return 0 for successful execution.
}
