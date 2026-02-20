/**
 * @file 670A.CPP
 * @brief Calculates the minimum and maximum possible rest days in a period of n days.
 *
 * This program determines the min/max number of weekend days within a period of
 * 'n' days. It works by calculating the number of full weeks and then analyzing
 * the remaining days to find the minimum and maximum possible weekend days.
 *
 * Algorithm:
 * 1. Read the total number of days, n.
 * 2. The base number of rest days is `(n / 7) * 2`.
 * 3. The number of additional min/max rest days depends on the remainder `n % 7`.
 *    - rem=0: min=0, max=0
 *    - rem=1: min=0, max=1
 *    - rem=2-5: min=0, max=2
 *    - rem=6: min=1, max=2
 * 4. The code implements this logic through a series of if-else checks on the remainder.
 *
 * Time Complexity: O(1), as it involves a few arithmetic operations and comparisons.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;

  /**
   * Block Logic: A series of checks based on the remainder of n / 7.
   * This determines the number of extra weekend days in the partial week.
   */
  if (n % 7 == 0) {
    // Case: Exactly n full weeks. No extra days.
    cout << (n / 7) * 2 << " " << (n / 7) * 2 << "\n";
  } else if ((n + 1) % 7 == 0) {
    // Case: Remainder is 6. This is an obscure way to write `n % 7 == 6`.
    // Min: Any 6-day period must contain at least 1 weekend day.
    // Max: A 6-day period can contain 2 weekend days (e.g., Sat-Thu).
    cout << (int)(n / 7) * 2 + 1 << " " << (int)((n / 7) * 2 + 2) << "\n";
  } else if (n % 7 >= 2) {
    // Case: Remainder is 2, 3, 4, or 5.
    // Min: We can arrange the days to have 0 weekend days (e.g., Mon-Fri).
    // Max: We can arrange the days to have 2 weekend days (e.g., starting on Sat).
    cout << (int)(n / 7) * 2 << " " << (int)((n / 7) * 2 + 2) << "\n";
  } else if (n % 7 == 1) {
    // Case: Remainder is 1.
    // Min: 0 extra weekend days.
    // Max: 1 extra weekend day (if it's a Saturday or Sunday).
    cout << (int)(n / 7) * 2 << " " << (int)((n / 7) * 2 + 1) << "\n";
  }
  return 0;
}
