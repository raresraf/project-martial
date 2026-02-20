/**
 * @file 670A.CPP
 * @brief Calculates the minimum and maximum possible rest days in a period of n days.
 *
 * This program determines the minimum and maximum number of weekend days (rest days)
 * that can occur within a given period of 'n' days, assuming a standard 7-day week
 * with 2 rest days.
 *
 * Algorithm:
 * 1. Read the total number of days, n.
 * 2. Calculate the number of full weeks (k = n / 7) and the remaining days (rem = n % 7).
 * 3. Calculate Minimum Rest Days:
 *    - Base: k * 2 days from the full weeks.
 *    - For the remaining `rem` days, the minimum additional rest days are 1 if `rem` is 6
 *      (as any 6-day period must contain at least one weekend day), and 0 otherwise.
 * 4. Calculate Maximum Rest Days:
 *    - Base: k * 2 days from the full weeks.
 *    - For the remaining `rem` days, the maximum additional rest days are achieved by
 *      starting the period on a Saturday. This gives 1 extra day if `rem` is 1, and 2
 *      extra days if `rem` is 2 or more.
 * 5. Print the minimum and maximum values.
 *
 * Time Complexity: O(1), as the calculation involves a few arithmetic operations.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
#define OPTIMASI                                                               \
  cin.tie(0);                                                                  \
  cout.tie(0);                                                                 \
  ios_base::sync_with_stdio(0);
typedef long long int lli;
using namespace std;

int main() {
  // Functional Utility: A macro for optimizing I/O in competitive programming.
  OPTIMASI

  long long int n; // The total number of days.
  long long int k; // The number of full 7-day weeks.
  cin >> n;

  // Calculate the number of full weeks.
  k = n / 7;
  // Calculate the number of remaining days.
  n = n - k * 7; // Equivalent to n = n % 7;

  /**
   * Block Logic: Calculate and print the minimum number of rest days.
   * A 7-day week has 2 rest days. For the remaining 'n' days, we can
   * choose the start day to minimize rest days.
   * If n < 6, we can arrange them to have 0 rest days (e.g., start on Monday).
   * If n = 6, we must have at least 1 rest day.
   */
  if (n == 6)
    cout << k * 2 + 1 << " ";
  else
    cout << k * 2 << " ";

  /**
   * Block Logic: Calculate and print the maximum number of rest days.
   * To maximize, we assume the remaining 'n' days start on a Saturday.
   * If n = 1, we get 1 extra rest day (Saturday).
   * If n >= 2, we get 2 extra rest days (Saturday and Sunday).
   */
  if (n > 2)
    cout << k * 2 + 2;
  else {
    cout << k * 2 + n;
  }
  cout << endl;

  return 0;
}
