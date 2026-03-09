/**
 * @file 670A.CPP
 * @brief Calculates the minimum and maximum possible weekend days in a given period.
 *
 * This program solves the "Holidays" problem from Codeforces. Given a total
 * number of days `n`, it calculates the minimum and maximum number of weekend
 * days (days off) that can occur within that period. A standard week is
 * considered to have 5 workdays and 2 weekend days.
 *
 * @algorithm Mathematical/Logical Calculation
 * @complexity
 * - Time: O(1) per test case, as it involves a few arithmetic operations.
 * - Space: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // n: total number of days.
  // m: number of full 7-day weeks in n.
  // r: number of remaining days after accounting for full weeks.
  // mn: minimum possible weekend days.
  // mx: maximum possible weekend days.
  // i: unused variable.
  long long i, n, m, r, mn, mx;

  // The while loop structure allows for processing multiple test cases.
  while (cin >> n) {
    r = n % 7; // Calculate remaining days.
    m = n / 7; // Calculate full weeks.

    // Base case: Every full week has exactly 2 weekend days.
    mx = mn = m * 2;

    /**
     * @brief Block Logic: Calculate minimum additional weekend days.
     * To minimize weekend days, we assume the period starts as late in the
     * week's work cycle as possible (e.g., on a Monday).
     */
    if (r == 6) {
      // If 6 days remain, we must cross at least one weekend day (e.g., Sat).
      mn++;
    }

    /**
     * @brief Block Logic: Calculate maximum additional weekend days.
     * To maximize weekend days, we assume the period starts at the beginning
     * of a weekend (e.g., on a Saturday).
     */
    if (r <= 2) {
      // If 1 or 2 days remain, they can both be weekend days.
      mx += r;
    } else {
      // If more than 2 days remain, we can still only have a maximum of 2
      // extra weekend days from that partial week.
      mx += 2;
    }
    cout << mn << " " << mx << endl;
  }
  return 0;
}