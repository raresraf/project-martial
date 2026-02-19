/**
 * @file 670A.CPP
 * @brief Solution to the "Holidays" problem.
 * @details This program calculates the minimum and maximum possible number of weekend days (Saturdays and Sundays)
 * that can occur in any continuous period of `n` days.
 *
 * Algorithm:
 * A week consists of 7 days, with 2 of them being weekend days.
 *
 * 1. **Base Calculation:**
 *    - First, calculate the number of full weeks within the `n` days: `weeks = n / 7`.
 *    - Each full week contributes exactly 2 weekend days. So, the base number of weekend days is `weeks * 2`.
 *    - Both minimum (`mn`) and maximum (`mx`) counters are initialized to this base value.
 *
 * 2. **Remaining Days:**
 *    - The variation in the total number of weekend days comes from the remaining days: `remainder = n % 7`.
 *
 * 3. **Maximum Weekend Days (`mx`):**
 *    - To maximize the number of weekend days in the remainder, we assume the `n`-day period starts on a Saturday.
 *    - If the remainder is 1, we get one extra weekend day (Saturday). `mx++`.
 *    - If the remainder is 2 or more, we get two extra weekend days (Saturday and Sunday). `mx += 2`.
 *      The maximum extra days can only be 2.
 *
 * 4. **Minimum Weekend Days (`mn`):**
 *    - To minimize the number of weekend days in the remainder, we assume the `n`-day period starts on a Monday.
 *    - If the remainder is 1 through 5, we can fit this period within the 5 weekdays, so there are no extra weekend days.
 *    - If the remainder is 6, the period must include at least one weekend day (e.g., Monday through Saturday). `mn++`.
 *    - If the remainder is 0, there are no extra days.
 *
 * 5. **Output:** The program prints the final calculated `mn` and `mx`.
 *
 * Time Complexity: O(1) for each test case.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  long int n, mn, mx;
  cin >> n;
  
  // Calculate the base number of weekend days from the full weeks.
  mn = (n / 7) * 2;
  mx = mn;

  /**
   * Block Logic: Calculate the maximum possible extra weekend days from the remainder.
   * This is achieved by assuming the n-day period starts on a Saturday.
   */
  if (n % 7 == 1)
    mx = mn + 1; // One remaining day could be a Saturday.
  else if (n % 7 > 1)
    mx = mn + 2; // Two or more remaining days could include Saturday and Sunday.

  /**
   * Block Logic: Calculate the minimum possible extra weekend days from the remainder.
   * This is achieved by assuming the n-day period starts on a Monday.
   */
  if (n % 7 == 6)
    mn = mn + 1; // A 6-day period must contain at least one weekend day.

  cout << mn << " " << mx << "\n";
}
