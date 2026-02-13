/**
 * @file 670A.CPP
 * @brief Calculates the minimum and maximum number of weekend days in a period of n days.
 * @details The problem is to find the minimum and maximum possible number of days off (Saturdays and Sundays)
 * over a continuous period of `n` days, depending on the starting day of the week.
 *
 * Algorithm:
 * 1. Read the total number of days, `n`.
 * 2. Calculate the number of full weeks in `n` days and the corresponding number of weekend days.
 *    - Each full week has 2 weekend days.
 *    - `d = (n / 7) * 2` gives the weekend days from full weeks.
 * 3. Calculate the remaining days, `r = n % 7`.
 * 4. Calculate minimum additional weekend days from the remainder `r`:
 *    - To minimize, we assume the period starts on a Monday. The first 5 days have no weekend.
 *    - Additional days off are `max(0, r - 5)`. For r in [0,6], this is equivalent to `(r == 6)`.
 *    - Minimum total days off = `d + (r == 6)`.
 * 5. Calculate maximum additional weekend days from the remainder `r`:
 *    - To maximize, we assume the period starts on a Saturday.
 *    - Additional days off are `min(r, 2)`.
 *    - Maximum total days off = `d + min(r, 2)`. The code uses `(r > 2 ? 2 : r)` which is equivalent to `min(r,2)` for r>=0.
 * 6. Print the minimum and maximum number of days off.
 *
 * Time Complexity: O(1).
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;
int main() {
  long n, d, r;
  cin >> n;
  
  // d: weekend days from full weeks.
  d = n / 7 * 2;
  // r: remaining days after accounting for full weeks.
  r = n % 7;
  
  // The minimum number of weekend days is calculated by assuming the period starts on a Monday.
  // This results in `d + max(0, r - 5)`, which for r in [0,6] simplifies to `d + (r == 6)`.
  // The maximum number of weekend days is calculated by assuming the period starts on a Saturday.
  // This results in `d + min(r, 2)`, which is implemented as `d + (r > 2 ? 2 : r)`.
  cout << d + (r == 6) << ' ' << d + (r > 2 ? 2 : r);
}
