/**
 * @file raw/45deebad-514c-4012-b716-f63372d5666d/670A.CPP
 * @brief Solution to Codeforces problem 670A - Holidays.
 *
 * @algorithm
 * The problem asks for the minimum and maximum possible number of weekend days
 * (holidays) in a period of `n` consecutive days.
 *
 * The core idea is to consider a 7-day week cycle where 2 days are weekends.
 * The number of holidays depends on which day of the week the n-day period begins.
 *
 * To find the minimum holidays, we assume the period starts on a Monday. This
 * places the weekend days (Saturday and Sunday) at the end of the 7-day cycle.
 *
 * To find the maximum holidays, we assume the period starts on a Saturday. This
 * places the weekend days at the beginning of the 7-day cycle.
 *
 * The provided code simulates this day-by-day instead of using a direct
 * mathematical formula.
 *
 * For Maximum Holidays (`mx`):
 * - The code iterates from day `i = 1` to `n`.
 * - It checks `if ((i - 1) % 7 == 0 || (i - 2) % 7 == 0)`.
 * - This condition is true for days 1, 2, 8, 9, 15, 16, ...
 * - This effectively simulates the `n`-day period starting on a Saturday (day 1).
 *
 * For Minimum Holidays (`mn`):
 * - It checks `if (i % 7 == 0 || (i + 1) % 7 == 0)`.
 * - This condition is true for days 6, 7, 13, 14, ...
 * - This simulates the period starting on a Monday, where days 6 (Saturday) and 7 (Sunday)
 *   are the holidays.
 *
 * A more efficient O(1) solution would be:
 * - `full_weeks = n / 7`
 * - `remaining_days = n % 7`
 * - `min_holidays = full_weeks * 2 + max(0, remaining_days - 5)`
 * - `max_holidays = full_weeks * 2 + min(remaining_days, 2)`
 *
 * @complexity
 * - Time Complexity: O(n), due to the loop that iterates from 1 to n.
 * - Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

typedef long long ll;

int main() {
  ll n; // The total number of days.
  ll mx = 0; // Counter for the maximum possible holidays.
  ll mn = 0; // Counter for the minimum possible holidays.
  
  scanf("%lld", &n);

  /**
   * @brief This loop simulates the n days to count min and max holidays.
   * This is an O(n) approach. A more optimal O(1) mathematical solution exists.
   */
  for (int i = 1; i <= n; ++i) {
    // To maximize holidays, assume the period starts on a Saturday.
    // The weekend days would be the 1st, 2nd, 8th, 9th, etc.
    // This logic correctly identifies days that would be Saturday or Sunday
    // if the period started on a Saturday.
    if ((i - 1) % 7 == 0 || (i - 2) % 7 == 0)
      mx++;

    // To minimize holidays, assume the period starts on a Monday.
    // The weekend days would be the 6th, 7th, 13th, 14th, etc.
    // `i % 7 == 0` catches the 7th, 14th... day (Sundays).
    // `(i + 1) % 7 == 0` implies `i % 7 == 6`. This catches the 6th, 13th... day (Saturdays).
    if (i % 7 == 0 || (i + 1) % 7 == 0)
      mn++;
  }

  cout << mn << " " << mx << endl;
  
  return 0;
}

/*
Author's note found in the code:
Seg Ter Quar Quint Sex Sab Dom
                        1   2
3    4    5    6    7   8   9
10  11   12    13   14  15  16
*/
