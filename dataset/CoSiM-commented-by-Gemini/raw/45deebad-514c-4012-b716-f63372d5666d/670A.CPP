/**
 * @file 670A.CPP
 * @brief Solution to a competitive programming problem.
 * @details This program calculates the minimum and maximum possible number of weekend days
 * (Saturdays and Sundays) within a period of 'n' consecutive days. The 'n' day period
 * can start on any day of the week.
 *
 * Algorithm:
 * The program iterates through each day from 1 to 'n' and calculates two scenarios:
 * 1. Maximum weekends ('mx'): This scenario assumes the period starts on a Saturday.
 *    Weekend days would then fall on days i such that i % 7 == 1 (Saturday) or i % 7 == 2 (Sunday).
 * 2. Minimum weekends ('mn'): This scenario assumes the period starts on a Monday.
 *    Weekend days would then fall on days i such that i % 7 == 6 (Saturday) or i % 7 == 0 (Sunday).
 *
 * The logic simulates counting these days over the 'n' day period.
 *
 * Time Complexity: O(n) due to the single loop from 1 to n.
 * Space Complexity: O(1).
 *
 * Note: While the O(n) approach is simple, a more optimal O(1) solution exists by
 * calculating the number of full weeks (n / 7) and then analyzing the remaining
 * days (n % 7).
 */
#include <bits/stdc++.h>

using namespace std;

typedef long long ll;

int main() {
  ll n, mx, mn;
  mx = mn = 0;
  scanf("%lld", &n);

  /**
   * @block
   * @description This loop iterates through each of the 'n' days to count the
   * minimum and maximum possible weekend days.
   * Invariant: At the end of each iteration 'i', 'mn' and 'mx' hold the counts
   * for the first 'i' days under their respective scenarios.
   */
  for (int i = 1; i <= n; ++i) {
    /**
     * @block
     * @description Calculates the maximum number of weekend days. This assumes
     * the n-day period begins on a Saturday. Weekend days in this scenario
     * correspond to the 1st and 2nd day of each 7-day cycle.
     */
    if ((i - 1) % 7 == 0 || (i - 2) % 7 == 0)
      mx++;
    
    /**
     * @block
     * @description Calculates the minimum number of weekend days. This assumes
     * the n-day period begins on a Monday. Weekend days in this scenario
     * correspond to the 6th and 7th day of each 7-day cycle.
     */
    if (i % 7 == 0 || (i + 1) % 7 == 0)
      mn++;
  }
  
  // Output the final minimum and maximum counts.
  cout << mn << " " << mx << endl;
  return 0;
}

/*
Seg Ter Quar Quint Sex Sab Dom
                        1   2
3    4    5    6    7   8   9
10  11   12    13   14  15  16
*/
