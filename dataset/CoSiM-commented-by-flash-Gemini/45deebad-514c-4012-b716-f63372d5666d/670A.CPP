/**
 * @file 670A.CPP
 * @brief Solution to a competitive programming problem involving day counting.
 *
 * This program calculates the minimum and maximum number of weekend days (Saturday and Sunday)
 * that can occur within a period of 'n' consecutive days. This is based on the starting day
 * of the 'n' day period.
 *
 * Algorithm: Iterative simulation / Pattern Recognition. The core idea is that in any 7-day
 * period, there are exactly 2 weekend days. The minimum and maximum arise from how the 'n'
 * days align with the start and end of weeks. The loop effectively simulates all possible
 * starting days of the week.
 * Time Complexity: O(N) due to the loop iterating 'n' times. For practical competitive programming constraints,
 * this is often acceptable if 'N' is not excessively large (e.g., up to 10^5-10^6).
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

typedef long long ll;

/**
 * @brief Main function to calculate minimum and maximum weekend days within 'n' days.
 *
 * Reads the total number of days 'n' from standard input. It then iterates
 * through 'n' days, simulating the counting of weekend days based on two
 * different alignment rules to find the minimum and maximum possible counts.
 *
 * @return 0 on successful execution.
 */
int main() {
  // Declare long long integers for the total number of days 'n',
  // and for the counts of maximum ('mx') and minimum ('mn') weekend days.
  ll n, mx, mn;
  // Initialize 'mx' and 'mn' to 0.
  mx = mn = 0;
  // Read the total number of days 'n' from standard input.
  scanf("%lld", &n);
  /**
   * @brief Block Logic: Simulates counting weekend days based on 'n' days.
   *
   * This loop iterates 'n' times, with 'i' representing each day from 1 to 'n'.
   * It calculates the maximum and minimum possible weekend days by checking
   * different alignment conditions within a 7-day week cycle.
   *
   * Invariant: 'i' represents the current day being processed (1-indexed).
   * 'mx' and 'mn' accumulate counts based on specific weekend alignment logic.
   */
  for (int i = 1; i <= n; ++i) {
    // Pre-condition: Check if day 'i' or 'i-1' falls on a weekend day (using modulo 7).
    // This condition models a scenario to maximize weekend days.
    if ((i - 1) % 7 == 0 || (i - 2) % 7 == 0)
      mx++;
    // Pre-condition: Check if day 'i' or 'i+1' falls on a weekend day (using modulo 7).
    // This condition models a scenario to minimize weekend days.
    if (i % 7 == 0 || (i + 1) % 7 == 0)
      mn++;
  }
  // Print the calculated minimum and maximum weekend days.
  cout << mn << " " << mx << endl;
  return 0;
}

/*
Seg Ter Quar Quint Sex Sab Dom
                        1   2
3    4    5    6    7   8   9
10  11   12    13   14  15  16
*/