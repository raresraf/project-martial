/**
 * @file 670A.CPP
 * @brief Solution for Codeforces Problem 670A - Holidays.
 *
 * This program calculates the minimum and maximum possible number of weekend
 * days (Saturday and Sunday) within a period of 'n' days, given that a week
 * has 7 days with 2 weekend days. The start day of the 'n' day period can vary.
 *
 * Algorithm: Modulo arithmetic and conditional logic. The core idea is to
 * determine how many full weeks are in 'n' days and then handle the remaining
 * days based on where they fall in a week to maximize or minimize weekend days.
 * Time Complexity: O(1) - constant time operations.
 * Space Complexity: O(1) - constant space usage.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @brief Main function to execute the Holidays problem logic.
   *
   * Reads the total number of days 'n' and then calculates and prints
   * the minimum and maximum possible number of weekend days within this period.
   * The calculations are based on dividing 'n' by 7 and handling the remainder.
   */
  // Declare an integer variable to store the total number of days.
  int n;
  // Read the total number of days.
  cin >> n;

  // Block Logic: Determine the minimum and maximum weekend days based on 'n'.
  // The logic considers different scenarios for the remainder of 'n' when divided by 7,
  // representing how many additional weekend days can be included beyond full weeks.

  // Scenario 1: 'n' is a perfect multiple of 7.
  // Pre-condition: The 'n' days form an exact number of full weeks.
  if (n % 7 == 0) {
    // Each full week has 2 weekend days (min and max are the same).
    cout << (n / 7) * 2 << " " << (n / 7) * 2 << "\n";
  }
  // Scenario 2: 'n+1' is a multiple of 7. This implies 'n' days end on a Friday,
  // meaning if we start on a Sunday (maximum weekend days), we get 2 weekend days
  // plus one more from the extra day.
  // Pre-condition: 'n' modulo 7 is 6 (e.g., n = 6, 13, 20...).
  else if ((n + 1) % 7 == 0) {
    // Minimum: (n/7)*2 full week weekends + 1 additional weekend day (e.g., Saturday if period ends Friday).
    // Maximum: (n/7)*2 full week weekends + 2 additional weekend days (e.g., if period starts Saturday/Sunday).
    cout << (int)(n / 7) * 2 + 1 << " " << (int)((n / 7) * 2 + 2) << "\n";
  }
  // Scenario 3: 'n' modulo 7 is 2 or more (but not 6, as covered above).
  // This handles cases where 'n' days include at least two days beyond a full week.
  // Pre-condition: 'n' modulo 7 is 2, 3, 4, or 5.
  else if (n % 7 >= 2) {
    // Minimum: (n/7)*2 full week weekends. The extra days might not include any weekend days for minimum.
    // Maximum: (n/7)*2 full week weekends + 2 additional weekend days (if the 2 extra days are Sat/Sun).
    cout << (int)(n / 7) * 2 << " " << (int)((n / 7) * 2 + 2) << "\n";
  }
  // Scenario 4: 'n' modulo 7 is 1.
  // This handles cases where 'n' days include one day beyond a full week.
  // Pre-condition: 'n' modulo 7 is 1 (e.g., n = 1, 8, 15...).
  else if (n % 7 == 1) {
    // Minimum: (n/7)*2 full week weekends. The single extra day might not be a weekend day.
    // Maximum: (n/7)*2 full week weekends + 1 additional weekend day (if the single extra day is Sat or Sun).
    cout << (int)(n / 7) * 2 << " " << (int)((n / 7) * 2 + 1) << "\n";
  }
  return 0;
}
