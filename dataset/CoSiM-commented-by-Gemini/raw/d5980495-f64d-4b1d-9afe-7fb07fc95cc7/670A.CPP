/**
 * @file 670A.CPP
 * @brief Solution for Codeforces Problem 670A, "Holidays".
 * @details The problem is to calculate the minimum and maximum possible number of holidays (weekends)
 * in a period of 'n' consecutive days. A week has 5 work days and 2 holidays.
 * The program calculates this based on the number of full weeks and the remaining days.
 *
 * @algorithm Case analysis based on modular arithmetic.
 * @complexity
 * Time: O(1), as all operations are constant time arithmetic.
 * Space: O(1), using a fixed amount of memory.
 */
// author @Nishant

#include <bits/stdc++.h> // Includes all standard C++ libraries
using namespace std;

int main() {
  int n;
  cin >> n;

  // Let 'weeks' be n / 7 and 'rem' be n % 7.

  // Case 1: The number of days is a perfect multiple of 7.
  if (!(n % 7)) {
    // Minimum and maximum holidays are the same: 2 for each full week.
    cout << (n / 7) * 2 << " " << (n / 7) * 2;
  } else {
    // Case 2: There are remaining days after the full weeks.
    
    // Subcase 2a: The remainder is 6.
    // Min holidays: The full weeks' holidays + 1 (if we start on a Tuesday, we get one extra holiday).
    // Max holidays: The full weeks' holidays + 2 (if we start on a Saturday, we get two extra holidays).
    if (n % 7 >= 6) {
      cout << (2 * (n / 7)) + 1 << " " << (2 * (n / 7)) + 2;
    } 
    // Subcase 2b: The remainder is between 2 and 5 (inclusive).
    // Min holidays: No extra holidays are guaranteed (e.g., start on Monday).
    // Max holidays: We can always fit a Saturday and Sunday in the remaining days.
    else if (n % 7 <= 5 && n % 7 >= 2) {
      cout << (n / 7) * 2 << " " << (2 * (n / 7)) + 2;
    }
    // Subcase 2c: The remainder is 1.
    // Min holidays: No extra holidays are guaranteed (e.g., start on Monday).
    // Max holidays: We can get one extra holiday (if we start on a Saturday).
    else if (n % 7 < 2) {
      cout << (n / 7) * 2 << " " << (2 * (n / 7)) + 1;
    }
  }
  cout << endl;
  return 0; // Indicate successful execution.
}