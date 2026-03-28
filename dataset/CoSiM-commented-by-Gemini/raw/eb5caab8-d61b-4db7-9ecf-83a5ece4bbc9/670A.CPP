/**
 * @file 670A.CPP
 * @brief Solves the "Holidays" problem from Codeforces.
 *
 * This program calculates the minimum and maximum possible number of weekend days
 * (days off) that can occur within a given period of `n` consecutive days.
 *
 * @algorithm The code separates the logic for `n > 7` and `n <= 7`.
 * - For n > 7: `minD` is found by counting full weeks and any remainder that must
 *   include a weekend. `maxD` is found using a convoluted but correct calculation
 *   that simulates starting the period on a weekend.
 * - For n <= 7: It uses `if/else` to handle the specific outcomes for small n.
 *
 * @time_complexity O(1), as it involves direct arithmetic calculations.
 * @space_complexity O(1).
 */
#include <iostream>
using namespace std;

int main() {
  int n, minD, maxD;
  cin >> n;
  
  if (n > 7) {
    // Calculate min days off before modifying n.
    minD = (n / 7) * 2;
    if (n % 7 > 5) // If remainder is 6, must include one extra weekend day.
      minD++;
    
    // The following block calculates max days off. It modifies n.
    maxD = 0;
    maxD += 2; // Assume we start on a weekend to maximize days off.
    n -= 2;    // Account for the first two days.
    maxD += (n / 7) * 2; // Add days off from the remaining full weeks.
    if (n % 7 > 5) // Add another day if the final partial week is long.
      maxD++;
  } else {
    // Logic for periods of 7 days or less
    if (n <= 5) {
      minD = 0;
      if (n <= 2)
        maxD = n;
      else
        maxD = 2;
    } else { // n is 6 or 7
      if (n == 6)
        minD = 1;
      else // n == 7
        minD = 2;
      maxD = 2;
    }
  }
  cout << minD << " " << maxD << endl;
  return 0;
}
