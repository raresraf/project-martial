/**
 * @file 670A.CPP
 * @brief This program calculates the minimum and maximum possible number of
 * weekend days (Saturday and Sunday) within a given total number of days 'n'.
 * It assumes a standard 7-day week cycle and that the 'n' days can start on any day of the week.
 * Algorithm: Case-based analysis depending on whether 'n' is greater than 7 or not,
 * and calculations based on integer division and modulo operations for weeks and remaining days.
 * Time Complexity: O(1) as it involves constant number of arithmetic operations.
 * Space Complexity: O(1) for storing a few integer variables.
 */
#include <iostream> // Required for input/output operations (std::cin, std::cout).
#include <algorithm> // Required for std::min (to simplify maxD calculation).
using namespace std;

int main() {
  int n;    // Stores the total number of days.
  int minD; // Stores the minimum possible number of weekend days.
  int maxD; // Stores the maximum possible number of weekend days.
  // Reads the total number of days 'n'.
  cin >> n;

  /**
   * Conditional Logic: Handles the calculation based on whether 'n' is greater than 7 days.
   * If 'n' > 7, it means 'n' spans at least one full week.
   */
  if (n > 7) {
    // Calculate minimum weekend days:
    // Each full 7-day period contributes 2 weekend days (one Saturday, one Sunday).
    // For the remaining days (n % 7), if there are 6 days, one additional weekend day (Saturday) is guaranteed.
    minD = (n / 7) * 2;
    if (n % 7 == 6) {
      minD++;
    }

    // Calculate maximum weekend days:
    // Each full 7-day period contributes 2 weekend days.
    // To maximize additional weekend days, consider starting the 'n' days on a Saturday or Sunday.
    // This allows for up to 2 extra weekend days from the remaining (n % 7) days.
    maxD = (n / 7) * 2;
    maxD += std::min(2, n % 7);

  }
  /**
   * Conditional Logic: Handles the calculation when 'n' is less than or equal to 7 days.
   * This involves specific logic for small 'n' values.
   */
  else { // n <= 7
    if (n <= 5) { // If n is 1, 2, 3, 4, or 5.
      minD = 0;   // It's possible to have 0 weekend days (e.g., start on Monday, end on Friday).
      if (n <= 2)
        maxD = n; // If n=1 (maxD=1), n=2 (maxD=2). (Can pick Sat or Sun, or Sat and Sun).
      else
        maxD = 2; // If n=3,4,5 (can still only have max 2 weekend days).
    } else { // n is 6 or 7.
      if (n == 6)
        minD = 1; // If n=6, min weekend days is 1 (e.g., Mon-Sat includes 1 Sat or Tue-Sun includes 1 Sun).
      else // n == 7
        minD = 2; // If n=7, min weekend days is 2 (a full week must include 1 Sat and 1 Sun).
      maxD = 2; // Max weekend days for 6 or 7 days is always 2.
    }
  }
  // Prints the calculated minimum and maximum weekend days separated by a space, followed by a newline.
  cout << minD << " " << maxD << endl;
  // Standard program exit.
  return 0;
}
