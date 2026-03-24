/**
 * @file 670A.CPP
 * @brief Solution to a competitive programming problem: "Holidays".
 *
 * Calculates the minimum and maximum possible number of weekend days (Saturday and Sunday)
 * within a period of 'n' days, given that the 'n' days can start on any day of the week.
 *
 * Algorithm: Mathematical calculation using integer division and modulo operations.
 * Time Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Main function to solve the "Holidays" problem.
 * Reads the total number of days 'n', calculates the minimum and maximum possible
 * weekend days within this period, and prints them to standard output.
 * @return 0 on successful execution (implicitly, as it's common in competitive programming).
 */
int main() {
  /**
   * Declares variables:
   * 'n': The total number of days in the period.
   * 'minimum': Stores the calculated minimum number of weekend days.
   * 'maximum': Stores the calculated maximum number of weekend days.
   * Reads the value for 'n' from standard input.
   * Pre-condition: Standard input provides a single integer 'n'.
   */
  long int n, minimum, maximum;
  cin >> n;
  /**
   * Block Logic: Calculates the minimum and maximum possible weekend days within 'n' days.
   * Base calculation: For every full week (n / 7), there are 2 weekend days.
   * Maximum calculation:
   *  - If there's 1 remaining day (n % 7 > 0), one additional weekend day is possible.
   *  - If there are 2 or more remaining days (n % 7 > 1), a second additional weekend day is possible.
   * Minimum calculation:
   *  - If exactly 6 days remain (n % 7 == 6), it's possible to include one additional weekend day
   *    by carefully choosing the start day of the 'n' day period.
   * Invariant: 'minimum' and 'maximum' will hold the calculated bounds for weekend days.
   * Pre-condition: 'n' is a non-negative integer.
   */
  minimum = ((n / 7) * 2);
  maximum = ((n / 7) * 2);
  if (n % 7 > 0)
    maximum++;
  if (n % 7 > 1)
    maximum++;
  if (n % 7 == 6)
    ++minimum;
  /**
   * Prints the calculated minimum and maximum number of weekend days.
   * Post-condition: Two integers (minimum and maximum) are printed to standard output, separated by a space.
   */
  cout << minimum << " " << maximum;
}
