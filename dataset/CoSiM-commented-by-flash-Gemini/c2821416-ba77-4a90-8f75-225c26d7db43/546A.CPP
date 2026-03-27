/**
 * @file 546A.CPP
 * @brief Calculates the amount of money a person needs to borrow to buy 'w' items (bananas),
 *        given that the first item costs 'k', the second '2k', and so on,
 *        and the person initially has 'n' money.
 *
 * Algorithm:
 * 1. Read the base cost (k), initial money (n), and number of items (w).
 * 2. Calculate the sum of the series 1 + 2 + ... + w.
 * 3. Multiply this sum by the base cost 'k' to get the total cost 'm'.
 * 4. Determine the amount to borrow:
 *    - If initial money 'n' is less than or equal to the total cost 'm', output 'm - n'.
 *    - Otherwise (initial money is sufficient or more), output 0.
 *
 * Time Complexity: O(1) - The operations are constant time arithmetic calculations.
 * Space Complexity: O(1) - Only a few integer variables are used.
 */

#include <iostream>
using namespace std;
/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates the input of problem parameters, calculates the total cost,
 * and determines the required borrowing amount.
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Declares variables for the number of items 'n',
  //              the number of items 'w', the base cost 'k',
  //              and the calculated total cost 'm'.
  long long int n, w, k, m = 0;
  // Block Logic: Reads the base cost 'k' from standard input.
  cin >> k;
  // Block Logic: Reads the initial money 'n' from standard input.
  cin >> n;
  // Block Logic: Reads the number of items 'w' from standard input.
  cin >> w;
  /**
   * Block Logic: Calculates the sum of the series 1 + 2 + ... + w.
   * Inline: The sum of the first 'w' natural numbers is given by the formula w * (w + 1) / 2.
   * Functional Utility: Computes the sum required for the total cost calculation.
   */
  m = (w * (w + 1)) / 2;
  // Block Logic: Multiplies the sum by the base cost 'k' to get the total cost 'm'.
  m = m * k;
  /**
   * Block Logic: Determines the amount of money the person needs to borrow.
   * If the initial money 'n' is less than or equal to the total cost 'm', the difference is output.
   * Otherwise, if the initial money is sufficient or more, 0 is output.
   *
   * Pre-condition: 'n' holds the initial money, and 'm' holds the total calculated cost.
   * Invariant: The output will be a non-negative integer representing the exact borrowing amount or zero.
   */
  if (n <= m)
    cout << m - n;
  else
    cout << 0;
}