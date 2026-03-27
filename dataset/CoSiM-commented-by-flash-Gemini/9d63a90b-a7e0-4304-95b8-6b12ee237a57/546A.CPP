/**
 * @file 546A.CPP
 * @brief Calculates the amount of money a person needs to borrow to buy 'w' items (bananas),
 *        given that the first item costs 'k', the second '2k', and so on,
 *        and the person initially has 'n' money.
 *
 * Algorithm:
 * 1. Read the base cost (k), initial money (n), and number of items (w).
 * 2. Calculate the total cost by iteratively summing `k * i` for each item `i` from 1 to `w`.
 * 3. Determine the amount to borrow by subtracting the initial money (n) from the total cost.
 * 4. If the result is positive, output the calculated borrowing amount.
 *    Otherwise (initial money is sufficient or more), output 0.
 *
 * Time Complexity: O(W) where W is the number of items, due to the iterative summation.
 * Space Complexity: O(1) as only a few integer variables are used.
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

  // Block Logic: Declares 'n' for initial money (long long), 'rst' for total cost (long long, initialized to 0),
  //              and 'k', 'w' for base cost and number of items (int).
  long long n, rst = 0;
  int k, w;
  // Block Logic: Reads the integer values for k, n, and w from standard input.
  cin >> k >> n >> w;
  /**
   * Block Logic: Calculates the total cost of buying 'w' items by iteratively summing `k * i`.
   * Each item 'i' costs `k * i`.
   *
   * Invariant: At each iteration, 'rst' accumulates the cost of items bought so far,
   * up to the current item 'i'. This approach is less efficient than a closed-form
   * arithmetic series formula but functionally equivalent.
   */
  for (int i = 1; i <= w; i++) {
    // Inline: Adds the cost of the current item (k * i) to the running total 'rst'.
    rst += k * i;
  }
  /**
   * Block Logic: Determines the amount of money the person needs to borrow.
   * If the total calculated cost ('rst') minus initial money ('n') is positive,
   * that difference is the amount to borrow. Otherwise, no borrowing is needed.
   *
   * Pre-condition: 'rst' holds the total cost of items and 'n' is the available money.
   * Invariant: The output will be a non-negative integer representing the exact borrowing amount or zero.
   */
  if (rst - n > 0)
    cout << rst - n;
  else
    cout << 0;
}
