/**
 * @file 546A.CPP
 * @brief Calculates the amount of money a person needs to borrow to buy 'w' items (bananas),
 *        given that the first item costs 'k', the second '2k', and so on,
 *        and the person initially has 'n' money.
 *
 * Algorithm:
 * 1. Read the base cost (k), initial money (n), and number of items (w).
 * 2. Calculate the total cost of 'w' items using the sum of an arithmetic progression:
 *    Total Cost = k * (1 + 2 + ... + w) = k * (w * (w + 1) / 2).
 * 3. Determine the amount to borrow by subtracting the initial money (n) from the total cost.
 * 4. If the result is negative, set it to 0 (meaning no money needs to be borrowed).
 *    Otherwise, use the calculated borrowing amount.
 * 5. Print the final borrowing amount.
 *
 * Time Complexity: O(1) - The operations are constant time arithmetic calculations.
 * Space Complexity: O(1) - Only a few integer variables are used.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates the input of problem parameters, calculates the total cost,
 * and determines the required borrowing amount.
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Declares integer variables. 'k' for base cost, 'n' for initial money,
  //              'w' for number of items, and 'b' for the borrowing amount.
  // Note: Variables 'i', 'j', 'l', 'o' are declared but not used, indicating potential
  //       leftovers from previous code or an oversight.
  int n, b, i, j, l, o = 0, k, w;
  // Block Logic: Reads the integer values for k, n, and w from standard input.
  cin >> k;
  cin >> n;
  cin >> w;
  /**
   * Block Logic: Calculates the total cost of buying 'w' items and the amount to borrow.
   * The total cost is `k * (sum of 1 to w)`.
   * Inline: The sum of the first 'w' natural numbers is given by the formula w * (w + 1) / 2.
   * Functional Utility: Computes the net financial requirement before adjusting for negative results.
   */
  b = (k * (w + 1) * w / 2) - n;
  /**
   * Block Logic: Adjusts the borrowing amount.
   * If the calculated amount 'b' is negative (meaning the person has enough or more money),
   * it is set to 0, indicating no borrowing is needed.
   *
   * Pre-condition: 'b' holds the raw difference between total cost and available money.
   * Invariant: 'b' will be a non-negative value representing the actual amount to be borrowed.
   */
  if (b < 0) {
    b = 0;
  }
  // Block Logic: Prints the final calculated borrowing amount to standard output.
  cout << b;
  return 0;
}
