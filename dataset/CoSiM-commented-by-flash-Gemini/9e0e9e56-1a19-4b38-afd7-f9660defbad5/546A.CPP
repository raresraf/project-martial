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
 * 4. If the result is negative (meaning enough or more money is available), output 0.
 *    Otherwise, output the calculated borrowing amount.
 *
 * Time Complexity: O(W) where W is the number of items, due to the iterative summation.
 * Space Complexity: O(1) as only a few integer variables are used.
 */

#include <iostream>
using namespace std;
// Block Logic: Global declarations for base cost (k), initial money (n),
//              number of items (w), and accumulated total cost (pas).
// Functional Utility: These variables store problem parameters and the running total cost.
// Note: While functional, global variable declarations are generally discouraged in modern C++
// for reasons like increased coupling and potential for unintended side effects.
int k, n, w, pas = 0;
int main() {
  // Block Logic: Reads the integer values for k, n, and w from standard input.
  cin >> k >> n >> w;
  /**
   * Block Logic: Calculates the total cost of buying 'w' items.
   * Each item 'i' costs `k * i`. This loop iteratively sums these costs.
   *
   * Invariant: At each iteration, 'pas' accumulates the cost of items bought so far,
   * up to the current item 'i'.
   */
  for (int i = 1; i <= w; i++) {
    // Inline: Adds the cost of the current item (k * i) to the running total 'pas'.
    pas += k * i;
  }
  /**
   * Block Logic: Determines the amount of money the person needs to borrow.
   * If the total calculated cost ('pas') exceeds the initial money ('n'),
   * the difference is the amount to borrow. Otherwise, no borrowing is needed.
   *
   * Pre-condition: 'pas' holds the total cost of items and 'n' is the available money.
   * Invariant: The output will be a non-negative integer representing the exact borrowing amount or zero.
   */
  if (pas - n < 0) {
    cout << 0;
  } else {
    cout << pas - n;
  }
}