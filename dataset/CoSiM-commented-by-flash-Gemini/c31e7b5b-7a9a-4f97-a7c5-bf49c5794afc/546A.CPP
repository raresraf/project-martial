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
 * 4. If the result is negative or zero (meaning enough or more money is available), output 0.
 *    Otherwise, output the calculated borrowing amount.
 *
 * Time Complexity: O(1) - The operations are constant time arithmetic calculations.
 * Space Complexity: O(1) - Only a few integer variables are used.
 */

#include <stdio.h>
using namespace std;

/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates the input of problem parameters, calculates the total cost,
 * and determines the required borrowing amount.
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Declares variables for the base cost (k), number of items (w),
  //              and initial money (n). 'r' is used for the calculated amount to borrow.
  int k, w;
  long n;
  // Block Logic: Reads the integer values for k, n, and w from standard input.
  // Note: '%lu' is used for 'unsigned long', but 'n' is declared as 'long'. This might be a mismatch.
  scanf("%d %lu %d", &k, &n, &w);
  /**
   * Block Logic: Calculates the total cost and then the amount to borrow.
   * The total cost of 'w' items is `k * (sum of 1 to w)`.
   * Inline: The sum of the first 'w' natural numbers is given by the formula w * (w + 1) / 2.
   * The amount to borrow is the total cost minus the initial money 'n'.
   * Functional Utility: Computes the net financial requirement after purchasing all items.
   */
  long r = (k * (w * (w + 1) / 2) - n);
  // Inline: Uses a ternary operator to ensure the borrowing amount is not negative.
  // If 'r' is positive, it means money needs to be borrowed; otherwise, it's 0.
  r = r > 0 ? r : 0;
  // Block Logic: Prints the calculated borrowing amount to standard output.
  printf("%lu", r);
  return 0;
}
