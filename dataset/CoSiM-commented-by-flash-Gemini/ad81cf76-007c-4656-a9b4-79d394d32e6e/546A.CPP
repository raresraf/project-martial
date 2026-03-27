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
 * 4. If the result is positive, output the calculated borrowing amount.
 *    Otherwise (initial money is sufficient or more), output 0.
 *
 * Time Complexity: O(1) - The operations are constant time arithmetic calculations.
 * Space Complexity: O(1) - Only a few integer variables are used.
 */

#include <stdio.h>
/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates the input of problem parameters, calculates the total cost,
 * and determines the required borrowing amount.
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Declares variables for the base cost (k), initial money (n),
  //              number of items (w), total cost (s), and the amount to borrow (ans).
  long long n, w, k, s, ans;
  // Block Logic: Reads the integer values for k, n, and w from standard input.
  scanf("%lld %lld %lld", &k, &n, &w);
  /**
   * Block Logic: Calculates the total cost of buying 'w' items.
   * The cost forms an arithmetic progression: k, 2k, 3k, ..., wk.
   * Functional Utility: Computes the cumulative cost for purchasing all desired items.
   */
  s = k * (w * (w + 1) / 2); // Inline: Uses the arithmetic series sum formula: k * (w * (w + 1) / 2).
  // Block Logic: Calculates the difference between the total cost and the initial money.
  ans = s - n;
  /**
   * Block Logic: Determines the amount of money the person needs to borrow.
   * If 'ans' is positive, it means money needs to be borrowed; otherwise,
   * no borrowing is needed, and 0 is printed.
   *
   * Pre-condition: 's' holds the total cost of items and 'n' is the available money.
   * Invariant: The output will be a non-negative integer representing the exact borrowing amount or zero.
   */
  if (ans > 0) {
    printf("%lld", ans);
  } else
    printf("0");
  return 0;
}