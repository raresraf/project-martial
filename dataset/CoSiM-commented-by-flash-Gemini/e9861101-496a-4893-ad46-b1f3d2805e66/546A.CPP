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

#include <bits/stdc++.h>

using namespace std;

#define ll long long

/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates the input of problem parameters, calculates the total cost,
 * and determines the required borrowing amount.
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Declares integer variables 'k' for base cost, 'n' for initial money,
  //              'w' for number of items.
  int k, n, w;
  // Block Logic: Reads the integer values for k, n, and w from standard input.
  cin >> k >> n >> w;
  /**
   * Block Logic: Calculates the total cost of buying 'w' items and the potential borrowing amount.
   * The total cost of 'w' items is `k * (sum of 1 to w)`.
   * Inline: Uses the arithmetic series sum formula: `k * w * (w + 1) / 2`.
   * Functional Utility: Determines the net financial requirement before adjusting for negative results.
   */
  int ris = k * (w) * (w + 1) / 2 - n;
  /**
   * Block Logic: Determines the amount of money the person needs to borrow.
   * If the calculated amount 'ris' is positive, it means money needs to be borrowed;
   * otherwise, no borrowing is needed, and 0 is printed.
   *
   * Pre-condition: 'ris' holds the raw difference between total cost and available money.
   * Invariant: The output will be a non-negative integer representing the exact borrowing amount or zero.
   */
  if (ris > 0)
    cout << ris << "\n";
  else
    cout << 0 << "\n";
  return 0;
}
