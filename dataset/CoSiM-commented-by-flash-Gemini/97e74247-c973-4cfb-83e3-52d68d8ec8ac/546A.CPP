/**
 * @file 546A.CPP
 * @brief Calculates the amount of money a soldier needs to borrow to buy 'w' bananas,
 *        given that the first banana costs 'k', the second '2k', and so on,
 *        and the soldier initially has 'n' money.
 *
 * Algorithm:
 * 1. Read the cost of the first banana (k), initial money (n), and number of bananas (w).
 * 2. Calculate the total cost of 'w' bananas using the sum of an arithmetic progression:
 *    Total Cost = k * (1 + 2 + ... + w) = k * (w * (w + 1) / 2).
 * 3. Determine the amount to borrow by subtracting the initial money (n) from the total cost.
 * 4. If the result is negative or zero (meaning enough or more money is available), output 0.
 *    Otherwise, output the calculated borrowing amount.
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
  // Block Logic: Declares variables to store the cost of the first banana (k),
  //              the soldier's initial money (n), and the number of bananas (w).
  int k, n, w;
  // Block Logic: Reads the integer values for k, n, and w from standard input.
  cin >> k >> n >> w;

  /**
   * Block Logic: Calculates the total cost of 'w' bananas.
   * The cost forms an arithmetic progression: k, 2k, 3k, ..., wk.
   * The sum of this series is k * (1 + 2 + ... + w).
   * Inline: The sum of the first 'w' natural numbers is given by the formula w * (w + 1) / 2.
   * Functional Utility: Computes the cumulative cost for purchasing all desired bananas.
   */
  int total = ((w * (w + 1)) / 2) * k;

  /**
   * Block Logic: Determines the amount of money the soldier needs to borrow.
   * If the total cost exceeds the initial money, the difference is the amount to borrow.
   * Otherwise, if the initial money is sufficient or more, no money needs to be borrowed.
   *
   * Pre-condition: `total` holds the calculated aggregate cost of bananas, and `n` is the available money.
   * Invariant: The output will be a non-negative integer representing the exact borrowing amount or zero.
   */
  if ((total - n) > 0)
    cout << total - n;
  else
    cout << "0";
}
