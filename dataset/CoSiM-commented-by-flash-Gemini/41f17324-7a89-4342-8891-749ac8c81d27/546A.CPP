/**
 * @file 546A.CPP
 * @brief Calculates the amount of money a soldier needs to borrow to buy bananas.
 *
 * This program solves a problem where a soldier wants to buy `w` bananas.
 * The first banana costs `k` dollars, the second `2k` dollars, and so on,
 * the `w`-th banana costs `wk` dollars. The soldier has `n` dollars initially.
 * The program calculates the total cost of all `w` bananas and then determines
 * how much money the soldier needs to borrow. If the soldier has enough money,
 * the amount to borrow is 0.
 *
 * Algorithm: Direct calculation using the sum of an arithmetic series. The total cost
 *            is `k + 2k + ... + wk = k * (1 + 2 + ... + w) = k * w * (w + 1) / 2`.
 *            The amount to borrow is `max(0, total_cost - initial_money)`.
 * Time Complexity: O(1) as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) as only a few scalar variables are used to store inputs and the result.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  /**
   * @brief Conditional block for redirecting standard I/O to files for local debugging.
   *
   * This block is typically active when the code is run outside an online judge environment.
   * It redirects standard input from "input.txt" and standard output to "output.txt".
   */
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  /**
   * @brief Main function to calculate the amount of money to borrow.
   *
   * Reads the cost of the first banana (`k`), initial money (`n`), and number of bananas (`w`).
   * Calculates the total cost using an arithmetic series formula.
   * Determines and prints the amount to borrow, or 0 if no borrowing is needed.
   */
  // @param k The cost of the first banana (subsequent bananas cost 2k, 3k, etc.).
  // @param n The amount of money the soldier initially has.
  // @param w The number of bananas the soldier wants to buy.
  // @param sum Stores the total cost of buying `w` bananas.
  long long int k, n, w, sum;
  // Reads k, n, and w from standard input.
  cin >> k >> n >> w;
  // @brief Calculates the total cost of `w` bananas using the arithmetic series sum formula.
  // Logic: The costs are k, 2k, ..., wk. The sum is k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
  sum = (k * (w + 1) * (w)) / 2;

  /**
   * @brief Conditional block to determine and print the amount to borrow.
   * Logic: If `total_cost - initial_money` is non-negative, that's the amount to borrow.
   *        Otherwise, if the soldier has enough or more money, 0 is printed.
   */
  if (sum - n >= 0) {
    cout << sum - n; // Prints the amount to borrow.
  } else {
    cout << "0"; // Prints 0 if no money needs to be borrowed.
  }

  return 0;
}