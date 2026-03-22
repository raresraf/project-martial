/**
 * @file 546A.CPP
 * @brief Solution to Codeforces problem 546A - Soldier and Bananas.
 *        This program calculates the amount of money a soldier needs to borrow from a friend
 *        to buy a certain number of bananas, given the cost of the first banana and his initial money.
 *        The cost of each subsequent banana increases arithmetically. This version calculates
 *        the total cost iteratively.
 *
 * Algorithm:
 *   1. Reads the cost of the first banana (`k`), the soldier's initial money (`n`),
 *      and the number of bananas to buy (`w`).
 *   2. Initializes a `total` variable to 0.
 *   3. Iterates from `i = 1` to `w`, adding `i * k` to `total` in each step. This computes
 *      the cumulative cost of buying `w` bananas.
 *   4. Calculates `result` as `total - n`. This represents the net amount needed (or surplus).
 *   5. If `result` is negative (meaning the soldier has enough money or more), sets `result` to 0.
 *   6. Prints the final `result`, which is the amount of money the soldier needs to borrow.
 *
 * Time Complexity: O(w), where `w` is the number of bananas, due to the loop.
 * Space Complexity: O(1) for storing a few integer variables.
 */

#include <iostream>
using namespace std;

int main() {
  /**
   * @brief Main function that solves the "Soldier and Bananas" problem.
   *
   * Reads problem parameters and iteratively calculates the total cost of bananas.
   * Determines the deficit (amount to borrow) and prints it.
   *
   * @return int Returns 0 upon successful execution.
   */
  int k, n, w; // k: cost of first banana, n: initial money, w: number of bananas to buy.
  cin >> k >> n >> w;

  int total = 0; // Accumulates the total cost of bananas.
  /**
   * Block Logic: Calculates the cumulative cost of buying `w` bananas.
   * The cost of the `i`-th banana is `i * k`.
   * Invariant: `total` stores the sum of costs for bananas from 1 to `i-1`.
   * Pre-condition: `k`, `n`, `w` are positive integers.
   */
  for (int i = 1; i <= w; i++)
    total += i * k; // Adds the cost of the current banana to the total.

  int result = total - n; // Calculates the money needed (or surplus).
  /**
   * Conditional Logic: Ensures that the output is non-negative.
   * If the soldier has enough money, the amount to borrow is 0.
   */
  if (result < 0)
    result = 0; // If there's a surplus, set borrowed amount to 0.
  cout << result << endl; // Prints the amount to be borrowed.

  return 0;
}