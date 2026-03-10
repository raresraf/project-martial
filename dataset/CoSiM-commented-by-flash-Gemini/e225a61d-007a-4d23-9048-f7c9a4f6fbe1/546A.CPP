/**
 * @file 546A.CPP
 * @brief This program solves the "Soldier and Bananas" problem (Codeforces 546A).
 *   A soldier wants to buy `w` bananas. The first costs `k` dollars, the second `2k`, and so on.
 *   He has `n` dollars. The task is to calculate how much money he needs to borrow.
 *   If he has enough money, the amount to borrow is 0.
 *   Problem description: https://codeforces.com/problemset/problem/546/A
 * * Algorithm: Iterative sum and conditional calculation.
 *   1. It calculates the total cost (`cost`) by summing the cost of each banana
 *      (`i * k` for `i` from 1 to `w`) using a loop.
 *   2. If the `cost` is less than or equal to the money the soldier has (`n`),
 *      he needs to borrow 0. Otherwise, he needs to borrow `cost - n`.
 * Time Complexity: O(W) where W is the number of bananas, due to the loop for summing individual costs.
 *   For typical competitive programming constraints (W up to 1000), this is efficient enough.
 * Space Complexity: O(1) for a few integer variables.
 */
//
// Created by Conor on 5/28/2020.
//

#include <iostream> // Required for standard input/output operations.

using namespace std;

/**
 * @brief Main function to calculate the amount of money a soldier needs to borrow.
 * Functional Utility: Reads the cost parameters, calculates the total cost of bananas iteratively,
 *                     determines the amount to borrow, and prints the result.
 */
int main() {
  int k, n, w; // `k`: cost of the first banana, `n`: money soldier has, `w`: number of bananas to buy.
  // Block Logic: Reads the input values for k, n, and w.
  cin >> k >> n >> w;

  int cost = 0; // Stores the total calculated cost of all bananas.

  /**
   * Block Logic: Calculates the total cost `cost` for `w` bananas iteratively.
   * Pre-condition: `k`, `w` are positive integers.
   * Invariant: `cost` accumulates the sum of `i * k` for `i` from 1 up to `w`.
   */
  for (int i = 1; i <= w; ++i) {
    cost += i * k; // Adds the cost of the `i`-th banana (which is `i * k`) to the total cost.
  }

  /**
   * Conditional Logic: Determines whether the soldier needs to borrow money.
   * If `cost` is less than or equal to `n` (money he has), he needs to borrow 0.
   * Otherwise, he needs to borrow `cost - n`.
   */
  if (cost <= n) {
    cout << 0; // Prints 0 if no money needs to be borrowed.
  } else {
    cout << cost - n; // Prints the positive amount to borrow.
  }

  return 0; // Indicates successful program termination.
}