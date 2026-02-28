/**
 * @file 546A.CPP
 * @brief Iterative solution for the "Soldier and Bananas" problem.
 * @details This program calculates the amount of money a soldier needs to borrow.
 * The cost of bananas increases linearly. This solution calculates the total cost
 * by iterating from 1 to `w` and summing up the cost of each banana.
 *
 * @algorithm Iterative Summation
 * The program uses a simple for-loop to simulate the purchase of each banana and
 * accumulate the total cost. The cost of the `i`-th banana is `k * i`.
 * The amount to borrow is the total accumulated cost minus the soldier's initial money.
 *
 * @complexity
 * - Time: O(W), where W is the number of bananas to buy. The loop runs W times.
 * - Space: O(1), for storing a few integer variables.
 */

#include <bits/stdc++.h> // Includes all standard C++ libraries.
using namespace std;

int main() {
  // k: cost of the first banana.
  // n: initial amount of money the soldier has.
  // w: number of bananas he wants to buy.
  int k, n, w;
  cin >> k >> n >> w;

  // `total` will accumulate the cost of all bananas.
  int total = 0;

  /**
   * @brief Calculate the total cost iteratively.
   * This loop simulates the purchase of each banana from 1 to w.
   * For each banana `i`, its cost (`k * i`) is added to the total.
   */
  // Invariant: `total` holds the sum of costs for bananas from 1 to `i-1`.
  for (int i = 1; i <= w; i++) {
    total += (k * i);
  }

  /**
   * @brief Print the amount to borrow.
   * A ternary operator checks if the initial money `n` is sufficient.
   * If `n` is greater than `total`, the soldier has extra money, so he borrows 0.
   * Otherwise, the amount to borrow is the difference, `total - n`.
   */
  n > total ? cout << "0" : cout << (total - n);
  cout << endl;

  return 0; // Indicate successful execution.
}
