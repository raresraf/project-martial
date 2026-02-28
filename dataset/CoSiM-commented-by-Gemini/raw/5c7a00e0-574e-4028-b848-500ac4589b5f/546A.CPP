/**
 * @file 546A.CPP
 * @brief Solution for the "Soldier and Bananas" problem using an arithmetic series formula.
 * @details This program calculates the amount of money a soldier needs to borrow to buy `w`
 * bananas, where the cost increases linearly (k, 2k, 3k, ...).
 *
 * @algorithm Arithmetic Series Summation
 * The total cost of the bananas forms an arithmetic series. The sum of an arithmetic
 * series can be calculated with the formula: `(first_term + last_term) * count / 2`.
 * In this problem:
 * - `first_term` = `k`
 * - `last_term` = `w * k`
 * - `count` = `w`
 * The total cost is `(k + w * k) * w / 2`. The program calculates this total,
 * subtracts the initial money `n`, and if the result is negative, outputs 0.
 *
 * @complexity
 * - Time: O(1), as the solution is a direct calculation using a mathematical formula.
 * - Space: O(1), for storing a few integer variables.
 */

#include <iostream>
using namespace std;

int main() {
  // k: cost of the first banana.
  // n: initial amount of money the soldier has.
  // w: number of bananas he wants to buy.
  // ans: the amount of money to borrow.
  long long int k, n, w, ans;
  cin >> k >> n >> w;

  /**
   * @brief Calculate the total cost and the amount to borrow.
   * This uses the formula for the sum of an arithmetic series to find the total
   * cost of `w` bananas, then subtracts the initial money `n`.
   */
  ans = (k + w * k) * w / 2 - n;

  // Pre-condition: Check if the soldier has enough money.
  // If `ans` is negative, it means the soldier has a surplus, so he needs to borrow 0.
  if (ans < 0) {
    ans = 0;
  }
  
  // Print the calculated amount to borrow.
  cout << ans << endl;
  
  return 0; // Indicate successful execution.
}
