/**
 * @file 546A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 546A "Soldier and Bananas".
 *
 * @details
 * This program calculates the amount of money a soldier needs to borrow to buy a
 * certain number of bananas. The cost of the bananas increases arithmetically:
 * the first costs `k`, the second `2k`, and so on, up to the `w`-th banana costing `w*k`.
 * The soldier initially has `n` dollars.
 *
 * @algorithm
 * The total cost of `w` bananas forms an arithmetic series: `k + 2k + ... + wk`,
 * which can be factored as `k * (1 + 2 + ... + w)`.
 * The sum of the first `w` positive integers is calculated using the formula `w * (w + 1) / 2`.
 * 1. The total cost is computed directly using the formula `k * w * (w + 1) / 2`.
 * 2. The amount to borrow is the total cost minus the money the soldier has (`n`).
 * 3. If the result is negative (meaning the soldier has enough money), the amount
 *    to borrow is adjusted to 0.
 *
 * @complexity
 * - Time Complexity: O(1), as the solution uses a direct mathematical formula without any loops
 *   dependent on the input size.
 * - Space Complexity: O(1), as it only uses a fixed number of variables.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  // Several variables are declared, but only k, n, w, and b are used.
  int n, b, i, j, l, o = 0, k, w;
  
  // k: The cost of the first banana.
  cin >> k;
  // n: The initial amount of money the soldier has.
  cin >> n;
  // w: The total number of bananas the soldier wants to buy.
  cin >> w;
  
  /**
   * @brief Calculate the amount of money to borrow.
   * Total cost = k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
   * `b` stores the difference between the total cost and the money on hand.
   */
  b = (k * (w + 1) * w / 2) - n;
  
  // If `b` is negative, the soldier has enough money, so no borrowing is needed.
  if (b < 0) {
    b = 0;
  }
  
  // Print the amount to be borrowed.
  cout << b;
  return 0;
}
