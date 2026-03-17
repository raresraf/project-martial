/**
 * @file 546A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 546A "Soldier and Bananas".
 *
 * @details
 * This program calculates the amount of money a soldier needs to borrow. The soldier
 * wants to buy `w` bananas. The cost of the first banana is `k`, the second is `2k`,
 * and so on. The soldier has `n` dollars. The program computes the difference between
 * the total cost and the money the soldier has.
 *
 * @algorithm
 * The total cost is the sum of an arithmetic series `k * (1 + 2 + ... + w)`.
 * This is calculated efficiently using the formula for the sum of the first `w`
 * integers: `Total Cost = k * w * (w + 1) / 2`.
 * The amount to borrow is then `Total Cost - n`. If this value is negative, it
 * means the soldier has enough money, so the amount to borrow is 0.
 *
 * @complexity
 * - Time Complexity: O(1), as the result is computed with a direct mathematical formula.
 * - Space Complexity: O(1), as only a few variables are used.
 */
#include <iostream>
using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  int k, n, w, a;
  // k: cost of the first banana
  // n: initial amount of money
  // w: number of bananas to buy
  cin >> k >> n >> w;

  /**
   * @brief Calculate the money to borrow in a single step.
   * `a` will hold the deficit.
   * The formula `(w * (w + 1)) / 2` calculates the sum of 1 to w.
   * This is multiplied by `k` for the total cost, and then `n` is subtracted.
   */
  a = ((((w * (w + 1)) / 2) * k) - n);

  // If the calculated deficit `a` is positive, that's the amount to borrow.
  // Otherwise, the soldier has enough money, so the borrow amount is 0.
  if (a > 0)
    cout << a << endl;
  else
    cout << 0 << endl;

  return 0;
}
