/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem (Codeforces 546A).
 * @link http://ideone.com/BQzJDe
 *
 * @details
 * This program calculates the amount of money a soldier needs to borrow to buy a
 * specific number of bananas. The cost of the bananas increases linearly: the first
 * costs `k`, the second `2k`, ..., and the `w`-th costs `w*k`.
 *
 * The total cost is the sum of an arithmetic progression:
 * Total Cost = k + 2k + 3k + ... + w*k
 *            = k * (1 + 2 + 3 + ... + w)
 *
 * The sum of the first `w` integers is given by the formula `w * (w + 1) / 2`.
 * Therefore, Total Cost = `k * (w * (w + 1) / 2)`.
 *
 * The program calculates this total cost and subtracts the soldier's initial money, `n`.
 * If the result is positive, that's the amount to borrow. If it's zero or negative,
 * the soldier has enough money, so the amount to borrow is 0.
 *
 * Algorithm:
 * 1. Read the initial cost `k`, initial money `n`, and number of bananas `w`.
 * 2. Calculate the total cost using the arithmetic series formula.
 * 3. Determine the deficit and print it (or 0 if there is no deficit).
 *
 * Time Complexity: O(1), as the solution involves a direct calculation.
 * Space Complexity: O(1).
 */
// 546A bananas

#include <iostream>

typedef long long ll;

using namespace std;

int main() {

  // k: cost of the first banana
  // n: initial money the soldier has
  // w: number of bananas to buy
  ll n, w, k, res;

  cin >> k >> n >> w;

  /**
   * @brief Calculate the total cost of `w` bananas.
   * @details This uses the formula for the sum of an arithmetic series,
   * `k * (1 + 2 + ... + w)`, which simplifies to `k * w * (w + 1) / 2`.
   */
  res = ((w * (w + 1)) / 2) * k;

  /**
   * @brief Determine the amount of money to borrow.
   * @details If the total cost `res` is less than or equal to the money the
   * soldier has `n`, he needs to borrow 0. Otherwise, the amount to borrow
   * is the difference `res - n`.
   */
  if (res <= n)
    cout << "0";
  else
    cout << res - n;

  return 0;
}