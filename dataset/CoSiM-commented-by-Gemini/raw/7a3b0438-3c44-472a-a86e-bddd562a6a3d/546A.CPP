/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A, "Soldier and Bananas".
 * @details This program calculates the amount of money a soldier needs to
 * borrow to buy a certain number of bananas. The cost of each subsequent
 * banana increases linearly.
 *
 * Algorithm:
 * 1. Read the cost of the first banana (`k`), the soldier's initial money (`n`),
 *    and the total number of bananas to buy (`w`).
 * 2. Calculate the total cost required to buy `w` bananas. The cost of the
 *    i-th banana is `i * k`. The code calculates the total by summing this
 *    cost in a loop from `i = 1` to `w`.
 *    - The total cost is an arithmetic series: `k * (1 + 2 + ... + w)`, which
 *      can be calculated directly with the formula `k * w * (w + 1) / 2`.
 * 3. Compare the total cost (`amount`) with the money the soldier has (`n`).
 * 4. If the total cost is greater than `n`, the amount to borrow is the
 *    difference (`amount - n`).
 * 5. If the soldier has enough money (`amount <= n`), he needs to borrow 0.
 *
 * Time Complexity: O(W), where W is the number of bananas.
 *                  Could be O(1) if the mathematical formula was used.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int k, n, w, amount = 0;
  cin >> k >> n >> w;
  /**
   * @brief This loop calculates the total cost of `w` bananas. The cost for
   * each banana `i` is `k * i`.
   */
  for (int i = 1; i <= w; i++) {
    amount += k * i;
  }
  /**
   * @brief This block determines the amount of money to borrow.
   * If the total cost (`amount`) is more than the money the soldier has (`n`),
   * it prints the difference. Otherwise, it prints 0.
   */
  if (amount <= n)
    cout << 0 << endl;
  else
    cout << amount - n << endl;
  return 0;
}
