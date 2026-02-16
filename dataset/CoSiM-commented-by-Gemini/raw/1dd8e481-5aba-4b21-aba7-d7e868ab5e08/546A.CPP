/**
 * @file 546A.CPP
 * @brief Solution for Codeforces problem 546A, "Soldier and Bananas".
 *
 * @description
 * This program calculates the amount of money a soldier must borrow to buy `w`
 * bananas. The cost of the i-th banana is `i*k`. The soldier starts with `n`
 * dollars. This solution uses a direct mathematical formula instead of a loop.
 *
 * @algorithm
 * The total cost of the bananas forms an arithmetic series:
 *   k * (1 + 2 + 3 + ... + w)
 * The sum of the first `w` integers is `w * (w + 1) / 2`.
 * Therefore, the total cost can be calculated in O(1) time.
 *
 * 1. Read the initial cost `k`, the soldier's money `n`, and the number of
 *    bananas `w`.
 * 2. Calculate the total cost using the formula: `(k * w * (w + 1)) / 2`.
 * 3. Calculate the deficit: `total_cost - n`.
 * 4. If the deficit is positive, that is the amount to borrow.
 * 5. If the deficit is zero or negative (meaning the soldier has enough money),
 *    the amount to borrow is 0.
 *
 * @time_complexity O(1), due to the direct mathematical calculation.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // k: base cost of a banana.
  // n: soldier's initial money.
  // w: number of bananas to buy.
  int k, n, w;
  cin >> k >> n >> w;

  // Block Logic: Calculate the borrowed amount using an arithmetic series formula.
  // `((w * (w + 1)) / 2)` is the sum of integers from 1 to w.
  // `* k` gives the total cost. `- n` gives the deficit.
  if ((((w * (w + 1)) / 2) * k) - n >= 0)
    cout << (((w * (w + 1)) / 2) * k) - n;
  else
    cout << "0";

  return 0;
}
