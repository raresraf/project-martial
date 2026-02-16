/**
 * @file 546A.CPP
 * @brief Solution for Codeforces problem 546A, "Soldier and Bananas".
 *
 * @description
 * A soldier buys `w` bananas. The cost of the bananas increases linearly: the
 * first costs `k`, the second `2k`, and so on, up to the `w`-th banana costing
 * `w*k`. Given the soldier's initial money `n`, this program calculates the
 * amount of money the soldier must borrow. If the soldier has enough money,
 * the result is 0.
 *
 * @algorithm
 * 1. Read the initial cost `k`, the soldier's money `n`, and the number of
 *    bananas `w`.
 * 2. Calculate the total cost of `w` bananas by summing the arithmetic series
 *    k + 2k + ... + w*k. This is done iteratively with a loop.
 * 3. Compare the total cost with the money `n`.
 * 4. If the cost is greater than `n`, print the difference (cost - n).
 * 5. Otherwise, print 0.
 *
 * @note The total cost can also be calculated directly using the formula for an
 * arithmetic series: `total_cost = k * (w * (w + 1) / 2)`.
 *
 * @time_complexity O(W), due to the loop iterating `w` times.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // k: cost of the first banana
  // n: initial money the soldier has
  // w: number of bananas to buy
  // c: total cost of all bananas
  int k, n, w, c = 0;
  cin >> k >> n >> w;

  // Block Logic: Calculate the total cost of 'w' bananas.
  // The cost of the i-th banana (1-indexed) is i * k.
  for (int i = 0; i < w; i++)
    c += (i + 1) * k;

  // Block Logic: Determine the amount of money to borrow.
  // If the total cost 'c' is less than or equal to the available money 'n',
  // no borrowing is needed. Otherwise, calculate the deficit.
  if (c <= n)
    cout << 0;
  else
    cout << c - n;
  return 0;
}
