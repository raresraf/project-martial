/**
 * @file 546A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Soldier and Bananas").
 * This program calculates the amount of money a soldier needs to borrow to buy a
 * specific number of bananas, where each subsequent banana costs more than the previous one.
 *
 * @algorithm Arithmetic Series Summation. The total cost of the bananas forms an
 * arithmetic series (k*1 + k*2 + ... + k*w). This can be calculated efficiently
 * using the formula for the sum of the first `w` integers: k * (w * (w + 1) / 2).
 *
 * @time_complexity O(1), as the solution relies on a direct mathematical formula.
 * @space_complexity O(1), as only a few variables are used.
 */
#include <bits/stdc++.h>
using namespace std;
main() {
  // k: cost of the first banana
  // n: initial money
  // w: number of bananas to buy
  int k, n, w;
  cin >> k >> n >> w;

  /**
   * @brief Calculates the total cost for 'w' bananas.
   * The cost of the i-th banana is i*k. The total cost is the sum of an
   * arithmetic progression: k * (1 + 2 + ... + w).
   */
  int total = ((w * (w + 1)) / 2) * k;

  /**
   * @brief Calculates and prints the amount to borrow.
   * If the total cost is greater than the money the soldier has (`n`),
   * the difference is printed. Otherwise, the soldier has enough money and
   * needs to borrow 0.
   */
  if ((total - n) > 0)
    cout << total - n;
  else
    cout << "0";
}
