/**
 * @file 546A.CPP
 * @brief Solution for a programming problem, "Soldier and Bananas".
 * @details The program calculates the amount of money a soldier needs to borrow to buy a
 * certain number of bananas. The cost of each banana increases arithmetically.
 *
 * Problem Statement:
 * A soldier wants to buy `w` bananas.
 * The first banana costs `k`.
 * The second costs `2k`.
 * The i-th banana costs `i*k`.
 * The soldier has `n` dollars.
 * Calculate the amount of money the soldier must borrow. If he has enough money,
 * he borrows nothing.
 *
 * Algorithm:
 * 1. Calculate the total cost of `w` bananas. The total cost is the sum of an
 *    arithmetic progression: `k + 2k + ... + w*k = k * (1 + 2 + ... + w)`.
 * 2. The sum of the first `w` integers is `w * (w + 1) / 2`.
 * 3. Therefore, the total cost is efficiently calculated as `k * w * (w + 1) / 2`.
 * 4. The amount to borrow is the total cost minus the money the soldier has (`n`).
 * 5. If the result is negative (he has more than enough money), the amount to borrow is 0.
 *    The `max(total_cost - n, 0)` function correctly determines this amount.
 *
 * Time Complexity: O(1).
 * Space Complexity: O(1).
 */
#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <map>
#include <math.h>
#include <set>
#include <stdio.h>
#include <string>

using namespace std;

int main() {
  int k, n, w;
  // k: cost of the first banana
  // n: initial money the soldier has
  // w: number of bananas to buy
  cin >> k >> n >> w;

  // Calculate the total cost of 'w' bananas using the formula for the sum of an arithmetic series.
  int total = k * w * (w + 1) / 2;
  
  /**
   * @block
   * @description Calculate the money to borrow. This is the total cost less the
   * money on hand. If the soldier has enough money, the result is 0.
   */
  cout << max(total - n, 0);

  return 0;
}
