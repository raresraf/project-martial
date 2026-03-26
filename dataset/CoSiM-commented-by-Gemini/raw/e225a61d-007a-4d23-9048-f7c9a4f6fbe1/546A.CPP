/**
 * @file 546A.CPP
 * @brief Solution to Codeforces problem 546A - Soldier and Bananas.
 * @description A soldier wants to buy `w` bananas. The cost increases linearly:
 *              the first banana costs `k`, the second `2k`, and so on, up to the
 *              `w`-th banana costing `w*k`. The soldier has `n` dollars. This program
 *              calculates the amount of money the soldier must borrow. If the total
 *              cost is less than or equal to `n`, the soldier borrows nothing (0).
 *
 * Algorithm: The total cost is the sum of an arithmetic progression:
 *            k * (1 + 2 + ... + w). This can be calculated directly with a loop or
 *            using the formula for the sum of the first `w` integers: k * w * (w + 1) / 2.
 *            The program uses a loop to compute the total cost, then subtracts the
 *            soldier's money `n` to find the deficit.
 *
 * Time Complexity: O(W), where W is the number of bananas to buy, due to the loop.
 * Space Complexity: O(1).
 */
//
// Created by Conor on 5/28/2020.
//

#include <iostream>

using namespace std;

int main() {
  // k: the cost of the first banana.
  // n: the initial amount of money the soldier has.
  // w: the number of bananas the soldier wants to buy.
  int k, n, w;
  cin >> k >> n >> w;

  // cost: variable to accumulate the total cost of all bananas.
  // Note: For the maximum constraints (k=1000, w=1000), the cost can reach ~5*10^8,
  // which fits within a 32-bit signed integer. Using long long would be safer.
  int cost = 0;

  /**
   * Block Logic: Calculate the total cost of `w` bananas by summing the cost of each one.
   * The cost of the i-th banana is i*k.
   */
  for (int i = 1; i <= w; ++i) {
    cost += i * k;
  }

  // Post-condition: Determine how much money to borrow.
  if (cost <= n) {
    // If the cost is less than or equal to the money the soldier has, no borrowing is needed.
    cout << 0;
  } else {
    // Otherwise, the amount to borrow is the difference between the cost and the available money.
    cout << cost - n;
  }

  return 0;
}
