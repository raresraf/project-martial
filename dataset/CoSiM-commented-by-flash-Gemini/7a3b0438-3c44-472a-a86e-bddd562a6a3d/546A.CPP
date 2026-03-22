/**
 * @file 546A.CPP
 * @brief Solution to Codeforces problem 546A - Soldier and Bananas.
 *        This program calculates the amount of money a soldier needs to borrow from a friend
 *        to buy a certain number of bananas, given the cost of the first banana and his initial money.
 *        The cost of each subsequent banana increases arithmetically.
 *
 * Algorithm:
 *   1. Reads the cost of the first banana (`k`), the soldier's initial money (`n`),
 *      and the number of bananas to buy (`w`).
 *   2. Iterates from 1 to `w` to calculate the total cost of `w` bananas.
 *      The cost of the `i`-th banana is `i * k`.
 *   3. If the total calculated cost is less than or equal to the soldier's initial money,
 *      it means he doesn't need to borrow any money, so prints 0.
 *   4. Otherwise, it prints the difference between the total cost and his initial money,
 *      which represents the amount he needs to borrow.
 *
 * Time Complexity: O(w), where `w` is the number of bananas, due to the loop.
 * Space Complexity: O(1) for storing a few integer variables.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * @brief Main function that solves the "Soldier and Bananas" problem.
   *
   * Calculates the total cost of `w` bananas, where the `i`-th banana costs `i * k`.
   * Determines if the soldier needs to borrow money and outputs the required amount.
   *
   * @return int Returns 0 upon successful execution.
   */
  int k, n, w, amount = 0;
  cin >> k >> n >> w;
  /**
   * Block Logic: Calculates the cumulative cost of buying `w` bananas.
   * Invariant: `amount` stores the sum of costs for bananas from 1 to `i-1`.
   * Pre-condition: `k`, `n`, `w` are positive integers.
   */
  for (int i = 1; i <= w; i++) {
    amount += k * i;
  }
  /**
   * Block Logic: Determines the amount of money to be borrowed.
   * Pre-condition: `amount` holds the total cost of bananas, `n` holds initial money.
   * Post-condition: Prints 0 if sufficient funds, otherwise prints the deficit.
   */
  if (amount <= n)
    cout << 0 << endl;
  else
    cout << amount - n << endl;
  return 0;
}
