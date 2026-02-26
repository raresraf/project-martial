/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem.
 * @details The problem requires calculating the amount of money a soldier needs
 * to borrow to buy a certain number of bananas. The cost of each banana
 * increases linearly: the first costs `k`, the second `2k`, and so on.
 *
 * @algorithm
 * 1. Read the initial cost `k`, the soldier's initial money `n`, and the
 *    number of bananas to buy `w`.
 * 2. Calculate the total cost for `w` bananas. This is the sum of an
 *    arithmetic series: k * (1 + 2 + ... + w).
 * 3. Compare the total cost with the money the soldier has (`n`).
 * 4. If the cost is greater than `n`, the difference is the amount to borrow.
 * 5. Otherwise, the soldier has enough money, and the amount to borrow is 0.
 *
 * @complexity
 * Time Complexity: O(W) due to the loop that calculates the sum. This could
 * also be implemented in O(1) using the formula for the sum of an arithmetic
 * series: total_cost = k * w * (w + 1) / 2.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // k: the cost of the first banana.
  // n: the initial amount of money the soldier has.
  // w: the number of bananas the soldier wants to buy.
  long long k, n, w;
  cin >> k >> n >> w;

  // Use long long for sum to prevent potential integer overflow if w and k are large.
  long long total_cost = 0;

  /**
   * @block
   * @brief Calculate the total cost of `w` bananas.
   * @details The cost of the i-th banana is `i * k`. This loop sums the costs
   * from the 1st to the w-th banana.
   */
  for (int i = 1; i <= w; i++)
    total_cost += i * k;

  /**
   * @block
   * @brief Determine the amount of money to borrow.
   * @pre `total_cost` holds the total price of all bananas.
   *      `n` is the money the soldier has.
   * @post Prints the required amount to borrow, or 0 if none is needed.
   */
  if (total_cost > n)
    cout << total_cost - n << endl;
  else
    cout << "0" << endl;
}