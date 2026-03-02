#include <bits/stdc++.h>
using namespace std;
/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem from Codeforces (Problem 546A).
 * @details A soldier buys `w` bananas. The first banana costs `k`, the second `2k`, and so on.
 * Given the soldier's initial money `n`, this program calculates how much money must be borrowed.
 *
 * Algorithm:
 * The total cost of the bananas is the sum of an arithmetic series. The cost of the i-th banana is
 * `i * k`. The total cost for `w` bananas is `k * (1 + 2 + ... + w)`.
 * The program calculates this total cost iteratively. It then determines the amount of money
 * the soldier needs to borrow by subtracting the initial money `n` from the total cost.
 * If the soldier has enough money (cost <= n), the amount to borrow is 0. Otherwise, it's `cost - n`.
 * The sum can also be calculated directly with the formula: `Total Cost = k * w * (w + 1) / 2`.
 *
 * Time Complexity: O(w) due to the loop that calculates the total cost by summing up the
 * price of each banana individually.
 * Space Complexity: O(1) as a fixed amount of memory is used.
 */
int main() {
  int k, n, w, cost = 0;
  cin >> k >> n >> w;
  int num = w, i = 1;

  /**
   * @brief Calculate the total cost of `w` bananas.
   * The loop iterates from the first banana to the w-th banana, summing their costs.
   * `i` tracks the banana number (1st, 2nd, ...), and `num` is the countdown of bananas to buy.
   */
  while (num > 0) {
    cost += i * k;
    i++;
    num--;
  }

  // Calculate the amount to borrow.
  int borrow = cost - n;
  
  /**
   * @brief Determine the output.
   * If the cost is less than or equal to the money the soldier has, they don't need to borrow.
   * Otherwise, print the calculated amount to borrow.
   */
  if (cost <= n)
    cout << 0;
  else
    cout << borrow;
}
