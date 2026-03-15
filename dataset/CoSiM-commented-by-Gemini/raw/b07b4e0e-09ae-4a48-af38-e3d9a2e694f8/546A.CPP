/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem.
 * @details This program calculates the amount of money a soldier needs to borrow to buy a certain number of bananas.
 * The cost of the first banana is `k`, the second is `2k`, and so on. The soldier has `n` dollars and wants to buy `w` bananas.
 *
 * @algorithm
 * 1. Reads the initial cost `k`, initial money `n`, and number of bananas `w`.
 * 2. Calculates the total cost by summing an arithmetic series. The cost of the i-th banana is `i*k`. The loop iterates from 1 to `w` to compute this sum.
 *    The total cost can also be calculated using the closed-form formula for an arithmetic series sum: `total = k * w * (w + 1) / 2`.
 * 3. Determines the amount to borrow by subtracting the initial money `n` from the `total` cost.
 * 4. If the total cost is less than or equal to the initial money, the soldier borrows nothing (output is 0). Otherwise, it outputs the difference.
 *
 * @complexity
 * - Time: O(W), where W is the number of bananas, due to the loop for summation.
 * - Space: O(1), as memory usage is constant.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // k: The cost of the first banana.
  // n: The initial amount of money the soldier has.
  // w: The number of bananas the soldier wants to buy.
  // total: The cumulative cost of all bananas.
  int k, n, w, total = 0;
  cin >> k >> n >> w;

  /**
   * @brief Block to calculate the total cost of `w` bananas.
   * The cost of each banana increases linearly. The cost of the i-th banana is `i * k`.
   * This loop sums the costs for all `w` bananas.
   * Invariant: At the end of iteration `i`, `total` holds the sum of costs for the first `i` bananas.
   */
  for (int i = 1; i <= w; i++)
    total += i * k;

  /**
   * @brief Block to determine the amount to borrow.
   * If the total cost is greater than the money the soldier has, the difference is the amount to borrow.
   * Otherwise, no money needs to be borrowed.
   */
  if (total - n > 0)
    cout << total - n;
  else
    cout << 0;

  return 0;
}