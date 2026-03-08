/**
 * @file 546A.CPP
 * @brief Solution for the "Soldier and Bananas" problem (e.g., Codeforces 546A).
 * @details This program calculates the amount of money a soldier needs to borrow to buy a
 * specific number of bananas. The cost of each subsequent banana increases linearly.
 *
 * @algorithm Iterative Summation of an Arithmetic Series.
 * @complexity
 * - Time: O(w), where 'w' is the number of bananas to buy, due to the loop.
 * - Space: O(1).
 *
 * @note A more efficient O(1) solution exists by using the formula for the sum of an
 * arithmetic series: `Total Cost = k * w * (w + 1) / 2`.
 */
#include <iostream>
using namespace std;
typedef long long ll; // Use long long to prevent potential integer overflow for large costs.

int main() {
  // k: the cost of the first banana.
  // n: the initial amount of money the soldier has.
  // w: the total number of bananas the soldier wants to buy.
  ll k, n, w;
  cin >> k >> n >> w;

  // 'dollar' will accumulate the total cost of all bananas.
  ll dollar = 0;

  /**
   * @block Total Cost Calculation
   * @logic The loop sums up the cost of each banana from the 1st to the w-th.
   * The cost of the i-th banana is `k * i`.
   */
  for (int i = 1; i <= w; i++) {
    dollar += k * i;
  }

  /**
   * @block Borrow Amount Calculation
   * @logic Calculates the difference between the total cost and the money the soldier has.
   * If the soldier has enough money, the amount to borrow is 0.
   */
  if ((dollar - n) < 0) {
    cout << "0";
  } else {
    cout << dollar - n;
  }

  return 0;
}
