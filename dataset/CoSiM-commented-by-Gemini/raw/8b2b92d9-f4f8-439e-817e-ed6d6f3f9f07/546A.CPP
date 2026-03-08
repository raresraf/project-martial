/**
 * @file 546A.CPP
 * @brief Solution for the "Soldier and Bananas" problem.
 *
 * @details
 * This program calculates the amount of money a soldier needs to borrow to buy
 * a certain number of bananas. The cost of the bananas increases arithmetically:
 * the first costs k, the second 2k, and so on, up to the w-th banana costing w*k.
 * The program calculates the total cost and subtracts the soldier's initial money
 * to find the deficit. If the soldier has enough money, the deficit is 0.
 *
 * @section
 * Algorithm: Iterative summation of an arithmetic series.
 * Time Complexity: O(W), where W is the number of bananas to buy.
 * Space Complexity: O(1).
 *
 * @note A more optimal O(1) solution exists by using the formula for the sum
 * of an arithmetic series: `totalCost = k * w * (w + 1) / 2`.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // k: the cost of the first banana.
  // n: the initial amount of money the soldier has.
  // w: the total number of bananas the soldier wants to buy.
  // totalCost: accumulator for the total cost of all bananas.
  int k, n, w, totalCost = 0;
  cin >> k >> n >> w;

  /**
   * @brief This loop calculates the total cost for 'w' bananas.
   *
   * Logic: The cost of the i-th banana is `i * k`. This loop sums the costs
   * for all bananas from 1 to w.
   */
  for (int i = 1; i <= w; i++)
    totalCost += (i * k);

  /**
   * @brief Determines the amount of money to borrow.
   *
   * Logic: If the total cost is greater than the money the soldier has ('n'),
   * the difference is the amount that needs to be borrowed. Otherwise, the
   * soldier has enough money and needs to borrow 0.
   */
  if (totalCost > n)
    cout << totalCost - n;
  else
    cout << 0;
}
