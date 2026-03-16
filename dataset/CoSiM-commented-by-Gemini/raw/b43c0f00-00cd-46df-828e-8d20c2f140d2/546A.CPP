/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem.
 *
 * This program calculates the amount of money a soldier needs to borrow to buy a
 * certain number of bananas. The cost of the bananas increases linearly.
 * The first banana costs k, the second 2k, and so on, up to the w-th banana costing w*k.
 * The soldier has n dollars. The program calculates the total cost and finds the
 * difference if the cost exceeds the soldier's money.
 *
 * @algorithm Arithmetic Series Summation
 * @details The total cost of `w` bananas forms an arithmetic series:
 * k + 2k + 3k + ... + wk = k * (1 + 2 + 3 + ... + w).
 * The sum of the first `w` positive integers is `w * (w + 1) / 2`.
 * Therefore, the total cost is `k * w * (w + 1) / 2`.
 * The program calculates this total cost and subtracts the initial money `n`
 * to find the amount to be borrowed. If the total cost is less than or equal to `n`,
 * no money needs to be borrowed.
 *
 * @complexity
 * Time Complexity: O(1), as the calculation is a direct formula.
 * Space Complexity: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  int k, n, w, t, b;
  cin >> k >> n >> w;
  
  /**
   * @brief Calculate the total cost of w bananas.
   * `t` represents the total cost, calculated using the sum of an arithmetic series.
   */
  t = (k * w * (w + 1)) / 2;
  
  /**
   * @brief Calculate the amount to be borrowed.
   * `b` is the amount to borrow, which is the total cost minus the initial money.
   */
  b = t - n;

  /**
   * @brief Output the result.
   * If `b` is positive, that's the amount to borrow.
   * Otherwise, the soldier has enough money, so 0 is printed.
   */
  if (b > 0)
    cout << b;
  else
    cout << "0";
  
  return 0;
}
