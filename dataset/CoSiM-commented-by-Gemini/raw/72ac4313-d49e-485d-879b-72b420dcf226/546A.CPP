/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A - "Soldier and Bananas".
 *
 * This program calculates the amount of money a soldier needs to borrow to buy
 * a certain number of bananas. The cost of the i-th banana is i*k. The total
 * cost is calculated using the formula for the sum of an arithmetic series.
 *
 * @algorithm Direct calculation using the sum of an arithmetic series.
 * @time_complexity O(1).
 * @space_complexity O(1).
 */
// 28.01.19 @odnaks
#include <iostream>

using namespace std;

int main() {
  // k: cost of the first banana
  // n: initial money
  // w: number of bananas to buy
  // d: total cost / debt
  int k;
  int n;
  int w;
  int d;

  cin >> k >> n >> w;
  /**
   * @brief Calculate the total cost of the bananas.
   * This uses the formula for the sum of an arithmetic series:
   * Sum = (first_term + last_term) * number_of_terms / 2.
   * Here, first_term is k*1, last_term is k*w, and number_of_terms is w.
   * Total cost = (k + w*k) * w / 2.
   */
  d = (k + w * k) * w / 2;
  /**
   * @brief Determine the amount to borrow.
   * If the total cost 'd' is greater than the soldier's money 'n', the
   * difference is the amount to borrow. Otherwise, no money needs to be
   * borrowed.
   */
  if (d > n)
    d = d - n;
  else
    d = 0;
  cout << d;
  return (0);
}
