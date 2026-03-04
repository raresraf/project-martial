/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A - "Soldier and Bananas".
 *
 * This program calculates the amount of money a soldier needs to borrow to buy
 * a specific number of bananas. The cost of each banana increases linearly.
 * The total cost is calculated using the formula for the sum of an arithmetic
 * series.
 *
 * @algorithm Direct calculation using the sum of an arithmetic series.
 * @time_complexity O(1), as it involves a constant number of arithmetic operations.
 * @space_complexity O(1).
 */
#include <iostream>

/*
 * Contest: Code Forces Round #304 (Task 546-A)
 * URL: http://codeforces.ru/contest/546/problem/A
 */

int main(int argc, char **argv) {
  // k: cost of the first banana
  // n: initial money the soldier has
  // w: number of bananas to buy
  int k, n, w, r;
  std::cin >> k >> n >> w;

  /**
   * @brief Calculate the amount to borrow.
   * The total cost of 'w' bananas is k * (1 + 2 + ... + w).
   * The sum of the first 'w' integers is w * (w + 1) / 2.
   * So, total cost = k * (w * (w + 1) / 2).
   * 'r' is the difference between the total cost and the soldier's money.
   */
  r = (k * (w * (w + 1) / 2)) - n;

  /**
   * @brief Print the result.
   * If 'r' is positive, it's the amount to borrow. If it's zero or negative,
   * the soldier has enough money, so 0 is printed.
   */
  std::cout << ((r >= 0) ? r : 0) << std::endl;
  return 0;
}