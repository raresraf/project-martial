/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem.
 * @details This program calculates the amount of money a soldier needs to borrow to buy
 * a specific number of bananas. The cost of each banana increases in an arithmetic
 * progression: the first costs k, the second 2k, and so on.
 *
 * @algorithm The core of the solution is the calculation of the total cost. The total cost
 * is the sum of an arithmetic series: k * (1 + 2 + ... + w). This is efficiently calculated
 * using the formula for the sum of the first 'w' positive integers: w * (w + 1) / 2.
 * The total cost is therefore `k * w * (w + 1) / 2`. The amount to borrow is the
 * total cost minus the initial money, or 0 if the soldier has enough.
 *
 * @time_complexity O(1), as the total cost is computed using a direct mathematical formula,
 * irrespective of the number of bananas.
 * @space_complexity O(1), using only a few variables to store input and results.
 */
#include <cstdio>

int main() {
  // k: The cost of the first banana.
  // n: The initial amount of money the soldier has.
  // w: The total number of bananas the soldier wants to buy.
  int k, n, w;
  scanf("%d %d %d", &k, &n, &w);

  /**
   * Block Logic: Calculate the total cost of 'w' bananas using the formula for an arithmetic series sum.
   * The cost progression is k, 2k, ..., wk, which sums to k * (1 + 2 + ... + w).
   */
  int money = w * (w + 1) / 2 * k;

  /**
   * Block Logic: Determine the amount to borrow.
   * If the total cost is greater than the soldier's money, the deficit is printed.
   * Otherwise, no money needs to be borrowed, so 0 is printed.
   */
  printf("%d\n", money <= n ? 0 : money - n);
  return 0;
}
