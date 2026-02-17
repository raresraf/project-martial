/**
 * @file 546A.CPP
 * @brief Solution to a problem about calculating the cost of bananas.
 * @details A person wants to buy 'w' bananas. The first banana costs 'k', the second costs '2k', and so on.
 * The person has 'n' dollars. This program calculates how much money they need to borrow to buy all the bananas.
 *
 * @algorithm The total cost is calculated by summing an arithmetic progression iteratively.
 * The cost for 'w' bananas is k * (1 + 2 + ... + w).
 * @time_complexity O(w), where 'w' is the number of bananas. The loop runs 'w' times.
 * @space_complexity O(1), as the memory usage is constant regardless of the input size.
 */
#include <iostream>

int main() {
  int k, n, w, cost = 0; // k: cost of first banana, n: initial money, w: number of bananas
  std::cin >> k >> n >> w;

  /**
   * Block Logic: Calculates the total cost for 'w' bananas.
   * The cost of the i-th banana is i * k. This loop sums the cost for all bananas from 1 to w.
   */
  for (int i = 1; i < w + 1; i++) {
    cost += (i * k);
  }

  /**
   * Block Logic: Determines the amount of money to borrow.
   * If the initial money 'n' is greater than or equal to the total 'cost', no money needs to be borrowed.
   * Otherwise, the difference is the amount to borrow.
   */
  if (n - cost > 0) {
    std::cout << 0;
  } else {
    // The amount to borrow is the absolute difference between the money on hand and the total cost.
    std::cout << abs(n - cost);
  }

  return 0;
}