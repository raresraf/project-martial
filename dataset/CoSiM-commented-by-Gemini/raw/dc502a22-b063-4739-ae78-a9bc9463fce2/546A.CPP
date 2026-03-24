/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A - "Soldier and Bananas".
 *
 * @details
 * This program calculates the amount of money a soldier needs to borrow to buy
 * a certain number of bananas. The cost of the bananas increases linearly: the
 * first costs `k`, the second `2k`, and so on.
 *
 * Algorithm:
 * The total cost forms an arithmetic series: k*1 + k*2 + ... + k*w.
 * The implementation calculates this sum procedurally using a loop.
 * 1. Read the cost of the first banana (k), the soldier's initial money (n),
 *    and the number of bananas to buy (w).
 * 2. Iterate from 1 to w, calculating the cost of each banana (i*k) and adding
 *    it to a running total `sum`.
 * 3. After the loop, compare the total cost `sum` with the soldier's money `n`.
 * 4. If the soldier has enough money (n >= sum), the amount to borrow is 0.
 *    Otherwise, the amount to borrow is `sum - n`.
 *
 * A more direct O(1) solution exists using the formula for an arithmetic series:
 * Total Cost = k * (w * (w + 1) / 2).
 *
 * Time Complexity: O(W), where W is the number of bananas, due to the loop.
 * Space Complexity: O(1).
 */
#include <iostream>

int main() {
  // k: cost of the first banana
  // n: initial money the soldier has
  // w: number of bananas to buy
  int k, n, w;

  // `sum` will accumulate the total cost of all bananas.
  int sum = 0;

  std::cin >> k >> n >> w;

  /**
   * Block Logic: Calculates the total cost of `w` bananas.
   * This loop simulates the purchase of each banana, adding its increasing
   * cost to the total sum, effectively summing an arithmetic series.
   */
  for (int i = 0; i < w; i++) {
    sum += ((i + 1) * k);
  }

  // Note from original author on the O(1) formula.
  // or using arithmetic progression sum formula => sum = (k + k*w )*w/2

  // Functional Utility: Determines the amount to borrow.
  // If the soldier's money `n` is not enough to cover the `sum`, print the
  // difference. Otherwise, print 0.
  std::cout << (n >= sum ? 0 : sum - n) << std::endl;

  return 0;
}
