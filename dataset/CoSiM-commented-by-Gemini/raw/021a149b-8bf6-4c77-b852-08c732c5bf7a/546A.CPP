/**
 * @file 546A.CPP
 * @brief Solution for competitive programming problem 546A "Soldier and Bananas" on Codeforces.
 *
 * This program calculates the amount of money a soldier needs to borrow to buy
 * a certain number of bananas, given the cost of the first banana, his initial
 * money, and the number of bananas he wants to buy. The cost of each subsequent
 * banana increases linearly.
 *
 * Algorithm: Iterative summation to calculate total cost, followed by a conditional
 * calculation of the amount to borrow.
 * Time Complexity: O(W) where W is the number of bananas.
 * Space Complexity: O(1)
 */
#include <iostream>
using namespace std;

int main() {
  int k, n, w;
  /**
   * Block Logic: Reads the input values.
   * `k`: the cost of the first banana.
   * `n`: the initial money the soldier has.
   * `w`: the number of bananas the soldier wants to buy.
   * Pre-condition: User provides three integer inputs.
   * Invariant: k, n, and w are positive integers.
   */
  cin >> k >> n >> w;

  int totalCost = 0;
  /**
   * Block Logic: Calculates the total cost for `w` bananas.
   * The cost of the i-th banana is i * k.
   * Invariant: `totalCost` accumulates the sum of costs for all bananas.
   */
  for (int i = 1; i <= w; i++) {
    totalCost += i * k;
  }

  /**
   * Block Logic: Determines if the soldier needs to borrow money and calculates the amount.
   * Pre-condition: `totalCost` and `n` (initial money) are known.
   * Invariant: If `totalCost` is less than or equal to `n`, no money needs to be borrowed (output is 0).
   * Otherwise, the difference `totalCost - n` is the amount to borrow.
   */
  if (totalCost <= n) {
    cout << "0" << endl;
  } else {
    int amountToBorrow = totalCost - n;
    cout << amountToBorrow << endl;
  }

  return 0;
}
