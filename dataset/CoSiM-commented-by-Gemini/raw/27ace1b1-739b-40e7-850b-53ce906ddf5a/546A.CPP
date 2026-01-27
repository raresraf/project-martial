/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem.
 * @details This program calculates the amount of money a soldier needs to borrow to buy a certain number of bananas.
 * The cost of the first banana is k, the second is 2k, and so on. The soldier has n dollars and wants to buy w bananas.
 * The program calculates the total cost and then determines how much money, if any, the soldier needs to borrow.
 *
 * @see https://codeforces.com/problemset/problem/546/A
 *
 * Algorithm:
 * 1. Read the cost of the first banana (k), the initial money the soldier has (n), and the number of bananas to buy (w).
 * 2. Calculate the total cost of w bananas by summing up the cost of each banana from 1 to w.
 * 3. If the soldier has enough money (n >= total cost), the amount to borrow is 0.
 * 4. Otherwise, the amount to borrow is the difference between the total cost and the money the soldier has.
 *
 * Time Complexity: O(w) because of the loop to calculate the total cost.
 * Space Complexity: O(1) as it uses a fixed amount of memory regardless of the input size.
 */

#include <iostream>

using namespace std;

int main() {
  // k: cost of the first banana
  // n: initial amount of money
  // w: number of bananas to buy
  int k, n, w;
  int total = 0, borrow = 0;

  // Read the input values
  cin >> k >> n >> w;

  /**
   * Block Logic: Calculate the total cost of w bananas.
   * The cost of the i-th banana is i*k. This loop sums the costs for all bananas from 1 to w.
   */
  for (int i = 1; i <= w; ++i) {
    total += k * i;
  }

  /**
   * Block Logic: Determine the amount to borrow.
   * If the soldier's money 'n' is greater than or equal to the total cost,
   * no money needs to be borrowed. Otherwise, calculate the difference.
   */
  if (n >= total) {
    // Output 0 if the soldier has enough money.
    cout << 0 << endl;
  } else {
    // Calculate and output the amount to borrow.
    borrow = total - n;
    cout << borrow << endl;
  }

  return 0;
}
