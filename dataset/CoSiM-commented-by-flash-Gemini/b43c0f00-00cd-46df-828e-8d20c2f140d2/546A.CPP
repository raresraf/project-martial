/**
 * @file 546A.CPP
 * @brief This program solves the "Soldier and Bananas" problem (Codeforces 546A).
 *   A soldier wants to buy `w` bananas. The first costs `k` dollars, the second `2k`, and so on.
 *   He has `n` dollars. The task is to calculate how much money he needs to borrow.
 *   If he has enough money, the amount to borrow is 0.
 *   Problem description: https://codeforces.com/problemset/problem/546/A
 * * Algorithm: Arithmetic series sum and conditional calculation.
 *   The total cost is calculated using the formula for the sum of an arithmetic series:
 *   `k + 2k + ... + wk = k * (1 + 2 + ... + w) = k * w * (w + 1) / 2`.
 *   The amount to borrow is then `total_cost - money_has`, clamped at a minimum of 0.
 * Time Complexity: O(1) - constant time, as it involves a few arithmetic operations.
 * Space Complexity: O(1) - constant space for a few integer variables.
 */
#include <iostream> // Required for standard input/output operations.

using namespace std;

/**
 * @brief Main function to calculate the amount of money a soldier needs to borrow.
 * Functional Utility: Reads the cost parameters, calculates the total cost of bananas,
 *                     determines the amount to borrow, and prints the result.
 */
int main() {
  int k, n, w; // `k`: cost of the first banana, `n`: money soldier has, `w`: number of bananas to buy.
  int t;       // `t`: total calculated cost of all bananas.
  int b;       // `b`: amount to borrow.

  // Block Logic: Reads the input values for k, n, and w.
  cin >> k >> n >> w;

  // Functional Utility: Calculates the total cost `t` for `w` bananas using the sum of an arithmetic series.
  // The cost increases linearly: k, 2k, 3k, ..., wk. The sum is k * (1 + 2 + ... + w).
  // The sum of 1 to w is w * (w + 1) / 2.
  t = (k * w * (w + 1)) / 2;

  // Functional Utility: Calculates the amount `b` that needs to be borrowed.
  // This is the total cost minus the money the soldier already has.
  b = t - n;

  /**
   * Conditional Logic: Determines whether the soldier needs to borrow money.
   * If `b` is greater than 0, it means he needs to borrow that amount.
   * Otherwise (if `b` is 0 or negative), he has enough money and needs to borrow 0.
   */
  if (b > 0)
    cout << b; // Prints the positive amount to borrow.
  else
    cout << "0"; // Prints 0 if no money needs to be borrowed.
  return 0;
}
