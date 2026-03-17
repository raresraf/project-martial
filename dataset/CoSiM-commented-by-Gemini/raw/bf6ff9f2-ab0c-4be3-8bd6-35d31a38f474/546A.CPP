/**
 * @file 546A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 546A "Soldier and Bananas".
 *
 * @details
 * This program calculates the amount of money a soldier needs to borrow to buy a
 * certain number of bananas. The cost of the bananas increases arithmetically:
 * the first costs `k`, the second `2k`, up to the `w`-th banana costing `w*k`.
 * The soldier initially has `n` dollars.
 *
 * @algorithm
 * The total cost of `w` bananas is the sum of an arithmetic series, which can be
 * calculated with a direct formula.
 * 1. The total cost is computed as `k * (1 + 2 + ... + w)`, which simplifies to
 *    `k * w * (w + 1) / 2`.
 * 2. The amount to borrow is calculated by subtracting the soldier's initial
 *    money (`n`) from this total cost.
 * 3. If the required amount to borrow is less than zero (i.e., the soldier
 *    has a surplus), the output is 0. Otherwise, the calculated amount is printed.
 *
 * @complexity
 * - Time Complexity: O(1), as the solution uses a mathematical formula with no
 *   loops dependent on the input size.
 * - Space Complexity: O(1), using only a fixed number of variables.
 */
#include <iostream>
using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
main() {
  int k, n, w;
  // k: cost of the first banana
  // n: initial money
  // w: number of bananas to buy
  cin >> k >> n >> w;
  
  int sum;
  /**
   * @brief Calculate the total cost of `w` bananas.
   * This uses the formula for the sum of the first `w` integers: w * (w + 1) / 2.
   * The total cost is this sum multiplied by the base cost `k`.
   */
  sum = (w * (w + 1) / 2) * k;
  
  // Calculate the amount the soldier needs to borrow.
  int borrow = sum - n;
  
  // Ensure the amount to borrow is not negative.
  if (borrow < 0)
    cout << "0" << endl;
  else
    cout << borrow << endl;
}
