/**
 * @file 546A.CPP
 * @brief Calculates the money a soldier needs to borrow using an O(1) formula.
 *
 * This program solves the "Soldier and Bananas" problem from Codeforces.
 * It calculates the total cost for `w` bananas where the price increases
 * arithmetically (k, 2k, 3k, ...). It uses the direct mathematical formula
 * for the sum of the first `w` integers, multiplied by `k`, to find the
 * total cost in constant time.
 *
 * @algorithm Arithmetic Series Summation (O(1) Formula)
 * @complexity
 * - Time: O(1), as the calculation is a single arithmetic expression.
 * - Space: O(1).
 */
#include <iostream>
#include <stdio.h>
using namespace ::std;
main() {
  // Fast I/O optimization.
  ios_base::sync_with_stdio(false); 
  
  // k: cost of the first banana.
  // n: initial amount of money the soldier has.
  // w: number of bananas to buy.
  int k, n, w;
  cin >> k >> n >> w;

  /**
   * @brief Functional Utility: Calculates the total cost in O(1) time.
   * The total cost is k * (1 + 2 + ... + w). The sum of the first w integers
   * is given by the formula w * (w + 1) / 2.
   */
  long int t = k * (w * (w + 1)) / 2;

  /**
   * @brief Block Logic: Determines the amount to borrow.
   */
  if (n >= t)
    // If the soldier has enough money, no borrowing is needed.
    cout << 0;
  else
    // The amount to borrow is the total cost minus the available money.
    cout << t - n;
    
  return 0;
}
