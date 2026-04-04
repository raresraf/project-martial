/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem from Codeforces.
 *
 * This program calculates the amount of money a soldier needs to borrow to buy 'w' bananas.
 * The cost of the first banana is 'k', the second is '2k', and so on, up to the 'w'-th banana costing 'wk'.
 * The soldier initially has 'n' dollars. The problem determines if he needs to borrow money,
 * and if so, how much.
 *
 * Algorithm: Arithmetic series sum calculation. The total cost of 'w' bananas is the sum
 *            of an arithmetic series: k + 2k + ... + wk = k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
 * Time Complexity: O(1) as it involves a constant number of arithmetic operations.
 * Space Complexity: O(1).
 */
/*
 * =====================================================================================
 *
 *       Filename:  template.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  Thursday 21 May 2015 03:51:34  IST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Shounak Dey (), dylandey1996@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include <bits/stdc++.h> // Includes most standard library headers.

using namespace std;     // Uses the standard namespace.

int main() {
  /**
   * @brief Main function: Calculates money to borrow for bananas.
   * @return 0 for successful execution.
   */
  long long int k, n, w; // 'k': cost of first banana, 'n': initial money, 'w': number of bananas to buy.
  cin >> k >> n >> w;    // Read input values.
  long long int i, ans = 0; // 'i': loop counter (unused in final optimized code), 'ans': total cost.
  
  // Functional Utility: Calculate the total cost of 'w' bananas using the sum of an arithmetic series.
  // The series is k + 2k + ... + wk = k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
  // The original formula (w * (2 * k + (w - 1) * k)) / 2 is equivalent to (w * (k + wk)) / 2,
  // which is also k * w * (w + 1) / 2.
  ans = (w * (2 * k + (w - 1) * k)) / 2;

  /**
   * @brief Conditional block to determine money to borrow.
   * @pre 'ans' holds the total cost, 'n' holds initial money.
   */
  if (ans < n) // If the total cost is less than or equal to initial money.
    cout << "0" << endl; // Output 0, no money needs to be borrowed.
  else // If the total cost is greater than initial money.
    cout << ans - n << endl; // Output the difference, which is the amount to borrow.
  return 0; // Return 0 for successful execution.
}
