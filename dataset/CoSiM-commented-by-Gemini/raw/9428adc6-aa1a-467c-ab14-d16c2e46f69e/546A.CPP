/**
 * @file 546A.CPP
 * @brief Calculates the money a soldier needs to borrow using an O(1) formula.
 * @description
 *   This program solves the "Soldier and Bananas" problem from Codeforces.
 *   It calculates the total cost for `w` bananas where the price increases
 *   arithmetically (k, 2k, 3k, ...). Instead of a loop, it uses the
 *   mathematical formula for the sum of an arithmetic series to find the
 *   total cost in constant time. It then determines the deficit.
 *
 * @version 1.0
 * @created Thursday 21 May 2015 03:51:34 IST
 * @author Shounak Dey (dylandey1996@gmail.com)
 *
 * @algorithm Arithmetic Series Summation (O(1) Formula)
 * @complexity
 * - Time: O(1), as the calculation is done with a direct formula.
 * - Space: O(1).
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  // k: cost of the first banana.
  // n: initial amount of money the soldier has.
  // w: number of bananas to buy.
  long long int k, n, w;
  cin >> k >> n >> w;
  
  // The variable `i` is declared but not used.
  long long int i, ans = 0;

  /**
   * @brief Functional Utility: Calculates the total cost in O(1) time.
   * This uses the formula for the sum of an arithmetic series: Sum = num_terms/2 * (2*first_term + (num_terms-1)*difference).
   * Here, num_terms=w, first_term=k, and difference=k.
   * This is equivalent to k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
   */
  ans = (w * (2 * k + (w - 1) * k)) / 2;

  /**
   * @brief Block Logic: Determines the amount to borrow.
   */
  if (ans < n)
    // If the total cost is less than or equal to the money available, no need to borrow.
    cout << "0" << endl;
  else
    // The amount to borrow is the total cost minus the available money.
    cout << ans - n << endl;
    
  return 0;
}