/**
 * @file 546A.CPP
 * @brief Efficient solution for Codeforces Problem 546A, "Soldier and Bananas".
 * @details This program calculates the amount of money a soldier must borrow to
 * buy a specific number of bananas, where the cost increases linearly for each
 * banana.
 *
 * Algorithm:
 * The solution uses a direct mathematical formula for an arithmetic series to
 * find the total cost, making it an O(1) solution.
 * 1. Read the initial cost factor (`k`), the soldier's initial money (`n`),
 *    and the number of bananas to buy (`w`).
 * 2. The total cost of the bananas is the sum of the series k*1 + k*2 + ... + k*w.
 *    This is equivalent to `k * (1 + 2 + ... + w)`.
 * 3. The sum of the first `w` integers is `w * (w + 1) / 2`.
 * 4. The code calculates the total cost directly as `k * w * (w + 1) / 2`.
 * 5. It then subtracts the soldier's money `n` to find the deficit (`b`).
 * 6. If the deficit `b` is positive, that amount is printed. Otherwise, the
 *    soldier has enough money, and 0 is printed.
 *
 * Time Complexity: O(1).
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int k, n, w, b;
  cin >> k >> n >> w;
  
  /**
   * @brief Calculates the amount to borrow in a single line.
   * `k * w * (w + 1) / 2` is the formula for the sum of the arithmetic series
   * representing the total cost of all bananas.
   * Subtracting `n` gives the deficit.
   */
  b = k * w * (w + 1) / 2 - n;

  /**
   * @brief If the calculated deficit `b` is positive, the soldier needs to
   * borrow. Otherwise, he has enough money.
   */
  if (b > 0)
    cout << b << endl;
  else
    cout << "0" << endl;
  return 0;
}
