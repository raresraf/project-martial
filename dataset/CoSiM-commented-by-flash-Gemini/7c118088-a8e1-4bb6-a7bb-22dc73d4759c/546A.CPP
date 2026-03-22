/**
 * @file 546A.CPP
 * @brief Solution to Codeforces problem 546A - Soldier and Bananas.
 *        This program calculates the amount of money a soldier needs to borrow from a friend
 *        to buy a certain number of bananas, given the cost of the first banana and his initial money.
 *        The cost of each subsequent banana increases arithmetically. This version uses a direct
 *        mathematical formula to calculate the total cost, optimizing for performance.
 *
 * Algorithm:
 *   1. Reads the cost of the first banana (`k`), the soldier's initial money (`n`),
 *      and the number of bananas to buy (`w`).
 *   2. Calculates the total cost of `w` bananas using the arithmetic series sum formula:
 *      `TotalCost = k * (1 + 2 + ... + w) = k * w * (w + 1) / 2`.
 *   3. Determines the amount to borrow (`b`) by subtracting the soldier's initial money (`n`)
 *      from the `TotalCost`.
 *   4. If `b` is greater than 0, it means money needs to be borrowed, so `b` is printed.
 *   5. Otherwise (if `b` is 0 or negative), it means the soldier has enough money, so 0 is printed.
 *
 * Time Complexity: O(1) as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) for storing a few integer variables.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * @brief Main function that solves the "Soldier and Bananas" problem using a direct formula.
   *
   * Reads the cost of the first banana, initial money, and desired number of bananas.
   * Calculates the total cost and the amount to be borrowed.
   *
   * @return int Returns 0 upon successful execution.
   */
  int k, n, w, b;
  cin >> k >> n >> w;
  // Calculates the total cost of `w` bananas using the sum of an arithmetic series formula.
  // The cost of the i-th banana is i*k. Sum is k*(1+2+...+w) = k*w*(w+1)/2.
  b = k * w * (w + 1) / 2 - n;
  /**
   * Conditional Logic: Determines if the soldier needs to borrow money.
   * Pre-condition: `b` holds the calculated deficit or surplus.
   * Post-condition: Prints the amount to borrow (if positive) or 0 (if sufficient funds).
   */
  if (b > 0)
    cout << b << endl;
  else
    cout << "0" << endl;
  return 0;
}
