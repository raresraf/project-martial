/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem from Codeforces (Problem 546A).
 * @details This program calculates the amount of money a soldier must borrow to buy
 * 'w' bananas. The cost increases linearly: the first banana costs 'k', the
 * second '2k', and so on, up to 'wk'. The soldier initially has 'n' dollars.
 *
 * @section algorithm
 * Algorithm: Direct Mathematical Formula
 * The total cost of the bananas forms an arithmetic series: k + 2k + ... + wk.
 * This can be expressed as k * (1 + 2 + ... + w).
 * The sum of the first 'w' integers is calculated using the formula w * (w + 1) / 2.
 * Therefore, Total Cost = k * w * (w + 1) / 2.
 * The amount to borrow is the Total Cost minus the initial money 'n'. If the
 * result is negative or zero, no money needs to be borrowed.
 *
 * @section complexity
 * - Time Complexity: O(1), as the solution involves a direct calculation with
 *   no loops or recursion dependent on the input values.
 * - Space Complexity: O(1), as a fixed number of variables are used.
 */
#include <iostream>
using namespace std;

int main() {
  long long k, n, w; // Use long long to prevent overflow for large w.
  cin >> k >> n >> w;
  
  // The total cost is calculated using the formula for the sum of an arithmetic series.
  // Total Cost = k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
  // 'a' is the amount to borrow, which is the total cost minus the money he has.
  long long a = k * w * (w + 1) / 2 - n;

  /**
   * @brief Determines the final amount to output.
   * If 'a' is positive, it's the amount the soldier must borrow.
   * If 'a' is zero or negative, he has enough money, so he borrows 0.
   */
  if (a > 0)
    cout << a;
  else
    cout << 0;
  
  return 0;
}
