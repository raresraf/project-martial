/**
 * @file 546A.CPP
 * @brief Calculates the amount of money to borrow for buying bananas using an arithmetic series formula.
 *
 * This program calculates the total cost of buying a certain number of bananas,
 * where the price increases linearly. It then determines how much money needs to be
 * borrowed if the initial amount is not sufficient. This version uses a mathematical
 * formula for efficiency.
 *
 * @b Algorithm: The total cost is calculated using the formula for the sum of an
 *    arithmetic series: `sum = n/2 * (2a + (n-1)d)`. In this problem, the series is
 *    k, 2k, 3k, ..., wk. The sum is `k * (1 + 2 + ... + w)`, which simplifies to
 *    `k * w * (w + 1) / 2`. This formula directly computes the total cost.
 *
 * @b Time_Complexity: O(1), as the calculation is done in a single step using a formula.
 * @b Space_Complexity: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  int k, n, w; // k: cost of first banana, n: initial money, w: number of bananas.
  cin >> k >> n >> w;
  
  /**
   * @brief This block calculates the amount to borrow using the arithmetic series sum formula.
   * Total cost = k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
   * If the initial money 'n' is sufficient, 0 is printed. Otherwise, the difference is printed.
   */
  if (n >= (k * w * (w + 1)) / 2)
    cout << 0;
  else
    cout << (k * w * (w + 1) / 2) - n;
}