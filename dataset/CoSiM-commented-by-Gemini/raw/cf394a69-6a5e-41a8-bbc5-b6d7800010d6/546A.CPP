/**
 * @file 546A.CPP
 * @brief Calculates the monetary deficit for purchasing items with linearly increasing cost.
 *
 * This program solves a problem where a soldier buys 'w' bananas. The cost increases
 * with each banana: the first costs 'k', the second '2k', and so on. Given the soldier's
 * initial money 'n', it calculates the amount that must be borrowed.
 *
 * @b Algorithm: Iterative summation of an arithmetic series.
 *   An O(1) solution is possible using the formula: Total Cost = k * w * (w + 1) / 2.
 * @b Time_Complexity: O(W), where W is the number of items.
 * @b Space_Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // k: base cost of the item.
  // n: initial money available.
  // w: number of items to purchase.
  // sum: accumulator for the total cost.
  int k, n, w, sum = 0;
  cin >> k >> n >> w;

  /**
   * @brief Iteratively calculates the total cost of 'w' items.
   * Invariant: 'sum' holds the cost for the first 'i-1' items.
   */
  for (int i = 1; i <= w; i++) {
    sum += k * i;
  }

  /**
   * @brief Calculates and outputs the amount of money to borrow.
   * If initial money 'n' is sufficient, the amount is 0.
   * Otherwise, it's the difference between the total cost and 'n'.
   */
  if (n >= sum) {
    cout << 0 << endl;
  } else {
    cout << sum - n << endl;
  }

  return 0;
}
