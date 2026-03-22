/**
 * @file 546A.CPP
 * @brief Calculates the monetary deficit for purchasing items with linearly increasing cost.
 *
 * This program solves the "Soldier and Bananas" problem. A soldier with 'n' dollars
 * wants to buy 'w' bananas. The cost increases with each banana: the first costs 'k',
 * the second '2k', and so on. This code calculates the amount of money to borrow.
 *
 * @b Algorithm: Iterative summation of an arithmetic series.
 *   An O(1) solution is possible using the formula: Total Cost = k * w * (w + 1) / 2.
 * @b Time_Complexity: O(W), where W is the number of items.
 * @b Space_Complexity: O(1).
 *
 * @note The `main` function is missing an explicit return type, which defaults to `int`.
 *   Several variables (`m`, `f1`, `f2`) are declared but never used.
 */
#include <cstdio>
#include <iostream>
#include <string.h>
using namespace std;

main() {
  // k: base cost of the item.
  // n: initial money available (long int).
  // w: number of items to purchase.
  // p: accumulator for the total cost.
  // x: the amount of money to borrow.
  // m, f1, f2: unused variables.
  int m, k, i, w, f1, f2, p = 0, x;
  long int n;
  cin >> k >> n >> w;

  /**
   * @brief Iteratively calculates the total cost 'p' of 'w' items.
   */
  for (i = 1; i <= w; i++) {
    p = p + k * i;
  }
  
  // Calculate the deficit.
  x = p - n;
  
  /**
   * @brief Determine and output the amount to borrow.
   * If the calculated deficit 'x' is zero or negative, the soldier has enough
   * money. Otherwise, the positive deficit is the amount to borrow.
   */
  if (x <= 0) {
    cout << 0;
  } else {
    cout << x;
  }
}
