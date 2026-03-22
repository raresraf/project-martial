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
 */
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[]) {
  // k: base cost of the item.
  // n: initial money available.
  // w: number of items to purchase.
  int k, n, w;
  cin >> k >> n >> w;

  // 'price' accumulates the total cost of all bananas.
  int price = 0;

  /**
   * @brief Iteratively calculates the total cost of 'w' items.
   * The cost of the i-th banana is k * i.
   */
  for (int i = 1; i <= w; i++) {
    price += (k * i);
  }

  /**
   * @brief Determine and output the amount to borrow.
   * If the total price exceeds the soldier's money 'n', print the difference.
   * Otherwise, no money is needed, so print 0.
   */
  if (price - n > 0)
    cout << price - n;
  else
    cout << "0";

  return 0;
}