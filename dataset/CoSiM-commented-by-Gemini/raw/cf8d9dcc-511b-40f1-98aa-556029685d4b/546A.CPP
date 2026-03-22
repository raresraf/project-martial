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
  int k, n, w, t;
  cin >> k >> n >> w;
  
  int borrow;
  // 'total' accumulates the cost of all bananas.
  int total = 0;
  
  /**
   * @brief Iteratively calculates the total cost of 'w' items.
   * Invariant: 'total' holds the cost for the first 'i-1' items.
   */
  for (int i = 1; i <= w; i++) {
    // 't' stores the cost of the current (i-th) banana.
    t = k * i;
    total = total + t;
  }
  
  // cout<<total; // A commented-out debug print statement.
  
  /**
   * @brief Calculates and outputs the amount of money to borrow.
   * If the total cost is greater than the available money 'n', the deficit is printed.
   * Otherwise, no money needs to be borrowed, and 0 is printed.
   */
  if (total > n) {
    borrow = total - n;
    cout << borrow;
  } else {
    cout << "0";
  }
}
