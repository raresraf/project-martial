/**
 * @file 546A.CPP
 * @brief Calculates the monetary deficit for purchasing items with linearly increasing cost.
 *
 * This program solves the "Soldier and Bananas" problem. A soldier with 'n' dollars
 * wants to buy 'w' bananas. The cost increases with each banana. This implementation
 * calculates the required loan amount by directly subtracting each banana's cost from
 * the initial money.
 *
 * @b Algorithm: Iterative Deficit Calculation
 *   Instead of summing the total cost first, this approach deducts the cost of each
 *   banana from the available money `n`. If the final amount `n` is negative, its
 *   absolute value represents the money that must be borrowed.
 *
 * @b Time_Complexity: O(W), where W is the number of items.
 * @b Space_Complexity: O(1).
 */
/*
 * Problem description:
 *
 * https://codeforces.com/problemset/problem/546/A
 *
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  // Fast I/O for competitive programming.
  ios::sync_with_stdio(0);
  cin.tie(0);

  // k: base cost, w: number of items.
  short k, w;
  // n: initial money available, later used to track the deficit.
  int n;

  cin >> k >> n >> w;

  /**
   * @brief Iteratively subtracts the cost of each banana from the available money.
   */
  for (int i = 1; i <= w; i++) {
    n -= i * k;
  }

  // If 'n' is still positive, the soldier had enough money, so the debt is 0.
  if (n > 0)
    n = 0;
  
  // The absolute value of the final 'n' is the amount to be borrowed.
  cout << abs(n) << '\n';
}
