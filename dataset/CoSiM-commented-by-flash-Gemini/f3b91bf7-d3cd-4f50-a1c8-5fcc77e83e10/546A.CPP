/**
 * @file 546A.CPP
 * @brief This program calculates the amount of money a soldier needs to borrow to buy a certain number of bananas.
 * The cost of the first banana is 'k', and each subsequent banana costs 'i * k' (where 'i' is the banana number).
 * Algorithm: Linear summation of banana costs, then comparison with initial money to determine borrowing amount.
 * Time Complexity: O(W) where W is the number of bananas.
 * Space Complexity: O(1) as only a few variables are used.
 */

#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience.
using namespace std;
// Macro to speed up input/output operations by disabling synchronization with C stdio and untie cin from cout.
#define FAST_INP                                                               \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL)

int main() {
  FAST_INP; // Applies fast input/output optimizations.
  int k, n, w; // k: cost of the first banana, n: initial money, w: number of bananas to buy.
  // Reads the cost of the first banana (k), the soldier's initial money (n), and the number of bananas (w).
  cin >> k >> n >> w;
  int t = 0; // 't' accumulates the total cost of all bananas.
  /**
   * Block Logic: Calculates the total cost for buying 'w' bananas.
   * The cost for the i-th banana is 'i * k'.
   * Invariant: 't' accurately reflects the sum of costs for bananas from 1 to 'i'.
   */
  for (int i = 1; i <= w; i++)
    t += i * k;
  // Prints the amount of money to borrow. If initial money 'n' is more than or equal to total cost 't', print 0.
  // Otherwise, print the difference 't - n'.
  cout << (n > t ? 0 : t - n) << "\n";
  // Standard program exit.
  return 0;
}
