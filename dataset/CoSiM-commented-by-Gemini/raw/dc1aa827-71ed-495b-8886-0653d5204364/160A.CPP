/**
 * @file 160A.CPP
 * @brief Solution for Codeforces Problem 160A - "Twins".
 *
 * @details
 * This program finds the minimum number of coins to take from a set such that
 * the sum of the taken coins is strictly greater than the sum of the remaining coins.
 *
 * Algorithm:
 * A greedy approach is used. To maximize the sum with the minimum number of coins,
 * the highest-value coins are taken first.
 * 1. Read all coin values and calculate their total sum.
 * 2. Sort the coins in descending order.
 * 3. Iterate through the sorted coins, adding each coin to a personal sum `c` and
 *    simultaneously subtracting it from the total sum `s`. `s` now represents
 *    the sum of the remaining coins.
 * 4. The loop continues as long as the personal sum `c` is not strictly greater
 *    than the remaining sum `s`.
 * 5. The number of iterations taken is the minimum number of coins required.
 *
 * Time Complexity: O(N log N), dominated by the sorting step.
 * Space Complexity: O(N), to store the coin values.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: number of coins, i: loop counter, s: initial total sum, c: sum of coins taken.
  int n, i, s = 0, c = 0;
  cin >> n;
  // VLA to store coin values.
  int x[n];

  /**
   * Block Logic: Reads all coin values and calculates their total sum.
   */
  for (i = 0; i < n; i++) {
    cin >> x[i];
    s += x[i];
  }
  
  // Sort the coins in descending order to enable a greedy approach.
  sort(x, x + n, greater<int>());

  /**
   * Block Logic: Greedily takes the largest coins.
   * The loop continues as long as the sum of taken coins (`c`) is not yet
   * strictly greater than the sum of the remaining coins (`s`).
   * The loop counter `i` tracks the number of coins taken.
   */
  for (i = 0; i < n && c <= s; i++) {
    // Add the largest available coin to the personal sum.
    c += x[i];
    // Subtract the same coin from the total sum, so `s` now represents the remaining sum.
    s -= x[i];
  }
  
  // After the loop, `i` holds the count of coins taken to satisfy the condition.
  cout << i;
}
