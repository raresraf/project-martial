/**
 * @file 160A.CPP
 * @brief A clean, greedy solution for the "Twins" problem (Codeforces 160A).
 *
 * @details The problem is to find the minimum number of coins to take from a set
 * so that the sum of the taken coins is strictly greater than the sum of the
 * remaining coins. This program implements an efficient greedy algorithm.
 *
 * @algorithm The program employs a greedy strategy:
 * 1. Read all coin values and store their total sum in `s1`.
 * 2. Sort the coins in ascending order.
 * 3. Iterate backward from the largest coin. In each step, move the current
 *    coin's value from the "remaining" sum (`s1`) to the "taken" sum (`s2`).
 * 4. The loop stops as soon as the "taken" sum (`s2`) exceeds the "remaining"
 *    sum (`s1`).
 * 5. The number of coins taken is `n - i`, which is printed as the result.
 *
 * @complexity
 * Time Complexity: O(N log N), where N is the number of coins, dominated by the
 *                  sorting operation.
 * Space Complexity: O(N), for the array `a` storing the coin values. (The array
 *                   is fixed-size `a[1000]`, sufficient for N <= 100).
 */
#include <bits/stdc++.h>
using namespace std;

// Global variables to store the sum of remaining coins (s1) and taken coins (s2).
// As globals, they are zero-initialized.
int s1, s2;

int main() {
  int n;
  cin >> n;
  int a[1000];
  // Read coins and calculate the total sum.
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    s1 += a[i];
  }

  // Sort coins in ascending order to easily access the largest ones.
  sort(a, a + n);
  
  /**
   * @brief Block Logic: Greedily take the largest coins until the sum is sufficient.
   * @invariant In each iteration, `s2` holds the sum of the `n - 1 - i` largest
   * coins, and `s1` holds the sum of the remaining coins.
   */
  for (int i = n - 1; i >= 0; i--) {
    s2 += a[i]; // Add the largest available coin to our share.
    s1 -= a[i]; // Remove it from the remaining share.
    if (s2 > s1) {
      // Print the number of coins taken and exit.
      cout << n - i;
      break;
    }
  }
  return 0;
}