/**
 * @file 1140D.CPP
 * @brief This program calculates a specific sum based on an input integer 'n'.
 * The sum `w` is accumulated by adding `i * (i - 1)` for all `i` from 3 to `n`.
 * This pattern is often seen in competitive programming problems related to
 * combinatorics, specifically involving combinations or triangular numbers,
 * or geometric problems like triangulating polygons.
 *
 * Algorithm: Iterative summation.
 * Time Complexity: O(N) where N is the input integer.
 * Space Complexity: O(1)
 */

// Problem Link : https://codeforces.com/problemset/problem/1140/D

#include <bits/stdc++.h> // Includes iostream, etc.
using namespace std;

typedef long long ll; // Functional Utility: Defines 'll' as an alias for 'long long' for convenience.

int main() {
  // Functional Utility: 'n' stores the upper limit for the summation.
  ll n;
  // Functional Utility: 'w' accumulates the sum. Initialized to 0.
  ll w = 0;
  // Functional Utility: 'i' is the loop counter for the summation.
  ll i;

  // Block Logic: Reads the integer 'n' from standard input.
  cin >> n;

  /**
   * Block Logic: Calculates the sum `w` by iterating from `i = 3` to `n`.
   * Invariant: In each iteration, `i * (i - 1)` is added to `w`.
   */
  for (i = 3; i <= n; i++)
    w = w + (i * (i - 1)); // Functional Utility: Adds the product of `i` and `(i-1)` to `w`.

  // Output the final calculated sum 'w'.
  cout << w;
  return 0;
}
