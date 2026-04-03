/**
 * @file 160A.CPP
 * @brief Solution to a competitive programming problem.
 * This program solves a problem where the goal is to find the minimum number of coins
 * to take from a set of coins such that the sum of the values of the taken coins is
 * strictly greater than the sum of the values of the remaining coins.
 *
 * Algorithm: Greedy Approach
 * 1. Read the number of coins and their values.
 * 2. Sort the coins in ascending order of their values.
 * 3. Iterate from the largest coin to the smallest, accumulating the sum of taken coins (`s1`).
 * 4. In each iteration, calculate the sum of the remaining coins (`s2`).
 * 5. If `s1` is greater than `s2`, the minimum number of coins has been found. Print the count of taken coins and exit.
 *
 * Time Complexity: O(N*N) due to the nested loop, where N is the number of coins. Sorting takes O(N log N).
 * Space Complexity: O(N) to store the coin values.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @brief n: number of coins, i, j: loop counters, s1: sum of taken coins, s2: sum of remaining coins.
   */
  int n, i, j, s1 = 0, s2;
  // Read the number of coins.
  cin >> n;
  int a[n];
  // Read the values of the coins.
  for (i = 0; i < n; i++) {
    cin >> a[i];
  }
  
  // Sort the coins in ascending order to easily pick the largest ones.
  sort(a, a + n);
  
  /**
   * @brief Greedily take the largest coins until the sum of taken coins is greater than the sum of the rest.
   * The loop iterates from the largest coin to the smallest.
   * `s1` accumulates the sum of the coins taken.
   * `s2` calculates the sum of the remaining coins in each iteration.
   */
  for (i = n - 1; i >= 0; i--) {
    // Add the current largest coin to the taken sum.
    s1 = s1 + a[i];
    s2 = 0;
    // Calculate the sum of the remaining coins.
    for (j = 0; j < i; j++) {
      s2 = s2 + a[j];
    }
    // Check if the condition is met.
    if (s1 > s2) {
      // Print the number of coins taken (n - i) and exit.
      cout << n - i;
      return 0;
    }
  }

  return 0;
}
