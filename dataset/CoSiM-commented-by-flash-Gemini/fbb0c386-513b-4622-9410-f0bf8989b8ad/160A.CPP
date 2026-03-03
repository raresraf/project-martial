/**
 * @file 160A.CPP
 * @brief Solution for Codeforces Problem 160A - "Twins".
 * This program solves the problem of dividing a set of coins between two brothers.
 * The goal is to find the minimum number of coins one brother (the "twin") must take
 * such that the total value of his coins is strictly greater than the total value
 * of the coins taken by the other brother.
 * Algorithm: Greedy approach. Sorts coins in ascending order, then the twin takes
 *            coins from largest to smallest until his sum strictly exceeds the remaining sum.
 * Time Complexity: O(N^2) - dominated by the nested loop which re-calculates `s2` in each iteration.
 *                  An optimized approach would achieve O(N log N) by calculating the total sum once.
 * Space Complexity: O(N) - to store the coin values in an array.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries (e.g., iostream, algorithm for sort).
using namespace std;     // Uses the standard namespace.

int main() {
  // n: total number of coins.
  // i, j: loop counters.
  // s1: sum of coins taken by the twin.
  // s2: sum of coins taken by the other brother.
  int n, i, j, s1 = 0, s2;
  
  cin >> n; // Reads the total number of coins.
  
  // Declares an array `a` of size `n` to store the coin values.
  // (Note: C-style array `int a[n]` is a Variable Length Array (VLA), a C99 feature
  // and C++ extension. `std::vector<int> a(n);` would be more standard C++).
  int a[n];
  
  /**
   * Block Logic: Reads the values of 'n' coins into the array.
   */
  for (i = 0; i < n; i++) {
    cin >> a[i]; // Reads each coin value.
  }
  
  // Sorts the array `a` in non-decreasing (ascending) order.
  // This allows the greedy strategy to pick the largest coins first.
  sort(a, a + n);
  
  /**
   * Block Logic: Implements the greedy strategy to find the minimum number of coins.
   * The twin takes coins from largest to smallest. In each step, it checks if his sum
   * is strictly greater than the sum of the remaining coins.
   * Invariant: `s1` holds the twin's current sum, and `s2` holds the other brother's sum.
   */
  for (i = n - 1; i >= 0; i--) { // Loop from the largest coin (index n-1) down to the smallest.
    s1 = s1 + a[i]; // The twin takes the current largest coin, adding it to his sum.
    s2 = 0;         // Reset sum for the other brother for recalculation.
    
    /**
     * Block Logic: Calculates the sum of coins remaining for the other brother.
     * This nested loop sums all coins with an index less than 'i' (i.e., smaller coins
     * that the twin has not yet taken). This recalculation makes the overall complexity O(N^2).
     */
    for (j = 0; j < i; j++) {
      s2 = s2 + a[j]; // Adds smaller coins to the other brother's sum.
    }
    
    // Checks if the twin's sum is strictly greater than the other brother's sum.
    if (s1 > s2) {
      // If true, `n - i` represents the number of coins the twin has taken.
      // (n - 1 - i + 1) = n - i.
      cout << n - i; // Prints the minimum number of coins the twin needs.
      return 0;      // Exits the program as the solution is found.
    }
  }

  return 0; // Should not be reached in typical problem constraints, but ensures main returns 0.
}
