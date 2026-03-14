/**
 * @61b69a47-23f7-4c42-85f6-503f976446cb/160A.CPP
 * @brief This program solves Codeforces problem 160A, "Twins".
 * Given 'n' coin values, the objective is to find the minimum number of coins
 * that can be taken such that their sum is strictly greater than the sum of
 * the remaining coins.
 *
 * Algorithm: Greedy approach. It calculates the total sum, sorts the coins
 * in descending order, and then iteratively takes the largest coins until
 * their sum exceeds the sum of the remaining coins.
 * Time Complexity: O(N log N) due to sorting.
 * Space Complexity: O(N) for storing coin values.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to execute the Twins problem logic.
 * Reads coin values, calculates total sum, sorts coins, and determines
 * the minimum number of coins needed to exceed half the total sum.
 * @return 0 on successful execution.
 */
int main() {
  int n;
  cin >> n;

  int a[n];
  int total = 0;
  int sum = 0;

  /**
   * Block Logic: Reads 'n' coin values into the array 'a' and calculates their total sum.
   * Invariant: After this loop, 'a' contains all coin values and 'total' holds their sum.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    total += a[i];
  }
  // Standard library sort is used, which typically has O(N log N) complexity.
  sort(a, a + n);

  int count = 0;

  /**
   * Block Logic: Iterates through the sorted coins from largest to smallest.
   * It greedily adds coins to 'sum' (coins taken by current player) and subtracts
   * them from 'total' (remaining coins). 'count' tracks the number of coins taken.
   * The loop breaks once 'sum' strictly exceeds 'total'.
   * Invariant: 'count' stores the minimum number of coins whose sum is strictly
   * greater than the sum of the remaining coins.
   */
  for (int i = n - 1; i >= 0; i--) {
    sum += a[i];    // Adds the largest available coin to the current player's sum.
    total -= a[i];  // Subtracts the taken coin from the total sum of all coins.
    count++;        // Increments the count of coins taken.
    // cout<<"sum = "<<sum<<", total = "<<total<<endl; // Debugging output.
    if (sum > total) // Pre-condition: Check if the current player's sum is strictly greater than the remaining sum.
      break;         // If condition met, stop taking coins.
  }

  cout << count << endl;

  return 0;
}
