/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" competitive programming problem.
 * @details The goal is to find the minimum number of coins to take from a given set
 * such that the sum of the chosen coins is strictly greater than the sum of the
 * remaining coins.
 *
 * @section algorithm Algorithm
 * A greedy approach provides the optimal solution:
 * 1. To minimize the number of coins taken, one should always pick the coins with the
 *    highest values first.
 * 2. The algorithm first calculates the total sum of all coin values. The target is to
 *    collect a sum that is strictly greater than half of this total sum.
 * 3. The coins are sorted in non-decreasing order.
 * 4. The algorithm then iterates, taking the largest available coin in each step,
 *    incrementing a counter, and adding the coin's value to a running sum.
 * 5. This process continues until the running sum exceeds half of the total sum.
 * 6. The final value of the counter is the minimum number of coins required.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n log n), where n is the number of coins. This is dominated by the sorting step.
 * - Space Complexity: O(n) to store the values of the coins.
 */
#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int n;             // Number of coins.
  int sum = 0;       // Holds the total sum, then the target sum.
  int count = 0;     // The number of coins taken.
  int ans = 0;       // The sum of the coins taken.
  cin >> n;

  int a[n];
  // Read all coin values and calculate their total sum.
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }

  // The target sum to strictly exceed is half of the total.
  sum = sum / 2;
  // Sort the coins in ascending order to easily access the largest ones.
  sort(a, a + n);

  // Greedily take the largest coins until the collected sum 'ans' exceeds the target 'sum'.
  while (ans <= sum) {
    // Increment the number of coins taken.
    ++count;
    // Add the value of the next largest coin to the collected sum.
    // a[n-count] accesses elements from the end of the sorted array (e.g., a[n-1], a[n-2], ...).
    ans += a[n - count];
  }
  // Output the minimum number of coins required to satisfy the condition.
  cout << count;
  return 0;
}