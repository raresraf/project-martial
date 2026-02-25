/**
 * @file raw/46578ddb-7beb-4aff-8ffc-77d4ac0d966c/160A.CPP
 * @brief Solution to Codeforces problem 160A - Twins.
 *
 * @algorithm
 * The problem is to find the minimum number of coins to take from a collection
 * such that the sum of the taken coins is strictly greater than the sum of the
 * remaining coins.
 *
 * This is a classic greedy algorithm problem. To minimize the number of coins
 * while maximizing their sum, the optimal strategy is to always pick the coins
 * with the highest values first.
 *
 * The algorithm is as follows:
 * 1. Read all `n` coin values into an array.
 * 2. Calculate the total sum of all coins. The goal is to obtain a sum that is
 *    strictly greater than `total_sum / 2`.
 * 3. Sort the coin values in ascending order.
 * 4. Greedily take coins starting from the largest value (the end of the sorted
 *    array) and add them to a running total (`ans`).
 * 5. Keep a counter (`cnt`) for the number of coins taken.
 * 6. Stop when the accumulated sum `ans` becomes strictly greater than
 *    `total_sum / 2`.
 * 7. The final value of `cnt` is the minimum number of coins required.
 *
 * @complexity
 * - Time Complexity: O(N log N), dominated by the sorting step.
 * - Space Complexity: O(N), to store the `n` coin values. The problem constraints
 *   allow for a fixed-size array `a[100]`, making it O(1) in this specific code.
 */
// Written By Deepesh Nimma
// Problem: 160A Codeforces
#include <bits/stdc++.h>
using namespace std;

// Macros for convenience.
#define ll long long
#define space " "
#define end1 "\n"
#define ar array

// Global variables to store state.
int n, i, a[100], sum = 0, ans = 0, cnt = 0;

int main() {
  // Read the number of coins.
  cin >> n;

  // Read the values of the `n` coins.
  for (i = 0; i < n; ++i) {
    cin >> a[i];
  }

  // Sort the coins in non-decreasing order.
  sort(a, a + n);

  // Calculate the total sum of all coins.
  for (i = 0; i < n; ++i) {
    sum += a[i];
  }

  // The target is to get a sum strictly greater than half of the total.
  sum = sum / 2;

  /**
   * @brief Greedily take the largest coins until the sum is sufficient.
   * The `while` loop continues as long as our sum `ans` is not strictly
   * greater than the half-sum `sum`.
   */
  while (ans <= sum) {
    // Increment the coin counter.
    ++cnt;
    // Add the next largest coin to our sum.
    // `a[n - cnt]` accesses the array from the end.
    ans += a[n - cnt];
  }

  // Print the minimum number of coins required.
  cout << cnt;
}