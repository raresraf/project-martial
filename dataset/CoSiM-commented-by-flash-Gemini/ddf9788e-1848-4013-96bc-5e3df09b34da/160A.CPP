/**
 * @file 160A.CPP
 * @brief This program solves the "Twins" problem (Codeforces 160A).
 *   It determines the minimum number of coins needed to take such that the sum of these coins
 *   is strictly greater than the sum of the remaining coins.
 *   Problem description: https://codeforces.com/problemset/problem/160/A
 * * Algorithm: Greedy approach with sorting.
 *   1. Calculate the total sum of all coins.
 *   2. Sort the coins in descending order.
 *   3. Iteratively take the largest remaining coins until the sum of taken coins
 *      is strictly greater than half of the total sum of all coins.
 * Time Complexity: O(N log N), dominated by the sorting step. Input reading and
 *   the greedy selection loop are O(N).
 * Space Complexity: O(N) for storing the array of coin values.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, such as <iostream>, <algorithm>, <vector>.
using namespace std;

/**
 * @brief Main function to solve the Twins problem.
 * Functional Utility: Reads coin values, calculates total sum, sorts coins,
 *                     and greedily selects coins to achieve a sum greater than half the total.
 */
int main() {
  int n; // Stores the number of coins.
  // Block Logic: Reads the total number of coins, `n`.
  cin >> n;
  // `arr`: A dynamic array (C99 feature, often supported as an extension) to store coin values.
  int arr[n] = {0};
  int cnt = 0;   // `cnt`: Counts the number of coins taken *before* the last coin that pushes the sum over the threshold.
  int sum = 0;   // `sum`: Stores the total value of all coins.
  int total = 0; // `total`: Stores the sum of coins taken by the "twin".
  /**
   * Block Logic: Reads `n` coin values and calculates their total sum.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: `arr` is populated with coin values, and `sum` holds their total.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }
  // Functional Utility: Sorts the array `arr` in descending order.
  // This is crucial for the greedy strategy to work (taking the largest coins first).
  sort(arr, arr + n, greater<int>());
  /**
   * Block Logic: Greedily selects coins starting from the largest until their sum
   *              is strictly greater than half of the total sum.
   * Pre-condition: `arr` is sorted in descending order, `sum` holds the total value.
   * Invariant: `total` accumulates selected coin values, `cnt` counts selected coins.
   */
  for (int i = 0; i < n; i++) {
    /**
     * Conditional Logic: Checks if adding the current largest coin (`arr[i]`)
     *                    would make `total` less than or equal to `sum / 2`.
     * The problem implies that the sum of taken coins must be strictly greater
     * than the sum of remaining coins, which is equivalent to `taken_sum > total_sum / 2`.
     * So, if `total + arr[i]` is still less than or equal to `sum / 2`, take it.
     */
    if (total + arr[i] <= sum / 2) {
      cnt++;        // Increment the count of coins taken.
      total += arr[i]; // Add the coin's value to the taken sum.
    } else {
      // If adding the current coin would make `total` strictly greater than `sum / 2`,
      // then this is the last coin needed. We break here, and `cnt + 1` will be the answer.
      break;
    }
  }
  // Block Logic: Prints the minimum number of coins needed.
  // Inline: `cnt + 1` is printed because `cnt` only counts coins *before* the last one
  //         that pushed the `total` sum to be strictly greater than `sum / 2`.
  cout << cnt + 1 << endl;
}
