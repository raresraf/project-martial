/**
 * @file 160A.CPP
 * @brief Solution for Codeforces Problem 160A - "Twins".
 * This program solves a problem where you need to divide a set of coins
 * (represented by integers) between two people (twins) such that one person's
 * sum of coins is strictly greater than the other's, and the number of coins
 * taken by the first person is minimized.
 *
 * @problem_statement
 * Given `n` coins with specific values, find the minimum number of coins
 * whose sum is strictly greater than the sum of the remaining coins.
 *
 * @algorithm
 * This problem is solved using a greedy approach:
 * 1. Calculate the total sum of all coins.
 * 2. Sort the coins in descending order.
 * 3. Iterate through the sorted coins, starting from the largest.
 * 4. In each iteration, add the current largest coin to a 'taken sum' and
 *    subtract it from the 'remaining sum'.
 * 5. Count how many coins are taken. Stop when the 'taken sum' becomes
 *    strictly greater than the 'remaining sum'.
 *
 * Time Complexity: O(N log N) due to the sorting step, where N is the number of coins.
 * Space Complexity: O(N) to store the coin values in an array.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @brief Main function implementing the greedy algorithm for coin selection.
   * Orchestrates reading input, calculating total sum, sorting coins, and
   * iteratively selecting coins to achieve the target sum condition.
   * @return 0 on successful execution.
   */
  int n; ///< @var n The total number of coins available.
  int i; ///< @var i Loop counter and final result (number of coins taken).
  int s = 0; ///< @var s Represents the sum of coins NOT yet taken (initially total sum).
  int c = 0; ///< @var c Represents the sum of coins already taken.

  // Read the total number of coins.
  cin >> n;

  int x[n]; ///< @var x Array to store the values of `n` coins.

  /**
   * @brief Loop to read coin values from standard input and calculate the total sum.
   * @invariant `i` increments from 0 to `n-1`, populating the `x` array.
   * `s` accumulates the sum of all coin values.
   * @post The `x` array contains all coin values, and `s` holds their total sum.
   */
  for (i = 0; i < n; i++) {
    cin >> x[i];
    s += x[i];
  }

  /**
   * @brief Sorts the coin values in descending order.
   * @details This is a crucial step for the greedy strategy, ensuring that
   * the largest available coins are considered first.
   * @post The `x` array is sorted from largest to smallest coin value.
   */
  sort(x, x + n, greater<int>());

  /**
   * @brief Greedy loop to select coins until their sum is strictly greater than the remainder.
   * @invariant `c` is the sum of coins taken, `s` is the sum of remaining coins.
   * In each iteration, the largest available coin is moved from `s` to `c`.
   * @post `c` will be strictly greater than `s`, and `i` will be the count of coins taken.
   * The loop terminates when `c > s` condition is met.
   */
  for (i = 0; i < n & c <= s; i++) {
    c += x[i]; // Add the current largest coin to the taken sum.
    s -= x[i]; // Subtract it from the remaining sum.
  }

  // Output the minimum number of coins required.
  cout << i;
}
