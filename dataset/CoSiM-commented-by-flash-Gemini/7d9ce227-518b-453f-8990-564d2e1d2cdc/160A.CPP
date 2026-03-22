/**
 * @file 160A.CPP
 * @brief Solution to Codeforces problem 160A - Twins.
 *        This program aims to find the minimum number of coins required from a given
 *        collection such that their sum is strictly greater than the sum of the remaining coins.
 *
 * Algorithm: Greedy Approach
 *   1. Reads `n`, the number of coins.
 *   2. Reads `n` coin values into an array `a` and calculates their `total_sum`.
 *   3. Calculates `target_sum = total_sum / 2`. The goal is to find a subset of coins
 *      whose sum is strictly greater than `target_sum`.
 *   4. Sorts the coin values in ascending order.
 *   5. Iterates from the largest coin (end of the sorted array) backwards, adding coins
 *      to a `current_group_sum` and counting them (`coins_taken`).
 *   6. Stops when `current_group_sum` becomes strictly greater than `target_sum`.
 *   7. Prints `coins_taken`, which represents the minimum number of coins needed.
 *
 * Time Complexity: O(N log N) due to the sorting of `N` coin values.
 *                  The subsequent loop runs at most `N` times.
 * Space Complexity: O(N) for storing the `N` coin values in array `a`.
 */

#include <bits/stdc++.h> // Includes common C++ standard library headers like iostream, algorithm, etc.
using namespace std;

// Using `main()` directly without specifying return type is a C99 feature,
// but for competitive programming, it implicitly defaults to `int main()`.
main() {
  /**
   * @brief Main function that solves the "Twins" problem.
   *
   * Reads a collection of coin values and determines the minimum number of coins
   * needed to form a sum strictly greater than half of the total sum.
   *
   * @return int (implicitly) Returns 0 upon successful execution.
   */
  int n; // Number of coins.
  cin >> n;
  int a[n];    // Array to store coin values. (Variable-length array, a C99 feature).
  int total_sum = 0; // Accumulator for the total sum of all coin values.
  /**
   * Block Logic: Reads coin values and calculates their total sum.
   * Invariant: `total_sum` accumulates the sum of `a[0]` to `a[i]`.
   * Pre-condition: `n` is a positive integer.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i];    // Reads coin value into the array.
    total_sum += a[i]; // Adds the current coin value to the total sum.
  }
  
  // Calculates the target sum: we need a sum strictly greater than this.
  int target_half_sum = total_sum / 2;
  
  int coins_taken = 0;   // Counter for the number of coins taken for our group.
  int current_group_sum = 0; // Sum of coin values in our chosen group.
  int current_coin_index = n - 1; // Index to iterate from the largest coin.
  
  // Sorts the array of coin values in ascending order.
  sort(a, a + n);

  /**
   * Block Logic: Greedily selects coins starting from the largest until their
   *              sum (`current_group_sum`) is strictly greater than `target_half_sum`.
   * Invariant: `current_group_sum` is increasing, `coins_taken` is increasing.
   * Pre-condition: `a` is sorted, `target_half_sum` is calculated.
   */
  while (true) { // Loop continues until `break` condition is met.
    coins_taken++;             // Increment the count of coins taken.
    current_group_sum += a[current_coin_index]; // Add the largest available coin to our group's sum.
    /**
     * Conditional Logic: Checks if the current group's sum has exceeded the target.
     * If so, the minimum number of coins has been found.
     */
    if (current_group_sum > target_half_sum) {
      cout << coins_taken; // Prints the minimum number of coins.
      break;               // Exits the loop.
    }
    current_coin_index--; // Move to the next largest coin (previous index).
  }
}
