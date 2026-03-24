/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" competitive programming problem.
 *
 * This program addresses a problem where a collection of coins needs to be divided
 * into two groups. The goal is to form a subset of coins whose sum is strictly
 * greater than the sum of the remaining coins, while minimizing the number of coins
 * in this chosen subset.
 *
 * Algorithm: Greedy approach after sorting. The coins are sorted in descending order,
 *            and then the largest coins are iteratively picked until their sum
 *            exceeds half of the total sum of all coins.
 * Time Complexity: O(N log N) due to the sorting step (Bubble Sort is O(N^2), but
 *                  if `std::sort` were used, it would be O(N log N)). Here it's N^2.
 * Space Complexity: O(N) to store the array of coins.
 */

#include <iostream> // Required for standard input/output operations.
// #include <algorithm> // Would be required for std::sort, but a bubble sort is implemented.
// #include <vector>    // Could be used instead of a C-style array.

/**
 * @brief Standard namespace for C++ functionalities.
 */
using namespace std;

/**
 * @brief Main function where the program execution begins.
 * @return int Returns 0 on successful execution.
 */
int main() {
  int n; // Declares 'n' to store the number of coins.
  int i, j; // Loop counters.
  int temp; // Temporary variable for swapping during sort.
  int sum = 0; // Accumulator for the total sum of all coins.

  /**
   * @brief Reads the number of coins.
   *
   * Pre-condition: An integer value representing `n` is available on standard input.
   * Post-condition: `n` stores the count of coins.
   */
  cin >> n;

  // Declaration of a variable-length array (VLA), a C99 feature and a GNU extension in C++.
  // For strict C++ compliance, a `std::vector` should be used.
  int a[n];

  /**
   * @brief Reads `n` coin values into array `a` and calculates their total sum.
   *
   * Invariant: `i` iterates from 0 to `n-1`, `a[i]` stores the current coin value,
   *            and `sum` accumulates the total.
   * Pre-condition: `n` integer values are available on standard input.
   * Post-condition: Array `a` is populated with coin values, and `sum` holds their total.
   */
  for (i = 0; i < n; i++) {
    cin >> a[i];
    sum = sum + a[i];
  }

  /**
   * @brief Calculates half of the total sum, rounded down.
   * Functional Utility: Establishes the target sum needed for one of the groups.
   */
  sum = sum / 2;

  /**
   * @brief Sorts the array `a` in ascending order using the Bubble Sort algorithm.
   *
   * Algorithm: Bubble Sort (iteratively compares adjacent elements and swaps them
   *            if they are in the wrong order).
   * Invariant: After each outer loop iteration, the largest unsorted element "bubbles"
   *            to its correct position at the end of the unsorted sub-array.
   * Time Complexity: O(N^2) where N is the number of coins.
   * Pre-condition: Array `a` contains `n` unsorted integer coin values.
   * Post-condition: Array `a` contains `n` sorted integer coin values in ascending order.
   */
  for (i = 0; i < n; i++) {
    for (j = 0; j < n - i - 1; j++) {
      // Conditional Logic: Swaps elements if the current is greater than the next.
      if (a[j] > a[j + 1]) {
        temp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = temp;
      }
    }
  }

  j = 0; // Resets `j` to count the number of coins picked.
  temp = 0; // Resets `temp` to accumulate the sum of picked coins.

  /**
   * @brief Iteratively picks the largest coins until their sum exceeds half of the total sum.
   *
   * Invariant: `i` iterates from the largest element down to the smallest. `temp` accumulates
   *            the sum of picked coins, and `j` counts them.
   * Pre-condition: Array `a` is sorted in ascending order.
   * Post-condition: `j` holds the minimum number of coins required to achieve a sum
   *                 greater than half the total sum.
   */
  for (i = n - 1; i >= 0; i--) {
    // Conditional Logic: Breaks the loop if the accumulated sum (`temp`) already exceeds `sum` (half of total).
    if (temp > sum)
      break;
    else {
      // Functional Utility: Adds the current (largest remaining) coin to `temp` and increments the count `j`.
      temp = temp + a[i];
      j++;
    }
  }
  /**
   * @brief Prints the minimum number of coins required.
   *
   * Pre-condition: `j` contains the final count of coins.
   * Post-condition: The computed count `j` is displayed on standard output.
   */
  cout << j;
  return 0;
}