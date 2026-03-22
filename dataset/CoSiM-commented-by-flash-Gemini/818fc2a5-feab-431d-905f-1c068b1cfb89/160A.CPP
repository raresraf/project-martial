/**
 * @file 160A.CPP
 * @brief Solution to Codeforces problem 160A - Twins.
 *        This program aims to find the minimum number of coins required from a given
 *        collection such that their sum is strictly greater than the sum of the remaining coins.
 *        This implementation uses a greedy approach by sorting coins in descending order.
 *
 * Algorithm: Greedy Approach with Sorting
 *   1. Reads `n`, the number of coins.
 *   2. Reads `n` coin values into a `std::vector<int>` and calculates their `total_sum`.
 *   3. Sorts the `coin` vector in descending order using a custom comparison function.
 *   4. Iterates through the sorted coins, accumulating their sum (`ansS`) and count (`ans`).
 *   5. The loop continues as long as `ansS` is not strictly greater than `total_sum / 2`.
 *   6. Once `ansS` exceeds `total_sum / 2`, `ans` (the minimum number of coins) is printed.
 *
 * Time Complexity: O(N log N) due to the sorting operation (where N is the number of coins).
 *                  The subsequent loop is linear, running at most N times.
 * Space Complexity: O(N) for storing the `std::vector<int> coin`. O(1) for auxiliary variables.
 */

#include <bits/stdc++.h> // Includes common C++ standard library headers like vector, algorithm, iostream.

using namespace std;

/**
 * @brief Custom comparison function for `std::sort` to achieve descending order.
 * @param a The first integer to compare.
 * @param b The second integer to compare.
 * @return bool True if `a` is greater than `b`, false otherwise.
 */
bool cmp(const int &a, const int &b) { return a > b; }

int main() {
  /**
   * @brief Main function that solves the "Twins" problem.
   *
   * Reads a collection of coin values, sorts them, and then selects the minimum
   * number of largest coins whose sum is greater than half of the total sum.
   *
   * @return int Returns 0 upon successful execution.
   */
  int n; // Number of coins.
  int ai; // Temporary variable to read individual coin values.
  int sum = 0; // Total sum of all coin values.
  int ansS = 0; // Sum of the coins selected for the current twin.
  int ans = 0; // Count of coins selected for the current twin.
  vector<int> coin; // Dynamic array to store coin values.
  scanf("%d", &n);  // Reads the number of coins.
  /**
   * Block Logic: Reads `n` coin values, stores them in `coin` vector, and calculates `sum`.
   * Invariant: `coin` contains all values, `sum` is their total.
   */
  for (int i = 0; i < n; i++)
    scanf("%d", &ai), coin.push_back(ai), sum += ai;

  // Sorts the `coin` vector in descending order using the custom comparison function `cmp`.
  sort(coin.begin(), coin.end(), cmp);

  /**
   * Block Logic: Greedily selects coins (starting from the largest) until their sum (`ansS`)
   *              is strictly greater than half of the total sum (`sum / 2`).
   * Invariant: `ansS` is increasing, `ans` is increasing.
   * Pre-condition: `coin` is sorted in descending order, `sum` is calculated.
   */
  for (int i = 0; i < n && ansS <= sum / 2; i++)
    ansS += coin[i], ans++; // Adds largest available coin to `ansS` and increments `ans`.

  printf("%d", ans); // Prints the minimum number of coins.
  return 0;
}