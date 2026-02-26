/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem using a greedy algorithm.
 * @details The problem is to find the minimum number of coins to take from a given
 * set so that the sum of the selected coins is strictly greater than the sum of
 * the remaining coins.
 *
 * @algorithm
 * The optimal strategy is a greedy one:
 * 1. Calculate the total sum of all coin values.
 * 2. Sort the coins in descending order.
 * 3. Iterate through the sorted coins, adding them to a personal sum and
 *    subtracting from the total sum, until the personal sum is strictly greater
 *    than the remaining sum.
 * The number of coins taken at that point is the answer.
 *
 * @complexity
 * Time Complexity: O(N log N), dominated by the sorting operation, where N is
 * the number of coins.
 * Space Complexity: O(N) to store the coin values in an array.
 */
#include <bits/stdc++.h>
using namespace std;

int arr[101]; // Global array to store coin values, assuming n <= 100.

int main() {
  int n;              // The number of coins.
  int total_sum = 0;  // The sum of all coin values.
  int my_sum = 0;     // The sum of coins taken.
  int remaining_sum = 0; // The sum of coins not yet taken.
  int found_solution = 0; // Flag to indicate if a solution is found and printed.

  cin >> n;
  /**
   * @block
   * @brief Read coin values and calculate the total sum.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    total_sum += arr[i];
  }

  // Sort the coins in ascending order to easily access the largest ones.
  sort(arr, arr + n);

  /**
   * @block
   * @brief Greedily take the largest coins until the condition is met.
   * @invariant Iterates from the largest coin to the smallest.
   */
  for (int i = 0; i < n; i++) {
    // Take the next largest coin (iterating from the end of the sorted array).
    my_sum += arr[n - 1 - i];
    // The remaining sum is the total minus what has been taken.
    remaining_sum = total_sum - my_sum;

    /**
     * @block
     * @brief Check if the greedy condition is satisfied.
     * @pre `my_sum` contains the sum of the `i+1` largest coins.
     *      `remaining_sum` contains the sum of the other coins.
     * @post If `my_sum` is strictly greater, print the count of coins taken
     *       and terminate the loop.
     */
    if (my_sum > remaining_sum) {
      cout << i + 1;
      found_solution = 1;
      break;
    }
  }

  /**
   * @block
   * @brief Handle the edge case where all coins must be taken.
   * @logic If the loop completes without the condition being met (e.g., if all
   * coins have a value of 0), it implies all `n` coins are needed.
   * This block is technically redundant as the loop's logic would correctly
   * handle this by breaking on the last iteration, but it's kept as a safeguard.
   */
  if (found_solution == 0)
    cout << n;
}