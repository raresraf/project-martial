/**
 * @file 160A.CPP
 * @brief Solution to Codeforces problem 160A - Twins.
 *        This program aims to find the minimum number of coins required from a given
 *        collection such that their sum is strictly greater than the sum of the remaining coins.
 *        This implementation uses a greedy approach by leveraging a `std::priority_queue`
 *        to efficiently select the largest coins.
 *
 * Algorithm: Greedy Approach using Priority Queue
 *   1. Reads `n`, the total number of coins.
 *   2. Initializes `myMoney = 0` (sum of coins taken for our group) and `twinMoney = 0`
 *      (sum of all coins, initially representing the other group's potential sum).
 *   3. Populates a `std::priority_queue` with all coin values. Simultaneously, calculates
 *      the `twinMoney` as the total sum of all coins.
 *   4. Iteratively extracts the largest coin from the `priority_queue`:
 *      a. Adds the extracted coin's value to `myMoney`.
 *      b. Subtracts the extracted coin's value from `twinMoney`.
 *      c. Increments `coinsNumber` (count of coins in our group).
 *   5. This iteration continues as long as `myMoney` is not strictly greater than `twinMoney`.
 *   6. Once `myMoney` > `twinMoney`, `coinsNumber` holds the minimum count of coins needed.
 *   7. Prints `coinsNumber`.
 *
 * Time Complexity: O(N log N), where N is the number of coins, due to N insertions and
 *                  N extractions from the `std::priority_queue` (each operation takes O(log N) time).
 * Space Complexity: O(N) for storing the coin values in the `std::priority_queue`.
 */

#include <iostream>
#include <queue> // Required for std::priority_queue.

using std::cin;
using std::cout;
using std::endl;

int main() {
  /**
   * @brief Main function that solves the "Twins" problem.
   *
   * Reads a collection of coin values, uses a priority queue to select the largest ones,
   * and determines the minimum number of coins whose sum is strictly greater than
   * the sum of the remaining coins.
   *
   * @return int Returns 0 upon successful execution.
   */
  int n; // Total number of coins.
  int coinsNumber = 0; // Count of coins taken for our group.
  int twinMoney = 0;   // Sum of coins in the "twin" (other) group.
  int myMoney = 0;     // Sum of coins in "our" group.
  std::priority_queue<int> q; // Max-priority queue to store coin values.
  cin >> n; // Reads the number of coins.
  /**
   * Block Logic: Reads `n` coin values, adds them to `twinMoney` (total sum),
   *              and pushes them into the priority queue.
   * Invariant: `twinMoney` accumulates the total sum of all coins. `q` stores all coins.
   */
  for (int val, i = 0; i != n; ++i) { // `val` is a temporary variable for each coin value.
    cin >> val;
    twinMoney += val; // Accumulates the total sum.
    q.push(val);      // Adds coin to the priority queue.
  }

  /**
   * Block Logic: Greedily picks the largest coins from the priority queue
   *              until `myMoney` (our sum) is strictly greater than `twinMoney`
   *              (the other group's sum).
   * Invariant: `myMoney` increases, `twinMoney` decreases, `coinsNumber` increases.
   * Pre-condition: `q` contains all coins, `myMoney` and `twinMoney` are initialized.
   */
  while (myMoney <= twinMoney) { // Loop condition: our sum is not yet strictly greater.
    myMoney += q.top();   // Add the largest available coin to our sum.
    twinMoney -= q.top(); // Subtract this coin from the other group's sum.
    q.pop();              // Remove the largest coin from the queue.
    coinsNumber++;        // Increment the count of coins in our group.
  }

  // Prints the minimum number of coins required.
  cout << coinsNumber;
  return 0;
}
