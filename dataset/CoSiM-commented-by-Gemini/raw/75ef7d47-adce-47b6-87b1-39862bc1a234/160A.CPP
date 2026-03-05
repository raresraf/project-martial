/**
 * @file 160A.CPP
 * @brief "Twins" - A greedy algorithm problem for splitting coins.
 *
 * This program solves a problem where the goal is to select the minimum number
 * of coins from a given set such that the sum of the selected coins is strictly
 * greater than the sum of the remaining coins.
 *
 * @algorithm
 * The problem can be solved using a greedy approach. To minimize the number of
 * coins taken while maximizing their sum, one should always pick the coins with
 * the highest value first.
 *
 * 1. Read the number of coins, `n`.
 * 2. Read all `n` coin values into an array. While reading, calculate the `total`
 *    sum of all coins.
 * 3. Sort the array of coins in non-decreasing (ascending) order.
 * 4. Initialize `sum` (for the coins taken) and `num` (the count of coins taken) to 0.
 * 5. Iterate through the sorted array in reverse (from the largest coin to the smallest).
 *    a. In each step, add the current coin's value to `sum`.
 *    b. Increment the coin counter `num`.
 *    c. Check if the sum of the taken coins (`sum`) is now strictly greater than the
 *       sum of the remaining coins (`total - sum`). This is equivalent to checking
 *       if `sum > total / 2`, or `2 * sum > total` to avoid floating-point issues.
 *    d. If the condition is met, the minimum number of coins has been found. Print `num`
 *       and terminate the program.
 *
 * @complexity
 * Time Complexity: O(n log n), dominated by the sorting of the coin array.
 * Space Complexity: O(n), to store the coin values in an array.
 */
#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  // `n` is the number of coins.
  int n;
  // `num` is the count of coins taken.
  int num = 0;
  // `coin` array to store the value of each coin.
  int coin[101];
  // `total` is the sum of all coin values.
  int total = 0;
  // `sum` is the sum of the coins taken by the twin.
  int sum = 0;

  cin >> n;
  
  // Read coin values and calculate the total sum.
  for (int i = 0; i < n; ++i) {
    cin >> coin[i];
    total += coin[i];
  }
  
  // Sort the coins in ascending order.
  sort(coin, coin + n);
  
  /**
   * @brief Greedily take the largest coins.
   *
   * Iterate from the end of the sorted array (largest values) to the beginning.
   */
  for (int i = n - 1; i >= 0; i--) {
    // Take the current largest coin.
    sum += coin[i];
    num += 1;
    
    // Check if the sum of taken coins is strictly greater than half of the total sum.
    // `sum * 2 > total` is an integer-safe way to check `sum > total / 2`.
    if (sum * 2 > total) {
      // If so, we have found the minimum number of coins required.
      cout << num;
      return 0; // Exit the program.
    }
  }
  // This part of the code should be unreachable given the problem constraints
  // (as taking all coins will always satisfy the condition).
  return 0;
}
