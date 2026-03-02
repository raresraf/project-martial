/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem on Codeforces (Problem 160A).
 * @details The goal is to determine the minimum number of coins one needs to take from a set
 * so that the sum of the taken coins is strictly greater than the sum of the remaining coins.
 * This is a classic greedy algorithm problem.
 *
 * Algorithm: Greedy Approach.
 * 1. Calculate the total sum of all coins.
 * 2. Sort the coins in descending order.
 * 3. Iterate through the sorted coins, adding the largest ones to a personal sum
 *    until this sum is greater than the sum of the remaining coins.
 *
 * Time Complexity: O(N log N), dominated by the sorting operation.
 * Space Complexity: O(N) to store the values of the N coins.
 */
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  // n: total number of coins.
  // a: temporary variable for reading each coin's value.
  // total_sum: the sum of all coin values.
  // coins_taken_count: counter for the minimum number of coins needed.
  // my_sum: the sum of coins taken.
  int n = 0, a = 0, total_sum = 0, coins_taken_count = 0, my_sum = 0;
  cin >> n;
  vector<int> v;

  /**
   * @brief Block Logic: Reads all coin values and calculates their total sum.
   * Invariant: After this loop, vector 'v' contains all coin values and 'total_sum'
   * holds their sum.
   */
  for (int i = 0; i < n; i++) {
    cin >> a;
    v.push_back(a);
    total_sum += a;
  }

  // Sorts the coins in ascending order.
  sort(v.begin(), v.end());
  // Reverses the vector to have coins in descending order, enabling the greedy strategy.
  reverse(v.begin(), v.end());

  /**
   * @brief Block Logic: Implements the greedy choice.
   * Iterates through the coins from largest to smallest, accumulating them.
   * Stops as soon as the accumulated sum is strictly greater than the sum of the rest.
   */
  for (int i = 0; i < n; i++) {
    // Add the largest available coin to my sum.
    my_sum += v[i];
    // Increment the count of coins taken.
    coins_taken_count++;
    // The sum of remaining coins is the original total minus my current sum.
    int remaining_sum = total_sum - my_sum;

    // Pre-condition: Check if my sum is now strictly greater than the remaining sum.
    if (my_sum > remaining_sum)
      break; // If so, we have found the minimum number of coins required.
  }

  // Output the minimum number of coins.
  cout << coins_taken_count;
  return 0;
}