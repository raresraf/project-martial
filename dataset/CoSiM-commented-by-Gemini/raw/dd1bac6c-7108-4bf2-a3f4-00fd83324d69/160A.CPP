#include <algorithm>
#include <iostream>
#include <vector>

/**
 * @file 160A.CPP
 * @brief Solution to a competitive programming problem, often called "Twins".
 *
 * Problem Statement:
 * Given a set of coins of varying values, find the minimum number of coins
 * you need to take such that the sum of the values of your coins is strictly
 * greater than the sum of the values of the remaining coins.
 *
 * Algorithm:
 * This problem can be solved using a greedy approach. To minimize the number
 * of coins taken while maximizing their sum, one should always pick the coins
* with the largest values first.
 *
 * 1. Read all coin values and calculate their total sum.
 * 2. Sort the coins in non-decreasing order.
 * 3. Iterate through the coins from largest to smallest, accumulating their
 *    sum and counting the number of coins taken.
 * 4. Stop when the accumulated sum is strictly greater than the sum of the
 *    remaining coins. The sum of remaining coins is (total_sum - my_sum).
 * 5. The count of coins taken at that point is the minimum required.
 *
 * Time Complexity: O(N log N) due to the sorting of coins, where N is the
 * number of coins.
 * Space Complexity: O(N) to store the coin values in a vector.
 */

using namespace std;

int main() {
  int num_coins;
  int total_sum{0}; // Use uniform initialization for safety.
  cin >> num_coins;

  vector<int> coins(num_coins);
  // Block Logic: Read coin values and calculate the total sum of all coins.
  for (size_t i{0}; i < num_coins; ++i) {
    cin >> coins[i];
    total_sum += coins[i];
  }

  // Sort the coins in ascending order to easily access the largest values.
  sort(begin(coins), end(coins));

  int min_coins_required{0}; // Counter for the minimum number of coins.
  int sum{0}; // The sum of the coins taken.

  // Block Logic: Greedily take the largest coins until our sum is greater
  // than the sum of the remaining coins.
  // Invariant: 'sum' contains the sum of the 'min_coins_required' largest coins.
  for (int i = num_coins - 1; i >= 0; --i) {
    sum += coins[i];
    ++min_coins_required;

    // Pre-condition: Check if the sum of coins taken is strictly greater
    // than the sum of the coins not taken.
    if (sum > total_sum - sum) {
      break; // Found the minimum number of coins required.
    }
  }

  cout << min_coins_required << endl;
  return 0;
}