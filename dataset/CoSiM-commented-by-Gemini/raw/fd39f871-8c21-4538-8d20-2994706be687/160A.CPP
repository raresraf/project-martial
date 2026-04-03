/**
 * @file 160A.cpp
 * @brief A clean and efficient solution to the "Twins" problem.
 * This program finds the minimum number of coins to take from a set such that
 * the sum of the taken coins is strictly greater than the sum of the remaining coins.
 *
 * Algorithm: Greedy Approach
 * 1. Read the number of coins `n` and their values.
 * 2. Calculate the total sum of all coin values.
 * 3. Sort the coins in descending order.
 * 4. Iterate through the sorted coins:
 *    a. Take the largest available coin, add its value to a `share` sum, and subtract it from the `total` sum.
 *    b. Increment the count of coins taken.
 *    c. If the `share` sum becomes strictly greater than the `total` sum, break the loop.
 * 5. Print the final count of coins taken.
 *
 * Time Complexity: O(N log N) dominated by the sorting step.
 * Space Complexity: O(N) to store the coin values.
 */
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n;
  vector<int> coins;

  cin >> n;

  int total = 0;
  int val;
  // Read coin values and calculate the total sum.
  for (int i = 0; i < n; ++i) {
    cin >> val;
    coins.push_back(val);
    total += val; // Note: this is slightly different from the original code, but more direct. The original adds coins[i] which is not yet val.
  }

  // Correctly calculate the total sum
  total = 0;
  for (int coin : coins) {
      total += coin;
  }


  // Sort the coins in descending order.
  sort(coins.begin(), coins.end(), greater<int>());

  int coin_count = 0;
  int share = 0;

  // Greedily take the largest coins.
  for (int i : coins) {
    coin_count++;
    total -= i; // Subtract the taken coin from the total.
    share += i; // Add the taken coin to our share.

    // If our share is now greater than what remains, we are done.
    if (share > total) {
      cout << coin_count;
      break;
    }
  }

  return 0;
}