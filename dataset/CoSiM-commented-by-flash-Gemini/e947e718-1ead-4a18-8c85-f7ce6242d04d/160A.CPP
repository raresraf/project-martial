/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem.
 *
 * This program aims to find the minimum number of coins that need to be taken
 * from a given set of coins such that the sum of the taken coins is strictly
 * greater than the sum of the remaining coins.
 *
 * Algorithm: Greedy approach with sorting. The strategy is to always pick the
 *            largest available coins first to reach the target sum with minimum count.
 * Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 *                  The subsequent loops are O(N).
 * Space Complexity: O(N) to store the coin values in a vector.
 */
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

/**
 * @brief Main function to solve the Twins problem.
 *
 * Functional Utility: Reads coin values, sorts them, and then greedily selects
 * the largest coins until their sum exceeds the sum of the remaining coins.
 */
int main() {
  int n, a; // 'n' for number of coins, 'a' for temporary coin value.
  int sum = 0, mine = 0, coins = 0; // 'sum' for total coins, 'mine' for taken coins sum, 'coins' for count of taken coins.
  vector<int> v; // Vector to store coin values.

  cin >> n; // Read the number of coins.

  /**
   * Block Logic: Reads 'n' coin values from standard input and stores them in a vector.
   * Pre-condition: 'n' (number of coins) has been read.
   * Invariant: After each iteration, one coin value is read and added to the vector.
   */
  for (int i = 0; i < n; ++i) {
    cin >> a;      // Read a coin value.
    v.push_back(a); // Add the coin to the vector.
  }

  /**
   * Block Logic: Calculates the total sum of all coin values.
   * Pre-condition: All coin values are stored in vector 'v'.
   * Invariant: 'sum' accumulates the total value of all coins.
   */
  for (auto x : v) {
    sum += x;
  }

  // Sort the coin values in ascending order.
  // This allows easy access to the largest coins by popping from the back.
  sort(v.begin(), v.end());

  /**
   * Block Logic: Greedily selects the largest coins from the sorted vector.
   * It continues taking coins until the sum of 'mine' (taken coins) is strictly
   * greater than 'sum' (remaining coins).
   * Invariant: 'mine' is the sum of selected coins, 'sum' is the sum of remaining coins,
   *            and 'coins' is the count of selected coins.
   */
  while (mine <= sum) {
    mine += v.back(); // Add the largest available coin to 'mine'.
    sum -= v.back();  // Subtract the taken coin from 'sum'.
    v.pop_back();     // Remove the taken coin from the vector.
    coins++;          // Increment the count of taken coins.
  }

  cout << coins << endl; // Output the minimum number of coins required.

  return 0;
}