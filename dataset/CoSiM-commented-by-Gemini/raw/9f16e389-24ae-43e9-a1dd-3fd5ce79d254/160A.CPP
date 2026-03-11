/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem using a greedy algorithm.
 *
 * The problem is to find the minimum number of coins to take from a set
 * such that the sum of their values is strictly greater than the sum of the
 * values of the remaining coins.
 *
 * Algorithm: Greedy approach. To minimize the number of coins, we should always
 * pick the coins with the largest values first. The process is as follows:
 * 1. Calculate the total sum of all coin values.
 * 2. Sort the coins in descending order.
 * 3. Iterate through the sorted coins, accumulating their sum.
 * 4. Stop when the accumulated sum is strictly greater than half of the total sum.
 * The number of coins taken is the answer.
 *
 * Time Complexity: O(N log N) dominated by the sorting step, where N is the number of coins.
 * Space Complexity: O(N) to store the coin values.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  // For local testing: redirect input from input.txt and output to output.txt.
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  int n; // The number of coins.
  cin >> n;
  vector<int> v(n); // Vector to store the values of the coins.
  int sum = 0; // Total sum of all coin values.
  int sum2 = 0; // The sum of the coins taken.
  
  /**
   * @brief Read coin values and calculate the total sum.
   */
  for (int i = 0; i < n; i++) {
    cin >> v[i];
    sum = sum + v[i];
  }

  // Sort the coins in descending order to apply the greedy strategy.
  sort(v.rbegin(), v.rend());

  /**
   * @brief Greedily take the largest coins until the sum is greater than half.
   *
   * Pre-condition: Coins in vector 'v' are sorted in descending order.
   * Invariant: 'sum2' holds the sum of the 'i' largest coins.
   */
  for (int i = 0; i < n; i++) {
    sum2 = sum2 + v[i];
    
    /**
     * @brief Check if the collected sum is a strict majority.
     * Logic: If the sum of the coins taken ('sum2') is more than half of the
     *        total sum, we have found the minimum number of coins required.
     */
    if (sum2 > sum / 2) {
      // Output the minimum number of coins needed.
      cout << i + 1;
      return 0; // Exit after finding the solution.
    }
  }

  return 0;
}
